use anyhow::{Context, Result};
use indicatif::{ProgressBar, ProgressStyle};
use sha2::{Digest, Sha256};
use std::path::Path;
use std::thread;
use std::time::Duration;
use tracing::{debug, warn};

/// Maximum number of retry attempts for downloads
const MAX_RETRIES: u32 = 3;

/// Initial delay in milliseconds for exponential backoff
const INITIAL_BACKOFF_MS: u64 = 1000;

/// Configuration for downloading files with retry and verification
#[derive(Debug, Clone)]
pub struct DownloadConfig {
    /// URL to download from
    pub url: String,
    /// Destination file path
    pub dest: std::path::PathBuf,
    /// Optional SHA-256 checksum to verify (hex string)
    pub checksum: Option<String>,
    /// Expected size in bytes (for progress bar)
    pub expected_size: Option<u64>,
    /// Description for progress messages
    pub description: String,
}

impl DownloadConfig {
    /// Create a new download configuration
    pub fn new(url: impl Into<String>, dest: impl AsRef<Path>) -> Self {
        Self {
            url: url.into(),
            dest: dest.as_ref().to_path_buf(),
            checksum: None,
            expected_size: None,
            description: "Downloading".to_string(),
        }
    }

    /// Set the expected checksum for verification
    pub fn with_checksum(mut self, checksum: impl Into<String>) -> Self {
        self.checksum = Some(checksum.into());
        self
    }

    /// Set the expected file size
    pub fn with_expected_size(mut self, size: u64) -> Self {
        self.expected_size = Some(size);
        self
    }

    /// Set the description for progress messages
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = description.into();
        self
    }
}

/// Download a file with retry logic and optional checksum verification
pub fn download_with_retry(config: &DownloadConfig) -> Result<()> {
    let mut attempt = 0;

    loop {
        attempt += 1;

        match try_download(config, attempt) {
            Ok(()) => {
                // Verify checksum if provided
                if let Some(expected_checksum) = &config.checksum {
                    debug!("Verifying checksum for {}", config.dest.display());
                    verify_checksum(&config.dest, expected_checksum)?;
                }
                return Ok(());
            }
            Err(e) if attempt >= MAX_RETRIES => {
                anyhow::bail!(
                    "Failed to download {} after {} attempts: {}",
                    config.url,
                    MAX_RETRIES,
                    e
                );
            }
            Err(e) => {
                let backoff_ms = INITIAL_BACKOFF_MS * 2u64.pow(attempt - 1);
                warn!(
                    "Download attempt {}/{} failed: {}. Retrying in {}ms...",
                    attempt, MAX_RETRIES, e, backoff_ms
                );
                thread::sleep(Duration::from_millis(backoff_ms));
            }
        }
    }
}

/// Attempt to download a file once
fn try_download(config: &DownloadConfig, attempt: u32) -> Result<()> {
    let attempt_msg = if attempt > 1 {
        format!(" (attempt {}/{})", attempt, MAX_RETRIES)
    } else {
        String::new()
    };

    debug!("Downloading from {}{}", config.url, attempt_msg);

    // Create progress bar
    let pb = ProgressBar::new(config.expected_size.unwrap_or(0));
    pb.set_style(
        ProgressStyle::default_bar()
            .template(&format!(
                "    [{{bar:40.cyan/blue}}] {{bytes}}/{{total_bytes}} ({{eta}}) {}{}",
                config.description, attempt_msg
            ))
            .unwrap()
            .progress_chars("#>-"),
    );

    // Create HTTP client with timeout
    let client = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(300)) // 5 minutes
        .build()
        .context("Failed to create HTTP client")?;

    // Send request
    let mut response = client
        .get(&config.url)
        .send()
        .context("Failed to send HTTP request")?;

    if !response.status().is_success() {
        anyhow::bail!("HTTP error: {}", response.status());
    }

    // Update progress bar with actual content length
    if let Some(len) = response.content_length() {
        pb.set_length(len);
    }

    // Create destination file
    let file = std::fs::File::create(&config.dest)
        .with_context(|| format!("Failed to create file: {}", config.dest.display()))?;
    let mut file = pb.wrap_write(file);

    // Copy response to file with progress
    std::io::copy(&mut response, &mut file)
        .with_context(|| format!("Failed to write to file: {}", config.dest.display()))?;

    let finish_msg = format!("✓ {}", config.description);
    pb.finish_with_message(finish_msg);

    Ok(())
}

/// Verify SHA-256 checksum of a file
fn verify_checksum(path: &Path, expected_hex: &str) -> Result<()> {
    let mut file =
        std::fs::File::open(path).with_context(|| format!("Failed to open {}", path.display()))?;

    let mut hasher = Sha256::new();
    std::io::copy(&mut file, &mut hasher)
        .with_context(|| format!("Failed to read {}", path.display()))?;

    let hash_bytes = hasher.finalize();
    let actual_hex = format!("{:x}", hash_bytes);

    if actual_hex.eq_ignore_ascii_case(expected_hex) {
        debug!("Checksum verified for {}", path.display());
        Ok(())
    } else {
        anyhow::bail!(
            "Checksum mismatch for {}\n  Expected: {}\n  Got:      {}",
            path.display(),
            expected_hex,
            actual_hex
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_download_config_builder() {
        let config = DownloadConfig::new("https://example.com/file.tar.gz", "/tmp/file.tar.gz")
            .with_checksum("abc123")
            .with_expected_size(1024)
            .with_description("Test file");

        assert_eq!(config.url, "https://example.com/file.tar.gz");
        assert_eq!(config.dest, Path::new("/tmp/file.tar.gz"));
        assert_eq!(config.checksum, Some("abc123".to_string()));
        assert_eq!(config.expected_size, Some(1024));
        assert_eq!(config.description, "Test file");
    }

    #[test]
    fn test_verify_checksum_empty_file() {
        // Empty file has known SHA-256 hash
        let temp_dir = tempfile::tempdir().unwrap();
        let file_path = temp_dir.path().join("empty.txt");
        std::fs::write(&file_path, b"").unwrap();

        let expected_hash = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";
        verify_checksum(&file_path, expected_hash).unwrap();
    }

    #[test]
    fn test_verify_checksum_mismatch() {
        let temp_dir = tempfile::tempdir().unwrap();
        let file_path = temp_dir.path().join("test.txt");
        std::fs::write(&file_path, b"hello").unwrap();

        let wrong_hash = "0000000000000000000000000000000000000000000000000000000000000000";
        let result = verify_checksum(&file_path, wrong_hash);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Checksum mismatch")
        );
    }
}
