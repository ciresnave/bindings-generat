pub mod archives;

use anyhow::Result;
use std::path::PathBuf;

/// Prepare the source for processing
/// - If it's a URL, download it
/// - If it's an archive, extract it
/// - If it's a directory, use it as-is
pub fn prepare_source(source: &str) -> Result<PreparedSource> {
    // Check if it's a URL
    if source.starts_with("http://") || source.starts_with("https://") {
        return archives::download_and_extract(source);
    }

    let path = PathBuf::from(source);

    // Check if it's a file (archive) or directory
    if path.is_file() {
        return archives::extract_archive(&path);
    }

    // It's a directory - use as-is
    if path.is_dir() {
        return Ok(PreparedSource {
            path,
            is_temporary: false,
        });
    }

    anyhow::bail!("Source path does not exist: {}", source);
}

/// Result of source preparation
pub struct PreparedSource {
    /// Path to the source directory
    pub path: PathBuf,

    /// Whether this is a temporary directory that should be cleaned up
    pub is_temporary: bool,
}

impl PreparedSource {
    /// Get the path to the source directory
    pub fn path(&self) -> &PathBuf {
        &self.path
    }
}

impl Drop for PreparedSource {
    fn drop(&mut self) {
        if self.is_temporary {
            // Clean up temporary directory
            if let Err(e) = std::fs::remove_dir_all(&self.path) {
                tracing::warn!(
                    "Failed to clean up temporary directory {}: {}",
                    self.path.display(),
                    e
                );
            }
        }
    }
}
