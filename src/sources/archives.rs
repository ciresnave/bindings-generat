use anyhow::{Context, Result};
use flate2::read::GzDecoder;
use std::fs::File;
use std::io::BufReader;
use std::path::{Path, PathBuf};
use tempfile::TempDir;
use tracing::info;

use super::PreparedSource;

/// Download a file from a URL and extract if it's an archive
pub fn download_and_extract(url: &str) -> Result<PreparedSource> {
    info!("Downloading from URL: {}", url);

    // Create temporary directory for extraction
    let temp_dir = TempDir::new().context("Failed to create temporary directory")?;

    let filename = extract_filename_from_url(url);
    let temp_file = temp_dir.path().join(&filename);

    // Download with retry logic
    let config = crate::llm::network::DownloadConfig::new(url, &temp_file)
        .with_description(format!("Downloading {}", filename));

    crate::llm::network::download_with_retry(&config).context("Failed to download archive")?;

    info!("Downloaded to temporary file: {}", temp_file.display());

    // Extract the archive
    extract_to_directory(&temp_file, temp_dir.path())?;

    // Find the actual source directory (might be nested in archive)
    let source_dir = find_source_directory(temp_dir.path())?;

    Ok(PreparedSource {
        path: source_dir,
        is_temporary: true,
    })
}

/// Extract an archive file to a temporary directory
pub fn extract_archive(archive_path: &Path) -> Result<PreparedSource> {
    info!("Extracting archive: {}", archive_path.display());

    let temp_dir = TempDir::new().context("Failed to create temporary directory")?;

    extract_to_directory(archive_path, temp_dir.path())?;

    // Find the actual source directory
    let source_dir = find_source_directory(temp_dir.path())?;

    Ok(PreparedSource {
        path: source_dir,
        is_temporary: true,
    })
}

/// Extract archive to a specific directory
fn extract_to_directory(archive_path: &Path, dest_dir: &Path) -> Result<()> {
    let file_name = archive_path
        .file_name()
        .and_then(|n| n.to_str())
        .context("Invalid archive filename")?;

    if file_name.ends_with(".zip") {
        extract_zip(archive_path, dest_dir)?;
    } else if file_name.ends_with(".tar.gz") || file_name.ends_with(".tgz") {
        extract_tar_gz(archive_path, dest_dir)?;
    } else if file_name.ends_with(".tar") {
        extract_tar(archive_path, dest_dir)?;
    } else if file_name.ends_with(".gz") {
        extract_gz(archive_path, dest_dir)?;
    } else {
        anyhow::bail!("Unsupported archive format: {}", file_name);
    }

    Ok(())
}

/// Extract a ZIP archive
fn extract_zip(archive_path: &Path, dest_dir: &Path) -> Result<()> {
    let file = File::open(archive_path).context("Failed to open ZIP archive")?;
    let mut archive = zip::ZipArchive::new(file).context("Failed to read ZIP archive")?;

    for i in 0..archive.len() {
        let mut file = archive.by_index(i).context("Failed to access ZIP entry")?;

        let outpath = match file.enclosed_name() {
            Some(path) => dest_dir.join(path),
            None => continue,
        };

        if file.name().ends_with('/') {
            // Directory
            std::fs::create_dir_all(&outpath).context("Failed to create directory")?;
        } else {
            // File
            if let Some(parent) = outpath.parent() {
                std::fs::create_dir_all(parent).context("Failed to create parent directory")?;
            }
            let mut outfile = File::create(&outpath).context("Failed to create file")?;
            std::io::copy(&mut file, &mut outfile).context("Failed to write file")?;
        }

        // Set permissions on Unix
        #[cfg(unix)]
        {
            use std::fs::Permissions;
            use std::os::unix::fs::PermissionsExt;

            if let Some(mode) = file.unix_mode() {
                std::fs::set_permissions(&outpath, Permissions::from_mode(mode))
                    .context("Failed to set permissions")?;
            }
        }
    }

    info!("Extracted {} files from ZIP archive", archive.len());
    Ok(())
}

/// Extract a tar.gz archive
fn extract_tar_gz(archive_path: &Path, dest_dir: &Path) -> Result<()> {
    let file = File::open(archive_path).context("Failed to open tar.gz archive")?;
    let gz = GzDecoder::new(BufReader::new(file));
    let mut archive = tar::Archive::new(gz);

    archive
        .unpack(dest_dir)
        .context("Failed to extract tar.gz archive")?;

    info!("Extracted tar.gz archive");
    Ok(())
}

/// Extract a tar archive
fn extract_tar(archive_path: &Path, dest_dir: &Path) -> Result<()> {
    let file = File::open(archive_path).context("Failed to open tar archive")?;
    let mut archive = tar::Archive::new(BufReader::new(file));

    archive
        .unpack(dest_dir)
        .context("Failed to extract tar archive")?;

    info!("Extracted tar archive");
    Ok(())
}

/// Extract a gzip file (not a tar.gz, just plain .gz)
fn extract_gz(archive_path: &Path, dest_dir: &Path) -> Result<()> {
    let file = File::open(archive_path).context("Failed to open gz file")?;
    let mut gz = GzDecoder::new(BufReader::new(file));

    // Extract to a file with the same name minus .gz
    let output_name = archive_path
        .file_stem()
        .and_then(|n| n.to_str())
        .context("Invalid filename")?;
    let output_path = dest_dir.join(output_name);

    let mut output_file = File::create(&output_path).context("Failed to create output file")?;

    std::io::copy(&mut gz, &mut output_file).context("Failed to decompress file")?;

    info!("Extracted gz file to {}", output_path.display());
    Ok(())
}

/// Find the actual source directory within extracted archive
/// Many archives have a single top-level directory, we want to use that
fn find_source_directory(extracted_dir: &Path) -> Result<PathBuf> {
    let entries: Vec<_> = std::fs::read_dir(extracted_dir)
        .context("Failed to read extracted directory")?
        .filter_map(|e| e.ok())
        .collect();

    // If there's only one entry and it's a directory, use that
    if entries.len() == 1 {
        let entry = &entries[0];
        let path = entry.path();
        if path.is_dir() {
            info!("Using nested directory as source: {}", path.display());
            return Ok(path);
        }
    }

    // Otherwise, use the extraction directory itself
    Ok(extracted_dir.to_path_buf())
}

/// Extract filename from URL
fn extract_filename_from_url(url: &str) -> String {
    url.split('/')
        .next_back()
        .unwrap_or("downloaded_file")
        .split('?') // Remove query parameters
        .next()
        .unwrap_or("downloaded_file")
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_filename_from_url() {
        assert_eq!(
            extract_filename_from_url("https://example.com/file.tar.gz"),
            "file.tar.gz"
        );
        assert_eq!(
            extract_filename_from_url("https://example.com/path/to/archive.zip?token=abc"),
            "archive.zip"
        );
    }
}
