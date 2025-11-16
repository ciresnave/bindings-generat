use anyhow::{Context, Result};
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

/// Find all header files in the source directory
pub fn find_headers(source_path: &Path) -> Result<Vec<PathBuf>> {
    let mut headers = Vec::new();

    for entry in WalkDir::new(source_path)
        .follow_links(true)
        .into_iter()
        .filter_map(|e| e.ok())
    {
        let path = entry.path();
        if path.is_file()
            && let Some(ext) = path.extension()
            && (ext == "h" || ext == "hpp" || ext == "hh")
        {
            headers.push(path.to_path_buf());
        }
    }

    if headers.is_empty() {
        anyhow::bail!("No header files found in {}", source_path.display());
    }

    Ok(headers)
}

/// Identify the main header file from a list of headers
/// Heuristics:
/// 1. File with same name as directory (e.g., cudnn/cudnn.h)
/// 2. Shortest filename (likely the main one)
/// 3. Most included by others
pub fn identify_main_header(headers: &[PathBuf]) -> Result<PathBuf> {
    if headers.is_empty() {
        anyhow::bail!("No headers provided");
    }

    // If only one header, that's it
    if headers.len() == 1 {
        return Ok(headers[0].clone());
    }

    // Try to find header with same name as its parent directory
    for header in headers {
        if let Some(parent) = header.parent()
            && let Some(parent_name) = parent.file_name()
            && let Some(file_stem) = header.file_stem()
            && parent_name == file_stem
        {
            return Ok(header.clone());
        }
    }

    // Fall back to shortest filename (heuristic: main headers are often simple)
    let main = headers
        .iter()
        .min_by_key(|h| {
            h.file_name()
                .and_then(|n| n.to_str())
                .map(|s| s.len())
                .unwrap_or(usize::MAX)
        })
        .context("Failed to find main header")?;

    Ok(main.clone())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_identify_main_header_single() {
        let headers = vec![PathBuf::from("test.h")];
        let main = identify_main_header(&headers).unwrap();
        assert_eq!(main, PathBuf::from("test.h"));
    }

    #[test]
    fn test_identify_main_header_shortest() {
        let headers = vec![
            PathBuf::from("very_long_header_name.h"),
            PathBuf::from("lib.h"),
            PathBuf::from("internal.h"),
        ];
        let main = identify_main_header(&headers).unwrap();
        assert_eq!(main, PathBuf::from("lib.h"));
    }
}
