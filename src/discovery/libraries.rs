use anyhow::{Context, Result};
use regex::Regex;
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

/// Find library files (.so, .dll, .dylib) in the source directory
pub fn find_library_file(source_path: &Path) -> Result<Option<PathBuf>> {
    let library_extensions = if cfg!(target_os = "windows") {
        vec!["dll", "lib"]
    } else if cfg!(target_os = "macos") {
        vec!["dylib", "a"]
    } else {
        vec!["so", "a"]
    };

    for entry in WalkDir::new(source_path)
        .follow_links(true)
        .into_iter()
        .filter_map(|e| e.ok())
    {
        let path = entry.path();
        if path.is_file()
            && let Some(ext) = path.extension()
        {
            let ext_str = ext.to_str().unwrap_or("");
            if library_extensions.contains(&ext_str) {
                return Ok(Some(path.to_path_buf()));
            }
            // Also check for versioned .so files (e.g., libcudnn.so.9)
            if ext_str.starts_with("so") {
                return Ok(Some(path.to_path_buf()));
            }
        }
    }

    // Library file is optional (might be installed system-wide)
    Ok(None)
}

/// Extract library name from library file or header
/// Examples:
/// - libcudnn.so -> cudnn
/// - libcudnn.so.9.16.0 -> cudnn
/// - cudnn.dll -> cudnn
/// - cudnn.h -> cudnn
pub fn extract_library_name(library_file: &Option<PathBuf>, main_header: &Path) -> Result<String> {
    // Try to extract from library file first
    if let Some(lib_path) = library_file
        && let Some(name) = lib_path.file_stem()
    {
        let name_str = name.to_str().context("Invalid library filename")?;

        // Remove 'lib' prefix (common on Unix)
        let name_clean = name_str.strip_prefix("lib").unwrap_or(name_str);

        // Remove version suffixes (.so.9, etc.)
        let re = Regex::new(r"^([a-zA-Z0-9_-]+)").unwrap();
        if let Some(cap) = re.captures(name_clean) {
            return Ok(cap[1].to_string());
        }

        return Ok(name_clean.to_string());
    }

    // Fall back to main header filename
    if let Some(stem) = main_header.file_stem() {
        let name = stem.to_str().context("Invalid header filename")?;
        return Ok(name.to_string());
    }

    anyhow::bail!("Could not determine library name");
}

/// Attempt to detect version from various sources
pub fn detect_version(source_path: &Path) -> Result<Option<String>> {
    // Try to find version in common locations:
    // 1. VERSION file
    let version_file = source_path.join("VERSION");
    if version_file.exists()
        && let Ok(content) = std::fs::read_to_string(&version_file)
    {
        return Ok(Some(content.trim().to_string()));
    }

    // 2. Parse from library filename (e.g., libcudnn.so.9.16.0)
    if let Ok(Some(lib_file)) = find_library_file(source_path)
        && let Some(name) = lib_file.file_name()
    {
        let name_str = name.to_str().unwrap_or("");
        let re = Regex::new(r"\.so\.(\d+(?:\.\d+)*)").unwrap();
        if let Some(cap) = re.captures(name_str) {
            return Ok(Some(cap[1].to_string()));
        }
    }

    // 3. TODO: Parse version from header comments/defines
    // #define LIB_VERSION "1.0.0"

    Ok(None)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_library_name() {
        let cases = vec![
            ("libcudnn.so", "cudnn"),
            ("cudnn.dll", "cudnn"),
            ("libmy_lib.so.1.2.3", "my_lib"),
        ];

        for (input, expected) in cases {
            let lib_file = Some(PathBuf::from(input));
            let header = PathBuf::from("dummy.h");
            let result = extract_library_name(&lib_file, &header).unwrap();
            assert_eq!(result, expected, "Failed for input: {}", input);
        }
    }

    #[test]
    fn test_extract_library_name_from_header() {
        let lib_file = None;
        let header = PathBuf::from("my_library.h");
        let result = extract_library_name(&lib_file, &header).unwrap();
        assert_eq!(result, "my_library");
    }
}
