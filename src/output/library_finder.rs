//! Intelligent library finder for resolving linker errors
//!
//! This module provides functionality to automatically locate library files
//! on the system based on linker errors.

use anyhow::Result;
use std::path::{Path, PathBuf};
use std::process::Command;
use tracing::{debug, info, warn};

use super::error_parser::BuildError;

/// Extract library name from object file name in linker error
///
/// Example: "cudnn64_9.8ap2pywpmnmvqie73zcskbas7.0732dr0.rcgu.o" -> "cudnn64_9"
pub fn extract_library_name(object_file: &str) -> Option<String> {
    // Object file format: libraryname.hash.rcgu.o
    // We want everything before the first dot
    if let Some(first_dot) = object_file.find('.') {
        let lib_name = &object_file[..first_dot];
        // Remove any path components
        let lib_name = Path::new(lib_name).file_name().and_then(|n| n.to_str())?;
        Some(lib_name.to_string())
    } else {
        None
    }
}

/// Search for a library file on the system
///
/// On Windows, searches for .lib and .dll files
/// On Unix, searches for .a and .so files
pub fn find_library_on_system(lib_name: &str) -> Result<Vec<PathBuf>> {
    info!("Searching system for library: {}", lib_name);

    if cfg!(target_os = "windows") {
        find_library_windows(lib_name)
    } else {
        find_library_unix(lib_name)
    }
}

/// Search for library on Windows using dir command
fn find_library_windows(lib_name: &str) -> Result<Vec<PathBuf>> {
    let mut found_paths = Vec::new();

    // Search for .lib file (static library for linking)
    let lib_filename = format!("{}.lib", lib_name);

    info!("Searching for {}...", lib_filename);

    // Common search locations
    let search_roots = vec![
        format!(
            "C:\\Users\\{}\\",
            std::env::var("USERNAME").unwrap_or_default()
        ),
        "C:\\Program Files\\".to_string(),
        "C:\\Program Files (x86)\\".to_string(),
        "C:\\".to_string(), // Last resort - will be slow
    ];

    for root in search_roots {
        debug!("Searching in {}...", root);

        // Use where.exe or dir /S to search
        let output = Command::new("cmd")
            .args(["/C", "dir", "/S", "/B", &lib_filename])
            .current_dir(&root)
            .output();

        match output {
            Ok(output) if output.status.success() => {
                let stdout = String::from_utf8_lossy(&output.stdout);
                for line in stdout.lines() {
                    let line = line.trim();
                    if !line.is_empty() && line.ends_with(&lib_filename) {
                        let path = PathBuf::from(line);
                        if path.exists() {
                            info!("Found library at: {}", path.display());
                            found_paths.push(path);
                        }
                    }
                }

                // If we found any in this root, don't search more roots
                if !found_paths.is_empty() {
                    break;
                }
            }
            Ok(_output) => {
                debug!("Search in {} returned no results", root);
            }
            Err(e) => {
                debug!("Failed to search {}: {}", root, e);
                continue;
            }
        }
    }

    if found_paths.is_empty() {
        warn!("Could not find {} on system", lib_filename);
    }

    Ok(found_paths)
}

/// Search for library on Unix using find command
fn find_library_unix(lib_name: &str) -> Result<Vec<PathBuf>> {
    let mut found_paths = Vec::new();

    // Search for .a and .so files
    let static_lib = format!("lib{}.a", lib_name);
    let shared_lib = format!("lib{}.so", lib_name);

    info!("Searching for {} or {}...", static_lib, shared_lib);

    // Common search locations
    let search_roots = vec![
        std::env::var("HOME").unwrap_or_else(|_| "/home".to_string()),
        "/usr/local/lib".to_string(),
        "/usr/lib".to_string(),
        "/opt".to_string(),
    ];

    for root in search_roots {
        for lib_filename in &[&static_lib, &shared_lib] {
            debug!("Searching for {} in {}...", lib_filename, root);

            let output = Command::new("find")
                .arg(&root)
                .arg("-name")
                .arg(lib_filename)
                .arg("-type")
                .arg("f")
                .output();

            match output {
                Ok(output) if output.status.success() => {
                    let stdout = String::from_utf8_lossy(&output.stdout);
                    for line in stdout.lines() {
                        let line = line.trim();
                        if !line.is_empty() {
                            let path = PathBuf::from(line);
                            if path.exists() {
                                info!("Found library at: {}", path.display());
                                found_paths.push(path);
                            }
                        }
                    }
                }
                Ok(_) => {
                    debug!(
                        "Search for {} in {} returned no results",
                        lib_filename, root
                    );
                }
                Err(e) => {
                    debug!("Failed to search {} for {}: {}", root, lib_filename, e);
                    continue;
                }
            }
        }

        // If we found any in this root, don't search more roots
        if !found_paths.is_empty() {
            break;
        }
    }

    if found_paths.is_empty() {
        warn!("Could not find {} or {} on system", static_lib, shared_lib);
    }

    Ok(found_paths)
}

/// Extract library names from build errors that we should search for
pub fn extract_libraries_to_find(errors: &[BuildError]) -> Vec<String> {
    let mut libraries = Vec::new();

    for error in errors {
        if let BuildError::UnresolvedSymbol {
            object_file: Some(lib_name),
            ..
        } = error
        {
            // The object_file has already been parsed to just the library name by the error parser
            // e.g., "cudnn64_9" not "cudnn64_9.8ap2pywpmnmvqie73zcskbas7.0732dr0.rcgu.o"
            if !libraries.contains(lib_name) {
                libraries.push(lib_name.clone());
            }
        }
    }

    libraries
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_library_name() {
        assert_eq!(
            extract_library_name("cudnn64_9.8ap2pywpmnmvqie73zcskbas7.0732dr0.rcgu.o"),
            Some("cudnn64_9".to_string())
        );

        assert_eq!(
            extract_library_name("libcuda.abc123.rcgu.o"),
            Some("libcuda".to_string())
        );

        assert_eq!(
            extract_library_name("simple_lib.o"),
            Some("simple_lib".to_string())
        );
    }
}
