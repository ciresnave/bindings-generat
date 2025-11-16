pub mod headers;
pub mod libraries;

use anyhow::Result;
use std::path::{Path, PathBuf};

/// Information about discovered C/C++ library
#[derive(Debug, Clone)]
pub struct DiscoveryResult {
    /// Main header file
    pub main_header: PathBuf,

    /// All header files found
    pub headers: Vec<PathBuf>,

    /// Library file (e.g., libcudnn.so)
    pub library_file: Option<PathBuf>,

    /// Detected library name (e.g., "cudnn")
    pub library_name: String,

    /// Version string if detected
    pub version: Option<String>,
}

/// Discover library structure from source directory
pub fn discover(source_path: &Path) -> Result<DiscoveryResult> {
    let headers = headers::find_headers(source_path)?;
    let main_header = headers::identify_main_header(&headers)?;
    let library_file = libraries::find_library_file(source_path)?;
    let library_name = libraries::extract_library_name(&library_file, &main_header)?;
    let version = libraries::detect_version(source_path)?;

    Ok(DiscoveryResult {
        main_header,
        headers,
        library_file,
        library_name,
        version,
    })
}
