use anyhow::Result;
use std::path::{Path, PathBuf};
use tracing::{debug, info, warn};

/// Common dependency detection patterns
#[derive(Debug, Clone)]
pub struct DependencyInfo {
    pub name: String,
    pub include_paths: Vec<PathBuf>,
    pub lib_paths: Vec<PathBuf>,
    pub link_libs: Vec<String>,
}

/// Detect dependencies based on library name and required headers
pub fn detect_dependencies(
    library_name: &str,
    headers: &[PathBuf],
    source_path: &Path,
) -> Result<Vec<DependencyInfo>> {
    let mut dependencies = Vec::new();

    info!("Detecting dependencies for library: {}", library_name);

    // Check for cuDNN dependency (also includes CUDA)
    if needs_cudnn(library_name, headers, source_path)? {
        if let Some(cudnn_info) = detect_cudnn(source_path)? {
            info!("✓ Auto-detected cuDNN dependency");
            dependencies.push(cudnn_info);
        } else {
            warn!("Library requires cuDNN but cuDNN installation not found, falling back to CUDA");
            // Fall back to just CUDA if cuDNN not found
            if let Some(cuda_info) = detect_cuda()? {
                info!("✓ Auto-detected CUDA dependency (fallback)");
                dependencies.push(cuda_info);
            }
        }
    }
    // Check for CUDA dependency (non-cuDNN)
    else if needs_cuda(library_name, headers, source_path)? {
        if let Some(cuda_info) = detect_cuda()? {
            info!("✓ Auto-detected CUDA dependency");
            dependencies.push(cuda_info);
        } else {
            warn!("Library requires CUDA but CUDA installation not found");
        }
    }

    // Check for other common dependencies
    // TODO: Add more dependency detection (OpenCL, Vulkan, DirectX, etc.)

    Ok(dependencies)
}

/// Check if the library needs cuDNN specifically
fn needs_cudnn(library_name: &str, headers: &[PathBuf], source_path: &Path) -> Result<bool> {
    // Check library name patterns
    let lib_name_lower = library_name.to_lowercase();
    if lib_name_lower.contains("cudnn") {
        debug!("Library name '{}' indicates cuDNN dependency", library_name);
        return Ok(true);
    }

    // Check for cuDNN headers being included
    if scan_for_cudnn_includes(headers, source_path)? {
        debug!("Found cuDNN header includes in library");
        return Ok(true);
    }

    Ok(false)
}

/// Check if the library needs CUDA
fn needs_cuda(library_name: &str, headers: &[PathBuf], source_path: &Path) -> Result<bool> {
    // Check library name patterns
    let lib_name_lower = library_name.to_lowercase();
    if lib_name_lower.contains("cublas")
        || lib_name_lower.contains("cufft")
        || lib_name_lower.contains("curand")
        || lib_name_lower.contains("cusparse")
        || lib_name_lower == "cuda"
    {
        debug!("Library name '{}' indicates CUDA dependency", library_name);
        return Ok(true);
    }

    // Check for CUDA headers being included
    if scan_for_cuda_includes(headers, source_path)? {
        debug!("Found CUDA header includes in library");
        return Ok(true);
    }

    Ok(false)
}

/// Scan headers for CUDA includes
fn scan_for_cuda_includes(headers: &[PathBuf], _source_path: &Path) -> Result<bool> {
    let cuda_headers = [
        "cuda_runtime.h",
        "cuda_runtime_api.h",
        "cuda.h",
        "driver_types.h",
        "vector_types.h",
    ];

    for header in headers {
        if let Ok(content) = std::fs::read_to_string(header) {
            for cuda_header in &cuda_headers {
                if content.contains(&format!("#include <{}>", cuda_header))
                    || content.contains(&format!("#include \"{}\"", cuda_header))
                {
                    debug!(
                        "Found CUDA include '{}' in {}",
                        cuda_header,
                        header.display()
                    );
                    return Ok(true);
                }
            }
        }
    }

    Ok(false)
}

/// Detect CUDA installation
fn detect_cuda() -> Result<Option<DependencyInfo>> {
    debug!("Searching for CUDA installation...");

    let cuda_info = if cfg!(target_os = "windows") {
        detect_cuda_windows()?
    } else {
        detect_cuda_unix()?
    };

    if let Some(info) = &cuda_info {
        info!(
            "Found CUDA at: {}",
            info.include_paths.first().unwrap().display()
        );
    }

    Ok(cuda_info)
}

/// Detect CUDA on Windows
fn detect_cuda_windows() -> Result<Option<DependencyInfo>> {
    // Common CUDA installation paths on Windows
    let cuda_paths = [
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA",
        r"C:\Program Files (x86)\NVIDIA GPU Computing Toolkit\CUDA",
        r"C:\CUDA",
    ];

    // Look for CUDA environment variable first
    if let Ok(cuda_path) = std::env::var("CUDA_PATH") {
        debug!("Checking CUDA_PATH: {}", cuda_path);
        let cuda_path = PathBuf::from(cuda_path);
        if let Some(info) = check_cuda_installation(&cuda_path)? {
            return Ok(Some(info));
        }
    }

    // Search common installation directories
    for base_path in &cuda_paths {
        let base_path = Path::new(base_path);
        if !base_path.exists() {
            continue;
        }

        debug!("Searching for CUDA versions in: {}", base_path.display());

        // Look for version directories (e.g., v11.0, v12.0, v13.0)
        if let Ok(entries) = std::fs::read_dir(base_path) {
            let mut versions = Vec::new();

            for entry in entries.flatten() {
                if let Ok(name) = entry.file_name().into_string()
                    && name.starts_with('v') && name.len() > 1 {
                        versions.push(name);
                    }
            }

            // Sort versions and try the latest first
            versions.sort_by(|a, b| b.cmp(a)); // Reverse sort for latest first

            for version in versions {
                let cuda_path = base_path.join(&version);
                debug!("Checking CUDA version: {}", cuda_path.display());

                if let Some(info) = check_cuda_installation(&cuda_path)? {
                    return Ok(Some(info));
                }
            }
        }
    }

    Ok(None)
}

/// Detect CUDA on Unix systems
fn detect_cuda_unix() -> Result<Option<DependencyInfo>> {
    // Common CUDA paths on Unix
    let cuda_paths = ["/usr/local/cuda", "/opt/cuda", "/usr/cuda"];

    // Check CUDA_HOME environment variable
    if let Ok(cuda_home) = std::env::var("CUDA_HOME") {
        let cuda_path = PathBuf::from(cuda_home);
        if let Some(info) = check_cuda_installation(&cuda_path)? {
            return Ok(Some(info));
        }
    }

    // Check common paths
    for path in &cuda_paths {
        let cuda_path = Path::new(path);
        if let Some(info) = check_cuda_installation(cuda_path)? {
            return Ok(Some(info));
        }
    }

    Ok(None)
}

/// Check if a path contains a valid CUDA installation
fn check_cuda_installation(cuda_path: &Path) -> Result<Option<DependencyInfo>> {
    let include_path = cuda_path.join("include");
    let cuda_runtime_header = include_path.join("cuda_runtime.h");

    if !cuda_runtime_header.exists() {
        debug!("CUDA headers not found at: {}", include_path.display());
        return Ok(None);
    }

    // Determine library paths based on platform
    let lib_paths = if cfg!(target_os = "windows") {
        if cfg!(target_arch = "x86_64") {
            vec![cuda_path.join("lib").join("x64")]
        } else {
            vec![cuda_path.join("lib").join("Win32")]
        }
    } else {
        vec![cuda_path.join("lib64"), cuda_path.join("lib")]
    };

    // Filter to existing library paths
    let existing_lib_paths: Vec<PathBuf> =
        lib_paths.into_iter().filter(|path| path.exists()).collect();

    if existing_lib_paths.is_empty() {
        debug!("No CUDA library paths found at: {}", cuda_path.display());
        return Ok(None);
    }

    debug!("Valid CUDA installation found at: {}", cuda_path.display());

    Ok(Some(DependencyInfo {
        name: "cuda".to_string(),
        include_paths: vec![include_path],
        lib_paths: existing_lib_paths,
        link_libs: vec!["cuda".to_string(), "cudart".to_string()],
    }))
}

/// Scan headers for cuDNN includes
fn scan_for_cudnn_includes(headers: &[PathBuf], _source_path: &Path) -> Result<bool> {
    let cudnn_headers = [
        "cudnn.h",
        "cudnn_v8.h",
        "cudnn_backend.h",
        "cudnn_frontend.h",
    ];

    for header in headers {
        if let Ok(content) = std::fs::read_to_string(header) {
            for cudnn_header in &cudnn_headers {
                if content.contains(&format!("#include <{}>", cudnn_header))
                    || content.contains(&format!("#include \"{}\"", cudnn_header))
                {
                    debug!(
                        "Found cuDNN include '{}' in {}",
                        cudnn_header,
                        header.display()
                    );
                    return Ok(true);
                }
            }
        }
    }

    Ok(false)
}

/// Detect cuDNN installation (includes CUDA as well)
fn detect_cudnn(source_path: &Path) -> Result<Option<DependencyInfo>> {
    debug!("Searching for cuDNN installation...");

    // First check if cuDNN is in the source path (like our case)
    if let Some(info) = check_cudnn_installation(source_path)? {
        return Ok(Some(info));
    }

    // Common cuDNN installation paths
    let cudnn_paths = if cfg!(target_os = "windows") {
        vec![
            PathBuf::from(r"C:\Program Files\NVIDIA\CUDNN"),
            PathBuf::from(r"C:\cudnn"),
        ]
    } else {
        vec![
            PathBuf::from("/usr/local/cudnn"),
            PathBuf::from("/opt/cudnn"),
        ]
    };

    for path in cudnn_paths {
        if let Some(info) = check_cudnn_installation(&path)? {
            return Ok(Some(info));
        }
    }

    Ok(None)
}

/// Check if a path contains a valid cuDNN installation
fn check_cudnn_installation(cudnn_path: &Path) -> Result<Option<DependencyInfo>> {
    // Look for cuDNN include directory
    let include_path = cudnn_path.join("include");
    let cudnn_header = include_path.join("cudnn.h");

    if !cudnn_header.exists() {
        debug!("cuDNN headers not found at: {}", include_path.display());
        return Ok(None);
    }

    // Look for cuDNN library directory
    let lib_path = cudnn_path.join("lib");
    if !lib_path.exists() {
        debug!("cuDNN lib directory not found at: {}", lib_path.display());
        return Ok(None);
    }

    // Check for cuDNN libraries
    let cudnn_lib = if cfg!(target_os = "windows") {
        lib_path.join("cudnn64_9.lib")
    } else {
        lib_path.join("libcudnn.so")
    };

    if !cudnn_lib.exists() {
        debug!("cuDNN library not found at: {}", cudnn_lib.display());
        return Ok(None);
    }

    debug!(
        "Valid cuDNN installation found at: {}",
        cudnn_path.display()
    );

    // Also include CUDA dependencies
    let mut include_paths = vec![include_path];
    let mut lib_paths = vec![lib_path];
    let mut link_libs = vec!["cudnn64_9".to_string()];

    // Add CUDA dependencies
    if let Some(cuda_info) = detect_cuda()? {
        include_paths.extend(cuda_info.include_paths);
        lib_paths.extend(cuda_info.lib_paths);
        link_libs.extend(cuda_info.link_libs);
    }

    Ok(Some(DependencyInfo {
        name: "cudnn".to_string(),
        include_paths,
        lib_paths,
        link_libs,
    }))
}
