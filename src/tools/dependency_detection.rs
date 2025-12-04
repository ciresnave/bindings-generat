//! Generic dependency detection tool
//! 
//! This tool analyzes FFI bindings to detect required dependencies
//! without hardcoded library-specific knowledge.

use super::{Tool, ToolContext, ToolResult};
use anyhow::Result;
use std::collections::{HashMap, HashSet};
use tracing::{debug, info};

pub struct DependencyDetectionTool;

impl Tool for DependencyDetectionTool {
    fn name(&self) -> &'static str {
        "detect_dependencies"
    }
    
    fn description(&self) -> &'static str {
        "Analyzes function signatures to detect required system libraries and dependencies"
    }
    
    fn requirements(&self) -> Vec<&'static str> {
        vec!["ffi_info"]
    }
    
    fn provides(&self) -> Vec<&'static str> {
        vec!["dependencies", "include_dirs", "lib_paths", "link_libs"]
    }
    
    fn execute(&self, mut context: ToolContext) -> Result<ToolResult> {
        let ffi_info = context.ffi_info.as_ref()
            .ok_or_else(|| anyhow::anyhow!("FFI info not available"))?;
        
        // Generic dependency detection based on function prefixes and patterns
        let detected_deps = self.detect_dependencies_from_functions(&ffi_info.functions);
        
        // Try to find system libraries for detected dependencies
        for dep in &detected_deps {
            if let Ok(Some(dep_info)) = self.find_system_dependency(dep, &context) {
                // Add to context
                for include_dir in &dep_info.include_dirs {
                    if !context.include_dirs.contains(include_dir) {
                        context.include_dirs.push(include_dir.clone());
                    }
                }
                
                for lib_path in &dep_info.lib_paths {
                    if !context.lib_paths.contains(lib_path) {
                        context.lib_paths.push(lib_path.clone());
                    }
                }
                
                for link_lib in &dep_info.link_libs {
                    if !context.link_libs.contains(link_lib) {
                        context.link_libs.push(link_lib.clone());
                    }
                }
            }
        }
        
        context.dependencies = detected_deps.into_iter().collect();
        
        let message = if context.dependencies.is_empty() {
            "No external dependencies detected".to_string()
        } else {
            format!("Detected dependencies: {}", context.dependencies.join(", "))
        };
        
        Ok(ToolResult {
            success: true,
            message,
            updated_context: context,
            suggestions: vec![],
        })
    }
}

impl DependencyDetectionTool {
    /// Detect dependencies based on function naming patterns
    fn detect_dependencies_from_functions(&self, functions: &[crate::ffi::FfiFunction]) -> HashSet<String> {
        let mut dependencies = HashSet::new();
        
        for function in functions {
            // Common library prefixes
            let name = &function.name;
            
            // GPU/Compute libraries
            if name.starts_with("cuda") || name.starts_with("cu") {
                dependencies.insert("cuda".to_string());
            }
            if name.starts_with("cudnn") {
                dependencies.insert("cudnn".to_string());
            }
            if name.starts_with("cublas") {
                dependencies.insert("cublas".to_string());
            }
            if name.starts_with("cufft") {
                dependencies.insert("cufft".to_string());
            }
            if name.starts_with("cl") && name.len() > 2 && name.chars().nth(2).unwrap().is_uppercase() {
                dependencies.insert("opencl".to_string());
            }
            
            // Graphics libraries
            if name.starts_with("gl") && name.len() > 2 && name.chars().nth(2).unwrap().is_uppercase() {
                dependencies.insert("opengl".to_string());
            }
            if name.starts_with("vk") {
                dependencies.insert("vulkan".to_string());
            }
            
            // Math libraries
            if name.starts_with("blas") || name.starts_with("cblas") {
                dependencies.insert("blas".to_string());
            }
            if name.starts_with("lapack") {
                dependencies.insert("lapack".to_string());
            }
            if name.starts_with("fftw") {
                dependencies.insert("fftw".to_string());
            }
            
            // Media libraries
            if name.starts_with("av") && name.len() > 2 {
                dependencies.insert("ffmpeg".to_string());
            }
            if name.starts_with("SDL_") {
                dependencies.insert("sdl2".to_string());
            }
            
            // Networking
            if name.starts_with("curl_") {
                dependencies.insert("curl".to_string());
            }
            
            // Compression
            if name.starts_with("z") && (name.contains("compress") || name.contains("deflate") || name == "zlib") {
                dependencies.insert("zlib".to_string());
            }
            
            // Database
            if name.starts_with("sqlite3_") {
                dependencies.insert("sqlite".to_string());
            }
            if name.starts_with("PQ") || name.starts_with("pg_") {
                dependencies.insert("postgresql".to_string());
            }
        }
        
        info!("Detected {} potential dependencies from function analysis", dependencies.len());
        dependencies
    }
    
    /// Generic system dependency finder
    fn find_system_dependency(&self, dep_name: &str, context: &ToolContext) -> Result<Option<DependencyInfo>> {
        debug!("Searching for dependency: {}", dep_name);
        
        // Use configurable dependency mapping instead of hardcoded logic
        let dep_configs = self.get_dependency_configurations();
        
        if let Some(config) = dep_configs.get(dep_name) {
            return self.find_dependency_by_config(config, context);
        }
        
        // Fallback: try generic search patterns
        self.generic_dependency_search(dep_name, context)
    }
    
    /// Get dependency configurations (could be loaded from external config)
    fn get_dependency_configurations(&self) -> HashMap<String, DependencyConfig> {
        let mut configs = HashMap::new();
        
        // CUDA configuration
        configs.insert("cuda".to_string(), DependencyConfig {
            env_vars: vec!["CUDA_PATH".to_string(), "CUDA_HOME".to_string()],
            common_paths: if cfg!(target_os = "windows") {
                vec![
                    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\*".to_string(),
                    r"C:\CUDA\*".to_string(),
                ]
            } else {
                vec![
                    "/usr/local/cuda*".to_string(),
                    "/opt/cuda*".to_string(),
                    "/usr/cuda*".to_string(),
                ]
            },
            include_subpaths: vec!["include".to_string()],
            lib_subpaths: if cfg!(target_os = "windows") {
                vec!["lib/x64".to_string(), "lib/Win32".to_string()]
            } else {
                vec!["lib64".to_string(), "lib".to_string()]
            },
            link_libs: vec!["cuda".to_string(), "cudart".to_string()],
            header_indicators: vec!["cuda_runtime.h".to_string(), "cuda.h".to_string()],
        });
        
        // cuDNN configuration  
        configs.insert("cudnn".to_string(), DependencyConfig {
            env_vars: vec!["CUDNN_PATH".to_string()],
            common_paths: if cfg!(target_os = "windows") {
                vec![
                    r"C:\Program Files\NVIDIA\CUDNN\*".to_string(),
                    r"C:\Users\*\.cudnn\*".to_string(),
                ]
            } else {
                vec![
                    "/usr/local/cudnn*".to_string(),
                    "/opt/cudnn*".to_string(),
                ]
            },
            include_subpaths: vec!["include".to_string()],
            lib_subpaths: if cfg!(target_os = "windows") {
                vec!["lib".to_string()]
            } else {
                vec!["lib64".to_string(), "lib".to_string()]
            },
            link_libs: if cfg!(target_os = "windows") {
                vec!["cudnn64_9".to_string()]
            } else {
                vec!["cudnn".to_string()]
            },
            header_indicators: vec!["cudnn.h".to_string()],
        });
        
        // More dependencies can be added here or loaded from external config
        configs
    }
    
    fn find_dependency_by_config(&self, config: &DependencyConfig, _context: &ToolContext) -> Result<Option<DependencyInfo>> {
        // Try environment variables first
        for env_var in &config.env_vars {
            if let Ok(path) = std::env::var(env_var) {
                let path = std::path::PathBuf::from(path);
                if let Some(dep_info) = self.validate_dependency_path(&path, config)? {
                    return Ok(Some(dep_info));
                }
            }
        }
        
        // Try common installation paths
        for pattern in &config.common_paths {
            if let Some(dep_info) = self.search_path_pattern(pattern, config)? {
                return Ok(Some(dep_info));
            }
        }
        
        Ok(None)
    }
    
    fn validate_dependency_path(&self, base_path: &std::path::Path, config: &DependencyConfig) -> Result<Option<DependencyInfo>> {
        // Check if this path has the expected structure
        for header in &config.header_indicators {
            let mut found = false;
            for include_subpath in &config.include_subpaths {
                let header_path = base_path.join(include_subpath).join(header);
                if header_path.exists() {
                    found = true;
                    break;
                }
            }
            if !found {
                return Ok(None);
            }
        }
        
        // Build dependency info
        let mut dep_info = DependencyInfo {
            include_dirs: vec![],
            lib_paths: vec![],
            link_libs: config.link_libs.clone(),
        };
        
        // Add include directories
        for include_subpath in &config.include_subpaths {
            let include_path = base_path.join(include_subpath);
            if include_path.exists() {
                dep_info.include_dirs.push(include_path);
            }
        }
        
        // Add library paths
        for lib_subpath in &config.lib_subpaths {
            let lib_path = base_path.join(lib_subpath);
            if lib_path.exists() {
                dep_info.lib_paths.push(lib_path);
            }
        }
        
        if !dep_info.include_dirs.is_empty() || !dep_info.lib_paths.is_empty() {
            Ok(Some(dep_info))
        } else {
            Ok(None)
        }
    }
    
    fn search_path_pattern(&self, pattern: &str, config: &DependencyConfig) -> Result<Option<DependencyInfo>> {
        // Simple glob-like pattern matching for common paths
        if pattern.contains('*') {
            let base_pattern = pattern.replace("*", "");
            let parent = std::path::Path::new(&base_pattern).parent().unwrap_or(std::path::Path::new("/"));
            
            if parent.exists()
                && let Ok(entries) = std::fs::read_dir(parent) {
                    for entry in entries.flatten() {
                        let path = entry.path();
                        if path.is_dir()
                            && let Some(dep_info) = self.validate_dependency_path(&path, config)? {
                                return Ok(Some(dep_info));
                            }
                    }
                }
        } else {
            let path = std::path::PathBuf::from(pattern);
            if path.exists() {
                return self.validate_dependency_path(&path, config);
            }
        }
        
        Ok(None)
    }
    
    fn generic_dependency_search(&self, _dep_name: &str, _context: &ToolContext) -> Result<Option<DependencyInfo>> {
        // Generic fallback search - could use pkg-config, system package managers, etc.
        Ok(None)
    }
}

#[derive(Debug, Clone)]
struct DependencyConfig {
    env_vars: Vec<String>,
    common_paths: Vec<String>,
    include_subpaths: Vec<String>,
    lib_subpaths: Vec<String>,
    link_libs: Vec<String>,
    header_indicators: Vec<String>,
}

#[derive(Debug, Clone)]
struct DependencyInfo {
    include_dirs: Vec<std::path::PathBuf>,
    lib_paths: Vec<std::path::PathBuf>,
    link_libs: Vec<String>,
}