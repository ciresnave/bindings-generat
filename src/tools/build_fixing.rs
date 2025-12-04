//! Build fixing tool - automatically applies fixes to resolve compilation errors

use super::{Tool, ToolContext, ToolResult};
use anyhow::Result;
use tracing::info;

pub struct BuildFixingTool;

impl Tool for BuildFixingTool {
    fn name(&self) -> &'static str {
        "fix_build_errors"
    }

    fn description(&self) -> &'static str {
        "Automatically applies fixes for common build errors like missing libraries or linker issues"
    }

    fn requirements(&self) -> Vec<&'static str> {
        vec!["build_errors"]
    }

    fn provides(&self) -> Vec<&'static str> {
        vec!["link_libs", "lib_paths", "include_dirs"]
    }

    fn execute(&self, mut context: ToolContext) -> Result<ToolResult> {
        if context.build_errors.is_empty() {
            return Ok(ToolResult {
                success: true,
                message: "No build errors to fix".to_string(),
                updated_context: context,
                suggestions: vec![],
            });
        }

        let mut fixes_applied = Vec::new();
        let mut suggestions = Vec::new();

        // Clone the errors to avoid borrowing issues
        let build_errors = context.build_errors.clone();

        // Analyze errors and apply automatic fixes
        for error in &build_errors {
            match error {
                crate::output::error_parser::BuildError::UnresolvedSymbol { symbol, .. } => {
                    if let Some(_fix) = self.fix_unresolved_symbol(symbol, &mut context) {
                        fixes_applied.push(format!("Added library for symbol: {}", symbol));
                        info!("Applied fix for unresolved symbol: {}", symbol);
                    } else {
                        suggestions.push(format!("Could not resolve symbol: {}. Consider adding the appropriate library manually.", symbol));
                    }
                }
                crate::output::error_parser::BuildError::LibraryNotFound { library_name } => {
                    if let Some(_fix) = self.fix_missing_library(library_name, &mut context) {
                        fixes_applied.push(format!("Added library path for: {}", library_name));
                        info!("Applied fix for missing library: {}", library_name);
                    } else {
                        suggestions.push(format!("Library not found: {}. Install the library or specify its path manually.", library_name));
                    }
                }
                crate::output::error_parser::BuildError::MissingHeader { header_name } => {
                    if let Some(_fix) = self.fix_missing_header(header_name, &mut context) {
                        fixes_applied.push(format!("Added include path for: {}", header_name));
                        info!("Applied fix for missing header: {}", header_name);
                    } else {
                        suggestions.push(format!("Header not found: {}. Install development headers or specify include path manually.", header_name));
                    }
                }
                crate::output::error_parser::BuildError::CompilationError { message } => {
                    suggestions.push(format!(
                        "Compilation error: {}. Manual intervention required.",
                        message
                    ));
                }
            }
        }

        let message = if fixes_applied.is_empty() {
            "No automatic fixes could be applied".to_string()
        } else {
            format!(
                "Applied {} automatic fixes: {}",
                fixes_applied.len(),
                fixes_applied.join(", ")
            )
        };

        Ok(ToolResult {
            success: !fixes_applied.is_empty(),
            message,
            updated_context: context,
            suggestions,
        })
    }
}

impl BuildFixingTool {
    /// Attempt to fix unresolved symbol by finding the appropriate library
    fn fix_unresolved_symbol(&self, symbol: &str, context: &mut ToolContext) -> Option<String> {
        // Known symbol-to-library mappings
        let symbol_mappings = self.get_symbol_library_mappings();

        // Try exact matches first
        for (pattern, libs) in &symbol_mappings {
            if self.symbol_matches_pattern(symbol, pattern) {
                for lib in libs {
                    if !context.link_libs.contains(lib) {
                        context.link_libs.push(lib.clone());
                        return Some(format!("Added library: {}", lib));
                    }
                }
            }
        }

        // Try to infer library from symbol prefix
        if let Some(lib) = self.infer_library_from_symbol(symbol)
            && !context.link_libs.contains(&lib) {
                context.link_libs.push(lib.clone());
                return Some(format!("Inferred and added library: {}", lib));
            }

        None
    }

    /// Attempt to fix missing library by finding it in system paths
    fn fix_missing_library(&self, library_name: &str, context: &mut ToolContext) -> Option<String> {
        // Search in common system library paths
        let search_paths = self.get_system_library_paths();

        for search_path in &search_paths {
            if let Ok(entries) = std::fs::read_dir(search_path) {
                for entry in entries.flatten() {
                    let path = entry.path();
                    if path.is_dir() {
                        let lib_files = self.find_library_files(&path, library_name);
                        if !lib_files.is_empty()
                            && !context.lib_paths.contains(&path) {
                                context.lib_paths.push(path.clone());
                                return Some(format!("Found library in: {}", path.display()));
                            }
                    }
                }
            }
        }

        None
    }

    /// Attempt to fix missing header by finding it in system paths  
    fn fix_missing_header(&self, header_path: &str, context: &mut ToolContext) -> Option<String> {
        // Search in common system include paths
        let search_paths = self.get_system_include_paths();

        let header_name = std::path::Path::new(header_path)
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or(header_path);

        for search_path in &search_paths {
            if let Ok(entries) = std::fs::read_dir(search_path) {
                for entry in entries.flatten() {
                    let path = entry.path();
                    if path.is_dir() {
                        let header_file = path.join(header_name);
                        if header_file.exists()
                            && !context.include_dirs.contains(&path) {
                                context.include_dirs.push(path.clone());
                                return Some(format!("Found header in: {}", path.display()));
                            }
                    }
                }
            }
        }

        None
    }

    /// Get known symbol-to-library mappings
    fn get_symbol_library_mappings(&self) -> Vec<(String, Vec<String>)> {
        vec![
            // CUDA symbols
            (
                "cuda*".to_string(),
                vec!["cuda".to_string(), "cudart".to_string()],
            ),
            ("cu*".to_string(), vec!["cuda".to_string()]),
            // cuDNN symbols
            (
                "cudnn*".to_string(),
                if cfg!(target_os = "windows") {
                    vec!["cudnn64_9".to_string()]
                } else {
                    vec!["cudnn".to_string()]
                },
            ),
            // Math libraries
            ("blas*".to_string(), vec!["blas".to_string()]),
            ("cblas*".to_string(), vec!["blas".to_string()]),
            ("lapack*".to_string(), vec!["lapack".to_string()]),
            // Graphics
            (
                "gl*".to_string(),
                vec!["GL".to_string(), "opengl32".to_string()],
            ),
            ("vk*".to_string(), vec!["vulkan".to_string()]),
            // System libraries
            ("pthread*".to_string(), vec!["pthread".to_string()]),
            ("m_*".to_string(), vec!["m".to_string()]),
        ]
    }

    fn symbol_matches_pattern(&self, symbol: &str, pattern: &str) -> bool {
        if let Some(prefix) = pattern.strip_suffix('*') {
            symbol.starts_with(prefix)
        } else {
            symbol == pattern
        }
    }

    fn infer_library_from_symbol(&self, symbol: &str) -> Option<String> {
        // Simple heuristics for inferring library names from symbols
        if symbol.starts_with("cuda") && symbol.len() > 4 {
            Some("cuda".to_string())
        } else if symbol.starts_with("cu")
            && symbol.len() > 2
            && symbol.chars().nth(2).unwrap().is_uppercase()
        {
            Some("cuda".to_string())
        } else if symbol.starts_with("gl")
            && symbol.len() > 2
            && symbol.chars().nth(2).unwrap().is_uppercase()
        {
            Some("GL".to_string())
        } else {
            None
        }
    }

    fn get_system_library_paths(&self) -> Vec<std::path::PathBuf> {
        if cfg!(target_os = "windows") {
            vec![
                std::path::PathBuf::from(r"C:\Program Files"),
                std::path::PathBuf::from(r"C:\Program Files (x86)"),
                std::path::PathBuf::from(r"C:\Windows\System32"),
            ]
        } else {
            vec![
                std::path::PathBuf::from("/usr/lib"),
                std::path::PathBuf::from("/usr/local/lib"),
                std::path::PathBuf::from("/lib"),
                std::path::PathBuf::from("/opt"),
            ]
        }
    }

    fn get_system_include_paths(&self) -> Vec<std::path::PathBuf> {
        if cfg!(target_os = "windows") {
            vec![
                std::path::PathBuf::from(r"C:\Program Files"),
                std::path::PathBuf::from(r"C:\Program Files (x86)"),
            ]
        } else {
            vec![
                std::path::PathBuf::from("/usr/include"),
                std::path::PathBuf::from("/usr/local/include"),
                std::path::PathBuf::from("/opt"),
            ]
        }
    }

    fn find_library_files(
        &self,
        dir: &std::path::Path,
        library_name: &str,
    ) -> Vec<std::path::PathBuf> {
        let mut found = Vec::new();

        if let Ok(entries) = std::fs::read_dir(dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if let Some(file_name) = path.file_name().and_then(|n| n.to_str()) {
                    // Check for various library naming patterns
                    let is_lib = if cfg!(target_os = "windows") {
                        file_name.ends_with(".lib") || file_name.ends_with(".dll")
                    } else {
                        file_name.starts_with("lib")
                            && (file_name.ends_with(".so") || file_name.ends_with(".a"))
                    };

                    if is_lib && file_name.contains(library_name) {
                        found.push(path);
                    }
                }
            }
        }

        found
    }
}
