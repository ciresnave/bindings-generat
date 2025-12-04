//! Library searching and validation methods for enhanced dependency detection

use super::enhanced_dependency_detection::*;
use anyhow::Result;
use std::path::{Path, PathBuf};
use std::process::Command;
use tracing::{debug, info, warn};

impl EnhancedDependencyDetectionTool {
    /// Search for libraries using environment variables
    pub fn search_env_vars(&self, pattern: &DependencyPattern) -> Result<Vec<LibraryCandidate>> {
        let mut candidates = Vec::new();

        for env_var in &pattern.env_vars {
            if let Ok(path) = std::env::var(env_var) {
                let base_path = PathBuf::from(path);
                if base_path.exists() {
                    candidates.extend(self.scan_directory_for_libs(&base_path, &pattern.link_libs, format!("env:{}", env_var))?);
                }
            }
        }

        Ok(candidates)
    }

    /// Search common installation paths
    pub fn search_common_paths(&self, pattern: &DependencyPattern) -> Result<Vec<LibraryCandidate>> {
        let mut candidates = Vec::new();

        for path_pattern in &pattern.common_paths {
            if path_pattern.contains('*') {
                // Handle glob patterns
                candidates.extend(self.search_glob_pattern(path_pattern, &pattern.link_libs)?);
            } else {
                let path = PathBuf::from(path_pattern);
                if path.exists() {
                    candidates.extend(self.scan_directory_for_libs(&path, &pattern.link_libs, format!("common_path:{}", path_pattern))?);
                }
            }
        }

        Ok(candidates)
    }

    /// Search using pkg-config
    pub fn search_pkg_config(&self, pkg_name: &str) -> Result<Vec<LibraryCandidate>> {
        let mut candidates = Vec::new();

        // Try pkg-config
        if let Ok(output) = Command::new("pkg-config")
            .args(&["--libs-only-L", pkg_name])
            .output()
        {
            if output.status.success() {
                let libs_output = String::from_utf8_lossy(&output.stdout);
                for line in libs_output.lines() {
                    if line.starts_with("-L") {
                        let lib_path = PathBuf::from(&line[2..]);
                        if lib_path.exists() {
                            // Get library names from --libs-only-l
                            if let Ok(lib_output) = Command::new("pkg-config")
                                .args(&["--libs-only-l", pkg_name])
                                .output()
                            {
                                if lib_output.status.success() {
                                    let lib_names_output = String::from_utf8_lossy(&lib_output.stdout);
                                    let lib_names: Vec<String> = lib_names_output
                                        .split_whitespace()
                                        .filter_map(|s| {
                                            if s.starts_with("-l") {
                                                Some(s[2..].to_string())
                                            } else {
                                                None
                                            }
                                        })
                                        .collect();
                                    
                                    candidates.extend(self.scan_directory_for_libs(&lib_path, &lib_names, format!("pkg-config:{}", pkg_name))?);
                                }
                            }
                        }
                    }
                }
            }
        }

        Ok(candidates)
    }

    /// System-wide library search
    pub fn system_wide_library_search(&self, lib_names: &[String]) -> Result<Vec<LibraryCandidate>> {
        let mut candidates = Vec::new();

        // Search common system library paths
        let system_paths = if cfg!(target_os = "windows") {
            vec![
                PathBuf::from(r"C:\Windows\System32"),
                PathBuf::from(r"C:\Windows\SysWOW64"),
            ]
        } else {
            vec![
                PathBuf::from("/usr/lib"),
                PathBuf::from("/usr/local/lib"),
                PathBuf::from("/lib"),
                PathBuf::from("/usr/lib/x86_64-linux-gnu"),
                PathBuf::from("/usr/lib64"),
            ]
        };

        for path in system_paths {
            if path.exists() {
                candidates.extend(self.scan_directory_for_libs(&path, lib_names, format!("system_path:{}", path.display()))?);
            }
        }

        // Also search LD_LIBRARY_PATH on Unix
        if !cfg!(target_os = "windows") {
            if let Ok(ld_path) = std::env::var("LD_LIBRARY_PATH") {
                for path_str in ld_path.split(':') {
                    let path = PathBuf::from(path_str);
                    if path.exists() {
                        candidates.extend(self.scan_directory_for_libs(&path, lib_names, format!("LD_LIBRARY_PATH:{}", path_str))?);
                    }
                }
            }
        }

        // Search PATH on Windows for DLLs
        if cfg!(target_os = "windows") {
            if let Ok(path_var) = std::env::var("PATH") {
                for path_str in path_var.split(';') {
                    let path = PathBuf::from(path_str);
                    if path.exists() {
                        candidates.extend(self.scan_directory_for_libs(&path, lib_names, format!("PATH:{}", path_str))?);
                    }
                }
            }
        }

        Ok(candidates)
    }

    /// Scan a directory for library files
    fn scan_directory_for_libs(&self, dir: &Path, lib_names: &[String], source: String) -> Result<Vec<LibraryCandidate>> {
        let mut candidates = Vec::new();

        if let Ok(entries) = std::fs::read_dir(dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_file() {
                    let file_name = path.file_name().unwrap().to_string_lossy();
                    
                    // Check if this file matches any of our library names
                    for lib_name in lib_names {
                        if self.is_library_file(&file_name, lib_name) {
                            candidates.push(LibraryCandidate {
                                name: lib_name.clone(),
                                path: path.clone(),
                                confidence: 0.7, // Default confidence for file name match
                                found_symbols: vec![], // Will be populated during validation
                                compilation_test_passed: None,
                                source: source.clone(),
                            });
                        }
                    }
                }
            }
        }

        Ok(candidates)
    }

    /// Check if a filename matches a library name
    fn is_library_file(&self, filename: &str, lib_name: &str) -> bool {
        if cfg!(target_os = "windows") {
            // Windows: look for .dll files
            filename.to_lowercase() == format!("{}.dll", lib_name.to_lowercase()) ||
            filename.to_lowercase().starts_with(&format!("{}", lib_name.to_lowercase())) && filename.ends_with(".dll")
        } else {
            // Unix: look for .so files
            filename == format!("lib{}.so", lib_name) ||
            filename.starts_with(&format!("lib{}.so.", lib_name)) ||
            filename == format!("lib{}.a", lib_name)
        }
    }

    /// Search using glob patterns
    fn search_glob_pattern(&self, pattern: &str, lib_names: &[String]) -> Result<Vec<LibraryCandidate>> {
        let mut candidates = Vec::new();

        // Simple glob implementation - replace * with actual directory names
        if let Some(star_pos) = pattern.find('*') {
            let before = &pattern[..star_pos];
            let after = &pattern[star_pos + 1..];
            
            if let Some(parent) = Path::new(before).parent() {
                if parent.exists() {
                    if let Ok(entries) = std::fs::read_dir(parent) {
                        for entry in entries.flatten() {
                            let path = entry.path();
                            if path.is_dir() {
                                let dir_name = path.file_name().unwrap().to_string_lossy();
                                let full_path = PathBuf::from(format!("{}{}{}", before, dir_name, after));
                                if full_path.exists() {
                                    candidates.extend(self.scan_directory_for_libs(&full_path, lib_names, format!("glob:{}", pattern))?);
                                }
                            }
                        }
                    }
                }
            }
        }

        Ok(candidates)
    }

    /// Validate library candidates by checking for symbols
    pub async fn validate_library_candidates(&self, candidates: &[LibraryCandidate], pattern: &DependencyPattern) -> Result<Vec<LibraryCandidate>> {
        let mut validated = Vec::new();

        for candidate in candidates {
            let mut validated_candidate = candidate.clone();
            
            // Check if the library file contains expected symbols
            if let Ok(symbols) = self.extract_library_symbols(&candidate.path) {
                let mut found_symbols = Vec::new();
                
                // Check which pattern functions are present in this library
                for func_pattern in &pattern.function_patterns {
                    for symbol in &symbols {
                        if self.matches_pattern(symbol, func_pattern) {
                            found_symbols.push(symbol.clone());
                        }
                    }
                }
                
                validated_candidate.found_symbols = found_symbols.clone();
                
                if !found_symbols.is_empty() {
                    // Increase confidence based on how many symbols we found
                    let symbol_ratio = found_symbols.len() as f32 / pattern.function_patterns.len() as f32;
                    validated_candidate.confidence = (candidate.confidence + symbol_ratio) / 2.0;
                    
                    // Test compilation if we have high confidence
                    if validated_candidate.confidence > 0.6 {
                        validated_candidate.compilation_test_passed = Some(self.test_compilation_with_library(&validated_candidate, pattern).await?);
                        
                        if validated_candidate.compilation_test_passed == Some(true) {
                            validated_candidate.confidence = validated_candidate.confidence.max(0.9);
                        }
                    }
                    
                    validated.push(validated_candidate);
                }
            }
        }

        // Sort by confidence
        validated.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());

        Ok(validated)
    }

    /// Extract symbols from a library file
    fn extract_library_symbols(&self, lib_path: &Path) -> Result<Vec<String>> {
        let mut symbols = Vec::new();

        if cfg!(target_os = "windows") {
            // On Windows, use dumpbin if available
            if let Ok(output) = Command::new("dumpbin")
                .args(&["/EXPORTS", &lib_path.to_string_lossy()])
                .output()
            {
                if output.status.success() {
                    let output_str = String::from_utf8_lossy(&output.stdout);
                    for line in output_str.lines() {
                        // Parse dumpbin output to extract symbol names
                        if let Some(symbol) = self.parse_dumpbin_line(line) {
                            symbols.push(symbol);
                        }
                    }
                }
            }
        } else {
            // On Unix, use nm or objdump
            if let Ok(output) = Command::new("nm")
                .args(&["-D", &lib_path.to_string_lossy()])
                .output()
            {
                if output.status.success() {
                    let output_str = String::from_utf8_lossy(&output.stdout);
                    for line in output_str.lines() {
                        // Parse nm output: "address type symbol_name"
                        if let Some(symbol) = line.split_whitespace().nth(2) {
                            symbols.push(symbol.to_string());
                        }
                    }
                }
            }
        }

        Ok(symbols)
    }

    /// Parse a dumpbin output line to extract symbol name
    fn parse_dumpbin_line(&self, line: &str) -> Option<String> {
        // Dumpbin output format varies, but typically:
        // "ordinal hint RVA      name"
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() >= 4 {
            // The last part is usually the symbol name
            Some(parts.last().unwrap().to_string())
        } else {
            None
        }
    }

    /// Test compilation with a specific library
    async fn test_compilation_with_library(&self, candidate: &LibraryCandidate, pattern: &DependencyPattern) -> Result<bool> {
        // Create a minimal test program that uses some functions from the pattern
        let test_code = self.generate_test_code(pattern)?;
        
        // Create a temporary directory for the test
        let temp_dir = tempfile::tempdir()?;
        let test_file = temp_dir.path().join("test.c");
        std::fs::write(&test_file, test_code)?;

        // Try to compile with this library
        let mut compile_cmd = Command::new("gcc"); // or appropriate compiler
        compile_cmd.arg("-c")
                   .arg(&test_file)
                   .arg("-o")
                   .arg(temp_dir.path().join("test.o"));

        // Add library path and link library
        if let Some(lib_dir) = candidate.path.parent() {
            compile_cmd.arg(format!("-L{}", lib_dir.display()));
        }
        compile_cmd.arg(format!("-l{}", candidate.name));

        let output = compile_cmd.output()?;
        Ok(output.status.success())
    }

    /// Generate test code for compilation testing
    fn generate_test_code(&self, pattern: &DependencyPattern) -> Result<String> {
        let mut code = String::new();
        
        // Add headers
        for header in &pattern.header_indicators {
            code.push_str(&format!("#include <{}>\n", header));
        }
        
        code.push_str("\nint main() {\n");
        
        // Add some basic function calls (just declarations, not actual calls to avoid runtime errors)
        for func_pattern in &pattern.function_patterns {
            if !func_pattern.contains('*') {
                code.push_str(&format!("    // {} function would be used here\n", func_pattern));
            }
        }
        
        code.push_str("    return 0;\n}\n");
        
        Ok(code)
    }

    /// Handle multiple library candidates - ask user to choose
    pub async fn handle_multiple_candidates(&self, candidates: &[LibraryCandidate], pattern: &DependencyPattern) -> Result<LibraryCandidate> {
        println!("\n🔍 Found multiple libraries for '{}':", pattern.name);
        
        for (i, candidate) in candidates.iter().enumerate() {
            println!("  {}. {} (confidence: {:.2}, source: {})", 
                i + 1, 
                candidate.path.display(),
                candidate.confidence,
                candidate.source
            );
            
            if !candidate.found_symbols.is_empty() {
                println!("     Found symbols: {}", candidate.found_symbols.join(", "));
            }
            
            if let Some(compilation_passed) = candidate.compilation_test_passed {
                println!("     Compilation test: {}", if compilation_passed { "✅ PASSED" } else { "❌ FAILED" });
            }
        }
        
        println!("Please choose a library (1-{}, or 'auto' for highest confidence): ", candidates.len());
        
        use std::io::{self, Write};
        io::stdout().flush()?;
        
        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let choice = input.trim();
        
        if choice == "auto" || choice.is_empty() {
            // Return the highest confidence candidate
            Ok(candidates[0].clone())
        } else if let Ok(index) = choice.parse::<usize>() {
            if index > 0 && index <= candidates.len() {
                Ok(candidates[index - 1].clone())
            } else {
                warn!("Invalid choice, using highest confidence candidate");
                Ok(candidates[0].clone())
            }
        } else {
            warn!("Invalid input, using highest confidence candidate");
            Ok(candidates[0].clone())
        }
    }

    /// Build DependencyInfo from a chosen candidate
    pub fn build_dependency_info(&self, pattern: &DependencyPattern, candidate: &LibraryCandidate) -> Result<DependencyInfo> {
        let mut dep_info = DependencyInfo {
            name: pattern.name.clone(),
            include_dirs: vec![],
            lib_paths: vec![],
            link_libs: vec![candidate.name.clone()],
            pattern_used: Some(pattern.clone()),
            library_candidates: vec![candidate.clone()],
            user_choice: Some(candidate.clone()),
        };

        // Add library path
        if let Some(lib_dir) = candidate.path.parent() {
            dep_info.lib_paths.push(lib_dir.to_path_buf());
        }

        // Try to find include directories
        // Look for include dirs relative to the library
        if let Some(lib_dir) = candidate.path.parent() {
            // Try going up directories to find include paths
            let mut current = lib_dir;
            for _ in 0..3 { // Search up to 3 levels up
                for include_subpath in &pattern.include_subpaths {
                    let include_path = current.join(include_subpath);
                    if include_path.exists() {
                        dep_info.include_dirs.push(include_path);
                    }
                }
                if let Some(parent) = current.parent() {
                    current = parent;
                } else {
                    break;
                }
            }
        }

        Ok(dep_info)
    }
}

#[derive(serde::Deserialize)]
struct LlmDependencyResponse {
    suggestions: Vec<LlmLibrarySuggestion>,
    user_questions: Option<Vec<LlmUserQuestion>>,
}

#[derive(serde::Deserialize)]
struct LlmLibrarySuggestion {
    library_name: String,
    confidence: f32,
    reasoning: String,
    function_patterns: Vec<String>,
    headers: Vec<String>,
    typical_paths: Vec<String>,
}

#[derive(serde::Deserialize)]
struct LlmUserQuestion {
    functions: Vec<String>,
    question: String,
}