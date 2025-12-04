use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use tracing::{debug, info};

/// Represents an actionable error from a build failure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BuildError {
    /// Missing library dependency (unresolved external symbol)
    UnresolvedSymbol {
        symbol: String,
        object_file: Option<String>,
    },
    /// Library not found by linker
    LibraryNotFound { library_name: String },
    /// Missing header file
    MissingHeader { header_name: String },
    /// Generic compilation error
    CompilationError { message: String },
}

/// Parse cargo build output to extract actionable errors
pub fn parse_build_errors(build_output: &str) -> Vec<BuildError> {
    let mut errors = Vec::new();

    // Count lines with "unresolved external symbol" for debugging
    let unresolved_count = build_output
        .lines()
        .filter(|l| l.contains("unresolved external symbol"))
        .count();
    if unresolved_count > 0 {
        info!(
            "Found {} lines with 'unresolved external symbol'",
            unresolved_count
        );
    }

    for line in build_output.lines() {
        // Windows MSVC linker errors: "unresolved external symbol"
        if line.contains("unresolved external symbol") {
            if let Some(error) = parse_msvc_unresolved_symbol(line) {
                errors.push(error);
            } else {
                debug!("Failed to parse unresolved symbol from: {}", line);
            }
        }
        // Linux/Mac ld linker errors: "undefined reference"
        else if line.contains("undefined reference to") {
            if let Some(error) = parse_ld_undefined_reference(line) {
                errors.push(error);
            }
        }
        // Library not found
        else if line.contains("cannot find -l") {
            if let Some(error) = parse_library_not_found(line) {
                errors.push(error);
            }
        }
        // Missing header
        else if line.contains("fatal error:") && line.contains(".h")
            && let Some(error) = parse_missing_header(line) {
                errors.push(error);
            }
    }

    info!("Parsed {} build errors", errors.len());
    errors
}

/// Extract function name from mangled C++ symbol
fn demangle_symbol(mangled: &str) -> String {
    // Try to extract a readable function name
    // This is a simplified version - could use cpp_demangle crate for full demangling

    // Remove common prefixes
    let cleaned = mangled.trim_start_matches("_Z").trim_start_matches("__Z");

    // Look for recognizable patterns
    if let Some(pos) = cleaned.find("E")
        && let Some(name_part) = cleaned.get(0..pos) {
            // Try to extract just the function name
            if let Some(digit_pos) = name_part.find(char::is_numeric)
                && let Some(after_digits) = name_part.get(digit_pos..) {
                    // Extract length-prefixed name
                    if let Some(len_str) = after_digits.split(|c: char| !c.is_numeric()).next()
                        && let Ok(len) = len_str.parse::<usize>() {
                            let start = digit_pos + len_str.len();
                            if let Some(func_name) = after_digits.get(start..start + len) {
                                return func_name.to_string();
                            }
                        }
                }
        }

    // If demangling fails, return original
    mangled.to_string()
}

/// Parse MSVC-style unresolved symbol error
fn parse_msvc_unresolved_symbol(line: &str) -> Option<BuildError> {
    // Example: "cudnn.o : error LNK2019: unresolved external symbol cudaFreeAsync referenced in function"
    // May have PowerShell redirection prefix: ">           cudnn.o : error LNK2019: ..."

    // Strip PowerShell output redirection prefix and whitespace
    let clean_line = line.trim_start_matches('>').trim();

    if let Some(symbol_start) = clean_line.find("unresolved external symbol ") {
        let after_prefix = &clean_line[symbol_start + 27..];

        // Extract symbol name (up to " referenced")
        let symbol = if let Some(end) = after_prefix.find(" referenced") {
            after_prefix[..end].trim()
        } else {
            after_prefix.trim()
        };

        // Extract object file if present (extract library name from object file name)
        let object_file = if let Some(colon_pos) = clean_line.find(" : error") {
            let obj_file = clean_line[..colon_pos].trim();
            // Extract library name from object file (e.g., "cudnn64_9.xyz.rcgu.o" -> "cudnn64_9")
            if let Some(dot_pos) = obj_file.find('.') {
                Some(obj_file[..dot_pos].to_string())
            } else {
                Some(obj_file.to_string())
            }
        } else {
            None
        };

        debug!(
            "Parsed unresolved symbol: {} from object: {:?}",
            symbol, object_file
        );

        return Some(BuildError::UnresolvedSymbol {
            symbol: symbol.to_string(),
            object_file,
        });
    }

    None
}

/// Parse ld-style undefined reference error
fn parse_ld_undefined_reference(line: &str) -> Option<BuildError> {
    // Example: "undefined reference to `cudaFreeAsync'"
    // May have PowerShell redirection prefix

    let clean_line = line.trim_start_matches('>').trim();

    if let Some(start) = clean_line.find("undefined reference to `") {
        let after_prefix = &clean_line[start + 24..];

        if let Some(end) = after_prefix.find('\'') {
            let symbol = after_prefix[..end].trim();

            // Try to demangle if it's a C++ symbol
            let clean_symbol = if symbol.starts_with("_Z") {
                demangle_symbol(symbol)
            } else {
                symbol.to_string()
            };

            debug!("Parsed undefined reference: {}", clean_symbol);

            return Some(BuildError::UnresolvedSymbol {
                symbol: clean_symbol,
                object_file: None,
            });
        }
    }

    None
}

/// Parse library not found error
fn parse_library_not_found(line: &str) -> Option<BuildError> {
    // Example: "cannot find -lcuda"
    // May have PowerShell redirection prefix

    let clean_line = line.trim_start_matches('>').trim();

    if let Some(start) = clean_line.find("cannot find -l") {
        let after_prefix = &clean_line[start + 14..];
        let lib_name = after_prefix.split_whitespace().next()?.trim();

        debug!("Parsed library not found: {}", lib_name);

        return Some(BuildError::LibraryNotFound {
            library_name: lib_name.to_string(),
        });
    }

    None
}

/// Parse missing header error
fn parse_missing_header(line: &str) -> Option<BuildError> {
    // Example: "fatal error: cuda.h: No such file or directory"
    // May have PowerShell redirection prefix

    let clean_line = line.trim_start_matches('>').trim();

    if let Some(start) = clean_line.find("fatal error:") {
        let after_prefix = &clean_line[start + 12..].trim();

        if let Some(colon_pos) = after_prefix.find(':') {
            let header = after_prefix[..colon_pos].trim();

            debug!("Parsed missing header: {}", header);

            return Some(BuildError::MissingHeader {
                header_name: header.to_string(),
            });
        }
    }

    None
}

/// Analyze errors to suggest missing libraries
pub fn suggest_missing_libraries(errors: &[BuildError]) -> Vec<String> {
    let mut symbol_counts: HashMap<String, usize> = HashMap::new();
    let mut suggested_libs: HashSet<String> = HashSet::new();

    // Count symbols by prefix to identify library patterns
    for error in errors {
        if let BuildError::UnresolvedSymbol { symbol, .. } = error {
            // Extract prefix (e.g., "cuda" from "cudaFreeAsync")
            let prefix = extract_prefix(symbol);
            *symbol_counts.entry(prefix.clone()).or_insert(0) += 1;

            // Direct library name suggestions
            if let Some(lib) = symbol_to_library(symbol) {
                suggested_libs.insert(lib);
            }
        } else if let BuildError::LibraryNotFound { library_name } = error {
            suggested_libs.insert(library_name.clone());
        }
    }

    // Suggest libraries based on high symbol counts
    for (prefix, count) in symbol_counts {
        if count >= 3 {
            // Multiple symbols with same prefix suggests missing library
            suggested_libs.insert(prefix);
        }
    }

    let mut suggestions: Vec<String> = suggested_libs.into_iter().collect();
    suggestions.sort();

    if !suggestions.is_empty() {
        info!("Suggested missing libraries: {:?}", suggestions);
    }

    suggestions
}

/// Extract prefix from function name
fn extract_prefix(symbol: &str) -> String {
    // Common patterns: cudaFreeAsync -> "cuda", cuBLAS_create -> "cublas"

    let lower = symbol.to_lowercase();

    // Check for common prefixes
    let known_prefixes = [
        "cuda", "cublas", "cufft", "curand", "cusparse", "cusolver", "cudnn", "nccl", "nvjpeg",
        "npp", "thrust", "gl", "vk", "d3d", "opencl",
    ];

    for prefix in &known_prefixes {
        if lower.starts_with(prefix) {
            return prefix.to_string();
        }
    }

    // Try to extract prefix before first uppercase after lowercase
    let chars: Vec<char> = symbol.chars().collect();
    for i in 1..chars.len() {
        if chars[i].is_uppercase() && chars[i - 1].is_lowercase() {
            return symbol[..i].to_lowercase();
        }
    }

    // Fallback: take first word-like segment
    if let Some(pos) = symbol.find(|c: char| c.is_uppercase() || c == '_')
        && pos > 0 {
            return symbol[..pos].to_lowercase();
        }

    symbol.to_lowercase()
}

/// Map a symbol name to likely library name
fn symbol_to_library(symbol: &str) -> Option<String> {
    let lower = symbol.to_lowercase();

    // CUDA ecosystem
    if lower.starts_with("cuda") && !lower.starts_with("cudnn") {
        return Some("cudart".to_string()); // CUDA runtime
    }
    if lower.starts_with("cudnn") {
        return Some("cudnn".to_string());
    }
    if lower.starts_with("cublas") {
        return Some("cublas".to_string());
    }
    if lower.starts_with("cufft") {
        return Some("cufft".to_string());
    }
    if lower.starts_with("curand") {
        return Some("curand".to_string());
    }
    if lower.starts_with("cusparse") {
        return Some("cusparse".to_string());
    }
    if lower.starts_with("cusolver") {
        return Some("cusolver".to_string());
    }

    // Graphics APIs
    if lower.starts_with("gl") {
        return Some("opengl".to_string());
    }
    if lower.starts_with("vk") {
        return Some("vulkan".to_string());
    }

    // Other common libraries
    if lower.starts_with("nccl") {
        return Some("nccl".to_string());
    }
    if lower.starts_with("nvjpeg") {
        return Some("nvjpeg".to_string());
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_msvc_unresolved_symbol() {
        let line = "cudnn.o : error LNK2019: unresolved external symbol cudaFreeAsync referenced in function";
        let errors = parse_build_errors(line);

        assert_eq!(errors.len(), 1);
        match &errors[0] {
            BuildError::UnresolvedSymbol { symbol, .. } => {
                assert_eq!(symbol, "cudaFreeAsync");
            }
            _ => panic!("Expected UnresolvedSymbol"),
        }
    }

    #[test]
    fn test_parse_ld_undefined_reference() {
        let line = "undefined reference to `cudaFreeAsync'";
        let errors = parse_build_errors(line);

        assert_eq!(errors.len(), 1);
        match &errors[0] {
            BuildError::UnresolvedSymbol { symbol, .. } => {
                assert_eq!(symbol, "cudaFreeAsync");
            }
            _ => panic!("Expected UnresolvedSymbol"),
        }
    }

    #[test]
    fn test_suggest_missing_libraries() {
        let errors = vec![
            BuildError::UnresolvedSymbol {
                symbol: "cudaFreeAsync".to_string(),
                object_file: None,
            },
            BuildError::UnresolvedSymbol {
                symbol: "cudaMalloc".to_string(),
                object_file: None,
            },
            BuildError::UnresolvedSymbol {
                symbol: "cudaMemcpy".to_string(),
                object_file: None,
            },
        ];

        let suggestions = suggest_missing_libraries(&errors);
        assert!(suggestions.contains(&"cudart".to_string()));
    }

    #[test]
    fn test_extract_prefix() {
        assert_eq!(extract_prefix("cudaFreeAsync"), "cuda");
        assert_eq!(extract_prefix("cuBLAS_create"), "cublas");
        assert_eq!(extract_prefix("glCreateShader"), "gl");
    }
}
