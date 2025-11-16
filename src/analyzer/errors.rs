use crate::ffi::{FfiEnum, FfiFunction, FfiInfo};
use tracing::{debug, info};

/// Detected error handling patterns
#[derive(Debug, Clone)]
pub struct ErrorPatterns {
    pub error_enums: Vec<ErrorEnum>,
    pub status_code_functions: Vec<String>,
    pub error_strategy: ErrorStrategy,
}

#[derive(Debug, Clone)]
pub struct ErrorEnum {
    pub name: String,
    pub success_variant: Option<String>,
    pub error_variants: Vec<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ErrorStrategy {
    /// Functions return status codes (int/enum)
    StatusCodes,
    /// Functions return bool (success/failure)
    Boolean,
    /// Functions set errno
    Errno,
    /// Functions return null on error
    NullPointer,
    /// Mixed strategies
    Mixed,
    /// Unknown/no clear pattern
    Unknown,
}

impl Default for ErrorPatterns {
    fn default() -> Self {
        Self::new()
    }
}

impl ErrorPatterns {
    pub fn new() -> Self {
        Self {
            error_enums: Vec::new(),
            status_code_functions: Vec::new(),
            error_strategy: ErrorStrategy::Unknown,
        }
    }
}

/// Detect error handling patterns in the FFI
pub fn detect_error_patterns(ffi_info: &FfiInfo) -> ErrorPatterns {
    info!("Detecting error handling patterns");

    let mut patterns = ErrorPatterns::new();

    // Step 1: Identify error/status enums
    patterns.error_enums = identify_error_enums(&ffi_info.enums);

    // Step 2: Analyze function return types
    let (status_functions, strategy) = analyze_return_types(&ffi_info.functions);
    patterns.status_code_functions = status_functions;
    patterns.error_strategy = strategy;

    info!("Detected error strategy: {:?}", patterns.error_strategy);
    info!("Found {} error enums", patterns.error_enums.len());

    patterns
}

fn identify_error_enums(enums: &[FfiEnum]) -> Vec<ErrorEnum> {
    let mut error_enums = Vec::new();

    // Common patterns for error enums
    let error_patterns = ["error", "status", "result", "code", "err"];
    let success_patterns = ["ok", "success", "none", "good", "valid"];

    for e in enums {
        let lower_name = e.name.to_lowercase();

        // Check if this is likely an error enum
        let is_error_enum = error_patterns.iter().any(|p| lower_name.contains(p));

        if is_error_enum {
            debug!("Found potential error enum: {}", e.name);

            // Find success variant (usually 0 or explicitly named)
            let success_variant = e
                .variants
                .iter()
                .find(|v| {
                    let v_lower = v.name.to_lowercase();
                    success_patterns.iter().any(|p| v_lower.contains(p)) || v.value == Some(0)
                })
                .map(|v| v.name.clone());

            // Other variants are errors
            let error_variants: Vec<String> = e
                .variants
                .iter()
                .filter(|v| Some(v.name.clone()) != success_variant)
                .map(|v| v.name.clone())
                .collect();

            error_enums.push(ErrorEnum {
                name: e.name.clone(),
                success_variant,
                error_variants,
            });
        }
    }

    error_enums
}

fn analyze_return_types(functions: &[FfiFunction]) -> (Vec<String>, ErrorStrategy) {
    let mut status_functions = Vec::new();
    let mut return_type_counts: std::collections::HashMap<String, usize> =
        std::collections::HashMap::new();

    for func in functions {
        let ret_type = normalize_return_type(&func.return_type);
        *return_type_counts.entry(ret_type.clone()).or_insert(0) += 1;

        // Check if function likely returns status code
        if is_status_return_type(&func.return_type) {
            status_functions.push(func.name.clone());
        }
    }

    // Determine overall strategy
    let strategy = determine_strategy(&return_type_counts);

    (status_functions, strategy)
}

fn normalize_return_type(ret_type: &str) -> String {
    ret_type.trim().to_lowercase().replace(" ", "")
}

fn is_status_return_type(ret_type: &str) -> bool {
    let normalized = normalize_return_type(ret_type);

    // Common status code return types
    normalized.contains("status")
        || normalized.contains("error")
        || normalized.contains("result")
        || normalized == "i32"
        || normalized == "c_int"
        || normalized == "bool"
}

fn determine_strategy(
    return_type_counts: &std::collections::HashMap<String, usize>,
) -> ErrorStrategy {
    let mut strategies = Vec::new();

    // Count functions by return type category
    let mut int_count = 0;
    let mut bool_count = 0;
    let mut pointer_count = 0;
    let mut void_count = 0;

    for (ret_type, count) in return_type_counts {
        if ret_type.contains("i32") || ret_type.contains("c_int") || ret_type.contains("status") {
            int_count += count;
        } else if ret_type.contains("bool") {
            bool_count += count;
        } else if ret_type.contains("*") {
            pointer_count += count;
        } else if ret_type == "()" || ret_type == "void" {
            void_count += count;
        }
    }

    debug!(
        "Return type distribution: int={}, bool={}, pointer={}, void={}",
        int_count, bool_count, pointer_count, void_count
    );

    // Determine primary strategy
    let total = int_count + bool_count + pointer_count + void_count;
    if total == 0 {
        return ErrorStrategy::Unknown;
    }

    if int_count as f32 / total as f32 > 0.5 {
        strategies.push(ErrorStrategy::StatusCodes);
    }
    if bool_count as f32 / total as f32 > 0.3 {
        strategies.push(ErrorStrategy::Boolean);
    }
    if pointer_count as f32 / total as f32 > 0.3 {
        strategies.push(ErrorStrategy::NullPointer);
    }

    match strategies.len() {
        0 => ErrorStrategy::Unknown,
        1 => strategies[0].clone(),
        _ => ErrorStrategy::Mixed,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ffi::FfiEnumVariant;

    #[test]
    fn test_identify_error_enums() {
        let enums = vec![FfiEnum {
            name: "MyStatus".to_string(),
            variants: vec![
                FfiEnumVariant {
                    name: "StatusOk".to_string(),
                    value: Some(0),
                },
                FfiEnumVariant {
                    name: "StatusError".to_string(),
                    value: Some(1),
                },
            ],
            docs: None,
        }];

        let error_enums = identify_error_enums(&enums);
        assert_eq!(error_enums.len(), 1);
        assert_eq!(error_enums[0].name, "MyStatus");
    }

    #[test]
    fn test_is_status_return_type() {
        assert!(is_status_return_type("i32"));
        assert!(is_status_return_type("MyStatus"));
        assert!(is_status_return_type("ErrorCode"));
        assert!(!is_status_return_type("*mut Foo"));
    }
}
