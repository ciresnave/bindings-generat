use crate::analyzer::error_semantics::ErrorSemantics;
use crate::analyzer::errors::ErrorEnum;
use crate::analyzer::{ErrorSeverity, SmartErrorAnalysis, SmartErrorCategory};
use crate::utils::doc_sanitizer::sanitize_doc;
use std::fmt::Write;
use tracing::{debug, info};

/// Generate a basic error enum when no FFI error enum is detected
pub fn generate_basic_error() -> String {
    info!("Generating basic error type (no error enum detected)");

    let mut code = String::new();

    writeln!(code, "/// Error type for this library").unwrap();
    writeln!(code, "#[derive(Debug, Clone, Copy, PartialEq, Eq)]").unwrap();
    writeln!(code, "pub enum Error {{").unwrap();
    writeln!(code, "    /// Null pointer returned").unwrap();
    writeln!(code, "    NullPointer,").unwrap();
    writeln!(code, "    /// Invalid parameter value").unwrap();
    writeln!(code, "    InvalidParameter,").unwrap();
    writeln!(code, "    /// Invalid string (contains null byte)").unwrap();
    writeln!(code, "    InvalidString,").unwrap();
    writeln!(code, "    /// FFI function returned an error status").unwrap();
    writeln!(code, "    FfiError(i32),").unwrap();
    writeln!(code, "    /// Unknown error").unwrap();
    writeln!(code, "    Unknown,").unwrap();
    writeln!(code, "}}").unwrap();
    writeln!(code).unwrap();

    // Generate From<i32> implementation
    writeln!(code, "impl From<i32> for Error {{").unwrap();
    writeln!(code, "    fn from(code: i32) -> Self {{").unwrap();
    writeln!(code, "        if code == 0 {{").unwrap();
    writeln!(
        code,
        "            // Success code should not become an error"
    )
    .unwrap();
    writeln!(code, "            Error::Unknown").unwrap();
    writeln!(code, "        }} else {{").unwrap();
    writeln!(code, "            Error::FfiError(code)").unwrap();
    writeln!(code, "        }}").unwrap();
    writeln!(code, "    }}").unwrap();
    writeln!(code, "}}").unwrap();
    writeln!(code).unwrap();

    // Generate Display implementation
    writeln!(code, "impl std::fmt::Display for Error {{").unwrap();
    writeln!(
        code,
        "    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {{"
    )
    .unwrap();
    writeln!(code, "        match self {{").unwrap();
    writeln!(
        code,
        "            Error::NullPointer => write!(f, \"Null pointer returned\"),"
    )
    .unwrap();
    writeln!(code, "            Error::InvalidParameter => write!(f, \"Invalid parameter value\"),").unwrap();
    writeln!(
        code,
        "            Error::InvalidString => write!(f, \"Invalid string: contains null byte\"),"
    )
    .unwrap();
    writeln!(
        code,
        "            Error::FfiError(code) => write!(f, \"FFI error: {{}}\", code),"
    )
    .unwrap();
    writeln!(
        code,
        "            Error::Unknown => write!(f, \"Unknown error\"),"
    )
    .unwrap();
    writeln!(code, "        }}").unwrap();
    writeln!(code, "    }}").unwrap();
    writeln!(code, "}}").unwrap();
    writeln!(code).unwrap();

    // Generate std::error::Error implementation
    writeln!(code, "impl std::error::Error for Error {{}}").unwrap();
    writeln!(code).unwrap();

    // Generate helper methods for error classification
    writeln!(code, "impl Error {{").unwrap();
    writeln!(
        code,
        "    /// Returns true if this error might be retryable"
    )
    .unwrap();
    writeln!(code, "    pub fn is_retryable(&self) -> bool {{").unwrap();
    writeln!(code, "        // Basic errors are generally not retryable").unwrap();
    writeln!(code, "        false").unwrap();
    writeln!(code, "    }}").unwrap();
    writeln!(code).unwrap();
    writeln!(
        code,
        "    /// Returns true if this error indicates a fatal condition"
    )
    .unwrap();
    writeln!(code, "    pub fn is_fatal(&self) -> bool {{").unwrap();
    writeln!(code, "        match self {{").unwrap();
    writeln!(code, "            Error::NullPointer => true,").unwrap();
    writeln!(code, "            Error::InvalidParameter => false,").unwrap();
    writeln!(code, "            Error::InvalidString => false,").unwrap();
    writeln!(code, "            Error::Unknown => true,").unwrap();
    writeln!(
        code,
        "            Error::FfiError(_) => false, // Unknown without enum details"
    )
    .unwrap();
    writeln!(code, "        }}").unwrap();
    writeln!(code, "    }}").unwrap();
    writeln!(code, "}}").unwrap();
    writeln!(code).unwrap();

    code
}

/// Generate Rust error enum from FFI error enum
pub fn generate_error_enum(
    error_enum: &ErrorEnum,
    enhancements: Option<&crate::llm::CodeEnhancements>,
    analysis: &crate::analyzer::AnalysisResult,
) -> String {
    // For now, pass None for smart_errors since we don't have it in the signature yet
    // TODO: Update all call sites to pass smart error analysis
    generate_error_enum_with_smart_analysis(error_enum, enhancements, analysis, None)
}

/// Generate error enum with optional smart error analysis
pub fn generate_error_enum_with_smart_analysis(
    error_enum: &ErrorEnum,
    enhancements: Option<&crate::llm::CodeEnhancements>,
    analysis: &crate::analyzer::AnalysisResult,
    smart_errors: Option<&SmartErrorAnalysis>,
) -> String {
    debug!("Generating error enum for {}", error_enum.name);

    let mut code = String::new();

    // Generate the error enum
    writeln!(code, "/// Error type for this library").unwrap();
    writeln!(code, "#[derive(Debug, Clone, Copy, PartialEq, Eq)]").unwrap();
    writeln!(code, "pub enum Error {{").unwrap();

    // Add null pointer variant
    writeln!(code, "    /// Null pointer returned").unwrap();
    writeln!(code, "    NullPointer,").unwrap();

    // Add invalid string variant
    writeln!(code, "    /// Invalid string (contains null byte)").unwrap();
    writeln!(code, "    InvalidString,").unwrap();

    // Add each error variant
    for variant in &error_enum.error_variants {
        let rust_variant = to_rust_variant_name(variant);

        // Priority: 1) Header docs, 2) LLM enhancement, 3) Auto-generated message
        let doc_comment = error_enum
            .variant_docs
            .get(variant)
            .map(|s| s.as_str())
            .or_else(|| {
                enhancements
                    .and_then(|e| e.get_error_message(variant))
                    .map(|s| s.as_str())
            })
            .unwrap_or(variant);

        let sanitized = sanitize_doc(doc_comment);
        for line in sanitized.lines() {
            writeln!(code, "    /// {}", line).unwrap();
        }
        writeln!(code, "    {},", rust_variant).unwrap();
    }

    // Add unknown variant for unrecognized status codes
    writeln!(code, "    /// Unknown error code").unwrap();
    writeln!(code, "    Unknown(i32),").unwrap();

    writeln!(code, "}}").unwrap();
    writeln!(code).unwrap();

    // Generate From implementation for the FFI enum
    writeln!(code, "impl From<ffi::{}> for Error {{", error_enum.name).unwrap();
    writeln!(
        code,
        "    fn from(status: ffi::{}) -> Self {{",
        error_enum.name
    )
    .unwrap();
    writeln!(code, "        match status {{").unwrap();

    // Success case should not be converted to Error (this is for error path)
    for variant in &error_enum.error_variants {
        let rust_variant = to_rust_variant_name(variant);
        writeln!(
            code,
            "            ffi::{}::{} => Error::{},",
            error_enum.name, variant, rust_variant
        )
        .unwrap();
    }

    writeln!(code, "            _ => Error::Unknown(status as i32),").unwrap();
    writeln!(code, "        }}").unwrap();
    writeln!(code, "    }}").unwrap();
    writeln!(code, "}}").unwrap();
    writeln!(code).unwrap();

    // Generate Display implementation with human-readable messages
    writeln!(code, "impl std::fmt::Display for Error {{").unwrap();
    writeln!(
        code,
        "    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {{"
    )
    .unwrap();
    writeln!(code, "        match self {{").unwrap();
    writeln!(
        code,
        "            Error::NullPointer => write!(f, \"Null pointer returned\"),"
    )
    .unwrap();
    writeln!(
        code,
        "            Error::InvalidString => write!(f, \"Invalid string: contains null byte\"),"
    )
    .unwrap();

    for variant in &error_enum.error_variants {
        let rust_variant = to_rust_variant_name(variant);

        // Get human-readable message: header docs > enhanced message > auto-generated
        let message = error_enum
            .variant_docs
            .get(variant)
            .map(|doc| {
                // Extract first line/sentence from docs for display message
                let first_line = doc.lines().next().unwrap_or(doc);
                // Remove common doc prefixes and clean up
                first_line
                    .trim()
                    .trim_start_matches("/**")
                    .trim_start_matches("/*")
                    .trim_start_matches('*')
                    .trim_start_matches("///")
                    .trim_start_matches("//")
                    .trim()
                    .to_string()
            })
            .unwrap_or_else(|| variant_to_message(variant));

        writeln!(
            code,
            "            Error::{} => write!(f, \"{}\"),",
            rust_variant,
            message.escape_default()
        )
        .unwrap();
    }

    writeln!(
        code,
        "            Error::Unknown(code) => write!(f, \"Unknown error code: {{}}\", code),"
    )
    .unwrap();
    writeln!(code, "        }}").unwrap();
    writeln!(code, "    }}").unwrap();
    writeln!(code, "}}").unwrap();
    writeln!(code).unwrap();

    // Generate std::error::Error implementation
    writeln!(code, "impl std::error::Error for Error {{}}").unwrap();
    writeln!(code).unwrap();

    // Generate helper methods for error classification
    writeln!(code, "impl Error {{").unwrap();

    // Generate is_retryable method
    if !analysis.function_contexts.is_empty() {
        // Use semantic error information from documentation analysis
        writeln!(
            code,
            "    /// Returns true if this error might be retryable"
        )
        .unwrap();
        writeln!(code, "    ///").unwrap();
        writeln!(
            code,
            "    /// Based on semantic analysis of error documentation and patterns."
        )
        .unwrap();
        writeln!(code, "    ///").unwrap();

        // Add examples of retryable errors from enrichment
        let mut retryable_examples: Vec<String> = Vec::new();
        for func_ctx in analysis.function_contexts.values() {
            if let Some(err_sem) = &func_ctx.error_semantics {
                for (variant, info) in &err_sem.errors {
                    if info.is_retryable && !retryable_examples.contains(variant) {
                        retryable_examples.push(variant.clone());
                        if retryable_examples.len() >= 3 {
                            break;
                        }
                    }
                }
            }
            if retryable_examples.len() >= 3 {
                break;
            }
        }

        if !retryable_examples.is_empty() {
            writeln!(
                code,
                "    /// Examples of retryable errors: {}",
                retryable_examples.join(", ")
            )
            .unwrap();
            writeln!(code, "    ///").unwrap();
        }
    } else {
        writeln!(
            code,
            "    /// Returns true if this error might be retryable"
        )
        .unwrap();
        writeln!(code, "    ///").unwrap();
        writeln!(
            code,
            "    /// This is a heuristic based on common error naming patterns."
        )
        .unwrap();
        writeln!(
            code,
            "    /// Errors related to resources, timeouts, or temporary conditions"
        )
        .unwrap();
        writeln!(code, "    /// are typically retryable.").unwrap();
    }

    writeln!(code, "    pub fn is_retryable(&self) -> bool {{").unwrap();
    writeln!(code, "        match self {{").unwrap();
    writeln!(code, "            Error::NullPointer => false,").unwrap();
    writeln!(code, "            Error::Unknown(_) => false,").unwrap();

    // Add retryability hints for each variant
    for variant in &error_enum.error_variants {
        let rust_variant = to_rust_variant_name(variant);

        // Try to get semantic info from enrichment
        let is_retryable = {
            // Check all functions for error semantics about this variant
            let mut found = false;
            let mut retryable = false;

            for func_ctx in analysis.function_contexts.values() {
                if let Some(err_sem) = &func_ctx.error_semantics
                    && let Some(err_info) = err_sem.errors.get(variant)
                {
                    found = true;
                    retryable = err_info.is_retryable;
                    break;
                }
            }

            if found {
                retryable
            } else {
                // Fallback to heuristics
                let variant_lower = variant.to_lowercase();
                variant_lower.contains("busy")
                    || variant_lower.contains("timeout")
                    || variant_lower.contains("again")
                    || variant_lower.contains("retry")
                    || variant_lower.contains("unavailable")
                    || variant_lower.contains("resource")
                    || variant_lower.contains("full")
                    || variant_lower.contains("limit")
                    || variant_lower.contains("temp")
                    || variant_lower.contains("transient")
            }
        };

        writeln!(
            code,
            "            Error::{} => {},",
            rust_variant, is_retryable
        )
        .unwrap();
    }

    writeln!(code, "        }}").unwrap();
    writeln!(code, "    }}").unwrap();
    writeln!(code).unwrap();

    // Generate is_fatal method
    if !analysis.function_contexts.is_empty() {
        writeln!(
            code,
            "    /// Returns true if this error indicates a fatal condition"
        )
        .unwrap();
        writeln!(code, "    ///").unwrap();
        writeln!(
            code,
            "    /// Based on semantic analysis of error documentation and patterns."
        )
        .unwrap();
        writeln!(code, "    ///").unwrap();

        // Add examples of fatal errors from enrichment
        let mut fatal_examples: Vec<String> = Vec::new();
        for func_ctx in analysis.function_contexts.values() {
            if let Some(err_sem) = &func_ctx.error_semantics {
                for (variant, info) in &err_sem.errors {
                    if !info.is_retryable
                        && info.description.to_lowercase().contains("fatal")
                        && !fatal_examples.contains(variant)
                    {
                        fatal_examples.push(variant.clone());
                        if fatal_examples.len() >= 3 {
                            break;
                        }
                    }
                }
            }
            if fatal_examples.len() >= 3 {
                break;
            }
        }

        if !fatal_examples.is_empty() {
            writeln!(
                code,
                "    /// Examples of fatal errors: {}",
                fatal_examples.join(", ")
            )
            .unwrap();
            writeln!(code, "    ///").unwrap();
        }
    } else {
        writeln!(
            code,
            "    /// Returns true if this error indicates a fatal condition"
        )
        .unwrap();
        writeln!(code, "    ///").unwrap();
        writeln!(
            code,
            "    /// Fatal errors typically indicate internal errors, memory exhaustion,"
        )
        .unwrap();
        writeln!(code, "    /// or unrecoverable system failures.").unwrap();
    }

    writeln!(code, "    pub fn is_fatal(&self) -> bool {{").unwrap();
    writeln!(code, "        match self {{").unwrap();
    writeln!(code, "            Error::NullPointer => true,").unwrap();
    writeln!(code, "            Error::Unknown(_) => true,").unwrap();

    // Add fatality hints for each variant
    for variant in &error_enum.error_variants {
        let rust_variant = to_rust_variant_name(variant);

        // Try to get semantic info from enrichment
        let is_fatal = {
            // Check all functions for error semantics about this variant
            let mut found = false;
            let mut fatal = false;

            for func_ctx in analysis.function_contexts.values() {
                if let Some(err_sem) = &func_ctx.error_semantics
                    && let Some(err_info) = err_sem.errors.get(variant)
                {
                    found = true;
                    fatal = err_info.is_fatal;
                    break;
                }
            }

            if found {
                fatal
            } else {
                // Fallback to heuristics
                let variant_lower = variant.to_lowercase();
                variant_lower.contains("fatal")
                    || variant_lower.contains("corrupt")
                    || variant_lower.contains("panic")
                    || variant_lower.contains("abort")
                    || variant_lower.contains("internal")
                    || variant_lower.contains("unrecoverable")
                    || variant_lower.contains("out_of_memory")
                    || variant_lower.contains("oom")
            }
        };

        writeln!(code, "            Error::{} => {},", rust_variant, is_fatal).unwrap();
    }

    writeln!(code, "        }}").unwrap();
    writeln!(code, "    }}").unwrap();

    // Add smart error analysis methods if available
    if let Some(smart_errors) = smart_errors {
        generate_smart_error_methods(&mut code, error_enum, smart_errors);
    }

    writeln!(code, "}}").unwrap();
    writeln!(code).unwrap();

    info!("Generated error enum from {}", error_enum.name);

    code
}

/// Convert C error variant to Rust variant name
fn to_rust_variant_name(c_variant: &str) -> String {
    let mut variant = c_variant
        .replace("ERROR_", "")
        .replace("ERR_", "")
        .replace("STATUS_", "");

    // Convert SCREAMING_SNAKE_CASE to PascalCase
    variant = variant
        .split('_')
        .map(|word| {
            if word.is_empty() {
                String::new()
            } else {
                let lower = word.to_lowercase();
                lower[0..1].to_uppercase() + &lower[1..]
            }
        })
        .collect::<Vec<_>>()
        .join("");

    // Ensure it starts with uppercase
    if !variant.is_empty() && !variant.chars().next().unwrap().is_uppercase() {
        variant = variant[0..1].to_uppercase() + &variant[1..];
    }

    variant
}

/// Convert variant name to human-readable message with context
fn variant_to_message(variant: &str) -> String {
    // First, clean up the variant name
    let cleaned = variant
        .replace("ERROR_", "")
        .replace("ERR_", "")
        .replace("STATUS_", "")
        .replace("CUDNN_STATUS_", "")
        .replace("CUDA_ERROR_", "");

    let lower = cleaned.replace("_", " ").to_lowercase();

    // Add context-aware enhancements for common error patterns
    let enhanced = match lower.as_str() {
        // Exact matches for common errors
        "not initialized" => {
            "Library or resource not initialized - call initialization function first"
        }
        "alloc failed" | "out of memory" => {
            "Memory allocation failed - insufficient memory available"
        }
        "bad param" | "invalid value" => "Invalid parameter value provided",
        "internal error" => "Internal library error - this may indicate a bug",
        "not supported" => "Operation not supported in this configuration or version",
        "license error" => "License validation failed - check license configuration",
        "runtime prerequisite missing" => "Required runtime component not found",
        "runtime in progress" => "Another operation is already in progress",
        "runtime fp overflow" => "Floating-point overflow occurred during computation",
        "not implemented" => "Feature not implemented in this version",

        // Memory errors
        s if s.contains("out of memory") || s.contains("alloc") => {
            "Memory allocation failed - insufficient memory available"
        }
        s if s.contains("bad alloc") => "Memory allocation failed - invalid allocation request",

        // Parameter errors
        s if s.contains("invalid") && s.contains("arg") => {
            "Invalid argument - one or more parameters are out of valid range"
        }
        s if s.contains("null") && s.contains("pointer") => {
            "Null pointer error - a required pointer parameter was null"
        }
        s if s.contains("bad param") => "Bad parameter - check parameter values and types",
        s if s.contains("invalid") && (s.contains("handle") || s.contains("descriptor")) => {
            "Invalid handle or descriptor - may be uninitialized or destroyed"
        }

        // Initialization errors
        s if s.contains("not initialized") => {
            "Not initialized - resource must be initialized before use"
        }
        s if s.contains("already initialized") => {
            "Already initialized - resource cannot be initialized twice"
        }

        // Resource errors
        s if s.contains("not found") => "Resource not found - the requested item does not exist",
        s if s.contains("already exists") => "Resource already exists - cannot create duplicate",
        s if s.contains("in use") || s.contains("busy") => {
            "Resource busy - currently in use by another operation"
        }

        // Compatibility errors
        s if s.contains("arch mismatch") => {
            "Architecture mismatch - binary compiled for different architecture"
        }
        s if s.contains("not supported") => "Operation not supported in this configuration",
        s if s.contains("version mismatch") => "Version mismatch - incompatible library versions",

        // Operation errors
        s if s.contains("not implemented") => "Feature not implemented in this version",
        s if s.contains("timeout") => "Operation timed out - took longer than allowed",
        s if s.contains("permission") || s.contains("access denied") => {
            "Access denied - insufficient permissions"
        }
        s if s.contains("mapping error") => "Memory mapping error - failed to map device memory",

        // State errors
        s if s.contains("invalid state") => "Invalid state - resource not ready for this operation",
        s if s.contains("failed") && !s.contains("alloc") => {
            "Operation failed - check parameters and state"
        }

        // Execution errors
        s if s.contains("execution failed") => "Execution failed - runtime error during operation",
        s if s.contains("launch failed") => "Launch failed - could not start operation",

        // Generic fallback with proper capitalization
        _ => return capitalize_first(&lower),
    };

    enhanced.to_string()
}

/// Capitalize the first character of a string
fn capitalize_first(s: &str) -> String {
    let mut chars = s.chars();
    match chars.next() {
        None => String::new(),
        Some(first) => first.to_uppercase().chain(chars).collect(),
    }
}

/// Generate enhanced error enum using error semantics analysis
///
/// This generates separate error types for fatal vs recoverable errors,
/// includes retry hints, and provides better error context.
pub fn generate_enhanced_error_enum(
    error_enum: &ErrorEnum,
    error_semantics: Option<&ErrorSemantics>,
    enhancements: Option<&crate::llm::CodeEnhancements>,
) -> String {
    debug!("Generating enhanced error enum for {}", error_enum.name);

    let mut code = String::new();

    // Separate fatal and recoverable errors if we have semantics
    let (fatal_variants, recoverable_variants, other_variants): (
        Vec<&String>,
        Vec<&String>,
        Vec<&String>,
    ) = if let Some(semantics) = error_semantics {
        let mut fatal = Vec::new();
        let mut recoverable = Vec::new();
        let mut other = Vec::new();

        for variant in &error_enum.error_variants {
            if let Some(error_info) = semantics.errors.get(variant) {
                if error_info.is_fatal {
                    fatal.push(variant);
                } else if error_info.is_retryable {
                    recoverable.push(variant);
                } else {
                    other.push(variant);
                }
            } else {
                other.push(variant);
            }
        }

        (fatal, recoverable, other)
    } else {
        (
            Vec::new(),
            Vec::new(),
            error_enum.error_variants.iter().collect(),
        )
    };

    // Generate main error enum
    writeln!(code, "/// Error type for this library").unwrap();
    writeln!(code, "///").unwrap();

    if error_semantics.is_some() && (!fatal_variants.is_empty() || !recoverable_variants.is_empty())
    {
        writeln!(code, "/// This enum is organized by error severity:").unwrap();
        if !fatal_variants.is_empty() {
            writeln!(
                code,
                "/// - **Fatal errors**: Cannot recover, operation must abort"
            )
            .unwrap();
        }
        if !recoverable_variants.is_empty() {
            writeln!(
                code,
                "/// - **Recoverable errors**: May succeed if retried or conditions change"
            )
            .unwrap();
        }
        if !other_variants.is_empty() {
            writeln!(
                code,
                "/// - **Other errors**: Require investigation or different approach"
            )
            .unwrap();
        }
    }

    writeln!(code, "#[derive(Debug, Clone, PartialEq, Eq)]").unwrap();
    writeln!(code, "pub enum Error {{").unwrap();

    // Null pointer variant
    writeln!(code, "    /// Null pointer returned").unwrap();
    writeln!(code, "    NullPointer,").unwrap();
    writeln!(code).unwrap();

    // Fatal errors section
    if !fatal_variants.is_empty() {
        writeln!(code, "    // === Fatal Errors (Cannot Recover) ===").unwrap();
        writeln!(code).unwrap();
        for variant in &fatal_variants {
            generate_error_variant(
                &mut code,
                variant,
                error_semantics,
                error_enum,
                enhancements,
            );
        }
        writeln!(code).unwrap();
    }

    // Recoverable errors section
    if !recoverable_variants.is_empty() {
        writeln!(
            code,
            "    // === Recoverable Errors (Retry May Succeed) ==="
        )
        .unwrap();
        writeln!(code).unwrap();
        for variant in &recoverable_variants {
            generate_error_variant(
                &mut code,
                variant,
                error_semantics,
                error_enum,
                enhancements,
            );
        }
        writeln!(code).unwrap();
    }

    // Other errors section
    if !other_variants.is_empty() {
        if !fatal_variants.is_empty() || !recoverable_variants.is_empty() {
            writeln!(code, "    // === Other Errors ===").unwrap();
            writeln!(code).unwrap();
        }
        for variant in &other_variants {
            generate_error_variant(
                &mut code,
                variant,
                error_semantics,
                error_enum,
                enhancements,
            );
        }
    }

    // Unknown variant
    writeln!(code, "    /// Unknown error code").unwrap();
    writeln!(code, "    Unknown(i32),").unwrap();

    writeln!(code, "}}").unwrap();
    writeln!(code).unwrap();

    // Generate implementation with retry helpers
    writeln!(code, "impl Error {{").unwrap();

    // is_fatal method
    writeln!(
        code,
        "    /// Returns true if this error is fatal and cannot be recovered from"
    )
    .unwrap();
    writeln!(code, "    pub fn is_fatal(&self) -> bool {{").unwrap();
    writeln!(code, "        match self {{").unwrap();
    for variant in &fatal_variants {
        let rust_variant = to_rust_variant_name(variant);
        writeln!(code, "            Error::{} => true,", rust_variant).unwrap();
    }
    writeln!(code, "            _ => false,").unwrap();
    writeln!(code, "        }}").unwrap();
    writeln!(code, "    }}").unwrap();
    writeln!(code).unwrap();

    // is_retryable method
    writeln!(
        code,
        "    /// Returns true if the operation should be retried"
    )
    .unwrap();
    writeln!(code, "    pub fn is_retryable(&self) -> bool {{").unwrap();
    writeln!(code, "        match self {{").unwrap();
    for variant in &recoverable_variants {
        let rust_variant = to_rust_variant_name(variant);
        writeln!(code, "            Error::{} => true,", rust_variant).unwrap();
    }
    writeln!(code, "            _ => false,").unwrap();
    writeln!(code, "        }}").unwrap();
    writeln!(code, "    }}").unwrap();
    writeln!(code).unwrap();

    // category method
    if let Some(semantics) = error_semantics {
        writeln!(
            code,
            "    /// Returns the error category for semantic grouping"
        )
        .unwrap();
        writeln!(code, "    pub fn category(&self) -> ErrorCategory {{").unwrap();
        writeln!(code, "        match self {{").unwrap();
        writeln!(
            code,
            "            Error::NullPointer => ErrorCategory::InvalidInput,"
        )
        .unwrap();

        for variant in fatal_variants
            .iter()
            .chain(recoverable_variants.iter())
            .chain(other_variants.iter())
        {
            if let Some(info) = semantics.errors.get(*variant) {
                let rust_variant = to_rust_variant_name(variant);
                let category = match info.category {
                    crate::analyzer::error_semantics::ErrorCategory::InvalidInput => "InvalidInput",
                    crate::analyzer::error_semantics::ErrorCategory::NotFound => {
                        "ResourceUnavailable"
                    }
                    crate::analyzer::error_semantics::ErrorCategory::ResourceExhausted => "Memory",
                    crate::analyzer::error_semantics::ErrorCategory::PermissionDenied => {
                        "PermissionDenied"
                    }
                    crate::analyzer::error_semantics::ErrorCategory::WouldBlock => "Timeout",
                    crate::analyzer::error_semantics::ErrorCategory::Timeout => "Timeout",
                    crate::analyzer::error_semantics::ErrorCategory::IoError => "IO",
                    crate::analyzer::error_semantics::ErrorCategory::InternalError => "Internal",
                    crate::analyzer::error_semantics::ErrorCategory::NotSupported => "NotSupported",
                    crate::analyzer::error_semantics::ErrorCategory::AlreadyExists => "State",
                    crate::analyzer::error_semantics::ErrorCategory::Cancelled => "State",
                    crate::analyzer::error_semantics::ErrorCategory::Unknown => "Other",
                };
                writeln!(
                    code,
                    "            Error::{} => ErrorCategory::{},",
                    rust_variant, category
                )
                .unwrap();
            }
        }

        writeln!(
            code,
            "            Error::Unknown(_) => ErrorCategory::Unknown,"
        )
        .unwrap();
        writeln!(code, "        }}").unwrap();
        writeln!(code, "    }}").unwrap();
        writeln!(code).unwrap();
    }

    // recovery_hint method
    writeln!(
        code,
        "    /// Returns a hint for how to recover from this error"
    )
    .unwrap();
    writeln!(
        code,
        "    pub fn recovery_hint(&self) -> Option<&'static str> {{"
    )
    .unwrap();
    writeln!(code, "        match self {{").unwrap();

    if let Some(semantics) = error_semantics {
        for variant in fatal_variants
            .iter()
            .chain(recoverable_variants.iter())
            .chain(other_variants.iter())
        {
            if let Some(info) = semantics.errors.get(*variant)
                && let Some(recovery) = &info.recovery_action
            {
                let rust_variant = to_rust_variant_name(variant);
                writeln!(
                    code,
                    "            Error::{} => Some(\"{}\"),",
                    rust_variant,
                    recovery.escape_default()
                )
                .unwrap();
            }
        }
    }

    writeln!(code, "            _ => None,").unwrap();
    writeln!(code, "        }}").unwrap();
    writeln!(code, "    }}").unwrap();

    writeln!(code, "}}").unwrap();
    writeln!(code).unwrap();

    // Generate From implementation for the FFI enum
    writeln!(code, "impl From<ffi::{}> for Error {{", error_enum.name).unwrap();
    writeln!(
        code,
        "    fn from(status: ffi::{}) -> Self {{",
        error_enum.name
    )
    .unwrap();
    writeln!(code, "        match status {{").unwrap();

    for variant in &error_enum.error_variants {
        let rust_variant = to_rust_variant_name(variant);
        writeln!(
            code,
            "            ffi::{}::{} => Error::{},",
            error_enum.name, variant, rust_variant
        )
        .unwrap();
    }

    writeln!(code, "            _ => Error::Unknown(status as i32),").unwrap();
    writeln!(code, "        }}").unwrap();
    writeln!(code, "    }}").unwrap();
    writeln!(code, "}}").unwrap();
    writeln!(code).unwrap();

    // Generate Display implementation
    writeln!(code, "impl std::fmt::Display for Error {{").unwrap();
    writeln!(
        code,
        "    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {{"
    )
    .unwrap();
    writeln!(code, "        match self {{").unwrap();
    writeln!(
        code,
        "            Error::NullPointer => write!(f, \"Null pointer returned\"),"
    )
    .unwrap();
    writeln!(
        code,
        "            Error::InvalidString => write!(f, \"Invalid string: contains null byte\"),"
    )
    .unwrap();

    for variant in &error_enum.error_variants {
        let rust_variant = to_rust_variant_name(variant);

        // Get message from semantics, docs, or enhancement
        let message = if let Some(semantics) = error_semantics {
            semantics
                .errors
                .get(variant)
                .map(|info| info.description.clone())
        } else {
            None
        }
        .or_else(|| {
            error_enum.variant_docs.get(variant).map(|doc| {
                let first_line = doc.lines().next().unwrap_or(doc);
                first_line
                    .trim()
                    .trim_start_matches("/**")
                    .trim_start_matches("/*")
                    .trim_start_matches('*')
                    .trim_start_matches("///")
                    .trim_start_matches("//")
                    .trim()
                    .to_string()
            })
        })
        .unwrap_or_else(|| variant_to_message(variant));

        writeln!(
            code,
            "            Error::{} => write!(f, \"{}\"),",
            rust_variant,
            message.escape_default()
        )
        .unwrap();
    }

    writeln!(
        code,
        "            Error::Unknown(code) => write!(f, \"Unknown error code: {{}}\", code),"
    )
    .unwrap();
    writeln!(code, "        }}").unwrap();
    writeln!(code, "    }}").unwrap();
    writeln!(code, "}}").unwrap();
    writeln!(code).unwrap();

    // Generate std::error::Error implementation
    writeln!(code, "impl std::error::Error for Error {{}}").unwrap();
    writeln!(code).unwrap();

    info!("Generated enhanced error enum from {}", error_enum.name);

    code
}

/// Helper to generate a single error variant with documentation
fn generate_error_variant(
    code: &mut String,
    variant: &str,
    error_semantics: Option<&ErrorSemantics>,
    error_enum: &ErrorEnum,
    enhancements: Option<&crate::llm::CodeEnhancements>,
) {
    let rust_variant = to_rust_variant_name(variant);

    // Get error info from semantics if available
    let info = error_semantics.and_then(|s| s.errors.get(variant));

    // Get documentation from multiple sources
    let doc = if let Some(info) = info {
        error_enum
            .variant_docs
            .get(variant)
            .map(|s| s.as_str())
            .or_else(|| {
                enhancements
                    .and_then(|e| e.get_error_message(variant))
                    .map(|s| s.as_str())
            })
            .unwrap_or(&info.description)
    } else {
        error_enum
            .variant_docs
            .get(variant)
            .map(|s| s.as_str())
            .or_else(|| {
                enhancements
                    .and_then(|e| e.get_error_message(variant))
                    .map(|s| s.as_str())
            })
            .unwrap_or(variant)
    };

    let sanitized = sanitize_doc(doc);
    for line in sanitized.lines() {
        writeln!(code, "    /// {}", line).unwrap();
    }

    // Add semantic information if available
    if let Some(info) = info {
        if info.is_fatal {
            writeln!(code, "    ///").unwrap();
            writeln!(
                code,
                "    /// ⚠️ **Fatal**: Operation cannot continue after this error"
            )
            .unwrap();
        } else if info.is_retryable {
            writeln!(code, "    ///").unwrap();
            writeln!(
                code,
                "    /// 🔄 **Retryable**: Operation may succeed if retried"
            )
            .unwrap();
        }

        if let Some(recovery) = &info.recovery_action {
            writeln!(code, "    ///").unwrap();
            writeln!(code, "    /// **Recovery**: {}", recovery).unwrap();
        }

        if !info.required_context.is_empty() {
            writeln!(code, "    ///").unwrap();
            writeln!(
                code,
                "    /// **Context needed**: {}",
                info.required_context.join(", ")
            )
            .unwrap();
        }
    }

    writeln!(code, "    {},", rust_variant).unwrap();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_to_rust_variant_name() {
        assert_eq!(
            to_rust_variant_name("ERROR_INVALID_ARGUMENT"),
            "InvalidArgument"
        );
        assert_eq!(to_rust_variant_name("ERR_OUT_OF_MEMORY"), "OutOfMemory");
        assert_eq!(to_rust_variant_name("STATUS_FAILED"), "Failed");
    }

    #[test]
    fn test_variant_to_message() {
        assert_eq!(
            variant_to_message("ERROR_INVALID_ARGUMENT"),
            "Invalid argument - one or more parameters are out of valid range"
        );
        assert_eq!(
            variant_to_message("ERR_OUT_OF_MEMORY"),
            "Memory allocation failed - insufficient memory available"
        );
        assert_eq!(
            variant_to_message("ERROR_NOT_INITIALIZED"),
            "Library or resource not initialized - call initialization function first"
        );
        assert_eq!(
            variant_to_message("CUDNN_STATUS_BAD_PARAM"),
            "Bad parameter - check parameter values and types"
        );
        assert_eq!(
            variant_to_message("STATUS_INTERNAL_ERROR"),
            "Internal library error - this may indicate a bug"
        );
        assert_eq!(variant_to_message("MY_CUSTOM_ERROR"), "My custom error");
    }

    #[test]
    fn test_variant_name_prefixes() {
        // Test different error prefix conventions
        assert_eq!(to_rust_variant_name("ERR_TIMEOUT"), "Timeout");
        assert_eq!(to_rust_variant_name("ERROR_TIMEOUT"), "Timeout");
        assert_eq!(to_rust_variant_name("STATUS_TIMEOUT"), "Timeout");
        // CUDA prefix is kept as it's part of the name
        assert_eq!(to_rust_variant_name("CUDA_ERROR_TIMEOUT"), "CudaTimeout");
        assert_eq!(to_rust_variant_name("CUDNN_STATUS_ERROR"), "CudnnError");
    }

    #[test]
    fn test_variant_name_acronyms() {
        // Test acronym handling
        assert_eq!(to_rust_variant_name("ERROR_HTTP_FAILED"), "HttpFailed");
        assert_eq!(to_rust_variant_name("ERROR_SSL_ERROR"), "SslError");
        assert_eq!(to_rust_variant_name("ERROR_IO_ERROR"), "IoError");
    }

    #[test]
    fn test_variant_name_numbers() {
        // Test numbers in variant names
        assert_eq!(
            to_rust_variant_name("ERROR_VERSION_2_REQUIRED"),
            "Version2Required"
        );
        assert_eq!(to_rust_variant_name("ERROR_CUDA_11_NEEDED"), "Cuda11Needed");
    }

    #[test]
    fn test_variant_name_underscores() {
        // Test multiple underscores
        assert_eq!(
            to_rust_variant_name("ERROR__DOUBLE__UNDERSCORE"),
            "DoubleUnderscore"
        );
        assert_eq!(
            to_rust_variant_name("___LEADING_UNDERSCORES"),
            "LeadingUnderscores"
        );
    }

    #[test]
    fn test_error_message_consistency() {
        // Ensure all standard errors have reasonable messages
        let test_cases = vec![
            "ERROR_INVALID_HANDLE",
            "ERROR_FILE_NOT_FOUND",
            "ERROR_PERMISSION_DENIED",
            "ERROR_ALREADY_EXISTS",
            "ERROR_TIMEOUT",
        ];

        for error in test_cases {
            let message = variant_to_message(error);
            // Message should not be empty and should be different from input
            assert!(!message.is_empty());
            assert_ne!(message.to_uppercase(), error);
        }
    }

    #[test]
    fn test_error_message_fallback() {
        // Test that unknown errors get reasonable fallback messages
        let message = variant_to_message("TOTALLY_UNKNOWN_ERROR_CODE_XYZ");
        assert!(!message.is_empty());
        // Should convert to title case
        assert!(message.contains("Totally") || message.contains("totally"));
    }
}

/// Generate additional methods using smart error analysis
fn generate_smart_error_methods(
    code: &mut String,
    error_enum: &ErrorEnum,
    smart_errors: &SmartErrorAnalysis,
) {
    // Find the smart error type that corresponds to this error enum
    let smart_type = smart_errors
        .error_types
        .iter()
        .find(|t| t.name == error_enum.name);

    if smart_type.is_none() {
        return;
    }

    let smart_type = smart_type.unwrap();

    writeln!(code).unwrap();
    writeln!(
        code,
        "    /// Get error category (from smart error analysis)"
    )
    .unwrap();
    writeln!(code, "    pub fn category(&self) -> ErrorCategory {{").unwrap();
    writeln!(code, "        match self {{").unwrap();
    writeln!(
        code,
        "            Error::NullPointer => ErrorCategory::InvalidInput,"
    )
    .unwrap();
    writeln!(
        code,
        "            Error::InvalidString => ErrorCategory::InvalidInput,"
    )
    .unwrap();
    writeln!(
        code,
        "            Error::Unknown(_) => ErrorCategory::Other,"
    )
    .unwrap();

    for variant in &smart_type.variants {
        let rust_variant = to_rust_variant_name(&variant.name);
        let category = match variant.category {
            SmartErrorCategory::InvalidInput => "InvalidInput",
            SmartErrorCategory::ResourceUnavailable => "ResourceUnavailable",
            SmartErrorCategory::PermissionDenied => "PermissionDenied",
            SmartErrorCategory::NotSupported => "NotSupported",
            SmartErrorCategory::Internal => "Internal",
            SmartErrorCategory::State => "State",
            SmartErrorCategory::Memory => "Memory",
            SmartErrorCategory::Timeout => "Timeout",
            SmartErrorCategory::IO => "IO",
        };
        writeln!(
            code,
            "            Error::{} => ErrorCategory::{},",
            rust_variant, category
        )
        .unwrap();
    }

    writeln!(code, "        }}").unwrap();
    writeln!(code, "    }}").unwrap();
    writeln!(code).unwrap();

    writeln!(code, "    /// Get error severity level").unwrap();
    writeln!(code, "    pub fn severity(&self) -> ErrorSeverity {{").unwrap();
    writeln!(code, "        match self {{").unwrap();
    writeln!(
        code,
        "            Error::NullPointer => ErrorSeverity::Error,"
    )
    .unwrap();
    writeln!(
        code,
        "            Error::InvalidString => ErrorSeverity::Error,"
    )
    .unwrap();
    writeln!(
        code,
        "            Error::Unknown(_) => ErrorSeverity::Error,"
    )
    .unwrap();

    for variant in &smart_type.variants {
        let rust_variant = to_rust_variant_name(&variant.name);
        let severity = match variant.severity {
            ErrorSeverity::Info => "Info",
            ErrorSeverity::Warning => "Warning",
            ErrorSeverity::Error => "Error",
            ErrorSeverity::Fatal => "Fatal",
        };
        writeln!(
            code,
            "            Error::{} => ErrorSeverity::{},",
            rust_variant, severity
        )
        .unwrap();
    }

    writeln!(code, "        }}").unwrap();
    writeln!(code, "    }}").unwrap();
    writeln!(code).unwrap();

    writeln!(code, "    /// Get recovery suggestions for this error").unwrap();
    writeln!(code, "    pub fn recovery_suggestions(&self) -> &[&str] {{").unwrap();
    writeln!(code, "        match self {{").unwrap();
    writeln!(code, "            Error::NullPointer => &[\"Check that the function succeeded before using the result\"],").unwrap();
    writeln!(code, "            Error::InvalidString => &[\"Ensure string is valid UTF-8 and null-terminated\"],").unwrap();
    writeln!(
        code,
        "            Error::Unknown(_) => &[\"Check library documentation for this error code\"],"
    )
    .unwrap();

    for variant in &smart_type.variants {
        let rust_variant = to_rust_variant_name(&variant.name);
        if !variant.recovery.is_empty() {
            // Take first 3 recovery suggestions
            let suggestions: Vec<_> = variant.recovery.iter().take(3).collect();
            let suggestions_str = suggestions
                .iter()
                .map(|s| format!("\"{}\"", s.replace("\"", "\\\"")))
                .collect::<Vec<_>>()
                .join(", ");
            writeln!(
                code,
                "            Error::{} => &[{}],",
                rust_variant, suggestions_str
            )
            .unwrap();
        } else {
            writeln!(code, "            Error::{} => &[],", rust_variant).unwrap();
        }
    }

    writeln!(code, "        }}").unwrap();
    writeln!(code, "    }}").unwrap();
}

/// Re-export SmartErrorCategory as ErrorCategory for generated code
pub use crate::analyzer::SmartErrorCategory as ErrorCategory;

#[cfg(test)]
mod smart_error_tests {
    use super::*;

    #[test]
    fn test_variant_to_message() {
        assert_eq!(
            variant_to_message("ERROR_NOT_SUPPORTED"),
            "Operation not supported in this configuration or version"
        );
    }
}
