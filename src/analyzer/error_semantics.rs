//! Error code semantics analyzer for understanding error handling patterns.
//!
//! This module analyzes documentation and code to extract:
//! - Error code meanings and categories
//! - Fatal vs recoverable errors
//! - Retry strategies
//! - Error context requirements

use std::collections::HashMap;

/// Represents semantic information about an error code or variant
#[derive(Debug, Clone, PartialEq)]
pub struct ErrorInfo {
    /// Error code name or identifier
    pub code: String,
    /// Human-readable description
    pub description: String,
    /// Whether this error is fatal (cannot continue)
    pub is_fatal: bool,
    /// Whether the operation should be retried
    pub is_retryable: bool,
    /// Category of error
    pub category: ErrorCategory,
    /// Additional context needed when error occurs
    pub required_context: Vec<String>,
    /// Suggested recovery action
    pub recovery_action: Option<String>,
}

/// Category of error for semantic grouping
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ErrorCategory {
    /// Invalid input parameters
    InvalidInput,
    /// Resource not found
    NotFound,
    /// Resource exhausted (memory, handles, etc.)
    ResourceExhausted,
    /// Permission denied
    PermissionDenied,
    /// Operation would block
    WouldBlock,
    /// Timeout occurred
    Timeout,
    /// Network or I/O error
    IoError,
    /// Internal error or bug
    InternalError,
    /// Operation not supported
    NotSupported,
    /// Resource already exists
    AlreadyExists,
    /// Operation cancelled
    Cancelled,
    /// Unknown error type
    Unknown,
}

/// Complete error semantics information
#[derive(Debug, Clone, PartialEq)]
pub struct ErrorSemantics {
    /// Map of error codes to their semantic information
    pub errors: HashMap<String, ErrorInfo>,
    /// Overall confidence in the analysis
    pub confidence: f64,
}

impl ErrorSemantics {
    /// Checks if there are any analyzed errors
    pub fn has_errors(&self) -> bool {
        !self.errors.is_empty()
    }

    /// Gets all fatal errors
    pub fn fatal_errors(&self) -> Vec<&ErrorInfo> {
        self.errors
            .values()
            .filter(|e| e.is_fatal)
            .collect()
    }

    /// Gets all retryable errors
    pub fn retryable_errors(&self) -> Vec<&ErrorInfo> {
        self.errors
            .values()
            .filter(|e| e.is_retryable)
            .collect()
    }

    /// Generates documentation for error handling
    pub fn generate_documentation(&self) -> String {
        if !self.has_errors() {
            return String::new();
        }

        let mut docs = String::from("/// # Error Handling\n");
        docs.push_str("///\n");

        // Group by category
        let mut by_category: HashMap<ErrorCategory, Vec<&ErrorInfo>> = HashMap::new();
        for error in self.errors.values() {
            by_category
                .entry(error.category)
                .or_default()
                .push(error);
        }

        // Fatal errors first
        let fatal: Vec<_> = self.errors.values().filter(|e| e.is_fatal).collect();
        if !fatal.is_empty() {
            docs.push_str("/// ## Fatal Errors (Operation Cannot Continue)\n");
            docs.push_str("///\n");
            for error in fatal {
                docs.push_str(&format!("/// - `{}`: {}\n", error.code, error.description));
                if let Some(recovery) = &error.recovery_action {
                    docs.push_str(&format!("///   Recovery: {}\n", recovery));
                }
            }
            docs.push_str("///\n");
        }

        // Retryable errors
        let retryable: Vec<_> = self.errors.values().filter(|e| e.is_retryable && !e.is_fatal).collect();
        if !retryable.is_empty() {
            docs.push_str("/// ## Retryable Errors (May Succeed on Retry)\n");
            docs.push_str("///\n");
            for error in retryable {
                docs.push_str(&format!("/// - `{}`: {}\n", error.code, error.description));
                if let Some(recovery) = &error.recovery_action {
                    docs.push_str(&format!("///   Retry: {}\n", recovery));
                }
            }
            docs.push_str("///\n");
        }

        // Other errors by category
        let other: Vec<_> = self.errors.values()
            .filter(|e| !e.is_fatal && !e.is_retryable)
            .collect();
        if !other.is_empty() {
            docs.push_str("/// ## Other Errors\n");
            docs.push_str("///\n");
            for error in other {
                docs.push_str(&format!("/// - `{}`: {}\n", error.code, error.description));
            }
            docs.push_str("///\n");
        }

        docs
    }
}

/// Main error semantics analyzer
#[derive(Debug)]
pub struct ErrorSemanticsAnalyzer {
    /// Cache of analyzed error semantics by function name
    cache: HashMap<String, ErrorSemantics>,
}

impl ErrorSemanticsAnalyzer {
    /// Creates a new error semantics analyzer
    pub fn new() -> Self {
        Self {
            cache: HashMap::new(),
        }
    }

    /// Analyzes error handling patterns from documentation
    ///
    /// # Arguments
    /// * `function_name` - Name of the function
    /// * `documentation` - Combined documentation text
    /// * `return_type` - Return type of the function (e.g., "int", "Status")
    ///
    /// # Returns
    /// Error semantics information
    pub fn analyze(
        &mut self,
        function_name: &str,
        documentation: &str,
        return_type: &str,
    ) -> ErrorSemantics {
        // Check cache
        if let Some(cached) = self.cache.get(function_name) {
            return cached.clone();
        }

        let mut errors = HashMap::new();
        let mut total_confidence = 0.0;
        let mut error_count = 0;

        // Detect error codes from documentation
        errors.extend(self.extract_error_codes(documentation));
        error_count += errors.len();
        total_confidence += errors.len() as f64 * 0.8;

        // Analyze return type
        if self.is_error_return_type(return_type) {
            // Add generic error handling info based on return type
            if errors.is_empty() {
                errors.extend(self.infer_from_return_type(return_type));
                total_confidence += 0.5;
                error_count += 1;
            }
        }

        let confidence = if error_count > 0 {
            total_confidence / error_count as f64
        } else {
            0.0
        };

        let semantics = ErrorSemantics { errors, confidence };

        // Cache the result
        self.cache.insert(function_name.to_string(), semantics.clone());
        semantics
    }

    /// Extracts error codes from documentation
    fn extract_error_codes(&self, docs: &str) -> HashMap<String, ErrorInfo> {
        let mut errors = HashMap::new();

        // Split by both newlines and periods to handle multiple errors per line
        let sentences: Vec<&str> = docs
            .split(['\n', '.'])
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .collect();

        for line in sentences {
            let lower = line.to_lowercase();

            // Pattern 1: "Returns ERROR_CODE on failure"
            if lower.contains("returns") && (lower.contains("error") || lower.contains("on failure"))
                && let Some(info) = self.parse_return_error(line) {
                    errors.insert(info.code.clone(), info);
                }

            // Pattern 2: "ERROR_CODE: description"
            if line.contains(':') && self.looks_like_error_code(line)
                && let Some(info) = self.parse_error_line(line) {
                    errors.insert(info.code.clone(), info);
                }

            // Pattern 3: "@retval ERROR_CODE description" (Doxygen)
            if (lower.contains("@retval") || lower.contains("\\retval"))
                && let Some(info) = self.parse_retval(line) {
                    errors.insert(info.code.clone(), info);
                }

            // Pattern 4: "Fails with ERROR_CODE if..."
            if (lower.contains("fails with") || lower.contains("returns"))
                && let Some(info) = self.parse_fails_with(line) {
                    errors.insert(info.code.clone(), info);
                }
        }

        errors
    }

    /// Parses "Returns ERROR_CODE on failure" patterns
    fn parse_return_error(&self, line: &str) -> Option<ErrorInfo> {
        // Extract error code (usually ALL_CAPS or prefixed)
        let words: Vec<&str> = line.split_whitespace().collect();
        for (i, word) in words.iter().enumerate() {
            if self.is_error_code_name(word) {
                let code = word.trim_matches(|c: char| !c.is_alphanumeric() && c != '_').to_string();
                
                // Get description (rest of line after code)
                let desc_start = i + 1;
                let description = if desc_start < words.len() {
                    words[desc_start..].join(" ")
                } else {
                    "Error occurred".to_string()
                };

                return Some(ErrorInfo {
                    code: code.clone(),
                    description: description.clone(),
                    is_fatal: self.is_fatal_error(&code, &description),
                    is_retryable: self.is_retryable_error(&code, &description),
                    category: self.categorize_error(&code, &description),
                    required_context: Vec::new(),
                    recovery_action: self.extract_recovery_action(&description),
                });
            }
        }
        None
    }

    /// Parses "ERROR_CODE: description" patterns
    fn parse_error_line(&self, line: &str) -> Option<ErrorInfo> {
        let parts: Vec<&str> = line.splitn(2, ':').collect();
        if parts.len() != 2 {
            return None;
        }

        let code = parts[0].trim();
        if !self.is_error_code_name(code) {
            return None;
        }

        let description = parts[1].trim().to_string();

        Some(ErrorInfo {
            code: code.to_string(),
            description: description.clone(),
            is_fatal: self.is_fatal_error(code, &description),
            is_retryable: self.is_retryable_error(code, &description),
            category: self.categorize_error(code, &description),
            required_context: Vec::new(),
            recovery_action: self.extract_recovery_action(&description),
        })
    }

    /// Parses "@retval ERROR_CODE description" patterns
    fn parse_retval(&self, line: &str) -> Option<ErrorInfo> {
        let after_retval = if line.to_lowercase().contains("@retval") {
            line.split("@retval").nth(1)?
        } else {
            line.split("\\retval").nth(1)?
        };

        let parts: Vec<&str> = after_retval.trim().splitn(2, ' ').collect();
        if parts.is_empty() {
            return None;
        }

        let code = parts[0].trim();
        let description = if parts.len() > 1 {
            parts[1].trim().to_string()
        } else {
            "Error occurred".to_string()
        };

        Some(ErrorInfo {
            code: code.to_string(),
            description: description.clone(),
            is_fatal: self.is_fatal_error(code, &description),
            is_retryable: self.is_retryable_error(code, &description),
            category: self.categorize_error(code, &description),
            required_context: Vec::new(),
            recovery_action: self.extract_recovery_action(&description),
        })
    }

    /// Parses "Fails with ERROR_CODE if..." patterns
    fn parse_fails_with(&self, line: &str) -> Option<ErrorInfo> {
        let lower = line.to_lowercase();
        let start_pos = if lower.contains("fails with") {
            lower.find("fails with")? + "fails with".len()
        } else if lower.contains("returns") {
            lower.find("returns")? + "returns".len()
        } else {
            return None;
        };

        let after = &line[start_pos..];
        let words: Vec<&str> = after.split_whitespace().collect();
        
        for (i, word) in words.iter().enumerate() {
            if self.is_error_code_name(word) {
                let code = word.trim_matches(|c: char| !c.is_alphanumeric() && c != '_').to_string();
                
                let desc_start = i + 1;
                let description = if desc_start < words.len() {
                    words[desc_start..].join(" ")
                } else {
                    "Error occurred".to_string()
                };

                return Some(ErrorInfo {
                    code: code.clone(),
                    description: description.clone(),
                    is_fatal: self.is_fatal_error(&code, &description),
                    is_retryable: self.is_retryable_error(&code, &description),
                    category: self.categorize_error(&code, &description),
                    required_context: Vec::new(),
                    recovery_action: self.extract_recovery_action(&description),
                });
            }
        }
        None
    }

    /// Checks if a word looks like an error code
    fn is_error_code_name(&self, word: &str) -> bool {
        let clean = word.trim_matches(|c: char| !c.is_alphanumeric() && c != '_');
        
        // Must be at least 3 characters
        if clean.len() < 3 {
            return false;
        }

        // Common error code patterns
        let lower = clean.to_lowercase();
        
        // Explicit error prefixes
        if lower.starts_with("err_") || lower.starts_with("error_") || 
           lower.starts_with("e_") || lower.ends_with("_error") ||
           lower.ends_with("_err") {
            return true;
        }

        // Status code patterns
        if lower.starts_with("status_") || lower.ends_with("_status") {
            return true;
        }

        // Common negative patterns
        if lower.starts_with("invalid_") || lower.starts_with("not_") ||
           lower.contains("_failed") || lower.contains("_failure") {
            return true;
        }

        // ALL_CAPS with underscores (common C error code style)
        if clean.chars().all(|c| c.is_uppercase() || c == '_' || c.is_numeric()) &&
           clean.contains('_') {
            return true;
        }

        false
    }

    /// Checks if line looks like it contains an error code
    fn looks_like_error_code(&self, line: &str) -> bool {
        let before_colon = line.split(':').next().unwrap_or("");
        self.is_error_code_name(before_colon.trim())
    }

    /// Determines if an error is fatal
    fn is_fatal_error(&self, code: &str, description: &str) -> bool {
        let lower_code = code.to_lowercase();
        let lower_desc = description.to_lowercase();

        // Fatal keywords
        lower_desc.contains("fatal") ||
        lower_desc.contains("cannot continue") ||
        lower_desc.contains("unrecoverable") ||
        lower_desc.contains("must restart") ||
        lower_desc.contains("must recreate") ||
        lower_code.contains("fatal") ||
        lower_code.contains("unrecoverable")
    }

    /// Determines if an error is retryable
    fn is_retryable_error(&self, code: &str, description: &str) -> bool {
        let lower_code = code.to_lowercase();
        let lower_desc = description.to_lowercase();

        // Retryable keywords
        lower_desc.contains("retry") ||
        lower_desc.contains("try again") ||
        lower_desc.contains("temporary") ||
        lower_desc.contains("transient") ||
        lower_code.contains("again") ||
        lower_code.contains("retry") ||
        lower_code.contains("timeout") ||
        lower_code.contains("would_block")
    }

    /// Categorizes an error based on code and description
    fn categorize_error(&self, code: &str, description: &str) -> ErrorCategory {
        let lower_code = code.to_lowercase();
        let lower_desc = description.to_lowercase();

        if lower_code.contains("invalid") || lower_desc.contains("invalid") ||
           lower_code.contains("argument") || lower_desc.contains("invalid input") {
            return ErrorCategory::InvalidInput;
        }

        if lower_code.contains("not_found") || lower_desc.contains("not found") ||
           lower_code.contains("notfound") || lower_code.contains("noent") {
            return ErrorCategory::NotFound;
        }

        if lower_code.contains("nomem") || lower_desc.contains("out of memory") ||
           lower_code.contains("exhausted") || lower_desc.contains("resource exhausted") {
            return ErrorCategory::ResourceExhausted;
        }

        if lower_code.contains("permission") || lower_desc.contains("permission denied") ||
           lower_code.contains("access") || lower_code.contains("forbidden") {
            return ErrorCategory::PermissionDenied;
        }

        if lower_code.contains("would_block") || lower_code.contains("wouldblock") ||
           lower_desc.contains("would block") {
            return ErrorCategory::WouldBlock;
        }

        if lower_code.contains("timeout") || lower_desc.contains("timed out") ||
           lower_desc.contains("timeout") {
            return ErrorCategory::Timeout;
        }

        if lower_code.contains("io") || lower_desc.contains("i/o error") ||
           lower_code.contains("network") || lower_desc.contains("network") {
            return ErrorCategory::IoError;
        }

        if lower_code.contains("internal") || lower_desc.contains("internal error") ||
           lower_code.contains("bug") {
            return ErrorCategory::InternalError;
        }

        if lower_code.contains("not_supported") || lower_desc.contains("not supported") ||
           lower_code.contains("unsupported") {
            return ErrorCategory::NotSupported;
        }

        if lower_code.contains("exists") || lower_desc.contains("already exists") {
            return ErrorCategory::AlreadyExists;
        }

        if lower_code.contains("cancel") || lower_desc.contains("cancelled") {
            return ErrorCategory::Cancelled;
        }

        ErrorCategory::Unknown
    }

    /// Extracts recovery action from description
    fn extract_recovery_action(&self, description: &str) -> Option<String> {
        let lower = description.to_lowercase();

        if lower.contains("retry") {
            return Some("Retry the operation".to_string());
        }
        if lower.contains("try again later") {
            return Some("Try again later".to_string());
        }
        if lower.contains("increase") {
            return Some("Increase resource limit".to_string());
        }
        if lower.contains("check") && lower.contains("parameter") {
            return Some("Check input parameters".to_string());
        }

        None
    }

    /// Infers error information from return type
    fn infer_from_return_type(&self, return_type: &str) -> HashMap<String, ErrorInfo> {
        let mut errors = HashMap::new();

        // Common error return types
        if return_type == "int" || return_type == "i32" {
            errors.insert(
                "negative".to_string(),
                ErrorInfo {
                    code: "negative".to_string(),
                    description: "Returns negative value on error".to_string(),
                    is_fatal: false,
                    is_retryable: false,
                    category: ErrorCategory::Unknown,
                    required_context: Vec::new(),
                    recovery_action: None,
                },
            );
        }

        errors
    }

    /// Checks if return type indicates error handling
    fn is_error_return_type(&self, return_type: &str) -> bool {
        let lower = return_type.to_lowercase();
        lower.contains("status") ||
        lower.contains("error") ||
        lower.contains("result") ||
        return_type == "int" ||
        return_type == "i32"
    }
}

impl Default for ErrorSemanticsAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_return_error() {
        let mut analyzer = ErrorSemanticsAnalyzer::new();
        let docs = "Returns ERROR_INVALID_ARGUMENT if the parameter is null.";
        
        let semantics = analyzer.analyze("test_func", docs, "int");
        assert!(semantics.has_errors());
        assert!(semantics.errors.contains_key("ERROR_INVALID_ARGUMENT"));
        
        let error = &semantics.errors["ERROR_INVALID_ARGUMENT"];
        assert_eq!(error.category, ErrorCategory::InvalidInput);
    }

    #[test]
    fn test_parse_error_line() {
        let mut analyzer = ErrorSemanticsAnalyzer::new();
        let docs = "ERROR_TIMEOUT: Operation timed out and should be retried.";
        
        let semantics = analyzer.analyze("test_func", docs, "int");
        assert!(semantics.has_errors());
        assert!(semantics.errors.contains_key("ERROR_TIMEOUT"));
        
        let error = &semantics.errors["ERROR_TIMEOUT"];
        assert_eq!(error.category, ErrorCategory::Timeout);
        assert!(error.is_retryable);
    }

    #[test]
    fn test_fatal_error_detection() {
        let mut analyzer = ErrorSemanticsAnalyzer::new();
        let docs = "ERROR_FATAL: Fatal error, operation cannot continue.";
        
        let semantics = analyzer.analyze("test_func", docs, "int");
        let error = &semantics.errors["ERROR_FATAL"];
        assert!(error.is_fatal);
    }

    #[test]
    fn test_retryable_error_detection() {
        let mut analyzer = ErrorSemanticsAnalyzer::new();
        let docs = "WOULD_BLOCK: Operation would block, try again later.";
        
        let semantics = analyzer.analyze("test_func", docs, "int");
        let error = &semantics.errors["WOULD_BLOCK"];
        assert!(error.is_retryable);
        assert_eq!(error.category, ErrorCategory::WouldBlock);
    }

    #[test]
    fn test_retval_parsing() {
        let mut analyzer = ErrorSemanticsAnalyzer::new();
        let docs = "@retval ERROR_NOT_FOUND The requested resource was not found.";
        
        let semantics = analyzer.analyze("test_func", docs, "int");
        assert!(semantics.errors.contains_key("ERROR_NOT_FOUND"));
        
        let error = &semantics.errors["ERROR_NOT_FOUND"];
        assert_eq!(error.category, ErrorCategory::NotFound);
    }

    #[test]
    fn test_documentation_generation() {
        let mut analyzer = ErrorSemanticsAnalyzer::new();
        let docs = "ERROR_FATAL: Fatal error. ERROR_RETRY: Retry the operation.";
        
        let semantics = analyzer.analyze("test_func", docs, "int");
        let doc = semantics.generate_documentation();
        
        assert!(doc.contains("# Error Handling"));
        assert!(doc.contains("Fatal Errors"));
        assert!(doc.contains("ERROR_FATAL"));
    }

    #[test]
    fn test_error_categorization() {
        let mut analyzer = ErrorSemanticsAnalyzer::new();
        
        let semantics = analyzer.analyze(
            "test",
            "ERROR_NOMEM: Out of memory. ERROR_IO: I/O error occurred.",
            "int"
        );
        
        // Check that errors were found
        assert!(semantics.has_errors(), "Should have found errors");
        assert!(semantics.errors.contains_key("ERROR_NOMEM"), "Should have ERROR_NOMEM");
        assert!(semantics.errors.contains_key("ERROR_IO"), "Should have ERROR_IO");
        
        assert_eq!(
            semantics.errors["ERROR_NOMEM"].category,
            ErrorCategory::ResourceExhausted
        );
        assert_eq!(
            semantics.errors["ERROR_IO"].category,
            ErrorCategory::IoError
        );
    }

    #[test]
    fn test_cache() {
        let mut analyzer = ErrorSemanticsAnalyzer::new();
        let docs = "ERROR_TEST: Test error.";
        
        let semantics1 = analyzer.analyze("test_func", docs, "int");
        let semantics2 = analyzer.analyze("test_func", docs, "int");
        
        assert_eq!(semantics1, semantics2);
    }

    #[test]
    fn test_no_errors() {
        let mut analyzer = ErrorSemanticsAnalyzer::new();
        let docs = "This function always succeeds.";
        
        let semantics = analyzer.analyze("test_func", docs, "void");
        assert!(!semantics.has_errors());
    }

    #[test]
    fn test_fatal_and_retryable_getters() {
        let mut analyzer = ErrorSemanticsAnalyzer::new();
        let docs = "ERROR_FATAL: Fatal. ERROR_TIMEOUT: Timeout, retry.";
        
        let semantics = analyzer.analyze("test", docs, "int");
        
        assert_eq!(semantics.fatal_errors().len(), 1);
        assert_eq!(semantics.retryable_errors().len(), 1);
    }
}
