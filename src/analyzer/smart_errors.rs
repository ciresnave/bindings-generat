//! Smart error type generation with recovery suggestions
//!
//! This module generates intelligent error types that include:
//! - Detailed error information with context
//! - Recovery suggestions
//! - Error categorization
//! - Automatic error chaining

use crate::ffi::{FfiEnum, FfiEnumVariant, FfiInfo};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::info;

/// Smart error type analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SmartErrorAnalysis {
    /// Error type definitions
    pub error_types: Vec<SmartErrorType>,
    /// Error recovery strategies
    pub recovery_strategies: HashMap<String, Vec<RecoveryStrategy>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SmartErrorType {
    /// Error enum name
    pub name: String,
    /// Error variants with enriched information
    pub variants: Vec<SmartErrorVariant>,
    /// Common error patterns
    pub patterns: Vec<ErrorPattern>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SmartErrorVariant {
    /// Variant name
    pub name: String,
    /// Original FFI value
    pub ffi_value: Option<i64>,
    /// Error category
    pub category: SmartErrorCategory,
    /// Error severity
    pub severity: ErrorSeverity,
    /// Detailed description
    pub description: String,
    /// Possible causes
    pub causes: Vec<String>,
    /// Recovery suggestions
    pub recovery: Vec<String>,
    /// Related errors
    pub related: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum SmartErrorCategory {
    /// Invalid input parameters
    InvalidInput,
    /// Resource not available
    ResourceUnavailable,
    /// Permission denied
    PermissionDenied,
    /// Operation not supported
    NotSupported,
    /// Internal error
    Internal,
    /// State error (wrong order of operations)
    State,
    /// Memory error
    Memory,
    /// Timeout error
    Timeout,
    /// Network/IO error
    IO,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum ErrorSeverity {
    /// Informational, not really an error
    Info,
    /// Warning, operation succeeded with caveats
    Warning,
    /// Recoverable error
    Error,
    /// Fatal error, cannot continue
    Fatal,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorPattern {
    /// Pattern name
    pub name: String,
    /// Error variants in this pattern
    pub variants: Vec<String>,
    /// Common cause
    pub common_cause: String,
    /// General recovery approach
    pub recovery_approach: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecoveryStrategy {
    /// Error variant name
    pub error: String,
    /// Strategy description
    pub strategy: String,
    /// Code example
    pub example: Option<String>,
}

/// Analyzes errors to generate smart error types
pub struct SmartErrorAnalyzer;

impl SmartErrorAnalyzer {
    pub fn new() -> Self {
        Self
    }

    /// Analyze error enums
    pub fn analyze(&self, ffi_info: &FfiInfo) -> SmartErrorAnalysis {
        let mut error_types = Vec::new();

        for error_enum in &ffi_info.enums {
            if self.is_error_enum(error_enum) {
                error_types.push(self.analyze_error_enum(error_enum));
            }
        }

        let recovery_strategies = self.generate_recovery_strategies(&error_types);

        info!("Smart error analysis: {} error types", error_types.len());

        SmartErrorAnalysis {
            error_types,
            recovery_strategies,
        }
    }

    /// Check if enum is an error enum
    fn is_error_enum(&self, enum_def: &FfiEnum) -> bool {
        let name_lower = enum_def.name.to_lowercase();
        name_lower.contains("error")
            || name_lower.contains("status")
            || name_lower.contains("result")
            || name_lower.contains("code")
    }

    /// Analyze single error enum
    fn analyze_error_enum(&self, error_enum: &FfiEnum) -> SmartErrorType {
        let mut variants = Vec::new();

        for variant in &error_enum.variants {
            variants.push(self.analyze_variant(variant, error_enum));
        }

        let patterns = self.identify_patterns(&variants);

        SmartErrorType {
            name: error_enum.name.clone(),
            variants,
            patterns,
        }
    }

    /// Analyze single error variant
    fn analyze_variant(&self, variant: &FfiEnumVariant, parent: &FfiEnum) -> SmartErrorVariant {
        let category = self.categorize_error(&variant.name);
        let severity = self.determine_severity(&variant.name, &category);
        let causes = self.infer_causes(&variant.name);
        let recovery = self.suggest_recovery(&variant.name, &category);
        let related = self.find_related_errors(variant, parent);

        let description = variant
            .docs
            .clone()
            .unwrap_or_else(|| self.generate_description(&variant.name, &category));

        SmartErrorVariant {
            name: variant.name.clone(),
            ffi_value: variant.value,
            category,
            severity,
            description,
            causes,
            recovery,
            related,
        }
    }

    /// Categorize error by name
    fn categorize_error(&self, name: &str) -> SmartErrorCategory {
        let name_lower = name.to_lowercase();

        if name_lower.contains("invalid") || name_lower.contains("bad") {
            SmartErrorCategory::InvalidInput
        } else if name_lower.contains("not_found")
            || name_lower.contains("unavailable")
            || name_lower.contains("busy")
        {
            SmartErrorCategory::ResourceUnavailable
        } else if name_lower.contains("permission") || name_lower.contains("denied") {
            SmartErrorCategory::PermissionDenied
        } else if name_lower.contains("not_supported") || name_lower.contains("unimplemented") {
            SmartErrorCategory::NotSupported
        } else if name_lower.contains("timeout") {
            SmartErrorCategory::Timeout
        } else if name_lower.contains("memory") || name_lower.contains("alloc") {
            SmartErrorCategory::Memory
        } else if name_lower.contains("io") || name_lower.contains("network") {
            SmartErrorCategory::IO
        } else if name_lower.contains("state") || name_lower.contains("order") {
            SmartErrorCategory::State
        } else {
            SmartErrorCategory::Internal
        }
    }

    /// Determine error severity
    fn determine_severity(&self, name: &str, category: &SmartErrorCategory) -> ErrorSeverity {
        let name_lower = name.to_lowercase();

        if name_lower.contains("fatal") || name_lower.contains("panic") {
            return ErrorSeverity::Fatal;
        }
        if name_lower.contains("warning") || name_lower.contains("warn") {
            return ErrorSeverity::Warning;
        }
        if name_lower.contains("info") {
            return ErrorSeverity::Info;
        }

        match category {
            SmartErrorCategory::Memory | SmartErrorCategory::Internal => ErrorSeverity::Fatal,
            SmartErrorCategory::InvalidInput | SmartErrorCategory::State => ErrorSeverity::Error,
            _ => ErrorSeverity::Error,
        }
    }

    /// Infer possible causes
    fn infer_causes(&self, name: &str) -> Vec<String> {
        let name_lower = name.to_lowercase();
        let mut causes = Vec::new();

        if name_lower.contains("invalid") {
            causes.push("Parameter validation failed".to_string());
            causes.push("Value outside acceptable range".to_string());
        }
        if name_lower.contains("null") {
            causes.push("Null pointer passed where non-null expected".to_string());
        }
        if name_lower.contains("memory") || name_lower.contains("alloc") {
            causes.push("Insufficient memory available".to_string());
            causes.push("Memory allocation failed".to_string());
        }
        if name_lower.contains("not_found") {
            causes.push("Resource does not exist".to_string());
            causes.push("Invalid identifier or handle".to_string());
        }
        if name_lower.contains("timeout") {
            causes.push("Operation took too long".to_string());
            causes.push("Deadline exceeded".to_string());
        }

        if causes.is_empty() {
            causes.push("Operation failed".to_string());
        }

        causes
    }

    /// Suggest recovery actions
    fn suggest_recovery(&self, _name: &str, category: &SmartErrorCategory) -> Vec<String> {
        let mut recovery = Vec::new();

        match category {
            SmartErrorCategory::InvalidInput => {
                recovery.push("Validate input parameters before calling".to_string());
                recovery.push("Check parameter documentation for valid ranges".to_string());
            }
            SmartErrorCategory::ResourceUnavailable => {
                recovery.push("Retry the operation after a delay".to_string());
                recovery.push("Check if resource exists before accessing".to_string());
            }
            SmartErrorCategory::PermissionDenied => {
                recovery.push("Verify required permissions are granted".to_string());
                recovery.push("Run with elevated privileges if appropriate".to_string());
            }
            SmartErrorCategory::NotSupported => {
                recovery.push("Check API version compatibility".to_string());
                recovery.push("Use alternative function if available".to_string());
            }
            SmartErrorCategory::Memory => {
                recovery.push("Free unused resources".to_string());
                recovery.push("Reduce memory usage or increase available memory".to_string());
            }
            SmartErrorCategory::Timeout => {
                recovery.push("Increase timeout duration".to_string());
                recovery.push("Retry with exponential backoff".to_string());
            }
            SmartErrorCategory::State => {
                recovery.push("Check function call order".to_string());
                recovery.push("Ensure prerequisites are met".to_string());
            }
            _ => {
                recovery.push("Check error documentation".to_string());
                recovery.push("Contact support if error persists".to_string());
            }
        }

        recovery
    }

    /// Find related error variants
    fn find_related_errors(&self, variant: &FfiEnumVariant, parent: &FfiEnum) -> Vec<String> {
        let mut related = Vec::new();
        let name_lower = variant.name.to_lowercase();

        for other in &parent.variants {
            if other.name == variant.name {
                continue;
            }

            let other_lower = other.name.to_lowercase();

            // Check for similar names
            if name_lower.contains("invalid") && other_lower.contains("invalid") {
                related.push(other.name.clone());
            } else if name_lower.contains("not_found") && other_lower.contains("not_found") {
                related.push(other.name.clone());
            }
        }

        related
    }

    /// Generate description if not available
    fn generate_description(&self, name: &str, category: &SmartErrorCategory) -> String {
        format!("{:?} error: {}", category, name)
    }

    /// Identify common error patterns
    fn identify_patterns(&self, variants: &[SmartErrorVariant]) -> Vec<ErrorPattern> {
        let mut patterns = Vec::new();

        // Invalid parameter pattern
        let invalid_variants: Vec<String> = variants
            .iter()
            .filter(|v| v.category == SmartErrorCategory::InvalidInput)
            .map(|v| v.name.clone())
            .collect();

        if !invalid_variants.is_empty() {
            patterns.push(ErrorPattern {
                name: "Invalid Input Pattern".to_string(),
                variants: invalid_variants,
                common_cause: "Parameter validation failed".to_string(),
                recovery_approach: "Validate parameters before calling the function".to_string(),
            });
        }

        // Resource pattern
        let resource_variants: Vec<String> = variants
            .iter()
            .filter(|v| v.category == SmartErrorCategory::ResourceUnavailable)
            .map(|v| v.name.clone())
            .collect();

        if !resource_variants.is_empty() {
            patterns.push(ErrorPattern {
                name: "Resource Unavailable Pattern".to_string(),
                variants: resource_variants,
                common_cause: "Required resource not available".to_string(),
                recovery_approach: "Check resource availability and retry if appropriate"
                    .to_string(),
            });
        }

        patterns
    }

    /// Generate recovery strategies
    fn generate_recovery_strategies(
        &self,
        error_types: &[SmartErrorType],
    ) -> HashMap<String, Vec<RecoveryStrategy>> {
        let mut strategies = HashMap::new();

        for error_type in error_types {
            let mut type_strategies = Vec::new();

            for variant in &error_type.variants {
                for (idx, recovery_text) in variant.recovery.iter().enumerate() {
                    type_strategies.push(RecoveryStrategy {
                        error: variant.name.clone(),
                        strategy: recovery_text.clone(),
                        example: if idx == 0 {
                            Some(self.generate_recovery_example(&variant.name, recovery_text))
                        } else {
                            None
                        },
                    });
                }
            }

            strategies.insert(error_type.name.clone(), type_strategies);
        }

        strategies
    }

    /// Generate code example for recovery
    fn generate_recovery_example(&self, error_name: &str, strategy: &str) -> String {
        if strategy.contains("retry") {
            format!(
                "loop {{\n    match operation() {{\n        Ok(result) => break result,\n        \
                 Err(Error::{}) => {{\n            std::thread::sleep(Duration::from_millis(100));\n            \
                 continue;\n        }}\n        Err(e) => return Err(e),\n    }}\n}}",
                error_name
            )
        } else if strategy.contains("validate") {
            format!(
                "if !is_valid(param) {{\n    return Err(Error::{});\n}}\noperation(param)",
                error_name
            )
        } else {
            format!("// Handle Error::{}\n// {}", error_name, strategy)
        }
    }

    /// Generate Rust error type code
    pub fn generate_error_type(&self, error_type: &SmartErrorType) -> String {
        let mut code = String::new();

        code.push_str(&format!("/// Smart error type for {}\n", error_type.name));
        code.push_str("#[derive(Debug, Clone, PartialEq, Eq)]\n");
        code.push_str(&format!("pub enum {} {{\n", error_type.name));

        for variant in &error_type.variants {
            code.push_str(&format!("    /// {}\n", variant.description));
            code.push_str(&format!("    ///\n"));
            code.push_str(&format!("    /// **Severity:** {:?}\n", variant.severity));
            code.push_str(&format!("    /// **Category:** {:?}\n", variant.category));

            if !variant.causes.is_empty() {
                code.push_str(&format!("    ///\n"));
                code.push_str(&format!("    /// **Possible causes:**\n"));
                for cause in &variant.causes {
                    code.push_str(&format!("    /// - {}\n", cause));
                }
            }

            if !variant.recovery.is_empty() {
                code.push_str(&format!("    ///\n"));
                code.push_str(&format!("    /// **Recovery:**\n"));
                for recovery in &variant.recovery {
                    code.push_str(&format!("    /// - {}\n", recovery));
                }
            }

            code.push_str(&format!("    {},\n", variant.name));
        }

        code.push_str("}\n\n");

        // Implement Display
        code.push_str(&format!(
            "impl std::fmt::Display for {} {{\n",
            error_type.name
        ));
        code.push_str("    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {\n");
        code.push_str("        match self {\n");

        for variant in &error_type.variants {
            code.push_str(&format!(
                "            Self::{} => write!(f, \"{}\"),\n",
                variant.name, variant.description
            ));
        }

        code.push_str("        }\n");
        code.push_str("    }\n");
        code.push_str("}\n\n");

        // Implement std::error::Error
        code.push_str(&format!(
            "impl std::error::Error for {} {{}}\n",
            error_type.name
        ));

        code
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_categorization() {
        let analyzer = SmartErrorAnalyzer::new();

        assert_eq!(
            analyzer.categorize_error("INVALID_PARAMETER"),
            SmartErrorCategory::InvalidInput
        );
        assert_eq!(
            analyzer.categorize_error("NOT_FOUND"),
            SmartErrorCategory::ResourceUnavailable
        );
        assert_eq!(
            analyzer.categorize_error("PERMISSION_DENIED"),
            SmartErrorCategory::PermissionDenied
        );
    }

    #[test]
    fn test_severity_determination() {
        let analyzer = SmartErrorAnalyzer::new();

        assert_eq!(
            analyzer.determine_severity("FATAL_ERROR", &SmartErrorCategory::Internal),
            ErrorSeverity::Fatal
        );
        assert_eq!(
            analyzer.determine_severity("WARNING", &SmartErrorCategory::InvalidInput),
            ErrorSeverity::Warning
        );
    }

    #[test]
    fn test_cause_inference() {
        let analyzer = SmartErrorAnalyzer::new();

        let causes = analyzer.infer_causes("INVALID_PARAMETER");
        assert!(!causes.is_empty());
        assert!(causes.iter().any(|c| c.contains("validation")));
    }

    #[test]
    fn test_recovery_suggestions() {
        let analyzer = SmartErrorAnalyzer::new();

        let recovery = analyzer.suggest_recovery("INVALID_PARAM", &SmartErrorCategory::InvalidInput);
        assert!(!recovery.is_empty());
        assert!(recovery.iter().any(|r| r.contains("Validate")));
    }

    #[test]
    fn test_pattern_identification() {
        let analyzer = SmartErrorAnalyzer::new();

        let variants = vec![
            SmartErrorVariant {
                name: "INVALID_A".to_string(),
                ffi_value: Some(1),
                category: SmartErrorCategory::InvalidInput,
                severity: ErrorSeverity::Error,
                description: "Test".to_string(),
                causes: vec![],
                recovery: vec![],
                related: vec![],
            },
            SmartErrorVariant {
                name: "INVALID_B".to_string(),
                ffi_value: Some(2),
                category: SmartErrorCategory::InvalidInput,
                severity: ErrorSeverity::Error,
                description: "Test".to_string(),
                causes: vec![],
                recovery: vec![],
                related: vec![],
            },
        ];

        let patterns = analyzer.identify_patterns(&variants);
        assert!(!patterns.is_empty());
    }
}
