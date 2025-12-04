//! Error code documentation enrichment
//!
//! This module enriches error enums with detailed documentation
//! about each error code, common causes, and solutions.

use crate::ffi::{FfiEnum, FfiInfo};
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::info;

/// Enriched error documentation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorDocumentation {
    /// Map from error enum name to documentation
    pub error_docs: HashMap<String, ErrorEnumDoc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorEnumDoc {
    /// Enum name
    pub name: String,
    /// Overall enum description
    pub description: String,
    /// Documentation for each variant
    pub variants: HashMap<String, ErrorVariantDoc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorVariantDoc {
    /// Variant name
    pub name: String,
    /// Human-readable description
    pub description: String,
    /// Common causes
    pub causes: Vec<String>,
    /// Suggested solutions
    pub solutions: Vec<String>,
    /// Related errors
    pub related: Vec<String>,
}

/// Analyzes error code documentation
pub struct ErrorDocAnalyzer;

impl ErrorDocAnalyzer {
    pub fn new() -> Self {
        Self
    }

    /// Analyze error documentation
    pub fn analyze(&self, ffi_info: &FfiInfo) -> ErrorDocumentation {
        let mut error_docs = HashMap::new();

        for enum_def in &ffi_info.enums {
            if self.is_error_enum(enum_def) {
                let doc = self.enrich_error_enum(enum_def);
                error_docs.insert(enum_def.name.clone(), doc);
            }
        }

        info!(
            "Error documentation enrichment: {} error enums documented",
            error_docs.len()
        );

        ErrorDocumentation { error_docs }
    }

    /// Check if an enum represents errors
    fn is_error_enum(&self, enum_def: &FfiEnum) -> bool {
        let name_lower = enum_def.name.to_lowercase();
        name_lower.contains("error")
            || name_lower.contains("status")
            || name_lower.contains("result")
            || name_lower.contains("code")
    }

    /// Enrich error enum documentation
    fn enrich_error_enum(&self, enum_def: &FfiEnum) -> ErrorEnumDoc {
        let description = format!("Error codes for {}", enum_def.name);

        let mut variants = HashMap::new();
        for variant in &enum_def.variants {
            let doc = self.enrich_variant_doc(&variant.name, &variant.docs);
            variants.insert(variant.name.clone(), doc);
        }

        ErrorEnumDoc {
            name: enum_def.name.clone(),
            description,
            variants,
        }
    }

    /// Enrich a single error variant
    fn enrich_variant_doc(&self, name: &str, docs: &Option<String>) -> ErrorVariantDoc {
        let description = docs
            .as_ref()
            .and_then(|d| d.lines().next())
            .unwrap_or(name)
            .to_string();

        let causes = self.extract_causes(docs);
        let solutions = self.extract_solutions(docs);
        let related = self.find_related_errors(name);

        ErrorVariantDoc {
            name: name.to_string(),
            description,
            causes,
            solutions,
            related,
        }
    }

    /// Extract common causes from documentation
    fn extract_causes(&self, docs: &Option<String>) -> Vec<String> {
        let mut causes = Vec::new();

        if let Some(doc_text) = docs {
            let re = Regex::new(r"(?i)caused by:?\s*(.+)").ok();
            if let Some(regex) = re {
                for cap in regex.captures_iter(doc_text) {
                    if let Some(cause) = cap.get(1) {
                        causes.push(cause.as_str().trim().to_string());
                    }
                }
            }

            // Look for "when" patterns
            let when_re = Regex::new(r"(?i)when\s+(.+)").ok();
            if let Some(regex) = when_re {
                for cap in regex.captures_iter(doc_text) {
                    if let Some(cause) = cap.get(1) {
                        let cause_text = cause.as_str().trim();
                        if cause_text.len() < 100 {
                            causes.push(cause_text.to_string());
                        }
                    }
                }
            }
        }

        // Add generic causes based on error name
        if causes.is_empty() {
            causes.extend(self.infer_causes_from_name(docs.as_deref().unwrap_or("")));
        }

        causes
    }

    /// Extract solutions from documentation
    fn extract_solutions(&self, docs: &Option<String>) -> Vec<String> {
        let mut solutions = Vec::new();

        if let Some(doc_text) = docs {
            let re = Regex::new(r"(?i)(?:solution|fix|try|use):?\s*(.+)").ok();
            if let Some(regex) = re {
                for cap in regex.captures_iter(doc_text) {
                    if let Some(solution) = cap.get(1) {
                        solutions.push(solution.as_str().trim().to_string());
                    }
                }
            }
        }

        solutions
    }

    /// Infer causes from error name
    fn infer_causes_from_name(&self, name: &str) -> Vec<String> {
        let name_lower = name.to_lowercase();
        let mut causes = Vec::new();

        if name_lower.contains("null") {
            causes.push("A null pointer was passed".to_string());
        }
        if name_lower.contains("invalid") || name_lower.contains("bad") {
            causes.push("Invalid parameter value".to_string());
        }
        if name_lower.contains("not_found") || name_lower.contains("missing") {
            causes.push("Required resource not found".to_string());
        }
        if name_lower.contains("timeout") {
            causes.push("Operation exceeded time limit".to_string());
        }
        if name_lower.contains("memory") || name_lower.contains("alloc") {
            causes.push("Insufficient memory".to_string());
        }
        if name_lower.contains("permission") || name_lower.contains("access") {
            causes.push("Insufficient permissions".to_string());
        }

        causes
    }

    /// Find related error variants
    fn find_related_errors(&self, _name: &str) -> Vec<String> {
        // TODO: Implement similarity matching
        Vec::new()
    }
}

/// Generate documentation for error variants
pub fn generate_error_variant_docs(variant_doc: &ErrorVariantDoc) -> String {
    let mut output = String::new();

    output.push_str(&format!("    /// {}\n", variant_doc.description));

    if !variant_doc.causes.is_empty() {
        output.push_str("    ///\n");
        output.push_str("    /// **Common Causes:**\n");
        for cause in &variant_doc.causes {
            output.push_str(&format!("    /// - {}\n", cause));
        }
    }

    if !variant_doc.solutions.is_empty() {
        output.push_str("    ///\n");
        output.push_str("    /// **Solutions:**\n");
        for solution in &variant_doc.solutions {
            output.push_str(&format!("    /// - {}\n", solution));
        }
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ffi::{FfiEnum, FfiEnumVariant};

    #[test]
    fn test_error_enum_detection() {
        let error_enum = FfiEnum {
            name: "MyError".to_string(),
            variants: vec![],
            docs: None,
        };

        let analyzer = ErrorDocAnalyzer::new();
        assert!(analyzer.is_error_enum(&error_enum));

        let normal_enum = FfiEnum {
            name: "DataType".to_string(),
            variants: vec![],
            docs: None,
        };
        assert!(!analyzer.is_error_enum(&normal_enum));
    }

    #[test]
    fn test_cause_inference() {
        let analyzer = ErrorDocAnalyzer::new();

        let causes = analyzer.infer_causes_from_name("NULL_POINTER_ERROR");
        assert!(!causes.is_empty());
        assert!(causes.iter().any(|c| c.contains("null")));

        let memory_causes = analyzer.infer_causes_from_name("OUT_OF_MEMORY");
        assert!(memory_causes.iter().any(|c| c.contains("memory")));
    }

    #[test]
    fn test_error_enrichment() {
        let mut ffi_info = FfiInfo::default();

        ffi_info.enums.push(FfiEnum {
            name: "MyStatus".to_string(),
            variants: vec![
                FfiEnumVariant {
                    name: "SUCCESS".to_string(),
                    value: Some(0),
                    docs: Some("Operation succeeded".to_string()),
                },
                FfiEnumVariant {
                    name: "NULL_POINTER".to_string(),
                    value: Some(1),
                    docs: Some("Null pointer passed".to_string()),
                },
            ],
            docs: None,
        });

        let analyzer = ErrorDocAnalyzer::new();
        let docs = analyzer.analyze(&ffi_info);

        assert!(docs.error_docs.contains_key("MyStatus"));
        let error_doc = &docs.error_docs["MyStatus"];
        assert_eq!(error_doc.variants.len(), 2);
    }

    #[test]
    fn test_variant_doc_generation() {
        let variant_doc = ErrorVariantDoc {
            name: "INVALID_PARAMETER".to_string(),
            description: "Parameter validation failed".to_string(),
            causes: vec!["Invalid value".to_string()],
            solutions: vec!["Check parameter constraints".to_string()],
            related: vec![],
        };

        let docs = generate_error_variant_docs(&variant_doc);
        assert!(docs.contains("Parameter validation"));
        assert!(docs.contains("Common Causes"));
        assert!(docs.contains("Solutions"));
    }
}
