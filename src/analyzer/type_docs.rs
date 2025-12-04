//! Type documentation enrichment
//!
//! This module enriches type definitions with comprehensive documentation
//! extracted from headers, examples, and related functions.

use crate::ffi::{FfiInfo, FfiType};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::info;

/// Enriched type documentation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypeDocumentation {
    /// Map from type name to documentation
    pub type_docs: HashMap<String, TypeDoc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypeDoc {
    /// Type name
    pub name: String,
    /// Summary description
    pub summary: String,
    /// Detailed description
    pub details: Vec<String>,
    /// Usage examples
    pub examples: Vec<String>,
    /// Related types
    pub related_types: Vec<String>,
    /// Functions that use this type
    pub related_functions: Vec<String>,
    /// Common usage patterns
    pub patterns: Vec<String>,
    /// Safety considerations
    pub safety_notes: Vec<String>,
}

/// Analyzes and enriches type documentation
pub struct TypeDocAnalyzer;

impl TypeDocAnalyzer {
    pub fn new() -> Self {
        Self
    }

    /// Analyze and enrich type documentation
    pub fn analyze(&self, ffi_info: &FfiInfo) -> TypeDocumentation {
        let mut type_docs = HashMap::new();

        for ty in &ffi_info.types {
            let doc = self.enrich_type_doc(ty, ffi_info);
            type_docs.insert(ty.name.clone(), doc);
        }

        // Also document opaque types
        for opaque_name in &ffi_info.opaque_types {
            let doc = self.create_opaque_type_doc(opaque_name, ffi_info);
            type_docs.insert(opaque_name.clone(), doc);
        }

        info!(
            "Type documentation enrichment: {} types documented",
            type_docs.len()
        );

        TypeDocumentation { type_docs }
    }

    /// Enrich documentation for a specific type
    fn enrich_type_doc(&self, ty: &FfiType, ffi_info: &FfiInfo) -> TypeDoc {
        let summary = ty
            .docs
            .as_ref()
            .and_then(|d| d.lines().next())
            .unwrap_or(&format!("{} type", ty.name))
            .to_string();

        let details = ty
            .docs
            .as_ref()
            .map(|d| d.lines().skip(1).map(|s| s.to_string()).collect())
            .unwrap_or_else(Vec::new);

        let related_functions = self.find_related_functions(&ty.name, ffi_info);
        let patterns = self.extract_usage_patterns(&ty.name, ffi_info);
        let safety_notes = self.generate_safety_notes(ty, ffi_info);

        TypeDoc {
            name: ty.name.clone(),
            summary,
            details,
            examples: vec![],
            related_types: vec![],
            related_functions,
            patterns,
            safety_notes,
        }
    }

    /// Create documentation for opaque types
    fn create_opaque_type_doc(&self, name: &str, ffi_info: &FfiInfo) -> TypeDoc {
        let related_functions = self.find_related_functions(name, ffi_info);
        let patterns = self.extract_usage_patterns(name, ffi_info);

        TypeDoc {
            name: name.to_string(),
            summary: format!("Opaque handle type: {}", name),
            details: vec![
                "This is an opaque type representing an internal resource.".to_string(),
                "It should only be accessed through the provided API functions.".to_string(),
            ],
            examples: vec![],
            related_types: vec![],
            related_functions,
            patterns,
            safety_notes: vec![
                "Do not dereference this handle directly.".to_string(),
                "Always use the API functions to manipulate this resource.".to_string(),
            ],
        }
    }

    /// Find functions that use this type
    fn find_related_functions(&self, type_name: &str, ffi_info: &FfiInfo) -> Vec<String> {
        ffi_info
            .functions
            .iter()
            .filter(|f| {
                f.params.iter().any(|p| p.ty.contains(type_name))
                    || f.return_type.contains(type_name)
            })
            .map(|f| f.name.clone())
            .take(10) // Limit to 10 functions
            .collect()
    }

    /// Extract common usage patterns
    fn extract_usage_patterns(&self, type_name: &str, ffi_info: &FfiInfo) -> Vec<String> {
        let mut patterns = Vec::new();

        // Check for create/destroy pattern
        let has_create = ffi_info.functions.iter().any(|f| {
            let name_lower = f.name.to_lowercase();
            (name_lower.contains("create") || name_lower.contains("init"))
                && f.params.iter().any(|p| p.ty.contains(type_name))
        });

        let has_destroy = ffi_info.functions.iter().any(|f| {
            let name_lower = f.name.to_lowercase();
            (name_lower.contains("destroy") || name_lower.contains("free"))
                && f.params.iter().any(|p| p.ty.contains(type_name))
        });

        if has_create && has_destroy {
            patterns.push("Follows RAII pattern: create, use, destroy".to_string());
        }

        // Check for getter/setter pattern
        let has_get = ffi_info.functions.iter().any(|f| {
            f.name.to_lowercase().contains("get")
                && f.params.iter().any(|p| p.ty.contains(type_name))
        });

        let has_set = ffi_info.functions.iter().any(|f| {
            f.name.to_lowercase().contains("set")
                && f.params.iter().any(|p| p.ty.contains(type_name))
        });

        if has_get && has_set {
            patterns.push("Supports property get/set operations".to_string());
        }

        patterns
    }

    /// Generate safety notes for a type
    fn generate_safety_notes(&self, ty: &FfiType, _ffi_info: &FfiInfo) -> Vec<String> {
        let mut notes = Vec::new();

        if ty.is_opaque {
            notes.push("Opaque type - do not access fields directly".to_string());
        }

        if ty.fields.iter().any(|f| f.ty.contains("*")) {
            notes.push("Contains pointer fields - be careful with lifetime management".to_string());
        }

        notes
    }
}

/// Generate Rust doc comments from TypeDoc
pub fn generate_type_docs(type_doc: &TypeDoc) -> String {
    let mut output = String::new();

    output.push_str(&format!("/// {}\n", type_doc.summary));
    output.push_str("///\n");

    for detail in &type_doc.details {
        output.push_str(&format!("/// {}\n", detail));
    }

    if !type_doc.patterns.is_empty() {
        output.push_str("///\n");
        output.push_str("/// # Usage Patterns\n");
        for pattern in &type_doc.patterns {
            output.push_str(&format!("/// - {}\n", pattern));
        }
    }

    if !type_doc.safety_notes.is_empty() {
        output.push_str("///\n");
        output.push_str("/// # Safety\n");
        for note in &type_doc.safety_notes {
            output.push_str(&format!("/// - {}\n", note));
        }
    }

    if !type_doc.related_functions.is_empty() {
        output.push_str("///\n");
        output.push_str("/// # Related Functions\n");
        for func in type_doc.related_functions.iter().take(5) {
            output.push_str(&format!("/// - [`{}`]\n", func));
        }
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ffi::{FfiFunction, FfiParam, FfiType};

    #[test]
    fn test_type_doc_enrichment() {
        let mut ffi_info = FfiInfo::default();

        ffi_info.types.push(FfiType {
            name: "TestType".to_string(),
            is_opaque: false,
            docs: Some("Test type documentation".to_string()),
            fields: vec![],
        });

        ffi_info.functions.push(FfiFunction {
            name: "testTypeCreate".to_string(),
            params: vec![FfiParam {
                name: "ptr".to_string(),
                ty: "*mut TestType".to_string(),
                is_pointer: true,
                is_mut: true,
            }],
            return_type: "int".to_string(),
            docs: None,
        });

        let analyzer = TypeDocAnalyzer::new();
        let docs = analyzer.analyze(&ffi_info);

        assert!(docs.type_docs.contains_key("TestType"));
        let test_doc = &docs.type_docs["TestType"];
        assert_eq!(test_doc.name, "TestType");
        assert!(!test_doc.related_functions.is_empty());
    }

    #[test]
    fn test_opaque_type_doc() {
        let mut ffi_info = FfiInfo::default();
        ffi_info.opaque_types.push("OpaqueHandle".to_string());

        let analyzer = TypeDocAnalyzer::new();
        let docs = analyzer.analyze(&ffi_info);

        assert!(docs.type_docs.contains_key("OpaqueHandle"));
        let doc = &docs.type_docs["OpaqueHandle"];
        assert!(doc.summary.contains("Opaque"));
        assert!(!doc.safety_notes.is_empty());
    }

    #[test]
    fn test_pattern_detection() {
        let mut ffi_info = FfiInfo::default();

        ffi_info.functions.push(FfiFunction {
            name: "createHandle".to_string(),
            params: vec![FfiParam {
                name: "handle".to_string(),
                ty: "*mut Handle_t".to_string(),
                is_pointer: true,
                is_mut: true,
            }],
            return_type: "int".to_string(),
            docs: None,
        });

        ffi_info.functions.push(FfiFunction {
            name: "destroyHandle".to_string(),
            params: vec![FfiParam {
                name: "handle".to_string(),
                ty: "Handle_t".to_string(),
                is_pointer: false,
                is_mut: false,
            }],
            return_type: "int".to_string(),
            docs: None,
        });

        let analyzer = TypeDocAnalyzer::new();
        let patterns = analyzer.extract_usage_patterns("Handle_t", &ffi_info);

        assert!(!patterns.is_empty());
        assert!(patterns.iter().any(|p| p.contains("RAII")));
    }

    #[test]
    fn test_doc_generation() {
        let type_doc = TypeDoc {
            name: "TestType".to_string(),
            summary: "A test type".to_string(),
            details: vec!["More details here".to_string()],
            examples: vec![],
            related_types: vec![],
            related_functions: vec!["test_func".to_string()],
            patterns: vec!["Pattern 1".to_string()],
            safety_notes: vec!["Be careful".to_string()],
        };

        let docs = generate_type_docs(&type_doc);
        assert!(docs.contains("A test type"));
        assert!(docs.contains("Usage Patterns"));
        assert!(docs.contains("Safety"));
        assert!(docs.contains("test_func"));
    }
}
