//! Cross-reference generation for documentation links
//!
//! This module generates comprehensive cross-reference documentation
//! to help users navigate between related functions, types, and modules.

use crate::ffi::{FfiFunction, FfiInfo, FfiType};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use tracing::info;

/// Cross-reference analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossReferences {
    /// Function cross-references
    pub function_refs: HashMap<String, FunctionRefs>,
    /// Type cross-references
    pub type_refs: HashMap<String, TypeRefs>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionRefs {
    /// Function name
    pub function: String,
    /// Functions that call this
    pub called_by: Vec<String>,
    /// Functions this calls
    pub calls: Vec<String>,
    /// Similar functions by signature
    pub similar: Vec<String>,
    /// Related types used
    pub uses_types: Vec<String>,
    /// See also suggestions
    pub see_also: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypeRefs {
    /// Type name
    pub type_name: String,
    /// Functions that use this type
    pub used_by_functions: Vec<String>,
    /// Related types
    pub related_types: Vec<String>,
    /// Parent type (if any)
    pub parent: Option<String>,
    /// Child types (if any)
    pub children: Vec<String>,
}

/// Performs cross-reference analysis
pub struct CrossReferenceAnalyzer;

impl CrossReferenceAnalyzer {
    pub fn new() -> Self {
        Self
    }

    /// Generate cross-references
    pub fn analyze(&self, ffi_info: &FfiInfo) -> CrossReferences {
        let function_refs = self.analyze_function_refs(ffi_info);
        let type_refs = self.analyze_type_refs(ffi_info);

        info!(
            "Generated {} function refs, {} type refs",
            function_refs.len(),
            type_refs.len()
        );

        CrossReferences {
            function_refs,
            type_refs,
        }
    }

    /// Analyze function cross-references
    fn analyze_function_refs(&self, ffi_info: &FfiInfo) -> HashMap<String, FunctionRefs> {
        let mut refs = HashMap::new();

        for func in &ffi_info.functions {
            let uses_types = self.extract_function_types(func, ffi_info);
            let similar = self.find_similar_functions(func, ffi_info);
            let see_also = self.generate_see_also(func, ffi_info);

            refs.insert(
                func.name.clone(),
                FunctionRefs {
                    function: func.name.clone(),
                    called_by: Vec::new(), // Would need call graph analysis
                    calls: Vec::new(),      // Would need call graph analysis
                    similar,
                    uses_types,
                    see_also,
                },
            );
        }

        refs
    }

    /// Extract types used by function
    fn extract_function_types(&self, func: &FfiFunction, ffi_info: &FfiInfo) -> Vec<String> {
        let mut types = HashSet::new();

        // Check parameters
        for param in &func.params {
            if let Some(ty) = ffi_info.types.iter().find(|t| param.ty.contains(&t.name)) {
                types.insert(ty.name.clone());
            }
        }

        // Check return type
        if let Some(ty) = ffi_info
            .types
            .iter()
            .find(|t| func.return_type.contains(&t.name))
        {
            types.insert(ty.name.clone());
        }

        types.into_iter().collect()
    }

    /// Find similar functions by signature
    fn find_similar_functions(&self, func: &FfiFunction, ffi_info: &FfiInfo) -> Vec<String> {
        let mut similar = Vec::new();

        for other in &ffi_info.functions {
            if other.name == func.name {
                continue;
            }

            let similarity = self.calculate_similarity(func, other);
            if similarity > 0.5 {
                similar.push(other.name.clone());
            }
        }

        similar
    }

    /// Calculate similarity between functions
    fn calculate_similarity(&self, func1: &FfiFunction, func2: &FfiFunction) -> f64 {
        let mut score = 0.0;

        // Name similarity
        let name1 = func1.name.to_lowercase();
        let name2 = func2.name.to_lowercase();
        if name1.contains(&name2[..name2.len().min(5)])
            || name2.contains(&name1[..name1.len().min(5)])
        {
            score += 0.3;
        }

        // Return type similarity
        if func1.return_type == func2.return_type {
            score += 0.3;
        }

        // Parameter count similarity
        if func1.params.len() == func2.params.len() {
            score += 0.4;
        }

        score
    }

    /// Generate see-also suggestions
    fn generate_see_also(&self, func: &FfiFunction, ffi_info: &FfiInfo) -> Vec<String> {
        let mut see_also = Vec::new();
        let name_lower = func.name.to_lowercase();

        // If this is a create function, suggest destroy
        if name_lower.contains("create") {
            let destroy_name = func.name.replace("create", "destroy").replace("Create", "Destroy");
            if ffi_info.functions.iter().any(|f| f.name == destroy_name) {
                see_also.push(destroy_name);
            }
        }

        // If this is a destroy function, suggest create
        if name_lower.contains("destroy") {
            let create_name = func.name.replace("destroy", "create").replace("Destroy", "Create");
            if ffi_info.functions.iter().any(|f| f.name == create_name) {
                see_also.push(create_name);
            }
        }

        // If this is a get function, suggest set
        if name_lower.contains("get") {
            let set_name = func.name.replace("get", "set").replace("Get", "Set");
            if ffi_info.functions.iter().any(|f| f.name == set_name) {
                see_also.push(set_name);
            }
        }

        see_also
    }

    /// Analyze type cross-references
    fn analyze_type_refs(&self, ffi_info: &FfiInfo) -> HashMap<String, TypeRefs> {
        let mut refs = HashMap::new();

        for ty in &ffi_info.types {
            let used_by_functions = self.find_functions_using_type(ty, ffi_info);
            let related_types = self.find_related_types(ty, ffi_info);

            refs.insert(
                ty.name.clone(),
                TypeRefs {
                    type_name: ty.name.clone(),
                    used_by_functions,
                    related_types,
                    parent: None,
                    children: Vec::new(),
                },
            );
        }

        refs
    }

    /// Find functions that use a type
    fn find_functions_using_type(&self, ty: &FfiType, ffi_info: &FfiInfo) -> Vec<String> {
        ffi_info
            .functions
            .iter()
            .filter(|f| {
                f.params.iter().any(|p| p.ty.contains(&ty.name))
                    || f.return_type.contains(&ty.name)
            })
            .map(|f| f.name.clone())
            .collect()
    }

    /// Find related types
    fn find_related_types(&self, ty: &FfiType, ffi_info: &FfiInfo) -> Vec<String> {
        let mut related = HashSet::new();

        // Types used in fields
        for field in &ty.fields {
            if let Some(other_ty) = ffi_info.types.iter().find(|t| field.ty.contains(&t.name)) {
                related.insert(other_ty.name.clone());
            }
        }

        related.into_iter().collect()
    }

    /// Generate documentation with cross-references
    pub fn generate_docs(&self, refs: &CrossReferences) -> String {
        let mut docs = String::new();

        docs.push_str("# Cross-References\n\n");

        // Function references
        for (name, func_ref) in &refs.function_refs {
            if !func_ref.see_also.is_empty() || !func_ref.similar.is_empty() {
                docs.push_str(&format!("## {}\n\n", name));

                if !func_ref.uses_types.is_empty() {
                    docs.push_str("**Uses types:** ");
                    docs.push_str(&func_ref.uses_types.join(", "));
                    docs.push_str("\n\n");
                }

                if !func_ref.see_also.is_empty() {
                    docs.push_str("**See also:** ");
                    docs.push_str(&func_ref.see_also.join(", "));
                    docs.push_str("\n\n");
                }

                if !func_ref.similar.is_empty() {
                    docs.push_str("**Similar:** ");
                    docs.push_str(&func_ref.similar.join(", "));
                    docs.push_str("\n\n");
                }
            }
        }

        docs
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ffi::FfiParam;

    #[test]
    fn test_function_cross_refs() {
        let mut ffi_info = FfiInfo::default();
        ffi_info.functions.push(FfiFunction {
            name: "createHandle".to_string(),
            params: vec![],
            return_type: "Handle*".to_string(),
            docs: None,
        });
        ffi_info.functions.push(FfiFunction {
            name: "destroyHandle".to_string(),
            params: vec![],
            return_type: "void".to_string(),
            docs: None,
        });

        let analyzer = CrossReferenceAnalyzer::new();
        let refs = analyzer.analyze(&ffi_info);

        assert_eq!(refs.function_refs.len(), 2);
        let create_refs = &refs.function_refs["createHandle"];
        assert!(create_refs.see_also.contains(&"destroyHandle".to_string()));
    }

    #[test]
    fn test_type_cross_refs() {
        let mut ffi_info = FfiInfo::default();
        ffi_info.types.push(FfiType {
            name: "MyType".to_string(),
            is_opaque: false,
            docs: None,
            fields: vec![],
        });
        ffi_info.functions.push(FfiFunction {
            name: "useType".to_string(),
            params: vec![FfiParam {
                name: "arg".to_string(),
                ty: "MyType*".to_string(),
                is_mut: false,
                is_pointer: true,
            }],
            return_type: "void".to_string(),
            docs: None,
        });

        let analyzer = CrossReferenceAnalyzer::new();
        let refs = analyzer.analyze(&ffi_info);

        assert!(refs.type_refs.contains_key("MyType"));
        let type_refs = &refs.type_refs["MyType"];
        assert!(type_refs.used_by_functions.contains(&"useType".to_string()));
    }

    #[test]
    fn test_similarity_calculation() {
        let func1 = FfiFunction {
            name: "createTensor".to_string(),
            params: vec![],
            return_type: "Tensor*".to_string(),
            docs: None,
        };
        let func2 = FfiFunction {
            name: "createMatrix".to_string(),
            params: vec![],
            return_type: "Matrix*".to_string(),
            docs: None,
        };

        let analyzer = CrossReferenceAnalyzer::new();
        let similarity = analyzer.calculate_similarity(&func1, &func2);

        assert!(similarity > 0.0);
    }

    #[test]
    fn test_see_also_generation() {
        let mut ffi_info = FfiInfo::default();
        ffi_info.functions.push(FfiFunction {
            name: "getValue".to_string(),
            params: vec![],
            return_type: "int".to_string(),
            docs: None,
        });
        ffi_info.functions.push(FfiFunction {
            name: "setValue".to_string(),
            params: vec![],
            return_type: "void".to_string(),
            docs: None,
        });

        let analyzer = CrossReferenceAnalyzer::new();
        let refs = analyzer.analyze(&ffi_info);

        let get_refs = &refs.function_refs["getValue"];
        assert!(get_refs.see_also.contains(&"setValue".to_string()));
    }
}
