//! Semantic code analysis for understanding API relationships
//!
//! This module performs deeper semantic analysis of the codebase
//! to understand relationships, dependencies, and usage patterns.

use crate::ffi::FfiInfo;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use tracing::info;

/// Semantic analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticAnalysis {
    /// Module organization
    pub modules: Vec<ModuleInfo>,
    /// Type relationships
    pub type_relationships: HashMap<String, TypeRelationship>,
    /// Function clusters by functionality
    pub function_clusters: Vec<FunctionCluster>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModuleInfo {
    /// Module name
    pub name: String,
    /// Module description
    pub description: String,
    /// Types in this module
    pub types: Vec<String>,
    /// Functions in this module
    pub functions: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypeRelationship {
    /// Type name
    pub name: String,
    /// Types this depends on
    pub depends_on: Vec<String>,
    /// Types that depend on this
    pub depended_by: Vec<String>,
    /// Relationship strength (0.0-1.0)
    pub strength: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionCluster {
    /// Cluster name (e.g., "Memory Management", "Tensor Operations")
    pub name: String,
    /// Functions in this cluster
    pub functions: Vec<String>,
    /// Common parameters across functions
    pub common_params: Vec<String>,
}

/// Performs semantic analysis
pub struct SemanticAnalyzer;

impl SemanticAnalyzer {
    pub fn new() -> Self {
        Self
    }

    /// Perform semantic analysis on FFI
    pub fn analyze(&self, ffi_info: &FfiInfo) -> SemanticAnalysis {
        let modules = self.organize_into_modules(ffi_info);
        let type_relationships = self.analyze_type_relationships(ffi_info);
        let function_clusters = self.cluster_functions(ffi_info);

        info!(
            "Semantic analysis: {} modules, {} type relationships, {} function clusters",
            modules.len(),
            type_relationships.len(),
            function_clusters.len()
        );

        SemanticAnalysis {
            modules,
            type_relationships,
            function_clusters,
        }
    }

    /// Organize functions and types into logical modules
    fn organize_into_modules(&self, ffi_info: &FfiInfo) -> Vec<ModuleInfo> {
        let mut modules: HashMap<String, ModuleInfo> = HashMap::new();

        // Group by function name prefixes
        for func in &ffi_info.functions {
            let module_name = self.infer_module_name(&func.name);

            modules
                .entry(module_name.clone())
                .or_insert_with(|| ModuleInfo {
                    name: module_name.clone(),
                    description: format!("{} operations", module_name),
                    types: Vec::new(),
                    functions: Vec::new(),
                })
                .functions
                .push(func.name.clone());
        }

        // Assign types to modules based on usage
        for ty in &ffi_info.types {
            let mut best_module = "core".to_string();
            let mut max_count = 0;

            for (module_name, module) in &modules {
                let count = module
                    .functions
                    .iter()
                    .filter(|f| {
                        ffi_info
                            .functions
                            .iter()
                            .find(|ff| &ff.name == *f)
                            .map(|ff| {
                                ff.params.iter().any(|p| p.ty.contains(&ty.name))
                                    || ff.return_type.contains(&ty.name)
                            })
                            .unwrap_or(false)
                    })
                    .count();

                if count > max_count {
                    max_count = count;
                    best_module = module_name.clone();
                }
            }

            if let Some(module) = modules.get_mut(&best_module) {
                module.types.push(ty.name.clone());
            }
        }

        modules.into_values().collect()
    }

    /// Infer module name from function name
    fn infer_module_name(&self, func_name: &str) -> String {
        let name_lower = func_name.to_lowercase();

        // Common module patterns
        if name_lower.contains("mem")
            || name_lower.contains("malloc")
            || name_lower.contains("free")
        {
            return "memory".to_string();
        }
        if name_lower.contains("tensor") {
            return "tensor".to_string();
        }
        if name_lower.contains("stream") {
            return "stream".to_string();
        }
        if name_lower.contains("event") {
            return "event".to_string();
        }
        if name_lower.contains("device") || name_lower.contains("gpu") {
            return "device".to_string();
        }
        if name_lower.contains("graph") {
            return "graph".to_string();
        }

        // Extract prefix
        if let Some(pos) = func_name.find(|c: char| c.is_uppercase() && !func_name.starts_with(c)) {
            return func_name[..pos].to_lowercase();
        }

        "core".to_string()
    }

    /// Analyze relationships between types
    fn analyze_type_relationships(&self, ffi_info: &FfiInfo) -> HashMap<String, TypeRelationship> {
        let mut relationships: HashMap<String, TypeRelationship> = HashMap::new();

        for ty in &ffi_info.types {
            let mut depends_on = HashSet::new();

            // Check field types
            for field in &ty.fields {
                if let Some(other_type) = ffi_info.types.iter().find(|t| field.ty.contains(&t.name))
                {
                    depends_on.insert(other_type.name.clone());
                }
            }

            relationships.insert(
                ty.name.clone(),
                TypeRelationship {
                    name: ty.name.clone(),
                    depends_on: depends_on.iter().cloned().collect(),
                    depended_by: Vec::new(),
                    strength: 1.0,
                },
            );
        }

        // Fill in reverse dependencies
        let deps_clone = relationships.clone();
        for (type_name, rel) in &mut relationships {
            for (other_name, other_rel) in &deps_clone {
                if other_rel.depends_on.contains(type_name) {
                    rel.depended_by.push(other_name.clone());
                }
            }
        }

        relationships
    }

    /// Cluster functions by common patterns
    fn cluster_functions(&self, ffi_info: &FfiInfo) -> Vec<FunctionCluster> {
        let mut clusters: HashMap<String, Vec<String>> = HashMap::new();

        for func in &ffi_info.functions {
            let cluster_name = self.determine_cluster(&func.name);
            clusters
                .entry(cluster_name)
                .or_insert_with(Vec::new)
                .push(func.name.clone());
        }

        clusters
            .into_iter()
            .map(|(name, functions)| FunctionCluster {
                name,
                common_params: Vec::new(),
                functions,
            })
            .collect()
    }

    /// Determine which cluster a function belongs to
    fn determine_cluster(&self, func_name: &str) -> String {
        let name_lower = func_name.to_lowercase();

        if name_lower.contains("create")
            || name_lower.contains("destroy")
            || name_lower.contains("init")
            || name_lower.contains("cleanup")
        {
            return "Resource Management".to_string();
        }
        if name_lower.contains("get") || name_lower.contains("set") || name_lower.contains("query")
        {
            return "Property Access".to_string();
        }
        if name_lower.contains("copy")
            || name_lower.contains("memcpy")
            || name_lower.contains("transfer")
        {
            return "Data Transfer".to_string();
        }
        if name_lower.contains("sync") || name_lower.contains("wait") || name_lower.contains("poll")
        {
            return "Synchronization".to_string();
        }

        "General Operations".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ffi::{FfiFunction, FfiType};

    #[test]
    fn test_module_organization() {
        let mut ffi_info = FfiInfo::default();
        ffi_info.functions.push(FfiFunction {
            name: "cudaMemcpy".to_string(),
            params: vec![],
            return_type: "int".to_string(),
            docs: None,
        });

        let analyzer = SemanticAnalyzer::new();
        let analysis = analyzer.analyze(&ffi_info);

        assert!(!analysis.modules.is_empty());
    }

    #[test]
    fn test_module_inference() {
        let analyzer = SemanticAnalyzer::new();

        assert_eq!(analyzer.infer_module_name("cudaMemcpy"), "memory");
        assert_eq!(analyzer.infer_module_name("tensorCreate"), "tensor");
        assert_eq!(analyzer.infer_module_name("streamSync"), "stream");
    }

    #[test]
    fn test_function_clustering() {
        let mut ffi_info = FfiInfo::default();
        ffi_info.functions.push(FfiFunction {
            name: "createHandle".to_string(),
            params: vec![],
            return_type: "int".to_string(),
            docs: None,
        });
        ffi_info.functions.push(FfiFunction {
            name: "destroyHandle".to_string(),
            params: vec![],
            return_type: "int".to_string(),
            docs: None,
        });

        let analyzer = SemanticAnalyzer::new();
        let analysis = analyzer.analyze(&ffi_info);

        assert!(!analysis.function_clusters.is_empty());
        assert!(
            analysis
                .function_clusters
                .iter()
                .any(|c| c.name == "Resource Management")
        );
    }

    #[test]
    fn test_type_relationships() {
        let mut ffi_info = FfiInfo::default();
        ffi_info.types.push(FfiType {
            name: "TypeA".to_string(),
            is_opaque: false,
            docs: None,
            fields: vec![],
        });

        let analyzer = SemanticAnalyzer::new();
        let analysis = analyzer.analyze(&ffi_info);

        assert!(analysis.type_relationships.contains_key("TypeA"));
    }
}
