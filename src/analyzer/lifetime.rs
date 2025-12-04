//! Lifetime tracking for borrowed FFI resources
//!
//! This module analyzes resource dependencies to determine when wrapper types
//! need lifetime parameters to prevent use-after-free bugs.

use crate::ffi::{FfiFunction, FfiInfo};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use tracing::{debug, info};

/// Lifetime dependency information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LifetimeDependencies {
    /// Map from dependent type to types it borrows from
    pub dependencies: HashMap<String, Vec<BorrowedFrom>>,
    /// Types that own resources and don't need lifetimes
    pub owning_types: HashSet<String>,
    /// Types that borrow and need lifetimes
    pub borrowing_types: HashSet<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BorrowedFrom {
    /// The type this borrows from
    pub owner_type: String,
    /// Why we think this is a borrow relationship
    pub reason: BorrowReason,
    /// Suggested lifetime parameter name
    pub lifetime_name: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum BorrowReason {
    /// Function takes both types as parameters
    PassedTogether,
    /// Created from another handle
    CreatedFrom,
    /// Documentation indicates dependency
    DocumentedDependency,
    /// Function name pattern (e.g., "get", "borrow")
    FunctionNamePattern,
    /// Type name pattern (e.g., "View", "Ref")
    TypeNamePattern,
}

/// Analyzes lifetime dependencies between types
pub struct LifetimeAnalyzer;

impl LifetimeAnalyzer {
    pub fn new() -> Self {
        Self
    }

    /// Analyze lifetime dependencies from FFI info
    pub fn analyze(&self, ffi_info: &FfiInfo) -> LifetimeDependencies {
        let mut dependencies: HashMap<String, Vec<BorrowedFrom>> = HashMap::new();
        let mut owning_types = HashSet::new();
        let mut borrowing_types = HashSet::new();

        // First pass: identify owning vs borrowing types by name patterns
        for func in &ffi_info.functions {
            self.analyze_function_for_ownership(func, &mut owning_types, &mut borrowing_types);
        }

        // Second pass: detect dependencies
        for func in &ffi_info.functions {
            if let Some((dependent, owner, reason)) =
                self.detect_dependency(func, &owning_types, &borrowing_types)
            {
                dependencies
                    .entry(dependent.clone())
                    .or_insert_with(Vec::new)
                    .push(BorrowedFrom {
                        owner_type: owner.clone(),
                        reason,
                        lifetime_name: self.generate_lifetime_name(&owner),
                    });

                borrowing_types.insert(dependent);
            }
        }

        // Remove self-dependencies
        dependencies.retain(|k, v| {
            v.retain(|dep| dep.owner_type != *k);
            !v.is_empty()
        });

        info!(
            "Lifetime analysis: {} owning types, {} borrowing types, {} dependencies",
            owning_types.len(),
            borrowing_types.len(),
            dependencies.len()
        );

        LifetimeDependencies {
            dependencies,
            owning_types,
            borrowing_types,
        }
    }

    /// Analyze a function to classify types as owning or borrowing
    fn analyze_function_for_ownership(
        &self,
        func: &FfiFunction,
        owning_types: &mut HashSet<String>,
        borrowing_types: &mut HashSet<String>,
    ) {
        let name_lower = func.name.to_lowercase();

        // Functions that create owning resources
        if name_lower.contains("create")
            || name_lower.contains("alloc")
            || name_lower.contains("new")
            || name_lower.contains("init")
        {
            // Last parameter that's a pointer is likely the owning handle
            if let Some(last) = func.params.last() {
                if last.is_pointer {
                    let base_type = extract_base_type(&last.ty);
                    owning_types.insert(base_type);
                }
            }
        }

        // Functions that create borrowing resources
        if name_lower.contains("get")
            || name_lower.contains("borrow")
            || name_lower.contains("view")
            || name_lower.contains("ref")
        {
            if let Some(last) = func.params.last() {
                if last.is_pointer {
                    let base_type = extract_base_type(&last.ty);
                    borrowing_types.insert(base_type);
                }
            }
        }

        // Type name patterns
        for param in &func.params {
            let type_name = extract_base_type(&param.ty);
            let type_lower = type_name.to_lowercase();

            if type_lower.contains("view")
                || type_lower.contains("ref")
                || type_lower.contains("iterator")
                || type_lower.ends_with("iter")
            {
                borrowing_types.insert(type_name.clone());
            }

            if type_lower.contains("handle")
                || type_lower.contains("context")
                || type_lower.contains("manager")
            {
                owning_types.insert(type_name.clone());
            }
        }
    }

    /// Detect dependency between types in a function
    fn detect_dependency(
        &self,
        func: &FfiFunction,
        owning_types: &HashSet<String>,
        borrowing_types: &HashSet<String>,
    ) -> Option<(String, String, BorrowReason)> {
        // Look for functions that take multiple handle parameters
        let handle_params: Vec<_> = func
            .params
            .iter()
            .filter(|p| p.is_pointer)
            .map(|p| extract_base_type(&p.ty))
            .collect();

        if handle_params.len() >= 2 {
            // Check for owner + borrower pattern
            for (i, potential_borrower) in handle_params.iter().enumerate() {
                if borrowing_types.contains(potential_borrower) {
                    // Find an owning type in earlier parameters
                    for potential_owner in &handle_params[..i] {
                        if owning_types.contains(potential_owner) {
                            debug!(
                                "Detected dependency: {} borrows from {} in {}",
                                potential_borrower, potential_owner, func.name
                            );
                            return Some((
                                potential_borrower.clone(),
                                potential_owner.clone(),
                                BorrowReason::PassedTogether,
                            ));
                        }
                    }
                }
            }
        }

        // Check function name patterns for creation from another type
        let name_lower = func.name.to_lowercase();
        if (name_lower.contains("create") || name_lower.contains("get"))
            && handle_params.len() >= 2
        {
            // Last param is usually output, first is source
            if let (Some(source), Some(dest)) = (handle_params.first(), handle_params.last()) {
                if source != dest {
                    return Some((
                        dest.clone(),
                        source.clone(),
                        BorrowReason::CreatedFrom,
                    ));
                }
            }
        }

        // Check documentation for dependency hints
        if let Some(doc) = &func.docs {
            let doc_lower = doc.to_lowercase();
            if doc_lower.contains("valid until")
                || doc_lower.contains("lifetime")
                || doc_lower.contains("must outlive")
                || doc_lower.contains("depends on")
            {
                if handle_params.len() >= 2 {
                    return Some((
                        handle_params[handle_params.len() - 1].clone(),
                        handle_params[0].clone(),
                        BorrowReason::DocumentedDependency,
                    ));
                }
            }
        }

        None
    }

    /// Generate a lifetime parameter name from a type name
    fn generate_lifetime_name(&self, type_name: &str) -> String {
        let base = type_name
            .to_lowercase()
            .replace("handle", "")
            .replace("_t", "")
            .replace("_", "");

        // Take first letter or first few chars
        if base.is_empty() {
            "'a".to_string()
        } else if base.len() <= 3 {
            format!("'{}", base)
        } else {
            format!("'{}", &base[..1])
        }
    }
}

/// Extract base type name from a type string
fn extract_base_type(ty: &str) -> String {
    ty.replace("*mut", "")
        .replace("*const", "")
        .replace("*", "")
        .trim()
        .to_string()
}

/// Generate lifetime parameters for a wrapper type
pub fn generate_lifetime_params(
    type_name: &str,
    dependencies: &LifetimeDependencies,
) -> Option<String> {
    if let Some(deps) = dependencies.dependencies.get(type_name) {
        if deps.is_empty() {
            return None;
        }

        let lifetimes: Vec<String> = deps.iter().map(|d| d.lifetime_name.clone()).collect();

        // Deduplicate
        let unique: HashSet<_> = lifetimes.into_iter().collect();
        let mut sorted: Vec<_> = unique.into_iter().collect();
        sorted.sort();

        if sorted.is_empty() {
            None
        } else {
            Some(format!("<{}>", sorted.join(", ")))
        }
    } else {
        None
    }
}

/// Generate documentation about lifetime requirements
pub fn generate_lifetime_docs(
    type_name: &str,
    dependencies: &LifetimeDependencies,
) -> Option<String> {
    if let Some(deps) = dependencies.dependencies.get(type_name) {
        if deps.is_empty() {
            return None;
        }

        let mut doc = String::from("/// # Lifetime Requirements\n///\n");
        doc.push_str(&format!(
            "/// This type borrows from the following resources:\n"
        ));

        for dep in deps {
            doc.push_str(&format!(
                "/// - `{}` (lifetime `{}`): ",
                dep.owner_type, dep.lifetime_name
            ));

            match dep.reason {
                BorrowReason::PassedTogether => {
                    doc.push_str("Used together in API calls\n");
                }
                BorrowReason::CreatedFrom => {
                    doc.push_str("Created from this resource\n");
                }
                BorrowReason::DocumentedDependency => {
                    doc.push_str("Documented as dependent\n");
                }
                BorrowReason::FunctionNamePattern => {
                    doc.push_str("Inferred from function naming\n");
                }
                BorrowReason::TypeNamePattern => {
                    doc.push_str("Inferred from type naming\n");
                }
            }
        }

        doc.push_str("///\n");
        doc.push_str("/// The borrowed resources must outlive this type.\n");

        Some(doc)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ffi::{FfiFunction, FfiParam};

    #[test]
    fn test_extract_base_type() {
        assert_eq!(extract_base_type("*mut Handle"), "Handle");
        assert_eq!(extract_base_type("*const Context"), "Context");
        assert_eq!(extract_base_type("Handle"), "Handle");
    }

    #[test]
    fn test_lifetime_name_generation() {
        let analyzer = LifetimeAnalyzer::new();
        assert_eq!(analyzer.generate_lifetime_name("ContextHandle"), "'c");
        assert_eq!(analyzer.generate_lifetime_name("Stream_t"), "'s");
        assert_eq!(analyzer.generate_lifetime_name("X"), "'x");
    }

    #[test]
    fn test_ownership_detection() {
        let create_func = FfiFunction {
            name: "cudaStreamCreate".to_string(),
            params: vec![FfiParam {
                name: "stream".to_string(),
                ty: "*mut cudaStream_t".to_string(),
                is_pointer: true,
                is_mut: true,
            }],
            return_type: "cudaError_t".to_string(),
            docs: None,
        };

        let mut owning = HashSet::new();
        let mut borrowing = HashSet::new();

        let analyzer = LifetimeAnalyzer::new();
        analyzer.analyze_function_for_ownership(&create_func, &mut owning, &mut borrowing);

        assert!(owning.contains("cudaStream_t"));
    }

    #[test]
    fn test_dependency_detection() {
        let func = FfiFunction {
            name: "cudaStreamGetContext".to_string(),
            params: vec![
                FfiParam {
                    name: "context".to_string(),
                    ty: "*mut cudaContext_t".to_string(),
                    is_pointer: true,
                    is_mut: false,
                },
                FfiParam {
                    name: "stream".to_string(),
                    ty: "*mut cudaStreamView_t".to_string(),
                    is_pointer: true,
                    is_mut: true,
                },
            ],
            return_type: "cudaError_t".to_string(),
            docs: None,
        };

        let mut owning = HashSet::new();
        owning.insert("cudaContext_t".to_string());

        let mut borrowing = HashSet::new();
        borrowing.insert("cudaStreamView_t".to_string());

        let analyzer = LifetimeAnalyzer::new();
        let dep = analyzer.detect_dependency(&func, &owning, &borrowing);

        assert!(dep.is_some());
        let (borrower, owner, _reason) = dep.unwrap();
        assert_eq!(borrower, "cudaStreamView_t");
        assert_eq!(owner, "cudaContext_t");
    }

    #[test]
    fn test_lifetime_params_generation() {
        let mut deps = LifetimeDependencies {
            dependencies: HashMap::new(),
            owning_types: HashSet::new(),
            borrowing_types: HashSet::new(),
        };

        deps.dependencies.insert(
            "StreamView".to_string(),
            vec![BorrowedFrom {
                owner_type: "Context".to_string(),
                reason: BorrowReason::CreatedFrom,
                lifetime_name: "'ctx".to_string(),
            }],
        );

        let params = generate_lifetime_params("StreamView", &deps);
        assert_eq!(params, Some("<'ctx>".to_string()));
    }

    #[test]
    fn test_multiple_lifetimes() {
        let mut deps = LifetimeDependencies {
            dependencies: HashMap::new(),
            owning_types: HashSet::new(),
            borrowing_types: HashSet::new(),
        };

        deps.dependencies.insert(
            "ComplexView".to_string(),
            vec![
                BorrowedFrom {
                    owner_type: "Context".to_string(),
                    reason: BorrowReason::PassedTogether,
                    lifetime_name: "'ctx".to_string(),
                },
                BorrowedFrom {
                    owner_type: "Stream".to_string(),
                    reason: BorrowReason::PassedTogether,
                    lifetime_name: "'s".to_string(),
                },
            ],
        );

        let params = generate_lifetime_params("ComplexView", &deps);
        assert!(params.is_some());
        let params_str = params.unwrap();
        assert!(params_str.contains("'ctx"));
        assert!(params_str.contains("'s"));
    }

    #[test]
    fn test_lifetime_docs_generation() {
        let mut deps = LifetimeDependencies {
            dependencies: HashMap::new(),
            owning_types: HashSet::new(),
            borrowing_types: HashSet::new(),
        };

        deps.dependencies.insert(
            "View".to_string(),
            vec![BorrowedFrom {
                owner_type: "Context".to_string(),
                reason: BorrowReason::DocumentedDependency,
                lifetime_name: "'c".to_string(),
            }],
        );

        let docs = generate_lifetime_docs("View", &deps);
        assert!(docs.is_some());
        let docs_str = docs.unwrap();
        assert!(docs_str.contains("Lifetime Requirements"));
        assert!(docs_str.contains("Context"));
        assert!(docs_str.contains("'c"));
    }
}
