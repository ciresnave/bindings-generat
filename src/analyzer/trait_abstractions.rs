//! Trait-based abstraction generation for common patterns
//!
//! This module detects common patterns across types and generates
//! trait definitions for generic programming over FFI types.

use crate::ffi::FfiInfo;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use tracing::info;

/// Detected trait abstractions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraitAbstractions {
    /// Common traits that can be implemented
    pub traits: Vec<TraitDefinition>,
    /// Map from type name to traits it can implement
    pub type_traits: HashMap<String, Vec<String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraitDefinition {
    /// Name of the trait
    pub name: String,
    /// Description of what the trait represents
    pub description: String,
    /// Methods in the trait
    pub methods: Vec<TraitMethod>,
    /// Types that implement this trait
    pub implementors: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraitMethod {
    /// Method name
    pub name: String,
    /// Method signature
    pub signature: String,
    /// What the method does
    pub description: String,
}

/// Analyzes FFI for trait opportunities
pub struct TraitAnalyzer;

impl TraitAnalyzer {
    pub fn new() -> Self {
        Self
    }

    /// Analyze FFI for common patterns that could be traits
    pub fn analyze(&self, ffi_info: &FfiInfo) -> TraitAbstractions {
        let mut traits = Vec::new();
        let mut type_traits: HashMap<String, Vec<String>> = HashMap::new();

        // Detect Stream-like traits
        if let Some(stream_trait) = self.detect_stream_trait(ffi_info) {
            for impl_type in &stream_trait.implementors {
                type_traits
                    .entry(impl_type.clone())
                    .or_insert_with(Vec::new)
                    .push(stream_trait.name.clone());
            }
            traits.push(stream_trait);
        }

        // Detect Resource traits (create/destroy)
        if let Some(resource_trait) = self.detect_resource_trait(ffi_info) {
            for impl_type in &resource_trait.implementors {
                type_traits
                    .entry(impl_type.clone())
                    .or_insert_with(Vec::new)
                    .push(resource_trait.name.clone());
            }
            traits.push(resource_trait);
        }

        // Detect Descriptor/Configuration traits
        if let Some(descriptor_trait) = self.detect_descriptor_trait(ffi_info) {
            for impl_type in &descriptor_trait.implementors {
                type_traits
                    .entry(impl_type.clone())
                    .or_insert_with(Vec::new)
                    .push(descriptor_trait.name.clone());
            }
            traits.push(descriptor_trait);
        }

        // Detect Tensor/Buffer traits
        if let Some(tensor_trait) = self.detect_tensor_trait(ffi_info) {
            for impl_type in &tensor_trait.implementors {
                type_traits
                    .entry(impl_type.clone())
                    .or_insert_with(Vec::new)
                    .push(tensor_trait.name.clone());
            }
            traits.push(tensor_trait);
        }

        info!(
            "Trait analysis: {} traits detected, {} types with traits",
            traits.len(),
            type_traits.len()
        );

        TraitAbstractions {
            traits,
            type_traits,
        }
    }

    /// Detect stream-like abstractions
    fn detect_stream_trait(&self, ffi_info: &FfiInfo) -> Option<TraitDefinition> {
        let mut stream_types = HashSet::new();

        // Look for types with stream-like operations
        for func in &ffi_info.functions {
            let name_lower = func.name.to_lowercase();

            if name_lower.contains("stream") {
                // Extract the type from function parameters
                for param in &func.params {
                    if param.ty.to_lowercase().contains("stream") {
                        let base_type = extract_base_type(&param.ty);
                        stream_types.insert(base_type);
                    }
                }
            }
        }

        if stream_types.is_empty() {
            return None;
        }

        Some(TraitDefinition {
            name: "Stream".to_string(),
            description: "Types that represent execution streams".to_string(),
            methods: vec![
                TraitMethod {
                    name: "synchronize".to_string(),
                    signature: "fn synchronize(&self) -> Result<(), Error>".to_string(),
                    description: "Wait for stream operations to complete".to_string(),
                },
                TraitMethod {
                    name: "query".to_string(),
                    signature: "fn query(&self) -> Result<bool, Error>".to_string(),
                    description: "Check if stream has completed".to_string(),
                },
            ],
            implementors: stream_types.into_iter().collect(),
        })
    }

    /// Detect resource management traits
    fn detect_resource_trait(&self, ffi_info: &FfiInfo) -> Option<TraitDefinition> {
        let mut resource_types = HashSet::new();

        // Look for create/destroy pairs
        for func in &ffi_info.functions {
            let name_lower = func.name.to_lowercase();

            if name_lower.contains("create") || name_lower.contains("init") {
                if let Some(last_param) = func.params.last() {
                    if last_param.is_pointer {
                        let base_type = extract_base_type(&last_param.ty);
                        resource_types.insert(base_type);
                    }
                }
            }
        }

        if resource_types.is_empty() {
            return None;
        }

        Some(TraitDefinition {
            name: "Resource".to_string(),
            description: "Types that manage system resources with RAII".to_string(),
            methods: vec![TraitMethod {
                name: "as_raw".to_string(),
                signature: "fn as_raw(&self) -> Self::Handle".to_string(),
                description: "Get the raw FFI handle".to_string(),
            }],
            implementors: resource_types.into_iter().collect(),
        })
    }

    /// Detect descriptor/configuration traits
    fn detect_descriptor_trait(&self, ffi_info: &FfiInfo) -> Option<TraitDefinition> {
        let mut descriptor_types = HashSet::new();

        for func in &ffi_info.functions {
            for param in &func.params {
                let type_lower = param.ty.to_lowercase();

                if type_lower.contains("descriptor")
                    || type_lower.contains("desc")
                    || type_lower.contains("config")
                {
                    let base_type = extract_base_type(&param.ty);
                    descriptor_types.insert(base_type);
                }
            }
        }

        if descriptor_types.is_empty() {
            return None;
        }

        Some(TraitDefinition {
            name: "Descriptor".to_string(),
            description: "Configuration and descriptor types".to_string(),
            methods: vec![
                TraitMethod {
                    name: "set".to_string(),
                    signature: "fn set(&mut self, ...) -> Result<(), Error>".to_string(),
                    description: "Configure descriptor properties".to_string(),
                },
                TraitMethod {
                    name: "get".to_string(),
                    signature: "fn get(&self) -> Result<Properties, Error>".to_string(),
                    description: "Query descriptor properties".to_string(),
                },
            ],
            implementors: descriptor_types.into_iter().collect(),
        })
    }

    /// Detect tensor/buffer traits
    fn detect_tensor_trait(&self, ffi_info: &FfiInfo) -> Option<TraitDefinition> {
        let mut tensor_types = HashSet::new();

        for func in &ffi_info.functions {
            for param in &func.params {
                let type_lower = param.ty.to_lowercase();

                if type_lower.contains("tensor")
                    || type_lower.contains("buffer")
                    || type_lower.contains("array")
                {
                    let base_type = extract_base_type(&param.ty);
                    tensor_types.insert(base_type);
                }
            }
        }

        if tensor_types.is_empty() {
            return None;
        }

        Some(TraitDefinition {
            name: "TensorLike".to_string(),
            description: "Multi-dimensional data containers".to_string(),
            methods: vec![
                TraitMethod {
                    name: "shape".to_string(),
                    signature: "fn shape(&self) -> &[usize]".to_string(),
                    description: "Get tensor dimensions".to_string(),
                },
                TraitMethod {
                    name: "dtype".to_string(),
                    signature: "fn dtype(&self) -> DataType".to_string(),
                    description: "Get element data type".to_string(),
                },
            ],
            implementors: tensor_types.into_iter().collect(),
        })
    }
}

/// Extract base type from pointer type
fn extract_base_type(ty: &str) -> String {
    ty.replace("*mut", "")
        .replace("*const", "")
        .replace("*", "")
        .trim()
        .to_string()
}

/// Generate trait definition code
pub fn generate_trait_code(trait_def: &TraitDefinition) -> String {
    let mut output = String::new();

    output.push_str(&format!("/// {}\n", trait_def.description));
    output.push_str(&format!("pub trait {} {{\n", trait_def.name));

    for method in &trait_def.methods {
        output.push_str(&format!("    /// {}\n", method.description));
        output.push_str(&format!("    {};\n", method.signature));
    }

    output.push_str("}\n\n");

    // Generate implementations for each type
    for impl_type in &trait_def.implementors {
        output.push_str(&format!("impl {} for {} {{\n", trait_def.name, impl_type));
        output.push_str("    // TODO: Implement trait methods\n");
        output.push_str("}\n\n");
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ffi::{FfiFunction, FfiParam};

    #[test]
    fn test_stream_trait_detection() {
        let mut ffi_info = FfiInfo::default();
        ffi_info.functions.push(FfiFunction {
            name: "cudaStreamSynchronize".to_string(),
            params: vec![FfiParam {
                name: "stream".to_string(),
                ty: "*mut cudaStream_t".to_string(),
                is_pointer: true,
                is_mut: true,
            }],
            return_type: "cudaError_t".to_string(),
            docs: None,
        });

        let analyzer = TraitAnalyzer::new();
        let abstractions = analyzer.analyze(&ffi_info);

        assert!(!abstractions.traits.is_empty());
        let stream_trait = abstractions.traits.iter().find(|t| t.name == "Stream");
        assert!(stream_trait.is_some());
    }

    #[test]
    fn test_resource_trait_detection() {
        let mut ffi_info = FfiInfo::default();
        ffi_info.functions.push(FfiFunction {
            name: "cudaContextCreate".to_string(),
            params: vec![FfiParam {
                name: "ctx".to_string(),
                ty: "*mut cudaContext_t".to_string(),
                is_pointer: true,
                is_mut: true,
            }],
            return_type: "cudaError_t".to_string(),
            docs: None,
        });

        let analyzer = TraitAnalyzer::new();
        let abstractions = analyzer.analyze(&ffi_info);

        let resource_trait = abstractions.traits.iter().find(|t| t.name == "Resource");
        assert!(resource_trait.is_some());
    }

    #[test]
    fn test_descriptor_trait_detection() {
        let mut ffi_info = FfiInfo::default();
        ffi_info.functions.push(FfiFunction {
            name: "cudnnSetTensorDescriptor".to_string(),
            params: vec![FfiParam {
                name: "desc".to_string(),
                ty: "*mut cudnnTensorDescriptor_t".to_string(),
                is_pointer: true,
                is_mut: true,
            }],
            return_type: "cudnnStatus_t".to_string(),
            docs: None,
        });

        let analyzer = TraitAnalyzer::new();
        let abstractions = analyzer.analyze(&ffi_info);

        let desc_trait = abstractions.traits.iter().find(|t| t.name == "Descriptor");
        assert!(desc_trait.is_some());
    }

    #[test]
    fn test_tensor_trait_detection() {
        let mut ffi_info = FfiInfo::default();
        ffi_info.functions.push(FfiFunction {
            name: "getTensorShape".to_string(),
            params: vec![FfiParam {
                name: "tensor".to_string(),
                ty: "*mut Tensor_t".to_string(),
                is_pointer: true,
                is_mut: false,
            }],
            return_type: "int".to_string(),
            docs: None,
        });

        let analyzer = TraitAnalyzer::new();
        let abstractions = analyzer.analyze(&ffi_info);

        let tensor_trait = abstractions.traits.iter().find(|t| t.name == "TensorLike");
        assert!(tensor_trait.is_some());
    }

    #[test]
    fn test_trait_code_generation() {
        let trait_def = TraitDefinition {
            name: "TestTrait".to_string(),
            description: "Test trait".to_string(),
            methods: vec![TraitMethod {
                name: "test_method".to_string(),
                signature: "fn test_method(&self)".to_string(),
                description: "Test method".to_string(),
            }],
            implementors: vec!["TestType".to_string()],
        };

        let code = generate_trait_code(&trait_def);
        assert!(code.contains("pub trait TestTrait"));
        assert!(code.contains("fn test_method"));
        assert!(code.contains("impl TestTrait for TestType"));
    }
}
