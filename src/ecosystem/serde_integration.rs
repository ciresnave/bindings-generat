// src/ecosystem/serde_integration.rs

//! Serde integration for generated types

use crate::ffi::{FfiInfo, FfiType};

/// Generates serde implementations for serializable types
pub struct SerdeIntegration;

impl SerdeIntegration {
    pub fn new() -> Self {
        Self
    }

    /// Determine which types can be serialized
    pub fn detect_serializable_types(&self, ffi_info: &FfiInfo) -> Vec<String> {
        let mut serializable = Vec::new();

        for ty in &ffi_info.types {
            if self.is_serializable(ty) {
                serializable.push(ty.name.clone());
            }
        }

        serializable
    }

    fn is_serializable(&self, ty: &FfiType) -> bool {
        // Types with no fields are not interesting
        if ty.fields.is_empty() {
            return false;
        }
        
        // Opaque types cannot be serialized
        if ty.is_opaque {
            return false;
        }
        
        // Types with fields are potentially serializable
        true
    }    /// Generate serde derive attributes
    pub fn generate_serde_attrs(&self, _type_name: &str) -> String {
        format!(
            r#"#[cfg(feature = "serde")]
#[derive(serde::Serialize, serde::Deserialize)]
"#
        )
    }
}

impl Default for SerdeIntegration {
    fn default() -> Self {
        Self::new()
    }
}
