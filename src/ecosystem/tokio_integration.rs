// src/ecosystem/tokio_integration.rs

//! Tokio async integration for generated wrappers

use crate::ffi::{FfiFunction, FfiInfo};

/// Generates async wrappers for blocking FFI functions
pub struct TokioIntegration;

impl TokioIntegration {
    pub fn new() -> Self {
        Self
    }
    
    /// Detect functions that should have async wrappers
    pub fn detect_async_candidates(&self, ffi_info: &FfiInfo) -> Vec<String> {
        let mut candidates = Vec::new();
        
        for func in &ffi_info.functions {
            if self.should_have_async_wrapper(func) {
                candidates.push(func.name.clone());
            }
        }
        
        candidates
    }
    
    fn should_have_async_wrapper(&self, func: &FfiFunction) -> bool {
        let name_lower = func.name.to_lowercase();
        
        // Functions that likely block
        name_lower.contains("read") 
            || name_lower.contains("write")
            || name_lower.contains("wait")
            || name_lower.contains("recv")
            || name_lower.contains("send")
            || name_lower.contains("connect")
            || name_lower.contains("accept")
    }
    
    /// Generate async wrapper for a method
    pub fn generate_async_method(&self, method_name: &str, sync_method: &str) -> String {
        format!(
            r#"#[cfg(feature = "tokio")]
    pub async fn {}_async(&self) -> Result<T, Error> {{
        let handle = self.clone();
        tokio::task::spawn_blocking(move || {{
            handle.{}()
        }})
        .await
        .map_err(|e| Error::AsyncError(e.to_string()))?
    }}
"#,
            method_name, sync_method
        )
    }
}

impl Default for TokioIntegration {
    fn default() -> Self {
        Self::new()
    }
}
