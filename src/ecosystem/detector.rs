// src/ecosystem/detector.rs

//! Detects which ecosystem integrations are relevant for a library

use super::{EcosystemCrate, LibraryCategory};
use crate::ffi::FfiInfo;
use anyhow::Result;
use tracing::info;

/// Detected ecosystem integrations for a library
#[derive(Debug, Clone)]
pub struct EcosystemIntegration {
    /// Category of library
    pub category: LibraryCategory,

    /// Standard ecosystem crates to integrate
    pub standard_crates: Vec<EcosystemCrate>,

    /// Library-specific related crates found on crates.io
    pub related_crates: Vec<RelatedCrate>,
}

/// A crate on crates.io related to the wrapped library
#[derive(Debug, Clone)]
pub struct RelatedCrate {
    pub name: String,
    pub description: String,
    pub downloads: u64,
    pub repository: Option<String>,
}

/// Detects relevant ecosystem integrations
pub struct IntegrationDetector;

impl IntegrationDetector {
    pub fn new() -> Self {
        Self
    }

    /// Detect integrations for a library
    pub fn detect(&self, ffi_info: &FfiInfo, library_name: &str) -> Result<EcosystemIntegration> {
        info!("Detecting ecosystem integrations for {}", library_name);

        // Detect category
        let category = self.detect_category(ffi_info, library_name);
        info!("Detected category: {:?}", category);

        // Get recommended standard crates
        let mut standard_crates = category.recommended_integrations();

        // Always include common ones
        standard_crates.push(EcosystemCrate::Tracing);
        standard_crates.push(EcosystemCrate::Thiserror);

        // Deduplicate
        standard_crates.sort_by_key(|c| c.crate_name());
        standard_crates.dedup();

        // TODO: Search crates.io for related crates
        let related_crates = Vec::new();

        Ok(EcosystemIntegration {
            category,
            standard_crates,
            related_crates,
        })
    }

    fn detect_category(&self, ffi_info: &FfiInfo, library_name: &str) -> LibraryCategory {
        let name_lower = library_name.to_lowercase();
        let all_text = format!(
            "{} {} {}",
            library_name,
            ffi_info
                .functions
                .iter()
                .map(|f| f.name.as_str())
                .collect::<Vec<_>>()
                .join(" "),
            ffi_info
                .types
                .iter()
                .map(|t| t.name.as_str())
                .collect::<Vec<_>>()
                .join(" ")
        )
        .to_lowercase();

        // Check for math libraries
        if name_lower.contains("blas")
            || name_lower.contains("lapack")
            || name_lower.contains("math")
            || name_lower.contains("linalg")
            || all_text.contains("matrix")
            || all_text.contains("vector")
            || all_text.contains("tensor")
        {
            return LibraryCategory::Mathematics;
        }

        // Check for graphics libraries
        if name_lower.contains("gl")
            || name_lower.contains("vulkan")
            || name_lower.contains("cuda")
            || name_lower.contains("directx")
            || name_lower.contains("metal")
            || all_text.contains("render")
            || all_text.contains("shader")
            || all_text.contains("texture")
            || all_text.contains("buffer")
        {
            return LibraryCategory::Graphics;
        }

        // Check for ML libraries
        if name_lower.contains("cudnn")
            || name_lower.contains("tensor")
            || name_lower.contains("onnx")
            || name_lower.contains("torch")
            || name_lower.contains("tensorflow")
            || all_text.contains("neural")
            || all_text.contains("inference")
            || all_text.contains("training")
        {
            return LibraryCategory::MachineLearning;
        }

        // Check for networking
        if name_lower.contains("curl")
            || name_lower.contains("http")
            || name_lower.contains("socket")
            || name_lower.contains("net")
            || all_text.contains("connect")
            || all_text.contains("request")
            || all_text.contains("response")
        {
            return LibraryCategory::Networking;
        }

        // Check for crypto
        if name_lower.contains("ssl")
            || name_lower.contains("tls")
            || name_lower.contains("crypto")
            || name_lower.contains("cipher")
            || all_text.contains("encrypt")
            || all_text.contains("decrypt")
            || all_text.contains("hash")
            || all_text.contains("sign")
        {
            return LibraryCategory::Cryptography;
        }

        // Check for multimedia
        if name_lower.contains("ffmpeg")
            || name_lower.contains("codec")
            || name_lower.contains("audio")
            || name_lower.contains("video")
            || name_lower.contains("image")
            || all_text.contains("decode")
            || all_text.contains("encode")
        {
            return LibraryCategory::Multimedia;
        }

        // Check for database
        if name_lower.contains("sql")
            || name_lower.contains("db")
            || name_lower.contains("database")
            || all_text.contains("query")
            || all_text.contains("transaction")
        {
            return LibraryCategory::Database;
        }

        // Check for system
        if name_lower.contains("sys")
            || name_lower.contains("posix")
            || name_lower.contains("win32")
            || all_text.contains("file")
            || all_text.contains("process")
        {
            return LibraryCategory::System;
        }

        LibraryCategory::General
    }

    /// Generate Cargo.toml feature section
    pub fn generate_cargo_features(&self, integration: &EcosystemIntegration) -> String {
        let mut output = String::new();

        output.push_str("[features]\n");
        output.push_str("default = []\n\n");

        // Group by tier
        output.push_str("# Tier 1: Universal (recommended for all crates)\n");
        for crate_ref in &integration.standard_crates {
            if crate_ref.tier() == 1 {
                output.push_str(&format!(
                    "{} = [\"dep:{}\"]\n",
                    crate_ref.feature_name(),
                    crate_ref.crate_name()
                ));
            }
        }

        output.push_str("\n# Ecosystem integrations (enable as needed)\n");
        for crate_ref in &integration.standard_crates {
            if crate_ref.tier() > 1 {
                output.push_str(&format!(
                    "{} = [\"dep:{}\"]\n",
                    crate_ref.feature_name(),
                    crate_ref.crate_name()
                ));
            }
        }

        output.push_str("\n[dependencies]\n");
        output.push_str("# Tier 1: Universal dependencies (always optional)\n");
        for crate_ref in &integration.standard_crates {
            if crate_ref.tier() == 1 {
                let features = if let Some(features) = crate_ref.cargo_features() {
                    format!(", features = {:?}", features)
                } else {
                    String::new()
                };

                output.push_str(&format!(
                    "{} = {{ version = \"{}\", optional = true{} }}\n",
                    crate_ref.crate_name(),
                    crate_ref.version(),
                    features
                ));
            }
        }

        output.push_str("\n# Ecosystem integrations\n");
        for crate_ref in &integration.standard_crates {
            if crate_ref.tier() > 1 {
                let features = if let Some(features) = crate_ref.cargo_features() {
                    format!(", features = {:?}", features)
                } else {
                    String::new()
                };

                output.push_str(&format!(
                    "{} = {{ version = \"{}\", optional = true{} }}\n",
                    crate_ref.crate_name(),
                    crate_ref.version(),
                    features
                ));
            }
        }

        output
    }
    /// Generate README section explaining integrations
    pub fn generate_readme_section(&self, integration: &EcosystemIntegration) -> String {
        let mut output = String::new();

        output.push_str("## Ecosystem Integrations\n\n");
        output.push_str(
            "This crate provides optional integrations with popular Rust ecosystem crates.\n\n",
        );
        output.push_str("Enable integrations by adding feature flags:\n\n");
        output.push_str("```toml\n");
        output.push_str("[dependencies]\n");
        output.push_str("my-lib-sys = { version = \"0.1\", features = [");

        let features: Vec<_> = integration
            .standard_crates
            .iter()
            .take(3)
            .map(|c| format!("\"{}\"", c.feature_name()))
            .collect();
        output.push_str(&features.join(", "));
        output.push_str("] }\n```\n\n");

        output.push_str("### Available Integrations\n\n");

        for crate_ref in &integration.standard_crates {
            output.push_str(&format!(
                "- **`{}`** - {}\n",
                crate_ref.feature_name(),
                crate_ref.description()
            ));
        }

        output.push_str("\n");
        output
    }
}

impl Default for IntegrationDetector {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_category_detection_cuda() {
        let detector = IntegrationDetector::new();
        let ffi_info = FfiInfo::default();

        let category = detector.detect_category(&ffi_info, "cuda");
        assert_eq!(category, LibraryCategory::Graphics);
    }

    #[test]
    fn test_category_detection_openssl() {
        let detector = IntegrationDetector::new();
        let ffi_info = FfiInfo::default();

        let category = detector.detect_category(&ffi_info, "openssl");
        assert_eq!(category, LibraryCategory::Cryptography);
    }

    #[test]
    fn test_category_detection_vulkan() {
        let detector = IntegrationDetector::new();
        let ffi_info = FfiInfo::default();

        let category = detector.detect_category(&ffi_info, "vulkan");
        assert_eq!(category, LibraryCategory::Graphics);
    }

    #[test]
    fn test_category_detection_cudnn() {
        let detector = IntegrationDetector::new();
        let ffi_info = FfiInfo::default();

        let category = detector.detect_category(&ffi_info, "cudnn");
        assert_eq!(category, LibraryCategory::MachineLearning);
    }

    #[test]
    fn test_category_detection_blas() {
        let detector = IntegrationDetector::new();
        let ffi_info = FfiInfo::default();

        let category = detector.detect_category(&ffi_info, "blas");
        assert_eq!(category, LibraryCategory::Mathematics);
    }

    #[test]
    fn test_category_detection_curl() {
        let detector = IntegrationDetector::new();
        let ffi_info = FfiInfo::default();

        let category = detector.detect_category(&ffi_info, "curl");
        assert_eq!(category, LibraryCategory::Networking);
    }

    #[test]
    fn test_category_detection_sqlite() {
        let detector = IntegrationDetector::new();
        let ffi_info = FfiInfo::default();

        let category = detector.detect_category(&ffi_info, "sqlite");
        assert_eq!(category, LibraryCategory::Database);
    }

    #[test]
    fn test_category_detection_zlib() {
        let detector = IntegrationDetector::new();
        let ffi_info = FfiInfo::default();

        let category = detector.detect_category(&ffi_info, "zlib");
        // zlib doesn't have a specific category, falls back to General
        assert_eq!(category, LibraryCategory::General);
    }

    #[test]
    fn test_category_detection_unknown() {
        let detector = IntegrationDetector::new();
        let ffi_info = FfiInfo::default();

        let category = detector.detect_category(&ffi_info, "unknown_lib");
        assert_eq!(category, LibraryCategory::General);
    }

    #[test]
    fn test_category_from_function_names() {
        let detector = IntegrationDetector::new();
        let mut ffi_info = FfiInfo::default();

        // Add functions with "render" in the name
        ffi_info.functions.push(crate::ffi::FfiFunction {
            name: "render_frame".to_string(),
            return_type: "void".to_string(),
            params: vec![],
            docs: None,
        });

        let category = detector.detect_category(&ffi_info, "testlib");
        assert_eq!(category, LibraryCategory::Graphics);
    }

    #[test]
    fn test_category_from_type_names() {
        let detector = IntegrationDetector::new();
        let mut ffi_info = FfiInfo::default();

        // Add types with "neural" in the name for ML detection
        ffi_info.types.push(crate::ffi::FfiType {
            name: "NeuralNetwork".to_string(),
            is_opaque: true,
            docs: None,
            fields: vec![],
        });

        let category = detector.detect_category(&ffi_info, "testlib");
        assert_eq!(category, LibraryCategory::MachineLearning);
    }

    #[test]
    fn test_detect_integration_success() {
        let detector = IntegrationDetector::new();
        let ffi_info = FfiInfo::default();

        let result = detector.detect(&ffi_info, "testlib");
        assert!(result.is_ok());

        let integration = result.unwrap();
        assert!(!integration.standard_crates.is_empty());
        // Should always include tracing and thiserror
        assert!(
            integration
                .standard_crates
                .iter()
                .any(|c| c.crate_name() == "tracing")
        );
        assert!(
            integration
                .standard_crates
                .iter()
                .any(|c| c.crate_name() == "thiserror")
        );
    }

    #[test]
    fn test_detect_integration_cuda() {
        let detector = IntegrationDetector::new();
        let ffi_info = FfiInfo::default();

        let result = detector.detect(&ffi_info, "cuda");
        assert!(result.is_ok());

        let integration = result.unwrap();
        assert_eq!(integration.category, LibraryCategory::Graphics);

        // Graphics category should recommend relevant crates
        let crate_names: Vec<_> = integration
            .standard_crates
            .iter()
            .map(|c| c.crate_name())
            .collect();
        assert!(crate_names.contains(&"tracing"));
        assert!(crate_names.contains(&"thiserror"));
    }

    #[test]
    fn test_detect_integration_openssl() {
        let detector = IntegrationDetector::new();
        let ffi_info = FfiInfo::default();

        let result = detector.detect(&ffi_info, "openssl");
        assert!(result.is_ok());

        let integration = result.unwrap();
        assert_eq!(integration.category, LibraryCategory::Cryptography);
    }

    #[test]
    fn test_category_detection_with_encrypt_functions() {
        let detector = IntegrationDetector::new();
        let mut ffi_info = FfiInfo::default();

        // Simulate a library with encryption functions
        ffi_info.functions.push(crate::ffi::FfiFunction {
            name: "encrypt_data".to_string(),
            return_type: "int".to_string(),
            params: vec![],
            docs: None,
        });

        let category = detector.detect_category(&ffi_info, "testlib");
        assert_eq!(category, LibraryCategory::Cryptography);
    }

    #[test]
    fn test_category_detection_with_matrix_types() {
        let detector = IntegrationDetector::new();
        let mut ffi_info = FfiInfo::default();

        // Simulate a library with matrix types
        ffi_info.types.push(crate::ffi::FfiType {
            name: "Matrix".to_string(),
            is_opaque: true,
            docs: None,
            fields: vec![],
        });

        let category = detector.detect_category(&ffi_info, "testlib");
        assert_eq!(category, LibraryCategory::Mathematics);
    }

    #[test]
    fn test_category_detection_with_shader_functions() {
        let detector = IntegrationDetector::new();
        let mut ffi_info = FfiInfo::default();

        // Simulate a library with shader functions
        ffi_info.functions.push(crate::ffi::FfiFunction {
            name: "compile_shader".to_string(),
            return_type: "int".to_string(),
            params: vec![],
            docs: None,
        });

        let category = detector.detect_category(&ffi_info, "testlib");
        assert_eq!(category, LibraryCategory::Graphics);
    }

    #[test]
    fn test_category_detection_with_neural_functions() {
        let detector = IntegrationDetector::new();
        let mut ffi_info = FfiInfo::default();

        // Simulate a library with neural network functions
        ffi_info.functions.push(crate::ffi::FfiFunction {
            name: "neural_forward".to_string(),
            return_type: "void".to_string(),
            params: vec![],
            docs: None,
        });

        let category = detector.detect_category(&ffi_info, "testlib");
        assert_eq!(category, LibraryCategory::MachineLearning);
    }

    #[test]
    fn test_category_detection_with_socket_functions() {
        let detector = IntegrationDetector::new();
        let mut ffi_info = FfiInfo::default();

        // Simulate a library with socket functions
        ffi_info.functions.push(crate::ffi::FfiFunction {
            name: "socket_connect".to_string(),
            return_type: "int".to_string(),
            params: vec![],
            docs: None,
        });

        let category = detector.detect_category(&ffi_info, "testlib");
        assert_eq!(category, LibraryCategory::Networking);
    }

    #[test]
    fn test_integration_includes_common_crates() {
        let detector = IntegrationDetector::new();
        let ffi_info = FfiInfo::default();

        let result = detector.detect(&ffi_info, "testlib");
        assert!(result.is_ok());

        let integration = result.unwrap();

        // Should always include tracing and thiserror
        let crate_names: Vec<_> = integration
            .standard_crates
            .iter()
            .map(|c| c.crate_name())
            .collect();

        assert!(crate_names.contains(&"tracing"));
        assert!(crate_names.contains(&"thiserror"));
    }

    #[test]
    fn test_integration_deduplicates_crates() {
        let detector = IntegrationDetector::new();
        let ffi_info = FfiInfo::default();

        let result = detector.detect(&ffi_info, "cuda");
        assert!(result.is_ok());

        let integration = result.unwrap();
        let crate_names: Vec<_> = integration
            .standard_crates
            .iter()
            .map(|c| c.crate_name())
            .collect();

        // Should not have duplicates
        let unique_count = crate_names
            .iter()
            .collect::<std::collections::HashSet<_>>()
            .len();
        assert_eq!(unique_count, crate_names.len());
    }

    #[test]
    fn test_empty_ffi_info_detection() {
        let detector = IntegrationDetector::new();
        let ffi_info = FfiInfo::default();

        let category = detector.detect_category(&ffi_info, "unknown");
        let result = detector.detect(&ffi_info, "unknown");

        // Should handle empty FFI info gracefully
        assert_eq!(category, LibraryCategory::General);
        assert!(result.is_ok());

        let integration = result.unwrap();
        // Should still have basic recommendations (tracing, thiserror)
        assert!(!integration.standard_crates.is_empty());
    }
}
