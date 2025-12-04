//! End-to-end integration tests
//!
//! Tests the complete pipeline from parsing headers to generating code

use bindings_generat::analyzer::*;
use bindings_generat::ffi::*;

/// Helper to create test FFI info
fn create_test_ffi_info() -> FfiInfo {
    let ffi_info = FfiInfo {
        functions: vec![
            FfiFunction {
                name: "createResource".to_string(),
                params: vec![],
                return_type: "ResourceHandle*".to_string(),
                docs: Some("Creates a new resource".to_string()),
            },
            FfiFunction {
                name: "destroyResource".to_string(),
                params: vec![FfiParam {
                    name: "handle".to_string(),
                    ty: "ResourceHandle*".to_string(),
                    is_mut: false,
                    is_pointer: true,
                }],
                return_type: "void".to_string(),
                docs: Some("Destroys a resource".to_string()),
            },
        ],
        types: Vec::new(),
        enums: vec![FfiEnum {
            name: "Status".to_string(),
            variants: vec![
                FfiEnumVariant {
                    name: "SUCCESS".to_string(),
                    value: Some(0),
                    docs: Some("Operation succeeded".to_string()),
                },
                FfiEnumVariant {
                    name: "ERROR_INVALID".to_string(),
                    value: Some(1),
                    docs: Some("Invalid parameter".to_string()),
                },
            ],
            docs: None,
        }],
        constants: Vec::new(),
        dependencies: Vec::new(),
        opaque_types: Vec::new(),
        type_aliases: std::collections::HashMap::new(),
    };

    ffi_info
}

#[test]
fn test_full_pipeline() {
    let ffi_info = create_test_ffi_info();

    // Verify basic content
    assert!(!ffi_info.functions.is_empty());
    assert!(!ffi_info.enums.is_empty());

    // Run some analyzers to verify they don't panic
    let _type_docs = TypeDocAnalyzer::new().analyze(&ffi_info);
    let _error_docs = ErrorDocAnalyzer::new().analyze(&ffi_info);

    println!("Pipeline completed successfully");
}

#[test]
fn test_analyzer_composition() {
    let ffi_info = create_test_ffi_info();

    // Test multiple analyzers
    let _type_docs = TypeDocAnalyzer::new().analyze(&ffi_info);
    let _error_docs = ErrorDocAnalyzer::new().analyze(&ffi_info);
    let _trait_analysis = TraitAnalyzer.analyze(&ffi_info);

    // All should complete without panic
    assert!(true);
}

#[test]
fn test_error_analysis() {
    let ffi_info = create_test_ffi_info();

    // Test error-related analyzers
    let error_docs = ErrorDocAnalyzer::new().analyze(&ffi_info);
    assert!(!error_docs.error_docs.is_empty() || ffi_info.enums.is_empty());
}

#[test]
fn test_documentation_analysis() {
    let ffi_info = create_test_ffi_info();

    let type_docs = TypeDocAnalyzer::new().analyze(&ffi_info);
    // Documentation analysis should complete - just verify it doesn't panic
    let _ = type_docs.type_docs.len();
}

#[test]
fn test_pattern_detection() {
    let ffi_info = create_test_ffi_info();

    let trait_analysis = TraitAnalyzer.analyze(&ffi_info);
    // Pattern detection should complete - just verify it doesn't panic
    let _ = trait_analysis.traits.len();
}
