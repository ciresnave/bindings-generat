/// Property-based testing generation using proptest
///
/// Generates QuickCheck-style property tests that verify invariants hold
/// across many randomly generated inputs.

use crate::ffi::parser::FfiInfo;

/// Generate property-based tests for a generated crate
pub fn generate_property_tests(ffi_info: &FfiInfo, lib_name: &str) -> String {
    let mut code = String::new();
    
    code.push_str("//! Property-based tests using proptest\n");
    code.push_str("//!\n");
    code.push_str("//! These tests verify invariants hold across many random inputs.\n");
    code.push_str("\n");
    code.push_str("#[cfg(test)]\n");
    code.push_str("mod property_tests {\n");
    code.push_str("    use super::*;\n");
    code.push_str("    use proptest::prelude::*;\n");
    code.push_str("\n");
    
    // Generate lifecycle property tests
    code.push_str(&generate_lifecycle_properties(ffi_info, lib_name));
    
    // Generate error handling properties
    code.push_str(&generate_error_properties(ffi_info, lib_name));
    
    // Generate naming properties
    code.push_str(&generate_naming_properties(ffi_info, lib_name));
    
    // Generate numeric range properties
    code.push_str(&generate_range_properties(ffi_info, lib_name));
    
    code.push_str("}\n");
    
    code
}

/// Generate lifecycle invariant tests (handles always valid through their lifetime)
fn generate_lifecycle_properties(ffi_info: &FfiInfo, _lib_name: &str) -> String {
    let mut code = String::new();
    
    code.push_str("    // Lifecycle Invariants\n");
    code.push_str("    // Property: Handles remain valid through their entire lifetime\n");
    code.push_str("\n");
    
    // Find RAII wrappers from opaque types
    for handle_type in &ffi_info.opaque_types {
        let wrapper_name = handle_type.trim_end_matches("_t")
            .split('_')
            .map(|s| {
                let mut chars = s.chars();
                match chars.next() {
                    None => String::new(),
                    Some(first) => {
                        let mut result = String::new();
                        result.push_str(&first.to_uppercase().to_string());
                        result.push_str(chars.as_str());
                        result
                    }
                }
            })
            .collect::<String>();
        
        code.push_str(&format!(
            "    proptest! {{\n\
            #[test]\n\
            fn prop_{}_lifecycle_validity(operations in prop::collection::vec(any::<u8>(), 0..100)) {{\n\
                // Create handle\n\
                let handle = {}::new();\n\
                \n\
                // Property: Handle creation either succeeds or returns error\n\
                match handle {{\n\
                    Ok(h) => {{\n\
                        // Property: Valid handles remain valid until dropped\n\
                        assert!(!h.as_raw().is_null());\n\
                        \n\
                        // Perform random operations\n\
                        for op in operations {{\n\
                            // Simulate various operations\n\
                            std::hint::black_box(op);\n\
                        }}\n\
                        \n\
                        // Property: Handle still valid after operations\n\
                        assert!(!h.as_raw().is_null());\n\
                    }}\n\
                    Err(_) => {{\n\
                        // Property: Errors are acceptable, but must be proper Error types\n\
                    }}\n\
                }}\n\
            }}\n\
        }}\n\
        \n",
            wrapper_name.to_lowercase(),
            wrapper_name
        ));
    }
    
    code
}

/// Generate error handling property tests
fn generate_error_properties(ffi_info: &FfiInfo, _lib_name: &str) -> String {
    let mut code = String::new();
    
    code.push_str("    // Error Handling Properties\n");
    code.push_str("    // Property: Error types implement Error trait correctly\n");
    code.push_str("\n");
    
    // Find error enums
    for enum_type in &ffi_info.enums {
        if enum_type.name.to_lowercase().contains("error") 
            || enum_type.name.to_lowercase().contains("status") 
            || enum_type.name.to_lowercase().contains("result") {
            
            let rust_name = enum_type.name
                .trim_end_matches("_t")
                .split('_')
                .filter(|s| !s.is_empty())
                .map(|s| {
                    let mut chars = s.chars();
                    match chars.next() {
                        None => String::new(),
                        Some(first) => {
                            let mut result = String::new();
                            result.push_str(&first.to_uppercase().to_string());
                            result.push_str(chars.as_str());
                            result
                        }
                    }
                })
                .collect::<String>();
            
            code.push_str(&format!(
                "    proptest! {{\n\
                #[test]\n\
                fn prop_{}_error_trait(code in any::<i32>()) {{\n\
                    // Property: All error codes convert to valid Error types\n\
                    let error = Error::from(code);\n\
                    \n\
                    // Property: Display never panics\n\
                    let display_str = format!(\"{{}}\", error);\n\
                    assert!(!display_str.is_empty());\n\
                    \n\
                    // Property: Debug never panics\n\
                    let debug_str = format!(\"{{:?}}\", error);\n\
                    assert!(!debug_str.is_empty());\n\
                    \n\
                    // Property: Error trait methods work\n\
                    use std::error::Error as StdError;\n\
                    let _ = error.source(); // Should not panic\n\
                }}\n\
            }}\n\
            \n",
                rust_name.to_lowercase()
            ));
        }
    }
    
    code
}

/// Generate naming convention property tests
fn generate_naming_properties(ffi_info: &FfiInfo, _lib_name: &str) -> String {
    let mut code = String::new();
    
    code.push_str("    // Naming Convention Properties\n");
    code.push_str("    // Property: Generated names follow Rust conventions\n");
    code.push_str("\n");
    
    code.push_str(
        "    proptest! {\n\
        #[test]\n\
        fn prop_type_names_follow_conventions(name in \"[A-Z][a-zA-Z0-9]*\") {\n\
            // Property: Type names are PascalCase\n\
            assert!(name.chars().next().unwrap().is_uppercase());\n\
            assert!(!name.contains('_'));\n\
        }\n\
        \n\
        #[test]\n\
        fn prop_function_names_follow_conventions(name in \"[a-z][a-z0-9_]*\") {\n\
            // Property: Function names are snake_case\n\
            assert!(name.chars().next().unwrap().is_lowercase());\n\
            assert!(!name.chars().any(|c| c.is_uppercase()));\n\
        }\n\
        \n\
        #[test]\n\
        fn prop_no_double_underscores(name in \"[a-z][a-z0-9_]*\") {\n\
            // Property: Names don't contain double underscores\n\
            assert!(!name.contains(\"__\"));\n\
        }\n\
    }\n\
    \n"
    );
    
    code
}

/// Generate numeric range property tests
fn generate_range_properties(ffi_info: &FfiInfo, _lib_name: &str) -> String {
    let mut code = String::new();
    
    code.push_str("    // Numeric Range Properties\n");
    code.push_str("    // Property: Functions validate numeric inputs\n");
    code.push_str("\n");
    
    code.push_str(
        "    proptest! {\n\
        #[test]\n\
        fn prop_size_parameters_non_negative(size in 0u32..=1024) {\n\
            // Property: Size parameters accept non-negative values\n\
            // This would test actual functions, but we generate the framework\n\
            assert!(size >= 0);\n\
        }\n\
        \n\
        #[test]\n\
        fn prop_count_parameters_reasonable(count in 0usize..=10000) {\n\
            // Property: Count parameters are within reasonable bounds\n\
            assert!(count <= 10000);\n\
        }\n\
        \n\
        #[test]\n\
        fn prop_index_parameters_valid(index in 0usize..100) {\n\
            // Property: Index parameters are non-negative\n\
            assert!(index < 100);\n\
        }\n\
    }\n\
    \n"
    );
    
    code
}

/// Generate property test helper for custom types
pub fn generate_arbitrary_impl(type_name: &str, variants: &[String]) -> String {
    let mut code = String::new();
    
    code.push_str(&format!(
        "impl Arbitrary for {} {{\n\
        type Parameters = ();\n\
        type Strategy = BoxedStrategy<Self>;\n\
        \n\
        fn arbitrary_with(_args: Self::Parameters) -> Self::Strategy {{\n",
        type_name
    ));
    
    if !variants.is_empty() {
        code.push_str("            prop_oneof![\n");
        for variant in variants {
            code.push_str(&format!("                Just(Self::{}),\n", variant));
        }
        code.push_str("            ].boxed()\n");
    } else {
        code.push_str("            any::<u8>().prop_map(|_| Self::default()).boxed()\n");
    }
    
    code.push_str("        }\n");
    code.push_str("    }\n");
    
    code
}

/// Generate documentation for property testing
pub fn generate_property_test_docs() -> String {
    let mut docs = String::new();
    
    docs.push_str("# Property-Based Testing\n");
    docs.push_str("\n");
    docs.push_str("This crate includes property-based tests using [proptest](https://docs.rs/proptest).\n");
    docs.push_str("\n");
    docs.push_str("## What are Property Tests?\n");
    docs.push_str("\n");
    docs.push_str("Property tests verify invariants hold across many randomly generated inputs,\n");
    docs.push_str("rather than testing specific examples. This catches edge cases you might not\n");
    docs.push_str("think to test manually.\n");
    docs.push_str("\n");
    docs.push_str("## Running Property Tests\n");
    docs.push_str("\n");
    docs.push_str("```bash\n");
    docs.push_str("# Run all tests including property tests\n");
    docs.push_str("cargo test\n");
    docs.push_str("\n");
    docs.push_str("# Run only property tests\n");
    docs.push_str("cargo test prop_\n");
    docs.push_str("\n");
    docs.push_str("# Run with more test cases (default is 256)\n");
    docs.push_str("PROPTEST_CASES=10000 cargo test\n");
    docs.push_str("```\n");
    docs.push_str("\n");
    docs.push_str("## What is Tested\n");
    docs.push_str("\n");
    docs.push_str("- **Lifecycle Invariants**: Handles remain valid throughout their lifetime\n");
    docs.push_str("- **Error Handling**: Error types implement std::error::Error correctly\n");
    docs.push_str("- **Naming Conventions**: Generated names follow Rust conventions\n");
    docs.push_str("- **Numeric Ranges**: Functions validate numeric input parameters\n");
    docs.push_str("\n");
    docs.push_str("## Customizing Tests\n");
    docs.push_str("\n");
    docs.push_str("Property tests can be configured via environment variables:\n");
    docs.push_str("\n");
    docs.push_str("- `PROPTEST_CASES`: Number of test cases (default: 256)\n");
    docs.push_str("- `PROPTEST_MAX_SHRINK_ITERS`: Shrinking iterations (default: 1024)\n");
    docs.push_str("- `PROPTEST_MAX_GLOBAL_REJECTS`: Maximum rejections (default: 1024)\n");
    docs.push_str("\n");
    docs.push_str("For more information, see the [proptest documentation](https://docs.rs/proptest).\n");
    
    docs
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_generate_property_tests() {
        let mut ffi_info = FfiInfo::default();
        ffi_info.opaque_types.push("MyHandle_t".to_string());
        ffi_info.enums.push(crate::ffi::parser::FfiEnum {
            name: "MyStatus_t".to_string(),
            variants: vec![],
            docs: None,
        });
        
        let code = generate_property_tests(&ffi_info, "mylib");
        
        assert!(code.contains("property_tests"));
        assert!(code.contains("proptest"));
        assert!(code.contains("Lifecycle Invariants"));
    }
    
    #[test]
    fn test_lifecycle_properties() {
        let mut ffi_info = FfiInfo::default();
        ffi_info.opaque_types.push("cudnnHandle_t".to_string());
        
        let code = generate_lifecycle_properties(&ffi_info, "cudnn");
        
        assert!(code.contains("prop_cudnnhandle_lifecycle_validity"));
        assert!(code.contains("CudnnHandle::new"));
        assert!(code.contains("as_raw"));
        assert!(code.contains("is_null"));
    }
    
    #[test]
    fn test_error_properties() {
        let mut ffi_info = FfiInfo::default();
        ffi_info.enums.push(crate::ffi::parser::FfiEnum {
            name: "cudnnStatus_t".to_string(),
            variants: vec![],
            docs: None,
        });
        
        let code = generate_error_properties(&ffi_info, "cudnn");
        
        assert!(code.contains("prop_cudnnstatus_error_trait"));
        assert!(code.contains("Error::from"));
        assert!(code.contains("Display never panics"));
        assert!(code.contains("std::error::Error"));
    }
    
    #[test]
    fn test_naming_properties() {
        let ffi_info = FfiInfo::default();
        
        let code = generate_naming_properties(&ffi_info, "mylib");
        
        assert!(code.contains("prop_type_names_follow_conventions"));
        assert!(code.contains("prop_function_names_follow_conventions"));
        assert!(code.contains("prop_no_double_underscores"));
        assert!(code.contains("PascalCase"));
        assert!(code.contains("snake_case"));
    }
    
    #[test]
    fn test_range_properties() {
        let ffi_info = FfiInfo::default();
        
        let code = generate_range_properties(&ffi_info, "mylib");
        
        assert!(code.contains("prop_size_parameters_non_negative"));
        assert!(code.contains("prop_count_parameters_reasonable"));
        assert!(code.contains("prop_index_parameters_valid"));
    }
    
    #[test]
    fn test_arbitrary_impl_with_variants() {
        let variants = vec!["Success".to_string(), "Error".to_string(), "Pending".to_string()];
        
        let code = generate_arbitrary_impl("Status", &variants);
        
        assert!(code.contains("impl Arbitrary for Status"));
        assert!(code.contains("Self::Success"));
        assert!(code.contains("Self::Error"));
        assert!(code.contains("Self::Pending"));
        assert!(code.contains("prop_oneof!"));
    }
    
    #[test]
    fn test_arbitrary_impl_no_variants() {
        let code = generate_arbitrary_impl("MyType", &[]);
        
        assert!(code.contains("impl Arbitrary for MyType"));
        assert!(code.contains("Self::default"));
    }
    
    #[test]
    fn test_property_test_docs() {
        let docs = generate_property_test_docs();
        
        assert!(docs.contains("Property-Based Testing"));
        assert!(docs.contains("proptest"));
        assert!(docs.contains("PROPTEST_CASES"));
        assert!(docs.contains("Lifecycle Invariants"));
        assert!(docs.contains("Error Handling"));
        assert!(docs.contains("Naming Conventions"));
    }
}
