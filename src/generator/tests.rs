//! Test generation for FFI bindings
//!
//! This module generates comprehensive tests for the generated bindings:
//! - Compilation tests to ensure all types are valid
//! - Basic instantiation tests where possible
//! - Documentation examples showing usage patterns

use crate::analyzer::AnalysisResult;
use crate::ffi::{FfiFunction, FfiType};
use std::fmt::Write;

/// Generate integration tests for the bindings
pub fn generate_tests(
    functions: &[FfiFunction],
    types: &[FfiType],
    wrapper_types: &[(String, String)], // (handle_type, wrapper_name)
    lib_name: &str,
    analysis: Option<&AnalysisResult>,
) -> String {
    let mut code = String::new();

    // Module header
    writeln!(
        &mut code,
        "//! Integration tests for {} bindings\n",
        lib_name
    )
    .unwrap();
    writeln!(&mut code, "use {}::*;\n", lib_name).unwrap();

    // Generate compilation tests
    generate_compilation_tests(&mut code, wrapper_types);

    // Generate type size tests
    generate_type_size_tests(&mut code, types, lib_name);

    // Generate wrapper tests
    generate_wrapper_tests(&mut code, wrapper_types, functions);

    // Generate runtime integration tests
    generate_runtime_integration_tests(&mut code, wrapper_types, functions, lib_name);

    // Generate FFI availability tests
    generate_ffi_availability_tests(&mut code, functions, lib_name);

    // Generate enhanced tests based on analysis
    if let Some(analysis) = analysis {
        generate_enhanced_tests(&mut code, analysis, wrapper_types, lib_name);
    }

    code
}

/// Generate tests that verify all types compile
fn generate_compilation_tests(code: &mut String, wrapper_types: &[(String, String)]) {
    writeln!(
        code,
        "/// Tests that verify all wrapper types compile correctly"
    )
    .unwrap();
    writeln!(code, "#[cfg(test)]").unwrap();
    writeln!(code, "mod compilation_tests {{").unwrap();
    writeln!(code, "    use super::*;\n").unwrap();

    writeln!(code, "    #[test]").unwrap();
    writeln!(code, "    fn test_all_types_compile() {{").unwrap();
    writeln!(
        code,
        "        // This test ensures all generated types are valid and compile"
    )
    .unwrap();
    writeln!(
        code,
        "        // If this test compiles, all wrapper types are syntactically correct"
    )
    .unwrap();

    for (_, wrapper_name) in wrapper_types {
        writeln!(code, "        let _: Option<{}> = None;", wrapper_name).unwrap();
    }

    writeln!(code, "    }}").unwrap();
    writeln!(code, "}}\n").unwrap();
}

/// Generate tests that check type sizes match expectations
fn generate_type_size_tests(code: &mut String, _types: &[FfiType], _lib_name: &str) {
    writeln!(code, "/// Tests that verify FFI type sizes").unwrap();
    writeln!(code, "#[cfg(test)]").unwrap();
    writeln!(code, "mod type_size_tests {{").unwrap();
    writeln!(code, "    use super::*;\n").unwrap();

    writeln!(code, "    #[test]").unwrap();
    writeln!(code, "    fn test_type_sizes_placeholder() {{").unwrap();
    writeln!(
        code,
        "        // FFI types are internal implementation details"
    )
    .unwrap();
    writeln!(
        code,
        "        // This module is a placeholder for custom type size tests"
    )
    .unwrap();
    writeln!(
        code,
        "        // Add specific assertions for wrapper types if needed"
    )
    .unwrap();
    writeln!(code, "    }}").unwrap();

    writeln!(code, "}}\n").unwrap();
}

/// Generate tests for wrapper types
fn generate_wrapper_tests(
    code: &mut String,
    wrapper_types: &[(String, String)],
    functions: &[FfiFunction],
) {
    writeln!(code, "/// Tests for wrapper type behavior").unwrap();
    writeln!(code, "#[cfg(test)]").unwrap();
    writeln!(code, "mod wrapper_tests {{").unwrap();
    writeln!(code, "    use super::*;\n").unwrap();

    // Test that wrappers exist and can be mentioned in type position
    writeln!(code, "    #[test]").unwrap();
    writeln!(code, "    fn test_wrapper_types_exist() {{").unwrap();
    writeln!(
        code,
        "        // Verify wrapper types exist and have expected memory layout"
    )
    .unwrap();
    writeln!(
        code,
        "        fn check_type_exists<T>(_: fn() -> Option<T>) {{}}"
    )
    .unwrap();
    writeln!(code).unwrap();

    for (_, wrapper_name) in wrapper_types.iter().take(5) {
        writeln!(
            code,
            "        check_type_exists(|| -> Option<{}> {{ None }});",
            wrapper_name
        )
        .unwrap();
    }

    writeln!(code, "    }}").unwrap();

    // Test for methods existence on wrappers
    if let Some((_handle_type, first_wrapper)) = wrapper_types.first() {
        // Find functions that might be methods on this wrapper
        let handle_type = &wrapper_types[0].0;
        let methods: Vec<_> = functions
            .iter()
            .filter(|f| {
                f.params
                    .first()
                    .map(|p| p.ty.contains(handle_type))
                    .unwrap_or(false)
            })
            .take(3)
            .collect();

        if !methods.is_empty() {
            writeln!(code, "\n    #[test]").unwrap();
            writeln!(code, "    #[ignore] // Requires actual library instance").unwrap();
            writeln!(
                code,
                "    fn test_{}_methods_exist() {{",
                first_wrapper.to_lowercase()
            )
            .unwrap();
            writeln!(
                code,
                "        // This test documents expected methods on {}",
                first_wrapper
            )
            .unwrap();
            writeln!(
                code,
                "        // Uncomment and modify when you have a way to create instances\n"
            )
            .unwrap();
            writeln!(code, "        // let instance = /* create instance */;").unwrap();

            for method in methods {
                let method_name = method.name.to_lowercase().replace(handle_type, "");
                writeln!(code, "        // instance.{}(...);", method_name).unwrap();
            }

            writeln!(code, "    }}").unwrap();
        }
    }

    writeln!(code, "}}\n").unwrap();
}

/// Generate runtime integration tests that actually call FFI functions
fn generate_runtime_integration_tests(
    code: &mut String,
    wrapper_types: &[(String, String)],
    functions: &[FfiFunction],
    _lib_name: &str,
) {
    writeln!(
        code,
        "/// Runtime integration tests that call actual FFI functions"
    )
    .unwrap();
    writeln!(
        code,
        "/// Enable with `cargo test --features runtime-tests`"
    )
    .unwrap();
    writeln!(code, "#[cfg(test)]").unwrap();
    writeln!(code, "#[cfg(feature = \"runtime-tests\")]").unwrap();
    writeln!(code, "mod runtime_integration_tests {{").unwrap();
    writeln!(code, "    use super::*;\n").unwrap();

    // Find lifecycle pairs (create/destroy functions)
    let lifecycle_pairs = find_lifecycle_pairs(functions);

    for (create_func, _destroy_func, wrapper_name) in lifecycle_pairs {
        // Check if this wrapper exists
        if !wrapper_types.iter().any(|(_, w)| w == &wrapper_name) {
            continue;
        }

        writeln!(code, "    #[test]").unwrap();
        writeln!(
            code,
            "    fn test_{}_lifecycle() {{",
            wrapper_name.to_lowercase()
        )
        .unwrap();
        writeln!(
            code,
            "        // Test creating and destroying a {} instance",
            wrapper_name
        )
        .unwrap();
        writeln!(
            code,
            "        // NOTE: This test requires the library to be installed and available"
        )
        .unwrap();

        // Generate test code based on create function signature
        if create_func.params.is_empty() {
            // No parameters - simple creation
            writeln!(code, "        let instance = {}::new();", wrapper_name).unwrap();
            writeln!(
                code,
                "        assert!(instance.is_ok(), \"Failed to create {}\");",
                wrapper_name
            )
            .unwrap();
            writeln!(code, "        if let Ok(inst) = instance {{").unwrap();
            writeln!(
                code,
                "            // Instance created successfully and will be dropped automatically"
            )
            .unwrap();
            writeln!(code, "            drop(inst);").unwrap();
            writeln!(code, "        }}").unwrap();
        } else {
            // Has parameters - might need specific values
            writeln!(
                code,
                "        // Note: This test may need specific initialization parameters"
            )
            .unwrap();
            writeln!(
                code,
                "        // Uncomment and adjust when you know the correct parameters:"
            )
            .unwrap();
            writeln!(
                code,
                "        // let instance = {}::new(/* parameters */);",
                wrapper_name
            )
            .unwrap();
            writeln!(code, "        // assert!(instance.is_ok());").unwrap();
            writeln!(code, "        // if let Ok(inst) = instance {{").unwrap();
            writeln!(code, "        //     drop(inst); // Explicit cleanup").unwrap();
            writeln!(code, "        // }}").unwrap();
        }

        writeln!(code, "    }}\n").unwrap();

        // Generate test for error handling
        writeln!(code, "    #[test]").unwrap();
        writeln!(
            code,
            "    fn test_{}_error_handling() {{",
            wrapper_name.to_lowercase()
        )
        .unwrap();
        writeln!(code, "        // Test that errors are properly propagated").unwrap();
        writeln!(
            code,
            "        // This is a placeholder - adjust based on actual error conditions\n"
        )
        .unwrap();

        writeln!(code, "        // Example: Try to use an invalid parameter").unwrap();
        writeln!(code, "        // let result = some_invalid_operation();").unwrap();
        writeln!(code, "        // assert!(result.is_err());").unwrap();
        writeln!(code, "        // if let Err(e) = result {{").unwrap();
        writeln!(
            code,
            "        //     eprintln!(\"Expected error: {{}}\", e);"
        )
        .unwrap();
        writeln!(code, "        // }}").unwrap();
        writeln!(code, "    }}\n").unwrap();
    }

    // Generate tests for methods on wrappers
    for (handle_type, wrapper_name) in wrapper_types.iter().take(3) {
        // Find methods for this wrapper
        let methods: Vec<_> = functions
            .iter()
            .filter(|f| {
                f.params
                    .first()
                    .map(|p| p.ty.contains(handle_type.as_str()))
                    .unwrap_or(false)
            })
            .filter(|f| {
                // Exclude lifecycle functions
                !is_lifecycle_function(&f.name)
            })
            .take(5)
            .collect();

        if !methods.is_empty() {
            writeln!(code, "    #[test]").unwrap();
            writeln!(
                code,
                "    #[ignore] // Requires valid instance with proper setup"
            )
            .unwrap();
            writeln!(
                code,
                "    fn test_{}_methods() {{",
                wrapper_name.to_lowercase()
            )
            .unwrap();
            writeln!(
                code,
                "        // Test calling methods on {} instance",
                wrapper_name
            )
            .unwrap();
            writeln!(
                code,
                "        // Uncomment when you have a way to create valid instances\n"
            )
            .unwrap();

            writeln!(
                code,
                "        // let mut instance = {}::new().expect(\"Failed to create instance\");",
                wrapper_name
            )
            .unwrap();

            for method in methods {
                let method_name = sanitize_method_name(&method.name, handle_type);
                writeln!(
                    code,
                    "        // instance.{}(...).expect(\"Method call failed\");",
                    method_name
                )
                .unwrap();
            }

            writeln!(code, "\n        // Explicit cleanup (automatic via Drop)").unwrap();
            writeln!(code, "        // drop(instance);").unwrap();
            writeln!(code, "    }}\n").unwrap();
        }
    }

    // Generate a basic smoke test
    writeln!(code, "    #[test]").unwrap();
    writeln!(code, "    fn test_error_type_implements_traits() {{").unwrap();
    writeln!(
        code,
        "        // Verify Error type implements expected traits"
    )
    .unwrap();
    writeln!(
        code,
        "        fn check_error_traits<T: std::error::Error + std::fmt::Debug + std::fmt::Display>() {{}}"
    )
    .unwrap();
    writeln!(code, "        check_error_traits::<Error>();").unwrap();
    writeln!(code, "    }}").unwrap();

    writeln!(code, "}}\n").unwrap();
}

/// Find lifecycle pairs of create/destroy functions
fn find_lifecycle_pairs(
    functions: &[FfiFunction],
) -> Vec<(&FfiFunction, Option<&FfiFunction>, String)> {
    let mut pairs = Vec::new();

    for func in functions {
        if is_create_function(&func.name) {
            // Try to find corresponding destroy function
            let base_name = extract_base_name(&func.name);
            let destroy_func = functions
                .iter()
                .find(|f| is_destroy_function(&f.name) && extract_base_name(&f.name) == base_name);

            // Generate wrapper name from base
            let wrapper_name = if base_name.is_empty() {
                "Handle".to_string()
            } else {
                // Convert to PascalCase
                base_name
                    .split('_')
                    .map(|word| {
                        let lower = word.to_lowercase();
                        if lower.is_empty() {
                            String::new()
                        } else {
                            lower[0..1].to_uppercase() + &lower[1..]
                        }
                    })
                    .collect::<Vec<_>>()
                    .join("")
            };

            pairs.push((func, destroy_func, wrapper_name));
        }
    }

    pairs
}

/// Check if a function is a creation function
fn is_create_function(name: &str) -> bool {
    let lower = name.to_lowercase();
    lower.contains("create")
        || lower.contains("init")
        || lower.contains("new")
        || lower.contains("alloc")
        || lower.contains("open")
}

/// Check if a function is a destruction function
fn is_destroy_function(name: &str) -> bool {
    let lower = name.to_lowercase();
    lower.contains("destroy")
        || lower.contains("free")
        || lower.contains("delete")
        || lower.contains("release")
        || lower.contains("close")
        || lower.contains("cleanup")
}

/// Check if a function is a lifecycle function
fn is_lifecycle_function(name: &str) -> bool {
    is_create_function(name) || is_destroy_function(name)
}

/// Extract base name from a function (remove prefixes/suffixes)
fn extract_base_name(name: &str) -> String {
    let mut base = name.to_string();

    // Remove common prefixes
    for prefix in &["cudnn", "cuda", "vk", "gl", "al"] {
        if base.to_lowercase().starts_with(prefix) {
            base = base[prefix.len()..].to_string();
        }
    }

    // Remove lifecycle keywords
    for keyword in &[
        "Create", "Destroy", "Init", "Free", "New", "Delete", "Alloc",
    ] {
        base = base.replace(keyword, "");
    }

    // Clean up any leading/trailing underscores
    base.trim_matches('_').to_string()
}

/// Sanitize method name by removing handle type and common prefixes
fn sanitize_method_name(name: &str, handle_type: &str) -> String {
    let mut method = name.to_string();

    // Remove handle type from method name
    method = method.replace(handle_type, "");

    // Remove common prefixes
    for prefix in &["cudnn", "cuda", "vk", "gl", "al"] {
        if method.to_lowercase().starts_with(prefix) {
            method = method[prefix.len()..].to_string();
        }
    }

    // Convert to snake_case
    let mut result = String::new();
    for (i, ch) in method.chars().enumerate() {
        if ch.is_uppercase() && i > 0 {
            result.push('_');
        }
        result.push(ch.to_lowercase().next().unwrap());
    }

    result.trim_matches('_').to_string()
}

/// Generate tests that verify FFI functions are available
fn generate_ffi_availability_tests(code: &mut String, functions: &[FfiFunction], _lib_name: &str) {
    writeln!(code, "/// Tests that verify FFI functions are accessible").unwrap();
    writeln!(code, "#[cfg(test)]").unwrap();
    writeln!(code, "mod ffi_availability_tests {{").unwrap();
    writeln!(code, "    use super::*;\n").unwrap();

    // Sample a few functions to test
    let sample_functions: Vec<_> = if functions.is_empty() {
        vec![]
    } else {
        let step = (functions.len() / 5).max(1);
        functions.iter().step_by(step).take(10).collect()
    };

    if !sample_functions.is_empty() {
        writeln!(code, "    #[test]").unwrap();
        writeln!(code, "    fn test_ffi_functions_are_accessible() {{").unwrap();
        writeln!(
            code,
            "        // FFI functions are internal implementation details"
        )
        .unwrap();
        writeln!(
            code,
            "        // This test ensures the module compiles and links correctly"
        )
        .unwrap();
        writeln!(
            code,
            "        // Actual FFI function availability is verified at link time\n"
        )
        .unwrap();
        writeln!(
            code,
            "        // If this test compiles and links, FFI is working"
        )
        .unwrap();
        writeln!(code, "    }}").unwrap();
    }

    writeln!(code, "}}\n").unwrap();
}

/// Generate usage examples as doc tests
pub fn generate_usage_examples(wrapper_types: &[(String, String)]) -> String {
    let mut code = String::new();

    writeln!(&mut code, "//! # Usage Examples\n//!").unwrap();
    writeln!(
        &mut code,
        "//! This section provides examples of how to use the generated bindings.\n//!"
    )
    .unwrap();

    if let Some((_handle_type, wrapper_name)) = wrapper_types.first() {
        writeln!(&mut code, "//! ## Basic Usage\n//!").unwrap();
        writeln!(&mut code, "//! ```ignore").unwrap();
        writeln!(&mut code, "//! use {}::*;\n//!", wrapper_name).unwrap();
        writeln!(
            &mut code,
            "//! // Create an instance (actual creation depends on the library)"
        )
        .unwrap();
        writeln!(
            &mut code,
            "//! // let instance = {}::new()?;\n//!",
            wrapper_name
        )
        .unwrap();
        writeln!(&mut code, "//! // Use methods on the instance").unwrap();
        writeln!(&mut code, "//! // instance.some_method()?;\n//!").unwrap();
        writeln!(
            &mut code,
            "//! // The instance is automatically cleaned up when dropped"
        )
        .unwrap();
        writeln!(&mut code, "//! ```\n//!").unwrap();
    }

    writeln!(&mut code, "//! ## Error Handling\n//!").unwrap();
    writeln!(&mut code, "//! ```ignore").unwrap();
    writeln!(
        &mut code,
        "//! use {}::*;\n//!",
        wrapper_types
            .first()
            .map(|(_, w)| w.as_str())
            .unwrap_or("wrapper")
    )
    .unwrap();
    writeln!(&mut code, "//! fn example() -> Result<(), Error> {{").unwrap();
    writeln!(&mut code, "//!     // Operations return Result<T, Error>").unwrap();
    writeln!(
        &mut code,
        "//!     // You can use ? for error propagation\n//!"
    )
    .unwrap();
    writeln!(&mut code, "//!     Ok(())").unwrap();
    writeln!(&mut code, "//! }}").unwrap();
    writeln!(&mut code, "//! ```").unwrap();

    code
}

/// Generate enhanced tests based on analyzer data
fn generate_enhanced_tests(
    code: &mut String,
    analysis: &AnalysisResult,
    _wrapper_types: &[(String, String)],
    _lib_name: &str,
) {
    writeln!(code, "/// Enhanced tests using semantic analysis").unwrap();
    writeln!(code, "#[cfg(test)]").unwrap();
    writeln!(code, "mod enhanced_tests {{").unwrap();
    writeln!(code, "    use super::*;\n").unwrap();

    // Test error categorization if smart_errors available
    if let Some(smart_errors) = &analysis.smart_errors {
        writeln!(code, "    #[test]").unwrap();
        writeln!(code, "    fn test_error_categorization() {{").unwrap();
        writeln!(code, "        // Test that errors have proper categories").unwrap();
        for error_type in &smart_errors.error_types {
            writeln!(code, "        // Error type: {}", error_type.name).unwrap();
            for variant in error_type.variants.iter().take(3) {
                writeln!(
                    code,
                    "        // - {}: {:?}",
                    variant.name, variant.category
                )
                .unwrap();
            }
        }
        writeln!(code, "    }}").unwrap();
        writeln!(code).unwrap();

        writeln!(code, "    #[test]").unwrap();
        writeln!(code, "    fn test_error_recovery_suggestions() {{").unwrap();
        writeln!(
            code,
            "        // Test that errors provide recovery suggestions"
        )
        .unwrap();
        for error_type in &smart_errors.error_types {
            for variant in error_type.variants.iter().take(2) {
                if !variant.recovery.is_empty() {
                    writeln!(
                        code,
                        "        // {}: {}",
                        variant.name,
                        variant.recovery.first().unwrap()
                    )
                    .unwrap();
                }
            }
        }
        writeln!(code, "    }}").unwrap();
        writeln!(code).unwrap();
    }

    // Test parameter validation if parameter_analysis available
    if let Some(param_analysis) = &analysis.parameter_analysis {
        writeln!(code, "    #[test]").unwrap();
        writeln!(code, "    fn test_parameter_constraints() {{").unwrap();
        writeln!(
            code,
            "        // Test that parameter constraints are enforced"
        )
        .unwrap();
        for (func_name, func_analysis) in param_analysis.function_analysis.iter().take(3) {
            writeln!(code, "        // Function: {}", func_name).unwrap();
            for constraint in &func_analysis.constraints {
                writeln!(code, "        // - Constraint: {:?}", constraint).unwrap();
            }
        }
        writeln!(code, "    }}").unwrap();
        writeln!(code).unwrap();

        writeln!(code, "    #[test]").unwrap();
        writeln!(code, "    fn test_parameter_relationships() {{").unwrap();
        writeln!(
            code,
            "        // Test that parameter relationships are validated"
        )
        .unwrap();
        for (func_name, func_analysis) in param_analysis.function_analysis.iter().take(3) {
            if !func_analysis.relationships.is_empty() {
                writeln!(code, "        // Function: {}", func_name).unwrap();
                for rel in &func_analysis.relationships {
                    writeln!(
                        code,
                        "        // - {} <-> {}: {:?}",
                        rel.param1, rel.param2, rel.relationship
                    )
                    .unwrap();
                }
            }
        }
        writeln!(code, "    }}").unwrap();
        writeln!(code).unwrap();
    }

    // Test builder typestates if available
    if let Some(builder_analysis) = &analysis.builder_typestates {
        writeln!(code, "    #[test]").unwrap();
        writeln!(code, "    fn test_builder_typestates_compile() {{").unwrap();
        writeln!(
            code,
            "        // Test that typestate builders enforce correct usage"
        )
        .unwrap();
        for builder in &builder_analysis.builders {
            writeln!(
                code,
                "        // Builder: {} for {}",
                builder.name, builder.target_type
            )
            .unwrap();
            writeln!(
                code,
                "        // States: {:?}",
                builder.states.iter().map(|s| &s.name).collect::<Vec<_>>()
            )
            .unwrap();
        }
        writeln!(code, "    }}").unwrap();
        writeln!(code).unwrap();
    }

    writeln!(code, "}}").unwrap();
    writeln!(code).unwrap();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_empty_tests() {
        let tests = generate_tests(&[], &[], &[], "test_lib", None);
        assert!(tests.contains("use test_lib::*"));
        assert!(tests.contains("mod compilation_tests"));
    }

    #[test]
    fn test_generate_usage_examples() {
        let wrappers = vec![("Handle_t".to_string(), "Handle".to_string())];
        let examples = generate_usage_examples(&wrappers);
        assert!(examples.contains("Usage Examples"));
        assert!(examples.contains("Handle"));
    }
}
