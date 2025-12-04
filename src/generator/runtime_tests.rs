///! Runtime integration test generation
///!
///! This module generates tests that actually call FFI functions and verify behavior.
///! Tests are conditional on the library being available at test time.
use crate::ffi::{FfiFunction, FfiType};
use std::fmt::Write;

/// Generate comprehensive runtime integration tests
pub fn generate_runtime_tests(
    functions: &[FfiFunction],
    _types: &[FfiType],
    wrapper_types: &[(String, String)], // (handle_type, wrapper_name)
    lib_name: &str,
) -> String {
    let mut code = String::new();

    // Module header
    writeln!(
        &mut code,
        "//! Runtime integration tests for {} bindings\n",
        lib_name
    )
    .unwrap();
    writeln!(
        &mut code,
        "//! These tests require the library to be installed."
    )
    .unwrap();
    writeln!(&mut code, "//! Run with: cargo test --release\n").unwrap();
    writeln!(&mut code, "#![cfg(test)]").unwrap();
    writeln!(&mut code, "use {}::*;\n", lib_name).unwrap();

    // Generate lifecycle tests
    generate_lifecycle_tests(&mut code, functions, wrapper_types, lib_name);

    // Generate error handling tests
    generate_error_handling_tests(&mut code, functions, wrapper_types);

    // Generate method call tests
    generate_method_tests(&mut code, functions, wrapper_types);

    // Generate resource leak tests
    generate_resource_leak_tests(&mut code, wrapper_types);

    // Generate concurrency tests
    generate_concurrency_tests(&mut code, wrapper_types);

    code
}

/// Generate tests for handle lifecycle (create/use/destroy)
fn generate_lifecycle_tests(
    code: &mut String,
    functions: &[FfiFunction],
    wrapper_types: &[(String, String)],
    _lib_name: &str,
) {
    writeln!(code, "#[cfg(test)]").unwrap();
    writeln!(code, "mod lifecycle_tests {{").unwrap();
    writeln!(code, "    use super::*;\n").unwrap();

    let lifecycle_pairs = find_lifecycle_pairs(functions);

    for (create_func, _destroy_func, wrapper_name) in lifecycle_pairs {
        // Check if this wrapper exists
        if !wrapper_types.iter().any(|(_, w)| w == &wrapper_name) {
            continue;
        }

        writeln!(code, "    #[test]").unwrap();
        writeln!(
            code,
            "    fn test_{}_create_and_drop() {{",
            wrapper_name.to_lowercase()
        )
        .unwrap();
        writeln!(
            code,
            "        // Test RAII pattern - resource automatically cleaned up"
        )
        .unwrap();

        // Check if create function needs parameters
        if create_func.params.is_empty() {
            writeln!(code, "        let result = {}::new();", wrapper_name).unwrap();
            writeln!(code, "        match result {{").unwrap();
            writeln!(code, "            Ok(handle) => {{").unwrap();
            writeln!(
                code,
                "                // Handle created successfully, will be dropped at end of scope"
            )
            .unwrap();
            writeln!(
                code,
                "                println!(\"✓ Created {{}} successfully\", stringify!({}));", wrapper_name
            )
            .unwrap();
            writeln!(
                code,
                "                drop(handle); // Explicit drop for clarity"
            )
            .unwrap();
            writeln!(
                code,
                "                println!(\"✓ Dropped {{}} successfully\", stringify!({}));", wrapper_name
            )
            .unwrap();
            writeln!(code, "            }}").unwrap();
            writeln!(code, "            Err(e) => {{").unwrap();
            writeln!(
                code,
                "                eprintln!(\"✗ Failed to create {}: {{}}\", e);",
                wrapper_name
            )
            .unwrap();
            writeln!(
                code,
                "                panic!(\"Library not available or initialization failed: {{}}\", e);"
            )
            .unwrap();
            writeln!(code, "            }}").unwrap();
            writeln!(code, "        }}").unwrap();
        } else {
            // Needs parameters - generate commented template
            writeln!(code, "        // Note: This function requires parameters").unwrap();
            writeln!(code, "        // Uncomment and provide appropriate values:").unwrap();
            write!(code, "        // let result = {}::new(", wrapper_name).unwrap();

            for (i, param) in create_func.params.iter().enumerate() {
                if i > 0 {
                    write!(code, ", ").unwrap();
                }
                write!(code, "/* {} */", param.name).unwrap();
            }
            writeln!(code, ");").unwrap();
            writeln!(code, "        // assert!(result.is_ok());").unwrap();
        }

        writeln!(code, "    }}\n").unwrap();

        // Generate test for multiple instances
        if create_func.params.is_empty() {
            writeln!(code, "    #[test]").unwrap();
            writeln!(
                code,
                "    fn test_{}_multiple_instances() {{",
                wrapper_name.to_lowercase()
            )
            .unwrap();
            writeln!(
                code,
                "        // Test creating multiple instances simultaneously"
            )
            .unwrap();
            writeln!(code, "        let instances: Vec<_> = (0..3)").unwrap();
            let _ = writeln!(code, "            .map(|i| {{");
            writeln!(
                code,
                "                {}::new().map(|h| (i, h))",
                wrapper_name
            )
            .unwrap();
            writeln!(code, "            }})").unwrap();
            writeln!(code, "            .collect::<Result<Vec<_>, _>>();\n").unwrap();
            writeln!(code, "        match instances {{").unwrap();
            writeln!(code, "            Ok(handles) => {{").unwrap();
            writeln!(
                code,
                "                println!(\"✓ Created {{}} {} instances\", handles.len());",
                wrapper_name
            )
            .unwrap();
            writeln!(
                code,
                "                assert_eq!(handles.len(), 3, \"Should create 3 instances\");"
            )
            .unwrap();
            writeln!(code, "                // All handles dropped automatically").unwrap();
            writeln!(code, "            }}").unwrap();
            writeln!(code, "            Err(e) => {{").unwrap();
            writeln!(
                code,
                "                panic!(\"Failed to create multiple instances: {{}}\", e);"
            )
            .unwrap();
            writeln!(code, "            }}").unwrap();
            writeln!(code, "        }}").unwrap();
            writeln!(code, "    }}\n").unwrap();
        }
    }

    writeln!(code, "}}\n").unwrap();
}

/// Generate error handling tests
fn generate_error_handling_tests(
    code: &mut String,
    _functions: &[FfiFunction],
    wrapper_types: &[(String, String)],
) {
    writeln!(code, "#[cfg(test)]").unwrap();
    writeln!(code, "mod error_handling_tests {{").unwrap();
    writeln!(code, "    use super::*;\n").unwrap();

    // Test error type implements expected traits
    writeln!(code, "    #[test]").unwrap();
    writeln!(code, "    fn test_error_implements_std_error() {{").unwrap();
    writeln!(
        code,
        "        fn assert_error_trait<T: std::error::Error>() {{}}"
    )
    .unwrap();
    writeln!(code, "        assert_error_trait::<Error>();").unwrap();
    writeln!(code, "    }}\n").unwrap();

    writeln!(code, "    #[test]").unwrap();
    writeln!(code, "    fn test_error_is_send_sync() {{").unwrap();
    writeln!(code, "        fn assert_send<T: Send>() {{}}").unwrap();
    writeln!(code, "        fn assert_sync<T: Sync>() {{}}").unwrap();
    writeln!(code, "        assert_send::<Error>();").unwrap();
    writeln!(code, "        assert_sync::<Error>();").unwrap();
    writeln!(code, "    }}\n").unwrap();

    // Test error display
    writeln!(code, "    #[test]").unwrap();
    writeln!(code, "    fn test_error_display_not_empty() {{").unwrap();
    writeln!(
        code,
        "        // Create a sample error and verify Display implementation"
    )
    .unwrap();
    writeln!(
        code,
        "        // This uses internal error codes - adjust based on actual error enum"
    )
    .unwrap();
    writeln!(
        code,
        "        let error = Error::from(1); // Assuming 1 is an error code"
    )
    .unwrap();
    writeln!(code, "        let display = format!(\"{{}}\", error);").unwrap();
    writeln!(
        code,
        "        assert!(!display.is_empty(), \"Error display should not be empty\");"
    )
    .unwrap();
    writeln!(
        code,
        "        assert!(display.len() > 5, \"Error message should be descriptive\");"
    )
    .unwrap();
    writeln!(code, "    }}\n").unwrap();

    // Test error debug
    writeln!(code, "    #[test]").unwrap();
    writeln!(code, "    fn test_error_debug_not_empty() {{").unwrap();
    writeln!(code, "        let error = Error::from(1);").unwrap();
    writeln!(code, "        let debug = format!(\"{{:?}}\", error);").unwrap();
    writeln!(
        code,
        "        assert!(!debug.is_empty(), \"Error debug should not be empty\");"
    )
    .unwrap();
    writeln!(code, "    }}\n").unwrap();

    // Test Result propagation
    if !wrapper_types.is_empty() {
        let (_, first_wrapper) = &wrapper_types[0];
        writeln!(code, "    #[test]").unwrap();
        writeln!(code, "    fn test_result_propagation() {{").unwrap();
        writeln!(
            code,
            "        // Test that errors propagate correctly with ?"
        )
        .unwrap();
        writeln!(
            code,
            "        fn create_and_return() -> Result<(), Error> {{"
        )
        .unwrap();
        writeln!(code, "            let _handle = {}::new()?;", first_wrapper).unwrap();
        writeln!(code, "            Ok(())").unwrap();
        writeln!(code, "        }}").unwrap();
        writeln!(code).unwrap();
        writeln!(code, "        match create_and_return() {{").unwrap();
        writeln!(
            code,
            "            Ok(()) => println!(\"✓ Result propagation works\"),"
        )
        .unwrap();
        writeln!(
            code,
            "            Err(e) => panic!(\"Error propagation failed: {{}}\", e),"
        )
        .unwrap();
        writeln!(code, "        }}").unwrap();
        writeln!(code, "    }}\n").unwrap();
    }

    writeln!(code, "}}\n").unwrap();
}

/// Generate method call tests
fn generate_method_tests(
    code: &mut String,
    functions: &[FfiFunction],
    wrapper_types: &[(String, String)],
) {
    writeln!(code, "#[cfg(test)]").unwrap();
    writeln!(code, "mod method_tests {{").unwrap();
    writeln!(code, "    use super::*;\n").unwrap();

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
            .filter(|f| !is_lifecycle_function(&f.name))
            .take(3)
            .collect();

        if !methods.is_empty() {
            writeln!(code, "    #[test]").unwrap();
            writeln!(code, "    #[ignore] // Requires specific setup").unwrap();
            writeln!(
                code,
                "    fn test_{}_has_methods() {{",
                wrapper_name.to_lowercase()
            )
            .unwrap();
            writeln!(
                code,
                "        // Verify {} has expected methods",
                wrapper_name
            )
            .unwrap();
            writeln!(
                code,
                "        // Uncomment when you can create valid instances:\n"
            )
            .unwrap();

            writeln!(
                code,
                "        // let mut instance = {}::new().expect(\"Failed to create instance\");",
                wrapper_name
            )
            .unwrap();

            for method in &methods {
                let method_name = sanitize_method_name(&method.name, handle_type);
                writeln!(code, "        // Call {}", method_name).unwrap();
                writeln!(
                    code,
                    "        // let result = instance.{}(/* parameters */);",
                    method_name
                )
                .unwrap();
                writeln!(code, "        // assert!(result.is_ok());\n").unwrap();
            }

            writeln!(code, "    }}\n").unwrap();
        }
    }

    writeln!(code, "}}\n").unwrap();
}

/// Generate resource leak detection tests
fn generate_resource_leak_tests(code: &mut String, wrapper_types: &[(String, String)]) {
    writeln!(code, "#[cfg(test)]").unwrap();
    writeln!(code, "mod resource_leak_tests {{").unwrap();
    writeln!(code, "    use super::*;\n").unwrap();

    if !wrapper_types.is_empty() {
        let (_, first_wrapper) = &wrapper_types[0];

        writeln!(code, "    #[test]").unwrap();
        writeln!(code, "    fn test_no_leaks_single_scope() {{").unwrap();
        writeln!(code, "        // Create and drop resource in single scope").unwrap();
        writeln!(code, "        for i in 0..10 {{").unwrap();
        writeln!(code, "            let result = {}::new();", first_wrapper).unwrap();
        writeln!(code, "            if let Ok(handle) = result {{").unwrap();
        writeln!(
            code,
            "                drop(handle); // Explicit drop, then scope ends"
        )
        .unwrap();
        writeln!(
            code,
            "                println!(\"✓ Iteration {{}}: Resource created and dropped\", i);"
        )
        .unwrap();
        writeln!(code, "            }}").unwrap();
        writeln!(code, "        }}").unwrap();
        writeln!(
            code,
            "        // If this completes without memory issues, no leaks detected"
        )
        .unwrap();
        writeln!(code, "    }}\n").unwrap();

        writeln!(code, "    #[test]").unwrap();
        writeln!(code, "    fn test_no_leaks_nested_scopes() {{").unwrap();
        writeln!(code, "        // Test RAII with nested scopes").unwrap();
        writeln!(code, "        let outer = {}::new();", first_wrapper).unwrap();
        writeln!(code, "        if let Ok(_outer_handle) = outer {{").unwrap();
        writeln!(code, "            {{").unwrap();
        writeln!(
            code,
            "                let inner = {}::new();",
            first_wrapper
        )
        .unwrap();
        writeln!(code, "                if let Ok(_inner_handle) = inner {{").unwrap();
        writeln!(code, "                    // Inner handle dropped here").unwrap();
        writeln!(code, "                }}").unwrap();
        writeln!(code, "            }}").unwrap();
        writeln!(
            code,
            "            // Outer handle still valid here, dropped at end"
        )
        .unwrap();
        writeln!(code, "        }}").unwrap();
        writeln!(code, "    }}\n").unwrap();
    }

    writeln!(code, "}}\n").unwrap();
}

/// Generate concurrency tests
fn generate_concurrency_tests(code: &mut String, wrapper_types: &[(String, String)]) {
    writeln!(code, "#[cfg(test)]").unwrap();
    writeln!(code, "mod concurrency_tests {{").unwrap();
    writeln!(code, "    use super::*;\n").unwrap();

    writeln!(code, "    #[test]").unwrap();
    writeln!(code, "    fn test_error_is_thread_safe() {{").unwrap();
    writeln!(code, "        // Verify error can be sent between threads").unwrap();
    writeln!(code, "        let error = Error::from(1);").unwrap();
    writeln!(code, "        let handle = std::thread::spawn(move || {{").unwrap();
    writeln!(code, "            format!(\"{{}}\", error)").unwrap();
    writeln!(code, "        }});").unwrap();
    writeln!(code, "        let result = handle.join().unwrap();").unwrap();
    writeln!(
        code,
        "        assert!(!result.is_empty(), \"Error should work across threads\");"
    )
    .unwrap();
    writeln!(code, "    }}\n").unwrap();

    if !wrapper_types.is_empty() {
        let (_, first_wrapper) = &wrapper_types[0];

        writeln!(code, "    #[test]").unwrap();
        writeln!(code, "    #[ignore] // May not be thread-safe").unwrap();
        writeln!(code, "    fn test_create_in_multiple_threads() {{").unwrap();
        writeln!(
            code,
            "        // Test creating handles in different threads"
        )
        .unwrap();
        writeln!(
            code,
            "        // WARNING: Only run if library is thread-safe!"
        )
        .unwrap();
        writeln!(code, "        let handles: Vec<_> = (0..3)").unwrap();
        writeln!(code, "            .map(|i| {{").unwrap();
        writeln!(code, "                std::thread::spawn(move || {{").unwrap();
        writeln!(
            code,
            "                    {}::new().map(|h| (i, h))",
            first_wrapper
        )
        .unwrap();
        writeln!(code, "                }})").unwrap();
        writeln!(code, "            }})").unwrap();
        writeln!(code, "            .collect();").unwrap();
        writeln!(code).unwrap();
        writeln!(
            code,
            "        for (i, handle) in handles.into_iter().enumerate() {{"
        )
        .unwrap();
        writeln!(code, "            match handle.join().unwrap() {{").unwrap();
        writeln!(
            code,
            "                Ok(_) => println!(\"✓ Thread {{}} created handle\", i),"
        )
        .unwrap();
        writeln!(
            code,
            "                Err(e) => println!(\"✗ Thread {{}} failed: {{}}\", i, e),"
        )
        .unwrap();
        writeln!(code, "            }}").unwrap();
        writeln!(code, "        }}").unwrap();
        writeln!(code, "    }}\n").unwrap();
    }

    writeln!(code, "}}\n").unwrap();
}

// Helper functions (same as in tests.rs)

fn find_lifecycle_pairs(
    functions: &[FfiFunction],
) -> Vec<(&FfiFunction, Option<&FfiFunction>, String)> {
    let mut pairs = Vec::new();

    for func in functions {
        if is_create_function(&func.name) {
            let base_name = extract_base_name(&func.name);
            let destroy_func = functions
                .iter()
                .find(|f| is_destroy_function(&f.name) && extract_base_name(&f.name) == base_name);

            let wrapper_name = if base_name.is_empty() {
                "Handle".to_string()
            } else {
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

fn is_create_function(name: &str) -> bool {
    let lower = name.to_lowercase();
    lower.contains("create")
        || lower.contains("init")
        || lower.contains("new")
        || lower.contains("alloc")
        || lower.contains("open")
}

fn is_destroy_function(name: &str) -> bool {
    let lower = name.to_lowercase();
    lower.contains("destroy")
        || lower.contains("free")
        || lower.contains("delete")
        || lower.contains("release")
        || lower.contains("close")
        || lower.contains("cleanup")
}

fn is_lifecycle_function(name: &str) -> bool {
    is_create_function(name) || is_destroy_function(name)
}

fn extract_base_name(name: &str) -> String {
    let mut base = name.to_string();

    for prefix in &["cudnn", "cuda", "vk", "gl", "al"] {
        if base.to_lowercase().starts_with(prefix) {
            base = base[prefix.len()..].to_string();
        }
    }

    for keyword in &[
        "Create", "Destroy", "Init", "Free", "New", "Delete", "Alloc",
    ] {
        base = base.replace(keyword, "");
    }

    base.trim_matches('_').to_string()
}

fn sanitize_method_name(name: &str, handle_type: &str) -> String {
    let mut method = name.to_string();
    method = method.replace(handle_type, "");

    for prefix in &["cudnn", "cuda", "vk", "gl", "al"] {
        if method.to_lowercase().starts_with(prefix) {
            method = method[prefix.len()..].to_string();
        }
    }

    let mut result = String::new();
    for (i, ch) in method.chars().enumerate() {
        if ch.is_uppercase() && i > 0 {
            result.push('_');
        }
        result.push(ch.to_lowercase().next().unwrap());
    }

    result.trim_matches('_').to_string()
}
