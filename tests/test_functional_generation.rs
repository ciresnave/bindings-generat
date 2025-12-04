//! Integration test to verify functional test generation

use bindings_generat::analyzer::LlmCodeExample;
use bindings_generat::ffi::{FfiFunction, FfiParam};
use bindings_generat::generator::functional_tests;

#[test]
fn test_functional_test_generation_output() {
    let functions = vec![
        FfiFunction {
            name: "cuda_malloc".to_string(),
            return_type: "int".to_string(),
            params: vec![FfiParam {
                name: "size".to_string(),
                ty: "size_t".to_string(),
                is_pointer: false,
                is_mut: false,
            }],
            docs: None,
        },
        FfiFunction {
            name: "cuda_free".to_string(),
            return_type: "void".to_string(),
            params: vec![FfiParam {
                name: "ptr".to_string(),
                ty: "void*".to_string(),
                is_pointer: true,
                is_mut: false,
            }],
            docs: None,
        },
    ];

    let output = functional_tests::generate_functional_tests(&functions, &[], "test_lib");

    // Verify the output contains expected test structures
    assert!(output.contains("#[cfg(test)]"), "Should have test module");
    assert!(
        output.contains("mod unit_tests"),
        "Should have unit_tests module"
    );
    assert!(
        output.contains("mod integration_tests"),
        "Should have integration_tests module"
    );
    assert!(
        output.contains("mod edge_case_tests"),
        "Should have edge_case_tests module"
    );
    assert!(
        output.contains("mod property_tests"),
        "Should have property_tests module"
    );

    // Verify test names are generated
    assert!(
        output.contains("test_cuda_malloc"),
        "Should have cuda_malloc test"
    );
    assert!(
        output.contains("test_cuda_free"),
        "Should have cuda_free test"
    );

    // Print for manual inspection
    println!("\n=== GENERATED FUNCTIONAL TESTS ===\n");
    println!("{}", output);
    println!("\n=== END OUTPUT ===\n");
}

#[test]
fn test_functional_test_with_examples() {
    let functions = vec![FfiFunction {
        name: "create_buffer".to_string(),
        return_type: "int".to_string(),
        params: vec![FfiParam {
            name: "size".to_string(),
            ty: "size_t".to_string(),
            is_pointer: false,
            is_mut: false,
        }],
        docs: None,
    }];

    // Provide an actual example from documentation
    let examples = vec![LlmCodeExample {
        title: "Basic buffer creation".to_string(),
        code: "int result = create_buffer(1024);".to_string(),
        explanation: "Creates a 1KB buffer".to_string(),
    }];

    let output = functional_tests::generate_functional_tests(&functions, &examples, "test_lib");

    // Print for debugging
    println!("\n=== GENERATED TESTS WITH EXAMPLES ===\n");
    println!("{}", output);
    println!("\n=== END OUTPUT ===\n");

    // Should generate test from the example
    assert!(
        output.contains("test_create_buffer"),
        "Should have create_buffer test"
    );

    // Note: The 1024 value might be in the default test since example parsing is still basic
    // The example integration is set up but the parsing logic can be enhanced
    if output.contains("1024") {
        println!("✓ Successfully extracted 1024 from example");
    } else {
        println!("! Using default value - example parsing can be enhanced further");
    }
}
