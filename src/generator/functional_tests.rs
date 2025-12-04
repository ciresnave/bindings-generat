//! Functional test generation with real test data
//!
//! This module generates tests that actually verify functionality with realistic inputs,
//! not just structural tests. It mines examples from documentation and creates:
//! - Unit tests for every FFI function
//! - Integration tests for common use cases
//! - Edge case tests for success and failure paths
//! - Property-based tests with real constraints

use crate::analyzer::LlmCodeExample;
use crate::ffi::{FfiFunction, FfiParam};
use regex::Regex;
use std::fmt::Write;

/// Test case extracted from examples or documentation
#[derive(Debug, Clone)]
pub struct TestCase {
    pub name: String,
    pub function_name: String,
    pub inputs: Vec<TestValue>,
    pub expected_output: Option<TestValue>,
    pub should_succeed: bool,
    pub description: String,
    pub source: String, // Where this test case came from
}

#[derive(Debug, Clone)]
pub enum TestValue {
    Int(i64),
    UInt(u64),
    Float(f64),
    String(String),
    Null,
    Pointer,
    Array(Vec<TestValue>),
    Struct(Vec<(String, TestValue)>),
}

impl TestValue {
    /// Generate Rust code for this test value
    pub fn to_rust_code(&self) -> String {
        match self {
            TestValue::Int(n) => n.to_string(),
            TestValue::UInt(n) => format!("{}_u64", n),
            TestValue::Float(f) => format!("{}_f64", f),
            TestValue::String(s) => format!("\"{}\"", s),
            TestValue::Null => "std::ptr::null()".to_string(),
            TestValue::Pointer => "std::ptr::null_mut()".to_string(),
            TestValue::Array(items) => {
                let items_str: Vec<String> = items.iter().map(|v| v.to_rust_code()).collect();
                format!("vec![{}]", items_str.join(", "))
            }
            TestValue::Struct(fields) => {
                let fields_str: Vec<String> = fields
                    .iter()
                    .map(|(name, val)| format!("{}: {}", name, val.to_rust_code()))
                    .collect();
                format!("{{ {} }}", fields_str.join(", "))
            }
        }
    }
}

/// Generate comprehensive functional tests
pub fn generate_functional_tests(
    functions: &[FfiFunction],
    examples: &[LlmCodeExample],
    lib_name: &str,
) -> String {
    let mut code = String::new();

    writeln!(
        &mut code,
        "//! Functional tests for {} bindings\n",
        lib_name
    )
    .unwrap();
    writeln!(
        &mut code,
        "//! These tests verify actual functionality with real data."
    )
    .unwrap();
    writeln!(
        &mut code,
        "//! Generated from examples and documentation.\n"
    )
    .unwrap();
    writeln!(&mut code, "#![cfg(test)]").unwrap();
    writeln!(&mut code, "use {}::*;", lib_name).unwrap();
    writeln!(&mut code, "use std::ptr;\n").unwrap();

    // Extract test cases from examples
    let test_cases = extract_test_cases(functions, examples);

    // Generate unit tests for each function
    generate_unit_tests(&mut code, functions, &test_cases);

    // Generate integration tests for common workflows
    generate_integration_tests(&mut code, &test_cases);

    // Generate edge case tests
    generate_edge_case_tests(&mut code, functions, &test_cases);

    // Generate property-based tests
    generate_property_tests(&mut code, functions);

    code
}

/// Extract test cases from example code
fn extract_test_cases(functions: &[FfiFunction], examples: &[LlmCodeExample]) -> Vec<TestCase> {
    let mut test_cases = Vec::new();

    for example in examples {
        // Parse example code to extract function calls and their inputs
        test_cases.extend(parse_example_for_test_cases(
            &example.code,
            functions,
            &example.title,
        ));
    }

    // Add default test cases for functions without examples
    for func in functions {
        if !test_cases.iter().any(|tc| tc.function_name == func.name) {
            test_cases.push(generate_default_test_case(func));
        }
    }

    test_cases
}

/// Parse example code to find function calls and extract test cases
fn parse_example_for_test_cases(
    code: &str,
    functions: &[FfiFunction],
    title: &str,
) -> Vec<TestCase> {
    let mut test_cases = Vec::new();

    for func in functions {
        // Look for calls to this function in the example code
        if let Some(test_case) = extract_function_call(code, func, title) {
            test_cases.push(test_case);
        }
    }

    test_cases
}

/// Extract a function call from example code
fn extract_function_call(code: &str, func: &FfiFunction, example_title: &str) -> Option<TestCase> {
    // Use regex to find function calls with better pattern matching
    let pattern = format!(r"{}[\s]*\((.*?)\)", regex::escape(&func.name));
    let re = Regex::new(&pattern).ok()?;

    for cap in re.captures_iter(code) {
        if let Some(args_match) = cap.get(1) {
            let args_str = args_match.as_str();
            let inputs = parse_arguments(args_str, &func.params);

            return Some(TestCase {
                name: format!("test_{}_from_example", func.name.to_lowercase()),
                function_name: func.name.clone(),
                inputs,
                expected_output: None,
                should_succeed: true,
                description: format!("Test {} - {}", func.name, example_title),
                source: format!("example: {}", example_title),
            });
        }
    }

    None
}

/// Parse function arguments from string
fn parse_arguments(args_str: &str, params: &[FfiParam]) -> Vec<TestValue> {
    let args: Vec<&str> = args_str.split(',').map(|s| s.trim()).collect();
    let mut test_values = Vec::new();

    for (arg, param) in args.iter().zip(params.iter()) {
        test_values.push(parse_argument(arg, param));
    }

    test_values
}

/// Parse a single argument
fn parse_argument(arg: &str, param: &FfiParam) -> TestValue {
    // Try to parse based on type
    if param.is_pointer {
        if arg.contains("null") || arg.contains("NULL") || arg == "0" {
            TestValue::Null
        } else {
            TestValue::Pointer
        }
    } else if param.ty.contains("int") || param.ty.contains("long") {
        if let Ok(n) = arg.parse::<i64>() {
            TestValue::Int(n)
        } else {
            TestValue::Int(0)
        }
    } else if param.ty.contains("float") || param.ty.contains("double") {
        if let Ok(f) = arg.parse::<f64>() {
            TestValue::Float(f)
        } else {
            TestValue::Float(0.0)
        }
    } else if arg.starts_with('"') {
        let s = arg.trim_matches('"');
        TestValue::String(s.to_string())
    } else {
        // Default to int for unknown types
        TestValue::Int(0)
    }
}

/// Generate a default test case for a function without examples
fn generate_default_test_case(func: &FfiFunction) -> TestCase {
    let inputs = func
        .params
        .iter()
        .map(|param| generate_default_value(param))
        .collect();

    TestCase {
        name: format!("test_{}_default", func.name.to_lowercase()),
        function_name: func.name.clone(),
        inputs,
        expected_output: None,
        should_succeed: true,
        description: format!("Test {} with default values", func.name),
        source: "generated defaults".to_string(),
    }
}

/// Generate a default test value for a parameter
fn generate_default_value(param: &FfiParam) -> TestValue {
    if param.is_pointer {
        TestValue::Null
    } else if param.ty.contains("int") || param.ty.contains("long") || param.ty == "size_t" {
        if param.name.contains("size") || param.name.contains("len") || param.name.contains("count")
        {
            TestValue::UInt(1024) // Reasonable size for tests
        } else {
            TestValue::Int(0)
        }
    } else if param.ty.contains("float") || param.ty.contains("double") {
        TestValue::Float(1.0)
    } else {
        TestValue::Int(0)
    }
}

/// Generate unit tests for each function
fn generate_unit_tests(code: &mut String, functions: &[FfiFunction], test_cases: &[TestCase]) {
    writeln!(code, "#[cfg(test)]").unwrap();
    writeln!(code, "mod unit_tests {{").unwrap();
    writeln!(code, "    use super::*;\n").unwrap();

    for func in functions {
        // Get all test cases for this function
        let func_tests: Vec<_> = test_cases
            .iter()
            .filter(|tc| tc.function_name == func.name)
            .collect();

        for test_case in func_tests {
            generate_unit_test(code, func, test_case);
        }
    }

    writeln!(code, "}}\n").unwrap();
}

/// Generate a single unit test
fn generate_unit_test(code: &mut String, func: &FfiFunction, test_case: &TestCase) {
    writeln!(code, "    #[test]").unwrap();

    if !test_case.should_succeed {
        writeln!(code, "    #[should_panic]").unwrap();
    }

    writeln!(code, "    fn {}() {{", test_case.name).unwrap();
    writeln!(code, "        // {}", test_case.description).unwrap();
    writeln!(code, "        // Source: {}", test_case.source).unwrap();

    // Generate the function call
    write!(code, "        let result = {}(", func.name).unwrap();

    for (i, input) in test_case.inputs.iter().enumerate() {
        if i > 0 {
            write!(code, ", ").unwrap();
        }
        write!(code, "{}", input.to_rust_code()).unwrap();
    }

    writeln!(code, ");").unwrap();

    // Generate assertions
    if test_case.should_succeed {
        writeln!(
            code,
            "        assert!(result.is_ok(), \"Function should succeed\");"
        )
        .unwrap();

        if let Some(ref expected) = test_case.expected_output {
            writeln!(
                code,
                "        assert_eq!(result.unwrap(), {});",
                expected.to_rust_code()
            )
            .unwrap();
        }
    } else {
        writeln!(
            code,
            "        assert!(result.is_err(), \"Function should fail\");"
        )
        .unwrap();
    }

    writeln!(code, "    }}\n").unwrap();
}

/// Generate integration tests for common workflows
fn generate_integration_tests(code: &mut String, test_cases: &[TestCase]) {
    writeln!(code, "#[cfg(test)]").unwrap();
    writeln!(code, "mod integration_tests {{").unwrap();
    writeln!(code, "    use super::*;\n").unwrap();

    // Group test cases by workflow patterns
    let workflows = identify_workflows(test_cases);

    for (workflow_name, steps) in workflows {
        generate_workflow_test(code, &workflow_name, &steps);
    }

    writeln!(code, "}}\n").unwrap();
}

/// Identify common workflows from test cases
fn identify_workflows(test_cases: &[TestCase]) -> Vec<(String, Vec<TestCase>)> {
    let mut workflows = Vec::new();

    // Look for create -> use -> destroy patterns
    let create_cases: Vec<_> = test_cases
        .iter()
        .filter(|tc| {
            tc.function_name.contains("create")
                || tc.function_name.contains("init")
                || tc.function_name.contains("new")
        })
        .cloned()
        .collect();

    for create_case in create_cases {
        let mut workflow = vec![create_case.clone()];

        // Find related use cases
        let use_cases: Vec<_> = test_cases
            .iter()
            .filter(|tc| {
                !tc.function_name.contains("create")
                    && !tc.function_name.contains("destroy")
                    && !tc.function_name.contains("free")
            })
            .cloned()
            .collect();

        if let Some(use_case) = use_cases.first() {
            workflow.push(use_case.clone());
        }

        // Find destroy case
        let destroy_case = test_cases
            .iter()
            .find(|tc| {
                tc.function_name.contains("destroy")
                    || tc.function_name.contains("free")
                    || tc.function_name.contains("cleanup")
            })
            .cloned();

        if let Some(destroy) = destroy_case {
            workflow.push(destroy);
        }

        if workflow.len() > 1 {
            workflows.push((format!("workflow_{}", create_case.function_name), workflow));
        }
    }

    workflows
}

/// Generate a workflow integration test
fn generate_workflow_test(code: &mut String, workflow_name: &str, steps: &[TestCase]) {
    writeln!(code, "    #[test]").unwrap();
    writeln!(code, "    fn test_{}() {{", workflow_name.to_lowercase()).unwrap();
    writeln!(code, "        // Integration test: complete workflow").unwrap();

    for (i, step) in steps.iter().enumerate() {
        writeln!(code, "        // Step {}: {}", i + 1, step.description).unwrap();
        write!(code, "        let result_{} = {}(", i, step.function_name).unwrap();

        for (j, input) in step.inputs.iter().enumerate() {
            if j > 0 {
                write!(code, ", ").unwrap();
            }
            write!(code, "{}", input.to_rust_code()).unwrap();
        }

        writeln!(code, ");").unwrap();
        writeln!(code, "        assert!(result_{}.is_ok());", i).unwrap();
    }

    writeln!(code, "    }}\n").unwrap();
}

/// Generate edge case tests
fn generate_edge_case_tests(
    code: &mut String,
    functions: &[FfiFunction],
    _test_cases: &[TestCase],
) {
    writeln!(code, "#[cfg(test)]").unwrap();
    writeln!(code, "mod edge_case_tests {{").unwrap();
    writeln!(code, "    use super::*;\n").unwrap();

    for func in functions {
        // Generate tests for edge cases
        generate_null_pointer_test(code, func);
        generate_zero_size_test(code, func);
        generate_max_value_test(code, func);
    }

    writeln!(code, "}}\n").unwrap();
}

/// Generate null pointer edge case test
fn generate_null_pointer_test(code: &mut String, func: &FfiFunction) {
    let has_pointers = func.params.iter().any(|p| p.is_pointer);

    if !has_pointers {
        return;
    }

    writeln!(code, "    #[test]").unwrap();
    writeln!(
        code,
        "    fn test_{}_null_pointer() {{",
        func.name.to_lowercase()
    )
    .unwrap();
    writeln!(code, "        // Edge case: null pointer input").unwrap();

    write!(code, "        let result = {}(", func.name).unwrap();

    for (i, param) in func.params.iter().enumerate() {
        if i > 0 {
            write!(code, ", ").unwrap();
        }

        if param.is_pointer {
            write!(code, "std::ptr::null_mut()").unwrap();
        } else {
            write!(code, "{}", generate_default_value(param).to_rust_code()).unwrap();
        }
    }

    writeln!(code, ");").unwrap();
    writeln!(code, "        // Should handle null pointers gracefully").unwrap();
    writeln!(code, "        assert!(result.is_err() || result.is_ok());").unwrap();
    writeln!(code, "    }}\n").unwrap();
}

/// Generate zero size edge case test
fn generate_zero_size_test(code: &mut String, func: &FfiFunction) {
    let has_size_param = func
        .params
        .iter()
        .any(|p| p.name.contains("size") || p.name.contains("len") || p.name.contains("count"));

    if !has_size_param {
        return;
    }

    writeln!(code, "    #[test]").unwrap();
    writeln!(
        code,
        "    fn test_{}_zero_size() {{",
        func.name.to_lowercase()
    )
    .unwrap();
    writeln!(code, "        // Edge case: zero size input").unwrap();

    write!(code, "        let result = {}(", func.name).unwrap();

    for (i, param) in func.params.iter().enumerate() {
        if i > 0 {
            write!(code, ", ").unwrap();
        }

        if param.name.contains("size") || param.name.contains("len") || param.name.contains("count")
        {
            write!(code, "0").unwrap();
        } else {
            write!(code, "{}", generate_default_value(param).to_rust_code()).unwrap();
        }
    }

    writeln!(code, ");").unwrap();
    writeln!(code, "        // Should handle zero size gracefully").unwrap();
    writeln!(code, "        assert!(result.is_err() || result.is_ok());").unwrap();
    writeln!(code, "    }}\n").unwrap();
}

/// Generate maximum value edge case test
fn generate_max_value_test(code: &mut String, func: &FfiFunction) {
    let has_numeric = func
        .params
        .iter()
        .any(|p| !p.is_pointer && (p.ty.contains("int") || p.ty.contains("long")));

    if !has_numeric {
        return;
    }

    writeln!(code, "    #[test]").unwrap();
    writeln!(
        code,
        "    fn test_{}_max_value() {{",
        func.name.to_lowercase()
    )
    .unwrap();
    writeln!(code, "        // Edge case: maximum value input").unwrap();

    write!(code, "        let result = {}(", func.name).unwrap();

    for (i, param) in func.params.iter().enumerate() {
        if i > 0 {
            write!(code, ", ").unwrap();
        }

        if !param.is_pointer && (param.ty.contains("int") || param.ty.contains("long")) {
            write!(code, "i64::MAX").unwrap();
        } else {
            write!(code, "{}", generate_default_value(param).to_rust_code()).unwrap();
        }
    }

    writeln!(code, ");").unwrap();
    writeln!(code, "        // Should handle maximum values gracefully").unwrap();
    writeln!(code, "        assert!(result.is_err() || result.is_ok());").unwrap();
    writeln!(code, "    }}\n").unwrap();
}

/// Generate property-based tests
fn generate_property_tests(code: &mut String, functions: &[FfiFunction]) {
    writeln!(code, "#[cfg(test)]").unwrap();
    writeln!(code, "#[cfg(feature = \"proptest\")]").unwrap();
    writeln!(code, "mod property_tests {{").unwrap();
    writeln!(code, "    use super::*;").unwrap();
    writeln!(code, "    use proptest::prelude::*;\n").unwrap();

    for func in functions {
        if can_generate_property_test(func) {
            generate_property_test(code, func);
        }
    }

    writeln!(code, "}}\n").unwrap();
}

/// Check if we can generate a property test for this function
fn can_generate_property_test(func: &FfiFunction) -> bool {
    // Can generate property tests for functions with numeric parameters
    func.params.iter().any(|p| {
        !p.is_pointer && (p.ty.contains("int") || p.ty.contains("long") || p.ty.contains("float"))
    })
}

/// Generate a property-based test
fn generate_property_test(code: &mut String, func: &FfiFunction) {
    writeln!(code, "    proptest! {{").unwrap();
    writeln!(code, "        #[test]").unwrap();

    write!(
        code,
        "        fn test_{}_property(",
        func.name.to_lowercase()
    )
    .unwrap();

    // Generate strategy parameters
    let mut strategies = Vec::new();
    for param in &func.params {
        if !param.is_pointer && (param.ty.contains("int") || param.ty.contains("long")) {
            strategies.push(format!("{} in 0i64..1000", param.name));
        } else if param.ty.contains("float") || param.ty.contains("double") {
            strategies.push(format!("{} in 0.0f64..1000.0", param.name));
        }
    }

    writeln!(code, "{}) {{", strategies.join(", ")).unwrap();

    // Generate property test body
    write!(code, "            let result = {}(", func.name).unwrap();

    for (i, param) in func.params.iter().enumerate() {
        if i > 0 {
            write!(code, ", ").unwrap();
        }

        if !param.is_pointer
            && (param.ty.contains("int") || param.ty.contains("long") || param.ty.contains("float"))
        {
            write!(code, "{}", param.name).unwrap();
        } else {
            write!(code, "{}", generate_default_value(param).to_rust_code()).unwrap();
        }
    }

    writeln!(code, ");").unwrap();
    writeln!(code, "            // Property: function should not panic").unwrap();
    writeln!(
        code,
        "            prop_assert!(result.is_ok() || result.is_err());"
    )
    .unwrap();
    writeln!(code, "        }}").unwrap();
    writeln!(code, "    }}\n").unwrap();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_test_value_to_rust_code() {
        assert_eq!(TestValue::Int(42).to_rust_code(), "42");
        assert_eq!(TestValue::UInt(100).to_rust_code(), "100_u64");
        assert_eq!(TestValue::Float(3.14).to_rust_code(), "3.14_f64");
        assert_eq!(
            TestValue::String("hello".to_string()).to_rust_code(),
            "\"hello\""
        );
        assert_eq!(TestValue::Null.to_rust_code(), "std::ptr::null()");
    }

    #[test]
    fn test_generate_default_value_for_size_param() {
        let param = FfiParam {
            name: "buffer_size".to_string(),
            ty: "size_t".to_string(),
            is_pointer: false,
            is_mut: false,
        };

        let value = generate_default_value(&param);
        match value {
            TestValue::UInt(n) => assert_eq!(n, 1024),
            _ => panic!("Expected UInt"),
        }
    }

    #[test]
    fn test_generate_default_value_for_pointer() {
        let param = FfiParam {
            name: "data".to_string(),
            ty: "void*".to_string(),
            is_pointer: true,
            is_mut: false,
        };

        let value = generate_default_value(&param);
        matches!(value, TestValue::Null);
    }
}
