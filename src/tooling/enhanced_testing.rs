//! Enhanced testing and validation support
//!
//! This module provides:
//! - Property-based testing integration (proptest)
//! - Fuzzing integration (cargo-fuzz)
//! - Mutation testing support
//! - Coverage tracking helpers

use crate::ffi::parser::{FfiFunction, FfiInfo};
use std::collections::HashSet;

/// Property test generator
pub struct PropertyTestGenerator {
    /// Functions to generate tests for
    functions: Vec<String>,
}

impl PropertyTestGenerator {
    /// Create new generator
    pub fn new() -> Self {
        Self {
            functions: Vec::new(),
        }
    }

    /// Analyze FFI info for property testing opportunities
    pub fn analyze(&mut self, ffi_info: &FfiInfo) {
        for func in &ffi_info.functions {
            // Functions with numeric parameters are good for property testing
            if Self::has_numeric_params(func) {
                self.functions.push(func.name.clone());
            }
        }
    }

    /// Check if function has numeric parameters
    fn has_numeric_params(func: &FfiFunction) -> bool {
        func.params.iter().any(|p| {
            p.ty.contains("int")
                || p.ty.contains("float")
                || p.ty.contains("double")
                || p.ty.contains("size")
        })
    }

    /// Generate property tests
    pub fn generate(&self) -> String {
        let mut output = String::from(
            r#"//! Property-based tests
//!
//! Run with: cargo test --features proptest

#[cfg(test)]
mod property_tests {
    use super::*;
    use proptest::prelude::*;

"#,
        );

        for func in &self.functions {
            output.push_str(&format!(
                r#"    proptest! {{
        #[test]
        fn test_{}_properties(x in 0i32..1000, y in 0i32..1000) {{
            // Property: function should not panic
            let result = std::panic::catch_unwind(|| {{
                // Call function here
            }});
            prop_assert!(result.is_ok());
        }}
    }}

"#,
                func
            ));
        }

        output.push_str("}\n");
        output
    }
}

impl Default for PropertyTestGenerator {
    fn default() -> Self {
        Self::new()
    }
}

/// Fuzz test generator
pub struct FuzzTestGenerator {
    /// Functions to fuzz
    fuzz_targets: Vec<String>,
}

impl FuzzTestGenerator {
    /// Create new generator
    pub fn new() -> Self {
        Self {
            fuzz_targets: Vec::new(),
        }
    }

    /// Analyze FFI info for fuzz testing
    pub fn analyze(&mut self, ffi_info: &FfiInfo) {
        for func in &ffi_info.functions {
            // Functions with buffer parameters need fuzzing
            if Self::needs_fuzzing(func) {
                self.fuzz_targets.push(func.name.clone());
            }
        }
    }

    /// Check if function needs fuzzing
    fn needs_fuzzing(func: &FfiFunction) -> bool {
        // Functions with pointer parameters and size parameters
        let has_pointer = func.params.iter().any(|p| p.is_pointer);
        let has_size = func
            .params
            .iter()
            .any(|p| p.ty.contains("size") || p.name.contains("len"));

        has_pointer && has_size
    }

    /// Generate fuzz target
    pub fn generate_fuzz_target(&self, function_name: &str) -> String {
        format!(
            r#"#![no_main]
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {{
    // TODO: Call {} with fuzzed data
    // Ensure proper bounds checking
}});
"#,
            function_name
        )
    }

    /// Generate Cargo.toml for fuzz targets
    pub fn generate_fuzz_cargo_toml(&self) -> String {
        r#"[package]
name = "fuzz"
version = "0.0.0"
edition = "2021"
publish = false

[dependencies]
libfuzzer-sys = "0.4"

[dependencies.my-bindings]
path = ".."

[[bin]]
name = "fuzz_target_1"
path = "fuzz_targets/fuzz_target_1.rs"
test = false
doc = false
"#
        .to_string()
    }
}

impl Default for FuzzTestGenerator {
    fn default() -> Self {
        Self::new()
    }
}

/// Test coverage helper
pub struct CoverageHelper;

impl CoverageHelper {
    /// Generate coverage collection script
    pub fn generate_coverage_script() -> String {
        r#"#!/bin/bash
# Coverage collection script
# Requires: cargo-tarpaulin or cargo-llvm-cov

# Using tarpaulin
cargo tarpaulin --out Html --output-dir coverage

# Or using llvm-cov
# cargo llvm-cov --html --output-dir coverage

echo "Coverage report generated in coverage/"
"#
        .to_string()
    }

    /// Generate coverage configuration
    pub fn generate_coverage_config() -> String {
        r#"# tarpaulin configuration
[coverage]
exclude-files = [
    "tests/*",
    "benches/*",
    "*/build.rs"
]

[report]
out-type = ["Html", "Lcov"]
output-dir = "coverage"
"#
        .to_string()
    }
}

/// Mutation testing helper
pub struct MutationTestHelper;

impl MutationTestHelper {
    /// Generate mutation testing configuration
    pub fn generate_mutagen_config() -> String {
        r#"# cargo-mutants configuration
[mutants]
# Timeout for each test run (seconds)
timeout = 300

# Patterns to exclude from mutation
exclude_globs = [
    "tests/*",
    "benches/*"
]

# Minimum number of tests that must pass
minimum_test_pass = 90
"#
        .to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ffi::parser::FfiParam;
    use std::collections::HashMap;

    #[test]
    fn test_property_test_generator_new() {
        let mut generator = PropertyTestGenerator::new();
        assert_eq!(generator.functions.len(), 0);

        let ffi_info = FfiInfo {
            functions: vec![FfiFunction {
                name: "add".to_string(),
                params: vec![
                    FfiParam {
                        name: "a".to_string(),
                        ty: "int".to_string(),
                        is_pointer: false,
                        is_mut: false,
                    },
                    FfiParam {
                        name: "b".to_string(),
                        ty: "int".to_string(),
                        is_pointer: false,
                        is_mut: false,
                    },
                ],
                return_type: "int".to_string(),
                docs: None,
            }],
            types: vec![],
            enums: vec![],
            constants: vec![],
            opaque_types: vec![],
            dependencies: vec![],
            type_aliases: HashMap::new(),
        };

        generator.analyze(&ffi_info);
        assert_eq!(generator.functions.len(), 1);
        assert_eq!(generator.functions[0], "add");
    }

    #[test]
    fn test_property_test_generation() {
        let mut generator = PropertyTestGenerator::new();
        generator.functions.push("example_func".to_string());

        let output = generator.generate();
        assert!(output.contains("property_tests"));
        assert!(output.contains("test_example_func_properties"));
        assert!(output.contains("proptest"));
    }

    #[test]
    fn test_fuzz_test_generator_new() {
        let mut generator = FuzzTestGenerator::new();
        assert_eq!(generator.fuzz_targets.len(), 0);

        let ffi_info = FfiInfo {
            functions: vec![FfiFunction {
                name: "process_buffer".to_string(),
                params: vec![
                    FfiParam {
                        name: "buf".to_string(),
                        ty: "u8".to_string(),
                        is_pointer: true,
                        is_mut: false,
                    },
                    FfiParam {
                        name: "size".to_string(),
                        ty: "size_t".to_string(),
                        is_pointer: false,
                        is_mut: false,
                    },
                ],
                return_type: "void".to_string(),
                docs: None,
            }],
            types: vec![],
            enums: vec![],
            constants: vec![],
            opaque_types: vec![],
            dependencies: vec![],
            type_aliases: HashMap::new(),
        };

        generator.analyze(&ffi_info);
        assert_eq!(generator.fuzz_targets.len(), 1);
    }

    #[test]
    fn test_fuzz_target_generation() {
        let generator = FuzzTestGenerator::new();
        let target = generator.generate_fuzz_target("test_func");

        assert!(target.contains("fuzz_target"));
        assert!(target.contains("test_func"));
        assert!(target.contains("libfuzzer_sys"));
    }

    #[test]
    fn test_coverage_script() {
        let script = CoverageHelper::generate_coverage_script();
        assert!(script.contains("cargo tarpaulin"));
        assert!(script.contains("coverage"));
    }

    #[test]
    fn test_mutation_config() {
        let config = MutationTestHelper::generate_mutagen_config();
        assert!(config.contains("cargo-mutants"));
        assert!(config.contains("timeout"));
    }
}
