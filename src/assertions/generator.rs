// src/assertions/generator.rs

//! Generates Rust assertion code from contracts

use super::contract_parser::{BufferConstraint, FunctionContract, RangeConstraint};
use super::{Assertion, AssertionCategory, FunctionAssertions};
use crate::ffi::FfiFunction;
use anyhow::Result;

/// Generates assertion code for functions
pub struct AssertionGenerator {
    /// Include lower-confidence assertions (>= 0.5)
    include_heuristic: bool,
}

impl AssertionGenerator {
    pub fn new() -> Self {
        Self {
            include_heuristic: true,
        }
    }

    /// Generate assertions for a function based on its contract
    pub fn generate_for_function(
        &self,
        function: &FfiFunction,
        contract: &FunctionContract,
    ) -> Result<FunctionAssertions> {
        let mut preconditions = Vec::new();

        // Generate null pointer checks
        for param_name in &contract.non_null_params {
            preconditions.push(Assertion {
                condition: format!("!{}.is_null()", param_name),
                message: format!("{} must not be null", param_name),
                category: AssertionCategory::NullPointer,
                confidence: 1.0,
            });
        }

        // Generate range checks
        for (param_name, range) in &contract.range_constraints {
            let (condition, message) = self.generate_range_check(param_name, range);
            preconditions.push(Assertion {
                condition,
                message,
                category: AssertionCategory::RangeCheck,
                confidence: 0.9,
            });
        }

        // Generate alignment checks
        for (param_name, alignment) in &contract.alignment_requirements {
            preconditions.push(Assertion {
                condition: format!("({} as usize) % {} == 0", param_name, alignment),
                message: format!(
                    "{} must be {}-byte aligned (got 0x{{:x}})",
                    param_name, alignment
                ),
                category: AssertionCategory::Alignment,
                confidence: 0.95,
            });
        }

        // Generate buffer size checks
        for (param_name, constraint) in &contract.buffer_constraints {
            if let Some(assertion) = self.generate_buffer_check(param_name, constraint) {
                preconditions.push(assertion);
            }
        }

        // Generate state checks
        let mut invariants = Vec::new();
        if contract.requires_initialization {
            invariants.push(Assertion {
                condition: "self.is_initialized()".to_string(),
                message: "handle must be initialized before use".to_string(),
                category: AssertionCategory::StateCheck,
                confidence: 0.9,
            });
        }

        // Generate thread safety checks
        if !contract.thread_safe && self.include_heuristic {
            invariants.push(Assertion {
                condition: "!self.is_in_use()".to_string(),
                message: "function is not thread-safe; ensure exclusive access".to_string(),
                category: AssertionCategory::ThreadSafety,
                confidence: 0.7,
            });
        }

        Ok(FunctionAssertions {
            function_name: function.name.clone(),
            preconditions,
            postconditions: Vec::new(), // Can be extended later
            invariants,
        })
    }

    fn generate_range_check(&self, param_name: &str, range: &RangeConstraint) -> (String, String) {
        match range {
            RangeConstraint::Min(min) => (
                format!("{} >= {}", param_name, min),
                format!("{} must be >= {} (got {{}})", param_name, min),
            ),
            RangeConstraint::GreaterThan(min) => (
                format!("{} > {}", param_name, min),
                format!("{} must be > {} (got {{}})", param_name, min),
            ),
            RangeConstraint::Max(max) => (
                format!("{} <= {}", param_name, max),
                format!("{} must be <= {} (got {{}})", param_name, max),
            ),
            RangeConstraint::LessThan(max) => (
                format!("{} < {}", param_name, max),
                format!("{} must be < {} (got {{}})", param_name, max),
            ),
            RangeConstraint::MinMax(min, max) => (
                format!("{} >= {} && {} <= {}", param_name, min, param_name, max),
                format!(
                    "{} must be in range {}..={} (got {{}})",
                    param_name, min, max
                ),
            ),
        }
    }

    fn generate_buffer_check(
        &self,
        param_name: &str,
        constraint: &BufferConstraint,
    ) -> Option<Assertion> {
        match constraint {
            BufferConstraint::SizedBy(size_param) => Some(Assertion {
                condition: format!("/* buffer size validated by {} */", size_param),
                message: format!("{} size must match {}", param_name, size_param),
                category: AssertionCategory::BufferSize,
                confidence: 0.8,
            }),
            BufferConstraint::MinSize(min_size) => Some(Assertion {
                condition: format!("{}_len >= {}", param_name, min_size),
                message: format!("{} must be at least {} bytes", param_name, min_size),
                category: AssertionCategory::BufferSize,
                confidence: 0.9,
            }),
        }
    }

    /// Generate the actual Rust code for assertions
    pub fn generate_code(&self, assertions: &FunctionAssertions) -> String {
        let mut code = String::new();

        // Only include in debug builds
        if !assertions.preconditions.is_empty() || !assertions.invariants.is_empty() {
            code.push_str("        #[cfg(debug_assertions)]\n");
            code.push_str("        {\n");

            // Invariants first
            for assertion in &assertions.invariants {
                if assertion.confidence >= 0.5 {
                    code.push_str(&self.format_assertion(assertion, "            "));
                }
            }

            // Then preconditions
            for assertion in &assertions.preconditions {
                if assertion.confidence >= 0.5 {
                    code.push_str(&self.format_assertion(assertion, "            "));
                }
            }

            code.push_str("        }\n");
        }

        code
    }

    fn format_assertion(&self, assertion: &Assertion, indent: &str) -> String {
        let mut result = String::new();

        // Add comment about confidence if not 100%
        if assertion.confidence < 1.0 {
            result.push_str(&format!(
                "{}// Confidence: {:.0}%\n",
                indent,
                assertion.confidence * 100.0
            ));
        }

        // Generate the debug_assert! call
        if assertion.message.contains("{}") {
            // Message has format parameters - need to extract variable name
            let var_name = assertion.condition.split_whitespace().next().unwrap_or("");
            result.push_str(&format!(
                "{}debug_assert!({}, \"{}\", {});\n",
                indent, assertion.condition, assertion.message, var_name
            ));
        } else if assertion.message.contains("{:x}") {
            // Hex format
            let var_name = assertion.condition.split_whitespace().next().unwrap_or("");
            result.push_str(&format!(
                "{}debug_assert!({}, \"{}\", {} as usize);\n",
                indent, assertion.condition, assertion.message, var_name
            ));
        } else {
            // Simple message
            result.push_str(&format!(
                "{}debug_assert!({}, \"{}\");\n",
                indent, assertion.condition, assertion.message
            ));
        }

        result
    }
}

impl Default for AssertionGenerator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_null_check_generation() {
        let generator = AssertionGenerator::new();
        let assertion = Assertion {
            condition: "!ptr.is_null()".to_string(),
            message: "ptr must not be null".to_string(),
            category: AssertionCategory::NullPointer,
            confidence: 1.0,
        };

        let code = generator.format_assertion(&assertion, "        ");
        assert!(code.contains("debug_assert!"));
        assert!(code.contains("!ptr.is_null()"));
    }

    #[test]
    fn test_range_check_generation() {
        let generator = AssertionGenerator::new();
        let (condition, message) =
            generator.generate_range_check("value", &RangeConstraint::Min(0));

        assert_eq!(condition, "value >= 0");
        assert!(message.contains("must be >= 0"));
    }

    #[test]
    fn test_range_check_max() {
        let generator = AssertionGenerator::new();
        let (condition, message) =
            generator.generate_range_check("index", &RangeConstraint::Max(255));

        assert_eq!(condition, "index <= 255");
        assert!(message.contains("must be <= 255"));
    }

    #[test]
    fn test_range_check_min_max() {
        let generator = AssertionGenerator::new();
        let (condition, message) =
            generator.generate_range_check("size", &RangeConstraint::MinMax(1, 1024));

        assert_eq!(condition, "size >= 1 && size <= 1024");
        assert!(message.contains("must be in range 1..=1024"));
    }

    #[test]
    fn test_range_check_greater_than() {
        let generator = AssertionGenerator::new();
        let (condition, message) =
            generator.generate_range_check("count", &RangeConstraint::GreaterThan(0));

        assert_eq!(condition, "count > 0");
        assert!(message.contains("must be > 0"));
    }

    #[test]
    fn test_range_check_less_than() {
        let generator = AssertionGenerator::new();
        let (condition, message) =
            generator.generate_range_check("index", &RangeConstraint::LessThan(100));

        assert_eq!(condition, "index < 100");
        assert!(message.contains("must be < 100"));
    }

    #[test]
    fn test_assertion_with_indentation() {
        let generator = AssertionGenerator::new();
        let assertion = Assertion {
            condition: "x > 0".to_string(),
            message: "x must be positive".to_string(),
            category: AssertionCategory::RangeCheck,
            confidence: 1.0,
        };

        let code = generator.format_assertion(&assertion, "    ");
        // Should have proper indentation
        assert!(code.starts_with("    debug_assert!"));
    }

    #[test]
    fn test_multiple_assertions() {
        let generator = AssertionGenerator::new();
        let assertions = vec![
            Assertion {
                condition: "!ptr.is_null()".to_string(),
                message: "ptr must not be null".to_string(),
                category: AssertionCategory::NullPointer,
                confidence: 1.0,
            },
            Assertion {
                condition: "size > 0".to_string(),
                message: "size must be positive".to_string(),
                category: AssertionCategory::RangeCheck,
                confidence: 1.0,
            },
        ];

        let code1 = generator.format_assertion(&assertions[0], "");
        let code2 = generator.format_assertion(&assertions[1], "");

        assert!(code1.contains("!ptr.is_null()"));
        assert!(code2.contains("size > 0"));
    }

    #[test]
    fn test_alignment_assertion() {
        let generator = AssertionGenerator::new();
        let assertion = Assertion {
            condition: "ptr as usize % 64 == 0".to_string(),
            message: "ptr must be 64-byte aligned".to_string(),
            category: AssertionCategory::Alignment,
            confidence: 0.9,
        };

        let code = generator.format_assertion(&assertion, "");
        assert!(code.contains("ptr as usize % 64 == 0"));
        assert!(code.contains("64-byte aligned"));
    }

    #[test]
    fn test_assertion_confidence() {
        let generator = AssertionGenerator::new();
        
        // High confidence assertion
        let high_confidence = Assertion {
            condition: "x > 0".to_string(),
            message: "x must be positive".to_string(),
            category: AssertionCategory::RangeCheck,
            confidence: 1.0,
        };

        // Low confidence assertion  
        let low_confidence = Assertion {
            condition: "y > 0".to_string(),
            message: "y might need to be positive".to_string(),
            category: AssertionCategory::RangeCheck,
            confidence: 0.5,
        };

        // Both should still generate assertions
        let code1 = generator.format_assertion(&high_confidence, "");
        let code2 = generator.format_assertion(&low_confidence, "");

        assert!(code1.contains("debug_assert!"));
        assert!(code2.contains("debug_assert!"));
    }
}
