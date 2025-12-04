//! Example pattern analysis from test files and documentation
//!
//! This module analyzes test code and examples to extract
//! common usage patterns and idiomatic API usage.

use crate::ffi::FfiInfo;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use tracing::{debug, info};

/// Extracted usage patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsagePatterns {
    /// Common API sequences
    pub sequences: Vec<PatternSequence>,
    /// Function usage examples
    pub examples: HashMap<String, Vec<CodeExample>>,
    /// Best practices extracted from examples
    pub best_practices: Vec<BestPractice>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternSequence {
    /// Description of what this sequence does
    pub description: String,
    /// Ordered list of function calls
    pub functions: Vec<String>,
    /// How often this pattern appears
    pub frequency: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeExample {
    /// Example code snippet
    pub code: String,
    /// What the example demonstrates
    pub description: String,
    /// Source of the example (test file, doc, etc.)
    pub source: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BestPractice {
    /// The practice description
    pub description: String,
    /// Example demonstrating the practice
    pub example: Option<String>,
    /// Why this is important
    pub rationale: Option<String>,
}

/// Analyzes examples and test code for patterns
pub struct ExampleAnalyzer;

impl ExampleAnalyzer {
    pub fn new() -> Self {
        Self
    }

    /// Analyze test files for usage patterns
    pub fn analyze(&self, ffi_info: &FfiInfo, test_dir: Option<&Path>) -> UsagePatterns {
        let mut sequences = Vec::new();
        let mut examples = HashMap::new();
        let mut best_practices = Vec::new();

        // Extract patterns from function documentation
        for func in &ffi_info.functions {
            if let Some(doc) = &func.docs {
                if let Some(example) = self.extract_example_from_doc(doc, &func.name) {
                    examples
                        .entry(func.name.clone())
                        .or_insert_with(Vec::new)
                        .push(example);
                }
            }
        }

        // If test directory provided, analyze test files
        if let Some(test_path) = test_dir {
            let test_patterns = self.analyze_test_directory(test_path, ffi_info);
            sequences.extend(test_patterns.sequences);
            for (func, ex) in test_patterns.examples {
                examples.entry(func).or_insert_with(Vec::new).extend(ex);
            }
        }

        // Extract best practices
        best_practices.extend(self.extract_best_practices(ffi_info));

        info!(
            "Example analysis: {} sequences, {} function examples, {} best practices",
            sequences.len(),
            examples.len(),
            best_practices.len()
        );

        UsagePatterns {
            sequences,
            examples,
            best_practices,
        }
    }

    /// Extract example from function documentation
    fn extract_example_from_doc(&self, doc: &str, func_name: &str) -> Option<CodeExample> {
        // Look for code blocks in documentation
        let code_block_re = Regex::new(r"```(?:c|cpp)?\s*\n(.*?)\n```").ok()?;

        for cap in code_block_re.captures_iter(doc) {
            if let Some(code) = cap.get(1) {
                let code_text = code.as_str();
                // Only include if it mentions the function
                if code_text.contains(func_name) {
                    return Some(CodeExample {
                        code: code_text.to_string(),
                        description: format!("Example usage of {}", func_name),
                        source: "documentation".to_string(),
                    });
                }
            }
        }

        // Look for @example tags
        if doc.contains("@example") || doc.contains("Example:") {
            let lines: Vec<&str> = doc.lines().collect();
            let mut in_example = false;
            let mut example_lines = Vec::new();

            for line in lines {
                if line.contains("@example") || line.contains("Example:") {
                    in_example = true;
                    continue;
                }
                if in_example {
                    if line.trim().is_empty() && !example_lines.is_empty() {
                        break;
                    }
                    example_lines.push(line.trim());
                }
            }

            if !example_lines.is_empty() {
                return Some(CodeExample {
                    code: example_lines.join("\n"),
                    description: format!("Example from {} documentation", func_name),
                    source: "documentation".to_string(),
                });
            }
        }

        None
    }

    /// Analyze test directory (stub for now)
    fn analyze_test_directory(&self, _test_path: &Path, _ffi_info: &FfiInfo) -> UsagePatterns {
        // TODO: Actually parse test files
        debug!("Test directory analysis not yet implemented");
        UsagePatterns {
            sequences: Vec::new(),
            examples: HashMap::new(),
            best_practices: Vec::new(),
        }
    }

    /// Extract best practices from FFI
    fn extract_best_practices(&self, ffi_info: &FfiInfo) -> Vec<BestPractice> {
        let mut practices = Vec::new();

        // Check for RAII patterns
        let has_create = ffi_info
            .functions
            .iter()
            .any(|f| f.name.to_lowercase().contains("create"));
        let has_destroy = ffi_info
            .functions
            .iter()
            .any(|f| f.name.to_lowercase().contains("destroy"));

        if has_create && has_destroy {
            practices.push(BestPractice {
                description: "Always pair create/destroy calls".to_string(),
                example: Some("Use RAII wrappers to ensure proper cleanup".to_string()),
                rationale: Some("Prevents resource leaks".to_string()),
            });
        }

        // Check for error handling
        let has_status_return = ffi_info
            .functions
            .iter()
            .any(|f| f.return_type.to_lowercase().contains("status"));

        if has_status_return {
            practices.push(BestPractice {
                description: "Always check return status codes".to_string(),
                example: Some("Use Result<T, Error> in Rust wrappers".to_string()),
                rationale: Some("Ensures errors are not silently ignored".to_string()),
            });
        }

        // Check for thread safety
        if ffi_info
            .functions
            .iter()
            .any(|f| f.name.to_lowercase().contains("thread"))
        {
            practices.push(BestPractice {
                description: "Consider thread safety requirements".to_string(),
                example: Some("Use appropriate synchronization primitives".to_string()),
                rationale: Some("Prevents race conditions and data races".to_string()),
            });
        }

        practices
    }
}

/// Generate documentation from usage patterns
pub fn generate_pattern_docs(patterns: &UsagePatterns) -> String {
    let mut output = String::new();

    if !patterns.sequences.is_empty() {
        output.push_str("# Common Usage Patterns\n\n");
        for seq in &patterns.sequences {
            output.push_str(&format!("## {}\n\n", seq.description));
            output.push_str("```rust\n");
            for func in &seq.functions {
                output.push_str(&format!("{}();\n", func));
            }
            output.push_str("```\n\n");
        }
    }

    if !patterns.best_practices.is_empty() {
        output.push_str("# Best Practices\n\n");
        for practice in &patterns.best_practices {
            output.push_str(&format!("- **{}**", practice.description));
            if let Some(rationale) = &practice.rationale {
                output.push_str(&format!(": {}", rationale));
            }
            output.push_str("\n");
        }
        output.push_str("\n");
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ffi::FfiFunction;

    #[test]
    fn test_example_extraction_from_doc() {
        let doc = r#"
        Creates a new handle.
        
        Example:
        ```c
        Handle h;
        createHandle(&h);
        ```
        "#;

        let analyzer = ExampleAnalyzer::new();
        let example = analyzer.extract_example_from_doc(doc, "createHandle");

        assert!(example.is_some());
        let ex = example.unwrap();
        assert!(ex.code.contains("createHandle"));
    }

    #[test]
    fn test_best_practice_extraction() {
        let mut ffi_info = FfiInfo::default();

        ffi_info.functions.push(FfiFunction {
            name: "createResource".to_string(),
            params: vec![],
            return_type: "int".to_string(),
            docs: None,
        });

        ffi_info.functions.push(FfiFunction {
            name: "destroyResource".to_string(),
            params: vec![],
            return_type: "int".to_string(),
            docs: None,
        });

        let analyzer = ExampleAnalyzer::new();
        let practices = analyzer.extract_best_practices(&ffi_info);

        assert!(!practices.is_empty());
        assert!(
            practices
                .iter()
                .any(|p| p.description.contains("create/destroy"))
        );
    }

    #[test]
    fn test_pattern_doc_generation() {
        let patterns = UsagePatterns {
            sequences: vec![],
            examples: HashMap::new(),
            best_practices: vec![BestPractice {
                description: "Test practice".to_string(),
                example: None,
                rationale: Some("Test rationale".to_string()),
            }],
        };

        let docs = generate_pattern_docs(&patterns);
        assert!(docs.contains("Best Practices"));
        assert!(docs.contains("Test practice"));
    }

    #[test]
    fn test_empty_analysis() {
        let ffi_info = FfiInfo::default();
        let analyzer = ExampleAnalyzer::new();
        let patterns = analyzer.analyze(&ffi_info, None);

        assert!(patterns.sequences.is_empty());
        assert!(patterns.examples.is_empty());
    }
}
