//! Test case mining for extracting valid usage patterns from examples and tests.
//!
//! This module analyzes test files, example code, and benchmarks to extract:
//! - Valid parameter values and configurations
//! - Common usage patterns
//! - Edge cases and boundary conditions
//! - Performance-tested configurations
//! - Common mistakes (from issue trackers)

use std::collections::HashMap;

/// Classification of a usage example
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExampleType {
    /// Basic, common usage pattern
    Basic,
    /// Advanced or specialized usage
    Advanced,
    /// Edge case or boundary condition
    EdgeCase,
    /// Performance-optimized configuration
    Performance,
    /// Common mistake to avoid
    AntiPattern,
    /// Minimal valid configuration
    Minimal,
    /// Maximum tested configuration
    Maximum,
}

/// Source of a usage example
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExampleSource {
    /// From official library examples
    OfficialExample,
    /// From unit tests
    UnitTest,
    /// From integration tests
    IntegrationTest,
    /// From benchmarks
    Benchmark,
    /// From documentation
    Documentation,
    /// From tutorial code
    Tutorial,
}

/// A single parameter value used in an example
#[derive(Debug, Clone, PartialEq)]
pub struct ParameterValue {
    /// Parameter name
    pub name: String,
    /// Value as string (e.g., "42", "DataType::Float", "&[1, 3, 224, 224]")
    pub value: String,
    /// Frequency this value appears (for statistical analysis)
    pub frequency: usize,
    /// Optional comment explaining the value
    pub comment: Option<String>,
}

/// A complete usage example extracted from code
#[derive(Debug, Clone, PartialEq)]
pub struct UsageExample {
    /// Function or method being called
    pub function_name: String,
    /// Type of example
    pub example_type: ExampleType,
    /// Source of the example
    pub source: ExampleSource,
    /// Parameter values used
    pub parameters: Vec<ParameterValue>,
    /// Complete code snippet
    pub code_snippet: String,
    /// Explanation or context
    pub description: Option<String>,
    /// Whether this example compiled/ran successfully
    pub verified: bool,
}

/// Statistics about parameter usage across examples
#[derive(Debug, Clone, PartialEq)]
pub struct ParameterStatistics {
    /// Parameter name
    pub name: String,
    /// All observed values with frequencies
    pub values: HashMap<String, usize>,
    /// Most common value
    pub most_common: String,
    /// Percentage of most common value (0.0-1.0)
    pub most_common_percentage: f64,
    /// Typical range (for numeric parameters)
    pub range: Option<(String, String)>,
}

/// Complete analysis of test cases and examples for a function
#[derive(Debug, Clone, PartialEq)]
pub struct TestCaseInfo {
    /// Function name
    pub function_name: String,
    /// All extracted examples
    pub examples: Vec<UsageExample>,
    /// Statistical analysis of parameters
    pub parameter_stats: Vec<ParameterStatistics>,
    /// Common mistakes detected
    pub anti_patterns: Vec<UsageExample>,
    /// Overall confidence in the analysis
    pub confidence: f64,
}

impl TestCaseInfo {
    /// Create new empty test case info
    pub fn new(function_name: String) -> Self {
        Self {
            function_name,
            examples: Vec::new(),
            parameter_stats: Vec::new(),
            anti_patterns: Vec::new(),
            confidence: 0.0,
        }
    }

    /// Check if any examples were found
    pub fn has_examples(&self) -> bool {
        !self.examples.is_empty()
    }

    /// Get examples of a specific type
    pub fn examples_by_type(&self, example_type: &ExampleType) -> Vec<&UsageExample> {
        self.examples
            .iter()
            .filter(|e| &e.example_type == example_type)
            .collect()
    }

    /// Get examples from a specific source
    pub fn examples_by_source(&self, source: &ExampleSource) -> Vec<&UsageExample> {
        self.examples
            .iter()
            .filter(|e| &e.source == source)
            .collect()
    }

    /// Get the most representative basic example
    pub fn primary_example(&self) -> Option<&UsageExample> {
        // Prefer official examples, then unit tests
        self.examples_by_source(&ExampleSource::OfficialExample)
            .into_iter()
            .find(|e| e.example_type == ExampleType::Basic)
            .or_else(|| self.examples_by_type(&ExampleType::Basic).first().copied())
    }

    /// Generate documentation from examples (for use in FunctionContext)
    pub fn generate_example_docs(&self) -> String {
        let mut docs = String::new();

        if !self.has_examples() {
            return docs;
        }

        docs.push_str("/// # Example Usage\n");
        docs.push_str("///\n");

        // Primary example (basic usage)
        if let Some(primary) = self.primary_example() {
            docs.push_str("/// ## Basic Usage\n");
            docs.push_str("/// ```rust\n");
            for line in primary.code_snippet.lines() {
                docs.push_str(&format!("/// {}\n", line));
            }
            docs.push_str("/// ```\n");
            docs.push_str("///\n");
        }

        // Parameter statistics
        if !self.parameter_stats.is_empty() {
            docs.push_str("/// ## Common Values\n");
            docs.push_str("///\n");
            for stat in &self.parameter_stats {
                let percentage = (stat.most_common_percentage * 100.0) as usize;
                docs.push_str(&format!(
                    "/// - `{}`: Most common value is `{}` ({}% of examples)\n",
                    stat.name, stat.most_common, percentage
                ));
            }
            docs.push_str("///\n");
        }

        // Performance examples
        let perf_examples = self.examples_by_type(&ExampleType::Performance);
        if !perf_examples.is_empty() {
            docs.push_str("/// ## Performance-Optimized Configuration\n");
            docs.push_str("/// ```rust\n");
            for line in perf_examples[0].code_snippet.lines() {
                docs.push_str(&format!("/// {}\n", line));
            }
            docs.push_str("/// ```\n");
            docs.push_str("///\n");
        }

        // Edge cases
        let edge_examples = self.examples_by_type(&ExampleType::EdgeCase);
        if !edge_examples.is_empty() {
            docs.push_str("/// ## Edge Cases\n");
            for example in edge_examples.iter().take(2) {
                // Limit to 2 edge cases
                docs.push_str("/// ```rust\n");
                for line in example.code_snippet.lines() {
                    docs.push_str(&format!("/// {}\n", line));
                }
                docs.push_str("/// ```\n");
            }
            docs.push_str("///\n");
        }

        // Anti-patterns
        if !self.anti_patterns.is_empty() {
            docs.push_str("/// ## Common Mistakes\n");
            docs.push_str("///\n");
            for anti in self.anti_patterns.iter().take(2) {
                // Limit to 2 anti-patterns
                docs.push_str("/// ```rust\n");
                for line in anti.code_snippet.lines() {
                    docs.push_str(&format!("/// {}\n", line));
                }
                docs.push_str("/// ```\n");
            }
        }

        docs
    }
}

impl Default for TestCaseInfo {
    fn default() -> Self {
        Self::new(String::new())
    }
}

/// Analyzer for mining test cases and usage examples
#[derive(Debug)]
pub struct TestCaseMiner {
    /// Cache of analyzed functions
    cache: HashMap<String, TestCaseInfo>,
}

impl TestCaseMiner {
    /// Create a new test case miner
    pub fn new() -> Self {
        Self {
            cache: HashMap::new(),
        }
    }

    /// Mine test cases from code snippets
    ///
    /// # Arguments
    /// * `function_name` - Name of the function to analyze
    /// * `code_snippets` - Collection of code snippets to analyze
    pub fn mine(
        &mut self,
        function_name: &str,
        code_snippets: &[(String, ExampleSource)],
    ) -> TestCaseInfo {
        // Check cache first
        if let Some(cached) = self.cache.get(function_name) {
            return cached.clone();
        }

        let mut info = TestCaseInfo::new(function_name.to_string());

        // Extract examples from each snippet
        for (snippet, source) in code_snippets {
            if let Some(example) = self.extract_example(function_name, snippet, source) {
                info.examples.push(example);
            }
        }

        // Calculate parameter statistics
        info.parameter_stats = self.calculate_parameter_stats(&info.examples);

        // Separate anti-patterns
        info.anti_patterns = info
            .examples
            .iter()
            .filter(|e| e.example_type == ExampleType::AntiPattern)
            .cloned()
            .collect();

        // Calculate confidence
        info.confidence = self.calculate_confidence(&info);

        // Cache the result
        self.cache.insert(function_name.to_string(), info.clone());

        info
    }

    /// Extract a usage example from a code snippet
    fn extract_example(
        &self,
        function_name: &str,
        code: &str,
        source: &ExampleSource,
    ) -> Option<UsageExample> {
        // Check if snippet contains the function
        if !code.contains(function_name) {
            return None;
        }

        // Determine example type based on source and content
        let example_type = self.classify_example(code, source);

        // Extract parameters (simplified - in real impl would use proper parsing)
        let parameters = self.extract_parameters(code);

        // Determine if example is verified (from tests that passed)
        let verified = matches!(
            source,
            ExampleSource::UnitTest | ExampleSource::IntegrationTest | ExampleSource::Benchmark
        );

        Some(UsageExample {
            function_name: function_name.to_string(),
            example_type,
            source: source.clone(),
            parameters,
            code_snippet: code.to_string(),
            description: None,
            verified,
        })
    }

    /// Classify the type of example based on content and source
    fn classify_example(&self, code: &str, source: &ExampleSource) -> ExampleType {
        let code_lower = code.to_lowercase();

        // Check for anti-patterns
        if code_lower.contains("wrong")
            || code_lower.contains("error")
            || code_lower.contains("bad")
        {
            return ExampleType::AntiPattern;
        }

        // Check for edge cases
        if code_lower.contains("edge")
            || code_lower.contains("boundary")
            || code_lower.contains("minimum")
            || code_lower.contains("maximum")
        {
            if code_lower.contains("minimum") || code_lower.contains("minimal") {
                return ExampleType::Minimal;
            }
            if code_lower.contains("maximum") || code_lower.contains("maximal") {
                return ExampleType::Maximum;
            }
            return ExampleType::EdgeCase;
        }

        // Check for performance examples
        if matches!(source, ExampleSource::Benchmark)
            || code_lower.contains("performance")
            || code_lower.contains("optimized")
            || code_lower.contains("fast")
        {
            return ExampleType::Performance;
        }

        // Check for advanced patterns
        if code_lower.contains("advanced") || code_lower.contains("complex") {
            return ExampleType::Advanced;
        }

        // Default to basic
        ExampleType::Basic
    }

    /// Extract parameter values from code (simplified implementation)
    fn extract_parameters(&self, code: &str) -> Vec<ParameterValue> {
        let mut parameters = Vec::new();

        // Simple pattern matching for common parameter patterns
        // In real implementation, would use proper AST parsing

        // Skip these common builder method names
        let skip_names = ["build", "new", "create", "unwrap", "expect", "ok"];

        // Look for builder patterns: .param_name(value)
        for line in code.lines() {
            if let Some(dot_idx) = line.find('.')
                && let Some(paren_idx) = line[dot_idx..].find('(') {
                    let start = dot_idx + 1;
                    let end = dot_idx + paren_idx;
                    let param_name = line[start..end].trim();

                    // Skip common builder finishers
                    if skip_names.contains(&param_name) {
                        continue;
                    }

                    if let Some(close_paren) = line[end..].find(')') {
                        let value_start = end + 1;
                        let value_end = end + close_paren;
                        let value = line[value_start..value_end].trim();

                        if !param_name.is_empty() && !value.is_empty() {
                            parameters.push(ParameterValue {
                                name: param_name.to_string(),
                                value: value.to_string(),
                                frequency: 1,
                                comment: None,
                            });
                        }
                    }
                }
        }

        parameters
    }

    /// Calculate statistical analysis of parameters across examples
    fn calculate_parameter_stats(&self, examples: &[UsageExample]) -> Vec<ParameterStatistics> {
        let mut stats_map: HashMap<String, HashMap<String, usize>> = HashMap::new();

        // Count occurrences of each parameter value
        for example in examples {
            // Skip anti-patterns for statistics
            if example.example_type == ExampleType::AntiPattern {
                continue;
            }

            for param in &example.parameters {
                stats_map
                    .entry(param.name.clone())
                    .or_default()
                    .entry(param.value.clone())
                    .and_modify(|count| *count += 1)
                    .or_insert(1);
            }
        }

        // Convert to ParameterStatistics
        let mut stats = Vec::new();
        for (name, values) in stats_map {
            if values.is_empty() {
                continue;
            }

            // Find most common value
            let (most_common, max_count) = values
                .iter()
                .max_by_key(|(_, count)| *count)
                .map(|(v, c)| (v.clone(), *c))
                .unwrap();

            let total_count: usize = values.values().sum();
            let percentage = max_count as f64 / total_count as f64;

            stats.push(ParameterStatistics {
                name,
                values,
                most_common,
                most_common_percentage: percentage,
                range: None, // Would be calculated for numeric types
            });
        }

        stats
    }

    /// Calculate confidence score based on quantity and quality of examples
    fn calculate_confidence(&self, info: &TestCaseInfo) -> f64 {
        if info.examples.is_empty() {
            return 0.0;
        }

        let mut confidence = 0.0;

        // Base confidence from number of examples
        let example_count = info.examples.len() as f64;
        confidence += (example_count / 10.0).min(0.5); // Up to 0.5 for having examples

        // Bonus for official examples
        let official_count = info
            .examples_by_source(&ExampleSource::OfficialExample)
            .len() as f64;
        confidence += (official_count * 0.15).min(0.3);

        // Bonus for verified examples (from tests)
        let verified_count = info.examples.iter().filter(|e| e.verified).count() as f64;
        confidence += (verified_count / example_count) * 0.2;

        confidence.min(1.0)
    }

    /// Generate comprehensive documentation from mined examples
    pub fn generate_docs(&self, info: &TestCaseInfo) -> String {
        let mut docs = String::new();

        if !info.has_examples() {
            return docs;
        }

        docs.push_str("/// # Example Usage\n");
        docs.push_str("///\n");

        // Primary example (basic usage)
        if let Some(primary) = info.primary_example() {
            docs.push_str("/// ## Basic Usage\n");
            docs.push_str("/// ```rust\n");
            for line in primary.code_snippet.lines() {
                docs.push_str(&format!("/// {}\n", line));
            }
            docs.push_str("/// ```\n");
            docs.push_str("///\n");
        }

        // Parameter statistics
        if !info.parameter_stats.is_empty() {
            docs.push_str("/// ## Common Values\n");
            docs.push_str("///\n");
            for stat in &info.parameter_stats {
                let percentage = (stat.most_common_percentage * 100.0) as usize;
                docs.push_str(&format!(
                    "/// - `{}`: Most common value is `{}` ({}% of examples)\n",
                    stat.name, stat.most_common, percentage
                ));
            }
            docs.push_str("///\n");
        }

        // Performance examples
        let perf_examples = info.examples_by_type(&ExampleType::Performance);
        if !perf_examples.is_empty() {
            docs.push_str("/// ## Performance-Optimized Configuration\n");
            docs.push_str("/// ```rust\n");
            for line in perf_examples[0].code_snippet.lines() {
                docs.push_str(&format!("/// {}\n", line));
            }
            docs.push_str("/// ```\n");
            docs.push_str("///\n");
        }

        // Edge cases
        let edge_examples = info.examples_by_type(&ExampleType::EdgeCase);
        if !edge_examples.is_empty() {
            docs.push_str("/// ## Edge Cases\n");
            for example in edge_examples {
                docs.push_str("/// ```rust\n");
                for line in example.code_snippet.lines() {
                    docs.push_str(&format!("/// {}\n", line));
                }
                docs.push_str("/// ```\n");
            }
            docs.push_str("///\n");
        }

        // Anti-patterns
        if !info.anti_patterns.is_empty() {
            docs.push_str("/// ## Common Mistakes\n");
            docs.push_str("///\n");
            for anti in &info.anti_patterns {
                docs.push_str("/// ```rust\n");
                for line in anti.code_snippet.lines() {
                    docs.push_str(&format!("/// {}\n", line));
                }
                docs.push_str("/// ```\n");
            }
        }

        docs
    }

    /// Clear the cache
    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }
}

impl Default for TestCaseMiner {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_classify_basic_example() {
        let miner = TestCaseMiner::new();
        let code = "let result = create_handle();";
        let example_type = miner.classify_example(code, &ExampleSource::OfficialExample);
        assert_eq!(example_type, ExampleType::Basic);
    }

    #[test]
    fn test_classify_anti_pattern() {
        let miner = TestCaseMiner::new();
        let code = "// WRONG: This will cause an error\nlet bad = create_handle(null);";
        let example_type = miner.classify_example(code, &ExampleSource::Documentation);
        assert_eq!(example_type, ExampleType::AntiPattern);
    }

    #[test]
    fn test_classify_performance() {
        let miner = TestCaseMiner::new();
        let code = "// Optimized for performance\nlet config = fast_config();";
        let example_type = miner.classify_example(code, &ExampleSource::Benchmark);
        assert_eq!(example_type, ExampleType::Performance);
    }

    #[test]
    fn test_classify_edge_case() {
        let miner = TestCaseMiner::new();
        let code = "// Edge case: minimum valid size\nlet min = create_buffer(1);";
        let example_type = miner.classify_example(code, &ExampleSource::UnitTest);
        assert_eq!(example_type, ExampleType::Minimal);
    }

    #[test]
    fn test_extract_parameters() {
        let miner = TestCaseMiner::new();
        let code = r#"
            let desc = TensorDescriptor::builder()
                .data_type(DataType::Float)
                .dimensions(&[1, 3, 224, 224])
                .build();
        "#;

        let params = miner.extract_parameters(code);
        // Should extract data_type and dimensions, but not build()
        assert_eq!(params.len(), 2);

        let data_type_param = params.iter().find(|p| p.name == "data_type");
        assert!(data_type_param.is_some());
        assert_eq!(data_type_param.unwrap().value, "DataType::Float");

        let dimensions_param = params.iter().find(|p| p.name == "dimensions");
        assert!(dimensions_param.is_some());
    }
    #[test]
    fn test_mine_with_multiple_snippets() {
        let mut miner = TestCaseMiner::new();

        let snippets = vec![
            ("let x = foo(42);".to_string(), ExampleSource::UnitTest),
            (
                "let y = foo(100);".to_string(),
                ExampleSource::IntegrationTest,
            ),
            (
                "let z = foo(42);".to_string(),
                ExampleSource::OfficialExample,
            ),
        ];

        let info = miner.mine("foo", &snippets);

        assert_eq!(info.examples.len(), 3);
        assert!(info.has_examples());
    }

    #[test]
    fn test_parameter_statistics() {
        let mut miner = TestCaseMiner::new();

        let snippets = vec![
            (
                "builder().size(100).build()".to_string(),
                ExampleSource::UnitTest,
            ),
            (
                "builder().size(100).build()".to_string(),
                ExampleSource::IntegrationTest,
            ),
            (
                "builder().size(200).build()".to_string(),
                ExampleSource::UnitTest,
            ),
        ];

        let info = miner.mine("builder", &snippets);

        assert!(!info.parameter_stats.is_empty());
        let size_stat = info.parameter_stats.iter().find(|s| s.name == "size");
        assert!(size_stat.is_some());

        let size_stat = size_stat.unwrap();
        assert_eq!(size_stat.most_common, "100");
        assert!((size_stat.most_common_percentage - 0.666).abs() < 0.01);
    }

    #[test]
    fn test_primary_example() {
        let mut miner = TestCaseMiner::new();

        let snippets = vec![
            ("advanced_usage()".to_string(), ExampleSource::UnitTest),
            ("basic_usage()".to_string(), ExampleSource::OfficialExample),
        ];

        let info = miner.mine("usage", &snippets);
        let primary = info.primary_example();

        assert!(primary.is_some());
        assert!(primary.unwrap().code_snippet.contains("basic"));
    }

    #[test]
    fn test_examples_by_type() {
        let mut miner = TestCaseMiner::new();

        let snippets = vec![
            ("test_basic()".to_string(), ExampleSource::OfficialExample),
            (
                "// WRONG: error\ntest_bad()".to_string(),
                ExampleSource::Documentation,
            ),
        ];

        let info = miner.mine("test", &snippets);

        let basics = info.examples_by_type(&ExampleType::Basic);
        assert_eq!(basics.len(), 1);

        let anti = info.examples_by_type(&ExampleType::AntiPattern);
        assert_eq!(anti.len(), 1);
    }

    #[test]
    fn test_confidence_calculation() {
        let mut miner = TestCaseMiner::new();

        // No examples = 0 confidence
        let empty_info = TestCaseInfo::new("test".to_string());
        assert_eq!(miner.calculate_confidence(&empty_info), 0.0);

        // With multiple examples including official = higher confidence
        let snippets = vec![
            ("test_example()".to_string(), ExampleSource::OfficialExample),
            ("test_one()".to_string(), ExampleSource::UnitTest),
            ("test_two()".to_string(), ExampleSource::IntegrationTest),
            ("test_three()".to_string(), ExampleSource::UnitTest),
        ];
        let info = miner.mine("test", &snippets);
        assert!(info.confidence > 0.5);
    }

    #[test]
    fn test_docs_generation() {
        let mut miner = TestCaseMiner::new();

        let snippets = vec![(
            "let x = create(100);".to_string(),
            ExampleSource::OfficialExample,
        )];

        let info = miner.mine("create", &snippets);
        let docs = miner.generate_docs(&info);

        assert!(docs.contains("# Example Usage"));
        assert!(docs.contains("## Basic Usage"));
        assert!(docs.contains("create(100)"));
    }

    #[test]
    fn test_cache() {
        let mut miner = TestCaseMiner::new();

        let snippets = vec![("example()".to_string(), ExampleSource::UnitTest)];

        let info1 = miner.mine("test", &snippets);
        let info2 = miner.mine("test", &snippets);

        assert_eq!(info1, info2);

        miner.clear_cache();
        let info3 = miner.mine("test", &snippets);
        assert_eq!(info1, info3);
    }
}
