//! Anti-pattern detection and common pitfall extraction.
//!
//! This module analyzes documentation, issue reports, and forum discussions to identify:
//! - Common usage mistakes
//! - Anti-patterns with corrections
//! - Frequently misunderstood APIs
//! - Error-prone function combinations

use std::collections::HashMap;

/// Represents a common pitfall or anti-pattern
#[derive(Debug, Clone, PartialEq)]
pub struct Pitfall {
    /// Title/summary of the pitfall
    pub title: String,
    /// Severity level
    pub severity: Severity,
    /// Wrong code example
    pub wrong_example: Option<String>,
    /// Correct code example
    pub correct_example: Option<String>,
    /// Detailed explanation
    pub explanation: String,
    /// Related issue numbers or references
    pub references: Vec<String>,
    /// Affected functions
    pub affected_functions: Vec<String>,
    /// Detection confidence (0.0-1.0)
    pub confidence: f64,
}

/// Severity of a pitfall
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Severity {
    /// Critical - causes crashes, UB, or data corruption
    Critical,
    /// High - causes incorrect behavior or memory leaks
    High,
    /// Medium - causes performance issues or confusing errors
    Medium,
    /// Low - minor issues or style concerns
    Low,
}

/// Information about common pitfalls for a function
#[derive(Debug, Clone, PartialEq)]
pub struct PitfallInfo {
    /// List of detected pitfalls
    pub pitfalls: Vec<Pitfall>,
    /// Overall confidence in the analysis
    pub confidence: f64,
}

impl PitfallInfo {
    /// Checks if there are any pitfalls
    pub fn has_pitfalls(&self) -> bool {
        !self.pitfalls.is_empty()
    }

    /// Gets pitfalls of a specific severity
    pub fn by_severity(&self, severity: Severity) -> Vec<&Pitfall> {
        self.pitfalls
            .iter()
            .filter(|p| p.severity == severity)
            .collect()
    }

    /// Generates documentation for all pitfalls
    pub fn generate_documentation(&self) -> String {
        if !self.has_pitfalls() {
            return String::new();
        }

        let mut docs = String::from("/// # Common Pitfalls\n");
        docs.push_str("///\n");

        // Sort by severity (Critical → High → Medium → Low)
        let mut sorted_pitfalls = self.pitfalls.clone();
        sorted_pitfalls.sort_by(|a, b| {
            let a_priority = match a.severity {
                Severity::Critical => 0,
                Severity::High => 1,
                Severity::Medium => 2,
                Severity::Low => 3,
            };
            let b_priority = match b.severity {
                Severity::Critical => 0,
                Severity::High => 1,
                Severity::Medium => 2,
                Severity::Low => 3,
            };
            a_priority.cmp(&b_priority)
        });

        for pitfall in sorted_pitfalls {
            // Emoji based on severity
            let emoji = match pitfall.severity {
                Severity::Critical => "🚨",
                Severity::High => "⚠️",
                Severity::Medium => "⚡",
                Severity::Low => "💡",
            };

            docs.push_str(&format!("/// {} **{}**\n", emoji, pitfall.title));
            docs.push_str("///\n");

            // Explanation
            for line in pitfall.explanation.lines() {
                docs.push_str(&format!("/// {}\n", line));
            }
            docs.push_str("///\n");

            // Wrong example
            if let Some(wrong) = &pitfall.wrong_example {
                docs.push_str("/// ```rust\n");
                docs.push_str("/// // ❌ WRONG\n");
                for line in wrong.lines() {
                    docs.push_str(&format!("/// {}\n", line));
                }
                docs.push_str("/// ```\n");
                docs.push_str("///\n");
            }

            // Correct example
            if let Some(correct) = &pitfall.correct_example {
                docs.push_str("/// ```rust\n");
                docs.push_str("/// // ✅ CORRECT\n");
                for line in correct.lines() {
                    docs.push_str(&format!("/// {}\n", line));
                }
                docs.push_str("/// ```\n");
                docs.push_str("///\n");
            }

            // References
            if !pitfall.references.is_empty() {
                docs.push_str(&format!(
                    "/// See also: {}\n",
                    pitfall.references.join(", ")
                ));
                docs.push_str("///\n");
            }
        }

        docs
    }
}

/// Main anti-pattern analyzer
#[derive(Debug)]
pub struct AntiPatternAnalyzer {
    /// Cache of analyzed pitfalls by function name
    cache: HashMap<String, PitfallInfo>,
}

impl AntiPatternAnalyzer {
    /// Creates a new anti-pattern analyzer
    pub fn new() -> Self {
        Self {
            cache: HashMap::new(),
        }
    }

    /// Analyzes function documentation and issue content for pitfalls
    ///
    /// # Arguments
    /// * `function_name` - Name of the function to analyze
    /// * `documentation` - Combined documentation from all sources
    /// * `issue_content` - Optional issue/forum content to analyze
    ///
    /// # Returns
    /// Information about detected pitfalls
    pub fn analyze(
        &mut self,
        function_name: &str,
        documentation: &str,
        issue_content: Option<&str>,
    ) -> PitfallInfo {
        // Check cache first
        if let Some(cached) = self.cache.get(function_name) {
            return cached.clone();
        }

        let mut pitfalls = Vec::new();

        // Analyze documentation for warnings and common mistakes
        pitfalls.extend(self.extract_from_documentation(function_name, documentation));

        // Analyze issue content if provided
        if let Some(issues) = issue_content {
            pitfalls.extend(self.extract_from_issues(function_name, issues));
        }

        // Calculate overall confidence
        let confidence = if pitfalls.is_empty() {
            0.0
        } else {
            pitfalls.iter().map(|p| p.confidence).sum::<f64>() / pitfalls.len() as f64
        };

        let info = PitfallInfo {
            pitfalls,
            confidence,
        };

        // Cache the result
        self.cache.insert(function_name.to_string(), info.clone());
        info
    }

    /// Extracts pitfalls from documentation
    fn extract_from_documentation(&self, function_name: &str, docs: &str) -> Vec<Pitfall> {
        let mut pitfalls = Vec::new();
        let lower_docs = docs.to_lowercase();

        // Pattern 1: "Do NOT" warnings
        if (lower_docs.contains("do not") || lower_docs.contains("don't"))
            && let Some(pitfall) = self.parse_do_not_warning(function_name, docs) {
                pitfalls.push(pitfall);
            }

        // Pattern 2: "Must call" requirements
        if (lower_docs.contains("must call") || lower_docs.contains("must be called"))
            && let Some(pitfall) = self.parse_must_call_requirement(function_name, docs) {
                pitfalls.push(pitfall);
            }

        // Pattern 3: "Cannot" restrictions
        if (lower_docs.contains("cannot") || lower_docs.contains("can not"))
            && let Some(pitfall) = self.parse_cannot_restriction(function_name, docs) {
                pitfalls.push(pitfall);
            }

        // Pattern 4: Multiple initialization warnings
        if lower_docs.contains("multiple")
            && (lower_docs.contains("init") || lower_docs.contains("call"))
            && let Some(pitfall) = self.parse_multiple_init_warning(function_name, docs) {
                pitfalls.push(pitfall);
            }

        // Pattern 5: Memory leak warnings
        if (lower_docs.contains("leak") || lower_docs.contains("memory leak"))
            && let Some(pitfall) = self.parse_memory_leak_warning(function_name, docs) {
                pitfalls.push(pitfall);
            }

        // Pattern 6: Race condition warnings
        if (lower_docs.contains("race") || lower_docs.contains("synchronize"))
            && let Some(pitfall) = self.parse_race_condition_warning(function_name, docs) {
                pitfalls.push(pitfall);
            }

        // Pattern 7: Order dependencies (must call X before Y)
        if lower_docs.contains("before") && lower_docs.contains("call")
            && let Some(pitfall) = self.parse_order_dependency(function_name, docs) {
                pitfalls.push(pitfall);
            }

        pitfalls
    }

    /// Extracts pitfalls from issue content
    fn extract_from_issues(&self, function_name: &str, issues: &str) -> Vec<Pitfall> {
        let mut pitfalls = Vec::new();
        let lower_issues = issues.to_lowercase();

        // Look for issue numbers and associated problems
        // Pattern: "Issue #1234: description"
        for line in issues.lines() {
            if line.to_lowercase().contains("issue") && line.contains("#")
                && let Some(pitfall) = self.parse_issue_line(function_name, line, issues) {
                    pitfalls.push(pitfall);
                }
        }

        // Look for "frequently" or "common" mentions
        if (lower_issues.contains("frequently") || lower_issues.contains("common"))
            && let Some(pitfall) = self.parse_common_mistake(function_name, issues) {
                pitfalls.push(pitfall);
            }

        pitfalls
    }

    /// Parses "Do NOT" warnings from documentation
    fn parse_do_not_warning(&self, function_name: &str, docs: &str) -> Option<Pitfall> {
        // Find the sentence containing "do not"
        for line in docs.lines() {
            let lower = line.to_lowercase();
            if lower.contains("do not") || lower.contains("don't") {
                let title = if lower.contains("multiple") {
                    format!("Do NOT call {} multiple times", function_name)
                } else {
                    line.trim().to_string()
                };

                return Some(Pitfall {
                    title,
                    severity: Severity::High,
                    wrong_example: None,
                    correct_example: None,
                    explanation: line.trim().to_string(),
                    references: Vec::new(),
                    affected_functions: vec![function_name.to_string()],
                    confidence: 0.8,
                });
            }
        }
        None
    }

    /// Parses "Must call" requirements
    fn parse_must_call_requirement(&self, function_name: &str, docs: &str) -> Option<Pitfall> {
        for line in docs.lines() {
            let lower = line.to_lowercase();
            if lower.contains("must call") || lower.contains("must be called") {
                // Extract the required function name
                let required_func = self.extract_function_from_text(line);

                return Some(Pitfall {
                    title: format!("Must call required function before {}", function_name),
                    severity: Severity::High,
                    wrong_example: None,
                    correct_example: None,
                    explanation: line.trim().to_string(),
                    references: Vec::new(),
                    affected_functions: vec![function_name.to_string(), required_func],
                    confidence: 0.85,
                });
            }
        }
        None
    }

    /// Parses "Cannot" restrictions
    fn parse_cannot_restriction(&self, function_name: &str, docs: &str) -> Option<Pitfall> {
        for line in docs.lines() {
            let lower = line.to_lowercase();
            if lower.contains("cannot") || lower.contains("can not") {
                return Some(Pitfall {
                    title: format!("Restriction on {}", function_name),
                    severity: Severity::Medium,
                    wrong_example: None,
                    correct_example: None,
                    explanation: line.trim().to_string(),
                    references: Vec::new(),
                    affected_functions: vec![function_name.to_string()],
                    confidence: 0.75,
                });
            }
        }
        None
    }

    /// Parses multiple initialization warnings
    fn parse_multiple_init_warning(&self, function_name: &str, docs: &str) -> Option<Pitfall> {
        let lower = docs.to_lowercase();
        if (lower.contains("multiple") || lower.contains("twice"))
            && (lower.contains("init") || lower.contains("call"))
        {
            return Some(Pitfall {
                title: format!("Do NOT call {} multiple times", function_name),
                severity: Severity::High,
                wrong_example: Some(format!(
                    "let mut handle = Handle::new()?;\nhandle.{}()?;  // First call\nhandle.{}()?;  // Second call - WRONG!",
                    function_name, function_name
                )),
                correct_example: Some(format!(
                    "let mut handle = Handle::new()?;\nhandle.{}()?;  // Called exactly once",
                    function_name
                )),
                explanation: format!(
                    "Calling {} multiple times can cause resource leaks or undefined behavior.",
                    function_name
                ),
                references: Vec::new(),
                affected_functions: vec![function_name.to_string()],
                confidence: 0.9,
            });
        }
        None
    }

    /// Parses memory leak warnings
    fn parse_memory_leak_warning(&self, function_name: &str, docs: &str) -> Option<Pitfall> {
        for line in docs.lines() {
            if line.to_lowercase().contains("leak") {
                return Some(Pitfall {
                    title: format!("Memory leak risk with {}", function_name),
                    severity: Severity::High,
                    wrong_example: None,
                    correct_example: None,
                    explanation: line.trim().to_string(),
                    references: Vec::new(),
                    affected_functions: vec![function_name.to_string()],
                    confidence: 0.85,
                });
            }
        }
        None
    }

    /// Parses race condition warnings
    fn parse_race_condition_warning(&self, function_name: &str, docs: &str) -> Option<Pitfall> {
        let lower = docs.to_lowercase();
        if lower.contains("race") || (lower.contains("sync") && lower.contains("before")) {
            return Some(Pitfall {
                title: format!("Must synchronize before using {} result", function_name),
                severity: Severity::Critical,
                wrong_example: Some(format!(
                    "{}()?;\nlet result = read_result()?;  // RACE CONDITION!",
                    function_name
                )),
                correct_example: Some(format!(
                    "{}()?;\nsync()?;  // Wait for completion\nlet result = read_result()?;",
                    function_name
                )),
                explanation: format!(
                    "Results from {} may not be available immediately. Always synchronize first.",
                    function_name
                ),
                references: Vec::new(),
                affected_functions: vec![function_name.to_string()],
                confidence: 0.9,
            });
        }
        None
    }

    /// Parses order dependencies (must call X before Y)
    fn parse_order_dependency(&self, function_name: &str, docs: &str) -> Option<Pitfall> {
        for line in docs.lines() {
            let lower = line.to_lowercase();
            if lower.contains("before") && lower.contains("call") {
                let required_func = self.extract_function_from_text(line);

                return Some(Pitfall {
                    title: format!("Must call {} before {}", required_func, function_name),
                    severity: Severity::High,
                    wrong_example: None,
                    correct_example: None,
                    explanation: line.trim().to_string(),
                    references: Vec::new(),
                    affected_functions: vec![function_name.to_string(), required_func],
                    confidence: 0.8,
                });
            }
        }
        None
    }

    /// Parses issue line for pitfall information
    fn parse_issue_line(
        &self,
        function_name: &str,
        line: &str,
        full_content: &str,
    ) -> Option<Pitfall> {
        // Extract issue number
        let issue_num = line
            .split('#')
            .nth(1)?
            .split(|c: char| !c.is_ascii_digit())
            .next()?
            .to_string();

        let reference = format!("Issue #{}", issue_num);

        // Get the description after the issue number
        let desc = line.split(':').nth(1).unwrap_or(line).trim();

        // Try to find code examples in the full content near this issue number
        let (wrong_example, correct_example) =
            self.extract_examples_from_context(full_content, &issue_num);

        // Determine severity based on keywords in both line and full content
        let severity = self.determine_severity_from_context(desc, full_content, &issue_num);

        Some(Pitfall {
            title: desc.to_string(),
            severity,
            wrong_example,
            correct_example,
            explanation: desc.to_string(),
            references: vec![reference],
            affected_functions: vec![function_name.to_string()],
            confidence: 0.75,
        })
    }

    /// Extracts code examples from the full content near an issue reference
    fn extract_examples_from_context(
        &self,
        full_content: &str,
        issue_num: &str,
    ) -> (Option<String>, Option<String>) {
        let mut wrong_example = None;
        let mut correct_example = None;

        // Find the section about this issue
        let issue_marker = format!("#{}", issue_num);
        if let Some(issue_pos) = full_content.find(&issue_marker) {
            // Look at the next 1000 characters after the issue reference
            let context =
                &full_content[issue_pos..std::cmp::min(issue_pos + 1000, full_content.len())];

            // Look for code blocks or examples
            let mut in_code_block = false;
            let mut current_code = String::new();
            let mut is_wrong_example = false;
            let mut is_correct_example = false;

            for line in context.lines() {
                let line_lower = line.to_lowercase();

                // Detect code block markers
                if line.starts_with("```") || line.starts_with("~~~") {
                    if in_code_block {
                        // End of code block
                        if is_wrong_example && wrong_example.is_none() {
                            wrong_example = Some(current_code.trim().to_string());
                        } else if is_correct_example && correct_example.is_none() {
                            correct_example = Some(current_code.trim().to_string());
                        }
                        current_code.clear();
                        is_wrong_example = false;
                        is_correct_example = false;
                        in_code_block = false;
                    } else {
                        // Start of code block
                        in_code_block = true;
                    }
                    continue;
                }

                if in_code_block {
                    current_code.push_str(line);
                    current_code.push('\n');
                } else {
                    // Check for example type markers
                    if line_lower.contains("wrong")
                        || line_lower.contains("incorrect")
                        || line_lower.contains("bad")
                        || line_lower.contains("don't")
                    {
                        is_wrong_example = true;
                    } else if line_lower.contains("correct")
                        || line_lower.contains("right")
                        || line_lower.contains("good")
                        || line_lower.contains("instead")
                    {
                        is_correct_example = true;
                    }
                }
            }
        }

        (wrong_example, correct_example)
    }

    /// Determines severity from the full context around an issue
    fn determine_severity_from_context(
        &self,
        desc: &str,
        full_content: &str,
        issue_num: &str,
    ) -> Severity {
        let desc_lower = desc.to_lowercase();

        // Check description first
        if desc_lower.contains("crash")
            || desc_lower.contains("undefined")
            || desc_lower.contains("segfault")
            || desc_lower.contains("memory corruption")
        {
            return Severity::Critical;
        }

        if desc_lower.contains("leak")
            || desc_lower.contains("corrupt")
            || desc_lower.contains("hang")
        {
            return Severity::High;
        }

        // Also check full context for severity indicators
        let issue_marker = format!("#{}", issue_num);
        if let Some(issue_pos) = full_content.find(&issue_marker) {
            let context =
                &full_content[issue_pos..std::cmp::min(issue_pos + 500, full_content.len())];
            let context_lower = context.to_lowercase();

            if context_lower.contains("critical") || context_lower.contains("severe") {
                return Severity::Critical;
            }
            if context_lower.contains("important") || context_lower.contains("major") {
                return Severity::High;
            }
            if context_lower.contains("minor") || context_lower.contains("low priority") {
                return Severity::Low;
            }
        }

        Severity::Medium
    }

    /// Parses common mistake patterns
    fn parse_common_mistake(&self, function_name: &str, content: &str) -> Option<Pitfall> {
        for line in content.lines() {
            let lower = line.to_lowercase();
            if (lower.contains("frequently") || lower.contains("common"))
                && (lower.contains("mistake")
                    || lower.contains("error")
                    || lower.contains("forget"))
            {
                return Some(Pitfall {
                    title: format!("Common mistake with {}", function_name),
                    severity: Severity::Medium,
                    wrong_example: None,
                    correct_example: None,
                    explanation: line.trim().to_string(),
                    references: Vec::new(),
                    affected_functions: vec![function_name.to_string()],
                    confidence: 0.7,
                });
            }
        }
        None
    }

    /// Extracts function name from text (looks for backtick-wrapped names or common patterns)
    fn extract_function_from_text(&self, text: &str) -> String {
        // Try to find backtick-wrapped function name
        if let Some(start) = text.find('`')
            && let Some(end) = text[start + 1..].find('`') {
                return text[start + 1..start + 1 + end]
                    .trim_end_matches("()")
                    .to_string();
            }

        // Fallback: look for common function-like words
        for word in text.split_whitespace() {
            if word.ends_with("()") {
                return word.trim_end_matches("()").to_string();
            }
        }

        String::from("unknown")
    }
}

impl Default for AntiPatternAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_do_not_multiple_times() {
        let mut analyzer = AntiPatternAnalyzer::new();
        let docs =
            "Do not call init() multiple times on the same handle. This causes memory leaks.";

        let info = analyzer.analyze("init", docs, None);
        assert!(info.has_pitfalls());
        assert_eq!(info.pitfalls.len(), 3); // "do not" + "multiple" + "leak"
        assert!(info.pitfalls.iter().any(|p| p.severity == Severity::High));
    }

    #[test]
    fn test_must_call_requirement() {
        let mut analyzer = AntiPatternAnalyzer::new();
        let docs = "You must call initialize() before calling this function.";

        let info = analyzer.analyze("process", docs, None);
        assert!(info.has_pitfalls());
        assert!(info.pitfalls.iter().any(|p| p.title.contains("Must call")));
    }

    #[test]
    fn test_race_condition_warning() {
        let mut analyzer = AntiPatternAnalyzer::new();
        let docs = "This function has a race condition if not synchronized properly.";

        let info = analyzer.analyze("compute_async", docs, None);
        assert!(info.has_pitfalls());
        assert!(
            info.pitfalls
                .iter()
                .any(|p| p.severity == Severity::Critical)
        );
    }

    #[test]
    fn test_memory_leak_warning() {
        let mut analyzer = AntiPatternAnalyzer::new();
        let docs = "Memory leak occurs if destroy() is not called.";

        let info = analyzer.analyze("create", docs, None);
        assert!(info.has_pitfalls());
        assert!(info.pitfalls.iter().any(|p| p.title.contains("leak")));
    }

    #[test]
    fn test_issue_extraction() {
        let mut analyzer = AntiPatternAnalyzer::new();
        let issues = "Issue #1234: Users frequently forget to synchronize before reading results.";

        let info = analyzer.analyze("read_result", "", Some(issues));
        assert!(info.has_pitfalls());
        assert!(
            info.pitfalls
                .iter()
                .any(|p| p.references.contains(&"Issue #1234".to_string()))
        );
    }

    #[test]
    fn test_by_severity() {
        let mut analyzer = AntiPatternAnalyzer::new();
        let docs = "Do not call this function multiple times. This function has a race condition.";

        let info = analyzer.analyze("test", docs, None);
        let critical = info.by_severity(Severity::Critical);
        let high = info.by_severity(Severity::High);

        assert!(!critical.is_empty() || !high.is_empty());
    }

    #[test]
    fn test_documentation_generation() {
        let pitfall = Pitfall {
            title: "Do NOT call multiple times".to_string(),
            severity: Severity::High,
            wrong_example: Some("init();\ninit();  // WRONG!".to_string()),
            correct_example: Some("init();  // Once only".to_string()),
            explanation: "Causes memory leak".to_string(),
            references: vec!["Issue #123".to_string()],
            affected_functions: vec!["init".to_string()],
            confidence: 0.9,
        };

        let info = PitfallInfo {
            pitfalls: vec![pitfall],
            confidence: 0.9,
        };

        let docs = info.generate_documentation();
        assert!(docs.contains("# Common Pitfalls"));
        assert!(docs.contains("⚠️"));
        assert!(docs.contains("❌ WRONG"));
        assert!(docs.contains("✅ CORRECT"));
        assert!(docs.contains("Issue #123"));
    }

    #[test]
    fn test_cache() {
        let mut analyzer = AntiPatternAnalyzer::new();
        let docs = "Do not call this multiple times.";

        // First call
        let info1 = analyzer.analyze("test", docs, None);
        // Second call (should use cache)
        let info2 = analyzer.analyze("test", docs, None);

        assert_eq!(info1, info2);
    }

    #[test]
    fn test_no_pitfalls() {
        let mut analyzer = AntiPatternAnalyzer::new();
        let docs = "This is a simple function that returns a value.";

        let info = analyzer.analyze("get_value", docs, None);
        assert!(!info.has_pitfalls());
        assert_eq!(info.confidence, 0.0);
    }

    #[test]
    fn test_severity_sorting() {
        let info = PitfallInfo {
            pitfalls: vec![
                Pitfall {
                    title: "Low issue".to_string(),
                    severity: Severity::Low,
                    wrong_example: None,
                    correct_example: None,
                    explanation: "".to_string(),
                    references: vec![],
                    affected_functions: vec![],
                    confidence: 0.5,
                },
                Pitfall {
                    title: "Critical issue".to_string(),
                    severity: Severity::Critical,
                    wrong_example: None,
                    correct_example: None,
                    explanation: "".to_string(),
                    references: vec![],
                    affected_functions: vec![],
                    confidence: 0.9,
                },
            ],
            confidence: 0.7,
        };

        let docs = info.generate_documentation();
        // Critical should appear before Low
        let critical_pos = docs.find("Critical issue").unwrap();
        let low_pos = docs.find("Low issue").unwrap();
        assert!(critical_pos < low_pos);
    }

    #[test]
    fn test_cannot_restriction() {
        let mut analyzer = AntiPatternAnalyzer::new();
        let docs = "This function cannot be called after finalize().";

        let info = analyzer.analyze("process", docs, None);
        assert!(info.has_pitfalls());
        assert!(
            info.pitfalls
                .iter()
                .any(|p| p.title.contains("Restriction"))
        );
    }

    #[test]
    fn test_confidence_calculation() {
        let mut analyzer = AntiPatternAnalyzer::new();
        let docs = "Do not call this multiple times. This causes a race condition.";

        let info = analyzer.analyze("test", docs, None);
        assert!(info.confidence > 0.0);
        assert!(info.confidence <= 1.0);
    }
}
