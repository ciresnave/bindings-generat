//! Precondition and constraint extraction from C/C++ documentation.
//!
//! This module analyzes function documentation to extract preconditions, constraints,
//! and invariants that must be satisfied for correct operation. These are crucial for
//! preventing undefined behavior and generating appropriate validation code.

use std::collections::HashMap;

/// Type of constraint that applies to a parameter or function state
#[derive(Debug, Clone, PartialEq)]
pub enum ConstraintType {
    /// Parameter must not be null
    NonNull,
    /// Parameter may be null under certain conditions
    Nullable { condition: Option<String> },
    /// Parameter must be within a specific range
    Range {
        min: Option<String>,
        max: Option<String>,
        inclusive: bool,
    },
    /// Parameter must satisfy a mathematical relationship
    Relationship { expression: String },
    /// Function requires specific prior state
    StateRequirement { required_state: String },
    /// Parameter must be a power of 2
    PowerOfTwo,
    /// Parameter must be a multiple of some value
    MultipleOf { value: String },
    /// Function call ordering constraint
    CallOrder {
        before: Option<String>,
        after: Option<String>,
    },
    /// Platform-specific constraint
    Platform { platforms: Vec<String> },
    /// Custom constraint that doesn't fit other categories
    Custom { description: String },
}

/// A single precondition or constraint on a function or parameter
#[derive(Debug, Clone, PartialEq)]
pub struct Precondition {
    /// Parameter name this applies to, or None for function-level constraints
    pub parameter: Option<String>,
    /// Type of constraint
    pub constraint_type: ConstraintType,
    /// Human-readable description
    pub description: String,
    /// Whether this can be validated at runtime
    pub can_validate: bool,
    /// Generated validation code (if applicable)
    pub validation_code: Option<String>,
    /// Confidence score (0.0-1.0) based on source clarity
    pub confidence: f64,
}

/// Complete precondition analysis for a function
#[derive(Debug, Clone, PartialEq)]
pub struct PreconditionInfo {
    /// All detected preconditions
    pub preconditions: Vec<Precondition>,
    /// Parameters that must not be null
    pub non_null_params: Vec<String>,
    /// Parameters that may be null
    pub nullable_params: Vec<String>,
    /// Undefined behavior scenarios
    pub undefined_behavior: Vec<String>,
    /// Performance constraints or recommendations
    pub performance_notes: Vec<String>,
    /// Overall confidence in the analysis
    pub confidence: f64,
}

impl PreconditionInfo {
    /// Create a new empty precondition analysis
    pub fn new() -> Self {
        Self {
            preconditions: Vec::new(),
            non_null_params: Vec::new(),
            nullable_params: Vec::new(),
            undefined_behavior: Vec::new(),
            performance_notes: Vec::new(),
            confidence: 0.0,
        }
    }

    /// Check if any preconditions were detected
    pub fn has_preconditions(&self) -> bool {
        !self.preconditions.is_empty()
    }

    /// Get all preconditions that can be validated at runtime
    pub fn validatable_preconditions(&self) -> Vec<&Precondition> {
        self.preconditions
            .iter()
            .filter(|p| p.can_validate)
            .collect()
    }

    /// Get all preconditions for a specific parameter
    pub fn preconditions_for_param(&self, param: &str) -> Vec<&Precondition> {
        self.preconditions
            .iter()
            .filter(|p| p.parameter.as_deref() == Some(param))
            .collect()
    }
}

impl Default for PreconditionInfo {
    fn default() -> Self {
        Self::new()
    }
}

/// Analyzer for extracting preconditions from function documentation
#[derive(Debug)]
pub struct PreconditionAnalyzer {
    /// Cache of analyzed functions
    cache: HashMap<String, PreconditionInfo>,
}

impl PreconditionAnalyzer {
    /// Create a new precondition analyzer
    pub fn new() -> Self {
        Self {
            cache: HashMap::new(),
        }
    }

    /// Analyze a function's documentation for preconditions
    ///
    /// # Arguments
    /// * `function_name` - Name of the function being analyzed
    /// * `documentation` - Combined documentation text to analyze
    /// * `param_docs` - Map of parameter names to their documentation
    pub fn analyze(
        &mut self,
        function_name: &str,
        documentation: &str,
        param_docs: &HashMap<String, String>,
    ) -> PreconditionInfo {
        // Check cache first
        if let Some(cached) = self.cache.get(function_name) {
            return cached.clone();
        }

        let mut info = PreconditionInfo::new();
        let doc_lower = documentation.to_lowercase();

        // Extract preconditions from annotations
        self.extract_explicit_preconditions(&doc_lower, &mut info);

        // Extract null checks
        self.extract_null_constraints(&doc_lower, param_docs, &mut info);

        // Extract range constraints
        self.extract_range_constraints(&doc_lower, param_docs, &mut info);

        // Extract state requirements
        self.extract_state_requirements(&doc_lower, &mut info);

        // Extract call order constraints
        self.extract_call_order(&doc_lower, &mut info);

        // Extract platform constraints
        self.extract_platform_constraints(&doc_lower, &mut info);

        // Extract undefined behavior warnings
        self.extract_undefined_behavior(&doc_lower, &mut info);

        // Extract performance constraints
        self.extract_performance_notes(&doc_lower, &mut info);

        // Calculate overall confidence
        info.confidence = self.calculate_confidence(&info);

        // Cache the result
        self.cache.insert(function_name.to_string(), info.clone());

        info
    }

    /// Extract explicit precondition annotations
    fn extract_explicit_preconditions(&self, doc: &str, info: &mut PreconditionInfo) {
        let patterns = [
            "@pre ",
            "@precondition ",
            "@requires ",
            "precondition:",
            "requires:",
            "prerequisite:",
            "must satisfy:",
            "constraint:",
        ];

        for pattern in &patterns {
            if let Some(idx) = doc.find(pattern) {
                // Extract text until end of line or sentence
                let rest = &doc[idx + pattern.len()..];
                if let Some(end) = rest.find(['\n', '.', ';'].as_ref()) {
                    let constraint_text = rest[..end].trim();
                    info.preconditions.push(Precondition {
                        parameter: None,
                        constraint_type: ConstraintType::Custom {
                            description: constraint_text.to_string(),
                        },
                        description: constraint_text.to_string(),
                        can_validate: false,
                        validation_code: None,
                        confidence: 0.9,
                    });
                }
            }
        }
    }

    /// Extract null pointer constraints
    fn extract_null_constraints(
        &self,
        doc: &str,
        param_docs: &HashMap<String, String>,
        info: &mut PreconditionInfo,
    ) {
        // Also check main documentation for general null pointer requirements
        let doc_lower = doc.to_lowercase();
        if doc_lower.contains("null pointer") || doc_lower.contains("nullptr") {
            // Extract which parameters are mentioned
            for param in param_docs.keys() {
                if doc_lower.contains(&param.to_lowercase()) {
                    // Already will be caught below, but check for general warnings
                    if doc_lower.contains(&format!("all {} must", param.to_lowercase()))
                        || doc_lower.contains(&format!("{} parameters must", param.to_lowercase()))
                    {
                        // This is a general requirement for this parameter type
                    }
                }
            }
        }

        // Check for non-null requirements
        let non_null_patterns = [
            "must not be null",
            "cannot be null",
            "must not be nullptr",
            "should not be null",
            "non-null",
            "nonnull",
        ];

        for (param, param_doc) in param_docs {
            let param_lower = param_doc.to_lowercase();

            for pattern in &non_null_patterns {
                if param_lower.contains(pattern) {
                    info.non_null_params.push(param.clone());
                    info.preconditions.push(Precondition {
                        parameter: Some(param.clone()),
                        constraint_type: ConstraintType::NonNull,
                        description: format!("{} must not be null", param),
                        can_validate: false, // Rust's type system handles this
                        validation_code: None,
                        confidence: 0.95,
                    });
                    break;
                }
            }
        }

        // Check for nullable parameters
        let nullable_patterns = [
            "may be null",
            "can be null",
            "nullable",
            "null is allowed",
            "optional",
        ];

        for (param, param_doc) in param_docs {
            let param_lower = param_doc.to_lowercase();

            for pattern in &nullable_patterns {
                if param_lower.contains(pattern) {
                    info.nullable_params.push(param.clone());
                    info.preconditions.push(Precondition {
                        parameter: Some(param.clone()),
                        constraint_type: ConstraintType::Nullable { condition: None },
                        description: format!("{} may be null", param),
                        can_validate: false,
                        validation_code: None,
                        confidence: 0.9,
                    });
                    break;
                }
            }
        }
    }

    /// Extract range constraints from documentation
    fn extract_range_constraints(
        &self,
        doc: &str,
        param_docs: &HashMap<String, String>,
        info: &mut PreconditionInfo,
    ) {
        // Check main documentation for general range requirements
        let doc_lower = doc.to_lowercase();

        // Look for general range statements like "all values must be positive"
        if doc_lower.contains("must be")
            && (doc_lower.contains("positive") || doc_lower.contains("> 0"))
        {
            for (param, param_doc) in param_docs {
                let param_doc_lower = param_doc.to_lowercase();

                // If the parameter looks numeric and doc mentions it
                // Check both main doc and parameter-specific doc for confirmation
                let in_main_doc = doc_lower.contains(&param.to_lowercase());
                let in_param_doc = param_doc_lower.contains("positive")
                    || param_doc_lower.contains("> 0")
                    || param_doc_lower.contains("greater than 0");

                if in_main_doc || in_param_doc {
                    // Higher confidence if mentioned in param-specific doc
                    let confidence = if in_param_doc { 0.90 } else { 0.75 };

                    info.preconditions.push(Precondition {
                        parameter: Some(param.clone()),
                        constraint_type: ConstraintType::Range {
                            min: Some("1".to_string()),
                            max: None,
                            inclusive: true,
                        },
                        description: format!(
                            "{} must be positive (extracted from {})",
                            param,
                            if in_param_doc {
                                "parameter documentation"
                            } else {
                                "general documentation"
                            }
                        ),
                        can_validate: true,
                        validation_code: Some(format!(
                            "if {} <= 0 {{ return Err(Error::InvalidParameter); }}",
                            param
                        )),
                        confidence,
                    });
                }
            }
        }

        for (param, param_doc) in param_docs {
            let param_lower = param_doc.to_lowercase();

            // Check for "must be > 0" pattern
            if param_lower.contains("must be > 0")
                || param_lower.contains("must be greater than 0")
                || param_lower.contains("must be positive")
            {
                info.preconditions.push(Precondition {
                    parameter: Some(param.clone()),
                    constraint_type: ConstraintType::Range {
                        min: Some("1".to_string()),
                        max: None,
                        inclusive: true,
                    },
                    description: format!("{} must be greater than 0", param),
                    can_validate: true,
                    validation_code: Some(format!(
                        "if {} == 0 {{\n    return Err(Error::InvalidParameter(\"{} must be > 0\".into()));\n}}",
                        param, param
                    )),
                    confidence: 0.95,
                });
            }

            // Check for "range: [min, max]" pattern
            if let Some(range_idx) = param_lower.find("range:") {
                let rest = &param_lower[range_idx + 6..];
                if let Some(bracket_start) = rest.find('[')
                    && let Some(bracket_end) = rest.find(']') {
                        let range_text = &rest[bracket_start + 1..bracket_end];
                        if let Some(comma_idx) = range_text.find(',') {
                            let min = range_text[..comma_idx].trim();
                            let max = range_text[comma_idx + 1..].trim();

                            info.preconditions.push(Precondition {
                                parameter: Some(param.clone()),
                                constraint_type: ConstraintType::Range {
                                    min: Some(min.to_string()),
                                    max: Some(max.to_string()),
                                    inclusive: true,
                                },
                                description: format!("{} must be in range [{}, {}]", param, min, max),
                                can_validate: true,
                                validation_code: Some(format!(
                                    "if {} < {} || {} > {} {{\n    return Err(Error::InvalidParameter(\"{} must be in range [{}, {}]\".into()));\n}}",
                                    param, min, param, max, param, min, max
                                )),
                                confidence: 0.95,
                            });
                        }
                    }
            }

            // Check for "power of 2" constraint
            if param_lower.contains("power of 2") || param_lower.contains("power of two") {
                info.preconditions.push(Precondition {
                    parameter: Some(param.clone()),
                    constraint_type: ConstraintType::PowerOfTwo,
                    description: format!("{} must be a power of 2", param),
                    can_validate: true,
                    validation_code: Some(format!(
                        "if {} == 0 || ({} & ({} - 1)) != 0 {{\n    return Err(Error::InvalidParameter(\"{} must be a power of 2\".into()));\n}}",
                        param, param, param, param
                    )),
                    confidence: 0.9,
                });
            }

            // Check for "multiple of X" constraint
            if let Some(mult_idx) = param_lower.find("multiple of ") {
                let rest = &param_lower[mult_idx + 12..];
                // Extract number - find first non-digit character
                let mut end_idx = 0;
                for (i, c) in rest.chars().enumerate() {
                    if c.is_ascii_digit() {
                        end_idx = i + 1;
                    } else if end_idx > 0 {
                        break;
                    }
                }

                if end_idx > 0 {
                    let value = &rest[..end_idx];
                    info.preconditions.push(Precondition {
                        parameter: Some(param.clone()),
                        constraint_type: ConstraintType::MultipleOf {
                            value: value.to_string(),
                        },
                        description: format!("{} must be a multiple of {}", param, value),
                        can_validate: true,
                        validation_code: Some(format!(
                            "if {} % {} != 0 {{\n    return Err(Error::InvalidParameter(\"{} must be a multiple of {}\".into()));\n}}",
                            param, value, param, value
                        )),
                        confidence: 0.85,
                    });
                }
            }
        }
    }

    /// Extract state requirements (initialization, prior calls, etc.)
    fn extract_state_requirements(&self, doc: &str, info: &mut PreconditionInfo) {
        let patterns = [
            (
                "must be initialized",
                "object must be initialized before use",
            ),
            (
                "requires initialization",
                "initialization required before calling",
            ),
            ("must call", "specific function must be called first"),
            ("call first", "another function must be called first"),
            ("after calling", "can only be called after specific state"),
            (
                "before calling",
                "must be called before specific operations",
            ),
        ];

        for (pattern, description) in &patterns {
            if doc.contains(pattern) {
                info.preconditions.push(Precondition {
                    parameter: None,
                    constraint_type: ConstraintType::StateRequirement {
                        required_state: pattern.to_string(),
                    },
                    description: description.to_string(),
                    can_validate: false,
                    validation_code: None,
                    confidence: 0.8,
                });
            }
        }
    }

    /// Extract call order constraints
    fn extract_call_order(&self, doc: &str, info: &mut PreconditionInfo) {
        if doc.contains("cannot be called after") || doc.contains("must not be called after") {
            info.preconditions.push(Precondition {
                parameter: None,
                constraint_type: ConstraintType::CallOrder {
                    before: None,
                    after: Some("specific operations".to_string()),
                },
                description: "Function has call order restrictions".to_string(),
                can_validate: false,
                validation_code: None,
                confidence: 0.75,
            });
        }
    }

    /// Extract platform-specific constraints
    fn extract_platform_constraints(&self, doc: &str, info: &mut PreconditionInfo) {
        let platforms = ["windows", "linux", "macos", "unix", "cuda", "gpu"];

        for platform in &platforms {
            let pattern = format!("only on {}", platform);
            if doc.contains(&pattern) || doc.contains(&format!("available on {}", platform)) {
                info.preconditions.push(Precondition {
                    parameter: None,
                    constraint_type: ConstraintType::Platform {
                        platforms: vec![platform.to_string()],
                    },
                    description: format!("Only available on {}", platform),
                    can_validate: false,
                    validation_code: None,
                    confidence: 0.85,
                });
            }
        }
    }

    /// Extract undefined behavior warnings
    fn extract_undefined_behavior(&self, doc: &str, info: &mut PreconditionInfo) {
        let patterns = [
            "undefined behavior",
            "undefined behaviour",
            "ub occurs",
            "results in ub",
            "causes undefined",
        ];

        for pattern in &patterns {
            if doc.contains(pattern) {
                // Extract sentence containing the UB warning
                if let Some(idx) = doc.find(pattern) {
                    // Find sentence boundaries
                    let before = &doc[..idx];
                    let after = &doc[idx..];

                    let sentence_start = before.rfind('.').map(|i| i + 1).unwrap_or(0);
                    let sentence_end = after.find('.').unwrap_or(after.len());

                    let sentence = &doc[sentence_start..idx + sentence_end];
                    let trimmed = sentence.trim();

                    if !trimmed.is_empty() {
                        info.undefined_behavior.push(trimmed.to_string());
                        return; // Only take first match to avoid duplicates
                    }
                }
            }
        }
    }

    /// Extract performance-related constraints or notes
    fn extract_performance_notes(&self, doc: &str, info: &mut PreconditionInfo) {
        let patterns = [
            ("optimal performance", "Performance optimization note"),
            ("for best performance", "Performance optimization note"),
            ("recommended for performance", "Performance recommendation"),
            ("should be <= ", "Recommended limit"),
            ("hardware limit", "Hardware constraint"),
        ];

        for (pattern, note_type) in &patterns {
            if let Some(idx) = doc.find(pattern) {
                let start = idx.saturating_sub(30);
                let end = (idx + 80).min(doc.len());
                let context = &doc[start..end].trim();

                info.performance_notes
                    .push(format!("{}: {}", note_type, context));
            }
        }
    }

    /// Calculate overall confidence based on detected preconditions
    fn calculate_confidence(&self, info: &PreconditionInfo) -> f64 {
        if info.preconditions.is_empty() {
            return 0.0;
        }

        let sum: f64 = info.preconditions.iter().map(|p| p.confidence).sum();
        let count = info.preconditions.len() as f64;

        sum / count
    }

    /// Generate comprehensive documentation for preconditions
    pub fn generate_docs(&self, info: &PreconditionInfo) -> String {
        let mut docs = String::new();

        if !info.preconditions.is_empty() {
            docs.push_str("/// # Preconditions\n");
            docs.push_str("///\n");

            for precond in &info.preconditions {
                docs.push_str(&format!("/// - {}\n", precond.description));
            }

            docs.push_str("///\n");
        }

        if !info.undefined_behavior.is_empty() {
            docs.push_str("/// # Undefined Behavior\n");
            docs.push_str("///\n");
            for ub in &info.undefined_behavior {
                docs.push_str(&format!("/// - {}\n", ub));
            }
            docs.push_str("///\n");
        }

        if !info.performance_notes.is_empty() {
            docs.push_str("/// # Performance Notes\n");
            docs.push_str("///\n");
            for note in &info.performance_notes {
                docs.push_str(&format!("/// - {}\n", note));
            }
            docs.push_str("///\n");
        }

        let validatable = info.validatable_preconditions();
        if !validatable.is_empty() {
            docs.push_str("/// # Validation\n");
            docs.push_str("///\n");
            docs.push_str("/// This method performs runtime validation:\n");
            docs.push_str("/// ```rust\n");
            for precond in validatable {
                if let Some(code) = &precond.validation_code {
                    docs.push_str(&format!("/// {}\n", code.replace('\n', "\n/// ")));
                }
            }
            docs.push_str("/// ```\n");
        }

        docs
    }

    /// Clear the cache (useful for testing or memory management)
    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }
}

impl Default for PreconditionAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_null_constraint_detection() {
        let mut analyzer = PreconditionAnalyzer::new();
        let doc = "Creates a handle";
        let mut params = HashMap::new();
        params.insert("ptr".to_string(), "Must not be NULL".to_string());

        let info = analyzer.analyze("test_func", doc, &params);

        assert!(info.has_preconditions());
        assert_eq!(info.non_null_params.len(), 1);
        assert_eq!(info.non_null_params[0], "ptr");
    }

    #[test]
    fn test_nullable_constraint() {
        let mut analyzer = PreconditionAnalyzer::new();
        let doc = "Optional parameter";
        let mut params = HashMap::new();
        params.insert("optional".to_string(), "May be null".to_string());

        let info = analyzer.analyze("test_func", doc, &params);

        assert_eq!(info.nullable_params.len(), 1);
        assert_eq!(info.nullable_params[0], "optional");
    }

    #[test]
    fn test_range_constraint_greater_than_zero() {
        let mut analyzer = PreconditionAnalyzer::new();
        let doc = "Allocates memory";
        let mut params = HashMap::new();
        params.insert("size".to_string(), "Must be > 0".to_string());

        let info = analyzer.analyze("test_func", doc, &params);

        let size_precond = info.preconditions_for_param("size");
        assert_eq!(size_precond.len(), 1);
        assert!(size_precond[0].can_validate);
        assert!(size_precond[0].validation_code.is_some());
    }

    #[test]
    fn test_range_constraint_with_bounds() {
        let mut analyzer = PreconditionAnalyzer::new();
        let doc = "Sets value";
        let mut params = HashMap::new();
        params.insert("count".to_string(), "Valid range: [0, 100]".to_string());

        let info = analyzer.analyze("test_func", doc, &params);

        let count_precond = info.preconditions_for_param("count");
        assert_eq!(count_precond.len(), 1);

        match &count_precond[0].constraint_type {
            ConstraintType::Range { min, max, .. } => {
                assert_eq!(min.as_deref(), Some("0"));
                assert_eq!(max.as_deref(), Some("100"));
            }
            _ => panic!("Expected Range constraint"),
        }
    }

    #[test]
    fn test_power_of_two_constraint() {
        let mut analyzer = PreconditionAnalyzer::new();
        let doc = "Alignment function";
        let mut params = HashMap::new();
        params.insert("alignment".to_string(), "Must be a power of 2".to_string());

        let info = analyzer.analyze("test_func", doc, &params);

        let alignment_precond = info.preconditions_for_param("alignment");
        assert_eq!(alignment_precond.len(), 1);
        assert!(matches!(
            alignment_precond[0].constraint_type,
            ConstraintType::PowerOfTwo
        ));
        assert!(alignment_precond[0].can_validate);
    }

    #[test]
    fn test_multiple_of_constraint() {
        let mut analyzer = PreconditionAnalyzer::new();
        let doc = "Memory operation";
        let mut params = HashMap::new();
        params.insert("size".to_string(), "Must be a multiple of 32".to_string());

        let info = analyzer.analyze("test_func", doc, &params);

        let size_precond = info.preconditions_for_param("size");
        assert_eq!(size_precond.len(), 1);

        match &size_precond[0].constraint_type {
            ConstraintType::MultipleOf { value } => {
                assert_eq!(value, "32");
            }
            _ => panic!("Expected MultipleOf constraint"),
        }
    }

    #[test]
    fn test_state_requirement() {
        let mut analyzer = PreconditionAnalyzer::new();
        let doc = "This function must be initialized before use";
        let params = HashMap::new();

        let info = analyzer.analyze("test_func", doc, &params);

        assert!(info.has_preconditions());
        let state_precond = info
            .preconditions
            .iter()
            .find(|p| matches!(p.constraint_type, ConstraintType::StateRequirement { .. }));
        assert!(state_precond.is_some());
    }

    #[test]
    fn test_undefined_behavior_detection() {
        let mut analyzer = PreconditionAnalyzer::new();
        let doc = "Passing null causes undefined behavior. Always check pointers.";
        let params = HashMap::new();

        let info = analyzer.analyze("test_func", doc, &params);

        assert!(!info.undefined_behavior.is_empty());
    }

    #[test]
    fn test_performance_notes() {
        let mut analyzer = PreconditionAnalyzer::new();
        let doc = "For optimal performance, size should be <= 8192";
        let params = HashMap::new();

        let info = analyzer.analyze("test_func", doc, &params);

        assert!(!info.performance_notes.is_empty());
    }

    #[test]
    fn test_platform_constraint() {
        let mut analyzer = PreconditionAnalyzer::new();
        let doc = "This function is only available on Linux";
        let params = HashMap::new();

        let info = analyzer.analyze("test_func", doc, &params);

        let platform_precond = info
            .preconditions
            .iter()
            .find(|p| matches!(p.constraint_type, ConstraintType::Platform { .. }));
        assert!(platform_precond.is_some());
    }

    #[test]
    fn test_docs_generation() {
        let mut analyzer = PreconditionAnalyzer::new();
        let doc = "Function with constraints";
        let mut params = HashMap::new();
        params.insert("size".to_string(), "Must be > 0".to_string());

        let info = analyzer.analyze("test_func", doc, &params);
        let docs = analyzer.generate_docs(&info);

        assert!(docs.contains("# Preconditions"));
        assert!(docs.contains("# Validation"));
        assert!(docs.contains("size"));
    }

    #[test]
    fn test_cache() {
        let mut analyzer = PreconditionAnalyzer::new();
        let doc = "Test function";
        let params = HashMap::new();

        let info1 = analyzer.analyze("cached_func", doc, &params);
        let info2 = analyzer.analyze("cached_func", doc, &params);

        assert_eq!(info1, info2);

        analyzer.clear_cache();
        let info3 = analyzer.analyze("cached_func", doc, &params);
        assert_eq!(info1, info3);
    }
}
