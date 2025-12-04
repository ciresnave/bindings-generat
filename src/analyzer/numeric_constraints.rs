use std::collections::HashMap;

/// Type of numeric constraint
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ConstraintType {
    Alignment,
    Range,
    PowerOfTwo,
    Multiple,
    NonZero,
    Positive,
    NonNegative,
    Overflow,
    Underflow,
}

/// Information about a numeric constraint
#[derive(Debug, Clone, PartialEq)]
pub struct NumericConstraint {
    pub constraint_type: ConstraintType,
    pub parameter_name: Option<String>,
    pub min_value: Option<i64>,
    pub max_value: Option<i64>,
    pub must_be_power_of_two: bool,
    pub must_be_multiple_of: Option<u64>,
    pub alignment_bytes: Option<u64>,
    pub can_overflow: bool,
    pub overflow_behavior: Option<String>,
    pub description: String,
}

/// Collection of numeric constraints
#[derive(Debug, Clone, PartialEq)]
pub struct NumericConstraints {
    pub constraints: Vec<NumericConstraint>,
}

impl NumericConstraints {
    pub fn new() -> Self {
        Self {
            constraints: Vec::new(),
        }
    }

    pub fn add_constraint(&mut self, constraint: NumericConstraint) {
        self.constraints.push(constraint);
    }

    pub fn has_constraints(&self) -> bool {
        !self.constraints.is_empty()
    }
}

impl Default for NumericConstraints {
    fn default() -> Self {
        Self::new()
    }
}

/// Analyzer for extracting numeric constraint information
#[derive(Debug)]
pub struct NumericConstraintsAnalyzer {
    cache: HashMap<String, NumericConstraints>,
}

impl NumericConstraintsAnalyzer {
    pub fn new() -> Self {
        Self {
            cache: HashMap::new(),
        }
    }

    /// Analyze documentation to extract numeric constraints
    pub fn analyze(&mut self, function_name: &str, docs: &str) -> NumericConstraints {
        if let Some(cached) = self.cache.get(function_name) {
            return cached.clone();
        }

        let mut constraints = NumericConstraints::new();
        let lower_docs = docs.to_lowercase();

        // Extract various constraint types
        self.extract_alignment_constraints(docs, &lower_docs, &mut constraints);
        self.extract_range_constraints(docs, &lower_docs, &mut constraints);
        self.extract_power_of_two_constraints(&lower_docs, &mut constraints);
        self.extract_multiple_constraints(docs, &lower_docs, &mut constraints);
        self.extract_sign_constraints(&lower_docs, &mut constraints);
        self.extract_overflow_constraints(&lower_docs, &mut constraints);

        self.cache
            .insert(function_name.to_string(), constraints.clone());
        constraints
    }

    fn extract_alignment_constraints(
        &self,
        docs: &str,
        lower_docs: &str,
        constraints: &mut NumericConstraints,
    ) {
        // Look for alignment requirements
        let patterns = [
            "must be aligned",
            "must be aligned to",
            "alignment",
            "byte-aligned",
            "aligned to",
        ];

        for pattern in &patterns {
            if lower_docs.contains(pattern) {
                let alignment = self.extract_alignment_value(docs);

                constraints.add_constraint(NumericConstraint {
                    constraint_type: ConstraintType::Alignment,
                    parameter_name: None,
                    min_value: None,
                    max_value: None,
                    must_be_power_of_two: false,
                    must_be_multiple_of: None,
                    alignment_bytes: alignment,
                    can_overflow: false,
                    overflow_behavior: None,
                    description: if let Some(align) = alignment {
                        format!("Must be aligned to {} bytes", align)
                    } else {
                        "Must be properly aligned".to_string()
                    },
                });
                break;
            }
        }
    }

    fn extract_alignment_value(&self, docs: &str) -> Option<u64> {
        // Look for patterns like "aligned to 8 bytes" or "8-byte aligned"
        if let Ok(re) = regex::Regex::new(r"(?:aligned to |alignment )?(\d+)(?:-byte| bytes?)")
            && let Some(cap) = re.find(docs) {
                let text = cap.as_str();
                if let Ok(num_re) = regex::Regex::new(r"\d+")
                    && let Some(num_match) = num_re.find(text) {
                        return num_match.as_str().parse().ok();
                    }
            }
        None
    }

    fn extract_range_constraints(
        &self,
        docs: &str,
        lower_docs: &str,
        constraints: &mut NumericConstraints,
    ) {
        // Look for range patterns
        if (lower_docs.contains("range")
            || lower_docs.contains("between")
            || lower_docs.contains("from") && lower_docs.contains("to"))
            && let Some((min, max)) = self.extract_range_values(docs) {
                constraints.add_constraint(NumericConstraint {
                    constraint_type: ConstraintType::Range,
                    parameter_name: None,
                    min_value: Some(min),
                    max_value: Some(max),
                    must_be_power_of_two: false,
                    must_be_multiple_of: None,
                    alignment_bytes: None,
                    can_overflow: false,
                    overflow_behavior: None,
                    description: format!("Must be in range {} to {}", min, max),
                });
            }

        // Look for minimum values
        if (lower_docs.contains("must be at least")
            || lower_docs.contains("minimum")
            || lower_docs.contains("greater than"))
            && let Some(min) = self.extract_minimum_value(docs) {
                constraints.add_constraint(NumericConstraint {
                    constraint_type: ConstraintType::Range,
                    parameter_name: None,
                    min_value: Some(min),
                    max_value: None,
                    must_be_power_of_two: false,
                    must_be_multiple_of: None,
                    alignment_bytes: None,
                    can_overflow: false,
                    overflow_behavior: None,
                    description: format!("Must be at least {}", min),
                });
            }

        // Look for maximum values
        if (lower_docs.contains("must not exceed")
            || lower_docs.contains("maximum")
            || lower_docs.contains("less than"))
            && let Some(max) = self.extract_maximum_value(docs) {
                constraints.add_constraint(NumericConstraint {
                    constraint_type: ConstraintType::Range,
                    parameter_name: None,
                    min_value: None,
                    max_value: Some(max),
                    must_be_power_of_two: false,
                    must_be_multiple_of: None,
                    alignment_bytes: None,
                    can_overflow: false,
                    overflow_behavior: None,
                    description: format!("Must not exceed {}", max),
                });
            }
    }

    fn extract_range_values(&self, docs: &str) -> Option<(i64, i64)> {
        // Pattern: "between X and Y" or "from X to Y"
        if let Ok(re) = regex::Regex::new(r"(?:between|from)\s+(-?\d+)\s+(?:and|to)\s+(-?\d+)")
            && let Some(cap) = re.captures(docs) {
                let min = cap.get(1)?.as_str().parse().ok()?;
                let max = cap.get(2)?.as_str().parse().ok()?;
                return Some((min, max));
            }

        // Pattern: "range X-Y" or "range X..Y"
        if let Ok(re) = regex::Regex::new(r"range\s+(-?\d+)(?:-|\.\.+)(-?\d+)")
            && let Some(cap) = re.captures(docs) {
                let min = cap.get(1)?.as_str().parse().ok()?;
                let max = cap.get(2)?.as_str().parse().ok()?;
                return Some((min, max));
            }

        None
    }

    fn extract_minimum_value(&self, docs: &str) -> Option<i64> {
        // Pattern: "at least N", "minimum N", "greater than N"
        let patterns = [
            r"at least\s+(-?\d+)",
            r"minimum\s+(?:of\s+)?(-?\d+)",
            r"greater than\s+(-?\d+)",
            r"must be\s+(-?\d+)\s+or\s+(?:more|greater)",
        ];

        for pattern in &patterns {
            if let Ok(re) = regex::Regex::new(pattern)
                && let Some(cap) = re.captures(docs) {
                    return cap.get(1)?.as_str().parse().ok();
                }
        }

        None
    }

    fn extract_maximum_value(&self, docs: &str) -> Option<i64> {
        // Pattern: "not exceed N", "maximum N", "less than N"
        let patterns = [
            r"not exceed\s+(-?\d+)",
            r"maximum\s+(?:of\s+)?(-?\d+)",
            r"less than\s+(-?\d+)",
            r"must be\s+(-?\d+)\s+or\s+(?:less|fewer)",
        ];

        for pattern in &patterns {
            if let Ok(re) = regex::Regex::new(pattern)
                && let Some(cap) = re.captures(docs) {
                    return cap.get(1)?.as_str().parse().ok();
                }
        }

        None
    }

    fn extract_power_of_two_constraints(
        &self,
        lower_docs: &str,
        constraints: &mut NumericConstraints,
    ) {
        if lower_docs.contains("power of 2")
            || lower_docs.contains("power of two")
            || lower_docs.contains("power-of-two")
        {
            constraints.add_constraint(NumericConstraint {
                constraint_type: ConstraintType::PowerOfTwo,
                parameter_name: None,
                min_value: None,
                max_value: None,
                must_be_power_of_two: true,
                must_be_multiple_of: None,
                alignment_bytes: None,
                can_overflow: false,
                overflow_behavior: None,
                description: "Must be a power of 2".to_string(),
            });
        }
    }

    fn extract_multiple_constraints(
        &self,
        docs: &str,
        lower_docs: &str,
        constraints: &mut NumericConstraints,
    ) {
        // Look for "must be a multiple of N" or "must be divisible by N"
        if (lower_docs.contains("multiple of") || lower_docs.contains("divisible by"))
            && let Some(multiple) = self.extract_multiple_value(docs) {
                constraints.add_constraint(NumericConstraint {
                    constraint_type: ConstraintType::Multiple,
                    parameter_name: None,
                    min_value: None,
                    max_value: None,
                    must_be_power_of_two: false,
                    must_be_multiple_of: Some(multiple),
                    alignment_bytes: None,
                    can_overflow: false,
                    overflow_behavior: None,
                    description: format!("Must be a multiple of {}", multiple),
                });
            }
    }

    fn extract_multiple_value(&self, docs: &str) -> Option<u64> {
        let patterns = [r"multiple of\s+(\d+)", r"divisible by\s+(\d+)"];

        for pattern in &patterns {
            if let Ok(re) = regex::Regex::new(pattern)
                && let Some(cap) = re.captures(docs) {
                    return cap.get(1)?.as_str().parse().ok();
                }
        }

        None
    }

    fn extract_sign_constraints(&self, lower_docs: &str, constraints: &mut NumericConstraints) {
        // Check for non-zero requirement
        if lower_docs.contains("non-zero")
            || lower_docs.contains("nonzero")
            || lower_docs.contains("must not be zero")
        {
            constraints.add_constraint(NumericConstraint {
                constraint_type: ConstraintType::NonZero,
                parameter_name: None,
                min_value: None,
                max_value: None,
                must_be_power_of_two: false,
                must_be_multiple_of: None,
                alignment_bytes: None,
                can_overflow: false,
                overflow_behavior: None,
                description: "Must not be zero".to_string(),
            });
        }

        // Check for positive requirement
        if lower_docs.contains("must be positive") || lower_docs.contains("positive value") {
            constraints.add_constraint(NumericConstraint {
                constraint_type: ConstraintType::Positive,
                parameter_name: None,
                min_value: Some(1),
                max_value: None,
                must_be_power_of_two: false,
                must_be_multiple_of: None,
                alignment_bytes: None,
                can_overflow: false,
                overflow_behavior: None,
                description: "Must be positive (> 0)".to_string(),
            });
        }

        // Check for non-negative requirement
        if lower_docs.contains("non-negative") || lower_docs.contains("nonnegative") {
            constraints.add_constraint(NumericConstraint {
                constraint_type: ConstraintType::NonNegative,
                parameter_name: None,
                min_value: Some(0),
                max_value: None,
                must_be_power_of_two: false,
                must_be_multiple_of: None,
                alignment_bytes: None,
                can_overflow: false,
                overflow_behavior: None,
                description: "Must be non-negative (>= 0)".to_string(),
            });
        }
    }

    fn extract_overflow_constraints(&self, lower_docs: &str, constraints: &mut NumericConstraints) {
        // Check for overflow warnings
        if lower_docs.contains("overflow") {
            let behavior =
                if lower_docs.contains("wraparound") || lower_docs.contains("wraps around") {
                    Some("Wraps around on overflow".to_string())
                } else if lower_docs.contains("saturate") {
                    Some("Saturates on overflow".to_string())
                } else if lower_docs.contains("undefined") {
                    Some("Undefined behavior on overflow".to_string())
                } else if lower_docs.contains("error") || lower_docs.contains("fails") {
                    Some("Returns error on overflow".to_string())
                } else {
                    None
                };

            constraints.add_constraint(NumericConstraint {
                constraint_type: ConstraintType::Overflow,
                parameter_name: None,
                min_value: None,
                max_value: None,
                must_be_power_of_two: false,
                must_be_multiple_of: None,
                alignment_bytes: None,
                can_overflow: true,
                overflow_behavior: behavior.clone(),
                description: behavior.unwrap_or_else(|| "May overflow".to_string()),
            });
        }

        // Check for underflow warnings
        if lower_docs.contains("underflow") {
            constraints.add_constraint(NumericConstraint {
                constraint_type: ConstraintType::Underflow,
                parameter_name: None,
                min_value: None,
                max_value: None,
                must_be_power_of_two: false,
                must_be_multiple_of: None,
                alignment_bytes: None,
                can_overflow: true,
                overflow_behavior: Some("May underflow".to_string()),
                description: "May underflow".to_string(),
            });
        }
    }

    /// Generate documentation from numeric constraints
    pub fn generate_documentation(&self, constraints: &NumericConstraints) -> String {
        if !constraints.has_constraints() {
            return String::new();
        }

        let mut doc = String::from("# Numeric Constraints\n\n");

        for constraint in &constraints.constraints {
            match constraint.constraint_type {
                ConstraintType::Alignment => {
                    doc.push_str(&format!("- **Alignment**: {}\n", constraint.description));
                }
                ConstraintType::Range => {
                    doc.push_str(&format!("- **Range**: {}\n", constraint.description));
                }
                ConstraintType::PowerOfTwo => {
                    doc.push_str("- **Power of 2**: Must be a power of 2\n");
                }
                ConstraintType::Multiple => {
                    doc.push_str(&format!("- **Multiple**: {}\n", constraint.description));
                }
                ConstraintType::NonZero => {
                    doc.push_str("- **Non-zero**: Value must not be zero\n");
                }
                ConstraintType::Positive => {
                    doc.push_str("- **Positive**: Value must be > 0\n");
                }
                ConstraintType::NonNegative => {
                    doc.push_str("- **Non-negative**: Value must be >= 0\n");
                }
                ConstraintType::Overflow => {
                    doc.push_str(&format!("- ⚠️ **Overflow**: {}\n", constraint.description));
                }
                ConstraintType::Underflow => {
                    doc.push_str(&format!("- ⚠️ **Underflow**: {}\n", constraint.description));
                }
            }
        }

        doc.push('\n');
        doc
    }

    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }
}

impl Default for NumericConstraintsAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_alignment_constraint() {
        let mut analyzer = NumericConstraintsAnalyzer::new();
        let docs = "Buffer must be aligned to 16 bytes.";

        let constraints = analyzer.analyze("allocate", docs);
        assert!(constraints.has_constraints());
        assert_eq!(
            constraints.constraints[0].constraint_type,
            ConstraintType::Alignment
        );
        assert_eq!(constraints.constraints[0].alignment_bytes, Some(16));
    }

    #[test]
    fn test_range_constraint() {
        let mut analyzer = NumericConstraintsAnalyzer::new();
        let docs = "Value must be between 0 and 100.";

        let constraints = analyzer.analyze("set_value", docs);
        assert!(constraints.has_constraints());
        assert_eq!(constraints.constraints[0].min_value, Some(0));
        assert_eq!(constraints.constraints[0].max_value, Some(100));
    }

    #[test]
    fn test_minimum_constraint() {
        let mut analyzer = NumericConstraintsAnalyzer::new();
        let docs = "Size must be at least 1024.";

        let constraints = analyzer.analyze("resize", docs);
        assert!(constraints.has_constraints());
        assert_eq!(constraints.constraints[0].min_value, Some(1024));
    }

    #[test]
    fn test_maximum_constraint() {
        let mut analyzer = NumericConstraintsAnalyzer::new();
        let docs = "Count must not exceed 256.";

        let constraints = analyzer.analyze("set_count", docs);
        assert!(constraints.has_constraints());
        assert_eq!(constraints.constraints[0].max_value, Some(256));
    }

    #[test]
    fn test_power_of_two() {
        let mut analyzer = NumericConstraintsAnalyzer::new();
        let docs = "Size must be a power of 2.";

        let constraints = analyzer.analyze("allocate", docs);
        assert!(constraints.has_constraints());
        assert_eq!(
            constraints.constraints[0].constraint_type,
            ConstraintType::PowerOfTwo
        );
        assert!(constraints.constraints[0].must_be_power_of_two);
    }

    #[test]
    fn test_multiple_constraint() {
        let mut analyzer = NumericConstraintsAnalyzer::new();
        let docs = "Length must be a multiple of 4.";

        let constraints = analyzer.analyze("write", docs);
        assert!(constraints.has_constraints());
        assert_eq!(
            constraints.constraints[0].constraint_type,
            ConstraintType::Multiple
        );
        assert_eq!(constraints.constraints[0].must_be_multiple_of, Some(4));
    }

    #[test]
    fn test_non_zero() {
        let mut analyzer = NumericConstraintsAnalyzer::new();
        let docs = "Divisor must be non-zero.";

        let constraints = analyzer.analyze("divide", docs);
        assert!(constraints.has_constraints());
        assert_eq!(
            constraints.constraints[0].constraint_type,
            ConstraintType::NonZero
        );
    }

    #[test]
    fn test_positive() {
        let mut analyzer = NumericConstraintsAnalyzer::new();
        let docs = "Count must be positive.";

        let constraints = analyzer.analyze("repeat", docs);
        assert!(constraints.has_constraints());
        assert_eq!(
            constraints.constraints[0].constraint_type,
            ConstraintType::Positive
        );
        assert_eq!(constraints.constraints[0].min_value, Some(1));
    }

    #[test]
    fn test_non_negative() {
        let mut analyzer = NumericConstraintsAnalyzer::new();
        let docs = "Offset must be non-negative.";

        let constraints = analyzer.analyze("seek", docs);
        assert!(constraints.has_constraints());
        assert_eq!(
            constraints.constraints[0].constraint_type,
            ConstraintType::NonNegative
        );
        assert_eq!(constraints.constraints[0].min_value, Some(0));
    }

    #[test]
    fn test_overflow_warning() {
        let mut analyzer = NumericConstraintsAnalyzer::new();
        let docs = "Operation may overflow. Wraps around on overflow.";

        let constraints = analyzer.analyze("add", docs);
        assert!(constraints.has_constraints());
        assert_eq!(
            constraints.constraints[0].constraint_type,
            ConstraintType::Overflow
        );
        assert!(constraints.constraints[0].can_overflow);
        assert!(constraints.constraints[0].overflow_behavior.is_some());
    }

    #[test]
    fn test_multiple_constraints() {
        let mut analyzer = NumericConstraintsAnalyzer::new();
        let docs = "Size must be at least 8 and a power of 2. Must be aligned to 16 bytes.";

        let constraints = analyzer.analyze("complex", docs);
        assert!(constraints.constraints.len() >= 2);
    }

    #[test]
    fn test_cache_functionality() {
        let mut analyzer = NumericConstraintsAnalyzer::new();
        let docs = "Must be positive.";

        let constraints1 = analyzer.analyze("test_fn", docs);
        let constraints2 = analyzer.analyze("test_fn", docs);

        assert_eq!(constraints1, constraints2);
    }

    #[test]
    fn test_generate_documentation() {
        let mut analyzer = NumericConstraintsAnalyzer::new();
        let docs = "Value must be between 0 and 100 and non-zero.";

        let constraints = analyzer.analyze("validate", docs);
        let doc = analyzer.generate_documentation(&constraints);

        assert!(doc.contains("Numeric Constraints"));
        assert!(doc.contains("Range") || doc.contains("Non-zero"));
    }
}
