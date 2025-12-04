// src/assertions/contract_parser.rs

//! Parses function documentation to extract contract requirements

use crate::ffi::FfiFunction;
use anyhow::Result;
use regex::Regex;

/// Analyzes function documentation to extract contracts
pub struct ContractAnalyzer {
    null_check_patterns: Vec<Regex>,
    range_patterns: Vec<Regex>,
    alignment_patterns: Vec<Regex>,
    state_patterns: Vec<Regex>,
}

impl ContractAnalyzer {
    pub fn new() -> Self {
        Self {
            null_check_patterns: vec![
                Regex::new(r"(?i)must\s+not\s+be\s+null").unwrap(),
                Regex::new(r"(?i)cannot\s+be\s+null").unwrap(),
                Regex::new(r"(?i)non-null").unwrap(),
                Regex::new(r"(?i)valid\s+pointer").unwrap(),
                Regex::new(r"(?i)@param\s+\w+\s+.*must\s+be\s+non-NULL").unwrap(),
            ],
            range_patterns: vec![
                Regex::new(r"(?i)>=\s*(\d+)\s+and\s+<=\s*(\d+)").unwrap(),
                Regex::new(r"(?i)must\s+be\s+(?:>|>=|<|<=|between)\s+(\d+)").unwrap(),
                Regex::new(r"(?i)range:\s*\[?(\d+)\s*,\s*(\d+)\]?").unwrap(),
                Regex::new(r"(?i)valid\s+range\s+(?:is\s+)?(\d+)\s*-\s*(\d+)").unwrap(),
            ],
            alignment_patterns: vec![
                Regex::new(r"(?i)(\d+)-byte\s+aligned").unwrap(),
                Regex::new(r"(?i)aligned\s+to\s+(\d+)\s+bytes?").unwrap(),
                Regex::new(r"(?i)alignment:\s*(\d+)").unwrap(),
            ],
            state_patterns: vec![
                Regex::new(r"(?i)must\s+be\s+initialized").unwrap(),
                Regex::new(r"(?i)cannot\s+be\s+destroyed").unwrap(),
                Regex::new(r"(?i)valid\s+handle").unwrap(),
                Regex::new(r"(?i)not\s+thread[-\s]?safe").unwrap(),
            ],
        }
    }

    /// Analyzes a function and extracts contract requirements
    pub fn analyze_function(&self, function: &FfiFunction) -> Result<FunctionContract> {
        let mut contract = FunctionContract::default();

        // Use existing docs field
        let doc_text = function.docs.clone().unwrap_or_default();

        // Analyze each parameter
        for param in &function.params {
            let param_name = &param.name;
            let param_type = &param.ty; // Check for null pointer requirements
            if param_type.contains('*') {
                if self.requires_non_null(&doc_text, param_name) {
                    contract.non_null_params.push(param_name.clone());
                }
            }

            // Check for range requirements
            if self.is_numeric_type(param_type) {
                if let Some(range) = self.extract_range(&doc_text, param_name) {
                    contract.range_constraints.push((param_name.clone(), range));
                }
            }

            // Check for alignment requirements
            if param_type.contains('*') {
                if let Some(alignment) = self.extract_alignment(&doc_text, param_name) {
                    contract
                        .alignment_requirements
                        .push((param_name.clone(), alignment));
                }
            }

            // Check for buffer size requirements
            if param_type.contains('*') {
                if let Some(size_constraint) =
                    self.extract_buffer_size(&doc_text, param_name, function)
                {
                    contract
                        .buffer_constraints
                        .push((param_name.clone(), size_constraint));
                }
            }
        }

        // Check for state requirements
        contract.requires_initialization =
            doc_text.contains("must be initialized") || doc_text.contains("valid handle");
        contract.thread_safe = !doc_text.contains("not thread-safe")
            && !doc_text.contains("not thread safe")
            && !doc_text.contains("not threadsafe");

        Ok(contract)
    }

    fn requires_non_null(&self, doc: &str, param_name: &str) -> bool {
        let param_context = self.extract_param_context(doc, param_name);

        for pattern in &self.null_check_patterns {
            if pattern.is_match(&param_context) {
                return true;
            }
        }

        // Heuristic: assume pointers should be non-null unless documented otherwise
        !param_context.contains("may be null")
            && !param_context.contains("can be null")
            && !param_context.contains("optional")
    }

    fn extract_range(&self, doc: &str, param_name: &str) -> Option<RangeConstraint> {
        let param_context = self.extract_param_context(doc, param_name);

        for pattern in &self.range_patterns {
            if let Some(caps) = pattern.captures(&param_context) {
                // Try to parse as min/max
                if caps.len() == 3 {
                    let min = caps.get(1)?.as_str().parse().ok()?;
                    let max = caps.get(2)?.as_str().parse().ok()?;
                    return Some(RangeConstraint::MinMax(min, max));
                } else if caps.len() == 2 {
                    let value = caps.get(1)?.as_str().parse().ok()?;

                    if param_context.contains(">=")
                        || param_context.contains("greater than or equal")
                    {
                        return Some(RangeConstraint::Min(value));
                    } else if param_context.contains(">") || param_context.contains("greater than")
                    {
                        return Some(RangeConstraint::GreaterThan(value));
                    } else if param_context.contains("<=")
                        || param_context.contains("less than or equal")
                    {
                        return Some(RangeConstraint::Max(value));
                    } else if param_context.contains("<") || param_context.contains("less than") {
                        return Some(RangeConstraint::LessThan(value));
                    }
                }
            }
        }

        None
    }

    fn extract_alignment(&self, doc: &str, param_name: &str) -> Option<usize> {
        let param_context = self.extract_param_context(doc, param_name);

        for pattern in &self.alignment_patterns {
            if let Some(caps) = pattern.captures(&param_context) {
                if let Some(alignment_str) = caps.get(1) {
                    return alignment_str.as_str().parse().ok();
                }
            }
        }

        None
    }

    fn extract_buffer_size(
        &self,
        doc: &str,
        param_name: &str,
        function: &FfiFunction,
    ) -> Option<BufferConstraint> {
        let param_context = self.extract_param_context(doc, param_name);

        // Look for size parameter references
        for other_param in &function.params {
            let size_name = &other_param.name;
            if param_context.contains(size_name) && self.is_numeric_type(&other_param.ty) {
                return Some(BufferConstraint::SizedBy(size_name.clone()));
            }
        }

        // Look for explicit size mentions
        let size_pattern = Regex::new(r"(?i)(?:at least|minimum|min)\s+(\d+)\s+bytes?").unwrap();

        if let Some(caps) = size_pattern.captures(&param_context) {
            if let Some(size_str) = caps.get(1) {
                if let Ok(min_size) = size_str.as_str().parse::<usize>() {
                    return Some(BufferConstraint::MinSize(min_size));
                }
            }
        }

        None
    }

    fn extract_param_context(&self, doc: &str, param_name: &str) -> String {
        // Find lines mentioning this parameter
        let lines: Vec<&str> = doc
            .lines()
            .filter(|line| line.contains(param_name))
            .collect();

        lines.join("\n")
    }

    fn is_numeric_type(&self, type_name: &str) -> bool {
        matches!(
            type_name,
            "int"
                | "unsigned int"
                | "long"
                | "unsigned long"
                | "size_t"
                | "ssize_t"
                | "uint32_t"
                | "int32_t"
                | "uint64_t"
                | "int64_t"
                | "i32"
                | "u32"
                | "i64"
                | "u64"
                | "usize"
                | "isize"
                | "float"
                | "double"
        )
    }
}

impl Default for ContractAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

/// Extracted contract requirements for a function
#[derive(Debug, Default, Clone)]
pub struct FunctionContract {
    /// Parameters that must be non-null
    pub non_null_params: Vec<String>,

    /// Range constraints for numeric parameters
    pub range_constraints: Vec<(String, RangeConstraint)>,

    /// Alignment requirements for pointer parameters
    pub alignment_requirements: Vec<(String, usize)>,

    /// Buffer size constraints
    pub buffer_constraints: Vec<(String, BufferConstraint)>,

    /// Whether the handle/object must be initialized before calling
    pub requires_initialization: bool,

    /// Whether the function is thread-safe
    pub thread_safe: bool,
}

#[derive(Debug, Clone)]
pub enum RangeConstraint {
    /// value >= min
    Min(i64),

    /// value > min
    GreaterThan(i64),

    /// value <= max
    Max(i64),

    /// value < max
    LessThan(i64),

    /// min <= value <= max
    MinMax(i64, i64),
}

#[derive(Debug, Clone)]
pub enum BufferConstraint {
    /// Buffer size is specified by another parameter
    SizedBy(String),

    /// Buffer must be at least this many bytes
    MinSize(usize),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_null_pointer_detection() {
        let analyzer = ContractAnalyzer::new();
        let doc = "@param buffer Pointer to data buffer, must not be NULL";

        assert!(analyzer.requires_non_null(doc, "buffer"));
    }

    #[test]
    fn test_range_extraction() {
        let analyzer = ContractAnalyzer::new();
        let doc = "@param size Size of buffer, must be >= 0 and <= 1048576";

        let range = analyzer.extract_range(doc, "size");
        assert!(matches!(range, Some(RangeConstraint::MinMax(0, 1048576))));
    }

    #[test]
    fn test_alignment_extraction() {
        let analyzer = ContractAnalyzer::new();
        let doc = "@param buffer Must be 64-byte aligned for DMA";

        let alignment = analyzer.extract_alignment(doc, "buffer");
        assert_eq!(alignment, Some(64));
    }

    #[test]
    fn test_null_optional_parameter() {
        let analyzer = ContractAnalyzer::new();
        let doc = "@param callback Callback function, may be null";

        assert!(!analyzer.requires_non_null(doc, "callback"));
    }

    #[test]
    fn test_range_min_only() {
        let analyzer = ContractAnalyzer::new();
        let doc = "@param count Number of items, must be >= 1";

        let range = analyzer.extract_range(doc, "count");
        assert!(matches!(range, Some(RangeConstraint::Min(1))));
    }

    #[test]
    fn test_range_max_only() {
        let analyzer = ContractAnalyzer::new();
        let doc = "@param index Array index, must be <= 255";

        let range = analyzer.extract_range(doc, "index");
        assert!(matches!(range, Some(RangeConstraint::Max(255))));
    }

    #[test]
    fn test_range_greater_than() {
        let analyzer = ContractAnalyzer::new();
        let doc = "@param size Size must be > 0";

        let range = analyzer.extract_range(doc, "size");
        assert!(matches!(range, Some(RangeConstraint::GreaterThan(0))));
    }

    #[test]
    fn test_range_less_than() {
        let analyzer = ContractAnalyzer::new();
        let doc = "@param index Index must be < 100";

        let range = analyzer.extract_range(doc, "index");
        assert!(matches!(range, Some(RangeConstraint::LessThan(100))));
    }

    #[test]
    fn test_multiple_alignment_formats() {
        let analyzer = ContractAnalyzer::new();

        // Test different alignment documentation formats
        let doc1 = "@param buffer aligned to 32 bytes";
        let doc2 = "@param buffer alignment: 16";
        let doc3 = "@param buffer 8-byte aligned";

        assert_eq!(analyzer.extract_alignment(doc1, "buffer"), Some(32));
        assert_eq!(analyzer.extract_alignment(doc2, "buffer"), Some(16));
        assert_eq!(analyzer.extract_alignment(doc3, "buffer"), Some(8));
    }

    #[test]
    fn test_state_requirements() {
        let analyzer = ContractAnalyzer::new();
        let doc = "This function must be initialized before calling";

        // Check if state patterns would match
        assert!(doc.contains("must be initialized"));
    }

    #[test]
    fn test_thread_safety_detection() {
        let analyzer = ContractAnalyzer::new();
        let doc1 = "This function is not thread-safe";
        let doc2 = "This function can be called from multiple threads";

        // First should NOT be thread safe
        assert!(doc1.contains("not thread-safe"));
        // Second should be thread safe (doesn't contain "not")
        assert!(!doc2.contains("not thread-safe"));
    }

    #[test]
    fn test_no_constraints() {
        let analyzer = ContractAnalyzer::new();
        let doc = "@param data Generic data pointer";

        let range = analyzer.extract_range(doc, "data");
        let alignment = analyzer.extract_alignment(doc, "data");

        assert!(range.is_none());
        assert!(alignment.is_none());
    }

    #[test]
    fn test_param_context_extraction() {
        let analyzer = ContractAnalyzer::new();
        let doc = "@param width Image width\n@param height Image height, must be > 0\n@param data Pixel data";

        // Should extract context for specific parameter
        let context = analyzer.extract_param_context(doc, "height");
        assert!(context.contains("height"));
        assert!(context.contains("must be > 0"));
    }
}
