//! Enhanced context for LLM prompts using enrichment data.

use crate::enrichment::{LibraryFiles, code_search, doc_parser, header_parser};
use std::collections::HashMap;

/// Enhanced context that combines multiple enrichment sources
#[derive(Debug, Clone, Default)]
pub struct EnhancedContext {
    /// Discovered documentation files
    pub library_files: Option<LibraryFiles>,
    /// Header comments from C/C++ files
    pub header_comments: HashMap<String, header_parser::FunctionComment>,
    /// Parsed documentation for functions
    pub parsed_docs: HashMap<String, doc_parser::FunctionDoc>,
    /// Usage patterns from code search
    pub usage_patterns: HashMap<String, code_search::UsagePattern>,
}

impl EnhancedContext {
    /// Create a new enhanced context
    pub fn new() -> Self {
        Self::default()
    }

    /// Set library files from discovery
    pub fn with_library_files(mut self, files: LibraryFiles) -> Self {
        self.library_files = Some(files);
        self
    }

    /// Add parsed documentation
    pub fn add_parsed_doc(&mut self, func_name: String, doc: doc_parser::FunctionDoc) {
        self.parsed_docs.insert(func_name, doc);
    }

    /// Add header comment
    pub fn add_header_comment(
        &mut self,
        func_name: String,
        comment: header_parser::FunctionComment,
    ) {
        self.header_comments.insert(func_name, comment);
    }

    /// Add usage pattern
    pub fn add_usage_pattern(&mut self, func_name: String, pattern: code_search::UsagePattern) {
        self.usage_patterns.insert(func_name, pattern);
    }

    /// Build enhanced context string for a function
    pub fn build_function_context(&self, function_name: &str, base_context: &str) -> String {
        let mut context = String::new();

        // Add base context
        context.push_str("Base Context:\n");
        context.push_str(base_context);
        context.push_str("\n\n");

        // Priority 1: Header comments (most authoritative - from source headers)
        if let Some(comment) = self.header_comments.get(function_name) {
            context.push_str("Header Documentation:\n");

            if let Some(brief) = &comment.brief {
                context.push_str(&format!("Brief: {}\n", brief));
            }

            if let Some(detailed) = &comment.detailed {
                context.push_str(&format!("Details: {}\n", detailed));
            }

            if !comment.param_docs.is_empty() {
                context.push_str("\nParameters:\n");
                for param in comment.param_docs.values() {
                    let dir_str = match param.direction {
                        Some(header_parser::ParamDirection::In) => " [in]",
                        Some(header_parser::ParamDirection::Out) => " [out]",
                        Some(header_parser::ParamDirection::InOut) => " [in/out]",
                        None => "",
                    };
                    context.push_str(&format!(
                        "  - {}{}: {}\n",
                        param.name, dir_str, param.description
                    ));
                }
            }

            if let Some(ret) = &comment.return_doc {
                context.push_str(&format!("\nReturns: {}\n", ret));
            }

            if !comment.notes.is_empty() {
                context.push_str("\nNotes:\n");
                for note in &comment.notes {
                    context.push_str(&format!("  - {}\n", note));
                }
            }

            if !comment.warnings.is_empty() {
                context.push_str("\nWarnings:\n");
                for warning in &comment.warnings {
                    context.push_str(&format!("  - {}\n", warning));
                }
            }

            if !comment.see_also.is_empty() {
                context.push_str("\nSee Also:\n");
                for see in &comment.see_also {
                    context.push_str(&format!("  - {}\n", see));
                }
            }

            if let Some(deprecated) = &comment.deprecated {
                context.push_str(&format!("\n⚠️ DEPRECATED: {}\n", deprecated));
            }

            context.push('\n');
        }

        // Priority 2: Parsed documentation (external docs - Doxygen/RST)
        if let Some(doc) = self.parsed_docs.get(function_name) {
            context.push_str("External Documentation:\n");

            if let Some(brief) = &doc.brief {
                context.push_str(&format!("Brief: {}\n", brief));
            }

            if let Some(detailed) = &doc.detailed {
                context.push_str(&format!("Details: {}\n", detailed));
            }

            if !doc.parameters.is_empty() {
                context.push_str("\nParameters:\n");
                for param in &doc.parameters {
                    context.push_str(&format!(
                        "  - {}: {} ({:?})\n",
                        param.name, param.description, param.direction
                    ));
                }
            }

            if let Some(ret) = &doc.return_doc {
                context.push_str(&format!("\nReturns: {}\n", ret));
            }

            context.push('\n');
        }

        // Add usage patterns if available
        if let Some(pattern) = self.usage_patterns.get(function_name) {
            context.push_str(&format!(
                "Real-world Usage ({} examples found):\n",
                pattern.occurrence_count
            ));

            // Add confidence score
            context.push_str(&format!("Confidence: {:?}\n", pattern.confidence));

            // Add best examples (limit to top 3)
            if !pattern.examples.is_empty() {
                context.push_str("\nCode Examples:\n");
                for (i, example) in pattern.examples.iter().take(3).enumerate() {
                    context.push_str(&format!(
                        "\nExample {} (from {}):\n",
                        i + 1,
                        example.repository
                    ));
                    context.push_str("```c\n");
                    context.push_str(&example.context);
                    context.push_str("\n```\n");
                }
            }

            // Add common error handling patterns
            if !pattern.error_handling.is_empty() {
                context.push_str("\nCommon error handling:\n");
                for eh in &pattern.error_handling {
                    context.push_str(&format!("  - {}\n", eh));
                }
            }

            context.push('\n');
        }

        // Add documentation file summary
        if let Some(files) = &self.library_files {
            if !files.documentation.is_empty() {
                context.push_str(&format!(
                    "Available documentation: {} files\n",
                    files.documentation.len()
                ));
            }
            if !files.examples.is_empty() {
                context.push_str(&format!(
                    "Available examples: {} files\n",
                    files.examples.len()
                ));
            }
        }

        context
    }

    /// Build enhanced context for error messages
    pub fn build_error_context(&self, error_name: &str, base_context: &str) -> String {
        let mut context = String::new();

        context.push_str(base_context);
        context.push_str("\n\n");

        // Check if we have documentation about this error
        for (func_name, doc) in &self.parsed_docs {
            if let Some(ret_doc) = &doc.return_doc
                && ret_doc.contains(error_name) {
                    context.push_str(&format!("Found in {} documentation:\n", func_name));
                    context.push_str(ret_doc);
                    context.push_str("\n\n");
                }
        }

        context
    }

    /// Check if we have any enrichment data
    pub fn has_enrichment(&self) -> bool {
        !self.header_comments.is_empty()
            || !self.parsed_docs.is_empty()
            || !self.usage_patterns.is_empty()
    }

    /// Get summary of available enrichment
    pub fn summary(&self) -> String {
        format!(
            "Enrichment: {} header comments, {} parsed docs, {} usage patterns, {} doc files, {} examples",
            self.header_comments.len(),
            self.parsed_docs.len(),
            self.usage_patterns.len(),
            self.library_files
                .as_ref()
                .map(|f| f.documentation.len())
                .unwrap_or(0),
            self.library_files
                .as_ref()
                .map(|f| f.examples.len())
                .unwrap_or(0)
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_enhanced_context_creation() {
        let context = EnhancedContext::new();
        assert!(!context.has_enrichment());
    }

    #[test]
    fn test_build_function_context() {
        let mut context = EnhancedContext::new();

        let mut func_doc = doc_parser::FunctionDoc::new("test_func".to_string());
        func_doc.brief = Some("Test function".to_string());
        func_doc.detailed = Some("This is a detailed description".to_string());

        context.add_parsed_doc("test_func".to_string(), func_doc);

        let enhanced = context.build_function_context("test_func", "Base info");
        assert!(enhanced.contains("Base Context"));
        assert!(enhanced.contains("Documentation"));
        assert!(enhanced.contains("Test function"));
    }

    #[test]
    fn test_has_enrichment() {
        let mut context = EnhancedContext::new();
        assert!(!context.has_enrichment());

        let func_doc = doc_parser::FunctionDoc::new("test".to_string());
        context.add_parsed_doc("test".to_string(), func_doc);

        assert!(context.has_enrichment());
    }

    #[test]
    fn test_header_comment_priority() {
        let mut context = EnhancedContext::new();

        // Add header comment
        let mut header_comment = header_parser::FunctionComment {
            function_name: "test_func".to_string(),
            brief: Some("Brief from header".to_string()),
            detailed: Some("Detailed from header".to_string()),
            param_docs: HashMap::new(),
            return_doc: Some("SUCCESS on success".to_string()),
            notes: vec!["Important note".to_string()],
            warnings: vec![],
            see_also: vec![],
            deprecated: None,
        };
        header_comment.param_docs.insert(
            "input".to_string(),
            header_parser::ParamDoc {
                name: "input".to_string(),
                description: "Input parameter".to_string(),
                direction: Some(header_parser::ParamDirection::In),
            },
        );
        context.add_header_comment("test_func".to_string(), header_comment);

        // Add parsed doc (should be lower priority)
        let mut func_doc = doc_parser::FunctionDoc::new("test_func".to_string());
        func_doc.brief = Some("Brief from parsed doc".to_string());
        context.add_parsed_doc("test_func".to_string(), func_doc);

        let enhanced = context.build_function_context("test_func", "Base info");

        // Header comment should appear first
        assert!(enhanced.contains("Header Documentation"));
        assert!(enhanced.contains("Brief from header"));
        assert!(enhanced.contains("input [in]: Input parameter"));
        assert!(enhanced.contains("Returns: SUCCESS on success"));
        assert!(enhanced.contains("Important note"));

        // Parsed doc should also appear but later
        assert!(enhanced.contains("External Documentation"));
    }

    #[test]
    fn test_header_comment_deprecated() {
        let mut context = EnhancedContext::new();

        let header_comment = header_parser::FunctionComment {
            function_name: "old_func".to_string(),
            brief: Some("Old function".to_string()),
            detailed: None,
            param_docs: HashMap::new(),
            return_doc: None,
            notes: vec![],
            warnings: vec!["Do not use in production".to_string()],
            see_also: vec!["new_func".to_string()],
            deprecated: Some("Use new_func instead".to_string()),
        };
        context.add_header_comment("old_func".to_string(), header_comment);

        let enhanced = context.build_function_context("old_func", "Base info");

        assert!(enhanced.contains("⚠️ DEPRECATED"));
        assert!(enhanced.contains("Use new_func instead"));
        assert!(enhanced.contains("Warnings"));
        assert!(enhanced.contains("Do not use in production"));
        assert!(enhanced.contains("See Also"));
        assert!(enhanced.contains("new_func"));
    }

    #[test]
    fn test_header_comment_only() {
        let mut context = EnhancedContext::new();

        let header_comment = header_parser::FunctionComment {
            function_name: "simple_func".to_string(),
            brief: Some("Simple function".to_string()),
            detailed: Some("Does something simple".to_string()),
            param_docs: HashMap::new(),
            return_doc: Some("Error code".to_string()),
            notes: vec![],
            warnings: vec![],
            see_also: vec![],
            deprecated: None,
        };
        context.add_header_comment("simple_func".to_string(), header_comment);

        let enhanced = context.build_function_context("simple_func", "Base info");

        assert!(enhanced.contains("Header Documentation"));
        assert!(enhanced.contains("Simple function"));
        assert!(enhanced.contains("Does something simple"));
        assert!(enhanced.contains("Returns: Error code"));

        // Should not have external docs section
        assert!(!enhanced.contains("External Documentation"));
    }

    #[test]
    fn test_enrichment_summary() {
        let mut context = EnhancedContext::new();

        // Add header comment
        let header_comment = header_parser::FunctionComment {
            function_name: "func1".to_string(),
            brief: Some("Function 1".to_string()),
            detailed: None,
            param_docs: HashMap::new(),
            return_doc: None,
            notes: vec![],
            warnings: vec![],
            see_also: vec![],
            deprecated: None,
        };
        context.add_header_comment("func1".to_string(), header_comment);

        // Add parsed doc
        let func_doc = doc_parser::FunctionDoc::new("func2".to_string());
        context.add_parsed_doc("func2".to_string(), func_doc);

        // Add usage pattern
        let pattern = code_search::UsagePattern::new("func3".to_string());
        context.add_usage_pattern("func3".to_string(), pattern);

        let summary = context.summary();
        assert!(summary.contains("1 header comment"));
        assert!(summary.contains("1 parsed doc"));
        assert!(summary.contains("1 usage pattern"));
    }
}
