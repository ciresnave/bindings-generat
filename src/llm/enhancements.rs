use std::collections::HashMap;

/// Storage for LLM-generated enhancements
#[derive(Debug, Clone, Default)]
pub struct CodeEnhancements {
    /// Enhanced documentation for functions (function_name -> docs)
    pub function_docs: HashMap<String, String>,

    /// Suggested names for functions (c_name -> rust_name)
    pub function_names: HashMap<String, String>,

    /// Enhanced error messages (variant_name -> message)
    pub error_messages: HashMap<String, String>,

    /// Usage examples for functions (function_name -> example_code)
    pub usage_examples: HashMap<String, String>,
}

impl CodeEnhancements {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add enhanced documentation for a function
    pub fn add_function_docs(&mut self, function_name: String, docs: String) {
        self.function_docs.insert(function_name, docs);
    }

    /// Add a suggested name for a function
    pub fn add_function_name(&mut self, c_name: String, rust_name: String) {
        self.function_names.insert(c_name, rust_name);
    }

    /// Add an enhanced error message
    pub fn add_error_message(&mut self, variant_name: String, message: String) {
        self.error_messages.insert(variant_name, message);
    }

    /// Add a usage example
    pub fn add_usage_example(&mut self, function_name: String, example: String) {
        self.usage_examples.insert(function_name, example);
    }

    /// Get documentation for a function
    pub fn get_function_docs(&self, function_name: &str) -> Option<&String> {
        self.function_docs.get(function_name)
    }

    /// Get suggested name for a function
    pub fn get_function_name(&self, c_name: &str) -> Option<&String> {
        self.function_names.get(c_name)
    }

    /// Get error message for a variant
    pub fn get_error_message(&self, variant_name: &str) -> Option<&String> {
        self.error_messages.get(variant_name)
    }

    /// Get usage example for a function
    pub fn get_usage_example(&self, function_name: &str) -> Option<&String> {
        self.usage_examples.get(function_name)
    }

    /// Check if enhancements are empty
    pub fn is_empty(&self) -> bool {
        self.function_docs.is_empty()
            && self.function_names.is_empty()
            && self.error_messages.is_empty()
            && self.usage_examples.is_empty()
    }

    /// Get total count of enhancements
    pub fn count(&self) -> usize {
        self.function_docs.len()
            + self.function_names.len()
            + self.error_messages.len()
            + self.usage_examples.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_enhancements_creation() {
        let enhancements = CodeEnhancements::new();
        assert!(enhancements.is_empty());
        assert_eq!(enhancements.count(), 0);
    }

    #[test]
    fn test_add_and_get_function_docs() {
        let mut enhancements = CodeEnhancements::new();
        enhancements.add_function_docs(
            "foo_create".to_string(),
            "Creates a new Foo instance".to_string(),
        );

        assert_eq!(
            enhancements.get_function_docs("foo_create"),
            Some(&"Creates a new Foo instance".to_string())
        );
        assert_eq!(enhancements.count(), 1);
    }

    #[test]
    fn test_add_function_name() {
        let mut enhancements = CodeEnhancements::new();
        enhancements.add_function_name("foo_bar_baz".to_string(), "bar_baz".to_string());

        assert_eq!(
            enhancements.get_function_name("foo_bar_baz"),
            Some(&"bar_baz".to_string())
        );
    }

    #[test]
    fn test_add_error_message() {
        let mut enhancements = CodeEnhancements::new();
        enhancements.add_error_message(
            "ErrorInvalidInput".to_string(),
            "The provided input was invalid".to_string(),
        );

        assert_eq!(
            enhancements.get_error_message("ErrorInvalidInput"),
            Some(&"The provided input was invalid".to_string())
        );
    }
}
