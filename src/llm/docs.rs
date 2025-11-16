use anyhow::Result;
use tracing::{info, warn};

use super::client::OllamaClient;
use super::prompts;

/// Documentation enhancer using LLM
pub struct DocsEnhancer {
    client: OllamaClient,
    model: String,
}

impl DocsEnhancer {
    /// Create a new documentation enhancer
    pub fn new(model: String, cache_dir: Option<std::path::PathBuf>) -> Result<Self> {
        let client = OllamaClient::new(cache_dir)?;
        Ok(Self { client, model })
    }

    /// Create a new documentation enhancer with custom Ollama URL
    pub fn with_base_url(
        model: String,
        base_url: &str,
        cache_dir: Option<std::path::PathBuf>,
    ) -> Result<Self> {
        let client = OllamaClient::with_base_url(base_url, cache_dir)?;
        Ok(Self { client, model })
    }

    /// Check if LLM is available
    pub fn is_available(&self) -> bool {
        self.client.is_available()
    }

    /// Enhance documentation for a function
    pub fn enhance_function_docs(
        &self,
        function_name: &str,
        signature: &str,
        context: &str,
    ) -> Result<Option<String>> {
        if !self.is_available() {
            info!("LLM not available, skipping documentation enhancement");
            return Ok(None);
        }

        let prompt = prompts::documentation_prompt(function_name, signature, context);

        match self.client.generate(&self.model, &prompt) {
            Ok(docs) => {
                info!("Generated enhanced documentation for {}", function_name);
                Ok(Some(docs.trim().to_string()))
            }
            Err(e) => {
                warn!("Failed to generate documentation: {}", e);
                Ok(None)
            }
        }
    }

    /// Suggest better names for an item
    pub fn suggest_names(
        &self,
        c_name: &str,
        context: &str,
        item_type: &str,
    ) -> Result<Option<Vec<String>>> {
        if !self.is_available() {
            return Ok(None);
        }

        let prompt = prompts::naming_prompt(c_name, context, item_type);

        match self.client.generate(&self.model, &prompt) {
            Ok(response) => {
                // Parse JSON array
                match serde_json::from_str::<Vec<String>>(&response) {
                    Ok(names) => {
                        info!("Generated {} name suggestions for {}", names.len(), c_name);
                        Ok(Some(names))
                    }
                    Err(e) => {
                        warn!("Failed to parse naming suggestions: {}", e);
                        Ok(None)
                    }
                }
            }
            Err(e) => {
                warn!("Failed to generate name suggestions: {}", e);
                Ok(None)
            }
        }
    }

    /// Generate usage example for a type
    pub fn generate_example(
        &self,
        type_name: &str,
        methods: &[String],
        context: &str,
    ) -> Result<Option<String>> {
        if !self.is_available() {
            return Ok(None);
        }

        let prompt = prompts::example_prompt(type_name, methods, context);

        match self.client.generate(&self.model, &prompt) {
            Ok(example) => {
                info!("Generated example for {}", type_name);
                Ok(Some(example.trim().to_string()))
            }
            Err(e) => {
                warn!("Failed to generate example: {}", e);
                Ok(None)
            }
        }
    }

    /// Generate better error message
    pub fn enhance_error_message(&self, error_code: &str, c_name: &str) -> Result<Option<String>> {
        if !self.is_available() {
            return Ok(None);
        }

        let prompt = prompts::error_message_prompt(error_code, c_name);

        match self.client.generate(&self.model, &prompt) {
            Ok(message) => {
                info!("Enhanced error message for {}", error_code);
                Ok(Some(message.trim().to_string()))
            }
            Err(e) => {
                warn!("Failed to enhance error message: {}", e);
                Ok(None)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_docs_enhancer_creation() {
        let temp_dir = TempDir::new().unwrap();
        let enhancer = DocsEnhancer::new(
            "test-model".to_string(),
            Some(temp_dir.path().to_path_buf()),
        );
        assert!(enhancer.is_ok());
    }

    #[test]
    fn test_enhancer_handles_unavailable_llm() {
        let temp_dir = TempDir::new().unwrap();
        let enhancer = DocsEnhancer::new(
            "test-model".to_string(),
            Some(temp_dir.path().to_path_buf()),
        )
        .unwrap();

        // Should return None when LLM is not available
        let result = enhancer.enhance_function_docs("test_func", "fn test_func()", "Test function");
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), None);
    }
}
