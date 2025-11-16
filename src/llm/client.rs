use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::time::Duration;
use tracing::{debug, info, warn};

use super::cache::Cache;

/// Ollama API endpoint
const OLLAMA_BASE_URL: &str = "http://localhost:11434";

/// Ollama generate request
#[derive(Debug, Serialize)]
struct GenerateRequest {
    model: String,
    prompt: String,
    stream: bool,
    options: GenerateOptions,
}

/// Generation options
#[derive(Debug, Serialize)]
struct GenerateOptions {
    temperature: f32,
    top_p: f32,
    top_k: i32,
}

impl Default for GenerateOptions {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            top_p: 0.9,
            top_k: 40,
        }
    }
}

/// Ollama generate response
#[derive(Debug, Deserialize)]
struct GenerateResponse {
    response: String,
    #[allow(dead_code)]
    done: bool,
}

/// Ollama client with retry logic and caching
pub struct OllamaClient {
    client: reqwest::blocking::Client,
    base_url: String,
    cache: Option<Cache>,
    max_retries: u32,
}

impl OllamaClient {
    /// Create a new Ollama client
    pub fn new(cache_dir: Option<std::path::PathBuf>) -> Result<Self> {
        Self::with_base_url(OLLAMA_BASE_URL, cache_dir)
    }

    /// Create a client with a custom base URL (for portable installations)
    pub fn with_base_url(base_url: &str, cache_dir: Option<std::path::PathBuf>) -> Result<Self> {
        let client = reqwest::blocking::Client::builder()
            .timeout(Duration::from_secs(120)) // 2 minute timeout for LLM responses
            .build()
            .context("Failed to create HTTP client")?;

        let cache = if let Some(dir) = cache_dir {
            Some(Cache::new(Some(dir))?)
        } else {
            Some(Cache::new(None)?)
        };

        Ok(Self {
            client,
            base_url: base_url.to_string(),
            cache,
            max_retries: 3,
        })
    }

    /// Create a client without caching
    pub fn without_cache() -> Result<Self> {
        let client = reqwest::blocking::Client::builder()
            .timeout(Duration::from_secs(120))
            .build()
            .context("Failed to create HTTP client")?;

        Ok(Self {
            client,
            base_url: OLLAMA_BASE_URL.to_string(),
            cache: None,
            max_retries: 3,
        })
    }

    /// Check if Ollama is available
    pub fn is_available(&self) -> bool {
        debug!("Checking if Ollama is available");

        let result = self
            .client
            .get(format!("{}/api/version", self.base_url))
            .timeout(Duration::from_secs(2))
            .send();

        match result {
            Ok(response) if response.status().is_success() => {
                info!("Ollama is available");
                true
            }
            Ok(response) => {
                warn!("Ollama returned status: {}", response.status());
                false
            }
            Err(e) => {
                debug!("Ollama is not available: {}", e);
                false
            }
        }
    }

    /// Generate completion with retry logic
    pub fn generate(&self, model: &str, prompt: &str) -> Result<String> {
        // Check cache first
        if let Some(cache) = &self.cache
            && let Some(cached) = cache.get(prompt, model)
        {
            info!("Using cached LLM response");
            return Ok(cached);
        }

        // Generate with retry
        let response = self.generate_with_retry(model, prompt)?;

        // Cache the response
        if let Some(cache) = &self.cache
            && let Err(e) = cache.set(prompt, model, response.clone())
        {
            warn!("Failed to cache LLM response: {}", e);
        }

        Ok(response)
    }

    /// Generate with exponential backoff retry
    fn generate_with_retry(&self, model: &str, prompt: &str) -> Result<String> {
        let mut last_error = None;

        for attempt in 0..self.max_retries {
            if attempt > 0 {
                let delay = Duration::from_secs(2u64.pow(attempt));
                info!(
                    "Retrying after {} seconds... (attempt {})",
                    delay.as_secs(),
                    attempt + 1
                );
                std::thread::sleep(delay);
            }

            match self.generate_once(model, prompt) {
                Ok(response) => return Ok(response),
                Err(e) => {
                    warn!("LLM generation attempt {} failed: {}", attempt + 1, e);
                    last_error = Some(e);
                }
            }
        }

        Err(last_error.unwrap_or_else(|| anyhow::anyhow!("All retry attempts failed")))
    }

    /// Single generation attempt
    fn generate_once(&self, model: &str, prompt: &str) -> Result<String> {
        debug!("Sending request to Ollama (model: {})", model);

        let request = GenerateRequest {
            model: model.to_string(),
            prompt: prompt.to_string(),
            stream: false,
            options: GenerateOptions::default(),
        };

        let response = self
            .client
            .post(format!("{}/api/generate", self.base_url))
            .json(&request)
            .send()
            .context("Failed to send request to Ollama")?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response
                .text()
                .unwrap_or_else(|_| String::from("(no body)"));
            anyhow::bail!("Ollama returned error {}: {}", status, body);
        }

        let generate_response: GenerateResponse =
            response.json().context("Failed to parse Ollama response")?;

        debug!(
            "Received response from Ollama ({} bytes)",
            generate_response.response.len()
        );

        Ok(generate_response.response)
    }

    /// List available models
    pub fn list_models(&self) -> Result<Vec<String>> {
        #[derive(Deserialize)]
        struct Model {
            name: String,
        }

        #[derive(Deserialize)]
        struct ModelsResponse {
            models: Vec<Model>,
        }

        let response = self
            .client
            .get(format!("{}/api/tags", self.base_url))
            .send()
            .context("Failed to list models")?;

        let models_response: ModelsResponse =
            response.json().context("Failed to parse models response")?;

        Ok(models_response.models.into_iter().map(|m| m.name).collect())
    }
}

/// Check if Ollama is available (convenience function)
pub fn is_ollama_available() -> bool {
    OllamaClient::without_cache()
        .map(|client| client.is_available())
        .unwrap_or(false)
}

/// Query Ollama for documentation enhancement
pub fn enhance_documentation(code: &str, model: &str) -> Result<String> {
    let client = OllamaClient::new(None)?;

    if !client.is_available() {
        anyhow::bail!("Ollama is not available");
    }

    let prompt = super::prompts::documentation_prompt(
        "wrapper_function",
        code,
        "Safe Rust wrapper for C library",
    );

    client.generate(model, &prompt)
}

/// Query Ollama for better naming suggestions
pub fn suggest_names(function_name: &str, context: &str, model: &str) -> Result<Vec<String>> {
    let client = OllamaClient::new(None)?;

    if !client.is_available() {
        anyhow::bail!("Ollama is not available");
    }

    let prompt = super::prompts::naming_prompt(function_name, context, "function");
    let response = client.generate(model, &prompt)?;

    // Parse JSON array response
    let names: Vec<String> =
        serde_json::from_str(&response).context("Failed to parse naming suggestions as JSON")?;

    Ok(names)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ollama_client_creation() {
        let client = OllamaClient::without_cache();
        assert!(client.is_ok());
    }

    #[test]
    fn test_is_available_does_not_panic() {
        // Should not panic even if Ollama is not running
        let _ = is_ollama_available();
    }
}
