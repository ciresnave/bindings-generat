use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs;
use std::path::PathBuf;
use tracing::{debug, info};

/// Cached LLM response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachedResponse {
    pub prompt_hash: String,
    pub model: String,
    pub response: String,
    pub timestamp: i64,
}

/// LLM response cache
pub struct Cache {
    cache_dir: PathBuf,
}

impl Cache {
    /// Create a new cache instance
    pub fn new(cache_dir: Option<PathBuf>) -> Result<Self> {
        let cache_dir = if let Some(dir) = cache_dir {
            dir
        } else {
            // Use platform-specific cache directory
            let mut dir = dirs::cache_dir().context("Failed to get system cache directory")?;
            dir.push("bindings-generat");
            dir.push("llm-cache");
            dir
        };

        // Create cache directory if it doesn't exist
        if !cache_dir.exists() {
            fs::create_dir_all(&cache_dir).context("Failed to create cache directory")?;
            info!("Created LLM cache directory: {}", cache_dir.display());
        }

        Ok(Self { cache_dir })
    }

    /// Generate cache key from prompt and model
    fn cache_key(&self, prompt: &str, model: &str) -> String {
        let mut hasher = Sha256::new();
        hasher.update(prompt.as_bytes());
        hasher.update(model.as_bytes());
        format!("{:x}", hasher.finalize())
    }

    /// Get cache file path for a given key
    fn cache_path(&self, key: &str) -> PathBuf {
        self.cache_dir.join(format!("{}.json", key))
    }

    /// Check if a cached response exists and is valid
    pub fn get(&self, prompt: &str, model: &str) -> Option<String> {
        let key = self.cache_key(prompt, model);
        let path = self.cache_path(&key);

        if !path.exists() {
            debug!("Cache miss: {} not found", key);
            return None;
        }

        // Read cached response
        let content = fs::read_to_string(&path).ok()?;
        let cached: CachedResponse = serde_json::from_str(&content).ok()?;

        // Verify model matches
        if cached.model != model {
            debug!("Cache miss: model mismatch");
            return None;
        }

        info!("Cache hit: {}", key);
        Some(cached.response)
    }

    /// Store a response in cache
    pub fn set(&self, prompt: &str, model: &str, response: String) -> Result<()> {
        let key = self.cache_key(prompt, model);
        let path = self.cache_path(&key);

        let cached = CachedResponse {
            prompt_hash: key.clone(),
            model: model.to_string(),
            response,
            timestamp: chrono::Utc::now().timestamp(),
        };

        let content =
            serde_json::to_string_pretty(&cached).context("Failed to serialize cached response")?;

        fs::write(&path, content).context("Failed to write cache file")?;

        debug!("Cached response: {}", key);
        Ok(())
    }

    /// Clear all cached responses
    pub fn clear(&self) -> Result<()> {
        if self.cache_dir.exists() {
            fs::remove_dir_all(&self.cache_dir).context("Failed to clear cache directory")?;
            fs::create_dir_all(&self.cache_dir).context("Failed to recreate cache directory")?;
            info!("Cleared LLM cache");
        }
        Ok(())
    }

    /// Get cache size in bytes
    pub fn size(&self) -> Result<u64> {
        let mut total_size = 0u64;

        if self.cache_dir.exists() {
            for entry in fs::read_dir(&self.cache_dir)? {
                let entry = entry?;
                if entry.file_type()?.is_file() {
                    total_size += entry.metadata()?.len();
                }
            }
        }

        Ok(total_size)
    }

    /// Get number of cached responses
    pub fn count(&self) -> Result<usize> {
        let mut count = 0;

        if self.cache_dir.exists() {
            for entry in fs::read_dir(&self.cache_dir)? {
                let entry = entry?;
                if entry.file_type()?.is_file() {
                    count += 1;
                }
            }
        }

        Ok(count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_cache_basic_operations() {
        let temp_dir = TempDir::new().unwrap();
        let cache = Cache::new(Some(temp_dir.path().to_path_buf())).unwrap();

        let prompt = "Test prompt";
        let model = "test-model";
        let response = "Test response";

        // Initially empty
        assert!(cache.get(prompt, model).is_none());
        assert_eq!(cache.count().unwrap(), 0);

        // Store response
        cache.set(prompt, model, response.to_string()).unwrap();

        // Should retrieve the same response
        assert_eq!(cache.get(prompt, model).unwrap(), response);
        assert_eq!(cache.count().unwrap(), 1);
    }

    #[test]
    fn test_cache_model_differentiation() {
        let temp_dir = TempDir::new().unwrap();
        let cache = Cache::new(Some(temp_dir.path().to_path_buf())).unwrap();

        let prompt = "Same prompt";
        let model1 = "model1";
        let model2 = "model2";

        cache.set(prompt, model1, "Response 1".to_string()).unwrap();
        cache.set(prompt, model2, "Response 2".to_string()).unwrap();

        // Different models should have different cached responses
        assert_eq!(cache.get(prompt, model1).unwrap(), "Response 1");
        assert_eq!(cache.get(prompt, model2).unwrap(), "Response 2");
        assert_eq!(cache.count().unwrap(), 2);
    }

    #[test]
    fn test_cache_clear() {
        let temp_dir = TempDir::new().unwrap();
        let cache = Cache::new(Some(temp_dir.path().to_path_buf())).unwrap();

        cache
            .set("prompt", "model", "response".to_string())
            .unwrap();
        assert_eq!(cache.count().unwrap(), 1);

        cache.clear().unwrap();
        assert_eq!(cache.count().unwrap(), 0);
    }
}
