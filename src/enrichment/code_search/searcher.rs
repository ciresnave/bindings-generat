//! Main code search interface and orchestration.

use super::types::{CodeSearchResult, UsagePattern};
use anyhow::Result;
use std::collections::HashMap;

/// Trait for platform-specific code searchers
pub trait PlatformSearcher: Send + Sync {
    /// Search for usage of a function on this platform
    fn search_function(&self, function_name: &str, library_name: &str) -> Result<Vec<CodeSearchResult>>;
    
    /// Get the platform name
    fn platform_name(&self) -> &str;
}

/// Multi-platform usage searcher
pub struct UsageSearcher {
    searchers: Vec<Box<dyn PlatformSearcher>>,
    max_results_per_platform: usize,
}

impl UsageSearcher {
    /// Create a new usage searcher with default platforms
    pub fn new() -> Self {
        Self {
            searchers: Vec::new(),
            max_results_per_platform: 50,
        }
    }

    /// Add a platform searcher
    pub fn add_searcher(&mut self, searcher: Box<dyn PlatformSearcher>) {
        self.searchers.push(searcher);
    }

    /// Set maximum results per platform
    pub fn set_max_results(&mut self, max: usize) {
        self.max_results_per_platform = max;
    }

    /// Search for usage patterns of a function across all platforms
    pub fn search_usage_patterns(
        &self,
        function_name: &str,
        library_name: &str,
    ) -> Result<UsagePattern> {
        let mut pattern = UsagePattern::new(function_name.to_string());

        // Search all platforms in parallel
        let results: Vec<Result<Vec<CodeSearchResult>>> = self
            .searchers
            .iter()
            .map(|searcher| searcher.search_function(function_name, library_name))
            .collect();

        // Collect all successful results
        for result in results {
            match result {
                Ok(search_results) => {
                    for search_result in search_results.into_iter().take(self.max_results_per_platform) {
                        pattern.add_result(search_result);
                    }
                }
                Err(e) => {
                    tracing::warn!("Platform search failed: {}", e);
                }
            }
        }

        Ok(pattern)
    }

    /// Search for multiple functions at once
    pub fn search_multiple_functions(
        &self,
        function_names: &[String],
        library_name: &str,
    ) -> Result<HashMap<String, UsagePattern>> {
        let mut patterns = HashMap::new();

        for function_name in function_names {
            match self.search_usage_patterns(function_name, library_name) {
                Ok(pattern) => {
                    patterns.insert(function_name.clone(), pattern);
                }
                Err(e) => {
                    tracing::warn!("Failed to search for {}: {}", function_name, e);
                }
            }
        }

        Ok(patterns)
    }
}

impl Default for UsageSearcher {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::enrichment::code_search::types::PlatformSource;
    use std::path::PathBuf;

    struct MockSearcher {
        name: String,
    }

    impl PlatformSearcher for MockSearcher {
        fn search_function(&self, function_name: &str, _library_name: &str) -> Result<Vec<CodeSearchResult>> {
            Ok(vec![CodeSearchResult {
                platform: PlatformSource::Other(self.name.clone()),
                repository: "test/repo".to_string(),
                stars: 100,
                file_path: PathBuf::from("test.c"),
                line_number: 1,
                code_snippet: format!("{}();", function_name),
                context: format!("/* test */\n{}();", function_name),
                url: "https://example.com".to_string(),
                last_updated: None,
            }])
        }

        fn platform_name(&self) -> &str {
            &self.name
        }
    }

    #[test]
    fn test_usage_searcher() {
        let mut searcher = UsageSearcher::new();
        searcher.add_searcher(Box::new(MockSearcher {
            name: "Mock1".to_string(),
        }));
        searcher.add_searcher(Box::new(MockSearcher {
            name: "Mock2".to_string(),
        }));

        let pattern = searcher.search_usage_patterns("test_func", "test_lib").unwrap();
        assert_eq!(pattern.function_name, "test_func");
        assert_eq!(pattern.occurrence_count, 2); // One from each mock searcher
    }
}
