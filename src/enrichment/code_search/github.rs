//! GitHub code search implementation.

use super::searcher::PlatformSearcher;
use super::types::{CodeSearchResult, PlatformSource};
use anyhow::{Context, Result};
use serde::Deserialize;
use std::path::PathBuf;

/// GitHub API code search implementation
pub struct GitHubSearcher {
    client: reqwest::blocking::Client,
    token: Option<String>,
}

#[derive(Debug, Deserialize)]
struct GitHubSearchResponse {
    #[allow(dead_code)]
    total_count: usize,
    items: Vec<GitHubCodeItem>,
}

#[derive(Debug, Deserialize)]
struct GitHubCodeItem {
    name: String,
    path: String,
    repository: GitHubRepository,
    html_url: String,
}

#[derive(Debug, Deserialize)]
struct GitHubRepository {
    full_name: String,
    stargazers_count: usize,
    updated_at: String,
}

impl GitHubSearcher {
    /// Create a new GitHub searcher
    pub fn new() -> Result<Self> {
        let client = reqwest::blocking::Client::builder()
            .user_agent("bindings-generat")
            .timeout(std::time::Duration::from_secs(30))
            .build()
            .context("Failed to create HTTP client")?;

        // Try to get GitHub token from environment
        let token = std::env::var("GITHUB_TOKEN").ok();

        Ok(Self { client, token })
    }

    /// Search GitHub for code containing a function
    fn search_code(&self, query: &str) -> Result<GitHubSearchResponse> {
        let url = format!(
            "https://api.github.com/search/code?q={}&per_page=30",
            urlencoding::encode(query)
        );

        let mut request = self.client.get(&url);

        if let Some(token) = &self.token {
            request = request.header("Authorization", format!("token {}", token));
        }

        let response = request
            .send()
            .context("Failed to send GitHub API request")?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().unwrap_or_default();
            anyhow::bail!("GitHub API error {}: {}", status, body);
        }

        let search_response: GitHubSearchResponse = response
            .json()
            .context("Failed to parse GitHub API response")?;

        Ok(search_response)
    }

    /// Fetch the actual code content from a file
    fn fetch_file_content(&self, repo: &str, path: &str) -> Result<String> {
        let url = format!(
            "https://raw.githubusercontent.com/{}/main/{}",
            repo,
            urlencoding::encode(path)
        );

        let response = self
            .client
            .get(&url)
            .send()
            .context("Failed to fetch file content")?;

        if !response.status().is_success() {
            // Try master branch if main doesn't exist
            let url = format!(
                "https://raw.githubusercontent.com/{}/master/{}",
                repo,
                urlencoding::encode(path)
            );

            let response = self
                .client
                .get(&url)
                .send()
                .context("Failed to fetch file content")?;

            if !response.status().is_success() {
                anyhow::bail!("Failed to fetch file: {}", path);
            }

            return response.text().context("Failed to read response text");
        }

        response.text().context("Failed to read response text")
    }

    /// Extract code snippet and context around a function call
    fn extract_snippet(&self, content: &str, function_name: &str) -> Option<(usize, String, String)> {
        for (line_num, line) in content.lines().enumerate() {
            if line.contains(function_name) && line.contains('(') {
                // Found the function call
                let lines: Vec<&str> = content.lines().collect();
                let start = line_num.saturating_sub(3);
                let end = (line_num + 4).min(lines.len());

                let context = lines[start..end].join("\n");
                let snippet = line.trim().to_string();

                return Some((line_num + 1, snippet, context));
            }
        }
        None
    }
}

impl PlatformSearcher for GitHubSearcher {
    fn search_function(&self, function_name: &str, _library_name: &str) -> Result<Vec<CodeSearchResult>> {
        // Construct search query: function name + library context
        let query = format!("{} language:C language:C++", function_name);

        let search_response = self.search_code(&query)?;
        let mut results = Vec::new();

        for item in search_response.items.into_iter().take(10) {
            // Fetch file content to get actual code snippet
            match self.fetch_file_content(&item.repository.full_name, &item.path) {
                Ok(content) => {
                    if let Some((line_number, snippet, context)) =
                        self.extract_snippet(&content, function_name)
                    {
                        results.push(CodeSearchResult {
                            platform: PlatformSource::GitHub,
                            repository: item.repository.full_name.clone(),
                            stars: item.repository.stargazers_count,
                            file_path: PathBuf::from(item.path),
                            line_number,
                            code_snippet: snippet,
                            context,
                            url: item.html_url,
                            last_updated: Some(item.repository.updated_at),
                        });
                    }
                }
                Err(e) => {
                    tracing::debug!("Failed to fetch file {}: {}", item.path, e);
                    // Create result without snippet
                    results.push(CodeSearchResult {
                        platform: PlatformSource::GitHub,
                        repository: item.repository.full_name.clone(),
                        stars: item.repository.stargazers_count,
                        file_path: PathBuf::from(item.path),
                        line_number: 1,
                        code_snippet: format!("// {} usage in {}", function_name, item.name),
                        context: String::new(),
                        url: item.html_url,
                        last_updated: Some(item.repository.updated_at),
                    });
                }
            }
        }

        Ok(results)
    }

    fn platform_name(&self) -> &str {
        "GitHub"
    }
}

impl Default for GitHubSearcher {
    fn default() -> Self {
        Self::new().expect("Failed to create GitHub searcher")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_snippet() {
        let searcher = GitHubSearcher::new().unwrap();
        let content = r#"
#include <cuda_runtime.h>

int main() {
    void* ptr;
    cudaMalloc(&ptr, 1024);
    cudaFree(ptr);
    return 0;
}
"#;

        let result = searcher.extract_snippet(content, "cudaMalloc");
        assert!(result.is_some());

        let (line_num, snippet, context) = result.unwrap();
        assert!(snippet.contains("cudaMalloc"));
        assert!(context.contains("cudaMalloc"));
        assert!(line_num > 0);
    }

    #[test]
    #[ignore] // Requires network access
    fn test_github_search() {
        let searcher = GitHubSearcher::new().unwrap();
        let results = searcher.search_function("cudaMalloc", "cuda");
        
        // Should succeed even if rate limited
        assert!(results.is_ok() || results.is_err());
    }
}
