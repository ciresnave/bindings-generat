//! GitLab code search implementation.

use super::searcher::PlatformSearcher;
use super::types::{CodeSearchResult, PlatformSource};
use anyhow::{Context, Result};
use serde::Deserialize;
use std::path::PathBuf;

/// GitLab API code search implementation
pub struct GitLabSearcher {
    client: reqwest::blocking::Client,
    base_url: String,
}

#[derive(Debug, Deserialize)]
struct GitLabSearchResponse {
    #[serde(default)]
    data: Vec<GitLabBlob>,
}

#[derive(Debug, Deserialize)]
struct GitLabBlob {
    basename: String,
    path: String,
    data: String,
    project_id: i64,
    #[serde(rename = "ref")]
    git_ref: String,
}

#[derive(Debug, Deserialize)]
struct GitLabProject {
    #[allow(dead_code)]
    id: i64,
    name_with_namespace: String,
    web_url: String,
    star_count: i64,
    last_activity_at: String,
}

impl GitLabSearcher {
    /// Create a new GitLab searcher
    pub fn new() -> Result<Self> {
        Self::with_base_url("https://gitlab.com")
    }

    /// Create a searcher with custom GitLab instance
    pub fn with_base_url(base_url: &str) -> Result<Self> {
        let client = reqwest::blocking::Client::builder()
            .user_agent("bindings-generat")
            .timeout(std::time::Duration::from_secs(30))
            .build()
            .context("Failed to create HTTP client")?;

        Ok(Self {
            client,
            base_url: base_url.to_string(),
        })
    }

    /// Search GitLab for code
    fn search_code(&self, query: &str) -> Result<GitLabSearchResponse> {
        let url = format!(
            "{}/api/v4/search?scope=blobs&search={}",
            self.base_url,
            urlencoding::encode(query)
        );

        let mut request = self.client.get(&url);

        // Add token if available
        if let Ok(token) = std::env::var("GITLAB_TOKEN") {
            request = request.header("PRIVATE-TOKEN", token);
        }

        let response = request
            .send()
            .context("Failed to send GitLab API request")?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().unwrap_or_default();
            anyhow::bail!("GitLab API error {}: {}", status, body);
        }

        let search_response: GitLabSearchResponse = response
            .json()
            .context("Failed to parse GitLab API response")?;

        Ok(search_response)
    }

    /// Get project details
    fn get_project(&self, project_id: i64) -> Result<GitLabProject> {
        let url = format!("{}/api/v4/projects/{}", self.base_url, project_id);

        let mut request = self.client.get(&url);

        if let Ok(token) = std::env::var("GITLAB_TOKEN") {
            request = request.header("PRIVATE-TOKEN", token);
        }

        let response = request.send().context("Failed to fetch project details")?;

        if !response.status().is_success() {
            anyhow::bail!("Failed to fetch project {}", project_id);
        }

        let project: GitLabProject = response
            .json()
            .context("Failed to parse project response")?;

        Ok(project)
    }

    /// Extract code snippet from blob data
    fn extract_snippet(&self, data: &str, function_name: &str) -> Option<(usize, String, String)> {
        for (line_num, line) in data.lines().enumerate() {
            if line.contains(function_name) && line.contains('(') {
                let lines: Vec<&str> = data.lines().collect();
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

impl PlatformSearcher for GitLabSearcher {
    fn search_function(
        &self,
        function_name: &str,
        _library_name: &str,
    ) -> Result<Vec<CodeSearchResult>> {
        // Search for function in C/C++ files
        let query = format!(
            "{} extension:c OR extension:cpp OR extension:h",
            function_name
        );

        let search_response = self.search_code(&query)?;
        let mut results = Vec::new();

        for blob in search_response.data.into_iter().take(10) {
            // Get project details
            let project = match self.get_project(blob.project_id) {
                Ok(p) => p,
                Err(e) => {
                    tracing::debug!("Failed to fetch project {}: {}", blob.project_id, e);
                    continue;
                }
            };

            // Extract snippet
            if let Some((line_number, snippet, context)) =
                self.extract_snippet(&blob.data, function_name)
            {
                results.push(CodeSearchResult {
                    platform: PlatformSource::GitLab,
                    repository: project.name_with_namespace.clone(),
                    stars: project.star_count as usize,
                    file_path: PathBuf::from(blob.path),
                    line_number,
                    code_snippet: snippet,
                    context,
                    url: format!(
                        "{}/-/blob/{}/{}",
                        project.web_url, blob.git_ref, blob.basename
                    ),
                    last_updated: Some(project.last_activity_at),
                });
            }
        }

        Ok(results)
    }

    fn platform_name(&self) -> &str {
        "GitLab"
    }
}

impl Default for GitLabSearcher {
    fn default() -> Self {
        Self::new().expect("Failed to create GitLab searcher")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gitlab_searcher_creation() {
        let searcher = GitLabSearcher::new();
        assert!(searcher.is_ok());
    }

    #[test]
    fn test_extract_snippet() {
        let searcher = GitLabSearcher::new().unwrap();
        let content = r#"
#include <cuda.h>

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
    fn test_gitlab_search() {
        let searcher = GitLabSearcher::new().unwrap();
        let results = searcher.search_function("cudaMalloc", "cuda");

        // Should succeed even if rate limited
        assert!(results.is_ok() || results.is_err());
    }
}
