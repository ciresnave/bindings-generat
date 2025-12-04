//! Google Custom Search integration for library discovery.
//!
//! Uses Google's Custom Search JSON API to search for unknown libraries and
//! extract metadata from search results.
//!
//! API documentation: <https://developers.google.com/custom-search/v1/overview>

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

const GOOGLE_SEARCH_API_BASE: &str = "https://www.googleapis.com/customsearch/v1";

/// A search result from Google Custom Search.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    /// Title of the result
    pub title: String,

    /// URL of the result
    pub link: String,

    /// Snippet/description of the result
    pub snippet: String,

    /// Display link (hostname)
    pub display_link: String,
}

/// Response from Google Custom Search API.
#[derive(Debug, Deserialize)]
struct GoogleSearchResponse {
    items: Option<Vec<GoogleSearchItem>>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GoogleSearchItem {
    title: String,
    link: String,
    snippet: String,
    display_link: String,
}

/// Search for a library using Google Custom Search.
///
/// # Arguments
///
/// * `library_name` - Name of the library to search for
/// * `api_key` - Google Custom Search API key
/// * `search_engine_id` - Custom Search Engine ID
///
/// # Returns
///
/// A vector of search results, limited to the top 10 results.
pub fn search_library(
    library_name: &str,
    api_key: &str,
    search_engine_id: &str,
) -> Result<Vec<SearchResult>> {
    // Construct search query
    let query = format!("{} library documentation", library_name);

    let client = reqwest::blocking::Client::new();
    let response = client
        .get(GOOGLE_SEARCH_API_BASE)
        .query(&[
            ("key", api_key),
            ("cx", search_engine_id),
            ("q", &query),
            ("num", "10"), // Get top 10 results
        ])
        .send()
        .context("Failed to send Google Search request")?;

    if !response.status().is_success() {
        let status = response.status();
        let text = response.text().unwrap_or_default();
        anyhow::bail!("Google Search API error: {} - {}", status, text);
    }

    let search_response: GoogleSearchResponse = response
        .json()
        .context("Failed to parse Google Search response")?;

    let results = search_response
        .items
        .unwrap_or_default()
        .into_iter()
        .map(|item| SearchResult {
            title: item.title,
            link: item.link,
            snippet: item.snippet,
            display_link: item.display_link,
        })
        .collect();

    Ok(results)
}

/// Perform an enhanced search for library information, including documentation, examples, and tutorials.
///
/// This performs multiple targeted searches to find comprehensive information about a library.
///
/// # Arguments
///
/// * `library_name` - Name of the library to search for
/// * `api_key` - Google Custom Search API key
/// * `search_engine_id` - Custom Search Engine ID
///
/// # Returns
///
/// LibraryInfo with categorized URLs for documentation, examples, and tutorials.
pub fn search_library_enhanced(
    library_name: &str,
    api_key: &str,
    search_engine_id: &str,
) -> Result<LibraryInfo> {
    // Perform multiple targeted searches
    let base_results = search_library(library_name, api_key, search_engine_id)?;
    let doc_results = search_with_query(
        &format!("{} documentation", library_name),
        api_key,
        search_engine_id,
    )?;
    let example_results = search_with_query(
        &format!("{} examples code", library_name),
        api_key,
        search_engine_id,
    )?;
    let tutorial_results = search_with_query(
        &format!("{} tutorial getting started", library_name),
        api_key,
        search_engine_id,
    )?;

    // Analyze base results for primary info
    let mut info = analyze_results(&base_results)
        .ok_or_else(|| anyhow::anyhow!("No search results found for {}", library_name))?;

    // Categorize documentation URLs
    info.documentation_urls = doc_results
        .iter()
        .filter(|r| is_documentation_url(&r.link))
        .map(|r| r.link.clone())
        .take(5)
        .collect();

    // Categorize example URLs
    info.example_urls = example_results
        .iter()
        .filter(|r| is_example_url(&r.link) || r.snippet.to_lowercase().contains("example"))
        .map(|r| r.link.clone())
        .take(5)
        .collect();

    // Categorize tutorial URLs
    info.tutorial_urls = tutorial_results
        .iter()
        .filter(|r| is_tutorial_url(&r.link) || r.snippet.to_lowercase().contains("tutorial"))
        .map(|r| r.link.clone())
        .take(5)
        .collect();

    Ok(info)
}

/// Perform a custom search query.
fn search_with_query(
    query: &str,
    api_key: &str,
    search_engine_id: &str,
) -> Result<Vec<SearchResult>> {
    let client = reqwest::blocking::Client::new();
    let response = client
        .get(GOOGLE_SEARCH_API_BASE)
        .query(&[
            ("key", api_key),
            ("cx", search_engine_id),
            ("q", query),
            ("num", "10"),
        ])
        .send()
        .context("Failed to send Google Search request")?;

    if !response.status().is_success() {
        let status = response.status();
        let text = response.text().unwrap_or_default();
        anyhow::bail!("Google Search API error: {} - {}", status, text);
    }

    let search_response: GoogleSearchResponse = response
        .json()
        .context("Failed to parse Google Search response")?;

    let results = search_response
        .items
        .unwrap_or_default()
        .into_iter()
        .map(|item| SearchResult {
            title: item.title,
            link: item.link,
            snippet: item.snippet,
            display_link: item.display_link,
        })
        .collect();

    Ok(results)
}

/// Check if a URL is likely a documentation page.
fn is_documentation_url(url: &str) -> bool {
    let lower = url.to_lowercase();
    lower.contains("docs") || lower.contains("documentation") || lower.contains("reference")
}

/// Check if a URL is likely an example page.
fn is_example_url(url: &str) -> bool {
    let lower = url.to_lowercase();
    lower.contains("example") || lower.contains("sample") || lower.contains("/examples/")
}

/// Check if a URL is likely a tutorial page.
fn is_tutorial_url(url: &str) -> bool {
    let lower = url.to_lowercase();
    lower.contains("tutorial")
        || lower.contains("getting-started")
        || lower.contains("guide")
        || lower.contains("howto")
}

/// Extract potential library information from search results.
///
/// This is a simple heuristic that looks for common patterns in search results
/// to identify official documentation, GitHub repositories, etc.
pub fn analyze_results(results: &[SearchResult]) -> Option<LibraryInfo> {
    if results.is_empty() {
        return None;
    }

    // Simple heuristic: prefer GitHub repos, then official documentation
    let github_result = results.iter().find(|r| r.link.contains("github.com"));
    let first_result = results.first()?;

    let primary_result = github_result.unwrap_or(first_result);

    Some(LibraryInfo {
        name: extract_library_name(&primary_result.title),
        homepage: primary_result.link.clone(),
        description: primary_result.snippet.clone(),
        github_repo: github_result.map(|r| r.link.clone()),
        documentation_urls: vec![],
        example_urls: vec![],
        tutorial_urls: vec![],
        rust_crates: vec![],
    })
}

/// Preliminary library information extracted from search results.
#[derive(Debug, Clone)]
pub struct LibraryInfo {
    pub name: String,
    pub homepage: String,
    pub description: String,
    pub github_repo: Option<String>,
    /// URLs to library documentation
    pub documentation_urls: Vec<String>,
    /// URLs to example code/tutorials
    pub example_urls: Vec<String>,
    /// URLs to tutorials
    pub tutorial_urls: Vec<String>,
    /// Known Rust crates that wrap this library
    pub rust_crates: Vec<super::crates_io::RustCrateInfo>,
}

fn extract_library_name(title: &str) -> String {
    // Simple extraction: take first word or phrase before common separators
    title
        .split(&['-', '|', ':', '—'][..])
        .next()
        .unwrap_or(title)
        .trim()
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_library_name() {
        assert_eq!(
            extract_library_name("OpenSSL - Cryptography Library"),
            "OpenSSL"
        );
        assert_eq!(
            extract_library_name("libpng | PNG Reference Library"),
            "libpng"
        );
        assert_eq!(extract_library_name("zlib"), "zlib");
    }

    #[test]
    fn test_analyze_results_empty() {
        let results = vec![];
        assert!(analyze_results(&results).is_none());
    }

    #[test]
    fn test_analyze_results_prefers_github() {
        let results = vec![
            SearchResult {
                title: "Library Docs".to_string(),
                link: "https://example.com/docs".to_string(),
                snippet: "Documentation".to_string(),
                display_link: "example.com".to_string(),
            },
            SearchResult {
                title: "Library GitHub".to_string(),
                link: "https://github.com/user/library".to_string(),
                snippet: "GitHub repository".to_string(),
                display_link: "github.com".to_string(),
            },
        ];

        let info = analyze_results(&results).unwrap();
        assert!(info.homepage.contains("github.com"));
        assert!(info.github_repo.is_some());
    }

    #[test]
    fn test_search_result_serialization() {
        let result = SearchResult {
            title: "Test".to_string(),
            link: "https://example.com".to_string(),
            snippet: "Test snippet".to_string(),
            display_link: "example.com".to_string(),
        };

        let json = serde_json::to_string(&result).unwrap();
        let deserialized: SearchResult = serde_json::from_str(&json).unwrap();

        assert_eq!(result.title, deserialized.title);
        assert_eq!(result.link, deserialized.link);
    }

    #[test]
    fn test_is_documentation_url() {
        assert!(is_documentation_url("https://example.com/docs/index.html"));
        assert!(is_documentation_url("https://docs.example.com"));
        assert!(is_documentation_url("https://example.com/documentation"));
        assert!(is_documentation_url("https://example.com/reference"));
        assert!(!is_documentation_url("https://example.com/blog"));
    }

    #[test]
    fn test_is_example_url() {
        assert!(is_example_url("https://example.com/examples/basic.c"));
        assert!(is_example_url("https://example.com/sample-code"));
        assert!(is_example_url(
            "https://github.com/user/repo/tree/main/examples"
        ));
        assert!(!is_example_url("https://mysite.com/documentation"));
    }

    #[test]
    fn test_is_tutorial_url() {
        assert!(is_tutorial_url("https://example.com/tutorial"));
        assert!(is_tutorial_url("https://example.com/getting-started"));
        assert!(is_tutorial_url("https://example.com/guide"));
        assert!(is_tutorial_url("https://example.com/howto"));
        assert!(!is_tutorial_url("https://example.com/about"));
    }
}
