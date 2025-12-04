//! Crates.io API integration for discovering existing Rust wrapper crates.
//!
//! Searches crates.io to find existing Rust wrappers for C/C++ libraries,
//! helping users avoid duplicate work by pointing them to established crates.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

const CRATES_IO_API_BASE: &str = "https://crates.io/api/v1";

/// Information about a Rust crate from crates.io.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RustCrateInfo {
    /// Crate name (e.g., "openssl-sys")
    pub name: String,

    /// Latest version published
    pub latest_version: String,

    /// Total downloads across all versions
    pub downloads: u64,

    /// Crate description
    pub description: Option<String>,

    /// Homepage URL
    pub homepage: Option<String>,

    /// Repository URL (usually GitHub)
    pub repository: Option<String>,

    /// Documentation URL (usually docs.rs)
    pub documentation: Option<String>,

    /// Crates.io URL
    pub crates_io_url: String,

    /// When the crate was last updated
    pub updated_at: String,

    /// License
    pub license: Option<String>,

    /// Keywords
    pub keywords: Vec<String>,
}

/// Response from crates.io API search
#[derive(Debug, Deserialize)]
struct CratesSearchResponse {
    crates: Vec<CrateSearchResult>,
    #[allow(dead_code)]
    meta: SearchMeta,
}

#[derive(Debug, Deserialize)]
struct CrateSearchResult {
    name: String,
    newest_version: String,
    downloads: u64,
    description: Option<String>,
    homepage: Option<String>,
    repository: Option<String>,
    documentation: Option<String>,
    updated_at: String,
    #[serde(default)]
    keywords: Vec<String>,
    #[serde(default)]
    #[allow(dead_code)]
    exact_match: bool,
}

#[derive(Debug, Deserialize)]
struct SearchMeta {
    #[allow(dead_code)]
    total: u64,
}

/// Search crates.io for existing Rust wrapper crates for a C/C++ library.
///
/// Searches for common naming patterns:
/// - `{library}-sys` - Low-level FFI bindings (most common)
/// - `{library}-rs` - Safe Rust wrapper
/// - `{library}` - Idiomatic wrapper
///
/// # Arguments
///
/// * `library_name` - Name of the C/C++ library (e.g., "openssl", "libpng")
///
/// # Returns
///
/// Vector of matching crates, sorted by relevance (exact matches first, then by downloads).
pub fn search_crates_io(library_name: &str) -> Result<Vec<RustCrateInfo>> {
    // Try multiple search patterns
    let search_patterns = vec![
        format!("{}-sys", library_name),
        format!("{}-rs", library_name),
        library_name.to_string(),
        // Also try without common prefixes
        library_name.trim_start_matches("lib").to_string(),
    ];

    let mut all_results = Vec::new();

    for pattern in search_patterns {
        match search_single_pattern(&pattern) {
            Ok(results) => {
                all_results.extend(results);
            }
            Err(e) => {
                tracing::debug!("Search for '{}' failed: {}", pattern, e);
                // Continue with other patterns
            }
        }
    }

    // Deduplicate by crate name
    let mut seen = std::collections::HashSet::new();
    all_results.retain(|crate_info| seen.insert(crate_info.name.clone()));

    // Sort by relevance: exact matches first, then by download count
    all_results.sort_by(|a, b| {
        // Prioritize -sys crates
        let a_is_sys = a.name.ends_with("-sys");
        let b_is_sys = b.name.ends_with("-sys");

        match (a_is_sys, b_is_sys) {
            (true, false) => std::cmp::Ordering::Less,
            (false, true) => std::cmp::Ordering::Greater,
            _ => b.downloads.cmp(&a.downloads), // Higher downloads first
        }
    });

    // Limit to top 10 results
    all_results.truncate(10);

    Ok(all_results)
}

fn search_single_pattern(query: &str) -> Result<Vec<RustCrateInfo>> {
    let client = reqwest::blocking::Client::new();
    let url = format!("{}/crates", CRATES_IO_API_BASE);

    let response = client
        .get(&url)
        .query(&[("q", query), ("per_page", "10")])
        .header(
            "User-Agent",
            "bindings-generat (github.com/ciresnave/bindings-generat)",
        )
        .send()
        .context("Failed to query crates.io API")?;

    if !response.status().is_success() {
        anyhow::bail!("Crates.io API returned error: {}", response.status());
    }

    let search_response: CratesSearchResponse = response
        .json()
        .context("Failed to parse crates.io response")?;

    let results = search_response
        .crates
        .into_iter()
        .map(|crate_result| {
            // Extract version from newest_version field
            let version = crate_result.newest_version;

            RustCrateInfo {
                name: crate_result.name.clone(),
                latest_version: version,
                downloads: crate_result.downloads,
                description: crate_result.description,
                homepage: crate_result.homepage,
                repository: crate_result.repository,
                documentation: crate_result
                    .documentation
                    .or_else(|| Some(format!("https://docs.rs/{}", crate_result.name))),
                crates_io_url: format!("https://crates.io/crates/{}", crate_result.name),
                updated_at: crate_result.updated_at,
                license: None, // Would need separate API call to get license
                keywords: crate_result.keywords,
            }
        })
        .collect();

    Ok(results)
}

/// Format download count in human-readable form.
pub fn format_downloads(downloads: u64) -> String {
    if downloads >= 1_000_000 {
        format!("{:.1}M", downloads as f64 / 1_000_000.0)
    } else if downloads >= 1_000 {
        format!("{:.1}K", downloads as f64 / 1_000.0)
    } else {
        downloads.to_string()
    }
}

/// Determine if a crate is likely an FFI binding based on name and keywords.
pub fn is_likely_ffi_crate(crate_info: &RustCrateInfo) -> bool {
    // Check name patterns
    if crate_info.name.ends_with("-sys")
        || crate_info.name.ends_with("-ffi")
        || crate_info.name.contains("bindings")
    {
        return true;
    }

    // Check keywords
    let ffi_keywords = ["ffi", "bindings", "sys", "c", "cpp"];
    crate_info
        .keywords
        .iter()
        .any(|kw| ffi_keywords.contains(&kw.as_str()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_downloads() {
        assert_eq!(format_downloads(500), "500");
        assert_eq!(format_downloads(1_500), "1.5K");
        assert_eq!(format_downloads(1_500_000), "1.5M");
        assert_eq!(format_downloads(5_234_123), "5.2M");
    }

    #[test]
    fn test_is_likely_ffi_crate() {
        let sys_crate = RustCrateInfo {
            name: "openssl-sys".to_string(),
            latest_version: "0.10.0".to_string(),
            downloads: 1000,
            description: None,
            homepage: None,
            repository: None,
            documentation: None,
            crates_io_url: "https://crates.io/crates/openssl-sys".to_string(),
            updated_at: "2025-01-01".to_string(),
            license: None,
            keywords: vec![],
        };
        assert!(is_likely_ffi_crate(&sys_crate));

        let ffi_crate = RustCrateInfo {
            name: "libz-ffi".to_string(),
            latest_version: "1.0.0".to_string(),
            downloads: 100,
            description: None,
            homepage: None,
            repository: None,
            documentation: None,
            crates_io_url: "https://crates.io/crates/libz-ffi".to_string(),
            updated_at: "2025-01-01".to_string(),
            license: None,
            keywords: vec!["ffi".to_string()],
        };
        assert!(is_likely_ffi_crate(&ffi_crate));

        let normal_crate = RustCrateInfo {
            name: "serde".to_string(),
            latest_version: "1.0.0".to_string(),
            downloads: 10000000,
            description: None,
            homepage: None,
            repository: None,
            documentation: None,
            crates_io_url: "https://crates.io/crates/serde".to_string(),
            updated_at: "2025-01-01".to_string(),
            license: None,
            keywords: vec!["serialization".to_string()],
        };
        assert!(!is_likely_ffi_crate(&normal_crate));
    }

    #[test]
    #[ignore] // Integration test - requires network
    fn test_search_crates_io_integration() {
        let results = search_crates_io("openssl").unwrap();
        assert!(!results.is_empty());

        // Should find openssl-sys
        let has_sys_crate = results.iter().any(|c| c.name == "openssl-sys");
        assert!(has_sys_crate, "Should find openssl-sys crate");
    }
}
