//! Configuration schema for bindings-generat.
//!
//! Defines the structure of the user's configuration file, including:
//! - Google Custom Search API settings for library discovery
//! - Community contribution preferences for sharing discovered libraries
//! - Submission method preferences (git CLI, GitHub token, or manual)

use serde::{Deserialize, Serialize};

/// Main configuration structure.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[derive(Default)]
pub struct Config {
    /// Google Custom Search API configuration for automatic library discovery.
    #[serde(default)]
    pub google_search: GoogleSearchConfig,

    /// Community contribution settings for sharing discovered libraries.
    #[serde(default)]
    pub community: CommunityConfig,
}


/// Configuration for Google Custom Search API.
///
/// Users can obtain API credentials at:
/// <https://developers.google.com/custom-search/v1/overview>
///
/// Free tier provides 100 queries per day.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GoogleSearchConfig {
    /// Google Custom Search API key.
    ///
    /// Optional: If not provided, automatic library discovery will be disabled.
    pub api_key: Option<String>,

    /// Google Custom Search Engine ID.
    ///
    /// Required when `api_key` is provided.
    pub search_engine_id: Option<String>,
}

impl GoogleSearchConfig {
    /// Check if Google Search is properly configured.
    pub fn is_configured(&self) -> bool {
        self.api_key.is_some() && self.search_engine_id.is_some()
    }
}

/// Configuration for community contributions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommunityConfig {
    /// Whether to automatically offer to submit discovered libraries to the community database.
    ///
    /// When enabled, bindings-generat will prompt to create a PR when it discovers
    /// a new library through Google Search. All submissions are reviewed by maintainers
    /// before being added to the database.
    #[serde(default)]
    pub contribute_discoveries: bool,

    /// Preferred method for submitting library discoveries.
    #[serde(default = "default_submission_method")]
    pub submission_method: SubmissionMethod,

    /// GitHub Personal Access Token for API-based submissions.
    ///
    /// Required when `submission_method` is `GithubToken`.
    /// Required scopes: `public_repo`
    pub github_token: Option<String>,

    /// Optional attribution information for contributions.
    #[serde(default)]
    pub attribution: Attribution,
}

impl Default for CommunityConfig {
    fn default() -> Self {
        Self {
            contribute_discoveries: false,
            submission_method: default_submission_method(),
            github_token: None,
            attribution: Attribution::default(),
        }
    }
}

/// Method for submitting library discoveries to the community database.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum SubmissionMethod {
    /// Use git and gh CLI tools (requires user to have these installed and authenticated).
    ///
    /// This is the recommended default as it uses the user's existing GitHub authentication
    /// without requiring them to create and store a Personal Access Token.
    GitCli,

    /// Use GitHub REST API with a Personal Access Token.
    ///
    /// Requires a GitHub PAT with `public_repo` scope to be configured in `github_token`.
    GithubToken,

    /// Generate the library TOML file locally and provide instructions for manual submission.
    ///
    /// This option requires no authentication but involves manual steps.
    Manual,
}

fn default_submission_method() -> SubmissionMethod {
    SubmissionMethod::GitCli
}

impl SubmissionMethod {
    /// Get a human-readable description of this submission method.
    pub fn description(&self) -> &'static str {
        match self {
            Self::GitCli => "Use git and gh CLI tools (recommended)",
            Self::GithubToken => "Use GitHub API with Personal Access Token",
            Self::Manual => "Generate file locally with manual submission instructions",
        }
    }
}

/// Attribution information for community contributions.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Attribution {
    /// Contributor's name for git commits.
    pub name: Option<String>,

    /// Contributor's email for git commits.
    pub email: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = Config::default();
        assert!(!config.google_search.is_configured());
        assert!(!config.community.contribute_discoveries);
        assert_eq!(config.community.submission_method, SubmissionMethod::GitCli);
    }

    #[test]
    fn test_google_search_configured() {
        let mut config = GoogleSearchConfig::default();
        assert!(!config.is_configured());

        config.api_key = Some("test_key".to_string());
        assert!(!config.is_configured()); // Still need search_engine_id

        config.search_engine_id = Some("test_id".to_string());
        assert!(config.is_configured());
    }

    #[test]
    fn test_submission_method_serialization() {
        // Test within a struct context (TOML doesn't support top-level enums)
        #[derive(Serialize, Deserialize)]
        struct TestConfig {
            method: SubmissionMethod,
        }

        let methods = vec![
            SubmissionMethod::GitCli,
            SubmissionMethod::GithubToken,
            SubmissionMethod::Manual,
        ];

        for method in methods {
            let test = TestConfig { method };
            let serialized = toml::to_string(&test).unwrap();
            let deserialized: TestConfig = toml::from_str(&serialized).unwrap();
            assert_eq!(test.method, deserialized.method);
        }
    }

    #[test]
    fn test_config_serialization() {
        let config = Config {
            google_search: GoogleSearchConfig {
                api_key: Some("test_api_key".to_string()),
                search_engine_id: Some("test_search_id".to_string()),
            },
            community: CommunityConfig {
                contribute_discoveries: true,
                submission_method: SubmissionMethod::GithubToken,
                github_token: Some("test_token".to_string()),
                attribution: Attribution {
                    name: Some("Test User".to_string()),
                    email: Some("test@example.com".to_string()),
                },
            },
        };

        let serialized = toml::to_string_pretty(&config).unwrap();
        let deserialized: Config = toml::from_str(&serialized).unwrap();

        assert_eq!(
            config.google_search.api_key,
            deserialized.google_search.api_key
        );
        assert_eq!(
            config.community.contribute_discoveries,
            deserialized.community.contribute_discoveries
        );
    }
}
