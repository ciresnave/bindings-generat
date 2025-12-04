//! Submission of discovered libraries to the community database.
//!
//! This module provides multiple methods for users to contribute discovered libraries
//! back to the community:
//!
//! - **Git CLI** ([`git_cli`]) - Uses `git` and `gh` commands (recommended)
//! - **GitHub Token** ([`github_token`]) - Uses GitHub REST API with Personal Access Token
//! - **Manual** ([`manual`]) - Generates file with manual submission instructions
//!
//! All submissions create pull requests to the bindings-generat repository that are
//! reviewed by maintainers before being merged.

use crate::user_config::{CommunityConfig, SubmissionMethod};
use anyhow::{Context, Result};

mod git_cli;
mod github_token;
mod manual;

pub use git_cli::{GitCliInfo, check_available};
pub use github_token::submit_via_api;
pub use manual::submit_manual;

/// Submit a library discovery to the community database.
///
/// # Arguments
///
/// * `library_name` - Name of the library (e.g., "openssl")
/// * `library_toml` - Complete TOML content for the library
/// * `config` - Community contribution configuration
///
/// # Returns
///
/// A success message indicating how the submission was handled.
pub fn submit_library(
    library_name: &str,
    library_toml: &str,
    config: &CommunityConfig,
) -> Result<SubmissionResult> {
    const REPO_OWNER: &str = "ciresnave";
    const REPO_NAME: &str = "bindings-generat";

    match config.submission_method {
        SubmissionMethod::GitCli => {
            // Check if git/gh are available
            match check_available() {
                Ok(info) => {
                    println!("✓ git found: {}", info.git_version);
                    println!("✓ gh found: {}", info.gh_version);
                    println!("✓ gh authenticated as: {}", info.username);
                    println!();

                    let attribution = config.attribution.name.as_ref().and_then(|name| {
                        config
                            .attribution
                            .email
                            .as_ref()
                            .map(|email| (name.as_str(), email.as_str()))
                    });

                    let pr_url = git_cli::submit_via_cli(
                        library_name,
                        library_toml,
                        REPO_OWNER,
                        REPO_NAME,
                        attribution,
                    )?;

                    Ok(SubmissionResult::PullRequestCreated(pr_url))
                }
                Err(e) => {
                    println!("⚠ Git CLI not available: {}", e);
                    println!("Falling back to manual submission...");
                    println!();

                    let path =
                        manual::submit_manual(library_name, library_toml, REPO_OWNER, REPO_NAME)?;

                    Ok(SubmissionResult::ManualInstructions(path))
                }
            }
        }

        SubmissionMethod::GithubToken => {
            let token = config.github_token.as_ref().context(
                "GitHub token not configured. Set in ~/.config/bindings-generat/config.toml",
            )?;

            let pr_url = github_token::submit_via_api(
                library_name,
                library_toml,
                REPO_OWNER,
                REPO_NAME,
                token,
            )?;

            Ok(SubmissionResult::PullRequestCreated(pr_url))
        }

        SubmissionMethod::Manual => {
            let path = manual::submit_manual(library_name, library_toml, REPO_OWNER, REPO_NAME)?;

            Ok(SubmissionResult::ManualInstructions(path))
        }
    }
}

/// Result of a library submission attempt.
#[derive(Debug, Clone)]
pub enum SubmissionResult {
    /// A pull request was successfully created.
    PullRequestCreated(String),

    /// Manual submission file was generated with instructions.
    ManualInstructions(std::path::PathBuf),
}

impl SubmissionResult {
    /// Get a user-friendly message about the submission result.
    pub fn message(&self) -> String {
        match self {
            Self::PullRequestCreated(url) => {
                format!(
                    "✓ Pull request created: {}\nThank you for contributing to the community!",
                    url
                )
            }
            Self::ManualInstructions(path) => {
                format!(
                    "✓ Submission file saved to: {}\nPlease follow the instructions above to submit manually.",
                    path.display()
                )
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    

    #[test]
    fn test_submission_result_message() {
        let pr_result = SubmissionResult::PullRequestCreated(
            "https://github.com/ciresnave/bindings-generat/pull/123".to_string(),
        );
        assert!(pr_result.message().contains("Pull request created"));

        let manual_result =
            SubmissionResult::ManualInstructions(std::path::PathBuf::from("/tmp/testlib.toml"));
        assert!(manual_result.message().contains("Submission file saved"));
    }

    #[test]
    fn test_repo_constants() {
        // Just verify the constants are accessible
        const REPO_OWNER: &str = "ciresnave";
        const REPO_NAME: &str = "bindings-generat";
        assert_eq!(REPO_OWNER, "ciresnave");
        assert_eq!(REPO_NAME, "bindings-generat");
    }
}
