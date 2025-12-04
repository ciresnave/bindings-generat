//! Multi-platform code search infrastructure.
//!
//! This module provides tools to search for real-world usage patterns of C/C++
//! libraries across multiple code hosting platforms (GitHub, GitLab, SourceHut, etc.).
//!
//! The search results help improve binding generation by:
//! - Discovering actual API usage patterns in the wild
//! - Validating detected RAII and error patterns
//! - Inferring parameter intent and common usage
//! - Finding edge cases and error handling approaches

pub mod github;
pub mod gitlab;
pub mod searcher;
pub mod types;
pub mod web;

pub use github::GitHubSearcher;
pub use gitlab::GitLabSearcher;
pub use searcher::UsageSearcher;
pub use types::{CodeSearchResult, ConfidenceScore, PlatformSource, UsagePattern};
pub use web::WebSearcher;
