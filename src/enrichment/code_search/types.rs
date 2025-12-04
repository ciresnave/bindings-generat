//! Type definitions for code search results.

use std::path::PathBuf;
use serde::{Deserialize, Serialize};

/// Source platform for code search results
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PlatformSource {
    GitHub,
    GitLab,
    SourceHut,
    Codeberg,
    BitBucket,
    Other(String),
}

/// Confidence score for usage patterns
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ConfidenceScore {
    /// Very low confidence (single occurrence, old code, unknown repo)
    VeryLow,
    /// Low confidence (few occurrences, questionable quality)
    Low,
    /// Medium confidence (multiple occurrences, decent repos)
    Medium,
    /// High confidence (many occurrences, quality repos, recent)
    High,
    /// Very high confidence (widespread pattern, official examples, docs)
    VeryHigh,
}

impl ConfidenceScore {
    /// Calculate confidence score from various factors
    pub fn calculate(
        occurrence_count: usize,
        repo_stars: usize,
        is_recent: bool,
        is_official: bool,
    ) -> Self {
        if is_official {
            return Self::VeryHigh;
        }

        let score = occurrence_count * 2 + repo_stars / 100 + if is_recent { 10 } else { 0 };

        match score {
            0..=5 => Self::VeryLow,
            6..=15 => Self::Low,
            16..=30 => Self::Medium,
            31..=50 => Self::High,
            _ => Self::VeryHigh,
        }
    }
}

/// A code snippet showing usage of a function or API
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeSearchResult {
    /// Platform where this code was found
    pub platform: PlatformSource,
    /// Repository name (e.g., "owner/repo")
    pub repository: String,
    /// Number of stars/popularity metric
    pub stars: usize,
    /// File path within the repository
    pub file_path: PathBuf,
    /// Line number where the usage appears
    pub line_number: usize,
    /// The code snippet showing usage
    pub code_snippet: String,
    /// Surrounding context (lines before/after)
    pub context: String,
    /// URL to view the code online
    pub url: String,
    /// Last update timestamp (if available)
    pub last_updated: Option<String>,
}

/// A detected usage pattern for an API
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsagePattern {
    /// The function or API being used
    pub function_name: String,
    /// Number of times this pattern was observed
    pub occurrence_count: usize,
    /// Confidence in this pattern
    pub confidence: ConfidenceScore,
    /// Code examples demonstrating this pattern
    pub examples: Vec<CodeSearchResult>,
    /// Common error handling approaches
    pub error_handling: Vec<String>,
    /// Common parameter patterns
    pub parameter_patterns: Vec<String>,
    /// Common setup/cleanup patterns
    pub setup_cleanup: Vec<String>,
}

impl UsagePattern {
    /// Create a new usage pattern
    pub fn new(function_name: String) -> Self {
        Self {
            function_name,
            occurrence_count: 0,
            confidence: ConfidenceScore::VeryLow,
            examples: Vec::new(),
            error_handling: Vec::new(),
            parameter_patterns: Vec::new(),
            setup_cleanup: Vec::new(),
        }
    }

    /// Add a search result to this pattern
    pub fn add_result(&mut self, result: CodeSearchResult) {
        self.examples.push(result);
        self.occurrence_count = self.examples.len();
        self.update_confidence();
    }

    /// Update confidence score based on current data
    fn update_confidence(&mut self) {
        let total_stars: usize = self.examples.iter().map(|e| e.stars).sum();
        let avg_stars = if self.examples.is_empty() {
            0
        } else {
            total_stars / self.examples.len()
        };

        let is_recent = self
            .examples
            .iter()
            .any(|e| e.last_updated.is_some());

        let is_official = self
            .examples
            .iter()
            .any(|e| e.repository.contains("official") || e.stars > 1000);

        self.confidence = ConfidenceScore::calculate(
            self.occurrence_count,
            avg_stars,
            is_recent,
            is_official,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_confidence_score_calculation() {
        // Official repos should get VeryHigh
        assert_eq!(
            ConfidenceScore::calculate(1, 0, false, true),
            ConfidenceScore::VeryHigh
        );

        // High occurrence + stars + recent = VeryHigh
        assert_eq!(
            ConfidenceScore::calculate(20, 5000, true, false),
            ConfidenceScore::VeryHigh
        );

        // Medium occurrence + stars + recent = High  
        assert_eq!(
            ConfidenceScore::calculate(10, 500, true, false),
            ConfidenceScore::High
        );

        // Single occurrence, no stars = VeryLow
        assert_eq!(
            ConfidenceScore::calculate(1, 0, false, false),
            ConfidenceScore::VeryLow
        );
    }

    #[test]
    fn test_usage_pattern_add_result() {
        let mut pattern = UsagePattern::new("cudaMalloc".to_string());
        assert_eq!(pattern.occurrence_count, 0);
        assert_eq!(pattern.confidence, ConfidenceScore::VeryLow);

        let result = CodeSearchResult {
            platform: PlatformSource::GitHub,
            repository: "nvidia/cuda-samples".to_string(),
            stars: 5000,
            file_path: PathBuf::from("samples/memory.cu"),
            line_number: 42,
            code_snippet: "cudaMalloc(&ptr, size);".to_string(),
            context: "// Allocate memory\ncudaMalloc(&ptr, size);".to_string(),
            url: "https://github.com/nvidia/cuda-samples/blob/main/samples/memory.cu#L42".to_string(),
            last_updated: Some("2025-11-20".to_string()),
        };

        pattern.add_result(result);
        assert_eq!(pattern.occurrence_count, 1);
        assert!(pattern.confidence > ConfidenceScore::VeryLow);
    }
}
