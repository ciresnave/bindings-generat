//! Data structures for enrichment results.

use std::collections::HashSet;
use std::path::PathBuf;

/// All discovered files for a library
#[derive(Debug, Default, Clone)]
pub struct LibraryFiles {
    /// Documentation files found
    pub documentation: Vec<DocumentFile>,

    /// Example code files found
    pub examples: Vec<ExampleFile>,

    /// Test files found
    pub tests: Vec<TestFile>,

    /// Paths already visited (for symlink loop protection)
    pub visited_paths: HashSet<PathBuf>,

    /// Total files scanned (safety limit to prevent runaway scanning)
    pub files_scanned: usize,
}

/// A documentation file
#[derive(Debug, Clone)]
pub struct DocumentFile {
    /// Path to the file
    pub path: PathBuf,

    /// Format of the documentation
    pub format: DocFormat,

    /// Category/purpose of the documentation
    pub category: DocCategory,
}

/// Documentation format
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DocFormat {
    Markdown,
    ReStructuredText,
    Html,
    Pdf,
    ManPage,
    PlainText,
    Unknown,
}

/// Documentation category
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DocCategory {
    /// Main README
    Readme,

    /// API reference documentation
    ApiReference,

    /// Getting started guide
    GettingStarted,

    /// Tutorial
    Tutorial,

    /// Architecture/design docs
    Architecture,

    /// Release notes/changelog
    ReleaseNotes,

    /// Unknown/other
    Other,
}

/// An example code file
#[derive(Debug, Clone)]
pub struct ExampleFile {
    /// Path to the file
    pub path: PathBuf,

    /// Programming language
    pub language: Language,

    /// Complexity level (heuristic)
    pub complexity: Complexity,
}

/// Programming language
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Language {
    C,
    Cpp,
    Python,
    Cuda,
    Other,
}

/// Example complexity level
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Complexity {
    Simple,
    Intermediate,
    Advanced,
}

/// A test file
#[derive(Debug, Clone)]
pub struct TestFile {
    /// Path to the file
    pub path: PathBuf,

    /// Programming language
    pub language: Language,
}

impl DocFormat {
    /// Detect format from file extension
    pub fn from_extension(ext: &str) -> Self {
        match ext.to_lowercase().as_str() {
            "md" | "markdown" => Self::Markdown,
            "rst" | "rest" => Self::ReStructuredText,
            "html" | "htm" => Self::Html,
            "pdf" => Self::Pdf,
            "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9" => Self::ManPage,
            "txt" | "text" => Self::PlainText,
            _ => Self::Unknown,
        }
    }
}

impl DocCategory {
    /// Classify document category from filename
    pub fn from_filename(name: &str) -> Self {
        let lower = name.to_lowercase();

        if lower.contains("readme") {
            Self::Readme
        } else if lower.contains("api") || lower.contains("reference") {
            Self::ApiReference
        } else if lower.contains("getting") && lower.contains("start") {
            Self::GettingStarted
        } else if lower.contains("tutorial") || lower.contains("guide") {
            Self::Tutorial
        } else if lower.contains("architecture") || lower.contains("design") {
            Self::Architecture
        } else if lower.contains("changelog") || lower.contains("release") || lower.contains("news")
        {
            Self::ReleaseNotes
        } else {
            Self::Other
        }
    }
}

impl Language {
    /// Detect language from file extension
    pub fn from_extension(ext: &str) -> Self {
        match ext.to_lowercase().as_str() {
            "c" | "h" => Self::C,
            "cpp" | "cc" | "cxx" | "hpp" | "hxx" | "h++" => Self::Cpp,
            "cu" | "cuh" => Self::Cuda,
            "py" => Self::Python,
            _ => Self::Other,
        }
    }
}

impl Complexity {
    /// Estimate complexity from filename
    pub fn from_filename(name: &str) -> Self {
        let lower = name.to_lowercase();

        if lower.contains("simple")
            || lower.contains("basic")
            || lower.contains("hello")
            || lower.contains("minimal")
        {
            Self::Simple
        } else if lower.contains("advanced") || lower.contains("complex") {
            Self::Advanced
        } else {
            Self::Intermediate
        }
    }
}
