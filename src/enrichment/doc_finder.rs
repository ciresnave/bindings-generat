//! Smart directory discovery for documentation, examples, and tests.
//!
//! This module implements intelligent directory traversal to find all relevant
//! files for a library, starting from a known header location.

use super::types::{
    Complexity, DocCategory, DocFormat, DocumentFile, ExampleFile, Language, LibraryFiles, TestFile,
};
use std::fs;
use std::path::{Path, PathBuf};
use tracing::debug;

/// Search patterns for different file categories
struct SearchPatterns {
    docs: Vec<&'static str>,
    examples: Vec<&'static str>,
    tests: Vec<&'static str>,
}

impl Default for SearchPatterns {
    fn default() -> Self {
        Self {
            docs: vec!["doc", "docs", "documentation", "man", "reference", "share"],
            examples: vec!["example", "examples", "sample", "samples", "demo", "demos"],
            tests: vec!["test", "tests", "testing", "spec"],
        }
    }
}

/// Check if a directory contains library-related documentation files
///
/// Looks for common documentation files and checks if their content
/// mentions library-related topics (API, functions, headers, etc.)
fn has_library_documentation(path: &Path) -> bool {
    // Common library documentation files
    let doc_files = [
        "readme.md",
        "readme.txt",
        "readme",
        "changelog.md",
        "changelog",
        "license",
        "license.md",
        "license.txt",
        "authors",
        "contributors",
        "install.md",
        "install.txt",
        "building.md",
        "api.md",
    ];

    for doc_file in &doc_files {
        let file_path = path.join(doc_file);
        if file_path.exists() && file_path.is_file() {
            // Check if the file contains library-related keywords
            if let Ok(content) = fs::read_to_string(&file_path) {
                // Only read first 2KB to avoid large files
                let preview = content
                    .chars()
                    .take(2048)
                    .collect::<String>()
                    .to_lowercase();

                // Library-related keywords
                let has_api_mentions = preview.contains("api") || preview.contains("function");
                let has_header_mentions = preview.contains("header") || preview.contains(".h\"");
                let has_library_mentions =
                    preview.contains("library") || preview.contains("linking");
                let has_include_mentions =
                    preview.contains("include") || preview.contains("#include");
                let has_build_mentions = preview.contains("build") || preview.contains("compile");

                if has_api_mentions
                    || has_header_mentions
                    || has_library_mentions
                    || has_include_mentions
                    || has_build_mentions
                {
                    debug!("Found library documentation in: {}", file_path.display());
                    return true;
                }
            }
        }
    }

    false
}

/// Find the likely root directory of a library given a header path
///
/// Walks up from the header location looking for indicators that suggest
/// we've reached the library root (presence of include/, doc/, examples/ siblings).
///
/// **Smart Strategy:**
/// - If we find library indicators going up, use that as root
/// - If we go up and find NO library-related content, return to the original path
/// - This prevents scanning parent directories that aren't part of the library
pub fn find_library_root(header_path: &Path) -> PathBuf {
    debug!("Finding library root from: {}", header_path.display());

    let start_path = if header_path.is_dir() {
        header_path.to_path_buf()
    } else {
        header_path.parent().unwrap_or(header_path).to_path_buf()
    };

    let mut current = start_path.clone();
    let mut found_any_library_content = false;

    // Try up to 3 levels above the starting point
    for level in 0..3 {
        if let Ok(entries) = fs::read_dir(&current) {
            let siblings: Vec<String> = entries
                .filter_map(|e| e.ok())
                .map(|e| e.file_name().to_string_lossy().to_lowercase())
                .collect();

            let has_include = siblings.iter().any(|s| s.contains("include"));
            let has_docs = siblings
                .iter()
                .any(|s| s == "doc" || s == "docs" || s == "documentation");
            let has_examples = siblings
                .iter()
                .any(|s| s.contains("example") || s.contains("sample"));
            let has_share = siblings.iter().any(|s| s == "share");
            let has_src = siblings.iter().any(|s| s == "src" || s == "source");
            let has_lib = siblings.iter().any(|s| s == "lib" || s.contains("library"));

            // Also check for library documentation files with relevant content
            let has_doc_files = has_library_documentation(&current);

            // Check if this directory has ANY library-related content
            let has_library_content = has_include
                || has_docs
                || has_examples
                || has_share
                || has_src
                || has_lib
                || has_doc_files;

            if has_library_content {
                found_any_library_content = true;

                // Strong indicators we found the root
                if (has_include && has_docs)
                    || (has_include && has_examples)
                    || (has_include && has_share)
                    || (has_include && has_doc_files)
                {
                    debug!(
                        "Found library root at level {}: {}",
                        level,
                        current.display()
                    );
                    return current;
                }
            } else if level > 0 && !found_any_library_content {
                // We went up a level and found NOTHING library-related
                // This means we're in a parent directory that's not part of the library
                // Return to the original starting path
                debug!(
                    "No library content found at level {}, returning to starting path: {}",
                    level,
                    start_path.display()
                );
                return start_path;
            }
        }

        // Move up one level
        if let Some(parent) = current.parent() {
            current = parent.to_path_buf();
        } else {
            break;
        }
    }

    // If we found library content but no perfect match, use current
    // Otherwise return to start_path
    let fallback = if found_any_library_content {
        current
    } else {
        start_path
    };

    debug!("Using fallback root: {}", fallback.display());
    fallback
}

/// Discover all documentation, examples, and tests for a library
///
/// Starting from a header path, finds the library root and recursively
/// searches for relevant files.
pub fn discover_library_files(header_path: &Path) -> LibraryFiles {
    debug!("Discovering library files from: {}", header_path.display());

    let root = find_library_root(header_path);
    debug!("Library root determined as: {}", root.display());

    let mut files = LibraryFiles::default();
    let patterns = SearchPatterns::default();

    // Reduced max_depth from 8 to 4 to prevent excessive scanning
    // Most library documentation is within 3-4 levels of the root
    walk_directory(&root, &patterns, &mut files, 0, 4, &root);

    debug!(
        "Discovery complete: {} docs, {} examples, {} tests",
        files.documentation.len(),
        files.examples.len(),
        files.tests.len()
    );

    files
}

/// Recursively walk a directory tree looking for relevant files
fn walk_directory(
    path: &Path,
    patterns: &SearchPatterns,
    files: &mut LibraryFiles,
    depth: usize,
    max_depth: usize,
    root: &Path, // Add root parameter to enforce boundaries
) {
    // Safety limit: Stop if we've scanned too many files (prevents runaway scanning)
    const MAX_FILES_TO_SCAN: usize = 10000;
    if files.files_scanned > MAX_FILES_TO_SCAN {
        debug!(
            "Reached safety limit of {} files scanned, stopping discovery",
            MAX_FILES_TO_SCAN
        );
        return;
    }

    if depth > max_depth {
        return;
    }

    // Safety check: Never walk outside the library root
    // This prevents scanning parent directories like C:\Users
    if let (Ok(canonical_path), Ok(canonical_root)) = (path.canonicalize(), root.canonicalize())
        && !canonical_path.starts_with(&canonical_root) {
            debug!(
                "Skipping path outside library root: {} (root: {})",
                path.display(),
                root.display()
            );
            return;
        }

    // Symlink loop protection
    if let Ok(canonical) = path.canonicalize() {
        if files.visited_paths.contains(&canonical) {
            debug!("Skipping already visited path: {}", path.display());
            return;
        }
        files.visited_paths.insert(canonical);
    }

    let entries = match fs::read_dir(path) {
        Ok(e) => e,
        Err(err) => {
            debug!("Cannot read directory {}: {}", path.display(), err);
            return;
        }
    };

    for entry in entries.filter_map(|e| e.ok()) {
        let entry_path = entry.path();
        let name = entry.file_name().to_string_lossy().to_lowercase();

        files.files_scanned += 1;

        if entry_path.is_dir() {
            // Prune obviously irrelevant directories
            if should_prune_directory(&name) {
                continue;
            }

            // Check if this directory matches our patterns
            let is_relevant = is_relevant_directory(&name, patterns);

            // Don't increment depth for relevant directories (search deeper within them)
            let new_depth = if is_relevant { depth } else { depth + 1 };

            walk_directory(&entry_path, patterns, files, new_depth, max_depth, root);
        } else if entry_path.is_file() {
            classify_and_add_file(&entry_path, files);
        }
    }
}

/// Check if a directory should be pruned (not searched)
fn should_prune_directory(name: &str) -> bool {
    // Hidden directories
    if name.starts_with('.') {
        return true;
    }

    // Build artifacts
    if matches!(name, "build" | "cmake" | "obj" | "target" | "out" | "bin") {
        return true;
    }

    // Version control
    if matches!(name, ".git" | ".svn" | ".hg" | "cvs") {
        return true;
    }

    // Dependencies
    if matches!(name, "node_modules" | "vendor" | "third_party" | "external") {
        return true;
    }

    // User directories that should never be searched
    if matches!(
        name,
        "desktop"
            | "downloads"
            | "documents"
            | "pictures"
            | "videos"
            | "music"
            | "appdata"
            | "application data"
            | "program files"
            | "program files (x86)"
            | "windows"
            | "system32"
            | "users"
            | "home"
            | "onedrive"
    ) {
        return true;
    }

    // Backup/temp
    if name.ends_with('~') || name.starts_with("tmp") {
        return true;
    }

    false
}

/// Check if a directory is relevant for our search
fn is_relevant_directory(name: &str, patterns: &SearchPatterns) -> bool {
    patterns.docs.iter().any(|p| name.contains(p))
        || patterns.examples.iter().any(|p| name.contains(p))
        || patterns.tests.iter().any(|p| name.contains(p))
}

/// Classify a file and add it to the appropriate category
fn classify_and_add_file(path: &Path, files: &mut LibraryFiles) {
    let name = path
        .file_name()
        .map(|n| n.to_string_lossy().to_lowercase())
        .unwrap_or_default();

    let ext = path
        .extension()
        .map(|e| e.to_string_lossy().to_lowercase())
        .unwrap_or_default();

    // Documentation files
    if is_documentation_file(&name, &ext) {
        let doc = DocumentFile {
            path: path.to_path_buf(),
            format: DocFormat::from_extension(&ext),
            category: DocCategory::from_filename(&name),
        };
        debug!("Found doc: {} ({:?})", path.display(), doc.category);
        files.documentation.push(doc);
        return;
    }

    // Example files (C, C++, Python, CUDA)
    if is_example_file(&name, &ext) {
        let example = ExampleFile {
            path: path.to_path_buf(),
            language: Language::from_extension(&ext),
            complexity: Complexity::from_filename(&name),
        };
        debug!("Found example: {} ({:?})", path.display(), example.language);
        files.examples.push(example);
        return;
    }

    // Test files
    if is_test_file(&name, &ext) {
        let test = TestFile {
            path: path.to_path_buf(),
            language: Language::from_extension(&ext),
        };
        debug!("Found test: {}", path.display());
        files.tests.push(test);
    }
}

/// Check if a file is a documentation file
fn is_documentation_file(name: &str, ext: &str) -> bool {
    // Documentation extensions
    let doc_extensions = [
        "md", "markdown", "rst", "rest", "html", "htm", "pdf", "txt", "text",
    ];
    if !doc_extensions.contains(&ext) && !ext.chars().all(|c| c.is_ascii_digit()) {
        return false;
    }

    // Documentation name patterns
    name.contains("readme")
        || name.contains("api")
        || name.contains("reference")
        || name.contains("doc")
        || name.contains("guide")
        || name.contains("tutorial")
        || name.contains("changelog")
        || name.contains("release")
        || name.contains("architecture")
        || name.contains("design")
}

/// Check if a file is an example file
fn is_example_file(name: &str, ext: &str) -> bool {
    // Source code extensions only
    let code_extensions = ["c", "cpp", "cc", "cxx", "cu", "py", "h", "hpp"];
    if !code_extensions.contains(&ext) {
        return false;
    }

    // Example name patterns
    name.contains("example") || name.contains("sample") || name.contains("demo")
}

/// Check if a file is a test file
fn is_test_file(name: &str, ext: &str) -> bool {
    // Source code extensions only
    let code_extensions = ["c", "cpp", "cc", "cxx"];
    if !code_extensions.contains(&ext) {
        return false;
    }

    // Test name patterns
    name.contains("test") || name.contains("spec")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_should_prune_directory() {
        assert!(should_prune_directory(".git"));
        assert!(should_prune_directory(".hidden"));
        assert!(should_prune_directory("build"));
        assert!(should_prune_directory("node_modules"));
        assert!(!should_prune_directory("docs"));
        assert!(!should_prune_directory("examples"));
    }

    #[test]
    fn test_is_documentation_file() {
        assert!(is_documentation_file("readme.md", "md"));
        assert!(is_documentation_file("api_reference.rst", "rst"));
        assert!(is_documentation_file("guide.html", "html"));
        assert!(!is_documentation_file("main.c", "c"));
        assert!(!is_documentation_file("test.cpp", "cpp"));
    }

    #[test]
    fn test_is_example_file() {
        assert!(is_example_file("example_basic.c", "c"));
        assert!(is_example_file("sample.cpp", "cpp"));
        assert!(is_example_file("demo.py", "py"));
        assert!(!is_example_file("main.c", "c"));
        assert!(!is_example_file("readme.md", "md"));
    }

    #[test]
    fn test_is_test_file() {
        assert!(is_test_file("test_functions.c", "c"));
        assert!(is_test_file("spec_utils.cpp", "cpp"));
        assert!(!is_test_file("example.c", "c"));
        assert!(!is_test_file("test.py", "py")); // Python tests not included
    }
}
