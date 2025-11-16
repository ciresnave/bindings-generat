use bindings_generat::sources;
use std::fs;
use tempfile::TempDir;

#[test]
fn test_prepare_source_directory() {
    let temp_dir = TempDir::new().unwrap();
    let source_dir = temp_dir.path().join("source");
    fs::create_dir(&source_dir).unwrap();

    // Create a dummy header file
    fs::write(source_dir.join("test.h"), "// test header").unwrap();

    let prepared = sources::prepare_source(source_dir.to_str().unwrap()).unwrap();
    assert_eq!(prepared.path(), &source_dir);
    assert!(!prepared.is_temporary);
}

#[test]
fn test_prepare_source_invalid_path() {
    let result = sources::prepare_source("/nonexistent/path/that/does/not/exist");
    assert!(result.is_err());
}

#[test]
fn test_extract_filename_from_url() {
    // Placeholder for testing URL filename extraction
    // The extract_filename_from_url function is private to archives module
}

#[test]
#[ignore] // Requires network access
fn test_download_from_url() {
    // Test downloading a small public archive
    // This would require a reliable test URL
}
