use bindings_generat::enrichment;
use std::path::Path;

#[test]
fn test_cuda_enrichment() {
    let cuda_header = Path::new(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\include\cuda.h");
    
    if !cuda_header.exists() {
        println!("CUDA not found, skipping test");
        return;
    }
    
    let library_files = enrichment::discover_library_files(cuda_header);
    
    println!("\n=== CUDA Enrichment Results ===");
    println!("Documentation files: {}", library_files.documentation.len());
    println!("Example files: {}", library_files.examples.len());
    println!("Test files: {}", library_files.tests.len());
    
    // Print first few docs
    if !library_files.documentation.is_empty() {
        println!("\nFirst documentation files:");
        for doc in library_files.documentation.iter().take(5) {
            println!("  {:?} ({:?}) - {}", doc.format, doc.category, doc.path.display());
        }
    }
    
    // Print first few examples
    if !library_files.examples.is_empty() {
        println!("\nFirst example files:");
        for example in library_files.examples.iter().take(5) {
            println!("  {:?} ({:?}) - {}", example.language, example.complexity, example.path.display());
        }
    }
    
    // Print first few tests
    if !library_files.tests.is_empty() {
        println!("\nFirst test files:");
        for test in library_files.tests.iter().take(5) {
            println!("  {:?} - {}", test.language, test.path.display());
        }
    }
    
    // At minimum, CUDA should have some documentation
    assert!(
        !library_files.documentation.is_empty() || !library_files.examples.is_empty(),
        "Expected to find at least some documentation or examples for CUDA"
    );
}

#[test]
fn test_library_root_detection() {
    let cuda_header = Path::new(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\include\cuda.h");
    
    if !cuda_header.exists() {
        println!("CUDA not found, skipping test");
        return;
    }
    
    let root_path = enrichment::find_library_root(cuda_header);
    
    println!("\n=== Library Root Detection ===");
    println!("Detected root: {}", root_path.display());
    
    // Should find v13.0 or CUDA directory
    assert!(
        root_path.to_string_lossy().contains("CUDA") || 
        root_path.to_string_lossy().contains("v13.0"),
        "Root should contain CUDA or version directory"
    );
}
