//! Demonstration of enriched LLM documentation enhancement
//!
//! This example shows how enrichment data (discovered docs, usage examples)
//! improves LLM-generated documentation quality.

use bindings_generat::enrichment::{doc_finder, doc_parser, code_search};
use bindings_generat::llm::{DocsEnhancer, EnhancedContext};
use std::path::PathBuf;
use anyhow::Result;

fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt::init();

    println!("Enhanced LLM Documentation Demo\n");
    println!("================================\n");

    // Step 1: Create enriched context
    let mut enhanced_context = EnhancedContext::new();

    // Step 2: Simulate documentation discovery
    println!("1. Discovering documentation files...");
    let library_path = PathBuf::from(".");
    let files = doc_finder::discover_library_files(&library_path);
    println!("   Found {} docs, {} examples, {} tests",
        files.documentation.len(),
        files.examples.len(),
        files.tests.len());
    enhanced_context = enhanced_context.with_library_files(files);

    // Step 3: Add parsed documentation (simulated)
    println!("\n2. Adding parsed documentation...");
    let mut func_doc = doc_parser::FunctionDoc::new("cudaMalloc".to_string());
    func_doc.brief = Some("Allocate memory on the device".to_string());
    func_doc.detailed = Some(
        "Allocates size bytes of linear memory on the device and returns \
         in *devPtr a pointer to the allocated memory. The allocated memory \
         is suitably aligned for any kind of variable."
            .to_string(),
    );
    func_doc.parameters.push(doc_parser::ParamDoc {
        name: "devPtr".to_string(),
        description: "Pointer to allocated device memory".to_string(),
        direction: doc_parser::types::ParamDirection::Out,
        optional: false,
    });
    func_doc.parameters.push(doc_parser::ParamDoc {
        name: "size".to_string(),
        description: "Requested allocation size in bytes".to_string(),
        direction: doc_parser::types::ParamDirection::In,
        optional: false,
    });
    func_doc.return_doc = Some("cudaSuccess on success, cudaErrorMemoryAllocation on failure".to_string());

    enhanced_context.add_parsed_doc("cudaMalloc".to_string(), func_doc);
    println!("   Added documentation for cudaMalloc");

    // Step 4: Add usage patterns (simulated)
    println!("\n3. Adding code search usage patterns...");
    let usage_pattern = code_search::UsagePattern {
        function_name: "cudaMalloc".to_string(),
        occurrence_count: 1234,
        confidence: code_search::ConfidenceScore::High,
        examples: vec![
            code_search::CodeSearchResult {
                platform: code_search::PlatformSource::GitHub,
                repository: "nvidia/cuda-samples".to_string(),
                stars: 5420,
                file_path: PathBuf::from("samples/vectorAdd/vectorAdd.cu"),
                line_number: 42,
                code_snippet: "cudaMalloc((void**)&d_A, size);".to_string(),
                context: r#"
float *d_A, *d_B, *d_C;
cudaMalloc((void**)&d_A, size);
cudaMalloc((void**)&d_B, size);
cudaMalloc((void**)&d_C, size);
if (cudaGetLastError() != cudaSuccess) {
    fprintf(stderr, "Failed to allocate device memory\n");
    exit(EXIT_FAILURE);
}
"#.to_string(),
                url: "https://github.com/nvidia/cuda-samples/blob/master/samples/vectorAdd/vectorAdd.cu".to_string(),
                last_updated: Some("2024-01-15".to_string()),
            },
        ],
        parameter_patterns: vec!["devPtr".to_string(), "size".to_string()],
        error_handling: vec![
            "Check cudaGetLastError() after allocation".to_string(),
            "Cast to (void**) for device pointer".to_string(),
        ],
        setup_cleanup: vec![
            "Free with cudaFree()".to_string(),
        ],
    };

    enhanced_context.add_usage_pattern("cudaMalloc".to_string(), usage_pattern);
    println!("   Added 1234 usage examples from code search");

    // Step 5: Show enrichment summary
    println!("\n4. Enrichment Summary:");
    println!("   {}", enhanced_context.summary());

    // Step 6: Create LLM enhancer
    println!("\n5. Creating DocsEnhancer...");
    match DocsEnhancer::new("qwen2.5-coder:3b".to_string(), None) {
        Ok(enhancer) => {
            if !enhancer.is_available() {
                println!("   LLM not available - would need Ollama running");
                println!("   (This is expected in CI/testing environments)");
            } else {
                println!("   LLM available!");

                // Step 7: Generate enhanced documentation
                println!("\n6. Generating enhanced documentation...");
                
                let base_context = "CUDA memory allocation function";
                let signature = "cudaError_t cudaMalloc(void **devPtr, size_t size)";
                
                match enhancer.enhance_function_docs_with_context(
                    "cudaMalloc",
                    signature,
                    base_context,
                    &enhanced_context,
                ) {
                    Ok(Some(docs)) => {
                        println!("\n   Generated Documentation:");
                        println!("   ========================");
                        println!("{}", docs);
                    }
                    Ok(None) => {
                        println!("   No documentation generated");
                    }
                    Err(e) => {
                        println!("   Error: {}", e);
                    }
                }
            }
        }
        Err(e) => {
            println!("   Could not create enhancer: {}", e);
        }
    }

    // Step 8: Show the enriched context that would be sent to LLM
    println!("\n7. Enhanced Context Example:");
    println!("   ==========================");
    let context_preview = enhanced_context.build_function_context(
        "cudaMalloc",
        "CUDA memory allocation function",
    );
    println!("{}", context_preview);

    println!("\n✓ Demo complete!");
    println!("\nKey Benefits of Enrichment:");
    println!("  • Real usage examples from 1234 repositories");
    println!("  • Official documentation with parameter descriptions");
    println!("  • Common error handling patterns from real code");
    println!("  • Direction info (In/Out) for parameters");
    println!("\nThis rich context enables LLMs to generate much better documentation!");

    Ok(())
}
