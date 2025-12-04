//! Example usage of the tool-based binding generation system

use anyhow::Result;
use bindings_generat::tools::{BindingOrchestrator, ExecutionMode, ToolContext};
use std::path::Path;

fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    // Example 1: Traditional sequential execution (like the old phase system)
    println!("=== Sequential Execution (Traditional) ===");
    sequential_example()?;

    // Example 2: LLM-guided intelligent execution
    println!("\n=== LLM-Guided Execution (Intelligent) ===");
    llm_guided_example()?;

    // Example 3: Generic library binding (not CUDA-specific)
    println!("\n=== Generic Library Binding ===");
    generic_library_example()?;

    Ok(())
}

fn sequential_example() -> Result<()> {
    // Create orchestrator with sequential execution
    let orchestrator = BindingOrchestrator::new(ExecutionMode::Sequential);

    // Set up context for CUDA binding
    let context = ToolContext::new(
        Path::new("./cuda-12.0").to_path_buf(),
        Path::new("./cuda-bindings").to_path_buf(),
        "cuda".to_string(),
    );

    // Execute - this runs the traditional pipeline:
    // 1. Discover headers
    // 2. Generate FFI
    // 3. Analyze patterns
    // 4. Generate wrappers
    // 5. Detect dependencies
    // 6. Validate build
    let result = orchestrator.execute(context)?;

    println!("Sequential execution completed!");
    println!("Headers found: {}", result.headers.len());
    println!("Build errors: {}", result.build_errors.len());

    Ok(())
}

fn llm_guided_example() -> Result<()> {
    // Create orchestrator with LLM guidance
    let orchestrator = BindingOrchestrator::new(ExecutionMode::LlmGuided {
        model: "llama3.2".to_string(),
        max_iterations: 15,
    });

    // Set up context for a complex library
    let context = ToolContext::new(
        Path::new("./complex-lib").to_path_buf(),
        Path::new("./complex-bindings").to_path_buf(),
        "complex_lib".to_string(),
    );

    // Execute - the LLM will intelligently choose tools:
    // - If build fails, it might run build_fixing tool
    // - If dependencies are missing, it might run dependency_detection
    // - It can iterate and retry until success
    // - It can adapt to different library structures
    let result = orchestrator.execute(context)?;

    println!("LLM-guided execution completed!");
    println!("Final result: {} build errors", result.build_errors.len());

    Ok(())
}

fn generic_library_example() -> Result<()> {
    println!("Demonstrating generic library support (no hardcoded CUDA logic):");

    // Example: Binding PyTorch C++ API
    let _pytorch_context = ToolContext::new(
        Path::new("./pytorch/torch/csrc").to_path_buf(),
        Path::new("./pytorch-bindings").to_path_buf(),
        "torch".to_string(),
    );

    // Example: Binding OpenCV
    let _opencv_context = ToolContext::new(
        Path::new("./opencv/include").to_path_buf(),
        Path::new("./opencv-bindings").to_path_buf(),
        "opencv".to_string(),
    );

    // Example: Binding FFmpeg
    let _ffmpeg_context = ToolContext::new(
        Path::new("./ffmpeg/libav*").to_path_buf(),
        Path::new("./ffmpeg-bindings").to_path_buf(),
        "ffmpeg".to_string(),
    );

    println!("✓ PyTorch context created (no CUDA assumptions)");
    println!("✓ OpenCV context created (no CUDA assumptions)");
    println!("✓ FFmpeg context created (no CUDA assumptions)");

    // The dependency detection tool will use configurable mappings
    // instead of hardcoded CUDA/cuDNN logic:
    //
    // For PyTorch: Look for torch_, at::, c10:: patterns
    // For OpenCV: Look for cv::, Mat, VideoCapture patterns
    // For FFmpeg: Look for av_, AVCodec, AVFormat patterns
    //
    // All configured through DependencyConfig structs!

    Ok(())
}

// Example of how build fixing works automatically
#[allow(dead_code)]
fn build_fixing_example() -> Result<()> {
    println!("=== Automatic Build Fixing Example ===");

    let orchestrator = BindingOrchestrator::new(ExecutionMode::LlmGuided {
        model: "llama3.2".to_string(),
        max_iterations: 10,
    });

    let mut context = ToolContext::new(
        Path::new("./some-library").to_path_buf(),
        Path::new("./bindings").to_path_buf(),
        "some_lib".to_string(),
    );

    // Simulate some initial build errors
    context.build_errors = vec![
        bindings_generat::output::error_parser::BuildError::LibraryNotFound {
            library_name: "cudart".to_string(),
        },
    ];

    println!("Starting with {} build errors", context.build_errors.len());

    // The LLM orchestrator will:
    // 1. See the build errors
    // 2. Choose to run the build_fixing tool
    // 3. The build_fixing tool will:
    //    - Map cuda_runtime_get_version to libcudart
    //    - Find libcudart.so in system paths
    //    - Find cuda_runtime.h in CUDA include paths
    //    - Generate appropriate linking flags
    // 4. Re-run validation
    // 5. If still failing, iterate with different fixes

    let result = orchestrator.execute(context)?;
    println!(
        "After automatic fixing: {} build errors",
        result.build_errors.len()
    );

    Ok(())
}
