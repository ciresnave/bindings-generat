# Tool-Based Architecture Migration

This document explains the major architectural transformation from rigid phase-based execution to a flexible, intelligent tool-based system.

## The Problem with Phases

The original system used a rigid 8-phase pipeline:

```
Phase 1: Discovery → Phase 2: FFI Gen → Phase 3: Pattern Analysis → ... → Phase 8: Done ✅
```

**Issues:**
- ❌ **Misleading Success**: Showed "🎉 Done!" even when builds failed
- ❌ **No Error Recovery**: When builds failed, tool just gave up
- ❌ **Hardcoded Logic**: CUDA/cuDNN assumptions baked into dependency detection
- ❌ **No Adaptability**: Couldn't handle different library structures or iterate on failures

## The New Tool-Based Solution

### Core Architecture

```rust
// Flexible execution modes
pub enum ExecutionMode {
    Sequential,           // Traditional pipeline
    LlmGuided {          // Intelligent orchestration
        model: String,
        max_iterations: usize,
    },
}

// Shared context between all tools
pub struct ToolContext {
    pub source_path: PathBuf,
    pub headers: Vec<PathBuf>,
    pub ffi_info: Option<FfiInfo>,
    pub build_errors: Vec<String>,
    pub dependencies: Vec<DependencyInfo>,
    // ... shared state
}

// Standardized tool interface
pub trait Tool {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    fn requirements(&self) -> Vec<String>;
    fn provides(&self) -> Vec<String>;
    fn execute(&self, context: ToolContext) -> Result<ToolResult>;
}
```

### Available Tools

1. **`discover_headers`** - Find header files
2. **`generate_ffi`** - Generate FFI bindings  
3. **`analyze_patterns`** - Pattern analysis
4. **`generate_wrappers`** - Safe wrapper generation
5. **`detect_dependencies`** - Generic dependency detection (no hardcoded CUDA)
6. **`validate_build`** - Build validation
7. **`fix_build_errors`** - Automatic build error resolution
8. **`enhance_documentation`** - LLM-powered documentation

### Generic Dependency Detection

**Before (Hardcoded):**
```rust
// Old system assumed CUDA everywhere
if function_name.contains("cuda") || function_name.contains("cuDNN") {
    add_cuda_dependency();
}
```

**After (Configurable):**
```rust
// New system uses configurable mappings
pub struct DependencyConfig {
    pub name: String,
    pub env_vars: Vec<String>,           // ["CUDA_PATH", "CUDA_HOME"]
    pub common_paths: Vec<String>,       // ["/usr/local/cuda/lib64"]
    pub header_indicators: Vec<String>,  // ["cuda_runtime.h", "cudnn.h"]
    pub function_patterns: Vec<String>,  // ["cuda*", "cu*", "*cuDNN*"]
    pub library_names: Vec<String>,      // ["cudart", "cublas", "cudnn"]
}
```

Now supports **any library**:
- PyTorch: `torch_`, `at::`, `c10::` patterns
- OpenCV: `cv::`, `Mat`, `VideoCapture` patterns
- FFmpeg: `av_`, `AVCodec`, `AVFormat` patterns

### LLM-Guided Execution

The LLM orchestrator can intelligently choose tools:

```
LLM: "I see build errors. Let me run build_fixing tool."
LLM: "Symbols resolved but still failing. Let me try dependency_detection."
LLM: "Dependencies found but headers missing. Let me discover more headers."
LLM: "Everything looks good now. Generation complete!"
```

**Key Benefits:**
- ✅ **Adaptive Problem Solving**: Can iterate and fix issues
- ✅ **Context Aware**: Understands what went wrong and how to fix it
- ✅ **Flexible Ordering**: Doesn't follow rigid phase sequence
- ✅ **Automatic Recovery**: Handles failures intelligently

## Usage Examples

### Sequential Mode (Traditional)
```rust
let orchestrator = BindingOrchestrator::new(ExecutionMode::Sequential);
let result = orchestrator.execute(context)?; // Runs fixed pipeline
```

### LLM-Guided Mode (Intelligent)
```rust
let orchestrator = BindingOrchestrator::new(ExecutionMode::LlmGuided {
    model: "llama3.2".to_string(),
    max_iterations: 15,
});
let result = orchestrator.execute(context)?; // LLM chooses tools
```

### Generic Library Support
```rust
// PyTorch (no CUDA assumptions!)
let context = ToolContext {
    source_path: Path::new("./pytorch/torch/csrc").to_path_buf(),
    library_name: "torch".to_string(),
    // ... tool will auto-detect PyTorch patterns
};

// OpenCV (no CUDA assumptions!)
let context = ToolContext {
    source_path: Path::new("./opencv/include").to_path_buf(), 
    library_name: "opencv".to_string(),
    // ... tool will auto-detect OpenCV patterns
};
```

## Migration Path

### Phase 1: Tool Creation ✅ DONE
- [x] Created tool-based architecture
- [x] Converted phases to individual tools
- [x] Implemented generic dependency detection
- [x] Added automatic build fixing

### Phase 2: LLM Integration ✅ DONE
- [x] Built LLM orchestrator
- [x] Added intelligent tool selection
- [x] Implemented iterative problem solving

### Phase 3: Integration (Next)
- [ ] Update main CLI to use new architecture
- [ ] Add configuration for execution modes
- [ ] Migration guide for existing users
- [ ] Performance testing and optimization

## Benefits Summary

| Feature | Old System | New System |
|---------|------------|------------|
| Error Handling | ❌ Misleading "Done!" | ✅ Proper error reporting |
| Failure Recovery | ❌ Give up on failure | ✅ Automatic fixing attempts |
| Library Support | ❌ Hardcoded CUDA | ✅ Generic, configurable |
| Adaptability | ❌ Fixed pipeline | ✅ Intelligent orchestration |
| Problem Solving | ❌ Linear progression | ✅ Iterative improvement |

The new architecture transforms bindings-generat from a rigid pipeline tool into an intelligent, adaptive system that can handle any C/C++ library with automatic error recovery and LLM-guided problem solving.