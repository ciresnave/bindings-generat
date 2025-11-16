# Architecture

## Overview

`bindings-generat` is a hybrid code generation tool that combines rule-based analysis (70%), LLM assistance (25%), and user interaction (5%) to automatically generate safe, idiomatic Rust wrappers from C/C++ libraries.

## Design Principles

1. **Deterministic First**: Use rule-based pattern matching for code generation, not LLMs
2. **LLM for Creative Work**: Use LLMs only for documentation, naming, and examples
3. **Offline Capable**: Core functionality works without LLM or internet
4. **Incremental Generation**: Support regeneration while preserving user customizations
5. **Graceful Degradation**: Each feature can fail independently without breaking the pipeline

## Architecture Layers

### 1. CLI Layer (`src/cli.rs`, `src/main.rs`)
- Parses command-line arguments using `clap`
- Validates input parameters
- Initializes logging and configuration

### 2. Configuration Layer (`src/config.rs`)
- Manages all configuration settings
- Supports both CLI arguments and TOML config files
- Provides defaults for all optional settings

### 3. Discovery Layer (`src/discovery/`)
- **Headers** (`headers.rs`): Finds and identifies C/C++ header files
- **Libraries** (`libraries.rs`): Locates library files and extracts metadata
- **Output**: `DiscoveryResult` with detected library structure

### 4. FFI Generation Layer (`src/ffi/`)
- **Bindgen** (`bindgen.rs`): Wraps bindgen to generate raw FFI bindings
- **Output**: Raw unsafe Rust FFI code

### 5. Analysis Layer (`src/analyzer/`)
- **AST** (`ast.rs`): Parses generated FFI using `syn`
- **Patterns** (`patterns.rs`): Detects common patterns
- **RAII** (`raii.rs`): Identifies create/destroy pairs for RAII wrappers
- **Errors** (`errors.rs`): Detects error handling patterns (status codes → Result)
- **Ownership** (`ownership.rs`): Analyzes pointer ownership and lifetimes
- **Ambiguity** (`ambiguity.rs`): Flags cases needing user input
- **Output**: `AnalysisResult` with detected patterns and types

### 6. Generation Layer (`src/generator/`)
- **Templates** (`templates.rs`): Handlebars template engine
- **Wrappers** (`wrappers.rs`): Generates RAII wrapper types
- **Errors** (`errors.rs`): Generates error enums from status codes
- **Methods** (`methods.rs`): Generates safe method wrappers
- **Builders** (`builders.rs`): Generates builder patterns for complex types
- **Output**: Generated Rust code (AST)

### 7. LLM Layer (`src/llm/`) - Optional
- **Client** (`client.rs`): Ollama API client with retry logic
- **Prompts** (`prompts.rs`): Prompt templates for various tasks
- **Cache** (`cache.rs`): SHA-256 based response caching
- **Docs** (`docs.rs`): Documentation generation orchestration
- **Output**: Enhanced documentation and examples

### 8. Interactive Layer (`src/interactive/`) - Optional
- **Questions** (`questions.rs`): User prompt system using `dialoguer`
- **Decisions** (`decisions.rs`): Stores and reuses user decisions
- **Output**: User choices for ambiguous cases

### 9. Output Layer (`src/output/`)
- **Writer** (`writer.rs`): Writes generated crate structure
- **Formatter** (`formatter.rs`): Runs rustfmt on generated code
- **Validator** (`validator.rs`): Runs cargo check and reports errors
- **Output**: Complete, formatted, validated Rust crate

## Data Flow

```
CLI Input
  ↓
Configuration
  ↓
Discovery → DiscoveryResult
  ↓
FFI Generation → Raw FFI Code
  ↓
Analysis → AnalysisResult
  ↓
Code Generation → Generated Code (AST)
  ↓
[Optional] LLM Enhancement → Enhanced Code + Docs
  ↓
[Optional] Interactive → User Decisions
  ↓
Output → Complete Crate
  ↓
Validation → Verified Crate
```

## Key Data Structures

### DiscoveryResult
```rust
pub struct DiscoveryResult {
    pub main_header: PathBuf,
    pub headers: Vec<PathBuf>,
    pub library_file: Option<PathBuf>,
    pub library_name: String,
    pub version: Option<String>,
}
```

### RaiiPattern (to be implemented)
```rust
pub struct RaiiPattern {
    pub type_name: String,
    pub create_fn: String,
    pub destroy_fn: String,
    pub is_thread_safe: bool,
    pub constructor_params: Vec<Param>,
}
```

### ErrorPattern (to be implemented)
```rust
pub struct ErrorPattern {
    pub status_type: String,
    pub success_variant: String,
    pub error_variants: Vec<ErrorVariant>,
}
```

## Pattern Detection Strategies

### RAII Detection
1. Function name matching: `create*/destroy*`, `init*/free*`, `new*/delete*`
2. Type analysis: Opaque pointer types (`*mut T`)
3. Parameter patterns: First param `T**` for create, `T` for destroy

### Error Handling Detection
1. Enum name matching: `*Status`, `*Error`, `*Result`, `*Code`
2. Success value detection: Usually `0` or first variant
3. Severity categorization: Warning, Error, Fatal

### Ownership Analysis
1. `const T*` → `&T` (immutable borrow)
2. `T*` (non-const) → `&mut T` (mutable borrow)
3. `T**` (output param) → returned in `Result<T>`
4. `T` (value) → `T` (owned)

## Testing Strategy

### Unit Tests
- Each analyzer module: `src/analyzer/*_test.rs`
- Generator modules: `src/generator/*_test.rs`
- Discovery: `tests/fixtures/simple/`

### Integration Tests
- `tests/integration_tests.rs`: End-to-end CLI testing
- `tests/fixtures/`: Test C libraries

### Test Fixtures
- `simple/`: Basic C library with RAII and error patterns
- `cuda_mock/` (future): CUDA-like API with complex patterns

## Current Implementation Status

### ✅ Completed
- Project structure and build system
- CLI argument parsing
- Configuration management
- Discovery layer (headers and libraries)
- Basic test infrastructure

### 🚧 In Progress
- FFI generation (bindgen integration)

### ⏳ Not Started
- Pattern analysis (RAII, errors, ownership)
- Code generation (wrappers, errors, methods)
- LLM integration (Ollama client)
- Interactive mode (questions, decisions)
- Output and validation

## Dependencies

### Core
- `clap`: CLI argument parsing
- `anyhow`: Error handling
- `walkdir`: Directory traversal
- `regex`: Pattern matching

### Code Generation
- `bindgen`: Raw FFI generation
- `syn`: Rust AST parsing
- `quote`: Code generation
- `handlebars`: Template engine
- `prettyplease`: Code formatting

### LLM (Optional)
- `reqwest`: HTTP client for Ollama
- `serde_json`: JSON serialization
- `sha2`: Cache key hashing

### User Interaction
- `dialoguer`: Interactive prompts
- `indicatif`: Progress bars

### Testing
- `assert_cmd`: CLI testing
- `predicates`: Assertions
- `tempfile`: Temporary directories

## Performance Targets

- Small library (<50 functions): <10s (no LLM), <60s (with LLM)
- Medium library (50-200 functions): <30s (no LLM), <5min (with LLM)
- Large library (200+ functions): <2min (no LLM), <15min (with LLM)

Caching reduces subsequent runs to <5 seconds.

## Future Enhancements

1. **C++ Support**: Namespaces, classes, templates
2. **Incremental Regeneration**: Preserve user customizations
3. **Plugin System**: Custom pattern detectors
4. **Multiple Backends**: Support languages beyond Rust
5. **CI/CD Integration**: GitHub Actions, pre-commit hooks
