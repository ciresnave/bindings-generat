# bindings-generat

![](bindings-generat.png)

Automatically generate safe, idiomatic Rust wrapper crates from C/C++ libraries with minimal user interaction.

## Overview

`bindings-generat` automatically generates safe Rust wrappers from C/C++ libraries using bindgen (for raw FFI), rule-based pattern analysis (for RAII and error handling), LLM enhancement (for documentation and naming), and optional interactive clarification.

**LLM integration is now fully featured and enabled by default** when Ollama is available. The tool automatically detects Ollama and uses it to enhance generated bindings with better documentation, idiomatic naming, and usage examples.

## Features

- **Automatic RAII Wrapper Generation**: Detects create/destroy pairs and generates Drop implementations
- **Error Handling Conversion**: Converts C status codes to Rust `Result` types
- **Smart Ownership Analysis**: Determines proper lifetime and ownership patterns
- **LLM-Enhanced Documentation**: Uses local Ollama to generate comprehensive docs and examples (enabled by default)
- **Idiomatic Naming**: LLM suggests Rust-friendly names for C-style functions
- **🎉 Ecosystem Integration (100+ Crates)**: Automatically integrates with the entire Rust ecosystem!
  - Detects library category (Math, Graphics, ML, Networking, etc.)
  - Recommends relevant crates from 100+ supported libraries
  - Generates Cargo.toml with proper feature flags
  - 12-tier prioritization system (Universal → Low-level Protocols)
  - **First FFI generator with comprehensive ecosystem awareness**
- **Archive Support**: Accepts .zip, .tar.gz, .tar, and .gz files as input
- **Remote Downloads**: Download archives directly from URLs (http/https)
- **Interactive Mode**: Asks clarifying questions for ambiguous cases (optional)
- **Offline Mode**: Works without LLM using rule-based generation only (use `--no-llm`)

## How It Works

**bindings-generat** uses a multi-layered approach:

1. **Rule-Based Analysis**:
   - Pattern matching to detect RAII pairs (e.g., `foo_create`/`foo_destroy`)
   - Status code to error enum conversion
   - Ownership inference from parameter types
   - Generates safe, compilable wrappers

2. **LLM Enhancement** (Default when Ollama is available):
   - Generates comprehensive documentation with examples
   - Suggests idiomatic Rust naming conventions
   - Creates usage examples for functions
   - Enhances error messages with context
   - Recommends API design improvements

3. **Interactive Refinement** (Optional):
   - Asks for clarification on ambiguous patterns
   - Allows user guidance on design decisions
   - Stores choices for future runs

## Quick Start

### Installation

```bash
cargo install bindings-generat
```

### Simplest Example

```bash
# Just provide a source - output goes to ./bindings-output/
bindings-generat libraryv1.0.zip
```

That's it! The tool will:
1. Extract the archive
2. Detect library structure automatically  
3. Prompt for LLM model selection (if needed)
4. Generate safe Rust bindings in `./bindings-output/`
5. Validate the generated code with `cargo check`

### First Run Experience

On first run, if Ollama is not installed, you'll see:

```
🔍 Ollama not found on your system.

bindings-generat uses AI to enhance generated bindings.
This requires Ollama (local LLM runtime).

Choose installation option:
  1. System-wide install (requires admin/sudo, persists after run)
  2. Portable install in temp directory (~1.3GB, auto-cleanup available)
  3. Skip LLM features (generates basic bindings only)

Your choice [1/2/3]: 
```

**Recommended:** Choose option 2 (portable) for a hassle-free experience with automatic cleanup.

### Basic Usage

```bash
# Simplest usage - just provide the source
# Output automatically goes to ./bindings-output/
bindings-generat libraryv1.0.zip

# From a directory
bindings-generat /path/to/library

# From a remote URL (e.g., GitHub release)
bindings-generat https://github.com/owner/repo/archive/v1.0.tar.gz

# Custom output directory
bindings-generat library.tar.gz --output custom-wrapper-rs

# With interactive mode
bindings-generat /path/to/library --interactive

# Without LLM (offline/fast mode)
bindings-generat /path/to/library --no-llm

# Specify a different LLM model
bindings-generat /path/to/library --model qwen2.5-coder:7b
```

## Requirements

- Rust 1.70 or later
- C/C++ headers for the library you want to wrap
- libclang (for bindgen)

### Ollama (for LLM features)

**bindings-generat will offer to install Ollama automatically** if not detected. You'll have three options:

1. **System-wide install** - Requires admin/sudo, persists after run
2. **Portable install** - No permissions needed, installed in temp directory with optional auto-cleanup
3. **Skip LLM features** - Uses basic rule-based generation only

The portable install option downloads Ollama (~300MB) and a model (~1GB) to a temporary directory, runs it during binding generation, and optionally cleans it up afterward. Perfect for CI/CD or restricted environments.

**Manual installation (optional):**
```bash
# macOS/Linux
curl https://ollama.ai/install.sh | sh
ollama pull qwen2.5-coder:1.5b

# Windows
winget install Ollama.Ollama
ollama pull qwen2.5-coder:1.5b
```

## Current Status

**Phase 1 (Core Functionality)** - ✅ COMPLETE  
**Phase 2 (LLM Integration)** - ✅ COMPLETE

All features are implemented and tested:

✅ **Completed:**
- Complete end-to-end pipeline (all 8 phases)
- CLI interface with all 13+ options
- Configuration management (CLI and TOML)
- Library discovery (headers, binaries, version detection)
- FFI generation with bindgen
- Pattern analysis (RAII, error handling, ownership)
- Safe wrapper code generation
- **LLM integration with Ollama** (caching, retry logic, auto-detection)
- **Enhanced documentation generation** (function docs, error messages)
- **Idiomatic naming suggestions** (C style → Rust style)
- Output and validation (Cargo.toml, build.rs, lib.rs)
- Source handling (directory, archives, URLs)
- Comprehensive test suite (26 passing tests)
- Generated code compiles successfully

📊 **Test Results:**
- 16 unit tests passing
- 10 LLM module tests passing (cache, prompts, client, enhancer)
- 3 source tests passing
- All CLI options tested
- Generated code compilation verified

📖 **Documentation:**
- [STATUS.md](STATUS.md) - Detailed development status
- [ARCHITECTURE.md](ARCHITECTURE.md) - System architecture
- [TEST_SUMMARY.md](TEST_SUMMARY.md) - Comprehensive test documentation
- [IMPLEMENTATION_COMPLETE.md](IMPLEMENTATION_COMPLETE.md) - Phase 1 notes

🎯 **Next Phase:**
- Phase 3: Interactive mode refinement (clarifying questions, user preferences)
- Continued improvements to pattern detection accuracy
- Additional LLM-powered features (type conversion suggestions, lifetime inference)

See [STATUS.md](STATUS.md) for detailed progress tracking.

## License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.

## Contributing

Contributions are welcome! Please read CONTRIBUTING.md for guidelines.
