# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2024-11-24

### Added
- Initial release of bindings-generat
- Automatic RAII wrapper generation with Drop implementations
- Error handling conversion (C status codes → Rust Result types)
- Smart ownership analysis for pointer parameters
- LLM-enhanced documentation generation via Ollama
- Idiomatic Rust naming suggestions for C-style functions
- Archive support (.zip, .tar.gz, .tar, .gz formats)
- Remote URL downloads (http/https)
- Interactive mode for ambiguous pattern clarification
- Offline mode (--no-llm flag) using rule-based generation only
- Library discovery system with community database
- Automated library installation and linking
- Build.rs generation with iterative refinement
- Context enrichment from headers, examples, and tests
- Precondition extraction and validation
- Anti-pattern detection from documentation
- API sequence analysis for proper usage patterns
- Callback detection and safe wrapper generation
- Type alias resolution for handle detection
- Builder pattern generation for complex types
- Enum wrapper generation with type safety
- Zero-cost abstractions with #[inline] and #[repr(transparent)]
- Comprehensive runtime tests generation
- Module-level documentation generation
- README generation with usage examples

### Features
- Multi-phase pipeline (Discovery → FFI → Analysis → Generation → Validation)
- CLI interface with 13+ options
- Configuration management (CLI arguments and TOML files)
- Pattern detection (RAII pairs, error enums, ownership semantics)
- Safe wrapper code generation with proper lifetimes
- Output validation with cargo check
- Source handling (directories, archives, remote URLs)
- Platform support (Windows, Linux, macOS)

### Testing
- 349 unit and integration tests passing
- Test coverage for all major functionality
- Real-world testing with CUDA and cuDNN libraries
- Generated code compilation verified

### Documentation
- Complete README with quick start guide
- Architecture documentation (ARCHITECTURE.md)
- Comprehensive roadmap (ROADMAP.md)
- Contributing guidelines (CONTRIBUTING.md)
- Pre-publication audit completed

### Known Limitations
- Some generated code contains `.unwrap()` on CString conversions (will panic on null bytes)
- Thread safety is not automatically detected
- Library discovery requires manual configuration for some proprietary libraries
- LLM features require Ollama to be installed (optional)

### Dependencies
- bindgen 0.72 for FFI generation
- syn 2.0 for Rust AST parsing
- clap 4.5 for CLI
- Ollama (optional) for LLM enhancements

## [0.0.1] - Development versions

Pre-release development work not published to crates.io.

[Unreleased]: https://github.com/ciresnave/bindings-generat/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/ciresnave/bindings-generat/releases/tag/v0.1.0
