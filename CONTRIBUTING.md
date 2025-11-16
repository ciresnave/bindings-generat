# Contributing to bindings-generat

Thank you for your interest in contributing to `bindings-generat`!

## Quick Start for Developers

### Prerequisites

- Rust 1.70 or later
- Git
- (Optional) Ollama for LLM features

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/bindings-generat.git
cd bindings-generat

# Build the project
cargo build

# Run tests
cargo test

# Run the CLI
cargo run -- --help
```

### Project Structure

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed architecture documentation.

```
bindings-generat/
├── src/
│   ├── main.rs           # CLI entry point
│   ├── lib.rs            # Main orchestrator
│   ├── cli.rs            # CLI argument parsing
│   ├── config.rs         # Configuration management
│   ├── discovery/        # Phase 1: Library discovery
│   ├── ffi/              # Phase 2: FFI generation
│   ├── analyzer/         # Phase 3: Pattern analysis
│   ├── generator/        # Phase 4: Code generation
│   ├── llm/              # Phase 5: LLM enhancement
│   ├── interactive/      # Phase 6: User interaction
│   ├── output/           # Phase 7: Output writing
│   └── utils/            # Utilities
├── tests/
│   ├── fixtures/         # Test C libraries
│   └── integration_tests.rs
└── templates/            # Code templates (TODO)
```

## Development Workflow

### Running Tests

```bash
# All tests
cargo test

# Specific test
cargo test test_cli_help

# With output
cargo test -- --nocapture

# Integration tests only
cargo test --test integration_tests
```

### Code Style

We use the standard Rust formatting:

```bash
# Format code
cargo fmt

# Check formatting
cargo fmt -- --check

# Lint with clippy
cargo clippy
```

### Adding a New Feature

1. **Create a branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Write tests first** (TDD approach)
   - Add unit tests in the relevant module
   - Add integration tests if applicable

3. **Implement the feature**
   - Follow the existing code structure
   - Add documentation comments (///)
   - Update ARCHITECTURE.md if needed

4. **Verify**
   ```bash
   cargo test
   cargo fmt
   cargo clippy
   ```

5. **Submit a PR**
   - Describe what the feature does
   - Reference any related issues
   - Ensure all tests pass

## Current Development Focus (Phase 1)

We're currently in Phase 1 (Proof of Concept). Priority tasks:

1. **FFI Generation** - Integrate bindgen
2. **Pattern Analysis** - Detect RAII and error patterns
3. **Code Generation** - Generate safe wrappers
4. **Output** - Write and validate generated crates

See [STATUS.md](STATUS.md) for detailed progress.

## How to Contribute

### Easy Wins (Good First Issues)

- Improve error messages
- Add more test fixtures
- Improve documentation
- Add more pattern detection heuristics

### Medium Complexity

- Implement bindgen integration
- Add template system
- Implement AST parsing
- Add more unit tests

### Advanced

- Design and implement code generation
- LLM integration
- Interactive mode
- C++ support

## Testing Guidelines

### Unit Tests

Place unit tests in the same file as the code being tested:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_something() {
        // Test code
    }
}
```

### Integration Tests

Add integration tests to `tests/integration_tests.rs` or create new files in `tests/`.

### Test Fixtures

C library fixtures go in `tests/fixtures/`. Each fixture should have:
- Header file(s)
- README explaining the test scenario
- Expected output (for comparison)

## Documentation

### Code Documentation

Use Rust doc comments (///) for public APIs:

```rust
/// Discovers C/C++ library structure from source directory.
///
/// # Arguments
/// * `source_path` - Path to the library source
///
/// # Returns
/// Result containing DiscoveryResult or error
///
/// # Examples
/// ```no_run
/// let result = discover(&PathBuf::from("/path/to/lib"))?;
/// ```
pub fn discover(source_path: &PathBuf) -> Result<DiscoveryResult> {
    // ...
}
```

### Architecture Documentation

Update `ARCHITECTURE.md` when:
- Adding new modules
- Changing data flow
- Adding new dependencies
- Modifying design patterns

### Status Documentation

Update `STATUS.md` when:
- Completing a major component
- Starting a new phase
- Changing project scope

## Communication

### Reporting Issues

When reporting bugs:
1. Describe what you expected to happen
2. Describe what actually happened
3. Provide a minimal reproduction case
4. Include your environment (OS, Rust version)

### Suggesting Features

When suggesting features:
1. Describe the use case
2. Explain why it's valuable
3. Provide examples if possible
4. Consider backward compatibility

## Code Review

All contributions go through code review. Expect:
- Constructive feedback on code structure
- Suggestions for tests
- Questions about design decisions
- Requests for documentation

This is normal and helps maintain code quality!

## License

By contributing, you agree that your contributions will be licensed under the same license as the project (MIT OR Apache-2.0).

## Questions?

- Open an issue for questions about the project
- Tag issues with `question` label
- Check existing issues first

## Thank You!

Every contribution, no matter how small, is appreciated!
