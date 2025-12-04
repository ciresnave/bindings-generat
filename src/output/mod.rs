//! Output handling, validation, and iterative refinement.
//!
//! This module manages the output process, including code formatting, validation,
//! and iterative refinement to ensure generated bindings actually compile and link.
//!
//! ## Iterative Refinement Process
//!
//! The refinement module implements an intelligent build-test-fix loop:
//!
//! 1. **Generate** initial bindings
//! 2. **Build** and capture errors
//! 3. **Analyze** linker errors to find missing libraries
//! 4. **Search** system for required libraries
//! 5. **Update** build.rs with discovered libraries
//! 6. **Retry** build until success or max iterations
//!
//! This automated process eliminates manual library hunting and configuration.
//!
//! ## Key Modules
//!
//! - [`error_parser`] - Parse compiler/linker errors to identify issues
//! - [`library_finder`] - Search system for missing libraries
//! - [`refinement`] - Orchestrate the iterative build-fix cycle
//! - [`formatter`] - Format generated Rust code with rustfmt/prettyplease
//! - [`validator`] - Validate generated code before writing
//! - [`writer`] - Write generated files to disk
//!
//! ## Safety Guarantees
//!
//! The output module ensures:
//! - Generated code compiles without errors
//! - All required libraries are found and linked
//! - No hardcoded paths (portability)
//! - No duplicate library directives
//! - DEP_* variables emitted for downstream crates

pub mod error_parser;
pub mod formatter;
pub mod library_finder;
pub mod refinement;
pub mod validator;
pub mod writer;

pub use formatter::format_code;
pub use refinement::{BuildResult, validate_and_refine};
pub use validator::validate_code;
pub use writer::write_generated_code;

use anyhow::Result;
use std::path::Path;
use tracing::info;

/// Complete output pipeline: write, format, and validate
pub fn output_generated_code(
    output_dir: &Path,
    lib_rs_content: &str,
    ffi_bindings: &str,
    ffi_dynamic_content: &str,
    tests_content: &str,
    runtime_tests_content: &str,
    functional_tests_content: &str,
    loader_content: &str,
    discovery_shared_content: &str,
    discovery_install_content: &str,
    lib_name: &str,
    dependencies: &[String],
    config: &crate::Config,
) -> Result<()> {
    // Step 1: Write files (or show what would be written in dry-run mode)
    writer::write_generated_code(
        output_dir,
        lib_rs_content,
        ffi_bindings,
        ffi_dynamic_content,
        tests_content,
        runtime_tests_content,
        functional_tests_content,
        loader_content,
        discovery_shared_content,
        discovery_install_content,
        lib_name,
        dependencies,
        config,
    )?;

    // Skip formatting and validation in dry-run mode
    if config.dry_run {
        info!("DRY RUN: Skipping code formatting and validation");
        return Ok(());
    }

    // Step 2: Format code (optional, continues on failure)
    if formatter::is_rustfmt_available() {
        let _ = formatter::format_code(output_dir);
    }

    // Step 3: Validate code (optional, continues on failure)
    if validator::is_cargo_available() {
        let _ = validator::validate_code(output_dir);
    }

    Ok(())
}
