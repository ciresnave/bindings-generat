//! FFI binding generation and parsing.
//!
//! This module handles the low-level FFI layer, generating raw bindings from C headers
//! and parsing them into structured data for analysis and code generation.
//!
//! ## Workflow
//!
//! 1. **bindgen** - Generate raw FFI bindings from C/C++ headers using bindgen
//! 2. **parser** - Parse the generated bindings into structured [`FfiInfo`]
//! 3. Analysis modules use [`FfiInfo`] to detect patterns and generate safe wrappers
//!
//! ## Key Types
//!
//! - [`FfiInfo`] - Complete representation of FFI bindings
//! - [`FfiFunction`] - Function signature with parameters and docs
//! - [`FfiType`] - Struct/union type with fields
//! - [`FfiEnum`] - Enum with variants and documentation
//! - [`FfiParam`] - Function parameter with type information
//!
//! ## Safety
//!
//! All types in this module represent unsafe FFI interfaces. The generator module
//! is responsible for creating safe wrappers around these unsafe primitives.

pub mod bindgen;
pub mod enricher;
pub mod parser;

pub use parser::{
    FfiConstant, FfiEnum, FfiEnumVariant, FfiField, FfiFunction, FfiInfo, FfiParam, FfiType,
};

use anyhow::Result;
use std::path::Path;

/// Complete FFI generation and parsing pipeline
/// Returns both the parsed FFI info and the raw bindings code
pub fn generate_and_parse_ffi(
    headers: &[std::path::PathBuf],
    lib_name: &str,
    source_path: &Path,
    config: &crate::Config,
) -> Result<(FfiInfo, String)> {
    // Step 1: Generate raw bindings with bindgen
    let bindings_code = bindgen::generate_bindings(headers, lib_name, source_path, config)?;

    // Step 2: Parse the generated bindings
    let ffi_info = parser::parse_ffi_bindings(&bindings_code)?;

    Ok((ffi_info, bindings_code))
}
