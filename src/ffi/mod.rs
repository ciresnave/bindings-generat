pub mod bindgen;
pub mod parser;

pub use parser::{
    FfiConstant, FfiEnum, FfiEnumVariant, FfiField, FfiFunction, FfiInfo, FfiParam, FfiType,
};

use anyhow::Result;
use std::path::Path;
use tracing::info;

/// Complete FFI generation and parsing pipeline
/// Returns both the parsed FFI info and the raw bindings code
pub fn generate_and_parse_ffi(
    headers: &[std::path::PathBuf],
    lib_name: &str,
    source_path: &Path,
) -> Result<(FfiInfo, String)> {
    info!("Starting FFI generation and parsing pipeline");

    // Step 1: Generate raw bindings with bindgen
    let bindings_code = bindgen::generate_bindings(headers, lib_name, source_path)?;

    // Step 2: Parse the generated bindings
    let ffi_info = parser::parse_ffi_bindings(&bindings_code)?;

    Ok((ffi_info, bindings_code))
}
