use anyhow::{Context, Result};
use std::path::{Path, PathBuf};
use tracing::{debug, info};

/// Generate raw FFI bindings using bindgen
pub fn generate_bindings(
    headers: &[PathBuf],
    lib_name: &str,
    source_path: &Path,
) -> Result<String> {
    info!("Generating FFI bindings for {} headers", headers.len());

    if headers.is_empty() {
        anyhow::bail!("No header files provided for bindgen");
    }

    let main_header = &headers[0];
    debug!("Using main header: {}", main_header.display());

    let mut builder = bindgen::Builder::default()
        .header(main_header.to_str().context("Invalid header path")?)
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        // Generate comments from header documentation
        .generate_comments(true)
        // Use core instead of std for no_std compatibility
        .use_core()
        // Generate Debug implementations
        .derive_debug(true)
        // Generate Default implementations where possible
        .derive_default(true)
        // Generate PartialEq/Eq where possible
        .derive_eq(true)
        // Generate PartialOrd/Ord where possible
        .derive_ord(true)
        // Generate Hash where possible
        .derive_hash(true)
        // Prepend the library name as module
        .module_raw_line("ffi", format!("//! Raw FFI bindings for {}", lib_name))
        // Make opaque types also derive Debug
        .opaque_type(".*_impl")
        .opaque_type(".*_internal");

    // Add include paths
    let include_dir = source_path.join("include");
    if include_dir.exists() {
        debug!("Adding include directory: {}", include_dir.display());
        builder = builder.clang_arg(format!("-I{}", include_dir.display()));
    }

    // Add the source directory itself
    builder = builder.clang_arg(format!("-I{}", source_path.display()));

    // Build and generate the bindings
    let bindings = builder
        .generate()
        .context("Failed to generate bindings with bindgen")?;

    // Convert to string
    let bindings_str = bindings.to_string();

    info!(
        "Successfully generated {} bytes of FFI bindings",
        bindings_str.len()
    );

    Ok(bindings_str)
}
