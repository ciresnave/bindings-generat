use anyhow::{Context, Result};
use std::fs;
use std::path::Path;
use tracing::{debug, info};

/// Write generated code to output directory
pub fn write_generated_code(
    output_dir: &Path,
    lib_rs_content: &str,
    ffi_bindings: &str,
    lib_name: &str,
) -> Result<()> {
    info!("Writing generated code to {}", output_dir.display());

    // Create output directory structure
    create_directory_structure(output_dir)?;

    // Write src/lib.rs
    let lib_rs_path = output_dir.join("src").join("lib.rs");
    debug!("Writing {}", lib_rs_path.display());
    fs::write(&lib_rs_path, lib_rs_content).context("Failed to write lib.rs")?;

    // Write src/ffi.rs with actual FFI bindings
    write_ffi_bindings(output_dir, ffi_bindings)?;

    // Write Cargo.toml
    write_cargo_toml(output_dir, lib_name)?;

    // Write build.rs (for bindgen integration)
    write_build_rs(output_dir)?;

    // Write .gitignore
    write_gitignore(output_dir)?;

    info!("Successfully wrote all files to {}", output_dir.display());

    Ok(())
}

fn create_directory_structure(output_dir: &Path) -> Result<()> {
    debug!("Creating directory structure");

    fs::create_dir_all(output_dir).context("Failed to create output directory")?;

    fs::create_dir_all(output_dir.join("src")).context("Failed to create src directory")?;

    Ok(())
}

fn write_cargo_toml(output_dir: &Path, lib_name: &str) -> Result<()> {
    let cargo_toml_path = output_dir.join("Cargo.toml");
    debug!("Writing {}", cargo_toml_path.display());

    let content = format!(
        r#"[package]
name = "{}"
version = "0.1.0"
edition = "2021"

[dependencies]

[build-dependencies]
bindgen = "0.70"

[lib]
crate-type = ["lib", "cdylib", "staticlib"]
"#,
        lib_name
    );

    fs::write(&cargo_toml_path, content).context("Failed to write Cargo.toml")?;

    Ok(())
}

fn write_build_rs(output_dir: &Path) -> Result<()> {
    let build_rs_path = output_dir.join("build.rs");
    debug!("Writing {}", build_rs_path.display());

    let content = r#"fn main() {
    // In a real scenario, you would run bindgen here
    // pointing to the C headers from the source library
    
    // Example:
    // let bindings = bindgen::Builder::default()
    //     .header("wrapper.h")
    //     .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
    //     .generate()
    //     .expect("Unable to generate bindings");
    //
    // let out_path = std::path::PathBuf::from(std::env::var("OUT_DIR").unwrap());
    // bindings
    //     .write_to_file(out_path.join("bindings.rs"))
    //     .expect("Couldn't write bindings!");
    
    println!("cargo:rerun-if-changed=build.rs");
}
"#;

    fs::write(&build_rs_path, content).context("Failed to write build.rs")?;

    Ok(())
}

fn write_gitignore(output_dir: &Path) -> Result<()> {
    let gitignore_path = output_dir.join(".gitignore");
    debug!("Writing {}", gitignore_path.display());

    let content = r#"/target
/Cargo.lock
**/*.rs.bk
*.pdb
.DS_Store
"#;

    fs::write(&gitignore_path, content).context("Failed to write .gitignore")?;

    Ok(())
}

fn write_ffi_bindings(output_dir: &Path, bindings: &str) -> Result<()> {
    let ffi_rs_path = output_dir.join("src").join("ffi.rs");
    debug!("Writing {}", ffi_rs_path.display());

    fs::write(&ffi_rs_path, bindings).context("Failed to write ffi.rs")?;

    Ok(())
}
