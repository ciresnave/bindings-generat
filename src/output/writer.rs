use crate::database;
use anyhow::{Context, Result};
use std::fs;
use std::path::Path;
use tracing::{debug, info};

/// Write generated code to output directory
pub fn write_generated_code(
    output_dir: &Path,
    lib_rs_content: &str,
    ffi_bindings: &str,
    ffi_dynamic: &str,
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
    if config.dry_run {
        info!(
            "DRY RUN MODE: Would write generated code to {}",
            output_dir.display()
        );
        let files = vec![
            "Cargo.toml",
            "build.rs",
            ".gitignore",
            "src/lib.rs",
            "src/ffi_bindings.rs",
            "src/ffi.rs",
            "src/loader.rs",
            "src/discovery_shared.rs",
            "src/discovery_install.rs",
            "tests/integration_tests.rs",
            "tests/runtime_tests.rs",
            "tests/functional_tests.rs",
        ];

        for file in files {
            info!("  Would create: {}", output_dir.join(file).display());
        }

        info!("  lib.rs content: {} bytes", lib_rs_content.len());
        info!("  ffi_bindings.rs content: {} bytes", ffi_bindings.len());
        info!(
            "  ffi.rs (dynamic shim) content: {} bytes",
            ffi_dynamic.len()
        );

        return Ok(());
    }

    // Ensure directories exist
    fs::create_dir_all(output_dir.join("src")).context("Failed to create src dir")?;
    fs::create_dir_all(output_dir.join("tests")).context("Failed to create tests dir")?;

    // Primary crate sources
    fs::write(output_dir.join("src").join("lib.rs"), lib_rs_content)
        .context("Failed to write lib.rs")?;
    write_ffi_bindings(output_dir, ffi_bindings)?;
    write_ffi_dynamic(output_dir, ffi_dynamic)?;
    fs::write(output_dir.join("src").join("loader.rs"), loader_content)
        .context("Failed to write loader.rs")?;
    fs::write(
        output_dir.join("src").join("discovery_shared.rs"),
        discovery_shared_content,
    )
    .context("Failed to write discovery_shared.rs")?;
    fs::write(
        output_dir.join("src").join("discovery_install.rs"),
        discovery_install_content,
    )
    .context("Failed to write discovery_install.rs")?;

    // Tests and supporting files
    write_tests(output_dir, tests_content)?;
    write_runtime_tests(output_dir, runtime_tests_content)?;
    write_functional_tests(output_dir, functional_tests_content)?;
    write_readme(output_dir, lib_name)?;
    write_gitignore(output_dir)?;

    // Minimal build.rs that reuses generated discovery_shared.rs at build time
    let build_rs = r#"include!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/discovery_shared.rs"));

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
}
"#;
    fs::write(output_dir.join("build.rs"), build_rs).context("Failed to write build.rs")?;

    // Write Cargo.toml, include runtime deps when dynamic shim present
    write_cargo_toml(
        output_dir,
        lib_name,
        dependencies,
        !ffi_dynamic.trim().is_empty(),
    )?;

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
    let ffi_bindings_path = output_dir.join("src").join("ffi_bindings.rs");
    debug!("Writing {}", ffi_bindings_path.display());

    fs::write(&ffi_bindings_path, bindings).context("Failed to write ffi_bindings.rs")?;
    Ok(())
}

fn write_ffi_dynamic(output_dir: &Path, shim: &str) -> Result<()> {
    let ffi_rs_path = output_dir.join("src").join("ffi.rs");
    debug!("Writing {}", ffi_rs_path.display());

    fs::write(&ffi_rs_path, shim).context("Failed to write ffi.rs (dynamic shim)")?;
    Ok(())
}

fn write_tests(output_dir: &Path, tests: &str) -> Result<()> {
    let tests_path = output_dir.join("tests").join("integration_tests.rs");
    debug!("Writing {}", tests_path.display());

    fs::write(&tests_path, tests).context("Failed to write integration_tests.rs")?;
    Ok(())
}

fn write_runtime_tests(output_dir: &Path, runtime_tests: &str) -> Result<()> {
    let tests_path = output_dir.join("tests").join("runtime_tests.rs");
    debug!("Writing {}", tests_path.display());

    fs::write(&tests_path, runtime_tests).context("Failed to write runtime_tests.rs")?;
    Ok(())
}

fn write_functional_tests(output_dir: &Path, functional_tests: &str) -> Result<()> {
    let tests_path = output_dir.join("tests").join("functional_tests.rs");
    debug!("Writing {}", tests_path.display());

    fs::write(&tests_path, functional_tests).context("Failed to write functional_tests.rs")?;
    Ok(())
}

fn write_readme(output_dir: &Path, lib_name: &str) -> Result<()> {
    let readme_path = output_dir.join("README.md");
    debug!("Writing {}", readme_path.display());

    let crate_name = lib_name.replace("-", "_").to_lowercase();
    let library_name = lib_name;

    let readme_content = format!(
        r#"# {}

Safe Rust bindings for {}, automatically generated.

## Installation

```toml
[dependencies]
{} = "0.1.0"
```

## Prerequisites

{} must be installed on your system.

## Usage

```rust
use {}::*;

fn main() -> Result<(), Error> {{
    // Use the generated bindings
    Ok(())
}}
```

## Documentation

Generate documentation with:

```bash
cargo doc --open
```

## License

Licensed under either of Apache License 2.0 or MIT license at your option.

## Acknowledgments

These bindings were automatically generated using bindings-generat.
"#,
        crate_name, library_name, crate_name, library_name, crate_name
    );

    fs::write(&readme_path, readme_content).context("Failed to write README.md")?;
    Ok(())
}

fn write_cargo_toml(
    output_dir: &Path,
    lib_name: &str,
    extra_deps: &[String],
    include_runtime: bool,
) -> Result<()> {
    let crate_name = lib_name.replace("-", "_").to_lowercase();
    let cargo_toml_path = output_dir.join("Cargo.toml");
    debug!("Writing {}", cargo_toml_path.display());

    // Base package metadata
    let mut toml = format!(
        r#"[package]
name = "{}"
version = "0.1.0"
edition = "2021"
description = "Generated bindings for {}"

[dependencies]
"#,
        crate_name, lib_name
    );

    if include_runtime {
        toml.push_str(
            r#"anyhow = "1.0"
libloading = "0.8"
once_cell = "1.18"
"#,
        );
    }

    // Append any extra dependency lines provided by the caller (already formatted)
    for dep in extra_deps {
        if dep.contains('=') {
            toml.push_str(&format!("{}\n", dep));
        }
    }

    fs::write(&cargo_toml_path, toml).context("Failed to write Cargo.toml")?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{env, fs};

    #[test]
    fn cargo_toml_includes_runtime_deps_when_dynamic_shim_present() {
        let tmp_dir = env::temp_dir().join(format!("bindings_toml_test_{}", std::process::id()));
        if tmp_dir.exists() {
            let _ = fs::remove_dir_all(&tmp_dir);
        }
        fs::create_dir_all(&tmp_dir).expect("create tmp dir");

        let lib_name = "testlib";
        let extra_deps: Vec<String> = vec!["serde = \"1.0\"".to_string()];
        write_cargo_toml(&tmp_dir, lib_name, &extra_deps, true).expect("write_cargo_toml failed");

        let content = fs::read_to_string(tmp_dir.join("Cargo.toml")).expect("read Cargo.toml");
        assert!(content.contains("anyhow = \"1.0\""));
        assert!(content.contains("libloading = \"0.8\""));
        assert!(content.contains("once_cell = \"1.18\""));

        let _ = fs::remove_dir_all(&tmp_dir);
    }
}
