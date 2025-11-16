use anyhow::{Context, Result};
use std::path::Path;
use std::process::Command;
use tracing::{debug, info, warn};

/// Validate generated code with cargo check
pub fn validate_code(output_dir: &Path) -> Result<bool> {
    info!("Validating generated code with cargo check");

    let cargo_toml = output_dir.join("Cargo.toml");
    if !cargo_toml.exists() {
        warn!("Cargo.toml not found, skipping validation");
        return Ok(false);
    }

    debug!("Running cargo check in {}", output_dir.display());

    let output = Command::new("cargo")
        .arg("check")
        .arg("--manifest-path")
        .arg(&cargo_toml)
        .arg("--quiet")
        .output()
        .context("Failed to run cargo check")?;

    if output.status.success() {
        info!("✓ Generated code compiles successfully");
        Ok(true)
    } else {
        warn!("Generated code has compilation errors");

        if !output.stderr.is_empty() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            debug!("cargo check stderr:\n{}", stderr);
        }

        Ok(false)
    }
}

/// Check if cargo is available
pub fn is_cargo_available() -> bool {
    Command::new("cargo")
        .arg("--version")
        .output()
        .map(|output| output.status.success())
        .unwrap_or(false)
}
