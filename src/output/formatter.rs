use anyhow::Result;
use std::path::Path;
use std::process::Command;
use tracing::{debug, info, warn};

/// Format generated code using rustfmt
pub fn format_code(output_dir: &Path) -> Result<()> {
    info!("Formatting generated code with rustfmt");

    let lib_rs_path = output_dir.join("src").join("lib.rs");

    if !lib_rs_path.exists() {
        warn!("lib.rs not found, skipping formatting");
        return Ok(());
    }

    debug!("Running rustfmt on {}", lib_rs_path.display());

    let output = Command::new("rustfmt")
        .arg(&lib_rs_path)
        .arg("--edition=2021")
        .output();

    match output {
        Ok(output) => {
            if output.status.success() {
                info!("Successfully formatted code");
            } else {
                warn!("rustfmt exited with non-zero status");
                if !output.stderr.is_empty() {
                    let stderr = String::from_utf8_lossy(&output.stderr);
                    debug!("rustfmt stderr: {}", stderr);
                }
            }
        }
        Err(e) => {
            warn!(
                "Failed to run rustfmt: {}. Continuing without formatting.",
                e
            );
        }
    }

    Ok(())
}

/// Check if rustfmt is available
pub fn is_rustfmt_available() -> bool {
    Command::new("rustfmt")
        .arg("--version")
        .output()
        .map(|output| output.status.success())
        .unwrap_or(false)
}
