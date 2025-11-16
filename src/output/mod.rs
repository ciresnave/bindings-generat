pub mod formatter;
pub mod validator;
pub mod writer;

pub use formatter::format_code;
pub use validator::validate_code;
pub use writer::write_generated_code;

use anyhow::Result;
use std::path::Path;
use tracing::info;

/// Complete output pipeline: write, format, and validate
pub fn output_generated_code(
    output_dir: &Path,
    lib_rs_content: &str,
    lib_name: &str,
) -> Result<()> {
    info!("Starting output pipeline");

    // Step 1: Write files
    writer::write_generated_code(output_dir, lib_rs_content, lib_name)?;

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
