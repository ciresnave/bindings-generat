use crate::analyzer::errors::ErrorEnum;
use std::fmt::Write;
use tracing::{debug, info};

/// Generate Rust error enum from FFI error enum
pub fn generate_error_enum(
    error_enum: &ErrorEnum,
    enhancements: Option<&crate::llm::CodeEnhancements>,
) -> String {
    debug!("Generating error enum for {}", error_enum.name);

    let mut code = String::new();

    // Generate the error enum
    writeln!(code, "/// Error type for this library").unwrap();
    writeln!(code, "#[derive(Debug, Clone, Copy, PartialEq, Eq)]").unwrap();
    writeln!(code, "pub enum Error {{").unwrap();

    // Add null pointer variant
    writeln!(code, "    /// Null pointer returned").unwrap();
    writeln!(code, "    NullPointer,").unwrap();

    // Add each error variant
    for variant in &error_enum.error_variants {
        let rust_variant = to_rust_variant_name(variant);
        // Use LLM-enhanced message if available, otherwise use variant name
        let doc_comment = enhancements
            .and_then(|e| e.get_error_message(variant))
            .unwrap_or(variant);
        writeln!(code, "    /// {}", doc_comment).unwrap();
        writeln!(code, "    {},", rust_variant).unwrap();
    }

    // Add unknown variant for unrecognized status codes
    writeln!(code, "    /// Unknown error code").unwrap();
    writeln!(code, "    Unknown(i32),").unwrap();

    writeln!(code, "}}").unwrap();
    writeln!(code).unwrap();

    // Generate From implementation for the FFI enum
    writeln!(code, "impl From<ffi::{}> for Error {{", error_enum.name).unwrap();
    writeln!(
        code,
        "    fn from(status: ffi::{}) -> Self {{",
        error_enum.name
    )
    .unwrap();
    writeln!(code, "        match status {{").unwrap();

    // Success case should not be converted to Error (this is for error path)
    for variant in &error_enum.error_variants {
        let rust_variant = to_rust_variant_name(variant);
        writeln!(
            code,
            "            ffi::{}::{} => Error::{},",
            error_enum.name, variant, rust_variant
        )
        .unwrap();
    }

    writeln!(code, "            _ => Error::Unknown(status as i32),").unwrap();
    writeln!(code, "        }}").unwrap();
    writeln!(code, "    }}").unwrap();
    writeln!(code, "}}").unwrap();
    writeln!(code).unwrap();

    // Generate Display implementation
    writeln!(code, "impl std::fmt::Display for Error {{").unwrap();
    writeln!(
        code,
        "    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {{"
    )
    .unwrap();
    writeln!(code, "        match self {{").unwrap();
    writeln!(
        code,
        "            Error::NullPointer => write!(f, \"Null pointer returned\"),"
    )
    .unwrap();

    for variant in &error_enum.error_variants {
        let rust_variant = to_rust_variant_name(variant);
        writeln!(
            code,
            "            Error::{} => write!(f, \"{}\"),",
            rust_variant,
            variant_to_message(variant)
        )
        .unwrap();
    }

    writeln!(
        code,
        "            Error::Unknown(code) => write!(f, \"Unknown error code: {{}}\", code),"
    )
    .unwrap();
    writeln!(code, "        }}").unwrap();
    writeln!(code, "    }}").unwrap();
    writeln!(code, "}}").unwrap();
    writeln!(code).unwrap();

    // Generate std::error::Error implementation
    writeln!(code, "impl std::error::Error for Error {{}}").unwrap();
    writeln!(code).unwrap();

    info!("Generated error enum from {}", error_enum.name);

    code
}

/// Convert C error variant to Rust variant name
fn to_rust_variant_name(c_variant: &str) -> String {
    let mut variant = c_variant
        .replace("ERROR_", "")
        .replace("ERR_", "")
        .replace("STATUS_", "");

    // Convert SCREAMING_SNAKE_CASE to PascalCase
    variant = variant
        .split('_')
        .map(|word| {
            if word.is_empty() {
                String::new()
            } else {
                let lower = word.to_lowercase();
                lower[0..1].to_uppercase() + &lower[1..]
            }
        })
        .collect::<Vec<_>>()
        .join("");

    // Ensure it starts with uppercase
    if !variant.is_empty() && !variant.chars().next().unwrap().is_uppercase() {
        variant = variant[0..1].to_uppercase() + &variant[1..];
    }

    variant
}

/// Convert variant name to human-readable message
fn variant_to_message(variant: &str) -> String {
    variant
        .replace("ERROR_", "")
        .replace("ERR_", "")
        .replace("_", " ")
        .to_lowercase()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_to_rust_variant_name() {
        assert_eq!(
            to_rust_variant_name("ERROR_INVALID_ARGUMENT"),
            "InvalidArgument"
        );
        assert_eq!(to_rust_variant_name("ERR_OUT_OF_MEMORY"), "OutOfMemory");
        assert_eq!(to_rust_variant_name("STATUS_FAILED"), "Failed");
    }

    #[test]
    fn test_variant_to_message() {
        assert_eq!(
            variant_to_message("ERROR_INVALID_ARGUMENT"),
            "invalid argument"
        );
        assert_eq!(variant_to_message("ERR_OUT_OF_MEMORY"), "out of memory");
    }
}
