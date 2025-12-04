//! Type-safe enum generation
//!
//! This module generates safe Rust enum wrappers for C enums, providing:
//! - Type-safe enum variants
//! - Conversion traits (From, TryFrom)
//! - Display implementation
//! - Unknown variant handling

use crate::ffi::FfiEnum;
use std::fmt::Write;
use tracing::{debug, info};

/// Generate a type-safe Rust enum wrapper for a C enum
pub fn generate_safe_enum(ffi_enum: &FfiEnum) -> String {
    info!("Generating safe enum wrapper for {}", ffi_enum.name);

    let mut code = String::new();
    let rust_name = to_rust_enum_name(&ffi_enum.name);

    // Generate documentation
    if let Some(docs) = &ffi_enum.docs {
        writeln!(code, "/// {}", docs).unwrap();
    } else {
        writeln!(code, "/// Type-safe wrapper for `{}`", ffi_enum.name).unwrap();
    }
    writeln!(code, "///").unwrap();
    writeln!(
        code,
        "/// This enum provides type-safe access to the C enum values,"
    )
    .unwrap();
    writeln!(
        code,
        "/// with an `Unknown` variant for forward compatibility."
    )
    .unwrap();
    writeln!(code, "#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]").unwrap();
    writeln!(code, "#[repr(i32)]").unwrap();
    writeln!(code, "pub enum {} {{", rust_name).unwrap();

    // Generate variants
    for variant in &ffi_enum.variants {
        let variant_name = to_rust_variant_name(&variant.name);
        if let Some(value) = variant.value {
            writeln!(code, "    /// Value: {}", value).unwrap();
            writeln!(code, "    {} = {},", variant_name, value).unwrap();
        } else {
            writeln!(code, "    {},", variant_name).unwrap();
        }
    }

    // Add Unknown variant for unrecognized values
    writeln!(code, "    /// Unknown or unsupported value").unwrap();
    writeln!(code, "    Unknown(i32),").unwrap();
    writeln!(code, "}}").unwrap();
    writeln!(code).unwrap();

    // Generate From<ffi enum> implementation
    writeln!(
        code,
        "impl From<ffi::{}> for {} {{",
        ffi_enum.name, rust_name
    )
    .unwrap();
    writeln!(code, "    #[inline]").unwrap();
    writeln!(
        code,
        "    fn from(value: ffi::{}) -> Self {{",
        ffi_enum.name
    )
    .unwrap();
    writeln!(code, "        match value {{").unwrap();

    for variant in &ffi_enum.variants {
        let variant_name = to_rust_variant_name(&variant.name);
        writeln!(
            code,
            "            ffi::{}::{} => {}::{},",
            ffi_enum.name, variant.name, rust_name, variant_name
        )
        .unwrap();
    }

    writeln!(
        code,
        "            _ => {}::Unknown(value as i32),",
        rust_name
    )
    .unwrap();
    writeln!(code, "        }}").unwrap();
    writeln!(code, "    }}").unwrap();
    writeln!(code, "}}").unwrap();
    writeln!(code).unwrap();

    // Generate Into<ffi enum> implementation
    writeln!(
        code,
        "impl From<{}> for ffi::{} {{",
        rust_name, ffi_enum.name
    )
    .unwrap();
    writeln!(code, "    #[inline]").unwrap();
    writeln!(code, "    fn from(value: {}) -> Self {{", rust_name).unwrap();
    writeln!(code, "        match value {{").unwrap();

    for variant in &ffi_enum.variants {
        let variant_name = to_rust_variant_name(&variant.name);
        writeln!(
            code,
            "            {}::{} => ffi::{}::{},",
            rust_name, variant_name, ffi_enum.name, variant.name
        )
        .unwrap();
    }

    writeln!(
        code,
        "            {}::Unknown(v) => unsafe {{ std::mem::transmute(v) }},",
        rust_name
    )
    .unwrap();
    writeln!(code, "        }}").unwrap();
    writeln!(code, "    }}").unwrap();
    writeln!(code, "}}").unwrap();
    writeln!(code).unwrap();

    // Generate Display implementation
    writeln!(code, "impl std::fmt::Display for {} {{", rust_name).unwrap();
    writeln!(
        code,
        "    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {{"
    )
    .unwrap();
    writeln!(code, "        match self {{").unwrap();

    for variant in &ffi_enum.variants {
        let variant_name = to_rust_variant_name(&variant.name);
        let display_name = variant_name_to_display(&variant.name);
        writeln!(
            code,
            "            {}::{} => write!(f, \"{}\"),",
            rust_name, variant_name, display_name
        )
        .unwrap();
    }

    writeln!(
        code,
        "            {}::Unknown(v) => write!(f, \"Unknown({{}})\", v),",
        rust_name
    )
    .unwrap();
    writeln!(code, "        }}").unwrap();
    writeln!(code, "    }}").unwrap();
    writeln!(code, "}}").unwrap();
    writeln!(code).unwrap();

    debug!("Generated safe enum wrapper for {}", ffi_enum.name);
    code
}

/// Convert C enum name to Rust enum name
fn to_rust_enum_name(c_name: &str) -> String {
    // Remove common suffixes like _t, _e, etc.
    let base = c_name
        .trim_end_matches("_t")
        .trim_end_matches("_e")
        .trim_end_matches("_enum");

    // Convert to PascalCase if it's not already
    if base.contains('_') {
        base.split('_')
            .filter(|s| !s.is_empty())
            .map(|word| {
                let lower = word.to_lowercase();
                lower[0..1].to_uppercase() + &lower[1..]
            })
            .collect()
    } else {
        // Already in PascalCase or camelCase - ensure first letter is uppercase
        let mut chars = base.chars();
        match chars.next() {
            None => String::new(),
            Some(first) => first.to_uppercase().chain(chars).collect(),
        }
    }
}

/// Convert C enum variant name to Rust variant name
fn to_rust_variant_name(c_variant: &str) -> String {
    // Remove common prefixes and convert to PascalCase
    let mut variant = c_variant.to_string();

    // Remove common prefixes (keep trying until no more matches)
    loop {
        let before = variant.clone();
        for prefix in &["ERROR_", "ERR_", "STATUS_", "FLAG_", "TYPE_", "MODE_"] {
            if variant.starts_with(prefix) {
                variant = variant[prefix.len()..].to_string();
                break;
            }
        }
        if before == variant {
            break;
        }
    }

    // Convert SCREAMING_SNAKE_CASE to PascalCase
    if variant.contains('_') {
        variant
            .split('_')
            .filter(|s| !s.is_empty())
            .map(|word| {
                let lower = word.to_lowercase();
                if lower.is_empty() {
                    String::new()
                } else {
                    lower[0..1].to_uppercase() + &lower[1..]
                }
            })
            .collect()
    } else if variant.chars().all(|c| c.is_uppercase() || c.is_numeric()) {
        // All uppercase - convert to PascalCase
        let lower = variant.to_lowercase();
        lower[0..1].to_uppercase() + &lower[1..]
    } else {
        // Mixed case or already PascalCase - ensure first letter is uppercase
        let mut chars = variant.chars();
        match chars.next() {
            None => String::new(),
            Some(first) => first.to_uppercase().chain(chars).collect(),
        }
    }
}

/// Convert variant name to display-friendly text
fn variant_name_to_display(c_variant: &str) -> String {
    // Remove common prefixes
    let cleaned = c_variant
        .replace("ERROR_", "")
        .replace("ERR_", "")
        .replace("STATUS_", "")
        .replace("FLAG_", "")
        .replace("TYPE_", "")
        .replace("MODE_", "");

    // Convert SCREAMING_SNAKE_CASE to Title Case
    cleaned
        .split('_')
        .filter(|s| !s.is_empty())
        .map(|word| {
            let lower = word.to_lowercase();
            if lower.is_empty() {
                String::new()
            } else {
                lower[0..1].to_uppercase() + &lower[1..]
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_to_rust_enum_name() {
        assert_eq!(to_rust_enum_name("cudnnStatus_t"), "CudnnStatus");
        assert_eq!(to_rust_enum_name("error_code_e"), "ErrorCode");
        assert_eq!(to_rust_enum_name("MyEnum"), "MyEnum");
        assert_eq!(to_rust_enum_name("some_enum_type"), "SomeEnumType");
    }

    #[test]
    fn test_to_rust_variant_name() {
        assert_eq!(to_rust_variant_name("ERROR_SUCCESS"), "Success");
        assert_eq!(
            to_rust_variant_name("STATUS_NOT_INITIALIZED"),
            "NotInitialized"
        );
        assert_eq!(to_rust_variant_name("FLAG_NONE"), "None");
        assert_eq!(to_rust_variant_name("SOME_VALUE"), "SomeValue");
    }

    #[test]
    fn test_variant_name_to_display() {
        assert_eq!(variant_name_to_display("ERROR_SUCCESS"), "Success");
        assert_eq!(
            variant_name_to_display("STATUS_NOT_INITIALIZED"),
            "Not Initialized"
        );
        assert_eq!(variant_name_to_display("FLAG_READ_ONLY"), "Read Only");
    }
}
