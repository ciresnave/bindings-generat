use crate::analyzer::raii::{HandleType, LifecyclePair};
use std::fmt::Write;
use tracing::{debug, info};

/// Generated RAII wrapper code
#[derive(Debug, Clone)]
pub struct RaiiWrapper {
    pub type_name: String,
    pub handle_type: String,
    pub code: String,
}

/// Generate RAII wrapper for a handle type
pub fn generate_raii_wrapper(
    handle: &HandleType,
    pair: &LifecyclePair,
    _lib_name: &str,
) -> RaiiWrapper {
    debug!("Generating RAII wrapper for {}", handle.name);

    let type_name = to_rust_type_name(&handle.name);
    let mut code = String::new();

    // Generate the struct
    writeln!(code, "/// Safe wrapper for `{}`", handle.name).unwrap();
    writeln!(code, "pub struct {} {{", type_name).unwrap();
    writeln!(code, "    handle: *mut {},", handle.name).unwrap();
    writeln!(code, "}}").unwrap();
    writeln!(code).unwrap();

    // Generate impl block with new() constructor
    writeln!(code, "impl {} {{", type_name).unwrap();
    writeln!(code, "    /// Create a new instance").unwrap();
    writeln!(code, "    pub fn new() -> Result<Self, Error> {{").unwrap();
    writeln!(code, "        unsafe {{").unwrap();
    writeln!(code, "            let handle = ffi::{}();", pair.create_fn).unwrap();
    writeln!(code, "            if handle.is_null() {{").unwrap();
    writeln!(code, "                Err(Error::NullPointer)").unwrap();
    writeln!(code, "            }} else {{").unwrap();
    writeln!(code, "                Ok(Self {{ handle }})").unwrap();
    writeln!(code, "            }}").unwrap();
    writeln!(code, "        }}").unwrap();
    writeln!(code, "    }}").unwrap();
    writeln!(code, "}}").unwrap();
    writeln!(code).unwrap();

    // Generate Drop implementation
    writeln!(code, "impl Drop for {} {{", type_name).unwrap();
    writeln!(code, "    fn drop(&mut self) {{").unwrap();
    writeln!(code, "        unsafe {{").unwrap();
    writeln!(code, "            if !self.handle.is_null() {{").unwrap();
    writeln!(
        code,
        "                ffi::{}(self.handle);",
        pair.destroy_fn
    )
    .unwrap();
    writeln!(code, "            }}").unwrap();
    writeln!(code, "        }}").unwrap();
    writeln!(code, "    }}").unwrap();
    writeln!(code, "}}").unwrap();
    writeln!(code).unwrap();

    // Generate Send/Sync if appropriate (conservative: don't implement by default)
    writeln!(code, "// Note: Implement Send/Sync manually if thread-safe").unwrap();
    writeln!(code, "// unsafe impl Send for {} {{}}", type_name).unwrap();
    writeln!(code, "// unsafe impl Sync for {} {{}}", type_name).unwrap();
    writeln!(code).unwrap();

    info!(
        "Generated RAII wrapper for {} -> {}",
        handle.name, type_name
    );

    RaiiWrapper {
        type_name,
        handle_type: handle.name.clone(),
        code,
    }
}

/// Generate a basic wrapper type for a handle without lifecycle functions
/// This creates a struct without create() or Drop implementation
pub fn generate_basic_wrapper(handle: &HandleType) -> RaiiWrapper {
    debug!(
        "Generating basic wrapper for {} (no lifecycle)",
        handle.name
    );

    let type_name = to_rust_type_name(&handle.name);
    let mut code = String::new();

    // Generate the struct
    writeln!(code, "/// Wrapper for `{}`", handle.name).unwrap();
    writeln!(
        code,
        "/// Note: No automatic resource management - handle cleanup manually"
    )
    .unwrap();
    writeln!(code, "pub struct {} {{", type_name).unwrap();
    writeln!(code, "    pub handle: *mut {},", handle.name).unwrap();
    writeln!(code, "}}").unwrap();
    writeln!(code).unwrap();

    info!(
        "Generated basic wrapper for {} -> {}",
        handle.name, type_name
    );

    RaiiWrapper {
        type_name,
        handle_type: handle.name.clone(),
        code,
    }
}

/// Convert C type name to idiomatic Rust type name
pub fn to_rust_type_name(c_name: &str) -> String {
    let mut name = c_name.replace("_t", "").replace("_", " ");

    // Convert to PascalCase
    name = name
        .split_whitespace()
        .map(|word| {
            let mut chars = word.chars();
            match chars.next() {
                None => String::new(),
                Some(first) => first.to_uppercase().collect::<String>() + chars.as_str(),
            }
        })
        .collect::<Vec<_>>()
        .join("");

    // Ensure it starts with uppercase
    if !name.is_empty() {
        name = name[0..1].to_uppercase() + &name[1..];
    }

    name
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_to_rust_type_name() {
        assert_eq!(to_rust_type_name("my_context_t"), "MyContext");
        assert_eq!(to_rust_type_name("simple"), "Simple");
        assert_eq!(to_rust_type_name("foo_bar_baz"), "FooBarBaz");
    }
}
