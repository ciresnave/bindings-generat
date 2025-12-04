//! Minimal generator templates used by the code generator.
//!
//! These implementations are intentionally small and return simple
//! placeholder strings. They satisfy compile-time requirements for
//! the generator pipeline and are sufficient for unit tests that don't
//! depend on the full template output.

use std::fmt::Write;

/// Runtime loader template (full implementation)
pub fn runtime_loader(lib_name: &str) -> String {
    let mut s = String::new();
    writeln!(s, "// Runtime loader for {lib}", lib = lib_name).unwrap();
    writeln!(s, "use once_cell::sync::OnceCell;").unwrap();
    writeln!(s, "use libloading::Library;").unwrap();
    writeln!(s, "use std::env;").unwrap();
    writeln!(s).unwrap();
    writeln!(s, "static LIB: OnceCell<Library> = OnceCell::new();").unwrap();
    writeln!(s, "fn find_library_path() -> String {{").unwrap();
    writeln!(
        s,
        "    if let Ok(p) = env::var(\"BINDINGS_GENERAT_LIBRARY\") {{"
    )
    .unwrap();
    writeln!(s, "        return p;").unwrap();
    writeln!(s, "    }}").unwrap();
    writeln!(
        s,
        "    #[cfg(target_os = \"windows\")]\n    return format!(\"{{}}.dll\", \"{name}\");",
        name = lib_name
    )
    .unwrap();
    writeln!(
        s,
        "    #[cfg(target_os = \"macos\")]\n    return format!(\"lib{{}}.dylib\", \"{name}\");",
        name = lib_name
    )
    .unwrap();
    writeln!(s, "    #[cfg(all(not(target_os = \"macos\"), not(target_os = \"windows\")))]\n    return format!(\"lib{{}}.so\", \"{name}\");", name = lib_name).unwrap();
    writeln!(s, "}}").unwrap();
    writeln!(s).unwrap();
    writeln!(s, "pub fn get_lib() -> &'static Library {{").unwrap();
    writeln!(s, "    LIB.get_or_init(|| {{").unwrap();
    writeln!(s, "        let path = find_library_path();").unwrap();
    writeln!(
        s,
        "        unsafe {{ Library::new(path).expect(\"Failed to load shared library\") }}"
    )
    .unwrap();
    writeln!(s, "    }})").unwrap();
    writeln!(s, "}}").unwrap();
    s
}

/// Shared discovery helper template (kept simple)
pub fn discovery_shared(_lib_name: &str) -> String {
    String::from("// discovery_shared placeholder (platform helpers go here)")
}

/// Discovery installer stub template (placeholder)
pub fn discovery_install_stub(_lib_name: &str) -> String {
    String::from("// discovery_install stub — installer not generated")
}

/// Discovery installer template — falls back to the stub currently
pub fn discovery_install(lib_name: &str, _sources: &Vec<crate::database::Source>) -> String {
    discovery_install_stub(lib_name)
}

/// Utility to make a safe Rust identifier for type aliases
fn sanitize_ident(s: &str) -> String {
    s.chars()
        .map(|c| if c.is_alphanumeric() { c } else { '_' })
        .collect()
}

/// Dynamic FFI shim template — emits `src/ffi.rs` content that:
/// - Re-exports `ffi_bindings.rs` types
/// - Provides a runtime loader
/// - Emits typed wrappers that call `lib.get::<libloading::Symbol<...>>()`
pub fn ffi_dynamic(lib_name: &str, ffi_info: &crate::ffi::FfiInfo) -> String {
    let mut s = String::new();
    writeln!(
        s,
        "// Auto-generated dynamic FFI shim for {name}",
        name = lib_name
    )
    .unwrap();
    writeln!(
        s,
        "#![allow(dead_code, non_camel_case_types, non_snake_case, non_upper_case_globals)]"
    )
    .unwrap();
    writeln!(s).unwrap();
    writeln!(
        s,
        "#[derive(Debug)]
    pub enum Error {{
        NullPointer,
        InvalidParameter,
        Specific(String),
    }}"
    )
    .unwrap();
    // Re-export bindgen output so types are available as `crate::ffi::Type`
    writeln!(s, "#[path = \"ffi_bindings.rs\"]").unwrap();
    writeln!(s, "mod ffi_bindings;").unwrap();
    writeln!(s, "pub use ffi_bindings::*;").unwrap();
    writeln!(s).unwrap();

    // Imports for runtime loader
    writeln!(s, "use once_cell::sync::OnceCell;").unwrap();
    writeln!(s, "use libloading::Library;").unwrap();
    writeln!(s, "use libloading::Symbol;").unwrap();
    writeln!(s, "use std::env;").unwrap();
    writeln!(s).unwrap();

    // find_library and singleton
    writeln!(s, "static LIB: OnceCell<Library> = OnceCell::new();").unwrap();
    writeln!(s).unwrap();
    writeln!(s, "fn find_library_path() -> String {{").unwrap();
    writeln!(
        s,
        "    if let Ok(p) = env::var(\"BINDINGS_GENERAT_LIBRARY\") {{"
    )
    .unwrap();
    writeln!(s, "        return p;").unwrap();
    writeln!(s, "    }}").unwrap();
    writeln!(
        s,
        "    #[cfg(target_os = \"windows\")]\n    return format!(\"{{}}.dll\", \"{name}\");",
        name = lib_name
    )
    .unwrap();
    writeln!(
        s,
        "    #[cfg(target_os = \"macos\")]\n    return format!(\"lib{{}}.dylib\", \"{name}\");",
        name = lib_name
    )
    .unwrap();
    writeln!(s, "    #[cfg(all(not(target_os = \"macos\"), not(target_os = \"windows\")))]\n    return format!(\"lib{{}}.so\", \"{name}\");", name = lib_name).unwrap();
    writeln!(s, "}}").unwrap();
    writeln!(s).unwrap();

    writeln!(s, "pub fn get_lib() -> &'static Library {{").unwrap();
    writeln!(s, "    LIB.get_or_init(|| {{").unwrap();
    writeln!(s, "        let path = find_library_path();").unwrap();
    writeln!(
        s,
        "        unsafe {{ Library::new(path).expect(\"Failed to load shared library\") }}"
    )
    .unwrap();
    writeln!(s, "    }})").unwrap();
    writeln!(s, "}}").unwrap();
    writeln!(s).unwrap();

    // For each FFI function, emit a typed wrapper that looks up the symbol at call time
    for func in &ffi_info.functions {
        // Build parameter signature and argument list
        let params_sig = func
            .params
            .iter()
            .map(|p| format!("{}: {}", p.name, p.ty))
            .collect::<Vec<_>>()
            .join(", ");
        let args_list = func
            .params
            .iter()
            .map(|p| p.name.clone())
            .collect::<Vec<_>>()
            .join(", ");

        // Sanitize a type name for the symbol alias
        let alias = sanitize_ident(&format!("{}_fn", func.name));

        // Use the return type verbatim from the parsed FFI info
        let ret = &func.return_type;

        // Emit the type alias for the symbol
        if params_sig.is_empty() {
            writeln!(
                s,
                "type {alias} = unsafe extern \"C\" fn() -> {ret};",
                alias = alias,
                ret = ret
            )
            .unwrap();
            writeln!(
                s,
                "pub unsafe fn {name}() -> {ret} {{",
                name = func.name,
                ret = ret
            )
            .unwrap();
            writeln!(s, "    let sym: Symbol<{alias}> = unsafe {{ get_lib().get(b\"{name}\\0\").expect(\"Missing symbol\") }};", alias = alias, name = func.name).unwrap();
            writeln!(s, "    unsafe {{ (*sym)() }}").unwrap();
            writeln!(s, "}}\n").unwrap();
        } else {
            writeln!(
                s,
                "type {alias} = unsafe extern \"C\" fn({params}) -> {ret};",
                alias = alias,
                params = params_sig,
                ret = ret
            )
            .unwrap();
            writeln!(
                s,
                "pub unsafe fn {name}({params}) -> {ret} {{",
                name = func.name,
                params = params_sig,
                ret = ret
            )
            .unwrap();
            writeln!(s, "    let sym: Symbol<{alias}> = unsafe {{ get_lib().get(b\"{name}\\0\").expect(\"Missing symbol\") }};", alias = alias, name = func.name).unwrap();
            if args_list.is_empty() {
                writeln!(s, "    unsafe {{ (*sym)() }}").unwrap();
            } else {
                writeln!(s, "    unsafe {{ (*sym)({args}) }}", args = args_list).unwrap();
            }
            writeln!(s, "}}\n").unwrap();
        }
    }
    s
}
