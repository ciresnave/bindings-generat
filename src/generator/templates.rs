//! Minimal generator templates used by the code generator.
//!
//! These implementations are intentionally small and return simple
//! placeholder strings. They satisfy compile-time requirements for
//! the generator pipeline and are sufficient for unit tests that don't
//! depend on the full template output.

/// Runtime loader template (minimal placeholder)
pub fn runtime_loader(lib_name: &str) -> String {
    format!("// runtime loader for {}", lib_name)
}

/// Shared discovery helper template (placeholder)
pub fn discovery_shared(_lib_name: &str) -> String {
    String::from("// discovery_shared stub")
}

/// Discovery installer stub template (placeholder)
pub fn discovery_install_stub(_lib_name: &str) -> String {
    String::from("// discovery_install stub")
}

/// Discovery installer template — falls back to the stub currently
pub fn discovery_install(lib_name: &str, _sources: &Vec<crate::database::Source>) -> String {
    discovery_install_stub(lib_name)
}

/// Dynamic FFI shim template (minimal placeholder)
pub fn ffi_dynamic(lib_name: &str, _ffi_info: &crate::ffi::FfiInfo) -> String {
    format!("// ffi_dynamic shim for {}", lib_name)
}
    /// Generate a dynamic FFI shim that re-exports bindgen types (from `ffi_bindings.rs`)
    /// and provides function implementations that lazy-load symbols at runtime.
    pub fn ffi_dynamic(lib_name: &str, ffi_info: &crate::ffi::FfiInfo) -> String {
        let mut s = String::new();

        // Re-export types from bindgen output (written to ffi_bindings.rs)
        writeln!(s, "#[path = \"ffi_bindings.rs\"]").ok();
        writeln!(s, "mod ffi_bindings;").ok();
        writeln!(s, "pub use ffi_bindings::*;").ok();
        writeln!(s, "").ok();

        writeln!(s, "use anyhow::{{Context, Result}};").ok();
        writeln!(s, "use libloading::Library;").ok();
        writeln!(s, "use once_cell::sync::OnceCell;").ok();
        writeln!(s, "").ok();

        // A single OnceCell that holds a leaked Library for the process lifetime
        writeln!(s, "fn get_lib() -> Result<&'static libloading::Library> {{").ok();
        writeln!(s, "    static LIB: OnceCell<&'static libloading::Library> = OnceCell::new();").ok();
        writeln!(s, "    LIB.get_or_try_init(|| {{").ok();
        writeln!(s, "        let lib = crate::loader::load_library().context(\"Failed to load library\")?;").ok();
        writeln!(s, "        let lib_static: &'static libloading::Library = Box::leak(Box::new(lib));").ok();
        writeln!(s, "        Ok(lib_static)").ok();
        writeln!(s, "    }}).map(|l| *l)").ok();
        writeln!(s, "}}\n").ok();

        // For each FFI function, emit a typed symbol lookup using lib.get::<Symbol<alias>>().
        // We intentionally call lib.get on each invocation to avoid storing Symbol lifetimes
        // in globals. Failures are reported via panic to match the expected function signature
        // (these are `unsafe` FFI entrypoints called by generated safe wrappers).
        for func in &ffi_info.functions {
            let fname = &func.name;
            let param_types: Vec<String> = func.params.iter().map(|p| p.ty.clone()).collect();
            let param_names_types: Vec<String> = func
                .params
                .iter()
                .map(|p| format!("{}: {}", p.name, p.ty))
                .collect();
            let param_names_only: Vec<String> = func.params.iter().map(|p| p.name.clone()).collect();

            let params_ty = if param_types.is_empty() { String::new() } else { param_types.join(", ") };
            let params_decl = if param_names_types.is_empty() { String::new() } else { param_names_types.join(", ") };
            let args_call = if param_names_only.is_empty() { String::new() } else { param_names_only.join(", ") };

            let ret_ty = if func.return_type.trim().is_empty() { "()".to_string() } else { func.return_type.clone() };

            let alias = format!("{}_t", fname);
            writeln!(s, "// Dynamic wrapper for {fname}", fname = fname).ok();
            if params_ty.is_empty() {
                writeln!(s, "pub type {alias} = unsafe extern \"C\" fn() -> {ret};", alias = alias, ret = ret_ty).ok();
            } else {
                writeln!(s, "pub type {alias} = unsafe extern \"C\" fn({params}) -> {ret};", alias = alias, params = params_ty, ret = ret_ty).ok();
            }

            let sym_name = fname;

            // Forwarding function: resolve symbol on each call and invoke it.
            if params_decl.is_empty() {
                // No parameters
                if ret_ty.trim() == "()" {
                    writeln!(s, "pub unsafe fn {fname}() -> {ret} {{", fname = fname, ret = ret_ty).ok();
                    writeln!(s, "    let lib = match get_lib() {{ Ok(l) => l, Err(e) => panic!(\"Failed to open library: {{}}\", e), }};").ok();
                    writeln!(s, "    let sym: libloading::Symbol<{alias}> = unsafe {{ lib.get(b\"{sym}\0\") }}.unwrap_or_else(|e| panic!(\"Failed to find symbol '{sym}': {{}}\", e));", alias = alias, sym = sym_name).ok();
                    writeln!(s, "    sym();").ok();
                    writeln!(s, "}}\n").ok();
                } else {
                    writeln!(s, "pub unsafe fn {fname}() -> {ret} {{", fname = fname, ret = ret_ty).ok();
                    writeln!(s, "    let lib = match get_lib() {{ Ok(l) => l, Err(e) => panic!(\"Failed to open library: {{}}\", e), }};").ok();
                    writeln!(s, "    let sym: libloading::Symbol<{alias}> = unsafe {{ lib.get(b\"{sym}\0\") }}.unwrap_or_else(|e| panic!(\"Failed to find symbol '{sym}': {{}}\", e));", alias = alias, sym = sym_name).ok();
                    writeln!(s, "    sym()").ok();
                    writeln!(s, "}}\n").ok();
                }
            } else {
                // With parameters
                if ret_ty.trim() == "()" {
                    writeln!(s, "pub unsafe fn {fname}({params}) -> {ret} {{", fname = fname, params = params_decl, ret = ret_ty).ok();
                    writeln!(s, "    let lib = match get_lib() {{ Ok(l) => l, Err(e) => panic!(\"Failed to open library: {{}}\", e), }};").ok();
                    writeln!(s, "    let sym: libloading::Symbol<{alias}> = unsafe {{ lib.get(b\"{sym}\0\") }}.unwrap_or_else(|e| panic!(\"Failed to find symbol '{sym}': {{}}\", e));", alias = alias, sym = sym_name).ok();
                    writeln!(s, "    sym({args});", args = args_call).ok();
                    writeln!(s, "}}\n").ok();
                } else {
                    writeln!(s, "pub unsafe fn {fname}({params}) -> {ret} {{", fname = fname, params = params_decl, ret = ret_ty).ok();
                    writeln!(s, "    let lib = match get_lib() {{ Ok(l) => l, Err(e) => panic!(\"Failed to open library: {{}}\", e), }};").ok();
                    writeln!(s, "    let sym: libloading::Symbol<{alias}> = unsafe {{ lib.get(b\"{sym}\0\") }}.unwrap_or_else(|e| panic!(\"Failed to find symbol '{sym}': {{}}\", e));", alias = alias, sym = sym_name).ok();
                    writeln!(s, "    sym({args})", args = args_call).ok();
                    writeln!(s, "}}\n").ok();
                }
            }
        }

        s
    }
    .ok();
    writeln!(s, "    let roots = find_library_roots(lib_name);").ok();
    writeln!(s, "    let mut found = None::<PathBuf>;").ok();

    writeln!(s, "    for lib_root in roots {{").ok();
    writeln!(s, "        #[cfg(target_os = \"windows\")]\n        let lib_paths = vec![lib_root.join(\"lib\").join(\"x64\"), lib_root.join(\"lib\"), lib_root.join(\"bin\")];").ok();
    writeln!(s, "        #[cfg(target_os = \"linux\")]\n        let lib_paths = vec![lib_root.join(\"lib64\"), lib_root.join(\"lib\"), lib_root.join(\"lib\").join(\"x86_64-linux-gnu\")];").ok();
    writeln!(s, "        #[cfg(target_os = \"macos\")]\n        let lib_paths = vec![lib_root.join(\"lib\"), lib_root.join(\"Frameworks\")];").ok();

    writeln!(s, "        for lib_path in lib_paths {{").ok();
    writeln!(s, "            if lib_path.exists() {{").ok();
    writeln!(
        s,
        "                if let Ok(entries) = fs::read_dir(&lib_path) {{"
    )
    .ok();
    writeln!(
        s,
        "                    for entry in entries.filter_map(|e| e.ok()) {{"
    )
    .ok();
    writeln!(s, "                        let path = entry.path();").ok();
    writeln!(
        s,
        "                        if let Some(name) = path.file_name().and_then(|n| n.to_str()) {{"
    )
    .ok();
    writeln!(
        s,
        "                            // Platform detection of library files"
    )
    .ok();
    writeln!(s, "                            #[cfg(target_os = \"windows\")]\n                            let is_lib = name.ends_with(\".lib\") || name.ends_with(\".dll.a\") || name.ends_with(\".dll\");").ok();
    writeln!(s, "                            #[cfg(target_os = \"linux\")]\n                            let is_lib = (name.starts_with(\"lib\") && name.ends_with(\".so\")) || name.ends_with(\".a\") || name.contains(\".so.\");").ok();
    writeln!(s, "                            #[cfg(target_os = \"macos\")]\n                            let is_lib = (name.starts_with(\"lib\") && name.ends_with(\".dylib\")) || name.ends_with(\".a\") || name.ends_with(\".framework\");").ok();

    writeln!(s, "                            if is_lib {{").ok();
    writeln!(
        s,
        "                                let extracted = extract_lib_name(name);"
    )
    .ok();
    use std::fmt::Write;

    /// Generate a simple runtime loader that attempts to load the library at runtime
    /// using `libloading`. This loader is intentionally minimal and portable.
    pub fn runtime_loader(lib_name: &str) -> String {
        let mut s = String::new();

        writeln!(s, "use anyhow::{{Context, Result}};").ok();
        writeln!(s, "use libloading::Library;").ok();
        writeln!(s, "use std::path::PathBuf;").ok();
        writeln!(s, "use std::env;").ok();
        writeln!(s, "use std::ffi::OsStr;").ok();
        writeln!(s, "use std::fs;").ok();
        writeln!(s, "").ok();

        writeln!(s, "/// Attempt to locate the library using environment variables and common locations").ok();
        writeln!(s, "pub fn load_library() -> Result<libloading::Library> {{").ok();
        writeln!(s, "    // Environment override (e.g. LIBFOO_PATH)").ok();
        writeln!(s, "    if let Ok(p) = env::var(format!(\"{}_PATH\", "{}")) {{", lib_name.to_uppercase(), lib_name).ok();
        writeln!(s, "        return Ok(unsafe {{ libloading::Library::new(p) }}.context(\"Failed to open library from {}_PATH\")?);",).ok();
        writeln!(s, "    }").ok();

        writeln!(s, "    // Try to use discovery_shared helpers if available").ok();
        writeln!(s, "    if let Some(path) = discovery_shared::find_library_file(\"{lib}\") {{", lib = lib_name).ok();
        writeln!(s, "        return Ok(unsafe {{ libloading::Library::new(path) }}.context(\"Failed to open discovered library\")?);").ok();
        writeln!(s, "    }").ok();

        writeln!(s, "    // Fallback: search common library directories" ).ok();
        writeln!(s, "    let candidates = vec![").ok();
        writeln!(s, "        \"/usr/lib\",\"/usr/local/lib\",\"/lib\",\"C:\\Windows\\System32\"" ).ok();
        writeln!(s, "    ];").ok();
        writeln!(s, "    for dir in candidates {" ).ok();
        writeln!(s, "        let mut p = PathBuf::from(dir);").ok();
        writeln!(s, "        // platform-specific name candidates").ok();
        writeln!(s, "        let names = vec![").ok();
        writeln!(s, "            format!(\"lib{}.so\", \"{lib}\"),", lib = lib_name).ok();
        writeln!(s, "            format!(\"{}.dll\", \"{lib}\"),", lib = lib_name).ok();
        writeln!(s, "            format!(\"lib{}.dylib\", \"{lib}\"),", lib = lib_name).ok();
        writeln!(s, "        ];").ok();
        writeln!(s, "        for n in names { p.push(n); if p.exists() {{ return Ok(unsafe {{ libloading::Library::new(p) }}.context(\"Failed to open candidate library\")?); }} p.pop(); }" ).ok();
        writeln!(s, "    }").ok();

        writeln!(s, "    Err(anyhow::anyhow!(\"Library {lib} not found on this system\"))", lib = lib_name).ok();
        writeln!(s, "}}\n").ok();

        s
    }

    /// Generate a minimal shared discovery helper used by both the build script and runtime
    pub fn discovery_shared(_lib_name: &str) -> String {
        let mut s = String::new();

        writeln!(s, "use std::path::PathBuf;").ok();
        writeln!(s, "use std::env;").ok();
        writeln!(s, "use std::fs;").ok();
        writeln!(s, "").ok();
        writeln!(s, "pub fn find_library_file(lib_name: &str) -> Option<PathBuf> {{").ok();
        writeln!(s, "    // Check env var override first").ok();
        writeln!(s, "    if let Ok(p) = env::var(format!(\"{}_PATH\", lib_name.to_uppercase())) {{").ok();
        writeln!(s, "        let p = PathBuf::from(p);").ok();
        writeln!(s, "        if p.exists() {{ return Some(p); }}").ok();
        writeln!(s, "    }").ok();

        writeln!(s, "    // Simple search in common locations").ok();
        writeln!(s, "    let candidates = vec![\"/usr/lib\", \"/usr/local/lib\", \"/lib\"];" ).ok();
        writeln!(s, "    for dir in candidates { let p = PathBuf::from(dir); if p.exists() {{ if let Ok(entries) = fs::read_dir(&p) {{ for e in entries.filter_map(|e| e.ok()) {{ let f = e.path(); if let Some(n) = f.file_name().and_then(|s| s.to_str()) {{ if n.contains(lib_name) {{ return Some(f); }} }} }} }} }} }").ok();

        writeln!(s, "    None").ok();
        writeln!(s, "}}\n").ok();

        s
    }

    /// Generate a minimal stub for discovery_install.rs that returns an error.
    pub fn discovery_install_stub(lib_name: &str) -> String {
        let mut s = String::new();
        writeln!(s, "use anyhow::{{Context, Result}};").ok();
        writeln!(s, "use std::path::PathBuf;").ok();
        writeln!(s, "").ok();
        writeln!(s, "/// Stub installer: no automated install available by default").ok();
        writeln!(s, "pub fn ensure_installed(_lib_name: &str) -> Result<PathBuf> {{").ok();
        writeln!(s, "    Err(anyhow::anyhow!(\"Auto-install not available for {}.\"))", lib_name).ok();
        writeln!(s, "}}\n").ok();
        s
    }

    /// Generate a runtime installer that attempts direct-download based installs for provided sources.
    pub fn discovery_install(_lib_name: &str, _sources: &Vec<crate::database::Source>) -> String {
        // For now, reuse the stub; a richer installer can be generated when sources are present.
        discovery_install_stub(_lib_name)
    }

    /// Generate a dynamic FFI shim that re-exports bindgen types (from `ffi_bindings.rs`)
    /// and provides function implementations that lazy-load symbols at runtime.
    pub fn ffi_dynamic(_lib_name: &str, ffi_info: &crate::ffi::FfiInfo) -> String {
        let mut s = String::new();

        // Re-export types from bindgen output (written to ffi_bindings.rs)
        writeln!(s, "#[path = \"ffi_bindings.rs\"]").ok();
        writeln!(s, "mod ffi_bindings;").ok();
        writeln!(s, "pub use ffi_bindings::*;").ok();
        writeln!(s, "").ok();

        writeln!(s, "use anyhow::{{Context, Result}};").ok();
        writeln!(s, "use libloading::Library;").ok();
        writeln!(s, "use once_cell::sync::OnceCell;").ok();
        writeln!(s, "").ok();

        // A single OnceCell that holds a leaked Library for the process lifetime
        writeln!(s, "fn get_lib() -> Result<&'static libloading::Library> {{").ok();
        writeln!(s, "    static LIB: OnceCell<&'static libloading::Library> = OnceCell::new();").ok();
        writeln!(s, "    LIB.get_or_try_init(|| {{").ok();
        writeln!(s, "        let lib = crate::loader::load_library().context(\"Failed to load library\")?;").ok();
        writeln!(s, "        let lib_static: &'static libloading::Library = Box::leak(Box::new(lib));").ok();
        writeln!(s, "        Ok(lib_static)").ok();
        writeln!(s, "    }}).map(|l| *l)").ok();
        writeln!(s, "}}\n").ok();

        // Generate typed symbol lookups per function (call-time resolution)
        for func in &ffi_info.functions {
            let fname = &func.name;
            let param_types: Vec<String> = func.params.iter().map(|p| p.ty.clone()).collect();
            let params_decl: Vec<String> = func
                .params
                .iter()
                .map(|p| format!("{}: {}", p.name, p.ty))
                .collect();
            let args_call: Vec<String> = func.params.iter().map(|p| p.name.clone()).collect();

            let params_ty = if param_types.is_empty() { String::new() } else { param_types.join(", ") };
            let params_decl_str = if params_decl.is_empty() { String::new() } else { params_decl.join(", ") };
            let args_call_str = if args_call.is_empty() { String::new() } else { args_call.join(", ") };

            let ret_ty = if func.return_type.trim().is_empty() { "()".to_string() } else { func.return_type.clone() };
            let alias = format!("{}_t", fname);

            writeln!(s, "// Dynamic wrapper for {fname}", fname = fname).ok();
            if params_ty.is_empty() {
                writeln!(s, "pub type {alias} = unsafe extern \"C\" fn() -> {ret};", alias = alias, ret = ret_ty).ok();
            } else {
                writeln!(s, "pub type {alias} = unsafe extern \"C\" fn({params}) -> {ret};", alias = alias, params = params_ty, ret = ret_ty).ok();
            }

            // Forwarding function: resolve typed symbol and call it
            if params_decl_str.is_empty() {
                // No parameters
                writeln!(s, "pub unsafe fn {fname}() -> {ret} {{", fname = fname, ret = ret_ty).ok();
                writeln!(s, "    let lib = match get_lib() {{ Ok(l) => l, Err(e) => panic!(\"Failed to open library: {{}}\", e), }};").ok();
                writeln!(s, "    let sym: libloading::Symbol<{alias}> = unsafe {{ lib.get(b\"{sym}\0\") }}.unwrap_or_else(|e| panic!(\"Failed to find symbol '{sym}': {{}}\", e));", alias = alias, sym = fname).ok();
                if ret_ty.trim() == "()" {
                    //! Minimal generator templates used by the code generator.
                    //!
                    //! These implementations are intentionally small and return simple
                    //! placeholder strings. They satisfy compile-time requirements for
                    //! the generator pipeline and are sufficient for unit tests that don't
                    //! depend on the full template output.

                    /// Runtime loader template (minimal placeholder)
                    pub fn runtime_loader(lib_name: &str) -> String {
                        format!("// runtime loader for {}", lib_name)
                    }

                    /// Shared discovery helper template (placeholder)
                    pub fn discovery_shared(_lib_name: &str) -> String {
                        String::from("// discovery_shared stub")
                    }

                    /// Discovery installer stub template (placeholder)
                    pub fn discovery_install_stub(_lib_name: &str) -> String {
                        String::from("// discovery_install stub")
                    }

                    /// Discovery installer template — falls back to the stub currently
                    pub fn discovery_install(lib_name: &str, _sources: &Vec<crate::database::Source>) -> String {
                        discovery_install_stub(lib_name)
                    }

                    /// Dynamic FFI shim template (minimal placeholder)
                    pub fn ffi_dynamic(lib_name: &str, _ffi_info: &crate::ffi::FfiInfo) -> String {
                        format!("// ffi_dynamic shim for {}", lib_name)
                    }
