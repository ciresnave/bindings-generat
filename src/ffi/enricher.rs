//! FFI bindings post-processor that enriches raw bindgen output with comprehensive documentation.
//!
//! This module transforms bare FFI declarations into well-documented, easy-to-understand
//! unsafe bindings by injecting documentation from multiple sources:
//! - Header comments and inline docs
//! - Enriched context (safety, performance, constraints)
//! - Common pitfalls and gotchas
//! - Usage examples and patterns
//!
//! The goal is to make the unsafe FFI layer so well-documented that it's safe to use
//! even without additional wrappers.

use crate::enrichment::context::{EnhancedContext, FunctionContext};
use crate::ffi::{FfiFunction, FfiInfo};
use anyhow::Result;
use syn::{File, Item, ItemForeignMod, ItemType, parse_str};
use tracing::info;

/// Enriches raw bindgen FFI output with comprehensive documentation
pub fn enrich_ffi_bindings(
    raw_bindings: &str,
    ffi_info: &FfiInfo,
    enriched_context: &EnhancedContext,
) -> Result<String> {
    info!("Enriching FFI bindings with documentation");

    // Parse the bindgen output as a Rust syntax tree
    let syntax_tree: File = parse_str(raw_bindings)?;

    let mut enriched_output = String::new();

    // Add comprehensive module-level documentation
    enriched_output.push_str(&generate_module_docs(ffi_info, enriched_context));

    // Process each item in the syntax tree
    for item in &syntax_tree.items {
        match item {
            Item::ForeignMod(foreign_mod) => {
                enriched_output.push_str(&enrich_foreign_mod(
                    foreign_mod,
                    ffi_info,
                    enriched_context,
                )?);
            }
            Item::Type(type_alias) => {
                enriched_output.push_str(&enrich_type_alias(
                    type_alias,
                    ffi_info,
                    enriched_context,
                )?);
            }
            Item::Struct(struct_item) => {
                enriched_output.push_str(&enrich_struct(struct_item, ffi_info, enriched_context)?);
            }
            Item::Enum(enum_item) => {
                enriched_output.push_str(&enrich_enum(enum_item, ffi_info, enriched_context)?);
            }
            Item::Const(const_item) => {
                enriched_output.push_str(&enrich_const(const_item)?);
            }
            _ => {
                // Pass through other items unchanged
                enriched_output.push_str(&quote::quote!(#item).to_string());
                enriched_output.push('\n');
            }
        }
    }

    info!("✓ FFI bindings enriched with documentation");
    Ok(enriched_output)
}

/// Generate comprehensive module-level documentation
fn generate_module_docs(ffi_info: &FfiInfo, enriched_context: &EnhancedContext) -> String {
    let mut docs = String::new();

    // Add strict safety lints at the module level
    docs.push_str("// Enforce safety best practices for FFI code\n");
    docs.push_str("#![deny(unsafe_op_in_unsafe_fn)]\n");
    docs.push_str("#![deny(improper_ctypes)]\n");
    docs.push_str("#![deny(improper_ctypes_definitions)]\n");
    docs.push_str("#![warn(missing_docs)]\n");
    docs.push_str("#![warn(clippy::missing_safety_doc)]\n\n");

    docs.push_str("//! # FFI Bindings\n//!\n");
    docs.push_str("//! Auto-generated unsafe FFI bindings with comprehensive documentation.\n");
    docs.push_str("//!\n");
    docs.push_str("//! ⚠️  **All functions in this module are unsafe** and require careful use.\n");
    docs.push_str("//! Read the documentation for each function thoroughly before calling.\n");
    docs.push_str("//!\n");

    // Add initialization sequence if we have context
    let init_functions: Vec<_> = enriched_context
        .functions
        .values()
        .filter(|ctx| {
            ctx.global_state
                .as_ref()
                .map_or(false, |gs| gs.requires_init)
        })
        .collect();

    if !init_functions.is_empty() {
        docs.push_str("//! ## Initialization\n//!\n");
        docs.push_str("//! This library requires initialization. Call functions in this order:\n");
        for (i, ctx) in init_functions.iter().enumerate() {
            docs.push_str(&format!("//! {}. `{}()`\n", i + 1, ctx.name));
        }
        docs.push_str("//!\n");
    }

    // Add thread safety summary
    let thread_unsafe: Vec<_> = enriched_context
        .functions
        .values()
        .filter(|ctx| {
            ctx.thread_safety
                .as_ref()
                .map_or(false, |ts| !ts.trait_bounds.sync)
        })
        .collect();

    if !thread_unsafe.is_empty() {
        docs.push_str("//! ## Thread Safety\n//!\n");
        docs.push_str("//! ⚠️  Some functions are NOT thread-safe:\n");
        for ctx in thread_unsafe.iter().take(5) {
            docs.push_str(&format!(
                "//! - `{}` - requires external synchronization\n",
                ctx.name
            ));
        }
        if thread_unsafe.len() > 5 {
            docs.push_str(&format!("//! - ... and {} more\n", thread_unsafe.len() - 5));
        }
        docs.push_str("//!\n");
    }

    // Add summary statistics
    docs.push_str("//! ## Library Statistics\n//!\n");
    docs.push_str(&format!("//! - Functions: {}\n", ffi_info.functions.len()));
    docs.push_str(&format!("//! - Types: {}\n", ffi_info.types.len()));
    docs.push_str(&format!("//! - Enums: {}\n", ffi_info.enums.len()));
    docs.push_str(&format!("//! - Constants: {}\n", ffi_info.constants.len()));
    docs.push_str("//!\n\n");

    docs
}

/// Enrich extern block with function documentation
fn enrich_foreign_mod(
    foreign_mod: &ItemForeignMod,
    ffi_info: &FfiInfo,
    enriched_context: &EnhancedContext,
) -> Result<String> {
    let mut output = String::new();

    // Extract the ABI and any attributes
    let abi = &foreign_mod.abi;

    // `quote!(#abi).to_string()` already renders the leading `extern` (eg `extern "C"`),
    // so avoid duplicating the keyword. Emit the ABI followed by the opening brace.
    output.push_str(&format!("{} {{\n", quote::quote!(#abi).to_string()));

    // Collect function names for alias generation
    let mut function_names = Vec::new();

    // Process each function in the extern block
    for item in &foreign_mod.items {
        if let syn::ForeignItem::Fn(func) = item {
            let func_name = func.sig.ident.to_string();
            function_names.push(func_name.clone());

            // Find matching FfiFunction and enriched context
            let ffi_func = ffi_info.functions.iter().find(|f| f.name == func_name);
            let context = enriched_context.functions.get(&func_name);

            // Generate enriched documentation
            if let Some(ctx) = context {
                output.push_str(&generate_function_docs(ctx, ffi_func));
            } else {
                // Minimal documentation if no context available
                output.push_str(&format!("    /// FFI binding for `{}`\n", func_name));
                output.push_str("    ///\n");
                output.push_str("    /// # Safety\n");
                output.push_str("    ///\n");
                output.push_str(
                    "    /// This function is unsafe. Refer to the original C documentation.\n",
                );
            }

            // Add the function declaration. `quote!(#func).to_string()` may include a
            // trailing semicolon already, so trim any existing trailing ';' to avoid
            // emitting double-semicolons in the output.
            let mut func_decl = quote::quote!(#func).to_string();
            func_decl = func_decl.trim_end().trim_end_matches(';').to_string();
            output.push_str("    ");
            output.push_str(&func_decl);
            output.push_str(";\n\n");
        }
    }

    output.push_str("}\n\n");

    // Generate Rust-style name aliases
    output.push_str(&generate_rust_style_aliases(
        &function_names,
        enriched_context,
    ));

    Ok(output)
}

/// Generate comprehensive documentation for a single FFI function
fn generate_function_docs(ctx: &FunctionContext, _ffi_func: Option<&FfiFunction>) -> String {
    let mut docs = String::new();

    // Main description
    docs.push_str(&format!(
        "    /// {}\n",
        ctx.description.as_deref().unwrap_or(&ctx.name)
    ));
    docs.push_str("    ///\n");

    // Safety documentation (always present for unsafe FFI)
    docs.push_str("    /// # Safety\n");
    docs.push_str("    ///\n");
    docs.push_str("    /// This function is unsafe because:\n");

    // Add specific safety concerns from context
    let mut safety_points = vec!["It directly calls C code with no runtime checks"];

    if let Some(precond) = &ctx.preconditions {
        if !precond.non_null_params.is_empty() {
            safety_points.push("Some pointer parameters must not be null");
        }
        if !precond.undefined_behavior.is_empty() {
            safety_points.push("Violating preconditions causes undefined behavior");
        }
    }

    if let Some(ownership) = &ctx.ownership {
        if ownership.return_ownership.requires_cleanup() {
            safety_points.push("Returns resources that must be manually freed");
        }
    }

    for point in safety_points {
        docs.push_str(&format!("    /// - {}\n", point));
    }
    docs.push_str("    ///\n");

    // Parameters documentation
    if !ctx.parameters.is_empty() {
        docs.push_str("    /// # Parameters\n");
        docs.push_str("    ///\n");
        for (param_name, param_doc) in &ctx.parameters {
            docs.push_str(&format!("    /// - `{}`: {}\n", param_name, param_doc));
        }
        docs.push_str("    ///\n");
    }

    // Add parameter constraints
    if let Some(precond) = &ctx.preconditions {
        if !precond.non_null_params.is_empty() || !precond.preconditions.is_empty() {
            docs.push_str("    /// # Preconditions\n");
            docs.push_str("    ///\n");
            for param in &precond.non_null_params {
                docs.push_str(&format!("    /// - `{}` must not be null\n", param));
            }
            for pc in precond.preconditions.iter().take(5) {
                docs.push_str(&format!("    /// - {}\n", pc.description));
            }
            docs.push_str("    ///\n");
        }
    }

    // Numeric constraints
    if let Some(constraints) = &ctx.numeric_constraints {
        if !constraints.constraints.is_empty() {
            docs.push_str("    /// # Constraints\n");
            docs.push_str("    ///\n");
            for constraint in constraints.constraints.iter().take(5) {
                if let Some(param) = &constraint.parameter_name {
                    if let Some(min) = constraint.min_value {
                        docs.push_str(&format!("    /// - `{}` >= {}\n", param, min));
                    }
                    if let Some(max) = constraint.max_value {
                        docs.push_str(&format!("    /// - `{}` <= {}\n", param, max));
                    }
                    if constraint.must_be_power_of_two {
                        docs.push_str(&format!("    /// - `{}` must be power of 2\n", param));
                    }
                }
            }
            docs.push_str("    ///\n");
        }
    }

    // Return value documentation
    if let Some(return_doc) = &ctx.return_doc {
        docs.push_str("    /// # Returns\n");
        docs.push_str("    ///\n");
        docs.push_str(&format!("    /// {}\n", return_doc));
        docs.push_str("    ///\n");
    }

    // Thread safety
    if let Some(thread_safety) = &ctx.thread_safety {
        docs.push_str("    /// # Thread Safety\n");
        docs.push_str("    ///\n");
        if thread_safety.trait_bounds.sync {
            docs.push_str("    /// ✓ Thread-safe - can be called from multiple threads\n");
        } else {
            docs.push_str("    /// ⚠️  NOT thread-safe - requires external synchronization\n");
        }
        if let Some(doc) = &thread_safety.documentation {
            docs.push_str(&format!("    /// {}\n", doc));
        }
        docs.push_str("    ///\n");
    }

    // Performance characteristics
    if let Some(perf) = &ctx.performance {
        docs.push_str("    /// # Performance\n");
        docs.push_str("    ///\n");
        docs.push_str(&format!("    /// Complexity: {:?}\n", perf.complexity));
        if !perf.warnings.is_empty() {
            for warning in perf.warnings.iter().take(3) {
                docs.push_str(&format!("    /// ⚠️  {}\n", warning));
            }
        }
        docs.push_str("    ///\n");
    }

    // API sequencing
    if let Some(api_seq) = &ctx.api_sequences {
        if !api_seq.prerequisites.is_empty() || !api_seq.requires_followup.is_empty() {
            docs.push_str("    /// # Call Order\n");
            docs.push_str("    ///\n");
            if !api_seq.prerequisites.is_empty() {
                docs.push_str("    /// Must call first:\n");
                for prereq in api_seq.prerequisites.iter().take(3) {
                    docs.push_str(&format!("    /// - `{}`\n", prereq));
                }
            }
            if !api_seq.requires_followup.is_empty() {
                docs.push_str("    /// Must call after:\n");
                for followup in api_seq.requires_followup.iter().take(3) {
                    docs.push_str(&format!("    /// - `{}`\n", followup));
                }
            }
            docs.push_str("    ///\n");
        }
    }

    // Common pitfalls
    if let Some(pitfalls) = &ctx.pitfalls {
        if !pitfalls.pitfalls.is_empty() {
            docs.push_str("    /// # Common Pitfalls\n");
            docs.push_str("    ///\n");
            for pitfall in pitfalls.pitfalls.iter().take(3) {
                docs.push_str(&format!(
                    "    /// ⚠️  **{}**: {}\n",
                    pitfall.title, pitfall.explanation
                ));
            }
            docs.push_str("    ///\n");
        }
    }

    // Examples from test cases
    if let Some(tests) = &ctx.test_cases {
        if !tests.examples.is_empty() {
            docs.push_str("    /// # Example\n");
            docs.push_str("    ///\n");
            docs.push_str("    /// ```c\n");
            // Show first example (simplified)
            for line in tests.examples[0].code_snippet.lines().take(10) {
                docs.push_str(&format!("    /// {}\n", line));
            }
            docs.push_str("    /// ```\n");
            docs.push_str("    ///\n");
        }
    }

    // Additional notes
    if !ctx.notes.is_empty() {
        docs.push_str("    /// # Notes\n");
        docs.push_str("    ///\n");
        for note in ctx.notes.iter().take(3) {
            docs.push_str(&format!("    /// - {}\n", note));
        }
        docs.push_str("    ///\n");
    }

    docs
}

/// Enrich type alias with documentation
fn enrich_type_alias(
    type_alias: &ItemType,
    _ffi_info: &FfiInfo,
    _enriched_context: &EnhancedContext,
) -> Result<String> {
    let mut output = String::new();

    let type_name = type_alias.ident.to_string();

    // Add basic documentation
    output.push_str(&format!("/// Type alias for `{}`\n", type_name));
    output.push_str("///\n");
    output.push_str("/// This is an opaque handle type from the C library.\n");

    // Output the type alias
    output.push_str(&quote::quote!(#type_alias).to_string());
    output.push('\n');

    Ok(output)
}

/// Enrich struct with documentation
fn enrich_struct(
    struct_item: &syn::ItemStruct,
    ffi_info: &FfiInfo,
    _enriched_context: &EnhancedContext,
) -> Result<String> {
    let mut output = String::new();

    let struct_name = struct_item.ident.to_string();

    // Find matching FfiType
    let ffi_type = ffi_info.types.iter().find(|t| t.name == struct_name);

    // Add documentation
    output.push_str(&format!("/// C struct `{}`\n", struct_name));
    if let Some(ffi_type) = ffi_type {
        if ffi_type.is_opaque {
            output.push_str("///\n");
            output.push_str("/// This is an opaque type - its fields are not exposed.\n");
            output.push_str("/// Use it only through pointers.\n");
        } else if !ffi_type.fields.is_empty() {
            output.push_str("///\n");
            output.push_str("/// # Fields\n");
            output.push_str("///\n");
            for field in &ffi_type.fields {
                output.push_str(&format!("/// - `{}`: {}\n", field.name, field.ty));
            }
        }
    }

    // Output the struct
    output.push_str(&quote::quote!(#struct_item).to_string());
    output.push('\n');

    Ok(output)
}

/// Enrich enum with comprehensive variant documentation
fn enrich_enum(
    enum_item: &syn::ItemEnum,
    ffi_info: &FfiInfo,
    _enriched_context: &EnhancedContext,
) -> Result<String> {
    let mut output = String::new();

    let enum_name = enum_item.ident.to_string();

    // Find matching FfiEnum
    let ffi_enum = ffi_info.enums.iter().find(|e| e.name == enum_name);

    // Add documentation
    output.push_str(&format!("/// C enum `{}`\n", enum_name));
    output.push_str("///\n");

    if let Some(ffi_enum) = ffi_enum {
        output.push_str("/// # Variants\n");
        output.push_str("///\n");
        for variant in &ffi_enum.variants {
            if let Some(value) = variant.value {
                output.push_str(&format!("/// - `{}` = {}\n", variant.name, value));
            } else {
                output.push_str(&format!("/// - `{}`\n", variant.name));
            }
        }
    }

    output.push_str("///\n");
    output.push_str("/// # Repr\n");
    output.push_str("///\n");
    output.push_str("/// `#[repr(C)]` - matches C ABI for FFI\n");

    // Output the enum
    output.push_str(&quote::quote!(#enum_item).to_string());
    output.push('\n');

    Ok(output)
}

/// Enrich const with documentation
fn enrich_const(const_item: &syn::ItemConst) -> Result<String> {
    let mut output = String::new();

    let const_name = const_item.ident.to_string();

    // Add documentation
    output.push_str(&format!("/// C constant `{}`\n", const_name));

    // Output the const
    output.push_str(&quote::quote!(#const_item).to_string());
    output.push('\n');

    Ok(output)
}

/// Generate Rust-style name aliases for C function names
///
/// Converts C naming conventions to idiomatic Rust:
/// - cudnnCreate → create_handle
/// - cudnnSetTensor4dDescriptor → set_tensor_4d_descriptor
/// - cv_Mat_new → mat_new
fn generate_rust_style_aliases(
    function_names: &[String],
    enriched_context: &EnhancedContext,
) -> String {
    let mut output = String::new();

    if function_names.is_empty() {
        return output;
    }

    output.push_str("// Rust-style name aliases for idiomatic usage\n");
    output.push_str("//\n");
    output.push_str("// These aliases provide snake_case names following Rust conventions\n");
    output.push_str("// while the original C-style names remain available for compatibility.\n\n");

    for func_name in function_names {
        let rust_name = c_name_to_rust_name(func_name);

        // Only create alias if the name is different (avoid duplicate like `foo` → `foo`)
        if rust_name != *func_name {
            // Add documentation for the alias
            output.push_str(&format!("/// Rust-style alias for [`{}`]\n", func_name));

            // Add context hint if available
            if let Some(ctx) = enriched_context.functions.get(func_name) {
                if let Some(desc) = &ctx.description {
                    // Take first line of description for alias
                    if let Some(first_line) = desc.lines().next() {
                        output.push_str(&format!("///\n/// {}\n", first_line));
                    }
                }
            }

            output.push_str(&format!("pub use {} as {};\n", func_name, rust_name));
        }
    }

    output.push('\n');
    output
}

/// Convert C-style function name to Rust snake_case
///
/// Examples:
/// - cudnnCreate → create
/// - cudnnSetTensor4dDescriptor → set_tensor_4d_descriptor
/// - cv_Mat_new → mat_new
/// - cuInit → init
fn c_name_to_rust_name(c_name: &str) -> String {
    // Common library prefixes to strip
    let prefixes = [
        "cudnn", "cuda", "cv_", "cu", "gl", "vk", "SDL_", "GLFW", "openssl_", "sqlite3_",
    ];

    let mut name = c_name.to_string();

    // Strip known prefixes (case-insensitive)
    for prefix in &prefixes {
        if name.to_lowercase().starts_with(&prefix.to_lowercase()) {
            name = name[prefix.len()..].to_string();
            break;
        }
    }

    // Handle empty name after prefix removal
    if name.is_empty() {
        return c_name.to_string();
    }

    // Convert to snake_case
    let mut result = String::new();
    let mut prev_was_lower = false;
    let mut prev_was_digit = false;

    for (i, ch) in name.chars().enumerate() {
        if ch == '_' {
            result.push('_');
            prev_was_lower = false;
            prev_was_digit = false;
        } else if ch.is_uppercase() {
            // Add underscore before uppercase if:
            // - Not the first character
            // - Previous was lowercase or digit
            // - Not part of an acronym (next char is lowercase)
            if i > 0 && (prev_was_lower || prev_was_digit) {
                result.push('_');
            }
            result.push(ch.to_lowercase().next().unwrap());
            prev_was_lower = false;
            prev_was_digit = false;
        } else if ch.is_ascii_digit() {
            // Add underscore before digit if previous was letter
            if i > 0 && !prev_was_digit && !result.ends_with('_') {
                // Check if this is part of a sequence like "4d"
                let next_is_lower = name.chars().nth(i + 1).map_or(false, |c| c.is_lowercase());
                if next_is_lower {
                    result.push('_');
                }
            }
            result.push(ch);
            prev_was_digit = true;
            prev_was_lower = false;
        } else {
            result.push(ch);
            prev_was_lower = true;
            prev_was_digit = false;
        }
    }

    // Clean up any double underscores
    while result.contains("__") {
        result = result.replace("__", "_");
    }

    // Remove leading/trailing underscores
    result = result.trim_matches('_').to_string();

    // If result is empty or very short, use original
    if result.is_empty() || result.len() < 2 {
        return c_name.to_string();
    }

    result
}
