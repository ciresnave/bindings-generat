//! Code generation for safe Rust wrappers.
//!
//! This module contains all code generation logic for transforming FFI bindings
//! into safe, idiomatic Rust APIs. It generates:
//!
//! - **RAII wrapper types** with automatic resource management
//! - **Type-safe error enums** with human-readable messages  
//! - **Safe method wrappers** that encapsulate unsafe FFI calls
//! - **Builder patterns** for complex constructors
//! - **Feature flags** for platform-specific APIs
//! - **Comprehensive documentation** including README and usage examples
//!
//! ## Architecture
//!
//! The generator follows a pipeline approach:
//!
//! 1. **Analysis** - Detect patterns in FFI (errors, RAII pairs, platforms)
//! 2. **Generation** - Transform patterns into Rust code
//! 3. **Enhancement** - Apply optimizations and add documentation
//! 4. **Output** - Write formatted code to files
//!
//! ## Key Modules
//!
//! - [`wrappers`] - RAII wrapper generation with Drop implementations
//! - [`errors`] - Error enum generation with Display/Error traits
//! - [`methods`] - Safe method wrappers for FFI functions
//! - [`builders`] - Builder pattern for complex constructors
//! - [`features`] - Platform-specific feature flag generation
//! - [`enums`] - Type-safe Rust enums from C enums
//! - [`tests`] - Test generation for generated code
//! - [`readme`] - README.md generation with examples

pub mod benchmarks;
pub mod builder_features;
pub mod builder_typestate;
pub mod builders;
pub mod cross_platform;
pub mod doc_generator;
pub mod enums;
pub mod ergonomics;
pub mod errors;
pub mod features;
pub mod functional_tests;
pub mod methods;
pub mod readme;
pub mod runtime_tests;
pub mod templates_minimal;
pub use templates_minimal as templates;
pub mod tests;
pub mod wrappers;

// Re-export key types for easier access
pub use builder_features::{BuilderConfig, BuilderGenerator, BuilderPreset, FieldValidation};
pub use builder_typestate::{BuilderTypestateAnalysis, BuilderTypestateGenerator};
pub use ergonomics::{ExtensionTraitGenerator, IteratorAdapterGenerator, OperatorOverloadAnalyzer};

use crate::analyzer::AnalysisResult;
use crate::ffi::FfiInfo;
use crate::llm::CodeEnhancements;
use anyhow::Result;
use std::fmt::Write;

/// Generated code output
#[derive(Debug, Clone)]
pub struct GeneratedCode {
    pub lib_rs: String,
    pub ffi_bindings: String,
    pub tests: String,
    pub runtime_tests: String,
    pub functional_tests: String,
    /// Runtime loader source (src/loader.rs) for Mode 3 dynamic loading
    pub loader_rs: String,
    /// Shared discovery module (src/discovery_shared.rs) used by build.rs and runtime
    pub discovery_shared_rs: String,
    /// Runtime install helper (src/discovery_install.rs) - stub by default, may be populated by writer
    pub discovery_install_rs: String,
    /// Dynamic FFI wrapper (src/ffi.rs) that lazy-loads symbols at runtime and re-exports types
    pub ffi_dynamic_rs: String,
}

/// Legacy function name for compatibility with tools
pub fn generate_safe_wrappers(
    ffi_info: &FfiInfo,
    analysis: &AnalysisResult,
    _code_style: &crate::config::CodeStyle,
) -> Result<String> {
    let generated = generate_code(ffi_info, analysis, "library", None)?;
    Ok(generated.lib_rs)
}

/// Generate safe Rust wrapper code from analysis
pub fn generate_code(
    ffi_info: &FfiInfo,
    analysis: &AnalysisResult,
    lib_name: &str,
    llm_enhancements: Option<&CodeEnhancements>,
) -> Result<GeneratedCode> {
    let mut lib_rs = String::new();

    // Generate header (includes FFI module declaration)
    generate_header(&mut lib_rs, lib_name);

    // Collect all FFI types used in generated code
    let ffi_types = collect_ffi_types(ffi_info, analysis);

    // Generate imports for FFI types
    generate_ffi_imports(&mut lib_rs, &ffi_types);

    // Always generate error enum (even if no error pattern detected)
    if let Some(error_enum) = analysis.error_patterns.error_enums.first() {
        lib_rs.push_str(&errors::generate_error_enum_with_smart_analysis(
            error_enum,
            llm_enhancements,
            analysis,
            analysis.smart_errors.as_ref(),
        ));
    } else {
        // Generate a basic error type
        lib_rs.push_str(&errors::generate_basic_error());
    }

    // Generate type-safe enum wrappers for all enums (except error enums)
    for ffi_enum in &ffi_info.enums {
        // Skip error enums - they're handled separately
        let is_error_enum = analysis
            .error_patterns
            .error_enums
            .iter()
            .any(|e| e.name == ffi_enum.name);

        if !is_error_enum {
            lib_rs.push_str(&enums::generate_safe_enum(ffi_enum));
        }
    }

    // Generate RAII wrappers for handle types with lifecycle pairs
    // Note: Multiple lifecycle pairs may exist for the same handle type (e.g., multiple create functions)
    // We only generate ONE wrapper per handle type, using the first/best lifecycle pair
    let mut generated_handles = std::collections::HashSet::new();
    // Track builder types we've emitted to avoid duplicates when multiple wrappers
    // could produce identical builder definitions. The wrappers return any
    // generated builder code and its canonical name so we can dedupe centrally.
    let mut emitted_builders = std::collections::HashSet::new();

    for pair in &analysis.raii_patterns.lifecycle_pairs {
        // Skip if we already generated a wrapper for this handle type
        if generated_handles.contains(&pair.handle_type) {
            continue;
        }

        if let Some(handle) = analysis
            .raii_patterns
            .handle_types
            .iter()
            .find(|h| h.name == pair.handle_type)
        {
            // Find the actual FFI functions for create and destroy
            let create_func = ffi_info.functions.iter().find(|f| f.name == pair.create_fn);
            let destroy_func = ffi_info
                .functions
                .iter()
                .find(|f| f.name == pair.destroy_fn);

            let wrapper = wrappers::generate_raii_wrapper(
                handle,
                pair,
                create_func,
                destroy_func,
                &analysis.error_patterns,
                lib_name,
                analysis,
                analysis.parameter_analysis.as_ref(),
            );
            lib_rs.push_str(&wrapper.code);
            // If the wrapper returned builder code, append it once across all wrappers
            if let Some(bc) = wrapper.builder_code.as_ref() {
                // Determine canonical builder name
                let bname = wrapper
                    .builder_name
                    .as_ref()
                    .cloned()
                    .unwrap_or_else(|| format!("{}Builder", wrapper.type_name));
                if !emitted_builders.contains(&bname) {
                    lib_rs.push_str(&bc);
                    lib_rs.push('\n');
                    emitted_builders.insert(bname);
                }
            }
            generated_handles.insert(pair.handle_type.clone());
        }
    }

    // Generate basic wrapper types for handle types WITHOUT lifecycle pairs
    // These don't have Drop implementation, but provide a safe container
    for handle in &analysis.raii_patterns.handle_types {
        // Skip if already generated with lifecycle pair
        if analysis
            .raii_patterns
            .lifecycle_pairs
            .iter()
            .any(|p| p.handle_type == handle.name)
        {
            continue;
        }

        let wrapper = wrappers::generate_basic_wrapper(handle);
        lib_rs.push_str(&wrapper.code);
    }

    // Generate methods for each wrapper
    for handle in &analysis.raii_patterns.handle_types {
        generate_methods_for_handle(&mut lib_rs, handle, ffi_info, analysis)?;
    }

    // Generate builder typestates if available
    if let Some(builder_analysis) = &analysis.builder_typestates {
        let generator = builder_typestate::BuilderTypestateGenerator::new();
        for builder in &builder_analysis.builders {
            let builder_code = generator.generate_builder_code(builder);
            lib_rs.push_str(&builder_code);
            lib_rs.push('\n');
        }
    }

    // Add platform detection utilities
    lib_rs.push_str("\n");
    lib_rs.push_str(&cross_platform::generate_platform_utils());

    // Generate tests for the bindings
    let wrapper_types: Vec<_> = analysis
        .raii_patterns
        .handle_types
        .iter()
        .map(|h| {
            let wrapper_name = wrappers::to_rust_type_name(&h.name);
            (h.name.clone(), wrapper_name)
        })
        .collect();

    let tests = tests::generate_tests(
        &ffi_info.functions,
        &ffi_info.types,
        &wrapper_types,
        lib_name,
        Some(analysis),
    );

    let runtime_tests = runtime_tests::generate_runtime_tests(
        &ffi_info.functions,
        &ffi_info.types,
        &wrapper_types,
        lib_name,
    );

    // Generate functional tests with examples from enhanced documentation
    let examples = analysis
        .enhanced_docs
        .as_ref()
        .map(|docs| docs.examples.as_slice())
        .unwrap_or(&[]);

    let functional_tests =
        functional_tests::generate_functional_tests(&ffi_info.functions, examples, lib_name);

    // Generate runtime loader template (Mode 3) — always generated by default
    let loader_rs = templates::runtime_loader(lib_name);
    // Generate shared discovery module (std-only) for reuse by build.rs and runtime loader
    let discovery_shared_rs = templates::discovery_shared(lib_name);
    // Default stub for install helpers - writer may replace this with a full installer if sources exist
    let discovery_install_rs = templates::discovery_install_stub(lib_name);
    // Generate dynamic FFI wrapper that lazily loads symbols and re-exports types from bindgen output
    let ffi_dynamic_rs = templates::ffi_dynamic(lib_name, ffi_info);

    Ok(GeneratedCode {
        lib_rs,
        ffi_bindings: String::new(), // Will be populated with bindgen output
        tests,
        runtime_tests,
        functional_tests,
        loader_rs,
        discovery_shared_rs,
        discovery_install_rs,
        ffi_dynamic_rs,
    })
}

fn generate_header(code: &mut String, lib_name: &str) {
    // Generate comprehensive crate-level documentation
    writeln!(code, "//! # {}", lib_name).unwrap();
    writeln!(code, "//!").unwrap();
    writeln!(code, "//! Safe Rust wrapper for {} C library.", lib_name).unwrap();
    writeln!(code, "//!").unwrap();
    writeln!(
        code,
        "//! This crate was automatically generated by [bindings-generat](https://github.com/your-repo/bindings-generat)."
    )
    .unwrap();
    writeln!(
        code,
        "//! It provides safe, idiomatic Rust wrappers around the raw FFI bindings,"
    )
    .unwrap();
    writeln!(
        code,
        "//! handling resource management, error conversion, and type safety."
    )
    .unwrap();
    writeln!(code, "//!").unwrap();
    writeln!(code, "//! ## Features").unwrap();
    writeln!(code, "//!").unwrap();
    writeln!(
        code,
        "//! - **RAII Resource Management**: Automatically manages handle lifecycles"
    )
    .unwrap();
    writeln!(
        code,
        "//!   with proper Drop implementations for leak-free usage"
    )
    .unwrap();
    writeln!(
        code,
        "//! - **Type-Safe Error Handling**: Converts C error codes to Rust Result types"
    )
    .unwrap();
    writeln!(code, "//!   with descriptive error messages").unwrap();
    writeln!(
        code,
        "//! - **Idiomatic Rust API**: Methods on wrapper types instead of raw C functions"
    )
    .unwrap();
    writeln!(
        code,
        "//! - **Comprehensive Safety**: All unsafe code is carefully encapsulated"
    )
    .unwrap();
    writeln!(code, "//!   with safe interfaces").unwrap();
    writeln!(code, "//!").unwrap();
    writeln!(code, "//! ## Usage").unwrap();
    writeln!(code, "//!").unwrap();
    writeln!(code, "//! ```ignore").unwrap();
    writeln!(code, "//! use {}::*;", lib_name).unwrap();
    writeln!(code, "//!").unwrap();
    writeln!(code, "//! fn main() -> Result<(), Error> {{").unwrap();
    writeln!(
        code,
        "//!     // Create a handle (specific constructor depends on library)"
    )
    .unwrap();
    writeln!(code, "//!     // let handle = Handle::new()?;").unwrap();
    writeln!(code, "//!     ").unwrap();
    writeln!(code, "//!     // Use methods on the handle").unwrap();
    writeln!(code, "//!     // handle.some_operation()?;").unwrap();
    writeln!(code, "//!     ").unwrap();
    writeln!(
        code,
        "//!     // Handle is automatically cleaned up when dropped"
    )
    .unwrap();
    writeln!(code, "//!     Ok(())").unwrap();
    writeln!(code, "//! }}").unwrap();
    writeln!(code, "//! ```").unwrap();
    writeln!(code, "//!").unwrap();
    writeln!(code, "//! ## Error Handling").unwrap();
    writeln!(code, "//!").unwrap();
    writeln!(
        code,
        "//! All fallible operations return `Result<T, Error>`. The `Error` type"
    )
    .unwrap();
    writeln!(
        code,
        "//! implements `std::error::Error` and provides human-readable error messages."
    )
    .unwrap();
    writeln!(code, "//!").unwrap();
    writeln!(code, "//! ```ignore").unwrap();
    writeln!(code, "//! match handle.operation() {{").unwrap();
    writeln!(code, "//!     Ok(result) => {{ /* success */ }},").unwrap();
    writeln!(
        code,
        "//!     Err(Error::NotInitialized) => {{ /* handle specific error */ }},"
    )
    .unwrap();
    writeln!(code, "//!     Err(e) => {{ /* handle other errors */ }},").unwrap();
    writeln!(code, "//! }}").unwrap();
    writeln!(code, "//! ```").unwrap();
    writeln!(code, "//!").unwrap();
    writeln!(code, "//! ## Safety").unwrap();
    writeln!(code, "//!").unwrap();
    writeln!(
        code,
        "//! This crate encapsulates all `unsafe` FFI calls behind safe Rust interfaces."
    )
    .unwrap();
    writeln!(
        code,
        "//! Resource management is handled automatically through RAII patterns:"
    )
    .unwrap();
    writeln!(code, "//!").unwrap();
    writeln!(code, "//! - Handles are created with safe constructors").unwrap();
    writeln!(
        code,
        "//! - Resources are automatically freed when handles are dropped"
    )
    .unwrap();
    writeln!(code, "//! - Null pointer checks prevent undefined behavior").unwrap();
    writeln!(code, "//! - Error codes are converted to Rust Result types").unwrap();
    writeln!(code, "//!").unwrap();
    writeln!(code, "//! ## Thread Safety").unwrap();
    writeln!(code, "//!").unwrap();
    writeln!(
        code,
        "//! Thread safety depends on the underlying C library. Check the"
    )
    .unwrap();
    writeln!(
        code,
        "//! original library documentation for threading requirements and restrictions."
    )
    .unwrap();
    writeln!(
        code,
        "//! Most handles are `!Send` and `!Sync` by default for safety."
    )
    .unwrap();
    writeln!(code, "//!").unwrap();
    writeln!(code, "//! ## Performance").unwrap();
    writeln!(code, "//!").unwrap();
    writeln!(code, "//! The wrapper layer has minimal overhead:").unwrap();
    writeln!(code, "//!").unwrap();
    writeln!(
        code,
        "//! - Wrapper types are zero-cost abstractions (transparent wrappers)"
    )
    .unwrap();
    writeln!(
        code,
        "//! - Methods are marked `#[inline]` for optimization"
    )
    .unwrap();
    writeln!(code, "//! - No runtime overhead beyond error code checks").unwrap();
    writeln!(code, "//!").unwrap();
    writeln!(code, "//! ## Documentation").unwrap();
    writeln!(code, "//!").unwrap();
    writeln!(
        code,
        "//! For detailed documentation about the underlying C library, please refer"
    )
    .unwrap();
    writeln!(code, "//! to the official {} documentation.", lib_name).unwrap();
    writeln!(code, "//!").unwrap();
    writeln!(code, "//! ## Generated Code").unwrap();
    writeln!(code, "//!").unwrap();
    writeln!(
        code,
        "//! This crate is automatically generated. To customize the bindings:"
    )
    .unwrap();
    writeln!(code, "//!").unwrap();
    writeln!(code, "//! 1. Modify the source headers or configuration").unwrap();
    writeln!(code, "//! 2. Re-run bindings-generat").unwrap();
    writeln!(code, "//! 3. Review the generated code for correctness").unwrap();
    writeln!(code, "//!").unwrap();
    writeln!(code, "//! ## License").unwrap();
    writeln!(code, "//!").unwrap();
    writeln!(
        code,
        "//! The generated bindings follow the same license as the original {} library.",
        lib_name
    )
    .unwrap();
    writeln!(
        code,
        "//! Please check the library's license before using these bindings."
    )
    .unwrap();
    writeln!(code).unwrap();

    // Compiler directives
    writeln!(code, "#![allow(dead_code)]").unwrap();
    writeln!(code, "#![allow(non_camel_case_types)]").unwrap();
    writeln!(code, "#![allow(non_snake_case)]").unwrap();
    writeln!(code, "#![allow(non_upper_case_globals)]").unwrap();
    writeln!(code).unwrap();

    // FFI module declaration
    writeln!(code, "// Note: FFI bindings should be in src/ffi.rs").unwrap();
    writeln!(
        code,
        "// Run bindgen on your headers and place the output there"
    )
    .unwrap();
    writeln!(code, "#[path = \"ffi.rs\"]").unwrap();
    writeln!(code, "mod ffi;").unwrap();
    writeln!(code).unwrap();
}

/// Collect all FFI types used in generated wrapper code
fn collect_ffi_types(
    ffi_info: &FfiInfo,
    analysis: &AnalysisResult,
) -> std::collections::HashSet<String> {
    use std::collections::HashSet;

    let mut types = HashSet::new();

    // Collect types from functions that operate on handles
    for handle in &analysis.raii_patterns.handle_types {
        let handle_functions: Vec<_> = ffi_info
            .functions
            .iter()
            .filter(|f| f.params.iter().any(|p| p.ty.contains(&handle.name)))
            .collect();

        for func in handle_functions {
            // Add return type if it's an FFI type
            if !func.return_type.is_empty()
                && func.return_type != "()"
                && func.return_type != "c_void"
                && !is_rust_primitive(&func.return_type)
            {
                types.insert(func.return_type.clone());
            }

            // Add parameter types
            for param in &func.params {
                // Extract base type from pointers
                let base_type = param
                    .ty
                    .replace("*const ", "")
                    .replace("*mut ", "")
                    .replace("* const ", "")
                    .replace("* mut ", "")
                    .trim()
                    .to_string();

                if !base_type.is_empty()
                    && !is_rust_primitive(&base_type)
                    && !base_type.contains("::")
                {
                    types.insert(base_type);
                }
            }
        }
    }

    types
}

/// Check if a type is a Rust primitive
fn is_rust_primitive(ty: &str) -> bool {
    matches!(
        ty,
        "i8" | "i16"
            | "i32"
            | "i64"
            | "i128"
            | "isize"
            | "u8"
            | "u16"
            | "u32"
            | "u64"
            | "u128"
            | "usize"
            | "f32"
            | "f64"
            | "bool"
            | "char"
            | "()"
            | "c_void"
            | "c_char"
            | "c_int"
            | "c_uint"
            | "c_long"
            | "c_ulong"
            | "c_short"
            | "c_ushort"
            | "c_longlong"
            | "c_ulonglong"
            | "c_float"
            | "c_double"
    )
}

/// Generate use statements for FFI types
fn generate_ffi_imports(code: &mut String, types: &std::collections::HashSet<String>) {
    // Avoid importing individual FFI type names into the crate root to
    // prevent name collisions with generated wrapper types. The FFI
    // bindings are available under the `ffi` module (declared by
    // `mod ffi;`), so reference them as `ffi::TypeName` from generated
    // code instead of importing into the root namespace.
    if types.is_empty() {
        return;
    }

    writeln!(
        code,
        "// FFI types are available via the `ffi` module (use `ffi::TypeName`)"
    )
    .unwrap();
    // Do NOT add a `use crate::ffi as ffi;` here - the crate already declares
    // `mod ffi;` at the top level and introducing a duplicate symbol named
    // `ffi` causes name collisions in the generated crate. Refer to types as
    // `ffi::TypeName` directly from generated code.
    writeln!(code).unwrap();
}

fn generate_methods_for_handle(
    code: &mut String,
    handle: &crate::analyzer::raii::HandleType,
    ffi_info: &FfiInfo,
    analysis: &AnalysisResult,
) -> Result<()> {
    // Find functions that operate on this handle
    let handle_functions: Vec<_> = ffi_info
        .functions
        .iter()
        .filter(|f| {
            // Check if function takes this handle as parameter
            f.params.iter().any(|p| p.ty.contains(&handle.name))
        })
        .collect();

    if handle_functions.is_empty() {
        return Ok(());
    }

    // Convert handle type name to Rust type name (PascalCase)
    let type_name = wrappers::to_rust_type_name(&handle.name);

    // Collect methods first
    let mut methods = Vec::new();
    for func in handle_functions {
        // Skip create/destroy functions (already handled)
        if handle.create_functions.contains(&func.name)
            || handle.destroy_functions.contains(&func.name)
        {
            continue;
        }

        if let Some(method_code) = methods::generate_safe_method(
            func,
            Some(&handle.name),
            analysis.function_contexts.get(&func.name),
            analysis.enhanced_docs.as_ref(),
        ) {
            methods.push(method_code);
        }
    }

    // Only generate impl block if there are methods
    if !methods.is_empty() {
        writeln!(code, "// Additional methods for {}", handle.name).unwrap();
        writeln!(code, "impl {} {{", type_name).unwrap();
        for method_code in methods {
            code.push_str(&method_code);
        }
        writeln!(code, "}}").unwrap();
        writeln!(code).unwrap();
    }

    Ok(())
}
