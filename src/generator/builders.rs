//! Builder pattern generation for complex constructors
//!
//! This module generates builder patterns for wrapper types that have
//! constructors with multiple parameters, providing a fluent, type-safe
//! API for object construction.

use crate::enrichment::context::FunctionContext;
use crate::ffi::FfiFunction;
use std::fmt::Write;
use tracing::debug;

/// Generates a builder pattern for a constructor function.
///
/// When `context` is provided, integrates comprehensive analyzer data:
/// - Numeric constraint validation (ranges, alignment, power-of-two)
/// - Precondition validation (non-null, non-zero, state requirements)
/// - Semantic grouping for related parameters
/// - API sequences and global state warnings
/// - Performance characteristics documentation
///
/// When `context` is None, generates a basic builder with standard validation.
///
/// Returns `None` if the function is not suitable for a builder pattern
/// (e.g., has fewer than 2 parameters).
pub fn generate_builder(
    wrapper_name: &str,
    create_func: &FfiFunction,
    handle_type: &str,
    context: Option<&FunctionContext>,
) -> Option<String> {
    let params = &create_func.params;

    // Only generate builders for functions with multiple parameters
    if params.len() < 2 {
        debug!(
            "Skipping builder for {}: only {} parameters",
            create_func.name,
            params.len()
        );
        return None;
    }

    debug!(
        "Generating builder for {} with {} parameters (enriched: {})",
        wrapper_name,
        params.len(),
        context.is_some()
    );

    let mut output = String::new();
    // Use the canonical wrapper type name helper from `wrappers` to ensure
    // consistent PascalCase naming across generator modules.
    let wrapper_ident = crate::generator::wrappers::to_rust_type_name(wrapper_name);
    let builder_name = format!("{}Builder", wrapper_ident);

    // Remove typestate marker emission: builder struct is not generic over state

    // Check if first param is output parameter
    let has_output_param = create_func
        .params
        .first()
        .map(|p| {
            let is_pointer_to_handle = p.ty.contains("*mut") && p.ty.contains(handle_type);
            let is_pointer_param = p.is_pointer && p.is_mut;
            is_pointer_to_handle || is_pointer_param
        })
        .unwrap_or(false);

    // Get the actual parameters (skip output param if present)
    let builder_params: Vec<_> = if has_output_param {
        params.iter().skip(1).collect()
    } else {
        params.iter().collect()
    };

    if builder_params.is_empty() {
        return None;
    }

    // Emit builder struct/impl for every invocation (no deduplication)
    generate_builder_struct(
        &mut output,
        &builder_name,
        &wrapper_ident,
        &builder_params,
        context,
    )
    .ok()?;
    generate_builder_impl(
        &mut output,
        &builder_name,
        &wrapper_ident,
        &builder_params,
        create_func,
        has_output_param,
        handle_type,
        context,
    )
    .ok()?;

    // Generate Default impl
    writeln!(&mut output).ok()?;
    writeln!(&mut output, "impl Default for {} {{", builder_name).ok()?;
    writeln!(&mut output, "    fn default() -> Self {{").ok()?;
    writeln!(&mut output, "        Self::new()").ok()?;
    writeln!(&mut output, "    }}").ok()?;
    writeln!(&mut output, "}}").ok()?;

    // Add convenience method to main type
    writeln!(&mut output).ok()?;
    writeln!(&mut output, "impl {} {{", wrapper_ident).ok()?;
    writeln!(&mut output, "    /// Creates a new builder for this type.").ok()?;

    // Add performance note if available
    if let Some(ctx) = context {
        if let Some(perf) = &ctx.performance {
            if perf.complexity != crate::analyzer::performance::ComplexityClass::Constant
                || !perf.warnings.is_empty()
            {
                writeln!(&mut output, "    ///").ok()?;
                writeln!(&mut output, "    /// # Performance").ok()?;
                writeln!(&mut output, "    /// Complexity: {:?}", perf.complexity).ok()?;
                if !perf.warnings.is_empty() {
                    writeln!(&mut output, "    /// Warnings:").ok()?;
                    for warning in &perf.warnings {
                        writeln!(&mut output, "    /// - {}", warning).ok()?;
                    }
                }
            }
        }
    }

    writeln!(&mut output, "    #[inline]").ok()?;
    writeln!(&mut output, "    pub fn builder() -> {} {{", builder_name).ok()?;
    writeln!(&mut output, "        {}::new()", builder_name).ok()?;
    writeln!(&mut output, "    }}").ok()?;
    writeln!(&mut output, "}}").ok()?;

    Some(output)
}

/// Generate the builder struct with documentation
fn generate_builder_struct(
    output: &mut String,
    builder_name: &str,
    wrapper_name: &str,
    params: &[&crate::ffi::FfiParam],
    context: Option<&FunctionContext>,
) -> Result<(), std::fmt::Error> {
    writeln!(output, "/// Builder for `{}`", wrapper_name)?;
    writeln!(output, "///")?;
    writeln!(
        output,
        "/// Provides a fluent API for constructing {} instances with",
        wrapper_name
    )?;
    writeln!(output, "/// multiple configuration options.")?;

    // Add semantic grouping if available
    if let Some(ctx) = context {
        if let Some(semantic) = &ctx.semantic_group {
            writeln!(output, "///")?;
            if let Some(module) = &semantic.module {
                writeln!(output, "/// Module: {}", module)?;
            }
            if let Some(feature_set) = &semantic.feature_set {
                writeln!(output, "/// Features: {}", feature_set)?;
            }
        }

        // Add API sequence warnings
        if let Some(api_seq) = &ctx.api_sequences {
            if !api_seq.prerequisites.is_empty() || !api_seq.requires_followup.is_empty() {
                writeln!(output, "///")?;
                writeln!(output, "/// # API Sequencing")?;
                if !api_seq.prerequisites.is_empty() {
                    writeln!(
                        output,
                        "/// Must call first: {}",
                        api_seq.prerequisites.join(", ")
                    )?;
                }
                if !api_seq.requires_followup.is_empty() {
                    writeln!(
                        output,
                        "/// Must call after: {}",
                        api_seq.requires_followup.join(", ")
                    )?;
                }
            }
        }

        // Add global state warnings
        if let Some(global) = &ctx.global_state {
            if !global.states.is_empty() || global.requires_init {
                writeln!(output, "///")?;
                writeln!(output, "/// # Global State")?;
                for state_info in &global.states {
                    let desc = &state_info.description;
                    if let Some(name) = &state_info.name {
                        writeln!(output, "/// ⚠️ {}: {}", name, desc)?;
                    } else {
                        writeln!(output, "/// ⚠️ {}", desc)?;
                    }
                }
                if global.requires_init {
                    if let Some(init_fn) = &global.init_function {
                        writeln!(
                            output,
                            "/// Requires initialization: call `{}` first",
                            init_fn
                        )?;
                    }
                }
            }
        }
    }

    writeln!(output, "#[derive(Debug)]")?;
    writeln!(output, "pub struct {} {{", builder_name)?;

    for param in params {
        let field_name = to_rust_field_name(&param.name);
        // Keep pointer tokens intact - to_rust_type returns pointer types as-is.
        let mut field_type = to_rust_type(&param.ty);
        // Normalize c_void occurrences to the canonical Rust path while preserving
        // any pointer qualifiers (e.g. `*mut *mut ::core::ffi::c_void`).
        if field_type.contains("c_void") {
            // If this is a pointer form (contains `*`), replace the base identifier
            // but keep the pointer qualifiers intact.
            if field_type.contains('*') {
                field_type = field_type.replace("c_void", "::core::ffi::c_void");
            } else {
                // Bare `c_void` -> prefer an opaque pointer
                field_type = "*mut ::core::ffi::c_void".to_string();
            }
        }
        writeln!(output, "    {}: Option<{}>,", field_name, field_type)?;
    }

    writeln!(output, "}}")?;
    writeln!(output)?;

    Ok(())
}

/// Generate the builder implementation
fn generate_builder_impl(
    output: &mut String,
    builder_name: &str,
    wrapper_name: &str,
    params: &[&crate::ffi::FfiParam],
    create_func: &FfiFunction,
    has_output_param: bool,
    _handle_type: &str,
    context: Option<&FunctionContext>,
) -> Result<(), std::fmt::Error> {
    writeln!(output, "impl {} {{", builder_name)?;
    writeln!(output, "    use super::Error;")?;

    // Generate new() method
    writeln!(output, "    /// Create a new builder")?;
    writeln!(output, "    #[inline]")?;
    writeln!(output, "    pub fn new() -> Self {{")?;
    writeln!(output, "        Self {{")?;
    for param in params {
        let field_name = to_rust_field_name(&param.name);
        writeln!(output, "            {}: None,", field_name)?;
    }
    writeln!(output, "        }}")?;
    writeln!(output, "    }}")?;
    writeln!(output)?;

    // Generate setter methods for each parameter
    for param in params {
        generate_setter_method(output, param, context)?;
    }

    // Generate build() method
    generate_build_method(
        output,
        wrapper_name,
        params,
        create_func,
        has_output_param,
        context,
    )?;

    writeln!(output, "}}")?;

    Ok(())
}

/// Generate a setter method for a parameter
fn generate_setter_method(
    output: &mut String,
    param: &crate::ffi::FfiParam,
    context: Option<&FunctionContext>,
) -> Result<(), std::fmt::Error> {
    let field_name = to_rust_field_name(&param.name);
    let field_type = to_rust_type(&param.ty);
    let method_name = field_name.clone();

    writeln!(output, "    /// Set {}", field_name)?;

    // Add numeric constraints if available
    if let Some(ctx) = context {
        if let Some(constraints) = &ctx.numeric_constraints {
            for constraint in &constraints.constraints {
                if let Some(param_name) = &constraint.parameter_name {
                    if param_name == &param.name {
                        writeln!(output, "    ///")?;
                        writeln!(output, "    /// # Constraints")?;

                        if let Some(min) = constraint.min_value {
                            writeln!(output, "    /// - Minimum value: {}", min)?;
                        }
                        if let Some(max) = constraint.max_value {
                            writeln!(output, "    /// - Maximum value: {}", max)?;
                        }
                        if constraint.must_be_power_of_two {
                            writeln!(output, "    /// - Must be a power of two")?;
                        }
                        if let Some(align) = constraint.alignment_bytes {
                            writeln!(output, "    /// - Must be aligned to {} bytes", align)?;
                        }
                        if constraint.constraint_type
                            == crate::analyzer::numeric_constraints::ConstraintType::Positive
                        {
                            writeln!(output, "    /// - Must be positive (> 0)")?;
                        }
                        if constraint.constraint_type
                            == crate::analyzer::numeric_constraints::ConstraintType::NonZero
                        {
                            writeln!(output, "    /// - Must be non-zero")?;
                        }
                        if !constraint.description.is_empty() {
                            writeln!(output, "    /// - {}", constraint.description)?;
                        }
                    }
                }
            }
        }

        // Add preconditions for this parameter
        if let Some(preconditions) = &ctx.preconditions {
            let param_preconditions: Vec<_> = preconditions
                .preconditions
                .iter()
                .filter(|p| p.parameter.as_ref() == Some(&param.name))
                .collect();

            if !param_preconditions.is_empty() {
                writeln!(output, "    ///")?;
                writeln!(output, "    /// # Requirements")?;
                for precond in param_preconditions {
                    writeln!(output, "    /// - {}", precond.description)?;
                }
            }
        }
    }

    writeln!(output, "    #[inline]")?;
    writeln!(
        output,
        "    pub fn {}(mut self, value: {}) -> Self {{",
        method_name, field_type
    )?;

    // Add null pointer checks for pointer parameters
    if field_type.contains("*const") || field_type.contains("*mut") {
        writeln!(output, "        // Safety check for null pointer")?;
        writeln!(
            output,
            "        assert!(!value.is_null(), \"{} cannot be null\");",
            param.name
        )?;
    }

    // Add runtime validation from numeric_constraints
    if let Some(ctx) = context {
        if let Some(constraints) = &ctx.numeric_constraints {
            for constraint in &constraints.constraints {
                if let Some(param_name) = &constraint.parameter_name {
                    if param_name == &param.name {
                        // Generate validation code based on constraint type
                        if let (Some(min), Some(max)) = (constraint.min_value, constraint.max_value)
                        {
                            writeln!(
                                output,
                                "        assert!(value as i64 >= {} && value as i64 <= {}, \"{} must be between {} and {}\");",
                                min, max, param.name, min, max
                            )?;
                        } else if let Some(min) = constraint.min_value {
                            writeln!(
                                output,
                                "        assert!(value as i64 >= {}, \"{} must be >= {}\");",
                                min, param.name, min
                            )?;
                        } else if let Some(max) = constraint.max_value {
                            writeln!(
                                output,
                                "        assert!(value as i64 <= {}, \"{} must be <= {}\");",
                                max, param.name, max
                            )?;
                        }

                        if constraint.must_be_power_of_two {
                            writeln!(
                                output,
                                "        assert!(value.is_power_of_two(), \"{} must be a power of two\");",
                                param.name
                            )?;
                        }

                        if let Some(align) = constraint.alignment_bytes {
                            writeln!(
                                output,
                                "        assert!(value as usize % {} == 0, \"{} must be aligned to {} bytes\");",
                                align, param.name, align
                            )?;
                        }

                        if constraint.constraint_type
                            == crate::analyzer::numeric_constraints::ConstraintType::Positive
                        {
                            writeln!(
                                output,
                                "        assert!(value as i64 > 0, \"{} must be positive (> 0)\");",
                                param.name
                            )?;
                        } else if constraint.constraint_type
                            == crate::analyzer::numeric_constraints::ConstraintType::NonZero
                        {
                            writeln!(
                                output,
                                "        assert!(value as i64 != 0, \"{} must be non-zero\");",
                                param.name
                            )?;
                        }
                    }
                }
            }
        }

        // Add precondition validation
        if let Some(preconditions) = &ctx.preconditions {
            // Check for custom constraints that might indicate non-zero requirements
            let param_has_nonzero = preconditions.preconditions.iter().any(|p| {
                p.parameter.as_ref() == Some(&param.name)
                    && p.description.to_lowercase().contains("non-zero")
            });

            if param_has_nonzero {
                writeln!(
                    output,
                    "        assert!(value as i64 != 0, \"{} must not be zero\");",
                    param.name
                )?;
            }
        }
    }

    writeln!(output, "        self.{} = Some(value);", field_name)?;
    writeln!(output, "        self")?;
    writeln!(output, "    }}")?;
    writeln!(output)?;

    Ok(())
}

/// Generate the build() method
fn generate_build_method(
    output: &mut String,
    wrapper_name: &str,
    params: &[&crate::ffi::FfiParam],
    create_func: &FfiFunction,
    has_output_param: bool,
    context: Option<&FunctionContext>,
) -> Result<(), std::fmt::Error> {
    writeln!(output, "    /// Build the {} instance", wrapper_name)?;
    writeln!(output, "    ///")?;
    writeln!(output, "    /// # Errors")?;
    writeln!(output, "    ///")?;
    writeln!(
        output,
        "    /// Returns an error if any parameters are not set"
    )?;
    writeln!(output, "    /// or if the FFI call fails.")?;

    // Add thread safety warnings if available
    if let Some(ctx) = context {
        if let Some(thread_safety) = &ctx.thread_safety {
            if !thread_safety.trait_bounds.sync {
                writeln!(output, "    ///")?;
                writeln!(output, "    /// # Thread Safety")?;
                writeln!(
                    output,
                    "    /// ⚠️ Not thread-safe - requires external synchronization"
                )?;
            }
        }

        // Add pitfalls
        if let Some(pitfalls) = &ctx.pitfalls {
            if !pitfalls.pitfalls.is_empty() {
                writeln!(output, "    ///")?;
                writeln!(output, "    /// # Common Pitfalls")?;
                for pitfall in &pitfalls.pitfalls {
                    writeln!(
                        output,
                        "    /// - {}: {}",
                        pitfall.title, pitfall.explanation
                    )?;
                }
            }
        }
    }

    writeln!(
        output,
        "    pub fn build(self) -> Result<{}, Error> {{",
        wrapper_name
    )?;

    // Validate all parameters are set
    writeln!(output, "        // Validate all parameters are set")?;
    for param in params {
        let field_name = to_rust_field_name(&param.name);
        writeln!(
            output,
            "        let {} = self.{}.ok_or(Error::InvalidParameter)?;",
            field_name, field_name
        )?;
    }
    writeln!(output)?;

    // Add precondition validation if available
    if let Some(ctx) = context {
        if let Some(precond) = &ctx.preconditions {
            if !precond.non_null_params.is_empty() || !precond.preconditions.is_empty() {
                writeln!(output, "        // Validate preconditions")?;

                // Validate non-null (for pointer types)
                for param_name in &precond.non_null_params {
                    if params.iter().any(|p| &p.name == param_name) {
                        let field = to_rust_field_name(param_name);
                        writeln!(output, "        if {}.is_null() {{", field)?;
                        writeln!(output, "            return Err(Error::NullPointer);")?;
                        writeln!(output, "        }}")?;
                    }
                }

                // Validate other preconditions
                for precond_item in &precond.preconditions {
                    if let Some(param_name) = &precond_item.parameter {
                        if params.iter().any(|p| &p.name == param_name) {
                            let field = to_rust_field_name(param_name);
                            // Check for specific types of constraints
                            if precond_item.description.to_lowercase().contains("non-zero") {
                                writeln!(output, "        if {} == 0 {{", field)?;
                                writeln!(
                                    output,
                                    "            return Err(Error::InvalidParameter);"
                                )?;
                                writeln!(output, "        }}")?;
                            }
                        }
                    }
                }
                writeln!(output)?;
            }
        }
    }

    // Call FFI function
    writeln!(output, "        unsafe {{")?;

    if has_output_param {
        // Need to create an uninitialized handle first
        writeln!(output, "            let mut handle = std::ptr::null_mut();")?;
        write!(
            output,
            "            let result = ffi::{}(&mut handle",
            create_func.name
        )?;
        for param in params {
            write!(output, ", {}", to_rust_field_name(&param.name))?;
        }
        writeln!(output, ");")?;
        writeln!(output)?;

        // Enhanced error handling with error_semantics
        if let Some(ctx) = context {
            if let Some(error_info) = &ctx.error_semantics {
                if !error_info.errors.is_empty() {
                    writeln!(output, "            if result != 0 {{")?;
                    writeln!(output, "                return Err(match result {{")?;
                    for (code, detail) in error_info.errors.iter().take(5) {
                        writeln!(
                            output,
                            "                    {} => Error::Specific(\"{}:{}\".to_string()),",
                            code, detail.description, code
                        )?;
                    }
                    writeln!(output, "                    _ => Error::from(result),")?;
                    writeln!(output, "                }});")?;
                    writeln!(output, "            }}")?;
                } else {
                    // Fallback to generic error handling
                    writeln!(output, "            if result != 0 {{")?;
                    writeln!(output, "                return Err(Error::from(result));")?;
                    writeln!(output, "            }}")?;
                }
            } else {
                writeln!(output, "            if result != 0 {{")?;
                writeln!(output, "                return Err(Error::from(result));")?;
                writeln!(output, "            }}")?;
            }
        } else {
            writeln!(output, "            if result != 0 {{")?;
            writeln!(output, "                return Err(Error::from(result));")?;
            writeln!(output, "            }}")?;
        }

        writeln!(output, "            if handle.is_null() {{")?;
        writeln!(output, "                return Err(Error::NullPointer);")?;
        writeln!(output, "            }}")?;
        writeln!(output)?;
        let expr_type = type_for_expr(wrapper_name);
        writeln!(output, "            Ok({} {{ handle }})", expr_type)?;
    } else {
        // Direct handle return
        write!(
            output,
            "            let handle = ffi::{}(",
            create_func.name
        )?;
        for (i, param) in params.iter().enumerate() {
            if i > 0 {
                write!(output, ", ")?;
            }
            write!(output, "{}", to_rust_field_name(&param.name))?;
        }
        writeln!(output, ");")?;
        writeln!(output)?;
        writeln!(output, "            if handle.is_null() {{")?;
        writeln!(output, "                return Err(Error::NullPointer);")?;
        writeln!(output, "            }}")?;
        writeln!(output)?;
        let expr_type = type_for_expr(wrapper_name);
        writeln!(output, "            Ok({} {{ handle }})", expr_type)?;
    }

    writeln!(output, "        }}")?;
    writeln!(output, "    }}")?;

    Ok(())
}

/// Convert C parameter name to Rust field name
fn to_rust_field_name(c_name: &str) -> String {
    // Remove common prefixes
    let name = c_name.trim_start_matches('p').trim_start_matches('_');

    // Convert to snake_case if needed
    if name.chars().any(|c| c.is_uppercase()) {
        // Has uppercase - convert camelCase to snake_case
        let mut result = String::new();
        for (i, ch) in name.chars().enumerate() {
            if ch.is_uppercase() && i > 0 {
                result.push('_');
            }
            result.push(ch.to_lowercase().next().unwrap());
        }
        result
    } else {
        name.to_string()
    }
}

/// Convert C type to Rust type
fn to_rust_type(c_type: &str) -> String {
    // Handle pointer types
    if c_type.contains('*') {
        // Preserve pointer types as-is (trim whitespace). Do not strip `mut`/`const` tokens,
        // they are meaningful in pointer types (e.g. `*mut ::core::ffi::c_void`).
        // Also normalize any spaced `::` tokens that may come from `quote` formatting
        // (e.g. `:: core :: ffi :: c_void`).
        return c_type
            .trim()
            .replace(":: ", "::")
            .replace(" ::", "::")
            .to_string();
    }

    // Simple type conversions - can be expanded
    let ty = c_type
        .replace("const ", "")
        .replace("*", "")
        .trim()
        .to_string();

    match ty.as_str() {
        "int" | "int32_t" => "i32".to_string(),
        "unsigned int" | "uint32_t" => "u32".to_string(),
        "long" | "int64_t" => "i64".to_string(),
        "unsigned long" | "uint64_t" => "u64".to_string(),
        "short" | "int16_t" => "i16".to_string(),
        "unsigned short" | "uint16_t" => "u16".to_string(),
        "char" | "int8_t" => "i8".to_string(),
        "unsigned char" | "uint8_t" => "u8".to_string(),
        "size_t" => "usize".to_string(),
        "float" => "f32".to_string(),
        "double" => "f64".to_string(),
        _ => ty,
    }
}

/// When emitting a generic type in an expression (struct literal) we must
/// use the turbofish form `Type::<Marker> {` to avoid the Rust parser
/// interpreting `<`/`>` as comparison operators. This helper inserts the
/// `::` before the first `<` if needed.
fn type_for_expr(s: &str) -> String {
    if s.contains("::<") || !s.contains('<') {
        s.to_string()
    } else {
        s.replacen("<", "::<", 1)
    }
}
