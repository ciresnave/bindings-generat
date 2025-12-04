use crate::analyzer::llm_doc_orchestrator::EnhancedDocumentation;
use crate::enrichment::context::FunctionContext;
use crate::ffi::FfiFunction;
use crate::generator::doc_generator;
use crate::tooling::cargo_features::{FeatureGuard, SafetyMode};
use crate::utils::doc_sanitizer::sanitize_doc;
use crate::utils::naming::{detect_library_prefix, to_idiomatic_rust_name};
use std::fmt::Write;
use tracing::debug;

/// Generate safe method wrapper for FFI function
///
/// When `func_context` is provided, generates enhanced documentation with comprehensive
/// safety information from all 13 analyzers (thread safety, preconditions, API sequences,
/// resource limits, numeric constraints, error semantics, anti-patterns, etc.).
///
/// When `func_context` is None, falls back to basic FFI documentation.
pub fn generate_safe_method(
    func: &FfiFunction,
    handle_type: Option<&str>,
    func_context: Option<&FunctionContext>,
    enhanced_docs: Option<&EnhancedDocumentation>,
) -> Option<String> {
    debug!(
        "Generating {} method for {}",
        if func_context.is_some() {
            "enhanced"
        } else {
            "safe"
        },
        func.name
    );

    let mut code = String::new();

    // Check if first parameter is an output parameter (pointer to handle type)
    let is_output_param = if let Some(handle) = handle_type {
        func.params
            .first()
            .map(|p| {
                let has_mut_ptr = p.ty.contains("* mut") || p.ty.contains("*mut");
                let has_handle = p.ty.contains(handle);
                has_mut_ptr && has_handle
            })
            .unwrap_or(false)
    } else {
        false
    };

    // If it's an output parameter, skip it - should be handled as a constructor
    if is_output_param {
        debug!("Skipping {} - first param is output parameter", func.name);
        return None;
    }

    // Determine if this is a method or a free function
    let is_method = handle_type.is_some()
        && func
            .params
            .first()
            .map(|p| p.ty.contains(handle_type.unwrap()))
            .unwrap_or(false);

    if is_method {
        generate_method_impl(
            func,
            handle_type.unwrap(),
            func_context,
            enhanced_docs,
            &mut code,
        );
    } else {
        generate_free_function_impl(func, func_context, enhanced_docs, &mut code);
    }

    Some(code)
}

fn generate_method_impl(
    func: &FfiFunction,
    handle_type: &str,
    func_context: Option<&FunctionContext>,
    enhanced_docs: Option<&EnhancedDocumentation>,
    code: &mut String,
) {
    let method_name = to_method_name(&func.name, handle_type);

    // Extract parameters (skip the first one which is the handle)
    let params: Vec<_> = func.params.iter().skip(1).collect();

    // Generate documentation - use enhanced if available, otherwise fall back to basic
    if let Some(context) = func_context {
        let docs = doc_generator::generate_enhanced_docs(context, "    ", enhanced_docs);
        code.push_str(&docs);

        // Add Rust attributes based on enriched context
        if let Some(attr_info) = &context.attributes {
            if attr_info.is_must_use() {
                writeln!(code, "    #[must_use]").unwrap();
            }
            if attr_info.is_deprecated() {
                // Try to extract deprecation message from attributes
                let deprecation_msg = attr_info.attributes.iter().find_map(|attr| {
                    if let crate::analyzer::attributes::AttributeType::Deprecated {
                        since: _,
                        note,
                    } = &attr.attr_type
                    {
                        note.clone()
                    } else {
                        None
                    }
                });
                if let Some(msg) = deprecation_msg {
                    writeln!(code, "    #[deprecated(note = \"{}\")]", msg).unwrap();
                } else {
                    writeln!(code, "    #[deprecated]").unwrap();
                }
            }
        }

        // Add deprecation from version_history if not already deprecated
        if !context.version_history.is_empty() {
            let deprecation = &context.version_history[0];
            if let (Some(reason), Some(replacement)) =
                (&deprecation.reason, &deprecation.replacement)
            {
                writeln!(
                    code,
                    "    #[deprecated(note = \"{} Use {} instead\")]",
                    reason, replacement
                )
                .unwrap();
            } else if let Some(replacement) = &deprecation.replacement {
                writeln!(
                    code,
                    "    #[deprecated(note = \"Use {} instead\")]",
                    replacement
                )
                .unwrap();
            }
        }

        // Add platform-specific compilation using cfg_attributes
        if let Some(platform_info) = &context.platform {
            if !platform_info.cfg_attributes.is_empty() {
                // Use the first cfg attribute
                writeln!(code, "    {}", platform_info.cfg_attributes[0]).unwrap();
            } else if !platform_info.available_on.is_empty() {
                // Generate cfg based on available_on platforms
                let platform_names: Vec<_> = platform_info
                    .available_on
                    .iter()
                    .map(|p| format!("{:?}", p).to_lowercase())
                    .collect();
                if !platform_names.is_empty() {
                    let first_platform = &platform_names[0];
                    if first_platform.contains("windows") {
                        writeln!(code, "    #[cfg(target_os = \"windows\")]").unwrap();
                    } else if first_platform.contains("linux") {
                        writeln!(code, "    #[cfg(target_os = \"linux\")]").unwrap();
                    } else if first_platform.contains("macos") || first_platform.contains("darwin")
                    {
                        writeln!(code, "    #[cfg(target_os = \"macos\")]").unwrap();
                    }
                }
            }
        }

        writeln!(code, "    #[inline]").unwrap();
    } else if let Some(docs) = &func.docs {
        let sanitized = sanitize_doc(docs);
        for line in sanitized.lines() {
            writeln!(code, "    /// {}", line).unwrap();
        }
    }
    writeln!(code, "    pub fn {}(", method_name).unwrap();
    writeln!(code, "        &mut self,").unwrap();

    // Add parameters
    for param in &params {
        let param_name = to_param_name(&param.name);
        let param_type = to_safe_type(&param.ty);
        writeln!(code, "        {}: {},", param_name, param_type).unwrap();
    }

    // Determine return type
    let return_type = if func.return_type == "()"
        || func.return_type == "c_void"
        || is_status_type(&func.return_type)
    {
        "Result<(), Error>".to_string()
    } else {
        format!("Result<{}, Error>", to_safe_type(&func.return_type))
    };

    writeln!(code, ") -> {} {{", return_type).unwrap();
    writeln!(code, "        unsafe {{").unwrap();

    // Generate feature-gated null pointer checks for raw pointer parameters
    // The checks vary by safety mode: strict (all), balanced (required), permissive (none)
    for param in &params {
        if is_raw_pointer_type(&param.ty) {
            let param_name = to_param_name(&param.name);
            let is_optional = is_optional_pointer(&param.name);

            // Generate conditional compilation for each mode
            // Strict mode: Check all pointers
            writeln!(code, "            #[cfg(feature = \"strict\")]").unwrap();
            writeln!(code, "            if {}.is_null() {{", param_name).unwrap();
            writeln!(code, "                return Err(Error::NullPointer);").unwrap();
            writeln!(code, "            }}").unwrap();

            // Balanced mode: Check required pointers only
            if !is_optional {
                writeln!(
                    code,
                    "            #[cfg(all(feature = \"balanced\", not(feature = \"strict\")))]"
                )
                .unwrap();
                writeln!(code, "            if {}.is_null() {{", param_name).unwrap();
                writeln!(code, "                return Err(Error::NullPointer);").unwrap();
                writeln!(code, "            }}").unwrap();
            }

            // Permissive mode: No checks
        }
    }

    // Add feature-gated numeric constraint validation
    // Only strict and balanced modes perform these checks
    if let Some(context) = func_context {
        if let Some(constraints) = &context.numeric_constraints {
            for param in &params {
                for constraint in &constraints.constraints {
                    if constraint.parameter_name.as_deref() == Some(&param.name) {
                        let param_name = to_param_name(&param.name);

                        // Generate validation based on constraint type
                        if let (Some(min), Some(max)) = (constraint.min_value, constraint.max_value)
                        {
                            writeln!(code, "            #[cfg(any(feature = \"strict\", feature = \"balanced\"))]").unwrap();
                            writeln!(
                                code,
                                "            if {} as i64 < {} || {} as i64 > {} {{",
                                param_name, min, param_name, max
                            )
                            .unwrap();
                            writeln!(code, "                return Err(Error::InvalidParameter);")
                                .unwrap();
                            writeln!(code, "            }}").unwrap();
                        }

                        if constraint.must_be_power_of_two {
                            writeln!(code, "            #[cfg(any(feature = \"strict\", feature = \"balanced\"))]").unwrap();
                            writeln!(
                                code,
                                "            if !({}).is_power_of_two() {{",
                                param_name
                            )
                            .unwrap();
                            writeln!(code, "                return Err(Error::InvalidParameter);")
                                .unwrap();
                            writeln!(code, "            }}").unwrap();
                        }

                        if let Some(align) = constraint.alignment_bytes {
                            writeln!(code, "            #[cfg(any(feature = \"strict\", feature = \"balanced\"))]").unwrap();
                            writeln!(
                                code,
                                "            if {} as usize % {} != 0 {{",
                                param_name, align
                            )
                            .unwrap();
                            writeln!(code, "                return Err(Error::InvalidParameter);")
                                .unwrap();
                            writeln!(code, "            }}").unwrap();
                        }
                    }
                }
            }
        }

        // Add precondition validation
        if let Some(precond) = &context.preconditions {
            for param in &params {
                // Check for non-zero constraints in preconditions
                let has_nonzero = precond.preconditions.iter().any(|p| {
                    p.parameter.as_ref() == Some(&param.name)
                        && matches!(
                            p.constraint_type,
                            crate::analyzer::preconditions::ConstraintType::Custom { .. }
                        )
                });

                if has_nonzero {
                    let param_name = to_param_name(&param.name);
                    writeln!(
                        code,
                        "            #[cfg(any(feature = \"strict\", feature = \"balanced\"))]"
                    )
                    .unwrap();
                    writeln!(code, "            if {} == 0 {{", param_name).unwrap();
                    writeln!(code, "                return Err(Error::InvalidParameter);").unwrap();
                    writeln!(code, "            }}").unwrap();
                }
            }

            // Generate debug assertions for documented preconditions
            // These are always active in debug builds (even permissive mode)
            for precond in &precond.preconditions {
                if let Some(param_name) = &precond.parameter {
                    let rust_param = to_param_name(param_name);
                    let assertion = match &precond.constraint_type {
                        crate::analyzer::preconditions::ConstraintType::NonNull => {
                            format!(
                                "debug_assert!(!{}.is_null(), \"{} must not be null\");",
                                rust_param, param_name
                            )
                        }
                        crate::analyzer::preconditions::ConstraintType::Range {
                            min,
                            max,
                            inclusive,
                        } => {
                            if let (Some(min_val), Some(max_val)) = (min, max) {
                                let op = if *inclusive { "=" } else { "" };
                                let left_op = format!(">{}", op);
                                let right_op = format!("<{}", op);
                                format!(
                                    "debug_assert!({}{}{} && {}{}{}, \"{} must be in range\");",
                                    rust_param,
                                    left_op,
                                    min_val,
                                    rust_param,
                                    right_op,
                                    max_val,
                                    param_name
                                )
                            } else if let Some(min_val) = min {
                                let op = if *inclusive { "=" } else { "" };
                                let left_op = format!(">{}", op);
                                format!(
                                    "debug_assert!({}{}{}, \"{} must be >{} {}\");",
                                    rust_param, left_op, min_val, param_name, op, min_val
                                )
                            } else if let Some(max_val) = max {
                                let op = if *inclusive { "=" } else { "" };
                                let right_op = format!("<{}", op);
                                format!(
                                    "debug_assert!({}{}{}, \"{} must be <{} {}\");",
                                    rust_param, right_op, max_val, param_name, op, max_val
                                )
                            } else {
                                continue;
                            }
                        }
                        crate::analyzer::preconditions::ConstraintType::PowerOfTwo => {
                            format!(
                                "debug_assert!({}.is_power_of_two(), \"{} must be a power of 2\");",
                                rust_param, param_name
                            )
                        }
                        crate::analyzer::preconditions::ConstraintType::MultipleOf { value } => {
                            format!(
                                "debug_assert!({} % {} == 0, \"{} must be a multiple of {}\");",
                                rust_param, value, param_name, value
                            )
                        }
                        _ => {
                            // For other constraint types, just add a descriptive comment
                            format!(
                                "// Precondition: {} - {:?}",
                                param_name, precond.constraint_type
                            )
                        }
                    };

                    writeln!(code, "            {}", assertion).unwrap();
                }
            }
        }
    }

    // Convert string parameters to CString
    for param in &params {
        if is_c_string_type(&param.ty) {
            let param_name = to_param_name(&param.name);
            writeln!(
                code,
                "            let {}_cstr = std::ffi::CString::new({}).map_err(|_| Error::NullPointer)?;",
                param_name, param_name
            )
            .unwrap();
        }
    }

    // Generate function call
    // Check if we need to pass a reference to the handle
    // If the first parameter expects a pointer to the handle type (e.g., *const HandleType),
    // we need to pass &self.handle instead of self.handle
    let first_param = func.params.first().unwrap();
    let needs_reference =
        first_param.ty.starts_with("*const ") || first_param.ty.starts_with("* const ");

    let handle_arg = if needs_reference {
        "&self.handle"
    } else {
        "self.handle"
    };

    // Add tracing span (if feature enabled)
    writeln!(code, "            #[cfg(feature = \"tracing\")]").unwrap();
    writeln!(
        code,
        "            let _span = tracing::trace_span!(\"ffi_call\", function = \"{}\").entered();",
        func.name
    )
    .unwrap();

    write!(
        code,
        "            let result = ffi::{}({}",
        func.name, handle_arg
    )
    .unwrap();
    for param in &params {
        let param_name = to_param_name(&param.name);
        if is_c_string_type(&param.ty) {
            write!(code, ", {}_cstr.as_ptr()", param_name).unwrap();
        } else {
            write!(code, ", {}", param_name).unwrap();
        }
    }
    writeln!(code, ");").unwrap();

    // Enhanced error handling with error_semantics
    if is_status_type(&func.return_type) {
        if let Some(context) = func_context {
            if let Some(error_info) = &context.error_semantics {
                if !error_info.errors.is_empty() {
                    writeln!(code, "            match result {{").unwrap();
                    writeln!(code, "                0 => Ok(()),").unwrap();
                    // Add specific error mappings (limit to top 5 to avoid bloat)
                    for (code_val, detail) in error_info.errors.iter().take(5) {
                        writeln!(
                            code,
                            "                {} => Err(Error::Specific(\"{}: {}\".to_string())),",
                            code_val, code_val, detail.description
                        )
                        .unwrap();
                    }
                    writeln!(code, "                _ => Err(Error::from(result)),").unwrap();
                    writeln!(code, "            }}").unwrap();
                } else {
                    // Fallback to generic error handling
                    writeln!(code, "            if result == 0 {{").unwrap();
                    writeln!(code, "                Ok(())").unwrap();
                    writeln!(code, "            }} else {{").unwrap();
                    writeln!(code, "                Err(Error::from(result))").unwrap();
                    writeln!(code, "            }}").unwrap();
                }
            } else {
                writeln!(code, "            if result == 0 {{").unwrap();
                writeln!(code, "                Ok(())").unwrap();
                writeln!(code, "            }} else {{").unwrap();
                writeln!(code, "                Err(Error::from(result))").unwrap();
                writeln!(code, "            }}").unwrap();
            }
        } else {
            writeln!(code, "            if result == 0 {{").unwrap();
            writeln!(code, "                Ok(())").unwrap();
            writeln!(code, "            }} else {{").unwrap();
            writeln!(code, "                Err(Error::from(result))").unwrap();
            writeln!(code, "            }}").unwrap();
        }
    } else {
        writeln!(code, "            Ok(result)").unwrap();
    }

    writeln!(code, "        }}").unwrap();
    writeln!(code, "    }}").unwrap();
    writeln!(code).unwrap();
}

fn generate_free_function_impl(
    func: &FfiFunction,
    func_context: Option<&FunctionContext>,
    enhanced_docs: Option<&EnhancedDocumentation>,
    code: &mut String,
) {
    let func_name = to_function_name(&func.name);

    // Generate documentation - use enhanced if available, otherwise fall back to basic
    if let Some(context) = func_context {
        let docs = doc_generator::generate_enhanced_docs(context, "", enhanced_docs);
        code.push_str(&docs);
    } else if let Some(docs) = &func.docs {
        let sanitized = sanitize_doc(docs);
        for line in sanitized.lines() {
            writeln!(code, "/// {}", line).unwrap();
        }
    }
    writeln!(code, "pub fn {}(", func_name).unwrap();

    // Add parameters
    for (i, param) in func.params.iter().enumerate() {
        let param_name = to_param_name(&param.name);
        let param_type = to_safe_type(&param.ty);
        if i < func.params.len() - 1 {
            writeln!(code, "    {}: {},", param_name, param_type).unwrap();
        } else {
            writeln!(code, "    {}: {}", param_name, param_type).unwrap();
        }
    }

    // Determine return type. Free functions use Result like methods do.
    let return_type = if func.return_type == "()"
        || func.return_type == "c_void"
        || is_status_type(&func.return_type)
    {
        "Result<(), Error>".to_string()
    } else {
        format!("Result<{}, Error>", to_safe_type(&func.return_type))
    };

    writeln!(code, ") -> {} {{", return_type).unwrap();
    writeln!(code, "    unsafe {{").unwrap();

    // Convert string parameters to CString
    for param in &func.params {
        if is_c_string_type(&param.ty) {
            let param_name = to_param_name(&param.name);
            writeln!(
                code,
                "        let {}_cstr = std::ffi::CString::new({})",
                param_name, param_name
            )
            .unwrap();
            writeln!(code, "            .map_err(|_| Error::InvalidString)?;").unwrap();
        }
    }

    // Generate function call
    write!(code, "        let result = ffi::{}(", func.name).unwrap();
    for (i, param) in func.params.iter().enumerate() {
        if i > 0 {
            write!(code, ", ").unwrap();
        }
        let param_name = to_param_name(&param.name);
        if is_c_string_type(&param.ty) {
            write!(code, "{}_cstr.as_ptr()", param_name).unwrap();
        } else {
            write!(code, "{}", param_name).unwrap();
        }
    }
    writeln!(code, ");").unwrap();

    // Handle status-like return types uniformly
    if is_status_type(&func.return_type) {
        writeln!(code, "        if result == 0 {{").unwrap();
        writeln!(code, "            Ok(())").unwrap();
        writeln!(code, "        }} else {{").unwrap();
        writeln!(code, "            Err(Error::from(result))").unwrap();
        writeln!(code, "        }}").unwrap();
    } else {
        writeln!(code, "        Ok(result)").unwrap();
    }

    writeln!(code, "    }}").unwrap();
    writeln!(code, "}}").unwrap();
    writeln!(code).unwrap();
}

fn to_method_name(func_name: &str, handle_type: &str) -> String {
    // Detect library prefix from function name
    let prefix = detect_library_prefix(func_name);

    // Use idiomatic naming conversion
    let mut name = to_idiomatic_rust_name(func_name, prefix.as_deref());

    // Also try to remove handle type name if it appears in the method
    let handle_lower = handle_type.to_lowercase();
    if name.starts_with(&handle_lower) {
        name = name[handle_lower.len()..]
            .trim_start_matches('_')
            .to_string();
    }

    // Ensure non-empty
    if name.is_empty() {
        name = func_name.to_lowercase();
    }

    name
}

fn to_function_name(func_name: &str) -> String {
    // Detect library prefix and convert to idiomatic Rust
    let prefix = detect_library_prefix(func_name);
    to_idiomatic_rust_name(func_name, prefix.as_deref())
}

fn to_param_name(param_name: &str) -> String {
    // Sanitize parameter name
    let mut name = param_name.to_lowercase();

    // Handle Rust keywords
    if matches!(
        name.as_str(),
        "type" | "mod" | "fn" | "let" | "mut" | "ref" | "self" | "Self"
    ) {
        name = format!("{}_", name);
    }

    name
}

fn is_c_string_type(ffi_type: &str) -> bool {
    // Check if this is a C string type (*const c_char)
    ffi_type.contains("*") && ffi_type.contains("const") && ffi_type.contains("c_char")
}

fn is_raw_pointer_type(ffi_type: &str) -> bool {
    // Check if this is a raw pointer (but not a C string)
    ffi_type.contains("*") && !ffi_type.contains("c_char")
}

fn is_optional_pointer(param_name: &str) -> bool {
    // Heuristic: parameters with certain names are likely optional
    let name_lower = param_name.to_lowercase();
    name_lower.contains("optional")
        || name_lower.contains("opt")
        || name_lower.ends_with("_opt")
        || name_lower.starts_with("opt_")
}

fn to_safe_type(ffi_type: &str) -> String {
    // Convert FFI types to safe Rust types
    // Normalize spacing around path separators produced by `quote::ToTokens`
    // (e.g. it may emit `:: core :: ffi :: c_void`). Collapse those to `::core::ffi::c_void`.
    let normalized = ffi_type
        .trim()
        .replace(":: ", "::")
        .replace(" ::", "::");
    // Handle C string pointers specially
    if normalized.contains("c_char") && normalized.contains("*") {
        // Treat `*const c_char` as borrowed `&str` input;
        // treat `*mut c_char` (mutable buffer) as a raw pointer to C char.
        if normalized.contains("const") {
            return "&str".to_string();
        } else {
            return "*mut ::core::ffi::c_char".to_string();
        }
    }

    // Helper to detect primitive / known types that should not be prefixed
    fn is_known_primitive(s: &str) -> bool {
        matches!(
            s,
            "c_int" | "i32" | "c_uint" | "u32" | "c_long" | "i64" | "c_ulong" | "u64" |
            "c_float" | "f32" | "c_double" | "f64" | "c_void" | "usize" | "isize" | "bool" | "()"
        )
    }

    // Normalize pointer spacing (handle `* mut` / `* const` variants)
    let normalized = normalized.replace("* mut", "*mut").replace("* const", "*const");

    // If this is a pointer type, preserve pointer qualifiers but qualify the base type with `ffi::` when appropriate
    if normalized.contains('*') {
        // Split on the last space to separate pointer qualifiers from the base type
        // e.g. "*mut cudaGraph_t" -> ("*mut", "cudaGraph_t")
        if let Some(idx) = normalized.rfind(' ') {
            let (prefix, base) = normalized.split_at(idx + 1);
            let base = base.trim();
            // Canonicalize c_void pointer types to ::core::ffi::c_void
            // Preserve all pointer qualifiers (e.g. `*mut *mut`) instead of collapsing to a single level.
            if base == "c_void" {
                let ptr = prefix.trim_end();
                // If there are no pointer qualifiers, treat bare c_void as an opaque pointer
                if ptr.is_empty() {
                    return "*mut ::core::ffi::c_void".to_string();
                } else {
                    return format!("{} ::core::ffi::c_void", ptr);
                }
            }
            let qualified_base = if base.contains("::") || base.starts_with("ffi::") || is_known_primitive(base) || base.starts_with("c_") {
                base.to_string()
            } else {
                format!("ffi::{}", base)
            };
            return format!("{} {}", prefix.trim_end(), qualified_base);
        } else {
            // No space found - fallback to qualifying whole token if needed
            let base = normalized.trim();
            if base == "c_void" {
                // Treat bare c_void as opaque pointer
                return "*mut ::core::ffi::c_void".to_string();
            }
            if base.contains("::") || base.starts_with("ffi::") || is_known_primitive(base) || base.starts_with("c_") {
                return base.to_string();
            } else {
                return format!("ffi::{}", base);
            }
        }
    }

    // Non-pointer primitives mapping
    match normalized.as_str() {
        "c_int" | "i32" => "i32".to_string(),
        "c_uint" | "u32" => "u32".to_string(),
        "c_long" => "i64".to_string(),
        "c_ulong" => "u64".to_string(),
        "c_float" | "f32" => "f32".to_string(),
        "c_double" | "f64" => "f64".to_string(),
        "c_void" => "()".to_string(),
        _ => {
            if normalized.contains("::") || normalized.starts_with("ffi::") || is_known_primitive(&normalized) || normalized.starts_with("c_") {
                normalized.to_string()
            } else {
                // Qualify unknown identifiers as FFI types
                format!("ffi::{}", normalized)
            }
        }
    }
}

fn is_status_type(type_str: &str) -> bool {
    let normalized = type_str.to_lowercase();
    normalized.contains("status")
        || normalized.contains("error")
        || normalized == "i32"
        || normalized == "c_int"
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_to_method_name() {
        assert_eq!(to_method_name("context_set_value", "context"), "set_value");
        assert_eq!(to_method_name("my_func", "other"), "my_func");
    }

    #[test]
    fn test_to_param_name() {
        assert_eq!(to_param_name("value"), "value");
        assert_eq!(to_param_name("type"), "type_");
    }

    #[test]
    fn test_to_safe_type() {
        assert_eq!(to_safe_type("c_int"), "i32");
        assert_eq!(to_safe_type("*const c_char"), "&str");
    }
}
