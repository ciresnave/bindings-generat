use crate::ffi::FfiFunction;
use crate::utils::doc_sanitizer::sanitize_doc;
use std::fmt::Write;
use tracing::debug;

/// Generate safe method wrapper for FFI function
pub fn generate_safe_method(func: &FfiFunction, handle_type: Option<&str>) -> Option<String> {
    debug!("Generating safe method for {}", func.name);

    let mut code = String::new();

    // Check if first parameter is an output parameter (pointer to handle type)
    let is_output_param = if let Some(handle) = handle_type {
        func.params
            .first()
            .map(|p| {
                // Check if it's a mutable pointer to the handle type
                // Output parameters are typically: *mut HandleType (where HandleType is itself a pointer)
                // So in practice: *mut cudaStream_t where cudaStream_t = *mut CUstream_st
                let has_mut_ptr = p.ty.contains("* mut") || p.ty.contains("*mut");
                let has_handle = p.ty.contains(handle);

                let is_output = has_mut_ptr && has_handle;

                if is_output {
                    debug!(
                        "Detected output parameter in {}: param '{}' has type '{}'",
                        func.name, p.name, p.ty
                    );
                }

                is_output
            })
            .unwrap_or(false)
    } else {
        false
    };

    // If it's an output parameter, skip it - should be handled as a constructor
    if is_output_param {
        debug!(
            "Skipping {} - first param is output parameter for {}",
            func.name,
            handle_type.unwrap()
        );
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
        // Generate as a method on the wrapper struct
        generate_method(func, handle_type.unwrap(), &mut code);
    } else {
        // Generate as a free function
        generate_free_function(func, &mut code);
    }

    Some(code)
}

fn generate_method(func: &FfiFunction, handle_type: &str, code: &mut String) {
    let method_name = to_method_name(&func.name, handle_type);

    // Extract parameters (skip the first one which is the handle)
    let params: Vec<_> = func.params.iter().skip(1).collect();

    // Generate method signature
    if let Some(docs) = &func.docs {
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

    writeln!(code, "    ) -> {} {{", return_type).unwrap();
    writeln!(code, "        unsafe {{").unwrap();

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

    write!(
        code,
        "            let result = ffi::{}({}",
        func.name, handle_arg
    )
    .unwrap();
    for param in &params {
        write!(code, ", {}", to_param_name(&param.name)).unwrap();
    }
    writeln!(code, ");").unwrap();

    // Handle result
    if is_status_type(&func.return_type) {
        writeln!(code, "            if result == 0 {{").unwrap();
        writeln!(code, "                Ok(())").unwrap();
        writeln!(code, "            }} else {{").unwrap();
        writeln!(code, "                Err(Error::from(result))").unwrap();
        writeln!(code, "            }}").unwrap();
    } else {
        writeln!(code, "            Ok(result)").unwrap();
    }

    writeln!(code, "        }}").unwrap();
    writeln!(code, "    }}").unwrap();
    writeln!(code).unwrap();
}

fn generate_free_function(func: &FfiFunction, code: &mut String) {
    let func_name = to_function_name(&func.name);

    // Generate function signature
    if let Some(docs) = &func.docs {
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

    // Determine return type
    let return_type = if func.return_type == "()" || func.return_type == "c_void" {
        "()".to_string()
    } else {
        to_safe_type(&func.return_type)
    };

    writeln!(code, ") -> {} {{", return_type).unwrap();
    writeln!(code, "    unsafe {{").unwrap();

    // Generate function call
    write!(code, "        ffi::{}(", func.name).unwrap();
    for (i, param) in func.params.iter().enumerate() {
        if i > 0 {
            write!(code, ", ").unwrap();
        }
        write!(code, "{}", to_param_name(&param.name)).unwrap();
    }
    writeln!(code, ")").unwrap();

    writeln!(code, "    }}").unwrap();
    writeln!(code, "}}").unwrap();
    writeln!(code).unwrap();
}

fn to_method_name(func_name: &str, handle_type: &str) -> String {
    let mut name = func_name.to_lowercase();

    // Remove handle type prefix if present
    let handle_lower = handle_type.to_lowercase();
    if name.starts_with(&handle_lower) {
        name = name[handle_lower.len()..]
            .trim_start_matches('_')
            .to_string();
    }

    // Convert to snake_case (already is)
    name
}

fn to_function_name(func_name: &str) -> String {
    func_name.to_lowercase()
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

fn to_safe_type(ffi_type: &str) -> String {
    // Convert FFI types to safe Rust types
    let normalized = ffi_type.trim();

    if normalized.contains("c_char") && normalized.contains("*") {
        if normalized.contains("const") {
            "&str".to_string()
        } else {
            "String".to_string()
        }
    } else if normalized.contains("*") {
        // Raw pointers - keep as is for now (could be improved)
        normalized.to_string()
    } else {
        // Primitive types
        match normalized {
            "c_int" | "i32" => "i32".to_string(),
            "c_uint" | "u32" => "u32".to_string(),
            "c_long" => "i64".to_string(),
            "c_ulong" => "u64".to_string(),
            "c_float" | "f32" => "f32".to_string(),
            "c_double" | "f64" => "f64".to_string(),
            "c_void" => "()".to_string(),
            _ => normalized.to_string(),
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
