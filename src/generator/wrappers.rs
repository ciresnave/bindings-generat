use crate::analyzer::AnalysisResult;
use crate::analyzer::errors::ErrorPatterns;
use crate::analyzer::llm_parameters::LlmParameterAnalysis;
use crate::analyzer::raii::{HandleType, LifecyclePair};
use crate::enrichment::context::FunctionContext;
use crate::ffi::FfiFunction;
use std::fmt::Write;
use tracing::{debug, info, warn};

/// Generated RAII wrapper code
#[derive(Debug, Clone)]
pub struct RaiiWrapper {
    pub type_name: String,
    pub handle_type: String,
    pub code: String,
    pub builder_name: Option<String>,
    pub builder_code: Option<String>,
}

/// Determine the pattern used by a create function
#[derive(Debug, Clone, Copy, PartialEq)]
enum CreatePattern {
    /// Function returns the handle directly: `handle_t create(...)`
    ReturnsHandle,
    /// Function uses output parameter: `status_t create(*mut handle_t, ...)`
    OutputParameter,
    /// Function returns status with no output param (unsupported)
    Unknown,
}

/// Determine the pattern used by a destroy function
#[derive(Debug, Clone, Copy, PartialEq)]
enum DestroyPattern {
    /// Function takes handle and returns void: `void destroy(handle_t)`
    TakesHandleReturnsVoid,
    /// Function takes handle and returns status: `status_t destroy(handle_t)`
    TakesHandleReturnsStatus,
    /// Function takes pointer to handle: `void destroy(*mut handle_t)`
    TakesPointerReturnsVoid,
    /// Unknown pattern
    Unknown,
}

/// Analyze create function to determine its pattern
fn analyze_create_pattern(func: &FfiFunction, handle_type: &str) -> CreatePattern {
    // Check if return type contains the handle type
    if func.return_type.contains(handle_type) {
        debug!(
            "Create function {} returns handle type {}",
            func.name, handle_type
        );
        return CreatePattern::ReturnsHandle;
    }

    // Check if first parameter is *mut handle_type (output parameter pattern)
    if let Some(first_param) = func.params.first() {
        let is_mut_ptr = first_param.ty.contains("*mut") || first_param.ty.contains("* mut");
        let contains_handle = first_param.ty.contains(handle_type);

        if is_mut_ptr && contains_handle {
            debug!(
                "Create function {} uses output parameter pattern",
                func.name
            );
            return CreatePattern::OutputParameter;
        }
    }

    warn!(
        "Unknown create pattern for {}: return_type={}, params={:?}",
        func.name, func.return_type, func.params
    );
    CreatePattern::Unknown
}

/// Analyze destroy function to determine its pattern
fn analyze_destroy_pattern(func: &FfiFunction, handle_type: &str) -> DestroyPattern {
    // Check if it takes the handle type as a parameter
    let takes_handle = func.params.iter().any(|p| p.ty.contains(handle_type));

    if !takes_handle {
        return DestroyPattern::Unknown;
    }

    // Check if first param is a pointer
    if let Some(first_param) = func.params.first() {
        let is_ptr = first_param.ty.contains("*mut") || first_param.ty.contains("* mut");

        if is_ptr {
            return DestroyPattern::TakesPointerReturnsVoid;
        }
    }

    // Check return type
    let returns_void =
        func.return_type == "()" || func.return_type == "c_void" || func.return_type.is_empty();

    if returns_void {
        DestroyPattern::TakesHandleReturnsVoid
    } else {
        DestroyPattern::TakesHandleReturnsStatus
    }
}

/// Check if a type looks like a status/error code
fn is_status_type(type_str: &str) -> bool {
    let lower = type_str.to_lowercase();
    lower.contains("status")
        || lower.contains("error")
        || lower.contains("result")
        || type_str == "i32"
        || type_str == "c_int"
}

/// Generate RAII wrapper for a handle type with flexible FFI pattern support
pub fn generate_raii_wrapper(
    handle: &HandleType,
    pair: &LifecyclePair,
    create_func: Option<&FfiFunction>,
    destroy_func: Option<&FfiFunction>,
    error_patterns: &ErrorPatterns,
    _lib_name: &str,
    analysis: &AnalysisResult,
    _parameter_analysis: Option<&LlmParameterAnalysis>,
) -> RaiiWrapper {
    debug!("Generating RAII wrapper for {}", handle.name);

    let type_name = to_rust_type_name(&handle.name);
    let mut code = String::new();

    // Try to get enriched context for create and destroy functions
    let create_ctx = create_func.and_then(|f| analysis.function_contexts.get(&f.name));

    let _destroy_ctx = destroy_func.and_then(|f| analysis.function_contexts.get(&f.name));

    // Generate the struct with optimization attributes
    writeln!(code, "/// Safe wrapper for `{}`", handle.name).unwrap();
    writeln!(code, "///").unwrap();

    // Add enriched description if available
    if let Some(ctx) = create_ctx.as_ref().and_then(|c| c.description.as_ref()) {
        for line in ctx.lines() {
            writeln!(code, "/// {}", line).unwrap();
        }
        writeln!(code, "///").unwrap();
    }

    writeln!(
        code,
        "/// This wrapper provides RAII-style resource management with automatic cleanup."
    )
    .unwrap();

    // Add thread safety information if available
    if let Some(ctx) = create_ctx {
        if let Some(ts) = &ctx.thread_safety
            && !ts.trait_bounds.sync
        {
            writeln!(code, "///").unwrap();
            writeln!(
                code,
                "/// ⚠️  **Thread Safety**: Not safe for concurrent access"
            )
            .unwrap();
            if let Some(doc) = &ts.documentation {
                writeln!(code, "/// {}", doc).unwrap();
            }
        }

        // Add ownership information
        if let Some(ownership) = &ctx.ownership
            && ownership.return_ownership.requires_cleanup()
        {
            writeln!(code, "///").unwrap();
            writeln!(
                code,
                "/// **Ownership**: {}",
                ownership.return_ownership.description()
            )
            .unwrap();
        }

        // Add preconditions
        if let Some(precond) = &ctx.preconditions
            && (!precond.non_null_params.is_empty() || !precond.preconditions.is_empty())
        {
            writeln!(code, "///").unwrap();
            writeln!(code, "/// **Preconditions**:").unwrap();
            for param in &precond.non_null_params {
                writeln!(code, "/// - `{}` must not be null", param).unwrap();
            }
            for pc in precond.preconditions.iter().take(3) {
                writeln!(code, "/// - {}", pc.description).unwrap();
            }
        }
    }

    // Use ffi:: prefix for the handle type to ensure proper scoping
    let ffi_type = format!("ffi::{}", handle.name);

    writeln!(
        code,
        "/// The wrapper is a zero-cost abstraction - it has the same memory layout"
    )
    .unwrap();
    writeln!(code, "/// as the underlying handle type.").unwrap();
    writeln!(code, "#[repr(transparent)]").unwrap();
    writeln!(code, "pub struct {} {{", type_name).unwrap();
    writeln!(code, "    handle: {},", ffi_type).unwrap();
    writeln!(code, "}}").unwrap();
    writeln!(code).unwrap();

    // Generate impl block with new() constructor
    writeln!(code, "impl {} {{", type_name).unwrap();

    if let Some(create_fn) = create_func {
        generate_constructor(
            &mut code,
            create_fn,
            &ffi_type,
            &pair.create_fn,
            error_patterns,
            create_ctx,
        );
    } else {
        // Fallback if we don't have function info
        generate_simple_constructor(&mut code, &pair.create_fn);
    }

    // Generate accessor methods for raw handle access
    writeln!(code).unwrap();
    writeln!(code, "    /// Returns the raw FFI handle").unwrap();
    writeln!(code, "    #[inline]").unwrap();
    writeln!(code, "    pub fn as_raw(&self) -> {} {{", ffi_type).unwrap();
    writeln!(code, "        self.handle").unwrap();
    writeln!(code, "    }}").unwrap();
    writeln!(code).unwrap();
    writeln!(
        code,
        "    /// Returns a mutable pointer to the raw FFI handle"
    )
    .unwrap();
    writeln!(code, "    #[inline]").unwrap();
    writeln!(
        code,
        "    pub fn as_raw_mut(&mut self) -> *mut {} {{",
        ffi_type
    )
    .unwrap();
    writeln!(code, "        &mut self.handle").unwrap();
    writeln!(code, "    }}").unwrap();
    writeln!(code).unwrap();
    writeln!(code, "    /// Constructs a wrapper from a raw FFI handle").unwrap();
    writeln!(code, "    ///").unwrap();
    writeln!(code, "    /// # Safety").unwrap();
    writeln!(code, "    ///").unwrap();
    writeln!(
        code,
        "    /// The caller must ensure the handle is valid and properly initialized."
    )
    .unwrap();
    writeln!(
        code,
        "    /// The wrapper will take ownership and call the destructor on drop."
    )
    .unwrap();
    writeln!(code, "    #[inline]").unwrap();
    writeln!(
        code,
        "    pub unsafe fn from_raw(handle: {}) -> Self {{",
        ffi_type
    )
    .unwrap();
    writeln!(code, "        Self {{ handle }}").unwrap();
    writeln!(code, "    }}").unwrap();

    writeln!(code, "}}").unwrap();
    writeln!(code).unwrap();

    // Generate Drop implementation
    writeln!(code, "impl Drop for {} {{", type_name).unwrap();
    writeln!(code, "    fn drop(&mut self) {{").unwrap();

    if let Some(destroy_fn) = destroy_func {
        generate_drop_body(&mut code, destroy_fn, &handle.name, &pair.destroy_fn);
    } else {
        // Fallback if we don't have function info
        generate_simple_drop(&mut code, &pair.destroy_fn, &handle.name);
    }

    writeln!(code, "    }}").unwrap();
    writeln!(code, "}}").unwrap();
    writeln!(code).unwrap();

    // Generate builder pattern for complex constructors (skip if typestate analysis exists)
    // Instead of inlining builder code here, return it in the RaiiWrapper so callers
    // (the central generator) can deduplicate identical builders across wrappers.
    let mut builder_name_opt: Option<String> = None;
    let mut builder_code_opt: Option<String> = None;
    if analysis.builder_typestates.is_none() {
        if let Some(create_fn) = create_func {
            if let Some(bc) = crate::generator::builders::generate_builder(
                &type_name,
                create_fn,
                &handle.name,
                create_ctx,
            ) {
                builder_name_opt = Some(format!("{}Builder", type_name));
                builder_code_opt = Some(bc);
            }
        }
    }
    writeln!(code).unwrap();

    // Generate Send/Sync trait implementations based on thread safety analysis
    if let Some(ctx) = create_ctx {
        if let Some(ts) = &ctx.thread_safety {
            // Generate negative trait implementations for non-thread-safe types
            if !ts.trait_bounds.send {
                writeln!(code, "// Not Send: {}", ts.source).unwrap();
                writeln!(code, "impl !Send for {} {{}}", type_name).unwrap();
                writeln!(code).unwrap();
            }
            if !ts.trait_bounds.sync {
                writeln!(code, "// Not Sync: {}", ts.source).unwrap();
                writeln!(code, "impl !Sync for {} {{}}", type_name).unwrap();
                writeln!(code).unwrap();
            }

            // If thread-safe, add a comment indicating it's safe
            if ts.trait_bounds.send && ts.trait_bounds.sync {
                writeln!(code, "// Thread-safe: implements Send + Sync by default").unwrap();
                writeln!(code, "// Source: {}", ts.source).unwrap();
                writeln!(code).unwrap();
            }
        }
    } else {
        // Conservative default: assume not thread-safe unless proven otherwise
        writeln!(code, "// Thread safety unknown - assuming not thread-safe").unwrap();
        writeln!(code, "impl !Send for {} {{}}", type_name).unwrap();
        writeln!(code, "impl !Sync for {} {{}}", type_name).unwrap();
        writeln!(code).unwrap();
    }

    info!(
        "Generated RAII wrapper for {} -> {}",
        handle.name, type_name
    );

    RaiiWrapper {
        type_name,
        handle_type: handle.name.clone(),
        code,
        builder_name: builder_name_opt,
        builder_code: builder_code_opt,
    }
}

/// Generate constructor based on the actual FFI function signature
fn generate_constructor(
    code: &mut String,
    func: &FfiFunction,
    handle_type: &str,
    func_name: &str,
    error_patterns: &ErrorPatterns,
    func_ctx: Option<&FunctionContext>,
) {
    let pattern = analyze_create_pattern(func, handle_type);

    writeln!(code, "    /// Create a new instance").unwrap();

    // Add enriched documentation if available
    if let Some(ctx) = func_ctx {
        if let Some(desc) = &ctx.description {
            writeln!(code, "    ///").unwrap();
            let lines: Vec<&str> = desc.lines().take(3).collect();
            for line in lines {
                // Limit to 3 lines for constructor
                writeln!(code, "    /// {}", line).unwrap();
            }
        }

        // Add safety notes from preconditions
        if let Some(precond) = &ctx.preconditions
            && !precond.undefined_behavior.is_empty()
        {
            writeln!(code, "    ///").unwrap();
            writeln!(code, "    /// # Prerequisites").unwrap();
            for ub in precond.undefined_behavior.iter().take(2) {
                writeln!(code, "    /// - {}", ub).unwrap();
            }
        }
    }

    writeln!(code, "    #[inline]").unwrap();

    match pattern {
        CreatePattern::ReturnsHandle => {
            generate_returns_handle_constructor(code, func, func_name);
        }
        CreatePattern::OutputParameter => {
            generate_output_param_constructor(code, func, handle_type, func_name, error_patterns);
        }
        CreatePattern::Unknown => {
            warn!("Unknown create pattern for {}, using fallback", func_name);
            generate_simple_constructor(code, func_name);
        }
    }
}

/// Generate constructor for pattern: `handle_t create(...params...)`
fn generate_returns_handle_constructor(code: &mut String, func: &FfiFunction, func_name: &str) {
    // Check if function has parameters beyond output
    let has_params = !func.params.is_empty();

    if has_params {
        // Generate constructor that requires parameters
        // For now, make it private or document that it needs parameters
        writeln!(
            code,
            "    // TODO: Function {} requires parameters:",
            func_name
        )
        .unwrap();
        for param in &func.params {
            writeln!(code, "    //   - {}: {}", param.name, param.ty).unwrap();
        }
        writeln!(
            code,
            "    // Implement custom constructor with required parameters"
        )
        .unwrap();
        writeln!(code, "    #[doc(hidden)]").unwrap();
    }

    writeln!(code, "    pub fn new() -> Result<Self, Error> {{").unwrap();
    writeln!(code, "        unsafe {{").unwrap();

    // Generate call with placeholder params if needed
    if has_params {
        writeln!(
            code,
            "            // TODO: Pass actual parameters instead of defaults"
        )
        .unwrap();
        write!(code, "            let handle = ffi::{}(", func_name).unwrap();
        for (i, param) in func.params.iter().enumerate() {
            if i > 0 {
                write!(code, ", ").unwrap();
            }
            // Generate placeholder based on type
            if param.ty.contains("*") {
                write!(code, "std::ptr::null_mut()").unwrap();
            } else if param.ty.contains("i32")
                || param.ty.contains("c_int")
                || param.ty.contains("u32")
                || param.ty.contains("c_uint")
                || param.ty.contains("usize")
                || param.ty.contains("size_t")
            {
                write!(code, "0").unwrap();
            } else if param.ty.contains("f32")
                || param.ty.contains("c_float")
                || param.ty.contains("f64")
                || param.ty.contains("c_double")
            {
                write!(code, "0.0").unwrap();
            } else if param.ty.contains("bool") {
                write!(code, "false").unwrap();
            } else {
                write!(code, "std::mem::zeroed()").unwrap();
            }
        }
        writeln!(code, ");").unwrap();
    } else {
        writeln!(code, "            let handle = ffi::{}();", func_name).unwrap();
    }

    writeln!(code, "            if handle.is_null() {{").unwrap();
    writeln!(code, "                Err(Error::NullPointer)").unwrap();
    writeln!(code, "            }} else {{").unwrap();
    writeln!(code, "                Ok(Self {{ handle }})").unwrap();
    writeln!(code, "            }}").unwrap();
    writeln!(code, "        }}").unwrap();
    writeln!(code, "    }}").unwrap();
    writeln!(code).unwrap();
}

/// Generate intelligent placeholder value for a parameter based on its name and type
fn generate_smart_placeholder(param: &crate::ffi::FfiParam) -> String {
    let param_name_lower = param.name.to_lowercase();
    let type_lower = param.ty.to_lowercase();

    // Check if it's a pointer type
    let is_const_ptr = param.ty.contains("*const") || param.ty.contains("* const");
    let is_mut_ptr = param.ty.contains("*mut") || param.ty.contains("* mut");

    // Pattern 1: Size/count parameters - these are often optional (can be 0)
    if param_name_lower.contains("size")
        || param_name_lower.contains("count")
        || param_name_lower.contains("len")
        || param_name_lower.contains("num")
    {
        return "0".to_string();
    }

    // Pattern 2: Flag parameters - often have default/zero value
    if param_name_lower.contains("flag") || param_name_lower.contains("option") {
        return "0".to_string();
    }

    // Pattern 3: Optional pointer parameters (e.g., *const void)
    // These are often nullable in C APIs
    if is_const_ptr {
        // Check common patterns for optional parameters
        if param_name_lower.contains("user")
            || param_name_lower.contains("context")
            || param_name_lower.contains("data")
            || param_name_lower.contains("extra")
            || param_name_lower.contains("arg")
            || param_name_lower == "p"
            || param_name_lower == "ptr"
        {
            return "std::ptr::null()".to_string();
        }
    }

    // Pattern 4: Mutable pointer for optional output parameters
    if is_mut_ptr && (param_name_lower.contains("out") || param_name_lower.contains("result")) {
        return "std::ptr::null_mut()".to_string();
    }

    // Fallback to type-based generation
    if is_const_ptr {
        "std::ptr::null()".to_string()
    } else if is_mut_ptr {
        "std::ptr::null_mut()".to_string()
    } else if type_lower.contains("i32") || type_lower.contains("c_int") {
        "0".to_string()
    } else if type_lower.contains("u32") || type_lower.contains("c_uint") {
        "0".to_string()
    } else if type_lower.contains("i64") || type_lower.contains("c_long") {
        "0".to_string()
    } else if type_lower.contains("u64") || type_lower.contains("c_ulong") {
        "0".to_string()
    } else if type_lower.contains("usize") || type_lower.contains("size_t") {
        "0".to_string()
    } else if type_lower.contains("isize") {
        "0".to_string()
    } else if type_lower.contains("f32") || type_lower.contains("c_float") {
        "0.0".to_string()
    } else if type_lower.contains("f64") || type_lower.contains("c_double") {
        "0.0".to_string()
    } else if type_lower.contains("bool") {
        "false".to_string()
    } else {
        // For complex types, use zeroed memory
        "std::mem::zeroed()".to_string()
    }
}

/// Generate constructor for pattern: `status_t create(*mut handle_t, ...params...)`
fn generate_output_param_constructor(
    code: &mut String,
    func: &FfiFunction,
    _handle_type: &str,
    func_name: &str,
    error_patterns: &ErrorPatterns,
) {
    // Check if there are parameters beyond the output parameter
    let extra_params: Vec<_> = func.params.iter().skip(1).collect();
    let has_extra_params = !extra_params.is_empty();

    if has_extra_params {
        writeln!(
            code,
            "    // Note: Function {} requires additional parameters:",
            func_name
        )
        .unwrap();
        for param in &extra_params {
            writeln!(code, "    //   - {}: {}", param.name, param.ty).unwrap();
        }
        writeln!(code, "    // Using default/placeholder values for now").unwrap();
    }

    writeln!(code, "    pub fn new() -> Result<Self, Error> {{").unwrap();
    writeln!(code, "        unsafe {{").unwrap();
    writeln!(code, "            let mut handle = std::ptr::null_mut();").unwrap();

    // Generate the function call
    write!(
        code,
        "            let status = ffi::{}(&mut handle",
        func_name
    )
    .unwrap();

    // Add extra parameters with placeholders if needed
    for param in &extra_params {
        write!(code, ", ").unwrap();
        write!(code, "{}", generate_smart_placeholder(param)).unwrap();
    }

    writeln!(code, ");").unwrap();
    writeln!(code).unwrap();

    // Check the status code
    let returns_status = is_status_type(&func.return_type);

    if returns_status {
        // Try to find the error enum for this return type
        let success_check = if let Some(error_enum) = error_patterns
            .error_enums
            .iter()
            .find(|e| func.return_type.contains(&e.name))
        {
            if let Some(ref success_variant) = error_enum.success_variant {
                // Use the detected success variant
                format!("status == ffi::{}::{}", error_enum.name, success_variant)
            } else {
                // Fallback to 0 if no success variant detected
                "status == 0".to_string()
            }
        } else {
            // No error enum found, use 0 as default
            "status == 0".to_string()
        };

        writeln!(code, "            // Check if creation succeeded").unwrap();
        writeln!(code, "            if {} {{", success_check).unwrap();
        writeln!(code, "                if handle.is_null() {{").unwrap();
        writeln!(code, "                    Err(Error::NullPointer)").unwrap();
        writeln!(code, "                }} else {{").unwrap();
        writeln!(code, "                    Ok(Self {{ handle }})").unwrap();
        writeln!(code, "                }}").unwrap();
        writeln!(code, "            }} else {{").unwrap();
        writeln!(code, "                Err(Error::FfiError(status as i32))").unwrap();
        writeln!(code, "            }}").unwrap();
    } else {
        // No status return, just check if handle is valid
        writeln!(code, "            if handle.is_null() {{").unwrap();
        writeln!(code, "                Err(Error::NullPointer)").unwrap();
        writeln!(code, "            }} else {{").unwrap();
        writeln!(code, "                Ok(Self {{ handle }})").unwrap();
        writeln!(code, "            }}").unwrap();
    }

    writeln!(code, "        }}").unwrap();
    writeln!(code, "    }}").unwrap();
    writeln!(code).unwrap();
}

/// Generate simple constructor as fallback (when no FFI info available)
fn generate_simple_constructor(code: &mut String, func_name: &str) {
    writeln!(code, "    /// Create a new instance").unwrap();
    writeln!(code, "    // Warning: Unable to analyze function signature").unwrap();
    writeln!(code, "    #[doc(hidden)]").unwrap();
    writeln!(code, "    pub fn new() -> Result<Self, Error> {{").unwrap();
    writeln!(code, "        unsafe {{").unwrap();
    writeln!(
        code,
        "            // TODO: Implement proper constructor for {}",
        func_name
    )
    .unwrap();
    writeln!(code, "            Err(Error::Unknown)").unwrap();
    writeln!(code, "        }}").unwrap();
    writeln!(code, "    }}").unwrap();
    writeln!(code).unwrap();
}

/// Generate Drop implementation body based on destroy function signature
fn generate_drop_body(code: &mut String, func: &FfiFunction, handle_type: &str, func_name: &str) {
    let pattern = analyze_destroy_pattern(func, handle_type);

    writeln!(code, "        unsafe {{").unwrap();

    match pattern {
        DestroyPattern::TakesHandleReturnsVoid => {
            // void destroy(handle_t h) or void destroy(handle_t h, extra_params...)
            writeln!(code, "            if !self.handle.is_null() {{").unwrap();

            // Check if there are additional parameters beyond the handle
            let additional_params = generate_additional_destroy_params(func, handle_type);
            if !additional_params.is_empty() {
                writeln!(
                    code,
                    "                ffi::{}(self.handle{});",
                    func_name, additional_params
                )
                .unwrap();
            } else {
                writeln!(code, "                ffi::{}(self.handle);", func_name).unwrap();
            }
            writeln!(code, "            }}").unwrap();
        }
        DestroyPattern::TakesHandleReturnsStatus => {
            // status_t destroy(handle_t h) or status_t destroy(handle_t h, extra_params...)
            writeln!(code, "            if !self.handle.is_null() {{").unwrap();

            // Check if there are additional parameters beyond the handle
            let additional_params = generate_additional_destroy_params(func, handle_type);
            if !additional_params.is_empty() {
                writeln!(
                    code,
                    "                let _status = ffi::{}(self.handle{});",
                    func_name, additional_params
                )
                .unwrap();
            } else {
                writeln!(
                    code,
                    "                let _status = ffi::{}(self.handle);",
                    func_name
                )
                .unwrap();
            }
            writeln!(
                code,
                "                // Ignoring status in drop - can't propagate errors"
            )
            .unwrap();
            writeln!(code, "            }}").unwrap();
        }
        DestroyPattern::TakesPointerReturnsVoid => {
            // void destroy(*mut handle_t)
            writeln!(code, "            if !self.handle.is_null() {{").unwrap();
            writeln!(
                code,
                "                ffi::{}(&mut self.handle);",
                func_name
            )
            .unwrap();
            writeln!(code, "            }}").unwrap();
        }
        DestroyPattern::Unknown => {
            warn!("Unknown destroy pattern for {}", func_name);
            writeln!(
                code,
                "            // TODO: Unable to determine proper calling convention for {}",
                func_name
            )
            .unwrap();
            writeln!(code, "            // Please implement manually").unwrap();
        }
    }

    writeln!(code, "        }}").unwrap();
}

/// Generate default values for additional parameters in destroy functions
fn generate_additional_destroy_params(func: &FfiFunction, handle_type: &str) -> String {
    let mut additional_params = String::new();

    // Find the first parameter that contains the handle type, then take all parameters after it
    let mut found_handle = false;
    for param in &func.params {
        if found_handle {
            // This is a parameter after the handle - add default value
            let default_value = if param.ty.contains("c_uint") || param.ty.contains("u32") {
                "1" // For count parameters, use 1
            } else if param.ty.contains("c_int") || param.ty.contains("i32") {
                "0"
            } else if param.ty.contains("*") {
                "std::ptr::null_mut()"
            } else if param.ty.contains("bool") {
                "false"
            } else {
                "0" // Generic fallback
            };

            additional_params.push_str(&format!(", {}", default_value));
        } else if param.ty.contains(handle_type) {
            found_handle = true;
        }
    }

    additional_params
}

/// Generate simple Drop as fallback (when no FFI info available)
fn generate_simple_drop(code: &mut String, func_name: &str, handle_type: &str) {
    writeln!(code, "        unsafe {{").unwrap();
    writeln!(code, "            if !self.handle.is_null() {{").unwrap();
    writeln!(
        code,
        "                // Warning: Unable to analyze function signature"
    )
    .unwrap();
    writeln!(
        code,
        "                // Assuming pattern: void {}({})",
        func_name, handle_type
    )
    .unwrap();
    writeln!(code, "                ffi::{}(self.handle);", func_name).unwrap();
    writeln!(code, "            }}").unwrap();
    writeln!(code, "        }}").unwrap();
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

    // Use ffi:: prefix for the handle type to ensure proper scoping
    let ffi_type = format!("ffi::{}", handle.name);

    // Generate the struct with optimization attributes
    writeln!(code, "/// Wrapper for `{}`", handle.name).unwrap();
    writeln!(
        code,
        "/// Note: No automatic resource management - handle cleanup manually"
    )
    .unwrap();
    writeln!(code, "///").unwrap();
    writeln!(
        code,
        "/// This wrapper is a zero-cost abstraction with the same memory layout"
    )
    .unwrap();
    writeln!(code, "/// as the underlying handle type.").unwrap();
    writeln!(code, "#[repr(transparent)]").unwrap();
    writeln!(code, "pub struct {} {{", type_name).unwrap();
    writeln!(code, "    handle: {},", ffi_type).unwrap();
    writeln!(code, "}}").unwrap();
    writeln!(code).unwrap();

    // Generate accessor methods for raw handle access
    writeln!(code, "impl {} {{", type_name).unwrap();
    writeln!(code, "    /// Returns the raw FFI handle").unwrap();
    writeln!(code, "    #[inline]").unwrap();
    writeln!(code, "    pub fn as_raw(&self) -> {} {{", ffi_type).unwrap();
    writeln!(code, "        self.handle").unwrap();
    writeln!(code, "    }}").unwrap();
    writeln!(code).unwrap();
    writeln!(
        code,
        "    /// Returns a mutable pointer to the raw FFI handle"
    )
    .unwrap();
    writeln!(code, "    #[inline]").unwrap();
    writeln!(
        code,
        "    pub fn as_raw_mut(&mut self) -> *mut {} {{",
        ffi_type
    )
    .unwrap();
    writeln!(code, "        &mut self.handle").unwrap();
    writeln!(code, "    }}").unwrap();
    writeln!(code).unwrap();
    writeln!(code, "    /// Constructs a wrapper from a raw FFI handle").unwrap();
    writeln!(code, "    ///").unwrap();
    writeln!(code, "    /// # Safety").unwrap();
    writeln!(code, "    ///").unwrap();
    writeln!(
        code,
        "    /// The caller must ensure the handle is valid and properly initialized."
    )
    .unwrap();
    writeln!(code, "    #[inline]").unwrap();
    writeln!(
        code,
        "    pub unsafe fn from_raw(handle: {}) -> Self {{",
        ffi_type
    )
    .unwrap();
    writeln!(code, "        Self {{ handle }}").unwrap();
    writeln!(code, "    }}").unwrap();
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
        builder_name: None,
        builder_code: None,
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

/// Generate an enhanced RAII wrapper with smarter error handling and documentation
///
/// This leverages enriched context from multiple analyzers:
/// - Error Semantics: Better Result<> types with fatal/recoverable distinction
/// - Callback Analysis: Lifetime parameters for callback-accepting functions
/// - API Sequences: Method chaining hints and prerequisites
/// - Resource Limits: Capacity validation
pub fn generate_enhanced_raii_wrapper(
    handle: &HandleType,
    pair: &LifecyclePair,
    create_func: Option<&FfiFunction>,
    destroy_func: Option<&FfiFunction>,
    error_patterns: &ErrorPatterns,
    create_context: Option<&FunctionContext>,
    _lib_name: &str,
) -> RaiiWrapper {
    debug!("Generating enhanced RAII wrapper for {}", handle.name);

    let type_name = to_rust_type_name(&handle.name);
    let mut code = String::new();

    // Generate documentation using enhanced context
    if let Some(ctx) = create_context
        && let Some(desc) = &ctx.description
    {
        writeln!(code, "/// {}", desc).unwrap();
        writeln!(code, "///").unwrap();
    }

    writeln!(code, "/// Safe RAII wrapper for `{}`", handle.name).unwrap();
    writeln!(code, "///").unwrap();
    writeln!(
        code,
        "/// This wrapper provides automatic resource management with compile-time safety."
    )
    .unwrap();

    // Add thread safety info from context
    if let Some(ctx) = create_context {
        if let Some(thread_safety) = &ctx.thread_safety {
            writeln!(code, "///").unwrap();
            writeln!(code, "/// # Thread Safety").unwrap();
            writeln!(code, "///").unwrap();
            match thread_safety.safety {
                crate::analyzer::thread_safety::ThreadSafety::Safe => {
                    writeln!(
                        code,
                        "/// This type is thread-safe and can be shared across threads."
                    )
                    .unwrap();
                }
                crate::analyzer::thread_safety::ThreadSafety::Unsafe => {
                    writeln!(
                        code,
                        "/// ⚠️ This type is NOT thread-safe. Use external synchronization."
                    )
                    .unwrap();
                }
                crate::analyzer::thread_safety::ThreadSafety::Reentrant => {
                    writeln!(code, "/// This type is reentrant-safe.").unwrap();
                }
                crate::analyzer::thread_safety::ThreadSafety::RequiresSync => {
                    writeln!(
                        code,
                        "/// This type requires external synchronization for concurrent access."
                    )
                    .unwrap();
                }
                crate::analyzer::thread_safety::ThreadSafety::PerThread => {
                    writeln!(code, "/// This type maintains per-thread state.").unwrap();
                }
                crate::analyzer::thread_safety::ThreadSafety::Unknown => {
                    writeln!(
                        code,
                        "/// ⚠️ Thread safety unknown - assume not thread-safe."
                    )
                    .unwrap();
                }
            }
        }

        // Add initialization requirements
        if let Some(global_state) = &ctx.global_state
            && global_state.requires_init
        {
            writeln!(code, "///").unwrap();
            writeln!(code, "/// # Initialization").unwrap();
            writeln!(code, "///").unwrap();
            if let Some(init_fn) = &global_state.init_function {
                writeln!(
                    code,
                    "/// **Required**: Call `{}()` before creating instances.",
                    init_fn
                )
                .unwrap();
            } else {
                writeln!(code, "/// **Required**: Initialize the library before use.").unwrap();
            }
        }

        // Add API sequence prerequisites
        if let Some(sequences) = &ctx.api_sequences
            && !sequences.prerequisites.is_empty()
        {
            writeln!(code, "///").unwrap();
            writeln!(code, "/// # Prerequisites").unwrap();
            writeln!(code, "///").unwrap();
            for prereq in &sequences.prerequisites {
                writeln!(code, "/// - Must call `{}` first", prereq).unwrap();
            }
        }

        // Add resource limits
        if let Some(limits) = &ctx.resource_limits
            && !limits.limits.is_empty()
        {
            writeln!(code, "///").unwrap();
            writeln!(code, "/// # Resource Limits").unwrap();
            writeln!(code, "///").unwrap();
            for (resource, limit_infos) in &limits.limits {
                for limit_info in limit_infos {
                    if let Some(val) = limit_info.value {
                        if limit_info.is_maximum {
                            writeln!(code, "/// - Maximum {}: {}", resource, val).unwrap();
                        } else if limit_info.is_minimum {
                            writeln!(code, "/// - Minimum {}: {}", resource, val).unwrap();
                        } else if limit_info.is_recommended {
                            writeln!(code, "/// - Recommended {}: {}", resource, val).unwrap();
                        }
                    }
                }
            }
        }
    }

    // Use ffi:: prefix for the handle type to ensure proper scoping
    let ffi_type = format!("ffi::{}", handle.name);

    writeln!(code, "#[repr(transparent)]").unwrap();
    writeln!(code, "pub struct {} {{", type_name).unwrap();
    writeln!(code, "    handle: {},", ffi_type).unwrap();
    writeln!(code, "}}").unwrap();
    writeln!(code).unwrap();

    // Use ffi:: prefix for the handle type to ensure proper scoping
    let ffi_type = format!("ffi::{}", handle.name);

    // Generate impl block
    writeln!(code, "impl {} {{", type_name).unwrap();

    if let Some(create_fn) = create_func {
        generate_enhanced_constructor(
            &mut code,
            create_fn,
            &ffi_type,
            &pair.create_fn,
            error_patterns,
            create_context,
        );
    } else {
        generate_simple_constructor(&mut code, &pair.create_fn);
    }

    // Generate accessor methods
    writeln!(code).unwrap();
    writeln!(code, "    /// Returns the raw FFI handle").unwrap();
    writeln!(code, "    #[inline]").unwrap();
    writeln!(code, "    pub fn as_raw(&self) -> {} {{", ffi_type).unwrap();
    writeln!(code, "        self.handle").unwrap();
    writeln!(code, "    }}").unwrap();
    writeln!(code).unwrap();
    writeln!(
        code,
        "    /// Returns a mutable pointer to the raw FFI handle"
    )
    .unwrap();
    writeln!(code, "    #[inline]").unwrap();
    writeln!(
        code,
        "    pub fn as_raw_mut(&mut self) -> *mut {} {{",
        ffi_type
    )
    .unwrap();
    writeln!(code, "        &mut self.handle").unwrap();
    writeln!(code, "    }}").unwrap();
    writeln!(code).unwrap();
    writeln!(code, "    /// Constructs a wrapper from a raw FFI handle").unwrap();
    writeln!(code, "    ///").unwrap();
    writeln!(code, "    /// # Safety").unwrap();
    writeln!(code, "    ///").unwrap();
    writeln!(
        code,
        "    /// The caller must ensure the handle is valid and properly initialized."
    )
    .unwrap();
    writeln!(
        code,
        "    /// The wrapper will take ownership and call the destructor on drop."
    )
    .unwrap();
    writeln!(code, "    #[inline]").unwrap();
    writeln!(
        code,
        "    pub unsafe fn from_raw(handle: {}) -> Self {{",
        ffi_type
    )
    .unwrap();
    writeln!(code, "        Self {{ handle }}").unwrap();
    writeln!(code, "    }}").unwrap();

    writeln!(code, "}}").unwrap();
    writeln!(code).unwrap();

    // Generate Drop implementation
    writeln!(code, "impl Drop for {} {{", type_name).unwrap();
    writeln!(code, "    fn drop(&mut self) {{").unwrap();

    if let Some(destroy_fn) = destroy_func {
        generate_drop_body(&mut code, destroy_fn, &handle.name, &pair.destroy_fn);
    } else {
        generate_simple_drop(&mut code, &pair.destroy_fn, &handle.name);
    }

    writeln!(code, "    }}").unwrap();
    writeln!(code, "}}").unwrap();
    writeln!(code).unwrap();

    // Conditionally implement Send/Sync based on thread safety analysis
    if let Some(ctx) = create_context
        && let Some(thread_safety) = &ctx.thread_safety
    {
        match thread_safety.safety {
            crate::analyzer::thread_safety::ThreadSafety::Safe => {
                writeln!(code, "// Thread-safe - can implement Send + Sync").unwrap();
                writeln!(code, "unsafe impl Send for {} {{}}", type_name).unwrap();
                writeln!(code, "unsafe impl Sync for {} {{}}", type_name).unwrap();
            }
            crate::analyzer::thread_safety::ThreadSafety::Reentrant => {
                writeln!(code, "// Reentrant - safe to send between threads").unwrap();
                writeln!(code, "unsafe impl Send for {} {{}}", type_name).unwrap();
            }
            _ => {
                writeln!(code, "// Not thread-safe - no Send/Sync implementation").unwrap();
            }
        }
        writeln!(code).unwrap();
    }

    // Enhanced RAII wrappers do not emit the legacy builder pattern here.
    // Builder typestate generation is handled centrally by the typestate
    // generator to avoid duplicate and conflicting builder definitions.

    info!(
        "Generated enhanced RAII wrapper for {} -> {}",
        handle.name, type_name
    );

    RaiiWrapper {
        type_name,
        handle_type: handle.name.clone(),
        code,
        builder_name: None,
        builder_code: None,
    }
}

/// Generate enhanced constructor with better error handling
fn generate_enhanced_constructor(
    code: &mut String,
    func: &FfiFunction,
    handle_type: &str,
    func_name: &str,
    error_patterns: &ErrorPatterns,
    context: Option<&FunctionContext>,
) {
    let pattern = analyze_create_pattern(func, handle_type);

    writeln!(code, "    /// Create a new instance").unwrap();

    // Add context-aware documentation
    if let Some(ctx) = context {
        if let Some(desc) = &ctx.description {
            writeln!(code, "    ///").unwrap();
            writeln!(code, "    /// {}", desc).unwrap();
        }

        // Add error semantics info
        if let Some(error_sem) = &ctx.error_semantics
            && !error_sem.errors.is_empty()
        {
            writeln!(code, "    ///").unwrap();
            writeln!(code, "    /// # Errors").unwrap();
            writeln!(code, "    ///").unwrap();

            // Separate fatal and recoverable errors
            let fatal_errors: Vec<_> = error_sem
                .errors
                .iter()
                .filter(|(_, info)| info.is_fatal)
                .collect();
            let recoverable_errors: Vec<_> = error_sem
                .errors
                .iter()
                .filter(|(_, info)| !info.is_fatal)
                .collect();

            if !fatal_errors.is_empty() {
                writeln!(code, "    /// **Fatal errors** (cannot recover):").unwrap();
                for (name, info) in &fatal_errors {
                    writeln!(code, "    /// - `{}`: {}", name, info.description).unwrap();
                }
            }

            if !recoverable_errors.is_empty() {
                if !fatal_errors.is_empty() {
                    writeln!(code, "    ///").unwrap();
                }
                writeln!(code, "    /// **Recoverable errors** (can retry):").unwrap();
                for (name, info) in &recoverable_errors {
                    writeln!(code, "    /// - `{}`: {}", name, info.description).unwrap();
                    if info.is_retryable {
                        writeln!(code, "    ///   (retryable)").unwrap();
                    }
                }
            }
        }

        // Add preconditions
        if let Some(precond) = &ctx.preconditions
            && (!precond.non_null_params.is_empty() || !precond.undefined_behavior.is_empty())
        {
            writeln!(code, "    ///").unwrap();
            writeln!(code, "    /// # Preconditions").unwrap();
            writeln!(code, "    ///").unwrap();
            for param in &precond.non_null_params {
                writeln!(code, "    /// - `{}` must not be null", param).unwrap();
            }
            for ub in &precond.undefined_behavior {
                writeln!(code, "    /// - ⚠️ UB: {}", ub).unwrap();
            }
        }
    }

    writeln!(code, "    #[inline]").unwrap();

    match pattern {
        CreatePattern::ReturnsHandle => {
            generate_enhanced_returns_handle_constructor(code, func, func_name, context);
        }
        CreatePattern::OutputParameter => {
            generate_enhanced_output_param_constructor(
                code,
                func,
                handle_type,
                func_name,
                error_patterns,
                context,
            );
        }
        CreatePattern::Unknown => {
            warn!("Unknown create pattern for {}, using fallback", func_name);
            generate_simple_constructor(code, func_name);
        }
    }
}

/// Generate enhanced constructor for return-handle pattern
fn generate_enhanced_returns_handle_constructor(
    code: &mut String,
    func: &FfiFunction,
    func_name: &str,
    context: Option<&FunctionContext>,
) {
    let has_params = !func.params.is_empty();

    // Check resource limits for parameter validation
    let mut needs_validation = false;
    if let Some(ctx) = context
        && let Some(limits) = &ctx.resource_limits
        && !limits.limits.is_empty()
    {
        needs_validation = true;
    }

    if has_params {
        writeln!(
            code,
            "    // Note: Function {} requires parameters:",
            func_name
        )
        .unwrap();
        for param in &func.params {
            writeln!(code, "    //   - {}: {}", param.name, param.ty).unwrap();
        }
    }

    writeln!(code, "    pub fn new() -> Result<Self, Error> {{").unwrap();

    if needs_validation {
        writeln!(code, "        // Validate resource limits").unwrap();
        if let Some(ctx) = context
            && let Some(limits) = &ctx.resource_limits
        {
            for (resource, limit_infos) in &limits.limits {
                for limit_info in limit_infos {
                    if let Some(val) = limit_info.value
                        && limit_info.is_maximum
                    {
                        writeln!(code, "        // TODO: Check {} <= {}", resource, val).unwrap();
                    }
                }
            }
        }
    }

    writeln!(code, "        unsafe {{").unwrap();

    if has_params {
        write!(code, "            let handle = ffi::{}(", func_name).unwrap();
        for (i, param) in func.params.iter().enumerate() {
            if i > 0 {
                write!(code, ", ").unwrap();
            }
            write!(code, "{}", generate_smart_placeholder(param)).unwrap();
        }
        writeln!(code, ");").unwrap();
    } else {
        writeln!(code, "            let handle = ffi::{}();", func_name).unwrap();
    }

    writeln!(code, "            if handle.is_null() {{").unwrap();
    writeln!(code, "                Err(Error::NullPointer)").unwrap();
    writeln!(code, "            }} else {{").unwrap();
    writeln!(code, "                Ok(Self {{ handle }})").unwrap();
    writeln!(code, "            }}").unwrap();
    writeln!(code, "        }}").unwrap();
    writeln!(code, "    }}").unwrap();
    writeln!(code).unwrap();
}

/// Generate enhanced constructor for output-parameter pattern
fn generate_enhanced_output_param_constructor(
    code: &mut String,
    func: &FfiFunction,
    _handle_type: &str,
    func_name: &str,
    error_patterns: &ErrorPatterns,
    context: Option<&FunctionContext>,
) {
    let extra_params: Vec<_> = func.params.iter().skip(1).collect();
    let has_extra_params = !extra_params.is_empty();

    if has_extra_params {
        writeln!(code, "    // Parameters required for {}:", func_name).unwrap();
        for param in &extra_params {
            writeln!(code, "    //   - {}: {}", param.name, param.ty).unwrap();
        }
    }

    writeln!(code, "    pub fn new() -> Result<Self, Error> {{").unwrap();
    writeln!(code, "        unsafe {{").unwrap();
    writeln!(code, "            let mut handle = std::ptr::null_mut();").unwrap();

    write!(
        code,
        "            let status = ffi::{}(&mut handle",
        func_name
    )
    .unwrap();

    for param in &extra_params {
        write!(code, ", ").unwrap();
        write!(code, "{}", generate_smart_placeholder(param)).unwrap();
    }

    writeln!(code, ");").unwrap();
    writeln!(code).unwrap();

    let returns_status = is_status_type(&func.return_type);

    if returns_status {
        // Use error patterns to find success value
        let success_check = if let Some(error_enum) = error_patterns
            .error_enums
            .iter()
            .find(|e| func.return_type.contains(&e.name))
        {
            if let Some(ref success_variant) = error_enum.success_variant {
                format!("status == ffi::{}::{}", error_enum.name, success_variant)
            } else {
                "status == 0".to_string()
            }
        } else {
            "status == 0".to_string()
        };

        writeln!(code, "            if {} {{", success_check).unwrap();
        writeln!(code, "                if handle.is_null() {{").unwrap();
        writeln!(code, "                    Err(Error::NullPointer)").unwrap();
        writeln!(code, "                }} else {{").unwrap();
        writeln!(code, "                    Ok(Self {{ handle }})").unwrap();
        writeln!(code, "                }}").unwrap();
        writeln!(code, "            }} else {{").unwrap();

        // Provide better error context from error semantics
        if let Some(ctx) = context
            && ctx.error_semantics.is_some()
        {
            writeln!(code, "                // Enhanced error context available").unwrap();
        }

        writeln!(code, "                Err(Error::FfiError(status as i32))").unwrap();
        writeln!(code, "            }}").unwrap();
    } else {
        writeln!(code, "            if handle.is_null() {{").unwrap();
        writeln!(code, "                Err(Error::NullPointer)").unwrap();
        writeln!(code, "            }} else {{").unwrap();
        writeln!(code, "                Ok(Self {{ handle }})").unwrap();
        writeln!(code, "            }}").unwrap();
    }

    writeln!(code, "        }}").unwrap();
    writeln!(code, "    }}").unwrap();
    writeln!(code).unwrap();
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

    #[test]
    fn test_to_rust_type_name_with_underscores() {
        // Leading underscores are stripped, treating rest as single word unless separated by _
        assert_eq!(
            to_rust_type_name("_leading_underscore"),
            "LeadingUnderscore"
        );
        assert_eq!(to_rust_type_name("trailing_"), "Trailing");
        assert_eq!(
            to_rust_type_name("multiple___underscores"),
            "MultipleUnderscores"
        );
    }

    #[test]
    fn test_to_rust_type_name_acronyms() {
        assert_eq!(to_rust_type_name("http_client"), "HttpClient");
        assert_eq!(to_rust_type_name("cuda_context"), "CudaContext");
        assert_eq!(to_rust_type_name("ssl_session"), "SslSession");
    }

    #[test]
    fn test_to_rust_type_name_numbers() {
        assert_eq!(to_rust_type_name("type_v2"), "TypeV2");
        assert_eq!(to_rust_type_name("vector3d"), "Vector3d");
        assert_eq!(to_rust_type_name("matrix4x4"), "Matrix4x4");
    }

    #[test]
    fn test_to_rust_type_name_empty_and_special() {
        // Edge cases
        assert_eq!(to_rust_type_name("a"), "A");
        assert_eq!(to_rust_type_name("_"), "");
    }
}
