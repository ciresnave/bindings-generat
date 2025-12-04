use regex::Regex;
use tracing::debug;

/// Convert C-style function names to idiomatic Rust naming
pub fn to_idiomatic_rust_name(c_name: &str, library_prefix: Option<&str>) -> String {
    let mut name = c_name.to_string();

    // Step 1: Remove library prefix if provided
    if let Some(prefix) = library_prefix {
        let prefix_lower = prefix.to_lowercase();
        let name_lower = name.to_lowercase();

        // Try exact prefix match
        if name_lower.starts_with(&prefix_lower) {
            name = name[prefix.len()..].to_string();
            // Remove leading underscore if present
            if name.starts_with('_') {
                name = name[1..].to_string();
            }
        }
    }

    // Step 2: Handle number suffixes BEFORE camel case conversion
    // This ensures "3D" is treated as a unit
    name = normalize_number_suffixes(&name);

    // Step 3: Convert camelCase/PascalCase to snake_case with proper word breaking
    name = camel_to_snake(&name);

    // Step 4: Handle consecutive uppercase acronyms
    name = normalize_acronyms(&name);

    // Step 5: Clean up any double underscores
    while name.contains("__") {
        name = name.replace("__", "_");
    }

    // Step 6: Remove leading/trailing underscores
    name = name.trim_matches('_').to_string();

    // Step 7: Ensure it's lowercase
    name = name.to_lowercase();

    debug!("Converted '{}' to '{}'", c_name, name);
    name
}

/// Convert camelCase or PascalCase to snake_case
fn camel_to_snake(input: &str) -> String {
    let mut result = String::new();
    let mut prev_was_lower = false;
    let mut prev_was_upper = false;

    for (i, ch) in input.chars().enumerate() {
        if ch.is_uppercase() {
            // Add underscore before uppercase if:
            // 1. Previous was lowercase (camelCase boundary)
            // 2. Next char is lowercase (end of acronym like "XMLParser" -> "xml_parser")
            if prev_was_lower
                || (prev_was_upper
                    && i + 1 < input.len()
                    && input.chars().nth(i + 1).map_or(false, |c| c.is_lowercase()))
            {
                if !result.is_empty() {
                    result.push('_');
                }
            }
            result.push(ch);
            prev_was_lower = false;
            prev_was_upper = true;
        } else if ch.is_lowercase() {
            result.push(ch);
            prev_was_lower = true;
            prev_was_upper = false;
        } else if ch.is_numeric() {
            // Add underscore before number if previous was a letter
            if prev_was_lower || prev_was_upper {
                if !result.is_empty() {
                    result.push('_');
                }
            }
            result.push(ch);
            prev_was_lower = false;
            prev_was_upper = false;
        } else {
            // Underscores and other chars
            result.push(ch);
            prev_was_lower = false;
            prev_was_upper = false;
        }
    }

    result
}

/// Normalize number suffixes by lowercasing D after digits
/// This converts "3D" → "3d", "4d" → "4d" so they're treated as units
fn normalize_number_suffixes(input: &str) -> String {
    // Lowercase both uppercase D and lowercase d patterns after digits
    // Handles: 3D, 4d, 2D, etc.
    let re = Regex::new(r"([0-9])[Dd]\b").unwrap();
    let result = re.replace_all(input, "${1}d");

    result.to_string()
}

/// Normalize consecutive uppercase acronyms
fn normalize_acronyms(input: &str) -> String {
    // This is handled by camel_to_snake, but we can add specific patterns
    // For example: "GPUAPI" -> "GPU_API"
    input.to_string()
}

/// Get common library prefix from function name
pub fn detect_library_prefix(func_name: &str) -> Option<String> {
    // Common patterns
    let prefixes = vec![
        "cuda", "cudnn", "cublas", "cusparse", "cufft", "gl", "glfw", "sdl", "zlib", "png", "jpeg",
        "ssl", "crypto", "sqlite",
    ];

    let name_lower = func_name.to_lowercase();
    for prefix in prefixes {
        if name_lower.starts_with(prefix) {
            return Some(prefix.to_string());
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cuda_function_names() {
        assert_eq!(
            to_idiomatic_rust_name("cudaMemcpy3D", Some("cuda")),
            "memcpy_3d"
        );
        assert_eq!(
            to_idiomatic_rust_name("cudaMemcpy2D", Some("cuda")),
            "memcpy_2d"
        );
        assert_eq!(
            to_idiomatic_rust_name("cudaMalloc3DArray", Some("cuda")),
            "malloc_3d_array"
        );
    }

    #[test]
    fn test_cudnn_function_names() {
        assert_eq!(
            to_idiomatic_rust_name("cudnnSetTensor4dDescriptor", Some("cudnn")),
            "set_tensor_4d_descriptor"
        );
        assert_eq!(
            to_idiomatic_rust_name("cudnnCreateTensorDescriptor", Some("cudnn")),
            "create_tensor_descriptor"
        );
    }

    #[test]
    fn test_graph_functions() {
        assert_eq!(
            to_idiomatic_rust_name("cudaGraphAddMemFreeNode", Some("cuda")),
            "graph_add_mem_free_node"
        );
        assert_eq!(
            to_idiomatic_rust_name("cudaGraphAddMemcpyNode", Some("cuda")),
            "graph_add_memcpy_node"
        );
    }

    #[test]
    fn test_acronyms() {
        assert_eq!(
            to_idiomatic_rust_name("cudaGetDevicePropertiesNDRange", Some("cuda")),
            "get_device_properties_nd_range"
        );
        assert_eq!(
            to_idiomatic_rust_name("GLFWCreateWindow", Some("glfw")),
            "create_window"
        );
    }

    #[test]
    fn test_no_prefix() {
        assert_eq!(
            to_idiomatic_rust_name("someFunction3D", None),
            "some_function_3d"
        );
        assert_eq!(
            to_idiomatic_rust_name("Array2DCreate", None),
            "array_2d_create"
        );
    }

    #[test]
    fn test_detect_prefix() {
        assert_eq!(
            detect_library_prefix("cudaMemcpy"),
            Some("cuda".to_string())
        );
        assert_eq!(
            detect_library_prefix("cudnnCreate"),
            Some("cudnn".to_string())
        );
        assert_eq!(
            detect_library_prefix("glCreateShader"),
            Some("gl".to_string())
        );
        assert_eq!(detect_library_prefix("someFunction"), None);
    }

    #[test]
    fn test_camel_to_snake() {
        assert_eq!(camel_to_snake("camelCase"), "camel_Case");
        assert_eq!(camel_to_snake("PascalCase"), "Pascal_Case");
        assert_eq!(camel_to_snake("XMLParser"), "XML_Parser");
        assert_eq!(camel_to_snake("HTMLElement"), "HTML_Element");
    }

    #[test]
    fn test_number_suffixes() {
        // normalize_number_suffixes just lowercases the D/d
        // The underscore is added later by camel_to_snake
        assert_eq!(normalize_number_suffixes("malloc3D"), "malloc3d");
        assert_eq!(normalize_number_suffixes("Array4d"), "Array4d");
        assert_eq!(normalize_number_suffixes("tensor2D"), "tensor2d");
        assert_eq!(
            normalize_number_suffixes("create4dTensor"),
            "create4dTensor"
        );
    }

    #[test]
    fn test_comprehensive_naming() {
        // Test the full pipeline
        assert_eq!(
            to_idiomatic_rust_name("cudnnSetTensor4dDescriptor", Some("cudnn")),
            "set_tensor_4d_descriptor"
        );
        assert_eq!(
            to_idiomatic_rust_name("cudaMemcpy3D", Some("cuda")),
            "memcpy_3d"
        );
        assert_eq!(
            to_idiomatic_rust_name("cudaGraphAddMemFreeNode", Some("cuda")),
            "graph_add_mem_free_node"
        );
        assert_eq!(
            to_idiomatic_rust_name("cudnnGetTensorNdDescriptor", Some("cudnn")),
            "get_tensor_nd_descriptor"
        );
    }
}
