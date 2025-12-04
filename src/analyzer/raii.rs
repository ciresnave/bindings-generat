use crate::ffi::{FfiFunction, FfiInfo};
use std::collections::{HashMap, HashSet};
use tracing::{debug, info};

/// Detected RAII patterns in the FFI
#[derive(Debug, Clone)]
pub struct RaiiPatterns {
    pub handle_types: Vec<HandleType>,
    pub lifecycle_pairs: Vec<LifecyclePair>,
}

#[derive(Debug, Clone)]
pub struct HandleType {
    pub name: String,
    pub is_pointer: bool,
    pub create_functions: Vec<String>,
    pub destroy_functions: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct LifecyclePair {
    pub handle_type: String,
    pub create_fn: String,
    pub destroy_fn: String,
    pub confidence: f32, // 0.0 to 1.0
}

/// Recursively resolve type aliases to their underlying type
///
/// Example: cudnnHandle_t -> *mut cudnnContext
fn resolve_type_alias(
    type_name: &str,
    type_aliases: &HashMap<String, String>,
    visited: &mut HashSet<String>,
) -> String {
    // Prevent infinite recursion from circular type aliases
    if visited.contains(type_name) {
        debug!("Circular type alias detected for: {}", type_name);
        return type_name.to_string();
    }

    visited.insert(type_name.to_string());

    // Look up the type alias
    if let Some(target_type) = type_aliases.get(type_name) {
        // Extract the base type name if it's a pointer type
        let base = extract_base_type(target_type);

        // If the target is also an alias, recursively resolve it
        if type_aliases.contains_key(&base) {
            return resolve_type_alias(&base, type_aliases, visited);
        }

        // Return the resolved target type
        return target_type.clone();
    }

    // Not an alias, return as-is
    type_name.to_string()
}
impl Default for RaiiPatterns {
    fn default() -> Self {
        Self::new()
    }
}

impl RaiiPatterns {
    pub fn new() -> Self {
        Self {
            handle_types: Vec::new(),
            lifecycle_pairs: Vec::new(),
        }
    }
}

/// Detect RAII patterns (create/destroy, init/cleanup, open/close pairs)
pub fn detect_raii_patterns(ffi_info: &FfiInfo) -> RaiiPatterns {
    info!(
        "Detecting RAII patterns in {} functions",
        ffi_info.functions.len()
    );

    let mut patterns = RaiiPatterns::new();

    // Step 1: Identify potential handle types (opaque pointers)
    let mut handle_types = identify_handle_types(ffi_info);

    // Step 2: Find create/destroy function pairs
    let lifecycle_pairs = find_lifecycle_pairs(&ffi_info.functions, &handle_types);

    // Step 3: Associate functions with handle types
    for pair in &lifecycle_pairs {
        if let Some(handle) = handle_types.iter_mut().find(|h| h.name == pair.handle_type) {
            if !handle.create_functions.contains(&pair.create_fn) {
                handle.create_functions.push(pair.create_fn.clone());
            }
            if !handle.destroy_functions.contains(&pair.destroy_fn) {
                handle.destroy_functions.push(pair.destroy_fn.clone());
            }
        }
    }

    patterns.handle_types = handle_types;
    patterns.lifecycle_pairs = lifecycle_pairs;

    info!(
        "Found {} handle types and {} lifecycle pairs",
        patterns.handle_types.len(),
        patterns.lifecycle_pairs.len()
    );

    patterns
}

fn identify_handle_types(ffi_info: &FfiInfo) -> Vec<HandleType> {
    let mut handles = Vec::new();

    // Check opaque types
    for opaque_type in &ffi_info.opaque_types {
        debug!("Found opaque type: {}", opaque_type);
        handles.push(HandleType {
            name: opaque_type.clone(),
            is_pointer: true,
            create_functions: Vec::new(),
            destroy_functions: Vec::new(),
        });
    }

    // Check type aliases that resolve to pointers
    for alias_name in ffi_info.type_aliases.keys() {
        // Resolve the alias to its underlying type
        let mut visited = HashSet::new();
        let resolved = resolve_type_alias(alias_name, &ffi_info.type_aliases, &mut visited);

        // Check if the resolved type is a pointer
        if resolved.contains("*") {
            debug!(
                "Type alias {} resolves to pointer type: {}",
                alias_name, resolved
            );

            // Skip if already added (might be in opaque_types)
            if handles.iter().any(|h| h.name == *alias_name) {
                continue;
            }

            // Add as handle type
            handles.push(HandleType {
                name: alias_name.clone(),
                is_pointer: true,
                create_functions: Vec::new(),
                destroy_functions: Vec::new(),
            });
        }
    }

    // Check for types that are used as pointers in functions
    let mut pointer_types: HashMap<String, usize> = HashMap::new();
    for func in &ffi_info.functions {
        for param in &func.params {
            if param.is_pointer {
                let base_type = extract_base_type(&param.ty);

                // Resolve if it's an alias
                let mut visited = HashSet::new();
                let resolved = resolve_type_alias(&base_type, &ffi_info.type_aliases, &mut visited);

                // Only count if the resolved type is also a pointer (handle pattern)
                if resolved.contains("*") {
                    *pointer_types.entry(base_type).or_insert(0) += 1;
                }
            }
        }

        // Check return type
        if func.return_type.contains("*") {
            let base_type = extract_base_type(&func.return_type);

            // Resolve if it's an alias
            let mut visited = HashSet::new();
            let resolved = resolve_type_alias(&base_type, &ffi_info.type_aliases, &mut visited);

            // Only count if the resolved type is also a pointer (handle pattern)
            if resolved.contains("*") {
                *pointer_types.entry(base_type).or_insert(0) += 1;
            }
        }
    }

    // Types used frequently as pointers are likely handles
    // But filter out primitive types, standard library types, and non-handle types
    let primitive_types = [
        "i8",
        "i16",
        "i32",
        "i64",
        "i128",
        "isize",
        "u8",
        "u16",
        "u32",
        "u64",
        "u128",
        "usize",
        "f32",
        "f64",
        "bool",
        "char",
        "c_void",
        "c_char",
        "c_int",
        "c_uint",
        "c_long",
        "c_ulong",
        "c_short",
        "c_ushort",
        "c_longlong",
        "c_ulonglong",
        "c_float",
        "c_double",
    ];

    // Common handle-like keywords in type names
    let handle_keywords = [
        "handle",
        "descriptor",
        "context",
        "stream",
        "event",
        "pool",
        "graph",
        "node",
        "texture",
        "surface",
        "array",
        "memory",
        "mem",
        "buffer",
        "queue",
        "device",
        "resource",
        "object",
        "instance",
        "session",
        "connection",
        "socket",
    ];

    for (type_name, count) in pointer_types {
        // Skip if the type is empty or a known primitive
        if type_name.is_empty() || primitive_types.contains(&type_name.as_str()) {
            continue;
        }

        // Skip if the type contains :: (it's already a Rust path, not a C type)
        if type_name.contains("::") {
            debug!("Skipping Rust path type: {}", type_name);
            continue;
        }

        // For types ending in _t, only consider them if they have handle-like keywords
        if type_name.ends_with("_t") {
            let lower_name = type_name.to_lowercase();
            let is_handle = handle_keywords
                .iter()
                .any(|keyword| lower_name.contains(keyword));
            if !is_handle {
                debug!("Skipping non-handle type ending in _t: {}", type_name);
                continue;
            }
        }

        // Skip if already added
        if handles.iter().any(|h| h.name == type_name) {
            continue;
        }

        // Must be used at least twice as a pointer
        if count >= 2 {
            debug!(
                "Identified potential handle type: {} (used {} times)",
                type_name, count
            );
            handles.push(HandleType {
                name: type_name,
                is_pointer: true,
                create_functions: Vec::new(),
                destroy_functions: Vec::new(),
            });
        }
    }

    handles
}

fn find_lifecycle_pairs(
    functions: &[FfiFunction],
    handle_types: &[HandleType],
) -> Vec<LifecyclePair> {
    let mut pairs = Vec::new();

    // Common patterns for create/destroy pairs
    let create_patterns = ["create", "new", "init", "open", "alloc", "make", "start"];
    let destroy_patterns = [
        "destroy", "delete", "free", "close", "cleanup", "finish", "stop", "release",
    ];

    // Exclude words that indicate operations rather than destruction
    // e.g., "cudaGraphAddMemFreeNode" has "free" but it's "add" + "free node", not "free" the handle
    let exclude_patterns = ["add", "insert", "append", "push", "enqueue", "get", "set"];

    // Group functions by their base name (without create/destroy prefix)
    for handle in handle_types {
        let create_funcs: Vec<_> = functions
            .iter()
            .filter(|f| {
                let lower_name = f.name.to_lowercase();
                // Check if function name suggests creation
                let has_create_pattern = create_patterns.iter().any(|p| lower_name.contains(p));

                if !has_create_pattern {
                    return false;
                }

                // Exclude compound operations like "AddCreate" or "GetCreate"
                let has_exclude = exclude_patterns.iter().any(|p| lower_name.contains(p));
                if has_exclude {
                    return false;
                }

                // Check if function returns a pointer to this handle type
                let returns_handle = f.return_type.contains(&handle.name);

                // Check if function has an output parameter (first param is *mut handle_type)
                // This is the most common C pattern: status_t create(*mut handle_t out)
                let has_output_param = f
                    .params
                    .first()
                    .map(|p| {
                        // Check if it's a mutable pointer to this handle type
                        let is_mut_ptr = p.ty.contains("*mut") || p.ty.contains("* mut");
                        let is_handle_type = p.ty.contains(&handle.name);
                        is_mut_ptr && is_handle_type
                    })
                    .unwrap_or(false);

                returns_handle || has_output_param
            })
            .collect();

        let destroy_funcs: Vec<_> = functions
            .iter()
            .filter(|f| {
                let lower_name = f.name.to_lowercase();
                // Check if function takes this handle type as parameter
                let takes_handle = f.params.iter().any(|p| p.ty.contains(&handle.name));
                // Check if function name suggests destruction
                let has_destroy_pattern = destroy_patterns.iter().any(|p| lower_name.contains(p));

                // Exclude compound operations like "AddFree" or "GetDestroy"
                let has_exclude = exclude_patterns.iter().any(|p| lower_name.contains(p));

                takes_handle && has_destroy_pattern && !has_exclude
            })
            .collect();

        debug!(
            "Handle type {}: {} create funcs, {} destroy funcs",
            handle.name,
            create_funcs.len(),
            destroy_funcs.len()
        );

        // Try to pair create and destroy functions
        for create_func in &create_funcs {
            for destroy_func in &destroy_funcs {
                let confidence = calculate_pair_confidence(create_func, destroy_func);
                if confidence > 0.5 {
                    pairs.push(LifecyclePair {
                        handle_type: handle.name.clone(),
                        create_fn: create_func.name.clone(),
                        destroy_fn: destroy_func.name.clone(),
                        confidence,
                    });
                }
            }
        }
    }

    pairs
}

fn calculate_pair_confidence(create_func: &FfiFunction, destroy_func: &FfiFunction) -> f32 {
    let mut confidence: f32 = 0.5; // Base confidence

    // Remove common prefixes/suffixes to find the core name
    // NOTE: Pass original names, not lowercased - extract_core_name needs camelCase boundaries!
    let create_core = extract_core_name(&create_func.name);
    let destroy_core = extract_core_name(&destroy_func.name);

    debug!(
        "calculate_pair_confidence: {} -> '{}' vs {} -> '{}'",
        create_func.name, create_core, destroy_func.name, destroy_core
    );

    // If core names match, high confidence
    if create_core == destroy_core {
        confidence += 0.3;
        debug!("  Core names match! confidence now {}", confidence);
    }

    // If names are similar (edit distance), medium confidence
    if names_similar(&create_core, &destroy_core) {
        confidence += 0.15;
    }

    // If parameter count matches (destroy typically takes what create returns), boost confidence
    if destroy_func.params.len() == 1 {
        confidence += 0.1;
    }

    confidence.min(1.0_f32)
}

fn extract_base_type(type_str: &str) -> String {
    type_str
        .replace("*", "")
        .replace("const", "")
        .replace("mut", "")
        .trim()
        .to_string()
}

fn extract_core_name(name: &str) -> String {
    let lower = name.to_lowercase();

    // List of lifecycle verbs to strip (both create and destroy verbs)
    // IMPORTANT: Order matters! Longer verbs (like "malloc") must come before shorter ones (like "alloc")
    // to prevent partial matches (e.g., "malloc" should match before "alloc")
    let lifecycle_verbs = [
        "create",
        "initialize",
        "allocate",
        "malloc", // Must come before "alloc" to match "cudaMallocArray" correctly
        "destroy",
        "release",
        "dispose",
        "cleanup",
        "finalize",
        "new",
        "init",
        "open",
        "alloc",
        "make",
        "start",
        "begin",
        "delete",
        "free",
        "close",
        "finish",
        "stop",
        "end",
    ];

    // Try to find and remove a lifecycle verb from the name
    // Handle both camelCase (cudnnCreatePooling) and snake_case (cudnn_create_pooling)
    for verb in &lifecycle_verbs {
        // Check for snake_case: prefix_verb_ or _verb_suffix
        let snake_prefix = format!("_{}_", verb);
        let snake_start = format!("{}_", verb);
        let snake_end = format!("_{}", verb);

        if lower.contains(&snake_prefix) {
            // e.g., "cudnn_create_pooling" -> "cudnn_pooling"
            return name.replace(&snake_prefix, "_");
        } else if lower.starts_with(&snake_start) {
            // e.g., "create_pooling" -> "pooling"
            return name[snake_start.len()..].to_string();
        } else if lower.ends_with(&snake_end) {
            // e.g., "pooling_create" -> "pooling"
            return name[..name.len() - snake_end.len()].to_string();
        }

        // Check for camelCase: PrefixVerbSuffix
        // Find the verb in lowercase version
        if let Some(verb_pos) = lower.find(verb) {
            // Make sure it's at a word boundary (preceded by lowercase or start, followed by uppercase or end)
            let is_word_start = verb_pos == 0
                || lower
                    .chars()
                    .nth(verb_pos - 1)
                    .map(|c| c.is_lowercase() || c == '_')
                    .unwrap_or(false);

            let verb_end = verb_pos + verb.len();
            let is_word_end = verb_end >= name.len()
                || name
                    .chars()
                    .nth(verb_end)
                    .map(|c| c.is_uppercase() || c == '_' || c.is_numeric())
                    .unwrap_or(false);

            if is_word_start && is_word_end {
                // Found a valid verb boundary - remove it
                // e.g., "cudnnCreatePoolingDescriptor"
                //   verb_pos = 5, verb = "create" (len 6)
                //   result: "cudnn" + "PoolingDescriptor"
                let before = &name[..verb_pos];
                let after = &name[verb_end..];

                // Keep the prefix, drop the verb, keep the suffix
                return format!("{}{}", before, after);
            }
        }
    }

    // No lifecycle verb found, return as-is
    name.to_string()
}

fn names_similar(name1: &str, name2: &str) -> bool {
    // Simple similarity check - share significant prefix or suffix
    if name1.is_empty() || name2.is_empty() {
        return false;
    }

    let min_len = name1.len().min(name2.len());
    if min_len < 3 {
        return false;
    }

    // Check if they share at least 60% of characters
    let shared_prefix_len = name1
        .chars()
        .zip(name2.chars())
        .take_while(|(a, b)| a == b)
        .count();

    shared_prefix_len as f32 / min_len as f32 > 0.6
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_core_name() {
        // snake_case
        assert_eq!(extract_core_name("create_context"), "context");
        assert_eq!(extract_core_name("destroy_context"), "context");
        assert_eq!(extract_core_name("context_create"), "context");
        assert_eq!(extract_core_name("context_destroy"), "context");
        
        // camelCase (cudnn-style)
        // The key insight: after stripping lifecycle verbs, the names should match
        assert_eq!(extract_core_name("cudnnCreatePoolingDescriptor"), "cudnnPoolingDescriptor");
        assert_eq!(extract_core_name("cudnnDestroyPoolingDescriptor"), "cudnnPoolingDescriptor");
        assert_eq!(extract_core_name("cudnnCreateTensorDescriptor"), "cudnnTensorDescriptor");
        assert_eq!(extract_core_name("cudnnDestroyTensorDescriptor"), "cudnnTensorDescriptor");
        
        // CUDA malloc/free pairs
        println!("cudaMallocArray -> '{}'", extract_core_name("cudaMallocArray"));
        println!("cudaFreeArray -> '{}'", extract_core_name("cudaFreeArray"));
        assert_eq!(extract_core_name("cudaMallocArray"), "cudaArray");
        assert_eq!(extract_core_name("cudaFreeArray"), "cudaArray");
        assert_eq!(extract_core_name("cudaMallocHost"), "cudaHost");
        assert_eq!(extract_core_name("cudaFreeHost"), "cudaHost");
        
        // Verify they match each other
        assert_eq!(
            extract_core_name("cudnnCreatePoolingDescriptor"),
            extract_core_name("cudnnDestroyPoolingDescriptor")
        );
        assert_eq!(
            extract_core_name("cudaMallocArray"),
            extract_core_name("cudaFreeArray")
        );
        
        // Mixed case
        assert_eq!(extract_core_name("cuGraphCreate"), "cuGraph");
        assert_eq!(extract_core_name("cuGraphDestroy"), "cuGraph");
        
        // Should NOT match partial words
        assert_eq!(extract_core_name("creative"), "creative"); // Not "ative"
        assert_eq!(extract_core_name("recreation"), "recreation"); // Not "reation"
    }

    #[test]
    fn test_names_similar() {
        assert!(names_similar("context", "context"));
        assert!(names_similar("context", "cont"));
        assert!(!names_similar("foo", "bar"));
    }
}
