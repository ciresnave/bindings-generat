use crate::ffi::{FfiFunction, FfiInfo};
use std::collections::HashMap;
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

    // Check for types that are used as pointers in functions
    let mut pointer_types: HashMap<String, usize> = HashMap::new();
    for func in &ffi_info.functions {
        for param in &func.params {
            if param.is_pointer {
                let base_type = extract_base_type(&param.ty);
                *pointer_types.entry(base_type).or_insert(0) += 1;
            }
        }

        // Check return type
        if func.return_type.contains("*") {
            let base_type = extract_base_type(&func.return_type);
            *pointer_types.entry(base_type).or_insert(0) += 1;
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
            let is_handle = handle_keywords.iter().any(|keyword| lower_name.contains(keyword));
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

    // Group functions by their base name (without create/destroy prefix)
    for handle in handle_types {
        let create_funcs: Vec<_> = functions
            .iter()
            .filter(|f| {
                let lower_name = f.name.to_lowercase();
                // Check if function returns a pointer to this handle type
                let returns_handle = f.return_type.contains(&handle.name);
                // Check if function name suggests creation
                let has_create_pattern = create_patterns.iter().any(|p| lower_name.contains(p));
                returns_handle && has_create_pattern
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
                takes_handle && has_destroy_pattern
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

    let create_lower = create_func.name.to_lowercase();
    let destroy_lower = destroy_func.name.to_lowercase();

    // Remove common prefixes/suffixes to find the core name
    let create_core = extract_core_name(&create_lower);
    let destroy_core = extract_core_name(&destroy_lower);

    // If core names match, high confidence
    if create_core == destroy_core {
        confidence += 0.3;
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
    let mut core = name.to_string();

    // Remove common prefixes
    let prefixes = [
        "create_", "new_", "init_", "open_", "alloc_", "make_", "start_", "destroy_", "delete_",
        "free_", "close_", "cleanup_", "finish_", "stop_", "release_",
    ];
    for prefix in &prefixes {
        if core.starts_with(prefix) {
            core = core[prefix.len()..].to_string();
            break;
        }
    }

    // Remove common suffixes
    let suffixes = [
        "_create", "_new", "_init", "_open", "_alloc", "_make", "_start", "_destroy", "_delete",
        "_free", "_close", "_cleanup", "_finish", "_stop", "_release",
    ];
    for suffix in &suffixes {
        if core.ends_with(suffix) {
            core = core[..core.len() - suffix.len()].to_string();
            break;
        }
    }

    core
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
        assert_eq!(extract_core_name("create_context"), "context");
        assert_eq!(extract_core_name("destroy_context"), "context");
        assert_eq!(extract_core_name("context_create"), "context");
        assert_eq!(extract_core_name("context_destroy"), "context");
    }

    #[test]
    fn test_names_similar() {
        assert!(names_similar("context", "context"));
        assert!(names_similar("context", "cont"));
        assert!(!names_similar("foo", "bar"));
    }
}
