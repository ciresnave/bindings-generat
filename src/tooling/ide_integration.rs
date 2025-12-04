//! IDE integration and developer tooling support
//!
//! This module provides features for better IDE experience:
//! - rust-analyzer hints and documentation
//! - IDE-friendly code structure
//! - Quick fixes and code actions
//! - Symbol navigation support

use crate::ffi::parser::{FfiFunction, FfiInfo};
use std::collections::HashMap;

/// IDE hint types
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IdeHint {
    /// Documentation hint
    Documentation(String),
    /// Type hint
    TypeHint(String),
    /// Parameter hint
    ParameterHint { name: String, ty: String },
    /// Return type hint
    ReturnTypeHint(String),
    /// Safety hint
    SafetyHint(String),
    /// Performance hint
    PerformanceHint(String),
}

/// IDE metadata for a function
#[derive(Debug, Clone)]
pub struct FunctionMetadata {
    /// Function name
    pub name: String,
    /// Documentation
    pub documentation: Vec<String>,
    /// Parameter hints
    pub parameter_hints: Vec<IdeHint>,
    /// Return type hint
    pub return_hint: Option<IdeHint>,
    /// Safety information
    pub safety_notes: Vec<String>,
    /// Usage examples
    pub examples: Vec<String>,
    /// Related functions
    pub see_also: Vec<String>,
}

impl FunctionMetadata {
    /// Create metadata from FFI function
    pub fn from_ffi_function(func: &FfiFunction) -> Self {
        let mut metadata = Self {
            name: func.name.clone(),
            documentation: vec![],
            parameter_hints: vec![],
            return_hint: None,
            safety_notes: vec![],
            examples: vec![],
            see_also: vec![],
        };

        // Extract docs
        if let Some(docs) = &func.docs {
            metadata.documentation.push(docs.clone());
        }

        // Add parameter hints
        for param in &func.params {
            metadata.parameter_hints.push(IdeHint::ParameterHint {
                name: param.name.clone(),
                ty: param.ty.clone(),
            });

            // Add safety notes for pointers
            if param.is_pointer {
                if param.is_mut {
                    metadata.safety_notes.push(format!(
                        "Parameter `{}` is a mutable pointer - ensure exclusive access",
                        param.name
                    ));
                } else {
                    metadata.safety_notes.push(format!(
                        "Parameter `{}` is a pointer - ensure it's valid and non-null",
                        param.name
                    ));
                }
            }
        }

        // Add return type hint
        if !func.return_type.is_empty() && func.return_type != "void" {
            metadata.return_hint = Some(IdeHint::ReturnTypeHint(func.return_type.clone()));
        }

        metadata
    }

    /// Generate rust-analyzer compatible documentation
    pub fn to_doc_comment(&self) -> String {
        let mut doc = String::new();

        // Main documentation
        for line in &self.documentation {
            doc.push_str(&format!("/// {}\n", line));
        }

        // Parameters section
        if !self.parameter_hints.is_empty() {
            doc.push_str("///\n/// # Parameters\n");
            for hint in &self.parameter_hints {
                if let IdeHint::ParameterHint { name, ty } = hint {
                    doc.push_str(&format!("/// - `{}`: {}\n", name, ty));
                }
            }
        }

        // Return section
        if let Some(IdeHint::ReturnTypeHint(ty)) = &self.return_hint {
            doc.push_str("///\n/// # Returns\n");
            doc.push_str(&format!("/// `{}`\n", ty));
        }

        // Safety section
        if !self.safety_notes.is_empty() {
            doc.push_str("///\n/// # Safety\n");
            for note in &self.safety_notes {
                doc.push_str(&format!("/// - {}\n", note));
            }
        }

        // Examples section
        if !self.examples.is_empty() {
            doc.push_str("///\n/// # Examples\n");
            for example in &self.examples {
                doc.push_str("/// ```\n");
                for line in example.lines() {
                    doc.push_str(&format!("/// {}\n", line));
                }
                doc.push_str("/// ```\n");
            }
        }

        // See also section
        if !self.see_also.is_empty() {
            doc.push_str("///\n/// # See Also\n");
            for related in &self.see_also {
                doc.push_str(&format!("/// - [`{}`]\n", related));
            }
        }

        doc
    }
}

/// IDE integration generator
pub struct IdeIntegration {
    /// Function metadata
    functions: HashMap<String, FunctionMetadata>,
    /// Module structure for navigation
    modules: HashMap<String, Vec<String>>,
}

impl IdeIntegration {
    /// Create new IDE integration
    pub fn new() -> Self {
        Self {
            functions: HashMap::new(),
            modules: HashMap::new(),
        }
    }

    /// Analyze FFI info for IDE integration
    pub fn analyze(&mut self, ffi_info: &FfiInfo) {
        // Generate metadata for each function
        for func in &ffi_info.functions {
            let metadata = FunctionMetadata::from_ffi_function(func);
            self.functions.insert(func.name.clone(), metadata);
        }

        // Organize into modules by prefix
        self.organize_modules(ffi_info);

        // Add cross-references
        self.add_cross_references();
    }

    /// Organize functions into logical modules
    fn organize_modules(&mut self, ffi_info: &FfiInfo) {
        for func in &ffi_info.functions {
            // Extract prefix (e.g., "cuda" from "cudaMalloc")
            let module_name = Self::extract_module_name(&func.name);
            self.modules
                .entry(module_name)
                .or_insert_with(Vec::new)
                .push(func.name.clone());
        }
    }

    /// Extract module name from function name
    fn extract_module_name(func_name: &str) -> String {
        // Common patterns: cuda_malloc, cuMalloc, SDL_CreateWindow
        if let Some(underscore_pos) = func_name.find('_') {
            func_name[..underscore_pos].to_lowercase()
        } else {
            // CamelCase: extract lowercase prefix
            let lowercase_chars: String = func_name
                .chars()
                .take_while(|c| c.is_lowercase())
                .collect();
            if !lowercase_chars.is_empty() {
                lowercase_chars
            } else {
                "general".to_string()
            }
        }
    }

    /// Add cross-references between related functions
    fn add_cross_references(&mut self) {
        let function_names: Vec<String> = self.functions.keys().cloned().collect();

        for func_name in &function_names {
            if let Some(metadata) = self.functions.get_mut(func_name) {
                // Find related functions (same module, similar names)
                let module = Self::extract_module_name(func_name);

                if let Some(module_funcs) = self.modules.get(&module) {
                    for related in module_funcs {
                        if related != func_name && Self::are_related(func_name, related) {
                            metadata.see_also.push(related.clone());
                        }
                    }
                }
            }
        }
    }

    /// Check if two function names are related
    fn are_related(name1: &str, name2: &str) -> bool {
        // Related if they share significant prefix or pattern
        let name1_lower = name1.to_lowercase();
        let name2_lower = name2.to_lowercase();

        // Check for create/destroy pairs
        if (name1_lower.contains("create") && name2_lower.contains("destroy"))
            || (name1_lower.contains("destroy") && name2_lower.contains("create"))
        {
            return true;
        }

        // Check for init/cleanup pairs
        if (name1_lower.contains("init") && name2_lower.contains("cleanup"))
            || (name1_lower.contains("cleanup") && name2_lower.contains("init"))
        {
            return true;
        }

        // Check for get/set pairs
        if name1_lower.contains("get") && name2_lower.contains("set") {
            let name1_suffix = name1_lower.trim_start_matches("get");
            let name2_suffix = name2_lower.trim_start_matches("set");
            if name1_suffix == name2_suffix {
                return true;
            }
        }

        false
    }

    /// Get metadata for a function
    pub fn get_metadata(&self, function_name: &str) -> Option<&FunctionMetadata> {
        self.functions.get(function_name)
    }

    /// Generate rust-analyzer workspace configuration
    pub fn generate_rust_analyzer_config(&self) -> String {
        format!(
            r#"{{
    "rust-analyzer.check.command": "clippy",
    "rust-analyzer.check.extraArgs": ["--", "-W", "clippy::all"],
    "rust-analyzer.cargo.features": "all",
    "rust-analyzer.inlayHints.parameterHints.enable": true,
    "rust-analyzer.inlayHints.typeHints.enable": true,
    "rust-analyzer.hover.actions.enable": true,
    "rust-analyzer.hover.documentation.enable": true,
    "rust-analyzer.completion.autoimport.enable": true,
    "rust-analyzer.diagnostics.enable": true
}}
"#
        )
    }

    /// Generate VSCode tasks.json for common operations
    pub fn generate_vscode_tasks(&self) -> String {
        r#"{
    "version": "2.0.0",
    "tasks": [
        {
            "label": "cargo build",
            "type": "shell",
            "command": "cargo",
            "args": ["build"],
            "group": {
                "kind": "build",
                "isDefault": true
            }
        },
        {
            "label": "cargo test",
            "type": "shell",
            "command": "cargo",
            "args": ["test"],
            "group": {
                "kind": "test",
                "isDefault": true
            }
        },
        {
            "label": "cargo clippy",
            "type": "shell",
            "command": "cargo",
            "args": ["clippy", "--", "-W", "clippy::all"]
        },
        {
            "label": "cargo doc",
            "type": "shell",
            "command": "cargo",
            "args": ["doc", "--open"]
        }
    ]
}"#
        .to_string()
    }

    /// Generate module documentation index
    pub fn generate_module_index(&self) -> String {
        let mut output = String::from("# Module Index\n\n");

        let mut module_names: Vec<_> = self.modules.keys().collect();
        module_names.sort();

        for module_name in module_names {
            if let Some(functions) = self.modules.get(module_name) {
                output.push_str(&format!("## `{}` module\n\n", module_name));
                output.push_str(&format!("{} functions:\n\n", functions.len()));

                let mut sorted_funcs = functions.clone();
                sorted_funcs.sort();

                for func in sorted_funcs {
                    output.push_str(&format!("- [`{}`]\n", func));
                }
                output.push_str("\n");
            }
        }

        output
    }
}

impl Default for IdeIntegration {
    fn default() -> Self {
        Self::new()
    }
}

/// Generate IDE-friendly prelude module
pub fn generate_prelude(common_imports: &[String]) -> String {
    let mut output = String::from(
        r#"//! Prelude module with commonly used items
//!
//! ```
//! use my_bindings::prelude::*;
//! ```

"#,
    );

    for import in common_imports {
        output.push_str(&format!("pub use crate::{};\n", import));
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ffi::parser::FfiParam;

    #[test]
    fn test_function_metadata_creation() {
        let func = FfiFunction {
            name: "test_func".to_string(),
            params: vec![FfiParam {
                name: "ptr".to_string(),
                ty: "int".to_string(),
                is_pointer: true,
                is_mut: false,
            }],
            return_type: "void".to_string(),
            docs: Some("Test function".to_string()),
        };

        let metadata = FunctionMetadata::from_ffi_function(&func);
        assert_eq!(metadata.name, "test_func");
        assert_eq!(metadata.documentation.len(), 1);
        assert!(!metadata.safety_notes.is_empty());
    }

    #[test]
    fn test_doc_comment_generation() {
        let mut metadata = FunctionMetadata {
            name: "example".to_string(),
            documentation: vec!["Example function".to_string()],
            parameter_hints: vec![IdeHint::ParameterHint {
                name: "x".to_string(),
                ty: "i32".to_string(),
            }],
            return_hint: Some(IdeHint::ReturnTypeHint("i32".to_string())),
            safety_notes: vec!["Safe to call".to_string()],
            examples: vec![],
            see_also: vec![],
        };

        let doc = metadata.to_doc_comment();
        assert!(doc.contains("Example function"));
        assert!(doc.contains("Parameters"));
        assert!(doc.contains("`x`: i32"));
        assert!(doc.contains("Returns"));
        assert!(doc.contains("Safety"));
    }

    #[test]
    fn test_extract_module_name() {
        assert_eq!(
            IdeIntegration::extract_module_name("cuda_malloc"),
            "cuda"
        );
        assert_eq!(IdeIntegration::extract_module_name("cuMalloc"), "cu");
        assert_eq!(
            IdeIntegration::extract_module_name("SDL_CreateWindow"),
            "sdl"
        );
    }

    #[test]
    fn test_are_related() {
        assert!(IdeIntegration::are_related(
            "create_context",
            "destroy_context"
        ));
        assert!(IdeIntegration::are_related("init_lib", "cleanup_lib"));
        assert!(IdeIntegration::are_related("get_value", "set_value"));
        assert!(!IdeIntegration::are_related("foo", "bar"));
    }

    #[test]
    fn test_ide_integration() {
        let ffi_info = FfiInfo {
            functions: vec![
                FfiFunction {
                    name: "test_create".to_string(),
                    params: vec![],
                    return_type: "Handle".to_string(),
                    docs: None,
                },
                FfiFunction {
                    name: "test_destroy".to_string(),
                    params: vec![FfiParam {
                        name: "handle".to_string(),
                        ty: "Handle".to_string(),
                        is_pointer: true,
                        is_mut: false,
                    }],
                    return_type: "void".to_string(),
                    docs: None,
                },
            ],
            types: vec![],
            enums: vec![],
            constants: vec![],
            opaque_types: vec![],
            dependencies: vec![],
            type_aliases: HashMap::new(),
        };

        let mut ide = IdeIntegration::new();
        ide.analyze(&ffi_info);

        assert_eq!(ide.functions.len(), 2);
        assert!(ide.get_metadata("test_create").is_some());

        // Check cross-references were added
        let create_metadata = ide.get_metadata("test_create").unwrap();
        assert!(create_metadata.see_also.contains(&"test_destroy".to_string()));
    }

    #[test]
    fn test_rust_analyzer_config() {
        let ide = IdeIntegration::new();
        let config = ide.generate_rust_analyzer_config();
        assert!(config.contains("rust-analyzer"));
        assert!(config.contains("clippy"));
    }

    #[test]
    fn test_vscode_tasks() {
        let ide = IdeIntegration::new();
        let tasks = ide.generate_vscode_tasks();
        assert!(tasks.contains("cargo build"));
        assert!(tasks.contains("cargo test"));
        assert!(tasks.contains("cargo clippy"));
    }

    #[test]
    fn test_module_index() {
        let mut ide = IdeIntegration::new();
        ide.modules.insert(
            "cuda".to_string(),
            vec!["cuda_malloc".to_string(), "cuda_free".to_string()],
        );

        let index = ide.generate_module_index();
        assert!(index.contains("cuda"));
        assert!(index.contains("cuda_malloc"));
        assert!(index.contains("2 functions"));
    }

    #[test]
    fn test_prelude_generation() {
        let imports = vec!["Error".to_string(), "Handle".to_string()];
        let prelude = generate_prelude(&imports);
        assert!(prelude.contains("pub use crate::Error"));
        assert!(prelude.contains("pub use crate::Handle"));
        assert!(prelude.contains("Prelude module"));
    }
}
