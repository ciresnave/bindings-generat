//! Builder typestate pattern generator
//!
//! Generates compile-time enforced builder patterns using Rust's type system
//! to ensure functions are called in the correct order.

use crate::ffi::FfiInfo;
use serde::{Deserialize, Serialize};
use tracing::info;

/// Builder typestate analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuilderTypestateAnalysis {
    /// Builder definitions
    pub builders: Vec<BuilderDefinition>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuilderDefinition {
    /// Builder name
    pub name: String,
    /// Type being built
    pub target_type: String,
    /// State machine
    pub states: Vec<BuilderState>,
    /// Required configuration steps
    pub required_steps: Vec<ConfigStep>,
    /// Optional configuration steps
    pub optional_steps: Vec<ConfigStep>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuilderState {
    /// State name
    pub name: String,
    /// Is this a valid terminal state?
    pub is_terminal: bool,
    /// Methods available in this state
    pub available_methods: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigStep {
    /// Step name
    pub name: String,
    /// Function to call
    pub function: String,
    /// Dependencies (must be called before this)
    pub depends_on: Vec<String>,
}

/// Analyzes and generates builder typestates
pub struct BuilderTypestateGenerator;

impl BuilderTypestateGenerator {
    pub fn new() -> Self {
        Self
    }

    /// Analyze FFI and identify builder opportunities
    pub fn analyze(&self, ffi_info: &FfiInfo) -> BuilderTypestateAnalysis {
        use std::collections::HashMap;

        // Look for builder patterns: create/init -> config* -> build/destroy
        let create_funcs: Vec<_> = ffi_info
            .functions
            .iter()
            .filter(|f| {
                let name_lower = f.name.to_lowercase();
                name_lower.contains("create") || name_lower.contains("init")
            })
            .collect();

        // Use a map to deduplicate builders targeting the same type. Multiple
        // create functions that produce the same target should result in a
        // single typestate builder with multiple available methods.
        let mut builders_map: HashMap<String, BuilderDefinition> = HashMap::new();

        // Look for builder patterns: create/init -> config* -> build/destroy
        let create_funcs: Vec<_> = ffi_info
            .functions
            .iter()
            .filter(|f| {
                let name_lower = f.name.to_lowercase();
                name_lower.contains("create") || name_lower.contains("init")
            })
            .collect();

        for create_func in create_funcs {
            // Determine the builder target type
            let mut target_type = None;
            if let Some(param) = create_func.params.iter().find(|p| p.is_pointer && p.is_mut) {
                let base = Self::extract_base_type_name(&param.ty);
                if !base.is_empty() {
                    target_type = Some(crate::generator::wrappers::to_rust_type_name(&base));
                }
            }

            if target_type.is_none() && create_func.return_type.contains("Handle") {
                let base = Self::extract_base_type_name(&create_func.return_type);
                if !base.is_empty() {
                    target_type = Some(crate::generator::wrappers::to_rust_type_name(&base));
                }
            }

            if target_type.is_none() {
                // Fallback: derive a reasonable Rust type name from the function name
                target_type = Some(crate::generator::wrappers::to_rust_type_name(
                    &create_func.name,
                ));
            }

            let sanitized = target_type.unwrap_or_else(|| "GeneratedType".to_string());
            let builder_name = format!("{}Builder", sanitized);

            if let Some(existing) = builders_map.get_mut(&builder_name) {
                // Add this create function to the initial state's available methods
                if let Some(initial_state) = existing.states.get_mut(0) {
                    if !initial_state.available_methods.contains(&create_func.name) {
                        initial_state
                            .available_methods
                            .push(create_func.name.clone());
                    }
                }
                // Also record as an additional required step for traceability
                existing.required_steps.push(ConfigStep {
                    name: create_func.name.clone(),
                    function: create_func.name.clone(),
                    depends_on: Vec::new(),
                });
            } else {
                builders_map.insert(
                    builder_name.clone(),
                    BuilderDefinition {
                        name: builder_name.clone(),
                        target_type: sanitized.clone(),
                        states: vec![
                            BuilderState {
                                name: "Initial".to_string(),
                                is_terminal: false,
                                available_methods: vec![create_func.name.clone()],
                            },
                            BuilderState {
                                name: "Built".to_string(),
                                is_terminal: true,
                                available_methods: Vec::new(),
                            },
                        ],
                        required_steps: vec![ConfigStep {
                            name: "create".to_string(),
                            function: create_func.name.clone(),
                            depends_on: Vec::new(),
                        }],
                        optional_steps: Vec::new(),
                    },
                );
            }
        }

        let builders: Vec<BuilderDefinition> = builders_map.into_values().collect();
        info!("Generated {} builder typestates", builders.len());

        BuilderTypestateAnalysis { builders }
    }

    /// Convert identifier pieces into a PascalCase Rust type name
    fn extract_base_type_name(s: &str) -> String {
        // Remove common pointer tokens and qualifiers, handling spaced variants
        let mut base = s.to_string();
        for pat in &["*const", "* const", "*mut", "* mut"] {
            base = base.replace(pat, "");
        }
        // Replace remaining '*' with whitespace and strip common qualifiers
        base = base.replace('*', " ");
        base = base.replace("const ", "");
        base = base.replace("mut ", "");

        // Split on non-alphanumeric (and underscore) and take the last meaningful token
        let token = base
            .split(|c: char| !c.is_alphanumeric() && c != '_')
            .filter(|p| !p.is_empty())
            .last()
            .map(|s| s.to_string())
            .unwrap_or_default();

        token
    }

    fn sanitize_type_name(s: &str) -> String {
        let base = Self::extract_base_type_name(s);
        if base.is_empty() {
            "GeneratedType".to_string()
        } else {
            crate::generator::wrappers::to_rust_type_name(&base)
        }
    }

    /// Generate Rust code for typestate builder
    pub fn generate_builder_code(&self, builder: &BuilderDefinition) -> String {
        let mut code = String::new();

        code.push_str(&format!(
            "// Typestate builder for {}\n",
            builder.target_type
        ));
        code.push_str(&format!(
            "// Ensures compile-time enforcement of builder order\n\n"
        ));

        // State marker types (namespaced by builder to avoid collisions)
        code.push_str("// State marker types\n");
        for state in &builder.states {
            let marker = format!("{}{}", builder.name, state.name);
            code.push_str(&format!("pub struct {};\n", marker));
        }
        code.push_str("\n");

        // Generate builder struct with phantom state
        code.push_str(&format!("pub struct {}<State> {{\n", builder.name));
        code.push_str("    _state: std::marker::PhantomData<State>,\n");
        code.push_str("    // Builder fields\n");
        code.push_str("}\n\n");

        // Generate initial constructor
        let initial_marker = format!("{}{}", builder.name, builder.states[0].name);
        code.push_str(&format!("impl {}<{}> {{\n", builder.name, initial_marker));
        code.push_str("    pub fn new() -> Self {\n");
        code.push_str("        Self {\n");
        code.push_str("            _state: std::marker::PhantomData,\n");
        code.push_str("        }\n");
        code.push_str("    }\n");
        code.push_str("}\n\n");

        // Generate state transition methods
        for (idx, state) in builder.states.iter().enumerate() {
            if idx < builder.states.len() - 1 {
                let next_state = &builder.states[idx + 1];
                let current_marker = format!("{}{}", builder.name, state.name);
                let next_marker = format!("{}{}", builder.name, next_state.name);
                code.push_str(&format!("impl {}<{}> {{\n", builder.name, current_marker));

                for method in &state.available_methods {
                    let method_name = self.to_snake_case(method);
                    code.push_str(&format!(
                        "    pub fn {}(self) -> {}<{}> {{\n",
                        method_name, builder.name, next_marker
                    ));
                    code.push_str(&format!("        // Call FFI: {}()\n", method));
                    // Instantiate the generic struct literal using turbofish
                    // to avoid `<`/`>` being parsed as comparison operators.
                    code.push_str(&format!("        {}::<{}> {{\n", builder.name, next_marker));
                    code.push_str("            _state: std::marker::PhantomData,\n");
                    code.push_str("        }\n");
                    code.push_str("    }\n");
                }

                code.push_str("}\n\n");
            }
        }

        // Generate terminal build method
        if let Some(terminal_state) = builder.states.iter().find(|s| s.is_terminal) {
            let terminal_marker = format!("{}{}", builder.name, terminal_state.name);
            code.push_str(&format!("impl {}<{}> {{\n", builder.name, terminal_marker));
            code.push_str(&format!(
                "    pub fn build(self) -> {} {{\n",
                builder.target_type
            ));
            code.push_str(&format!(
                "        // Finalize and return {}\n",
                builder.target_type
            ));
            code.push_str(&format!(
                "        todo!(\"Implement {}  construction\")\n",
                builder.target_type
            ));
            code.push_str("    }\n");
            code.push_str("}\n\n");
        }

        // Generate usage example
        code.push_str("// Usage example:\n");
        code.push_str("// let obj = ");
        code.push_str(&format!("{}::new()", builder.name));
        for step in &builder.required_steps {
            code.push_str(&format!("\n//     .{}()", self.to_snake_case(&step.name)));
        }
        code.push_str("\n//     .build();\n");

        code
    }

    /// Convert CamelCase to snake_case
    fn to_snake_case(&self, s: &str) -> String {
        let mut result = String::new();
        for (i, ch) in s.chars().enumerate() {
            if ch.is_uppercase() {
                if i > 0 {
                    result.push('_');
                }
                result.push(ch.to_lowercase().next().unwrap());
            } else {
                result.push(ch);
            }
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ffi::FfiFunction;

    #[test]
    fn test_builder_detection() {
        let generator = BuilderTypestateGenerator::new();

        let ffi_info = FfiInfo {
            functions: vec![FfiFunction {
                name: "createHandle".to_string(),
                return_type: "Handle*".to_string(),
                params: Vec::new(),
                docs: None,
            }],
            types: Vec::new(),
            enums: Vec::new(),
            constants: Vec::new(),
            dependencies: Vec::new(),
            opaque_types: Vec::new(),
            type_aliases: std::collections::HashMap::new(),
        };

        let analysis = generator.analyze(&ffi_info);
        assert!(!analysis.builders.is_empty());
    }

    #[test]
    fn test_code_generation() {
        let generator = BuilderTypestateGenerator::new();

        let builder = BuilderDefinition {
            name: "HandleBuilder".to_string(),
            target_type: "Handle".to_string(),
            states: vec![
                BuilderState {
                    name: "Initial".to_string(),
                    is_terminal: false,
                    available_methods: vec!["create".to_string()],
                },
                BuilderState {
                    name: "Built".to_string(),
                    is_terminal: true,
                    available_methods: vec![],
                },
            ],
            required_steps: vec![],
            optional_steps: vec![],
        };

        let code = generator.generate_builder_code(&builder);

        assert!(code.contains("HandleBuilder"));
        assert!(code.contains("Initial"));
        assert!(code.contains("Built"));
    }

    #[test]
    fn test_to_snake_case() {
        let generator = BuilderTypestateGenerator::new();

        assert_eq!(generator.to_snake_case("createHandle"), "create_handle");
        assert_eq!(generator.to_snake_case("setOption"), "set_option");
        assert_eq!(generator.to_snake_case("build"), "build");
    }
}
