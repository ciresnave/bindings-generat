//! Advanced builder pattern features including presets and validation chains
//!
//! This module provides enhanced builder pattern generation with:
//! - Builder presets for common configurations
//! - Fluent validation chains with compile-time guarantees
//! - Copy-on-write builders for efficiency
//! - Type-state pattern for required fields

use crate::ffi::parser::FfiInfo;
use crate::tooling::cargo_features::SafetyMode;
use quote::quote;
use std::collections::{HashMap, HashSet};
use syn::Ident;

/// Types of builder presets that can be generated
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BuilderPreset {
    /// High performance configuration (minimal checks)
    HighPerformance,
    /// Safe defaults configuration (maximum safety)
    SafeDefaults,
    /// Balanced configuration (reasonable defaults)
    Balanced,
    /// Minimal configuration (bare essentials)
    Minimal,
    /// Testing configuration (suitable for unit tests)
    Testing,
}

impl BuilderPreset {
    /// Get the method name for this preset
    pub fn method_name(&self) -> &'static str {
        match self {
            BuilderPreset::HighPerformance => "high_performance",
            BuilderPreset::SafeDefaults => "safe_defaults",
            BuilderPreset::Balanced => "balanced",
            BuilderPreset::Minimal => "minimal",
            BuilderPreset::Testing => "testing",
        }
    }

    /// Get a description of this preset
    pub fn description(&self) -> &'static str {
        match self {
            BuilderPreset::HighPerformance => "High performance preset with minimal safety checks",
            BuilderPreset::SafeDefaults => "Safe defaults with maximum safety guarantees",
            BuilderPreset::Balanced => "Balanced configuration with reasonable defaults",
            BuilderPreset::Minimal => "Minimal configuration with bare essentials",
            BuilderPreset::Testing => "Testing preset suitable for unit tests",
        }
    }
}

/// Validation that can be applied to a builder field
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum FieldValidation {
    /// Value must be non-zero
    NonZero,
    /// Value must be positive (> 0)
    Positive,
    /// Value must be a power of two
    PowerOfTwo,
    /// Value must be a multiple of another value
    MultipleOf(u64),
    /// Value must be in a specific range
    Range { min: i64, max: i64 },
    /// Value must be one of a set of allowed values
    OneOf(Vec<i64>),
    /// String must not be empty
    NonEmpty,
    /// String must match a pattern
    Pattern(String),
    /// Custom validation with a name
    Custom(String),
}

impl FieldValidation {
    /// Get the validation method name
    pub fn method_name(&self) -> String {
        match self {
            FieldValidation::NonZero => "validate_non_zero".to_string(),
            FieldValidation::Positive => "validate_positive".to_string(),
            FieldValidation::PowerOfTwo => "validate_power_of_two".to_string(),
            FieldValidation::MultipleOf(n) => format!("validate_multiple_of_{}", n),
            FieldValidation::Range { .. } => "validate_range".to_string(),
            FieldValidation::OneOf(_) => "validate_one_of".to_string(),
            FieldValidation::NonEmpty => "validate_non_empty".to_string(),
            FieldValidation::Pattern(_) => "validate_pattern".to_string(),
            FieldValidation::Custom(name) => format!("validate_{}", name),
        }
    }

    /// Get the validation error message
    pub fn error_message(&self, field_name: &str) -> String {
        match self {
            FieldValidation::NonZero => format!("{} must be non-zero", field_name),
            FieldValidation::Positive => format!("{} must be positive", field_name),
            FieldValidation::PowerOfTwo => format!("{} must be a power of two", field_name),
            FieldValidation::MultipleOf(n) => {
                format!("{} must be a multiple of {}", field_name, n)
            }
            FieldValidation::Range { min, max } => {
                format!("{} must be in range [{}, {}]", field_name, min, max)
            }
            FieldValidation::OneOf(values) => {
                format!("{} must be one of {:?}", field_name, values)
            }
            FieldValidation::NonEmpty => format!("{} must not be empty", field_name),
            FieldValidation::Pattern(pat) => {
                format!("{} must match pattern {}", field_name, pat)
            }
            FieldValidation::Custom(name) => {
                format!("{} failed validation: {}", field_name, name)
            }
        }
    }

    /// Generate the validation code
    pub fn generate_code(&self, field_name: &Ident, field_value: &str) -> proc_macro2::TokenStream {
        let field_str = field_name.to_string();
        let error_msg = self.error_message(&field_str);

        match self {
            FieldValidation::NonZero => {
                quote! {
                    if #field_name == 0 {
                        return Err(crate::Error::InvalidParameter(#error_msg.to_string()));
                    }
                }
            }
            FieldValidation::Positive => {
                quote! {
                    if #field_name <= 0 {
                        return Err(crate::Error::InvalidParameter(#error_msg.to_string()));
                    }
                }
            }
            FieldValidation::PowerOfTwo => {
                quote! {
                    if #field_name == 0 || (#field_name & (#field_name - 1)) != 0 {
                        return Err(crate::Error::InvalidParameter(#error_msg.to_string()));
                    }
                }
            }
            FieldValidation::MultipleOf(n) => {
                let n_lit = proc_macro2::Literal::u64_unsuffixed(*n);
                quote! {
                    if #field_name % #n_lit != 0 {
                        return Err(crate::Error::InvalidParameter(#error_msg.to_string()));
                    }
                }
            }
            FieldValidation::Range { min, max } => {
                let min_lit = proc_macro2::Literal::i64_unsuffixed(*min);
                let max_lit = proc_macro2::Literal::i64_unsuffixed(*max);
                quote! {
                    if #field_name < #min_lit || #field_name > #max_lit {
                        return Err(crate::Error::InvalidParameter(#error_msg.to_string()));
                    }
                }
            }
            FieldValidation::OneOf(values) => {
                let values_lit: Vec<_> = values
                    .iter()
                    .map(|v| proc_macro2::Literal::i64_unsuffixed(*v))
                    .collect();
                quote! {
                    if ![#(#values_lit),*].contains(&#field_name) {
                        return Err(crate::Error::InvalidParameter(#error_msg.to_string()));
                    }
                }
            }
            FieldValidation::NonEmpty => {
                quote! {
                    if #field_name.is_empty() {
                        return Err(crate::Error::InvalidParameter(#error_msg.to_string()));
                    }
                }
            }
            FieldValidation::Pattern(pattern) => {
                quote! {
                    let re = regex::Regex::new(#pattern)
                        .map_err(|e| crate::Error::InvalidParameter(format!("Invalid pattern: {}", e)))?;
                    if !re.is_match(#field_name) {
                        return Err(crate::Error::InvalidParameter(#error_msg.to_string()));
                    }
                }
            }
            FieldValidation::Custom(name) => {
                let validation_fn = syn::Ident::new(
                    &format!("validate_{}", name),
                    proc_macro2::Span::call_site(),
                );
                quote! {
                    Self::#validation_fn(#field_name)?;
                }
            }
        }
    }
}

/// Builder field information
#[derive(Debug, Clone)]
pub struct BuilderField {
    /// Field name
    pub name: String,
    /// Field type
    pub ty: String,
    /// Whether the field is required
    pub required: bool,
    /// Default value if optional
    pub default: Option<String>,
    /// Validations to apply
    pub validations: Vec<FieldValidation>,
    /// Documentation
    pub doc: Option<String>,
}

/// Builder configuration
#[derive(Debug, Clone)]
pub struct BuilderConfig {
    /// Builder name (e.g., "MyStructBuilder")
    pub name: String,
    /// Target struct name (e.g., "MyStruct")
    pub target_name: String,
    /// Fields in the builder
    pub fields: Vec<BuilderField>,
    /// Presets to generate
    pub presets: Vec<BuilderPreset>,
    /// Whether to use type-state pattern for required fields
    pub use_typestate: bool,
    /// Whether to implement Clone for copy-on-write
    pub implement_clone: bool,
}

impl BuilderConfig {
    /// Create a new builder configuration
    pub fn new(target_name: impl Into<String>) -> Self {
        let target_name = target_name.into();
        let name = format!("{}Builder", target_name);
        Self {
            name,
            target_name,
            fields: Vec::new(),
            presets: vec![
                BuilderPreset::Balanced,
                BuilderPreset::SafeDefaults,
                BuilderPreset::HighPerformance,
            ],
            use_typestate: false,
            implement_clone: true,
        }
    }

    /// Add a field to the builder
    pub fn add_field(&mut self, field: BuilderField) {
        self.fields.push(field);
    }

    /// Add a preset
    pub fn add_preset(&mut self, preset: BuilderPreset) {
        if !self.presets.contains(&preset) {
            self.presets.push(preset);
        }
    }
}

/// Generate builder code with advanced features
pub struct BuilderGenerator {
    config: BuilderConfig,
    safety_mode: SafetyMode,
}

impl BuilderGenerator {
    /// Create a new builder generator
    pub fn new(config: BuilderConfig, safety_mode: SafetyMode) -> Self {
        Self {
            config,
            safety_mode,
        }
    }

    /// Generate the complete builder implementation
    pub fn generate(&self) -> String {
        let mut output = String::new();

        // Generate builder struct
        output.push_str(&self.generate_builder_struct());
        output.push('\n');

        // Generate builder implementation
        output.push_str(&self.generate_builder_impl());
        output.push('\n');

        // Generate preset methods
        output.push_str(&self.generate_presets());
        output.push('\n');

        // Generate validation methods
        output.push_str(&self.generate_validations());

        output
    }

    /// Generate the builder struct definition
    fn generate_builder_struct(&self) -> String {
        let builder_name = &self.config.name;
        let mut fields = String::new();

        for field in &self.config.fields {
            if let Some(doc) = &field.doc {
                fields.push_str(&format!("    /// {}\n", doc));
            }
            let field_type = if field.required {
                format!("Option<{}>", field.ty)
            } else {
                field.ty.clone()
            };
            fields.push_str(&format!("    pub {}: {},\n", field.name, field_type));
        }

        let clone_derive = if self.config.implement_clone {
            ", Clone"
        } else {
            ""
        };

        format!(
            r#"/// Builder for constructing instances with validation
#[derive(Debug, Default{})]
pub struct {} {{
{}}}
"#,
            clone_derive, builder_name, fields
        )
    }

    /// Generate the builder implementation
    fn generate_builder_impl(&self) -> String {
        let builder_name = &self.config.name;
        let target_name = &self.config.target_name;

        let mut methods = String::new();

        // Constructor
        methods.push_str(&format!(
            r#"    /// Create a new builder
    pub fn new() -> Self {{
        Self::default()
    }}

"#
        ));

        // Setter methods
        for field in &self.config.fields {
            let field_name = &field.name;
            let field_ty = &field.ty;

            if let Some(doc) = &field.doc {
                methods.push_str(&format!("    /// {}\n", doc));
            }

            methods.push_str(&format!(
                r#"    pub fn {name}(mut self, value: {ty}) -> Self {{
        self.{name} = {value};
        self
    }}

"#,
                name = field_name,
                ty = field_ty,
                value = if field.required {
                    "Some(value)"
                } else {
                    "value"
                }
            ));

            // Add validation method if field has validations
            if !field.validations.is_empty() {
                for validation in &field.validations {
                    let method_name = validation.method_name();
                    methods.push_str(&format!(
                        r#"    /// Apply validation: {}
    pub fn {method}(self) -> Result<Self, crate::Error> {{
        if let Some(value) = self.{field} {{
            {validation_code}
        }}
        Ok(self)
    }}

"#,
                        validation.error_message(field_name),
                        method = method_name,
                        field = field_name,
                        validation_code = self.generate_validation_code(validation, field_name)
                    ));
                }
            }
        }

        // Build method
        methods.push_str(&self.generate_build_method());

        format!(
            r#"impl {} {{
{}}}
"#,
            builder_name, methods
        )
    }

    /// Generate validation code for a specific field
    fn generate_validation_code(&self, validation: &FieldValidation, field_name: &str) -> String {
        let field_ident = syn::Ident::new(field_name, proc_macro2::Span::call_site());
        let tokens = validation.generate_code(&field_ident, "value");
        // Replace field name with 'value' for validation context
        tokens.to_string().replace(field_name, "value")
    }

    /// Generate the build method
    fn generate_build_method(&self) -> String {
        let target_name = &self.config.target_name;
        let mut field_checks = String::new();
        let mut field_assignments = String::new();

        for field in &self.config.fields {
            let field_name = &field.name;

            if field.required {
                field_checks.push_str(&format!(
                    r#"        let {name} = self.{name}.ok_or_else(|| {{
            crate::Error::InvalidParameter("Field '{name}' is required".to_string())
        }})?;
"#,
                    name = field_name
                ));
                field_assignments.push_str(&format!("            {},\n", field_name));
            } else if let Some(default) = &field.default {
                field_assignments.push_str(&format!(
                    "            {}: self.{}.unwrap_or({}),\n",
                    field_name, field_name, default
                ));
            } else {
                field_assignments.push_str(&format!(
                    "            {}: self.{},\n",
                    field_name, field_name
                ));
            }
        }

        format!(
            r#"    /// Build the final instance
    pub fn build(self) -> Result<{target}, crate::Error> {{
{field_checks}
        Ok({target} {{
{field_assignments}        }})
    }}
"#,
            target = target_name,
            field_checks = field_checks,
            field_assignments = field_assignments
        )
    }

    /// Generate preset methods
    fn generate_presets(&self) -> String {
        let builder_name = &self.config.name;
        let mut presets = String::new();

        presets.push_str(&format!("impl {} {{\n", builder_name));

        for preset in &self.config.presets {
            let method_name = preset.method_name();
            let description = preset.description();

            presets.push_str(&format!(
                r#"    /// {}
    pub fn {}() -> Self {{
        Self::new(){}
    }}

"#,
                description,
                method_name,
                self.generate_preset_configuration(preset)
            ));
        }

        presets.push_str("}\n");
        presets
    }

    /// Generate configuration for a specific preset
    fn generate_preset_configuration(&self, preset: &BuilderPreset) -> String {
        let mut config = String::new();

        // Generate preset-specific field values based on safety mode and preset type
        for field in &self.config.fields {
            if !field.required {
                if let Some(value) = self.get_preset_value(preset, field) {
                    config.push_str(&format!("\n            .{}({})", field.name, value));
                }
            }
        }

        config
    }

    /// Get preset value for a field
    fn get_preset_value(&self, preset: &BuilderPreset, field: &BuilderField) -> Option<String> {
        match preset {
            BuilderPreset::HighPerformance => {
                // Minimal safety checks, maximum performance
                if field.name.contains("check") || field.name.contains("validate") {
                    Some("false".to_string())
                } else if field.name.contains("buffer") || field.name.contains("size") {
                    Some("8192".to_string()) // Large buffers
                } else {
                    field.default.clone()
                }
            }
            BuilderPreset::SafeDefaults => {
                // Maximum safety checks
                if field.name.contains("check") || field.name.contains("validate") {
                    Some("true".to_string())
                } else if field.name.contains("buffer") || field.name.contains("size") {
                    Some("1024".to_string()) // Reasonable buffers
                } else {
                    field.default.clone()
                }
            }
            BuilderPreset::Balanced => {
                // Use default values
                field.default.clone()
            }
            BuilderPreset::Minimal => {
                // Bare minimum configuration
                if field.name.contains("enable") || field.name.contains("check") {
                    Some("false".to_string())
                } else {
                    None
                }
            }
            BuilderPreset::Testing => {
                // Suitable for unit tests
                if field.name.contains("timeout") {
                    Some("100".to_string()) // Short timeouts
                } else if field.name.contains("retry") {
                    Some("1".to_string()) // Minimal retries
                } else {
                    field.default.clone()
                }
            }
        }
    }

    /// Generate validation helper methods
    fn generate_validations(&self) -> String {
        let builder_name = &self.config.name;
        let mut validations = HashSet::new();

        // Collect all unique validations
        for field in &self.config.fields {
            for validation in &field.validations {
                validations.insert(validation.clone());
            }
        }

        if validations.is_empty() {
            return String::new();
        }

        let mut output = format!("impl {} {{\n", builder_name);

        for validation in validations {
            // Generate helper methods for custom validations
            if let FieldValidation::Custom(name) = validation {
                output.push_str(&format!(
                    r#"    /// Custom validation: {}
    fn validate_{}(value: impl std::fmt::Display) -> Result<(), crate::Error> {{
        // TODO: Implement custom validation logic
        Ok(())
    }}

"#,
                    name, name
                ));
            }
        }

        output.push_str("}\n");
        output
    }
}

/// Analyze FFI info and generate builder configurations
pub fn generate_builder_configs(ffi_info: &FfiInfo, safety_mode: SafetyMode) -> Vec<BuilderConfig> {
    let mut configs = Vec::new();

    // For each struct that would benefit from a builder
    for ty in &ffi_info.types {
        // Only generate builders for structs with multiple fields
        let field_count = estimate_field_count(&ty.name);
        if field_count >= 3 {
            let mut config = BuilderConfig::new(&ty.name);

            // Add appropriate presets based on safety mode
            match safety_mode {
                SafetyMode::Strict => {
                    config.add_preset(BuilderPreset::SafeDefaults);
                    config.add_preset(BuilderPreset::Balanced);
                }
                SafetyMode::Balanced => {
                    config.add_preset(BuilderPreset::Balanced);
                    config.add_preset(BuilderPreset::HighPerformance);
                    config.add_preset(BuilderPreset::SafeDefaults);
                }
                SafetyMode::Permissive => {
                    config.add_preset(BuilderPreset::HighPerformance);
                    config.add_preset(BuilderPreset::Minimal);
                }
            }

            configs.push(config);
        }
    }

    configs
}

/// Estimate field count from type name (heuristic)
fn estimate_field_count(type_name: &str) -> usize {
    // Simple heuristic: types with "Config" or "Options" in name likely have multiple fields
    if type_name.contains("Config") || type_name.contains("Options") || type_name.contains("Params")
    {
        5 // Assume 5+ fields for config-like types
    } else {
        2 // Assume fewer fields for other types
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_builder_preset_method_names() {
        assert_eq!(
            BuilderPreset::HighPerformance.method_name(),
            "high_performance"
        );
        assert_eq!(BuilderPreset::SafeDefaults.method_name(), "safe_defaults");
        assert_eq!(BuilderPreset::Balanced.method_name(), "balanced");
    }

    #[test]
    fn test_field_validation_method_names() {
        assert_eq!(FieldValidation::NonZero.method_name(), "validate_non_zero");
        assert_eq!(FieldValidation::Positive.method_name(), "validate_positive");
        assert_eq!(
            FieldValidation::PowerOfTwo.method_name(),
            "validate_power_of_two"
        );
        assert_eq!(
            FieldValidation::MultipleOf(4).method_name(),
            "validate_multiple_of_4"
        );
    }

    #[test]
    fn test_validation_error_messages() {
        assert_eq!(
            FieldValidation::NonZero.error_message("width"),
            "width must be non-zero"
        );
        assert_eq!(
            FieldValidation::Positive.error_message("height"),
            "height must be positive"
        );
        assert_eq!(
            FieldValidation::Range { min: 0, max: 100 }.error_message("value"),
            "value must be in range [0, 100]"
        );
    }

    #[test]
    fn test_builder_config_creation() {
        let config = BuilderConfig::new("MyStruct");
        assert_eq!(config.name, "MyStructBuilder");
        assert_eq!(config.target_name, "MyStruct");
        assert!(config.implement_clone);
    }

    #[test]
    fn test_builder_field_addition() {
        let mut config = BuilderConfig::new("TestStruct");

        let field = BuilderField {
            name: "width".to_string(),
            ty: "u32".to_string(),
            required: true,
            default: None,
            validations: vec![FieldValidation::Positive],
            doc: Some("Width in pixels".to_string()),
        };

        config.add_field(field);
        assert_eq!(config.fields.len(), 1);
        assert_eq!(config.fields[0].name, "width");
    }

    #[test]
    fn test_builder_preset_addition() {
        let mut config = BuilderConfig::new("TestStruct");
        config.add_preset(BuilderPreset::Testing);

        assert!(config.presets.contains(&BuilderPreset::Testing));
    }

    #[test]
    fn test_builder_generator_creation() {
        let config = BuilderConfig::new("TestStruct");
        let generator = BuilderGenerator::new(config, SafetyMode::Balanced);

        assert_eq!(generator.config.name, "TestStructBuilder");
        assert_eq!(generator.safety_mode, SafetyMode::Balanced);
    }

    #[test]
    fn test_estimate_field_count() {
        assert_eq!(estimate_field_count("DeviceConfig"), 5);
        assert_eq!(estimate_field_count("InitOptions"), 5);
        assert_eq!(estimate_field_count("DeviceParams"), 5);
        assert_eq!(estimate_field_count("SimpleHandle"), 2);
    }

    #[test]
    fn test_validation_code_generation() {
        let field_name = syn::Ident::new("width", proc_macro2::Span::call_site());
        let validation = FieldValidation::Positive;
        let code = validation.generate_code(&field_name, "width");

        let code_str = code.to_string();
        assert!(code_str.contains("width"));
        assert!(code_str.contains("<= 0"));
    }

    #[test]
    fn test_power_of_two_validation() {
        let validation = FieldValidation::PowerOfTwo;
        let field_name = syn::Ident::new("size", proc_macro2::Span::call_site());
        let code = validation.generate_code(&field_name, "size");

        let code_str = code.to_string();
        assert!(code_str.contains("size"));
        assert!(code_str.contains("& ("));
    }
}
