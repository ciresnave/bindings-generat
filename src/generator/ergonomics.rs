//! Ergonomic convenience features for generated bindings
//!
//! This module provides:
//! - Extension traits for common conversions
//! - Iterator adapters for FFI results
//! - Operator overloading where safe and idiomatic
//! - Convenience macros for common patterns

use crate::ffi::parser::{FfiFunction, FfiInfo};
use quote::quote;
use std::collections::HashMap;

/// Type of extension trait to generate
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ExtensionTraitType {
    /// String conversion (C string ↔ Rust String)
    StringConversion,
    /// Slice conversion (C array ↔ Rust slice)
    SliceConversion,
    /// Numeric conversion (between different numeric types)
    NumericConversion,
    /// Result conversion (status code → Result)
    ResultConversion,
    /// Iterator adapter (C iterator → Rust Iterator)
    IteratorAdapter,
}

/// Operator overloading opportunity
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OperatorOverload {
    /// Type to implement operator for
    pub type_name: String,
    /// Operator to overload (Add, Sub, Mul, etc.)
    pub operator: String,
    /// Trait to implement (std::ops::Add, etc.)
    pub trait_name: String,
    /// Implementation code
    pub implementation: String,
}

/// Iterator adapter configuration
#[derive(Debug, Clone)]
pub struct IteratorAdapter {
    /// Adapter name
    pub name: String,
    /// Item type
    pub item_type: String,
    /// Source FFI function
    pub source_function: String,
    /// Iteration logic
    pub iteration_code: String,
}

/// Extension trait generator
pub struct ExtensionTraitGenerator {
    traits: HashMap<ExtensionTraitType, Vec<String>>,
}

impl ExtensionTraitGenerator {
    /// Create a new extension trait generator
    pub fn new() -> Self {
        Self {
            traits: HashMap::new(),
        }
    }

    /// Analyze FFI info and identify extension trait opportunities
    pub fn analyze(&mut self, ffi_info: &FfiInfo) {
        for func in &ffi_info.functions {
            self.analyze_function(func);
        }
    }

    /// Analyze a single function for extension trait opportunities
    fn analyze_function(&mut self, func: &FfiFunction) {
        // Check for string conversion opportunities
        if self.has_string_params(func) {
            self.add_trait(ExtensionTraitType::StringConversion, func.name.clone());
        }

        // Check for slice conversion opportunities
        if self.has_slice_params(func) {
            self.add_trait(ExtensionTraitType::SliceConversion, func.name.clone());
        }

        // Check for result conversion opportunities
        if self.returns_status_code(func) {
            self.add_trait(ExtensionTraitType::ResultConversion, func.name.clone());
        }
    }

    /// Check if function has string parameters
    fn has_string_params(&self, func: &FfiFunction) -> bool {
        func.params
            .iter()
            .any(|p| p.ty.contains("char") && p.is_pointer)
    }

    /// Check if function has slice parameters
    fn has_slice_params(&self, func: &FfiFunction) -> bool {
        func.params
            .iter()
            .any(|p| p.is_pointer && !p.ty.contains("char") && !p.ty.contains("void"))
    }

    /// Check if function returns a status code
    fn returns_status_code(&self, func: &FfiFunction) -> bool {
        func.return_type.contains("Status")
            || func.return_type.contains("Error")
            || func.return_type.contains("Result")
            || func.return_type == "i32"
            || func.return_type == "c_int"
    }

    /// Add a trait type with associated function
    fn add_trait(&mut self, trait_type: ExtensionTraitType, function: String) {
        self.traits
            .entry(trait_type)
            .or_insert_with(Vec::new)
            .push(function);
    }

    /// Generate extension trait code
    pub fn generate(&self) -> String {
        let mut output = String::new();

        if self
            .traits
            .contains_key(&ExtensionTraitType::StringConversion)
        {
            output.push_str(&self.generate_string_trait());
            output.push_str("\n\n");
        }

        if self
            .traits
            .contains_key(&ExtensionTraitType::SliceConversion)
        {
            output.push_str(&self.generate_slice_trait());
            output.push_str("\n\n");
        }

        if self
            .traits
            .contains_key(&ExtensionTraitType::ResultConversion)
        {
            output.push_str(&self.generate_result_trait());
            output.push_str("\n\n");
        }

        output
    }

    /// Generate string conversion extension trait
    fn generate_string_trait(&self) -> String {
        r#"/// Extension trait for convenient string conversions
pub trait StringConversionExt {
    /// Convert to a C-compatible string
    fn to_c_string(&self) -> Result<std::ffi::CString, crate::Error>;
}

impl StringConversionExt for str {
    fn to_c_string(&self) -> Result<std::ffi::CString, crate::Error> {
        std::ffi::CString::new(self)
            .map_err(|e| crate::Error::InvalidParameter(format!("Invalid string: {}", e)))
    }
}

impl StringConversionExt for String {
    fn to_c_string(&self) -> Result<std::ffi::CString, crate::Error> {
        self.as_str().to_c_string()
    }
}

/// Extension trait for converting from C strings
pub trait FromCStringExt {
    /// Convert from a C string pointer (unsafe)
    ///
    /// # Safety
    /// - `ptr` must be a valid, null-terminated C string
    /// - `ptr` must remain valid for the duration of the conversion
    unsafe fn from_c_str(ptr: *const std::os::raw::c_char) -> Result<String, crate::Error>;
}

impl FromCStringExt for String {
    unsafe fn from_c_str(ptr: *const std::os::raw::c_char) -> Result<String, crate::Error> {
        if ptr.is_null() {
            return Err(crate::Error::NullPointer("C string pointer is null".to_string()));
        }

        std::ffi::CStr::from_ptr(ptr)
            .to_str()
            .map(|s| s.to_string())
            .map_err(|e| crate::Error::InvalidParameter(format!("Invalid UTF-8: {}", e)))
    }
}"#
        .to_string()
    }

    /// Generate slice conversion extension trait
    fn generate_slice_trait(&self) -> String {
        r#"/// Extension trait for convenient slice conversions
pub trait SliceConversionExt<T> {
    /// Get a pointer to the slice data
    fn as_ptr_len(&self) -> (*const T, usize);

    /// Get a mutable pointer to the slice data
    fn as_mut_ptr_len(&mut self) -> (*mut T, usize);
}

impl<T> SliceConversionExt<T> for [T] {
    fn as_ptr_len(&self) -> (*const T, usize) {
        (self.as_ptr(), self.len())
    }

    fn as_mut_ptr_len(&mut self) -> (*mut T, usize) {
        (self.as_mut_ptr(), self.len())
    }
}

impl<T> SliceConversionExt<T> for Vec<T> {
    fn as_ptr_len(&self) -> (*const T, usize) {
        (self.as_ptr(), self.len())
    }

    fn as_mut_ptr_len(&mut self) -> (*mut T, usize) {
        (self.as_mut_ptr(), self.len())
    }
}

/// Extension trait for creating slices from raw pointers
pub trait FromRawSliceExt<T> {
    /// Create a slice from a raw pointer and length (unsafe)
    ///
    /// # Safety
    /// - `ptr` must be valid for reads of `len * size_of::<T>()` bytes
    /// - `ptr` must be properly aligned for type `T`
    /// - The memory referenced must not be mutated for the lifetime of the slice
    unsafe fn from_raw_parts_safe(ptr: *const T, len: usize) -> Result<&'static [T], crate::Error>;
}

impl<T> FromRawSliceExt<T> for [T] {
    unsafe fn from_raw_parts_safe(ptr: *const T, len: usize) -> Result<&'static [T], crate::Error> {
        if ptr.is_null() {
            return Err(crate::Error::NullPointer("Slice pointer is null".to_string()));
        }

        if len == 0 {
            return Ok(&[]);
        }

        Ok(std::slice::from_raw_parts(ptr, len))
    }
}"#
        .to_string()
    }

    /// Generate result conversion extension trait
    fn generate_result_trait(&self) -> String {
        r#"/// Extension trait for converting status codes to Results
pub trait StatusCodeExt {
    /// Convert to a Result type
    fn to_result(self) -> Result<(), crate::Error>;

    /// Convert to a Result with a value
    fn to_result_with<T>(self, value: T) -> Result<T, crate::Error>;
}

impl StatusCodeExt for i32 {
    fn to_result(self) -> Result<(), crate::Error> {
        if self == 0 {
            Ok(())
        } else {
            Err(crate::Error::FfiError(self))
        }
    }

    fn to_result_with<T>(self, value: T) -> Result<T, crate::Error> {
        if self == 0 {
            Ok(value)
        } else {
            Err(crate::Error::FfiError(self))
        }
    }
}"#
        .to_string()
    }
}

impl Default for ExtensionTraitGenerator {
    fn default() -> Self {
        Self::new()
    }
}

/// Operator overloading analyzer
pub struct OperatorOverloadAnalyzer {
    overloads: Vec<OperatorOverload>,
}

impl OperatorOverloadAnalyzer {
    /// Create a new analyzer
    pub fn new() -> Self {
        Self {
            overloads: Vec::new(),
        }
    }

    /// Analyze FFI info for operator overloading opportunities
    pub fn analyze(&mut self, ffi_info: &FfiInfo) {
        for func in &ffi_info.functions {
            self.analyze_function(func);
        }
    }

    /// Analyze a function for operator overloading
    fn analyze_function(&mut self, func: &FfiFunction) {
        let name_lower = func.name.to_lowercase();

        // Look for arithmetic operations
        if name_lower.contains("add") {
            self.suggest_overload(func, "Add", "std::ops::Add");
        } else if name_lower.contains("sub") || name_lower.contains("subtract") {
            self.suggest_overload(func, "Sub", "std::ops::Sub");
        } else if name_lower.contains("mul") || name_lower.contains("multiply") {
            self.suggest_overload(func, "Mul", "std::ops::Mul");
        } else if name_lower.contains("div") || name_lower.contains("divide") {
            self.suggest_overload(func, "Div", "std::ops::Div");
        }

        // Look for comparison operations
        if name_lower.contains("equal") || name_lower.contains("compare") {
            self.suggest_comparison(func);
        }
    }

    /// Suggest an arithmetic operator overload
    fn suggest_overload(&mut self, func: &FfiFunction, operator: &str, trait_name: &str) {
        // Only suggest if function has 2 parameters of the same type
        if func.params.len() == 2 {
            let type1 = &func.params[0].ty;
            let type2 = &func.params[1].ty;

            if type1 == type2 && !func.params[0].is_pointer && !func.params[1].is_pointer {
                let implementation = format!(
                    r#"impl {} for {} {{
    type Output = {};

    fn {}(self, rhs: Self) -> Self::Output {{
        unsafe {{ ffi::{}(self, rhs) }}
    }}
}}"#,
                    trait_name,
                    type1,
                    type1,
                    operator.to_lowercase(),
                    func.name
                );

                self.overloads.push(OperatorOverload {
                    type_name: type1.clone(),
                    operator: operator.to_string(),
                    trait_name: trait_name.to_string(),
                    implementation,
                });
            }
        }
    }

    /// Suggest comparison operator overload
    fn suggest_comparison(&mut self, func: &FfiFunction) {
        if func.params.len() == 2 {
            let type1 = &func.params[0].ty;
            let type2 = &func.params[1].ty;

            if type1 == type2 && !func.params[0].is_pointer && !func.params[1].is_pointer {
                let implementation = format!(
                    r#"impl PartialEq for {} {{
    fn eq(&self, other: &Self) -> bool {{
        unsafe {{ ffi::{}(*self, *other) != 0 }}
    }}
}}"#,
                    type1, func.name
                );

                self.overloads.push(OperatorOverload {
                    type_name: type1.clone(),
                    operator: "Eq".to_string(),
                    trait_name: "std::cmp::PartialEq".to_string(),
                    implementation,
                });
            }
        }
    }

    /// Get all suggested overloads
    pub fn overloads(&self) -> &[OperatorOverload] {
        &self.overloads
    }

    /// Generate operator overload implementations
    pub fn generate(&self) -> String {
        self.overloads
            .iter()
            .map(|o| o.implementation.clone())
            .collect::<Vec<_>>()
            .join("\n\n")
    }
}

impl Default for OperatorOverloadAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

/// Iterator adapter generator
pub struct IteratorAdapterGenerator {
    adapters: Vec<IteratorAdapter>,
}

impl IteratorAdapterGenerator {
    /// Create a new generator
    pub fn new() -> Self {
        Self {
            adapters: Vec::new(),
        }
    }

    /// Analyze FFI info for iterator opportunities
    pub fn analyze(&mut self, ffi_info: &FfiInfo) {
        // Look for patterns like: get_first, get_next, has_next
        let mut iterator_sets: HashMap<String, Vec<&FfiFunction>> = HashMap::new();

        for func in &ffi_info.functions {
            let name_lower = func.name.to_lowercase();

            // Extract base name (remove first/next/has prefixes)
            let base_name = if name_lower.contains("first") {
                name_lower.replace("first", "").replace("get_", "")
            } else if name_lower.contains("next") {
                name_lower.replace("next", "").replace("get_", "")
            } else if name_lower.contains("has") {
                name_lower.replace("has_", "")
            } else {
                continue;
            };

            iterator_sets
                .entry(base_name)
                .or_insert_with(Vec::new)
                .push(func);
        }

        // Generate adapters for complete sets
        for (base_name, funcs) in iterator_sets {
            if funcs.len() >= 2 {
                // Has at least get_first and get_next or similar
                self.generate_adapter(&base_name, &funcs);
            }
        }
    }

    /// Generate an iterator adapter
    fn generate_adapter(&mut self, base_name: &str, funcs: &[&FfiFunction]) {
        let adapter_name = format!("{}Iterator", Self::to_pascal_case(base_name));
        let item_type = Self::infer_item_type(funcs);

        let iteration_code = format!(
            r#"pub struct {} {{
    current: Option<{}>,
}}

impl {} {{
    pub fn new() -> Self {{
        Self {{
            current: Some(unsafe {{ ffi::get_first_{}() }}),
        }}
    }}
}}

impl Iterator for {} {{
    type Item = {};

    fn next(&mut self) -> Option<Self::Item> {{
        self.current.take().and_then(|current| {{
            let next = unsafe {{ ffi::get_next_{}(current) }};
            self.current = if !next.is_null() {{ Some(next) }} else {{ None }};
            if !current.is_null() {{ Some(current) }} else {{ None }}
        }})
    }}
}}"#,
            adapter_name, item_type, adapter_name, base_name, adapter_name, item_type, base_name
        );

        self.adapters.push(IteratorAdapter {
            name: adapter_name,
            item_type,
            source_function: format!("get_first_{}", base_name),
            iteration_code,
        });
    }

    /// Convert snake_case to PascalCase
    fn to_pascal_case(s: &str) -> String {
        s.split('_')
            .filter(|s| !s.is_empty())
            .map(|word| {
                let mut chars = word.chars();
                match chars.next() {
                    Some(first) => first.to_uppercase().chain(chars).collect::<String>(),
                    None => String::new(),
                }
            })
            .collect()
    }

    /// Infer item type from function signatures
    fn infer_item_type(funcs: &[&FfiFunction]) -> String {
        for func in funcs {
            let rt = &func.return_type;
            if rt.contains('*') {
                return rt.replace('*', "").trim().to_string();
            }
        }
        "Item".to_string()
    }

    /// Generate all iterator adapters
    pub fn generate(&self) -> String {
        self.adapters
            .iter()
            .map(|a| a.iteration_code.clone())
            .collect::<Vec<_>>()
            .join("\n\n")
    }
}

impl Default for IteratorAdapterGenerator {
    fn default() -> Self {
        Self::new()
    }
}

/// Generate convenience macros
pub fn generate_convenience_macros() -> String {
    r#"/// Convenience macro for calling FFI functions with automatic error handling
///
/// # Example
/// ```ignore
/// ffi_call!(my_function(arg1, arg2))?;
/// ```
#[macro_export]
macro_rules! ffi_call {
    ($func:ident($($arg:expr),*)) => {
        {
            let result = unsafe { ffi::$func($($arg),*) };
            if result != 0 {
                Err($crate::Error::FfiError(result))
            } else {
                Ok(())
            }
        }
    };
}

/// Convenience macro for calling FFI functions that return values
///
/// # Example
/// ```ignore
/// let value = ffi_call_with!(my_function(arg1, arg2), default_value)?;
/// ```
#[macro_export]
macro_rules! ffi_call_with {
    ($func:ident($($arg:expr),*), $default:expr) => {
        {
            let result = unsafe { ffi::$func($($arg),*) };
            if result != 0 {
                Err($crate::Error::FfiError(result))
            } else {
                Ok($default)
            }
        }
    };
}

/// Convenience macro for working with C strings
///
/// # Example
/// ```ignore
/// with_c_string!("hello" => |ptr| {
///     unsafe { ffi::my_function(ptr) }
/// })?;
/// ```
#[macro_export]
macro_rules! with_c_string {
    ($s:expr => |$ptr:ident| $body:expr) => {
        {
            use $crate::StringConversionExt;
            let c_string = $s.to_c_string()?;
            let $ptr = c_string.as_ptr();
            $body
        }
    };
}"#
    .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extension_trait_generator_creation() {
        let generator = ExtensionTraitGenerator::new();
        assert_eq!(generator.traits.len(), 0);
    }

    #[test]
    fn test_string_trait_generation() {
        let generator = ExtensionTraitGenerator::new();
        let output = generator.generate_string_trait();

        assert!(output.contains("StringConversionExt"));
        assert!(output.contains("to_c_string"));
        assert!(output.contains("from_c_str"));
    }

    #[test]
    fn test_slice_trait_generation() {
        let generator = ExtensionTraitGenerator::new();
        let output = generator.generate_slice_trait();

        assert!(output.contains("SliceConversionExt"));
        assert!(output.contains("as_ptr_len"));
        assert!(output.contains("from_raw_parts_safe"));
    }

    #[test]
    fn test_result_trait_generation() {
        let generator = ExtensionTraitGenerator::new();
        let output = generator.generate_result_trait();

        assert!(output.contains("StatusCodeExt"));
        assert!(output.contains("to_result"));
        assert!(output.contains("to_result_with"));
    }

    #[test]
    fn test_operator_overload_analyzer() {
        let analyzer = OperatorOverloadAnalyzer::new();
        assert_eq!(analyzer.overloads().len(), 0);
    }

    #[test]
    fn test_iterator_adapter_generator() {
        let generator = IteratorAdapterGenerator::new();
        assert_eq!(generator.adapters.len(), 0);
    }

    #[test]
    fn test_to_pascal_case() {
        assert_eq!(
            IteratorAdapterGenerator::to_pascal_case("my_item"),
            "MyItem"
        );
        assert_eq!(IteratorAdapterGenerator::to_pascal_case("device"), "Device");
        assert_eq!(
            IteratorAdapterGenerator::to_pascal_case("multi_word_name"),
            "MultiWordName"
        );
    }

    #[test]
    fn test_convenience_macros_generation() {
        let macros = generate_convenience_macros();

        assert!(macros.contains("ffi_call!"));
        assert!(macros.contains("ffi_call_with!"));
        assert!(macros.contains("with_c_string!"));
    }

    #[test]
    fn test_has_string_params() {
        let generator = ExtensionTraitGenerator::new();

        let func = FfiFunction {
            name: "test_func".to_string(),
            params: vec![crate::ffi::parser::FfiParam {
                name: "str".to_string(),
                ty: "char".to_string(),
                is_pointer: true,
                is_mut: false,
            }],
            return_type: "void".to_string(),
            docs: None,
        };

        assert!(generator.has_string_params(&func));
    }

    #[test]
    fn test_returns_status_code() {
        let generator = ExtensionTraitGenerator::new();

        let func = FfiFunction {
            name: "test_func".to_string(),
            params: vec![],
            return_type: "i32".to_string(),
            docs: None,
        };

        assert!(generator.returns_status_code(&func));
    }
}
