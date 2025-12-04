//! Runtime safety instrumentation and validation
//!
//! This module provides:
//! - Bounds checking validation
//! - Sanitizer integration (AddressSanitizer, MemorySanitizer, etc.)
//! - Runtime safety instrumentation
//! - Dynamic safety checks

use crate::ffi::parser::{FfiFunction, FfiParam};

/// Safety check type
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SafetyCheck {
    /// Null pointer check
    NullCheck(String),
    /// Bounds check for buffer access
    BoundsCheck { buffer: String, size: String },
    /// Alignment check
    AlignmentCheck(String),
    /// Lifetime validity check
    LifetimeCheck(String),
    /// Thread safety check
    ThreadSafetyCheck,
}

/// Runtime safety instrumentation
pub struct SafetyInstrumentation {
    /// Checks to insert
    checks: Vec<SafetyCheck>,
}

impl SafetyInstrumentation {
    /// Create new instrumentation
    pub fn new() -> Self {
        Self { checks: Vec::new() }
    }

    /// Analyze function for safety instrumentation needs
    pub fn analyze_function(&mut self, func: &FfiFunction) {
        for param in &func.params {
            if param.is_pointer {
                // Add null check
                self.checks
                    .push(SafetyCheck::NullCheck(param.name.clone()));

                // Add bounds check if there's a size parameter
                if let Some(size_param) = Self::find_size_param(&func.params, &param.name) {
                    self.checks.push(SafetyCheck::BoundsCheck {
                        buffer: param.name.clone(),
                        size: size_param,
                    });
                }

                // Check alignment for typed pointers
                if !param.ty.contains("void") && !param.ty.contains("char") {
                    self.checks
                        .push(SafetyCheck::AlignmentCheck(param.name.clone()));
                }
            }
        }
    }

    /// Find size parameter for a buffer
    fn find_size_param(params: &[FfiParam], buffer_name: &str) -> Option<String> {
        // Look for common patterns: size, len, count, n
        let size_names = ["size", "len", "count", "n"];

        for param in params {
            // Skip the buffer parameter itself
            if param.name == buffer_name {
                continue;
            }

            let param_lower = param.name.to_lowercase();

            // Check if parameter name suggests it's a size
            if size_names.iter().any(|&name| param_lower.contains(name)) {
                return Some(param.name.clone());
            }

            // Check if it's named after the buffer (e.g., "data" -> "data_size")
            if param_lower.contains(&buffer_name.to_lowercase()) {
                return Some(param.name.clone());
            }
        }

        None
    }

    /// Generate safety check code
    pub fn generate_checks(&self) -> String {
        let mut output = String::new();

        for check in &self.checks {
            output.push_str(&self.generate_check_code(check));
            output.push('\n');
        }

        output
    }

    /// Generate code for a specific check
    fn generate_check_code(&self, check: &SafetyCheck) -> String {
        match check {
            SafetyCheck::NullCheck(param) => {
                format!(
                    r#"    if {}.is_null() {{
        return Err(crate::Error::NullPointer("{} is null".to_string()));
    }}"#,
                    param, param
                )
            }
            SafetyCheck::BoundsCheck { buffer, size } => {
                format!(
                    r#"    if {} == 0 {{
        return Err(crate::Error::InvalidParameter("{} size is zero".to_string()));
    }}"#,
                    size, buffer
                )
            }
            SafetyCheck::AlignmentCheck(param) => {
                format!(
                    r#"    if ({} as usize) % std::mem::align_of::<T>() != 0 {{
        return Err(crate::Error::InvalidParameter("{} is not properly aligned".to_string()));
    }}"#,
                    param, param
                )
            }
            SafetyCheck::LifetimeCheck(_) => {
                "    // Lifetime check placeholder\n".to_string()
            }
            SafetyCheck::ThreadSafetyCheck => {
                "    // Thread safety check placeholder\n".to_string()
            }
        }
    }
}

impl Default for SafetyInstrumentation {
    fn default() -> Self {
        Self::new()
    }
}

/// Sanitizer integration
pub struct SanitizerIntegration;

impl SanitizerIntegration {
    /// Generate sanitizer build configuration
    pub fn generate_build_config() -> String {
        r#"# Sanitizer build configurations

# AddressSanitizer (memory errors)
[profile.asan]
inherits = "dev"
rustflags = ["-Zsanitizer=address"]

# MemorySanitizer (uninitialized reads)
[profile.msan]
inherits = "dev"
rustflags = ["-Zsanitizer=memory"]

# ThreadSanitizer (data races)
[profile.tsan]
inherits = "dev"
rustflags = ["-Zsanitizer=thread"]

# LeakSanitizer (memory leaks)
[profile.lsan]
inherits = "dev"
rustflags = ["-Zsanitizer=leak"]
"#
        .to_string()
    }

    /// Generate sanitizer test script
    pub fn generate_test_script() -> String {
        r#"#!/bin/bash
# Run tests with various sanitizers

echo "Running with AddressSanitizer..."
RUSTFLAGS="-Z sanitizer=address" cargo +nightly test

echo "Running with ThreadSanitizer..."
RUSTFLAGS="-Z sanitizer=thread" cargo +nightly test

echo "Running with LeakSanitizer..."
RUSTFLAGS="-Z sanitizer=leak" cargo +nightly test

echo "All sanitizer tests complete!"
"#
        .to_string()
    }
}

/// Bounds checking validator
pub struct BoundsValidator {
    /// Buffer-size pairs to validate
    bounds: Vec<(String, String)>,
}

impl BoundsValidator {
    /// Create new validator
    pub fn new() -> Self {
        Self { bounds: Vec::new() }
    }

    /// Add bounds check
    pub fn add_bounds_check(&mut self, buffer: String, size: String) {
        self.bounds.push((buffer, size));
    }

    /// Generate validation code
    pub fn generate_validation(&self) -> String {
        let mut output = String::from(
            r#"/// Runtime bounds validation
#[inline]
fn validate_bounds(ptr: *const u8, size: usize) -> Result<(), crate::Error> {
    if ptr.is_null() {
        return Err(crate::Error::NullPointer("Buffer pointer is null".to_string()));
    }
    if size == 0 {
        return Err(crate::Error::InvalidParameter("Buffer size is zero".to_string()));
    }
    Ok(())
}

"#,
        );

        // Generate specific validators
        for (buffer, size) in &self.bounds {
            output.push_str(&format!(
                r#"/// Validate {} bounds
#[inline]
fn validate_{}_bounds({}: *const u8, {}: usize) -> Result<(), crate::Error> {{
    validate_bounds({}, {})?;
    Ok(())
}}

"#,
                buffer, buffer, buffer, size, buffer, size
            ));
        }

        output
    }
}

impl Default for BoundsValidator {
    fn default() -> Self {
        Self::new()
    }
}

/// Debug build instrumentation
pub struct DebugInstrumentation;

impl DebugInstrumentation {
    /// Generate debug-only checks
    pub fn generate_debug_checks() -> String {
        r#"/// Debug-only assertion macro
#[cfg(debug_assertions)]
macro_rules! debug_check {
    ($cond:expr, $msg:expr) => {
        if !$cond {
            panic!("Debug assertion failed: {}", $msg);
        }
    };
}

#[cfg(not(debug_assertions))]
macro_rules! debug_check {
    ($cond:expr, $msg:expr) => {{}};
}

/// Debug-only null pointer check
#[cfg(debug_assertions)]
#[inline]
fn debug_check_null(ptr: *const ()) -> bool {
    if ptr.is_null() {
        eprintln!("Warning: Null pointer detected!");
        false
    } else {
        true
    }
}

#[cfg(not(debug_assertions))]
#[inline]
fn debug_check_null(_ptr: *const ()) -> bool {
    true
}
"#
        .to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_safety_instrumentation() {
        let func = FfiFunction {
            name: "process".to_string(),
            params: vec![
                FfiParam {
                    name: "data".to_string(),
                    ty: "u8".to_string(),
                    is_pointer: true,
                    is_mut: false,
                },
                FfiParam {
                    name: "size".to_string(),
                    ty: "size_t".to_string(),
                    is_pointer: false,
                    is_mut: false,
                },
            ],
            return_type: "void".to_string(),
            docs: None,
        };

        let mut instr = SafetyInstrumentation::new();
        instr.analyze_function(&func);

        assert!(!instr.checks.is_empty());
        assert!(instr
            .checks
            .iter()
            .any(|c| matches!(c, SafetyCheck::NullCheck(_))));
    }

    #[test]
    fn test_find_size_param() {
        let params = vec![
            FfiParam {
                name: "data".to_string(),
                ty: "u8".to_string(),
                is_pointer: true,
                is_mut: false,
            },
            FfiParam {
                name: "data_size".to_string(),
                ty: "size_t".to_string(),
                is_pointer: false,
                is_mut: false,
            },
        ];

        let size = SafetyInstrumentation::find_size_param(&params, "data");
        assert_eq!(size, Some("data_size".to_string()));
    }

    #[test]
    fn test_check_generation() {
        let instr = SafetyInstrumentation::new();

        let null_check = instr.generate_check_code(&SafetyCheck::NullCheck("ptr".to_string()));
        assert!(null_check.contains("is_null"));
        assert!(null_check.contains("NullPointer"));

        let bounds_check = instr.generate_check_code(&SafetyCheck::BoundsCheck {
            buffer: "buf".to_string(),
            size: "len".to_string(),
        });
        assert!(bounds_check.contains("len"));
        assert!(bounds_check.contains("zero"));
    }

    #[test]
    fn test_sanitizer_config() {
        let config = SanitizerIntegration::generate_build_config();
        assert!(config.contains("address"));
        assert!(config.contains("thread"));
        assert!(config.contains("leak"));
    }

    #[test]
    fn test_bounds_validator() {
        let mut validator = BoundsValidator::new();
        validator.add_bounds_check("buffer".to_string(), "size".to_string());

        let code = validator.generate_validation();
        assert!(code.contains("validate_bounds"));
        assert!(code.contains("validate_buffer_bounds"));
    }

    #[test]
    fn test_debug_instrumentation() {
        let code = DebugInstrumentation::generate_debug_checks();
        assert!(code.contains("macro_rules! debug_check"));
        assert!(code.contains("debug_check_null"));
        assert!(code.contains("#[cfg(debug_assertions)]"));
    }
}
