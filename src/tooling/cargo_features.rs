/// Cargo feature flag management for generated crates
///
/// Provides different safety and checking levels to balance safety with flexibility.

use crate::config::GeneratorConfig;
use std::io::Write;

/// Safety and checking modes available via feature flags
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SafetyMode {
    /// Maximum safety checks, may reject valid code
    Strict,
    /// Reasonable defaults (recommended)
    Balanced,
    /// Minimal checks, maximum compatibility
    Permissive,
}

/// Generate feature flags section for Cargo.toml
pub fn generate_cargo_features(_config: &GeneratorConfig) -> String {
    let mut toml = String::new();
    
    toml.push_str("\n[features]\n");
    toml.push_str("# Safety modes (mutually exclusive)\n");
    toml.push_str("strict = []\n");
    toml.push_str("balanced = []\n");
    toml.push_str("permissive = []\n");
    toml.push_str("\n");
    
    toml.push_str("# Additional features\n");
    toml.push_str("debug-extra = []  # Extra runtime checks in debug builds\n");
    toml.push_str("tracing = [\"dep:tracing\"]  # Structured logging for FFI calls\n");
    toml.push_str("leak-detector = []  # Track resource allocations in debug\n");
    toml.push_str("\n");
    
    toml.push_str("# Default: balanced mode\n");
    toml.push_str("default = [\"balanced\"]\n");
    
    toml
}

/// Generate feature-gated code documentation
pub fn generate_feature_docs() -> String {
    let mut docs = String::new();
    
    docs.push_str("//! # Feature Flags\n");
    docs.push_str("//!\n");
    docs.push_str("//! This crate supports multiple feature flags to control safety vs flexibility:\n");
    docs.push_str("//!\n");
    docs.push_str("//! ## Safety Modes (choose one)\n");
    docs.push_str("//!\n");
    docs.push_str("//! - **`balanced`** (default): Reasonable safety checks with good compatibility\n");
    docs.push_str("//!   - Null pointer checks on required parameters\n");
    docs.push_str("//!   - Basic invariant validation\n");
    docs.push_str("//!   - Runtime assertions in debug builds\n");
    docs.push_str("//!\n");
    docs.push_str("//! - **`strict`**: Maximum safety, may reject valid code\n");
    docs.push_str("//!   - Aggressive null pointer checks\n");
    docs.push_str("//!   - Comprehensive precondition validation\n");
    docs.push_str("//!   - Runtime checks even in release builds\n");
    docs.push_str("//!   - Type-state builders for compile-time safety\n");
    docs.push_str("//!\n");
    docs.push_str("//! - **`permissive`**: Minimal checks, maximum compatibility\n");
    docs.push_str("//!   - Only essential null checks\n");
    docs.push_str("//!   - Trust caller to provide valid inputs\n");
    docs.push_str("//!   - No runtime overhead\n");
    docs.push_str("//!\n");
    docs.push_str("//! ## Additional Features\n");
    docs.push_str("//!\n");
    docs.push_str("//! - **`debug-extra`**: Enhanced debugging information\n");
    docs.push_str("//!   - Parameter value logging\n");
    docs.push_str("//!   - Call stack capture on errors\n");
    docs.push_str("//!   - Detailed assertion messages\n");
    docs.push_str("//!\n");
    docs.push_str("//! - **`tracing`**: Structured logging (requires `tracing` crate)\n");
    docs.push_str("//!   - Log all FFI calls with timing\n");
    docs.push_str("//!   - Track resource allocations\n");
    docs.push_str("//!   - Performance metrics\n");
    docs.push_str("//!\n");
    docs.push_str("//! - **`leak-detector`**: Resource leak detection in debug builds\n");
    docs.push_str("//!   - Track all handle allocations\n");
    docs.push_str("//!   - Detect unclosed resources\n");
    docs.push_str("//!   - Report leaks with allocation sites\n");
    docs.push_str("//!\n");
    docs.push_str("//! # Examples\n");
    docs.push_str("//!\n");
    docs.push_str("//! ```toml\n");
    docs.push_str("//! # Use strict mode with leak detection\n");
    docs.push_str("//! [dependencies]\n");
    docs.push_str("//! my-sys = { version = \"0.1\", features = [\"strict\", \"leak-detector\"] }\n");
    docs.push_str("//! ```\n");
    docs.push_str("//!\n");
    docs.push_str("//! ```toml\n");
    docs.push_str("//! # Use permissive mode for maximum performance\n");
    docs.push_str("//! [dependencies]\n");
    docs.push_str("//! my-sys = { version = \"0.1\", features = [\"permissive\"] }\n");
    docs.push_str("//! ```\n");
    
    docs
}

/// Generate conditional compilation attributes based on active features
pub struct FeatureGuard {
    mode: SafetyMode,
    debug_extra: bool,
    tracing: bool,
    leak_detector: bool,
}

impl FeatureGuard {
    pub fn new(mode: SafetyMode) -> Self {
        Self {
            mode,
            debug_extra: false,
            tracing: false,
            leak_detector: false,
        }
    }
    
    pub fn with_debug_extra(mut self) -> Self {
        self.debug_extra = true;
        self
    }
    
    pub fn with_tracing(mut self) -> Self {
        self.tracing = true;
        self
    }
    
    pub fn with_leak_detector(mut self) -> Self {
        self.leak_detector = true;
        self
    }
    
    /// Generate null pointer check code based on safety mode
    pub fn null_check(&self, param_name: &str, is_optional: bool) -> String {
        match (self.mode, is_optional) {
            // Strict: Always check, even optional
            (SafetyMode::Strict, _) => {
                format!(
                    "if {}.is_null() {{\n\
                     return Err(Error::NullPointer);\n\
                    }}",
                    param_name
                )
            }
            // Balanced: Check required parameters
            (SafetyMode::Balanced, false) => {
                format!(
                    "if {}.is_null() {{\n\
                     return Err(Error::NullPointer);\n\
                    }}",
                    param_name
                )
            }
            // Permissive or optional: No check
            _ => String::new(),
        }
    }
    
    /// Generate range validation code
    pub fn range_check(&self, param: &str, min: Option<i64>, max: Option<i64>) -> String {
        match self.mode {
            SafetyMode::Strict | SafetyMode::Balanced => {
                let mut checks = Vec::new();
                
                if let Some(min_val) = min {
                    checks.push(format!(
                        "if {} < {} {{\n\
                         return Err(Error::InvalidValue);\n\
                        }}",
                        param, min_val
                    ));
                }
                
                if let Some(max_val) = max {
                    checks.push(format!(
                        "if {} > {} {{\n\
                         return Err(Error::InvalidValue);\n\
                        }}",
                        param, max_val
                    ));
                }
                
                checks.join("\n")
            }
            SafetyMode::Permissive => String::new(),
        }
    }
    
    /// Generate debug assertion with detailed message
    pub fn debug_assert(&self, condition: &str, message: &str) -> String {
        if self.mode == SafetyMode::Permissive {
            return String::new();
        }
        
        let msg = if self.debug_extra {
            format!("{} ({})", message, condition)
        } else {
            message.to_string()
        };
        
        format!("debug_assert!({}, {:?});", condition, msg)
    }
    
    /// Generate tracing span for FFI call
    pub fn trace_span(&self, fn_name: &str) -> String {
        if !self.tracing {
            return String::new();
        }
        
        format!(
            "let _span = tracing::trace_span!(\"ffi_call\", function = {:?}).entered();",
            fn_name
        )
    }
    
    /// Generate leak tracking code
    pub fn track_allocation(&self, handle: &str, alloc_site: &str) -> String {
        if !self.leak_detector {
            return String::new();
        }
        
        format!(
            "#[cfg(all(debug_assertions, feature = \"leak-detector\"))]\n\
            crate::__leak_tracker::track({}, {:?});",
            handle, alloc_site
        )
    }
    
    pub fn track_deallocation(&self, handle: &str) -> String {
        if !self.leak_detector {
            return String::new();
        }
        
        format!(
            "#[cfg(all(debug_assertions, feature = \"leak-detector\"))]\n\
            crate::__leak_tracker::untrack({});",
            handle
        )
    }
}

/// Generate leak detector infrastructure
pub fn generate_leak_detector() -> String {
    let mut code = String::new();
    
    code.push_str("#[cfg(all(debug_assertions, feature = \"leak-detector\"))]\n");
    code.push_str("pub(crate) mod __leak_tracker {\n");
    code.push_str("    use std::sync::Mutex;\n");
    code.push_str("    use std::collections::HashMap;\n");
    code.push_str("    use std::backtrace::Backtrace;\n");
    code.push_str("\n");
    code.push_str("    struct Allocation {\n");
    code.push_str("        site: &'static str,\n");
    code.push_str("        backtrace: Backtrace,\n");
    code.push_str("    }\n");
    code.push_str("\n");
    code.push_str("    static ALLOCATIONS: Mutex<HashMap<usize, Allocation>> = Mutex::new(HashMap::new());\n");
    code.push_str("\n");
    code.push_str("    pub fn track<T>(handle: *mut T, site: &'static str) {\n");
    code.push_str("        let addr = handle as usize;\n");
    code.push_str("        let mut allocs = ALLOCATIONS.lock().unwrap();\n");
    code.push_str("        allocs.insert(addr, Allocation {\n");
    code.push_str("            site,\n");
    code.push_str("            backtrace: Backtrace::capture(),\n");
    code.push_str("        });\n");
    code.push_str("    }\n");
    code.push_str("\n");
    code.push_str("    pub fn untrack<T>(handle: *mut T) {\n");
    code.push_str("        let addr = handle as usize;\n");
    code.push_str("        let mut allocs = ALLOCATIONS.lock().unwrap();\n");
    code.push_str("        allocs.remove(&addr);\n");
    code.push_str("    }\n");
    code.push_str("\n");
    code.push_str("    pub fn report_leaks() {\n");
    code.push_str("        let allocs = ALLOCATIONS.lock().unwrap();\n");
    code.push_str("        if !allocs.is_empty() {\n");
    code.push_str("            eprintln!(\"⚠️  RESOURCE LEAKS DETECTED: {} handles not freed\", allocs.len());\n");
    code.push_str("            for (addr, alloc) in allocs.iter() {\n");
    code.push_str("                eprintln!(\"  • Handle {:#x} allocated at {}\", addr, alloc.site);\n");
    code.push_str("                eprintln!(\"    Backtrace:\\n{}\", alloc.backtrace);\n");
    code.push_str("            }\n");
    code.push_str("        }\n");
    code.push_str("    }\n");
    code.push_str("}\n");
    code.push_str("\n");
    code.push_str("#[cfg(all(debug_assertions, feature = \"leak-detector\"))]\n");
    code.push_str("impl Drop for LeakDetectorGuard {\n");
    code.push_str("    fn drop(&mut self) {\n");
    code.push_str("        __leak_tracker::report_leaks();\n");
    code.push_str("    }\n");
    code.push_str("}\n");
    
    code
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_generate_cargo_features() {
        let config = GeneratorConfig::default();
        let features = generate_cargo_features(&config);
        
        assert!(features.contains("strict"));
        assert!(features.contains("balanced"));
        assert!(features.contains("permissive"));
        assert!(features.contains("debug-extra"));
        assert!(features.contains("default = [\"balanced\"]"));
    }
    
    #[test]
    fn test_feature_docs() {
        let docs = generate_feature_docs();
        
        assert!(docs.contains("Feature Flags"));
        assert!(docs.contains("strict"));
        assert!(docs.contains("balanced"));
        assert!(docs.contains("permissive"));
        assert!(docs.contains("debug-extra"));
    }
    
    #[test]
    fn test_null_check_strict() {
        let guard = FeatureGuard::new(SafetyMode::Strict);
        
        // Strict checks even optional parameters
        let check = guard.null_check("ptr", true);
        assert!(check.contains("is_null"));
        assert!(check.contains("NullPointer"));
    }
    
    #[test]
    fn test_null_check_balanced() {
        let guard = FeatureGuard::new(SafetyMode::Balanced);
        
        // Balanced checks required parameters
        let check = guard.null_check("ptr", false);
        assert!(check.contains("is_null"));
        
        // But not optional ones
        let check_opt = guard.null_check("ptr", true);
        assert!(check_opt.is_empty());
    }
    
    #[test]
    fn test_null_check_permissive() {
        let guard = FeatureGuard::new(SafetyMode::Permissive);
        
        // Permissive never checks
        let check = guard.null_check("ptr", false);
        assert!(check.is_empty());
    }
    
    #[test]
    fn test_range_checks() {
        let guard = FeatureGuard::new(SafetyMode::Strict);
        
        let check = guard.range_check("value", Some(0), Some(100));
        assert!(check.contains("< 0"));
        assert!(check.contains("> 100"));
        assert!(check.contains("InvalidValue"));
    }
    
    #[test]
    fn test_range_checks_permissive() {
        let guard = FeatureGuard::new(SafetyMode::Permissive);
        
        let check = guard.range_check("value", Some(0), Some(100));
        assert!(check.is_empty());
    }
    
    #[test]
    fn test_tracing_span() {
        let guard = FeatureGuard::new(SafetyMode::Balanced).with_tracing();
        
        let span = guard.trace_span("cudnnCreate");
        assert!(span.contains("tracing::trace_span"));
        assert!(span.contains("cudnnCreate"));
    }
    
    #[test]
    fn test_leak_tracking() {
        let guard = FeatureGuard::new(SafetyMode::Balanced).with_leak_detector();
        
        let track = guard.track_allocation("handle", "CudnnHandle::new");
        assert!(track.contains("leak-detector"));
        assert!(track.contains("__leak_tracker::track"));
        
        let untrack = guard.track_deallocation("handle");
        assert!(untrack.contains("__leak_tracker::untrack"));
    }
    
    #[test]
    fn test_leak_detector_code() {
        let code = generate_leak_detector();
        
        assert!(code.contains("mod __leak_tracker"));
        assert!(code.contains("Backtrace"));
        assert!(code.contains("report_leaks"));
        assert!(code.contains("RESOURCE LEAKS DETECTED"));
    }
}
