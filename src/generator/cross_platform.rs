//! Cross-platform code generation and testing support.
//!
//! This module provides functionality for generating platform-aware Rust code:
//! - Platform-specific #[cfg(...)] attributes
//! - Conditional compilation for different OSes
//! - Platform-specific type aliases and constants
//! - Cross-platform testing guidance

use crate::analyzer::platform::{Architecture, Platform, PlatformInfo};
use crate::enrichment::context::FunctionContext;
use crate::ffi::{FfiFunction, FfiType};
use std::collections::HashMap;
use std::fmt::Write;

/// Configuration for cross-platform code generation
#[derive(Debug, Clone)]
pub struct CrossPlatformConfig {
    /// Generate platform-specific test modules
    pub generate_platform_tests: bool,
    /// Include platform detection utilities
    pub include_platform_utils: bool,
    /// Generate stubs for unsupported platforms
    pub generate_platform_stubs: bool,
}

impl Default for CrossPlatformConfig {
    fn default() -> Self {
        Self {
            generate_platform_tests: true,
            include_platform_utils: true,
            generate_platform_stubs: false,
        }
    }
}

/// Platform-specific code section
#[derive(Debug, Clone)]
pub struct PlatformSection {
    pub platforms: Vec<Platform>,
    pub code: String,
    pub is_exclusive: bool, // true = only these platforms, false = all except these
}

/// Generate #[cfg(...)] attribute from PlatformInfo
pub fn generate_cfg_attribute(platform_info: &PlatformInfo) -> Option<String> {
    if !platform_info.is_platform_specific() {
        return None;
    }

    let mut conditions = Vec::new();

    // Add platform conditions
    for platform in &platform_info.available_on {
        if let Some(target_os) = platform.to_cfg_target_os() {
            conditions.push(format!("target_os = \"{}\"", target_os));
        } else if let Some(target_family) = platform.to_cfg_target_family() {
            conditions.push(format!("target_family = \"{}\"", target_family));
        }
    }

    // Add architecture conditions
    for arch in &platform_info.architectures {
        conditions.push(format!("target_arch = \"{}\"", arch.to_cfg_target_arch()));
    }

    if conditions.is_empty() {
        None
    } else if conditions.len() == 1 {
        Some(format!("#[cfg({})]", conditions[0]))
    } else {
        // Multiple conditions - use any()
        Some(format!("#[cfg(any({}))]", conditions.join(", ")))
    }
}

/// Generate platform-specific wrapper function
pub fn generate_platform_specific_wrapper(
    func: &FfiFunction,
    platform_info: &PlatformInfo,
    impl_code: &str,
) -> String {
    let mut output = String::new();

    // Generate doc comment noting platform specificity
    if let Some(cfg_attr) = generate_cfg_attribute(platform_info) {
        writeln!(&mut output, "    /// **Platform-specific:** Available on:").unwrap();
        for platform in &platform_info.available_on {
            writeln!(&mut output, "    /// - {:?}", platform).unwrap();
        }
        
        for note in &platform_info.platform_notes {
            writeln!(&mut output, "    /// - {}: {}", 
                     if note.is_limitation { "⚠️" } else { "ℹ️" },
                     note.note).unwrap();
        }
        writeln!(&mut output).unwrap();
        
        // Add cfg attribute
        writeln!(&mut output, "    {}", cfg_attr).unwrap();
    }

    // Add the actual implementation
    output.push_str(impl_code);
    
    output
}

/// Generate platform detection utilities
pub fn generate_platform_utils() -> String {
    r#"/// Platform detection utilities
#[cfg(test)]
mod platform_utils {
    /// Check if running on Windows
    #[cfg(target_os = "windows")]
    pub fn is_windows() -> bool { true }
    #[cfg(not(target_os = "windows"))]
    pub fn is_windows() -> bool { false }

    /// Check if running on Linux
    #[cfg(target_os = "linux")]
    pub fn is_linux() -> bool { true }
    #[cfg(not(target_os = "linux"))]
    pub fn is_linux() -> bool { false }

    /// Check if running on macOS
    #[cfg(target_os = "macos")]
    pub fn is_macos() -> bool { true }
    #[cfg(not(target_os = "macos"))]
    pub fn is_macos() -> bool { false }

    /// Check if running on Unix-like system
    #[cfg(unix)]
    pub fn is_unix() -> bool { true }
    #[cfg(not(unix))]
    pub fn is_unix() -> bool { false }

    /// Get current platform name
    pub fn current_platform() -> &'static str {
        if cfg!(target_os = "windows") {
            "Windows"
        } else if cfg!(target_os = "linux") {
            "Linux"
        } else if cfg!(target_os = "macos") {
            "macOS"
        } else if cfg!(unix) {
            "Unix"
        } else {
            "Unknown"
        }
    }
}
"#.to_string()
}

/// Generate cross-platform tests for a function
pub fn generate_cross_platform_tests(
    func: &FfiFunction,
    platform_info: Option<&PlatformInfo>,
) -> String {
    let mut code = String::new();

    let func_test_name = format!("test_{}_cross_platform", func.name.to_lowercase());
    
    writeln!(&mut code, "    #[test]").unwrap();
    writeln!(&mut code, "    fn {}() {{", func_test_name).unwrap();
    
    if let Some(info) = platform_info {
        if info.is_platform_specific() {
            writeln!(&mut code, "        // This function is platform-specific").unwrap();
            writeln!(&mut code, "        #[cfg(any(").unwrap();
            for (i, platform) in info.available_on.iter().enumerate() {
                if i > 0 {
                    write!(&mut code, ",").unwrap();
                }
                if let Some(target_os) = platform.to_cfg_target_os() {
                    write!(&mut code, "target_os = \"{}\"", target_os).unwrap();
                }
            }
            writeln!(&mut code, "))]").unwrap();
            writeln!(&mut code, "        {{").unwrap();
            writeln!(&mut code, "            // Test on supported platform").unwrap();
            writeln!(&mut code, "            let result = {}(/* params */);", func.name).unwrap();
            writeln!(&mut code, "            // Platform-specific assertions").unwrap();
            writeln!(&mut code, "        }}").unwrap();
            writeln!(&mut code).unwrap();
            writeln!(&mut code, "        #[cfg(not(any(").unwrap();
            for (i, platform) in info.available_on.iter().enumerate() {
                if i > 0 {
                    write!(&mut code, ",").unwrap();
                }
                if let Some(target_os) = platform.to_cfg_target_os() {
                    write!(&mut code, "target_os = \"{}\"", target_os).unwrap();
                }
            }
            writeln!(&mut code, ")))]").unwrap();
            writeln!(&mut code, "        {{").unwrap();
            writeln!(&mut code, "            // Not available on this platform").unwrap();
            writeln!(&mut code, "            // Verify compile error or provide stub").unwrap();
            writeln!(&mut code, "        }}").unwrap();
        } else {
            // Available on all platforms
            writeln!(&mut code, "        // This function is available on all platforms").unwrap();
            writeln!(&mut code, "        let result = {}(/* params */);", func.name).unwrap();
            writeln!(&mut code, "        // Cross-platform assertions").unwrap();
        }
    } else {
        // No platform info available, assume cross-platform
        writeln!(&mut code, "        // Assuming cross-platform availability").unwrap();
        writeln!(&mut code, "        let result = {}(/* params */);", func.name).unwrap();
    }
    
    writeln!(&mut code, "    }}").unwrap();
    
    code
}

/// Generate platform compatibility documentation
pub fn generate_platform_docs(
    functions: &[FfiFunction],
    platform_contexts: &HashMap<String, PlatformInfo>,
) -> String {
    let mut doc = String::new();

    writeln!(&mut doc, "# Platform Compatibility\n").unwrap();
    writeln!(&mut doc, "This document describes platform-specific behavior and requirements.\n").unwrap();

    // Group functions by platform availability
    let mut windows_only = Vec::new();
    let mut linux_only = Vec::new();
    let mut macos_only = Vec::new();
    let mut cross_platform = Vec::new();

    for func in functions {
        if let Some(info) = platform_contexts.get(&func.name) {
            if info.available_on.contains(&Platform::Windows) && info.available_on.len() == 1 {
                windows_only.push(&func.name);
            } else if info.available_on.contains(&Platform::Linux) && info.available_on.len() == 1 {
                linux_only.push(&func.name);
            } else if info.available_on.contains(&Platform::MacOS) && info.available_on.len() == 1 {
                macos_only.push(&func.name);
            } else if !info.available_on.is_empty() {
                cross_platform.push((&func.name, info));
            }
        }
    }

    if !windows_only.is_empty() {
        writeln!(&mut doc, "## Windows-Only Functions\n").unwrap();
        for func in windows_only {
            writeln!(&mut doc, "- `{}`", func).unwrap();
        }
        writeln!(&mut doc).unwrap();
    }

    if !linux_only.is_empty() {
        writeln!(&mut doc, "## Linux-Only Functions\n").unwrap();
        for func in linux_only {
            writeln!(&mut doc, "- `{}`", func).unwrap();
        }
        writeln!(&mut doc).unwrap();
    }

    if !macos_only.is_empty() {
        writeln!(&mut doc, "## macOS-Only Functions\n").unwrap();
        for func in macos_only {
            writeln!(&mut doc, "- `{}`", func).unwrap();
        }
        writeln!(&mut doc).unwrap();
    }

    writeln!(&mut doc, "## Testing Recommendations\n").unwrap();
    writeln!(&mut doc, "- Test on all target platforms before release").unwrap();
    writeln!(&mut doc, "- Use conditional compilation for platform-specific code").unwrap();
    writeln!(&mut doc, "- Provide clear error messages for unsupported platforms").unwrap();
    writeln!(&mut doc, "- Consider CI/CD testing on Windows, Linux, and macOS").unwrap();

    doc
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn test_cfg_attribute_generation() {
        let mut info = PlatformInfo::new("test_func".to_string());
        info.available_on.insert(Platform::Windows);
        
        let attr = generate_cfg_attribute(&info);
        assert!(attr.is_some());
        assert!(attr.unwrap().contains("target_os = \"windows\""));
    }

    #[test]
    fn test_multiple_platforms() {
        let mut info = PlatformInfo::new("test_func".to_string());
        info.available_on.insert(Platform::Windows);
        info.available_on.insert(Platform::Linux);
        
        let attr = generate_cfg_attribute(&info);
        assert!(attr.is_some());
        let attr_str = attr.unwrap();
        assert!(attr_str.contains("any("));
        assert!(attr_str.contains("target_os = \"windows\""));
        assert!(attr_str.contains("target_os = \"linux\""));
    }

    #[test]
    fn test_platform_utils_generation() {
        let utils = generate_platform_utils();
        assert!(utils.contains("is_windows"));
        assert!(utils.contains("is_linux"));
        assert!(utils.contains("is_macos"));
        assert!(utils.contains("current_platform"));
    }
}
