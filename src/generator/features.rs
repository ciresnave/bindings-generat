//! Feature flag generation for platform-specific and optional API gating.
//!
//! This module provides functionality to generate Cargo feature flags for:
//! - Platform-specific functions (Windows, Linux, macOS)
//! - Optional function groups to reduce binary size
//!
//! All functions present in the parsed headers are included by default.
//! Feature flags allow selective compilation for optimization.

use crate::ffi::FfiFunction;
use std::collections::BTreeMap;

/// Represents a feature with its associated functions.
#[derive(Debug, Clone)]
pub struct Feature {
    pub name: String,
    pub description: String,
    pub functions: Vec<String>,
    pub enabled_by_default: bool,
}

impl Feature {
    /// Generate the cfg attribute for this feature.
    pub fn cfg_attr(&self) -> String {
        format!(r#"#[cfg(feature = "{}")]"#, self.name)
    }
}

/// Platform detection from function documentation or naming patterns.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Platform {
    Windows,
    Linux,
    MacOS,
    Unix, // Linux + macOS
}

impl Platform {
    pub fn feature_name(&self) -> &'static str {
        match self {
            Platform::Windows => "platform-windows",
            Platform::Linux => "platform-linux",
            Platform::MacOS => "platform-macos",
            Platform::Unix => "platform-unix",
        }
    }

    pub fn cfg_attr(&self) -> &'static str {
        match self {
            Platform::Windows => r#"#[cfg(target_os = "windows")]"#,
            Platform::Linux => r#"#[cfg(target_os = "linux")]"#,
            Platform::MacOS => r#"#[cfg(target_os = "macos")]"#,
            Platform::Unix => r#"#[cfg(unix)]"#,
        }
    }
}

/// Detect platform-specific functions from documentation or naming.
///
/// Patterns detected:
/// - Documentation: "Windows only", "Linux-specific", "macOS only"
/// - Function names: "Win32", "linux_", "darwin_", "unix_"
/// - Platform-specific types: HANDLE, HINSTANCE (Windows)
pub fn detect_platform(func: &FfiFunction) -> Option<Platform> {
    let name_lower = func.name.to_lowercase();

    // Check documentation for platform markers
    if let Some(docs) = &func.docs {
        let docs_lower = docs.to_lowercase();

        if docs_lower.contains("windows only") || docs_lower.contains("win32 only") {
            return Some(Platform::Windows);
        }
        if docs_lower.contains("linux only") || docs_lower.contains("linux-specific") {
            return Some(Platform::Linux);
        }
        if docs_lower.contains("macos only") || docs_lower.contains("darwin only") {
            return Some(Platform::MacOS);
        }
        if docs_lower.contains("unix only") {
            return Some(Platform::Unix);
        }
    }

    // Check function name patterns
    if name_lower.contains("win32") || name_lower.contains("windows") {
        return Some(Platform::Windows);
    }
    if name_lower.contains("linux") {
        return Some(Platform::Linux);
    }
    if name_lower.contains("darwin") || name_lower.contains("macos") {
        return Some(Platform::MacOS);
    }
    if name_lower.contains("unix") || name_lower.contains("posix") {
        return Some(Platform::Unix);
    }

    // Check for Windows-specific types in parameters
    for param in &func.params {
        let ty_upper = param.ty.to_uppercase();
        if ty_upper.contains("HANDLE")
            || ty_upper.contains("HWND")
            || ty_upper.contains("HINSTANCE")
            || ty_upper.contains("HMODULE")
        {
            return Some(Platform::Windows);
        }
    }

    None
}

/// Group functions by detected platform.
pub fn group_by_platform(functions: &[FfiFunction]) -> BTreeMap<Platform, Vec<String>> {
    let mut platform_map: BTreeMap<Platform, Vec<String>> = BTreeMap::new();

    for func in functions {
        if let Some(platform) = detect_platform(func) {
            platform_map
                .entry(platform)
                .or_default()
                .push(func.name.clone());
        }
    }

    platform_map
}

/// Generate Cargo.toml feature definitions for platform-specific APIs.
pub fn generate_platform_features(functions: &[FfiFunction]) -> String {
    let platform_map = group_by_platform(functions);

    if platform_map.is_empty() {
        return String::new();
    }

    let mut features = String::new();
    features.push_str("# Platform-specific feature flags\n");
    features.push_str("# These are automatically enabled on their respective platforms\n");
    features.push_str("[features]\n");

    for (platform, funcs) in &platform_map {
        features.push_str(&format!(
            "{} = []  # {} functions\n",
            platform.feature_name(),
            funcs.len()
        ));
    }

    features.push('\n');
    features
}

/// Generate feature documentation for README or module docs.
pub fn generate_feature_docs(functions: &[FfiFunction]) -> String {
    let platform_map = group_by_platform(functions);

    if platform_map.is_empty() {
        return String::new();
    }

    let mut docs = String::new();
    docs.push_str("## Feature Flags\n\n");
    docs.push_str("This crate uses feature flags for platform-specific APIs:\n\n");

    for (platform, funcs) in &platform_map {
        docs.push_str(&format!(
            "- `{}`: {} platform-specific functions\n",
            platform.feature_name(),
            funcs.len()
        ));
    }

    docs.push_str("\nPlatform features are automatically enabled on the target platform.\n");
    docs.push_str("You can also manually enable them in `Cargo.toml`:\n\n");
    docs.push_str("```toml\n");
    docs.push_str("[dependencies]\n");
    docs.push_str(
        "generated-bindings = { version = \"0.1\", features = [\"platform-windows\"] }\n",
    );
    docs.push_str("```\n");

    docs
}

/// Add platform-specific cfg attributes to generated code.
pub fn add_platform_gate(code: &str, func: &FfiFunction) -> String {
    if let Some(platform) = detect_platform(func) {
        // Add both the feature gate AND the platform cfg
        // This ensures the function is only available on the right platform
        // even if the feature is manually enabled
        let feature_cfg = format!(r#"#[cfg(feature = "{}")]"#, platform.feature_name());
        let platform_cfg = platform.cfg_attr();
        format!("{}\n{}\n{}", feature_cfg, platform_cfg, code)
    } else {
        code.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ffi::FfiParam;

    #[test]
    fn test_detect_platform_from_docs_windows() {
        let func = FfiFunction {
            name: "someFunc".to_string(),
            params: vec![],
            return_type: "int".to_string(),
            docs: Some("Does something. Windows only.".to_string()),
        };
        assert_eq!(detect_platform(&func), Some(Platform::Windows));
    }

    #[test]
    fn test_detect_platform_from_docs_linux() {
        let func = FfiFunction {
            name: "someFunc".to_string(),
            params: vec![],
            return_type: "void".to_string(),
            docs: Some("Linux-specific implementation".to_string()),
        };
        assert_eq!(detect_platform(&func), Some(Platform::Linux));
    }

    #[test]
    fn test_detect_platform_from_name() {
        let func = FfiFunction {
            name: "get_win32_handle".to_string(),
            params: vec![],
            return_type: "void*".to_string(),
            docs: None,
        };
        assert_eq!(detect_platform(&func), Some(Platform::Windows));
    }

    #[test]
    fn test_detect_platform_from_params() {
        let func = FfiFunction {
            name: "doSomething".to_string(),
            params: vec![FfiParam {
                name: "handle".to_string(),
                ty: "HANDLE".to_string(),
                is_pointer: false,
                is_mut: false,
            }],
            return_type: "int".to_string(),
            docs: None,
        };
        assert_eq!(detect_platform(&func), Some(Platform::Windows));
    }

    #[test]
    fn test_no_platform_detected() {
        let func = FfiFunction {
            name: "genericFunc".to_string(),
            params: vec![],
            return_type: "int".to_string(),
            docs: Some("Cross-platform function".to_string()),
        };
        assert_eq!(detect_platform(&func), None);
    }

    #[test]
    fn test_group_by_platform() {
        let functions = vec![
            FfiFunction {
                name: "win32_func".to_string(),
                params: vec![],
                return_type: "int".to_string(),
                docs: None,
            },
            FfiFunction {
                name: "linux_func".to_string(),
                params: vec![],
                return_type: "int".to_string(),
                docs: None,
            },
            FfiFunction {
                name: "another_win32".to_string(),
                params: vec![],
                return_type: "int".to_string(),
                docs: None,
            },
        ];

        let platform_map = group_by_platform(&functions);
        assert_eq!(platform_map.get(&Platform::Windows).unwrap().len(), 2);
        assert_eq!(platform_map.get(&Platform::Linux).unwrap().len(), 1);
    }

    #[test]
    fn test_add_platform_gate() {
        let func = FfiFunction {
            name: "win32_func".to_string(),
            params: vec![],
            return_type: "int".to_string(),
            docs: None,
        };

        let code = "pub fn func() {}";
        let gated = add_platform_gate(code, &func);
        assert!(gated.contains(r#"#[cfg(feature = "platform-windows")]"#));
        assert!(gated.contains(r#"#[cfg(target_os = "windows")]"#));
        assert!(gated.contains("pub fn func() {}"));
    }

    #[test]
    fn test_generate_platform_features() {
        let functions = vec![
            FfiFunction {
                name: "win32_func".to_string(),
                params: vec![],
                return_type: "int".to_string(),
                docs: None,
            },
            FfiFunction {
                name: "linux_func".to_string(),
                params: vec![],
                return_type: "int".to_string(),
                docs: None,
            },
        ];

        let features = generate_platform_features(&functions);
        assert!(features.contains("platform-windows"));
        assert!(features.contains("platform-linux"));
    }
}
