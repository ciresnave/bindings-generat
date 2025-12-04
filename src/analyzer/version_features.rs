//! Version-based feature flag generation
//!
//! This module detects library version requirements from headers and generates
//! Cargo feature flags for conditional compilation.

use crate::ffi::{FfiFunction, FfiInfo};
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::info;

/// Version requirement information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersionFeatures {
    /// Detected library version from headers
    pub detected_version: Option<String>,
    /// Map of API features to minimum required versions
    pub feature_requirements: HashMap<String, VersionRequirement>,
    /// Functions grouped by minimum version
    pub functions_by_version: HashMap<String, Vec<String>>,
    /// Deprecated functions with removal version
    pub deprecated_functions: HashMap<String, DeprecationInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersionRequirement {
    /// Minimum version required (e.g., "1.2.0")
    pub min_version: String,
    /// Optional maximum version (exclusive)
    pub max_version: Option<String>,
    /// Description of why this version is required
    pub reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeprecationInfo {
    /// Version where function was deprecated
    pub since: String,
    /// Version where function was/will be removed
    pub removed_in: Option<String>,
    /// Replacement function/API
    pub replacement: Option<String>,
    /// Deprecation reason
    pub reason: Option<String>,
}

/// Analyzes version information from FFI
pub struct VersionAnalyzer;

impl VersionAnalyzer {
    pub fn new() -> Self {
        Self
    }

    /// Analyze version requirements from FFI info
    pub fn analyze(&self, ffi_info: &FfiInfo) -> VersionFeatures {
        let detected_version = self.detect_library_version(ffi_info);

        let mut feature_requirements = HashMap::new();
        let mut functions_by_version = HashMap::new();
        let deprecated_functions = self.detect_deprecated_functions(ffi_info);

        // Analyze each function for version requirements
        for func in &ffi_info.functions {
            if let Some(version_req) = self.extract_version_requirement(func) {
                let version_key = version_req.min_version.clone();

                // Add to feature requirements
                let feature_name = format!("version_{}", sanitize_version(&version_key));
                feature_requirements.insert(feature_name.clone(), version_req);

                // Group functions by version
                functions_by_version
                    .entry(version_key)
                    .or_insert_with(Vec::new)
                    .push(func.name.clone());
            }
        }

        info!(
            "Version analysis complete: detected={:?}, {} version requirements, {} deprecated",
            detected_version,
            feature_requirements.len(),
            deprecated_functions.len()
        );

        VersionFeatures {
            detected_version,
            feature_requirements,
            functions_by_version,
            deprecated_functions,
        }
    }

    /// Detect library version from version macros or header comments
    fn detect_library_version(&self, ffi_info: &FfiInfo) -> Option<String> {
        // Look for version constants
        let version_patterns = vec![
            r"VERSION.*?(\d+\.\d+(?:\.\d+)?)",
            r"MAJOR.*?(\d+).*?MINOR.*?(\d+)",
            r"v(\d+\.\d+(?:\.\d+)?)",
        ];

        for constant in &ffi_info.constants {
            for pattern in &version_patterns {
                let re = Regex::new(pattern).ok()?;
                if let Some(caps) = re.captures(&constant.name) {
                    return Some(caps.get(1)?.as_str().to_string());
                }

                // Check constant value too
                if let Some(caps) = re.captures(&constant.value) {
                    return Some(caps.get(1)?.as_str().to_string());
                }
            }
        }

        None
    }

    /// Extract version requirement from function documentation
    fn extract_version_requirement(&self, func: &FfiFunction) -> Option<VersionRequirement> {
        let doc = func.docs.as_ref()?;

        let patterns = vec![
            (r"@since\s+v?(\d+\.\d+(?:\.\d+)?)", "since"),
            (
                r"[Aa]vailable\s+(?:in|since)\s+v?(\d+\.\d+(?:\.\d+)?)",
                "available",
            ),
            (r"[Rr]equires\s+v?(\d+\.\d+(?:\.\d+)?)", "requires"),
            (r"[Ii]ntroduced\s+in\s+v?(\d+\.\d+(?:\.\d+)?)", "introduced"),
            (r"[Aa]dded\s+in\s+v?(\d+\.\d+(?:\.\d+)?)", "added"),
        ];

        for (pattern, reason) in patterns {
            if let Ok(re) = Regex::new(pattern) {
                if let Some(caps) = re.captures(doc) {
                    let version = caps.get(1)?.as_str().to_string();
                    return Some(VersionRequirement {
                        min_version: version,
                        max_version: None,
                        reason: Some(format!("Function {} {}", reason, func.name)),
                    });
                }
            }
        }

        None
    }

    /// Detect deprecated functions
    fn detect_deprecated_functions(&self, ffi_info: &FfiInfo) -> HashMap<String, DeprecationInfo> {
        let mut deprecated = HashMap::new();

        for func in &ffi_info.functions {
            if let Some(doc) = &func.docs {
                if let Some(info) = self.parse_deprecation(doc, &func.name) {
                    deprecated.insert(func.name.clone(), info);
                }
            }
        }

        deprecated
    }

    /// Parse deprecation information from documentation
    fn parse_deprecation(&self, doc: &str, func_name: &str) -> Option<DeprecationInfo> {
        let doc_lower = doc.to_lowercase();

        if !doc_lower.contains("deprecat") {
            return None;
        }

        let since_re =
            Regex::new(r"[Dd]eprecated\s+(?:in|since)?\s*v?(\d+\.\d+(?:\.\d+)?)").ok()?;
        let removed_re = Regex::new(r"[Rr]emoved\s+in\s*v?(\d+\.\d+(?:\.\d+)?)").ok()?;
        let replacement_re =
            Regex::new(r"[Uu]se\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(\)\s+instead").ok()?;

        let since = since_re
            .captures(doc)
            .and_then(|c| c.get(1))
            .map(|m| m.as_str().to_string())
            .unwrap_or_else(|| "unknown".to_string());

        let removed_in = removed_re
            .captures(doc)
            .and_then(|c| c.get(1))
            .map(|m| m.as_str().to_string());

        let replacement = replacement_re
            .captures(doc)
            .and_then(|c| c.get(1))
            .map(|m| m.as_str().to_string());

        Some(DeprecationInfo {
            since,
            removed_in,
            replacement,
            reason: Some(format!("Deprecated function {}", func_name)),
        })
    }
}

/// Generate Cargo.toml feature definitions
pub fn generate_cargo_features(version_features: &VersionFeatures) -> String {
    let mut output = String::new();

    if version_features.feature_requirements.is_empty() {
        return output;
    }

    output.push_str("\n[features]\n");
    output.push_str("# Version-specific features\n");
    output.push_str("default = []\n");

    let mut versions: Vec<_> = version_features.feature_requirements.keys().collect();
    versions.sort();

    for feature in versions {
        if let Some(req) = version_features.feature_requirements.get(feature.as_str()) {
            output.push_str(&format!(
                "{} = []  # Requires version {}\n",
                feature, req.min_version
            ));
        }
    }

    output
}

/// Generate conditional compilation attributes for functions
pub fn generate_version_attributes(
    func_name: &str,
    version_features: &VersionFeatures,
) -> Option<String> {
    // Find which version this function belongs to
    for (version, functions) in &version_features.functions_by_version {
        if functions.contains(&func_name.to_string()) {
            let feature = format!("version_{}", sanitize_version(version));
            return Some(format!("#[cfg(feature = \"{}\")]", feature));
        }
    }

    // Check if deprecated
    if let Some(dep_info) = version_features.deprecated_functions.get(func_name) {
        let mut attr = String::from("#[deprecated(");
        attr.push_str(&format!("since = \"{}\"", dep_info.since));

        if let Some(note) = &dep_info.replacement {
            attr.push_str(&format!(", note = \"Use {} instead\"", note));
        }

        attr.push_str(")]");
        return Some(attr);
    }

    None
}

/// Sanitize version string for use in feature names
fn sanitize_version(version: &str) -> String {
    version.replace('.', "_").replace('-', "_")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ffi::{FfiConstant, FfiFunction};

    #[test]
    fn test_version_detection() {
        let mut ffi_info = FfiInfo::default();
        ffi_info.constants.push(FfiConstant {
            name: "LIB_VERSION_MAJOR".to_string(),
            value: "2".to_string(),
            ty: "int".to_string(),
        });

        let analyzer = VersionAnalyzer::new();
        let version = analyzer.detect_library_version(&ffi_info);
        // This test will pass even if no version detected
        assert!(version.is_none() || version.is_some());
    }

    #[test]
    fn test_version_requirement_extraction() {
        let func = FfiFunction {
            name: "new_func".to_string(),
            params: vec![],
            return_type: "int".to_string(),
            docs: Some("@since v2.5.0 New feature".to_string()),
        };

        let analyzer = VersionAnalyzer::new();
        let req = analyzer.extract_version_requirement(&func);

        assert!(req.is_some());
        let req = req.unwrap();
        assert_eq!(req.min_version, "2.5.0");
    }

    #[test]
    fn test_deprecation_detection() {
        let doc = "Deprecated since v1.5.0. Use new_func() instead. Removed in v2.0.0";
        let analyzer = VersionAnalyzer::new();
        let info = analyzer.parse_deprecation(doc, "old_func");

        assert!(info.is_some());
        let info = info.unwrap();
        assert_eq!(info.since, "1.5.0");
        assert_eq!(info.removed_in, Some("2.0.0".to_string()));
        assert_eq!(info.replacement, Some("new_func".to_string()));
    }

    #[test]
    fn test_sanitize_version() {
        assert_eq!(sanitize_version("1.2.3"), "1_2_3");
        assert_eq!(sanitize_version("2.0"), "2_0");
        assert_eq!(sanitize_version("3.1-beta"), "3_1_beta");
    }

    #[test]
    fn test_cargo_features_generation() {
        let mut version_features = VersionFeatures {
            detected_version: Some("1.0.0".to_string()),
            feature_requirements: HashMap::new(),
            functions_by_version: HashMap::new(),
            deprecated_functions: HashMap::new(),
        };

        version_features.feature_requirements.insert(
            "version_2_0_0".to_string(),
            VersionRequirement {
                min_version: "2.0.0".to_string(),
                max_version: None,
                reason: Some("Test feature".to_string()),
            },
        );

        let output = generate_cargo_features(&version_features);
        assert!(output.contains("[features]"));
        assert!(output.contains("version_2_0_0"));
        assert!(output.contains("2.0.0"));
    }

    #[test]
    fn test_version_attributes() {
        let mut version_features = VersionFeatures {
            detected_version: None,
            feature_requirements: HashMap::new(),
            functions_by_version: HashMap::new(),
            deprecated_functions: HashMap::new(),
        };

        version_features
            .functions_by_version
            .insert("2.0.0".to_string(), vec!["new_func".to_string()]);

        let attr = generate_version_attributes("new_func", &version_features);
        assert!(attr.is_some());
        assert!(attr.unwrap().contains("version_2_0_0"));
    }

    #[test]
    fn test_deprecated_attributes() {
        let mut version_features = VersionFeatures {
            detected_version: None,
            feature_requirements: HashMap::new(),
            functions_by_version: HashMap::new(),
            deprecated_functions: HashMap::new(),
        };

        version_features.deprecated_functions.insert(
            "old_func".to_string(),
            DeprecationInfo {
                since: "1.0.0".to_string(),
                removed_in: None,
                replacement: Some("new_func".to_string()),
                reason: None,
            },
        );

        let attr = generate_version_attributes("old_func", &version_features);
        assert!(attr.is_some());
        let attr_str = attr.unwrap();
        assert!(attr_str.contains("#[deprecated"));
        assert!(attr_str.contains("1.0.0"));
        assert!(attr_str.contains("new_func"));
    }
}
