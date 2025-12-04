//! Version compatibility tracking across library versions
//!
//! This module tracks API changes, deprecations, and compatibility
//! across different versions of the FFI library.

use crate::ffi::FfiInfo;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::info;

/// Version compatibility information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersionCompatibility {
    /// API version information
    pub versions: Vec<ApiVersion>,
    /// Compatibility matrix
    pub compatibility: CompatibilityMatrix,
    /// Migration guides
    pub migrations: Vec<MigrationGuide>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiVersion {
    /// Version number (e.g., "9.0", "10.0")
    pub version: String,
    /// Release date
    pub release_date: Option<String>,
    /// Added functions
    pub added_functions: Vec<String>,
    /// Removed functions
    pub removed_functions: Vec<String>,
    /// Changed functions
    pub changed_functions: Vec<FunctionChange>,
    /// Added types
    pub added_types: Vec<String>,
    /// Deprecated items
    pub deprecated: Vec<DeprecationInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionChange {
    /// Function name
    pub name: String,
    /// What changed
    pub change_type: ChangeType,
    /// Description
    pub description: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ChangeType {
    /// Signature changed
    SignatureChange,
    /// Behavior changed
    BehaviorChange,
    /// Performance improved
    PerformanceChange,
    /// Bug fix
    BugFix,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeprecationInfo {
    /// Deprecated item name
    pub name: String,
    /// Version deprecated in
    pub deprecated_in: String,
    /// Version removed in (if known)
    pub removed_in: Option<String>,
    /// Replacement
    pub replacement: Option<String>,
    /// Reason for deprecation
    pub reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompatibilityMatrix {
    /// Minimum supported version
    pub min_version: String,
    /// Maximum tested version
    pub max_version: String,
    /// Version-specific requirements
    pub requirements: HashMap<String, Vec<String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MigrationGuide {
    /// From version
    pub from_version: String,
    /// To version
    pub to_version: String,
    /// Migration steps
    pub steps: Vec<MigrationStep>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MigrationStep {
    /// Step description
    pub description: String,
    /// Old code example
    pub old_code: Option<String>,
    /// New code example
    pub new_code: Option<String>,
    /// Breaking change?
    pub breaking: bool,
}

/// Analyzes version compatibility
pub struct VersionCompatibilityAnalyzer;

impl VersionCompatibilityAnalyzer {
    pub fn new() -> Self {
        Self
    }

    /// Analyze version compatibility
    pub fn analyze(&self, ffi_info: &FfiInfo) -> VersionCompatibility {
        let versions = self.extract_version_info(ffi_info);
        let compatibility = self.build_compatibility_matrix(ffi_info, &versions);
        let migrations = self.generate_migration_guides(&versions);

        info!(
            "Tracked {} API versions with {} migrations",
            versions.len(),
            migrations.len()
        );

        VersionCompatibility {
            versions,
            compatibility,
            migrations,
        }
    }

    /// Extract version information from FFI
    fn extract_version_info(&self, ffi_info: &FfiInfo) -> Vec<ApiVersion> {
        let mut versions = Vec::new();
        let mut version_map: HashMap<String, ApiVersion> = HashMap::new();

        // Extract from function documentation
        for func in &ffi_info.functions {
            if let Some(ref docs) = func.docs {
                if let Some(version) = self.extract_since_version(docs) {
                    version_map
                        .entry(version.clone())
                        .or_insert_with(|| ApiVersion {
                            version: version.clone(),
                            release_date: None,
                            added_functions: Vec::new(),
                            removed_functions: Vec::new(),
                            changed_functions: Vec::new(),
                            added_types: Vec::new(),
                            deprecated: Vec::new(),
                        })
                        .added_functions
                        .push(func.name.clone());
                }

                if let Some(deprecation) = self.extract_deprecation_info(docs, &func.name) {
                    if let Some(version) = version_map.get_mut(&deprecation.deprecated_in) {
                        version.deprecated.push(deprecation);
                    }
                }
            }
        }

        versions.extend(version_map.into_values());
        versions.sort_by(|a, b| a.version.cmp(&b.version));
        versions
    }

    /// Extract @since version from documentation
    fn extract_since_version(&self, docs: &str) -> Option<String> {
        let patterns = [
            r"@since\s+(\d+\.\d+)",
            r"Available since\s+version\s+(\d+\.\d+)",
            r"Added in\s+(\d+\.\d+)",
        ];

        for pattern in &patterns {
            if let Some(captures) = regex::Regex::new(pattern).ok()?.captures(docs) {
                return Some(captures.get(1)?.as_str().to_string());
            }
        }

        None
    }

    /// Extract deprecation information
    fn extract_deprecation_info(&self, docs: &str, name: &str) -> Option<DeprecationInfo> {
        if !docs.to_lowercase().contains("deprecated") {
            return None;
        }

        let deprecated_in = self
            .extract_since_version(docs)
            .unwrap_or_else(|| "unknown".to_string());

        let replacement = if let Some(pos) = docs.find("use") {
            let remaining = &docs[pos + 3..];
            if let Some(end) = remaining.find(|c: char| c.is_whitespace() || c == '.' || c == ')') {
                Some(remaining[..end].trim().to_string())
            } else {
                None
            }
        } else {
            None
        };

        Some(DeprecationInfo {
            name: name.to_string(),
            deprecated_in,
            removed_in: None,
            replacement,
            reason: "See documentation for details".to_string(),
        })
    }

    /// Build compatibility matrix
    fn build_compatibility_matrix(
        &self,
        _ffi_info: &FfiInfo,
        versions: &[ApiVersion],
    ) -> CompatibilityMatrix {
        let min_version = versions
            .first()
            .map(|v| v.version.clone())
            .unwrap_or_else(|| "1.0".to_string());
        let max_version = versions
            .last()
            .map(|v| v.version.clone())
            .unwrap_or_else(|| "1.0".to_string());

        CompatibilityMatrix {
            min_version,
            max_version,
            requirements: HashMap::new(),
        }
    }

    /// Generate migration guides between versions
    fn generate_migration_guides(&self, versions: &[ApiVersion]) -> Vec<MigrationGuide> {
        let mut guides = Vec::new();

        for i in 0..versions.len().saturating_sub(1) {
            let from = &versions[i];
            let to = &versions[i + 1];

            let mut steps = Vec::new();

            // Add steps for deprecated items
            for deprecated in &from.deprecated {
                if let Some(ref replacement) = deprecated.replacement {
                    steps.push(MigrationStep {
                        description: format!("Replace {} with {}", deprecated.name, replacement),
                        old_code: Some(format!("{}(...)", deprecated.name)),
                        new_code: Some(format!("{}(...)", replacement)),
                        breaking: false,
                    });
                }
            }

            // Add steps for removed items
            for removed in &to.removed_functions {
                steps.push(MigrationStep {
                    description: format!("Remove usage of {}", removed),
                    old_code: Some(format!("{}(...)", removed)),
                    new_code: None,
                    breaking: true,
                });
            }

            if !steps.is_empty() {
                guides.push(MigrationGuide {
                    from_version: from.version.clone(),
                    to_version: to.version.clone(),
                    steps,
                });
            }
        }

        guides
    }

    /// Generate compatibility documentation
    pub fn generate_docs(&self, compat: &VersionCompatibility) -> String {
        let mut docs = String::new();

        docs.push_str("# Version Compatibility\n\n");
        docs.push_str(&format!(
            "Supported versions: {} - {}\n\n",
            compat.compatibility.min_version, compat.compatibility.max_version
        ));

        // Version history
        docs.push_str("## Version History\n\n");
        for version in &compat.versions {
            docs.push_str(&format!("### Version {}\n\n", version.version));

            if !version.added_functions.is_empty() {
                docs.push_str(&format!(
                    "- Added {} functions\n",
                    version.added_functions.len()
                ));
            }

            if !version.deprecated.is_empty() {
                docs.push_str(&format!(
                    "- Deprecated {} items\n",
                    version.deprecated.len()
                ));
            }

            docs.push('\n');
        }

        // Migration guides
        if !compat.migrations.is_empty() {
            docs.push_str("## Migration Guides\n\n");
            for guide in &compat.migrations {
                docs.push_str(&format!(
                    "### Migrating from {} to {}\n\n",
                    guide.from_version, guide.to_version
                ));

                for (i, step) in guide.steps.iter().enumerate() {
                    docs.push_str(&format!("{}. {}\n", i + 1, step.description));
                    if step.breaking {
                        docs.push_str("   **⚠️ Breaking change**\n");
                    }
                    docs.push('\n');
                }
            }
        }

        docs
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ffi::FfiFunction;

    #[test]
    fn test_version_extraction() {
        let mut ffi_info = FfiInfo::default();
        ffi_info.functions.push(FfiFunction {
            name: "newFunc".to_string(),
            params: vec![],
            return_type: "void".to_string(),
            docs: Some("@since 2.0".to_string()),
        });

        let analyzer = VersionCompatibilityAnalyzer::new();
        let compat = analyzer.analyze(&ffi_info);

        assert!(!compat.versions.is_empty());
        assert!(compat.versions.iter().any(|v| v.version == "2.0"));
    }

    #[test]
    fn test_deprecation_extraction() {
        let analyzer = VersionCompatibilityAnalyzer::new();
        let docs = "Deprecated in 3.0. Use newFunc instead.";
        let deprecation = analyzer.extract_deprecation_info(docs, "oldFunc");

        assert!(deprecation.is_some());
        let dep = deprecation.unwrap();
        assert_eq!(dep.name, "oldFunc");
    }

    #[test]
    fn test_migration_guide_generation() {
        let versions = vec![
            ApiVersion {
                version: "1.0".to_string(),
                release_date: None,
                added_functions: vec![],
                removed_functions: vec![],
                changed_functions: vec![],
                added_types: vec![],
                deprecated: vec![DeprecationInfo {
                    name: "oldFunc".to_string(),
                    deprecated_in: "1.0".to_string(),
                    removed_in: Some("2.0".to_string()),
                    replacement: Some("newFunc".to_string()),
                    reason: "Old API".to_string(),
                }],
            },
            ApiVersion {
                version: "2.0".to_string(),
                release_date: None,
                added_functions: vec![],
                removed_functions: vec!["oldFunc".to_string()],
                changed_functions: vec![],
                added_types: vec![],
                deprecated: vec![],
            },
        ];

        let analyzer = VersionCompatibilityAnalyzer::new();
        let guides = analyzer.generate_migration_guides(&versions);

        assert!(!guides.is_empty());
        assert!(guides[0].steps.iter().any(|s| s.breaking));
    }

    #[test]
    fn test_compatibility_matrix() {
        let ffi_info = FfiInfo::default();
        let versions = vec![ApiVersion {
            version: "1.0".to_string(),
            release_date: None,
            added_functions: vec![],
            removed_functions: vec![],
            changed_functions: vec![],
            added_types: vec![],
            deprecated: vec![],
        }];

        let analyzer = VersionCompatibilityAnalyzer::new();
        let matrix = analyzer.build_compatibility_matrix(&ffi_info, &versions);

        assert_eq!(matrix.min_version, "1.0");
    }
}
