//! Platform-specific documentation generation
//!
//! This module generates platform-specific documentation for Windows, Linux, and macOS,
//! including platform differences, build instructions, and compatibility notes.

use crate::ffi::FfiInfo;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::info;

/// Platform-specific documentation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlatformDocs {
    /// Platform-specific information
    pub platforms: Vec<PlatformInfo>,
    /// Platform differences
    pub differences: Vec<PlatformDifference>,
    /// Build instructions per platform
    pub build_instructions: HashMap<String, BuildInstructions>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlatformInfo {
    /// Platform name (Windows, Linux, macOS, etc.)
    pub name: String,
    /// Supported?
    pub supported: bool,
    /// Minimum version requirements
    pub requirements: Vec<String>,
    /// Platform-specific notes
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlatformDifference {
    /// What differs
    pub category: DifferenceCategory,
    /// Function or type name
    pub name: String,
    /// Description of difference
    pub description: String,
    /// Affected platforms
    pub platforms: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum DifferenceCategory {
    /// Function availability
    Availability,
    /// Behavior difference
    Behavior,
    /// Performance difference
    Performance,
    /// Linking requirement
    Linking,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuildInstructions {
    /// Platform name
    pub platform: String,
    /// Prerequisites
    pub prerequisites: Vec<String>,
    /// Build steps
    pub steps: Vec<BuildStep>,
    /// Environment variables
    pub env_vars: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuildStep {
    /// Step description
    pub description: String,
    /// Command to run
    pub command: Option<String>,
}

/// Analyzes platform-specific features
pub struct PlatformDocsAnalyzer;

impl PlatformDocsAnalyzer {
    pub fn new() -> Self {
        Self
    }

    /// Analyze platform-specific features
    pub fn analyze(&self, ffi_info: &FfiInfo) -> PlatformDocs {
        let platforms = self.detect_platforms(ffi_info);
        let differences = self.analyze_differences(ffi_info);
        let build_instructions = self.generate_build_instructions(ffi_info);

        info!(
            "Generated docs for {} platforms with {} differences",
            platforms.len(),
            differences.len()
        );

        PlatformDocs {
            platforms,
            differences,
            build_instructions,
        }
    }

    /// Detect supported platforms
    fn detect_platforms(&self, _ffi_info: &FfiInfo) -> Vec<PlatformInfo> {
        let mut platforms = Vec::new();

        // Windows
        platforms.push(PlatformInfo {
            name: "Windows".to_string(),
            supported: true,
            requirements: vec![
                "Visual Studio 2019 or later".to_string(),
                "Windows 10 or later".to_string(),
            ],
            notes: vec!["Requires MSVC toolchain".to_string()],
        });

        // Linux
        platforms.push(PlatformInfo {
            name: "Linux".to_string(),
            supported: true,
            requirements: vec![
                "GCC 9.0 or later".to_string(),
                "glibc 2.27 or later".to_string(),
            ],
            notes: vec!["Tested on Ubuntu 20.04+".to_string()],
        });

        // macOS
        platforms.push(PlatformInfo {
            name: "macOS".to_string(),
            supported: true,
            requirements: vec![
                "Xcode 12 or later".to_string(),
                "macOS 11.0 or later".to_string(),
            ],
            notes: vec!["Universal binary support".to_string()],
        });

        platforms
    }

    /// Analyze platform differences
    fn analyze_differences(&self, ffi_info: &FfiInfo) -> Vec<PlatformDifference> {
        let mut differences = Vec::new();

        for func in &ffi_info.functions {
            if let Some(ref docs) = func.docs {
                // Check for platform-specific mentions
                if docs.to_lowercase().contains("windows only") {
                    differences.push(PlatformDifference {
                        category: DifferenceCategory::Availability,
                        name: func.name.clone(),
                        description: "Only available on Windows".to_string(),
                        platforms: vec!["Windows".to_string()],
                    });
                }

                if docs.to_lowercase().contains("linux only") {
                    differences.push(PlatformDifference {
                        category: DifferenceCategory::Availability,
                        name: func.name.clone(),
                        description: "Only available on Linux".to_string(),
                        platforms: vec!["Linux".to_string()],
                    });
                }

                // Check for behavior differences
                if docs.contains("platform-dependent") || docs.contains("platform specific") {
                    differences.push(PlatformDifference {
                        category: DifferenceCategory::Behavior,
                        name: func.name.clone(),
                        description: "Behavior may vary by platform".to_string(),
                        platforms: vec![
                            "Windows".to_string(),
                            "Linux".to_string(),
                            "macOS".to_string(),
                        ],
                    });
                }
            }
        }

        differences
    }

    /// Generate platform-specific build instructions
    fn generate_build_instructions(
        &self,
        _ffi_info: &FfiInfo,
    ) -> HashMap<String, BuildInstructions> {
        let mut instructions = HashMap::new();

        // Windows instructions
        instructions.insert(
            "Windows".to_string(),
            BuildInstructions {
                platform: "Windows".to_string(),
                prerequisites: vec![
                    "Visual Studio 2019 or later with C++ support".to_string(),
                    "Rust toolchain (MSVC)".to_string(),
                ],
                steps: vec![
                    BuildStep {
                        description: "Install dependencies".to_string(),
                        command: Some("choco install llvm".to_string()),
                    },
                    BuildStep {
                        description: "Set up MSVC environment".to_string(),
                        command: Some("vcvarsall.bat x64".to_string()),
                    },
                    BuildStep {
                        description: "Build the bindings".to_string(),
                        command: Some("cargo build --release".to_string()),
                    },
                ],
                env_vars: HashMap::from([(
                    "LIBCLANG_PATH".to_string(),
                    "C:\\Program Files\\LLVM\\bin".to_string(),
                )]),
            },
        );

        // Linux instructions
        instructions.insert(
            "Linux".to_string(),
            BuildInstructions {
                platform: "Linux".to_string(),
                prerequisites: vec![
                    "GCC 9.0 or later".to_string(),
                    "Clang/LLVM for bindgen".to_string(),
                    "Rust toolchain".to_string(),
                ],
                steps: vec![
                    BuildStep {
                        description: "Install dependencies".to_string(),
                        command: Some(
                            "sudo apt-get install llvm-dev libclang-dev clang".to_string(),
                        ),
                    },
                    BuildStep {
                        description: "Build the bindings".to_string(),
                        command: Some("cargo build --release".to_string()),
                    },
                ],
                env_vars: HashMap::new(),
            },
        );

        // macOS instructions
        instructions.insert(
            "macOS".to_string(),
            BuildInstructions {
                platform: "macOS".to_string(),
                prerequisites: vec![
                    "Xcode Command Line Tools".to_string(),
                    "Homebrew".to_string(),
                    "Rust toolchain".to_string(),
                ],
                steps: vec![
                    BuildStep {
                        description: "Install dependencies".to_string(),
                        command: Some("brew install llvm".to_string()),
                    },
                    BuildStep {
                        description: "Build the bindings".to_string(),
                        command: Some("cargo build --release".to_string()),
                    },
                ],
                env_vars: HashMap::from([(
                    "LIBCLANG_PATH".to_string(),
                    "/usr/local/opt/llvm/lib".to_string(),
                )]),
            },
        );

        instructions
    }

    /// Generate platform documentation
    pub fn generate_docs(&self, platform_docs: &PlatformDocs) -> String {
        let mut docs = String::new();

        docs.push_str("# Platform Support\n\n");

        // Platform information
        docs.push_str("## Supported Platforms\n\n");
        for platform in &platform_docs.platforms {
            docs.push_str(&format!("### {}\n\n", platform.name));

            if platform.supported {
                docs.push_str("✅ **Supported**\n\n");
            } else {
                docs.push_str("❌ **Not Supported**\n\n");
            }

            if !platform.requirements.is_empty() {
                docs.push_str("**Requirements:**\n");
                for req in &platform.requirements {
                    docs.push_str(&format!("- {}\n", req));
                }
                docs.push('\n');
            }

            if !platform.notes.is_empty() {
                docs.push_str("**Notes:**\n");
                for note in &platform.notes {
                    docs.push_str(&format!("- {}\n", note));
                }
                docs.push('\n');
            }
        }

        // Platform differences
        if !platform_docs.differences.is_empty() {
            docs.push_str("## Platform Differences\n\n");

            for diff in &platform_docs.differences {
                docs.push_str(&format!("### {}\n\n", diff.name));
                docs.push_str(&format!("**Category:** {:?}\n\n", diff.category));
                docs.push_str(&format!("**Description:** {}\n\n", diff.description));
                docs.push_str(&format!("**Platforms:** {}\n\n", diff.platforms.join(", ")));
            }
        }

        // Build instructions
        docs.push_str("## Build Instructions\n\n");
        for (platform_name, instructions) in &platform_docs.build_instructions {
            docs.push_str(&format!("### {}\n\n", platform_name));

            docs.push_str("**Prerequisites:**\n");
            for prereq in &instructions.prerequisites {
                docs.push_str(&format!("- {}\n", prereq));
            }
            docs.push('\n');

            docs.push_str("**Build Steps:**\n\n");
            for (i, step) in instructions.steps.iter().enumerate() {
                docs.push_str(&format!("{}. {}\n", i + 1, step.description));
                if let Some(ref cmd) = step.command {
                    docs.push_str(&format!("   ```bash\n   {}\n   ```\n", cmd));
                }
                docs.push('\n');
            }

            if !instructions.env_vars.is_empty() {
                docs.push_str("**Environment Variables:**\n");
                for (key, value) in &instructions.env_vars {
                    docs.push_str(&format!("- `{}={}`\n", key, value));
                }
                docs.push('\n');
            }
        }

        docs
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ffi::{FfiFunction, FfiInfo};

    #[test]
    fn test_platform_detection() {
        let ffi_info = FfiInfo::default();
        let analyzer = PlatformDocsAnalyzer::new();
        let docs = analyzer.analyze(&ffi_info);

        assert_eq!(docs.platforms.len(), 3);
        assert!(docs.platforms.iter().any(|p| p.name == "Windows"));
        assert!(docs.platforms.iter().any(|p| p.name == "Linux"));
        assert!(docs.platforms.iter().any(|p| p.name == "macOS"));
    }

    #[test]
    fn test_platform_differences() {
        let mut ffi_info = FfiInfo::default();
        ffi_info.functions.push(FfiFunction {
            name: "windowsFunc".to_string(),
            params: vec![],
            return_type: "void".to_string(),
            docs: Some("Windows only function".to_string()),
        });

        let analyzer = PlatformDocsAnalyzer::new();
        let docs = analyzer.analyze(&ffi_info);

        assert!(!docs.differences.is_empty());
        assert!(
            docs.differences
                .iter()
                .any(|d| d.name == "windowsFunc" && d.category == DifferenceCategory::Availability)
        );
    }

    #[test]
    fn test_build_instructions() {
        let ffi_info = FfiInfo::default();
        let analyzer = PlatformDocsAnalyzer::new();
        let docs = analyzer.analyze(&ffi_info);

        assert!(docs.build_instructions.contains_key("Windows"));
        assert!(docs.build_instructions.contains_key("Linux"));
        assert!(docs.build_instructions.contains_key("macOS"));

        let windows_instr = &docs.build_instructions["Windows"];
        assert!(!windows_instr.steps.is_empty());
    }

    #[test]
    fn test_docs_generation() {
        let ffi_info = FfiInfo::default();
        let analyzer = PlatformDocsAnalyzer::new();
        let platform_docs = analyzer.analyze(&ffi_info);
        let docs = analyzer.generate_docs(&platform_docs);

        assert!(docs.contains("Platform Support"));
        assert!(docs.contains("Windows"));
        assert!(docs.contains("Linux"));
        assert!(docs.contains("macOS"));
    }
}
