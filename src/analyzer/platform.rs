//! Platform and version requirement analysis.
//!
//! This module detects platform-specific and version-dependent code by analyzing:
//! - Preprocessor conditionals (#ifdef, #if, etc.)
//! - Documentation annotations (@since, @available, etc.)
//! - Platform-specific header locations
//! - Function naming patterns
//!
//! The analysis generates appropriate Rust `#[cfg(...)]` attributes and documentation.

use std::collections::{HashMap, HashSet};

/// Target platform
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Platform {
    Windows,
    Linux,
    MacOS,
    FreeBSD,
    Android,
    IOS,
    Unix,     // Generic Unix
    Posix,    // POSIX-compliant
    Unknown(String),
}

impl Platform {
    /// Convert to Rust cfg target_os value
    pub fn to_cfg_target_os(&self) -> Option<String> {
        match self {
            Platform::Windows => Some("windows".to_string()),
            Platform::Linux => Some("linux".to_string()),
            Platform::MacOS => Some("macos".to_string()),
            Platform::FreeBSD => Some("freebsd".to_string()),
            Platform::Android => Some("android".to_string()),
            Platform::IOS => Some("ios".to_string()),
            Platform::Unix | Platform::Posix => None, // Use target_family instead
            Platform::Unknown(_) => None,
        }
    }

    /// Convert to Rust cfg target_family value
    pub fn to_cfg_target_family(&self) -> Option<String> {
        match self {
            Platform::Unix | Platform::Posix => Some("unix".to_string()),
            Platform::Windows => Some("windows".to_string()),
            _ => None,
        }
    }
}

/// CPU architecture
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Architecture {
    X86,
    X86_64,
    ARM,
    ARM64,
    MIPS,
    PowerPC,
    Unknown(String),
}

impl Architecture {
    /// Convert to Rust cfg target_arch value
    pub fn to_cfg_target_arch(&self) -> String {
        match self {
            Architecture::X86 => "x86".to_string(),
            Architecture::X86_64 => "x86_64".to_string(),
            Architecture::ARM => "arm".to_string(),
            Architecture::ARM64 => "aarch64".to_string(),
            Architecture::MIPS => "mips".to_string(),
            Architecture::PowerPC => "powerpc".to_string(),
            Architecture::Unknown(s) => s.clone(),
        }
    }
}

/// Version constraint type
#[derive(Debug, Clone, PartialEq)]
pub enum VersionConstraint {
    /// Exact version (== X)
    Exact { major: u32, minor: u32, patch: u32 },
    /// Minimum version (>= X)
    AtLeast { major: u32, minor: u32, patch: u32 },
    /// Maximum version (<= X)
    AtMost { major: u32, minor: u32, patch: u32 },
    /// Range (>= X && <= Y)
    Range {
        min_major: u32,
        min_minor: u32,
        min_patch: u32,
        max_major: u32,
        max_minor: u32,
        max_patch: u32,
    },
}

impl VersionConstraint {
    /// Generate human-readable version string
    pub fn to_string(&self) -> String {
        match self {
            VersionConstraint::Exact {
                major,
                minor,
                patch,
            } => format!("{}.{}.{}", major, minor, patch),
            VersionConstraint::AtLeast {
                major,
                minor,
                patch,
            } => format!(">= {}.{}.{}", major, minor, patch),
            VersionConstraint::AtMost {
                major,
                minor,
                patch,
            } => format!("<= {}.{}.{}", major, minor, patch),
            VersionConstraint::Range {
                min_major,
                min_minor,
                min_patch,
                max_major,
                max_minor,
                max_patch,
            } => format!(
                "{}.{}.{} - {}.{}.{}",
                min_major, min_minor, min_patch, max_major, max_minor, max_patch
            ),
        }
    }
}

/// Version requirement for a library
#[derive(Debug, Clone, PartialEq)]
pub struct VersionRequirement {
    /// Library or API name
    pub library: String,
    /// Version constraint
    pub constraint: VersionConstraint,
    /// Source of requirement (e.g., "#if CUDA_VERSION >= 11000")
    pub source: String,
    /// Confidence (0.0-1.0)
    pub confidence: f64,
}

/// Platform-specific note or requirement
#[derive(Debug, Clone, PartialEq)]
pub struct PlatformNote {
    /// Target platform
    pub platform: Platform,
    /// Note text
    pub note: String,
    /// Whether this is a limitation/restriction
    pub is_limitation: bool,
}

/// Complete platform analysis for a function or type
#[derive(Debug, Clone, PartialEq)]
pub struct PlatformInfo {
    /// Function/type name
    pub name: String,
    /// Platforms this is available on (empty = all platforms)
    pub available_on: HashSet<Platform>,
    /// Architectures this is available on (empty = all architectures)
    pub architectures: HashSet<Architecture>,
    /// Version requirements
    pub version_requirements: Vec<VersionRequirement>,
    /// Platform-specific notes
    pub platform_notes: Vec<PlatformNote>,
    /// Rust #[cfg(...)] attributes to generate
    pub cfg_attributes: Vec<String>,
    /// Feature flags required
    pub required_features: Vec<String>,
    /// Overall confidence (0.0-1.0)
    pub confidence: f64,
}

impl PlatformInfo {
    /// Create new empty platform info
    pub fn new(name: String) -> Self {
        Self {
            name,
            available_on: HashSet::new(),
            architectures: HashSet::new(),
            version_requirements: Vec::new(),
            platform_notes: Vec::new(),
            cfg_attributes: Vec::new(),
            required_features: Vec::new(),
            confidence: 0.0,
        }
    }

    /// Check if platform-specific
    pub fn is_platform_specific(&self) -> bool {
        !self.available_on.is_empty() || !self.architectures.is_empty()
    }

    /// Check if version-gated
    pub fn is_version_gated(&self) -> bool {
        !self.version_requirements.is_empty()
    }

    /// Check if has any restrictions
    pub fn has_restrictions(&self) -> bool {
        self.is_platform_specific() || self.is_version_gated() || !self.required_features.is_empty()
    }

    /// Generate documentation for platform requirements
    pub fn generate_documentation(&self) -> String {
        let mut doc = String::new();

        // Platform availability
        if !self.available_on.is_empty() {
            let platforms: Vec<String> = self
                .available_on
                .iter()
                .map(|p| match p {
                    Platform::Windows => "Windows",
                    Platform::Linux => "Linux",
                    Platform::MacOS => "macOS",
                    Platform::FreeBSD => "FreeBSD",
                    Platform::Android => "Android",
                    Platform::IOS => "iOS",
                    Platform::Unix => "Unix",
                    Platform::Posix => "POSIX",
                    Platform::Unknown(s) => s.as_str(),
                })
                .map(String::from)
                .collect();
            doc.push_str(&format!("/// **Platform Availability:** {}\n", platforms.join(", ")));
        }

        // Architecture restrictions
        if !self.architectures.is_empty() {
            let archs: Vec<String> = self
                .architectures
                .iter()
                .map(|a| match a {
                    Architecture::X86 => "x86",
                    Architecture::X86_64 => "x86-64",
                    Architecture::ARM => "ARM",
                    Architecture::ARM64 => "ARM64",
                    Architecture::MIPS => "MIPS",
                    Architecture::PowerPC => "PowerPC",
                    Architecture::Unknown(s) => s.as_str(),
                })
                .map(String::from)
                .collect();
            doc.push_str(&format!("/// **Architectures:** {}\n", archs.join(", ")));
        }

        // Version requirements
        if !self.version_requirements.is_empty() {
            doc.push_str("///\n");
            doc.push_str("/// # Version Requirements\n");
            for req in &self.version_requirements {
                doc.push_str(&format!(
                    "/// - **{}**: {}\n",
                    req.library,
                    req.constraint.to_string()
                ));
            }
        }

        // Required features
        if !self.required_features.is_empty() {
            doc.push_str("///\n");
            doc.push_str(&format!(
                "/// **Requires Features:** {}\n",
                self.required_features.join(", ")
            ));
        }

        // Platform-specific notes
        if !self.platform_notes.is_empty() {
            doc.push_str("///\n");
            doc.push_str("/// # Platform Notes\n");
            for note in &self.platform_notes {
                let platform_name = match &note.platform {
                    Platform::Windows => "Windows",
                    Platform::Linux => "Linux",
                    Platform::MacOS => "macOS",
                    Platform::FreeBSD => "FreeBSD",
                    Platform::Android => "Android",
                    Platform::IOS => "iOS",
                    Platform::Unix => "Unix",
                    Platform::Posix => "POSIX",
                    Platform::Unknown(s) => s.as_str(),
                };
                doc.push_str(&format!("/// - **{}**: {}\n", platform_name, note.note));
            }
        }

        doc
    }
}

impl Default for PlatformInfo {
    fn default() -> Self {
        Self::new(String::new())
    }
}

/// Platform and version analyzer
#[derive(Debug)]
pub struct PlatformAnalyzer {
    /// Cache of analyzed items
    cache: HashMap<String, PlatformInfo>,
}

impl PlatformAnalyzer {
    /// Create new analyzer
    pub fn new() -> Self {
        Self {
            cache: HashMap::new(),
        }
    }

    /// Analyze platform requirements from source and documentation
    pub fn analyze(
        &mut self,
        name: &str,
        declaration: &str,
        documentation: Option<&str>,
    ) -> PlatformInfo {
        // Check cache first
        if let Some(cached) = self.cache.get(name) {
            return cached.clone();
        }

        let mut info = PlatformInfo::new(name.to_string());

        // Extract from preprocessor directives
        self.extract_preprocessor_platforms(declaration, &mut info);
        self.extract_preprocessor_versions(declaration, &mut info);

        // Extract from documentation
        if let Some(doc) = documentation {
            self.extract_doc_platforms(doc, &mut info);
            self.extract_doc_versions(doc, &mut info);
            self.extract_doc_features(doc, &mut info);
        }

        // Generate cfg attributes
        self.generate_cfg_attributes(&mut info);

        // Calculate confidence
        info.confidence = self.calculate_confidence(&info);

        // Cache result
        self.cache.insert(name.to_string(), info.clone());

        info
    }

    /// Extract platform requirements from preprocessor directives
    fn extract_preprocessor_platforms(&self, declaration: &str, info: &mut PlatformInfo) {
        // Windows detection
        if declaration.contains("#ifdef _WIN32")
            || declaration.contains("#ifdef _WIN64")
            || declaration.contains("#if defined(_WIN32)")
            || declaration.contains("#if defined(_WIN64)")
        {
            info.available_on.insert(Platform::Windows);
        }

        // Linux detection
        if declaration.contains("#ifdef __linux__")
            || declaration.contains("#if defined(__linux__)")
            || declaration.contains("defined(__linux__)")
        {
            info.available_on.insert(Platform::Linux);
        }

        // macOS detection
        if declaration.contains("#ifdef __APPLE__")
            || declaration.contains("#if defined(__APPLE__)")
            || declaration.contains("defined(__APPLE__)")
            || declaration.contains("#ifdef __MACH__")
        {
            info.available_on.insert(Platform::MacOS);
        }

        // FreeBSD detection
        if declaration.contains("#ifdef __FreeBSD__")
            || declaration.contains("#if defined(__FreeBSD__)")
        {
            info.available_on.insert(Platform::FreeBSD);
        }

        // Unix detection
        if declaration.contains("#ifdef __unix__")
            || declaration.contains("#if defined(__unix__)")
            || declaration.contains("#ifdef unix")
        {
            info.available_on.insert(Platform::Unix);
        }

        // POSIX detection
        if declaration.contains("#ifdef _POSIX_VERSION")
            || declaration.contains("#if defined(_POSIX_VERSION)")
        {
            info.available_on.insert(Platform::Posix);
        }

        // Architecture detection
        if declaration.contains("#ifdef __x86_64__")
            || declaration.contains("#ifdef _M_X64")
        {
            info.architectures.insert(Architecture::X86_64);
        }

        if declaration.contains("#ifdef __i386__") || declaration.contains("#ifdef _M_IX86") {
            info.architectures.insert(Architecture::X86);
        }

        if declaration.contains("#ifdef __aarch64__")
            || declaration.contains("#ifdef __arm64__")
            || declaration.contains("#ifdef _M_ARM64")
        {
            info.architectures.insert(Architecture::ARM64);
        }

        if declaration.contains("#ifdef __arm__") || declaration.contains("#ifdef _M_ARM") {
            info.architectures.insert(Architecture::ARM);
        }
    }

    /// Extract version requirements from preprocessor directives
    fn extract_preprocessor_versions(&self, declaration: &str, info: &mut PlatformInfo) {
        // Common version check patterns
        let version_patterns = [
            // CUDA_VERSION >= 11000
            ("CUDA_VERSION", "CUDA"),
            // CUDNN_VERSION >= 8000
            ("CUDNN_VERSION", "cuDNN"),
            // OPENGL_VERSION >= 4
            ("OPENGL_VERSION", "OpenGL"),
            // VULKAN_VERSION >= 1000000
            ("VK_API_VERSION", "Vulkan"),
        ];

        for (version_macro, lib_name) in &version_patterns {
            // Only extract if this specific macro is present
            if declaration.contains(version_macro)
                && let Some(version_num) = self.extract_version_number(declaration, version_macro) {
                    let (major, minor, patch) = self.parse_version_components(version_num);
                    info.version_requirements.push(VersionRequirement {
                        library: lib_name.to_string(),
                        constraint: VersionConstraint::AtLeast {
                            major,
                            minor,
                            patch,
                        },
                        source: format!("#if {} >= {}", version_macro, version_num),
                        confidence: 0.95,
                    });
                }
        }
    }

    /// Extract platform info from documentation
    fn extract_doc_platforms(&self, doc: &str, info: &mut PlatformInfo) {
        let doc_lower = doc.to_lowercase();

        // Platform availability annotations
        if doc_lower.contains("@available")
            || doc_lower.contains("platform:")
            || doc_lower.contains("supported on")
        {
            if doc_lower.contains("windows") {
                info.available_on.insert(Platform::Windows);
            }
            if doc_lower.contains("linux") {
                info.available_on.insert(Platform::Linux);
            }
            if doc_lower.contains("macos") || doc_lower.contains("mac os") {
                info.available_on.insert(Platform::MacOS);
            }
            if doc_lower.contains("unix") {
                info.available_on.insert(Platform::Unix);
            }
        }

        // Platform-specific notes
        if doc_lower.contains("windows:")
            && let Some(note) = self.extract_platform_note(doc, "windows") {
                info.platform_notes.push(PlatformNote {
                    platform: Platform::Windows,
                    note,
                    is_limitation: doc_lower.contains("not available")
                        || doc_lower.contains("not supported"),
                });
            }

        if doc_lower.contains("linux:")
            && let Some(note) = self.extract_platform_note(doc, "linux") {
                info.platform_notes.push(PlatformNote {
                    platform: Platform::Linux,
                    note,
                    is_limitation: doc_lower.contains("not available")
                        || doc_lower.contains("not supported"),
                });
            }

        if (doc_lower.contains("macos:") || doc_lower.contains("mac os:"))
            && let Some(note) = self.extract_platform_note(doc, "macos") {
                info.platform_notes.push(PlatformNote {
                    platform: Platform::MacOS,
                    note,
                    is_limitation: doc_lower.contains("not available")
                        || doc_lower.contains("not supported"),
                });
            }

        // Architecture notes
        if doc_lower.contains("x86-64 only") || doc_lower.contains("x86_64 only") {
            info.architectures.insert(Architecture::X86_64);
        }
        if doc_lower.contains("arm64") || doc_lower.contains("aarch64") {
            info.architectures.insert(Architecture::ARM64);
        }
        if doc_lower.contains("not available on m1")
            || doc_lower.contains("not available on arm")
        {
            info.architectures.insert(Architecture::X86_64);
            info.platform_notes.push(PlatformNote {
                platform: Platform::MacOS,
                note: "Not available on ARM (M1/M2) - x86_64 only".to_string(),
                is_limitation: true,
            });
        }
    }

    /// Extract version requirements from documentation
    fn extract_doc_versions(&self, doc: &str, info: &mut PlatformInfo) {
        let doc_lower = doc.to_lowercase();

        // @since version X.Y.Z
        if (doc_lower.contains("@since") || doc_lower.contains("since version"))
            && let Some((lib, major, minor, patch)) = self.extract_since_version(doc) {
                info.version_requirements.push(VersionRequirement {
                    library: lib,
                    constraint: VersionConstraint::AtLeast {
                        major,
                        minor,
                        patch,
                    },
                    source: "documentation".to_string(),
                    confidence: 0.8,
                });
            }

        // Minimum version: X.Y
        if (doc_lower.contains("minimum version")
            || doc_lower.contains("requires version")
            || doc_lower.contains("version >="))
            && let Some((lib, major, minor, patch)) = self.extract_minimum_version(doc) {
                info.version_requirements.push(VersionRequirement {
                    library: lib,
                    constraint: VersionConstraint::AtLeast {
                        major,
                        minor,
                        patch,
                    },
                    source: "documentation".to_string(),
                    confidence: 0.85,
                });
            }
    }

    /// Extract required features from documentation
    fn extract_doc_features(&self, doc: &str, info: &mut PlatformInfo) {
        let doc_lower = doc.to_lowercase();

        if doc_lower.contains("@requires") || doc_lower.contains("requires:") {
            // Look for common feature requirements
            if doc_lower.contains("cuda toolkit") {
                info.required_features.push("cuda".to_string());
            }
            if doc_lower.contains("cudnn") {
                info.required_features.push("cudnn".to_string());
            }
            if doc_lower.contains("opengl") {
                info.required_features.push("opengl".to_string());
            }
            if doc_lower.contains("vulkan") {
                info.required_features.push("vulkan".to_string());
            }
        }
    }

    /// Generate Rust #[cfg(...)] attributes
    fn generate_cfg_attributes(&self, info: &mut PlatformInfo) {
        // Generate target_os cfg
        if !info.available_on.is_empty() {
            let mut os_cfgs = Vec::new();
            for platform in &info.available_on {
                if let Some(target_os) = platform.to_cfg_target_os() {
                    os_cfgs.push(format!("target_os = \"{}\"", target_os));
                } else if let Some(target_family) = platform.to_cfg_target_family() {
                    os_cfgs.push(format!("target_family = \"{}\"", target_family));
                }
            }

            if os_cfgs.len() == 1 {
                info.cfg_attributes
                    .push(format!("#[cfg({})]", os_cfgs[0]));
            } else if os_cfgs.len() > 1 {
                info.cfg_attributes
                    .push(format!("#[cfg(any({}))]", os_cfgs.join(", ")));
            }
        }

        // Generate target_arch cfg
        if !info.architectures.is_empty() {
            let mut arch_cfgs = Vec::new();
            for arch in &info.architectures {
                arch_cfgs.push(format!(
                    "target_arch = \"{}\"",
                    arch.to_cfg_target_arch()
                ));
            }

            if arch_cfgs.len() == 1 {
                info.cfg_attributes
                    .push(format!("#[cfg({})]", arch_cfgs[0]));
            } else if arch_cfgs.len() > 1 {
                info.cfg_attributes
                    .push(format!("#[cfg(any({}))]", arch_cfgs.join(", ")));
            }
        }

        // Generate feature cfg
        if !info.required_features.is_empty() {
            for feature in &info.required_features {
                info.cfg_attributes
                    .push(format!("#[cfg(feature = \"{}\")]", feature));
            }
        }
    }

    /// Extract version number from pattern
    fn extract_version_number(&self, text: &str, _pattern: &str) -> Option<u32> {
        // Simplified extraction - look for numbers after >= or >
        if let Some(pos) = text.find(">=") {
            let after = &text[pos + 2..];
            if let Some(num_str) = after.split_whitespace().next() {
                return num_str.trim().parse().ok();
            }
        }
        None
    }

    /// Parse version number into components
    fn parse_version_components(&self, version: u32) -> (u32, u32, u32) {
        // Common patterns:
        // CUDA: 11050 = 11.5.0 (major * 1000 + minor * 10)
        // cuDNN: 8006 = 8.0.6 (major * 1000 + minor * 10 + patch)
        let major = version / 1000;
        let minor = (version % 1000) / 10;
        let patch = version % 10;
        (major, minor, patch)
    }

    /// Extract platform-specific note from documentation
    fn extract_platform_note(&self, doc: &str, platform: &str) -> Option<String> {
        let platform_marker = format!("{}:", platform);
        if let Some(start) = doc.to_lowercase().find(&platform_marker) {
            let after = &doc[start + platform_marker.len()..];
            // Extract until newline or period
            let note = after
                .lines()
                .next()?
                .trim()
                .split('.')
                .next()?
                .trim()
                .to_string();
            if !note.is_empty() {
                return Some(note);
            }
        }
        None
    }

    /// Extract @since version from documentation
    fn extract_since_version(&self, doc: &str) -> Option<(String, u32, u32, u32)> {
        let doc_lower = doc.to_lowercase();
        if let Some(pos) = doc_lower.find("@since") {
            let after = &doc[pos + 6..];
            // Look for version pattern: X.Y.Z
            self.parse_version_string(after, "API")
        } else {
            None
        }
    }

    /// Extract minimum version from documentation
    fn extract_minimum_version(&self, doc: &str) -> Option<(String, u32, u32, u32)> {
        // Look for "minimum version: X.Y" or "requires version X.Y"
        let patterns = ["minimum version", "requires version"];
        for pattern in &patterns {
            if let Some(pos) = doc.to_lowercase().find(pattern) {
                let after = &doc[pos + pattern.len()..];
                return self.parse_version_string(after, "API");
            }
        }
        None
    }

    /// Parse version string like "11.5.0" or "8.0"
    fn parse_version_string(&self, text: &str, default_lib: &str) -> Option<(String, u32, u32, u32)> {
        // Find first number sequence with dots
        let trimmed = text.trim();
        let words: Vec<&str> = trimmed.split_whitespace().collect();

        for word in words {
            let parts: Vec<&str> = word.split('.').collect();
            if parts.len() >= 2
                && let Ok(major) = parts[0].parse::<u32>() {
                    let minor = parts.get(1).and_then(|s| s.parse::<u32>().ok()).unwrap_or(0);
                    let patch = parts.get(2).and_then(|s| s.parse::<u32>().ok()).unwrap_or(0);
                    return Some((default_lib.to_string(), major, minor, patch));
                }
        }
        None
    }

    /// Calculate overall confidence
    fn calculate_confidence(&self, info: &PlatformInfo) -> f64 {
        let mut total = 0.0;
        let mut count = 0;

        // High confidence if we have preprocessor directives
        if !info.available_on.is_empty() {
            total += 0.95;
            count += 1;
        }

        if !info.architectures.is_empty() {
            total += 0.9;
            count += 1;
        }

        // Version requirements
        for req in &info.version_requirements {
            total += req.confidence;
            count += 1;
        }

        // Platform notes (lower confidence)
        if !info.platform_notes.is_empty() {
            total += 0.7;
            count += 1;
        }

        if count > 0 {
            total / count as f64
        } else {
            0.0
        }
    }

    /// Clear the cache
    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }
}

impl Default for PlatformAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_windows_ifdef() {
        let mut analyzer = PlatformAnalyzer::new();
        let decl = "#ifdef _WIN32\nvoid func();\n#endif";
        let info = analyzer.analyze("func", decl, None);

        assert!(info.available_on.contains(&Platform::Windows));
        assert!(info.is_platform_specific());
        assert_eq!(info.cfg_attributes.len(), 1);
        assert!(info.cfg_attributes[0].contains("windows"));
    }

    #[test]
    fn test_linux_ifdef() {
        let mut analyzer = PlatformAnalyzer::new();
        let decl = "#ifdef __linux__\nvoid func();\n#endif";
        let info = analyzer.analyze("func", decl, None);

        assert!(info.available_on.contains(&Platform::Linux));
        assert!(info.cfg_attributes[0].contains("linux"));
    }

    #[test]
    fn test_macos_ifdef() {
        let mut analyzer = PlatformAnalyzer::new();
        let decl = "#ifdef __APPLE__\nvoid func();\n#endif";
        let info = analyzer.analyze("func", decl, None);

        assert!(info.available_on.contains(&Platform::MacOS));
        assert!(info.cfg_attributes[0].contains("macos"));
    }

    #[test]
    fn test_multiple_platforms() {
        let mut analyzer = PlatformAnalyzer::new();
        let decl = "#if defined(__linux__) || defined(__APPLE__)\nvoid func();\n#endif";
        let info = analyzer.analyze("func", decl, None);

        assert!(info.available_on.contains(&Platform::Linux));
        assert!(info.available_on.contains(&Platform::MacOS));
        assert!(info.cfg_attributes[0].contains("any"));
    }

    #[test]
    fn test_architecture_detection() {
        let mut analyzer = PlatformAnalyzer::new();
        let decl = "#ifdef __x86_64__\nvoid func();\n#endif";
        let info = analyzer.analyze("func", decl, None);

        assert!(info.architectures.contains(&Architecture::X86_64));
        assert!(info.cfg_attributes.iter().any(|c| c.contains("x86_64")));
    }

    #[test]
    fn test_version_requirement() {
        let mut analyzer = PlatformAnalyzer::new();
        let decl = "#if CUDA_VERSION >= 11050\nvoid func();\n#endif";
        let info = analyzer.analyze("func", decl, None);

        assert_eq!(info.version_requirements.len(), 1);
        assert_eq!(info.version_requirements[0].library, "CUDA");
        assert!(info.is_version_gated());
    }

    #[test]
    fn test_doc_platform_annotation() {
        let mut analyzer = PlatformAnalyzer::new();
        let doc = "@available on Windows and Linux only";
        let info = analyzer.analyze("func", "", Some(doc));

        assert!(info.available_on.contains(&Platform::Windows));
        assert!(info.available_on.contains(&Platform::Linux));
    }

    #[test]
    fn test_doc_since_version() {
        let mut analyzer = PlatformAnalyzer::new();
        let doc = "@since 11.5.0";
        let info = analyzer.analyze("func", "", Some(doc));

        assert_eq!(info.version_requirements.len(), 1);
        let req = &info.version_requirements[0];
        if let VersionConstraint::AtLeast {
            major,
            minor,
            patch,
        } = req.constraint
        {
            assert_eq!(major, 11);
            assert_eq!(minor, 5);
            assert_eq!(patch, 0);
        } else {
            panic!("Expected AtLeast constraint");
        }
    }

    #[test]
    fn test_platform_notes() {
        let mut analyzer = PlatformAnalyzer::new();
        let doc = "Windows: Requires Visual Studio 2019+";
        let info = analyzer.analyze("func", "", Some(doc));

        assert_eq!(info.platform_notes.len(), 1);
        assert_eq!(info.platform_notes[0].platform, Platform::Windows);
        assert!(info.platform_notes[0].note.contains("Visual Studio"));
    }

    #[test]
    fn test_required_features() {
        let mut analyzer = PlatformAnalyzer::new();
        let doc = "@requires CUDA Toolkit with cuDNN enabled";
        let info = analyzer.analyze("func", "", Some(doc));

        assert!(info.required_features.contains(&"cuda".to_string()));
        assert!(info.required_features.contains(&"cudnn".to_string()));
        assert!(info.cfg_attributes.iter().any(|c| c.contains("cuda")));
    }

    #[test]
    fn test_docs_generation() {
        let mut analyzer = PlatformAnalyzer::new();
        let decl = "#ifdef _WIN32\nvoid func();\n#endif";
        let doc = "@since 11.0.0\nRequires: CUDA Toolkit";
        let info = analyzer.analyze("func", decl, Some(doc));

        let docs = info.generate_documentation();
        assert!(docs.contains("Platform Availability"));
        assert!(docs.contains("Windows"));
        assert!(docs.contains("Version Requirements"));
    }

    #[test]
    fn test_confidence_calculation() {
        let mut analyzer = PlatformAnalyzer::new();
        let decl = "#ifdef _WIN32\nvoid func();\n#endif";
        let info = analyzer.analyze("func", decl, None);

        assert!(info.confidence > 0.8);
    }

    #[test]
    fn test_cache() {
        let mut analyzer = PlatformAnalyzer::new();
        let decl = "#ifdef _WIN32\nvoid func();\n#endif";

        let info1 = analyzer.analyze("func", decl, None);
        let info2 = analyzer.analyze("func", decl, None);

        assert_eq!(info1.name, info2.name);
        assert_eq!(info1.available_on, info2.available_on);
    }

    #[test]
    fn test_no_restrictions() {
        let mut analyzer = PlatformAnalyzer::new();
        let info = analyzer.analyze("func", "void func();", None);

        assert!(!info.has_restrictions());
        assert!(!info.is_platform_specific());
        assert!(!info.is_version_gated());
        assert!(info.cfg_attributes.is_empty());
    }
}
