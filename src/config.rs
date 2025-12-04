use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Configuration for the bindings generator
///
/// This is the configuration for a specific binding generation run,
/// as opposed to the user's persistent preferences (which are in config/mod.rs).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneratorConfig {
    /// Source identifier (path, URL, or archive)
    pub source: String,

    /// Output directory for the generated crate
    pub output_path: PathBuf,

    /// Library name (auto-detected if None)
    pub lib_name: Option<String>,

    /// Specific header files to process
    pub headers: Vec<PathBuf>,

    /// Additional include directories for dependencies
    pub include_dirs: Vec<PathBuf>,

    /// Link libraries for dependencies
    pub link_libs: Vec<String>,

    /// Link library search paths
    pub lib_paths: Vec<PathBuf>,

    /// Whether to use LLM enhancement
    pub use_llm: bool,

    /// LLM model name (for Ollama)
    pub llm_model: String,

    /// Whether to ask questions interactively
    pub interactive: bool,

    /// Code style profile
    pub style: CodeStyle,

    /// Cache directory for LLM responses
    pub cache_dir: PathBuf,

    /// Dry run mode (don't write files)
    pub dry_run: bool,

    /// Verbose logging
    pub verbose: bool,

    /// Whether to bundle the library into the generated crate (opt-in; obeys license)
    #[serde(default)]
    pub bundle_library: bool,

    /// Whether to enable build-time discovery/download (deprecated; opt-in)
    #[serde(default)]
    pub enable_build_discovery: bool,

    /// Bindgen timeout in seconds (default: 300 = 5 minutes)
    ///
    /// If bindgen takes longer than this, it will be terminated.
    /// Increase for extremely large header files (10k+ functions).
    #[serde(default = "default_bindgen_timeout")]
    pub bindgen_timeout_secs: u64,
}

fn default_bindgen_timeout() -> u64 {
    300 // 5 minutes
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub enum CodeStyle {
    #[default]
    /// Minimal wrappers, close to raw FFI
    Minimal,
    /// Ergonomic Rust API with builders and conveniences
    Ergonomic,
    /// Zero-cost abstractions, no overhead
    ZeroCost,
}

impl std::str::FromStr for CodeStyle {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "minimal" => Ok(CodeStyle::Minimal),
            "ergonomic" => Ok(CodeStyle::Ergonomic),
            "zero-cost" => Ok(CodeStyle::ZeroCost),
            _ => Err(anyhow::anyhow!(
                "Invalid style: {}. Use minimal, ergonomic, or zero-cost",
                s
            )),
        }
    }
}

impl GeneratorConfig {
    /// Create a new configuration from CLI arguments
    pub fn from_cli(cli: &crate::cli::Cli) -> Result<Self> {
        let cache_dir = cli.cache_dir.clone().unwrap_or_else(|| {
            dirs::cache_dir()
                .unwrap_or_else(|| PathBuf::from(".cache"))
                .join("bindings-generat")
        });

        let style = cli.style.parse()?;

        Ok(Self {
            source: cli.source.clone(),
            output_path: cli.get_output_dir(),
            lib_name: cli.lib_name.clone(),
            headers: Vec::new(), // Will be populated during discovery
            include_dirs: cli.include_dirs.clone(),
            link_libs: cli.link_libs.clone(),
            lib_paths: cli.lib_paths.clone(),
            use_llm: !cli.no_llm,
            llm_model: cli.model.clone(),
            interactive: cli.interactive || (!cli.non_interactive),
            style,
            cache_dir,
            dry_run: cli.dry_run,
            verbose: cli.verbose,
            bundle_library: cli.bundle_library,
            enable_build_discovery: cli.enable_build_discovery,
            bindgen_timeout_secs: default_bindgen_timeout(),
        })
    }

    /// Load configuration from a TOML file
    pub fn from_file(path: &PathBuf) -> Result<Self> {
        let content = std::fs::read_to_string(path).context("Failed to read config file")?;
        let config: Self = toml::from_str(&content).context("Failed to parse config file")?;
        Ok(config)
    }

    /// Save configuration to a TOML file
    pub fn save(&self, path: &PathBuf) -> Result<()> {
        let content = toml::to_string_pretty(self).context("Failed to serialize config")?;
        std::fs::write(path, content).context("Failed to write config file")?;
        Ok(())
    }
}

impl Default for GeneratorConfig {
    fn default() -> Self {
        Self {
            source: String::new(),
            output_path: PathBuf::from("bindings-output"),
            lib_name: None,
            headers: Vec::new(),
            include_dirs: Vec::new(),
            link_libs: Vec::new(),
            lib_paths: Vec::new(),
            use_llm: false,
            llm_model: "llama3.2".to_string(),
            interactive: false,
            style: CodeStyle::default(),
            cache_dir: PathBuf::from(".cache"),
            dry_run: false,
            verbose: false,
            bundle_library: false,
            enable_build_discovery: false,
            bindgen_timeout_secs: default_bindgen_timeout(),
        }
    }
}

// Add dirs crate to Cargo.toml dependencies
