use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Configuration for the bindings generator
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    /// Source identifier (path, URL, or archive)
    pub source: String,

    /// Output directory for the generated crate
    pub output_path: PathBuf,

    /// Library name (auto-detected if None)
    pub lib_name: Option<String>,

    /// Specific header files to process
    pub headers: Vec<PathBuf>,

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
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum CodeStyle {
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

impl Config {
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
            use_llm: !cli.no_llm,
            llm_model: cli.model.clone(),
            interactive: cli.interactive || (!cli.non_interactive),
            style,
            cache_dir,
            dry_run: cli.dry_run,
            verbose: cli.verbose,
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

// Add dirs crate to Cargo.toml dependencies
