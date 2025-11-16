use clap::Parser;
use std::path::PathBuf;

/// Automatically generate safe, idiomatic Rust wrapper crates from C/C++ libraries
#[derive(Parser, Debug, Clone)]
#[command(name = "bindings-generat")]
#[command(version, about, long_about = None)]
pub struct Cli {
    /// Path or URL to C/C++ library source
    ///
    /// Can be:
    /// - A directory containing headers and libraries
    /// - A local archive file (.zip, .tar.gz, .tar, .gz)
    /// - A URL to a remote archive (http:// or https://)
    ///
    /// Archives are automatically extracted to a temporary directory.
    #[arg(value_name = "SOURCE")]
    pub source: String,

    /// Output directory for generated crate (default: ./bindings-output)
    #[arg(short = 'o', long, value_name = "PATH")]
    pub output: Option<PathBuf>,

    /// Specific headers to include (glob pattern)
    #[arg(long, value_name = "GLOB")]
    pub headers: Option<String>,

    /// Library name override (default: auto-detect)
    #[arg(long, value_name = "NAME")]
    pub lib_name: Option<String>,

    /// LLM model for Ollama (default: deepseek-coder:6.7b)
    #[arg(long, value_name = "MODEL", default_value = "deepseek-coder:6.7b")]
    pub model: String,

    /// Skip LLM enhancement (faster, less documentation)
    #[arg(long)]
    pub no_llm: bool,

    /// Ask questions for ambiguous cases
    #[arg(short = 'i', long)]
    pub interactive: bool,

    /// Make best guesses, no questions
    #[arg(long)]
    pub non_interactive: bool,

    /// Code style profile (minimal/ergonomic/zero-cost)
    #[arg(long, value_name = "PROFILE", default_value = "ergonomic")]
    pub style: String,

    /// LLM response cache location
    #[arg(long, value_name = "PATH")]
    pub cache_dir: Option<PathBuf>,

    /// Show what would be generated without writing files
    #[arg(long)]
    pub dry_run: bool,

    /// Verbose output
    #[arg(short = 'v', long)]
    pub verbose: bool,
}

impl Cli {
    /// Get the output directory, using default if not specified
    pub fn get_output_dir(&self) -> PathBuf {
        self.output
            .clone()
            .unwrap_or_else(|| PathBuf::from("./bindings-output"))
    }

    /// Validate CLI arguments
    pub fn validate(&self) -> anyhow::Result<()> {
        // Validate source (only if it's not a URL)
        if !self.source.starts_with("http://") && !self.source.starts_with("https://") {
            let path = PathBuf::from(&self.source);
            if !path.exists() {
                anyhow::bail!("Source path does not exist: {}", self.source);
            }
        }

        let output_path = self.get_output_dir();
        if output_path.exists() && !self.dry_run {
            anyhow::bail!(
                "Output path already exists: {}. Please choose a different location or remove the existing directory.",
                output_path.display()
            );
        }

        if self.interactive && self.non_interactive {
            anyhow::bail!("Cannot use both --interactive and --non-interactive");
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_output_dir_default() {
        let cli = Cli {
            source: "test.zip".to_string(),
            output: None,
            headers: None,
            lib_name: None,
            model: "qwen2.5-coder:1.5b".to_string(),
            no_llm: false,
            interactive: false,
            non_interactive: false,
            style: "ergonomic".to_string(),
            cache_dir: None,
            dry_run: false,
            verbose: false,
        };

        assert_eq!(cli.get_output_dir(), PathBuf::from("./bindings-output"));
    }

    #[test]
    fn test_get_output_dir_custom() {
        let cli = Cli {
            source: "test.zip".to_string(),
            output: Some(PathBuf::from("/custom/path")),
            headers: None,
            lib_name: None,
            model: "qwen2.5-coder:1.5b".to_string(),
            no_llm: false,
            interactive: false,
            non_interactive: false,
            style: "ergonomic".to_string(),
            cache_dir: None,
            dry_run: false,
            verbose: false,
        };

        assert_eq!(cli.get_output_dir(), PathBuf::from("/custom/path"));
    }
}
