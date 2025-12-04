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

    /// Additional include directories for dependencies
    #[arg(long = "include", value_name = "PATH", action = clap::ArgAction::Append)]
    pub include_dirs: Vec<PathBuf>,

    /// Link libraries for dependencies (e.g. 'cuda', 'cudnn')  
    #[arg(long = "link", value_name = "LIB", action = clap::ArgAction::Append)]
    pub link_libs: Vec<String>,

    /// Link library search paths
    #[arg(long = "lib-path", value_name = "PATH", action = clap::ArgAction::Append)]
    pub lib_paths: Vec<PathBuf>,

    /// LLM model for Ollama (default: llama3.2:1b)
    #[arg(long, value_name = "MODEL", default_value = "llama3.2:1b")]
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

    /// Run publishing wizard after generation
    #[arg(long)]
    pub publish: bool,

    /// Verbose output
    #[arg(short = 'v', long)]
    pub verbose: bool,
    /// Bundle the library into the generated crate (opt-in; obeys license)
    #[arg(long = "bundle-library")]
    pub bundle_library: bool,
    /// Enable build-time library discovery (deprecated; opt-in)
    #[arg(long = "enable-build-discovery")]
    pub enable_build_discovery: bool,
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
            include_dirs: Vec::new(),
            link_libs: Vec::new(),
            lib_paths: Vec::new(),
            model: "qwen2.5-coder:1.5b".to_string(),
            no_llm: false,
            interactive: false,
            non_interactive: false,
            style: "ergonomic".to_string(),
            cache_dir: None,
            dry_run: false,
            publish: false,
            verbose: false,
            bundle_library: false,
            enable_build_discovery: false,
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
            include_dirs: Vec::new(),
            link_libs: Vec::new(),
            lib_paths: Vec::new(),
            model: "qwen2.5-coder:1.5b".to_string(),
            no_llm: false,
            interactive: false,
            non_interactive: false,
            style: "ergonomic".to_string(),
            cache_dir: None,
            dry_run: false,
            publish: false,
            verbose: false,
            bundle_library: false,
            enable_build_discovery: false,
        };

        assert_eq!(cli.get_output_dir(), PathBuf::from("/custom/path"));
    }

    #[test]
    fn test_default_values() {
        let cli = Cli {
            source: "test.h".to_string(),
            output: None,
            headers: None,
            lib_name: None,
            include_dirs: Vec::new(),
            link_libs: Vec::new(),
            lib_paths: Vec::new(),
            model: "qwen2.5-coder:1.5b".to_string(),
            no_llm: false,
            interactive: false,
            non_interactive: false,
            style: "ergonomic".to_string(),
            cache_dir: None,
            dry_run: false,
            publish: false,
            verbose: false,
            bundle_library: false,
            enable_build_discovery: false,
        };

        // Verify default values
        assert!(!cli.no_llm);
        assert!(!cli.interactive);
        assert!(!cli.non_interactive);
        assert_eq!(cli.style, "ergonomic");
        assert!(!cli.dry_run);
        assert!(!cli.publish);
        assert!(!cli.verbose);
    }

    #[test]
    fn test_llm_flag() {
        let cli = Cli {
            source: "test.h".to_string(),
            output: None,
            headers: None,
            lib_name: None,
            include_dirs: Vec::new(),
            link_libs: Vec::new(),
            lib_paths: Vec::new(),
            model: "qwen2.5-coder:1.5b".to_string(),
            no_llm: true,
            interactive: false,
            non_interactive: false,
            style: "ergonomic".to_string(),
            cache_dir: None,
            dry_run: false,
            publish: false,
            verbose: false,
            bundle_library: false,
            enable_build_discovery: false,
        };

        assert!(cli.no_llm);
    }

    #[test]
    fn test_interactive_modes() {
        // Interactive mode
        let interactive = Cli {
            source: "test.h".to_string(),
            output: None,
            headers: None,
            lib_name: None,
            include_dirs: Vec::new(),
            link_libs: Vec::new(),
            lib_paths: Vec::new(),
            model: "qwen2.5-coder:1.5b".to_string(),
            no_llm: false,
            interactive: true,
            non_interactive: false,
            style: "ergonomic".to_string(),
            cache_dir: None,
            dry_run: false,
            publish: false,
            verbose: false,
            bundle_library: false,
            enable_build_discovery: false,
        };

        assert!(interactive.interactive);
        assert!(!interactive.non_interactive);

        // Non-interactive mode
        let non_interactive = Cli {
            source: "test.h".to_string(),
            output: None,
            headers: None,
            lib_name: None,
            include_dirs: Vec::new(),
            link_libs: Vec::new(),
            lib_paths: Vec::new(),
            model: "qwen2.5-coder:1.5b".to_string(),
            no_llm: false,
            interactive: false,
            non_interactive: true,
            style: "ergonomic".to_string(),
            cache_dir: None,
            dry_run: false,
            publish: false,
            verbose: false,
            bundle_library: false,
            enable_build_discovery: false,
        };

        assert!(!non_interactive.interactive);
        assert!(non_interactive.non_interactive);
    }

    #[test]
    fn test_style_options() {
        let styles = vec!["ergonomic", "unsafe", "minimal"];

        for style in styles {
            let cli = Cli {
                source: "test.h".to_string(),
                output: None,
                headers: None,
                lib_name: None,
                include_dirs: Vec::new(),
                link_libs: Vec::new(),
                lib_paths: Vec::new(),
                model: "qwen2.5-coder:1.5b".to_string(),
                no_llm: false,
                interactive: false,
                non_interactive: false,
                style: style.to_string(),
                cache_dir: None,
                dry_run: false,
                publish: false,
                verbose: false,
                bundle_library: false,
                enable_build_discovery: false,
            };

            assert_eq!(cli.style, style);
        }
    }

    #[test]
    fn test_verbose_flag() {
        let cli = Cli {
            source: "test.h".to_string(),
            output: None,
            headers: None,
            lib_name: None,
            include_dirs: Vec::new(),
            link_libs: Vec::new(),
            lib_paths: Vec::new(),
            model: "qwen2.5-coder:1.5b".to_string(),
            no_llm: false,
            interactive: false,
            non_interactive: false,
            style: "ergonomic".to_string(),
            cache_dir: None,
            dry_run: false,
            publish: false,
            verbose: true,
            bundle_library: false,
            enable_build_discovery: false,
        };

        assert!(cli.verbose);
    }

    #[test]
    fn test_dry_run_flag() {
        let cli = Cli {
            source: "test.h".to_string(),
            output: None,
            headers: None,
            lib_name: None,
            include_dirs: Vec::new(),
            link_libs: Vec::new(),
            lib_paths: Vec::new(),
            model: "qwen2.5-coder:1.5b".to_string(),
            no_llm: false,
            interactive: false,
            non_interactive: false,
            style: "ergonomic".to_string(),
            cache_dir: None,
            dry_run: true,
            publish: false,
            verbose: false,
            bundle_library: false,
            enable_build_discovery: false,
        };

        assert!(cli.dry_run);
    }

    #[test]
    fn test_multiple_include_dirs() {
        let cli = Cli {
            source: "test.h".to_string(),
            output: None,
            headers: None,
            lib_name: None,
            include_dirs: vec![
                PathBuf::from("/usr/include"),
                PathBuf::from("/usr/local/include"),
                PathBuf::from("./include"),
            ],
            link_libs: Vec::new(),
            lib_paths: Vec::new(),
            model: "qwen2.5-coder:1.5b".to_string(),
            no_llm: false,
            interactive: false,
            non_interactive: false,
            style: "ergonomic".to_string(),
            cache_dir: None,
            dry_run: false,
            publish: false,
            verbose: false,
            bundle_library: false,
            enable_build_discovery: false,
        };

        assert_eq!(cli.include_dirs.len(), 3);
        assert_eq!(cli.include_dirs[0], PathBuf::from("/usr/include"));
        assert_eq!(cli.include_dirs[1], PathBuf::from("/usr/local/include"));
        assert_eq!(cli.include_dirs[2], PathBuf::from("./include"));
    }

    #[test]
    fn test_link_libraries() {
        let cli = Cli {
            source: "test.h".to_string(),
            output: None,
            headers: None,
            lib_name: None,
            include_dirs: Vec::new(),
            link_libs: vec!["cuda".to_string(), "cudnn".to_string()],
            lib_paths: vec![PathBuf::from("/usr/lib"), PathBuf::from("/usr/local/lib")],
            model: "qwen2.5-coder:1.5b".to_string(),
            no_llm: false,
            interactive: false,
            non_interactive: false,
            style: "ergonomic".to_string(),
            cache_dir: None,
            dry_run: false,
            publish: false,
            verbose: false,
            bundle_library: false,
            enable_build_discovery: false,
        };

        assert_eq!(cli.link_libs.len(), 2);
        assert_eq!(cli.link_libs[0], "cuda");
        assert_eq!(cli.link_libs[1], "cudnn");
        assert_eq!(cli.lib_paths.len(), 2);
    }

    #[test]
    fn test_custom_cache_dir() {
        let custom_cache = PathBuf::from("/tmp/custom-cache");
        let cli = Cli {
            source: "test.h".to_string(),
            output: None,
            headers: None,
            lib_name: None,
            include_dirs: Vec::new(),
            link_libs: Vec::new(),
            lib_paths: Vec::new(),
            model: "qwen2.5-coder:1.5b".to_string(),
            no_llm: false,
            interactive: false,
            non_interactive: false,
            style: "ergonomic".to_string(),
            cache_dir: Some(custom_cache.clone()),
            dry_run: false,
            publish: false,
            verbose: false,
            bundle_library: false,
            enable_build_discovery: false,
        };

        assert_eq!(cli.cache_dir, Some(custom_cache));
    }

    #[test]
    fn test_lib_name_override() {
        let cli = Cli {
            source: "test.h".to_string(),
            output: None,
            headers: None,
            lib_name: Some("my_custom_lib".to_string()),
            include_dirs: Vec::new(),
            link_libs: Vec::new(),
            lib_paths: Vec::new(),
            model: "qwen2.5-coder:1.5b".to_string(),
            no_llm: false,
            interactive: false,
            non_interactive: false,
            style: "ergonomic".to_string(),
            cache_dir: None,
            dry_run: false,
            publish: false,
            verbose: false,
            bundle_library: false,
            enable_build_discovery: false,
        };

        assert_eq!(cli.lib_name, Some("my_custom_lib".to_string()));
    }
}
