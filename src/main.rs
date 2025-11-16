use anyhow::Result;
use bindings_generat::{BindingsGenerator, Cli, Config};
use clap::Parser;
use tracing_subscriber::{EnvFilter, fmt};

fn main() -> Result<()> {
    // Parse CLI arguments
    let cli = Cli::parse();

    // Set up logging
    let filter = if cli.verbose {
        EnvFilter::new("debug")
    } else {
        EnvFilter::new("info")
    };

    fmt()
        .with_env_filter(filter)
        .with_target(false)
        .without_time()
        .init();

    // Validate arguments
    cli.validate()?;

    // For config file, only check if source is a local directory
    let config_file = if !cli.source.starts_with("http://") && !cli.source.starts_with("https://") {
        let source_path = std::path::PathBuf::from(&cli.source);
        if source_path.is_dir() {
            let cfg_file = source_path.join("bindings-generat.toml");
            if cfg_file.exists() {
                Some(cfg_file)
            } else {
                None
            }
        } else {
            None
        }
    } else {
        None
    };

    let config = if let Some(config_file) = config_file {
        Config::from_file(&config_file)?
    } else {
        Config::from_cli(&cli)?
    };

    // Run the generator
    let mut generator = BindingsGenerator::new(config);
    generator.run()?;

    Ok(())
}
