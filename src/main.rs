use anyhow::Result;
use bindings_generat::{BindingsGenerator, Cli, Config};
use clap::Parser;
use tracing_indicatif::IndicatifLayer;
use tracing_subscriber::{EnvFilter, fmt, prelude::*};

fn main() -> Result<()> {
    // Parse CLI arguments
    let cli = Cli::parse();

    // Set up logging with progress bar awareness
    // When verbose, enable debug for our crate but keep bindgen quiet
    let filter = if cli.verbose {
        EnvFilter::new("bindings_generat=debug,info")
    } else {
        EnvFilter::new("info")
    };

    // Create indicatif layer for progress bars
    let indicatif_layer = IndicatifLayer::new();

    tracing_subscriber::registry()
        .with(filter)
        .with(fmt::layer().with_target(false).without_time())
        .with(indicatif_layer)
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
    let mut generator = BindingsGenerator::new(config.clone());
    generator.run()?;

    // Run publishing wizard if requested
    if cli.publish {
        use bindings_generat::publishing::wizard;
        let output_dir = config.output_path;
        println!("\n"); // Visual separator
        wizard::run_wizard(output_dir)?;
    }

    Ok(())
}
