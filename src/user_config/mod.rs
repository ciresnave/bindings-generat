//! Configuration management for bindings-generat.
//!
//! Handles loading and saving user configuration from platform-appropriate directories:
//! - Linux/macOS: `~/.config/bindings-generat/config.toml`
//! - Windows: `%APPDATA%\bindings-generat\config.toml`

use anyhow::{Context, Result};
use std::path::PathBuf;

mod schema;

pub use schema::{Attribution, CommunityConfig, Config, GoogleSearchConfig, SubmissionMethod};

/// Get the path to the configuration file.
///
/// Returns the platform-appropriate path:
/// - Linux/macOS: `~/.config/bindings-generat/config.toml`
/// - Windows: `%APPDATA%\bindings-generat\config.toml`
pub fn config_path() -> Result<PathBuf> {
    let config_dir = dirs::config_dir().context("Failed to determine config directory")?;
    Ok(config_dir.join("bindings-generat").join("config.toml"))
}

/// Load configuration from disk.
///
/// If the configuration file doesn't exist, returns a default configuration.
pub fn load() -> Result<Config> {
    let path = config_path()?;

    if !path.exists() {
        return Ok(Config::default());
    }

    let contents = std::fs::read_to_string(&path)
        .with_context(|| format!("Failed to read config file: {}", path.display()))?;

    let config: Config = toml::from_str(&contents)
        .with_context(|| format!("Failed to parse config file: {}", path.display()))?;

    Ok(config)
}

/// Save configuration to disk.
///
/// Creates the configuration directory if it doesn't exist.
pub fn save(config: &Config) -> Result<()> {
    let path = config_path()?;

    // Create parent directory if it doesn't exist
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("Failed to create config directory: {}", parent.display()))?;
    }

    let contents = toml::to_string_pretty(config).context("Failed to serialize config")?;

    std::fs::write(&path, contents)
        .with_context(|| format!("Failed to write config file: {}", path.display()))?;

    Ok(())
}

/// Check if configuration file exists.
pub fn exists() -> bool {
    config_path().ok().is_some_and(|path| path.exists())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_config_path() {
        let path = config_path().unwrap();
        assert!(
            path.ends_with("bindings-generat/config.toml")
                || path.ends_with("bindings-generat\\config.toml")
        );
    }

    #[test]
    fn test_load_nonexistent() {
        // Loading non-existent config should return default
        let config = load().unwrap();
        assert!(!config.google_search.is_configured());
        assert!(!config.community.contribute_discoveries);
    }

    #[test]
    fn test_save_and_load() {
        // Create a temporary directory for testing
        let temp_dir = TempDir::new().unwrap();
        let config_file = temp_dir.path().join("config.toml");

        // Create test config
        let original = Config {
            google_search: GoogleSearchConfig {
                api_key: Some("test_key".to_string()),
                search_engine_id: Some("test_id".to_string()),
            },
            community: CommunityConfig {
                contribute_discoveries: true,
                submission_method: SubmissionMethod::Manual,
                github_token: Some("test_token".to_string()),
                attribution: Attribution {
                    name: Some("Test User".to_string()),
                    email: Some("test@example.com".to_string()),
                },
            },
        };

        // Serialize and save
        let contents = toml::to_string_pretty(&original).unwrap();
        std::fs::write(&config_file, contents).unwrap();

        // Load and verify
        let loaded_contents = std::fs::read_to_string(&config_file).unwrap();
        let loaded: Config = toml::from_str(&loaded_contents).unwrap();

        assert_eq!(original.google_search.api_key, loaded.google_search.api_key);
        assert_eq!(
            original.google_search.search_engine_id,
            loaded.google_search.search_engine_id
        );
        assert_eq!(
            original.community.contribute_discoveries,
            loaded.community.contribute_discoveries
        );
        assert_eq!(
            original.community.submission_method,
            loaded.community.submission_method
        );
        assert_eq!(
            original.community.github_token,
            loaded.community.github_token
        );
    }

    #[test]
    fn test_default_config_serialization() {
        let config = Config::default();
        let serialized = toml::to_string_pretty(&config).unwrap();

        // Verify it can be deserialized back
        let deserialized: Config = toml::from_str(&serialized).unwrap();
        assert_eq!(
            config.community.submission_method,
            deserialized.community.submission_method
        );
    }
}
