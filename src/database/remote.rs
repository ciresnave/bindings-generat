// Remote database fetching from GitHub
// Fetches TOML files from bindings-generat-db repository

use super::{LibraryEntry, PlatformInfo, Source};
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};
use tracing::{debug, info, warn};

/// GitHub repository information
const GITHUB_OWNER: &str = "ciresnave";
const GITHUB_REPO: &str = "bindings-generat-db";
const GITHUB_BRANCH: &str = "main";

/// Cache freshness duration (7 days)
const CACHE_FRESHNESS: Duration = Duration::from_secs(7 * 24 * 60 * 60);

/// Remote library metadata (from bindings-generat-db TOML files)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RemoteLibraryMetadata {
    pub name: String,
    pub display_name: String,
    pub description: String,
    pub homepage: String,
    pub license: String,
    pub redistribution_allowed: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub version_detection: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub notes: Option<String>,
    pub platforms: HashMap<String, RemotePlatformInfo>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dependencies: Option<RemoteDependencies>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RemotePlatformInfo {
    pub symbols: Vec<String>,
    pub filenames: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub header_files: Option<Vec<String>>,
    pub sources: Vec<RemoteInstallSource>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum RemoteInstallSource {
    DirectDownload {
        url: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        file_format: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        requires_account: Option<bool>,
        install_instructions: Vec<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        notes: Option<String>,
    },
    PackageManager {
        manager: String,
        package: String,
        command: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        notes: Option<String>,
    },
    SourceBuild {
        url: String,
        build_instructions: Vec<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        notes: Option<String>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RemoteDependencies {
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub runtime: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub build: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub libraries: Vec<String>,
}

/// Remote database fetcher
pub struct RemoteDatabaseFetcher {
    cache_dir: PathBuf,
    client: Option<reqwest::blocking::Client>,
}

impl RemoteDatabaseFetcher {
    /// Create a new fetcher with default cache location
    pub fn new() -> Result<Self> {
        let cache_dir = Self::get_cache_dir()?;
        fs::create_dir_all(&cache_dir).context("Failed to create cache directory")?;

        let client = Self::create_http_client();

        Ok(Self {
            cache_dir,
            client: Some(client),
        })
    }

    /// Create a new fetcher for offline use (no HTTP client)
    pub fn new_offline() -> Result<Self> {
        let cache_dir = Self::get_cache_dir()?;
        Ok(Self {
            cache_dir,
            client: None,
        })
    }

    /// Get the default cache directory
    fn get_cache_dir() -> Result<PathBuf> {
        let home_dir = dirs::home_dir().context("Failed to determine home directory")?;

        Ok(home_dir
            .join(".cache")
            .join("bindings-generat")
            .join("db"))
    }

    /// Create HTTP client with reasonable timeouts
    fn create_http_client() -> reqwest::blocking::Client {
        reqwest::blocking::Client::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .expect("Failed to create HTTP client")
    }

    /// Fetch library metadata by name
    /// First tries cache, then fetches from GitHub if needed
    pub fn get_library(&self, library_name: &str) -> Result<RemoteLibraryMetadata> {
        debug!("Looking up library: {}", library_name);

        // Try loading from cache first
        if let Ok(metadata) = self.load_from_cache(library_name) {
            if self.is_cache_fresh(library_name).unwrap_or(false) {
                debug!("Using fresh cached metadata for {}", library_name);
                return Ok(metadata);
            } else {
                debug!("Cache is stale for {}", library_name);
            }
        }

        // Cache miss or stale - fetch from GitHub
        if let Some(client) = &self.client {
            info!("Fetching {} metadata from GitHub", library_name);
            let metadata = self.fetch_from_github(library_name, client)?;
            self.save_to_cache(library_name, &metadata)?;
            Ok(metadata)
        } else {
            // Offline mode - try cache even if stale
            warn!(
                "Offline mode: using stale cache for {} if available",
                library_name
            );
            self.load_from_cache(library_name)
                .context("Library not in cache and offline mode enabled")
        }
    }

    /// Search for a library by symbols
    /// Returns all matching libraries from cache
    pub fn search_by_symbols(&self, symbols: &[String]) -> Result<Vec<RemoteLibraryMetadata>> {
        let mut matches = Vec::new();

        // Search in cache (avoids unnecessary GitHub API calls)
        if !self.cache_dir.exists() {
            return Ok(matches);
        }

        for entry in fs::read_dir(&self.cache_dir)? {
            let entry = entry?;
            let path = entry.path();

            if path.extension().and_then(|s| s.to_str()) == Some("toml")
                && let Ok(metadata) = self.load_toml_file(&path)
                    && self.matches_symbols(&metadata, symbols) {
                        matches.push(metadata);
                    }
        }

        Ok(matches)
    }

    /// Check if library metadata matches any of the given symbols
    fn matches_symbols(&self, metadata: &RemoteLibraryMetadata, symbols: &[String]) -> bool {
        for platform_info in metadata.platforms.values() {
            for symbol in symbols {
                if platform_info.symbols.iter().any(|s| s == symbol) {
                    return true;
                }
            }
        }
        false
    }

    /// Fetch library TOML from GitHub raw content API
    fn fetch_from_github(
        &self,
        library_name: &str,
        client: &reqwest::blocking::Client,
    ) -> Result<RemoteLibraryMetadata> {
        // Try each category directory
        let categories = [
            "audio",
            "compression",
            "crypto",
            "database",
            "graphics",
            "ml",
            "network",
            "system",
        ];

        let mut last_error = None;

        for category in &categories {
            let url = format!(
                "https://raw.githubusercontent.com/{}/{}/{}/libraries/{}/{}.toml",
                GITHUB_OWNER, GITHUB_REPO, GITHUB_BRANCH, category, library_name
            );

            debug!("Trying URL: {}", url);

            match client.get(&url).send() {
                Ok(response) => {
                    if response.status() == 404 {
                        // Not in this category, try next
                        debug!("Not found in category {}", category);
                        continue;
                    }

                    let toml_content = response
                        .text()
                        .context("Failed to read response body")?;
                    let metadata: RemoteLibraryMetadata = toml::from_str(&toml_content)
                        .context("Failed to parse TOML from GitHub")?;
                    info!(
                        "Successfully fetched {} from category {}",
                        library_name, category
                    );
                    return Ok(metadata);
                }
                Err(e) => {
                    warn!("Error fetching from {}: {}", category, e);
                    last_error = Some(e);
                }
            }
        }

        if let Some(e) = last_error {
            Err(e.into())
        } else {
            anyhow::bail!("Library '{}' not found in database", library_name)
        }
    }

    /// Load library metadata from cache
    fn load_from_cache(&self, library_name: &str) -> Result<RemoteLibraryMetadata> {
        let cache_path = self.cache_dir.join(format!("{}.toml", library_name));
        self.load_toml_file(&cache_path)
    }

    /// Load TOML file from filesystem
    fn load_toml_file(&self, path: &Path) -> Result<RemoteLibraryMetadata> {
        let content = fs::read_to_string(path)
            .with_context(|| format!("Failed to read file: {}", path.display()))?;

        toml::from_str(&content).context("Failed to parse TOML")
    }

    /// Save library metadata to cache
    fn save_to_cache(
        &self,
        library_name: &str,
        metadata: &RemoteLibraryMetadata,
    ) -> Result<()> {
        let cache_path = self.cache_dir.join(format!("{}.toml", library_name));
        let toml_content =
            toml::to_string_pretty(metadata).context("Failed to serialize metadata")?;

        fs::write(&cache_path, toml_content)
            .with_context(|| format!("Failed to write cache file: {}", cache_path.display()))?;

        debug!("Cached {} metadata to {:?}", library_name, cache_path);
        Ok(())
    }

    /// Check if cached library metadata is fresh
    fn is_cache_fresh(&self, library_name: &str) -> Result<bool> {
        let cache_path = self.cache_dir.join(format!("{}.toml", library_name));

        let metadata =
            fs::metadata(&cache_path).context("Failed to read cache file metadata")?;

        let modified = metadata
            .modified()
            .context("Failed to get cache file modification time")?;

        let age = SystemTime::now()
            .duration_since(modified)
            .unwrap_or(Duration::MAX);

        Ok(age < CACHE_FRESHNESS)
    }

    /// Clear the entire cache
    pub fn clear_cache(&self) -> Result<()> {
        if self.cache_dir.exists() {
            fs::remove_dir_all(&self.cache_dir).context("Failed to remove cache directory")?;
            fs::create_dir_all(&self.cache_dir).context("Failed to recreate cache directory")?;
            info!("Cleared library database cache");
        }
        Ok(())
    }

    /// Get cache directory path
    pub fn cache_path(&self) -> &Path {
        &self.cache_dir
    }
}

impl Default for RemoteDatabaseFetcher {
    fn default() -> Self {
        Self::new().expect("Failed to create default RemoteDatabaseFetcher")
    }
}

/// Convert remote metadata to local LibraryEntry format
impl From<RemoteLibraryMetadata> for LibraryEntry {
    fn from(remote: RemoteLibraryMetadata) -> Self {
        let platforms = remote
            .platforms
            .into_iter()
            .map(|(name, info)| {
                let platform_info = PlatformInfo {
                    symbols: info.symbols,
                    filenames: info.filenames,
                    sources: Some(
                        info.sources
                            .into_iter()
                            .map(Source::from)
                            .collect(),
                    ),
                    dependencies: None,
                    minimum_version: None,
                    install_path_hints: None,
                };
                (name, platform_info)
            })
            .collect();

        LibraryEntry {
            name: remote.name,
            display_name: remote.display_name,
            version: None,
            description: remote.description,
            homepage: Some(remote.homepage),
            documentation: None,
            license: Some(remote.license),
            redistributable: remote.redistribution_allowed,
            rust_wrappers: vec![],
            platforms,
        }
    }
}

/// Convert remote source to local Source format
impl From<RemoteInstallSource> for Source {
    fn from(remote: RemoteInstallSource) -> Self {
        match remote {
            RemoteInstallSource::DirectDownload {
                url,
                file_format,
                requires_account,
                install_instructions,
                notes,
            } => Source {
                source_type: super::SourceType::DirectDownload,
                url: Some(url),
                requires_account: requires_account.unwrap_or(false),
                requires_login: requires_account.unwrap_or(false),
                notes,
                file_format,
                manager: None,
                package: None,
                command: None,
                repository: None,
                install_instructions: Some(install_instructions),
                build_instructions: None,
                checksum: None,
            },
            RemoteInstallSource::PackageManager {
                manager,
                package,
                command,
                notes,
            } => Source {
                source_type: super::SourceType::PackageManager,
                url: None,
                requires_account: false,
                requires_login: false,
                notes,
                file_format: None,
                manager: Some(manager),
                package: Some(package),
                command: Some(command),
                repository: None,
                install_instructions: None,
                build_instructions: None,
                checksum: None,
            },
            RemoteInstallSource::SourceBuild {
                url,
                build_instructions,
                notes,
            } => Source {
                source_type: super::SourceType::SourceBuild,
                url: Some(url),
                requires_account: false,
                requires_login: false,
                notes,
                file_format: None,
                manager: None,
                package: None,
                command: None,
                repository: None,
                install_instructions: None,
                build_instructions: Some(build_instructions),
                checksum: None,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_dir_creation() {
        let fetcher = RemoteDatabaseFetcher::new().unwrap();
        assert!(fetcher.cache_dir.exists());
    }

    #[test]
    fn test_offline_mode() {
        let fetcher = RemoteDatabaseFetcher::new_offline().unwrap();
        assert!(fetcher.client.is_none());
    }

    #[test]
    fn test_github_url_format() {
        let category = "crypto";
        let library = "openssl";
        let url = format!(
            "https://raw.githubusercontent.com/{}/{}/{}/libraries/{}/{}.toml",
            GITHUB_OWNER, GITHUB_REPO, GITHUB_BRANCH, category, library
        );
        assert_eq!(
            url,
            "https://raw.githubusercontent.com/ciresnave/bindings-generat-db/main/libraries/crypto/openssl.toml"
        );
    }
}
