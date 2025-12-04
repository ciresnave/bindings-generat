//! Library database for automated discovery and installation
//!
//! This module provides access to a curated database of C/C++ libraries,
//! including metadata about installation sources, platform-specific files,
//! and detection symbols.

pub mod remote;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::{debug, info};

/// Complete library database
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LibraryDatabase {
    pub version: String,
    pub libraries: Vec<LibraryEntry>,
}

/// Entry for a single library
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LibraryEntry {
    pub name: String,
    pub display_name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub version: Option<String>,
    pub description: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub homepage: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub documentation: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub license: Option<String>,
    #[serde(default)]
    pub redistributable: bool,
    /// Known Rust crates that wrap this library
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub rust_wrappers: Vec<RustWrapper>,
    pub platforms: HashMap<String, PlatformInfo>,
}

/// Information about a known Rust wrapper crate
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RustWrapper {
    pub crate_name: String,
    pub version: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub repository: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub documentation: Option<String>,
}

/// Platform-specific library information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlatformInfo {
    pub symbols: Vec<String>,
    pub filenames: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sources: Option<Vec<Source>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dependencies: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub minimum_version: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub install_path_hints: Option<Vec<String>>,
}

/// Source for obtaining a library
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Source {
    #[serde(rename = "type")]
    pub source_type: SourceType,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
    #[serde(default)]
    pub requires_account: bool,
    #[serde(default)]
    pub requires_login: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub notes: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub file_format: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub manager: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub package: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub command: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub repository: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub install_instructions: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub build_instructions: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub checksum: Option<Checksum>,
}

/// Type of source
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SourceType {
    DirectDownload,
    Ftp,
    PackageManager,
    SourceBuild,
    GithubRelease,
    GitRepo,
    ArchiveOrg,
    VendorSpecific,
}

/// File checksum for verification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Checksum {
    pub algorithm: String,
    pub value: String,
}

impl LibraryDatabase {
    /// Load the database from embedded JSON
    pub fn load_embedded() -> Result<Self> {
        info!("Loading embedded library database");
        let json = include_str!("../../library-database.json");
        let db: LibraryDatabase =
            serde_json::from_str(json).context("Failed to parse embedded library database")?;
        debug!(
            "Loaded library database v{} with {} libraries",
            db.version,
            db.libraries.len()
        );
        Ok(db)
    }

    /// Find a library by name (case-insensitive)
    pub fn find_by_name(&self, name: &str) -> Option<&LibraryEntry> {
        let name_lower = name.to_lowercase();
        self.libraries.iter().find(|lib| {
            lib.name.to_lowercase() == name_lower || lib.display_name.to_lowercase() == name_lower
        })
    }

    /// Find libraries by symbol (useful for linker error matching)
    pub fn find_by_symbol(&self, symbol: &str, platform: &str) -> Vec<&LibraryEntry> {
        self.libraries
            .iter()
            .filter(|lib| {
                lib.platforms
                    .get(platform)
                    .map(|p| p.symbols.iter().any(|s| s == symbol))
                    .unwrap_or(false)
            })
            .collect()
    }

    /// Find libraries by filename pattern
    pub fn find_by_filename(&self, filename: &str, platform: &str) -> Vec<&LibraryEntry> {
        let filename_lower = filename.to_lowercase();
        self.libraries
            .iter()
            .filter(|lib| {
                lib.platforms
                    .get(platform)
                    .map(|p| {
                        p.filenames.iter().any(|f| {
                            f.to_lowercase().contains(&filename_lower)
                                || filename_lower.contains(&f.to_lowercase())
                        })
                    })
                    .unwrap_or(false)
            })
            .collect()
    }

    /// Get installation instructions for a library on current platform
    pub fn get_install_instructions(&self, name: &str) -> Option<Vec<String>> {
        let lib = self.find_by_name(name)?;
        let platform = get_current_platform();
        let platform_info = lib.platforms.get(&platform)?;

        // Collect instructions from all sources
        let mut instructions = Vec::new();

        instructions.push(format!("Installing {}", lib.display_name));
        instructions.push(String::new());

        if let Some(sources) = &platform_info.sources {
            for (i, source) in sources.iter().enumerate() {
                instructions.push(format!(
                    "Option {}: {}",
                    i + 1,
                    source_type_display(&source.source_type)
                ));

                if let Some(notes) = &source.notes {
                    instructions.push(format!("  Note: {}", notes));
                }

                if let Some(cmd) = &source.command {
                    instructions.push(format!("  Command: {}", cmd));
                }

                if let Some(install) = &source.install_instructions {
                    instructions.push("  Instructions:".to_string());
                    for step in install {
                        instructions.push(format!("    - {}", step));
                    }
                }

                instructions.push(String::new());
            }
        }

        Some(instructions)
    }

    /// Check if a library is available for the current platform
    pub fn is_available_on_platform(&self, name: &str) -> bool {
        self.find_by_name(name)
            .and_then(|lib| lib.platforms.get(&get_current_platform()))
            .is_some()
    }

    /// Try to discover an unknown library using Google Search (if configured).
    ///
    /// This is called when a library is not found in the database.
    /// If Google Search is configured and enabled, it will search for the library
    /// and prompt the user to add it to their local database and optionally
    /// submit it to the community.
    pub fn try_discover_unknown_library(
        &self,
        library_name: &str,
        config: &crate::user_config::Config,
    ) -> Result<Option<LibraryEntry>> {
        use dialoguer::{Confirm, theme::ColorfulTheme};

        println!("\n⚠ Unknown library: {}", library_name);

        // Step 1: Check remote database
        println!("Checking community database...");
        if let Ok(fetcher) = remote::RemoteDatabaseFetcher::new() {
            match fetcher.get_library(library_name) {
                Ok(remote_lib) => {
                    println!("✓ Found in community database!");
                    println!("  Name: {}", remote_lib.display_name);
                    println!("  Description: {}", remote_lib.description);
                    println!("  Homepage: {}", remote_lib.homepage);
                    println!("  License: {}", remote_lib.license);

                    // Convert remote metadata to LibraryEntry
                    let library_entry: LibraryEntry = remote_lib.into();

                    // Ask if they want to add it locally for faster access
                    let add_locally = Confirm::with_theme(&ColorfulTheme::default())
                        .with_prompt("Add this library to your local database for faster access?")
                        .default(true)
                        .interact()?;

                    if add_locally {
                        println!("✓ Added to local database");
                        return Ok(Some(library_entry));
                    } else {
                        // Still return the entry so it can be used, just not saved locally
                        return Ok(Some(library_entry));
                    }
                }
                Err(e) => {
                    debug!(
                        "Library '{}' not found in remote database: {}",
                        library_name, e
                    );
                    println!("Library not found in community database.");
                }
            }
        } else {
            debug!("Could not initialize remote database fetcher");
        }

        // Step 2: Search crates.io for existing Rust wrappers
        println!("Searching crates.io for existing Rust wrappers...");
        let crates = crate::discovery::search_crates_io(library_name)?;

        if !crates.is_empty() {
            // Let user choose if they want to use an existing crate
            let used_existing =
                crate::interactive::handle_existing_crates_workflow(library_name, &crates)?;

            if used_existing {
                // User selected an existing crate, no need to generate bindings
                return Ok(None);
            }
        } else {
            println!("No existing Rust wrappers found on crates.io.");
        }

        // Step 3: If no existing crates or user wants to generate anyway, search for library info
        if !config.google_search.is_configured() {
            println!("\n💡 Configure Google Custom Search API for automatic library discovery.");
            return Ok(None);
        }

        println!("\nSearching for library information...");

        // Use enhanced search to get documentation, examples, etc.
        let mut lib_info = crate::discovery::search_library_enhanced(
            library_name,
            config.google_search.api_key.as_ref().unwrap(),
            config.google_search.search_engine_id.as_ref().unwrap(),
        )?;

        // Add the crates we found to the library info
        lib_info.rust_crates = crates;

        println!("\n✓ Found potential match:");
        println!("  Name: {}", lib_info.name);
        println!("  Homepage: {}", lib_info.homepage);
        println!("  Description: {}", lib_info.description);
        if let Some(repo) = &lib_info.github_repo {
            println!("  GitHub: {}", repo);
        }
        if !lib_info.documentation_urls.is_empty() {
            println!(
                "  Documentation: {} link(s)",
                lib_info.documentation_urls.len()
            );
        }
        if !lib_info.example_urls.is_empty() {
            println!("  Examples: {} link(s)", lib_info.example_urls.len());
        }

        // Ask user if they want to add it locally
        let add_locally = Confirm::with_theme(&ColorfulTheme::default())
            .with_prompt("Add this library to your local database?")
            .default(true)
            .interact()?;

        if !add_locally {
            return Ok(None);
        }

        // Convert found Rust crates to RustWrapper format
        let rust_wrappers = lib_info
            .rust_crates
            .iter()
            .map(|c| RustWrapper {
                crate_name: c.name.clone(),
                version: c.latest_version.clone(),
                description: c.description.clone(),
                repository: c.repository.clone(),
                documentation: c.documentation.clone(),
            })
            .collect();

        // Generate a basic library entry
        let library_entry = LibraryEntry {
            name: library_name.to_lowercase(),
            display_name: lib_info.name.clone(),
            version: None,
            description: lib_info.description.clone(),
            homepage: Some(lib_info.homepage.clone()),
            documentation: lib_info.documentation_urls.first().cloned(),
            license: None,
            redistributable: false,
            rust_wrappers,
            platforms: HashMap::new(), // Would need to be populated
        };

        // If community contributions are enabled, offer to submit
        if config.community.contribute_discoveries {
            println!();
            let submit_to_community = Confirm::with_theme(&ColorfulTheme::default())
                .with_prompt("Submit this discovery to help the community?")
                .default(true)
                .interact()?;

            if submit_to_community {
                println!("\nCreating pull request...");

                // Generate TOML for the library
                let library_toml = generate_library_toml(&library_entry)?;

                // Submit using configured method
                match crate::submission::submit_library(
                    &library_entry.name,
                    &library_toml,
                    &config.community,
                ) {
                    Ok(result) => {
                        println!("\n{}", result.message());
                    }
                    Err(e) => {
                        println!("\n⚠ Failed to submit: {}", e);
                        println!("You can submit manually later if desired.");
                    }
                }
            }
        }

        Ok(Some(library_entry))
    }
}

/// Generate TOML content for a library entry.
fn generate_library_toml(entry: &LibraryEntry) -> Result<String> {
    // Simple TOML generation (could be enhanced with proper serialization)
    let mut toml = String::new();
    toml.push_str("[library]\n");
    toml.push_str(&format!("name = \"{}\"\n", entry.name));
    toml.push_str(&format!("display_name = \"{}\"\n", entry.display_name));
    toml.push_str(&format!("description = \"{}\"\n", entry.description));

    if let Some(homepage) = &entry.homepage {
        toml.push_str(&format!("homepage = \"{}\"\n", homepage));
    }

    toml.push_str(&format!("redistributable = {}\n", entry.redistributable));

    // Add rust wrappers if any
    if !entry.rust_wrappers.is_empty() {
        toml.push_str("\n[[rust_wrappers]]\n");
        for wrapper in &entry.rust_wrappers {
            toml.push_str(&format!("crate_name = \"{}\"\n", wrapper.crate_name));
            toml.push_str(&format!("version = \"{}\"\n", wrapper.version));
            if let Some(desc) = &wrapper.description {
                toml.push_str(&format!("description = \"{}\"\n", desc));
            }
            if let Some(repo) = &wrapper.repository {
                toml.push_str(&format!("repository = \"{}\"\n", repo));
            }
            if let Some(docs) = &wrapper.documentation {
                toml.push_str(&format!("documentation = \"{}\"\n", docs));
            }
            toml.push('\n');
        }
    }

    toml.push_str("\n# TODO: Add platform-specific information\n");
    toml.push_str("# [platforms.windows]\n");
    toml.push_str("# [platforms.linux]\n");
    toml.push_str("# [platforms.macos]\n");

    Ok(toml)
}

/// Get the current platform identifier
fn get_current_platform() -> String {
    #[cfg(target_os = "windows")]
    return "windows".to_string();

    #[cfg(target_os = "linux")]
    return "linux".to_string();

    #[cfg(target_os = "macos")]
    return "macos".to_string();

    #[cfg(not(any(target_os = "windows", target_os = "linux", target_os = "macos")))]
    return "unknown".to_string();
}

/// Convert source type to display string
fn source_type_display(source_type: &SourceType) -> &'static str {
    match source_type {
        SourceType::DirectDownload => "Direct Download",
        SourceType::Ftp => "FTP",
        SourceType::PackageManager => "Package Manager",
        SourceType::SourceBuild => "Build from Source",
        SourceType::GithubRelease => "GitHub Release",
        SourceType::GitRepo => "Git Repository",
        SourceType::ArchiveOrg => "Archive.org",
        SourceType::VendorSpecific => "Vendor-Specific",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_embedded() {
        let db = LibraryDatabase::load_embedded().expect("Failed to load database");
        assert!(!db.libraries.is_empty());
        assert_eq!(db.version, "1.0.0");
    }

    #[test]
    fn test_find_by_name() {
        let db = LibraryDatabase::load_embedded().unwrap();

        // Test exact match
        assert!(db.find_by_name("openssl").is_some());

        // Test case-insensitive
        assert!(db.find_by_name("OpenSSL").is_some());

        // Test display name
        assert!(db.find_by_name("NVIDIA cuDNN").is_some());

        // Test not found
        assert!(db.find_by_name("nonexistent").is_none());
    }

    #[test]
    fn test_find_by_symbol() {
        let db = LibraryDatabase::load_embedded().unwrap();
        let results = db.find_by_symbol("SSL_library_init", "windows");
        assert!(!results.is_empty());
        assert_eq!(results[0].name, "openssl");
    }

    #[test]
    fn test_find_by_filename() {
        let db = LibraryDatabase::load_embedded().unwrap();

        // Test with libpng which exists on all platforms
        let results = db.find_by_filename("libpng", "linux");
        assert!(!results.is_empty(), "Should find libpng on linux");
        assert_eq!(results[0].name, "libpng");
    }
    #[test]
    fn test_get_install_instructions() {
        let db = LibraryDatabase::load_embedded().unwrap();
        let instructions = db.get_install_instructions("openssl");
        assert!(instructions.is_some());
        let instructions = instructions.unwrap();
        assert!(!instructions.is_empty());
    }

    #[test]
    fn test_is_available_on_platform() {
        let db = LibraryDatabase::load_embedded().unwrap();
        assert!(db.is_available_on_platform("openssl"));
        assert!(db.is_available_on_platform("libpng"));
    }
}
