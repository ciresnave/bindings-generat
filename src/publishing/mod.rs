//! Publishing automation for generated bindings crates.
//!
//! This module provides a complete workflow for publishing generated bindings:
//! - License file generation
//! - GitHub repository creation
//! - CI/CD workflow setup
//! - cargo publish automation
//! - Interactive publishing wizard

pub mod wizard;

use anyhow::{anyhow, Context, Result};
use std::fs;
use std::path::PathBuf;
use std::process::Command;
use tracing::{info, warn};

/// Result of a publishing operation
#[derive(Debug, Clone)]
pub enum PublishResult {
    /// Successfully published to crates.io
    Published { crate_name: String, version: String },
    /// Created GitHub repository
    RepositoryCreated { url: String },
    /// Generated files but didn't publish
    DryRun { files_created: Vec<PathBuf> },
    /// User cancelled
    Cancelled,
}

/// Status check before publishing
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PublishStatus {
    /// Ready to publish
    Ready,
    /// Not logged into cargo
    NotLoggedIn,
    /// Uncommitted git changes
    UncommittedChanges,
    /// Tests failed
    TestsFailed,
    /// Missing required metadata
    MissingMetadata(Vec<String>),
    /// Git not available
    GitNotAvailable,
}

/// Publishing configuration
#[derive(Debug, Clone)]
pub struct PublishConfig {
    /// Crate directory
    pub crate_dir: PathBuf,
    /// Whether to create GitHub repository
    pub create_github_repo: bool,
    /// Whether to publish to crates.io
    pub publish_to_crates_io: bool,
    /// Whether to add CI/CD workflows
    pub add_ci_workflows: bool,
    /// GitHub username (for repo creation)
    pub github_username: Option<String>,
    /// License to use
    pub license: String,
    /// Dry run mode (don't actually publish)
    pub dry_run: bool,
}

impl Default for PublishConfig {
    fn default() -> Self {
        Self {
            crate_dir: PathBuf::from("."),
            create_github_repo: true,
            publish_to_crates_io: true,
            add_ci_workflows: true,
            github_username: None,
            license: "MIT OR Apache-2.0".to_string(),
            dry_run: false,
        }
    }
}

/// Publisher for managing the publishing workflow
pub struct Publisher {
    config: PublishConfig,
}

impl Publisher {
    /// Create a new publisher
    pub fn new(config: PublishConfig) -> Self {
        Self { config }
    }

    /// Check prerequisites for publishing
    pub fn check_prerequisites(&self) -> Result<PublishStatus> {
        // Check cargo login
        if self.config.publish_to_crates_io && !self.is_cargo_logged_in()? {
            return Ok(PublishStatus::NotLoggedIn);
        }

        // Check for uncommitted changes
        if self.has_uncommitted_changes()? {
            return Ok(PublishStatus::UncommittedChanges);
        }

        // Check required Cargo.toml fields
        if let Some(missing) = self.check_cargo_metadata()? {
            return Ok(PublishStatus::MissingMetadata(missing));
        }

        // Check if tests pass
        if !self.config.dry_run && !self.run_tests()? {
            return Ok(PublishStatus::TestsFailed);
        }

        Ok(PublishStatus::Ready)
    }

    /// Run the complete publishing workflow
    pub fn publish(&self) -> Result<Vec<PublishResult>> {
        let mut results = Vec::new();

        info!("Starting publishing workflow for {:?}", self.config.crate_dir);

        // 1. Generate license file
        if !self.license_file_exists()? {
            info!("Generating license file");
            self.generate_license_file()?;
        }

        // 2. Add CI/CD workflows
        if self.config.add_ci_workflows {
            info!("Setting up CI/CD workflows");
            self.setup_ci_workflows()?;
        }

        // 3. Create GitHub repository
        if self.config.create_github_repo {
            info!("Creating GitHub repository");
            let repo_url = self.create_github_repo()?;
            results.push(PublishResult::RepositoryCreated { url: repo_url });
        }

        // 4. Publish to crates.io
        if self.config.publish_to_crates_io {
            info!("Publishing to crates.io");
            let (name, version) = self.cargo_publish()?;
            results.push(PublishResult::Published {
                crate_name: name,
                version,
            });
        }

        Ok(results)
    }

    /// Check if user is logged into cargo
    fn is_cargo_logged_in(&self) -> Result<bool> {
        let _output = Command::new("cargo")
            .args(["login", "--help"])
            .output()
            .context("Failed to run cargo")?;

        // Check if token is configured
        let home_dir = dirs::home_dir().context("Failed to get home directory")?;
        let credentials_file = home_dir.join(".cargo").join("credentials.toml");

        Ok(credentials_file.exists())
    }

    /// Check for uncommitted git changes
    fn has_uncommitted_changes(&self) -> Result<bool> {
        if !self.is_git_repo()? {
            return Ok(false); // Not a git repo, no uncommitted changes
        }

        let output = Command::new("git")
            .args(["status", "--porcelain"])
            .current_dir(&self.config.crate_dir)
            .output()
            .context("Failed to run git status")?;

        Ok(!output.stdout.is_empty())
    }

    /// Check if directory is a git repository
    fn is_git_repo(&self) -> Result<bool> {
        Ok(self.config.crate_dir.join(".git").exists())
    }

    /// Check Cargo.toml for required metadata
    fn check_cargo_metadata(&self) -> Result<Option<Vec<String>>> {
        let cargo_toml_path = self.config.crate_dir.join("Cargo.toml");
        let content = fs::read_to_string(&cargo_toml_path)
            .context("Failed to read Cargo.toml")?;

        let manifest: toml::Value = toml::from_str(&content)
            .context("Failed to parse Cargo.toml")?;

        let mut missing = Vec::new();

        // Required fields for crates.io
        let required_fields = vec![
            ("package.name", "name"),
            ("package.version", "version"),
            ("package.description", "description"),
            ("package.license", "license"),
        ];

        for (path, field_name) in required_fields {
            if !self.has_toml_field(&manifest, path) {
                missing.push(field_name.to_string());
            }
        }

        if missing.is_empty() {
            Ok(None)
        } else {
            Ok(Some(missing))
        }
    }

    /// Check if TOML has a field at the given path
    fn has_toml_field(&self, value: &toml::Value, path: &str) -> bool {
        let parts: Vec<&str> = path.split('.').collect();
        let mut current = value;

        for part in parts {
            if let Some(table) = current.as_table() {
                if let Some(next) = table.get(part) {
                    current = next;
                } else {
                    return false;
                }
            } else {
                return false;
            }
        }

        true
    }

    /// Run tests
    fn run_tests(&self) -> Result<bool> {
        info!("Running tests...");

        let output = Command::new("cargo")
            .args(["test", "--all"])
            .current_dir(&self.config.crate_dir)
            .output()
            .context("Failed to run cargo test")?;

        Ok(output.status.success())
    }

    /// Check if license file exists
    fn license_file_exists(&self) -> Result<bool> {
        let license_files = vec!["LICENSE", "LICENSE.md", "LICENSE.txt", "LICENSE-MIT", "LICENSE-APACHE"];
        
        for file in license_files {
            if self.config.crate_dir.join(file).exists() {
                return Ok(true);
            }
        }

        Ok(false)
    }

    /// Generate license file
    fn generate_license_file(&self) -> Result<()> {
        let license_content = match self.config.license.as_str() {
            "MIT" => include_str!("../../LICENSE-MIT"),
            "Apache-2.0" => include_str!("../../LICENSE-APACHE"),
            "MIT OR Apache-2.0" | "MIT/Apache-2.0" => {
                // Dual license - create both files
                let mit_path = self.config.crate_dir.join("LICENSE-MIT");
                let apache_path = self.config.crate_dir.join("LICENSE-APACHE");

                fs::write(&mit_path, include_str!("../../LICENSE-MIT"))
                    .context("Failed to write LICENSE-MIT")?;
                fs::write(&apache_path, include_str!("../../LICENSE-APACHE"))
                    .context("Failed to write LICENSE-APACHE")?;

                info!("Created LICENSE-MIT and LICENSE-APACHE");
                return Ok(());
            }
            _ => {
                warn!("Unknown license type: {}, skipping license generation", self.config.license);
                return Ok(());
            }
        };

        let license_path = self.config.crate_dir.join("LICENSE");
        fs::write(&license_path, license_content)
            .context("Failed to write LICENSE file")?;

        info!("Created LICENSE file");
        Ok(())
    }

    /// Setup CI/CD workflows
    fn setup_ci_workflows(&self) -> Result<()> {
        let workflows_dir = self.config.crate_dir.join(".github").join("workflows");
        fs::create_dir_all(&workflows_dir)
            .context("Failed to create .github/workflows directory")?;

        // Create CI workflow
        let ci_workflow = self.generate_ci_workflow();
        let ci_path = workflows_dir.join("ci.yml");
        fs::write(&ci_path, ci_workflow)
            .context("Failed to write CI workflow")?;

        info!("Created .github/workflows/ci.yml");

        Ok(())
    }

    /// Generate CI workflow content
    fn generate_ci_workflow(&self) -> String {
        r#"name: CI

on:
  push:
    branches: [ main, master ]
  pull_request:
    branches: [ main, master ]

env:
  CARGO_TERM_COLOR: always

jobs:
  test:
    name: Test
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, windows-latest, macos-latest]
        rust: [stable, beta]
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Install Rust
      uses: dtolnay/rust-toolchain@master
      with:
        toolchain: ${{ matrix.rust }}
    
    - name: Cache cargo registry
      uses: actions/cache@v4
      with:
        path: ~/.cargo/registry
        key: ${{ runner.os }}-cargo-registry-${{ hashFiles('**/Cargo.lock') }}
    
    - name: Cache cargo index
      uses: actions/cache@v4
      with:
        path: ~/.cargo/git
        key: ${{ runner.os }}-cargo-git-${{ hashFiles('**/Cargo.lock') }}
    
    - name: Cache cargo build
      uses: actions/cache@v4
      with:
        path: target
        key: ${{ runner.os }}-cargo-build-target-${{ hashFiles('**/Cargo.lock') }}
    
    - name: Check formatting
      run: cargo fmt --all -- --check
      if: matrix.rust == 'stable'
    
    - name: Run clippy
      run: cargo clippy --all-targets --all-features -- -D warnings
      if: matrix.rust == 'stable'
    
    - name: Build
      run: cargo build --verbose
    
    - name: Run tests
      run: cargo test --verbose

  build-release:
    name: Build Release
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Install Rust
      uses: dtolnay/rust-toolchain@stable
    
    - name: Build release
      run: cargo build --release --verbose
    
    - name: Check package
      run: cargo package --verbose
"#.to_string()
    }

    /// Create GitHub repository
    fn create_github_repo(&self) -> Result<String> {
        // Get crate name from Cargo.toml
        let cargo_toml_path = self.config.crate_dir.join("Cargo.toml");
        let content = fs::read_to_string(&cargo_toml_path)?;
        let manifest: toml::Value = toml::from_str(&content)?;

        let crate_name = manifest
            .get("package")
            .and_then(|p| p.get("name"))
            .and_then(|n| n.as_str())
            .context("Failed to get crate name from Cargo.toml")?;

        let description = manifest
            .get("package")
            .and_then(|p| p.get("description"))
            .and_then(|d| d.as_str())
            .unwrap_or("Rust FFI bindings");

        if self.config.dry_run {
            info!("DRY RUN: Would create GitHub repo: {}", crate_name);
            return Ok(format!("https://github.com/{}/{}", 
                self.config.github_username.as_deref().unwrap_or("user"), 
                crate_name));
        }

        // Check if gh CLI is available
        let gh_check = Command::new("gh")
            .args(["--version"])
            .output();

        if gh_check.is_err() {
            warn!("GitHub CLI (gh) not found. Skipping repository creation.");
            warn!("Install gh from: https://cli.github.com/");
            return Err(anyhow!("gh CLI not available"));
        }

        // Create repository using gh CLI
        let output = Command::new("gh")
            .args([
                "repo",
                "create",
                crate_name,
                "--public",
                "--description",
                description,
                "--source",
                ".",
                "--push",
            ])
            .current_dir(&self.config.crate_dir)
            .output()
            .context("Failed to create GitHub repository")?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(anyhow!("Failed to create GitHub repo: {}", stderr));
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        let repo_url = stdout
            .lines()
            .find(|line| line.starts_with("https://github.com/"))
            .map(|s| s.trim().to_string())
            .unwrap_or_else(|| format!("https://github.com/{}/{}", 
                self.config.github_username.as_deref().unwrap_or("user"), 
                crate_name));

        info!("Created GitHub repository: {}", repo_url);
        Ok(repo_url)
    }

    /// Publish to crates.io
    fn cargo_publish(&self) -> Result<(String, String)> {
        // Get crate info
        let cargo_toml_path = self.config.crate_dir.join("Cargo.toml");
        let content = fs::read_to_string(&cargo_toml_path)?;
        let manifest: toml::Value = toml::from_str(&content)?;

        let name = manifest
            .get("package")
            .and_then(|p| p.get("name"))
            .and_then(|n| n.as_str())
            .context("Failed to get crate name")?
            .to_string();

        let version = manifest
            .get("package")
            .and_then(|p| p.get("version"))
            .and_then(|v| v.as_str())
            .context("Failed to get crate version")?
            .to_string();

        if self.config.dry_run {
            info!("DRY RUN: Would publish {} v{} to crates.io", name, version);
            return Ok((name, version));
        }

        info!("Publishing {} v{} to crates.io...", name, version);

        let output = Command::new("cargo")
            .args(["publish"])
            .current_dir(&self.config.crate_dir)
            .output()
            .context("Failed to run cargo publish")?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(anyhow!("cargo publish failed: {}", stderr));
        }

        info!("Successfully published {} v{}", name, version);
        Ok((name, version))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_publish_config_default() {
        let config = PublishConfig::default();
        assert_eq!(config.license, "MIT OR Apache-2.0");
        assert!(config.create_github_repo);
        assert!(config.publish_to_crates_io);
    }

    #[test]
    fn test_publish_status() {
        assert_eq!(PublishStatus::Ready, PublishStatus::Ready);
        assert_ne!(PublishStatus::Ready, PublishStatus::NotLoggedIn);
    }
}
