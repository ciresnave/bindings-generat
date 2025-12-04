//! Interactive publishing wizard

use super::{PublishConfig, PublishResult, PublishStatus, Publisher};
use anyhow::{Context, Result};
use std::io::{self, Write};
use std::path::PathBuf;

/// Run the interactive publishing wizard
pub fn run_wizard(crate_dir: PathBuf) -> Result<Vec<PublishResult>> {
    println!("\n🚀 Publishing Wizard");
    println!("==================\n");

    // Build configuration through prompts
    let mut config = PublishConfig {
        crate_dir: crate_dir.clone(),
        ..Default::default()
    };

    // Get crate info
    let crate_info = get_crate_info(&config.crate_dir)?;
    println!("📦 Crate: {} v{}", crate_info.name, crate_info.version);
    println!("   {}\n", crate_info.description);

    // License selection
    config.license = prompt_license()?;

    // GitHub repo creation
    config.create_github_repo = prompt_yes_no("Create GitHub repository?", true)?;
    if config.create_github_repo {
        config.github_username = Some(get_github_username()?);
    }

    // CI/CD workflows
    config.add_ci_workflows = prompt_yes_no("Add CI/CD workflows?", true)?;

    // Publish to crates.io
    config.publish_to_crates_io = prompt_yes_no("Publish to crates.io?", true)?;

    // Dry run option
    config.dry_run = prompt_yes_no("Dry run (don't actually publish)?", false)?;

    println!("\n📋 Publishing Plan");
    println!("==================");
    print_publishing_plan(&config);

    // Confirm
    if !prompt_yes_no("\nProceed with publishing?", true)? {
        println!("❌ Publishing cancelled");
        return Ok(vec![PublishResult::Cancelled]);
    }

    println!("\n🔨 Starting publishing workflow...\n");

    // Create publisher and check prerequisites
    let publisher = Publisher::new(config.clone());

    print!("⏳ Checking prerequisites... ");
    io::stdout().flush()?;

    match publisher.check_prerequisites()? {
        PublishStatus::Ready => {
            println!("✅\n");
        }
        PublishStatus::NotLoggedIn => {
            println!("❌\n");
            println!("⚠️  Not logged into cargo");
            println!("   Run: cargo login");
            return Err(anyhow::anyhow!("Not logged into cargo"));
        }
        PublishStatus::UncommittedChanges => {
            println!("⚠️\n");
            println!("⚠️  Uncommitted changes detected");
            if !prompt_yes_no("Continue anyway?", false)? {
                return Ok(vec![PublishResult::Cancelled]);
            }
            println!();
        }
        PublishStatus::TestsFailed => {
            println!("❌\n");
            println!("⚠️  Tests failed");
            if !prompt_yes_no("Continue anyway?", false)? {
                return Ok(vec![PublishResult::Cancelled]);
            }
            println!();
        }
        PublishStatus::MissingMetadata(fields) => {
            println!("❌\n");
            println!("⚠️  Missing required metadata in Cargo.toml:");
            for field in &fields {
                println!("   - {}", field);
            }
            return Err(anyhow::anyhow!("Missing required metadata"));
        }
        PublishStatus::GitNotAvailable => {
            println!("⚠️\n");
            println!("⚠️  Git not available");
            println!();
        }
    }

    // Execute publishing
    let results = publisher.publish()?;

    // Display results
    println!("\n✅ Publishing Complete!");
    println!("======================\n");

    for result in &results {
        match result {
            PublishResult::Published { crate_name, version } => {
                println!("✅ Published to crates.io: {} v{}", crate_name, version);
                println!("   View at: https://crates.io/crates/{}", crate_name);
            }
            PublishResult::RepositoryCreated { url } => {
                println!("✅ Created GitHub repository: {}", url);
            }
            PublishResult::DryRun { files_created } => {
                println!("✅ Dry run completed. Created {} file(s)", files_created.len());
            }
            PublishResult::Cancelled => {
                println!("❌ Publishing cancelled");
            }
        }
    }

    println!("\n🎉 All done!");

    Ok(results)
}

/// Prompt for yes/no answer
fn prompt_yes_no(question: &str, default: bool) -> Result<bool> {
    let default_str = if default { "Y/n" } else { "y/N" };
    print!("{} [{}]: ", question, default_str);
    io::stdout().flush()?;

    let mut input = String::new();
    io::stdin().read_line(&mut input)?;
    let input = input.trim().to_lowercase();

    Ok(match input.as_str() {
        "" => default,
        "y" | "yes" => true,
        "n" | "no" => false,
        _ => default,
    })
}

/// Prompt for license selection
fn prompt_license() -> Result<String> {
    println!("Select license:");
    println!("  1. MIT OR Apache-2.0 (recommended for Rust)");
    println!("  2. MIT");
    println!("  3. Apache-2.0");
    println!("  4. Custom (enter SPDX identifier)");
    print!("Choice [1]: ");
    io::stdout().flush()?;

    let mut input = String::new();
    io::stdin().read_line(&mut input)?;
    let input = input.trim();

    Ok(match input {
        "" | "1" => "MIT OR Apache-2.0".to_string(),
        "2" => "MIT".to_string(),
        "3" => "Apache-2.0".to_string(),
        "4" => {
            print!("Enter SPDX license identifier: ");
            io::stdout().flush()?;
            let mut license = String::new();
            io::stdin().read_line(&mut license)?;
            license.trim().to_string()
        }
        _ => "MIT OR Apache-2.0".to_string(),
    })
}

/// Get GitHub username from gh CLI or prompt
fn get_github_username() -> Result<String> {
    // Try to get from gh CLI
    if let Ok(output) = std::process::Command::new("gh")
        .args(["api", "user", "--jq", ".login"])
        .output()
        && output.status.success() {
            let username = String::from_utf8_lossy(&output.stdout);
            let username = username.trim();
            if !username.is_empty() {
                return Ok(username.to_string());
            }
        }

    // Prompt user
    print!("Enter GitHub username: ");
    io::stdout().flush()?;

    let mut username = String::new();
    io::stdin().read_line(&mut username)?;
    Ok(username.trim().to_string())
}

/// Get crate information from Cargo.toml
fn get_crate_info(crate_dir: &PathBuf) -> Result<CrateInfo> {
    let cargo_toml = crate_dir.join("Cargo.toml");
    let content = std::fs::read_to_string(&cargo_toml)
        .context("Failed to read Cargo.toml")?;

    let manifest: toml::Value = toml::from_str(&content)
        .context("Failed to parse Cargo.toml")?;

    let package = manifest
        .get("package")
        .context("No [package] section in Cargo.toml")?;

    let name = package
        .get("name")
        .and_then(|v| v.as_str())
        .context("No package.name in Cargo.toml")?
        .to_string();

    let version = package
        .get("version")
        .and_then(|v| v.as_str())
        .context("No package.version in Cargo.toml")?
        .to_string();

    let description = package
        .get("description")
        .and_then(|v| v.as_str())
        .unwrap_or("No description")
        .to_string();

    Ok(CrateInfo {
        name,
        version,
        description,
    })
}

/// Print the publishing plan
fn print_publishing_plan(config: &PublishConfig) {
    println!("  • License: {}", config.license);

    if config.create_github_repo {
        println!("  • Create GitHub repository");
        if let Some(username) = &config.github_username {
            println!("    └─ Owner: {}", username);
        }
    }

    if config.add_ci_workflows {
        println!("  • Add CI/CD workflows");
    }

    if config.publish_to_crates_io {
        println!("  • Publish to crates.io");
    }

    if config.dry_run {
        println!("  • DRY RUN (no actual publishing)");
    }
}

/// Crate information
struct CrateInfo {
    name: String,
    version: String,
    description: String,
}

#[cfg(test)]
mod tests {
    

    #[test]
    fn test_crate_info_parsing() {
        // This would need a test Cargo.toml file
        // Skipped for now
    }
}
