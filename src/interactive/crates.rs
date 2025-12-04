//! Interactive prompts for selecting from existing Rust crates.

use crate::discovery::crates_io::RustCrateInfo;
use anyhow::Result;
use dialoguer::{theme::ColorfulTheme, Confirm, Select};

/// Present found crates to the user and let them choose one.
///
/// Returns the selected crate, or None if the user declines all options.
pub fn prompt_select_existing_crate(
    library_name: &str,
    crates: &[RustCrateInfo],
) -> Result<Option<RustCrateInfo>> {
    if crates.is_empty() {
        return Ok(None);
    }

    println!("\n🦀 Found existing Rust crates for '{}':", library_name);
    println!();

    // Display crate details
    for (i, crate_info) in crates.iter().enumerate() {
        println!("{}. {}", i + 1, crate_info.name);
        println!("   Version: {}", crate_info.latest_version);
        println!(
            "   Downloads: {}",
            crate::discovery::crates_io::format_downloads(crate_info.downloads)
        );
        if let Some(desc) = &crate_info.description {
            println!("   Description: {}", desc);
        }
        if let Some(repo) = &crate_info.repository {
            println!("   Repository: {}", repo);
        }
        if let Some(docs) = &crate_info.documentation {
            println!("   Documentation: {}", docs);
        }
        println!();
    }

    // Ask if user wants to use one of these crates
    let use_existing = Confirm::with_theme(&ColorfulTheme::default())
        .with_prompt("Would you like to use one of these existing crates instead of generating new bindings?")
        .default(true)
        .interact()?;

    if !use_existing {
        return Ok(None);
    }

    // Let user select which crate
    let selections: Vec<String> = crates
        .iter()
        .map(|c| {
            format!(
                "{} (v{}, {} downloads)",
                c.name,
                c.latest_version,
                crate::discovery::crates_io::format_downloads(c.downloads)
            )
        })
        .collect();

    let selection = Select::with_theme(&ColorfulTheme::default())
        .with_prompt("Select a crate to use")
        .items(&selections)
        .default(0)
        .interact()?;

    Ok(Some(crates[selection].clone()))
}

/// Display instructions for adding the selected crate to Cargo.toml.
pub fn show_cargo_instructions(crate_info: &RustCrateInfo) {
    println!("\n✨ To use {} in your project, add to your Cargo.toml:", crate_info.name);
    println!();
    println!("[dependencies]");
    println!("{} = \"{}\"", crate_info.name, crate_info.latest_version);
    println!();

    if let Some(repo) = &crate_info.repository {
        println!("📦 Repository: {}", repo);
    }
    if let Some(docs) = &crate_info.documentation {
        println!("📚 Documentation: {}", docs);
    }
    if let Some(homepage) = &crate_info.homepage {
        println!("🏠 Homepage: {}", homepage);
    }
}

/// Full workflow: search, prompt, and display instructions.
///
/// Returns true if user selected an existing crate (and won't need new bindings).
pub fn handle_existing_crates_workflow(
    library_name: &str,
    crates: &[RustCrateInfo],
) -> Result<bool> {
    if let Some(selected) = prompt_select_existing_crate(library_name, crates)? {
        show_cargo_instructions(&selected);
        Ok(true) // User chose existing crate
    } else {
        println!(
            "\n✓ Continuing with binding generation for '{}'",
            library_name
        );
        Ok(false) // User wants to generate new bindings
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_crates_returns_none() {
        let result = prompt_select_existing_crate("testlib", &[]);
        // Can't test interactively, but we can verify it doesn't panic
        assert!(result.is_ok());
    }

    #[test]
    fn test_cargo_instructions_display() {
        let crate_info = RustCrateInfo {
            name: "test-crate".to_string(),
            latest_version: "1.0.0".to_string(),
            description: Some("A test crate".to_string()),
            repository: Some("https://github.com/user/test-crate".to_string()),
            documentation: Some("https://docs.rs/test-crate".to_string()),
            homepage: Some("https://example.com".to_string()),
            downloads: 10000,
            crates_io_url: "https://crates.io/crates/test-crate".to_string(),
            updated_at: "2024-01-01T00:00:00Z".to_string(),
            license: Some("MIT".to_string()),
            keywords: vec!["ffi".to_string()],
        };

        // Just verify it doesn't panic
        show_cargo_instructions(&crate_info);
    }
}
