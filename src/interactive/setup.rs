//! First-run setup wizard for bindings-generat.
//!
//! Guides users through initial configuration, including:
//! - Google Custom Search API setup (optional)
//! - Community contribution preferences
//! - Submission method selection

use crate::user_config::{
    self, Attribution, CommunityConfig, Config, GoogleSearchConfig, SubmissionMethod,
};
use anyhow::Result;
use dialoguer::{Confirm, Input, Select, theme::ColorfulTheme};

/// Run the first-run setup wizard.
///
/// Prompts the user for configuration options and saves them to the config file.
pub fn run_setup_wizard() -> Result<Config> {
    println!("\n=== Welcome to bindings-generat! ===\n");
    println!("Let's set up some optional features to improve your experience.\n");

    let config = Config {
        google_search: setup_google_search()?,
        community: setup_community_contributions()?,
    };

    // Save configuration
    user_config::save(&config)?;

    let config_path = user_config::config_path()?;
    println!("\n✓ Configuration saved to: {}", config_path.display());
    println!("You can change these settings anytime by editing the config file.\n");

    Ok(config)
}

/// Check if this is the first run (config doesn't exist).
pub fn is_first_run() -> bool {
    !user_config::exists()
}

/// Prompt for first-run setup if needed.
///
/// If this is the first run, prompts the user whether they want to configure optional features.
/// If they decline, creates a default config file.
pub fn prompt_first_run_if_needed() -> Result<Config> {
    if !is_first_run() {
        return user_config::load();
    }

    println!("\nThis appears to be your first time running bindings-generat.");

    let should_setup = Confirm::with_theme(&ColorfulTheme::default())
        .with_prompt("Would you like to configure optional features now?")
        .default(true)
        .interact()?;

    if should_setup {
        run_setup_wizard()
    } else {
        // Create default config
        let config = Config::default();
        user_config::save(&config)?;
        println!("Using default configuration. You can run setup later if needed.\n");
        Ok(config)
    }
}

fn setup_google_search() -> Result<GoogleSearchConfig> {
    println!("--- Google Custom Search API ---");
    println!("This enables automatic discovery of unknown libraries.");
    println!("Get API credentials at: https://developers.google.com/custom-search/v1/overview");
    println!("(100 free queries per day)\n");

    let enable = Confirm::with_theme(&ColorfulTheme::default())
        .with_prompt("Enable automatic library discovery?")
        .default(false)
        .interact()?;

    if !enable {
        return Ok(GoogleSearchConfig::default());
    }

    let api_key: String = Input::with_theme(&ColorfulTheme::default())
        .with_prompt("Enter your Google Custom Search API key")
        .allow_empty(true)
        .interact_text()?;

    if api_key.is_empty() {
        println!("Skipping Google Search setup.");
        return Ok(GoogleSearchConfig::default());
    }

    let search_engine_id: String = Input::with_theme(&ColorfulTheme::default())
        .with_prompt("Enter your Custom Search Engine ID")
        .allow_empty(true)
        .interact_text()?;

    if search_engine_id.is_empty() {
        println!("Skipping Google Search setup (both API key and Search Engine ID are required).");
        return Ok(GoogleSearchConfig::default());
    }

    Ok(GoogleSearchConfig {
        api_key: Some(api_key),
        search_engine_id: Some(search_engine_id),
    })
}

fn setup_community_contributions() -> Result<CommunityConfig> {
    println!("\n--- Community Contributions ---");
    println!("Help improve bindings-generat by sharing discovered libraries.");
    println!("Submissions are reviewed by maintainers before being published.\n");

    let contribute = Confirm::with_theme(&ColorfulTheme::default())
        .with_prompt("Contribute discovered libraries back to the community?")
        .default(false)
        .interact()?;

    if !contribute {
        return Ok(CommunityConfig::default());
    }

    // Select submission method
    let methods = [SubmissionMethod::GitCli,
        SubmissionMethod::GithubToken,
        SubmissionMethod::Manual];

    let method_names: Vec<&str> = methods.iter().map(|m| m.description()).collect();

    let selection = Select::with_theme(&ColorfulTheme::default())
        .with_prompt("Select submission method")
        .items(&method_names)
        .default(0)
        .interact()?;

    let submission_method = methods[selection];

    // Get GitHub token if needed
    let github_token = if submission_method == SubmissionMethod::GithubToken {
        let token: String = Input::with_theme(&ColorfulTheme::default())
            .with_prompt("Enter your GitHub Personal Access Token (with public_repo scope)")
            .allow_empty(true)
            .interact_text()?;

        if token.is_empty() { None } else { Some(token) }
    } else {
        None
    };

    // Optional attribution
    let set_attribution = Confirm::with_theme(&ColorfulTheme::default())
        .with_prompt("Set attribution information for commits? (optional)")
        .default(false)
        .interact()?;

    let attribution = if set_attribution {
        let name: String = Input::with_theme(&ColorfulTheme::default())
            .with_prompt("Your name")
            .allow_empty(true)
            .interact_text()?;

        let email: String = Input::with_theme(&ColorfulTheme::default())
            .with_prompt("Your email")
            .allow_empty(true)
            .interact_text()?;

        Attribution {
            name: if name.is_empty() { None } else { Some(name) },
            email: if email.is_empty() { None } else { Some(email) },
        }
    } else {
        Attribution::default()
    };

    Ok(CommunityConfig {
        contribute_discoveries: contribute,
        submission_method,
        github_token,
        attribution,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_first_run() {
        // We can't easily test this without mocking the filesystem,
        // but we can verify the function signature
        let _ = is_first_run();
    }
}
