//! Git CLI-based submission of library discoveries.
//!
//! Uses the system's `git` and `gh` CLI tools to automate the process of:
//! 1. Forking the repository (if needed)
//! 2. Cloning the fork to a temporary directory
//! 3. Creating a new branch
//! 4. Adding the library file
//! 5. Committing and pushing
//! 6. Creating a pull request
//!
//! This method leverages the user's existing GitHub authentication via `gh`,
//! avoiding the need to store a Personal Access Token.

use anyhow::{Context, Result, bail};
use std::path::Path;
use std::process::{Command, Output};
use tempfile::TempDir;

/// Check if git and gh CLI tools are available and properly authenticated.
pub fn check_available() -> Result<GitCliInfo> {
    // Check git
    let git_output = Command::new("git")
        .arg("--version")
        .output()
        .context("Failed to execute git. Is git installed?")?;

    if !git_output.status.success() {
        bail!("git command failed");
    }

    let git_version = parse_version_output(&git_output)?;

    // Check gh
    let gh_output = Command::new("gh")
        .arg("--version")
        .output()
        .context("Failed to execute gh. Install from: https://cli.github.com/")?;

    if !gh_output.status.success() {
        bail!("gh command failed");
    }

    let gh_version = parse_version_output(&gh_output)?;

    // Check gh auth status
    let auth_output = Command::new("gh")
        .args(["auth", "status"])
        .output()
        .context("Failed to check gh authentication")?;

    if !auth_output.status.success() {
        bail!("gh is not authenticated. Run: gh auth login");
    }

    // Extract username from auth status
    let auth_text = String::from_utf8_lossy(&auth_output.stderr);
    let username = extract_username(&auth_text).unwrap_or_else(|| "unknown".to_string());

    Ok(GitCliInfo {
        git_version,
        gh_version,
        username,
    })
}

/// Information about available git CLI tools.
#[derive(Debug, Clone)]
pub struct GitCliInfo {
    pub git_version: String,
    pub gh_version: String,
    pub username: String,
}

/// Submit a library discovery using git and gh CLI tools.
///
/// # Arguments
///
/// * `library_name` - Name of the library (e.g., "openssl")
/// * `library_toml` - Complete TOML content for the library
/// * `repo_owner` - GitHub repository owner
/// * `repo_name` - GitHub repository name
/// * `attribution` - Optional git author information
///
/// # Returns
///
/// URL of the created pull request.
pub fn submit_via_cli(
    library_name: &str,
    library_toml: &str,
    repo_owner: &str,
    repo_name: &str,
    attribution: Option<(&str, &str)>, // (name, email)
) -> Result<String> {
    // Create temporary directory for the clone
    let temp_dir = TempDir::new().context("Failed to create temporary directory")?;
    let repo_path = temp_dir.path();

    println!("  → Checking fork status...");
    ensure_fork(repo_owner, repo_name)?;

    println!("  → Cloning repository...");
    clone_fork(repo_owner, repo_name, repo_path)?;

    // Configure git author if provided
    if let Some((name, email)) = attribution {
        configure_git_author(repo_path, name, email)?;
    }

    let branch_name = format!("library/{}-discovery", library_name);
    println!("  → Creating branch: {}", branch_name);
    create_branch(repo_path, &branch_name)?;

    println!("  → Adding library file...");
    add_library_file(repo_path, library_name, library_toml)?;

    println!("  → Committing changes...");
    commit_changes(repo_path, library_name)?;

    println!("  → Pushing to fork...");
    push_branch(repo_path, &branch_name)?;

    println!("  → Creating pull request...");
    let pr_url = create_pull_request(repo_path, repo_owner, repo_name, library_name, &branch_name)?;

    Ok(pr_url)
}

fn parse_version_output(output: &Output) -> Result<String> {
    let text = String::from_utf8_lossy(&output.stdout);
    let version = text
        .lines()
        .next()
        .context("No output from version command")?
        .to_string();
    Ok(version)
}

fn extract_username(auth_text: &str) -> Option<String> {
    // Parse "Logged in to github.com as USERNAME (oauth_token)"
    for line in auth_text.lines() {
        if line.contains("Logged in") && line.contains(" as ")
            && let Some(start) = line.find(" as ") {
                let after_as = &line[start + 4..];
                if let Some(end) = after_as.find(' ') {
                    return Some(after_as[..end].to_string());
                }
            }
    }
    None
}

fn ensure_fork(repo_owner: &str, repo_name: &str) -> Result<()> {
    // Try to view the fork; if it doesn't exist, create it
    let check = Command::new("gh")
        .args([
            "repo",
            "view",
            &format!("{}/{}", repo_owner, repo_name),
            "--json",
            "isFork",
        ])
        .output()
        .context("Failed to check fork status")?;

    if !check.status.success() {
        // Fork doesn't exist, create it
        let fork = Command::new("gh")
            .args([
                "repo",
                "fork",
                &format!("{}/{}", repo_owner, repo_name),
                "--clone=false",
            ])
            .output()
            .context("Failed to create fork")?;

        if !fork.status.success() {
            let stderr = String::from_utf8_lossy(&fork.stderr);
            bail!("Failed to create fork: {}", stderr);
        }
    }

    Ok(())
}

fn clone_fork(repo_owner: &str, repo_name: &str, dest: &Path) -> Result<()> {
    let output = Command::new("gh")
        .args([
            "repo",
            "clone",
            &format!("{}/{}", repo_owner, repo_name),
            dest.to_str().context("Invalid path")?,
        ])
        .output()
        .context("Failed to clone repository")?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        bail!("Failed to clone repository: {}", stderr);
    }

    Ok(())
}

fn configure_git_author(repo_path: &Path, name: &str, email: &str) -> Result<()> {
    Command::new("git")
        .args(["-C", repo_path.to_str().context("Invalid path")?])
        .args(["config", "user.name", name])
        .output()
        .context("Failed to configure git user.name")?;

    Command::new("git")
        .args(["-C", repo_path.to_str().context("Invalid path")?])
        .args(["config", "user.email", email])
        .output()
        .context("Failed to configure git user.email")?;

    Ok(())
}

fn create_branch(repo_path: &Path, branch_name: &str) -> Result<()> {
    let output = Command::new("git")
        .args(["-C", repo_path.to_str().context("Invalid path")?])
        .args(["checkout", "-b", branch_name])
        .output()
        .context("Failed to create branch")?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        bail!("Failed to create branch: {}", stderr);
    }

    Ok(())
}

fn add_library_file(repo_path: &Path, library_name: &str, library_toml: &str) -> Result<()> {
    // Create libraries directory if it doesn't exist
    let libraries_dir = repo_path.join("libraries");
    std::fs::create_dir_all(&libraries_dir).context("Failed to create libraries directory")?;

    // Write library file
    let library_file = libraries_dir.join(format!("{}.toml", library_name));
    std::fs::write(&library_file, library_toml).context("Failed to write library file")?;

    // Git add
    let output = Command::new("git")
        .args(["-C", repo_path.to_str().context("Invalid path")?])
        .args(["add", &format!("libraries/{}.toml", library_name)])
        .output()
        .context("Failed to git add")?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        bail!("Failed to git add: {}", stderr);
    }

    Ok(())
}

fn commit_changes(repo_path: &Path, library_name: &str) -> Result<()> {
    let message = format!("Add {} to library database", library_name);

    let output = Command::new("git")
        .args(["-C", repo_path.to_str().context("Invalid path")?])
        .args(["commit", "-m", &message])
        .output()
        .context("Failed to commit")?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        bail!("Failed to commit: {}", stderr);
    }

    Ok(())
}

fn push_branch(repo_path: &Path, branch_name: &str) -> Result<()> {
    let output = Command::new("git")
        .args(["-C", repo_path.to_str().context("Invalid path")?])
        .args(["push", "-u", "origin", branch_name])
        .output()
        .context("Failed to push")?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        bail!("Failed to push: {}", stderr);
    }

    Ok(())
}

fn create_pull_request(
    repo_path: &Path,
    repo_owner: &str,
    repo_name: &str,
    library_name: &str,
    branch_name: &str,
) -> Result<String> {
    let title = format!("Add {} to library database", library_name);
    let body = format!(
        "This PR adds `{}` to the library database.\n\n\
         Automatically generated by bindings-generat.",
        library_name
    );

    let output = Command::new("gh")
        .args(["-C", repo_path.to_str().context("Invalid path")?])
        .args([
            "pr",
            "create",
            "--repo",
            &format!("{}/{}", repo_owner, repo_name),
            "--base",
            "main",
            "--head",
            branch_name,
            "--title",
            &title,
            "--body",
            &body,
        ])
        .output()
        .context("Failed to create pull request")?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        bail!("Failed to create pull request: {}", stderr);
    }

    let pr_url = String::from_utf8_lossy(&output.stdout).trim().to_string();
    Ok(pr_url)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_username() {
        let auth_text = "  ✓ Logged in to github.com as testuser (oauth_token)\n";
        assert_eq!(extract_username(auth_text), Some("testuser".to_string()));

        let no_match = "Some other text";
        assert_eq!(extract_username(no_match), None);
    }

    #[test]
    fn test_parse_version_output() {
        let output = Output {
            status: std::process::ExitStatus::default(),
            stdout: b"git version 2.42.0\n".to_vec(),
            stderr: vec![],
        };
        let version = parse_version_output(&output).unwrap();
        assert!(version.contains("2.42.0"));
    }
}
