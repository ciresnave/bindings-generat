//! GitHub API-based submission of library discoveries.
//!
//! Uses GitHub's REST API with a Personal Access Token to automate the process of:
//! 1. Forking the repository (if needed)
//! 2. Creating a new branch
//! 3. Creating/updating the library file
//! 4. Creating a pull request
//!
//! This method requires a GitHub PAT with `public_repo` scope.

use anyhow::{Context, Result, bail};
use base64::{Engine, engine::general_purpose::STANDARD};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

const GITHUB_API_BASE: &str = "https://api.github.com";

/// Submit a library discovery using GitHub REST API with a Personal Access Token.
///
/// # Arguments
///
/// * `library_name` - Name of the library (e.g., "openssl")
/// * `library_toml` - Complete TOML content for the library
/// * `repo_owner` - GitHub repository owner
/// * `repo_name` - GitHub repository name
/// * `github_token` - Personal Access Token with `public_repo` scope
///
/// # Returns
///
/// URL of the created pull request.
pub fn submit_via_api(
    library_name: &str,
    library_toml: &str,
    repo_owner: &str,
    repo_name: &str,
    github_token: &str,
) -> Result<String> {
    let client = reqwest::blocking::Client::new();

    println!("  → Getting authenticated user info...");
    let user = get_authenticated_user(&client, github_token)?;

    println!("  → Checking fork status...");
    let fork_owner = ensure_fork(&client, github_token, repo_owner, repo_name, &user.login)?;

    println!("  → Getting default branch...");
    let default_branch = get_default_branch(&client, github_token, repo_owner, repo_name)?;

    let branch_name = format!("library/{}-discovery", library_name);
    println!("  → Creating branch: {}", branch_name);
    create_branch(
        &client,
        github_token,
        &fork_owner,
        repo_name,
        &branch_name,
        &default_branch,
    )?;

    println!("  → Adding library file...");
    create_or_update_file(
        &client,
        github_token,
        &fork_owner,
        repo_name,
        &branch_name,
        library_name,
        library_toml,
    )?;

    println!("  → Creating pull request...");
    let pr_url = create_pull_request(
        &client,
        github_token,
        repo_owner,
        repo_name,
        &default_branch,
        &fork_owner,
        &branch_name,
        library_name,
    )?;

    Ok(pr_url)
}

#[derive(Debug, Deserialize)]
struct User {
    login: String,
}

#[derive(Debug, Deserialize)]
struct Repository {
    #[allow(dead_code)]
    owner: Owner,
    default_branch: String,
}

#[derive(Debug, Deserialize)]
struct Owner {
    #[allow(dead_code)]
    login: String,
}

#[derive(Debug, Deserialize)]
struct GitRef {
    object: GitObject,
}

#[derive(Debug, Deserialize)]
struct GitObject {
    sha: String,
}

#[derive(Debug, Deserialize)]
struct PullRequest {
    html_url: String,
}

fn get_authenticated_user(client: &reqwest::blocking::Client, token: &str) -> Result<User> {
    let response = client
        .get(format!("{}/user", GITHUB_API_BASE))
        .header("Authorization", format!("Bearer {}", token))
        .header("User-Agent", "bindings-generat")
        .header("Accept", "application/vnd.github+json")
        .header("X-GitHub-Api-Version", "2022-11-28")
        .send()
        .context("Failed to get authenticated user")?;

    if !response.status().is_success() {
        bail!("Failed to authenticate with GitHub: {}", response.status());
    }

    let user: User = response.json().context("Failed to parse user response")?;
    Ok(user)
}

fn ensure_fork(
    client: &reqwest::blocking::Client,
    token: &str,
    repo_owner: &str,
    repo_name: &str,
    username: &str,
) -> Result<String> {
    // Check if fork already exists
    let fork_url = format!("{}/repos/{}/{}", GITHUB_API_BASE, username, repo_name);
    let check_response = client
        .get(&fork_url)
        .header("Authorization", format!("Bearer {}", token))
        .header("User-Agent", "bindings-generat")
        .header("Accept", "application/vnd.github+json")
        .header("X-GitHub-Api-Version", "2022-11-28")
        .send()
        .context("Failed to check fork status")?;

    if check_response.status().is_success() {
        // Fork exists
        return Ok(username.to_string());
    }

    // Fork doesn't exist, create it
    let fork_url = format!(
        "{}/repos/{}/{}/forks",
        GITHUB_API_BASE, repo_owner, repo_name
    );
    let fork_response = client
        .post(&fork_url)
        .header("Authorization", format!("Bearer {}", token))
        .header("User-Agent", "bindings-generat")
        .header("Accept", "application/vnd.github+json")
        .header("X-GitHub-Api-Version", "2022-11-28")
        .send()
        .context("Failed to create fork")?;

    if !fork_response.status().is_success() {
        bail!("Failed to create fork: {}", fork_response.status());
    }

    // Wait a moment for fork to be ready
    std::thread::sleep(std::time::Duration::from_secs(2));

    Ok(username.to_string())
}

fn get_default_branch(
    client: &reqwest::blocking::Client,
    token: &str,
    repo_owner: &str,
    repo_name: &str,
) -> Result<String> {
    let url = format!("{}/repos/{}/{}", GITHUB_API_BASE, repo_owner, repo_name);
    let response = client
        .get(&url)
        .header("Authorization", format!("Bearer {}", token))
        .header("User-Agent", "bindings-generat")
        .header("Accept", "application/vnd.github+json")
        .header("X-GitHub-Api-Version", "2022-11-28")
        .send()
        .context("Failed to get repository info")?;

    if !response.status().is_success() {
        bail!("Failed to get repository: {}", response.status());
    }

    let repo: Repository = response
        .json()
        .context("Failed to parse repository response")?;
    Ok(repo.default_branch)
}

fn create_branch(
    client: &reqwest::blocking::Client,
    token: &str,
    owner: &str,
    repo_name: &str,
    branch_name: &str,
    base_branch: &str,
) -> Result<()> {
    // Get SHA of base branch
    let ref_url = format!(
        "{}/repos/{}/{}/git/ref/heads/{}",
        GITHUB_API_BASE, owner, repo_name, base_branch
    );
    let ref_response = client
        .get(&ref_url)
        .header("Authorization", format!("Bearer {}", token))
        .header("User-Agent", "bindings-generat")
        .header("Accept", "application/vnd.github+json")
        .header("X-GitHub-Api-Version", "2022-11-28")
        .send()
        .context("Failed to get base branch ref")?;

    if !ref_response.status().is_success() {
        bail!("Failed to get base branch: {}", ref_response.status());
    }

    let git_ref: GitRef = ref_response
        .json()
        .context("Failed to parse ref response")?;
    let base_sha = git_ref.object.sha;

    // Create new branch
    let create_url = format!("{}/repos/{}/{}/git/refs", GITHUB_API_BASE, owner, repo_name);
    let mut body = HashMap::new();
    body.insert("ref", format!("refs/heads/{}", branch_name));
    body.insert("sha", base_sha);

    let create_response = client
        .post(&create_url)
        .header("Authorization", format!("Bearer {}", token))
        .header("User-Agent", "bindings-generat")
        .header("Accept", "application/vnd.github+json")
        .header("X-GitHub-Api-Version", "2022-11-28")
        .json(&body)
        .send()
        .context("Failed to create branch")?;

    if !create_response.status().is_success() {
        let status = create_response.status();
        let text = create_response.text().unwrap_or_default();
        bail!("Failed to create branch: {} - {}", status, text);
    }

    Ok(())
}

fn create_or_update_file(
    client: &reqwest::blocking::Client,
    token: &str,
    owner: &str,
    repo_name: &str,
    branch_name: &str,
    library_name: &str,
    library_toml: &str,
) -> Result<()> {
    let file_path = format!("libraries/{}.toml", library_name);
    let url = format!(
        "{}/repos/{}/{}/contents/{}",
        GITHUB_API_BASE, owner, repo_name, file_path
    );

    let message = format!("Add {} to library database", library_name);
    let content = STANDARD.encode(library_toml);

    #[derive(Serialize)]
    struct CreateFileRequest {
        message: String,
        content: String,
        branch: String,
    }

    let body = CreateFileRequest {
        message,
        content,
        branch: branch_name.to_string(),
    };

    let response = client
        .put(&url)
        .header("Authorization", format!("Bearer {}", token))
        .header("User-Agent", "bindings-generat")
        .header("Accept", "application/vnd.github+json")
        .header("X-GitHub-Api-Version", "2022-11-28")
        .json(&body)
        .send()
        .context("Failed to create file")?;

    if !response.status().is_success() {
        let status = response.status();
        let text = response.text().unwrap_or_default();
        bail!("Failed to create file: {} - {}", status, text);
    }

    Ok(())
}

fn create_pull_request(
    client: &reqwest::blocking::Client,
    token: &str,
    base_owner: &str,
    repo_name: &str,
    base_branch: &str,
    head_owner: &str,
    head_branch: &str,
    library_name: &str,
) -> Result<String> {
    let url = format!(
        "{}/repos/{}/{}/pulls",
        GITHUB_API_BASE, base_owner, repo_name
    );

    #[derive(Serialize)]
    struct CreatePullRequest {
        title: String,
        body: String,
        head: String,
        base: String,
    }

    let body = CreatePullRequest {
        title: format!("Add {} to library database", library_name),
        body: format!(
            "This PR adds `{}` to the library database.\n\n\
             Automatically generated by bindings-generat.",
            library_name
        ),
        head: format!("{}:{}", head_owner, head_branch),
        base: base_branch.to_string(),
    };

    let response = client
        .post(&url)
        .header("Authorization", format!("Bearer {}", token))
        .header("User-Agent", "bindings-generat")
        .header("Accept", "application/vnd.github+json")
        .header("X-GitHub-Api-Version", "2022-11-28")
        .json(&body)
        .send()
        .context("Failed to create pull request")?;

    if !response.status().is_success() {
        let status = response.status();
        let text = response.text().unwrap_or_default();
        bail!("Failed to create pull request: {} - {}", status, text);
    }

    let pr: PullRequest = response
        .json()
        .context("Failed to parse pull request response")?;
    Ok(pr.html_url)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_api_constants() {
        assert_eq!(GITHUB_API_BASE, "https://api.github.com");
    }
}
