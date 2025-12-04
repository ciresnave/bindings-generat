//! library_db: local library symbol index + GitHub sync helpers
//!
//! Provides a small local database of library entries (name + symbols) and
//! helpers to (1) query the local DB for the most-likely libraries given a
//! set of symbol names, (2) update the local DB from a remote GitHub JSON
//! resource, and (3) submit a new library entry as a pull request against
//! the remote GitHub repo (requires a write-capable token).

use anyhow::{Context, Result, anyhow};
use base64::Engine as _;
use base64::engine::general_purpose::STANDARD as BASE64_ENGINE;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct LibraryEntry {
    pub name: String,
    pub description: Option<String>,
    pub symbols: Vec<String>,
    pub homepage: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LibraryDb {
    pub entries: Vec<LibraryEntry>,
}

impl LibraryDb {
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Load a library DB from a JSON file that contains an array of LibraryEntry
    pub fn load_from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let bytes =
            fs::read(&path).with_context(|| format!("reading {}", path.as_ref().display()))?;
        let entries: Vec<LibraryEntry> = serde_json::from_slice(&bytes).context("parsing JSON")?;
        Ok(Self { entries })
    }

    /// Write the DB back to disk as JSON
    pub fn save_to_file<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let text = serde_json::to_string_pretty(&self.entries)?;
        fs::write(path, text).context("writing db file")?;
        Ok(())
    }

    /// Naive greedy set-cover: given a list of symbol names, return an ordered
    /// list of libraries that together cover the most symbols using as few
    /// libraries as possible (greedy heuristic).
    pub fn suggest_libraries_for_symbols(
        &self,
        symbols: &[String],
    ) -> Vec<(LibraryEntry, Vec<String>)> {
        let mut uncovered: HashSet<String> = symbols.iter().map(|s| s.to_string()).collect();
        let mut chosen: Vec<(LibraryEntry, Vec<String>)> = Vec::new();

        while !uncovered.is_empty() {
            // find the library that covers the most uncovered symbols
            let mut best_idx: Option<usize> = None;
            let mut best_match_count: usize = 0;
            for (i, lib) in self.entries.iter().enumerate() {
                let matched: usize = lib
                    .symbols
                    .iter()
                    .filter(|s| uncovered.contains(*s))
                    .count();
                if matched > best_match_count {
                    best_match_count = matched;
                    best_idx = Some(i);
                }
            }

            if let Some(idx) = best_idx {
                let lib = self.entries[idx].clone();
                let matched_symbols: Vec<String> = lib
                    .symbols
                    .iter()
                    .filter(|s| uncovered.contains(*s))
                    .cloned()
                    .collect();

                if matched_symbols.is_empty() {
                    break;
                }

                // remove matched symbols from uncovered
                for s in &matched_symbols {
                    uncovered.remove(s);
                }

                chosen.push((lib, matched_symbols));
            } else {
                break;
            }
        }

        chosen
    }

    /// Merge (append/overwrite) entries from a remote list into the local DB.
    /// Uses `name` as the unique key; incoming entries replace local entries
    /// with the same name.
    pub fn merge_entries(&mut self, incoming: Vec<LibraryEntry>) {
        let mut map: HashMap<String, LibraryEntry> = HashMap::new();
        for e in self.entries.drain(..) {
            map.insert(e.name.clone(), e);
        }
        for incoming_e in incoming.into_iter() {
            map.insert(incoming_e.name.clone(), incoming_e);
        }
        self.entries = map.into_values().collect();
        // keep deterministic order
        self.entries.sort_by(|a, b| a.name.cmp(&b.name));
    }

    /// Update the local database from a JSON file hosted on GitHub (raw URL).
    /// `owner` and `repo` identify the GitHub repository, `branch` defaults to "main",
    /// and `path` is the path to the JSON file inside the repository.
    pub fn update_from_github(
        &mut self,
        owner: &str,
        repo: &str,
        branch: &str,
        path: &str,
    ) -> Result<()> {
        let url = format!(
            "https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{path}",
            owner = owner,
            repo = repo,
            branch = branch,
            path = path
        );

        let resp = reqwest::blocking::get(&url).with_context(|| format!("fetching {}", url))?;
        if !resp.status().is_success() {
            return Err(anyhow!("failed to fetch remote DB: {}", resp.status()));
        }
        let text = resp.text()?;
        let incoming: Vec<LibraryEntry> =
            serde_json::from_str(&text).context("parsing remote JSON")?;
        self.merge_entries(incoming);
        Ok(())
    }

    /// Submit a library entry to the remote GitHub database by creating a new
    /// branch, committing an updated JSON file and opening a pull request.
    /// This uses the GitHub REST API and requires a personal access token with
    /// repository access (passed as `token`). `db_path` is the path to the
    /// JSON file inside the repo. Returns the URL of the created PR on success.
    pub fn submit_entry_as_pr(
        &self,
        entry: &LibraryEntry,
        owner: &str,
        repo: &str,
        db_path: &str,
        token: &str,
        pr_title: &str,
        pr_body: &str,
    ) -> Result<String> {
        // 1) Get repo info to find default branch
        let client = reqwest::blocking::Client::new();
        let repo_api = format!(
            "https://api.github.com/repos/{owner}/{repo}",
            owner = owner,
            repo = repo
        );
        let repo_resp = client
            .get(&repo_api)
            .header("User-Agent", "library_db-client")
            .bearer_auth(token)
            .send()?;
        if !repo_resp.status().is_success() {
            return Err(anyhow!("failed to read repo info: {}", repo_resp.status()));
        }
        let repo_json: serde_json::Value = repo_resp.json()?;
        let default_branch = repo_json
            .get("default_branch")
            .and_then(|v| v.as_str())
            .unwrap_or("main")
            .to_string();

        // 2) Read existing db contents from the repo to update
        let contents_api = format!(
            "https://api.github.com/repos/{owner}/{repo}/contents/{path}?ref={branch}",
            owner = owner,
            repo = repo,
            path = db_path,
            branch = default_branch
        );
        let contents_resp = client
            .get(&contents_api)
            .header("User-Agent", "library_db-client")
            .bearer_auth(token)
            .send()?;
        if !contents_resp.status().is_success() {
            return Err(anyhow!(
                "failed to read db contents: {}",
                contents_resp.status()
            ));
        }
        let contents_json: serde_json::Value = contents_resp.json()?;
        let sha = contents_json
            .get("sha")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("missing sha in contents response"))?;
        let encoded = contents_json
            .get("content")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("missing content in contents response"))?;
        let decoded = BASE64_ENGINE.decode(encoded.replace("\n", ""))?;
        let mut existing: Vec<LibraryEntry> =
            serde_json::from_slice(&decoded).context("parsing existing db JSON")?;

        // add or replace the entry locally
        let mut map: HashMap<String, LibraryEntry> =
            existing.drain(..).map(|e| (e.name.clone(), e)).collect();
        map.insert(entry.name.clone(), entry.clone());
        let mut new_entries: Vec<LibraryEntry> = map.into_values().collect();
        new_entries.sort_by(|a, b| a.name.cmp(&b.name));
        let new_text = serde_json::to_string_pretty(&new_entries)?;
        let new_content_b64 = BASE64_ENGINE.encode(new_text.as_bytes());

        // 3) Create a new branch from default_branch
        // fetch reference for default branch
        let refs_api = format!(
            "https://api.github.com/repos/{owner}/{repo}/git/refs/heads/{branch}",
            owner = owner,
            repo = repo,
            branch = default_branch
        );
        let refs_resp = client
            .get(&refs_api)
            .header("User-Agent", "library_db-client")
            .bearer_auth(token)
            .send()?;
        if !refs_resp.status().is_success() {
            return Err(anyhow!("failed to get ref: {}", refs_resp.status()));
        }
        let refs_json: serde_json::Value = refs_resp.json()?;
        let commit_sha = refs_json
            .get("object")
            .and_then(|o| o.get("sha"))
            .and_then(|s| s.as_str())
            .ok_or_else(|| anyhow!("missing object.sha"))?;

        let new_branch = format!(
            "library-db-add-{}",
            chrono::Utc::now().format("%Y%m%d%H%M%S")
        );
        let create_ref_api = format!(
            "https://api.github.com/repos/{owner}/{repo}/git/refs",
            owner = owner,
            repo = repo
        );
        let create_ref_body = serde_json::json!({
            "ref": format!("refs/heads/{b}", b = new_branch),
            "sha": commit_sha
        });
        let create_ref_resp = client
            .post(&create_ref_api)
            .header("User-Agent", "library_db-client")
            .bearer_auth(token)
            .json(&create_ref_body)
            .send()?;
        if !create_ref_resp.status().is_success() {
            return Err(anyhow!(
                "failed to create branch: {}",
                create_ref_resp.status()
            ));
        }

        // 4) Update the file on the new branch using the contents API (PUT)
        let put_api = format!(
            "https://api.github.com/repos/{owner}/{repo}/contents/{path}",
            owner = owner,
            repo = repo,
            path = db_path
        );
        let put_body = serde_json::json!({
            "message": pr_title,
            "content": new_content_b64,
            "branch": new_branch,
            "sha": sha
        });
        let put_resp = client
            .put(&put_api)
            .header("User-Agent", "library_db-client")
            .bearer_auth(token)
            .json(&put_body)
            .send()?;
        if !put_resp.status().is_success() {
            return Err(anyhow!(
                "failed to create file on branch: {}",
                put_resp.status()
            ));
        }

        // 5) Create a pull request
        let pr_api = format!(
            "https://api.github.com/repos/{owner}/{repo}/pulls",
            owner = owner,
            repo = repo
        );
        let pr_body = serde_json::json!({
            "title": pr_title,
            "body": pr_body,
            "head": new_branch,
            "base": default_branch
        });
        let pr_resp = client
            .post(&pr_api)
            .header("User-Agent", "library_db-client")
            .bearer_auth(token)
            .json(&pr_body)
            .send()?;
        if !pr_resp.status().is_success() {
            return Err(anyhow!("failed to create PR: {}", pr_resp.status()));
        }
        let pr_json: serde_json::Value = pr_resp.json()?;
        let html_url = pr_json
            .get("html_url")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("missing html_url in PR response"))?;
        Ok(html_url.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[test]
    fn test_suggest_libraries_greedy() {
        let mut db = LibraryDb::new();
        db.entries.push(LibraryEntry {
            name: "liba".to_string(),
            description: None,
            symbols: vec!["foo".to_string(), "bar".to_string()],
            homepage: None,
        });
        db.entries.push(LibraryEntry {
            name: "libb".to_string(),
            description: None,
            symbols: vec!["baz".to_string(), "qux".to_string(), "foo".to_string()],
            homepage: None,
        });

        let symbols = vec!["foo".to_string(), "baz".to_string()];
        let suggestion = db.suggest_libraries_for_symbols(&symbols);
        // greedy should pick libb first (covers foo & baz)
        assert!(!suggestion.is_empty());
        assert_eq!(suggestion[0].0.name, "libb");
    }

    #[test]
    fn test_load_and_save_roundtrip() {
        let mut db = LibraryDb::new();
        db.entries.push(LibraryEntry {
            name: "libx".to_string(),
            description: Some("desc".to_string()),
            symbols: vec!["a".to_string()],
            homepage: None,
        });
        let tmp = NamedTempFile::new().unwrap();
        let path = tmp.path();
        db.save_to_file(path).unwrap();
        let loaded = LibraryDb::load_from_file(path).unwrap();
        assert_eq!(loaded.entries.len(), 1);
        assert_eq!(loaded.entries[0].name, "libx");
    }
}
