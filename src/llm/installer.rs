use anyhow::{Context, Result};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use tracing::{info, warn};

/// Types of Ollama installations
#[derive(Debug, Clone, PartialEq)]
pub enum OllamaInstallType {
    /// Ollama was already installed on the system
    AlreadyInstalled,
    /// We installed Ollama system-wide (requires sudo/admin)
    SystemWide,
    /// We installed Ollama portably in a temp directory
    Portable { path: PathBuf, process: Option<u32> },
}

/// Manages Ollama installation and lifecycle
pub struct OllamaInstaller {
    install_type: Option<OllamaInstallType>,
    temp_dir: Option<PathBuf>,
}

impl Default for OllamaInstaller {
    fn default() -> Self {
        Self::new()
    }
}

impl OllamaInstaller {
    pub fn new() -> Self {
        Self {
            install_type: None,
            temp_dir: None,
        }
    }

    /// Check if Ollama is already available on the system
    pub fn is_available() -> bool {
        // Try to run ollama --version
        if let Ok(output) = Command::new("ollama").arg("--version").output() {
            return output.status.success();
        }
        false
    }

    /// Prompt user for installation preference
    pub fn prompt_installation(&mut self) -> Result<OllamaInstallType> {
        println!();
        println!("🔍 Ollama not found on your system.");
        println!();
        println!("bindings-generat uses AI to enhance generated bindings.");
        println!("This requires Ollama (local LLM runtime).");
        println!();
        println!("Choose installation option:");
        println!("  1. System-wide install (requires admin/sudo, persists after run)");
        println!("  2. Portable install in temp directory (~1.3GB, auto-cleanup available)");
        println!("  3. Skip LLM features (generates basic bindings only)");
        println!();

        loop {
            print!("Your choice [1/2/3]: ");
            io::stdout().flush()?;

            let mut input = String::new();
            io::stdin().read_line(&mut input)?;

            match input.trim() {
                "1" => {
                    return Ok(OllamaInstallType::SystemWide);
                }
                "2" => {
                    let temp_path = self.create_temp_directory()?;
                    return Ok(OllamaInstallType::Portable {
                        path: temp_path,
                        process: None,
                    });
                }
                "3" => {
                    println!();
                    println!("⚠ Skipping LLM features. Bindings will have minimal documentation.");
                    return Err(anyhow::anyhow!("User skipped LLM installation"));
                }
                _ => {
                    println!("Invalid choice. Please enter 1, 2, or 3.");
                    continue;
                }
            }
        }
    }

    /// Create a temporary directory for portable installation
    fn create_temp_directory(&mut self) -> Result<PathBuf> {
        let temp_base = std::env::temp_dir();
        let uuid = uuid::Uuid::new_v4();
        let temp_path = temp_base.join(format!("bindings-generat-ollama-{}", uuid));

        std::fs::create_dir_all(&temp_path)
            .context("Failed to create temporary directory for Ollama")?;

        info!("Created temp directory: {}", temp_path.display());
        self.temp_dir = Some(temp_path.clone());
        Ok(temp_path)
    }

    /// Install Ollama system-wide
    pub fn install_system_wide(&mut self) -> Result<()> {
        println!();
        println!("📦 Installing Ollama system-wide...");
        println!("   This may require administrator/sudo permissions.");
        println!();

        let install_cmd = if cfg!(target_os = "windows") {
            // Windows: download installer and run it
            "winget install Ollama.Ollama"
        } else if cfg!(target_os = "macos") {
            // macOS: use homebrew or direct download
            "curl https://ollama.ai/install.sh | sh"
        } else {
            // Linux: use install script
            "curl https://ollama.ai/install.sh | sh"
        };

        println!("Running: {}", install_cmd);

        let status = if cfg!(target_os = "windows") {
            Command::new("cmd")
                .args(["/C", install_cmd])
                .status()
                .context("Failed to run Ollama installer")?
        } else {
            Command::new("sh")
                .args(["-c", install_cmd])
                .status()
                .context("Failed to run Ollama installer")?
        };

        if !status.success() {
            anyhow::bail!("Ollama installation failed");
        }

        println!("✓ Ollama installed successfully");
        self.install_type = Some(OllamaInstallType::SystemWide);
        Ok(())
    }

    /// Install Ollama portably in temp directory
    pub fn install_portable(
        &mut self,
        temp_path: PathBuf,
        model: &super::models::LlmModel,
    ) -> Result<Child> {
        println!();
        println!("📦 Setting up portable Ollama...");

        // Download Ollama binary
        let binary_path = self.download_ollama_binary(&temp_path)?;

        // Create models directory
        let models_dir = temp_path.join("models");
        std::fs::create_dir_all(&models_dir)?;

        println!("  • Starting Ollama server...");

        // Start Ollama server with custom environment
        let mut child = Command::new(&binary_path)
            .arg("serve")
            .env("OLLAMA_MODELS", &models_dir)
            .env("OLLAMA_HOST", "127.0.0.1:11435") // Use non-default port
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .context("Failed to start Ollama server")?;

        // Wait a bit for server to start
        std::thread::sleep(std::time::Duration::from_secs(3));

        // Check if process is still running
        if let Ok(Some(_)) = child.try_wait() {
            anyhow::bail!("Ollama server failed to start");
        }

        let pid = child.id();
        println!("  • Ollama server started (PID: {})", pid);

        // Download model
        self.download_model(&binary_path, &models_dir, model)?;

        println!("✓ Portable Ollama ready at {}", temp_path.display());
        println!();

        self.install_type = Some(OllamaInstallType::Portable {
            path: temp_path,
            process: Some(pid),
        });

        Ok(child)
    }

    /// Download Ollama binary for the current platform
    fn download_ollama_binary(&self, dest_dir: &Path) -> Result<PathBuf> {
        let (url, filename) = if cfg!(target_os = "windows") {
            (
                "https://github.com/ollama/ollama/releases/latest/download/ollama-windows-amd64.exe",
                "ollama.exe",
            )
        } else if cfg!(target_os = "macos") {
            (
                "https://github.com/ollama/ollama/releases/latest/download/ollama-darwin",
                "ollama",
            )
        } else {
            (
                "https://github.com/ollama/ollama/releases/latest/download/ollama-linux-amd64",
                "ollama",
            )
        };

        let binary_path = dest_dir.join(filename);

        println!("  • Downloading Ollama binary (~300MB)...");

        // Use network utilities with retry logic
        let config = super::network::DownloadConfig::new(url, &binary_path)
            .with_expected_size(300_000_000) // ~300MB estimate
            .with_description("Ollama binary");

        super::network::download_with_retry(&config).context("Failed to download Ollama binary")?;

        // Make executable on Unix
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut perms = std::fs::metadata(&binary_path)?.permissions();
            perms.set_mode(0o755);
            std::fs::set_permissions(&binary_path, perms)?;
        }

        Ok(binary_path)
    }

    /// Download the LLM model
    fn download_model(
        &self,
        ollama_binary: &PathBuf,
        models_dir: &PathBuf,
        model: &super::models::LlmModel,
    ) -> Result<()> {
        println!(
            "  • Downloading model {} ({})...",
            model.name(),
            model.size()
        );

        // Run ollama pull and show its output (it has built-in progress)
        let status = Command::new(ollama_binary)
            .arg("pull")
            .arg(model.name())
            .env("OLLAMA_MODELS", models_dir)
            .env("OLLAMA_HOST", "127.0.0.1:11435")
            .stdout(Stdio::inherit()) // Show ollama's progress output
            .stderr(Stdio::inherit())
            .status()
            .context("Failed to download model")?;

        if !status.success() {
            anyhow::bail!("Model download failed");
        }

        println!("    ✓ Model downloaded");
        Ok(())
    }

    /// Prompt user to clean up portable installation
    pub fn prompt_cleanup(&mut self) -> Result<()> {
        if let Some(OllamaInstallType::Portable { ref path, process }) = self.install_type {
            println!();
            print!("🧹 Clean up portable Ollama install? (~1.3GB freed) [Y/n]: ");
            io::stdout().flush()?;

            let mut input = String::new();
            io::stdin().read_line(&mut input)?;

            match input.trim().to_lowercase().as_str() {
                "" | "y" | "yes" => {
                    self.cleanup_portable(path.clone(), process)?;
                }
                _ => {
                    println!("   Keeping portable install at: {}", path.display());
                    println!("   (You can manually delete this directory later)");
                }
            }
        }

        Ok(())
    }

    /// Clean up portable installation
    fn cleanup_portable(&mut self, path: PathBuf, process_id: Option<u32>) -> Result<()> {
        println!("  • Stopping Ollama server...");

        // Kill the Ollama process if we have the PID
        if let Some(pid) = process_id {
            #[cfg(unix)]
            {
                let _ = Command::new("kill").arg(pid.to_string()).status();
            }
            #[cfg(windows)]
            {
                let _ = Command::new("taskkill")
                    .args(["/PID", &pid.to_string(), "/F"])
                    .status();
            }
        }

        println!("  • Removing {}...", path.display());
        std::fs::remove_dir_all(&path).context("Failed to remove temp directory")?;

        println!("✓ Cleanup complete");
        self.install_type = None;
        Ok(())
    }

    /// Get the install type
    pub fn install_type(&self) -> Option<&OllamaInstallType> {
        self.install_type.as_ref()
    }

    /// Check if we should use a custom Ollama host
    pub fn ollama_host(&self) -> String {
        if let Some(OllamaInstallType::Portable { .. }) = self.install_type {
            "http://127.0.0.1:11435".to_string()
        } else {
            "http://localhost:11434".to_string()
        }
    }
}

impl Drop for OllamaInstaller {
    fn drop(&mut self) {
        // Clean up temp directory if it exists and user hasn't been prompted
        if let Some(ref temp_dir) = self.temp_dir
            && temp_dir.exists()
        {
            warn!(
                "Temp directory still exists at drop: {}",
                temp_dir.display()
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_installer_creation() {
        let installer = OllamaInstaller::new();
        assert!(installer.install_type.is_none());
        assert!(installer.temp_dir.is_none());
    }

    #[test]
    fn test_ollama_host_default() {
        let installer = OllamaInstaller::new();
        assert_eq!(installer.ollama_host(), "http://localhost:11434");
    }

    #[test]
    fn test_ollama_host_portable() {
        let mut installer = OllamaInstaller::new();
        installer.install_type = Some(OllamaInstallType::Portable {
            path: PathBuf::from("/tmp/test"),
            process: Some(12345),
        });
        assert_eq!(installer.ollama_host(), "http://127.0.0.1:11435");
    }
}
