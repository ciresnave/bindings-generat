use anyhow::{Context, Result};
use console::style;
use dialoguer::{Confirm, Input, MultiSelect};
use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use tracing::{debug, info, warn};

use super::error_parser::{BuildError, parse_build_errors, suggest_missing_libraries};
use super::library_finder::{extract_libraries_to_find, find_library_on_system};

const MAX_RETRY_ATTEMPTS: usize = 5;

/// Result of an iterative build attempt
#[derive(Debug)]
pub enum BuildResult {
    Success,
    FailedAfterRetries { final_errors: Vec<BuildError> },
    UserCancelled,
}

/// Iteratively build, detect errors, fix, and retry
pub fn iterative_build_and_refine(
    output_dir: &Path,
    lib_name: &str,
    initial_dependencies: &[String],
    interactive: bool,
) -> Result<BuildResult> {
    let mut known_dependencies = initial_dependencies.to_vec();
    let mut attempted_libs: HashSet<String> = HashSet::new();

    for attempt in 1..=MAX_RETRY_ATTEMPTS {
        info!("→ Build attempt {}/{}", attempt, MAX_RETRY_ATTEMPTS);

        // Try to build
        let build_output = run_cargo_build(output_dir)?;

        // Check if successful
        if build_output.status.success() {
            info!("✓ Build successful!");

            // Run cargo fix to clean up warnings automatically
            info!("→ Running cargo fix to resolve warnings...");
            match run_cargo_fix(output_dir) {
                Ok(fixes_applied) => {
                    if fixes_applied {
                        info!("✓ cargo fix applied automatic improvements");
                    } else {
                        debug!("No automatic fixes needed");
                    }
                }
                Err(e) => {
                    warn!("cargo fix failed (non-critical): {}", e);
                }
            }

            return Ok(BuildResult::Success);
        }

        // Parse errors from both stderr and stdout (linker errors often go to stdout on Windows)
        let stderr = String::from_utf8_lossy(&build_output.stderr);
        let stdout = String::from_utf8_lossy(&build_output.stdout);

        // Show what we're getting
        info!(
            "Build output: stderr={} bytes, stdout={} bytes",
            build_output.stderr.len(),
            build_output.stdout.len()
        );

        // Debug: show lines containing "unresolved" to diagnose parsing
        let unresolved_lines: Vec<&str> = stderr
            .lines()
            .filter(|line| line.to_lowercase().contains("unresolved"))
            .collect();
        if !unresolved_lines.is_empty() {
            info!("Found {} lines with 'unresolved':", unresolved_lines.len());
            for (i, line) in unresolved_lines.iter().take(3).enumerate() {
                info!("  Unresolved #{}: {:?}", i + 1, line);
            }
        } else {
            info!("No lines containing 'unresolved' found in stderr");
        }

        // Combine both streams for error parsing
        let combined_output = format!("{}\n{}", stderr, stdout);

        let errors = parse_build_errors(&combined_output);
        if errors.is_empty() {
            warn!("Build failed but no actionable errors detected");
            return Ok(BuildResult::FailedAfterRetries {
                final_errors: vec![],
            });
        }

        info!("→ Detected {} errors, analyzing...", errors.len());

        // Debug: show what the first error looks like
        if let Some(first_error) = errors.first() {
            info!("  First error structure: {:?}", first_error);
        }

        // First, try to intelligently find missing libraries on the system
        let libs_to_find = extract_libraries_to_find(&errors);
        info!(
            "→ Extracted {} unique library names to search for",
            libs_to_find.len()
        );
        for lib in &libs_to_find {
            info!("  - {}", lib);
        }

        let mut found_lib_paths: Vec<PathBuf> = Vec::new();

        for lib_name in &libs_to_find {
            info!("→ Attempting to locate library: {}", lib_name);
            match find_library_on_system(lib_name) {
                Ok(paths) if !paths.is_empty() => {
                    info!("✓ Found {} at {} location(s)", lib_name, paths.len());
                    found_lib_paths.extend(paths);
                }
                Ok(_) => {
                    warn!("✗ Could not locate {} on system", lib_name);
                }
                Err(e) => {
                    warn!("Error searching for {}: {}", lib_name, e);
                }
            }
        }

        // If we found library paths, add them to build.rs
        if !found_lib_paths.is_empty() {
            info!(
                "→ Adding {} library path(s) to build.rs",
                found_lib_paths.len()
            );
            if let Err(e) = add_library_paths_to_build_rs(output_dir, &found_lib_paths) {
                warn!("Failed to add library paths to build.rs: {}", e);
            } else {
                // Clean and retry immediately since we found libraries
                clean_build_artifacts(output_dir)?;
                continue;
            }
        }

        // Suggest missing libraries
        let suggestions = suggest_missing_libraries(&errors);

        if suggestions.is_empty() {
            info!("⚠ No automatic suggestions available");

            if interactive && attempt < MAX_RETRY_ATTEMPTS {
                // Ask user for help
                match prompt_user_for_dependencies(&errors, lib_name)? {
                    Some(user_deps) => {
                        known_dependencies.extend(user_deps);
                    }
                    None => {
                        return Ok(BuildResult::UserCancelled);
                    }
                }
            } else {
                return Ok(BuildResult::FailedAfterRetries {
                    final_errors: errors,
                });
            }
        } else {
            // Filter out already attempted libraries
            let new_suggestions: Vec<String> = suggestions
                .into_iter()
                .filter(|lib| !attempted_libs.contains(lib))
                .collect();

            if new_suggestions.is_empty() {
                info!("All suggestions already attempted");

                if interactive && attempt < MAX_RETRY_ATTEMPTS {
                    match prompt_user_for_dependencies(&errors, lib_name)? {
                        Some(user_deps) => {
                            known_dependencies.extend(user_deps);
                        }
                        None => {
                            return Ok(BuildResult::UserCancelled);
                        }
                    }
                } else {
                    return Ok(BuildResult::FailedAfterRetries {
                        final_errors: errors,
                    });
                }
            } else {
                info!("Adding suggested dependencies: {:?}", new_suggestions);

                // Add to attempted set
                for lib in &new_suggestions {
                    attempted_libs.insert(lib.clone());
                }

                // Add to dependencies list
                known_dependencies.extend(new_suggestions.clone());

                // Update build.rs
                update_build_rs_dependencies(output_dir, lib_name, &known_dependencies)?;
            }
        }

        // Clean build artifacts before retry
        clean_build_artifacts(output_dir)?;
    }

    info!("Max retry attempts reached");

    // One final attempt to get errors
    let build_output = run_cargo_build(output_dir)?;
    let stderr = String::from_utf8_lossy(&build_output.stderr);
    let final_errors = parse_build_errors(&stderr);

    Ok(BuildResult::FailedAfterRetries { final_errors })
}

/// Run cargo build and capture output
fn run_cargo_build(output_dir: &Path) -> Result<std::process::Output> {
    debug!("Running cargo build in {}", output_dir.display());

    Command::new("cargo")
        .arg("build")
        // Don't use --message-format=short as it may suppress linker errors
        .current_dir(output_dir)
        .output()
        .context("Failed to run cargo build")
}

/// Run cargo fix to automatically fix warnings
fn run_cargo_fix(output_dir: &Path) -> Result<bool> {
    debug!("Running cargo fix in {}", output_dir.display());

    let fix_output = Command::new("cargo")
        .arg("fix")
        .arg("--allow-dirty")
        .arg("--allow-staged")
        .current_dir(output_dir)
        .output()
        .context("Failed to run cargo fix")?;

    if fix_output.status.success() {
        let stdout = String::from_utf8_lossy(&fix_output.stdout);
        let stderr = String::from_utf8_lossy(&fix_output.stderr);

        // Check if any fixes were applied
        let fixes_applied = stdout.contains("Fixed")
            || stderr.contains("Fixed")
            || stdout.contains("fixed")
            || stderr.contains("fixed");

        if fixes_applied {
            info!("cargo fix applied automatic fixes");
        } else {
            debug!("cargo fix ran but no fixes were needed");
        }

        Ok(fixes_applied)
    } else {
        let stderr = String::from_utf8_lossy(&fix_output.stderr);
        warn!("cargo fix failed: {}", stderr);
        Ok(false)
    }
}

/// Clean build artifacts
fn clean_build_artifacts(output_dir: &Path) -> Result<()> {
    debug!("Cleaning build artifacts");

    let target_dir = output_dir.join("target");
    if target_dir.exists() {
        fs::remove_dir_all(&target_dir).context("Failed to clean target directory")?;
    }

    Ok(())
}

/// Update build.rs with new dependencies
fn update_build_rs_dependencies(
    output_dir: &Path,
    _lib_name: &str,
    dependencies: &[String],
) -> Result<()> {
    info!("Updating build.rs with dependencies: {:?}", dependencies);

    // Read current build.rs
    let build_rs_path = output_dir.join("build.rs");
    let current_content = fs::read_to_string(&build_rs_path).context("Failed to read build.rs")?;

    // Generate new dependency search code
    let dep_search_code = generate_dependency_search_code(dependencies);

    // Find where to inject the code (after "let mut found_libs = Vec::new();")
    let injection_point = "let mut found_libs = Vec::new();";

    if let Some(pos) = current_content.find(injection_point) {
        let insert_pos = pos + injection_point.len();

        // Check if we already have dependency search code
        let after_injection = &current_content[insert_pos..];

        if after_injection.contains("// Search for detected dependencies") {
            // Replace existing dependency search code
            if let Some(dep_start) = after_injection.find("// Search for detected dependencies")
                && let Some(dep_end) = after_injection[dep_start..]
                    .find("\n    // Try to find the library installation")
                {
                    // Remove old dependency code
                    let before = &current_content[..insert_pos + dep_start];
                    let after = &current_content[insert_pos + dep_start + dep_end..];

                    let new_content = format!("{}{}{}", before, dep_search_code, after);
                    fs::write(&build_rs_path, new_content)
                        .context("Failed to write updated build.rs")?;

                    info!("Updated existing dependency search code");
                    return Ok(());
                }
        }

        // No existing code, inject new
        let before = &current_content[..insert_pos];
        let after = &current_content[insert_pos..];

        let new_content = format!("{}{}{}", before, dep_search_code, after);
        fs::write(&build_rs_path, new_content).context("Failed to write updated build.rs")?;

        info!("Injected new dependency search code");
    } else {
        warn!("Could not find injection point in build.rs");
    }

    Ok(())
}

/// Generate dependency search code for build.rs
fn generate_dependency_search_code(dependencies: &[String]) -> String {
    let mut code = String::from("\n\n    // Search for detected dependencies\n");

    for dep in dependencies {
        code.push_str(&format!(r#"    
    // Search for {} library
    let {}_roots = find_library_roots("{}");
    for dep_root in {}_roots {{
        eprintln!("Found {} libraries at: {{}}", dep_root.display());
        
        #[cfg(target_os = "windows")]
        let dep_paths = vec![
            dep_root.join("lib").join("x64"),
            dep_root.join("lib"),
            dep_root.join("bin"),
        ];
        
        #[cfg(target_os = "linux")]
        let dep_paths = vec![
            dep_root.join("lib64"),
            dep_root.join("lib"),
            dep_root.join("lib").join("x86_64-linux-gnu"),
        ];
        
        #[cfg(target_os = "macos")]
        let dep_paths = vec![
            dep_root.join("lib"),
            dep_root.join("Frameworks"),
        ];
        
        for dep_path in dep_paths {{
            if dep_path.exists() && !search_paths.contains(&dep_path) {{
                search_paths.push(dep_path.clone());
                
                // Scan for library files
                if let Ok(entries) = fs::read_dir(&dep_path) {{
                    for entry in entries.filter_map(|e| e.ok()) {{
                        let path = entry.path();
                        if let Some(name) = path.file_name().and_then(|n| n.to_str()) {{
                            #[cfg(target_os = "windows")]
                            let is_lib = name.ends_with(".lib") || name.ends_with(".dll.a");
                            
                            #[cfg(target_os = "linux")]
                            let is_lib = (name.starts_with("lib") && name.ends_with(".so")) || 
                                         name.ends_with(".a") ||
                                         name.contains(".so.");
                            
                            #[cfg(target_os = "macos")]
                            let is_lib = (name.starts_with("lib") && name.ends_with(".dylib")) || 
                                         name.ends_with(".a") ||
                                         name.ends_with(".framework");
                            
                            if is_lib {{
                                let extracted_name = extract_lib_name(name);
                                if !extracted_name.is_empty() && !found_libs.contains(&extracted_name.to_string()) {{
                                    found_libs.push(extracted_name.to_string());
                                }}
                            }}
                        }}
                    }}
                }}
            }}
        }}
    }}
"#, dep, dep, dep, dep, dep));
    }

    code
}

/// Add library search paths to build.rs based on discovered library locations
fn add_library_paths_to_build_rs(output_dir: &Path, lib_paths: &[PathBuf]) -> Result<()> {
    info!("Adding {} library path(s) to build.rs", lib_paths.len());

    // Read current build.rs
    let build_rs_path = output_dir.join("build.rs");
    let current_content = fs::read_to_string(&build_rs_path).context("Failed to read build.rs")?;

    // Extract the parent directories (lib directories) and library names
    let mut lib_dirs: HashSet<PathBuf> = HashSet::new();
    let mut lib_names: HashSet<String> = HashSet::new();

    for lib_path in lib_paths {
        // Skip paths that look like Rust build artifacts (target directories)
        let path_str = lib_path.to_string_lossy();
        if path_str.contains("\\target\\") || path_str.contains("/target/") {
            info!(
                "⊗ Skipping Rust build artifact path: {}",
                lib_path.display()
            );
            continue;
        }

        // Extract directory
        if let Some(parent) = lib_path.parent() {
            lib_dirs.insert(parent.to_path_buf());
        }

        // Extract library name from filename
        // e.g., "cudnn64_9.lib" -> "cudnn64_9"
        if let Some(file_name) = lib_path.file_name().and_then(|n| n.to_str()) {
            let lib_name = file_name
                .trim_end_matches(".lib")
                .trim_end_matches(".dll.a")
                .trim_start_matches("lib") // Unix libraries start with "lib"
                .trim_end_matches(".so")
                .trim_end_matches(".a")
                .trim_end_matches(".dylib");

            if !lib_name.is_empty() {
                lib_names.insert(lib_name.to_string());
            }
        }
    }

    info!("Extracted {} unique library directories", lib_dirs.len());
    for dir in &lib_dirs {
        info!("  • {}", dir.display());
    }

    info!("Extracted {} unique library names to link", lib_names.len());
    for name in &lib_names {
        info!("  • {}", name);
    }

    // Generate code to add these paths and link directives
    let mut path_code = String::from("\n    // Auto-discovered library paths\n");
    for dir in &lib_dirs {
        let dir_str = dir.to_string_lossy().replace('\\', "\\\\");
        path_code.push_str(&format!(
            r#"    println!("cargo:rustc-link-search=native={}");"#,
            dir_str
        ));
        path_code.push('\n');
    }

    // Add link directives for the discovered libraries
    path_code.push_str("\n    // Auto-discovered library link directives\n");
    for lib_name in &lib_names {
        // Generate conditional code that will run in the build.rs
        path_code.push_str(&format!(
            r#"    #[cfg(target_os = "windows")]
    println!("cargo:rustc-link-lib=dylib={}");
    #[cfg(not(target_os = "windows"))]
    println!("cargo:rustc-link-lib={}");
"#,
            lib_name, lib_name
        ));
    }

    // Emit DEP_* environment variables for downstream crates
    if !lib_dirs.is_empty() {
        path_code.push_str("\n    // Emit environment variables for downstream crates\n");
        // Emit the first library path (usually the most relevant one)
        if let Some(first_lib_dir) = lib_dirs.iter().next() {
            let dir_str = first_lib_dir.to_string_lossy().replace('\\', "\\\\");
            path_code.push_str(&format!(r#"    println!("cargo:lib={}");"#, dir_str));
            path_code.push('\n');

            // Also emit include path if we can find it
            if let Some(parent) = first_lib_dir.parent() {
                let include_path = parent.join("include");
                if include_path.exists() {
                    let include_str = include_path.to_string_lossy().replace('\\', "\\\\");
                    path_code.push_str(&format!(
                        r#"    println!("cargo:include={}");"#,
                        include_str
                    ));
                    path_code.push('\n');
                }
            }
        }
    }

    // Add blank lines to mark the end of the auto-discovered section
    path_code.push_str("\n\n");

    // Find where to inject the code - right after "fn main() {"
    let injection_point = "fn main() {";

    if let Some(pos) = current_content.find(injection_point) {
        let insert_pos = pos + injection_point.len();

        // Check if we already have auto-discovered paths
        if current_content.contains("// Auto-discovered library paths") {
            // Remove old auto-discovered section (everything from the start marker to the next major section)
            if let Some(start) = current_content.find("// Auto-discovered library paths") {
                // Find the end - look for the next major comment that's NOT part of auto-discovered
                let after_start = &current_content[start..];

                // Look for end markers that indicate we're past the auto-discovered section
                // This should find the blank line AFTER the link directives
                let end_offset = if let Some(link_section) =
                    after_start.find("// Auto-discovered library link directives")
                {
                    // We have a link directives section, find its end
                    let after_link = &after_start[link_section..];
                    link_section + after_link.find("\n\n").unwrap_or(after_link.len())
                } else {
                    // No link section, just find the end of paths section
                    after_start
                        .find("\n\n")
                        .or_else(|| after_start.find("\n    //"))
                        .or_else(|| after_start.find("\n    let"))
                        .unwrap_or(after_start.len())
                };

                let before = &current_content[..start];
                let after = &current_content[start + end_offset..];

                let new_content = format!("{}{}{}", before, path_code, after);
                fs::write(&build_rs_path, new_content)
                    .context("Failed to write updated build.rs")?;

                info!("✓ Updated existing auto-discovered library paths");
                return Ok(());
            }
        }

        // No existing code, inject new
        let before = &current_content[..insert_pos];
        let after = &current_content[insert_pos..];

        let new_content = format!("{}{}{}", before, path_code, after);
        fs::write(&build_rs_path, new_content).context("Failed to write updated build.rs")?;

        info!("✓ Injected new library search paths");
    } else {
        warn!("Could not find injection point in build.rs");
        return Err(anyhow::anyhow!("Could not find 'fn main() {{' in build.rs"));
    }

    Ok(())
}

/// Prompt user for missing dependencies
fn prompt_user_for_dependencies(
    errors: &[BuildError],
    lib_name: &str,
) -> Result<Option<Vec<String>>> {
    println!("\n{}", style("Build Failed - Need Your Help!").red().bold());
    println!("{}", style("─".repeat(50)).dim());

    // Show error summary
    let symbol_preview: Vec<String> = errors
        .iter()
        .filter_map(|e| match e {
            BuildError::UnresolvedSymbol { symbol, .. } => Some(symbol.clone()),
            _ => None,
        })
        .take(10)
        .collect();

    if !symbol_preview.is_empty() {
        println!("\n{}", style("Missing symbols (first 10):").yellow());
        for symbol in &symbol_preview {
            println!("  • {}", style(symbol).cyan());
        }

        if errors.len() > 10 {
            println!("  ... and {} more", errors.len() - 10);
        }
    }

    println!("\n{}", style("The generated bindings for").white());
    println!(
        "{}",
        style(format!("'{}' require additional libraries.", lib_name)).white()
    );

    // Ask if user wants to continue
    let should_continue = Confirm::new()
        .with_prompt("Would you like to specify the missing libraries?")
        .default(true)
        .interact()?;

    if !should_continue {
        return Ok(None);
    }

    // Common library suggestions based on patterns
    let common_deps = vec![
        "cudart (CUDA Runtime)",
        "cuda (CUDA Driver API)",
        "cublas (cuBLAS)",
        "cufft (cuFFT)",
        "curand (cuRAND)",
        "cusparse (cuSPARSE)",
        "cusolver (cuSOLVER)",
        "opengl (OpenGL)",
        "vulkan (Vulkan)",
        "Other (I'll type the name)",
    ];

    let selections = MultiSelect::new()
        .with_prompt("Select the libraries your bindings depend on")
        .items(&common_deps)
        .interact()?;

    let mut dependencies = Vec::new();

    for &idx in &selections {
        let selected = common_deps[idx];

        if selected.starts_with("Other") {
            // Ask user to type custom library name
            loop {
                let custom: String = Input::new()
                    .with_prompt("Enter library name (or empty to skip)")
                    .allow_empty(true)
                    .interact_text()?;

                if custom.is_empty() {
                    break;
                }

                dependencies.push(custom.trim().to_string());

                let more = Confirm::new()
                    .with_prompt("Add another library?")
                    .default(false)
                    .interact()?;

                if !more {
                    break;
                }
            }
        } else {
            // Extract library name from selection
            if let Some(name) = selected.split_whitespace().next() {
                dependencies.push(name.to_lowercase());
            }
        }
    }

    if dependencies.is_empty() {
        println!(
            "\n{}",
            style("No libraries specified. Will try again...").yellow()
        );
        return Ok(None);
    }

    println!(
        "\n{}",
        style(format!("Will try adding: {:?}", dependencies)).green()
    );

    Ok(Some(dependencies))
}

/// High-level function to validate and optionally refine build
pub fn validate_and_refine(
    output_dir: &Path,
    lib_name: &str,
    initial_dependencies: &[String],
    enable_refinement: bool,
    interactive: bool,
) -> Result<BuildResult> {
    if !enable_refinement {
        // Just try once
        let build_output = run_cargo_build(output_dir)?;

        if build_output.status.success() {
            // Run cargo fix to clean up warnings automatically
            info!("Running cargo fix to resolve warnings...");
            match run_cargo_fix(output_dir) {
                Ok(fixes_applied) => {
                    if fixes_applied {
                        info!("✓ cargo fix applied automatic improvements");
                    } else {
                        debug!("No automatic fixes needed");
                    }
                }
                Err(e) => {
                    warn!("cargo fix failed (non-critical): {}", e);
                }
            }

            return Ok(BuildResult::Success);
        }

        let stderr = String::from_utf8_lossy(&build_output.stderr);
        let errors = parse_build_errors(&stderr);

        return Ok(BuildResult::FailedAfterRetries {
            final_errors: errors,
        });
    }

    // Use iterative refinement
    iterative_build_and_refine(output_dir, lib_name, initial_dependencies, interactive)
}
