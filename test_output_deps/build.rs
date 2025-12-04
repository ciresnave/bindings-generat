use std::env;
use std::path::PathBuf;
use std::fs;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    // User-specified dependencies
    println!("cargo:rustc-link-search=native=C:\MyLibs\lib");
    println!("cargo:rustc-link-lib=mydep1");
    println!("cargo:rustc-link-lib=mydep2");

    
    // Get the library name we're wrapping
    let lib_name = "simple";
    let lib_name_upper = lib_name.to_uppercase();
    
    // Watch for environment variable changes
    println!("cargo:rerun-if-env-changed={}_PATH", lib_name_upper);
    println!("cargo:rerun-if-env-changed={}_ROOT", lib_name_upper);
    println!("cargo:rerun-if-env-changed=LIBRARY_PATH");
    println!("cargo:rerun-if-env-changed=LD_LIBRARY_PATH");
    
    let mut search_paths = Vec::new();
    let mut found_libs = Vec::new();

    // Try to find the library installation
    let lib_roots = find_library_roots(lib_name);
    
    if lib_roots.is_empty() {
        eprintln!("Warning: {} installation not found.", lib_name);
        eprintln!("Set {}_PATH or {}_ROOT environment variable to the installation directory.", 
                  lib_name_upper, lib_name_upper);
        eprintln!("Or ensure the library is in a standard system location.");
    }
    
    // Scan each root directory for libraries
    for lib_root in lib_roots {
        eprintln!("Found {} libraries at: {}", lib_name, lib_root.display());
        
        // Determine platform-specific lib directories
        #[cfg(target_os = "windows")]
        let lib_paths = vec![
            lib_root.join("lib").join("x64"),
            lib_root.join("lib"),
            lib_root.join("bin"),
        ];
        
        #[cfg(target_os = "linux")]
        let lib_paths = vec![
            lib_root.join("lib64"),
            lib_root.join("lib"),
            lib_root.join("lib").join("x86_64-linux-gnu"),
        ];
        
        #[cfg(target_os = "macos")]
        let lib_paths = vec![
            lib_root.join("lib"),
            lib_root.join("Frameworks"),
        ];
        
        for lib_path in lib_paths {
            if lib_path.exists() {
                if !search_paths.contains(&lib_path) {
                    search_paths.push(lib_path.clone());
                }
                
                // Scan for library files
                if let Ok(entries) = fs::read_dir(&lib_path) {
                    for entry in entries.filter_map(|e| e.ok()) {
                        let path = entry.path();
                        if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                            // Detect library files by extension
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
                            
                            if is_lib {
                                // Extract library name
                                let lib_name = extract_lib_name(name);
                                
                                if !lib_name.is_empty() && !found_libs.contains(&lib_name.to_string()) {
                                    found_libs.push(lib_name.to_string());
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    
    // Output all search paths
    for path in &search_paths {
        println!("cargo:rustc-link-search=native={}", path.display());
    }
    
    // Link discovered libraries
    if !found_libs.is_empty() {
        // Sort to ensure main library is linked first if present
        found_libs.sort_by_key(|name| {
            if name == "simple" {
                0  // Main library first
            } else if name.starts_with("simple") {
                1  // Related libraries second
            } else {
                2  // Other libraries last
            }
        });
        
        for lib in &found_libs {
            #[cfg(target_os = "windows")]
            println!("cargo:rustc-link-lib=dylib={}", lib);
            
            #[cfg(not(target_os = "windows"))]
            println!("cargo:rustc-link-lib={}", lib);
            
            eprintln!("Linking library: {}", lib);
        }
    } else {
        eprintln!("Warning: No library files found. Linking will likely fail.");
        eprintln!("Please ensure {} and its dependencies are properly installed.", "simple");
    }
}

/// Extract library name from filename
fn extract_lib_name(filename: &str) -> String {
    #[cfg(target_os = "windows")]
    {
        // Windows: remove .lib or .dll.a extension
        filename
            .trim_end_matches(".lib")
            .trim_end_matches(".dll.a")
            .to_string()
    }
    
    #[cfg(target_os = "linux")]
    {
        // Linux: remove lib prefix and .so* or .a extension
        let name = filename.trim_start_matches("lib");
        // Handle versioned .so files: libfoo.so.1.2.3 -> foo
        if let Some(so_pos) = name.find(".so") {
            name[..so_pos].to_string()
        } else {
            name.trim_end_matches(".a").to_string()
        }
    }
    
    #[cfg(target_os = "macos")]
    {
        // macOS: remove lib prefix and .dylib or .a extension
        filename
            .trim_start_matches("lib")
            .trim_end_matches(".dylib")
            .trim_end_matches(".a")
            .trim_end_matches(".framework")
            .to_string()
    }
}

/// Find potential library installation directories
fn find_library_roots(lib_name: &str) -> Vec<PathBuf> {
    let lib_name_upper = lib_name.to_uppercase();
    let mut roots = Vec::new();
    
    // 1. Try LIB_NAME_PATH environment variable
    if let Ok(path) = env::var(format!("{}_PATH", lib_name_upper)) {
        let p = PathBuf::from(path);
        if p.exists() {
            roots.push(p);
        }
    }
    
    // 2. Try LIB_NAME_ROOT environment variable
    if let Ok(path) = env::var(format!("{}_ROOT", lib_name_upper)) {
        let p = PathBuf::from(path);
        if p.exists() {
            roots.push(p);
        }
    }
    
    // 3. Check user home directory for ~/.lib_name/
    if let Ok(home) = env::var("HOME").or_else(|_| env::var("USERPROFILE")) {
        let home_path = PathBuf::from(home);
        let dot_lib = home_path.join(format!(".{}", lib_name));
        
        if dot_lib.exists() {
            // If versioned subdirectories exist, find the latest
            if let Ok(entries) = fs::read_dir(&dot_lib) {
                let mut versions: Vec<_> = entries
                    .filter_map(|e| e.ok())
                    .filter(|e| e.path().is_dir())
                    .collect();
                
                versions.sort_by_key(|e| e.path());
                
                if let Some(latest) = versions.last() {
                    roots.push(latest.path());
                }
            }
            
            // Also add the base directory
            roots.push(dot_lib);
        }
    }
    
    // 4. Check standard system locations
    #[cfg(target_os = "windows")]
    {
        if let Ok(program_files) = env::var("ProgramFiles") {
            let paths = vec![
                PathBuf::from(&program_files).join(lib_name),
                PathBuf::from(&program_files).join(lib_name.to_uppercase()),
            ];
            
            for path in paths {
                if path.exists() {
                    roots.push(path);
                }
            }
        }
    }
    
    #[cfg(target_os = "linux")]
    {
        let paths = vec![
            PathBuf::from(format!("/usr/local/{}", lib_name)),
            PathBuf::from(format!("/opt/{}", lib_name)),
            PathBuf::from(format!("/usr/lib/{}", lib_name)),
            PathBuf::from("/usr/local"),
            PathBuf::from("/usr"),
        ];
        
        for path in paths {
            if path.exists() {
                roots.push(path);
            }
        }
    }
    
    #[cfg(target_os = "macos")]
    {
        let paths = vec![
            PathBuf::from(format!("/usr/local/{}", lib_name)),
            PathBuf::from(format!("/opt/{}", lib_name)),
            PathBuf::from("/usr/local"),
            PathBuf::from(format!("/Library/Frameworks/{}.framework", lib_name)),
        ];
        
        for path in paths {
            if path.exists() {
                roots.push(path);
            }
        }
    }
    
    // Deduplicate roots
    roots.sort();
    roots.dedup();
    
    roots
}
