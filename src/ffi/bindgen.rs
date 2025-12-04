use anyhow::{Context, Result};
use bindgen::callbacks::{ItemKind, ParseCallbacks};
use std::path::{Path, PathBuf};
use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};
use std::time::{Duration, Instant};
use tracing::{debug, error, info, warn};

/// Progress monitoring callback for bindgen
#[derive(Debug)]
struct ProgressCallback {
    functions_processed: Arc<AtomicUsize>,
    types_processed: Arc<AtomicUsize>,
    last_update: std::sync::Mutex<Instant>,
}

impl ProgressCallback {
    fn new() -> Self {
        Self {
            functions_processed: Arc::new(AtomicUsize::new(0)),
            types_processed: Arc::new(AtomicUsize::new(0)),
            last_update: std::sync::Mutex::new(Instant::now()),
        }
    }

    fn get_counters(&self) -> (Arc<AtomicUsize>, Arc<AtomicUsize>) {
        (
            self.functions_processed.clone(),
            self.types_processed.clone(),
        )
    }

    fn report_progress(&self) {
        let mut last_update = self.last_update.lock().unwrap();
        let now = Instant::now();

        // Only report progress every 10 seconds to avoid spam
        if now.duration_since(*last_update) > Duration::from_secs(10) {
            let functions = self.functions_processed.load(Ordering::Relaxed);
            let types = self.types_processed.load(Ordering::Relaxed);

            // Use eprint! for progress to avoid cluttering logs
            eprint!(
                "\r⚡ Processing: {} functions, {} types...",
                functions, types
            );
            std::io::Write::flush(&mut std::io::stderr()).ok();
            *last_update = now;
        }
    }
}

impl ParseCallbacks for ProgressCallback {
    fn include_file(&self, filename: &str) {
        // Called when a file is included
        debug!("Processing file: {}", filename);
    }

    fn item_name(&self, info: bindgen::callbacks::ItemInfo<'_>) -> Option<String> {
        // Called for each item bindgen processes - track ALL items
        let name = info.name;

        // Skip compiler-generated items but track all library items
        if !name.starts_with("_bindgen_") && !name.starts_with("__") {
            match info.kind {
                ItemKind::Function => {
                    self.functions_processed.fetch_add(1, Ordering::Relaxed);
                }
                _ => {
                    self.types_processed.fetch_add(1, Ordering::Relaxed);
                }
            }
            self.report_progress();
        }

        None // Don't modify the name
    }

    // Note: Additional diagnostic suppression methods removed due to bindgen version compatibility

    // Override to suppress verbose diagnostic output
    fn generated_name_override(&self, _info: bindgen::callbacks::ItemInfo<'_>) -> Option<String> {
        None // Don't override names, just suppress messages
    }
}

/// Quick estimation of header complexity by counting function/type declarations
fn estimate_header_complexity(header_path: &Path) -> Result<usize> {
    let content = std::fs::read_to_string(header_path)
        .context("Failed to read header for complexity estimation")?;

    let mut count = 0;

    // Count function declarations (rough heuristic)
    for line in content.lines() {
        let line = line.trim();
        if line.starts_with("CUDNN_DEPRECATED")
            || line.starts_with("cudnn")
            || line.starts_with("CUDA")
            || line.starts_with("cuda")
            || line.contains("typedef")
            || line.contains("enum")
            || line.contains("struct")
        {
            count += 1;
        }
    }

    // Also scan included headers briefly
    if count < 100 {
        // This might be a header that includes the real headers
        let lines: Vec<&str> = content.lines().collect();
        let include_count = lines
            .iter()
            .filter(|line| line.trim().starts_with("#include"))
            .count();
        if include_count > 5 {
            // Estimate based on typical CUDA header size
            count = include_count * 500; // Rough estimate
        }
    }

    Ok(count.max(100)) // Minimum estimate
}

/// Generate raw FFI bindings using bindgen with timeout and progress monitoring
pub fn generate_bindings(
    headers: &[PathBuf],
    lib_name: &str,
    source_path: &Path,
    config: &crate::Config,
) -> Result<String> {
    // Get timeout from config (default 5 minutes for large headers like CUDA)
    let bindgen_timeout = Duration::from_secs(config.bindgen_timeout_secs);

    if headers.is_empty() {
        anyhow::bail!("No header files provided for bindgen");
    }

    let main_header = &headers[0];
    debug!("Using main header: {}", main_header.display());

    let start_time = Instant::now();

    // Clear bindgen debug environment variables that might cause verbose output
    unsafe {
        std::env::remove_var("BINDGEN_DEBUG");
        std::env::remove_var("BINDGEN_DUMP_AST");
        std::env::remove_var("BINDGEN_TRACE");
        // Keep BINDGEN_EXTRA_CLANG_ARGS - user might need it for includes
    }

    // Create progress monitoring callback
    let progress_callback = ProgressCallback::new();
    let (functions_counter, types_counter) = progress_callback.get_counters();

    let mut builder = bindgen::Builder::default()
        .header(main_header.to_str().context("Invalid header path")?)
        // Temporarily disable callbacks to see if they're causing AST dump
        // .parse_callbacks(Box::new(progress_callback))
        // Performance optimizations
        .layout_tests(false) // Skip layout tests for faster generation
        .generate_inline_functions(false) // Skip inline functions for speed
        .size_t_is_usize(true) // Faster size_t handling
        // Ensure quiet operation - no debug dumps
        .record_matches(false) // Disable match recording that might cause verbose output
        // Essential features only for speed
        .generate_comments(true) // Keep comments for documentation
        .use_core() // Use core instead of std
        .derive_debug(true) // Enable Debug for builder compatibility
        .derive_default(false) // Skip Default to speed up generation
        // Prepend the library name as module
        .module_raw_line("ffi", format!("//! Raw FFI bindings for {}", lib_name))
        // Make opaque types for faster processing
        .opaque_type(".*_impl")
        .opaque_type(".*_internal")
        // Add performance-oriented clang args
        .clang_arg("-fparse-all-comments") // Ensure we get documentation
        .clang_arg("-w") // Suppress clang warnings for speed
        .clang_arg("-Wno-everything") // Suppress ALL warnings
        // Additional flags to suppress verbose output
        .clang_arg("-fsyntax-only") // Only check syntax, don't generate anything extra
        .clang_arg("-fno-diagnostics-show-option") // Don't show diagnostic options
        ; // Remove deprecated rust_target setting

    // Add include paths
    let include_dir = source_path.join("include");
    if include_dir.exists() {
        debug!("Adding include directory: {}", include_dir.display());
        builder = builder.clang_arg(format!("-I{}", include_dir.display()));
    }

    // Add the source directory itself
    builder = builder.clang_arg(format!("-I{}", source_path.display()));

    // Add user-specified include directories for dependencies
    for include_path in &config.include_dirs {
        if include_path.exists() {
            debug!("Adding user include directory: {}", include_path.display());
            builder = builder.clang_arg(format!("-I{}", include_path.display()));
        } else {
            debug!(
                "Warning: include directory does not exist: {}",
                include_path.display()
            );
        }
    }

    // Add library search paths as clang args (for header resolution)
    for lib_path in &config.lib_paths {
        if lib_path.exists() {
            debug!("Adding library search path: {}", lib_path.display());
            builder = builder.clang_arg(format!("-L{}", lib_path.display()));
        } else {
            debug!(
                "Warning: library path does not exist: {}",
                lib_path.display()
            );
        }
    }

    // Pre-scan header to estimate work
    info!("Scanning header to estimate processing requirements...");
    let estimated_functions = estimate_header_complexity(main_header)?;

    // Monitor progress with periodic status updates
    info!(
        "Running bindgen on ~{} estimated items (this may take several minutes for large libraries like CUDA)...",
        estimated_functions
    );

    // Build and generate the bindings with progress monitoring and timeout
    info!(
        "Starting bindgen generation (timeout configured for {} seconds)...",
        bindgen_timeout.as_secs()
    );

    // Create a watchdog thread to monitor for hangs
    let watchdog_start = Arc::new(Instant::now());
    let watchdog_running = Arc::new(AtomicBool::new(true));

    let watchdog_start_clone = Arc::clone(&watchdog_start);
    let watchdog_running_clone = Arc::clone(&watchdog_running);
    let timeout_clone = bindgen_timeout;

    let watchdog = std::thread::spawn(move || {
        let mut last_warning = Duration::ZERO;
        loop {
            std::thread::sleep(Duration::from_secs(30)); // Check every 30 seconds

            if !watchdog_running_clone.load(Ordering::Relaxed) {
                break;
            }

            let elapsed = watchdog_start_clone.elapsed();

            // Warn every minute after 2 minutes
            if elapsed > Duration::from_secs(120)
                && elapsed - last_warning > Duration::from_secs(60)
            {
                warn!(
                    "Bindgen still processing after {:.1} minutes (timeout at {} minutes)...",
                    elapsed.as_secs_f64() / 60.0,
                    timeout_clone.as_secs() / 60
                );
                last_warning = elapsed;
            }

            // Check for timeout
            if elapsed > timeout_clone {
                error!(
                    "TIMEOUT: Bindgen exceeded {} minute timeout. This likely indicates:\n\
                     1. Headers are extremely complex (thousands of types/functions)\n\
                     2. Bindgen encountered infinite recursion in type resolution\n\
                     3. System is low on resources\n\
                     \n\
                     Consider: simplifying headers, adding --opaque-type flags, or increasing timeout.",
                    timeout_clone.as_secs() / 60
                );
                std::process::exit(1); // Force exit on timeout
            }
        }
    });

    let bindings = builder
        .generate()
        .context("Failed to generate bindings with bindgen")?;
    
    // Stop watchdog
    watchdog_running.store(false, Ordering::Relaxed);
    let _ = watchdog.join();

    let elapsed = start_time.elapsed();

    // Get final progress counts
    let final_functions = functions_counter.load(Ordering::Relaxed);
    let final_types = types_counter.load(Ordering::Relaxed);

    info!("bindgen completed in {:.2} seconds", elapsed.as_secs_f64());
    info!(
        "Processed {} functions and {} types",
        final_functions, final_types
    );

    // Convert to string
    let bindings_str = bindings.to_string();

    info!(
        "Successfully generated {} bytes of FFI bindings",
        bindings_str.len()
    );

    Ok(bindings_str)
}
