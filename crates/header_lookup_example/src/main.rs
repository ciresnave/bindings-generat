use anyhow::{Context, Result};
use library_db::{LibraryDb, LibraryEntry};
use serde_json;
use std::collections::HashSet;

fn print_usage(program: &str) {
    eprintln!("Usage: {} <header.h> [db.json]\n", program);
}

fn main() -> Result<()> {
    let mut args: Vec<String> = std::env::args().collect();

    // Verbosity flags: `--full` prefers full path (e.g. `std::os::raw::c_char`),
    // `--short` prefers the base name (e.g. `c_char`). Remove them from
    // `args` so they don't interfere with header/db/clang-arg parsing below.
    let mut prefer_full: bool = true;
    if let Some(pos) = args.iter().position(|a| a == "--full") {
        prefer_full = true;
        args.remove(pos);
    }
    if let Some(pos) = args.iter().position(|a| a == "--short") {
        prefer_full = false;
        args.remove(pos);
    }

    if args.len() < 2 {
        print_usage(&args[0]);
        return Ok(());
    }

    let header = &args[1];

    // optional second arg is db path unless it looks like a clang arg (starts with `-`)
    let mut db_path: Option<&String> = None;
    let mut clang_args: Vec<String> = Vec::new();
    if args.len() >= 3 {
        if args[2].starts_with('-') {
            // everything from args[2].. are clang args
            clang_args = args[2..].to_vec();
        } else {
            db_path = Some(&args[2]);
            if args.len() > 3 {
                clang_args = args[3..].to_vec();
            }
        }
    }

    if !std::path::Path::new(header).exists() {
        eprintln!("Header '{}' not found", header);
        std::process::exit(2);
    }

    println!("Analyzing header: {}", header);
    let ffi = c_header_analyzer::analyze_headers(&[header.as_str()], &clang_args)
        .context("failed to analyze headers; ensure libclang is available for bindgen and clang args are correct")?;

    // Print raw parsed structure as JSON for inspection
    println!(
        "\n--- Parsed FFI info (JSON) ---\n{}
",
        serde_json::to_string_pretty(&ffi)?
    );

    println!(
        "Discovered {} functions, {} constants",
        ffi.functions.len(),
        ffi.constants.len()
    );

    // Also print a concise human-readable listing of parsed items
    println!("\n--- Parsed Items ---");
    if !ffi.functions.is_empty() {
        println!("Functions:");
        for f in &ffi.functions {
            if f.name.trim().is_empty() || f.name == "_" {
                continue;
            }

            let ret = if let Some(rs) = &f.return_spec {
                rs.to_rust_string(prefer_full)
            } else {
                f.return_type.clone()
            };

            println!("- {} -> {}", f.name, ret);
            if let Some(docs) = &f.docs {
                println!("    docs: {}", docs.trim());
            }
            for p in &f.params {
                let ty_str = p.ty_spec.to_rust_string(prefer_full);
                println!("    param: {}: {}", p.name, ty_str);
            }
        }
    }
    if !ffi.constants.is_empty() {
        println!("Constants:");
        for c in &ffi.constants {
            if c.name.trim().is_empty() || c.name == "_" {
                continue;
            }
            let ty = c.ty_spec.to_rust_string(prefer_full);
            println!("- {} = {} : {}", c.name, c.value, ty);
        }
    }

    // collect symbol names we will query the DB with
    let mut symbols: Vec<String> = ffi
        .functions
        .iter()
        .filter(|f| !f.name.trim().is_empty() && f.name != "_")
        .map(|f| f.name.clone())
        .collect();
    for c in &ffi.constants {
        if c.name.trim().is_empty() || c.name == "_" {
            continue;
        }
        symbols.push(c.name.clone());
    }

    if symbols.is_empty() {
        println!("No symbols discovered in header.");
        return Ok(());
    }

    // load or build a small demo DB
    let db: LibraryDb = if let Some(p) = db_path {
        println!("Loading library DB from {}", p);
        LibraryDb::load_from_file(p).context("loading library DB")?
    } else {
        println!("Using built-in demo library DB");
        let mut db = LibraryDb::new();
        db.entries.push(LibraryEntry {
            name: "cuda".to_string(),
            description: Some("NVIDIA CUDA runtime".to_string()),
            symbols: vec![
                "cudaMalloc".to_string(),
                "cudaFree".to_string(),
                "cudaMemcpy".to_string(),
                "cudaGetErrorString".to_string(),
            ],
            homepage: Some("https://developer.nvidia.com/cuda".to_string()),
        });
        db.entries.push(LibraryEntry {
            name: "cudnn".to_string(),
            description: Some("NVIDIA cuDNN".to_string()),
            symbols: vec![
                "cudnnCreate".to_string(),
                "cudnnDestroy".to_string(),
                "cudnnGetErrorString".to_string(),
            ],
            homepage: Some("https://developer.nvidia.com/cudnn".to_string()),
        });
        db.entries.push(LibraryEntry {
            name: "cublas".to_string(),
            description: Some("NVIDIA cuBLAS".to_string()),
            symbols: vec![
                "cublasCreate".to_string(),
                "cublasDestroy".to_string(),
                "cublasSgemm".to_string(),
            ],
            homepage: None,
        });
        db
    };

    // ask the DB for suggestions
    let suggestions = db.suggest_libraries_for_symbols(&symbols);

    if suggestions.is_empty() {
        println!("No library candidates found for discovered symbols.");
        return Ok(());
    }

    println!("\nLibrary suggestions:");
    let mut matched_all: HashSet<String> = HashSet::new();
    for (entry, matched) in &suggestions {
        println!(
            "- {} ({} matches){}",
            entry.name,
            matched.len(),
            entry
                .description
                .as_ref()
                .map(|d| format!(" — {}", d))
                .unwrap_or_default()
        );
        for s in matched {
            println!("    {}", s);
            matched_all.insert(s.clone());
        }
    }

    let unmatched: Vec<String> = symbols
        .into_iter()
        .filter(|s| !matched_all.contains(s))
        .collect();
    if !unmatched.is_empty() {
        println!("\nUnmatched symbols (no DB entry contained them):");
        for s in unmatched {
            println!("  {}", s);
        }
    }

    Ok(())
}
