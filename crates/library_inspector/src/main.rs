use anyhow::Context;
use clap::Parser;
use std::path::PathBuf;

/// Inspect a library file (local path or remote URL) and optionally compare to
/// an FFI JSON file produced by `c_header_analyzer`.
#[derive(Parser, Debug)]
#[command(author, version, about)]
struct Args {
    /// Local path to library file
    #[arg(long)]
    path: Option<PathBuf>,

    /// Remote URL to download (http/https)
    #[arg(long)]
    url: Option<String>,

    /// Optional FFI JSON (as produced by `c_header_analyzer::analyze_headers` + serde)
    #[arg(long)]
    ffi: Option<PathBuf>,

    /// Write output JSON to this file (inspection + optional coverage)
    #[arg(long)]
    out: Option<PathBuf>,

    /// Optional library name hint for generated DB entry
    #[arg(long)]
    name: Option<String>,

    /// Emit a `library_db::LibraryEntry` JSON object to `--out` (or stdout)
    #[arg(long)]
    emit_db_entry: bool,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    let local_path = if let Some(url) = args.url.as_ref() {
        println!("Downloading {} ...", url);
        let (p, _sum) = library_inspector::download_and_cache(url).context("download failed")?;
        p
    } else if let Some(p) = args.path.as_ref() {
        p.clone()
    } else {
        anyhow::bail!("either --path or --url must be provided");
    };

    println!("Inspecting: {}", local_path.display());
    let inspection = library_inspector::inspect_library(&local_path).context("inspect failed")?;

    // Optionally compare to FFI JSON
    let coverage = if let Some(ffi_path) = args.ffi.as_ref() {
        let json = std::fs::read_to_string(ffi_path).context("reading ffi json")?;
        let ffi: c_header_analyzer::FfiInfo = serde_json::from_str(&json).context("parse ffi json")?;
        Some(library_inspector::compare_with_ffi(&inspection, &ffi))
    } else {
        None
    };

    // Build output structure
    let mut output = serde_json::json!({
        "inspection": inspection,
    });
    if let Some(cov) = coverage {
        output["coverage"] = serde_json::to_value(&cov)?;
    }

    if args.emit_db_entry {
        let entry = library_inspector::library_entry_from_inspection(&inspection, args.name.as_deref(), None);
        if let Some(outp) = args.out.as_ref() {
            std::fs::write(outp, serde_json::to_string_pretty(&entry)?)?;
            println!("Wrote library_db entry to {}", outp.display());
        } else {
            println!("Library DB entry:\n{}", serde_json::to_string_pretty(&entry)?);
        }
        return Ok(());
    }

    if let Some(outp) = args.out.as_ref() {
        std::fs::write(outp, serde_json::to_string_pretty(&output)?)?;
        println!("Wrote output to {}", outp.display());
    } else {
        println!("{}", serde_json::to_string_pretty(&output)?);
    }

    Ok(())
}
