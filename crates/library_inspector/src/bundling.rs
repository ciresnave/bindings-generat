use anyhow::Result;
use std::fs;
use std::path::Path;

/// Simple heuristic license check for whether bundling is allowed.
///
/// This is intentionally conservative: it returns `true` for common permissive
/// licenses and `false` otherwise. Callers should perform a real license audit
/// for production use.
pub fn license_allows_bundling(license_identifier: &str) -> bool {
    let id = license_identifier.to_ascii_lowercase();
    matches!(
        id.as_str(),
        "mit" | "apache-2.0" | "apache-2" | "bsd-3-clause" | "bsd-2-clause" | "zlib" | "isc"
    )
}

/// Bundle a library file into `dest_dir`. Performs a conservative license check
/// using `license_identifier` and copies the file into the destination.
pub fn bundle_library<P: AsRef<Path>, Q: AsRef<Path>>(
    src: P,
    dest_dir: Q,
    license_identifier: &str,
) -> Result<()> {
    if !license_allows_bundling(license_identifier) {
        anyhow::bail!("license '{}' does not permit bundling by default", license_identifier);
    }
    let src = src.as_ref();
    let dest_dir = dest_dir.as_ref();
    fs::create_dir_all(dest_dir)?;
    let filename = src
        .file_name()
        .ok_or_else(|| anyhow::anyhow!("invalid source path"))?;
    fs::copy(src, dest_dir.join(filename))?;
    Ok(())
}
