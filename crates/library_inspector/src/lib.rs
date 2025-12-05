use anyhow::Result;
use cpp_demangle::Symbol as CppSymbol;
use goblin::Object as GoblinObject;
use object::Object;
use object::ObjectSection;
use object::ObjectSymbol;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::process::Command;
use undname::Flags as UndnameFlags;

mod loader;
mod bundling;

pub use loader::Loader;
pub use bundling::{bundle_library, license_allows_bundling};

use object::SymbolKind;

/// Per-export metadata discovered in a library/binary.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportInfo {
    pub raw_name: String,
    pub demangled: Option<String>,
    pub normalized: String,
    /// kind: "func" | "data" | "unknown"
    pub kind: Option<String>,
    /// ordinal when available (PE exports)
    pub ordinal: Option<u64>,
    /// forwarded target like "OTHER.DLL.FuncName"
    pub forwarded_to: Option<String>,
    /// calling convention hint e.g. "stdcall", "cdecl", "system", "unknown"
    pub calling_convention_hint: Option<String>,
    /// Candidate name variants to try (ordered)
    pub candidates: Vec<String>,
    /// Confidence score 0.0..1.0 about mapping to header-level name
    pub confidence: f32,
}

/// Basic inspection result for a library file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LibraryInspection {
    pub path: PathBuf,
    pub checksum: String,
    pub format: String,
    pub arch: String,
    /// file_type: "shared_lib" | "import_lib" | "static_lib" | "archive" | "unknown"
    pub file_type: String,
    /// ELF SONAME when present
    pub soname: Option<String>,
    /// Referenced DLLs (from import descriptors / heuristics)
    pub referenced_dlls: Vec<String>,
    /// Per-export rich metadata
    pub exports: Vec<ExportInfo>,
    /// Backwards-compatible simple symbol list (normalized)
    pub symbols: Vec<String>,
}

/// Coverage report comparing a header's requested symbols against a library's exports.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoverageReport {
    pub expected_count: usize,
    pub matched_count: usize,
    pub percent: f64,
    pub matched: Vec<String>,
    pub missing: Vec<String>,
}
/// Per-export metadata discovered in a library/binary.

/// Compute sha256 hex of bytes
fn sha256_hex(data: &[u8]) -> String {
    use sha2::Digest;
    let mut hasher = sha2::Sha256::new();
    hasher.update(data);
    let sum = hasher.finalize();
    hex::encode(sum)
}

/// Detect simple container format from magic bytes
fn detect_format(data: &[u8]) -> String {
    if data.len() >= 4 && &data[0..4] == b"\x7fELF" {
        "elf".to_string()
    } else if data.len() >= 2 && &data[0..2] == b"MZ" {
        "pe".to_string()
    } else if data.len() >= 8 && &data[0..8] == b"!<arch>\n" {
        "ar".to_string()
    } else if data.len() >= 4
        && (&data[0..4] == [0xca, 0xfe, 0xba, 0xbe] || &data[0..4] == [0xfe, 0xed, 0xfa, 0xcf])
    {
        "mach".to_string()
    } else {
        "unknown".to_string()
    }
}

/// Normalize a raw exported symbol name into a likely header-level identifier.
///
/// - Strips a single leading underscore where appropriate
/// - Removes stdcall "@N" decorations
/// - Attempts C++ demangling via `cpp_demangle`
fn normalize_symbol(s: &str) -> String {
    // quick noise filtering on raw symbol text
    let raw = s;
    let low = raw.to_ascii_lowercase();
    if low.starts_with(".idata")
        || low.starts_with(".rsrc")
        || low.starts_with("import_descriptor_")
        || low.contains("null_import_descriptor")
        || low.contains("_null_thunk_data")
        || low.starts_with("@comp.id")
    {
        return String::new();
    }

    let mut name = raw.to_string();

    // strip a single leading underscore for C-style exports (but keep MSVC '?' mangled names)
    if name.starts_with('_') && !name.starts_with('?') && !name.starts_with("??") {
        name = name.trim_start_matches('_').to_string();
    }

    // strip stdcall decoration like `Func@8`
    if let Some(idx) = name.rfind('@') {
        let tail = &name[idx + 1..];
        if !tail.is_empty() && tail.chars().all(|c| c.is_ascii_digit()) {
            name.truncate(idx);
        }
    }

    // Try to demangle C++/MSVC names — if demangling succeeds, simplify the demangled
    // Try MSVC demangling for names starting with '?' which is common for MSVC
    if name.starts_with('?') || name.starts_with("??") {
        if let Ok(demangled) = undname::demangle(&name, UndnameFlags::default()) {
            let dem = demangled.trim();
            if let Some(paren_idx) = dem.find('(') {
                let before = &dem[..paren_idx].trim();
                if let Some(last_space) = before.rfind(' ') {
                    return before[last_space + 1..].to_string();
                } else {
                    return before.to_string();
                }
            }
            return dem.to_string();
        }
    }

    // Try to demangle Itanium-style / gcc/clang C++ mangled names
    if let Ok(sym) = CppSymbol::new(&name) {
        if let Ok(demangled) = sym.demangle() {
            // demangled might include return type and parameter list; strip parameters
            let dem = demangled.trim();
            if let Some(paren_idx) = dem.find('(') {
                // substring before '('
                let before = &dem[..paren_idx].trim();
                // if there's a return type, take the last token after whitespace
                if let Some(last_space) = before.rfind(' ') {
                    return before[last_space + 1..].to_string();
                } else {
                    return before.to_string();
                }
            }

            return dem.to_string();
        }
    }

    name
}

/// Very small AR archive member parser. Yields (name, bytes) for each member.
fn parse_ar_members<'a>(data: &'a [u8]) -> Vec<(String, &'a [u8])> {
    let mut out = Vec::new();
    if data.len() < 8 || &data[0..8] != b"!<arch>\n" {
        return out;
    }

    let mut off: usize = 8;
    while off + 60 <= data.len() {
        let header = &data[off..off + 60];
        // name is bytes 0..16
        let name_raw = &header[0..16];
        let name = match std::str::from_utf8(name_raw) {
            Ok(s) => s.trim().trim_end_matches('/').to_string(),
            Err(_) => String::new(),
        };

        // size is bytes 48..58 (10 bytes)
        let size_raw = &header[48..58];
        let size_str = match std::str::from_utf8(size_raw) {
            Ok(s) => s.trim(),
            Err(_) => "0",
        };

        let size = size_str.parse::<usize>().unwrap_or(0);
        let start = off + 60;
        if start + size > data.len() {
            break;
        }

        let member_bytes = &data[start..start + size];
        out.push((name, member_bytes));

        let mut next = start + size;
        // entries are 2-byte aligned
        if next % 2 == 1 {
            next += 1;
        }
        off = next;
    }

    out
}

/// Find printable ASCII zero-terminated or space-separated strings in `data` that end with `suffix` (case-insensitive).
fn find_ascii_strings_with_suffix(data: &[u8], suffix: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut buf = Vec::new();
    for &b in data.iter() {
        if (0x20..=0x7e).contains(&b) {
            buf.push(b);
        } else {
            if !buf.is_empty() {
                if let Ok(s) = std::str::from_utf8(&buf) {
                    if s.to_ascii_lowercase()
                        .ends_with(&suffix.to_ascii_lowercase())
                    {
                        out.push(s.to_string());
                    }
                }
                buf.clear();
            }
        }
    }
    // tail
    if !buf.is_empty() {
        if let Ok(s) = std::str::from_utf8(&buf) {
            if s.to_ascii_lowercase()
                .ends_with(&suffix.to_ascii_lowercase())
            {
                out.push(s.to_string());
            }
        }
    }
    out
}

/// Normalize a DLL hint string: remove control / padding characters, surrounding
/// quotes and whitespace, and trim trailing slashes. Leaves base names like
/// `cudnn64_9` and full names like `libpng16-16.dll` intact.
fn normalize_dll_name(s: &str) -> String {
    // Trim outer whitespace and quotes first
    let mut t = s
        .trim_matches(|c: char| c.is_whitespace() || c == '"' || c == '\'')
        .to_string();

    // Remove control characters (including DEL) that sometimes appear as padding
    t = t.chars().filter(|c| !c.is_control()).collect();

    // Trim again and remove any trailing path separators
    let t = t
        .trim()
        .trim_end_matches(|c| c == '/' || c == '\\')
        .to_string();

    // Finally, collapse internal repeated spaces and return
    let collapsed = t.split_whitespace().collect::<Vec<_>>().join(" ");
    collapsed
}

/// Convert a goblin `Reexport` debug representation into a compact readable string when possible.
fn reexport_readable<T: std::fmt::Debug>(r: &Option<T>) -> Option<String> {
    if let Some(v) = r {
        let s = format!("{:?}", v);
        // try to extract first quoted substring
        if let Some(first_quote) = s.find('"') {
            if let Some(second_quote_rel) = s[first_quote + 1..].find('"') {
                let extracted = &s[first_quote + 1..first_quote + 1 + second_quote_rel];
                return Some(extracted.to_string());
            }
        }
        // fallback to the plain debug string
        Some(s)
    } else {
        None
    }
}

/// Parse a potential COFF object member (from a .lib archive) and
/// extract an optional DLL hint plus any imported symbol names present
/// in the object's symbol table or sections.
fn parse_coff_import_member(data: &[u8]) -> Option<(Option<String>, Vec<(String, String)>)> {
    if let Ok(obj) = object::File::parse(data) {
        let mut dll_hint: Option<String> = None;
        // pairs of (raw_name, normalized_name)
        let mut syms: Vec<(String, String)> = Vec::new();

        // Inspect symbol table for import-related symbols
        for sym in obj.symbols() {
            if let Ok(name) = sym.name() {
                if name.is_empty() {
                    continue;
                }

                let lname = name.to_ascii_lowercase();

                // IMPORT_DESCRIPTOR_<dll>
                if lname.starts_with("import_descriptor_") || lname.contains("import_descriptor_") {
                    if let Some(pos) = name.find("IMPORT_DESCRIPTOR_") {
                        let tail = &name[pos + "IMPORT_DESCRIPTOR_".len()..];
                        let tail_trim = tail.trim_start_matches('_').trim_matches('"').to_string();
                        let n = normalize_dll_name(&tail_trim);
                        if !n.is_empty() {
                            dll_hint = Some(n);
                        }
                    }
                    continue;
                }

                // <dll>_NULL_THUNK_DATA
                if lname.ends_with("_null_thunk_data") {
                    let tail = name
                        .trim_end_matches("_NULL_THUNK_DATA")
                        .trim_start_matches('_')
                        .to_string();
                    let n = normalize_dll_name(&tail);
                    if !n.is_empty() && dll_hint.is_none() {
                        dll_hint = Some(n);
                    }
                    continue;
                }

                // Imported symbol stubs use __imp_ / _imp_ prefixes
                if name.starts_with("__imp_")
                    || name.starts_with("_imp_")
                    || name.starts_with("__imp")
                    || name.starts_with("_imp")
                {
                    let cleaned = name
                        .trim_start_matches("__imp_")
                        .trim_start_matches("_imp_")
                        .trim_start_matches("__imp")
                        .trim_start_matches("_imp")
                        .trim_start_matches("imp_")
                        .to_string();
                    let n = normalize_symbol(&cleaned);
                    if !n.is_empty() {
                        syms.push((cleaned, n));
                    }
                    continue;
                }

                // MSVC-mangled names starting with '?' may decode to useful identifiers
                if name.starts_with('?') || name.starts_with("??") {
                    let n = normalize_symbol(name);
                    if !n.is_empty() {
                        syms.push((name.to_string(), n));
                    }
                    continue;
                }

                // If a raw symbol literally contains a dll name (rare), capture it
                if name.to_ascii_lowercase().ends_with(".dll") && dll_hint.is_none() {
                    let n = normalize_dll_name(&name);
                    if !n.is_empty() {
                        dll_hint = Some(n);
                    }
                    continue;
                }
            }
        }

        // Also scan section data for ASCII strings ending in `.dll`
        for section in obj.sections() {
            if let Ok(secdata) = section.data() {
                for cand in find_ascii_strings_with_suffix(&secdata, ".dll") {
                    if dll_hint.is_none() {
                        let n = normalize_dll_name(&cand);
                        if !n.is_empty() {
                            dll_hint = Some(n);
                        }
                    }
                }
            }
        }

        if !syms.is_empty() || dll_hint.is_some() {
            syms.sort_by(|a, b| a.0.cmp(&b.0));
            syms.dedup_by(|a, b| a.0 == b.0 && a.1 == b.1);
            return Some((dll_hint, syms));
        }
    }

    None
}

/// Generate plausible candidate DLL paths to try given a base lib path and a hint like `openblas` or `openblas.dll`.
fn generate_candidate_dll_paths(base_lib: &std::path::Path, hint: &str) -> Vec<std::path::PathBuf> {
    use std::env;
    let mut out = Vec::new();
    let mut names = Vec::new();
    let hint_trim = hint.trim_matches(|c: char| c == '"' || c == '\'' || c == ' ');
    if hint_trim.to_ascii_lowercase().ends_with(".dll") {
        names.push(hint_trim.to_string());
    } else {
        names.push(format!("{}.dll", hint_trim));
        names.push(format!("lib{}.dll", hint_trim));
    }

    let parent = base_lib
        .parent()
        .unwrap_or_else(|| std::path::Path::new("."));
    for name in &names {
        out.push(parent.join(name));
        out.push(parent.join("bin").join(name));
        out.push(parent.join("..").join("bin").join(name));
    }

    // also check PATH entries
    if let Ok(path_var) = env::var("PATH") {
        for entry in path_var.split(';') {
            if !entry.is_empty() {
                let p = std::path::Path::new(entry).join(&names[0]);
                out.push(p);
            }
        }
    }

    // unique
    out.sort();
    out.dedup();
    out
}

/// Given a base library path (`.lib`) and a DLL hint, try to locate and inspect candidate DLL files and insert their exports into `set`.
fn try_inspect_referenced_dlls(
    base_lib: &std::path::Path,
    hint: &str,
    set: &mut std::collections::HashSet<String>,
    seen: &mut std::collections::HashSet<std::path::PathBuf>,
) {
    // First try direct candidate paths
    let candidates = generate_candidate_dll_paths(base_lib, hint);
    for cand in candidates.iter() {
        if cand.exists() {
            let canonical = cand.canonicalize().unwrap_or(cand.clone());
            if seen.contains(&canonical) {
                continue;
            }
            if let Ok(bytes) = std::fs::read(&canonical) {
                if let Ok(obj) = object::File::parse(&*bytes) {
                    for sym in obj.dynamic_symbols() {
                        if let Ok(name) = sym.name() {
                            if !name.is_empty() {
                                let s = normalize_symbol(&name);
                                if !s.is_empty() {
                                    set.insert(s);
                                }
                            }
                        }
                    }
                    for sym in obj.symbols() {
                        if let Ok(name) = sym.name() {
                            if !name.is_empty() {
                                let s = normalize_symbol(&name);
                                if !s.is_empty() {
                                    set.insert(s);
                                }
                            }
                        }
                    }
                }
            }
            seen.insert(canonical);
        }
    }

    // If direct candidates didn't find anything, search within the package root (e.g. installed/<triplet>)
    let maybe_root = base_lib
        .parent()
        .and_then(|p| p.parent())
        .unwrap_or_else(|| base_lib.parent().unwrap_or(std::path::Path::new(".")));

    // attempt search for the primary candidate name
    let primary_name = if hint.to_ascii_lowercase().ends_with(".dll") {
        hint.to_string()
    } else {
        format!("{}.dll", hint)
    };

    if let Some(found) = search_for_file_within(maybe_root, &primary_name, 4) {
        let canonical = found.canonicalize().unwrap_or(found.clone());
        if !seen.contains(&canonical) {
            if let Ok(bytes) = std::fs::read(&canonical) {
                if let Ok(obj) = object::File::parse(&*bytes) {
                    for sym in obj.dynamic_symbols() {
                        if let Ok(name) = sym.name() {
                            if !name.is_empty() {
                                let s = normalize_symbol(&name);
                                if !s.is_empty() {
                                    set.insert(s);
                                }
                            }
                        }
                    }
                    for sym in obj.symbols() {
                        if let Ok(name) = sym.name() {
                            if !name.is_empty() {
                                let s = normalize_symbol(&name);
                                if !s.is_empty() {
                                    set.insert(s);
                                }
                            }
                        }
                    }
                }
            }
            seen.insert(canonical);
        }
    }
}

/// Recursively search `root` up to `max_depth` for a file named `filename`.
fn search_for_file_within(
    root: &std::path::Path,
    filename: &str,
    max_depth: usize,
) -> Option<std::path::PathBuf> {
    use std::collections::VecDeque;
    let mut q: VecDeque<(std::path::PathBuf, usize)> = VecDeque::new();
    q.push_back((root.to_path_buf(), 0));
    while let Some((dir, depth)) = q.pop_front() {
        if depth > max_depth {
            continue;
        }
        if let Ok(rd) = std::fs::read_dir(&dir) {
            for entry in rd.flatten() {
                let p = entry.path();
                if p.is_file() {
                    if let Some(name) = p.file_name().and_then(|n| n.to_str()) {
                        if name.eq_ignore_ascii_case(filename) {
                            return Some(p);
                        }
                    }
                } else if p.is_dir() {
                    q.push_back((p, depth + 1));
                }
            }
        }
    }
    None
}

/// Try to run `dumpbin.exe /exports <path>` and parse exported names (Windows fallback).
fn try_dumpbin_exports(path: &std::path::Path) -> Option<Vec<String>> {
    let p = path.to_str()?;
    let out = Command::new("dumpbin.exe")
        .arg("/exports")
        .arg(p)
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let stdout = String::from_utf8_lossy(&out.stdout);
    let mut names = Vec::new();
    for line in stdout.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        // heuristic: last token on the line is likely the name
        let parts: Vec<&str> = trimmed.split_whitespace().collect();
        if parts.len() >= 1 {
            let last = parts.last().unwrap().to_string();
            // skip lines that look like headers or numeric-only
            if last.chars().any(|c| c.is_ascii_alphabetic()) {
                names.push(last);
            }
        }
    }
    if names.is_empty() { None } else { Some(names) }
}

/// Compute the ordinal for an export given optional goblin ExportData and the
/// export index (name-table index). Falls back to `ordinal_base + index` when
/// ordinal table is missing.
fn compute_export_ordinal(
    export_data_opt: Option<&goblin::pe::export::ExportData>,
    index: usize,
) -> u64 {
    let export_base = export_data_opt
        .map(|ed| ed.export_directory_table.ordinal_base)
        .unwrap_or(1);

    export_data_opt
        .and_then(|ed| {
            ed.export_ordinal_table
                .get(index)
                .map(|o| ed.export_directory_table.ordinal_base as u64 + *o as u64)
        })
        .unwrap_or(export_base as u64 + index as u64)
}

/// Download bytes and write to a temp cache file (by checksum). Returns (path, checksum).
pub fn download_and_cache(url: &str) -> Result<(PathBuf, String)> {
    let resp = reqwest::blocking::get(url)?;
    if !resp.status().is_success() {
        anyhow::bail!("download failed: {}", resp.status());
    }
    let bytes = resp.bytes()?;
    let checksum = sha256_hex(&bytes);

    let mut path = std::env::temp_dir();
    path.push(format!("library_inspector_{}.bin", &checksum[..16]));
    if path.exists() {
        return Ok((path, checksum));
    }

    std::fs::write(&path, &bytes)?;
    Ok((path, checksum))
}

/// Inspect a local library file and extract exported symbol names.
pub fn inspect_library<P: AsRef<std::path::Path>>(path: P) -> Result<LibraryInspection> {
    let path = path.as_ref().to_path_buf();
    let data = std::fs::read(&path)?;
    let checksum = sha256_hex(&data);
    let format = detect_format(&data);

    use std::collections::{HashMap, HashSet};
    let mut set: HashSet<String> = HashSet::new();
    // preserve raw exported symbol names as discovered
    let mut raw_set: HashSet<String> = HashSet::new();
    // collect any DLL hints we discover while parsing .lib archive members
    let mut referenced_dlls_set: std::collections::HashSet<String> =
        std::collections::HashSet::new();
    // symbol -> kind mapping from object::Symbol
    let mut symbol_kind_map: HashMap<String, SymbolKind> = HashMap::new();
    // goblin export metadata map: raw_name -> (ordinal, forwarded_to)
    let mut goblin_export_map: HashMap<String, (Option<u64>, Option<String>)> = HashMap::new();

    // Try parsing with `object` crate first. If it fails or yields no symbols,
    // fall back to goblin-based PE/archive parsing heuristics below.
    let mut arch = String::new();
    if let Ok(obj) = object::File::parse(&*data) {
        arch = format!("{:?}", obj.architecture());

        // collect dynamic symbols first
        for sym in obj.dynamic_symbols() {
            if let Ok(name) = sym.name() {
                if !name.is_empty() {
                    raw_set.insert(name.to_string());
                    symbol_kind_map
                        .entry(name.to_string())
                        .or_insert(sym.kind());
                    let s = normalize_symbol(&name);
                    if !s.is_empty() {
                        set.insert(s);
                    }
                }
            }
        }

        // fallback to all symbols
        for sym in obj.symbols() {
            if let Ok(name) = sym.name() {
                if !name.is_empty() {
                    raw_set.insert(name.to_string());
                    symbol_kind_map
                        .entry(name.to_string())
                        .or_insert(sym.kind());
                    let s = normalize_symbol(&name);
                    if !s.is_empty() {
                        set.insert(s);
                    }
                }
            }
        }
    }

    // Try to parse PE export directory with goblin to capture ordinals and forwarders
    if format == "pe" {
        if let Ok(gob) = GoblinObject::parse(&data) {
            if let GoblinObject::PE(pe) = gob {
                // Compute ordinals using goblin's ExportData when available.
                for (i, exp) in pe.exports.iter().enumerate() {
                    let ordinal = Some(compute_export_ordinal(pe.export_data.as_ref(), i));
                    if let Some(name) = exp.name.as_ref() {
                        raw_set.insert(name.to_string());
                        let s = normalize_symbol(name);
                        if !s.is_empty() {
                            set.insert(s);
                        }
                        let fwd = reexport_readable(&exp.reexport);
                        goblin_export_map.insert(name.to_string(), (ordinal, fwd));
                    } else if exp.reexport.is_some() {
                        // export forwarded without name; record forwarder for diagnostics
                        let fwd = reexport_readable(&exp.reexport);
                        goblin_export_map.insert(format!("#rva:{}", exp.rva), (ordinal, fwd));
                    }
                }
            }
        }
    }

    // If object parsing didn't yield any symbols, attempt goblin fallbacks
    if set.is_empty() {
        // If PE, try goblin::pe to read export directory
        if format == "pe" {
            if let Ok(gob) = GoblinObject::parse(&data) {
                if let GoblinObject::PE(pe) = gob {
                    // try to read export names if present
                    for exp in pe.exports.iter() {
                        if let Some(name) = exp.name.as_ref() {
                            let s = normalize_symbol(name);
                            if !s.is_empty() {
                                set.insert(s);
                            }
                        }
                    }
                }
            }
        }

        // If archive (.lib / ar) try a simple ar parser to extract members
        if format == "ar" {
            // track DLLs we've already inspected to avoid duplicates
            let mut seen_inspected_dlls: std::collections::HashSet<std::path::PathBuf> =
                std::collections::HashSet::new();

            // parse ar headers and iterate members
            for (member_name, member_bytes) in parse_ar_members(&data) {
                // If the member name implies an import descriptor, attempt to inspect referenced DLLs.
                if member_name.starts_with("IMPORT_DESCRIPTOR_") {
                    let dll_hint_raw = member_name.trim_start_matches("IMPORT_DESCRIPTOR_");
                    let dll_hint = normalize_dll_name(dll_hint_raw);
                    if !dll_hint.is_empty() {
                        referenced_dlls_set.insert(dll_hint.clone());
                        try_inspect_referenced_dlls(
                            &path,
                            &dll_hint,
                            &mut set,
                            &mut seen_inspected_dlls,
                        );
                    }
                }

                // Attempt a focused COFF import parse for this member. Many .lib members are
                // small COFF object files that contain import descriptors and stub symbols
                // (e.g. __imp_Foo, IMPORT_DESCRIPTOR_Bar). Extract any discovered symbols and
                // a DLL hint if present.
                if let Some((maybe_dll, imported_pairs)) = parse_coff_import_member(member_bytes) {
                    if let Some(dll_hint) = maybe_dll {
                        let n = normalize_dll_name(&dll_hint);
                        if !n.is_empty() {
                            referenced_dlls_set.insert(n.clone());
                            try_inspect_referenced_dlls(
                                &path,
                                &n,
                                &mut set,
                                &mut seen_inspected_dlls,
                            );
                        }
                    }
                    for (raw, norm) in imported_pairs {
                        if !raw.is_empty() {
                            raw_set.insert(raw.clone());
                        }
                        if !norm.is_empty() {
                            set.insert(norm.clone());
                        }
                    }
                }

                // Look for embedded ASCII DLL names inside the member as an extra heuristic.
                for cand in find_ascii_strings_with_suffix(member_bytes, ".dll") {
                    let n = normalize_dll_name(&cand);
                    if !n.is_empty() {
                        referenced_dlls_set.insert(n.clone());
                        try_inspect_referenced_dlls(&path, &n, &mut set, &mut seen_inspected_dlls);
                    }
                }

                // Also try parsing the member with `object` to grab symbol names the crate
                // understands (keeps previous fallback behavior).
                if let Ok(inner) = object::File::parse(member_bytes) {
                    if arch.is_empty() {
                        arch = format!("{:?}", inner.architecture());
                    }
                    for sym in inner.dynamic_symbols() {
                        if let Ok(name) = sym.name() {
                            if !name.is_empty() {
                                raw_set.insert(name.to_string());
                                symbol_kind_map
                                    .entry(name.to_string())
                                    .or_insert(sym.kind());
                                let s = normalize_symbol(&name);
                                if !s.is_empty() {
                                    set.insert(s);
                                }
                            }
                        }
                    }
                    for sym in inner.symbols() {
                        if let Ok(name) = sym.name() {
                            if !name.is_empty() {
                                raw_set.insert(name.to_string());
                                symbol_kind_map
                                    .entry(name.to_string())
                                    .or_insert(sym.kind());
                                let s = normalize_symbol(&name);
                                if !s.is_empty() {
                                    set.insert(s);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Final fallback: use dumpbin.exe on Windows for PE files when available
    if set.is_empty() && format == "pe" {
        if let Some(exts) = try_dumpbin_exports(&path) {
            for name in exts {
                let s = normalize_symbol(&name);
                if !s.is_empty() {
                    set.insert(s);
                }
            }
        }
    }

    let mut symbols: Vec<String> = set
        .into_iter()
        .filter(|s| !s.is_empty())
        .filter(|s| {
            let low = s.to_ascii_lowercase();
            !(low.starts_with("import_descriptor_")
                || low.starts_with(".idata")
                || low.starts_with(".rsrc")
                || low.contains("null_import_descriptor")
                || low.contains("_null_thunk_data")
                || low.starts_with("@comp.id"))
        })
        .collect();
    symbols.sort();

    let arch = if arch.is_empty() {
        format.clone()
    } else {
        arch
    };

    // Prepare referenced DLL list
    let mut referenced_dlls: Vec<String> = referenced_dlls_set
        .into_iter()
        .filter(|s| !s.is_empty())
        .collect();
    referenced_dlls.sort();

    // Heuristic file type
    let file_type = if format == "ar" {
        if !referenced_dlls.is_empty() {
            "import_lib".to_string()
        } else {
            "archive".to_string()
        }
    } else if format == "pe" {
        if symbols.is_empty() && !referenced_dlls.is_empty() {
            "import_lib".to_string()
        } else {
            "shared_lib".to_string()
        }
    } else if format == "elf" {
        "shared_lib".to_string()
    } else {
        "unknown".to_string()
    };

    // Build export metadata entries (best-effort) using preserved raw names
    let mut raw_names: Vec<String> = raw_set.into_iter().filter(|s| !s.is_empty()).collect();
    raw_names.sort();

    let exports: Vec<ExportInfo> = raw_names
        .iter()
        .map(|raw_name| {
            // Compute normalized form first (used in multiple places)
            let normalized = normalize_symbol(raw_name);

            // try demangling: prefer MSVC demangler for `?`-style names, then Itanium-style
            let demangled = {
                // MSVC-style demangle if it looks like an MSVC mangled name
                if raw_name.starts_with('?') || raw_name.starts_with("??") {
                    undname::demangle(raw_name, UndnameFlags::default()).ok()
                } else {
                    None
                }
            }
            .or_else(|| {
                CppSymbol::new(raw_name)
                    .ok()
                    .and_then(|sym| sym.demangle().ok())
            })
            .or_else(|| {
                // try normalized form with MSVC, then with Cpp demangler
                undname::demangle(&normalized, UndnameFlags::default())
                    .ok()
                    .or_else(|| {
                        CppSymbol::new(&normalized)
                            .ok()
                            .and_then(|sym| sym.demangle().ok())
                    })
            });

            // lookup ordinal/forwarder from goblin map (try raw then normalized)
            let (ordinal, forwarded_to) = goblin_export_map
                .get(raw_name)
                .cloned()
                .or_else(|| goblin_export_map.get(&normalized).cloned())
                .unwrap_or((None, None));

            // kind: try to map from object symbol kinds collected earlier
            let kind = symbol_kind_map
                .get(raw_name)
                .or_else(|| symbol_kind_map.get(&normalized))
                .map(|k| match k {
                    SymbolKind::Text => "func".to_string(),
                    SymbolKind::Data => "data".to_string(),
                    _ => "unknown".to_string(),
                });

            // calling convention hint (simple heuristic)
            let calling_convention_hint = if raw_name.contains('@') {
                Some("stdcall".to_string())
            } else {
                None
            };

            // Candidate generation: prefer normalized -> raw -> underscored -> demangled -> ordinal
            let mut candidates: Vec<String> = Vec::new();
            if !normalized.is_empty() && normalized != *raw_name {
                candidates.push(normalized.clone());
            }
            candidates.push(raw_name.clone());
            if !raw_name.starts_with('_') {
                candidates.push(format!("_{}", raw_name));
            } else {
                let trimmed = raw_name.trim_start_matches('_').to_string();
                if !trimmed.is_empty() && trimmed != *raw_name {
                    candidates.push(trimmed);
                }
            }
            if let Some(dem) = &demangled {
                if !candidates.contains(dem) {
                    candidates.push(dem.clone());
                }
            }
            if let Some(ord) = ordinal {
                candidates.push(format!("ordinal:{}", ord));
            }

            let confidence = if ordinal.is_some() { 1.0 } else { 0.85 };

            ExportInfo {
                raw_name: raw_name.clone(),
                demangled,
                normalized: normalized.clone(),
                kind,
                ordinal,
                forwarded_to,
                calling_convention_hint,
                candidates,
                confidence,
            }
        })
        .collect();

    Ok(LibraryInspection {
        path,
        checksum,
        format,
        arch,
        file_type,
        soname: None,
        referenced_dlls,
        exports,
        symbols,
    })
}

/// Compare an inspection against an FFI info structure and produce coverage info.
pub fn compare_with_ffi(
    inspection: &LibraryInspection,
    ffi: &c_header_analyzer::FfiInfo,
) -> CoverageReport {
    use std::collections::HashSet;
    let mut expected: HashSet<String> = HashSet::new();
    for f in &ffi.functions {
        expected.insert(f.name.clone());
    }
    for c in &ffi.constants {
        expected.insert(c.name.clone());
    }

    let exported: HashSet<String> = inspection.symbols.iter().cloned().collect();

    let mut matched: Vec<String> = expected
        .iter()
        .filter(|s| exported.contains(*s))
        .cloned()
        .collect();
    matched.sort();

    let mut missing: Vec<String> = expected
        .iter()
        .filter(|s| !exported.contains(*s))
        .cloned()
        .collect();
    missing.sort();

    let expected_count = expected.len();
    let matched_count = matched.len();
    let percent = if expected_count == 0 {
        100.0
    } else {
        (matched_count as f64) / (expected_count as f64) * 100.0
    };

    CoverageReport {
        expected_count,
        matched_count,
        percent,
        matched,
        missing,
    }
}

/// Create a `library_db::LibraryEntry` from the inspection.
pub fn library_entry_from_inspection(
    inspection: &LibraryInspection,
    name_hint: Option<&str>,
    homepage: Option<&str>,
) -> library_db::LibraryEntry {
    let file_name = inspection
        .path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("library")
        .to_string();

    let name = name_hint
        .map(|s| s.to_string())
        .unwrap_or(file_name.clone());

    library_db::LibraryEntry {
        name,
        description: Some(format!(
            "Auto-generated from {} (format={}, arch={})",
            inspection.path.display(),
            inspection.format,
            inspection.arch
        )),
        symbols: inspection.symbols.clone(),
        homepage: homepage.map(|s| s.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sha256_roundtrip() {
        let data = b"hello world";
        let hex = sha256_hex(data);
        assert!(!hex.is_empty());
    }

    #[test]
    fn normalize_dll_name_strips_padding_and_quotes() {
        // Leading DEL (0x7f) should be removed
        assert_eq!(normalize_dll_name("\u{7f}cudnn64_9"), "cudnn64_9");
        // Surrounding quotes trimmed
        assert_eq!(normalize_dll_name("\"libpng16-16.dll\""), "libpng16-16.dll");
        // Whitespace trimmed
        assert_eq!(normalize_dll_name("  openblas  "), "openblas");
    }

    #[test]
    fn compute_export_ordinal_basic() {
        use goblin::pe::export::{ExportData, ExportDirectoryTable};

        let mut edt = ExportDirectoryTable::default();
        edt.ordinal_base = 5;

        let ed = ExportData {
            name: None,
            export_directory_table: edt,
            export_name_pointer_table: vec![],
            export_ordinal_table: vec![2u16, 3u16],
            export_address_table: vec![],
        };

        // index 0 -> base + table[0] = 5 + 2 = 7
        assert_eq!(compute_export_ordinal(Some(&ed), 0), 7);
        // index 1 -> 5 + 3 = 8
        assert_eq!(compute_export_ordinal(Some(&ed), 1), 8);
        // index 2 -> fallback base + index = 5 + 2 = 7
        assert_eq!(compute_export_ordinal(Some(&ed), 2), 7);
        // no export data -> fallback to base 1 + index
        assert_eq!(compute_export_ordinal(None, 3), 4);
    }

    #[test]
    fn msvc_demangle_via_normalize_symbol() {
        // Example from undname docs
        let mangled = "?world@@YA?AUhello@@XZ";
        assert_eq!(normalize_symbol(mangled), "world");
    }

    // Prefer local fixture if available; fallback to the previous env-gated remote test.
    #[test]
    fn fixture_zlib_or_remote_if_enabled() {
        use std::path::PathBuf;

        // try local fixture path first
        let fixture_path = PathBuf::from(env!("CARGO_MANIFEST_DIR").to_string() + "/../tests/fixtures/zlib1.dll");
        if fixture_path.exists() {
            let insp = inspect_library(&fixture_path).expect("inspect local zlib fixture");
            assert!(!insp.exports.is_empty(), "no exports found in local fixture");
            assert!(insp.exports.iter().any(|e| e.ordinal.is_some()), "expect at least one ordinal");
            return;
        }

        // fallback: behave like the old remote test when env var enabled
        if std::env::var("LIBRARY_INSPECTOR_RUN_REMOTE_TESTS").unwrap_or_default() != "1" {
            eprintln!("Skipping zlib fixture test (no local fixture and remote tests disabled)");
            return;
        }

        let url = match std::env::var("LIBRARY_INSPECTOR_ZLIB_DLL_URL") {
            Ok(v) => v,
            Err(_) => {
                panic!(
                    "Set LIBRARY_INSPECTOR_ZLIB_DLL_URL to a redistributable DLL URL to run this test"
                );
            }
        };

        let (path, _checksum) = download_and_cache(&url).expect("download zlib dll");
        let insp = inspect_library(&path).expect("inspect zlib");
        assert!(!insp.exports.is_empty(), "no exports found");
        assert!(insp.exports.iter().any(|e| e.ordinal.is_some()));
    }
}
