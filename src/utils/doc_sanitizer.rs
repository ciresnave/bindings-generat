//! Documentation sanitization utilities
//!
//! Converts doxygen/LaTeX-style documentation to Markdown-compatible format
//! for Rust doc comments.

use regex::Regex;

/// Sanitize documentation text by converting doxygen/LaTeX commands to Markdown
///
/// This function handles common documentation patterns found in C/C++ headers:
/// - `\p NAME` → `` `NAME` `` (parameter references)
/// - `\c NAME` → `` `NAME` `` (code references)
/// - `\em TEXT` → `*TEXT*` (emphasis)
/// - `\b TEXT` → `**TEXT**` (bold)
/// - `\brief` → (removed, redundant in Rust)
/// - `\return`, `\returns` → `Returns:`
/// - `\param NAME DESC` → `- `NAME`: DESC`
/// - Other backslash commands are removed
///
/// # Examples
///
/// ```
/// # fn sanitize_doc(s: &str) -> String { String::new() }
/// let input = r#"\brief Sets parameter \p alpha and returns \b success"#;
/// let output = sanitize_doc(input);
/// // Output will have `alpha` instead of \p alpha
/// // and **success** instead of \b success
/// ```
pub fn sanitize_doc(s: &str) -> String {
    let mut result = s.to_string();

    // Remove \brief (redundant in Rust)
    let re_brief = Regex::new(r"(?m)\\brief\s+").unwrap();
    result = re_brief.replace_all(&result, "").into_owned();

    // Convert \return, \returns to "Returns:"
    let re_return = Regex::new(r"(?m)\\returns?\s+").unwrap();
    result = re_return.replace_all(&result, "Returns: ").into_owned();

    // Convert \param NAME DESC to "- `NAME`: DESC"
    // Use a simpler regex without lookahead, handle leading whitespace
    let re_param = Regex::new(r"(?m)^\s*\\param\s+([A-Za-z0-9_]+)\s+(.+)$").unwrap();
    result = re_param.replace_all(&result, "- `$1`: $2").into_owned();

    // Convert \p NAME to `NAME` (parameter reference)
    let re_p = Regex::new(r"\\p\s+([A-Za-z0-9_*]+)").unwrap();
    result = re_p.replace_all(&result, "`$1`").into_owned();

    // Convert \c NAME to `NAME` (code reference)
    let re_c = Regex::new(r"\\c\s+([A-Za-z0-9_*]+)").unwrap();
    result = re_c.replace_all(&result, "`$1`").into_owned();

    // Convert \em text to *text* (emphasis)
    // Capture single word after \em
    let re_em = Regex::new(r"\\em\s+([A-Za-z0-9_]+)").unwrap();
    result = re_em.replace_all(&result, "*$1*").into_owned();

    // Convert \b text to **text** (bold)
    // Capture single word after \b
    let re_b = Regex::new(r"\\b\s+([A-Za-z0-9_]+)").unwrap();
    result = re_b.replace_all(&result, "**$1**").into_owned();

    // Convert \note to "Note:"
    let re_note = Regex::new(r"(?m)\\note\s+").unwrap();
    result = re_note.replace_all(&result, "Note: ").into_owned();

    // Convert \warning to "Warning:"
    let re_warning = Regex::new(r"(?m)\\warning\s+").unwrap();
    result = re_warning.replace_all(&result, "Warning: ").into_owned();

    // Convert \sa (see also) to "See also:"
    let re_sa = Regex::new(r"(?m)\\sa\s+").unwrap();
    result = re_sa.replace_all(&result, "See also: ").into_owned();

    // Remove other common single-word backslash commands that don't need translation
    // (like \notefnerr, \note_init_rt, \note_callback, \ingroup, etc.)
    let re_cmd = Regex::new(r"\\(?:notefnerr|note_init_rt|note_callback|note_string_api_versions|note_null_stream|ingroup|internal|deprecated|hideinitializer|showinitializer)\b").unwrap();
    result = re_cmd.replace_all(&result, "").into_owned();

    // Clean up any remaining unknown backslash commands (conservative)
    // Only remove if they're clearly command-like (backslash followed by letters)
    let re_unknown = Regex::new(r"\\([A-Za-z_][A-Za-z0-9_]*)\b").unwrap();
    result = re_unknown.replace_all(&result, "").into_owned();

    // Clean up multiple consecutive blank lines
    let re_blanks = Regex::new(r"\n{3,}").unwrap();
    result = re_blanks.replace_all(&result, "\n\n").into_owned();

    // Trim leading/trailing whitespace
    result.trim().to_string()
}

/// Emit documentation as line-based doc comments (`///`)
///
/// This is the preferred method for emitting documentation as it doesn't
/// require escaping backslashes or quotes.
///
/// # Arguments
///
/// * `output` - The output buffer to write to
/// * `doc` - The documentation text (should be pre-sanitized)
///
/// # Examples
///
/// ```
/// # fn sanitize_doc(s: &str) -> String { String::new() }
/// # fn emit_doc_lines(output: &mut String, doc: &str) -> std::fmt::Result { Ok(()) }
/// let raw = r#"\brief Sets \p alpha"#;
/// let sanitized = sanitize_doc(raw);
/// let mut output = String::new();
/// emit_doc_lines(&mut output, &sanitized).unwrap();
/// // Output will contain "/// Sets `alpha`"
/// ```
pub fn emit_doc_lines(output: &mut String, doc: &str) -> std::fmt::Result {
    use std::fmt::Write;

    for line in doc.lines() {
        writeln!(output, "/// {}", line)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sanitize_param_reference() {
        let input = r#"Sets parameter \p alpha and \p beta"#;
        let output = sanitize_doc(input);
        assert!(output.contains("`alpha`"));
        assert!(output.contains("`beta`"));
        assert!(!output.contains(r"\p"));
    }

    #[test]
    fn test_sanitize_code_reference() {
        let input = r#"Call \c cudaFree when done"#;
        let output = sanitize_doc(input);
        assert!(output.contains("`cudaFree`"));
        assert!(!output.contains(r"\c"));
    }

    #[test]
    fn test_sanitize_emphasis_and_bold() {
        let input = r#"This is \em important and \b critical"#;
        let output = sanitize_doc(input);
        assert!(output.contains("*important*"));
        assert!(output.contains("**critical**"));
        assert!(!output.contains(r"\em"));
        assert!(!output.contains(r"\b"));
    }

    #[test]
    fn test_sanitize_brief() {
        let input = r#"\brief Returns a handle to a device"#;
        let output = sanitize_doc(input);
        assert_eq!(output, "Returns a handle to a device");
        assert!(!output.contains(r"\brief"));
    }

    #[test]
    fn test_sanitize_return() {
        let input = r#"\returns Success code"#;
        let output = sanitize_doc(input);
        assert!(output.contains("Returns: Success code"));
        assert!(!output.contains(r"\returns"));
    }

    #[test]
    fn test_sanitize_param() {
        let input = r#"\param device Returned device ordinal
\param pciBusId String in PCI bus ID format"#;
        let output = sanitize_doc(input);
        assert!(output.contains("- `device`: Returned device ordinal"));
        assert!(output.contains("- `pciBusId`: String in PCI bus ID format"));
        assert!(!output.contains(r"\param"));
    }

    #[test]
    fn test_sanitize_note_commands() {
        let input = r#"\note This is important
\notefnerr
\note_init_rt
\note_callback"#;
        let output = sanitize_doc(input);
        assert!(output.contains("Note: This is important"));
        assert!(!output.contains(r"\notefnerr"));
        assert!(!output.contains(r"\note_init_rt"));
        assert!(!output.contains(r"\note_callback"));
    }

    #[test]
    fn test_sanitize_see_also() {
        let input = r#"\sa cudaDeviceGetPCIBusId"#;
        let output = sanitize_doc(input);
        assert!(output.contains("See also: cudaDeviceGetPCIBusId"));
        assert!(!output.contains(r"\sa"));
    }

    #[test]
    fn test_complex_cuda_doc() {
        let input = r#"\brief Returns a handle to a compute device

 Returns in \p *device a device ordinal given a PCI bus ID string.

 \param device   - Returned device ordinal

 \param pciBusId - String in one of the following forms:
 [domain]:[bus]:[device].[function]
 where \p domain, \p bus, \p device, and \p function are all hexadecimal values

 \return
 ::cudaSuccess,
 ::cudaErrorInvalidValue
 \notefnerr
 \note_init_rt

 \sa
 ::cudaDeviceGetPCIBusId"#;

        let output = sanitize_doc(input);

        // Should contain sanitized content
        assert!(output.contains("`*device`"), "Missing `*device`");
        assert!(output.contains("- `device`"), "Missing - `device`");
        assert!(output.contains("- `pciBusId`"), "Missing - `pciBusId`");
        assert!(output.contains("`domain`"), "Missing `domain`");
        assert!(output.contains("Returns:"));
        assert!(output.contains("See also:"));

        // Should NOT contain raw backslash commands
        assert!(!output.contains(r"\p "));
        assert!(!output.contains(r"\param"));
        assert!(!output.contains(r"\brief"));
        assert!(!output.contains(r"\notefnerr"));
        assert!(!output.contains(r"\sa"));
    }

    #[test]
    fn test_emit_doc_lines() {
        let doc = "This is line one\nThis is line two\nThis is line three";
        let mut output = String::new();
        emit_doc_lines(&mut output, doc).unwrap();

        assert!(output.contains("/// This is line one"));
        assert!(output.contains("/// This is line two"));
        assert!(output.contains("/// This is line three"));
    }
}
