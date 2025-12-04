//! C/C++ header comment parser for extracting inline documentation.
//!
//! This module parses Doxygen-style comments directly from C/C++ headers to extract
//! function, type, and parameter documentation that may not be available in external
//! documentation files.

use anyhow::{Context, Result};
use regex::Regex;
use std::collections::HashMap;
use std::fs;
use std::path::Path;

/// A parsed comment for a function
#[derive(Debug, Clone, Default)]
pub struct FunctionComment {
    pub function_name: String,
    pub brief: Option<String>,
    pub detailed: Option<String>,
    pub param_docs: HashMap<String, ParamDoc>,
    pub return_doc: Option<String>,
    pub notes: Vec<String>,
    pub warnings: Vec<String>,
    pub see_also: Vec<String>,
    pub deprecated: Option<String>,
}

/// Documentation for a function parameter
#[derive(Debug, Clone)]
pub struct ParamDoc {
    pub name: String,
    pub description: String,
    pub direction: Option<ParamDirection>,
}

/// Parameter direction (in, out, inout)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParamDirection {
    In,
    Out,
    InOut,
}

impl ParamDirection {
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "in" => Some(Self::In),
            "out" => Some(Self::Out),
            "inout" | "in,out" | "in-out" | "in_out" => Some(Self::InOut),
            _ => None,
        }
    }
}

/// Parser for C/C++ header comments
pub struct HeaderCommentParser {
    /// Regex for block comments /** ... */
    #[allow(dead_code)]
    block_comment_re: Regex,
    /// Regex for line comments ///
    #[allow(dead_code)]
    line_comment_re: Regex,
    /// Regex for Doxygen @param tag
    param_re: Regex,
    /// Regex for Doxygen @return/@returns tag
    return_re: Regex,
    /// Regex for Doxygen @brief tag
    brief_re: Regex,
    /// Regex for Doxygen @details tag
    details_re: Regex,
    /// Regex for Doxygen @note tag
    note_re: Regex,
    /// Regex for Doxygen @warning tag
    warning_re: Regex,
    /// Regex for Doxygen @see tag
    see_re: Regex,
    /// Regex for Doxygen @deprecated tag
    deprecated_re: Regex,
    /// Regex for inline comments /* ... */
    inline_comment_re: Regex,
}

impl HeaderCommentParser {
    pub fn new() -> Result<Self> {
        Ok(Self {
            block_comment_re: Regex::new(r"/\*\*(.*?)\*/")?,
            line_comment_re: Regex::new(r"///(.*)$")?,
            param_re: Regex::new(r"@param(?:\[([^\]]+)\])?\s+(\w+)\s+(.+)")?,
            return_re: Regex::new(r"@returns?\s+(.+)")?,
            brief_re: Regex::new(r"@brief\s+(.+)")?,
            details_re: Regex::new(r"@details?\s+(.+)")?,
            note_re: Regex::new(r"@note\s+(.+)")?,
            warning_re: Regex::new(r"@warning\s+(.+)")?,
            see_re: Regex::new(r"@see\s+(.+)")?,
            deprecated_re: Regex::new(r"@deprecated\s+(.+)")?,
            inline_comment_re: Regex::new(r"/\*\s*(.+?)\s*\*/")?,
        })
    }

    /// Parse all comments from a header file
    pub fn parse_header_file(&self, path: &Path) -> Result<Vec<FunctionComment>> {
        let content = fs::read_to_string(path)
            .with_context(|| format!("Failed to read header file: {}", path.display()))?;
        self.parse_header_content(&content)
    }

    /// Parse comments from header content string (useful for testing)
    pub fn parse_header_content(&self, content: &str) -> Result<Vec<FunctionComment>> {
        let mut comments = Vec::new();
        let lines: Vec<&str> = content.lines().collect();
        let mut processed_functions: std::collections::HashSet<String> =
            std::collections::HashSet::new();

        let mut i = 0;
        while i < lines.len() {
            let line = lines[i].trim();

            // Look for Doxygen-style comment blocks first (higher priority)
            if line.starts_with("/**") || line.starts_with("///") {
                if let Some(comment_content) = self.extract_comment_block(&lines, &mut i) {
                    // Parse the comment
                    let mut func_comment = FunctionComment::default();
                    self.parse_comment_content(&comment_content, &mut func_comment);

                    // Try to associate with following function
                    if let Some(func_name) = self.find_next_function(&lines, i) {
                        func_comment.function_name = func_name.clone();
                        processed_functions.insert(func_name);
                        comments.push(func_comment);
                    }
                }
            } else {
                // Look for inline-commented function declarations (like cuDNN style)
                if let Some(func_comment) = self.extract_inline_commented_function(&lines, i) {
                    // Check if we've already processed this function
                    if !processed_functions.contains(&func_comment.function_name) {
                        processed_functions.insert(func_comment.function_name.clone());
                        comments.push(func_comment);
                    }
                }
            }
            i += 1;
        }

        Ok(comments)
    }

    /// Extract a comment block starting at the current position
    fn extract_comment_block(&self, lines: &[&str], pos: &mut usize) -> Option<String> {
        let line = lines[*pos].trim();

        // Check for /** ... */ style block comment
        if line.starts_with("/**") {
            let mut comment = String::new();
            let mut end_pos = *pos;

            // Single line block comment
            if line.ends_with("*/") {
                let content = line.trim_start_matches("/**").trim_end_matches("*/").trim();
                return Some(content.to_string());
            }

            // Multi-line block comment
            while end_pos < lines.len() {
                let current = lines[end_pos];

                // Add line content (strip leading * and whitespace)
                let cleaned = current
                    .trim()
                    .trim_start_matches("/**")
                    .trim_start_matches("/*")
                    .trim_start_matches('*')
                    .trim()
                    .trim_end_matches("*/");

                if !cleaned.is_empty() {
                    comment.push_str(cleaned);
                    comment.push('\n');
                }

                if current.contains("*/") {
                    *pos = end_pos;
                    return Some(comment);
                }

                end_pos += 1;
            }
        }

        // Check for /// style line comments
        if line.starts_with("///") {
            let mut comment = String::new();
            let mut end_pos = *pos;

            while end_pos < lines.len() && lines[end_pos].trim().starts_with("///") {
                let content = lines[end_pos].trim().trim_start_matches("///").trim();

                if !content.is_empty() {
                    comment.push_str(content);
                    comment.push('\n');
                }

                end_pos += 1;
            }

            *pos = end_pos - 1;
            return Some(comment);
        }

        None
    }

    /// Parse comment content and extract Doxygen tags
    fn parse_comment_content(&self, content: &str, func_comment: &mut FunctionComment) {
        let mut current_section = String::new();
        let mut in_details = false;
        let mut in_return = false;
        let mut current_param: Option<String> = None;

        for line in content.lines() {
            let line = line.trim();

            // Check for @param tag
            if let Some(captures) = self.param_re.captures(line) {
                in_details = false; // End any previous section
                in_return = false;
                let direction = captures.get(1).map(|m| m.as_str());
                let param_name = captures.get(2).map(|m| m.as_str()).unwrap_or("");
                let description = captures.get(3).map(|m| m.as_str()).unwrap_or("");

                func_comment.param_docs.insert(
                    param_name.to_string(),
                    ParamDoc {
                        name: param_name.to_string(),
                        description: description.to_string(),
                        direction: direction.and_then(ParamDirection::from_str),
                    },
                );
                current_param = Some(param_name.to_string());
                continue;
            }

            // Check for @return/@returns tag
            if let Some(captures) = self.return_re.captures(line) {
                in_details = false; // End any previous section
                current_param = None;
                let return_text = captures.get(1).map(|m| m.as_str().to_string()).unwrap();
                func_comment.return_doc = Some(return_text);
                in_return = true; // Mark that we're in a @return section
                continue;
            }

            // Check for @brief tag
            if let Some(captures) = self.brief_re.captures(line) {
                in_details = false; // End any previous section
                in_return = false;
                current_param = None;
                func_comment.brief = captures.get(1).map(|m| m.as_str().to_string());
                continue;
            }

            // Check for @details/@detail tag
            if let Some(captures) = self.details_re.captures(line) {
                in_return = false;
                current_param = None;
                let details_text = captures.get(1).map(|m| m.as_str().to_string()).unwrap();
                if let Some(ref mut detailed) = func_comment.detailed {
                    detailed.push(' ');
                    detailed.push_str(&details_text);
                } else {
                    func_comment.detailed = Some(details_text);
                }
                in_details = true; // Mark that we're in a @details section
                continue;
            }

            // Check for @note tag
            if let Some(captures) = self.note_re.captures(line) {
                in_details = false; // End any previous section
                in_return = false;
                current_param = None;
                func_comment
                    .notes
                    .push(captures.get(1).map(|m| m.as_str().to_string()).unwrap());
                continue;
            }

            // Check for @warning tag
            if let Some(captures) = self.warning_re.captures(line) {
                in_details = false; // End any previous section
                in_return = false;
                current_param = None;
                func_comment
                    .warnings
                    .push(captures.get(1).map(|m| m.as_str().to_string()).unwrap());
                continue;
            }

            // Check for @see tag
            if let Some(captures) = self.see_re.captures(line) {
                in_details = false; // End any previous section
                in_return = false;
                current_param = None;
                func_comment
                    .see_also
                    .push(captures.get(1).map(|m| m.as_str().to_string()).unwrap());
                continue;
            }

            // Check for @deprecated tag
            if let Some(captures) = self.deprecated_re.captures(line) {
                in_details = false; // End any previous section
                in_return = false;
                current_param = None;
                func_comment.deprecated = captures.get(1).map(|m| m.as_str().to_string());
                continue;
            }

            // If no tag, it's a continuation of the previous section
            if !line.is_empty() && !line.starts_with('@') {
                if in_details {
                    // Continuation of @details
                    if let Some(ref mut detailed) = func_comment.detailed {
                        detailed.push(' ');
                        detailed.push_str(line);
                    }
                } else if in_return {
                    // Continuation of @return
                    if let Some(ref mut return_doc) = func_comment.return_doc {
                        return_doc.push(' ');
                        return_doc.push_str(line);
                    }
                } else if let Some(ref param_name) = current_param {
                    // Continuation of @param
                    if let Some(param_doc) = func_comment.param_docs.get_mut(param_name) {
                        param_doc.description.push(' ');
                        param_doc.description.push_str(line);
                    }
                } else {
                    // If we don't have a brief yet, use first line as brief
                    if func_comment.brief.is_none() && current_section.is_empty() {
                        func_comment.brief = Some(line.to_string());
                    } else {
                        if !current_section.is_empty() {
                            current_section.push(' ');
                        }
                        current_section.push_str(line);
                    }
                }
            }
        }

        // Set detailed description if we collected any in current_section
        // Only use current_section if we don't already have @details
        if !current_section.is_empty() && func_comment.detailed.is_none() {
            if func_comment.brief.is_some() {
                func_comment.detailed = Some(current_section);
            } else {
                // No brief, so current_section becomes brief
                func_comment.brief = Some(current_section);
            }
        }
    }

    /// Extract inline comments from a function declaration (e.g., cuDNN style)
    /// Example:
    /// ```c
    /// cudnnStatus_t cudnnSetTensor4dDescriptor(
    ///     cudnnTensorDescriptor_t tensorDesc,
    ///     cudnnTensorFormat_t format,
    ///     cudnnDataType_t dataType,  /* image data type */
    ///     int n,                      /* number of inputs (batch size) */
    ///     int c);                     /* number of input feature maps */
    /// ```
    fn extract_inline_commented_function(
        &self,
        lines: &[&str],
        start_pos: usize,
    ) -> Option<FunctionComment> {
        let line = lines[start_pos].trim();

        // Skip comments, preprocessor directives first
        if line.starts_with("//") || line.starts_with("/*") || line.starts_with('#') {
            return None;
        }

        // Remove inline comments before checking for parentheses to avoid false positives
        // from comments containing '(' like /* number of inputs (batch size) */
        let line_without_comments = if let Some(comment_start) = line.find("/*") {
            &line[..comment_start]
        } else {
            line
        };

        // Must contain opening parenthesis for function declaration
        if !line_without_comments.contains('(') {
            return None;
        }

        // Skip typedef, struct, enum, and lines that are clearly parameter continuations
        if line.starts_with("typedef")
            || line.starts_with("struct")
            || line.starts_with("enum")
            || line.starts_with("union")
        {
            return None;
        }

        // If the line starts with whitespace and has '(' but no clear return type before it,
        // it's likely a continuation line (parameter), not the function declaration start
        if lines[start_pos].starts_with(char::is_whitespace) {
            // Check if there's a type before the parenthesis (use line without comments)
            let before_paren = line_without_comments.split('(').next().unwrap_or("");
            let tokens: Vec<&str> = before_paren.split_whitespace().collect();

            // A function declaration should have at least a return type and function name
            // If we only have one token before '(', it's likely just a parameter name
            if tokens.len() < 2 {
                return None;
            }
        }

        // Extract function name (from original line since we want the actual text)
        let func_name = if let Some(paren_pos) = line_without_comments.find('(') {
            let before_paren = &line_without_comments[..paren_pos];
            before_paren
                .split_whitespace()
                .last()?
                .trim_start_matches('*')
                .to_string()
        } else {
            return None;
        };

        if func_name.is_empty() || func_name.contains(')') {
            return None;
        }

        let mut func_comment = FunctionComment {
            function_name: func_name.clone(),
            ..Default::default()
        };

        // Collect the full function declaration (may span multiple lines)
        let mut declaration = String::new();
        let mut current_line = start_pos;
        let mut found_end = false;

        while current_line < lines.len() && !found_end {
            let line_text = lines[current_line];
            declaration.push_str(line_text);
            declaration.push('\n');

            if line_text.contains(';') || line_text.contains('{') {
                found_end = true;
            }

            current_line += 1;

            // Safety limit: don't scan too far
            if current_line - start_pos > 50 {
                return None;
            }
        }

        if !found_end {
            return None;
        }

        // Extract inline comments from parameters
        self.extract_inline_param_comments(&declaration, &mut func_comment);

        // Extract any preceding block comment for brief/detailed description
        if start_pos > 0 {
            self.extract_preceding_block_comment(lines, start_pos, &mut func_comment);
        }

        // Only return if we found at least some documentation
        if !func_comment.param_docs.is_empty() || func_comment.brief.is_some() {
            Some(func_comment)
        } else {
            None
        }
    }

    /// Extract inline comments from parameter lines
    fn extract_inline_param_comments(&self, declaration: &str, func_comment: &mut FunctionComment) {
        // First, extract all parameters from the declaration
        let mut all_params = Vec::new();
        let mut param_comments: HashMap<String, String> = HashMap::new();

        // Split into lines and process each
        for line in declaration.lines() {
            let line = line.trim();

            // Skip lines that don't look like parameters
            if line.is_empty() || line.contains("typedef") || line.contains("struct") {
                continue;
            }

            // Extract inline comment if present
            let (param_line, comment) =
                if let Some(comment_match) = self.inline_comment_re.captures(line) {
                    let comment_text = comment_match
                        .get(1)
                        .map(|m| m.as_str())
                        .unwrap_or("")
                        .trim();
                    let before_comment = &line[..line.find("/*").unwrap_or(line.len())];
                    (
                        before_comment,
                        if comment_text.is_empty() {
                            None
                        } else {
                            Some(comment_text.to_string())
                        },
                    )
                } else {
                    (line, None)
                };

            // Extract parameter name from the line
            if let Some(param_name) = self.extract_param_name(param_line) {
                // Avoid duplicates
                if !all_params.contains(&param_name) {
                    all_params.push(param_name.clone());

                    // Store comment if we have one
                    if let Some(comment_text) = comment {
                        param_comments.insert(param_name, comment_text);
                    }
                }
            }
        }

        // Create ParamDoc entries for all parameters
        for param_name in all_params {
            let description = param_comments.get(&param_name).cloned().unwrap_or_default();

            func_comment.param_docs.insert(
                param_name.clone(),
                ParamDoc {
                    name: param_name,
                    description,
                    direction: None,
                },
            );
        }
    }

    /// Extract parameter name from a declaration line
    fn extract_param_name(&self, text: &str) -> Option<String> {
        // Remove trailing punctuation and whitespace
        // Process multiple times to handle cases like "int c);" where we need to remove both ) and ;
        let mut cleaned = text.trim();
        loop {
            let before = cleaned;
            cleaned = cleaned
                .trim_end_matches(',')
                .trim_end_matches(')')
                .trim_end_matches(';')
                .trim();
            if cleaned == before {
                break;
            }
        }

        if cleaned.is_empty() {
            return None;
        }

        // Split by whitespace and get the last word (the parameter name)
        let parts: Vec<&str> = cleaned.split_whitespace().collect();

        if parts.is_empty() {
            return None;
        } // The parameter name is typically the last token
        // But handle cases like "int *param" or "const char* name"
        let mut param = parts
            .last()?
            .trim_start_matches('*')
            .trim_start_matches('&')
            .trim();

        // Handle array syntax: name[size]
        if let Some(bracket_pos) = param.find('[') {
            param = &param[..bracket_pos];
        }

        // Handle function pointers and other complex types - just get the name part
        if param.contains('(') {
            // For function pointers like (*callback)(args)
            if let Some(star_pos) = param.find('*') {
                let after_star = &param[star_pos + 1..];
                if let Some(paren_pos) = after_star.find('(') {
                    param = &after_star[..paren_pos];
                }
            }
        }

        let param = param.trim();

        // Validate it looks like an identifier
        if param.is_empty() {
            return None;
        }

        // Must start with letter or underscore
        let first_char = param.chars().next()?;
        if !first_char.is_alphabetic() && first_char != '_' {
            return None;
        }

        // All characters must be alphanumeric or underscore
        if !param.chars().all(|c| c.is_alphanumeric() || c == '_') {
            return None;
        }

        Some(param.to_string())
    }

    /// Extract any block comment preceding the function declaration
    fn extract_preceding_block_comment(
        &self,
        lines: &[&str],
        func_pos: usize,
        func_comment: &mut FunctionComment,
    ) {
        // Look back a few lines for a block comment
        let start = func_pos.saturating_sub(10);

        let mut comment_lines = Vec::new();
        let mut in_comment = false;
        let mut found_comment = false;

        for i in (start..func_pos).rev() {
            let line = lines[i].trim();

            // Skip empty lines immediately before function
            if i == func_pos - 1 && line.is_empty() {
                continue;
            }

            // Found end of comment block
            if line.contains("*/") && !line.starts_with("/**") {
                in_comment = true;
                found_comment = true;

                // Extract content from this line if it's a single-line comment
                if line.starts_with("/*") && line.ends_with("*/") {
                    let content = line.trim_start_matches("/*").trim_end_matches("*/").trim();
                    if !content.is_empty() && !content.starts_with('*') {
                        comment_lines.push(content.to_string());
                    }
                    break;
                }
                continue;
            }

            // Collect lines while in comment block
            if in_comment {
                if line.starts_with("/*") {
                    // Start of comment block
                    let content = line.trim_start_matches("/*").trim_start_matches('*').trim();
                    if !content.is_empty() {
                        comment_lines.push(content.to_string());
                    }
                    break;
                }

                // Middle of comment block
                let content = line.trim_start_matches('*').trim();
                if !content.is_empty() {
                    comment_lines.push(content.to_string());
                }
            } else if !line.is_empty() && !line.starts_with("//") {
                // Hit non-comment code, stop looking
                break;
            }
        }

        if found_comment && !comment_lines.is_empty() {
            // Reverse since we collected backwards
            comment_lines.reverse();

            // Use first line as brief, rest as detailed
            if !comment_lines.is_empty() {
                func_comment.brief = Some(comment_lines[0].clone());

                if comment_lines.len() > 1 {
                    func_comment.detailed = Some(comment_lines[1..].join(" "));
                }
            }
        }
    }

    /// Find the next function declaration after the comment
    fn find_next_function(&self, lines: &[&str], start_pos: usize) -> Option<String> {
        // Look ahead a few lines for a function declaration
        for i in start_pos..std::cmp::min(start_pos + 10, lines.len()) {
            let line = lines[i].trim();

            // Skip empty lines and preprocessor directives
            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            // Look for function pattern: type name(params)
            // This is simplified - in reality we'd need a proper C parser
            if line.contains('(') && !line.starts_with("//") && !line.starts_with("/*") {
                // Extract function name (word before the parenthesis)
                if let Some(paren_pos) = line.find('(') {
                    let before_paren = &line[..paren_pos];
                    if let Some(func_name) = before_paren.split_whitespace().last() {
                        // Clean up function pointers and other syntax
                        let cleaned = func_name
                            .trim_start_matches('*')
                            .trim_start_matches('(')
                            .trim()
                            .to_string();

                        if !cleaned.is_empty() && !cleaned.contains(')') {
                            return Some(cleaned);
                        }
                    }
                }
            }
        }

        None
    }
}

impl Default for HeaderCommentParser {
    fn default() -> Self {
        Self::new().expect("Failed to create HeaderCommentParser")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_parse_block_comment() {
        let parser = HeaderCommentParser::new().unwrap();
        let content = r#"
/**
 * Create a new handle
 * @param[out] handle - Pointer to the created handle
 * @return 0 on success
 */
int createHandle(void** handle);
"#;

        let mut file = NamedTempFile::new().unwrap();
        file.write_all(content.as_bytes()).unwrap();
        file.flush().unwrap();

        let comments = parser.parse_header_file(file.path()).unwrap();

        assert_eq!(comments.len(), 1);
        assert_eq!(comments[0].function_name, "createHandle");
        assert!(comments[0].brief.is_some());
        assert_eq!(comments[0].param_docs.len(), 1);
        assert!(comments[0].return_doc.is_some());
    }

    #[test]
    fn test_parse_line_comment() {
        let parser = HeaderCommentParser::new().unwrap();
        let content = r#"
/// Destroy a handle
/// @param handle - Handle to destroy
void destroyHandle(void* handle);
"#;

        let mut file = NamedTempFile::new().unwrap();
        file.write_all(content.as_bytes()).unwrap();
        file.flush().unwrap();

        let comments = parser.parse_header_file(file.path()).unwrap();

        assert_eq!(comments.len(), 1);
        assert_eq!(comments[0].function_name, "destroyHandle");
        assert!(comments[0].brief.is_some());
    }

    #[test]
    fn test_param_direction() {
        assert_eq!(ParamDirection::from_str("in"), Some(ParamDirection::In));
        assert_eq!(ParamDirection::from_str("out"), Some(ParamDirection::Out));
        assert_eq!(
            ParamDirection::from_str("inout"),
            Some(ParamDirection::InOut)
        );
        assert_eq!(
            ParamDirection::from_str("in-out"),
            Some(ParamDirection::InOut)
        );
    }

    #[test]
    fn test_inline_comments_cudnn_style() {
        let parser = HeaderCommentParser::new().unwrap();
        let content = r#"
cudnnStatus_t CUDNNWINAPI
cudnnSetTensor4dDescriptor(cudnnTensorDescriptor_t tensorDesc,
                           cudnnTensorFormat_t format,
                           cudnnDataType_t dataType, /* image data type */
                           int n,                    /* number of inputs (batch size) */
                           int c,                    /* number of input feature maps */
                           int h,                    /* height of input section */
                           int w);                   /* width of input section */
"#;

        let comments = parser.parse_header_content(content).unwrap();

        assert_eq!(comments.len(), 1);
        assert_eq!(comments[0].function_name, "cudnnSetTensor4dDescriptor");

        // Check we extracted inline parameter comments
        assert!(comments[0].param_docs.contains_key("dataType"));
        assert!(comments[0].param_docs.contains_key("n"));
        assert!(comments[0].param_docs.contains_key("c"));
        assert!(comments[0].param_docs.contains_key("h"));
        assert!(comments[0].param_docs.contains_key("w"));

        assert_eq!(
            comments[0].param_docs.get("dataType").unwrap().description,
            "image data type"
        );
        assert_eq!(
            comments[0].param_docs.get("n").unwrap().description,
            "number of inputs (batch size)"
        );
        assert_eq!(
            comments[0].param_docs.get("c").unwrap().description,
            "number of input feature maps"
        );
    }

    #[test]
    fn test_inline_comments_with_block_comment() {
        let parser = HeaderCommentParser::new().unwrap();
        let content = r#"
/* Creates a 4D tensor descriptor */
cudnnStatus_t cudnnSetTensor4dDescriptor(
    cudnnTensorDescriptor_t tensorDesc,
    cudnnDataType_t dataType, /* image data type */
    int n,                    /* number of inputs */
    int c);                   /* number of channels */
"#;

        let comments = parser.parse_header_content(content).unwrap();

        assert_eq!(comments.len(), 1);
        assert_eq!(comments[0].function_name, "cudnnSetTensor4dDescriptor");

        // Should have both brief from block comment and inline param docs
        assert!(comments[0].brief.is_some());
        assert!(
            comments[0]
                .brief
                .as_ref()
                .unwrap()
                .contains("Creates a 4D tensor descriptor")
        );

        assert!(comments[0].param_docs.contains_key("dataType"));
        assert!(comments[0].param_docs.contains_key("n"));
        assert!(comments[0].param_docs.contains_key("c"));
    }

    #[test]
    fn test_inline_comments_mixed_styles() {
        let parser = HeaderCommentParser::new().unwrap();
        let content = r#"
void* memcpy_safe(void* dest,      /* destination buffer */
                  const void* src,  /* source buffer */
                  size_t n);        /* number of bytes */
"#;

        let comments = parser.parse_header_content(content).unwrap();

        assert_eq!(comments.len(), 1);
        assert_eq!(comments[0].function_name, "memcpy_safe");
        assert_eq!(comments[0].param_docs.len(), 3);

        assert_eq!(
            comments[0].param_docs.get("dest").unwrap().description,
            "destination buffer"
        );
        assert_eq!(
            comments[0].param_docs.get("src").unwrap().description,
            "source buffer"
        );
        assert_eq!(
            comments[0].param_docs.get("n").unwrap().description,
            "number of bytes"
        );
    }

    #[test]
    fn test_extract_param_name() {
        let parser = HeaderCommentParser::new().unwrap();

        assert_eq!(parser.extract_param_name("int n,").unwrap(), "n");
        assert_eq!(
            parser.extract_param_name("const char* name)").unwrap(),
            "name"
        );
        assert_eq!(
            parser.extract_param_name("void **handle").unwrap(),
            "handle"
        );
        assert_eq!(parser.extract_param_name("size_t size").unwrap(), "size");
        assert_eq!(parser.extract_param_name("int array[10]").unwrap(), "array");
    }
}
