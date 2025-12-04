//! Changelog parser for extracting version history, breaking changes, and migration guides.
//!
//! This module parses CHANGELOG.md, HISTORY.txt, and similar files to extract:
//! - Version-specific changes
//! - Breaking changes with migration guides
//! - Deprecation warnings
//! - API additions and removals

use std::collections::HashMap;

/// Represents a single version entry in the changelog
#[derive(Debug, Clone, PartialEq)]
pub struct ChangelogEntry {
    /// Version number (e.g., "9.0.0")
    pub version: String,
    /// Release date if available
    pub date: Option<String>,
    /// Breaking changes in this version
    pub breaking_changes: Vec<BreakingChange>,
    /// New features added
    pub features: Vec<String>,
    /// Bug fixes
    pub fixes: Vec<String>,
    /// Deprecations introduced
    pub deprecations: Vec<DeprecationInfo>,
    /// Performance improvements
    pub performance: Vec<String>,
    /// Raw changelog text for this version
    pub raw_text: String,
}

/// Represents a breaking change with migration information
#[derive(Debug, Clone, PartialEq)]
pub struct BreakingChange {
    /// Description of what changed
    pub description: String,
    /// Old API example (before the breaking change)
    pub old_api: Option<String>,
    /// New API example (after the breaking change)
    pub new_api: Option<String>,
    /// Migration guide text
    pub migration_guide: Option<String>,
    /// Affected functions or types
    pub affected_items: Vec<String>,
    /// Reason for the change
    pub reason: Option<String>,
}

/// Represents a deprecation warning
#[derive(Debug, Clone, PartialEq)]
pub struct DeprecationInfo {
    /// Deprecated item name
    pub item: String,
    /// Version when deprecated
    pub since: String,
    /// Version when it will be removed
    pub removal_version: Option<String>,
    /// Replacement API
    pub replacement: Option<String>,
    /// Deprecation reason
    pub reason: Option<String>,
    /// Migration example
    pub migration: Option<String>,
}

/// Represents a migration guide between versions
#[derive(Debug, Clone, PartialEq)]
pub struct MigrationGuide {
    /// Source version (upgrading from)
    pub from_version: String,
    /// Target version (upgrading to)
    pub to_version: String,
    /// Step-by-step migration instructions
    pub steps: Vec<String>,
    /// Code examples showing before/after
    pub examples: Vec<(String, String)>, // (old_code, new_code)
    /// Known issues during migration
    pub known_issues: Vec<String>,
}

/// Main changelog parser
#[derive(Debug)]
pub struct ChangelogParser {
    /// Cache of parsed changelogs by file path
    cache: HashMap<String, Vec<ChangelogEntry>>,
}

impl ChangelogParser {
    /// Creates a new changelog parser
    pub fn new() -> Self {
        Self {
            cache: HashMap::new(),
        }
    }

    /// Parses a changelog file and returns version entries
    ///
    /// # Arguments
    /// * `path` - Path to the changelog file (for caching)
    /// * `content` - Content of the changelog file
    ///
    /// # Returns
    /// Vector of changelog entries, ordered from newest to oldest
    pub fn parse(&mut self, path: &str, content: &str) -> Vec<ChangelogEntry> {
        // Check cache first
        if let Some(cached) = self.cache.get(path) {
            return cached.clone();
        }

        let mut entries = Vec::new();
        let lines: Vec<&str> = content.lines().collect();
        let mut i = 0;

        while i < lines.len() {
            // Look for version headers
            if let Some(version) = self.extract_version(lines[i]) {
                let (entry, next_idx) = self.parse_version_section(&lines, i, &version);
                entries.push(entry);
                i = next_idx;
            } else {
                i += 1;
            }
        }

        // Cache the result
        self.cache.insert(path.to_string(), entries.clone());
        entries
    }

    /// Extracts version number from a line
    ///
    /// Handles formats like:
    /// - "## [9.0.0] - 2024-01-15"
    /// - "Version 9.0.0 (January 15, 2024)"
    /// - "v9.0.0"
    /// - "9.0.0"
    fn extract_version(&self, line: &str) -> Option<String> {
        let line = line.trim();
        
        // Match "## [9.0.0]" or "## 9.0.0"
        if line.starts_with("##") || line.starts_with("Version") || line.starts_with("v") {
            // Extract version pattern: X.Y.Z or X.Y
            let parts: Vec<&str> = line.split_whitespace().collect();
            for part in parts {
                let clean = part.trim_matches(|c: char| !c.is_ascii_digit() && c != '.');
                if self.is_valid_version(clean) {
                    return Some(clean.to_string());
                }
            }
        }
        
        None
    }

    /// Checks if a string looks like a valid version number
    fn is_valid_version(&self, s: &str) -> bool {
        let parts: Vec<&str> = s.split('.').collect();
        parts.len() >= 2 && parts.iter().all(|p| p.chars().all(|c| c.is_ascii_digit()))
    }

    /// Parses a complete version section
    fn parse_version_section(
        &self,
        lines: &[&str],
        start_idx: usize,
        version: &str,
    ) -> (ChangelogEntry, usize) {
        let mut entry = ChangelogEntry {
            version: version.to_string(),
            date: self.extract_date(lines[start_idx]),
            breaking_changes: Vec::new(),
            features: Vec::new(),
            fixes: Vec::new(),
            deprecations: Vec::new(),
            performance: Vec::new(),
            raw_text: String::new(),
        };

        let mut i = start_idx + 1;
        let mut raw_lines = vec![lines[start_idx]];
        let mut current_section = SectionType::None;

        // Parse until next version or end of file
        while i < lines.len() {
            let line = lines[i].trim();

            // Stop if we hit another version
            if self.extract_version(lines[i]).is_some() {
                break;
            }

            raw_lines.push(lines[i]);

            // Identify section headers
            if line.starts_with("###") || line.starts_with("**") {
                current_section = self.identify_section(line);
                i += 1;
                continue;
            }

            // Parse content based on section
            match current_section {
                SectionType::Breaking => {
                    if let Some(change) = self.parse_breaking_change(lines, &mut i) {
                        entry.breaking_changes.push(change);
                    }
                }
                SectionType::Features => {
                    if let Some(feature) = self.parse_list_item(line) {
                        entry.features.push(feature);
                    }
                }
                SectionType::Fixes => {
                    if let Some(fix) = self.parse_list_item(line) {
                        entry.fixes.push(fix);
                    }
                }
                SectionType::Deprecations => {
                    if let Some(dep) = self.parse_deprecation(line) {
                        entry.deprecations.push(dep);
                    }
                }
                SectionType::Performance => {
                    if let Some(perf) = self.parse_list_item(line) {
                        entry.performance.push(perf);
                    }
                }
                SectionType::None => {}
            }

            i += 1;
        }

        entry.raw_text = raw_lines.join("\n");
        (entry, i)
    }

    /// Extracts date from a version header line
    fn extract_date(&self, line: &str) -> Option<String> {
        // Match "- 2024-01-15" or "(January 15, 2024)"
        // Look for pattern: "- YYYY-MM-DD"
        if let Some(pos) = line.find(" - ") {
            let after_dash = &line[pos + 3..].trim();
            // Take first word-like sequence (the date)
            if let Some(date_end) = after_dash.find(|c: char| c.is_whitespace()) {
                let date = &after_dash[..date_end];
                // Validate it looks like a date (YYYY-MM-DD format)
                if date.matches('-').count() == 2 {
                    return Some(date.to_string());
                }
            } else if after_dash.matches('-').count() == 2 {
                // No whitespace after, entire string is date
                return Some(after_dash.to_string());
            }
        }
        None
    }

    /// Identifies the section type from a header
    fn identify_section(&self, line: &str) -> SectionType {
        let lower = line.to_lowercase();
        
        if lower.contains("breaking") {
            SectionType::Breaking
        } else if lower.contains("feature") || lower.contains("added") {
            SectionType::Features
        } else if lower.contains("fix") || lower.contains("bug") {
            SectionType::Fixes
        } else if lower.contains("deprecat") {
            SectionType::Deprecations
        } else if lower.contains("performance") || lower.contains("optim") {
            SectionType::Performance
        } else {
            SectionType::None
        }
    }

    /// Parses a breaking change with code examples
    fn parse_breaking_change(&self, lines: &[&str], idx: &mut usize) -> Option<BreakingChange> {
        let line = lines[*idx].trim();
        
        if !line.starts_with('-') && !line.starts_with('*') {
            return None;
        }

        let description = self.parse_list_item(line)?;
        let mut change = BreakingChange {
            description: description.clone(),
            old_api: None,
            new_api: None,
            migration_guide: None,
            affected_items: Vec::new(),
            reason: None,
        };

        // Extract affected items from the description line
        let items = self.extract_backtick_items(line);
        change.affected_items.extend(items);

        // Look ahead for code examples or migration guides
        *idx += 1;
        let mut last_context = String::new();
        
        while *idx < lines.len() {
            let next_line = lines[*idx].trim();
            
            // Stop at next list item or section header
            if next_line.starts_with('-') || next_line.starts_with('*') || next_line.starts_with("##") {
                *idx -= 1;
                break;
            }

            // Track context lines before code blocks
            if !next_line.is_empty() && !next_line.starts_with("```") {
                last_context = next_line.to_lowercase();
            }

            // Extract code blocks
            if next_line.starts_with("```") {
                let (code, end_idx) = self.extract_code_block(lines, *idx);
                *idx = end_idx;
                
                // Determine if this is old or new API based on last context line
                if last_context.contains("old") || last_context.contains("before") || last_context.contains("❌") {
                    change.old_api = Some(code);
                } else if last_context.contains("new") || last_context.contains("after") || last_context.contains("✅") {
                    change.new_api = Some(code);
                } else if change.old_api.is_none() {
                    // First code block without clear context = old API
                    change.old_api = Some(code);
                } else {
                    // Second code block = new API
                    change.new_api = Some(code);
                }
            }

            // Extract more affected items from additional lines
            if next_line.contains('`') {
                let items = self.extract_backtick_items(next_line);
                change.affected_items.extend(items);
            }

            *idx += 1;
        }

        Some(change)
    }

    /// Parses a list item (removes bullet points)
    fn parse_list_item(&self, line: &str) -> Option<String> {
        let trimmed = line.trim();
        if trimmed.starts_with('-') || trimmed.starts_with('*') {
            let content = trimmed.trim_start_matches('-').trim_start_matches('*').trim();
            if !content.is_empty() {
                return Some(content.to_string());
            }
        }
        None
    }

    /// Parses a deprecation notice
    fn parse_deprecation(&self, line: &str) -> Option<DeprecationInfo> {
        let content = self.parse_list_item(line)?;
        
        // Extract item name (usually first backtick-wrapped word)
        let items = self.extract_backtick_items(&content);
        let item = items.first()?.clone();
        
        let mut deprecation = DeprecationInfo {
            item: item.clone(),
            since: String::new(),
            removal_version: None,
            replacement: None,
            reason: None,
            migration: None,
        };

        // Extract replacement (often second backtick-wrapped word)
        if items.len() > 1 {
            deprecation.replacement = Some(items[1].clone());
        }

        // Extract version info
        if (content.contains("removal in") || content.contains("removed in"))
            && let Some(version) = self.extract_removal_version(&content) {
                deprecation.removal_version = Some(version);
            }

        Some(deprecation)
    }

    /// Extracts removal version from text
    fn extract_removal_version(&self, text: &str) -> Option<String> {
        // Look for "removal in 10.0" or "removed in v10.0"
        let lower = text.to_lowercase();
        if let Some(pos) = lower.find("removal in").or_else(|| lower.find("removed in")) {
            let after = &text[pos..];
            let parts: Vec<&str> = after.split_whitespace().collect();
            for part in parts.iter().skip(2) {
                let clean = part.trim_matches(|c: char| !c.is_ascii_digit() && c != '.');
                if self.is_valid_version(clean) {
                    return Some(clean.to_string());
                }
            }
        }
        None
    }

    /// Extracts code block content
    fn extract_code_block(&self, lines: &[&str], start_idx: usize) -> (String, usize) {
        let mut code = String::new();
        let mut i = start_idx + 1; // Skip opening ```
        
        while i < lines.len() {
            if lines[i].trim().starts_with("```") {
                return (code.trim().to_string(), i);
            }
            code.push_str(lines[i]);
            code.push('\n');
            i += 1;
        }
        
        (code.trim().to_string(), i)
    }

    /// Extracts all backtick-wrapped items from text
    fn extract_backtick_items(&self, text: &str) -> Vec<String> {
        let mut items = Vec::new();
        let mut in_backtick = false;
        let mut current = String::new();
        
        for ch in text.chars() {
            if ch == '`' {
                if in_backtick
                    && !current.is_empty() {
                        items.push(current.clone());
                        current.clear();
                    }
                in_backtick = !in_backtick;
            } else if in_backtick {
                current.push(ch);
            }
        }
        
        items
    }

    /// Finds the changelog entry for a specific version
    pub fn get_version<'a>(&self, entries: &'a [ChangelogEntry], version: &str) -> Option<&'a ChangelogEntry> {
        entries.iter().find(|e| e.version == version)
    }

    /// Generates a migration guide between two versions
    pub fn generate_migration_guide(
        &self,
        entries: &[ChangelogEntry],
        from_version: &str,
        to_version: &str,
    ) -> Option<MigrationGuide> {
        // Find all versions between from and to
        // Entries are ordered newest to oldest, so to_idx < from_idx when upgrading
        let from_idx = entries.iter().position(|e| e.version == from_version)?;
        let to_idx = entries.iter().position(|e| e.version == to_version)?;
        
        if from_idx <= to_idx {
            return None; // Invalid version range (can't downgrade)
        }

        let mut guide = MigrationGuide {
            from_version: from_version.to_string(),
            to_version: to_version.to_string(),
            steps: Vec::new(),
            examples: Vec::new(),
            known_issues: Vec::new(),
        };

        // Collect all breaking changes in the range (from newest to from_version)
        for entry in &entries[to_idx..=from_idx] {
            for breaking in &entry.breaking_changes {
                guide.steps.push(format!(
                    "Version {}: {}",
                    entry.version, breaking.description
                ));
                
                if let (Some(old), Some(new)) = (&breaking.old_api, &breaking.new_api) {
                    guide.examples.push((old.clone(), new.clone()));
                }
            }
        }

        Some(guide)
    }
}

impl Default for ChangelogParser {
    fn default() -> Self {
        Self::new()
    }
}

/// Section types in changelog
#[derive(Debug, Clone, Copy, PartialEq)]
enum SectionType {
    None,
    Breaking,
    Features,
    Fixes,
    Deprecations,
    Performance,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_version() {
        let mut parser = ChangelogParser::new();
        let changelog = r#"
## [9.0.0] - 2024-01-15

### Breaking Changes
- `data_type` parameter is now required
- Default data type changed from Int8 to Float
"#;

        let entries = parser.parse("test.md", changelog);
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].version, "9.0.0");
        assert_eq!(entries[0].date, Some("2024-01-15".to_string()));
        assert_eq!(entries[0].breaking_changes.len(), 2);
    }

    #[test]
    fn test_extract_version_various_formats() {
        let parser = ChangelogParser::new();
        
        assert_eq!(parser.extract_version("## [9.0.0] - 2024-01-15"), Some("9.0.0".to_string()));
        assert_eq!(parser.extract_version("Version 9.0.0 (January 15, 2024)"), Some("9.0.0".to_string()));
        assert_eq!(parser.extract_version("v9.0.0"), Some("9.0.0".to_string()));
        assert_eq!(parser.extract_version("## 9.0"), Some("9.0".to_string()));
        assert_eq!(parser.extract_version("Not a version"), None);
    }

    #[test]
    fn test_parse_breaking_change_with_code() {
        let mut parser = ChangelogParser::new();
        let changelog = r#"
## [9.0.0]

### Breaking Changes
- `new()` constructor removed - use builder pattern

Old API:
```rust
let desc = TensorDescriptor::new(dims)?;
```

New API:
```rust
let desc = TensorDescriptor::builder()
    .dimensions(dims)
    .build()?;
```
"#;

        let entries = parser.parse("test.md", changelog);
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].breaking_changes.len(), 1);
        
        let breaking = &entries[0].breaking_changes[0];
        assert!(breaking.old_api.is_some());
        assert!(breaking.new_api.is_some());
        assert!(breaking.old_api.as_ref().unwrap().contains("new(dims)"));
        assert!(breaking.new_api.as_ref().unwrap().contains("builder()"));
    }

    #[test]
    fn test_parse_deprecation() {
        let mut parser = ChangelogParser::new();
        let changelog = r#"
## [9.0.0]

### Deprecations
- `set_format()` deprecated - use `with_format()` instead (removal in 10.0)
"#;

        let entries = parser.parse("test.md", changelog);
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].deprecations.len(), 1);
        
        let dep = &entries[0].deprecations[0];
        assert_eq!(dep.item, "set_format()");
        assert_eq!(dep.replacement, Some("with_format()".to_string()));
        assert_eq!(dep.removal_version, Some("10.0".to_string()));
    }

    #[test]
    fn test_parse_multiple_sections() {
        let mut parser = ChangelogParser::new();
        let changelog = r#"
## [9.0.0] - 2024-01-15

### Features
- Added new `builder()` method
- Support for asynchronous operations

### Bug Fixes
- Fixed memory leak in `cleanup()`
- Resolved race condition

### Performance
- 2x faster tensor operations
"#;

        let entries = parser.parse("test.md", changelog);
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].features.len(), 2);
        assert_eq!(entries[0].fixes.len(), 2);
        assert_eq!(entries[0].performance.len(), 1);
    }

    #[test]
    fn test_parse_multiple_versions() {
        let mut parser = ChangelogParser::new();
        let changelog = r#"
## [9.0.0] - 2024-01-15

### Breaking Changes
- Breaking change in 9.0

## [8.0.0] - 2023-12-01

### Features
- New feature in 8.0
"#;

        let entries = parser.parse("test.md", changelog);
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].version, "9.0.0");
        assert_eq!(entries[1].version, "8.0.0");
    }

    #[test]
    fn test_extract_backtick_items() {
        let parser = ChangelogParser::new();
        let text = "The `old_func()` is replaced by `new_func()` in version 2.0";
        let items = parser.extract_backtick_items(text);
        
        assert_eq!(items.len(), 2);
        assert_eq!(items[0], "old_func()");
        assert_eq!(items[1], "new_func()");
    }

    #[test]
    fn test_generate_migration_guide() {
        let mut parser = ChangelogParser::new();
        let changelog = r#"
## [10.0.0]

### Breaking Changes
- Removed deprecated `old_api()`

## [9.0.0]

### Breaking Changes
- Changed signature of `process()`

Old:
```rust
process(data)
```

New:
```rust
process(data, config)
```
"#;

        let entries = parser.parse("test.md", changelog);
        let guide = parser.generate_migration_guide(&entries, "9.0.0", "10.0.0");
        
        assert!(guide.is_some());
        let guide = guide.unwrap();
        assert_eq!(guide.from_version, "9.0.0");
        assert_eq!(guide.to_version, "10.0.0");
        assert!(!guide.steps.is_empty());
    }

    #[test]
    fn test_cache() {
        let mut parser = ChangelogParser::new();
        let changelog = "## [1.0.0]\n### Features\n- Test";
        
        // First parse
        let entries1 = parser.parse("test.md", changelog);
        assert_eq!(entries1.len(), 1);
        
        // Second parse should use cache
        let entries2 = parser.parse("test.md", changelog);
        assert_eq!(entries2.len(), 1);
        assert_eq!(entries1, entries2);
    }

    #[test]
    fn test_extract_affected_items() {
        let mut parser = ChangelogParser::new();
        let changelog = r#"
## [9.0.0]

### Breaking Changes
- The `old_func()` and `deprecated_api()` were removed
"#;

        let entries = parser.parse("test.md", changelog);
        let breaking = &entries[0].breaking_changes[0];
        
        assert_eq!(breaking.affected_items.len(), 2);
        assert!(breaking.affected_items.contains(&"old_func()".to_string()));
        assert!(breaking.affected_items.contains(&"deprecated_api()".to_string()));
    }
}
