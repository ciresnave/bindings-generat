//! API sequence and ordering analyzer.
//!
//! This module analyzes documentation to extract:
//! - "Must call X before Y" relationships
//! - Mutually exclusive API calls
//! - State machine patterns
//! - Alternative/equivalent implementations

use std::collections::HashMap;

/// Information about API call ordering requirements
#[derive(Debug, Clone, PartialEq)]
pub struct ApiSequence {
    /// Functions that must be called before this one
    pub prerequisites: Vec<String>,
    /// Functions that cannot be called after this one
    pub invalidates: Vec<String>,
    /// Functions that must be called after this one (for cleanup, etc.)
    pub requires_followup: Vec<String>,
    /// Functions that are mutually exclusive with this one
    pub mutually_exclusive: Vec<String>,
    /// Alternative functions that provide similar functionality
    pub alternatives: Vec<String>,
    /// State transitions this function participates in
    pub state_transitions: Vec<StateTransition>,
    /// Additional ordering notes
    pub notes: Vec<String>,
}

/// Represents a state transition in an API state machine
#[derive(Debug, Clone, PartialEq)]
pub struct StateTransition {
    /// State before calling this function
    pub from_state: String,
    /// State after calling this function
    pub to_state: String,
    /// Conditions for this transition
    pub conditions: Vec<String>,
}

impl ApiSequence {
    /// Creates an empty API sequence
    pub fn new() -> Self {
        Self {
            prerequisites: Vec::new(),
            invalidates: Vec::new(),
            requires_followup: Vec::new(),
            mutually_exclusive: Vec::new(),
            alternatives: Vec::new(),
            state_transitions: Vec::new(),
            notes: Vec::new(),
        }
    }

    /// Checks if this function has any ordering requirements
    pub fn has_requirements(&self) -> bool {
        !self.prerequisites.is_empty()
            || !self.invalidates.is_empty()
            || !self.requires_followup.is_empty()
            || !self.mutually_exclusive.is_empty()
    }

    /// Generates documentation for API sequencing
    pub fn generate_documentation(&self) -> String {
        if !self.has_requirements()
            && self.alternatives.is_empty()
            && self.state_transitions.is_empty()
        {
            return String::new();
        }

        let mut docs = String::from("/// # API Sequencing\n");
        docs.push_str("///\n");

        // Prerequisites
        if !self.prerequisites.is_empty() {
            docs.push_str("/// ## Prerequisites\n");
            docs.push_str("///\n");
            docs.push_str("/// The following functions must be called before this one:\n");
            for prereq in &self.prerequisites {
                docs.push_str(&format!("/// - `{}`\n", prereq));
            }
            docs.push_str("///\n");
        }

        // Follow-up required
        if !self.requires_followup.is_empty() {
            docs.push_str("/// ## Required Follow-up\n");
            docs.push_str("///\n");
            docs.push_str("/// After calling this function, you must call:\n");
            for followup in &self.requires_followup {
                docs.push_str(&format!("/// - `{}`\n", followup));
            }
            docs.push_str("///\n");
        }

        // Invalidates
        if !self.invalidates.is_empty() {
            docs.push_str("/// ## Invalidates\n");
            docs.push_str("///\n");
            docs.push_str("/// This function invalidates/renders unusable:\n");
            for invalid in &self.invalidates {
                docs.push_str(&format!("/// - `{}`\n", invalid));
            }
            docs.push_str("///\n");
        }

        // Mutually exclusive
        if !self.mutually_exclusive.is_empty() {
            docs.push_str("/// ## Mutually Exclusive\n");
            docs.push_str("///\n");
            docs.push_str("/// Cannot be used with:\n");
            for excl in &self.mutually_exclusive {
                docs.push_str(&format!("/// - `{}`\n", excl));
            }
            docs.push_str("///\n");
        }

        // Alternatives
        if !self.alternatives.is_empty() {
            docs.push_str("/// ## Alternatives\n");
            docs.push_str("///\n");
            docs.push_str("/// Similar functionality provided by:\n");
            for alt in &self.alternatives {
                docs.push_str(&format!("/// - `{}`\n", alt));
            }
            docs.push_str("///\n");
        }

        // State transitions
        if !self.state_transitions.is_empty() {
            docs.push_str("/// ## State Transitions\n");
            docs.push_str("///\n");
            for transition in &self.state_transitions {
                docs.push_str(&format!(
                    "/// - {} → {}\n",
                    transition.from_state, transition.to_state
                ));
                for condition in &transition.conditions {
                    docs.push_str(&format!("///   (if {})\n", condition));
                }
            }
            docs.push_str("///\n");
        }

        // Additional notes
        for note in &self.notes {
            docs.push_str(&format!("/// - {}\n", note));
        }

        docs
    }
}

impl Default for ApiSequence {
    fn default() -> Self {
        Self::new()
    }
}

/// Main API sequence analyzer
#[derive(Debug)]
pub struct ApiSequenceAnalyzer {
    /// Cache of analyzed sequences by function name
    cache: HashMap<String, ApiSequence>,
}

impl ApiSequenceAnalyzer {
    /// Creates a new API sequence analyzer
    pub fn new() -> Self {
        Self {
            cache: HashMap::new(),
        }
    }

    /// Analyzes API sequencing requirements from documentation
    ///
    /// # Arguments
    /// * `function_name` - Name of the function
    /// * `documentation` - Combined documentation text
    ///
    /// # Returns
    /// API sequence information
    pub fn analyze(&mut self, function_name: &str, documentation: &str) -> ApiSequence {
        // Check cache
        if let Some(cached) = self.cache.get(function_name) {
            return cached.clone();
        }

        let mut sequence = ApiSequence::new();

        // Extract various relationships
        sequence.prerequisites = self.extract_prerequisites(documentation);
        sequence.invalidates = self.extract_invalidates(documentation);
        sequence.requires_followup = self.extract_followup(documentation);
        sequence.mutually_exclusive = self.extract_mutually_exclusive(documentation);
        sequence.alternatives = self.extract_alternatives(documentation);
        sequence.state_transitions = self.extract_state_transitions(documentation);
        sequence.notes = self.extract_sequencing_notes(documentation);

        // Cache the result
        self.cache
            .insert(function_name.to_string(), sequence.clone());
        sequence
    }

    /// Extracts prerequisite functions that must be called first
    fn extract_prerequisites(&self, docs: &str) -> Vec<String> {
        let mut prereqs = Vec::new();

        for line in docs.lines() {
            let lower = line.to_lowercase();

            // Pattern: "must call X before" or "X must be called first"
            if lower.contains("must") && lower.contains("before") {
                // For "before", function name comes BEFORE the keyword
                if let Some(func) = self.extract_function_before_keyword(line, "before")
                    && !prereqs.contains(&func) {
                        prereqs.push(func);
                    }
            }

            // Pattern: "after calling X"
            if (lower.contains("after calling") || lower.contains("after") && lower.contains("call"))
                && let Some(func) = self.extract_function_name(line, "after")
                    && !prereqs.contains(&func) {
                        prereqs.push(func);
                    }

            // Pattern: "requires X to be called first"
            if lower.contains("requires")
                && (lower.contains("first") || lower.contains("initialized"))
                && let Some(func) = self.extract_function_name(line, "requires")
                    && !prereqs.contains(&func) {
                        prereqs.push(func);
                    }
        }

        prereqs
    }

    /// Extracts functions/resources that this call invalidates
    fn extract_invalidates(&self, docs: &str) -> Vec<String> {
        let mut invalidates = Vec::new();

        for line in docs.lines() {
            let lower = line.to_lowercase();

            // Pattern: "invalidates X"
            if lower.contains("invalidate")
                && let Some(func) = self.extract_function_name(line, "invalidate")
                    && !invalidates.contains(&func) {
                        invalidates.push(func);
                    }

            // Pattern: "makes X unusable"
            if lower.contains("makes") && lower.contains("unusable")
                && let Some(func) = self.extract_function_name(line, "makes")
                    && !invalidates.contains(&func) {
                        invalidates.push(func);
                    }

            // Pattern: "X can no longer be used"
            if (lower.contains("no longer") || lower.contains("cannot be used"))
                && let Some(func) = self.extract_function_name(line, "no longer")
                    && !invalidates.contains(&func) {
                        invalidates.push(func);
                    }
        }

        invalidates
    }

    /// Extracts functions that must be called after this one
    fn extract_followup(&self, docs: &str) -> Vec<String> {
        let mut followup = Vec::new();

        for line in docs.lines() {
            let lower = line.to_lowercase();

            // Pattern: "must call X after" or "X must be called after"  or "after X, must call Y"
            if lower.contains("after") && lower.contains("call") && !lower.contains("before") {
                // Look for function after "call"
                if let Some(func) = self.extract_function_name(line, "call")
                    && !followup.contains(&func) {
                        followup.push(func);
                    }
            }

            // Pattern: "requires cleanup with X"
            if (lower.contains("cleanup") || lower.contains("clean up"))
                && let Some(func) = self.extract_function_name(line, "cleanup")
                    && !followup.contains(&func) {
                        followup.push(func);
                    }

            // Pattern: "followed by X"
            if lower.contains("followed by")
                && let Some(func) = self.extract_function_name(line, "followed by")
                    && !followup.contains(&func) {
                        followup.push(func);
                    }
        }

        followup
    }

    /// Extracts mutually exclusive functions
    fn extract_mutually_exclusive(&self, docs: &str) -> Vec<String> {
        let mut exclusive = Vec::new();

        for line in docs.lines() {
            let lower = line.to_lowercase();

            // Pattern: "cannot be used with X" or "incompatible with X"
            if (lower.contains("cannot") && lower.contains("with") || lower.contains("incompatible"))
                && let Some(func) = self.extract_function_name(line, "with")
                    && !exclusive.contains(&func) {
                        exclusive.push(func);
                    }

            // Pattern: "mutually exclusive with X"
            if lower.contains("mutually exclusive")
                && let Some(func) = self.extract_function_name(line, "exclusive")
                    && !exclusive.contains(&func) {
                        exclusive.push(func);
                    }

            // Pattern: "either X or Y"
            if lower.contains("either") && lower.contains("or") {
                // This is complex - would need more sophisticated parsing
                // For now, skip
            }
        }

        exclusive
    }

    /// Extracts alternative functions with similar functionality
    fn extract_alternatives(&self, docs: &str) -> Vec<String> {
        let mut alternatives = Vec::new();

        for line in docs.lines() {
            let lower = line.to_lowercase();

            // Pattern: "use X instead"
            if lower.contains("instead")
                && let Some(func) = self.extract_function_name(line, "use")
                    && !alternatives.contains(&func) {
                        alternatives.push(func);
                    }

            // Pattern: "see also X"
            if lower.contains("see also")
                && let Some(func) = self.extract_function_name(line, "also")
                    && !alternatives.contains(&func) {
                        alternatives.push(func);
                    }

            // Pattern: "similar to X"
            if lower.contains("similar to")
                && let Some(func) = self.extract_function_name(line, "similar to")
                    && !alternatives.contains(&func) {
                        alternatives.push(func);
                    }
        }

        alternatives
    }

    /// Extracts state machine transitions
    fn extract_state_transitions(&self, docs: &str) -> Vec<StateTransition> {
        let mut transitions = Vec::new();

        for line in docs.lines() {
            let lower = line.to_lowercase();

            // Pattern: "transitions from X to Y" or "X -> Y"
            if lower.contains("transition") && lower.contains("from") && lower.contains("to") {
                // Try to extract states
                if let Some(from) = self.extract_state_name(line, "from")
                    && let Some(to) = self.extract_state_name(line, "to") {
                        transitions.push(StateTransition {
                            from_state: from,
                            to_state: to,
                            conditions: Vec::new(),
                        });
                    }
            }

            // Pattern: "moves to X state"
            if (lower.contains("moves to") || lower.contains("enters"))
                && let Some(to) = self.extract_state_name(line, "to") {
                    transitions.push(StateTransition {
                        from_state: "any".to_string(),
                        to_state: to,
                        conditions: Vec::new(),
                    });
                }
        }

        transitions
    }

    /// Extracts additional sequencing notes
    fn extract_sequencing_notes(&self, docs: &str) -> Vec<String> {
        let mut notes = Vec::new();

        for line in docs.lines() {
            let lower = line.to_lowercase();

            // Look for important sequencing information
            if (lower.contains("order") || lower.contains("sequence") || lower.contains("timing"))
                && (lower.contains("important")
                    || lower.contains("note")
                    || lower.contains("must")
                    || lower.contains("warning"))
            {
                let trimmed = line.trim();
                if !trimmed.is_empty() && !notes.contains(&trimmed.to_string()) {
                    notes.push(trimmed.to_string());
                }
            }
        }

        notes
    }

    /// Extracts a function name from a line near a keyword
    fn extract_function_name(&self, line: &str, near_keyword: &str) -> Option<String> {
        let lower = line.to_lowercase();

        // Find the keyword position
        let keyword_pos = if let Some(pos) = lower.find(near_keyword) {
            pos
        } else {
            0 // If keyword not found, search whole line
        };

        // Search primarily AFTER the keyword
        let after_keyword = &line[keyword_pos..];

        // First pass: Look for explicit function calls with ()
        for word in after_keyword.split_whitespace() {
            let cleaned = word
                .trim_matches(|c: char| !c.is_alphanumeric() && c != '_' && c != '(' && c != ')');

            if let Some(func_name) = cleaned.strip_suffix("()") {
                if func_name.len() >= 3 {
                    return Some(func_name.to_string());
                }
            }
        }

        // Second pass: Look for identifier-like words
        for word in after_keyword.split_whitespace() {
            let clean_alpha = word.trim_matches(|c: char| !c.is_alphanumeric() && c != '_');

            // Skip articles and common prepositions
            if matches!(clean_alpha, "the" | "a" | "an" | "and" | "or" | "it") {
                continue;
            }

            // Relax condition - accept any word >= 3 chars that's not common
            if clean_alpha.len() >= 3 && !self.is_common_word(clean_alpha) {
                return Some(clean_alpha.to_string());
            }
        }

        None
    }

    /// Extracts a function name that appears BEFORE a keyword
    fn extract_function_before_keyword(&self, line: &str, keyword: &str) -> Option<String> {
        let lower = line.to_lowercase();

        // Find the keyword position
        let keyword_pos = lower.find(keyword)?;

        // Search BEFORE the keyword
        let before_keyword = &line[..keyword_pos];

        // First pass: Look for explicit function calls with () (search from end)
        for word in before_keyword.split_whitespace().rev() {
            let cleaned = word
                .trim_matches(|c: char| !c.is_alphanumeric() && c != '_' && c != '(' && c != ')');

            if let Some(func_name) = cleaned.strip_suffix("()") {
                if func_name.len() >= 3 {
                    return Some(func_name.to_string());
                }
            }
        }

        // Second pass: Look for identifier-like words (search from end)
        for word in before_keyword.split_whitespace().rev() {
            let clean_alpha = word.trim_matches(|c: char| !c.is_alphanumeric() && c != '_');

            // Skip articles and common prepositions
            if matches!(clean_alpha, "the" | "a" | "an" | "and" | "or" | "it") {
                continue;
            }

            // Check if word looks like an identifier (function, variable, type, etc.)
            if self.looks_like_identifier(clean_alpha) {
                return Some(clean_alpha.to_string());
            }

            // Fallback: accept any word >= 3 chars that's not common
            if clean_alpha.len() >= 3 && !self.is_common_word(clean_alpha) {
                return Some(clean_alpha.to_string());
            }
        }

        None
    }

    /// Checks if a word looks like an identifier (function, variable, type, etc.)
    fn looks_like_identifier(&self, word: &str) -> bool {
        if word.len() < 3 {
            return false;
        }

        // Accept anything with underscores (snake_case)
        if word.contains('_') {
            return true;
        }

        // Accept camelCase or PascalCase
        let has_upper = word.chars().any(|c| c.is_uppercase());
        let has_lower = word.chars().any(|c| c.is_lowercase());
        if has_upper && has_lower {
            return true;
        }

        // Accept if it's all lowercase and looks technical
        if word.chars().all(|c| c.is_lowercase() || c.is_numeric()) {
            return word.len() >= 5; // Longer lowercase words are likely identifiers
        }

        false
    }

    /// Checks if a word is a common English word (not a function name)
    fn is_common_word(&self, word: &str) -> bool {
        let lower = word.to_lowercase();

        // Short words that are clearly English (but allow 4-letter technical terms)
        if word.len() < 3 {
            return true;
        }

        // Common English words that might appear in documentation
        matches!(
            lower.as_str(),
            "this"
                | "that"
                | "with"
                | "from"
                | "into"
                | "call"
                | "calling"
                | "called"
                | "function"
                | "method"
                | "before"
                | "after"
                | "must"
                | "should"
                | "will"
                | "can"
                | "may"
                | "might"
                | "then"
                | "also"
                | "only"
                | "first"
                | "last"
                | "completed"
                | "finished"
                | "done"
                | "used"
                | "using"
                | "returns"
                | "return"
                | "makes"
                | "unusable"
                | "resources"
        )
    }

    /// Extracts a state name from a line near a keyword
    fn extract_state_name(&self, line: &str, near_keyword: &str) -> Option<String> {
        let lower = line.to_lowercase();
        let keyword_pos = lower.find(near_keyword)?;

        // Search after the keyword
        let search_start = keyword_pos + near_keyword.len();
        if search_start >= line.len() {
            return None;
        }

        let search_area = &line[search_start..];

        // Look for state names (usually capitalized or ALL_CAPS)
        for word in search_area.split_whitespace() {
            let cleaned = word.trim_matches(|c: char| !c.is_alphanumeric() && c != '_');

            if self.looks_like_state_name(cleaned) {
                return Some(cleaned.to_string());
            }
        }

        None
    }

    /// Checks if a word looks like a state name
    fn looks_like_state_name(&self, word: &str) -> bool {
        if word.len() < 3 {
            return false;
        }

        // State names are often ALL_CAPS or PascalCase
        word.chars().all(|c| c.is_uppercase() || c == '_')
            || (word.chars().next().unwrap().is_uppercase()
                && word.chars().skip(1).any(|c| c.is_lowercase()))
    }
}

impl Default for ApiSequenceAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_prerequisites() {
        let mut analyzer = ApiSequenceAnalyzer::new();
        let docs = "Must call initialize() before calling this function.";

        let sequence = analyzer.analyze("test_func", docs);
        assert!(sequence.prerequisites.contains(&"initialize".to_string()));
    }

    #[test]
    fn test_extract_invalidates() {
        let mut analyzer = ApiSequenceAnalyzer::new();
        let docs = "This function invalidates the old_iterator() handle.";

        let sequence = analyzer.analyze("test_func", docs);
        assert!(sequence.invalidates.contains(&"old_iterator".to_string()));
    }

    #[test]
    fn test_extract_followup() {
        let mut analyzer = ApiSequenceAnalyzer::new();
        let docs = "After calling this function, you must call cleanup() to free resources.";

        let sequence = analyzer.analyze("test_func", docs);
        assert!(sequence.requires_followup.contains(&"cleanup".to_string()));
    }

    #[test]
    fn test_extract_mutually_exclusive() {
        let mut analyzer = ApiSequenceAnalyzer::new();
        let docs = "This function cannot be used with manual_mode().";

        let sequence = analyzer.analyze("test_func", docs);
        assert!(
            sequence
                .mutually_exclusive
                .contains(&"manual_mode".to_string())
        );
    }

    #[test]
    fn test_extract_alternatives() {
        let mut analyzer = ApiSequenceAnalyzer::new();
        let docs = "For better performance, use fast_variant() instead.";

        let sequence = analyzer.analyze("test_func", docs);
        assert!(sequence.alternatives.contains(&"fast_variant".to_string()));
    }

    #[test]
    fn test_state_transitions() {
        let mut analyzer = ApiSequenceAnalyzer::new();
        let docs = "This function transitions from IDLE to RUNNING state.";

        let sequence = analyzer.analyze("test_func", docs);
        assert_eq!(sequence.state_transitions.len(), 1);
        assert_eq!(sequence.state_transitions[0].from_state, "IDLE");
        assert_eq!(sequence.state_transitions[0].to_state, "RUNNING");
    }

    #[test]
    fn test_documentation_generation() {
        let mut analyzer = ApiSequenceAnalyzer::new();
        let docs = "Must call init() before this. After this, call cleanup().";

        let sequence = analyzer.analyze("test_func", docs);
        let doc = sequence.generate_documentation();

        assert!(doc.contains("# API Sequencing"));
        assert!(doc.contains("Prerequisites"));
        assert!(doc.contains("init"));
    }

    #[test]
    fn test_no_requirements() {
        let mut analyzer = ApiSequenceAnalyzer::new();
        let docs = "This function can be called at any time.";

        let sequence = analyzer.analyze("test_func", docs);
        assert!(!sequence.has_requirements());
    }

    #[test]
    fn test_cache() {
        let mut analyzer = ApiSequenceAnalyzer::new();
        let docs = "Must call setup() first.";

        let seq1 = analyzer.analyze("test_func", docs);
        let seq2 = analyzer.analyze("test_func", docs);

        assert_eq!(seq1, seq2);
    }

    #[test]
    fn test_complex_sequencing() {
        let mut analyzer = ApiSequenceAnalyzer::new();
        let docs = "Must call init() before calling this.\n\
                    This invalidates old_handle().\n\
                    After calling this, you must call finalize() to cleanup.\n\
                    Cannot be used with async_mode().";

        let sequence = analyzer.analyze("test_func", docs);

        assert!(
            !sequence.prerequisites.is_empty(),
            "Should have prerequisites"
        );
        assert!(!sequence.invalidates.is_empty(), "Should have invalidates");
        assert!(
            !sequence.requires_followup.is_empty(),
            "Should have followup: {:?}",
            sequence.requires_followup
        );
        assert!(
            !sequence.mutually_exclusive.is_empty(),
            "Should have mutually exclusive"
        );
    }
}
