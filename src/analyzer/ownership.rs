//! Memory Ownership and Lifetime Analysis
//!
//! This module analyzes C/C++ documentation to extract memory ownership semantics
//! and lifetime requirements, enabling generation of safe Rust wrappers with proper
//! lifetime parameters and ownership documentation.
//!
//! This is THE most critical safety feature for FFI bindings, as incorrect ownership
//! handling leads to memory leaks, use-after-free, and double-free bugs.

use std::collections::HashMap;

/// Memory ownership semantics for a value
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OwnershipSemantics {
    /// Caller owns the returned value and must free it
    /// Example: `malloc()`, object constructors
    CallerOwns,

    /// Callee takes ownership and will free it
    /// Example: Functions that consume handles
    CalleeOwns,

    /// Borrowed reference - caller retains ownership
    /// Example: Functions that read but don't modify
    Borrowed,

    /// Shared ownership - reference counted or managed
    /// Example: Ref-counted objects, static data
    Shared,

    /// Ownership transfer from callee to caller
    /// Example: Functions that return newly allocated objects
    TransferToCallee,

    /// Unknown ownership semantics
    Unknown,
}

impl OwnershipSemantics {
    /// Get a human-readable description
    pub fn description(&self) -> &'static str {
        match self {
            Self::CallerOwns => "CALLER OWNS RESULT",
            Self::CalleeOwns => "CALLEE TAKES OWNERSHIP",
            Self::Borrowed => "BORROWS INPUT",
            Self::Shared => "SHARED OWNERSHIP",
            Self::TransferToCallee => "OWNERSHIP TRANSFERRED",
            Self::Unknown => "UNKNOWN OWNERSHIP",
        }
    }

    /// Whether this requires manual cleanup
    pub fn requires_cleanup(&self) -> bool {
        matches!(self, Self::CallerOwns | Self::TransferToCallee)
    }

    /// Whether this can be represented with Rust lifetimes
    pub fn supports_lifetimes(&self) -> bool {
        matches!(self, Self::Borrowed)
    }
}

/// Lifetime requirements for a parameter or return value
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LifetimeInfo {
    /// Whether a lifetime parameter is needed
    pub needs_lifetime: bool,

    /// Suggested lifetime parameter name (e.g., 'a, 'input)
    pub lifetime_name: String,

    /// What this lifetime must outlive
    pub outlives: Vec<String>,

    /// Documentation about lifetime requirements
    pub documentation: String,

    /// When this lifetime is invalidated
    pub invalidated_by: Vec<String>,
}

impl LifetimeInfo {
    /// Create a new lifetime requirement
    pub fn new(lifetime_name: impl Into<String>, doc: impl Into<String>) -> Self {
        Self {
            needs_lifetime: true,
            lifetime_name: lifetime_name.into(),
            outlives: Vec::new(),
            documentation: doc.into(),
            invalidated_by: Vec::new(),
        }
    }

    /// No lifetime needed
    pub fn none() -> Self {
        Self {
            needs_lifetime: false,
            lifetime_name: String::new(),
            outlives: Vec::new(),
            documentation: String::new(),
            invalidated_by: Vec::new(),
        }
    }
}

/// Complete ownership analysis for a function
#[derive(Debug, Clone)]
pub struct OwnershipInfo {
    /// Ownership semantics for return value
    pub return_ownership: OwnershipSemantics,

    /// Ownership semantics for each parameter (by name)
    pub param_ownership: HashMap<String, OwnershipSemantics>,

    /// Lifetime information for return value
    pub return_lifetime: LifetimeInfo,

    /// Lifetime information for parameters
    pub param_lifetimes: HashMap<String, LifetimeInfo>,

    /// Detected lifecycle pairs (e.g., create -> destroy)
    pub lifecycle_pair: Option<LifecyclePair>,

    /// Confidence score (0.0 - 1.0)
    pub confidence: f64,

    /// Source of ownership information
    pub source: String,

    /// Additional documentation about ownership
    pub documentation: String,
}

impl OwnershipInfo {
    /// Create ownership info with unknown semantics
    pub fn unknown() -> Self {
        Self {
            return_ownership: OwnershipSemantics::Unknown,
            param_ownership: HashMap::new(),
            return_lifetime: LifetimeInfo::none(),
            param_lifetimes: HashMap::new(),
            lifecycle_pair: None,
            confidence: 0.0,
            source: "no ownership annotations found".to_string(),
            documentation: String::new(),
        }
    }

    /// Generate Rust documentation for ownership semantics
    pub fn generate_docs(&self) -> String {
        let mut docs = String::new();

        // Main ownership classification
        docs.push_str(&format!(
            "/// **Memory Ownership:** {}\n",
            self.return_ownership.description()
        ));
        docs.push_str("///\n");

        // Custom documentation if provided
        if !self.documentation.is_empty() {
            for line in self.documentation.lines() {
                docs.push_str(&format!("/// {}\n", line));
            }
            docs.push_str("///\n");
        }

        // Lifecycle information
        if let Some(pair) = &self.lifecycle_pair {
            docs.push_str("/// **Lifecycle:**\n");
            docs.push_str(&format!("/// 1. Created by: `{}()`\n", pair.creator));
            if !pair.users.is_empty() {
                docs.push_str(&format!("/// 2. Used by: {}\n", pair.users.join(", ")));
            }
            docs.push_str(&format!("/// 3. Destroyed by: `{}()`\n", pair.destroyer));
            docs.push_str("///\n");
        }

        // Rust safety implications
        docs.push_str("/// **Rust Safety:**\n");
        match self.return_ownership {
            OwnershipSemantics::CallerOwns => {
                docs.push_str(
                    "/// The RAII wrapper automatically manages cleanup, preventing leaks.\n",
                );
                docs.push_str("/// Use `into_raw()` for manual ownership transfer.\n");
            }
            OwnershipSemantics::Borrowed => {
                docs.push_str("/// This borrows the input. Lifetime parameters ensure safety.\n");
                docs.push_str("/// The input must remain valid for the duration of use.\n");
            }
            OwnershipSemantics::CalleeOwns => {
                docs.push_str(
                    "/// This transfers ownership to the callee. Do NOT use after calling.\n",
                );
                docs.push_str(
                    "/// Consider using `ManuallyDrop` if you need to prevent cleanup.\n",
                );
            }
            OwnershipSemantics::Shared => {
                docs.push_str("/// This uses shared ownership. Multiple references are safe.\n");
            }
            OwnershipSemantics::TransferToCallee => {
                docs.push_str(
                    "/// Ownership is transferred to the caller. Must call cleanup function.\n",
                );
            }
            OwnershipSemantics::Unknown => {
                docs.push_str(
                    "/// ⚠️  Ownership semantics unclear. Verify documentation carefully.\n",
                );
            }
        }

        // Lifetime warnings
        if self.return_lifetime.needs_lifetime {
            docs.push_str("///\n");
            docs.push_str(&format!(
                "/// **Lifetime:** {}\n",
                self.return_lifetime.documentation
            ));
        }

        docs
    }
}

/// Detected lifecycle function pair (create/destroy, alloc/free, etc.)
#[derive(Debug, Clone, PartialEq)]
pub struct LifecyclePair {
    /// Function that creates the resource
    pub creator: String,

    /// Function that destroys the resource
    pub destroyer: String,

    /// Functions that use this resource
    pub users: Vec<String>,

    /// Confidence this is a correct pairing (0.0 - 1.0)
    pub confidence: f64,
}

/// Analyzer for extracting ownership and lifetime information
#[derive(Debug)]
pub struct OwnershipAnalyzer {
    /// Cache of analyzed functions
    cache: HashMap<String, OwnershipInfo>,

    /// Known lifecycle pairs
    lifecycle_pairs: HashMap<String, LifecyclePair>,
}

impl OwnershipAnalyzer {
    /// Create a new ownership analyzer
    pub fn new() -> Self {
        Self {
            cache: HashMap::new(),
            lifecycle_pairs: HashMap::new(),
        }
    }

    /// Analyze ownership semantics for a function
    pub fn analyze(&mut self, function_name: &str, doc_comment: &str) -> OwnershipInfo {
        // Check cache
        if let Some(cached) = self.cache.get(function_name) {
            return cached.clone();
        }

        let doc_lower = doc_comment.to_lowercase();
        let mut info = OwnershipInfo::unknown();

        // Analyze return value ownership
        info.return_ownership = self.analyze_return_ownership(&doc_lower, function_name);

        // Detect lifecycle pairs
        if let Some(pair) = self.detect_lifecycle_pair(&doc_lower, function_name) {
            info.lifecycle_pair = Some(pair);
            info.confidence = 0.9;
        }

        // Check for ownership transfer annotations
        if self.has_ownership_annotation(&doc_lower) {
            info.source = "explicit ownership annotation".to_string();
            info.confidence = 0.95;
        }

        // Extract lifetime requirements
        if self.requires_lifetime(&doc_lower) {
            info.return_lifetime = self.extract_lifetime_info(&doc_lower);
        }

        // Generate documentation
        info.documentation = self.extract_ownership_docs(&doc_lower);

        // Cache result
        self.cache.insert(function_name.to_string(), info.clone());
        info
    }

    /// Analyze return value ownership
    fn analyze_return_ownership(&self, doc: &str, function_name: &str) -> OwnershipSemantics {
        // Check for explicit annotations
        if doc.contains("@ownership-transfer") || doc.contains("@transfer") {
            return OwnershipSemantics::TransferToCallee;
        }

        if doc.contains("@caller-owns") || doc.contains("caller must free") {
            return OwnershipSemantics::CallerOwns;
        }

        if doc.contains("@callee-owns") || doc.contains("takes ownership") {
            return OwnershipSemantics::CalleeOwns;
        }

        if doc.contains("@borrows") || doc.contains("borrowed reference") {
            return OwnershipSemantics::Borrowed;
        }

        if doc.contains("@shared") || doc.contains("shared ownership") {
            return OwnershipSemantics::Shared;
        }

        // Infer from function name
        if function_name.ends_with("create")
            || function_name.ends_with("alloc")
            || function_name.ends_with("new")
            || function_name.contains("_create_")
        {
            return OwnershipSemantics::CallerOwns;
        }

        // Check documentation phrases
        if doc.contains("returns newly allocated") || doc.contains("caller must free") {
            return OwnershipSemantics::CallerOwns;
        }

        if doc.contains("does not take ownership") || doc.contains("borrows") {
            return OwnershipSemantics::Borrowed;
        }

        OwnershipSemantics::Unknown
    }

    /// Detect lifecycle function pairs (create/destroy, alloc/free)
    fn detect_lifecycle_pair(&mut self, doc: &str, function_name: &str) -> Option<LifecyclePair> {
        // Common lifecycle patterns (check both lowercase and PascalCase)
        let patterns = [
            ("create", "destroy"),
            ("Create", "Destroy"),
            ("alloc", "free"),
            ("Alloc", "Free"),
            ("init", "cleanup"),
            ("Init", "Cleanup"),
            ("open", "close"),
            ("Open", "Close"),
            ("begin", "end"),
            ("Begin", "End"),
            ("start", "stop"),
            ("Start", "Stop"),
            ("acquire", "release"),
            ("Acquire", "Release"),
        ];

        for (creator_suffix, destroyer_suffix) in patterns {
            if function_name.ends_with(creator_suffix) {
                let base = function_name.strip_suffix(creator_suffix)?;
                let destroyer = format!("{}{}", base, destroyer_suffix);
                let destroyer_lower = destroyer.to_lowercase();

                // Check if destroyer is mentioned in docs (case-insensitive)
                if doc.contains(&destroyer_lower)
                    || doc.contains(&format!("freed by {}", destroyer_lower))
                    || doc.contains(&format!("destroyed by {}", destroyer_lower))
                    || doc.contains(&format!("freed with {}", destroyer_lower))
                    || doc.contains(&format!("destroyed with {}", destroyer_lower))
                {
                    let pair = LifecyclePair {
                        creator: function_name.to_string(),
                        destroyer,
                        users: Vec::new(),
                        confidence: 0.9,
                    };
                    self.lifecycle_pairs
                        .insert(function_name.to_string(), pair.clone());
                    return Some(pair);
                }
            }
        }

        None
    }

    /// Check for explicit ownership annotations
    fn has_ownership_annotation(&self, doc: &str) -> bool {
        doc.contains("@ownership")
            || doc.contains("@caller-owns")
            || doc.contains("@callee-owns")
            || doc.contains("@transfer")
            || doc.contains("@borrows")
            || doc.contains("@shared")
    }

    /// Check if lifetime parameters are needed
    fn requires_lifetime(&self, doc: &str) -> bool {
        doc.contains("@lifetime")
            || doc.contains("must remain valid")
            || doc.contains("must outlive")
            || doc.contains("valid until")
            || doc.contains("pointer remains valid")
    }

    /// Extract lifetime information
    fn extract_lifetime_info(&self, doc: &str) -> LifetimeInfo {
        let mut lifetime = LifetimeInfo::new("a", "Input must remain valid");

        // Check for invalidation conditions
        if doc.contains("invalidated by") || doc.contains("freed by") {
            for line in doc.lines() {
                if line.contains("invalidated by") || line.contains("freed by") {
                    lifetime.invalidated_by.push(line.trim().to_string());
                }
            }
        }

        // Extract outlives requirements
        if doc.contains("must outlive") {
            for line in doc.lines() {
                if line.contains("must outlive") {
                    lifetime.outlives.push(line.trim().to_string());
                }
            }
        }

        lifetime
    }

    /// Extract ownership documentation
    fn extract_ownership_docs(&self, doc: &str) -> String {
        let mut docs = String::new();

        // Look for ownership-related sentences
        for line in doc.lines() {
            let line_lower = line.to_lowercase();
            if line_lower.contains("ownership")
                || line_lower.contains("must free")
                || line_lower.contains("must be freed")
                || line_lower.contains("caller must")
                || line_lower.contains("do not free")
                || line_lower.contains("managed internally")
            {
                if !docs.is_empty() {
                    docs.push('\n');
                }
                docs.push_str(line.trim());
            }
        }

        docs
    }

    /// Get known lifecycle pairs
    pub fn get_lifecycle_pairs(&self) -> &HashMap<String, LifecyclePair> {
        &self.lifecycle_pairs
    }
}

impl Default for OwnershipAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_caller_owns_annotation() {
        let mut analyzer = OwnershipAnalyzer::new();
        let doc = "Creates a new handle. @caller-owns - Caller must free with cudnnDestroy()";

        let info = analyzer.analyze("cudnnCreate", doc);

        assert_eq!(info.return_ownership, OwnershipSemantics::CallerOwns);
        assert!(info.confidence > 0.9);
    }

    #[test]
    fn test_callee_owns_annotation() {
        let mut analyzer = OwnershipAnalyzer::new();
        let doc = "Consumes the tensor. @callee-owns - Takes ownership and frees internally";

        let info = analyzer.analyze("consume_tensor", doc);

        assert_eq!(info.return_ownership, OwnershipSemantics::CalleeOwns);
    }

    #[test]
    fn test_borrowed_annotation() {
        let mut analyzer = OwnershipAnalyzer::new();
        let doc = "Reads the tensor data. @borrows - Does not take ownership";

        let info = analyzer.analyze("read_tensor", doc);

        assert_eq!(info.return_ownership, OwnershipSemantics::Borrowed);
    }

    #[test]
    fn test_lifecycle_pair_detection() {
        let mut analyzer = OwnershipAnalyzer::new();
        let doc = "Creates a new context. Must be freed with cudnnDestroy()";

        let info = analyzer.analyze("cudnnCreate", doc);

        assert!(info.lifecycle_pair.is_some());
        let pair = info.lifecycle_pair.unwrap();
        assert_eq!(pair.creator, "cudnnCreate");
        assert_eq!(pair.destroyer, "cudnnDestroy");
    }

    #[test]
    fn test_infer_from_function_name() {
        let mut analyzer = OwnershipAnalyzer::new();
        let doc = "Creates a handle";

        let info = analyzer.analyze("my_object_create", doc);

        assert_eq!(info.return_ownership, OwnershipSemantics::CallerOwns);
    }

    #[test]
    fn test_lifetime_detection() {
        let mut analyzer = OwnershipAnalyzer::new();
        let doc = "Borrows input. Pointer must remain valid until operation completes. @lifetime";

        let info = analyzer.analyze("process_data", doc);

        assert!(info.return_lifetime.needs_lifetime);
    }

    #[test]
    fn test_docs_generation() {
        let mut analyzer = OwnershipAnalyzer::new();
        let doc = "Creates a handle. Caller must free with destroy()";

        let info = analyzer.analyze("create_handle", doc);
        let docs = info.generate_docs();

        assert!(docs.contains("CALLER OWNS"));
        assert!(docs.contains("RAII wrapper"));
    }

    #[test]
    fn test_unknown_ownership() {
        let mut analyzer = OwnershipAnalyzer::new();
        let doc = "Does something with data";

        let info = analyzer.analyze("mystery_function", doc);

        assert_eq!(info.return_ownership, OwnershipSemantics::Unknown);
        assert!(info.confidence < 0.1);
    }

    #[test]
    fn test_cache() {
        let mut analyzer = OwnershipAnalyzer::new();
        let doc = "@caller-owns";

        let info1 = analyzer.analyze("test_func", doc);
        let info2 = analyzer.analyze("test_func", doc);

        assert_eq!(info1.return_ownership, info2.return_ownership);
        assert_eq!(analyzer.cache.len(), 1);
    }
}
