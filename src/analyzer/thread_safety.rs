//! Thread safety analysis for FFI functions
//!
//! This module extracts thread safety information from:
//! - Header comments (@threadsafe, @reentrant annotations)
//! - Function attributes (__attribute__((thread_safe)))
//! - Documentation patterns ("this function is thread-safe")
//! - Code analysis (static variables, thread-local storage)
//!
//! The extracted information is critical for Rust's Send/Sync trait bounds
//! and helps prevent data races in concurrent code.

use std::collections::HashMap;

/// Thread safety classification for FFI functions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ThreadSafety {
    /// Function is safe to call concurrently from multiple threads
    Safe,

    /// Function is NOT safe for concurrent use - requires external synchronization
    Unsafe,

    /// Function is reentrant (safe for signal handlers, recursive calls)
    Reentrant,

    /// Function requires external synchronization (mutex, lock)
    RequiresSync,

    /// One instance per thread required
    PerThread,

    /// Thread safety is unknown - assume unsafe
    Unknown,
}

/// Detailed thread safety information for a function
#[derive(Debug, Clone)]
pub struct ThreadSafetyInfo {
    /// Thread safety classification
    pub safety: ThreadSafety,

    /// Confidence level (0.0-1.0)
    pub confidence: f64,

    /// Source of information (e.g., "header comment", "static variable analysis")
    pub source: String,

    /// Full documentation text
    pub documentation: Option<String>,

    /// Specific synchronization requirements (if any)
    pub sync_requirements: Option<String>,

    /// Rust trait implications (Send, Sync, neither)
    pub trait_bounds: TraitBounds,
}

/// Rust trait bounds based on thread safety
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TraitBounds {
    pub send: bool, // Can be sent to other threads
    pub sync: bool, // Can be accessed from multiple threads
}

impl ThreadSafetyInfo {
    /// Create new thread safety info
    pub fn new(safety: ThreadSafety, confidence: f64, source: String) -> Self {
        let trait_bounds = match safety {
            ThreadSafety::Safe | ThreadSafety::Reentrant => TraitBounds {
                send: true,
                sync: true,
            },
            ThreadSafety::PerThread => TraitBounds {
                send: true,
                sync: false,
            },
            ThreadSafety::Unsafe | ThreadSafety::RequiresSync | ThreadSafety::Unknown => {
                TraitBounds {
                    send: false,
                    sync: false,
                }
            }
        };

        Self {
            safety,
            confidence,
            source,
            documentation: None,
            sync_requirements: None,
            trait_bounds,
        }
    }

    /// Generate Rust documentation for thread safety
    pub fn generate_docs(&self) -> String {
        let mut docs = String::new();

        // Safety classification
        docs.push_str(&format!(
            "/// **Thread Safety:** {}\n",
            self.safety_description()
        ));
        docs.push_str("///\n");

        // Details
        if let Some(doc) = &self.documentation {
            docs.push_str(&format!("/// {}\n", doc));
            docs.push_str("///\n");
        }

        // Synchronization requirements
        if let Some(req) = &self.sync_requirements {
            docs.push_str(&format!("/// **Synchronization:** {}\n", req));
            docs.push_str("///\n");
        }

        // Rust implications
        docs.push_str("/// **Rust Implications:**\n");
        if self.trait_bounds.send && self.trait_bounds.sync {
            docs.push_str("/// - Implements `Send + Sync` - can be used safely across threads\n");
        } else if self.trait_bounds.send {
            docs.push_str("/// - Implements `Send` - can be sent to other threads\n");
            docs.push_str("/// - Does NOT implement `Sync` - cannot be shared between threads\n");
        } else {
            docs.push_str("/// - Does NOT implement `Send` or `Sync`\n");
            docs.push_str("/// - Cannot be sent to other threads\n");
            docs.push_str("/// - Cannot be shared between threads\n");
        }

        docs
    }

    fn safety_description(&self) -> &'static str {
        match self.safety {
            ThreadSafety::Safe => "THREAD-SAFE",
            ThreadSafety::Unsafe => "NOT THREAD-SAFE",
            ThreadSafety::Reentrant => "REENTRANT",
            ThreadSafety::RequiresSync => "REQUIRES SYNCHRONIZATION",
            ThreadSafety::PerThread => "ONE INSTANCE PER THREAD",
            ThreadSafety::Unknown => "UNKNOWN (assume not thread-safe)",
        }
    }
}

/// Analyzer for extracting thread safety information
#[derive(Debug)]
pub struct ThreadSafetyAnalyzer {
    /// Cache of analyzed functions
    cache: HashMap<String, ThreadSafetyInfo>,
}

impl ThreadSafetyAnalyzer {
    pub fn new() -> Self {
        Self {
            cache: HashMap::new(),
        }
    }

    /// Analyze a function's thread safety based on its documentation
    pub fn analyze(&mut self, function_name: &str, doc_comment: &str) -> ThreadSafetyInfo {
        // Check cache first
        if let Some(cached) = self.cache.get(function_name) {
            return cached.clone();
        }

        let doc_lower = doc_comment.to_lowercase();

        // Check for not-thread-safe annotation FIRST (higher priority than thread-safe)
        if self.has_unsafe_annotation(&doc_lower) {
            let mut info = ThreadSafetyInfo::new(
                ThreadSafety::Unsafe,
                0.9,
                "explicit not-thread-safe annotation".to_string(),
            );

            // Check for synchronization requirements
            if let Some(sync) = self.extract_sync_requirements(&doc_lower) {
                info.safety = ThreadSafety::RequiresSync;
                info.sync_requirements = Some(sync);
            }

            self.cache.insert(function_name.to_string(), info.clone());
            return info;
        }

        // Check for per-thread requirement
        if self.has_per_thread_annotation(&doc_lower) {
            let info = ThreadSafetyInfo::new(
                ThreadSafety::PerThread,
                0.8,
                "per-thread annotation".to_string(),
            );
            self.cache.insert(function_name.to_string(), info.clone());
            return info;
        }

        // Check for explicit thread-safe annotations
        if self.has_thread_safe_annotation(&doc_lower) {
            let info =
                ThreadSafetyInfo::new(ThreadSafety::Safe, 0.9, "explicit annotation".to_string());
            self.cache.insert(function_name.to_string(), info.clone());
            return info;
        }

        // Check for reentrant annotation
        if self.has_reentrant_annotation(&doc_lower) {
            let info = ThreadSafetyInfo::new(
                ThreadSafety::Reentrant,
                0.9,
                "reentrant annotation".to_string(),
            );
            self.cache.insert(function_name.to_string(), info.clone());
            return info;
        }

        // Default: unknown (assume unsafe)
        let info = ThreadSafetyInfo::new(
            ThreadSafety::Unknown,
            0.5,
            "no annotation found".to_string(),
        );
        self.cache.insert(function_name.to_string(), info.clone());
        info
    }

    /// Check for thread-safe annotations
    fn has_thread_safe_annotation(&self, doc: &str) -> bool {
        // Must NOT have "not thread-safe" or similar negations
        if doc.contains("not thread-safe")
            || doc.contains("not thread safe")
            || doc.contains("isn't thread-safe")
            || doc.contains("is not thread-safe")
        {
            return false;
        }

        // Check each pattern - be careful with word boundaries!
        // "thread safe" could match "thread safety" so we check more specific patterns first
        if doc.contains("@threadsafe") {
            return true;
        }
        if doc.contains("@thread-safe") {
            return true;
        }
        if doc.contains("thread-safe") {
            // Check hyphenated first
            return true;
        }
        // For "thread safe" with space, need to ensure it's not "thread safety"
        if doc.contains("thread safe") && !doc.contains("thread safety") {
            return true;
        }
        if doc.contains("safe for concurrent use") {
            return true;
        }
        if doc.contains("may be called concurrently") {
            return true;
        }
        if doc.contains("safe to call from multiple threads") {
            return true;
        }

        false
    }
    /// Check for reentrant annotations
    fn has_reentrant_annotation(&self, doc: &str) -> bool {
        doc.contains("@reentrant")
            || doc.contains("reentrant")
            || doc.contains("signal safe")
            || doc.contains("async-signal-safe")
    }

    /// Check for not-thread-safe annotations
    fn has_unsafe_annotation(&self, doc: &str) -> bool {
        doc.contains("@not-thread-safe")
            || doc.contains("not thread-safe")
            || doc.contains("not thread safe")
            || doc.contains("single-threaded")
            || doc.contains("not safe for concurrent")
            || doc.contains("not reentrant")
            || doc.contains("requires external synchronization")
            || doc.contains("must be synchronized")
    }

    /// Check for per-thread annotations
    fn has_per_thread_annotation(&self, doc: &str) -> bool {
        doc.contains("one per thread")
            || doc.contains("one instance per thread")
            || doc.contains("thread-local")
            || doc.contains("per-thread instance")
    }

    /// Extract synchronization requirements from documentation
    fn extract_sync_requirements(&self, doc: &str) -> Option<String> {
        // Look for mutex/lock mentions
        if doc.contains("mutex") {
            return Some("Requires external mutex for thread safety".to_string());
        }
        if doc.contains("lock") || doc.contains("locking") {
            return Some("Requires external locking mechanism".to_string());
        }
        if doc.contains("synchroniz") {
            // Matches both "synchronized" and "synchronization"
            return Some("Must be externally synchronized".to_string());
        }
        None
    }

    /// Analyze function signature for thread-unsafe patterns
    pub fn analyze_signature(&self, _function_sig: &str) -> Option<ThreadSafetyInfo> {
        // TODO: Implement signature analysis
        // - Check for static variables
        // - Check for thread-local storage
        // - Check for global state access
        None
    }
}

impl Default for ThreadSafetyAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_thread_safe_annotation() {
        let mut analyzer = ThreadSafetyAnalyzer::new();

        let doc = "This function is thread-safe and may be called concurrently.";
        let info = analyzer.analyze("foo", doc);

        assert_eq!(info.safety, ThreadSafety::Safe);
        assert!(info.confidence > 0.8);
        assert!(info.trait_bounds.send);
        assert!(info.trait_bounds.sync);
    }

    #[test]
    fn test_not_thread_safe_annotation() {
        let mut analyzer = ThreadSafetyAnalyzer::new();

        let doc = "This function is not thread-safe. Use external synchronization.";
        let info = analyzer.analyze("bar", doc);

        assert_eq!(info.safety, ThreadSafety::RequiresSync);
        assert!(!info.trait_bounds.send);
        assert!(!info.trait_bounds.sync);
    }

    #[test]
    fn test_reentrant_annotation() {
        let mut analyzer = ThreadSafetyAnalyzer::new();

        let doc = "@reentrant This function is safe for signal handlers.";
        let info = analyzer.analyze("baz", doc);

        assert_eq!(info.safety, ThreadSafety::Reentrant);
        assert!(info.trait_bounds.send);
        assert!(info.trait_bounds.sync);
    }

    #[test]
    fn test_per_thread_annotation() {
        let mut analyzer = ThreadSafetyAnalyzer::new();

        let doc = "Create one instance per thread for optimal performance.";
        let info = analyzer.analyze("qux", doc);

        assert_eq!(info.safety, ThreadSafety::PerThread);
        assert!(info.trait_bounds.send);
        assert!(!info.trait_bounds.sync);
    }

    #[test]
    fn test_unknown_safety() {
        let mut analyzer = ThreadSafetyAnalyzer::new();

        let doc = "This is a normal function with no thread safety info.";
        let info = analyzer.analyze("quux", doc);

        assert_eq!(info.safety, ThreadSafety::Unknown);
        assert!(!info.trait_bounds.send);
        assert!(!info.trait_bounds.sync);
    }

    #[test]
    fn test_cache() {
        let mut analyzer = ThreadSafetyAnalyzer::new();

        let doc = "This function is thread-safe.";
        let info1 = analyzer.analyze("foo", doc);
        let info2 = analyzer.analyze("foo", "different doc");

        // Should return cached result
        assert_eq!(info1.safety, info2.safety);
        assert_eq!(info1.source, info2.source);
    }

    #[test]
    fn test_sync_requirements() {
        let mut analyzer = ThreadSafetyAnalyzer::new();

        let doc = "Not thread-safe. Requires mutex protection.";
        let info = analyzer.analyze("foo", doc);

        assert_eq!(info.safety, ThreadSafety::RequiresSync);
        assert!(info.sync_requirements.is_some());
        assert!(info.sync_requirements.unwrap().contains("mutex"));
    }

    #[test]
    fn test_docs_generation() {
        let info = ThreadSafetyInfo::new(ThreadSafety::Safe, 0.9, "test".to_string());

        let docs = info.generate_docs();
        assert!(docs.contains("THREAD-SAFE"));
        assert!(docs.contains("Send + Sync"));
    }
}
