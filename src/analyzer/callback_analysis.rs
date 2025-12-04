//! Callback lifetime and semantics analyzer.
//!
//! This module analyzes function pointer parameters to determine:
//! - How long callbacks must remain valid
//! - How many times they will be invoked
//! - Thread safety requirements
//! - Ownership of callback context data

use std::collections::HashMap;

/// Information about a callback parameter
#[derive(Debug, Clone, PartialEq)]
pub struct CallbackInfo {
    /// Parameter name
    pub param_name: String,
    /// Lifetime requirement for the callback
    pub lifetime: CallbackLifetime,
    /// How many times the callback will be invoked
    pub invocation_count: InvocationCount,
    /// Thread safety requirements
    pub thread_safety: CallbackThreadSafety,
    /// Who owns the callback context data
    pub context_ownership: ContextOwnership,
    /// Additional notes from documentation
    pub notes: Vec<String>,
}

/// Lifetime requirement for a callback
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CallbackLifetime {
    /// Callback is only used during the function call
    CallDuration,
    /// Callback must remain valid until a specific event (e.g., until operation completes)
    UntilEvent,
    /// Callback must remain valid until explicitly unregistered
    UntilUnregister,
    /// Callback must remain valid for the lifetime of an object
    ObjectLifetime,
    /// Lifetime is unclear from documentation
    Unknown,
}

/// How many times a callback will be invoked
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InvocationCount {
    /// Called exactly once
    Once,
    /// Called zero or one times
    ZeroOrOnce,
    /// Called multiple times (specific count)
    Multiple(Option<usize>),
    /// Called repeatedly until some condition
    Repeated,
    /// Unknown invocation pattern
    Unknown,
}

/// Thread safety requirements for callbacks
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CallbackThreadSafety {
    /// Callback is always invoked from the same thread
    SingleThreaded,
    /// Callback may be invoked from any thread
    MultiThreaded,
    /// Callback is invoked from a specific thread pool
    ThreadPool,
    /// Unknown thread safety requirements
    Unknown,
}

/// Who owns the callback context data
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContextOwnership {
    /// Caller owns and must free context
    CallerOwned,
    /// Library owns context after registration
    LibraryOwned,
    /// Shared ownership (refcounted)
    Shared,
    /// No context data
    NoContext,
    /// Unclear ownership
    Unknown,
}

/// Complete callback analysis for a function
#[derive(Debug, Clone, PartialEq)]
pub struct CallbackSemantics {
    /// Map of parameter names to their callback information
    pub callbacks: HashMap<String, CallbackInfo>,
    /// Overall confidence in the analysis
    pub confidence: f64,
}

impl CallbackSemantics {
    /// Checks if the function has any callbacks
    pub fn has_callbacks(&self) -> bool {
        !self.callbacks.is_empty()
    }

    /// Generates documentation for callback usage
    pub fn generate_documentation(&self) -> String {
        if !self.has_callbacks() {
            return String::new();
        }

        let mut docs = String::from("/// # Callbacks\n");
        docs.push_str("///\n");

        for (param_name, info) in &self.callbacks {
            docs.push_str(&format!("/// ## `{}`\n", param_name));
            docs.push_str("///\n");

            // Lifetime
            match info.lifetime {
                CallbackLifetime::CallDuration => {
                    docs.push_str(
                        "/// **Lifetime:** Only used during function call (stack-safe)\n",
                    );
                }
                CallbackLifetime::UntilEvent => {
                    docs.push_str(
                        "/// **Lifetime:** Must remain valid until operation completes\n",
                    );
                }
                CallbackLifetime::UntilUnregister => {
                    docs.push_str(
                        "/// **Lifetime:** Must remain valid until explicitly unregistered\n",
                    );
                }
                CallbackLifetime::ObjectLifetime => {
                    docs.push_str(
                        "/// **Lifetime:** Must remain valid for lifetime of associated object\n",
                    );
                }
                CallbackLifetime::Unknown => {
                    docs.push_str("/// **Lifetime:** Unknown (assume must persist beyond call)\n");
                }
            }

            // Invocation count
            match info.invocation_count {
                InvocationCount::Once => {
                    docs.push_str("/// **Invocations:** Called exactly once\n");
                }
                InvocationCount::ZeroOrOnce => {
                    docs.push_str("/// **Invocations:** Called zero or one times\n");
                }
                InvocationCount::Multiple(Some(n)) => {
                    docs.push_str(&format!("/// **Invocations:** Called {} times\n", n));
                }
                InvocationCount::Multiple(None) => {
                    docs.push_str("/// **Invocations:** Called multiple times\n");
                }
                InvocationCount::Repeated => {
                    docs.push_str("/// **Invocations:** Called repeatedly until condition met\n");
                }
                InvocationCount::Unknown => {
                    docs.push_str(
                        "/// **Invocations:** Unknown (assume may be called multiple times)\n",
                    );
                }
            }

            // Thread safety
            match info.thread_safety {
                CallbackThreadSafety::SingleThreaded => {
                    docs.push_str("/// **Thread Safety:** Called from same thread only\n");
                }
                CallbackThreadSafety::MultiThreaded => {
                    docs.push_str("/// **Thread Safety:** May be called from any thread (must be Send + Sync)\n");
                }
                CallbackThreadSafety::ThreadPool => {
                    docs.push_str(
                        "/// **Thread Safety:** Called from thread pool (must be Send + Sync)\n",
                    );
                }
                CallbackThreadSafety::Unknown => {}
            }

            // Context ownership
            match info.context_ownership {
                ContextOwnership::CallerOwned => {
                    docs.push_str("/// **Context:** Caller owns and must free context data\n");
                }
                ContextOwnership::LibraryOwned => {
                    docs.push_str("/// **Context:** Library takes ownership and frees context\n");
                }
                ContextOwnership::Shared => {
                    docs.push_str("/// **Context:** Shared ownership (refcounted)\n");
                }
                ContextOwnership::NoContext => {}
                ContextOwnership::Unknown => {}
            }

            // Additional notes
            for note in &info.notes {
                docs.push_str(&format!("/// - {}\n", note));
            }

            docs.push_str("///\n");
        }

        docs
    }
}

/// Main callback analyzer
#[derive(Debug)]
pub struct CallbackAnalyzer {
    /// Cache of analyzed callback semantics by function name
    cache: HashMap<String, CallbackSemantics>,
}

impl CallbackAnalyzer {
    /// Creates a new callback analyzer
    pub fn new() -> Self {
        Self {
            cache: HashMap::new(),
        }
    }

    /// Analyzes callback parameters from function documentation and signature
    ///
    /// # Arguments
    /// * `function_name` - Name of the function
    /// * `documentation` - Combined documentation text
    /// * `parameters` - Parameter names and types
    ///
    /// # Returns
    /// Callback semantics information
    pub fn analyze(
        &mut self,
        function_name: &str,
        documentation: &str,
        parameters: &[(String, String)],
    ) -> CallbackSemantics {
        // Check cache
        if let Some(cached) = self.cache.get(function_name) {
            return cached.clone();
        }

        let mut callbacks = HashMap::new();
        let mut total_confidence = 0.0;
        let mut callback_count = 0;

        // Find callback parameters
        for (param_name, param_type) in parameters {
            if self.is_callback_type(param_type) {
                let info = self.analyze_callback(param_name, param_type, documentation);
                callbacks.insert(param_name.clone(), info);
                total_confidence += 0.7; // Base confidence for finding callback
                callback_count += 1;
            }
        }

        let confidence = if callback_count > 0 {
            total_confidence / callback_count as f64
        } else {
            0.0
        };

        let semantics = CallbackSemantics {
            callbacks,
            confidence,
        };

        // Cache the result
        self.cache
            .insert(function_name.to_string(), semantics.clone());
        semantics
    }

    /// Checks if a type is a callback/function pointer
    fn is_callback_type(&self, type_str: &str) -> bool {
        let lower = type_str.to_lowercase();

        // Function pointer syntax
        if type_str.contains("(*") || type_str.contains("fn(") {
            return true;
        }

        // Common callback type names
        lower.contains("callback")
            || lower.contains("handler")
            || lower.contains("listener")
            || lower.contains("hook")
            || lower.contains("notify")
            || lower.contains("observer")
            || lower.ends_with("_fn")
            || lower.ends_with("_func")
    }

    /// Analyzes a specific callback parameter
    fn analyze_callback(&self, param_name: &str, param_type: &str, docs: &str) -> CallbackInfo {
        CallbackInfo {
            param_name: param_name.to_string(),
            lifetime: self.determine_lifetime(param_name, docs),
            invocation_count: self.determine_invocation_count(param_name, docs),
            thread_safety: self.determine_thread_safety(param_name, param_type, docs),
            context_ownership: self.determine_context_ownership(param_name, docs),
            notes: self.extract_callback_notes(param_name, docs),
        }
    }

    /// Determines callback lifetime from documentation
    fn determine_lifetime(&self, param_name: &str, docs: &str) -> CallbackLifetime {
        let param_lower = param_name.to_lowercase();

        // Look for lifetime mentions near the parameter
        for line in docs.lines() {
            let line_lower = line.to_lowercase();
            if !line_lower.contains(&param_lower) && !line_lower.contains("callback") {
                continue;
            }

            // Call duration patterns - use lower for case-insensitive matching
            if line_lower.contains("only during")
                || line_lower.contains("only used during")
                || line_lower.contains("synchronous")
                || line_lower.contains("before returning")
            {
                return CallbackLifetime::CallDuration;
            }

            // Until event patterns
            if line_lower.contains("until")
                && (line_lower.contains("complete")
                    || line_lower.contains("finish")
                    || line_lower.contains("done"))
            {
                return CallbackLifetime::UntilEvent;
            }

            // Until unregister patterns
            if line_lower.contains("unregister")
                || line_lower.contains("remove")
                || line_lower.contains("deregister")
            {
                return CallbackLifetime::UntilUnregister;
            }

            // Object lifetime patterns
            if line_lower.contains("lifetime of")
                || line_lower.contains("as long as")
                || line_lower.contains("for the duration of")
            {
                return CallbackLifetime::ObjectLifetime;
            }
        }

        CallbackLifetime::Unknown
    }

    /// Determines how many times callback is invoked
    fn determine_invocation_count(&self, param_name: &str, docs: &str) -> InvocationCount {
        let param_lower = param_name.to_lowercase();

        for line in docs.lines() {
            let line_lower = line.to_lowercase();
            if !line_lower.contains(&param_lower) && !line_lower.contains("callback") {
                continue;
            }

            // Once patterns - use lower for case-insensitive matching
            if line_lower.contains("called once")
                || line_lower.contains("invoked once")
                || line_lower.contains("single call")
                || line_lower.contains("one-shot")
            {
                return InvocationCount::Once;
            }

            // Zero or once patterns
            if line_lower.contains("may be called") || line_lower.contains("optionally called") {
                return InvocationCount::ZeroOrOnce;
            }

            // Repeated patterns
            if line_lower.contains("repeatedly")
                || line_lower.contains("each time")
                || line_lower.contains("every time")
                || line_lower.contains("for each")
            {
                return InvocationCount::Repeated;
            }

            // Multiple with count
            if line_lower.contains("called") {
                // Try to extract number
                let words: Vec<&str> = line_lower.split_whitespace().collect();
                for (i, word) in words.iter().enumerate() {
                    if *word == "called" && i > 0
                        && let Ok(n) = words[i - 1].parse::<usize>() {
                            return InvocationCount::Multiple(Some(n));
                        }
                }
                return InvocationCount::Multiple(None);
            }
        }

        InvocationCount::Unknown
    }

    /// Determines thread safety requirements
    fn determine_thread_safety(
        &self,
        param_name: &str,
        _param_type: &str,
        docs: &str,
    ) -> CallbackThreadSafety {
        let param_lower = param_name.to_lowercase();

        for line in docs.lines() {
            let line_lower = line.to_lowercase();
            if !line_lower.contains(&param_lower) && !line_lower.contains("callback") {
                continue;
            }

            // Multi-threaded patterns - use lower for case-insensitive matching
            if line_lower.contains("any thread")
                || line_lower.contains("different thread")
                || line_lower.contains("thread-safe")
                || line_lower.contains("from any thread")
            {
                return CallbackThreadSafety::MultiThreaded;
            }

            // Thread pool patterns
            if line_lower.contains("thread pool") || line_lower.contains("worker thread") {
                return CallbackThreadSafety::ThreadPool;
            }

            // Single-threaded patterns
            if line_lower.contains("same thread")
                || line_lower.contains("calling thread")
                || line_lower.contains("not thread-safe")
            {
                return CallbackThreadSafety::SingleThreaded;
            }
        }

        CallbackThreadSafety::Unknown
    }

    /// Determines who owns callback context data
    fn determine_context_ownership(&self, param_name: &str, docs: &str) -> ContextOwnership {
        let lower = docs.to_lowercase();
        let param_lower = param_name.to_lowercase();

        // Look for context parameter (usually next to callback)
        let has_context = lower.contains("context")
            || lower.contains("user_data")
            || lower.contains("userdata")
            || lower.contains("user data");

        if !has_context {
            return ContextOwnership::NoContext;
        }

        for line in docs.lines() {
            let line_lower = line.to_lowercase();
            if !line_lower.contains(&param_lower) && !line_lower.contains("context") {
                continue;
            }

            // Caller owned patterns
            if line_lower.contains("caller must") && line_lower.contains("free")
                || line_lower.contains("caller owns")
                || line_lower.contains("user must free")
            {
                return ContextOwnership::CallerOwned;
            }

            // Library owned patterns
            if line_lower.contains("library frees")
                || line_lower.contains("automatically freed")
                || line_lower.contains("takes ownership")
            {
                return ContextOwnership::LibraryOwned;
            }

            // Shared ownership patterns
            if line_lower.contains("reference count")
                || line_lower.contains("refcount")
                || line_lower.contains("shared")
            {
                return ContextOwnership::Shared;
            }
        }

        ContextOwnership::Unknown
    }

    /// Extracts additional notes about callback usage
    fn extract_callback_notes(&self, param_name: &str, docs: &str) -> Vec<String> {
        let mut notes = Vec::new();
        let param_lower = param_name.to_lowercase();

        for line in docs.lines() {
            let line_lower = line.to_lowercase();

            // Look for lines mentioning the callback
            if !line_lower.contains(&param_lower) && !line_lower.contains("callback") {
                continue;
            }

            // Skip lines we've already categorized
            if line_lower.contains("lifetime")
                || line_lower.contains("thread")
                || line_lower.contains("called")
                || line_lower.contains("invoked")
            {
                continue;
            }

            // Important notes
            if line_lower.contains("must")
                || line_lower.contains("should")
                || line_lower.contains("note")
                || line_lower.contains("warning")
            {
                let trimmed = line.trim();
                if !trimmed.is_empty() {
                    notes.push(trimmed.to_string());
                }
            }
        }

        notes
    }
}

impl Default for CallbackAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_callback_type() {
        let analyzer = CallbackAnalyzer::new();

        assert!(analyzer.is_callback_type("void (*callback)(int)"));
        assert!(analyzer.is_callback_type("CallbackFn"));
        assert!(analyzer.is_callback_type("event_handler_fn"));
        assert!(analyzer.is_callback_type("NotifyFunc"));
        assert!(!analyzer.is_callback_type("int"));
        assert!(!analyzer.is_callback_type("void*"));
    }

    #[test]
    fn test_call_duration_lifetime() {
        let mut analyzer = CallbackAnalyzer::new();
        let docs =
            "The callback is only used during the function call and does not need to persist.";
        let params = vec![("callback".to_string(), "void (*)(int)".to_string())];

        let semantics = analyzer.analyze("test_func", docs, &params);
        assert!(semantics.has_callbacks());

        let cb = &semantics.callbacks["callback"];
        assert_eq!(cb.lifetime, CallbackLifetime::CallDuration);
    }

    #[test]
    fn test_until_event_lifetime() {
        let mut analyzer = CallbackAnalyzer::new();
        let docs = "The callback must remain valid until the operation completes.";
        let params = vec![("callback".to_string(), "void (*)(int)".to_string())];

        let semantics = analyzer.analyze("test_func", docs, &params);
        let cb = &semantics.callbacks["callback"];
        assert_eq!(cb.lifetime, CallbackLifetime::UntilEvent);
    }

    #[test]
    fn test_until_unregister_lifetime() {
        let mut analyzer = CallbackAnalyzer::new();
        let docs = "The callback remains active until explicitly unregistered.";
        let params = vec![("handler".to_string(), "HandlerFn".to_string())];

        let semantics = analyzer.analyze("test_func", docs, &params);
        let cb = &semantics.callbacks["handler"];
        assert_eq!(cb.lifetime, CallbackLifetime::UntilUnregister);
    }

    #[test]
    fn test_once_invocation() {
        let mut analyzer = CallbackAnalyzer::new();
        let docs = "The callback is called once when the operation completes.";
        let params = vec![("callback".to_string(), "void (*)(int)".to_string())];

        let semantics = analyzer.analyze("test_func", docs, &params);
        let cb = &semantics.callbacks["callback"];
        assert_eq!(cb.invocation_count, InvocationCount::Once);
    }

    #[test]
    fn test_repeated_invocation() {
        let mut analyzer = CallbackAnalyzer::new();
        let docs = "The callback is called repeatedly for each item in the list.";
        let params = vec![("callback".to_string(), "void (*)(int)".to_string())];

        let semantics = analyzer.analyze("test_func", docs, &params);
        let cb = &semantics.callbacks["callback"];
        assert_eq!(cb.invocation_count, InvocationCount::Repeated);
    }

    #[test]
    fn test_multithread_safety() {
        let mut analyzer = CallbackAnalyzer::new();
        let docs = "The callback may be invoked from any thread and must be thread-safe.";
        let params = vec![("callback".to_string(), "void (*)(int)".to_string())];

        let semantics = analyzer.analyze("test_func", docs, &params);
        let cb = &semantics.callbacks["callback"];
        assert_eq!(cb.thread_safety, CallbackThreadSafety::MultiThreaded);
    }

    #[test]
    fn test_caller_owned_context() {
        let mut analyzer = CallbackAnalyzer::new();
        let docs =
            "The callback receives a context pointer. The caller owns and must free the context.";
        let params = vec![("callback".to_string(), "void (*)(void*)".to_string())];

        let semantics = analyzer.analyze("test_func", docs, &params);
        let cb = &semantics.callbacks["callback"];
        assert_eq!(cb.context_ownership, ContextOwnership::CallerOwned);
    }

    #[test]
    fn test_library_owned_context() {
        let mut analyzer = CallbackAnalyzer::new();
        let docs = "The callback receives context data. The library takes ownership and frees the context.";
        let params = vec![("callback".to_string(), "void (*)(void*)".to_string())];

        let semantics = analyzer.analyze("test_func", docs, &params);
        let cb = &semantics.callbacks["callback"];
        assert_eq!(cb.context_ownership, ContextOwnership::LibraryOwned);
    }

    #[test]
    fn test_no_callbacks() {
        let mut analyzer = CallbackAnalyzer::new();
        let docs = "This function takes regular parameters.";
        let params = vec![("x".to_string(), "int".to_string())];

        let semantics = analyzer.analyze("test_func", docs, &params);
        assert!(!semantics.has_callbacks());
    }

    #[test]
    fn test_documentation_generation() {
        let mut analyzer = CallbackAnalyzer::new();
        let docs = "The callback is called once and must be thread-safe.";
        let params = vec![("callback".to_string(), "void (*)(int)".to_string())];

        let semantics = analyzer.analyze("test_func", docs, &params);
        let doc = semantics.generate_documentation();

        assert!(doc.contains("# Callbacks"));
        assert!(doc.contains("callback"));
    }

    #[test]
    fn test_cache() {
        let mut analyzer = CallbackAnalyzer::new();
        let docs = "The callback is used during the call.";
        let params = vec![("cb".to_string(), "void (*)(int)".to_string())];

        let semantics1 = analyzer.analyze("test_func", docs, &params);
        let semantics2 = analyzer.analyze("test_func", docs, &params);

        assert_eq!(semantics1, semantics2);
    }
}
