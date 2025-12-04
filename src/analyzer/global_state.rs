use std::collections::HashMap;

/// Types of global state that can be detected
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum GlobalStateType {
    Initialization,
    ThreadLocalStorage,
    EnvironmentVariable,
    GlobalVariable,
    Singleton,
    Registry,
    Configuration,
}

/// Information about global state usage
#[derive(Debug, Clone, PartialEq)]
pub struct GlobalStateInfo {
    pub state_type: GlobalStateType,
    pub name: Option<String>,
    pub must_initialize: bool,
    pub initialization_function: Option<String>,
    pub cleanup_function: Option<String>,
    pub is_thread_safe: bool,
    pub requires_synchronization: bool,
    pub description: String,
}

/// Collection of global state information
#[derive(Debug, Clone, PartialEq)]
pub struct GlobalState {
    pub states: Vec<GlobalStateInfo>,
    pub requires_init: bool,
    pub init_function: Option<String>,
    pub cleanup_function: Option<String>,
}

impl GlobalState {
    pub fn new() -> Self {
        Self {
            states: Vec::new(),
            requires_init: false,
            init_function: None,
            cleanup_function: None,
        }
    }

    pub fn add_state(&mut self, state: GlobalStateInfo) {
        self.states.push(state);
    }

    pub fn has_global_state(&self) -> bool {
        !self.states.is_empty() || self.requires_init
    }
}

impl Default for GlobalState {
    fn default() -> Self {
        Self::new()
    }
}

/// Analyzer for detecting global state patterns
#[derive(Debug)]
pub struct GlobalStateAnalyzer {
    cache: HashMap<String, GlobalState>,
}

impl GlobalStateAnalyzer {
    pub fn new() -> Self {
        Self {
            cache: HashMap::new(),
        }
    }

    /// Analyze documentation to extract global state information
    pub fn analyze(&mut self, function_name: &str, docs: &str) -> GlobalState {
        if let Some(cached) = self.cache.get(function_name) {
            return cached.clone();
        }

        let mut state = GlobalState::new();
        let lower_docs = docs.to_lowercase();

        // Check for initialization requirements
        if self.requires_initialization(function_name, &lower_docs) {
            state.requires_init = true;
            state.init_function = self.extract_init_function(&lower_docs);
        }

        // Extract different types of global state
        self.extract_thread_local_storage(docs, &lower_docs, &mut state);
        self.extract_environment_variables(docs, &lower_docs, &mut state);
        self.extract_global_variables(docs, &lower_docs, &mut state);
        self.extract_singletons(&lower_docs, &mut state);
        self.extract_registries(&lower_docs, &mut state);
        self.extract_configuration(&lower_docs, &mut state);

        // Check for cleanup functions
        state.cleanup_function = self.extract_cleanup_function(&lower_docs);

        self.cache.insert(function_name.to_string(), state.clone());
        state
    }

    fn requires_initialization(&self, function_name: &str, docs: &str) -> bool {
        // Check function name patterns
        let is_init_func = function_name.contains("init")
            || function_name.contains("setup")
            || function_name.contains("start")
            || function_name == "initialize";

        // Check documentation patterns
        let requires_init = docs.contains("must be initialized")
            || docs.contains("must call init")
            || docs.contains("requires initialization")
            || docs.contains("before first use")
            || docs.contains("initialize before")
            || docs.contains("call init() first");

        is_init_func || requires_init
    }

    fn extract_init_function(&self, docs: &str) -> Option<String> {
        let patterns = ["call ", "initialize with ", "must call ", "first call "];

        for pattern in &patterns {
            if let Some(pos) = docs.find(pattern) {
                let start = pos + pattern.len();
                let rest = &docs[start..];

                // Extract function name
                if let Some(func_name) = self.extract_function_name(rest) {
                    return Some(func_name);
                }
            }
        }

        // Check for common init function names
        if docs.contains("init()") {
            return Some("init".to_string());
        }
        if docs.contains("initialize()") {
            return Some("initialize".to_string());
        }
        if docs.contains("setup()") {
            return Some("setup".to_string());
        }

        None
    }

    fn extract_cleanup_function(&self, docs: &str) -> Option<String> {
        let patterns = ["cleanup", "finalize", "shutdown", "destroy", "deinit"];

        for pattern in &patterns {
            if docs.contains(pattern)
                && docs.contains(&format!("{}()", pattern)) {
                    return Some(pattern.to_string());
                }
        }

        None
    }

    fn extract_thread_local_storage(&self, _docs: &str, lower_docs: &str, state: &mut GlobalState) {
        if lower_docs.contains("thread-local")
            || lower_docs.contains("thread local")
            || lower_docs.contains("tls")
            || lower_docs.contains("per-thread")
        {
            state.add_state(GlobalStateInfo {
                state_type: GlobalStateType::ThreadLocalStorage,
                name: None,
                must_initialize: false,
                initialization_function: None,
                cleanup_function: None,
                is_thread_safe: true,
                requires_synchronization: false,
                description: "Uses thread-local storage".to_string(),
            });
        }
    }

    fn extract_environment_variables(&self, docs: &str, lower_docs: &str, state: &mut GlobalState) {
        // Look for environment variable patterns
        let env_patterns = [
            "environment variable",
            "env var",
            "getenv",
            "$HOME",
            "$PATH",
            "%",
        ];

        for pattern in &env_patterns {
            if lower_docs.contains(pattern) {
                let var_name = self.extract_env_var_name(docs);
                state.add_state(GlobalStateInfo {
                    state_type: GlobalStateType::EnvironmentVariable,
                    name: var_name,
                    must_initialize: false,
                    initialization_function: None,
                    cleanup_function: None,
                    is_thread_safe: true,
                    requires_synchronization: false,
                    description: "Reads environment variables".to_string(),
                });
                break;
            }
        }
    }

    fn extract_global_variables(&self, _docs: &str, lower_docs: &str, state: &mut GlobalState) {
        if lower_docs.contains("global variable")
            || lower_docs.contains("global state")
            || lower_docs.contains("shared state")
        {
            let has_negation = lower_docs.contains("not thread-safe")
                || lower_docs.contains("isn't thread-safe")
                || lower_docs.contains("not threadsafe");

            let is_thread_safe = if has_negation {
                false
            } else {
                lower_docs.contains("thread-safe")
                    || lower_docs.contains("threadsafe")
                    || lower_docs.contains("synchronized")
            };

            state.add_state(GlobalStateInfo {
                state_type: GlobalStateType::GlobalVariable,
                name: None,
                must_initialize: false,
                initialization_function: None,
                cleanup_function: None,
                is_thread_safe,
                requires_synchronization: !is_thread_safe,
                description: "Uses global variables".to_string(),
            });
        }
    }

    fn extract_singletons(&self, lower_docs: &str, state: &mut GlobalState) {
        if lower_docs.contains("singleton") || lower_docs.contains("single instance") {
            state.add_state(GlobalStateInfo {
                state_type: GlobalStateType::Singleton,
                name: None,
                must_initialize: true,
                initialization_function: None,
                cleanup_function: None,
                is_thread_safe: false,
                requires_synchronization: true,
                description: "Uses singleton pattern".to_string(),
            });
        }
    }

    fn extract_registries(&self, lower_docs: &str, state: &mut GlobalState) {
        if lower_docs.contains("registry")
            || lower_docs.contains("register")
            || lower_docs.contains("global registry")
        {
            state.add_state(GlobalStateInfo {
                state_type: GlobalStateType::Registry,
                name: None,
                must_initialize: false,
                initialization_function: None,
                cleanup_function: None,
                is_thread_safe: false,
                requires_synchronization: true,
                description: "Accesses global registry".to_string(),
            });
        }
    }

    fn extract_configuration(&self, lower_docs: &str, state: &mut GlobalState) {
        if lower_docs.contains("configuration")
            || lower_docs.contains("config file")
            || lower_docs.contains("settings")
        {
            state.add_state(GlobalStateInfo {
                state_type: GlobalStateType::Configuration,
                name: None,
                must_initialize: false,
                initialization_function: None,
                cleanup_function: None,
                is_thread_safe: true,
                requires_synchronization: false,
                description: "Reads configuration".to_string(),
            });
        }
    }

    fn extract_function_name(&self, text: &str) -> Option<String> {
        // Look for identifier followed by ()
        if let Some(paren_pos) = text.find('(') {
            let before = &text[..paren_pos];
            let words: Vec<&str> = before.split_whitespace().collect();
            if let Some(last_word) = words.last()
                && self.looks_like_function_name(last_word) {
                    return Some(last_word.to_string());
                }
        }
        None
    }

    fn looks_like_function_name(&self, word: &str) -> bool {
        // Simple check: contains underscore or is mixed case
        word.contains('_')
            || (word.chars().any(|c| c.is_lowercase()) && word.chars().any(|c| c.is_uppercase()))
    }

    fn extract_env_var_name(&self, docs: &str) -> Option<String> {
        // Look for patterns like $VAR or %VAR%
        if let Some(pos) = docs.find('$') {
            let rest = &docs[pos + 1..];
            let end = rest
                .find(|c: char| !c.is_alphanumeric() && c != '_')
                .unwrap_or(rest.len());
            if end > 0 {
                return Some(rest[..end].to_string());
            }
        }

        // Look for ALL_CAPS words that might be env vars
        for word in docs.split_whitespace() {
            if word.len() > 2 && word.chars().all(|c| c.is_uppercase() || c == '_') {
                return Some(word.to_string());
            }
        }

        None
    }

    /// Generate documentation from global state
    pub fn generate_documentation(&self, state: &GlobalState) -> String {
        if !state.has_global_state() {
            return String::new();
        }

        let mut doc = String::from("# Global State\n\n");

        if state.requires_init {
            doc.push_str("⚠️ **Requires Initialization**\n\n");
            if let Some(init_func) = &state.init_function {
                doc.push_str(&format!(
                    "Call `{}()` before using this function.\n\n",
                    init_func
                ));
            }
        }

        if !state.states.is_empty() {
            doc.push_str("## State Usage\n\n");

            for state_info in &state.states {
                doc.push_str(&format!("### {:?}\n\n", state_info.state_type));
                doc.push_str(&format!("{}\n\n", state_info.description));

                if let Some(name) = &state_info.name {
                    doc.push_str(&format!("- Name: `{}`\n", name));
                }

                if state_info.must_initialize {
                    doc.push_str("- ⚠️ Must be initialized\n");
                }

                if !state_info.is_thread_safe {
                    doc.push_str("- ⚠️ Not thread-safe\n");
                }

                if state_info.requires_synchronization {
                    doc.push_str("- 🔒 Requires synchronization\n");
                }

                doc.push('\n');
            }
        }

        if let Some(cleanup_func) = &state.cleanup_function {
            doc.push_str(&format!(
                "**Cleanup**: Call `{}()` when done.\n\n",
                cleanup_func
            ));
        }

        doc
    }

    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }
}

impl Default for GlobalStateAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_initialization_required() {
        let mut analyzer = GlobalStateAnalyzer::new();
        let docs = "Must be initialized before first use.";

        let state = analyzer.analyze("process", docs);
        assert!(state.requires_init);
    }

    #[test]
    fn test_init_function_extraction() {
        let mut analyzer = GlobalStateAnalyzer::new();
        let docs = "Must call init() before using this function.";

        let state = analyzer.analyze("process", docs);
        assert_eq!(state.init_function, Some("init".to_string()));
    }

    #[test]
    fn test_thread_local_storage() {
        let mut analyzer = GlobalStateAnalyzer::new();
        let docs = "Uses thread-local storage for state.";

        let state = analyzer.analyze("get_tls", docs);
        assert!(!state.states.is_empty());
        assert_eq!(
            state.states[0].state_type,
            GlobalStateType::ThreadLocalStorage
        );
        assert!(state.states[0].is_thread_safe);
    }

    #[test]
    fn test_environment_variables() {
        let mut analyzer = GlobalStateAnalyzer::new();
        let docs = "Reads the $HOME environment variable.";

        let state = analyzer.analyze("get_home", docs);
        assert!(!state.states.is_empty());
        assert_eq!(
            state.states[0].state_type,
            GlobalStateType::EnvironmentVariable
        );
        assert_eq!(state.states[0].name, Some("HOME".to_string()));
    }

    #[test]
    fn test_global_variables() {
        let mut analyzer = GlobalStateAnalyzer::new();
        let docs = "Accesses global state that is not thread-safe.";

        let state = analyzer.analyze("get_global", docs);
        assert!(!state.states.is_empty());
        assert_eq!(state.states[0].state_type, GlobalStateType::GlobalVariable);
        assert!(!state.states[0].is_thread_safe);
        assert!(state.states[0].requires_synchronization);
    }

    #[test]
    fn test_singleton_pattern() {
        let mut analyzer = GlobalStateAnalyzer::new();
        let docs = "Returns the singleton instance.";

        let state = analyzer.analyze("get_instance", docs);
        assert!(!state.states.is_empty());
        assert_eq!(state.states[0].state_type, GlobalStateType::Singleton);
        assert!(state.states[0].must_initialize);
    }

    #[test]
    fn test_registry_access() {
        let mut analyzer = GlobalStateAnalyzer::new();
        let docs = "Register handler in global registry.";

        let state = analyzer.analyze("register_handler", docs);
        assert!(!state.states.is_empty());
        assert_eq!(state.states[0].state_type, GlobalStateType::Registry);
    }

    #[test]
    fn test_configuration() {
        let mut analyzer = GlobalStateAnalyzer::new();
        let docs = "Loads settings from configuration file.";

        let state = analyzer.analyze("load_config", docs);
        assert!(!state.states.is_empty());
        assert_eq!(state.states[0].state_type, GlobalStateType::Configuration);
    }

    #[test]
    fn test_cleanup_function() {
        let mut analyzer = GlobalStateAnalyzer::new();
        let docs = "Initialize library. Call cleanup() when done.";

        let state = analyzer.analyze("init", docs);
        assert_eq!(state.cleanup_function, Some("cleanup".to_string()));
    }

    #[test]
    fn test_multiple_states() {
        let mut analyzer = GlobalStateAnalyzer::new();
        let docs = "Uses thread-local storage and reads environment variables.";

        let state = analyzer.analyze("complex", docs);
        assert!(state.states.len() >= 2);
    }

    #[test]
    fn test_cache_functionality() {
        let mut analyzer = GlobalStateAnalyzer::new();
        let docs = "Must be initialized.";

        let state1 = analyzer.analyze("test_fn", docs);
        let state2 = analyzer.analyze("test_fn", docs);

        assert_eq!(state1, state2);
    }

    #[test]
    fn test_generate_documentation() {
        let mut analyzer = GlobalStateAnalyzer::new();
        let docs = "Must call init() first. Uses global state.";

        let state = analyzer.analyze("process", docs);
        let doc = analyzer.generate_documentation(&state);

        assert!(doc.contains("Global State"));
        assert!(doc.contains("Requires Initialization"));
    }
}
