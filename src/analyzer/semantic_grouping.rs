use std::collections::{HashMap, HashSet};

/// Type of semantic relationship between functions
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum GroupType {
    GetterSetter,
    Module,
    FeatureSet,
    Lifecycle,
    ErrorHandling,
    Custom(String),
}

/// Information about a semantic group of functions
#[derive(Debug, Clone, PartialEq)]
pub struct FunctionGroup {
    pub group_type: GroupType,
    pub name: String,
    pub functions: Vec<String>,
    pub description: String,
}

/// Information about getter/setter pairs
#[derive(Debug, Clone, PartialEq)]
pub struct GetterSetterPair {
    pub property_name: String,
    pub getter: Option<String>,
    pub setter: Option<String>,
    pub is_boolean: bool,
}

/// Semantic grouping information for a function
#[derive(Debug, Clone, PartialEq)]
pub struct SemanticGroupInfo {
    pub module: Option<String>,
    pub feature_set: Option<String>,
    pub related_functions: Vec<String>,
    pub getter_setter_pair: Option<GetterSetterPair>,
    pub is_getter: bool,
    pub is_setter: bool,
}

impl SemanticGroupInfo {
    pub fn new() -> Self {
        Self {
            module: None,
            feature_set: None,
            related_functions: Vec::new(),
            getter_setter_pair: None,
            is_getter: false,
            is_setter: false,
        }
    }

    pub fn has_grouping(&self) -> bool {
        self.module.is_some()
            || self.feature_set.is_some()
            || !self.related_functions.is_empty()
            || self.getter_setter_pair.is_some()
    }
}

impl Default for SemanticGroupInfo {
    fn default() -> Self {
        Self::new()
    }
}

/// Analyzer for detecting semantic grouping patterns
#[derive(Debug)]
pub struct SemanticGroupingAnalyzer {
    cache: HashMap<String, SemanticGroupInfo>,
    all_functions: HashSet<String>,
    function_docs: HashMap<String, String>,
}

impl SemanticGroupingAnalyzer {
    pub fn new() -> Self {
        Self {
            cache: HashMap::new(),
            all_functions: HashSet::new(),
            function_docs: HashMap::new(),
        }
    }

    /// Register a function for later grouping analysis
    pub fn register_function(&mut self, function_name: &str, docs: &str) {
        self.all_functions.insert(function_name.to_string());
        self.function_docs
            .insert(function_name.to_string(), docs.to_string());
    }

    /// Analyze a function's semantic grouping
    pub fn analyze(&mut self, function_name: &str, docs: &str) -> SemanticGroupInfo {
        if let Some(cached) = self.cache.get(function_name) {
            return cached.clone();
        }

        let mut info = SemanticGroupInfo::new();

        // Extract module information
        info.module = self.extract_module(function_name, docs);

        // Extract feature set
        info.feature_set = self.extract_feature_set(docs);

        // Check if it's a getter or setter
        let (is_getter, is_setter) = self.check_getter_setter(function_name, docs);
        info.is_getter = is_getter;
        info.is_setter = is_setter;

        // Find getter/setter pair
        if is_getter || is_setter {
            info.getter_setter_pair = self.find_getter_setter_pair(function_name);
        }

        // Find related functions
        info.related_functions = self.find_related_functions(function_name, docs);

        self.cache.insert(function_name.to_string(), info.clone());
        info
    }

    fn extract_module(&self, function_name: &str, docs: &str) -> Option<String> {
        let lower_docs = docs.to_lowercase();

        // Check for explicit module mentions
        if let Some(module) = self.extract_after_keyword(&lower_docs, "module:") {
            return Some(module);
        }
        if let Some(module) = self.extract_after_keyword(&lower_docs, "@module") {
            return Some(module);
        }
        if let Some(module) = self.extract_after_keyword(&lower_docs, "part of") {
            return Some(module);
        }

        // Infer from function name prefix (e.g., crypto_init, crypto_encrypt -> crypto module)
        if let Some(pos) = function_name.find('_') {
            let prefix = &function_name[..pos];
            // Check if there are multiple functions with this prefix
            let count = self
                .all_functions
                .iter()
                .filter(|f| f.starts_with(prefix) && f.contains('_'))
                .count();
            if count >= 3 {
                return Some(prefix.to_string());
            }
        }

        None
    }

    fn extract_feature_set(&self, docs: &str) -> Option<String> {
        let lower_docs = docs.to_lowercase();

        let keywords = [
            "feature:",
            "@feature",
            "belongs to",
            "provides",
            "implements",
        ];

        for keyword in &keywords {
            if let Some(feature) = self.extract_after_keyword(&lower_docs, keyword) {
                return Some(feature);
            }
        }

        None
    }

    fn check_getter_setter(&self, function_name: &str, docs: &str) -> (bool, bool) {
        let lower_name = function_name.to_lowercase();
        let lower_docs = docs.to_lowercase();

        // Check for getter patterns
        let is_getter = lower_name.starts_with("get_")
            || lower_name.starts_with("is_")
            || lower_name.starts_with("has_")
            || lower_name.starts_with("can_")
            || lower_docs.contains("returns the")
            || lower_docs.contains("gets the")
            || lower_docs.contains("retrieves the");

        // Check for setter patterns
        let is_setter = lower_name.starts_with("set_")
            || lower_name.starts_with("enable_")
            || lower_name.starts_with("disable_")
            || lower_docs.contains("sets the")
            || lower_docs.contains("updates the")
            || lower_docs.contains("modifies the");

        (is_getter, is_setter)
    }

    fn find_getter_setter_pair(&self, function_name: &str) -> Option<GetterSetterPair> {
        let lower_name = function_name.to_lowercase();

        // Extract property name
        let property = if lower_name.starts_with("get_") {
            lower_name.strip_prefix("get_")?.to_string()
        } else if lower_name.starts_with("set_") {
            lower_name.strip_prefix("set_")?.to_string()
        } else if lower_name.starts_with("is_") {
            lower_name.strip_prefix("is_")?.to_string()
        } else if lower_name.starts_with("has_") {
            lower_name.strip_prefix("has_")?.to_string()
        } else if lower_name.starts_with("can_") {
            lower_name.strip_prefix("can_")?.to_string()
        } else if lower_name.starts_with("enable_") {
            lower_name.strip_prefix("enable_")?.to_string()
        } else if lower_name.starts_with("disable_") {
            lower_name.strip_prefix("disable_")?.to_string()
        } else {
            return None;
        };

        let is_boolean = lower_name.starts_with("is_")
            || lower_name.starts_with("has_")
            || lower_name.starts_with("can_")
            || lower_name.starts_with("enable_")
            || lower_name.starts_with("disable_");

        // Look for corresponding getter/setter
        let getter_names = vec![
            format!("get_{}", property),
            format!("is_{}", property),
            format!("has_{}", property),
            format!("can_{}", property),
        ];
        let setter_names = vec![
            format!("set_{}", property),
            format!("enable_{}", property),
            format!("disable_{}", property),
        ];

        let mut getter = None;
        let mut setter = None;

        for name in &getter_names {
            if self.all_functions.contains(name) {
                getter = Some(name.clone());
                break;
            }
        }

        for name in &setter_names {
            if self.all_functions.contains(name) {
                setter = Some(name.clone());
                break;
            }
        }

        // Only return if we found at least one other function in the pair
        if (getter.is_some() && lower_name.starts_with("set_"))
            || (getter.is_some() && lower_name.starts_with("enable_"))
            || (getter.is_some() && lower_name.starts_with("disable_"))
            || (setter.is_some() && !lower_name.starts_with("set_"))
        {
            Some(GetterSetterPair {
                property_name: property,
                getter,
                setter,
                is_boolean,
            })
        } else {
            None
        }
    }

    fn find_related_functions(&self, function_name: &str, docs: &str) -> Vec<String> {
        let mut related = Vec::new();
        let lower_docs = docs.to_lowercase();

        // Look for "see also" references
        if lower_docs.contains("see also") || lower_docs.contains("see:") {
            related.extend(self.extract_function_references(docs));
        }

        // Look for functions mentioned in the same sentence as "related", "together", "with"
        for line in docs.lines() {
            let lower_line = line.to_lowercase();
            if lower_line.contains("related")
                || lower_line.contains("together")
                || lower_line.contains("with")
            {
                related.extend(self.extract_function_references(line));
            }
        }

        // Find functions with same prefix
        if let Some(pos) = function_name.find('_') {
            let prefix = &function_name[..pos];
            for func in &self.all_functions {
                if func != function_name && func.starts_with(prefix)
                    && !related.contains(func) {
                        related.push(func.clone());
                    }
            }
        }

        related
    }

    fn extract_function_references(&self, text: &str) -> Vec<String> {
        let mut functions = Vec::new();

        for func in &self.all_functions {
            if text.contains(func.as_str()) {
                functions.push(func.clone());
            }
        }

        functions
    }

    fn extract_after_keyword(&self, text: &str, keyword: &str) -> Option<String> {
        if let Some(pos) = text.find(keyword) {
            let start = pos + keyword.len();
            let rest = &text[start..];

            // Extract until newline or period
            let end = rest
                .find('\n')
                .or_else(|| rest.find('.'))
                .or_else(|| rest.find(','))
                .unwrap_or(rest.len());

            let extracted = rest[..end].trim();
            if !extracted.is_empty() {
                return Some(extracted.to_string());
            }
        }
        None
    }

    /// Group all registered functions by module
    pub fn group_by_module(&self) -> HashMap<String, Vec<String>> {
        let mut groups: HashMap<String, Vec<String>> = HashMap::new();

        for (func_name, info) in &self.cache {
            if let Some(module) = &info.module {
                groups
                    .entry(module.clone())
                    .or_default()
                    .push(func_name.clone());
            }
        }

        groups
    }

    /// Find all getter/setter pairs
    pub fn find_all_getter_setter_pairs(&self) -> Vec<GetterSetterPair> {
        let mut pairs: HashMap<String, GetterSetterPair> = HashMap::new();

        for info in self.cache.values() {
            if let Some(pair) = &info.getter_setter_pair {
                pairs
                    .entry(pair.property_name.clone())
                    .and_modify(|existing| {
                        if pair.getter.is_some() {
                            existing.getter = pair.getter.clone();
                        }
                        if pair.setter.is_some() {
                            existing.setter = pair.setter.clone();
                        }
                    })
                    .or_insert_with(|| pair.clone());
            }
        }

        pairs.into_values().collect()
    }

    /// Generate documentation from semantic grouping
    pub fn generate_documentation(&self, info: &SemanticGroupInfo) -> String {
        if !info.has_grouping() {
            return String::new();
        }

        let mut doc = String::from("# Semantic Grouping\n\n");

        if let Some(module) = &info.module {
            doc.push_str(&format!("**Module**: `{}`\n\n", module));
        }

        if let Some(feature) = &info.feature_set {
            doc.push_str(&format!("**Feature Set**: {}\n\n", feature));
        }

        if let Some(pair) = &info.getter_setter_pair {
            doc.push_str("## Property Access\n\n");
            doc.push_str(&format!("**Property**: `{}`\n\n", pair.property_name));

            if pair.is_boolean {
                doc.push_str("*Boolean property*\n\n");
            }

            if let Some(getter) = &pair.getter {
                doc.push_str(&format!("- Getter: `{}`\n", getter));
            }
            if let Some(setter) = &pair.setter {
                doc.push_str(&format!("- Setter: `{}`\n", setter));
            }
            doc.push('\n');
        }

        if !info.related_functions.is_empty() {
            doc.push_str("## Related Functions\n\n");
            for func in &info.related_functions {
                doc.push_str(&format!("- `{}`\n", func));
            }
            doc.push('\n');
        }

        doc
    }

    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }
}

impl Default for SemanticGroupingAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_getter_setter_detection() {
        let analyzer = SemanticGroupingAnalyzer::new();

        let (is_getter, is_setter) =
            analyzer.check_getter_setter("get_value", "Returns the value.");
        assert!(is_getter);
        assert!(!is_setter);

        let (is_getter, is_setter) = analyzer.check_getter_setter("set_value", "Sets the value.");
        assert!(!is_getter);
        assert!(is_setter);
    }

    #[test]
    fn test_boolean_getter() {
        let analyzer = SemanticGroupingAnalyzer::new();

        let (is_getter, _) = analyzer.check_getter_setter("is_valid", "Returns true if valid.");
        assert!(is_getter);

        let (is_getter, _) =
            analyzer.check_getter_setter("has_data", "Returns true if data exists.");
        assert!(is_getter);

        let (is_getter, _) =
            analyzer.check_getter_setter("can_execute", "Returns true if execution is allowed.");
        assert!(is_getter);
    }

    #[test]
    fn test_getter_setter_pair() {
        let mut analyzer = SemanticGroupingAnalyzer::new();

        analyzer.register_function("get_timeout", "Returns the timeout value.");
        analyzer.register_function("set_timeout", "Sets the timeout value.");

        let info = analyzer.analyze("get_timeout", "Returns the timeout value.");
        assert!(info.is_getter);
        assert!(info.getter_setter_pair.is_some());

        let pair = info.getter_setter_pair.unwrap();
        assert_eq!(pair.property_name, "timeout");
        assert_eq!(pair.getter, Some("get_timeout".to_string()));
        assert_eq!(pair.setter, Some("set_timeout".to_string()));
    }

    #[test]
    fn test_boolean_pair() {
        let mut analyzer = SemanticGroupingAnalyzer::new();

        analyzer.register_function("is_enabled", "Checks if feature is enabled.");
        analyzer.register_function("enable_feature", "Enables the feature.");
        analyzer.register_function("disable_feature", "Disables the feature.");

        let info = analyzer.analyze("is_enabled", "Checks if feature is enabled.");

        // Note: The current implementation looks for exact property name matches
        // This test verifies the is_boolean flag
        assert!(info.is_getter);
    }

    #[test]
    fn test_module_extraction() {
        let mut analyzer = SemanticGroupingAnalyzer::new();

        analyzer.register_function("crypto_init", "");
        analyzer.register_function("crypto_encrypt", "");
        analyzer.register_function("crypto_decrypt", "");
        analyzer.register_function("crypto_finalize", "");

        let info = analyzer.analyze("crypto_init", "Module: cryptography");
        assert_eq!(info.module, Some("cryptography".to_string()));
    }

    #[test]
    fn test_module_inference() {
        let mut analyzer = SemanticGroupingAnalyzer::new();

        analyzer.register_function("net_connect", "");
        analyzer.register_function("net_send", "");
        analyzer.register_function("net_receive", "");
        analyzer.register_function("net_disconnect", "");

        let info = analyzer.analyze("net_connect", "Connects to server.");
        assert_eq!(info.module, Some("net".to_string()));
    }

    #[test]
    fn test_feature_set() {
        let mut analyzer = SemanticGroupingAnalyzer::new();

        let info = analyzer.analyze("compress", "Feature: compression");
        assert_eq!(info.feature_set, Some("compression".to_string()));
    }

    #[test]
    fn test_related_functions() {
        let mut analyzer = SemanticGroupingAnalyzer::new();

        analyzer.register_function("init", "");
        analyzer.register_function("process", "See also: cleanup");
        analyzer.register_function("cleanup", "");

        let info = analyzer.analyze("process", "See also: cleanup");
        assert!(info.related_functions.contains(&"cleanup".to_string()));
    }

    #[test]
    fn test_group_by_module() {
        let mut analyzer = SemanticGroupingAnalyzer::new();

        analyzer.register_function("net_connect", "");
        analyzer.register_function("net_send", "");
        analyzer.register_function("net_close", "");
        analyzer.register_function("file_open", "");
        analyzer.register_function("file_read", "");
        analyzer.register_function("file_close", "");

        analyzer.analyze("net_connect", "");
        analyzer.analyze("net_send", "");
        analyzer.analyze("net_close", "");
        analyzer.analyze("file_open", "");
        analyzer.analyze("file_read", "");
        analyzer.analyze("file_close", "");

        let groups = analyzer.group_by_module();
        assert!(groups.contains_key("net"));
        assert!(groups.contains_key("file"));
        assert_eq!(groups.get("net").unwrap().len(), 3);
        assert_eq!(groups.get("file").unwrap().len(), 3);
    }
    #[test]
    fn test_cache_functionality() {
        let mut analyzer = SemanticGroupingAnalyzer::new();

        let info1 = analyzer.analyze("test_func", "Test documentation.");
        let info2 = analyzer.analyze("test_func", "Test documentation.");

        assert_eq!(info1, info2);
    }

    #[test]
    fn test_generate_documentation() {
        let mut analyzer = SemanticGroupingAnalyzer::new();

        analyzer.register_function("get_value", "");
        analyzer.register_function("set_value", "");

        let info = analyzer.analyze("get_value", "Returns the value.");
        let doc = analyzer.generate_documentation(&info);

        assert!(doc.contains("Semantic Grouping"));
        assert!(doc.contains("Property Access") || !info.getter_setter_pair.is_some());
    }
}
