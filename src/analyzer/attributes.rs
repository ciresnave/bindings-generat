//! Compiler attribute extraction from C/C++ headers.
//!
//! This module analyzes function declarations to extract compiler-specific attributes
//! (GCC/Clang, MSVC, C23) and maps them to appropriate Rust equivalents.

use std::collections::HashMap;

/// Type of compiler attribute
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AttributeType {
    /// Function is deprecated
    Deprecated { since: Option<String>, note: Option<String> },
    /// Return value must be used
    MustUse { reason: Option<String> },
    /// Function is const (no side effects, deterministic)
    Const,
    /// Function is pure (no side effects, but may read globals)
    Pure,
    /// Function returns newly allocated memory
    Malloc,
    /// Specified parameters must not be null
    NonNull { param_indices: Vec<usize> },
    /// Function never returns
    NoReturn,
    /// Function has restricted pointer semantics
    Restrict,
    /// Sentinel value marks end of variadic args
    Sentinel { position: usize },
    /// Custom or library-specific attribute
    Custom { name: String, value: Option<String> },
}

/// Source of the attribute
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AttributeSource {
    /// GCC/Clang __attribute__
    GccAttribute,
    /// MSVC __declspec
    MsvcDeclspec,
    /// C23 [[attribute]]
    C23Attribute,
    /// Custom annotation in documentation
    Documentation,
}

/// A single extracted attribute
#[derive(Debug, Clone, PartialEq)]
pub struct Attribute {
    /// Type of attribute
    pub attr_type: AttributeType,
    /// Source of the attribute
    pub source: AttributeSource,
    /// Raw attribute text
    pub raw_text: String,
    /// Confidence in the extraction (0.0-1.0)
    pub confidence: f64,
}

/// Complete attribute analysis for a function
#[derive(Debug, Clone, PartialEq)]
pub struct AttributeInfo {
    /// Function name
    pub function_name: String,
    /// All extracted attributes
    pub attributes: Vec<Attribute>,
    /// Generated Rust attributes
    pub rust_attributes: Vec<String>,
    /// Documentation about attributes
    pub documentation: Vec<String>,
    /// Overall confidence
    pub confidence: f64,
}

impl AttributeInfo {
    /// Create new empty attribute info
    pub fn new(function_name: String) -> Self {
        Self {
            function_name,
            attributes: Vec::new(),
            rust_attributes: Vec::new(),
            documentation: Vec::new(),
            confidence: 0.0,
        }
    }

    /// Check if any attributes were found
    pub fn has_attributes(&self) -> bool {
        !self.attributes.is_empty()
    }

    /// Check if function is deprecated
    pub fn is_deprecated(&self) -> bool {
        self.attributes
            .iter()
            .any(|a| matches!(a.attr_type, AttributeType::Deprecated { .. }))
    }

    /// Check if return value must be used
    pub fn is_must_use(&self) -> bool {
        self.attributes
            .iter()
            .any(|a| matches!(a.attr_type, AttributeType::MustUse { .. }))
    }

    /// Check if function is const or pure
    pub fn is_const_or_pure(&self) -> bool {
        self.attributes.iter().any(|a| {
            matches!(a.attr_type, AttributeType::Const | AttributeType::Pure)
        })
    }

    /// Get non-null parameter indices
    pub fn non_null_params(&self) -> Vec<usize> {
        self.attributes
            .iter()
            .filter_map(|a| {
                if let AttributeType::NonNull { param_indices } = &a.attr_type {
                    Some(param_indices.clone())
                } else {
                    None
                }
            })
            .flatten()
            .collect()
    }

    /// Generate documentation string for attributes
    pub fn generate_documentation(&self) -> String {
        let mut doc = String::new();
        
        for line in &self.documentation {
            doc.push_str(&format!("/// {}\n", line));
        }
        
        doc
    }
}

impl Default for AttributeInfo {
    fn default() -> Self {
        Self::new(String::new())
    }
}

/// Analyzer for extracting compiler attributes
#[derive(Debug)]
pub struct AttributeAnalyzer {
    /// Cache of analyzed functions
    cache: HashMap<String, AttributeInfo>,
}

impl AttributeAnalyzer {
    /// Create a new attribute analyzer
    pub fn new() -> Self {
        Self {
            cache: HashMap::new(),
        }
    }

    /// Analyze function declaration for attributes
    ///
    /// # Arguments
    /// * `function_name` - Name of the function
    /// * `declaration` - Function declaration text
    /// * `documentation` - Optional documentation text
    pub fn analyze(
        &mut self,
        function_name: &str,
        declaration: &str,
        documentation: Option<&str>,
    ) -> AttributeInfo {
        // Check cache first
        if let Some(cached) = self.cache.get(function_name) {
            return cached.clone();
        }

        let mut info = AttributeInfo::new(function_name.to_string());

        // Extract GCC/Clang attributes
        self.extract_gcc_attributes(declaration, &mut info);

        // Extract MSVC attributes
        self.extract_msvc_attributes(declaration, &mut info);

        // Extract C23 attributes
        self.extract_c23_attributes(declaration, &mut info);

        // Extract from documentation
        if let Some(doc) = documentation {
            self.extract_doc_attributes(doc, &mut info);
        }

        // Generate Rust attributes
        self.generate_rust_attributes(&mut info);

        // Generate documentation
        self.generate_documentation(&mut info);

        // Calculate confidence
        info.confidence = self.calculate_confidence(&info);

        // Cache the result
        self.cache.insert(function_name.to_string(), info.clone());

        info
    }

    /// Extract GCC/Clang __attribute__ annotations
    fn extract_gcc_attributes(&self, declaration: &str, info: &mut AttributeInfo) {
        // Look for __attribute__((name))
        if let Some(start) = declaration.find("__attribute__((") {
            let rest = &declaration[start + 15..];
            if let Some(end) = rest.find("))") {
                let attrs_text = &rest[..end];
                
                // Parse individual attributes
                for attr in attrs_text.split(',') {
                    let attr = attr.trim();
                    
                    if attr.starts_with("deprecated") {
                        let (since, note) = self.parse_deprecated_params(attr);
                        info.attributes.push(Attribute {
                            attr_type: AttributeType::Deprecated { since, note },
                            source: AttributeSource::GccAttribute,
                            raw_text: format!("__attribute__(({})))", attr),
                            confidence: 0.95,
                        });
                    } else if attr == "warn_unused_result" {
                        info.attributes.push(Attribute {
                            attr_type: AttributeType::MustUse { reason: None },
                            source: AttributeSource::GccAttribute,
                            raw_text: "__attribute__((warn_unused_result))".to_string(),
                            confidence: 0.95,
                        });
                    } else if attr == "const" {
                        info.attributes.push(Attribute {
                            attr_type: AttributeType::Const,
                            source: AttributeSource::GccAttribute,
                            raw_text: "__attribute__((const))".to_string(),
                            confidence: 0.9,
                        });
                    } else if attr == "pure" {
                        info.attributes.push(Attribute {
                            attr_type: AttributeType::Pure,
                            source: AttributeSource::GccAttribute,
                            raw_text: "__attribute__((pure))".to_string(),
                            confidence: 0.9,
                        });
                    } else if attr == "malloc" {
                        info.attributes.push(Attribute {
                            attr_type: AttributeType::Malloc,
                            source: AttributeSource::GccAttribute,
                            raw_text: "__attribute__((malloc))".to_string(),
                            confidence: 0.9,
                        });
                    } else if attr.starts_with("nonnull") {
                        let indices = self.parse_nonnull_params(attr);
                        info.attributes.push(Attribute {
                            attr_type: AttributeType::NonNull { param_indices: indices },
                            source: AttributeSource::GccAttribute,
                            raw_text: format!("__attribute__(({})))", attr),
                            confidence: 0.9,
                        });
                    } else if attr == "noreturn" {
                        info.attributes.push(Attribute {
                            attr_type: AttributeType::NoReturn,
                            source: AttributeSource::GccAttribute,
                            raw_text: "__attribute__((noreturn))".to_string(),
                            confidence: 0.95,
                        });
                    }
                }
            }
        }
    }

    /// Extract MSVC __declspec attributes
    fn extract_msvc_attributes(&self, declaration: &str, info: &mut AttributeInfo) {
        // Look for __declspec(name)
        if let Some(start) = declaration.find("__declspec(") {
            let rest = &declaration[start + 11..];
            if let Some(end) = rest.find(')') {
                let attr = rest[..end].trim();
                
                if attr.starts_with("deprecated") {
                    let (since, note) = self.parse_deprecated_params(attr);
                    info.attributes.push(Attribute {
                        attr_type: AttributeType::Deprecated { since, note },
                        source: AttributeSource::MsvcDeclspec,
                        raw_text: format!("__declspec({})", attr),
                        confidence: 0.95,
                    });
                } else if attr == "noreturn" {
                    info.attributes.push(Attribute {
                        attr_type: AttributeType::NoReturn,
                        source: AttributeSource::MsvcDeclspec,
                        raw_text: "__declspec(noreturn)".to_string(),
                        confidence: 0.95,
                    });
                } else if attr == "restrict" {
                    info.attributes.push(Attribute {
                        attr_type: AttributeType::Restrict,
                        source: AttributeSource::MsvcDeclspec,
                        raw_text: "__declspec(restrict)".to_string(),
                        confidence: 0.9,
                    });
                }
            }
        }
    }

    /// Extract C23 [[attribute]] syntax
    fn extract_c23_attributes(&self, declaration: &str, info: &mut AttributeInfo) {
        // Look for [[name]]
        let mut pos = 0;
        while let Some(start) = declaration[pos..].find("[[") {
            let abs_start = pos + start;
            let rest = &declaration[abs_start + 2..];
            
            if let Some(end) = rest.find("]]") {
                let attr = rest[..end].trim();
                
                if attr.starts_with("deprecated") {
                    let (since, note) = self.parse_deprecated_params(attr);
                    info.attributes.push(Attribute {
                        attr_type: AttributeType::Deprecated { since, note },
                        source: AttributeSource::C23Attribute,
                        raw_text: format!("[[{}]]", attr),
                        confidence: 0.95,
                    });
                } else if attr == "nodiscard" {
                    info.attributes.push(Attribute {
                        attr_type: AttributeType::MustUse { reason: None },
                        source: AttributeSource::C23Attribute,
                        raw_text: "[[nodiscard]]".to_string(),
                        confidence: 0.95,
                    });
                } else if attr == "noreturn" {
                    info.attributes.push(Attribute {
                        attr_type: AttributeType::NoReturn,
                        source: AttributeSource::C23Attribute,
                        raw_text: "[[noreturn]]".to_string(),
                        confidence: 0.95,
                    });
                }
                
                pos = abs_start + end + 2;
            } else {
                break;
            }
        }
    }

    /// Extract attributes from documentation
    fn extract_doc_attributes(&self, doc: &str, info: &mut AttributeInfo) {
        let doc_lower = doc.to_lowercase();

        // Check for deprecation notices
        if doc_lower.contains("deprecated") || doc_lower.contains("obsolete") {
            let note = if let Some(idx) = doc_lower.find("use") {
                // Try to extract what to use instead
                let rest = &doc[idx..];
                rest.find(['.', '\n'].as_ref()).map(|end| rest[..end].trim().to_string())
            } else {
                None
            };

            info.attributes.push(Attribute {
                attr_type: AttributeType::Deprecated { since: None, note },
                source: AttributeSource::Documentation,
                raw_text: "deprecated (from docs)".to_string(),
                confidence: 0.7,
            });
        }

        // Check for must-use hints
        if doc_lower.contains("must use") || doc_lower.contains("must check") 
            || doc_lower.contains("ignoring") && doc_lower.contains("may cause") {
            info.attributes.push(Attribute {
                attr_type: AttributeType::MustUse { reason: Some("return value must be checked".to_string()) },
                source: AttributeSource::Documentation,
                raw_text: "must_use (from docs)".to_string(),
                confidence: 0.6,
            });
        }

        // Check for const/pure hints
        if doc_lower.contains("no side effects") || doc_lower.contains("pure function") {
            info.attributes.push(Attribute {
                attr_type: AttributeType::Pure,
                source: AttributeSource::Documentation,
                raw_text: "pure (from docs)".to_string(),
                confidence: 0.5,
            });
        }
    }

    /// Parse deprecated attribute parameters
    fn parse_deprecated_params(&self, attr: &str) -> (Option<String>, Option<String>) {
        let since = None;
        let mut note = None;

        // Look for quoted strings (GCC/Clang style)
        if let Some(quote_start) = attr.find('"')
            && let Some(quote_end) = attr[quote_start + 1..].find('"') {
                let message = &attr[quote_start + 1..quote_start + 1 + quote_end];
                note = Some(message.to_string());
            }

        (since, note)
    }

    /// Parse nonnull attribute parameters
    fn parse_nonnull_params(&self, attr: &str) -> Vec<usize> {
        let mut indices = Vec::new();

        // Look for nonnull(1, 2, 3) pattern
        if let Some(paren_start) = attr.find('(')
            && let Some(paren_end) = attr[paren_start..].find(')') {
                let params = &attr[paren_start + 1..paren_start + paren_end];
                for param in params.split(',') {
                    if let Ok(idx) = param.trim().parse::<usize>() {
                        indices.push(idx);
                    }
                }
            }

        // If no specific indices, it applies to all pointer parameters
        if indices.is_empty() {
            indices.push(0); // Sentinel value meaning "all"
        }

        indices
    }

    /// Generate Rust attribute strings
    fn generate_rust_attributes(&self, info: &mut AttributeInfo) {
        for attr in &info.attributes {
            match &attr.attr_type {
                AttributeType::Deprecated { since, note } => {
                    let mut rust_attr = "#[deprecated".to_string();
                    if let Some(s) = since {
                        rust_attr.push_str(&format!("(since = \"{}\"", s));
                        if let Some(n) = note {
                            rust_attr.push_str(&format!(", note = \"{}\")", n));
                        } else {
                            rust_attr.push(')');
                        }
                    } else if let Some(n) = note {
                        rust_attr.push_str(&format!("(note = \"{}\")", n));
                    }
                    rust_attr.push(']');
                    info.rust_attributes.push(rust_attr);
                }
                AttributeType::MustUse { reason } => {
                    if let Some(r) = reason {
                        info.rust_attributes.push(format!("#[must_use = \"{}\"]", r));
                    } else {
                        info.rust_attributes.push("#[must_use]".to_string());
                    }
                }
                _ => {
                    // Other attributes are documented but not mapped to Rust attributes
                }
            }
        }
    }

    /// Generate documentation about attributes
    fn generate_documentation(&self, info: &mut AttributeInfo) {
        for attr in &info.attributes {
            match &attr.attr_type {
                AttributeType::Deprecated { note, .. } => {
                    let mut doc = "**Deprecated:** ".to_string();
                    if let Some(n) = note {
                        doc.push_str(n);
                    } else {
                        doc.push_str("This function is deprecated and should not be used in new code.");
                    }
                    info.documentation.push(doc);
                }
                AttributeType::MustUse { reason } => {
                    let mut doc = "**Must Use Result:** ".to_string();
                    if let Some(r) = reason {
                        doc.push_str(r);
                    } else {
                        doc.push_str("The return value must be used; ignoring it may cause errors.");
                    }
                    info.documentation.push(doc);
                }
                AttributeType::Const => {
                    info.documentation.push(
                        "**Const Function:** Pure function with no side effects. \
                         Result depends only on input parameters. May be marked `const fn` in future Rust versions."
                            .to_string(),
                    );
                }
                AttributeType::Pure => {
                    info.documentation.push(
                        "**Pure Function:** Function has no side effects but may read global state. \
                         Safe to call multiple times with same arguments."
                            .to_string(),
                    );
                }
                AttributeType::Malloc => {
                    info.documentation.push(
                        "**Allocates Memory:** Returns newly allocated memory that must be freed by the caller."
                            .to_string(),
                    );
                }
                AttributeType::NonNull { param_indices } => {
                    if param_indices.contains(&0) {
                        info.documentation.push(
                            "**Non-null Parameters:** All pointer parameters must be valid (non-null)."
                                .to_string(),
                        );
                    } else {
                        info.documentation.push(format!(
                            "**Non-null Parameters:** Parameters at positions {:?} must not be null.",
                            param_indices
                        ));
                    }
                }
                AttributeType::NoReturn => {
                    info.documentation.push(
                        "**Never Returns:** This function never returns to the caller (e.g., exits the program)."
                            .to_string(),
                    );
                }
                AttributeType::Restrict => {
                    info.documentation.push(
                        "**Restrict Semantics:** Pointer parameters have restricted aliasing (no overlapping memory)."
                            .to_string(),
                    );
                }
                AttributeType::Sentinel { position } => {
                    info.documentation.push(format!(
                        "**Sentinel Value:** Variadic arguments must be terminated with a sentinel value at position {}.",
                        position
                    ));
                }
                AttributeType::Custom { name, value } => {
                    let mut doc = format!("**Custom Attribute:** {}", name);
                    if let Some(v) = value {
                        doc.push_str(&format!(" = {}", v));
                    }
                    info.documentation.push(doc);
                }
            }
        }
    }

    /// Calculate overall confidence
    fn calculate_confidence(&self, info: &AttributeInfo) -> f64 {
        if info.attributes.is_empty() {
            return 0.0;
        }

        let sum: f64 = info.attributes.iter().map(|a| a.confidence).sum();
        let count = info.attributes.len() as f64;

        (sum / count).min(1.0)
    }

    /// Generate complete documentation including attributes
    pub fn generate_docs(&self, info: &AttributeInfo) -> String {
        let mut docs = String::new();

        if info.documentation.is_empty() {
            return docs;
        }

        for doc in &info.documentation {
            docs.push_str(&format!("/// {}\n", doc));
        }
        docs.push_str("///\n");

        docs
    }

    /// Clear the cache
    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }
}

impl Default for AttributeAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gcc_deprecated() {
        let mut analyzer = AttributeAnalyzer::new();
        let decl = r#"void old_func() __attribute__((deprecated));"#;
        
        let info = analyzer.analyze("old_func", decl, None);
        
        assert!(info.is_deprecated());
        assert!(!info.rust_attributes.is_empty());
        assert!(info.rust_attributes[0].contains("#[deprecated"));
    }

    #[test]
    fn test_gcc_warn_unused_result() {
        let mut analyzer = AttributeAnalyzer::new();
        let decl = r#"int important() __attribute__((warn_unused_result));"#;
        
        let info = analyzer.analyze("important", decl, None);
        
        assert!(info.is_must_use());
        assert!(info.rust_attributes.contains(&"#[must_use]".to_string()));
    }

    #[test]
    fn test_gcc_const() {
        let mut analyzer = AttributeAnalyzer::new();
        let decl = r#"int compute(int x) __attribute__((const));"#;
        
        let info = analyzer.analyze("compute", decl, None);
        
        assert!(info.is_const_or_pure());
        assert!(!info.documentation.is_empty());
        assert!(info.documentation[0].contains("Const Function"));
    }

    #[test]
    fn test_gcc_pure() {
        let mut analyzer = AttributeAnalyzer::new();
        let decl = r#"int calculate(int x) __attribute__((pure));"#;
        
        let info = analyzer.analyze("calculate", decl, None);
        
        assert!(info.is_const_or_pure());
        assert!(info.documentation[0].contains("Pure Function"));
    }

    #[test]
    fn test_gcc_malloc() {
        let mut analyzer = AttributeAnalyzer::new();
        let decl = r#"void* allocate(size_t n) __attribute__((malloc));"#;
        
        let info = analyzer.analyze("allocate", decl, None);
        
        assert!(info.has_attributes());
        let malloc_attr = info.attributes.iter().find(|a| matches!(a.attr_type, AttributeType::Malloc));
        assert!(malloc_attr.is_some());
    }

    #[test]
    fn test_gcc_nonnull() {
        let mut analyzer = AttributeAnalyzer::new();
        let decl = r#"void process(void* data) __attribute__((nonnull));"#;
        
        let info = analyzer.analyze("process", decl, None);
        
        let nonnull_params = info.non_null_params();
        assert!(!nonnull_params.is_empty());
    }

    #[test]
    fn test_msvc_deprecated() {
        let mut analyzer = AttributeAnalyzer::new();
        let decl = r#"__declspec(deprecated) void old_func();"#;
        
        let info = analyzer.analyze("old_func", decl, None);
        
        assert!(info.is_deprecated());
    }

    #[test]
    fn test_msvc_noreturn() {
        let mut analyzer = AttributeAnalyzer::new();
        let decl = r#"__declspec(noreturn) void exit_program();"#;
        
        let info = analyzer.analyze("exit_program", decl, None);
        
        assert!(info.has_attributes());
        let noreturn = info.attributes.iter().find(|a| matches!(a.attr_type, AttributeType::NoReturn));
        assert!(noreturn.is_some());
    }

    #[test]
    fn test_c23_deprecated() {
        let mut analyzer = AttributeAnalyzer::new();
        let decl = r#"[[deprecated]] void old_func();"#;
        
        let info = analyzer.analyze("old_func", decl, None);
        
        assert!(info.is_deprecated());
    }

    #[test]
    fn test_c23_nodiscard() {
        let mut analyzer = AttributeAnalyzer::new();
        let decl = r#"[[nodiscard]] int critical();"#;
        
        let info = analyzer.analyze("critical", decl, None);
        
        assert!(info.is_must_use());
    }

    #[test]
    fn test_doc_deprecated() {
        let mut analyzer = AttributeAnalyzer::new();
        let decl = "void func();";
        let doc = "This function is deprecated. Use new_func() instead.";
        
        let info = analyzer.analyze("func", decl, Some(doc));
        
        assert!(info.is_deprecated());
        assert!(!info.documentation.is_empty());
    }

    #[test]
    fn test_doc_must_use() {
        let mut analyzer = AttributeAnalyzer::new();
        let decl = "int func();";
        let doc = "Return value must be checked as ignoring may cause errors.";
        
        let info = analyzer.analyze("func", decl, Some(doc));
        
        assert!(info.is_must_use());
    }

    #[test]
    fn test_multiple_attributes() {
        let mut analyzer = AttributeAnalyzer::new();
        let decl = r#"void* alloc(size_t n) __attribute__((malloc, warn_unused_result));"#;
        
        let info = analyzer.analyze("alloc", decl, None);
        
        assert!(info.is_must_use());
        assert!(info.attributes.len() >= 2);
    }

    #[test]
    fn test_confidence_calculation() {
        let mut analyzer = AttributeAnalyzer::new();
        
        // No attributes = 0 confidence
        let empty_info = AttributeInfo::new("test".to_string());
        assert_eq!(analyzer.calculate_confidence(&empty_info), 0.0);
        
        // With attributes = average confidence
        let decl = r#"void func() __attribute__((deprecated));"#;
        let info = analyzer.analyze("func", decl, None);
        assert!(info.confidence > 0.5);
    }

    #[test]
    fn test_cache() {
        let mut analyzer = AttributeAnalyzer::new();
        let decl = r#"void func() __attribute__((const));"#;
        
        let info1 = analyzer.analyze("func", decl, None);
        let info2 = analyzer.analyze("func", decl, None);
        
        assert_eq!(info1, info2);
        
        analyzer.clear_cache();
        let info3 = analyzer.analyze("func", decl, None);
        assert_eq!(info1, info3);
    }
}
