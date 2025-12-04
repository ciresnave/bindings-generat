//! Type definitions for parsed documentation.

use std::collections::HashMap;
use std::path::PathBuf;

/// Trait for documentation parsers
pub trait DocParser: Send + Sync {
    /// Parse a documentation file
    fn parse(&self, path: &PathBuf) -> anyhow::Result<ParsedDoc>;
    
    /// Check if this parser can handle the given file
    fn can_parse(&self, path: &PathBuf) -> bool;
    
    /// Get parser name
    fn name(&self) -> &str;
}

/// Parsed documentation container
#[derive(Debug, Clone, Default)]
pub struct ParsedDoc {
    /// Source file path
    pub source_path: PathBuf,
    /// Function documentation entries
    pub functions: HashMap<String, FunctionDoc>,
    /// General documentation sections
    pub sections: Vec<DocSection>,
    /// Extracted code examples
    pub examples: Vec<CodeExample>,
}

/// Documentation for a single function
#[derive(Debug, Clone)]
pub struct FunctionDoc {
    /// Function name
    pub name: String,
    /// Brief description
    pub brief: Option<String>,
    /// Detailed description
    pub detailed: Option<String>,
    /// Parameter documentation
    pub parameters: Vec<ParamDoc>,
    /// Return value documentation
    pub return_doc: Option<String>,
    /// Example usage
    pub examples: Vec<String>,
    /// Related functions
    pub see_also: Vec<String>,
    /// Deprecation notice
    pub deprecated: Option<String>,
}

/// Documentation for a parameter
#[derive(Debug, Clone)]
pub struct ParamDoc {
    /// Parameter name
    pub name: String,
    /// Parameter description
    pub description: String,
    /// Parameter direction (in, out, inout)
    pub direction: ParamDirection,
    /// Whether parameter is optional
    pub optional: bool,
}

/// Parameter direction
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
            "inout" | "in-out" | "in_out" => Some(Self::InOut),
            _ => None,
        }
    }
}

/// A documentation section
#[derive(Debug, Clone)]
pub struct DocSection {
    /// Section title
    pub title: String,
    /// Section content
    pub content: String,
    /// Nesting level
    pub level: usize,
}

/// A code example from documentation
#[derive(Debug, Clone)]
pub struct CodeExample {
    /// Example title/description
    pub title: Option<String>,
    /// The code snippet
    pub code: String,
    /// Programming language
    pub language: Option<String>,
    /// Line number in source
    pub line_number: Option<usize>,
}

impl ParsedDoc {
    /// Create a new parsed documentation container
    pub fn new(source_path: PathBuf) -> Self {
        Self {
            source_path,
            functions: HashMap::new(),
            sections: Vec::new(),
            examples: Vec::new(),
        }
    }

    /// Add function documentation
    pub fn add_function(&mut self, func_doc: FunctionDoc) {
        self.functions.insert(func_doc.name.clone(), func_doc);
    }

    /// Get function documentation by name
    pub fn get_function(&self, name: &str) -> Option<&FunctionDoc> {
        self.functions.get(name)
    }

    /// Add a documentation section
    pub fn add_section(&mut self, section: DocSection) {
        self.sections.push(section);
    }

    /// Add a code example
    pub fn add_example(&mut self, example: CodeExample) {
        self.examples.push(example);
    }
}

impl FunctionDoc {
    /// Create new function documentation
    pub fn new(name: String) -> Self {
        Self {
            name,
            brief: None,
            detailed: None,
            parameters: Vec::new(),
            return_doc: None,
            examples: Vec::new(),
            see_also: Vec::new(),
            deprecated: None,
        }
    }

    /// Add parameter documentation
    pub fn add_parameter(&mut self, param: ParamDoc) {
        self.parameters.push(param);
    }

    /// Check if function is deprecated
    pub fn is_deprecated(&self) -> bool {
        self.deprecated.is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_param_direction_parsing() {
        assert_eq!(ParamDirection::from_str("in"), Some(ParamDirection::In));
        assert_eq!(ParamDirection::from_str("out"), Some(ParamDirection::Out));
        assert_eq!(ParamDirection::from_str("inout"), Some(ParamDirection::InOut));
        assert_eq!(ParamDirection::from_str("in-out"), Some(ParamDirection::InOut));
        assert_eq!(ParamDirection::from_str("invalid"), None);
    }

    #[test]
    fn test_parsed_doc_operations() {
        let mut doc = ParsedDoc::new(PathBuf::from("test.xml"));
        
        let mut func = FunctionDoc::new("test_function".to_string());
        func.brief = Some("Test function".to_string());
        func.add_parameter(ParamDoc {
            name: "param1".to_string(),
            description: "First parameter".to_string(),
            direction: ParamDirection::In,
            optional: false,
        });
        
        doc.add_function(func);
        
        let retrieved = doc.get_function("test_function");
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().brief, Some("Test function".to_string()));
    }
}
