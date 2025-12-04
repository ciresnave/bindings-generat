//! Doxygen XML documentation parser.

use super::types::{
    DocParser, FunctionDoc, ParamDirection, ParamDoc, ParsedDoc,
};
use anyhow::{Context, Result};
use std::path::PathBuf;

/// Doxygen XML parser
pub struct DoxygenParser;

impl DoxygenParser {
    pub fn new() -> Self {
        Self
    }

    /// Parse a doxygen compound XML file
    fn parse_xml(&self, content: &str) -> Result<ParsedDoc> {
        // For now, use a simple text-based parser
        // TODO: Use proper XML parser like quick-xml
        let mut doc = ParsedDoc::new(PathBuf::new());

        // Extract function documentation using simple pattern matching
        self.extract_functions(content, &mut doc)?;

        Ok(doc)
    }

    fn extract_functions(&self, content: &str, doc: &mut ParsedDoc) -> Result<()> {
        // Look for <memberdef kind="function"> blocks
        for section in content.split("<memberdef kind=\"function\">") {
            if let Some(end_pos) = section.find("</memberdef>") {
                let func_section = &section[..end_pos];
                if let Some(func_doc) = self.parse_function_section(func_section) {
                    doc.add_function(func_doc);
                }
            }
        }
        Ok(())
    }

    fn parse_function_section(&self, section: &str) -> Option<FunctionDoc> {
        // Extract function name
        let name = self.extract_tag_content(section, "name")?;
        let mut func = FunctionDoc::new(name);

        // Extract brief description
        if let Some(brief) = self.extract_tag_content(section, "briefdescription")
            && !brief.trim().is_empty() {
                func.brief = Some(self.clean_html(&brief));
            }

        // Extract detailed description
        if let Some(detailed) = self.extract_tag_content(section, "detaileddescription")
            && !detailed.trim().is_empty() {
                func.detailed = Some(self.clean_html(&detailed));
            }

        // Extract parameters
        for param_section in section.split("<param>").skip(1) {
            if let Some(end_pos) = param_section.find("</param>") {
                let param_content = &param_section[..end_pos];
                
                if let Some(param_name) = self.extract_tag_content(param_content, "declname") {
                    // Find parameter documentation in detaileddescription
                    let param_desc = self.find_param_description(section, &param_name);
                    
                    func.add_parameter(ParamDoc {
                        name: param_name,
                        description: param_desc.unwrap_or_default(),
                        direction: ParamDirection::In, // Default, can be refined
                        optional: false,
                    });
                }
            }
        }

        Some(func)
    }

    fn extract_tag_content(&self, text: &str, tag: &str) -> Option<String> {
        let start_tag = format!("<{}>", tag);
        let end_tag = format!("</{}>", tag);
        
        if let Some(start_pos) = text.find(&start_tag) {
            let content_start = start_pos + start_tag.len();
            if let Some(end_pos) = text[content_start..].find(&end_tag) {
                return Some(text[content_start..content_start + end_pos].trim().to_string());
            }
        }
        None
    }

    fn find_param_description(&self, section: &str, param_name: &str) -> Option<String> {
        // Look for <parameteritem> with matching parameter name
        for param_item in section.split("<parameteritem>") {
            if param_item.contains(&format!("<parametername>{}</parametername>", param_name))
                && let Some(desc) = self.extract_tag_content(param_item, "parameterdescription") {
                    return Some(self.clean_html(&desc));
                }
        }
        None
    }

    fn clean_html(&self, text: &str) -> String {
        // Remove XML/HTML tags
        let mut result = String::new();
        let mut in_tag = false;
        
        for c in text.chars() {
            match c {
                '<' => in_tag = true,
                '>' => in_tag = false,
                _ if !in_tag => result.push(c),
                _ => {}
            }
        }
        
        // Clean up whitespace
        result.split_whitespace().collect::<Vec<_>>().join(" ")
    }
}

impl DocParser for DoxygenParser {
    fn parse(&self, path: &PathBuf) -> Result<ParsedDoc> {
        let content = std::fs::read_to_string(path)
            .context(format!("Failed to read doxygen file: {}", path.display()))?;
        
        let mut doc = self.parse_xml(&content)?;
        doc.source_path = path.clone();
        
        Ok(doc)
    }

    fn can_parse(&self, path: &PathBuf) -> bool {
        if let Some(ext) = path.extension()
            && ext == "xml" {
                // Check if it's a doxygen file by looking for doxygen markers
                if let Ok(content) = std::fs::read_to_string(path) {
                    return content.contains("<doxygen") || content.contains("doxygenindex");
                }
            }
        false
    }

    fn name(&self) -> &str {
        "Doxygen"
    }
}

impl Default for DoxygenParser {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_doxygen_parser_creation() {
        let parser = DoxygenParser::new();
        assert_eq!(parser.name(), "Doxygen");
    }

    #[test]
    fn test_can_parse_xml() {
        let parser = DoxygenParser::new();
        
        // Create temp XML file
        let temp_dir = std::env::temp_dir();
        let xml_path = temp_dir.join("test_doxygen.xml");
        std::fs::write(&xml_path, "<doxygen><compound></compound></doxygen>").unwrap();
        
        assert!(parser.can_parse(&xml_path));
        
        // Cleanup
        let _ = std::fs::remove_file(&xml_path);
    }

    #[test]
    fn test_clean_html() {
        let parser = DoxygenParser::new();
        let html = "<para>This is <bold>bold</bold> text.</para>";
        let cleaned = parser.clean_html(html);
        assert_eq!(cleaned, "This is bold text.");
    }

    #[test]
    fn test_extract_tag_content() {
        let parser = DoxygenParser::new();
        let xml = "<name>test_function</name>";
        let content = parser.extract_tag_content(xml, "name");
        assert_eq!(content, Some("test_function".to_string()));
    }
}
