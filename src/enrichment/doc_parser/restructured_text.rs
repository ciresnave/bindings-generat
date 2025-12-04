//! reStructuredText and Sphinx documentation parser.

use super::types::{
    CodeExample, DocParser, DocSection, FunctionDoc, ParamDirection, ParamDoc, ParsedDoc,
};
use anyhow::{Context, Result};
use std::path::PathBuf;

/// reStructuredText/Sphinx parser
pub struct RstParser;

impl RstParser {
    pub fn new() -> Self {
        Self
    }

    /// Parse RST content
    fn parse_rst(&self, content: &str, source_path: PathBuf) -> Result<ParsedDoc> {
        let mut doc = ParsedDoc::new(source_path);
        let lines: Vec<&str> = content.lines().collect();

        let mut i = 0;
        while i < lines.len() {
            let line = lines[i].trim();

            // Check for function documentation (.. c:function:: or .. function::)
            if (line.starts_with(".. c:function::") || line.starts_with(".. function::"))
                && let Some((func_doc, next_i)) = self.parse_function_directive(&lines, i) {
                    doc.add_function(func_doc);
                    i = next_i;
                    continue;
                }

            // Check for code blocks (.. code-block::)
            if line.starts_with(".. code-block::")
                && let Some((example, next_i)) = self.parse_code_block(&lines, i) {
                    doc.add_example(example);
                    i = next_i;
                    continue;
                }

            // Check for section headers
            if i + 1 < lines.len() {
                let next_line = lines[i + 1];
                if self.is_section_underline(next_line) {
                    let level = self.get_section_level(next_line);
                    doc.add_section(DocSection {
                        title: line.to_string(),
                        content: String::new(),
                        level,
                    });
                }
            }

            i += 1;
        }

        Ok(doc)
    }

    fn parse_function_directive(&self, lines: &[&str], start: usize) -> Option<(FunctionDoc, usize)> {
        let directive_line = lines[start];
        
        // Extract function signature
        let sig_start = directive_line.find("::")? + 2;
        let signature = directive_line[sig_start..].trim();
        
        // Extract function name (simple approach - find first identifier before '(')
        let func_name = if let Some(paren_pos) = signature.find('(') {
            signature[..paren_pos].split_whitespace().last()?.to_string()
        } else {
            signature.split_whitespace().last()?.to_string()
        };

        let mut func = FunctionDoc::new(func_name);
        let mut i = start + 1;
        let mut current_section = None;
        let mut content_buffer = String::new();

        // Parse the indented content block
        while i < lines.len() {
            let line = lines[i];
            
            // Check if we've exited the indented block
            if !line.is_empty() && !line.starts_with(' ') && !line.starts_with('\t') {
                break;
            }

            let trimmed = line.trim();

            // Check for field lists (:param:, :returns:, etc.)
            if trimmed.starts_with(":param ") {
                // Save previous section
                if let Some(section) = current_section.take() {
                    self.apply_section_content(&mut func, section, &content_buffer);
                    content_buffer.clear();
                }

                // Parse parameter
                if let Some(param) = self.parse_param_field(trimmed) {
                    func.add_parameter(param);
                }
            } else if trimmed.starts_with(":returns:") || trimmed.starts_with(":return:") {
                // Save previous section
                if let Some(section) = current_section.take() {
                    self.apply_section_content(&mut func, section, &content_buffer);
                    content_buffer.clear();
                }

                current_section = Some("returns");
                // Get content after :returns:
                let content = trimmed.split(':').nth(2).unwrap_or("").trim();
                if !content.is_empty() {
                    content_buffer.push_str(content);
                }
            } else if !trimmed.is_empty() {
                // Regular content line
                if current_section.is_none() && content_buffer.is_empty() {
                    current_section = Some("brief");
                }
                if !content_buffer.is_empty() {
                    content_buffer.push(' ');
                }
                content_buffer.push_str(trimmed);
            }

            i += 1;
        }

        // Apply final section
        if let Some(section) = current_section {
            self.apply_section_content(&mut func, section, &content_buffer);
        }

        Some((func, i))
    }

    fn parse_param_field(&self, field: &str) -> Option<ParamDoc> {
        // Format: :param name: description
        // or: :param type name: description
        let parts: Vec<&str> = field.split(':').collect();
        if parts.len() < 3 {
            return None;
        }

        let param_part = parts[1].trim();
        let description = parts[2..].join(":").trim().to_string();

        // Extract parameter name (handle both "param name" and "param type name")
        let name = param_part
            .strip_prefix("param ")?
            .split_whitespace()
            .last()?
            .to_string();

        Some(ParamDoc {
            name,
            description,
            direction: ParamDirection::In,
            optional: false,
        })
    }

    fn apply_section_content(&self, func: &mut FunctionDoc, section: &str, content: &str) {
        match section {
            "brief" if func.brief.is_none() => {
                func.brief = Some(content.to_string());
            }
            "brief" => {
                // Add to detailed if brief already exists
                func.detailed = Some(content.to_string());
            }
            "returns" => {
                func.return_doc = Some(content.to_string());
            }
            _ => {}
        }
    }

    fn parse_code_block(&self, lines: &[&str], start: usize) -> Option<(CodeExample, usize)> {
        let directive_line = lines[start];
        
        // Extract language
        let language = if let Some(lang_start) = directive_line.find("::") {
            let lang = directive_line[lang_start + 2..].trim();
            if lang.is_empty() {
                None
            } else {
                Some(lang.to_string())
            }
        } else {
            None
        };

        let mut code_lines = Vec::new();
        let mut i = start + 1;

        // Skip empty line after directive
        if i < lines.len() && lines[i].trim().is_empty() {
            i += 1;
        }

        // Collect indented code lines
        while i < lines.len() {
            let line = lines[i];
            if !line.is_empty() && !line.starts_with(' ') && !line.starts_with('\t') {
                break;
            }
            if !line.trim().is_empty() {
                code_lines.push(line);
            }
            i += 1;
        }

        let code = code_lines.join("\n");
        if code.is_empty() {
            return None;
        }

        Some((
            CodeExample {
                title: None,
                code,
                language,
                line_number: Some(start + 1),
            },
            i,
        ))
    }

    fn is_section_underline(&self, line: &str) -> bool {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            return false;
        }

        // Check if line consists only of section marker characters
        let first_char = trimmed.chars().next().unwrap();
        matches!(first_char, '=' | '-' | '~' | '^' | '#' | '*' | '+')
            && trimmed.chars().all(|c| c == first_char)
    }

    fn get_section_level(&self, underline: &str) -> usize {
        let first_char = underline.trim().chars().next().unwrap_or('=');
        match first_char {
            '=' => 1,
            '-' => 2,
            '~' => 3,
            '^' => 4,
            _ => 5,
        }
    }
}

impl DocParser for RstParser {
    fn parse(&self, path: &PathBuf) -> Result<ParsedDoc> {
        let content = std::fs::read_to_string(path)
            .context(format!("Failed to read RST file: {}", path.display()))?;
        
        self.parse_rst(&content, path.clone())
    }

    fn can_parse(&self, path: &PathBuf) -> bool {
        if let Some(ext) = path.extension() {
            return ext == "rst" || ext == "txt";
        }
        false
    }

    fn name(&self) -> &str {
        "reStructuredText"
    }
}

impl Default for RstParser {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rst_parser_creation() {
        let parser = RstParser::new();
        assert_eq!(parser.name(), "reStructuredText");
    }

    #[test]
    fn test_can_parse_rst() {
        let parser = RstParser::new();
        assert!(parser.can_parse(&PathBuf::from("test.rst")));
        assert!(parser.can_parse(&PathBuf::from("README.txt")));
        assert!(!parser.can_parse(&PathBuf::from("test.xml")));
    }

    #[test]
    fn test_is_section_underline() {
        let parser = RstParser::new();
        assert!(parser.is_section_underline("====="));
        assert!(parser.is_section_underline("-----"));
        assert!(parser.is_section_underline("~~~~~"));
        assert!(!parser.is_section_underline("abc"));
        assert!(!parser.is_section_underline("==abc=="));
    }

    #[test]
    fn test_parse_param_field() {
        let parser = RstParser::new();
        
        let param = parser.parse_param_field(":param name: The parameter description");
        assert!(param.is_some());
        let param = param.unwrap();
        assert_eq!(param.name, "name");
        assert_eq!(param.description, "The parameter description");
    }

    #[test]
    fn test_parse_function_directive() {
        let parser = RstParser::new();
        let content = vec![
            ".. c:function:: int test_func(int param1)",
            "   ",
            "   Brief description of the function.",
            "   ",
            "   :param param1: First parameter",
            "   :returns: Status code",
        ];

        let result = parser.parse_function_directive(&content, 0);
        assert!(result.is_some());
        
        let (func, _) = result.unwrap();
        assert_eq!(func.name, "test_func");
        assert!(func.brief.is_some());
        assert_eq!(func.parameters.len(), 1);
        assert!(func.return_doc.is_some());
    }
}
