//! LLM-enhanced documentation orchestration
//!
//! This module orchestrates the integration of LLM-enhanced documentation
//! into the code generation pipeline, coordinating between analyzers and generators.

use crate::analyzer::{
    error_docs::ErrorDocAnalyzer, example_patterns::ExampleAnalyzer, type_docs::TypeDocAnalyzer,
};
use crate::ffi::FfiInfo;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use tracing::{info, warn};

// Placeholder for LLM client
pub struct LlmClient;

impl LlmClient {
    pub async fn query(&self, _prompt: &str) -> Result<String> {
        Ok(String::new())
    }
}

/// LLM-enhanced documentation orchestration result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnhancedDocumentation {
    /// Enhanced function documentation
    pub function_docs: Vec<EnhancedFunctionDoc>,
    /// Enhanced type documentation
    pub type_docs: Vec<EnhancedTypeDoc>,
    /// Enhanced error documentation
    pub error_docs: Vec<EnhancedErrorDoc>,
    /// Code examples
    pub examples: Vec<LlmCodeExample>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnhancedFunctionDoc {
    /// Function name
    pub name: String,
    /// LLM-enhanced description
    pub description: String,
    /// Parameter explanations
    pub parameters: Vec<ParameterDoc>,
    /// Return value explanation
    pub returns: String,
    /// Usage examples
    pub examples: Vec<String>,
    /// Common pitfalls
    pub pitfalls: Vec<String>,
    /// Best practices
    pub best_practices: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterDoc {
    /// Parameter name
    pub name: String,
    /// LLM-enhanced description
    pub description: String,
    /// Valid value ranges
    pub valid_values: Option<String>,
    /// Common mistakes
    pub common_mistakes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnhancedTypeDoc {
    /// Type name
    pub name: String,
    /// LLM-enhanced description
    pub description: String,
    /// Usage patterns
    pub usage_patterns: Vec<String>,
    /// Field descriptions
    pub fields: Vec<FieldDoc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldDoc {
    /// Field name
    pub name: String,
    /// Description
    pub description: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnhancedErrorDoc {
    /// Error name
    pub name: String,
    /// Detailed explanation
    pub explanation: String,
    /// When it occurs
    pub when_occurs: Vec<String>,
    /// How to fix
    pub how_to_fix: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmCodeExample {
    /// Example title
    pub title: String,
    /// Code snippet
    pub code: String,
    /// Explanation
    pub explanation: String,
}

/// Orchestrates LLM-enhanced documentation generation
pub struct LlmDocOrchestrator {
    llm_client: Option<LlmClient>,
    type_doc_analyzer: TypeDocAnalyzer,
    error_doc_analyzer: ErrorDocAnalyzer,
    example_analyzer: ExampleAnalyzer,
}

impl LlmDocOrchestrator {
    pub fn new(llm_client: Option<LlmClient>) -> Self {
        Self {
            llm_client,
            type_doc_analyzer: TypeDocAnalyzer::new(),
            error_doc_analyzer: ErrorDocAnalyzer::new(),
            example_analyzer: ExampleAnalyzer::new(),
        }
    }

    /// Generate enhanced documentation
    pub async fn enhance_documentation(&self, ffi_info: &FfiInfo) -> Result<EnhancedDocumentation> {
        info!("Starting LLM-enhanced documentation generation");

        // Run base analyzers
        let type_analysis = self.type_doc_analyzer.analyze(ffi_info);
        let error_analysis = self.error_doc_analyzer.analyze(ffi_info);
        let example_analysis = self.example_analyzer.analyze(ffi_info, None);

        // Enhance with LLM if available
        let function_docs = if self.llm_client.is_some() {
            self.enhance_function_docs(ffi_info).await?
        } else {
            self.fallback_function_docs(ffi_info)
        };

        let type_docs = if self.llm_client.is_some() {
            self.enhance_type_docs(&type_analysis).await?
        } else {
            self.fallback_type_docs(&type_analysis)
        };

        let error_docs = if self.llm_client.is_some() {
            self.enhance_error_docs(&error_analysis).await?
        } else {
            self.fallback_error_docs(&error_analysis)
        };

        let examples = self.extract_examples(&example_analysis);

        info!(
            "Enhanced documentation: {} functions, {} types, {} errors, {} examples",
            function_docs.len(),
            type_docs.len(),
            error_docs.len(),
            examples.len()
        );

        Ok(EnhancedDocumentation {
            function_docs,
            type_docs,
            error_docs,
            examples,
        })
    }

    /// Enhance function documentation with LLM
    async fn enhance_function_docs(&self, ffi_info: &FfiInfo) -> Result<Vec<EnhancedFunctionDoc>> {
        let llm = self.llm_client.as_ref().unwrap();
        let mut docs = Vec::new();

        for func in &ffi_info.functions {
            let prompt = format!(
                "Enhance documentation for this C function:\n\
                 Function: {}\n\
                 Return type: {}\n\
                 Parameters: {}\n\
                 Existing docs: {:?}\n\n\
                 Provide:\n\
                 1. Clear description\n\
                 2. Parameter explanations\n\
                 3. Return value explanation\n\
                 4. Usage examples\n\
                 5. Common pitfalls\n\
                 6. Best practices",
                func.name,
                func.return_type,
                func.params
                    .iter()
                    .map(|p| format!("{}: {}", p.name, p.ty))
                    .collect::<Vec<_>>()
                    .join(", "),
                func.docs
            );

            match llm.query(&prompt).await {
                Ok(response) => {
                    if let Some(doc) = self.parse_function_doc_response(&response, &func.name) {
                        docs.push(doc);
                    }
                }
                Err(e) => {
                    warn!("LLM enhancement failed for {}: {}", func.name, e);
                }
            }
        }

        Ok(docs)
    }

    /// Parse LLM response for function documentation
    fn parse_function_doc_response(
        &self,
        response: &str,
        func_name: &str,
    ) -> Option<EnhancedFunctionDoc> {
        // Simple parsing - in production would use structured output
        Some(EnhancedFunctionDoc {
            name: func_name.to_string(),
            description: response.lines().next()?.to_string(),
            parameters: Vec::new(),
            returns: "See documentation".to_string(),
            examples: Vec::new(),
            pitfalls: Vec::new(),
            best_practices: Vec::new(),
        })
    }

    /// Fallback function documentation without LLM
    fn fallback_function_docs(&self, ffi_info: &FfiInfo) -> Vec<EnhancedFunctionDoc> {
        ffi_info
            .functions
            .iter()
            .map(|func| EnhancedFunctionDoc {
                name: func.name.clone(),
                description: func
                    .docs
                    .clone()
                    .unwrap_or_else(|| format!("Function {}", func.name)),
                parameters: func
                    .params
                    .iter()
                    .map(|p| ParameterDoc {
                        name: p.name.clone(),
                        description: format!("Parameter of type {}", p.ty),
                        valid_values: None,
                        common_mistakes: Vec::new(),
                    })
                    .collect(),
                returns: format!("Returns {}", func.return_type),
                examples: Vec::new(),
                pitfalls: Vec::new(),
                best_practices: Vec::new(),
            })
            .collect()
    }

    /// Enhance type documentation with LLM
    async fn enhance_type_docs(
        &self,
        _type_analysis: &crate::analyzer::type_docs::TypeDocumentation,
    ) -> Result<Vec<EnhancedTypeDoc>> {
        // Similar to function docs enhancement
        Ok(Vec::new())
    }

    /// Fallback type documentation
    fn fallback_type_docs(
        &self,
        type_analysis: &crate::analyzer::type_docs::TypeDocumentation,
    ) -> Vec<EnhancedTypeDoc> {
        type_analysis
            .type_docs
            .values()
            .map(|ty| EnhancedTypeDoc {
                name: ty.name.clone(),
                description: ty.summary.clone(),
                usage_patterns: Vec::new(),
                fields: Vec::new(),
            })
            .collect()
    }

    /// Enhance error documentation with LLM
    async fn enhance_error_docs(
        &self,
        _error_analysis: &crate::analyzer::error_docs::ErrorDocumentation,
    ) -> Result<Vec<EnhancedErrorDoc>> {
        Ok(Vec::new())
    }

    /// Fallback error documentation
    fn fallback_error_docs(
        &self,
        error_analysis: &crate::analyzer::error_docs::ErrorDocumentation,
    ) -> Vec<EnhancedErrorDoc> {
        error_analysis
            .error_docs
            .values()
            .flat_map(|e| {
                e.variants.values().map(|v| EnhancedErrorDoc {
                    name: v.name.clone(),
                    explanation: v.description.clone(),
                    when_occurs: Vec::new(),
                    how_to_fix: Vec::new(),
                })
            })
            .collect()
    }

    /// Extract code examples
    fn extract_examples(
        &self,
        example_analysis: &crate::analyzer::example_patterns::UsagePatterns,
    ) -> Vec<LlmCodeExample> {
        example_analysis
            .examples
            .values()
            .flatten()
            .map(|ex| LlmCodeExample {
                title: ex.description.clone(),
                code: ex.code.clone(),
                explanation: ex.source.clone(),
            })
            .collect()
    }

    /// Generate Rust doc comments from enhanced documentation
    pub fn generate_doc_comments(&self, func_doc: &EnhancedFunctionDoc) -> String {
        let mut doc = String::new();

        doc.push_str(&format!("/// {}\n", func_doc.description));
        doc.push_str("///\n");

        if !func_doc.parameters.is_empty() {
            doc.push_str("/// # Parameters\n");
            doc.push_str("///\n");
            for param in &func_doc.parameters {
                doc.push_str(&format!("/// * `{}` - {}\n", param.name, param.description));
            }
            doc.push_str("///\n");
        }

        if !func_doc.returns.is_empty() {
            doc.push_str(&format!("/// # Returns\n"));
            doc.push_str("///\n");
            doc.push_str(&format!("/// {}\n", func_doc.returns));
            doc.push_str("///\n");
        }

        if !func_doc.examples.is_empty() {
            doc.push_str("/// # Examples\n");
            doc.push_str("///\n");
            for example in &func_doc.examples {
                doc.push_str("/// ```rust\n");
                for line in example.lines() {
                    doc.push_str(&format!("/// {}\n", line));
                }
                doc.push_str("/// ```\n");
                doc.push_str("///\n");
            }
        }

        if !func_doc.pitfalls.is_empty() {
            doc.push_str("/// # Common Pitfalls\n");
            doc.push_str("///\n");
            for pitfall in &func_doc.pitfalls {
                doc.push_str(&format!("/// * {}\n", pitfall));
            }
            doc.push_str("///\n");
        }

        doc
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ffi::FfiFunction;

    #[test]
    fn test_fallback_function_docs() {
        let orchestrator = LlmDocOrchestrator::new(None);
        let mut ffi_info = FfiInfo::default();

        ffi_info.functions.push(FfiFunction {
            name: "test_func".to_string(),
            params: vec![],
            return_type: "int".to_string(),
            docs: Some("Test function".to_string()),
        });

        let docs = orchestrator.fallback_function_docs(&ffi_info);
        assert_eq!(docs.len(), 1);
        assert_eq!(docs[0].name, "test_func");
    }

    #[test]
    fn test_doc_comment_generation() {
        let orchestrator = LlmDocOrchestrator::new(None);
        let func_doc = EnhancedFunctionDoc {
            name: "test".to_string(),
            description: "Test function".to_string(),
            parameters: vec![ParameterDoc {
                name: "param".to_string(),
                description: "A parameter".to_string(),
                valid_values: None,
                common_mistakes: vec![],
            }],
            returns: "Success code".to_string(),
            examples: vec!["let result = test(param);".to_string()],
            pitfalls: vec!["Don't forget to check errors".to_string()],
            best_practices: vec![],
        };

        let doc = orchestrator.generate_doc_comments(&func_doc);
        assert!(doc.contains("Test function"));
        assert!(doc.contains("Parameters"));
        assert!(doc.contains("param"));
    }
}
