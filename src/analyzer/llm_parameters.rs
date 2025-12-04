//! LLM-powered parameter analysis
//!
//! This module uses LLM to analyze parameter relationships, constraints,
//! and semantic meaning beyond what can be extracted from documentation alone.

use crate::ffi::{FfiFunction, FfiInfo};
use anyhow::Result;

// Placeholder for LLM client
pub struct LlmClient;

impl LlmClient {
    pub async fn query(&self, _prompt: &str) -> Result<String> {
        Ok(String::new())
    }
}
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::{info, warn};

/// LLM-analyzed parameter information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmParameterAnalysis {
    /// Analysis per function
    pub function_analysis: HashMap<String, FunctionParamAnalysis>,
    /// Global parameter patterns
    pub common_patterns: Vec<ParameterPattern>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionParamAnalysis {
    /// Function name
    pub function: String,
    /// Parameter semantic roles
    pub param_roles: Vec<ParameterRole>,
    /// Parameter relationships
    pub relationships: Vec<ParameterRelationship>,
    /// Inferred constraints
    pub constraints: Vec<InferredConstraint>,
    /// Usage recommendations
    pub recommendations: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterRole {
    /// Parameter name
    pub param_name: String,
    /// Semantic role
    pub role: SemanticRole,
    /// Confidence (0.0-1.0)
    pub confidence: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum SemanticRole {
    /// Input data source
    Input,
    /// Output destination
    Output,
    /// Configuration/settings
    Configuration,
    /// Size/dimension
    Size,
    /// Flags/options
    Flags,
    /// Handle/context
    Handle,
    /// Callback function
    Callback,
    /// User data for callback
    UserData,
    /// Error reporting
    Error,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterRelationship {
    /// First parameter
    pub param1: String,
    /// Second parameter
    pub param2: String,
    /// Relationship type
    pub relationship: RelationshipType,
    /// Description
    pub description: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum RelationshipType {
    /// param1 is size of param2
    SizeOf,
    /// param1 must match dimension of param2
    DimensionMatch,
    /// param1 and param2 are mutually exclusive
    MutuallyExclusive,
    /// param1 required when param2 is set
    RequiredWith,
    /// param1 and param2 must have same alignment
    AlignmentMatch,
    /// param1 describes format of param2
    FormatDescriptor,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferredConstraint {
    /// Parameter name
    pub param_name: String,
    /// Constraint type
    pub constraint_type: ConstraintType,
    /// Constraint value/description
    pub value: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ConstraintType {
    /// Must be power of 2
    PowerOfTwo,
    /// Must be aligned to boundary
    Alignment,
    /// Must be multiple of value
    MultipleOf,
    /// Valid range
    Range,
    /// Specific valid values
    Enum,
    /// Must not be null
    NonNull,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterPattern {
    /// Pattern name
    pub name: String,
    /// Description
    pub description: String,
    /// Example functions
    pub examples: Vec<String>,
}

/// LLM-powered parameter analyzer
pub struct LlmParameterAnalyzer {
    llm_client: Option<LlmClient>,
}

impl LlmParameterAnalyzer {
    pub fn new(llm_client: Option<LlmClient>) -> Self {
        Self { llm_client }
    }

    /// Analyze parameters using LLM
    pub async fn analyze(&self, ffi_info: &FfiInfo) -> Result<LlmParameterAnalysis> {
        if self.llm_client.is_none() {
            warn!("LLM client not available, using fallback analysis");
            return Ok(self.fallback_analysis(ffi_info));
        }

        let mut function_analysis = HashMap::new();

        for func in &ffi_info.functions {
            if let Ok(analysis) = self.analyze_function(func).await {
                function_analysis.insert(func.name.clone(), analysis);
            }
        }

        let common_patterns = self.identify_common_patterns(&function_analysis);

        info!(
            "LLM parameter analysis: {} functions, {} patterns",
            function_analysis.len(),
            common_patterns.len()
        );

        Ok(LlmParameterAnalysis {
            function_analysis,
            common_patterns,
        })
    }

    /// Analyze single function's parameters
    async fn analyze_function(&self, func: &FfiFunction) -> Result<FunctionParamAnalysis> {
        let llm_client = self.llm_client.as_ref().unwrap();

        // Build prompt for LLM
        let prompt = self.build_analysis_prompt(func);

        // Query LLM
        let response = llm_client.query(&prompt).await?;

        // Parse response
        self.parse_llm_response(&response, func)
    }

    /// Build prompt for LLM analysis
    fn build_analysis_prompt(&self, func: &FfiFunction) -> String {
        let mut prompt = format!(
            "Analyze the parameters of this C function:\n\n\
             Function: {}\n\
             Return type: {}\n\
             Parameters:\n",
            func.name, func.return_type
        );

        for param in &func.params {
            prompt.push_str(&format!("  - {} (type: {})\n", param.name, param.ty));
        }

        if let Some(ref docs) = func.docs {
            prompt.push_str(&format!("\nDocumentation:\n{}\n", docs));
        }

        prompt.push_str(
            "\nProvide:\n\
             1. Semantic role of each parameter (input/output/config/size/etc)\n\
             2. Relationships between parameters (e.g., param1 is size of param2)\n\
             3. Inferred constraints (alignment, power of 2, non-null, etc)\n\
             4. Usage recommendations\n\n\
             Format as JSON.",
        );

        prompt
    }

    /// Parse LLM response into structured analysis
    fn parse_llm_response(
        &self,
        response: &str,
        func: &FfiFunction,
    ) -> Result<FunctionParamAnalysis> {
        // Try to parse as JSON first
        if let Ok(analysis) = serde_json::from_str::<FunctionParamAnalysis>(response) {
            return Ok(analysis);
        }

        // Fallback: parse text response
        Ok(self.parse_text_response(response, func))
    }

    /// Parse text response from LLM
    fn parse_text_response(&self, _response: &str, func: &FfiFunction) -> FunctionParamAnalysis {
        let mut param_roles = Vec::new();
        let relationships = Vec::new();
        let constraints = Vec::new();
        let recommendations = Vec::new();

        // Simple heuristic parsing
        for param in &func.params {
            let role = self.infer_role_from_name(&param.name, &param.ty);
            param_roles.push(ParameterRole {
                param_name: param.name.clone(),
                role,
                confidence: 0.7,
            });
        }

        FunctionParamAnalysis {
            function: func.name.clone(),
            param_roles,
            relationships,
            constraints,
            recommendations,
        }
    }

    /// Fallback analysis without LLM
    fn fallback_analysis(&self, ffi_info: &FfiInfo) -> LlmParameterAnalysis {
        let mut function_analysis = HashMap::new();

        for func in &ffi_info.functions {
            let mut param_roles = Vec::new();

            for param in &func.params {
                let role = self.infer_role_from_name(&param.name, &param.ty);
                param_roles.push(ParameterRole {
                    param_name: param.name.clone(),
                    role,
                    confidence: 0.5,
                });
            }

            function_analysis.insert(
                func.name.clone(),
                FunctionParamAnalysis {
                    function: func.name.clone(),
                    param_roles,
                    relationships: Vec::new(),
                    constraints: Vec::new(),
                    recommendations: Vec::new(),
                },
            );
        }

        LlmParameterAnalysis {
            function_analysis,
            common_patterns: Vec::new(),
        }
    }

    /// Infer semantic role from parameter name and type
    fn infer_role_from_name(&self, name: &str, ty: &str) -> SemanticRole {
        let name_lower = name.to_lowercase();

        if name_lower.contains("size") || name_lower.contains("len") || name_lower.contains("count")
        {
            return SemanticRole::Size;
        }
        if name_lower.contains("out") || name_lower.contains("result") || name_lower.contains("dst")
        {
            return SemanticRole::Output;
        }
        if name_lower.contains("in") || name_lower.contains("src") || name_lower.contains("input") {
            return SemanticRole::Input;
        }
        if name_lower.contains("config")
            || name_lower.contains("desc")
            || name_lower.contains("attr")
        {
            return SemanticRole::Configuration;
        }
        if name_lower.contains("flag")
            || name_lower.contains("mode")
            || name_lower.contains("option")
        {
            return SemanticRole::Flags;
        }
        if name_lower.contains("callback") || name_lower.contains("handler") {
            return SemanticRole::Callback;
        }
        if name_lower.contains("userdata")
            || name_lower.contains("context")
            || name_lower.contains("ctx")
        {
            return SemanticRole::UserData;
        }
        if name_lower.contains("error") || name_lower.contains("status") {
            return SemanticRole::Error;
        }
        if ty.contains("Handle") || ty.contains("*") && name_lower.contains("handle") {
            return SemanticRole::Handle;
        }

        SemanticRole::Input
    }

    /// Identify common parameter patterns across functions
    fn identify_common_patterns(
        &self,
        analyses: &HashMap<String, FunctionParamAnalysis>,
    ) -> Vec<ParameterPattern> {
        let mut patterns = Vec::new();

        // Size parameter pattern
        let size_funcs: Vec<String> = analyses
            .iter()
            .filter(|(_, analysis)| {
                analysis
                    .param_roles
                    .iter()
                    .any(|r| r.role == SemanticRole::Size)
            })
            .map(|(name, _)| name.clone())
            .take(3)
            .collect();

        if !size_funcs.is_empty() {
            patterns.push(ParameterPattern {
                name: "Size Parameter Pattern".to_string(),
                description: "Functions that take size/length parameters for arrays".to_string(),
                examples: size_funcs,
            });
        }

        // Output parameter pattern
        let output_funcs: Vec<String> = analyses
            .iter()
            .filter(|(_, analysis)| {
                analysis
                    .param_roles
                    .iter()
                    .any(|r| r.role == SemanticRole::Output)
            })
            .map(|(name, _)| name.clone())
            .take(3)
            .collect();

        if !output_funcs.is_empty() {
            patterns.push(ParameterPattern {
                name: "Output Parameter Pattern".to_string(),
                description: "Functions that write results to output parameters".to_string(),
                examples: output_funcs,
            });
        }

        patterns
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ffi::{FfiFunction, FfiParam};

    #[test]
    fn test_role_inference() {
        let analyzer = LlmParameterAnalyzer::new(None);

        assert_eq!(
            analyzer.infer_role_from_name("size", "int"),
            SemanticRole::Size
        );
        assert_eq!(
            analyzer.infer_role_from_name("output_buffer", "void*"),
            SemanticRole::Output
        );
        assert_eq!(
            analyzer.infer_role_from_name("callback", "void*"),
            SemanticRole::Callback
        );
    }

    #[test]
    fn test_fallback_analysis() {
        let mut ffi_info = FfiInfo::default();
        ffi_info.functions.push(FfiFunction {
            name: "test_func".to_string(),
            params: vec![FfiParam {
                name: "size".to_string(),
                ty: "int".to_string(),
                is_mut: false,
                is_pointer: false,
            }],
            return_type: "void".to_string(),
            docs: None,
        });

        let analyzer = LlmParameterAnalyzer::new(None);
        let analysis = analyzer.fallback_analysis(&ffi_info);

        assert_eq!(analysis.function_analysis.len(), 1);
        assert!(analysis.function_analysis.contains_key("test_func"));
    }

    #[test]
    fn test_pattern_identification() {
        let analyzer = LlmParameterAnalyzer::new(None);
        let mut analyses = HashMap::new();

        analyses.insert(
            "func1".to_string(),
            FunctionParamAnalysis {
                function: "func1".to_string(),
                param_roles: vec![ParameterRole {
                    param_name: "size".to_string(),
                    role: SemanticRole::Size,
                    confidence: 0.8,
                }],
                relationships: Vec::new(),
                constraints: Vec::new(),
                recommendations: Vec::new(),
            },
        );

        let patterns = analyzer.identify_common_patterns(&analyses);
        assert!(!patterns.is_empty());
    }

    #[test]
    fn test_prompt_building() {
        let analyzer = LlmParameterAnalyzer::new(None);
        let func = FfiFunction {
            name: "test".to_string(),
            params: vec![FfiParam {
                name: "size".to_string(),
                ty: "int".to_string(),
                is_mut: false,
                is_pointer: false,
            }],
            return_type: "void".to_string(),
            docs: Some("Test function".to_string()),
        };

        let prompt = analyzer.build_analysis_prompt(&func);
        assert!(prompt.contains("test"));
        assert!(prompt.contains("size"));
        assert!(prompt.contains("Test function"));
    }
}
