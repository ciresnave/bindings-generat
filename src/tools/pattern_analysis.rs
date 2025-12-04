//! Pattern analysis tool
use super::{Tool, ToolContext, ToolResult};
use anyhow::Result;

pub struct PatternAnalysisTool;

impl Tool for PatternAnalysisTool {
    fn name(&self) -> &'static str {
        "analyze_patterns"
    }
    fn description(&self) -> &'static str {
        "Analyzes FFI functions to detect RAII patterns and error handling"
    }
    fn requirements(&self) -> Vec<&'static str> {
        vec!["ffi_info"]
    }
    fn provides(&self) -> Vec<&'static str> {
        vec!["patterns"]
    }

    fn execute(&self, mut context: ToolContext) -> Result<ToolResult> {
        let ffi_info = context.ffi_info.as_ref().unwrap();
        let analysis_result = crate::analyzer::analyze_patterns(&ffi_info.functions)?;
        // Convert analysis result to a string summary for now
        let patterns_summary = format!(
            "RAII patterns: {} handle types, {} lifecycle pairs; Error patterns: {:?}",
            analysis_result.raii_patterns.handle_types.len(),
            analysis_result.raii_patterns.lifecycle_pairs.len(),
            analysis_result.error_patterns.error_strategy
        );
        context.patterns = Some(patterns_summary);

        Ok(ToolResult {
            success: true,
            message: "Pattern analysis completed".to_string(),
            updated_context: context,
            suggestions: vec![],
        })
    }
}
