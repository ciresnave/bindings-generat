//! Wrapper generation tool
use super::{Tool, ToolContext, ToolResult};
use anyhow::Result;

pub struct WrapperGenerationTool;

impl Tool for WrapperGenerationTool {
    fn name(&self) -> &'static str {
        "generate_wrappers"
    }
    fn description(&self) -> &'static str {
        "Generates safe Rust wrappers around FFI functions"
    }
    fn requirements(&self) -> Vec<&'static str> {
        vec!["ffi_info", "patterns"]
    }
    fn provides(&self) -> Vec<&'static str> {
        vec!["wrapper_code"]
    }

    fn execute(&self, mut context: ToolContext) -> Result<ToolResult> {
        let ffi_info = context.ffi_info.as_ref().unwrap();
        // Re-analyze the ffi_info to get AnalysisResult (patterns is just a summary string)
        let analysis = crate::analyzer::analyze_ffi(ffi_info)?;

        let wrapper_code = crate::generator::generate_safe_wrappers(
            ffi_info,
            &analysis,
            &crate::config::CodeStyle::default(),
        )?;

        context.wrapper_code = Some(wrapper_code);

        Ok(ToolResult {
            success: true,
            message: "Safe wrappers generated".to_string(),
            updated_context: context,
            suggestions: vec![],
        })
    }
}
