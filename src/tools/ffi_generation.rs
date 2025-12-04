//! FFI generation tool
use super::{Tool, ToolContext, ToolResult};
use anyhow::Result;

pub struct FfiGenerationTool;

impl Tool for FfiGenerationTool {
    fn name(&self) -> &'static str {
        "generate_ffi"
    }
    fn description(&self) -> &'static str {
        "Generates raw FFI bindings from C headers using bindgen"
    }
    fn requirements(&self) -> Vec<&'static str> {
        vec!["headers"]
    }
    fn provides(&self) -> Vec<&'static str> {
        vec!["ffi_code", "ffi_info"]
    }

    fn execute(&self, mut context: ToolContext) -> Result<ToolResult> {
        // Use existing FFI generation logic
        let (ffi_info, bindings_code) = crate::ffi::generate_and_parse_ffi(
            &context.headers,
            &context.lib_name,
            &context.source_path,
            &crate::Config::default(), // TODO: pass proper config
        )?;

        context.ffi_code = Some(bindings_code);
        context.ffi_info = Some(ffi_info);

        Ok(ToolResult {
            success: true,
            message: "FFI bindings generated successfully".to_string(),
            updated_context: context,
            suggestions: vec![],
        })
    }
}
