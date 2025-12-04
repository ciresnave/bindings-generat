//! Header discovery tool

use super::{Tool, ToolContext, ToolResult};
use anyhow::Result;

pub struct HeaderDiscoveryTool;

impl Tool for HeaderDiscoveryTool {
    fn name(&self) -> &'static str {
        "discover_headers"
    }

    fn description(&self) -> &'static str {
        "Discovers and analyzes header files in the source directory"
    }

    fn requirements(&self) -> Vec<&'static str> {
        vec!["source_path"]
    }

    fn provides(&self) -> Vec<&'static str> {
        vec!["headers"]
    }

    fn execute(&self, mut context: ToolContext) -> Result<ToolResult> {
        // Use existing discovery logic
        let discovery_result = crate::discovery::discover(&context.source_path)?;

        context.headers = discovery_result.headers;

        // Update lib_name if not already set and we found one
        if context.lib_name.is_empty() {
            context.lib_name = if !discovery_result.library_name.is_empty() {
                discovery_result.library_name
            } else {
                context
                    .source_path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("unknown")
                    .to_string()
            };
        }

        let message = format!("Discovered {} header files", context.headers.len());
        let suggestions = if context.headers.is_empty() {
            vec![
                "No headers found. Check if the source path contains C/C++ header files."
                    .to_string(),
            ]
        } else {
            vec![]
        };

        Ok(ToolResult {
            success: !context.headers.is_empty(),
            message,
            updated_context: context,
            suggestions,
        })
    }
}
