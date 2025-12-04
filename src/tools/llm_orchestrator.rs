//! LLM-guided execution for intelligent binding generation

use super::{ToolContext, ToolRegistry, ToolResult};
use anyhow::Result;
use serde_json::json;
use tracing::{info, warn};

pub struct LlmOrchestrator {
    registry: ToolRegistry,
    llm_client: crate::llm::OllamaClient,
    model: String,
}

impl LlmOrchestrator {
    pub fn new(model: String, cache_dir: Option<std::path::PathBuf>) -> Result<Self> {
        Ok(Self {
            registry: ToolRegistry::new(),
            llm_client: crate::llm::OllamaClient::new(cache_dir)?,
            model,
        })
    }

    pub fn execute(&self, mut context: ToolContext, max_iterations: usize) -> Result<ToolContext> {
        info!("Starting LLM-guided binding generation");

        for iteration in 0..max_iterations {
            info!("Iteration {} of {}", iteration + 1, max_iterations);

            // Get the current situation and available tools
            let situation = self.assess_situation(&context);
            let available_tools = self.registry.list_tools();

            // Ask LLM what to do next
            let llm_decision = self.get_llm_decision(&situation, &available_tools, &context)?;

            match llm_decision {
                LlmDecision::RunTool {
                    tool_name,
                    reasoning,
                } => {
                    info!("LLM decided to run tool '{}': {}", tool_name, reasoning);

                    let tool = self
                        .registry
                        .get_tool(&tool_name)
                        .ok_or_else(|| anyhow::anyhow!("Unknown tool: {}", tool_name))?;

                    match tool.execute(context) {
                        Ok(result) => {
                            context = result.updated_context;
                            if !result.success {
                                warn!("Tool '{}' failed: {}", tool_name, result.message);
                                // Continue to next iteration to let LLM handle the failure
                            } else {
                                info!("Tool '{}' succeeded: {}", tool_name, result.message);
                            }
                        }
                        Err(e) => {
                            warn!("Tool '{}' error: {}", tool_name, e);
                            // Continue to let LLM handle the error
                        }
                    }
                }
                LlmDecision::Complete { reason } => {
                    info!("LLM decided to complete: {}", reason);
                    break;
                }
                LlmDecision::Error { error } => {
                    return Err(anyhow::anyhow!("LLM decision error: {}", error));
                }
            }

            // Check if we've achieved success
            if self.is_successful(&context) {
                info!("Binding generation completed successfully");
                break;
            }
        }

        Ok(context)
    }

    fn assess_situation(&self, context: &ToolContext) -> String {
        let mut situation = Vec::new();

        if context.headers.is_empty() {
            situation.push("No headers discovered yet".to_string());
        } else {
            situation.push(format!("Found {} headers", context.headers.len()));
        }

        if context.ffi_info.is_none() {
            situation.push("FFI bindings not generated".to_string());
        } else {
            let ffi_info = context.ffi_info.as_ref().unwrap();
            situation.push(format!(
                "FFI generated: {} functions, {} types",
                ffi_info.functions.len(),
                ffi_info.types.len()
            ));
        }

        if context.patterns.is_none() {
            situation.push("Pattern analysis not done".to_string());
        } else {
            situation.push("Pattern analysis complete".to_string());
        }

        if context.wrapper_code.is_none() {
            situation.push("Safe wrappers not generated".to_string());
        } else {
            situation.push("Safe wrappers generated".to_string());
        }

        if !context.build_errors.is_empty() {
            situation.push(format!(
                "Build errors present: {}",
                context.build_errors.len()
            ));
        }

        situation.join(", ")
    }

    fn get_llm_decision(
        &self,
        situation: &str,
        available_tools: &[(&str, &str)],
        context: &ToolContext,
    ) -> Result<LlmDecision> {
        let tools_list = available_tools
            .iter()
            .map(|(name, desc)| format!("- {}: {}", name, desc))
            .collect::<Vec<_>>()
            .join("\n");

        let build_errors_info = if context.build_errors.is_empty() {
            "No build errors".to_string()
        } else {
            format!(
                "Build errors:\n{}",
                context
                    .build_errors
                    .iter()
                    .map(|e| format!("- {:?}", e))
                    .collect::<Vec<_>>()
                    .join("\n")
            )
        };

        let prompt = format!(
            r#"You are an expert at generating Rust FFI bindings. 

Current situation: {situation}

{build_errors_info}

Available tools:
{tools_list}

Your task is to decide what to do next to successfully generate working Rust bindings.

Respond in JSON format with one of these actions:

{{"action": "run_tool", "tool_name": "tool_name", "reasoning": "why you chose this tool"}}
{{"action": "complete", "reason": "why the binding generation is complete"}}
{{"action": "error", "error": "description of unrecoverable error"}}

Choose the most appropriate next step to progress toward working bindings."#,
            situation = situation,
            build_errors_info = build_errors_info,
            tools_list = tools_list
        );

        let response = self.llm_client.generate(&self.model, &prompt)?;

        // Parse LLM response
        match serde_json::from_str::<serde_json::Value>(&response) {
            Ok(json) => match json["action"].as_str() {
                Some("run_tool") => {
                    let tool_name = json["tool_name"]
                        .as_str()
                        .ok_or_else(|| anyhow::anyhow!("Missing tool_name"))?;
                    let reasoning = json["reasoning"]
                        .as_str()
                        .unwrap_or("No reasoning provided");

                    Ok(LlmDecision::RunTool {
                        tool_name: tool_name.to_string(),
                        reasoning: reasoning.to_string(),
                    })
                }
                Some("complete") => {
                    let reason = json["reason"].as_str().unwrap_or("No reason provided");
                    Ok(LlmDecision::Complete {
                        reason: reason.to_string(),
                    })
                }
                Some("error") => {
                    let error = json["error"].as_str().unwrap_or("Unknown error");
                    Ok(LlmDecision::Error {
                        error: error.to_string(),
                    })
                }
                _ => Err(anyhow::anyhow!("Invalid action in LLM response")),
            },
            Err(_) => {
                // Fallback: try to extract action from unstructured response
                if response.contains("complete")
                    || response.contains("done")
                    || response.contains("success")
                {
                    Ok(LlmDecision::Complete {
                        reason: "LLM indicated completion".to_string(),
                    })
                } else {
                    // Default to header discovery if nothing else makes sense
                    Ok(LlmDecision::RunTool {
                        tool_name: "discover_headers".to_string(),
                        reasoning: "Fallback: starting with header discovery".to_string(),
                    })
                }
            }
        }
    }

    fn is_successful(&self, context: &ToolContext) -> bool {
        // Consider it successful if we have generated code and no build errors
        context.wrapper_code.is_some() && context.build_errors.is_empty()
    }
}

#[derive(Debug)]
enum LlmDecision {
    RunTool {
        tool_name: String,
        reasoning: String,
    },
    Complete {
        reason: String,
    },
    Error {
        error: String,
    },
}
