//! Tool-based architecture for bindings generation
//!
//! This module provides a collection of discrete tools that can be orchestrated
//! either in a fixed pipeline (without LLM) or intelligently by an LLM to
//! adaptively solve binding generation challenges.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use tracing::info;

pub mod build_fixing;
pub mod build_validation;
pub mod dependency_detection;
pub mod discovery;
pub mod documentation;
// pub mod enhanced_dependency_detection;
pub mod ffi_generation;
// pub mod library_search;
// pub mod llm_orchestrator;
pub mod pattern_analysis;
// pub mod pattern_storage;
pub mod wrapper_generation;

/// Context shared between tools during execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolContext {
    /// Source path being processed
    pub source_path: PathBuf,
    /// Output directory
    pub output_path: PathBuf,
    /// Library name
    pub lib_name: String,
    /// Discovered headers
    pub headers: Vec<PathBuf>,
    /// Generated FFI bindings code
    pub ffi_code: Option<String>,
    /// Parsed FFI information
    pub ffi_info: Option<crate::ffi::FfiInfo>,
    /// Detected patterns (placeholder)
    pub patterns: Option<String>,
    /// Generated wrapper code
    pub wrapper_code: Option<String>,
    /// Detected dependencies
    pub dependencies: Vec<String>,
    /// Include directories
    pub include_dirs: Vec<PathBuf>,
    /// Library paths
    pub lib_paths: Vec<PathBuf>,
    /// Link libraries
    pub link_libs: Vec<String>,
    /// Build errors (if any)
    pub build_errors: Vec<crate::output::error_parser::BuildError>,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

impl ToolContext {
    pub fn new(source_path: PathBuf, output_path: PathBuf, lib_name: String) -> Self {
        Self {
            source_path,
            output_path,
            lib_name,
            headers: Vec::new(),
            ffi_code: None,
            ffi_info: None,
            patterns: None,
            wrapper_code: None,
            dependencies: Vec::new(),
            include_dirs: Vec::new(),
            lib_paths: Vec::new(),
            link_libs: Vec::new(),
            build_errors: Vec::new(),
            metadata: HashMap::new(),
        }
    }
}

/// Result of a tool execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult {
    pub success: bool,
    pub message: String,
    pub updated_context: ToolContext,
    pub suggestions: Vec<String>,
}

/// Trait for all binding generation tools
pub trait Tool {
    /// Name of the tool for LLM interaction
    fn name(&self) -> &'static str;

    /// Description of what this tool does
    fn description(&self) -> &'static str;

    /// Execute the tool with the given context
    fn execute(&self, context: ToolContext) -> Result<ToolResult>;

    /// Get the tool's input requirements
    fn requirements(&self) -> Vec<&'static str>;

    /// Get what this tool provides/modifies
    fn provides(&self) -> Vec<&'static str>;
}

/// Tool registry for managing available tools
pub struct ToolRegistry {
    tools: HashMap<String, Box<dyn Tool>>,
}

impl Default for ToolRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl ToolRegistry {
    pub fn new() -> Self {
        let mut registry = Self {
            tools: HashMap::new(),
        };

        // Register all available tools
        registry.register(Box::new(discovery::HeaderDiscoveryTool));
        registry.register(Box::new(ffi_generation::FfiGenerationTool));
        registry.register(Box::new(pattern_analysis::PatternAnalysisTool));
        registry.register(Box::new(wrapper_generation::WrapperGenerationTool));
        registry.register(Box::new(dependency_detection::DependencyDetectionTool));
        // registry.register(Box::new(
        //     enhanced_dependency_detection::EnhancedDependencyDetectionTool::new(None).unwrap(),
        // ));
        registry.register(Box::new(build_validation::BuildValidationTool));
        registry.register(Box::new(build_fixing::BuildFixingTool));
        registry.register(Box::new(documentation::DocumentationTool));

        registry
    }

    pub fn register(&mut self, tool: Box<dyn Tool>) {
        self.tools.insert(tool.name().to_string(), tool);
    }

    pub fn get_tool(&self, name: &str) -> Option<&dyn Tool> {
        self.tools.get(name).map(|t| t.as_ref())
    }

    pub fn list_tools(&self) -> Vec<(&str, &str)> {
        self.tools
            .values()
            .map(|tool| (tool.name(), tool.description()))
            .collect()
    }

    pub fn get_tool_info(&self, name: &str) -> Option<ToolInfo> {
        self.tools.get(name).map(|tool| ToolInfo {
            name: tool.name().to_string(),
            description: tool.description().to_string(),
            requirements: tool.requirements().iter().map(|s| s.to_string()).collect(),
            provides: tool.provides().iter().map(|s| s.to_string()).collect(),
        })
    }
}

/// Information about a tool for LLM interaction
#[derive(Debug, Serialize, Deserialize)]
pub struct ToolInfo {
    pub name: String,
    pub description: String,
    pub requirements: Vec<String>,
    pub provides: Vec<String>,
}

/// Execution mode for the binding generation
#[derive(Debug, Clone)]
pub enum ExecutionMode {
    /// Fixed pipeline without LLM
    Sequential,
    /// LLM-guided intelligent execution
    LlmGuided {
        model: String,
        max_iterations: usize,
    },
}

/// Main orchestrator that can run in different modes
pub struct BindingOrchestrator {
    registry: ToolRegistry,
    mode: ExecutionMode,
}

impl BindingOrchestrator {
    pub fn new(mode: ExecutionMode) -> Self {
        Self {
            registry: ToolRegistry::new(),
            mode,
        }
    }

    /// Execute binding generation with the configured mode
    pub fn execute(&self, context: ToolContext) -> Result<ToolContext> {
        match &self.mode {
            ExecutionMode::Sequential => self.execute_sequential(context),
            ExecutionMode::LlmGuided {
                model,
                max_iterations,
            } => self.execute_llm_guided(context, model, *max_iterations),
        }
    }

    /// Execute the traditional sequential pipeline
    fn execute_sequential(&self, mut context: ToolContext) -> Result<ToolContext> {
        let pipeline = [
            "discover_headers",
            "generate_ffi",
            "analyze_patterns",
            "generate_wrappers",
            "detect_dependencies",
            "validate_build",
        ];

        for tool_name in &pipeline {
            let tool = self
                .registry
                .get_tool(tool_name)
                .ok_or_else(|| anyhow::anyhow!("Tool not found: {}", tool_name))?;

            let result = tool.execute(context)?;
            context = result.updated_context;

            if !result.success {
                // In sequential mode, we stop on first failure
                return Err(anyhow::anyhow!(
                    "Tool {} failed: {}",
                    tool_name,
                    result.message
                ));
            }
        }

        Ok(context)
    }

    /// Execute with LLM guidance for intelligent problem-solving
    fn execute_llm_guided(
        &self,
        context: ToolContext,
        _model: &str,
        _max_iterations: usize,
    ) -> Result<ToolContext> {
        // TODO: Implement LLM-guided execution
        info!("LLM-guided execution not yet implemented, falling back to sequential");
        self.execute_sequential(context)
    }
}
