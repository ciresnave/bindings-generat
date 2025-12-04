//! Pattern detection and analysis of FFI bindings.
//!
//! This module analyzes parsed FFI information to detect common patterns and
//! idioms that can be transformed into safe, idiomatic Rust code.
//!
//! ## Detected Patterns
//!
//! ### RAII Patterns ([`raii`])
//! Detects resource management patterns:
//! - Constructor/destructor pairs (create/destroy, init/cleanup, etc.)
//! - Handle types that need automatic cleanup
//! - Resource lifecycle management
//!
//! ### Error Patterns ([`errors`])
//! Detects error handling strategies:
//! - Status code enums (success/error variants)
//! - Functions returning error codes
//! - Boolean success/failure patterns
//! - Null pointer error returns
//!
//! ### Ownership Patterns ([`ownership`])
//! Analyzes ownership and borrowing:
//! - Which types own resources
//! - Which functions transfer ownership
//! - Lifetime dependencies between types
//!
//! ### Ambiguity Detection ([`ambiguity`])
//! Identifies unclear patterns that need clarification:
//! - Multiple possible RAII pairings
//! - Unclear ownership semantics
//! - Ambiguous error handling
//!
//! ## Usage
//!
//! ```ignore
//! use analyzer::{detect_raii_patterns, detect_error_patterns};
//!
//! let raii = detect_raii_patterns(&ffi_info);
//! let errors = detect_error_patterns(&ffi_info);
//! ```

pub mod ambiguity;
pub mod anti_patterns;
pub mod api_sequences;
pub mod ast;
pub mod async_patterns;
pub mod attributes;
pub mod callback_analysis;
pub mod cross_references;
pub mod error_docs;
pub mod error_semantics;
pub mod errors;
pub mod example_patterns;
pub mod global_state;
pub mod lifetime;
pub mod llm_doc_orchestrator;
pub mod llm_parameters;
pub mod numeric_constraints;
pub mod ownership;
pub mod patterns;
pub mod performance;
pub mod platform;
pub mod platform_docs;
pub mod preconditions;
pub mod raii;
pub mod resource_limits;
pub mod semantic_analysis;
pub mod semantic_grouping;
pub mod smart_errors;
pub mod test_mining;
pub mod thread_safety;
pub mod trait_abstractions;
pub mod type_docs;
pub mod version_compat;
pub mod version_features;

pub use anti_patterns::{AntiPatternAnalyzer, Pitfall, PitfallInfo, Severity};
pub use api_sequences::{ApiSequence, ApiSequenceAnalyzer, StateTransition};
pub use async_patterns::{
    AsyncAnalyzer, AsyncOperation, AsyncPatterns, CallbackPattern, CompletionMethod, EventPattern,
    PollingPattern, generate_async_wrapper,
};
pub use attributes::{Attribute, AttributeAnalyzer, AttributeInfo, AttributeSource, AttributeType};
pub use callback_analysis::{
    CallbackAnalyzer, CallbackInfo, CallbackLifetime, CallbackSemantics, CallbackThreadSafety,
    ContextOwnership, InvocationCount,
};
pub use cross_references::{CrossReferenceAnalyzer, CrossReferences, FunctionRefs, TypeRefs};
pub use error_docs::{
    ErrorDocAnalyzer, ErrorDocumentation, ErrorEnumDoc, ErrorVariantDoc,
    generate_error_variant_docs,
};
pub use error_semantics::{ErrorCategory, ErrorInfo, ErrorSemantics, ErrorSemanticsAnalyzer};
pub use errors::{ErrorPatterns, detect_error_patterns};
pub use example_patterns::{
    BestPractice, CodeExample, ExampleAnalyzer, PatternSequence, UsagePatterns,
    generate_pattern_docs,
};
pub use global_state::{GlobalState, GlobalStateAnalyzer, GlobalStateInfo, GlobalStateType};
pub use lifetime::{
    BorrowReason, BorrowedFrom, LifetimeAnalyzer, LifetimeDependencies, generate_lifetime_docs,
    generate_lifetime_params,
};
pub use llm_doc_orchestrator::{
    EnhancedDocumentation, EnhancedErrorDoc, EnhancedFunctionDoc, EnhancedTypeDoc, FieldDoc,
    LlmCodeExample, LlmDocOrchestrator, ParameterDoc,
};
pub use llm_parameters::{
    ConstraintType as LlmConstraintType, InferredConstraint, LlmParameterAnalysis,
    LlmParameterAnalyzer, ParameterPattern, ParameterRelationship, ParameterRole, RelationshipType,
    SemanticRole,
};
pub use numeric_constraints::{
    ConstraintType as NumericConstraintType, NumericConstraint, NumericConstraints,
    NumericConstraintsAnalyzer,
};
pub use ownership::{
    LifecyclePair, LifetimeInfo, OwnershipAnalyzer, OwnershipInfo, OwnershipSemantics,
};
pub use performance::{
    BlockingBehavior, ComplexityClass, OperationType, PerformanceAnalyzer, PerformanceCost,
    PerformanceInfo, PerformanceTip, TimingInfo,
};
pub use platform::{
    Architecture, Platform, PlatformAnalyzer, PlatformInfo, PlatformNote, VersionConstraint,
    VersionRequirement,
};
pub use platform_docs::{
    BuildInstructions, BuildStep, DifferenceCategory, PlatformDifference, PlatformDocs,
    PlatformDocsAnalyzer, PlatformInfo as PlatformInfoDocs,
};
pub use preconditions::{ConstraintType, Precondition, PreconditionAnalyzer, PreconditionInfo};
pub use raii::{RaiiPatterns, detect_raii_patterns};
pub use resource_limits::{
    LimitInfo, LimitType, LimitUnit, ResourceLimits, ResourceLimitsAnalyzer,
};
pub use semantic_analysis::{
    FunctionCluster, ModuleInfo, SemanticAnalysis, SemanticAnalyzer, TypeRelationship,
};
pub use semantic_grouping::{
    FunctionGroup, GetterSetterPair, GroupType, SemanticGroupInfo, SemanticGroupingAnalyzer,
};
pub use smart_errors::{
    ErrorPattern, ErrorSeverity, RecoveryStrategy, SmartErrorAnalysis, SmartErrorAnalyzer,
    SmartErrorCategory, SmartErrorType, SmartErrorVariant,
};
pub use test_mining::{
    ExampleSource, ExampleType, ParameterStatistics, ParameterValue, TestCaseInfo, TestCaseMiner,
    UsageExample,
};
pub use thread_safety::{ThreadSafety, ThreadSafetyAnalyzer, ThreadSafetyInfo};
pub use trait_abstractions::{
    TraitAbstractions, TraitAnalyzer, TraitDefinition, TraitMethod, generate_trait_code,
};
pub use type_docs::{TypeDoc, TypeDocAnalyzer, TypeDocumentation, generate_type_docs};
pub use version_compat::{
    ApiVersion, ChangeType, CompatibilityMatrix, DeprecationInfo as DeprecationInfoCompat,
    FunctionChange, MigrationGuide, MigrationStep, VersionCompatibility,
    VersionCompatibilityAnalyzer,
};
pub use version_features::{
    DeprecationInfo, VersionAnalyzer, VersionFeatures, VersionRequirement as VersionReq,
    generate_cargo_features, generate_version_attributes,
};

use crate::ffi::FfiInfo;
use anyhow::Result;

// Import generator types needed for analysis
use crate::generator::builder_typestate::{BuilderTypestateAnalysis, BuilderTypestateGenerator};

/// Complete analysis result
#[derive(Debug, Clone)]
pub struct AnalysisResult {
    // Core pattern detection
    pub raii_patterns: RaiiPatterns,
    pub error_patterns: ErrorPatterns,

    // Advanced pattern analysis
    /// LLM-powered parameter analysis with semantic relationships
    pub parameter_analysis: Option<LlmParameterAnalysis>,
    /// Smart error type analysis with recovery suggestions
    pub smart_errors: Option<SmartErrorAnalysis>,
    /// LLM-enhanced documentation with examples and best practices
    pub enhanced_docs: Option<EnhancedDocumentation>,
    /// Builder typestate pattern analysis
    pub builder_typestates: Option<BuilderTypestateAnalysis>,

    // Per-function enriched context (merged from EnhancedContext)
    /// Detailed analysis for each function including thread safety, preconditions, etc.
    pub function_contexts:
        std::collections::HashMap<String, crate::enrichment::context::FunctionContext>,

    // Global context
    /// Changelog entries for version tracking
    pub changelog_entries: Vec<crate::discovery::ChangelogEntry>,
}

/// Perform complete pattern analysis on FFI
pub fn analyze_ffi(ffi_info: &FfiInfo) -> Result<AnalysisResult> {
    use tracing::info;

    // Original analyzers
    let raii_patterns = detect_raii_patterns(ffi_info);
    let error_patterns = detect_error_patterns(ffi_info);

    // New enhanced analyzers
    info!("Running enhanced analyzers...");

    // LLM parameter analysis (skip for now - requires async)
    // TODO: Make analyze_ffi async or run this in a separate phase
    let parameter_analysis = None;

    // Smart error analysis
    let smart_errors = {
        let analyzer = SmartErrorAnalyzer::new();
        let analysis = analyzer.analyze(ffi_info);
        Some(analysis)
    };

    // LLM documentation orchestration (skip for now - requires async)
    // TODO: Make analyze_ffi async or run this in a separate phase
    let enhanced_docs = None;

    // Builder typestate analysis
    let builder_typestates = {
        let generator = BuilderTypestateGenerator::new();
        let analysis = generator.analyze(ffi_info);
        Some(analysis)
    };

    if smart_errors.is_some() {
        info!("✓ Smart error analysis completed");
    }
    if builder_typestates.is_some() {
        info!("✓ Builder typestate analysis completed");
    }

    Ok(AnalysisResult {
        raii_patterns,
        error_patterns,
        parameter_analysis,
        smart_errors,
        enhanced_docs,
        builder_typestates,
        function_contexts: std::collections::HashMap::new(),
        changelog_entries: Vec::new(),
    })
}

/// Perform async LLM-powered analysis phase
///
/// This function runs the async analyzers (LlmParameterAnalyzer and LlmDocOrchestrator)
/// and updates the AnalysisResult with their findings.
pub async fn analyze_ffi_async(
    ffi_info: &FfiInfo,
    analysis: &mut AnalysisResult,
    use_llm: bool,
) -> Result<()> {
    use tracing::{info, warn};

    if !use_llm {
        info!("Skipping async LLM analysis (LLM disabled)");
        return Ok(());
    }

    info!("Running async LLM-powered analyzers...");

    // LLM parameter analysis
    let parameter_analysis = {
        let analyzer = LlmParameterAnalyzer::new(None); // Pass LLM client when available
        match analyzer.analyze(ffi_info).await {
            Ok(result) => {
                info!(
                    "✓ LLM parameter analysis completed: {} functions analyzed",
                    result.function_analysis.len()
                );
                Some(result)
            }
            Err(e) => {
                warn!("LLM parameter analysis failed: {}", e);
                None
            }
        }
    };

    // LLM documentation orchestration
    let enhanced_docs = {
        let orchestrator = LlmDocOrchestrator::new(None); // Pass LLM client when available
        match orchestrator.enhance_documentation(ffi_info).await {
            Ok(result) => {
                info!(
                    "✓ LLM documentation enhancement completed: {} function docs, {} type docs",
                    result.function_docs.len(),
                    result.type_docs.len()
                );
                Some(result)
            }
            Err(e) => {
                warn!("LLM documentation enhancement failed: {}", e);
                None
            }
        }
    };

    // Update analysis result
    analysis.parameter_analysis = parameter_analysis;
    analysis.enhanced_docs = enhanced_docs;

    Ok(())
}

/// Legacy function name for compatibility with tools
pub fn analyze_patterns(functions: &[crate::ffi::FfiFunction]) -> Result<AnalysisResult> {
    // Create a minimal FfiInfo with just functions for backward compatibility
    let ffi_info = FfiInfo {
        functions: functions.to_vec(),
        types: Vec::new(),
        enums: Vec::new(),
        constants: Vec::new(),
        opaque_types: Vec::new(),
        dependencies: Vec::new(),
        type_aliases: std::collections::HashMap::new(),
    };
    analyze_ffi(&ffi_info)
}
