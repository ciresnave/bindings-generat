//! Enhanced context combining all enrichment sources
//!
//! This module provides the `EnhancedContext` type which combines information from:
//! - Header comments (inline documentation)
//! - External documentation files
//! - Code examples and tests
//! - Thread safety analysis
//! - Ownership semantics
//! - Preconditions and constraints
//!
//! The enriched context is used during code generation to produce high-quality
//! documentation and safer wrapper APIs.

use crate::analyzer::anti_patterns::{AntiPatternAnalyzer, PitfallInfo};
use crate::analyzer::api_sequences::{ApiSequence, ApiSequenceAnalyzer};
use crate::analyzer::attributes::{AttributeAnalyzer, AttributeInfo};
use crate::analyzer::callback_analysis::{CallbackAnalyzer, CallbackSemantics};
use crate::analyzer::error_semantics::{ErrorSemantics, ErrorSemanticsAnalyzer};
use crate::analyzer::global_state::{GlobalState, GlobalStateAnalyzer};
use crate::analyzer::numeric_constraints::{NumericConstraints, NumericConstraintsAnalyzer};
use crate::analyzer::ownership::{OwnershipAnalyzer, OwnershipInfo};
use crate::analyzer::performance::{PerformanceAnalyzer, PerformanceInfo};
use crate::analyzer::platform::{PlatformAnalyzer, PlatformInfo};
use crate::analyzer::preconditions::{PreconditionAnalyzer, PreconditionInfo};
use crate::analyzer::resource_limits::{ResourceLimits, ResourceLimitsAnalyzer};
use crate::analyzer::semantic_grouping::{SemanticGroupInfo, SemanticGroupingAnalyzer};
use crate::analyzer::test_mining::{TestCaseInfo, TestCaseMiner};
use crate::analyzer::thread_safety::{ThreadSafetyAnalyzer, ThreadSafetyInfo};
use crate::discovery::{ChangelogEntry, ChangelogParser, DeprecationInfo};
use crate::enrichment::header_parser::FunctionComment;
use std::collections::HashMap;

/// Enhanced documentation context for a function
#[derive(Debug, Clone)]
pub struct FunctionContext {
    /// Function name
    pub name: String,

    /// Primary description (best available)
    pub description: Option<String>,

    /// Parameter documentation
    pub parameters: HashMap<String, String>,

    /// Return value documentation
    pub return_doc: Option<String>,

    /// Thread safety information
    pub thread_safety: Option<ThreadSafetyInfo>,

    /// Memory ownership information
    pub ownership: Option<OwnershipInfo>,

    /// Preconditions and constraints
    pub preconditions: Option<PreconditionInfo>,

    /// Test case examples and usage patterns
    pub test_cases: Option<TestCaseInfo>,

    /// Compiler attributes and annotations
    pub attributes: Option<AttributeInfo>,

    /// Platform and version requirements
    pub platform: Option<PlatformInfo>,

    /// Performance characteristics
    pub performance: Option<PerformanceInfo>,

    /// Version history and deprecation info
    pub version_history: Vec<DeprecationInfo>,

    /// Common pitfalls and anti-patterns
    pub pitfalls: Option<PitfallInfo>,

    /// Error code semantics
    pub error_semantics: Option<ErrorSemantics>,

    /// Callback analysis (lifetimes, invocation counts)
    pub callback_info: Option<CallbackSemantics>,

    /// API sequencing requirements
    pub api_sequences: Option<ApiSequence>,

    /// Resource limits (connections, memory, etc.)
    pub resource_limits: Option<ResourceLimits>,

    /// Semantic grouping (modules, getter/setter pairs)
    pub semantic_group: Option<SemanticGroupInfo>,

    /// Global state usage
    pub global_state: Option<GlobalState>,

    /// Numeric constraints (alignment, ranges, etc.)
    pub numeric_constraints: Option<NumericConstraints>,

    /// Source priority (where the info came from)
    pub source: ContextSource,

    /// Additional notes and warnings
    pub notes: Vec<String>,
}

/// Source of enrichment information (in priority order)
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ContextSource {
    /// From header file inline comments (highest priority)
    HeaderComment,

    /// From external API documentation
    ExternalDocs,

    /// From usage examples
    UsageExamples,

    /// From test cases
    TestCases,

    /// Inferred from code analysis
    CodeAnalysis,

    /// No information available
    Unknown,
}

impl FunctionContext {
    /// Create new empty context
    pub fn new(name: String) -> Self {
        Self {
            name,
            description: None,
            parameters: HashMap::new(),
            return_doc: None,
            thread_safety: None,
            ownership: None,
            preconditions: None,
            test_cases: None,
            attributes: None,
            platform: None,
            performance: None,
            version_history: Vec::new(),
            pitfalls: None,
            error_semantics: None,
            callback_info: None,
            api_sequences: None,
            resource_limits: None,
            semantic_group: None,
            global_state: None,
            numeric_constraints: None,
            source: ContextSource::Unknown,
            notes: Vec::new(),
        }
    }

    /// Create from header comment (highest priority)
    pub fn from_header_comment(name: String, comment: &FunctionComment) -> Self {
        let mut ctx = Self::new(name);
        ctx.source = ContextSource::HeaderComment;

        // Use detailed if available, otherwise brief
        ctx.description = comment.detailed.clone().or_else(|| comment.brief.clone());

        // Copy parameter docs
        for (param_name, param_doc) in &comment.param_docs {
            ctx.parameters
                .insert(param_name.clone(), param_doc.description.clone());
        }

        // Copy return doc
        ctx.return_doc = comment.return_doc.clone();

        // Copy notes
        ctx.notes = comment.notes.clone();

        ctx
    }

    /// Merge with lower-priority context (only fill in missing fields)
    pub fn merge_with(&mut self, other: &Self) {
        // Only merge if other is lower or equal priority
        if other.source <= self.source {
            return;
        }

        // Fill in description if missing
        if self.description.is_none() && other.description.is_some() {
            self.description = other.description.clone();
            self.source = other.source;
        }

        // Add missing parameter docs
        for (name, doc) in &other.parameters {
            self.parameters
                .entry(name.clone())
                .or_insert_with(|| doc.clone());
        }

        // Fill in return doc if missing
        if self.return_doc.is_none() && other.return_doc.is_some() {
            self.return_doc = other.return_doc.clone();
        }

        // Add thread safety if missing
        if self.thread_safety.is_none() && other.thread_safety.is_some() {
            self.thread_safety = other.thread_safety.clone();
        }

        // Add ownership if missing
        if self.ownership.is_none() && other.ownership.is_some() {
            self.ownership = other.ownership.clone();
        }

        // Add preconditions if missing
        if self.preconditions.is_none() && other.preconditions.is_some() {
            self.preconditions = other.preconditions.clone();
        }

        // Add test cases if missing
        if self.test_cases.is_none() && other.test_cases.is_some() {
            self.test_cases = other.test_cases.clone();
        }

        // Add attributes if missing
        if self.attributes.is_none() && other.attributes.is_some() {
            self.attributes = other.attributes.clone();
        }

        // Add platform info if missing
        if self.platform.is_none() && other.platform.is_some() {
            self.platform = other.platform.clone();
        }

        // Add performance info if missing
        if self.performance.is_none() && other.performance.is_some() {
            self.performance = other.performance.clone();
        }

        // Add pitfalls if missing
        if self.pitfalls.is_none() && other.pitfalls.is_some() {
            self.pitfalls = other.pitfalls.clone();
        }

        // Merge version history (combine all deprecations)
        self.version_history
            .extend(other.version_history.iter().cloned());

        // Merge notes
        self.notes.extend(other.notes.iter().cloned());
    }

    /// Analyze thread safety from documentation
    pub fn analyze_thread_safety(&mut self, analyzer: &mut ThreadSafetyAnalyzer) {
        // Skip if already analyzed
        if self.thread_safety.is_some() {
            return;
        }

        // Combine all documentation for analysis
        let mut full_doc = String::new();
        if let Some(desc) = &self.description {
            full_doc.push_str(desc);
            full_doc.push('\n');
        }
        for note in &self.notes {
            full_doc.push_str(note);
            full_doc.push('\n');
        }

        // Analyze if we have any documentation
        if !full_doc.is_empty() {
            self.thread_safety = Some(analyzer.analyze(&self.name, &full_doc));
        }
    }

    /// Analyze ownership semantics from documentation
    pub fn analyze_ownership(&mut self, analyzer: &mut OwnershipAnalyzer) {
        // Skip if already analyzed
        if self.ownership.is_some() {
            return;
        }

        // Combine all documentation for analysis
        let mut full_doc = String::new();
        if let Some(desc) = &self.description {
            full_doc.push_str(desc);
            full_doc.push('\n');
        }
        for note in &self.notes {
            full_doc.push_str(note);
            full_doc.push('\n');
        }

        // Analyze if we have any documentation
        if !full_doc.is_empty() {
            self.ownership = Some(analyzer.analyze(&self.name, &full_doc));
        }
    }

    /// Analyze preconditions and constraints from documentation
    pub fn analyze_preconditions(&mut self, analyzer: &mut PreconditionAnalyzer) {
        // Skip if already analyzed
        if self.preconditions.is_some() {
            return;
        }

        // Combine all documentation for analysis
        let mut full_doc = String::new();
        if let Some(desc) = &self.description {
            full_doc.push_str(desc);
            full_doc.push('\n');
        }
        for note in &self.notes {
            full_doc.push_str(note);
            full_doc.push('\n');
        }

        // Analyze if we have any documentation
        if !full_doc.is_empty() {
            self.preconditions = Some(analyzer.analyze(&self.name, &full_doc, &self.parameters));
        }
    }

    /// Analyze compiler attributes from declaration
    pub fn analyze_attributes(&mut self, analyzer: &mut AttributeAnalyzer, declaration: &str) {
        // Skip if already analyzed
        if self.attributes.is_some() {
            return;
        }

        // Combine documentation for context
        let mut full_doc = String::new();
        if let Some(desc) = &self.description {
            full_doc.push_str(desc);
            full_doc.push('\n');
        }
        for note in &self.notes {
            full_doc.push_str(note);
            full_doc.push('\n');
        }

        // Analyze attributes from declaration and documentation
        self.attributes = Some(analyzer.analyze(&self.name, declaration, Some(&full_doc)));
    }

    /// Analyze platform requirements from declaration
    pub fn analyze_platform(&mut self, analyzer: &mut PlatformAnalyzer, declaration: &str) {
        // Skip if already analyzed
        if self.platform.is_some() {
            return;
        }

        // Combine documentation for context
        let mut full_doc = String::new();
        if let Some(desc) = &self.description {
            full_doc.push_str(desc);
            full_doc.push('\n');
        }
        for note in &self.notes {
            full_doc.push_str(note);
            full_doc.push('\n');
        }

        // Analyze platform requirements
        self.platform = Some(analyzer.analyze(&self.name, declaration, Some(&full_doc)));
    }

    /// Analyze performance characteristics
    pub fn analyze_performance(&mut self, analyzer: &mut PerformanceAnalyzer) {
        // Skip if already analyzed
        if self.performance.is_some() {
            return;
        }

        // Combine all documentation for analysis
        let mut full_doc = String::new();
        if let Some(desc) = &self.description {
            full_doc.push_str(desc);
            full_doc.push('\n');
        }
        for note in &self.notes {
            full_doc.push_str(note);
            full_doc.push('\n');
        }

        // Analyze performance
        self.performance = Some(analyzer.analyze(&self.name, Some(&full_doc)));
    }

    /// Generate comprehensive Rust documentation
    pub fn generate_docs(&self) -> String {
        let mut docs = String::new();

        // Description
        if let Some(desc) = &self.description {
            docs.push_str(&format!("/// {}\n", desc));
            docs.push_str("///\n");
        }

        // Thread safety (critical information)
        if let Some(safety) = &self.thread_safety {
            docs.push_str(&safety.generate_docs());
            docs.push_str("///\n");
        }

        // Ownership semantics (critical information)
        if let Some(ownership) = &self.ownership {
            docs.push_str(&ownership.generate_docs());
            docs.push_str("///\n");
        }

        // Platform and version requirements (critical information)
        if let Some(platform) = &self.platform
            && platform.has_restrictions() {
                docs.push_str(&platform.generate_documentation());
                docs.push_str("///\n");
            }

        // Performance characteristics (important information)
        if let Some(performance) = &self.performance
            && performance.has_info() {
                docs.push_str(&performance.generate_documentation());
                docs.push_str("///\n");
            }

        // Preconditions and constraints (critical information)
        if let Some(preconditions) = &self.preconditions {
            if preconditions.has_preconditions() {
                docs.push_str("/// # Preconditions\n");
                docs.push_str("///\n");
                for precond in &preconditions.preconditions {
                    docs.push_str(&format!("/// - {}\n", precond.description));
                }
                docs.push_str("///\n");
            }

            if !preconditions.undefined_behavior.is_empty() {
                docs.push_str("/// # Undefined Behavior\n");
                docs.push_str("///\n");
                for ub in &preconditions.undefined_behavior {
                    docs.push_str(&format!("/// - {}\n", ub));
                }
                docs.push_str("///\n");
            }

            if !preconditions.performance_notes.is_empty() {
                docs.push_str("/// # Performance Notes\n");
                docs.push_str("///\n");
                for note in &preconditions.performance_notes {
                    docs.push_str(&format!("/// - {}\n", note));
                }
                docs.push_str("///\n");
            }
        }

        // Compiler attributes and annotations
        if let Some(attrs) = &self.attributes
            && attrs.has_attributes() {
                docs.push_str(&attrs.generate_documentation());
                docs.push_str("///\n");
            }

        // Version history and deprecations (critical information)
        if !self.version_history.is_empty() {
            docs.push_str("/// # Version History\n");
            docs.push_str("///\n");
            for dep in &self.version_history {
                if let Some(removal_version) = &dep.removal_version {
                    docs.push_str(&format!(
                        "/// ⚠️ **DEPRECATED since {}, will be removed in {}**\n",
                        dep.since, removal_version
                    ));
                } else {
                    docs.push_str(&format!("/// ⚠️ **DEPRECATED since {}**\n", dep.since));
                }

                if let Some(replacement) = &dep.replacement {
                    docs.push_str(&format!("///\n/// Use `{}` instead.\n", replacement));
                }

                if let Some(reason) = &dep.reason {
                    docs.push_str(&format!("///\n/// Reason: {}\n", reason));
                }

                if let Some(migration) = &dep.migration {
                    docs.push_str("///\n/// # Migration\n");
                    docs.push_str(&"/// ```rust\n".to_string());
                    for line in migration.lines() {
                        docs.push_str(&format!("/// {}\n", line));
                    }
                    docs.push_str(&"/// ```\n".to_string());
                }

                docs.push_str("///\n");
            }
        }

        // Common pitfalls and anti-patterns (critical safety information)
        if let Some(pitfalls) = &self.pitfalls
            && pitfalls.has_pitfalls() {
                docs.push_str(&pitfalls.generate_documentation());
                docs.push_str("///\n");
            }

        // Test case examples and usage patterns
        if let Some(test_cases) = &self.test_cases
            && test_cases.has_examples() {
                docs.push_str(&test_cases.generate_example_docs());
                docs.push_str("///\n");
            }

        // Parameters
        if !self.parameters.is_empty() {
            docs.push_str("/// # Parameters\n");
            docs.push_str("///\n");
            for (name, doc) in &self.parameters {
                docs.push_str(&format!("/// * `{}` - {}\n", name, doc));
            }
            docs.push_str("///\n");
        }

        // Return value
        if let Some(ret) = &self.return_doc {
            docs.push_str("/// # Returns\n");
            docs.push_str("///\n");
            docs.push_str(&format!("/// {}\n", ret));
            docs.push_str("///\n");
        }

        // Notes and warnings
        if !self.notes.is_empty() {
            docs.push_str("/// # Notes\n");
            docs.push_str("///\n");
            for note in &self.notes {
                docs.push_str(&format!("/// {}\n", note));
            }
        }

        docs
    }

    /// Analyze error code semantics from documentation
    pub fn analyze_error_semantics(&mut self, analyzer: &mut ErrorSemanticsAnalyzer) {
        if self.error_semantics.is_some() {
            return;
        }

        let mut full_doc = String::new();
        if let Some(desc) = &self.description {
            full_doc.push_str(desc);
            full_doc.push('\n');
        }
        if let Some(ret) = &self.return_doc {
            full_doc.push_str(ret);
            full_doc.push('\n');
        }

        if !full_doc.is_empty() {
            let return_type = self.return_doc.as_deref().unwrap_or("");
            self.error_semantics = Some(analyzer.analyze(&self.name, &full_doc, return_type));
        }
    }

    /// Analyze callback semantics from documentation
    pub fn analyze_callbacks(&mut self, analyzer: &mut CallbackAnalyzer) {
        if self.callback_info.is_some() {
            return;
        }

        let mut full_doc = String::new();
        if let Some(desc) = &self.description {
            full_doc.push_str(desc);
            full_doc.push('\n');
        }
        for (param, doc) in &self.parameters {
            full_doc.push_str(&format!("{}: {}\n", param, doc));
        }

        if !full_doc.is_empty() {
            // Convert parameters to slice of tuples
            let params: Vec<(String, String)> = self
                .parameters
                .iter()
                .map(|(name, doc)| (name.clone(), doc.clone()))
                .collect();
            self.callback_info = Some(analyzer.analyze(&self.name, &full_doc, &params));
        }
    }

    /// Analyze API sequencing requirements from documentation
    pub fn analyze_api_sequences(&mut self, analyzer: &mut ApiSequenceAnalyzer) {
        if self.api_sequences.is_some() {
            return;
        }

        let mut full_doc = String::new();
        if let Some(desc) = &self.description {
            full_doc.push_str(desc);
            full_doc.push('\n');
        }
        for note in &self.notes {
            full_doc.push_str(note);
            full_doc.push('\n');
        }

        if !full_doc.is_empty() {
            self.api_sequences = Some(analyzer.analyze(&self.name, &full_doc));
        }
    }

    /// Analyze resource limits from documentation
    pub fn analyze_resource_limits(&mut self, analyzer: &mut ResourceLimitsAnalyzer) {
        if self.resource_limits.is_some() {
            return;
        }

        let mut full_doc = String::new();
        if let Some(desc) = &self.description {
            full_doc.push_str(desc);
            full_doc.push('\n');
        }
        for note in &self.notes {
            full_doc.push_str(note);
            full_doc.push('\n');
        }

        if !full_doc.is_empty() {
            self.resource_limits = Some(analyzer.analyze(&self.name, &full_doc));
        }
    }

    /// Analyze semantic grouping from documentation
    pub fn analyze_semantic_grouping(&mut self, analyzer: &mut SemanticGroupingAnalyzer) {
        if self.semantic_group.is_some() {
            return;
        }

        let mut full_doc = String::new();
        if let Some(desc) = &self.description {
            full_doc.push_str(desc);
            full_doc.push('\n');
        }

        // Register with analyzer for later grouping
        analyzer.register_function(&self.name, &full_doc);

        if !full_doc.is_empty() {
            self.semantic_group = Some(analyzer.analyze(&self.name, &full_doc));
        }
    }

    /// Analyze global state usage from documentation
    pub fn analyze_global_state(&mut self, analyzer: &mut GlobalStateAnalyzer) {
        if self.global_state.is_some() {
            return;
        }

        let mut full_doc = String::new();
        if let Some(desc) = &self.description {
            full_doc.push_str(desc);
            full_doc.push('\n');
        }
        for note in &self.notes {
            full_doc.push_str(note);
            full_doc.push('\n');
        }

        if !full_doc.is_empty() {
            self.global_state = Some(analyzer.analyze(&self.name, &full_doc));
        }
    }

    /// Analyze numeric constraints from documentation
    pub fn analyze_numeric_constraints(&mut self, analyzer: &mut NumericConstraintsAnalyzer) {
        if self.numeric_constraints.is_some() {
            return;
        }

        let mut full_doc = String::new();
        if let Some(desc) = &self.description {
            full_doc.push_str(desc);
            full_doc.push('\n');
        }
        for (param, doc) in &self.parameters {
            full_doc.push_str(&format!("{}: {}\n", param, doc));
        }

        if !full_doc.is_empty() {
            self.numeric_constraints = Some(analyzer.analyze(&self.name, &full_doc));
        }
    }
}

/// Complete enhanced context for all functions
#[derive(Debug)]
pub struct EnhancedContext {
    /// Per-function context
    pub functions: HashMap<String, FunctionContext>,

    /// Thread safety analyzer
    pub thread_safety_analyzer: ThreadSafetyAnalyzer,

    /// Ownership analyzer
    pub ownership_analyzer: OwnershipAnalyzer,

    /// Precondition analyzer
    pub precondition_analyzer: PreconditionAnalyzer,

    /// Test case miner
    pub test_case_miner: TestCaseMiner,

    /// Attribute analyzer
    pub attribute_analyzer: AttributeAnalyzer,

    /// Platform analyzer
    pub platform_analyzer: PlatformAnalyzer,

    /// Performance analyzer
    pub performance_analyzer: PerformanceAnalyzer,

    /// Anti-pattern analyzer
    pub anti_pattern_analyzer: AntiPatternAnalyzer,

    /// Error semantics analyzer
    pub error_semantics_analyzer: ErrorSemanticsAnalyzer,

    /// Callback analysis analyzer
    pub callback_analyzer: CallbackAnalyzer,

    /// API sequence analyzer
    pub api_sequence_analyzer: ApiSequenceAnalyzer,

    /// Resource limits analyzer
    pub resource_limits_analyzer: ResourceLimitsAnalyzer,

    /// Semantic grouping analyzer
    pub semantic_grouping_analyzer: SemanticGroupingAnalyzer,

    /// Global state analyzer
    pub global_state_analyzer: GlobalStateAnalyzer,

    /// Numeric constraints analyzer
    pub numeric_constraints_analyzer: NumericConstraintsAnalyzer,

    /// Changelog parser
    pub changelog_parser: ChangelogParser,

    /// Parsed changelog entries
    pub changelog_entries: Vec<ChangelogEntry>,
}

impl EnhancedContext {
    /// Create new empty context
    pub fn new() -> Self {
        Self {
            functions: HashMap::new(),
            thread_safety_analyzer: ThreadSafetyAnalyzer::new(),
            ownership_analyzer: OwnershipAnalyzer::new(),
            precondition_analyzer: PreconditionAnalyzer::new(),
            test_case_miner: TestCaseMiner::new(),
            attribute_analyzer: AttributeAnalyzer::new(),
            platform_analyzer: PlatformAnalyzer::new(),
            performance_analyzer: PerformanceAnalyzer::new(),
            anti_pattern_analyzer: AntiPatternAnalyzer::new(),
            error_semantics_analyzer: ErrorSemanticsAnalyzer::new(),
            callback_analyzer: CallbackAnalyzer::new(),
            api_sequence_analyzer: ApiSequenceAnalyzer::new(),
            resource_limits_analyzer: ResourceLimitsAnalyzer::new(),
            semantic_grouping_analyzer: SemanticGroupingAnalyzer::new(),
            global_state_analyzer: GlobalStateAnalyzer::new(),
            numeric_constraints_analyzer: NumericConstraintsAnalyzer::new(),
            changelog_parser: ChangelogParser::new(),
            changelog_entries: Vec::new(),
        }
    }

    /// Add or update function context
    pub fn add_function(&mut self, ctx: FunctionContext) {
        let name = ctx.name.clone();

        // Merge with existing if present
        if let Some(existing) = self.functions.get_mut(&name) {
            existing.merge_with(&ctx);
        } else {
            self.functions.insert(name, ctx);
        }
    }

    /// Get function context (creates if missing)
    pub fn get_or_create(&mut self, name: &str) -> &mut FunctionContext {
        self.functions
            .entry(name.to_string())
            .or_insert_with(|| FunctionContext::new(name.to_string()))
    }

    /// Analyze thread safety for all functions
    pub fn analyze_all_thread_safety(&mut self) {
        let names: Vec<String> = self.functions.keys().cloned().collect();

        for name in names {
            if let Some(ctx) = self.functions.get_mut(&name) {
                ctx.analyze_thread_safety(&mut self.thread_safety_analyzer);
            }
        }
    }

    /// Parse changelog file and extract version history
    ///
    /// # Arguments
    /// * `path` - Path to the changelog file (e.g., "CHANGELOG.md")
    /// * `content` - Content of the changelog file
    pub fn parse_changelog(&mut self, path: &str, content: &str) {
        self.changelog_entries = self.changelog_parser.parse(path, content);
    }

    /// Check if a function is deprecated and update context
    ///
    /// Call this after parsing the changelog to enrich function contexts
    /// with deprecation information.
    pub fn apply_deprecations(&mut self) {
        // Build a map of deprecated functions
        let mut deprecations: HashMap<String, Vec<DeprecationInfo>> = HashMap::new();

        for entry in &self.changelog_entries {
            for dep in &entry.deprecations {
                // Extract function name from item (remove backticks, parentheses)
                let func_name = dep
                    .item
                    .trim_matches('`')
                    .trim_end_matches("()")
                    .to_string();

                deprecations
                    .entry(func_name)
                    .or_default()
                    .push(dep.clone());
            }
        }

        // Apply deprecations to function contexts
        for (func_name, deps) in deprecations {
            if let Some(ctx) = self.functions.get_mut(&func_name) {
                ctx.version_history.extend(deps);
            }
        }
    }

    /// Analyze function for common pitfalls
    ///
    /// Call this after gathering documentation to detect anti-patterns and common mistakes.
    ///
    /// # Arguments
    /// * `function_name` - Name of the function to analyze
    /// * `issue_content` - Optional issue/forum content to analyze
    pub fn analyze_pitfalls(&mut self, function_name: &str, issue_content: Option<&str>) {
        // Get or create function context
        let ctx = self.get_or_create(function_name);

        // Combine all documentation for analysis
        let mut full_doc = String::new();
        if let Some(desc) = &ctx.description {
            full_doc.push_str(desc);
            full_doc.push('\n');
        }
        for note in &ctx.notes {
            full_doc.push_str(note);
            full_doc.push('\n');
        }

        // Analyze if we have any documentation
        if !full_doc.is_empty() || issue_content.is_some() {
            let pitfalls =
                self.anti_pattern_analyzer
                    .analyze(function_name, &full_doc, issue_content);

            // Only store if pitfalls were found
            if pitfalls.has_pitfalls() {
                self.functions.get_mut(function_name).unwrap().pitfalls = Some(pitfalls);
            }
        }
    }
}

impl Default for EnhancedContext {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::enrichment::header_parser::{FunctionComment, ParamDoc};

    #[test]
    fn test_context_from_header() {
        let mut comment = FunctionComment::default();
        comment.brief = Some("Creates a new handle".to_string());
        comment.param_docs.insert(
            "flags".to_string(),
            ParamDoc {
                name: "flags".to_string(),
                description: "Configuration flags".to_string(),
                direction: None,
            },
        );
        comment.return_doc = Some("Handle on success, null on error".to_string());

        let ctx = FunctionContext::from_header_comment("create_handle".to_string(), &comment);

        assert_eq!(ctx.name, "create_handle");
        assert_eq!(ctx.description, Some("Creates a new handle".to_string()));
        assert_eq!(
            ctx.parameters.get("flags"),
            Some(&"Configuration flags".to_string())
        );
        assert_eq!(
            ctx.return_doc,
            Some("Handle on success, null on error".to_string())
        );
        assert_eq!(ctx.source, ContextSource::HeaderComment);
    }

    #[test]
    fn test_context_merge() {
        let mut ctx1 = FunctionContext::new("foo".to_string());
        ctx1.description = Some("From header".to_string());
        ctx1.source = ContextSource::HeaderComment;

        let mut ctx2 = FunctionContext::new("foo".to_string());
        ctx2.description = Some("From docs".to_string());
        ctx2.return_doc = Some("Returns int".to_string());
        ctx2.source = ContextSource::ExternalDocs;

        ctx1.merge_with(&ctx2);

        // Should keep header description (higher priority)
        assert_eq!(ctx1.description, Some("From header".to_string()));
        // Should add return doc (missing)
        assert_eq!(ctx1.return_doc, Some("Returns int".to_string()));
    }

    #[test]
    fn test_thread_safety_analysis() {
        let mut comment = FunctionComment::default();
        comment.brief = Some("Thread-safe function".to_string());
        comment
            .notes
            .push("This function is thread-safe and may be called concurrently.".to_string());

        let mut ctx = FunctionContext::from_header_comment("safe_func".to_string(), &comment);
        let mut analyzer = ThreadSafetyAnalyzer::new();

        ctx.analyze_thread_safety(&mut analyzer);

        assert!(ctx.thread_safety.is_some());
        let safety = ctx.thread_safety.as_ref().unwrap();
        assert_eq!(
            safety.safety,
            crate::analyzer::thread_safety::ThreadSafety::Safe
        );
    }

    #[test]
    fn test_ownership_analysis() {
        let mut comment = FunctionComment::default();
        comment.brief = Some("Creates a new handle".to_string());
        comment
            .notes
            .push("Caller must free with cudnnDestroy()".to_string());

        let mut ctx = FunctionContext::from_header_comment("cudnnCreate".to_string(), &comment);
        let mut analyzer = OwnershipAnalyzer::new();

        ctx.analyze_ownership(&mut analyzer);

        assert!(ctx.ownership.is_some());
        let ownership = ctx.ownership.as_ref().unwrap();
        assert_eq!(
            ownership.return_ownership,
            crate::analyzer::ownership::OwnershipSemantics::CallerOwns
        );
        assert!(ownership.lifecycle_pair.is_some());
    }

    #[test]
    fn test_precondition_analysis() {
        let mut comment = FunctionComment::default();
        comment.brief = Some("Allocates memory".to_string());
        comment.param_docs.insert(
            "size".to_string(),
            crate::enrichment::header_parser::ParamDoc {
                name: "size".to_string(),
                description: "Must be > 0".to_string(),
                direction: None,
            },
        );
        comment
            .notes
            .push("Passing null causes undefined behavior".to_string());

        let mut ctx = FunctionContext::from_header_comment("allocate".to_string(), &comment);
        let mut analyzer = PreconditionAnalyzer::new();

        ctx.analyze_preconditions(&mut analyzer);

        assert!(ctx.preconditions.is_some());
        let preconditions = ctx.preconditions.as_ref().unwrap();
        assert!(preconditions.has_preconditions());
        assert!(!preconditions.undefined_behavior.is_empty());

        // Should have detected the > 0 constraint
        let size_precond = preconditions.preconditions_for_param("size");
        assert!(!size_precond.is_empty());
        assert!(size_precond[0].can_validate);
    }

    #[test]
    fn test_enhanced_context() {
        let mut enhanced = EnhancedContext::new();

        let ctx1 = FunctionContext::new("func1".to_string());
        let ctx2 = FunctionContext::new("func2".to_string());

        enhanced.add_function(ctx1);
        enhanced.add_function(ctx2);

        assert_eq!(enhanced.functions.len(), 2);
        assert!(enhanced.functions.contains_key("func1"));
        assert!(enhanced.functions.contains_key("func2"));
    }
}
