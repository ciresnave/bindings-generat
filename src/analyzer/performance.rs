//! Performance characteristic analysis.
//!
//! This module detects performance-related properties by analyzing:
//! - Documentation keywords (blocking, async, expensive, etc.)
//! - Function naming patterns (*Async, *Sync, Wait*)
//! - Complexity annotations (O(n), O(n²), etc.)
//! - Performance warnings and tips
//!
//! The analysis generates appropriate performance documentation and warnings.

use std::collections::HashMap;

/// Blocking behavior of an operation
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BlockingBehavior {
    /// Operation blocks the calling thread
    Blocking,
    /// Operation returns immediately (asynchronous)
    NonBlocking,
    /// Operation may block depending on conditions
    MayBlock,
    /// Unknown blocking behavior
    Unknown,
}

/// Type of operation
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OperationType {
    /// CPU computation
    Compute,
    /// Memory operation
    Memory,
    /// I/O operation (disk, network)
    IO,
    /// GPU operation
    GPU,
    /// Synchronization primitive
    Synchronization,
    /// Unknown or mixed
    Unknown,
}

/// Computational complexity class
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ComplexityClass {
    /// O(1) - constant time
    Constant,
    /// O(log n) - logarithmic
    Logarithmic,
    /// O(n) - linear
    Linear,
    /// O(n log n)
    Linearithmic,
    /// O(n²) - quadratic
    Quadratic,
    /// O(n³) - cubic
    Cubic,
    /// O(2ⁿ) - exponential
    Exponential,
    /// O(n!) - factorial
    Factorial,
    /// Unknown complexity
    Unknown,
}

impl ComplexityClass {
    /// Get human-readable description
    pub fn description(&self) -> &str {
        match self {
            ComplexityClass::Constant => "O(1) - constant time",
            ComplexityClass::Logarithmic => "O(log n) - logarithmic",
            ComplexityClass::Linear => "O(n) - linear time",
            ComplexityClass::Linearithmic => "O(n log n) - linearithmic",
            ComplexityClass::Quadratic => "O(n²) - quadratic time",
            ComplexityClass::Cubic => "O(n³) - cubic time",
            ComplexityClass::Exponential => "O(2ⁿ) - exponential time",
            ComplexityClass::Factorial => "O(n!) - factorial time",
            ComplexityClass::Unknown => "complexity not specified",
        }
    }
}

/// Performance cost indicator
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum PerformanceCost {
    /// Very fast, negligible cost
    Negligible,
    /// Fast, low cost
    Low,
    /// Moderate cost
    Moderate,
    /// Expensive operation
    High,
    /// Very expensive, should be avoided in hot paths
    VeryHigh,
    /// Unknown cost
    Unknown,
}

/// Performance tip or optimization suggestion
#[derive(Debug, Clone, PartialEq)]
pub struct PerformanceTip {
    /// The tip text
    pub tip: String,
    /// Priority/importance (0.0-1.0)
    pub priority: f64,
}

/// Timing information
#[derive(Debug, Clone, PartialEq)]
pub struct TimingInfo {
    /// Description of the scenario
    pub scenario: String,
    /// Typical timing
    pub timing: String,
}

/// Complete performance analysis
#[derive(Debug, Clone, PartialEq)]
pub struct PerformanceInfo {
    /// Function name
    pub function_name: String,
    /// Blocking behavior
    pub blocking: BlockingBehavior,
    /// Operation type
    pub operation_type: OperationType,
    /// Computational complexity
    pub complexity: ComplexityClass,
    /// Overall performance cost
    pub cost: PerformanceCost,
    /// Performance tips
    pub tips: Vec<PerformanceTip>,
    /// Timing information
    pub timing_info: Vec<TimingInfo>,
    /// Alternative faster functions
    pub alternatives: Vec<String>,
    /// Performance warnings
    pub warnings: Vec<String>,
    /// Overall confidence (0.0-1.0)
    pub confidence: f64,
}

impl PerformanceInfo {
    /// Create new empty performance info
    pub fn new(function_name: String) -> Self {
        Self {
            function_name,
            blocking: BlockingBehavior::Unknown,
            operation_type: OperationType::Unknown,
            complexity: ComplexityClass::Unknown,
            cost: PerformanceCost::Unknown,
            tips: Vec::new(),
            timing_info: Vec::new(),
            alternatives: Vec::new(),
            warnings: Vec::new(),
            confidence: 0.0,
        }
    }

    /// Check if has performance information
    pub fn has_info(&self) -> bool {
        !matches!(self.blocking, BlockingBehavior::Unknown)
            || !matches!(self.operation_type, OperationType::Unknown)
            || !matches!(self.complexity, ComplexityClass::Unknown)
            || !matches!(self.cost, PerformanceCost::Unknown)
            || !self.tips.is_empty()
            || !self.timing_info.is_empty()
            || !self.alternatives.is_empty()
            || !self.warnings.is_empty()
    }

    /// Generate documentation string
    pub fn generate_documentation(&self) -> String {
        let mut doc = String::new();

        if !self.has_info() {
            return doc;
        }

        doc.push_str("/// # Performance\n");
        doc.push_str("///\n");

        // Blocking behavior
        match self.blocking {
            BlockingBehavior::Blocking => {
                doc.push_str("/// **⚠️ BLOCKING OPERATION**\n");
                doc.push_str("///\n");
                doc.push_str("/// This function blocks the calling thread until completion.\n");
            }
            BlockingBehavior::NonBlocking => {
                doc.push_str("/// **✓ NON-BLOCKING OPERATION**\n");
                doc.push_str("///\n");
                doc.push_str("/// This function returns immediately and executes asynchronously.\n");
            }
            BlockingBehavior::MayBlock => {
                doc.push_str("/// **MAY BLOCK**\n");
                doc.push_str("///\n");
                doc.push_str("/// This function may block depending on conditions.\n");
            }
            BlockingBehavior::Unknown => {}
        }

        // Operation type
        if !matches!(self.operation_type, OperationType::Unknown) {
            let op_type = match self.operation_type {
                OperationType::Compute => "CPU Computation",
                OperationType::Memory => "Memory Operation",
                OperationType::IO => "I/O Operation",
                OperationType::GPU => "GPU Operation",
                OperationType::Synchronization => "Synchronization",
                OperationType::Unknown => "",
            };
            if !op_type.is_empty() {
                doc.push_str(&format!("/// **Operation Type:** {}\n", op_type));
            }
        }

        // Complexity
        if !matches!(self.complexity, ComplexityClass::Unknown) {
            doc.push_str(&format!("/// **Complexity:** {}\n", self.complexity.description()));
        }

        // Cost
        match self.cost {
            PerformanceCost::VeryHigh => {
                doc.push_str("/// **Cost:** ⚠️ VERY HIGH - Avoid in hot paths\n");
            }
            PerformanceCost::High => {
                doc.push_str("/// **Cost:** High - Expensive operation\n");
            }
            PerformanceCost::Moderate => {
                doc.push_str("/// **Cost:** Moderate\n");
            }
            PerformanceCost::Low => {
                doc.push_str("/// **Cost:** Low - Fast operation\n");
            }
            PerformanceCost::Negligible => {
                doc.push_str("/// **Cost:** Negligible - Very fast\n");
            }
            PerformanceCost::Unknown => {}
        }

        // Timing information
        if !self.timing_info.is_empty() {
            doc.push_str("///\n");
            doc.push_str("/// **Typical Performance:**\n");
            for timing in &self.timing_info {
                doc.push_str(&format!("/// - {}: {}\n", timing.scenario, timing.timing));
            }
        }

        // Alternatives
        if !self.alternatives.is_empty() {
            doc.push_str("///\n");
            doc.push_str("/// **Faster Alternatives:**\n");
            for alt in &self.alternatives {
                doc.push_str(&format!("/// - `{}`\n", alt));
            }
        }

        // Performance tips
        if !self.tips.is_empty() {
            doc.push_str("///\n");
            doc.push_str("/// **Performance Tips:**\n");
            for tip in &self.tips {
                doc.push_str(&format!("/// - {}\n", tip.tip));
            }
        }

        // Warnings
        if !self.warnings.is_empty() {
            doc.push_str("///\n");
            doc.push_str("/// **Performance Warnings:**\n");
            for warning in &self.warnings {
                doc.push_str(&format!("/// - ⚠️ {}\n", warning));
            }
        }

        doc
    }
}

impl Default for PerformanceInfo {
    fn default() -> Self {
        Self::new(String::new())
    }
}

/// Performance analyzer
#[derive(Debug)]
pub struct PerformanceAnalyzer {
    /// Cache of analyzed functions
    cache: HashMap<String, PerformanceInfo>,
}

impl PerformanceAnalyzer {
    /// Create new analyzer
    pub fn new() -> Self {
        Self {
            cache: HashMap::new(),
        }
    }

    /// Analyze performance characteristics
    pub fn analyze(&mut self, name: &str, documentation: Option<&str>) -> PerformanceInfo {
        // Check cache
        if let Some(cached) = self.cache.get(name) {
            return cached.clone();
        }

        let mut info = PerformanceInfo::new(name.to_string());

        // Analyze from function name
        self.analyze_from_name(name, &mut info);

        // Analyze from documentation
        if let Some(doc) = documentation {
            self.analyze_from_documentation(doc, &mut info);
        }

        // Calculate overall confidence
        info.confidence = self.calculate_confidence(&info);

        // Cache result
        self.cache.insert(name.to_string(), info.clone());

        info
    }

    /// Analyze from function name patterns
    fn analyze_from_name(&self, name: &str, info: &mut PerformanceInfo) {
        let name_lower = name.to_lowercase();

        // Async/sync patterns
        if name_lower.ends_with("async") || name_lower.contains("_async_") {
            info.blocking = BlockingBehavior::NonBlocking;
        } else if name_lower.ends_with("sync") || name_lower.contains("_sync_") {
            info.blocking = BlockingBehavior::Blocking;
        } else if name_lower.starts_with("wait") || name_lower.contains("_wait_") {
            info.blocking = BlockingBehavior::Blocking;
        }

        // Operation type patterns
        if name_lower.contains("memcpy")
            || name_lower.contains("copy")
            || name_lower.contains("alloc")
        {
            info.operation_type = OperationType::Memory;
        } else if name_lower.contains("compute")
            || name_lower.contains("calculate")
            || name_lower.contains("process")
        {
            info.operation_type = OperationType::Compute;
        } else if name_lower.contains("read")
            || name_lower.contains("write")
            || name_lower.contains("file")
            || name_lower.contains("socket")
        {
            info.operation_type = OperationType::IO;
        } else if name_lower.contains("gpu")
            || name_lower.contains("cuda")
            || name_lower.contains("kernel")
        {
            info.operation_type = OperationType::GPU;
        } else if name_lower.contains("lock")
            || name_lower.contains("mutex")
            || name_lower.contains("barrier")
        {
            info.operation_type = OperationType::Synchronization;
        }
    }

    /// Analyze from documentation
    fn analyze_from_documentation(&self, doc: &str, info: &mut PerformanceInfo) {
        let doc_lower = doc.to_lowercase();

        // Blocking behavior
        if doc_lower.contains("blocking")
            || doc_lower.contains("blocks")
            || doc_lower.contains("will block")
        {
            info.blocking = BlockingBehavior::Blocking;
        } else if doc_lower.contains("non-blocking")
            || doc_lower.contains("asynchronous")
            || doc_lower.contains("does not block")
            || doc_lower.contains("returns immediately")
        {
            info.blocking = BlockingBehavior::NonBlocking;
        } else if doc_lower.contains("may block") {
            info.blocking = BlockingBehavior::MayBlock;
        }

        // Operation type
        if doc_lower.contains("gpu operation") || doc_lower.contains("device operation") {
            info.operation_type = OperationType::GPU;
        } else if doc_lower.contains("i/o operation") || doc_lower.contains("disk access") {
            info.operation_type = OperationType::IO;
        } else if doc_lower.contains("cpu-intensive") || doc_lower.contains("computation") {
            info.operation_type = OperationType::Compute;
        }

        // Cost indicators
        if doc_lower.contains("very expensive")
            || doc_lower.contains("extremely slow")
            || doc_lower.contains("avoid in hot")
        {
            info.cost = PerformanceCost::VeryHigh;
        } else if doc_lower.contains("expensive") || doc_lower.contains("slow") {
            info.cost = PerformanceCost::High;
        } else if doc_lower.contains("fast") || doc_lower.contains("efficient") {
            info.cost = PerformanceCost::Low;
        }

        // Complexity analysis
        self.extract_complexity(doc, info);

        // Extract timing information
        self.extract_timing(doc, info);

        // Extract tips
        self.extract_tips(doc, info);

        // Extract alternatives
        self.extract_alternatives(doc, info);

        // Extract warnings
        self.extract_warnings(doc, info);
    }

    /// Extract complexity notation
    fn extract_complexity(&self, doc: &str, info: &mut PerformanceInfo) {
        let doc_lower = doc.to_lowercase();

        if doc_lower.contains("o(1)") || doc_lower.contains("constant time") {
            info.complexity = ComplexityClass::Constant;
        } else if doc_lower.contains("o(log n)") || doc_lower.contains("logarithmic") {
            info.complexity = ComplexityClass::Logarithmic;
        } else if doc_lower.contains("o(n log n)") {
            info.complexity = ComplexityClass::Linearithmic;
        } else if doc_lower.contains("o(n)") || doc_lower.contains("linear time") {
            info.complexity = ComplexityClass::Linear;
        } else if doc_lower.contains("o(n²)")
            || doc_lower.contains("o(n^2)")
            || doc_lower.contains("quadratic")
        {
            info.complexity = ComplexityClass::Quadratic;
        } else if doc_lower.contains("o(n³)")
            || doc_lower.contains("o(n^3)")
            || doc_lower.contains("cubic")
        {
            info.complexity = ComplexityClass::Cubic;
        } else if doc_lower.contains("o(2^n)")
            || doc_lower.contains("o(2ⁿ)")
            || doc_lower.contains("exponential")
        {
            info.complexity = ComplexityClass::Exponential;
        } else if doc_lower.contains("o(n!)") || doc_lower.contains("factorial") {
            info.complexity = ComplexityClass::Factorial;
        }
    }

    /// Extract timing information
    fn extract_timing(&self, doc: &str, info: &mut PerformanceInfo) {
        // Look for timing patterns like "~10-50μs", "1-5ms", etc.
        let lines: Vec<&str> = doc.lines().collect();

        for line in lines {
            let line_lower = line.to_lowercase();

            // Look for timing indicators
            if line_lower.contains("μs")
                || line_lower.contains("ms")
                || line_lower.contains("seconds")
                || line_lower.contains("gb/s")
            {
                // Extract scenario and timing
                if let Some(colon_pos) = line.find(':') {
                    let scenario = line[..colon_pos].trim().to_string();
                    let timing = line[colon_pos + 1..].trim().to_string();

                    // Clean up common prefixes
                    let scenario = scenario
                        .trim_start_matches('-')
                        .trim_start_matches('*')
                        .trim()
                        .to_string();

                    if !scenario.is_empty() && !timing.is_empty() {
                        info.timing_info.push(TimingInfo { scenario, timing });
                    }
                }
            }
        }
    }

    /// Extract performance tips
    fn extract_tips(&self, doc: &str, info: &mut PerformanceInfo) {
        let doc_lower = doc.to_lowercase();

        // Look for tip sections
        if doc_lower.contains("performance tip") || doc_lower.contains("optimization") {
            let lines: Vec<&str> = doc.lines().collect();
            let mut in_tips_section = false;

            for line in lines {
                let line_lower = line.to_lowercase();

                if line_lower.contains("performance tip")
                    || line_lower.contains("optimization")
                    || line_lower.contains("best practice")
                {
                    in_tips_section = true;
                    continue;
                }

                if in_tips_section {
                    // Stop at next section
                    if line.starts_with('#') {
                        break;
                    }

                    // Extract bullet points
                    if line.trim_start().starts_with('-') || line.trim_start().starts_with('*') {
                        let tip = line
                            .trim_start()
                            .trim_start_matches('-')
                            .trim_start_matches('*')
                            .trim()
                            .to_string();

                        if !tip.is_empty() {
                            info.tips.push(PerformanceTip {
                                tip,
                                priority: 0.7,
                            });
                        }
                    }
                }
            }
        }

        // Common optimization patterns
        if doc_lower.contains("use pinned memory") {
            info.tips.push(PerformanceTip {
                tip: "Use pinned memory for faster transfers".to_string(),
                priority: 0.8,
            });
        }
        if doc_lower.contains("batch") {
            info.tips.push(PerformanceTip {
                tip: "Batch multiple operations for better performance".to_string(),
                priority: 0.7,
            });
        }
    }

    /// Extract alternative functions
    fn extract_alternatives(&self, doc: &str, info: &mut PerformanceInfo) {
        let doc_lower = doc.to_lowercase();

        // Look for alternative suggestions
        if doc_lower.contains("faster") || doc_lower.contains("alternative") {
            let lines: Vec<&str> = doc.lines().collect();

            for line in lines {
                let line_lower = line.to_lowercase();

                if (line_lower.contains("use") || line_lower.contains("consider"))
                    && (line_lower.contains("faster") || line_lower.contains("instead"))
                {
                    // Extract function names in backticks
                    let mut in_backtick = false;
                    let mut func_name = String::new();

                    for ch in line.chars() {
                        if ch == '`' {
                            if in_backtick {
                                if !func_name.is_empty() && func_name.contains('(') {
                                    // Remove () suffix
                                    if let Some(paren_pos) = func_name.find('(') {
                                        func_name = func_name[..paren_pos].to_string();
                                    }
                                    info.alternatives.push(func_name.clone());
                                }
                                func_name.clear();
                                in_backtick = false;
                            } else {
                                in_backtick = true;
                            }
                        } else if in_backtick {
                            func_name.push(ch);
                        }
                    }
                }
            }
        }
    }

    /// Extract performance warnings
    fn extract_warnings(&self, doc: &str, info: &mut PerformanceInfo) {
        let doc_lower = doc.to_lowercase();

        if (doc_lower.contains("avoid calling in") || doc_lower.contains("do not call in"))
            && (doc_lower.contains("hot path") || doc_lower.contains("tight loop")) {
                info.warnings
                    .push("Avoid calling in performance-critical hot paths".to_string());
            }

        if doc_lower.contains("requires synchronization") {
            info.warnings
                .push("Requires synchronization - may impact performance".to_string());
        }

        if doc_lower.contains("allocates memory") || doc_lower.contains("heap allocation") {
            info.warnings
                .push("Allocates memory - may cause allocator contention".to_string());
        }
    }

    /// Calculate overall confidence
    fn calculate_confidence(&self, info: &PerformanceInfo) -> f64 {
        let mut score = 0.0;
        let mut count = 0;

        // High confidence if we detected blocking behavior
        if !matches!(info.blocking, BlockingBehavior::Unknown) {
            score += 0.9;
            count += 1;
        }

        // Medium confidence for operation type
        if !matches!(info.operation_type, OperationType::Unknown) {
            score += 0.7;
            count += 1;
        }

        // High confidence for complexity
        if !matches!(info.complexity, ComplexityClass::Unknown) {
            score += 0.9;
            count += 1;
        }

        // Medium confidence for cost
        if !matches!(info.cost, PerformanceCost::Unknown) {
            score += 0.7;
            count += 1;
        }

        // Tips and timing add confidence
        if !info.tips.is_empty() {
            score += 0.8;
            count += 1;
        }

        if !info.timing_info.is_empty() {
            score += 0.95;
            count += 1;
        }

        if count > 0 {
            score / count as f64
        } else {
            0.0
        }
    }

    /// Clear cache
    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }
}

impl Default for PerformanceAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_async_from_name() {
        let mut analyzer = PerformanceAnalyzer::new();
        let info = analyzer.analyze("memcpy_async", None);

        assert_eq!(info.blocking, BlockingBehavior::NonBlocking);
    }

    #[test]
    fn test_sync_from_name() {
        let mut analyzer = PerformanceAnalyzer::new();
        let info = analyzer.analyze("device_sync", None);

        assert_eq!(info.blocking, BlockingBehavior::Blocking);
    }

    #[test]
    fn test_wait_from_name() {
        let mut analyzer = PerformanceAnalyzer::new();
        let info = analyzer.analyze("wait_for_completion", None);

        assert_eq!(info.blocking, BlockingBehavior::Blocking);
    }

    #[test]
    fn test_memory_operation() {
        let mut analyzer = PerformanceAnalyzer::new();
        let info = analyzer.analyze("cudaMemcpy", None);

        assert_eq!(info.operation_type, OperationType::Memory);
    }

    #[test]
    fn test_gpu_operation() {
        let mut analyzer = PerformanceAnalyzer::new();
        let info = analyzer.analyze("cuda_kernel_launch", None);

        assert_eq!(info.operation_type, OperationType::GPU);
    }

    #[test]
    fn test_blocking_from_doc() {
        let mut analyzer = PerformanceAnalyzer::new();
        let doc = "This function blocks the calling thread until completion.";
        let info = analyzer.analyze("func", Some(doc));

        assert_eq!(info.blocking, BlockingBehavior::Blocking);
    }

    #[test]
    fn test_non_blocking_from_doc() {
        let mut analyzer = PerformanceAnalyzer::new();
        let doc = "This function returns immediately and executes asynchronously.";
        let info = analyzer.analyze("func", Some(doc));

        assert_eq!(info.blocking, BlockingBehavior::NonBlocking);
    }

    #[test]
    fn test_complexity_linear() {
        let mut analyzer = PerformanceAnalyzer::new();
        let doc = "This operation runs in O(n) linear time.";
        let info = analyzer.analyze("func", Some(doc));

        assert_eq!(info.complexity, ComplexityClass::Linear);
    }

    #[test]
    fn test_complexity_quadratic() {
        let mut analyzer = PerformanceAnalyzer::new();
        let doc = "Complexity: O(n²) quadratic time.";
        let info = analyzer.analyze("func", Some(doc));

        assert_eq!(info.complexity, ComplexityClass::Quadratic);
    }

    #[test]
    fn test_cost_expensive() {
        let mut analyzer = PerformanceAnalyzer::new();
        let doc = "This is an expensive operation that should be avoided.";
        let info = analyzer.analyze("func", Some(doc));

        assert_eq!(info.cost, PerformanceCost::High);
    }

    #[test]
    fn test_extract_timing() {
        let mut analyzer = PerformanceAnalyzer::new();
        let doc = "Typical timing:\n- Small transfers (<1MB): ~10-50μs\n- Large transfers: 1-5ms";
        let info = analyzer.analyze("func", Some(doc));

        assert_eq!(info.timing_info.len(), 2);
        assert!(info.timing_info[0].scenario.contains("Small"));
        assert!(info.timing_info[0].timing.contains("μs"));
    }

    #[test]
    fn test_extract_tips() {
        let mut analyzer = PerformanceAnalyzer::new();
        let doc = "Performance tips:\n- Use pinned memory\n- Batch operations";
        let info = analyzer.analyze("func", Some(doc));

        assert!(!info.tips.is_empty());
    }

    #[test]
    fn test_extract_alternatives() {
        let mut analyzer = PerformanceAnalyzer::new();
        let doc = "For faster performance, use `memcpy_async()` instead.";
        let info = analyzer.analyze("func", Some(doc));

        assert_eq!(info.alternatives.len(), 1);
        assert_eq!(info.alternatives[0], "memcpy_async");
    }

    #[test]
    fn test_docs_generation() {
        let mut analyzer = PerformanceAnalyzer::new();
        let doc = "This function blocks. O(n) complexity. Expensive operation.";
        let info = analyzer.analyze("func", Some(doc));

        let docs = info.generate_documentation();
        assert!(docs.contains("Performance"));
        assert!(docs.contains("BLOCKING"));
        assert!(docs.contains("O(n)"));
    }

    #[test]
    fn test_cache() {
        let mut analyzer = PerformanceAnalyzer::new();
        let doc = "Blocking operation.";

        let info1 = analyzer.analyze("func", Some(doc));
        let info2 = analyzer.analyze("func", Some(doc));

        assert_eq!(info1.function_name, info2.function_name);
        assert_eq!(info1.blocking, info2.blocking);
    }

    #[test]
    fn test_no_performance_info() {
        let mut analyzer = PerformanceAnalyzer::new();
        let info = analyzer.analyze("func", None);

        assert!(!info.has_info());
        assert!(info.generate_documentation().is_empty());
    }
}
