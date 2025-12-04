//! Performance optimization features for generated bindings
//!
//! This module provides:
//! - Zero-cost abstraction verification
//! - Inline hint generation and placement
//! - Benchmarking framework integration
//! - Hot path identification
//! - Profile-guided optimization support

use crate::ffi::parser::{FfiFunction, FfiInfo};
use std::collections::{HashMap, HashSet};

/// Inline hint strategy for a function
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum InlineHint {
    /// Always inline (trivial getters, simple operations)
    Always,
    /// Suggest inlining, let compiler decide
    Suggest,
    /// Never inline (large functions, cold paths)
    Never,
    /// No hint (default compiler behavior)
    None,
}

impl InlineHint {
    /// Get the Rust attribute string
    pub fn to_attribute(&self) -> Option<&'static str> {
        match self {
            InlineHint::Always => Some("#[inline(always)]"),
            InlineHint::Suggest => Some("#[inline]"),
            InlineHint::Never => Some("#[inline(never)]"),
            InlineHint::None => None,
        }
    }
}

/// Function performance characteristics
#[derive(Debug, Clone)]
pub struct FunctionPerformance {
    /// Function name
    pub name: String,
    /// Estimated complexity (0-100, higher = more complex)
    pub complexity: u32,
    /// Whether function is on hot path
    pub is_hot_path: bool,
    /// Recommended inline hint
    pub inline_hint: InlineHint,
    /// Estimated instruction count
    pub instruction_count: u32,
    /// Whether function allocates
    pub allocates: bool,
    /// Whether function has side effects
    pub has_side_effects: bool,
}

impl FunctionPerformance {
    /// Create performance analysis for a function
    pub fn analyze(func: &FfiFunction) -> Self {
        let complexity = Self::estimate_complexity(func);
        let instruction_count = Self::estimate_instructions(func);
        let inline_hint = Self::recommend_inline(complexity, instruction_count);
        let allocates = Self::likely_allocates(func);
        let has_side_effects = Self::has_side_effects(func);

        Self {
            name: func.name.clone(),
            complexity,
            is_hot_path: false, // Will be determined by profiling
            inline_hint,
            instruction_count,
            allocates,
            has_side_effects,
        }
    }

    /// Estimate function complexity (0-100)
    fn estimate_complexity(func: &FfiFunction) -> u32 {
        let mut complexity = 0u32;

        // Parameter count contributes to complexity
        complexity += (func.params.len() as u32) * 3;

        // Pointer parameters add complexity
        let pointer_count = func.params.iter().filter(|p| p.is_pointer).count() as u32;
        complexity += pointer_count * 5;

        // Void* adds extra complexity (type-unsafe)
        let void_ptr_count = func
            .params
            .iter()
            .filter(|p| p.ty.contains("void") && p.is_pointer)
            .count() as u32;
        complexity += void_ptr_count * 10;

        // Cap at 100
        complexity.min(100)
    }

    /// Estimate instruction count
    fn estimate_instructions(func: &FfiFunction) -> u32 {
        // Base: FFI call overhead (~10-20 instructions)
        let mut count = 15;

        // Each parameter adds ~2-3 instructions
        count += (func.params.len() as u32) * 2;

        // Pointer parameters need null checks in safe mode
        let pointer_count = func.params.iter().filter(|p| p.is_pointer).count() as u32;
        count += pointer_count * 5; // null check + branch

        // Return value processing
        if !func.return_type.is_empty() && func.return_type != "void" {
            count += 5;
        }

        count
    }

    /// Recommend inline strategy
    fn recommend_inline(complexity: u32, instruction_count: u32) -> InlineHint {
        if instruction_count <= 5 && complexity <= 10 {
            // Trivial functions: always inline
            InlineHint::Always
        } else if instruction_count <= 30 && complexity <= 40 {
            // Small functions: suggest inline
            InlineHint::Suggest
        } else if instruction_count > 100 || complexity > 70 {
            // Large/complex functions: never inline
            InlineHint::Never
        } else {
            // Medium functions: let compiler decide
            InlineHint::None
        }
    }

    /// Check if function likely allocates
    fn likely_allocates(func: &FfiFunction) -> bool {
        let name_lower = func.name.to_lowercase();

        // Functions with these patterns likely allocate
        name_lower.contains("create")
            || name_lower.contains("alloc")
            || name_lower.contains("new")
            || name_lower.contains("init")
            || name_lower.contains("copy")
            || name_lower.contains("clone")
    }

    /// Check if function has side effects
    fn has_side_effects(func: &FfiFunction) -> bool {
        // Functions with mutable parameters or non-const operations have side effects
        func.params.iter().any(|p| p.is_mut)
            || func.name.to_lowercase().contains("set")
            || func.name.to_lowercase().contains("write")
            || func.name.to_lowercase().contains("modify")
    }
}

/// Performance optimizer for FFI bindings
pub struct PerformanceOptimizer {
    /// Performance data for each function
    functions: HashMap<String, FunctionPerformance>,
    /// Hot path functions (frequently called)
    hot_paths: HashSet<String>,
}

impl PerformanceOptimizer {
    /// Create a new optimizer
    pub fn new() -> Self {
        Self {
            functions: HashMap::new(),
            hot_paths: HashSet::new(),
        }
    }

    /// Analyze FFI info for performance optimization
    pub fn analyze(&mut self, ffi_info: &FfiInfo) {
        for func in &ffi_info.functions {
            let perf = FunctionPerformance::analyze(func);
            self.functions.insert(func.name.clone(), perf);
        }

        // Identify potential hot paths (heuristics)
        self.identify_hot_paths();
    }

    /// Identify likely hot path functions
    fn identify_hot_paths(&mut self) {
        for (name, perf) in &self.functions {
            let name_lower = name.to_lowercase();

            // Getters are often hot paths
            if name_lower.starts_with("get_") && perf.complexity < 20 {
                self.hot_paths.insert(name.clone());
            }

            // Status/query functions are hot paths
            if name_lower.contains("status")
                || name_lower.contains("query")
                || name_lower.contains("is_")
                || name_lower.contains("has_")
            {
                self.hot_paths.insert(name.clone());
            }

            // Simple operations are often called frequently
            if perf.complexity < 15 && !perf.allocates {
                self.hot_paths.insert(name.clone());
            }
        }
    }

    /// Mark a function as hot path (from profiling data)
    pub fn mark_hot_path(&mut self, function_name: &str) {
        self.hot_paths.insert(function_name.to_string());
        if let Some(perf) = self.functions.get_mut(function_name) {
            perf.is_hot_path = true;
        }
    }

    /// Get inline hint for a function
    pub fn get_inline_hint(&self, function_name: &str) -> InlineHint {
        self.functions
            .get(function_name)
            .map(|p| p.inline_hint)
            .unwrap_or(InlineHint::None)
    }

    /// Get all hot path functions
    pub fn hot_paths(&self) -> &HashSet<String> {
        &self.hot_paths
    }

    /// Generate performance report
    pub fn generate_report(&self) -> PerformanceReport {
        let mut hot_path_functions = Vec::new();
        let mut inline_always = Vec::new();
        let mut inline_never = Vec::new();
        let mut allocating_functions = Vec::new();

        for (name, perf) in &self.functions {
            if self.hot_paths.contains(name) {
                hot_path_functions.push(perf.clone());
            }

            match perf.inline_hint {
                InlineHint::Always => inline_always.push(name.clone()),
                InlineHint::Never => inline_never.push(name.clone()),
                _ => {}
            }

            if perf.allocates {
                allocating_functions.push(perf.clone());
            }
        }

        // Sort by complexity
        hot_path_functions.sort_by_key(|f| f.complexity);
        allocating_functions.sort_by_key(|f| std::cmp::Reverse(f.complexity));

        PerformanceReport {
            total_functions: self.functions.len(),
            hot_path_count: hot_path_functions.len(),
            inline_always_count: inline_always.len(),
            inline_never_count: inline_never.len(),
            allocating_count: allocating_functions.len(),
            hot_path_functions,
            inline_always,
            inline_never,
            allocating_functions,
        }
    }

    /// Generate benchmark code for hot paths
    pub fn generate_benchmarks(&self) -> String {
        let mut output = String::from(
            r#"//! Generated benchmarks for hot path functions
//!
//! Run with: cargo bench

use criterion::{black_box, criterion_group, criterion_main, Criterion};

"#,
        );

        for func_name in &self.hot_paths {
            if let Some(perf) = self.functions.get(func_name) {
                output.push_str(&format!(
                    r#"fn bench_{}(c: &mut Criterion) {{
    c.bench_function("{}", |b| {{
        b.iter(|| {{
            // TODO: Initialize test data
            // let handle = create_test_handle();
            // black_box(handle.{}());
        }});
    }});
}}

"#,
                    func_name, func_name, func_name
                ));
            }
        }

        // Add criterion group
        output.push_str("criterion_group!(benches");
        for func_name in &self.hot_paths {
            output.push_str(&format!(", bench_{}", func_name));
        }
        output.push_str(");\n");
        output.push_str("criterion_main!(benches);\n");

        output
    }
}

impl Default for PerformanceOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

/// Performance analysis report
#[derive(Debug, Clone)]
pub struct PerformanceReport {
    /// Total number of functions
    pub total_functions: usize,
    /// Number of hot path functions
    pub hot_path_count: usize,
    /// Functions with inline(always)
    pub inline_always_count: usize,
    /// Functions with inline(never)
    pub inline_never_count: usize,
    /// Functions that allocate
    pub allocating_count: usize,
    /// Hot path function details
    pub hot_path_functions: Vec<FunctionPerformance>,
    /// Functions to always inline
    pub inline_always: Vec<String>,
    /// Functions to never inline
    pub inline_never: Vec<String>,
    /// Allocating function details
    pub allocating_functions: Vec<FunctionPerformance>,
}

impl PerformanceReport {
    /// Generate markdown report
    pub fn to_markdown(&self) -> String {
        let mut output = String::from("# Performance Analysis Report\n\n");

        // Executive summary
        output.push_str("## Executive Summary\n\n");
        output.push_str(&format!("- **Total Functions**: {}\n", self.total_functions));
        output.push_str(&format!(
            "- **Hot Path Functions**: {} ({:.1}%)\n",
            self.hot_path_count,
            (self.hot_path_count as f64 / self.total_functions as f64) * 100.0
        ));
        output.push_str(&format!(
            "- **Always Inline**: {}\n",
            self.inline_always_count
        ));
        output.push_str(&format!("- **Never Inline**: {}\n", self.inline_never_count));
        output.push_str(&format!(
            "- **Allocating Functions**: {} ({:.1}%)\n\n",
            self.allocating_count,
            (self.allocating_count as f64 / self.total_functions as f64) * 100.0
        ));

        // Hot paths
        if !self.hot_path_functions.is_empty() {
            output.push_str("## Hot Path Functions\n\n");
            output.push_str(
                "These functions are likely called frequently and should be optimized:\n\n",
            );
            output.push_str("| Function | Complexity | Instructions | Inline | Allocates |\n");
            output.push_str("|----------|------------|--------------|--------|----------|\n");

            for func in &self.hot_path_functions {
                output.push_str(&format!(
                    "| `{}` | {} | ~{} | {:?} | {} |\n",
                    func.name,
                    func.complexity,
                    func.instruction_count,
                    func.inline_hint,
                    if func.allocates { "Yes" } else { "No" }
                ));
            }
            output.push_str("\n");
        }

        // Inline recommendations
        if !self.inline_always.is_empty() {
            output.push_str("## Always Inline (Trivial Functions)\n\n");
            output.push_str("These functions are simple enough to always inline:\n\n");
            for func in &self.inline_always {
                output.push_str(&format!("- `{}`\n", func));
            }
            output.push_str("\n");
        }

        if !self.inline_never.is_empty() {
            output.push_str("## Never Inline (Complex Functions)\n\n");
            output.push_str("These functions are too large/complex to inline:\n\n");
            for func in &self.inline_never {
                output.push_str(&format!("- `{}`\n", func));
            }
            output.push_str("\n");
        }

        // Allocating functions
        if !self.allocating_functions.is_empty() {
            output.push_str("## Allocating Functions\n\n");
            output.push_str("These functions likely allocate memory. Consider:\n");
            output.push_str("- Pre-allocating buffers\n");
            output.push_str("- Using memory pools\n");
            output.push_str("- Reusing allocations\n\n");

            output.push_str("| Function | Complexity | Hot Path |\n");
            output.push_str("|----------|------------|----------|\n");

            for func in self.allocating_functions.iter().take(10) {
                output.push_str(&format!(
                    "| `{}` | {} | {} |\n",
                    func.name,
                    func.complexity,
                    if func.is_hot_path { "Yes ⚠️" } else { "No" }
                ));
            }
            output.push_str("\n");
        }

        // Recommendations
        output.push_str("## Optimization Recommendations\n\n");

        if self.hot_path_count > 0 {
            output.push_str("### Hot Path Optimization\n");
            output.push_str(&format!(
                "1. Profile hot paths with `cargo bench` (use generated benchmarks)\n"
            ));
            output.push_str("2. Consider `#[inline]` for frequently called functions\n");
            output.push_str("3. Minimize allocations in hot paths\n\n");
        }

        if self.allocating_count > self.total_functions / 4 {
            output.push_str("### Memory Optimization\n");
            output.push_str(&format!(
                "1. High allocation rate ({:.1}% of functions)\n",
                (self.allocating_count as f64 / self.total_functions as f64) * 100.0
            ));
            output.push_str("2. Consider implementing memory pools\n");
            output.push_str("3. Reuse buffers where possible\n\n");
        }

        output.push_str("### Zero-Cost Abstraction Verification\n");
        output.push_str("1. Compare assembly output: `cargo asm <function>`\n");
        output.push_str("2. Verify wrapper overhead is minimal\n");
        output.push_str("3. Use `#[inline(always)]` for getters\n\n");

        output
    }
}

/// Zero-cost abstraction verifier
pub struct ZeroCostVerifier {
    /// Functions verified as zero-cost
    verified: HashSet<String>,
    /// Functions with overhead
    overhead: HashMap<String, String>,
}

impl ZeroCostVerifier {
    /// Create a new verifier
    pub fn new() -> Self {
        Self {
            verified: HashSet::new(),
            overhead: HashMap::new(),
        }
    }

    /// Mark function as verified zero-cost
    pub fn mark_verified(&mut self, function_name: &str) {
        self.verified.insert(function_name.to_string());
        self.overhead.remove(function_name);
    }

    /// Mark function as having overhead
    pub fn mark_overhead(&mut self, function_name: &str, reason: String) {
        self.overhead.insert(function_name.to_string(), reason);
        self.verified.remove(function_name);
    }

    /// Check if function is verified
    pub fn is_verified(&self, function_name: &str) -> bool {
        self.verified.contains(function_name)
    }

    /// Generate verification report
    pub fn generate_report(&self) -> String {
        let mut output = String::from("# Zero-Cost Abstraction Report\n\n");

        output.push_str(&format!(
            "- **Verified**: {} functions\n",
            self.verified.len()
        ));
        output.push_str(&format!(
            "- **With Overhead**: {} functions\n\n",
            self.overhead.len()
        ));

        if !self.overhead.is_empty() {
            output.push_str("## Functions with Overhead\n\n");
            output.push_str("| Function | Reason |\n");
            output.push_str("|----------|--------|\n");

            for (func, reason) in &self.overhead {
                output.push_str(&format!("| `{}` | {} |\n", func, reason));
            }
            output.push_str("\n");
        }

        output
    }
}

impl Default for ZeroCostVerifier {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ffi::parser::FfiParam;

    #[test]
    fn test_inline_hint_attributes() {
        assert_eq!(
            InlineHint::Always.to_attribute(),
            Some("#[inline(always)]")
        );
        assert_eq!(InlineHint::Suggest.to_attribute(), Some("#[inline]"));
        assert_eq!(InlineHint::Never.to_attribute(), Some("#[inline(never)]"));
        assert_eq!(InlineHint::None.to_attribute(), None);
    }

    #[test]
    fn test_function_complexity() {
        let func = FfiFunction {
            name: "simple_func".to_string(),
            params: vec![],
            return_type: "void".to_string(),
            docs: None,
        };

        let perf = FunctionPerformance::analyze(&func);
        assert_eq!(perf.complexity, 0);
    }

    #[test]
    fn test_function_complexity_with_params() {
        let func = FfiFunction {
            name: "complex_func".to_string(),
            params: vec![
                FfiParam {
                    name: "a".to_string(),
                    ty: "int".to_string(),
                    is_pointer: false,
                    is_mut: false,
                },
                FfiParam {
                    name: "ptr".to_string(),
                    ty: "void".to_string(),
                    is_pointer: true,
                    is_mut: false,
                },
            ],
            return_type: "int".to_string(),
            docs: None,
        };

        let perf = FunctionPerformance::analyze(&func);
        assert!(perf.complexity > 0);
        assert!(perf.instruction_count > 15);
    }

    #[test]
    fn test_likely_allocates() {
        let create_func = FfiFunction {
            name: "create_context".to_string(),
            params: vec![],
            return_type: "void*".to_string(),
            docs: None,
        };

        let perf = FunctionPerformance::analyze(&create_func);
        assert!(perf.allocates);
    }

    #[test]
    fn test_has_side_effects() {
        let mut_func = FfiFunction {
            name: "set_value".to_string(),
            params: vec![FfiParam {
                name: "handle".to_string(),
                ty: "Handle".to_string(),
                is_pointer: true,
                is_mut: true,
            }],
            return_type: "void".to_string(),
            docs: None,
        };

        let perf = FunctionPerformance::analyze(&mut_func);
        assert!(perf.has_side_effects);
    }

    #[test]
    fn test_inline_recommendations() {
        // Trivial function
        assert_eq!(
            FunctionPerformance::recommend_inline(5, 3),
            InlineHint::Always
        );

        // Small function
        assert_eq!(
            FunctionPerformance::recommend_inline(20, 25),
            InlineHint::Suggest
        );

        // Large function
        assert_eq!(
            FunctionPerformance::recommend_inline(80, 150),
            InlineHint::Never
        );

        // Medium function
        assert_eq!(
            FunctionPerformance::recommend_inline(50, 60),
            InlineHint::None
        );
    }

    #[test]
    fn test_performance_optimizer() {
        let mut optimizer = PerformanceOptimizer::new();
        assert_eq!(optimizer.functions.len(), 0);
        assert_eq!(optimizer.hot_paths.len(), 0);
    }

    #[test]
    fn test_hot_path_identification() {
        let ffi_info = FfiInfo {
            functions: vec![
                FfiFunction {
                    name: "get_status".to_string(),
                    params: vec![],
                    return_type: "int".to_string(),
                    docs: None,
                },
                FfiFunction {
                    name: "complex_operation".to_string(),
                    params: vec![
                        FfiParam {
                            name: "a".to_string(),
                            ty: "int".to_string(),
                            is_pointer: false,
                            is_mut: false,
                        },
                        FfiParam {
                            name: "b".to_string(),
                            ty: "int".to_string(),
                            is_pointer: false,
                            is_mut: false,
                        },
                    ],
                    return_type: "void".to_string(),
                    docs: None,
                },
            ],
            types: vec![],
            enums: vec![],
            constants: vec![],
            opaque_types: vec![],
            dependencies: vec![],
            type_aliases: HashMap::new(),
        };

        let mut optimizer = PerformanceOptimizer::new();
        optimizer.analyze(&ffi_info);

        // get_status should be identified as hot path
        assert!(optimizer.hot_paths().contains("get_status"));
    }

    #[test]
    fn test_zero_cost_verifier() {
        let mut verifier = ZeroCostVerifier::new();

        verifier.mark_verified("fast_func");
        assert!(verifier.is_verified("fast_func"));

        verifier.mark_overhead("slow_func", "Bounds checking overhead".to_string());
        assert!(!verifier.is_verified("slow_func"));
    }

    #[test]
    fn test_performance_report_generation() {
        let report = PerformanceReport {
            total_functions: 10,
            hot_path_count: 3,
            inline_always_count: 2,
            inline_never_count: 1,
            allocating_count: 4,
            hot_path_functions: vec![],
            inline_always: vec!["getter".to_string()],
            inline_never: vec!["huge_func".to_string()],
            allocating_functions: vec![],
        };

        let markdown = report.to_markdown();
        assert!(markdown.contains("**Total Functions**: 10"));
        assert!(markdown.contains("**Hot Path Functions**: 3"));
        assert!(markdown.contains("`getter`"));
        assert!(markdown.contains("`huge_func`"));
    }
}
