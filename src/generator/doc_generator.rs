//! Documentation generation from enriched context
//!
//! This module generates comprehensive safety documentation using all available analyzers.

use crate::analyzer::llm_doc_orchestrator::EnhancedDocumentation;
use crate::enrichment::context::FunctionContext;
use std::fmt::Write;

/// Generate enhanced documentation block from enriched context
pub fn generate_enhanced_docs(
    context: &FunctionContext,
    indent: &str,
    enhanced_docs: Option<&EnhancedDocumentation>,
) -> String {
    let mut doc = String::new();

    // Primary description
    if let Some(desc) = &context.description {
        for line in desc.lines() {
            writeln!(doc, "{}/// {}", indent, line).unwrap();
        }
        writeln!(doc, "{}///", indent).unwrap();
    }

    // Safety section - the most important part
    let safety_items = collect_safety_items(context);
    if !safety_items.is_empty() {
        writeln!(doc, "{}/// # Safety", indent).unwrap();
        writeln!(doc, "{}///", indent).unwrap();
        for item in &safety_items {
            writeln!(doc, "{}/// {}", indent, item).unwrap();
        }
        writeln!(doc, "{}///", indent).unwrap();
    }

    // Additional notes
    for note in &context.notes {
        writeln!(doc, "{}/// **Note**: {}", indent, note).unwrap();
    }

    // Add parameter documentation from parameters HashMap
    if !context.parameters.is_empty() {
        writeln!(doc, "{}///", indent).unwrap();
        writeln!(doc, "{}/// # Parameters", indent).unwrap();
        writeln!(doc, "{}///", indent).unwrap();
        for (param_name, param_doc) in &context.parameters {
            writeln!(doc, "{}/// - `{}`: {}", indent, param_name, param_doc).unwrap();

            // Add numeric constraints for this parameter
            if let Some(constraints) = &context.numeric_constraints {
                for constraint in &constraints.constraints {
                    if constraint.parameter_name.as_deref() == Some(param_name.as_str()) {
                        if let (Some(min), Some(max)) = (constraint.min_value, constraint.max_value)
                        {
                            writeln!(doc, "{}///   - Valid range: [{}, {}]", indent, min, max)
                                .unwrap();
                        } else if let Some(min) = constraint.min_value {
                            writeln!(doc, "{}///   - Minimum: {}", indent, min).unwrap();
                        } else if let Some(max) = constraint.max_value {
                            writeln!(doc, "{}///   - Maximum: {}", indent, max).unwrap();
                        }
                        if constraint.must_be_power_of_two {
                            writeln!(doc, "{}///   - Must be a power of two", indent).unwrap();
                        }
                        if let Some(align) = constraint.alignment_bytes {
                            writeln!(doc, "{}///   - Must be aligned to {} bytes", indent, align)
                                .unwrap();
                        }
                    }
                }
            }
        }
        writeln!(doc, "{}///", indent).unwrap();
    }

    // Add return value documentation
    if let Some(return_doc) = &context.return_doc {
        writeln!(doc, "{}///", indent).unwrap();
        writeln!(doc, "{}/// # Returns", indent).unwrap();
        writeln!(doc, "{}///", indent).unwrap();
        for line in return_doc.lines() {
            writeln!(doc, "{}/// {}", indent, line).unwrap();
        }
        writeln!(doc, "{}///", indent).unwrap();
    }

    // Add error documentation from error_semantics
    if let Some(error_info) = &context.error_semantics {
        if !error_info.errors.is_empty() {
            writeln!(doc, "{}///", indent).unwrap();
            writeln!(doc, "{}/// # Errors", indent).unwrap();
            writeln!(doc, "{}///", indent).unwrap();
            for (code, error_detail) in &error_info.errors {
                writeln!(
                    doc,
                    "{}/// - `{}`: {}",
                    indent, code, error_detail.description
                )
                .unwrap();
                if error_detail.is_fatal {
                    writeln!(doc, "{}///   (FATAL - cannot recover)", indent).unwrap();
                }
            }
            writeln!(doc, "{}///", indent).unwrap();
        }
    }

    // Add usage examples from test_cases
    if let Some(test_info) = &context.test_cases {
        if !test_info.examples.is_empty() {
            writeln!(doc, "{}///", indent).unwrap();
            writeln!(doc, "{}/// # Examples", indent).unwrap();
            writeln!(doc, "{}///", indent).unwrap();
            for example in &test_info.examples {
                writeln!(doc, "{}/// ```rust", indent).unwrap();
                for line in example.code_snippet.lines() {
                    writeln!(doc, "{}/// {}", indent, line).unwrap();
                }
                writeln!(doc, "{}/// ```", indent).unwrap();
                if let Some(desc) = &example.description {
                    writeln!(doc, "{}/// {}", indent, desc).unwrap();
                }
                writeln!(doc, "{}///", indent).unwrap();
            }
        }
    }

    // Add LLM-enhanced documentation if available
    if let Some(enhanced) = enhanced_docs {
        // Find function-specific docs
        if let Some(func_doc) = enhanced
            .function_docs
            .iter()
            .find(|d| d.name == context.name)
        {
            // Add examples (only if test_cases didn't provide them)
            if !func_doc.examples.is_empty() && context.test_cases.is_none() {
                writeln!(doc, "{}///", indent).unwrap();
                writeln!(doc, "{}/// # Examples", indent).unwrap();
                writeln!(doc, "{}///", indent).unwrap();
                for example in &func_doc.examples {
                    writeln!(doc, "{}/// ```rust", indent).unwrap();
                    for line in example.lines() {
                        writeln!(doc, "{}/// {}", indent, line).unwrap();
                    }
                    writeln!(doc, "{}/// ```", indent).unwrap();
                    writeln!(doc, "{}///", indent).unwrap();
                }
            }

            // Add pitfalls
            if !func_doc.pitfalls.is_empty() {
                writeln!(doc, "{}/// # Common Pitfalls", indent).unwrap();
                writeln!(doc, "{}///", indent).unwrap();
                for pitfall in &func_doc.pitfalls {
                    writeln!(doc, "{}/// - {}", indent, pitfall).unwrap();
                }
                writeln!(doc, "{}///", indent).unwrap();
            }

            // Add best practices
            if !func_doc.best_practices.is_empty() {
                writeln!(doc, "{}/// # Best Practices", indent).unwrap();
                writeln!(doc, "{}///", indent).unwrap();
                for practice in &func_doc.best_practices {
                    writeln!(doc, "{}/// - {}", indent, practice).unwrap();
                }
                writeln!(doc, "{}///", indent).unwrap();
            }
        }
    }

    // Add platform-specific information
    if let Some(platform_info) = &context.platform {
        if !platform_info.available_on.is_empty() {
            writeln!(doc, "{}///", indent).unwrap();
            writeln!(doc, "{}/// # Platform Support", indent).unwrap();
            writeln!(doc, "{}///", indent).unwrap();
            for platform in &platform_info.available_on {
                writeln!(doc, "{}/// - {:?}", indent, platform).unwrap();
            }
            if !platform_info.version_requirements.is_empty() {
                writeln!(
                    doc,
                    "{}/// Version requirements: {:?}",
                    indent, platform_info.version_requirements[0]
                )
                .unwrap();
            }
            writeln!(doc, "{}///", indent).unwrap();
        }
    }

    // Add callback safety documentation
    if let Some(callback_semantics) = &context.callback_info {
        if !callback_semantics.callbacks.is_empty() {
            writeln!(doc, "{}///", indent).unwrap();
            writeln!(doc, "{}/// # Callback Safety", indent).unwrap();
            writeln!(doc, "{}///", indent).unwrap();
            for (param_name, callback) in &callback_semantics.callbacks {
                writeln!(doc, "{}/// Callback parameter `{}`:", indent, param_name).unwrap();
                writeln!(doc, "{}/// - Lifetime: {:?}", indent, callback.lifetime).unwrap();
                writeln!(
                    doc,
                    "{}/// - Invocation: {:?}",
                    indent, callback.invocation_count
                )
                .unwrap();
                writeln!(
                    doc,
                    "{}/// - Thread safety: {:?}",
                    indent, callback.thread_safety
                )
                .unwrap();
                if !callback.notes.is_empty() {
                    for note in &callback.notes {
                        writeln!(doc, "{}/// - {}", indent, note).unwrap();
                    }
                }
            }
            writeln!(doc, "{}///", indent).unwrap();
        }
    }

    // Add resource limits
    if let Some(limits) = &context.resource_limits {
        if !limits.limits.is_empty() {
            writeln!(doc, "{}///", indent).unwrap();
            writeln!(doc, "{}/// # Resource Limits", indent).unwrap();
            writeln!(doc, "{}///", indent).unwrap();
            for (_key, limit_infos) in &limits.limits {
                for limit_info in limit_infos {
                    write!(doc, "{}/// - {}", indent, limit_info.description).unwrap();
                    if let Some(val) = limit_info.value {
                        writeln!(doc, " ({} {:?})", val, limit_info.unit).unwrap();
                    } else {
                        writeln!(doc).unwrap();
                    }
                }
            }
            writeln!(doc, "{}///", indent).unwrap();
        }
    }

    // Add deprecation warnings from version_history
    if !context.version_history.is_empty() {
        writeln!(doc, "{}///", indent).unwrap();
        for deprecation in &context.version_history {
            if let Some(reason) = &deprecation.reason {
                writeln!(doc, "{}/// ⚠️  **Deprecated**: {}", indent, reason).unwrap();
            } else {
                writeln!(doc, "{}/// ⚠️  **Deprecated**", indent).unwrap();
            }
            if let Some(replacement) = &deprecation.replacement {
                writeln!(doc, "{}/// Use `{}` instead", indent, replacement).unwrap();
            }
            writeln!(doc, "{}/// (since version {})", indent, deprecation.since).unwrap();
        }
        writeln!(doc, "{}///", indent).unwrap();
    }

    doc
}

fn collect_safety_items(context: &FunctionContext) -> Vec<String> {
    let mut items = Vec::new();

    // 1. Preconditions - null checks and UB warnings
    if let Some(precond) = &context.preconditions {
        for param in &precond.non_null_params {
            items.push(format!("- `{}` must not be null", param));
        }
        // Include ALL undefined behavior warnings, not just some
        for ub in &precond.undefined_behavior {
            items.push(format!("- **UB**: {}", ub));
        }
        // Add all preconditions (which may include non-zero requirements)
        for precond_item in &precond.preconditions {
            if let Some(param) = &precond_item.parameter {
                items.push(format!("- `{}`: {}", param, precond_item.description));
            } else {
                items.push(format!("- {}", precond_item.description));
            }
        }
    }

    // 2. Thread safety - critical for Rust's safety guarantees
    if let Some(thread_safety) = &context.thread_safety {
        use crate::analyzer::thread_safety::ThreadSafety;
        match thread_safety.safety {
            ThreadSafety::Safe => {
                items.push("- Thread-safe: can be called concurrently".to_string());
            }
            ThreadSafety::Unsafe => {
                items.push("- **Not thread-safe**: requires external synchronization".to_string());
            }
            ThreadSafety::Reentrant => {
                items.push("- Reentrant: safe for recursive calls".to_string());
            }
            ThreadSafety::RequiresSync => {
                items.push("- Requires synchronization (mutex, lock)".to_string());
            }
            ThreadSafety::PerThread => {
                items.push("- Per-thread instance required".to_string());
            }
            _ => {}
        }
    }

    // 3. Global state - initialization requirements
    if let Some(global_state) = &context.global_state
        && global_state.requires_init
    {
        if let Some(init_fn) = &global_state.init_function {
            items.push(format!(
                "- **Initialization required**: call `{}` first",
                init_fn
            ));
        } else {
            items.push("- **Initialization required** before use".to_string());
        }
    }

    // 4. API sequencing - call order requirements
    if let Some(api_seq) = &context.api_sequences {
        for prereq in &api_seq.prerequisites {
            items.push(format!("- Must call `{}` before this", prereq));
        }
        for exclusive in &api_seq.mutually_exclusive {
            items.push(format!(
                "- Cannot use with `{}` (mutually exclusive)",
                exclusive
            ));
        }
    }

    // 5. Anti-patterns - critical warnings
    if let Some(pitfalls) = &context.pitfalls {
        for pitfall in &pitfalls.pitfalls {
            use crate::analyzer::anti_patterns::Severity;
            let marker = match pitfall.severity {
                Severity::Critical | Severity::High => "⚠️",
                _ => "ℹ️",
            };
            items.push(format!("{} {}", marker, pitfall.title));
        }
    }

    // 6. Error information - if available
    if let Some(error_info) = &context.error_semantics
        && !error_info.errors.is_empty()
    {
        items.push("- Returns error codes (see documentation)".to_string());
        // Check for fatal errors
        if error_info.errors.values().any(|e| e.is_fatal) {
            items.push("- Some errors are FATAL and cannot be recovered".to_string());
        }
    }

    items
}
