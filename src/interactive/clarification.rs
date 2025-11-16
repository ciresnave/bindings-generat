use anyhow::Result;
use dialoguer::{Confirm, Select};
use tracing::info;

use crate::analyzer::raii::{LifecyclePair, RaiiPatterns};

/// Threshold for low-confidence lifecycle pairs that need confirmation
const LOW_CONFIDENCE_THRESHOLD: f32 = 0.7;

/// Interactive clarification results
#[derive(Debug, Clone)]
pub struct ClarificationResults {
    pub confirmed_pairs: Vec<LifecyclePair>,
    pub rejected_pairs: Vec<String>, // Pairs rejected by user (by create_fn name)
}

impl Default for ClarificationResults {
    fn default() -> Self {
        Self::new()
    }
}

impl ClarificationResults {
    pub fn new() -> Self {
        Self {
            confirmed_pairs: Vec::new(),
            rejected_pairs: Vec::new(),
        }
    }
}

/// Request user confirmation for ambiguous RAII patterns
pub fn clarify_patterns(patterns: &RaiiPatterns) -> Result<ClarificationResults> {
    let mut results = ClarificationResults::new();

    // Find low-confidence pairs that need confirmation
    let low_confidence_pairs: Vec<_> = patterns
        .lifecycle_pairs
        .iter()
        .filter(|pair| pair.confidence < LOW_CONFIDENCE_THRESHOLD)
        .collect();

    if low_confidence_pairs.is_empty() {
        info!("All lifecycle pairs have high confidence, no clarification needed");
        results.confirmed_pairs = patterns.lifecycle_pairs.clone();
        return Ok(results);
    }

    println!("\n⚠️  Some lifecycle pairs need confirmation:");
    println!();

    for pair in low_confidence_pairs {
        let should_confirm = Confirm::new()
            .with_prompt(format!(
                "Is '{}' paired with '{}' for handle '{}'? (confidence: {:.0}%)",
                pair.create_fn,
                pair.destroy_fn,
                pair.handle_type,
                pair.confidence * 100.0
            ))
            .default(pair.confidence > 0.5)
            .interact()?;

        if should_confirm {
            results.confirmed_pairs.push(pair.clone());
        } else {
            results.rejected_pairs.push(pair.create_fn.clone());
        }
    }

    // Add all high-confidence pairs automatically
    for pair in &patterns.lifecycle_pairs {
        if pair.confidence >= LOW_CONFIDENCE_THRESHOLD {
            results.confirmed_pairs.push(pair.clone());
        }
    }

    println!();
    Ok(results)
}

/// Ask user to select from multiple possible destroy functions for a create function
pub fn select_destroy_function(
    create_fn: &str,
    candidates: &[String],
    handle_type: &str,
) -> Result<Option<String>> {
    if candidates.is_empty() {
        return Ok(None);
    }

    if candidates.len() == 1 {
        return Ok(Some(candidates[0].clone()));
    }

    println!();
    let selection = Select::new()
        .with_prompt(format!(
            "Multiple possible destroy functions for '{}' (handle: '{}'). Which is correct?",
            create_fn, handle_type
        ))
        .items(candidates)
        .default(0)
        .interact_opt()?;

    Ok(selection.map(|idx| candidates[idx].clone()))
}

/// Ask user to confirm which functions are truly create/destroy pairs
pub fn select_lifecycle_pairs(
    handle_type: &str,
    create_candidates: &[String],
    destroy_candidates: &[String],
) -> Result<Vec<(String, String)>> {
    println!();
    println!(
        "🔍 Ambiguous lifecycle functions detected for handle '{}'",
        handle_type
    );
    println!();

    if create_candidates.is_empty() || destroy_candidates.is_empty() {
        return Ok(Vec::new());
    }

    let mut pairs = Vec::new();

    // For each create function, ask which destroy function it pairs with
    for create_fn in create_candidates {
        let options: Vec<String> = destroy_candidates
            .iter()
            .map(|d| d.to_string())
            .chain(std::iter::once("(None - skip this function)".to_string()))
            .collect();

        let selection = Select::new()
            .with_prompt(format!(
                "Which function destroys resources created by '{}'?",
                create_fn
            ))
            .items(&options)
            .default(0)
            .interact()?;

        if selection < destroy_candidates.len() {
            pairs.push((create_fn.clone(), destroy_candidates[selection].clone()));
        }
    }

    Ok(pairs)
}

/// Check if user wants to manually add a lifecycle pair not detected automatically
pub fn prompt_manual_pair() -> Result<Option<(String, String, String)>> {
    println!();
    let add_manual = Confirm::new()
        .with_prompt("Would you like to manually specify a lifecycle pair that wasn't detected?")
        .default(false)
        .interact()?;

    if !add_manual {
        return Ok(None);
    }

    let handle_type: String = dialoguer::Input::new()
        .with_prompt("Handle type name (e.g., 'MyHandle')")
        .interact_text()?;

    let create_fn: String = dialoguer::Input::new()
        .with_prompt("Create function name")
        .interact_text()?;

    let destroy_fn: String = dialoguer::Input::new()
        .with_prompt("Destroy function name")
        .interact_text()?;

    Ok(Some((handle_type, create_fn, destroy_fn)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clarification_results_creation() {
        let results = ClarificationResults::new();
        assert!(results.confirmed_pairs.is_empty());
        assert!(results.rejected_pairs.is_empty());
    }

    const _: () = {
        assert!(LOW_CONFIDENCE_THRESHOLD > 0.0);
        assert!(LOW_CONFIDENCE_THRESHOLD <= 1.0);
    };
}
