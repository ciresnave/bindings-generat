// src/assertions/mod.rs

//! Debug assertion generation for contract validation
//!
//! This module generates `debug_assert!` statements that validate function
//! contracts at runtime in debug builds. All assertions are compiled out
//! in release builds, providing zero-cost safety verification during development.

pub mod contract_parser;
pub mod generator;

pub use contract_parser::ContractAnalyzer;
pub use generator::AssertionGenerator;

use crate::ffi::FfiInfo;
use anyhow::Result;

/// Generates debug assertions for all functions in the FFI info
pub fn generate_assertions(ffi_info: &FfiInfo) -> Result<Vec<FunctionAssertions>> {
    let analyzer = ContractAnalyzer::new();
    let generator = AssertionGenerator::new();

    let mut all_assertions = Vec::new();

    for function in &ffi_info.functions {
        let contracts = analyzer.analyze_function(function)?;
        let assertions = generator.generate_for_function(function, &contracts)?;
        all_assertions.push(assertions);
    }

    Ok(all_assertions)
}

/// Assertions generated for a single function
#[derive(Debug, Clone)]
pub struct FunctionAssertions {
    /// Function name
    pub function_name: String,

    /// Assertions to run before the FFI call
    pub preconditions: Vec<Assertion>,

    /// Assertions to run after the FFI call
    pub postconditions: Vec<Assertion>,

    /// Invariants to check (if applicable)
    pub invariants: Vec<Assertion>,
}

/// A single assertion
#[derive(Debug, Clone)]
pub struct Assertion {
    /// The condition to check (Rust expression)
    pub condition: String,

    /// Error message if assertion fails
    pub message: String,

    /// Category of assertion
    pub category: AssertionCategory,

    /// Confidence level (0.0 - 1.0)
    /// 1.0 = certain from explicit documentation
    /// 0.5 = inferred from patterns
    pub confidence: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AssertionCategory {
    /// Null pointer check
    NullPointer,

    /// Numeric range validation
    RangeCheck,

    /// State validation (initialized, not destroyed, etc.)
    StateCheck,

    /// Memory alignment requirement
    Alignment,

    /// Enum variant validation
    EnumValid,

    /// Buffer size validation
    BufferSize,

    /// Thread safety check
    ThreadSafety,

    /// General invariant
    Invariant,
}

impl AssertionCategory {
    /// Priority for this category (higher = more important)
    pub fn priority(self) -> u8 {
        match self {
            AssertionCategory::NullPointer => 10, // Critical
            AssertionCategory::StateCheck => 9,   // Very important
            AssertionCategory::BufferSize => 8,   // Important
            AssertionCategory::RangeCheck => 7,
            AssertionCategory::Alignment => 6,
            AssertionCategory::EnumValid => 5,
            AssertionCategory::ThreadSafety => 4,
            AssertionCategory::Invariant => 3,
        }
    }
}
