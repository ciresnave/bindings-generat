pub mod ambiguity;
pub mod ast;
pub mod errors;
pub mod ownership;
pub mod patterns;
pub mod raii;

pub use errors::{ErrorPatterns, detect_error_patterns};
pub use raii::{RaiiPatterns, detect_raii_patterns};

use crate::ffi::FfiInfo;
use anyhow::Result;
use tracing::info;

/// Complete analysis result
#[derive(Debug, Clone)]
pub struct AnalysisResult {
    pub raii_patterns: RaiiPatterns,
    pub error_patterns: ErrorPatterns,
}

/// Perform complete pattern analysis on FFI
pub fn analyze_ffi(ffi_info: &FfiInfo) -> Result<AnalysisResult> {
    info!("Starting FFI pattern analysis");

    let raii_patterns = detect_raii_patterns(ffi_info);
    let error_patterns = detect_error_patterns(ffi_info);

    Ok(AnalysisResult {
        raii_patterns,
        error_patterns,
    })
}
