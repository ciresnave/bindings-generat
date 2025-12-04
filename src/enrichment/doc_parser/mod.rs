//! Documentation parsing and extraction.
//!
//! This module provides parsers for structured documentation formats
//! to extract API documentation, parameter descriptions, and examples.

pub mod doxygen;
pub mod restructured_text;
pub mod types;

pub use types::{ParsedDoc, ParamDoc, FunctionDoc, DocParser};
