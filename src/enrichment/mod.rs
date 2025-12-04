//! Context enrichment from documentation, examples, and usage patterns.
//!
//! This module discovers and analyzes additional context beyond just C headers:
//! - Documentation files (Markdown, RST, HTML, PDF, man pages)
//! - Example code (C, C++, Python samples)
//! - Test files (unit tests, integration tests)
//! - Real-world usage patterns from code repositories
//!
//! ## Modules
//!
//! - [`doc_finder`] - Smart directory discovery for docs, examples, tests
//! - [`doc_parser`] - Documentation parsing (Doxygen, Sphinx/RST)
//! - [`code_search`] - Multi-platform code search for usage patterns
//! - [`types`] - Data structures for enrichment results
//!
//! ## Philosophy
//!
//! Bindgen gives us correct FFI syntax, but we need semantic understanding
//! for quality wrapper generation. Context enrichment bridges this gap by:
//!
//! 1. Finding comprehensive documentation beyond header comments
//! 2. Extracting usage patterns from real example code
//! 3. Validating pattern detection with actual usage
//! 4. Improving parameter classification from examples
//! 5. Generating better tests from discovered examples

pub mod code_search;
pub mod context;
pub mod doc_finder;
pub mod doc_parser;
pub mod header_parser;
pub mod types;

pub use context::{ContextSource, EnhancedContext, FunctionContext};
pub use doc_finder::{discover_library_files, find_library_root};
pub use header_parser::{FunctionComment, HeaderCommentParser, ParamDirection, ParamDoc};
pub use types::{
    DocCategory, DocFormat, DocumentFile, ExampleFile, Language, LibraryFiles, TestFile,
};
