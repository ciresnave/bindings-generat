pub mod doc_sanitizer;
pub mod naming;
pub mod progress;

pub use doc_sanitizer::sanitize_doc;
pub use naming::{detect_library_prefix, to_idiomatic_rust_name};

// Utility types and functions will be implemented here
