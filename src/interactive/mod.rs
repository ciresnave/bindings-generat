pub mod clarification;
pub mod crates;
pub mod decisions;
pub mod questions;
pub mod setup;

pub use clarification::{ClarificationResults, clarify_patterns};
pub use crates::{
    handle_existing_crates_workflow, prompt_select_existing_crate, show_cargo_instructions,
};
pub use setup::{is_first_run, prompt_first_run_if_needed, run_setup_wizard};
