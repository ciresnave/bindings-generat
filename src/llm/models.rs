use dialoguer::Select;

/// Available LLM models for code generation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LlmModel {
    /// Fastest, smallest model (~1GB) - good for quick iterations
    #[default]
    Qwen25Coder15b,
    /// Balanced model (~4GB) - better quality, reasonable speed
    Qwen25Coder7b,
    /// Highest quality (~8GB) - best results, slower
    Qwen25Coder14b,
}

impl LlmModel {
    /// Get the model name for Ollama
    pub fn name(&self) -> &'static str {
        match self {
            Self::Qwen25Coder15b => "qwen2.5-coder:1.5b",
            Self::Qwen25Coder7b => "qwen2.5-coder:7b",
            Self::Qwen25Coder14b => "qwen2.5-coder:14b",
        }
    }

    /// Get the approximate download size
    pub fn size(&self) -> &'static str {
        match self {
            Self::Qwen25Coder15b => "~1GB",
            Self::Qwen25Coder7b => "~4GB",
            Self::Qwen25Coder14b => "~8GB",
        }
    }

    /// Get a description of the model's characteristics
    pub fn description(&self) -> &'static str {
        match self {
            Self::Qwen25Coder15b => "Fast, lightweight - good for quick iterations",
            Self::Qwen25Coder7b => "Balanced - better quality, reasonable speed",
            Self::Qwen25Coder14b => "Best quality - most accurate, slower",
        }
    }

    /// Get the display string for selection UI
    pub fn display_string(&self) -> String {
        format!("{} - {} ({})", self.name(), self.description(), self.size())
    }

    /// List all available models
    pub fn all() -> Vec<Self> {
        vec![
            Self::Qwen25Coder15b,
            Self::Qwen25Coder7b,
            Self::Qwen25Coder14b,
        ]
    }
}

/// Prompt user to select an LLM model
pub fn select_model() -> anyhow::Result<LlmModel> {
    let models = LlmModel::all();
    let display_strings: Vec<String> = models.iter().map(|m| m.display_string()).collect();

    println!();
    println!("🤖 Select LLM model for code enhancement:");
    println!();

    let selection = Select::new()
        .with_prompt("Model selection")
        .items(&display_strings)
        .default(0)
        .interact()?;

    Ok(models[selection])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_names() {
        assert_eq!(LlmModel::Qwen25Coder15b.name(), "qwen2.5-coder:1.5b");
        assert_eq!(LlmModel::Qwen25Coder7b.name(), "qwen2.5-coder:7b");
        assert_eq!(LlmModel::Qwen25Coder14b.name(), "qwen2.5-coder:14b");
    }

    #[test]
    fn test_model_sizes() {
        assert_eq!(LlmModel::Qwen25Coder15b.size(), "~1GB");
        assert_eq!(LlmModel::Qwen25Coder7b.size(), "~4GB");
        assert_eq!(LlmModel::Qwen25Coder14b.size(), "~8GB");
    }

    #[test]
    fn test_default_model() {
        let default = LlmModel::default();
        assert_eq!(default, LlmModel::Qwen25Coder15b);
    }

    #[test]
    fn test_all_models() {
        let models = LlmModel::all();
        assert_eq!(models.len(), 3);
        assert_eq!(models[0], LlmModel::Qwen25Coder15b);
        assert_eq!(models[1], LlmModel::Qwen25Coder7b);
        assert_eq!(models[2], LlmModel::Qwen25Coder14b);
    }

    #[test]
    fn test_display_string() {
        let model = LlmModel::Qwen25Coder15b;
        let display = model.display_string();
        assert!(display.contains("qwen2.5-coder:1.5b"));
        assert!(display.contains("~1GB"));
    }
}
