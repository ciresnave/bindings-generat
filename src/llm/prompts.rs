/// Prompt templates for LLM interactions
///
/// Generate documentation prompt for a function
pub fn documentation_prompt(function_name: &str, signature: &str, context: &str) -> String {
    format!(
        r#"You are a Rust documentation expert. Generate clear, concise documentation for this FFI wrapper function.

Function name: {function_name}
Signature: {signature}
Context: {context}

Generate a documentation comment in Rust doc comment format (/// or //!) that includes:
1. A one-line summary
2. Description of what the function does
3. Parameter descriptions
4. Return value description
5. Any important notes about safety or usage
6. A simple usage example if applicable

Output ONLY the documentation comment, no additional explanation."#,
        function_name = function_name,
        signature = signature,
        context = context,
    )
}

/// Generate naming suggestion prompt
pub fn naming_prompt(c_name: &str, context: &str, item_type: &str) -> String {
    format!(
        r#"You are a Rust naming expert. Suggest an idiomatic Rust name for this C {item_type}.

C name: {c_name}
Context: {context}
Type: {item_type}

Suggest 3 alternative Rust names that:
1. Follow Rust naming conventions
2. Are clear and descriptive
3. Are more idiomatic than a direct translation

Output ONLY a JSON array of 3 strings, like: ["name1", "name2", "name3"]
No additional explanation."#,
        c_name = c_name,
        context = context,
        item_type = item_type,
    )
}

/// Generate usage example prompt
pub fn example_prompt(type_name: &str, methods: &[String], context: &str) -> String {
    let methods_list = methods.join(", ");
    format!(
        r#"You are a Rust example code expert. Generate a simple, realistic usage example for this wrapper type.

Type name: {type_name}
Available methods: {methods_list}
Context: {context}

Generate a complete, runnable Rust example showing:
1. Creating an instance
2. Using 1-2 of the most common methods
3. Proper error handling
4. Resource cleanup (if Drop is not automatic)

Output ONLY the example code block, no additional explanation.
Use realistic but simple values."#,
        type_name = type_name,
        methods_list = methods_list,
        context = context,
    )
}

/// Generate API design improvement prompt
pub fn api_improvement_prompt(current_design: &str, context: &str) -> String {
    format!(
        r#"You are a Rust API design expert. Analyze this generated wrapper API and suggest improvements.

Current API:
{current_design}

Context: {context}

Suggest improvements for:
1. More idiomatic naming
2. Better use of Rust's type system
3. Improved ergonomics
4. Better error types
5. More Rust-like patterns (builders, iterators, etc.)

Output your suggestions as a JSON object with:
{{
  "naming": ["suggestion1", "suggestion2"],
  "types": ["suggestion1", "suggestion2"],
  "ergonomics": ["suggestion1", "suggestion2"],
  "errors": ["suggestion1", "suggestion2"],
  "patterns": ["suggestion1", "suggestion2"]
}}

Keep suggestions concise and actionable. No additional explanation."#,
        current_design = current_design,
        context = context,
    )
}

/// Generate error message improvement prompt
pub fn error_message_prompt(error_code: &str, c_name: &str) -> String {
    format!(
        r#"You are a Rust error message expert. Generate a clear, helpful error message for this C error code.

C error code: {c_name}
Error variant name: {error_code}

Generate a user-friendly error message that:
1. Clearly describes what went wrong
2. Is concise (one sentence preferred)
3. Uses active voice
4. Avoids technical jargon when possible

Output ONLY the error message string, no quotes, no additional explanation."#,
        error_code = error_code,
        c_name = c_name,
    )
}

/// Generate type conversion suggestion prompt  
pub fn type_conversion_prompt(c_type: &str, current_rust_type: &str, context: &str) -> String {
    format!(
        r#"You are a Rust FFI expert. Suggest the most appropriate Rust type for this C type.

C type: {c_type}
Current Rust type: {current_rust_type}
Context: {context}

Consider:
1. Safety (avoid raw pointers if possible)
2. Ergonomics (use Rust idioms)
3. Performance (minimize copies)
4. Correctness (proper lifetimes)

Output ONLY the suggested Rust type as a string, no additional explanation.
If the current type is already good, output it unchanged."#,
        c_type = c_type,
        current_rust_type = current_rust_type,
        context = context,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_documentation_prompt() {
        let prompt = documentation_prompt(
            "simple_create",
            "fn simple_create() -> Result<Simple>",
            "Creates a new Simple instance",
        );
        assert!(prompt.contains("simple_create"));
        assert!(prompt.contains("Result<Simple>"));
    }

    #[test]
    fn test_naming_prompt() {
        let prompt = naming_prompt("simple_get_value", "Gets a value", "function");
        assert!(prompt.contains("simple_get_value"));
        assert!(prompt.contains("JSON array"));
    }

    #[test]
    fn test_example_prompt() {
        let methods = vec!["new".to_string(), "process".to_string()];
        let prompt = example_prompt("Simple", &methods, "Test library");
        assert!(prompt.contains("Simple"));
        assert!(prompt.contains("new"));
        assert!(prompt.contains("process"));
    }
}
