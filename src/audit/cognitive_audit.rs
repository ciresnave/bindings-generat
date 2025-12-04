// Cognitive load audit system for generated bindings
// Analyzes API complexity and suggests simplifications

use crate::ffi::parser::{FfiFunction, FfiInfo, FfiParam};
use std::collections::HashMap;

/// Complexity level assessment
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ComplexityLevel {
    /// Very simple, easy to understand
    VeryLow,
    /// Simple, straightforward
    Low,
    /// Moderate complexity
    Medium,
    /// High complexity, may be confusing
    High,
    /// Very high complexity, difficult to use
    VeryHigh,
}

impl ComplexityLevel {
    pub fn as_str(&self) -> &'static str {
        match self {
            ComplexityLevel::VeryLow => "VERY LOW",
            ComplexityLevel::Low => "LOW",
            ComplexityLevel::Medium => "MEDIUM",
            ComplexityLevel::High => "HIGH",
            ComplexityLevel::VeryHigh => "VERY HIGH",
        }
    }

    pub fn emoji(&self) -> &'static str {
        match self {
            ComplexityLevel::VeryLow => "🟢",
            ComplexityLevel::Low => "🟢",
            ComplexityLevel::Medium => "🟡",
            ComplexityLevel::High => "🟠",
            ComplexityLevel::VeryHigh => "🔴",
        }
    }
}

/// Cognitive load issue
#[derive(Debug, Clone)]
pub struct CognitiveIssue {
    /// Complexity level
    pub complexity: ComplexityLevel,
    /// Function or API element
    pub element_name: String,
    /// Description of the issue
    pub description: String,
    /// Suggested simplification
    pub suggestion: String,
    /// Impact on usability
    pub impact: String,
}

/// Cognitive load metrics for a function
#[derive(Debug, Clone)]
pub struct FunctionMetrics {
    /// Function name
    pub name: String,
    /// Number of parameters
    pub param_count: usize,
    /// Cyclomatic complexity estimate
    pub cyclomatic_complexity: u32,
    /// Number of pointer parameters
    pub pointer_count: usize,
    /// Number of preconditions
    pub precondition_count: usize,
    /// Overall complexity score (0-100, higher is more complex)
    pub complexity_score: u32,
}

/// Cognitive load audit report
#[derive(Debug, Clone)]
pub struct CognitiveAuditReport {
    /// Total functions analyzed
    pub total_functions: usize,
    /// Issues found
    pub issues: Vec<CognitiveIssue>,
    /// Function metrics
    pub function_metrics: Vec<FunctionMetrics>,
    /// Overall cognitive load score (0-100, higher is easier to use)
    pub usability_score: u32,
    /// Recommendations summary
    pub recommendations: Vec<String>,
}

impl CognitiveAuditReport {
    /// Generate markdown report
    pub fn to_markdown(&self) -> String {
        let mut md = String::new();
        
        md.push_str("# Cognitive Load Audit Report\n\n");
        md.push_str(&format!("**Generated**: {}\n\n", chrono::Local::now().format("%Y-%m-%d %H:%M:%S")));
        
        // Executive summary
        md.push_str("## Executive Summary\n\n");
        md.push_str(&format!("- **Usability Score**: {}/100 {}\n", 
            self.usability_score,
            Self::score_emoji(self.usability_score)
        ));
        md.push_str(&format!("- **Functions Analyzed**: {}\n", self.total_functions));
        md.push_str(&format!("- **Issues Found**: {}\n\n", self.issues.len()));

        // Complexity distribution
        let mut complexity_counts = HashMap::new();
        for issue in &self.issues {
            *complexity_counts.entry(issue.complexity).or_insert(0) += 1;
        }

        if !complexity_counts.is_empty() {
            md.push_str("### Complexity Distribution\n\n");
            md.push_str("| Complexity Level | Count |\n");
            md.push_str("|------------------|-------|\n");
            
            for level in [ComplexityLevel::VeryHigh, ComplexityLevel::High, ComplexityLevel::Medium, ComplexityLevel::Low, ComplexityLevel::VeryLow] {
                if let Some(&count) = complexity_counts.get(&level) {
                    md.push_str(&format!("| {} {} | {} |\n", level.emoji(), level.as_str(), count));
                }
            }
            md.push_str("\n");
        }

        // Top 10 most complex functions
        md.push_str("## Most Complex Functions\n\n");
        let mut sorted_metrics = self.function_metrics.clone();
        sorted_metrics.sort_by(|a, b| b.complexity_score.cmp(&a.complexity_score));
        
        md.push_str("| Function | Params | Pointers | Preconditions | Complexity |\n");
        md.push_str("|----------|--------|----------|---------------|------------|\n");
        
        for metric in sorted_metrics.iter().take(10) {
            md.push_str(&format!("| `{}` | {} | {} | {} | {} |\n",
                metric.name,
                metric.param_count,
                metric.pointer_count,
                metric.precondition_count,
                metric.complexity_score
            ));
        }
        md.push_str("\n");

        // Detailed issues
        if !self.issues.is_empty() {
            md.push_str("## Detailed Issues\n\n");
            
            for level in [ComplexityLevel::VeryHigh, ComplexityLevel::High, ComplexityLevel::Medium] {
                let issues: Vec<_> = self.issues.iter()
                    .filter(|i| i.complexity == level)
                    .collect();
                
                if !issues.is_empty() {
                    md.push_str(&format!("### {} {} Complexity\n\n", level.emoji(), level.as_str()));
                    
                    for (idx, issue) in issues.iter().enumerate() {
                        md.push_str(&format!("#### {}.{} `{}`\n\n", level.as_str(), idx + 1, issue.element_name));
                        md.push_str(&format!("**Issue**: {}\n\n", issue.description));
                        md.push_str(&format!("**Impact**: {}\n\n", issue.impact));
                        md.push_str(&format!("**Suggestion**: {}\n\n", issue.suggestion));
                    }
                }
            }
        }

        // Recommendations
        md.push_str("## Recommendations\n\n");
        if self.recommendations.is_empty() {
            md.push_str("No specific recommendations - API appears well-designed.\n\n");
        } else {
            for (idx, rec) in self.recommendations.iter().enumerate() {
                md.push_str(&format!("{}. {}\n", idx + 1, rec));
            }
            md.push_str("\n");
        }

        // Best practices
        md.push_str("## API Design Best Practices\n\n");
        md.push_str("1. **Parameter Count**: Keep functions under 4 parameters when possible\n");
        md.push_str("2. **Builder Pattern**: Consider builders for functions with many optional parameters\n");
        md.push_str("3. **Type Safety**: Use strong types instead of raw integers/pointers\n");
        md.push_str("4. **Documentation**: Provide clear examples for complex functions\n");
        md.push_str("5. **Sensible Defaults**: Offer preset configurations for common use cases\n");
        md.push_str("6. **Grouping**: Group related functions into modules or traits\n");
        md.push_str("7. **Naming**: Use descriptive, self-documenting names\n");
        md.push_str("8. **Error Messages**: Provide actionable error messages\n\n");

        md
    }

    fn score_emoji(score: u32) -> &'static str {
        match score {
            80..=100 => "🟢",
            60..=79 => "🟡",
            40..=59 => "🟠",
            _ => "🔴",
        }
    }
}

/// Cognitive load audit analyzer
pub struct CognitiveAudit;

impl CognitiveAudit {
    /// Perform comprehensive cognitive load audit
    pub fn analyze(ffi_info: &FfiInfo) -> CognitiveAuditReport {
        let mut issues = Vec::new();
        let mut function_metrics = Vec::new();

        // Analyze each function
        for func in &ffi_info.functions {
            let metrics = Self::calculate_function_metrics(func);
            function_metrics.push(metrics.clone());
            
            issues.extend(Self::analyze_function(func, &metrics));
        }

        // Generate recommendations
        let recommendations = Self::generate_recommendations(&issues, &function_metrics);

        // Calculate usability score
        let usability_score = Self::calculate_usability_score(&function_metrics, &issues);

        CognitiveAuditReport {
            total_functions: ffi_info.functions.len(),
            issues,
            function_metrics,
            usability_score,
            recommendations,
        }
    }

    /// Calculate metrics for a function
    fn calculate_function_metrics(func: &FfiFunction) -> FunctionMetrics {
        let param_count = func.params.len();
        let pointer_count = func.params.iter()
            .filter(|p| p.is_pointer)
            .count();
        // Estimate preconditions from pointer parameters
        let precondition_count = pointer_count;

        // Estimate cyclomatic complexity
        let cyclomatic_complexity = 1 + precondition_count as u32;

        // Calculate complexity score (0-100)
        let mut score = 0;
        score += param_count as u32 * 5;  // Each param adds 5 points
        score += pointer_count as u32 * 10; // Pointers add more complexity
        score += precondition_count as u32 * 3; // Preconditions add complexity

        FunctionMetrics {
            name: func.name.clone(),
            param_count,
            cyclomatic_complexity,
            pointer_count,
            precondition_count,
            complexity_score: score.min(100),
        }
    }

    /// Analyze a function for cognitive load issues
    fn analyze_function(func: &FfiFunction, metrics: &FunctionMetrics) -> Vec<CognitiveIssue> {
        let mut issues = Vec::new();

        // Check parameter count
        if metrics.param_count > 7 {
            issues.push(CognitiveIssue {
                complexity: ComplexityLevel::VeryHigh,
                element_name: func.name.clone(),
                description: format!("Function has {} parameters - exceeds recommended maximum of 7", metrics.param_count),
                suggestion: "Consider using a configuration struct or builder pattern".to_string(),
                impact: "Users may struggle to remember parameter order and meaning".to_string(),
            });
        } else if metrics.param_count > 4 {
            issues.push(CognitiveIssue {
                complexity: ComplexityLevel::High,
                element_name: func.name.clone(),
                description: format!("Function has {} parameters - consider simplification", metrics.param_count),
                suggestion: "Group related parameters into a struct or use builder pattern".to_string(),
                impact: "May be difficult to use without IDE assistance".to_string(),
            });
        }

        // Check pointer density
        if metrics.pointer_count > 3 {
            issues.push(CognitiveIssue {
                complexity: ComplexityLevel::High,
                element_name: func.name.clone(),
                description: format!("Function has {} pointer parameters", metrics.pointer_count),
                suggestion: "Consider using Rust slice types or safe wrapper structs".to_string(),
                impact: "Users must carefully manage lifetimes and null safety".to_string(),
            });
        }

        // Check for complex parameter types
        for param in &func.params {
            if Self::is_complex_type(&param.ty) {
                issues.push(CognitiveIssue {
                    complexity: ComplexityLevel::Medium,
                    element_name: format!("{}::{}", func.name, param.name),
                    description: format!("Parameter '{}' has complex type: {}", param.name, param.ty),
                    suggestion: "Add type alias or wrapper type with clear documentation".to_string(),
                    impact: "Users may not understand how to construct this parameter".to_string(),
                });
            }
        }

        // Check precondition complexity
        if metrics.precondition_count > 5 {
            issues.push(CognitiveIssue {
                complexity: ComplexityLevel::High,
                element_name: func.name.clone(),
                description: format!("Function has {} preconditions - high validation burden", metrics.precondition_count),
                suggestion: "Consider splitting function or providing validated wrapper".to_string(),
                impact: "Users must remember many constraints, increasing error risk".to_string(),
            });
        }

        // Check naming clarity
        if !Self::is_clear_name(&func.name) {
            issues.push(CognitiveIssue {
                complexity: ComplexityLevel::Medium,
                element_name: func.name.clone(),
                description: "Function name may be unclear or too abbreviated".to_string(),
                suggestion: "Use descriptive, self-documenting names".to_string(),
                impact: "Users may not understand function purpose without docs".to_string(),
            });
        }

        issues
    }

    /// Check if type is complex
    fn is_complex_type(type_name: &str) -> bool {
        // Multiple indirection levels
        let pointer_count = type_name.matches('*').count();
        if pointer_count > 1 {
            return true;
        }

        // Function pointers
        if type_name.contains("fn(") || type_name.contains("->") {
            return true;
        }

        // Generic types with multiple parameters
        if type_name.matches('<').count() > 1 {
            return true;
        }

        false
    }

    /// Check if name is clear
    fn is_clear_name(name: &str) -> bool {
        // Should have reasonable length
        if name.len() < 3 || name.len() > 50 {
            return false;
        }

        // Should not be mostly abbreviations
        let uppercase_count = name.chars().filter(|c| c.is_uppercase()).count();
        let total_alpha = name.chars().filter(|c| c.is_alphabetic()).count();
        
        if total_alpha > 0 && uppercase_count as f64 / total_alpha as f64 > 0.8 {
            return false; // Mostly uppercase (abbreviations)
        }

        true
    }

    /// Generate recommendations based on issues
    fn generate_recommendations(issues: &[CognitiveIssue], metrics: &[FunctionMetrics]) -> Vec<String> {
        let mut recommendations = Vec::new();

        // Check for common patterns
        let high_param_count = issues.iter()
            .filter(|i| i.description.contains("parameters"))
            .count();
        
        if high_param_count > 3 {
            recommendations.push(
                "Consider implementing builder pattern for complex functions".to_string()
            );
        }

        let high_pointer_usage = issues.iter()
            .filter(|i| i.description.contains("pointer"))
            .count();
        
        if high_pointer_usage > 5 {
            recommendations.push(
                "Reduce pointer usage by using Rust slice types and references".to_string()
            );
        }

        // Check average complexity
        let avg_complexity: f64 = metrics.iter()
            .map(|m| m.complexity_score as f64)
            .sum::<f64>() / metrics.len().max(1) as f64;
        
        if avg_complexity > 40.0 {
            recommendations.push(
                "Overall API complexity is high - consider providing facade/simplified interface".to_string()
            );
        }

        // Check for very complex functions
        let very_complex = metrics.iter()
            .filter(|m| m.complexity_score > 70)
            .count();
        
        if very_complex > 0 {
            recommendations.push(
                format!("Review {} highly complex function(s) for simplification opportunities", very_complex)
            );
        }

        recommendations
    }

    /// Calculate overall usability score
    fn calculate_usability_score(metrics: &[FunctionMetrics], issues: &[CognitiveIssue]) -> u32 {
        if metrics.is_empty() {
            return 100;
        }

        // Start with 100 and deduct points for issues
        let mut score = 100i32;

        for issue in issues {
            score -= match issue.complexity {
                ComplexityLevel::VeryHigh => 10,
                ComplexityLevel::High => 5,
                ComplexityLevel::Medium => 2,
                ComplexityLevel::Low => 1,
                ComplexityLevel::VeryLow => 0,
            };
        }

        // Also factor in average complexity
        let avg_complexity: f64 = metrics.iter()
            .map(|m| m.complexity_score as f64)
            .sum::<f64>() / metrics.len() as f64;
        
        score -= (avg_complexity / 2.0) as i32;

        score.max(0).min(100) as u32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_complexity_level_ordering() {
        assert!(ComplexityLevel::VeryHigh > ComplexityLevel::High);
        assert!(ComplexityLevel::High > ComplexityLevel::Medium);
        assert!(ComplexityLevel::Medium > ComplexityLevel::Low);
        assert!(ComplexityLevel::Low > ComplexityLevel::VeryLow);
    }

    #[test]
    fn test_calculate_function_metrics() {
        let func = FfiFunction {
            name: "test_func".to_string(),
            return_type: "int".to_string(),
            params: vec![
                FfiParam {
                    name: "ptr1".to_string(),
                    ty: "*const i32".to_string(),
                    is_pointer: true,
                    is_mut: false,
                },
                FfiParam {
                    name: "ptr2".to_string(),
                    ty: "*mut u8".to_string(),
                    is_pointer: true,
                    is_mut: true,
                },
                FfiParam {
                    name: "value".to_string(),
                    ty: "i32".to_string(),
                    is_pointer: false,
                    is_mut: false,
                },
            ],
            docs: None,
        };

        let metrics = CognitiveAudit::calculate_function_metrics(&func);
        
        assert_eq!(metrics.param_count, 3);
        assert_eq!(metrics.pointer_count, 2);
        assert!(metrics.complexity_score > 0);
    }

    #[test]
    fn test_is_complex_type() {
        assert!(CognitiveAudit::is_complex_type("**mut i32"));
        assert!(CognitiveAudit::is_complex_type("fn(*const i32) -> i32"));
        assert!(!CognitiveAudit::is_complex_type("*const i32"));
        assert!(!CognitiveAudit::is_complex_type("i32"));
    }

    #[test]
    fn test_is_clear_name() {
        assert!(CognitiveAudit::is_clear_name("create_context"));
        assert!(CognitiveAudit::is_clear_name("cudnnCreate"));
        assert!(!CognitiveAudit::is_clear_name("cc")); // Too short
        assert!(!CognitiveAudit::is_clear_name("CNN")); // All caps
    }

    #[test]
    fn test_analyze_high_param_count() {
        let func = FfiFunction {
            name: "test_func".to_string(),
            return_type: "void".to_string(),
            params: vec![
                FfiParam { name: "p1".to_string(), ty: "i32".to_string(), is_pointer: false, is_mut: false },
                FfiParam { name: "p2".to_string(), ty: "i32".to_string(), is_pointer: false, is_mut: false },
                FfiParam { name: "p3".to_string(), ty: "i32".to_string(), is_pointer: false, is_mut: false },
                FfiParam { name: "p4".to_string(), ty: "i32".to_string(), is_pointer: false, is_mut: false },
                FfiParam { name: "p5".to_string(), ty: "i32".to_string(), is_pointer: false, is_mut: false },
                FfiParam { name: "p6".to_string(), ty: "i32".to_string(), is_pointer: false, is_mut: false },
            ],
            docs: None,
        };

        let metrics = CognitiveAudit::calculate_function_metrics(&func);
        let issues = CognitiveAudit::analyze_function(&func, &metrics);
        
        // Should flag high parameter count
        assert!(!issues.is_empty());
        assert!(issues.iter().any(|i| i.description.contains("parameters")));
    }

    #[test]
    fn test_calculate_usability_score_perfect() {
        let metrics = vec![FunctionMetrics {
            name: "simple".to_string(),
            param_count: 2,
            cyclomatic_complexity: 1,
            pointer_count: 0,
            precondition_count: 0,
            complexity_score: 10,
        }];

        let score = CognitiveAudit::calculate_usability_score(&metrics, &[]);
        assert!(score > 90); // Should be high for simple API
    }
}
