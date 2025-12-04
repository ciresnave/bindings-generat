// Safety audit system for generated bindings
// Analyzes unsafe operations and provides risk assessments

use crate::ffi::parser::{FfiFunction, FfiInfo, FfiParam};
use std::collections::HashMap;

/// Risk level for safety issues
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum RiskLevel {
    /// No safety concerns
    Safe,
    /// Minor safety concern, unlikely to cause issues
    Low,
    /// Moderate safety concern, could cause issues with incorrect usage
    Medium,
    /// High safety concern, likely to cause issues
    High,
    /// Critical safety concern, almost certain to cause issues
    Critical,
}

impl RiskLevel {
    pub fn as_str(&self) -> &'static str {
        match self {
            RiskLevel::Safe => "SAFE",
            RiskLevel::Low => "LOW",
            RiskLevel::Medium => "MEDIUM",
            RiskLevel::High => "HIGH",
            RiskLevel::Critical => "CRITICAL",
        }
    }

    pub fn emoji(&self) -> &'static str {
        match self {
            RiskLevel::Safe => "✅",
            RiskLevel::Low => "🟢",
            RiskLevel::Medium => "🟡",
            RiskLevel::High => "🟠",
            RiskLevel::Critical => "🔴",
        }
    }
}

/// Safety issue identified during audit
#[derive(Debug, Clone)]
pub struct SafetyIssue {
    /// Risk level of this issue
    pub risk_level: RiskLevel,
    /// Function name where issue was found
    pub function_name: String,
    /// Description of the safety issue
    pub description: String,
    /// Suggested mitigation
    pub mitigation: String,
    /// Line number or location (if applicable)
    pub location: Option<String>,
}

/// Comprehensive safety audit report
#[derive(Debug, Clone)]
pub struct SafetyAuditReport {
    /// Total functions analyzed
    pub total_functions: usize,
    /// Total unsafe operations
    pub unsafe_operations: usize,
    /// Issues found, grouped by risk level
    pub issues: Vec<SafetyIssue>,
    /// Overall risk assessment
    pub overall_risk: RiskLevel,
    /// Mitigation summary
    pub mitigation_summary: String,
}

impl SafetyAuditReport {
    /// Generate markdown report
    pub fn to_markdown(&self) -> String {
        let mut md = String::new();
        
        md.push_str("# Safety Audit Report\n\n");
        md.push_str(&format!("**Generated**: {}\n\n", chrono::Local::now().format("%Y-%m-%d %H:%M:%S")));
        
        // Executive summary
        md.push_str("## Executive Summary\n\n");
        md.push_str(&format!("- **Overall Risk Level**: {} {}\n", self.overall_risk.emoji(), self.overall_risk.as_str()));
        md.push_str(&format!("- **Functions Analyzed**: {}\n", self.total_functions));
        md.push_str(&format!("- **Unsafe Operations**: {}\n", self.unsafe_operations));
        md.push_str(&format!("- **Issues Found**: {}\n\n", self.issues.len()));

        // Risk distribution
        let mut risk_counts = HashMap::new();
        for issue in &self.issues {
            *risk_counts.entry(issue.risk_level).or_insert(0) += 1;
        }

        md.push_str("### Risk Distribution\n\n");
        md.push_str("| Risk Level | Count |\n");
        md.push_str("|------------|-------|\n");
        for risk in [RiskLevel::Critical, RiskLevel::High, RiskLevel::Medium, RiskLevel::Low, RiskLevel::Safe] {
            if let Some(&count) = risk_counts.get(&risk) {
                md.push_str(&format!("| {} {} | {} |\n", risk.emoji(), risk.as_str(), count));
            }
        }
        md.push_str("\n");

        // Detailed issues
        if !self.issues.is_empty() {
            md.push_str("## Detailed Issues\n\n");
            
            // Group by risk level
            for risk in [RiskLevel::Critical, RiskLevel::High, RiskLevel::Medium, RiskLevel::Low] {
                let issues_at_level: Vec<_> = self.issues.iter()
                    .filter(|i| i.risk_level == risk)
                    .collect();
                
                if !issues_at_level.is_empty() {
                    md.push_str(&format!("### {} {} Issues\n\n", risk.emoji(), risk.as_str()));
                    
                    for (idx, issue) in issues_at_level.iter().enumerate() {
                        md.push_str(&format!("#### {}.{} `{}`\n\n", risk.as_str(), idx + 1, issue.function_name));
                        md.push_str(&format!("**Issue**: {}\n\n", issue.description));
                        md.push_str(&format!("**Mitigation**: {}\n\n", issue.mitigation));
                        if let Some(ref location) = issue.location {
                            md.push_str(&format!("**Location**: {}\n\n", location));
                        }
                    }
                }
            }
        }

        // Mitigation summary
        md.push_str("## Mitigation Summary\n\n");
        md.push_str(&self.mitigation_summary);
        md.push_str("\n\n");

        // Recommendations
        md.push_str("## Recommendations\n\n");
        md.push_str("1. **Enable Strict Mode**: Use `features = [\"strict\"]` for critical systems\n");
        md.push_str("2. **Add Property Tests**: Verify invariants with property-based testing\n");
        md.push_str("3. **Review High-Risk Functions**: Manually review functions flagged as HIGH or CRITICAL\n");
        md.push_str("4. **Document Assumptions**: Add SAFETY comments for all unsafe operations\n");
        md.push_str("5. **Enable Leak Detection**: Use `features = [\"leak-detector\"]` during testing\n\n");

        md
    }

    /// Count issues by risk level
    pub fn count_by_risk(&self, level: RiskLevel) -> usize {
        self.issues.iter().filter(|i| i.risk_level == level).count()
    }
}

/// Safety audit analyzer
pub struct SafetyAudit;

impl SafetyAudit {
    /// Perform comprehensive safety audit
    pub fn analyze(ffi_info: &FfiInfo) -> SafetyAuditReport {
        let mut issues = Vec::new();

        // Analyze each function
        for func in &ffi_info.functions {
            issues.extend(Self::analyze_function(func));
        }

        // Determine overall risk
        let overall_risk = Self::calculate_overall_risk(&issues);

        // Generate mitigation summary
        let mitigation_summary = Self::generate_mitigation_summary(&issues);

        SafetyAuditReport {
            total_functions: ffi_info.functions.len(),
            unsafe_operations: ffi_info.functions.len(), // Each FFI call is unsafe
            issues,
            overall_risk,
            mitigation_summary,
        }
    }

    /// Analyze a single function for safety issues
    fn analyze_function(func: &FfiFunction) -> Vec<SafetyIssue> {
        let mut issues = Vec::new();

        // Check for pointer parameters
        for param in &func.params {
            if param.is_pointer {
                // All pointer parameters are potentially unsafe
                issues.push(SafetyIssue {
                    risk_level: RiskLevel::Medium,
                    function_name: func.name.clone(),
                    description: format!(
                        "Pointer parameter '{}' requires null safety checks",
                        param.name
                    ),
                    mitigation: "Add null pointer validation in generated wrapper".to_string(),
                    location: Some(format!("parameter '{}'", param.name)),
                });
            }
        }

        // Check for raw pointer return types
        if Self::is_pointer_type(&func.return_type) {
            issues.push(SafetyIssue {
                risk_level: RiskLevel::Medium,
                function_name: func.name.clone(),
                description: "Returns raw pointer - caller must handle null checks".to_string(),
                mitigation: "Wrap return value in Option<NonNull<T>> or similar".to_string(),
                location: Some("return type".to_string()),
            });
        }

        // Check for mutable pointers
        for param in &func.params {
            if param.is_mut && param.is_pointer && !Self::is_opaque_pointer(&param.ty) {
                issues.push(SafetyIssue {
                    risk_level: RiskLevel::High,
                    function_name: func.name.clone(),
                    description: format!(
                        "Mutable pointer parameter '{}' may lack size/bounds constraint",
                        param.name
                    ),
                    mitigation: "Add size parameter and bounds checking in wrapper".to_string(),
                    location: Some(format!("parameter '{}'", param.name)),
                });
            }
        }

        // Check for buffer parameters without length
        if Self::has_buffer_without_length(func) {
            issues.push(SafetyIssue {
                risk_level: RiskLevel::Critical,
                function_name: func.name.clone(),
                description: "Buffer parameter without corresponding length parameter".to_string(),
                mitigation: "Add length parameter or use Rust slice types".to_string(),
                location: Some("function signature".to_string()),
            });
        }

        // Check for void* parameters (type-unsafe)
        for param in &func.params {
            if param.ty.contains("void") && param.is_pointer {
                issues.push(SafetyIssue {
                    risk_level: RiskLevel::Medium,
                    function_name: func.name.clone(),
                    description: format!(
                        "Type-unsafe void* parameter '{}'",
                        param.name
                    ),
                    mitigation: "Use typed pointer or opaque handle instead".to_string(),
                    location: Some(format!("parameter '{}'", param.name)),
                });
            }
        }

        // Check for missing error handling
        if !Self::has_error_return(func) {
            issues.push(SafetyIssue {
                risk_level: RiskLevel::Low,
                function_name: func.name.clone(),
                description: "No error return value - failures may be silent".to_string(),
                mitigation: "Document failure modes or add Result return type".to_string(),
                location: Some("return type".to_string()),
            });
        }

        issues
    }

    /// Check if type is a pointer
    fn is_pointer_type(type_name: &str) -> bool {
        type_name.contains('*') || type_name.ends_with("Ptr")
    }



    /// Check if type is an opaque pointer (handle)
    fn is_opaque_pointer(type_name: &str) -> bool {
        // Heuristic: types ending in Handle, Context, or T
        type_name.ends_with("Handle") || 
        type_name.ends_with("Context") ||
        type_name.ends_with("_t")
    }

    /// Check if function has buffer without length
    fn has_buffer_without_length(func: &FfiFunction) -> bool {
        let has_buffer = func.params.iter().any(|p| {
            p.is_pointer && 
            !Self::is_opaque_pointer(&p.ty) &&
            (p.name.contains("buffer") || p.name.contains("data") || p.name.contains("array"))
        });

        let has_length = func.params.iter().any(|p| {
            p.name.contains("len") || 
            p.name.contains("size") || 
            p.name.contains("count") ||
            p.name.contains("num")
        });

        has_buffer && !has_length
    }

    /// Check if function has error return
    fn has_error_return(func: &FfiFunction) -> bool {
        // Check for common error return patterns
        func.return_type.contains("Result") ||
        func.return_type.contains("Status") ||
        func.return_type.contains("Error") ||
        func.return_type.contains("int") || // C error codes
        !func.return_type.contains("void")
    }

    /// Calculate overall risk level
    fn calculate_overall_risk(issues: &[SafetyIssue]) -> RiskLevel {
        if issues.is_empty() {
            return RiskLevel::Safe;
        }

        // If any critical issues, overall is critical
        if issues.iter().any(|i| i.risk_level == RiskLevel::Critical) {
            return RiskLevel::Critical;
        }

        // If multiple high issues, escalate to critical
        let high_count = issues.iter().filter(|i| i.risk_level == RiskLevel::High).count();
        if high_count > 5 {
            return RiskLevel::Critical;
        }
        if high_count > 0 {
            return RiskLevel::High;
        }

        // Count medium issues
        let medium_count = issues.iter().filter(|i| i.risk_level == RiskLevel::Medium).count();
        if medium_count > 10 {
            return RiskLevel::High;
        }
        if medium_count > 0 {
            return RiskLevel::Medium;
        }

        // Only low-risk issues
        RiskLevel::Low
    }

    /// Generate mitigation summary
    fn generate_mitigation_summary(issues: &[SafetyIssue]) -> String {
        if issues.is_empty() {
            return "No safety issues identified. Bindings appear safe to use.".to_string();
        }

        let mut summary = String::new();
        summary.push_str("The following mitigation strategies are recommended:\n\n");

        // Group by mitigation type
        let mut mitigations: HashMap<String, usize> = HashMap::new();
        for issue in issues {
            *mitigations.entry(issue.mitigation.clone()).or_insert(0) += 1;
        }

        // Sort by frequency
        let mut sorted: Vec<_> = mitigations.into_iter().collect();
        sorted.sort_by(|a, b| b.1.cmp(&a.1));

        for (idx, (mitigation, count)) in sorted.iter().enumerate() {
            summary.push_str(&format!("{}. {} ({} occurrence{})\n", 
                idx + 1, 
                mitigation, 
                count,
                if *count == 1 { "" } else { "s" }
            ));
        }

        summary
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_risk_level_ordering() {
        assert!(RiskLevel::Critical > RiskLevel::High);
        assert!(RiskLevel::High > RiskLevel::Medium);
        assert!(RiskLevel::Medium > RiskLevel::Low);
        assert!(RiskLevel::Low > RiskLevel::Safe);
    }

    #[test]
    fn test_is_pointer_type() {
        assert!(SafetyAudit::is_pointer_type("*const i32"));
        assert!(SafetyAudit::is_pointer_type("*mut u8"));
        assert!(SafetyAudit::is_pointer_type("HandlePtr"));
        assert!(!SafetyAudit::is_pointer_type("i32"));
    }

    #[test]
    fn test_is_opaque_pointer() {
        assert!(SafetyAudit::is_opaque_pointer("cudnnHandle_t"));
        assert!(SafetyAudit::is_opaque_pointer("MyContext"));
        assert!(SafetyAudit::is_opaque_pointer("FileHandle"));
        assert!(!SafetyAudit::is_opaque_pointer("*mut i32"));
    }

    #[test]
    fn test_analyze_pointer_without_null_check() {
        let func = FfiFunction {
            name: "test_func".to_string(),
            return_type: "void".to_string(),
            params: vec![FfiParam {
                name: "ptr".to_string(),
                ty: "*const i32".to_string(),
                is_pointer: true,
                is_mut: false,
            }],
            docs: None,
        };

        let issues = SafetyAudit::analyze_function(&func);
        assert!(!issues.is_empty());
        assert!(issues[0].description.contains("null"));
    }



    #[test]
    fn test_calculate_overall_risk_critical() {
        let issues = vec![SafetyIssue {
            risk_level: RiskLevel::Critical,
            function_name: "test".to_string(),
            description: "test".to_string(),
            mitigation: "test".to_string(),
            location: None,
        }];

        assert_eq!(SafetyAudit::calculate_overall_risk(&issues), RiskLevel::Critical);
    }

    #[test]
    fn test_calculate_overall_risk_multiple_high() {
        let issues = vec![
            SafetyIssue {
                risk_level: RiskLevel::High,
                function_name: "test1".to_string(),
                description: "test".to_string(),
                mitigation: "test".to_string(),
                location: None,
            };
            6 // More than 5 high-risk issues
        ];

        assert_eq!(SafetyAudit::calculate_overall_risk(&issues), RiskLevel::Critical);
    }

    #[test]
    fn test_safety_report_markdown() {
        let report = SafetyAuditReport {
            total_functions: 10,
            unsafe_operations: 10,
            issues: vec![SafetyIssue {
                risk_level: RiskLevel::High,
                function_name: "test_func".to_string(),
                description: "Test issue".to_string(),
                mitigation: "Test mitigation".to_string(),
                location: Some("line 42".to_string()),
            }],
            overall_risk: RiskLevel::High,
            mitigation_summary: "Test summary".to_string(),
        };

        let markdown = report.to_markdown();
        assert!(markdown.contains("Safety Audit Report"));
        assert!(markdown.contains("HIGH"));
        assert!(markdown.contains("test_func"));
        assert!(markdown.contains("Test mitigation"));
    }
}
