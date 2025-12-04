// Security audit system for generated bindings
// Identifies potential security vulnerabilities

use crate::ffi::parser::{FfiFunction, FfiInfo, FfiParam};
use std::collections::HashMap;

/// Type of security vulnerability
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VulnerabilityType {
    /// Buffer overflow potential
    BufferOverflow,
    /// Integer overflow/underflow
    IntegerOverflow,
    /// Use-after-free potential
    UseAfterFree,
    /// Double-free potential
    DoubleFree,
    /// Null pointer dereference
    NullDeref,
    /// Uninitialized memory access
    UninitMemory,
    /// Race condition potential
    RaceCondition,
    /// Injection vulnerability (SQL, command, etc.)
    Injection,
    /// Information leak
    InfoLeak,
    /// Type confusion
    TypeConfusion,
}

impl VulnerabilityType {
    pub fn as_str(&self) -> &'static str {
        match self {
            VulnerabilityType::BufferOverflow => "Buffer Overflow",
            VulnerabilityType::IntegerOverflow => "Integer Overflow",
            VulnerabilityType::UseAfterFree => "Use-After-Free",
            VulnerabilityType::DoubleFree => "Double-Free",
            VulnerabilityType::NullDeref => "Null Pointer Dereference",
            VulnerabilityType::UninitMemory => "Uninitialized Memory",
            VulnerabilityType::RaceCondition => "Race Condition",
            VulnerabilityType::Injection => "Injection Attack",
            VulnerabilityType::InfoLeak => "Information Leak",
            VulnerabilityType::TypeConfusion => "Type Confusion",
        }
    }

    pub fn severity(&self) -> &'static str {
        match self {
            VulnerabilityType::BufferOverflow => "CRITICAL",
            VulnerabilityType::IntegerOverflow => "HIGH",
            VulnerabilityType::UseAfterFree => "CRITICAL",
            VulnerabilityType::DoubleFree => "CRITICAL",
            VulnerabilityType::NullDeref => "HIGH",
            VulnerabilityType::UninitMemory => "HIGH",
            VulnerabilityType::RaceCondition => "HIGH",
            VulnerabilityType::Injection => "CRITICAL",
            VulnerabilityType::InfoLeak => "MEDIUM",
            VulnerabilityType::TypeConfusion => "HIGH",
        }
    }

    pub fn cwe_id(&self) -> Option<u32> {
        match self {
            VulnerabilityType::BufferOverflow => Some(120), // CWE-120: Buffer Copy without Checking Size of Input
            VulnerabilityType::IntegerOverflow => Some(190), // CWE-190: Integer Overflow or Wraparound
            VulnerabilityType::UseAfterFree => Some(416),    // CWE-416: Use After Free
            VulnerabilityType::DoubleFree => Some(415),      // CWE-415: Double Free
            VulnerabilityType::NullDeref => Some(476),       // CWE-476: NULL Pointer Dereference
            VulnerabilityType::UninitMemory => Some(457),    // CWE-457: Use of Uninitialized Variable
            VulnerabilityType::RaceCondition => Some(362),   // CWE-362: Concurrent Execution using Shared Resource
            VulnerabilityType::Injection => Some(77),        // CWE-77: Command Injection
            VulnerabilityType::InfoLeak => Some(200),        // CWE-200: Exposure of Sensitive Information
            VulnerabilityType::TypeConfusion => Some(843),   // CWE-843: Access of Resource Using Incompatible Type
        }
    }
}

/// Security vulnerability found during audit
#[derive(Debug, Clone)]
pub struct SecurityVulnerability {
    /// Type of vulnerability
    pub vuln_type: VulnerabilityType,
    /// Function where vulnerability was found
    pub function_name: String,
    /// Description of the vulnerability
    pub description: String,
    /// Recommended fix
    pub fix: String,
    /// Location in code
    pub location: Option<String>,
    /// Exploitation scenario
    pub exploitation: String,
}

/// Security audit report
#[derive(Debug, Clone)]
pub struct SecurityAuditReport {
    /// Total functions analyzed
    pub total_functions: usize,
    /// Vulnerabilities found
    pub vulnerabilities: Vec<SecurityVulnerability>,
    /// Overall security rating (0-100, higher is better)
    pub security_score: u32,
}

impl SecurityAuditReport {
    /// Generate markdown report
    pub fn to_markdown(&self) -> String {
        let mut md = String::new();
        
        md.push_str("# Security Audit Report\n\n");
        md.push_str(&format!("**Generated**: {}\n\n", chrono::Local::now().format("%Y-%m-%d %H:%M:%S")));
        
        // Executive summary
        md.push_str("## Executive Summary\n\n");
        md.push_str(&format!("- **Security Score**: {}/100 {}\n", 
            self.security_score,
            Self::score_emoji(self.security_score)
        ));
        md.push_str(&format!("- **Functions Analyzed**: {}\n", self.total_functions));
        md.push_str(&format!("- **Vulnerabilities Found**: {}\n\n", self.vulnerabilities.len()));

        // Vulnerability distribution
        let mut vuln_counts: HashMap<VulnerabilityType, usize> = HashMap::new();
        for vuln in &self.vulnerabilities {
            *vuln_counts.entry(vuln.vuln_type).or_insert(0) += 1;
        }

        if !vuln_counts.is_empty() {
            md.push_str("### Vulnerability Distribution\n\n");
            md.push_str("| Vulnerability Type | Severity | Count | CWE |\n");
            md.push_str("|--------------------|----------|-------|-----|\n");
            
            let mut sorted: Vec<_> = vuln_counts.iter().collect();
            sorted.sort_by_key(|(t, _)| t.severity());
            sorted.reverse();
            
            for (vuln_type, count) in sorted {
                let cwe = vuln_type.cwe_id()
                    .map(|id| format!("CWE-{}", id))
                    .unwrap_or_else(|| "N/A".to_string());
                md.push_str(&format!("| {} | {} | {} | {} |\n", 
                    vuln_type.as_str(),
                    vuln_type.severity(),
                    count,
                    cwe
                ));
            }
            md.push_str("\n");
        }

        // Detailed vulnerabilities
        if !self.vulnerabilities.is_empty() {
            md.push_str("## Detailed Vulnerabilities\n\n");
            
            // Group by severity
            for severity in ["CRITICAL", "HIGH", "MEDIUM", "LOW"] {
                let vulns: Vec<_> = self.vulnerabilities.iter()
                    .filter(|v| v.vuln_type.severity() == severity)
                    .collect();
                
                if !vulns.is_empty() {
                    md.push_str(&format!("### {} Severity\n\n", severity));
                    
                    for (idx, vuln) in vulns.iter().enumerate() {
                        md.push_str(&format!("#### {}.{} {} in `{}`\n\n", 
                            severity, 
                            idx + 1,
                            vuln.vuln_type.as_str(),
                            vuln.function_name
                        ));
                        
                        if let Some(cwe) = vuln.vuln_type.cwe_id() {
                            md.push_str(&format!("**CWE**: CWE-{} ({})\n\n", cwe, vuln.vuln_type.as_str()));
                        }
                        
                        md.push_str(&format!("**Description**: {}\n\n", vuln.description));
                        md.push_str(&format!("**Exploitation**: {}\n\n", vuln.exploitation));
                        md.push_str(&format!("**Fix**: {}\n\n", vuln.fix));
                        
                        if let Some(ref location) = vuln.location {
                            md.push_str(&format!("**Location**: {}\n\n", location));
                        }
                    }
                }
            }
        }

        // Recommendations
        md.push_str("## Security Recommendations\n\n");
        md.push_str("1. **Input Validation**: Validate all parameters, especially sizes and indices\n");
        md.push_str("2. **Bounds Checking**: Always check array/buffer bounds before access\n");
        md.push_str("3. **Integer Overflow**: Use checked arithmetic for size calculations\n");
        md.push_str("4. **Null Safety**: Check all pointers before dereferencing\n");
        md.push_str("5. **Memory Safety**: Ensure proper initialization and cleanup\n");
        md.push_str("6. **Thread Safety**: Use proper synchronization for shared resources\n");
        md.push_str("7. **Fuzzing**: Run fuzzing tests to discover edge cases\n");
        md.push_str("8. **Code Review**: Have security expert review high-risk functions\n\n");

        md
    }

    fn score_emoji(score: u32) -> &'static str {
        match score {
            90..=100 => "🟢",
            70..=89 => "🟡",
            50..=69 => "🟠",
            _ => "🔴",
        }
    }

    /// Count vulnerabilities by type
    pub fn count_by_type(&self, vuln_type: VulnerabilityType) -> usize {
        self.vulnerabilities.iter().filter(|v| v.vuln_type == vuln_type).count()
    }
}

/// Security audit analyzer
pub struct SecurityAudit;

impl SecurityAudit {
    /// Perform comprehensive security audit
    pub fn analyze(ffi_info: &FfiInfo) -> SecurityAuditReport {
        let mut vulnerabilities = Vec::new();

        // Analyze each function
        for func in &ffi_info.functions {
            vulnerabilities.extend(Self::analyze_function(func));
        }

        // Calculate security score (0-100, higher is better)
        let security_score = Self::calculate_security_score(
            ffi_info.functions.len(),
            &vulnerabilities,
        );

        SecurityAuditReport {
            total_functions: ffi_info.functions.len(),
            vulnerabilities,
            security_score,
        }
    }

    /// Analyze a single function for security vulnerabilities
    fn analyze_function(func: &FfiFunction) -> Vec<SecurityVulnerability> {
        let mut vulns = Vec::new();

        // Check for buffer overflow risks
        vulns.extend(Self::check_buffer_overflow(func));

        // Check for integer overflow risks
        vulns.extend(Self::check_integer_overflow(func));

        // Check for null dereference risks
        vulns.extend(Self::check_null_deref(func));

        // Check for use-after-free risks
        vulns.extend(Self::check_use_after_free(func));

        // Check for race condition risks
        vulns.extend(Self::check_race_condition(func));

        // Check for type confusion risks
        vulns.extend(Self::check_type_confusion(func));

        vulns
    }

    /// Check for buffer overflow vulnerabilities
    fn check_buffer_overflow(func: &FfiFunction) -> Vec<SecurityVulnerability> {
        let mut vulns = Vec::new();

        // Look for buffer parameters without size constraints
        for param in &func.params {
            if Self::is_buffer_param(&param.ty, &param.name) {
                // Check if there's a corresponding size parameter
                let has_size = func.params.iter().any(|p| {
                    Self::is_size_param(&p.name, &param.name)
                });

                if !has_size {
                    vulns.push(SecurityVulnerability {
                        vuln_type: VulnerabilityType::BufferOverflow,
                        function_name: func.name.clone(),
                        description: format!(
                            "Buffer parameter '{}' has no size constraint - potential buffer overflow",
                            param.name
                        ),
                        fix: "Add size parameter and validate bounds before access".to_string(),
                        location: Some(format!("parameter '{}'", param.name)),
                        exploitation: "Attacker could provide oversized buffer causing out-of-bounds write".to_string(),
                    });
                }
            }
        }

        vulns
    }

    /// Check for integer overflow vulnerabilities
    fn check_integer_overflow(func: &FfiFunction) -> Vec<SecurityVulnerability> {
        let mut vulns = Vec::new();

        // Look for size/count parameters that might be used in calculations
        for param in &func.params {
            if Self::is_size_param(&param.name, "") && Self::is_integer_type(&param.ty) {
                // Conservative: flag any unbounded size parameter
                vulns.push(SecurityVulnerability {
                    vuln_type: VulnerabilityType::IntegerOverflow,
                    function_name: func.name.clone(),
                    description: format!(
                        "Size parameter '{}' may cause integer overflow in calculations",
                        param.name
                    ),
                    fix: "Add maximum bound constraint and use checked arithmetic".to_string(),
                    location: Some(format!("parameter '{}'", param.name)),
                    exploitation: "Attacker could provide large size causing overflow in calculations".to_string(),
                });
            }
        }

        vulns
    }

    /// Check for null dereference vulnerabilities
    fn check_null_deref(func: &FfiFunction) -> Vec<SecurityVulnerability> {
        let mut vulns = Vec::new();

        // Look for pointer parameters without null checks
        for param in &func.params {
            if param.is_pointer && !Self::is_optional_pointer(&param.name) {
                vulns.push(SecurityVulnerability {
                    vuln_type: VulnerabilityType::NullDeref,
                    function_name: func.name.clone(),
                    description: format!(
                        "Pointer parameter '{}' may be dereferenced without null check",
                        param.name
                    ),
                    fix: "Add null check in wrapper or document null handling".to_string(),
                    location: Some(format!("parameter '{}'", param.name)),
                    exploitation: "Passing null pointer could cause crash or be exploited".to_string(),
                });
            }
        }

        vulns
    }

    /// Check for use-after-free vulnerabilities
    fn check_use_after_free(func: &FfiFunction) -> Vec<SecurityVulnerability> {
        let mut vulns = Vec::new();

        // Look for functions that might free memory but allow continued use
        if func.name.to_lowercase().contains("destroy") || 
           func.name.to_lowercase().contains("free") ||
           func.name.to_lowercase().contains("delete") {
            
            vulns.push(SecurityVulnerability {
                vuln_type: VulnerabilityType::UseAfterFree,
                function_name: func.name.clone(),
                description: "Function frees memory - ensure handle is invalidated".to_string(),
                fix: "Implement Drop trait to consume handle and prevent reuse".to_string(),
                location: Some("function semantics".to_string()),
                exploitation: "Using handle after free could cause crash or memory corruption".to_string(),
            });
        }

        vulns
    }

    /// Check for race condition vulnerabilities
    fn check_race_condition(func: &FfiFunction) -> Vec<SecurityVulnerability> {
        let mut vulns = Vec::new();

        // Look for functions that modify shared state
        let has_mutable_ptr = func.params.iter().any(|p| {
            p.is_mut && p.is_pointer &&
            (p.name.contains("context") || p.name.contains("state") || p.name.contains("handle"))
        });

        if has_mutable_ptr {
            vulns.push(SecurityVulnerability {
                vuln_type: VulnerabilityType::RaceCondition,
                function_name: func.name.clone(),
                description: "Function modifies shared state - potential race condition".to_string(),
                fix: "Add synchronization or document thread-safety requirements".to_string(),
                location: Some("mutable shared state".to_string()),
                exploitation: "Concurrent access could cause data corruption or crashes".to_string(),
            });
        }

        vulns
    }

    /// Check for type confusion vulnerabilities
    fn check_type_confusion(func: &FfiFunction) -> Vec<SecurityVulnerability> {
        let mut vulns = Vec::new();

        // Look for void* parameters (type-unsafe)
        for param in &func.params {
            if param.ty.contains("void") && param.is_pointer {
                vulns.push(SecurityVulnerability {
                    vuln_type: VulnerabilityType::TypeConfusion,
                    function_name: func.name.clone(),
                    description: format!(
                        "Type-unsafe void* parameter '{}' - potential type confusion",
                        param.name
                    ),
                    fix: "Use typed pointer or opaque handle with type checking".to_string(),
                    location: Some(format!("parameter '{}'", param.name)),
                    exploitation: "Passing wrong type could cause memory corruption".to_string(),
                });
            }
        }

        vulns
    }

    /// Check if parameter is a buffer
    fn is_buffer_param(param_type: &str, param_name: &str) -> bool {
        let is_ptr = param_type.contains('*');
        let name_indicates_buffer = param_name.contains("buffer") || 
                                   param_name.contains("data") ||
                                   param_name.contains("array") ||
                                   param_name.contains("mem");
        is_ptr && name_indicates_buffer
    }

    /// Check if parameter is a size parameter for a buffer
    fn is_size_param(param_name: &str, buffer_name: &str) -> bool {
        let is_size = param_name.contains("size") || 
                     param_name.contains("len") ||
                     param_name.contains("count") ||
                     param_name.contains("num");
        
        if buffer_name.is_empty() {
            is_size
        } else {
            is_size && param_name.to_lowercase().contains(&buffer_name.to_lowercase())
        }
    }



    /// Check if type is an integer
    fn is_integer_type(type_name: &str) -> bool {
        type_name.contains("int") || 
        type_name.contains("size") ||
        type_name.contains("i32") ||
        type_name.contains("u32") ||
        type_name.contains("i64") ||
        type_name.contains("u64")
    }

    /// Check if pointer is optional (nullable by design)
    fn is_optional_pointer(param_name: &str) -> bool {
        param_name.to_lowercase().contains("optional") ||
        param_name.to_lowercase().contains("opt")
    }

    /// Calculate security score (0-100)
    fn calculate_security_score(total_functions: usize, vulnerabilities: &[SecurityVulnerability]) -> u32 {
        if total_functions == 0 {
            return 100;
        }

        // Weight vulnerabilities by severity
        let mut penalty = 0;
        for vuln in vulnerabilities {
            penalty += match vuln.vuln_type.severity() {
                "CRITICAL" => 20,
                "HIGH" => 10,
                "MEDIUM" => 5,
                "LOW" => 2,
                _ => 1,
            };
        }

        // Normalize by number of functions
        let penalty_per_function = penalty as f64 / total_functions as f64;
        let score = (100.0 - penalty_per_function).max(0.0).min(100.0);
        
        score as u32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vulnerability_severity() {
        assert_eq!(VulnerabilityType::BufferOverflow.severity(), "CRITICAL");
        assert_eq!(VulnerabilityType::NullDeref.severity(), "HIGH");
        assert_eq!(VulnerabilityType::InfoLeak.severity(), "MEDIUM");
    }

    #[test]
    fn test_vulnerability_cwe() {
        assert_eq!(VulnerabilityType::BufferOverflow.cwe_id(), Some(120));
        assert_eq!(VulnerabilityType::UseAfterFree.cwe_id(), Some(416));
    }

    #[test]
    fn test_is_buffer_param() {
        assert!(SecurityAudit::is_buffer_param("*mut u8", "buffer"));
        assert!(SecurityAudit::is_buffer_param("*const u8", "data"));
        assert!(!SecurityAudit::is_buffer_param("i32", "buffer"));
        assert!(!SecurityAudit::is_buffer_param("*mut i32", "ptr"));
    }

    #[test]
    fn test_is_size_param() {
        assert!(SecurityAudit::is_size_param("buffer_size", "buffer"));
        assert!(SecurityAudit::is_size_param("data_len", "data"));
        assert!(SecurityAudit::is_size_param("count", ""));
        assert!(!SecurityAudit::is_size_param("ptr", ""));
    }

    #[test]
    fn test_check_buffer_overflow() {
        let func = FfiFunction {
            name: "test_func".to_string(),
            return_type: "void".to_string(),
            params: vec![FfiParam {
                name: "buffer".to_string(),
                ty: "*mut u8".to_string(),
                is_pointer: true,
                is_mut: true,
            }],
            docs: None,
        };

        let vulns = SecurityAudit::check_buffer_overflow(&func);
        assert!(!vulns.is_empty());
        assert_eq!(vulns[0].vuln_type, VulnerabilityType::BufferOverflow);
    }

    #[test]
    fn test_calculate_security_score_perfect() {
        let score = SecurityAudit::calculate_security_score(10, &[]);
        assert_eq!(score, 100);
    }

    #[test]
    fn test_calculate_security_score_with_vulns() {
        let vulns = vec![SecurityVulnerability {
            vuln_type: VulnerabilityType::BufferOverflow, // 20 points penalty
            function_name: "test".to_string(),
            description: "test".to_string(),
            fix: "test".to_string(),
            location: None,
            exploitation: "test".to_string(),
        }];

        let score = SecurityAudit::calculate_security_score(10, &vulns);
        assert_eq!(score, 98); // 100 - (20 / 10) = 98
    }
}
