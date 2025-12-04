use std::collections::HashMap;

/// Types of resource limits that can be extracted
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum LimitType {
    Connections,
    Threads,
    Handles,
    FileDescriptors,
    Memory,
    BufferSize,
    QueueSize,
    PoolSize,
    Timeout,
    Retries,
    Custom(String),
}

/// Units for numeric limits
#[derive(Debug, Clone, PartialEq)]
pub enum LimitUnit {
    Count,
    Bytes,
    Kilobytes,
    Megabytes,
    Gigabytes,
    Milliseconds,
    Seconds,
    Minutes,
}

/// Information about a resource limit
#[derive(Debug, Clone, PartialEq)]
pub struct LimitInfo {
    pub limit_type: LimitType,
    pub value: Option<u64>,
    pub unit: LimitUnit,
    pub is_maximum: bool,
    pub is_minimum: bool,
    pub is_recommended: bool,
    pub cleanup_required: bool,
    pub description: String,
}

/// Collection of resource limits extracted from documentation
#[derive(Debug, Clone, PartialEq)]
pub struct ResourceLimits {
    pub limits: HashMap<String, Vec<LimitInfo>>,
    pub confidence: f64,
}

impl ResourceLimits {
    pub fn new() -> Self {
        Self {
            limits: HashMap::new(),
            confidence: 1.0,
        }
    }

    pub fn add_limit(&mut self, key: String, limit: LimitInfo) {
        self.limits.entry(key).or_default().push(limit);
    }

    pub fn has_limits(&self) -> bool {
        !self.limits.is_empty()
    }

    pub fn get_limit(&self, key: &str) -> Option<&Vec<LimitInfo>> {
        self.limits.get(key)
    }
}

impl Default for ResourceLimits {
    fn default() -> Self {
        Self::new()
    }
}

/// Analyzer for extracting resource limit information
#[derive(Debug)]
pub struct ResourceLimitsAnalyzer {
    cache: HashMap<String, ResourceLimits>,
}

impl ResourceLimitsAnalyzer {
    pub fn new() -> Self {
        Self {
            cache: HashMap::new(),
        }
    }

    /// Analyze documentation to extract resource limits
    pub fn analyze(&mut self, function_name: &str, docs: &str) -> ResourceLimits {
        if let Some(cached) = self.cache.get(function_name) {
            return cached.clone();
        }

        let mut limits = ResourceLimits::new();
        let lower_docs = docs.to_lowercase();

        // Extract various limit patterns
        self.extract_connection_limits(docs, &lower_docs, &mut limits);
        self.extract_memory_limits(docs, &lower_docs, &mut limits);
        self.extract_thread_limits(docs, &lower_docs, &mut limits);
        self.extract_buffer_limits(docs, &lower_docs, &mut limits);
        self.extract_timeout_limits(docs, &lower_docs, &mut limits);
        self.extract_retry_limits(docs, &lower_docs, &mut limits);
        self.extract_pool_limits(docs, &lower_docs, &mut limits);
        self.extract_handle_limits(docs, &lower_docs, &mut limits);
        self.extract_cleanup_requirements(docs, &lower_docs, &mut limits);

        self.cache.insert(function_name.to_string(), limits.clone());
        limits
    }

    fn extract_connection_limits(&self, docs: &str, lower_docs: &str, limits: &mut ResourceLimits) {
        // Look for patterns like "maximum N connections", "N connections", "maximum of N connections"
        if lower_docs.contains("connection")
            && let Some(value) = self.extract_number_with_keyword(lower_docs, "connection") {
                let is_max = lower_docs.contains("maximum")
                    || lower_docs.contains("max")
                    || lower_docs.contains("limit");
                let is_min = lower_docs.contains("minimum") || lower_docs.contains("min");

                limits.add_limit(
                    "connections".to_string(),
                    LimitInfo {
                        limit_type: LimitType::Connections,
                        value: Some(value),
                        unit: LimitUnit::Count,
                        is_maximum: is_max,
                        is_minimum: is_min,
                        is_recommended: lower_docs.contains("recommend"),
                        cleanup_required: self.check_cleanup_needed(docs, "connection"),
                        description: format!("Maximum {} connections", value),
                    },
                );
            }
    }

    fn extract_memory_limits(&self, docs: &str, _lower_docs: &str, limits: &mut ResourceLimits) {
        // Pattern: "N MB", "N megabytes", "N GB"
        let memory_patterns = vec![
            (r"(\d+)\s*(?:mb|megabytes?)", LimitUnit::Megabytes),
            (r"(\d+)\s*(?:gb|gigabytes?)", LimitUnit::Gigabytes),
            (r"(\d+)\s*(?:kb|kilobytes?)", LimitUnit::Kilobytes),
            (r"(\d+)\s*bytes?", LimitUnit::Bytes),
        ];

        for line in docs.lines() {
            let lower_line = line.to_lowercase();

            for (pattern, unit) in &memory_patterns {
                if let Some(cap) = regex::Regex::new(pattern)
                    .ok()
                    .and_then(|re| re.captures(&lower_line))
                    && let Some(value_str) = cap.get(1)
                        && let Ok(value) = value_str.as_str().parse::<u64>() {
                            let is_max = lower_line.contains("maximum")
                                || lower_line.contains("max")
                                || lower_line.contains("limit");
                            let is_min = lower_line.contains("minimum")
                                || lower_line.contains("min")
                                || lower_line.contains("at least");

                            limits.add_limit(
                                "memory".to_string(),
                                LimitInfo {
                                    limit_type: LimitType::Memory,
                                    value: Some(value),
                                    unit: unit.clone(),
                                    is_maximum: is_max,
                                    is_minimum: is_min,
                                    is_recommended: lower_line.contains("recommend"),
                                    cleanup_required: self.check_cleanup_needed(docs, "memory"),
                                    description: line.trim().to_string(),
                                },
                            );
                        }
            }
        }
    }

    fn extract_thread_limits(&self, _docs: &str, lower_docs: &str, limits: &mut ResourceLimits) {
        if lower_docs.contains("thread")
            && let Some(value) = self.extract_number_with_keyword(lower_docs, "thread") {
                let is_max = lower_docs.contains("maximum") || lower_docs.contains("max");
                let is_min = lower_docs.contains("minimum") || lower_docs.contains("min");

                limits.add_limit(
                    "threads".to_string(),
                    LimitInfo {
                        limit_type: LimitType::Threads,
                        value: Some(value),
                        unit: LimitUnit::Count,
                        is_maximum: is_max,
                        is_minimum: is_min,
                        is_recommended: false,
                        cleanup_required: false,
                        description: format!("{} threads", value),
                    },
                );
            }
    }

    fn extract_buffer_limits(&self, docs: &str, lower_docs: &str, limits: &mut ResourceLimits) {
        let patterns = [
            "buffer size",
            "buffer length",
            "max buffer",
            "buffer capacity",
        ];

        for pattern in &patterns {
            if let Some(value) = self.extract_number_near(pattern, lower_docs) {
                limits.add_limit(
                    "buffer".to_string(),
                    LimitInfo {
                        limit_type: LimitType::BufferSize,
                        value: Some(value),
                        unit: LimitUnit::Bytes,
                        is_maximum: pattern.contains("max"),
                        is_minimum: false,
                        is_recommended: false,
                        cleanup_required: self.check_cleanup_needed(docs, "buffer"),
                        description: format!("Buffer size: {} bytes", value),
                    },
                );
            }
        }
    }

    fn extract_timeout_limits(&self, _docs: &str, lower_docs: &str, limits: &mut ResourceLimits) {
        // Pattern: "timeout of N seconds/milliseconds"
        let timeout_patterns = vec![
            (
                r"timeout.*?(\d+)\s*(?:ms|milliseconds?)",
                LimitUnit::Milliseconds,
            ),
            (r"timeout.*?(\d+)\s*(?:s|seconds?)", LimitUnit::Seconds),
            (r"timeout.*?(\d+)\s*(?:m|minutes?)", LimitUnit::Minutes),
        ];

        for (pattern, unit) in &timeout_patterns {
            if let Some(cap) = regex::Regex::new(pattern)
                .ok()
                .and_then(|re| re.captures(lower_docs))
                && let Some(value_str) = cap.get(1)
                    && let Ok(value) = value_str.as_str().parse::<u64>() {
                        limits.add_limit(
                            "timeout".to_string(),
                            LimitInfo {
                                limit_type: LimitType::Timeout,
                                value: Some(value),
                                unit: unit.clone(),
                                is_maximum: true,
                                is_minimum: false,
                                is_recommended: false,
                                cleanup_required: false,
                                description: format!("Timeout: {} {:?}", value, unit),
                            },
                        );
                    }
        }
    }

    fn extract_retry_limits(&self, _docs: &str, lower_docs: &str, limits: &mut ResourceLimits) {
        let patterns = [
            "maximum retries",
            "max retries",
            "retry limit",
            "retry count",
        ];

        for pattern in &patterns {
            if let Some(value) = self.extract_number_near(pattern, lower_docs) {
                limits.add_limit(
                    "retries".to_string(),
                    LimitInfo {
                        limit_type: LimitType::Retries,
                        value: Some(value),
                        unit: LimitUnit::Count,
                        is_maximum: true,
                        is_minimum: false,
                        is_recommended: false,
                        cleanup_required: false,
                        description: format!("Maximum {} retries", value),
                    },
                );
            }
        }
    }

    fn extract_pool_limits(&self, _docs: &str, lower_docs: &str, limits: &mut ResourceLimits) {
        let patterns = [
            "pool size",
            "connection pool",
            "thread pool",
            "maximum pool",
        ];

        for pattern in &patterns {
            if let Some(value) = self.extract_number_near(pattern, lower_docs) {
                limits.add_limit(
                    "pool".to_string(),
                    LimitInfo {
                        limit_type: LimitType::PoolSize,
                        value: Some(value),
                        unit: LimitUnit::Count,
                        is_maximum: pattern.contains("max"),
                        is_minimum: false,
                        is_recommended: false,
                        cleanup_required: false,
                        description: format!("Pool size: {}", value),
                    },
                );
            }
        }
    }

    fn extract_handle_limits(&self, docs: &str, lower_docs: &str, limits: &mut ResourceLimits) {
        // Check for file descriptors
        if lower_docs.contains("file descriptor") || lower_docs.contains("open file") {
            let keyword = if lower_docs.contains("file descriptor") {
                "file descriptor"
            } else {
                "file"
            };

            if let Some(value) = self.extract_number_with_keyword(lower_docs, keyword) {
                limits.add_limit(
                    "file descriptors".to_string(),
                    LimitInfo {
                        limit_type: LimitType::FileDescriptors,
                        value: Some(value),
                        unit: LimitUnit::Count,
                        is_maximum: true,
                        is_minimum: false,
                        is_recommended: false,
                        cleanup_required: self.check_cleanup_needed(docs, "file"),
                        description: format!("Maximum {} file descriptors", value),
                    },
                );
            }
        }
        // Check for handles
        else if lower_docs.contains("handle")
            && let Some(value) = self.extract_number_with_keyword(lower_docs, "handle") {
                limits.add_limit(
                    "handles".to_string(),
                    LimitInfo {
                        limit_type: LimitType::Handles,
                        value: Some(value),
                        unit: LimitUnit::Count,
                        is_maximum: true,
                        is_minimum: false,
                        is_recommended: false,
                        cleanup_required: self.check_cleanup_needed(docs, "handle"),
                        description: format!("Maximum {} handles", value),
                    },
                );
            }
    }

    fn extract_cleanup_requirements(
        &self,
        docs: &str,
        lower_docs: &str,
        limits: &mut ResourceLimits,
    ) {
        let cleanup_keywords = [
            "must be freed",
            "must free",
            "cleanup required",
            "close on error",
            "release on failure",
            "must release",
        ];

        for keyword in &cleanup_keywords {
            if lower_docs.contains(keyword) {
                limits.add_limit(
                    "cleanup".to_string(),
                    LimitInfo {
                        limit_type: LimitType::Custom("cleanup".to_string()),
                        value: None,
                        unit: LimitUnit::Count,
                        is_maximum: false,
                        is_minimum: false,
                        is_recommended: false,
                        cleanup_required: true,
                        description: self.extract_sentence_containing(docs, keyword),
                    },
                );
            }
        }
    }

    fn extract_number_with_keyword(&self, docs: &str, keyword: &str) -> Option<u64> {
        // Look for a number near the keyword
        self.extract_number_near(keyword, docs)
    }

    fn extract_number_near(&self, pattern: &str, docs: &str) -> Option<u64> {
        // Find the pattern and look for a number nearby (before or after)
        if let Some(pos) = docs.find(pattern) {
            let start = pos.saturating_sub(50);
            let end = (pos + pattern.len() + 50).min(docs.len());
            let context = &docs[start..end];

            // Look for numbers in context
            if let Ok(re) = regex::Regex::new(r"\d+") {
                for cap in re.find_iter(context) {
                    if let Ok(value) = cap.as_str().parse::<u64>() {
                        // Return the first reasonable number found
                        if value > 0 && value < 1_000_000_000 {
                            return Some(value);
                        }
                    }
                }
            }
        }
        None
    }

    fn check_cleanup_needed(&self, docs: &str, resource: &str) -> bool {
        let lower_docs = docs.to_lowercase();
        let cleanup_phrases = [
            format!("{} must be freed", resource),
            format!("free {}", resource),
            format!("close {}", resource),
            format!("release {}", resource),
            format!("cleanup {}", resource),
        ];

        cleanup_phrases
            .iter()
            .any(|phrase| lower_docs.contains(phrase.as_str()))
    }

    fn extract_sentence_containing(&self, docs: &str, keyword: &str) -> String {
        for line in docs.lines() {
            if line.to_lowercase().contains(keyword) {
                return line.trim().to_string();
            }
        }
        keyword.to_string()
    }

    /// Generate documentation from resource limits
    pub fn generate_documentation(&self, limits: &ResourceLimits) -> String {
        if !limits.has_limits() {
            return String::new();
        }

        let mut doc = String::from("# Resource Limits\n\n");

        // Group by category
        let mut categories: HashMap<&str, Vec<(&String, &Vec<LimitInfo>)>> = HashMap::new();

        for (key, limit_list) in &limits.limits {
            let category = match limit_list.first() {
                Some(info) => match &info.limit_type {
                    LimitType::Connections => "Connections",
                    LimitType::Threads => "Threading",
                    LimitType::Memory | LimitType::BufferSize => "Memory",
                    LimitType::Timeout => "Timeouts",
                    LimitType::Retries => "Retry Logic",
                    LimitType::PoolSize => "Resource Pools",
                    LimitType::Handles | LimitType::FileDescriptors => "File Handles",
                    LimitType::Custom(_) => "Cleanup",
                    _ => "Other",
                },
                None => continue,
            };
            categories
                .entry(category)
                .or_default()
                .push((key, limit_list));
        }

        for (category, items) in categories.iter() {
            doc.push_str(&format!("## {}\n\n", category));

            for (_key, limit_list) in items {
                for limit in *limit_list {
                    if let Some(value) = limit.value {
                        let limit_str = if limit.is_maximum {
                            "Maximum"
                        } else if limit.is_minimum {
                            "Minimum"
                        } else if limit.is_recommended {
                            "Recommended"
                        } else {
                            "Limit"
                        };

                        doc.push_str(&format!("- {}: {} {:?}\n", limit_str, value, limit.unit));
                    }

                    if !limit.description.is_empty()
                        && limit.description != format!("{:?}", limit.limit_type)
                    {
                        doc.push_str(&format!("  - {}\n", limit.description));
                    }

                    if limit.cleanup_required {
                        doc.push_str("  - ⚠️ Cleanup required\n");
                    }
                }
            }
            doc.push('\n');
        }

        doc
    }

    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }
}

impl Default for ResourceLimitsAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_connection_limits() {
        let mut analyzer = ResourceLimitsAnalyzer::new();
        let docs = "This function supports a maximum of 100 connections.";

        let limits = analyzer.analyze("connect", docs);
        assert!(limits.has_limits());

        let conn_limits = limits.get_limit("connections").unwrap();
        assert_eq!(conn_limits.len(), 1);
        assert_eq!(conn_limits[0].value, Some(100));
        assert!(conn_limits[0].is_maximum);
    }

    #[test]
    fn test_extract_memory_limits() {
        let mut analyzer = ResourceLimitsAnalyzer::new();
        let docs = "Buffer must be at least 512 MB in size.";

        let limits = analyzer.analyze("allocate", docs);
        assert!(limits.has_limits());

        let mem_limits = limits.get_limit("memory").unwrap();
        assert_eq!(mem_limits[0].value, Some(512));
        assert_eq!(mem_limits[0].unit, LimitUnit::Megabytes);
    }

    #[test]
    fn test_extract_thread_limits() {
        let mut analyzer = ResourceLimitsAnalyzer::new();
        let docs = "Maximum 8 threads can be used.";

        let limits = analyzer.analyze("start_threads", docs);
        assert!(limits.has_limits());

        let thread_limits = limits.get_limit("threads").unwrap();
        assert_eq!(thread_limits[0].value, Some(8));
        assert!(thread_limits[0].is_maximum);
    }

    #[test]
    fn test_extract_buffer_limits() {
        let mut analyzer = ResourceLimitsAnalyzer::new();
        let docs = "Buffer size must be 4096 bytes.";

        let limits = analyzer.analyze("read", docs);
        assert!(limits.has_limits());

        let buffer_limits = limits.get_limit("buffer").unwrap();
        assert_eq!(buffer_limits[0].value, Some(4096));
    }

    #[test]
    fn test_extract_timeout() {
        let mut analyzer = ResourceLimitsAnalyzer::new();
        let docs = "Operation has a timeout of 30 seconds.";

        let limits = analyzer.analyze("wait", docs);
        assert!(limits.has_limits());

        let timeout_limits = limits.get_limit("timeout").unwrap();
        assert_eq!(timeout_limits[0].value, Some(30));
        assert_eq!(timeout_limits[0].unit, LimitUnit::Seconds);
    }

    #[test]
    fn test_extract_retries() {
        let mut analyzer = ResourceLimitsAnalyzer::new();
        let docs = "Maximum retries is 3 attempts.";

        let limits = analyzer.analyze("retry", docs);
        assert!(limits.has_limits());

        let retry_limits = limits.get_limit("retries").unwrap();
        assert_eq!(retry_limits[0].value, Some(3));
        assert!(retry_limits[0].is_maximum);
    }

    #[test]
    fn test_extract_pool_size() {
        let mut analyzer = ResourceLimitsAnalyzer::new();
        let docs = "Connection pool size is 20.";

        let limits = analyzer.analyze("init_pool", docs);
        assert!(limits.has_limits());

        let pool_limits = limits.get_limit("pool").unwrap();
        assert_eq!(pool_limits[0].value, Some(20));
    }

    #[test]
    fn test_cleanup_required() {
        let mut analyzer = ResourceLimitsAnalyzer::new();
        let docs = "Allocated memory must be freed by caller.";

        let limits = analyzer.analyze("malloc", docs);
        assert!(limits.has_limits());

        let cleanup_limits = limits.get_limit("cleanup").unwrap();
        assert!(cleanup_limits[0].cleanup_required);
    }

    #[test]
    fn test_file_descriptor_limits() {
        let mut analyzer = ResourceLimitsAnalyzer::new();
        let docs = "Maximum 1024 file descriptors can be open.";

        let limits = analyzer.analyze("open", docs);
        assert!(limits.has_limits());

        let fd_limits = limits.get_limit("file descriptors").unwrap();
        assert_eq!(fd_limits[0].value, Some(1024));
        assert_eq!(fd_limits[0].limit_type, LimitType::FileDescriptors);
    }

    #[test]
    fn test_multiple_limits() {
        let mut analyzer = ResourceLimitsAnalyzer::new();
        let docs = "Maximum 100 connections with 512 MB buffer size and timeout of 30 seconds.";

        let limits = analyzer.analyze("connect", docs);
        assert!(limits.has_limits());

        // Should find connections, memory, and timeout
        assert!(limits.get_limit("connections").is_some());
        assert!(limits.get_limit("memory").is_some());
        assert!(limits.get_limit("timeout").is_some());
    }

    #[test]
    fn test_cache_functionality() {
        let mut analyzer = ResourceLimitsAnalyzer::new();
        let docs = "Maximum 10 connections.";

        let limits1 = analyzer.analyze("test_fn", docs);
        let limits2 = analyzer.analyze("test_fn", docs);

        assert_eq!(limits1, limits2);
    }

    #[test]
    fn test_generate_documentation() {
        let mut analyzer = ResourceLimitsAnalyzer::new();
        let docs = "Maximum 50 connections. Timeout of 60 seconds. Buffer must be freed.";

        let limits = analyzer.analyze("complex", docs);
        let doc = analyzer.generate_documentation(&limits);

        assert!(doc.contains("Resource Limits"));
        assert!(doc.contains("Connections"));
        assert!(doc.contains("Timeouts"));
    }
}
