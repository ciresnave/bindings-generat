#[cfg(test)]
mod thread_safety_tests {
    use bindings_generat::analyzer::AnalysisResult;
    use bindings_generat::analyzer::errors::ErrorPatterns;
    use bindings_generat::analyzer::raii::{HandleType, LifecyclePair};
    use bindings_generat::analyzer::thread_safety::ThreadSafetyAnalyzer;
    use bindings_generat::enrichment::{EnhancedContext, FunctionContext};
    use bindings_generat::ffi::FfiFunction;
    use bindings_generat::generator::wrappers::generate_raii_wrapper;

    fn create_mock_handle() -> HandleType {
        HandleType {
            name: "TestHandle_t".to_string(),
            is_pointer: true,
            create_functions: vec!["testHandleCreate".to_string()],
            destroy_functions: vec!["testHandleDestroy".to_string()],
        }
    }

    fn create_mock_lifecycle() -> LifecyclePair {
        LifecyclePair {
            handle_type: "TestHandle_t".to_string(),
            create_fn: "testHandleCreate".to_string(),
            destroy_fn: "testHandleDestroy".to_string(),
            confidence: 0.9,
        }
    }

    fn create_mock_function(name: &str) -> FfiFunction {
        FfiFunction {
            name: name.to_string(),
            return_type: "TestHandle_t".to_string(),
            params: vec![],
            docs: None,
        }
    }

    #[test]
    fn test_thread_safe_wrapper() {
        let handle = create_mock_handle();
        let pair = create_mock_lifecycle();
        let create_fn = create_mock_function("testHandleCreate");
        let destroy_fn = create_mock_function("testHandleDestroy");
        let error_patterns = ErrorPatterns::default();

        // Create context with thread-safe annotation
        let mut context = EnhancedContext::new();
        let mut func_ctx = FunctionContext::new("testHandleCreate".to_string());
        func_ctx.description = Some("This function is thread-safe".to_string());

        let mut analyzer = ThreadSafetyAnalyzer::new();
        func_ctx.analyze_thread_safety(&mut analyzer);
        context
            .functions
            .insert("testHandleCreate".to_string(), func_ctx);

        let analysis = AnalysisResult {
            raii_patterns: bindings_generat::analyzer::raii::RaiiPatterns::default(),
            error_patterns: error_patterns.clone(),
            parameter_analysis: None,
            smart_errors: None,
            enhanced_docs: None,
            builder_typestates: None,
            function_contexts: context.functions.clone(),
            changelog_entries: Vec::new(),
        };
        let wrapper = generate_raii_wrapper(
            &handle,
            &pair,
            Some(&create_fn),
            Some(&destroy_fn),
            &error_patterns,
            "test",
            &analysis,
            None,
        );

        // Should contain thread safety documentation
        assert!(wrapper.code.contains("Thread-safe"));
        // Should NOT contain !Send or !Sync
        assert!(!wrapper.code.contains("impl !Send"));
        assert!(!wrapper.code.contains("impl !Sync"));
    }

    #[test]
    fn test_not_thread_safe_wrapper() {
        let handle = create_mock_handle();
        let pair = create_mock_lifecycle();
        let create_fn = create_mock_function("testHandleCreate");
        let destroy_fn = create_mock_function("testHandleDestroy");
        let error_patterns = ErrorPatterns::default();

        // Create context with not-thread-safe annotation
        let mut context = EnhancedContext::new();
        let mut func_ctx = FunctionContext::new("testHandleCreate".to_string());
        func_ctx.description =
            Some("WARNING: Not thread-safe. Requires external synchronization.".to_string());

        let mut analyzer = ThreadSafetyAnalyzer::new();
        func_ctx.analyze_thread_safety(&mut analyzer);
        context
            .functions
            .insert("testHandleCreate".to_string(), func_ctx);

        let analysis = AnalysisResult {
            raii_patterns: bindings_generat::analyzer::raii::RaiiPatterns::default(),
            error_patterns: error_patterns.clone(),
            parameter_analysis: None,
            smart_errors: None,
            enhanced_docs: None,
            builder_typestates: None,
            function_contexts: context.functions.clone(),
            changelog_entries: Vec::new(),
        };
        let wrapper = generate_raii_wrapper(
            &handle,
            &pair,
            Some(&create_fn),
            Some(&destroy_fn),
            &error_patterns,
            "test",
            &analysis,
            None,
        );

        // Should contain negative trait implementations
        assert!(wrapper.code.contains("impl !Send"));
        assert!(wrapper.code.contains("impl !Sync"));
        // Should contain warning
        assert!(wrapper.code.contains("Thread Safety"));
    }

    #[test]
    fn test_reentrant_wrapper() {
        let handle = create_mock_handle();
        let pair = create_mock_lifecycle();
        let create_fn = create_mock_function("testHandleCreate");
        let destroy_fn = create_mock_function("testHandleDestroy");
        let error_patterns = ErrorPatterns::default();

        // Create context with reentrant annotation
        let mut context = EnhancedContext::new();
        let mut func_ctx = FunctionContext::new("testHandleCreate".to_string());
        func_ctx.description =
            Some("@reentrant This function is reentrant and safe for signal handlers.".to_string());

        let mut analyzer = ThreadSafetyAnalyzer::new();
        func_ctx.analyze_thread_safety(&mut analyzer);
        context
            .functions
            .insert("testHandleCreate".to_string(), func_ctx);

        let analysis = AnalysisResult {
            raii_patterns: bindings_generat::analyzer::raii::RaiiPatterns::default(),
            error_patterns: error_patterns.clone(),
            parameter_analysis: None,
            smart_errors: None,
            enhanced_docs: None,
            builder_typestates: None,
            function_contexts: context.functions.clone(),
            changelog_entries: Vec::new(),
        };
        let wrapper = generate_raii_wrapper(
            &handle,
            &pair,
            Some(&create_fn),
            Some(&destroy_fn),
            &error_patterns,
            "test",
            &analysis,
            None,
        );

        // Reentrant should be thread-safe (Send + Sync)
        assert!(!wrapper.code.contains("impl !Send"));
        assert!(!wrapper.code.contains("impl !Sync"));
    }

    #[test]
    fn test_per_thread_wrapper() {
        let handle = create_mock_handle();
        let pair = create_mock_lifecycle();
        let create_fn = create_mock_function("testHandleCreate");
        let destroy_fn = create_mock_function("testHandleDestroy");
        let error_patterns = ErrorPatterns::default();

        // Create context with per-thread annotation
        let mut context = EnhancedContext::new();
        let mut func_ctx = FunctionContext::new("testHandleCreate".to_string());
        func_ctx.description = Some("One instance per thread is required.".to_string());

        let mut analyzer = ThreadSafetyAnalyzer::new();
        func_ctx.analyze_thread_safety(&mut analyzer);
        context
            .functions
            .insert("testHandleCreate".to_string(), func_ctx);

        let analysis = AnalysisResult {
            raii_patterns: bindings_generat::analyzer::raii::RaiiPatterns::default(),
            error_patterns: error_patterns.clone(),
            parameter_analysis: None,
            smart_errors: None,
            enhanced_docs: None,
            builder_typestates: None,
            function_contexts: context.functions.clone(),
            changelog_entries: Vec::new(),
        };
        let wrapper = generate_raii_wrapper(
            &handle,
            &pair,
            Some(&create_fn),
            Some(&destroy_fn),
            &error_patterns,
            "test",
            &analysis,
            None,
        );

        // Per-thread should be Send but not Sync
        assert!(!wrapper.code.contains("impl !Send"), "Should be Send");
        assert!(wrapper.code.contains("impl !Sync"), "Should not be Sync");
    }

    #[test]
    fn test_unknown_thread_safety_conservative() {
        let handle = create_mock_handle();
        let pair = create_mock_lifecycle();
        let create_fn = create_mock_function("testHandleCreate");
        let destroy_fn = create_mock_function("testHandleDestroy");
        let error_patterns = ErrorPatterns::default();

        // No context provided - should be conservative
        let context = EnhancedContext::new();
        let analysis = AnalysisResult {
            raii_patterns: bindings_generat::analyzer::raii::RaiiPatterns::default(),
            error_patterns: error_patterns.clone(),
            parameter_analysis: None,
            smart_errors: None,
            enhanced_docs: None,
            builder_typestates: None,
            function_contexts: context.functions.clone(),
            changelog_entries: Vec::new(),
        };
        let wrapper = generate_raii_wrapper(
            &handle,
            &pair,
            Some(&create_fn),
            Some(&destroy_fn),
            &error_patterns,
            "test",
            &analysis,
            None,
        );

        // Should be conservative: not Send, not Sync
        assert!(wrapper.code.contains("impl !Send"));
        assert!(wrapper.code.contains("impl !Sync"));
        assert!(wrapper.code.contains("Thread safety unknown"));
    }
}
