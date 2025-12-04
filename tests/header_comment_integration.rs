/// Integration test for header comment extraction and context building
use bindings_generat::enrichment::header_parser::HeaderCommentParser;
use bindings_generat::llm::enhanced_context::EnhancedContext;

#[test]
fn test_cuda_style_header_comments() {
    // Realistic CUDA-style header with comprehensive Doxygen comments
    let header_content = r#"
/**
 * @brief Creates a CUDA runtime handle
 * 
 * @details This function initializes a new CUDA runtime handle that must be
 * used for all subsequent CUDA operations on a specific GPU device. The handle
 * maintains internal state including stream associations and memory allocations.
 * 
 * @param[out] handle Pointer to where the created handle will be stored. Must not be NULL.
 * @param[in] device_id The CUDA device ID to associate with this handle (0-indexed).
 * 
 * @return CUDA_SUCCESS on successful creation, or an error code:
 *   - CUDA_ERROR_INVALID_VALUE if handle is NULL or device_id is invalid
 *   - CUDA_ERROR_OUT_OF_MEMORY if allocation fails
 *   - CUDA_ERROR_NO_DEVICE if the specified device doesn't exist
 * 
 * @note This function must be called before any other operations using this handle.
 * @warning Not thread-safe. Each thread must create its own handle.
 * @warning Memory allocated by this handle must be freed with cudaDestroy().
 * 
 * @see cudaDestroy
 * @see cudaSetDevice
 * 
 * @deprecated This function is deprecated in CUDA 12.0. Use cudaCreateV2() instead.
 */
cudaError_t cudaCreate(cudaHandle_t* handle, int device_id);

/**
 * @brief Destroys a CUDA runtime handle
 * 
 * @param[in] handle The handle to destroy
 * 
 * @return CUDA_SUCCESS or CUDA_ERROR_INVALID_HANDLE
 */
cudaError_t cudaDestroy(cudaHandle_t handle);
"#;

    // Parse the header
    let parser = HeaderCommentParser::new().expect("Failed to create parser");
    let comments = parser.parse_header_content(header_content).expect("Failed to parse header");

    // Verify we extracted both functions
    assert_eq!(comments.len(), 2, "Should extract 2 function comments");

    // Verify cudaCreate function
    let cuda_create = comments.iter()
        .find(|c| c.function_name == "cudaCreate")
        .expect("Should find cudaCreate comment");

    assert_eq!(
        cuda_create.brief.as_ref().unwrap(),
        "Creates a CUDA runtime handle"
    );

    assert!(cuda_create.detailed.as_ref().unwrap().contains("initializes a new CUDA runtime handle"));
    
    // Verify parameters
    assert_eq!(cuda_create.param_docs.len(), 2, "Should have 2 parameters");
    
    let handle_param = cuda_create.param_docs.get("handle").unwrap();
    assert!(handle_param.description.contains("Pointer to where the created handle"));
    assert_eq!(
        handle_param.direction,
        Some(bindings_generat::enrichment::header_parser::ParamDirection::Out)
    );

    let device_param = cuda_create.param_docs.get("device_id").unwrap();
    assert!(device_param.description.contains("CUDA device ID"));
    assert_eq!(
        device_param.direction,
        Some(bindings_generat::enrichment::header_parser::ParamDirection::In)
    );

    // Verify return documentation
    assert!(cuda_create.return_doc.as_ref().unwrap().contains("CUDA_SUCCESS"));
    assert!(cuda_create.return_doc.as_ref().unwrap().contains("CUDA_ERROR_INVALID_VALUE"));

    // Verify notes and warnings
    assert_eq!(cuda_create.notes.len(), 1);
    assert!(cuda_create.notes[0].contains("must be called before"));

    assert_eq!(cuda_create.warnings.len(), 2);
    assert!(cuda_create.warnings[0].contains("Not thread-safe"));
    assert!(cuda_create.warnings[1].contains("Memory allocated"));

    // Verify see also references
    assert_eq!(cuda_create.see_also.len(), 2);
    assert!(cuda_create.see_also.contains(&"cudaDestroy".to_string()));
    assert!(cuda_create.see_also.contains(&"cudaSetDevice".to_string()));

    // Verify deprecated notice
    assert!(cuda_create.deprecated.is_some());
    assert!(cuda_create.deprecated.as_ref().unwrap().contains("CUDA 12.0"));

    // Verify cudaDestroy function
    let cuda_destroy = comments.iter()
        .find(|c| c.function_name == "cudaDestroy")
        .expect("Should find cudaDestroy comment");

    assert_eq!(
        cuda_destroy.brief.as_ref().unwrap(),
        "Destroys a CUDA runtime handle"
    );
}

#[test]
fn test_enhanced_context_with_real_header() {
    let header_content = r#"
/**
 * @brief Allocates device memory
 * 
 * @param[out] devPtr Pointer to allocated device memory
 * @param[in] size Size in bytes to allocate
 * 
 * @return cudaSuccess or cudaErrorMemoryAllocation
 * 
 * @note Must call cudaFree() to deallocate
 * @warning Do not use after cudaFree()
 */
cudaError_t cudaMalloc(void** devPtr, size_t size);
"#;

    // Parse header
    let parser = HeaderCommentParser::new().expect("Failed to create parser");
    let comments = parser.parse_header_content(header_content).expect("Failed to parse header");
    
    assert_eq!(comments.len(), 1);

    // Create enhanced context and add header comment
    let mut context = EnhancedContext::new();
    context.add_header_comment("cudaMalloc".to_string(), comments[0].clone());

    // Build function context
    let enhanced = context.build_function_context(
        "cudaMalloc",
        "fn cudaMalloc(dev_ptr: *mut *mut c_void, size: usize) -> cudaError_t"
    );

    // Verify the enhanced context includes all header information
    assert!(enhanced.contains("Header Documentation"));
    assert!(enhanced.contains("Allocates device memory"));
    assert!(enhanced.contains("devPtr [out]"));
    assert!(enhanced.contains("size [in]"));
    assert!(enhanced.contains("cudaSuccess or cudaErrorMemoryAllocation"));
    assert!(enhanced.contains("Must call cudaFree()"));
    assert!(enhanced.contains("Do not use after cudaFree()"));

    // Verify summary
    let summary = context.summary();
    assert!(summary.contains("1 header comment"));
}

#[test]
fn test_openssl_style_header_comments() {
    // OpenSSL-style documentation
    let header_content = r#"
/**
 * @brief Initialize an SSL context
 * 
 * @details Creates and initializes a new SSL_CTX object which holds various
 * configuration and data relevant to SSL/TLS or DTLS session establishment.
 * 
 * @param[in] method The SSL/TLS protocol method to use. Common values include:
 *                   TLS_client_method(), TLS_server_method(), TLS_method()
 * 
 * @return A pointer to the SSL_CTX structure on success, NULL on failure.
 *         Use ERR_get_error() to retrieve the error code.
 * 
 * @note The returned context must be freed with SSL_CTX_free() when no longer needed.
 * @note This function is thread-safe as of OpenSSL 1.1.0.
 * 
 * @see SSL_CTX_free
 * @see SSL_CTX_set_options
 * @see ERR_get_error
 */
SSL_CTX* SSL_CTX_new(const SSL_METHOD* method);
"#;

    let parser = HeaderCommentParser::new().expect("Failed to create parser");
    let comments = parser.parse_header_content(header_content).expect("Failed to parse header");

    assert_eq!(comments.len(), 1);
    let ssl_ctx_new = &comments[0];

    assert_eq!(ssl_ctx_new.function_name, "SSL_CTX_new");
    assert!(ssl_ctx_new.brief.as_ref().unwrap().contains("Initialize an SSL context"));
    assert!(ssl_ctx_new.detailed.as_ref().unwrap().contains("SSL_CTX object"));

    // Verify method parameter with detailed description
    let method_param = ssl_ctx_new.param_docs.get("method").unwrap();
    assert!(method_param.description.contains("TLS_client_method()"));
    assert_eq!(
        method_param.direction,
        Some(bindings_generat::enrichment::header_parser::ParamDirection::In)
    );

    // Verify return documentation
    assert!(ssl_ctx_new.return_doc.as_ref().unwrap().contains("SSL_CTX structure"));
    assert!(ssl_ctx_new.return_doc.as_ref().unwrap().contains("ERR_get_error"));

    // Verify multiple notes
    assert_eq!(ssl_ctx_new.notes.len(), 2);
    assert!(ssl_ctx_new.notes[0].contains("SSL_CTX_free()"));
    assert!(ssl_ctx_new.notes[1].contains("thread-safe"));

    // Verify see also references
    assert_eq!(ssl_ctx_new.see_also.len(), 3);
    assert!(ssl_ctx_new.see_also.contains(&"SSL_CTX_free".to_string()));
    assert!(ssl_ctx_new.see_also.contains(&"ERR_get_error".to_string()));
}

#[test]
fn test_line_comment_style() {
    let header_content = r#"
/// @brief Simple memory copy function
/// @param[out] dest Destination buffer
/// @param[in] src Source buffer  
/// @param[in] n Number of bytes to copy
/// @return Number of bytes copied
/// @note Buffers must not overlap
void* memcpy_safe(void* dest, const void* src, size_t n);
"#;

    let parser = HeaderCommentParser::new().expect("Failed to create parser");
    let comments = parser.parse_header_content(header_content).expect("Failed to parse header");

    assert_eq!(comments.len(), 1);
    let memcpy = &comments[0];

    assert_eq!(memcpy.function_name, "memcpy_safe");
    assert!(memcpy.brief.as_ref().unwrap().contains("Simple memory copy"));

    // Verify all three parameters
    assert_eq!(memcpy.param_docs.len(), 3);
    assert!(memcpy.param_docs.contains_key("dest"));
    assert!(memcpy.param_docs.contains_key("src"));
    assert!(memcpy.param_docs.contains_key("n"));

    // Verify return doc
    assert!(memcpy.return_doc.as_ref().unwrap().contains("bytes copied"));

    // Verify note
    assert_eq!(memcpy.notes.len(), 1);
    assert!(memcpy.notes[0].contains("must not overlap"));
}

#[test]
fn test_mixed_directions() {
    let header_content = r#"
/**
 * @brief Read and update counter
 * @param[in,out] counter Pointer to counter (read and incremented)
 * @param[in] increment Amount to add
 * @param[out] old_value Receives the value before increment
 * @return Status code
 */
int update_counter(int* counter, int increment, int* old_value);
"#;

    let parser = HeaderCommentParser::new().expect("Failed to create parser");
    let comments = parser.parse_header_content(header_content).expect("Failed to parse header");

    assert_eq!(comments.len(), 1);
    let func = &comments[0];

    // Verify in/out parameter
    let counter_param = func.param_docs.get("counter").unwrap();
    assert_eq!(
        counter_param.direction,
        Some(bindings_generat::enrichment::header_parser::ParamDirection::InOut)
    );

    // Verify in parameter
    let increment_param = func.param_docs.get("increment").unwrap();
    assert_eq!(
        increment_param.direction,
        Some(bindings_generat::enrichment::header_parser::ParamDirection::In)
    );

    // Verify out parameter
    let old_value_param = func.param_docs.get("old_value").unwrap();
    assert_eq!(
        old_value_param.direction,
        Some(bindings_generat::enrichment::header_parser::ParamDirection::Out)
    );
}
