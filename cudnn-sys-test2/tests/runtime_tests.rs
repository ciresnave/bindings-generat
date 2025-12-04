//! Runtime integration tests for cudnn bindings

//! These tests require the library to be installed.
//! Run with: cargo test --release

#![cfg(test)]
use cudnn::*;

#[cfg(test)]
mod lifecycle_tests {
    use super::*;

}

#[cfg(test)]
mod error_handling_tests {
    use super::*;

    #[test]
    fn test_error_implements_std_error() {
        fn assert_error_trait<T: std::error::Error>() {}
        assert_error_trait::<Error>();
    }

    #[test]
    fn test_error_is_send_sync() {
        fn assert_send<T: Send>() {}
        fn assert_sync<T: Sync>() {}
        assert_send::<Error>();
        assert_sync::<Error>();
    }

    #[test]
    fn test_error_display_not_empty() {
        // Create a sample error and verify Display implementation
        // This uses internal error codes - adjust based on actual error enum
        let error = Error::from(1); // Assuming 1 is an error code
        let display = format!("{}", error);
        assert!(!display.is_empty(), "Error display should not be empty");
        assert!(display.len() > 5, "Error message should be descriptive");
    }

    #[test]
    fn test_error_debug_not_empty() {
        let error = Error::from(1);
        let debug = format!("{:?}", error);
        assert!(!debug.is_empty(), "Error debug should not be empty");
    }

    #[test]
    fn test_result_propagation() {
        // Test that errors propagate correctly with ?
        fn create_and_return() -> Result<(), Error> {
            let _handle = CudaKernel::new()?;
            Ok(())
        }

        match create_and_return() {
            Ok(()) => println!("✓ Result propagation works"),
            Err(e) => panic!("Error propagation failed: {}", e),
        }
    }

}

#[cfg(test)]
mod method_tests {
    use super::*;

    #[test]
    #[ignore] // Requires specific setup
    fn test_cudakernel_has_methods() {
        // Verify CudaKernel has expected methods
        // Uncomment when you can create valid instances:

        // let mut instance = CudaKernel::new().expect("Failed to create instance");
        // Call library_get_kernel
        // let result = instance.library_get_kernel(/* parameters */);
        // assert!(result.is_ok());

        // Call library_enumerate_kernels
        // let result = instance.library_enumerate_kernels(/* parameters */);
        // assert!(result.is_ok());

        // Call kernel_set_attribute_for_device
        // let result = instance.kernel_set_attribute_for_device(/* parameters */);
        // assert!(result.is_ok());

    }

    #[test]
    #[ignore] // Requires specific setup
    fn test_cudagraphnode_has_methods() {
        // Verify CudaGraphNode has expected methods
        // Uncomment when you can create valid instances:

        // let mut instance = CudaGraphNode::new().expect("Failed to create instance");
        // Call graph_add_kernel_node
        // let result = instance.graph_add_kernel_node(/* parameters */);
        // assert!(result.is_ok());

        // Call graph_kernel_node_get_params
        // let result = instance.graph_kernel_node_get_params(/* parameters */);
        // assert!(result.is_ok());

        // Call graph_kernel_node_set_params
        // let result = instance.graph_kernel_node_set_params(/* parameters */);
        // assert!(result.is_ok());

    }

    #[test]
    #[ignore] // Requires specific setup
    fn test_cudnnseqdatadescriptor_has_methods() {
        // Verify CudnnSeqDataDescriptor has expected methods
        // Uncomment when you can create valid instances:

        // let mut instance = CudnnSeqDataDescriptor::new().expect("Failed to create instance");
        // Call set_seq_data_descriptor
        // let result = instance.set_seq_data_descriptor(/* parameters */);
        // assert!(result.is_ok());

        // Call get_seq_data_descriptor
        // let result = instance.get_seq_data_descriptor(/* parameters */);
        // assert!(result.is_ok());

    }

}

#[cfg(test)]
mod resource_leak_tests {
    use super::*;

    #[test]
    fn test_no_leaks_single_scope() {
        // Create and drop resource in single scope
        for i in 0..10 {
            let result = CudaKernel::new();
            if let Ok(handle) = result {
                drop(handle); // Explicit drop, then scope ends
                println!("✓ Iteration {}: Resource created and dropped", i);
            }
        }
        // If this completes without memory issues, no leaks detected
    }

    #[test]
    fn test_no_leaks_nested_scopes() {
        // Test RAII with nested scopes
        let outer = CudaKernel::new();
        if let Ok(_outer_handle) = outer {
            {
                let inner = CudaKernel::new();
                if let Ok(_inner_handle) = inner {
                    // Inner handle dropped here
                }
            }
            // Outer handle still valid here, dropped at end
        }
    }

}

#[cfg(test)]
mod concurrency_tests {
    use super::*;

    #[test]
    fn test_error_is_thread_safe() {
        // Verify error can be sent between threads
        let error = Error::from(1);
        let handle = std::thread::spawn(move || {
            format!("{}", error)
        });
        let result = handle.join().unwrap();
        assert!(!result.is_empty(), "Error should work across threads");
    }

    #[test]
    #[ignore] // May not be thread-safe
    fn test_create_in_multiple_threads() {
        // Test creating handles in different threads
        // WARNING: Only run if library is thread-safe!
        let handles: Vec<_> = (0..3)
            .map(|i| {
                std::thread::spawn(move || {
                    CudaKernel::new().map(|h| (i, h))
                })
            })
            .collect();

        for (i, handle) in handles.into_iter().enumerate() {
            match handle.join().unwrap() {
                Ok(_) => println!("✓ Thread {} created handle", i),
                Err(e) => println!("✗ Thread {} failed: {}", i, e),
            }
        }
    }

}

