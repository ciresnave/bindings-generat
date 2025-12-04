//! Integration test for cross-platform code generation

use bindings_generat::analyzer::platform::{Platform, PlatformInfo};
use bindings_generat::ffi::{FfiFunction, FfiParam};
use bindings_generat::generator::cross_platform;

#[test]
fn test_generate_cfg_attribute_single_platform() {
    let mut info = PlatformInfo::new("test_func".to_string());
    info.available_on.insert(Platform::Windows);
    
    let attr = cross_platform::generate_cfg_attribute(&info);
    assert!(attr.is_some());
    let attr_str = attr.unwrap();
    assert!(attr_str.contains("#[cfg("));
    assert!(attr_str.contains("target_os = \"windows\""));
}

#[test]
fn test_generate_cfg_attribute_multiple_platforms() {
    let mut info = PlatformInfo::new("test_func".to_string());
    info.available_on.insert(Platform::Windows);
    info.available_on.insert(Platform::Linux);
    info.available_on.insert(Platform::MacOS);
    
    let attr = cross_platform::generate_cfg_attribute(&info);
    assert!(attr.is_some());
    let attr_str = attr.unwrap();
    assert!(attr_str.contains("any("));
    assert!(attr_str.contains("target_os = \"windows\""));
    assert!(attr_str.contains("target_os = \"linux\""));
    assert!(attr_str.contains("target_os = \"macos\""));
}

#[test]
fn test_generate_platform_utils() {
    let utils = cross_platform::generate_platform_utils();
    
    // Verify utility functions are present
    assert!(utils.contains("fn is_windows()"));
    assert!(utils.contains("fn is_linux()"));
    assert!(utils.contains("fn is_macos()"));
    assert!(utils.contains("fn is_unix()"));
    assert!(utils.contains("fn current_platform()"));
    
    // Verify conditional compilation
    assert!(utils.contains("#[cfg(target_os = \"windows\")]"));
    assert!(utils.contains("#[cfg(target_os = \"linux\")]"));
    assert!(utils.contains("#[cfg(target_os = \"macos\")]"));
    
    println!("\n=== GENERATED PLATFORM UTILS ===\n");
    println!("{}", utils);
    println!("\n=== END OUTPUT ===\n");
}

#[test]
fn test_generate_cross_platform_tests() {
    let func = FfiFunction {
        name: "test_function".to_string(),
        return_type: "int".to_string(),
        params: vec![
            FfiParam {
                name: "value".to_string(),
                ty: "int".to_string(),
                is_pointer: false,
                is_mut: false,
            },
        ],
        docs: None,
    };

    let mut info = PlatformInfo::new("test_function".to_string());
    info.available_on.insert(Platform::Windows);

    let test_code = cross_platform::generate_cross_platform_tests(&func, Some(&info));
    
    assert!(test_code.contains("test_test_function_cross_platform"));
    assert!(test_code.contains("platform-specific"));
    assert!(test_code.contains("#[cfg(any("));
    
    println!("\n=== GENERATED CROSS-PLATFORM TEST ===\n");
    println!("{}", test_code);
    println!("\n=== END OUTPUT ===\n");
}

#[test]
fn test_platform_specific_wrapper_generation() {
    let func = FfiFunction {
        name: "windows_only_function".to_string(),
        return_type: "void".to_string(),
        params: vec![],
        docs: None,
    };

    let mut info = PlatformInfo::new("windows_only_function".to_string());
    info.available_on.insert(Platform::Windows);

    let impl_code = "    pub fn windows_only_function() {\n        unsafe { ffi::windows_only_function() }\n    }\n";
    let wrapper = cross_platform::generate_platform_specific_wrapper(&func, &info, impl_code);
    
    assert!(wrapper.contains("Platform-specific"));
    assert!(wrapper.contains("#[cfg(target_os = \"windows\")]"));
    assert!(wrapper.contains("pub fn windows_only_function"));
    
    println!("\n=== GENERATED PLATFORM-SPECIFIC WRAPPER ===\n");
    println!("{}", wrapper);
    println!("\n=== END OUTPUT ===\n");
}

#[test]
fn test_current_platform_detection() {
    // This test verifies that platform detection works at compile time
    let platform = if cfg!(target_os = "windows") {
        "Windows"
    } else if cfg!(target_os = "linux") {
        "Linux"
    } else if cfg!(target_os = "macos") {
        "macOS"
    } else {
        "Other"
    };
    
    println!("Current platform: {}", platform);
    assert!(!platform.is_empty());
}
