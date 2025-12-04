// Auto-generated dynamic FFI shim for simple
#![allow(dead_code, non_camel_case_types, non_snake_case, non_upper_case_globals)]

#[path = "ffi_bindings.rs"]
mod ffi_bindings;
pub use ffi_bindings::*;

use once_cell::sync::OnceCell;
use libloading::Library;
use libloading::Symbol;
use std::env;

static LIB: OnceCell<Library> = OnceCell::new();

fn find_library_path() -> String {
    if let Ok(p) = env::var("BINDINGS_GENERAT_LIBRARY") {
        return p;
    }
    #[cfg(target_os = "windows")]
    return format!("{}.dll", "simple");
    #[cfg(target_os = "macos")]
    return format!("lib{}.dylib", "simple");
    #[cfg(all(not(target_os = "macos"), not(target_os = "windows")))]
    return format!("lib{}.so", "simple");
}

pub fn get_lib() -> &'static Library {
    LIB.get_or_init(|| {
        let path = find_library_path();
        unsafe { Library::new(path).expect("Failed to load shared library") }
    })
}

type simple_create_fn = unsafe extern "C" fn(handle: * mut SimpleHandle) -> SimpleStatus;
pub unsafe fn simple_create(handle: * mut SimpleHandle) -> SimpleStatus {
    let sym: Symbol<simple_create_fn> = unsafe { get_lib().get(b"simple_create\0").expect("Missing symbol") };
    unsafe { (*sym)(handle) }
}

type simple_destroy_fn = unsafe extern "C" fn(handle: SimpleHandle) -> ();
pub unsafe fn simple_destroy(handle: SimpleHandle) -> () {
    let sym: Symbol<simple_destroy_fn> = unsafe { get_lib().get(b"simple_destroy\0").expect("Missing symbol") };
    unsafe { (*sym)(handle) }
}

type simple_process_fn = unsafe extern "C" fn(handle: SimpleHandle, value: :: core :: ffi :: c_int) -> SimpleStatus;
pub unsafe fn simple_process(handle: SimpleHandle, value: :: core :: ffi :: c_int) -> SimpleStatus {
    let sym: Symbol<simple_process_fn> = unsafe { get_lib().get(b"simple_process\0").expect("Missing symbol") };
    unsafe { (*sym)(handle, value) }
}

type simple_get_value_fn = unsafe extern "C" fn(handle: SimpleHandle, out_value: * mut :: core :: ffi :: c_int) -> SimpleStatus;
pub unsafe fn simple_get_value(handle: SimpleHandle, out_value: * mut :: core :: ffi :: c_int) -> SimpleStatus {
    let sym: Symbol<simple_get_value_fn> = unsafe { get_lib().get(b"simple_get_value\0").expect("Missing symbol") };
    unsafe { (*sym)(handle, out_value) }
}

type simple_set_name_fn = unsafe extern "C" fn(handle: SimpleHandle, name: * const :: core :: ffi :: c_char) -> SimpleStatus;
pub unsafe fn simple_set_name(handle: SimpleHandle, name: * const :: core :: ffi :: c_char) -> SimpleStatus {
    let sym: Symbol<simple_set_name_fn> = unsafe { get_lib().get(b"simple_set_name\0").expect("Missing symbol") };
    unsafe { (*sym)(handle, name) }
}

type simple_get_name_fn = unsafe extern "C" fn(handle: SimpleHandle, buffer: * mut :: core :: ffi :: c_char, buffer_size: :: core :: ffi :: c_int) -> SimpleStatus;
pub unsafe fn simple_get_name(handle: SimpleHandle, buffer: * mut :: core :: ffi :: c_char, buffer_size: :: core :: ffi :: c_int) -> SimpleStatus {
    let sym: Symbol<simple_get_name_fn> = unsafe { get_lib().get(b"simple_get_name\0").expect("Missing symbol") };
    unsafe { (*sym)(handle, buffer, buffer_size) }
}

