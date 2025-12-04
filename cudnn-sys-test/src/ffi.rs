// Auto-generated dynamic FFI shim for cudnn64_9
#![allow(
    dead_code,
    non_camel_case_types,
    non_snake_case,
    non_upper_case_globals
)]

#[derive(Debug)]
pub enum Error {
    NullPointer,
    InvalidParameter,
    Specific(String),
}
#[path = "ffi_bindings.rs"]
mod ffi_bindings;
pub use ffi_bindings::*;

use libloading::Library;
use libloading::Symbol;
use once_cell::sync::OnceCell;
use std::env;

static LIB: OnceCell<Library> = OnceCell::new();

fn find_library_path() -> String {
    if let Ok(p) = env::var("BINDINGS_GENERAT_LIBRARY") {
        return p;
    }
    #[cfg(target_os = "windows")]
    return format!("{}.dll", "cudnn64_9");
    #[cfg(target_os = "macos")]
    return format!("lib{}.dylib", "cudnn64_9");
    #[cfg(all(not(target_os = "macos"), not(target_os = "windows")))]
    return format!("lib{}.so", "cudnn64_9");
}

pub fn get_lib() -> &'static Library {
    LIB.get_or_init(|| {
        let path = find_library_path();
        unsafe { Library::new(path).expect("Failed to load shared library") }
    })
}

type __va_start_fn = unsafe extern "C" fn(arg1: *mut *mut ::core::ffi::c_char) -> ();
pub unsafe fn __va_start(arg1: *mut *mut ::core::ffi::c_char) -> () {
    let sym: Symbol<__va_start_fn> =
        unsafe { get_lib().get(b"__va_start\0").expect("Missing symbol") };
    unsafe { (*sym)(arg1) }
}

type __security_init_cookie_fn = unsafe extern "C" fn() -> ();
pub unsafe fn __security_init_cookie() -> () {
    let sym: Symbol<__security_init_cookie_fn> = unsafe {
        get_lib()
            .get(b"__security_init_cookie\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)() }
}

type __security_check_cookie_fn = unsafe extern "C" fn(_StackCookie: usize) -> ();
pub unsafe fn __security_check_cookie(_StackCookie: usize) -> () {
    let sym: Symbol<__security_check_cookie_fn> = unsafe {
        get_lib()
            .get(b"__security_check_cookie\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(_StackCookie) }
}

type __report_gsfailure_fn = unsafe extern "C" fn(_StackCookie: usize) -> !;
pub unsafe fn __report_gsfailure(_StackCookie: usize) -> ! {
    let sym: Symbol<__report_gsfailure_fn> = unsafe {
        get_lib()
            .get(b"__report_gsfailure\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(_StackCookie) }
}

type cudaDeviceReset_fn = unsafe extern "C" fn() -> cudaError_t;
pub unsafe fn cudaDeviceReset() -> cudaError_t {
    let sym: Symbol<cudaDeviceReset_fn> =
        unsafe { get_lib().get(b"cudaDeviceReset\0").expect("Missing symbol") };
    unsafe { (*sym)() }
}

type cudaDeviceSynchronize_fn = unsafe extern "C" fn() -> cudaError_t;
pub unsafe fn cudaDeviceSynchronize() -> cudaError_t {
    let sym: Symbol<cudaDeviceSynchronize_fn> = unsafe {
        get_lib()
            .get(b"cudaDeviceSynchronize\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)() }
}

type cudaDeviceSetLimit_fn = unsafe extern "C" fn(limit: cudaLimit, value: usize) -> cudaError_t;
pub unsafe fn cudaDeviceSetLimit(limit: cudaLimit, value: usize) -> cudaError_t {
    let sym: Symbol<cudaDeviceSetLimit_fn> = unsafe {
        get_lib()
            .get(b"cudaDeviceSetLimit\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(limit, value) }
}

type cudaDeviceGetLimit_fn =
    unsafe extern "C" fn(pValue: *mut usize, limit: cudaLimit) -> cudaError_t;
pub unsafe fn cudaDeviceGetLimit(pValue: *mut usize, limit: cudaLimit) -> cudaError_t {
    let sym: Symbol<cudaDeviceGetLimit_fn> = unsafe {
        get_lib()
            .get(b"cudaDeviceGetLimit\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(pValue, limit) }
}

type cudaDeviceGetTexture1DLinearMaxWidth_fn = unsafe extern "C" fn(
    maxWidthInElements: *mut usize,
    fmtDesc: *const cudaChannelFormatDesc,
    device: ::core::ffi::c_int,
) -> cudaError_t;
pub unsafe fn cudaDeviceGetTexture1DLinearMaxWidth(
    maxWidthInElements: *mut usize,
    fmtDesc: *const cudaChannelFormatDesc,
    device: ::core::ffi::c_int,
) -> cudaError_t {
    let sym: Symbol<cudaDeviceGetTexture1DLinearMaxWidth_fn> = unsafe {
        get_lib()
            .get(b"cudaDeviceGetTexture1DLinearMaxWidth\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(maxWidthInElements, fmtDesc, device) }
}

type cudaDeviceGetCacheConfig_fn =
    unsafe extern "C" fn(pCacheConfig: *mut cudaFuncCache) -> cudaError_t;
pub unsafe fn cudaDeviceGetCacheConfig(pCacheConfig: *mut cudaFuncCache) -> cudaError_t {
    let sym: Symbol<cudaDeviceGetCacheConfig_fn> = unsafe {
        get_lib()
            .get(b"cudaDeviceGetCacheConfig\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(pCacheConfig) }
}

type cudaDeviceGetStreamPriorityRange_fn = unsafe extern "C" fn(
    leastPriority: *mut ::core::ffi::c_int,
    greatestPriority: *mut ::core::ffi::c_int,
) -> cudaError_t;
pub unsafe fn cudaDeviceGetStreamPriorityRange(
    leastPriority: *mut ::core::ffi::c_int,
    greatestPriority: *mut ::core::ffi::c_int,
) -> cudaError_t {
    let sym: Symbol<cudaDeviceGetStreamPriorityRange_fn> = unsafe {
        get_lib()
            .get(b"cudaDeviceGetStreamPriorityRange\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(leastPriority, greatestPriority) }
}

type cudaDeviceSetCacheConfig_fn = unsafe extern "C" fn(cacheConfig: cudaFuncCache) -> cudaError_t;
pub unsafe fn cudaDeviceSetCacheConfig(cacheConfig: cudaFuncCache) -> cudaError_t {
    let sym: Symbol<cudaDeviceSetCacheConfig_fn> = unsafe {
        get_lib()
            .get(b"cudaDeviceSetCacheConfig\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(cacheConfig) }
}

type cudaDeviceGetByPCIBusId_fn = unsafe extern "C" fn(
    device: *mut ::core::ffi::c_int,
    pciBusId: *const ::core::ffi::c_char,
) -> cudaError_t;
pub unsafe fn cudaDeviceGetByPCIBusId(
    device: *mut ::core::ffi::c_int,
    pciBusId: *const ::core::ffi::c_char,
) -> cudaError_t {
    let sym: Symbol<cudaDeviceGetByPCIBusId_fn> = unsafe {
        get_lib()
            .get(b"cudaDeviceGetByPCIBusId\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(device, pciBusId) }
}

type cudaDeviceGetPCIBusId_fn = unsafe extern "C" fn(
    pciBusId: *mut ::core::ffi::c_char,
    len: ::core::ffi::c_int,
    device: ::core::ffi::c_int,
) -> cudaError_t;
pub unsafe fn cudaDeviceGetPCIBusId(
    pciBusId: *mut ::core::ffi::c_char,
    len: ::core::ffi::c_int,
    device: ::core::ffi::c_int,
) -> cudaError_t {
    let sym: Symbol<cudaDeviceGetPCIBusId_fn> = unsafe {
        get_lib()
            .get(b"cudaDeviceGetPCIBusId\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(pciBusId, len, device) }
}

type cudaIpcGetEventHandle_fn =
    unsafe extern "C" fn(handle: *mut cudaIpcEventHandle_t, event: cudaEvent_t) -> cudaError_t;
pub unsafe fn cudaIpcGetEventHandle(
    handle: *mut cudaIpcEventHandle_t,
    event: cudaEvent_t,
) -> cudaError_t {
    let sym: Symbol<cudaIpcGetEventHandle_fn> = unsafe {
        get_lib()
            .get(b"cudaIpcGetEventHandle\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(handle, event) }
}

type cudaIpcOpenEventHandle_fn =
    unsafe extern "C" fn(event: *mut cudaEvent_t, handle: cudaIpcEventHandle_t) -> cudaError_t;
pub unsafe fn cudaIpcOpenEventHandle(
    event: *mut cudaEvent_t,
    handle: cudaIpcEventHandle_t,
) -> cudaError_t {
    let sym: Symbol<cudaIpcOpenEventHandle_fn> = unsafe {
        get_lib()
            .get(b"cudaIpcOpenEventHandle\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(event, handle) }
}

type cudaIpcGetMemHandle_fn = unsafe extern "C" fn(
    handle: *mut cudaIpcMemHandle_t,
    devPtr: *mut ::core::ffi::c_void,
) -> cudaError_t;
pub unsafe fn cudaIpcGetMemHandle(
    handle: *mut cudaIpcMemHandle_t,
    devPtr: *mut ::core::ffi::c_void,
) -> cudaError_t {
    let sym: Symbol<cudaIpcGetMemHandle_fn> = unsafe {
        get_lib()
            .get(b"cudaIpcGetMemHandle\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(handle, devPtr) }
}

type cudaIpcOpenMemHandle_fn = unsafe extern "C" fn(
    devPtr: *mut *mut ::core::ffi::c_void,
    handle: cudaIpcMemHandle_t,
    flags: ::core::ffi::c_uint,
) -> cudaError_t;
pub unsafe fn cudaIpcOpenMemHandle(
    devPtr: *mut *mut ::core::ffi::c_void,
    handle: cudaIpcMemHandle_t,
    flags: ::core::ffi::c_uint,
) -> cudaError_t {
    let sym: Symbol<cudaIpcOpenMemHandle_fn> = unsafe {
        get_lib()
            .get(b"cudaIpcOpenMemHandle\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(devPtr, handle, flags) }
}

type cudaIpcCloseMemHandle_fn =
    unsafe extern "C" fn(devPtr: *mut ::core::ffi::c_void) -> cudaError_t;
pub unsafe fn cudaIpcCloseMemHandle(devPtr: *mut ::core::ffi::c_void) -> cudaError_t {
    let sym: Symbol<cudaIpcCloseMemHandle_fn> = unsafe {
        get_lib()
            .get(b"cudaIpcCloseMemHandle\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(devPtr) }
}

type cudaDeviceFlushGPUDirectRDMAWrites_fn = unsafe extern "C" fn(
    target: cudaFlushGPUDirectRDMAWritesTarget,
    scope: cudaFlushGPUDirectRDMAWritesScope,
) -> cudaError_t;
pub unsafe fn cudaDeviceFlushGPUDirectRDMAWrites(
    target: cudaFlushGPUDirectRDMAWritesTarget,
    scope: cudaFlushGPUDirectRDMAWritesScope,
) -> cudaError_t {
    let sym: Symbol<cudaDeviceFlushGPUDirectRDMAWrites_fn> = unsafe {
        get_lib()
            .get(b"cudaDeviceFlushGPUDirectRDMAWrites\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(target, scope) }
}

type cudaDeviceRegisterAsyncNotification_fn = unsafe extern "C" fn(
    device: ::core::ffi::c_int,
    callbackFunc: cudaAsyncCallback,
    userData: *mut ::core::ffi::c_void,
    callback: *mut cudaAsyncCallbackHandle_t,
) -> cudaError_t;
pub unsafe fn cudaDeviceRegisterAsyncNotification(
    device: ::core::ffi::c_int,
    callbackFunc: cudaAsyncCallback,
    userData: *mut ::core::ffi::c_void,
    callback: *mut cudaAsyncCallbackHandle_t,
) -> cudaError_t {
    let sym: Symbol<cudaDeviceRegisterAsyncNotification_fn> = unsafe {
        get_lib()
            .get(b"cudaDeviceRegisterAsyncNotification\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(device, callbackFunc, userData, callback) }
}

type cudaDeviceUnregisterAsyncNotification_fn = unsafe extern "C" fn(
    device: ::core::ffi::c_int,
    callback: cudaAsyncCallbackHandle_t,
) -> cudaError_t;
pub unsafe fn cudaDeviceUnregisterAsyncNotification(
    device: ::core::ffi::c_int,
    callback: cudaAsyncCallbackHandle_t,
) -> cudaError_t {
    let sym: Symbol<cudaDeviceUnregisterAsyncNotification_fn> = unsafe {
        get_lib()
            .get(b"cudaDeviceUnregisterAsyncNotification\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(device, callback) }
}

type cudaDeviceGetSharedMemConfig_fn =
    unsafe extern "C" fn(pConfig: *mut cudaSharedMemConfig) -> cudaError_t;
pub unsafe fn cudaDeviceGetSharedMemConfig(pConfig: *mut cudaSharedMemConfig) -> cudaError_t {
    let sym: Symbol<cudaDeviceGetSharedMemConfig_fn> = unsafe {
        get_lib()
            .get(b"cudaDeviceGetSharedMemConfig\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(pConfig) }
}

type cudaDeviceSetSharedMemConfig_fn =
    unsafe extern "C" fn(config: cudaSharedMemConfig) -> cudaError_t;
pub unsafe fn cudaDeviceSetSharedMemConfig(config: cudaSharedMemConfig) -> cudaError_t {
    let sym: Symbol<cudaDeviceSetSharedMemConfig_fn> = unsafe {
        get_lib()
            .get(b"cudaDeviceSetSharedMemConfig\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(config) }
}

type cudaGetLastError_fn = unsafe extern "C" fn() -> cudaError_t;
pub unsafe fn cudaGetLastError() -> cudaError_t {
    let sym: Symbol<cudaGetLastError_fn> = unsafe {
        get_lib()
            .get(b"cudaGetLastError\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)() }
}

type cudaPeekAtLastError_fn = unsafe extern "C" fn() -> cudaError_t;
pub unsafe fn cudaPeekAtLastError() -> cudaError_t {
    let sym: Symbol<cudaPeekAtLastError_fn> = unsafe {
        get_lib()
            .get(b"cudaPeekAtLastError\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)() }
}

type cudaGetErrorName_fn = unsafe extern "C" fn(error: cudaError_t) -> *const ::core::ffi::c_char;
pub unsafe fn cudaGetErrorName(error: cudaError_t) -> *const ::core::ffi::c_char {
    let sym: Symbol<cudaGetErrorName_fn> = unsafe {
        get_lib()
            .get(b"cudaGetErrorName\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(error) }
}

type cudaGetErrorString_fn = unsafe extern "C" fn(error: cudaError_t) -> *const ::core::ffi::c_char;
pub unsafe fn cudaGetErrorString(error: cudaError_t) -> *const ::core::ffi::c_char {
    let sym: Symbol<cudaGetErrorString_fn> = unsafe {
        get_lib()
            .get(b"cudaGetErrorString\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(error) }
}

type cudaGetDeviceCount_fn = unsafe extern "C" fn(count: *mut ::core::ffi::c_int) -> cudaError_t;
pub unsafe fn cudaGetDeviceCount(count: *mut ::core::ffi::c_int) -> cudaError_t {
    let sym: Symbol<cudaGetDeviceCount_fn> = unsafe {
        get_lib()
            .get(b"cudaGetDeviceCount\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(count) }
}

type cudaGetDeviceProperties_fn =
    unsafe extern "C" fn(prop: *mut cudaDeviceProp, device: ::core::ffi::c_int) -> cudaError_t;
pub unsafe fn cudaGetDeviceProperties(
    prop: *mut cudaDeviceProp,
    device: ::core::ffi::c_int,
) -> cudaError_t {
    let sym: Symbol<cudaGetDeviceProperties_fn> = unsafe {
        get_lib()
            .get(b"cudaGetDeviceProperties\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(prop, device) }
}

type cudaDeviceGetAttribute_fn = unsafe extern "C" fn(
    value: *mut ::core::ffi::c_int,
    attr: cudaDeviceAttr,
    device: ::core::ffi::c_int,
) -> cudaError_t;
pub unsafe fn cudaDeviceGetAttribute(
    value: *mut ::core::ffi::c_int,
    attr: cudaDeviceAttr,
    device: ::core::ffi::c_int,
) -> cudaError_t {
    let sym: Symbol<cudaDeviceGetAttribute_fn> = unsafe {
        get_lib()
            .get(b"cudaDeviceGetAttribute\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(value, attr, device) }
}

type cudaDeviceGetHostAtomicCapabilities_fn = unsafe extern "C" fn(
    capabilities: *mut ::core::ffi::c_uint,
    operations: *const cudaAtomicOperation,
    count: ::core::ffi::c_uint,
    device: ::core::ffi::c_int,
) -> cudaError_t;
pub unsafe fn cudaDeviceGetHostAtomicCapabilities(
    capabilities: *mut ::core::ffi::c_uint,
    operations: *const cudaAtomicOperation,
    count: ::core::ffi::c_uint,
    device: ::core::ffi::c_int,
) -> cudaError_t {
    let sym: Symbol<cudaDeviceGetHostAtomicCapabilities_fn> = unsafe {
        get_lib()
            .get(b"cudaDeviceGetHostAtomicCapabilities\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(capabilities, operations, count, device) }
}

type cudaDeviceGetDefaultMemPool_fn =
    unsafe extern "C" fn(memPool: *mut cudaMemPool_t, device: ::core::ffi::c_int) -> cudaError_t;
pub unsafe fn cudaDeviceGetDefaultMemPool(
    memPool: *mut cudaMemPool_t,
    device: ::core::ffi::c_int,
) -> cudaError_t {
    let sym: Symbol<cudaDeviceGetDefaultMemPool_fn> = unsafe {
        get_lib()
            .get(b"cudaDeviceGetDefaultMemPool\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(memPool, device) }
}

type cudaDeviceSetMemPool_fn =
    unsafe extern "C" fn(device: ::core::ffi::c_int, memPool: cudaMemPool_t) -> cudaError_t;
pub unsafe fn cudaDeviceSetMemPool(
    device: ::core::ffi::c_int,
    memPool: cudaMemPool_t,
) -> cudaError_t {
    let sym: Symbol<cudaDeviceSetMemPool_fn> = unsafe {
        get_lib()
            .get(b"cudaDeviceSetMemPool\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(device, memPool) }
}

type cudaDeviceGetMemPool_fn =
    unsafe extern "C" fn(memPool: *mut cudaMemPool_t, device: ::core::ffi::c_int) -> cudaError_t;
pub unsafe fn cudaDeviceGetMemPool(
    memPool: *mut cudaMemPool_t,
    device: ::core::ffi::c_int,
) -> cudaError_t {
    let sym: Symbol<cudaDeviceGetMemPool_fn> = unsafe {
        get_lib()
            .get(b"cudaDeviceGetMemPool\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(memPool, device) }
}

type cudaDeviceGetNvSciSyncAttributes_fn = unsafe extern "C" fn(
    nvSciSyncAttrList: *mut ::core::ffi::c_void,
    device: ::core::ffi::c_int,
    flags: ::core::ffi::c_int,
) -> cudaError_t;
pub unsafe fn cudaDeviceGetNvSciSyncAttributes(
    nvSciSyncAttrList: *mut ::core::ffi::c_void,
    device: ::core::ffi::c_int,
    flags: ::core::ffi::c_int,
) -> cudaError_t {
    let sym: Symbol<cudaDeviceGetNvSciSyncAttributes_fn> = unsafe {
        get_lib()
            .get(b"cudaDeviceGetNvSciSyncAttributes\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(nvSciSyncAttrList, device, flags) }
}

type cudaDeviceGetP2PAttribute_fn = unsafe extern "C" fn(
    value: *mut ::core::ffi::c_int,
    attr: cudaDeviceP2PAttr,
    srcDevice: ::core::ffi::c_int,
    dstDevice: ::core::ffi::c_int,
) -> cudaError_t;
pub unsafe fn cudaDeviceGetP2PAttribute(
    value: *mut ::core::ffi::c_int,
    attr: cudaDeviceP2PAttr,
    srcDevice: ::core::ffi::c_int,
    dstDevice: ::core::ffi::c_int,
) -> cudaError_t {
    let sym: Symbol<cudaDeviceGetP2PAttribute_fn> = unsafe {
        get_lib()
            .get(b"cudaDeviceGetP2PAttribute\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(value, attr, srcDevice, dstDevice) }
}

type cudaDeviceGetP2PAtomicCapabilities_fn = unsafe extern "C" fn(
    capabilities: *mut ::core::ffi::c_uint,
    operations: *const cudaAtomicOperation,
    count: ::core::ffi::c_uint,
    srcDevice: ::core::ffi::c_int,
    dstDevice: ::core::ffi::c_int,
) -> cudaError_t;
pub unsafe fn cudaDeviceGetP2PAtomicCapabilities(
    capabilities: *mut ::core::ffi::c_uint,
    operations: *const cudaAtomicOperation,
    count: ::core::ffi::c_uint,
    srcDevice: ::core::ffi::c_int,
    dstDevice: ::core::ffi::c_int,
) -> cudaError_t {
    let sym: Symbol<cudaDeviceGetP2PAtomicCapabilities_fn> = unsafe {
        get_lib()
            .get(b"cudaDeviceGetP2PAtomicCapabilities\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(capabilities, operations, count, srcDevice, dstDevice) }
}

type cudaChooseDevice_fn = unsafe extern "C" fn(
    device: *mut ::core::ffi::c_int,
    prop: *const cudaDeviceProp,
) -> cudaError_t;
pub unsafe fn cudaChooseDevice(
    device: *mut ::core::ffi::c_int,
    prop: *const cudaDeviceProp,
) -> cudaError_t {
    let sym: Symbol<cudaChooseDevice_fn> = unsafe {
        get_lib()
            .get(b"cudaChooseDevice\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(device, prop) }
}

type cudaInitDevice_fn = unsafe extern "C" fn(
    device: ::core::ffi::c_int,
    deviceFlags: ::core::ffi::c_uint,
    flags: ::core::ffi::c_uint,
) -> cudaError_t;
pub unsafe fn cudaInitDevice(
    device: ::core::ffi::c_int,
    deviceFlags: ::core::ffi::c_uint,
    flags: ::core::ffi::c_uint,
) -> cudaError_t {
    let sym: Symbol<cudaInitDevice_fn> =
        unsafe { get_lib().get(b"cudaInitDevice\0").expect("Missing symbol") };
    unsafe { (*sym)(device, deviceFlags, flags) }
}

type cudaSetDevice_fn = unsafe extern "C" fn(device: ::core::ffi::c_int) -> cudaError_t;
pub unsafe fn cudaSetDevice(device: ::core::ffi::c_int) -> cudaError_t {
    let sym: Symbol<cudaSetDevice_fn> =
        unsafe { get_lib().get(b"cudaSetDevice\0").expect("Missing symbol") };
    unsafe { (*sym)(device) }
}

type cudaGetDevice_fn = unsafe extern "C" fn(device: *mut ::core::ffi::c_int) -> cudaError_t;
pub unsafe fn cudaGetDevice(device: *mut ::core::ffi::c_int) -> cudaError_t {
    let sym: Symbol<cudaGetDevice_fn> =
        unsafe { get_lib().get(b"cudaGetDevice\0").expect("Missing symbol") };
    unsafe { (*sym)(device) }
}

type cudaSetValidDevices_fn = unsafe extern "C" fn(
    device_arr: *mut ::core::ffi::c_int,
    len: ::core::ffi::c_int,
) -> cudaError_t;
pub unsafe fn cudaSetValidDevices(
    device_arr: *mut ::core::ffi::c_int,
    len: ::core::ffi::c_int,
) -> cudaError_t {
    let sym: Symbol<cudaSetValidDevices_fn> = unsafe {
        get_lib()
            .get(b"cudaSetValidDevices\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(device_arr, len) }
}

type cudaSetDeviceFlags_fn = unsafe extern "C" fn(flags: ::core::ffi::c_uint) -> cudaError_t;
pub unsafe fn cudaSetDeviceFlags(flags: ::core::ffi::c_uint) -> cudaError_t {
    let sym: Symbol<cudaSetDeviceFlags_fn> = unsafe {
        get_lib()
            .get(b"cudaSetDeviceFlags\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(flags) }
}

type cudaGetDeviceFlags_fn = unsafe extern "C" fn(flags: *mut ::core::ffi::c_uint) -> cudaError_t;
pub unsafe fn cudaGetDeviceFlags(flags: *mut ::core::ffi::c_uint) -> cudaError_t {
    let sym: Symbol<cudaGetDeviceFlags_fn> = unsafe {
        get_lib()
            .get(b"cudaGetDeviceFlags\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(flags) }
}

type cudaStreamCreate_fn = unsafe extern "C" fn(pStream: *mut cudaStream_t) -> cudaError_t;
pub unsafe fn cudaStreamCreate(pStream: *mut cudaStream_t) -> cudaError_t {
    let sym: Symbol<cudaStreamCreate_fn> = unsafe {
        get_lib()
            .get(b"cudaStreamCreate\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(pStream) }
}

type cudaStreamCreateWithFlags_fn =
    unsafe extern "C" fn(pStream: *mut cudaStream_t, flags: ::core::ffi::c_uint) -> cudaError_t;
pub unsafe fn cudaStreamCreateWithFlags(
    pStream: *mut cudaStream_t,
    flags: ::core::ffi::c_uint,
) -> cudaError_t {
    let sym: Symbol<cudaStreamCreateWithFlags_fn> = unsafe {
        get_lib()
            .get(b"cudaStreamCreateWithFlags\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(pStream, flags) }
}

type cudaStreamCreateWithPriority_fn = unsafe extern "C" fn(
    pStream: *mut cudaStream_t,
    flags: ::core::ffi::c_uint,
    priority: ::core::ffi::c_int,
) -> cudaError_t;
pub unsafe fn cudaStreamCreateWithPriority(
    pStream: *mut cudaStream_t,
    flags: ::core::ffi::c_uint,
    priority: ::core::ffi::c_int,
) -> cudaError_t {
    let sym: Symbol<cudaStreamCreateWithPriority_fn> = unsafe {
        get_lib()
            .get(b"cudaStreamCreateWithPriority\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(pStream, flags, priority) }
}

type cudaStreamGetPriority_fn =
    unsafe extern "C" fn(hStream: cudaStream_t, priority: *mut ::core::ffi::c_int) -> cudaError_t;
pub unsafe fn cudaStreamGetPriority(
    hStream: cudaStream_t,
    priority: *mut ::core::ffi::c_int,
) -> cudaError_t {
    let sym: Symbol<cudaStreamGetPriority_fn> = unsafe {
        get_lib()
            .get(b"cudaStreamGetPriority\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(hStream, priority) }
}

type cudaStreamGetFlags_fn =
    unsafe extern "C" fn(hStream: cudaStream_t, flags: *mut ::core::ffi::c_uint) -> cudaError_t;
pub unsafe fn cudaStreamGetFlags(
    hStream: cudaStream_t,
    flags: *mut ::core::ffi::c_uint,
) -> cudaError_t {
    let sym: Symbol<cudaStreamGetFlags_fn> = unsafe {
        get_lib()
            .get(b"cudaStreamGetFlags\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(hStream, flags) }
}

type cudaStreamGetId_fn = unsafe extern "C" fn(
    hStream: cudaStream_t,
    streamId: *mut ::core::ffi::c_ulonglong,
) -> cudaError_t;
pub unsafe fn cudaStreamGetId(
    hStream: cudaStream_t,
    streamId: *mut ::core::ffi::c_ulonglong,
) -> cudaError_t {
    let sym: Symbol<cudaStreamGetId_fn> =
        unsafe { get_lib().get(b"cudaStreamGetId\0").expect("Missing symbol") };
    unsafe { (*sym)(hStream, streamId) }
}

type cudaStreamGetDevice_fn =
    unsafe extern "C" fn(hStream: cudaStream_t, device: *mut ::core::ffi::c_int) -> cudaError_t;
pub unsafe fn cudaStreamGetDevice(
    hStream: cudaStream_t,
    device: *mut ::core::ffi::c_int,
) -> cudaError_t {
    let sym: Symbol<cudaStreamGetDevice_fn> = unsafe {
        get_lib()
            .get(b"cudaStreamGetDevice\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(hStream, device) }
}

type cudaCtxResetPersistingL2Cache_fn = unsafe extern "C" fn() -> cudaError_t;
pub unsafe fn cudaCtxResetPersistingL2Cache() -> cudaError_t {
    let sym: Symbol<cudaCtxResetPersistingL2Cache_fn> = unsafe {
        get_lib()
            .get(b"cudaCtxResetPersistingL2Cache\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)() }
}

type cudaStreamCopyAttributes_fn =
    unsafe extern "C" fn(dst: cudaStream_t, src: cudaStream_t) -> cudaError_t;
pub unsafe fn cudaStreamCopyAttributes(dst: cudaStream_t, src: cudaStream_t) -> cudaError_t {
    let sym: Symbol<cudaStreamCopyAttributes_fn> = unsafe {
        get_lib()
            .get(b"cudaStreamCopyAttributes\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(dst, src) }
}

type cudaStreamGetAttribute_fn = unsafe extern "C" fn(
    hStream: cudaStream_t,
    attr: cudaLaunchAttributeID,
    value_out: *mut cudaLaunchAttributeValue,
) -> cudaError_t;
pub unsafe fn cudaStreamGetAttribute(
    hStream: cudaStream_t,
    attr: cudaLaunchAttributeID,
    value_out: *mut cudaLaunchAttributeValue,
) -> cudaError_t {
    let sym: Symbol<cudaStreamGetAttribute_fn> = unsafe {
        get_lib()
            .get(b"cudaStreamGetAttribute\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(hStream, attr, value_out) }
}

type cudaStreamSetAttribute_fn = unsafe extern "C" fn(
    hStream: cudaStream_t,
    attr: cudaLaunchAttributeID,
    value: *const cudaLaunchAttributeValue,
) -> cudaError_t;
pub unsafe fn cudaStreamSetAttribute(
    hStream: cudaStream_t,
    attr: cudaLaunchAttributeID,
    value: *const cudaLaunchAttributeValue,
) -> cudaError_t {
    let sym: Symbol<cudaStreamSetAttribute_fn> = unsafe {
        get_lib()
            .get(b"cudaStreamSetAttribute\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(hStream, attr, value) }
}

type cudaStreamDestroy_fn = unsafe extern "C" fn(stream: cudaStream_t) -> cudaError_t;
pub unsafe fn cudaStreamDestroy(stream: cudaStream_t) -> cudaError_t {
    let sym: Symbol<cudaStreamDestroy_fn> = unsafe {
        get_lib()
            .get(b"cudaStreamDestroy\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(stream) }
}

type cudaStreamWaitEvent_fn = unsafe extern "C" fn(
    stream: cudaStream_t,
    event: cudaEvent_t,
    flags: ::core::ffi::c_uint,
) -> cudaError_t;
pub unsafe fn cudaStreamWaitEvent(
    stream: cudaStream_t,
    event: cudaEvent_t,
    flags: ::core::ffi::c_uint,
) -> cudaError_t {
    let sym: Symbol<cudaStreamWaitEvent_fn> = unsafe {
        get_lib()
            .get(b"cudaStreamWaitEvent\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(stream, event, flags) }
}

type cudaStreamAddCallback_fn = unsafe extern "C" fn(
    stream: cudaStream_t,
    callback: cudaStreamCallback_t,
    userData: *mut ::core::ffi::c_void,
    flags: ::core::ffi::c_uint,
) -> cudaError_t;
pub unsafe fn cudaStreamAddCallback(
    stream: cudaStream_t,
    callback: cudaStreamCallback_t,
    userData: *mut ::core::ffi::c_void,
    flags: ::core::ffi::c_uint,
) -> cudaError_t {
    let sym: Symbol<cudaStreamAddCallback_fn> = unsafe {
        get_lib()
            .get(b"cudaStreamAddCallback\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(stream, callback, userData, flags) }
}

type cudaStreamSynchronize_fn = unsafe extern "C" fn(stream: cudaStream_t) -> cudaError_t;
pub unsafe fn cudaStreamSynchronize(stream: cudaStream_t) -> cudaError_t {
    let sym: Symbol<cudaStreamSynchronize_fn> = unsafe {
        get_lib()
            .get(b"cudaStreamSynchronize\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(stream) }
}

type cudaStreamQuery_fn = unsafe extern "C" fn(stream: cudaStream_t) -> cudaError_t;
pub unsafe fn cudaStreamQuery(stream: cudaStream_t) -> cudaError_t {
    let sym: Symbol<cudaStreamQuery_fn> =
        unsafe { get_lib().get(b"cudaStreamQuery\0").expect("Missing symbol") };
    unsafe { (*sym)(stream) }
}

type cudaStreamAttachMemAsync_fn = unsafe extern "C" fn(
    stream: cudaStream_t,
    devPtr: *mut ::core::ffi::c_void,
    length: usize,
    flags: ::core::ffi::c_uint,
) -> cudaError_t;
pub unsafe fn cudaStreamAttachMemAsync(
    stream: cudaStream_t,
    devPtr: *mut ::core::ffi::c_void,
    length: usize,
    flags: ::core::ffi::c_uint,
) -> cudaError_t {
    let sym: Symbol<cudaStreamAttachMemAsync_fn> = unsafe {
        get_lib()
            .get(b"cudaStreamAttachMemAsync\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(stream, devPtr, length, flags) }
}

type cudaStreamBeginCapture_fn =
    unsafe extern "C" fn(stream: cudaStream_t, mode: cudaStreamCaptureMode) -> cudaError_t;
pub unsafe fn cudaStreamBeginCapture(
    stream: cudaStream_t,
    mode: cudaStreamCaptureMode,
) -> cudaError_t {
    let sym: Symbol<cudaStreamBeginCapture_fn> = unsafe {
        get_lib()
            .get(b"cudaStreamBeginCapture\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(stream, mode) }
}

type cudaStreamBeginCaptureToGraph_fn = unsafe extern "C" fn(
    stream: cudaStream_t,
    graph: cudaGraph_t,
    dependencies: *const cudaGraphNode_t,
    dependencyData: *const cudaGraphEdgeData,
    numDependencies: usize,
    mode: cudaStreamCaptureMode,
) -> cudaError_t;
pub unsafe fn cudaStreamBeginCaptureToGraph(
    stream: cudaStream_t,
    graph: cudaGraph_t,
    dependencies: *const cudaGraphNode_t,
    dependencyData: *const cudaGraphEdgeData,
    numDependencies: usize,
    mode: cudaStreamCaptureMode,
) -> cudaError_t {
    let sym: Symbol<cudaStreamBeginCaptureToGraph_fn> = unsafe {
        get_lib()
            .get(b"cudaStreamBeginCaptureToGraph\0")
            .expect("Missing symbol")
    };
    unsafe {
        (*sym)(
            stream,
            graph,
            dependencies,
            dependencyData,
            numDependencies,
            mode,
        )
    }
}

type cudaThreadExchangeStreamCaptureMode_fn =
    unsafe extern "C" fn(mode: *mut cudaStreamCaptureMode) -> cudaError_t;
pub unsafe fn cudaThreadExchangeStreamCaptureMode(mode: *mut cudaStreamCaptureMode) -> cudaError_t {
    let sym: Symbol<cudaThreadExchangeStreamCaptureMode_fn> = unsafe {
        get_lib()
            .get(b"cudaThreadExchangeStreamCaptureMode\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(mode) }
}

type cudaStreamEndCapture_fn =
    unsafe extern "C" fn(stream: cudaStream_t, pGraph: *mut cudaGraph_t) -> cudaError_t;
pub unsafe fn cudaStreamEndCapture(stream: cudaStream_t, pGraph: *mut cudaGraph_t) -> cudaError_t {
    let sym: Symbol<cudaStreamEndCapture_fn> = unsafe {
        get_lib()
            .get(b"cudaStreamEndCapture\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(stream, pGraph) }
}

type cudaStreamIsCapturing_fn = unsafe extern "C" fn(
    stream: cudaStream_t,
    pCaptureStatus: *mut cudaStreamCaptureStatus,
) -> cudaError_t;
pub unsafe fn cudaStreamIsCapturing(
    stream: cudaStream_t,
    pCaptureStatus: *mut cudaStreamCaptureStatus,
) -> cudaError_t {
    let sym: Symbol<cudaStreamIsCapturing_fn> = unsafe {
        get_lib()
            .get(b"cudaStreamIsCapturing\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(stream, pCaptureStatus) }
}

type cudaStreamGetCaptureInfo_fn = unsafe extern "C" fn(
    stream: cudaStream_t,
    captureStatus_out: *mut cudaStreamCaptureStatus,
    id_out: *mut ::core::ffi::c_ulonglong,
    graph_out: *mut cudaGraph_t,
    dependencies_out: *mut *const cudaGraphNode_t,
    edgeData_out: *mut *const cudaGraphEdgeData,
    numDependencies_out: *mut usize,
) -> cudaError_t;
pub unsafe fn cudaStreamGetCaptureInfo(
    stream: cudaStream_t,
    captureStatus_out: *mut cudaStreamCaptureStatus,
    id_out: *mut ::core::ffi::c_ulonglong,
    graph_out: *mut cudaGraph_t,
    dependencies_out: *mut *const cudaGraphNode_t,
    edgeData_out: *mut *const cudaGraphEdgeData,
    numDependencies_out: *mut usize,
) -> cudaError_t {
    let sym: Symbol<cudaStreamGetCaptureInfo_fn> = unsafe {
        get_lib()
            .get(b"cudaStreamGetCaptureInfo\0")
            .expect("Missing symbol")
    };
    unsafe {
        (*sym)(
            stream,
            captureStatus_out,
            id_out,
            graph_out,
            dependencies_out,
            edgeData_out,
            numDependencies_out,
        )
    }
}

type cudaStreamUpdateCaptureDependencies_fn = unsafe extern "C" fn(
    stream: cudaStream_t,
    dependencies: *mut cudaGraphNode_t,
    dependencyData: *const cudaGraphEdgeData,
    numDependencies: usize,
    flags: ::core::ffi::c_uint,
) -> cudaError_t;
pub unsafe fn cudaStreamUpdateCaptureDependencies(
    stream: cudaStream_t,
    dependencies: *mut cudaGraphNode_t,
    dependencyData: *const cudaGraphEdgeData,
    numDependencies: usize,
    flags: ::core::ffi::c_uint,
) -> cudaError_t {
    let sym: Symbol<cudaStreamUpdateCaptureDependencies_fn> = unsafe {
        get_lib()
            .get(b"cudaStreamUpdateCaptureDependencies\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(stream, dependencies, dependencyData, numDependencies, flags) }
}

type cudaEventCreate_fn = unsafe extern "C" fn(event: *mut cudaEvent_t) -> cudaError_t;
pub unsafe fn cudaEventCreate(event: *mut cudaEvent_t) -> cudaError_t {
    let sym: Symbol<cudaEventCreate_fn> =
        unsafe { get_lib().get(b"cudaEventCreate\0").expect("Missing symbol") };
    unsafe { (*sym)(event) }
}

type cudaEventCreateWithFlags_fn =
    unsafe extern "C" fn(event: *mut cudaEvent_t, flags: ::core::ffi::c_uint) -> cudaError_t;
pub unsafe fn cudaEventCreateWithFlags(
    event: *mut cudaEvent_t,
    flags: ::core::ffi::c_uint,
) -> cudaError_t {
    let sym: Symbol<cudaEventCreateWithFlags_fn> = unsafe {
        get_lib()
            .get(b"cudaEventCreateWithFlags\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(event, flags) }
}

type cudaEventRecord_fn =
    unsafe extern "C" fn(event: cudaEvent_t, stream: cudaStream_t) -> cudaError_t;
pub unsafe fn cudaEventRecord(event: cudaEvent_t, stream: cudaStream_t) -> cudaError_t {
    let sym: Symbol<cudaEventRecord_fn> =
        unsafe { get_lib().get(b"cudaEventRecord\0").expect("Missing symbol") };
    unsafe { (*sym)(event, stream) }
}

type cudaEventRecordWithFlags_fn = unsafe extern "C" fn(
    event: cudaEvent_t,
    stream: cudaStream_t,
    flags: ::core::ffi::c_uint,
) -> cudaError_t;
pub unsafe fn cudaEventRecordWithFlags(
    event: cudaEvent_t,
    stream: cudaStream_t,
    flags: ::core::ffi::c_uint,
) -> cudaError_t {
    let sym: Symbol<cudaEventRecordWithFlags_fn> = unsafe {
        get_lib()
            .get(b"cudaEventRecordWithFlags\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(event, stream, flags) }
}

type cudaEventQuery_fn = unsafe extern "C" fn(event: cudaEvent_t) -> cudaError_t;
pub unsafe fn cudaEventQuery(event: cudaEvent_t) -> cudaError_t {
    let sym: Symbol<cudaEventQuery_fn> =
        unsafe { get_lib().get(b"cudaEventQuery\0").expect("Missing symbol") };
    unsafe { (*sym)(event) }
}

type cudaEventSynchronize_fn = unsafe extern "C" fn(event: cudaEvent_t) -> cudaError_t;
pub unsafe fn cudaEventSynchronize(event: cudaEvent_t) -> cudaError_t {
    let sym: Symbol<cudaEventSynchronize_fn> = unsafe {
        get_lib()
            .get(b"cudaEventSynchronize\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(event) }
}

type cudaEventDestroy_fn = unsafe extern "C" fn(event: cudaEvent_t) -> cudaError_t;
pub unsafe fn cudaEventDestroy(event: cudaEvent_t) -> cudaError_t {
    let sym: Symbol<cudaEventDestroy_fn> = unsafe {
        get_lib()
            .get(b"cudaEventDestroy\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(event) }
}

type cudaEventElapsedTime_fn =
    unsafe extern "C" fn(ms: *mut f32, start: cudaEvent_t, end: cudaEvent_t) -> cudaError_t;
pub unsafe fn cudaEventElapsedTime(
    ms: *mut f32,
    start: cudaEvent_t,
    end: cudaEvent_t,
) -> cudaError_t {
    let sym: Symbol<cudaEventElapsedTime_fn> = unsafe {
        get_lib()
            .get(b"cudaEventElapsedTime\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(ms, start, end) }
}

type cudaImportExternalMemory_fn = unsafe extern "C" fn(
    extMem_out: *mut cudaExternalMemory_t,
    memHandleDesc: *const cudaExternalMemoryHandleDesc,
) -> cudaError_t;
pub unsafe fn cudaImportExternalMemory(
    extMem_out: *mut cudaExternalMemory_t,
    memHandleDesc: *const cudaExternalMemoryHandleDesc,
) -> cudaError_t {
    let sym: Symbol<cudaImportExternalMemory_fn> = unsafe {
        get_lib()
            .get(b"cudaImportExternalMemory\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(extMem_out, memHandleDesc) }
}

type cudaExternalMemoryGetMappedBuffer_fn = unsafe extern "C" fn(
    devPtr: *mut *mut ::core::ffi::c_void,
    extMem: cudaExternalMemory_t,
    bufferDesc: *const cudaExternalMemoryBufferDesc,
) -> cudaError_t;
pub unsafe fn cudaExternalMemoryGetMappedBuffer(
    devPtr: *mut *mut ::core::ffi::c_void,
    extMem: cudaExternalMemory_t,
    bufferDesc: *const cudaExternalMemoryBufferDesc,
) -> cudaError_t {
    let sym: Symbol<cudaExternalMemoryGetMappedBuffer_fn> = unsafe {
        get_lib()
            .get(b"cudaExternalMemoryGetMappedBuffer\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(devPtr, extMem, bufferDesc) }
}

type cudaExternalMemoryGetMappedMipmappedArray_fn = unsafe extern "C" fn(
    mipmap: *mut cudaMipmappedArray_t,
    extMem: cudaExternalMemory_t,
    mipmapDesc: *const cudaExternalMemoryMipmappedArrayDesc,
) -> cudaError_t;
pub unsafe fn cudaExternalMemoryGetMappedMipmappedArray(
    mipmap: *mut cudaMipmappedArray_t,
    extMem: cudaExternalMemory_t,
    mipmapDesc: *const cudaExternalMemoryMipmappedArrayDesc,
) -> cudaError_t {
    let sym: Symbol<cudaExternalMemoryGetMappedMipmappedArray_fn> = unsafe {
        get_lib()
            .get(b"cudaExternalMemoryGetMappedMipmappedArray\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(mipmap, extMem, mipmapDesc) }
}

type cudaDestroyExternalMemory_fn =
    unsafe extern "C" fn(extMem: cudaExternalMemory_t) -> cudaError_t;
pub unsafe fn cudaDestroyExternalMemory(extMem: cudaExternalMemory_t) -> cudaError_t {
    let sym: Symbol<cudaDestroyExternalMemory_fn> = unsafe {
        get_lib()
            .get(b"cudaDestroyExternalMemory\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(extMem) }
}

type cudaImportExternalSemaphore_fn = unsafe extern "C" fn(
    extSem_out: *mut cudaExternalSemaphore_t,
    semHandleDesc: *const cudaExternalSemaphoreHandleDesc,
) -> cudaError_t;
pub unsafe fn cudaImportExternalSemaphore(
    extSem_out: *mut cudaExternalSemaphore_t,
    semHandleDesc: *const cudaExternalSemaphoreHandleDesc,
) -> cudaError_t {
    let sym: Symbol<cudaImportExternalSemaphore_fn> = unsafe {
        get_lib()
            .get(b"cudaImportExternalSemaphore\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(extSem_out, semHandleDesc) }
}

type cudaSignalExternalSemaphoresAsync_fn = unsafe extern "C" fn(
    extSemArray: *const cudaExternalSemaphore_t,
    paramsArray: *const cudaExternalSemaphoreSignalParams,
    numExtSems: ::core::ffi::c_uint,
    stream: cudaStream_t,
) -> cudaError_t;
pub unsafe fn cudaSignalExternalSemaphoresAsync(
    extSemArray: *const cudaExternalSemaphore_t,
    paramsArray: *const cudaExternalSemaphoreSignalParams,
    numExtSems: ::core::ffi::c_uint,
    stream: cudaStream_t,
) -> cudaError_t {
    let sym: Symbol<cudaSignalExternalSemaphoresAsync_fn> = unsafe {
        get_lib()
            .get(b"cudaSignalExternalSemaphoresAsync\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(extSemArray, paramsArray, numExtSems, stream) }
}

type cudaWaitExternalSemaphoresAsync_fn = unsafe extern "C" fn(
    extSemArray: *const cudaExternalSemaphore_t,
    paramsArray: *const cudaExternalSemaphoreWaitParams,
    numExtSems: ::core::ffi::c_uint,
    stream: cudaStream_t,
) -> cudaError_t;
pub unsafe fn cudaWaitExternalSemaphoresAsync(
    extSemArray: *const cudaExternalSemaphore_t,
    paramsArray: *const cudaExternalSemaphoreWaitParams,
    numExtSems: ::core::ffi::c_uint,
    stream: cudaStream_t,
) -> cudaError_t {
    let sym: Symbol<cudaWaitExternalSemaphoresAsync_fn> = unsafe {
        get_lib()
            .get(b"cudaWaitExternalSemaphoresAsync\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(extSemArray, paramsArray, numExtSems, stream) }
}

type cudaDestroyExternalSemaphore_fn =
    unsafe extern "C" fn(extSem: cudaExternalSemaphore_t) -> cudaError_t;
pub unsafe fn cudaDestroyExternalSemaphore(extSem: cudaExternalSemaphore_t) -> cudaError_t {
    let sym: Symbol<cudaDestroyExternalSemaphore_fn> = unsafe {
        get_lib()
            .get(b"cudaDestroyExternalSemaphore\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(extSem) }
}

type cudaLaunchKernel_fn = unsafe extern "C" fn(
    func: *const ::core::ffi::c_void,
    gridDim: dim3,
    blockDim: dim3,
    args: *mut *mut ::core::ffi::c_void,
    sharedMem: usize,
    stream: cudaStream_t,
) -> cudaError_t;
pub unsafe fn cudaLaunchKernel(
    func: *const ::core::ffi::c_void,
    gridDim: dim3,
    blockDim: dim3,
    args: *mut *mut ::core::ffi::c_void,
    sharedMem: usize,
    stream: cudaStream_t,
) -> cudaError_t {
    let sym: Symbol<cudaLaunchKernel_fn> = unsafe {
        get_lib()
            .get(b"cudaLaunchKernel\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(func, gridDim, blockDim, args, sharedMem, stream) }
}

type cudaLaunchKernelExC_fn = unsafe extern "C" fn(
    config: *const cudaLaunchConfig_t,
    func: *const ::core::ffi::c_void,
    args: *mut *mut ::core::ffi::c_void,
) -> cudaError_t;
pub unsafe fn cudaLaunchKernelExC(
    config: *const cudaLaunchConfig_t,
    func: *const ::core::ffi::c_void,
    args: *mut *mut ::core::ffi::c_void,
) -> cudaError_t {
    let sym: Symbol<cudaLaunchKernelExC_fn> = unsafe {
        get_lib()
            .get(b"cudaLaunchKernelExC\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(config, func, args) }
}

type cudaLaunchCooperativeKernel_fn = unsafe extern "C" fn(
    func: *const ::core::ffi::c_void,
    gridDim: dim3,
    blockDim: dim3,
    args: *mut *mut ::core::ffi::c_void,
    sharedMem: usize,
    stream: cudaStream_t,
) -> cudaError_t;
pub unsafe fn cudaLaunchCooperativeKernel(
    func: *const ::core::ffi::c_void,
    gridDim: dim3,
    blockDim: dim3,
    args: *mut *mut ::core::ffi::c_void,
    sharedMem: usize,
    stream: cudaStream_t,
) -> cudaError_t {
    let sym: Symbol<cudaLaunchCooperativeKernel_fn> = unsafe {
        get_lib()
            .get(b"cudaLaunchCooperativeKernel\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(func, gridDim, blockDim, args, sharedMem, stream) }
}

type cudaFuncSetCacheConfig_fn = unsafe extern "C" fn(
    func: *const ::core::ffi::c_void,
    cacheConfig: cudaFuncCache,
) -> cudaError_t;
pub unsafe fn cudaFuncSetCacheConfig(
    func: *const ::core::ffi::c_void,
    cacheConfig: cudaFuncCache,
) -> cudaError_t {
    let sym: Symbol<cudaFuncSetCacheConfig_fn> = unsafe {
        get_lib()
            .get(b"cudaFuncSetCacheConfig\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(func, cacheConfig) }
}

type cudaFuncGetAttributes_fn = unsafe extern "C" fn(
    attr: *mut cudaFuncAttributes,
    func: *const ::core::ffi::c_void,
) -> cudaError_t;
pub unsafe fn cudaFuncGetAttributes(
    attr: *mut cudaFuncAttributes,
    func: *const ::core::ffi::c_void,
) -> cudaError_t {
    let sym: Symbol<cudaFuncGetAttributes_fn> = unsafe {
        get_lib()
            .get(b"cudaFuncGetAttributes\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(attr, func) }
}

type cudaFuncSetAttribute_fn = unsafe extern "C" fn(
    func: *const ::core::ffi::c_void,
    attr: cudaFuncAttribute,
    value: ::core::ffi::c_int,
) -> cudaError_t;
pub unsafe fn cudaFuncSetAttribute(
    func: *const ::core::ffi::c_void,
    attr: cudaFuncAttribute,
    value: ::core::ffi::c_int,
) -> cudaError_t {
    let sym: Symbol<cudaFuncSetAttribute_fn> = unsafe {
        get_lib()
            .get(b"cudaFuncSetAttribute\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(func, attr, value) }
}

type cudaFuncGetName_fn = unsafe extern "C" fn(
    name: *mut *const ::core::ffi::c_char,
    func: *const ::core::ffi::c_void,
) -> cudaError_t;
pub unsafe fn cudaFuncGetName(
    name: *mut *const ::core::ffi::c_char,
    func: *const ::core::ffi::c_void,
) -> cudaError_t {
    let sym: Symbol<cudaFuncGetName_fn> =
        unsafe { get_lib().get(b"cudaFuncGetName\0").expect("Missing symbol") };
    unsafe { (*sym)(name, func) }
}

type cudaFuncGetParamInfo_fn = unsafe extern "C" fn(
    func: *const ::core::ffi::c_void,
    paramIndex: usize,
    paramOffset: *mut usize,
    paramSize: *mut usize,
) -> cudaError_t;
pub unsafe fn cudaFuncGetParamInfo(
    func: *const ::core::ffi::c_void,
    paramIndex: usize,
    paramOffset: *mut usize,
    paramSize: *mut usize,
) -> cudaError_t {
    let sym: Symbol<cudaFuncGetParamInfo_fn> = unsafe {
        get_lib()
            .get(b"cudaFuncGetParamInfo\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(func, paramIndex, paramOffset, paramSize) }
}

type cudaLaunchHostFunc_fn = unsafe extern "C" fn(
    stream: cudaStream_t,
    fn_: cudaHostFn_t,
    userData: *mut ::core::ffi::c_void,
) -> cudaError_t;
pub unsafe fn cudaLaunchHostFunc(
    stream: cudaStream_t,
    fn_: cudaHostFn_t,
    userData: *mut ::core::ffi::c_void,
) -> cudaError_t {
    let sym: Symbol<cudaLaunchHostFunc_fn> = unsafe {
        get_lib()
            .get(b"cudaLaunchHostFunc\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(stream, fn_, userData) }
}

type cudaFuncSetSharedMemConfig_fn = unsafe extern "C" fn(
    func: *const ::core::ffi::c_void,
    config: cudaSharedMemConfig,
) -> cudaError_t;
pub unsafe fn cudaFuncSetSharedMemConfig(
    func: *const ::core::ffi::c_void,
    config: cudaSharedMemConfig,
) -> cudaError_t {
    let sym: Symbol<cudaFuncSetSharedMemConfig_fn> = unsafe {
        get_lib()
            .get(b"cudaFuncSetSharedMemConfig\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(func, config) }
}

type cudaOccupancyMaxActiveBlocksPerMultiprocessor_fn = unsafe extern "C" fn(
    numBlocks: *mut ::core::ffi::c_int,
    func: *const ::core::ffi::c_void,
    blockSize: ::core::ffi::c_int,
    dynamicSMemSize: usize,
) -> cudaError_t;
pub unsafe fn cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    numBlocks: *mut ::core::ffi::c_int,
    func: *const ::core::ffi::c_void,
    blockSize: ::core::ffi::c_int,
    dynamicSMemSize: usize,
) -> cudaError_t {
    let sym: Symbol<cudaOccupancyMaxActiveBlocksPerMultiprocessor_fn> = unsafe {
        get_lib()
            .get(b"cudaOccupancyMaxActiveBlocksPerMultiprocessor\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(numBlocks, func, blockSize, dynamicSMemSize) }
}

type cudaOccupancyAvailableDynamicSMemPerBlock_fn = unsafe extern "C" fn(
    dynamicSmemSize: *mut usize,
    func: *const ::core::ffi::c_void,
    numBlocks: ::core::ffi::c_int,
    blockSize: ::core::ffi::c_int,
) -> cudaError_t;
pub unsafe fn cudaOccupancyAvailableDynamicSMemPerBlock(
    dynamicSmemSize: *mut usize,
    func: *const ::core::ffi::c_void,
    numBlocks: ::core::ffi::c_int,
    blockSize: ::core::ffi::c_int,
) -> cudaError_t {
    let sym: Symbol<cudaOccupancyAvailableDynamicSMemPerBlock_fn> = unsafe {
        get_lib()
            .get(b"cudaOccupancyAvailableDynamicSMemPerBlock\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(dynamicSmemSize, func, numBlocks, blockSize) }
}

type cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags_fn =
    unsafe extern "C" fn(
        numBlocks: *mut ::core::ffi::c_int,
        func: *const ::core::ffi::c_void,
        blockSize: ::core::ffi::c_int,
        dynamicSMemSize: usize,
        flags: ::core::ffi::c_uint,
    ) -> cudaError_t;
pub unsafe fn cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
    numBlocks: *mut ::core::ffi::c_int,
    func: *const ::core::ffi::c_void,
    blockSize: ::core::ffi::c_int,
    dynamicSMemSize: usize,
    flags: ::core::ffi::c_uint,
) -> cudaError_t {
    let sym: Symbol<cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags_fn> = unsafe {
        get_lib()
            .get(b"cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(numBlocks, func, blockSize, dynamicSMemSize, flags) }
}

type cudaOccupancyMaxPotentialClusterSize_fn = unsafe extern "C" fn(
    clusterSize: *mut ::core::ffi::c_int,
    func: *const ::core::ffi::c_void,
    launchConfig: *const cudaLaunchConfig_t,
) -> cudaError_t;
pub unsafe fn cudaOccupancyMaxPotentialClusterSize(
    clusterSize: *mut ::core::ffi::c_int,
    func: *const ::core::ffi::c_void,
    launchConfig: *const cudaLaunchConfig_t,
) -> cudaError_t {
    let sym: Symbol<cudaOccupancyMaxPotentialClusterSize_fn> = unsafe {
        get_lib()
            .get(b"cudaOccupancyMaxPotentialClusterSize\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(clusterSize, func, launchConfig) }
}

type cudaOccupancyMaxActiveClusters_fn = unsafe extern "C" fn(
    numClusters: *mut ::core::ffi::c_int,
    func: *const ::core::ffi::c_void,
    launchConfig: *const cudaLaunchConfig_t,
) -> cudaError_t;
pub unsafe fn cudaOccupancyMaxActiveClusters(
    numClusters: *mut ::core::ffi::c_int,
    func: *const ::core::ffi::c_void,
    launchConfig: *const cudaLaunchConfig_t,
) -> cudaError_t {
    let sym: Symbol<cudaOccupancyMaxActiveClusters_fn> = unsafe {
        get_lib()
            .get(b"cudaOccupancyMaxActiveClusters\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(numClusters, func, launchConfig) }
}

type cudaMallocManaged_fn = unsafe extern "C" fn(
    devPtr: *mut *mut ::core::ffi::c_void,
    size: usize,
    flags: ::core::ffi::c_uint,
) -> cudaError_t;
pub unsafe fn cudaMallocManaged(
    devPtr: *mut *mut ::core::ffi::c_void,
    size: usize,
    flags: ::core::ffi::c_uint,
) -> cudaError_t {
    let sym: Symbol<cudaMallocManaged_fn> = unsafe {
        get_lib()
            .get(b"cudaMallocManaged\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(devPtr, size, flags) }
}

type cudaMalloc_fn =
    unsafe extern "C" fn(devPtr: *mut *mut ::core::ffi::c_void, size: usize) -> cudaError_t;
pub unsafe fn cudaMalloc(devPtr: *mut *mut ::core::ffi::c_void, size: usize) -> cudaError_t {
    let sym: Symbol<cudaMalloc_fn> =
        unsafe { get_lib().get(b"cudaMalloc\0").expect("Missing symbol") };
    unsafe { (*sym)(devPtr, size) }
}

type cudaMallocHost_fn =
    unsafe extern "C" fn(ptr: *mut *mut ::core::ffi::c_void, size: usize) -> cudaError_t;
pub unsafe fn cudaMallocHost(ptr: *mut *mut ::core::ffi::c_void, size: usize) -> cudaError_t {
    let sym: Symbol<cudaMallocHost_fn> =
        unsafe { get_lib().get(b"cudaMallocHost\0").expect("Missing symbol") };
    unsafe { (*sym)(ptr, size) }
}

type cudaMallocPitch_fn = unsafe extern "C" fn(
    devPtr: *mut *mut ::core::ffi::c_void,
    pitch: *mut usize,
    width: usize,
    height: usize,
) -> cudaError_t;
pub unsafe fn cudaMallocPitch(
    devPtr: *mut *mut ::core::ffi::c_void,
    pitch: *mut usize,
    width: usize,
    height: usize,
) -> cudaError_t {
    let sym: Symbol<cudaMallocPitch_fn> =
        unsafe { get_lib().get(b"cudaMallocPitch\0").expect("Missing symbol") };
    unsafe { (*sym)(devPtr, pitch, width, height) }
}

type cudaMallocArray_fn = unsafe extern "C" fn(
    array: *mut cudaArray_t,
    desc: *const cudaChannelFormatDesc,
    width: usize,
    height: usize,
    flags: ::core::ffi::c_uint,
) -> cudaError_t;
pub unsafe fn cudaMallocArray(
    array: *mut cudaArray_t,
    desc: *const cudaChannelFormatDesc,
    width: usize,
    height: usize,
    flags: ::core::ffi::c_uint,
) -> cudaError_t {
    let sym: Symbol<cudaMallocArray_fn> =
        unsafe { get_lib().get(b"cudaMallocArray\0").expect("Missing symbol") };
    unsafe { (*sym)(array, desc, width, height, flags) }
}

type cudaFree_fn = unsafe extern "C" fn(devPtr: *mut ::core::ffi::c_void) -> cudaError_t;
pub unsafe fn cudaFree(devPtr: *mut ::core::ffi::c_void) -> cudaError_t {
    let sym: Symbol<cudaFree_fn> = unsafe { get_lib().get(b"cudaFree\0").expect("Missing symbol") };
    unsafe { (*sym)(devPtr) }
}

type cudaFreeHost_fn = unsafe extern "C" fn(ptr: *mut ::core::ffi::c_void) -> cudaError_t;
pub unsafe fn cudaFreeHost(ptr: *mut ::core::ffi::c_void) -> cudaError_t {
    let sym: Symbol<cudaFreeHost_fn> =
        unsafe { get_lib().get(b"cudaFreeHost\0").expect("Missing symbol") };
    unsafe { (*sym)(ptr) }
}

type cudaFreeArray_fn = unsafe extern "C" fn(array: cudaArray_t) -> cudaError_t;
pub unsafe fn cudaFreeArray(array: cudaArray_t) -> cudaError_t {
    let sym: Symbol<cudaFreeArray_fn> =
        unsafe { get_lib().get(b"cudaFreeArray\0").expect("Missing symbol") };
    unsafe { (*sym)(array) }
}

type cudaFreeMipmappedArray_fn =
    unsafe extern "C" fn(mipmappedArray: cudaMipmappedArray_t) -> cudaError_t;
pub unsafe fn cudaFreeMipmappedArray(mipmappedArray: cudaMipmappedArray_t) -> cudaError_t {
    let sym: Symbol<cudaFreeMipmappedArray_fn> = unsafe {
        get_lib()
            .get(b"cudaFreeMipmappedArray\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(mipmappedArray) }
}

type cudaHostAlloc_fn = unsafe extern "C" fn(
    pHost: *mut *mut ::core::ffi::c_void,
    size: usize,
    flags: ::core::ffi::c_uint,
) -> cudaError_t;
pub unsafe fn cudaHostAlloc(
    pHost: *mut *mut ::core::ffi::c_void,
    size: usize,
    flags: ::core::ffi::c_uint,
) -> cudaError_t {
    let sym: Symbol<cudaHostAlloc_fn> =
        unsafe { get_lib().get(b"cudaHostAlloc\0").expect("Missing symbol") };
    unsafe { (*sym)(pHost, size, flags) }
}

type cudaHostRegister_fn = unsafe extern "C" fn(
    ptr: *mut ::core::ffi::c_void,
    size: usize,
    flags: ::core::ffi::c_uint,
) -> cudaError_t;
pub unsafe fn cudaHostRegister(
    ptr: *mut ::core::ffi::c_void,
    size: usize,
    flags: ::core::ffi::c_uint,
) -> cudaError_t {
    let sym: Symbol<cudaHostRegister_fn> = unsafe {
        get_lib()
            .get(b"cudaHostRegister\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(ptr, size, flags) }
}

type cudaHostUnregister_fn = unsafe extern "C" fn(ptr: *mut ::core::ffi::c_void) -> cudaError_t;
pub unsafe fn cudaHostUnregister(ptr: *mut ::core::ffi::c_void) -> cudaError_t {
    let sym: Symbol<cudaHostUnregister_fn> = unsafe {
        get_lib()
            .get(b"cudaHostUnregister\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(ptr) }
}

type cudaHostGetDevicePointer_fn = unsafe extern "C" fn(
    pDevice: *mut *mut ::core::ffi::c_void,
    pHost: *mut ::core::ffi::c_void,
    flags: ::core::ffi::c_uint,
) -> cudaError_t;
pub unsafe fn cudaHostGetDevicePointer(
    pDevice: *mut *mut ::core::ffi::c_void,
    pHost: *mut ::core::ffi::c_void,
    flags: ::core::ffi::c_uint,
) -> cudaError_t {
    let sym: Symbol<cudaHostGetDevicePointer_fn> = unsafe {
        get_lib()
            .get(b"cudaHostGetDevicePointer\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(pDevice, pHost, flags) }
}

type cudaHostGetFlags_fn = unsafe extern "C" fn(
    pFlags: *mut ::core::ffi::c_uint,
    pHost: *mut ::core::ffi::c_void,
) -> cudaError_t;
pub unsafe fn cudaHostGetFlags(
    pFlags: *mut ::core::ffi::c_uint,
    pHost: *mut ::core::ffi::c_void,
) -> cudaError_t {
    let sym: Symbol<cudaHostGetFlags_fn> = unsafe {
        get_lib()
            .get(b"cudaHostGetFlags\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(pFlags, pHost) }
}

type cudaMalloc3D_fn =
    unsafe extern "C" fn(pitchedDevPtr: *mut cudaPitchedPtr, extent: cudaExtent) -> cudaError_t;
pub unsafe fn cudaMalloc3D(pitchedDevPtr: *mut cudaPitchedPtr, extent: cudaExtent) -> cudaError_t {
    let sym: Symbol<cudaMalloc3D_fn> =
        unsafe { get_lib().get(b"cudaMalloc3D\0").expect("Missing symbol") };
    unsafe { (*sym)(pitchedDevPtr, extent) }
}

type cudaMalloc3DArray_fn = unsafe extern "C" fn(
    array: *mut cudaArray_t,
    desc: *const cudaChannelFormatDesc,
    extent: cudaExtent,
    flags: ::core::ffi::c_uint,
) -> cudaError_t;
pub unsafe fn cudaMalloc3DArray(
    array: *mut cudaArray_t,
    desc: *const cudaChannelFormatDesc,
    extent: cudaExtent,
    flags: ::core::ffi::c_uint,
) -> cudaError_t {
    let sym: Symbol<cudaMalloc3DArray_fn> = unsafe {
        get_lib()
            .get(b"cudaMalloc3DArray\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(array, desc, extent, flags) }
}

type cudaMallocMipmappedArray_fn = unsafe extern "C" fn(
    mipmappedArray: *mut cudaMipmappedArray_t,
    desc: *const cudaChannelFormatDesc,
    extent: cudaExtent,
    numLevels: ::core::ffi::c_uint,
    flags: ::core::ffi::c_uint,
) -> cudaError_t;
pub unsafe fn cudaMallocMipmappedArray(
    mipmappedArray: *mut cudaMipmappedArray_t,
    desc: *const cudaChannelFormatDesc,
    extent: cudaExtent,
    numLevels: ::core::ffi::c_uint,
    flags: ::core::ffi::c_uint,
) -> cudaError_t {
    let sym: Symbol<cudaMallocMipmappedArray_fn> = unsafe {
        get_lib()
            .get(b"cudaMallocMipmappedArray\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(mipmappedArray, desc, extent, numLevels, flags) }
}

type cudaGetMipmappedArrayLevel_fn = unsafe extern "C" fn(
    levelArray: *mut cudaArray_t,
    mipmappedArray: cudaMipmappedArray_const_t,
    level: ::core::ffi::c_uint,
) -> cudaError_t;
pub unsafe fn cudaGetMipmappedArrayLevel(
    levelArray: *mut cudaArray_t,
    mipmappedArray: cudaMipmappedArray_const_t,
    level: ::core::ffi::c_uint,
) -> cudaError_t {
    let sym: Symbol<cudaGetMipmappedArrayLevel_fn> = unsafe {
        get_lib()
            .get(b"cudaGetMipmappedArrayLevel\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(levelArray, mipmappedArray, level) }
}

type cudaMemcpy3D_fn = unsafe extern "C" fn(p: *const cudaMemcpy3DParms) -> cudaError_t;
pub unsafe fn cudaMemcpy3D(p: *const cudaMemcpy3DParms) -> cudaError_t {
    let sym: Symbol<cudaMemcpy3D_fn> =
        unsafe { get_lib().get(b"cudaMemcpy3D\0").expect("Missing symbol") };
    unsafe { (*sym)(p) }
}

type cudaMemcpy3DPeer_fn = unsafe extern "C" fn(p: *const cudaMemcpy3DPeerParms) -> cudaError_t;
pub unsafe fn cudaMemcpy3DPeer(p: *const cudaMemcpy3DPeerParms) -> cudaError_t {
    let sym: Symbol<cudaMemcpy3DPeer_fn> = unsafe {
        get_lib()
            .get(b"cudaMemcpy3DPeer\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(p) }
}

type cudaMemcpy3DAsync_fn =
    unsafe extern "C" fn(p: *const cudaMemcpy3DParms, stream: cudaStream_t) -> cudaError_t;
pub unsafe fn cudaMemcpy3DAsync(p: *const cudaMemcpy3DParms, stream: cudaStream_t) -> cudaError_t {
    let sym: Symbol<cudaMemcpy3DAsync_fn> = unsafe {
        get_lib()
            .get(b"cudaMemcpy3DAsync\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(p, stream) }
}

type cudaMemcpy3DPeerAsync_fn =
    unsafe extern "C" fn(p: *const cudaMemcpy3DPeerParms, stream: cudaStream_t) -> cudaError_t;
pub unsafe fn cudaMemcpy3DPeerAsync(
    p: *const cudaMemcpy3DPeerParms,
    stream: cudaStream_t,
) -> cudaError_t {
    let sym: Symbol<cudaMemcpy3DPeerAsync_fn> = unsafe {
        get_lib()
            .get(b"cudaMemcpy3DPeerAsync\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(p, stream) }
}

type cudaMemGetInfo_fn = unsafe extern "C" fn(free: *mut usize, total: *mut usize) -> cudaError_t;
pub unsafe fn cudaMemGetInfo(free: *mut usize, total: *mut usize) -> cudaError_t {
    let sym: Symbol<cudaMemGetInfo_fn> =
        unsafe { get_lib().get(b"cudaMemGetInfo\0").expect("Missing symbol") };
    unsafe { (*sym)(free, total) }
}

type cudaArrayGetInfo_fn = unsafe extern "C" fn(
    desc: *mut cudaChannelFormatDesc,
    extent: *mut cudaExtent,
    flags: *mut ::core::ffi::c_uint,
    array: cudaArray_t,
) -> cudaError_t;
pub unsafe fn cudaArrayGetInfo(
    desc: *mut cudaChannelFormatDesc,
    extent: *mut cudaExtent,
    flags: *mut ::core::ffi::c_uint,
    array: cudaArray_t,
) -> cudaError_t {
    let sym: Symbol<cudaArrayGetInfo_fn> = unsafe {
        get_lib()
            .get(b"cudaArrayGetInfo\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(desc, extent, flags, array) }
}

type cudaArrayGetPlane_fn = unsafe extern "C" fn(
    pPlaneArray: *mut cudaArray_t,
    hArray: cudaArray_t,
    planeIdx: ::core::ffi::c_uint,
) -> cudaError_t;
pub unsafe fn cudaArrayGetPlane(
    pPlaneArray: *mut cudaArray_t,
    hArray: cudaArray_t,
    planeIdx: ::core::ffi::c_uint,
) -> cudaError_t {
    let sym: Symbol<cudaArrayGetPlane_fn> = unsafe {
        get_lib()
            .get(b"cudaArrayGetPlane\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(pPlaneArray, hArray, planeIdx) }
}

type cudaArrayGetMemoryRequirements_fn = unsafe extern "C" fn(
    memoryRequirements: *mut cudaArrayMemoryRequirements,
    array: cudaArray_t,
    device: ::core::ffi::c_int,
) -> cudaError_t;
pub unsafe fn cudaArrayGetMemoryRequirements(
    memoryRequirements: *mut cudaArrayMemoryRequirements,
    array: cudaArray_t,
    device: ::core::ffi::c_int,
) -> cudaError_t {
    let sym: Symbol<cudaArrayGetMemoryRequirements_fn> = unsafe {
        get_lib()
            .get(b"cudaArrayGetMemoryRequirements\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(memoryRequirements, array, device) }
}

type cudaMipmappedArrayGetMemoryRequirements_fn = unsafe extern "C" fn(
    memoryRequirements: *mut cudaArrayMemoryRequirements,
    mipmap: cudaMipmappedArray_t,
    device: ::core::ffi::c_int,
) -> cudaError_t;
pub unsafe fn cudaMipmappedArrayGetMemoryRequirements(
    memoryRequirements: *mut cudaArrayMemoryRequirements,
    mipmap: cudaMipmappedArray_t,
    device: ::core::ffi::c_int,
) -> cudaError_t {
    let sym: Symbol<cudaMipmappedArrayGetMemoryRequirements_fn> = unsafe {
        get_lib()
            .get(b"cudaMipmappedArrayGetMemoryRequirements\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(memoryRequirements, mipmap, device) }
}

type cudaArrayGetSparseProperties_fn = unsafe extern "C" fn(
    sparseProperties: *mut cudaArraySparseProperties,
    array: cudaArray_t,
) -> cudaError_t;
pub unsafe fn cudaArrayGetSparseProperties(
    sparseProperties: *mut cudaArraySparseProperties,
    array: cudaArray_t,
) -> cudaError_t {
    let sym: Symbol<cudaArrayGetSparseProperties_fn> = unsafe {
        get_lib()
            .get(b"cudaArrayGetSparseProperties\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(sparseProperties, array) }
}

type cudaMipmappedArrayGetSparseProperties_fn = unsafe extern "C" fn(
    sparseProperties: *mut cudaArraySparseProperties,
    mipmap: cudaMipmappedArray_t,
) -> cudaError_t;
pub unsafe fn cudaMipmappedArrayGetSparseProperties(
    sparseProperties: *mut cudaArraySparseProperties,
    mipmap: cudaMipmappedArray_t,
) -> cudaError_t {
    let sym: Symbol<cudaMipmappedArrayGetSparseProperties_fn> = unsafe {
        get_lib()
            .get(b"cudaMipmappedArrayGetSparseProperties\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(sparseProperties, mipmap) }
}

type cudaMemcpy_fn = unsafe extern "C" fn(
    dst: *mut ::core::ffi::c_void,
    src: *const ::core::ffi::c_void,
    count: usize,
    kind: cudaMemcpyKind,
) -> cudaError_t;
pub unsafe fn cudaMemcpy(
    dst: *mut ::core::ffi::c_void,
    src: *const ::core::ffi::c_void,
    count: usize,
    kind: cudaMemcpyKind,
) -> cudaError_t {
    let sym: Symbol<cudaMemcpy_fn> =
        unsafe { get_lib().get(b"cudaMemcpy\0").expect("Missing symbol") };
    unsafe { (*sym)(dst, src, count, kind) }
}

type cudaMemcpyPeer_fn = unsafe extern "C" fn(
    dst: *mut ::core::ffi::c_void,
    dstDevice: ::core::ffi::c_int,
    src: *const ::core::ffi::c_void,
    srcDevice: ::core::ffi::c_int,
    count: usize,
) -> cudaError_t;
pub unsafe fn cudaMemcpyPeer(
    dst: *mut ::core::ffi::c_void,
    dstDevice: ::core::ffi::c_int,
    src: *const ::core::ffi::c_void,
    srcDevice: ::core::ffi::c_int,
    count: usize,
) -> cudaError_t {
    let sym: Symbol<cudaMemcpyPeer_fn> =
        unsafe { get_lib().get(b"cudaMemcpyPeer\0").expect("Missing symbol") };
    unsafe { (*sym)(dst, dstDevice, src, srcDevice, count) }
}

type cudaMemcpy2D_fn = unsafe extern "C" fn(
    dst: *mut ::core::ffi::c_void,
    dpitch: usize,
    src: *const ::core::ffi::c_void,
    spitch: usize,
    width: usize,
    height: usize,
    kind: cudaMemcpyKind,
) -> cudaError_t;
pub unsafe fn cudaMemcpy2D(
    dst: *mut ::core::ffi::c_void,
    dpitch: usize,
    src: *const ::core::ffi::c_void,
    spitch: usize,
    width: usize,
    height: usize,
    kind: cudaMemcpyKind,
) -> cudaError_t {
    let sym: Symbol<cudaMemcpy2D_fn> =
        unsafe { get_lib().get(b"cudaMemcpy2D\0").expect("Missing symbol") };
    unsafe { (*sym)(dst, dpitch, src, spitch, width, height, kind) }
}

type cudaMemcpy2DToArray_fn = unsafe extern "C" fn(
    dst: cudaArray_t,
    wOffset: usize,
    hOffset: usize,
    src: *const ::core::ffi::c_void,
    spitch: usize,
    width: usize,
    height: usize,
    kind: cudaMemcpyKind,
) -> cudaError_t;
pub unsafe fn cudaMemcpy2DToArray(
    dst: cudaArray_t,
    wOffset: usize,
    hOffset: usize,
    src: *const ::core::ffi::c_void,
    spitch: usize,
    width: usize,
    height: usize,
    kind: cudaMemcpyKind,
) -> cudaError_t {
    let sym: Symbol<cudaMemcpy2DToArray_fn> = unsafe {
        get_lib()
            .get(b"cudaMemcpy2DToArray\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(dst, wOffset, hOffset, src, spitch, width, height, kind) }
}

type cudaMemcpy2DFromArray_fn = unsafe extern "C" fn(
    dst: *mut ::core::ffi::c_void,
    dpitch: usize,
    src: cudaArray_const_t,
    wOffset: usize,
    hOffset: usize,
    width: usize,
    height: usize,
    kind: cudaMemcpyKind,
) -> cudaError_t;
pub unsafe fn cudaMemcpy2DFromArray(
    dst: *mut ::core::ffi::c_void,
    dpitch: usize,
    src: cudaArray_const_t,
    wOffset: usize,
    hOffset: usize,
    width: usize,
    height: usize,
    kind: cudaMemcpyKind,
) -> cudaError_t {
    let sym: Symbol<cudaMemcpy2DFromArray_fn> = unsafe {
        get_lib()
            .get(b"cudaMemcpy2DFromArray\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(dst, dpitch, src, wOffset, hOffset, width, height, kind) }
}

type cudaMemcpy2DArrayToArray_fn = unsafe extern "C" fn(
    dst: cudaArray_t,
    wOffsetDst: usize,
    hOffsetDst: usize,
    src: cudaArray_const_t,
    wOffsetSrc: usize,
    hOffsetSrc: usize,
    width: usize,
    height: usize,
    kind: cudaMemcpyKind,
) -> cudaError_t;
pub unsafe fn cudaMemcpy2DArrayToArray(
    dst: cudaArray_t,
    wOffsetDst: usize,
    hOffsetDst: usize,
    src: cudaArray_const_t,
    wOffsetSrc: usize,
    hOffsetSrc: usize,
    width: usize,
    height: usize,
    kind: cudaMemcpyKind,
) -> cudaError_t {
    let sym: Symbol<cudaMemcpy2DArrayToArray_fn> = unsafe {
        get_lib()
            .get(b"cudaMemcpy2DArrayToArray\0")
            .expect("Missing symbol")
    };
    unsafe {
        (*sym)(
            dst, wOffsetDst, hOffsetDst, src, wOffsetSrc, hOffsetSrc, width, height, kind,
        )
    }
}

type cudaMemcpyToSymbol_fn = unsafe extern "C" fn(
    symbol: *const ::core::ffi::c_void,
    src: *const ::core::ffi::c_void,
    count: usize,
    offset: usize,
    kind: cudaMemcpyKind,
) -> cudaError_t;
pub unsafe fn cudaMemcpyToSymbol(
    symbol: *const ::core::ffi::c_void,
    src: *const ::core::ffi::c_void,
    count: usize,
    offset: usize,
    kind: cudaMemcpyKind,
) -> cudaError_t {
    let sym: Symbol<cudaMemcpyToSymbol_fn> = unsafe {
        get_lib()
            .get(b"cudaMemcpyToSymbol\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(symbol, src, count, offset, kind) }
}

type cudaMemcpyFromSymbol_fn = unsafe extern "C" fn(
    dst: *mut ::core::ffi::c_void,
    symbol: *const ::core::ffi::c_void,
    count: usize,
    offset: usize,
    kind: cudaMemcpyKind,
) -> cudaError_t;
pub unsafe fn cudaMemcpyFromSymbol(
    dst: *mut ::core::ffi::c_void,
    symbol: *const ::core::ffi::c_void,
    count: usize,
    offset: usize,
    kind: cudaMemcpyKind,
) -> cudaError_t {
    let sym: Symbol<cudaMemcpyFromSymbol_fn> = unsafe {
        get_lib()
            .get(b"cudaMemcpyFromSymbol\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(dst, symbol, count, offset, kind) }
}

type cudaMemcpyAsync_fn = unsafe extern "C" fn(
    dst: *mut ::core::ffi::c_void,
    src: *const ::core::ffi::c_void,
    count: usize,
    kind: cudaMemcpyKind,
    stream: cudaStream_t,
) -> cudaError_t;
pub unsafe fn cudaMemcpyAsync(
    dst: *mut ::core::ffi::c_void,
    src: *const ::core::ffi::c_void,
    count: usize,
    kind: cudaMemcpyKind,
    stream: cudaStream_t,
) -> cudaError_t {
    let sym: Symbol<cudaMemcpyAsync_fn> =
        unsafe { get_lib().get(b"cudaMemcpyAsync\0").expect("Missing symbol") };
    unsafe { (*sym)(dst, src, count, kind, stream) }
}

type cudaMemcpyPeerAsync_fn = unsafe extern "C" fn(
    dst: *mut ::core::ffi::c_void,
    dstDevice: ::core::ffi::c_int,
    src: *const ::core::ffi::c_void,
    srcDevice: ::core::ffi::c_int,
    count: usize,
    stream: cudaStream_t,
) -> cudaError_t;
pub unsafe fn cudaMemcpyPeerAsync(
    dst: *mut ::core::ffi::c_void,
    dstDevice: ::core::ffi::c_int,
    src: *const ::core::ffi::c_void,
    srcDevice: ::core::ffi::c_int,
    count: usize,
    stream: cudaStream_t,
) -> cudaError_t {
    let sym: Symbol<cudaMemcpyPeerAsync_fn> = unsafe {
        get_lib()
            .get(b"cudaMemcpyPeerAsync\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(dst, dstDevice, src, srcDevice, count, stream) }
}

type cudaMemcpyBatchAsync_fn = unsafe extern "C" fn(
    dsts: *const *mut ::core::ffi::c_void,
    srcs: *const *const ::core::ffi::c_void,
    sizes: *const usize,
    count: usize,
    attrs: *mut cudaMemcpyAttributes,
    attrsIdxs: *mut usize,
    numAttrs: usize,
    stream: cudaStream_t,
) -> cudaError_t;
pub unsafe fn cudaMemcpyBatchAsync(
    dsts: *const *mut ::core::ffi::c_void,
    srcs: *const *const ::core::ffi::c_void,
    sizes: *const usize,
    count: usize,
    attrs: *mut cudaMemcpyAttributes,
    attrsIdxs: *mut usize,
    numAttrs: usize,
    stream: cudaStream_t,
) -> cudaError_t {
    let sym: Symbol<cudaMemcpyBatchAsync_fn> = unsafe {
        get_lib()
            .get(b"cudaMemcpyBatchAsync\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(dsts, srcs, sizes, count, attrs, attrsIdxs, numAttrs, stream) }
}

type cudaMemcpy3DBatchAsync_fn = unsafe extern "C" fn(
    numOps: usize,
    opList: *mut cudaMemcpy3DBatchOp,
    flags: ::core::ffi::c_ulonglong,
    stream: cudaStream_t,
) -> cudaError_t;
pub unsafe fn cudaMemcpy3DBatchAsync(
    numOps: usize,
    opList: *mut cudaMemcpy3DBatchOp,
    flags: ::core::ffi::c_ulonglong,
    stream: cudaStream_t,
) -> cudaError_t {
    let sym: Symbol<cudaMemcpy3DBatchAsync_fn> = unsafe {
        get_lib()
            .get(b"cudaMemcpy3DBatchAsync\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(numOps, opList, flags, stream) }
}

type cudaMemcpy2DAsync_fn = unsafe extern "C" fn(
    dst: *mut ::core::ffi::c_void,
    dpitch: usize,
    src: *const ::core::ffi::c_void,
    spitch: usize,
    width: usize,
    height: usize,
    kind: cudaMemcpyKind,
    stream: cudaStream_t,
) -> cudaError_t;
pub unsafe fn cudaMemcpy2DAsync(
    dst: *mut ::core::ffi::c_void,
    dpitch: usize,
    src: *const ::core::ffi::c_void,
    spitch: usize,
    width: usize,
    height: usize,
    kind: cudaMemcpyKind,
    stream: cudaStream_t,
) -> cudaError_t {
    let sym: Symbol<cudaMemcpy2DAsync_fn> = unsafe {
        get_lib()
            .get(b"cudaMemcpy2DAsync\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(dst, dpitch, src, spitch, width, height, kind, stream) }
}

type cudaMemcpy2DToArrayAsync_fn = unsafe extern "C" fn(
    dst: cudaArray_t,
    wOffset: usize,
    hOffset: usize,
    src: *const ::core::ffi::c_void,
    spitch: usize,
    width: usize,
    height: usize,
    kind: cudaMemcpyKind,
    stream: cudaStream_t,
) -> cudaError_t;
pub unsafe fn cudaMemcpy2DToArrayAsync(
    dst: cudaArray_t,
    wOffset: usize,
    hOffset: usize,
    src: *const ::core::ffi::c_void,
    spitch: usize,
    width: usize,
    height: usize,
    kind: cudaMemcpyKind,
    stream: cudaStream_t,
) -> cudaError_t {
    let sym: Symbol<cudaMemcpy2DToArrayAsync_fn> = unsafe {
        get_lib()
            .get(b"cudaMemcpy2DToArrayAsync\0")
            .expect("Missing symbol")
    };
    unsafe {
        (*sym)(
            dst, wOffset, hOffset, src, spitch, width, height, kind, stream,
        )
    }
}

type cudaMemcpy2DFromArrayAsync_fn = unsafe extern "C" fn(
    dst: *mut ::core::ffi::c_void,
    dpitch: usize,
    src: cudaArray_const_t,
    wOffset: usize,
    hOffset: usize,
    width: usize,
    height: usize,
    kind: cudaMemcpyKind,
    stream: cudaStream_t,
) -> cudaError_t;
pub unsafe fn cudaMemcpy2DFromArrayAsync(
    dst: *mut ::core::ffi::c_void,
    dpitch: usize,
    src: cudaArray_const_t,
    wOffset: usize,
    hOffset: usize,
    width: usize,
    height: usize,
    kind: cudaMemcpyKind,
    stream: cudaStream_t,
) -> cudaError_t {
    let sym: Symbol<cudaMemcpy2DFromArrayAsync_fn> = unsafe {
        get_lib()
            .get(b"cudaMemcpy2DFromArrayAsync\0")
            .expect("Missing symbol")
    };
    unsafe {
        (*sym)(
            dst, dpitch, src, wOffset, hOffset, width, height, kind, stream,
        )
    }
}

type cudaMemcpyToSymbolAsync_fn = unsafe extern "C" fn(
    symbol: *const ::core::ffi::c_void,
    src: *const ::core::ffi::c_void,
    count: usize,
    offset: usize,
    kind: cudaMemcpyKind,
    stream: cudaStream_t,
) -> cudaError_t;
pub unsafe fn cudaMemcpyToSymbolAsync(
    symbol: *const ::core::ffi::c_void,
    src: *const ::core::ffi::c_void,
    count: usize,
    offset: usize,
    kind: cudaMemcpyKind,
    stream: cudaStream_t,
) -> cudaError_t {
    let sym: Symbol<cudaMemcpyToSymbolAsync_fn> = unsafe {
        get_lib()
            .get(b"cudaMemcpyToSymbolAsync\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(symbol, src, count, offset, kind, stream) }
}

type cudaMemcpyFromSymbolAsync_fn = unsafe extern "C" fn(
    dst: *mut ::core::ffi::c_void,
    symbol: *const ::core::ffi::c_void,
    count: usize,
    offset: usize,
    kind: cudaMemcpyKind,
    stream: cudaStream_t,
) -> cudaError_t;
pub unsafe fn cudaMemcpyFromSymbolAsync(
    dst: *mut ::core::ffi::c_void,
    symbol: *const ::core::ffi::c_void,
    count: usize,
    offset: usize,
    kind: cudaMemcpyKind,
    stream: cudaStream_t,
) -> cudaError_t {
    let sym: Symbol<cudaMemcpyFromSymbolAsync_fn> = unsafe {
        get_lib()
            .get(b"cudaMemcpyFromSymbolAsync\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(dst, symbol, count, offset, kind, stream) }
}

type cudaMemset_fn = unsafe extern "C" fn(
    devPtr: *mut ::core::ffi::c_void,
    value: ::core::ffi::c_int,
    count: usize,
) -> cudaError_t;
pub unsafe fn cudaMemset(
    devPtr: *mut ::core::ffi::c_void,
    value: ::core::ffi::c_int,
    count: usize,
) -> cudaError_t {
    let sym: Symbol<cudaMemset_fn> =
        unsafe { get_lib().get(b"cudaMemset\0").expect("Missing symbol") };
    unsafe { (*sym)(devPtr, value, count) }
}

type cudaMemset2D_fn = unsafe extern "C" fn(
    devPtr: *mut ::core::ffi::c_void,
    pitch: usize,
    value: ::core::ffi::c_int,
    width: usize,
    height: usize,
) -> cudaError_t;
pub unsafe fn cudaMemset2D(
    devPtr: *mut ::core::ffi::c_void,
    pitch: usize,
    value: ::core::ffi::c_int,
    width: usize,
    height: usize,
) -> cudaError_t {
    let sym: Symbol<cudaMemset2D_fn> =
        unsafe { get_lib().get(b"cudaMemset2D\0").expect("Missing symbol") };
    unsafe { (*sym)(devPtr, pitch, value, width, height) }
}

type cudaMemset3D_fn = unsafe extern "C" fn(
    pitchedDevPtr: cudaPitchedPtr,
    value: ::core::ffi::c_int,
    extent: cudaExtent,
) -> cudaError_t;
pub unsafe fn cudaMemset3D(
    pitchedDevPtr: cudaPitchedPtr,
    value: ::core::ffi::c_int,
    extent: cudaExtent,
) -> cudaError_t {
    let sym: Symbol<cudaMemset3D_fn> =
        unsafe { get_lib().get(b"cudaMemset3D\0").expect("Missing symbol") };
    unsafe { (*sym)(pitchedDevPtr, value, extent) }
}

type cudaMemsetAsync_fn = unsafe extern "C" fn(
    devPtr: *mut ::core::ffi::c_void,
    value: ::core::ffi::c_int,
    count: usize,
    stream: cudaStream_t,
) -> cudaError_t;
pub unsafe fn cudaMemsetAsync(
    devPtr: *mut ::core::ffi::c_void,
    value: ::core::ffi::c_int,
    count: usize,
    stream: cudaStream_t,
) -> cudaError_t {
    let sym: Symbol<cudaMemsetAsync_fn> =
        unsafe { get_lib().get(b"cudaMemsetAsync\0").expect("Missing symbol") };
    unsafe { (*sym)(devPtr, value, count, stream) }
}

type cudaMemset2DAsync_fn = unsafe extern "C" fn(
    devPtr: *mut ::core::ffi::c_void,
    pitch: usize,
    value: ::core::ffi::c_int,
    width: usize,
    height: usize,
    stream: cudaStream_t,
) -> cudaError_t;
pub unsafe fn cudaMemset2DAsync(
    devPtr: *mut ::core::ffi::c_void,
    pitch: usize,
    value: ::core::ffi::c_int,
    width: usize,
    height: usize,
    stream: cudaStream_t,
) -> cudaError_t {
    let sym: Symbol<cudaMemset2DAsync_fn> = unsafe {
        get_lib()
            .get(b"cudaMemset2DAsync\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(devPtr, pitch, value, width, height, stream) }
}

type cudaMemset3DAsync_fn = unsafe extern "C" fn(
    pitchedDevPtr: cudaPitchedPtr,
    value: ::core::ffi::c_int,
    extent: cudaExtent,
    stream: cudaStream_t,
) -> cudaError_t;
pub unsafe fn cudaMemset3DAsync(
    pitchedDevPtr: cudaPitchedPtr,
    value: ::core::ffi::c_int,
    extent: cudaExtent,
    stream: cudaStream_t,
) -> cudaError_t {
    let sym: Symbol<cudaMemset3DAsync_fn> = unsafe {
        get_lib()
            .get(b"cudaMemset3DAsync\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(pitchedDevPtr, value, extent, stream) }
}

type cudaGetSymbolAddress_fn = unsafe extern "C" fn(
    devPtr: *mut *mut ::core::ffi::c_void,
    symbol: *const ::core::ffi::c_void,
) -> cudaError_t;
pub unsafe fn cudaGetSymbolAddress(
    devPtr: *mut *mut ::core::ffi::c_void,
    symbol: *const ::core::ffi::c_void,
) -> cudaError_t {
    let sym: Symbol<cudaGetSymbolAddress_fn> = unsafe {
        get_lib()
            .get(b"cudaGetSymbolAddress\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(devPtr, symbol) }
}

type cudaGetSymbolSize_fn =
    unsafe extern "C" fn(size: *mut usize, symbol: *const ::core::ffi::c_void) -> cudaError_t;
pub unsafe fn cudaGetSymbolSize(
    size: *mut usize,
    symbol: *const ::core::ffi::c_void,
) -> cudaError_t {
    let sym: Symbol<cudaGetSymbolSize_fn> = unsafe {
        get_lib()
            .get(b"cudaGetSymbolSize\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(size, symbol) }
}

type cudaMemPrefetchAsync_fn = unsafe extern "C" fn(
    devPtr: *const ::core::ffi::c_void,
    count: usize,
    location: cudaMemLocation,
    flags: ::core::ffi::c_uint,
    stream: cudaStream_t,
) -> cudaError_t;
pub unsafe fn cudaMemPrefetchAsync(
    devPtr: *const ::core::ffi::c_void,
    count: usize,
    location: cudaMemLocation,
    flags: ::core::ffi::c_uint,
    stream: cudaStream_t,
) -> cudaError_t {
    let sym: Symbol<cudaMemPrefetchAsync_fn> = unsafe {
        get_lib()
            .get(b"cudaMemPrefetchAsync\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(devPtr, count, location, flags, stream) }
}

type cudaMemPrefetchBatchAsync_fn = unsafe extern "C" fn(
    dptrs: *mut *mut ::core::ffi::c_void,
    sizes: *mut usize,
    count: usize,
    prefetchLocs: *mut cudaMemLocation,
    prefetchLocIdxs: *mut usize,
    numPrefetchLocs: usize,
    flags: ::core::ffi::c_ulonglong,
    stream: cudaStream_t,
) -> cudaError_t;
pub unsafe fn cudaMemPrefetchBatchAsync(
    dptrs: *mut *mut ::core::ffi::c_void,
    sizes: *mut usize,
    count: usize,
    prefetchLocs: *mut cudaMemLocation,
    prefetchLocIdxs: *mut usize,
    numPrefetchLocs: usize,
    flags: ::core::ffi::c_ulonglong,
    stream: cudaStream_t,
) -> cudaError_t {
    let sym: Symbol<cudaMemPrefetchBatchAsync_fn> = unsafe {
        get_lib()
            .get(b"cudaMemPrefetchBatchAsync\0")
            .expect("Missing symbol")
    };
    unsafe {
        (*sym)(
            dptrs,
            sizes,
            count,
            prefetchLocs,
            prefetchLocIdxs,
            numPrefetchLocs,
            flags,
            stream,
        )
    }
}

type cudaMemDiscardBatchAsync_fn = unsafe extern "C" fn(
    dptrs: *mut *mut ::core::ffi::c_void,
    sizes: *mut usize,
    count: usize,
    flags: ::core::ffi::c_ulonglong,
    stream: cudaStream_t,
) -> cudaError_t;
pub unsafe fn cudaMemDiscardBatchAsync(
    dptrs: *mut *mut ::core::ffi::c_void,
    sizes: *mut usize,
    count: usize,
    flags: ::core::ffi::c_ulonglong,
    stream: cudaStream_t,
) -> cudaError_t {
    let sym: Symbol<cudaMemDiscardBatchAsync_fn> = unsafe {
        get_lib()
            .get(b"cudaMemDiscardBatchAsync\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(dptrs, sizes, count, flags, stream) }
}

type cudaMemDiscardAndPrefetchBatchAsync_fn = unsafe extern "C" fn(
    dptrs: *mut *mut ::core::ffi::c_void,
    sizes: *mut usize,
    count: usize,
    prefetchLocs: *mut cudaMemLocation,
    prefetchLocIdxs: *mut usize,
    numPrefetchLocs: usize,
    flags: ::core::ffi::c_ulonglong,
    stream: cudaStream_t,
) -> cudaError_t;
pub unsafe fn cudaMemDiscardAndPrefetchBatchAsync(
    dptrs: *mut *mut ::core::ffi::c_void,
    sizes: *mut usize,
    count: usize,
    prefetchLocs: *mut cudaMemLocation,
    prefetchLocIdxs: *mut usize,
    numPrefetchLocs: usize,
    flags: ::core::ffi::c_ulonglong,
    stream: cudaStream_t,
) -> cudaError_t {
    let sym: Symbol<cudaMemDiscardAndPrefetchBatchAsync_fn> = unsafe {
        get_lib()
            .get(b"cudaMemDiscardAndPrefetchBatchAsync\0")
            .expect("Missing symbol")
    };
    unsafe {
        (*sym)(
            dptrs,
            sizes,
            count,
            prefetchLocs,
            prefetchLocIdxs,
            numPrefetchLocs,
            flags,
            stream,
        )
    }
}

type cudaMemAdvise_fn = unsafe extern "C" fn(
    devPtr: *const ::core::ffi::c_void,
    count: usize,
    advice: cudaMemoryAdvise,
    location: cudaMemLocation,
) -> cudaError_t;
pub unsafe fn cudaMemAdvise(
    devPtr: *const ::core::ffi::c_void,
    count: usize,
    advice: cudaMemoryAdvise,
    location: cudaMemLocation,
) -> cudaError_t {
    let sym: Symbol<cudaMemAdvise_fn> =
        unsafe { get_lib().get(b"cudaMemAdvise\0").expect("Missing symbol") };
    unsafe { (*sym)(devPtr, count, advice, location) }
}

type cudaMemRangeGetAttribute_fn = unsafe extern "C" fn(
    data: *mut ::core::ffi::c_void,
    dataSize: usize,
    attribute: cudaMemRangeAttribute,
    devPtr: *const ::core::ffi::c_void,
    count: usize,
) -> cudaError_t;
pub unsafe fn cudaMemRangeGetAttribute(
    data: *mut ::core::ffi::c_void,
    dataSize: usize,
    attribute: cudaMemRangeAttribute,
    devPtr: *const ::core::ffi::c_void,
    count: usize,
) -> cudaError_t {
    let sym: Symbol<cudaMemRangeGetAttribute_fn> = unsafe {
        get_lib()
            .get(b"cudaMemRangeGetAttribute\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(data, dataSize, attribute, devPtr, count) }
}

type cudaMemRangeGetAttributes_fn = unsafe extern "C" fn(
    data: *mut *mut ::core::ffi::c_void,
    dataSizes: *mut usize,
    attributes: *mut cudaMemRangeAttribute,
    numAttributes: usize,
    devPtr: *const ::core::ffi::c_void,
    count: usize,
) -> cudaError_t;
pub unsafe fn cudaMemRangeGetAttributes(
    data: *mut *mut ::core::ffi::c_void,
    dataSizes: *mut usize,
    attributes: *mut cudaMemRangeAttribute,
    numAttributes: usize,
    devPtr: *const ::core::ffi::c_void,
    count: usize,
) -> cudaError_t {
    let sym: Symbol<cudaMemRangeGetAttributes_fn> = unsafe {
        get_lib()
            .get(b"cudaMemRangeGetAttributes\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(data, dataSizes, attributes, numAttributes, devPtr, count) }
}

type cudaMemcpyToArray_fn = unsafe extern "C" fn(
    dst: cudaArray_t,
    wOffset: usize,
    hOffset: usize,
    src: *const ::core::ffi::c_void,
    count: usize,
    kind: cudaMemcpyKind,
) -> cudaError_t;
pub unsafe fn cudaMemcpyToArray(
    dst: cudaArray_t,
    wOffset: usize,
    hOffset: usize,
    src: *const ::core::ffi::c_void,
    count: usize,
    kind: cudaMemcpyKind,
) -> cudaError_t {
    let sym: Symbol<cudaMemcpyToArray_fn> = unsafe {
        get_lib()
            .get(b"cudaMemcpyToArray\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(dst, wOffset, hOffset, src, count, kind) }
}

type cudaMemcpyFromArray_fn = unsafe extern "C" fn(
    dst: *mut ::core::ffi::c_void,
    src: cudaArray_const_t,
    wOffset: usize,
    hOffset: usize,
    count: usize,
    kind: cudaMemcpyKind,
) -> cudaError_t;
pub unsafe fn cudaMemcpyFromArray(
    dst: *mut ::core::ffi::c_void,
    src: cudaArray_const_t,
    wOffset: usize,
    hOffset: usize,
    count: usize,
    kind: cudaMemcpyKind,
) -> cudaError_t {
    let sym: Symbol<cudaMemcpyFromArray_fn> = unsafe {
        get_lib()
            .get(b"cudaMemcpyFromArray\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(dst, src, wOffset, hOffset, count, kind) }
}

type cudaMemcpyArrayToArray_fn = unsafe extern "C" fn(
    dst: cudaArray_t,
    wOffsetDst: usize,
    hOffsetDst: usize,
    src: cudaArray_const_t,
    wOffsetSrc: usize,
    hOffsetSrc: usize,
    count: usize,
    kind: cudaMemcpyKind,
) -> cudaError_t;
pub unsafe fn cudaMemcpyArrayToArray(
    dst: cudaArray_t,
    wOffsetDst: usize,
    hOffsetDst: usize,
    src: cudaArray_const_t,
    wOffsetSrc: usize,
    hOffsetSrc: usize,
    count: usize,
    kind: cudaMemcpyKind,
) -> cudaError_t {
    let sym: Symbol<cudaMemcpyArrayToArray_fn> = unsafe {
        get_lib()
            .get(b"cudaMemcpyArrayToArray\0")
            .expect("Missing symbol")
    };
    unsafe {
        (*sym)(
            dst, wOffsetDst, hOffsetDst, src, wOffsetSrc, hOffsetSrc, count, kind,
        )
    }
}

type cudaMemcpyToArrayAsync_fn = unsafe extern "C" fn(
    dst: cudaArray_t,
    wOffset: usize,
    hOffset: usize,
    src: *const ::core::ffi::c_void,
    count: usize,
    kind: cudaMemcpyKind,
    stream: cudaStream_t,
) -> cudaError_t;
pub unsafe fn cudaMemcpyToArrayAsync(
    dst: cudaArray_t,
    wOffset: usize,
    hOffset: usize,
    src: *const ::core::ffi::c_void,
    count: usize,
    kind: cudaMemcpyKind,
    stream: cudaStream_t,
) -> cudaError_t {
    let sym: Symbol<cudaMemcpyToArrayAsync_fn> = unsafe {
        get_lib()
            .get(b"cudaMemcpyToArrayAsync\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(dst, wOffset, hOffset, src, count, kind, stream) }
}

type cudaMemcpyFromArrayAsync_fn = unsafe extern "C" fn(
    dst: *mut ::core::ffi::c_void,
    src: cudaArray_const_t,
    wOffset: usize,
    hOffset: usize,
    count: usize,
    kind: cudaMemcpyKind,
    stream: cudaStream_t,
) -> cudaError_t;
pub unsafe fn cudaMemcpyFromArrayAsync(
    dst: *mut ::core::ffi::c_void,
    src: cudaArray_const_t,
    wOffset: usize,
    hOffset: usize,
    count: usize,
    kind: cudaMemcpyKind,
    stream: cudaStream_t,
) -> cudaError_t {
    let sym: Symbol<cudaMemcpyFromArrayAsync_fn> = unsafe {
        get_lib()
            .get(b"cudaMemcpyFromArrayAsync\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(dst, src, wOffset, hOffset, count, kind, stream) }
}

type cudaMallocAsync_fn = unsafe extern "C" fn(
    devPtr: *mut *mut ::core::ffi::c_void,
    size: usize,
    hStream: cudaStream_t,
) -> cudaError_t;
pub unsafe fn cudaMallocAsync(
    devPtr: *mut *mut ::core::ffi::c_void,
    size: usize,
    hStream: cudaStream_t,
) -> cudaError_t {
    let sym: Symbol<cudaMallocAsync_fn> =
        unsafe { get_lib().get(b"cudaMallocAsync\0").expect("Missing symbol") };
    unsafe { (*sym)(devPtr, size, hStream) }
}

type cudaFreeAsync_fn =
    unsafe extern "C" fn(devPtr: *mut ::core::ffi::c_void, hStream: cudaStream_t) -> cudaError_t;
pub unsafe fn cudaFreeAsync(
    devPtr: *mut ::core::ffi::c_void,
    hStream: cudaStream_t,
) -> cudaError_t {
    let sym: Symbol<cudaFreeAsync_fn> =
        unsafe { get_lib().get(b"cudaFreeAsync\0").expect("Missing symbol") };
    unsafe { (*sym)(devPtr, hStream) }
}

type cudaMemPoolTrimTo_fn =
    unsafe extern "C" fn(memPool: cudaMemPool_t, minBytesToKeep: usize) -> cudaError_t;
pub unsafe fn cudaMemPoolTrimTo(memPool: cudaMemPool_t, minBytesToKeep: usize) -> cudaError_t {
    let sym: Symbol<cudaMemPoolTrimTo_fn> = unsafe {
        get_lib()
            .get(b"cudaMemPoolTrimTo\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(memPool, minBytesToKeep) }
}

type cudaMemPoolSetAttribute_fn = unsafe extern "C" fn(
    memPool: cudaMemPool_t,
    attr: cudaMemPoolAttr,
    value: *mut ::core::ffi::c_void,
) -> cudaError_t;
pub unsafe fn cudaMemPoolSetAttribute(
    memPool: cudaMemPool_t,
    attr: cudaMemPoolAttr,
    value: *mut ::core::ffi::c_void,
) -> cudaError_t {
    let sym: Symbol<cudaMemPoolSetAttribute_fn> = unsafe {
        get_lib()
            .get(b"cudaMemPoolSetAttribute\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(memPool, attr, value) }
}

type cudaMemPoolGetAttribute_fn = unsafe extern "C" fn(
    memPool: cudaMemPool_t,
    attr: cudaMemPoolAttr,
    value: *mut ::core::ffi::c_void,
) -> cudaError_t;
pub unsafe fn cudaMemPoolGetAttribute(
    memPool: cudaMemPool_t,
    attr: cudaMemPoolAttr,
    value: *mut ::core::ffi::c_void,
) -> cudaError_t {
    let sym: Symbol<cudaMemPoolGetAttribute_fn> = unsafe {
        get_lib()
            .get(b"cudaMemPoolGetAttribute\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(memPool, attr, value) }
}

type cudaMemPoolSetAccess_fn = unsafe extern "C" fn(
    memPool: cudaMemPool_t,
    descList: *const cudaMemAccessDesc,
    count: usize,
) -> cudaError_t;
pub unsafe fn cudaMemPoolSetAccess(
    memPool: cudaMemPool_t,
    descList: *const cudaMemAccessDesc,
    count: usize,
) -> cudaError_t {
    let sym: Symbol<cudaMemPoolSetAccess_fn> = unsafe {
        get_lib()
            .get(b"cudaMemPoolSetAccess\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(memPool, descList, count) }
}

type cudaMemPoolGetAccess_fn = unsafe extern "C" fn(
    flags: *mut cudaMemAccessFlags,
    memPool: cudaMemPool_t,
    location: *mut cudaMemLocation,
) -> cudaError_t;
pub unsafe fn cudaMemPoolGetAccess(
    flags: *mut cudaMemAccessFlags,
    memPool: cudaMemPool_t,
    location: *mut cudaMemLocation,
) -> cudaError_t {
    let sym: Symbol<cudaMemPoolGetAccess_fn> = unsafe {
        get_lib()
            .get(b"cudaMemPoolGetAccess\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(flags, memPool, location) }
}

type cudaMemPoolCreate_fn = unsafe extern "C" fn(
    memPool: *mut cudaMemPool_t,
    poolProps: *const cudaMemPoolProps,
) -> cudaError_t;
pub unsafe fn cudaMemPoolCreate(
    memPool: *mut cudaMemPool_t,
    poolProps: *const cudaMemPoolProps,
) -> cudaError_t {
    let sym: Symbol<cudaMemPoolCreate_fn> = unsafe {
        get_lib()
            .get(b"cudaMemPoolCreate\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(memPool, poolProps) }
}

type cudaMemPoolDestroy_fn = unsafe extern "C" fn(memPool: cudaMemPool_t) -> cudaError_t;
pub unsafe fn cudaMemPoolDestroy(memPool: cudaMemPool_t) -> cudaError_t {
    let sym: Symbol<cudaMemPoolDestroy_fn> = unsafe {
        get_lib()
            .get(b"cudaMemPoolDestroy\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(memPool) }
}

type cudaMemGetDefaultMemPool_fn = unsafe extern "C" fn(
    memPool: *mut cudaMemPool_t,
    location: *mut cudaMemLocation,
    type_: cudaMemAllocationType,
) -> cudaError_t;
pub unsafe fn cudaMemGetDefaultMemPool(
    memPool: *mut cudaMemPool_t,
    location: *mut cudaMemLocation,
    type_: cudaMemAllocationType,
) -> cudaError_t {
    let sym: Symbol<cudaMemGetDefaultMemPool_fn> = unsafe {
        get_lib()
            .get(b"cudaMemGetDefaultMemPool\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(memPool, location, type_) }
}

type cudaMemGetMemPool_fn = unsafe extern "C" fn(
    memPool: *mut cudaMemPool_t,
    location: *mut cudaMemLocation,
    type_: cudaMemAllocationType,
) -> cudaError_t;
pub unsafe fn cudaMemGetMemPool(
    memPool: *mut cudaMemPool_t,
    location: *mut cudaMemLocation,
    type_: cudaMemAllocationType,
) -> cudaError_t {
    let sym: Symbol<cudaMemGetMemPool_fn> = unsafe {
        get_lib()
            .get(b"cudaMemGetMemPool\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(memPool, location, type_) }
}

type cudaMemSetMemPool_fn = unsafe extern "C" fn(
    location: *mut cudaMemLocation,
    type_: cudaMemAllocationType,
    memPool: cudaMemPool_t,
) -> cudaError_t;
pub unsafe fn cudaMemSetMemPool(
    location: *mut cudaMemLocation,
    type_: cudaMemAllocationType,
    memPool: cudaMemPool_t,
) -> cudaError_t {
    let sym: Symbol<cudaMemSetMemPool_fn> = unsafe {
        get_lib()
            .get(b"cudaMemSetMemPool\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(location, type_, memPool) }
}

type cudaMallocFromPoolAsync_fn = unsafe extern "C" fn(
    ptr: *mut *mut ::core::ffi::c_void,
    size: usize,
    memPool: cudaMemPool_t,
    stream: cudaStream_t,
) -> cudaError_t;
pub unsafe fn cudaMallocFromPoolAsync(
    ptr: *mut *mut ::core::ffi::c_void,
    size: usize,
    memPool: cudaMemPool_t,
    stream: cudaStream_t,
) -> cudaError_t {
    let sym: Symbol<cudaMallocFromPoolAsync_fn> = unsafe {
        get_lib()
            .get(b"cudaMallocFromPoolAsync\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(ptr, size, memPool, stream) }
}

type cudaMemPoolExportToShareableHandle_fn = unsafe extern "C" fn(
    shareableHandle: *mut ::core::ffi::c_void,
    memPool: cudaMemPool_t,
    handleType: cudaMemAllocationHandleType,
    flags: ::core::ffi::c_uint,
) -> cudaError_t;
pub unsafe fn cudaMemPoolExportToShareableHandle(
    shareableHandle: *mut ::core::ffi::c_void,
    memPool: cudaMemPool_t,
    handleType: cudaMemAllocationHandleType,
    flags: ::core::ffi::c_uint,
) -> cudaError_t {
    let sym: Symbol<cudaMemPoolExportToShareableHandle_fn> = unsafe {
        get_lib()
            .get(b"cudaMemPoolExportToShareableHandle\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(shareableHandle, memPool, handleType, flags) }
}

type cudaMemPoolImportFromShareableHandle_fn = unsafe extern "C" fn(
    memPool: *mut cudaMemPool_t,
    shareableHandle: *mut ::core::ffi::c_void,
    handleType: cudaMemAllocationHandleType,
    flags: ::core::ffi::c_uint,
) -> cudaError_t;
pub unsafe fn cudaMemPoolImportFromShareableHandle(
    memPool: *mut cudaMemPool_t,
    shareableHandle: *mut ::core::ffi::c_void,
    handleType: cudaMemAllocationHandleType,
    flags: ::core::ffi::c_uint,
) -> cudaError_t {
    let sym: Symbol<cudaMemPoolImportFromShareableHandle_fn> = unsafe {
        get_lib()
            .get(b"cudaMemPoolImportFromShareableHandle\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(memPool, shareableHandle, handleType, flags) }
}

type cudaMemPoolExportPointer_fn = unsafe extern "C" fn(
    exportData: *mut cudaMemPoolPtrExportData,
    ptr: *mut ::core::ffi::c_void,
) -> cudaError_t;
pub unsafe fn cudaMemPoolExportPointer(
    exportData: *mut cudaMemPoolPtrExportData,
    ptr: *mut ::core::ffi::c_void,
) -> cudaError_t {
    let sym: Symbol<cudaMemPoolExportPointer_fn> = unsafe {
        get_lib()
            .get(b"cudaMemPoolExportPointer\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(exportData, ptr) }
}

type cudaMemPoolImportPointer_fn = unsafe extern "C" fn(
    ptr: *mut *mut ::core::ffi::c_void,
    memPool: cudaMemPool_t,
    exportData: *mut cudaMemPoolPtrExportData,
) -> cudaError_t;
pub unsafe fn cudaMemPoolImportPointer(
    ptr: *mut *mut ::core::ffi::c_void,
    memPool: cudaMemPool_t,
    exportData: *mut cudaMemPoolPtrExportData,
) -> cudaError_t {
    let sym: Symbol<cudaMemPoolImportPointer_fn> = unsafe {
        get_lib()
            .get(b"cudaMemPoolImportPointer\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(ptr, memPool, exportData) }
}

type cudaPointerGetAttributes_fn = unsafe extern "C" fn(
    attributes: *mut cudaPointerAttributes,
    ptr: *const ::core::ffi::c_void,
) -> cudaError_t;
pub unsafe fn cudaPointerGetAttributes(
    attributes: *mut cudaPointerAttributes,
    ptr: *const ::core::ffi::c_void,
) -> cudaError_t {
    let sym: Symbol<cudaPointerGetAttributes_fn> = unsafe {
        get_lib()
            .get(b"cudaPointerGetAttributes\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(attributes, ptr) }
}

type cudaDeviceCanAccessPeer_fn = unsafe extern "C" fn(
    canAccessPeer: *mut ::core::ffi::c_int,
    device: ::core::ffi::c_int,
    peerDevice: ::core::ffi::c_int,
) -> cudaError_t;
pub unsafe fn cudaDeviceCanAccessPeer(
    canAccessPeer: *mut ::core::ffi::c_int,
    device: ::core::ffi::c_int,
    peerDevice: ::core::ffi::c_int,
) -> cudaError_t {
    let sym: Symbol<cudaDeviceCanAccessPeer_fn> = unsafe {
        get_lib()
            .get(b"cudaDeviceCanAccessPeer\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(canAccessPeer, device, peerDevice) }
}

type cudaDeviceEnablePeerAccess_fn =
    unsafe extern "C" fn(peerDevice: ::core::ffi::c_int, flags: ::core::ffi::c_uint) -> cudaError_t;
pub unsafe fn cudaDeviceEnablePeerAccess(
    peerDevice: ::core::ffi::c_int,
    flags: ::core::ffi::c_uint,
) -> cudaError_t {
    let sym: Symbol<cudaDeviceEnablePeerAccess_fn> = unsafe {
        get_lib()
            .get(b"cudaDeviceEnablePeerAccess\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(peerDevice, flags) }
}

type cudaDeviceDisablePeerAccess_fn =
    unsafe extern "C" fn(peerDevice: ::core::ffi::c_int) -> cudaError_t;
pub unsafe fn cudaDeviceDisablePeerAccess(peerDevice: ::core::ffi::c_int) -> cudaError_t {
    let sym: Symbol<cudaDeviceDisablePeerAccess_fn> = unsafe {
        get_lib()
            .get(b"cudaDeviceDisablePeerAccess\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(peerDevice) }
}

type cudaGraphicsUnregisterResource_fn =
    unsafe extern "C" fn(resource: cudaGraphicsResource_t) -> cudaError_t;
pub unsafe fn cudaGraphicsUnregisterResource(resource: cudaGraphicsResource_t) -> cudaError_t {
    let sym: Symbol<cudaGraphicsUnregisterResource_fn> = unsafe {
        get_lib()
            .get(b"cudaGraphicsUnregisterResource\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(resource) }
}

type cudaGraphicsResourceSetMapFlags_fn = unsafe extern "C" fn(
    resource: cudaGraphicsResource_t,
    flags: ::core::ffi::c_uint,
) -> cudaError_t;
pub unsafe fn cudaGraphicsResourceSetMapFlags(
    resource: cudaGraphicsResource_t,
    flags: ::core::ffi::c_uint,
) -> cudaError_t {
    let sym: Symbol<cudaGraphicsResourceSetMapFlags_fn> = unsafe {
        get_lib()
            .get(b"cudaGraphicsResourceSetMapFlags\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(resource, flags) }
}

type cudaGraphicsMapResources_fn = unsafe extern "C" fn(
    count: ::core::ffi::c_int,
    resources: *mut cudaGraphicsResource_t,
    stream: cudaStream_t,
) -> cudaError_t;
pub unsafe fn cudaGraphicsMapResources(
    count: ::core::ffi::c_int,
    resources: *mut cudaGraphicsResource_t,
    stream: cudaStream_t,
) -> cudaError_t {
    let sym: Symbol<cudaGraphicsMapResources_fn> = unsafe {
        get_lib()
            .get(b"cudaGraphicsMapResources\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(count, resources, stream) }
}

type cudaGraphicsUnmapResources_fn = unsafe extern "C" fn(
    count: ::core::ffi::c_int,
    resources: *mut cudaGraphicsResource_t,
    stream: cudaStream_t,
) -> cudaError_t;
pub unsafe fn cudaGraphicsUnmapResources(
    count: ::core::ffi::c_int,
    resources: *mut cudaGraphicsResource_t,
    stream: cudaStream_t,
) -> cudaError_t {
    let sym: Symbol<cudaGraphicsUnmapResources_fn> = unsafe {
        get_lib()
            .get(b"cudaGraphicsUnmapResources\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(count, resources, stream) }
}

type cudaGraphicsResourceGetMappedPointer_fn = unsafe extern "C" fn(
    devPtr: *mut *mut ::core::ffi::c_void,
    size: *mut usize,
    resource: cudaGraphicsResource_t,
) -> cudaError_t;
pub unsafe fn cudaGraphicsResourceGetMappedPointer(
    devPtr: *mut *mut ::core::ffi::c_void,
    size: *mut usize,
    resource: cudaGraphicsResource_t,
) -> cudaError_t {
    let sym: Symbol<cudaGraphicsResourceGetMappedPointer_fn> = unsafe {
        get_lib()
            .get(b"cudaGraphicsResourceGetMappedPointer\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(devPtr, size, resource) }
}

type cudaGraphicsSubResourceGetMappedArray_fn = unsafe extern "C" fn(
    array: *mut cudaArray_t,
    resource: cudaGraphicsResource_t,
    arrayIndex: ::core::ffi::c_uint,
    mipLevel: ::core::ffi::c_uint,
) -> cudaError_t;
pub unsafe fn cudaGraphicsSubResourceGetMappedArray(
    array: *mut cudaArray_t,
    resource: cudaGraphicsResource_t,
    arrayIndex: ::core::ffi::c_uint,
    mipLevel: ::core::ffi::c_uint,
) -> cudaError_t {
    let sym: Symbol<cudaGraphicsSubResourceGetMappedArray_fn> = unsafe {
        get_lib()
            .get(b"cudaGraphicsSubResourceGetMappedArray\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(array, resource, arrayIndex, mipLevel) }
}

type cudaGraphicsResourceGetMappedMipmappedArray_fn = unsafe extern "C" fn(
    mipmappedArray: *mut cudaMipmappedArray_t,
    resource: cudaGraphicsResource_t,
) -> cudaError_t;
pub unsafe fn cudaGraphicsResourceGetMappedMipmappedArray(
    mipmappedArray: *mut cudaMipmappedArray_t,
    resource: cudaGraphicsResource_t,
) -> cudaError_t {
    let sym: Symbol<cudaGraphicsResourceGetMappedMipmappedArray_fn> = unsafe {
        get_lib()
            .get(b"cudaGraphicsResourceGetMappedMipmappedArray\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(mipmappedArray, resource) }
}

type cudaGetChannelDesc_fn =
    unsafe extern "C" fn(desc: *mut cudaChannelFormatDesc, array: cudaArray_const_t) -> cudaError_t;
pub unsafe fn cudaGetChannelDesc(
    desc: *mut cudaChannelFormatDesc,
    array: cudaArray_const_t,
) -> cudaError_t {
    let sym: Symbol<cudaGetChannelDesc_fn> = unsafe {
        get_lib()
            .get(b"cudaGetChannelDesc\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(desc, array) }
}

type cudaCreateChannelDesc_fn = unsafe extern "C" fn(
    x: ::core::ffi::c_int,
    y: ::core::ffi::c_int,
    z: ::core::ffi::c_int,
    w: ::core::ffi::c_int,
    f: cudaChannelFormatKind,
) -> cudaChannelFormatDesc;
pub unsafe fn cudaCreateChannelDesc(
    x: ::core::ffi::c_int,
    y: ::core::ffi::c_int,
    z: ::core::ffi::c_int,
    w: ::core::ffi::c_int,
    f: cudaChannelFormatKind,
) -> cudaChannelFormatDesc {
    let sym: Symbol<cudaCreateChannelDesc_fn> = unsafe {
        get_lib()
            .get(b"cudaCreateChannelDesc\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(x, y, z, w, f) }
}

type cudaCreateTextureObject_fn = unsafe extern "C" fn(
    pTexObject: *mut cudaTextureObject_t,
    pResDesc: *const cudaResourceDesc,
    pTexDesc: *const cudaTextureDesc,
    pResViewDesc: *const cudaResourceViewDesc,
) -> cudaError_t;
pub unsafe fn cudaCreateTextureObject(
    pTexObject: *mut cudaTextureObject_t,
    pResDesc: *const cudaResourceDesc,
    pTexDesc: *const cudaTextureDesc,
    pResViewDesc: *const cudaResourceViewDesc,
) -> cudaError_t {
    let sym: Symbol<cudaCreateTextureObject_fn> = unsafe {
        get_lib()
            .get(b"cudaCreateTextureObject\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(pTexObject, pResDesc, pTexDesc, pResViewDesc) }
}

type cudaDestroyTextureObject_fn =
    unsafe extern "C" fn(texObject: cudaTextureObject_t) -> cudaError_t;
pub unsafe fn cudaDestroyTextureObject(texObject: cudaTextureObject_t) -> cudaError_t {
    let sym: Symbol<cudaDestroyTextureObject_fn> = unsafe {
        get_lib()
            .get(b"cudaDestroyTextureObject\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(texObject) }
}

type cudaGetTextureObjectResourceDesc_fn = unsafe extern "C" fn(
    pResDesc: *mut cudaResourceDesc,
    texObject: cudaTextureObject_t,
) -> cudaError_t;
pub unsafe fn cudaGetTextureObjectResourceDesc(
    pResDesc: *mut cudaResourceDesc,
    texObject: cudaTextureObject_t,
) -> cudaError_t {
    let sym: Symbol<cudaGetTextureObjectResourceDesc_fn> = unsafe {
        get_lib()
            .get(b"cudaGetTextureObjectResourceDesc\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(pResDesc, texObject) }
}

type cudaGetTextureObjectTextureDesc_fn = unsafe extern "C" fn(
    pTexDesc: *mut cudaTextureDesc,
    texObject: cudaTextureObject_t,
) -> cudaError_t;
pub unsafe fn cudaGetTextureObjectTextureDesc(
    pTexDesc: *mut cudaTextureDesc,
    texObject: cudaTextureObject_t,
) -> cudaError_t {
    let sym: Symbol<cudaGetTextureObjectTextureDesc_fn> = unsafe {
        get_lib()
            .get(b"cudaGetTextureObjectTextureDesc\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(pTexDesc, texObject) }
}

type cudaGetTextureObjectResourceViewDesc_fn = unsafe extern "C" fn(
    pResViewDesc: *mut cudaResourceViewDesc,
    texObject: cudaTextureObject_t,
) -> cudaError_t;
pub unsafe fn cudaGetTextureObjectResourceViewDesc(
    pResViewDesc: *mut cudaResourceViewDesc,
    texObject: cudaTextureObject_t,
) -> cudaError_t {
    let sym: Symbol<cudaGetTextureObjectResourceViewDesc_fn> = unsafe {
        get_lib()
            .get(b"cudaGetTextureObjectResourceViewDesc\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(pResViewDesc, texObject) }
}

type cudaCreateSurfaceObject_fn = unsafe extern "C" fn(
    pSurfObject: *mut cudaSurfaceObject_t,
    pResDesc: *const cudaResourceDesc,
) -> cudaError_t;
pub unsafe fn cudaCreateSurfaceObject(
    pSurfObject: *mut cudaSurfaceObject_t,
    pResDesc: *const cudaResourceDesc,
) -> cudaError_t {
    let sym: Symbol<cudaCreateSurfaceObject_fn> = unsafe {
        get_lib()
            .get(b"cudaCreateSurfaceObject\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(pSurfObject, pResDesc) }
}

type cudaDestroySurfaceObject_fn =
    unsafe extern "C" fn(surfObject: cudaSurfaceObject_t) -> cudaError_t;
pub unsafe fn cudaDestroySurfaceObject(surfObject: cudaSurfaceObject_t) -> cudaError_t {
    let sym: Symbol<cudaDestroySurfaceObject_fn> = unsafe {
        get_lib()
            .get(b"cudaDestroySurfaceObject\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(surfObject) }
}

type cudaGetSurfaceObjectResourceDesc_fn = unsafe extern "C" fn(
    pResDesc: *mut cudaResourceDesc,
    surfObject: cudaSurfaceObject_t,
) -> cudaError_t;
pub unsafe fn cudaGetSurfaceObjectResourceDesc(
    pResDesc: *mut cudaResourceDesc,
    surfObject: cudaSurfaceObject_t,
) -> cudaError_t {
    let sym: Symbol<cudaGetSurfaceObjectResourceDesc_fn> = unsafe {
        get_lib()
            .get(b"cudaGetSurfaceObjectResourceDesc\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(pResDesc, surfObject) }
}

type cudaDriverGetVersion_fn =
    unsafe extern "C" fn(driverVersion: *mut ::core::ffi::c_int) -> cudaError_t;
pub unsafe fn cudaDriverGetVersion(driverVersion: *mut ::core::ffi::c_int) -> cudaError_t {
    let sym: Symbol<cudaDriverGetVersion_fn> = unsafe {
        get_lib()
            .get(b"cudaDriverGetVersion\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(driverVersion) }
}

type cudaRuntimeGetVersion_fn =
    unsafe extern "C" fn(runtimeVersion: *mut ::core::ffi::c_int) -> cudaError_t;
pub unsafe fn cudaRuntimeGetVersion(runtimeVersion: *mut ::core::ffi::c_int) -> cudaError_t {
    let sym: Symbol<cudaRuntimeGetVersion_fn> = unsafe {
        get_lib()
            .get(b"cudaRuntimeGetVersion\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(runtimeVersion) }
}

type cudaLogsRegisterCallback_fn = unsafe extern "C" fn(
    callbackFunc: cudaLogsCallback_t,
    userData: *mut ::core::ffi::c_void,
    callback_out: *mut cudaLogsCallbackHandle,
) -> cudaError_t;
pub unsafe fn cudaLogsRegisterCallback(
    callbackFunc: cudaLogsCallback_t,
    userData: *mut ::core::ffi::c_void,
    callback_out: *mut cudaLogsCallbackHandle,
) -> cudaError_t {
    let sym: Symbol<cudaLogsRegisterCallback_fn> = unsafe {
        get_lib()
            .get(b"cudaLogsRegisterCallback\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(callbackFunc, userData, callback_out) }
}

type cudaLogsUnregisterCallback_fn =
    unsafe extern "C" fn(callback: cudaLogsCallbackHandle) -> cudaError_t;
pub unsafe fn cudaLogsUnregisterCallback(callback: cudaLogsCallbackHandle) -> cudaError_t {
    let sym: Symbol<cudaLogsUnregisterCallback_fn> = unsafe {
        get_lib()
            .get(b"cudaLogsUnregisterCallback\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(callback) }
}

type cudaLogsCurrent_fn = unsafe extern "C" fn(
    iterator_out: *mut cudaLogIterator,
    flags: ::core::ffi::c_uint,
) -> cudaError_t;
pub unsafe fn cudaLogsCurrent(
    iterator_out: *mut cudaLogIterator,
    flags: ::core::ffi::c_uint,
) -> cudaError_t {
    let sym: Symbol<cudaLogsCurrent_fn> =
        unsafe { get_lib().get(b"cudaLogsCurrent\0").expect("Missing symbol") };
    unsafe { (*sym)(iterator_out, flags) }
}

type cudaLogsDumpToFile_fn = unsafe extern "C" fn(
    iterator: *mut cudaLogIterator,
    pathToFile: *const ::core::ffi::c_char,
    flags: ::core::ffi::c_uint,
) -> cudaError_t;
pub unsafe fn cudaLogsDumpToFile(
    iterator: *mut cudaLogIterator,
    pathToFile: *const ::core::ffi::c_char,
    flags: ::core::ffi::c_uint,
) -> cudaError_t {
    let sym: Symbol<cudaLogsDumpToFile_fn> = unsafe {
        get_lib()
            .get(b"cudaLogsDumpToFile\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(iterator, pathToFile, flags) }
}

type cudaLogsDumpToMemory_fn = unsafe extern "C" fn(
    iterator: *mut cudaLogIterator,
    buffer: *mut ::core::ffi::c_char,
    size: *mut usize,
    flags: ::core::ffi::c_uint,
) -> cudaError_t;
pub unsafe fn cudaLogsDumpToMemory(
    iterator: *mut cudaLogIterator,
    buffer: *mut ::core::ffi::c_char,
    size: *mut usize,
    flags: ::core::ffi::c_uint,
) -> cudaError_t {
    let sym: Symbol<cudaLogsDumpToMemory_fn> = unsafe {
        get_lib()
            .get(b"cudaLogsDumpToMemory\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(iterator, buffer, size, flags) }
}

type cudaGraphCreate_fn =
    unsafe extern "C" fn(pGraph: *mut cudaGraph_t, flags: ::core::ffi::c_uint) -> cudaError_t;
pub unsafe fn cudaGraphCreate(pGraph: *mut cudaGraph_t, flags: ::core::ffi::c_uint) -> cudaError_t {
    let sym: Symbol<cudaGraphCreate_fn> =
        unsafe { get_lib().get(b"cudaGraphCreate\0").expect("Missing symbol") };
    unsafe { (*sym)(pGraph, flags) }
}

type cudaGraphAddKernelNode_fn = unsafe extern "C" fn(
    pGraphNode: *mut cudaGraphNode_t,
    graph: cudaGraph_t,
    pDependencies: *const cudaGraphNode_t,
    numDependencies: usize,
    pNodeParams: *const cudaKernelNodeParams,
) -> cudaError_t;
pub unsafe fn cudaGraphAddKernelNode(
    pGraphNode: *mut cudaGraphNode_t,
    graph: cudaGraph_t,
    pDependencies: *const cudaGraphNode_t,
    numDependencies: usize,
    pNodeParams: *const cudaKernelNodeParams,
) -> cudaError_t {
    let sym: Symbol<cudaGraphAddKernelNode_fn> = unsafe {
        get_lib()
            .get(b"cudaGraphAddKernelNode\0")
            .expect("Missing symbol")
    };
    unsafe {
        (*sym)(
            pGraphNode,
            graph,
            pDependencies,
            numDependencies,
            pNodeParams,
        )
    }
}

type cudaGraphKernelNodeGetParams_fn = unsafe extern "C" fn(
    node: cudaGraphNode_t,
    pNodeParams: *mut cudaKernelNodeParams,
) -> cudaError_t;
pub unsafe fn cudaGraphKernelNodeGetParams(
    node: cudaGraphNode_t,
    pNodeParams: *mut cudaKernelNodeParams,
) -> cudaError_t {
    let sym: Symbol<cudaGraphKernelNodeGetParams_fn> = unsafe {
        get_lib()
            .get(b"cudaGraphKernelNodeGetParams\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(node, pNodeParams) }
}

type cudaGraphKernelNodeSetParams_fn = unsafe extern "C" fn(
    node: cudaGraphNode_t,
    pNodeParams: *const cudaKernelNodeParams,
) -> cudaError_t;
pub unsafe fn cudaGraphKernelNodeSetParams(
    node: cudaGraphNode_t,
    pNodeParams: *const cudaKernelNodeParams,
) -> cudaError_t {
    let sym: Symbol<cudaGraphKernelNodeSetParams_fn> = unsafe {
        get_lib()
            .get(b"cudaGraphKernelNodeSetParams\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(node, pNodeParams) }
}

type cudaGraphKernelNodeCopyAttributes_fn =
    unsafe extern "C" fn(hDst: cudaGraphNode_t, hSrc: cudaGraphNode_t) -> cudaError_t;
pub unsafe fn cudaGraphKernelNodeCopyAttributes(
    hDst: cudaGraphNode_t,
    hSrc: cudaGraphNode_t,
) -> cudaError_t {
    let sym: Symbol<cudaGraphKernelNodeCopyAttributes_fn> = unsafe {
        get_lib()
            .get(b"cudaGraphKernelNodeCopyAttributes\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(hDst, hSrc) }
}

type cudaGraphKernelNodeGetAttribute_fn = unsafe extern "C" fn(
    hNode: cudaGraphNode_t,
    attr: cudaLaunchAttributeID,
    value_out: *mut cudaLaunchAttributeValue,
) -> cudaError_t;
pub unsafe fn cudaGraphKernelNodeGetAttribute(
    hNode: cudaGraphNode_t,
    attr: cudaLaunchAttributeID,
    value_out: *mut cudaLaunchAttributeValue,
) -> cudaError_t {
    let sym: Symbol<cudaGraphKernelNodeGetAttribute_fn> = unsafe {
        get_lib()
            .get(b"cudaGraphKernelNodeGetAttribute\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(hNode, attr, value_out) }
}

type cudaGraphKernelNodeSetAttribute_fn = unsafe extern "C" fn(
    hNode: cudaGraphNode_t,
    attr: cudaLaunchAttributeID,
    value: *const cudaLaunchAttributeValue,
) -> cudaError_t;
pub unsafe fn cudaGraphKernelNodeSetAttribute(
    hNode: cudaGraphNode_t,
    attr: cudaLaunchAttributeID,
    value: *const cudaLaunchAttributeValue,
) -> cudaError_t {
    let sym: Symbol<cudaGraphKernelNodeSetAttribute_fn> = unsafe {
        get_lib()
            .get(b"cudaGraphKernelNodeSetAttribute\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(hNode, attr, value) }
}

type cudaGraphAddMemcpyNode_fn = unsafe extern "C" fn(
    pGraphNode: *mut cudaGraphNode_t,
    graph: cudaGraph_t,
    pDependencies: *const cudaGraphNode_t,
    numDependencies: usize,
    pCopyParams: *const cudaMemcpy3DParms,
) -> cudaError_t;
pub unsafe fn cudaGraphAddMemcpyNode(
    pGraphNode: *mut cudaGraphNode_t,
    graph: cudaGraph_t,
    pDependencies: *const cudaGraphNode_t,
    numDependencies: usize,
    pCopyParams: *const cudaMemcpy3DParms,
) -> cudaError_t {
    let sym: Symbol<cudaGraphAddMemcpyNode_fn> = unsafe {
        get_lib()
            .get(b"cudaGraphAddMemcpyNode\0")
            .expect("Missing symbol")
    };
    unsafe {
        (*sym)(
            pGraphNode,
            graph,
            pDependencies,
            numDependencies,
            pCopyParams,
        )
    }
}

type cudaGraphAddMemcpyNodeToSymbol_fn = unsafe extern "C" fn(
    pGraphNode: *mut cudaGraphNode_t,
    graph: cudaGraph_t,
    pDependencies: *const cudaGraphNode_t,
    numDependencies: usize,
    symbol: *const ::core::ffi::c_void,
    src: *const ::core::ffi::c_void,
    count: usize,
    offset: usize,
    kind: cudaMemcpyKind,
) -> cudaError_t;
pub unsafe fn cudaGraphAddMemcpyNodeToSymbol(
    pGraphNode: *mut cudaGraphNode_t,
    graph: cudaGraph_t,
    pDependencies: *const cudaGraphNode_t,
    numDependencies: usize,
    symbol: *const ::core::ffi::c_void,
    src: *const ::core::ffi::c_void,
    count: usize,
    offset: usize,
    kind: cudaMemcpyKind,
) -> cudaError_t {
    let sym: Symbol<cudaGraphAddMemcpyNodeToSymbol_fn> = unsafe {
        get_lib()
            .get(b"cudaGraphAddMemcpyNodeToSymbol\0")
            .expect("Missing symbol")
    };
    unsafe {
        (*sym)(
            pGraphNode,
            graph,
            pDependencies,
            numDependencies,
            symbol,
            src,
            count,
            offset,
            kind,
        )
    }
}

type cudaGraphAddMemcpyNodeFromSymbol_fn = unsafe extern "C" fn(
    pGraphNode: *mut cudaGraphNode_t,
    graph: cudaGraph_t,
    pDependencies: *const cudaGraphNode_t,
    numDependencies: usize,
    dst: *mut ::core::ffi::c_void,
    symbol: *const ::core::ffi::c_void,
    count: usize,
    offset: usize,
    kind: cudaMemcpyKind,
) -> cudaError_t;
pub unsafe fn cudaGraphAddMemcpyNodeFromSymbol(
    pGraphNode: *mut cudaGraphNode_t,
    graph: cudaGraph_t,
    pDependencies: *const cudaGraphNode_t,
    numDependencies: usize,
    dst: *mut ::core::ffi::c_void,
    symbol: *const ::core::ffi::c_void,
    count: usize,
    offset: usize,
    kind: cudaMemcpyKind,
) -> cudaError_t {
    let sym: Symbol<cudaGraphAddMemcpyNodeFromSymbol_fn> = unsafe {
        get_lib()
            .get(b"cudaGraphAddMemcpyNodeFromSymbol\0")
            .expect("Missing symbol")
    };
    unsafe {
        (*sym)(
            pGraphNode,
            graph,
            pDependencies,
            numDependencies,
            dst,
            symbol,
            count,
            offset,
            kind,
        )
    }
}

type cudaGraphAddMemcpyNode1D_fn = unsafe extern "C" fn(
    pGraphNode: *mut cudaGraphNode_t,
    graph: cudaGraph_t,
    pDependencies: *const cudaGraphNode_t,
    numDependencies: usize,
    dst: *mut ::core::ffi::c_void,
    src: *const ::core::ffi::c_void,
    count: usize,
    kind: cudaMemcpyKind,
) -> cudaError_t;
pub unsafe fn cudaGraphAddMemcpyNode1D(
    pGraphNode: *mut cudaGraphNode_t,
    graph: cudaGraph_t,
    pDependencies: *const cudaGraphNode_t,
    numDependencies: usize,
    dst: *mut ::core::ffi::c_void,
    src: *const ::core::ffi::c_void,
    count: usize,
    kind: cudaMemcpyKind,
) -> cudaError_t {
    let sym: Symbol<cudaGraphAddMemcpyNode1D_fn> = unsafe {
        get_lib()
            .get(b"cudaGraphAddMemcpyNode1D\0")
            .expect("Missing symbol")
    };
    unsafe {
        (*sym)(
            pGraphNode,
            graph,
            pDependencies,
            numDependencies,
            dst,
            src,
            count,
            kind,
        )
    }
}

type cudaGraphMemcpyNodeGetParams_fn =
    unsafe extern "C" fn(node: cudaGraphNode_t, pNodeParams: *mut cudaMemcpy3DParms) -> cudaError_t;
pub unsafe fn cudaGraphMemcpyNodeGetParams(
    node: cudaGraphNode_t,
    pNodeParams: *mut cudaMemcpy3DParms,
) -> cudaError_t {
    let sym: Symbol<cudaGraphMemcpyNodeGetParams_fn> = unsafe {
        get_lib()
            .get(b"cudaGraphMemcpyNodeGetParams\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(node, pNodeParams) }
}

type cudaGraphMemcpyNodeSetParams_fn = unsafe extern "C" fn(
    node: cudaGraphNode_t,
    pNodeParams: *const cudaMemcpy3DParms,
) -> cudaError_t;
pub unsafe fn cudaGraphMemcpyNodeSetParams(
    node: cudaGraphNode_t,
    pNodeParams: *const cudaMemcpy3DParms,
) -> cudaError_t {
    let sym: Symbol<cudaGraphMemcpyNodeSetParams_fn> = unsafe {
        get_lib()
            .get(b"cudaGraphMemcpyNodeSetParams\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(node, pNodeParams) }
}

type cudaGraphMemcpyNodeSetParamsToSymbol_fn = unsafe extern "C" fn(
    node: cudaGraphNode_t,
    symbol: *const ::core::ffi::c_void,
    src: *const ::core::ffi::c_void,
    count: usize,
    offset: usize,
    kind: cudaMemcpyKind,
) -> cudaError_t;
pub unsafe fn cudaGraphMemcpyNodeSetParamsToSymbol(
    node: cudaGraphNode_t,
    symbol: *const ::core::ffi::c_void,
    src: *const ::core::ffi::c_void,
    count: usize,
    offset: usize,
    kind: cudaMemcpyKind,
) -> cudaError_t {
    let sym: Symbol<cudaGraphMemcpyNodeSetParamsToSymbol_fn> = unsafe {
        get_lib()
            .get(b"cudaGraphMemcpyNodeSetParamsToSymbol\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(node, symbol, src, count, offset, kind) }
}

type cudaGraphMemcpyNodeSetParamsFromSymbol_fn = unsafe extern "C" fn(
    node: cudaGraphNode_t,
    dst: *mut ::core::ffi::c_void,
    symbol: *const ::core::ffi::c_void,
    count: usize,
    offset: usize,
    kind: cudaMemcpyKind,
) -> cudaError_t;
pub unsafe fn cudaGraphMemcpyNodeSetParamsFromSymbol(
    node: cudaGraphNode_t,
    dst: *mut ::core::ffi::c_void,
    symbol: *const ::core::ffi::c_void,
    count: usize,
    offset: usize,
    kind: cudaMemcpyKind,
) -> cudaError_t {
    let sym: Symbol<cudaGraphMemcpyNodeSetParamsFromSymbol_fn> = unsafe {
        get_lib()
            .get(b"cudaGraphMemcpyNodeSetParamsFromSymbol\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(node, dst, symbol, count, offset, kind) }
}

type cudaGraphMemcpyNodeSetParams1D_fn = unsafe extern "C" fn(
    node: cudaGraphNode_t,
    dst: *mut ::core::ffi::c_void,
    src: *const ::core::ffi::c_void,
    count: usize,
    kind: cudaMemcpyKind,
) -> cudaError_t;
pub unsafe fn cudaGraphMemcpyNodeSetParams1D(
    node: cudaGraphNode_t,
    dst: *mut ::core::ffi::c_void,
    src: *const ::core::ffi::c_void,
    count: usize,
    kind: cudaMemcpyKind,
) -> cudaError_t {
    let sym: Symbol<cudaGraphMemcpyNodeSetParams1D_fn> = unsafe {
        get_lib()
            .get(b"cudaGraphMemcpyNodeSetParams1D\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(node, dst, src, count, kind) }
}

type cudaGraphAddMemsetNode_fn = unsafe extern "C" fn(
    pGraphNode: *mut cudaGraphNode_t,
    graph: cudaGraph_t,
    pDependencies: *const cudaGraphNode_t,
    numDependencies: usize,
    pMemsetParams: *const cudaMemsetParams,
) -> cudaError_t;
pub unsafe fn cudaGraphAddMemsetNode(
    pGraphNode: *mut cudaGraphNode_t,
    graph: cudaGraph_t,
    pDependencies: *const cudaGraphNode_t,
    numDependencies: usize,
    pMemsetParams: *const cudaMemsetParams,
) -> cudaError_t {
    let sym: Symbol<cudaGraphAddMemsetNode_fn> = unsafe {
        get_lib()
            .get(b"cudaGraphAddMemsetNode\0")
            .expect("Missing symbol")
    };
    unsafe {
        (*sym)(
            pGraphNode,
            graph,
            pDependencies,
            numDependencies,
            pMemsetParams,
        )
    }
}

type cudaGraphMemsetNodeGetParams_fn =
    unsafe extern "C" fn(node: cudaGraphNode_t, pNodeParams: *mut cudaMemsetParams) -> cudaError_t;
pub unsafe fn cudaGraphMemsetNodeGetParams(
    node: cudaGraphNode_t,
    pNodeParams: *mut cudaMemsetParams,
) -> cudaError_t {
    let sym: Symbol<cudaGraphMemsetNodeGetParams_fn> = unsafe {
        get_lib()
            .get(b"cudaGraphMemsetNodeGetParams\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(node, pNodeParams) }
}

type cudaGraphMemsetNodeSetParams_fn = unsafe extern "C" fn(
    node: cudaGraphNode_t,
    pNodeParams: *const cudaMemsetParams,
) -> cudaError_t;
pub unsafe fn cudaGraphMemsetNodeSetParams(
    node: cudaGraphNode_t,
    pNodeParams: *const cudaMemsetParams,
) -> cudaError_t {
    let sym: Symbol<cudaGraphMemsetNodeSetParams_fn> = unsafe {
        get_lib()
            .get(b"cudaGraphMemsetNodeSetParams\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(node, pNodeParams) }
}

type cudaGraphAddHostNode_fn = unsafe extern "C" fn(
    pGraphNode: *mut cudaGraphNode_t,
    graph: cudaGraph_t,
    pDependencies: *const cudaGraphNode_t,
    numDependencies: usize,
    pNodeParams: *const cudaHostNodeParams,
) -> cudaError_t;
pub unsafe fn cudaGraphAddHostNode(
    pGraphNode: *mut cudaGraphNode_t,
    graph: cudaGraph_t,
    pDependencies: *const cudaGraphNode_t,
    numDependencies: usize,
    pNodeParams: *const cudaHostNodeParams,
) -> cudaError_t {
    let sym: Symbol<cudaGraphAddHostNode_fn> = unsafe {
        get_lib()
            .get(b"cudaGraphAddHostNode\0")
            .expect("Missing symbol")
    };
    unsafe {
        (*sym)(
            pGraphNode,
            graph,
            pDependencies,
            numDependencies,
            pNodeParams,
        )
    }
}

type cudaGraphHostNodeGetParams_fn = unsafe extern "C" fn(
    node: cudaGraphNode_t,
    pNodeParams: *mut cudaHostNodeParams,
) -> cudaError_t;
pub unsafe fn cudaGraphHostNodeGetParams(
    node: cudaGraphNode_t,
    pNodeParams: *mut cudaHostNodeParams,
) -> cudaError_t {
    let sym: Symbol<cudaGraphHostNodeGetParams_fn> = unsafe {
        get_lib()
            .get(b"cudaGraphHostNodeGetParams\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(node, pNodeParams) }
}

type cudaGraphHostNodeSetParams_fn = unsafe extern "C" fn(
    node: cudaGraphNode_t,
    pNodeParams: *const cudaHostNodeParams,
) -> cudaError_t;
pub unsafe fn cudaGraphHostNodeSetParams(
    node: cudaGraphNode_t,
    pNodeParams: *const cudaHostNodeParams,
) -> cudaError_t {
    let sym: Symbol<cudaGraphHostNodeSetParams_fn> = unsafe {
        get_lib()
            .get(b"cudaGraphHostNodeSetParams\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(node, pNodeParams) }
}

type cudaGraphAddChildGraphNode_fn = unsafe extern "C" fn(
    pGraphNode: *mut cudaGraphNode_t,
    graph: cudaGraph_t,
    pDependencies: *const cudaGraphNode_t,
    numDependencies: usize,
    childGraph: cudaGraph_t,
) -> cudaError_t;
pub unsafe fn cudaGraphAddChildGraphNode(
    pGraphNode: *mut cudaGraphNode_t,
    graph: cudaGraph_t,
    pDependencies: *const cudaGraphNode_t,
    numDependencies: usize,
    childGraph: cudaGraph_t,
) -> cudaError_t {
    let sym: Symbol<cudaGraphAddChildGraphNode_fn> = unsafe {
        get_lib()
            .get(b"cudaGraphAddChildGraphNode\0")
            .expect("Missing symbol")
    };
    unsafe {
        (*sym)(
            pGraphNode,
            graph,
            pDependencies,
            numDependencies,
            childGraph,
        )
    }
}

type cudaGraphChildGraphNodeGetGraph_fn =
    unsafe extern "C" fn(node: cudaGraphNode_t, pGraph: *mut cudaGraph_t) -> cudaError_t;
pub unsafe fn cudaGraphChildGraphNodeGetGraph(
    node: cudaGraphNode_t,
    pGraph: *mut cudaGraph_t,
) -> cudaError_t {
    let sym: Symbol<cudaGraphChildGraphNodeGetGraph_fn> = unsafe {
        get_lib()
            .get(b"cudaGraphChildGraphNodeGetGraph\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(node, pGraph) }
}

type cudaGraphAddEmptyNode_fn = unsafe extern "C" fn(
    pGraphNode: *mut cudaGraphNode_t,
    graph: cudaGraph_t,
    pDependencies: *const cudaGraphNode_t,
    numDependencies: usize,
) -> cudaError_t;
pub unsafe fn cudaGraphAddEmptyNode(
    pGraphNode: *mut cudaGraphNode_t,
    graph: cudaGraph_t,
    pDependencies: *const cudaGraphNode_t,
    numDependencies: usize,
) -> cudaError_t {
    let sym: Symbol<cudaGraphAddEmptyNode_fn> = unsafe {
        get_lib()
            .get(b"cudaGraphAddEmptyNode\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(pGraphNode, graph, pDependencies, numDependencies) }
}

type cudaGraphAddEventRecordNode_fn = unsafe extern "C" fn(
    pGraphNode: *mut cudaGraphNode_t,
    graph: cudaGraph_t,
    pDependencies: *const cudaGraphNode_t,
    numDependencies: usize,
    event: cudaEvent_t,
) -> cudaError_t;
pub unsafe fn cudaGraphAddEventRecordNode(
    pGraphNode: *mut cudaGraphNode_t,
    graph: cudaGraph_t,
    pDependencies: *const cudaGraphNode_t,
    numDependencies: usize,
    event: cudaEvent_t,
) -> cudaError_t {
    let sym: Symbol<cudaGraphAddEventRecordNode_fn> = unsafe {
        get_lib()
            .get(b"cudaGraphAddEventRecordNode\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(pGraphNode, graph, pDependencies, numDependencies, event) }
}

type cudaGraphEventRecordNodeGetEvent_fn =
    unsafe extern "C" fn(node: cudaGraphNode_t, event_out: *mut cudaEvent_t) -> cudaError_t;
pub unsafe fn cudaGraphEventRecordNodeGetEvent(
    node: cudaGraphNode_t,
    event_out: *mut cudaEvent_t,
) -> cudaError_t {
    let sym: Symbol<cudaGraphEventRecordNodeGetEvent_fn> = unsafe {
        get_lib()
            .get(b"cudaGraphEventRecordNodeGetEvent\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(node, event_out) }
}

type cudaGraphEventRecordNodeSetEvent_fn =
    unsafe extern "C" fn(node: cudaGraphNode_t, event: cudaEvent_t) -> cudaError_t;
pub unsafe fn cudaGraphEventRecordNodeSetEvent(
    node: cudaGraphNode_t,
    event: cudaEvent_t,
) -> cudaError_t {
    let sym: Symbol<cudaGraphEventRecordNodeSetEvent_fn> = unsafe {
        get_lib()
            .get(b"cudaGraphEventRecordNodeSetEvent\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(node, event) }
}

type cudaGraphAddEventWaitNode_fn = unsafe extern "C" fn(
    pGraphNode: *mut cudaGraphNode_t,
    graph: cudaGraph_t,
    pDependencies: *const cudaGraphNode_t,
    numDependencies: usize,
    event: cudaEvent_t,
) -> cudaError_t;
pub unsafe fn cudaGraphAddEventWaitNode(
    pGraphNode: *mut cudaGraphNode_t,
    graph: cudaGraph_t,
    pDependencies: *const cudaGraphNode_t,
    numDependencies: usize,
    event: cudaEvent_t,
) -> cudaError_t {
    let sym: Symbol<cudaGraphAddEventWaitNode_fn> = unsafe {
        get_lib()
            .get(b"cudaGraphAddEventWaitNode\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(pGraphNode, graph, pDependencies, numDependencies, event) }
}

type cudaGraphEventWaitNodeGetEvent_fn =
    unsafe extern "C" fn(node: cudaGraphNode_t, event_out: *mut cudaEvent_t) -> cudaError_t;
pub unsafe fn cudaGraphEventWaitNodeGetEvent(
    node: cudaGraphNode_t,
    event_out: *mut cudaEvent_t,
) -> cudaError_t {
    let sym: Symbol<cudaGraphEventWaitNodeGetEvent_fn> = unsafe {
        get_lib()
            .get(b"cudaGraphEventWaitNodeGetEvent\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(node, event_out) }
}

type cudaGraphEventWaitNodeSetEvent_fn =
    unsafe extern "C" fn(node: cudaGraphNode_t, event: cudaEvent_t) -> cudaError_t;
pub unsafe fn cudaGraphEventWaitNodeSetEvent(
    node: cudaGraphNode_t,
    event: cudaEvent_t,
) -> cudaError_t {
    let sym: Symbol<cudaGraphEventWaitNodeSetEvent_fn> = unsafe {
        get_lib()
            .get(b"cudaGraphEventWaitNodeSetEvent\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(node, event) }
}

type cudaGraphAddExternalSemaphoresSignalNode_fn = unsafe extern "C" fn(
    pGraphNode: *mut cudaGraphNode_t,
    graph: cudaGraph_t,
    pDependencies: *const cudaGraphNode_t,
    numDependencies: usize,
    nodeParams: *const cudaExternalSemaphoreSignalNodeParams,
) -> cudaError_t;
pub unsafe fn cudaGraphAddExternalSemaphoresSignalNode(
    pGraphNode: *mut cudaGraphNode_t,
    graph: cudaGraph_t,
    pDependencies: *const cudaGraphNode_t,
    numDependencies: usize,
    nodeParams: *const cudaExternalSemaphoreSignalNodeParams,
) -> cudaError_t {
    let sym: Symbol<cudaGraphAddExternalSemaphoresSignalNode_fn> = unsafe {
        get_lib()
            .get(b"cudaGraphAddExternalSemaphoresSignalNode\0")
            .expect("Missing symbol")
    };
    unsafe {
        (*sym)(
            pGraphNode,
            graph,
            pDependencies,
            numDependencies,
            nodeParams,
        )
    }
}

type cudaGraphExternalSemaphoresSignalNodeGetParams_fn = unsafe extern "C" fn(
    hNode: cudaGraphNode_t,
    params_out: *mut cudaExternalSemaphoreSignalNodeParams,
) -> cudaError_t;
pub unsafe fn cudaGraphExternalSemaphoresSignalNodeGetParams(
    hNode: cudaGraphNode_t,
    params_out: *mut cudaExternalSemaphoreSignalNodeParams,
) -> cudaError_t {
    let sym: Symbol<cudaGraphExternalSemaphoresSignalNodeGetParams_fn> = unsafe {
        get_lib()
            .get(b"cudaGraphExternalSemaphoresSignalNodeGetParams\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(hNode, params_out) }
}

type cudaGraphExternalSemaphoresSignalNodeSetParams_fn = unsafe extern "C" fn(
    hNode: cudaGraphNode_t,
    nodeParams: *const cudaExternalSemaphoreSignalNodeParams,
) -> cudaError_t;
pub unsafe fn cudaGraphExternalSemaphoresSignalNodeSetParams(
    hNode: cudaGraphNode_t,
    nodeParams: *const cudaExternalSemaphoreSignalNodeParams,
) -> cudaError_t {
    let sym: Symbol<cudaGraphExternalSemaphoresSignalNodeSetParams_fn> = unsafe {
        get_lib()
            .get(b"cudaGraphExternalSemaphoresSignalNodeSetParams\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(hNode, nodeParams) }
}

type cudaGraphAddExternalSemaphoresWaitNode_fn = unsafe extern "C" fn(
    pGraphNode: *mut cudaGraphNode_t,
    graph: cudaGraph_t,
    pDependencies: *const cudaGraphNode_t,
    numDependencies: usize,
    nodeParams: *const cudaExternalSemaphoreWaitNodeParams,
) -> cudaError_t;
pub unsafe fn cudaGraphAddExternalSemaphoresWaitNode(
    pGraphNode: *mut cudaGraphNode_t,
    graph: cudaGraph_t,
    pDependencies: *const cudaGraphNode_t,
    numDependencies: usize,
    nodeParams: *const cudaExternalSemaphoreWaitNodeParams,
) -> cudaError_t {
    let sym: Symbol<cudaGraphAddExternalSemaphoresWaitNode_fn> = unsafe {
        get_lib()
            .get(b"cudaGraphAddExternalSemaphoresWaitNode\0")
            .expect("Missing symbol")
    };
    unsafe {
        (*sym)(
            pGraphNode,
            graph,
            pDependencies,
            numDependencies,
            nodeParams,
        )
    }
}

type cudaGraphExternalSemaphoresWaitNodeGetParams_fn = unsafe extern "C" fn(
    hNode: cudaGraphNode_t,
    params_out: *mut cudaExternalSemaphoreWaitNodeParams,
) -> cudaError_t;
pub unsafe fn cudaGraphExternalSemaphoresWaitNodeGetParams(
    hNode: cudaGraphNode_t,
    params_out: *mut cudaExternalSemaphoreWaitNodeParams,
) -> cudaError_t {
    let sym: Symbol<cudaGraphExternalSemaphoresWaitNodeGetParams_fn> = unsafe {
        get_lib()
            .get(b"cudaGraphExternalSemaphoresWaitNodeGetParams\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(hNode, params_out) }
}

type cudaGraphExternalSemaphoresWaitNodeSetParams_fn = unsafe extern "C" fn(
    hNode: cudaGraphNode_t,
    nodeParams: *const cudaExternalSemaphoreWaitNodeParams,
) -> cudaError_t;
pub unsafe fn cudaGraphExternalSemaphoresWaitNodeSetParams(
    hNode: cudaGraphNode_t,
    nodeParams: *const cudaExternalSemaphoreWaitNodeParams,
) -> cudaError_t {
    let sym: Symbol<cudaGraphExternalSemaphoresWaitNodeSetParams_fn> = unsafe {
        get_lib()
            .get(b"cudaGraphExternalSemaphoresWaitNodeSetParams\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(hNode, nodeParams) }
}

type cudaGraphAddMemAllocNode_fn = unsafe extern "C" fn(
    pGraphNode: *mut cudaGraphNode_t,
    graph: cudaGraph_t,
    pDependencies: *const cudaGraphNode_t,
    numDependencies: usize,
    nodeParams: *mut cudaMemAllocNodeParams,
) -> cudaError_t;
pub unsafe fn cudaGraphAddMemAllocNode(
    pGraphNode: *mut cudaGraphNode_t,
    graph: cudaGraph_t,
    pDependencies: *const cudaGraphNode_t,
    numDependencies: usize,
    nodeParams: *mut cudaMemAllocNodeParams,
) -> cudaError_t {
    let sym: Symbol<cudaGraphAddMemAllocNode_fn> = unsafe {
        get_lib()
            .get(b"cudaGraphAddMemAllocNode\0")
            .expect("Missing symbol")
    };
    unsafe {
        (*sym)(
            pGraphNode,
            graph,
            pDependencies,
            numDependencies,
            nodeParams,
        )
    }
}

type cudaGraphMemAllocNodeGetParams_fn = unsafe extern "C" fn(
    node: cudaGraphNode_t,
    params_out: *mut cudaMemAllocNodeParams,
) -> cudaError_t;
pub unsafe fn cudaGraphMemAllocNodeGetParams(
    node: cudaGraphNode_t,
    params_out: *mut cudaMemAllocNodeParams,
) -> cudaError_t {
    let sym: Symbol<cudaGraphMemAllocNodeGetParams_fn> = unsafe {
        get_lib()
            .get(b"cudaGraphMemAllocNodeGetParams\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(node, params_out) }
}

type cudaGraphAddMemFreeNode_fn = unsafe extern "C" fn(
    pGraphNode: *mut cudaGraphNode_t,
    graph: cudaGraph_t,
    pDependencies: *const cudaGraphNode_t,
    numDependencies: usize,
    dptr: *mut ::core::ffi::c_void,
) -> cudaError_t;
pub unsafe fn cudaGraphAddMemFreeNode(
    pGraphNode: *mut cudaGraphNode_t,
    graph: cudaGraph_t,
    pDependencies: *const cudaGraphNode_t,
    numDependencies: usize,
    dptr: *mut ::core::ffi::c_void,
) -> cudaError_t {
    let sym: Symbol<cudaGraphAddMemFreeNode_fn> = unsafe {
        get_lib()
            .get(b"cudaGraphAddMemFreeNode\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(pGraphNode, graph, pDependencies, numDependencies, dptr) }
}

type cudaGraphMemFreeNodeGetParams_fn =
    unsafe extern "C" fn(node: cudaGraphNode_t, dptr_out: *mut ::core::ffi::c_void) -> cudaError_t;
pub unsafe fn cudaGraphMemFreeNodeGetParams(
    node: cudaGraphNode_t,
    dptr_out: *mut ::core::ffi::c_void,
) -> cudaError_t {
    let sym: Symbol<cudaGraphMemFreeNodeGetParams_fn> = unsafe {
        get_lib()
            .get(b"cudaGraphMemFreeNodeGetParams\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(node, dptr_out) }
}

type cudaDeviceGraphMemTrim_fn = unsafe extern "C" fn(device: ::core::ffi::c_int) -> cudaError_t;
pub unsafe fn cudaDeviceGraphMemTrim(device: ::core::ffi::c_int) -> cudaError_t {
    let sym: Symbol<cudaDeviceGraphMemTrim_fn> = unsafe {
        get_lib()
            .get(b"cudaDeviceGraphMemTrim\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(device) }
}

type cudaDeviceGetGraphMemAttribute_fn = unsafe extern "C" fn(
    device: ::core::ffi::c_int,
    attr: cudaGraphMemAttributeType,
    value: *mut ::core::ffi::c_void,
) -> cudaError_t;
pub unsafe fn cudaDeviceGetGraphMemAttribute(
    device: ::core::ffi::c_int,
    attr: cudaGraphMemAttributeType,
    value: *mut ::core::ffi::c_void,
) -> cudaError_t {
    let sym: Symbol<cudaDeviceGetGraphMemAttribute_fn> = unsafe {
        get_lib()
            .get(b"cudaDeviceGetGraphMemAttribute\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(device, attr, value) }
}

type cudaDeviceSetGraphMemAttribute_fn = unsafe extern "C" fn(
    device: ::core::ffi::c_int,
    attr: cudaGraphMemAttributeType,
    value: *mut ::core::ffi::c_void,
) -> cudaError_t;
pub unsafe fn cudaDeviceSetGraphMemAttribute(
    device: ::core::ffi::c_int,
    attr: cudaGraphMemAttributeType,
    value: *mut ::core::ffi::c_void,
) -> cudaError_t {
    let sym: Symbol<cudaDeviceSetGraphMemAttribute_fn> = unsafe {
        get_lib()
            .get(b"cudaDeviceSetGraphMemAttribute\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(device, attr, value) }
}

type cudaGraphClone_fn =
    unsafe extern "C" fn(pGraphClone: *mut cudaGraph_t, originalGraph: cudaGraph_t) -> cudaError_t;
pub unsafe fn cudaGraphClone(
    pGraphClone: *mut cudaGraph_t,
    originalGraph: cudaGraph_t,
) -> cudaError_t {
    let sym: Symbol<cudaGraphClone_fn> =
        unsafe { get_lib().get(b"cudaGraphClone\0").expect("Missing symbol") };
    unsafe { (*sym)(pGraphClone, originalGraph) }
}

type cudaGraphNodeFindInClone_fn = unsafe extern "C" fn(
    pNode: *mut cudaGraphNode_t,
    originalNode: cudaGraphNode_t,
    clonedGraph: cudaGraph_t,
) -> cudaError_t;
pub unsafe fn cudaGraphNodeFindInClone(
    pNode: *mut cudaGraphNode_t,
    originalNode: cudaGraphNode_t,
    clonedGraph: cudaGraph_t,
) -> cudaError_t {
    let sym: Symbol<cudaGraphNodeFindInClone_fn> = unsafe {
        get_lib()
            .get(b"cudaGraphNodeFindInClone\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(pNode, originalNode, clonedGraph) }
}

type cudaGraphNodeGetType_fn =
    unsafe extern "C" fn(node: cudaGraphNode_t, pType: *mut cudaGraphNodeType) -> cudaError_t;
pub unsafe fn cudaGraphNodeGetType(
    node: cudaGraphNode_t,
    pType: *mut cudaGraphNodeType,
) -> cudaError_t {
    let sym: Symbol<cudaGraphNodeGetType_fn> = unsafe {
        get_lib()
            .get(b"cudaGraphNodeGetType\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(node, pType) }
}

type cudaGraphGetNodes_fn = unsafe extern "C" fn(
    graph: cudaGraph_t,
    nodes: *mut cudaGraphNode_t,
    numNodes: *mut usize,
) -> cudaError_t;
pub unsafe fn cudaGraphGetNodes(
    graph: cudaGraph_t,
    nodes: *mut cudaGraphNode_t,
    numNodes: *mut usize,
) -> cudaError_t {
    let sym: Symbol<cudaGraphGetNodes_fn> = unsafe {
        get_lib()
            .get(b"cudaGraphGetNodes\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(graph, nodes, numNodes) }
}

type cudaGraphGetRootNodes_fn = unsafe extern "C" fn(
    graph: cudaGraph_t,
    pRootNodes: *mut cudaGraphNode_t,
    pNumRootNodes: *mut usize,
) -> cudaError_t;
pub unsafe fn cudaGraphGetRootNodes(
    graph: cudaGraph_t,
    pRootNodes: *mut cudaGraphNode_t,
    pNumRootNodes: *mut usize,
) -> cudaError_t {
    let sym: Symbol<cudaGraphGetRootNodes_fn> = unsafe {
        get_lib()
            .get(b"cudaGraphGetRootNodes\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(graph, pRootNodes, pNumRootNodes) }
}

type cudaGraphGetEdges_fn = unsafe extern "C" fn(
    graph: cudaGraph_t,
    from: *mut cudaGraphNode_t,
    to: *mut cudaGraphNode_t,
    edgeData: *mut cudaGraphEdgeData,
    numEdges: *mut usize,
) -> cudaError_t;
pub unsafe fn cudaGraphGetEdges(
    graph: cudaGraph_t,
    from: *mut cudaGraphNode_t,
    to: *mut cudaGraphNode_t,
    edgeData: *mut cudaGraphEdgeData,
    numEdges: *mut usize,
) -> cudaError_t {
    let sym: Symbol<cudaGraphGetEdges_fn> = unsafe {
        get_lib()
            .get(b"cudaGraphGetEdges\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(graph, from, to, edgeData, numEdges) }
}

type cudaGraphNodeGetDependencies_fn = unsafe extern "C" fn(
    node: cudaGraphNode_t,
    pDependencies: *mut cudaGraphNode_t,
    edgeData: *mut cudaGraphEdgeData,
    pNumDependencies: *mut usize,
) -> cudaError_t;
pub unsafe fn cudaGraphNodeGetDependencies(
    node: cudaGraphNode_t,
    pDependencies: *mut cudaGraphNode_t,
    edgeData: *mut cudaGraphEdgeData,
    pNumDependencies: *mut usize,
) -> cudaError_t {
    let sym: Symbol<cudaGraphNodeGetDependencies_fn> = unsafe {
        get_lib()
            .get(b"cudaGraphNodeGetDependencies\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(node, pDependencies, edgeData, pNumDependencies) }
}

type cudaGraphNodeGetDependentNodes_fn = unsafe extern "C" fn(
    node: cudaGraphNode_t,
    pDependentNodes: *mut cudaGraphNode_t,
    edgeData: *mut cudaGraphEdgeData,
    pNumDependentNodes: *mut usize,
) -> cudaError_t;
pub unsafe fn cudaGraphNodeGetDependentNodes(
    node: cudaGraphNode_t,
    pDependentNodes: *mut cudaGraphNode_t,
    edgeData: *mut cudaGraphEdgeData,
    pNumDependentNodes: *mut usize,
) -> cudaError_t {
    let sym: Symbol<cudaGraphNodeGetDependentNodes_fn> = unsafe {
        get_lib()
            .get(b"cudaGraphNodeGetDependentNodes\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(node, pDependentNodes, edgeData, pNumDependentNodes) }
}

type cudaGraphAddDependencies_fn = unsafe extern "C" fn(
    graph: cudaGraph_t,
    from: *const cudaGraphNode_t,
    to: *const cudaGraphNode_t,
    edgeData: *const cudaGraphEdgeData,
    numDependencies: usize,
) -> cudaError_t;
pub unsafe fn cudaGraphAddDependencies(
    graph: cudaGraph_t,
    from: *const cudaGraphNode_t,
    to: *const cudaGraphNode_t,
    edgeData: *const cudaGraphEdgeData,
    numDependencies: usize,
) -> cudaError_t {
    let sym: Symbol<cudaGraphAddDependencies_fn> = unsafe {
        get_lib()
            .get(b"cudaGraphAddDependencies\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(graph, from, to, edgeData, numDependencies) }
}

type cudaGraphRemoveDependencies_fn = unsafe extern "C" fn(
    graph: cudaGraph_t,
    from: *const cudaGraphNode_t,
    to: *const cudaGraphNode_t,
    edgeData: *const cudaGraphEdgeData,
    numDependencies: usize,
) -> cudaError_t;
pub unsafe fn cudaGraphRemoveDependencies(
    graph: cudaGraph_t,
    from: *const cudaGraphNode_t,
    to: *const cudaGraphNode_t,
    edgeData: *const cudaGraphEdgeData,
    numDependencies: usize,
) -> cudaError_t {
    let sym: Symbol<cudaGraphRemoveDependencies_fn> = unsafe {
        get_lib()
            .get(b"cudaGraphRemoveDependencies\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(graph, from, to, edgeData, numDependencies) }
}

type cudaGraphDestroyNode_fn = unsafe extern "C" fn(node: cudaGraphNode_t) -> cudaError_t;
pub unsafe fn cudaGraphDestroyNode(node: cudaGraphNode_t) -> cudaError_t {
    let sym: Symbol<cudaGraphDestroyNode_fn> = unsafe {
        get_lib()
            .get(b"cudaGraphDestroyNode\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(node) }
}

type cudaGraphInstantiate_fn = unsafe extern "C" fn(
    pGraphExec: *mut cudaGraphExec_t,
    graph: cudaGraph_t,
    flags: ::core::ffi::c_ulonglong,
) -> cudaError_t;
pub unsafe fn cudaGraphInstantiate(
    pGraphExec: *mut cudaGraphExec_t,
    graph: cudaGraph_t,
    flags: ::core::ffi::c_ulonglong,
) -> cudaError_t {
    let sym: Symbol<cudaGraphInstantiate_fn> = unsafe {
        get_lib()
            .get(b"cudaGraphInstantiate\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(pGraphExec, graph, flags) }
}

type cudaGraphInstantiateWithFlags_fn = unsafe extern "C" fn(
    pGraphExec: *mut cudaGraphExec_t,
    graph: cudaGraph_t,
    flags: ::core::ffi::c_ulonglong,
) -> cudaError_t;
pub unsafe fn cudaGraphInstantiateWithFlags(
    pGraphExec: *mut cudaGraphExec_t,
    graph: cudaGraph_t,
    flags: ::core::ffi::c_ulonglong,
) -> cudaError_t {
    let sym: Symbol<cudaGraphInstantiateWithFlags_fn> = unsafe {
        get_lib()
            .get(b"cudaGraphInstantiateWithFlags\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(pGraphExec, graph, flags) }
}

type cudaGraphInstantiateWithParams_fn = unsafe extern "C" fn(
    pGraphExec: *mut cudaGraphExec_t,
    graph: cudaGraph_t,
    instantiateParams: *mut cudaGraphInstantiateParams,
) -> cudaError_t;
pub unsafe fn cudaGraphInstantiateWithParams(
    pGraphExec: *mut cudaGraphExec_t,
    graph: cudaGraph_t,
    instantiateParams: *mut cudaGraphInstantiateParams,
) -> cudaError_t {
    let sym: Symbol<cudaGraphInstantiateWithParams_fn> = unsafe {
        get_lib()
            .get(b"cudaGraphInstantiateWithParams\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(pGraphExec, graph, instantiateParams) }
}

type cudaGraphExecGetFlags_fn = unsafe extern "C" fn(
    graphExec: cudaGraphExec_t,
    flags: *mut ::core::ffi::c_ulonglong,
) -> cudaError_t;
pub unsafe fn cudaGraphExecGetFlags(
    graphExec: cudaGraphExec_t,
    flags: *mut ::core::ffi::c_ulonglong,
) -> cudaError_t {
    let sym: Symbol<cudaGraphExecGetFlags_fn> = unsafe {
        get_lib()
            .get(b"cudaGraphExecGetFlags\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(graphExec, flags) }
}

type cudaGraphExecKernelNodeSetParams_fn = unsafe extern "C" fn(
    hGraphExec: cudaGraphExec_t,
    node: cudaGraphNode_t,
    pNodeParams: *const cudaKernelNodeParams,
) -> cudaError_t;
pub unsafe fn cudaGraphExecKernelNodeSetParams(
    hGraphExec: cudaGraphExec_t,
    node: cudaGraphNode_t,
    pNodeParams: *const cudaKernelNodeParams,
) -> cudaError_t {
    let sym: Symbol<cudaGraphExecKernelNodeSetParams_fn> = unsafe {
        get_lib()
            .get(b"cudaGraphExecKernelNodeSetParams\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(hGraphExec, node, pNodeParams) }
}

type cudaGraphExecMemcpyNodeSetParams_fn = unsafe extern "C" fn(
    hGraphExec: cudaGraphExec_t,
    node: cudaGraphNode_t,
    pNodeParams: *const cudaMemcpy3DParms,
) -> cudaError_t;
pub unsafe fn cudaGraphExecMemcpyNodeSetParams(
    hGraphExec: cudaGraphExec_t,
    node: cudaGraphNode_t,
    pNodeParams: *const cudaMemcpy3DParms,
) -> cudaError_t {
    let sym: Symbol<cudaGraphExecMemcpyNodeSetParams_fn> = unsafe {
        get_lib()
            .get(b"cudaGraphExecMemcpyNodeSetParams\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(hGraphExec, node, pNodeParams) }
}

type cudaGraphExecMemcpyNodeSetParamsToSymbol_fn = unsafe extern "C" fn(
    hGraphExec: cudaGraphExec_t,
    node: cudaGraphNode_t,
    symbol: *const ::core::ffi::c_void,
    src: *const ::core::ffi::c_void,
    count: usize,
    offset: usize,
    kind: cudaMemcpyKind,
) -> cudaError_t;
pub unsafe fn cudaGraphExecMemcpyNodeSetParamsToSymbol(
    hGraphExec: cudaGraphExec_t,
    node: cudaGraphNode_t,
    symbol: *const ::core::ffi::c_void,
    src: *const ::core::ffi::c_void,
    count: usize,
    offset: usize,
    kind: cudaMemcpyKind,
) -> cudaError_t {
    let sym: Symbol<cudaGraphExecMemcpyNodeSetParamsToSymbol_fn> = unsafe {
        get_lib()
            .get(b"cudaGraphExecMemcpyNodeSetParamsToSymbol\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(hGraphExec, node, symbol, src, count, offset, kind) }
}

type cudaGraphExecMemcpyNodeSetParamsFromSymbol_fn = unsafe extern "C" fn(
    hGraphExec: cudaGraphExec_t,
    node: cudaGraphNode_t,
    dst: *mut ::core::ffi::c_void,
    symbol: *const ::core::ffi::c_void,
    count: usize,
    offset: usize,
    kind: cudaMemcpyKind,
) -> cudaError_t;
pub unsafe fn cudaGraphExecMemcpyNodeSetParamsFromSymbol(
    hGraphExec: cudaGraphExec_t,
    node: cudaGraphNode_t,
    dst: *mut ::core::ffi::c_void,
    symbol: *const ::core::ffi::c_void,
    count: usize,
    offset: usize,
    kind: cudaMemcpyKind,
) -> cudaError_t {
    let sym: Symbol<cudaGraphExecMemcpyNodeSetParamsFromSymbol_fn> = unsafe {
        get_lib()
            .get(b"cudaGraphExecMemcpyNodeSetParamsFromSymbol\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(hGraphExec, node, dst, symbol, count, offset, kind) }
}

type cudaGraphExecMemcpyNodeSetParams1D_fn = unsafe extern "C" fn(
    hGraphExec: cudaGraphExec_t,
    node: cudaGraphNode_t,
    dst: *mut ::core::ffi::c_void,
    src: *const ::core::ffi::c_void,
    count: usize,
    kind: cudaMemcpyKind,
) -> cudaError_t;
pub unsafe fn cudaGraphExecMemcpyNodeSetParams1D(
    hGraphExec: cudaGraphExec_t,
    node: cudaGraphNode_t,
    dst: *mut ::core::ffi::c_void,
    src: *const ::core::ffi::c_void,
    count: usize,
    kind: cudaMemcpyKind,
) -> cudaError_t {
    let sym: Symbol<cudaGraphExecMemcpyNodeSetParams1D_fn> = unsafe {
        get_lib()
            .get(b"cudaGraphExecMemcpyNodeSetParams1D\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(hGraphExec, node, dst, src, count, kind) }
}

type cudaGraphExecMemsetNodeSetParams_fn = unsafe extern "C" fn(
    hGraphExec: cudaGraphExec_t,
    node: cudaGraphNode_t,
    pNodeParams: *const cudaMemsetParams,
) -> cudaError_t;
pub unsafe fn cudaGraphExecMemsetNodeSetParams(
    hGraphExec: cudaGraphExec_t,
    node: cudaGraphNode_t,
    pNodeParams: *const cudaMemsetParams,
) -> cudaError_t {
    let sym: Symbol<cudaGraphExecMemsetNodeSetParams_fn> = unsafe {
        get_lib()
            .get(b"cudaGraphExecMemsetNodeSetParams\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(hGraphExec, node, pNodeParams) }
}

type cudaGraphExecHostNodeSetParams_fn = unsafe extern "C" fn(
    hGraphExec: cudaGraphExec_t,
    node: cudaGraphNode_t,
    pNodeParams: *const cudaHostNodeParams,
) -> cudaError_t;
pub unsafe fn cudaGraphExecHostNodeSetParams(
    hGraphExec: cudaGraphExec_t,
    node: cudaGraphNode_t,
    pNodeParams: *const cudaHostNodeParams,
) -> cudaError_t {
    let sym: Symbol<cudaGraphExecHostNodeSetParams_fn> = unsafe {
        get_lib()
            .get(b"cudaGraphExecHostNodeSetParams\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(hGraphExec, node, pNodeParams) }
}

type cudaGraphExecChildGraphNodeSetParams_fn = unsafe extern "C" fn(
    hGraphExec: cudaGraphExec_t,
    node: cudaGraphNode_t,
    childGraph: cudaGraph_t,
) -> cudaError_t;
pub unsafe fn cudaGraphExecChildGraphNodeSetParams(
    hGraphExec: cudaGraphExec_t,
    node: cudaGraphNode_t,
    childGraph: cudaGraph_t,
) -> cudaError_t {
    let sym: Symbol<cudaGraphExecChildGraphNodeSetParams_fn> = unsafe {
        get_lib()
            .get(b"cudaGraphExecChildGraphNodeSetParams\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(hGraphExec, node, childGraph) }
}

type cudaGraphExecEventRecordNodeSetEvent_fn = unsafe extern "C" fn(
    hGraphExec: cudaGraphExec_t,
    hNode: cudaGraphNode_t,
    event: cudaEvent_t,
) -> cudaError_t;
pub unsafe fn cudaGraphExecEventRecordNodeSetEvent(
    hGraphExec: cudaGraphExec_t,
    hNode: cudaGraphNode_t,
    event: cudaEvent_t,
) -> cudaError_t {
    let sym: Symbol<cudaGraphExecEventRecordNodeSetEvent_fn> = unsafe {
        get_lib()
            .get(b"cudaGraphExecEventRecordNodeSetEvent\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(hGraphExec, hNode, event) }
}

type cudaGraphExecEventWaitNodeSetEvent_fn = unsafe extern "C" fn(
    hGraphExec: cudaGraphExec_t,
    hNode: cudaGraphNode_t,
    event: cudaEvent_t,
) -> cudaError_t;
pub unsafe fn cudaGraphExecEventWaitNodeSetEvent(
    hGraphExec: cudaGraphExec_t,
    hNode: cudaGraphNode_t,
    event: cudaEvent_t,
) -> cudaError_t {
    let sym: Symbol<cudaGraphExecEventWaitNodeSetEvent_fn> = unsafe {
        get_lib()
            .get(b"cudaGraphExecEventWaitNodeSetEvent\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(hGraphExec, hNode, event) }
}

type cudaGraphExecExternalSemaphoresSignalNodeSetParams_fn = unsafe extern "C" fn(
    hGraphExec: cudaGraphExec_t,
    hNode: cudaGraphNode_t,
    nodeParams: *const cudaExternalSemaphoreSignalNodeParams,
) -> cudaError_t;
pub unsafe fn cudaGraphExecExternalSemaphoresSignalNodeSetParams(
    hGraphExec: cudaGraphExec_t,
    hNode: cudaGraphNode_t,
    nodeParams: *const cudaExternalSemaphoreSignalNodeParams,
) -> cudaError_t {
    let sym: Symbol<cudaGraphExecExternalSemaphoresSignalNodeSetParams_fn> = unsafe {
        get_lib()
            .get(b"cudaGraphExecExternalSemaphoresSignalNodeSetParams\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(hGraphExec, hNode, nodeParams) }
}

type cudaGraphExecExternalSemaphoresWaitNodeSetParams_fn = unsafe extern "C" fn(
    hGraphExec: cudaGraphExec_t,
    hNode: cudaGraphNode_t,
    nodeParams: *const cudaExternalSemaphoreWaitNodeParams,
) -> cudaError_t;
pub unsafe fn cudaGraphExecExternalSemaphoresWaitNodeSetParams(
    hGraphExec: cudaGraphExec_t,
    hNode: cudaGraphNode_t,
    nodeParams: *const cudaExternalSemaphoreWaitNodeParams,
) -> cudaError_t {
    let sym: Symbol<cudaGraphExecExternalSemaphoresWaitNodeSetParams_fn> = unsafe {
        get_lib()
            .get(b"cudaGraphExecExternalSemaphoresWaitNodeSetParams\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(hGraphExec, hNode, nodeParams) }
}

type cudaGraphNodeSetEnabled_fn = unsafe extern "C" fn(
    hGraphExec: cudaGraphExec_t,
    hNode: cudaGraphNode_t,
    isEnabled: ::core::ffi::c_uint,
) -> cudaError_t;
pub unsafe fn cudaGraphNodeSetEnabled(
    hGraphExec: cudaGraphExec_t,
    hNode: cudaGraphNode_t,
    isEnabled: ::core::ffi::c_uint,
) -> cudaError_t {
    let sym: Symbol<cudaGraphNodeSetEnabled_fn> = unsafe {
        get_lib()
            .get(b"cudaGraphNodeSetEnabled\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(hGraphExec, hNode, isEnabled) }
}

type cudaGraphNodeGetEnabled_fn = unsafe extern "C" fn(
    hGraphExec: cudaGraphExec_t,
    hNode: cudaGraphNode_t,
    isEnabled: *mut ::core::ffi::c_uint,
) -> cudaError_t;
pub unsafe fn cudaGraphNodeGetEnabled(
    hGraphExec: cudaGraphExec_t,
    hNode: cudaGraphNode_t,
    isEnabled: *mut ::core::ffi::c_uint,
) -> cudaError_t {
    let sym: Symbol<cudaGraphNodeGetEnabled_fn> = unsafe {
        get_lib()
            .get(b"cudaGraphNodeGetEnabled\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(hGraphExec, hNode, isEnabled) }
}

type cudaGraphExecUpdate_fn = unsafe extern "C" fn(
    hGraphExec: cudaGraphExec_t,
    hGraph: cudaGraph_t,
    resultInfo: *mut cudaGraphExecUpdateResultInfo,
) -> cudaError_t;
pub unsafe fn cudaGraphExecUpdate(
    hGraphExec: cudaGraphExec_t,
    hGraph: cudaGraph_t,
    resultInfo: *mut cudaGraphExecUpdateResultInfo,
) -> cudaError_t {
    let sym: Symbol<cudaGraphExecUpdate_fn> = unsafe {
        get_lib()
            .get(b"cudaGraphExecUpdate\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(hGraphExec, hGraph, resultInfo) }
}

type cudaGraphUpload_fn =
    unsafe extern "C" fn(graphExec: cudaGraphExec_t, stream: cudaStream_t) -> cudaError_t;
pub unsafe fn cudaGraphUpload(graphExec: cudaGraphExec_t, stream: cudaStream_t) -> cudaError_t {
    let sym: Symbol<cudaGraphUpload_fn> =
        unsafe { get_lib().get(b"cudaGraphUpload\0").expect("Missing symbol") };
    unsafe { (*sym)(graphExec, stream) }
}

type cudaGraphLaunch_fn =
    unsafe extern "C" fn(graphExec: cudaGraphExec_t, stream: cudaStream_t) -> cudaError_t;
pub unsafe fn cudaGraphLaunch(graphExec: cudaGraphExec_t, stream: cudaStream_t) -> cudaError_t {
    let sym: Symbol<cudaGraphLaunch_fn> =
        unsafe { get_lib().get(b"cudaGraphLaunch\0").expect("Missing symbol") };
    unsafe { (*sym)(graphExec, stream) }
}

type cudaGraphExecDestroy_fn = unsafe extern "C" fn(graphExec: cudaGraphExec_t) -> cudaError_t;
pub unsafe fn cudaGraphExecDestroy(graphExec: cudaGraphExec_t) -> cudaError_t {
    let sym: Symbol<cudaGraphExecDestroy_fn> = unsafe {
        get_lib()
            .get(b"cudaGraphExecDestroy\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(graphExec) }
}

type cudaGraphDestroy_fn = unsafe extern "C" fn(graph: cudaGraph_t) -> cudaError_t;
pub unsafe fn cudaGraphDestroy(graph: cudaGraph_t) -> cudaError_t {
    let sym: Symbol<cudaGraphDestroy_fn> = unsafe {
        get_lib()
            .get(b"cudaGraphDestroy\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(graph) }
}

type cudaGraphDebugDotPrint_fn = unsafe extern "C" fn(
    graph: cudaGraph_t,
    path: *const ::core::ffi::c_char,
    flags: ::core::ffi::c_uint,
) -> cudaError_t;
pub unsafe fn cudaGraphDebugDotPrint(
    graph: cudaGraph_t,
    path: *const ::core::ffi::c_char,
    flags: ::core::ffi::c_uint,
) -> cudaError_t {
    let sym: Symbol<cudaGraphDebugDotPrint_fn> = unsafe {
        get_lib()
            .get(b"cudaGraphDebugDotPrint\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(graph, path, flags) }
}

type cudaUserObjectCreate_fn = unsafe extern "C" fn(
    object_out: *mut cudaUserObject_t,
    ptr: *mut ::core::ffi::c_void,
    destroy: cudaHostFn_t,
    initialRefcount: ::core::ffi::c_uint,
    flags: ::core::ffi::c_uint,
) -> cudaError_t;
pub unsafe fn cudaUserObjectCreate(
    object_out: *mut cudaUserObject_t,
    ptr: *mut ::core::ffi::c_void,
    destroy: cudaHostFn_t,
    initialRefcount: ::core::ffi::c_uint,
    flags: ::core::ffi::c_uint,
) -> cudaError_t {
    let sym: Symbol<cudaUserObjectCreate_fn> = unsafe {
        get_lib()
            .get(b"cudaUserObjectCreate\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(object_out, ptr, destroy, initialRefcount, flags) }
}

type cudaUserObjectRetain_fn =
    unsafe extern "C" fn(object: cudaUserObject_t, count: ::core::ffi::c_uint) -> cudaError_t;
pub unsafe fn cudaUserObjectRetain(
    object: cudaUserObject_t,
    count: ::core::ffi::c_uint,
) -> cudaError_t {
    let sym: Symbol<cudaUserObjectRetain_fn> = unsafe {
        get_lib()
            .get(b"cudaUserObjectRetain\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(object, count) }
}

type cudaUserObjectRelease_fn =
    unsafe extern "C" fn(object: cudaUserObject_t, count: ::core::ffi::c_uint) -> cudaError_t;
pub unsafe fn cudaUserObjectRelease(
    object: cudaUserObject_t,
    count: ::core::ffi::c_uint,
) -> cudaError_t {
    let sym: Symbol<cudaUserObjectRelease_fn> = unsafe {
        get_lib()
            .get(b"cudaUserObjectRelease\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(object, count) }
}

type cudaGraphRetainUserObject_fn = unsafe extern "C" fn(
    graph: cudaGraph_t,
    object: cudaUserObject_t,
    count: ::core::ffi::c_uint,
    flags: ::core::ffi::c_uint,
) -> cudaError_t;
pub unsafe fn cudaGraphRetainUserObject(
    graph: cudaGraph_t,
    object: cudaUserObject_t,
    count: ::core::ffi::c_uint,
    flags: ::core::ffi::c_uint,
) -> cudaError_t {
    let sym: Symbol<cudaGraphRetainUserObject_fn> = unsafe {
        get_lib()
            .get(b"cudaGraphRetainUserObject\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(graph, object, count, flags) }
}

type cudaGraphReleaseUserObject_fn = unsafe extern "C" fn(
    graph: cudaGraph_t,
    object: cudaUserObject_t,
    count: ::core::ffi::c_uint,
) -> cudaError_t;
pub unsafe fn cudaGraphReleaseUserObject(
    graph: cudaGraph_t,
    object: cudaUserObject_t,
    count: ::core::ffi::c_uint,
) -> cudaError_t {
    let sym: Symbol<cudaGraphReleaseUserObject_fn> = unsafe {
        get_lib()
            .get(b"cudaGraphReleaseUserObject\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(graph, object, count) }
}

type cudaGraphAddNode_fn = unsafe extern "C" fn(
    pGraphNode: *mut cudaGraphNode_t,
    graph: cudaGraph_t,
    pDependencies: *const cudaGraphNode_t,
    dependencyData: *const cudaGraphEdgeData,
    numDependencies: usize,
    nodeParams: *mut cudaGraphNodeParams,
) -> cudaError_t;
pub unsafe fn cudaGraphAddNode(
    pGraphNode: *mut cudaGraphNode_t,
    graph: cudaGraph_t,
    pDependencies: *const cudaGraphNode_t,
    dependencyData: *const cudaGraphEdgeData,
    numDependencies: usize,
    nodeParams: *mut cudaGraphNodeParams,
) -> cudaError_t {
    let sym: Symbol<cudaGraphAddNode_fn> = unsafe {
        get_lib()
            .get(b"cudaGraphAddNode\0")
            .expect("Missing symbol")
    };
    unsafe {
        (*sym)(
            pGraphNode,
            graph,
            pDependencies,
            dependencyData,
            numDependencies,
            nodeParams,
        )
    }
}

type cudaGraphNodeSetParams_fn = unsafe extern "C" fn(
    node: cudaGraphNode_t,
    nodeParams: *mut cudaGraphNodeParams,
) -> cudaError_t;
pub unsafe fn cudaGraphNodeSetParams(
    node: cudaGraphNode_t,
    nodeParams: *mut cudaGraphNodeParams,
) -> cudaError_t {
    let sym: Symbol<cudaGraphNodeSetParams_fn> = unsafe {
        get_lib()
            .get(b"cudaGraphNodeSetParams\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(node, nodeParams) }
}

type cudaGraphExecNodeSetParams_fn = unsafe extern "C" fn(
    graphExec: cudaGraphExec_t,
    node: cudaGraphNode_t,
    nodeParams: *mut cudaGraphNodeParams,
) -> cudaError_t;
pub unsafe fn cudaGraphExecNodeSetParams(
    graphExec: cudaGraphExec_t,
    node: cudaGraphNode_t,
    nodeParams: *mut cudaGraphNodeParams,
) -> cudaError_t {
    let sym: Symbol<cudaGraphExecNodeSetParams_fn> = unsafe {
        get_lib()
            .get(b"cudaGraphExecNodeSetParams\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(graphExec, node, nodeParams) }
}

type cudaGraphConditionalHandleCreate_fn = unsafe extern "C" fn(
    pHandle_out: *mut cudaGraphConditionalHandle,
    graph: cudaGraph_t,
    defaultLaunchValue: ::core::ffi::c_uint,
    flags: ::core::ffi::c_uint,
) -> cudaError_t;
pub unsafe fn cudaGraphConditionalHandleCreate(
    pHandle_out: *mut cudaGraphConditionalHandle,
    graph: cudaGraph_t,
    defaultLaunchValue: ::core::ffi::c_uint,
    flags: ::core::ffi::c_uint,
) -> cudaError_t {
    let sym: Symbol<cudaGraphConditionalHandleCreate_fn> = unsafe {
        get_lib()
            .get(b"cudaGraphConditionalHandleCreate\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(pHandle_out, graph, defaultLaunchValue, flags) }
}

type cudaGetDriverEntryPoint_fn = unsafe extern "C" fn(
    symbol: *const ::core::ffi::c_char,
    funcPtr: *mut *mut ::core::ffi::c_void,
    flags: ::core::ffi::c_ulonglong,
    driverStatus: *mut cudaDriverEntryPointQueryResult,
) -> cudaError_t;
pub unsafe fn cudaGetDriverEntryPoint(
    symbol: *const ::core::ffi::c_char,
    funcPtr: *mut *mut ::core::ffi::c_void,
    flags: ::core::ffi::c_ulonglong,
    driverStatus: *mut cudaDriverEntryPointQueryResult,
) -> cudaError_t {
    let sym: Symbol<cudaGetDriverEntryPoint_fn> = unsafe {
        get_lib()
            .get(b"cudaGetDriverEntryPoint\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(symbol, funcPtr, flags, driverStatus) }
}

type cudaGetDriverEntryPointByVersion_fn = unsafe extern "C" fn(
    symbol: *const ::core::ffi::c_char,
    funcPtr: *mut *mut ::core::ffi::c_void,
    cudaVersion: ::core::ffi::c_uint,
    flags: ::core::ffi::c_ulonglong,
    driverStatus: *mut cudaDriverEntryPointQueryResult,
) -> cudaError_t;
pub unsafe fn cudaGetDriverEntryPointByVersion(
    symbol: *const ::core::ffi::c_char,
    funcPtr: *mut *mut ::core::ffi::c_void,
    cudaVersion: ::core::ffi::c_uint,
    flags: ::core::ffi::c_ulonglong,
    driverStatus: *mut cudaDriverEntryPointQueryResult,
) -> cudaError_t {
    let sym: Symbol<cudaGetDriverEntryPointByVersion_fn> = unsafe {
        get_lib()
            .get(b"cudaGetDriverEntryPointByVersion\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(symbol, funcPtr, cudaVersion, flags, driverStatus) }
}

type cudaLibraryLoadData_fn = unsafe extern "C" fn(
    library: *mut cudaLibrary_t,
    code: *const ::core::ffi::c_void,
    jitOptions: *mut cudaJitOption,
    jitOptionsValues: *mut *mut ::core::ffi::c_void,
    numJitOptions: ::core::ffi::c_uint,
    libraryOptions: *mut cudaLibraryOption,
    libraryOptionValues: *mut *mut ::core::ffi::c_void,
    numLibraryOptions: ::core::ffi::c_uint,
) -> cudaError_t;
pub unsafe fn cudaLibraryLoadData(
    library: *mut cudaLibrary_t,
    code: *const ::core::ffi::c_void,
    jitOptions: *mut cudaJitOption,
    jitOptionsValues: *mut *mut ::core::ffi::c_void,
    numJitOptions: ::core::ffi::c_uint,
    libraryOptions: *mut cudaLibraryOption,
    libraryOptionValues: *mut *mut ::core::ffi::c_void,
    numLibraryOptions: ::core::ffi::c_uint,
) -> cudaError_t {
    let sym: Symbol<cudaLibraryLoadData_fn> = unsafe {
        get_lib()
            .get(b"cudaLibraryLoadData\0")
            .expect("Missing symbol")
    };
    unsafe {
        (*sym)(
            library,
            code,
            jitOptions,
            jitOptionsValues,
            numJitOptions,
            libraryOptions,
            libraryOptionValues,
            numLibraryOptions,
        )
    }
}

type cudaLibraryLoadFromFile_fn = unsafe extern "C" fn(
    library: *mut cudaLibrary_t,
    fileName: *const ::core::ffi::c_char,
    jitOptions: *mut cudaJitOption,
    jitOptionsValues: *mut *mut ::core::ffi::c_void,
    numJitOptions: ::core::ffi::c_uint,
    libraryOptions: *mut cudaLibraryOption,
    libraryOptionValues: *mut *mut ::core::ffi::c_void,
    numLibraryOptions: ::core::ffi::c_uint,
) -> cudaError_t;
pub unsafe fn cudaLibraryLoadFromFile(
    library: *mut cudaLibrary_t,
    fileName: *const ::core::ffi::c_char,
    jitOptions: *mut cudaJitOption,
    jitOptionsValues: *mut *mut ::core::ffi::c_void,
    numJitOptions: ::core::ffi::c_uint,
    libraryOptions: *mut cudaLibraryOption,
    libraryOptionValues: *mut *mut ::core::ffi::c_void,
    numLibraryOptions: ::core::ffi::c_uint,
) -> cudaError_t {
    let sym: Symbol<cudaLibraryLoadFromFile_fn> = unsafe {
        get_lib()
            .get(b"cudaLibraryLoadFromFile\0")
            .expect("Missing symbol")
    };
    unsafe {
        (*sym)(
            library,
            fileName,
            jitOptions,
            jitOptionsValues,
            numJitOptions,
            libraryOptions,
            libraryOptionValues,
            numLibraryOptions,
        )
    }
}

type cudaLibraryUnload_fn = unsafe extern "C" fn(library: cudaLibrary_t) -> cudaError_t;
pub unsafe fn cudaLibraryUnload(library: cudaLibrary_t) -> cudaError_t {
    let sym: Symbol<cudaLibraryUnload_fn> = unsafe {
        get_lib()
            .get(b"cudaLibraryUnload\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(library) }
}

type cudaLibraryGetKernel_fn = unsafe extern "C" fn(
    pKernel: *mut cudaKernel_t,
    library: cudaLibrary_t,
    name: *const ::core::ffi::c_char,
) -> cudaError_t;
pub unsafe fn cudaLibraryGetKernel(
    pKernel: *mut cudaKernel_t,
    library: cudaLibrary_t,
    name: *const ::core::ffi::c_char,
) -> cudaError_t {
    let sym: Symbol<cudaLibraryGetKernel_fn> = unsafe {
        get_lib()
            .get(b"cudaLibraryGetKernel\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(pKernel, library, name) }
}

type cudaLibraryGetGlobal_fn = unsafe extern "C" fn(
    dptr: *mut *mut ::core::ffi::c_void,
    bytes: *mut usize,
    library: cudaLibrary_t,
    name: *const ::core::ffi::c_char,
) -> cudaError_t;
pub unsafe fn cudaLibraryGetGlobal(
    dptr: *mut *mut ::core::ffi::c_void,
    bytes: *mut usize,
    library: cudaLibrary_t,
    name: *const ::core::ffi::c_char,
) -> cudaError_t {
    let sym: Symbol<cudaLibraryGetGlobal_fn> = unsafe {
        get_lib()
            .get(b"cudaLibraryGetGlobal\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(dptr, bytes, library, name) }
}

type cudaLibraryGetManaged_fn = unsafe extern "C" fn(
    dptr: *mut *mut ::core::ffi::c_void,
    bytes: *mut usize,
    library: cudaLibrary_t,
    name: *const ::core::ffi::c_char,
) -> cudaError_t;
pub unsafe fn cudaLibraryGetManaged(
    dptr: *mut *mut ::core::ffi::c_void,
    bytes: *mut usize,
    library: cudaLibrary_t,
    name: *const ::core::ffi::c_char,
) -> cudaError_t {
    let sym: Symbol<cudaLibraryGetManaged_fn> = unsafe {
        get_lib()
            .get(b"cudaLibraryGetManaged\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(dptr, bytes, library, name) }
}

type cudaLibraryGetUnifiedFunction_fn = unsafe extern "C" fn(
    fptr: *mut *mut ::core::ffi::c_void,
    library: cudaLibrary_t,
    symbol: *const ::core::ffi::c_char,
) -> cudaError_t;
pub unsafe fn cudaLibraryGetUnifiedFunction(
    fptr: *mut *mut ::core::ffi::c_void,
    library: cudaLibrary_t,
    symbol: *const ::core::ffi::c_char,
) -> cudaError_t {
    let sym: Symbol<cudaLibraryGetUnifiedFunction_fn> = unsafe {
        get_lib()
            .get(b"cudaLibraryGetUnifiedFunction\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(fptr, library, symbol) }
}

type cudaLibraryGetKernelCount_fn =
    unsafe extern "C" fn(count: *mut ::core::ffi::c_uint, lib: cudaLibrary_t) -> cudaError_t;
pub unsafe fn cudaLibraryGetKernelCount(
    count: *mut ::core::ffi::c_uint,
    lib: cudaLibrary_t,
) -> cudaError_t {
    let sym: Symbol<cudaLibraryGetKernelCount_fn> = unsafe {
        get_lib()
            .get(b"cudaLibraryGetKernelCount\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(count, lib) }
}

type cudaLibraryEnumerateKernels_fn = unsafe extern "C" fn(
    kernels: *mut cudaKernel_t,
    numKernels: ::core::ffi::c_uint,
    lib: cudaLibrary_t,
) -> cudaError_t;
pub unsafe fn cudaLibraryEnumerateKernels(
    kernels: *mut cudaKernel_t,
    numKernels: ::core::ffi::c_uint,
    lib: cudaLibrary_t,
) -> cudaError_t {
    let sym: Symbol<cudaLibraryEnumerateKernels_fn> = unsafe {
        get_lib()
            .get(b"cudaLibraryEnumerateKernels\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(kernels, numKernels, lib) }
}

type cudaKernelSetAttributeForDevice_fn = unsafe extern "C" fn(
    kernel: cudaKernel_t,
    attr: cudaFuncAttribute,
    value: ::core::ffi::c_int,
    device: ::core::ffi::c_int,
) -> cudaError_t;
pub unsafe fn cudaKernelSetAttributeForDevice(
    kernel: cudaKernel_t,
    attr: cudaFuncAttribute,
    value: ::core::ffi::c_int,
    device: ::core::ffi::c_int,
) -> cudaError_t {
    let sym: Symbol<cudaKernelSetAttributeForDevice_fn> = unsafe {
        get_lib()
            .get(b"cudaKernelSetAttributeForDevice\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(kernel, attr, value, device) }
}

type cudaGetExportTable_fn = unsafe extern "C" fn(
    ppExportTable: *mut *const ::core::ffi::c_void,
    pExportTableId: *const cudaUUID_t,
) -> cudaError_t;
pub unsafe fn cudaGetExportTable(
    ppExportTable: *mut *const ::core::ffi::c_void,
    pExportTableId: *const cudaUUID_t,
) -> cudaError_t {
    let sym: Symbol<cudaGetExportTable_fn> = unsafe {
        get_lib()
            .get(b"cudaGetExportTable\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(ppExportTable, pExportTableId) }
}

type cudaGetFuncBySymbol_fn = unsafe extern "C" fn(
    functionPtr: *mut cudaFunction_t,
    symbolPtr: *const ::core::ffi::c_void,
) -> cudaError_t;
pub unsafe fn cudaGetFuncBySymbol(
    functionPtr: *mut cudaFunction_t,
    symbolPtr: *const ::core::ffi::c_void,
) -> cudaError_t {
    let sym: Symbol<cudaGetFuncBySymbol_fn> = unsafe {
        get_lib()
            .get(b"cudaGetFuncBySymbol\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(functionPtr, symbolPtr) }
}

type cudaGetKernel_fn = unsafe extern "C" fn(
    kernelPtr: *mut cudaKernel_t,
    entryFuncAddr: *const ::core::ffi::c_void,
) -> cudaError_t;
pub unsafe fn cudaGetKernel(
    kernelPtr: *mut cudaKernel_t,
    entryFuncAddr: *const ::core::ffi::c_void,
) -> cudaError_t {
    let sym: Symbol<cudaGetKernel_fn> =
        unsafe { get_lib().get(b"cudaGetKernel\0").expect("Missing symbol") };
    unsafe { (*sym)(kernelPtr, entryFuncAddr) }
}

type cudnnGetVersion_fn = unsafe extern "C" fn() -> usize;
pub unsafe fn cudnnGetVersion() -> usize {
    let sym: Symbol<cudnnGetVersion_fn> =
        unsafe { get_lib().get(b"cudnnGetVersion\0").expect("Missing symbol") };
    unsafe { (*sym)() }
}

type cudnnGetMaxDeviceVersion_fn = unsafe extern "C" fn() -> usize;
pub unsafe fn cudnnGetMaxDeviceVersion() -> usize {
    let sym: Symbol<cudnnGetMaxDeviceVersion_fn> = unsafe {
        get_lib()
            .get(b"cudnnGetMaxDeviceVersion\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)() }
}

type cudnnGetCudartVersion_fn = unsafe extern "C" fn() -> usize;
pub unsafe fn cudnnGetCudartVersion() -> usize {
    let sym: Symbol<cudnnGetCudartVersion_fn> = unsafe {
        get_lib()
            .get(b"cudnnGetCudartVersion\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)() }
}

type cudnnGetErrorString_fn =
    unsafe extern "C" fn(status: cudnnStatus_t) -> *const ::core::ffi::c_char;
pub unsafe fn cudnnGetErrorString(status: cudnnStatus_t) -> *const ::core::ffi::c_char {
    let sym: Symbol<cudnnGetErrorString_fn> = unsafe {
        get_lib()
            .get(b"cudnnGetErrorString\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(status) }
}

type cudnnGetLastErrorString_fn =
    unsafe extern "C" fn(message: *mut ::core::ffi::c_char, max_size: usize) -> ();
pub unsafe fn cudnnGetLastErrorString(message: *mut ::core::ffi::c_char, max_size: usize) -> () {
    let sym: Symbol<cudnnGetLastErrorString_fn> = unsafe {
        get_lib()
            .get(b"cudnnGetLastErrorString\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(message, max_size) }
}

type cudnnQueryRuntimeError_fn = unsafe extern "C" fn(
    handle: cudnnHandle_t,
    rstatus: *mut cudnnStatus_t,
    mode: cudnnErrQueryMode_t,
    tag: *mut cudnnRuntimeTag_t,
) -> cudnnStatus_t;
pub unsafe fn cudnnQueryRuntimeError(
    handle: cudnnHandle_t,
    rstatus: *mut cudnnStatus_t,
    mode: cudnnErrQueryMode_t,
    tag: *mut cudnnRuntimeTag_t,
) -> cudnnStatus_t {
    let sym: Symbol<cudnnQueryRuntimeError_fn> = unsafe {
        get_lib()
            .get(b"cudnnQueryRuntimeError\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(handle, rstatus, mode, tag) }
}

type cudnnGetProperty_fn = unsafe extern "C" fn(
    type_: libraryPropertyType,
    value: *mut ::core::ffi::c_int,
) -> cudnnStatus_t;
pub unsafe fn cudnnGetProperty(
    type_: libraryPropertyType,
    value: *mut ::core::ffi::c_int,
) -> cudnnStatus_t {
    let sym: Symbol<cudnnGetProperty_fn> = unsafe {
        get_lib()
            .get(b"cudnnGetProperty\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(type_, value) }
}

type cudnnCreate_fn = unsafe extern "C" fn(handle: *mut cudnnHandle_t) -> cudnnStatus_t;
pub unsafe fn cudnnCreate(handle: *mut cudnnHandle_t) -> cudnnStatus_t {
    let sym: Symbol<cudnnCreate_fn> =
        unsafe { get_lib().get(b"cudnnCreate\0").expect("Missing symbol") };
    unsafe { (*sym)(handle) }
}

type cudnnDestroy_fn = unsafe extern "C" fn(handle: cudnnHandle_t) -> cudnnStatus_t;
pub unsafe fn cudnnDestroy(handle: cudnnHandle_t) -> cudnnStatus_t {
    let sym: Symbol<cudnnDestroy_fn> =
        unsafe { get_lib().get(b"cudnnDestroy\0").expect("Missing symbol") };
    unsafe { (*sym)(handle) }
}

type cudnnSetStream_fn =
    unsafe extern "C" fn(handle: cudnnHandle_t, streamId: cudaStream_t) -> cudnnStatus_t;
pub unsafe fn cudnnSetStream(handle: cudnnHandle_t, streamId: cudaStream_t) -> cudnnStatus_t {
    let sym: Symbol<cudnnSetStream_fn> =
        unsafe { get_lib().get(b"cudnnSetStream\0").expect("Missing symbol") };
    unsafe { (*sym)(handle, streamId) }
}

type cudnnGetStream_fn =
    unsafe extern "C" fn(handle: cudnnHandle_t, streamId: *mut cudaStream_t) -> cudnnStatus_t;
pub unsafe fn cudnnGetStream(handle: cudnnHandle_t, streamId: *mut cudaStream_t) -> cudnnStatus_t {
    let sym: Symbol<cudnnGetStream_fn> =
        unsafe { get_lib().get(b"cudnnGetStream\0").expect("Missing symbol") };
    unsafe { (*sym)(handle, streamId) }
}

type cudnnSetCallback_fn = unsafe extern "C" fn(
    mask: ::core::ffi::c_uint,
    udata: *mut ::core::ffi::c_void,
    fptr: cudnnCallback_t,
) -> cudnnStatus_t;
pub unsafe fn cudnnSetCallback(
    mask: ::core::ffi::c_uint,
    udata: *mut ::core::ffi::c_void,
    fptr: cudnnCallback_t,
) -> cudnnStatus_t {
    let sym: Symbol<cudnnSetCallback_fn> = unsafe {
        get_lib()
            .get(b"cudnnSetCallback\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(mask, udata, fptr) }
}

type cudnnGetCallback_fn = unsafe extern "C" fn(
    mask: *mut ::core::ffi::c_uint,
    udata: *mut *mut ::core::ffi::c_void,
    fptr: *mut cudnnCallback_t,
) -> cudnnStatus_t;
pub unsafe fn cudnnGetCallback(
    mask: *mut ::core::ffi::c_uint,
    udata: *mut *mut ::core::ffi::c_void,
    fptr: *mut cudnnCallback_t,
) -> cudnnStatus_t {
    let sym: Symbol<cudnnGetCallback_fn> = unsafe {
        get_lib()
            .get(b"cudnnGetCallback\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(mask, udata, fptr) }
}

type cudnnGraphVersionCheck_fn = unsafe extern "C" fn() -> cudnnStatus_t;
pub unsafe fn cudnnGraphVersionCheck() -> cudnnStatus_t {
    let sym: Symbol<cudnnGraphVersionCheck_fn> = unsafe {
        get_lib()
            .get(b"cudnnGraphVersionCheck\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)() }
}

type cudnnBackendCreateDescriptor_fn = unsafe extern "C" fn(
    descriptorType: cudnnBackendDescriptorType_t,
    descriptor: *mut cudnnBackendDescriptor_t,
) -> cudnnStatus_t;
pub unsafe fn cudnnBackendCreateDescriptor(
    descriptorType: cudnnBackendDescriptorType_t,
    descriptor: *mut cudnnBackendDescriptor_t,
) -> cudnnStatus_t {
    let sym: Symbol<cudnnBackendCreateDescriptor_fn> = unsafe {
        get_lib()
            .get(b"cudnnBackendCreateDescriptor\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(descriptorType, descriptor) }
}

type cudnnBackendDestroyDescriptor_fn =
    unsafe extern "C" fn(descriptor: cudnnBackendDescriptor_t) -> cudnnStatus_t;
pub unsafe fn cudnnBackendDestroyDescriptor(descriptor: cudnnBackendDescriptor_t) -> cudnnStatus_t {
    let sym: Symbol<cudnnBackendDestroyDescriptor_fn> = unsafe {
        get_lib()
            .get(b"cudnnBackendDestroyDescriptor\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(descriptor) }
}

type cudnnBackendInitialize_fn =
    unsafe extern "C" fn(descriptor: cudnnBackendDescriptor_t) -> cudnnStatus_t;
pub unsafe fn cudnnBackendInitialize(descriptor: cudnnBackendDescriptor_t) -> cudnnStatus_t {
    let sym: Symbol<cudnnBackendInitialize_fn> = unsafe {
        get_lib()
            .get(b"cudnnBackendInitialize\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(descriptor) }
}

type cudnnBackendFinalize_fn =
    unsafe extern "C" fn(descriptor: cudnnBackendDescriptor_t) -> cudnnStatus_t;
pub unsafe fn cudnnBackendFinalize(descriptor: cudnnBackendDescriptor_t) -> cudnnStatus_t {
    let sym: Symbol<cudnnBackendFinalize_fn> = unsafe {
        get_lib()
            .get(b"cudnnBackendFinalize\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(descriptor) }
}

type cudnnBackendSetAttribute_fn = unsafe extern "C" fn(
    descriptor: cudnnBackendDescriptor_t,
    attributeName: cudnnBackendAttributeName_t,
    attributeType: cudnnBackendAttributeType_t,
    elementCount: i64,
    arrayOfElements: *const ::core::ffi::c_void,
) -> cudnnStatus_t;
pub unsafe fn cudnnBackendSetAttribute(
    descriptor: cudnnBackendDescriptor_t,
    attributeName: cudnnBackendAttributeName_t,
    attributeType: cudnnBackendAttributeType_t,
    elementCount: i64,
    arrayOfElements: *const ::core::ffi::c_void,
) -> cudnnStatus_t {
    let sym: Symbol<cudnnBackendSetAttribute_fn> = unsafe {
        get_lib()
            .get(b"cudnnBackendSetAttribute\0")
            .expect("Missing symbol")
    };
    unsafe {
        (*sym)(
            descriptor,
            attributeName,
            attributeType,
            elementCount,
            arrayOfElements,
        )
    }
}

type cudnnBackendGetAttribute_fn = unsafe extern "C" fn(
    descriptor: cudnnBackendDescriptor_t,
    attributeName: cudnnBackendAttributeName_t,
    attributeType: cudnnBackendAttributeType_t,
    requestedElementCount: i64,
    elementCount: *mut i64,
    arrayOfElements: *mut ::core::ffi::c_void,
) -> cudnnStatus_t;
pub unsafe fn cudnnBackendGetAttribute(
    descriptor: cudnnBackendDescriptor_t,
    attributeName: cudnnBackendAttributeName_t,
    attributeType: cudnnBackendAttributeType_t,
    requestedElementCount: i64,
    elementCount: *mut i64,
    arrayOfElements: *mut ::core::ffi::c_void,
) -> cudnnStatus_t {
    let sym: Symbol<cudnnBackendGetAttribute_fn> = unsafe {
        get_lib()
            .get(b"cudnnBackendGetAttribute\0")
            .expect("Missing symbol")
    };
    unsafe {
        (*sym)(
            descriptor,
            attributeName,
            attributeType,
            requestedElementCount,
            elementCount,
            arrayOfElements,
        )
    }
}

type cudnnBackendExecute_fn = unsafe extern "C" fn(
    handle: cudnnHandle_t,
    executionPlan: cudnnBackendDescriptor_t,
    variantPack: cudnnBackendDescriptor_t,
) -> cudnnStatus_t;
pub unsafe fn cudnnBackendExecute(
    handle: cudnnHandle_t,
    executionPlan: cudnnBackendDescriptor_t,
    variantPack: cudnnBackendDescriptor_t,
) -> cudnnStatus_t {
    let sym: Symbol<cudnnBackendExecute_fn> = unsafe {
        get_lib()
            .get(b"cudnnBackendExecute\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(handle, executionPlan, variantPack) }
}

type cudnnBackendPopulateCudaGraph_fn = unsafe extern "C" fn(
    handle: cudnnHandle_t,
    executionPlan: cudnnBackendDescriptor_t,
    variantPack: cudnnBackendDescriptor_t,
    graph: cudaGraph_t,
) -> cudnnStatus_t;
pub unsafe fn cudnnBackendPopulateCudaGraph(
    handle: cudnnHandle_t,
    executionPlan: cudnnBackendDescriptor_t,
    variantPack: cudnnBackendDescriptor_t,
    graph: cudaGraph_t,
) -> cudnnStatus_t {
    let sym: Symbol<cudnnBackendPopulateCudaGraph_fn> = unsafe {
        get_lib()
            .get(b"cudnnBackendPopulateCudaGraph\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(handle, executionPlan, variantPack, graph) }
}

type cudnnBackendUpdateCudaGraph_fn = unsafe extern "C" fn(
    handle: cudnnHandle_t,
    executionPlan: cudnnBackendDescriptor_t,
    variantPack: cudnnBackendDescriptor_t,
    graph: cudaGraph_t,
) -> cudnnStatus_t;
pub unsafe fn cudnnBackendUpdateCudaGraph(
    handle: cudnnHandle_t,
    executionPlan: cudnnBackendDescriptor_t,
    variantPack: cudnnBackendDescriptor_t,
    graph: cudaGraph_t,
) -> cudnnStatus_t {
    let sym: Symbol<cudnnBackendUpdateCudaGraph_fn> = unsafe {
        get_lib()
            .get(b"cudnnBackendUpdateCudaGraph\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(handle, executionPlan, variantPack, graph) }
}

type cudnnCreateTensorDescriptor_fn =
    unsafe extern "C" fn(tensorDesc: *mut cudnnTensorDescriptor_t) -> cudnnStatus_t;
pub unsafe fn cudnnCreateTensorDescriptor(
    tensorDesc: *mut cudnnTensorDescriptor_t,
) -> cudnnStatus_t {
    let sym: Symbol<cudnnCreateTensorDescriptor_fn> = unsafe {
        get_lib()
            .get(b"cudnnCreateTensorDescriptor\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(tensorDesc) }
}

type cudnnSetTensor4dDescriptor_fn = unsafe extern "C" fn(
    tensorDesc: cudnnTensorDescriptor_t,
    format: cudnnTensorFormat_t,
    dataType: cudnnDataType_t,
    n: ::core::ffi::c_int,
    c: ::core::ffi::c_int,
    h: ::core::ffi::c_int,
    w: ::core::ffi::c_int,
) -> cudnnStatus_t;
pub unsafe fn cudnnSetTensor4dDescriptor(
    tensorDesc: cudnnTensorDescriptor_t,
    format: cudnnTensorFormat_t,
    dataType: cudnnDataType_t,
    n: ::core::ffi::c_int,
    c: ::core::ffi::c_int,
    h: ::core::ffi::c_int,
    w: ::core::ffi::c_int,
) -> cudnnStatus_t {
    let sym: Symbol<cudnnSetTensor4dDescriptor_fn> = unsafe {
        get_lib()
            .get(b"cudnnSetTensor4dDescriptor\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(tensorDesc, format, dataType, n, c, h, w) }
}

type cudnnSetTensor4dDescriptorEx_fn = unsafe extern "C" fn(
    tensorDesc: cudnnTensorDescriptor_t,
    dataType: cudnnDataType_t,
    n: ::core::ffi::c_int,
    c: ::core::ffi::c_int,
    h: ::core::ffi::c_int,
    w: ::core::ffi::c_int,
    nStride: ::core::ffi::c_int,
    cStride: ::core::ffi::c_int,
    hStride: ::core::ffi::c_int,
    wStride: ::core::ffi::c_int,
) -> cudnnStatus_t;
pub unsafe fn cudnnSetTensor4dDescriptorEx(
    tensorDesc: cudnnTensorDescriptor_t,
    dataType: cudnnDataType_t,
    n: ::core::ffi::c_int,
    c: ::core::ffi::c_int,
    h: ::core::ffi::c_int,
    w: ::core::ffi::c_int,
    nStride: ::core::ffi::c_int,
    cStride: ::core::ffi::c_int,
    hStride: ::core::ffi::c_int,
    wStride: ::core::ffi::c_int,
) -> cudnnStatus_t {
    let sym: Symbol<cudnnSetTensor4dDescriptorEx_fn> = unsafe {
        get_lib()
            .get(b"cudnnSetTensor4dDescriptorEx\0")
            .expect("Missing symbol")
    };
    unsafe {
        (*sym)(
            tensorDesc, dataType, n, c, h, w, nStride, cStride, hStride, wStride,
        )
    }
}

type cudnnGetTensor4dDescriptor_fn = unsafe extern "C" fn(
    tensorDesc: cudnnTensorDescriptor_t,
    dataType: *mut cudnnDataType_t,
    n: *mut ::core::ffi::c_int,
    c: *mut ::core::ffi::c_int,
    h: *mut ::core::ffi::c_int,
    w: *mut ::core::ffi::c_int,
    nStride: *mut ::core::ffi::c_int,
    cStride: *mut ::core::ffi::c_int,
    hStride: *mut ::core::ffi::c_int,
    wStride: *mut ::core::ffi::c_int,
) -> cudnnStatus_t;
pub unsafe fn cudnnGetTensor4dDescriptor(
    tensorDesc: cudnnTensorDescriptor_t,
    dataType: *mut cudnnDataType_t,
    n: *mut ::core::ffi::c_int,
    c: *mut ::core::ffi::c_int,
    h: *mut ::core::ffi::c_int,
    w: *mut ::core::ffi::c_int,
    nStride: *mut ::core::ffi::c_int,
    cStride: *mut ::core::ffi::c_int,
    hStride: *mut ::core::ffi::c_int,
    wStride: *mut ::core::ffi::c_int,
) -> cudnnStatus_t {
    let sym: Symbol<cudnnGetTensor4dDescriptor_fn> = unsafe {
        get_lib()
            .get(b"cudnnGetTensor4dDescriptor\0")
            .expect("Missing symbol")
    };
    unsafe {
        (*sym)(
            tensorDesc, dataType, n, c, h, w, nStride, cStride, hStride, wStride,
        )
    }
}

type cudnnSetTensorNdDescriptor_fn = unsafe extern "C" fn(
    tensorDesc: cudnnTensorDescriptor_t,
    dataType: cudnnDataType_t,
    nbDims: ::core::ffi::c_int,
    dimA: *const ::core::ffi::c_int,
    strideA: *const ::core::ffi::c_int,
) -> cudnnStatus_t;
pub unsafe fn cudnnSetTensorNdDescriptor(
    tensorDesc: cudnnTensorDescriptor_t,
    dataType: cudnnDataType_t,
    nbDims: ::core::ffi::c_int,
    dimA: *const ::core::ffi::c_int,
    strideA: *const ::core::ffi::c_int,
) -> cudnnStatus_t {
    let sym: Symbol<cudnnSetTensorNdDescriptor_fn> = unsafe {
        get_lib()
            .get(b"cudnnSetTensorNdDescriptor\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(tensorDesc, dataType, nbDims, dimA, strideA) }
}

type cudnnSetTensorNdDescriptorEx_fn = unsafe extern "C" fn(
    tensorDesc: cudnnTensorDescriptor_t,
    format: cudnnTensorFormat_t,
    dataType: cudnnDataType_t,
    nbDims: ::core::ffi::c_int,
    dimA: *const ::core::ffi::c_int,
) -> cudnnStatus_t;
pub unsafe fn cudnnSetTensorNdDescriptorEx(
    tensorDesc: cudnnTensorDescriptor_t,
    format: cudnnTensorFormat_t,
    dataType: cudnnDataType_t,
    nbDims: ::core::ffi::c_int,
    dimA: *const ::core::ffi::c_int,
) -> cudnnStatus_t {
    let sym: Symbol<cudnnSetTensorNdDescriptorEx_fn> = unsafe {
        get_lib()
            .get(b"cudnnSetTensorNdDescriptorEx\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(tensorDesc, format, dataType, nbDims, dimA) }
}

type cudnnGetTensorNdDescriptor_fn = unsafe extern "C" fn(
    tensorDesc: cudnnTensorDescriptor_t,
    nbDimsRequested: ::core::ffi::c_int,
    dataType: *mut cudnnDataType_t,
    nbDims: *mut ::core::ffi::c_int,
    dimA: *mut ::core::ffi::c_int,
    strideA: *mut ::core::ffi::c_int,
) -> cudnnStatus_t;
pub unsafe fn cudnnGetTensorNdDescriptor(
    tensorDesc: cudnnTensorDescriptor_t,
    nbDimsRequested: ::core::ffi::c_int,
    dataType: *mut cudnnDataType_t,
    nbDims: *mut ::core::ffi::c_int,
    dimA: *mut ::core::ffi::c_int,
    strideA: *mut ::core::ffi::c_int,
) -> cudnnStatus_t {
    let sym: Symbol<cudnnGetTensorNdDescriptor_fn> = unsafe {
        get_lib()
            .get(b"cudnnGetTensorNdDescriptor\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(tensorDesc, nbDimsRequested, dataType, nbDims, dimA, strideA) }
}

type cudnnGetTensorSizeInBytes_fn =
    unsafe extern "C" fn(tensorDesc: cudnnTensorDescriptor_t, size: *mut usize) -> cudnnStatus_t;
pub unsafe fn cudnnGetTensorSizeInBytes(
    tensorDesc: cudnnTensorDescriptor_t,
    size: *mut usize,
) -> cudnnStatus_t {
    let sym: Symbol<cudnnGetTensorSizeInBytes_fn> = unsafe {
        get_lib()
            .get(b"cudnnGetTensorSizeInBytes\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(tensorDesc, size) }
}

type cudnnDestroyTensorDescriptor_fn =
    unsafe extern "C" fn(tensorDesc: cudnnTensorDescriptor_t) -> cudnnStatus_t;
pub unsafe fn cudnnDestroyTensorDescriptor(tensorDesc: cudnnTensorDescriptor_t) -> cudnnStatus_t {
    let sym: Symbol<cudnnDestroyTensorDescriptor_fn> = unsafe {
        get_lib()
            .get(b"cudnnDestroyTensorDescriptor\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(tensorDesc) }
}

type cudnnInitTransformDest_fn = unsafe extern "C" fn(
    transformDesc: cudnnTensorTransformDescriptor_t,
    srcDesc: cudnnTensorDescriptor_t,
    destDesc: cudnnTensorDescriptor_t,
    destSizeInBytes: *mut usize,
) -> cudnnStatus_t;
pub unsafe fn cudnnInitTransformDest(
    transformDesc: cudnnTensorTransformDescriptor_t,
    srcDesc: cudnnTensorDescriptor_t,
    destDesc: cudnnTensorDescriptor_t,
    destSizeInBytes: *mut usize,
) -> cudnnStatus_t {
    let sym: Symbol<cudnnInitTransformDest_fn> = unsafe {
        get_lib()
            .get(b"cudnnInitTransformDest\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(transformDesc, srcDesc, destDesc, destSizeInBytes) }
}

type cudnnCreateTensorTransformDescriptor_fn =
    unsafe extern "C" fn(transformDesc: *mut cudnnTensorTransformDescriptor_t) -> cudnnStatus_t;
pub unsafe fn cudnnCreateTensorTransformDescriptor(
    transformDesc: *mut cudnnTensorTransformDescriptor_t,
) -> cudnnStatus_t {
    let sym: Symbol<cudnnCreateTensorTransformDescriptor_fn> = unsafe {
        get_lib()
            .get(b"cudnnCreateTensorTransformDescriptor\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(transformDesc) }
}

type cudnnSetTensorTransformDescriptor_fn = unsafe extern "C" fn(
    transformDesc: cudnnTensorTransformDescriptor_t,
    nbDims: u32,
    destFormat: cudnnTensorFormat_t,
    padBeforeA: *const i32,
    padAfterA: *const i32,
    foldA: *const u32,
    direction: cudnnFoldingDirection_t,
) -> cudnnStatus_t;
pub unsafe fn cudnnSetTensorTransformDescriptor(
    transformDesc: cudnnTensorTransformDescriptor_t,
    nbDims: u32,
    destFormat: cudnnTensorFormat_t,
    padBeforeA: *const i32,
    padAfterA: *const i32,
    foldA: *const u32,
    direction: cudnnFoldingDirection_t,
) -> cudnnStatus_t {
    let sym: Symbol<cudnnSetTensorTransformDescriptor_fn> = unsafe {
        get_lib()
            .get(b"cudnnSetTensorTransformDescriptor\0")
            .expect("Missing symbol")
    };
    unsafe {
        (*sym)(
            transformDesc,
            nbDims,
            destFormat,
            padBeforeA,
            padAfterA,
            foldA,
            direction,
        )
    }
}

type cudnnGetTensorTransformDescriptor_fn = unsafe extern "C" fn(
    transformDesc: cudnnTensorTransformDescriptor_t,
    nbDimsRequested: u32,
    destFormat: *mut cudnnTensorFormat_t,
    padBeforeA: *mut i32,
    padAfterA: *mut i32,
    foldA: *mut u32,
    direction: *mut cudnnFoldingDirection_t,
) -> cudnnStatus_t;
pub unsafe fn cudnnGetTensorTransformDescriptor(
    transformDesc: cudnnTensorTransformDescriptor_t,
    nbDimsRequested: u32,
    destFormat: *mut cudnnTensorFormat_t,
    padBeforeA: *mut i32,
    padAfterA: *mut i32,
    foldA: *mut u32,
    direction: *mut cudnnFoldingDirection_t,
) -> cudnnStatus_t {
    let sym: Symbol<cudnnGetTensorTransformDescriptor_fn> = unsafe {
        get_lib()
            .get(b"cudnnGetTensorTransformDescriptor\0")
            .expect("Missing symbol")
    };
    unsafe {
        (*sym)(
            transformDesc,
            nbDimsRequested,
            destFormat,
            padBeforeA,
            padAfterA,
            foldA,
            direction,
        )
    }
}

type cudnnDestroyTensorTransformDescriptor_fn =
    unsafe extern "C" fn(transformDesc: cudnnTensorTransformDescriptor_t) -> cudnnStatus_t;
pub unsafe fn cudnnDestroyTensorTransformDescriptor(
    transformDesc: cudnnTensorTransformDescriptor_t,
) -> cudnnStatus_t {
    let sym: Symbol<cudnnDestroyTensorTransformDescriptor_fn> = unsafe {
        get_lib()
            .get(b"cudnnDestroyTensorTransformDescriptor\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(transformDesc) }
}

type cudnnTransformTensor_fn = unsafe extern "C" fn(
    handle: cudnnHandle_t,
    alpha: *const ::core::ffi::c_void,
    xDesc: cudnnTensorDescriptor_t,
    x: *const ::core::ffi::c_void,
    beta: *const ::core::ffi::c_void,
    yDesc: cudnnTensorDescriptor_t,
    y: *mut ::core::ffi::c_void,
) -> cudnnStatus_t;
pub unsafe fn cudnnTransformTensor(
    handle: cudnnHandle_t,
    alpha: *const ::core::ffi::c_void,
    xDesc: cudnnTensorDescriptor_t,
    x: *const ::core::ffi::c_void,
    beta: *const ::core::ffi::c_void,
    yDesc: cudnnTensorDescriptor_t,
    y: *mut ::core::ffi::c_void,
) -> cudnnStatus_t {
    let sym: Symbol<cudnnTransformTensor_fn> = unsafe {
        get_lib()
            .get(b"cudnnTransformTensor\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(handle, alpha, xDesc, x, beta, yDesc, y) }
}

type cudnnTransformTensorEx_fn = unsafe extern "C" fn(
    handle: cudnnHandle_t,
    transDesc: cudnnTensorTransformDescriptor_t,
    alpha: *const ::core::ffi::c_void,
    srcDesc: cudnnTensorDescriptor_t,
    srcData: *const ::core::ffi::c_void,
    beta: *const ::core::ffi::c_void,
    destDesc: cudnnTensorDescriptor_t,
    destData: *mut ::core::ffi::c_void,
) -> cudnnStatus_t;
pub unsafe fn cudnnTransformTensorEx(
    handle: cudnnHandle_t,
    transDesc: cudnnTensorTransformDescriptor_t,
    alpha: *const ::core::ffi::c_void,
    srcDesc: cudnnTensorDescriptor_t,
    srcData: *const ::core::ffi::c_void,
    beta: *const ::core::ffi::c_void,
    destDesc: cudnnTensorDescriptor_t,
    destData: *mut ::core::ffi::c_void,
) -> cudnnStatus_t {
    let sym: Symbol<cudnnTransformTensorEx_fn> = unsafe {
        get_lib()
            .get(b"cudnnTransformTensorEx\0")
            .expect("Missing symbol")
    };
    unsafe {
        (*sym)(
            handle, transDesc, alpha, srcDesc, srcData, beta, destDesc, destData,
        )
    }
}

type cudnnAddTensor_fn = unsafe extern "C" fn(
    handle: cudnnHandle_t,
    alpha: *const ::core::ffi::c_void,
    aDesc: cudnnTensorDescriptor_t,
    A: *const ::core::ffi::c_void,
    beta: *const ::core::ffi::c_void,
    cDesc: cudnnTensorDescriptor_t,
    C: *mut ::core::ffi::c_void,
) -> cudnnStatus_t;
pub unsafe fn cudnnAddTensor(
    handle: cudnnHandle_t,
    alpha: *const ::core::ffi::c_void,
    aDesc: cudnnTensorDescriptor_t,
    A: *const ::core::ffi::c_void,
    beta: *const ::core::ffi::c_void,
    cDesc: cudnnTensorDescriptor_t,
    C: *mut ::core::ffi::c_void,
) -> cudnnStatus_t {
    let sym: Symbol<cudnnAddTensor_fn> =
        unsafe { get_lib().get(b"cudnnAddTensor\0").expect("Missing symbol") };
    unsafe { (*sym)(handle, alpha, aDesc, A, beta, cDesc, C) }
}

type cudnnCreateOpTensorDescriptor_fn =
    unsafe extern "C" fn(opTensorDesc: *mut cudnnOpTensorDescriptor_t) -> cudnnStatus_t;
pub unsafe fn cudnnCreateOpTensorDescriptor(
    opTensorDesc: *mut cudnnOpTensorDescriptor_t,
) -> cudnnStatus_t {
    let sym: Symbol<cudnnCreateOpTensorDescriptor_fn> = unsafe {
        get_lib()
            .get(b"cudnnCreateOpTensorDescriptor\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(opTensorDesc) }
}

type cudnnSetOpTensorDescriptor_fn = unsafe extern "C" fn(
    opTensorDesc: cudnnOpTensorDescriptor_t,
    opTensorOp: cudnnOpTensorOp_t,
    opTensorCompType: cudnnDataType_t,
    opTensorNanOpt: cudnnNanPropagation_t,
) -> cudnnStatus_t;
pub unsafe fn cudnnSetOpTensorDescriptor(
    opTensorDesc: cudnnOpTensorDescriptor_t,
    opTensorOp: cudnnOpTensorOp_t,
    opTensorCompType: cudnnDataType_t,
    opTensorNanOpt: cudnnNanPropagation_t,
) -> cudnnStatus_t {
    let sym: Symbol<cudnnSetOpTensorDescriptor_fn> = unsafe {
        get_lib()
            .get(b"cudnnSetOpTensorDescriptor\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(opTensorDesc, opTensorOp, opTensorCompType, opTensorNanOpt) }
}

type cudnnGetOpTensorDescriptor_fn = unsafe extern "C" fn(
    opTensorDesc: cudnnOpTensorDescriptor_t,
    opTensorOp: *mut cudnnOpTensorOp_t,
    opTensorCompType: *mut cudnnDataType_t,
    opTensorNanOpt: *mut cudnnNanPropagation_t,
) -> cudnnStatus_t;
pub unsafe fn cudnnGetOpTensorDescriptor(
    opTensorDesc: cudnnOpTensorDescriptor_t,
    opTensorOp: *mut cudnnOpTensorOp_t,
    opTensorCompType: *mut cudnnDataType_t,
    opTensorNanOpt: *mut cudnnNanPropagation_t,
) -> cudnnStatus_t {
    let sym: Symbol<cudnnGetOpTensorDescriptor_fn> = unsafe {
        get_lib()
            .get(b"cudnnGetOpTensorDescriptor\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(opTensorDesc, opTensorOp, opTensorCompType, opTensorNanOpt) }
}

type cudnnDestroyOpTensorDescriptor_fn =
    unsafe extern "C" fn(opTensorDesc: cudnnOpTensorDescriptor_t) -> cudnnStatus_t;
pub unsafe fn cudnnDestroyOpTensorDescriptor(
    opTensorDesc: cudnnOpTensorDescriptor_t,
) -> cudnnStatus_t {
    let sym: Symbol<cudnnDestroyOpTensorDescriptor_fn> = unsafe {
        get_lib()
            .get(b"cudnnDestroyOpTensorDescriptor\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(opTensorDesc) }
}

type cudnnOpTensor_fn = unsafe extern "C" fn(
    handle: cudnnHandle_t,
    opTensorDesc: cudnnOpTensorDescriptor_t,
    alpha1: *const ::core::ffi::c_void,
    aDesc: cudnnTensorDescriptor_t,
    A: *const ::core::ffi::c_void,
    alpha2: *const ::core::ffi::c_void,
    bDesc: cudnnTensorDescriptor_t,
    B: *const ::core::ffi::c_void,
    beta: *const ::core::ffi::c_void,
    cDesc: cudnnTensorDescriptor_t,
    C: *mut ::core::ffi::c_void,
) -> cudnnStatus_t;
pub unsafe fn cudnnOpTensor(
    handle: cudnnHandle_t,
    opTensorDesc: cudnnOpTensorDescriptor_t,
    alpha1: *const ::core::ffi::c_void,
    aDesc: cudnnTensorDescriptor_t,
    A: *const ::core::ffi::c_void,
    alpha2: *const ::core::ffi::c_void,
    bDesc: cudnnTensorDescriptor_t,
    B: *const ::core::ffi::c_void,
    beta: *const ::core::ffi::c_void,
    cDesc: cudnnTensorDescriptor_t,
    C: *mut ::core::ffi::c_void,
) -> cudnnStatus_t {
    let sym: Symbol<cudnnOpTensor_fn> =
        unsafe { get_lib().get(b"cudnnOpTensor\0").expect("Missing symbol") };
    unsafe {
        (*sym)(
            handle,
            opTensorDesc,
            alpha1,
            aDesc,
            A,
            alpha2,
            bDesc,
            B,
            beta,
            cDesc,
            C,
        )
    }
}

type cudnnCreateReduceTensorDescriptor_fn =
    unsafe extern "C" fn(reduceTensorDesc: *mut cudnnReduceTensorDescriptor_t) -> cudnnStatus_t;
pub unsafe fn cudnnCreateReduceTensorDescriptor(
    reduceTensorDesc: *mut cudnnReduceTensorDescriptor_t,
) -> cudnnStatus_t {
    let sym: Symbol<cudnnCreateReduceTensorDescriptor_fn> = unsafe {
        get_lib()
            .get(b"cudnnCreateReduceTensorDescriptor\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(reduceTensorDesc) }
}

type cudnnSetReduceTensorDescriptor_fn = unsafe extern "C" fn(
    reduceTensorDesc: cudnnReduceTensorDescriptor_t,
    reduceTensorOp: cudnnReduceTensorOp_t,
    reduceTensorCompType: cudnnDataType_t,
    reduceTensorNanOpt: cudnnNanPropagation_t,
    reduceTensorIndices: cudnnReduceTensorIndices_t,
    reduceTensorIndicesType: cudnnIndicesType_t,
) -> cudnnStatus_t;
pub unsafe fn cudnnSetReduceTensorDescriptor(
    reduceTensorDesc: cudnnReduceTensorDescriptor_t,
    reduceTensorOp: cudnnReduceTensorOp_t,
    reduceTensorCompType: cudnnDataType_t,
    reduceTensorNanOpt: cudnnNanPropagation_t,
    reduceTensorIndices: cudnnReduceTensorIndices_t,
    reduceTensorIndicesType: cudnnIndicesType_t,
) -> cudnnStatus_t {
    let sym: Symbol<cudnnSetReduceTensorDescriptor_fn> = unsafe {
        get_lib()
            .get(b"cudnnSetReduceTensorDescriptor\0")
            .expect("Missing symbol")
    };
    unsafe {
        (*sym)(
            reduceTensorDesc,
            reduceTensorOp,
            reduceTensorCompType,
            reduceTensorNanOpt,
            reduceTensorIndices,
            reduceTensorIndicesType,
        )
    }
}

type cudnnGetReduceTensorDescriptor_fn = unsafe extern "C" fn(
    reduceTensorDesc: cudnnReduceTensorDescriptor_t,
    reduceTensorOp: *mut cudnnReduceTensorOp_t,
    reduceTensorCompType: *mut cudnnDataType_t,
    reduceTensorNanOpt: *mut cudnnNanPropagation_t,
    reduceTensorIndices: *mut cudnnReduceTensorIndices_t,
    reduceTensorIndicesType: *mut cudnnIndicesType_t,
) -> cudnnStatus_t;
pub unsafe fn cudnnGetReduceTensorDescriptor(
    reduceTensorDesc: cudnnReduceTensorDescriptor_t,
    reduceTensorOp: *mut cudnnReduceTensorOp_t,
    reduceTensorCompType: *mut cudnnDataType_t,
    reduceTensorNanOpt: *mut cudnnNanPropagation_t,
    reduceTensorIndices: *mut cudnnReduceTensorIndices_t,
    reduceTensorIndicesType: *mut cudnnIndicesType_t,
) -> cudnnStatus_t {
    let sym: Symbol<cudnnGetReduceTensorDescriptor_fn> = unsafe {
        get_lib()
            .get(b"cudnnGetReduceTensorDescriptor\0")
            .expect("Missing symbol")
    };
    unsafe {
        (*sym)(
            reduceTensorDesc,
            reduceTensorOp,
            reduceTensorCompType,
            reduceTensorNanOpt,
            reduceTensorIndices,
            reduceTensorIndicesType,
        )
    }
}

type cudnnDestroyReduceTensorDescriptor_fn =
    unsafe extern "C" fn(reduceTensorDesc: cudnnReduceTensorDescriptor_t) -> cudnnStatus_t;
pub unsafe fn cudnnDestroyReduceTensorDescriptor(
    reduceTensorDesc: cudnnReduceTensorDescriptor_t,
) -> cudnnStatus_t {
    let sym: Symbol<cudnnDestroyReduceTensorDescriptor_fn> = unsafe {
        get_lib()
            .get(b"cudnnDestroyReduceTensorDescriptor\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(reduceTensorDesc) }
}

type cudnnGetReductionIndicesSize_fn = unsafe extern "C" fn(
    handle: cudnnHandle_t,
    reduceTensorDesc: cudnnReduceTensorDescriptor_t,
    aDesc: cudnnTensorDescriptor_t,
    cDesc: cudnnTensorDescriptor_t,
    sizeInBytes: *mut usize,
) -> cudnnStatus_t;
pub unsafe fn cudnnGetReductionIndicesSize(
    handle: cudnnHandle_t,
    reduceTensorDesc: cudnnReduceTensorDescriptor_t,
    aDesc: cudnnTensorDescriptor_t,
    cDesc: cudnnTensorDescriptor_t,
    sizeInBytes: *mut usize,
) -> cudnnStatus_t {
    let sym: Symbol<cudnnGetReductionIndicesSize_fn> = unsafe {
        get_lib()
            .get(b"cudnnGetReductionIndicesSize\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(handle, reduceTensorDesc, aDesc, cDesc, sizeInBytes) }
}

type cudnnGetReductionWorkspaceSize_fn = unsafe extern "C" fn(
    handle: cudnnHandle_t,
    reduceTensorDesc: cudnnReduceTensorDescriptor_t,
    aDesc: cudnnTensorDescriptor_t,
    cDesc: cudnnTensorDescriptor_t,
    sizeInBytes: *mut usize,
) -> cudnnStatus_t;
pub unsafe fn cudnnGetReductionWorkspaceSize(
    handle: cudnnHandle_t,
    reduceTensorDesc: cudnnReduceTensorDescriptor_t,
    aDesc: cudnnTensorDescriptor_t,
    cDesc: cudnnTensorDescriptor_t,
    sizeInBytes: *mut usize,
) -> cudnnStatus_t {
    let sym: Symbol<cudnnGetReductionWorkspaceSize_fn> = unsafe {
        get_lib()
            .get(b"cudnnGetReductionWorkspaceSize\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(handle, reduceTensorDesc, aDesc, cDesc, sizeInBytes) }
}

type cudnnReduceTensor_fn = unsafe extern "C" fn(
    handle: cudnnHandle_t,
    reduceTensorDesc: cudnnReduceTensorDescriptor_t,
    indices: *mut ::core::ffi::c_void,
    indicesSizeInBytes: usize,
    workspace: *mut ::core::ffi::c_void,
    workspaceSizeInBytes: usize,
    alpha: *const ::core::ffi::c_void,
    aDesc: cudnnTensorDescriptor_t,
    A: *const ::core::ffi::c_void,
    beta: *const ::core::ffi::c_void,
    cDesc: cudnnTensorDescriptor_t,
    C: *mut ::core::ffi::c_void,
) -> cudnnStatus_t;
pub unsafe fn cudnnReduceTensor(
    handle: cudnnHandle_t,
    reduceTensorDesc: cudnnReduceTensorDescriptor_t,
    indices: *mut ::core::ffi::c_void,
    indicesSizeInBytes: usize,
    workspace: *mut ::core::ffi::c_void,
    workspaceSizeInBytes: usize,
    alpha: *const ::core::ffi::c_void,
    aDesc: cudnnTensorDescriptor_t,
    A: *const ::core::ffi::c_void,
    beta: *const ::core::ffi::c_void,
    cDesc: cudnnTensorDescriptor_t,
    C: *mut ::core::ffi::c_void,
) -> cudnnStatus_t {
    let sym: Symbol<cudnnReduceTensor_fn> = unsafe {
        get_lib()
            .get(b"cudnnReduceTensor\0")
            .expect("Missing symbol")
    };
    unsafe {
        (*sym)(
            handle,
            reduceTensorDesc,
            indices,
            indicesSizeInBytes,
            workspace,
            workspaceSizeInBytes,
            alpha,
            aDesc,
            A,
            beta,
            cDesc,
            C,
        )
    }
}

type cudnnSetTensor_fn = unsafe extern "C" fn(
    handle: cudnnHandle_t,
    yDesc: cudnnTensorDescriptor_t,
    y: *mut ::core::ffi::c_void,
    valuePtr: *const ::core::ffi::c_void,
) -> cudnnStatus_t;
pub unsafe fn cudnnSetTensor(
    handle: cudnnHandle_t,
    yDesc: cudnnTensorDescriptor_t,
    y: *mut ::core::ffi::c_void,
    valuePtr: *const ::core::ffi::c_void,
) -> cudnnStatus_t {
    let sym: Symbol<cudnnSetTensor_fn> =
        unsafe { get_lib().get(b"cudnnSetTensor\0").expect("Missing symbol") };
    unsafe { (*sym)(handle, yDesc, y, valuePtr) }
}

type cudnnScaleTensor_fn = unsafe extern "C" fn(
    handle: cudnnHandle_t,
    yDesc: cudnnTensorDescriptor_t,
    y: *mut ::core::ffi::c_void,
    alpha: *const ::core::ffi::c_void,
) -> cudnnStatus_t;
pub unsafe fn cudnnScaleTensor(
    handle: cudnnHandle_t,
    yDesc: cudnnTensorDescriptor_t,
    y: *mut ::core::ffi::c_void,
    alpha: *const ::core::ffi::c_void,
) -> cudnnStatus_t {
    let sym: Symbol<cudnnScaleTensor_fn> = unsafe {
        get_lib()
            .get(b"cudnnScaleTensor\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(handle, yDesc, y, alpha) }
}

type cudnnCreateFilterDescriptor_fn =
    unsafe extern "C" fn(filterDesc: *mut cudnnFilterDescriptor_t) -> cudnnStatus_t;
pub unsafe fn cudnnCreateFilterDescriptor(
    filterDesc: *mut cudnnFilterDescriptor_t,
) -> cudnnStatus_t {
    let sym: Symbol<cudnnCreateFilterDescriptor_fn> = unsafe {
        get_lib()
            .get(b"cudnnCreateFilterDescriptor\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(filterDesc) }
}

type cudnnSetFilter4dDescriptor_fn = unsafe extern "C" fn(
    filterDesc: cudnnFilterDescriptor_t,
    dataType: cudnnDataType_t,
    format: cudnnTensorFormat_t,
    k: ::core::ffi::c_int,
    c: ::core::ffi::c_int,
    h: ::core::ffi::c_int,
    w: ::core::ffi::c_int,
) -> cudnnStatus_t;
pub unsafe fn cudnnSetFilter4dDescriptor(
    filterDesc: cudnnFilterDescriptor_t,
    dataType: cudnnDataType_t,
    format: cudnnTensorFormat_t,
    k: ::core::ffi::c_int,
    c: ::core::ffi::c_int,
    h: ::core::ffi::c_int,
    w: ::core::ffi::c_int,
) -> cudnnStatus_t {
    let sym: Symbol<cudnnSetFilter4dDescriptor_fn> = unsafe {
        get_lib()
            .get(b"cudnnSetFilter4dDescriptor\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(filterDesc, dataType, format, k, c, h, w) }
}

type cudnnGetFilter4dDescriptor_fn = unsafe extern "C" fn(
    filterDesc: cudnnFilterDescriptor_t,
    dataType: *mut cudnnDataType_t,
    format: *mut cudnnTensorFormat_t,
    k: *mut ::core::ffi::c_int,
    c: *mut ::core::ffi::c_int,
    h: *mut ::core::ffi::c_int,
    w: *mut ::core::ffi::c_int,
) -> cudnnStatus_t;
pub unsafe fn cudnnGetFilter4dDescriptor(
    filterDesc: cudnnFilterDescriptor_t,
    dataType: *mut cudnnDataType_t,
    format: *mut cudnnTensorFormat_t,
    k: *mut ::core::ffi::c_int,
    c: *mut ::core::ffi::c_int,
    h: *mut ::core::ffi::c_int,
    w: *mut ::core::ffi::c_int,
) -> cudnnStatus_t {
    let sym: Symbol<cudnnGetFilter4dDescriptor_fn> = unsafe {
        get_lib()
            .get(b"cudnnGetFilter4dDescriptor\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(filterDesc, dataType, format, k, c, h, w) }
}

type cudnnSetFilterNdDescriptor_fn = unsafe extern "C" fn(
    filterDesc: cudnnFilterDescriptor_t,
    dataType: cudnnDataType_t,
    format: cudnnTensorFormat_t,
    nbDims: ::core::ffi::c_int,
    filterDimA: *const ::core::ffi::c_int,
) -> cudnnStatus_t;
pub unsafe fn cudnnSetFilterNdDescriptor(
    filterDesc: cudnnFilterDescriptor_t,
    dataType: cudnnDataType_t,
    format: cudnnTensorFormat_t,
    nbDims: ::core::ffi::c_int,
    filterDimA: *const ::core::ffi::c_int,
) -> cudnnStatus_t {
    let sym: Symbol<cudnnSetFilterNdDescriptor_fn> = unsafe {
        get_lib()
            .get(b"cudnnSetFilterNdDescriptor\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(filterDesc, dataType, format, nbDims, filterDimA) }
}

type cudnnGetFilterNdDescriptor_fn = unsafe extern "C" fn(
    filterDesc: cudnnFilterDescriptor_t,
    nbDimsRequested: ::core::ffi::c_int,
    dataType: *mut cudnnDataType_t,
    format: *mut cudnnTensorFormat_t,
    nbDims: *mut ::core::ffi::c_int,
    filterDimA: *mut ::core::ffi::c_int,
) -> cudnnStatus_t;
pub unsafe fn cudnnGetFilterNdDescriptor(
    filterDesc: cudnnFilterDescriptor_t,
    nbDimsRequested: ::core::ffi::c_int,
    dataType: *mut cudnnDataType_t,
    format: *mut cudnnTensorFormat_t,
    nbDims: *mut ::core::ffi::c_int,
    filterDimA: *mut ::core::ffi::c_int,
) -> cudnnStatus_t {
    let sym: Symbol<cudnnGetFilterNdDescriptor_fn> = unsafe {
        get_lib()
            .get(b"cudnnGetFilterNdDescriptor\0")
            .expect("Missing symbol")
    };
    unsafe {
        (*sym)(
            filterDesc,
            nbDimsRequested,
            dataType,
            format,
            nbDims,
            filterDimA,
        )
    }
}

type cudnnGetFilterSizeInBytes_fn =
    unsafe extern "C" fn(filterDesc: cudnnFilterDescriptor_t, size: *mut usize) -> cudnnStatus_t;
pub unsafe fn cudnnGetFilterSizeInBytes(
    filterDesc: cudnnFilterDescriptor_t,
    size: *mut usize,
) -> cudnnStatus_t {
    let sym: Symbol<cudnnGetFilterSizeInBytes_fn> = unsafe {
        get_lib()
            .get(b"cudnnGetFilterSizeInBytes\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(filterDesc, size) }
}

type cudnnTransformFilter_fn = unsafe extern "C" fn(
    handle: cudnnHandle_t,
    transDesc: cudnnTensorTransformDescriptor_t,
    alpha: *const ::core::ffi::c_void,
    srcDesc: cudnnFilterDescriptor_t,
    srcData: *const ::core::ffi::c_void,
    beta: *const ::core::ffi::c_void,
    destDesc: cudnnFilterDescriptor_t,
    destData: *mut ::core::ffi::c_void,
) -> cudnnStatus_t;
pub unsafe fn cudnnTransformFilter(
    handle: cudnnHandle_t,
    transDesc: cudnnTensorTransformDescriptor_t,
    alpha: *const ::core::ffi::c_void,
    srcDesc: cudnnFilterDescriptor_t,
    srcData: *const ::core::ffi::c_void,
    beta: *const ::core::ffi::c_void,
    destDesc: cudnnFilterDescriptor_t,
    destData: *mut ::core::ffi::c_void,
) -> cudnnStatus_t {
    let sym: Symbol<cudnnTransformFilter_fn> = unsafe {
        get_lib()
            .get(b"cudnnTransformFilter\0")
            .expect("Missing symbol")
    };
    unsafe {
        (*sym)(
            handle, transDesc, alpha, srcDesc, srcData, beta, destDesc, destData,
        )
    }
}

type cudnnDestroyFilterDescriptor_fn =
    unsafe extern "C" fn(filterDesc: cudnnFilterDescriptor_t) -> cudnnStatus_t;
pub unsafe fn cudnnDestroyFilterDescriptor(filterDesc: cudnnFilterDescriptor_t) -> cudnnStatus_t {
    let sym: Symbol<cudnnDestroyFilterDescriptor_fn> = unsafe {
        get_lib()
            .get(b"cudnnDestroyFilterDescriptor\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(filterDesc) }
}

type cudnnSoftmaxForward_fn = unsafe extern "C" fn(
    handle: cudnnHandle_t,
    algo: cudnnSoftmaxAlgorithm_t,
    mode: cudnnSoftmaxMode_t,
    alpha: *const ::core::ffi::c_void,
    xDesc: cudnnTensorDescriptor_t,
    x: *const ::core::ffi::c_void,
    beta: *const ::core::ffi::c_void,
    yDesc: cudnnTensorDescriptor_t,
    y: *mut ::core::ffi::c_void,
) -> cudnnStatus_t;
pub unsafe fn cudnnSoftmaxForward(
    handle: cudnnHandle_t,
    algo: cudnnSoftmaxAlgorithm_t,
    mode: cudnnSoftmaxMode_t,
    alpha: *const ::core::ffi::c_void,
    xDesc: cudnnTensorDescriptor_t,
    x: *const ::core::ffi::c_void,
    beta: *const ::core::ffi::c_void,
    yDesc: cudnnTensorDescriptor_t,
    y: *mut ::core::ffi::c_void,
) -> cudnnStatus_t {
    let sym: Symbol<cudnnSoftmaxForward_fn> = unsafe {
        get_lib()
            .get(b"cudnnSoftmaxForward\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(handle, algo, mode, alpha, xDesc, x, beta, yDesc, y) }
}

type cudnnCreatePoolingDescriptor_fn =
    unsafe extern "C" fn(poolingDesc: *mut cudnnPoolingDescriptor_t) -> cudnnStatus_t;
pub unsafe fn cudnnCreatePoolingDescriptor(
    poolingDesc: *mut cudnnPoolingDescriptor_t,
) -> cudnnStatus_t {
    let sym: Symbol<cudnnCreatePoolingDescriptor_fn> = unsafe {
        get_lib()
            .get(b"cudnnCreatePoolingDescriptor\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(poolingDesc) }
}

type cudnnSetPooling2dDescriptor_fn = unsafe extern "C" fn(
    poolingDesc: cudnnPoolingDescriptor_t,
    mode: cudnnPoolingMode_t,
    maxpoolingNanOpt: cudnnNanPropagation_t,
    windowHeight: ::core::ffi::c_int,
    windowWidth: ::core::ffi::c_int,
    verticalPadding: ::core::ffi::c_int,
    horizontalPadding: ::core::ffi::c_int,
    verticalStride: ::core::ffi::c_int,
    horizontalStride: ::core::ffi::c_int,
) -> cudnnStatus_t;
pub unsafe fn cudnnSetPooling2dDescriptor(
    poolingDesc: cudnnPoolingDescriptor_t,
    mode: cudnnPoolingMode_t,
    maxpoolingNanOpt: cudnnNanPropagation_t,
    windowHeight: ::core::ffi::c_int,
    windowWidth: ::core::ffi::c_int,
    verticalPadding: ::core::ffi::c_int,
    horizontalPadding: ::core::ffi::c_int,
    verticalStride: ::core::ffi::c_int,
    horizontalStride: ::core::ffi::c_int,
) -> cudnnStatus_t {
    let sym: Symbol<cudnnSetPooling2dDescriptor_fn> = unsafe {
        get_lib()
            .get(b"cudnnSetPooling2dDescriptor\0")
            .expect("Missing symbol")
    };
    unsafe {
        (*sym)(
            poolingDesc,
            mode,
            maxpoolingNanOpt,
            windowHeight,
            windowWidth,
            verticalPadding,
            horizontalPadding,
            verticalStride,
            horizontalStride,
        )
    }
}

type cudnnGetPooling2dDescriptor_fn = unsafe extern "C" fn(
    poolingDesc: cudnnPoolingDescriptor_t,
    mode: *mut cudnnPoolingMode_t,
    maxpoolingNanOpt: *mut cudnnNanPropagation_t,
    windowHeight: *mut ::core::ffi::c_int,
    windowWidth: *mut ::core::ffi::c_int,
    verticalPadding: *mut ::core::ffi::c_int,
    horizontalPadding: *mut ::core::ffi::c_int,
    verticalStride: *mut ::core::ffi::c_int,
    horizontalStride: *mut ::core::ffi::c_int,
) -> cudnnStatus_t;
pub unsafe fn cudnnGetPooling2dDescriptor(
    poolingDesc: cudnnPoolingDescriptor_t,
    mode: *mut cudnnPoolingMode_t,
    maxpoolingNanOpt: *mut cudnnNanPropagation_t,
    windowHeight: *mut ::core::ffi::c_int,
    windowWidth: *mut ::core::ffi::c_int,
    verticalPadding: *mut ::core::ffi::c_int,
    horizontalPadding: *mut ::core::ffi::c_int,
    verticalStride: *mut ::core::ffi::c_int,
    horizontalStride: *mut ::core::ffi::c_int,
) -> cudnnStatus_t {
    let sym: Symbol<cudnnGetPooling2dDescriptor_fn> = unsafe {
        get_lib()
            .get(b"cudnnGetPooling2dDescriptor\0")
            .expect("Missing symbol")
    };
    unsafe {
        (*sym)(
            poolingDesc,
            mode,
            maxpoolingNanOpt,
            windowHeight,
            windowWidth,
            verticalPadding,
            horizontalPadding,
            verticalStride,
            horizontalStride,
        )
    }
}

type cudnnSetPoolingNdDescriptor_fn = unsafe extern "C" fn(
    poolingDesc: cudnnPoolingDescriptor_t,
    mode: cudnnPoolingMode_t,
    maxpoolingNanOpt: cudnnNanPropagation_t,
    nbDims: ::core::ffi::c_int,
    windowDimA: *const ::core::ffi::c_int,
    paddingA: *const ::core::ffi::c_int,
    strideA: *const ::core::ffi::c_int,
) -> cudnnStatus_t;
pub unsafe fn cudnnSetPoolingNdDescriptor(
    poolingDesc: cudnnPoolingDescriptor_t,
    mode: cudnnPoolingMode_t,
    maxpoolingNanOpt: cudnnNanPropagation_t,
    nbDims: ::core::ffi::c_int,
    windowDimA: *const ::core::ffi::c_int,
    paddingA: *const ::core::ffi::c_int,
    strideA: *const ::core::ffi::c_int,
) -> cudnnStatus_t {
    let sym: Symbol<cudnnSetPoolingNdDescriptor_fn> = unsafe {
        get_lib()
            .get(b"cudnnSetPoolingNdDescriptor\0")
            .expect("Missing symbol")
    };
    unsafe {
        (*sym)(
            poolingDesc,
            mode,
            maxpoolingNanOpt,
            nbDims,
            windowDimA,
            paddingA,
            strideA,
        )
    }
}

type cudnnGetPoolingNdDescriptor_fn = unsafe extern "C" fn(
    poolingDesc: cudnnPoolingDescriptor_t,
    nbDimsRequested: ::core::ffi::c_int,
    mode: *mut cudnnPoolingMode_t,
    maxpoolingNanOpt: *mut cudnnNanPropagation_t,
    nbDims: *mut ::core::ffi::c_int,
    windowDimA: *mut ::core::ffi::c_int,
    paddingA: *mut ::core::ffi::c_int,
    strideA: *mut ::core::ffi::c_int,
) -> cudnnStatus_t;
pub unsafe fn cudnnGetPoolingNdDescriptor(
    poolingDesc: cudnnPoolingDescriptor_t,
    nbDimsRequested: ::core::ffi::c_int,
    mode: *mut cudnnPoolingMode_t,
    maxpoolingNanOpt: *mut cudnnNanPropagation_t,
    nbDims: *mut ::core::ffi::c_int,
    windowDimA: *mut ::core::ffi::c_int,
    paddingA: *mut ::core::ffi::c_int,
    strideA: *mut ::core::ffi::c_int,
) -> cudnnStatus_t {
    let sym: Symbol<cudnnGetPoolingNdDescriptor_fn> = unsafe {
        get_lib()
            .get(b"cudnnGetPoolingNdDescriptor\0")
            .expect("Missing symbol")
    };
    unsafe {
        (*sym)(
            poolingDesc,
            nbDimsRequested,
            mode,
            maxpoolingNanOpt,
            nbDims,
            windowDimA,
            paddingA,
            strideA,
        )
    }
}

type cudnnGetPoolingNdForwardOutputDim_fn = unsafe extern "C" fn(
    poolingDesc: cudnnPoolingDescriptor_t,
    inputTensorDesc: cudnnTensorDescriptor_t,
    nbDims: ::core::ffi::c_int,
    outputTensorDimA: *mut ::core::ffi::c_int,
) -> cudnnStatus_t;
pub unsafe fn cudnnGetPoolingNdForwardOutputDim(
    poolingDesc: cudnnPoolingDescriptor_t,
    inputTensorDesc: cudnnTensorDescriptor_t,
    nbDims: ::core::ffi::c_int,
    outputTensorDimA: *mut ::core::ffi::c_int,
) -> cudnnStatus_t {
    let sym: Symbol<cudnnGetPoolingNdForwardOutputDim_fn> = unsafe {
        get_lib()
            .get(b"cudnnGetPoolingNdForwardOutputDim\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(poolingDesc, inputTensorDesc, nbDims, outputTensorDimA) }
}

type cudnnGetPooling2dForwardOutputDim_fn = unsafe extern "C" fn(
    poolingDesc: cudnnPoolingDescriptor_t,
    inputTensorDesc: cudnnTensorDescriptor_t,
    n: *mut ::core::ffi::c_int,
    c: *mut ::core::ffi::c_int,
    h: *mut ::core::ffi::c_int,
    w: *mut ::core::ffi::c_int,
) -> cudnnStatus_t;
pub unsafe fn cudnnGetPooling2dForwardOutputDim(
    poolingDesc: cudnnPoolingDescriptor_t,
    inputTensorDesc: cudnnTensorDescriptor_t,
    n: *mut ::core::ffi::c_int,
    c: *mut ::core::ffi::c_int,
    h: *mut ::core::ffi::c_int,
    w: *mut ::core::ffi::c_int,
) -> cudnnStatus_t {
    let sym: Symbol<cudnnGetPooling2dForwardOutputDim_fn> = unsafe {
        get_lib()
            .get(b"cudnnGetPooling2dForwardOutputDim\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(poolingDesc, inputTensorDesc, n, c, h, w) }
}

type cudnnDestroyPoolingDescriptor_fn =
    unsafe extern "C" fn(poolingDesc: cudnnPoolingDescriptor_t) -> cudnnStatus_t;
pub unsafe fn cudnnDestroyPoolingDescriptor(
    poolingDesc: cudnnPoolingDescriptor_t,
) -> cudnnStatus_t {
    let sym: Symbol<cudnnDestroyPoolingDescriptor_fn> = unsafe {
        get_lib()
            .get(b"cudnnDestroyPoolingDescriptor\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(poolingDesc) }
}

type cudnnPoolingForward_fn = unsafe extern "C" fn(
    handle: cudnnHandle_t,
    poolingDesc: cudnnPoolingDescriptor_t,
    alpha: *const ::core::ffi::c_void,
    xDesc: cudnnTensorDescriptor_t,
    x: *const ::core::ffi::c_void,
    beta: *const ::core::ffi::c_void,
    yDesc: cudnnTensorDescriptor_t,
    y: *mut ::core::ffi::c_void,
) -> cudnnStatus_t;
pub unsafe fn cudnnPoolingForward(
    handle: cudnnHandle_t,
    poolingDesc: cudnnPoolingDescriptor_t,
    alpha: *const ::core::ffi::c_void,
    xDesc: cudnnTensorDescriptor_t,
    x: *const ::core::ffi::c_void,
    beta: *const ::core::ffi::c_void,
    yDesc: cudnnTensorDescriptor_t,
    y: *mut ::core::ffi::c_void,
) -> cudnnStatus_t {
    let sym: Symbol<cudnnPoolingForward_fn> = unsafe {
        get_lib()
            .get(b"cudnnPoolingForward\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(handle, poolingDesc, alpha, xDesc, x, beta, yDesc, y) }
}

type cudnnCreateActivationDescriptor_fn =
    unsafe extern "C" fn(activationDesc: *mut cudnnActivationDescriptor_t) -> cudnnStatus_t;
pub unsafe fn cudnnCreateActivationDescriptor(
    activationDesc: *mut cudnnActivationDescriptor_t,
) -> cudnnStatus_t {
    let sym: Symbol<cudnnCreateActivationDescriptor_fn> = unsafe {
        get_lib()
            .get(b"cudnnCreateActivationDescriptor\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(activationDesc) }
}

type cudnnSetActivationDescriptor_fn = unsafe extern "C" fn(
    activationDesc: cudnnActivationDescriptor_t,
    mode: cudnnActivationMode_t,
    reluNanOpt: cudnnNanPropagation_t,
    coef: f64,
) -> cudnnStatus_t;
pub unsafe fn cudnnSetActivationDescriptor(
    activationDesc: cudnnActivationDescriptor_t,
    mode: cudnnActivationMode_t,
    reluNanOpt: cudnnNanPropagation_t,
    coef: f64,
) -> cudnnStatus_t {
    let sym: Symbol<cudnnSetActivationDescriptor_fn> = unsafe {
        get_lib()
            .get(b"cudnnSetActivationDescriptor\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(activationDesc, mode, reluNanOpt, coef) }
}

type cudnnGetActivationDescriptor_fn = unsafe extern "C" fn(
    activationDesc: cudnnActivationDescriptor_t,
    mode: *mut cudnnActivationMode_t,
    reluNanOpt: *mut cudnnNanPropagation_t,
    coef: *mut f64,
) -> cudnnStatus_t;
pub unsafe fn cudnnGetActivationDescriptor(
    activationDesc: cudnnActivationDescriptor_t,
    mode: *mut cudnnActivationMode_t,
    reluNanOpt: *mut cudnnNanPropagation_t,
    coef: *mut f64,
) -> cudnnStatus_t {
    let sym: Symbol<cudnnGetActivationDescriptor_fn> = unsafe {
        get_lib()
            .get(b"cudnnGetActivationDescriptor\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(activationDesc, mode, reluNanOpt, coef) }
}

type cudnnSetActivationDescriptorSwishBeta_fn = unsafe extern "C" fn(
    activationDesc: cudnnActivationDescriptor_t,
    swish_beta: f64,
) -> cudnnStatus_t;
pub unsafe fn cudnnSetActivationDescriptorSwishBeta(
    activationDesc: cudnnActivationDescriptor_t,
    swish_beta: f64,
) -> cudnnStatus_t {
    let sym: Symbol<cudnnSetActivationDescriptorSwishBeta_fn> = unsafe {
        get_lib()
            .get(b"cudnnSetActivationDescriptorSwishBeta\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(activationDesc, swish_beta) }
}

type cudnnGetActivationDescriptorSwishBeta_fn = unsafe extern "C" fn(
    activationDesc: cudnnActivationDescriptor_t,
    swish_beta: *mut f64,
) -> cudnnStatus_t;
pub unsafe fn cudnnGetActivationDescriptorSwishBeta(
    activationDesc: cudnnActivationDescriptor_t,
    swish_beta: *mut f64,
) -> cudnnStatus_t {
    let sym: Symbol<cudnnGetActivationDescriptorSwishBeta_fn> = unsafe {
        get_lib()
            .get(b"cudnnGetActivationDescriptorSwishBeta\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(activationDesc, swish_beta) }
}

type cudnnDestroyActivationDescriptor_fn =
    unsafe extern "C" fn(activationDesc: cudnnActivationDescriptor_t) -> cudnnStatus_t;
pub unsafe fn cudnnDestroyActivationDescriptor(
    activationDesc: cudnnActivationDescriptor_t,
) -> cudnnStatus_t {
    let sym: Symbol<cudnnDestroyActivationDescriptor_fn> = unsafe {
        get_lib()
            .get(b"cudnnDestroyActivationDescriptor\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(activationDesc) }
}

type cudnnActivationForward_fn = unsafe extern "C" fn(
    handle: cudnnHandle_t,
    activationDesc: cudnnActivationDescriptor_t,
    alpha: *const ::core::ffi::c_void,
    xDesc: cudnnTensorDescriptor_t,
    x: *const ::core::ffi::c_void,
    beta: *const ::core::ffi::c_void,
    yDesc: cudnnTensorDescriptor_t,
    y: *mut ::core::ffi::c_void,
) -> cudnnStatus_t;
pub unsafe fn cudnnActivationForward(
    handle: cudnnHandle_t,
    activationDesc: cudnnActivationDescriptor_t,
    alpha: *const ::core::ffi::c_void,
    xDesc: cudnnTensorDescriptor_t,
    x: *const ::core::ffi::c_void,
    beta: *const ::core::ffi::c_void,
    yDesc: cudnnTensorDescriptor_t,
    y: *mut ::core::ffi::c_void,
) -> cudnnStatus_t {
    let sym: Symbol<cudnnActivationForward_fn> = unsafe {
        get_lib()
            .get(b"cudnnActivationForward\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(handle, activationDesc, alpha, xDesc, x, beta, yDesc, y) }
}

type cudnnCreateLRNDescriptor_fn =
    unsafe extern "C" fn(normDesc: *mut cudnnLRNDescriptor_t) -> cudnnStatus_t;
pub unsafe fn cudnnCreateLRNDescriptor(normDesc: *mut cudnnLRNDescriptor_t) -> cudnnStatus_t {
    let sym: Symbol<cudnnCreateLRNDescriptor_fn> = unsafe {
        get_lib()
            .get(b"cudnnCreateLRNDescriptor\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(normDesc) }
}

type cudnnSetLRNDescriptor_fn = unsafe extern "C" fn(
    normDesc: cudnnLRNDescriptor_t,
    lrnN: ::core::ffi::c_uint,
    lrnAlpha: f64,
    lrnBeta: f64,
    lrnK: f64,
) -> cudnnStatus_t;
pub unsafe fn cudnnSetLRNDescriptor(
    normDesc: cudnnLRNDescriptor_t,
    lrnN: ::core::ffi::c_uint,
    lrnAlpha: f64,
    lrnBeta: f64,
    lrnK: f64,
) -> cudnnStatus_t {
    let sym: Symbol<cudnnSetLRNDescriptor_fn> = unsafe {
        get_lib()
            .get(b"cudnnSetLRNDescriptor\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(normDesc, lrnN, lrnAlpha, lrnBeta, lrnK) }
}

type cudnnGetLRNDescriptor_fn = unsafe extern "C" fn(
    normDesc: cudnnLRNDescriptor_t,
    lrnN: *mut ::core::ffi::c_uint,
    lrnAlpha: *mut f64,
    lrnBeta: *mut f64,
    lrnK: *mut f64,
) -> cudnnStatus_t;
pub unsafe fn cudnnGetLRNDescriptor(
    normDesc: cudnnLRNDescriptor_t,
    lrnN: *mut ::core::ffi::c_uint,
    lrnAlpha: *mut f64,
    lrnBeta: *mut f64,
    lrnK: *mut f64,
) -> cudnnStatus_t {
    let sym: Symbol<cudnnGetLRNDescriptor_fn> = unsafe {
        get_lib()
            .get(b"cudnnGetLRNDescriptor\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(normDesc, lrnN, lrnAlpha, lrnBeta, lrnK) }
}

type cudnnDestroyLRNDescriptor_fn =
    unsafe extern "C" fn(lrnDesc: cudnnLRNDescriptor_t) -> cudnnStatus_t;
pub unsafe fn cudnnDestroyLRNDescriptor(lrnDesc: cudnnLRNDescriptor_t) -> cudnnStatus_t {
    let sym: Symbol<cudnnDestroyLRNDescriptor_fn> = unsafe {
        get_lib()
            .get(b"cudnnDestroyLRNDescriptor\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(lrnDesc) }
}

type cudnnLRNCrossChannelForward_fn = unsafe extern "C" fn(
    handle: cudnnHandle_t,
    normDesc: cudnnLRNDescriptor_t,
    lrnMode: cudnnLRNMode_t,
    alpha: *const ::core::ffi::c_void,
    xDesc: cudnnTensorDescriptor_t,
    x: *const ::core::ffi::c_void,
    beta: *const ::core::ffi::c_void,
    yDesc: cudnnTensorDescriptor_t,
    y: *mut ::core::ffi::c_void,
) -> cudnnStatus_t;
pub unsafe fn cudnnLRNCrossChannelForward(
    handle: cudnnHandle_t,
    normDesc: cudnnLRNDescriptor_t,
    lrnMode: cudnnLRNMode_t,
    alpha: *const ::core::ffi::c_void,
    xDesc: cudnnTensorDescriptor_t,
    x: *const ::core::ffi::c_void,
    beta: *const ::core::ffi::c_void,
    yDesc: cudnnTensorDescriptor_t,
    y: *mut ::core::ffi::c_void,
) -> cudnnStatus_t {
    let sym: Symbol<cudnnLRNCrossChannelForward_fn> = unsafe {
        get_lib()
            .get(b"cudnnLRNCrossChannelForward\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(handle, normDesc, lrnMode, alpha, xDesc, x, beta, yDesc, y) }
}

type cudnnDivisiveNormalizationForward_fn = unsafe extern "C" fn(
    handle: cudnnHandle_t,
    normDesc: cudnnLRNDescriptor_t,
    mode: cudnnDivNormMode_t,
    alpha: *const ::core::ffi::c_void,
    xDesc: cudnnTensorDescriptor_t,
    x: *const ::core::ffi::c_void,
    means: *const ::core::ffi::c_void,
    temp: *mut ::core::ffi::c_void,
    temp2: *mut ::core::ffi::c_void,
    beta: *const ::core::ffi::c_void,
    yDesc: cudnnTensorDescriptor_t,
    y: *mut ::core::ffi::c_void,
) -> cudnnStatus_t;
pub unsafe fn cudnnDivisiveNormalizationForward(
    handle: cudnnHandle_t,
    normDesc: cudnnLRNDescriptor_t,
    mode: cudnnDivNormMode_t,
    alpha: *const ::core::ffi::c_void,
    xDesc: cudnnTensorDescriptor_t,
    x: *const ::core::ffi::c_void,
    means: *const ::core::ffi::c_void,
    temp: *mut ::core::ffi::c_void,
    temp2: *mut ::core::ffi::c_void,
    beta: *const ::core::ffi::c_void,
    yDesc: cudnnTensorDescriptor_t,
    y: *mut ::core::ffi::c_void,
) -> cudnnStatus_t {
    let sym: Symbol<cudnnDivisiveNormalizationForward_fn> = unsafe {
        get_lib()
            .get(b"cudnnDivisiveNormalizationForward\0")
            .expect("Missing symbol")
    };
    unsafe {
        (*sym)(
            handle, normDesc, mode, alpha, xDesc, x, means, temp, temp2, beta, yDesc, y,
        )
    }
}

type cudnnDeriveBNTensorDescriptor_fn = unsafe extern "C" fn(
    derivedBnDesc: cudnnTensorDescriptor_t,
    xDesc: cudnnTensorDescriptor_t,
    mode: cudnnBatchNormMode_t,
) -> cudnnStatus_t;
pub unsafe fn cudnnDeriveBNTensorDescriptor(
    derivedBnDesc: cudnnTensorDescriptor_t,
    xDesc: cudnnTensorDescriptor_t,
    mode: cudnnBatchNormMode_t,
) -> cudnnStatus_t {
    let sym: Symbol<cudnnDeriveBNTensorDescriptor_fn> = unsafe {
        get_lib()
            .get(b"cudnnDeriveBNTensorDescriptor\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(derivedBnDesc, xDesc, mode) }
}

type cudnnBatchNormalizationForwardInference_fn = unsafe extern "C" fn(
    handle: cudnnHandle_t,
    mode: cudnnBatchNormMode_t,
    alpha: *const ::core::ffi::c_void,
    beta: *const ::core::ffi::c_void,
    xDesc: cudnnTensorDescriptor_t,
    x: *const ::core::ffi::c_void,
    yDesc: cudnnTensorDescriptor_t,
    y: *mut ::core::ffi::c_void,
    bnScaleBiasMeanVarDesc: cudnnTensorDescriptor_t,
    bnScale: *const ::core::ffi::c_void,
    bnBias: *const ::core::ffi::c_void,
    estimatedMean: *const ::core::ffi::c_void,
    estimatedVariance: *const ::core::ffi::c_void,
    epsilon: f64,
) -> cudnnStatus_t;
pub unsafe fn cudnnBatchNormalizationForwardInference(
    handle: cudnnHandle_t,
    mode: cudnnBatchNormMode_t,
    alpha: *const ::core::ffi::c_void,
    beta: *const ::core::ffi::c_void,
    xDesc: cudnnTensorDescriptor_t,
    x: *const ::core::ffi::c_void,
    yDesc: cudnnTensorDescriptor_t,
    y: *mut ::core::ffi::c_void,
    bnScaleBiasMeanVarDesc: cudnnTensorDescriptor_t,
    bnScale: *const ::core::ffi::c_void,
    bnBias: *const ::core::ffi::c_void,
    estimatedMean: *const ::core::ffi::c_void,
    estimatedVariance: *const ::core::ffi::c_void,
    epsilon: f64,
) -> cudnnStatus_t {
    let sym: Symbol<cudnnBatchNormalizationForwardInference_fn> = unsafe {
        get_lib()
            .get(b"cudnnBatchNormalizationForwardInference\0")
            .expect("Missing symbol")
    };
    unsafe {
        (*sym)(
            handle,
            mode,
            alpha,
            beta,
            xDesc,
            x,
            yDesc,
            y,
            bnScaleBiasMeanVarDesc,
            bnScale,
            bnBias,
            estimatedMean,
            estimatedVariance,
            epsilon,
        )
    }
}

type cudnnDeriveNormTensorDescriptor_fn = unsafe extern "C" fn(
    derivedNormScaleBiasDesc: cudnnTensorDescriptor_t,
    derivedNormMeanVarDesc: cudnnTensorDescriptor_t,
    xDesc: cudnnTensorDescriptor_t,
    mode: cudnnNormMode_t,
    groupCnt: ::core::ffi::c_int,
) -> cudnnStatus_t;
pub unsafe fn cudnnDeriveNormTensorDescriptor(
    derivedNormScaleBiasDesc: cudnnTensorDescriptor_t,
    derivedNormMeanVarDesc: cudnnTensorDescriptor_t,
    xDesc: cudnnTensorDescriptor_t,
    mode: cudnnNormMode_t,
    groupCnt: ::core::ffi::c_int,
) -> cudnnStatus_t {
    let sym: Symbol<cudnnDeriveNormTensorDescriptor_fn> = unsafe {
        get_lib()
            .get(b"cudnnDeriveNormTensorDescriptor\0")
            .expect("Missing symbol")
    };
    unsafe {
        (*sym)(
            derivedNormScaleBiasDesc,
            derivedNormMeanVarDesc,
            xDesc,
            mode,
            groupCnt,
        )
    }
}

type cudnnNormalizationForwardInference_fn = unsafe extern "C" fn(
    handle: cudnnHandle_t,
    mode: cudnnNormMode_t,
    normOps: cudnnNormOps_t,
    algo: cudnnNormAlgo_t,
    alpha: *const ::core::ffi::c_void,
    beta: *const ::core::ffi::c_void,
    xDesc: cudnnTensorDescriptor_t,
    x: *const ::core::ffi::c_void,
    normScaleBiasDesc: cudnnTensorDescriptor_t,
    normScale: *const ::core::ffi::c_void,
    normBias: *const ::core::ffi::c_void,
    normMeanVarDesc: cudnnTensorDescriptor_t,
    estimatedMean: *const ::core::ffi::c_void,
    estimatedVariance: *const ::core::ffi::c_void,
    zDesc: cudnnTensorDescriptor_t,
    z: *const ::core::ffi::c_void,
    activationDesc: cudnnActivationDescriptor_t,
    yDesc: cudnnTensorDescriptor_t,
    y: *mut ::core::ffi::c_void,
    epsilon: f64,
    groupCnt: ::core::ffi::c_int,
) -> cudnnStatus_t;
pub unsafe fn cudnnNormalizationForwardInference(
    handle: cudnnHandle_t,
    mode: cudnnNormMode_t,
    normOps: cudnnNormOps_t,
    algo: cudnnNormAlgo_t,
    alpha: *const ::core::ffi::c_void,
    beta: *const ::core::ffi::c_void,
    xDesc: cudnnTensorDescriptor_t,
    x: *const ::core::ffi::c_void,
    normScaleBiasDesc: cudnnTensorDescriptor_t,
    normScale: *const ::core::ffi::c_void,
    normBias: *const ::core::ffi::c_void,
    normMeanVarDesc: cudnnTensorDescriptor_t,
    estimatedMean: *const ::core::ffi::c_void,
    estimatedVariance: *const ::core::ffi::c_void,
    zDesc: cudnnTensorDescriptor_t,
    z: *const ::core::ffi::c_void,
    activationDesc: cudnnActivationDescriptor_t,
    yDesc: cudnnTensorDescriptor_t,
    y: *mut ::core::ffi::c_void,
    epsilon: f64,
    groupCnt: ::core::ffi::c_int,
) -> cudnnStatus_t {
    let sym: Symbol<cudnnNormalizationForwardInference_fn> = unsafe {
        get_lib()
            .get(b"cudnnNormalizationForwardInference\0")
            .expect("Missing symbol")
    };
    unsafe {
        (*sym)(
            handle,
            mode,
            normOps,
            algo,
            alpha,
            beta,
            xDesc,
            x,
            normScaleBiasDesc,
            normScale,
            normBias,
            normMeanVarDesc,
            estimatedMean,
            estimatedVariance,
            zDesc,
            z,
            activationDesc,
            yDesc,
            y,
            epsilon,
            groupCnt,
        )
    }
}

type cudnnCreateSpatialTransformerDescriptor_fn =
    unsafe extern "C" fn(stDesc: *mut cudnnSpatialTransformerDescriptor_t) -> cudnnStatus_t;
pub unsafe fn cudnnCreateSpatialTransformerDescriptor(
    stDesc: *mut cudnnSpatialTransformerDescriptor_t,
) -> cudnnStatus_t {
    let sym: Symbol<cudnnCreateSpatialTransformerDescriptor_fn> = unsafe {
        get_lib()
            .get(b"cudnnCreateSpatialTransformerDescriptor\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(stDesc) }
}

type cudnnSetSpatialTransformerNdDescriptor_fn = unsafe extern "C" fn(
    stDesc: cudnnSpatialTransformerDescriptor_t,
    samplerType: cudnnSamplerType_t,
    dataType: cudnnDataType_t,
    nbDims: ::core::ffi::c_int,
    dimA: *const ::core::ffi::c_int,
) -> cudnnStatus_t;
pub unsafe fn cudnnSetSpatialTransformerNdDescriptor(
    stDesc: cudnnSpatialTransformerDescriptor_t,
    samplerType: cudnnSamplerType_t,
    dataType: cudnnDataType_t,
    nbDims: ::core::ffi::c_int,
    dimA: *const ::core::ffi::c_int,
) -> cudnnStatus_t {
    let sym: Symbol<cudnnSetSpatialTransformerNdDescriptor_fn> = unsafe {
        get_lib()
            .get(b"cudnnSetSpatialTransformerNdDescriptor\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(stDesc, samplerType, dataType, nbDims, dimA) }
}

type cudnnDestroySpatialTransformerDescriptor_fn =
    unsafe extern "C" fn(stDesc: cudnnSpatialTransformerDescriptor_t) -> cudnnStatus_t;
pub unsafe fn cudnnDestroySpatialTransformerDescriptor(
    stDesc: cudnnSpatialTransformerDescriptor_t,
) -> cudnnStatus_t {
    let sym: Symbol<cudnnDestroySpatialTransformerDescriptor_fn> = unsafe {
        get_lib()
            .get(b"cudnnDestroySpatialTransformerDescriptor\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(stDesc) }
}

type cudnnSpatialTfGridGeneratorForward_fn = unsafe extern "C" fn(
    handle: cudnnHandle_t,
    stDesc: cudnnSpatialTransformerDescriptor_t,
    theta: *const ::core::ffi::c_void,
    grid: *mut ::core::ffi::c_void,
) -> cudnnStatus_t;
pub unsafe fn cudnnSpatialTfGridGeneratorForward(
    handle: cudnnHandle_t,
    stDesc: cudnnSpatialTransformerDescriptor_t,
    theta: *const ::core::ffi::c_void,
    grid: *mut ::core::ffi::c_void,
) -> cudnnStatus_t {
    let sym: Symbol<cudnnSpatialTfGridGeneratorForward_fn> = unsafe {
        get_lib()
            .get(b"cudnnSpatialTfGridGeneratorForward\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(handle, stDesc, theta, grid) }
}

type cudnnSpatialTfSamplerForward_fn = unsafe extern "C" fn(
    handle: cudnnHandle_t,
    stDesc: cudnnSpatialTransformerDescriptor_t,
    alpha: *const ::core::ffi::c_void,
    xDesc: cudnnTensorDescriptor_t,
    x: *const ::core::ffi::c_void,
    grid: *const ::core::ffi::c_void,
    beta: *const ::core::ffi::c_void,
    yDesc: cudnnTensorDescriptor_t,
    y: *mut ::core::ffi::c_void,
) -> cudnnStatus_t;
pub unsafe fn cudnnSpatialTfSamplerForward(
    handle: cudnnHandle_t,
    stDesc: cudnnSpatialTransformerDescriptor_t,
    alpha: *const ::core::ffi::c_void,
    xDesc: cudnnTensorDescriptor_t,
    x: *const ::core::ffi::c_void,
    grid: *const ::core::ffi::c_void,
    beta: *const ::core::ffi::c_void,
    yDesc: cudnnTensorDescriptor_t,
    y: *mut ::core::ffi::c_void,
) -> cudnnStatus_t {
    let sym: Symbol<cudnnSpatialTfSamplerForward_fn> = unsafe {
        get_lib()
            .get(b"cudnnSpatialTfSamplerForward\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(handle, stDesc, alpha, xDesc, x, grid, beta, yDesc, y) }
}

type cudnnCreateDropoutDescriptor_fn =
    unsafe extern "C" fn(dropoutDesc: *mut cudnnDropoutDescriptor_t) -> cudnnStatus_t;
pub unsafe fn cudnnCreateDropoutDescriptor(
    dropoutDesc: *mut cudnnDropoutDescriptor_t,
) -> cudnnStatus_t {
    let sym: Symbol<cudnnCreateDropoutDescriptor_fn> = unsafe {
        get_lib()
            .get(b"cudnnCreateDropoutDescriptor\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(dropoutDesc) }
}

type cudnnDestroyDropoutDescriptor_fn =
    unsafe extern "C" fn(dropoutDesc: cudnnDropoutDescriptor_t) -> cudnnStatus_t;
pub unsafe fn cudnnDestroyDropoutDescriptor(
    dropoutDesc: cudnnDropoutDescriptor_t,
) -> cudnnStatus_t {
    let sym: Symbol<cudnnDestroyDropoutDescriptor_fn> = unsafe {
        get_lib()
            .get(b"cudnnDestroyDropoutDescriptor\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(dropoutDesc) }
}

type cudnnDropoutGetStatesSize_fn =
    unsafe extern "C" fn(handle: cudnnHandle_t, sizeInBytes: *mut usize) -> cudnnStatus_t;
pub unsafe fn cudnnDropoutGetStatesSize(
    handle: cudnnHandle_t,
    sizeInBytes: *mut usize,
) -> cudnnStatus_t {
    let sym: Symbol<cudnnDropoutGetStatesSize_fn> = unsafe {
        get_lib()
            .get(b"cudnnDropoutGetStatesSize\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(handle, sizeInBytes) }
}

type cudnnDropoutGetReserveSpaceSize_fn =
    unsafe extern "C" fn(xdesc: cudnnTensorDescriptor_t, sizeInBytes: *mut usize) -> cudnnStatus_t;
pub unsafe fn cudnnDropoutGetReserveSpaceSize(
    xdesc: cudnnTensorDescriptor_t,
    sizeInBytes: *mut usize,
) -> cudnnStatus_t {
    let sym: Symbol<cudnnDropoutGetReserveSpaceSize_fn> = unsafe {
        get_lib()
            .get(b"cudnnDropoutGetReserveSpaceSize\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(xdesc, sizeInBytes) }
}

type cudnnSetDropoutDescriptor_fn = unsafe extern "C" fn(
    dropoutDesc: cudnnDropoutDescriptor_t,
    handle: cudnnHandle_t,
    dropout: f32,
    states: *mut ::core::ffi::c_void,
    stateSizeInBytes: usize,
    seed: ::core::ffi::c_ulonglong,
) -> cudnnStatus_t;
pub unsafe fn cudnnSetDropoutDescriptor(
    dropoutDesc: cudnnDropoutDescriptor_t,
    handle: cudnnHandle_t,
    dropout: f32,
    states: *mut ::core::ffi::c_void,
    stateSizeInBytes: usize,
    seed: ::core::ffi::c_ulonglong,
) -> cudnnStatus_t {
    let sym: Symbol<cudnnSetDropoutDescriptor_fn> = unsafe {
        get_lib()
            .get(b"cudnnSetDropoutDescriptor\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(dropoutDesc, handle, dropout, states, stateSizeInBytes, seed) }
}

type cudnnRestoreDropoutDescriptor_fn = unsafe extern "C" fn(
    dropoutDesc: cudnnDropoutDescriptor_t,
    handle: cudnnHandle_t,
    dropout: f32,
    states: *mut ::core::ffi::c_void,
    stateSizeInBytes: usize,
    seed: ::core::ffi::c_ulonglong,
) -> cudnnStatus_t;
pub unsafe fn cudnnRestoreDropoutDescriptor(
    dropoutDesc: cudnnDropoutDescriptor_t,
    handle: cudnnHandle_t,
    dropout: f32,
    states: *mut ::core::ffi::c_void,
    stateSizeInBytes: usize,
    seed: ::core::ffi::c_ulonglong,
) -> cudnnStatus_t {
    let sym: Symbol<cudnnRestoreDropoutDescriptor_fn> = unsafe {
        get_lib()
            .get(b"cudnnRestoreDropoutDescriptor\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(dropoutDesc, handle, dropout, states, stateSizeInBytes, seed) }
}

type cudnnGetDropoutDescriptor_fn = unsafe extern "C" fn(
    dropoutDesc: cudnnDropoutDescriptor_t,
    handle: cudnnHandle_t,
    dropout: *mut f32,
    states: *mut *mut ::core::ffi::c_void,
    seed: *mut ::core::ffi::c_ulonglong,
) -> cudnnStatus_t;
pub unsafe fn cudnnGetDropoutDescriptor(
    dropoutDesc: cudnnDropoutDescriptor_t,
    handle: cudnnHandle_t,
    dropout: *mut f32,
    states: *mut *mut ::core::ffi::c_void,
    seed: *mut ::core::ffi::c_ulonglong,
) -> cudnnStatus_t {
    let sym: Symbol<cudnnGetDropoutDescriptor_fn> = unsafe {
        get_lib()
            .get(b"cudnnGetDropoutDescriptor\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(dropoutDesc, handle, dropout, states, seed) }
}

type cudnnDropoutForward_fn = unsafe extern "C" fn(
    handle: cudnnHandle_t,
    dropoutDesc: cudnnDropoutDescriptor_t,
    xdesc: cudnnTensorDescriptor_t,
    x: *const ::core::ffi::c_void,
    ydesc: cudnnTensorDescriptor_t,
    y: *mut ::core::ffi::c_void,
    reserveSpace: *mut ::core::ffi::c_void,
    reserveSpaceSizeInBytes: usize,
) -> cudnnStatus_t;
pub unsafe fn cudnnDropoutForward(
    handle: cudnnHandle_t,
    dropoutDesc: cudnnDropoutDescriptor_t,
    xdesc: cudnnTensorDescriptor_t,
    x: *const ::core::ffi::c_void,
    ydesc: cudnnTensorDescriptor_t,
    y: *mut ::core::ffi::c_void,
    reserveSpace: *mut ::core::ffi::c_void,
    reserveSpaceSizeInBytes: usize,
) -> cudnnStatus_t {
    let sym: Symbol<cudnnDropoutForward_fn> = unsafe {
        get_lib()
            .get(b"cudnnDropoutForward\0")
            .expect("Missing symbol")
    };
    unsafe {
        (*sym)(
            handle,
            dropoutDesc,
            xdesc,
            x,
            ydesc,
            y,
            reserveSpace,
            reserveSpaceSizeInBytes,
        )
    }
}

type cudnnOpsVersionCheck_fn = unsafe extern "C" fn() -> cudnnStatus_t;
pub unsafe fn cudnnOpsVersionCheck() -> cudnnStatus_t {
    let sym: Symbol<cudnnOpsVersionCheck_fn> = unsafe {
        get_lib()
            .get(b"cudnnOpsVersionCheck\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)() }
}

type cudnnSoftmaxBackward_fn = unsafe extern "C" fn(
    handle: cudnnHandle_t,
    algo: cudnnSoftmaxAlgorithm_t,
    mode: cudnnSoftmaxMode_t,
    alpha: *const ::core::ffi::c_void,
    yDesc: cudnnTensorDescriptor_t,
    y: *const ::core::ffi::c_void,
    dyDesc: cudnnTensorDescriptor_t,
    dy: *const ::core::ffi::c_void,
    beta: *const ::core::ffi::c_void,
    dxDesc: cudnnTensorDescriptor_t,
    dx: *mut ::core::ffi::c_void,
) -> cudnnStatus_t;
pub unsafe fn cudnnSoftmaxBackward(
    handle: cudnnHandle_t,
    algo: cudnnSoftmaxAlgorithm_t,
    mode: cudnnSoftmaxMode_t,
    alpha: *const ::core::ffi::c_void,
    yDesc: cudnnTensorDescriptor_t,
    y: *const ::core::ffi::c_void,
    dyDesc: cudnnTensorDescriptor_t,
    dy: *const ::core::ffi::c_void,
    beta: *const ::core::ffi::c_void,
    dxDesc: cudnnTensorDescriptor_t,
    dx: *mut ::core::ffi::c_void,
) -> cudnnStatus_t {
    let sym: Symbol<cudnnSoftmaxBackward_fn> = unsafe {
        get_lib()
            .get(b"cudnnSoftmaxBackward\0")
            .expect("Missing symbol")
    };
    unsafe {
        (*sym)(
            handle, algo, mode, alpha, yDesc, y, dyDesc, dy, beta, dxDesc, dx,
        )
    }
}

type cudnnPoolingBackward_fn = unsafe extern "C" fn(
    handle: cudnnHandle_t,
    poolingDesc: cudnnPoolingDescriptor_t,
    alpha: *const ::core::ffi::c_void,
    yDesc: cudnnTensorDescriptor_t,
    y: *const ::core::ffi::c_void,
    dyDesc: cudnnTensorDescriptor_t,
    dy: *const ::core::ffi::c_void,
    xDesc: cudnnTensorDescriptor_t,
    x: *const ::core::ffi::c_void,
    beta: *const ::core::ffi::c_void,
    dxDesc: cudnnTensorDescriptor_t,
    dx: *mut ::core::ffi::c_void,
) -> cudnnStatus_t;
pub unsafe fn cudnnPoolingBackward(
    handle: cudnnHandle_t,
    poolingDesc: cudnnPoolingDescriptor_t,
    alpha: *const ::core::ffi::c_void,
    yDesc: cudnnTensorDescriptor_t,
    y: *const ::core::ffi::c_void,
    dyDesc: cudnnTensorDescriptor_t,
    dy: *const ::core::ffi::c_void,
    xDesc: cudnnTensorDescriptor_t,
    x: *const ::core::ffi::c_void,
    beta: *const ::core::ffi::c_void,
    dxDesc: cudnnTensorDescriptor_t,
    dx: *mut ::core::ffi::c_void,
) -> cudnnStatus_t {
    let sym: Symbol<cudnnPoolingBackward_fn> = unsafe {
        get_lib()
            .get(b"cudnnPoolingBackward\0")
            .expect("Missing symbol")
    };
    unsafe {
        (*sym)(
            handle,
            poolingDesc,
            alpha,
            yDesc,
            y,
            dyDesc,
            dy,
            xDesc,
            x,
            beta,
            dxDesc,
            dx,
        )
    }
}

type cudnnActivationBackward_fn = unsafe extern "C" fn(
    handle: cudnnHandle_t,
    activationDesc: cudnnActivationDescriptor_t,
    alpha: *const ::core::ffi::c_void,
    yDesc: cudnnTensorDescriptor_t,
    y: *const ::core::ffi::c_void,
    dyDesc: cudnnTensorDescriptor_t,
    dy: *const ::core::ffi::c_void,
    xDesc: cudnnTensorDescriptor_t,
    x: *const ::core::ffi::c_void,
    beta: *const ::core::ffi::c_void,
    dxDesc: cudnnTensorDescriptor_t,
    dx: *mut ::core::ffi::c_void,
) -> cudnnStatus_t;
pub unsafe fn cudnnActivationBackward(
    handle: cudnnHandle_t,
    activationDesc: cudnnActivationDescriptor_t,
    alpha: *const ::core::ffi::c_void,
    yDesc: cudnnTensorDescriptor_t,
    y: *const ::core::ffi::c_void,
    dyDesc: cudnnTensorDescriptor_t,
    dy: *const ::core::ffi::c_void,
    xDesc: cudnnTensorDescriptor_t,
    x: *const ::core::ffi::c_void,
    beta: *const ::core::ffi::c_void,
    dxDesc: cudnnTensorDescriptor_t,
    dx: *mut ::core::ffi::c_void,
) -> cudnnStatus_t {
    let sym: Symbol<cudnnActivationBackward_fn> = unsafe {
        get_lib()
            .get(b"cudnnActivationBackward\0")
            .expect("Missing symbol")
    };
    unsafe {
        (*sym)(
            handle,
            activationDesc,
            alpha,
            yDesc,
            y,
            dyDesc,
            dy,
            xDesc,
            x,
            beta,
            dxDesc,
            dx,
        )
    }
}

type cudnnLRNCrossChannelBackward_fn = unsafe extern "C" fn(
    handle: cudnnHandle_t,
    normDesc: cudnnLRNDescriptor_t,
    lrnMode: cudnnLRNMode_t,
    alpha: *const ::core::ffi::c_void,
    yDesc: cudnnTensorDescriptor_t,
    y: *const ::core::ffi::c_void,
    dyDesc: cudnnTensorDescriptor_t,
    dy: *const ::core::ffi::c_void,
    xDesc: cudnnTensorDescriptor_t,
    x: *const ::core::ffi::c_void,
    beta: *const ::core::ffi::c_void,
    dxDesc: cudnnTensorDescriptor_t,
    dx: *mut ::core::ffi::c_void,
) -> cudnnStatus_t;
pub unsafe fn cudnnLRNCrossChannelBackward(
    handle: cudnnHandle_t,
    normDesc: cudnnLRNDescriptor_t,
    lrnMode: cudnnLRNMode_t,
    alpha: *const ::core::ffi::c_void,
    yDesc: cudnnTensorDescriptor_t,
    y: *const ::core::ffi::c_void,
    dyDesc: cudnnTensorDescriptor_t,
    dy: *const ::core::ffi::c_void,
    xDesc: cudnnTensorDescriptor_t,
    x: *const ::core::ffi::c_void,
    beta: *const ::core::ffi::c_void,
    dxDesc: cudnnTensorDescriptor_t,
    dx: *mut ::core::ffi::c_void,
) -> cudnnStatus_t {
    let sym: Symbol<cudnnLRNCrossChannelBackward_fn> = unsafe {
        get_lib()
            .get(b"cudnnLRNCrossChannelBackward\0")
            .expect("Missing symbol")
    };
    unsafe {
        (*sym)(
            handle, normDesc, lrnMode, alpha, yDesc, y, dyDesc, dy, xDesc, x, beta, dxDesc, dx,
        )
    }
}

type cudnnDivisiveNormalizationBackward_fn = unsafe extern "C" fn(
    handle: cudnnHandle_t,
    normDesc: cudnnLRNDescriptor_t,
    mode: cudnnDivNormMode_t,
    alpha: *const ::core::ffi::c_void,
    xDesc: cudnnTensorDescriptor_t,
    x: *const ::core::ffi::c_void,
    means: *const ::core::ffi::c_void,
    dy: *const ::core::ffi::c_void,
    temp: *mut ::core::ffi::c_void,
    temp2: *mut ::core::ffi::c_void,
    beta: *const ::core::ffi::c_void,
    dXdMeansDesc: cudnnTensorDescriptor_t,
    dx: *mut ::core::ffi::c_void,
    dMeans: *mut ::core::ffi::c_void,
) -> cudnnStatus_t;
pub unsafe fn cudnnDivisiveNormalizationBackward(
    handle: cudnnHandle_t,
    normDesc: cudnnLRNDescriptor_t,
    mode: cudnnDivNormMode_t,
    alpha: *const ::core::ffi::c_void,
    xDesc: cudnnTensorDescriptor_t,
    x: *const ::core::ffi::c_void,
    means: *const ::core::ffi::c_void,
    dy: *const ::core::ffi::c_void,
    temp: *mut ::core::ffi::c_void,
    temp2: *mut ::core::ffi::c_void,
    beta: *const ::core::ffi::c_void,
    dXdMeansDesc: cudnnTensorDescriptor_t,
    dx: *mut ::core::ffi::c_void,
    dMeans: *mut ::core::ffi::c_void,
) -> cudnnStatus_t {
    let sym: Symbol<cudnnDivisiveNormalizationBackward_fn> = unsafe {
        get_lib()
            .get(b"cudnnDivisiveNormalizationBackward\0")
            .expect("Missing symbol")
    };
    unsafe {
        (*sym)(
            handle,
            normDesc,
            mode,
            alpha,
            xDesc,
            x,
            means,
            dy,
            temp,
            temp2,
            beta,
            dXdMeansDesc,
            dx,
            dMeans,
        )
    }
}

type cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize_fn =
    unsafe extern "C" fn(
        handle: cudnnHandle_t,
        mode: cudnnBatchNormMode_t,
        bnOps: cudnnBatchNormOps_t,
        xDesc: cudnnTensorDescriptor_t,
        zDesc: cudnnTensorDescriptor_t,
        yDesc: cudnnTensorDescriptor_t,
        bnScaleBiasMeanVarDesc: cudnnTensorDescriptor_t,
        activationDesc: cudnnActivationDescriptor_t,
        sizeInBytes: *mut usize,
    ) -> cudnnStatus_t;
pub unsafe fn cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize(
    handle: cudnnHandle_t,
    mode: cudnnBatchNormMode_t,
    bnOps: cudnnBatchNormOps_t,
    xDesc: cudnnTensorDescriptor_t,
    zDesc: cudnnTensorDescriptor_t,
    yDesc: cudnnTensorDescriptor_t,
    bnScaleBiasMeanVarDesc: cudnnTensorDescriptor_t,
    activationDesc: cudnnActivationDescriptor_t,
    sizeInBytes: *mut usize,
) -> cudnnStatus_t {
    let sym: Symbol<cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize_fn> = unsafe {
        get_lib()
            .get(b"cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize\0")
            .expect("Missing symbol")
    };
    unsafe {
        (*sym)(
            handle,
            mode,
            bnOps,
            xDesc,
            zDesc,
            yDesc,
            bnScaleBiasMeanVarDesc,
            activationDesc,
            sizeInBytes,
        )
    }
}

type cudnnGetBatchNormalizationBackwardExWorkspaceSize_fn = unsafe extern "C" fn(
    handle: cudnnHandle_t,
    mode: cudnnBatchNormMode_t,
    bnOps: cudnnBatchNormOps_t,
    xDesc: cudnnTensorDescriptor_t,
    yDesc: cudnnTensorDescriptor_t,
    dyDesc: cudnnTensorDescriptor_t,
    dzDesc: cudnnTensorDescriptor_t,
    dxDesc: cudnnTensorDescriptor_t,
    dBnScaleBiasDesc: cudnnTensorDescriptor_t,
    activationDesc: cudnnActivationDescriptor_t,
    sizeInBytes: *mut usize,
) -> cudnnStatus_t;
pub unsafe fn cudnnGetBatchNormalizationBackwardExWorkspaceSize(
    handle: cudnnHandle_t,
    mode: cudnnBatchNormMode_t,
    bnOps: cudnnBatchNormOps_t,
    xDesc: cudnnTensorDescriptor_t,
    yDesc: cudnnTensorDescriptor_t,
    dyDesc: cudnnTensorDescriptor_t,
    dzDesc: cudnnTensorDescriptor_t,
    dxDesc: cudnnTensorDescriptor_t,
    dBnScaleBiasDesc: cudnnTensorDescriptor_t,
    activationDesc: cudnnActivationDescriptor_t,
    sizeInBytes: *mut usize,
) -> cudnnStatus_t {
    let sym: Symbol<cudnnGetBatchNormalizationBackwardExWorkspaceSize_fn> = unsafe {
        get_lib()
            .get(b"cudnnGetBatchNormalizationBackwardExWorkspaceSize\0")
            .expect("Missing symbol")
    };
    unsafe {
        (*sym)(
            handle,
            mode,
            bnOps,
            xDesc,
            yDesc,
            dyDesc,
            dzDesc,
            dxDesc,
            dBnScaleBiasDesc,
            activationDesc,
            sizeInBytes,
        )
    }
}

type cudnnGetBatchNormalizationTrainingExReserveSpaceSize_fn =
    unsafe extern "C" fn(
        handle: cudnnHandle_t,
        mode: cudnnBatchNormMode_t,
        bnOps: cudnnBatchNormOps_t,
        activationDesc: cudnnActivationDescriptor_t,
        xDesc: cudnnTensorDescriptor_t,
        sizeInBytes: *mut usize,
    ) -> cudnnStatus_t;
pub unsafe fn cudnnGetBatchNormalizationTrainingExReserveSpaceSize(
    handle: cudnnHandle_t,
    mode: cudnnBatchNormMode_t,
    bnOps: cudnnBatchNormOps_t,
    activationDesc: cudnnActivationDescriptor_t,
    xDesc: cudnnTensorDescriptor_t,
    sizeInBytes: *mut usize,
) -> cudnnStatus_t {
    let sym: Symbol<cudnnGetBatchNormalizationTrainingExReserveSpaceSize_fn> = unsafe {
        get_lib()
            .get(b"cudnnGetBatchNormalizationTrainingExReserveSpaceSize\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(handle, mode, bnOps, activationDesc, xDesc, sizeInBytes) }
}

type cudnnBatchNormalizationForwardTraining_fn = unsafe extern "C" fn(
    handle: cudnnHandle_t,
    mode: cudnnBatchNormMode_t,
    alpha: *const ::core::ffi::c_void,
    beta: *const ::core::ffi::c_void,
    xDesc: cudnnTensorDescriptor_t,
    x: *const ::core::ffi::c_void,
    yDesc: cudnnTensorDescriptor_t,
    y: *mut ::core::ffi::c_void,
    bnScaleBiasMeanVarDesc: cudnnTensorDescriptor_t,
    bnScale: *const ::core::ffi::c_void,
    bnBias: *const ::core::ffi::c_void,
    exponentialAverageFactor: f64,
    resultRunningMean: *mut ::core::ffi::c_void,
    resultRunningVariance: *mut ::core::ffi::c_void,
    epsilon: f64,
    resultSaveMean: *mut ::core::ffi::c_void,
    resultSaveInvVariance: *mut ::core::ffi::c_void,
) -> cudnnStatus_t;
pub unsafe fn cudnnBatchNormalizationForwardTraining(
    handle: cudnnHandle_t,
    mode: cudnnBatchNormMode_t,
    alpha: *const ::core::ffi::c_void,
    beta: *const ::core::ffi::c_void,
    xDesc: cudnnTensorDescriptor_t,
    x: *const ::core::ffi::c_void,
    yDesc: cudnnTensorDescriptor_t,
    y: *mut ::core::ffi::c_void,
    bnScaleBiasMeanVarDesc: cudnnTensorDescriptor_t,
    bnScale: *const ::core::ffi::c_void,
    bnBias: *const ::core::ffi::c_void,
    exponentialAverageFactor: f64,
    resultRunningMean: *mut ::core::ffi::c_void,
    resultRunningVariance: *mut ::core::ffi::c_void,
    epsilon: f64,
    resultSaveMean: *mut ::core::ffi::c_void,
    resultSaveInvVariance: *mut ::core::ffi::c_void,
) -> cudnnStatus_t {
    let sym: Symbol<cudnnBatchNormalizationForwardTraining_fn> = unsafe {
        get_lib()
            .get(b"cudnnBatchNormalizationForwardTraining\0")
            .expect("Missing symbol")
    };
    unsafe {
        (*sym)(
            handle,
            mode,
            alpha,
            beta,
            xDesc,
            x,
            yDesc,
            y,
            bnScaleBiasMeanVarDesc,
            bnScale,
            bnBias,
            exponentialAverageFactor,
            resultRunningMean,
            resultRunningVariance,
            epsilon,
            resultSaveMean,
            resultSaveInvVariance,
        )
    }
}

type cudnnBatchNormalizationForwardTrainingEx_fn = unsafe extern "C" fn(
    handle: cudnnHandle_t,
    mode: cudnnBatchNormMode_t,
    bnOps: cudnnBatchNormOps_t,
    alpha: *const ::core::ffi::c_void,
    beta: *const ::core::ffi::c_void,
    xDesc: cudnnTensorDescriptor_t,
    xData: *const ::core::ffi::c_void,
    zDesc: cudnnTensorDescriptor_t,
    zData: *const ::core::ffi::c_void,
    yDesc: cudnnTensorDescriptor_t,
    yData: *mut ::core::ffi::c_void,
    bnScaleBiasMeanVarDesc: cudnnTensorDescriptor_t,
    bnScale: *const ::core::ffi::c_void,
    bnBias: *const ::core::ffi::c_void,
    exponentialAverageFactor: f64,
    resultRunningMean: *mut ::core::ffi::c_void,
    resultRunningVariance: *mut ::core::ffi::c_void,
    epsilon: f64,
    resultSaveMean: *mut ::core::ffi::c_void,
    resultSaveInvVariance: *mut ::core::ffi::c_void,
    activationDesc: cudnnActivationDescriptor_t,
    workspace: *mut ::core::ffi::c_void,
    workSpaceSizeInBytes: usize,
    reserveSpace: *mut ::core::ffi::c_void,
    reserveSpaceSizeInBytes: usize,
) -> cudnnStatus_t;
pub unsafe fn cudnnBatchNormalizationForwardTrainingEx(
    handle: cudnnHandle_t,
    mode: cudnnBatchNormMode_t,
    bnOps: cudnnBatchNormOps_t,
    alpha: *const ::core::ffi::c_void,
    beta: *const ::core::ffi::c_void,
    xDesc: cudnnTensorDescriptor_t,
    xData: *const ::core::ffi::c_void,
    zDesc: cudnnTensorDescriptor_t,
    zData: *const ::core::ffi::c_void,
    yDesc: cudnnTensorDescriptor_t,
    yData: *mut ::core::ffi::c_void,
    bnScaleBiasMeanVarDesc: cudnnTensorDescriptor_t,
    bnScale: *const ::core::ffi::c_void,
    bnBias: *const ::core::ffi::c_void,
    exponentialAverageFactor: f64,
    resultRunningMean: *mut ::core::ffi::c_void,
    resultRunningVariance: *mut ::core::ffi::c_void,
    epsilon: f64,
    resultSaveMean: *mut ::core::ffi::c_void,
    resultSaveInvVariance: *mut ::core::ffi::c_void,
    activationDesc: cudnnActivationDescriptor_t,
    workspace: *mut ::core::ffi::c_void,
    workSpaceSizeInBytes: usize,
    reserveSpace: *mut ::core::ffi::c_void,
    reserveSpaceSizeInBytes: usize,
) -> cudnnStatus_t {
    let sym: Symbol<cudnnBatchNormalizationForwardTrainingEx_fn> = unsafe {
        get_lib()
            .get(b"cudnnBatchNormalizationForwardTrainingEx\0")
            .expect("Missing symbol")
    };
    unsafe {
        (*sym)(
            handle,
            mode,
            bnOps,
            alpha,
            beta,
            xDesc,
            xData,
            zDesc,
            zData,
            yDesc,
            yData,
            bnScaleBiasMeanVarDesc,
            bnScale,
            bnBias,
            exponentialAverageFactor,
            resultRunningMean,
            resultRunningVariance,
            epsilon,
            resultSaveMean,
            resultSaveInvVariance,
            activationDesc,
            workspace,
            workSpaceSizeInBytes,
            reserveSpace,
            reserveSpaceSizeInBytes,
        )
    }
}

type cudnnBatchNormalizationBackward_fn = unsafe extern "C" fn(
    handle: cudnnHandle_t,
    mode: cudnnBatchNormMode_t,
    alphaDataDiff: *const ::core::ffi::c_void,
    betaDataDiff: *const ::core::ffi::c_void,
    alphaParamDiff: *const ::core::ffi::c_void,
    betaParamDiff: *const ::core::ffi::c_void,
    xDesc: cudnnTensorDescriptor_t,
    x: *const ::core::ffi::c_void,
    dyDesc: cudnnTensorDescriptor_t,
    dy: *const ::core::ffi::c_void,
    dxDesc: cudnnTensorDescriptor_t,
    dx: *mut ::core::ffi::c_void,
    dBnScaleBiasDesc: cudnnTensorDescriptor_t,
    bnScale: *const ::core::ffi::c_void,
    dBnScaleResult: *mut ::core::ffi::c_void,
    dBnBiasResult: *mut ::core::ffi::c_void,
    epsilon: f64,
    savedMean: *const ::core::ffi::c_void,
    savedInvVariance: *const ::core::ffi::c_void,
) -> cudnnStatus_t;
pub unsafe fn cudnnBatchNormalizationBackward(
    handle: cudnnHandle_t,
    mode: cudnnBatchNormMode_t,
    alphaDataDiff: *const ::core::ffi::c_void,
    betaDataDiff: *const ::core::ffi::c_void,
    alphaParamDiff: *const ::core::ffi::c_void,
    betaParamDiff: *const ::core::ffi::c_void,
    xDesc: cudnnTensorDescriptor_t,
    x: *const ::core::ffi::c_void,
    dyDesc: cudnnTensorDescriptor_t,
    dy: *const ::core::ffi::c_void,
    dxDesc: cudnnTensorDescriptor_t,
    dx: *mut ::core::ffi::c_void,
    dBnScaleBiasDesc: cudnnTensorDescriptor_t,
    bnScale: *const ::core::ffi::c_void,
    dBnScaleResult: *mut ::core::ffi::c_void,
    dBnBiasResult: *mut ::core::ffi::c_void,
    epsilon: f64,
    savedMean: *const ::core::ffi::c_void,
    savedInvVariance: *const ::core::ffi::c_void,
) -> cudnnStatus_t {
    let sym: Symbol<cudnnBatchNormalizationBackward_fn> = unsafe {
        get_lib()
            .get(b"cudnnBatchNormalizationBackward\0")
            .expect("Missing symbol")
    };
    unsafe {
        (*sym)(
            handle,
            mode,
            alphaDataDiff,
            betaDataDiff,
            alphaParamDiff,
            betaParamDiff,
            xDesc,
            x,
            dyDesc,
            dy,
            dxDesc,
            dx,
            dBnScaleBiasDesc,
            bnScale,
            dBnScaleResult,
            dBnBiasResult,
            epsilon,
            savedMean,
            savedInvVariance,
        )
    }
}

type cudnnBatchNormalizationBackwardEx_fn = unsafe extern "C" fn(
    handle: cudnnHandle_t,
    mode: cudnnBatchNormMode_t,
    bnOps: cudnnBatchNormOps_t,
    alphaDataDiff: *const ::core::ffi::c_void,
    betaDataDiff: *const ::core::ffi::c_void,
    alphaParamDiff: *const ::core::ffi::c_void,
    betaParamDiff: *const ::core::ffi::c_void,
    xDesc: cudnnTensorDescriptor_t,
    xData: *const ::core::ffi::c_void,
    yDesc: cudnnTensorDescriptor_t,
    yData: *const ::core::ffi::c_void,
    dyDesc: cudnnTensorDescriptor_t,
    dyData: *const ::core::ffi::c_void,
    dzDesc: cudnnTensorDescriptor_t,
    dzData: *mut ::core::ffi::c_void,
    dxDesc: cudnnTensorDescriptor_t,
    dxData: *mut ::core::ffi::c_void,
    dBnScaleBiasDesc: cudnnTensorDescriptor_t,
    bnScaleData: *const ::core::ffi::c_void,
    bnBiasData: *const ::core::ffi::c_void,
    dBnScaleData: *mut ::core::ffi::c_void,
    dBnBiasData: *mut ::core::ffi::c_void,
    epsilon: f64,
    savedMean: *const ::core::ffi::c_void,
    savedInvVariance: *const ::core::ffi::c_void,
    activationDesc: cudnnActivationDescriptor_t,
    workSpace: *mut ::core::ffi::c_void,
    workSpaceSizeInBytes: usize,
    reserveSpace: *mut ::core::ffi::c_void,
    reserveSpaceSizeInBytes: usize,
) -> cudnnStatus_t;
pub unsafe fn cudnnBatchNormalizationBackwardEx(
    handle: cudnnHandle_t,
    mode: cudnnBatchNormMode_t,
    bnOps: cudnnBatchNormOps_t,
    alphaDataDiff: *const ::core::ffi::c_void,
    betaDataDiff: *const ::core::ffi::c_void,
    alphaParamDiff: *const ::core::ffi::c_void,
    betaParamDiff: *const ::core::ffi::c_void,
    xDesc: cudnnTensorDescriptor_t,
    xData: *const ::core::ffi::c_void,
    yDesc: cudnnTensorDescriptor_t,
    yData: *const ::core::ffi::c_void,
    dyDesc: cudnnTensorDescriptor_t,
    dyData: *const ::core::ffi::c_void,
    dzDesc: cudnnTensorDescriptor_t,
    dzData: *mut ::core::ffi::c_void,
    dxDesc: cudnnTensorDescriptor_t,
    dxData: *mut ::core::ffi::c_void,
    dBnScaleBiasDesc: cudnnTensorDescriptor_t,
    bnScaleData: *const ::core::ffi::c_void,
    bnBiasData: *const ::core::ffi::c_void,
    dBnScaleData: *mut ::core::ffi::c_void,
    dBnBiasData: *mut ::core::ffi::c_void,
    epsilon: f64,
    savedMean: *const ::core::ffi::c_void,
    savedInvVariance: *const ::core::ffi::c_void,
    activationDesc: cudnnActivationDescriptor_t,
    workSpace: *mut ::core::ffi::c_void,
    workSpaceSizeInBytes: usize,
    reserveSpace: *mut ::core::ffi::c_void,
    reserveSpaceSizeInBytes: usize,
) -> cudnnStatus_t {
    let sym: Symbol<cudnnBatchNormalizationBackwardEx_fn> = unsafe {
        get_lib()
            .get(b"cudnnBatchNormalizationBackwardEx\0")
            .expect("Missing symbol")
    };
    unsafe {
        (*sym)(
            handle,
            mode,
            bnOps,
            alphaDataDiff,
            betaDataDiff,
            alphaParamDiff,
            betaParamDiff,
            xDesc,
            xData,
            yDesc,
            yData,
            dyDesc,
            dyData,
            dzDesc,
            dzData,
            dxDesc,
            dxData,
            dBnScaleBiasDesc,
            bnScaleData,
            bnBiasData,
            dBnScaleData,
            dBnBiasData,
            epsilon,
            savedMean,
            savedInvVariance,
            activationDesc,
            workSpace,
            workSpaceSizeInBytes,
            reserveSpace,
            reserveSpaceSizeInBytes,
        )
    }
}

type cudnnGetNormalizationForwardTrainingWorkspaceSize_fn = unsafe extern "C" fn(
    handle: cudnnHandle_t,
    mode: cudnnNormMode_t,
    normOps: cudnnNormOps_t,
    algo: cudnnNormAlgo_t,
    xDesc: cudnnTensorDescriptor_t,
    zDesc: cudnnTensorDescriptor_t,
    yDesc: cudnnTensorDescriptor_t,
    normScaleBiasDesc: cudnnTensorDescriptor_t,
    activationDesc: cudnnActivationDescriptor_t,
    normMeanVarDesc: cudnnTensorDescriptor_t,
    sizeInBytes: *mut usize,
    groupCnt: ::core::ffi::c_int,
) -> cudnnStatus_t;
pub unsafe fn cudnnGetNormalizationForwardTrainingWorkspaceSize(
    handle: cudnnHandle_t,
    mode: cudnnNormMode_t,
    normOps: cudnnNormOps_t,
    algo: cudnnNormAlgo_t,
    xDesc: cudnnTensorDescriptor_t,
    zDesc: cudnnTensorDescriptor_t,
    yDesc: cudnnTensorDescriptor_t,
    normScaleBiasDesc: cudnnTensorDescriptor_t,
    activationDesc: cudnnActivationDescriptor_t,
    normMeanVarDesc: cudnnTensorDescriptor_t,
    sizeInBytes: *mut usize,
    groupCnt: ::core::ffi::c_int,
) -> cudnnStatus_t {
    let sym: Symbol<cudnnGetNormalizationForwardTrainingWorkspaceSize_fn> = unsafe {
        get_lib()
            .get(b"cudnnGetNormalizationForwardTrainingWorkspaceSize\0")
            .expect("Missing symbol")
    };
    unsafe {
        (*sym)(
            handle,
            mode,
            normOps,
            algo,
            xDesc,
            zDesc,
            yDesc,
            normScaleBiasDesc,
            activationDesc,
            normMeanVarDesc,
            sizeInBytes,
            groupCnt,
        )
    }
}

type cudnnGetNormalizationBackwardWorkspaceSize_fn = unsafe extern "C" fn(
    handle: cudnnHandle_t,
    mode: cudnnNormMode_t,
    normOps: cudnnNormOps_t,
    algo: cudnnNormAlgo_t,
    xDesc: cudnnTensorDescriptor_t,
    yDesc: cudnnTensorDescriptor_t,
    dyDesc: cudnnTensorDescriptor_t,
    dzDesc: cudnnTensorDescriptor_t,
    dxDesc: cudnnTensorDescriptor_t,
    dNormScaleBiasDesc: cudnnTensorDescriptor_t,
    activationDesc: cudnnActivationDescriptor_t,
    normMeanVarDesc: cudnnTensorDescriptor_t,
    sizeInBytes: *mut usize,
    groupCnt: ::core::ffi::c_int,
) -> cudnnStatus_t;
pub unsafe fn cudnnGetNormalizationBackwardWorkspaceSize(
    handle: cudnnHandle_t,
    mode: cudnnNormMode_t,
    normOps: cudnnNormOps_t,
    algo: cudnnNormAlgo_t,
    xDesc: cudnnTensorDescriptor_t,
    yDesc: cudnnTensorDescriptor_t,
    dyDesc: cudnnTensorDescriptor_t,
    dzDesc: cudnnTensorDescriptor_t,
    dxDesc: cudnnTensorDescriptor_t,
    dNormScaleBiasDesc: cudnnTensorDescriptor_t,
    activationDesc: cudnnActivationDescriptor_t,
    normMeanVarDesc: cudnnTensorDescriptor_t,
    sizeInBytes: *mut usize,
    groupCnt: ::core::ffi::c_int,
) -> cudnnStatus_t {
    let sym: Symbol<cudnnGetNormalizationBackwardWorkspaceSize_fn> = unsafe {
        get_lib()
            .get(b"cudnnGetNormalizationBackwardWorkspaceSize\0")
            .expect("Missing symbol")
    };
    unsafe {
        (*sym)(
            handle,
            mode,
            normOps,
            algo,
            xDesc,
            yDesc,
            dyDesc,
            dzDesc,
            dxDesc,
            dNormScaleBiasDesc,
            activationDesc,
            normMeanVarDesc,
            sizeInBytes,
            groupCnt,
        )
    }
}

type cudnnGetNormalizationTrainingReserveSpaceSize_fn = unsafe extern "C" fn(
    handle: cudnnHandle_t,
    mode: cudnnNormMode_t,
    normOps: cudnnNormOps_t,
    algo: cudnnNormAlgo_t,
    activationDesc: cudnnActivationDescriptor_t,
    xDesc: cudnnTensorDescriptor_t,
    sizeInBytes: *mut usize,
    groupCnt: ::core::ffi::c_int,
) -> cudnnStatus_t;
pub unsafe fn cudnnGetNormalizationTrainingReserveSpaceSize(
    handle: cudnnHandle_t,
    mode: cudnnNormMode_t,
    normOps: cudnnNormOps_t,
    algo: cudnnNormAlgo_t,
    activationDesc: cudnnActivationDescriptor_t,
    xDesc: cudnnTensorDescriptor_t,
    sizeInBytes: *mut usize,
    groupCnt: ::core::ffi::c_int,
) -> cudnnStatus_t {
    let sym: Symbol<cudnnGetNormalizationTrainingReserveSpaceSize_fn> = unsafe {
        get_lib()
            .get(b"cudnnGetNormalizationTrainingReserveSpaceSize\0")
            .expect("Missing symbol")
    };
    unsafe {
        (*sym)(
            handle,
            mode,
            normOps,
            algo,
            activationDesc,
            xDesc,
            sizeInBytes,
            groupCnt,
        )
    }
}

type cudnnNormalizationForwardTraining_fn = unsafe extern "C" fn(
    handle: cudnnHandle_t,
    mode: cudnnNormMode_t,
    normOps: cudnnNormOps_t,
    algo: cudnnNormAlgo_t,
    alpha: *const ::core::ffi::c_void,
    beta: *const ::core::ffi::c_void,
    xDesc: cudnnTensorDescriptor_t,
    xData: *const ::core::ffi::c_void,
    normScaleBiasDesc: cudnnTensorDescriptor_t,
    normScale: *const ::core::ffi::c_void,
    normBias: *const ::core::ffi::c_void,
    exponentialAverageFactor: f64,
    normMeanVarDesc: cudnnTensorDescriptor_t,
    resultRunningMean: *mut ::core::ffi::c_void,
    resultRunningVariance: *mut ::core::ffi::c_void,
    epsilon: f64,
    resultSaveMean: *mut ::core::ffi::c_void,
    resultSaveInvVariance: *mut ::core::ffi::c_void,
    activationDesc: cudnnActivationDescriptor_t,
    zDesc: cudnnTensorDescriptor_t,
    zData: *const ::core::ffi::c_void,
    yDesc: cudnnTensorDescriptor_t,
    yData: *mut ::core::ffi::c_void,
    workspace: *mut ::core::ffi::c_void,
    workSpaceSizeInBytes: usize,
    reserveSpace: *mut ::core::ffi::c_void,
    reserveSpaceSizeInBytes: usize,
    groupCnt: ::core::ffi::c_int,
) -> cudnnStatus_t;
pub unsafe fn cudnnNormalizationForwardTraining(
    handle: cudnnHandle_t,
    mode: cudnnNormMode_t,
    normOps: cudnnNormOps_t,
    algo: cudnnNormAlgo_t,
    alpha: *const ::core::ffi::c_void,
    beta: *const ::core::ffi::c_void,
    xDesc: cudnnTensorDescriptor_t,
    xData: *const ::core::ffi::c_void,
    normScaleBiasDesc: cudnnTensorDescriptor_t,
    normScale: *const ::core::ffi::c_void,
    normBias: *const ::core::ffi::c_void,
    exponentialAverageFactor: f64,
    normMeanVarDesc: cudnnTensorDescriptor_t,
    resultRunningMean: *mut ::core::ffi::c_void,
    resultRunningVariance: *mut ::core::ffi::c_void,
    epsilon: f64,
    resultSaveMean: *mut ::core::ffi::c_void,
    resultSaveInvVariance: *mut ::core::ffi::c_void,
    activationDesc: cudnnActivationDescriptor_t,
    zDesc: cudnnTensorDescriptor_t,
    zData: *const ::core::ffi::c_void,
    yDesc: cudnnTensorDescriptor_t,
    yData: *mut ::core::ffi::c_void,
    workspace: *mut ::core::ffi::c_void,
    workSpaceSizeInBytes: usize,
    reserveSpace: *mut ::core::ffi::c_void,
    reserveSpaceSizeInBytes: usize,
    groupCnt: ::core::ffi::c_int,
) -> cudnnStatus_t {
    let sym: Symbol<cudnnNormalizationForwardTraining_fn> = unsafe {
        get_lib()
            .get(b"cudnnNormalizationForwardTraining\0")
            .expect("Missing symbol")
    };
    unsafe {
        (*sym)(
            handle,
            mode,
            normOps,
            algo,
            alpha,
            beta,
            xDesc,
            xData,
            normScaleBiasDesc,
            normScale,
            normBias,
            exponentialAverageFactor,
            normMeanVarDesc,
            resultRunningMean,
            resultRunningVariance,
            epsilon,
            resultSaveMean,
            resultSaveInvVariance,
            activationDesc,
            zDesc,
            zData,
            yDesc,
            yData,
            workspace,
            workSpaceSizeInBytes,
            reserveSpace,
            reserveSpaceSizeInBytes,
            groupCnt,
        )
    }
}

type cudnnNormalizationBackward_fn = unsafe extern "C" fn(
    handle: cudnnHandle_t,
    mode: cudnnNormMode_t,
    normOps: cudnnNormOps_t,
    algo: cudnnNormAlgo_t,
    alphaDataDiff: *const ::core::ffi::c_void,
    betaDataDiff: *const ::core::ffi::c_void,
    alphaParamDiff: *const ::core::ffi::c_void,
    betaParamDiff: *const ::core::ffi::c_void,
    xDesc: cudnnTensorDescriptor_t,
    xData: *const ::core::ffi::c_void,
    yDesc: cudnnTensorDescriptor_t,
    yData: *const ::core::ffi::c_void,
    dyDesc: cudnnTensorDescriptor_t,
    dyData: *const ::core::ffi::c_void,
    dzDesc: cudnnTensorDescriptor_t,
    dzData: *mut ::core::ffi::c_void,
    dxDesc: cudnnTensorDescriptor_t,
    dxData: *mut ::core::ffi::c_void,
    dNormScaleBiasDesc: cudnnTensorDescriptor_t,
    normScaleData: *const ::core::ffi::c_void,
    normBiasData: *const ::core::ffi::c_void,
    dNormScaleData: *mut ::core::ffi::c_void,
    dNormBiasData: *mut ::core::ffi::c_void,
    epsilon: f64,
    normMeanVarDesc: cudnnTensorDescriptor_t,
    savedMean: *const ::core::ffi::c_void,
    savedInvVariance: *const ::core::ffi::c_void,
    activationDesc: cudnnActivationDescriptor_t,
    workSpace: *mut ::core::ffi::c_void,
    workSpaceSizeInBytes: usize,
    reserveSpace: *mut ::core::ffi::c_void,
    reserveSpaceSizeInBytes: usize,
    groupCnt: ::core::ffi::c_int,
) -> cudnnStatus_t;
pub unsafe fn cudnnNormalizationBackward(
    handle: cudnnHandle_t,
    mode: cudnnNormMode_t,
    normOps: cudnnNormOps_t,
    algo: cudnnNormAlgo_t,
    alphaDataDiff: *const ::core::ffi::c_void,
    betaDataDiff: *const ::core::ffi::c_void,
    alphaParamDiff: *const ::core::ffi::c_void,
    betaParamDiff: *const ::core::ffi::c_void,
    xDesc: cudnnTensorDescriptor_t,
    xData: *const ::core::ffi::c_void,
    yDesc: cudnnTensorDescriptor_t,
    yData: *const ::core::ffi::c_void,
    dyDesc: cudnnTensorDescriptor_t,
    dyData: *const ::core::ffi::c_void,
    dzDesc: cudnnTensorDescriptor_t,
    dzData: *mut ::core::ffi::c_void,
    dxDesc: cudnnTensorDescriptor_t,
    dxData: *mut ::core::ffi::c_void,
    dNormScaleBiasDesc: cudnnTensorDescriptor_t,
    normScaleData: *const ::core::ffi::c_void,
    normBiasData: *const ::core::ffi::c_void,
    dNormScaleData: *mut ::core::ffi::c_void,
    dNormBiasData: *mut ::core::ffi::c_void,
    epsilon: f64,
    normMeanVarDesc: cudnnTensorDescriptor_t,
    savedMean: *const ::core::ffi::c_void,
    savedInvVariance: *const ::core::ffi::c_void,
    activationDesc: cudnnActivationDescriptor_t,
    workSpace: *mut ::core::ffi::c_void,
    workSpaceSizeInBytes: usize,
    reserveSpace: *mut ::core::ffi::c_void,
    reserveSpaceSizeInBytes: usize,
    groupCnt: ::core::ffi::c_int,
) -> cudnnStatus_t {
    let sym: Symbol<cudnnNormalizationBackward_fn> = unsafe {
        get_lib()
            .get(b"cudnnNormalizationBackward\0")
            .expect("Missing symbol")
    };
    unsafe {
        (*sym)(
            handle,
            mode,
            normOps,
            algo,
            alphaDataDiff,
            betaDataDiff,
            alphaParamDiff,
            betaParamDiff,
            xDesc,
            xData,
            yDesc,
            yData,
            dyDesc,
            dyData,
            dzDesc,
            dzData,
            dxDesc,
            dxData,
            dNormScaleBiasDesc,
            normScaleData,
            normBiasData,
            dNormScaleData,
            dNormBiasData,
            epsilon,
            normMeanVarDesc,
            savedMean,
            savedInvVariance,
            activationDesc,
            workSpace,
            workSpaceSizeInBytes,
            reserveSpace,
            reserveSpaceSizeInBytes,
            groupCnt,
        )
    }
}

type cudnnSpatialTfGridGeneratorBackward_fn = unsafe extern "C" fn(
    handle: cudnnHandle_t,
    stDesc: cudnnSpatialTransformerDescriptor_t,
    dgrid: *const ::core::ffi::c_void,
    dtheta: *mut ::core::ffi::c_void,
) -> cudnnStatus_t;
pub unsafe fn cudnnSpatialTfGridGeneratorBackward(
    handle: cudnnHandle_t,
    stDesc: cudnnSpatialTransformerDescriptor_t,
    dgrid: *const ::core::ffi::c_void,
    dtheta: *mut ::core::ffi::c_void,
) -> cudnnStatus_t {
    let sym: Symbol<cudnnSpatialTfGridGeneratorBackward_fn> = unsafe {
        get_lib()
            .get(b"cudnnSpatialTfGridGeneratorBackward\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(handle, stDesc, dgrid, dtheta) }
}

type cudnnSpatialTfSamplerBackward_fn = unsafe extern "C" fn(
    handle: cudnnHandle_t,
    stDesc: cudnnSpatialTransformerDescriptor_t,
    alpha: *const ::core::ffi::c_void,
    xDesc: cudnnTensorDescriptor_t,
    x: *const ::core::ffi::c_void,
    beta: *const ::core::ffi::c_void,
    dxDesc: cudnnTensorDescriptor_t,
    dx: *mut ::core::ffi::c_void,
    alphaDgrid: *const ::core::ffi::c_void,
    dyDesc: cudnnTensorDescriptor_t,
    dy: *const ::core::ffi::c_void,
    grid: *const ::core::ffi::c_void,
    betaDgrid: *const ::core::ffi::c_void,
    dgrid: *mut ::core::ffi::c_void,
) -> cudnnStatus_t;
pub unsafe fn cudnnSpatialTfSamplerBackward(
    handle: cudnnHandle_t,
    stDesc: cudnnSpatialTransformerDescriptor_t,
    alpha: *const ::core::ffi::c_void,
    xDesc: cudnnTensorDescriptor_t,
    x: *const ::core::ffi::c_void,
    beta: *const ::core::ffi::c_void,
    dxDesc: cudnnTensorDescriptor_t,
    dx: *mut ::core::ffi::c_void,
    alphaDgrid: *const ::core::ffi::c_void,
    dyDesc: cudnnTensorDescriptor_t,
    dy: *const ::core::ffi::c_void,
    grid: *const ::core::ffi::c_void,
    betaDgrid: *const ::core::ffi::c_void,
    dgrid: *mut ::core::ffi::c_void,
) -> cudnnStatus_t {
    let sym: Symbol<cudnnSpatialTfSamplerBackward_fn> = unsafe {
        get_lib()
            .get(b"cudnnSpatialTfSamplerBackward\0")
            .expect("Missing symbol")
    };
    unsafe {
        (*sym)(
            handle, stDesc, alpha, xDesc, x, beta, dxDesc, dx, alphaDgrid, dyDesc, dy, grid,
            betaDgrid, dgrid,
        )
    }
}

type cudnnDropoutBackward_fn = unsafe extern "C" fn(
    handle: cudnnHandle_t,
    dropoutDesc: cudnnDropoutDescriptor_t,
    dydesc: cudnnTensorDescriptor_t,
    dy: *const ::core::ffi::c_void,
    dxdesc: cudnnTensorDescriptor_t,
    dx: *mut ::core::ffi::c_void,
    reserveSpace: *mut ::core::ffi::c_void,
    reserveSpaceSizeInBytes: usize,
) -> cudnnStatus_t;
pub unsafe fn cudnnDropoutBackward(
    handle: cudnnHandle_t,
    dropoutDesc: cudnnDropoutDescriptor_t,
    dydesc: cudnnTensorDescriptor_t,
    dy: *const ::core::ffi::c_void,
    dxdesc: cudnnTensorDescriptor_t,
    dx: *mut ::core::ffi::c_void,
    reserveSpace: *mut ::core::ffi::c_void,
    reserveSpaceSizeInBytes: usize,
) -> cudnnStatus_t {
    let sym: Symbol<cudnnDropoutBackward_fn> = unsafe {
        get_lib()
            .get(b"cudnnDropoutBackward\0")
            .expect("Missing symbol")
    };
    unsafe {
        (*sym)(
            handle,
            dropoutDesc,
            dydesc,
            dy,
            dxdesc,
            dx,
            reserveSpace,
            reserveSpaceSizeInBytes,
        )
    }
}

type cudnnCreateRNNDescriptor_fn =
    unsafe extern "C" fn(rnnDesc: *mut cudnnRNNDescriptor_t) -> cudnnStatus_t;
pub unsafe fn cudnnCreateRNNDescriptor(rnnDesc: *mut cudnnRNNDescriptor_t) -> cudnnStatus_t {
    let sym: Symbol<cudnnCreateRNNDescriptor_fn> = unsafe {
        get_lib()
            .get(b"cudnnCreateRNNDescriptor\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(rnnDesc) }
}

type cudnnDestroyRNNDescriptor_fn =
    unsafe extern "C" fn(rnnDesc: cudnnRNNDescriptor_t) -> cudnnStatus_t;
pub unsafe fn cudnnDestroyRNNDescriptor(rnnDesc: cudnnRNNDescriptor_t) -> cudnnStatus_t {
    let sym: Symbol<cudnnDestroyRNNDescriptor_fn> = unsafe {
        get_lib()
            .get(b"cudnnDestroyRNNDescriptor\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(rnnDesc) }
}

type cudnnSetRNNDescriptor_v8_fn = unsafe extern "C" fn(
    rnnDesc: cudnnRNNDescriptor_t,
    algo: cudnnRNNAlgo_t,
    cellMode: cudnnRNNMode_t,
    biasMode: cudnnRNNBiasMode_t,
    dirMode: cudnnDirectionMode_t,
    inputMode: cudnnRNNInputMode_t,
    dataType: cudnnDataType_t,
    mathPrec: cudnnDataType_t,
    mathType: cudnnMathType_t,
    inputSize: i32,
    hiddenSize: i32,
    projSize: i32,
    numLayers: i32,
    dropoutDesc: cudnnDropoutDescriptor_t,
    auxFlags: u32,
) -> cudnnStatus_t;
pub unsafe fn cudnnSetRNNDescriptor_v8(
    rnnDesc: cudnnRNNDescriptor_t,
    algo: cudnnRNNAlgo_t,
    cellMode: cudnnRNNMode_t,
    biasMode: cudnnRNNBiasMode_t,
    dirMode: cudnnDirectionMode_t,
    inputMode: cudnnRNNInputMode_t,
    dataType: cudnnDataType_t,
    mathPrec: cudnnDataType_t,
    mathType: cudnnMathType_t,
    inputSize: i32,
    hiddenSize: i32,
    projSize: i32,
    numLayers: i32,
    dropoutDesc: cudnnDropoutDescriptor_t,
    auxFlags: u32,
) -> cudnnStatus_t {
    let sym: Symbol<cudnnSetRNNDescriptor_v8_fn> = unsafe {
        get_lib()
            .get(b"cudnnSetRNNDescriptor_v8\0")
            .expect("Missing symbol")
    };
    unsafe {
        (*sym)(
            rnnDesc,
            algo,
            cellMode,
            biasMode,
            dirMode,
            inputMode,
            dataType,
            mathPrec,
            mathType,
            inputSize,
            hiddenSize,
            projSize,
            numLayers,
            dropoutDesc,
            auxFlags,
        )
    }
}

type cudnnGetRNNDescriptor_v8_fn = unsafe extern "C" fn(
    rnnDesc: cudnnRNNDescriptor_t,
    algo: *mut cudnnRNNAlgo_t,
    cellMode: *mut cudnnRNNMode_t,
    biasMode: *mut cudnnRNNBiasMode_t,
    dirMode: *mut cudnnDirectionMode_t,
    inputMode: *mut cudnnRNNInputMode_t,
    dataType: *mut cudnnDataType_t,
    mathPrec: *mut cudnnDataType_t,
    mathType: *mut cudnnMathType_t,
    inputSize: *mut i32,
    hiddenSize: *mut i32,
    projSize: *mut i32,
    numLayers: *mut i32,
    dropoutDesc: *mut cudnnDropoutDescriptor_t,
    auxFlags: *mut u32,
) -> cudnnStatus_t;
pub unsafe fn cudnnGetRNNDescriptor_v8(
    rnnDesc: cudnnRNNDescriptor_t,
    algo: *mut cudnnRNNAlgo_t,
    cellMode: *mut cudnnRNNMode_t,
    biasMode: *mut cudnnRNNBiasMode_t,
    dirMode: *mut cudnnDirectionMode_t,
    inputMode: *mut cudnnRNNInputMode_t,
    dataType: *mut cudnnDataType_t,
    mathPrec: *mut cudnnDataType_t,
    mathType: *mut cudnnMathType_t,
    inputSize: *mut i32,
    hiddenSize: *mut i32,
    projSize: *mut i32,
    numLayers: *mut i32,
    dropoutDesc: *mut cudnnDropoutDescriptor_t,
    auxFlags: *mut u32,
) -> cudnnStatus_t {
    let sym: Symbol<cudnnGetRNNDescriptor_v8_fn> = unsafe {
        get_lib()
            .get(b"cudnnGetRNNDescriptor_v8\0")
            .expect("Missing symbol")
    };
    unsafe {
        (*sym)(
            rnnDesc,
            algo,
            cellMode,
            biasMode,
            dirMode,
            inputMode,
            dataType,
            mathPrec,
            mathType,
            inputSize,
            hiddenSize,
            projSize,
            numLayers,
            dropoutDesc,
            auxFlags,
        )
    }
}

type cudnnRNNSetClip_v8_fn = unsafe extern "C" fn(
    rnnDesc: cudnnRNNDescriptor_t,
    clipMode: cudnnRNNClipMode_t,
    clipNanOpt: cudnnNanPropagation_t,
    lclip: f64,
    rclip: f64,
) -> cudnnStatus_t;
pub unsafe fn cudnnRNNSetClip_v8(
    rnnDesc: cudnnRNNDescriptor_t,
    clipMode: cudnnRNNClipMode_t,
    clipNanOpt: cudnnNanPropagation_t,
    lclip: f64,
    rclip: f64,
) -> cudnnStatus_t {
    let sym: Symbol<cudnnRNNSetClip_v8_fn> = unsafe {
        get_lib()
            .get(b"cudnnRNNSetClip_v8\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(rnnDesc, clipMode, clipNanOpt, lclip, rclip) }
}

type cudnnRNNSetClip_v9_fn = unsafe extern "C" fn(
    rnnDesc: cudnnRNNDescriptor_t,
    clipMode: cudnnRNNClipMode_t,
    lclip: f64,
    rclip: f64,
) -> cudnnStatus_t;
pub unsafe fn cudnnRNNSetClip_v9(
    rnnDesc: cudnnRNNDescriptor_t,
    clipMode: cudnnRNNClipMode_t,
    lclip: f64,
    rclip: f64,
) -> cudnnStatus_t {
    let sym: Symbol<cudnnRNNSetClip_v9_fn> = unsafe {
        get_lib()
            .get(b"cudnnRNNSetClip_v9\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(rnnDesc, clipMode, lclip, rclip) }
}

type cudnnRNNGetClip_v8_fn = unsafe extern "C" fn(
    rnnDesc: cudnnRNNDescriptor_t,
    clipMode: *mut cudnnRNNClipMode_t,
    clipNanOpt: *mut cudnnNanPropagation_t,
    lclip: *mut f64,
    rclip: *mut f64,
) -> cudnnStatus_t;
pub unsafe fn cudnnRNNGetClip_v8(
    rnnDesc: cudnnRNNDescriptor_t,
    clipMode: *mut cudnnRNNClipMode_t,
    clipNanOpt: *mut cudnnNanPropagation_t,
    lclip: *mut f64,
    rclip: *mut f64,
) -> cudnnStatus_t {
    let sym: Symbol<cudnnRNNGetClip_v8_fn> = unsafe {
        get_lib()
            .get(b"cudnnRNNGetClip_v8\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(rnnDesc, clipMode, clipNanOpt, lclip, rclip) }
}

type cudnnRNNGetClip_v9_fn = unsafe extern "C" fn(
    rnnDesc: cudnnRNNDescriptor_t,
    clipMode: *mut cudnnRNNClipMode_t,
    lclip: *mut f64,
    rclip: *mut f64,
) -> cudnnStatus_t;
pub unsafe fn cudnnRNNGetClip_v9(
    rnnDesc: cudnnRNNDescriptor_t,
    clipMode: *mut cudnnRNNClipMode_t,
    lclip: *mut f64,
    rclip: *mut f64,
) -> cudnnStatus_t {
    let sym: Symbol<cudnnRNNGetClip_v9_fn> = unsafe {
        get_lib()
            .get(b"cudnnRNNGetClip_v9\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(rnnDesc, clipMode, lclip, rclip) }
}

type cudnnBuildRNNDynamic_fn = unsafe extern "C" fn(
    handle: cudnnHandle_t,
    rnnDesc: cudnnRNNDescriptor_t,
    miniBatch: ::core::ffi::c_int,
) -> cudnnStatus_t;
pub unsafe fn cudnnBuildRNNDynamic(
    handle: cudnnHandle_t,
    rnnDesc: cudnnRNNDescriptor_t,
    miniBatch: ::core::ffi::c_int,
) -> cudnnStatus_t {
    let sym: Symbol<cudnnBuildRNNDynamic_fn> = unsafe {
        get_lib()
            .get(b"cudnnBuildRNNDynamic\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(handle, rnnDesc, miniBatch) }
}

type cudnnGetRNNTempSpaceSizes_fn = unsafe extern "C" fn(
    handle: cudnnHandle_t,
    rnnDesc: cudnnRNNDescriptor_t,
    fwdMode: cudnnForwardMode_t,
    xDesc: cudnnRNNDataDescriptor_t,
    workSpaceSize: *mut usize,
    reserveSpaceSize: *mut usize,
) -> cudnnStatus_t;
pub unsafe fn cudnnGetRNNTempSpaceSizes(
    handle: cudnnHandle_t,
    rnnDesc: cudnnRNNDescriptor_t,
    fwdMode: cudnnForwardMode_t,
    xDesc: cudnnRNNDataDescriptor_t,
    workSpaceSize: *mut usize,
    reserveSpaceSize: *mut usize,
) -> cudnnStatus_t {
    let sym: Symbol<cudnnGetRNNTempSpaceSizes_fn> = unsafe {
        get_lib()
            .get(b"cudnnGetRNNTempSpaceSizes\0")
            .expect("Missing symbol")
    };
    unsafe {
        (*sym)(
            handle,
            rnnDesc,
            fwdMode,
            xDesc,
            workSpaceSize,
            reserveSpaceSize,
        )
    }
}

type cudnnGetRNNWeightSpaceSize_fn = unsafe extern "C" fn(
    handle: cudnnHandle_t,
    rnnDesc: cudnnRNNDescriptor_t,
    weightSpaceSize: *mut usize,
) -> cudnnStatus_t;
pub unsafe fn cudnnGetRNNWeightSpaceSize(
    handle: cudnnHandle_t,
    rnnDesc: cudnnRNNDescriptor_t,
    weightSpaceSize: *mut usize,
) -> cudnnStatus_t {
    let sym: Symbol<cudnnGetRNNWeightSpaceSize_fn> = unsafe {
        get_lib()
            .get(b"cudnnGetRNNWeightSpaceSize\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(handle, rnnDesc, weightSpaceSize) }
}

type cudnnGetRNNWeightParams_fn = unsafe extern "C" fn(
    handle: cudnnHandle_t,
    rnnDesc: cudnnRNNDescriptor_t,
    pseudoLayer: i32,
    weightSpaceSize: usize,
    weightSpace: *const ::core::ffi::c_void,
    linLayerID: i32,
    mDesc: cudnnTensorDescriptor_t,
    mAddr: *mut *mut ::core::ffi::c_void,
    bDesc: cudnnTensorDescriptor_t,
    bAddr: *mut *mut ::core::ffi::c_void,
) -> cudnnStatus_t;
pub unsafe fn cudnnGetRNNWeightParams(
    handle: cudnnHandle_t,
    rnnDesc: cudnnRNNDescriptor_t,
    pseudoLayer: i32,
    weightSpaceSize: usize,
    weightSpace: *const ::core::ffi::c_void,
    linLayerID: i32,
    mDesc: cudnnTensorDescriptor_t,
    mAddr: *mut *mut ::core::ffi::c_void,
    bDesc: cudnnTensorDescriptor_t,
    bAddr: *mut *mut ::core::ffi::c_void,
) -> cudnnStatus_t {
    let sym: Symbol<cudnnGetRNNWeightParams_fn> = unsafe {
        get_lib()
            .get(b"cudnnGetRNNWeightParams\0")
            .expect("Missing symbol")
    };
    unsafe {
        (*sym)(
            handle,
            rnnDesc,
            pseudoLayer,
            weightSpaceSize,
            weightSpace,
            linLayerID,
            mDesc,
            mAddr,
            bDesc,
            bAddr,
        )
    }
}

type cudnnCreateRNNDataDescriptor_fn =
    unsafe extern "C" fn(rnnDataDesc: *mut cudnnRNNDataDescriptor_t) -> cudnnStatus_t;
pub unsafe fn cudnnCreateRNNDataDescriptor(
    rnnDataDesc: *mut cudnnRNNDataDescriptor_t,
) -> cudnnStatus_t {
    let sym: Symbol<cudnnCreateRNNDataDescriptor_fn> = unsafe {
        get_lib()
            .get(b"cudnnCreateRNNDataDescriptor\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(rnnDataDesc) }
}

type cudnnDestroyRNNDataDescriptor_fn =
    unsafe extern "C" fn(rnnDataDesc: cudnnRNNDataDescriptor_t) -> cudnnStatus_t;
pub unsafe fn cudnnDestroyRNNDataDescriptor(
    rnnDataDesc: cudnnRNNDataDescriptor_t,
) -> cudnnStatus_t {
    let sym: Symbol<cudnnDestroyRNNDataDescriptor_fn> = unsafe {
        get_lib()
            .get(b"cudnnDestroyRNNDataDescriptor\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(rnnDataDesc) }
}

type cudnnSetRNNDataDescriptor_fn = unsafe extern "C" fn(
    rnnDataDesc: cudnnRNNDataDescriptor_t,
    dataType: cudnnDataType_t,
    layout: cudnnRNNDataLayout_t,
    maxSeqLength: ::core::ffi::c_int,
    batchSize: ::core::ffi::c_int,
    vectorSize: ::core::ffi::c_int,
    seqLengthArray: *const ::core::ffi::c_int,
    paddingFill: *mut ::core::ffi::c_void,
) -> cudnnStatus_t;
pub unsafe fn cudnnSetRNNDataDescriptor(
    rnnDataDesc: cudnnRNNDataDescriptor_t,
    dataType: cudnnDataType_t,
    layout: cudnnRNNDataLayout_t,
    maxSeqLength: ::core::ffi::c_int,
    batchSize: ::core::ffi::c_int,
    vectorSize: ::core::ffi::c_int,
    seqLengthArray: *const ::core::ffi::c_int,
    paddingFill: *mut ::core::ffi::c_void,
) -> cudnnStatus_t {
    let sym: Symbol<cudnnSetRNNDataDescriptor_fn> = unsafe {
        get_lib()
            .get(b"cudnnSetRNNDataDescriptor\0")
            .expect("Missing symbol")
    };
    unsafe {
        (*sym)(
            rnnDataDesc,
            dataType,
            layout,
            maxSeqLength,
            batchSize,
            vectorSize,
            seqLengthArray,
            paddingFill,
        )
    }
}

type cudnnGetRNNDataDescriptor_fn = unsafe extern "C" fn(
    rnnDataDesc: cudnnRNNDataDescriptor_t,
    dataType: *mut cudnnDataType_t,
    layout: *mut cudnnRNNDataLayout_t,
    maxSeqLength: *mut ::core::ffi::c_int,
    batchSize: *mut ::core::ffi::c_int,
    vectorSize: *mut ::core::ffi::c_int,
    arrayLengthRequested: ::core::ffi::c_int,
    seqLengthArray: *mut ::core::ffi::c_int,
    paddingFill: *mut ::core::ffi::c_void,
) -> cudnnStatus_t;
pub unsafe fn cudnnGetRNNDataDescriptor(
    rnnDataDesc: cudnnRNNDataDescriptor_t,
    dataType: *mut cudnnDataType_t,
    layout: *mut cudnnRNNDataLayout_t,
    maxSeqLength: *mut ::core::ffi::c_int,
    batchSize: *mut ::core::ffi::c_int,
    vectorSize: *mut ::core::ffi::c_int,
    arrayLengthRequested: ::core::ffi::c_int,
    seqLengthArray: *mut ::core::ffi::c_int,
    paddingFill: *mut ::core::ffi::c_void,
) -> cudnnStatus_t {
    let sym: Symbol<cudnnGetRNNDataDescriptor_fn> = unsafe {
        get_lib()
            .get(b"cudnnGetRNNDataDescriptor\0")
            .expect("Missing symbol")
    };
    unsafe {
        (*sym)(
            rnnDataDesc,
            dataType,
            layout,
            maxSeqLength,
            batchSize,
            vectorSize,
            arrayLengthRequested,
            seqLengthArray,
            paddingFill,
        )
    }
}

type cudnnRNNForward_fn = unsafe extern "C" fn(
    handle: cudnnHandle_t,
    rnnDesc: cudnnRNNDescriptor_t,
    fwdMode: cudnnForwardMode_t,
    devSeqLengths: *const i32,
    xDesc: cudnnRNNDataDescriptor_t,
    x: *const ::core::ffi::c_void,
    yDesc: cudnnRNNDataDescriptor_t,
    y: *mut ::core::ffi::c_void,
    hDesc: cudnnTensorDescriptor_t,
    hx: *const ::core::ffi::c_void,
    hy: *mut ::core::ffi::c_void,
    cDesc: cudnnTensorDescriptor_t,
    cx: *const ::core::ffi::c_void,
    cy: *mut ::core::ffi::c_void,
    weightSpaceSize: usize,
    weightSpace: *const ::core::ffi::c_void,
    workSpaceSize: usize,
    workSpace: *mut ::core::ffi::c_void,
    reserveSpaceSize: usize,
    reserveSpace: *mut ::core::ffi::c_void,
) -> cudnnStatus_t;
pub unsafe fn cudnnRNNForward(
    handle: cudnnHandle_t,
    rnnDesc: cudnnRNNDescriptor_t,
    fwdMode: cudnnForwardMode_t,
    devSeqLengths: *const i32,
    xDesc: cudnnRNNDataDescriptor_t,
    x: *const ::core::ffi::c_void,
    yDesc: cudnnRNNDataDescriptor_t,
    y: *mut ::core::ffi::c_void,
    hDesc: cudnnTensorDescriptor_t,
    hx: *const ::core::ffi::c_void,
    hy: *mut ::core::ffi::c_void,
    cDesc: cudnnTensorDescriptor_t,
    cx: *const ::core::ffi::c_void,
    cy: *mut ::core::ffi::c_void,
    weightSpaceSize: usize,
    weightSpace: *const ::core::ffi::c_void,
    workSpaceSize: usize,
    workSpace: *mut ::core::ffi::c_void,
    reserveSpaceSize: usize,
    reserveSpace: *mut ::core::ffi::c_void,
) -> cudnnStatus_t {
    let sym: Symbol<cudnnRNNForward_fn> =
        unsafe { get_lib().get(b"cudnnRNNForward\0").expect("Missing symbol") };
    unsafe {
        (*sym)(
            handle,
            rnnDesc,
            fwdMode,
            devSeqLengths,
            xDesc,
            x,
            yDesc,
            y,
            hDesc,
            hx,
            hy,
            cDesc,
            cx,
            cy,
            weightSpaceSize,
            weightSpace,
            workSpaceSize,
            workSpace,
            reserveSpaceSize,
            reserveSpace,
        )
    }
}

type cudnnCreateSeqDataDescriptor_fn =
    unsafe extern "C" fn(seqDataDesc: *mut cudnnSeqDataDescriptor_t) -> cudnnStatus_t;
pub unsafe fn cudnnCreateSeqDataDescriptor(
    seqDataDesc: *mut cudnnSeqDataDescriptor_t,
) -> cudnnStatus_t {
    let sym: Symbol<cudnnCreateSeqDataDescriptor_fn> = unsafe {
        get_lib()
            .get(b"cudnnCreateSeqDataDescriptor\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(seqDataDesc) }
}

type cudnnDestroySeqDataDescriptor_fn =
    unsafe extern "C" fn(seqDataDesc: cudnnSeqDataDescriptor_t) -> cudnnStatus_t;
pub unsafe fn cudnnDestroySeqDataDescriptor(
    seqDataDesc: cudnnSeqDataDescriptor_t,
) -> cudnnStatus_t {
    let sym: Symbol<cudnnDestroySeqDataDescriptor_fn> = unsafe {
        get_lib()
            .get(b"cudnnDestroySeqDataDescriptor\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(seqDataDesc) }
}

type cudnnSetSeqDataDescriptor_fn = unsafe extern "C" fn(
    seqDataDesc: cudnnSeqDataDescriptor_t,
    dataType: cudnnDataType_t,
    nbDims: ::core::ffi::c_int,
    dimA: *const ::core::ffi::c_int,
    axes: *const cudnnSeqDataAxis_t,
    seqLengthArraySize: usize,
    seqLengthArray: *const ::core::ffi::c_int,
    paddingFill: *mut ::core::ffi::c_void,
) -> cudnnStatus_t;
pub unsafe fn cudnnSetSeqDataDescriptor(
    seqDataDesc: cudnnSeqDataDescriptor_t,
    dataType: cudnnDataType_t,
    nbDims: ::core::ffi::c_int,
    dimA: *const ::core::ffi::c_int,
    axes: *const cudnnSeqDataAxis_t,
    seqLengthArraySize: usize,
    seqLengthArray: *const ::core::ffi::c_int,
    paddingFill: *mut ::core::ffi::c_void,
) -> cudnnStatus_t {
    let sym: Symbol<cudnnSetSeqDataDescriptor_fn> = unsafe {
        get_lib()
            .get(b"cudnnSetSeqDataDescriptor\0")
            .expect("Missing symbol")
    };
    unsafe {
        (*sym)(
            seqDataDesc,
            dataType,
            nbDims,
            dimA,
            axes,
            seqLengthArraySize,
            seqLengthArray,
            paddingFill,
        )
    }
}

type cudnnGetSeqDataDescriptor_fn = unsafe extern "C" fn(
    seqDataDesc: cudnnSeqDataDescriptor_t,
    dataType: *mut cudnnDataType_t,
    nbDims: *mut ::core::ffi::c_int,
    nbDimsRequested: ::core::ffi::c_int,
    dimA: *mut ::core::ffi::c_int,
    axes: *mut cudnnSeqDataAxis_t,
    seqLengthArraySize: *mut usize,
    seqLengthSizeRequested: usize,
    seqLengthArray: *mut ::core::ffi::c_int,
    paddingFill: *mut ::core::ffi::c_void,
) -> cudnnStatus_t;
pub unsafe fn cudnnGetSeqDataDescriptor(
    seqDataDesc: cudnnSeqDataDescriptor_t,
    dataType: *mut cudnnDataType_t,
    nbDims: *mut ::core::ffi::c_int,
    nbDimsRequested: ::core::ffi::c_int,
    dimA: *mut ::core::ffi::c_int,
    axes: *mut cudnnSeqDataAxis_t,
    seqLengthArraySize: *mut usize,
    seqLengthSizeRequested: usize,
    seqLengthArray: *mut ::core::ffi::c_int,
    paddingFill: *mut ::core::ffi::c_void,
) -> cudnnStatus_t {
    let sym: Symbol<cudnnGetSeqDataDescriptor_fn> = unsafe {
        get_lib()
            .get(b"cudnnGetSeqDataDescriptor\0")
            .expect("Missing symbol")
    };
    unsafe {
        (*sym)(
            seqDataDesc,
            dataType,
            nbDims,
            nbDimsRequested,
            dimA,
            axes,
            seqLengthArraySize,
            seqLengthSizeRequested,
            seqLengthArray,
            paddingFill,
        )
    }
}

type cudnnCreateAttnDescriptor_fn =
    unsafe extern "C" fn(attnDesc: *mut cudnnAttnDescriptor_t) -> cudnnStatus_t;
pub unsafe fn cudnnCreateAttnDescriptor(attnDesc: *mut cudnnAttnDescriptor_t) -> cudnnStatus_t {
    let sym: Symbol<cudnnCreateAttnDescriptor_fn> = unsafe {
        get_lib()
            .get(b"cudnnCreateAttnDescriptor\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(attnDesc) }
}

type cudnnDestroyAttnDescriptor_fn =
    unsafe extern "C" fn(attnDesc: cudnnAttnDescriptor_t) -> cudnnStatus_t;
pub unsafe fn cudnnDestroyAttnDescriptor(attnDesc: cudnnAttnDescriptor_t) -> cudnnStatus_t {
    let sym: Symbol<cudnnDestroyAttnDescriptor_fn> = unsafe {
        get_lib()
            .get(b"cudnnDestroyAttnDescriptor\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(attnDesc) }
}

type cudnnSetAttnDescriptor_fn = unsafe extern "C" fn(
    attnDesc: cudnnAttnDescriptor_t,
    attnMode: ::core::ffi::c_uint,
    nHeads: ::core::ffi::c_int,
    smScaler: f64,
    dataType: cudnnDataType_t,
    computePrec: cudnnDataType_t,
    mathType: cudnnMathType_t,
    attnDropoutDesc: cudnnDropoutDescriptor_t,
    postDropoutDesc: cudnnDropoutDescriptor_t,
    qSize: ::core::ffi::c_int,
    kSize: ::core::ffi::c_int,
    vSize: ::core::ffi::c_int,
    qProjSize: ::core::ffi::c_int,
    kProjSize: ::core::ffi::c_int,
    vProjSize: ::core::ffi::c_int,
    oProjSize: ::core::ffi::c_int,
    qoMaxSeqLength: ::core::ffi::c_int,
    kvMaxSeqLength: ::core::ffi::c_int,
    maxBatchSize: ::core::ffi::c_int,
    maxBeamSize: ::core::ffi::c_int,
) -> cudnnStatus_t;
pub unsafe fn cudnnSetAttnDescriptor(
    attnDesc: cudnnAttnDescriptor_t,
    attnMode: ::core::ffi::c_uint,
    nHeads: ::core::ffi::c_int,
    smScaler: f64,
    dataType: cudnnDataType_t,
    computePrec: cudnnDataType_t,
    mathType: cudnnMathType_t,
    attnDropoutDesc: cudnnDropoutDescriptor_t,
    postDropoutDesc: cudnnDropoutDescriptor_t,
    qSize: ::core::ffi::c_int,
    kSize: ::core::ffi::c_int,
    vSize: ::core::ffi::c_int,
    qProjSize: ::core::ffi::c_int,
    kProjSize: ::core::ffi::c_int,
    vProjSize: ::core::ffi::c_int,
    oProjSize: ::core::ffi::c_int,
    qoMaxSeqLength: ::core::ffi::c_int,
    kvMaxSeqLength: ::core::ffi::c_int,
    maxBatchSize: ::core::ffi::c_int,
    maxBeamSize: ::core::ffi::c_int,
) -> cudnnStatus_t {
    let sym: Symbol<cudnnSetAttnDescriptor_fn> = unsafe {
        get_lib()
            .get(b"cudnnSetAttnDescriptor\0")
            .expect("Missing symbol")
    };
    unsafe {
        (*sym)(
            attnDesc,
            attnMode,
            nHeads,
            smScaler,
            dataType,
            computePrec,
            mathType,
            attnDropoutDesc,
            postDropoutDesc,
            qSize,
            kSize,
            vSize,
            qProjSize,
            kProjSize,
            vProjSize,
            oProjSize,
            qoMaxSeqLength,
            kvMaxSeqLength,
            maxBatchSize,
            maxBeamSize,
        )
    }
}

type cudnnGetAttnDescriptor_fn = unsafe extern "C" fn(
    attnDesc: cudnnAttnDescriptor_t,
    attnMode: *mut ::core::ffi::c_uint,
    nHeads: *mut ::core::ffi::c_int,
    smScaler: *mut f64,
    dataType: *mut cudnnDataType_t,
    computePrec: *mut cudnnDataType_t,
    mathType: *mut cudnnMathType_t,
    attnDropoutDesc: *mut cudnnDropoutDescriptor_t,
    postDropoutDesc: *mut cudnnDropoutDescriptor_t,
    qSize: *mut ::core::ffi::c_int,
    kSize: *mut ::core::ffi::c_int,
    vSize: *mut ::core::ffi::c_int,
    qProjSize: *mut ::core::ffi::c_int,
    kProjSize: *mut ::core::ffi::c_int,
    vProjSize: *mut ::core::ffi::c_int,
    oProjSize: *mut ::core::ffi::c_int,
    qoMaxSeqLength: *mut ::core::ffi::c_int,
    kvMaxSeqLength: *mut ::core::ffi::c_int,
    maxBatchSize: *mut ::core::ffi::c_int,
    maxBeamSize: *mut ::core::ffi::c_int,
) -> cudnnStatus_t;
pub unsafe fn cudnnGetAttnDescriptor(
    attnDesc: cudnnAttnDescriptor_t,
    attnMode: *mut ::core::ffi::c_uint,
    nHeads: *mut ::core::ffi::c_int,
    smScaler: *mut f64,
    dataType: *mut cudnnDataType_t,
    computePrec: *mut cudnnDataType_t,
    mathType: *mut cudnnMathType_t,
    attnDropoutDesc: *mut cudnnDropoutDescriptor_t,
    postDropoutDesc: *mut cudnnDropoutDescriptor_t,
    qSize: *mut ::core::ffi::c_int,
    kSize: *mut ::core::ffi::c_int,
    vSize: *mut ::core::ffi::c_int,
    qProjSize: *mut ::core::ffi::c_int,
    kProjSize: *mut ::core::ffi::c_int,
    vProjSize: *mut ::core::ffi::c_int,
    oProjSize: *mut ::core::ffi::c_int,
    qoMaxSeqLength: *mut ::core::ffi::c_int,
    kvMaxSeqLength: *mut ::core::ffi::c_int,
    maxBatchSize: *mut ::core::ffi::c_int,
    maxBeamSize: *mut ::core::ffi::c_int,
) -> cudnnStatus_t {
    let sym: Symbol<cudnnGetAttnDescriptor_fn> = unsafe {
        get_lib()
            .get(b"cudnnGetAttnDescriptor\0")
            .expect("Missing symbol")
    };
    unsafe {
        (*sym)(
            attnDesc,
            attnMode,
            nHeads,
            smScaler,
            dataType,
            computePrec,
            mathType,
            attnDropoutDesc,
            postDropoutDesc,
            qSize,
            kSize,
            vSize,
            qProjSize,
            kProjSize,
            vProjSize,
            oProjSize,
            qoMaxSeqLength,
            kvMaxSeqLength,
            maxBatchSize,
            maxBeamSize,
        )
    }
}

type cudnnGetMultiHeadAttnBuffers_fn = unsafe extern "C" fn(
    handle: cudnnHandle_t,
    attnDesc: cudnnAttnDescriptor_t,
    weightSizeInBytes: *mut usize,
    workSpaceSizeInBytes: *mut usize,
    reserveSpaceSizeInBytes: *mut usize,
) -> cudnnStatus_t;
pub unsafe fn cudnnGetMultiHeadAttnBuffers(
    handle: cudnnHandle_t,
    attnDesc: cudnnAttnDescriptor_t,
    weightSizeInBytes: *mut usize,
    workSpaceSizeInBytes: *mut usize,
    reserveSpaceSizeInBytes: *mut usize,
) -> cudnnStatus_t {
    let sym: Symbol<cudnnGetMultiHeadAttnBuffers_fn> = unsafe {
        get_lib()
            .get(b"cudnnGetMultiHeadAttnBuffers\0")
            .expect("Missing symbol")
    };
    unsafe {
        (*sym)(
            handle,
            attnDesc,
            weightSizeInBytes,
            workSpaceSizeInBytes,
            reserveSpaceSizeInBytes,
        )
    }
}

type cudnnGetMultiHeadAttnWeights_fn = unsafe extern "C" fn(
    handle: cudnnHandle_t,
    attnDesc: cudnnAttnDescriptor_t,
    wKind: cudnnMultiHeadAttnWeightKind_t,
    weightSizeInBytes: usize,
    weights: *const ::core::ffi::c_void,
    wDesc: cudnnTensorDescriptor_t,
    wAddr: *mut *mut ::core::ffi::c_void,
) -> cudnnStatus_t;
pub unsafe fn cudnnGetMultiHeadAttnWeights(
    handle: cudnnHandle_t,
    attnDesc: cudnnAttnDescriptor_t,
    wKind: cudnnMultiHeadAttnWeightKind_t,
    weightSizeInBytes: usize,
    weights: *const ::core::ffi::c_void,
    wDesc: cudnnTensorDescriptor_t,
    wAddr: *mut *mut ::core::ffi::c_void,
) -> cudnnStatus_t {
    let sym: Symbol<cudnnGetMultiHeadAttnWeights_fn> = unsafe {
        get_lib()
            .get(b"cudnnGetMultiHeadAttnWeights\0")
            .expect("Missing symbol")
    };
    unsafe {
        (*sym)(
            handle,
            attnDesc,
            wKind,
            weightSizeInBytes,
            weights,
            wDesc,
            wAddr,
        )
    }
}

type cudnnMultiHeadAttnForward_fn = unsafe extern "C" fn(
    handle: cudnnHandle_t,
    attnDesc: cudnnAttnDescriptor_t,
    currIdx: ::core::ffi::c_int,
    loWinIdx: *const ::core::ffi::c_int,
    hiWinIdx: *const ::core::ffi::c_int,
    devSeqLengthsQO: *const ::core::ffi::c_int,
    devSeqLengthsKV: *const ::core::ffi::c_int,
    qDesc: cudnnSeqDataDescriptor_t,
    queries: *const ::core::ffi::c_void,
    residuals: *const ::core::ffi::c_void,
    kDesc: cudnnSeqDataDescriptor_t,
    keys: *const ::core::ffi::c_void,
    vDesc: cudnnSeqDataDescriptor_t,
    values: *const ::core::ffi::c_void,
    oDesc: cudnnSeqDataDescriptor_t,
    out: *mut ::core::ffi::c_void,
    weightSizeInBytes: usize,
    weights: *const ::core::ffi::c_void,
    workSpaceSizeInBytes: usize,
    workSpace: *mut ::core::ffi::c_void,
    reserveSpaceSizeInBytes: usize,
    reserveSpace: *mut ::core::ffi::c_void,
) -> cudnnStatus_t;
pub unsafe fn cudnnMultiHeadAttnForward(
    handle: cudnnHandle_t,
    attnDesc: cudnnAttnDescriptor_t,
    currIdx: ::core::ffi::c_int,
    loWinIdx: *const ::core::ffi::c_int,
    hiWinIdx: *const ::core::ffi::c_int,
    devSeqLengthsQO: *const ::core::ffi::c_int,
    devSeqLengthsKV: *const ::core::ffi::c_int,
    qDesc: cudnnSeqDataDescriptor_t,
    queries: *const ::core::ffi::c_void,
    residuals: *const ::core::ffi::c_void,
    kDesc: cudnnSeqDataDescriptor_t,
    keys: *const ::core::ffi::c_void,
    vDesc: cudnnSeqDataDescriptor_t,
    values: *const ::core::ffi::c_void,
    oDesc: cudnnSeqDataDescriptor_t,
    out: *mut ::core::ffi::c_void,
    weightSizeInBytes: usize,
    weights: *const ::core::ffi::c_void,
    workSpaceSizeInBytes: usize,
    workSpace: *mut ::core::ffi::c_void,
    reserveSpaceSizeInBytes: usize,
    reserveSpace: *mut ::core::ffi::c_void,
) -> cudnnStatus_t {
    let sym: Symbol<cudnnMultiHeadAttnForward_fn> = unsafe {
        get_lib()
            .get(b"cudnnMultiHeadAttnForward\0")
            .expect("Missing symbol")
    };
    unsafe {
        (*sym)(
            handle,
            attnDesc,
            currIdx,
            loWinIdx,
            hiWinIdx,
            devSeqLengthsQO,
            devSeqLengthsKV,
            qDesc,
            queries,
            residuals,
            kDesc,
            keys,
            vDesc,
            values,
            oDesc,
            out,
            weightSizeInBytes,
            weights,
            workSpaceSizeInBytes,
            workSpace,
            reserveSpaceSizeInBytes,
            reserveSpace,
        )
    }
}

type cudnnAdvVersionCheck_fn = unsafe extern "C" fn() -> cudnnStatus_t;
pub unsafe fn cudnnAdvVersionCheck() -> cudnnStatus_t {
    let sym: Symbol<cudnnAdvVersionCheck_fn> = unsafe {
        get_lib()
            .get(b"cudnnAdvVersionCheck\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)() }
}

type cudnnRNNBackwardData_v8_fn = unsafe extern "C" fn(
    handle: cudnnHandle_t,
    rnnDesc: cudnnRNNDescriptor_t,
    devSeqLengths: *const i32,
    yDesc: cudnnRNNDataDescriptor_t,
    y: *const ::core::ffi::c_void,
    dy: *const ::core::ffi::c_void,
    xDesc: cudnnRNNDataDescriptor_t,
    dx: *mut ::core::ffi::c_void,
    hDesc: cudnnTensorDescriptor_t,
    hx: *const ::core::ffi::c_void,
    dhy: *const ::core::ffi::c_void,
    dhx: *mut ::core::ffi::c_void,
    cDesc: cudnnTensorDescriptor_t,
    cx: *const ::core::ffi::c_void,
    dcy: *const ::core::ffi::c_void,
    dcx: *mut ::core::ffi::c_void,
    weightSpaceSize: usize,
    weightSpace: *const ::core::ffi::c_void,
    workSpaceSize: usize,
    workSpace: *mut ::core::ffi::c_void,
    reserveSpaceSize: usize,
    reserveSpace: *mut ::core::ffi::c_void,
) -> cudnnStatus_t;
pub unsafe fn cudnnRNNBackwardData_v8(
    handle: cudnnHandle_t,
    rnnDesc: cudnnRNNDescriptor_t,
    devSeqLengths: *const i32,
    yDesc: cudnnRNNDataDescriptor_t,
    y: *const ::core::ffi::c_void,
    dy: *const ::core::ffi::c_void,
    xDesc: cudnnRNNDataDescriptor_t,
    dx: *mut ::core::ffi::c_void,
    hDesc: cudnnTensorDescriptor_t,
    hx: *const ::core::ffi::c_void,
    dhy: *const ::core::ffi::c_void,
    dhx: *mut ::core::ffi::c_void,
    cDesc: cudnnTensorDescriptor_t,
    cx: *const ::core::ffi::c_void,
    dcy: *const ::core::ffi::c_void,
    dcx: *mut ::core::ffi::c_void,
    weightSpaceSize: usize,
    weightSpace: *const ::core::ffi::c_void,
    workSpaceSize: usize,
    workSpace: *mut ::core::ffi::c_void,
    reserveSpaceSize: usize,
    reserveSpace: *mut ::core::ffi::c_void,
) -> cudnnStatus_t {
    let sym: Symbol<cudnnRNNBackwardData_v8_fn> = unsafe {
        get_lib()
            .get(b"cudnnRNNBackwardData_v8\0")
            .expect("Missing symbol")
    };
    unsafe {
        (*sym)(
            handle,
            rnnDesc,
            devSeqLengths,
            yDesc,
            y,
            dy,
            xDesc,
            dx,
            hDesc,
            hx,
            dhy,
            dhx,
            cDesc,
            cx,
            dcy,
            dcx,
            weightSpaceSize,
            weightSpace,
            workSpaceSize,
            workSpace,
            reserveSpaceSize,
            reserveSpace,
        )
    }
}

type cudnnRNNBackwardWeights_v8_fn = unsafe extern "C" fn(
    handle: cudnnHandle_t,
    rnnDesc: cudnnRNNDescriptor_t,
    addGrad: cudnnWgradMode_t,
    devSeqLengths: *const i32,
    xDesc: cudnnRNNDataDescriptor_t,
    x: *const ::core::ffi::c_void,
    hDesc: cudnnTensorDescriptor_t,
    hx: *const ::core::ffi::c_void,
    yDesc: cudnnRNNDataDescriptor_t,
    y: *const ::core::ffi::c_void,
    weightSpaceSize: usize,
    dweightSpace: *mut ::core::ffi::c_void,
    workSpaceSize: usize,
    workSpace: *mut ::core::ffi::c_void,
    reserveSpaceSize: usize,
    reserveSpace: *mut ::core::ffi::c_void,
) -> cudnnStatus_t;
pub unsafe fn cudnnRNNBackwardWeights_v8(
    handle: cudnnHandle_t,
    rnnDesc: cudnnRNNDescriptor_t,
    addGrad: cudnnWgradMode_t,
    devSeqLengths: *const i32,
    xDesc: cudnnRNNDataDescriptor_t,
    x: *const ::core::ffi::c_void,
    hDesc: cudnnTensorDescriptor_t,
    hx: *const ::core::ffi::c_void,
    yDesc: cudnnRNNDataDescriptor_t,
    y: *const ::core::ffi::c_void,
    weightSpaceSize: usize,
    dweightSpace: *mut ::core::ffi::c_void,
    workSpaceSize: usize,
    workSpace: *mut ::core::ffi::c_void,
    reserveSpaceSize: usize,
    reserveSpace: *mut ::core::ffi::c_void,
) -> cudnnStatus_t {
    let sym: Symbol<cudnnRNNBackwardWeights_v8_fn> = unsafe {
        get_lib()
            .get(b"cudnnRNNBackwardWeights_v8\0")
            .expect("Missing symbol")
    };
    unsafe {
        (*sym)(
            handle,
            rnnDesc,
            addGrad,
            devSeqLengths,
            xDesc,
            x,
            hDesc,
            hx,
            yDesc,
            y,
            weightSpaceSize,
            dweightSpace,
            workSpaceSize,
            workSpace,
            reserveSpaceSize,
            reserveSpace,
        )
    }
}

type cudnnMultiHeadAttnBackwardData_fn = unsafe extern "C" fn(
    handle: cudnnHandle_t,
    attnDesc: cudnnAttnDescriptor_t,
    loWinIdx: *const ::core::ffi::c_int,
    hiWinIdx: *const ::core::ffi::c_int,
    devSeqLengthsDQDO: *const ::core::ffi::c_int,
    devSeqLengthsDKDV: *const ::core::ffi::c_int,
    doDesc: cudnnSeqDataDescriptor_t,
    dout: *const ::core::ffi::c_void,
    dqDesc: cudnnSeqDataDescriptor_t,
    dqueries: *mut ::core::ffi::c_void,
    queries: *const ::core::ffi::c_void,
    dkDesc: cudnnSeqDataDescriptor_t,
    dkeys: *mut ::core::ffi::c_void,
    keys: *const ::core::ffi::c_void,
    dvDesc: cudnnSeqDataDescriptor_t,
    dvalues: *mut ::core::ffi::c_void,
    values: *const ::core::ffi::c_void,
    weightSizeInBytes: usize,
    weights: *const ::core::ffi::c_void,
    workSpaceSizeInBytes: usize,
    workSpace: *mut ::core::ffi::c_void,
    reserveSpaceSizeInBytes: usize,
    reserveSpace: *mut ::core::ffi::c_void,
) -> cudnnStatus_t;
pub unsafe fn cudnnMultiHeadAttnBackwardData(
    handle: cudnnHandle_t,
    attnDesc: cudnnAttnDescriptor_t,
    loWinIdx: *const ::core::ffi::c_int,
    hiWinIdx: *const ::core::ffi::c_int,
    devSeqLengthsDQDO: *const ::core::ffi::c_int,
    devSeqLengthsDKDV: *const ::core::ffi::c_int,
    doDesc: cudnnSeqDataDescriptor_t,
    dout: *const ::core::ffi::c_void,
    dqDesc: cudnnSeqDataDescriptor_t,
    dqueries: *mut ::core::ffi::c_void,
    queries: *const ::core::ffi::c_void,
    dkDesc: cudnnSeqDataDescriptor_t,
    dkeys: *mut ::core::ffi::c_void,
    keys: *const ::core::ffi::c_void,
    dvDesc: cudnnSeqDataDescriptor_t,
    dvalues: *mut ::core::ffi::c_void,
    values: *const ::core::ffi::c_void,
    weightSizeInBytes: usize,
    weights: *const ::core::ffi::c_void,
    workSpaceSizeInBytes: usize,
    workSpace: *mut ::core::ffi::c_void,
    reserveSpaceSizeInBytes: usize,
    reserveSpace: *mut ::core::ffi::c_void,
) -> cudnnStatus_t {
    let sym: Symbol<cudnnMultiHeadAttnBackwardData_fn> = unsafe {
        get_lib()
            .get(b"cudnnMultiHeadAttnBackwardData\0")
            .expect("Missing symbol")
    };
    unsafe {
        (*sym)(
            handle,
            attnDesc,
            loWinIdx,
            hiWinIdx,
            devSeqLengthsDQDO,
            devSeqLengthsDKDV,
            doDesc,
            dout,
            dqDesc,
            dqueries,
            queries,
            dkDesc,
            dkeys,
            keys,
            dvDesc,
            dvalues,
            values,
            weightSizeInBytes,
            weights,
            workSpaceSizeInBytes,
            workSpace,
            reserveSpaceSizeInBytes,
            reserveSpace,
        )
    }
}

type cudnnMultiHeadAttnBackwardWeights_fn = unsafe extern "C" fn(
    handle: cudnnHandle_t,
    attnDesc: cudnnAttnDescriptor_t,
    addGrad: cudnnWgradMode_t,
    qDesc: cudnnSeqDataDescriptor_t,
    queries: *const ::core::ffi::c_void,
    kDesc: cudnnSeqDataDescriptor_t,
    keys: *const ::core::ffi::c_void,
    vDesc: cudnnSeqDataDescriptor_t,
    values: *const ::core::ffi::c_void,
    doDesc: cudnnSeqDataDescriptor_t,
    dout: *const ::core::ffi::c_void,
    weightSizeInBytes: usize,
    weights: *const ::core::ffi::c_void,
    dweights: *mut ::core::ffi::c_void,
    workSpaceSizeInBytes: usize,
    workSpace: *mut ::core::ffi::c_void,
    reserveSpaceSizeInBytes: usize,
    reserveSpace: *mut ::core::ffi::c_void,
) -> cudnnStatus_t;
pub unsafe fn cudnnMultiHeadAttnBackwardWeights(
    handle: cudnnHandle_t,
    attnDesc: cudnnAttnDescriptor_t,
    addGrad: cudnnWgradMode_t,
    qDesc: cudnnSeqDataDescriptor_t,
    queries: *const ::core::ffi::c_void,
    kDesc: cudnnSeqDataDescriptor_t,
    keys: *const ::core::ffi::c_void,
    vDesc: cudnnSeqDataDescriptor_t,
    values: *const ::core::ffi::c_void,
    doDesc: cudnnSeqDataDescriptor_t,
    dout: *const ::core::ffi::c_void,
    weightSizeInBytes: usize,
    weights: *const ::core::ffi::c_void,
    dweights: *mut ::core::ffi::c_void,
    workSpaceSizeInBytes: usize,
    workSpace: *mut ::core::ffi::c_void,
    reserveSpaceSizeInBytes: usize,
    reserveSpace: *mut ::core::ffi::c_void,
) -> cudnnStatus_t {
    let sym: Symbol<cudnnMultiHeadAttnBackwardWeights_fn> = unsafe {
        get_lib()
            .get(b"cudnnMultiHeadAttnBackwardWeights\0")
            .expect("Missing symbol")
    };
    unsafe {
        (*sym)(
            handle,
            attnDesc,
            addGrad,
            qDesc,
            queries,
            kDesc,
            keys,
            vDesc,
            values,
            doDesc,
            dout,
            weightSizeInBytes,
            weights,
            dweights,
            workSpaceSizeInBytes,
            workSpace,
            reserveSpaceSizeInBytes,
            reserveSpace,
        )
    }
}

type cudnnCreateCTCLossDescriptor_fn =
    unsafe extern "C" fn(ctcLossDesc: *mut cudnnCTCLossDescriptor_t) -> cudnnStatus_t;
pub unsafe fn cudnnCreateCTCLossDescriptor(
    ctcLossDesc: *mut cudnnCTCLossDescriptor_t,
) -> cudnnStatus_t {
    let sym: Symbol<cudnnCreateCTCLossDescriptor_fn> = unsafe {
        get_lib()
            .get(b"cudnnCreateCTCLossDescriptor\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(ctcLossDesc) }
}

type cudnnSetCTCLossDescriptor_fn = unsafe extern "C" fn(
    ctcLossDesc: cudnnCTCLossDescriptor_t,
    compType: cudnnDataType_t,
) -> cudnnStatus_t;
pub unsafe fn cudnnSetCTCLossDescriptor(
    ctcLossDesc: cudnnCTCLossDescriptor_t,
    compType: cudnnDataType_t,
) -> cudnnStatus_t {
    let sym: Symbol<cudnnSetCTCLossDescriptor_fn> = unsafe {
        get_lib()
            .get(b"cudnnSetCTCLossDescriptor\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(ctcLossDesc, compType) }
}

type cudnnSetCTCLossDescriptorEx_fn = unsafe extern "C" fn(
    ctcLossDesc: cudnnCTCLossDescriptor_t,
    compType: cudnnDataType_t,
    normMode: cudnnLossNormalizationMode_t,
    gradMode: cudnnNanPropagation_t,
) -> cudnnStatus_t;
pub unsafe fn cudnnSetCTCLossDescriptorEx(
    ctcLossDesc: cudnnCTCLossDescriptor_t,
    compType: cudnnDataType_t,
    normMode: cudnnLossNormalizationMode_t,
    gradMode: cudnnNanPropagation_t,
) -> cudnnStatus_t {
    let sym: Symbol<cudnnSetCTCLossDescriptorEx_fn> = unsafe {
        get_lib()
            .get(b"cudnnSetCTCLossDescriptorEx\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(ctcLossDesc, compType, normMode, gradMode) }
}

type cudnnSetCTCLossDescriptor_v8_fn = unsafe extern "C" fn(
    ctcLossDesc: cudnnCTCLossDescriptor_t,
    compType: cudnnDataType_t,
    normMode: cudnnLossNormalizationMode_t,
    gradMode: cudnnNanPropagation_t,
    maxLabelLength: ::core::ffi::c_int,
) -> cudnnStatus_t;
pub unsafe fn cudnnSetCTCLossDescriptor_v8(
    ctcLossDesc: cudnnCTCLossDescriptor_t,
    compType: cudnnDataType_t,
    normMode: cudnnLossNormalizationMode_t,
    gradMode: cudnnNanPropagation_t,
    maxLabelLength: ::core::ffi::c_int,
) -> cudnnStatus_t {
    let sym: Symbol<cudnnSetCTCLossDescriptor_v8_fn> = unsafe {
        get_lib()
            .get(b"cudnnSetCTCLossDescriptor_v8\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(ctcLossDesc, compType, normMode, gradMode, maxLabelLength) }
}

type cudnnSetCTCLossDescriptor_v9_fn = unsafe extern "C" fn(
    ctcLossDesc: cudnnCTCLossDescriptor_t,
    compType: cudnnDataType_t,
    normMode: cudnnLossNormalizationMode_t,
    ctcGradMode: cudnnCTCGradMode_t,
    maxLabelLength: ::core::ffi::c_int,
) -> cudnnStatus_t;
pub unsafe fn cudnnSetCTCLossDescriptor_v9(
    ctcLossDesc: cudnnCTCLossDescriptor_t,
    compType: cudnnDataType_t,
    normMode: cudnnLossNormalizationMode_t,
    ctcGradMode: cudnnCTCGradMode_t,
    maxLabelLength: ::core::ffi::c_int,
) -> cudnnStatus_t {
    let sym: Symbol<cudnnSetCTCLossDescriptor_v9_fn> = unsafe {
        get_lib()
            .get(b"cudnnSetCTCLossDescriptor_v9\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(ctcLossDesc, compType, normMode, ctcGradMode, maxLabelLength) }
}

type cudnnGetCTCLossDescriptor_fn = unsafe extern "C" fn(
    ctcLossDesc: cudnnCTCLossDescriptor_t,
    compType: *mut cudnnDataType_t,
) -> cudnnStatus_t;
pub unsafe fn cudnnGetCTCLossDescriptor(
    ctcLossDesc: cudnnCTCLossDescriptor_t,
    compType: *mut cudnnDataType_t,
) -> cudnnStatus_t {
    let sym: Symbol<cudnnGetCTCLossDescriptor_fn> = unsafe {
        get_lib()
            .get(b"cudnnGetCTCLossDescriptor\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(ctcLossDesc, compType) }
}

type cudnnGetCTCLossDescriptorEx_fn = unsafe extern "C" fn(
    ctcLossDesc: cudnnCTCLossDescriptor_t,
    compType: *mut cudnnDataType_t,
    normMode: *mut cudnnLossNormalizationMode_t,
    gradMode: *mut cudnnNanPropagation_t,
) -> cudnnStatus_t;
pub unsafe fn cudnnGetCTCLossDescriptorEx(
    ctcLossDesc: cudnnCTCLossDescriptor_t,
    compType: *mut cudnnDataType_t,
    normMode: *mut cudnnLossNormalizationMode_t,
    gradMode: *mut cudnnNanPropagation_t,
) -> cudnnStatus_t {
    let sym: Symbol<cudnnGetCTCLossDescriptorEx_fn> = unsafe {
        get_lib()
            .get(b"cudnnGetCTCLossDescriptorEx\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(ctcLossDesc, compType, normMode, gradMode) }
}

type cudnnGetCTCLossDescriptor_v8_fn = unsafe extern "C" fn(
    ctcLossDesc: cudnnCTCLossDescriptor_t,
    compType: *mut cudnnDataType_t,
    normMode: *mut cudnnLossNormalizationMode_t,
    gradMode: *mut cudnnNanPropagation_t,
    maxLabelLength: *mut ::core::ffi::c_int,
) -> cudnnStatus_t;
pub unsafe fn cudnnGetCTCLossDescriptor_v8(
    ctcLossDesc: cudnnCTCLossDescriptor_t,
    compType: *mut cudnnDataType_t,
    normMode: *mut cudnnLossNormalizationMode_t,
    gradMode: *mut cudnnNanPropagation_t,
    maxLabelLength: *mut ::core::ffi::c_int,
) -> cudnnStatus_t {
    let sym: Symbol<cudnnGetCTCLossDescriptor_v8_fn> = unsafe {
        get_lib()
            .get(b"cudnnGetCTCLossDescriptor_v8\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(ctcLossDesc, compType, normMode, gradMode, maxLabelLength) }
}

type cudnnGetCTCLossDescriptor_v9_fn = unsafe extern "C" fn(
    ctcLossDesc: cudnnCTCLossDescriptor_t,
    compType: *mut cudnnDataType_t,
    normMode: *mut cudnnLossNormalizationMode_t,
    ctcGradMode: *mut cudnnCTCGradMode_t,
    maxLabelLength: *mut ::core::ffi::c_int,
) -> cudnnStatus_t;
pub unsafe fn cudnnGetCTCLossDescriptor_v9(
    ctcLossDesc: cudnnCTCLossDescriptor_t,
    compType: *mut cudnnDataType_t,
    normMode: *mut cudnnLossNormalizationMode_t,
    ctcGradMode: *mut cudnnCTCGradMode_t,
    maxLabelLength: *mut ::core::ffi::c_int,
) -> cudnnStatus_t {
    let sym: Symbol<cudnnGetCTCLossDescriptor_v9_fn> = unsafe {
        get_lib()
            .get(b"cudnnGetCTCLossDescriptor_v9\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(ctcLossDesc, compType, normMode, ctcGradMode, maxLabelLength) }
}

type cudnnDestroyCTCLossDescriptor_fn =
    unsafe extern "C" fn(ctcLossDesc: cudnnCTCLossDescriptor_t) -> cudnnStatus_t;
pub unsafe fn cudnnDestroyCTCLossDescriptor(
    ctcLossDesc: cudnnCTCLossDescriptor_t,
) -> cudnnStatus_t {
    let sym: Symbol<cudnnDestroyCTCLossDescriptor_fn> = unsafe {
        get_lib()
            .get(b"cudnnDestroyCTCLossDescriptor\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(ctcLossDesc) }
}

type cudnnCTCLoss_fn = unsafe extern "C" fn(
    handle: cudnnHandle_t,
    probsDesc: cudnnTensorDescriptor_t,
    probs: *const ::core::ffi::c_void,
    hostLabels: *const ::core::ffi::c_int,
    hostLabelLengths: *const ::core::ffi::c_int,
    hostInputLengths: *const ::core::ffi::c_int,
    costs: *mut ::core::ffi::c_void,
    gradientsDesc: cudnnTensorDescriptor_t,
    gradients: *mut ::core::ffi::c_void,
    algo: cudnnCTCLossAlgo_t,
    ctcLossDesc: cudnnCTCLossDescriptor_t,
    workspace: *mut ::core::ffi::c_void,
    workSpaceSizeInBytes: usize,
) -> cudnnStatus_t;
pub unsafe fn cudnnCTCLoss(
    handle: cudnnHandle_t,
    probsDesc: cudnnTensorDescriptor_t,
    probs: *const ::core::ffi::c_void,
    hostLabels: *const ::core::ffi::c_int,
    hostLabelLengths: *const ::core::ffi::c_int,
    hostInputLengths: *const ::core::ffi::c_int,
    costs: *mut ::core::ffi::c_void,
    gradientsDesc: cudnnTensorDescriptor_t,
    gradients: *mut ::core::ffi::c_void,
    algo: cudnnCTCLossAlgo_t,
    ctcLossDesc: cudnnCTCLossDescriptor_t,
    workspace: *mut ::core::ffi::c_void,
    workSpaceSizeInBytes: usize,
) -> cudnnStatus_t {
    let sym: Symbol<cudnnCTCLoss_fn> =
        unsafe { get_lib().get(b"cudnnCTCLoss\0").expect("Missing symbol") };
    unsafe {
        (*sym)(
            handle,
            probsDesc,
            probs,
            hostLabels,
            hostLabelLengths,
            hostInputLengths,
            costs,
            gradientsDesc,
            gradients,
            algo,
            ctcLossDesc,
            workspace,
            workSpaceSizeInBytes,
        )
    }
}

type cudnnCTCLoss_v8_fn = unsafe extern "C" fn(
    handle: cudnnHandle_t,
    algo: cudnnCTCLossAlgo_t,
    ctcLossDesc: cudnnCTCLossDescriptor_t,
    probsDesc: cudnnTensorDescriptor_t,
    probs: *const ::core::ffi::c_void,
    labels: *const ::core::ffi::c_int,
    labelLengths: *const ::core::ffi::c_int,
    inputLengths: *const ::core::ffi::c_int,
    costs: *mut ::core::ffi::c_void,
    gradientsDesc: cudnnTensorDescriptor_t,
    gradients: *mut ::core::ffi::c_void,
    workSpaceSizeInBytes: usize,
    workspace: *mut ::core::ffi::c_void,
) -> cudnnStatus_t;
pub unsafe fn cudnnCTCLoss_v8(
    handle: cudnnHandle_t,
    algo: cudnnCTCLossAlgo_t,
    ctcLossDesc: cudnnCTCLossDescriptor_t,
    probsDesc: cudnnTensorDescriptor_t,
    probs: *const ::core::ffi::c_void,
    labels: *const ::core::ffi::c_int,
    labelLengths: *const ::core::ffi::c_int,
    inputLengths: *const ::core::ffi::c_int,
    costs: *mut ::core::ffi::c_void,
    gradientsDesc: cudnnTensorDescriptor_t,
    gradients: *mut ::core::ffi::c_void,
    workSpaceSizeInBytes: usize,
    workspace: *mut ::core::ffi::c_void,
) -> cudnnStatus_t {
    let sym: Symbol<cudnnCTCLoss_v8_fn> =
        unsafe { get_lib().get(b"cudnnCTCLoss_v8\0").expect("Missing symbol") };
    unsafe {
        (*sym)(
            handle,
            algo,
            ctcLossDesc,
            probsDesc,
            probs,
            labels,
            labelLengths,
            inputLengths,
            costs,
            gradientsDesc,
            gradients,
            workSpaceSizeInBytes,
            workspace,
        )
    }
}

type cudnnGetCTCLossWorkspaceSize_fn = unsafe extern "C" fn(
    handle: cudnnHandle_t,
    probsDesc: cudnnTensorDescriptor_t,
    gradientsDesc: cudnnTensorDescriptor_t,
    labels: *const ::core::ffi::c_int,
    labelLengths: *const ::core::ffi::c_int,
    inputLengths: *const ::core::ffi::c_int,
    algo: cudnnCTCLossAlgo_t,
    ctcLossDesc: cudnnCTCLossDescriptor_t,
    sizeInBytes: *mut usize,
) -> cudnnStatus_t;
pub unsafe fn cudnnGetCTCLossWorkspaceSize(
    handle: cudnnHandle_t,
    probsDesc: cudnnTensorDescriptor_t,
    gradientsDesc: cudnnTensorDescriptor_t,
    labels: *const ::core::ffi::c_int,
    labelLengths: *const ::core::ffi::c_int,
    inputLengths: *const ::core::ffi::c_int,
    algo: cudnnCTCLossAlgo_t,
    ctcLossDesc: cudnnCTCLossDescriptor_t,
    sizeInBytes: *mut usize,
) -> cudnnStatus_t {
    let sym: Symbol<cudnnGetCTCLossWorkspaceSize_fn> = unsafe {
        get_lib()
            .get(b"cudnnGetCTCLossWorkspaceSize\0")
            .expect("Missing symbol")
    };
    unsafe {
        (*sym)(
            handle,
            probsDesc,
            gradientsDesc,
            labels,
            labelLengths,
            inputLengths,
            algo,
            ctcLossDesc,
            sizeInBytes,
        )
    }
}

type cudnnGetCTCLossWorkspaceSize_v8_fn = unsafe extern "C" fn(
    handle: cudnnHandle_t,
    algo: cudnnCTCLossAlgo_t,
    ctcLossDesc: cudnnCTCLossDescriptor_t,
    probsDesc: cudnnTensorDescriptor_t,
    gradientsDesc: cudnnTensorDescriptor_t,
    sizeInBytes: *mut usize,
) -> cudnnStatus_t;
pub unsafe fn cudnnGetCTCLossWorkspaceSize_v8(
    handle: cudnnHandle_t,
    algo: cudnnCTCLossAlgo_t,
    ctcLossDesc: cudnnCTCLossDescriptor_t,
    probsDesc: cudnnTensorDescriptor_t,
    gradientsDesc: cudnnTensorDescriptor_t,
    sizeInBytes: *mut usize,
) -> cudnnStatus_t {
    let sym: Symbol<cudnnGetCTCLossWorkspaceSize_v8_fn> = unsafe {
        get_lib()
            .get(b"cudnnGetCTCLossWorkspaceSize_v8\0")
            .expect("Missing symbol")
    };
    unsafe {
        (*sym)(
            handle,
            algo,
            ctcLossDesc,
            probsDesc,
            gradientsDesc,
            sizeInBytes,
        )
    }
}

type cudnnCreateConvolutionDescriptor_fn =
    unsafe extern "C" fn(convDesc: *mut cudnnConvolutionDescriptor_t) -> cudnnStatus_t;
pub unsafe fn cudnnCreateConvolutionDescriptor(
    convDesc: *mut cudnnConvolutionDescriptor_t,
) -> cudnnStatus_t {
    let sym: Symbol<cudnnCreateConvolutionDescriptor_fn> = unsafe {
        get_lib()
            .get(b"cudnnCreateConvolutionDescriptor\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(convDesc) }
}

type cudnnDestroyConvolutionDescriptor_fn =
    unsafe extern "C" fn(convDesc: cudnnConvolutionDescriptor_t) -> cudnnStatus_t;
pub unsafe fn cudnnDestroyConvolutionDescriptor(
    convDesc: cudnnConvolutionDescriptor_t,
) -> cudnnStatus_t {
    let sym: Symbol<cudnnDestroyConvolutionDescriptor_fn> = unsafe {
        get_lib()
            .get(b"cudnnDestroyConvolutionDescriptor\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(convDesc) }
}

type cudnnSetConvolutionMathType_fn = unsafe extern "C" fn(
    convDesc: cudnnConvolutionDescriptor_t,
    mathType: cudnnMathType_t,
) -> cudnnStatus_t;
pub unsafe fn cudnnSetConvolutionMathType(
    convDesc: cudnnConvolutionDescriptor_t,
    mathType: cudnnMathType_t,
) -> cudnnStatus_t {
    let sym: Symbol<cudnnSetConvolutionMathType_fn> = unsafe {
        get_lib()
            .get(b"cudnnSetConvolutionMathType\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(convDesc, mathType) }
}

type cudnnGetConvolutionMathType_fn = unsafe extern "C" fn(
    convDesc: cudnnConvolutionDescriptor_t,
    mathType: *mut cudnnMathType_t,
) -> cudnnStatus_t;
pub unsafe fn cudnnGetConvolutionMathType(
    convDesc: cudnnConvolutionDescriptor_t,
    mathType: *mut cudnnMathType_t,
) -> cudnnStatus_t {
    let sym: Symbol<cudnnGetConvolutionMathType_fn> = unsafe {
        get_lib()
            .get(b"cudnnGetConvolutionMathType\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(convDesc, mathType) }
}

type cudnnSetConvolutionGroupCount_fn = unsafe extern "C" fn(
    convDesc: cudnnConvolutionDescriptor_t,
    groupCount: ::core::ffi::c_int,
) -> cudnnStatus_t;
pub unsafe fn cudnnSetConvolutionGroupCount(
    convDesc: cudnnConvolutionDescriptor_t,
    groupCount: ::core::ffi::c_int,
) -> cudnnStatus_t {
    let sym: Symbol<cudnnSetConvolutionGroupCount_fn> = unsafe {
        get_lib()
            .get(b"cudnnSetConvolutionGroupCount\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(convDesc, groupCount) }
}

type cudnnGetConvolutionGroupCount_fn = unsafe extern "C" fn(
    convDesc: cudnnConvolutionDescriptor_t,
    groupCount: *mut ::core::ffi::c_int,
) -> cudnnStatus_t;
pub unsafe fn cudnnGetConvolutionGroupCount(
    convDesc: cudnnConvolutionDescriptor_t,
    groupCount: *mut ::core::ffi::c_int,
) -> cudnnStatus_t {
    let sym: Symbol<cudnnGetConvolutionGroupCount_fn> = unsafe {
        get_lib()
            .get(b"cudnnGetConvolutionGroupCount\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(convDesc, groupCount) }
}

type cudnnSetConvolutionReorderType_fn = unsafe extern "C" fn(
    convDesc: cudnnConvolutionDescriptor_t,
    reorderType: cudnnReorderType_t,
) -> cudnnStatus_t;
pub unsafe fn cudnnSetConvolutionReorderType(
    convDesc: cudnnConvolutionDescriptor_t,
    reorderType: cudnnReorderType_t,
) -> cudnnStatus_t {
    let sym: Symbol<cudnnSetConvolutionReorderType_fn> = unsafe {
        get_lib()
            .get(b"cudnnSetConvolutionReorderType\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(convDesc, reorderType) }
}

type cudnnGetConvolutionReorderType_fn = unsafe extern "C" fn(
    convDesc: cudnnConvolutionDescriptor_t,
    reorderType: *mut cudnnReorderType_t,
) -> cudnnStatus_t;
pub unsafe fn cudnnGetConvolutionReorderType(
    convDesc: cudnnConvolutionDescriptor_t,
    reorderType: *mut cudnnReorderType_t,
) -> cudnnStatus_t {
    let sym: Symbol<cudnnGetConvolutionReorderType_fn> = unsafe {
        get_lib()
            .get(b"cudnnGetConvolutionReorderType\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(convDesc, reorderType) }
}

type cudnnSetConvolution2dDescriptor_fn = unsafe extern "C" fn(
    convDesc: cudnnConvolutionDescriptor_t,
    pad_h: ::core::ffi::c_int,
    pad_w: ::core::ffi::c_int,
    u: ::core::ffi::c_int,
    v: ::core::ffi::c_int,
    dilation_h: ::core::ffi::c_int,
    dilation_w: ::core::ffi::c_int,
    mode: cudnnConvolutionMode_t,
    computeType: cudnnDataType_t,
) -> cudnnStatus_t;
pub unsafe fn cudnnSetConvolution2dDescriptor(
    convDesc: cudnnConvolutionDescriptor_t,
    pad_h: ::core::ffi::c_int,
    pad_w: ::core::ffi::c_int,
    u: ::core::ffi::c_int,
    v: ::core::ffi::c_int,
    dilation_h: ::core::ffi::c_int,
    dilation_w: ::core::ffi::c_int,
    mode: cudnnConvolutionMode_t,
    computeType: cudnnDataType_t,
) -> cudnnStatus_t {
    let sym: Symbol<cudnnSetConvolution2dDescriptor_fn> = unsafe {
        get_lib()
            .get(b"cudnnSetConvolution2dDescriptor\0")
            .expect("Missing symbol")
    };
    unsafe {
        (*sym)(
            convDesc,
            pad_h,
            pad_w,
            u,
            v,
            dilation_h,
            dilation_w,
            mode,
            computeType,
        )
    }
}

type cudnnGetConvolution2dDescriptor_fn = unsafe extern "C" fn(
    convDesc: cudnnConvolutionDescriptor_t,
    pad_h: *mut ::core::ffi::c_int,
    pad_w: *mut ::core::ffi::c_int,
    u: *mut ::core::ffi::c_int,
    v: *mut ::core::ffi::c_int,
    dilation_h: *mut ::core::ffi::c_int,
    dilation_w: *mut ::core::ffi::c_int,
    mode: *mut cudnnConvolutionMode_t,
    computeType: *mut cudnnDataType_t,
) -> cudnnStatus_t;
pub unsafe fn cudnnGetConvolution2dDescriptor(
    convDesc: cudnnConvolutionDescriptor_t,
    pad_h: *mut ::core::ffi::c_int,
    pad_w: *mut ::core::ffi::c_int,
    u: *mut ::core::ffi::c_int,
    v: *mut ::core::ffi::c_int,
    dilation_h: *mut ::core::ffi::c_int,
    dilation_w: *mut ::core::ffi::c_int,
    mode: *mut cudnnConvolutionMode_t,
    computeType: *mut cudnnDataType_t,
) -> cudnnStatus_t {
    let sym: Symbol<cudnnGetConvolution2dDescriptor_fn> = unsafe {
        get_lib()
            .get(b"cudnnGetConvolution2dDescriptor\0")
            .expect("Missing symbol")
    };
    unsafe {
        (*sym)(
            convDesc,
            pad_h,
            pad_w,
            u,
            v,
            dilation_h,
            dilation_w,
            mode,
            computeType,
        )
    }
}

type cudnnSetConvolutionNdDescriptor_fn = unsafe extern "C" fn(
    convDesc: cudnnConvolutionDescriptor_t,
    arrayLength: ::core::ffi::c_int,
    padA: *const ::core::ffi::c_int,
    filterStrideA: *const ::core::ffi::c_int,
    dilationA: *const ::core::ffi::c_int,
    mode: cudnnConvolutionMode_t,
    computeType: cudnnDataType_t,
) -> cudnnStatus_t;
pub unsafe fn cudnnSetConvolutionNdDescriptor(
    convDesc: cudnnConvolutionDescriptor_t,
    arrayLength: ::core::ffi::c_int,
    padA: *const ::core::ffi::c_int,
    filterStrideA: *const ::core::ffi::c_int,
    dilationA: *const ::core::ffi::c_int,
    mode: cudnnConvolutionMode_t,
    computeType: cudnnDataType_t,
) -> cudnnStatus_t {
    let sym: Symbol<cudnnSetConvolutionNdDescriptor_fn> = unsafe {
        get_lib()
            .get(b"cudnnSetConvolutionNdDescriptor\0")
            .expect("Missing symbol")
    };
    unsafe {
        (*sym)(
            convDesc,
            arrayLength,
            padA,
            filterStrideA,
            dilationA,
            mode,
            computeType,
        )
    }
}

type cudnnGetConvolutionNdDescriptor_fn = unsafe extern "C" fn(
    convDesc: cudnnConvolutionDescriptor_t,
    arrayLengthRequested: ::core::ffi::c_int,
    arrayLength: *mut ::core::ffi::c_int,
    padA: *mut ::core::ffi::c_int,
    strideA: *mut ::core::ffi::c_int,
    dilationA: *mut ::core::ffi::c_int,
    mode: *mut cudnnConvolutionMode_t,
    computeType: *mut cudnnDataType_t,
) -> cudnnStatus_t;
pub unsafe fn cudnnGetConvolutionNdDescriptor(
    convDesc: cudnnConvolutionDescriptor_t,
    arrayLengthRequested: ::core::ffi::c_int,
    arrayLength: *mut ::core::ffi::c_int,
    padA: *mut ::core::ffi::c_int,
    strideA: *mut ::core::ffi::c_int,
    dilationA: *mut ::core::ffi::c_int,
    mode: *mut cudnnConvolutionMode_t,
    computeType: *mut cudnnDataType_t,
) -> cudnnStatus_t {
    let sym: Symbol<cudnnGetConvolutionNdDescriptor_fn> = unsafe {
        get_lib()
            .get(b"cudnnGetConvolutionNdDescriptor\0")
            .expect("Missing symbol")
    };
    unsafe {
        (*sym)(
            convDesc,
            arrayLengthRequested,
            arrayLength,
            padA,
            strideA,
            dilationA,
            mode,
            computeType,
        )
    }
}

type cudnnGetConvolution2dForwardOutputDim_fn = unsafe extern "C" fn(
    convDesc: cudnnConvolutionDescriptor_t,
    inputTensorDesc: cudnnTensorDescriptor_t,
    filterDesc: cudnnFilterDescriptor_t,
    n: *mut ::core::ffi::c_int,
    c: *mut ::core::ffi::c_int,
    h: *mut ::core::ffi::c_int,
    w: *mut ::core::ffi::c_int,
) -> cudnnStatus_t;
pub unsafe fn cudnnGetConvolution2dForwardOutputDim(
    convDesc: cudnnConvolutionDescriptor_t,
    inputTensorDesc: cudnnTensorDescriptor_t,
    filterDesc: cudnnFilterDescriptor_t,
    n: *mut ::core::ffi::c_int,
    c: *mut ::core::ffi::c_int,
    h: *mut ::core::ffi::c_int,
    w: *mut ::core::ffi::c_int,
) -> cudnnStatus_t {
    let sym: Symbol<cudnnGetConvolution2dForwardOutputDim_fn> = unsafe {
        get_lib()
            .get(b"cudnnGetConvolution2dForwardOutputDim\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(convDesc, inputTensorDesc, filterDesc, n, c, h, w) }
}

type cudnnGetConvolutionNdForwardOutputDim_fn = unsafe extern "C" fn(
    convDesc: cudnnConvolutionDescriptor_t,
    inputTensorDesc: cudnnTensorDescriptor_t,
    filterDesc: cudnnFilterDescriptor_t,
    nbDims: ::core::ffi::c_int,
    tensorOuputDimA: *mut ::core::ffi::c_int,
) -> cudnnStatus_t;
pub unsafe fn cudnnGetConvolutionNdForwardOutputDim(
    convDesc: cudnnConvolutionDescriptor_t,
    inputTensorDesc: cudnnTensorDescriptor_t,
    filterDesc: cudnnFilterDescriptor_t,
    nbDims: ::core::ffi::c_int,
    tensorOuputDimA: *mut ::core::ffi::c_int,
) -> cudnnStatus_t {
    let sym: Symbol<cudnnGetConvolutionNdForwardOutputDim_fn> = unsafe {
        get_lib()
            .get(b"cudnnGetConvolutionNdForwardOutputDim\0")
            .expect("Missing symbol")
    };
    unsafe {
        (*sym)(
            convDesc,
            inputTensorDesc,
            filterDesc,
            nbDims,
            tensorOuputDimA,
        )
    }
}

type cudnnGetConvolutionForwardAlgorithmMaxCount_fn =
    unsafe extern "C" fn(handle: cudnnHandle_t, count: *mut ::core::ffi::c_int) -> cudnnStatus_t;
pub unsafe fn cudnnGetConvolutionForwardAlgorithmMaxCount(
    handle: cudnnHandle_t,
    count: *mut ::core::ffi::c_int,
) -> cudnnStatus_t {
    let sym: Symbol<cudnnGetConvolutionForwardAlgorithmMaxCount_fn> = unsafe {
        get_lib()
            .get(b"cudnnGetConvolutionForwardAlgorithmMaxCount\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(handle, count) }
}

type cudnnGetConvolutionForwardAlgorithm_v7_fn = unsafe extern "C" fn(
    handle: cudnnHandle_t,
    srcDesc: cudnnTensorDescriptor_t,
    filterDesc: cudnnFilterDescriptor_t,
    convDesc: cudnnConvolutionDescriptor_t,
    destDesc: cudnnTensorDescriptor_t,
    requestedAlgoCount: ::core::ffi::c_int,
    returnedAlgoCount: *mut ::core::ffi::c_int,
    perfResults: *mut cudnnConvolutionFwdAlgoPerf_t,
) -> cudnnStatus_t;
pub unsafe fn cudnnGetConvolutionForwardAlgorithm_v7(
    handle: cudnnHandle_t,
    srcDesc: cudnnTensorDescriptor_t,
    filterDesc: cudnnFilterDescriptor_t,
    convDesc: cudnnConvolutionDescriptor_t,
    destDesc: cudnnTensorDescriptor_t,
    requestedAlgoCount: ::core::ffi::c_int,
    returnedAlgoCount: *mut ::core::ffi::c_int,
    perfResults: *mut cudnnConvolutionFwdAlgoPerf_t,
) -> cudnnStatus_t {
    let sym: Symbol<cudnnGetConvolutionForwardAlgorithm_v7_fn> = unsafe {
        get_lib()
            .get(b"cudnnGetConvolutionForwardAlgorithm_v7\0")
            .expect("Missing symbol")
    };
    unsafe {
        (*sym)(
            handle,
            srcDesc,
            filterDesc,
            convDesc,
            destDesc,
            requestedAlgoCount,
            returnedAlgoCount,
            perfResults,
        )
    }
}

type cudnnFindConvolutionForwardAlgorithm_fn = unsafe extern "C" fn(
    handle: cudnnHandle_t,
    xDesc: cudnnTensorDescriptor_t,
    wDesc: cudnnFilterDescriptor_t,
    convDesc: cudnnConvolutionDescriptor_t,
    yDesc: cudnnTensorDescriptor_t,
    requestedAlgoCount: ::core::ffi::c_int,
    returnedAlgoCount: *mut ::core::ffi::c_int,
    perfResults: *mut cudnnConvolutionFwdAlgoPerf_t,
) -> cudnnStatus_t;
pub unsafe fn cudnnFindConvolutionForwardAlgorithm(
    handle: cudnnHandle_t,
    xDesc: cudnnTensorDescriptor_t,
    wDesc: cudnnFilterDescriptor_t,
    convDesc: cudnnConvolutionDescriptor_t,
    yDesc: cudnnTensorDescriptor_t,
    requestedAlgoCount: ::core::ffi::c_int,
    returnedAlgoCount: *mut ::core::ffi::c_int,
    perfResults: *mut cudnnConvolutionFwdAlgoPerf_t,
) -> cudnnStatus_t {
    let sym: Symbol<cudnnFindConvolutionForwardAlgorithm_fn> = unsafe {
        get_lib()
            .get(b"cudnnFindConvolutionForwardAlgorithm\0")
            .expect("Missing symbol")
    };
    unsafe {
        (*sym)(
            handle,
            xDesc,
            wDesc,
            convDesc,
            yDesc,
            requestedAlgoCount,
            returnedAlgoCount,
            perfResults,
        )
    }
}

type cudnnFindConvolutionForwardAlgorithmEx_fn = unsafe extern "C" fn(
    handle: cudnnHandle_t,
    xDesc: cudnnTensorDescriptor_t,
    x: *const ::core::ffi::c_void,
    wDesc: cudnnFilterDescriptor_t,
    w: *const ::core::ffi::c_void,
    convDesc: cudnnConvolutionDescriptor_t,
    yDesc: cudnnTensorDescriptor_t,
    y: *mut ::core::ffi::c_void,
    requestedAlgoCount: ::core::ffi::c_int,
    returnedAlgoCount: *mut ::core::ffi::c_int,
    perfResults: *mut cudnnConvolutionFwdAlgoPerf_t,
    workSpace: *mut ::core::ffi::c_void,
    workSpaceSizeInBytes: usize,
) -> cudnnStatus_t;
pub unsafe fn cudnnFindConvolutionForwardAlgorithmEx(
    handle: cudnnHandle_t,
    xDesc: cudnnTensorDescriptor_t,
    x: *const ::core::ffi::c_void,
    wDesc: cudnnFilterDescriptor_t,
    w: *const ::core::ffi::c_void,
    convDesc: cudnnConvolutionDescriptor_t,
    yDesc: cudnnTensorDescriptor_t,
    y: *mut ::core::ffi::c_void,
    requestedAlgoCount: ::core::ffi::c_int,
    returnedAlgoCount: *mut ::core::ffi::c_int,
    perfResults: *mut cudnnConvolutionFwdAlgoPerf_t,
    workSpace: *mut ::core::ffi::c_void,
    workSpaceSizeInBytes: usize,
) -> cudnnStatus_t {
    let sym: Symbol<cudnnFindConvolutionForwardAlgorithmEx_fn> = unsafe {
        get_lib()
            .get(b"cudnnFindConvolutionForwardAlgorithmEx\0")
            .expect("Missing symbol")
    };
    unsafe {
        (*sym)(
            handle,
            xDesc,
            x,
            wDesc,
            w,
            convDesc,
            yDesc,
            y,
            requestedAlgoCount,
            returnedAlgoCount,
            perfResults,
            workSpace,
            workSpaceSizeInBytes,
        )
    }
}

type cudnnIm2Col_fn = unsafe extern "C" fn(
    handle: cudnnHandle_t,
    xDesc: cudnnTensorDescriptor_t,
    x: *const ::core::ffi::c_void,
    wDesc: cudnnFilterDescriptor_t,
    convDesc: cudnnConvolutionDescriptor_t,
    colBuffer: *mut ::core::ffi::c_void,
) -> cudnnStatus_t;
pub unsafe fn cudnnIm2Col(
    handle: cudnnHandle_t,
    xDesc: cudnnTensorDescriptor_t,
    x: *const ::core::ffi::c_void,
    wDesc: cudnnFilterDescriptor_t,
    convDesc: cudnnConvolutionDescriptor_t,
    colBuffer: *mut ::core::ffi::c_void,
) -> cudnnStatus_t {
    let sym: Symbol<cudnnIm2Col_fn> =
        unsafe { get_lib().get(b"cudnnIm2Col\0").expect("Missing symbol") };
    unsafe { (*sym)(handle, xDesc, x, wDesc, convDesc, colBuffer) }
}

type cudnnReorderFilterAndBias_fn = unsafe extern "C" fn(
    handle: cudnnHandle_t,
    filterDesc: cudnnFilterDescriptor_t,
    reorderType: cudnnReorderType_t,
    filterData: *const ::core::ffi::c_void,
    reorderedFilterData: *mut ::core::ffi::c_void,
    reorderBias: ::core::ffi::c_int,
    biasData: *const ::core::ffi::c_void,
    reorderedBiasData: *mut ::core::ffi::c_void,
) -> cudnnStatus_t;
pub unsafe fn cudnnReorderFilterAndBias(
    handle: cudnnHandle_t,
    filterDesc: cudnnFilterDescriptor_t,
    reorderType: cudnnReorderType_t,
    filterData: *const ::core::ffi::c_void,
    reorderedFilterData: *mut ::core::ffi::c_void,
    reorderBias: ::core::ffi::c_int,
    biasData: *const ::core::ffi::c_void,
    reorderedBiasData: *mut ::core::ffi::c_void,
) -> cudnnStatus_t {
    let sym: Symbol<cudnnReorderFilterAndBias_fn> = unsafe {
        get_lib()
            .get(b"cudnnReorderFilterAndBias\0")
            .expect("Missing symbol")
    };
    unsafe {
        (*sym)(
            handle,
            filterDesc,
            reorderType,
            filterData,
            reorderedFilterData,
            reorderBias,
            biasData,
            reorderedBiasData,
        )
    }
}

type cudnnGetConvolutionForwardWorkspaceSize_fn = unsafe extern "C" fn(
    handle: cudnnHandle_t,
    xDesc: cudnnTensorDescriptor_t,
    wDesc: cudnnFilterDescriptor_t,
    convDesc: cudnnConvolutionDescriptor_t,
    yDesc: cudnnTensorDescriptor_t,
    algo: cudnnConvolutionFwdAlgo_t,
    sizeInBytes: *mut usize,
) -> cudnnStatus_t;
pub unsafe fn cudnnGetConvolutionForwardWorkspaceSize(
    handle: cudnnHandle_t,
    xDesc: cudnnTensorDescriptor_t,
    wDesc: cudnnFilterDescriptor_t,
    convDesc: cudnnConvolutionDescriptor_t,
    yDesc: cudnnTensorDescriptor_t,
    algo: cudnnConvolutionFwdAlgo_t,
    sizeInBytes: *mut usize,
) -> cudnnStatus_t {
    let sym: Symbol<cudnnGetConvolutionForwardWorkspaceSize_fn> = unsafe {
        get_lib()
            .get(b"cudnnGetConvolutionForwardWorkspaceSize\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(handle, xDesc, wDesc, convDesc, yDesc, algo, sizeInBytes) }
}

type cudnnConvolutionForward_fn = unsafe extern "C" fn(
    handle: cudnnHandle_t,
    alpha: *const ::core::ffi::c_void,
    xDesc: cudnnTensorDescriptor_t,
    x: *const ::core::ffi::c_void,
    wDesc: cudnnFilterDescriptor_t,
    w: *const ::core::ffi::c_void,
    convDesc: cudnnConvolutionDescriptor_t,
    algo: cudnnConvolutionFwdAlgo_t,
    workSpace: *mut ::core::ffi::c_void,
    workSpaceSizeInBytes: usize,
    beta: *const ::core::ffi::c_void,
    yDesc: cudnnTensorDescriptor_t,
    y: *mut ::core::ffi::c_void,
) -> cudnnStatus_t;
pub unsafe fn cudnnConvolutionForward(
    handle: cudnnHandle_t,
    alpha: *const ::core::ffi::c_void,
    xDesc: cudnnTensorDescriptor_t,
    x: *const ::core::ffi::c_void,
    wDesc: cudnnFilterDescriptor_t,
    w: *const ::core::ffi::c_void,
    convDesc: cudnnConvolutionDescriptor_t,
    algo: cudnnConvolutionFwdAlgo_t,
    workSpace: *mut ::core::ffi::c_void,
    workSpaceSizeInBytes: usize,
    beta: *const ::core::ffi::c_void,
    yDesc: cudnnTensorDescriptor_t,
    y: *mut ::core::ffi::c_void,
) -> cudnnStatus_t {
    let sym: Symbol<cudnnConvolutionForward_fn> = unsafe {
        get_lib()
            .get(b"cudnnConvolutionForward\0")
            .expect("Missing symbol")
    };
    unsafe {
        (*sym)(
            handle,
            alpha,
            xDesc,
            x,
            wDesc,
            w,
            convDesc,
            algo,
            workSpace,
            workSpaceSizeInBytes,
            beta,
            yDesc,
            y,
        )
    }
}

type cudnnConvolutionBiasActivationForward_fn = unsafe extern "C" fn(
    handle: cudnnHandle_t,
    alpha1: *const ::core::ffi::c_void,
    xDesc: cudnnTensorDescriptor_t,
    x: *const ::core::ffi::c_void,
    wDesc: cudnnFilterDescriptor_t,
    w: *const ::core::ffi::c_void,
    convDesc: cudnnConvolutionDescriptor_t,
    algo: cudnnConvolutionFwdAlgo_t,
    workSpace: *mut ::core::ffi::c_void,
    workSpaceSizeInBytes: usize,
    alpha2: *const ::core::ffi::c_void,
    zDesc: cudnnTensorDescriptor_t,
    z: *const ::core::ffi::c_void,
    biasDesc: cudnnTensorDescriptor_t,
    bias: *const ::core::ffi::c_void,
    activationDesc: cudnnActivationDescriptor_t,
    yDesc: cudnnTensorDescriptor_t,
    y: *mut ::core::ffi::c_void,
) -> cudnnStatus_t;
pub unsafe fn cudnnConvolutionBiasActivationForward(
    handle: cudnnHandle_t,
    alpha1: *const ::core::ffi::c_void,
    xDesc: cudnnTensorDescriptor_t,
    x: *const ::core::ffi::c_void,
    wDesc: cudnnFilterDescriptor_t,
    w: *const ::core::ffi::c_void,
    convDesc: cudnnConvolutionDescriptor_t,
    algo: cudnnConvolutionFwdAlgo_t,
    workSpace: *mut ::core::ffi::c_void,
    workSpaceSizeInBytes: usize,
    alpha2: *const ::core::ffi::c_void,
    zDesc: cudnnTensorDescriptor_t,
    z: *const ::core::ffi::c_void,
    biasDesc: cudnnTensorDescriptor_t,
    bias: *const ::core::ffi::c_void,
    activationDesc: cudnnActivationDescriptor_t,
    yDesc: cudnnTensorDescriptor_t,
    y: *mut ::core::ffi::c_void,
) -> cudnnStatus_t {
    let sym: Symbol<cudnnConvolutionBiasActivationForward_fn> = unsafe {
        get_lib()
            .get(b"cudnnConvolutionBiasActivationForward\0")
            .expect("Missing symbol")
    };
    unsafe {
        (*sym)(
            handle,
            alpha1,
            xDesc,
            x,
            wDesc,
            w,
            convDesc,
            algo,
            workSpace,
            workSpaceSizeInBytes,
            alpha2,
            zDesc,
            z,
            biasDesc,
            bias,
            activationDesc,
            yDesc,
            y,
        )
    }
}

type cudnnGetConvolutionBackwardDataAlgorithmMaxCount_fn =
    unsafe extern "C" fn(handle: cudnnHandle_t, count: *mut ::core::ffi::c_int) -> cudnnStatus_t;
pub unsafe fn cudnnGetConvolutionBackwardDataAlgorithmMaxCount(
    handle: cudnnHandle_t,
    count: *mut ::core::ffi::c_int,
) -> cudnnStatus_t {
    let sym: Symbol<cudnnGetConvolutionBackwardDataAlgorithmMaxCount_fn> = unsafe {
        get_lib()
            .get(b"cudnnGetConvolutionBackwardDataAlgorithmMaxCount\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(handle, count) }
}

type cudnnFindConvolutionBackwardDataAlgorithm_fn = unsafe extern "C" fn(
    handle: cudnnHandle_t,
    wDesc: cudnnFilterDescriptor_t,
    dyDesc: cudnnTensorDescriptor_t,
    convDesc: cudnnConvolutionDescriptor_t,
    dxDesc: cudnnTensorDescriptor_t,
    requestedAlgoCount: ::core::ffi::c_int,
    returnedAlgoCount: *mut ::core::ffi::c_int,
    perfResults: *mut cudnnConvolutionBwdDataAlgoPerf_t,
) -> cudnnStatus_t;
pub unsafe fn cudnnFindConvolutionBackwardDataAlgorithm(
    handle: cudnnHandle_t,
    wDesc: cudnnFilterDescriptor_t,
    dyDesc: cudnnTensorDescriptor_t,
    convDesc: cudnnConvolutionDescriptor_t,
    dxDesc: cudnnTensorDescriptor_t,
    requestedAlgoCount: ::core::ffi::c_int,
    returnedAlgoCount: *mut ::core::ffi::c_int,
    perfResults: *mut cudnnConvolutionBwdDataAlgoPerf_t,
) -> cudnnStatus_t {
    let sym: Symbol<cudnnFindConvolutionBackwardDataAlgorithm_fn> = unsafe {
        get_lib()
            .get(b"cudnnFindConvolutionBackwardDataAlgorithm\0")
            .expect("Missing symbol")
    };
    unsafe {
        (*sym)(
            handle,
            wDesc,
            dyDesc,
            convDesc,
            dxDesc,
            requestedAlgoCount,
            returnedAlgoCount,
            perfResults,
        )
    }
}

type cudnnFindConvolutionBackwardDataAlgorithmEx_fn = unsafe extern "C" fn(
    handle: cudnnHandle_t,
    wDesc: cudnnFilterDescriptor_t,
    w: *const ::core::ffi::c_void,
    dyDesc: cudnnTensorDescriptor_t,
    dy: *const ::core::ffi::c_void,
    convDesc: cudnnConvolutionDescriptor_t,
    dxDesc: cudnnTensorDescriptor_t,
    dx: *mut ::core::ffi::c_void,
    requestedAlgoCount: ::core::ffi::c_int,
    returnedAlgoCount: *mut ::core::ffi::c_int,
    perfResults: *mut cudnnConvolutionBwdDataAlgoPerf_t,
    workSpace: *mut ::core::ffi::c_void,
    workSpaceSizeInBytes: usize,
) -> cudnnStatus_t;
pub unsafe fn cudnnFindConvolutionBackwardDataAlgorithmEx(
    handle: cudnnHandle_t,
    wDesc: cudnnFilterDescriptor_t,
    w: *const ::core::ffi::c_void,
    dyDesc: cudnnTensorDescriptor_t,
    dy: *const ::core::ffi::c_void,
    convDesc: cudnnConvolutionDescriptor_t,
    dxDesc: cudnnTensorDescriptor_t,
    dx: *mut ::core::ffi::c_void,
    requestedAlgoCount: ::core::ffi::c_int,
    returnedAlgoCount: *mut ::core::ffi::c_int,
    perfResults: *mut cudnnConvolutionBwdDataAlgoPerf_t,
    workSpace: *mut ::core::ffi::c_void,
    workSpaceSizeInBytes: usize,
) -> cudnnStatus_t {
    let sym: Symbol<cudnnFindConvolutionBackwardDataAlgorithmEx_fn> = unsafe {
        get_lib()
            .get(b"cudnnFindConvolutionBackwardDataAlgorithmEx\0")
            .expect("Missing symbol")
    };
    unsafe {
        (*sym)(
            handle,
            wDesc,
            w,
            dyDesc,
            dy,
            convDesc,
            dxDesc,
            dx,
            requestedAlgoCount,
            returnedAlgoCount,
            perfResults,
            workSpace,
            workSpaceSizeInBytes,
        )
    }
}

type cudnnGetConvolutionBackwardDataAlgorithm_v7_fn = unsafe extern "C" fn(
    handle: cudnnHandle_t,
    filterDesc: cudnnFilterDescriptor_t,
    diffDesc: cudnnTensorDescriptor_t,
    convDesc: cudnnConvolutionDescriptor_t,
    gradDesc: cudnnTensorDescriptor_t,
    requestedAlgoCount: ::core::ffi::c_int,
    returnedAlgoCount: *mut ::core::ffi::c_int,
    perfResults: *mut cudnnConvolutionBwdDataAlgoPerf_t,
) -> cudnnStatus_t;
pub unsafe fn cudnnGetConvolutionBackwardDataAlgorithm_v7(
    handle: cudnnHandle_t,
    filterDesc: cudnnFilterDescriptor_t,
    diffDesc: cudnnTensorDescriptor_t,
    convDesc: cudnnConvolutionDescriptor_t,
    gradDesc: cudnnTensorDescriptor_t,
    requestedAlgoCount: ::core::ffi::c_int,
    returnedAlgoCount: *mut ::core::ffi::c_int,
    perfResults: *mut cudnnConvolutionBwdDataAlgoPerf_t,
) -> cudnnStatus_t {
    let sym: Symbol<cudnnGetConvolutionBackwardDataAlgorithm_v7_fn> = unsafe {
        get_lib()
            .get(b"cudnnGetConvolutionBackwardDataAlgorithm_v7\0")
            .expect("Missing symbol")
    };
    unsafe {
        (*sym)(
            handle,
            filterDesc,
            diffDesc,
            convDesc,
            gradDesc,
            requestedAlgoCount,
            returnedAlgoCount,
            perfResults,
        )
    }
}

type cudnnGetConvolutionBackwardDataWorkspaceSize_fn = unsafe extern "C" fn(
    handle: cudnnHandle_t,
    wDesc: cudnnFilterDescriptor_t,
    dyDesc: cudnnTensorDescriptor_t,
    convDesc: cudnnConvolutionDescriptor_t,
    dxDesc: cudnnTensorDescriptor_t,
    algo: cudnnConvolutionBwdDataAlgo_t,
    sizeInBytes: *mut usize,
) -> cudnnStatus_t;
pub unsafe fn cudnnGetConvolutionBackwardDataWorkspaceSize(
    handle: cudnnHandle_t,
    wDesc: cudnnFilterDescriptor_t,
    dyDesc: cudnnTensorDescriptor_t,
    convDesc: cudnnConvolutionDescriptor_t,
    dxDesc: cudnnTensorDescriptor_t,
    algo: cudnnConvolutionBwdDataAlgo_t,
    sizeInBytes: *mut usize,
) -> cudnnStatus_t {
    let sym: Symbol<cudnnGetConvolutionBackwardDataWorkspaceSize_fn> = unsafe {
        get_lib()
            .get(b"cudnnGetConvolutionBackwardDataWorkspaceSize\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(handle, wDesc, dyDesc, convDesc, dxDesc, algo, sizeInBytes) }
}

type cudnnConvolutionBackwardData_fn = unsafe extern "C" fn(
    handle: cudnnHandle_t,
    alpha: *const ::core::ffi::c_void,
    wDesc: cudnnFilterDescriptor_t,
    w: *const ::core::ffi::c_void,
    dyDesc: cudnnTensorDescriptor_t,
    dy: *const ::core::ffi::c_void,
    convDesc: cudnnConvolutionDescriptor_t,
    algo: cudnnConvolutionBwdDataAlgo_t,
    workSpace: *mut ::core::ffi::c_void,
    workSpaceSizeInBytes: usize,
    beta: *const ::core::ffi::c_void,
    dxDesc: cudnnTensorDescriptor_t,
    dx: *mut ::core::ffi::c_void,
) -> cudnnStatus_t;
pub unsafe fn cudnnConvolutionBackwardData(
    handle: cudnnHandle_t,
    alpha: *const ::core::ffi::c_void,
    wDesc: cudnnFilterDescriptor_t,
    w: *const ::core::ffi::c_void,
    dyDesc: cudnnTensorDescriptor_t,
    dy: *const ::core::ffi::c_void,
    convDesc: cudnnConvolutionDescriptor_t,
    algo: cudnnConvolutionBwdDataAlgo_t,
    workSpace: *mut ::core::ffi::c_void,
    workSpaceSizeInBytes: usize,
    beta: *const ::core::ffi::c_void,
    dxDesc: cudnnTensorDescriptor_t,
    dx: *mut ::core::ffi::c_void,
) -> cudnnStatus_t {
    let sym: Symbol<cudnnConvolutionBackwardData_fn> = unsafe {
        get_lib()
            .get(b"cudnnConvolutionBackwardData\0")
            .expect("Missing symbol")
    };
    unsafe {
        (*sym)(
            handle,
            alpha,
            wDesc,
            w,
            dyDesc,
            dy,
            convDesc,
            algo,
            workSpace,
            workSpaceSizeInBytes,
            beta,
            dxDesc,
            dx,
        )
    }
}

type cudnnGetFoldedConvBackwardDataDescriptors_fn = unsafe extern "C" fn(
    handle: cudnnHandle_t,
    filterDesc: cudnnFilterDescriptor_t,
    diffDesc: cudnnTensorDescriptor_t,
    convDesc: cudnnConvolutionDescriptor_t,
    gradDesc: cudnnTensorDescriptor_t,
    transformFormat: cudnnTensorFormat_t,
    foldedFilterDesc: cudnnFilterDescriptor_t,
    paddedDiffDesc: cudnnTensorDescriptor_t,
    foldedConvDesc: cudnnConvolutionDescriptor_t,
    foldedGradDesc: cudnnTensorDescriptor_t,
    filterFoldTransDesc: cudnnTensorTransformDescriptor_t,
    diffPadTransDesc: cudnnTensorTransformDescriptor_t,
    gradFoldTransDesc: cudnnTensorTransformDescriptor_t,
    gradUnfoldTransDesc: cudnnTensorTransformDescriptor_t,
) -> cudnnStatus_t;
pub unsafe fn cudnnGetFoldedConvBackwardDataDescriptors(
    handle: cudnnHandle_t,
    filterDesc: cudnnFilterDescriptor_t,
    diffDesc: cudnnTensorDescriptor_t,
    convDesc: cudnnConvolutionDescriptor_t,
    gradDesc: cudnnTensorDescriptor_t,
    transformFormat: cudnnTensorFormat_t,
    foldedFilterDesc: cudnnFilterDescriptor_t,
    paddedDiffDesc: cudnnTensorDescriptor_t,
    foldedConvDesc: cudnnConvolutionDescriptor_t,
    foldedGradDesc: cudnnTensorDescriptor_t,
    filterFoldTransDesc: cudnnTensorTransformDescriptor_t,
    diffPadTransDesc: cudnnTensorTransformDescriptor_t,
    gradFoldTransDesc: cudnnTensorTransformDescriptor_t,
    gradUnfoldTransDesc: cudnnTensorTransformDescriptor_t,
) -> cudnnStatus_t {
    let sym: Symbol<cudnnGetFoldedConvBackwardDataDescriptors_fn> = unsafe {
        get_lib()
            .get(b"cudnnGetFoldedConvBackwardDataDescriptors\0")
            .expect("Missing symbol")
    };
    unsafe {
        (*sym)(
            handle,
            filterDesc,
            diffDesc,
            convDesc,
            gradDesc,
            transformFormat,
            foldedFilterDesc,
            paddedDiffDesc,
            foldedConvDesc,
            foldedGradDesc,
            filterFoldTransDesc,
            diffPadTransDesc,
            gradFoldTransDesc,
            gradUnfoldTransDesc,
        )
    }
}

type cudnnCnnVersionCheck_fn = unsafe extern "C" fn() -> cudnnStatus_t;
pub unsafe fn cudnnCnnVersionCheck() -> cudnnStatus_t {
    let sym: Symbol<cudnnCnnVersionCheck_fn> = unsafe {
        get_lib()
            .get(b"cudnnCnnVersionCheck\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)() }
}

type cudnnGetConvolutionBackwardFilterAlgorithmMaxCount_fn =
    unsafe extern "C" fn(handle: cudnnHandle_t, count: *mut ::core::ffi::c_int) -> cudnnStatus_t;
pub unsafe fn cudnnGetConvolutionBackwardFilterAlgorithmMaxCount(
    handle: cudnnHandle_t,
    count: *mut ::core::ffi::c_int,
) -> cudnnStatus_t {
    let sym: Symbol<cudnnGetConvolutionBackwardFilterAlgorithmMaxCount_fn> = unsafe {
        get_lib()
            .get(b"cudnnGetConvolutionBackwardFilterAlgorithmMaxCount\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(handle, count) }
}

type cudnnFindConvolutionBackwardFilterAlgorithm_fn = unsafe extern "C" fn(
    handle: cudnnHandle_t,
    xDesc: cudnnTensorDescriptor_t,
    dyDesc: cudnnTensorDescriptor_t,
    convDesc: cudnnConvolutionDescriptor_t,
    dwDesc: cudnnFilterDescriptor_t,
    requestedAlgoCount: ::core::ffi::c_int,
    returnedAlgoCount: *mut ::core::ffi::c_int,
    perfResults: *mut cudnnConvolutionBwdFilterAlgoPerf_t,
) -> cudnnStatus_t;
pub unsafe fn cudnnFindConvolutionBackwardFilterAlgorithm(
    handle: cudnnHandle_t,
    xDesc: cudnnTensorDescriptor_t,
    dyDesc: cudnnTensorDescriptor_t,
    convDesc: cudnnConvolutionDescriptor_t,
    dwDesc: cudnnFilterDescriptor_t,
    requestedAlgoCount: ::core::ffi::c_int,
    returnedAlgoCount: *mut ::core::ffi::c_int,
    perfResults: *mut cudnnConvolutionBwdFilterAlgoPerf_t,
) -> cudnnStatus_t {
    let sym: Symbol<cudnnFindConvolutionBackwardFilterAlgorithm_fn> = unsafe {
        get_lib()
            .get(b"cudnnFindConvolutionBackwardFilterAlgorithm\0")
            .expect("Missing symbol")
    };
    unsafe {
        (*sym)(
            handle,
            xDesc,
            dyDesc,
            convDesc,
            dwDesc,
            requestedAlgoCount,
            returnedAlgoCount,
            perfResults,
        )
    }
}

type cudnnFindConvolutionBackwardFilterAlgorithmEx_fn = unsafe extern "C" fn(
    handle: cudnnHandle_t,
    xDesc: cudnnTensorDescriptor_t,
    x: *const ::core::ffi::c_void,
    dyDesc: cudnnTensorDescriptor_t,
    y: *const ::core::ffi::c_void,
    convDesc: cudnnConvolutionDescriptor_t,
    dwDesc: cudnnFilterDescriptor_t,
    dw: *mut ::core::ffi::c_void,
    requestedAlgoCount: ::core::ffi::c_int,
    returnedAlgoCount: *mut ::core::ffi::c_int,
    perfResults: *mut cudnnConvolutionBwdFilterAlgoPerf_t,
    workSpace: *mut ::core::ffi::c_void,
    workSpaceSizeInBytes: usize,
) -> cudnnStatus_t;
pub unsafe fn cudnnFindConvolutionBackwardFilterAlgorithmEx(
    handle: cudnnHandle_t,
    xDesc: cudnnTensorDescriptor_t,
    x: *const ::core::ffi::c_void,
    dyDesc: cudnnTensorDescriptor_t,
    y: *const ::core::ffi::c_void,
    convDesc: cudnnConvolutionDescriptor_t,
    dwDesc: cudnnFilterDescriptor_t,
    dw: *mut ::core::ffi::c_void,
    requestedAlgoCount: ::core::ffi::c_int,
    returnedAlgoCount: *mut ::core::ffi::c_int,
    perfResults: *mut cudnnConvolutionBwdFilterAlgoPerf_t,
    workSpace: *mut ::core::ffi::c_void,
    workSpaceSizeInBytes: usize,
) -> cudnnStatus_t {
    let sym: Symbol<cudnnFindConvolutionBackwardFilterAlgorithmEx_fn> = unsafe {
        get_lib()
            .get(b"cudnnFindConvolutionBackwardFilterAlgorithmEx\0")
            .expect("Missing symbol")
    };
    unsafe {
        (*sym)(
            handle,
            xDesc,
            x,
            dyDesc,
            y,
            convDesc,
            dwDesc,
            dw,
            requestedAlgoCount,
            returnedAlgoCount,
            perfResults,
            workSpace,
            workSpaceSizeInBytes,
        )
    }
}

type cudnnGetConvolutionBackwardFilterAlgorithm_v7_fn = unsafe extern "C" fn(
    handle: cudnnHandle_t,
    srcDesc: cudnnTensorDescriptor_t,
    diffDesc: cudnnTensorDescriptor_t,
    convDesc: cudnnConvolutionDescriptor_t,
    gradDesc: cudnnFilterDescriptor_t,
    requestedAlgoCount: ::core::ffi::c_int,
    returnedAlgoCount: *mut ::core::ffi::c_int,
    perfResults: *mut cudnnConvolutionBwdFilterAlgoPerf_t,
) -> cudnnStatus_t;
pub unsafe fn cudnnGetConvolutionBackwardFilterAlgorithm_v7(
    handle: cudnnHandle_t,
    srcDesc: cudnnTensorDescriptor_t,
    diffDesc: cudnnTensorDescriptor_t,
    convDesc: cudnnConvolutionDescriptor_t,
    gradDesc: cudnnFilterDescriptor_t,
    requestedAlgoCount: ::core::ffi::c_int,
    returnedAlgoCount: *mut ::core::ffi::c_int,
    perfResults: *mut cudnnConvolutionBwdFilterAlgoPerf_t,
) -> cudnnStatus_t {
    let sym: Symbol<cudnnGetConvolutionBackwardFilterAlgorithm_v7_fn> = unsafe {
        get_lib()
            .get(b"cudnnGetConvolutionBackwardFilterAlgorithm_v7\0")
            .expect("Missing symbol")
    };
    unsafe {
        (*sym)(
            handle,
            srcDesc,
            diffDesc,
            convDesc,
            gradDesc,
            requestedAlgoCount,
            returnedAlgoCount,
            perfResults,
        )
    }
}

type cudnnGetConvolutionBackwardFilterWorkspaceSize_fn = unsafe extern "C" fn(
    handle: cudnnHandle_t,
    xDesc: cudnnTensorDescriptor_t,
    dyDesc: cudnnTensorDescriptor_t,
    convDesc: cudnnConvolutionDescriptor_t,
    gradDesc: cudnnFilterDescriptor_t,
    algo: cudnnConvolutionBwdFilterAlgo_t,
    sizeInBytes: *mut usize,
) -> cudnnStatus_t;
pub unsafe fn cudnnGetConvolutionBackwardFilterWorkspaceSize(
    handle: cudnnHandle_t,
    xDesc: cudnnTensorDescriptor_t,
    dyDesc: cudnnTensorDescriptor_t,
    convDesc: cudnnConvolutionDescriptor_t,
    gradDesc: cudnnFilterDescriptor_t,
    algo: cudnnConvolutionBwdFilterAlgo_t,
    sizeInBytes: *mut usize,
) -> cudnnStatus_t {
    let sym: Symbol<cudnnGetConvolutionBackwardFilterWorkspaceSize_fn> = unsafe {
        get_lib()
            .get(b"cudnnGetConvolutionBackwardFilterWorkspaceSize\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(handle, xDesc, dyDesc, convDesc, gradDesc, algo, sizeInBytes) }
}

type cudnnConvolutionBackwardFilter_fn = unsafe extern "C" fn(
    handle: cudnnHandle_t,
    alpha: *const ::core::ffi::c_void,
    xDesc: cudnnTensorDescriptor_t,
    x: *const ::core::ffi::c_void,
    dyDesc: cudnnTensorDescriptor_t,
    dy: *const ::core::ffi::c_void,
    convDesc: cudnnConvolutionDescriptor_t,
    algo: cudnnConvolutionBwdFilterAlgo_t,
    workSpace: *mut ::core::ffi::c_void,
    workSpaceSizeInBytes: usize,
    beta: *const ::core::ffi::c_void,
    dwDesc: cudnnFilterDescriptor_t,
    dw: *mut ::core::ffi::c_void,
) -> cudnnStatus_t;
pub unsafe fn cudnnConvolutionBackwardFilter(
    handle: cudnnHandle_t,
    alpha: *const ::core::ffi::c_void,
    xDesc: cudnnTensorDescriptor_t,
    x: *const ::core::ffi::c_void,
    dyDesc: cudnnTensorDescriptor_t,
    dy: *const ::core::ffi::c_void,
    convDesc: cudnnConvolutionDescriptor_t,
    algo: cudnnConvolutionBwdFilterAlgo_t,
    workSpace: *mut ::core::ffi::c_void,
    workSpaceSizeInBytes: usize,
    beta: *const ::core::ffi::c_void,
    dwDesc: cudnnFilterDescriptor_t,
    dw: *mut ::core::ffi::c_void,
) -> cudnnStatus_t {
    let sym: Symbol<cudnnConvolutionBackwardFilter_fn> = unsafe {
        get_lib()
            .get(b"cudnnConvolutionBackwardFilter\0")
            .expect("Missing symbol")
    };
    unsafe {
        (*sym)(
            handle,
            alpha,
            xDesc,
            x,
            dyDesc,
            dy,
            convDesc,
            algo,
            workSpace,
            workSpaceSizeInBytes,
            beta,
            dwDesc,
            dw,
        )
    }
}

type cudnnConvolutionBackwardBias_fn = unsafe extern "C" fn(
    handle: cudnnHandle_t,
    alpha: *const ::core::ffi::c_void,
    dyDesc: cudnnTensorDescriptor_t,
    dy: *const ::core::ffi::c_void,
    beta: *const ::core::ffi::c_void,
    dbDesc: cudnnTensorDescriptor_t,
    db: *mut ::core::ffi::c_void,
) -> cudnnStatus_t;
pub unsafe fn cudnnConvolutionBackwardBias(
    handle: cudnnHandle_t,
    alpha: *const ::core::ffi::c_void,
    dyDesc: cudnnTensorDescriptor_t,
    dy: *const ::core::ffi::c_void,
    beta: *const ::core::ffi::c_void,
    dbDesc: cudnnTensorDescriptor_t,
    db: *mut ::core::ffi::c_void,
) -> cudnnStatus_t {
    let sym: Symbol<cudnnConvolutionBackwardBias_fn> = unsafe {
        get_lib()
            .get(b"cudnnConvolutionBackwardBias\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(handle, alpha, dyDesc, dy, beta, dbDesc, db) }
}

type cudnnCreateFusedOpsConstParamPack_fn = unsafe extern "C" fn(
    constPack: *mut cudnnFusedOpsConstParamPack_t,
    ops: cudnnFusedOps_t,
) -> cudnnStatus_t;
pub unsafe fn cudnnCreateFusedOpsConstParamPack(
    constPack: *mut cudnnFusedOpsConstParamPack_t,
    ops: cudnnFusedOps_t,
) -> cudnnStatus_t {
    let sym: Symbol<cudnnCreateFusedOpsConstParamPack_fn> = unsafe {
        get_lib()
            .get(b"cudnnCreateFusedOpsConstParamPack\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(constPack, ops) }
}

type cudnnDestroyFusedOpsConstParamPack_fn =
    unsafe extern "C" fn(constPack: cudnnFusedOpsConstParamPack_t) -> cudnnStatus_t;
pub unsafe fn cudnnDestroyFusedOpsConstParamPack(
    constPack: cudnnFusedOpsConstParamPack_t,
) -> cudnnStatus_t {
    let sym: Symbol<cudnnDestroyFusedOpsConstParamPack_fn> = unsafe {
        get_lib()
            .get(b"cudnnDestroyFusedOpsConstParamPack\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(constPack) }
}

type cudnnSetFusedOpsConstParamPackAttribute_fn = unsafe extern "C" fn(
    constPack: cudnnFusedOpsConstParamPack_t,
    paramLabel: cudnnFusedOpsConstParamLabel_t,
    param: *const ::core::ffi::c_void,
) -> cudnnStatus_t;
pub unsafe fn cudnnSetFusedOpsConstParamPackAttribute(
    constPack: cudnnFusedOpsConstParamPack_t,
    paramLabel: cudnnFusedOpsConstParamLabel_t,
    param: *const ::core::ffi::c_void,
) -> cudnnStatus_t {
    let sym: Symbol<cudnnSetFusedOpsConstParamPackAttribute_fn> = unsafe {
        get_lib()
            .get(b"cudnnSetFusedOpsConstParamPackAttribute\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(constPack, paramLabel, param) }
}

type cudnnGetFusedOpsConstParamPackAttribute_fn = unsafe extern "C" fn(
    constPack: cudnnFusedOpsConstParamPack_t,
    paramLabel: cudnnFusedOpsConstParamLabel_t,
    param: *mut ::core::ffi::c_void,
    isNULL: *mut ::core::ffi::c_int,
) -> cudnnStatus_t;
pub unsafe fn cudnnGetFusedOpsConstParamPackAttribute(
    constPack: cudnnFusedOpsConstParamPack_t,
    paramLabel: cudnnFusedOpsConstParamLabel_t,
    param: *mut ::core::ffi::c_void,
    isNULL: *mut ::core::ffi::c_int,
) -> cudnnStatus_t {
    let sym: Symbol<cudnnGetFusedOpsConstParamPackAttribute_fn> = unsafe {
        get_lib()
            .get(b"cudnnGetFusedOpsConstParamPackAttribute\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(constPack, paramLabel, param, isNULL) }
}

type cudnnCreateFusedOpsVariantParamPack_fn = unsafe extern "C" fn(
    varPack: *mut cudnnFusedOpsVariantParamPack_t,
    ops: cudnnFusedOps_t,
) -> cudnnStatus_t;
pub unsafe fn cudnnCreateFusedOpsVariantParamPack(
    varPack: *mut cudnnFusedOpsVariantParamPack_t,
    ops: cudnnFusedOps_t,
) -> cudnnStatus_t {
    let sym: Symbol<cudnnCreateFusedOpsVariantParamPack_fn> = unsafe {
        get_lib()
            .get(b"cudnnCreateFusedOpsVariantParamPack\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(varPack, ops) }
}

type cudnnDestroyFusedOpsVariantParamPack_fn =
    unsafe extern "C" fn(varPack: cudnnFusedOpsVariantParamPack_t) -> cudnnStatus_t;
pub unsafe fn cudnnDestroyFusedOpsVariantParamPack(
    varPack: cudnnFusedOpsVariantParamPack_t,
) -> cudnnStatus_t {
    let sym: Symbol<cudnnDestroyFusedOpsVariantParamPack_fn> = unsafe {
        get_lib()
            .get(b"cudnnDestroyFusedOpsVariantParamPack\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(varPack) }
}

type cudnnSetFusedOpsVariantParamPackAttribute_fn = unsafe extern "C" fn(
    varPack: cudnnFusedOpsVariantParamPack_t,
    paramLabel: cudnnFusedOpsVariantParamLabel_t,
    ptr: *mut ::core::ffi::c_void,
) -> cudnnStatus_t;
pub unsafe fn cudnnSetFusedOpsVariantParamPackAttribute(
    varPack: cudnnFusedOpsVariantParamPack_t,
    paramLabel: cudnnFusedOpsVariantParamLabel_t,
    ptr: *mut ::core::ffi::c_void,
) -> cudnnStatus_t {
    let sym: Symbol<cudnnSetFusedOpsVariantParamPackAttribute_fn> = unsafe {
        get_lib()
            .get(b"cudnnSetFusedOpsVariantParamPackAttribute\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(varPack, paramLabel, ptr) }
}

type cudnnGetFusedOpsVariantParamPackAttribute_fn = unsafe extern "C" fn(
    varPack: cudnnFusedOpsVariantParamPack_t,
    paramLabel: cudnnFusedOpsVariantParamLabel_t,
    ptr: *mut ::core::ffi::c_void,
) -> cudnnStatus_t;
pub unsafe fn cudnnGetFusedOpsVariantParamPackAttribute(
    varPack: cudnnFusedOpsVariantParamPack_t,
    paramLabel: cudnnFusedOpsVariantParamLabel_t,
    ptr: *mut ::core::ffi::c_void,
) -> cudnnStatus_t {
    let sym: Symbol<cudnnGetFusedOpsVariantParamPackAttribute_fn> = unsafe {
        get_lib()
            .get(b"cudnnGetFusedOpsVariantParamPackAttribute\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(varPack, paramLabel, ptr) }
}

type cudnnCreateFusedOpsPlan_fn =
    unsafe extern "C" fn(plan: *mut cudnnFusedOpsPlan_t, ops: cudnnFusedOps_t) -> cudnnStatus_t;
pub unsafe fn cudnnCreateFusedOpsPlan(
    plan: *mut cudnnFusedOpsPlan_t,
    ops: cudnnFusedOps_t,
) -> cudnnStatus_t {
    let sym: Symbol<cudnnCreateFusedOpsPlan_fn> = unsafe {
        get_lib()
            .get(b"cudnnCreateFusedOpsPlan\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(plan, ops) }
}

type cudnnDestroyFusedOpsPlan_fn = unsafe extern "C" fn(plan: cudnnFusedOpsPlan_t) -> cudnnStatus_t;
pub unsafe fn cudnnDestroyFusedOpsPlan(plan: cudnnFusedOpsPlan_t) -> cudnnStatus_t {
    let sym: Symbol<cudnnDestroyFusedOpsPlan_fn> = unsafe {
        get_lib()
            .get(b"cudnnDestroyFusedOpsPlan\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(plan) }
}

type cudnnMakeFusedOpsPlan_fn = unsafe extern "C" fn(
    handle: cudnnHandle_t,
    plan: cudnnFusedOpsPlan_t,
    constPack: cudnnFusedOpsConstParamPack_t,
    workspaceSizeInBytes: *mut usize,
) -> cudnnStatus_t;
pub unsafe fn cudnnMakeFusedOpsPlan(
    handle: cudnnHandle_t,
    plan: cudnnFusedOpsPlan_t,
    constPack: cudnnFusedOpsConstParamPack_t,
    workspaceSizeInBytes: *mut usize,
) -> cudnnStatus_t {
    let sym: Symbol<cudnnMakeFusedOpsPlan_fn> = unsafe {
        get_lib()
            .get(b"cudnnMakeFusedOpsPlan\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(handle, plan, constPack, workspaceSizeInBytes) }
}

type cudnnFusedOpsExecute_fn = unsafe extern "C" fn(
    handle: cudnnHandle_t,
    plan: cudnnFusedOpsPlan_t,
    varPack: cudnnFusedOpsVariantParamPack_t,
) -> cudnnStatus_t;
pub unsafe fn cudnnFusedOpsExecute(
    handle: cudnnHandle_t,
    plan: cudnnFusedOpsPlan_t,
    varPack: cudnnFusedOpsVariantParamPack_t,
) -> cudnnStatus_t {
    let sym: Symbol<cudnnFusedOpsExecute_fn> = unsafe {
        get_lib()
            .get(b"cudnnFusedOpsExecute\0")
            .expect("Missing symbol")
    };
    unsafe { (*sym)(handle, plan, varPack) }
}
