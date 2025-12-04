// Runtime loader for cudnn64_9
use once_cell::sync::OnceCell;
use libloading::Library;
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
