use anyhow::Result;
use hex;
use libloading::Library;
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// Simple runtime loader wrapper around `libloading::Library`.
///
/// Provides basic checksum verification and a safe handle to resolve symbols.
#[derive(Clone)]
pub struct Loader {
    lib: Arc<Library>,
    pub path: PathBuf,
    pub checksum: String,
}

impl Loader {
    /// Load a library from `path`. If `verify_checksum` is Some, the loader will
    /// compute the SHA-256 of the file and compare it to the expected value.
    pub fn load<P: AsRef<Path>>(path: P, verify_checksum: Option<&str>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let bytes = std::fs::read(&path)?;
        let mut hasher = Sha256::new();
        hasher.update(&bytes);
        let checksum = hex::encode(hasher.finalize());

        if let Some(expected) = verify_checksum {
            if expected != checksum {
                anyhow::bail!("checksum mismatch: expected {} got {}", expected, checksum);
            }
        }

        // Safety: libloading::Library::new is unsafe but we only call it on a path
        // provided by the user/test harness; callers must ensure it's a real library.
        let lib = unsafe { Library::new(&path)? };
        Ok(Loader {
            lib: Arc::new(lib),
            path,
            checksum,
        })
    }

    /// Get the raw address of a symbol. The pointer is valid while the Loader
    /// is alive (because it owns the `Library`). This returns a raw `*mut c_void`.
    ///
    /// # Safety
    /// Caller must transmute the pointer to an appropriate function pointer
    /// type before calling.
    pub unsafe fn get_symbol_addr(&self, name: &str) -> Result<*mut std::ffi::c_void> {
        let sym = self.lib.get::<*mut std::ffi::c_void>(name.as_bytes())?;
        // deref the Symbol to get the raw pointer value
        Ok(*sym)
    }
}
