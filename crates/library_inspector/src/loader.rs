use anyhow::Result;
use libloading::Library;
use sha2::{Digest, Sha256};
use std::ffi::c_void;
use std::path::PathBuf;
use std::sync::Arc;

/// Loader options to control prompting and offline behavior.
#[derive(Debug, Clone, Default)]
pub struct LoaderOptions {
    pub no_prompt: bool,
    pub offline: bool,
}

/// Runtime loader wrapper around `libloading::Library` with a small symbol cache
/// and checksum verification support.
#[derive(Clone)]
pub struct Loader {
    lib: Arc<Library>,
    pub path: PathBuf,
    pub checksum: String,
    // symbol_cache: Arc<Mutex<HashMap<String, *mut c_void>>>,
    pub options: LoaderOptions,
}

impl Loader {
    /// Load a library from `path`. If `verify_checksum` is Some, the loader will
    /// compute the SHA-256 of the file and compare it to the expected value.
    pub fn load<P: AsRef<std::path::Path>>(path: P, verify_checksum: Option<&str>) -> Result<Self> {
        Self::load_with_options(path, verify_checksum, LoaderOptions::default())
    }

    /// Load with explicit options
    pub fn load_with_options<P: AsRef<std::path::Path>>(
        path: P,
        verify_checksum: Option<&str>,
        options: LoaderOptions,
    ) -> Result<Self> {
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

        // Safety: calling Library::new on a known path
        let lib = unsafe { Library::new(&path)? };

        Ok(Loader {
            lib: Arc::new(lib),
            path,
            checksum,
            options,
        })
    }

    // Note: symbol cache removed to simplify safe typed lookups. Use
    // `libloading::Library::get` directly for typed function pointers.

    /// Get a typed function pointer for `name`.
    ///
    /// # Safety
    /// Caller must ensure `T` is the correct function pointer type matching the
    /// symbol's ABI and signature. The returned `Option<T>` is `None` when the
    /// symbol is not found.
    pub unsafe fn get_typed<T: Copy>(&self, name: &str) -> Result<Option<T>> {
        // Attempt to get the symbol as type `T` directly from libloading. If
        // the symbol is not present an error is returned by `get`; map that
        // to `Ok(None)` so callers can distinguish missing symbols from
        // other failures if desired.
        match self.lib.get::<T>(name.as_bytes()) {
            Ok(sym) => Ok(Some(*sym)),
            Err(_) => Ok(None),
        }
    }

    /// Convenience wrapper: get raw address (unsafe)
    pub unsafe fn get_symbol_addr(&self, name: &str) -> Result<*mut c_void> {
        match self.lib.get::<*mut c_void>(name.as_bytes()) {
            Ok(sym) => Ok(*sym),
            Err(e) => Err(e.into()),
        }
    }
}
