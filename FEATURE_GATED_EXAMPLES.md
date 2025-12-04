# Feature-Gated Code Generation Examples

This document shows how generated code varies based on the selected safety mode.

## Example Function: `cudnnSetTensor4dDescriptor`

### Source FFI Function
```rust
// From bindgen
extern "C" {
    pub fn cudnnSetTensor4dDescriptor(
        tensorDesc: cudnnTensorDescriptor_t,
        format: cudnnTensorFormat_t,
        dataType: cudnnDataType_t,
        n: ::std::os::raw::c_int,
        c: ::std::os::raw::c_int,
        h: ::std::os::raw::c_int,
        w: ::std::os::raw::c_int,
    ) -> cudnnStatus_t;
}
```

## Generated Safe Wrapper Code

### With `balanced` Mode (Default)

```rust
pub fn set_tensor_4d_descriptor(
    &mut self,
    format: TensorFormat,
    data_type: DataType,
    n: i32,
    c: i32,
    h: i32,
    w: i32,
) -> Result<(), Error> {
    unsafe {
        // Null pointer checks for required parameters (balanced mode)
        #[cfg(all(feature = "balanced", not(feature = "strict")))]
        if self.handle.is_null() {
            return Err(Error::NullPointer);
        }
        
        // Numeric constraint validation (balanced mode)
        #[cfg(any(feature = "strict", feature = "balanced"))]
        if n as i64 < 1 || n as i64 > 1024 {
            return Err(Error::InvalidParameter);
        }
        
        #[cfg(any(feature = "strict", feature = "balanced"))]
        if c as i64 < 1 || c as i64 > 1024 {
            return Err(Error::InvalidParameter);
        }
        
        #[cfg(any(feature = "strict", feature = "balanced"))]
        if h as i64 < 1 || h as i64 > 8192 {
            return Err(Error::InvalidParameter);
        }
        
        #[cfg(any(feature = "strict", feature = "balanced"))]
        if w as i64 < 1 || w as i64 > 8192 {
            return Err(Error::InvalidParameter);
        }
        
        // Debug assertions (always active in debug builds)
        debug_assert!(n > 0, "n must be positive");
        debug_assert!(c > 0, "c must be positive");
        debug_assert!(h > 0, "h must be positive");
        debug_assert!(w > 0, "w must be positive");
        
        // Tracing span (if tracing feature enabled)
        #[cfg(feature = "tracing")]
        let _span = tracing::trace_span!("ffi_call", function = "cudnnSetTensor4dDescriptor").entered();
        
        let result = ffi::cudnnSetTensor4dDescriptor(
            self.handle,
            format.into(),
            data_type.into(),
            n,
            c,
            h,
            w,
        );
        
        if result == 0 {
            Ok(())
        } else {
            Err(Error::from(result))
        }
    }
}
```

### With `strict` Mode

```rust
pub fn set_tensor_4d_descriptor(
    &mut self,
    format: TensorFormat,
    data_type: DataType,
    n: i32,
    c: i32,
    h: i32,
    w: i32,
) -> Result<(), Error> {
    unsafe {
        // Null pointer checks for ALL parameters (strict mode)
        #[cfg(feature = "strict")]
        if self.handle.is_null() {
            return Err(Error::NullPointer);
        }
        
        // ALL validation checks active in strict mode
        #[cfg(any(feature = "strict", feature = "balanced"))]
        if n as i64 < 1 || n as i64 > 1024 {
            return Err(Error::InvalidParameter);
        }
        
        #[cfg(any(feature = "strict", feature = "balanced"))]
        if c as i64 < 1 || c as i64 > 1024 {
            return Err(Error::InvalidParameter);
        }
        
        #[cfg(any(feature = "strict", feature = "balanced"))]
        if h as i64 < 1 || h as i64 > 8192 {
            return Err(Error::InvalidParameter);
        }
        
        #[cfg(any(feature = "strict", feature = "balanced"))]
        if w as i64 < 1 || w as i64 > 8192 {
            return Err(Error::InvalidParameter);
        }
        
        // Debug assertions (active even in release builds with strict mode)
        debug_assert!(n > 0, "n must be positive");
        debug_assert!(c > 0, "c must be positive");
        debug_assert!(h > 0, "h must be positive");
        debug_assert!(w > 0, "w must be positive");
        
        #[cfg(feature = "tracing")]
        let _span = tracing::trace_span!("ffi_call", function = "cudnnSetTensor4dDescriptor").entered();
        
        let result = ffi::cudnnSetTensor4dDescriptor(
            self.handle,
            format.into(),
            data_type.into(),
            n,
            c,
            h,
            w,
        );
        
        if result == 0 {
            Ok(())
        } else {
            Err(Error::from(result))
        }
    }
}
```

### With `permissive` Mode

```rust
pub fn set_tensor_4d_descriptor(
    &mut self,
    format: TensorFormat,
    data_type: DataType,
    n: i32,
    c: i32,
    h: i32,
    w: i32,
) -> Result<(), Error> {
    unsafe {
        // NO null pointer checks (permissive mode trusts caller)
        
        // NO numeric validation (permissive mode)
        
        // Debug assertions still active in debug builds
        debug_assert!(n > 0, "n must be positive");
        debug_assert!(c > 0, "c must be positive");
        debug_assert!(h > 0, "h must be positive");
        debug_assert!(w > 0, "w must be positive");
        
        #[cfg(feature = "tracing")]
        let _span = tracing::trace_span!("ffi_call", function = "cudnnSetTensor4dDescriptor").entered();
        
        // Direct FFI call - maximum performance
        let result = ffi::cudnnSetTensor4dDescriptor(
            self.handle,
            format.into(),
            data_type.into(),
            n,
            c,
            h,
            w,
        );
        
        if result == 0 {
            Ok(())
        } else {
            Err(Error::from(result))
        }
    }
}
```

### With `tracing` Feature Enabled

When compiled with `--features balanced,tracing`:

```rust
pub fn set_tensor_4d_descriptor(
    &mut self,
    format: TensorFormat,
    data_type: DataType,
    n: i32,
    c: i32,
    h: i32,
    w: i32,
) -> Result<(), Error> {
    unsafe {
        // ... null checks and validation ...
        
        // Tracing is now active
        #[cfg(feature = "tracing")]
        let _span = tracing::trace_span!(
            "ffi_call",
            function = "cudnnSetTensor4dDescriptor",
            n = n,
            c = c,
            h = h,
            w = w
        ).entered();
        
        let result = ffi::cudnnSetTensor4dDescriptor(
            self.handle,
            format.into(),
            data_type.into(),
            n,
            c,
            h,
            w,
        );
        
        // Tracing will automatically log when the span is dropped
        
        if result == 0 {
            Ok(())
        } else {
            Err(Error::from(result))
        }
    }
}
```

## Cargo.toml Configuration

Generated crates will have:

```toml
[features]
# Safety modes (mutually exclusive)
strict = []
balanced = []
permissive = []

# Additional features
debug-extra = []  # Extra runtime checks in debug builds
tracing = ["dep:tracing"]  # Structured logging for FFI calls
leak-detector = []  # Track resource allocations in debug

# Default: balanced mode
default = ["balanced"]

[dependencies]
# Core dependencies always present
# ...

# Optional dependencies activated by features
tracing = { version = "0.1", optional = true }
```

## Usage Examples

### Development (Maximum Safety)
```toml
[dependencies]
cudnn-sys = { version = "0.1", features = ["strict", "tracing", "leak-detector"] }
```

### Production (Balanced)
```toml
[dependencies]
cudnn-sys = "0.1"  # Uses default = ["balanced"]
```

### Performance-Critical (Minimal Overhead)
```toml
[dependencies]
cudnn-sys = { version = "0.1", features = ["permissive"] }
```

## Performance Impact

| Mode | Null Checks | Range Checks | Debug Asserts | Release Overhead |
|------|-------------|--------------|---------------|------------------|
| **strict** | All params | All | Always | ~5-10% |
| **balanced** | Required only | All | Debug only | <1% |
| **permissive** | None | None | Debug only | 0% |

## Safety vs Performance Tradeoff

```
Safety          ◄─────────────────────────────────────► Performance
Strict          Balanced (Default)                      Permissive
│               │                                        │
├─ Always safe  ├─ Safe with good performance          ├─ Maximum speed
├─ May reject   ├─ Practical for most use cases        ├─ Trust caller
│  valid code   ├─ Recommended for production          ├─ Expert users
└─ Development  └─ Zero overhead in release            └─ Profiled code
```

## Recommendations

1. **Start with `balanced`** - Good default for most projects
2. **Use `strict` during development** - Catch bugs early
3. **Switch to `permissive` only after profiling** - Only if needed
4. **Always enable `tracing` when debugging** - Invaluable for FFI issues
5. **Use `leak-detector` in test suites** - Catch resource leaks early
