# Safe Wrapper Enrichment Implementation Plan

## Executive Summary

The safe wrapper generators (wrappers.rs, methods.rs, builders.rs) are currently using **only 26% of available enrichment data comprehensively**. This plan outlines systematic improvements to leverage all enrichment fields, making the safe wrappers as "spit-and-polish perfect" as the FFI layer.

## Current State Analysis

### Enrichment Fields: 19 Total

**Fully Utilized (5 fields, 26%)**:
- ✅ `thread_safety` - Generates Send/Sync traits
- ✅ `preconditions` - Documentation only (no runtime validation!)
- ✅ `api_sequences` - Builder documentation
- ✅ `global_state` - Builder documentation
- ✅ `performance` - Builder and wrapper documentation

**Partially Utilized (5 fields, 26%)**:
- ⚠️ `description` - Used but not parameterized
- ⚠️ `ownership` - Used but incompletely
- ⚠️ `semantic_group` - Builder only
- ⚠️ `pitfalls` - Doc generator only
- ⚠️ `numeric_constraints` - Builder docs, no validation

**Completely Unused (9 fields, 47%)**:
- ❌ `parameters` (HashMap<String, String>) - Rich parameter docs
- ❌ `return_doc` - Return value documentation
- ❌ `test_cases` - Usage examples
- ❌ `attributes` - #[must_use], #[deprecated], etc.
- ❌ `platform` - Platform-specific warnings
- ❌ `version_history` - Deprecation notices
- ❌ `callback_info` - Callback lifetime/safety
- ❌ `resource_limits` - Resource constraint validation
- ❌ `error_semantics` - Specific error mappings

---

## Critical Issues

### Issue #1: No Runtime Validation
**Impact**: Critical - Allows invalid values to reach FFI layer

**Current State**:
- `numeric_constraints` contains ranges, alignment, power-of-two requirements
- `preconditions` contains non-null, non-zero, state requirements
- Builder setters accept ANY value without validation
- Methods don't validate parameters before FFI calls

**Example Gap**:
```rust
// Current (builders.rs ~lines 280-300)
pub fn with_size(mut self, size: usize) -> Self {
    self.size = Some(size);
    self  // NO VALIDATION! Even if constraints say "must be power of 2, >= 1024"
}
```

### Issue #2: Generic Error Handling Only
**Impact**: High - Users get unhelpful error messages

**Current State**:
- `error_semantics` contains specific error code meanings
- Code only does: `Err(Error::from(result))`
- No context about WHY an error occurred

**Example Gap**:
```rust
// Current (methods.rs ~line 182)
if result == 0 {
    Ok(())
} else {
    Err(Error::from(result))  // What does this error mean? No context!
}
```

### Issue #3: Missing Parameter Documentation
**Impact**: Medium - Poor documentation for complex parameters

**Current State**:
- `parameters: HashMap<String, String>` completely unused
- Methods show type but not what the parameter means
- No constraints or acceptable ranges documented

### Issue #4: No Usage Examples
**Impact**: Medium - Users don't know how to use the API

**Current State**:
- `test_cases` contains actual usage patterns
- No examples generated in method/wrapper docs
- Users must guess correct API usage

### Issue #5: Missing Rust Attributes
**Impact**: Medium - Missing compile-time safety

**Current State**:
- `attributes` contains #[must_use], #[deprecated], etc.
- `version_history` contains deprecation info
- No attributes generated on methods/types

---

## Implementation Plan

### Phase 1: Critical Runtime Validation (P0)

#### 1.1 Builder Validation
**File**: `src/generator/builders.rs`
**Lines**: Setter methods (~280-350) and build() method (~420-470)

**Enhancements**:
1. Add validation in setter methods:
```rust
pub fn with_size(mut self, size: usize) -> Self {
    // NEW: Validate numeric constraints
    if let Some(ctx) = self.context {
        if let Some(constraints) = &ctx.numeric_constraints {
            for constraint in &constraints.constraints {
                if constraint.parameter_name.as_deref() == Some("size") {
                    // Validate range
                    if let (Some(min), Some(max)) = (constraint.min_value, constraint.max_value) {
                        assert!(size >= min as usize && size <= max as usize,
                                "size must be between {} and {}", min, max);
                    }
                    // Validate power of two
                    if constraint.must_be_power_of_two {
                        assert!(size.is_power_of_two(), "size must be a power of two");
                    }
                    // Validate alignment
                    if let Some(align) = constraint.must_be_aligned_to {
                        assert!(size % align as usize == 0, 
                                "size must be aligned to {}", align);
                    }
                }
            }
        }
    }
    self.size = Some(size);
    self
}
```

2. Add precondition validation in build():
```rust
pub fn build(self) -> Result<WrapperType, Error> {
    // NEW: Validate preconditions
    if let Some(ctx) = self.context {
        if let Some(precond) = &ctx.preconditions {
            // Validate non-null
            for param in &precond.non_null_params {
                // Check corresponding Option<T> field is Some
            }
            // Validate non-zero
            for param in &precond.non_zero_params {
                // Check value != 0
            }
        }
    }
    
    // Existing validation
    // ... rest of build logic
}
```

#### 1.2 Method Pre-call Validation
**File**: `src/generator/methods.rs`
**Lines**: Method generation (~120-190)

**Enhancements**:
1. Add parameter validation before FFI call:
```rust
pub fn set_config(&mut self, flags: u32, mode: i32) -> Result<(), Error> {
    // NEW: Validate numeric constraints
    validate_numeric_constraints(&flags, "flags", &self.context)?;
    validate_numeric_constraints(&mode, "mode", &self.context)?;
    
    unsafe {
        let result = ffi::cudnnSetConfig(self.handle, flags, mode);
        // ... rest of method
    }
}
```

2. Generate helper validation functions:
```rust
fn validate_numeric_constraints<T>(value: &T, param_name: &str, ctx: Option<&FunctionContext>) -> Result<(), Error> {
    if let Some(ctx) = ctx {
        if let Some(constraints) = &ctx.numeric_constraints {
            for constraint in &constraints.constraints {
                if constraint.parameter_name.as_deref() == Some(param_name) {
                    // Perform validation
                }
            }
        }
    }
    Ok(())
}
```

#### 1.3 Enhanced Error Handling
**File**: `src/generator/methods.rs` and `src/generator/wrappers.rs`
**Lines**: Error handling blocks (~180-185, ~600-650)

**Enhancements**:
1. Use `error_semantics` for context-aware errors:
```rust
// Instead of:
if result == 0 {
    Ok(())
} else {
    Err(Error::from(result))
}

// Generate:
if result == 0 {
    Ok(())
} else {
    // NEW: Use error_semantics for rich error info
    match result {
        1 => Err(Error::NotInitialized {
            context: "cudnnCreate must be called before this function",
            error_code: result,
        }),
        2 => Err(Error::InvalidValue {
            context: "one or more parameters are out of valid range",
            parameter: Some("see function preconditions"),
            error_code: result,
        }),
        _ => Err(Error::Unknown(result)),
    }
}
```

### Phase 2: Documentation Enhancements (P1)

#### 2.1 Parameter Documentation
**File**: `src/generator/methods.rs` and `src/generator/doc_generator.rs`
**Lines**: Documentation generation (~85-105)

**Enhancements**:
1. Use `parameters` HashMap:
```rust
// For each parameter:
if let Some(param_doc) = context.parameters.get(&param.name) {
    writeln!(code, "/// - `{}`: {}", param.name, param_doc)?;
    
    // Add constraint info
    if let Some(constraints) = &context.numeric_constraints {
        for constraint in &constraints.constraints {
            if constraint.parameter_name.as_deref() == Some(&param.name) {
                if let (Some(min), Some(max)) = (constraint.min_value, constraint.max_value) {
                    writeln!(code, "///   - Valid range: [{}, {}]", min, max)?;
                }
                if constraint.must_be_power_of_two {
                    writeln!(code, "///   - Must be a power of two")?;
                }
            }
        }
    }
}
```

#### 2.2 Return Value Documentation
**File**: `src/generator/methods.rs` and `src/generator/doc_generator.rs`

**Enhancements**:
1. Use `return_doc`:
```rust
if let Some(return_doc) = &context.return_doc {
    writeln!(code, "///")?;
    writeln!(code, "/// # Returns")?;
    writeln!(code, "///")?;
    writeln!(code, "/// {}", return_doc)?;
    
    // Add success/error info from error_semantics
    if let Some(error_info) = &context.error_semantics {
        writeln!(code, "///")?;
        writeln!(code, "/// # Errors")?;
        writeln!(code, "///")?;
        for (code_val, meaning) in &error_info.error_codes {
            writeln!(code, "/// - `{}`: {}", code_val, meaning)?;
        }
    }
}
```

#### 2.3 Usage Examples
**File**: `src/generator/doc_generator.rs` and wrapper/method generators

**Enhancements**:
1. Use `test_cases`:
```rust
if let Some(test_info) = &context.test_cases {
    writeln!(code, "///")?;
    writeln!(code, "/// # Examples")?;
    writeln!(code, "///")?;
    
    // Generate example from test patterns
    writeln!(code, "/// ```rust")?;
    writeln!(code, "/// {}", test_info.example_usage)?;
    writeln!(code, "/// ```")?;
}
```

### Phase 3: Rust Attributes & Metadata (P2)

#### 3.1 Add Rust Attributes
**File**: All generators

**Enhancements**:
1. Use `attributes` field:
```rust
if let Some(attr_info) = &context.attributes {
    if attr_info.must_use {
        writeln!(code, "#[must_use]")?;
    }
    if attr_info.deprecated {
        if let Some(msg) = &attr_info.deprecation_message {
            writeln!(code, "#[deprecated(note = \"{}\")]", msg)?;
        } else {
            writeln!(code, "#[deprecated]")?;
        }
    }
}
```

#### 3.2 Version & Deprecation Warnings
**File**: All generators

**Enhancements**:
1. Use `version_history`:
```rust
if !context.version_history.is_empty() {
    for deprecation in &context.version_history {
        writeln!(code, "///")?;
        writeln!(code, "/// ⚠️  **Deprecated**: {}", deprecation.reason)?;
        if let Some(alternative) = &deprecation.alternative {
            writeln!(code, "/// Use `{}` instead", alternative)?;
        }
    }
}
```

#### 3.3 Platform Awareness
**File**: `src/generator/wrappers.rs` and `src/generator/methods.rs`

**Enhancements**:
1. Use `platform`:
```rust
if let Some(platform_info) = &context.platform {
    // Add conditional compilation
    if !platform_info.supported_platforms.is_empty() {
        let platforms = platform_info.supported_platforms.join("\", \"");
        writeln!(code, "#[cfg(target_os = \"{}\")]", platforms)?;
    }
    
    // Add platform notes in docs
    writeln!(code, "///")?;
    writeln!(code, "/// # Platform Support")?;
    for platform in &platform_info.supported_platforms {
        writeln!(code, "/// - {}", platform)?;
    }
}
```

### Phase 4: Specialized Features (P2)

#### 4.1 Callback Safety
**File**: `src/generator/wrappers.rs` and `src/generator/methods.rs`

**Enhancements**:
1. Use `callback_info`:
```rust
if let Some(callback) = &context.callback_info {
    writeln!(code, "///")?;
    writeln!(code, "/// # Callback Safety")?;
    writeln!(code, "///")?;
    writeln!(code, "/// Callback lifetime: {:?}", callback.lifetime)?;
    writeln!(code, "/// Invocation: {:?}", callback.invocation_count)?;
    
    if callback.can_be_recursive {
        writeln!(code, "/// ⚠️  Callback may be invoked recursively")?;
    }
    if callback.must_not_block {
        writeln!(code, "/// ⚠️  Callback must not block")?;
    }
}
```

#### 4.2 Resource Limit Enforcement
**File**: `src/generator/builders.rs` and `src/generator/wrappers.rs`

**Enhancements**:
1. Use `resource_limits`:
```rust
if let Some(limits) = &context.resource_limits {
    // Document limits
    writeln!(code, "///")?;
    writeln!(code, "/// # Resource Limits")?;
    for limit in &limits.limits {
        writeln!(code, "/// - {}: {}", limit.resource_type, limit.description)?;
        if let Some(max) = limit.max_count {
            writeln!(code, "///   Maximum: {}", max)?;
        }
    }
    
    // Add runtime checks if possible
    if let Some(max_connections) = limits.limits.iter()
        .find(|l| l.resource_type == "connections")
        .and_then(|l| l.max_count) 
    {
        // Generate connection counting logic
    }
}
```

---

## Integration with Polished FFI Layer

### Using Rust-Style Aliases

**Current Issue**: Wrappers use raw C names like `ffi::cudnnCreate`

**Enhancement**: Use the new Rust-style aliases:
```rust
// Instead of:
let result = ffi::cudnnCreate(&mut handle);

// Use:
let result = ffi::create_handle(&mut handle);
```

**Implementation**:
1. Update all FFI call sites in wrappers.rs, methods.rs, builders.rs
2. Detect which alias to use based on function name
3. Fall back to C name if alias doesn't exist

---

## Success Metrics

### Before (Current State)
- 26% enrichment utilization
- No runtime validation
- Generic errors only
- Minimal parameter documentation
- No usage examples
- No Rust attributes

### After (Target State)
- 95%+ enrichment utilization
- Comprehensive runtime validation
- Context-aware error messages
- Rich parameter documentation with constraints
- Usage examples from test cases
- Full Rust attributes (#[must_use], #[deprecated], etc.)
- Platform-aware compilation
- Resource limit enforcement
- Callback safety documentation

---

## Implementation Priority

### Week 1: Critical Runtime Validation (P0)
1. Builder validation (numeric constraints + preconditions)
2. Method pre-call validation
3. Enhanced error handling with error_semantics

### Week 2: Documentation Enhancements (P1)
1. Parameter documentation from parameters HashMap
2. Return value documentation
3. Usage examples from test_cases

### Week 3: Attributes & Metadata (P2)
1. Rust attributes generation
2. Version/deprecation warnings
3. Platform awareness

### Week 4: Specialized Features (P2)
1. Callback safety documentation
2. Resource limit enforcement
3. FFI alias integration

---

## Testing Strategy

1. **Unit Tests**: Validate each enrichment field is used
2. **Integration Tests**: Ensure wrappers work with real libraries
3. **Documentation Tests**: Verify all doc comments compile
4. **Comparison Tests**: Before/after wrapper quality metrics

---

## Notes

- All enhancements should be backward compatible
- Generate fallback code when enrichment data unavailable
- Maintain zero-cost abstraction principles
- Preserve existing functionality while adding improvements
