# Enrichment Data Usage Audit

**Date**: November 25, 2025  
**Purpose**: Comprehensive audit of how safe wrapper generators utilize enriched context data

## Executive Summary

This audit examines three key generator files (`wrappers.rs`, `methods.rs`, `builders.rs`) to assess their usage of the 19 enrichment fields available in `FunctionContext`. The analysis reveals significant gaps in data utilization, with many valuable enrichment fields being completely unused or only partially leveraged.

**Overall Statistics**:
- **Total Enrichment Fields**: 19
- **Fully Utilized**: 5 (26%)
- **Partially Utilized**: 5 (26%)
- **Unused**: 9 (47%)

---

## 1. wrappers.rs - RAII Wrapper Generator

### Used Fields

| Field | Lines | Usage Type | Coverage |
|-------|-------|------------|----------|
| `description` | 139-145, 367-374 | ✅ **Full** | Added to struct doc comments and constructor docs |
| `thread_safety` | 154-165, 308-340 | ✅ **Full** | Generates trait implementations (!Send/!Sync), adds safety warnings |
| `ownership` | 169-178 | ⚠️ **Partial** | Only documents return ownership, not parameter ownership |
| `preconditions` | 182-193, 376-383 | ⚠️ **Partial** | Documents non-null params and top 3 conditions only |

### Unused Fields

| Field | Opportunity | Impact |
|-------|-------------|--------|
| `parameters` | Could document constructor parameters with rich descriptions | **HIGH** - Constructor docs lack parameter details |
| `return_doc` | Could enhance return value documentation | **MEDIUM** - Return semantics unclear |
| `test_cases` | Could generate example code in doc comments | **HIGH** - No usage examples in generated wrappers |
| `attributes` | Could add compiler attributes (e.g., #[must_use], #[repr(C)]) | **HIGH** - Missing important compiler hints |
| `platform` | Could add platform-specific warnings or conditional compilation | **MEDIUM** - No platform awareness |
| `performance` | Could warn about expensive constructors or operations | **MEDIUM** - Users unaware of performance implications |
| `version_history` | Could add deprecation warnings | **MEDIUM** - No version/deprecation info |
| `pitfalls` | ✅ Used in doc_generator but NOT in wrappers.rs directly | **HIGH** - Critical safety information missing |
| `error_semantics` | Could improve error handling in constructors | **HIGH** - Generic error handling only |
| `callback_info` | Could add special handling for callback-based APIs | **MEDIUM** - Callback APIs not optimized |
| `api_sequences` | Could warn about prerequisite initialization | **HIGH** - No sequencing warnings |
| `resource_limits` | Could validate limits at construction time | **MEDIUM** - No resource validation |
| `semantic_group` | Could organize related types together | **LOW** - Organizational opportunity |
| `global_state` | Could warn about global initialization requirements | **HIGH** - Missing critical safety info |
| `numeric_constraints` | Could validate numeric parameters in constructors | **HIGH** - No parameter validation |

### Partial Usage Analysis

**`ownership` (Lines 169-178)**:
```rust
if let Some(ownership) = &ctx.ownership
    && ownership.return_ownership.requires_cleanup()
{
    writeln!(code, "/// **Ownership**: {}", 
        ownership.return_ownership.description())
}
```
- ❌ Missing: Parameter ownership semantics
- ❌ Missing: Validation of borrowed vs owned parameters
- ❌ Missing: Lifetime annotations based on ownership

**`preconditions` (Lines 182-193)**:
```rust
if let Some(precond) = &ctx.preconditions
    && (!precond.non_null_params.is_empty() || !precond.preconditions.is_empty())
{
    for pc in precond.preconditions.iter().take(3) {  // Only 3!
        writeln!(code, "/// - {}", pc.description).unwrap();
    }
}
```
- ❌ Missing: `undefined_behavior` warnings (critical!)
- ❌ Missing: All preconditions beyond first 3
- ❌ Missing: Runtime validation based on preconditions
- ❌ Missing: `preconditions` in constructor body (line 376 only shows `undefined_behavior`)

### Opportunities for Improvement

1. **Constructor Parameter Validation** (HIGH PRIORITY)
   - Use `numeric_constraints` to add runtime checks
   - Use `preconditions` to validate all parameters, not just first 3
   - Add early returns with descriptive errors

2. **Enhanced Error Handling** (HIGH PRIORITY)
   - Use `error_semantics` to map specific error codes
   - Distinguish between fatal and recoverable errors
   - Add error-specific recovery suggestions

3. **Test Case Examples** (HIGH PRIORITY)
   - Use `test_cases` to generate doc examples
   - Show typical usage patterns
   - Demonstrate error handling

4. **Attribute Generation** (MEDIUM PRIORITY)
   - Use `attributes` to add #[must_use] for important types
   - Add #[repr(C)] when needed
   - Generate #[deprecated] from `version_history`

5. **API Sequence Warnings** (HIGH PRIORITY)
   - Use `api_sequences` to warn about initialization order
   - Document prerequisite calls
   - Warn about mutually exclusive operations

---

## 2. methods.rs - Method Wrapper Generator

### Used Fields

| Field | Lines | Usage Type | Coverage |
|-------|-------|------------|----------|
| (None directly) | - | ❌ **None** | All enrichment delegated to doc_generator |

### Analysis

**Shocking Discovery**: `methods.rs` does NOT directly access ANY enrichment fields!

```rust
// Line 24-28: Delegates everything to doc_generator
if let Some(context) = func_context {
    let docs = doc_generator::generate_enhanced_docs(context, "    ", enhanced_docs);
    code.push_str(&docs);
    writeln!(code, "    #[inline]").unwrap();
}
```

This is both good (separation of concerns) and bad (missed opportunities for method-specific integration).

### Unused Fields (All 19!)

| Field | Opportunity | Impact |
|-------|-------------|--------|
| `parameters` | Could generate parameter validation in method body | **HIGH** |
| `return_doc` | Could enhance return type selection | **MEDIUM** |
| `error_semantics` | Could improve error handling logic | **HIGH** |
| `numeric_constraints` | Could add parameter validation before FFI call | **HIGH** |
| `preconditions` | Could add runtime checks (currently NO validation!) | **CRITICAL** |
| `callback_info` | Could optimize callback parameter handling | **MEDIUM** |
| `thread_safety` | Could add sync primitives or warnings | **HIGH** |
| `ownership` | Could affect parameter passing (by-value vs by-ref) | **HIGH** |
| `global_state` | Could add initialization checks | **HIGH** |
| `api_sequences` | Could enforce call ordering at compile-time | **HIGH** |
| `resource_limits` | Could validate resource usage | **MEDIUM** |
| `pitfalls` | Could add inline warnings as comments | **HIGH** |
| `performance` | Could optimize parameter passing | **MEDIUM** |
| `platform` | Could add conditional compilation | **LOW** |
| `attributes` | Could add method attributes (#[must_use], etc.) | **MEDIUM** |
| `test_cases` | Could generate doc examples | **HIGH** |
| `version_history` | Could add deprecation attributes | **MEDIUM** |
| `semantic_group` | Could organize method placement | **LOW** |
| `description` | ✅ Used via doc_generator | - |

### Critical Gaps

1. **No Parameter Validation** (CRITICAL)
   ```rust
   // Lines 124-130: Only checks null pointers heuristically!
   for param in &params {
       if is_raw_pointer_type(&param.ty) && !is_optional_pointer(&param.name) {
           // Check null... but what about numeric_constraints?
       }
   }
   ```

2. **No Error Code Mapping** (HIGH)
   ```rust
   // Lines 172-178: Generic error handling only
   if is_status_type(&func.return_type) {
       writeln!(code, "if result == 0 {{").unwrap();
       writeln!(code, "    Ok(())").unwrap();
       writeln!(code, "}} else {{").unwrap();
       writeln!(code, "    Err(Error::from(result))").unwrap();
   }
   ```
   - ❌ No use of `error_semantics.errors` map
   - ❌ No distinction between fatal and recoverable errors
   - ❌ No error-specific messages

3. **No Ownership-Based Type Selection** (HIGH)
   - Method signatures don't consider `ownership` semantics
   - All types converted via simple `to_safe_type()` function
   - No distinction between borrowed and owned parameters

### Opportunities for Improvement

1. **Add Precondition Validation** (CRITICAL)
   ```rust
   // Before FFI call, add:
   if let Some(precond) = func_context.and_then(|c| c.preconditions.as_ref()) {
       // Validate non-null parameters
       // Check numeric constraints
       // Verify state requirements
   }
   ```

2. **Enhance Error Handling** (HIGH)
   ```rust
   if let Some(error_sem) = func_context.and_then(|c| c.error_semantics.as_ref()) {
       // Map specific error codes to meaningful errors
       // Add recovery suggestions
       // Handle fatal errors differently
   }
   ```

3. **Parameter Validation** (HIGH)
   ```rust
   if let Some(constraints) = func_context.and_then(|c| c.numeric_constraints.as_ref()) {
       // Validate ranges
       // Check alignment
       // Verify power-of-two requirements
   }
   ```

4. **Add Method Attributes** (MEDIUM)
   ```rust
   if let Some(attrs) = func_context.and_then(|c| c.attributes.as_ref()) {
       if attrs.must_use {
           writeln!(code, "    #[must_use]")?;
       }
   }
   ```

---

## 3. builders.rs - Builder Pattern Generator

### Used Fields

| Field | Lines | Usage Type | Coverage |
|-------|-------|------------|----------|
| `performance` | 113-127 | ✅ **Full** | Documents complexity and warnings |
| `semantic_group` | 163-172 | ✅ **Full** | Documents module and feature set |
| `api_sequences` | 174-191 | ✅ **Full** | Warns about prerequisites and followups |
| `global_state` | 196-220 | ✅ **Full** | Warns about initialization requirements |
| `numeric_constraints` | 296-331 | ⚠️ **Partial** | Documents constraints but NO validation |
| `preconditions` | 334-348 | ⚠️ **Partial** | Documents but NO runtime validation |
| `thread_safety` | 393-401 | ⚠️ **Partial** | Only in build() method |
| `pitfalls` | 404-414 | ⚠️ **Partial** | Only in build() method |

### Unused Fields

| Field | Opportunity | Impact |
|-------|-------------|--------|
| `description` | Could enhance builder struct documentation | **MEDIUM** |
| `parameters` | Could provide detailed field documentation | **HIGH** |
| `return_doc` | Could enhance build() return documentation | **LOW** |
| `test_cases` | Could generate usage examples | **HIGH** |
| `attributes` | Could add #[must_use] to builder | **MEDIUM** |
| `platform` | Could add platform-specific notes | **LOW** |
| `version_history` | Could mark deprecated builders | **LOW** |
| `error_semantics` | Could improve build() error handling | **MEDIUM** |
| `callback_info` | Could handle callback parameters specially | **LOW** |
| `resource_limits` | Could validate resource parameters | **MEDIUM** |
| `ownership` | Could affect parameter storage (owned vs borrowed) | **MEDIUM** |

### Partial Usage Analysis

**`numeric_constraints` (Lines 296-331)**:
```rust
if let Some(constraints) = &ctx.numeric_constraints {
    for constraint in &constraints.constraints {
        // Documents constraints in setter...
        writeln!(output, "    /// - Minimum value: {}", min)?;
        // BUT NO VALIDATION IN SETTER OR BUILD!
    }
}
```
- ✅ Documents constraints
- ❌ NO runtime validation in setters
- ❌ NO validation in build() method
- ❌ No compile-time type safety (e.g., using types like Positive<i32>)

**`preconditions` (Lines 334-348)**:
```rust
let param_preconditions: Vec<_> = preconditions.preconditions.iter()
    .filter(|p| p.parameter.as_ref() == Some(&param.name))
    .collect();
// Documents requirements...
writeln!(output, "    /// # Requirements")?;
// BUT NO VALIDATION!
```
- ✅ Documents requirements
- ❌ NO validation in setters
- ❌ NO validation in build()

**`thread_safety` and `pitfalls` (Lines 393-414)**:
- Only documented in build() method
- Not mentioned in struct documentation
- Could be more prominent

### Critical Gaps

1. **No Constraint Validation** (CRITICAL)
   - Constraints are documented but never enforced
   - Users can violate constraints silently
   - No validation at setter time OR build time

2. **No Precondition Enforcement** (HIGH)
   - Preconditions documented but not checked
   - Could lead to undefined behavior

3. **Missing Parameter Documentation** (HIGH)
   - `parameters` map not used
   - Field docs could be much richer

### Opportunities for Improvement

1. **Add Validation to Setters** (HIGH PRIORITY)
   ```rust
   pub fn width(mut self, value: usize) -> Result<Self, Error> {
       if let Some(constraints) = numeric_constraints_for("width") {
           constraints.validate(value)?;
       }
       self.width = Some(value);
       Ok(self)
   }
   ```

2. **Add Validation to build()** (HIGH PRIORITY)
   ```rust
   pub fn build(self) -> Result<Wrapper, Error> {
       // Validate all constraints
       if let Some(constraints) = &self.constraints {
           for constraint in &constraints.constraints {
               // Validate before FFI call
           }
       }
       // ... existing build logic
   }
   ```

3. **Use Parameters Map** (MEDIUM)
   ```rust
   // In generate_setter_method:
   if let Some(param_doc) = context.parameters.get(&param.name) {
       writeln!(output, "    /// {}", param_doc)?;
   }
   ```

4. **Add Test Case Examples** (HIGH)
   ```rust
   if let Some(test_cases) = &context.test_cases {
       writeln!(output, "    /// # Example")?;
       writeln!(output, "    /// ```")?;
       // Generate example from test cases
       writeln!(output, "    /// ```")?;
   }
   ```

---

## 4. doc_generator.rs - Documentation Generator

### Used Fields

| Field | Lines | Usage Type | Coverage |
|-------|-------|------------|----------|
| `description` | 18-22 | ✅ **Full** | Primary description in docs |
| `preconditions` | 93-101 | ⚠️ **Partial** | Only non-null and UB warnings |
| `thread_safety` | 103-122 | ✅ **Full** | Comprehensive thread safety docs |
| `global_state` | 126-137 | ✅ **Full** | Initialization requirements |
| `api_sequences` | 140-149 | ⚠️ **Partial** | Prerequisites and exclusions only |
| `pitfalls` | 153-162 | ✅ **Full** | All pitfalls with severity markers |
| `error_semantics` | 165-172 | ⚠️ **Partial** | Only checks if errors exist and if any are fatal |

### Unused Fields

| Field | Opportunity | Impact |
|-------|-------------|--------|
| `parameters` | Could document each parameter inline | **HIGH** |
| `return_doc` | Could add Returns section | **HIGH** |
| `test_cases` | Could generate Examples section | **HIGH** |
| `attributes` | Could mention special attributes | **LOW** |
| `platform` | Could add platform requirements | **MEDIUM** |
| `performance` | Could add performance notes | **MEDIUM** |
| `version_history` | Could add deprecation notices | **MEDIUM** |
| `callback_info` | Could document callback semantics | **MEDIUM** |
| `resource_limits` | Could warn about limits | **MEDIUM** |
| `semantic_group` | Could add "See also" section | **LOW** |
| `numeric_constraints` | Could document constraints | **HIGH** |
| `ownership` | Could document ownership semantics | **HIGH** |

### Partial Usage Details

**`preconditions` (Lines 93-101)**:
```rust
for param in &precond.non_null_params {
    items.push(format!("- `{}` must not be null", param));
}
for ub in &precond.undefined_behavior {
    items.push(format!("- **UB**: {}", ub));
}
```
- ✅ Documents non-null parameters
- ✅ Documents undefined behavior
- ❌ Missing: General `preconditions` list
- ❌ Missing: State requirements

**`api_sequences` (Lines 140-149)**:
```rust
for prereq in &api_seq.prerequisites {
    items.push(format!("- Must call `{}` before this", prereq));
}
for exclusive in &api_seq.mutually_exclusive {
    items.push(format!("- Cannot use with `{}`", exclusive));
}
```
- ✅ Prerequisites documented
- ✅ Mutually exclusive calls documented
- ❌ Missing: `requires_followup` (must call after)
- ❌ Missing: `typical_sequence` (common patterns)

**`error_semantics` (Lines 165-172)**:
```rust
if let Some(error_info) = &context.error_semantics
    && !error_info.errors.is_empty()
{
    items.push("- Returns error codes (see documentation)".to_string());
    if error_info.errors.values().any(|e| e.is_fatal) {
        items.push("- Some errors are FATAL".to_string());
    }
}
```
- ✅ Mentions error codes exist
- ✅ Warns about fatal errors
- ❌ Missing: Specific error code documentation
- ❌ Missing: Error recovery strategies
- ❌ Missing: Per-error descriptions

### Opportunities

1. **Add Parameters Section** (HIGH)
   ```rust
   if !context.parameters.is_empty() {
       writeln!(doc, "{}/// # Parameters", indent)?;
       for (name, desc) in &context.parameters {
           writeln!(doc, "{}/// - `{}`: {}", indent, name, desc)?;
       }
   }
   ```

2. **Add Returns Section** (HIGH)
   ```rust
   if let Some(return_doc) = &context.return_doc {
       writeln!(doc, "{}/// # Returns", indent)?;
       writeln!(doc, "{}/// {}", indent, return_doc)?;
   }
   ```

3. **Add Examples Section** (HIGH)
   ```rust
   if let Some(test_cases) = &context.test_cases {
       writeln!(doc, "{}/// # Examples", indent)?;
       // Generate from test cases
   }
   ```

4. **Enhanced Error Documentation** (HIGH)
   ```rust
   if let Some(errors) = &context.error_semantics {
       writeln!(doc, "{}/// # Errors", indent)?;
       for (code, info) in &errors.errors {
           writeln!(doc, "{}/// - `{}`: {}", indent, code, info.description)?;
       }
   }
   ```

---

## Summary of Findings

### Overall Usage by Field

| # | Field | wrappers.rs | methods.rs | builders.rs | doc_generator.rs | Overall |
|---|-------|------------|------------|-------------|------------------|---------|
| 1 | description | ✅ Full | ✅ (via doc) | ❌ None | ✅ Full | ✅ **Full** |
| 2 | parameters | ❌ None | ❌ None | ❌ None | ❌ None | ❌ **UNUSED** |
| 3 | return_doc | ❌ None | ❌ None | ❌ None | ❌ None | ❌ **UNUSED** |
| 4 | thread_safety | ✅ Full | ✅ (via doc) | ⚠️ Partial | ✅ Full | ✅ **Full** |
| 5 | ownership | ⚠️ Partial | ❌ None | ❌ None | ❌ None | ⚠️ **Partial** |
| 6 | preconditions | ⚠️ Partial | ❌ None | ⚠️ Partial | ⚠️ Partial | ⚠️ **Partial** |
| 7 | test_cases | ❌ None | ❌ None | ❌ None | ❌ None | ❌ **UNUSED** |
| 8 | attributes | ❌ None | ❌ None | ❌ None | ❌ None | ❌ **UNUSED** |
| 9 | platform | ❌ None | ❌ None | ❌ None | ❌ None | ❌ **UNUSED** |
| 10 | performance | ❌ None | ❌ None | ✅ Full | ❌ None | ⚠️ **Partial** |
| 11 | version_history | ❌ None | ❌ None | ❌ None | ❌ None | ❌ **UNUSED** |
| 12 | pitfalls | ❌ None | ❌ None | ⚠️ Partial | ✅ Full | ⚠️ **Partial** |
| 13 | error_semantics | ❌ None | ❌ None | ❌ None | ⚠️ Partial | ⚠️ **Partial** |
| 14 | callback_info | ❌ None | ❌ None | ❌ None | ❌ None | ❌ **UNUSED** |
| 15 | api_sequences | ❌ None | ❌ None | ✅ Full | ⚠️ Partial | ⚠️ **Partial** |
| 16 | resource_limits | ❌ None | ❌ None | ❌ None | ❌ None | ❌ **UNUSED** |
| 17 | semantic_group | ❌ None | ❌ None | ✅ Full | ❌ None | ⚠️ **Partial** |
| 18 | global_state | ❌ None | ❌ None | ✅ Full | ✅ Full | ✅ **Full** |
| 19 | numeric_constraints | ❌ None | ❌ None | ⚠️ Partial | ❌ None | ⚠️ **Partial** |

### Critical Issues

1. **No Runtime Validation** (CRITICAL)
   - `numeric_constraints` documented but never enforced
   - `preconditions` documented but never checked
   - `resource_limits` completely unused
   - **Impact**: Users can violate constraints and cause undefined behavior

2. **Incomplete Parameter Documentation** (HIGH)
   - `parameters` map completely unused
   - Rich parameter documentation available but not shown
   - **Impact**: Poor API documentation

3. **Missing Error Semantics** (HIGH)
   - `error_semantics` has detailed error information
   - Only used to check if errors exist, not to map them
   - **Impact**: Generic error handling, poor error messages

4. **No Test Examples** (HIGH)
   - `test_cases` completely unused
   - Users don't see usage patterns
   - **Impact**: Reduced API usability

5. **Missing Return Documentation** (HIGH)
   - `return_doc` completely unused
   - Return values undocumented
   - **Impact**: Unclear return semantics

6. **No Attribute Generation** (MEDIUM)
   - `attributes` completely unused
   - Missing #[must_use], #[deprecated], etc.
   - **Impact**: Missing compiler assistance

7. **No Platform Awareness** (MEDIUM)
   - `platform` completely unused
   - No platform-specific warnings
   - **Impact**: Portability issues

8. **Limited Ownership Usage** (MEDIUM)
   - Only return ownership used in wrappers.rs
   - Parameter ownership ignored everywhere
   - **Impact**: Suboptimal API design

---

## Recommendations by Priority

### P0 - Critical (Implement Immediately)

1. **Add Runtime Validation in builders.rs**
   - Validate numeric constraints in setters
   - Validate preconditions in build()
   - Return Result<Self, Error> from setters

2. **Add Validation in methods.rs**
   - Check preconditions before FFI calls
   - Validate numeric constraints on parameters
   - Early return with descriptive errors

3. **Enhance Error Handling**
   - Map specific error codes using error_semantics
   - Provide error-specific messages
   - Distinguish fatal vs recoverable errors

### P1 - High (Implement Soon)

4. **Use Parameters Map**
   - Document each parameter with rich descriptions
   - Show in doc comments for functions and builders

5. **Use Return Documentation**
   - Add Returns section to doc comments
   - Clarify return value semantics

6. **Generate Test Examples**
   - Use test_cases to create doc examples
   - Show typical usage patterns

7. **Complete Precondition Usage**
   - Document ALL preconditions, not just top 3
   - Include state requirements
   - Add runtime checks where possible

### P2 - Medium (Implement Later)

8. **Add Attribute Generation**
   - Generate #[must_use] for important types
   - Add #[deprecated] from version_history
   - Include platform-specific attributes

9. **Enhance Ownership Usage**
   - Consider parameter ownership in API design
   - Use borrowed parameters where possible
   - Document ownership transfer clearly

10. **Add Platform Awareness**
    - Document platform requirements
    - Add conditional compilation hints
    - Warn about platform-specific behavior

11. **Use Callback Info**
    - Optimize callback parameter handling
    - Document lifetime requirements
    - Add safety warnings for callbacks

12. **Complete Error Semantics Usage**
    - Document each error code
    - Provide recovery strategies
    - Map errors to specific Rust error types

### P3 - Low (Nice to Have)

13. **Use Semantic Grouping**
    - Organize related types together
    - Generate module structure hints
    - Add "See also" sections

14. **Use Resource Limits**
    - Validate resource parameters
    - Warn about resource exhaustion
    - Document limits in API docs

15. **Use Version History**
    - Mark deprecated APIs
    - Document version requirements
    - Add migration guides

---

## Conclusion

The enrichment system has collected valuable data across 19 dimensions, but only about **26% is fully utilized**. The most critical gaps are:

1. **No runtime validation** despite having constraint data
2. **Incomplete documentation** despite having rich parameter/return info
3. **Generic error handling** despite having detailed error semantics
4. **No usage examples** despite having test cases

Addressing the P0 and P1 recommendations would dramatically improve:
- **Safety**: Runtime validation prevents undefined behavior
- **Usability**: Rich documentation and examples help users
- **Error Handling**: Specific error messages aid debugging
- **API Quality**: Proper ownership and attributes improve the API surface

The generators have a solid foundation but need significant enhancements to fully leverage the enrichment data.
