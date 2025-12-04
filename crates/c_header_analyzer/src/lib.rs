//! C header analysis crate
//!
//! This crate provides a small library and CLI to analyze C header files
//! (via `bindgen`) and emit a JSON representation of the discovered FFI
//! surface. The output schema is intentionally simple and matches the
//! `FfiInfo` structure used by the generator, so it can be consumed by
//! downstream tools (unsafe bindings generation and safe wrapper
//! generation).

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

/// A small, structured representation of a parsed type expression.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArraySpec {
    /// The exact source string parsed for the array type.
    pub parsed_from: String,
    /// Optional length for fixed-size C arrays (None for slices).
    pub len: Option<usize>,
    /// Element type spec.
    pub elem: Box<TypeSpec>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionPointerSpec {
    /// The exact source string parsed for the function pointer.
    pub parsed_from: String,
    /// Parameter types.
    pub params: Vec<TypeSpec>,
    /// Return type.
    pub ret: Box<TypeSpec>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypeSpec {
    /// The exact source string this TypeSpec was parsed from.
    pub parsed_from: String,
    /// The base identifier (e.g. `cudnnContext` for `*mut cudnnContext`).
    /// `None` for non-path-first types (e.g. bare fn pointers).
    pub base: Option<String>,
    /// Full path for path types (e.g. `std::os::raw::c_char`).
    /// `None` when not applicable.
    pub full_path: Option<String>,
    /// Number of pointer indirections.
    pub pointer_depth: usize,
    /// Mutability flags for each pointer level (len == pointer_depth).
    pub pointer_mut: Vec<bool>,
    /// Qualifiers observed on the type (e.g. `const`).
    pub qualifiers: Vec<String>,
    /// Optional array information.
    pub array: Option<ArraySpec>,
    /// Optional function-pointer information.
    pub fn_ptr: Option<FunctionPointerSpec>,
}

impl TypeSpec {
    /// Render a canonical Rust-like string for this TypeSpec.
    ///
    /// If `prefer_full_path` is true and `full_path` is available, the
    /// renderer will use the full path (e.g. `std::os::raw::c_char`) instead
    /// of the short `base` identifier. Pointer mutability is rendered from
    /// outermost -> innermost (e.g. `*mut *const T`).
    pub fn to_rust_string(&self, prefer_full_path: bool) -> String {
        // helper that renders the core (non-pointer) representation
        fn core_repr(ts: &TypeSpec, prefer_full: bool) -> String {
            if let Some(arr) = &ts.array {
                let elem = arr.elem.to_rust_string(prefer_full);
                if let Some(len) = arr.len {
                    return format!("[{}; {}]", elem, len);
                } else {
                    return format!("[{}]", elem);
                }
            }

            if let Some(fp) = &ts.fn_ptr {
                let params = fp
                    .params
                    .iter()
                    .map(|p| p.to_rust_string(prefer_full))
                    .collect::<Vec<_>>()
                    .join(", ");
                let ret = fp.ret.to_rust_string(prefer_full);
                return format!("fn({}) -> {}", params, ret);
            }

            if prefer_full {
                if let Some(full) = &ts.full_path {
                    return full.clone();
                }
                if let Some(base) = &ts.base {
                    return base.clone();
                }
            } else {
                if let Some(base) = &ts.base {
                    return base.clone();
                }
                if let Some(full) = &ts.full_path {
                    return full.clone();
                }
            }

            ts.parsed_from.clone()
        }

        let mut out = String::new();
        // pointer_mut is ordered outermost -> innermost
        for &is_mut in &self.pointer_mut {
            if is_mut {
                out.push_str("*mut ");
            } else {
                out.push_str("*const ");
            }
        }

        out.push_str(&core_repr(self, prefer_full_path));
        out
    }
}

impl fmt::Display for TypeSpec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_rust_string(true))
    }
}

/// Parsed FFI information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FfiInfo {
    pub functions: Vec<FfiFunction>,
    pub types: Vec<FfiType>,
    pub enums: Vec<FfiEnum>,
    pub constants: Vec<FfiConstant>,
    pub opaque_types: Vec<String>,
    pub dependencies: Vec<String>,
    /// Type aliases (e.g., "cudnnHandle_t" -> TypeSpec for `*mut cudnnContext`)
    pub type_aliases: HashMap<String, TypeSpec>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FfiFunction {
    /// The exact source string parsed for this function (for debugging).
    pub parsed_from: String,
    pub name: String,
    pub params: Vec<FfiParam>,
    /// Base return type (e.g. `cudnnStatus_t`), without pointer decorators.
    pub return_type: String,
    /// Structured return type information (pointers, qualifiers, arrays, fn ptrs).
    pub return_spec: Option<TypeSpec>,
    pub docs: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FfiParam {
    /// Exact source string parsed for this parameter.
    pub parsed_from: String,
    pub name: String,
    /// Base type (e.g. `::std::os::raw::c_char`) with normalized path if available.
    pub ty: String,
    /// Number of pointer indirections (0 for non-pointer types).
    pub pointer_depth: usize,
    /// Mutability flags per pointer level (len == pointer_depth).
    pub pointer_mut: Vec<bool>,
    /// Full parsed TypeSpec for this parameter.
    pub ty_spec: TypeSpec,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FfiType {
    /// Exact source string parsed for this type (struct decl or alias).
    pub parsed_from: String,
    pub name: String,
    pub is_opaque: bool,
    pub docs: Option<String>,
    pub fields: Vec<FfiField>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FfiField {
    /// Exact source string parsed for this field.
    pub parsed_from: String,
    pub name: String,
    pub ty: String,
    /// Structured type information for this field.
    pub ty_spec: TypeSpec,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FfiEnum {
    /// Exact source string parsed for this enum.
    pub parsed_from: String,
    pub name: String,
    pub variants: Vec<FfiEnumVariant>,
    pub docs: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FfiEnumVariant {
    /// Exact source string parsed for this variant.
    pub parsed_from: String,
    pub name: String,
    pub value: Option<i64>,
    pub docs: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FfiConstant {
    /// Exact source string parsed for this constant.
    pub parsed_from: String,
    pub name: String,
    /// The literal/text value of the constant (expression as string).
    pub value: String,
    /// Base type of the constant (e.g. `u32`).
    pub ty: String,
    /// Structured type information for the constant's type.
    pub ty_spec: TypeSpec,
}

impl Default for FfiInfo {
    fn default() -> Self {
        Self::new()
    }
}

impl FfiInfo {
    pub fn new() -> Self {
        Self {
            functions: Vec::new(),
            types: Vec::new(),
            enums: Vec::new(),
            constants: Vec::new(),
            opaque_types: Vec::new(),
            dependencies: Vec::new(),
            type_aliases: HashMap::new(),
        }
    }
}

// --- Parsing of bindgen-generated Rust bindings ---

use quote::quote;
use syn::{File, ForeignItem, ForeignItemFn, Item, ItemEnum, ItemStruct, ItemType};

/// Parse the generated FFI bindings to extract information
pub fn parse_ffi_bindings(bindings_code: &str) -> Result<FfiInfo> {
    // Parse the generated Rust code using `syn`.
    let syntax_tree: File =
        syn::parse_str(bindings_code).context("Failed to parse generated bindings as Rust code")?;

    let mut ffi_info = FfiInfo::new();

    for item in syntax_tree.items {
        match item {
            Item::ForeignMod(foreign_mod) => {
                for foreign_item in foreign_mod.items {
                    if let ForeignItem::Fn(func) = foreign_item {
                        if let Some(ffi_func) = parse_foreign_function(func) {
                            ffi_info.functions.push(ffi_func);
                        }
                    }
                }
            }
            Item::Struct(item_struct) => {
                if let Some(ffi_type) = parse_struct(item_struct) {
                    ffi_info.types.push(ffi_type);
                }
            }
            Item::Enum(item_enum) => {
                if let Some(ffi_enum) = parse_enum(item_enum) {
                    ffi_info.enums.push(ffi_enum);
                }
            }
            Item::Type(item_type) => {
                // Store the type alias mapping
                let alias_name = item_type.ident.to_string();
                let ty = &*item_type.ty;
                let spec = parse_type_spec(ty);
                ffi_info.type_aliases.insert(alias_name.clone(), spec);

                // Also check if this is an opaque type
                if is_opaque_type(&item_type) {
                    ffi_info.opaque_types.push(alias_name);
                }
            }
            Item::Const(item_const) => {
                let parsed = quote!(#item_const).to_string();
                // Try to extract a clean literal value (e.g. 13000) when possible
                let expr = &*item_const.expr;
                let value_str = match expr {
                    syn::Expr::Lit(expr_lit) => match &expr_lit.lit {
                        syn::Lit::Int(lit_int) => lit_int.base10_digits().to_string(),
                        syn::Lit::Float(lit_float) => lit_float.base10_digits().to_string(),
                        syn::Lit::Str(lit_str) => lit_str.value(),
                        other => quote!(#other).to_string(),
                    },
                    syn::Expr::Unary(unary) => {
                        // handle negative integer literals like `-1`
                        if let syn::UnOp::Neg(_) = unary.op {
                            if let syn::Expr::Lit(inner_lit) = &*unary.expr {
                                if let syn::Lit::Int(lit_int) = &inner_lit.lit {
                                    format!("-{}", lit_int.base10_digits())
                                } else {
                                    quote!(#expr).to_string()
                                }
                            } else {
                                quote!(#expr).to_string()
                            }
                        } else {
                            quote!(#expr).to_string()
                        }
                    }
                    _ => quote!(#expr).to_string(),
                };

                let ty_spec = parse_type_spec(&*item_const.ty);
                let ty_base = ty_spec
                    .base
                    .clone()
                    .unwrap_or_else(|| ty_spec.parsed_from.clone());

                ffi_info.constants.push(FfiConstant {
                    parsed_from: parsed.clone(),
                    name: item_const.ident.to_string(),
                    value: value_str,
                    ty: ty_base,
                    ty_spec,
                });
            }
            _ => {}
        }
    }

    // Detect external library dependencies by analyzing function name prefixes
    ffi_info.dependencies = detect_dependencies(&ffi_info.functions);

    Ok(ffi_info)
}

/// Detect external library dependencies by analyzing function name prefixes
fn detect_dependencies(functions: &[FfiFunction]) -> Vec<String> {
    use std::collections::HashMap;

    let known_prefixes: HashMap<&str, &str> = [
        ("cuda", "cuda"),
        ("cublas", "cublas"),
        ("cufft", "cufft"),
        ("curand", "curand"),
        ("cusparse", "cusparse"),
        ("cusolver", "cusolver"),
        ("nvjpeg", "nvjpeg"),
        ("npp", "npp"),
    ]
    .iter()
    .copied()
    .collect();

    let mut prefix_counts: HashMap<String, usize> = HashMap::new();

    for func in functions {
        let func_name = func.name.to_lowercase();
        for prefix in known_prefixes.keys() {
            if func_name.starts_with(prefix) {
                *prefix_counts.entry(prefix.to_string()).or_insert(0) += 1;
                break;
            }
        }
    }

    let mut dependencies: Vec<String> = prefix_counts
        .iter()
        .filter(|(_, count)| **count >= 5)
        .map(|(prefix, _)| known_prefixes.get(prefix.as_str()).unwrap().to_string())
        .collect();

    dependencies.sort();
    dependencies.dedup();
    dependencies
}

fn parse_foreign_function(func: ForeignItemFn) -> Option<FfiFunction> {
    let parsed = quote!(#func).to_string();
    let name = func.sig.ident.to_string();

    let params: Vec<FfiParam> = func
        .sig
        .inputs
        .iter()
        .filter_map(|arg| {
            if let syn::FnArg::Typed(pat_type) = arg {
                if let syn::Pat::Ident(pat_ident) = &*pat_type.pat {
                    let param_name = pat_ident.ident.to_string();
                    let ty = &pat_type.ty;
                    let param_parsed = quote!(#pat_type).to_string();

                    let ty_spec = parse_type_spec(&*ty);
                    let ty_base = ty_spec
                        .base
                        .clone()
                        .unwrap_or_else(|| ty_spec.parsed_from.clone());

                    return Some(FfiParam {
                        parsed_from: param_parsed,
                        name: param_name,
                        ty: ty_base,
                        pointer_depth: ty_spec.pointer_depth,
                        pointer_mut: ty_spec.pointer_mut.clone(),
                        ty_spec,
                    });
                }
            }
            None
        })
        .collect();

    let (return_type, return_spec) = match &func.sig.output {
        syn::ReturnType::Default => ("()".to_string(), None),
        syn::ReturnType::Type(_, ty) => {
            let spec = parse_type_spec(&*ty);
            let base = spec
                .base
                .clone()
                .unwrap_or_else(|| spec.parsed_from.clone());
            (base, Some(spec))
        }
    };

    let docs = extract_docs(&func.attrs);

    Some(FfiFunction {
        parsed_from: parsed,
        name,
        params,
        return_type,
        return_spec,
        docs,
    })
}

fn parse_struct(item_struct: ItemStruct) -> Option<FfiType> {
    let name = item_struct.ident.to_string();
    let docs = extract_docs(&item_struct.attrs);
    let parsed = quote!(#item_struct).to_string();

    let fields: Vec<FfiField> = match item_struct.fields {
        syn::Fields::Named(fields_named) => fields_named
            .named
            .iter()
            .map(|field| {
                let parsed_from = quote!(#field).to_string();
                let name = field.ident.as_ref().unwrap().to_string();
                let ty_spec = parse_type_spec(&field.ty);
                let ty = ty_spec
                    .base
                    .clone()
                    .unwrap_or_else(|| ty_spec.parsed_from.clone());

                FfiField {
                    parsed_from,
                    name,
                    ty,
                    ty_spec,
                }
            })
            .collect(),
        syn::Fields::Unnamed(_) => Vec::new(),
        syn::Fields::Unit => Vec::new(),
    };

    let is_opaque = fields.is_empty();

    Some(FfiType {
        parsed_from: parsed,
        name,
        is_opaque,
        docs,
        fields,
    })
}

fn parse_enum(item_enum: ItemEnum) -> Option<FfiEnum> {
    let name = item_enum.ident.to_string();
    let docs = extract_docs(&item_enum.attrs);
    let parsed = quote!(#item_enum).to_string();

    let variants: Vec<FfiEnumVariant> = item_enum
        .variants
        .iter()
        .map(|variant| {
            let variant_name = variant.ident.to_string();
            let variant_docs = extract_docs(&variant.attrs);
            let value = match &variant.discriminant {
                Some((_, expr)) => {
                    if let syn::Expr::Lit(expr_lit) = expr {
                        if let syn::Lit::Int(lit_int) = &expr_lit.lit {
                            lit_int.base10_parse::<i64>().ok()
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                }
                None => None,
            };

            FfiEnumVariant {
                parsed_from: quote!(#variant).to_string(),
                name: variant_name,
                value,
                docs: variant_docs,
            }
        })
        .collect();

    Some(FfiEnum {
        parsed_from: parsed,
        name,
        variants,
        docs,
    })
}

fn is_opaque_type(item_type: &ItemType) -> bool {
    if let syn::Type::Path(type_path) = &*item_type.ty {
        if let Some(segment) = type_path.path.segments.first() {
            let ident = segment.ident.to_string();
            return ident == "c_void" || ident.ends_with("_impl") || ident.ends_with("_internal");
        }
    }
    false
}

/// Parse a syn::Type into a small, serializable TypeSpec.
fn parse_type_spec(ty: &syn::Type) -> TypeSpec {
    // Recursively parse into a TypeSpec.
    fn rec(ty: &syn::Type) -> TypeSpec {
        let parsed = quote!(#ty).to_string();

        match ty {
            syn::Type::Array(type_array) => {
                let elem_spec = rec(&*type_array.elem);
                // try to extract length if literal integer
                let len = match &type_array.len {
                    syn::Expr::Lit(expr_lit) => {
                        if let syn::Lit::Int(lit_int) = &expr_lit.lit {
                            lit_int.base10_parse::<usize>().ok()
                        } else {
                            None
                        }
                    }
                    _ => None,
                };

                TypeSpec {
                    parsed_from: parsed.clone(),
                    base: elem_spec.base.clone(),
                    full_path: elem_spec.full_path.clone(),
                    pointer_depth: elem_spec.pointer_depth,
                    pointer_mut: elem_spec.pointer_mut.clone(),
                    qualifiers: elem_spec.qualifiers.clone(),
                    array: Some(ArraySpec {
                        parsed_from: parsed.clone(),
                        len,
                        elem: Box::new(elem_spec),
                    }),
                    fn_ptr: None,
                }
            }
            syn::Type::Slice(type_slice) => {
                let elem_spec = rec(&*type_slice.elem);
                TypeSpec {
                    parsed_from: parsed.clone(),
                    base: elem_spec.base.clone(),
                    full_path: elem_spec.full_path.clone(),
                    pointer_depth: elem_spec.pointer_depth,
                    pointer_mut: elem_spec.pointer_mut.clone(),
                    qualifiers: elem_spec.qualifiers.clone(),
                    array: Some(ArraySpec {
                        parsed_from: parsed.clone(),
                        len: None,
                        elem: Box::new(elem_spec),
                    }),
                    fn_ptr: None,
                }
            }
            syn::Type::BareFn(barefn) => {
                // function pointer
                let params = barefn
                    .inputs
                    .iter()
                    .map(|arg| match &arg.ty {
                        ty => rec(ty),
                    })
                    .collect::<Vec<_>>();

                let ret_spec = match &barefn.output {
                    syn::ReturnType::Default => TypeSpec {
                        parsed_from: "()".to_string(),
                        base: Some("()".to_string()),
                        full_path: Some("()".to_string()),
                        pointer_depth: 0,
                        pointer_mut: Vec::new(),
                        qualifiers: Vec::new(),
                        array: None,
                        fn_ptr: None,
                    },
                    syn::ReturnType::Type(_, ty) => rec(&*ty),
                };

                TypeSpec {
                    parsed_from: parsed.clone(),
                    base: None,
                    full_path: None,
                    pointer_depth: 0,
                    pointer_mut: Vec::new(),
                    qualifiers: Vec::new(),
                    array: None,
                    fn_ptr: Some(FunctionPointerSpec {
                        parsed_from: parsed.clone(),
                        params,
                        ret: Box::new(ret_spec),
                    }),
                }
            }
            syn::Type::Ptr(type_ptr) => {
                let mut inner = rec(&*type_ptr.elem);
                // this pointer adds one level at the front
                inner.pointer_depth += 1;
                inner.pointer_mut.insert(0, type_ptr.mutability.is_some());
                inner.parsed_from = parsed.clone();
                inner
            }
            syn::Type::Reference(type_ref) => {
                let mut inner = rec(&*type_ref.elem);
                inner.pointer_depth += 1;
                inner.pointer_mut.insert(0, type_ref.mutability.is_some());
                inner.parsed_from = parsed.clone();
                inner
            }
            syn::Type::Path(type_path) => {
                let base = type_path
                    .path
                    .segments
                    .last()
                    .map(|seg| seg.ident.to_string());

                // Build a normalized full path like "std::os::raw::c_char"
                let mut parts: Vec<String> = Vec::new();
                for seg in type_path.path.segments.iter() {
                    parts.push(seg.ident.to_string());
                }
                let full = if parts.is_empty() {
                    None
                } else {
                    Some(parts.join("::"))
                };

                // crude qualifier detection (e.g. "const") using textual search
                let mut qualifiers = Vec::new();
                if parsed.contains("const ") {
                    qualifiers.push("const".to_string());
                }
                if parsed.contains("volatile") {
                    qualifiers.push("volatile".to_string());
                }

                TypeSpec {
                    parsed_from: parsed.clone(),
                    base,
                    full_path: full,
                    pointer_depth: 0,
                    pointer_mut: Vec::new(),
                    qualifiers,
                    array: None,
                    fn_ptr: None,
                }
            }
            syn::Type::Paren(tp) => rec(&*tp.elem),
            syn::Type::Group(tp) => rec(&*tp.elem),
            other => TypeSpec {
                parsed_from: parsed.clone(),
                base: Some(quote!(#other).to_string()),
                full_path: Some(quote!(#other).to_string()),
                pointer_depth: 0,
                pointer_mut: Vec::new(),
                qualifiers: Vec::new(),
                array: None,
                fn_ptr: None,
            },
        }
    }

    rec(ty)
}

fn extract_docs(attrs: &[syn::Attribute]) -> Option<String> {
    let mut docs = Vec::new();

    for attr in attrs {
        if attr.path().is_ident("doc") {
            if let syn::Meta::NameValue(meta_name_value) = &attr.meta {
                if let syn::Expr::Lit(expr_lit) = &meta_name_value.value {
                    if let syn::Lit::Str(lit_str) = &expr_lit.lit {
                        docs.push(lit_str.value().trim().to_string());
                    }
                }
            }
        }
    }

    if docs.is_empty() {
        None
    } else {
        Some(docs.join("\n"))
    }
}

// --- High level API: generate bindings via bindgen then parse to FfiInfo ---

/// Run `bindgen` on the provided header files and return the parsed `FfiInfo`.
pub fn analyze_headers<P: AsRef<std::path::Path>>(
    headers: &[P],
    clang_args: &[String],
) -> Result<FfiInfo> {
    if headers.is_empty() {
        return Ok(FfiInfo::new());
    }

    let mut builder = bindgen::Builder::default()
        .generate_comments(true)
        .derive_default(true);

    for header in headers {
        let path = header.as_ref().to_string_lossy().to_string();
        builder = builder.header(path);
    }

    for arg in clang_args {
        builder = builder.clang_arg(arg.clone());
    }

    let bindings = builder
        .generate()
        .context("bindgen failed to generate bindings (is libclang available?)")?;

    let code = bindings.to_string();
    let info = parse_ffi_bindings(&code)?;
    Ok(info)
}

// note: the single `analyze_headers` API now accepts explicit clang args.

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_function() {
        let code = r#"
            extern "C" {
                pub fn foo(x: i32) -> i32;
            }
        "#;

        let result = parse_ffi_bindings(code);
        assert!(result.is_ok());
        let ffi_info = result.unwrap();
        assert_eq!(ffi_info.functions.len(), 1);
        assert_eq!(ffi_info.functions[0].name, "foo");
    }
}
