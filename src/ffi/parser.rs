use anyhow::{Context, Result};
use syn::{File, ForeignItem, ForeignItemFn, Item, ItemEnum, ItemStruct, ItemType};
use tracing::{debug, info};

/// Parsed FFI information
#[derive(Debug, Clone)]
pub struct FfiInfo {
    pub functions: Vec<FfiFunction>,
    pub types: Vec<FfiType>,
    pub enums: Vec<FfiEnum>,
    pub constants: Vec<FfiConstant>,
    pub opaque_types: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct FfiFunction {
    pub name: String,
    pub params: Vec<FfiParam>,
    pub return_type: String,
    pub docs: Option<String>,
}

#[derive(Debug, Clone)]
pub struct FfiParam {
    pub name: String,
    pub ty: String,
    pub is_pointer: bool,
    pub is_mut: bool,
}

#[derive(Debug, Clone)]
pub struct FfiType {
    pub name: String,
    pub is_opaque: bool,
    pub docs: Option<String>,
    pub fields: Vec<FfiField>,
}

#[derive(Debug, Clone)]
pub struct FfiField {
    pub name: String,
    pub ty: String,
}

#[derive(Debug, Clone)]
pub struct FfiEnum {
    pub name: String,
    pub variants: Vec<FfiEnumVariant>,
    pub docs: Option<String>,
}

#[derive(Debug, Clone)]
pub struct FfiEnumVariant {
    pub name: String,
    pub value: Option<i64>,
}

#[derive(Debug, Clone)]
pub struct FfiConstant {
    pub name: String,
    pub value: String,
    pub ty: String,
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
        }
    }
}

/// Parse the generated FFI bindings to extract information
pub fn parse_ffi_bindings(bindings_code: &str) -> Result<FfiInfo> {
    info!("Parsing FFI bindings");

    let syntax_tree: File =
        syn::parse_str(bindings_code).context("Failed to parse generated bindings as Rust code")?;

    let mut ffi_info = FfiInfo::new();

    for item in syntax_tree.items {
        match item {
            Item::ForeignMod(foreign_mod) => {
                debug!(
                    "Found foreign module with {} items",
                    foreign_mod.items.len()
                );
                for foreign_item in foreign_mod.items {
                    if let ForeignItem::Fn(func) = foreign_item
                        && let Some(ffi_func) = parse_foreign_function(func)
                    {
                        ffi_info.functions.push(ffi_func);
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
                // Check if this is an opaque type
                if is_opaque_type(&item_type) {
                    ffi_info.opaque_types.push(item_type.ident.to_string());
                }
            }
            Item::Const(item_const) => {
                ffi_info.constants.push(FfiConstant {
                    name: item_const.ident.to_string(),
                    value: quote::quote!(#item_const).to_string(),
                    ty: quote::quote!(#item_const.ty).to_string(),
                });
            }
            _ => {}
        }
    }

    info!(
        "Parsed {} functions, {} types, {} enums, {} constants",
        ffi_info.functions.len(),
        ffi_info.types.len(),
        ffi_info.enums.len(),
        ffi_info.constants.len()
    );

    Ok(ffi_info)
}

fn parse_foreign_function(func: ForeignItemFn) -> Option<FfiFunction> {
    let name = func.sig.ident.to_string();

    let params: Vec<FfiParam> = func
        .sig
        .inputs
        .iter()
        .filter_map(|arg| {
            if let syn::FnArg::Typed(pat_type) = arg
                && let syn::Pat::Ident(pat_ident) = &*pat_type.pat
            {
                let param_name = pat_ident.ident.to_string();
                let type_str = quote::quote!(#pat_type.ty).to_string();

                let (is_pointer, is_mut) = analyze_pointer_type(&pat_type.ty);

                return Some(FfiParam {
                    name: param_name,
                    ty: type_str,
                    is_pointer,
                    is_mut,
                });
            }
            None
        })
        .collect();

    let return_type = match &func.sig.output {
        syn::ReturnType::Default => "()".to_string(),
        syn::ReturnType::Type(_, ty) => quote::quote!(#ty).to_string(),
    };

    let docs = extract_docs(&func.attrs);

    Some(FfiFunction {
        name,
        params,
        return_type,
        docs,
    })
}

fn parse_struct(item_struct: ItemStruct) -> Option<FfiType> {
    let name = item_struct.ident.to_string();
    let docs = extract_docs(&item_struct.attrs);

    let fields: Vec<FfiField> = match item_struct.fields {
        syn::Fields::Named(fields_named) => fields_named
            .named
            .iter()
            .map(|field| FfiField {
                name: field.ident.as_ref().unwrap().to_string(),
                ty: quote::quote!(#field.ty).to_string(),
            })
            .collect(),
        syn::Fields::Unnamed(_) => Vec::new(),
        syn::Fields::Unit => Vec::new(),
    };

    let is_opaque = fields.is_empty();

    Some(FfiType {
        name,
        is_opaque,
        docs,
        fields,
    })
}

fn parse_enum(item_enum: ItemEnum) -> Option<FfiEnum> {
    let name = item_enum.ident.to_string();
    let docs = extract_docs(&item_enum.attrs);

    let variants: Vec<FfiEnumVariant> = item_enum
        .variants
        .iter()
        .map(|variant| {
            let variant_name = variant.ident.to_string();
            let value = match &variant.discriminant {
                Some((_, expr)) => {
                    // Try to extract integer value
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
                name: variant_name,
                value,
            }
        })
        .collect();

    Some(FfiEnum {
        name,
        variants,
        docs,
    })
}

fn is_opaque_type(item_type: &ItemType) -> bool {
    // Check if this is a type alias to an opaque struct
    if let syn::Type::Path(type_path) = &*item_type.ty
        && let Some(segment) = type_path.path.segments.first()
    {
        // Common opaque type patterns
        let ident = segment.ident.to_string();
        return ident == "c_void" || ident.ends_with("_impl") || ident.ends_with("_internal");
    }
    false
}

fn analyze_pointer_type(ty: &syn::Type) -> (bool, bool) {
    match ty {
        syn::Type::Ptr(type_ptr) => {
            let is_mut = type_ptr.mutability.is_some();
            (true, is_mut)
        }
        _ => (false, false),
    }
}

fn extract_docs(attrs: &[syn::Attribute]) -> Option<String> {
    let mut docs = Vec::new();

    for attr in attrs {
        if attr.path().is_ident("doc")
            && let syn::Meta::NameValue(meta_name_value) = &attr.meta
            && let syn::Expr::Lit(expr_lit) = &meta_name_value.value
            && let syn::Lit::Str(lit_str) = &expr_lit.lit
        {
            docs.push(lit_str.value().trim().to_string());
        }
    }

    if docs.is_empty() {
        None
    } else {
        Some(docs.join("\n"))
    }
}

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
