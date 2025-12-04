# c_header_analyzer

Standalone crate that analyzes C header files (via `bindgen`) and emits a JSON
representation of the FFI surface. The JSON schema follows the `FfiInfo` type
exported by the library.

## JSON Schema (informal)

- functions: array of objects { name: string, params: [ { name, ty, is_pointer, is_mut } ], return_type, docs }
- types: array of objects { name, is_opaque, docs, fields: [ { name, ty } ] }
- enums: array of objects { name, variants: [ { name, value?, docs? } ], docs }
- constants: array of objects { name, value, ty }
- opaque_types: array of strings
- dependencies: array of strings (library names inferred from function prefixes)
- type_aliases: mapping string -> string

## Usage

Build and run locally (requires libclang on the system for bindgen):

```powershell
cd crates/c_header_analyzer
cargo run -- --output ffi.json path\to\header.h
```

Or, print JSON to stdout:

```powershell
cargo run -- path\to\header.h
```

## Notes

This crate is intentionally small and focused. Later we will move/replace the
existing header analysis code in `bindings-generat` to use this crate as a
path dependency so the generator can concentrate on unsafe binding emission
and safe wrapper generation.
