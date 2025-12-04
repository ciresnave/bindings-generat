//! Quick test to verify functional test generation output

use bindings_generat::ffi::{FfiFunction, FfiParam};
use bindings_generat::generator::functional_tests;

fn main() {
    let functions = vec![
        FfiFunction {
            name: "cuda_malloc".to_string(),
            return_type: "int".to_string(),
            params: vec![
                FfiParam {
                    name: "size".to_string(),
                    ty: "size_t".to_string(),
                    is_pointer: false,
                    is_mut: false,
                },
            ],
            docs: None,
        },
        FfiFunction {
            name: "cuda_free".to_string(),
            return_type: "void".to_string(),
            params: vec![
                FfiParam {
                    name: "ptr".to_string(),
                    ty: "void*".to_string(),
                    is_pointer: true,
                    is_mut: false,
                },
            ],
            docs: None,
        },
    ];

    let output = functional_tests::generate_functional_tests(&functions, &[], "test_lib");
    
    println!("=== GENERATED FUNCTIONAL TESTS ===\n");
    println!("{}", output);
    println!("\n=== END OUTPUT ===");
}
