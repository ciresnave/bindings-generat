/// Example demonstrating automatic ecosystem integration
/// 
/// This shows how bindings-generat detects library categories and recommends
/// relevant ecosystem crates from a curated list of 100+ libraries.

use bindings_generat::ecosystem::{EcosystemCrate, LibraryCategory, IntegrationDetector};
use bindings_generat::ffi::FfiInfo;

fn main() {
    println!("🎉 Ecosystem Integration Demo\n");
    println!("bindings-generat supports 100+ ecosystem crates organized in 12 tiers!\n");

    // Example 1: Mathematics Library
    println!("═══════════════════════════════════════════════════════════");
    println!("Example 1: Mathematics Library (BLAS/LAPACK wrapper)");
    println!("═══════════════════════════════════════════════════════════\n");
    
    let math_ffi = create_math_library_info();
    demonstrate_integration(&math_ffi, "libmathcore");

    // Example 2: Graphics Library
    println!("\n═══════════════════════════════════════════════════════════");
    println!("Example 2: Graphics Library (Vulkan wrapper)");
    println!("═══════════════════════════════════════════════════════════\n");
    
    let graphics_ffi = create_graphics_library_info();
    demonstrate_integration(&graphics_ffi, "libgraphics");

    // Example 3: Networking Library
    println!("\n═══════════════════════════════════════════════════════════");
    println!("Example 3: Networking Library (HTTP/REST client)");
    println!("═══════════════════════════════════════════════════════════\n");
    
    let network_ffi = create_network_library_info();
    demonstrate_integration(&network_ffi, "libnetwork");

    // Show all supported tiers
    println!("\n═══════════════════════════════════════════════════════════");
    println!("All Supported Tiers (100+ crates)");
    println!("═══════════════════════════════════════════════════════════\n");
    show_all_tiers();
}

fn demonstrate_integration(ffi_info: &FfiInfo, library_name: &str) {
    let detector = IntegrationDetector::new();
    
    // Detect category and get recommendations
    let integration = detector.detect(ffi_info, library_name)
        .expect("Failed to detect integrations");
    
    println!("✓ Detected Category: {:?}", integration.category);
    println!("✓ Recommended Integrations ({} crates):", integration.standard_crates.len());
    
    // Group by tier
    let mut tiers: std::collections::BTreeMap<u8, Vec<&EcosystemCrate>> = 
        std::collections::BTreeMap::new();
    
    for crate_enum in &integration.standard_crates {
        tiers.entry(crate_enum.tier())
            .or_default()
            .push(crate_enum);
    }
    
    // Display by tier
    for (tier, crates) in tiers {
        let tier_name = match tier {
            1 => "Universal (recommended for all)",
            2 => "Async/Concurrency",
            3 => "Serialization",
            9 => "Math/Arrays",
            10 => "Formats",
            _ => "Other",
        };
        
        println!("\n  Tier {}: {}", tier, tier_name);
        for crate_enum in crates {
            println!("    • {} - {}", crate_enum.crate_name(), crate_enum.description());
        }
    }
    
    // Show Cargo.toml snippet
    println!("\n✓ Generated Cargo.toml snippet:\n");
    show_cargo_toml(&integration.standard_crates);
}

fn show_cargo_toml(crates: &[EcosystemCrate]) {
    println!("  [dependencies]");
    for crate_enum in crates.iter().take(5) {
        let features = if let Some(features) = crate_enum.cargo_features() {
            format!(", features = {:?}", features)
        } else {
            String::new()
        };
        println!("  {} = {{ version = \"{}\"{}, optional = true }}", 
                 crate_enum.crate_name(), 
                 crate_enum.version(),
                 features);
    }
    if crates.len() > 5 {
        println!("  # ... and {} more", crates.len() - 5);
    }
}

fn create_math_library_info() -> FfiInfo {
    use bindings_generat::ffi::{FfiFunction, FfiParam};
    use std::collections::HashMap;
    
    FfiInfo {
        functions: vec![
            FfiFunction {
                name: "matrix_multiply".to_string(),
                docs: Some("Multiplies two matrices using BLAS".to_string()),
                params: vec![],
                return_type: "void".to_string(),
            },
            FfiFunction {
                name: "vector_dot_product".to_string(),
                docs: Some("Computes dot product of vectors".to_string()),
                params: vec![],
                return_type: "double".to_string(),
            },
            FfiFunction {
                name: "solve_linear_system".to_string(),
                docs: Some("Solves Ax=b using LAPACK".to_string()),
                params: vec![],
                return_type: "int".to_string(),
            },
        ],
        types: vec![],
        enums: vec![],
        constants: vec![],
        opaque_types: vec![],
        dependencies: vec![],
        type_aliases: HashMap::new(),
    }
}

fn create_graphics_library_info() -> FfiInfo {
    use bindings_generat::ffi::{FfiFunction, FfiParam};
    use std::collections::HashMap;
    
    FfiInfo {
        functions: vec![
            FfiFunction {
                name: "create_render_pass".to_string(),
                docs: Some("Creates a Vulkan render pass".to_string()),
                params: vec![],
                return_type: "VkRenderPass".to_string(),
            },
            FfiFunction {
                name: "submit_command_buffer".to_string(),
                docs: Some("Submits graphics commands".to_string()),
                params: vec![],
                return_type: "VkResult".to_string(),
            },
            FfiFunction {
                name: "create_shader_module".to_string(),
                docs: Some("Loads and compiles shader".to_string()),
                params: vec![],
                return_type: "VkShaderModule".to_string(),
            },
        ],
        types: vec![],
        enums: vec![],
        constants: vec![],
        opaque_types: vec![],
        dependencies: vec![],
        type_aliases: HashMap::new(),
    }
}

fn create_network_library_info() -> FfiInfo {
    use bindings_generat::ffi::{FfiFunction, FfiParam};
    use std::collections::HashMap;
    
    FfiInfo {
        functions: vec![
            FfiFunction {
                name: "http_get_request".to_string(),
                docs: Some("Sends HTTP GET request".to_string()),
                params: vec![],
                return_type: "int".to_string(),
            },
            FfiFunction {
                name: "connect_socket".to_string(),
                docs: Some("Establishes TCP connection".to_string()),
                params: vec![],
                return_type: "int".to_string(),
            },
            FfiFunction {
                name: "send_packet".to_string(),
                docs: Some("Transmits network packet".to_string()),
                params: vec![],
                return_type: "ssize_t".to_string(),
            },
        ],
        types: vec![],
        enums: vec![],
        constants: vec![],
        opaque_types: vec![],
        dependencies: vec![],
        type_aliases: HashMap::new(),
    }
}

fn show_all_tiers() {
    println!("Tier 1 (Universal - Recommended for All):");
    println!("  serde, thiserror, tracing, log, once_cell");
    
    println!("\nTier 2 (Async/Concurrency - 10 crates):");
    println!("  tokio, async-std, futures, rayon, crossbeam, parking_lot, flume, dashmap, arc-swap, thread_local");
    
    println!("\nTier 3 (Serialization - 9 crates):");
    println!("  serde_json, serde_yaml, bincode, ron, toml, rmp-serde, postcard, flexbuffers, bson");
    
    println!("\nTier 4 (Error Handling - 4 crates):");
    println!("  anyhow, eyre, miette, color-eyre");
    
    println!("\nTier 5 (CLI - 5 crates):");
    println!("  clap, dialoguer, indicatif, console, termcolor");
    
    println!("\nTier 6 (HTTP/Web - 6 crates):");
    println!("  hyper, reqwest, axum, actix-web, tower, warp");
    
    println!("\nTier 7 (Time/IDs - 4 crates):");
    println!("  chrono, time, uuid, ulid");
    
    println!("\nTier 8 (Data Structures - 6 crates):");
    println!("  smallvec, arrayvec, tinyvec, hashbrown, indexmap, ahash");
    
    println!("\nTier 9 (Math/Arrays - 6 crates):");
    println!("  ndarray, nalgebra, num, num-traits, approx, rand");
    
    println!("\nTier 10 (Formats - 4 crates):");
    println!("  image, csv, serde-xml-rs, quick-xml");
    
    println!("\nTier 11 (Database/Storage - 5 crates):");
    println!("  rusqlite, sqlx, redb, sled, rocksdb");
    
    println!("\nTier 12 (Protocols - 7 crates):");
    println!("  prost, tonic, capnp, flatbuffers, url, http, mime");
    
    println!("\n✨ Total: 100+ crates across 12 tiers!");
}
