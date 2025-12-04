include!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/discovery_shared.rs"));

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
}
