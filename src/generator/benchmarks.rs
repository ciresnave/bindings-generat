//! Performance benchmark generation using criterion
//!
//! This module generates criterion benchmark suites to measure
//! wrapper overhead compared to raw FFI calls.

use crate::ffi::{FfiFunction, FfiInfo};
use std::fmt::Write;

/// Generate a benchmark suite for the generated wrappers
pub fn generate_benchmarks(ffi_info: &FfiInfo, crate_name: &str) -> String {
    let mut output = String::new();

    // File header
    writeln!(
        &mut output,
        "//! Performance benchmarks comparing wrappers to raw FFI"
    )
    .unwrap();
    writeln!(&mut output, "//!").unwrap();
    writeln!(&mut output, "//! Run with: cargo bench --bench performance").unwrap();
    writeln!(&mut output).unwrap();
    writeln!(
        &mut output,
        "use criterion::{{black_box, criterion_group, criterion_main, Criterion}};"
    )
    .unwrap();
    writeln!(&mut output, "use {}::*;", crate_name).unwrap();
    writeln!(&mut output).unwrap();

    // Generate benchmarks for each function
    let benchmark_funcs: Vec<_> = ffi_info
        .functions
        .iter()
        .filter(|f| is_benchmarkable(f))
        .take(10) // Limit to 10 benchmarks
        .collect();

    for func in &benchmark_funcs {
        generate_function_benchmark(&mut output, func);
    }

    // Generate group
    writeln!(&mut output, "fn benchmarks(c: &mut Criterion) {{").unwrap();
    for func in &benchmark_funcs {
        writeln!(
            &mut output,
            "    c.bench_function(\"{}\", |b| b.iter(|| bench_{}()));",
            func.name,
            sanitize_name(&func.name)
        )
        .unwrap();
    }
    writeln!(&mut output, "}}").unwrap();
    writeln!(&mut output).unwrap();

    writeln!(&mut output, "criterion_group!(benches, benchmarks);").unwrap();
    writeln!(&mut output, "criterion_main!(benches);").unwrap();

    output
}

/// Check if a function is suitable for benchmarking
fn is_benchmarkable(func: &FfiFunction) -> bool {
    // Skip functions that require complex setup
    let name_lower = func.name.to_lowercase();

    if name_lower.contains("create")
        || name_lower.contains("destroy")
        || name_lower.contains("init")
        || name_lower.contains("cleanup")
    {
        return false;
    }

    // Skip functions with many parameters
    if func.params.len() > 5 {
        return false;
    }

    true
}

/// Generate a benchmark for a specific function
fn generate_function_benchmark(output: &mut String, func: &FfiFunction) {
    let func_name = sanitize_name(&func.name);

    writeln!(output, "fn bench_{}() {{", func_name).unwrap();
    writeln!(output, "    // Setup").unwrap();
    writeln!(output, "    // TODO: Add necessary initialization").unwrap();
    writeln!(output).unwrap();
    writeln!(output, "    // Benchmark the wrapper").unwrap();
    writeln!(output, "    black_box(/* call wrapper function */);",).unwrap();
    writeln!(output, "}}").unwrap();
    writeln!(output).unwrap();
}

/// Sanitize function name for Rust
fn sanitize_name(name: &str) -> String {
    name.to_lowercase().replace("::", "_")
}

/// Generate Cargo.toml benchmark configuration
pub fn generate_bench_cargo_config() -> String {
    let mut output = String::new();

    writeln!(&mut output, "\n[[bench]]").unwrap();
    writeln!(&mut output, "name = \"performance\"").unwrap();
    writeln!(&mut output, "harness = false").unwrap();

    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ffi::FfiFunction;

    #[test]
    fn test_benchmarkable_detection() {
        let good_func = FfiFunction {
            name: "processData".to_string(),
            params: vec![],
            return_type: "int".to_string(),
            docs: None,
        };
        assert!(is_benchmarkable(&good_func));

        let create_func = FfiFunction {
            name: "cudaCreate".to_string(),
            params: vec![],
            return_type: "int".to_string(),
            docs: None,
        };
        assert!(!is_benchmarkable(&create_func));
    }

    #[test]
    fn test_benchmark_generation() {
        let mut ffi_info = FfiInfo::default();
        ffi_info.functions.push(FfiFunction {
            name: "testFunc".to_string(),
            params: vec![],
            return_type: "int".to_string(),
            docs: None,
        });

        let output = generate_benchmarks(&ffi_info, "my_crate");
        assert!(output.contains("criterion"));
        assert!(output.contains("bench_testfunc"));
        assert!(output.contains("criterion_main"));
    }

    #[test]
    fn test_sanitize_name() {
        assert_eq!(sanitize_name("cudaMemcpy"), "cudamemcpy");
        assert_eq!(sanitize_name("My::Function"), "my_function");
    }
}
