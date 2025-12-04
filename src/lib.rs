//! Automated generation of safe, idiomatic Rust bindings from C/C++ libraries.
//!
//! `bindings-generat` analyzes C/C++ header files and generates complete Rust crates
//! with safe wrappers, RAII resource management, type-safe error handling, and
//! comprehensive documentation.
//!
//! ## Features
//!
//! - **Automatic Pattern Detection** - Identifies RAII pairs, error enums, handle types
//! - **Safe Wrapper Generation** - Creates Rust wrappers with automatic resource cleanup
//! - **Smart Library Discovery** - Finds and links required system libraries automatically
//! - **Iterative Refinement** - Build-test-fix loop ensures generated code compiles
//! - **LLM Enhancement** - Optional AI-powered documentation and naming improvements
//! - **Platform Support** - Windows, Linux, macOS with platform-specific features
//!
//! ## Quick Start
//!
//! ```ignore
//! use bindings_generat::{BindingsGenerator, Config};
//!
//! let config = Config {
//!     header_path: "/path/to/library.h".into(),
//!     output_dir: "generated-crate".into(),
//!     library_name: Some("mylib".into()),
//!     ..Default::default()
//! };
//!
//! let mut generator = BindingsGenerator::new(config);
//! generator.generate()?;
//! ```
//!
//! ## Architecture
//!
//! The generation process follows a pipeline:
//!
//! 1. **FFI Generation** ([`ffi`]) - Parse C headers with bindgen
//! 2. **Analysis** ([`analyzer`]) - Detect RAII patterns, errors, ownership
//! 3. **Code Generation** ([`generator`]) - Create safe wrappers and documentation
//! 4. **Validation** ([`output`]) - Build, test, and refine until it compiles
//! 5. **Enhancement** ([`llm`]) - Optional AI improvements (requires Ollama)
//!
//! ## Key Modules
//!
//! - [`ffi`] - FFI binding generation and parsing
//! - [`analyzer`] - Pattern detection (RAII, errors, ownership)
//! - [`generator`] - Safe wrapper code generation
//! - [`output`] - Validation, formatting, and iterative refinement
//! - [`database`] - Library metadata for automated installation
//! - [`discovery`] - Automatic library discovery via Google Custom Search
//! - [`enrichment`] - Context enrichment from documentation, examples, and tests
//! - [`submission`] - Community contribution of discovered libraries
//! - [`llm`] - LLM-powered enhancements (documentation, naming)
//! - [`cli`] - Command-line interface
//! - [`config`] - User configuration and preferences
//!
//! ## Safety
//!
//! Generated bindings encapsulate all `unsafe` code and provide safe APIs:
//! - RAII types with automatic Drop implementations
//! - Error handling via Result types
//! - Type-safe function wrappers
//! - Proper lifetime management

pub mod analyzer;
pub mod assertions;
pub mod audit;
pub mod cli;
pub mod config;
pub mod database;
pub mod dependency_detection;
pub mod discovery;
pub mod ecosystem;
pub mod enrichment;
pub mod ffi;
pub mod generator;
pub mod interactive;
pub mod llm;
pub mod output;
pub mod publishing;
pub mod sources;
pub mod submission;
pub mod testing;
pub mod tooling;
pub mod tools;
pub mod user_config;
pub mod utils;

pub use cli::Cli;
pub use config::GeneratorConfig as Config;
pub use user_config::Config as UserConfig;

use anyhow::Result;
use tracing::{info, info_span, warn};

/// Main orchestrator for the bindings generation process
pub struct BindingsGenerator {
    config: Config,
    // Store intermediate results
    ffi_info: Option<ffi::FfiInfo>,
    ffi_bindings_code: Option<String>,
    analysis: Option<analyzer::AnalysisResult>,
    generated_code: Option<generator::GeneratedCode>,
    // Context enrichment
    library_files: Option<enrichment::LibraryFiles>,
    enriched_context: Option<enrichment::EnhancedContext>,
    // LLM enhancements
    llm_enhancements: Option<llm::CodeEnhancements>,
    // Ollama installation state
    ollama_installer: Option<llm::OllamaInstaller>,
    ollama_process: Option<std::process::Child>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::database;
    use crate::ffi::FfiInfo;
    use crate::generator::GeneratedCode;

    #[test]
    fn bundling_rejected_when_non_redistributable() {
        // Find an entry in the embedded DB that is marked non-redistributable
        let db = database::LibraryDatabase::load_embedded().expect("Failed to load DB");
        let entry = db.libraries.iter().find(|e| !e.redistributable).expect(
            "Embedded DB must contain at least one non-redistributable entry for this test",
        );

        // Create a generator configured to bundle
        let mut cfg = Config::default();
        cfg.bundle_library = true;
        cfg.lib_name = Some(entry.name.clone());
        cfg.dry_run = true; // avoid writing files

        let mut generator = BindingsGenerator::new(cfg);

        // Provide minimal FFI info and bindgen output so phase_output will run
        generator.ffi_info = Some(FfiInfo::default());
        generator.ffi_bindings_code = Some(String::new());
        generator.generated_code = Some(GeneratedCode {
            lib_rs: String::new(),
            ffi_bindings: String::new(),
            tests: String::new(),
            runtime_tests: String::new(),
            functional_tests: String::new(),
            loader_rs: String::new(),
            discovery_shared_rs: String::new(),
            discovery_install_rs: String::new(),
            ffi_dynamic_rs: String::new(),
        });

        // phase_output should reject bundling for non-redistributable libraries
        let res = generator.phase_output(&entry.name);
        assert!(
            res.is_err(),
            "Expected bundling to be rejected when redistributable == false"
        );
    }
}

impl BindingsGenerator {
    pub fn new(config: Config) -> Self {
        Self {
            config,
            ffi_info: None,
            ffi_bindings_code: None,
            analysis: None,
            generated_code: None,
            library_files: None,
            enriched_context: None,
            llm_enhancements: None,
            ollama_installer: None,
            ollama_process: None,
        }
    }

    pub fn run(&mut self) -> Result<()> {
        println!(
            "🔧 bindings-generat v{} - Automatic FFI Wrapper Generator\n",
            env!("CARGO_PKG_VERSION")
        );

        // Phase 0: Ollama Setup (if LLM is enabled)
        if self.config.use_llm {
            self.phase_ollama_setup()?;
        }

        // Phase 1: Discovery
        let (headers, lib_name, source_path) = self.phase_discovery()?;

        // Phase 2: FFI Generation
        self.phase_ffi_generation(&headers, &lib_name, &source_path)?;

        // Phase 3: Pattern Analysis
        self.phase_pattern_analysis()?;

        // Phase 3.25: Async LLM Analysis (parameter and documentation)
        if self.config.use_llm {
            self.phase_async_llm_analysis()?;
        }

        // Phase 3.5: Context Enrichment
        self.phase_context_enrichment()?;

        // Phase 4: Code Generation
        self.phase_code_generation(&lib_name)?;

        // Phase 5: LLM Enhancement (optional)
        if self.config.use_llm {
            self.phase_llm_enhancement()?;
        }

        // Phase 6: Interactive Clarification (optional)
        if self.config.interactive {
            self.phase_interactive()?;
        }

        // Phase 7: Output Generation
        self.phase_output(&lib_name)?;

        // Phase 8: Validation
        self.phase_validation()?;

        // Phase 9: Cleanup (if we installed Ollama portably)
        self.phase_cleanup()?;

        println!("\n🎉 Done! Your safe Rust wrapper is ready.");
        println!("   Location: {}", self.config.output_path.display());
        println!("\nNext steps:");
        println!("   cd {}", self.config.output_path.display());
        println!("   cargo doc --open");

        Ok(())
    }

    fn phase_discovery(&mut self) -> Result<(Vec<std::path::PathBuf>, String, std::path::PathBuf)> {
        info_span!("Phase 1: Discovering library").in_scope(|| {
            // Prepare source (download/extract if needed)
            let prepared_source = sources::prepare_source(&self.config.source)?;
            info!("✓ Source prepared: {}", prepared_source.path().display());

            let discovery_result = discovery::discover(&prepared_source.path().to_path_buf())?;
            info!("✓ Found {} header files", discovery_result.headers.len());

            // Enrich context with documentation, examples, and tests
            let library_files = if let Some(first_header) = discovery_result.headers.first() {
                let files = enrichment::discover_library_files(first_header);
                info!(
                    "✓ Enrichment: {} docs, {} examples, {} tests",
                    files.documentation.len(),
                    files.examples.len(),
                    files.tests.len()
                );
                Some(files)
            } else {
                None
            };

            let lib_name = if let Some(name) = &self.config.lib_name {
                name.clone()
            } else {
                discovery_result.library_name.clone()
            };
            info!("✓ Detected library: {}", lib_name);

            // Store enrichment data
            self.library_files = library_files;

            // Auto-detect dependencies
            let dependencies = dependency_detection::detect_dependencies(
                &lib_name,
                &discovery_result.headers,
                &prepared_source.path().to_path_buf(),
            )?;

            if !dependencies.is_empty() {
                info!("✓ Auto-detected {} dependencies", dependencies.len());

                // Add detected include paths to configuration
                for dep in &dependencies {
                    for include_path in &dep.include_paths {
                        if !self.config.include_dirs.contains(include_path) {
                            info!(
                                "Adding auto-detected include path: {}",
                                include_path.display()
                            );
                            self.config.include_dirs.push(include_path.clone());
                        }
                    }

                    // Add detected library paths
                    for lib_path in &dep.lib_paths {
                        if !self.config.lib_paths.contains(lib_path) {
                            info!("Adding auto-detected library path: {}", lib_path.display());
                            self.config.lib_paths.push(lib_path.clone());
                        }
                    }

                    // Add detected link libraries
                    for link_lib in &dep.link_libs {
                        if !self.config.link_libs.contains(link_lib) {
                            info!("Adding auto-detected link library: {}", link_lib);
                            self.config.link_libs.push(link_lib.clone());
                        }
                    }
                }
            }

            Ok((
                discovery_result.headers,
                lib_name,
                prepared_source.path().to_path_buf(),
            ))
        })
    }

    fn phase_ffi_generation(
        &mut self,
        headers: &[std::path::PathBuf],
        lib_name: &str,
        source_path: &std::path::Path,
    ) -> Result<()> {
        info_span!("Phase 2: Running bindgen").in_scope(|| {
            let (ffi_info, bindings_code) =
                ffi::generate_and_parse_ffi(headers, lib_name, source_path, &self.config)?;

            let function_count = ffi_info.functions.len();
            let type_count = ffi_info.types.len();

            self.ffi_info = Some(ffi_info);
            self.ffi_bindings_code = Some(bindings_code);

            info!(
                "✓ Generated FFI bindings ({} functions, {} types)",
                function_count, type_count
            );
            Ok(())
        })
    }

    fn phase_pattern_analysis(&mut self) -> Result<()> {
        info_span!("Phase 3: Analyzing patterns").in_scope(|| {
            let ffi_info = self.ffi_info.as_ref().expect("FFI info not generated");
            let analysis = analyzer::analyze_ffi(ffi_info)?;

            let handle_count = analysis.raii_patterns.handle_types.len();
            let lifecycle_count = analysis.raii_patterns.lifecycle_pairs.len();

            self.analysis = Some(analysis);

            info!(
                "✓ Found {} handle types, {} lifecycle pairs",
                handle_count, lifecycle_count
            );
            Ok(())
        })
    }

    fn phase_async_llm_analysis(&mut self) -> Result<()> {
        info_span!("Phase 3.25: Running async LLM analyzers").in_scope(|| {
            let ffi_info = self.ffi_info.as_ref().expect("FFI info not generated");
            let analysis = self.analysis.as_mut().expect("Analysis not completed");

            // Run async analysis in a blocking context
            let runtime = tokio::runtime::Runtime::new()?;
            runtime.block_on(async {
                analyzer::analyze_ffi_async(ffi_info, analysis, self.config.use_llm).await
            })?;

            // Log results
            if let Some(param_analysis) = &analysis.parameter_analysis {
                info!(
                    "✓ Parameter analysis: {} functions analyzed",
                    param_analysis.function_analysis.len()
                );
            }
            if let Some(doc_analysis) = &analysis.enhanced_docs {
                info!(
                    "✓ Documentation enhancement: {} function docs, {} type docs",
                    doc_analysis.function_docs.len(),
                    doc_analysis.type_docs.len()
                );
            }

            Ok(())
        })
    }

    fn phase_context_enrichment(&mut self) -> Result<()> {
        info_span!("Phase 3.5: Building enriched context").in_scope(|| {
            let mut context = enrichment::EnhancedContext::new();

            // Get the source directory from config
            let source_path = std::path::PathBuf::from(&self.config.source);
            let library_root = if source_path.is_dir() {
                source_path.clone()
            } else if let Some(parent) = source_path.parent() {
                parent.to_path_buf()
            } else {
                std::path::PathBuf::from(".")
            };

            info!(
                "Discovering documentation and examples in {:?}",
                library_root
            );

            // 1. Discover documentation files, examples, and tests
            let library_files = enrichment::discover_library_files(&library_root);
            info!(
                "Found {} documentation files, {} examples, {} tests",
                library_files.documentation.len(),
                library_files.examples.len(),
                library_files.tests.len()
            );

            // 2. Parse header comments from all header files
            info!("Parsing header comments from FFI functions");
            let ffi_info = self.ffi_info.as_ref().expect("FFI info not generated");

            // Create header parser
            let header_parser = match enrichment::HeaderCommentParser::new() {
                Ok(parser) => parser,
                Err(e) => {
                    warn!("Failed to create header parser: {}", e);
                    // Continue without header parsing but with basic contexts
                    for func in &ffi_info.functions {
                        let func_ctx = enrichment::FunctionContext::new(func.name.clone());
                        context.add_function(func_ctx);
                    }
                    self.enriched_context = Some(context);
                    info!(
                        "✓ Enriched context ready: {} functions with basic contexts",
                        ffi_info.functions.len()
                    );
                    return Ok(());
                }
            };

            // Parse comments from discovered header files
            let mut all_comments = Vec::new();
            for header_path in &self.config.headers {
                match header_parser.parse_header_file(header_path) {
                    Ok(comments) => {
                        info!("Parsed {} comments from {:?}", comments.len(), header_path);
                        all_comments.extend(comments);
                    }
                    Err(e) => {
                        warn!("Failed to parse header {:?}: {}", header_path, e);
                    }
                }
            }

            // Create function contexts from parsed comments
            for comment in all_comments {
                let func_ctx = enrichment::FunctionContext::from_header_comment(
                    comment.function_name.clone(),
                    &comment,
                );
                context.add_function(func_ctx);
            }

            // For functions without comments, create basic contexts
            for func in &ffi_info.functions {
                if !context.functions.contains_key(&func.name) {
                    let func_ctx = enrichment::FunctionContext::new(func.name.clone());
                    context.add_function(func_ctx);
                }
            }

            // 3. Run analyzers on each function context
            info!(
                "Running semantic analyzers on {} functions",
                context.functions.len()
            );

            // Thread safety analysis
            context.analyze_all_thread_safety();

            // Analyze each function with all remaining analyzers
            let func_names: Vec<String> = context.functions.keys().cloned().collect();
            for func_name in &func_names {
                if let Some(func_ctx) = context.functions.get_mut(func_name) {
                    // Run each analyzer (these analyze the documentation text)
                    func_ctx.analyze_ownership(&mut context.ownership_analyzer);
                    func_ctx.analyze_preconditions(&mut context.precondition_analyzer);
                    func_ctx.analyze_error_semantics(&mut context.error_semantics_analyzer);
                    func_ctx.analyze_callbacks(&mut context.callback_analyzer);
                    func_ctx.analyze_api_sequences(&mut context.api_sequence_analyzer);
                    func_ctx.analyze_resource_limits(&mut context.resource_limits_analyzer);
                    func_ctx.analyze_semantic_grouping(&mut context.semantic_grouping_analyzer);
                    func_ctx.analyze_global_state(&mut context.global_state_analyzer);
                    func_ctx.analyze_numeric_constraints(&mut context.numeric_constraints_analyzer);
                    func_ctx.analyze_performance(&mut context.performance_analyzer);

                    // Anti-pattern detection needs full doc text
                    if let Some(desc) = &func_ctx.description {
                        let pitfalls = context.anti_pattern_analyzer.analyze(func_name, desc, None);
                        func_ctx.pitfalls = Some(pitfalls);
                    }
                }
            }

            // 4. Parse changelog if available
            for doc_file in &library_files.documentation {
                if doc_file
                    .path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .map(|n| {
                        let lower = n.to_lowercase();
                        lower.contains("changelog") || lower.contains("changes")
                    })
                    .unwrap_or(false)
                    && let Ok(content) = std::fs::read_to_string(&doc_file.path)
                {
                    info!("Parsing changelog: {:?}", doc_file.path);
                    context
                        .parse_changelog(doc_file.path.to_str().unwrap_or("CHANGELOG"), &content);
                    context.apply_deprecations();
                }
            }

            let enriched_count = context
                .functions
                .values()
                .filter(|ctx| {
                    ctx.description.is_some()
                        || ctx.thread_safety.is_some()
                        || ctx.ownership.is_some()
                        || ctx.preconditions.is_some()
                })
                .count();

            // Preserve a copy of the enriched context for later stages (FFI enricher,
            // external doc generation, etc.). We clone only the function contexts and
            // changelog entries so the heavy analyzer instances remain owned by
            // `context` while allowing both `analysis` and `self.enriched_context` to
            // access the semantic data they need.
            let mut preserved = enrichment::EnhancedContext::new();
            preserved.functions = context.functions.clone();
            preserved.changelog_entries = context.changelog_entries.clone();
            self.enriched_context = Some(preserved);

            // MERGE enriched context into analysis result (move ownership)
            if let Some(analysis) = &mut self.analysis {
                analysis.function_contexts = context.functions;
                analysis.changelog_entries = context.changelog_entries;
                info!("✓ Merged enriched context into analysis result");
            }

            info!(
                "✓ Enriched context ready: {} functions analyzed, {} with semantic data",
                ffi_info.functions.len(),
                enriched_count
            );
            Ok(())
        })
    }

    fn phase_code_generation(&mut self, lib_name: &str) -> Result<()> {
        info_span!("Phase 4: Generating safe wrappers").in_scope(|| {
            let ffi_info = self.ffi_info.as_ref().expect("FFI info not generated");
            let analysis = self.analysis.as_ref().expect("Analysis not performed");

            let mut generated = generator::generate_code(
                ffi_info,
                analysis,
                lib_name,
                self.llm_enhancements.as_ref(),
            )?;

            // Populate GeneratedCode.ffi_bindings with the raw bindgen output
            // produced earlier in Phase 2 so the writer can emit
            // `src/ffi_bindings.rs`.
            if let Some(bindings) = &self.ffi_bindings_code {
                generated.ffi_bindings = bindings.clone();
            }

            self.generated_code = Some(generated);

            info!("✓ Created safe Rust wrappers");
            Ok(())
        })
    }

    fn phase_llm_enhancement(&mut self) -> Result<()> {
        if !self.config.use_llm {
            info!("LLM enhancement disabled via config");
            return Ok(());
        }

        let _span = info_span!("Phase 5: Enhancing with LLM").entered();

        // Check if Ollama is available
        if !llm::is_ollama_available() {
            info!("⚠ Ollama not detected, skipping LLM enhancement");
            info!("   Install Ollama from https://ollama.ai to enable this feature");
            return Ok(());
        }

        // Create enhancement storage
        let mut enhancements = llm::CodeEnhancements::new();

        // Create documentation enhancer (with custom URL if portable install)
        let enhancer = if let Some(ref installer) = self.ollama_installer {
            let base_url = installer.ollama_host();
            match llm::docs::DocsEnhancer::with_base_url(
                self.config.llm_model.clone(),
                &base_url,
                Some(self.config.cache_dir.clone()),
            ) {
                Ok(e) => e,
                Err(e) => {
                    info!("⚠ Failed to create LLM client: {}", e);
                    return Ok(());
                }
            }
        } else {
            match llm::docs::DocsEnhancer::new(
                self.config.llm_model.clone(),
                Some(self.config.cache_dir.clone()),
            ) {
                Ok(e) => e,
                Err(e) => {
                    info!("⚠ Failed to create LLM client: {}", e);
                    return Ok(());
                }
            }
        };

        let mut enhanced_count = 0;

        // Enhance error messages
        if let Some(analysis) = &self.analysis
            && let Some(error_enum) = analysis.error_patterns.error_enums.first()
        {
            // Enhancing error messages
            for variant in &error_enum.error_variants {
                if let Ok(Some(message)) = enhancer.enhance_error_message(variant, &error_enum.name)
                {
                    info!("Enhanced error message for {}: {}", variant, message);
                    enhancements.add_error_message(variant.clone(), message);
                    enhanced_count += 1;
                }
            }
        }

        // Enhance function documentation
        if let Some(ffi_info) = &self.ffi_info {
            // Enhancing function documentation
            let sample_funcs: Vec<_> = ffi_info
                .functions
                .iter()
                .take(5) // Sample first 5 functions to avoid too many LLM calls
                .collect();

            for func in sample_funcs {
                let signature = format!(
                    "{}({}) -> {}",
                    func.name,
                    func.params
                        .iter()
                        .map(|p| format!("{}: {}", p.name, p.ty))
                        .collect::<Vec<_>>()
                        .join(", "),
                    func.return_type
                );

                let context = format!(
                    "C FFI function from {} library",
                    self.config.lib_name.as_deref().unwrap_or("unknown")
                );

                if let Ok(Some(enhanced_docs)) =
                    enhancer.enhance_function_docs(&func.name, &signature, &context)
                {
                    info!("Enhanced docs for {}: {}", func.name, enhanced_docs);
                    enhancements.add_function_docs(func.name.clone(), enhanced_docs);
                    enhanced_count += 1;
                }
            }
        }

        // Suggest better naming for C-style functions
        if let Some(ffi_info) = &self.ffi_info {
            // Suggesting idiomatic names
            let sample_funcs: Vec<_> = ffi_info
                .functions
                .iter()
                .filter(|f| f.name.contains('_')) // C-style names with underscores
                .take(3)
                .collect();

            for func in sample_funcs {
                let context = format!(
                    "C FFI function from {} library: {}",
                    self.config.lib_name.as_deref().unwrap_or("unknown"),
                    func.name
                );

                if let Ok(Some(suggestions)) =
                    enhancer.suggest_names(&func.name, &context, "function")
                {
                    info!("Naming suggestions for {}: {:?}", func.name, suggestions);
                    // Store the first suggestion as the preferred name
                    if let Some(first_suggestion) = suggestions.first() {
                        enhancements.add_function_name(func.name.clone(), first_suggestion.clone());
                    }
                    enhanced_count += 1;
                }
            }
        }

        info!(
            "✓ LLM enhancement complete ({} items enhanced)",
            enhanced_count
        );

        // Store the enhancements for use in code generation
        self.llm_enhancements = Some(enhancements);

        Ok(())
    }

    fn phase_interactive(&mut self) -> Result<()> {
        // Use println instead of tracing to avoid interference with indicatif spinners
        println!("Phase 6: Clarifying ambiguities");
        std::io::Write::flush(&mut std::io::stdout())?;

        // Check if we have analysis results with RAII patterns
        if let Some(analysis) = &self.analysis {
            // Check if there are low-confidence patterns that need clarification
            let has_low_confidence = analysis
                .raii_patterns
                .lifecycle_pairs
                .iter()
                .any(|pair| pair.confidence < 0.7);

            if has_low_confidence {
                println!("⚠ Low-confidence patterns detected, requesting clarification...");
                std::io::Write::flush(&mut std::io::stdout())?;

                // Request user clarification
                let clarifications = interactive::clarify_patterns(&analysis.raii_patterns)?;

                let confirmed_count = clarifications.confirmed_pairs.len();

                // Update analysis with confirmed pairs
                if let Some(mut_analysis) = &mut self.analysis {
                    mut_analysis.raii_patterns.lifecycle_pairs = clarifications.confirmed_pairs;
                }

                println!(
                    "✓ Clarification complete ({} pairs confirmed)",
                    confirmed_count
                );
                std::io::Write::flush(&mut std::io::stdout())?;
            } else {
                println!("ℹ No ambiguities detected");
                std::io::Write::flush(&mut std::io::stdout())?;
            }
        } else {
            println!("ℹ No analysis available for clarification");
            std::io::Write::flush(&mut std::io::stdout())?;
        }

        Ok(())
    }

    fn phase_output(&self, lib_name: &str) -> Result<()> {
        // Use println to avoid indicatif spinner interference
        println!(
            "Phase 7: Writing output (path: {})",
            self.config.output_path.display()
        );
        std::io::Write::flush(&mut std::io::stdout())?;

        let generated = self.generated_code.as_ref().expect("Code not generated");
        let raw_bindings = self
            .ffi_bindings_code
            .as_ref()
            .expect("FFI bindings not generated");
        let ffi_info = self.ffi_info.as_ref().expect("FFI info not generated");

        // ENRICH FFI BINDINGS WITH DOCUMENTATION
        let enriched_bindings = if let Some(enriched_ctx) = &self.enriched_context {
            println!("  Enriching FFI bindings with comprehensive documentation...");
            std::io::Write::flush(&mut std::io::stdout())?;

            match ffi::enricher::enrich_ffi_bindings(raw_bindings, ffi_info, enriched_ctx) {
                Ok(enriched) => {
                    println!("  ✓ FFI bindings enriched");
                    std::io::Write::flush(&mut std::io::stdout())?;
                    enriched
                }
                Err(e) => {
                    warn!("Failed to enrich FFI bindings: {}, using raw bindings", e);
                    raw_bindings.clone()
                }
            }
        } else {
            info!("No enriched context available, using raw bindings");
            raw_bindings.clone()
        };

        // If the user requested bundling, ensure the database allows redistribution
        if self.config.bundle_library {
            // Load embedded library DB and check redistributable flag
            let db = crate::database::LibraryDatabase::load_embedded()?;
            if let Some(entry) = db.find_by_name(lib_name) {
                if !entry.redistributable {
                    return Err(anyhow::anyhow!(
                        "Cannot bundle '{}' - license prohibits redistribution",
                        lib_name
                    ));
                }
            } else {
                return Err(anyhow::anyhow!(
                    "Cannot bundle '{}' - no license metadata found",
                    lib_name
                ));
            }
        }

        output::output_generated_code(
            &self.config.output_path,
            &generated.lib_rs,
            &enriched_bindings,
            &generated.ffi_dynamic_rs,
            &generated.tests,
            &generated.runtime_tests,
            &generated.functional_tests,
            &generated.loader_rs,
            &generated.discovery_shared_rs,
            &generated.discovery_install_rs,
            lib_name,
            &ffi_info.dependencies,
            &self.config,
        )?;

        println!("✓ Created Rust crate files");
        std::io::Write::flush(&mut std::io::stdout())?;
        Ok(())
    }

    fn phase_validation(&self) -> Result<()> {
        // Use println to avoid indicatif spinner interference
        println!("Phase 8: Validating");
        std::io::Write::flush(&mut std::io::stdout())?;

        // Skip validation in dry-run mode
        if self.config.dry_run {
            info!("DRY RUN: Skipping validation");
            return Ok(());
        }

        if !output::validator::is_cargo_available() {
            info!("⚠ cargo not available, skipping validation");
            return Ok(());
        }

        // Get the library name from the output directory
        let lib_name = self
            .config
            .output_path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("generated");

        // Get initial dependencies from FFI info
        let ffi_info = self.ffi_info.as_ref().expect("FFI info not generated");
        let initial_deps = &ffi_info.dependencies;

        // Use iterative refinement system
        println!("→ Running iterative build validation...");
        std::io::Write::flush(&mut std::io::stdout())?;
        match output::validate_and_refine(
            &self.config.output_path,
            lib_name,
            initial_deps,
            true,  // enable_refinement
            false, // interactive (allow user prompts)
        ) {
            Ok(output::BuildResult::Success) => {
                println!("✓ Build validation... SUCCESS!");
                std::io::Write::flush(&mut std::io::stdout())?;
                Ok(())
            }
            Ok(output::BuildResult::FailedAfterRetries { final_errors }) => {
                eprintln!(
                    "⚠ Build validation... FAILED after retries ({} errors)",
                    final_errors.len()
                );
                for (i, err) in final_errors.iter().take(5).enumerate() {
                    eprintln!("   Error {}: {:?}", i + 1, err);
                }
                if final_errors.len() > 5 {
                    eprintln!("   ... and {} more", final_errors.len() - 5);
                }
                std::io::Write::flush(&mut std::io::stderr())?;

                // Try LLM diagnosis of build errors
                if let Err(diagnosis_err) = self.phase_llm_error_diagnosis(&final_errors) {
                    warn!("LLM error diagnosis failed: {}", diagnosis_err);
                }

                Err(anyhow::anyhow!(
                    "Generated code failed to compile. See errors and diagnosis above."
                ))
            }
            Ok(output::BuildResult::UserCancelled) => {
                Err(anyhow::anyhow!("Build validation was cancelled by user"))
            }
            Err(e) => Err(anyhow::anyhow!("Validation error: {}", e)),
        }
    }

    fn phase_ollama_setup(&mut self) -> Result<()> {
        info!("Checking Ollama availability");

        // Check if Ollama is already installed
        if llm::OllamaInstaller::is_available() {
            info!("Ollama already installed");
            let installer = llm::OllamaInstaller::new();
            self.ollama_installer = Some(installer);
            return Ok(());
        }

        // Ollama not found, prompt user for installation
        let mut installer = llm::OllamaInstaller::new();

        match installer.prompt_installation() {
            Ok(llm::OllamaInstallType::SystemWide) => {
                installer.install_system_wide()?;
                self.ollama_installer = Some(installer);
                Ok(())
            }
            Ok(llm::OllamaInstallType::Portable { path, .. }) => {
                // Prompt for model selection
                let model = llm::select_model()?;
                let process = installer.install_portable(path, &model)?;
                self.ollama_process = Some(process);
                self.ollama_installer = Some(installer);
                Ok(())
            }
            Ok(llm::OllamaInstallType::AlreadyInstalled) => {
                self.ollama_installer = Some(installer);
                Ok(())
            }
            Err(e) => {
                // User chose to skip LLM features
                info!("User skipped Ollama installation: {}", e);
                self.config.use_llm = false;
                Ok(())
            }
        }
    }

    fn phase_llm_error_diagnosis(&self, errors: &[output::error_parser::BuildError]) -> Result<()> {
        info!("🔍 Diagnosing build errors with LLM...");

        // Check if LLM is available
        if !self.config.use_llm {
            info!("LLM not enabled, skipping error diagnosis");
            return Ok(());
        }

        // Check if Ollama is available
        if !llm::OllamaInstaller::is_available() {
            info!("Ollama not available, skipping error diagnosis");
            return Ok(());
        }

        // Format errors for LLM analysis
        let error_summary = errors
            .iter()
            .map(|err| format!("- {:?}", err))
            .collect::<Vec<_>>()
            .join("\n");

        let diagnosis_prompt = format!(
            r#"I generated Rust FFI bindings that failed to compile with these errors:

{}

The generated code includes:
- FFI function declarations from C headers
- Safe Rust wrapper structs and functions
- A build.rs file for linking

Please analyze these compilation errors and:
1. Identify the most likely root cause
2. Suggest specific fixes if possible
3. If unfixable, explain why the errors are irrecoverable

Be concise and focus on actionable solutions."#,
            error_summary
        );

        // Try to get diagnosis from LLM
        let client = match llm::OllamaClient::new(Some(self.config.cache_dir.clone())) {
            Ok(client) => client,
            Err(e) => {
                warn!("Failed to create Ollama client: {}", e);
                return Ok(());
            }
        };

        match client.generate(&self.config.llm_model, &diagnosis_prompt) {
            Ok(diagnosis) => {
                println!("\n🤖 LLM Error Diagnosis:");
                println!("────────────────────────");
                println!("{}", diagnosis.trim());
                println!("────────────────────────\n");
            }
            Err(e) => {
                warn!("LLM diagnosis failed: {}", e);
            }
        }

        Ok(())
    }

    fn phase_cleanup(&mut self) -> Result<()> {
        if let Some(ref mut installer) = self.ollama_installer {
            // Only prompt for cleanup if we installed portably
            if matches!(
                installer.install_type(),
                Some(llm::OllamaInstallType::Portable { .. })
            ) {
                installer.prompt_cleanup()?;
            }
        }
        Ok(())
    }
}
