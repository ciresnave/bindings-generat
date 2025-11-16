pub mod analyzer;
pub mod cli;
pub mod config;
pub mod discovery;
pub mod ffi;
pub mod generator;
pub mod interactive;
pub mod llm;
pub mod output;
pub mod sources;
pub mod utils;

pub use cli::Cli;
pub use config::Config;

use anyhow::Result;
use indicatif::{ProgressBar, ProgressStyle};
use tracing::info;

/// Main orchestrator for the bindings generation process
pub struct BindingsGenerator {
    config: Config,
    // Store intermediate results
    ffi_info: Option<ffi::FfiInfo>,
    analysis: Option<analyzer::AnalysisResult>,
    generated_code: Option<generator::GeneratedCode>,
    // LLM enhancements
    llm_enhancements: Option<llm::CodeEnhancements>,
    // Ollama installation state
    ollama_installer: Option<llm::OllamaInstaller>,
    ollama_process: Option<std::process::Child>,
}

impl BindingsGenerator {
    pub fn new(config: Config) -> Self {
        Self {
            config,
            ffi_info: None,
            analysis: None,
            generated_code: None,
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

    fn phase_discovery(&self) -> Result<(Vec<std::path::PathBuf>, String, std::path::PathBuf)> {
        let pb = Self::create_progress_bar("Phase 1: Preparing source...");

        info!("Starting source preparation phase");

        // Prepare source (download/extract if needed)
        let prepared_source = sources::prepare_source(&self.config.source)?;
        pb.println(format!(
            "   ✓ Source prepared: {}",
            prepared_source.path().display()
        ));

        // Now discover the library structure
        pb.set_message("Phase 1: Discovering library...".to_string());

        let discovery_result = discovery::discover(&prepared_source.path().to_path_buf())?;
        pb.println(format!(
            "   ✓ Found {} header files",
            discovery_result.headers.len()
        ));

        let lib_name = if let Some(name) = &self.config.lib_name {
            name.clone()
        } else {
            discovery_result.library_name.clone()
        };
        pb.println(format!("   ✓ Detected library: {}", lib_name));

        pb.finish_and_clear();
        Ok((
            discovery_result.headers,
            lib_name,
            prepared_source.path().to_path_buf(),
        ))
    }

    fn phase_ffi_generation(
        &mut self,
        headers: &[std::path::PathBuf],
        lib_name: &str,
        source_path: &std::path::Path,
    ) -> Result<()> {
        let pb = Self::create_progress_bar("Phase 2: Running bindgen...");

        info!("Starting FFI generation phase");

        let ffi_info = ffi::generate_and_parse_ffi(headers, lib_name, source_path)?;
        pb.println(format!(
            "   ✓ Generated FFI bindings ({} functions, {} types)",
            ffi_info.functions.len(),
            ffi_info.types.len()
        ));

        self.ffi_info = Some(ffi_info);

        pb.finish_and_clear();
        Ok(())
    }

    fn phase_pattern_analysis(&mut self) -> Result<()> {
        let pb = Self::create_progress_bar("Phase 3: Analyzing patterns...");

        info!("Starting pattern analysis phase");

        let ffi_info = self.ffi_info.as_ref().expect("FFI info not generated");
        let analysis = analyzer::analyze_ffi(ffi_info)?;

        pb.println(format!(
            "   ✓ Found {} handle types",
            analysis.raii_patterns.handle_types.len()
        ));
        pb.println(format!(
            "   ✓ Detected {} lifecycle pairs",
            analysis.raii_patterns.lifecycle_pairs.len()
        ));

        self.analysis = Some(analysis);

        pb.finish_and_clear();
        Ok(())
    }

    fn phase_code_generation(&mut self, lib_name: &str) -> Result<()> {
        let pb = Self::create_progress_bar("Phase 4: Generating safe wrappers...");

        info!("Starting code generation phase");

        let ffi_info = self.ffi_info.as_ref().expect("FFI info not generated");
        let analysis = self.analysis.as_ref().expect("Analysis not performed");

        let generated =
            generator::generate_code(ffi_info, analysis, lib_name, self.llm_enhancements.as_ref())?;

        pb.println("   ✓ Created RAII wrapper types");
        pb.println("   ✓ Converted status codes to Error");

        self.generated_code = Some(generated);

        pb.finish_and_clear();
        Ok(())
    }

    fn phase_llm_enhancement(&mut self) -> Result<()> {
        if !self.config.use_llm {
            info!("LLM enhancement disabled via config");
            return Ok(());
        }

        let pb = Self::create_progress_bar("Phase 5: Enhancing with LLM...");

        info!("Starting LLM enhancement phase");

        // Check if Ollama is available
        if !llm::is_ollama_available() {
            pb.println("   ⚠ Ollama not detected, skipping LLM enhancement");
            pb.println("      Install Ollama from https://ollama.ai to enable this feature");
            pb.finish_and_clear();
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
                    pb.println(format!("   ⚠ Failed to create LLM client: {}", e));
                    pb.finish_and_clear();
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
                    pb.println(format!("   ⚠ Failed to create LLM client: {}", e));
                    pb.finish_and_clear();
                    return Ok(());
                }
            }
        };

        let mut enhanced_count = 0;

        // Enhance error messages
        if let Some(analysis) = &self.analysis
            && let Some(error_enum) = analysis.error_patterns.error_enums.first()
        {
            pb.set_message("Enhancing error messages...");
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
            pb.set_message("Enhancing function documentation...");
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
            pb.set_message("Suggesting idiomatic names...");
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

        pb.println(format!(
            "   ✓ LLM enhancement complete ({} items enhanced)",
            enhanced_count
        ));
        pb.finish_and_clear();

        // Store the enhancements for use in code generation
        self.llm_enhancements = Some(enhancements);

        Ok(())
    }

    fn phase_interactive(&mut self) -> Result<()> {
        let pb = Self::create_progress_bar("Phase 6: Clarifying ambiguities...");

        info!("Starting interactive phase");

        // Check if we have analysis results with RAII patterns
        if let Some(analysis) = &self.analysis {
            // Check if there are low-confidence patterns that need clarification
            let has_low_confidence = analysis
                .raii_patterns
                .lifecycle_pairs
                .iter()
                .any(|pair| pair.confidence < 0.7);

            if has_low_confidence {
                pb.println("   ⚠ Low-confidence patterns detected, requesting clarification...");
                pb.finish_and_clear();

                // Request user clarification
                let clarifications = interactive::clarify_patterns(&analysis.raii_patterns)?;

                let confirmed_count = clarifications.confirmed_pairs.len();

                // Update analysis with confirmed pairs
                if let Some(mut_analysis) = &mut self.analysis {
                    mut_analysis.raii_patterns.lifecycle_pairs = clarifications.confirmed_pairs;
                }

                println!(
                    "   ✓ Clarification complete ({} pairs confirmed)",
                    confirmed_count
                );
            } else {
                pb.println("   ℹ No ambiguities detected");
                pb.finish_and_clear();
            }
        } else {
            pb.println("   ℹ No analysis available for clarification");
            pb.finish_and_clear();
        }

        Ok(())
    }

    fn phase_output(&self, lib_name: &str) -> Result<()> {
        let pb = Self::create_progress_bar(&format!(
            "Phase 7: Writing output to {}...",
            self.config.output_path.display()
        ));

        info!("Starting output phase");

        let generated = self.generated_code.as_ref().expect("Code not generated");

        output::output_generated_code(&self.config.output_path, &generated.lib_rs, lib_name)?;

        pb.println("   ✓ Created Cargo.toml");
        pb.println("   ✓ Created src/lib.rs");
        pb.println("   ✓ Created build.rs");

        pb.finish_and_clear();
        Ok(())
    }

    fn phase_validation(&self) -> Result<()> {
        let pb = Self::create_progress_bar("Phase 8: Validating...");

        info!("Starting validation phase");

        if output::validator::is_cargo_available() {
            match output::validate_code(&self.config.output_path) {
                Ok(true) => pb.println("   ✓ Checking with cargo check... SUCCESS!"),
                Ok(false) => pb.println("   ⚠ Checking with cargo check... FAILED (see logs)"),
                Err(e) => pb.println(format!("   ⚠ Validation error: {}", e)),
            }
        } else {
            pb.println("   ⚠ cargo not available, skipping validation");
        }

        pb.finish_and_clear();
        Ok(())
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

    fn create_progress_bar(message: &str) -> ProgressBar {
        let pb = ProgressBar::new_spinner();
        pb.set_style(
            ProgressStyle::default_spinner()
                .template("{spinner:.green} {msg}")
                .unwrap(),
        );
        pb.set_message(message.to_string());
        pb.enable_steady_tick(std::time::Duration::from_millis(100));
        pb
    }
}
