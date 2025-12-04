//! Pattern storage and management for dependency detection
//!
//! This system manages multiple sources of dependency patterns:
//! - Hardcoded patterns for common libraries
//! - Learned patterns from successful detections
//! - LLM-generated patterns
//! - User-provided patterns

use super::enhanced_dependency_detection::*;
use anyhow::Result;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use tracing::{debug, info, warn};

pub struct PatternStorage {
    cache_dir: Option<PathBuf>,
    patterns: HashMap<String, DependencyPattern>,
}

impl PatternStorage {
    pub fn new(cache_dir: Option<PathBuf>) -> Result<Self> {
        let mut storage = Self {
            cache_dir,
            patterns: HashMap::new(),
        };
        
        // Load hardcoded patterns
        storage.load_hardcoded_patterns();
        
        // Load saved patterns from disk
        if let Some(ref cache_dir) = storage.cache_dir {
            storage.load_saved_patterns(cache_dir)?;
        }
        
        Ok(storage)
    }

    /// Load built-in patterns for common libraries
    fn load_hardcoded_patterns(&mut self) {
        // CUDA patterns
        self.patterns.insert("cuda".to_string(), DependencyPattern {
            name: "cuda".to_string(),
            source: PatternSource::Hardcoded,
            confidence: 0.95,
            function_patterns: vec![
                "cuda*".to_string(),
                "cu*".to_string(),
                "__cuda*".to_string(),
            ],
            type_patterns: vec![
                "cuda*".to_string(),
                "CU*".to_string(),
            ],
            header_indicators: vec!["cuda_runtime.h".to_string(), "cuda.h".to_string()],
            env_vars: vec!["CUDA_PATH".to_string(), "CUDA_HOME".to_string()],
            common_paths: if cfg!(target_os = "windows") {
                vec![
                    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\*".to_string(),
                    r"C:\CUDA\*".to_string(),
                ]
            } else {
                vec![
                    "/usr/local/cuda*".to_string(),
                    "/opt/cuda*".to_string(),
                    "/usr/cuda*".to_string(),
                ]
            },
            include_subpaths: vec!["include".to_string()],
            lib_subpaths: if cfg!(target_os = "windows") {
                vec!["lib/x64".to_string(), "lib".to_string()]
            } else {
                vec!["lib64".to_string(), "lib".to_string()]
            },
            link_libs: vec!["cuda".to_string(), "cudart".to_string()],
            pkg_config_name: Some("cuda".to_string()),
            created_at: chrono::Utc::now(),
            success_count: 0,
            failure_count: 0,
        });

        // cuDNN patterns
        self.patterns.insert("cudnn".to_string(), DependencyPattern {
            name: "cudnn".to_string(),
            source: PatternSource::Hardcoded,
            confidence: 0.95,
            function_patterns: vec![
                "cudnn*".to_string(),
            ],
            type_patterns: vec![
                "cudnn*".to_string(),
            ],
            header_indicators: vec!["cudnn.h".to_string()],
            env_vars: vec!["CUDNN_PATH".to_string(), "CUDNN_HOME".to_string()],
            common_paths: if cfg!(target_os = "windows") {
                vec![
                    r"C:\Program Files\NVIDIA\CUDNN\*".to_string(),
                    r"C:\Users\*\.cudnn\*".to_string(),
                ]
            } else {
                vec![
                    "/usr/local/cudnn*".to_string(),
                    "/opt/cudnn*".to_string(),
                ]
            },
            include_subpaths: vec!["include".to_string()],
            lib_subpaths: if cfg!(target_os = "windows") {
                vec!["lib".to_string()]
            } else {
                vec!["lib64".to_string(), "lib".to_string()]
            },
            link_libs: if cfg!(target_os = "windows") {
                vec!["cudnn64_9".to_string(), "cudnn".to_string()]
            } else {
                vec!["cudnn".to_string()]
            },
            pkg_config_name: Some("cudnn".to_string()),
            created_at: chrono::Utc::now(),
            success_count: 0,
            failure_count: 0,
        });

        // OpenCV patterns
        self.patterns.insert("opencv".to_string(), DependencyPattern {
            name: "opencv".to_string(),
            source: PatternSource::Hardcoded,
            confidence: 0.9,
            function_patterns: vec![
                "cv*".to_string(),
                "CV_*".to_string(),
            ],
            type_patterns: vec![
                "cv::*".to_string(),
                "Mat*".to_string(),
            ],
            header_indicators: vec!["opencv2/opencv.hpp".to_string(), "cv.h".to_string()],
            env_vars: vec!["OPENCV_DIR".to_string()],
            common_paths: vec![
                "/usr/local/opencv*".to_string(),
                "/opt/opencv*".to_string(),
                "/usr/include/opencv*".to_string(),
            ],
            include_subpaths: vec!["include".to_string(), "include/opencv2".to_string()],
            lib_subpaths: vec!["lib".to_string(), "lib64".to_string()],
            link_libs: vec!["opencv_core".to_string(), "opencv_imgproc".to_string()],
            pkg_config_name: Some("opencv4".to_string()),
            created_at: chrono::Utc::now(),
            success_count: 0,
            failure_count: 0,
        });

        // PyTorch patterns
        self.patterns.insert("pytorch".to_string(), DependencyPattern {
            name: "pytorch".to_string(),
            source: PatternSource::Hardcoded,
            confidence: 0.9,
            function_patterns: vec![
                "torch_*".to_string(),
                "at::*".to_string(),
                "c10::*".to_string(),
                "THC*".to_string(),
            ],
            type_patterns: vec![
                "torch::*".to_string(),
                "at::*".to_string(),
                "c10::*".to_string(),
            ],
            header_indicators: vec!["torch/torch.h".to_string(), "ATen/ATen.h".to_string()],
            env_vars: vec!["PYTORCH_PATH".to_string(), "TORCH_PATH".to_string()],
            common_paths: vec![
                "/usr/local/libtorch*".to_string(),
                "/opt/pytorch*".to_string(),
            ],
            include_subpaths: vec!["include".to_string(), "include/torch/csrc/api/include".to_string()],
            lib_subpaths: vec!["lib".to_string(), "lib64".to_string()],
            link_libs: vec!["torch".to_string(), "torch_cpu".to_string()],
            pkg_config_name: Some("torch".to_string()),
            created_at: chrono::Utc::now(),
            success_count: 0,
            failure_count: 0,
        });

        // FFmpeg patterns
        self.patterns.insert("ffmpeg".to_string(), DependencyPattern {
            name: "ffmpeg".to_string(),
            source: PatternSource::Hardcoded,
            confidence: 0.9,
            function_patterns: vec![
                "av_*".to_string(),
                "avcodec_*".to_string(),
                "avformat_*".to_string(),
                "avutil_*".to_string(),
            ],
            type_patterns: vec![
                "AVCodec*".to_string(),
                "AVFormat*".to_string(),
                "AVFrame*".to_string(),
            ],
            header_indicators: vec!["libavcodec/avcodec.h".to_string(), "libavformat/avformat.h".to_string()],
            env_vars: vec!["FFMPEG_PATH".to_string()],
            common_paths: vec![
                "/usr/local/ffmpeg*".to_string(),
                "/opt/ffmpeg*".to_string(),
            ],
            include_subpaths: vec!["include".to_string()],
            lib_subpaths: vec!["lib".to_string(), "lib64".to_string()],
            link_libs: vec!["avcodec".to_string(), "avformat".to_string(), "avutil".to_string()],
            pkg_config_name: Some("libavcodec".to_string()),
            created_at: chrono::Utc::now(),
            success_count: 0,
            failure_count: 0,
        });

        // OpenSSL patterns
        self.patterns.insert("openssl".to_string(), DependencyPattern {
            name: "openssl".to_string(),
            source: PatternSource::Hardcoded,
            confidence: 0.9,
            function_patterns: vec![
                "SSL_*".to_string(),
                "EVP_*".to_string(),
                "BN_*".to_string(),
                "RSA_*".to_string(),
            ],
            type_patterns: vec![
                "SSL*".to_string(),
                "EVP_*".to_string(),
            ],
            header_indicators: vec!["openssl/ssl.h".to_string(), "openssl/evp.h".to_string()],
            env_vars: vec!["OPENSSL_DIR".to_string()],
            common_paths: vec![
                "/usr/local/ssl*".to_string(),
                "/opt/openssl*".to_string(),
            ],
            include_subpaths: vec!["include".to_string()],
            lib_subpaths: vec!["lib".to_string(), "lib64".to_string()],
            link_libs: vec!["ssl".to_string(), "crypto".to_string()],
            pkg_config_name: Some("openssl".to_string()),
            created_at: chrono::Utc::now(),
            success_count: 0,
            failure_count: 0,
        });

        info!("Loaded {} hardcoded patterns", self.patterns.len());
    }

    /// Load saved patterns from disk
    fn load_saved_patterns(&mut self, cache_dir: &Path) -> Result<()> {
        let patterns_file = cache_dir.join("dependency_patterns.json");
        if patterns_file.exists() {
            let content = std::fs::read_to_string(&patterns_file)?;
            let saved_patterns: HashMap<String, DependencyPattern> = serde_json::from_str(&content)?;
            
            for (name, pattern) in saved_patterns {
                // Only add if we don't have this pattern yet, or if the saved one has better stats
                if let Some(existing) = self.patterns.get(&name) {
                    if pattern.success_count > existing.success_count {
                        self.patterns.insert(name, pattern);
                    }
                } else {
                    self.patterns.insert(name, pattern);
                }
            }
            
            info!("Loaded saved patterns from {}", patterns_file.display());
        }
        
        Ok(())
    }

    /// Save a new pattern to disk
    pub fn save_pattern(&mut self, pattern: &DependencyPattern) -> Result<()> {
        // Add to memory
        self.patterns.insert(pattern.name.clone(), pattern.clone());
        
        // Save to disk if we have a cache directory
        if let Some(ref cache_dir) = self.cache_dir {
            std::fs::create_dir_all(cache_dir)?;
            let patterns_file = cache_dir.join("dependency_patterns.json");
            
            // Load existing patterns and merge
            let mut all_patterns = if patterns_file.exists() {
                let content = std::fs::read_to_string(&patterns_file)?;
                serde_json::from_str::<HashMap<String, DependencyPattern>>(&content).unwrap_or_default()
            } else {
                HashMap::new()
            };
            
            all_patterns.insert(pattern.name.clone(), pattern.clone());
            
            let json = serde_json::to_string_pretty(&all_patterns)?;
            std::fs::write(&patterns_file, json)?;
            
            info!("Saved pattern '{}' to {}", pattern.name, patterns_file.display());
        }
        
        Ok(())
    }

    /// Get all patterns, sorted by confidence and success rate
    pub fn get_all_patterns(&self) -> Result<Vec<DependencyPattern>> {
        let mut patterns: Vec<DependencyPattern> = self.patterns.values().cloned().collect();
        
        // Sort by a combination of confidence and success rate
        patterns.sort_by(|a, b| {
            let score_a = a.confidence + (a.success_count as f32 / (a.success_count + a.failure_count + 1) as f32);
            let score_b = b.confidence + (b.success_count as f32 / (b.success_count + b.failure_count + 1) as f32);
            score_b.partial_cmp(&score_a).unwrap()
        });
        
        Ok(patterns)
    }

    /// Get patterns for a specific library
    pub fn get_patterns_for_library(&self, lib_name: &str) -> Vec<DependencyPattern> {
        self.patterns.values()
            .filter(|p| p.name.to_lowercase().contains(&lib_name.to_lowercase()))
            .cloned()
            .collect()
    }

    /// Update pattern statistics after successful/failed use
    pub fn update_pattern_stats(&mut self, pattern_name: &str, success: bool) -> Result<()> {
        if let Some(pattern) = self.patterns.get_mut(pattern_name) {
            if success {
                pattern.success_count += 1;
            } else {
                pattern.failure_count += 1;
            }
            
            // Adjust confidence based on success rate
            let total = pattern.success_count + pattern.failure_count;
            if total > 0 {
                let success_rate = pattern.success_count as f32 / total as f32;
                pattern.confidence = (pattern.confidence + success_rate) / 2.0;
            }
            
            // Save updated pattern to disk
            self.save_pattern(pattern)?;
        }
        
        Ok(())
    }

    /// Clean up old or poorly performing patterns
    pub fn cleanup_patterns(&mut self) -> Result<()> {
        let mut to_remove = Vec::new();
        
        for (name, pattern) in &self.patterns {
            // Remove patterns that have consistently failed
            let total_uses = pattern.success_count + pattern.failure_count;
            if total_uses > 10 {
                let success_rate = pattern.success_count as f32 / total_uses as f32;
                if success_rate < 0.1 {
                    to_remove.push(name.clone());
                }
            }
            
            // Remove very old patterns with low confidence (except hardcoded ones)
            if pattern.source != PatternSource::Hardcoded {
                let age_days = (chrono::Utc::now() - pattern.created_at).num_days();
                if age_days > 365 && pattern.confidence < 0.3 {
                    to_remove.push(name.clone());
                }
            }
        }
        
        for name in to_remove {
            self.patterns.remove(&name);
            info!("Removed poorly performing pattern: {}", name);
        }
        
        Ok(())
    }

    /// Export patterns for sharing
    pub fn export_patterns(&self, export_path: &Path) -> Result<()> {
        // Only export learned and user-provided patterns (not hardcoded ones)
        let exportable_patterns: HashMap<String, DependencyPattern> = self.patterns.iter()
            .filter(|(_, pattern)| {
                matches!(pattern.source, PatternSource::Learned | PatternSource::UserProvided | PatternSource::LlmGenerated)
            })
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        
        let json = serde_json::to_string_pretty(&exportable_patterns)?;
        std::fs::write(export_path, json)?;
        
        info!("Exported {} patterns to {}", exportable_patterns.len(), export_path.display());
        Ok(())
    }

    /// Import patterns from a file
    pub fn import_patterns(&mut self, import_path: &Path) -> Result<()> {
        let content = std::fs::read_to_string(import_path)?;
        let imported_patterns: HashMap<String, DependencyPattern> = serde_json::from_str(&content)?;
        
        let mut imported_count = 0;
        for (name, mut pattern) in imported_patterns {
            // Mark as learned since they're coming from external source
            pattern.source = PatternSource::Learned;
            
            // Only import if we don't have this pattern or if the imported one seems better
            let should_import = if let Some(existing) = self.patterns.get(&name) {
                pattern.success_count > existing.success_count || 
                (pattern.success_count == existing.success_count && pattern.confidence > existing.confidence)
            } else {
                true
            };
            
            if should_import {
                self.patterns.insert(name.clone(), pattern);
                imported_count += 1;
            }
        }
        
        info!("Imported {} new/better patterns from {}", imported_count, import_path.display());
        Ok(())
    }
}