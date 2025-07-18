pub mod loader;
pub mod preprocessor;
pub mod schema;
pub mod sequence;
pub mod structures;
pub mod target_converter;

use serde::{Deserialize, Serialize};

pub use loader::DataLoader;
pub use preprocessor::DataPreprocessor;
pub use schema::{CryptoDataSchema, DataValidationError};
pub use sequence::SequenceGenerator;
pub use target_converter::TargetConverter;

use crate::config::training::ClassWeightStrategy;
use crate::targets::PreparedTargets;
use crate::targets::TargetType;
use crate::utils::error::Result;

use std::collections::HashMap;
use std::path::Path;

/// Main data pipeline orchestrator
pub struct DataPipeline {
    loader: DataLoader,
    preprocessor: DataPreprocessor,
    sequence_generator: SequenceGenerator,
}

impl Default for DataPipeline {
    fn default() -> Self {
        Self::new()
    }
}

impl DataPipeline {
    pub fn new() -> Self {
        Self {
            loader: DataLoader::new(),
            preprocessor: DataPreprocessor::new(),
            sequence_generator: SequenceGenerator::default(), // Uses no overlap by default
        }
    }

    /// Load and preprocess data for training with walk-forward analysis (default)
    pub async fn prepare_training_data<P: AsRef<Path>>(
        &self,
        data_path: P,
        config: &crate::config::TrainingConfig,
    ) -> Result<Vec<TrainingWindow>> {
        // Load raw data
        let raw_data = self.loader.load_csv(data_path).await?;

        // Validate schema
        CryptoDataSchema::validate(&raw_data)?;

        // Preprocess data (features, normalization, etc.)
        let (processed_data, normalization_stats) = self
            .preprocessor
            .process_for_training(raw_data, &config.data, Some(&config.features))
            .await?;

        // CRITICAL: Use walk-forward analysis to maximize data utilization
        let windows = self
            .create_walk_forward_windows(&processed_data, &normalization_stats, config)
            .await?;

        log::info!(
            "📊 Walk-forward analysis: {} windows created for progressive training",
            windows.len()
        );

        Ok(windows)
    }

    /// Calculate class weights for a specific training window
    /// Reuses the same logic as the LSTM model's class weight calculation
    fn calculate_window_class_weights(
        &self,
        train_data: &PreparedData,
        target_type: &TargetType,
        horizon: &str,
        config: &crate::config::TrainingConfig,
    ) -> Result<Option<Vec<f32>>> {
        // Get the target data for the specific target type and horizon
        let targets = match target_type {
            TargetType::PriceLevel => train_data.targets.price_levels.get(horizon),
            TargetType::Direction => train_data.targets.directions.get(horizon),
            TargetType::Volatility => train_data.targets.volatility.get(horizon),
        };

        let targets = match targets {
            Some(t) => t,
            None => {
                log::warn!(
                    "⚠️ No target data available for {:?} horizon {}, skipping class weights",
                    target_type,
                    horizon
                );
                return Ok(None);
            }
        };

        if targets.is_empty() {
            log::warn!(
                "⚠️ Empty target data for {:?} horizon {}, skipping class weights",
                target_type,
                horizon
            );
            return Ok(None);
        }

        // Get the correct number of classes from model configuration (same logic as LSTM model)
        let num_classes = match target_type {
            TargetType::PriceLevel => {
                if config.model.output_heads.price_levels.enabled {
                    config.model.output_heads.price_levels.bins as usize
                } else {
                    // Fallback: calculate from data but this should not happen
                    let max_class = targets.iter().max().unwrap_or(&0);
                    (*max_class + 1) as usize
                }
            }
            TargetType::Direction => 3,  // Down=0, Sideways=1, Up=2
            TargetType::Volatility => 3, // Low=0, Medium=1, High=2
        };

        // Count class frequencies
        let mut class_counts: HashMap<i32, usize> = HashMap::new();
        let mut total_samples = 0;

        for &target in targets.iter() {
            let class_id = target;
            *class_counts.entry(class_id).or_insert(0) += 1;
            total_samples += 1;
        }

        if num_classes < 2 {
            log::warn!(
                "⚠️ Only {} classes configured for {:?} horizon {}, skipping class weights",
                num_classes,
                target_type,
                horizon
            );
            return Ok(None);
        }

        // Calculate balanced class weights using sklearn's "balanced" strategy
        // weight[i] = total_samples / (num_classes * class_count[i])
        let mut weights = vec![1.0f32; num_classes];

        for (class_id, weight) in weights.iter_mut().enumerate().take(num_classes) {
            let class_count = class_counts.get(&(class_id as i32)).unwrap_or(&1);
            if *class_count > 0 {
                *weight = total_samples as f32 / (num_classes as f32 * *class_count as f32);
            }
        }

        log::debug!(
            "🎯 Window class weights for {:?} horizon {}: {:?} (from {} samples, {} classes configured)",
            target_type,
            horizon,
            weights,
            total_samples,
            num_classes
        );

        Ok(Some(weights))
    }

    /// Create walk-forward analysis windows using existing validation_split config
    async fn create_walk_forward_windows(
        &self,
        df: &polars::prelude::DataFrame,
        normalization_stats: &NormalizationStats,
        config: &crate::config::TrainingConfig,
    ) -> Result<Vec<TrainingWindow>> {
        let total_samples = df.height();
        let validation_size = (total_samples as f64 * config.training.validation_split) as usize;
        let min_train_size = total_samples / 2; // Start with at least 50% for initial training

        if validation_size == 0 || min_train_size + validation_size > total_samples {
            return Err(crate::utils::error::VangaError::DataError(
                "Insufficient data for walk-forward analysis".to_string(),
            ));
        }

        let mut windows = Vec::new();
        let mut train_end = min_train_size;

        // CRITICAL FIX: Calculate proper gap for walk-forward validation to prevent data leakage
        let sequence_length = match &config.model.sequence_length {
            crate::config::model::SequenceLengthConfig::Fixed(len) => *len as usize,
            crate::config::model::SequenceLengthConfig::Auto { min_length, .. } => {
                *min_length as usize
            }
            crate::config::model::SequenceLengthConfig::Adaptive => 60,
        };

        let max_horizon_steps = if !config.horizons.is_empty() {
            config
                .horizons
                .iter()
                .map(|h| crate::targets::volatility::parse_horizon_to_steps(h).unwrap_or(1))
                .max()
                .unwrap_or(72)
        } else {
            72
        };

        let gap_size = sequence_length + max_horizon_steps;

        log::info!(
            "🔒 Walk-forward gap calculation: sequence_length({}) + max_horizon_steps({}) = {} total gap",
            sequence_length,
            max_horizon_steps,
            gap_size
        );

        // Create progressive windows with proper gap to prevent data leakage
        while train_end + gap_size + validation_size <= total_samples {
            let val_start = train_end + gap_size; // PROPER GAP ADDED
            let val_end = val_start + validation_size;

            let train_df = df.slice(0, train_end);
            let val_df = df.slice(val_start as i64, validation_size);

            // Generate sequences for this window
            let train_sequences = self
                .sequence_generator
                .generate_training_sequences(
                    train_df,
                    normalization_stats.clone(),
                    &config.horizons,
                    &config.model,
                    &config.data,
                )
                .await?;

            let val_sequences = self
                .sequence_generator
                .generate_training_sequences(
                    val_df,
                    normalization_stats.clone(),
                    &config.horizons,
                    &config.model,
                    &config.data,
                )
                .await?;

            // Calculate per-window class weights based on configuration strategy
            let window_class_weights = match config.training.class_weight_strategy {
                ClassWeightStrategy::PerWindow => {
                    if let Some(primary_horizon) = config.horizons.first() {
                        self.calculate_window_class_weights(
                            &train_sequences,
                            &TargetType::PriceLevel,
                            primary_horizon,
                            config,
                        )
                        .unwrap_or_else(|e| {
                            log::warn!("⚠️ Failed to calculate window class weights: {}", e);
                            None
                        })
                    } else {
                        log::warn!("⚠️ No horizons configured, skipping window class weights");
                        None
                    }
                }
                ClassWeightStrategy::Global => {
                    // Global weights will be calculated once in the LSTM model
                    None
                }
                ClassWeightStrategy::None => {
                    // No class weighting
                    None
                }
            };

            windows.push(TrainingWindow {
                train_data: train_sequences,
                val_data: val_sequences,
                window_id: windows.len(),
                train_samples: train_end,
                val_samples: validation_size,
                class_weights: window_class_weights,
            });

            log::info!(
                "📊 Window {}: Train[0-{}] → Gap[{}-{}] → Val[{}-{}] | Gap: {} steps (prevents data leakage)",
                windows.len(),
                train_end,
                train_end,
                val_start,
                val_start,
                val_end,
                gap_size
            );

            // Move window forward by validation_size (no overlap in test sets)
            train_end = val_end;
        }

        // Calculate comprehensive data split statistics
        let total_train_sequences: usize = windows
            .iter()
            .map(|w| w.train_data.sequences.shape()[0])
            .sum();
        let total_val_sequences: usize = windows
            .iter()
            .map(|w| w.val_data.sequences.shape()[0])
            .sum();
        let avg_train_per_window = if !windows.is_empty() {
            total_train_sequences / windows.len()
        } else {
            0
        };
        let avg_val_per_window = if !windows.is_empty() {
            total_val_sequences / windows.len()
        } else {
            0
        };

        log::info!("📊 WALK-FORWARD ANALYSIS SUMMARY:");
        log::info!(
            "   • Windows: {} created with {:.1}% validation split",
            windows.len(),
            config.training.validation_split * 100.0
        );
        log::info!(
            "   • Total Sequences: {} train + {} validation = {}",
            total_train_sequences,
            total_val_sequences,
            total_train_sequences + total_val_sequences
        );
        log::info!(
            "   • Per Window Avg: {} train, {} validation sequences",
            avg_train_per_window,
            avg_val_per_window
        );
        log::info!(
            "   • Data Split Ratio: {:.1}% train / {:.1}% validation",
            (total_train_sequences as f64 / (total_train_sequences + total_val_sequences) as f64)
                * 100.0,
            (total_val_sequences as f64 / (total_train_sequences + total_val_sequences) as f64)
                * 100.0
        );
        log::info!("   • Chronological Order: Maintained (no data leakage)");
        log::info!("   • Test Split: Reserved for final evaluation (separate from validation)");
        log::info!("   • Validation Strategy: Walk-forward windows prevent overfitting");

        Ok(windows)
    }

    /// Load and preprocess data for prediction
    pub async fn prepare_prediction_data<P: AsRef<Path>>(
        &self,
        data_path: P,
        config: &crate::config::PredictionConfig,
    ) -> Result<PreparedPredictionData> {
        // Load raw data
        let raw_data = self.loader.load_csv(data_path).await?;

        // Validate schema
        CryptoDataSchema::validate(&raw_data)?;

        // Load model to get training config
        let model_path = crate::utils::model_path::get_model_path(&config.symbols[0]);
        let model = crate::model::multi_target::MultiTargetLSTMModel::load(&model_path)?;

        // Use stored training config for consistent preprocessing
        let processed_data = if let Some(training_config) = model.get_training_config() {
            log::info!("Using stored training config for consistent preprocessing");

            // STEP 1: Apply EXACT same preprocessing as training
            let (mut df, _normalization_stats) = self
                .preprocessor
                .process_for_training(
                    raw_data,
                    &training_config.data,
                    Some(&training_config.features),
                )
                .await?;

            log::info!(
                "✅ Applied same preprocessing as training: {} rows, {} columns",
                df.height(),
                df.width()
            );

            // STEP 2: CRITICAL - Apply normalization using stored training stats
            // This is the missing step that caused the bug!
            if let Some(normalization_stats) = model.get_normalization_stats() {
                log::info!("🔧 Applying normalization using stored training statistics");

                df = self.preprocessor.apply_normalization_with_stats(
                    df,
                    normalization_stats,
                    &training_config.data.normalization,
                )?;

                log::info!("✅ Normalization applied - model will receive properly scaled inputs");
            } else {
                log::warn!("⚠️  No normalization stats found in model - model may receive wrong input scale");
                log::warn!("    This suggests the model was trained with an older version that didn't store normalization stats");
            }

            // STEP 3: Extract most recent data for prediction (AFTER normalization)
            let sequence_length = match &training_config.model.sequence_length {
                crate::config::model::SequenceLengthConfig::Fixed(len) => *len as usize,
                crate::config::model::SequenceLengthConfig::Auto { min_length, .. } => {
                    *min_length as usize
                }
                crate::config::model::SequenceLengthConfig::Adaptive => 60,
            };

            let required_rows = sequence_length + 1;

            // Validate we have enough data after preprocessing
            if df.height() < required_rows {
                return Err(crate::utils::error::VangaError::DataError(format!(
                    "Insufficient data after preprocessing: {} rows available, {} required for prediction",
                    df.height(),
                    required_rows
                )));
            }

            // Take most recent data for prediction
            let start_idx = df.height() - required_rows;
            df = df.slice(start_idx as i64, required_rows);

            log::info!("🔍 PREDICTION PIPELINE SUMMARY:");
            log::info!("   ✅ Used exact same preprocessing as training");
            log::info!("   ✅ Applied normalization with stored training stats");
            log::info!(
                "   ✅ Extracted {} most recent rows for prediction",
                required_rows
            );
            log::info!("   ✅ Model will receive properly normalized inputs");

            df
        } else {
            // Fallback for old models without stored training config
            log::warn!("No training config found in model - using basic preprocessing (may cause feature mismatch)");
            self.preprocessor
                .process_for_prediction(raw_data, &config.symbols[0])
                .await?
        };

        // Generate prediction sequences using model config from training
        let model_config = if let Some(training_config) = model.get_training_config() {
            &training_config.model
        } else {
            // Fallback for old models
            &crate::config::ModelConfig::default()
        };

        let sequences = self
            .sequence_generator
            .generate_prediction_sequences(processed_data, &config.symbols[0], model_config)
            .await?;

        Ok(sequences)
    }

    /// Load and preprocess data for multi-symbol cross-asset prediction
    pub async fn prepare_cross_asset_prediction_data(
        &self,
        symbol_paths: &std::collections::HashMap<String, std::path::PathBuf>,
        _config: &crate::config::PredictionConfig,
        features_config: &crate::config::FeatureConfig,
    ) -> Result<std::collections::HashMap<String, PreparedPredictionData>> {
        log::info!(
            "Preparing cross-asset prediction data for {} symbols",
            symbol_paths.len()
        );

        // Load raw data for all symbols
        let mut symbol_data = std::collections::HashMap::new();
        for (symbol, path) in symbol_paths {
            let raw_data = self.loader.load_csv(path).await?;
            CryptoDataSchema::validate(&raw_data)?;
            symbol_data.insert(symbol.clone(), raw_data);
        }

        // Apply cross-asset preprocessing
        let processed_symbol_data = self
            .preprocessor
            .process_for_cross_asset_prediction(symbol_data, features_config)
            .await?;

        // Generate prediction sequences for each symbol
        let mut prepared_data = std::collections::HashMap::new();
        let default_model_config = crate::config::ModelConfig::default();

        for (symbol, processed_df) in processed_symbol_data {
            let sequences = self
                .sequence_generator
                .generate_prediction_sequences(processed_df, &symbol, &default_model_config)
                .await?;
            prepared_data.insert(symbol, sequences);
        }

        Ok(prepared_data)
    }
}

/// Prepared training data with sequences and targets
#[derive(Debug)]
pub struct PreparedData {
    pub sequences: ndarray::Array3<f64>, // [batch, sequence, features]
    pub targets: PreparedTargets,
    pub feature_names: Vec<String>,
    pub normalization_stats: NormalizationStats,
    pub metadata: DataMetadata,
}

/// Prepared prediction data
#[derive(Debug)]
pub struct PreparedPredictionData {
    pub sequences: ndarray::Array3<f64>, // [batch, sequence, features]
    pub feature_names: Vec<String>,
    pub metadata: DataMetadata,
}

/// Normalization statistics for features
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NormalizationStats {
    pub means: Vec<f64>,
    pub stds: Vec<f64>,
    pub mins: Vec<f64>,
    pub maxs: Vec<f64>,
    pub medians: Vec<f64>,
    pub q25: Vec<f64>,
    pub q75: Vec<f64>,
}

/// Data metadata
#[derive(Debug, Clone)]
pub struct DataMetadata {
    pub symbol: String,
    pub start_time: chrono::DateTime<chrono::Utc>,
    pub end_time: chrono::DateTime<chrono::Utc>,
    pub total_records: usize,
    pub feature_count: usize,
    pub sequence_length: usize,
    pub horizons: Vec<String>,
}

/// Walk-forward training window
#[derive(Debug)]
pub struct TrainingWindow {
    pub train_data: PreparedData,
    pub val_data: PreparedData,
    pub window_id: usize,
    pub train_samples: usize,
    pub val_samples: usize,
    /// Per-window class weights for balanced training
    pub class_weights: Option<Vec<f32>>,
}
