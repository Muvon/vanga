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

        // Apply feature engineering but NO global normalization
        let processed_data = self
            .preprocessor
            .process_features_only(raw_data, &config.data, Some(&config.features))
            .await?;

        // Create windows with raw data - normalization happens per-sequence
        let windows = self
            .create_walk_forward_windows(processed_data, config)
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
                    crate::config::model::NUM_CLASSES // Use unified 5-class system
                } else {
                    // Fallback: calculate from data but this should not happen
                    let max_class = targets.iter().max().unwrap_or(&0);
                    (*max_class + 1) as usize
                }
            }
            TargetType::Direction => crate::config::model::NUM_CLASSES, // Dump/Down/Sideways/Up/Pump
            TargetType::Volatility => crate::config::model::NUM_CLASSES, // VeryLow/Low/Medium/High/VeryHigh
        };

        // Count class frequencies
        let mut class_counts: HashMap<i32, usize> = HashMap::new();
        let mut total_samples = 0;

        for &target in targets.iter() {
            let class_id = target;
            *class_counts.entry(class_id).or_insert(0) += 1;
            total_samples += 1;
        }

        // Debug: Log detailed class distribution for this window
        log::debug!(
            "🔍 Window class distribution for {:?} horizon {}: {} total samples",
            target_type,
            horizon,
            total_samples
        );
        for (class_id, count) in &class_counts {
            let percentage = (*count as f64 / total_samples as f64) * 100.0;
            log::debug!(
                "   Class {}: {} samples ({:.2}%)",
                class_id,
                count,
                percentage
            );
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

        // Use advanced class weighting (same as price levels) for all target types
        use crate::targets::imbalance_mitigation::{
            AdvancedClassWeighter, ClassDistributionAnalysis, ImbalanceMitigationConfig,
        };

        let mitigation_config = ImbalanceMitigationConfig::default();
        let analysis = ClassDistributionAnalysis::analyze(targets, num_classes, &mitigation_config);
        let weights = AdvancedClassWeighter::calculate_weights(
            &analysis,
            &mitigation_config.weighting_strategy,
        )?;

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

    /// Calculate class weights for all target types and horizons
    fn calculate_all_target_class_weights(
        &self,
        train_data: &PreparedData,
        config: &crate::config::TrainingConfig,
    ) -> Result<HashMap<String, Vec<f32>>> {
        let mut target_weights = HashMap::new();

        // Define all target types to calculate weights for
        let target_types = [
            TargetType::PriceLevel,
            TargetType::Direction,
            TargetType::Volatility,
        ];

        for target_type in &target_types {
            for horizon in &config.horizons {
                // Calculate weights for this specific target type and horizon
                if let Ok(Some(weights)) =
                    self.calculate_window_class_weights(train_data, target_type, horizon, config)
                {
                    let key = format!("{:?}_{}", target_type, horizon);
                    target_weights.insert(key, weights);

                    log::debug!(
                        "📊 Calculated class weights for {:?} horizon {}: {} classes",
                        target_type,
                        horizon,
                        target_weights
                            .get(&format!("{:?}_{}", target_type, horizon))
                            .unwrap()
                            .len()
                    );
                }
            }
        }

        log::info!(
            "🎯 Calculated class weights for {} target-horizon combinations",
            target_weights.len()
        );

        Ok(target_weights)
    }

    /// Create walk-forward analysis windows with proper three-way split
    /// Reserves test_split for final evaluation while maximizing training data utilization
    async fn create_walk_forward_windows(
        &self,
        raw_processed_data: polars::prelude::DataFrame, // Has features but NOT normalized
        config: &crate::config::TrainingConfig,
    ) -> Result<Vec<TrainingWindow>> {
        let total_samples = raw_processed_data.height();

        // STEP 1: Reserve test set (never touched during training/validation)
        let test_size = (total_samples as f64 * config.training.test_split) as usize;
        let available_for_training = total_samples - test_size;

        // STEP 2: Calculate validation size from remaining data
        let validation_size =
            (available_for_training as f64 * config.training.validation_split) as usize;
        let min_train_size = available_for_training / 2; // Start with at least 50% for initial training

        if validation_size == 0 || min_train_size + validation_size > available_for_training {
            return Err(crate::utils::error::VangaError::DataError(
                format!(
                    "Insufficient data for walk-forward analysis: total={}, test_reserved={}, available={}, min_train={}, val={}",
                    total_samples, test_size, available_for_training, min_train_size, validation_size
                )
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

        log::info!(
            "📊 Three-way split setup: total={}, test_reserved={} ({:.1}%), available_for_training={}, val_size={} ({:.1}%)",
            total_samples,
            test_size,
            config.training.test_split * 100.0,
            available_for_training,
            validation_size,
            config.training.validation_split * 100.0
        );

        // STEP 3: Create progressive windows with proper gap to prevent data leakage
        // Only use available_for_training data, keep test set completely separate
        while train_end + gap_size + validation_size <= available_for_training {
            let val_start = train_end + gap_size; // PROPER GAP ADDED
            let _val_end = val_start + validation_size;

            let train_df = raw_processed_data.slice(0, train_end);
            let val_df = raw_processed_data.slice(val_start as i64, validation_size);

            // Test data is reserved - only include in final window
            let _test_df =
                if train_end + gap_size + validation_size + gap_size >= available_for_training {
                    // Final window - include test data for final evaluation
                    Some(raw_processed_data.slice(available_for_training as i64, test_size))
                } else {
                    // Intermediate window - no test data
                    None
                };

            // Generate sequences with per-sequence normalization
            let train_sequences = self
                .sequence_generator
                .generate_training_sequences(
                    train_df, // RAW data
                    &config.horizons,
                    &config.model,
                    &config.data,
                )
                .await?;

            let val_sequences = self
                .sequence_generator
                .generate_training_sequences(
                    val_df, // RAW data
                    &config.horizons,
                    &config.model,
                    &config.data,
                )
                .await?;

            // Generate test sequences - empty for intermediate windows, populated for final window
            let test_sequences =
                if train_end + gap_size + validation_size + gap_size >= available_for_training {
                    // Final window - include test data for final evaluation
                    self.sequence_generator
                        .generate_training_sequences(
                            raw_processed_data.slice(available_for_training as i64, test_size),
                            &config.horizons,
                            &config.model,
                            &config.data,
                        )
                        .await?
                } else {
                    // Intermediate window - create empty test data with same structure
                    PreparedData {
                        sequences: ndarray::Array3::zeros((
                            0,
                            train_sequences.sequences.shape()[1],
                            train_sequences.sequences.shape()[2],
                        )),
                        targets: crate::targets::PreparedTargets::new(0),
                        feature_names: train_sequences.feature_names.clone(),
                        normalization_stats: train_sequences.normalization_stats.clone(),
                        metadata: train_sequences.metadata.clone(),
                    }
                };

            // Calculate target-specific per-window class weights based on configuration strategy
            let target_class_weights = match config.training.class_weight_strategy {
                ClassWeightStrategy::PerWindow => self
                    .calculate_all_target_class_weights(&train_sequences, config)
                    .unwrap_or_else(|e| {
                        log::warn!(
                            "⚠️ Failed to calculate target-specific class weights: {}",
                            e
                        );
                        HashMap::new()
                    }),
                ClassWeightStrategy::Global => {
                    // Global weights will be calculated once in the LSTM model
                    HashMap::new()
                }
                ClassWeightStrategy::None => {
                    // No class weighting
                    HashMap::new()
                }
                ClassWeightStrategy::Advanced => {
                    // Use advanced imbalance mitigation strategies
                    self.calculate_all_target_class_weights(&train_sequences, config)
                        .unwrap_or_else(|e| {
                            log::warn!("⚠️ Failed to calculate advanced class weights: {}", e);
                            HashMap::new()
                        })
                }
            };

            // Log target class weights summary for this window
            if !target_class_weights.is_empty() {
                log::info!(
                    "🎯 Window {} class weights: {} target-horizon combinations calculated",
                    windows.len() + 1,
                    target_class_weights.len()
                );
                for (key, weights) in &target_class_weights {
                    log::debug!("   {}: {:?}", key, weights);
                }
            } else {
                log::info!(
                    "🎯 Window {}: No class weights calculated (strategy: {:?})",
                    windows.len() + 1,
                    config.training.class_weight_strategy
                );
            }

            windows.push(TrainingWindow {
                train_data: train_sequences,
                val_data: val_sequences,
                test_data: test_sequences.clone(),
                window_id: windows.len(),
                train_samples: train_end,
                val_samples: validation_size,
                test_samples: test_sequences.sequences.shape()[0],
                target_class_weights,
            });

            // Progress to next window - increase training data size
            train_end += validation_size / 4; // Increment by 25% of validation size for smooth progression

            log::debug!(
                "📊 Window {}: train_samples={}, val_samples={}, test_samples={}, val_start={}",
                windows.len(),
                train_end,
                validation_size,
                test_sequences.sequences.shape()[0],
                val_start
            );
        }

        if windows.is_empty() {
            return Err(crate::utils::error::VangaError::DataError(
                "No valid walk-forward windows could be created".to_string(),
            ));
        }

        log::info!(
            "📊 Walk-forward windows created: {} windows with per-sequence normalization",
            windows.len()
        );

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

            // Apply EXACT same preprocessing as training (feature engineering + remove_nan_rows)
            let df = self
                .preprocessor
                .process_features_only(
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
            log::info!("✅ Per-sequence normalization will be applied during sequence generation");

            df
        } else {
            // Fallback for old models without stored training config
            log::warn!("No training config found in model - using basic preprocessing (may cause feature mismatch)");
            self.preprocessor
                .process_for_prediction(raw_data, &config.symbols[0], None)
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
#[derive(Debug, Clone)]
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
    /// OHLC data for the sequence used in prediction (for ATR calculation)
    pub sequence_ohlc: Option<Vec<crate::data::structures::MarketDataRow>>,
}

/// Normalization statistics for features
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
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

/// Walk-forward training window with proper three-way split
#[derive(Debug)]
pub struct TrainingWindow {
    pub train_data: PreparedData,
    pub val_data: PreparedData,
    /// Test data - empty for intermediate windows, populated for final evaluation
    pub test_data: PreparedData,
    pub window_id: usize,
    pub train_samples: usize,
    pub val_samples: usize,
    pub test_samples: usize,
    /// Target-specific class weights for balanced training
    /// Key format: "{target_type}_{horizon}" (e.g., "PriceLevel_1h", "Direction_4h")
    pub target_class_weights: HashMap<String, Vec<f32>>,
}
