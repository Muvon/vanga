pub mod loader;
pub mod preprocessor;
pub mod schema;
pub mod sequence;
pub mod structures;
pub mod target_converter;

pub use loader::DataLoader;
pub use preprocessor::DataPreprocessor;
pub use schema::{CryptoDataSchema, DataValidationError};
pub use sequence::SequenceGenerator;
pub use target_converter::TargetConverter;

use crate::targets::PreparedTargets;
use crate::utils::error::Result;

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
        let processed_data = self
            .preprocessor
            .process_for_training(raw_data, &config.data, Some(&config.features))
            .await?;

        // CRITICAL: Use walk-forward analysis to maximize data utilization
        let windows = self
            .create_walk_forward_windows(&processed_data, config)
            .await?;

        log::info!(
            "📊 Walk-forward analysis: {} windows created for progressive training",
            windows.len()
        );

        Ok(windows)
    }

    /// Create walk-forward analysis windows using existing validation_split config
    async fn create_walk_forward_windows(
        &self,
        df: &polars::prelude::DataFrame,
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

        // Create progressive windows
        while train_end + validation_size <= total_samples {
            let val_start = train_end;
            let val_end = train_end + validation_size;

            let train_df = df.slice(0, train_end);
            let val_df = df.slice(val_start as i64, validation_size);

            // Generate sequences for this window
            let train_sequences = self
                .sequence_generator
                .generate_training_sequences(
                    train_df,
                    &config.horizons,
                    &config.model,
                    &config.data,
                )
                .await?;

            let val_sequences = self
                .sequence_generator
                .generate_training_sequences(val_df, &config.horizons, &config.model, &config.data)
                .await?;

            windows.push(TrainingWindow {
                train_data: train_sequences,
                val_data: val_sequences,
                window_id: windows.len(),
                train_samples: train_end,
                val_samples: validation_size,
            });

            log::info!(
                "📊 Window {}: Train[0-{}] ({} samples) → Val[{}-{}] ({} samples) | Chronological Split",
                windows.len(),
                train_end,
                train_end,
                val_start,
                val_end,
                validation_size
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

            // Apply SAME preprocessing as training
            let df = self
                .preprocessor
                .process_for_training(
                    raw_data,
                    &training_config.data,
                    Some(&training_config.features),
                )
                .await?;

            // Apply SAME feature engineering as training
            let feature_engineer =
                crate::features::FeatureEngineer::new(training_config.features.clone());
            let mut df = feature_engineer.generate_features(df).await?;

            // CRITICAL: Use SAME logic as training - remove initial NaN rows
            // This removes the initial rows where technical indicators are NaN (e.g., first 200 rows)
            // But does NOT validate the entire dataset like training does
            let original_len = df.height();

            // Find the first row where ALL columns have valid values (same as training)
            let mut first_valid_row = None;
            for row_idx in 0..original_len {
                let mut all_valid = true;
                for series in df.get_columns() {
                    if series.dtype().is_numeric() {
                        if let Ok(float_series) = series.f64() {
                            if let Some(value) = float_series.get(row_idx) {
                                if !value.is_finite() {
                                    all_valid = false;
                                    break;
                                }
                            } else {
                                all_valid = false;
                                break;
                            }
                        }
                    }
                }
                if all_valid {
                    first_valid_row = Some(row_idx);
                    break;
                }
            }

            let first_valid_idx = match first_valid_row {
                Some(idx) => idx,
                None => {
                    return Err(crate::utils::error::VangaError::DataError(
                        "No rows found with all valid (non-NaN) values after feature engineering."
                            .to_string(),
                    ));
                }
            };

            // Remove initial NaN rows (same as training)
            let clean_data_len = original_len - first_valid_idx;
            df = df.slice(first_valid_idx as i64, clean_data_len);

            log::info!(
                "Removed {} initial NaN rows, {} clean rows remaining",
                first_valid_idx,
                clean_data_len
            );

            // Calculate required rows for prediction
            let sequence_length = match &training_config.model.sequence_length {
                crate::config::model::SequenceLengthConfig::Fixed(len) => *len as usize,
                crate::config::model::SequenceLengthConfig::Auto { min_length, .. } => {
                    *min_length as usize
                }
                crate::config::model::SequenceLengthConfig::Adaptive => 60,
            };

            let required_rows = sequence_length + 1; // Only need sequence + buffer after NaN removal

            // Validate we have enough clean data
            if df.height() < required_rows {
                return Err(crate::utils::error::VangaError::DataError(format!(
                    "Insufficient clean data for prediction: {} rows available, {} required",
                    df.height(),
                    required_rows
                )));
            }

            // Take most recent data for prediction
            let start_idx = df.height() - required_rows;
            df = df.slice(start_idx as i64, required_rows);

            log::info!("Using most recent {} rows for prediction", required_rows);

            log::info!("🔍 PREDICTION DATA PROCESSING:");
            log::info!("   • Original dataset: {} rows", original_len);
            log::info!("   • Removed initial NaN rows: {} rows", first_valid_idx);
            log::info!("   • Clean data available: {} rows", clean_data_len);
            log::info!("   • Using most recent: {} rows", required_rows);
            log::info!("   • Sequence length: {} periods", sequence_length);
            log::info!("   ✅ Using same logic as training (no extra validation)");

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
#[derive(Debug, Clone)]
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
}
