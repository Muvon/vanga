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

            log::debug!(
                "📊 Window {}: Train[0-{}] → Val[{}-{}]",
                windows.len(),
                train_end,
                val_start,
                val_end
            );

            // Move window forward by validation_size (no overlap in test sets)
            train_end = val_end;
        }

        log::info!(
            "📊 Created {} walk-forward windows (validation_split: {:.1}%)",
            windows.len(),
            config.training.validation_split * 100.0
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

        // Preprocess data using training statistics
        let processed_data = self
            .preprocessor
            .process_for_prediction(raw_data, &config.symbols[0])
            .await?;

        // Generate prediction sequences (use default model config for now)
        let default_model_config = crate::config::ModelConfig::default();
        let sequences = self
            .sequence_generator
            .generate_prediction_sequences(
                processed_data,
                &config.symbols[0],
                &default_model_config,
            )
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
