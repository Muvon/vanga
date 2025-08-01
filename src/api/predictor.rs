// Unified model predictor - handles both single and multi-target models
use crate::config::PredictionConfig;
use crate::data::{DataPipeline, PreparedPredictionData};
use crate::model::lstm_simple::LSTMModel;
use crate::model::multi_target::MultiTargetLSTMModel;
use crate::output::{OutputFormatter, PostProcessor, PredictionResult};
use crate::utils::error::{Result, VangaError};
use ndarray::Array2;

/// Unified model wrapper for both single and multi-target models
pub enum ModelWrapper<'a> {
    Single(&'a LSTMModel),
    MultiTarget(&'a MultiTargetLSTMModel),
}

impl<'a> ModelWrapper<'a> {
    /// Make predictions using the wrapped model
    pub async fn predict(&self, sequences: &ndarray::Array3<f64>) -> Result<Array2<f64>> {
        match self {
            ModelWrapper::Single(model) => model.predict(sequences).await,
            ModelWrapper::MultiTarget(model) => model.predict(sequences).await,
        }
    }

    /// Get input size expected by the model
    pub fn get_input_size(&self) -> usize {
        match self {
            ModelWrapper::Single(model) => model.get_input_size(),
            ModelWrapper::MultiTarget(model) => model.get_input_size(),
        }
    }

    /// Get training configuration if available
    pub fn get_training_config(&self) -> Option<&crate::config::TrainingConfig> {
        match self {
            ModelWrapper::Single(_model) => {
                // Single models don't store training config yet
                None
            }
            ModelWrapper::MultiTarget(model) => model.get_training_config(),
        }
    }

    /// Get trained horizons
    pub fn get_trained_horizons(&self) -> Vec<String> {
        match self {
            ModelWrapper::Single(model) => {
                // Try to get horizons from model's training config if available
                if let Some(training_config) = model.get_training_config() {
                    training_config.horizons.clone()
                } else {
                    // Fallback to 1h for models without stored config
                    vec!["1h".to_string()]
                }
            }
            ModelWrapper::MultiTarget(model) => model.get_trained_horizons().to_vec(),
        }
    }

    /// Get target names
    pub fn get_target_names(&self) -> Vec<String> {
        match self {
            ModelWrapper::Single(_model) => {
                // Single models have one unnamed target
                vec!["prediction".to_string()]
            }
            ModelWrapper::MultiTarget(model) => model.get_target_names().to_vec(),
        }
    }

    /// Check if this is a multi-target model
    pub fn is_multi_target(&self) -> bool {
        matches!(self, ModelWrapper::MultiTarget(_))
    }
}

pub struct Predictor {
    config: PredictionConfig,
}

impl Predictor {
    pub fn new(config: PredictionConfig) -> Self {
        Self { config }
    }

    /// Make predictions using either single or multi-target model
    pub async fn predict(&self, model: ModelWrapper<'_>) -> Result<Vec<PredictionResult>> {
        let symbol = &self.config.symbols[0]; // Use first symbol for single-symbol prediction
        log::info!("Starting prediction for symbol: {}", symbol);

        // Initialize device from configuration
        let device_string = self.config.device.to_device_string();
        let device = crate::utils::device::DeviceManager::create_device(&device_string)?;
        log::info!(
            "🔧 Using device: {} ({})",
            device_string,
            match device {
                candle_core::Device::Cpu => "CPU",
                candle_core::Device::Cuda(_) => "NVIDIA CUDA GPU",
                candle_core::Device::Metal(_) => "Apple Metal GPU",
            }
        );

        // Resolve data file path using the same logic as training
        let data_file_path = crate::utils::file_discovery::resolve_symbol_data_path(
            &self.config.input_path,
            symbol,
        )?;

        log::info!("📂 Using data file: {}", data_file_path.display());

        // Initialize data pipeline
        let data_pipeline = DataPipeline::new();

        // Load and prepare prediction data
        log::info!("Loading prediction data from: {}", data_file_path.display());
        let prepared_data = data_pipeline
            .prepare_prediction_data(&data_file_path, &self.config)
            .await?;

        log::info!(
            "Prediction data prepared: {} sequences, {} features",
            prepared_data.sequences.shape()[0],
            prepared_data.sequences.shape()[2]
        );

        // Capture metadata for later use
        let input_feature_count = prepared_data.sequences.shape()[2];
        let sequence_length = prepared_data.sequences.shape()[1];

        // Validate input compatibility with model
        let expected_features = model.get_input_size();
        let actual_features = prepared_data.sequences.shape()[2];

        if actual_features != expected_features {
            // Get training configuration from model for debugging
            let config_info = if let Some(training_config) = model.get_training_config() {
                format!(
                    "Technical indicators: {}, Custom features: {}",
                    training_config.features.technical_indicators.enabled,
                    training_config.features.custom_features.auto_include_all
                )
            } else {
                "No training config stored (old model format)".to_string()
            };

            log::error!("🚨 FEATURE MISMATCH DETECTED:");
            log::error!(
                "   Expected: {} features (from trained model)",
                expected_features
            );
            log::error!(
                "   Received: {} features (from current data)",
                actual_features
            );
            log::error!("   Config: {}", config_info);
            log::error!(
                "   Generated features: {}",
                prepared_data.feature_names.len()
            );

            // Show first 10 and last 10 features for debugging
            let feature_preview = if prepared_data.feature_names.len() > 20 {
                format!(
                    "First 10: {:?} ... Last 10: {:?}",
                    &prepared_data.feature_names[..10],
                    &prepared_data.feature_names[prepared_data.feature_names.len() - 10..]
                )
            } else {
                format!("All features: {:?}", prepared_data.feature_names)
            };
            log::error!("   {}", feature_preview);

            return Err(VangaError::ModelError(format!(
                "Feature count mismatch: model expects {} features but data has {}. This indicates inconsistent feature processing between training and prediction. Check logs above for details.",
                expected_features,
                actual_features
            )));
        }

        // Extract current price from the most recent data point (before cleanup)
        let current_price = self.extract_current_price_from_data(&prepared_data)?;

        // Make predictions
        log::info!("Generating predictions...");
        let raw_predictions = model.predict(&prepared_data.sequences).await?;

        log::info!("Generated {} predictions", raw_predictions.nrows());

        // Confidence calculation for predictions should use model uncertainty, not target generation
        // Target generation is inappropriate for prediction data as it requires full historical context
        // Extract sequence data for order generation before cleanup - REQUIRED!
        let sequence_ohlc = prepared_data.sequence_ohlc.clone()
            .ok_or_else(|| VangaError::PredictionError(
                "FATAL: No sequence OHLC data available for order generation. This is required for proper ATR calculation and sequence-aware orders.".to_string()
            ))?;

        // Extract OHLCV data for VWAP-based range calculation (matches training approach)
        let sequence_ohlcv = sequence_ohlc.clone();

        log::info!(
            "✅ Sequence OHLCV data loaded: {} rows for VWAP-based order generation",
            sequence_ohlcv.len()
        );

        // Also extract close prices for backward compatibility and logging
        let sequence_prices: Vec<f64> = sequence_ohlcv.iter().map(|row| row.close).collect();
        log::debug!(
            "Sequence price range: {:.2} - {:.2}",
            sequence_prices.iter().fold(f64::INFINITY, |a, &b| a.min(b)),
            sequence_prices
                .iter()
                .fold(f64::NEG_INFINITY, |a, &b| a.max(b))
        );

        // Explicit memory cleanup after prediction and data extraction
        drop(prepared_data);

        // Format predictions using output formatter with 5-class system
        let mut formatter = OutputFormatter::new(self.config.output_config.clone());

        // Pass OHLCV data to formatter for VWAP-based range calculation - REQUIRED!
        formatter = formatter.with_sequence_ohlcv(sequence_ohlcv);

        // Pass metadata to formatter for accurate PredictionResult creation
        formatter = formatter.with_metadata(input_feature_count, sequence_length);

        // Configure formatter with model's targets config for proper 5-class parsing
        let targets_config = if let Some(training_config) = model.get_training_config() {
            // Use targets config from model's training configuration
            log::info!("✅ Using training configuration for prediction parameters");
            Some(training_config.model.targets.clone())
        } else {
            // Fallback to default TargetsConfig - all targets are always enabled with NUM_CLASSES=5
            log::warn!("⚠️  No training configuration available, using fallback defaults. This may cause prediction inconsistency with training.");
            None // Let reconstruction methods use their defaults
        };

        // Pass training config to formatter for enhanced reconstruction
        if let Some(ref config) = targets_config {
            formatter = formatter.with_training_config(config.clone());
            log::debug!(
                "🔧 Training config for reconstruction: base_sensitivity={:.3}, extreme_multiplier={:.1}",
                config.base_sensitivity,
                config.extreme_multiplier
            );
        }

        // Determine horizons to process based on configuration
        let horizons_to_process = if self.config.all_horizons {
            // Process ALL trained horizons
            let trained_horizons = model.get_trained_horizons();
            log::info!(
                "Processing all {} trained horizons: {:?}",
                trained_horizons.len(),
                trained_horizons
            );
            trained_horizons.to_vec()
        } else if let Some(requested_horizon) = &self.config.horizon {
            // Process SPECIFIC horizon (with strict validation)
            let trained_horizons = model.get_trained_horizons();
            if !trained_horizons.contains(requested_horizon) {
                return Err(VangaError::ConfigError(format!(
                    "Requested horizon '{}' was not trained. Available horizons: {:?}. Use one of the available horizons or --all-horizons to predict all.",
                    requested_horizon, trained_horizons
                )));
            }
            log::info!(
                "Processing specific requested horizon: {}",
                requested_horizon
            );
            vec![requested_horizon.clone()]
        } else {
            // Process FIRST horizon (default behavior for backward compatibility)
            let trained_horizons = model.get_trained_horizons();
            let default_horizon = trained_horizons
                .first()
                .unwrap_or(&"1h".to_string())
                .clone();
            log::info!(
                "No horizon specified, using primary horizon: {}",
                default_horizon
            );
            vec![default_horizon]
        };

        // Process each horizon and collect all predictions
        let mut all_predictions = Vec::new();

        for horizon in horizons_to_process {
            log::info!("🎯 Processing predictions for horizon: {}", horizon);

            // Generate predictions for this specific horizon
            let formatted_predictions = formatter.format_predictions(
                &raw_predictions,
                &self.config.symbols[0],
                &horizon,
                current_price,
                None, // No prepared targets available during prediction - use model uncertainty instead
            )?;

            log::info!(
                "Generated {} predictions for horizon: {}",
                formatted_predictions.len(),
                horizon
            );
            all_predictions.extend(formatted_predictions);
        }

        log::info!(
            "Total predictions generated across all horizons: {}",
            all_predictions.len()
        );

        // Apply post-processing if configured
        let post_processor = PostProcessor::new(self.config.post_processing.clone());
        let final_predictions = if self.config.min_confidence > 0.0 {
            log::info!(
                "Applying confidence threshold: {} (predictions with confidence below this will be filtered out)",
                self.config.min_confidence
            );
            let processed = post_processor.process(all_predictions)?;

            // Log confidence values before filtering (promote to INFO level for debugging)
            for (i, pred) in processed.iter().enumerate() {
                log::info!(
                    "Prediction {} ({}): confidence = {:.3}",
                    i,
                    pred.horizon,
                    pred.confidence
                );
            }

            let processed_count = processed.len();
            let filtered =
                post_processor.filter_by_confidence(processed, self.config.min_confidence);

            // Warn if all predictions are filtered out
            if filtered.is_empty() && processed_count > 0 {
                log::warn!(
                    "⚠️  All {} predictions filtered out by confidence threshold {:.1}%. Consider lowering min_confidence or check model confidence calculation.",
                    processed_count,
                    self.config.min_confidence * 100.0
                );
            }

            log::info!(
                "Confidence filtering: {} predictions before, {} predictions after (threshold: {:.1}%)",
                processed_count,
                filtered.len(),
                self.config.min_confidence * 100.0
            );
            filtered
        } else {
            log::info!("No confidence threshold applied (min_confidence = 0.0)");
            post_processor.process(all_predictions)?
        };

        log::info!("Prediction completed successfully");
        Ok(final_predictions)
    }

    /// Extract current price from prepared prediction data
    fn extract_current_price_from_data(&self, data: &PreparedPredictionData) -> Result<f64> {
        // REQUIRE OHLC data - no fallback to normalized data
        if let Some(ref ohlc_data) = data.sequence_ohlc {
            if let Some(last_row) = ohlc_data.last() {
                let current_price = last_row.close;
                if current_price > 0.0 {
                    log::debug!(
                        "Extracted current price from OHLC data: {:.2}",
                        current_price
                    );
                    return Ok(current_price);
                }
            }
        }

        // FATAL ERROR: No OHLC data available
        Err(crate::utils::error::VangaError::DataError(
            "OHLC data is required for price extraction but not available. Cannot use normalized tensor data for price calculations.".to_string(),
        ))
    }
}

/// High-level prediction function for single-target models
pub async fn predict(config: PredictionConfig, model: &LSTMModel) -> Result<Vec<PredictionResult>> {
    let predictor = Predictor::new(config);
    predictor.predict(ModelWrapper::Single(model)).await
}
