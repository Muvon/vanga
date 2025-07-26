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
            ModelWrapper::Single(_model) => {
                // Single models default to 1h
                vec!["1h".to_string()]
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

        // Generate targets for confidence calculation (before cleanup) - make optional
        let targets_config = match self.generate_targets_for_confidence(&prepared_data).await {
            Ok(config) => Some(config),
            Err(e) => {
                log::warn!("Could not generate targets for confidence calculation: {}. Predictions will proceed without confidence scores.", e);
                None
            }
        };

        // Extract sequence data for order generation before cleanup - REQUIRED!
        let sequence_ohlc = prepared_data.sequence_ohlc.clone()
            .ok_or_else(|| VangaError::PredictionError(
                "FATAL: No sequence OHLC data available for order generation. This is required for proper ATR calculation and sequence-aware orders.".to_string()
            ))?;
        
        let sequence_prices: Vec<f64> = sequence_ohlc.iter().map(|row| row.close).collect();

        log::info!("✅ Sequence OHLC data loaded: {} rows for order generation", sequence_ohlc.len());
        log::debug!("Sequence price range: {:.2} - {:.2}", 
            sequence_prices.iter().fold(f64::INFINITY, |a, &b| a.min(b)),
            sequence_prices.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b))
        );

        // Explicit memory cleanup after prediction and data extraction
        drop(prepared_data);

        // Format predictions using output formatter with 5-class system
        let mut formatter = OutputFormatter::new(self.config.output_config.clone());

        // Pass sequence data to formatter for order generation - REQUIRED!
        formatter = formatter.with_sequence_data(sequence_prices);

        // Pass metadata to formatter for accurate PredictionResult creation
        formatter = formatter.with_metadata(input_feature_count, sequence_length);

        // Configure formatter with model's output heads for proper 5-class parsing
        let output_heads = if let Some(training_config) = model.get_training_config() {
            // Use output heads from model's training configuration
            training_config.model.output_heads.clone()
        } else {
            // Fallback to default configuration for older models
            crate::config::model::OutputHeadsConfig {
                price_levels: crate::config::model::PriceLevelHead {
                    enabled: true,
                    bandwidth_size: Some(1.0),
                    distribution_type: crate::config::model::DistributionType::Categorical,
                },
                direction: crate::config::model::DirectionHead {
                    enabled: true,
                    bandwidth_size: Some(0.8),
                    base_threshold_factor: 0.5,
                    extreme_multiplier: 2.5,
                },
                volatility: crate::config::model::VolatilityHead {
                    enabled: true,
                    bandwidth_size: Some(1.2),
                    base_percentiles: [0.20, 0.40, 0.60, 0.80],
                },
            }
        };

        formatter = formatter.with_output_heads(output_heads);

        // TODO: Add sequence data for sequence-aware price level calculations
        // formatter = formatter.with_sequence_data(sequence_data);

        // Determine horizon using smart selection logic
        let horizon = if let Some(requested_horizon) = &self.config.horizon {
            // Validate requested horizon against trained horizons
            let trained_horizons = model.get_trained_horizons();
            if !trained_horizons.contains(requested_horizon) {
                log::warn!(
                    "Requested horizon '{}' was not trained. Available horizons: {:?}. Using first available horizon.",
                    requested_horizon, trained_horizons
                );
                trained_horizons.first().unwrap_or(&"1h".to_string()).clone()
            } else {
                requested_horizon.clone()
            }
        } else {
            // Use first trained horizon or default to 1h
            let trained_horizons = model.get_trained_horizons();
            trained_horizons.first().unwrap_or(&"1h".to_string()).clone()
        };

        // Generate targets for the prediction data to enable confidence calculation
        let formatted_predictions = formatter.format_predictions(
            &raw_predictions,
            &self.config.symbols[0],
            &horizon,
            current_price,
            targets_config.as_ref(), // Pass targets config if available for confidence calculation
        )?;

        // Apply post-processing if configured
        let post_processor = PostProcessor::new(self.config.post_processing.clone());
        let final_predictions = if self.config.min_confidence > 0.0 {
            log::debug!(
                "Applying confidence threshold: {}",
                self.config.min_confidence
            );
            let processed = post_processor.process(formatted_predictions)?;
            post_processor.filter_by_confidence(processed, self.config.min_confidence)
        } else {
            post_processor.process(formatted_predictions)?
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
                    log::debug!("Extracted current price from OHLC data: {:.2}", current_price);
                    return Ok(current_price);
                }
            }
        }

        // FATAL ERROR: No OHLC data available
        Err(crate::utils::error::VangaError::DataError(
            "OHLC data is required for price extraction but not available. Cannot use normalized tensor data for price calculations.".to_string(),
        ))
    }

    /// Generate targets from prediction data for confidence calculation
    async fn generate_targets_for_confidence(
        &self,
        prepared_data: &PreparedPredictionData,
    ) -> Result<crate::targets::PreparedTargets> {
        // Create a minimal DataFrame from the prepared data for target generation
        // This allows us to calculate target statistics for confidence assessment

        // Extract the most recent data point for target generation
        let last_sequence_idx = prepared_data.sequences.shape()[0].saturating_sub(1);
        let sequence = prepared_data
            .sequences
            .slice(ndarray::s![last_sequence_idx, .., ..]);

        // Find close price feature (assuming it's one of the features)
        let close_feature_idx = prepared_data
            .feature_names
            .iter()
            .position(|name| name.contains("close"))
            .unwrap_or(0); // Default to first feature if close not found

        // Extract close prices from the sequence
        let close_prices: Vec<f64> = sequence.column(close_feature_idx).to_vec();

        // Create a minimal DataFrame for target generation
        let close_series = polars::prelude::Series::from_iter(close_prices.iter().cloned());
        let df = polars::prelude::DataFrame::new(vec![close_series.with_name("close")]).map_err(
            |e| {
                crate::utils::error::VangaError::DataError(format!(
                    "Failed to create DataFrame for target generation: {}",
                    e
                ))
            },
        )?;

        // Generate targets using the default configuration
        let target_generator = crate::targets::TargetGenerator::with_defaults();
        target_generator.generate_all_targets(&df, None).await
    }
}

/// High-level prediction function for single-target models
pub async fn predict(config: PredictionConfig, model: &LSTMModel) -> Result<Vec<PredictionResult>> {
    let predictor = Predictor::new(config);
    predictor.predict(ModelWrapper::Single(model)).await
}


