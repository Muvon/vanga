// Multi-target prediction API
use crate::config::PredictionConfig;
use crate::data::DataPipeline;
use crate::model::multi_target::MultiTargetLSTMModel;
use crate::output::{OutputFormatter, PredictionResult};
use crate::utils::error::{Result, VangaError};
use ndarray::Array2;

/// Multi-target predictor for cryptocurrency forecasting
pub struct MultiTargetPredictor {
    config: PredictionConfig,
}

impl MultiTargetPredictor {
    /// Create new multi-target predictor
    pub fn new(config: PredictionConfig) -> Self {
        Self { config }
    }

    /// Make predictions using trained multi-target model
    pub async fn predict(&self, model: &MultiTargetLSTMModel) -> Result<MultiTargetPredictions> {
        log::info!(
            "Starting multi-target prediction for symbol: {}",
            &self.config.symbols[0]
        );

        // Initialize data pipeline
        let data_pipeline = DataPipeline::new();

        // Load and prepare prediction data
        log::info!(
            "Loading prediction data from: {}",
            self.config.input_path.display()
        );
        let prepared_data = data_pipeline
            .prepare_prediction_data(&self.config.input_path, &self.config)
            .await?;

        // Extract current price from the input data (last close price)
        let current_price = self.extract_current_price(&self.config.input_path).await?;

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

        // Make predictions using multi-target model
        log::info!(
            "Making predictions using multi-target model with {} targets",
            model.get_num_targets()
        );
        let raw_predictions = model.predict(&prepared_data.sequences).await?;

        // Format predictions with target names and metadata
        let predictions = MultiTargetPredictions::new_with_metadata(
            raw_predictions,
            model.get_target_names().to_vec(),
            self.config.symbols[0].clone(),
            current_price,
            input_feature_count,
            sequence_length,
        );

        log::info!("✅ Multi-target predictions completed successfully");
        Ok(predictions)
    }

    /// Extract current price from input data (latest close price)
    async fn extract_current_price<P: AsRef<std::path::Path>>(&self, data_path: P) -> Result<f64> {
        use crate::data::DataLoader;

        let loader = DataLoader::new();
        let df = loader.load_csv(data_path).await?;

        // Get the last close price
        let close_series = df
            .column("close")
            .map_err(|e| VangaError::DataError(format!("Failed to get close column: {}", e)))?;

        let close_values = close_series
            .f64()
            .map_err(|e| VangaError::DataError(format!("Failed to convert close to f64: {}", e)))?;

        // Get the last non-null value using to_vec()
        let close_vec = close_values.to_vec();
        let current_price = close_vec
            .iter()
            .rev()
            .filter_map(|v| *v)
            .next()
            .ok_or_else(|| {
                VangaError::DataError("No valid close price found in data".to_string())
            })?;

        log::info!("Extracted current price: ${:.2}", current_price);
        Ok(current_price)
    }
}

/// Multi-target prediction results
#[derive(Debug, Clone)]
pub struct MultiTargetPredictions {
    /// Raw prediction values [samples, targets]
    pub predictions: Array2<f64>,
    /// Target names corresponding to prediction columns
    pub target_names: Vec<String>,
    /// Symbol these predictions are for
    pub symbol: String,
    /// Current price at prediction time
    pub current_price: f64,
    /// Number of input features used
    pub input_feature_count: usize,
    /// Sequence length used for prediction
    pub sequence_length: usize,
}

impl MultiTargetPredictions {
    /// Create new multi-target predictions
    pub fn new(
        predictions: Array2<f64>,
        target_names: Vec<String>,
        symbol: String,
        current_price: f64,
    ) -> Self {
        Self {
            predictions,
            target_names,
            symbol,
            current_price,
            input_feature_count: 0,
            sequence_length: 0,
        }
    }

    /// Create new multi-target predictions with metadata
    pub fn new_with_metadata(
        predictions: Array2<f64>,
        target_names: Vec<String>,
        symbol: String,
        current_price: f64,
        input_feature_count: usize,
        sequence_length: usize,
    ) -> Self {
        Self {
            predictions,
            target_names,
            symbol,
            current_price,
            input_feature_count,
            sequence_length,
        }
    }

    /// Get predictions for a specific target
    pub fn get_target_predictions(&self, target_name: &str) -> Option<Vec<f64>> {
        self.target_names
            .iter()
            .position(|name| name == target_name)
            .map(|target_idx| self.predictions.column(target_idx).to_vec())
    }

    /// Get all target names
    pub fn get_target_names(&self) -> &[String] {
        &self.target_names
    }

    /// Get number of samples
    pub fn num_samples(&self) -> usize {
        self.predictions.shape()[0]
    }

    /// Get number of targets
    pub fn num_targets(&self) -> usize {
        self.predictions.shape()[1]
    }

    /// Get prediction for specific sample and target
    pub fn get_prediction(&self, sample_idx: usize, target_name: &str) -> Option<f64> {
        if let Some(target_idx) = self
            .target_names
            .iter()
            .position(|name| name == target_name)
        {
            if sample_idx < self.predictions.shape()[0] {
                Some(self.predictions[[sample_idx, target_idx]])
            } else {
                None
            }
        } else {
            None
        }
    }

    /// Convert raw predictions to structured format using OutputFormatter
    pub async fn to_structured_predictions(
        &self,
        config: &PredictionConfig,
        model: &MultiTargetLSTMModel,
    ) -> Result<Vec<PredictionResult>> {
        log::info!("Converting raw predictions to structured format");

        // Create output formatter with config
        let mut formatter = OutputFormatter::new(config.output_config.clone());

        // Get output heads configuration from stored model config
        let output_heads = model.get_training_config()
            .ok_or_else(|| {
                crate::utils::error::VangaError::ConfigError(
                    "Training configuration not found in model. Model may be corrupted or from an older version.".to_string()
                )
            })?
            .model.output_heads.clone();

        formatter = formatter.with_output_heads(output_heads);

        // Smart horizon selection logic
        let horizons_to_predict = if let Some(requested_horizon) = &config.horizon {
            // Validate requested horizon against trained horizons
            let trained_horizons = model.get_trained_horizons();
            if !trained_horizons.contains(requested_horizon) {
                return Err(crate::utils::error::VangaError::ConfigError(format!(
                    "Requested horizon '{}' was not trained. Available horizons: {:?}",
                    requested_horizon, trained_horizons
                )));
            }
            vec![requested_horizon.clone()]
        } else if config.all_horizons {
            // For all_horizons, use all trained horizons
            let trained_horizons = model.get_trained_horizons();
            if trained_horizons.is_empty() {
                vec!["1h".to_string()] // Fallback for backward compatibility
            } else {
                trained_horizons.to_vec()
            }
        } else {
            // No specific horizon requested - use first trained horizon
            let trained_horizons = model.get_trained_horizons();
            if trained_horizons.is_empty() {
                vec!["1h".to_string()] // Fallback for backward compatibility
            } else {
                vec![trained_horizons[0].clone()]
            }
        };

        log::info!(
            "Generating predictions for horizons: {:?}",
            horizons_to_predict
        );

        // Create structured predictions with correct metadata for each horizon
        let mut results = Vec::new();

        // Generate predictions for each requested horizon
        for horizon in &horizons_to_predict {
            for batch_idx in 0..self.predictions.nrows() {
                let mut result = PredictionResult::new_with_metadata(
                    self.symbol.clone(),
                    horizon.clone(),
                    self.current_price,
                    self.input_feature_count,
                    self.sequence_length,
                );

                // Extract predictions for this batch
                let batch_predictions = self.predictions.row(batch_idx);

                // Use the formatter's MultiTargetParser
                if formatter.has_parser() {
                    // Parse the raw predictions using the multi-target parser
                    let parsed_output = formatter.parse_raw_predictions(batch_predictions)?;

                    // Convert parsed output to structured predictions
                    if let Some(price_level_probs) = parsed_output.price_levels {
                        result =
                            result.with_price_levels(formatter.create_price_level_prediction(
                                &price_level_probs,
                                self.current_price,
                                formatter.get_sequence_data(),
                                formatter.get_bandwidth_size(),
                            )?);
                    }

                    if let Some(direction_output) = parsed_output.direction {
                        result = result.with_direction(formatter.create_direction_prediction(
                            &direction_output,
                            Some(horizon),
                            Some(60),  // Default sequence length
                            Some(3.0), // Default bandwidth percent
                        )?);
                    }

                    if let Some(volatility_output) = parsed_output.volatility {
                        result = result.with_volatility(formatter.create_volatility_prediction(
                            &volatility_output,
                            Some(horizon),
                            Some(3.0),  // Default bandwidth percent
                            Some(50.0), // Default volatility percentile
                        )?);
                    }
                } else {
                    return Err(VangaError::PredictionError(
                        "MultiTargetParser not configured. All predictions must use 5-class system.".to_string()
                    ));
                }

                result = result.with_confidence(0.7); // Default confidence
                results.push(result);
            }
        }

        log::info!(
            "✅ Successfully converted {} raw predictions to structured format for {} horizon(s): {:?}",
            results.len(),
            horizons_to_predict.len(),
            horizons_to_predict
        );

        Ok(results)
    }

    /// Get all predictions for a specific sample
    pub fn get_sample_predictions(&self, sample_idx: usize) -> Option<Vec<(String, f64)>> {
        if sample_idx < self.predictions.shape()[0] {
            Some(
                self.target_names
                    .iter()
                    .enumerate()
                    .map(|(target_idx, name)| {
                        (name.clone(), self.predictions[[sample_idx, target_idx]])
                    })
                    .collect(),
            )
        } else {
            None
        }
    }

    /// Format predictions as human-readable string
    pub fn format_predictions(&self, sample_idx: usize) -> String {
        if let Some(sample_preds) = self.get_sample_predictions(sample_idx) {
            let mut result = format!("Predictions for {} (sample {}):\n", self.symbol, sample_idx);
            for (target_name, value) in sample_preds {
                result.push_str(&format!("  {}: {:.4}\n", target_name, value));
            }
            result
        } else {
            format!("Invalid sample index: {}", sample_idx)
        }
    }
}

/// High-level prediction function for multi-target models
pub async fn predict_multi_target(
    config: PredictionConfig,
    model: &MultiTargetLSTMModel,
) -> Result<MultiTargetPredictions> {
    let predictor = MultiTargetPredictor::new(config);
    predictor.predict(model).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::prediction::OutputConfig;
    use crate::output::OutputFormatter;

    #[test]
    fn test_output_formatter_has_parser_after_configuration() {
        // This test verifies the fix for "MultiTargetParser not configured" error

        // Create output formatter with config (same as multi_target_predictor.rs line 258)
        let output_config = OutputConfig::default();
        let mut formatter = OutputFormatter::new(output_config);

        // Initially, formatter should not have parser
        assert!(
            !formatter.has_parser(),
            "Formatter should not have parser initially"
        );

        // Configure output heads (same as the fix we applied)
        let output_heads = crate::config::model::OutputHeadsConfig {
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
        };

        formatter = formatter.with_output_heads(output_heads);

        // After configuration, formatter should have parser
        assert!(
            formatter.has_parser(),
            "Formatter should have parser after with_output_heads()"
        );
    }

    #[test]
    fn test_multi_target_predictor_creation() {
        // Test that MultiTargetPredictor can be created without errors
        let prediction_config = PredictionConfig::default();
        let _predictor = MultiTargetPredictor::new(prediction_config);

        // If we reach here without panicking, the test passes
    }
}
