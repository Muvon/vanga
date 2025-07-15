// Model predictor
use crate::config::PredictionConfig;
use crate::data::{DataPipeline, PreparedPredictionData};
use crate::model::lstm_simple::LSTMModel;
use crate::output::{OutputFormatter, PostProcessor, PredictionResult};
use crate::utils::error::Result;

pub struct Predictor {
    config: PredictionConfig,
}

impl Predictor {
    pub fn new(config: PredictionConfig) -> Self {
        Self { config }
    }

    pub async fn predict(&self, model: &LSTMModel) -> Result<Vec<PredictionResult>> {
        let symbol = &self.config.symbols[0]; // Use first symbol for single-symbol prediction
        log::info!("Starting prediction for symbol: {}", symbol);

        // Initialize device from configuration
        let device_string = self.config.device.to_device_string();
        let device = crate::utils::device::DeviceManager::new(&device_string)?;
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

        // Extract current price from the most recent data point (before cleanup)
        let current_price = self.extract_current_price_from_data(&prepared_data)?;

        // Make predictions
        log::info!("Generating predictions...");
        let raw_predictions = model.predict(&prepared_data.sequences).await?;

        log::info!("Generated {} predictions", raw_predictions.nrows());

        // Generate targets for confidence calculation (before cleanup)
        let targets_config = self.generate_targets_for_confidence(&prepared_data).await?;

        // Explicit memory cleanup after prediction and data extraction
        drop(prepared_data);

        // Format predictions using output formatter
        let formatter = OutputFormatter::new(self.config.output_config.clone());

        // Determine horizon
        let horizon = self
            .config
            .horizon
            .clone()
            .unwrap_or_else(|| "1h".to_string());

        // Generate targets for the prediction data to enable confidence calculation
        let formatted_predictions = formatter.format_predictions(
            &raw_predictions,
            &self.config.symbols[0],
            &horizon,
            current_price,
            Some(&targets_config), // Pass actual targets config for confidence calculation
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
        // The last sequence contains the most recent data
        // Assuming 'close' price is the last feature in the sequence
        let last_sequence_idx = data.sequences.shape()[0] - 1;
        let last_time_step = data.sequences.shape()[1] - 1;
        let close_price_idx = data.sequences.shape()[2] - 1; // Assuming close is last feature

        let current_price = data.sequences[[last_sequence_idx, last_time_step, close_price_idx]];

        if current_price <= 0.0 {
            return Err(crate::utils::error::VangaError::DataError(
                "Invalid current price extracted from data".to_string(),
            ));
        }

        log::debug!("Extracted current price: {:.2}", current_price);
        Ok(current_price)
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
        target_generator.generate_all_targets(&df).await
    }
}

/// High-level prediction function
pub async fn predict(config: PredictionConfig, model: &LSTMModel) -> Result<Vec<PredictionResult>> {
    let predictor = Predictor::new(config);
    predictor.predict(model).await
}
