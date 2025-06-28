// Model predictor
use crate::config::PredictionConfig;
use crate::data::DataPipeline;
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
        log::info!("Starting prediction for symbol: {}", self.config.symbol);

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

        log::info!(
            "Prediction data prepared: {} sequences, {} features",
            prepared_data.sequences.shape()[0],
            prepared_data.sequences.shape()[2]
        );

        // Make predictions
        log::info!("Generating predictions...");
        let raw_predictions = model.predict(&prepared_data.sequences).await?;

        log::info!("Generated {} predictions", raw_predictions.nrows());

        // Format predictions using output formatter
        let formatter = OutputFormatter::new(self.config.output_config.clone());

        // Get current price (simplified - use last close price)
        let current_price = 50000.0; // TODO: Extract from actual data

        // Determine horizon
        let horizon = self
            .config
            .horizon
            .clone()
            .unwrap_or_else(|| "1h".to_string());

        let formatted_predictions = formatter.format_predictions(
            &raw_predictions,
            &self.config.symbol,
            &horizon,
            current_price,
            None, // TODO: Pass actual targets config
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
}

/// High-level prediction function
pub async fn predict(config: PredictionConfig, model: &LSTMModel) -> Result<Vec<PredictionResult>> {
    let predictor = Predictor::new(config);
    predictor.predict(model).await
}
