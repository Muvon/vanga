// Model predictor
use crate::config::PredictionConfig;
use crate::data::DataPipeline;
use crate::model::lstm_simple::LSTMModel;
use crate::utils::error::Result;
use ndarray::Array2;

pub struct Predictor {
    config: PredictionConfig,
}

impl Predictor {
    pub fn new(config: PredictionConfig) -> Self {
        Self { config }
    }

    pub async fn predict(&self, model: &LSTMModel) -> Result<Array2<f64>> {
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
        let predictions = model.predict(&prepared_data.sequences).await?;

        log::info!("Generated {} predictions", predictions.nrows());

        // Apply post-processing if configured
        let final_predictions = if self.config.min_confidence > 0.0 {
            // Filter predictions by confidence (simplified - just return all for now)
            log::debug!(
                "Applying confidence threshold: {}",
                self.config.min_confidence
            );
            predictions
        } else {
            predictions
        };

        log::info!("Prediction completed successfully");
        Ok(final_predictions)
    }
}

/// High-level prediction function
pub async fn predict(config: PredictionConfig, model: &LSTMModel) -> Result<Array2<f64>> {
    let predictor = Predictor::new(config);
    predictor.predict(model).await
}
