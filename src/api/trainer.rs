// Model trainer
use crate::config::TrainingConfig;
use crate::data::DataPipeline;
use crate::model::lstm_simple::LSTMModel;
use crate::targets::TargetGenerator;
use crate::utils::error::Result;

pub struct ModelTrainer {
    config: TrainingConfig,
}

impl ModelTrainer {
    pub fn new(config: TrainingConfig) -> Self {
        Self { config }
    }

    pub async fn train(&self) -> Result<LSTMModel> {
        log::info!("Starting model training for symbol: {}", self.config.symbol);

        // Initialize data pipeline
        let data_pipeline = DataPipeline::new();

        // Load and prepare training data
        log::info!(
            "Loading training data from: {}",
            self.config.data_path.display()
        );
        let prepared_data = data_pipeline
            .prepare_training_data(&self.config.data_path, &self.config)
            .await?;

        log::info!(
            "Training data prepared: {} sequences, {} features",
            prepared_data.sequences.shape()[0],
            prepared_data.sequences.shape()[2]
        );

        // Generate targets
        let target_generator = TargetGenerator::with_defaults();
        let df = crate::data::loader::DataLoader::new()
            .load_csv(&self.config.data_path)
            .await?;
        let targets = target_generator.generate_all_targets(&df).await?;

        // Create LSTM model
        let input_size = prepared_data.sequences.shape()[2]; // Number of features
        let mut model = LSTMModel::from_model_config(&self.config.model_config, input_size)?;

        // Train the model
        log::info!("Starting LSTM training...");

        // For now, use price level targets as the main training target
        if let Some(price_targets) = targets.price_levels.get("1h") {
            // Convert targets to the format expected by LSTM (batch, output_size)
            let target_array = ndarray::Array2::from_shape_vec(
                (price_targets.len(), 1),
                price_targets.iter().map(|&x| x as f64).collect(),
            )
            .map_err(|e| {
                crate::utils::error::VangaError::DataError(format!(
                    "Target conversion error: {}",
                    e
                ))
            })?;

            model.train(&prepared_data.sequences, &target_array).await?;
        } else {
            return Err(crate::utils::error::VangaError::ModelError(
                "No price level targets available for training".to_string(),
            ));
        }

        log::info!("Model training completed successfully");
        Ok(model)
    }
}

/// High-level training function
pub async fn train_model(config: TrainingConfig) -> Result<LSTMModel> {
    let trainer = ModelTrainer::new(config);
    trainer.train().await
}
