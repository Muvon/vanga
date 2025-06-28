// Model trainer
use crate::config::TrainingConfig;
use crate::data::{DataPipeline, TargetConverter};
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

        // Create target converter for multi-target training
        let target_converter = TargetConverter::new(self.config.model_config.output_heads.clone());

        // Validate targets are compatible with output configuration
        target_converter.validate_targets(&targets)?;

        // Convert targets to training array format
        let training_targets =
            target_converter.convert_to_training_array(&targets, &targets.valid_indices)?;

        log::info!(
            "Training targets prepared: {} samples x {} outputs",
            training_targets.shape()[0],
            training_targets.shape()[1]
        );

        // Train the LSTM model with multi-target outputs
        model
            .train(&prepared_data.sequences, &training_targets)
            .await?;

        log::info!("Model training completed successfully");
        Ok(model)
    }
}

/// High-level training function
pub async fn train_model(config: TrainingConfig) -> Result<LSTMModel> {
    let trainer = ModelTrainer::new(config);
    trainer.train().await
}
