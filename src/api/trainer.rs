// Model trainer
use crate::config::TrainingConfig;
use crate::data::{DataPipeline, TargetConverter};
use crate::model::lstm_simple::LSTMModel;
use crate::targets::TargetGenerator;
use crate::utils::error::{Result, VangaError};

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

        // Generate targets with training config horizons
        let target_config = crate::targets::MultiTargetConfig {
            price_level_config: crate::targets::PriceLevelConfig::default(),
            direction_config: crate::targets::DirectionConfig::default(),
            volatility_config: crate::targets::VolatilityConfig::default(),
            horizons: self.config.horizons.clone(),
        };
        let target_generator = TargetGenerator::new(target_config);
        let df = crate::data::loader::DataLoader::new()
            .load_csv(&self.config.data_path)
            .await?;
        let targets = target_generator.generate_all_targets(&df).await?;

        // Create target converter with synchronized horizons
        let mut output_heads = self.config.model_config.output_heads.clone();
        output_heads.volatility.horizons = self.config.horizons.clone();
        let target_converter = TargetConverter::new(output_heads);

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

        // Handle model creation or loading based on training configuration
        let input_size = prepared_data.sequences.shape()[2]; // Number of features
        let output_size = training_targets.shape()[1]; // Number of target outputs
        let model_path = format!("./models/{}_model.bin", self.config.symbol);
        let mut model = self
            .get_or_create_model(&model_path, input_size, output_size)
            .await?;

        // Train the LSTM model with multi-target outputs
        model
            .train(&prepared_data.sequences, &training_targets)
            .await?;

        log::info!("Model training completed successfully");
        Ok(model)
    }

    /// Get existing model or create new one based on training configuration
    async fn get_or_create_model(
        &self,
        model_path: &str,
        input_size: usize,
        output_size: usize,
    ) -> Result<LSTMModel> {
        let model_exists = std::path::Path::new(model_path).exists();

        match (self.config.fresh_training, self.config.continue_training, model_exists) {
            // Fresh training requested - always create new model
            (true, _, _) => {
                log::info!("Fresh training requested - creating new model");
                LSTMModel::from_model_config(&self.config.model_config, input_size, output_size)
            }

            // Continue training requested - must have existing model
            (false, true, false) => {
                Err(VangaError::ModelError(format!(
                    "Continue training requested but no existing model found for symbol: {}. Train without --continue-training first, or use --fresh to start new training.",
                    self.config.symbol
                )))
            }
            (false, true, true) => {
                log::info!("Continue training requested - loading existing model from: {}", model_path);
                let model = LSTMModel::load(model_path)?;

                // Validate model compatibility
                if model.get_input_size() != input_size {
                    return Err(VangaError::ConfigError(format!(
                        "Model input size mismatch: existing model expects {} features, but data has {} features. Use --fresh to retrain with new data structure.",
                        model.get_input_size(),
                        input_size
                    )));
                }

                Ok(model)
            }

            // Default behavior - continue if model exists, otherwise create new
            (false, false, true) => {
                log::info!("Existing model found - continuing training from: {}", model_path);
                let model = LSTMModel::load(model_path)?;

                // Check compatibility, fallback to fresh if incompatible
                if model.get_input_size() != input_size {
                    log::warn!(
                        "Model input size mismatch (expected {}, got {}). Creating fresh model.",
                        model.get_input_size(),
                        input_size
                    );
                    LSTMModel::from_model_config(&self.config.model_config, input_size, output_size)
                } else {
                    Ok(model)
                }
            }
            (false, false, false) => {
                log::info!("No existing model found - creating new model");
                LSTMModel::from_model_config(&self.config.model_config, input_size, output_size)
            }
        }
    }
}

/// High-level training function
pub async fn train_model(config: TrainingConfig) -> Result<LSTMModel> {
    let trainer = ModelTrainer::new(config);
    trainer.train().await
}
