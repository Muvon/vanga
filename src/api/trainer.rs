// Model trainer
use crate::config::TrainingConfig;
use crate::data::DataPipeline;
use crate::model::multi_target::MultiTargetLSTMModel;
use crate::targets::{PreparedTargets, TargetGenerator};
use crate::utils::error::{Result, VangaError};
use ndarray::Array2;

pub struct ModelTrainer {
    config: TrainingConfig,
}

impl ModelTrainer {
    pub fn new(config: TrainingConfig) -> Self {
        Self { config }
    }

    pub async fn train(&self) -> Result<MultiTargetLSTMModel> {
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

        // Get target names for multi-target model
        let target_names = target_generator.get_target_names();
        log::info!(
            "Generated {} targets: {:?}",
            target_names.len(),
            target_names
        );

        // CRITICAL FIX: Use raw targets directly for multi-target model
        // Skip TargetConverter which expands 3 targets to 14 one-hot encoded outputs
        // MultiTargetLSTMModel expects raw target values, not one-hot encoded

        // Convert raw targets to Array2 format for training
        let training_targets = convert_raw_targets_to_array2(&targets, &target_names)?;

        log::info!(
            "Training targets prepared: {} samples x {} outputs (raw targets, not one-hot)",
            training_targets.shape()[0],
            training_targets.shape()[1]
        );

        // CRITICAL FIX: Use MultiTargetLSTMModel instead of single-target model
        // This eliminates the 93% data loss issue
        let input_size = prepared_data.sequences.shape()[2]; // Number of features
        let model_path = format!("./models/{}_multi_model", self.config.symbol); // Different path for multi-target

        let mut model = self
            .get_or_create_multi_target_model(&model_path, input_size, target_names)
            .await?;

        // Train the multi-target LSTM model with intelligent early stopping
        log::info!(
            "🚀 Starting intelligent multi-target training with early stopping - using ALL {} targets (0% data loss)",
            training_targets.shape()[1]
        );
        model
            .train_with_early_stopping(&prepared_data.sequences, &training_targets, &self.config)
            .await?;

        // Save the trained multi-target model
        log::info!("💾 Saving multi-target model to: {}", model_path);
        model.save(&model_path)?;

        log::info!("✅ Multi-target model training completed successfully!");
        Ok(model)
    }

    /// Get existing multi-target model or create new one based on training configuration
    async fn get_or_create_multi_target_model(
        &self,
        model_path: &str,
        input_size: usize,
        target_names: Vec<String>,
    ) -> Result<MultiTargetLSTMModel> {
        let model_exists = std::path::Path::new(&format!("{}.meta", model_path)).exists();

        match (self.config.fresh_training, self.config.continue_training, model_exists) {
            // Fresh training requested - always create new model
            (true, _, _) => {
                log::info!("🆕 Fresh training requested - creating new multi-target model");
                MultiTargetLSTMModel::new(&self.config.model_config, input_size, target_names)
            }

            // Continue training requested - must have existing model
            (false, true, false) => {
                Err(VangaError::ModelError(format!(
                    "Continue training requested but no existing multi-target model found for symbol: {}. Train without --continue-training first, or use --fresh to start new training.",
                    self.config.symbol
                )))
            }
            (false, true, true) => {
                log::info!("📂 Continue training requested - loading existing multi-target model from: {}", model_path);
                let model = MultiTargetLSTMModel::load(model_path)?;

                // Validate model compatibility
                if model.get_input_size() != input_size {
                    return Err(VangaError::ConfigError(format!(
                        "Model input size mismatch: existing model expects {} features, but data has {} features. Use --fresh to retrain with new data structure.",
                        model.get_input_size(),
                        input_size
                    )));
                }

                if model.get_num_targets() != target_names.len() {
                    return Err(VangaError::ConfigError(format!(
                        "Model target count mismatch: existing model has {} targets, but config needs {} targets. Use --fresh to retrain with new target configuration.",
                        model.get_num_targets(),
                        target_names.len()
                    )));
                }

                Ok(model)
            }

            // Default behavior - continue if model exists, otherwise create new
            (false, false, true) => {
                log::info!("📂 Existing multi-target model found - continuing training from: {}", model_path);
                match MultiTargetLSTMModel::load(model_path) {
                    Ok(model) => {
                        // Check compatibility, fallback to fresh if incompatible
                        if model.get_input_size() != input_size {
                            log::warn!(
                                "Model input size mismatch (expected {}, got {}). Creating fresh multi-target model.",
                                model.get_input_size(),
                                input_size
                            );
                            MultiTargetLSTMModel::new(&self.config.model_config, input_size, target_names)
                        } else if model.get_num_targets() != target_names.len() {
                            log::warn!(
                                "Model target count mismatch (expected {}, got {}). Creating fresh multi-target model.",
                                model.get_num_targets(),
                                target_names.len()
                            );
                            MultiTargetLSTMModel::new(&self.config.model_config, input_size, target_names)
                        } else {
                            Ok(model)
                        }
                    }
                    Err(e) => {
                        log::warn!("Failed to load existing multi-target model: {}. Creating fresh model.", e);
                        MultiTargetLSTMModel::new(&self.config.model_config, input_size, target_names)
                    }
                }
            }
            (false, false, false) => {
                log::info!("🆕 No existing multi-target model found - creating new model");
                MultiTargetLSTMModel::new(&self.config.model_config, input_size, target_names)
            }
        }
    }
}

/// High-level training function
pub async fn train_model(config: TrainingConfig) -> Result<MultiTargetLSTMModel> {
    let trainer = ModelTrainer::new(config);
    trainer.train().await
}

/// Convert raw targets to Array2 format for multi-target training
/// This bypasses the TargetConverter which creates one-hot encoded outputs
fn convert_raw_targets_to_array2(
    targets: &PreparedTargets,
    target_names: &[String],
) -> Result<Array2<f64>> {
    let num_samples = targets.valid_indices.len();
    let num_targets = target_names.len();

    if num_samples == 0 {
        return Err(VangaError::DataError(
            "No valid samples for target conversion".to_string(),
        ));
    }

    let mut training_array = Array2::<f64>::zeros((num_samples, num_targets));

    // Extract targets in the order specified by target_names
    for (target_idx, target_name) in target_names.iter().enumerate() {
        // Parse target name format: "price_level_1h", "direction_1h", "volatility_1h"
        let parts: Vec<&str> = target_name.split('_').collect();
        if parts.len() < 2 {
            return Err(VangaError::DataError(format!(
                "Invalid target name format: {}",
                target_name
            )));
        }

        // Handle compound target types like "price_level"
        let (target_type, horizon) =
            if parts.len() == 3 && parts[0] == "price" && parts[1] == "level" {
                ("price_level", parts[2])
            } else if parts.len() == 2 {
                (parts[0], parts[1])
            } else {
                return Err(VangaError::DataError(format!(
                    "Invalid target name format: {}",
                    target_name
                )));
            };

        // Get the appropriate target data based on type
        let target_data = match target_type {
            "price_level" => targets.price_levels.get(horizon),
            "direction" => targets.directions.get(horizon),
            "volatility" => targets.volatility.get(horizon),
            _ => {
                return Err(VangaError::DataError(format!(
                    "Unknown target type: {}",
                    target_type
                )))
            }
        };

        if let Some(data) = target_data {
            // Fill the training array with raw target values
            for (sample_idx, &data_idx) in targets.valid_indices.iter().enumerate() {
                if data_idx < data.len() {
                    training_array[[sample_idx, target_idx]] = data[data_idx] as f64;
                }
            }
        } else {
            return Err(VangaError::DataError(format!(
                "Target data not found for: {}",
                target_name
            )));
        }
    }

    log::info!(
        "Converted {} raw targets to Array2: {:?}",
        num_targets,
        training_array.shape()
    );
    Ok(training_array)
}
