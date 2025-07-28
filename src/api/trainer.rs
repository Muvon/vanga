// Model trainer
use crate::config::TrainingConfig;
use crate::data::DataPipeline;
use crate::model::multi_target::{MultiTargetLSTMModel, TrainingContext};
use crate::targets::PreparedTargets;
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

        // Initialize device from configuration
        let device_string = self.config.training.device.to_device_string();
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

        // Initialize data pipeline
        let data_pipeline = DataPipeline::new();

        // Load and prepare training data with chronological split
        log::info!(
            "Loading training data from: {}",
            self.config.data_path.display()
        );
        let windows = data_pipeline
            .prepare_training_data(&self.config.data_path, &self.config)
            .await?;

        log::info!(
            "Walk-forward training: {} windows prepared for progressive learning",
            windows.len()
        );

        // Train model using walk-forward analysis
        let mut model = None;

        for (i, window) in windows.iter().enumerate() {
            log::info!(
                "🔄 Walk-forward window {}/{}: {} train samples → {} validation samples",
                i + 1,
                windows.len(),
                window.train_samples,
                window.val_samples
            );

            if i == 0 {
                // First window: train from scratch
                model = Some(self.train_window_from_scratch(window).await?);
            } else {
                // Subsequent windows: continue training on expanded data
                model = Some(
                    self.continue_training_window(model.unwrap(), window)
                        .await?,
                );
            }
        }

        let mut final_model = model.unwrap();

        // CRITICAL FIX: Set complete training config on model before saving
        // This ensures prediction can regenerate the same features and settings as training
        final_model.set_training_config(self.config.clone());

        // CRITICAL FIX: Set normalization stats from training data
        // This ensures prediction uses the same normalization as training
        if let Some(first_window) = windows.first() {
            final_model
                .set_normalization_stats(first_window.train_data.normalization_stats.clone());
            log::info!(
                "✅ Normalization stats saved with model for consistent prediction preprocessing"
            );
        } else {
            log::warn!("⚠️  No training windows available - normalization stats not saved");
        }

        // Save the trained multi-target model
        log::info!("✅ Walk-forward multi-target model training completed successfully!");
        log::info!("🔧 Complete training config saved with model for consistent prediction");
        Ok(final_model)
    }

    /// Train model from scratch on first window
    async fn train_window_from_scratch(
        &self,
        window: &crate::data::TrainingWindow,
    ) -> Result<MultiTargetLSTMModel> {
        log::info!(
            "🎯 [train_window_from_scratch] Training config horizons: {:?} (count: {})",
            self.config.horizons,
            self.config.horizons.len()
        );

        // FIXED: Use target names from prepared data (eliminates redundant TargetGenerator creation)
        let target_names = &window.train_data.targets.target_names;
        log::info!(
            "Using targets from prepared data: {} targets: {:?}",
            target_names.len(),
            target_names
        );

        // Convert raw targets to Array2 format for training and validation separately
        let train_targets =
            convert_raw_targets_to_array2(&window.train_data.targets, target_names)?;
        let val_targets = convert_raw_targets_to_array2(&window.val_data.targets, target_names)?;

        log::info!(
            "Window {} training targets: {} train samples x {} outputs, {} validation samples",
            window.window_id + 1,
            train_targets.shape()[0],
            train_targets.shape()[1],
            val_targets.shape()[0]
        );

        // Create multi-target model
        let input_size = window.train_data.sequences.shape()[2];
        let mut model = self
            .get_or_create_multi_target_model(input_size, target_names)
            .await?;

        // Train the model with chronological validation
        model
            .train(
                TrainingContext::Standard {
                    sequences: &window.train_data.sequences,
                    targets: &train_targets,
                    val_sequences: Some(&window.val_data.sequences),
                    val_targets: Some(&val_targets),
                    target_class_weights: Some(&window.target_class_weights),
                },
                &self.config,
            )
            .await?;

        log::info!("✅ Window {} training completed", window.window_id + 1);
        Ok(model)
    }

    /// Continue training existing model on new window
    async fn continue_training_window(
        &self,
        mut model: MultiTargetLSTMModel,
        window: &crate::data::TrainingWindow,
    ) -> Result<MultiTargetLSTMModel> {
        log::info!(
            "🎯 [continue_training_window] Training config horizons: {:?} (count: {})",
            self.config.horizons,
            self.config.horizons.len()
        );

        // FIXED: Use target names from prepared data (eliminates redundant TargetGenerator creation)
        let target_names = &window.train_data.targets.target_names;

        // FIXED: Use targets directly from prepared data (already aligned with sequences)
        let train_targets =
            convert_raw_targets_to_array2(&window.train_data.targets, target_names)?;
        let val_targets = convert_raw_targets_to_array2(&window.val_data.targets, target_names)?;

        log::info!(
            "Window {} continue training: {} new train samples, {} validation samples",
            window.window_id + 1,
            train_targets.shape()[0],
            val_targets.shape()[0]
        );

        // Continue training with new data
        model
            .train(
                TrainingContext::Continue {
                    new_sequences: &window.train_data.sequences,
                    new_targets: &train_targets,
                    target_class_weights: Some(&window.target_class_weights),
                },
                &self.config,
            )
            .await?;

        log::info!(
            "✅ Window {} continue training completed",
            window.window_id + 1
        );
        Ok(model)
    }

    /// Get existing multi-target model or create new one based on training configuration
    async fn get_or_create_multi_target_model(
        &self,
        input_size: usize,
        target_names: &[String],
    ) -> Result<MultiTargetLSTMModel> {
        // Create new model since we're not loading from file anymore
        // The caller (main.rs) will handle loading/saving based on training config
        log::info!("🆕 Creating new multi-target model for training");
        MultiTargetLSTMModel::new(
            &self.config.model,
            input_size,
            target_names.to_vec(),
            self.config.horizons.clone(),
        )
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
