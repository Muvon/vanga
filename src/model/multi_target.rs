// Multi-target LSTM model implementation
// Trains separate LSTM models for each target to overcome rust-lstm single-target limitation

use crate::config::ModelConfig;
use crate::model::lstm_simple::LSTMModel;
use crate::targets::TargetType;
use crate::utils::error::{Result, VangaError};
use ndarray::{Array2, Array3, Axis};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Defines the context for a training session.
pub enum TrainingContext<'a> {
    /// Standard training with an optional validation set.
    Standard {
        sequences: &'a Array3<f64>,
        targets: &'a Array2<f64>,
        val_sequences: Option<&'a Array3<f64>>,
        val_targets: Option<&'a Array2<f64>>,
    },
    /// Continues training from a previous state.
    Continue {
        new_sequences: &'a Array3<f64>,
        new_targets: &'a Array2<f64>,
    },
}
/// Multi-target LSTM model that trains separate models for each target
pub struct MultiTargetLSTMModel {
    /// Individual LSTM models, one per target
    models: Vec<LSTMModel>,
    /// Names/descriptions of each target
    target_names: Vec<String>,
    /// Input feature size (shared across all models)
    input_size: usize,
    /// Number of targets
    num_targets: usize,
    /// Prediction horizons the model was trained on
    trained_horizons: Vec<String>,
    /// Complete training configuration used during training
    training_config: Option<crate::config::TrainingConfig>,
    /// Feature configuration used during training (kept for backward compatibility)
    feature_config: Option<crate::config::FeatureConfig>,
    /// Normalization statistics used during training (for consistent prediction preprocessing)
    normalization_stats: Option<crate::data::NormalizationStats>,
}

/// Serializable state for multi-target model persistence
#[derive(Debug, Clone, Serialize, Deserialize)]
struct MultiTargetModelState {
    target_names: Vec<String>,
    input_size: usize,
    num_targets: usize,
    /// Prediction horizons the model was trained on (optional for backward compatibility)
    #[serde(default)]
    trained_horizons: Option<Vec<String>>,
    /// Complete training configuration used during training (includes data preprocessing, features, etc.)
    #[serde(default)]
    training_config: Option<crate::config::TrainingConfig>,
    /// Feature configuration used during training (kept for backward compatibility)
    #[serde(default)]
    feature_config: Option<crate::config::FeatureConfig>,
    /// Normalization statistics used during training (for consistent prediction preprocessing)
    #[serde(default)]
    normalization_stats: Option<crate::data::NormalizationStats>,
}

impl MultiTargetLSTMModel {
    /// Calculate output size for individual target model based on configuration
    /// Each target gets its own separate LSTM model with appropriate output size
    fn get_output_size_for_target(target_type: TargetType, model_config: &ModelConfig) -> usize {
        match target_type {
            TargetType::PriceLevel => {
                let bins = model_config.output_heads.price_levels.bins as usize;
                log::debug!(
                    "PriceLevel target: {} output classes (bins from config)",
                    bins
                );
                bins
            }
            TargetType::Direction => {
                log::debug!("Direction target: 3 output classes (Up/Down/Sideways)");
                3 // Fixed: Up, Down, Sideways
            }
            TargetType::Volatility => {
                log::debug!("Volatility target: 3 output classes (Low/Medium/High)");
                3 // Fixed: Low, Medium, High volatility regimes
            }
        }
    }

    /// Determine target type from target name based on actual naming convention
    /// Target names follow pattern: "price_level_1h", "direction_4h", "volatility_1d"
    fn get_target_type_from_name(target_name: &str) -> TargetType {
        if target_name.starts_with("price_level_") {
            TargetType::PriceLevel
        } else if target_name.starts_with("direction_") {
            TargetType::Direction
        } else if target_name.starts_with("volatility_") {
            TargetType::Volatility
        } else {
            // Fallback for unknown patterns - log warning and default to PriceLevel
            log::warn!(
                "Unknown target name pattern '{}', defaulting to PriceLevel",
                target_name
            );
            TargetType::PriceLevel
        }
    }

    /// Create new multi-target LSTM model
    pub fn new(
        model_config: &ModelConfig,
        input_size: usize,
        target_names: Vec<String>,
        trained_horizons: Vec<String>,
    ) -> Result<Self> {
        let num_targets = target_names.len();

        log::info!(
            "🏗️  Creating multi-target LSTM model with {} targets",
            num_targets
        );

        // Create individual LSTM model for each target
        let mut models = Vec::with_capacity(num_targets);
        for target_name in target_names.iter() {
            // Determine target type and calculate proper output size
            let target_type = Self::get_target_type_from_name(target_name);
            let output_size = Self::get_output_size_for_target(target_type, model_config);

            // Create model with proper output size for target type
            let mut model = LSTMModel::from_model_config(model_config, input_size, output_size)?;

            // CRITICAL: Verify the model was created with correct output_size
            let actual_output_size = model.get_output_size();

            // Set target context for proper loss calculation
            model.set_target_context(target_name.clone(), target_type);

            // Compact structured logging per target
            let config_info = match target_type {
                TargetType::PriceLevel => {
                    format!("bins={}", model_config.output_heads.price_levels.bins)
                }
                TargetType::Direction => "classes=3".to_string(),
                TargetType::Volatility => "classes=3".to_string(),
            };

            log::info!(
                "📊 {} [{:?}] → output_size={} ({})",
                target_name,
                target_type,
                actual_output_size,
                config_info
            );

            if actual_output_size != output_size {
                log::error!(
                    "🚨 SIZE MISMATCH: {} expected {} but got {}",
                    target_name,
                    output_size,
                    actual_output_size
                );
            }

            // Reconfigure attention with target context for better logging
            if model_config.attention.enabled {
                model.configure_attention(&model_config.attention, Some(target_name))?;
            }

            models.push(model);
        }

        Ok(Self {
            models,
            target_names,
            input_size,
            num_targets,
            trained_horizons,
            training_config: None,     // Will be set during training
            feature_config: None,      // Will be set during training
            normalization_stats: None, // Will be set during training
        })
    }

    /// Train all target models based on the provided context and configuration.
    pub async fn train(
        &mut self,
        context: TrainingContext<'_>,
        config: &crate::config::TrainingConfig,
    ) -> Result<()> {
        match context {
            TrainingContext::Standard {
                sequences,
                targets,
                val_sequences,
                val_targets,
            } => {
                self.train_internal(sequences, targets, val_sequences, val_targets, config)
                    .await
            }
            TrainingContext::Continue {
                new_sequences,
                new_targets,
            } => {
                self.continue_training(new_sequences, new_targets, config)
                    .await
            }
        }
    }

    /// Continue training with new data for all target models (incremental learning)
    pub async fn continue_training(
        &mut self,
        new_sequences: &Array3<f64>,
        new_targets: &Array2<f64>,
        config: &crate::config::TrainingConfig,
    ) -> Result<()> {
        log::info!(
            "🔄 INCREMENTAL multi-target training: {} models with {} new samples",
            self.models.len(),
            new_sequences.shape()[0]
        );

        // Validate input dimensions
        if new_targets.shape()[1] != self.num_targets {
            return Err(VangaError::ModelError(format!(
                "Target dimension mismatch: expected {} targets, got {}",
                self.num_targets,
                new_targets.shape()[1]
            )));
        }

        // Continue training for each target model
        for (i, model) in self.models.iter_mut().enumerate() {
            let target_name = &self.target_names[i];
            log::info!(
                "Incremental training model {}/{}: {}",
                i + 1,
                self.num_targets,
                target_name
            );

            // Extract single target column for this model
            let single_target = new_targets
                .column(i)
                .into_owned()
                .insert_axis(ndarray::Axis(1));

            // Continue training with new data
            model
                .train(new_sequences, &single_target, config, None, None)
                .await?;

            log::info!(
                "✅ Incremental training completed for target: {}",
                target_name
            );
        }

        log::info!("🎉 Multi-target incremental training completed successfully!");
        Ok(())
    }

    /// Internal training logic for all target models.
    async fn train_internal(
        &mut self,
        sequences: &Array3<f64>,
        targets: &Array2<f64>,
        val_sequences: Option<&Array3<f64>>,
        val_targets: Option<&Array2<f64>>,
        config: &crate::config::TrainingConfig,
    ) -> Result<()> {
        log::info!(
            "Starting multi-target training: {} models for {} targets",
            self.models.len(),
            self.num_targets
        );

        // Validate input dimensions
        if targets.shape()[1] != self.num_targets {
            return Err(VangaError::ModelError(format!(
                "Target dimension mismatch: expected {} targets, got {}",
                self.num_targets,
                targets.shape()[1]
            )));
        }

        if sequences.shape()[2] != self.input_size {
            return Err(VangaError::ModelError(format!(
                "Input dimension mismatch: expected {} features, got {}",
                self.input_size,
                sequences.shape()[2]
            )));
        }

        log::info!(
            "Training data validation: {} sequences, {} features, {} targets",
            sequences.shape()[0],
            sequences.shape()[2],
            targets.shape()[1]
        );

        // Train each model with its corresponding target
        for (i, model) in self.models.iter_mut().enumerate() {
            let target_name = &self.target_names[i];
            log::info!(
                "Training model {}/{}: {}",
                i + 1,
                self.num_targets,
                target_name
            );

            // Extract single target column for this model
            let single_target = targets.column(i).to_owned().insert_axis(Axis(1));
            let val_single_target =
                val_targets.map(|vt| vt.column(i).to_owned().insert_axis(Axis(1)));

            log::debug!(
                "Target {} shape: {:?}, values range: [{:.4}, {:.4}]",
                target_name,
                single_target.shape(),
                single_target.iter().fold(f64::INFINITY, |a, &b| a.min(b)),
                single_target
                    .iter()
                    .fold(f64::NEG_INFINITY, |a, &b| a.max(b))
            );

            // Train individual model
            log::info!("📊 TARGET {} TRAINING SUMMARY:", target_name.to_uppercase());
            log::info!(
                "   • Training Data: {} sequences × {} features",
                sequences.shape()[0],
                sequences.shape()[2]
            );
            log::info!("   • Target Shape: {:?}", single_target.shape());
            if let Some(val_seq) = val_sequences {
                log::info!(
                    "   • Validation Data: {} sequences (chronological split)",
                    val_seq.shape()[0]
                );
            }
            log::info!(
                "   • Optimization: {}",
                if config.optimization.method != crate::config::training::OptimizationMethod::None {
                    format!(
                        "{:?} with {} trials",
                        config.optimization.method, config.optimization.n_trials
                    )
                } else {
                    "Disabled (using default parameters)".to_string()
                }
            );
            match model
                .train(
                    sequences,
                    &single_target,
                    config,
                    val_sequences,
                    val_single_target.as_ref(),
                )
                .await
            {
                Ok(_) => {
                    log::info!("✅ Successfully trained model for target: {}", target_name);
                }
                Err(e) => {
                    log::error!("❌ Failed to train model for target {}: {}", target_name, e);
                    return Err(VangaError::ModelError(format!(
                        "Training failed for target '{}': {}",
                        target_name, e
                    )));
                }
            }
        }

        log::info!(
            "🎉 Multi-target training completed successfully for all {} targets!",
            self.num_targets
        );
        Ok(())
    }

    /// Make predictions using all trained models (memory-optimized)
    pub async fn predict(&self, sequences: &Array3<f64>) -> Result<Array2<f64>> {
        log::info!(
            "Making multi-target predictions for {} sequences using {} models",
            sequences.shape()[0],
            self.models.len()
        );

        let batch_size = sequences.shape()[0];
        let mut all_predictions = Array2::zeros((batch_size, self.num_targets));

        // Process each model sequentially to avoid memory accumulation
        for (i, model) in self.models.iter().enumerate() {
            let target_name = &self.target_names[i];
            log::debug!(
                "Processing model {}/{}: {}",
                i + 1,
                self.num_targets,
                target_name
            );

            match model.predict(sequences).await {
                Ok(predictions) => {
                    // predictions should be [batch_size, 1] since each model has single output
                    if predictions.shape()[1] != 1 {
                        return Err(VangaError::ModelError(format!(
                            "Model {} returned unexpected prediction shape: {:?}, expected [batch, 1]",
                            target_name, predictions.shape()
                        )));
                    }

                    // Copy predictions to the appropriate column
                    for batch_idx in 0..batch_size {
                        all_predictions[[batch_idx, i]] = predictions[[batch_idx, 0]];
                    }

                    log::debug!(
                        "Model {} predictions: range [{:.4}, {:.4}]",
                        target_name,
                        predictions.iter().fold(f64::INFINITY, |a, &b| a.min(b)),
                        predictions.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b))
                    );

                    // Explicit memory cleanup after each model prediction
                    drop(predictions);

                    // Force garbage collection hint between models
                    if i < self.models.len() - 1 {
                        std::hint::black_box(());
                    }
                }
                Err(e) => {
                    log::error!("Prediction failed for model {}: {}", target_name, e);
                    return Err(VangaError::ModelError(format!(
                        "Prediction failed for target '{}': {}",
                        target_name, e
                    )));
                }
            }
        }

        log::info!(
            "✅ Multi-target predictions completed: {} predictions for {} targets",
            batch_size,
            self.num_targets
        );

        Ok(all_predictions)
    }

    /// Save all models to disk
    pub fn save<P: AsRef<Path>>(&self, base_path: P) -> Result<()> {
        let base_path = base_path.as_ref();

        log::info!("Saving multi-target model to: {}", base_path.display());

        // Create directory if it doesn't exist
        if let Some(parent) = base_path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| VangaError::DataError(format!("Failed to create directory: {}", e)))?;
        }

        // Save metadata
        let state = MultiTargetModelState {
            target_names: self.target_names.clone(),
            input_size: self.input_size,
            num_targets: self.num_targets,
            trained_horizons: Some(self.trained_horizons.clone()),
            training_config: self.training_config.clone(),
            feature_config: self.feature_config.clone(),
            normalization_stats: self.normalization_stats.clone(),
        };

        let metadata_path = base_path.with_extension("meta");
        let serialized = bincode::serialize(&state).map_err(|e| {
            VangaError::SerializationError(format!("Failed to serialize metadata: {}", e))
        })?;

        std::fs::write(&metadata_path, serialized)
            .map_err(|e| VangaError::DataError(format!("Failed to write metadata: {}", e)))?;

        // Save each individual model
        for (i, model) in self.models.iter().enumerate() {
            let model_path = format!("{}_{}.bin", base_path.to_string_lossy(), i);
            model.save(&model_path)?;
            log::debug!("Saved model {} to: {}", i + 1, model_path);
        }

        log::info!("✅ Multi-target model saved successfully");
        Ok(())
    }

    /// Load multi-target model from disk
    pub fn load<P: AsRef<Path>>(base_path: P) -> Result<Self> {
        let base_path = base_path.as_ref();

        log::info!("Loading multi-target model from: {}", base_path.display());

        // Load metadata
        let metadata_path = base_path.with_extension("meta");
        let serialized = std::fs::read(&metadata_path)
            .map_err(|e| VangaError::DataError(format!("Failed to read metadata: {}", e)))?;

        let state: MultiTargetModelState = bincode::deserialize(&serialized).map_err(|e| {
            VangaError::SerializationError(format!("Failed to deserialize metadata: {}", e))
        })?;

        // Load individual models
        let mut models = Vec::with_capacity(state.num_targets);
        for i in 0..state.num_targets {
            let model_path = format!("{}_{}.bin", base_path.to_string_lossy(), i);
            log::debug!("Loading model {} from: {}", i + 1, model_path);

            let mut model = LSTMModel::load(&model_path)?;

            // CRITICAL FIX: Restore target context after loading
            // The target_context is lost during serialization/deserialization
            let target_name = &state.target_names[i];
            let target_type = Self::get_target_type_from_name(target_name);
            model.set_target_context(target_name.clone(), target_type);

            log::debug!(
                "Restored target context for model {}: {} -> {:?}",
                i + 1,
                target_name,
                target_type
            );

            models.push(model);
        }

        log::info!(
            "✅ Multi-target model loaded successfully with {} targets",
            state.num_targets
        );

        // Handle backward compatibility - if no trained_horizons in metadata, use default
        let trained_horizons = state
            .trained_horizons
            .unwrap_or_else(|| vec!["1h".to_string()]);

        Ok(Self {
            models,
            target_names: state.target_names,
            input_size: state.input_size,
            num_targets: state.num_targets,
            trained_horizons,
            training_config: state.training_config,
            feature_config: state.feature_config,
            normalization_stats: state.normalization_stats,
        })
    }

    /// Get target names
    pub fn get_target_names(&self) -> &[String] {
        &self.target_names
    }

    /// Get trained horizons
    pub fn get_trained_horizons(&self) -> &[String] {
        &self.trained_horizons
    }

    /// Get number of targets
    pub fn get_num_targets(&self) -> usize {
        self.num_targets
    }

    /// Get input size
    pub fn get_input_size(&self) -> usize {
        self.input_size
    }

    /// Get feature configuration used during training
    pub fn get_feature_config(&self) -> Option<&crate::config::FeatureConfig> {
        self.feature_config.as_ref()
    }

    /// Set feature configuration (used during training)
    pub fn set_feature_config(&mut self, config: crate::config::FeatureConfig) {
        self.feature_config = Some(config);
    }

    /// Get training configuration used during training
    pub fn get_training_config(&self) -> Option<&crate::config::TrainingConfig> {
        self.training_config.as_ref()
    }

    /// Set training configuration (used during training)
    pub fn set_training_config(&mut self, config: crate::config::TrainingConfig) {
        self.training_config = Some(config);
    }

    /// Set normalization statistics (used during training)
    pub fn set_normalization_stats(&mut self, stats: crate::data::NormalizationStats) {
        self.normalization_stats = Some(stats);
    }

    /// Get normalization statistics (used during prediction)
    pub fn get_normalization_stats(&self) -> Option<&crate::data::NormalizationStats> {
        self.normalization_stats.as_ref()
    }

    /// Check if all models are trained (have networks)
    pub fn is_trained(&self) -> bool {
        // This would need to be implemented in LSTMModel to check if network exists
        // For now, assume trained if models exist
        !self.models.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;

    #[tokio::test]
    async fn test_multi_target_creation() {
        let model_config = ModelConfig::default();
        let target_names = vec![
            "price_1h".to_string(),
            "direction".to_string(),
            "volatility".to_string(),
        ];

        let result = MultiTargetLSTMModel::new(
            &model_config,
            10,
            target_names.clone(),
            vec!["1h".to_string()],
        );
        assert!(result.is_ok());

        let model = result.unwrap();
        assert_eq!(model.get_num_targets(), 3);
        assert_eq!(model.get_input_size(), 10);
        assert_eq!(model.get_target_names(), &target_names);
    }

    #[tokio::test]
    async fn test_multi_target_training_validation() {
        let model_config = ModelConfig::default();
        let target_names = vec!["target1".to_string(), "target2".to_string()];
        let mut model =
            MultiTargetLSTMModel::new(&model_config, 5, target_names, vec!["1h".to_string()])
                .unwrap();

        // Create a test config
        let config = crate::config::TrainingConfig {
            symbol: "BTCUSDT".to_string(),
            data_path: std::path::PathBuf::from("test.csv"),
            fresh_training: true,
            continue_training: false,
            horizons: vec!["1h".to_string()],
            features: crate::config::FeatureConfig::default(),
            model: ModelConfig::default(),
            training: crate::config::training::TrainingParams {
                epochs: crate::config::training::EpochConfig::Fixed(1),
                batch_size: crate::config::training::BatchSizeConfig::Fixed(32),
                learning_rate: crate::config::training::LearningRateConfig::Fixed(0.01),
                optimizer: crate::config::training::OptimizerType::AdamW {
                    weight_decay: 0.01,
                    beta1: 0.9,
                    beta2: 0.999,
                },
                warmup_epochs: 0,
                learning_schedule: None,
                validation_split: 0.0,
                test_split: 0.0,
                early_stopping: crate::config::training::EarlyStoppingConfig {
                    patience: 10,
                    min_delta: 0.0001,
                },
                device: crate::config::training::DeviceConfig::Auto,
                gradient_clip: Some(1.0),
                print_every: 1, // Add missing print_every field
            },
            data: crate::config::training::DataConfig::default(),
            optimization: crate::config::training::OptimizationConfig::default(),
        };

        // Create test data with wrong target dimensions
        let sequences = Array3::zeros((10, 30, 5)); // [batch, seq_len, features]
        let wrong_targets = Array2::zeros((10, 3)); // Wrong: 3 targets instead of 2

        let result = model
            .train(
                TrainingContext::Standard {
                    sequences: &sequences,
                    targets: &wrong_targets,
                    val_sequences: None,
                    val_targets: None,
                },
                &config,
            )
            .await;
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Target dimension mismatch"));
    }
}
