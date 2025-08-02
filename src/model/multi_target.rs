// Multi-target LSTM model implementation
// Trains separate LSTM models for each target to overcome rust-lstm single-target limitation

use crate::config::ModelConfig;
use crate::model::lstm_simple::LSTMModel;
use crate::targets::TargetType;
use crate::utils::error::{Result, VangaError};
use ndarray::{Array2, Array3, Axis};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

/// Defines the context for a training session.
pub enum TrainingContext<'a> {
    /// Standard training with an optional validation set.
    Standard {
        sequences: &'a Array3<f64>,
        targets: &'a Array2<f64>,
        val_sequences: Option<&'a Array3<f64>>,
        val_targets: Option<&'a Array2<f64>>,
        /// Target-specific class weights for balanced training
        /// Key format: "{target_type}_{horizon}" (e.g., "PriceLevel_1h", "Direction_4h")
        target_class_weights: Option<&'a HashMap<String, Vec<f32>>>,
    },
    /// Continues training from a previous state.
    Continue {
        new_sequences: &'a Array3<f64>,
        new_targets: &'a Array2<f64>,
        /// Target-specific class weights for balanced training
        /// Key format: "{target_type}_{horizon}" (e.g., "PriceLevel_1h", "Direction_4h")
        target_class_weights: Option<&'a HashMap<String, Vec<f32>>>,
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
    fn get_output_size_for_target(target_type: TargetType, _model_config: &ModelConfig) -> usize {
        match target_type {
            TargetType::PriceLevel => {
                let bins = crate::config::model::NUM_CLASSES; // Use unified 5-class system
                log::debug!(
                    "PriceLevel target: {} output classes (5-class unified system)",
                    bins
                );
                bins
            }
            TargetType::Direction => {
                log::debug!(
                    "Direction target: {} output classes (Dump/Down/Sideways/Up/Pump)",
                    crate::config::model::NUM_CLASSES
                );
                crate::config::model::NUM_CLASSES // Use unified 5-class system
            }
            TargetType::Volatility => {
                log::debug!(
                    "Volatility target: {} output classes (VeryLow/Low/Medium/High/VeryHigh)",
                    crate::config::model::NUM_CLASSES
                );
                crate::config::model::NUM_CLASSES // Use unified 5-class system volatility
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
                    "bins=5".to_string() // Use unified 5-class system
                }
                TargetType::Direction => "classes=5".to_string(),
                TargetType::Volatility => "classes=5".to_string(),
            };

            log::info!(
                "📊 {} [{:?}] → output_size={} ({})",
                target_name,
                target_type,
                actual_output_size,
                config_info
            );

            if actual_output_size != output_size {
                return Err(VangaError::ModelError(format!(
                    "🚨 CRITICAL: Target '{}' output size mismatch! Expected {} classes but model has {} classes. \
                    This indicates a bins configuration mismatch between target generation and model creation.",
                    target_name,
                    output_size,
                    actual_output_size
                )));
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
                target_class_weights,
            } => {
                self.train_internal(
                    sequences,
                    targets,
                    val_sequences,
                    val_targets,
                    target_class_weights,
                    config,
                )
                .await
            }
            TrainingContext::Continue {
                new_sequences,
                new_targets,
                target_class_weights: _, // Unused in continue training
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
                .train(new_sequences, &single_target, config, None, None, None)
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
        target_class_weights: Option<&HashMap<String, Vec<f32>>>,
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

            // CRITICAL VALIDATION: Ensure target dimensions match model output dimensions
            let model_output_size = model.get_output_size();
            let target_type = Self::get_target_type_from_name(target_name);
            let expected_output_size = Self::get_output_size_for_target(target_type, &config.model);

            if model_output_size != expected_output_size {
                return Err(VangaError::ModelError(format!(
                    "🚨 TRAINING VALIDATION FAILED: Target '{}' has model output size {} but expected {} for {:?}. \
                    This indicates a bins configuration mismatch. Check your configuration file.",
                    target_name,
                    model_output_size,
                    expected_output_size,
                    target_type
                )));
            }

            log::info!(
                "Training model {}/{}: {} (output size: {} classes)",
                i + 1,
                self.num_targets,
                target_name,
                model_output_size
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

            // Get target-specific class weights for this model
            let target_specific_weights = if let Some(weights_map) = target_class_weights {
                // Extract target type and horizon from target name (format: "target_type_horizon")
                let target_type = Self::get_target_type_from_name(target_name);

                // Find the horizon from target name (assumes format like "price_level_1h")
                let horizon = if let Some(last_underscore) = target_name.rfind('_') {
                    &target_name[last_underscore + 1..]
                } else {
                    // Fallback to first horizon if parsing fails
                    config.horizons.first().map(|h| h.as_str()).unwrap_or("1h")
                };

                let weights_key = format!("{:?}_{}", target_type, horizon);
                let weights = weights_map.get(&weights_key);

                if let Some(w) = weights {
                    log::debug!(
                        "🎯 Using target-specific class weights for {}: {} classes",
                        target_name,
                        w.len()
                    );
                } else {
                    log::debug!(
                        "⚠️ No target-specific class weights found for key '{}', using None",
                        weights_key
                    );
                }

                weights
            } else {
                None
            };

            match model
                .train(
                    sequences,
                    &single_target,
                    config,
                    val_sequences,
                    val_single_target.as_ref(),
                    target_specific_weights,
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
        // Calculate total output size: each target has NUM_CLASSES outputs (5-class system)
        let total_output_size = self.num_targets * crate::config::model::NUM_CLASSES;
        let mut all_predictions = Array2::zeros((batch_size, total_output_size));

        // Process each model sequentially to avoid memory accumulation
        for (i, model) in self.models.iter().enumerate() {
            let target_name = &self.target_names[i];
            let has_xgboost = model.xgboost_model.is_some();
            log::info!(
                "🎯 Processing model {}/{}: {} (XGBoost: {})",
                i + 1,
                self.num_targets,
                target_name,
                if has_xgboost {
                    "✅ Enabled"
                } else {
                    "❌ Disabled"
                }
            );

            match model.predict(sequences).await {
                Ok(predictions) => {
                    // predictions should be [batch_size, NUM_CLASSES] since each model has 5 outputs (5-class system)
                    let expected_classes = crate::config::model::NUM_CLASSES;
                    if predictions.shape()[1] != expected_classes {
                        return Err(VangaError::ModelError(format!(
                            "Model {} returned unexpected prediction shape: {:?}, expected [batch, {}] for 5-class system",
                            target_name, predictions.shape(), expected_classes
                        )));
                    }

                    // Copy predictions to the appropriate columns (5 columns per target)
                    let start_col = i * expected_classes;
                    for batch_idx in 0..batch_size {
                        for class_idx in 0..expected_classes {
                            all_predictions[[batch_idx, start_col + class_idx]] =
                                predictions[[batch_idx, class_idx]];
                        }
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
        let serialized = serde_json::to_string_pretty(&state).map_err(|e| {
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
        let serialized = std::fs::read_to_string(&metadata_path)
            .map_err(|e| VangaError::DataError(format!("Failed to read metadata: {}", e)))?;

        let state: MultiTargetModelState = serde_json::from_str(&serialized).map_err(|e| {
            VangaError::SerializationError(format!("Failed to deserialize metadata: {}", e))
        })?;

        // Load individual models
        let mut models = Vec::with_capacity(state.num_targets);
        for i in 0..state.num_targets {
            // Use the correct base path without extension - LSTMModel methods will add extensions
            let model_base_path = format!("{}_{}", base_path.to_string_lossy(), i);
            log::debug!(
                "Loading model {} from base path: {}",
                i + 1,
                model_base_path
            );

            // UNIFIED APPROACH: Use load_with_model_config to respect stored architecture
            let mut model = if let Some(training_config) = &state.training_config {
                log::debug!(
                    "🔧 Loading model {} with stored training_config.model architecture",
                    i + 1
                );
                LSTMModel::load_with_model_config(
                    &model_base_path,
                    &training_config.model,
                    state.input_size,
                    crate::config::model::NUM_CLASSES,
                )?
            } else {
                log::warn!("⚠️ Loading legacy model {} without training config", i + 1);
                LSTMModel::load(&model_base_path)?
            };

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
