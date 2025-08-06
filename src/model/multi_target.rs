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
    /// CRITICAL: new_sequences is CUMULATIVE (Window1: 2880, Window2: 3744, etc.) - NOT just new data
    Continue {
        new_sequences: &'a Array3<f64>, // CUMULATIVE sequences (preserves + extends)
        new_targets: &'a Array2<f64>,
        /// CRITICAL: Validation data prevents overfitting in continuation training
        val_sequences: Option<&'a Array3<f64>>,
        val_targets: Option<&'a Array2<f64>>,
    },
}
/// Multi-target LSTM model that trains separate models for each target
///
/// ARCHITECTURE CLARIFICATION:
/// - This is a WRAPPER containing multiple individual LSTMModel instances (one per target)
/// - Each LSTMModel inside handles ONE target (e.g., price_level_1h, direction_4h, etc.)
/// - In target-specific training, this wrapper may contain only ONE LSTMModel
/// - This design avoids training PriceLevel models on Direction-specific sequences
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
        Self::new_with_seed(
            model_config,
            input_size,
            target_names,
            trained_horizons,
            None,
        )
    }

    /// Create new multi-target LSTM model with seed for reproducible training
    pub fn new_with_seed(
        model_config: &ModelConfig,
        input_size: usize,
        target_names: Vec<String>,
        trained_horizons: Vec<String>,
        seed: Option<u64>,
    ) -> Result<Self> {
        let num_targets = target_names.len();

        log::info!(
            "🏗️  Creating multi-target LSTM model with {} targets",
            num_targets
        );

        if let Some(seed_value) = seed {
            log::info!("🎲 Multi-target model using seed: {}", seed_value);
            if seed_value == 0 {
                log::info!("🎲 Seed = 0: Each target model will use random initialization");
            } else {
                log::info!("🎲 Seed = {}: Each target model will use reproducible initialization with incremental seeds", seed_value);
            }
        } else {
            log::info!("🎲 Multi-target model using random initialization for all targets");
        }

        // Create individual LSTM model for each target
        let mut models = Vec::with_capacity(num_targets);
        for (i, target_name) in target_names.iter().enumerate() {
            // Determine target type and calculate proper output size
            let target_type = Self::get_target_type_from_name(target_name);
            let output_size = Self::get_output_size_for_target(target_type, model_config);

            // Calculate per-target seed for consistency across targets
            let target_seed = seed.and_then(|s| if s == 0 { None } else { Some(s + i as u64) });

            // Create model with proper output size for target type and seed
            let mut model = if let Some(target_seed_value) = target_seed {
                log::debug!(
                    "🎲 Target '{}' using seed: {}",
                    target_name,
                    target_seed_value
                );
                LSTMModel::from_model_config_with_seed(
                    model_config,
                    input_size,
                    output_size,
                    Some(target_seed_value),
                )?
            } else {
                LSTMModel::from_model_config(model_config, input_size, output_size)?
            };

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
            training_config: None, // Will be set during training
            feature_config: None,  // Will be set during training
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
                val_sequences,
                val_targets,
            } => {
                self.continue_training(
                    new_sequences,
                    new_targets,
                    val_sequences,
                    val_targets,
                    config,
                )
                .await
            }
        }
    }

    /// Continue training with new data for all target models (incremental learning)
    /// CRITICAL: new_sequences is CUMULATIVE data (Window1: 2880, Window2: 3744, etc.)
    /// CRITICAL: LSTMModel.trained flag prevents weight reinitialization - preserves learned patterns
    pub async fn continue_training(
        &mut self,
        new_sequences: &Array3<f64>, // CUMULATIVE sequences (not just new data)
        new_targets: &Array2<f64>,
        val_sequences: Option<&Array3<f64>>, // CRITICAL: Prevents overfitting
        val_targets: Option<&Array2<f64>>,   // CRITICAL: Prevents overfitting
        config: &crate::config::TrainingConfig,
    ) -> Result<()> {
        log::info!(
            "🔄 INCREMENTAL multi-target training: {} models with {} new samples",
            self.models.len(),
            new_sequences.shape()[0]
        );

        // Log validation data usage for continuation training
        if let (Some(val_seq), Some(_val_tgt)) = (val_sequences, val_targets) {
            log::info!(
                "✅ CONTINUATION TRAINING with validation: {} train, {} val samples (prevents overfitting)",
                new_sequences.shape()[0],
                val_seq.shape()[0]
            );
        } else {
            log::warn!(
                "⚠️  CONTINUATION TRAINING without validation: {} train samples (risk of overfitting)",
                new_sequences.shape()[0]
            );
        }

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

            // Extract validation target if validation data is provided
            let single_val_target = val_targets.map(|val_targets| {
                val_targets
                    .column(i)
                    .into_owned()
                    .insert_axis(ndarray::Axis(1))
            });

            // Continue training with new data and validation (prevents overfitting)
            // CRITICAL: LSTMModel.train() with val_sequences prevents overfitting in continuation
            model
                .train(
                    new_sequences,
                    &single_target,
                    config,
                    val_sequences, // CRITICAL: Validation prevents overfitting
                    single_val_target.as_ref(), // CRITICAL: Validation prevents overfitting
                )
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
    #[allow(clippy::too_many_arguments)]
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

            // Extract target-specific validation sequences using balanced indices
            let (target_val_sequences, target_val_targets) =
                if let (Some(val_seq), Some(val_tgt)) = (val_sequences, val_targets) {
                    let target_val_seq = val_seq.to_owned();
                    let target_val_tgt = val_tgt.slice(ndarray::s![.., i..i + 1]).to_owned();
                    (Some(target_val_seq), Some(target_val_tgt))
                } else {
                    (None, None)
                };

            match model
                .train(
                    sequences,
                    &single_target,
                    config,
                    target_val_sequences.as_ref(),
                    target_val_targets.as_ref(),
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

    /// Check if all models are trained (have networks)
    pub fn is_trained(&self) -> bool {
        // This would need to be implemented in LSTMModel to check if network exists
        // For now, assume trained if models exist
        !self.models.is_empty()
    }

    /// EXTRACT MODELS: Get individual LSTM models from MultiTargetLSTMModel wrapper
    /// PURPOSE: Used in training loop to extract single-target models for combination
    /// FLOW: Single-target wrapper → extract LSTM → add to collection → combine all
    pub fn extract_models(self) -> Result<Vec<LSTMModel>> {
        Ok(self.models)
    }

    /// COMBINE MODELS: Create MultiTargetLSTMModel from multiple pre-trained LSTM models
    /// PURPOSE: Fixes critical bug where only first target was kept
    /// ARCHITECTURE: Vec<LSTMModel> → MultiTargetLSTMModel wrapper (all targets preserved)
    /// USAGE: Called after training all targets separately to combine into final model
    pub fn from_trained_models(
        models: Vec<LSTMModel>,        // Pre-trained LSTM models (one per target)
        target_names: Vec<String>,     // ["price_level_12h", "direction_12h", "volatility_12h"]
        trained_horizons: Vec<String>, // ["12h", "12h", "12h"]
        input_size: usize,             // Feature count (same for all models)
    ) -> Result<Self> {
        let num_targets = models.len();

        if num_targets != target_names.len() {
            return Err(VangaError::ModelError(format!(
                "Model count ({}) doesn't match target names count ({})",
                num_targets,
                target_names.len()
            )));
        }

        log::info!(
            "🔗 Creating MultiTargetLSTMModel from {} pre-trained models: {:?}",
            num_targets,
            target_names
        );

        Ok(Self {
            models,
            target_names,
            input_size,
            num_targets,
            trained_horizons,
            feature_config: None,
            training_config: None,
        })
    }
}
