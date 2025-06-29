// Multi-target LSTM model implementation
// Trains separate LSTM models for each target to overcome rust-lstm single-target limitation

use crate::config::ModelConfig;
use crate::model::lstm_simple::LSTMModel;
use crate::utils::error::{Result, VangaError};
use ndarray::{Array2, Array3, Axis};
use serde::{Deserialize, Serialize};
use std::path::Path;

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
}

/// Serializable state for multi-target model persistence
#[derive(Debug, Clone, Serialize, Deserialize)]
struct MultiTargetModelState {
    target_names: Vec<String>,
    input_size: usize,
    num_targets: usize,
}

impl MultiTargetLSTMModel {
    /// Create new multi-target LSTM model
    pub fn new(
        model_config: &ModelConfig,
        input_size: usize,
        target_names: Vec<String>,
    ) -> Result<Self> {
        let num_targets = target_names.len();

        log::info!(
            "Creating multi-target LSTM model with {} targets: {:?}",
            num_targets,
            target_names
        );

        // Create individual LSTM model for each target
        let mut models = Vec::with_capacity(num_targets);
        for (i, target_name) in target_names.iter().enumerate() {
            log::debug!("Creating LSTM model {} for target: {}", i + 1, target_name);

            // Each model has single output (1) since rust-lstm limitation
            let model = LSTMModel::from_model_config(model_config, input_size, 1)?;
            models.push(model);
        }

        Ok(Self {
            models,
            target_names,
            input_size,
            num_targets,
        })
    }

    /// Train all target models with the provided data
    pub async fn train(&mut self, sequences: &Array3<f64>, targets: &Array2<f64>) -> Result<()> {
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
            match model.train(sequences, &single_target).await {
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

    /// Make predictions using all trained models
    pub async fn predict(&self, sequences: &Array3<f64>) -> Result<Array2<f64>> {
        log::info!(
            "Making multi-target predictions for {} sequences using {} models",
            sequences.shape()[0],
            self.models.len()
        );

        let batch_size = sequences.shape()[0];
        let mut all_predictions = Array2::zeros((batch_size, self.num_targets));

        // Get predictions from each model
        for (i, model) in self.models.iter().enumerate() {
            let target_name = &self.target_names[i];
            log::debug!("Getting predictions from model {}: {}", i + 1, target_name);

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

            let model = LSTMModel::load(&model_path)?;
            models.push(model);
        }

        log::info!(
            "✅ Multi-target model loaded successfully with {} targets",
            state.num_targets
        );

        Ok(Self {
            models,
            target_names: state.target_names,
            input_size: state.input_size,
            num_targets: state.num_targets,
        })
    }

    /// Get target names
    pub fn get_target_names(&self) -> &[String] {
        &self.target_names
    }

    /// Get number of targets
    pub fn get_num_targets(&self) -> usize {
        self.num_targets
    }

    /// Get input size
    pub fn get_input_size(&self) -> usize {
        self.input_size
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

        let result = MultiTargetLSTMModel::new(&model_config, 10, target_names.clone());
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
        let mut model = MultiTargetLSTMModel::new(&model_config, 5, target_names).unwrap();

        // Create test data with wrong target dimensions
        let sequences = Array3::zeros((10, 30, 5)); // [batch, seq_len, features]
        let wrong_targets = Array2::zeros((10, 3)); // Wrong: 3 targets instead of 2

        let result = model.train(&sequences, &wrong_targets).await;
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Target dimension mismatch"));
    }
}
