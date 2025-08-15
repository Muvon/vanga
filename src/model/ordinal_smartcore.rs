//! Ordinal-aware wrapper for SmartCore backend
//!
//! This module implements ordinal loss awareness for SmartCore models
//! by using sample weighting and post-processing to simulate the
//! trading-aware ordinal loss used in LSTM models.

use crate::config::model::XGBoostConfig;
use crate::utils::error::{Result, VangaError};

use candle_core::{Device, Tensor};
use ndarray::{Array1, Array2};

/// Trading-aware penalty matrix (same as LSTM)
/// Rows = true class, Columns = predicted class
/// Higher values = worse mistakes for trading
pub const ORDINAL_PENALTY_MATRIX: [[f32; 5]; 5] = [
    // True = 0 (VeryDown)
    [0.0, 0.3, 0.5, 1.5, 2.0], // Predicting UP in crash is worst
    // True = 1 (Down)
    [0.3, 0.0, 0.3, 1.2, 1.8], // Still bad to predict UP
    // True = 2 (Sideways)
    [0.8, 0.4, 0.0, 0.4, 0.8], // Symmetric penalties for unnecessary trades
    // True = 3 (Up)
    [1.8, 1.2, 0.3, 0.0, 0.3], // Bad to predict DOWN
    // True = 4 (VeryUp)
    [2.0, 1.5, 0.5, 0.3, 0.0], // Shorting in rally is worst
];

/// Ordinal-aware SmartCore wrapper
/// This struct provides ordinal-aware functionality without containing the backend
pub struct OrdinalSmartCore {
    /// Configuration for the model
    config: XGBoostConfig,

    /// Sample weights for ordinal-aware training
    sample_weights: Option<Vec<f32>>,

    /// Last training targets for weight calculation
    last_targets: Option<Vec<i32>>,

    /// Lambda parameter for ordinal penalty weight (default: 0.3)
    lambda: f32,
}

impl OrdinalSmartCore {
    /// Create new ordinal-aware SmartCore model
    pub fn new(config: XGBoostConfig) -> Self {
        log::info!("🎯 Creating Ordinal-Aware SmartCore model with trading penalty matrix");
        Self {
            config,
            sample_weights: None,
            last_targets: None,
            lambda: 0.3, // Same as LSTM
        }
    }

    /// Calculate sample weights based on ordinal penalties
    /// This simulates ordinal loss by weighting samples during training
    fn calculate_ordinal_weights(
        &mut self,
        targets: &[i32],
        predictions: Option<&[i32]>,
    ) -> Vec<f32> {
        let num_samples = targets.len();
        let mut weights = vec![1.0f32; num_samples];

        // If we have predictions (iterative training), calculate penalties
        if let Some(preds) = predictions {
            // Log ordinal penalty distribution for analysis
            self.log_ordinal_penalties(targets, preds);

            for i in 0..num_samples {
                let true_class = targets[i] as usize;
                let pred_class = preds[i] as usize;

                if true_class < 5 && pred_class < 5 {
                    let penalty = ORDINAL_PENALTY_MATRIX[true_class][pred_class];
                    // Convert penalty to weight: higher penalty = higher weight for this sample
                    // This makes the model focus more on samples with high ordinal penalties
                    weights[i] = 1.0 + self.lambda * penalty;
                }
            }

            // Normalize weights to sum to num_samples (maintain overall scale)
            let sum: f32 = weights.iter().sum();
            let scale = num_samples as f32 / sum;
            for w in weights.iter_mut() {
                *w *= scale;
            }

            log::debug!(
                "📊 Ordinal weights calculated: min={:.3}, max={:.3}, mean={:.3}",
                weights.iter().fold(f32::INFINITY, |a, &b| a.min(b)),
                weights.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b)),
                weights.iter().sum::<f32>() / num_samples as f32
            );
        }

        weights
    }

    /// Apply ordinal constraints to predictions (post-processing)
    /// Ensures predictions respect ordinal relationships
    fn apply_ordinal_constraints(&self, probabilities: &mut Array2<f32>) {
        let num_samples = probabilities.nrows();
        let num_classes = probabilities.ncols();

        if num_classes != 5 {
            return; // Only apply to 5-class ordinal problems
        }

        // Apply smoothing to respect ordinal relationships
        for i in 0..num_samples {
            let mut row = probabilities.row_mut(i);

            // Find the peak probability class
            let mut max_prob = 0.0f32;
            let mut max_class = 0;
            for (j, &prob) in row.iter().enumerate() {
                if prob > max_prob {
                    max_prob = prob;
                    max_class = j;
                }
            }

            // Apply Gaussian-like smoothing around the peak
            // This ensures nearby classes get some probability mass
            for j in 0..num_classes {
                let distance = (j as i32 - max_class as i32).abs() as f32;
                let smoothing_factor = (-0.5 * distance * distance).exp();
                row[j] = row[j] * 0.7 + max_prob * smoothing_factor * 0.3;
            }

            // Renormalize to sum to 1
            let sum: f32 = row.iter().sum();
            if sum > 0.0 {
                for j in 0..num_classes {
                    row[j] /= sum;
                }
            }
        }
    }

    /// Calculate ordinal weights for training samples
    /// Returns weights that can be used by the training algorithm
    pub fn calculate_ordinal_weights_for_training(
        &mut self,
        targets: &Tensor,
        predictions: Option<&Tensor>,
    ) -> Result<Vec<f32>> {
        // Convert targets to array
        let targets_array = self.tensor_to_array1(targets)?;
        let target_classes: Vec<i32> = targets_array.iter().map(|&x| x as i32).collect();

        // Get predictions if available
        let pred_classes = if let Some(preds) = predictions {
            Some(self.get_predicted_classes(preds)?)
        } else {
            None
        };

        // Calculate weights
        let weights = self.calculate_ordinal_weights(&target_classes, pred_classes.as_deref());
        self.sample_weights = Some(weights.clone());
        self.last_targets = Some(target_classes);

        Ok(weights)
    }

    /// Apply ordinal constraints to prediction probabilities
    pub fn apply_ordinal_constraints_to_predictions(&self, predictions: &Tensor) -> Result<Tensor> {
        let mut probs_array = self.tensor_to_array2(predictions)?;
        self.apply_ordinal_constraints(&mut probs_array);
        self.array2_to_tensor(&probs_array, predictions.device())
    }

    /// Calculate ordinal loss for evaluation (same as LSTM)
    ///
    /// Mathematical Framework (per paper):
    /// - Input: predictions ŷ ∈ ℝ^(N×C) where N=samples, C=5 classes
    /// - Target: y ∈ ℝ^N class indices [0,1,2,3,4]
    /// - Loss: L = CE(ŷ,y) + λ * Ordinal_Penalty(ŷ,y)
    ///
    /// Where:
    /// - CE(ŷ,y) = -log(ŷ[y]) (cross-entropy)
    /// - Ordinal_Penalty = Σ_c ŷ[c] * P[y,c] (trading-aware penalty)
    /// - P[y,c] = ORDINAL_PENALTY_MATRIX[y][c] (asymmetric trading penalties)
    /// - λ = 0.3 (penalty weight, same as LSTM)
    pub fn calculate_ordinal_loss(&self, predictions: &Tensor, targets: &Tensor) -> Result<f32> {
        let pred_array = self.tensor_to_array2(predictions)?;
        let target_array = self.tensor_to_array1(targets)?;

        let num_samples = pred_array.nrows();
        let mut total_loss = 0.0f32;

        for i in 0..num_samples {
            let true_class = target_array[i] as usize;
            let pred_probs = pred_array.row(i);

            // Calculate cross-entropy loss
            let mut ce_loss = 0.0f32;
            if true_class < 5 {
                let true_prob = pred_probs[true_class].max(1e-7);
                ce_loss = -true_prob.ln();
            }

            // Calculate ordinal penalty
            let mut ordinal_penalty = 0.0f32;
            for (pred_class, &prob) in pred_probs.iter().enumerate() {
                if true_class < 5 && pred_class < 5 {
                    ordinal_penalty += prob * ORDINAL_PENALTY_MATRIX[true_class][pred_class];
                }
            }

            // Combine losses (same formula as LSTM)
            total_loss += ce_loss + self.lambda * ordinal_penalty;
        }

        Ok(total_loss / num_samples as f32)
    }

    /// Get predicted classes from probability tensor
    fn get_predicted_classes(&self, predictions: &Tensor) -> Result<Vec<i32>> {
        let pred_array = self.tensor_to_array2(predictions)?;
        let mut classes = Vec::new();

        for i in 0..pred_array.nrows() {
            let row = pred_array.row(i);
            let mut max_prob = 0.0f32;
            let mut max_class = 0i32;

            for (j, &prob) in row.iter().enumerate() {
                if prob > max_prob {
                    max_prob = prob;
                    max_class = j as i32;
                }
            }
            classes.push(max_class);
        }

        Ok(classes)
    }

    /// Log ordinal penalty distribution for debugging
    fn log_ordinal_penalties(&self, targets: &[i32], predictions: &[i32]) {
        let mut penalty_counts = [[0u32; 5]; 5];
        let mut total_penalty = 0.0f32;
        let mut count = 0u32;

        for (i, &true_class) in targets.iter().enumerate() {
            if let Some(&pred_class) = predictions.get(i) {
                let true_idx = true_class as usize;
                let pred_idx = pred_class as usize;

                if true_idx < 5 && pred_idx < 5 {
                    penalty_counts[true_idx][pred_idx] += 1;
                    total_penalty += ORDINAL_PENALTY_MATRIX[true_idx][pred_idx];
                    count += 1;
                }
            }
        }

        if count > 0 {
            let avg_penalty = total_penalty / count as f32;
            log::info!(
                "📊 Ordinal Penalty Stats: Average={:.3}, Total samples={}",
                avg_penalty,
                count
            );

            // Log confusion matrix with penalties
            log::debug!("🎯 Ordinal Confusion Matrix (True vs Predicted):");
            for (i, row) in penalty_counts.iter().enumerate() {
                let row_str: String = row
                    .iter()
                    .map(|count| format!("{:4}", count))
                    .collect::<Vec<_>>()
                    .join(" ");
                log::debug!("  Class {}: {}", i, row_str);
            }
        }
    }

    // Helper methods for tensor/array conversion
    fn tensor_to_array1(&self, tensor: &Tensor) -> Result<Array1<f32>> {
        let shape = tensor.shape();

        let data: Vec<f32> = if shape.dims().len() == 1 {
            tensor
                .to_vec1::<f32>()
                .map_err(|e| VangaError::model(format!("Failed to convert 1D tensor: {}", e)))?
        } else if shape.dims().len() == 2 && shape.dims()[1] == 1 {
            // Flatten [N,1] -> [N]
            tensor
                .to_vec2::<f32>()
                .map_err(|e| VangaError::model(format!("Failed to convert 2D tensor: {}", e)))?
                .into_iter()
                .flatten()
                .collect()
        } else if shape.dims().len() == 2 && shape.dims()[1] > 1 {
            // One-hot [N,C] -> argmax indices [N]
            log::info!(
                "🔄 Converting one-hot encoded targets to class indices for OrdinalSmartCore"
            );
            let targets_2d = tensor
                .to_vec2::<f32>()
                .map_err(|e| VangaError::model(format!("Failed to convert 2D tensor: {}", e)))?;

            let mut class_indices = Vec::new();
            for row in targets_2d {
                let (max_idx, max_val) = row
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .unwrap();

                if *max_val < 0.5 {
                    log::warn!(
                        "⚠️ Target row {:?} doesn't look like one-hot encoding (max={:.3})",
                        row,
                        max_val
                    );
                }

                class_indices.push(max_idx as f32);
            }

            class_indices
        } else {
            return Err(VangaError::model(format!(
                "Cannot convert tensor with shape {:?} to 1D array",
                shape
            )));
        };

        Ok(Array1::from_vec(data))
    }

    fn tensor_to_array2(&self, tensor: &Tensor) -> Result<Array2<f32>> {
        let shape = tensor.shape();
        let data = tensor.to_vec2::<f32>()?;
        let rows = shape.dims()[0];
        let cols = if shape.dims().len() > 1 {
            shape.dims()[1]
        } else {
            1
        };

        let mut array = Array2::zeros((rows, cols));
        for (i, row_data) in data.iter().enumerate() {
            for (j, &val) in row_data.iter().enumerate() {
                array[[i, j]] = val;
            }
        }
        Ok(array)
    }

    fn array2_to_tensor(&self, array: &Array2<f32>, device: &Device) -> Result<Tensor> {
        let shape = array.dim();
        let data: Vec<f32> = array.iter().cloned().collect();
        Ok(Tensor::from_vec(data, (shape.0, shape.1), device)?)
    }

    /// Update feature dimension to match actual LSTM output
    pub fn update_feature_dimension(&mut self, actual_feature_dim: usize) {
        self.config.feature_dim = actual_feature_dim;
    }
}

/// Public function to get ordinal penalty for a given true/predicted class pair
pub fn get_ordinal_penalty(true_class: usize, pred_class: usize) -> f32 {
    if true_class < 5 && pred_class < 5 {
        ORDINAL_PENALTY_MATRIX[true_class][pred_class]
    } else {
        0.0
    }
}
