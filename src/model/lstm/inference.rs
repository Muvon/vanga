//! Inference and prediction pipeline
//!
//! This module contains forward pass, prediction methods,
//! and tensor conversion utilities.

use super::config::LSTMModel;
use crate::targets::TargetType;
use crate::utils::error::{Result, VangaError};

use candle_core::Tensor;
use candle_nn::{ops::dropout, Module, RNN};
use ndarray::{Array2, Array3};

impl LSTMModel {
    pub fn convert_sequences_to_tensors(
        &self,
        sequences: &Array3<f64>,
        targets: &Array2<f64>,
    ) -> Result<(Tensor, Tensor)> {
        // sequences shape: [batch, sequence_length, features] - SAME as original
        // targets shape: [batch, output_size] - SAME as original

        // Use minimum batch size to ensure alignment - SAME logic as original
        let batch_size = std::cmp::min(sequences.shape()[0], targets.shape()[0]);

        log::debug!(
            "Converting batch: {} sequences, {} targets, processing {} samples",
            sequences.shape()[0],
            targets.shape()[0],
            batch_size
        );

        // Convert sequences to proper LSTM input format [batch, sequence_length, features]
        let seq_len = sequences.shape()[1];
        let features = sequences.shape()[2];

        let mut seq_data: Vec<f32> = Vec::with_capacity(batch_size * seq_len * features);
        for batch_idx in 0..batch_size {
            for seq_idx in 0..seq_len {
                for feature_idx in 0..features {
                    seq_data.push(sequences[[batch_idx, seq_idx, feature_idx]] as f32);
                }
            }
        }

        let seq_tensor = Tensor::from_vec(seq_data, (batch_size, seq_len, features), &self.device)
            .map_err(|e| {
                VangaError::ModelError(format!("Sequence tensor conversion failed: {}", e))
            })?;

        // Convert targets - use only first target for rust-lstm compatibility - SAME logic as original
        let target_data: Vec<f32> = (0..batch_size)
            .map(|i| targets[[i, 0]] as f32) // Take first target only (single output)
            .collect();
        let target_tensor =
            Tensor::from_vec(target_data, (batch_size, 1), &self.device).map_err(|e| {
                VangaError::ModelError(format!("Target tensor conversion failed: {}", e))
            })?;

        // Log warning about multi-target limitation - SAME as original
        if targets.shape()[1] > 1 {
            log::warn!(
                "Candle LSTM limitation: Using only first target out of {} targets. Consider implementing separate models for each target or using a different ML library for true multi-target support.",
                targets.shape()[1]
            );
        }

        log::debug!(
            "Training data converted: {} samples with sequence length {} (using single target output instead of {})",
            batch_size,
            seq_len,
            targets.shape()[1]
        );

        Ok((seq_tensor, target_tensor))
    }

    /// Forward pass through multi-layer LSTM network - Enhanced with bidirectional support and dropout
    pub fn forward(&self, input: &Tensor, training: bool) -> Result<Tensor> {
        let forward_lstm_layers = self.lstm_layers.as_ref().ok_or_else(|| {
            VangaError::ModelError("Forward LSTM layers not initialized".to_string())
        })?;

        let output_layer = self
            .output_layer
            .as_ref()
            .ok_or_else(|| VangaError::ModelError("Output layer not initialized".to_string()))?;

        // Check if this is bidirectional
        let is_bidirectional = matches!(
            self.architecture,
            Some(crate::config::model::LSTMArchitecture::BidirectionalLSTM { .. })
        );

        let lstm_output = if is_bidirectional {
            // Bidirectional processing
            let backward_lstm_layers = self.backward_lstm_layers.as_ref().ok_or_else(|| {
                VangaError::ModelError(
                    "Backward LSTM layers not initialized for bidirectional model".to_string(),
                )
            })?;

            // Process each layer bidirectionally
            let mut current_input = input.clone();

            for (layer_idx, (forward_layer, backward_layer)) in forward_lstm_layers
                .iter()
                .zip(backward_lstm_layers.iter())
                .enumerate()
            {
                // Process forward direction
                let forward_states = forward_layer.seq(&current_input)?;
                if forward_states.is_empty() {
                    return Err(VangaError::ModelError(format!(
                        "Forward layer {} produced no states",
                        layer_idx
                    )));
                }

                let mut forward_hidden_states = Vec::new();
                for state in &forward_states {
                    forward_hidden_states.push(state.h().clone());
                }
                let forward_output = Tensor::stack(&forward_hidden_states, 1)?.contiguous()?;

                // Process backward direction
                let backward_states = backward_layer.seq(&current_input)?;
                if backward_states.is_empty() {
                    return Err(VangaError::ModelError(format!(
                        "Backward layer {} produced no states",
                        layer_idx
                    )));
                }

                let mut backward_hidden_states = Vec::new();
                for state in &backward_states {
                    backward_hidden_states.push(state.h().clone());
                }
                let backward_output = Tensor::stack(&backward_hidden_states, 1)?.contiguous()?;

                // Concatenate forward and backward outputs along the feature dimension
                // forward_output: [batch_size, seq_len, hidden_size]
                // backward_output: [batch_size, seq_len, hidden_size]
                // Result: [batch_size, seq_len, 2*hidden_size]
                current_input =
                    Tensor::cat(&[&forward_output, &backward_output], 2)?.contiguous()?;

                // Apply dropout between layers if enabled and in training mode
                if training
                    && self.dropout_config.as_ref().is_some_and(|d| d.enabled)
                    && layer_idx < forward_lstm_layers.len() - 1
                {
                    current_input = self.apply_dropout(&current_input)?;
                }

                log::debug!(
                    "Bidirectional layer {} - Forward: {:?}, Backward: {:?}, Concatenated: {:?}",
                    layer_idx,
                    forward_output.shape(),
                    backward_output.shape(),
                    current_input.shape()
                );
            }

            current_input
        } else {
            // Unidirectional processing (original logic)
            let mut current_output = input.clone();
            for (i, lstm_layer) in forward_lstm_layers.iter().enumerate() {
                let layer_states = lstm_layer.seq(&current_output)?;

                if layer_states.is_empty() {
                    return Err(VangaError::ModelError(format!(
                        "Layer {} produced no states",
                        i
                    )));
                }

                let mut hidden_states = Vec::new();
                for state in &layer_states {
                    hidden_states.push(state.h().clone());
                }

                current_output = Tensor::stack(&hidden_states, 1)?.contiguous()?;

                // Apply dropout between layers if enabled and in training mode
                if training
                    && self.dropout_config.as_ref().is_some_and(|d| d.enabled)
                    && i < forward_lstm_layers.len() - 1
                {
                    current_output = self.apply_dropout(&current_output)?;
                }

                log::debug!(
                    "Unidirectional layer {} output shape: {:?}",
                    i,
                    current_output.shape()
                );

                // Validate output dimensions
                let output_shape = current_output.shape();
                if output_shape.dims().len() != 3 {
                    return Err(VangaError::ModelError(format!(
                        "Layer {} output has wrong dimensions: expected 3D tensor, got {:?}",
                        i, output_shape
                    )));
                }
            }
            current_output
        };

        // Apply attention if enabled
        let final_output = if self.use_attention && self.attention_layers.is_some() {
            let attention = self.attention_layers.as_ref().unwrap();

            // Ensure LSTM output is contiguous before passing to attention
            let contiguous_lstm_output = lstm_output.contiguous()?;
            let (attended_output, _attention_weights) =
                attention.forward(&contiguous_lstm_output)?;

            // For sequence-to-one prediction, take the last timestep from attended output
            let seq_len = attended_output.dim(1).map_err(|e| {
                VangaError::ModelError(format!("Failed to get attended sequence length: {}", e))
            })?;

            attended_output
                .narrow(1, seq_len - 1, 1)
                .map_err(|e| {
                    VangaError::ModelError(format!(
                        "Failed to extract last timestep from attended output: {}",
                        e
                    ))
                })?
                .contiguous()?
                .squeeze(1)
                .map_err(|e| {
                    VangaError::ModelError(format!(
                        "Failed to squeeze attended last timestep: {}",
                        e
                    ))
                })?
                .contiguous()?
        } else {
            // Standard LSTM: For sequence-to-one prediction, we need the last timestep
            // LSTM output should be [batch_size, seq_len, hidden_size] or [batch_size, seq_len, 2*hidden_size] for bidirectional
            let seq_len = lstm_output.dim(1).map_err(|e| {
                VangaError::ModelError(format!("Failed to get sequence length: {}", e))
            })?;

            // Take the last timestep hidden state for sequence-to-one prediction
            lstm_output
                .narrow(1, seq_len - 1, 1)?
                .contiguous()?
                .squeeze(1)?
                .contiguous()
                .map_err(|e| {
                    VangaError::ModelError(format!("Failed to squeeze last timestep: {}", e))
                })?
        };

        // Apply output layer to final hidden state
        let predictions = output_layer
            .forward(&final_output)
            .map_err(|e| VangaError::ModelError(format!("Output layer forward failed: {}", e)))?;

        log::debug!(
            "Forward pass complete: input_shape={:?}, final_output_shape={:?}, predictions_shape={:?}, bidirectional={}",
            input.shape(),
            final_output.shape(),
            predictions.shape(),
            is_bidirectional
        );

        Ok(predictions)
    }

    pub async fn predict(&self, sequences: &Array3<f64>) -> Result<Array2<f64>> {
        log::info!("Making predictions with LSTM model");

        // Check if network is trained - SAME logic as original
        if !self.trained {
            return Err(VangaError::ModelError(
                "Network not initialized - cannot make predictions".to_string(),
            ));
        }

        // Ensure network is initialized (defensive programming for loaded models)
        if self.lstm_layers.is_none() || self.output_layer.is_none() {
            return Err(VangaError::ModelError(
                "LSTM network not properly initialized - model may not be loaded correctly"
                    .to_string(),
            ));
        }

        // Convert sequences to tensor (prediction-optimized version)
        let input_tensor = self.convert_sequences_to_prediction_tensor(sequences)?;

        // Forward pass through network (inference mode - no dropout)
        let predictions_tensor = self.forward(&input_tensor, false)?;

        // CRITICAL FIX: Handle multi-class outputs for categorical targets
        let final_predictions_tensor = if let Some((_, target_type)) = &self.target_context {
            log::debug!("Target context found: {:?}", target_type);
            match target_type {
                TargetType::PriceLevel => {
                    // For Price Level: Keep multi-class probabilities (don't convert to indices)
                    let tensor_shape = predictions_tensor.shape();
                    log::debug!("Price Level prediction shape: {:?}", tensor_shape);
                    if tensor_shape.dims().len() == 2 && tensor_shape.dims()[1] > 1 {
                        log::info!(
                            "Keeping Price Level multi-class output {:?} as probabilities",
                            tensor_shape
                        );
                        // Return the full probability distribution for multi-target parsing
                        predictions_tensor
                    } else {
                        log::debug!(
                            "Price Level output already in correct shape: {:?}",
                            tensor_shape
                        );
                        predictions_tensor
                    }
                }
                TargetType::Direction => {
                    // For Direction: Keep multi-class probabilities (don't convert to indices)
                    let tensor_shape = predictions_tensor.shape();
                    log::debug!("Direction prediction shape: {:?}", tensor_shape);
                    if tensor_shape.dims().len() == 2 && tensor_shape.dims()[1] > 1 {
                        log::info!(
                            "Keeping Direction multi-class output {:?} as probabilities",
                            tensor_shape
                        );
                        // Return the full probability distribution for multi-target parsing
                        predictions_tensor
                    } else {
                        predictions_tensor
                    }
                }
                TargetType::Volatility => {
                    // For Volatility: Keep multi-class probabilities (don't convert to indices)
                    let tensor_shape = predictions_tensor.shape();
                    log::debug!("Volatility prediction shape: {:?}", tensor_shape);
                    if tensor_shape.dims().len() == 2 && tensor_shape.dims()[1] > 1 {
                        log::info!(
                            "Keeping Volatility multi-class output {:?} as probabilities",
                            tensor_shape
                        );
                        // Return the full probability distribution for multi-target parsing
                        predictions_tensor
                    } else {
                        predictions_tensor
                    }
                }
            }
        } else {
            // No target context - keep multi-class outputs as probabilities for multi-target parsing
            let tensor_shape = predictions_tensor.shape();
            log::warn!(
                "No target context set during prediction! Tensor shape: {:?}",
                tensor_shape
            );

            if tensor_shape.dims().len() == 2 && tensor_shape.dims()[1] > 1 {
                log::info!(
                    "Auto-detecting multi-class output {:?}, keeping as probabilities for multi-target parsing",
                    tensor_shape
                );
                // Keep the full probability distribution for multi-target parsing
                predictions_tensor
            } else {
                predictions_tensor
            }
        };

        // Convert back to ndarray
        let predictions = self.tensor_to_array2(&final_predictions_tensor)?;

        // Explicit memory cleanup for prediction tensors
        drop(input_tensor);
        // Note: predictions_tensor and final_predictions_tensor are dropped automatically

        log::info!("Generated {} predictions", predictions.nrows());
        Ok(predictions)
    }

    /// Convert sequences to tensor for prediction (memory-optimized, no targets needed)
    fn convert_sequences_to_prediction_tensor(&self, sequences: &Array3<f64>) -> Result<Tensor> {
        let batch_size = sequences.shape()[0];
        let seq_len = sequences.shape()[1];
        let features = sequences.shape()[2];

        log::debug!(
            "Converting prediction batch: {} sequences with {} features, sequence length {}",
            batch_size,
            features,
            seq_len
        );

        // Pre-allocate vector with exact capacity to avoid reallocations
        let mut seq_data: Vec<f32> = Vec::with_capacity(batch_size * seq_len * features);

        // Convert sequences to proper LSTM input format [batch, sequence_length, features]
        for batch_idx in 0..batch_size {
            for seq_idx in 0..seq_len {
                for feature_idx in 0..features {
                    seq_data.push(sequences[[batch_idx, seq_idx, feature_idx]] as f32);
                }
            }
        }

        // Create tensor and immediately drop the vector to free memory
        let seq_tensor = Tensor::from_vec(seq_data, (batch_size, seq_len, features), &self.device)
            .map_err(|e| {
                VangaError::ModelError(format!("Prediction tensor conversion failed: {}", e))
            })?;

        log::debug!(
            "Prediction tensor created: shape {:?}, memory usage: ~{} MB",
            seq_tensor.shape(),
            (batch_size * seq_len * features * 4) / 1_048_576 // 4 bytes per f32, convert to MB
        );

        Ok(seq_tensor)
    }

    /// Convert Candle tensor to ndarray Array2 - helper method (memory-optimized)
    fn tensor_to_array2(&self, tensor: &Tensor) -> Result<Array2<f64>> {
        // Get tensor shape
        let shape = tensor.shape();
        if shape.dims().len() != 2 {
            return Err(VangaError::ModelError(format!(
                "Expected 2D tensor, got {}D tensor with shape {:?}",
                shape.dims().len(),
                shape.dims()
            )));
        }

        let rows = shape.dims()[0];
        let cols = shape.dims()[1];

        // Convert tensor to Vec<f32> then to f64
        let data: Vec<f32> = tensor
            .flatten_all()
            .map_err(|e| VangaError::ModelError(format!("Failed to flatten tensor: {}", e)))?
            .to_vec1()
            .map_err(|e| {
                VangaError::ModelError(format!("Failed to convert tensor to vec: {}", e))
            })?;

        // Convert f32 to f64 and create Array2
        let data_f64: Vec<f64> = data.iter().map(|&x| x as f64).collect();

        // Explicit cleanup of intermediate data
        drop(data);

        Array2::from_shape_vec((rows, cols), data_f64)
            .map_err(|e| VangaError::ModelError(format!("Failed to create Array2: {}", e)))
    }

    /// Apply dropout with proper rate calculation based on configuration
    fn apply_dropout(&self, tensor: &Tensor) -> Result<Tensor> {
        let dropout_config = self
            .dropout_config
            .as_ref()
            .ok_or_else(|| VangaError::ModelError("Dropout configuration not set".to_string()))?;

        // Calculate dropout rate based on configuration
        let dropout_rate = match &dropout_config.rate {
            crate::config::model::DropoutRate::Fixed(rate) => *rate,
            crate::config::model::DropoutRate::Auto { min_rate, max_rate } => {
                // Use middle value for auto rate (could be enhanced with adaptive logic)
                (min_rate + max_rate) / 2.0
            }
            crate::config::model::DropoutRate::Adaptive => {
                // Default adaptive rate - could be enhanced based on training progress
                0.2
            }
        };

        // Apply dropout using candle's dropout function
        let dropped_tensor = dropout(tensor, dropout_rate as f32)?;

        log::debug!(
            "Applied dropout with rate {:.3} to tensor shape {:?}",
            dropout_rate,
            tensor.shape()
        );

        Ok(dropped_tensor)
    }
}
