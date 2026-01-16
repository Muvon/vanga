//! Inference and prediction pipeline
//!
//! This module contains forward pass, prediction methods,
//! and tensor conversion utilities.

use super::config::LSTMModel;
use crate::targets::TargetType;
use crate::utils::error::{Result, VangaError};

use candle_core::Tensor;
use candle_nn::{Module, RNN};
use ndarray::{s, Array2, Array3};

// Import deterministic dropout
use super::seeded_weights::SeededTensorUtils;

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

        // CRITICAL: Log device being used for tensor creation
        let device_name = match &self.device {
            candle_core::Device::Cpu => "CPU",
            candle_core::Device::Cuda(_) => "CUDA GPU",
            candle_core::Device::Metal(_) => "Metal GPU",
        };
        log::debug!("🔧 Creating sequence tensor on device: {}", device_name);

        let seq_tensor = Tensor::from_vec(seq_data, (batch_size, seq_len, features), &self.device)
            .map_err(|e| {
                VangaError::ModelError(format!("Sequence tensor conversion failed: {}", e))
            })?;

        // Convert targets - handle both one-hot encoded and raw class indices
        let num_target_cols = targets.shape()[1];
        let target_data: Vec<f32> = if num_target_cols > 1 {
            // Assume one-hot encoded targets - convert to class indices
            log::debug!(
                "Converting one-hot encoded targets ({} classes) to class indices",
                num_target_cols
            );
            (0..batch_size)
                .map(|i| {
                    // Find which class is hot (has value 1.0 or highest value)
                    let mut max_val = -1.0;
                    let mut max_idx = 0;
                    for class_idx in 0..num_target_cols {
                        let val = targets[[i, class_idx]];
                        if val > max_val {
                            max_val = val;
                            max_idx = class_idx;
                        }
                    }
                    max_idx as f32
                })
                .collect()
        } else {
            // Already class indices - just convert to f32
            log::debug!("Using raw class indices (already in correct format)");
            (0..batch_size).map(|i| targets[[i, 0]] as f32).collect()
        };

        // Log sample of converted targets for verification
        if batch_size > 0 {
            let sample_size = std::cmp::min(5, batch_size);
            let sample_targets: Vec<f32> = target_data.iter().take(sample_size).copied().collect();
            log::debug!(
                "Sample converted targets (first {} values): {:?}",
                sample_size,
                sample_targets
            );
        }

        log::debug!("🔧 Creating target tensor on device: {}", device_name);
        let target_tensor =
            Tensor::from_vec(target_data, (batch_size, 1), &self.device).map_err(|e| {
                VangaError::ModelError(format!("Target tensor conversion failed: {}", e))
            })?;

        log::debug!(
            "Training data converted: {} samples with sequence length {} (converted {} target columns to class indices)",
            batch_size,
            seq_len,
            targets.shape()[1]
        );

        Ok((seq_tensor, target_tensor))
    }

    /// Forward pass through multi-layer LSTM network - Enhanced with bidirectional support and consistent dropout
    ///
    /// CRITICAL FIX: Dropout consistency between training and validation
    /// - training=true: Apply dropout for regularization during training
    /// - training=false: NO dropout for consistent validation behavior
    pub fn forward(&self, input: &Tensor, training: bool) -> Result<Tensor> {
        // CRITICAL: Verify device before forward pass
        let device_name = match &self.device {
            candle_core::Device::Cpu => "CPU",
            candle_core::Device::Cuda(_) => "CUDA GPU",
            candle_core::Device::Metal(_) => "Metal GPU",
        };
        let input_device_name = match input.device() {
            candle_core::Device::Cpu => "CPU",
            candle_core::Device::Cuda(_) => "CUDA GPU",
            candle_core::Device::Metal(_) => "Metal GPU",
        };
        log::debug!(
            "🔍 Forward pass - model.device: {}, input.device: {}",
            device_name,
            input_device_name
        );

        // Apply DAIN normalization if enabled
        let normalized_input = if let Some(ref dain) = self.dain_normalization {
            let input_f64 = input.to_dtype(candle_core::DType::F64)?;
            let normalized = dain.forward(&input_f64)?;
            log::debug!("🎯 Applied DAIN normalization to input");
            normalized.to_dtype(input.dtype())?
        } else {
            input.clone()
        };

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
            let mut current_input = normalized_input.clone();

            for (layer_idx, (forward_layer, backward_layer)) in forward_lstm_layers
                .iter()
                .zip(backward_lstm_layers.iter())
                .enumerate()
            {
                // CRITICAL FIX: Use seq_init() with zero_state() for validation to prevent hidden state contamination
                // During training, we want to maintain states for temporal learning
                // During validation, we want fresh states for each batch to get accurate metrics
                let forward_states = if training {
                    // Training mode: maintain hidden states between sequences (good for learning)
                    forward_layer.seq(&current_input)?
                } else {
                    // Validation mode: reset hidden states for each batch (prevents contamination)
                    let batch_size = current_input.dim(0)?;
                    let zero_state = forward_layer.zero_state(batch_size)?;
                    forward_layer.seq_init(&current_input, &zero_state)?
                };

                if forward_states.is_empty() {
                    return Err(VangaError::ModelError(format!(
                        "Forward layer {} produced no states",
                        layer_idx
                    )));
                }

                let forward_hidden_states: Vec<Tensor> = forward_states
                    .iter()
                    .map(|state| state.h().clone())
                    .collect();
                let forward_output = Tensor::stack(&forward_hidden_states, 1)?.contiguous()?;
                drop(forward_hidden_states);
                drop(forward_states);

                // Process backward direction with same logic
                let backward_states = if training {
                    // Training mode: maintain hidden states
                    backward_layer.seq(&current_input)?
                } else {
                    // Validation mode: reset hidden states
                    let batch_size = current_input.dim(0)?;
                    let zero_state = backward_layer.zero_state(batch_size)?;
                    backward_layer.seq_init(&current_input, &zero_state)?
                };

                if backward_states.is_empty() {
                    return Err(VangaError::ModelError(format!(
                        "Backward layer {} produced no states",
                        layer_idx
                    )));
                }

                let backward_hidden_states: Vec<Tensor> = backward_states
                    .iter()
                    .map(|state| state.h().clone())
                    .collect();
                let backward_output = Tensor::stack(&backward_hidden_states, 1)?.contiguous()?;
                drop(backward_hidden_states);
                drop(backward_states);

                current_input =
                    Tensor::cat(&[&forward_output, &backward_output], 2)?.contiguous()?;
                drop(forward_output);
                drop(backward_output);

                // Apply Layer Normalization if enabled (Ba et al., 2016)
                // LayerNorm stabilizes training in deep LSTMs
                // CRITICAL FIX: Use configured position (Pre or Post)
                if let Some(ref config) = self.layer_norm_config {
                    if config.enabled && config.lstm_cell {
                        let position = config.position;
                        log::debug!(
                            "Applying LayerNorm (position: {:?}, layer: {})",
                            position,
                            layer_idx
                        );
                        current_input =
                            self.apply_layer_norm(&current_input, config, layer_idx, position)?;
                    }
                }

                // Apply consistent dropout between layers if enabled AND in training mode
                let should_apply_dropout = if let Some(dropout_config) = &self.dropout_config {
                    dropout_config.enabled && training // Only apply dropout during training
                } else {
                    false
                };

                if should_apply_dropout {
                    // Generate sequence ID for variational dropout consistency
                    let sequence_id = format!("bidirectional_layer_{}", layer_idx);
                    current_input = self.apply_dropout_with_context(
                        &current_input,
                        training,
                        Some(&sequence_id),
                    )?;
                    log::debug!(
                        "🔧 Applied LSTM layer dropout (layer: {}, training: {}, sequence: {})",
                        layer_idx,
                        training,
                        sequence_id
                    );
                }

                // Track dropout behavior in metrics collector if available
                // Note: This is a simplified tracking - in practice, you'd need access to the metrics collector
                // which would require passing it through the forward pass or storing it in the model

                log::debug!(
                    "Bidirectional layer {} - Concatenated: {:?}",
                    layer_idx,
                    current_input.shape()
                );
            }

            current_input
        } else {
            // Unidirectional processing (original logic)
            let mut current_output = normalized_input.clone();
            for (i, lstm_layer) in forward_lstm_layers.iter().enumerate() {
                // CRITICAL FIX: Use seq_init() with zero_state() for validation to prevent hidden state contamination
                // During training, we want to maintain states for temporal learning
                // During validation, we want fresh states for each batch to get accurate metrics
                let layer_states = if training {
                    // Training mode: maintain hidden states between sequences (good for learning)
                    lstm_layer.seq(&current_output)?
                } else {
                    // Validation mode: reset hidden states for each batch (prevents contamination)
                    let batch_size = current_output.dim(0)?;
                    let zero_state = lstm_layer.zero_state(batch_size)?;
                    lstm_layer.seq_init(&current_output, &zero_state)?
                };

                if layer_states.is_empty() {
                    return Err(VangaError::ModelError(format!(
                        "Layer {} produced no states",
                        i
                    )));
                }

                let hidden_states: Vec<Tensor> =
                    layer_states.iter().map(|state| state.h().clone()).collect();
                current_output = Tensor::stack(&hidden_states, 1)?.contiguous()?;
                drop(hidden_states);
                drop(layer_states);

                // Apply Layer Normalization if enabled (Ba et al., 2016)
                // LayerNorm stabilizes training in deep LSTMs by normalizing activations
                // across features for each sample independently
                // CRITICAL FIX: Use configured position (Pre or Post)
                if let Some(ref config) = self.layer_norm_config {
                    if config.enabled && config.lstm_cell {
                        let position = config.position;
                        log::debug!(
                            "Applying LayerNorm (position: {:?}, layer: {})",
                            position,
                            i
                        );
                        current_output =
                            self.apply_layer_norm(&current_output, config, i, position)?;
                    }
                }

                // Apply consistent dropout between layers if enabled AND in training mode
                let should_apply_dropout = self
                    .dropout_config
                    .as_ref()
                    .map(|d| d.enabled && training) // Only apply dropout during training
                    .unwrap_or(false);

                if should_apply_dropout && i < forward_lstm_layers.len() - 1 {
                    // Generate sequence ID for variational dropout consistency
                    let sequence_id = format!("unidirectional_layer_{}", i);
                    current_output = self.apply_dropout_with_context(
                        &current_output,
                        training,
                        Some(&sequence_id),
                    )?;
                    log::debug!(
                        "🔧 Applied LSTM layer dropout (layer: {}, training: {}, sequence: {})",
                        i,
                        training,
                        sequence_id
                    );
                }

                // Track dropout behavior in metrics collector if available
                // Note: This is a simplified tracking - in practice, you'd need access to the metrics collector
                // which would require passing it through the forward pass or storing it in the model

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
        let final_output = if self.use_attention {
            if let Some(attention) = &self.attention_module {
                // Ensure LSTM output is contiguous before passing to attention
                let contiguous_lstm_output = lstm_output.contiguous()?;
                let (attended_output, _attention_weights) =
                    attention.forward(&contiguous_lstm_output, training)?;

                let attention_dropout_rate = attention.get_config().dropout_rate;
                log::debug!(
                    "🎯 Applied attention with dropout rate: {:.3}, training: {}",
                    attention_dropout_rate,
                    training
                );

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
                lstm_output
            }
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

        // DEBUG: Check if LSTM hidden states are diverse across samples
        // If hidden states are similar for all samples, output will be similar too
        if !training {
            let hs = final_output.flatten_all()?;
            let hs_data: Vec<f32> = hs.to_vec1()?;
            let sample_size = self
                .config
                .get_hidden_size_for_layer(self.config.num_layers - 1);
            if hs_data.len() >= sample_size * 2 {
                let first_sample = &hs_data[0..sample_size];
                let second_sample = &hs_data[sample_size..sample_size * 2];
                let diff: f32 = first_sample
                    .iter()
                    .zip(second_sample.iter())
                    .map(|(a, b)| (a - b).abs())
                    .sum();
                let avg_diff = diff / sample_size as f32;
                log::debug!(
                    "🔍 Hidden state diversity: avg diff between sample 0 and 1 = {:.6}",
                    avg_diff
                );
            }
        }

        // DEBUG: Check output layer weights (if possible)
        if !training {
            log::debug!("🔍 Checking output layer weights...");
        }

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

        let total_sequences = sequences.shape()[0];
        let batch_size = self.training_config.batch_size;

        // If total sequences fit in one batch, process directly
        if total_sequences <= batch_size {
            return self.predict_batch(sequences).await;
        }

        // Process in batches to avoid OOM
        log::info!(
            "Processing {} sequences in batches of {}",
            total_sequences,
            batch_size
        );

        let num_batches = total_sequences.div_ceil(batch_size);
        let mut all_predictions = Vec::new();

        for batch_idx in 0..num_batches {
            let start_idx = batch_idx * batch_size;
            let end_idx = std::cmp::min(start_idx + batch_size, total_sequences);

            log::debug!(
                "Processing batch {}/{}: sequences {}-{}",
                batch_idx + 1,
                num_batches,
                start_idx,
                end_idx
            );

            let batch_sequences = sequences.slice(ndarray::s![start_idx..end_idx, .., ..]);
            let batch_predictions = self.predict_batch(&batch_sequences.to_owned()).await?;

            all_predictions.push(batch_predictions);
        }

        // Concatenate all batch predictions
        let total_rows: usize = all_predictions.iter().map(|p| p.nrows()).sum();
        let num_cols = all_predictions[0].ncols();

        let mut result = Array2::zeros((total_rows, num_cols));
        let mut current_row = 0;

        for batch_pred in all_predictions {
            let batch_rows = batch_pred.nrows();
            result
                .slice_mut(ndarray::s![current_row..current_row + batch_rows, ..])
                .assign(&batch_pred);
            current_row += batch_rows;
        }

        log::info!(
            "Generated {} predictions across {} batches",
            total_rows,
            num_batches
        );
        Ok(result)
    }

    async fn predict_batch(&self, sequences: &Array3<f64>) -> Result<Array2<f64>> {
        // Convert sequences to tensor (prediction-optimized version)
        let input_tensor = self.convert_sequences_to_prediction_tensor(sequences)?;

        // Hybrid Prediction: Use XGBoost if available, otherwise pure LSTM
        let predictions_tensor = if let Some(xgb_model) = &self.xgboost_model {
            log::info!("🔄 Using hybrid LSTM+XGBoost prediction (as per paper)");

            // Extract LSTM features (hidden state z)
            // Equation (8): z = h_n ∈ ℝ^k
            let lstm_features = self.extract_lstm_features(&input_tensor)?;
            log::info!(
                "📊 [LSTM] Extracted latent vector z = h_n with shape {:?}",
                lstm_features.shape()
            );
            log::debug!("   • z is LSTM hidden state (NOT predictions)");
            log::debug!("   • z ∈ ℝ^k where k = hidden_size");

            // XGBoost prediction: ŷ = f(z)
            // Equation (9): ŷ = f(z) = Σ f_m(z)
            log::info!("🎯 [XGBoost] Computing predictions ŷ = f(z) from LSTM features");
            let xgb_predictions = xgb_model.predict(&lstm_features)?;
            log::info!(
                "📊 [XGBoost] Final predictions ŷ with shape {:?}",
                xgb_predictions.shape()
            );
            log::debug!("   • ŷ are actual predictions (NOT features)");
            log::debug!("   • ŷ ∈ ℝ^(N×5) for 5-class classification");

            // IMPROVEMENT: Blend XGBoost with LSTM predictions for better stability
            // While paper uses pure XGBoost output, blending improves robustness
            let lstm_predictions = self.forward(&input_tensor, false)?;
            log::debug!(
                "📊 LSTM predictions (for blending): {:?}",
                lstm_predictions.shape()
            );

            // Use weighted average: 20% LSTM + 80% XGBoost
            // This preserves some LSTM temporal patterns while leveraging XGBoost's power
            self.blend_predictions(&lstm_predictions, &xgb_predictions, 0.2)?
        } else {
            log::info!("🔄 Using pure LSTM prediction");

            // Forward pass through network (inference mode - no dropout)
            let raw_logits = self.forward(&input_tensor, false)?;

            // CRITICAL: Apply POST-HOC ensemble calibration if available
            // Temperature scaling is applied ONLY during inference, never during training
            // Research: Guo et al. 2017 - temperature scaling is post-processing method
            if let Some(ref ensemble_cal) = self.ensemble_calibrator {
                if ensemble_cal.is_calibrated {
                    // Convert logits tensor to ndarray for temperature scaling
                    let logits_shape = raw_logits.shape();
                    let batch_size = logits_shape.dims()[0];
                    let logits_data: Vec<f64> = raw_logits
                        .flatten_all()?
                        .to_vec1::<f32>()?
                        .iter()
                        .map(|&v| v as f64)
                        .collect();

                    let logits_array = Array2::from_shape_vec((batch_size, 5), logits_data)
                        .map_err(|e| {
                            VangaError::ModelError(format!("Failed to create logits array: {}", e))
                        })?;

                    // Apply temperature scaling to logits
                    let calibrated_logits_array = ensemble_cal.apply_to_logits(&logits_array)?;

                    // Convert back to tensor
                    let calibrated_data: Vec<f32> =
                        calibrated_logits_array.iter().map(|&v| v as f32).collect();

                    let calibrated_logits =
                        Tensor::from_vec(calibrated_data, (batch_size, 5), &self.device)?;

                    log::debug!(
                        "🌡️ Applied temperature scaling (T={:.3}) during inference",
                        ensemble_cal.temperature_scaling.temperature
                    );

                    calibrated_logits
                } else {
                    raw_logits
                }
            } else {
                raw_logits
            }
        };

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
                TargetType::Sentiment => {
                    // For Sentiment: Keep multi-class probabilities (don't convert to indices)
                    let tensor_shape = predictions_tensor.shape();
                    log::debug!("Sentiment prediction shape: {:?}", tensor_shape);
                    if tensor_shape.dims().len() == 2 && tensor_shape.dims()[1] > 1 {
                        log::info!(
                            "Keeping Sentiment multi-class output {:?} as probabilities",
                            tensor_shape
                        );
                        // Return the full probability distribution for multi-target parsing
                        predictions_tensor
                    } else {
                        predictions_tensor
                    }
                }
                TargetType::Volume => {
                    // For Volume: Keep multi-class probabilities (don't convert to indices)
                    let tensor_shape = predictions_tensor.shape();
                    log::debug!("Volume prediction shape: {:?}", tensor_shape);
                    if tensor_shape.dims().len() == 2 && tensor_shape.dims()[1] > 1 {
                        log::info!(
                            "Keeping Volume multi-class output {:?} as probabilities",
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

        // CRITICAL FIX: Apply softmax to convert logits to probabilities before bias correction
        // The output layer produces raw logits, but bias correction expects probabilities
        let predictions_probs = candle_nn::ops::softmax(&final_predictions_tensor, 1)?;

        // Convert back to ndarray
        let mut predictions_array = self.tensor_to_array2(&predictions_probs)?;

        // Apply bias correction using LinearBiasCorrector with proper tensor method
        if let Some(ref corrector) = self.bias_corrector {
            if corrector.is_calibrated {
                log::info!(
                    "🔧 Applying LinearBiasCorrector to {} predictions",
                    predictions_array.nrows()
                );

                // Convert back to tensor for proper bias correction
                let predictions_tensor = self.array2_to_tensor(&predictions_array)?;
                let corrected_tensor = corrector.apply_correction_tensor(&predictions_tensor)?;
                predictions_array = self.tensor_to_array2(&corrected_tensor)?;

                log::info!("✅ Bias correction applied successfully");
            } else {
                log::debug!("ℹ️ LinearBiasCorrector not calibrated - skipping correction");
            }
        } else if let Some(correction_factors) = &self.bias_correction_factors {
            // Fallback to simple bias correction for backward compatibility
            log::info!(
                "🔧 Applying simple bias correction with factors: {:?}",
                correction_factors
            );
            self.apply_simple_bias_correction(&mut predictions_array, correction_factors)?;
            log::info!("✅ Simple bias correction applied successfully");
        } else {
            log::debug!("ℹ️ No bias correction available");
        }

        // Explicit memory cleanup for prediction tensors
        drop(input_tensor);

        log::debug!(
            "Generated {} predictions for batch",
            predictions_array.nrows()
        );
        Ok(predictions_array)
    }

    /// Apply simple bias correction to predictions
    pub fn apply_simple_bias_correction(
        &self,
        predictions: &mut Array2<f64>,
        correction_factors: &[f64; 5],
    ) -> Result<()> {
        // Validate input dimensions
        let num_classes = predictions.ncols();
        if num_classes != 5 {
            log::error!(
                "❌ Expected 5 classes in predictions for simple bias correction, got {}",
                num_classes
            );
            return Err(crate::utils::error::VangaError::ModelError(format!(
                "Simple bias correction requires 5-class predictions, got {}",
                num_classes
            )));
        }

        // Apply correction factors to each class
        for (class_idx, &factor) in correction_factors.iter().enumerate() {
            predictions
                .column_mut(class_idx)
                .mapv_inplace(|x| x * factor);
        }

        // Renormalize probabilities to sum to 1.0
        for mut row in predictions.axis_iter_mut(ndarray::Axis(0)) {
            let sum: f64 = row.sum();
            if sum > 1e-10 {
                row /= sum;
            } else {
                // If all probabilities are near zero, set to uniform distribution
                row.fill(0.2); // 1/5 for each class
            }
        }

        Ok(())
    }

    /// Convert sequences to tensor for prediction (memory-optimized, no targets needed)
    pub fn convert_sequences_to_prediction_tensor(
        &self,
        sequences: &Array3<f64>,
    ) -> Result<Tensor> {
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

    /// Convert ndarray Array2 to Candle tensor - helper method
    fn array2_to_tensor(&self, array: &Array2<f64>) -> Result<Tensor> {
        let (rows, cols) = array.dim();

        // Convert f64 to f32 and flatten
        let data_f32: Vec<f32> = array.iter().map(|&x| x as f32).collect();

        // Create tensor
        Tensor::from_vec(data_f32, (rows, cols), &self.device).map_err(|e| {
            VangaError::ModelError(format!("Failed to create tensor from Array2: {}", e))
        })
    }

    /// Apply dropout with sequence context for variational dropout
    ///
    /// # Arguments
    /// * `tensor` - Input tensor to apply dropout to
    /// * `training` - Whether model is in training mode
    /// * `sequence_context` - Optional sequence ID for variational dropout consistency
    fn apply_dropout_with_context(
        &self,
        tensor: &Tensor,
        training: bool,
        sequence_context: Option<&str>,
    ) -> Result<Tensor> {
        let dropout_config = self
            .dropout_config
            .as_ref()
            .ok_or_else(|| VangaError::ModelError("Dropout configuration not set".to_string()))?;

        // CRITICAL FIX: Only apply dropout if enabled AND in training mode
        if !dropout_config.enabled || !training {
            log::trace!(
                "🔧 Dropout skipped - enabled: {}, training: {}",
                dropout_config.enabled,
                training
            );
            return Ok(tensor.clone());
        }

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

        // Choose dropout method based on configuration
        let dropped_tensor = if let Some(sequence_id) = sequence_context {
            // Use variational dropout with sequence context
            log::debug!(
                "🔧 Applying variational dropout with rate {:.3} to tensor shape {:?} [sequence: {}]",
                dropout_rate,
                tensor.shape(),
                sequence_id
            );
            SeededTensorUtils::variational_dropout(
                tensor,
                dropout_rate as f32,
                training,
                sequence_id,
            )?
        } else if dropout_config.recurrent {
            // Use recurrent dropout (for hidden state connections)
            log::debug!(
                "🔧 Applying recurrent dropout with rate {:.3} to tensor shape {:?}",
                dropout_rate,
                tensor.shape()
            );
            SeededTensorUtils::recurrent_dropout(tensor, dropout_rate as f32, training)?
        } else {
            // Fall back to standard dropout
            log::debug!(
                "🔧 Applying standard dropout with rate {:.3} to tensor shape {:?}",
                dropout_rate,
                tensor.shape()
            );
            SeededTensorUtils::deterministic_dropout(tensor, dropout_rate as f32, training)?
        };

        log::debug!(
            "🔧 Applied LSTM dropout with rate {:.3} to tensor shape {:?} [training={}, variational={}, recurrent={}]",
            dropout_rate,
            tensor.shape(),
            training,
            dropout_config.variational,
            dropout_config.recurrent
        );

        Ok(dropped_tensor)
    }

    /// Extract LSTM features for XGBoost (z = h_n from paper)
    ///
    /// This method performs a forward pass through the LSTM layers and extracts
    /// the final hidden state as features for XGBoost regression.
    ///
    /// # Arguments
    /// * `sequences` - Input sequences tensor [batch_size, seq_len, features]
    ///
    /// # Returns
    /// * `Result<Tensor>` - LSTM features tensor [batch_size, feature_dim]
    pub fn extract_lstm_features(&self, sequences: &Tensor) -> Result<Tensor> {
        log::debug!("🔍 Extracting LSTM features for XGBoost");

        // Ensure model is initialized
        let forward_lstm_layers = self
            .lstm_layers
            .as_ref()
            .ok_or_else(|| VangaError::model("LSTM layers not initialized"))?;

        let batch_size = sequences.dim(0)?;
        let seq_len = sequences.dim(1)?;
        let input_size = sequences.dim(2)?;

        log::debug!(
            "📊 Input shape: batch={}, seq_len={}, features={}",
            batch_size,
            seq_len,
            input_size
        );

        // Check if this is bidirectional architecture
        let is_bidirectional = matches!(
            self.architecture,
            Some(crate::config::model::LSTMArchitecture::BidirectionalLSTM { .. })
        );

        let mut current_input = sequences.clone();

        if is_bidirectional {
            // Bidirectional processing - mirror the forward() method logic
            let backward_lstm_layers = self.backward_lstm_layers.as_ref().ok_or_else(|| {
                VangaError::ModelError(
                    "Backward LSTM layers not initialized for bidirectional model".to_string(),
                )
            })?;

            log::debug!("🔄 Processing bidirectional LSTM for feature extraction");

            // Process each layer bidirectionally
            for (layer_idx, (forward_layer, backward_layer)) in forward_lstm_layers
                .iter()
                .zip(backward_lstm_layers.iter())
                .enumerate()
            {
                // Process forward direction - CRITICAL FIX: Use seq_init with zero_state for deterministic predictions
                let batch_size = current_input.dim(0)?;
                let forward_zero_state = forward_layer.zero_state(batch_size)?;
                let forward_states = forward_layer.seq_init(&current_input, &forward_zero_state)?;
                if forward_states.is_empty() {
                    return Err(VangaError::ModelError(format!(
                        "Forward layer {} produced no states",
                        layer_idx
                    )));
                }

                // CRITICAL OOM FIX: Only extract last timestep for feature extraction
                let forward_last = forward_states.last().ok_or_else(|| {
                    VangaError::ModelError(format!("Forward layer {} has no states", layer_idx))
                })?;
                let forward_output = forward_last.h().unsqueeze(1)?.contiguous()?;

                // Process backward direction - CRITICAL FIX: Use seq_init with zero_state for deterministic predictions
                let backward_zero_state = backward_layer.zero_state(batch_size)?;
                let backward_states =
                    backward_layer.seq_init(&current_input, &backward_zero_state)?;
                if backward_states.is_empty() {
                    return Err(VangaError::ModelError(format!(
                        "Backward layer {} produced no states",
                        layer_idx
                    )));
                }

                // CRITICAL OOM FIX: Only extract last timestep for feature extraction
                let backward_last = backward_states.last().ok_or_else(|| {
                    VangaError::ModelError(format!("Backward layer {} has no states", layer_idx))
                })?;
                let backward_output = backward_last.h().unsqueeze(1)?.contiguous()?;

                // Concatenate forward and backward outputs along the feature dimension
                // forward_output: [batch_size, 1, hidden_size]
                // backward_output: [batch_size, 1, hidden_size]
                // Result: [batch_size, 1, 2*hidden_size]
                current_input =
                    Tensor::cat(&[&forward_output, &backward_output], 2)?.contiguous()?;

                log::debug!(
                    "🔄 Bidirectional layer {} - Concatenated shape: {:?} (last timestep only)",
                    layer_idx,
                    current_input.shape()
                );
            }

            // Extract final timestep features from bidirectional output
            // Already have [batch_size, 1, 2*hidden_size] from last layer
            let lstm_features = current_input.squeeze(1)?; // Remove sequence dimension to get [batch_size, 2*hidden_size]

            log::info!(
                "✅ Extracted LSTM latent vector z_test shape: {:?}",
                lstm_features.shape()
            );

            return Ok(lstm_features);
        }

        // Standard (non-bidirectional) processing - CRITICAL: Only keep last layer's output
        let mut current_output = current_input;

        for (layer_idx, lstm_layer) in forward_lstm_layers.iter().enumerate() {
            // LSTM forward pass - CRITICAL FIX: Use seq_init with zero_state for deterministic predictions
            let batch_size = current_output.dim(0)?;
            let zero_state = lstm_layer.zero_state(batch_size)?;
            let lstm_states = lstm_layer.seq_init(&current_output, &zero_state)?;

            if lstm_states.is_empty() {
                return Err(VangaError::ModelError(format!(
                    "LSTM layer {} produced no states",
                    layer_idx
                )));
            }

            // CRITICAL OOM FIX: Only extract last timestep, don't stack all timesteps
            // For feature extraction, we only need h_n (final hidden state)
            let last_state = lstm_states.last().ok_or_else(|| {
                VangaError::ModelError(format!("Layer {} has no states", layer_idx))
            })?;

            // Get last hidden state and expand to [batch_size, 1, hidden_size] for next layer
            let last_hidden = last_state.h();
            current_output = last_hidden.unsqueeze(1)?.contiguous()?;

            log::debug!(
                "🔄 Standard layer {} output shape: {:?} (last timestep only)",
                layer_idx,
                current_output.shape()
            );
        }

        // Extract final timestep features: z = h_n (equation 8 from paper)
        // Already have [batch_size, 1, hidden_size] from last layer
        let lstm_features = current_output;

        log::info!(
            "✅ Extracted standard LSTM features shape: {:?}",
            lstm_features.shape()
        );

        // Apply attention if enabled (optional enhancement)
        let features = if self.use_attention {
            if let Some(attention) = &self.attention_module {
                log::debug!("🎯 Applying attention to LSTM features");
                let attention_result = attention.forward(&lstm_features, false)?; // inference mode
                                                                                  // Handle attention output (may be tuple)
                let (attended_features, _) = attention_result;
                attended_features
            } else {
                lstm_features
            }
        } else {
            lstm_features
        };

        // Ensure feature dimension matches configuration
        // For 3D tensor [batch_size, 1, hidden_size], calculate total feature dim
        let expected_dim = self.get_xgboost_feature_dim();
        let features_dims = features.dims();
        let actual_dim = if features_dims.len() == 3 {
            features_dims[1] * features_dims[2] // seq_len * hidden_size (1 * hidden_size = hidden_size)
        } else {
            features_dims[1] // 2D case
        };

        if actual_dim != expected_dim {
            log::warn!(
                "⚠️  Feature dimension mismatch: expected={}, actual={}",
                expected_dim,
                actual_dim
            );

            // For 3D features, flatten to 2D [batch_size, seq_len*hidden_size]
            let features_2d = if features_dims.len() == 3 {
                features.reshape((features_dims[0], features_dims[1] * features_dims[2]))?
            } else {
                features.clone()
            };

            // Add projection layer if needed (simple linear transformation)
            if let Some(output_layer) = &self.output_layer {
                log::debug!("🔄 Applying output projection to match feature dimension");
                let projected = output_layer.forward(&features_2d)?;

                // Take first expected_dim features if output is larger
                if projected.dim(1)? >= expected_dim {
                    let final_features = projected.narrow(1, 0, expected_dim)?;
                    log::debug!("✅ Final LSTM features shape: {:?}", final_features.shape());
                    return Ok(final_features);
                }
            }

            // Fallback: pad or truncate to match expected dimension
            return self.adjust_feature_dimension(features_2d, expected_dim);
        }

        // Flatten 3D to 2D [batch_size, seq_len*hidden_size] for XGBoost
        let features_2d = if features_dims.len() == 3 {
            features.reshape((features_dims[0], features_dims[1] * features_dims[2]))?
        } else {
            features
        };

        log::debug!("✅ Final LSTM features shape: {:?}", features_2d.shape());
        Ok(features_2d)
    }

    /// Extract LSTM features for all sequences in a batch (for training)
    ///
    /// # Arguments
    /// * `sequences` - Input sequences array [batch_size, seq_len, features]
    ///
    /// # Returns
    /// * `Result<Tensor>` - LSTM features tensor [batch_size, feature_dim]
    pub fn extract_all_lstm_features(&self, sequences: &Array3<f64>) -> Result<Tensor> {
        let total_sequences = sequences.shape()[0];
        let seq_len = sequences.shape()[1];
        let features = sequences.shape()[2];

        log::info!(
            "🔄 Extracting LSTM features for {} sequences (batched processing)",
            total_sequences
        );

        // CRITICAL FIX: Process in batches to avoid CUDA OOM
        // Use configured batch size from training config for consistency
        let batch_size = self.training_config.batch_size;
        let num_batches = total_sequences.div_ceil(batch_size);

        // Calculate memory requirements
        let memory_per_sequence_mb = (seq_len * features * 4) / 1_048_576; // 4 bytes per f32
        let total_memory_mb = memory_per_sequence_mb * total_sequences;
        let batch_memory_mb = memory_per_sequence_mb * batch_size;

        log::info!(
            "📦 Processing {} sequences in {} batches of size {} (from config)",
            total_sequences,
            num_batches,
            batch_size
        );
        log::info!(
            "💾 Memory estimate: ~{} MB total, ~{} MB per batch",
            total_memory_mb,
            batch_memory_mb
        );

        let mut all_features = Vec::new();

        for batch_idx in 0..num_batches {
            let start_idx = batch_idx * batch_size;
            let end_idx = std::cmp::min(start_idx + batch_size, total_sequences);
            let current_batch_size = end_idx - start_idx;

            log::debug!(
                "🔄 Processing batch {}/{}: sequences {}-{} ({} sequences)",
                batch_idx + 1,
                num_batches,
                start_idx,
                end_idx,
                current_batch_size
            );

            // Extract batch slice
            let batch_sequences = sequences.slice(s![start_idx..end_idx, .., ..]);

            // Convert batch to tensor
            let mut seq_data: Vec<f32> =
                Vec::with_capacity(current_batch_size * seq_len * features);
            for batch_idx in 0..current_batch_size {
                for seq_idx in 0..seq_len {
                    for feature_idx in 0..features {
                        seq_data.push(batch_sequences[[batch_idx, seq_idx, feature_idx]] as f32);
                    }
                }
            }

            let seq_tensor = Tensor::from_vec(
                seq_data,
                (current_batch_size, seq_len, features),
                &self.device,
            )
            .map_err(|e| VangaError::model(format!("Failed to create sequence tensor: {}", e)))?;

            // Extract features for this batch
            let batch_features = self.extract_lstm_features(&seq_tensor)?;

            // CRITICAL: Drop intermediate tensors immediately to free memory
            drop(seq_tensor);

            // Convert to CPU and store (to free GPU memory immediately)
            let batch_features_cpu = batch_features.to_device(&candle_core::Device::Cpu)?;
            // Get shape as owned values before moving
            let cpu_shape = (batch_features_cpu.dim(0)?, batch_features_cpu.dim(1)?);
            drop(batch_features);
            all_features.push(batch_features_cpu);

            log::debug!(
                "✅ Batch {}/{} completed, features shape: {:?}",
                batch_idx + 1,
                num_batches,
                cpu_shape
            );
        }

        // Concatenate all batch features on CPU first
        log::info!(
            "🔗 Concatenating {} batches of features...",
            all_features.len()
        );
        let concatenated_cpu = Tensor::cat(&all_features, 0)?;

        // Move final result back to original device
        let final_features = concatenated_cpu.to_device(&self.device)?;

        log::info!(
            "✅ Extracted LSTM features for all {} sequences, final shape: {:?}",
            total_sequences,
            final_features.shape()
        );

        Ok(final_features)
    }

    /// Get expected XGBoost feature dimension from configuration
    pub fn get_xgboost_feature_dim(&self) -> usize {
        // Use XGBoost config feature_dim if available, otherwise calculate from LSTM architecture
        if let Some(ref xgboost_model) = self.xgboost_model {
            xgboost_model.get_config().feature_dim
        } else {
            // Calculate based on LSTM architecture
            let base_hidden_size = self.config.hidden_sizes.last().copied().unwrap_or(64);

            // Check if this is bidirectional architecture - doubles the feature dimension
            let is_bidirectional = matches!(
                self.architecture,
                Some(crate::config::model::LSTMArchitecture::BidirectionalLSTM { .. })
            );

            if is_bidirectional {
                base_hidden_size * 2 // Bidirectional concatenates forward + backward
            } else {
                base_hidden_size // Standard architectures use base size
            }
        }
    }

    /// Get expected XGBoost feature dimension with explicit config parameter
    /// This method prioritizes the provided XGBoost config over calculated values
    pub fn get_xgboost_feature_dim_with_config(
        &self,
        xgb_config: &crate::config::model::XGBoostConfig,
    ) -> usize {
        // ALWAYS prioritize the explicit config value
        xgb_config.feature_dim
    }

    /// Adjust feature dimension to match expected size
    fn adjust_feature_dimension(&self, features: Tensor, expected_dim: usize) -> Result<Tensor> {
        let actual_dim = features.dim(1)?;

        if actual_dim > expected_dim {
            // Truncate to expected dimension
            log::debug!(
                "🔧 Truncating features from {} to {}",
                actual_dim,
                expected_dim
            );
            Ok(features.narrow(1, 0, expected_dim)?)
        } else if actual_dim < expected_dim {
            // Pad with zeros to expected dimension
            log::debug!(
                "🔧 Padding features from {} to {}",
                actual_dim,
                expected_dim
            );
            let batch_size = features.dim(0)?;
            let padding_size = expected_dim - actual_dim;

            let zeros = Tensor::zeros((batch_size, padding_size), features.dtype(), &self.device)?;
            let padded = Tensor::cat(&[&features, &zeros], 1)?;
            Ok(padded)
        } else {
            // Dimension already matches
            Ok(features)
        }
    }
}
impl LSTMModel {
    /// Blend LSTM and XGBoost predictions using weighted average
    /// This is an improvement over the paper's pure XGBoost approach
    fn blend_predictions(
        &self,
        lstm_predictions: &Tensor,
        xgb_predictions: &Tensor,
        lstm_weight: f32,
    ) -> Result<Tensor> {
        // Ensure shapes match
        if lstm_predictions.shape() != xgb_predictions.shape() {
            return Err(VangaError::model(format!(
                "Prediction shape mismatch: LSTM {:?} vs XGBoost {:?}",
                lstm_predictions.shape(),
                xgb_predictions.shape()
            )));
        }

        let xgb_weight = 1.0 - lstm_weight;

        // Weighted average: lstm_weight * LSTM + (1 - lstm_weight) * XGBoost
        let lstm_weight_tensor =
            Tensor::new(&[lstm_weight], &self.device)?.broadcast_as(lstm_predictions.shape())?;
        let xgb_weight_tensor =
            Tensor::new(&[xgb_weight], &self.device)?.broadcast_as(xgb_predictions.shape())?;

        let lstm_scaled = lstm_predictions.mul(&lstm_weight_tensor)?;
        let xgb_scaled = xgb_predictions.mul(&xgb_weight_tensor)?;
        let combined = lstm_scaled.add(&xgb_scaled)?;

        log::info!(
            "🎯 Blended predictions: {:.0}% LSTM + {:.0}% XGBoost (improves stability)",
            lstm_weight * 100.0,
            xgb_weight * 100.0
        );

        Ok(combined)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::model::XGBoostConfig;

    #[test]
    fn test_xgboost_feature_dim_with_config() {
        // Create a test LSTM model
        let config = crate::model::lstm::config::LSTMConfig {
            input_size: 10,
            hidden_sizes: vec![128],
            output_size: 1,
            sequence_length: 60,
            learning_rate: 0.001,
            num_layers: 1,
        };

        let model = LSTMModel::new(config).unwrap();

        // Create XGBoost config with custom feature_dim
        let xgb_config = XGBoostConfig {
            enabled: true,
            feature_dim: 256, // Custom value different from default
            n_estimators: 100,
            max_depth: 6,
            objective: "RandomForest".to_string(),
            eval_metric: "multiclass_accuracy".to_string(),
            save_feature_importance: true,
            importance_method: "permutation".to_string(),
        };

        // Test that the method returns the config value, not calculated value
        let feature_dim = model.get_xgboost_feature_dim_with_config(&xgb_config);
        assert_eq!(feature_dim, 256, "Should return config feature_dim value");

        // Test with different config value
        let mut xgb_config2 = xgb_config.clone();
        xgb_config2.feature_dim = 512;
        let feature_dim2 = model.get_xgboost_feature_dim_with_config(&xgb_config2);
        assert_eq!(
            feature_dim2, 512,
            "Should return updated config feature_dim value"
        );
    }

    #[test]
    fn test_config_loading_integration() {
        // Test that config loading works with custom feature_dim
        let toml_content = r#"
[model]
[model.xgboost]
enabled = true
feature_dim = 999
n_estimators = 50
max_depth = 4
learning_rate = 0.1
subsample = 0.8
colsample_bytree = 0.8
reg_alpha = 0.0
reg_lambda = 1.0
early_stopping_rounds = 5
eval_metric = "rmse"
objective = "reg:squarederror"
save_feature_importance = true
importance_type = "gain"
importance_method = "permutation"
"#;

        // Parse the TOML config
        let parsed: toml::Value = toml::from_str(toml_content).unwrap();
        let xgb_config: XGBoostConfig = parsed["model"]["xgboost"].clone().try_into().unwrap();

        // Verify the custom feature_dim is loaded correctly
        assert_eq!(
            xgb_config.feature_dim, 999,
            "Config should load custom feature_dim"
        );
        assert_eq!(
            xgb_config.n_estimators, 50,
            "Config should load custom n_estimators"
        );

        // Test with LSTM model
        let lstm_config = crate::model::lstm::config::LSTMConfig {
            input_size: 10,
            hidden_sizes: vec![128],
            output_size: 1,
            sequence_length: 60,
            learning_rate: 0.001,
            num_layers: 1,
        };

        let model = LSTMModel::new(lstm_config).unwrap();
        let feature_dim = model.get_xgboost_feature_dim_with_config(&xgb_config);
        assert_eq!(
            feature_dim, 999,
            "Should use config value, not calculated value"
        );
    }
}
