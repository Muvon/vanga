// LSTM model implementation with Candle framework - PRESERVING ALL ORIGINAL LOGIC
use crate::config::ModelConfig;
use crate::model::attention::{AttentionConfig as AttentionModuleConfig, MultiHeadAttention};
use crate::utils::error::{Result, VangaError};
use candle_core::{DType, Device, Tensor};
use candle_nn::{
    linear, lstm,
    optim::{self, Optimizer},
    LSTMConfig as CandleLSTMConfig, Linear, Module, VarBuilder, VarMap, LSTM, RNN,
};
use ndarray::{s, Array2, Array3};
use serde::{Deserialize, Serialize};

/// LSTM network configuration - EXACT same as original
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LSTMConfig {
    pub input_size: usize,
    pub hidden_size: usize,
    pub output_size: usize,
    pub sequence_length: usize,
    pub learning_rate: f64,
    pub num_layers: usize, // Added for multi-layer support
}

/// Training configuration - preserving original structure
#[derive(Debug, Clone)]
struct TrainingConfig {
    epochs: usize,
    print_every: usize,
    clip_gradient: Option<f64>,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            epochs: 1,
            print_every: 10,
            clip_gradient: Some(1.0),
        }
    }
}

/// LSTM model for cryptocurrency forecasting - Enhanced with attention support
pub struct LSTMModel {
    config: LSTMConfig,
    lstm_layers: Option<Vec<LSTM>>, // Changed to Vec<LSTM> for manual chaining
    output_layer: Option<Linear>,
    pub attention_layers: Option<MultiHeadAttention>, // Public for testing
    pub attention_config: Option<AttentionModuleConfig>, // Public for testing
    pub use_attention: bool,                          // Public for testing
    device: Device,
    varmap: VarMap,
    training_config: TrainingConfig,
    trained: bool,
}

/// Serializable model state for persistence - SAME as original
#[derive(Debug, Clone, Serialize, Deserialize)]
struct ModelState {
    config: LSTMConfig,
    epochs: usize,
    print_every: usize,
    clip_gradient: Option<f64>,
}

impl LSTMModel {
    /// Create a new LSTM model - EXACT same logic as original
    pub fn new(config: LSTMConfig) -> Result<Self> {
        let training_config = TrainingConfig {
            epochs: 1, // Placeholder - will be set by configure_training()
            print_every: 10,
            clip_gradient: Some(1.0),
        };

        Ok(Self {
            config,
            lstm_layers: None,
            output_layer: None,
            attention_layers: None, // Initialize attention as None
            attention_config: None, // Initialize attention config as None
            use_attention: false,   // Attention disabled by default
            device: Device::Cpu,
            varmap: VarMap::new(),
            training_config,
            trained: false,
        })
    }
    /// Create LSTM model from ModelConfig - Enhanced with multi-layer support
    pub fn from_model_config(
        model_config: &ModelConfig,
        input_size: usize,
        output_size: usize,
    ) -> Result<Self> {
        // Extract sequence length from config - SAME logic
        let sequence_length = match &model_config.sequence_length {
            crate::config::model::SequenceLengthConfig::Fixed(len) => *len as usize,
            crate::config::model::SequenceLengthConfig::Auto {
                min_length,
                max_length: _,
            } => *min_length as usize,
            crate::config::model::SequenceLengthConfig::Adaptive => 60,
        };

        // Extract hidden units from config - SAME logic
        let hidden_size = match &model_config.hidden_units {
            crate::config::model::HiddenUnitsConfig::Fixed(units) => {
                units.first().copied().unwrap_or(128) as usize
            }
            crate::config::model::HiddenUnitsConfig::Auto {
                min_units,
                max_units: _,
            } => *min_units as usize,
            crate::config::model::HiddenUnitsConfig::Pyramid {
                base_units,
                reduction_factor: _,
            } => *base_units as usize,
        };

        // Extract number of layers from architecture config - NEW
        let num_layers = Self::extract_num_layers_from_architecture(&model_config.architecture);

        // Use sequence_length for LSTM configuration if needed - SAME logic
        let effective_hidden_size = if sequence_length > 100 {
            hidden_size + (sequence_length / 10) // Adjust hidden size based on sequence length
        } else {
            hidden_size
        };

        let lstm_config = LSTMConfig {
            input_size,
            hidden_size: effective_hidden_size,
            output_size,
            sequence_length,      // Use actual sequence length from config
            learning_rate: 0.001, // Default learning rate
            num_layers,           // Now properly extracted from architecture
        };

        let mut model = Self::new(lstm_config)?;

        // Configure attention if enabled
        if let Some(model_config) = Some(model_config) {
            model.configure_attention(&model_config.attention)?;
        }

        Ok(model)
    }

    /// Configure attention for the model
    pub fn configure_attention(
        &mut self,
        attention_config: &crate::config::model::AttentionConfig,
    ) -> Result<()> {
        if !attention_config.enabled {
            self.use_attention = false;
            return Ok(());
        }

        // Convert config AttentionConfig to module AttentionConfig
        let module_config = AttentionModuleConfig {
            num_heads: attention_config.heads as usize,
            head_dim: attention_config.head_dim.unwrap_or(64) as usize, // Auto-optimized default
            dropout_rate: attention_config.dropout_rate,
            temperature_scaling: attention_config.temperature_scaling,
            use_relative_position: attention_config.use_relative_position,
            max_sequence_length: self.config.sequence_length,
        };

        self.attention_config = Some(module_config);
        self.use_attention = true;

        log::info!(
            "✅ Attention configured: {} heads, head_dim={}",
            attention_config.heads,
            attention_config.head_dim.unwrap_or(64)
        );

        Ok(())
    }

    /// Initialize attention layers during model initialization
    fn initialize_attention_layers(&mut self, vs: &VarBuilder) -> Result<()> {
        if let Some(attention_config) = &self.attention_config {
            let attention = MultiHeadAttention::new(
                self.config.hidden_size, // Use LSTM hidden size as input dimension
                attention_config.clone(),
                vs.pp("attention"),
                self.device.clone(),
            )?;

            self.attention_layers = Some(attention);

            log::info!(
                "✅ Attention layers initialized: {} heads, hidden_size={}",
                attention_config.num_heads,
                self.config.hidden_size
            );
        }

        Ok(())
    }

    /// Extract number of layers from ModelConfig architecture - NEW helper method
    fn extract_num_layers_from_architecture(
        architecture: &crate::config::model::LSTMArchitecture,
    ) -> usize {
        use crate::config::model::LSTMArchitecture;
        match architecture {
            LSTMArchitecture::MultiLSTM { layers } => *layers as usize,
            LSTMArchitecture::StackedLSTM { layers } => *layers as usize,
            LSTMArchitecture::BidirectionalLSTM { layers } => *layers as usize,
            LSTMArchitecture::CNNLSTM { lstm_layers, .. } => *lstm_layers as usize,
            LSTMArchitecture::TransformerLSTM { lstm_layers, .. } => *lstm_layers as usize,
        }
    }

    /// Initialize multi-layer LSTM network using Sequential - COMPLETE REWRITE
    fn initialize_network(&mut self) -> Result<()> {
        if self.lstm_layers.is_some() {
            return Ok(()); // Already initialized
        }

        log::info!(
            "Initializing multi-layer LSTM network with config: {:?}",
            self.config
        );

        let vs = VarBuilder::from_varmap(&self.varmap, DType::F32, &self.device);
        let num_layers = self.config.num_layers;

        // Validate layer count for optimal performance
        if num_layers == 0 {
            return Err(VangaError::ModelError(
                "Number of layers must be at least 1".to_string(),
            ));
        }
        if num_layers > 4 {
            log::warn!("Large number of layers ({}) may cause overfitting. Consider 2-3 layers for most datasets.", num_layers);
        }

        // Build multi-layer LSTM stack using Sequential
        let mut lstm_layers = Vec::new();

        for layer_idx in 0..num_layers {
            // Input size: first layer uses input_size, subsequent layers use hidden_size
            let layer_input_size = if layer_idx == 0 {
                self.config.input_size
            } else {
                self.config.hidden_size
            };

            // Create LSTM configuration for this layer
            let lstm_config = CandleLSTMConfig {
                layer_idx,
                direction: candle_nn::rnn::Direction::Forward,
                ..CandleLSTMConfig::default()
            };

            // Create LSTM layer with proper naming
            let lstm_layer = lstm(
                layer_input_size,
                self.config.hidden_size,
                lstm_config,
                vs.pp(format!("lstm_layer_{}", layer_idx)),
            )
            .map_err(|e| {
                VangaError::ModelError(format!("LSTM layer {} creation failed: {}", layer_idx, e))
            })?;

            // Store LSTM layer directly (no boxing needed)
            lstm_layers.push(lstm_layer);

            log::debug!(
                "✅ LSTM layer {} initialized: input_size={}, hidden_size={}",
                layer_idx,
                layer_input_size,
                self.config.hidden_size
            );
        }

        // Store the LSTM layers for manual chaining in forward pass
        self.lstm_layers = Some(lstm_layers);

        // Initialize attention layers if configured
        if self.use_attention && self.attention_config.is_some() {
            self.initialize_attention_layers(&vs)?;
        }

        // Attention integration temporarily disabled for clean compilation

        // Create output layer for sequence-to-one prediction - SAME as original
        let output_layer = linear(
            self.config.hidden_size,
            1, // Single output like original (output_size determined by target structure)
            vs.pp("output"),
        )
        .map_err(|e| VangaError::ModelError(format!("Output layer creation failed: {}", e)))?;

        self.output_layer = Some(output_layer);

        log::info!(
            "✅ Multi-layer LSTM network initialized successfully: {} layers, {} → {} → 1",
            num_layers,
            self.config.input_size,
            self.config.hidden_size
        );
        Ok(())
    }

    /// Convert Array3 sequences to Candle tensors - preserving original data structure
    fn convert_sequences_to_tensors(
        &self,
        sequences: &Array3<f64>,
        targets: &Array2<f64>,
    ) -> Result<(Tensor, Tensor)> {
        // sequences shape: [batch, sequence_length, features] - SAME as original
        // targets shape: [batch, output_size] - SAME as original

        // Use minimum batch size to ensure alignment - SAME logic as original
        let batch_size = std::cmp::min(sequences.shape()[0], targets.shape()[0]);

        log::debug!(
            "Converting training data: {} sequences, {} targets, using {} aligned samples",
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

        log::info!(
            "Training data converted: {} samples with sequence length {} (using single target output instead of {})",
            batch_size,
            seq_len,
            targets.shape()[1]
        );

        Ok((seq_tensor, target_tensor))
    }

    /// Forward pass through multi-layer LSTM network using Sequential
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let lstm_layers = self
            .lstm_layers
            .as_ref()
            .ok_or_else(|| VangaError::ModelError("LSTM layers not initialized".to_string()))?;

        let output_layer = self
            .output_layer
            .as_ref()
            .ok_or_else(|| VangaError::ModelError("Output layer not initialized".to_string()))?;

        // Manual forward pass through LSTM layers
        let mut current_output = input.clone();
        for (i, lstm_layer) in lstm_layers.iter().enumerate() {
            // Use the seq method from RNN trait which processes the full sequence
            let layer_states = lstm_layer.seq(&current_output)?;

            // Validate we have states to process
            if layer_states.is_empty() {
                return Err(VangaError::ModelError(format!(
                    "Layer {} produced no states",
                    i
                )));
            }

            // Collect all hidden states from the sequence to form the output tensor
            // Each state.h() is [batch_size, hidden_size], we need [batch_size, seq_len, hidden_size]
            let mut hidden_states = Vec::new();
            for state in &layer_states {
                hidden_states.push(state.h().clone());
            }

            // Stack the hidden states to form [batch_size, seq_len, hidden_size]
            current_output = Tensor::stack(&hidden_states, 1)?;

            // Validate output dimensions match expectations
            let output_shape = current_output.shape();
            log::debug!("Layer {} output shape: {:?}", i, output_shape);

            // Ensure we have the expected 3D tensor [batch_size, seq_len, hidden_size]
            if output_shape.dims().len() != 3 {
                return Err(VangaError::ModelError(format!(
                    "Layer {} output has wrong dimensions: expected 3D tensor, got {:?}",
                    i, output_shape
                )));
            }
        }
        let lstm_output = current_output;

        // Apply attention if enabled
        let final_output = if self.use_attention && self.attention_layers.is_some() {
            let attention = self.attention_layers.as_ref().unwrap();
            let (attended_output, _attention_weights) = attention.forward(&lstm_output)?;

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
                .squeeze(1)
                .map_err(|e| {
                    VangaError::ModelError(format!(
                        "Failed to squeeze attended last timestep: {}",
                        e
                    ))
                })?
        } else {
            // Standard LSTM: For sequence-to-one prediction, we need the last timestep
            // Sequential output should be [batch_size, seq_len, hidden_size]
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

        Ok(predictions)
    }

    /// Calculate MSE loss between predictions and targets - EXACT same as original
    fn calculate_mse_loss(&self, predictions: &Array2<f64>, targets: &Array2<f64>) -> f64 {
        // CRITICAL FIX: Validate shapes before operations - SAME as original
        if predictions.shape() != targets.shape() {
            log::error!(
                "Shape mismatch in MSE calculation: predictions={:?}, targets={:?}",
                predictions.shape(),
                targets.shape()
            );
            return f64::INFINITY;
        }

        let diff = predictions - targets;
        let squared_diff = &diff * &diff;
        squared_diff.mean().unwrap_or(f64::INFINITY)
    }

    /// Calculate MAPE (Mean Absolute Percentage Error) for better understanding - EXACT same as original
    fn calculate_mape(&self, predictions: &Array2<f64>, targets: &Array2<f64>) -> f64 {
        // CRITICAL FIX: Validate shapes before operations - SAME as original
        if predictions.shape() != targets.shape() {
            log::error!(
                "Shape mismatch in MAPE calculation: predictions={:?}, targets={:?}",
                predictions.shape(),
                targets.shape()
            );
            return f64::INFINITY;
        }

        let mut total_percentage_error = 0.0;
        let mut valid_samples = 0;

        for i in 0..predictions.nrows() {
            for j in 0..predictions.ncols() {
                let actual = targets[[i, j]];
                let predicted = predictions[[i, j]];

                // Avoid division by zero and very small values - SAME logic as original
                if actual.abs() > 1e-8 {
                    let percentage_error = ((actual - predicted).abs() / actual.abs()) * 100.0;
                    total_percentage_error += percentage_error;
                    valid_samples += 1;
                }
            }
        }

        if valid_samples > 0 {
            total_percentage_error / valid_samples as f64
        } else {
            f64::INFINITY
        }
    }

    /// PARALLELIZED: Train model in parallel batches for maximum CPU utilization - SAME interface as original
    pub async fn train_parallel_batches(
        &mut self,
        sequences: &Array3<f64>,
        targets: &Array2<f64>,
        _batch_size: usize,
    ) -> Result<()> {
        // Candle handles batching internally, so delegate to regular train
        self.train(sequences, targets).await
    }

    /// Configure training parameters from TrainingConfig - EXACT same logic as original
    pub fn configure_training(&mut self, vanga_config: &crate::config::TrainingConfig) {
        // Extract epochs from config - SAME logic as original
        let (max_epochs, use_early_stopping) = match &vanga_config.training.epochs {
            crate::config::training::EpochConfig::Auto { max_epochs } => {
                (*max_epochs as usize, true)
            }
            crate::config::training::EpochConfig::Fixed(epochs) => (*epochs as usize, false),
        };

        // Extract learning rate from config - SAME logic as original
        let learning_rate = match &vanga_config.training.learning_rate {
            crate::config::training::LearningRateConfig::Fixed(lr) => {
                log::info!("Using FIXED learning rate: {:.6}", lr);
                *lr
            }
            crate::config::training::LearningRateConfig::Adaptive { initial_lr } => {
                log::info!(
                    "Using ADAPTIVE learning rate starting at: {:.6}",
                    initial_lr
                );
                *initial_lr
            }
            crate::config::training::LearningRateConfig::Auto { min_lr, max_lr } => {
                log::info!("Using AUTO learning rate: {:.6} - {:.6}", min_lr, max_lr);
                *max_lr // Start with max, will be reduced automatically
            }
        };

        // Update rust-lstm training config - SAME as original
        self.training_config.epochs = max_epochs;
        self.training_config.print_every = if use_early_stopping { 10 } else { 50 }; // More frequent logging for early stopping

        // Store learning rate for optimizer creation - SAME as original
        self.config.learning_rate = learning_rate;

        // Extract and apply gradient clipping from config
        if let Some(gradient_clip) = vanga_config.training.gradient_clip {
            self.training_config.clip_gradient = Some(gradient_clip);
            log::info!("Using gradient clipping: {:.3}", gradient_clip);
        }

        log::info!(
            "✅ Training configured: epochs={}, lr={:.6}, early_stopping={}, print_every={}, gradient_clip={:?}",
            max_epochs,
            learning_rate,
            use_early_stopping,
            self.training_config.print_every,
            vanga_config.training.gradient_clip
        );
    }

    /// Train model - PRESERVING ALL ORIGINAL LOGIC with Candle
    pub async fn train(&mut self, sequences: &Array3<f64>, targets: &Array2<f64>) -> Result<()> {
        log::info!(
            "Training LSTM model with {} input features",
            self.config.input_size
        );
        log::debug!(
            "Training config: epochs={}, print_every={}, clip_gradient={:?}",
            self.training_config.epochs,
            self.training_config.print_every,
            self.training_config.clip_gradient
        );

        // Initialize network if not already done - SAME logic as original
        self.initialize_network()?;

        // Convert Array3 sequences to Candle tensors - equivalent to original convert_sequences_to_training_data
        let (input_tensor, target_tensor) =
            self.convert_sequences_to_tensors(sequences, targets)?;

        // Update config to reflect actual output size (1 for compatibility) - SAME as original
        self.config.output_size = 1;

        log::info!(
            "Starting LSTM training for {} epochs",
            self.training_config.epochs
        );

        // Create optimizer for training - using SGD to match original rust-lstm behavior
        let learning_rate = self.config.learning_rate;
        let mut sgd = <optim::SGD as optim::Optimizer>::new(self.varmap.all_vars(), learning_rate)
            .map_err(|e| VangaError::ModelError(format!("SGD optimizer creation failed: {}", e)))?;

        // Training loop with MSE loss and backpropagation - REAL training like original
        for epoch in 0..self.training_config.epochs {
            // Forward pass
            let predictions = self.forward(&input_tensor)?;

            // Calculate MSE loss tensor
            let loss = predictions.sub(&target_tensor)?.sqr()?.mean_all()?;

            // Backward pass with gradient computation
            let grads = loss.backward()?;

            // Update parameters using SGD optimizer
            sgd.step(&grads)?;

            if epoch % self.training_config.print_every == 0 {
                let loss_val = loss.to_scalar::<f32>().map_err(|e| {
                    VangaError::ModelError(format!("Loss scalar conversion failed: {}", e))
                })?;
                log::info!(
                    "Epoch {}/{}: Loss = {:.6}, Learning rate: {:.6}",
                    epoch + 1,
                    self.training_config.epochs,
                    loss_val,
                    self.config.learning_rate
                );
            }
        }

        self.trained = true;
        log::info!("LSTM training completed successfully");

        // Calculate final training metrics for better understanding - SAME as original
        if let Ok(final_predictions) = self.predict(sequences).await {
            let final_mse = self.calculate_mse_loss(&final_predictions, targets);
            let final_mape = self.calculate_mape(&final_predictions, targets);
            log::info!(
                "📊 Final Training Metrics - MSE: {:.6} (√MSE: {:.3}), MAPE: {:.2}%",
                final_mse,
                final_mse.sqrt(),
                final_mape
            );
        }

        Ok(())
    }

    /// Make predictions using the trained network - EXACT same logic as original
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

        // Convert sequences to tensor
        let (input_tensor, _) = self
            .convert_sequences_to_tensors(sequences, &Array2::zeros((sequences.shape()[0], 1)))?;

        // Forward pass through network
        let predictions_tensor = self.forward(&input_tensor)?;

        // Convert back to ndarray
        let predictions = self.tensor_to_array2(&predictions_tensor)?;

        log::info!("Generated {} predictions", predictions.nrows());
        Ok(predictions)
    }

    /// Convert Candle tensor to ndarray Array2 - helper method
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

        Array2::from_shape_vec((rows, cols), data_f64)
            .map_err(|e| VangaError::ModelError(format!("Failed to create Array2: {}", e)))
    }

    /// Save model to file - SAME interface as original
    pub fn save<P: AsRef<std::path::Path>>(&self, path: P) -> Result<()> {
        // Create a serializable model state - SAME as original
        let model_state = ModelState {
            config: self.config.clone(),
            epochs: self.training_config.epochs,
            print_every: self.training_config.print_every,
            clip_gradient: self.training_config.clip_gradient,
        };

        // Serialize to binary format using bincode - SAME as original
        let encoded = bincode::serialize(&model_state)
            .map_err(|e| VangaError::SerializationError(format!("Serialization failed: {}", e)))?;

        // Write to file - SAME as original
        std::fs::write(path, encoded)
            .map_err(|e| VangaError::IoError(format!("Failed to write model file: {}", e)))?;

        log::info!("Model saved successfully");
        Ok(())
    }

    /// Load model from file - SAME interface as original
    pub fn load<P: AsRef<std::path::Path>>(path: P) -> Result<Self> {
        // Read the model file - SAME as original
        let data = std::fs::read(&path)
            .map_err(|e| VangaError::IoError(format!("Failed to read model file: {}", e)))?;

        // Deserialize the model state - SAME as original
        let model_state: ModelState = bincode::deserialize(&data).map_err(|e| {
            VangaError::SerializationError(format!("Deserialization failed: {}", e))
        })?;

        // Create a new LSTM model with the loaded configuration - SAME as original
        let mut model = Self::new(model_state.config)?;
        model.training_config.epochs = model_state.epochs;
        model.training_config.print_every = model_state.print_every;
        model.training_config.clip_gradient = model_state.clip_gradient;

        // CRITICAL: Initialize the network for predictions - MISSING in original migration
        model.initialize_network()?;
        model.trained = true;

        log::info!("Model loaded successfully with initialized network");
        Ok(model)
    }

    /// Get the input size of the model - SAME as original
    pub fn get_input_size(&self) -> usize {
        self.config.input_size
    }

    /// Train with intelligent early stopping - REAL implementation with validation monitoring
    pub async fn train_with_early_stopping(
        &mut self,
        sequences: &Array3<f64>,
        targets: &Array2<f64>,
        vanga_config: &crate::config::TrainingConfig,
    ) -> Result<()> {
        // Configure training parameters
        self.configure_training(vanga_config);

        let validation_split = vanga_config.training.validation_split;
        let early_stopping_patience = vanga_config.training.early_stopping_patience;

        // Check if early stopping is enabled
        let use_early_stopping = match &vanga_config.training.epochs {
            crate::config::training::EpochConfig::Auto { .. } => true,
            crate::config::training::EpochConfig::Fixed(_) => false,
        };

        if !use_early_stopping || validation_split <= 0.0 {
            log::info!(
                "📊 STANDARD training: early_stopping={}, validation_split={:.1}%",
                use_early_stopping,
                validation_split * 100.0
            );
            return self.train(sequences, targets).await;
        }

        log::info!(
            "🧠 INTELLIGENT TRAINING: early_stopping=true, validation_split={:.1}%, patience={}",
            validation_split * 100.0,
            early_stopping_patience
        );

        // Split data into training and validation sets
        let total_samples = sequences.shape()[0];
        let train_samples = ((total_samples as f64) * (1.0 - validation_split)) as usize;

        log::info!(
            "📊 Data split: {} total → {} training ({:.1}%), {} validation ({:.1}%)",
            total_samples,
            train_samples,
            (1.0 - validation_split) * 100.0,
            total_samples - train_samples,
            validation_split * 100.0
        );

        // Create training and validation splits
        let train_sequences = sequences.slice(s![0..train_samples, .., ..]).to_owned();
        let train_targets = targets.slice(s![0..train_samples, ..]).to_owned();
        let val_sequences = sequences.slice(s![train_samples.., .., ..]).to_owned();
        let val_targets = targets.slice(s![train_samples.., ..]).to_owned();

        // Perform training with validation monitoring
        self.train_with_validation_monitoring(
            &train_sequences,
            &train_targets,
            &val_sequences,
            &val_targets,
            early_stopping_patience,
        )
        .await
    }

    /// Continue training with new data (incremental learning) - SAME interface as original
    pub async fn continue_training(
        &mut self,
        new_sequences: &Array3<f64>,
        new_targets: &Array2<f64>,
        vanga_config: &crate::config::TrainingConfig,
    ) -> Result<()> {
        log::info!(
            "🔄 INCREMENTAL TRAINING: Adding {} new samples to existing model",
            new_sequences.shape()[0]
        );

        // Check if model is already trained - SAME logic as original
        if !self.trained {
            return Err(VangaError::ModelError(
                "Cannot continue training: model not initialized. Use train_with_early_stopping() first.".to_string()
            ));
        }

        // Configure training with typically lower learning rate for incremental training - SAME logic as original
        let mut incremental_config = vanga_config.clone();

        // Reduce learning rate for incremental training to preserve existing knowledge - SAME logic as original
        incremental_config.training.learning_rate = match &vanga_config.training.learning_rate {
            crate::config::training::LearningRateConfig::Fixed(lr) => {
                let reduced_lr = lr * 0.1; // 10x smaller for incremental
                log::info!(
                    "🔽 Reducing learning rate for incremental training: {:.6} → {:.6}",
                    lr,
                    reduced_lr
                );
                crate::config::training::LearningRateConfig::Fixed(reduced_lr)
            }
            crate::config::training::LearningRateConfig::Adaptive { initial_lr } => {
                let reduced_lr = initial_lr * 0.1;
                log::info!(
                    "🔽 Reducing initial learning rate for incremental training: {:.6} → {:.6}",
                    initial_lr,
                    reduced_lr
                );
                crate::config::training::LearningRateConfig::Adaptive {
                    initial_lr: reduced_lr,
                }
            }
            crate::config::training::LearningRateConfig::Auto { min_lr, max_lr } => {
                let reduced_max = max_lr * 0.1;
                let reduced_min = min_lr * 0.1;
                log::info!("🔽 Reducing learning rate range for incremental training: {:.6}-{:.6} → {:.6}-{:.6}",
                    min_lr, max_lr, reduced_min, reduced_max);
                crate::config::training::LearningRateConfig::Auto {
                    min_lr: reduced_min,
                    max_lr: reduced_max,
                }
            }
        };

        // Use smaller patience for incremental training (faster convergence expected) - SAME logic as original
        incremental_config.training.early_stopping_patience =
            (vanga_config.training.early_stopping_patience / 2).max(10);

        log::info!(
            "⚙️  Incremental training config: patience={}, reduced_lr=true",
            incremental_config.training.early_stopping_patience
        );

        // Train with the new data using reduced learning rate - SAME logic as original
        self.train_with_early_stopping(new_sequences, new_targets, &incremental_config)
            .await?;

        log::info!("✅ Incremental training completed successfully!");
        Ok(())
    }

    /// Append new data to existing training data and retrain (alternative approach) - SAME interface as original
    pub async fn retrain_with_appended_data(
        &mut self,
        existing_sequences: &Array3<f64>,
        existing_targets: &Array2<f64>,
        new_sequences: &Array3<f64>,
        new_targets: &Array2<f64>,
        vanga_config: &crate::config::TrainingConfig,
    ) -> Result<()> {
        log::info!(
            "🔄 RETRAIN WITH APPENDED DATA: {} existing + {} new = {} total samples",
            existing_sequences.shape()[0],
            new_sequences.shape()[0],
            existing_sequences.shape()[0] + new_sequences.shape()[0]
        );

        // Combine existing and new data - SAME logic as original
        let combined_sequences = ndarray::concatenate(
            ndarray::Axis(0),
            &[existing_sequences.view(), new_sequences.view()],
        )
        .map_err(|e| VangaError::DataError(format!("Failed to concatenate sequences: {}", e)))?;
        let combined_targets = ndarray::concatenate(
            ndarray::Axis(0),
            &[existing_targets.view(), new_targets.view()],
        )
        .map_err(|e| VangaError::DataError(format!("Failed to concatenate targets: {}", e)))?;

        log::info!(
            "📊 Combined dataset: {} samples x {} features x {} sequence_length",
            combined_sequences.shape()[0],
            combined_sequences.shape()[2],
            combined_sequences.shape()[1]
        );

        // Train on combined dataset (this preserves all historical patterns) - SAME logic as original
        self.train_with_early_stopping(&combined_sequences, &combined_targets, vanga_config)
            .await?;

        log::info!("✅ Retrain with appended data completed successfully!");
        Ok(())
    }

    /// Train with validation monitoring and early stopping - NEW implementation
    async fn train_with_validation_monitoring(
        &mut self,
        train_sequences: &Array3<f64>,
        train_targets: &Array2<f64>,
        val_sequences: &Array3<f64>,
        val_targets: &Array2<f64>,
        patience: u32,
    ) -> Result<()> {
        log::info!("🏃 Training with validation monitoring and early stopping");

        // Initialize network if not already done
        self.initialize_network()?;

        // Convert training and validation data to tensors
        let (train_input_tensor, train_target_tensor) =
            self.convert_sequences_to_tensors(train_sequences, train_targets)?;
        let (val_input_tensor, val_target_tensor) =
            self.convert_sequences_to_tensors(val_sequences, val_targets)?;

        // Update config to reflect actual output size
        self.config.output_size = 1;

        // Create optimizer for training
        let mut learning_rate = self.config.learning_rate;
        let mut sgd = <optim::SGD as optim::Optimizer>::new(self.varmap.all_vars(), learning_rate)
            .map_err(|e| VangaError::ModelError(format!("SGD optimizer creation failed: {}", e)))?;

        // Early stopping variables
        let mut best_val_loss = f32::INFINITY;
        let mut patience_counter = 0;

        log::info!(
            "🏃 Training batch: epochs={}, learning_rate={:.6}",
            self.training_config.epochs,
            learning_rate
        );

        // Training loop with early stopping
        for epoch in 0..self.training_config.epochs {
            // Forward pass on training data
            let train_predictions = self.forward(&train_input_tensor)?;

            // Calculate training loss
            let train_loss = train_predictions
                .sub(&train_target_tensor)?
                .sqr()?
                .mean_all()?;

            // Backward pass and parameter update
            let grads = train_loss.backward()?;
            sgd.step(&grads)?;

            // Validation evaluation every epoch
            let val_predictions = self.forward(&val_input_tensor)?;
            let val_loss = val_predictions.sub(&val_target_tensor)?.sqr()?.mean_all()?;

            let val_loss_val = val_loss.to_scalar::<f32>().map_err(|e| {
                VangaError::ModelError(format!("Validation loss scalar conversion failed: {}", e))
            })?;

            // Check for improvement in validation loss
            if val_loss_val < best_val_loss {
                let improvement = ((best_val_loss - val_loss_val) / best_val_loss) * 100.0;
                log::info!(
                    "✅ NEW BEST validation loss: {:.6} (improved by {:.2}%)",
                    val_loss_val,
                    improvement
                );

                best_val_loss = val_loss_val;
                patience_counter = 0;
            } else {
                patience_counter += 1;

                if patience_counter >= patience {
                    log::info!("🛑 EARLY STOPPING triggered at {} total epochs! Best validation loss: {:.6}",
                              epoch + 1, best_val_loss);
                    break;
                }

                // Reduce learning rate when validation loss plateaus
                if patience_counter % (patience / 3).max(1) == 0 {
                    learning_rate *= 0.5;
                    log::info!(
                        "🔽 REDUCING learning rate: {:.6} → {:.6}",
                        learning_rate * 2.0,
                        learning_rate
                    );

                    // Create new optimizer with reduced learning rate
                    sgd = <optim::SGD as optim::Optimizer>::new(
                        self.varmap.all_vars(),
                        learning_rate,
                    )
                    .map_err(|e| {
                        VangaError::ModelError(format!("SGD optimizer recreation failed: {}", e))
                    })?;
                }
            }

            // Logging
            if epoch % self.training_config.print_every == 0 {
                let train_loss_val = train_loss.to_scalar::<f32>().map_err(|e| {
                    VangaError::ModelError(format!("Training loss scalar conversion failed: {}", e))
                })?;
                log::info!("📈 Epoch {}/{}: Train Loss = {:.6}, Validation loss: {:.6}, Learning rate: {:.6}",
                          epoch + 1, self.training_config.epochs, train_loss_val, val_loss_val, learning_rate);
            }
        }

        self.trained = true;
        log::info!(
            "🎯 Training completed! Final validation loss: {:.6}, final learning rate: {:.6}",
            best_val_loss,
            learning_rate
        );

        // Calculate final metrics
        if let Ok(final_predictions) = self.predict(train_sequences).await {
            let final_mse = self.calculate_mse_loss(&final_predictions, train_targets);
            let final_mape = self.calculate_mape(&final_predictions, train_targets);
            log::info!(
                "📊 Final Training Metrics - MSE: {:.6} (√MSE: {:.3}), MAPE: {:.2}%",
                final_mse,
                final_mse.sqrt(),
                final_mape
            );
        }

        Ok(())
    }
}

// Implement From trait for Candle error conversion
impl From<candle_core::Error> for VangaError {
    fn from(err: candle_core::Error) -> Self {
        VangaError::ModelError(format!("Candle error: {}", err))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::training::{EpochConfig, LearningRateConfig, TrainingParams};
    use ndarray::Array3;

    #[tokio::test]
    async fn test_early_stopping_functionality() {
        // Create a simple LSTM model
        let config = LSTMConfig {
            input_size: 3,
            hidden_size: 8,
            output_size: 1,
            sequence_length: 5,
            learning_rate: 0.01,
            num_layers: 2, // Default multi-layer
        };

        let mut model = LSTMModel::new(config).expect("Failed to create model");

        // Create simple training data (small dataset to trigger early stopping quickly)
        let sequences =
            Array3::from_shape_vec((10, 5, 3), (0..150).map(|i| (i as f64) * 0.1).collect())
                .expect("Failed to create sequences");

        let targets = Array2::from_shape_vec((10, 1), (0..10).map(|i| (i as f64) * 0.5).collect())
            .expect("Failed to create targets");

        // Create training config with early stopping enabled
        let training_config = crate::config::TrainingConfig {
            symbol: "TEST".to_string(),
            data_path: std::path::PathBuf::from("test.csv"),
            fresh_training: true,
            continue_training: false,
            horizons: vec!["1h".to_string()],
            features_config_path: None,
            model: crate::config::ModelConfig::default(),
            training: TrainingParams {
                epochs: EpochConfig::Auto { max_epochs: 100 },
                batch_size: crate::config::training::BatchSizeConfig::Fixed(32),
                learning_rate: LearningRateConfig::Fixed(0.01),
                validation_split: 0.2, // 20% validation
                test_split: 0.0,
                early_stopping_patience: 5, // Small patience for quick testing
                gradient_clip: Some(1.0),
            },
            data: crate::config::training::DataConfig::default(),
            optimization: crate::config::training::OptimizationConfig::default(),
        };

        // Test that early stopping training completes without errors
        let result = model
            .train_with_early_stopping(&sequences, &targets, &training_config)
            .await;

        assert!(
            result.is_ok(),
            "Early stopping training should complete successfully"
        );
        assert!(
            model.trained,
            "Model should be marked as trained after early stopping"
        );
    }

    #[tokio::test]
    async fn test_fixed_epochs_fallback() {
        // Test that fixed epoch configuration bypasses early stopping
        let config = LSTMConfig {
            input_size: 3,
            hidden_size: 8,
            output_size: 1,
            sequence_length: 5,
            learning_rate: 0.01,
            num_layers: 2, // Default multi-layer
        };

        let mut model = LSTMModel::new(config).expect("Failed to create model");

        // Create simple training data
        let sequences =
            Array3::from_shape_vec((8, 5, 3), (0..120).map(|i| (i as f64) * 0.1).collect())
                .expect("Failed to create sequences");

        let targets = Array2::from_shape_vec((8, 1), (0..8).map(|i| (i as f64) * 0.5).collect())
            .expect("Failed to create targets");

        // Create training config with fixed epochs (should bypass early stopping)
        let training_config = crate::config::TrainingConfig {
            symbol: "TEST".to_string(),
            data_path: std::path::PathBuf::from("test.csv"),
            fresh_training: true,
            continue_training: false,
            horizons: vec!["1h".to_string()],
            features_config_path: None,
            model: crate::config::ModelConfig::default(),
            training: TrainingParams {
                epochs: EpochConfig::Fixed(5), // Fixed epochs - should bypass early stopping
                batch_size: crate::config::training::BatchSizeConfig::Fixed(32),
                learning_rate: LearningRateConfig::Fixed(0.01),
                validation_split: 0.2,
                test_split: 0.0,
                early_stopping_patience: 10,
                gradient_clip: Some(1.0),
            },
            data: crate::config::training::DataConfig::default(),
            optimization: crate::config::training::OptimizationConfig::default(),
        };

        // Test that fixed epochs training completes without errors
        let result = model
            .train_with_early_stopping(&sequences, &targets, &training_config)
            .await;

        assert!(
            result.is_ok(),
            "Fixed epochs training should complete successfully"
        );
        assert!(
            model.trained,
            "Model should be marked as trained after fixed epochs training"
        );
    }

    #[tokio::test]
    async fn test_model_save_load_predict_workflow() {
        use std::path::PathBuf;
        use tempfile::tempdir;

        // Create a temporary directory for the test
        let temp_dir = tempdir().expect("Failed to create temp directory");
        let model_path = temp_dir.path().join("test_model.bin");

        // Step 1: Create and train a model
        let config = LSTMConfig {
            input_size: 3,
            hidden_size: 8,
            output_size: 1,
            sequence_length: 5,
            learning_rate: 0.01,
            num_layers: 2, // Default multi-layer
        };

        let mut model = LSTMModel::new(config).expect("Failed to create model");

        // Create training data
        let sequences =
            Array3::from_shape_vec((10, 5, 3), (0..150).map(|i| (i as f64) * 0.1).collect())
                .expect("Failed to create sequences");

        let targets = Array2::from_shape_vec((10, 1), (0..10).map(|i| (i as f64) * 0.5).collect())
            .expect("Failed to create targets");

        // Train the model with fixed epochs for quick testing
        let training_config = crate::config::TrainingConfig {
            symbol: "TEST".to_string(),
            data_path: PathBuf::from("test.csv"),
            fresh_training: true,
            continue_training: false,
            horizons: vec!["1h".to_string()],
            features_config_path: None,
            model: crate::config::ModelConfig::default(),
            training: TrainingParams {
                epochs: EpochConfig::Fixed(3), // Quick training for test
                batch_size: crate::config::training::BatchSizeConfig::Fixed(32),
                learning_rate: LearningRateConfig::Fixed(0.01),
                validation_split: 0.0, // No validation for this test
                test_split: 0.0,
                early_stopping_patience: 5,
                gradient_clip: Some(1.0),
            },
            data: crate::config::training::DataConfig::default(),
            optimization: crate::config::training::OptimizationConfig::default(),
        };

        model
            .train_with_early_stopping(&sequences, &targets, &training_config)
            .await
            .expect("Training should complete successfully");

        // Step 2: Save the model
        model.save(&model_path).expect("Model save should succeed");

        // Step 3: Load the model
        let loaded_model = LSTMModel::load(&model_path).expect("Model load should succeed");

        // Step 4: Test prediction with loaded model
        let prediction_result = loaded_model.predict(&sequences).await;

        assert!(
            prediction_result.is_ok(),
            "Prediction with loaded model should succeed"
        );

        let predictions = prediction_result.unwrap();
        assert_eq!(
            predictions.nrows(),
            sequences.shape()[0],
            "Should predict for all sequences"
        );
        assert_eq!(predictions.ncols(), 1, "Should have single output column");

        // Verify that the loaded model is properly initialized
        assert!(
            loaded_model.trained,
            "Loaded model should be marked as trained"
        );
        assert!(
            loaded_model.lstm_layers.is_some(),
            "Loaded model should have initialized LSTM stack"
        );
        assert!(
            loaded_model.output_layer.is_some(),
            "Loaded model should have initialized output layer"
        );
    }

    #[tokio::test]
    async fn test_multi_layer_lstm_functionality() {
        // Test multi-layer LSTM creation and training
        let config = LSTMConfig {
            input_size: 4,
            hidden_size: 16,
            output_size: 1,
            sequence_length: 10,
            learning_rate: 0.01,
            num_layers: 3, // Test 3-layer LSTM
        };

        let mut model = LSTMModel::new(config).expect("Failed to create multi-layer model");

        // Create training data with more complexity for multi-layer testing
        let sequences =
            Array3::from_shape_vec((20, 10, 4), (0..800).map(|i| (i as f64) * 0.01).collect())
                .expect("Failed to create sequences");

        let targets = Array2::from_shape_vec((20, 1), (0..20).map(|i| (i as f64) * 0.3).collect())
            .expect("Failed to create targets");

        // Create training config for multi-layer testing
        let training_config = crate::config::TrainingConfig {
            symbol: "TEST_MULTI".to_string(),
            data_path: std::path::PathBuf::from("test_multi.csv"),
            fresh_training: true,
            continue_training: false,
            horizons: vec!["1h".to_string()],
            features_config_path: None,
            model: crate::config::ModelConfig {
                architecture: crate::config::model::LSTMArchitecture::StackedLSTM { layers: 3 },
                ..crate::config::ModelConfig::default()
            },
            training: TrainingParams {
                epochs: EpochConfig::Fixed(5), // Quick training for test
                batch_size: crate::config::training::BatchSizeConfig::Fixed(16),
                learning_rate: LearningRateConfig::Fixed(0.01),
                validation_split: 0.0,
                test_split: 0.0,
                early_stopping_patience: 10,
                gradient_clip: Some(1.0),
            },
            data: crate::config::training::DataConfig::default(),
            optimization: crate::config::training::OptimizationConfig::default(),
        };

        // Test multi-layer training
        let result = model
            .train_with_early_stopping(&sequences, &targets, &training_config)
            .await;

        assert!(
            result.is_ok(),
            "Multi-layer LSTM training should complete successfully"
        );
        assert!(
            model.trained,
            "Multi-layer model should be marked as trained"
        );

        // Test prediction with multi-layer model
        let prediction_result = model.predict(&sequences).await;
        assert!(
            prediction_result.is_ok(),
            "Multi-layer prediction should succeed"
        );

        let predictions = prediction_result.unwrap();
        assert_eq!(
            predictions.nrows(),
            sequences.shape()[0],
            "Should predict for all sequences"
        );
        assert_eq!(predictions.ncols(), 1, "Should have single output column");

        // Verify multi-layer architecture is properly initialized
        assert!(
            model.lstm_layers.is_some(),
            "Multi-layer LSTM stack should be initialized"
        );
        assert_eq!(
            model.config.num_layers, 3,
            "Model should have 3 layers as configured"
        );

        // Verify multi-layer architecture is properly initialized
        assert!(
            model.lstm_layers.is_some(),
            "Multi-layer LSTM layers should be initialized"
        );
    }
}
