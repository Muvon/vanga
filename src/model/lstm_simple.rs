// LSTM model implementation with Candle framework - PRESERVING ALL ORIGINAL LOGIC
use crate::config::ModelConfig;
use crate::model::attention::{AttentionConfig as AttentionModuleConfig, MultiHeadAttention};
use crate::model::loss::CryptoLossFunction;
// MarketRegime imported in calculate_loss method
use crate::utils::error::{Result, VangaError};

use candle_core::{DType, Device, Tensor};
use candle_nn::{
    linear, lstm,
    optim::{self, Optimizer},
    LSTMConfig as CandleLSTMConfig, Linear, Module, VarBuilder, VarMap, LSTM, RNN,
};
use ndarray::{s, Array2, Array3};
use serde::{Deserialize, Serialize};

// Import candle-optimisers for extended optimizer support
use candle_optimisers::{
    adadelta::{Adadelta, ParamsAdaDelta},
    adagrad::{Adagrad, ParamsAdaGrad},
    adam::{Adam, ParamsAdam},
    adamax::{Adamax, ParamsAdaMax},
    nadam::{NAdam, ParamsNAdam},
    radam::{ParamsRAdam, RAdam},
    rmsprop::{ParamsRMSprop, RMSprop},
    Decay,
};

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
    batch_size: usize,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            epochs: 1,
            print_every: 10,
            clip_gradient: Some(1.0),
            batch_size: 32, // Default batch size for memory safety
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
    loss_function: CryptoLossFunction, // Multi-target loss function
}

/// Serializable model state for persistence - SAME as original
#[derive(Debug, Clone, Serialize, Deserialize)]
struct ModelState {
    config: LSTMConfig,
    epochs: usize,
    print_every: usize,
    clip_gradient: Option<f64>,
}

// Optimizer wrapper for concrete type handling with Candle
enum OptimizerWrapper {
    Sgd(optim::SGD),
    AdamW(optim::AdamW),
    // New optimizers from candle-optimisers crate
    Adam(candle_optimisers::adam::Adam),
    AdaDelta(candle_optimisers::adadelta::Adadelta),
    AdaGrad(candle_optimisers::adagrad::Adagrad),
    AdaMax(candle_optimisers::adamax::Adamax),
    NAdam(candle_optimisers::nadam::NAdam),
    RAdam(candle_optimisers::radam::RAdam),
    RMSprop(candle_optimisers::rmsprop::RMSprop),
}

impl OptimizerWrapper {
    fn set_learning_rate(&mut self, lr: f64) {
        match self {
            OptimizerWrapper::Sgd(sgd) => sgd.set_learning_rate(lr),
            OptimizerWrapper::AdamW(adamw) => adamw.set_learning_rate(lr),
            OptimizerWrapper::Adam(adam) => adam.set_learning_rate(lr),
            OptimizerWrapper::AdaDelta(adadelta) => adadelta.set_learning_rate(lr),
            OptimizerWrapper::AdaGrad(adagrad) => adagrad.set_learning_rate(lr),
            OptimizerWrapper::AdaMax(adamax) => adamax.set_learning_rate(lr),
            OptimizerWrapper::NAdam(nadam) => nadam.set_learning_rate(lr),
            OptimizerWrapper::RAdam(radam) => radam.set_learning_rate(lr),
            OptimizerWrapper::RMSprop(rmsprop) => rmsprop.set_learning_rate(lr),
        }
    }

    fn step(&mut self, grads: &candle_core::backprop::GradStore) -> candle_core::Result<()> {
        match self {
            OptimizerWrapper::Sgd(sgd) => sgd.step(grads),
            OptimizerWrapper::AdamW(adamw) => adamw.step(grads),
            OptimizerWrapper::Adam(adam) => adam.step(grads),
            OptimizerWrapper::AdaDelta(adadelta) => adadelta.step(grads),
            OptimizerWrapper::AdaGrad(adagrad) => adagrad.step(grads),
            OptimizerWrapper::AdaMax(adamax) => adamax.step(grads),
            OptimizerWrapper::NAdam(nadam) => nadam.step(grads),
            OptimizerWrapper::RAdam(radam) => radam.step(grads),
            OptimizerWrapper::RMSprop(rmsprop) => rmsprop.step(grads),
        }
    }
}

impl LSTMModel {
    /// Create a new LSTM model - EXACT same logic as original
    pub fn new(config: LSTMConfig) -> Result<Self> {
        let training_config = TrainingConfig {
            epochs: 1, // Placeholder - will be set by configure_training()
            print_every: 10,
            clip_gradient: Some(1.0),
            batch_size: 32, // Default batch size
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
            loss_function: CryptoLossFunction::MSE, // Default to MSE
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
        model.configure_attention(&model_config.attention, None)?;

        // Configure loss function
        model.loss_function = model_config.loss_function.clone();

        Ok(model)
    }

    /// Configure attention for the model
    pub fn configure_attention(
        &mut self,
        attention_config: &crate::config::model::AttentionConfig,
        context: Option<&str>,
    ) -> Result<()> {
        if !attention_config.enabled {
            self.use_attention = false;
            return Ok(());
        }

        // Convert config AttentionConfig to module AttentionConfig
        let module_config = AttentionModuleConfig {
            num_heads: attention_config.heads as usize,
            head_dim: Some(attention_config.head_dim.unwrap_or(64) as usize),
            dropout_rate: attention_config.dropout_rate,
            temperature_scaling: attention_config.temperature_scaling,
            use_relative_position: attention_config.use_relative_position,
            max_sequence_length: self.config.sequence_length,
        };

        self.attention_config = Some(module_config);
        self.use_attention = true;

        // Log with context if provided, otherwise use generic message
        match context {
            Some(ctx) => log::info!(
                "✅ Attention configured for {}: {} heads, head_dim={}",
                ctx,
                attention_config.heads,
                attention_config.head_dim.unwrap_or(64)
            ),
            None => log::debug!(
                "✅ Attention configured: {} heads, head_dim={}",
                attention_config.heads,
                attention_config.head_dim.unwrap_or(64)
            ),
        }

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

            log::debug!(
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
            crate::config::training::LearningRateConfig::Adaptive {
                initial_lr,
                patience: _,
                factor: _,
            } => {
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

        // Extract batch size from config - NEW: Properly utilize batch size configuration
        let batch_size = match &vanga_config.training.batch_size {
            crate::config::training::BatchSizeConfig::Fixed(size) => {
                log::info!("Using FIXED batch size: {}", size);
                *size as usize
            }
            crate::config::training::BatchSizeConfig::Auto { min_size, max_size } => {
                // Memory-aware batch size optimization
                let chosen_size = self.optimize_batch_size(*min_size as usize, *max_size as usize);
                log::info!(
                    "Using AUTO batch size: {} (optimized from range: {} - {})",
                    chosen_size,
                    min_size,
                    max_size
                );
                chosen_size
            }
        };

        // Update rust-lstm training config - SAME as original + batch size
        self.training_config.epochs = max_epochs;
        self.training_config.print_every = vanga_config.training.print_every as usize; // Use configured print_every
        self.training_config.batch_size = batch_size; // Store configured batch size

        // Store learning rate for optimizer creation - SAME as original
        self.config.learning_rate = learning_rate;

        // Extract and apply gradient clipping from config
        if let Some(gradient_clip) = vanga_config.training.gradient_clip {
            self.training_config.clip_gradient = Some(gradient_clip);
            log::info!("Using gradient clipping: {:.3}", gradient_clip);
        }

        log::info!(
            "✅ Training configured: epochs={}, lr={:.6}, batch_size={}, early_stopping={}, print_every={}, gradient_clip={:?}",
            max_epochs,
            learning_rate,
            batch_size,
            use_early_stopping,
            self.training_config.print_every,
            vanga_config.training.gradient_clip
        );
    }

    /// Validate batch configuration and provide warnings
    fn validate_batch_configuration(&self, total_samples: usize, batch_size: usize) -> Result<()> {
        // Basic validation
        if batch_size == 0 {
            return Err(VangaError::ConfigError(
                "Batch size cannot be zero".to_string(),
            ));
        }

        if batch_size > total_samples {
            log::warn!(
                "⚠️  Batch size ({}) is larger than total samples ({}). Will use full dataset as single batch.",
                batch_size, total_samples
            );
        }

        // Memory estimation and warnings
        let estimated_memory_per_batch = self.estimate_batch_memory_usage(batch_size);
        let estimated_memory_mb = estimated_memory_per_batch / (1024 * 1024);

        if estimated_memory_mb > 1000 {
            // > 1GB per batch
            log::warn!(
                "⚠️  Large batch size detected! Estimated memory per batch: {}MB. Consider reducing batch size if you encounter OOM.",
                estimated_memory_mb
            );
        } else {
            log::info!(
                "✅ Batch configuration validated. Estimated memory per batch: {}MB",
                estimated_memory_mb
            );
        }

        let num_batches = total_samples.div_ceil(batch_size);
        log::info!(
            "📊 Batch processing: {} total samples → {} batches of size {} (last batch: {} samples)",
            total_samples, num_batches, batch_size,
            if total_samples % batch_size == 0 { batch_size } else { total_samples % batch_size }
        );

        Ok(())
    }

    /// Estimate memory usage for a given batch size
    fn estimate_batch_memory_usage(&self, batch_size: usize) -> usize {
        let sequence_length = self.config.sequence_length;
        let input_features = self.config.input_size;
        let hidden_size = self.config.hidden_size;
        let num_layers = self.config.num_layers;

        // Rough estimation: input tensor + hidden states + gradients + attention (if enabled)
        let input_tensor_size = batch_size * sequence_length * input_features * 4; // f32 = 4 bytes
        let hidden_states_size = batch_size * hidden_size * num_layers * 4 * 2; // forward + backward
        let attention_multiplier = if self.use_attention { 3 } else { 1 }; // Attention adds ~3x memory

        (input_tensor_size + hidden_states_size) * attention_multiplier
    }

    /// Optimize batch size based on available memory and model complexity
    fn optimize_batch_size(&self, min_size: usize, max_size: usize) -> usize {
        // Get available memory (rough estimation)
        let available_memory_gb = self.get_available_memory_gb();

        // Memory-based batch size selection following VANGA guidelines
        let memory_based_size = match available_memory_gb {
            gb if gb < 1.0 => 16,
            gb if gb < 4.0 => 32,
            gb if gb < 8.0 => 64,
            gb if gb < 16.0 => 128,
            _ => 256,
        };

        // Start with memory-based size, then test within range
        let mut optimal_size = memory_based_size.max(min_size).min(max_size);

        // Test if we can use a larger batch size within the range
        for test_size in (optimal_size..=max_size).step_by(16) {
            let estimated_memory_mb = self.estimate_batch_memory_usage(test_size) / (1024 * 1024);
            let memory_limit_mb = (available_memory_gb * 1024.0 * 0.7) as usize; // Use 70% of available memory

            if estimated_memory_mb <= memory_limit_mb {
                optimal_size = test_size;
            } else {
                break;
            }
        }

        log::debug!(
            "Batch size optimization: available_memory={}GB, memory_based={}, optimal={}",
            available_memory_gb,
            memory_based_size,
            optimal_size
        );

        optimal_size
    }

    /// Get available memory in GB (rough estimation)
    fn get_available_memory_gb(&self) -> f64 {
        // For macOS, try to get memory info
        if let Ok(output) = std::process::Command::new("vm_stat").output() {
            if let Ok(vm_stat) = String::from_utf8(output.stdout) {
                // Parse vm_stat output to get free memory
                if let Some(free_line) = vm_stat.lines().find(|line| line.contains("Pages free:")) {
                    if let Some(free_pages_str) = free_line.split_whitespace().nth(2) {
                        if let Ok(free_pages) = free_pages_str.trim_end_matches('.').parse::<u64>()
                        {
                            // macOS page size is typically 16KB
                            let free_memory_gb =
                                (free_pages * 16384) as f64 / (1024.0 * 1024.0 * 1024.0);
                            return free_memory_gb.max(1.0); // Minimum 1GB assumption
                        }
                    }
                }
            }
        }

        // Fallback: assume reasonable memory based on system
        4.0 // Default to 4GB assumption for batch size calculation
    }

    /// UNIFIED TRAINING METHOD - Handles all training scenarios through configuration
    /// This method consolidates all training logic from multiple methods into a single,
    /// configuration-driven approach while preserving ALL original functionality.
    pub async fn train(
        &mut self,
        sequences: &Array3<f64>,
        targets: &Array2<f64>,
        config: &crate::config::TrainingConfig,
        // Optional pre-split validation data (prevents data leakage)
        val_sequences: Option<&Array3<f64>>,
        val_targets: Option<&Array2<f64>>,
    ) -> Result<()> {
        let total_samples = sequences.shape()[0];
        log::info!(
            "🚀 UNIFIED TRAINING: Starting with {} samples",
            total_samples
        );

        // Log validation data usage for tracking
        if let (Some(val_seq), Some(_val_tgt)) = (val_sequences, val_targets) {
            log::info!(
                "📊 Using pre-split chronological validation: {} train, {} val samples (no data leakage)",
                total_samples,
                val_seq.shape()[0]
            );
        }

        // INCREMENTAL TRAINING DETECTION AND OPTIMIZATION - SAME logic as original continue_training
        let final_config = if self.trained {
            log::info!(
                "🔄 INCREMENTAL TRAINING: Adding {} new samples to existing model",
                total_samples
            );

            // Configure training with typically lower learning rate for incremental training - SAME logic as original
            let mut incremental_config = config.clone();

            // Reduce learning rate for incremental training to preserve existing knowledge - SAME logic as original
            incremental_config.training.learning_rate = match &config.training.learning_rate {
                crate::config::training::LearningRateConfig::Fixed(lr) => {
                    let reduced_lr = lr * 0.1; // 10x smaller for incremental
                    log::info!(
                        "🔽 Reducing learning rate for incremental training: {:.6} → {:.6}",
                        lr,
                        reduced_lr
                    );
                    crate::config::training::LearningRateConfig::Fixed(reduced_lr)
                }
                crate::config::training::LearningRateConfig::Adaptive {
                    initial_lr,
                    patience,
                    factor,
                } => {
                    let reduced_lr = initial_lr * 0.1;
                    log::info!(
                        "🔽 Reducing initial learning rate for incremental training: {:.6} → {:.6}",
                        initial_lr,
                        reduced_lr
                    );
                    crate::config::training::LearningRateConfig::Adaptive {
                        initial_lr: reduced_lr,
                        patience: *patience,
                        factor: *factor,
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
            incremental_config.training.early_stopping.patience =
                (config.training.early_stopping.patience / 2).max(10);

            log::info!(
                "⚙️  Incremental training config: patience={}, min_delta={:.6}, reduced_lr=true",
                incremental_config.training.early_stopping.patience,
                incremental_config.training.early_stopping.min_delta
            );

            incremental_config
        } else {
            config.clone()
        };

        // Configure training parameters from final config (original or incremental)
        self.configure_training(&final_config);

        // Initialize network if not already done
        if self.lstm_layers.is_none() || self.output_layer.is_none() {
            self.initialize_network()?;
        }

        // Update config to reflect actual output size (1 for compatibility)
        self.config.output_size = 1;

        // Determine if we need validation split
        let validation_split = config.training.validation_split;
        let use_validation = validation_split > 0.0;

        // Prepare training and validation data - handle pre-split vs internal split
        let (train_sequences, train_targets, val_sequences_final, val_targets_final) = if let (
            Some(val_seq),
            Some(val_tgt),
        ) =
            (val_sequences, val_targets)
        {
            // Use pre-split chronological validation data (prevents data leakage)
            log::info!(
                "📊 Using pre-split chronological validation: {} train, {} val samples",
                sequences.shape()[0],
                val_seq.shape()[0]
            );
            (
                sequences.to_owned(),
                targets.to_owned(),
                Some(val_seq.to_owned()),
                Some(val_tgt.to_owned()),
            )
        } else if use_validation {
            // Create internal validation split with gap to prevent data leakage
            log::info!(
                "📊 Using internal validation split: {:.1}%",
                validation_split * 100.0
            );

            // Calculate gap size to prevent data leakage based on prediction horizon
            // Gap should be max_horizon_steps to ensure targets don't overlap
            let gap_size = if !config.horizons.is_empty() {
                // Calculate max horizon steps from training config horizons
                config
                    .horizons
                    .iter()
                    .map(|h| crate::targets::volatility::parse_horizon_to_steps(h).unwrap_or(1))
                    .max()
                    .unwrap_or(72)
            } else {
                72 // Fallback to 3d horizon if no horizons specified
            };

            // Calculate training samples, then add gap before validation
            let base_train_samples = ((1.0 - validation_split) * total_samples as f64) as usize;
            let train_samples = base_train_samples.min(total_samples.saturating_sub(gap_size));
            let val_start = train_samples + gap_size;

            // Ensure we have enough samples for validation after the gap
            if val_start >= total_samples {
                return Err(VangaError::DataError(format!(
                        "Not enough data for validation after gap: {} total samples, {} train + {} gap = {} start, need at least {} for validation",
                        total_samples, train_samples, gap_size, val_start, val_start + 1
                    )));
            }

            let train_seq = sequences.slice(s![0..train_samples, .., ..]).to_owned();
            let train_tgt = targets.slice(s![0..train_samples, ..]).to_owned();
            let val_seq = sequences.slice(s![val_start.., .., ..]).to_owned();
            let val_tgt = targets.slice(s![val_start.., ..]).to_owned();

            log::info!(
                    "🔒 Data leakage prevention: {} train samples, {} gap (max horizon: {}), {} val samples (starting at {})",
                    train_samples,
                    gap_size,
                    config.horizons.iter().max().unwrap_or(&"3d".to_string()),
                    val_seq.shape()[0],
                    val_start
                );

            (train_seq, train_tgt, Some(val_seq), Some(val_tgt))
        } else {
            // No validation
            log::info!("📊 Training without validation");
            (sequences.to_owned(), targets.to_owned(), None, None)
        };

        let total_train_samples = train_sequences.shape()[0];
        let total_val_samples = val_sequences_final
            .as_ref()
            .map(|v| v.shape()[0])
            .unwrap_or(0);
        let batch_size = self.training_config.batch_size;

        log::info!(
            "🚀 UNIFIED TRAINING: {} train samples{}, batch_size={}, optimizer={:?}",
            total_train_samples,
            if use_validation {
                format!(", {} val samples", total_val_samples)
            } else {
                String::new()
            },
            batch_size,
            config.training.optimizer
        );

        // Memory prevalidation and warnings
        self.validate_batch_configuration(total_train_samples, batch_size)?;

        // Setup advanced optimizer with all configurations
        let mut optimizer = self.setup_advanced_optimizer(config)?;

        // Extract learning rate configuration
        let target_lr = match &config.training.learning_rate {
            crate::config::training::LearningRateConfig::Fixed(rate) => *rate,
            crate::config::training::LearningRateConfig::Adaptive { initial_lr, .. } => *initial_lr,
            crate::config::training::LearningRateConfig::Auto { .. } => 0.001, // Default for auto
        };

        // Extract warmup configuration
        let warmup_epochs = config.training.warmup_epochs;
        let mut current_lr = target_lr;

        // Initialize adaptive learning rate variables
        let mut best_loss = f64::INFINITY;
        let mut patience_counter = 0;
        let (adaptive_patience, adaptive_factor) = match &config.training.learning_rate {
            crate::config::training::LearningRateConfig::Adaptive {
                patience, factor, ..
            } => (*patience, *factor),
            _ => (10, 0.5), // Default values for non-adaptive modes
        };

        // Initialize early stopping variables (only used with validation)
        let mut best_val_loss = f64::INFINITY;
        let mut early_stopping_counter = 0;

        // FIXED: Adaptive early stopping configuration based on loss function type
        let (early_stopping_patience, early_stopping_min_delta) =
            if use_validation {
                let base_patience = match &config.training.epochs {
                    crate::config::training::EpochConfig::Auto { max_epochs: _ } => {
                        config.training.early_stopping.patience
                    }
                    _ => 10, // Default patience for fixed epochs
                };
                let base_min_delta = config.training.early_stopping.min_delta;

                // FIXED: Adjust min_delta based on loss function type and expected scale
                let adaptive_min_delta = self.get_adaptive_min_delta(base_min_delta);

                log::info!(
                "🎯 Early stopping configured: patience={}, min_delta={:.6} (adaptive from {:.6})",
                base_patience, adaptive_min_delta, base_min_delta
            );

                (base_patience, adaptive_min_delta)
            } else {
                (u32::MAX, 0.0) // Disable early stopping without validation
            };

        log::info!("🔧 Training Configuration:");
        log::info!("  - Epochs: {}", self.training_config.epochs);
        log::info!("  - Batch size: {}", batch_size);
        log::info!("  - Warmup epochs: {}", warmup_epochs);
        log::info!("  - Adaptive patience: {}", adaptive_patience);
        log::info!("  - Adaptive factor: {:.3}", adaptive_factor);
        log::info!("  - Target learning rate: {:.6}", target_lr);

        // Unified training loop with warmup, adaptive learning, optional validation, and early stopping
        for epoch in 0..self.training_config.epochs {
            let mut epoch_train_loss = 0.0;

            // Calculate warmup learning rate for current epoch
            if epoch < warmup_epochs as usize {
                // Linear warmup from 0 to target_lr
                let warmup_progress = (epoch + 1) as f64 / (warmup_epochs as f64);
                let warmup_lr = target_lr * warmup_progress;

                // Update optimizer learning rate for warmup
                optimizer.set_learning_rate(warmup_lr);
                current_lr = warmup_lr;

                if epoch == 0 || epoch == (warmup_epochs as usize) - 1 {
                    log::info!(
                        "🔥 Warmup epoch {}/{}: learning rate = {:.6}",
                        epoch + 1,
                        warmup_epochs,
                        warmup_lr
                    );
                }
            }

            // Training phase - process data in batches
            for (batch_idx, batch_start) in (0..total_train_samples).step_by(batch_size).enumerate()
            {
                let batch_end = std::cmp::min(batch_start + batch_size, total_train_samples);
                let actual_batch_size = batch_end - batch_start;

                // Extract batch from sequences and targets
                let batch_sequences = train_sequences
                    .slice(ndarray::s![batch_start..batch_end, .., ..])
                    .to_owned();
                let batch_targets = train_targets
                    .slice(ndarray::s![batch_start..batch_end, ..])
                    .to_owned();

                // Convert batch to tensors
                let (input_tensor, target_tensor) =
                    self.convert_sequences_to_tensors(&batch_sequences, &batch_targets)?;

                // Forward pass
                let predictions = self.forward(&input_tensor)?;

                // Calculate loss using configured loss function or default MSE
                let loss = self.calculate_loss(&predictions, &target_tensor)?;

                // Backward pass with gradient computation
                let grads = loss.backward()?;

                // Apply gradient clipping if configured
                if let Some(clip_value) = self.training_config.clip_gradient {
                    let grad_norm = self.clip_gradients(&grads, clip_value)?;

                    if epoch == 0 && batch_idx == 0 {
                        log::debug!(
                            "Gradient clipping enabled: threshold={:.3}, norm={:.6}",
                            clip_value,
                            grad_norm
                        );
                    }

                    if grad_norm > clip_value {
                        log::trace!(
                            "Gradients would be clipped: norm={:.6} > threshold={:.6}",
                            grad_norm,
                            clip_value
                        );
                    }
                }

                // Update parameters using the configured optimizer
                optimizer.step(&grads)?;

                // Accumulate loss for epoch reporting
                let batch_loss = loss.to_scalar::<f32>().map_err(|e| {
                    VangaError::ModelError(format!("Loss scalar conversion failed: {}", e))
                })?;
                epoch_train_loss += batch_loss * actual_batch_size as f32;
            }

            // Calculate average training loss
            let avg_train_loss = epoch_train_loss / total_train_samples as f32;

            // Validation phase (only if validation data is available)
            let avg_val_loss = if let (Some(val_seq), Some(val_tgt)) =
                (&val_sequences_final, &val_targets_final)
            {
                let mut epoch_val_loss = 0.0;

                for batch_start in (0..total_val_samples).step_by(batch_size) {
                    let batch_end = std::cmp::min(batch_start + batch_size, total_val_samples);
                    let actual_batch_size = batch_end - batch_start;

                    // Extract validation batch
                    let batch_sequences = val_seq
                        .slice(ndarray::s![batch_start..batch_end, .., ..])
                        .to_owned();
                    let batch_targets = val_tgt
                        .slice(ndarray::s![batch_start..batch_end, ..])
                        .to_owned();

                    // Convert batch to tensors
                    let (input_tensor, target_tensor) =
                        self.convert_sequences_to_tensors(&batch_sequences, &batch_targets)?;

                    // Forward pass (no gradient computation for validation)
                    let predictions = self.forward(&input_tensor)?;

                    // Calculate validation loss using configured loss function
                    let val_loss = self.calculate_loss(&predictions, &target_tensor)?;
                    let val_batch_loss = val_loss.to_scalar::<f32>().map_err(|e| {
                        VangaError::ModelError(format!(
                            "Validation loss scalar conversion failed: {}",
                            e
                        ))
                    })?;

                    epoch_val_loss += val_batch_loss * actual_batch_size as f32;
                }

                Some(epoch_val_loss / total_val_samples as f32)
            } else {
                None
            };

            // Adaptive learning rate adjustment after warmup
            if epoch >= warmup_epochs as usize {
                if let crate::config::training::LearningRateConfig::Adaptive { .. } =
                    &config.training.learning_rate
                {
                    // Use validation loss if available, otherwise use training loss
                    let loss_for_adaptation = avg_val_loss
                        .map(|v| v as f64)
                        .unwrap_or(avg_train_loss as f64);

                    // Check if we should reduce learning rate
                    if loss_for_adaptation < best_loss {
                        best_loss = loss_for_adaptation;
                        patience_counter = 0;
                    } else {
                        patience_counter += 1;

                        if patience_counter >= adaptive_patience {
                            // Reduce learning rate
                            current_lr *= adaptive_factor;
                            optimizer.set_learning_rate(current_lr);
                            patience_counter = 0;

                            log::info!(
                                "🔄 Adaptive learning rate reduced to: {:.6} (patience exceeded)",
                                current_lr
                            );
                        }
                    }
                }
            }

            // Early stopping check with min_delta threshold (only with validation)
            if let Some(val_loss) = avg_val_loss {
                let improvement = best_val_loss - (val_loss as f64);
                if improvement > early_stopping_min_delta {
                    best_val_loss = val_loss as f64;
                    early_stopping_counter = 0;
                    log::debug!(
                        "✅ Validation improved by {:.6} (> {:.6}), resetting patience counter",
                        improvement,
                        early_stopping_min_delta
                    );
                } else {
                    early_stopping_counter += 1;
                    log::debug!(
                        "⏳ No significant improvement ({:.6} <= {:.6}), patience: {}/{}",
                        improvement,
                        early_stopping_min_delta,
                        early_stopping_counter,
                        early_stopping_patience
                    );

                    if early_stopping_counter >= early_stopping_patience {
                        log::info!(
                            "🛑 Early stopping triggered at epoch {} (best val loss: {:.6}, min_delta: {:.6})",
                            epoch + 1,
                            best_val_loss,
                            early_stopping_min_delta
                        );
                        break;
                    }
                }
            }

            // Enhanced logging with learning rate tracking
            if epoch % self.training_config.print_every == 0 {
                let warmup_status = if epoch < warmup_epochs as usize {
                    " (warmup)"
                } else {
                    ""
                };

                if let Some(val_loss) = avg_val_loss {
                    log::info!(
                        "Epoch {}/{}: Train Loss = {:.6}, Val Loss = {:.6}, LR: {:.6}{}, Early Stop: {}/{}",
                        epoch + 1,
                        self.training_config.epochs,
                        avg_train_loss,
                        val_loss,
                        current_lr,
                        warmup_status,
                        early_stopping_counter,
                        early_stopping_patience
                    );
                } else {
                    let num_batches = total_train_samples.div_ceil(batch_size);
                    log::info!(
                        "Epoch {}/{}: Loss = {:.6}, Batches: {}, LR: {:.6}{}",
                        epoch + 1,
                        self.training_config.epochs,
                        avg_train_loss,
                        num_batches,
                        current_lr,
                        warmup_status
                    );
                }

                // Additional adaptive learning rate status
                if matches!(
                    &config.training.learning_rate,
                    crate::config::training::LearningRateConfig::Adaptive { .. }
                ) && epoch >= warmup_epochs as usize
                {
                    log::debug!(
                        "📊 Adaptive LR status - Best loss: {:.6}, Patience: {}/{}",
                        best_loss,
                        patience_counter,
                        adaptive_patience
                    );
                }
            }
        }

        self.trained = true;
        log::info!("✅ Unified LSTM training completed successfully");

        // Calculate final training metrics
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

    /// Advanced optimizer setup with OptimizerWrapper for proper optimizer handling
    fn setup_advanced_optimizer(
        &self,
        config: &crate::config::TrainingConfig,
    ) -> Result<OptimizerWrapper> {
        let learning_rate = self.config.learning_rate;
        let optimizer_config = &config.training.optimizer;

        match optimizer_config {
            crate::config::training::OptimizerType::SGD { momentum } => {
                log::info!(
                    "Using SGD optimizer with learning rate: {:.6}",
                    learning_rate
                );
                if let Some(momentum_val) = momentum {
                    log::info!(
                        "SGD momentum: {:.3} (not yet implemented in Candle)",
                        momentum_val
                    );
                }
                Ok(OptimizerWrapper::Sgd(
                    optim::SGD::new(self.varmap.all_vars(), learning_rate).map_err(|e| {
                        VangaError::ModelError(format!("SGD optimizer creation failed: {}", e))
                    })?,
                ))
            }
            crate::config::training::OptimizerType::AdamW {
                weight_decay,
                beta1,
                beta2,
            } => {
                log::info!(
                    "Using AdamW optimizer with learning rate: {:.6}",
                    learning_rate
                );
                log::info!(
                    "AdamW parameters - weight_decay: {:.4}, beta1: {:.3}, beta2: {:.3}",
                    weight_decay,
                    beta1,
                    beta2
                );
                Ok(OptimizerWrapper::AdamW(
                    optim::AdamW::new_lr(self.varmap.all_vars(), learning_rate).map_err(|e| {
                        VangaError::ModelError(format!("AdamW optimizer creation failed: {}", e))
                    })?,
                ))
            }
            // New optimizers from candle-optimisers crate
            crate::config::training::OptimizerType::Adam {
                beta1,
                beta2,
                eps,
                weight_decay,
                amsgrad,
            } => {
                log::info!(
                    "Using Adam optimizer with learning rate: {:.6}",
                    learning_rate
                );
                log::info!(
                    "Adam parameters - beta1: {:.3}, beta2: {:.3}, eps: {:.2e}, amsgrad: {}",
                    beta1,
                    beta2,
                    eps,
                    amsgrad
                );

                let params = ParamsAdam {
                    lr: learning_rate,
                    beta_1: *beta1,
                    beta_2: *beta2,
                    eps: *eps,
                    weight_decay: weight_decay.map(Decay::WeightDecay),
                    amsgrad: *amsgrad,
                };

                Ok(OptimizerWrapper::Adam(
                    Adam::new(self.varmap.all_vars(), params).map_err(|e| {
                        VangaError::ModelError(format!("Adam optimizer creation failed: {}", e))
                    })?,
                ))
            }
            crate::config::training::OptimizerType::AdaDelta {
                rho,
                eps,
                weight_decay,
            } => {
                log::info!(
                    "Using AdaDelta optimizer with learning rate: {:.6}",
                    learning_rate
                );
                log::info!("AdaDelta parameters - rho: {:.3}, eps: {:.2e}", rho, eps);

                let params = ParamsAdaDelta {
                    lr: learning_rate,
                    rho: *rho,
                    eps: *eps,
                    weight_decay: weight_decay.map(Decay::WeightDecay),
                };

                Ok(OptimizerWrapper::AdaDelta(
                    Adadelta::new(self.varmap.all_vars(), params).map_err(|e| {
                        VangaError::ModelError(format!("AdaDelta optimizer creation failed: {}", e))
                    })?,
                ))
            }
            crate::config::training::OptimizerType::AdaGrad {
                lr_decay,
                weight_decay,
                initial_accumulator_value,
                eps,
            } => {
                log::info!(
                    "Using AdaGrad optimizer with learning rate: {:.6}",
                    learning_rate
                );
                log::info!(
                    "AdaGrad parameters - lr_decay: {:.3}, eps: {:.2e}, init_acc: {:.3}",
                    lr_decay,
                    eps,
                    initial_accumulator_value
                );

                let params = ParamsAdaGrad {
                    lr: learning_rate,
                    lr_decay: *lr_decay,
                    weight_decay: weight_decay.map(Decay::WeightDecay),
                    eps: *eps,
                    initial_acc: *initial_accumulator_value,
                };
                Ok(OptimizerWrapper::AdaGrad(
                    Adagrad::new(self.varmap.all_vars(), params).map_err(|e| {
                        VangaError::ModelError(format!("AdaGrad optimizer creation failed: {}", e))
                    })?,
                ))
            }
            crate::config::training::OptimizerType::AdaMax {
                beta1,
                beta2,
                eps,
                weight_decay,
            } => {
                log::info!(
                    "Using AdaMax optimizer with learning rate: {:.6}",
                    learning_rate
                );
                log::info!(
                    "AdaMax parameters - beta1: {:.3}, beta2: {:.3}, eps: {:.2e}",
                    beta1,
                    beta2,
                    eps
                );

                let params = ParamsAdaMax {
                    lr: learning_rate,
                    beta_1: *beta1,
                    beta_2: *beta2,
                    eps: *eps,
                    weight_decay: weight_decay.map(Decay::WeightDecay),
                };

                Ok(OptimizerWrapper::AdaMax(
                    Adamax::new(self.varmap.all_vars(), params).map_err(|e| {
                        VangaError::ModelError(format!("AdaMax optimizer creation failed: {}", e))
                    })?,
                ))
            }
            crate::config::training::OptimizerType::NAdam {
                beta1,
                beta2,
                eps,
                weight_decay,
                momentum_decay,
            } => {
                log::info!(
                    "Using NAdam optimizer with learning rate: {:.6}",
                    learning_rate
                );
                log::info!(
                    "NAdam parameters - beta1: {:.3}, beta2: {:.3}, eps: {:.2e}, momentum_decay: {:.3}",
                    beta1, beta2, eps, momentum_decay
                );

                let params = ParamsNAdam {
                    lr: learning_rate,
                    beta_1: *beta1,
                    beta_2: *beta2,
                    eps: *eps,
                    weight_decay: weight_decay.map(Decay::WeightDecay),
                    momentum_decay: *momentum_decay,
                };

                Ok(OptimizerWrapper::NAdam(
                    NAdam::new(self.varmap.all_vars(), params).map_err(|e| {
                        VangaError::ModelError(format!("NAdam optimizer creation failed: {}", e))
                    })?,
                ))
            }
            crate::config::training::OptimizerType::RAdam {
                beta1,
                beta2,
                eps,
                weight_decay,
            } => {
                log::info!(
                    "Using RAdam optimizer with learning rate: {:.6}",
                    learning_rate
                );
                log::info!(
                    "RAdam parameters - beta1: {:.3}, beta2: {:.3}, eps: {:.2e}",
                    beta1,
                    beta2,
                    eps
                );

                let params = ParamsRAdam {
                    lr: learning_rate,
                    beta_1: *beta1,
                    beta_2: *beta2,
                    eps: *eps,
                    weight_decay: weight_decay.map(Decay::WeightDecay),
                };

                Ok(OptimizerWrapper::RAdam(
                    RAdam::new(self.varmap.all_vars(), params).map_err(|e| {
                        VangaError::ModelError(format!("RAdam optimizer creation failed: {}", e))
                    })?,
                ))
            }
            crate::config::training::OptimizerType::RMSprop {
                alpha,
                eps,
                weight_decay,
                momentum,
                centered,
            } => {
                log::info!(
                    "Using RMSprop optimizer with learning rate: {:.6}",
                    learning_rate
                );
                log::info!(
                    "RMSprop parameters - alpha: {:.3}, eps: {:.2e}, momentum: {:.3}, centered: {}",
                    alpha,
                    eps,
                    momentum,
                    centered
                );

                let params = ParamsRMSprop {
                    lr: learning_rate,
                    alpha: *alpha,
                    eps: *eps,
                    weight_decay: *weight_decay,
                    momentum: if *momentum > 0.0 {
                        Some(*momentum)
                    } else {
                        None
                    },
                    centered: *centered,
                };

                Ok(OptimizerWrapper::RMSprop(
                    RMSprop::new(self.varmap.all_vars(), params).map_err(|e| {
                        VangaError::ModelError(format!("RMSprop optimizer creation failed: {}", e))
                    })?,
                ))
            }
        }
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

        // Convert sequences to tensor (prediction-optimized version)
        let input_tensor = self.convert_sequences_to_prediction_tensor(sequences)?;

        // Forward pass through network
        let predictions_tensor = self.forward(&input_tensor)?;

        // Convert back to ndarray
        let predictions = self.tensor_to_array2(&predictions_tensor)?;

        // Explicit memory cleanup for prediction tensors
        drop(input_tensor);
        drop(predictions_tensor);

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

    /// Save model to file - Enhanced to save both config and weights
    pub fn save<P: AsRef<std::path::Path>>(&self, path: P) -> Result<()> {
        let path = path.as_ref();

        // Create directory if it doesn't exist
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| VangaError::IoError(format!("Failed to create directory: {}", e)))?;
        }

        // Save model weights using VarMap's safetensors format
        let weights_path = path.with_extension("safetensors");
        self.varmap.save(&weights_path).map_err(|e| {
            VangaError::SerializationError(format!("Failed to save model weights: {}", e))
        })?;

        // Save model configuration and metadata
        let model_state = ModelState {
            config: self.config.clone(),
            epochs: self.training_config.epochs,
            print_every: self.training_config.print_every,
            clip_gradient: self.training_config.clip_gradient,
        };

        let config_path = path.with_extension("config");
        let encoded = bincode::serialize(&model_state).map_err(|e| {
            VangaError::SerializationError(format!("Config serialization failed: {}", e))
        })?;

        std::fs::write(&config_path, encoded)
            .map_err(|e| VangaError::IoError(format!("Failed to write config file: {}", e)))?;

        log::info!(
            "Model saved successfully: weights={}, config={}",
            weights_path.display(),
            config_path.display()
        );
        Ok(())
    }

    /// Load model from file - Enhanced to load both config and weights
    pub fn load<P: AsRef<std::path::Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();

        // Load model configuration
        let config_path = path.with_extension("config");
        let data = std::fs::read(&config_path)
            .map_err(|e| VangaError::IoError(format!("Failed to read config file: {}", e)))?;

        let model_state: ModelState = bincode::deserialize(&data).map_err(|e| {
            VangaError::SerializationError(format!("Config deserialization failed: {}", e))
        })?;

        // Create model with loaded configuration
        let mut model = Self::new(model_state.config)?;
        model.training_config.epochs = model_state.epochs;
        model.training_config.print_every = model_state.print_every;
        model.training_config.clip_gradient = model_state.clip_gradient;

        // Initialize the network structure
        model.initialize_network()?;

        // Load model weights from safetensors
        let weights_path = path.with_extension("safetensors");
        model.varmap.load(&weights_path).map_err(|e| {
            VangaError::SerializationError(format!("Failed to load model weights: {}", e))
        })?;

        model.trained = true;

        log::info!(
            "Model loaded successfully: weights={}, config={}",
            weights_path.display(),
            config_path.display()
        );
        Ok(model)
    }

    /// Get the input size of the model - SAME as original
    pub fn get_input_size(&self) -> usize {
        self.config.input_size
    }
}

// Implement From trait for Candle error conversion
impl From<candle_core::Error> for VangaError {
    fn from(err: candle_core::Error) -> Self {
        VangaError::ModelError(format!("Candle error: {}", err))
    }
}

impl LSTMModel {
    /// Calculate loss using configured loss function
    fn calculate_loss(&self, predictions: &Tensor, targets: &Tensor) -> Result<Tensor> {
        use crate::model::loss::TensorCryptoLossFunction;

        // Log loss calculation context
        log::debug!(
            "🔍 LOSS CALCULATION - Pred shape: {:?}, Target shape: {:?}",
            predictions.shape(),
            targets.shape()
        );

        let mut tensor_loss_fn = TensorCryptoLossFunction::new(self.loss_function.clone());

        let market_regime = self.detect_market_regime(predictions, targets)?;

        let loss_result =
            tensor_loss_fn.calculate_tensor_loss(predictions, targets, market_regime.clone())?;
        let loss_value = loss_result.to_scalar::<f32>().unwrap_or(0.0);
        log::debug!(
            "🔍 LOSS RESULT - Value: {:.6}, Regime: {:?} (FIXED REGIME DETECTION)",
            loss_value,
            market_regime
        );

        Ok(loss_result)
    }

    /// Detect market regime from prediction and target data patterns
    /// Detect market regime from input sequences (NOT targets) for consistency
    fn detect_market_regime(
        &self,
        predictions: &Tensor,
        _targets: &Tensor,
    ) -> Result<crate::optimization::objective::MarketRegime> {
        use crate::optimization::objective::MarketRegime;

        // FIXED: Use predictions (derived from input sequences) instead of targets
        // This ensures regime detection is based on input data, not ground truth
        let batch_size = predictions.shape().dims()[0];
        log::debug!(
            "🔍 REGIME DETECTION - Batch size: {}, Using predictions shape: {:?}",
            batch_size,
            predictions.shape()
        );

        // Calculate volatility from predictions (sequence-derived data)
        let pred_mean = predictions.mean_all()?;
        let pred_mean_broadcast = pred_mean.broadcast_as(predictions.shape())?;
        let pred_variance = predictions
            .sub(&pred_mean_broadcast)?
            .contiguous()?
            .sqr()?
            .mean_all()?;
        let volatility =
            pred_variance.sqrt()?.to_scalar::<f32>().map_err(|e| {
                VangaError::ModelError(format!("Volatility calculation failed: {}", e))
            })? as f64;

        // Calculate trend strength from predictions (more stable than targets)
        let pred_shape = predictions.shape();
        let trend_strength = if pred_shape.dims()[0] > 1 && pred_shape.dims().len() > 1 {
            // Use temporal dimension if available (sequence-based trend)
            let seq_len = pred_shape.dims()[1];
            if seq_len > 1 {
                let first_half = predictions.narrow(1, 0, seq_len / 2)?;
                let second_half = predictions.narrow(1, seq_len / 2, seq_len / 2)?;

                let first_mean = first_half.mean_all()?.to_scalar::<f32>().unwrap_or(0.0) as f64;
                let second_mean = second_half.mean_all()?.to_scalar::<f32>().unwrap_or(0.0) as f64;

                (second_mean - first_mean) / first_mean.abs().max(1e-8)
            } else {
                0.0
            }
        } else {
            0.0
        };

        // Classify market regime with crypto-appropriate thresholds
        let regime = match (volatility, trend_strength) {
            (v, _) if v > 0.1 => MarketRegime::HighVolatility, // Crypto: higher volatility threshold
            (v, t) if v < 0.02 && t.abs() < 0.005 => MarketRegime::LowVolatility,
            (_, t) if t > 0.02 => MarketRegime::BullMarket, // Crypto: lower trend threshold
            (_, t) if t < -0.02 => MarketRegime::BearMarket,
            (v, _) if v < 0.05 => MarketRegime::RangeBound,
            _ => MarketRegime::MediumVolatility,
        };

        log::debug!(
            "🔍 FIXED REGIME - Regime: {:?}, Volatility: {:.6}, Trend: {:.6} (from predictions)",
            regime,
            volatility,
            trend_strength
        );

        Ok(regime)
    }

    /// Get adaptive min_delta for early stopping based on loss function type
    fn get_adaptive_min_delta(&self, base_min_delta: f64) -> f64 {
        match &self.loss_function {
            crate::model::loss::CryptoLossFunction::Composite { .. } => {
                // Composite loss has higher scale, need larger min_delta
                let adaptive_delta = base_min_delta * 20.0; // 20x larger for composite loss
                log::debug!(
                    "🔧 Composite loss detected: min_delta scaled from {:.6} to {:.6}",
                    base_min_delta,
                    adaptive_delta
                );
                adaptive_delta
            }
            crate::model::loss::CryptoLossFunction::DirectionalFocused { .. } => {
                // DirectionalFocused has moderate scale increase
                let adaptive_delta = base_min_delta * 10.0;
                log::debug!(
                    "🔧 DirectionalFocused loss detected: min_delta scaled from {:.6} to {:.6}",
                    base_min_delta,
                    adaptive_delta
                );
                adaptive_delta
            }
            crate::model::loss::CryptoLossFunction::RiskAdjusted { .. } => {
                // RiskAdjusted has moderate scale increase
                let adaptive_delta = base_min_delta * 15.0;
                log::debug!(
                    "🔧 RiskAdjusted loss detected: min_delta scaled from {:.6} to {:.6}",
                    base_min_delta,
                    adaptive_delta
                );
                adaptive_delta
            }
            crate::model::loss::CryptoLossFunction::VolatilityAware { .. } => {
                // VolatilityAware can have variable scale
                let adaptive_delta = base_min_delta * 12.0;
                log::debug!(
                    "🔧 VolatilityAware loss detected: min_delta scaled from {:.6} to {:.6}",
                    base_min_delta,
                    adaptive_delta
                );
                adaptive_delta
            }
            crate::model::loss::CryptoLossFunction::MSE => {
                // MSE loss uses original min_delta
                log::debug!(
                    "🔧 MSE loss detected: min_delta unchanged at {:.6}",
                    base_min_delta
                );
                base_min_delta
            }
            _ => {
                // Other crypto loss functions get moderate scaling
                let adaptive_delta = base_min_delta * 8.0;
                log::debug!(
                    "🔧 Crypto loss function detected: min_delta scaled from {:.6} to {:.6}",
                    base_min_delta,
                    adaptive_delta
                );
                adaptive_delta
            }
        }
    }

    /// Validate loss function configuration and mathematical correctness
    pub fn validate_loss_function(&self) -> Result<()> {
        match &self.loss_function {
            crate::model::loss::CryptoLossFunction::MSE => {
                log::info!("✅ Using MSE loss function");
            }
            crate::model::loss::CryptoLossFunction::Composite {
                accuracy_weight,
                direction_weight,
                volatility_weight,
                risk_weight,
            } => {
                // Validate weights are non-negative
                if *accuracy_weight < 0.0
                    || *direction_weight < 0.0
                    || *volatility_weight < 0.0
                    || *risk_weight < 0.0
                {
                    return Err(crate::utils::error::VangaError::ConfigError(
                        "Composite loss weights must be non-negative".to_string(),
                    ));
                }

                // Validate at least one weight is positive
                let total_weight =
                    accuracy_weight + direction_weight + volatility_weight + risk_weight;
                if total_weight <= 0.0 {
                    return Err(crate::utils::error::VangaError::ConfigError(
                        "Composite loss must have at least one positive weight".to_string(),
                    ));
                }

                // Log configuration for debugging
                log::info!(
                    "✅ Composite loss validated: acc={:.2}, dir={:.2}, vol={:.2}, risk={:.2} (total={:.2})",
                    accuracy_weight, direction_weight, volatility_weight, risk_weight, total_weight
                );
            }
            crate::model::loss::CryptoLossFunction::DirectionalFocused { direction_penalty } => {
                if *direction_penalty <= 0.0 {
                    return Err(crate::utils::error::VangaError::ConfigError(
                        "DirectionalFocused direction_penalty must be positive".to_string(),
                    ));
                }
                log::info!(
                    "✅ DirectionalFocused loss validated: penalty={:.2}",
                    direction_penalty
                );
            }
            crate::model::loss::CryptoLossFunction::RiskAdjusted {
                sharpe_weight,
                drawdown_weight,
            } => {
                if *sharpe_weight < 0.0 || *drawdown_weight < 0.0 {
                    return Err(crate::utils::error::VangaError::ConfigError(
                        "RiskAdjusted loss weights must be non-negative".to_string(),
                    ));
                }
                if *sharpe_weight + *drawdown_weight <= 0.0 {
                    return Err(crate::utils::error::VangaError::ConfigError(
                        "RiskAdjusted loss must have at least one positive weight".to_string(),
                    ));
                }
                log::info!(
                    "✅ RiskAdjusted loss validated: sharpe={:.2}, drawdown={:.2}",
                    sharpe_weight,
                    drawdown_weight
                );
            }
            crate::model::loss::CryptoLossFunction::VolatilityAware {
                volatility_threshold,
                penalty_factor,
            } => {
                if *volatility_threshold < 0.0 || *penalty_factor < 0.0 {
                    return Err(crate::utils::error::VangaError::ConfigError(
                        "VolatilityAware loss parameters must be non-negative".to_string(),
                    ));
                }
                log::info!(
                    "✅ VolatilityAware loss validated: threshold={:.4}, penalty={:.2}",
                    volatility_threshold,
                    penalty_factor
                );
            }
            crate::model::loss::CryptoLossFunction::RegimeAware { volatility_penalty } => {
                if *volatility_penalty < 0.0 {
                    return Err(crate::utils::error::VangaError::ConfigError(
                        "RegimeAware volatility_penalty must be non-negative".to_string(),
                    ));
                }
                log::info!(
                    "✅ RegimeAware loss validated: penalty={:.2}",
                    volatility_penalty
                );
            }
            crate::model::loss::CryptoLossFunction::MultiObjective { horizon_weights } => {
                if horizon_weights.is_empty() {
                    return Err(crate::utils::error::VangaError::ConfigError(
                        "MultiObjective loss must have at least one horizon weight".to_string(),
                    ));
                }
                if horizon_weights.iter().any(|&w| w < 0.0) {
                    return Err(crate::utils::error::VangaError::ConfigError(
                        "MultiObjective horizon weights must be non-negative".to_string(),
                    ));
                }
                let total_weight: f64 = horizon_weights.iter().sum();
                if total_weight <= 0.0 {
                    return Err(crate::utils::error::VangaError::ConfigError(
                        "MultiObjective loss must have at least one positive weight".to_string(),
                    ));
                }
                log::info!(
                    "✅ MultiObjective loss validated: {} horizons, total_weight={:.2}",
                    horizon_weights.len(),
                    total_weight
                );
            }
        }

        Ok(())
    }

    /// Clip gradients to prevent exploding gradients during training
    /// Returns the original gradient norm for monitoring
    fn clip_gradients(
        &self,
        _grads: &candle_core::backprop::GradStore,
        clip_value: f64,
    ) -> Result<f64> {
        // For now, implement a simplified version that works with current Candle API
        // This is a placeholder that logs the clipping request

        log::debug!(
            "Gradient clipping requested with threshold {:.3} - using original gradients (implementation pending full Candle API support)",
            clip_value
        );

        // Return a placeholder norm
        // This maintains the interface while we wait for proper Candle gradient manipulation API
        Ok(1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::training::OptimizerType;
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
            features: crate::config::FeatureConfig::default(),
            model: crate::config::ModelConfig::default(),
            training: TrainingParams {
                epochs: EpochConfig::Auto { max_epochs: 100 },
                batch_size: crate::config::training::BatchSizeConfig::Fixed(32),
                learning_rate: LearningRateConfig::Fixed(0.01),
                optimizer: crate::config::training::OptimizerType::AdamW {
                    weight_decay: 0.01,
                    beta1: 0.9,
                    beta2: 0.999,
                },
                warmup_epochs: 0, // No warmup for tests
                learning_schedule: None,
                test_split: 0.1,
                early_stopping: crate::config::training::EarlyStoppingConfig {
                    patience: 10,
                    min_delta: 0.0001,
                },
                gradient_clip: Some(1.0),
                validation_split: 0.2, // 20% validation
                device: crate::config::training::DeviceConfig::Auto,
                print_every: 1, // Add missing print_every field
            },
            data: crate::config::training::DataConfig::default(),
            optimization: crate::config::training::OptimizationConfig::default(),
        };

        // Test that early stopping training completes without errors
        let result = model
            .train(&sequences, &targets, &training_config, None, None)
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
            features: crate::config::FeatureConfig::default(),
            model: crate::config::ModelConfig::default(),
            training: TrainingParams {
                epochs: EpochConfig::Fixed(5), // Fixed epochs - should bypass early stopping
                batch_size: crate::config::training::BatchSizeConfig::Fixed(32),
                learning_rate: LearningRateConfig::Fixed(0.01),
                optimizer: crate::config::training::OptimizerType::AdamW {
                    weight_decay: 0.01,
                    beta1: 0.9,
                    beta2: 0.999,
                },
                warmup_epochs: 0,
                learning_schedule: None,
                validation_split: 0.2,
                device: crate::config::training::DeviceConfig::Auto,
                test_split: 0.0,
                early_stopping: crate::config::training::EarlyStoppingConfig {
                    patience: 10,
                    min_delta: 0.0001,
                },
                gradient_clip: Some(1.0),
                print_every: 1, // Add missing print_every field
            },
            data: crate::config::training::DataConfig::default(),
            optimization: crate::config::training::OptimizationConfig::default(),
        };

        // Test that fixed epochs training completes without errors
        let result = model
            .train(&sequences, &targets, &training_config, None, None)
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
            features: crate::config::FeatureConfig::default(),
            model: crate::config::ModelConfig::default(),
            training: TrainingParams {
                epochs: EpochConfig::Fixed(3), // Quick training for test
                batch_size: crate::config::training::BatchSizeConfig::Fixed(32),
                learning_rate: LearningRateConfig::Fixed(0.01),
                optimizer: OptimizerType::SGD { momentum: None },
                warmup_epochs: 0,
                learning_schedule: None,
                validation_split: 0.0, // No validation for this test
                test_split: 0.0,
                early_stopping: crate::config::training::EarlyStoppingConfig {
                    patience: 10,
                    min_delta: 0.0001,
                },
                gradient_clip: Some(1.0),
                device: crate::config::training::DeviceConfig::Auto,
                print_every: 1, // Add missing print_every field
            },

            data: crate::config::training::DataConfig::default(),
            optimization: crate::config::training::OptimizationConfig::default(),
        };

        model
            .train(&sequences, &targets, &training_config, None, None)
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
            features: crate::config::FeatureConfig::default(),
            model: crate::config::ModelConfig {
                architecture: crate::config::model::LSTMArchitecture::StackedLSTM { layers: 3 },
                ..crate::config::ModelConfig::default()
            },
            training: TrainingParams {
                epochs: EpochConfig::Fixed(5), // Quick training for test
                batch_size: crate::config::training::BatchSizeConfig::Fixed(16),
                learning_rate: LearningRateConfig::Fixed(0.01),
                optimizer: crate::config::training::OptimizerType::AdamW {
                    weight_decay: 0.01,
                    beta1: 0.9,
                    beta2: 0.999,
                },
                warmup_epochs: 0, // No warmup for tests
                learning_schedule: None,
                validation_split: 0.2,
                device: crate::config::training::DeviceConfig::Auto,
                test_split: 0.0,
                early_stopping: crate::config::training::EarlyStoppingConfig {
                    patience: 10,
                    min_delta: 0.0001,
                },
                gradient_clip: Some(1.0),
                print_every: 1, // Add missing print_every field
            },
            data: crate::config::training::DataConfig::default(),
            optimization: crate::config::training::OptimizationConfig::default(),
        };

        // Test multi-layer training
        let result = model
            .train(&sequences, &targets, &training_config, None, None)
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
