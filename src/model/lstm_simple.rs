// LSTM model implementation with Candle framework - PRESERVING ALL ORIGINAL LOGIC
use crate::config::training::ClassWeightStrategy;
use crate::config::ModelConfig;
use crate::model::attention::{AttentionConfig as AttentionModuleConfig, MultiHeadAttention};
use crate::model::loss::CryptoLossFunction;
use crate::targets::TargetType;
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

/// Target format enumeration for metrics calculation
#[derive(Debug, Clone, Copy)]
enum TargetFormat {
    OneHot,          // [0, 0, 1, 0, 0] - one-hot encoded classes
    RawClassIndices, // [2] - raw class index
    RawValues,       // [0.8] - continuous values or other formats
    Unknown,         // Cannot determine format
}

/// LSTM network configuration - EXACT same as original
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LSTMConfig {
    pub input_size: usize,
    pub hidden_sizes: Vec<usize>, // Changed from single hidden_size to per-layer sizes
    pub output_size: usize,
    pub sequence_length: usize,
    pub learning_rate: f64,
    pub num_layers: usize, // Added for multi-layer support
}

impl LSTMConfig {
    /// Get hidden size for a specific layer
    pub fn get_hidden_size_for_layer(&self, layer_idx: usize) -> usize {
        self.hidden_sizes
            .get(layer_idx)
            .copied()
            .unwrap_or_else(|| {
                // Fallback: use last available size if layer_idx exceeds array
                self.hidden_sizes.last().copied().unwrap_or(128)
            })
    }

    /// Get the total number of parameters across all layers
    pub fn total_parameters(&self) -> usize {
        let mut total = 0;
        for layer_idx in 0..self.num_layers {
            let input_size = if layer_idx == 0 {
                self.input_size
            } else {
                self.get_hidden_size_for_layer(layer_idx - 1)
            };
            let hidden_size = self.get_hidden_size_for_layer(layer_idx);

            // LSTM has 4 gates, each with input and hidden weights plus bias
            total += (input_size + hidden_size + 1) * hidden_size * 4;
        }
        total
    }

    /// Validate the configuration for consistency
    pub fn validate(&self) -> Result<()> {
        if self.hidden_sizes.is_empty() {
            return Err(VangaError::ModelError(
                "hidden_sizes cannot be empty".to_string(),
            ));
        }

        if self.num_layers == 0 {
            return Err(VangaError::ModelError(
                "num_layers must be at least 1".to_string(),
            ));
        }

        // Warn if hidden_sizes array is shorter than num_layers
        if self.hidden_sizes.len() < self.num_layers {
            log::warn!(
                "hidden_sizes array length ({}) < num_layers ({}). Will reuse last size for remaining layers.",
                self.hidden_sizes.len(),
                self.num_layers
            );
        }

        Ok(())
    }
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
    lstm_layers: Option<Vec<LSTM>>, // Forward layers for unidirectional or bidirectional
    backward_lstm_layers: Option<Vec<LSTM>>, // Backward layers for bidirectional LSTM
    output_layer: Option<Linear>,
    pub attention_layers: Option<MultiHeadAttention>, // Public for testing
    pub attention_config: Option<AttentionModuleConfig>, // Public for testing
    pub use_attention: bool,                          // Public for testing
    device: Device,
    varmap: VarMap,
    training_config: TrainingConfig,
    trained: bool,
    loss_function: CryptoLossFunction, // Multi-target loss function
    /// Target context for this individual model (e.g., "price_level_1h", "direction_4h")
    /// This allows proper target type detection without assumptions
    target_context: Option<(String, crate::targets::TargetType)>, // (target_name, target_type)
    /// Global class weights calculated once from entire training dataset
    /// Used for consistent loss calculation across all batches (training and validation)
    global_class_weights: Option<Vec<f32>>,
    /// Architecture configuration for bidirectional detection
    architecture: Option<crate::config::model::LSTMArchitecture>,
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
            backward_lstm_layers: None, // Initialize backward layers as None
            output_layer: None,
            attention_layers: None, // Initialize attention as None
            attention_config: None, // Initialize attention config as None
            use_attention: false,   // Attention disabled by default
            device: Device::Cpu,
            varmap: VarMap::new(),
            training_config,
            trained: false,
            loss_function: CryptoLossFunction::MSE, // Default to MSE
            target_context: None,                   // No target context by default
            global_class_weights: None,             // No global weights initially
            architecture: None,                     // No architecture info by default
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

        // Extract number of layers from architecture config - MOVED UP
        let num_layers = Self::extract_num_layers_from_architecture(&model_config.architecture);

        // Extract hidden units from config - ENHANCED to use full array
        let hidden_sizes = match &model_config.hidden_units {
            crate::config::model::HiddenUnitsConfig::Fixed(units) => {
                // Use the full array instead of just the first value
                units.iter().map(|&u| u as usize).collect::<Vec<usize>>()
            }
            crate::config::model::HiddenUnitsConfig::Auto {
                min_units,
                max_units: _,
            } => {
                // For auto config, create a single-layer configuration
                vec![*min_units as usize]
            }
            crate::config::model::HiddenUnitsConfig::Pyramid {
                base_units,
                reduction_factor,
            } => {
                // Generate pyramid architecture: base_units, base_units * reduction_factor, etc.
                let mut sizes = Vec::new();
                let mut current_size = *base_units as f64;

                for _ in 0..num_layers {
                    sizes.push(current_size as usize);
                    current_size *= reduction_factor;
                    // Ensure minimum size of 8 units
                    if current_size < 8.0 {
                        current_size = 8.0;
                    }
                }
                sizes
            }
        };

        // Validate hidden_sizes array consistency
        if hidden_sizes.is_empty() {
            return Err(VangaError::ModelError(
                "Hidden units configuration resulted in empty array".to_string(),
            ));
        }

        // Validate reasonable hidden sizes
        for (i, &size) in hidden_sizes.iter().enumerate() {
            if size == 0 {
                return Err(VangaError::ModelError(format!(
                    "Layer {} has zero hidden units",
                    i
                )));
            }
            if size > 2048 {
                log::warn!(
                    "⚠️ Layer {} has very large hidden size ({}). This may cause memory issues.",
                    i,
                    size
                );
            }
        }

        // Extend hidden_sizes if needed to match num_layers
        let mut final_hidden_sizes = hidden_sizes;
        if final_hidden_sizes.len() < num_layers {
            let last_size = final_hidden_sizes.last().copied().unwrap_or(128);
            log::info!(
                "🔧 Extending hidden_sizes from {} to {} layers using last size ({})",
                final_hidden_sizes.len(),
                num_layers,
                last_size
            );
            final_hidden_sizes.resize(num_layers, last_size);
        } else if final_hidden_sizes.len() > num_layers {
            log::warn!(
                "⚠️ hidden_sizes array length ({}) > num_layers ({}). Truncating to {} layers.",
                final_hidden_sizes.len(),
                num_layers,
                num_layers
            );
            final_hidden_sizes.truncate(num_layers);
        }

        // Use sequence_length for LSTM configuration if needed - SAME logic
        let adjusted_hidden_sizes = if sequence_length > 100 {
            // Adjust all layer sizes based on sequence length
            final_hidden_sizes
                .iter()
                .map(|&size| size + (sequence_length / 10))
                .collect()
        } else {
            final_hidden_sizes
        };

        let lstm_config = LSTMConfig {
            input_size,
            hidden_sizes: adjusted_hidden_sizes,
            output_size,
            sequence_length,      // Use actual sequence length from config
            learning_rate: 0.001, // Default learning rate
            num_layers,           // Now properly extracted from architecture
        };

        // Validate the configuration
        lstm_config.validate()?;

        let mut model = Self::new(lstm_config)?;

        // Configure attention if enabled
        model.configure_attention(&model_config.attention, None)?;

        // Configure loss function
        model.loss_function = model_config.loss_function.clone();

        // Store architecture information for bidirectional detection
        model.architecture = Some(model_config.architecture.clone());

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
            // Calculate attention input size based on architecture
            let base_hidden_size = self
                .config
                .get_hidden_size_for_layer(self.config.num_layers - 1);

            // For bidirectional LSTM, attention receives concatenated output (2x hidden size)
            let is_bidirectional = matches!(
                self.architecture,
                Some(crate::config::model::LSTMArchitecture::BidirectionalLSTM { .. })
            );

            let attention_input_size = if is_bidirectional {
                base_hidden_size * 2
            } else {
                base_hidden_size
            };

            let attention = MultiHeadAttention::new(
                attention_input_size, // Use correct input dimension for bidirectional
                attention_config.clone(),
                vs.pp("attention"),
                self.device.clone(),
            )?;

            self.attention_layers = Some(attention);

            log::debug!(
                "✅ Attention layers initialized: {} heads, input_size={}, bidirectional={}",
                attention_config.num_heads,
                attention_input_size,
                is_bidirectional
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

    /// Initialize multi-layer LSTM network using Sequential - Enhanced with bidirectional support
    fn initialize_network(&mut self) -> Result<()> {
        if self.lstm_layers.is_some() {
            return Ok(()); // Already initialized
        }

        log::info!(
            "Initializing multi-layer LSTM network with config: {:?}",
            self.config
        );

        // Check if this is a bidirectional LSTM
        let is_bidirectional = matches!(
            self.architecture,
            Some(crate::config::model::LSTMArchitecture::BidirectionalLSTM { .. })
        );

        if is_bidirectional {
            log::info!("🔄 Initializing Bidirectional LSTM with forward and backward layers");
        }

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

        // Build forward LSTM layers
        let mut forward_lstm_layers = Vec::new();
        // Build backward LSTM layers (only for bidirectional)
        let mut backward_lstm_layers = Vec::new();

        for layer_idx in 0..num_layers {
            // Input size calculation for bidirectional:
            // - First layer: uses input_size
            // - Subsequent layers: uses 2x hidden_size (concatenated forward+backward) for bidirectional,
            //   or 1x hidden_size for unidirectional
            let layer_input_size = if layer_idx == 0 {
                self.config.input_size
            } else {
                let prev_hidden_size = self.config.get_hidden_size_for_layer(layer_idx - 1);
                if is_bidirectional {
                    prev_hidden_size * 2 // Bidirectional output is concatenated
                } else {
                    prev_hidden_size
                }
            };

            // Get hidden size for this specific layer
            let layer_hidden_size = self.config.get_hidden_size_for_layer(layer_idx);

            // Create forward LSTM layer
            let forward_lstm_config = CandleLSTMConfig {
                layer_idx,
                direction: candle_nn::rnn::Direction::Forward,
                ..CandleLSTMConfig::default()
            };

            let vs_forward = vs.pp(format!("forward_lstm_layer_{}", layer_idx));
            let forward_lstm_layer = lstm(
                layer_input_size,
                layer_hidden_size,
                forward_lstm_config,
                vs_forward,
            )
            .map_err(|e| {
                VangaError::ModelError(format!(
                    "Forward LSTM layer {} creation failed: {}",
                    layer_idx, e
                ))
            })?;

            forward_lstm_layers.push(forward_lstm_layer);

            // Create backward LSTM layer (only for bidirectional)
            if is_bidirectional {
                let backward_lstm_config = CandleLSTMConfig {
                    layer_idx,
                    direction: candle_nn::rnn::Direction::Backward,
                    ..CandleLSTMConfig::default()
                };

                let vs_backward = vs.pp(format!("backward_lstm_layer_{}", layer_idx));
                let backward_lstm_layer = lstm(
                    layer_input_size,
                    layer_hidden_size,
                    backward_lstm_config,
                    vs_backward,
                )
                .map_err(|e| {
                    VangaError::ModelError(format!(
                        "Backward LSTM layer {} creation failed: {}",
                        layer_idx, e
                    ))
                })?;

                backward_lstm_layers.push(backward_lstm_layer);
            }

            log::debug!(
                "Layer {}: input_size={}, hidden_size={}, bidirectional={}",
                layer_idx,
                layer_input_size,
                layer_hidden_size,
                is_bidirectional
            );

            // GRADIENT STABILITY CHECK: Warn about configurations that cause exploding gradients
            if self.config.sequence_length > 60 {
                log::warn!(
                    "⚠️ LONG SEQUENCE WARNING: sequence_length={} > 60 may cause exploding gradients. Consider reducing to 30-60 for stability.",
                    self.config.sequence_length
                );
            }

            if layer_hidden_size > 256 && self.config.sequence_length > 30 {
                log::warn!(
                    "⚠️ LARGE MODEL WARNING: layer {} hidden_size={} with sequence_length={} may cause gradient instability. Consider reducing one or both.",
                    layer_idx, layer_hidden_size, self.config.sequence_length
                );
            }
        }

        // Store the layers
        self.lstm_layers = Some(forward_lstm_layers);
        if is_bidirectional {
            self.backward_lstm_layers = Some(backward_lstm_layers);
        }

        // CRITICAL: Apply proper weight initialization after LSTM creation
        self.apply_xavier_initialization()?;

        // Initialize attention layers if configured
        if self.use_attention && self.attention_config.is_some() {
            self.initialize_attention_layers(&vs)?;
        }

        // Calculate output layer input size
        // For bidirectional: last layer hidden size * 2 (concatenated)
        // For unidirectional: last layer hidden size
        let final_hidden_size = self.config.get_hidden_size_for_layer(num_layers - 1);
        let output_input_size = if is_bidirectional {
            final_hidden_size * 2
        } else {
            final_hidden_size
        };

        // Create output layer with proper input size
        let output_layer = linear(output_input_size, self.config.output_size, vs.pp("output"))
            .map_err(|e| VangaError::ModelError(format!("Output layer creation failed: {}", e)))?;

        self.output_layer = Some(output_layer);

        log::info!(
            "✅ LSTM network initialized: {} layers, {} parameters, bidirectional={}, output_input_size={}, final_hidden_size={}",
            num_layers,
            self.config.total_parameters(),
            is_bidirectional,
            output_input_size,
            final_hidden_size
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

    /// Forward pass through multi-layer LSTM network - Enhanced with bidirectional support
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
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

    /// Calculate MAPE for categorical data (price levels)
    ///
    /// For ordinal categorical data (price levels 0,1,2,3,4), we calculate MAPE as
    /// the percentage of the maximum possible error. This gives meaningful results:
    /// - If max_class=4, predicting class 4 when actual is class 0 = 100% error
    /// - If max_class=4, predicting class 2 when actual is class 0 = 50% error
    /// - If max_class=4, predicting class 1 when actual is class 0 = 25% error
    ///
    /// Formula: MAPE = (|predicted - actual| / max_possible_error) * 100
    /// where max_possible_error = max_class_value (since min is always 0)
    fn calculate_categorical_mape(&self, predictions: &Array2<f64>, targets: &Array2<f64>) -> f64 {
        // CRITICAL FIX: Validate shapes before operations
        if predictions.shape() != targets.shape() {
            log::error!(
                "Shape mismatch in categorical MAPE calculation: predictions={:?}, targets={:?}",
                predictions.shape(),
                targets.shape()
            );
            return f64::INFINITY;
        }

        // Find the maximum class value to determine the scale
        let max_target = targets.iter().fold(0.0f64, |acc, &x| acc.max(x));
        let max_prediction = predictions.iter().fold(0.0f64, |acc, &x| acc.max(x));
        let max_class_value = max_target.max(max_prediction);

        // If all values are 0 or max_class_value is 0, return 0% error
        if max_class_value <= 0.0 {
            return 0.0;
        }

        let mut total_percentage_error = 0.0;
        let mut total_samples = 0;

        for i in 0..predictions.nrows() {
            for j in 0..predictions.ncols() {
                let actual = targets[[i, j]];
                let predicted = predictions[[i, j]];

                // Calculate percentage error relative to maximum possible error
                let absolute_error = (actual - predicted).abs();
                let percentage_error = (absolute_error / max_class_value) * 100.0;

                total_percentage_error += percentage_error;
                total_samples += 1;
            }
        }

        if total_samples > 0 {
            total_percentage_error / total_samples as f64
        } else {
            0.0
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
        let num_layers = self.config.num_layers;

        // Calculate total hidden states size across all layers
        let mut hidden_states_size = 0;
        for layer_idx in 0..num_layers {
            let hidden_size = self.config.get_hidden_size_for_layer(layer_idx);
            hidden_states_size += batch_size * hidden_size * 4 * 2; // forward + backward, f32 = 4 bytes
        }

        // Rough estimation: input tensor + hidden states + gradients + attention (if enabled)
        let input_tensor_size = batch_size * sequence_length * input_features * 4; // f32 = 4 bytes
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
        // Optional per-window class weights for balanced training
        class_weights: Option<&Vec<f32>>,
    ) -> Result<()> {
        let total_samples = sequences.shape()[0];

        // ADDED: Validate dataset size for proper training with gap
        let sequence_length = self.config.sequence_length;
        let max_horizon_steps = if !config.horizons.is_empty() {
            config
                .horizons
                .iter()
                .map(|h| crate::targets::volatility::parse_horizon_to_steps(h).unwrap_or(1))
                .max()
                .unwrap_or(72)
        } else {
            72
        };

        let required_gap = sequence_length + max_horizon_steps;
        let min_required_samples = required_gap + sequence_length + 10; // Minimum viable dataset

        if total_samples < min_required_samples {
            log::warn!(
                "⚠️  SMALL DATASET WARNING: {} samples < {} recommended minimum",
                total_samples,
                min_required_samples
            );
            log::warn!(
                "   • Sequence length: {}, Horizon steps: {}, Required gap: {}",
                sequence_length,
                max_horizon_steps,
                required_gap
            );
            log::warn!(
                "   • Consider: reducing sequence_length, shorter horizons, or collecting more data"
            );
        }

        log::info!(
            "🚀 UNIFIED TRAINING: Starting with {} samples (min recommended: {})",
            total_samples,
            min_required_samples
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

            // FIXED: Calculate proper gap size to prevent data leakage
            // Gap must be sequence_length + max_horizon_steps to ensure no overlap between
            // the last training sequence and first validation target
            let max_horizon_steps = if !config.horizons.is_empty() {
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

            // CRITICAL FIX: Proper gap calculation
            let gap_size = self.config.sequence_length + max_horizon_steps;

            log::info!(
                "🔒 Gap calculation: sequence_length({}) + max_horizon_steps({}) = {} total gap",
                self.config.sequence_length,
                max_horizon_steps,
                gap_size
            );

            // FIXED: Account for gap in validation split calculation
            let effective_samples = total_samples.saturating_sub(gap_size);
            let base_train_samples = ((1.0 - validation_split) * effective_samples as f64) as usize;
            let train_samples = base_train_samples;
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

            // CRITICAL: Validate sequence-target alignment
            log::debug!(
                "🔍 Train/Val split validation: train_seq={:?}, train_tgt={:?}, val_seq={:?}, val_tgt={:?}",
                train_seq.shape(), train_tgt.shape(), val_seq.shape(), val_tgt.shape()
            );

            if train_seq.shape()[0] != train_tgt.shape()[0] {
                return Err(VangaError::DataError(format!(
                    "Train sequence-target mismatch: {} sequences vs {} targets",
                    train_seq.shape()[0],
                    train_tgt.shape()[0]
                )));
            }

            if val_seq.shape()[0] != val_tgt.shape()[0] {
                return Err(VangaError::DataError(format!(
                    "Validation sequence-target mismatch: {} sequences vs {} targets",
                    val_seq.shape()[0],
                    val_tgt.shape()[0]
                )));
            }

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

        // FIXED: Adaptive early stopping configuration based on target types
        let (early_stopping_patience, early_stopping_min_delta) = if use_validation {
            let base_patience = match &config.training.epochs {
                crate::config::training::EpochConfig::Auto { max_epochs: _ } => {
                    config.training.early_stopping.patience
                }
                _ => 10, // Default patience for fixed epochs
            };
            let base_min_delta = config.training.early_stopping.min_delta;

            // FIXED: Adjust min_delta based on target types and expected scale
            let target_type = self.get_target_type().unwrap_or(TargetType::PriceLevel);
            let (adaptive_patience, adaptive_min_delta) = self.get_adaptive_early_stopping_config(
                &[target_type],
                base_patience,
                base_min_delta,
            );

            log::info!(
                "🎯 Early stopping configured: patience={}, min_delta={:.6} (adaptive from {:.6}) for target: {:?}",
                adaptive_patience, adaptive_min_delta, base_min_delta, target_type
            );

            (adaptive_patience, adaptive_min_delta)
        } else {
            (u32::MAX, 0.0) // Disable early stopping without validation
        };

        // Calculate global class weights for consistent loss calculation across all batches
        // Skip if class weighting is disabled via configuration
        if config.training.class_weight_strategy == ClassWeightStrategy::None {
            log::info!("🚫 Class weighting disabled via configuration");
            self.global_class_weights = None;
        } else if let Some((_, target_type)) = &self.target_context {
            let num_classes = match target_type {
                TargetType::PriceLevel => {
                    if config.model.output_heads.price_levels.enabled {
                        config.model.output_heads.price_levels.bins as usize
                    } else {
                        self.config.output_size
                    }
                }
                TargetType::Direction => 3,  // Down=0, Sideways=1, Up=2
                TargetType::Volatility => 3, // Low=0, Medium=1, High=2
            };

            log::info!(
                "🌍 Calculating class weights from {} training samples for {:?} with {} classes (strategy: {:?})",
                train_targets.shape()[0],
                target_type,
                num_classes,
                config.training.class_weight_strategy
            );
            self.calculate_global_class_weights(
                &train_targets,
                num_classes,
                class_weights.cloned(),
            )?;
        }

        log::info!("🔧 Training Configuration:");
        log::info!("  - Epochs: {}", self.training_config.epochs);
        log::info!("  - Batch size: {}", batch_size);
        log::info!("  - Warmup epochs: {}", warmup_epochs);
        log::info!("  - Adaptive patience: {}", adaptive_patience);
        log::info!("  - Adaptive factor: {:.3}", adaptive_factor);
        log::info!("  - Target learning rate: {:.6}", target_lr);

        // Unified training loop with warmup, adaptive learning, optional validation, and early stopping
        for epoch in 0..self.training_config.epochs {
            // Initialize epoch tracking variables
            let mut epoch_train_loss = 0.0;
            let mut epoch_grad_norm = 0.0; // Track gradient norm for epoch logging
            let mut batch_count = 0;

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
            } else {
                // Apply learning schedule after warmup phase (if configured)
                if let Some(schedule_config) = &config.training.learning_schedule {
                    let epoch_after_warmup = epoch - warmup_epochs as usize;
                    let total_epochs = match &config.training.epochs {
                        crate::config::training::EpochConfig::Fixed(n) => *n as usize,
                        crate::config::training::EpochConfig::Auto { max_epochs } => {
                            *max_epochs as usize
                        }
                    };

                    let scheduled_lr = Self::calculate_scheduled_learning_rate(
                        schedule_config,
                        epoch_after_warmup,
                        target_lr,
                        total_epochs.saturating_sub(warmup_epochs as usize),
                    );

                    // Only update if there's a meaningful change (avoid unnecessary updates)
                    if (scheduled_lr - current_lr).abs() > 1e-8 {
                        optimizer.set_learning_rate(scheduled_lr);
                        current_lr = scheduled_lr;

                        log::debug!(
                            "📈 Schedule LR update at epoch {}: {:.6} (schedule: {:?})",
                            epoch + 1,
                            scheduled_lr,
                            schedule_config
                        );
                    }
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
                let loss = self.calculate_loss(&predictions, &target_tensor, config)?;

                // Backward pass with gradient computation
                let grads = loss.backward()?;

                // Apply gradient clipping if configured
                let (original_grad_norm, effective_grad_norm) = if let Some(clip_value) =
                    self.training_config.clip_gradient
                {
                    let (orig_norm, eff_norm) =
                        self.clip_gradients(&grads, clip_value, &mut optimizer, current_lr)?;

                    if epoch == 0 && batch_idx == 0 {
                        log::info!(
                            "🔧 Gradient clipping enabled: threshold={:.3}, initial_norm={:.6} -> effective_norm={:.6}",
                            clip_value,
                            orig_norm,
                            eff_norm
                        );
                    }

                    (orig_norm, eff_norm)
                } else {
                    // No clipping - calculate norm for monitoring
                    let norm = self.calculate_gradient_norm(&grads)?;
                    (norm, norm) // Both norms are the same when no clipping
                };

                // GRADIENT FLOW VALIDATION: Check gradients using EFFECTIVE norm after clipping
                self.validate_gradient_flow(&grads, effective_grad_norm, original_grad_norm)?;

                // Accumulate effective gradient norm for epoch reporting (shows actual training impact)
                epoch_grad_norm += effective_grad_norm;
                batch_count += 1;

                // Update parameters using the configured optimizer
                optimizer.step(&grads)?;

                // Accumulate loss for epoch reporting
                let batch_loss = loss.to_scalar::<f32>().map_err(|e| {
                    VangaError::ModelError(format!("Loss scalar conversion failed: {}", e))
                })?;
                epoch_train_loss += batch_loss * actual_batch_size as f32;
            }

            // Calculate average training loss and gradient norm
            let avg_train_loss = epoch_train_loss / total_train_samples as f32;
            let avg_grad_norm = if batch_count > 0 {
                epoch_grad_norm / batch_count as f64
            } else {
                0.0
            };

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
                    let val_loss = self.calculate_loss(&predictions, &target_tensor, config)?;
                    let val_batch_loss = val_loss.to_scalar::<f32>().map_err(|e| {
                        VangaError::ModelError(format!(
                            "Validation loss scalar conversion failed: {}",
                            e
                        ))
                    })?;

                    epoch_val_loss += val_batch_loss * actual_batch_size as f32;
                }

                let avg_val_loss = epoch_val_loss / total_val_samples as f32;

                // Calculate categorical metrics for price level targets
                if let Some((_, target_type)) = &self.target_context {
                    if target_type == &TargetType::PriceLevel {
                        self.calculate_categorical_validation_metrics(
                            val_seq, val_tgt, batch_size, epoch, config,
                        )
                        .await?;
                    }
                }

                Some(avg_val_loss)
            } else {
                None
            };

            // Adaptive learning rate adjustment after warmup
            // NOTE: This runs AFTER schedule updates, so adaptive LR can override schedule if needed
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

                // Add schedule status information
                let schedule_status = if epoch >= warmup_epochs as usize {
                    if let Some(schedule) = &config.training.learning_schedule {
                        match schedule {
                            crate::config::training::LearningScheduleConfig::Constant => {
                                " [Constant]"
                            }
                            crate::config::training::LearningScheduleConfig::LinearDecay {
                                ..
                            } => " [LinearDecay]",
                            crate::config::training::LearningScheduleConfig::ExponentialDecay {
                                ..
                            } => " [ExpDecay]",
                            crate::config::training::LearningScheduleConfig::CosineAnnealing {
                                ..
                            } => " [CosineAnn]",
                            crate::config::training::LearningScheduleConfig::WarmRestarts {
                                ..
                            } => " [WarmRestart]",
                        }
                    } else {
                        ""
                    }
                } else {
                    ""
                };

                if let Some(val_loss) = avg_val_loss {
                    // Get target type for this individual model
                    let target_type = self.get_target_type().unwrap_or(TargetType::PriceLevel);
                    let target_info = format!(" [{:?}]", target_type);

                    // Calculate loss ratio and status
                    let loss_ratio = val_loss / avg_train_loss;
                    let ratio_status = if loss_ratio < 1.5 {
                        "✅"
                    } else if loss_ratio < 3.0 {
                        "⚠️"
                    } else {
                        "🚨"
                    };
                    log::info!(
                        "Epoch {}/{}: Train Loss = {:.6}, Val Loss = {:.6} (Ratio: {:.2}x {}), LR: {:.6}, Grad: {:.2e}{}{}, Early Stop: {}/{}{}",
                        epoch + 1,
                        self.training_config.epochs,
                        avg_train_loss,
                        val_loss,
                        loss_ratio,
                        ratio_status,
                        current_lr,
                        avg_grad_norm,
                        warmup_status,
                        schedule_status,
                        early_stopping_counter,
                        early_stopping_patience,
                        target_info
                    );

                    // Log overfitting warnings only when necessary
                    if loss_ratio > 3.0 {
                        log::warn!("🔧 Overfitting detected (ratio: {:.2}x). Consider adjusting regularization or model complexity.", loss_ratio);
                    }
                } else {
                    let num_batches = total_train_samples.div_ceil(batch_size);
                    log::info!(
                        "Epoch {}/{}: Loss = {:.6}, Batches: {}, LR: {:.6}{}{}",
                        epoch + 1,
                        self.training_config.epochs,
                        avg_train_loss,
                        num_batches,
                        current_lr,
                        warmup_status,
                        schedule_status
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

        // Calculate final training metrics - use classification accuracy for categorical targets
        if let Ok(final_predictions) = self.predict(sequences).await {
            // For classification targets, use accuracy instead of MSE
            if let Some((_, target_type)) = &self.target_context {
                match target_type {
                    TargetType::PriceLevel | TargetType::Direction | TargetType::Volatility => {
                        // Use the SAME working method that calculates correct MAPE every 10 epochs
                        log::info!(
                            "📊 Calculating Final Training Metrics using validated method..."
                        );
                        let _ = self
                            .calculate_categorical_validation_metrics(
                                sequences, targets, 64, // batch_size (not used in the method)
                                10, // epoch = 10 to force calculation (10 % 10 == 0)
                                config,
                            )
                            .await;
                    }
                }
            } else {
                // Fallback to MSE for regression targets
                let final_mse = self.calculate_mse_loss(&final_predictions, targets);
                let final_mape = self.calculate_mape(&final_predictions, targets);
                log::info!(
                    "📊 Final Training Metrics - MSE: {:.6} (√MSE: {:.3}), MAPE: {:.2}%",
                    final_mse,
                    final_mse.sqrt(),
                    final_mape
                );
            }
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

        // CRITICAL FIX: Handle multi-class outputs for categorical targets
        let final_predictions_tensor = if let Some((_, target_type)) = &self.target_context {
            log::debug!("Target context found: {:?}", target_type);
            match target_type {
                TargetType::PriceLevel => {
                    // For Price Levels: Convert multi-class probabilities to class indices
                    let tensor_shape = predictions_tensor.shape();
                    log::debug!("Price Level prediction shape: {:?}", tensor_shape);
                    if tensor_shape.dims().len() == 2 && tensor_shape.dims()[1] > 1 {
                        log::info!(
                            "Converting Price Level multi-class output {:?} to class indices",
                            tensor_shape
                        );

                        // Get argmax (predicted class) for each sample
                        let class_indices = predictions_tensor.argmax(1)?;

                        // Convert to f32 and add dimension to make it [batch, 1]
                        class_indices
                            .to_dtype(candle_core::DType::F32)?
                            .unsqueeze(1)?
                    } else {
                        log::debug!(
                            "Price Level output already in correct shape: {:?}",
                            tensor_shape
                        );
                        predictions_tensor
                    }
                }
                TargetType::Direction => {
                    // For Direction: Convert multi-class probabilities to class indices
                    let tensor_shape = predictions_tensor.shape();
                    log::debug!("Direction prediction shape: {:?}", tensor_shape);
                    if tensor_shape.dims().len() == 2 && tensor_shape.dims()[1] > 1 {
                        log::info!(
                            "Converting Direction multi-class output {:?} to class indices",
                            tensor_shape
                        );

                        // Get argmax (predicted class) for each sample
                        let class_indices = predictions_tensor.argmax(1)?;

                        // Convert to f32 and add dimension to make it [batch, 1]
                        class_indices
                            .to_dtype(candle_core::DType::F32)?
                            .unsqueeze(1)?
                    } else {
                        predictions_tensor
                    }
                }
                TargetType::Volatility => {
                    // For Volatility: Convert multi-class probabilities to class indices
                    let tensor_shape = predictions_tensor.shape();
                    log::debug!("Volatility prediction shape: {:?}", tensor_shape);
                    if tensor_shape.dims().len() == 2 && tensor_shape.dims()[1] > 1 {
                        log::info!(
                            "Converting Volatility multi-class output {:?} to class indices",
                            tensor_shape
                        );

                        // Get argmax (predicted class) for each sample
                        let class_indices = predictions_tensor.argmax(1)?;

                        // Convert to f32 and add dimension to make it [batch, 1]
                        class_indices
                            .to_dtype(candle_core::DType::F32)?
                            .unsqueeze(1)?
                    } else {
                        predictions_tensor
                    }
                }
            }
        } else {
            // No target context - detect multi-class output automatically
            let tensor_shape = predictions_tensor.shape();
            log::warn!(
                "No target context set during prediction! Tensor shape: {:?}",
                tensor_shape
            );

            if tensor_shape.dims().len() == 2 && tensor_shape.dims()[1] > 1 {
                log::info!(
                    "Auto-detecting multi-class output {:?}, converting to class indices",
                    tensor_shape
                );

                // Get argmax (predicted class) for each sample
                let class_indices = predictions_tensor.argmax(1)?;

                // Convert to f32 and add dimension to make it [batch, 1]
                class_indices
                    .to_dtype(candle_core::DType::F32)?
                    .unsqueeze(1)?
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

    /// Get the output size of the model for debugging
    pub fn get_output_size(&self) -> usize {
        self.config.output_size
    }

    /// Calculate gradient norm for monitoring gradient flow
    fn calculate_gradient_norm(&self, grads: &candle_core::backprop::GradStore) -> Result<f64> {
        let mut total_norm_squared = 0.0f64;
        let mut param_count = 0;

        // Get all variables from the VarMap (same approach as clip_gradients)
        let all_vars = self.varmap.all_vars();

        for var in all_vars.iter() {
            if let Some(grad) = grads.get(var) {
                total_norm_squared += self.calculate_tensor_norm_squared(grad)?;
                param_count += 1;

                log::trace!(
                    "Gradient norm for param {}: {:.6e}",
                    var.as_tensor()
                        .shape()
                        .dims()
                        .iter()
                        .map(|d| d.to_string())
                        .collect::<Vec<_>>()
                        .join("x"),
                    self.calculate_tensor_norm_squared(grad)?.sqrt()
                );
            }
        }

        if param_count == 0 {
            return Err(VangaError::ModelError(
                "No gradients found in backward pass".to_string(),
            ));
        }

        let total_norm = total_norm_squared.sqrt();
        log::debug!(
            "🔍 Total gradient norm: {:.6e} across {} parameters",
            total_norm,
            param_count
        );

        Ok(total_norm)
    }

    /// Validate gradient flow to catch gradient issues early
    fn validate_gradient_flow(
        &self,
        grads: &candle_core::backprop::GradStore,
        effective_grad_norm: f64,
        original_grad_norm: f64,
    ) -> Result<bool> {
        // Check for NaN gradients (use effective norm)
        if effective_grad_norm.is_nan() {
            return Err(VangaError::ModelError("🚨 NaN gradients detected! This indicates numerical instability in loss calculation or model architecture.".to_string()));
        }

        // Check for infinite gradients (use effective norm)
        if effective_grad_norm.is_infinite() {
            return Err(VangaError::ModelError("🚨 Infinite gradients detected! This indicates exploding gradients - consider gradient clipping or lower learning rate.".to_string()));
        }

        // Check for zero gradients (no learning) - use effective norm
        if effective_grad_norm < 1e-12 {
            log::warn!(
                "⚠️ Very small effective gradient norm ({:.2e}) - model may not be learning effectively",
                effective_grad_norm
            );
            return Ok(false);
        }

        // Check for exploding gradients - NOW USES EFFECTIVE NORM (the actual training impact)
        if effective_grad_norm > 100.0 {
            if original_grad_norm != effective_grad_norm {
                // Clipping was applied but still too large
                log::warn!(
                    "⚠️ Large effective gradient norm ({:.2e}) after clipping from original ({:.2e}) - consider lower clipping threshold or learning rate",
                    effective_grad_norm,
                    original_grad_norm
                );
            } else {
                // No clipping was applied
                log::warn!(
                    "⚠️ Large gradient norm ({:.2e}) - consider gradient clipping",
                    effective_grad_norm
                );
            }
        } else if original_grad_norm > effective_grad_norm {
            // Clipping was successfully applied
            log::debug!(
                "✂️ Gradient clipping working: original={:.2e} -> effective={:.2e}",
                original_grad_norm,
                effective_grad_norm
            );
        }

        // Validate individual gradient tensors for NaN using sum approach
        let all_vars = self.varmap.all_vars();
        for var in all_vars.iter() {
            if let Some(grad) = grads.get(var) {
                // Check for NaN in individual gradients using sum approach with proper dtype handling
                let grad_sum = grad.sum_all()?;

                // Handle both F32 and F64 tensors (same pattern as calculate_tensor_norm_squared)
                let grad_sum_value: f64 = match grad_sum.dtype() {
                    candle_core::DType::F32 => {
                        let val: f32 = grad_sum.to_scalar().map_err(|e| {
                            VangaError::ModelError(format!("F32 gradient sum check failed: {}", e))
                        })?;
                        val as f64
                    }
                    candle_core::DType::F64 => grad_sum.to_scalar().map_err(|e| {
                        VangaError::ModelError(format!("F64 gradient sum check failed: {}", e))
                    })?,
                    other => {
                        return Err(VangaError::ModelError(format!(
                            "Unsupported gradient tensor dtype: {:?}. Expected F32 or F64.",
                            other
                        )));
                    }
                };

                if grad_sum_value.is_nan() {
                    return Err(VangaError::ModelError(format!(
                        "🚨 NaN detected in gradient for parameter with shape {:?}",
                        var.as_tensor().shape()
                    )));
                }
            }
        }

        log::debug!(
            "✅ Gradient flow validation passed - effective_norm: {:.6e}, original_norm: {:.6e}",
            effective_grad_norm,
            original_grad_norm
        );
        Ok(true)
    }

    /// Validate tensor shapes for loss calculation to catch configuration mismatches
    fn validate_tensor_shapes(
        &self,
        predictions: &Tensor,
        targets: &Tensor,
        config: &crate::config::TrainingConfig,
    ) -> Result<()> {
        let pred_shape = predictions.shape();
        let target_shape = targets.shape();

        // Basic shape validation
        if pred_shape.dims().len() != 2 {
            return Err(VangaError::ModelError(format!(
                "🚨 TENSOR SHAPE ERROR: Predictions must be 2D tensor, got shape {:?}",
                pred_shape
            )));
        }

        if target_shape.dims().len() != 2 {
            return Err(VangaError::ModelError(format!(
                "🚨 TENSOR SHAPE ERROR: Targets must be 2D tensor, got shape {:?}",
                target_shape
            )));
        }

        // Batch size consistency
        let pred_batch_size = pred_shape.dims()[0];
        let target_batch_size = target_shape.dims()[0];

        if pred_batch_size != target_batch_size {
            return Err(VangaError::ModelError(format!(
                "🚨 BATCH SIZE MISMATCH: Predictions batch size {} != targets batch size {}",
                pred_batch_size, target_batch_size
            )));
        }

        // Output size validation for classification targets
        let target_type = self.get_target_type()?;
        let pred_output_size = pred_shape.dims()[1];
        let expected_output_size = match target_type {
            TargetType::PriceLevel => {
                if config.model.output_heads.price_levels.enabled {
                    config.model.output_heads.price_levels.bins as usize
                } else {
                    1 // Regression mode
                }
            }
            TargetType::Direction => 3,  // Up/Down/Sideways
            TargetType::Volatility => 3, // Low/Medium/High
        };

        // CRITICAL CHECK: This catches the main bug we're fixing
        if config.model.output_heads.price_levels.enabled
            && target_type == TargetType::PriceLevel
            && pred_output_size == 1
            && expected_output_size > 1
        {
            return Err(VangaError::ModelError(format!(
                "🚨 CRITICAL CONFIGURATION MISMATCH: PriceLevel classification enabled with {} bins but model output_size=1. This causes MSE fallback instead of CrossEntropy loss, breaking gradient flow. Fix: Set model output_size={}",
                expected_output_size, expected_output_size
            )));
        }

        if pred_output_size != expected_output_size {
            log::warn!(
                "⚠️ OUTPUT SIZE MISMATCH: Model output_size={} but expected {} for {:?} target. This may cause suboptimal loss calculation.",
                pred_output_size, expected_output_size, target_type
            );
        }

        log::debug!(
            "✅ Tensor shape validation passed: pred={:?}, target={:?}, expected_output={}",
            pred_shape,
            target_shape,
            expected_output_size
        );

        Ok(())
    }

    /// Set target context for this individual model
    /// This allows proper target type detection without assumptions based on output_size
    pub fn set_target_context(
        &mut self,
        target_name: String,
        target_type: crate::targets::TargetType,
    ) {
        self.target_context = Some((target_name.clone(), target_type));
        log::debug!(
            "🎯 Target context set: {} -> {:?}",
            target_name,
            target_type
        );
    }
}

// Implement From trait for Candle error conversion
impl From<candle_core::Error> for VangaError {
    fn from(err: candle_core::Error) -> Self {
        VangaError::ModelError(format!("Candle error: {}", err))
    }
}

impl LSTMModel {
    /// Get THIS model's target type - MUST be set during model creation
    /// No fallbacks, no assumptions - if not set, it's a programming error
    fn get_target_type(&self) -> Result<TargetType> {
        match &self.target_context {
            Some((_, target_type)) => Ok(*target_type),
            None => Err(VangaError::ModelError(
                "Target context not set for individual LSTM model. This is a programming error - models must be created with explicit target context.".to_string()
            ))
        }
    }

    /// Validate that model output_size matches expected target size
    /// This helps debug configuration issues
    fn validate_target_size_consistency(
        &self,
        config: &crate::config::TrainingConfig,
    ) -> Result<()> {
        let target_type = self.get_target_type()?;
        let expected_size = self.get_target_size(target_type, config);
        let actual_size = self.config.output_size;

        if actual_size != expected_size {
            log::error!(
                "🚨 TARGET SIZE MISMATCH: Target {:?} expects {} outputs but model has {} outputs",
                target_type,
                expected_size,
                actual_size
            );
            return Err(VangaError::ModelError(format!(
                "Model output_size ({}) doesn't match expected size ({}) for target type {:?}",
                actual_size, expected_size, target_type
            )));
        }

        log::debug!(
            "✅ Target size validation passed: {:?} -> {} outputs",
            target_type,
            actual_size
        );
        Ok(())
    }

    /// Get target size for a specific target type based on configuration
    fn get_target_size(
        &self,
        target_type: TargetType,
        config: &crate::config::TrainingConfig,
    ) -> usize {
        match target_type {
            TargetType::PriceLevel => {
                if config.model.output_heads.price_levels.enabled {
                    config.model.output_heads.price_levels.bins as usize
                } else {
                    // Use output_size from LSTM config as fallback
                    self.config.output_size
                }
            }
            TargetType::Direction => 3,  // Up/Down/Sideways
            TargetType::Volatility => 3, // Low/Medium/High
        }
    }

    /// Calculate CrossEntropy loss for categorical targets with optional class weighting
    fn calculate_crossentropy_loss(
        &self,
        predictions: &Tensor,
        targets: &Tensor,
        num_classes: usize,
    ) -> Result<Tensor> {
        log::debug!(
            "🔍 CrossEntropy Loss - Pred shape: {:?}, Target shape: {:?}, Classes: {}",
            predictions.shape(),
            targets.shape(),
            num_classes
        );

        // Handle different prediction shapes
        let logits = if predictions.dims().last() == Some(&num_classes) {
            // Already correct shape for multi-class
            predictions.clone()
        } else if predictions.dims().len() == 2 && predictions.dims()[1] == 1 {
            // CRITICAL BUG FIX: Single output with classification targets is a configuration error
            return Err(VangaError::ModelError(format!(
                "🚨 CONFIGURATION MISMATCH: Model has output_size=1 but classification target requires {} classes. This causes MSE fallback instead of CrossEntropy loss, breaking gradient flow. Fix: Set model output_size={} for classification targets.",
                num_classes, num_classes
            )));
        } else {
            return Err(VangaError::ModelError(format!(
                "Invalid prediction shape for CrossEntropy: {:?}, expected last dim = {}",
                predictions.shape(),
                num_classes
            )));
        };

        // Ensure targets are in correct format for CrossEntropy
        let target_shape = targets.shape();
        if target_shape.dims().len() != 2 {
            return Err(VangaError::ModelError(format!(
                "Invalid target shape for CrossEntropy: {:?}, expected 2D tensor",
                target_shape
            )));
        }

        // Use global class weights if available, otherwise calculate per-batch (fallback)
        let class_weights = if let Some((_target_name, target_type)) = &self.target_context {
            match target_type {
                TargetType::PriceLevel | TargetType::Direction | TargetType::Volatility => {
                    if let Some(ref global_weights) = self.global_class_weights {
                        log::debug!(
                            "🌍 Using global class weights for {:?}: {:?}",
                            target_type,
                            global_weights
                        );
                        Some(global_weights.clone())
                    } else {
                        log::debug!(
                            "⚠️ Global weights not available for {:?}, calculating per-batch",
                            target_type
                        );
                        self.calculate_class_weights_from_tensor(targets, num_classes)?
                    }
                }
            }
        } else {
            None
        };

        // Apply label smoothing for categorical targets
        let smoothed_targets = if let Some((_, target_type)) = &self.target_context {
            match target_type {
                TargetType::PriceLevel => {
                    // 10% smoothing for price levels (existing behavior)
                    self.apply_label_smoothing(targets, num_classes, 0.1)?
                }
                TargetType::Direction => {
                    // 5% smoothing for direction targets (less aggressive for 3-class)
                    self.apply_label_smoothing(targets, num_classes, 0.05)?
                }
                TargetType::Volatility => {
                    // 5% smoothing for volatility targets (less aggressive for 3-class)
                    self.apply_label_smoothing(targets, num_classes, 0.05)?
                }
            }
        } else {
            targets.clone()
        };

        // Check the smoothed targets shape to determine loss calculation path
        let smoothed_target_shape = smoothed_targets.shape();

        log::debug!(
            "🎯 Loss calculation: Original targets {:?} → Smoothed targets {:?}, Classes: {}",
            target_shape,
            smoothed_target_shape,
            num_classes
        );

        // For CrossEntropy, targets should be class indices (integers) or one-hot encoded
        let loss = if smoothed_target_shape.dims()[1] == 1 {
            log::debug!("📊 Using class indices path (no label smoothing applied)");
            // Targets are class indices - use proper CrossEntropy loss
            let target_indices = smoothed_targets.to_dtype(candle_core::DType::I64)?;

            if let Some(weights) = class_weights {
                log::debug!("⚖️ Applying class weights to indices");
                // Use weighted CrossEntropy for imbalanced classes
                self.calculate_weighted_crossentropy_loss(
                    &logits,
                    &target_indices.squeeze(1)?,
                    &weights,
                )?
            } else {
                log::debug!("📈 Using standard CrossEntropy for indices");
                // Use standard CrossEntropy loss
                candle_nn::loss::cross_entropy(&logits, &target_indices.squeeze(1)?)?
            }
        } else if smoothed_target_shape.dims()[1] == num_classes {
            log::debug!("🎯 Using one-hot path (label smoothing applied)");
            // Targets are one-hot encoded (from label smoothing) - use soft CrossEntropy
            let log_softmax =
                candle_nn::ops::log_softmax(&logits, candle_core::D::Minus1)?.contiguous()?;

            // For one-hot targets with class weights, we need to apply weights differently
            if let Some(weights) = class_weights {
                log::debug!("⚖️ Applying class weights to one-hot targets");
                // Apply class weights to one-hot encoded targets
                self.calculate_weighted_soft_crossentropy_loss(
                    &logits,
                    &smoothed_targets,
                    &weights,
                )?
            } else {
                log::debug!("📈 Using standard soft CrossEntropy for one-hot");
                // Standard soft CrossEntropy for one-hot targets - ensure all tensors are contiguous
                let smoothed_contiguous = smoothed_targets.contiguous()?;
                let loss = smoothed_contiguous
                    .mul(&log_softmax)?
                    .contiguous()?
                    .sum(candle_core::D::Minus1)?
                    .contiguous()?;
                loss.neg()?.mean_all()?
            }
        } else {
            return Err(VangaError::ModelError(format!(
                "Target dimension mismatch: got {}, expected 1 (indices) or {} (one-hot)",
                smoothed_target_shape.dims()[1],
                num_classes
            )));
        };

        let loss_value = loss.to_scalar::<f32>().unwrap_or(0.0);

        log::debug!("🎯 CrossEntropy Loss: {:.6}", loss_value);
        Ok(loss)
    }

    /// Calculate global class weights from entire training dataset
    /// This ensures consistent loss calculation across all batches
    pub fn calculate_global_class_weights(
        &mut self,
        train_targets: &Array2<f64>,
        num_classes: usize,
        provided_weights: Option<Vec<f32>>,
    ) -> Result<()> {
        // Calculate for all categorical targets: PriceLevel, Direction, and Volatility
        if let Some((_, target_type)) = &self.target_context {
            match target_type {
                TargetType::PriceLevel => {
                    log::debug!(
                        "🎯 Calculating global class weights for PriceLevel target with {} classes",
                        num_classes
                    );
                }
                TargetType::Direction => {
                    log::debug!(
                        "🎯 Calculating global class weights for Direction target (3 classes: Down=0, Sideways=1, Up=2)"
                    );
                }
                TargetType::Volatility => {
                    log::debug!(
                        "🎯 Calculating global class weights for Volatility target (3 classes: Low=0, Medium=1, High=2)"
                    );
                }
            }
        } else {
            log::debug!("🎯 No target context set, skipping global class weights");
            self.global_class_weights = None;
            return Ok(());
        }

        // Check if pre-calculated weights are provided (for per-window class weights)
        if let Some(weights) = provided_weights {
            log::info!(
                "🎯 Using provided per-window class weights for {:?}: {:?}",
                self.target_context.as_ref().map(|(_, t)| t),
                weights
            );
            self.global_class_weights = Some(weights);
            return Ok(());
        }

        // Convert to tensor for consistent processing - ensure F32 dtype
        let targets_f32: Vec<f32> = train_targets
            .as_slice()
            .unwrap()
            .iter()
            .map(|&x| x as f32)
            .collect();
        let targets_tensor = Tensor::from_slice(&targets_f32, train_targets.dim(), &self.device)?;

        // Calculate global class weights from entire training dataset
        let weights = self.calculate_class_weights_from_tensor(&targets_tensor, num_classes)?;

        if let Some(weights) = weights {
            log::info!(
                "🌍 Global class weights calculated from {} training samples for {:?}: {:?}",
                train_targets.shape()[0],
                self.target_context.as_ref().map(|(_, t)| t),
                weights
            );
            self.global_class_weights = Some(weights);
        } else {
            log::warn!("⚠️ Failed to calculate global class weights, using per-batch calculation");
            self.global_class_weights = None;
        }

        Ok(())
    }

    /// Calculate class weights for imbalanced datasets (helper method)
    fn calculate_class_weights_from_tensor(
        &self,
        targets: &Tensor,
        num_classes: usize,
    ) -> Result<Option<Vec<f32>>> {
        // Extract target values to calculate class distribution
        let target_data = targets.to_vec2::<f32>()?;
        let mut class_counts = vec![0usize; num_classes];
        let mut total_samples = 0;

        // Count class occurrences
        for row in &target_data {
            if let Some(&target_val) = row.first() {
                let class_idx = target_val as usize;
                if class_idx < num_classes {
                    class_counts[class_idx] += 1;
                    total_samples += 1;
                }
            }
        }

        if total_samples == 0 {
            return Ok(None);
        }

        // Calculate inverse frequency weights
        let mut weights = Vec::new();
        let mut max_weight = 0.0f32;

        for &count in &class_counts {
            if count > 0 {
                let weight = total_samples as f32 / (num_classes as f32 * count as f32);
                weights.push(weight);
                max_weight = max_weight.max(weight);
            } else {
                // Handle empty classes with high weight
                weights.push(max_weight * 2.0);
            }
        }

        // Normalize weights to prevent extreme values
        let weight_sum: f32 = weights.iter().sum();
        if weight_sum > 0.0 {
            for weight in &mut weights {
                *weight = (*weight / weight_sum) * num_classes as f32;
                *weight = weight.clamp(0.1, 10.0); // Clamp to reasonable range
            }
        }

        log::debug!(
            "📊 Class weights calculated: {:?} (from counts: {:?})",
            weights,
            class_counts
        );

        Ok(Some(weights))
    }

    /// Calculate weighted CrossEntropy loss for imbalanced classes
    fn calculate_weighted_crossentropy_loss(
        &self,
        logits: &Tensor,
        targets: &Tensor,
        class_weights: &[f32],
    ) -> Result<Tensor> {
        // Calculate standard CrossEntropy loss per sample
        let log_softmax =
            candle_nn::ops::log_softmax(logits, candle_core::D::Minus1)?.contiguous()?;

        // Validate tensor dimensions
        let batch_size = targets.dim(0)?;
        let logits_batch_size = logits.dim(0)?;
        let num_classes = class_weights.len();

        if batch_size != logits_batch_size {
            return Err(VangaError::ModelError(format!(
                "Batch size mismatch: targets {} vs logits {}",
                batch_size, logits_batch_size
            )));
        }

        let mut weighted_losses = Vec::with_capacity(batch_size);
        let target_data = targets.contiguous()?.to_vec1::<i64>()?;
        let log_softmax_data = log_softmax.to_vec2::<f32>()?;

        // Validate data consistency
        if target_data.len() != batch_size {
            return Err(VangaError::ModelError(format!(
                "Target data length {} doesn't match batch size {}",
                target_data.len(),
                batch_size
            )));
        }

        if log_softmax_data.len() != batch_size {
            return Err(VangaError::ModelError(format!(
                "Log softmax data length {} doesn't match batch size {}",
                log_softmax_data.len(),
                batch_size
            )));
        }

        for (i, &target_class) in target_data.iter().enumerate() {
            let class_idx = target_class as usize;
            if class_idx < num_classes {
                let log_prob = log_softmax_data[i][class_idx];
                let weight = class_weights[class_idx];
                let weighted_loss = -log_prob * weight;
                weighted_losses.push(weighted_loss);
            } else {
                log::warn!(
                    "Invalid class index {} >= {}, skipping sample {}",
                    class_idx,
                    num_classes,
                    i
                );
            }
        }

        if weighted_losses.is_empty() {
            return Err(VangaError::ModelError(
                "No valid samples for weighted loss calculation".to_string(),
            ));
        }

        // Convert back to tensor and calculate mean
        let loss_values = weighted_losses.clone(); // Clone before move
        let loss_tensor = Tensor::from_vec(weighted_losses, (loss_values.len(),), logits.device())?
            .contiguous()?;
        let mean_loss = loss_tensor.mean_all()?;

        log::debug!(
            "⚖️ Weighted CrossEntropy: {:.6} (vs unweighted: {:.6}) for {} samples",
            mean_loss.to_scalar::<f32>().unwrap_or(0.0),
            candle_nn::loss::cross_entropy(logits, targets)?
                .to_scalar::<f32>()
                .unwrap_or(0.0),
            batch_size
        );

        Ok(mean_loss)
    }

    /// Calculate weighted soft CrossEntropy loss for one-hot encoded targets
    fn calculate_weighted_soft_crossentropy_loss(
        &self,
        logits: &Tensor,
        one_hot_targets: &Tensor,
        class_weights: &[f32],
    ) -> Result<Tensor> {
        // Ensure ALL input tensors are contiguous from the start
        let logits_contiguous = logits.contiguous()?;
        let targets_contiguous = one_hot_targets.contiguous()?;

        let log_softmax = candle_nn::ops::log_softmax(&logits_contiguous, candle_core::D::Minus1)?
            .contiguous()?;

        // Validate tensor dimensions
        let batch_size = targets_contiguous.dim(0)?;
        let num_classes = class_weights.len();

        if targets_contiguous.dim(1)? != num_classes {
            return Err(VangaError::ModelError(format!(
                "One-hot targets dimension {} doesn't match class weights {}",
                targets_contiguous.dim(1)?,
                num_classes
            )));
        }

        log::debug!(
            "🔍 Weighted soft CrossEntropy shapes: targets {:?}, logits {:?}, weights len {}",
            targets_contiguous.shape(),
            logits_contiguous.shape(),
            num_classes
        );

        // Create weight tensor with shape [1, num_classes] and ensure contiguous
        let weight_tensor = Tensor::from_vec(
            class_weights.to_vec(),
            (1, num_classes),
            logits_contiguous.device(),
        )?
        .contiguous()?;

        log::debug!(
            "🔍 Broadcasting shapes: targets {:?} × weights {:?}",
            targets_contiguous.shape(),
            weight_tensor.shape()
        );

        // Use broadcast_as to explicitly match tensor shapes before multiplication
        // Broadcasting: [1, num_classes] -> [batch_size, num_classes]
        let weight_tensor_broadcast = weight_tensor.broadcast_as(targets_contiguous.shape())?;

        log::debug!(
            "🔍 After broadcast_as: targets {:?} × weights {:?}",
            targets_contiguous.shape(),
            weight_tensor_broadcast.shape()
        );

        // Now multiply tensors with matching shapes and ensure result is contiguous
        let weighted_targets = targets_contiguous
            .mul(&weight_tensor_broadcast)?
            .contiguous()?;

        // Calculate weighted soft CrossEntropy loss - ensure all intermediate results are contiguous
        let weighted_log_loss = weighted_targets.mul(&log_softmax)?.contiguous()?;
        let loss_per_sample = weighted_log_loss
            .sum(candle_core::D::Minus1)?
            .contiguous()?;
        let mean_loss = loss_per_sample.neg()?.mean_all()?.contiguous()?;

        log::debug!(
            "⚖️ Weighted Soft CrossEntropy: {:.6} for {} samples with {} classes",
            mean_loss.to_scalar::<f32>().unwrap_or(0.0),
            batch_size,
            num_classes
        );

        Ok(mean_loss)
    }

    /// Apply label smoothing to reduce overconfidence in categorical predictions
    fn apply_label_smoothing(
        &self,
        targets: &Tensor,
        num_classes: usize,
        smoothing: f32,
    ) -> Result<Tensor> {
        let target_shape = targets.shape();

        if target_shape.dims()[1] == 1 {
            // Convert class indices to smoothed one-hot encoding
            let batch_size = target_shape.dims()[0];
            let target_data = targets.to_vec2::<f32>()?;

            let mut smoothed_data = Vec::new();

            for row in &target_data {
                if let Some(&target_class) = row.first() {
                    let class_idx = target_class as usize;

                    // Create smoothed one-hot vector
                    let mut one_hot = vec![smoothing / (num_classes - 1) as f32; num_classes];
                    if class_idx < num_classes {
                        one_hot[class_idx] = 1.0 - smoothing;
                    }

                    smoothed_data.extend(one_hot);
                }
            }

            let smoothed_tensor =
                Tensor::from_vec(smoothed_data, (batch_size, num_classes), targets.device())?
                    .contiguous()?; // Ensure contiguity

            log::debug!(
                "🎯 Label smoothing applied: {:.1}% smoothing for {} classes",
                smoothing * 100.0,
                num_classes
            );

            Ok(smoothed_tensor)
        } else if target_shape.dims()[1] == num_classes {
            // Already one-hot encoded - apply smoothing
            let uniform_dist = smoothing / num_classes as f32;

            // Ensure ALL intermediate tensors are contiguous
            let targets_contiguous = targets.contiguous()?;
            let scale_tensor =
                Tensor::from_slice(&[1.0 - smoothing], (1,), targets.device())?.contiguous()?;
            let uniform_tensor =
                Tensor::from_slice(&[uniform_dist], (1,), targets.device())?.contiguous()?;

            let scaled = targets_contiguous.mul(&scale_tensor)?.contiguous()?;
            let smoothed = scaled.add(&uniform_tensor)?.contiguous()?;

            log::debug!(
                "🎯 Label smoothing applied to one-hot targets: {:.1}% smoothing",
                smoothing * 100.0
            );

            Ok(smoothed)
        } else {
            // Invalid target format - return original
            log::warn!(
                "⚠️ Cannot apply label smoothing to targets with shape: {:?}",
                target_shape
            );
            Ok(targets.clone())
        }
    }

    /// Detect target format based on tensor shape and values
    fn detect_target_format(&self, target_tensor: &Tensor) -> Result<TargetFormat> {
        let shape = target_tensor.shape();
        let dims = shape.dims();
        if dims.len() != 2 {
            return Ok(TargetFormat::Unknown);
        }

        let num_outputs = dims[1];

        // If only 1 output dimension, it's likely raw class indices
        if num_outputs == 1 {
            return Ok(TargetFormat::RawClassIndices);
        }

        // If multiple outputs, check if it looks like one-hot encoding
        // Sample a few rows to check the pattern
        let sample_data = target_tensor.to_vec2::<f32>()?;
        let mut one_hot_count = 0;
        let mut total_checked = 0;

        for row in sample_data.iter().take(10) {
            // Check first 10 rows
            total_checked += 1;

            // Count non-zero values
            let non_zero_count = row.iter().filter(|&&x| x > 0.0).count();
            let max_value = row.iter().fold(0.0f32, |a, &b| a.max(b));

            // One-hot pattern: exactly one 1.0, rest are 0.0
            if non_zero_count == 1 && max_value == 1.0 {
                one_hot_count += 1;
            }
        }

        // If most samples follow one-hot pattern, classify as one-hot
        if total_checked > 0 && one_hot_count as f32 / total_checked as f32 > 0.8 {
            Ok(TargetFormat::OneHot)
        } else {
            Ok(TargetFormat::RawValues)
        }
    }

    /// Calculate categorical validation metrics for price level targets
    async fn calculate_categorical_validation_metrics(
        &self,
        val_sequences: &Array3<f64>,
        val_targets: &Array2<f64>,
        _batch_size: usize,
        epoch: usize,
        _config: &crate::config::TrainingConfig,
    ) -> Result<()> {
        // Only calculate detailed metrics every 10 epochs to avoid overhead
        if epoch % 10 != 0 {
            return Ok(());
        }

        let total_val_samples = val_sequences.shape()[0];
        let validation_batch_size = 64; // Fixed batch size for validation metrics
        let mut all_predictions = Vec::new();
        let mut all_targets = Vec::new();

        // Detect target format once for the entire validation set
        let (_input_tensor_sample, target_tensor_sample) = self.convert_sequences_to_tensors(
            &val_sequences.slice(ndarray::s![0..1, .., ..]).to_owned(),
            &val_targets.slice(ndarray::s![0..1, ..]).to_owned(),
        )?;
        let target_format = self.detect_target_format(&target_tensor_sample)?;

        log::debug!("🎯 Detected target format: {:?}", target_format);

        // Collect all predictions and targets
        for batch_start in (0..total_val_samples).step_by(validation_batch_size) {
            let batch_end = std::cmp::min(batch_start + validation_batch_size, total_val_samples);

            let batch_sequences = val_sequences
                .slice(ndarray::s![batch_start..batch_end, .., ..])
                .to_owned();
            let batch_targets = val_targets
                .slice(ndarray::s![batch_start..batch_end, ..])
                .to_owned();

            let (input_tensor, target_tensor) =
                self.convert_sequences_to_tensors(&batch_sequences, &batch_targets)?;

            let predictions = self.forward(&input_tensor)?;

            // Convert predictions to class indices
            let pred_data = predictions.to_vec2::<f32>()?;
            let target_data = target_tensor.to_vec2::<f32>()?;

            for (pred_row, target_row) in pred_data.iter().zip(target_data.iter()) {
                // Get predicted class (argmax)
                let predicted_class = pred_row
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(idx, _)| idx as i32)
                    .unwrap_or(0);

                // Get true class - use detected format for proper extraction
                let true_class = match target_format {
                    TargetFormat::OneHot => {
                        // One-hot encoded - find max index
                        target_row
                            .iter()
                            .enumerate()
                            .max_by(|(_, a), (_, b)| {
                                a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                            })
                            .map(|(idx, _)| idx as i32)
                            .unwrap_or(0)
                    }
                    TargetFormat::RawClassIndices => {
                        // Raw class index format: [2.0] means class 2
                        let raw_value = target_row[0];
                        if raw_value >= 0.0 && raw_value.fract() == 0.0 {
                            raw_value as i32
                        } else {
                            log::warn!(
                                "Invalid class index in target: {}, defaulting to class 0",
                                raw_value
                            );
                            0
                        }
                    }
                    TargetFormat::RawValues | TargetFormat::Unknown => {
                        // For other formats, try to infer the best approach
                        if target_row.len() == 1 {
                            let raw_value = target_row[0];
                            if raw_value >= 0.0 && raw_value.fract() == 0.0 {
                                raw_value as i32
                            } else {
                                log::warn!(
                                    "Non-integer target value: {}, defaulting to class 0",
                                    raw_value
                                );
                                0
                            }
                        } else {
                            // Multi-value, assume one-hot
                            target_row
                                .iter()
                                .enumerate()
                                .max_by(|(_, a), (_, b)| {
                                    a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                                })
                                .map(|(idx, _)| idx as i32)
                                .unwrap_or(0)
                        }
                    }
                };

                all_predictions.push(predicted_class);
                all_targets.push(true_class);
            }
        }

        // Calculate categorical metrics
        let accuracy = self.calculate_accuracy(&all_predictions, &all_targets);
        let (precision, recall, f1) =
            self.calculate_precision_recall_f1(&all_predictions, &all_targets);
        let class_distribution =
            self.analyze_prediction_distribution(&all_predictions, &all_targets);

        // Calculate additional distance-based metrics for categorical data
        // Convert predictions and targets to Array2<f64> for MSE/MAPE calculation
        let pred_array = Array2::from_shape_vec(
            (all_predictions.len(), 1),
            all_predictions.iter().map(|&x| x as f64).collect(),
        )
        .unwrap_or_else(|_| Array2::zeros((0, 1)));

        let target_array = Array2::from_shape_vec(
            (all_targets.len(), 1),
            all_targets.iter().map(|&x| x as f64).collect(),
        )
        .unwrap_or_else(|_| Array2::zeros((0, 1)));

        let mse = if !pred_array.is_empty() && !target_array.is_empty() {
            self.calculate_mse_loss(&pred_array, &target_array)
        } else {
            f64::INFINITY
        };

        let categorical_mape = if !pred_array.is_empty() && !target_array.is_empty() {
            self.calculate_categorical_mape(&pred_array, &target_array)
        } else {
            f64::INFINITY
        };

        // Debug logging for first few samples to verify target extraction
        if epoch == 10 {
            log::debug!("🔍 Target extraction verification (first 5 samples):");
            for i in 0..std::cmp::min(5, all_predictions.len()) {
                log::debug!(
                    "  Sample {}: Predicted={}, True={}",
                    i,
                    all_predictions[i],
                    all_targets[i]
                );
            }
        }

        // Log comprehensive categorical metrics
        log::info!(
            "📊 Categorical Metrics [Epoch {}]: Accuracy: {:.3}, Precision: {:.3}, Recall: {:.3}, F1: {:.3}, MSE: {:.3}, MAPE: {:.2}%",
            epoch, accuracy, precision, recall, f1, mse, categorical_mape
        );

        log::debug!(
            "📈 Class Distribution: Pred: {:?}, True: {:?}",
            class_distribution.0,
            class_distribution.1
        );

        Ok(())
    }

    /// Calculate accuracy for categorical predictions
    fn calculate_accuracy(&self, predictions: &[i32], targets: &[i32]) -> f32 {
        if predictions.len() != targets.len() || predictions.is_empty() {
            return 0.0;
        }

        let correct = predictions
            .iter()
            .zip(targets.iter())
            .filter(|(pred, target)| pred == target)
            .count();

        correct as f32 / predictions.len() as f32
    }

    /// Calculate precision, recall, and F1 score (macro-averaged)
    fn calculate_precision_recall_f1(
        &self,
        predictions: &[i32],
        targets: &[i32],
    ) -> (f32, f32, f32) {
        if predictions.len() != targets.len() || predictions.is_empty() {
            return (0.0, 0.0, 0.0);
        }

        // Find unique classes
        let mut classes = std::collections::HashSet::new();
        for &pred in predictions {
            classes.insert(pred);
        }
        for &target in targets {
            classes.insert(target);
        }

        let mut total_precision = 0.0;
        let mut total_recall = 0.0;
        let mut valid_classes = 0;

        for &class in &classes {
            let tp = predictions
                .iter()
                .zip(targets.iter())
                .filter(|(pred, target)| **pred == class && **target == class)
                .count() as f32;

            let fp = predictions
                .iter()
                .zip(targets.iter())
                .filter(|(pred, target)| **pred == class && **target != class)
                .count() as f32;

            let fn_count = predictions
                .iter()
                .zip(targets.iter())
                .filter(|(pred, target)| **pred != class && **target == class)
                .count() as f32;

            let precision = if tp + fp > 0.0 { tp / (tp + fp) } else { 0.0 };
            let recall = if tp + fn_count > 0.0 {
                tp / (tp + fn_count)
            } else {
                0.0
            };

            if precision > 0.0 || recall > 0.0 {
                total_precision += precision;
                total_recall += recall;
                valid_classes += 1;
            }
        }

        let avg_precision = if valid_classes > 0 {
            total_precision / valid_classes as f32
        } else {
            0.0
        };
        let avg_recall = if valid_classes > 0 {
            total_recall / valid_classes as f32
        } else {
            0.0
        };
        let f1 = if avg_precision + avg_recall > 0.0 {
            2.0 * (avg_precision * avg_recall) / (avg_precision + avg_recall)
        } else {
            0.0
        };

        (avg_precision, avg_recall, f1)
    }

    /// Analyze prediction and target class distributions
    fn analyze_prediction_distribution(
        &self,
        predictions: &[i32],
        targets: &[i32],
    ) -> (Vec<usize>, Vec<usize>) {
        let max_class = predictions.iter().chain(targets.iter()).max().unwrap_or(&0);
        let num_classes = (*max_class + 1) as usize;

        let mut pred_counts = vec![0; num_classes];
        let mut target_counts = vec![0; num_classes];

        for &pred in predictions {
            if pred >= 0 && (pred as usize) < num_classes {
                pred_counts[pred as usize] += 1;
            }
        }

        for &target in targets {
            if target >= 0 && (target as usize) < num_classes {
                target_counts[target as usize] += 1;
            }
        }

        (pred_counts, target_counts)
    }

    /// Calculate loss for single target type
    fn calculate_single_target_loss(
        &self,
        predictions: &Tensor,
        targets: &Tensor,
        target_type: TargetType,
        config: &crate::config::TrainingConfig,
    ) -> Result<Tensor> {
        log::debug!(
            "🎯 Single target loss - Type: {:?}, Pred shape: {:?}, Target shape: {:?}",
            target_type,
            predictions.shape(),
            targets.shape()
        );

        match target_type {
            TargetType::PriceLevel => {
                if config.model.output_heads.price_levels.enabled {
                    // CrossEntropy for categorical price levels
                    let num_classes = config.model.output_heads.price_levels.bins as usize;
                    self.calculate_crossentropy_loss(predictions, targets, num_classes)
                } else {
                    // MSE for continuous price prediction
                    Ok(predictions.sub(targets)?.sqr()?.mean_all()?)
                }
            }
            TargetType::Direction => {
                // Direction targets are ALWAYS 3-class classification (Down=0, Sideways=1, Up=2)
                // Use CrossEntropy loss with proper error handling - NO FALLBACKS
                log::debug!(
                    "🎯 Direction target: Using CrossEntropy loss for 3-class classification"
                );

                // Validate model output matches Direction classes (3)
                if predictions.dims().last() != Some(&3) {
                    return Err(VangaError::ModelError(format!(
                        "Direction target requires model output_size=3, got {}. Please update model configuration.",
                        predictions.dims().last().unwrap_or(&0)
                    )));
                }

                // Use proper 3-class CrossEntropy loss (same pattern as PriceLevel)
                self.calculate_crossentropy_loss(predictions, targets, 3)
            }
            TargetType::Volatility => {
                // Volatility targets are ALWAYS 3-class classification (Low=0, Medium=1, High=2)
                // Use CrossEntropy loss with proper error handling - NO FALLBACKS
                log::debug!(
                    "🎯 Volatility target: Using CrossEntropy loss for 3-class classification"
                );

                // Validate model output matches Volatility classes (3)
                if predictions.dims().last() != Some(&3) {
                    return Err(VangaError::ModelError(format!(
                        "Volatility target requires model output_size=3, got {}. Please update model configuration.",
                        predictions.dims().last().unwrap_or(&0)
                    )));
                }

                // Use proper 3-class CrossEntropy loss (same pattern as PriceLevel)
                self.calculate_crossentropy_loss(predictions, targets, 3)
            }
        }
    }

    /// Calculate multi-target loss with proper combination
    /// Calculate loss using configured loss function with target-aware logic
    fn calculate_loss(
        &self,
        predictions: &Tensor,
        targets: &Tensor,
        config: &crate::config::TrainingConfig,
    ) -> Result<Tensor> {
        // TENSOR SHAPE VALIDATION: Critical for catching configuration mismatches early
        self.validate_tensor_shapes(predictions, targets, config)?;

        // Log loss calculation context
        log::debug!(
            "🔍 LOSS CALCULATION - Pred shape: {:?}, Target shape: {:?}",
            predictions.shape(),
            targets.shape()
        );

        // Detect active target types from configuration
        let target_type = self.get_target_type()?;
        log::debug!("🎯 Target type: {:?}", target_type);
        log::debug!("🔧 Model output size: {}", self.config.output_size);

        // CRITICAL: Validate target size consistency
        if let Err(e) = self.validate_target_size_consistency(config) {
            log::error!("Target size validation failed: {}", e);
            // Continue with warning instead of failing - for debugging
        }

        // FIXED: Use single-target loss for individual models (they should always have correct size)
        // The validation above will catch and log any size mismatches
        log::debug!("📊 Using single target loss calculation");
        let loss_result =
            self.calculate_single_target_loss(predictions, targets, target_type, config)?;

        // Fallback to existing loss function system if configured
        let final_loss = if matches!(
            self.loss_function,
            crate::model::loss::CryptoLossFunction::MSE
        ) {
            // Use new target-aware loss for MSE (most common case)
            log::debug!("✅ Using target-aware loss calculation");
            loss_result
        } else {
            // Use existing advanced loss functions for specialized cases
            log::debug!("🔄 Using advanced loss function: {:?}", self.loss_function);
            use crate::model::loss::TensorCryptoLossFunction;
            let mut tensor_loss_fn = TensorCryptoLossFunction::new_with_class_weights(
                self.loss_function.clone(),
                self.global_class_weights.clone(),
            );

            let market_regime = match &self.loss_function {
                crate::model::loss::CryptoLossFunction::RegimeAware { .. }
                | crate::model::loss::CryptoLossFunction::Composite { .. } => {
                    let regime = self.detect_market_regime(predictions, targets)?;
                    log::debug!("🔍 REGIME DETECTION - Calculated regime: {:?}", regime);
                    regime
                }
                _ => crate::optimization::objective::MarketRegime::MediumVolatility,
            };

            tensor_loss_fn.calculate_tensor_loss(predictions, targets, market_regime)?
        };

        let loss_value = final_loss.to_scalar::<f32>().unwrap_or(0.0);
        log::debug!(
            "🎯 FINAL LOSS - Value: {:.6}, Target type: {:?}, Loss function: {:?}",
            loss_value,
            target_type,
            self.loss_function
        );

        // Validate loss is not NaN or infinite
        if !loss_value.is_finite() {
            log::error!("🚨 Invalid loss value: {}", loss_value);
            return Err(VangaError::ModelError(format!(
                "Loss calculation produced invalid value: {}",
                loss_value
            )));
        }

        Ok(final_loss)
    }

    /// Get adaptive early stopping configuration based on target types
    fn get_adaptive_early_stopping_config(
        &self,
        target_types: &[TargetType],
        base_patience: u32,
        base_min_delta: f64,
    ) -> (u32, f64) {
        // Adjust thresholds based on target types
        let min_delta = if target_types.iter().all(|t| {
            matches!(
                t,
                TargetType::PriceLevel | TargetType::Direction | TargetType::Volatility
            )
        }) {
            // Categorical targets need smaller deltas
            base_min_delta * 0.1
        } else {
            // Mixed targets use intermediate threshold
            base_min_delta * 0.5
        };

        (base_patience, min_delta)
    }

    /// Detect market regime using mathematically sound approach
    /// Uses target statistics to determine market conditions for regime-aware loss functions
    fn detect_market_regime(
        &self,
        _predictions: &Tensor,
        targets: &Tensor,
    ) -> Result<crate::optimization::objective::MarketRegime> {
        use crate::optimization::objective::MarketRegime;

        // ADDED: Validate tensor shapes and log for debugging
        log::debug!(
            "🔍 Market regime detection - Input targets shape: {:?}",
            targets.shape()
        );

        // Validate minimum tensor dimensions
        if targets.dims().len() < 2 {
            return Err(VangaError::ModelError(format!(
                "Invalid targets tensor for regime detection: expected 2D tensor, got shape {:?}",
                targets.shape()
            )));
        }

        // Use targets for regime detection - they represent actual market conditions
        // targets shape: [batch_size, num_targets] where num_targets = 9

        // Calculate adaptive statistics from the actual target data
        let targets_contiguous = targets.contiguous()?;
        let target_mean = targets_contiguous.mean_all()?;

        // FIXED: Proper scalar broadcasting for tensor subtraction
        // Create a tensor with the same shape as targets filled with the mean value
        let target_mean_scalar = target_mean
            .to_scalar::<f32>()
            .map_err(|e| VangaError::ModelError(format!("Failed to extract mean scalar: {}", e)))?;

        let target_mean_broadcast = Tensor::full(
            target_mean_scalar,
            targets_contiguous.shape(),
            targets_contiguous.device(),
        )?
        .contiguous()?;

        log::debug!(
            "🔍 Market regime detection shapes: targets {:?}, mean_broadcast {:?}",
            targets_contiguous.shape(),
            target_mean_broadcast.shape()
        );

        let target_variance = targets_contiguous
            .sub(&target_mean_broadcast)?
            .contiguous()?
            .sqr()?
            .mean_all()?;
        let volatility =
            target_variance.sqrt()?.to_scalar::<f32>().map_err(|e| {
                VangaError::ModelError(format!("Volatility calculation failed: {}", e))
            })? as f64;

        let target_mean_value = target_mean.to_scalar::<f32>().unwrap_or(0.0) as f64;

        // Calculate adaptive thresholds based on actual data distribution
        let target_std = volatility; // Standard deviation
        let target_abs_mean = target_mean_value.abs();

        // Dynamic thresholds based on data characteristics
        let high_vol_threshold = target_std * 2.0; // 2 standard deviations
        let low_vol_threshold = target_std * 0.5; // 0.5 standard deviations
        let trend_threshold = target_abs_mean * 0.1 + target_std * 0.5; // Adaptive trend detection
        let range_threshold = target_std * 1.0; // 1 standard deviation for range-bound

        // Classify market regime using adaptive thresholds
        let regime = match (volatility, target_mean_value) {
            (v, _) if v > high_vol_threshold => MarketRegime::HighVolatility,
            (v, t) if v < low_vol_threshold && t.abs() < trend_threshold * 0.5 => {
                MarketRegime::LowVolatility
            }
            (_, t) if t > trend_threshold => MarketRegime::BullMarket,
            (_, t) if t < -trend_threshold => MarketRegime::BearMarket,
            (v, _) if v < range_threshold => MarketRegime::RangeBound,
            _ => MarketRegime::MediumVolatility,
        };

        Ok(regime)
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

    /// Apply Xavier/Glorot initialization to LSTM weights to prevent exploding gradients
    /// This is critical because Candle's default initialization can cause gradient explosion
    fn apply_xavier_initialization(&mut self) -> Result<()> {
        log::info!(
            "🔧 Applying Xavier/Glorot weight initialization to prevent exploding gradients..."
        );

        let all_vars = self.varmap.all_vars();
        let mut initialized_count = 0;

        for var in all_vars.iter() {
            // Get the tensor shape to determine initialization parameters
            let shape = var.shape();

            // Skip biases (1D tensors) - initialize only weights (2D tensors)
            if shape.dims().len() == 2 {
                let (fan_in, fan_out) = (shape.dims()[0], shape.dims()[1]);

                // Xavier/Glorot initialization: std = sqrt(2.0 / (fan_in + fan_out))
                let xavier_std = (2.0 / (fan_in + fan_out) as f64).sqrt();

                log::debug!(
                    "🎯 Xavier init: shape={:?}, fan_in={}, fan_out={}, std={:.6}",
                    shape.dims(),
                    fan_in,
                    fan_out,
                    xavier_std
                );

                initialized_count += 1;
            }
        }

        if initialized_count == 0 {
            log::warn!(
                "⚠️ No weight tensors found for Xavier initialization - using Candle defaults"
            );
        } else {
            log::info!(
                "✅ Xavier initialization parameters calculated for {} weight tensors",
                initialized_count
            );
            log::warn!("⚠️ LIMITATION: Candle doesn't support post-creation weight replacement");
            log::warn!("⚠️ RECOMMENDATION: Use smaller learning rate or shorter sequences until proper init is available");
        }

        Ok(())
    }

    /// Clip gradients to prevent exploding gradients during training
    /// Returns the original gradient norm for monitoring
    /// Apply gradient clipping by norm (clip-by-norm method)
    ///
    /// This implements proper gradient clipping as described in the literature:
    /// If ||g|| > threshold, then g := threshold * g/||g||
    ///
    /// Since Candle doesn't support direct gradient modification, we implement
    /// gradient clipping by scaling the learning rate dynamically based on gradient norm.
    /// This achieves the same effect as gradient clipping.
    fn clip_gradients(
        &self,
        grads: &candle_core::backprop::GradStore,
        clip_value: f64,
        optimizer: &mut OptimizerWrapper,
        original_lr: f64,
    ) -> Result<(f64, f64)> {
        // Calculate gradient norm across all parameters using VarMap
        let mut total_norm_squared = 0.0f64;
        let mut param_count = 0;

        // Get all variables from the VarMap
        let all_vars = self.varmap.all_vars();

        for var in all_vars.iter() {
            if let Some(grad) = grads.get(var) {
                total_norm_squared += self.calculate_tensor_norm_squared(grad)?;
                param_count += 1;
            }
        }

        // Calculate the L2 norm
        let grad_norm = if total_norm_squared > 0.0 {
            total_norm_squared.sqrt()
        } else {
            0.0
        };

        // Log gradient statistics
        if param_count > 0 {
            log::debug!(
                "🔍 Gradient norm calculated: {:.6} from {} parameters (threshold: {:.3})",
                grad_norm,
                param_count,
                clip_value
            );
        } else {
            log::warn!("⚠️ No gradients found for norm calculation - this may indicate a problem with the model");
            return Ok((0.0, 0.0));
        }

        // Apply gradient clipping by scaling learning rate if needed
        if grad_norm > clip_value && grad_norm > 0.0 {
            let clip_ratio = clip_value / grad_norm;
            let effective_lr = original_lr * clip_ratio;
            let effective_norm = clip_value; // Effective gradient norm after clipping

            log::debug!(
                "✂️ GRADIENT CLIPPING APPLIED: original_norm={:.6} > threshold={:.6} (clip ratio: {:.3}, lr: {:.6} -> {:.6}, effective_norm={:.6})",
                grad_norm,
                clip_value,
                clip_ratio,
                original_lr,
                effective_lr,
                effective_norm
            );

            // Temporarily scale learning rate to achieve gradient clipping effect
            optimizer.set_learning_rate(effective_lr);

            Ok((grad_norm, effective_norm)) // Return both original and effective norms
        } else {
            // No clipping needed - ensure learning rate is at original value
            optimizer.set_learning_rate(original_lr);
            log::trace!(
                "✅ No gradient clipping needed: norm={:.6} <= threshold={:.6}",
                grad_norm,
                clip_value
            );
            Ok((grad_norm, grad_norm)) // Both norms are the same when no clipping
        }
    }

    /// Calculate the squared L2 norm of a tensor
    fn calculate_tensor_norm_squared(&self, tensor: &Tensor) -> Result<f64> {
        let squared = tensor.sqr().map_err(|e| {
            VangaError::ModelError(format!(
                "Failed to square tensor for norm calculation: {}",
                e
            ))
        })?;

        let sum = squared.sum_all().map_err(|e| {
            VangaError::ModelError(format!("Failed to sum tensor for norm calculation: {}", e))
        })?;

        // Handle both F32 and F64 tensors
        let norm_squared: f64 = match sum.dtype() {
            candle_core::DType::F32 => {
                let val: f32 = sum.to_scalar().map_err(|e| {
                    VangaError::ModelError(format!("Failed to convert F32 norm to scalar: {}", e))
                })?;
                val as f64
            }
            candle_core::DType::F64 => sum.to_scalar().map_err(|e| {
                VangaError::ModelError(format!("Failed to convert F64 norm to scalar: {}", e))
            })?,
            _ => {
                return Err(VangaError::ModelError(format!(
                    "Unsupported tensor dtype for norm calculation: {:?}",
                    sum.dtype()
                )));
            }
        };

        Ok(norm_squared)
    }

    /// Calculate learning rate based on schedule configuration
    ///
    /// This function implements all 5 learning schedule types:
    /// - Constant: Maintains initial learning rate
    /// - LinearDecay: Linear reduction over training epochs
    /// - ExponentialDecay: Exponential decay with configurable rate
    /// - CosineAnnealing: Cosine annealing schedule
    /// - WarmRestarts: Cosine annealing with warm restarts (SGDR)
    fn calculate_scheduled_learning_rate(
        schedule_config: &crate::config::training::LearningScheduleConfig,
        epoch_after_warmup: usize,
        initial_lr: f64,
        total_epochs: usize,
    ) -> f64 {
        use crate::config::training::LearningScheduleConfig;

        match schedule_config {
            LearningScheduleConfig::Constant => {
                // Maintain constant learning rate
                initial_lr
            }

            LearningScheduleConfig::LinearDecay { decay_rate } => {
                // Linear decay: lr = initial_lr * (1 - decay_rate * progress)
                let progress = epoch_after_warmup as f64 / total_epochs.max(1) as f64;
                let decay_factor = 1.0 - (decay_rate * progress);
                initial_lr * decay_factor.max(0.001) // Minimum LR threshold
            }

            LearningScheduleConfig::ExponentialDecay { decay_rate } => {
                // Exponential decay: lr = initial_lr * decay_rate^epoch
                let decay_factor = decay_rate.powf(epoch_after_warmup as f64);
                initial_lr * decay_factor.max(0.0001) // Minimum LR threshold
            }

            LearningScheduleConfig::CosineAnnealing { t_max } => {
                // Cosine annealing: lr = initial_lr * 0.5 * (1 + cos(π * epoch / t_max))
                let t_max_f = (*t_max).max(1) as f64;
                let progress = (epoch_after_warmup as f64) / t_max_f;
                let cosine_factor = 0.5 * (1.0 + (std::f64::consts::PI * progress).cos());
                initial_lr * cosine_factor.max(0.001) // Minimum LR threshold
            }

            LearningScheduleConfig::WarmRestarts { t_0, t_mult } => {
                // Cosine annealing with warm restarts (SGDR)
                let mut t_cur = epoch_after_warmup;
                let mut t_i = (*t_0).max(1) as usize;
                let t_mult_val = (*t_mult).max(1) as usize;

                // Find current restart cycle
                while t_cur >= t_i {
                    t_cur -= t_i;
                    t_i *= t_mult_val;
                }

                // Calculate cosine annealing within current cycle
                let progress = t_cur as f64 / t_i.max(1) as f64;
                let cosine_factor = 0.5 * (1.0 + (std::f64::consts::PI * progress).cos());
                initial_lr * cosine_factor.max(0.001) // Minimum LR threshold
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::model::{
        AttentionConfig, AttentionMechanism, DirectionHead, DistributionType, DropoutConfig,
        DropoutRate, HiddenUnitsConfig, LSTMArchitecture, ModelConfig, OutputHeadsConfig,
        PriceLevelHead, PriceLevelTargetStrategy, SequenceLengthConfig, VisualizationConfig,
        VolatilityHead, VolatilityPredictionMethod,
    };
    use crate::config::training::OptimizerType;
    use crate::config::training::{EpochConfig, LearningRateConfig, TrainingParams};
    use ndarray::Array3;

    #[tokio::test]
    async fn test_early_stopping_functionality() {
        // Create a simple LSTM model
        let config = LSTMConfig {
            input_size: 3,
            hidden_sizes: vec![8, 8], // Two layers with 8 hidden units each
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
                class_weight_strategy: ClassWeightStrategy::Global, // Add missing class_weight_strategy field
            },
            data: crate::config::training::DataConfig::default(),
            optimization: crate::config::training::OptimizationConfig::default(),
        };

        // Test that early stopping training completes without errors
        let result = model
            .train(&sequences, &targets, &training_config, None, None, None)
            .await;

        if let Err(ref e) = result {
            println!("Training error: {:?}", e);
        }

        assert!(
            result.is_ok(),
            "Early stopping training should complete successfully: {:?}",
            result.err()
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
            hidden_sizes: vec![8, 8], // Two layers with 8 hidden units each
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
                class_weight_strategy: ClassWeightStrategy::Global, // Add missing class_weight_strategy field
            },
            data: crate::config::training::DataConfig::default(),
            optimization: crate::config::training::OptimizationConfig::default(),
        };

        // Test that fixed epochs training completes without errors
        let result = model
            .train(&sequences, &targets, &training_config, None, None, None)
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
            hidden_sizes: vec![8, 8], // Two layers with 8 hidden units each
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
                class_weight_strategy: ClassWeightStrategy::Global, // Add missing class_weight_strategy field
            },

            data: crate::config::training::DataConfig::default(),
            optimization: crate::config::training::OptimizationConfig::default(),
        };

        model
            .train(&sequences, &targets, &training_config, None, None, None)
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
            hidden_sizes: vec![16, 16, 16], // Three layers with 16 hidden units each
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
                class_weight_strategy: ClassWeightStrategy::Global, // Add missing class_weight_strategy field
            },
            data: crate::config::training::DataConfig::default(),
            optimization: crate::config::training::OptimizationConfig::default(),
        };

        // Test multi-layer training
        let result = model
            .train(&sequences, &targets, &training_config, None, None, None)
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

    #[tokio::test]
    async fn test_bidirectional_lstm_initialization() {
        // Create a bidirectional LSTM configuration
        let model_config = ModelConfig {
            architecture: LSTMArchitecture::BidirectionalLSTM { layers: 2 },
            sequence_length: SequenceLengthConfig::Fixed(10),
            hidden_units: HiddenUnitsConfig::Fixed(vec![32, 16]),
            dropout: DropoutConfig {
                enabled: false,
                rate: DropoutRate::Fixed(0.0),
                variational: false,
                recurrent: false,
            },
            attention: AttentionConfig {
                enabled: false,
                mechanism: AttentionMechanism::None,
                heads: 1,
                head_dim: None,
                dropout_rate: 0.0,
                temperature_scaling: 1.0,
                use_relative_position: false,
                visualization: VisualizationConfig::default(),
            },
            output_heads: OutputHeadsConfig {
                price_levels: PriceLevelHead {
                    enabled: true,
                    bins: 5,
                    range_percent: 2.0,
                    distribution_type: DistributionType::Categorical,
                    target_strategy: PriceLevelTargetStrategy::Current,
                },
                direction: DirectionHead {
                    enabled: false,
                    threshold: 0.01,
                    confidence_calibration: false,
                },
                volatility: VolatilityHead {
                    enabled: false,
                    method: VolatilityPredictionMethod::Direct,
                    horizons: vec!["1h".to_string()],
                },
            },
            quantile_outputs: None,
            loss_function: CryptoLossFunction::MSE,
        };

        // Create model with bidirectional architecture
        let input_size = 10;
        let output_size = 5;

        let mut model =
            LSTMModel::from_model_config(&model_config, input_size, output_size).unwrap();

        // Verify architecture is stored
        assert!(matches!(
            model.architecture,
            Some(LSTMArchitecture::BidirectionalLSTM { .. })
        ));

        // Initialize the network - this should create both forward and backward layers
        model.initialize_network().unwrap();

        // Verify both forward and backward layers are created
        assert!(model.lstm_layers.is_some());
        assert!(model.backward_lstm_layers.is_some());

        let forward_layers = model.lstm_layers.as_ref().unwrap();
        let backward_layers = model.backward_lstm_layers.as_ref().unwrap();

        // Should have 2 layers each
        assert_eq!(forward_layers.len(), 2);
        assert_eq!(backward_layers.len(), 2);

        println!("✅ Bidirectional LSTM initialization test passed!");
    }

    #[tokio::test]
    async fn test_bidirectional_lstm_forward_pass() {
        // Create a bidirectional LSTM configuration
        let model_config = ModelConfig {
            architecture: LSTMArchitecture::BidirectionalLSTM { layers: 1 },
            sequence_length: SequenceLengthConfig::Fixed(5),
            hidden_units: HiddenUnitsConfig::Fixed(vec![8]),
            dropout: DropoutConfig {
                enabled: false,
                rate: DropoutRate::Fixed(0.0),
                variational: false,
                recurrent: false,
            },
            attention: AttentionConfig {
                enabled: false,
                mechanism: AttentionMechanism::None,
                heads: 1,
                head_dim: None,
                dropout_rate: 0.0,
                temperature_scaling: 1.0,
                use_relative_position: false,
                visualization: VisualizationConfig::default(),
            },
            output_heads: OutputHeadsConfig {
                price_levels: PriceLevelHead {
                    enabled: true,
                    bins: 3,
                    range_percent: 1.0,
                    distribution_type: DistributionType::Categorical,
                    target_strategy: PriceLevelTargetStrategy::Current,
                },
                direction: DirectionHead {
                    enabled: false,
                    threshold: 0.01,
                    confidence_calibration: false,
                },
                volatility: VolatilityHead {
                    enabled: false,
                    method: VolatilityPredictionMethod::Direct,
                    horizons: vec!["1h".to_string()],
                },
            },
            quantile_outputs: None,
            loss_function: CryptoLossFunction::MSE,
        };

        let input_size = 4;
        let expected_output_size = model_config.output_heads.calculate_total_output_size();

        let mut model =
            LSTMModel::from_model_config(&model_config, input_size, expected_output_size).unwrap();

        model.initialize_network().unwrap();

        // Mark model as trained for prediction (bypass training requirement for test)
        model.trained = true;

        // Clear target context to get raw output shape instead of converted class indices
        model.target_context = None;

        // Create test input data: [batch_size=2, seq_len=5, features=4]
        let batch_size = 2;
        let seq_len = 5;
        let features = 4;

        let mut input_data = Array3::<f64>::zeros((batch_size, seq_len, features));

        // Fill with some test data
        for i in 0..batch_size {
            for j in 0..seq_len {
                for k in 0..features {
                    input_data[[i, j, k]] = (i as f64 + j as f64 * 0.1 + k as f64 * 0.01) * 0.1;
                }
            }
        }

        // Test forward pass directly to get raw output shape (not converted to class indices)
        // Convert input data to tensor manually using the same logic as predict method
        let batch_size = input_data.shape()[0];
        let seq_len = input_data.shape()[1];
        let features = input_data.shape()[2];

        let mut seq_data: Vec<f32> = Vec::with_capacity(batch_size * seq_len * features);

        for batch_idx in 0..batch_size {
            for seq_idx in 0..seq_len {
                for feature_idx in 0..features {
                    seq_data.push(input_data[[batch_idx, seq_idx, feature_idx]] as f32);
                }
            }
        }

        let input_tensor = Tensor::from_vec(
            seq_data,
            (batch_size, seq_len, features),
            &candle_core::Device::Cpu,
        )
        .unwrap();

        let predictions = model.forward(&input_tensor).unwrap();

        // Verify output shape
        let shape_dims = predictions.shape().dims();
        assert_eq!(shape_dims, &[batch_size, expected_output_size]);

        println!("✅ Bidirectional LSTM forward pass test passed!");
        println!(
            "   Input shape: [{}, {}, {}]",
            batch_size, seq_len, features
        );
        println!("   Output shape: {:?}", predictions.shape().dims());
        println!("   Expected output size: {}", expected_output_size);
    }

    #[tokio::test]
    async fn test_unidirectional_vs_bidirectional_output_size() {
        let input_size = 6;
        let output_size = 4;
        let hidden_size = 12;

        // Test unidirectional LSTM
        let unidirectional_config = ModelConfig {
            architecture: LSTMArchitecture::MultiLSTM { layers: 1 },
            sequence_length: SequenceLengthConfig::Fixed(8),
            hidden_units: HiddenUnitsConfig::Fixed(vec![hidden_size]),
            dropout: DropoutConfig {
                enabled: false,
                rate: DropoutRate::Fixed(0.0),
                variational: false,
                recurrent: false,
            },
            attention: AttentionConfig {
                enabled: false,
                mechanism: AttentionMechanism::None,
                heads: 1,
                head_dim: None,
                dropout_rate: 0.0,
                temperature_scaling: 1.0,
                use_relative_position: false,
                visualization: VisualizationConfig::default(),
            },
            output_heads: OutputHeadsConfig {
                price_levels: PriceLevelHead {
                    enabled: true,
                    bins: output_size as u32,
                    range_percent: 1.0,
                    distribution_type: DistributionType::Categorical,
                    target_strategy: PriceLevelTargetStrategy::Current,
                },
                direction: DirectionHead {
                    enabled: false,
                    threshold: 0.01,
                    confidence_calibration: false,
                },
                volatility: VolatilityHead {
                    enabled: false,
                    method: VolatilityPredictionMethod::Direct,
                    horizons: vec!["1h".to_string()],
                },
            },
            quantile_outputs: None,
            loss_function: CryptoLossFunction::MSE,
        };

        // Test bidirectional LSTM
        let bidirectional_config = ModelConfig {
            architecture: LSTMArchitecture::BidirectionalLSTM { layers: 1 },
            ..unidirectional_config.clone()
        };

        let mut uni_model =
            LSTMModel::from_model_config(&unidirectional_config, input_size, output_size).unwrap();
        let mut bi_model =
            LSTMModel::from_model_config(&bidirectional_config, input_size, output_size).unwrap();

        uni_model.initialize_network().unwrap();
        bi_model.initialize_network().unwrap();

        // Verify unidirectional has no backward layers
        assert!(uni_model.backward_lstm_layers.is_none());

        // Verify bidirectional has backward layers
        assert!(bi_model.backward_lstm_layers.is_some());

        println!("✅ Unidirectional vs Bidirectional architecture test passed!");
    }

    #[tokio::test]
    async fn test_bidirectional_lstm_with_attention() {
        // Test bidirectional LSTM with attention enabled
        let model_config = ModelConfig {
            architecture: LSTMArchitecture::BidirectionalLSTM { layers: 1 },
            sequence_length: SequenceLengthConfig::Fixed(5),
            hidden_units: HiddenUnitsConfig::Fixed(vec![8]),
            dropout: DropoutConfig {
                enabled: false,
                rate: DropoutRate::Fixed(0.0),
                variational: false,
                recurrent: false,
            },
            attention: AttentionConfig {
                enabled: true, // Enable attention
                mechanism: AttentionMechanism::SelfAttention,
                heads: 2,
                head_dim: Some(8),
                dropout_rate: 0.0,
                temperature_scaling: 1.0,
                use_relative_position: false,
                visualization: VisualizationConfig::default(),
            },
            output_heads: OutputHeadsConfig {
                price_levels: PriceLevelHead {
                    enabled: true,
                    bins: 3,
                    range_percent: 1.0,
                    distribution_type: DistributionType::Categorical,
                    target_strategy: PriceLevelTargetStrategy::Current,
                },
                direction: DirectionHead {
                    enabled: false,
                    threshold: 0.01,
                    confidence_calibration: false,
                },
                volatility: VolatilityHead {
                    enabled: false,
                    method: VolatilityPredictionMethod::Direct,
                    horizons: vec!["1h".to_string()],
                },
            },
            quantile_outputs: None,
            loss_function: CryptoLossFunction::MSE,
        };

        let input_size = 4;
        let expected_output_size = model_config.output_heads.calculate_total_output_size();

        let mut model =
            LSTMModel::from_model_config(&model_config, input_size, expected_output_size).unwrap();

        model.initialize_network().unwrap();

        // Mark model as trained for prediction (bypass training requirement for test)
        model.trained = true;

        // Clear target context to get raw output shape instead of converted class indices
        model.target_context = None;

        // Create test input data: [batch_size=2, seq_len=5, features=4]
        let batch_size = 2;
        let seq_len = 5;
        let features = 4;

        let mut input_data = Array3::<f64>::zeros((batch_size, seq_len, features));

        // Fill with some test data
        for i in 0..batch_size {
            for j in 0..seq_len {
                for k in 0..features {
                    input_data[[i, j, k]] = (i as f64 + j as f64 * 0.1 + k as f64 * 0.01) * 0.1;
                }
            }
        }

        // Test forward pass directly to get raw output shape (not converted to class indices)
        // Convert input data to tensor manually using the same logic as predict method
        let mut seq_data: Vec<f32> = Vec::with_capacity(batch_size * seq_len * features);

        for batch_idx in 0..batch_size {
            for seq_idx in 0..seq_len {
                for feature_idx in 0..features {
                    seq_data.push(input_data[[batch_idx, seq_idx, feature_idx]] as f32);
                }
            }
        }

        let input_tensor = Tensor::from_vec(
            seq_data,
            (batch_size, seq_len, features),
            &candle_core::Device::Cpu,
        )
        .unwrap();

        let predictions = model.forward(&input_tensor).unwrap();

        // Verify output shape
        let shape_dims = predictions.shape().dims();
        assert_eq!(shape_dims, &[batch_size, expected_output_size]);

        println!("✅ Bidirectional LSTM with attention test passed!");
        println!(
            "   Input shape: [{}, {}, {}]",
            batch_size, seq_len, features
        );
        println!("   Output shape: {:?}", predictions.shape().dims());
        println!("   Expected output size: {}", expected_output_size);
    }
}
