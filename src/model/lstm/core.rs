//! Core LSTM model implementation - model lifecycle methods
//!
//! This module contains model creation, initialization, persistence,
//! and other core lifecycle methods.

use super::config::{LSTMConfig, LSTMModel, ModelState, TrainingConfig};
use crate::config::ModelConfig;
use crate::model::attention::{AttentionConfig as AttentionModuleConfig, MultiHeadAttention};
use crate::model::loss::CryptoLossFunction;
use crate::utils::error::{Result, VangaError};

use candle_core::{DType, Device};
use candle_nn::{linear, lstm, LSTMConfig as CandleLSTMConfig, VarBuilder, VarMap};

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
    pub fn initialize_network(&mut self) -> Result<()> {
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
}
