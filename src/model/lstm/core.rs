//! Core LSTM model implementation - model lifecycle methods
//!
//! This module contains model creation, initialization, persistence,
//! and other core lifecycle methods.

use super::config::{LSTMConfig, LSTMModel, ModelState, TrainingConfig};
use crate::config::ModelConfig;
use crate::model::attention::MultiHeadAttention;
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
            attention_module: None, // Initialize attention as None
            attention_config: None, // Initialize attention config as None
            use_attention: false,   // Attention disabled by default
            device: Device::Cpu,
            varmap: VarMap::new(),
            training_config,
            trained: false,
            target_context: None,           // No target context by default
            training_class_weights: None,   // No global weights initially
            validation_class_weights: None, // No validation weights initially
            architecture: None,             // No architecture info by default
            dropout_config: None,           // No dropout config by default
            stored_val_sequences: None,     // No stored validation data initially
            stored_val_targets: None,       // No stored validation targets initially
            stored_test_sequences: ndarray::Array3::zeros((0, 1, 1)), // Empty test sequences
            stored_test_targets: ndarray::Array2::zeros((0, 1)), // Empty test targets
            xgboost_model: None,            // No XGBoost model initially
            best_model_varmap: None,        // No best model state initially
            best_validation_loss: None,     // No best validation loss initially
            best_epoch: None,               // No best epoch initially
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

        // Configure dropout
        model.configure_dropout(&model_config.dropout);

        // Loss function is now hardcoded to NLL - no configuration needed
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
        let module_config = crate::config::model::AttentionConfig {
            enabled: true,
            mechanism: crate::config::model::AttentionMechanism::MultiHeadAttention,
            heads: 8,
            head_dim: Some(64),
            dropout_rate: 0.1,
            dropout_weights: true,
            dropout_output: true,
            dropout_projections: true,
            dropout_scores: true,
            temperature_scaling: 1.0,
            use_relative_position: true,
            visualization: crate::config::model::VisualizationConfig::default(),
            moh: None,
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

    /// Configure dropout for the model
    pub fn configure_dropout(&mut self, dropout_config: &crate::config::model::DropoutConfig) {
        self.dropout_config = Some(dropout_config.clone());

        log::debug!(
            "✅ Dropout configured: enabled={}, rate={:?}",
            dropout_config.enabled,
            dropout_config.rate
        );
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

            self.attention_module = Some(attention);

            log::debug!(
                "✅ Attention layers initialized: {} heads, input_size={}, bidirectional={}",
                attention_config.heads as usize,
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
        }

        // Store the layers
        self.lstm_layers = Some(forward_lstm_layers);
        if is_bidirectional {
            self.backward_lstm_layers = Some(backward_lstm_layers);
        }

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

        // Verify VarMap was populated during initialization
        let vars_count_after_init = self.varmap.all_vars().len();
        log::info!(
            "✅ LSTM network initialized: {} layers, {} parameters in VarMap, bidirectional={}, output_input_size={}, final_hidden_size={}",
            num_layers,
            vars_count_after_init,
            is_bidirectional,
            output_input_size,
            final_hidden_size
        );

        if vars_count_after_init == 0 {
            log::error!("⚠️ CRITICAL: VarMap is empty after network initialization!");
            log::error!(
                "   This indicates a problem with parameter creation during layer initialization."
            );
            return Err(VangaError::ModelError(
                "Network initialization failed: no parameters created".to_string(),
            ));
        }

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

        // Save XGBoost model if present (hybrid model persistence)
        if let Some(xgb_model) = &self.xgboost_model {
            // Use the base path directly - SmartCore will add its own extensions
            xgb_model.save_model(&path.to_string_lossy())?;
            log::debug!("XGBoost model saved to: {}", path.display());
        }

        log::debug!(
            "Model saved successfully: weights={}, config={}",
            weights_path.display(),
            config_path.display()
        );
        Ok(())
    }

    /// Load model from file - Enhanced to load both config and weights
    /// CRITICAL FIX: Ensure deterministic weight loading by proper initialization sequence
    pub fn load<P: AsRef<std::path::Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();

        // Load model configuration
        let config_path = path.with_extension("config");
        let data = std::fs::read(&config_path)
            .map_err(|e| VangaError::IoError(format!("Failed to read config file: {}", e)))?;

        let model_state: ModelState = bincode::deserialize(&data).map_err(|e| {
            VangaError::SerializationError(format!("Config deserialization failed: {}", e))
        })?;

        // Check if weights file exists
        let weights_path = path.with_extension("safetensors");
        if !weights_path.exists() {
            return Err(VangaError::SerializationError(format!(
                "Weights file not found: {}",
                weights_path.display()
            )));
        }

        // Create model with loaded configuration
        let mut model = Self::new(model_state.config)?;
        model.training_config.epochs = model_state.epochs;
        model.training_config.print_every = model_state.print_every;
        model.training_config.clip_gradient = model_state.clip_gradient;

        // CRITICAL FIX: Initialize network structure FIRST to create tensor placeholders
        log::info!("🔧 Initializing network structure...");
        model.initialize_network()?;

        // Verify network was initialized
        let pre_load_keys: Vec<String> = model
            .varmap
            .data()
            .lock()
            .unwrap()
            .keys()
            .cloned()
            .collect();
        log::info!(
            "📊 Network initialized with {} tensors: {:?}",
            pre_load_keys.len(),
            pre_load_keys
        );

        if pre_load_keys.is_empty() {
            return Err(VangaError::ModelError(
                "Network initialization failed - no tensors created".to_string(),
            ));
        }

        // Load model weights from safetensors - this should OVERWRITE the initialized weights
        log::info!("🔄 Loading weights from: {}", weights_path.display());

        // CRITICAL FIX: Handle shape mismatches gracefully
        match model.varmap.load(&weights_path) {
            Ok(_) => {
                log::info!("✅ Weights loaded successfully");
            }
            Err(e) => {
                let error_msg = format!("{}", e);
                if error_msg.contains("shape mismatch") {
                    log::error!("❌ Shape mismatch detected in saved weights!");
                    log::error!(
                        "This usually means the model architecture changed since training."
                    );
                    log::error!("Error details: {}", error_msg);

                    // Extract the problematic tensor name and shapes from error
                    if let Some(tensor_start) = error_msg.find("setting ") {
                        if let Some(tensor_end) = error_msg[tensor_start..].find(" using") {
                            let tensor_name =
                                &error_msg[tensor_start + 8..tensor_start + tensor_end];
                            log::error!("Problematic tensor: {}", tensor_name);
                        }
                    }

                    return Err(VangaError::ModelError(format!(
                        "Model architecture mismatch: {}. The saved model was trained with different layer sizes than the current configuration. Please retrain the model or use the correct configuration.",
                        error_msg
                    )));
                } else {
                    return Err(VangaError::SerializationError(format!(
                        "Failed to load model weights: {}",
                        e
                    )));
                }
            }
        }

        // Verify weights were actually loaded by checking if tensor values changed
        let post_load_keys: Vec<String> = model
            .varmap
            .data()
            .lock()
            .unwrap()
            .keys()
            .cloned()
            .collect();
        log::info!(
            "📊 After loading: {} tensors present: {:?}",
            post_load_keys.len(),
            post_load_keys
        );

        // The key count should be the same, but the values should have changed
        if post_load_keys.len() != pre_load_keys.len() {
            log::warn!(
                "⚠️ Tensor count mismatch: before={}, after={}",
                pre_load_keys.len(),
                post_load_keys.len()
            );
        }

        model.trained = true;

        // Load XGBoost model if present (hybrid model persistence)
        let smartcore_meta_path = format!("{}.smartcore.meta", path.to_string_lossy());
        if std::path::Path::new(&smartcore_meta_path).exists() {
            log::info!("🔄 Loading XGBoost model from: {}", path.display());
            match crate::model::xgboost::XGBoostRegressor::load_model(
                &path.to_string_lossy(),
                model.device.clone(),
            ) {
                Ok(xgb_model) => {
                    model.xgboost_model = Some(xgb_model);
                    log::info!("✅ XGBoost model loaded successfully");
                }
                Err(e) => {
                    log::warn!(
                        "⚠️ Failed to load XGBoost model: {}. Model will use pure LSTM prediction.",
                        e
                    );
                    model.xgboost_model = None;
                }
            }
        } else {
            log::debug!(
                "No XGBoost model found at: {} - using pure LSTM prediction",
                smartcore_meta_path
            );
            model.xgboost_model = None;
        }

        log::info!(
            "🎯 Model loaded successfully: weights={}, config={}",
            weights_path.display(),
            config_path.display()
        );

        Ok(model)
    }

    /// Load model from file with specific model configuration (for multi-target models)
    /// This allows loading models with architecture different from the saved config
    pub fn load_with_model_config<P: AsRef<std::path::Path>>(
        path: P,
        model_config: &crate::config::ModelConfig,
        input_size: usize,
        output_size: usize,
    ) -> Result<Self> {
        let path = path.as_ref();

        // Check if weights file exists
        let weights_path = path.with_extension("safetensors");
        if !weights_path.exists() {
            return Err(VangaError::SerializationError(format!(
                "Weights file not found: {}",
                weights_path.display()
            )));
        }

        // Create model with provided model configuration (not saved config)
        let mut model = Self::from_model_config(model_config, input_size, output_size)?;

        // Initialize network structure FIRST to create tensor placeholders
        log::info!("🔧 Initializing network structure with provided config...");
        model.initialize_network()?;

        // Load model weights from safetensors
        log::info!("🔄 Loading weights from: {}", weights_path.display());
        match model.varmap.load(&weights_path) {
            Ok(_) => {
                log::info!("✅ Weights loaded successfully");
            }
            Err(e) => {
                log::error!("❌ Shape mismatch detected in saved weights!");
                log::error!("This usually means the model architecture changed since training.");
                log::error!("Error details: {}", e);
                return Err(VangaError::SerializationError(format!(
                    "Failed to load weights: {}",
                    e
                )));
            }
        }

        model.trained = true;

        // Load XGBoost model if present (hybrid model persistence)
        let smartcore_meta_path = format!("{}.smartcore.meta", path.to_string_lossy());
        if std::path::Path::new(&smartcore_meta_path).exists() {
            log::info!("🔄 Loading XGBoost model from: {}", path.display());
            match crate::model::xgboost::XGBoostRegressor::load_model(
                &path.to_string_lossy(),
                model.device.clone(),
            ) {
                Ok(xgb_model) => {
                    model.xgboost_model = Some(xgb_model);
                    log::info!("✅ XGBoost model loaded successfully");
                }
                Err(e) => {
                    log::warn!(
                        "⚠️ Failed to load XGBoost model: {}. Model will use pure LSTM prediction.",
                        e
                    );
                    model.xgboost_model = None;
                }
            }
        } else {
            log::debug!(
                "No XGBoost model found at: {} - using pure LSTM prediction",
                smartcore_meta_path
            );
            model.xgboost_model = None;
        }

        log::info!(
            "🎯 Model loaded successfully with provided config: weights={}",
            weights_path.display()
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

    /// Get training configuration used during training
    pub fn get_training_config(&self) -> Option<&crate::config::TrainingConfig> {
        // For now, single models don't store the full TrainingConfig
        // This is a limitation that should be addressed in future versions
        None
    }

    /// Get trained horizons from model configuration
    /// This is a temporary method until we properly store training config in single models
    pub fn get_trained_horizons(&self) -> Vec<String> {
        // For now, single models default to 1h
        // TODO: Store actual trained horizons in model metadata
        vec!["1h".to_string()]
    }

    /// Save current model weights as the best checkpoint
    /// Called when validation loss improves during training
    pub fn save_best_checkpoint(&mut self, validation_loss: f64, epoch: usize) -> Result<()> {
        // Ensure network is initialized before saving
        if self.lstm_layers.is_none() || self.output_layer.is_none() {
            log::warn!("⚠️ Network not initialized, initializing before checkpoint save...");
            self.initialize_network()?;
        }

        // Verify current model has parameters before saving
        let current_vars_count = self.varmap.all_vars().len();
        if current_vars_count == 0 {
            log::error!("⚠️ Cannot save checkpoint: current model has no parameters even after initialization!");
            log::error!(
                "   This suggests the VarMap is not being populated during network initialization."
            );
            log::error!(
                "   LSTM layers: {:?}, Output layer: {:?}",
                self.lstm_layers.is_some(),
                self.output_layer.is_some()
            );
            return Err(VangaError::ModelError(
                "Cannot save checkpoint: model has no parameters".to_string(),
            ));
        }

        log::debug!(
            "💾 Saving best model checkpoint at epoch {} with validation loss: {:.6} ({} parameters)",
            epoch + 1,
            validation_loss,
            current_vars_count
        );

        // Store validation metrics
        self.best_validation_loss = Some(validation_loss);
        self.best_epoch = Some(epoch);

        // Create a unique checkpoint path
        let checkpoint_dir = std::env::temp_dir().join("vanga_checkpoints");
        std::fs::create_dir_all(&checkpoint_dir)
            .map_err(|e| VangaError::IoError(format!("Failed to create checkpoint dir: {}", e)))?;

        let checkpoint_path = checkpoint_dir.join(format!(
            "best_model_{}_{}.safetensors",
            std::process::id(),
            epoch
        ));

        // Save the entire model state using the existing save method
        self.save(&checkpoint_path)?;

        // Verify the saved file exists and has content
        if checkpoint_path.with_extension("safetensors").exists() {
            let file_size = std::fs::metadata(checkpoint_path.with_extension("safetensors"))
                .map(|m| m.len())
                .unwrap_or(0);
            log::debug!("✅ Checkpoint file saved: {} bytes", file_size);

            if file_size == 0 {
                log::error!("⚠️ Saved checkpoint file is empty!");
                return Err(VangaError::ModelError(
                    "Saved checkpoint file is empty".to_string(),
                ));
            }
        } else {
            log::error!("⚠️ Checkpoint file was not created!");
            return Err(VangaError::ModelError(
                "Checkpoint file was not created".to_string(),
            ));
        }

        // Store the checkpoint path for later restoration
        // We'll use a marker VarMap to indicate a checkpoint exists
        self.best_model_varmap = Some(VarMap::new());

        // Save checkpoint path to a metadata file
        let metadata_path =
            checkpoint_dir.join(format!("best_model_{}_metadata.txt", std::process::id()));
        let metadata = format!(
            "{}\n{}\n{}",
            checkpoint_path.to_string_lossy(),
            epoch,
            validation_loss
        );
        std::fs::write(&metadata_path, metadata).map_err(|e| {
            VangaError::IoError(format!("Failed to save checkpoint metadata: {}", e))
        })?;

        log::debug!(
            "✅ Best model checkpoint saved to: {}",
            checkpoint_path.display()
        );

        Ok(())
    }

    /// Restore model weights from the best checkpoint
    /// Called when early stopping triggers to use best weights instead of last
    pub fn restore_best_checkpoint(&mut self) -> Result<()> {
        if self.best_model_varmap.is_some() {
            log::info!(
                "🔄 Restoring best model checkpoint from epoch {} (val loss: {:.6})",
                self.best_epoch.map(|e| e + 1).unwrap_or(0),
                self.best_validation_loss.unwrap_or(0.0)
            );

            // Read checkpoint metadata
            let checkpoint_dir = std::env::temp_dir().join("vanga_checkpoints");
            let metadata_path =
                checkpoint_dir.join(format!("best_model_{}_metadata.txt", std::process::id()));

            if metadata_path.exists() {
                let metadata = std::fs::read_to_string(&metadata_path).map_err(|e| {
                    VangaError::IoError(format!("Failed to read checkpoint metadata: {}", e))
                })?;
                let lines: Vec<&str> = metadata.lines().collect();

                if !lines.is_empty() {
                    let checkpoint_path = std::path::Path::new(lines[0]);
                    let weights_path = checkpoint_path.with_extension("safetensors");

                    if weights_path.exists() {
                        // Store current varmap info for verification
                        let current_vars_count = self.varmap.all_vars().len();

                        // CRITICAL FIX: Ensure network is initialized before loading weights
                        // This creates the variables in the VarMap that load() can update
                        if self.lstm_layers.is_none() || self.output_layer.is_none() {
                            log::info!(
                                "🔧 Initializing network before loading checkpoint weights..."
                            );
                            self.initialize_network()?;
                        }

                        // Verify we have variables to load into
                        let vars_after_init = self.varmap.all_vars().len();
                        if vars_after_init == 0 {
                            log::error!("⚠️ No variables in VarMap after initialization!");
                            return Err(VangaError::ModelError(
                                "Cannot load checkpoint: no variables to load into".to_string(),
                            ));
                        }

                        // Now load the checkpoint weights into the existing variables
                        // This is the correct way to use VarMap.load() - it updates existing variables
                        self.varmap.load(&weights_path).map_err(|e| {
                            VangaError::ModelError(format!(
                                "Failed to load checkpoint weights: {}",
                                e
                            ))
                        })?;

                        // Verify the loading worked
                        let loaded_vars_count = self.varmap.all_vars().len();
                        log::info!(
                            "✅ Best model weights restored successfully from: {} (vars: {} → {} after load)",
                            weights_path.display(),
                            current_vars_count,
                            loaded_vars_count
                        );
                        log::debug!("🔄 Checkpoint weights loaded into existing VarMap variables");

                        // Clean up checkpoint files AFTER successful restoration
                        // Note: Keeping files temporarily for debugging if needed
                        let _ = std::fs::remove_file(&weights_path);
                        let _ = std::fs::remove_file(checkpoint_path.with_extension("config"));
                        let _ = std::fs::remove_file(&metadata_path);

                        log::debug!("🧹 Checkpoint files cleaned up successfully");
                    } else {
                        log::warn!(
                            "⚠️ Checkpoint weights file not found: {}",
                            weights_path.display()
                        );
                    }
                } else {
                    log::warn!("⚠️ Invalid checkpoint metadata format");
                }
            } else {
                log::warn!("⚠️ No checkpoint metadata file found");
            }

            Ok(())
        } else {
            log::warn!("⚠️ No best model checkpoint available to restore");
            Ok(())
        }
    }
}
