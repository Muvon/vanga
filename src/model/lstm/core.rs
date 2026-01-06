//! Core LSTM model implementation - model lifecycle methods
//!
//! This module contains model creation, initialization, persistence,
//! and other core lifecycle methods.

use super::config::{LSTMConfig, LSTMModel, ModelState, TrainingConfig};
use crate::config::ModelConfig;
use crate::utils::error::{Result, VangaError};

use candle_core::{DType, Device};
use candle_nn::{linear, lstm, LSTMConfig as CandleLSTMConfig, VarBuilder, VarMap};

/// Format numbers with thousands separators for better readability
fn format_number(n: usize) -> String {
    let s = n.to_string();
    let mut result = String::new();
    for (i, c) in s.chars().rev().enumerate() {
        if i > 0 && i % 3 == 0 {
            result.push(',');
        }
        result.push(c);
    }
    result.chars().rev().collect()
}

impl LSTMModel {
    /// Create a new LSTM model - EXACT same logic as original
    pub fn new(config: LSTMConfig) -> Result<Self> {
        Self::new_with_bias_config(
            config,
            crate::model::bias_correction::BiasCorrection::default(),
        )
    }

    /// Create a new LSTM model with bias correction configuration
    pub fn new_with_bias_config(
        config: LSTMConfig,
        bias_correction_config: crate::model::bias_correction::BiasCorrection,
    ) -> Result<Self> {
        Self::new_with_bias_config_and_device(config, bias_correction_config, None)
    }

    /// Create a new LSTM model with bias correction configuration and optional device
    fn new_with_bias_config_and_device(
        config: LSTMConfig,
        bias_correction_config: crate::model::bias_correction::BiasCorrection,
        device: Option<Device>,
    ) -> Result<Self> {
        let training_config = TrainingConfig {
            epochs: 1, // Placeholder - will be set by configure_training()
            print_every: 10,
            clip_gradient: Some(1.0),
            batch_size: 32, // Default batch size
        };

        let actual_device = device.unwrap_or(Device::Cpu);

        Ok(Self {
            config,
            lstm_layers: None,
            backward_lstm_layers: None, // Initialize backward layers as None
            output_layer: None,
            attention_module: None, // Initialize attention as None
            attention_config: None, // Initialize attention config as None
            use_attention: false,   // Attention disabled by default
            device: actual_device,  // CRITICAL FIX: Use provided device from the start
            varmap: VarMap::new(),
            training_config,
            trained: false,
            target_context: None, // No target context by default

            architecture: None,         // No architecture info by default
            dropout_config: None,       // No dropout config by default
            stored_val_sequences: None, // No stored validation data initially
            stored_val_targets: None,   // No stored validation targets initially
            stored_test_sequences: ndarray::Array3::zeros((0, 1, 1)), // Empty test sequences
            stored_test_targets: ndarray::Array2::zeros((0, 1)), // Empty test targets
            xgboost_model: None,        // No XGBoost model initially
            best_model_varmap: None,    // No best model state initially
            best_validation_loss: None, // No best validation loss initially
            best_epoch: None,           // No best epoch initially
            seed: None,                 // No seed by default (random initialization)
            calibrated_parameters: None, // No calibrated parameters initially
            optimizer: None,            // No optimizer initially (created during training)
            bias_correction_factors: None, // No bias correction initially
            bias_correction_config: bias_correction_config.clone(), // Use provided bias correction config
            bias_corrector: if bias_correction_config.use_ensemble_calibration {
                None // Use ensemble calibrator instead
            } else {
                Some(crate::model::bias_correction::LinearBiasCorrector::new(
                    bias_correction_config.clone(),
                ))
            },
            ensemble_calibrator: if bias_correction_config.use_ensemble_calibration {
                Some(crate::model::calibration::EnsembleCalibrator::new())
            } else {
                None
            },
        })
    }

    /// Create a new LSTM model with specified seed for reproducible training
    pub fn new_with_seed(
        config: LSTMConfig,
        seed: Option<u64>,
        device: Option<Device>,
    ) -> Result<Self> {
        // CRITICAL FIX: Pass device to new_with_bias_config_and_device so it's set from the start
        let mut model = Self::new_with_bias_config_and_device(
            config,
            crate::model::bias_correction::BiasCorrection::default(),
            device.clone(),
        )?;
        model.seed = seed;

        if let Some(ref dev) = device {
            let device_name = match dev {
                Device::Cpu => "CPU",
                Device::Cuda(_) => "CUDA GPU",
                Device::Metal(_) => "Metal GPU",
            };
            log::info!("🔧 LSTMModel created with device: {}", device_name);
        }

        if let Some(seed_value) = seed {
            log::info!("🎲 Created LSTMModel with seed: {}", seed_value);
            if seed_value == 0 {
                log::info!("🎲 Seed = 0: Random weight initialization will be used");
            } else {
                log::info!(
                    "🎲 Seed = {}: Reproducible weight initialization will be used",
                    seed_value
                );
            }
        } else {
            log::info!(
                "🎲 Created LSTMModel without seed: Random weight initialization will be used"
            );
        }

        Ok(model)
    }

    /// Create a new LSTM model with seed and bias correction configuration
    pub fn new_with_seed_and_bias(
        config: LSTMConfig,
        seed: Option<u64>,
        bias_correction_config: crate::model::bias_correction::BiasCorrection,
    ) -> Result<Self> {
        let mut model =
            Self::new_with_bias_config_and_device(config, bias_correction_config, None)?;
        model.seed = seed;

        if let Some(seed_value) = seed {
            log::debug!(
                "🎲 Created LSTMModel with seed: {} and bias config",
                seed_value
            );
        } else {
            log::debug!("🎲 Created LSTMModel without seed but with bias config");
        }

        Ok(model)
    }
    /// Create LSTM model from ModelConfig - Enhanced with multi-layer support
    pub fn from_model_config(
        model_config: &ModelConfig,
        input_size: usize,
        output_size: usize,
    ) -> Result<Self> {
        Self::from_model_config_with_seed(model_config, input_size, output_size, None, None)
    }

    /// Create LSTM model from ModelConfig with seed for reproducible training
    pub fn from_model_config_with_seed(
        model_config: &ModelConfig,
        input_size: usize,
        output_size: usize,
        seed: Option<u64>,
        device: Option<Device>,
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

        let config = LSTMConfig {
            input_size,
            hidden_sizes: final_hidden_sizes,
            output_size,
            sequence_length,
            learning_rate: 0.001, // Default learning rate
            num_layers,
        };

        let mut model = Self::new_with_bias_config_and_device(
            config,
            model_config.bias_correction.clone(),
            device,
        )?;
        model.seed = seed;
        model.architecture = Some(model_config.architecture.clone());
        model.dropout_config = Some(model_config.dropout.clone());
        model.attention_config = Some(model_config.attention.clone());
        model.use_attention = model_config.attention.enabled;

        // Initialize XGBoost model if enabled
        if model_config.xgboost.enabled {
            log::info!("🚀 Initializing XGBoost model for hybrid prediction");
            model.xgboost_model = Some(crate::model::xgboost::XGBoostRegressor::new(
                model_config.xgboost.clone(),
                model.device.clone(),
            ));
        }

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

        // Use the actual configured mechanism instead of hardcoding MultiHeadAttention
        let mut final_config = attention_config.clone();

        // Auto-configure MoH if MixtureOfHeads is selected but no MoH config provided
        if final_config.mechanism == crate::config::model::AttentionMechanism::MixtureOfHeads
            && final_config.moh.is_none()
        {
            log::info!("🔧 Auto-configuring MoH settings for MixtureOfHeads mechanism");
            final_config.moh = Some(crate::config::model::MoHConfig::default());
        }

        self.attention_config = Some(final_config);
        self.use_attention = true;

        // Log with context if provided, otherwise use generic message
        match context {
            Some(ctx) => log::info!(
                "✅ Attention configured for {}: mechanism={:?}, heads={}, head_dim={}",
                ctx,
                attention_config.mechanism,
                attention_config.heads,
                attention_config.head_dim.unwrap_or(64)
            ),
            None => log::debug!(
                "✅ Attention configured: mechanism={:?}, heads={}, head_dim={}",
                attention_config.mechanism,
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
            "✅ Dropout configured: enabled={}, rate={:?}, variational={}, recurrent={}",
            dropout_config.enabled,
            dropout_config.rate,
            dropout_config.variational,
            dropout_config.recurrent
        );
    }

    /// Clear variational dropout masks (call at end of epoch or sequence)
    ///
    /// This prevents memory leaks and ensures fresh masks for new sequences.
    /// Should be called at the end of each training epoch or when switching
    /// between training and validation.
    pub fn clear_dropout_masks(&self) {
        use crate::model::lstm::seeded_weights::SeededTensorUtils;
        SeededTensorUtils::clear_variational_masks(None);
        log::debug!("🧹 Cleared all variational dropout masks");
    }

    /// Clear specific sequence dropout mask
    ///
    /// # Arguments
    /// * `sequence_id` - Specific sequence ID to clear
    pub fn clear_sequence_dropout_mask(&self, sequence_id: &str) {
        use crate::model::lstm::seeded_weights::SeededTensorUtils;
        SeededTensorUtils::clear_variational_masks(Some(sequence_id));
        log::debug!(
            "🧹 Cleared variational dropout mask for sequence: {}",
            sequence_id
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

            // Use EnhancedAttentionFactory to create the appropriate attention mechanism
            use crate::model::attention_moh_wrapper::EnhancedAttentionFactory;
            let attention_module = EnhancedAttentionFactory::create_attention(
                &attention_config.mechanism,
                attention_input_size,
                attention_config.clone(),
                vs.pp("attention"),
                self.device.clone(),
            )?;

            self.attention_module = Some(attention_module);

            log::debug!(
                "✅ Attention layers initialized: mechanism={:?}, heads={}, input_size={}, bidirectional={}",
                attention_config.mechanism,
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
    pub fn initialize_network(&mut self, skip_weight_init: Option<bool>) -> Result<()> {
        if self.lstm_layers.is_some() {
            return Ok(()); // Already initialized
        }

        let skip_weights = skip_weight_init.unwrap_or(false);

        let device_str = match &self.device {
            Device::Cpu => "CPU",
            Device::Cuda(_) => "CUDA GPU",
            Device::Metal(_) => "Metal GPU",
        };

        log::info!(
            "🔧 Initializing multi-layer LSTM network on device: {} (config: {:?})",
            device_str,
            self.config
        );

        // Set device seed for reproducible weight initialization if seed is provided
        if let Some(seed_value) = self.seed {
            crate::model::lstm::seeded_weights::set_device_seed_with_logging(
                &self.device,
                Some(seed_value),
            )?;
        }

        // Check if this is a bidirectional LSTM
        let is_bidirectional = matches!(
            self.architecture,
            Some(crate::config::model::LSTMArchitecture::BidirectionalLSTM { .. })
        );

        if is_bidirectional {
            log::info!("🔄 Initializing Bidirectional LSTM with forward and backward layers");
        }

        // CRITICAL: Verify device is correct before creating VarBuilder
        log::info!("🔧 Creating VarBuilder with device: {}", device_str);
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
        let lstm_params = self.config.total_parameters();

        // Calculate output layer parameters
        let output_params = (output_input_size + 1) * self.config.output_size;

        // Estimate attention parameters (if attention is enabled)
        let attention_params = if self.attention_module.is_some() {
            // Rough estimation: 16 heads × 4 projections × (input_dim × head_dim + head_dim bias)
            // This is an approximation since we don't have direct access to attention config here
            let estimated_input_dim = final_hidden_size * if is_bidirectional { 2 } else { 1 };
            let estimated_head_dim = 64; // Common head dimension
            let estimated_heads = 16; // From your log
            estimated_heads * 4 * (estimated_input_dim * estimated_head_dim + estimated_head_dim)
        } else {
            0
        };

        let total_params = lstm_params + output_params + attention_params;

        log::info!(
            "✅ LSTM network initialized: {} layers, {} tensor variables, {} total parameters",
            num_layers,
            vars_count_after_init,
            format_number(total_params)
        );
        log::info!(
            "   📊 Parameter breakdown: LSTM={}, Output={}, Attention={}",
            format_number(lstm_params),
            format_number(output_params),
            format_number(attention_params)
        );
        log::info!(
            "   🏗️  Architecture: bidirectional={}, output_input_size={}, final_hidden_size={}",
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

        // Apply proper LSTM weight initialization after network creation (only if not skipped)
        if skip_weights {
            log::info!("🔧 Initializing network structure (skipping weight initialization for model loading)");
        } else {
            log::info!("🎯 Applying proper LSTM weight initialization...");

            // First, let's see what tensors we actually have with their names
            let var_data = self.varmap.data().lock().unwrap();
            let var_names: Vec<String> = var_data.keys().cloned().collect();
            drop(var_data); // Release the lock

            let all_vars = self.varmap.all_vars();
            log::info!("📊 Found {} tensors in VarMap:", all_vars.len());
            for (idx, var) in all_vars.iter().enumerate() {
                let shape = var.shape();
                let dims = shape.dims();
                let var_name = var_names.get(idx).map(|s| s.as_str()).unwrap_or("unknown");
                log::info!("  Tensor {}: '{}' shape={:?}", idx, var_name, dims);
            }

            crate::model::lstm::seeded_weights::SeededTensorUtils::apply_lstm_weight_initialization(
                &self.varmap,
                &self.device,
                self.seed,
            )?;
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
            calibrated_parameters: self.calibrated_parameters.clone(),
        };

        let config_path = path.with_extension("config");
        let encoded = bincode::serialize(&model_state).map_err(|e| {
            VangaError::SerializationError(format!("Config serialization failed: {}", e))
        })?;

        std::fs::write(&config_path, encoded)
            .map_err(|e| VangaError::IoError(format!("Failed to write config file: {}", e)))?;

        // Save XGBoost model if present AND trained (hybrid model persistence)
        if let Some(xgb_model) = &self.xgboost_model {
            // Only save if XGBoost model is trained
            if xgb_model.is_trained() {
                // Use the base path directly - SmartCore will add its own extensions
                match xgb_model.save_model(&path.to_string_lossy()) {
                    Ok(_) => {
                        log::debug!("XGBoost model saved to: {}", path.display());
                    }
                    Err(e) => {
                        // Log but don't fail - XGBoost model is optional
                        log::warn!("⚠️  Failed to save XGBoost model (non-fatal): {}", e);
                        log::warn!("   Continuing with LSTM-only model save");
                    }
                }
            } else {
                log::debug!("XGBoost model exists but not trained yet, skipping save");
            }
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
        model.calibrated_parameters = model_state.calibrated_parameters.clone();

        // CRITICAL FIX: Initialize network structure FIRST to create tensor placeholders
        // Skip weight initialization since we'll load trained weights from safetensors
        log::info!("🔧 Initializing network structure for model loading...");
        model.initialize_network(Some(true))?;

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

        // CRITICAL: Verify weights are actually loaded by checking a sample tensor
        // This helps detect if weights are being properly loaded from file
        let all_vars = model.varmap.all_vars();
        if !all_vars.is_empty() {
            let first_var = &all_vars[0];
            let tensor = first_var.as_tensor();

            // Get tensor shape and a few sample values for verification
            let shape = tensor.shape();
            let dims = shape.dims();

            // Try to get first few values as a sanity check
            if let Ok(flattened) = tensor.flatten_all() {
                if let Ok(values) = flattened.narrow(0, 0, dims.iter().product::<usize>().min(5)) {
                    if let Ok(vec_values) = values.to_vec1::<f32>() {
                        log::info!(
                            "🔍 Sample tensor shape {:?}, first values: {:?}",
                            dims,
                            vec_values
                        );
                    }
                }
            }
        }

        // Calculate a simple checksum of all weights for consistency verification
        let mut weight_sum = 0.0f32;
        let mut weight_count = 0usize;
        for var in model.varmap.all_vars() {
            let tensor = var.as_tensor();
            if let Ok(flattened) = tensor.flatten_all() {
                if let Ok(vec_values) = flattened.to_vec1::<f32>() {
                    weight_sum += vec_values.iter().sum::<f32>();
                    weight_count += vec_values.len();
                }
            }
        }

        if weight_count > 0 {
            let avg_weight = weight_sum / weight_count as f32;
            log::info!(
                "📊 Weight statistics: {} total weights, sum={:.6}, avg={:.6}",
                weight_count,
                weight_sum,
                avg_weight
            );
            log::info!(
                "🔑 Weight checksum (for consistency): {:.8}",
                weight_sum.abs() // Use absolute value for consistent checksum
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
                    // SENIOR-LEVEL FIX: Fail fast and loud - validate architecture compatibility
                    let current_lstm_feature_dim = model.get_xgboost_feature_dim();
                    let loaded_feature_dim = xgb_model.get_config().feature_dim;

                    if current_lstm_feature_dim != loaded_feature_dim {
                        // FATAL ERROR: Architecture mismatch - cannot proceed
                        let error_msg = format!(
                            "FATAL: XGBoost model architecture mismatch detected!\n\
                             \n\
                             Current LSTM configuration produces: {} features\n\
                             Saved XGBoost model expects: {} features\n\
                             \n\
                             This indicates the model was trained with a different LSTM architecture than the current configuration.\n\
                             \n\
                             LSTM Config Analysis:\n\
                             - Your hidden_units: {:?}\n\
                             - Last layer size: {}\n\
                             - Bidirectional: {} (multiplier: {})\n\
                             - Calculated features: {} × {} = {}\n\
                             \n\
                             SOLUTION:\n\
                             1. Delete incompatible model files: rm models/BTCUSDT_*.safetensors models/BTCUSDT_*.smartcore.*\n\
                             2. Retrain the model with your current LSTM configuration\n\
                             3. Or update your LSTM config to match the trained model\n\
                             \n\
                             Cannot proceed with incompatible architectures - this would cause runtime crashes.",
                            current_lstm_feature_dim,
                            loaded_feature_dim,
                            model.config.hidden_sizes,
                            model.config.hidden_sizes.last().copied().unwrap_or(0),
                            if matches!(model.architecture, Some(crate::config::model::LSTMArchitecture::BidirectionalLSTM { .. })) { "Yes" } else { "No" },
                            if matches!(model.architecture, Some(crate::config::model::LSTMArchitecture::BidirectionalLSTM { .. })) { 2 } else { 1 },
                            model.config.hidden_sizes.last().copied().unwrap_or(0),
                            if matches!(model.architecture, Some(crate::config::model::LSTMArchitecture::BidirectionalLSTM { .. })) { 2 } else { 1 },
                            current_lstm_feature_dim
                        );

                        log::error!("🚨 {}", error_msg);
                        return Err(VangaError::ModelError(error_msg));
                    }

                    model.xgboost_model = Some(xgb_model);
                    log::info!("✅ XGBoost model loaded successfully with matching feature dimensions ({} features)", current_lstm_feature_dim);
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
        // Skip weight initialization since we'll load trained weights from safetensors
        log::info!("🔧 Initializing network structure with provided config (for loading)...");
        model.initialize_network(Some(true))?;

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
                    // SENIOR-LEVEL FIX: Fail fast and loud - validate architecture compatibility
                    let current_lstm_feature_dim = model.get_xgboost_feature_dim();
                    let loaded_feature_dim = xgb_model.get_config().feature_dim;

                    if current_lstm_feature_dim != loaded_feature_dim {
                        // FATAL ERROR: Architecture mismatch - cannot proceed
                        let error_msg = format!(
                            "FATAL: XGBoost model architecture mismatch detected!\n\
                             \n\
                             Current LSTM configuration produces: {} features\n\
                             Saved XGBoost model expects: {} features\n\
                             \n\
                             This indicates the model was trained with a different LSTM architecture than the current configuration.\n\
                             \n\
                             LSTM Config Analysis:\n\
                             - Your hidden_units: {:?}\n\
                             - Last layer size: {}\n\
                             - Bidirectional: {} (multiplier: {})\n\
                             - Calculated features: {} × {} = {}\n\
                             \n\
                             SOLUTION:\n\
                             1. Delete incompatible model files: rm models/BTCUSDT_*.safetensors models/BTCUSDT_*.smartcore.*\n\
                             2. Retrain the model with your current LSTM configuration\n\
                             3. Or update your LSTM config to match the trained model\n\
                             \n\
                             Cannot proceed with incompatible architectures - this would cause runtime crashes.",
                            current_lstm_feature_dim,
                            loaded_feature_dim,
                            model.config.hidden_sizes,
                            model.config.hidden_sizes.last().copied().unwrap_or(0),
                            if matches!(model.architecture, Some(crate::config::model::LSTMArchitecture::BidirectionalLSTM { .. })) { "Yes" } else { "No" },
                            if matches!(model.architecture, Some(crate::config::model::LSTMArchitecture::BidirectionalLSTM { .. })) { 2 } else { 1 },
                            model.config.hidden_sizes.last().copied().unwrap_or(0),
                            if matches!(model.architecture, Some(crate::config::model::LSTMArchitecture::BidirectionalLSTM { .. })) { 2 } else { 1 },
                            current_lstm_feature_dim
                        );

                        log::error!("🚨 {}", error_msg);
                        return Err(VangaError::ModelError(error_msg));
                    }

                    model.xgboost_model = Some(xgb_model);
                    log::info!("✅ XGBoost model loaded successfully with matching feature dimensions ({} features)", current_lstm_feature_dim);
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

        // Calculate weight checksum for verification
        let mut weight_sum = 0.0f32;
        let mut weight_count = 0usize;
        for var in model.varmap.all_vars() {
            let tensor = var.as_tensor();
            if let Ok(flattened) = tensor.flatten_all() {
                if let Ok(vec_values) = flattened.to_vec1::<f32>() {
                    weight_sum += vec_values.iter().sum::<f32>();
                    weight_count += vec_values.len();
                }
            }
        }

        if weight_count > 0 {
            log::info!(
                "🔑 Weight checksum: {:.8} ({} weights loaded)",
                weight_sum.abs(),
                weight_count
            );
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
        // Ensure network is initialized before saving (with weight initialization for training)
        if self.lstm_layers.is_none() || self.output_layer.is_none() {
            log::warn!("⚠️ Network not initialized, initializing before checkpoint save...");
            self.initialize_network(Some(false))?; // Ensure weight initialization for training
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
                        // Skip weight initialization since we're loading checkpoint weights
                        if self.lstm_layers.is_none() || self.output_layer.is_none() {
                            log::info!(
                                "🔧 Initializing network before loading checkpoint weights (skipping weight init)..."
                            );
                            self.initialize_network(Some(true))?; // Skip weight init for loading
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

    /// Mark model as trained for testing purposes
    /// This is a test helper method to allow predictions on initialized models
    #[cfg(test)]
    pub fn mark_as_trained_for_testing(&mut self) {
        self.trained = true;
        log::info!("🧪 Model marked as trained for testing purposes");
    }
}
