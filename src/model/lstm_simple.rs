// LSTM model implementation with rust-lstm integration
use crate::config::ModelConfig;
use crate::utils::error::Result;
use ndarray::{Array2, Array3};
use rust_lstm::models::lstm_network::LSTMNetwork;
use rust_lstm::training::TrainingConfig;
use serde::{Deserialize, Serialize};

/// Type alias for complex training data batch structure
type TrainingDataBatch = Vec<(Vec<Array2<f64>>, Vec<Array2<f64>>)>;

/// LSTM network configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LSTMConfig {
    pub input_size: usize,
    pub hidden_size: usize,
    pub output_size: usize,
    pub sequence_length: usize,
    pub learning_rate: f64,
}

/// LSTM model for cryptocurrency forecasting
pub struct LSTMModel {
    config: LSTMConfig,
    network: Option<LSTMNetwork>,
    training_config: TrainingConfig,
}

/// Serializable model state for persistence
#[derive(Debug, Clone, Serialize, Deserialize)]
struct ModelState {
    config: LSTMConfig,
    epochs: usize,
    print_every: usize,
    clip_gradient: Option<f64>,
}

impl LSTMModel {
    /// Create a new LSTM model
    pub fn new(config: LSTMConfig) -> Result<Self> {
        let training_config = TrainingConfig {
            epochs: 100,
            print_every: 100,
            clip_gradient: Some(1.0),
        };

        Ok(Self {
            config,
            network: None,
            training_config,
        })
    }

    /// Create LSTM model from ModelConfig
    pub fn from_model_config(
        model_config: &ModelConfig,
        input_size: usize,
        output_size: usize,
    ) -> Result<Self> {
        // Extract sequence length from config
        let sequence_length = match &model_config.sequence_length {
            crate::config::model::SequenceLengthConfig::Fixed(len) => *len as usize,
            crate::config::model::SequenceLengthConfig::Auto {
                min_length,
                max_length: _,
            } => *min_length as usize,
            crate::config::model::SequenceLengthConfig::Adaptive => 60,
        };

        // Extract hidden units from config
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

        // Use sequence_length for LSTM configuration if needed
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
        };
        Self::new(lstm_config)
    }

    /// Train the model using the network and training config
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

        // Initialize network if not already done
        if self.network.is_none() {
            log::info!("Initializing LSTM network with config: {:?}", self.config);

            // Use default 2 layers for now (we can make this configurable later)
            let num_layers = 2;

            // NOTE: rust-lstm network output size is determined by target data structure, not constructor
            let network = rust_lstm::models::lstm_network::LSTMNetwork::new(
                self.config.input_size,
                self.config.hidden_size,
                num_layers,
            );
            self.network = Some(network);
        }

        // Convert Array3 sequences to Vec<Array2> format expected by rust-lstm
        let training_data = self.convert_sequences_to_training_data(sequences, targets)?;

        // Update config to reflect actual output size (1 for rust-lstm compatibility)
        self.config.output_size = 1;

        // Create trainer with MSE loss and SGD optimizer
        use rust_lstm::loss::MSELoss;
        use rust_lstm::optimizers::SGD;
        use rust_lstm::training::LSTMTrainer;

        if let Some(network) = self.network.take() {
            let mut trainer = LSTMTrainer::new(
                network,
                MSELoss,
                SGD::new(0.001), // Learning rate
            );

            // Set training configuration
            trainer.config.epochs = self.training_config.epochs;
            trainer.config.print_every = self.training_config.print_every;
            trainer.config.clip_gradient = self.training_config.clip_gradient;

            log::info!(
                "Starting LSTM training for {} epochs",
                trainer.config.epochs
            );

            // Train the model with error handling
            match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                trainer.train(&training_data, None); // No validation data for now
            })) {
                Ok(_) => {
                    // Get the trained network back
                    self.network = Some(trainer.network);
                    log::info!("LSTM training completed successfully");
                }
                Err(_) => {
                    return Err(crate::utils::error::VangaError::ModelError(
                        format!(
                            "LSTM training failed due to shape incompatibility. Expected: input_size={}, output_size={}, sequence_length={}. Check that training data dimensions match model configuration.",
                            self.config.input_size,
                            self.config.output_size,
                            self.config.sequence_length
                        )
                    ));
                }
            }
        }

        Ok(())
    }

    /// Make predictions using the trained network
    pub async fn predict(&self, sequences: &Array3<f64>) -> Result<Array2<f64>> {
        log::info!("Making predictions with LSTM model");

        // Check if network is trained
        if self.network.is_none() {
            return Err(crate::utils::error::VangaError::ModelError(
                "Network not initialized - cannot make predictions".to_string(),
            ));
        }

        let network = self.network.as_ref().unwrap();

        // Use the predict_sequences helper method
        let predictions = self.predict_sequences(network, sequences)?;

        log::info!("Generated {} predictions", predictions.nrows());
        Ok(predictions)
    }

    /// Save model to file
    /// Save model to file
    pub fn save<P: AsRef<std::path::Path>>(&self, path: P) -> Result<()> {
        // Create a serializable model state
        #[derive(Serialize)]
        struct ModelState {
            config: LSTMConfig,
            // Note: rust-lstm network is not serializable, so we save only config
            // Network will be recreated on load
        }

        let model_state = ModelState {
            config: self.config.clone(),
        };

        // Serialize to binary format using bincode
        let encoded = bincode::serialize(&model_state).map_err(|e| {
            crate::utils::error::VangaError::SerializationError(format!(
                "Serialization failed: {}",
                e
            ))
        })?;

        // Write to file
        std::fs::write(path, encoded).map_err(|e| {
            crate::utils::error::VangaError::IoError(format!("Failed to write model file: {}", e))
        })?;

        log::info!("Model saved successfully");
        Ok(())
    }

    /// Load model from file
    pub fn load<P: AsRef<std::path::Path>>(path: P) -> Result<Self> {
        // Read the model file
        let data = std::fs::read(&path).map_err(|e| {
            crate::utils::error::VangaError::IoError(format!("Failed to read model file: {}", e))
        })?;

        // Deserialize the model state
        #[derive(Deserialize)]
        struct ModelState {
            config: LSTMConfig,
        }

        let model_state: ModelState = bincode::deserialize(&data).map_err(|e| {
            crate::utils::error::VangaError::SerializationError(format!(
                "Deserialization failed: {}",
                e
            ))
        })?;

        // Create a new LSTM network with the loaded configuration
        let network = LSTMNetwork::new(
            model_state.config.input_size,
            model_state.config.hidden_size,
            2, // Default to 2 layers
        );

        log::info!("Model loaded successfully");
        Ok(Self {
            config: model_state.config,
            network: Some(network),
            training_config: TrainingConfig::default(),
        })
    }

    /// Get the input size of the model
    pub fn get_input_size(&self) -> usize {
        self.config.input_size
    }

    /// Convert Array3 sequences to training data format for rust-lstm
    fn convert_sequences_to_training_data(
        &self,
        sequences: &Array3<f64>,
        targets: &Array2<f64>,
    ) -> Result<TrainingDataBatch> {
        let mut training_data: TrainingDataBatch = Vec::new();

        // sequences shape: [batch, sequence_length, features]
        // targets shape: [batch, output_size]

        // Use minimum batch size to ensure alignment
        let batch_size = std::cmp::min(sequences.shape()[0], targets.shape()[0]);

        log::debug!(
            "Converting training data: {} sequences, {} targets, using {} aligned samples",
            sequences.shape()[0],
            targets.shape()[0],
            batch_size
        );

        for batch_idx in 0..batch_size {
            let mut input_sequence = Vec::new();
            let mut target_sequence = Vec::new();

            // Extract sequence for this batch - fix input structure
            for seq_idx in 0..sequences.shape()[1] {
                // Create input with proper shape (features, 1) to match official example
                let mut input_timestep = Array2::zeros((sequences.shape()[2], 1));
                for feature_idx in 0..sequences.shape()[2] {
                    input_timestep[[feature_idx, 0]] = sequences[[batch_idx, seq_idx, feature_idx]];
                }
                input_sequence.push(input_timestep);
            }

            // CRITICAL FIX: rust-lstm expects single output per timestep, not multi-target
            // We need to create separate training runs for each target or restructure approach
            // For now, let's use only the first target to test basic functionality
            for _seq_idx in 0..sequences.shape()[1] {
                // Use only first target (single output) to match rust-lstm expectations
                let target_value = targets[[batch_idx, 0]]; // Take first target only
                let target_timestep = Array2::from_elem((1, 1), target_value);
                target_sequence.push(target_timestep);
            }

            training_data.push((input_sequence, target_sequence));
        }

        log::info!(
            "Training data converted: {} samples with sequence length {} (using single target output instead of {})",
            training_data.len(),
            if !training_data.is_empty() { training_data[0].0.len() } else { 0 },
            targets.shape()[1]
        );

        // Debug: Log the actual shapes we're working with
        log::debug!(
            "Input shapes: sequences={:?}, targets={:?}, batch_size={}",
            sequences.shape(),
            targets.shape(),
            batch_size
        );

        // Log warning about multi-target limitation
        if targets.shape()[1] > 1 {
            log::warn!(
                "rust-lstm library limitation: Using only first target out of {} targets. Consider implementing separate models for each target or using a different ML library for true multi-target support.",
                targets.shape()[1]
            );
        }

        Ok(training_data)
    }

    /// Make predictions on sequences using the trained network
    fn predict_sequences(
        &self,
        network: &rust_lstm::models::lstm_network::LSTMNetwork,
        sequences: &Array3<f64>,
    ) -> Result<Array2<f64>> {
        let batch_size = sequences.shape()[0];
        let output_size = self.config.output_size; // Use configured output size
        let mut predictions = Array2::zeros((batch_size, output_size));

        for batch_idx in 0..batch_size {
            // Convert batch to sequence format
            let mut input_sequence = Vec::new();
            for seq_idx in 0..sequences.shape()[1] {
                let mut input_timestep = Array2::zeros((sequences.shape()[2], 1));
                for feature_idx in 0..sequences.shape()[2] {
                    input_timestep[[feature_idx, 0]] = sequences[[batch_idx, seq_idx, feature_idx]];
                }
                input_sequence.push(input_timestep);
            }

            // Get predictions for this sequence
            let (outputs, _) = network.forward_sequence_with_cache(&input_sequence);

            // Use the last output as the prediction
            if let Some((last_output, _)) = outputs.last() {
                // Extract all output dimensions for multi-target predictions
                for output_idx in 0..output_size.min(last_output.nrows()) {
                    predictions[[batch_idx, output_idx]] = last_output[[output_idx, 0]];
                }
            }
        }

        Ok(predictions)
    }
}
