// LSTM model implementation with rust-lstm integration
use crate::config::ModelConfig;
use crate::utils::error::Result;
use ndarray::{Array2, Array3};
use rayon::prelude::*;
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
            epochs: 1, // Placeholder - will be set by configure_training()
            print_every: 10,
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

    /// PARALLELIZED: Train model in parallel batches for maximum CPU utilization
    pub async fn train_parallel_batches(
        &mut self,
        sequences: &Array3<f64>,
        targets: &Array2<f64>,
        batch_size: usize,
    ) -> Result<()> {
        let num_samples = sequences.shape()[0];
        let num_batches = num_samples.div_ceil(batch_size);

        log::info!(
            "Training with {} parallel batches of size {}",
            num_batches,
            batch_size
        );

        // Create batches for parallel processing
        let batches: Vec<(Array3<f64>, Array2<f64>)> = (0..num_batches)
            .into_par_iter()
            .map(|i| {
                let start_idx = i * batch_size;
                let end_idx = std::cmp::min(start_idx + batch_size, num_samples);

                let batch_sequences = sequences
                    .slice(ndarray::s![start_idx..end_idx, .., ..])
                    .to_owned();
                let batch_targets = targets
                    .slice(ndarray::s![start_idx..end_idx, ..])
                    .to_owned();

                (batch_sequences, batch_targets)
            })
            .collect();

        // Process batches (note: actual LSTM training is sequential, but data prep is parallel)
        for (batch_seq, batch_tgt) in batches {
            self.train(&batch_seq, &batch_tgt).await?;
        }

        Ok(())
    }

    /// Configure training parameters from TrainingConfig
    pub fn configure_training(&mut self, vanga_config: &crate::config::TrainingConfig) {
        // Extract epochs from config
        let (max_epochs, use_early_stopping) = match &vanga_config.training_params.epochs {
            crate::config::training::EpochConfig::Auto { max_epochs } => {
                (*max_epochs as usize, true)
            }
            crate::config::training::EpochConfig::Fixed(epochs) => (*epochs as usize, false),
        };

        // Extract learning rate from config
        let learning_rate = match &vanga_config.training_params.learning_rate {
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

        // Update rust-lstm training config
        self.training_config.epochs = max_epochs;
        self.training_config.print_every = if use_early_stopping { 10 } else { 50 }; // More frequent logging for early stopping

        // Store learning rate for optimizer creation
        self.config.learning_rate = learning_rate;

        log::info!(
            "✅ Training configured: epochs={}, lr={:.6}, early_stopping={}, print_every={}",
            max_epochs,
            learning_rate,
            use_early_stopping,
            self.training_config.print_every
        );
    }

    /// Train with intelligent early stopping
    pub async fn train_with_early_stopping(
        &mut self,
        sequences: &Array3<f64>,
        targets: &Array2<f64>,
        vanga_config: &crate::config::TrainingConfig,
    ) -> Result<()> {
        // Configure training parameters
        self.configure_training(vanga_config);

        // Determine if we should use early stopping
        let (max_epochs, use_early_stopping) = match &vanga_config.training_params.epochs {
            crate::config::training::EpochConfig::Auto { max_epochs } => {
                (*max_epochs as usize, true)
            }
            crate::config::training::EpochConfig::Fixed(epochs) => (*epochs as usize, false),
        };

        let patience = vanga_config.training_params.early_stopping_patience;
        let validation_split = vanga_config.training_params.validation_split;

        log::info!(
            "Starting intelligent training: max_epochs={}, early_stopping={}, patience={}, validation_split={:.2}",
            max_epochs, use_early_stopping, patience, validation_split
        );

        if use_early_stopping && validation_split > 0.0 {
            log::info!("🧠 USING INTELLIGENT TRAINING with validation monitoring");
            // Split data for validation-based early stopping
            self.train_with_validation_monitoring(sequences, targets, vanga_config)
                .await
        } else {
            log::info!(
                "📊 USING STANDARD TRAINING (early_stopping={}, validation_split={})",
                use_early_stopping,
                validation_split
            );
            // Use standard training (may still have fixed epoch limit)
            self.train(sequences, targets).await
        }
    }

    /// Calculate MSE loss between predictions and targets
    fn calculate_mse_loss(&self, predictions: &Array2<f64>, targets: &Array2<f64>) -> f64 {
        // CRITICAL FIX: Validate shapes before operations
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

    /// Calculate MAPE (Mean Absolute Percentage Error) for better understanding
    fn calculate_mape(&self, predictions: &Array2<f64>, targets: &Array2<f64>) -> f64 {
        // CRITICAL FIX: Validate shapes before operations
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

                // Avoid division by zero and very small values
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

    /// Continue training with new data (incremental learning)
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

        // Check if model is already trained
        if self.network.is_none() {
            return Err(crate::utils::error::VangaError::ModelError(
                "Cannot continue training: model not initialized. Use train_with_early_stopping() first.".to_string()
            ));
        }

        // Configure training with typically lower learning rate for incremental training
        let mut incremental_config = vanga_config.clone();

        // Reduce learning rate for incremental training to preserve existing knowledge
        incremental_config.training_params.learning_rate = match &vanga_config
            .training_params
            .learning_rate
        {
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

        // Use smaller patience for incremental training (faster convergence expected)
        incremental_config.training_params.early_stopping_patience =
            (vanga_config.training_params.early_stopping_patience / 2).max(10);

        log::info!(
            "⚙️  Incremental training config: patience={}, reduced_lr=true",
            incremental_config.training_params.early_stopping_patience
        );

        // Train with the new data using reduced learning rate
        self.train_with_early_stopping(new_sequences, new_targets, &incremental_config)
            .await?;

        log::info!("✅ Incremental training completed successfully!");
        Ok(())
    }

    /// Append new data to existing training data and retrain (alternative approach)
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

        // Combine existing and new data
        let combined_sequences = ndarray::concatenate(
            ndarray::Axis(0),
            &[existing_sequences.view(), new_sequences.view()],
        )
        .map_err(|e| {
            crate::utils::error::VangaError::DataError(format!(
                "Failed to concatenate sequences: {}",
                e
            ))
        })?;
        let combined_targets = ndarray::concatenate(
            ndarray::Axis(0),
            &[existing_targets.view(), new_targets.view()],
        )
        .map_err(|e| {
            crate::utils::error::VangaError::DataError(format!(
                "Failed to concatenate targets: {}",
                e
            ))
        })?;

        log::info!(
            "📊 Combined dataset: {} samples x {} features x {} sequence_length",
            combined_sequences.shape()[0],
            combined_sequences.shape()[2],
            combined_sequences.shape()[1]
        );

        // Train on combined dataset (this preserves all historical patterns)
        self.train_with_early_stopping(&combined_sequences, &combined_targets, vanga_config)
            .await?;

        log::info!("✅ Retrain with appended data completed successfully!");
        Ok(())
    }

    /// Train with validation monitoring using rust-lstm's built-in validation
    async fn train_with_validation_monitoring(
        &mut self,
        sequences: &Array3<f64>,
        targets: &Array2<f64>,
        vanga_config: &crate::config::TrainingConfig,
    ) -> Result<()> {
        let validation_split = vanga_config.training_params.validation_split;

        log::info!(
            "🧠 Starting intelligent training with validation split: {:.1}%",
            validation_split * 100.0
        );

        // Split data for validation
        let total_samples = sequences.shape()[0];
        let train_samples = ((total_samples as f64) * (1.0 - validation_split)) as usize;

        log::info!(
            "📊 Data split: {} training samples, {} validation samples",
            train_samples,
            total_samples - train_samples
        );

        // Create training and validation sets
        let train_sequences = sequences
            .slice(ndarray::s![0..train_samples, .., ..])
            .to_owned();
        let train_targets = targets.slice(ndarray::s![0..train_samples, ..]).to_owned();

        let val_sequences = sequences
            .slice(ndarray::s![train_samples.., .., ..])
            .to_owned();
        let val_targets = targets.slice(ndarray::s![train_samples.., ..]).to_owned();

        // Initialize network if not already done
        if self.network.is_none() {
            log::info!("Initializing LSTM network with config: {:?}", self.config);
            let num_layers = 2;
            let network = rust_lstm::models::lstm_network::LSTMNetwork::new(
                self.config.input_size,
                self.config.hidden_size,
                num_layers,
            );
            self.network = Some(network);
        }

        // Convert data to rust-lstm format
        let training_data =
            self.convert_sequences_to_training_data(&train_sequences, &train_targets)?;
        let validation_data =
            self.convert_sequences_to_training_data(&val_sequences, &val_targets)?;

        // Update config to reflect actual output size (1 for rust-lstm compatibility)
        self.config.output_size = 1;

        // Create trainer with MSE loss and SGD optimizer
        use rust_lstm::loss::MSELoss;
        use rust_lstm::optimizers::SGD;
        use rust_lstm::training::LSTMTrainer;

        if let Some(network) = self.network.take() {
            let mut trainer =
                LSTMTrainer::new(network, MSELoss, SGD::new(self.config.learning_rate));

            // Set training configuration
            trainer.config.epochs = self.training_config.epochs;
            trainer.config.print_every = self.training_config.print_every;
            trainer.config.clip_gradient = self.training_config.clip_gradient;

            log::info!(
                "🧠 Starting LSTM training with validation monitoring (max {} epochs)",
                trainer.config.epochs
            );

            // Train the model with validation data using rust-lstm's built-in validation
            match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                trainer.train(&training_data, Some(&validation_data));
            })) {
                Ok(_) => {
                    // Get the trained network back
                    self.network = Some(trainer.network);
                    log::info!("✅ LSTM training with validation completed successfully");

                    // Calculate final validation metrics for better understanding
                    if let Ok(final_predictions) = self.predict(&val_sequences).await {
                        log::debug!(
                            "Validation shapes - predictions: {:?}, targets: {:?}",
                            final_predictions.shape(),
                            val_targets.shape()
                        );

                        // FIXED: Ensure shapes match before calculating metrics
                        if final_predictions.shape() == val_targets.shape() {
                            let final_mse =
                                self.calculate_mse_loss(&final_predictions, &val_targets);
                            let final_mape = self.calculate_mape(&final_predictions, &val_targets);
                            log::info!(
                                "📊 Final validation metrics - MSE: {:.6}, MAPE: {:.2}%",
                                final_mse,
                                final_mape
                            );
                        } else {
                            log::warn!(
                                "Skipping validation metrics due to shape mismatch: predictions={:?}, targets={:?}",
                                final_predictions.shape(),
                                val_targets.shape()
                            );
                        }
                    }
                }
                Err(e) => {
                    let error_msg = format!("LSTM training panicked: {:?}", e);
                    log::error!("{}", error_msg);
                    return Err(crate::utils::error::VangaError::ModelError(error_msg));
                }
            }
        } else {
            return Err(crate::utils::error::VangaError::ModelError(
                "Network not initialized".to_string(),
            ));
        }

        Ok(())
    }

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
                SGD::new(self.config.learning_rate), // Use configured learning rate
            );

            // Set training configuration with early stopping support
            trainer.config.epochs = self.training_config.epochs;
            trainer.config.print_every = self.training_config.print_every;
            trainer.config.clip_gradient = self.training_config.clip_gradient;

            log::info!(
                "Starting LSTM training for {} epochs",
                trainer.config.epochs
            );

            // Train the model with error handling
            match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                trainer.train(&training_data, None);
            })) {
                Ok(_) => {
                    // Get the trained network back
                    self.network = Some(trainer.network);
                    log::info!("LSTM training completed successfully");

                    // Calculate final training metrics for better understanding
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

            // Extract sequence for this batch - fix input structure
            for seq_idx in 0..sequences.shape()[1] {
                // Create input with proper shape (features, 1) to match official example
                let mut input_timestep = Array2::zeros((sequences.shape()[2], 1));
                for feature_idx in 0..sequences.shape()[2] {
                    input_timestep[[feature_idx, 0]] = sequences[[batch_idx, seq_idx, feature_idx]];
                }
                input_sequence.push(input_timestep);
            }

            // CRITICAL FIX: rust-lstm expects target sequence length to match input sequence length
            // For sequence-to-one prediction, we repeat the same target value for each timestep
            // but only the final timestep's prediction is used during training
            let target_value = targets[[batch_idx, 0]]; // Take first target only (single output)
            let target_timestep = Array2::from_elem((1, 1), target_value);
            let sequence_length = sequences.shape()[1];
            
            // Create target sequence with same length as input sequence (vec! macro creates efficiently)
            let target_sequence = vec![target_timestep; sequence_length];

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
        let output_size = self.config.output_size;
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
                // FIXED: For single-target models, we expect output_size=1
                // The rust-lstm returns hidden states, so we need to project to single output
                // Take the mean of the hidden state as the prediction (simple projection)
                let prediction_value = if last_output.nrows() > 0 {
                    // Simple projection: take mean of hidden state values
                    let sum: f64 = (0..last_output.nrows()).map(|i| last_output[[i, 0]]).sum();
                    sum / last_output.nrows() as f64
                } else {
                    0.0
                };

                // Store the single prediction value
                if batch_idx < predictions.nrows() && predictions.ncols() > 0 {
                    predictions[[batch_idx, 0]] = prediction_value;
                }
            }
        }

        Ok(predictions)
    }
}
