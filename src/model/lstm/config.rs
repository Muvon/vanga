//! LSTM configuration types and validation
//!
//! This module contains configuration structs, enums, and validation
//! for LSTM model setup and training parameters.

use crate::config::model::{AttentionConfig, DropoutConfig};
use crate::model::attention::AttentionModule;
use crate::utils::error::{Result, VangaError};

use candle_core::Device;
use candle_nn::{
    optim::{self, Optimizer},
    Linear, VarMap, LSTM,
};
use serde::{Deserialize, Serialize};

/// Target format enumeration for metrics calculation
#[derive(Debug, Clone, Copy)]
pub enum TargetFormat {
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
pub struct TrainingConfig {
    pub epochs: usize,
    pub print_every: usize,
    pub clip_gradient: Option<f64>,
    pub batch_size: usize,
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
    pub config: LSTMConfig,
    pub lstm_layers: Option<Vec<LSTM>>, // Forward layers for unidirectional or bidirectional
    pub backward_lstm_layers: Option<Vec<LSTM>>, // Backward layers for bidirectional LSTM
    pub output_layer: Option<Linear>,
    pub attention_module: Option<Box<dyn AttentionModule>>,
    pub attention_config: Option<AttentionConfig>,
    pub use_attention: bool, // Public for testing
    pub device: Device,
    pub varmap: VarMap,
    pub training_config: TrainingConfig,
    pub trained: bool,
    /// Target context for this individual model (e.g., "price_level_1h", "direction_4h")
    /// This allows proper target type detection without assumptions
    pub target_context: Option<(String, crate::targets::TargetType)>, // (target_name, target_type)
    /// Global class weights calculated once from entire training dataset
    /// Used for consistent loss calculation across all batches (training and validation)

    /// Architecture configuration for bidirectional detection
    pub architecture: Option<crate::config::model::LSTMArchitecture>,
    /// Dropout configuration for regularization
    pub dropout_config: Option<DropoutConfig>,
    /// Stored validation data for consistent metrics calculation
    /// Used to ensure epoch metrics and final metrics use the same data
    pub stored_val_sequences: Option<ndarray::Array3<f64>>,
    pub stored_val_targets: Option<ndarray::Array2<f64>>,
    /// Stored test data for final evaluation - empty arrays if no test data
    /// Check sequences.shape()[0] > 0 to determine if test data is available
    pub stored_test_sequences: ndarray::Array3<f64>,
    pub stored_test_targets: ndarray::Array2<f64>,
    /// None if XGBoost is disabled in configuration
    pub xgboost_model: Option<crate::model::xgboost::XGBoostRegressor>,
    /// Best model weights saved during training (for early stopping)
    /// Stores the VarMap state when validation loss improves
    pub best_model_varmap: Option<VarMap>,
    /// Best validation loss achieved during training
    pub best_validation_loss: Option<f64>,
    /// Epoch at which best validation loss was achieved
    pub best_epoch: Option<usize>,
    /// Random seed for reproducible training
    /// None = random initialization, Some(0) = random, Some(>0) = reproducible
    pub seed: Option<u64>,
}

/// Serializable model state for persistence - SAME as original
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelState {
    pub config: LSTMConfig,
    pub epochs: usize,
    pub print_every: usize,
    pub clip_gradient: Option<f64>,
}

// Optimizer wrapper for concrete type handling with Candle
pub enum OptimizerWrapper {
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
    pub fn set_learning_rate(&mut self, lr: f64) {
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

    pub fn step(&mut self, grads: &candle_core::backprop::GradStore) -> candle_core::Result<()> {
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
