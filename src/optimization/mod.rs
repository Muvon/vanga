//! Auto-optimization system for VANGA LSTM
//!
//! This module provides automatic hyperparameter optimization, feature selection,
//! and model architecture tuning specifically designed for cryptocurrency forecasting.
//!
//! Key Features:
//! - Bayesian hyperparameter optimization
//! - Correlation-based feature selection
//! - Crypto-specific loss functions
//! - Data-driven architecture selection

pub mod auto_tuner;
pub mod feature_selection;
pub mod frac_adam;
#[cfg(test)]
mod frac_adam_test;
#[cfg(test)]
mod frac_integration_test;
pub mod frac_nadam;
#[cfg(test)]
mod frac_nadam_test;
pub mod fractional;
pub mod hyperparameter;
pub mod objective;
pub mod optimizer_selector;

// Re-export main optimization components
pub use auto_tuner::{
    BayesianOptimizer, SearchSpace as TunerSearchSpace, TrialConfig, TrialResult,
};
pub use feature_selection::{
    CorrelationMatrix, FeatureSelector, ImportanceMethod, ImportanceScores,
};
pub use frac_adam::{FracAdam, ParamsFracAdam};
pub use frac_nadam::{FracNAdam, ParamsFracNAdam};
pub use fractional::{FractionalConfig, FractionalDerivative};
pub use hyperparameter::{HyperparameterOptimizer, OptimizationMethod, SearchSpace};
pub use objective::{ObjectiveFunction, OptimizationMetric};
pub use optimizer_selector::{
    apply_optimizer_recommendation, recommend_optimizer_for_data, DataCharacteristics,
    MarketRegime, OptimizerRecommendation, OptimizerSelector, PerformanceExpectation,
};

use crate::utils::error::Result;
use polars::prelude::*;
use serde::{Deserialize, Serialize};

/// Main optimization coordinator that orchestrates all optimization processes
#[derive(Debug, Clone)]
pub struct AutoOptimizer {
    pub hyperparameter_optimizer: HyperparameterOptimizer,
    pub feature_selector: FeatureSelector,
    pub optimization: OptimizationConfig,
}

impl AutoOptimizer {
    /// Create new auto-optimizer with default configuration
    pub fn new() -> Self {
        Self {
            hyperparameter_optimizer: HyperparameterOptimizer::new(),
            feature_selector: FeatureSelector::new(),
            optimization: OptimizationConfig::default(),
        }
    }

    /// Create auto-optimizer with custom configuration
    pub fn with_config(config: crate::config::training::OptimizationConfig) -> Self {
        // Convert training config to internal optimization config
        let internal_config = OptimizationConfig {
            hyperparameter_config: HyperparameterConfig {
                method: match config.method {
                    crate::config::training::OptimizationMethod::Bayesian => {
                        hyperparameter::OptimizationMethod::Bayesian
                    }
                    crate::config::training::OptimizationMethod::Grid => {
                        hyperparameter::OptimizationMethod::Grid
                    }
                    crate::config::training::OptimizationMethod::Random => {
                        hyperparameter::OptimizationMethod::Random
                    }
                    crate::config::training::OptimizationMethod::None => {
                        hyperparameter::OptimizationMethod::Random
                    } // Default fallback
                },
                n_trials: config.n_trials,
                sequence_length_range: (10, 200),  // Default range
                hidden_units_range: (32, 512),     // Default range
                learning_rate_range: (1e-5, 1e-2), // Default range
                batch_size_options: vec![16, 32, 64, 128, 256], // Default options
            },
            feature_selection_config: FeatureSelectionConfig::default(),
            max_optimization_time: config.timeout_seconds.unwrap_or(3600),
            parallel_trials: 4,
            early_stopping_patience: 10,
        };

        log::info!(
            "Auto-optimizer configured: method={:?}, trials={}, timeout={}s",
            config.method,
            config.n_trials,
            config.timeout_seconds.unwrap_or(3600)
        );

        Self {
            hyperparameter_optimizer: HyperparameterOptimizer::with_config(&internal_config),
            feature_selector: FeatureSelector::with_config(&internal_config),
            optimization: internal_config,
        }
    }

    /// Perform complete optimization pipeline
    pub async fn optimize_complete_pipeline(
        &self,
        data: &DataFrame,
        symbol: &str,
    ) -> Result<OptimizationResult> {
        log::info!(
            "Starting complete optimization pipeline for symbol: {}",
            symbol
        );

        // Phase 1: Feature selection and correlation analysis
        let selected_features = self.feature_selector.select_optimal_features(data).await?;
        log::info!(
            "Selected {} features from {} available",
            selected_features.len(),
            data.width()
        );

        // Phase 2: Hyperparameter optimization
        let sequence_length = self
            .hyperparameter_optimizer
            .optimize_sequence_length(data)
            .await?;

        let architecture = self
            .hyperparameter_optimizer
            .optimize_architecture(data.height())
            .await?;

        let learning_schedule = self
            .hyperparameter_optimizer
            .optimize_learning_schedule(data)
            .await?;

        let batch_size = self
            .hyperparameter_optimizer
            .optimize_batch_size(8192)
            .await?; // 8GB memory limit default

        log::info!(
            "Optimization completed: seq_len={}, batch_size={}, features={}",
            sequence_length,
            batch_size,
            selected_features.len()
        );

        Ok(OptimizationResult {
            selected_features,
            sequence_length,
            architecture,
            learning_schedule,
            batch_size,
            optimization_score: 0.0, // Will be calculated during validation
        })
    }
}

impl Default for AutoOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

/// Complete optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationConfig {
    /// Hyperparameter optimization settings
    pub hyperparameter_config: HyperparameterConfig,

    /// Feature selection settings
    pub feature_selection_config: FeatureSelectionConfig,

    /// Maximum optimization time in seconds
    pub max_optimization_time: u64,

    /// Number of parallel optimization trials
    pub parallel_trials: usize,

    /// Early stopping patience
    pub early_stopping_patience: u32,
}

impl Default for OptimizationConfig {
    fn default() -> Self {
        Self {
            hyperparameter_config: HyperparameterConfig::default(),
            feature_selection_config: FeatureSelectionConfig::default(),
            max_optimization_time: 3600, // 1 hour
            parallel_trials: 4,
            early_stopping_patience: 10,
        }
    }
}

/// Hyperparameter optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HyperparameterConfig {
    pub method: OptimizationMethod,
    pub n_trials: u32,
    pub sequence_length_range: (u32, u32),
    pub hidden_units_range: (u32, u32),
    pub learning_rate_range: (f64, f64),
    pub batch_size_options: Vec<u32>,
}

impl Default for HyperparameterConfig {
    fn default() -> Self {
        Self {
            method: OptimizationMethod::Bayesian,
            n_trials: 100,
            sequence_length_range: (10, 200),
            hidden_units_range: (32, 512),
            learning_rate_range: (1e-5, 1e-2),
            batch_size_options: vec![16, 32, 64, 128, 256],
        }
    }
}

/// Feature selection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureSelectionConfig {
    pub correlation_threshold: f64,
    pub importance_method: ImportanceMethod,
    pub min_features: usize,
    pub max_features: usize,
    pub recursive_elimination_step: f64,
}

impl Default for FeatureSelectionConfig {
    fn default() -> Self {
        Self {
            correlation_threshold: 0.95,
            importance_method: ImportanceMethod::Correlation,
            min_features: 5,
            max_features: 50,
            recursive_elimination_step: 0.1,
        }
    }
}

/// Result of complete optimization process
#[derive(Debug, Clone)]
pub struct OptimizationResult {
    pub selected_features: Vec<String>,
    pub sequence_length: u32,
    pub architecture: ArchitectureConfig,
    pub learning_schedule: LearningSchedule,
    pub batch_size: u32,
    pub optimization_score: f64,
}

/// Architecture configuration from optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArchitectureConfig {
    pub hidden_units: u32,
    pub num_layers: u32,
    pub dropout_rate: f64,
    pub use_bidirectional: bool,
    pub activation: String,
}

impl Default for ArchitectureConfig {
    fn default() -> Self {
        Self {
            hidden_units: 128,
            num_layers: 2,
            dropout_rate: 0.2,
            use_bidirectional: false,
            activation: "tanh".to_string(),
        }
    }
}

/// Learning schedule configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningSchedule {
    pub initial_lr: f64,
    pub schedule_type: ScheduleType,
    pub warmup_steps: u32,
    pub decay_rate: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ScheduleType {
    Constant,
    LinearDecay,
    ExponentialDecay,
    CosineAnnealing,
    WarmRestarts,
}

impl Default for LearningSchedule {
    fn default() -> Self {
        Self {
            initial_lr: 1e-3,
            schedule_type: ScheduleType::CosineAnnealing,
            warmup_steps: 100,
            decay_rate: 0.95,
        }
    }
}
