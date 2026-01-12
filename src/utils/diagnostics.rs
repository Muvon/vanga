//! Training diagnostics and logging utilities
//!
//! This module provides comprehensive diagnostic functions for training pipeline components,
//! including optimizer parameters, regularization analysis, data configuration, and capacity assessment.
//!
//! # Features
//! - Comprehensive optimizer parameter display for all 9 supported optimizers
//! - Weight decay analysis with threshold warnings
//! - Regularization diagnostics (dropout, L2)
//! - Data configuration diagnostics (samples, validation, batch size)
//! - Model capacity assessment for LSTM time series models
//! - Reusable across different training contexts
//!
//! # Usage
//! ```rust
//! use crate::utils::diagnostics::TrainingDiagnostics;
//! use crate::config::training::{TrainingConfig, OptimizerType};
//!
//! // Display optimizer configuration
//! TrainingDiagnostics::log_optimizer_config(&config.training.optimizer, config.training.learning_rate);
//!
//! // Display regularization settings
//! TrainingDiagnostics::log_regularization_config(true, Some(0.25));
//!
//! // Display data configuration
//! TrainingDiagnostics::log_data_config(1000, 200, 32, true);
//!
//! // Display capacity assessment
//! TrainingDiagnostics::log_capacity_assessment(1000, 60, 50, 100000);
//! ```

use crate::config::training::OptimizerType;

/// Comprehensive diagnostics for training pipeline components
///
/// This struct provides static methods for logging detailed information about:
/// - Optimizer parameters and configuration
/// - Regularization settings (dropout, weight decay)
/// - Data configuration (samples, validation, batching)
/// - Model capacity assessment for time series data
pub struct TrainingDiagnostics;

impl TrainingDiagnostics {
    /// Log comprehensive optimizer configuration with parameter display
    ///
    /// Displays the actual optimizer type and all its specific parameters,
    /// including weight decay analysis and threshold warnings.
    ///
    /// # Arguments
    /// * `optimizer` - The optimizer configuration to analyze
    /// * `learning_rate` - The base learning rate
    ///
    /// # Example Output
    /// ```text
    /// ⚙️ OPTIMIZER DIAGNOSTICS:
    ///    📈 Learning rate: 0.001000
    ///    🔧 Optimizer: AdamW
    ///    🏋️ Weight decay: 0.0100 (L2 regularization strength)
    ///    📊 AdamW params: β1=0.900, β2=0.999, ε=1.00e-8
    /// ```
    pub fn log_optimizer_config(optimizer: &OptimizerType, learning_rate: f64) {
        log::info!("⚙️ OPTIMIZER DIAGNOSTICS:");
        log::info!("   📈 Learning rate: {:.6}", learning_rate);

        match optimizer {
            OptimizerType::AdamW {
                weight_decay,
                beta1,
                beta2,
                eps,
            } => {
                log::info!("   🔧 Optimizer: AdamW");
                log::info!(
                    "   🏋️ Weight decay: {:.4} (L2 regularization strength)",
                    weight_decay
                );
                log::info!(
                    "   📊 AdamW params: β1={:.3}, β2={:.3}, ε={:.2e}",
                    beta1,
                    beta2,
                    eps
                );

                Self::log_weight_decay_warnings(*weight_decay);
            }
            OptimizerType::SGD { momentum } => {
                log::info!("   🔧 Optimizer: SGD");
                if let Some(momentum_val) = momentum {
                    log::info!("   🚀 Momentum: {:.3}", momentum_val);
                } else {
                    log::info!("   🚀 Momentum: DISABLED");
                }
                log::info!("   🏋️ Weight decay: N/A (SGD doesn't support weight decay)");
                log::warn!("   ⚠️ No weight decay regularization - consider AdamW for better regularization");
            }
            OptimizerType::Adam {
                beta1,
                beta2,
                eps,
                weight_decay,
                amsgrad,
            } => {
                log::info!("   🔧 Optimizer: Adam");
                log::info!(
                    "   📊 Adam params: β1={:.3}, β2={:.3}, ε={:.2e}, AMSGrad={}",
                    beta1,
                    beta2,
                    eps,
                    amsgrad
                );
                Self::log_optional_weight_decay(weight_decay);
            }
            OptimizerType::AdaDelta {
                rho,
                eps,
                weight_decay,
            } => {
                log::info!("   🔧 Optimizer: AdaDelta");
                log::info!("   📊 AdaDelta params: ρ={:.3}, ε={:.2e}", rho, eps);
                Self::log_optional_weight_decay(weight_decay);
            }
            OptimizerType::AdaGrad {
                lr_decay,
                weight_decay,
                initial_accumulator_value,
                eps,
            } => {
                log::info!("   🔧 Optimizer: AdaGrad");
                log::info!(
                    "   📊 AdaGrad params: lr_decay={:.4}, init_acc={:.4}, ε={:.2e}",
                    lr_decay,
                    initial_accumulator_value,
                    eps
                );
                Self::log_optional_weight_decay(weight_decay);
            }
            OptimizerType::AdaMax {
                beta1,
                beta2,
                eps,
                weight_decay,
            } => {
                log::info!("   🔧 Optimizer: AdaMax");
                log::info!(
                    "   📊 AdaMax params: β1={:.3}, β2={:.3}, ε={:.2e}",
                    beta1,
                    beta2,
                    eps
                );
                Self::log_optional_weight_decay(weight_decay);
            }
            OptimizerType::NAdam {
                beta1,
                beta2,
                eps,
                weight_decay,
                momentum_decay,
            } => {
                log::info!("   🔧 Optimizer: NAdam");
                log::info!(
                    "   📊 NAdam params: β1={:.3}, β2={:.3}, ε={:.2e}, momentum_decay={:.4}",
                    beta1,
                    beta2,
                    eps,
                    momentum_decay
                );
                Self::log_optional_weight_decay(weight_decay);
            }
            OptimizerType::RAdam {
                beta1,
                beta2,
                eps,
                weight_decay,
            } => {
                log::info!("   🔧 Optimizer: RAdam");
                log::info!(
                    "   📊 RAdam params: β1={:.3}, β2={:.3}, ε={:.2e}",
                    beta1,
                    beta2,
                    eps
                );
                Self::log_optional_weight_decay(weight_decay);
            }
            OptimizerType::RMSprop {
                alpha,
                eps,
                weight_decay,
                momentum,
                centered,
            } => {
                log::info!("   🔧 Optimizer: RMSprop");
                log::info!(
                    "   📊 RMSprop params: α={:.4}, ε={:.2e}, momentum={:.3}, centered={}",
                    alpha,
                    eps,
                    momentum,
                    centered
                );
                Self::log_optional_weight_decay(weight_decay);
            }
            OptimizerType::FracAdam {
                beta1,
                beta2,
                eps,
                weight_decay,
                alpha,
                memory_window,
                step_size,
            } => {
                log::info!("   🔧 Optimizer: FracAdam (Fractional Adam)");
                log::info!(
                    "   📊 FracAdam params: β1={:.3}, β2={:.3}, ε={:.2e}",
                    beta1,
                    beta2,
                    eps
                );
                log::info!(
                    "   🧮 Fractional params: α={:.3} (memory strength), window={} steps, step_size={:.3}",
                    alpha,
                    memory_window,
                    step_size
                );
                Self::log_optional_weight_decay(weight_decay);
                log::info!("   💡 FracAdam uses long-term memory effects for better time-series performance");
            }
            OptimizerType::FracNAdam {
                beta1,
                beta2,
                eps,
                weight_decay,
                momentum_decay,
                alpha,
                memory_window,
                step_size,
            } => {
                log::info!("   🔧 Optimizer: FracNAdam (Fractional NAdam with Nesterov)");
                log::info!(
                    "   📊 FracNAdam params: β1={:.3}, β2={:.3}, ε={:.2e}, momentum_decay={:.4}",
                    beta1,
                    beta2,
                    eps,
                    momentum_decay
                );
                log::info!(
                    "   🧮 Fractional params: α={:.3} (memory strength), window={} steps, step_size={:.3}",
                    alpha,
                    memory_window,
                    step_size
                );
                Self::log_optional_weight_decay(weight_decay);
                log::info!("   💡 FracNAdam combines Nesterov acceleration with fractional memory for fast convergence");
            }
            OptimizerType::Prodigy {
                d_coef,
                growth_rate,
                beta1,
                beta2,
                eps,
                weight_decay,
                safeguard_warmup,
            } => {
                log::info!("   🔧 Optimizer: Prodigy (ICLR 2024) - Learning-Rate-Free!");
                log::info!(
                    "   🚀 Automatic LR adaptation: lr={:.1} (will auto-adjust)",
                    learning_rate
                );
                log::info!(
                    "   📊 D estimate: coef={:.3}, growth_rate={}",
                    d_coef,
                    if growth_rate.is_infinite() {
                        "unlimited".to_string()
                    } else {
                        format!("{:.2}", growth_rate)
                    }
                );
                log::info!(
                    "   📊 Adam-like params: β1={:.3}, β2={:.3}, ε={:.2e}",
                    beta1,
                    beta2,
                    eps
                );
                log::info!("   🏋️ Weight decay: {:.4}", weight_decay);
                log::info!("   🔥 Safeguard warmup: {}", safeguard_warmup);
                log::info!(
                    "   💡 Prodigy automatically finds optimal learning rate - no tuning needed!"
                );
                log::info!("   📄 Paper: https://arxiv.org/abs/2306.06101");
            }
            OptimizerType::FracProdigy {
                beta1,
                beta2,
                eps,
                weight_decay,
                momentum_decay,
                d_coef,
                growth_rate,
                alpha,
                memory_window,
                step_size,
            } => {
                log::info!("   🔧 Optimizer: FracProdigy - Fractional Memory + Automatic LR!");
                log::info!(
                    "   🚀 Automatic LR adaptation: lr={:.1} (will auto-adjust)",
                    learning_rate
                );

                // Validate fractional parameters
                if *alpha <= 0.0 || *alpha > 1.0 {
                    log::error!(
                        "   ❌ Invalid fractional order α={:.2} (must be in (0, 1])",
                        alpha
                    );
                }
                if *memory_window == 0 || *memory_window > 200 {
                    log::warn!(
                        "   ⚠️ Memory window {} is outside recommended range [1, 200]",
                        memory_window
                    );
                }
                if *d_coef <= 0.0 {
                    log::error!("   ❌ Invalid d_coef={:.3} (must be > 0)", d_coef);
                }
                if *growth_rate <= 0.0 && growth_rate.is_finite() {
                    log::error!(
                        "   ❌ Invalid growth_rate={:.3} (must be > 0 or infinite)",
                        growth_rate
                    );
                }

                // Warn if learning_rate != 1.0 for Prodigy-based optimizer
                if (learning_rate - 1.0).abs() > 0.01 {
                    log::warn!(
                        "   ⚠️ Learning rate {:.3} != 1.0 - Prodigy works best with lr=1.0",
                        learning_rate
                    );
                }

                log::info!(
                    "   🧮 Fractional memory: α={:.2}, window={}, step={:.1}",
                    alpha,
                    memory_window,
                    step_size
                );
                log::info!(
                    "   📊 Prodigy D-estimate: coef={:.3}, growth_rate={}",
                    d_coef,
                    if growth_rate.is_infinite() {
                        "unlimited".to_string()
                    } else {
                        format!("{:.2}", growth_rate)
                    }
                );
                log::info!(
                    "   📊 NAdam params: β1={:.3}, β2={:.3}, momentum_decay={:.4}, ε={:.2e}",
                    beta1,
                    beta2,
                    momentum_decay,
                    eps
                );

                if let Some(wd) = weight_decay {
                    log::info!("   🏋️ Weight decay: {:.4}", wd);
                    Self::log_weight_decay_warnings(*wd);
                } else {
                    log::info!("   🏋️ Weight decay: DISABLED");
                }

                // Memory usage estimate
                let estimated_tensors = memory_window * 2; // First + second moments
                log::info!(
                    "   💾 Estimated memory: ~{} gradient tensors in history",
                    estimated_tensors
                );

                log::info!(
                    "   💡 FracProdigy combines fractional memory with automatic LR - best of both worlds!"
                );
                log::info!("   📄 Combines: FracNAdam memory + Prodigy automatic LR");
            }
        }
    }

    /// Log weight decay warnings for mandatory weight decay (AdamW)
    fn log_weight_decay_warnings(weight_decay: f64) {
        if weight_decay < 0.001 {
            log::warn!(
                "   ⚠️ Weight decay ({:.4}) is very low - may not prevent overfitting",
                weight_decay
            );
        } else if weight_decay > 0.1 {
            log::warn!(
                "   ⚠️ Weight decay ({:.4}) is very high - may cause underfitting",
                weight_decay
            );
        }
    }

    /// Log optional weight decay configuration and warnings
    fn log_optional_weight_decay(weight_decay: &Option<f64>) {
        if let Some(wd) = weight_decay {
            log::info!("   🏋️ Weight decay: {:.4}", wd);
            Self::log_weight_decay_warnings(*wd);
        } else {
            log::info!("   🏋️ Weight decay: DISABLED");
            log::warn!("   ⚠️ No weight decay regularization - overfitting risk increased");
        }
    }

    /// Log regularization configuration diagnostics
    ///
    /// # Arguments
    /// * `dropout_enabled` - Whether dropout is enabled
    /// * `dropout_rate` - Optional dropout rate if enabled
    ///
    /// # Example Output
    /// ```text
    /// 🛡️ REGULARIZATION DIAGNOSTICS:
    ///    💧 Dropout enabled: true
    ///    💧 Dropout rate: 0.25
    /// ```
    pub fn log_regularization_config(dropout_enabled: bool, dropout_rate: Option<f64>) {
        log::info!("🛡️ REGULARIZATION DIAGNOSTICS:");
        if dropout_enabled {
            log::info!("   💧 Dropout enabled: {}", dropout_enabled);
            if let Some(rate) = dropout_rate {
                log::info!("   💧 Dropout rate: {:.2}", rate);
            }
        } else {
            log::info!("   💧 Dropout: DISABLED");
        }
    }

    /// Log data configuration diagnostics
    ///
    /// # Arguments
    /// * `total_train_samples` - Number of training samples
    /// * `total_val_samples` - Number of validation samples (0 if disabled)
    /// * `batch_size` - Training batch size
    /// * `use_validation` - Whether validation is enabled
    ///
    /// # Example Output
    /// ```text
    /// 📊 DATA DIAGNOSTICS:
    ///    🎯 Training samples: 1000
    ///    ✅ Validation samples: 200
    ///    📊 Validation ratio: 20.0% (of training data)
    ///    📦 Batch size: 32
    /// ```
    pub fn log_data_config(
        total_train_samples: usize,
        total_val_samples: usize,
        batch_size: usize,
        use_validation: bool,
    ) {
        log::info!("📊 DATA DIAGNOSTICS:");
        log::info!("   🎯 Training samples: {}", total_train_samples);

        if use_validation {
            log::info!("   ✅ Validation samples: {}", total_val_samples);
            // Validation ratio is calculated as % of training data (after test split)
            let val_ratio = total_val_samples as f64 / total_train_samples as f64;
            log::info!(
                "   📊 Validation ratio: {:.1}% (of training data)",
                val_ratio * 100.0
            );
        } else {
            log::info!("   ❌ Validation: DISABLED");
        }

        log::info!("   📦 Batch size: {}", batch_size);
    }

    /// Log LSTM model capacity assessment for time series data
    ///
    /// # Arguments
    /// * `total_train_samples` - Number of training sequences
    /// * `sequence_length` - Length of each sequence
    /// * `num_features` - Number of features per timestep
    /// * `total_params` - Total model parameters
    ///
    /// # Example Output
    /// ```text
    /// 🧮 LSTM TIME SERIES CAPACITY ASSESSMENT:
    ///    📊 Training sequences: 1000
    ///    📏 Sequence length: 60
    ///    🔢 Features per timestep: 50
    ///    📈 Effective data points: 3000000 (1000 × 60 × 50)
    ///    🧮 Model parameters: 50000
    ///    📊 Data points per parameter: 60.0
    /// ```
    pub fn log_capacity_assessment(
        total_train_samples: usize,
        sequence_length: usize,
        num_features: usize,
        total_params: usize,
    ) {
        let effective_data_points = total_train_samples * sequence_length * num_features;
        let samples_per_param = if total_params > 0 {
            effective_data_points as f64 / total_params as f64
        } else {
            f64::INFINITY
        };

        log::info!("🧮 LSTM TIME SERIES CAPACITY ASSESSMENT:");
        log::info!("   📊 Training sequences: {}", total_train_samples);
        log::info!("   📏 Sequence length: {}", sequence_length);
        log::info!("   🔢 Features per timestep: {}", num_features);
        log::info!(
            "   📈 Effective data points: {} ({} × {} × {})",
            effective_data_points,
            total_train_samples,
            sequence_length,
            num_features
        );
        log::info!("   🧮 Model parameters: {}", total_params);
        log::info!("   📊 Data points per parameter: {:.1}", samples_per_param);

        // LSTM-specific capacity assessment (different from traditional ML)
        if total_params == 0 {
            log::error!("   🚨 CRITICAL: Model has 0 parameters! Configuration error!");
        } else if total_train_samples == 0 {
            log::error!("   🚨 CRITICAL: No training sequences! Data loading error!");
        } else if samples_per_param < 1.0 {
            log::warn!(
                "   ⚠️ LOW data density: {:.1} data points per parameter",
                samples_per_param
            );
            log::warn!("   💡 Consider: More sequences, longer sequences, or smaller model");
        } else if samples_per_param < 10.0 {
            log::warn!(
                "   ⚠️ MODERATE data density: {:.1} data points per parameter",
                samples_per_param
            );
            log::warn!("   💡 Consider: More training data or regularization techniques");
        } else if samples_per_param > 1000.0 {
            log::info!(
                "   ✅ EXCELLENT data density: {:.1} data points per parameter",
                samples_per_param
            );
            log::info!("   💡 Consider: Larger model capacity for better performance");
        } else {
            log::info!(
                "   ✅ GOOD data density: {:.1} data points per parameter",
                samples_per_param
            );
        }
    }

    /// Get optimizer name as string for external use
    ///
    /// # Arguments
    /// * `optimizer` - The optimizer configuration
    ///
    /// # Returns
    /// String representation of the optimizer name
    ///
    /// # Example
    /// ```rust
    /// let name = OptimizerDiagnostics::get_optimizer_name(&optimizer);
    /// assert_eq!(name, "AdamW");
    /// ```
    pub fn get_optimizer_name(optimizer: &OptimizerType) -> &'static str {
        match optimizer {
            OptimizerType::AdamW { .. } => "AdamW",
            OptimizerType::SGD { .. } => "SGD",
            OptimizerType::Adam { .. } => "Adam",
            OptimizerType::AdaDelta { .. } => "AdaDelta",
            OptimizerType::AdaGrad { .. } => "AdaGrad",
            OptimizerType::AdaMax { .. } => "AdaMax",
            OptimizerType::NAdam { .. } => "NAdam",
            OptimizerType::RAdam { .. } => "RAdam",
            OptimizerType::RMSprop { .. } => "RMSprop",
            OptimizerType::FracAdam { .. } => "FracAdam",
            OptimizerType::FracNAdam { .. } => "FracNAdam",
            OptimizerType::Prodigy { .. } => "Prodigy",
            OptimizerType::FracProdigy { .. } => "FracProdigy",
        }
    }

    /// Check if optimizer supports weight decay
    ///
    /// # Arguments
    /// * `optimizer` - The optimizer configuration
    ///
    /// # Returns
    /// True if optimizer supports weight decay, false otherwise
    pub fn supports_weight_decay(optimizer: &OptimizerType) -> bool {
        !matches!(optimizer, OptimizerType::SGD { .. })
    }

    /// Get weight decay value if supported and enabled
    ///
    /// # Arguments
    /// * `optimizer` - The optimizer configuration
    ///
    /// # Returns
    /// Some(weight_decay) if supported and enabled, None otherwise
    pub fn get_weight_decay(optimizer: &OptimizerType) -> Option<f64> {
        match optimizer {
            OptimizerType::AdamW { weight_decay, .. } => Some(*weight_decay),
            OptimizerType::SGD { .. } => None,
            OptimizerType::Adam { weight_decay, .. } => *weight_decay,
            OptimizerType::AdaDelta { weight_decay, .. } => *weight_decay,
            OptimizerType::AdaGrad { weight_decay, .. } => *weight_decay,
            OptimizerType::AdaMax { weight_decay, .. } => *weight_decay,
            OptimizerType::NAdam { weight_decay, .. } => *weight_decay,
            OptimizerType::RAdam { weight_decay, .. } => *weight_decay,
            OptimizerType::RMSprop { weight_decay, .. } => *weight_decay,
            OptimizerType::FracAdam { weight_decay, .. } => *weight_decay,
            OptimizerType::FracNAdam { weight_decay, .. } => *weight_decay,
            OptimizerType::Prodigy { weight_decay, .. } => Some(*weight_decay),
            OptimizerType::FracProdigy { weight_decay, .. } => *weight_decay,
        }
    }
}
