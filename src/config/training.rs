use crate::config::{FeatureConfig, ModelConfig};
use crate::utils::error::{Result, VangaError};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Complete configuration for VANGA training pipeline
/// This is a coordinator that loads and manages all configuration sections
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    /// Trading symbol (e.g., BTCUSDT)
    pub symbol: String,

    /// Path to training data CSV file
    pub data_path: PathBuf,

    /// Whether to start fresh training (ignore existing model)
    pub fresh_training: bool,

    /// Whether to continue training existing model
    pub continue_training: bool,

    /// Prediction horizons to train for
    pub horizons: Vec<String>,

    /// Feature engineering configuration
    pub features: FeatureConfig,

    /// Model architecture configuration
    pub model: ModelConfig,

    /// Training hyperparameters
    pub training: TrainingParams,

    /// Data preprocessing configuration
    pub data: DataConfig,

    /// Optimization configuration
    pub optimization: OptimizationConfig,

    /// Target configuration (enable/disable targets)
    pub targets: TargetsConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingParams {
    /// Device to use for training
    pub device: DeviceConfig,
    /// Maximum number of epochs ("auto" for early stopping)
    pub epochs: EpochConfig,

    /// Batch size ("auto" for optimization)
    pub batch_size: BatchSizeConfig,

    /// Base learning rate (e.g., 0.001, 1e-5)
    pub learning_rate: f64,

    /// Optimizer type selection
    pub optimizer: OptimizerType,

    /// Warmup epochs for gradual learning rate increase
    pub warmup_epochs: u32,

    /// Learning rate schedule configuration
    pub learning_schedule: Option<LearningScheduleConfig>,

    /// Validation split ratio
    pub validation_split: f64,

    /// Gap between training and validation samples (e.g., "1h", "30m", "0")
    /// Prevents information leakage from features with lookback periods
    pub validation_gap: String,

    /// Test split ratio
    pub test_split: f64,

    /// Early stopping configuration
    pub early_stopping: EarlyStoppingConfig,

    /// Gradient clipping threshold
    pub gradient_clip: Option<f64>,

    /// Print training progress every N epochs (1 = every epoch, 10 = every 10 epochs)
    pub print_every: u32,

    /// Class weight strategy for handling imbalanced datasets

    /// Window-based learning rate decay for walk-forward training
    /// 0.8 = 20% reduction per window, 1.0 = no decay
    #[serde(default = "default_window_decay")]
    pub window_decay: f64,

    /// Minimum training data ratio for initial window (0.3-0.6 range recommended)
    /// 0.4 = start with 40% of available data, 0.5 = start with 50%
    #[serde(default = "default_min_train_ratio")]
    pub min_train_ratio: f64,

    /// Minimum increment ratio for subsequent windows (0.2-0.5 range recommended)
    /// 0.3 = each window must add at least 30% more data than previous window
    /// This prevents overfitting by ensuring sufficient new information per window
    #[serde(default = "default_min_increment_ratio")]
    pub min_increment_ratio: f64,

    /// Random seed for reproducible training
    /// 0 = random initialization (different weights each run)
    /// >0 = reproducible initialization (same weights each run)
    #[serde(default = "default_seed")]
    pub seed: u64,
}

/// Strategy for calculating class weights in imbalanced datasets
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub enum DeviceConfig {
    #[default]
    Auto,
    CPU,
    GPU(usize),
    Metal(usize),
}

impl DeviceConfig {
    /// Convert DeviceConfig to string format expected by DeviceManager
    pub fn to_device_string(&self) -> String {
        match self {
            DeviceConfig::Auto => "auto".to_string(),
            DeviceConfig::CPU => "cpu".to_string(),
            DeviceConfig::GPU(index) => format!("gpu:{}", index),
            DeviceConfig::Metal(index) => format!("metal:{}", index),
        }
    }
}

impl std::fmt::Display for DeviceConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_device_string())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EarlyStoppingConfig {
    /// Number of epochs to wait for improvement before stopping
    pub patience: u32,
    /// Minimum improvement required to reset patience counter
    pub min_delta: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataConfig {
    /// Normalization method
    pub normalization: NormalizationMethod,

    /// Sequence overlap ratio
    pub sequence_overlap: f64,

    /// Outlier detection and handling
    pub outlier_handling: OutlierHandling,

    /// Feature selection configuration
    pub feature_selection: FeatureSelectionConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationConfig {
    /// Hyperparameter optimization method
    pub method: OptimizationMethod,

    /// Number of optimization trials
    pub n_trials: u32,

    /// Optimization timeout in seconds
    pub timeout_seconds: Option<u64>,

    /// Optimization metric to maximize
    pub metric: OptimizationMetric,
}
#[derive(Eq, Ord, PartialEq, PartialOrd, Debug, Clone, Serialize, Deserialize)]
pub enum EpochConfig {
    Auto { max_epochs: u32 },
    Fixed(u32),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BatchSizeConfig {
    Auto { min_size: u32, max_size: u32 },
    Fixed(u32),
}

/// Default window decay (no decay)
fn default_window_decay() -> f64 {
    1.0
}

/// Default minimum training ratio (40% for efficiency)
fn default_min_train_ratio() -> f64 {
    0.4
}

/// Default minimum increment ratio (30% for sufficient new data)
fn default_min_increment_ratio() -> f64 {
    0.3
}

/// Default seed (0 = random initialization)
fn default_seed() -> u64 {
    0
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizerType {
    SGD {
        momentum: Option<f64>,
    },
    AdamW {
        weight_decay: f64,
        beta1: f64,
        beta2: f64,
        eps: f64,
    },
    // New optimizers from candle-optimisers crate
    Adam {
        beta1: f64,
        beta2: f64,
        eps: f64,
        weight_decay: Option<f64>,
        amsgrad: bool,
    },
    AdaDelta {
        rho: f64,
        eps: f64,
        weight_decay: Option<f64>,
    },
    AdaGrad {
        lr_decay: f64,
        weight_decay: Option<f64>,
        initial_accumulator_value: f64,
        eps: f64,
    },
    AdaMax {
        beta1: f64,
        beta2: f64,
        eps: f64,
        weight_decay: Option<f64>,
    },
    NAdam {
        beta1: f64,
        beta2: f64,
        eps: f64,
        weight_decay: Option<f64>,
        momentum_decay: f64,
    },
    RAdam {
        beta1: f64,
        beta2: f64,
        eps: f64,
        weight_decay: Option<f64>,
    },
    RMSprop {
        alpha: f64,
        eps: f64,
        weight_decay: Option<f64>,
        momentum: f64,
        centered: bool,
    },
    // Fractional optimizers with long-term memory effects
    FracAdam {
        beta1: f64,
        beta2: f64,
        eps: f64,
        weight_decay: Option<f64>,
        alpha: f64,           // Fractional order (0 < α ≤ 1)
        memory_window: usize, // Memory window size (30-90 recommended)
        step_size: f64,       // Discretization step size (typically 1.0)
    },
    FracNAdam {
        beta1: f64,
        beta2: f64,
        eps: f64,
        weight_decay: Option<f64>,
        momentum_decay: f64,
        alpha: f64,           // Fractional order (0 < α ≤ 1)
        memory_window: usize, // Memory window size (30-90 recommended)
        step_size: f64,       // Discretization step size (typically 1.0)
    },
    /// Prodigy: Learning-rate-free optimizer (ICLR 2024)
    /// Set learning_rate = 1.0 in config, Prodigy handles automatic adaptation
    Prodigy {
        d_coef: f64,            // D estimate coefficient (default: 1.0)
        growth_rate: f64,       // Max D growth rate (default: inf)
        beta1: f64,             // First moment decay (default: 0.9)
        beta2: f64,             // Second moment decay (default: 0.999)
        eps: f64,               // Numerical stability (default: 1e-8)
        weight_decay: f64,      // L2 regularization (default: 0.0)
        safeguard_warmup: bool, // Enable warmup (default: false)
    },
    /// FracProdigy: Fractional Prodigy with long-term memory
    /// Combines fractional derivatives (FracNAdam) with automatic LR (Prodigy)
    /// Set learning_rate = 1.0 in config, Prodigy handles automatic adaptation
    FracProdigy {
        beta1: f64,                // First moment decay (default: 0.9)
        beta2: f64,                // Second moment decay (default: 0.999)
        eps: f64,                  // Numerical stability (default: 1e-8)
        weight_decay: Option<f64>, // L2 regularization (default: None)
        momentum_decay: f64,       // NAdam momentum decay (default: 0.004)
        d_coef: f64,               // D estimate coefficient (default: 1.0)
        growth_rate: f64,          // Max D growth rate (default: inf)
        alpha: f64,                // Fractional order (0 < α ≤ 1, default: 0.5)
        memory_window: usize,      // Memory window size (default: 20)
        step_size: f64,            // Discretization step size (default: 1.0)
    },
}

impl OptimizerType {
    /// Get default parameters for each optimizer type
    pub fn default_for_type(optimizer_name: &str) -> Self {
        match optimizer_name {
            "SGD" => OptimizerType::SGD {
                momentum: Some(0.9),
            },
            "AdamW" => OptimizerType::AdamW {
                weight_decay: 0.01,
                beta1: 0.9,
                beta2: 0.999,
                eps: 1e-8,
            },
            "Adam" => OptimizerType::Adam {
                beta1: 0.9,
                beta2: 0.999,
                eps: 1e-8,
                weight_decay: None,
                amsgrad: false,
            },
            "AdaDelta" => OptimizerType::AdaDelta {
                rho: 0.9,
                eps: 1e-6,
                weight_decay: None,
            },
            "AdaGrad" => OptimizerType::AdaGrad {
                lr_decay: 0.0,
                weight_decay: None,
                initial_accumulator_value: 0.0,
                eps: 1e-10,
            },
            "AdaMax" => OptimizerType::AdaMax {
                beta1: 0.9,
                beta2: 0.999,
                eps: 1e-8,
                weight_decay: None,
            },
            "NAdam" => OptimizerType::NAdam {
                beta1: 0.9,
                beta2: 0.999,
                eps: 1e-8,
                weight_decay: None,
                momentum_decay: 0.004,
            },
            "RAdam" => OptimizerType::RAdam {
                beta1: 0.9,
                beta2: 0.999,
                eps: 1e-8,
                weight_decay: None,
            },
            "RMSprop" => OptimizerType::RMSprop {
                alpha: 0.99,
                eps: 1e-8,
                weight_decay: None,
                momentum: 0.0,
                centered: false,
            },
            "FracAdam" => OptimizerType::FracAdam {
                beta1: 0.9,
                beta2: 0.999,
                eps: 1e-8,
                weight_decay: None,
                alpha: 0.9,        // Good balance between memory and stability
                memory_window: 60, // Moderate memory window
                step_size: 1.0,    // Standard discrete step
            },
            "FracNAdam" => OptimizerType::FracNAdam {
                beta1: 0.9,
                beta2: 0.999,
                eps: 1e-8,
                weight_decay: None,
                momentum_decay: 0.004,
                alpha: 0.9,        // Good balance between memory and stability
                memory_window: 60, // Moderate memory window
                step_size: 1.0,    // Standard discrete step
            },
            "Prodigy" => OptimizerType::Prodigy {
                d_coef: 1.0,
                growth_rate: f64::INFINITY,
                beta1: 0.9,
                beta2: 0.999,
                eps: 1e-8,
                weight_decay: 0.0,
                safeguard_warmup: false,
            },
            "FracProdigy" => OptimizerType::FracProdigy {
                beta1: 0.9,
                beta2: 0.999,
                eps: 1e-8,
                weight_decay: None,
                momentum_decay: 0.004,
                d_coef: 1.0,
                growth_rate: f64::INFINITY,
                alpha: 0.5,        // Balanced memory (less aggressive than FracNAdam)
                memory_window: 20, // Efficient memory window
                step_size: 1.0,    // Standard discrete step
            },
            _ => OptimizerType::AdamW {
                weight_decay: 0.01,
                beta1: 0.9,
                beta2: 0.999,
                eps: 1e-8,
            }, // Default to AdamW
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LearningScheduleConfig {
    /// No learning rate scheduling (constant rate)
    Constant,
    /// Reduce learning rate when validation loss plateaus (formerly "Adaptive")
    ReduceOnPlateau {
        patience: u32,
        factor: f64,
        min_lr: Option<f64>,
        /// Metric to monitor: "loss", "accuracy", "f1_score"
        monitor: Option<String>,
        /// Threshold for significant improvement
        threshold: Option<f64>,
    },
    /// Linear decay over training epochs
    LinearDecay {
        decay_rate: f64,
        min_lr: Option<f64>,
    },
    /// Exponential decay over training epochs
    ExponentialDecay {
        gamma: f64, // Renamed from decay_rate for clarity
        min_lr: Option<f64>,
    },
    /// Step decay at specific milestones
    StepDecay {
        step_size: u32,
        gamma: f64,
        milestones: Option<Vec<u32>>,
        min_lr: Option<f64>,
    },
    /// Polynomial decay with configurable power
    PolynomialDecay { power: f64, min_lr: Option<f64> },
    /// Cosine annealing schedule with proper eta_min
    CosineAnnealing { t_max: u32, eta_min: Option<f64> },
    /// Warm restarts with cosine annealing (SGDR)
    WarmRestarts {
        t_0: u32,
        t_mult: u32,
        eta_min: Option<f64>,
    },
    /// One Cycle Learning Rate (super-convergence)
    OneCycle {
        max_lr: f64,
        pct_start: Option<f64>, // Percentage of cycle spent increasing LR
        anneal_strategy: Option<String>, // "cos" or "linear"
        div_factor: Option<f64>, // initial_lr = max_lr / div_factor
        final_div_factor: Option<f64>, // final_lr = initial_lr / final_div_factor
    },
    /// Cyclical Learning Rate with different policies
    CyclicalLR {
        base_lr: f64,
        max_lr: f64,
        step_size_up: u32,
        step_size_down: Option<u32>,
        mode: Option<String>, // "triangular", "triangular2", "exp_range"
        gamma: Option<f64>,   // For exp_range mode
    },
    /// Noam scheduler (Transformer-style)
    NoamLR {
        model_size: u32,
        warmup_steps: u32,
        factor: Option<f64>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NormalizationMethod {
    Robust,
    MinMax,
    Standard,
    Quantile,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MissingDataStrategy {
    Interpolate,
    Drop,
    ForwardFill,
    BackwardFill,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutlierHandling {
    pub enabled: bool,
    pub method: OutlierMethod,
    pub threshold: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OutlierMethod {
    IQR,
    ZScore,
    ModifiedZScore,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureSelectionConfig {
    pub enabled: bool,
    pub max_features: Option<usize>,
    pub correlation_threshold: f64,
    pub importance_threshold: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum OptimizationMethod {
    Bayesian,
    Grid,
    Random,
    None,
}

// Re-export OptimizationMetric from objective module to avoid duplication
pub use crate::optimization::objective::OptimizationMetric;

impl TrainingParams {
    /// Validate training parameters for correctness
    pub fn validate(&self) -> Result<()> {
        // Validate validation split
        if self.validation_split <= 0.0 || self.validation_split >= 1.0 {
            return Err(crate::utils::error::VangaError::ConfigError(format!(
                "validation_split must be between 0.0 and 1.0, got: {}",
                self.validation_split
            )));
        }

        // Validate test split
        if self.test_split < 0.0 || self.test_split >= 1.0 {
            return Err(crate::utils::error::VangaError::ConfigError(format!(
                "test_split must be between 0.0 and 1.0, got: {}",
                self.test_split
            )));
        }

        // Validate that validation + test splits don't exceed 1.0
        if self.validation_split + self.test_split >= 1.0 {
            return Err(crate::utils::error::VangaError::ConfigError(format!(
                "validation_split + test_split must be less than 1.0, got: {} + {} = {}",
                self.validation_split,
                self.test_split,
                self.validation_split + self.test_split
            )));
        }

        // Validate optimizer parameters
        self.validate_optimizer()?;

        // Validate batch size parameters
        self.validate_batch_size()?;

        // Validate early stopping parameters
        self.validate_early_stopping()?;

        // Validate learning schedule parameters
        self.validate_learning_schedule()?;

        Ok(())
    }

    /// Validate optimizer-specific parameters
    fn validate_optimizer(&self) -> Result<()> {
        match &self.optimizer {
            OptimizerType::SGD { momentum } => {
                if let Some(m) = momentum {
                    if *m < 0.0 || *m >= 1.0 {
                        return Err(VangaError::ConfigError(format!(
                            "SGD momentum must be between 0.0 and 1.0, got: {}",
                            m
                        )));
                    }
                }
            }
            OptimizerType::AdamW {
                weight_decay,
                beta1,
                beta2,
                eps,
            } => {
                if *weight_decay < 0.0 {
                    return Err(VangaError::ConfigError(format!(
                        "AdamW weight_decay must be non-negative, got: {}",
                        weight_decay
                    )));
                }
                if *beta1 <= 0.0 || *beta1 >= 1.0 {
                    return Err(VangaError::ConfigError(format!(
                        "AdamW beta1 must be between 0.0 and 1.0, got: {}",
                        beta1
                    )));
                }
                if *beta2 <= 0.0 || *beta2 >= 1.0 {
                    return Err(VangaError::ConfigError(format!(
                        "AdamW beta2 must be between 0.0 and 1.0, got: {}",
                        beta2
                    )));
                }
                if *eps <= 0.0 {
                    return Err(VangaError::ConfigError(format!(
                        "AdamW eps must be positive, got: {}",
                        eps
                    )));
                }
            }
            OptimizerType::Adam {
                beta1,
                beta2,
                eps,
                weight_decay,
                amsgrad: _,
            } => {
                if *beta1 <= 0.0 || *beta1 >= 1.0 {
                    return Err(VangaError::ConfigError(format!(
                        "Adam beta1 must be between 0.0 and 1.0, got: {}",
                        beta1
                    )));
                }
                if *beta2 <= 0.0 || *beta2 >= 1.0 {
                    return Err(VangaError::ConfigError(format!(
                        "Adam beta2 must be between 0.0 and 1.0, got: {}",
                        beta2
                    )));
                }
                if *eps <= 0.0 {
                    return Err(VangaError::ConfigError(format!(
                        "Adam eps must be positive, got: {}",
                        eps
                    )));
                }
                if let Some(wd) = weight_decay {
                    if *wd < 0.0 {
                        return Err(VangaError::ConfigError(format!(
                            "Adam weight_decay must be non-negative, got: {}",
                            wd
                        )));
                    }
                }
            }
            OptimizerType::AdaDelta {
                rho,
                eps,
                weight_decay,
            } => {
                if *rho <= 0.0 || *rho >= 1.0 {
                    return Err(VangaError::ConfigError(format!(
                        "AdaDelta rho must be between 0.0 and 1.0, got: {}",
                        rho
                    )));
                }
                if *eps <= 0.0 {
                    return Err(VangaError::ConfigError(format!(
                        "AdaDelta eps must be positive, got: {}",
                        eps
                    )));
                }
                if let Some(wd) = weight_decay {
                    if *wd < 0.0 {
                        return Err(VangaError::ConfigError(format!(
                            "AdaDelta weight_decay must be non-negative, got: {}",
                            wd
                        )));
                    }
                }
            }
            OptimizerType::AdaGrad {
                lr_decay,
                weight_decay,
                initial_accumulator_value,
                eps,
            } => {
                if *lr_decay < 0.0 {
                    return Err(VangaError::ConfigError(format!(
                        "AdaGrad lr_decay must be non-negative, got: {}",
                        lr_decay
                    )));
                }
                if *initial_accumulator_value < 0.0 {
                    return Err(VangaError::ConfigError(format!(
                        "AdaGrad initial_accumulator_value must be non-negative, got: {}",
                        initial_accumulator_value
                    )));
                }
                if *eps <= 0.0 {
                    return Err(VangaError::ConfigError(format!(
                        "AdaGrad eps must be positive, got: {}",
                        eps
                    )));
                }
                if let Some(wd) = weight_decay {
                    if *wd < 0.0 {
                        return Err(VangaError::ConfigError(format!(
                            "AdaGrad weight_decay must be non-negative, got: {}",
                            wd
                        )));
                    }
                }
            }
            OptimizerType::AdaMax {
                beta1,
                beta2,
                eps,
                weight_decay,
            } => {
                if *beta1 <= 0.0 || *beta1 >= 1.0 {
                    return Err(VangaError::ConfigError(format!(
                        "AdaMax beta1 must be between 0.0 and 1.0, got: {}",
                        beta1
                    )));
                }
                if *beta2 <= 0.0 || *beta2 >= 1.0 {
                    return Err(VangaError::ConfigError(format!(
                        "AdaMax beta2 must be between 0.0 and 1.0, got: {}",
                        beta2
                    )));
                }
                if *eps <= 0.0 {
                    return Err(VangaError::ConfigError(format!(
                        "AdaMax eps must be positive, got: {}",
                        eps
                    )));
                }
                if let Some(wd) = weight_decay {
                    if *wd < 0.0 {
                        return Err(VangaError::ConfigError(format!(
                            "AdaMax weight_decay must be non-negative, got: {}",
                            wd
                        )));
                    }
                }
            }
            OptimizerType::NAdam {
                beta1,
                beta2,
                eps,
                weight_decay,
                momentum_decay,
            } => {
                if *beta1 <= 0.0 || *beta1 >= 1.0 {
                    return Err(VangaError::ConfigError(format!(
                        "NAdam beta1 must be between 0.0 and 1.0, got: {}",
                        beta1
                    )));
                }
                if *beta2 <= 0.0 || *beta2 >= 1.0 {
                    return Err(VangaError::ConfigError(format!(
                        "NAdam beta2 must be between 0.0 and 1.0, got: {}",
                        beta2
                    )));
                }
                if *eps <= 0.0 {
                    return Err(VangaError::ConfigError(format!(
                        "NAdam eps must be positive, got: {}",
                        eps
                    )));
                }
                if *momentum_decay < 0.0 {
                    return Err(VangaError::ConfigError(format!(
                        "NAdam momentum_decay must be non-negative, got: {}",
                        momentum_decay
                    )));
                }
                if let Some(wd) = weight_decay {
                    if *wd < 0.0 {
                        return Err(VangaError::ConfigError(format!(
                            "NAdam weight_decay must be non-negative, got: {}",
                            wd
                        )));
                    }
                }
            }
            OptimizerType::RAdam {
                beta1,
                beta2,
                eps,
                weight_decay,
            } => {
                if *beta1 <= 0.0 || *beta1 >= 1.0 {
                    return Err(VangaError::ConfigError(format!(
                        "RAdam beta1 must be between 0.0 and 1.0, got: {}",
                        beta1
                    )));
                }
                if *beta2 <= 0.0 || *beta2 >= 1.0 {
                    return Err(VangaError::ConfigError(format!(
                        "RAdam beta2 must be between 0.0 and 1.0, got: {}",
                        beta2
                    )));
                }
                if *eps <= 0.0 {
                    return Err(VangaError::ConfigError(format!(
                        "RAdam eps must be positive, got: {}",
                        eps
                    )));
                }
                if let Some(wd) = weight_decay {
                    if *wd < 0.0 {
                        return Err(VangaError::ConfigError(format!(
                            "RAdam weight_decay must be non-negative, got: {}",
                            wd
                        )));
                    }
                }
            }
            OptimizerType::RMSprop {
                alpha,
                eps,
                weight_decay,
                momentum,
                centered: _,
            } => {
                if *alpha <= 0.0 || *alpha >= 1.0 {
                    return Err(VangaError::ConfigError(format!(
                        "RMSprop alpha must be between 0.0 and 1.0, got: {}",
                        alpha
                    )));
                }
                if *eps <= 0.0 {
                    return Err(VangaError::ConfigError(format!(
                        "RMSprop eps must be positive, got: {}",
                        eps
                    )));
                }
                if *momentum < 0.0 || *momentum >= 1.0 {
                    return Err(VangaError::ConfigError(format!(
                        "RMSprop momentum must be between 0.0 and 1.0, got: {}",
                        momentum
                    )));
                }
                if let Some(wd) = weight_decay {
                    if *wd < 0.0 {
                        return Err(VangaError::ConfigError(format!(
                            "RMSprop weight_decay must be non-negative, got: {}",
                            wd
                        )));
                    }
                }
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
                if *beta1 <= 0.0 || *beta1 >= 1.0 {
                    return Err(VangaError::ConfigError(format!(
                        "FracAdam beta1 must be between 0.0 and 1.0, got: {}",
                        beta1
                    )));
                }
                if *beta2 <= 0.0 || *beta2 >= 1.0 {
                    return Err(VangaError::ConfigError(format!(
                        "FracAdam beta2 must be between 0.0 and 1.0, got: {}",
                        beta2
                    )));
                }
                if *eps <= 0.0 {
                    return Err(VangaError::ConfigError(format!(
                        "FracAdam eps must be positive, got: {}",
                        eps
                    )));
                }
                if let Some(wd) = weight_decay {
                    if *wd < 0.0 {
                        return Err(VangaError::ConfigError(format!(
                            "FracAdam weight_decay must be non-negative, got: {}",
                            wd
                        )));
                    }
                }
                if *alpha <= 0.0 || *alpha > 1.0 {
                    return Err(VangaError::ConfigError(format!(
                        "FracAdam alpha (fractional order) must be in (0, 1], got: {}",
                        alpha
                    )));
                }
                if *memory_window == 0 {
                    return Err(VangaError::ConfigError(
                        "FracAdam memory_window must be positive".to_string(),
                    ));
                }
                if *memory_window > 200 {
                    return Err(VangaError::ConfigError(format!(
                        "FracAdam memory_window too large ({}), maximum recommended: 200",
                        memory_window
                    )));
                }
                if *step_size <= 0.0 {
                    return Err(VangaError::ConfigError(format!(
                        "FracAdam step_size must be positive, got: {}",
                        step_size
                    )));
                }
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
                if *beta1 <= 0.0 || *beta1 >= 1.0 {
                    return Err(VangaError::ConfigError(format!(
                        "FracNAdam beta1 must be between 0.0 and 1.0, got: {}",
                        beta1
                    )));
                }
                if *beta2 <= 0.0 || *beta2 >= 1.0 {
                    return Err(VangaError::ConfigError(format!(
                        "FracNAdam beta2 must be between 0.0 and 1.0, got: {}",
                        beta2
                    )));
                }
                if *eps <= 0.0 {
                    return Err(VangaError::ConfigError(format!(
                        "FracNAdam eps must be positive, got: {}",
                        eps
                    )));
                }
                if let Some(wd) = weight_decay {
                    if *wd < 0.0 {
                        return Err(VangaError::ConfigError(format!(
                            "FracNAdam weight_decay must be non-negative, got: {}",
                            wd
                        )));
                    }
                }
                if *momentum_decay < 0.0 {
                    return Err(VangaError::ConfigError(format!(
                        "FracNAdam momentum_decay must be non-negative, got: {}",
                        momentum_decay
                    )));
                }
                if *alpha <= 0.0 || *alpha > 1.0 {
                    return Err(VangaError::ConfigError(format!(
                        "FracNAdam alpha (fractional order) must be in (0, 1], got: {}",
                        alpha
                    )));
                }
                if *memory_window == 0 {
                    return Err(VangaError::ConfigError(
                        "FracNAdam memory_window must be positive".to_string(),
                    ));
                }
                if *memory_window > 200 {
                    return Err(VangaError::ConfigError(format!(
                        "FracNAdam memory_window too large ({}), maximum recommended: 200",
                        memory_window
                    )));
                }
                if *step_size <= 0.0 {
                    return Err(VangaError::ConfigError(format!(
                        "FracNAdam step_size must be positive, got: {}",
                        step_size
                    )));
                }
            }
            OptimizerType::Prodigy {
                d_coef,
                growth_rate,
                beta1,
                beta2,
                eps,
                weight_decay,
                safeguard_warmup: _,
            } => {
                if *d_coef <= 0.0 {
                    return Err(VangaError::ConfigError(format!(
                        "Prodigy d_coef must be positive, got: {}",
                        d_coef
                    )));
                }
                if growth_rate.is_finite() && *growth_rate <= 1.0 {
                    return Err(VangaError::ConfigError(format!(
                        "Prodigy growth_rate must be > 1.0 or infinite, got: {}",
                        growth_rate
                    )));
                }
                if *beta1 <= 0.0 || *beta1 >= 1.0 {
                    return Err(VangaError::ConfigError(format!(
                        "Prodigy beta1 must be between 0.0 and 1.0, got: {}",
                        beta1
                    )));
                }
                if *beta2 <= 0.0 || *beta2 >= 1.0 {
                    return Err(VangaError::ConfigError(format!(
                        "Prodigy beta2 must be between 0.0 and 1.0, got: {}",
                        beta2
                    )));
                }
                if *eps <= 0.0 {
                    return Err(VangaError::ConfigError(format!(
                        "Prodigy eps must be positive, got: {}",
                        eps
                    )));
                }
                if *weight_decay < 0.0 {
                    return Err(VangaError::ConfigError(format!(
                        "Prodigy weight_decay must be non-negative, got: {}",
                        weight_decay
                    )));
                }
                log::info!("✅ Prodigy optimizer validated (learning-rate-free)");
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
                // Validate fractional parameters
                if *alpha <= 0.0 || *alpha > 1.0 {
                    return Err(VangaError::ConfigError(format!(
                        "FracProdigy alpha must be in (0, 1], got: {}",
                        alpha
                    )));
                }
                if *memory_window == 0 {
                    return Err(VangaError::ConfigError(
                        "FracProdigy memory_window must be positive".to_string(),
                    ));
                }
                if *memory_window > 200 {
                    log::warn!(
                        "⚠️ FracProdigy memory_window {} is very large (recommended: 10-50)",
                        memory_window
                    );
                }
                if *step_size <= 0.0 {
                    return Err(VangaError::ConfigError(format!(
                        "FracProdigy step_size must be positive, got: {}",
                        step_size
                    )));
                }

                // Validate Prodigy parameters
                if *d_coef <= 0.0 {
                    return Err(VangaError::ConfigError(format!(
                        "FracProdigy d_coef must be positive, got: {}",
                        d_coef
                    )));
                }
                if growth_rate.is_finite() && *growth_rate <= 1.0 {
                    return Err(VangaError::ConfigError(format!(
                        "FracProdigy growth_rate must be > 1.0 or infinite, got: {}",
                        growth_rate
                    )));
                }

                // Validate NAdam parameters
                if *beta1 <= 0.0 || *beta1 >= 1.0 {
                    return Err(VangaError::ConfigError(format!(
                        "FracProdigy beta1 must be between 0.0 and 1.0, got: {}",
                        beta1
                    )));
                }
                if *beta2 <= 0.0 || *beta2 >= 1.0 {
                    return Err(VangaError::ConfigError(format!(
                        "FracProdigy beta2 must be between 0.0 and 1.0, got: {}",
                        beta2
                    )));
                }
                if *momentum_decay < 0.0 || *momentum_decay > 1.0 {
                    return Err(VangaError::ConfigError(format!(
                        "FracProdigy momentum_decay must be between 0.0 and 1.0, got: {}",
                        momentum_decay
                    )));
                }
                if *eps <= 0.0 {
                    return Err(VangaError::ConfigError(format!(
                        "FracProdigy eps must be positive, got: {}",
                        eps
                    )));
                }

                // Validate weight decay
                if let Some(wd) = weight_decay {
                    if *wd < 0.0 {
                        return Err(VangaError::ConfigError(format!(
                            "FracProdigy weight_decay must be non-negative, got: {}",
                            wd
                        )));
                    }
                }

                log::info!("✅ FracProdigy optimizer validated (fractional memory + automatic LR)");
            }
        }
        Ok(())
    }
    /// Validate batch size configuration parameters
    fn validate_batch_size(&self) -> Result<()> {
        match &self.batch_size {
            BatchSizeConfig::Fixed(size) => {
                if *size == 0 {
                    return Err(VangaError::ConfigError(
                        "Fixed batch_size must be greater than 0".to_string(),
                    ));
                }
                if *size > 1024 {
                    return Err(VangaError::ConfigError(format!(
                        "Fixed batch_size {} is too large (max: 1024)",
                        size
                    )));
                }
            }
            BatchSizeConfig::Auto { min_size, max_size } => {
                if *min_size == 0 {
                    return Err(VangaError::ConfigError(
                        "Auto batch_size min_size must be greater than 0".to_string(),
                    ));
                }
                if *max_size == 0 {
                    return Err(VangaError::ConfigError(
                        "Auto batch_size max_size must be greater than 0".to_string(),
                    ));
                }
                if min_size >= max_size {
                    return Err(VangaError::ConfigError(format!(
                        "Auto batch_size min_size ({}) must be less than max_size ({})",
                        min_size, max_size
                    )));
                }
                if *max_size > 1024 {
                    return Err(VangaError::ConfigError(format!(
                        "Auto batch_size max_size {} is too large (max: 1024)",
                        max_size
                    )));
                }
            }
        }
        Ok(())
    }

    /// Validate early stopping configuration parameters
    fn validate_early_stopping(&self) -> Result<()> {
        if self.early_stopping.patience == 0 {
            return Err(VangaError::ConfigError(
                "early_stopping.patience must be greater than 0".to_string(),
            ));
        }

        if self.early_stopping.min_delta < 0.0 {
            return Err(VangaError::ConfigError(format!(
                "early_stopping.min_delta must be non-negative, got: {}",
                self.early_stopping.min_delta
            )));
        }

        Ok(())
    }

    /// Validate learning schedule configuration parameters
    fn validate_learning_schedule(&self) -> Result<()> {
        // Validate base learning rate
        if self.learning_rate <= 0.0 {
            return Err(VangaError::ConfigError(format!(
                "learning_rate must be positive, got: {}",
                self.learning_rate
            )));
        }

        // Validate learning schedule if present
        if let Some(schedule) = &self.learning_schedule {
            match schedule {
                LearningScheduleConfig::Constant => {
                    // No parameters to validate for constant schedule
                }

                LearningScheduleConfig::ReduceOnPlateau {
                    patience,
                    factor,
                    min_lr,
                    monitor: _,
                    threshold: _,
                } => {
                    if *patience == 0 {
                        return Err(VangaError::ConfigError(
                            "ReduceOnPlateau patience must be greater than 0".to_string(),
                        ));
                    }
                    if *factor <= 0.0 || *factor >= 1.0 {
                        return Err(VangaError::ConfigError(format!(
                            "ReduceOnPlateau factor must be between 0.0 and 1.0, got: {}",
                            factor
                        )));
                    }
                    if let Some(min_lr_val) = min_lr {
                        if *min_lr_val <= 0.0 {
                            return Err(VangaError::ConfigError(format!(
                                "ReduceOnPlateau min_lr must be positive, got: {}",
                                min_lr_val
                            )));
                        }
                        if *min_lr_val >= self.learning_rate {
                            return Err(VangaError::ConfigError(format!(
                                "ReduceOnPlateau min_lr must be less than base learning_rate, got: {} >= {}",
                                min_lr_val, self.learning_rate
                            )));
                        }
                    }
                }

                LearningScheduleConfig::LinearDecay { decay_rate, min_lr } => {
                    if *decay_rate < 0.0 || *decay_rate > 1.0 {
                        return Err(VangaError::ConfigError(format!(
                            "LinearDecay decay_rate must be between 0.0 and 1.0, got: {}",
                            decay_rate
                        )));
                    }
                    if let Some(min_lr_val) = min_lr {
                        if *min_lr_val <= 0.0 {
                            return Err(VangaError::ConfigError(format!(
                                "LinearDecay min_lr must be positive, got: {}",
                                min_lr_val
                            )));
                        }
                    }
                }

                LearningScheduleConfig::ExponentialDecay { gamma, min_lr } => {
                    if *gamma <= 0.0 || *gamma > 1.0 {
                        return Err(VangaError::ConfigError(format!(
                            "ExponentialDecay gamma must be between 0.0 and 1.0, got: {}",
                            gamma
                        )));
                    }
                    if let Some(min_lr_val) = min_lr {
                        if *min_lr_val <= 0.0 {
                            return Err(VangaError::ConfigError(format!(
                                "ExponentialDecay min_lr must be positive, got: {}",
                                min_lr_val
                            )));
                        }
                    }
                }

                LearningScheduleConfig::CosineAnnealing { t_max, eta_min } => {
                    if *t_max == 0 {
                        return Err(VangaError::ConfigError(
                            "CosineAnnealing t_max must be greater than 0".to_string(),
                        ));
                    }
                    if let Some(eta_min_val) = eta_min {
                        if *eta_min_val <= 0.0 {
                            return Err(VangaError::ConfigError(format!(
                                "CosineAnnealing eta_min must be positive, got: {}",
                                eta_min_val
                            )));
                        }
                    }
                }

                LearningScheduleConfig::WarmRestarts {
                    t_0,
                    t_mult,
                    eta_min,
                } => {
                    if *t_0 == 0 {
                        return Err(VangaError::ConfigError(
                            "WarmRestarts t_0 must be greater than 0".to_string(),
                        ));
                    }
                    if *t_mult == 0 {
                        return Err(VangaError::ConfigError(
                            "WarmRestarts t_mult must be greater than 0".to_string(),
                        ));
                    }
                    if let Some(eta_min_val) = eta_min {
                        if *eta_min_val <= 0.0 {
                            return Err(VangaError::ConfigError(format!(
                                "WarmRestarts eta_min must be positive, got: {}",
                                eta_min_val
                            )));
                        }
                    }
                }

                // Add validation for new schedule types
                LearningScheduleConfig::StepDecay {
                    step_size,
                    gamma,
                    milestones,
                    min_lr,
                } => {
                    if *step_size == 0 {
                        return Err(VangaError::ConfigError(
                            "StepDecay step_size must be greater than 0".to_string(),
                        ));
                    }
                    if *gamma <= 0.0 || *gamma > 1.0 {
                        return Err(VangaError::ConfigError(format!(
                            "StepDecay gamma must be between 0.0 and 1.0, got: {}",
                            gamma
                        )));
                    }
                    if let Some(milestones_vec) = milestones {
                        if milestones_vec.is_empty() {
                            return Err(VangaError::ConfigError(
                                "StepDecay milestones cannot be empty".to_string(),
                            ));
                        }
                        // Check ascending order
                        for window in milestones_vec.windows(2) {
                            if window[0] >= window[1] {
                                return Err(VangaError::ConfigError(
                                    "StepDecay milestones must be in ascending order".to_string(),
                                ));
                            }
                        }
                    }
                    if let Some(min_lr_val) = min_lr {
                        if *min_lr_val <= 0.0 {
                            return Err(VangaError::ConfigError(format!(
                                "StepDecay min_lr must be positive, got: {}",
                                min_lr_val
                            )));
                        }
                    }
                }

                LearningScheduleConfig::PolynomialDecay { power, min_lr } => {
                    if *power <= 0.0 {
                        return Err(VangaError::ConfigError(format!(
                            "PolynomialDecay power must be positive, got: {}",
                            power
                        )));
                    }
                    if let Some(min_lr_val) = min_lr {
                        if *min_lr_val <= 0.0 {
                            return Err(VangaError::ConfigError(format!(
                                "PolynomialDecay min_lr must be positive, got: {}",
                                min_lr_val
                            )));
                        }
                    }
                }

                LearningScheduleConfig::OneCycle {
                    max_lr,
                    pct_start,
                    div_factor,
                    final_div_factor,
                    ..
                } => {
                    if *max_lr <= 0.0 {
                        return Err(VangaError::ConfigError(format!(
                            "OneCycle max_lr must be positive, got: {}",
                            max_lr
                        )));
                    }
                    if let Some(pct) = pct_start {
                        if *pct <= 0.0 || *pct >= 1.0 {
                            return Err(VangaError::ConfigError(format!(
                                "OneCycle pct_start must be between 0.0 and 1.0, got: {}",
                                pct
                            )));
                        }
                    }
                    if let Some(div) = div_factor {
                        if *div <= 1.0 {
                            return Err(VangaError::ConfigError(format!(
                                "OneCycle div_factor must be greater than 1.0, got: {}",
                                div
                            )));
                        }
                    }
                    if let Some(final_div) = final_div_factor {
                        if *final_div <= 1.0 {
                            return Err(VangaError::ConfigError(format!(
                                "OneCycle final_div_factor must be greater than 1.0, got: {}",
                                final_div
                            )));
                        }
                    }
                }

                LearningScheduleConfig::CyclicalLR {
                    base_lr,
                    max_lr,
                    step_size_up,
                    ..
                } => {
                    if *base_lr <= 0.0 {
                        return Err(VangaError::ConfigError(format!(
                            "CyclicalLR base_lr must be positive, got: {}",
                            base_lr
                        )));
                    }
                    if *max_lr <= 0.0 {
                        return Err(VangaError::ConfigError(format!(
                            "CyclicalLR max_lr must be positive, got: {}",
                            max_lr
                        )));
                    }
                    if *max_lr <= *base_lr {
                        return Err(VangaError::ConfigError(format!(
                            "CyclicalLR max_lr ({}) must be greater than base_lr ({})",
                            max_lr, base_lr
                        )));
                    }
                    if *step_size_up == 0 {
                        return Err(VangaError::ConfigError(
                            "CyclicalLR step_size_up must be greater than 0".to_string(),
                        ));
                    }
                }

                LearningScheduleConfig::NoamLR {
                    model_size,
                    warmup_steps,
                    ..
                } => {
                    if *model_size == 0 {
                        return Err(VangaError::ConfigError(
                            "NoamLR model_size must be greater than 0".to_string(),
                        ));
                    }
                    if *warmup_steps == 0 {
                        return Err(VangaError::ConfigError(
                            "NoamLR warmup_steps must be greater than 0".to_string(),
                        ));
                    }
                }
            }
        }

        Ok(())
    }
}

impl OptimizationConfig {
    /// Validate optimization configuration parameters
    pub fn validate(&self) -> Result<()> {
        // Validate number of trials
        if self.n_trials == 0 {
            return Err(crate::utils::error::VangaError::ConfigError(
                "n_trials must be greater than 0".to_string(),
            ));
        }

        // Validate timeout
        if let Some(timeout) = self.timeout_seconds {
            if timeout == 0 {
                return Err(crate::utils::error::VangaError::ConfigError(
                    "timeout_seconds must be greater than 0".to_string(),
                ));
            }
        }

        Ok(())
    }
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            symbol: String::new(),
            data_path: PathBuf::new(),
            fresh_training: false,
            continue_training: false,
            horizons: vec!["1h".to_string()], // FIXED: Default to single horizon instead of multiple
            features: FeatureConfig::default(),
            model: ModelConfig::default(),
            training: TrainingParams::default(),
            data: DataConfig::default(),
            optimization: OptimizationConfig::default(),
            targets: TargetsConfig::default(), // Add targets config with all enabled by default
        }
    }
}

impl Default for TrainingParams {
    fn default() -> Self {
        Self {
            epochs: EpochConfig::Auto { max_epochs: 1000 }, // Auto early stopping by default
            batch_size: BatchSizeConfig::Auto {
                min_size: 32,

                max_size: 512,
            },
            learning_rate: 0.001, // Simple float value - 1e-3 default
            optimizer: OptimizerType::AdamW {
                weight_decay: 0.01,
                beta1: 0.9,
                beta2: 0.999,
                eps: 1e-8,
            }, // AdamW by default for better performance
            warmup_epochs: 5,     // 5 epochs warmup by default
            learning_schedule: Some(LearningScheduleConfig::ReduceOnPlateau {
                patience: 10,
                factor: 0.5,
                min_lr: Some(1e-6),
                monitor: Some("loss".to_string()),
                threshold: Some(0.001),
            }), // Adaptive scheduling by default (formerly "Adaptive" learning_rate)
            validation_split: 0.2, // 20% validation for early stopping
            validation_gap: "1h".to_string(), // 1 hour gap by default for feature independence
            test_split: 0.1,
            device: DeviceConfig::Auto,
            early_stopping: EarlyStoppingConfig {
                patience: 50,
                min_delta: 0.001, // Updated for Composite loss scale compatibility
            },
            gradient_clip: Some(1.0), // Prevent exploding gradients
            print_every: 1,           // Print every epoch by default for better monitoring
            window_decay: 1.0,        // No decay by default
            min_train_ratio: 0.4,     // Start with 40% for efficiency
            min_increment_ratio: 0.3, // Ensure 30% new data per window
            seed: default_seed(),     // Random seed by default
        }
    }
}

impl Default for DataConfig {
    fn default() -> Self {
        Self {
            normalization: NormalizationMethod::Robust,
            sequence_overlap: 0.8,
            outlier_handling: OutlierHandling {
                enabled: true,
                method: OutlierMethod::ModifiedZScore,
                threshold: 3.5,
            },
            feature_selection: FeatureSelectionConfig {
                enabled: true,
                max_features: None,
                correlation_threshold: 0.95,
                importance_threshold: 0.001,
            },
        }
    }
}

impl Default for OptimizationConfig {
    fn default() -> Self {
        Self {
            method: OptimizationMethod::Bayesian,
            n_trials: 100,
            timeout_seconds: Some(3600),     // 1 hour
            metric: OptimizationMetric::MAE, // Use MAE for hyperparameter optimization
        }
    }
}

// Builder pattern implementation
impl TrainingConfig {
    pub fn symbol<S: Into<String>>(mut self, symbol: S) -> Self {
        self.symbol = symbol.into();
        self
    }

    pub fn data_path<P: Into<PathBuf>>(mut self, path: P) -> Self {
        self.data_path = path.into();
        self
    }

    pub fn fresh_training(mut self, fresh: bool) -> Self {
        self.fresh_training = fresh;
        self
    }

    pub fn continue_training(mut self, continue_train: bool) -> Self {
        self.continue_training = continue_train;
        self
    }

    pub fn horizons(mut self, horizons: Vec<String>) -> Self {
        self.horizons = horizons;
        self
    }

    /// Validate configuration against provided symbols
    pub fn validate_for_symbols(&self, symbols: &[String]) -> Result<()> {
        // Validate cross-asset features
        if self.features.cross_asset.enabled {
            self.validate_cross_asset_requirements(symbols)?;
        }

        Ok(())
    }

    /// Validate cross-asset feature requirements
    fn validate_cross_asset_requirements(&self, symbols: &[String]) -> Result<()> {
        let config = &self.features.cross_asset;

        // Check minimum symbol count
        if symbols.len() < config.min_symbols_required {
            return Err(VangaError::ConfigError(format!(
                "Cross-asset features enabled but only {} symbols provided. Minimum required: {}. Use: --symbol BTCUSDT,ETHUSDT,DOTUSDT",
                symbols.len(),
                config.min_symbols_required
            )));
        }

        // Check required symbols
        for required in &config.required_symbols {
            if !symbols.contains(required) {
                return Err(VangaError::ConfigError(format!(
                    "Cross-asset features require '{}' but not found in symbols: {:?}",
                    required, symbols
                )));
            }
        }

        // Check BTC dominance requirements
        if config.btc_dominance_enabled && !symbols.contains(&"BTCUSDT".to_string()) {
            return Err(VangaError::ConfigError(format!(
                "BTC dominance enabled but BTCUSDT not in symbols: {:?}",
                symbols
            )));
        }

        // Check ETH/BTC ratio requirements (only warn if enabled but missing symbols)
        if config.eth_btc_ratio_enabled {
            let has_btc = symbols.contains(&"BTCUSDT".to_string());
            let has_eth = symbols.contains(&"ETHUSDT".to_string());

            if !has_btc || !has_eth {
                log::warn!(
                    "ETH/BTC ratio enabled but missing symbols (BTCUSDT: {}, ETHUSDT: {}). Feature will be skipped.",
                    has_btc, has_eth
                );
                // Don't error - just warn and skip this feature
            }
        }

        log::info!(
            "✅ Cross-asset validation passed for symbols: {:?}",
            symbols
        );
        Ok(())
    }

    /// Set device configuration from string
    pub fn with_device_config(mut self, device_str: &str) -> Result<Self> {
        self.training.device = match device_str.to_lowercase().as_str() {
            "auto" => DeviceConfig::Auto,
            "cpu" => DeviceConfig::CPU,
            device_str if device_str.starts_with("gpu:") => {
                let index = device_str[4..].parse::<usize>().map_err(|_| {
                    VangaError::ConfigError(format!("Invalid GPU index in device config: {}", device_str))
                })?;
                DeviceConfig::GPU(index)
            }
            device_str if device_str.starts_with("metal:") => {
                let index = device_str[6..].parse::<usize>().map_err(|_| {
                    VangaError::ConfigError(format!("Invalid Metal index in device config: {}", device_str))
                })?;
                DeviceConfig::Metal(index)
            }
            _ => return Err(VangaError::ConfigError(format!(
                "Invalid device configuration: '{}'. Supported options: 'auto', 'cpu', 'gpu:0', 'metal:0'",
                device_str
            ))),
        };
        log::info!("🔧 Device configuration set to: {}", self.training.device);
        Ok(self)
    }

    /// Enable or disable attention mechanism
    pub fn with_attention_enabled(mut self, enabled: bool) -> Self {
        self.model.attention.enabled = enabled;
        if enabled {
            log::info!("✅ Attention mechanism enabled in model configuration");
        }
        self
    }

    /// Enable or disable TFT (Temporal Fusion Transformer) features
    pub fn with_tft_enabled(mut self, enabled: bool) -> Self {
        if enabled {
            // Enable TFT Variable Selection attention mechanism
            self.model.attention.enabled = true;
            self.model.attention.mechanism =
                crate::config::model::AttentionMechanism::VariableSelection;
            log::info!("✅ TFT Variable Selection attention enabled in model configuration");

            // Enable quantile regression outputs for uncertainty quantification
            self.model.quantile_outputs =
                Some(crate::config::model::TFTQuantileOutputConfig::default());

            log::info!("✅ TFT (Temporal Fusion Transformer) enabled in model configuration");
        }
        self
    }

    /// Enable or disable XGBoost hybrid model
    pub fn with_xgboost_enabled(mut self, enabled: bool) -> Self {
        self.model.xgboost.enabled = enabled;
        if enabled {
            log::info!("✅ XGBoost hybrid model enabled in model configuration");
            log::info!("🔄 Two-phase training: LSTM → XGBoost regression");
        } else {
            log::info!("❌ XGBoost hybrid model disabled - using pure LSTM");
        }
        self
    }

    /// Load training configuration from TOML file
    pub fn from_file<P: AsRef<std::path::Path>>(path: P) -> crate::utils::error::Result<Self> {
        let content = std::fs::read_to_string(&path).map_err(|e| {
            crate::utils::error::VangaError::IoError(format!("Failed to read config file: {}", e))
        })?;

        let config = toml::from_str::<TrainingConfig>(&content).map_err(|e| {
            crate::utils::error::VangaError::ConfigError(format!(
                "Failed to parse config file: {}",
                e
            ))
        })?;

        // Validate configuration parameters
        config.training.validate()?;
        config.optimization.validate()?;

        log::info!(
            "✅ Configuration loaded and validated from: {}",
            path.as_ref().display()
        );
        Ok(config)
    }

    /// Load configuration sections from TOML file and merge with base config
    pub fn with_config_from_file<P: AsRef<std::path::Path>>(
        mut self,
        path: P,
    ) -> crate::utils::error::Result<Self> {
        let content = std::fs::read_to_string(&path).map_err(|e| {
            crate::utils::error::VangaError::IoError(format!("Failed to read config file: {}", e))
        })?;

        // Parse the TOML content
        let parsed: toml::Value = toml::from_str(&content).map_err(|e| {
            crate::utils::error::VangaError::ConfigError(format!(
                "Failed to parse config file: {}",
                e
            ))
        })?;

        // Load training section if present
        if let Some(training_value) = parsed.get("training") {
            let training_params: TrainingParams =
                training_value.clone().try_into().map_err(|e| {
                    crate::utils::error::VangaError::ConfigError(format!(
                        "Failed to parse training: {}",
                        e
                    ))
                })?;
            self.training = training_params;
        }

        // Load model section if present
        if let Some(model_value) = parsed.get("model") {
            let model: ModelConfig = model_value.clone().try_into().map_err(|e| {
                crate::utils::error::VangaError::ConfigError(format!(
                    "Failed to parse model: {}",
                    e
                ))
            })?;
            self.model = model;
        }

        // Load data section if present
        if let Some(data_value) = parsed.get("data") {
            let data_config: DataConfig = data_value.clone().try_into().map_err(|e| {
                crate::utils::error::VangaError::ConfigError(format!("Failed to parse data: {}", e))
            })?;
            self.data = data_config;
        }

        // Load optimization section if present
        if let Some(optimization_value) = parsed.get("optimization") {
            let optimization: OptimizationConfig =
                optimization_value.clone().try_into().map_err(|e| {
                    crate::utils::error::VangaError::ConfigError(format!(
                        "Failed to parse optimization: {}",
                        e
                    ))
                })?;
            self.optimization = optimization;
        }

        // Load features section if present
        if let Some(features_value) = parsed.get("features") {
            let features: FeatureConfig = features_value.clone().try_into().map_err(|e| {
                crate::utils::error::VangaError::ConfigError(format!(
                    "Failed to parse features: {}",
                    e
                ))
            })?;
            self.features = features;
        }

        // Load targets section from [model.targets]
        if let Some(model_section) = parsed.get("model") {
            if let Some(targets_value) = model_section.get("targets") {
                let targets: TargetsConfig = targets_value.clone().try_into().map_err(|e| {
                    crate::utils::error::VangaError::ConfigError(format!(
                        "Failed to parse targets: {}",
                        e
                    ))
                })?;
                self.targets = targets;
                log::info!("📋 Loaded targets configuration from file");
                log::info!("   - price_level: {}", self.targets.price_level);
                log::info!("   - direction: {}", self.targets.direction);
                log::info!("   - volatility: {}", self.targets.volatility);
                log::info!("   - sentiment: {}", self.targets.sentiment);
                log::info!("   - volume: {}", self.targets.volume);
            } else {
                log::info!(
                    "📋 No [model.targets] section in config file, using defaults (all enabled)"
                );
            }
        } else {
            log::info!("📋 No [model] section in config file, using defaults (all enabled)");
        }

        // Configuration loaded successfully

        log::info!(
            "✅ Configuration loaded and validated from: {}",
            path.as_ref().display()
        );
        Ok(self)
    }

    /// Validate the complete training configuration
    pub fn validate(&self) -> Result<()> {
        // Validate targets configuration
        self.targets.validate()?;

        // Validate training parameters
        self.training.validate()?;

        // Validate model configuration
        self.model.validate()?;

        // Validate that we have at least one horizon
        if self.horizons.is_empty() {
            return Err(VangaError::ConfigError(
                "At least one prediction horizon must be specified".to_string(),
            ));
        }

        log::info!("✅ Training configuration validated successfully");
        log::info!("   - Symbol: {}", self.symbol);
        log::info!("   - Horizons: {:?}", self.horizons);
        log::info!(
            "   - Enabled targets: {:?}",
            self.targets.get_enabled_targets()
        );

        Ok(())
    }

    /// Create a minimal configuration suitable for testing
    pub fn default_for_testing() -> Result<Self> {
        let config = Self {
            symbol: "TESTUSDT".to_string(),
            data_path: std::path::PathBuf::from("test_data.csv"),
            targets: TargetsConfig::default(), // Add targets config for testing
            training: TrainingParams {
                epochs: EpochConfig::Fixed(5),
                batch_size: BatchSizeConfig::Fixed(16),
                learning_rate: 0.01,     // Simple float value for testing
                learning_schedule: None, // No scheduling for testing
                ..Default::default()
            },
            model: ModelConfig {
                xgboost: crate::config::model::XGBoostConfig {
                    enabled: true,
                    n_estimators: 10, // Small for testing
                    max_depth: 3,
                    ..Default::default()
                },
                ..Default::default()
            },
            ..Default::default()
        };

        Ok(config)
    }
}

/// Target configuration - simple enable/disable flags
/// All parameters come from automatic calibration system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TargetsConfig {
    /// Enable price level targets (VWAP-weighted price classification)
    pub price_level: bool,

    /// Enable direction targets (directional movement classification)
    pub direction: bool,

    /// Enable volatility targets (volatility regime classification)
    pub volatility: bool,

    /// Enable sentiment targets (market sentiment classification)
    pub sentiment: bool,

    /// Enable volume targets (volume regime classification)
    pub volume: bool,
}

impl Default for TargetsConfig {
    fn default() -> Self {
        Self {
            price_level: true,
            direction: true,
            volatility: true,
            sentiment: true,
            volume: true,
        }
    }
}

impl TargetsConfig {
    /// Validate that at least one target is enabled
    pub fn validate(&self) -> Result<()> {
        if !self.price_level
            && !self.direction
            && !self.volatility
            && !self.sentiment
            && !self.volume
        {
            return Err(VangaError::ConfigError(
                "At least one target must be enabled for training".to_string(),
            ));
        }
        Ok(())
    }

    /// Get list of enabled target types
    pub fn get_enabled_targets(&self) -> Vec<&'static str> {
        let mut enabled = Vec::new();
        if self.price_level {
            enabled.push("price_level");
        }
        if self.direction {
            enabled.push("direction");
        }
        if self.volatility {
            enabled.push("volatility");
        }
        if self.sentiment {
            enabled.push("sentiment");
        }
        if self.volume {
            enabled.push("volume");
        }
        enabled
    }

    /// Count number of enabled targets
    pub fn count_enabled(&self) -> usize {
        let mut count = 0;
        if self.price_level {
            count += 1;
        }
        if self.direction {
            count += 1;
        }
        if self.volatility {
            count += 1;
        }
        if self.sentiment {
            count += 1;
        }
        if self.volume {
            count += 1;
        }
        count
    }
}
