use crate::config::ModelConfig;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

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

    /// Path to custom features configuration
    pub features_config_path: Option<PathBuf>,

    /// Model architecture configuration
    pub model_config: ModelConfig,

    /// Training hyperparameters
    pub training_params: TrainingParams,

    /// Data preprocessing configuration
    pub data_config: DataConfig,

    /// Optimization configuration
    pub optimization_config: OptimizationConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingParams {
    /// Maximum number of epochs ("auto" for early stopping)
    pub epochs: EpochConfig,

    /// Batch size ("auto" for optimization)
    pub batch_size: BatchSizeConfig,

    /// Learning rate configuration
    pub learning_rate: LearningRateConfig,

    /// Validation split ratio
    pub validation_split: f64,

    /// Test split ratio
    pub test_split: f64,

    /// Early stopping patience
    pub early_stopping_patience: u32,

    /// Gradient clipping threshold
    pub gradient_clip: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataConfig {
    /// Normalization method
    pub normalization: NormalizationMethod,

    /// Sequence overlap ratio
    pub sequence_overlap: f64,

    /// Missing data handling strategy
    pub missing_data_strategy: MissingDataStrategy,

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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EpochConfig {
    Auto { max_epochs: u32 },
    Fixed(u32),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BatchSizeConfig {
    Auto { min_size: u32, max_size: u32 },
    Fixed(u32),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LearningRateConfig {
    Auto { min_lr: f64, max_lr: f64 },
    Adaptive { initial_lr: f64 },
    Fixed(f64),
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizationMethod {
    Bayesian,
    Grid,
    Random,
    None,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizationMetric {
    Accuracy,
    SharpeRatio,
    MaxDrawdown,
    ProfitFactor,
    Custom(String),
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            symbol: String::new(),
            data_path: PathBuf::new(),
            fresh_training: false,
            continue_training: false,
            horizons: vec!["1h".to_string(), "4h".to_string(), "1d".to_string()],
            features_config_path: None,
            model_config: ModelConfig::default(),
            training_params: TrainingParams::default(),
            data_config: DataConfig::default(),
            optimization_config: OptimizationConfig::default(),
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
            learning_rate: LearningRateConfig::Adaptive { initial_lr: 0.001 }, // Adaptive by default
            validation_split: 0.2, // 20% validation for early stopping
            test_split: 0.1,
            early_stopping_patience: 50, // Reasonable patience
            gradient_clip: Some(1.0),    // Prevent exploding gradients
        }
    }
}

impl Default for DataConfig {
    fn default() -> Self {
        Self {
            normalization: NormalizationMethod::Robust,
            sequence_overlap: 0.8,
            missing_data_strategy: MissingDataStrategy::Interpolate,
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
            timeout_seconds: Some(3600), // 1 hour
            metric: OptimizationMetric::SharpeRatio,
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

    pub fn features_config_path<P: Into<PathBuf>>(mut self, path: P) -> Self {
        self.features_config_path = Some(path.into());
        self
    }
}
