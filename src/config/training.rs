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
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingParams {
    /// Maximum number of epochs ("auto" for early stopping)
    pub epochs: EpochConfig,

    /// Batch size ("auto" for optimization)
    pub batch_size: BatchSizeConfig,

    /// Learning rate configuration
    pub learning_rate: LearningRateConfig,

    /// Optimizer type selection
    pub optimizer: OptimizerType,

    /// Warmup epochs for gradual learning rate increase
    pub warmup_epochs: u32,

    /// Learning rate schedule configuration
    pub learning_schedule: Option<LearningScheduleConfig>,

    /// Validation split ratio
    pub validation_split: f64,

    /// Test split ratio
    pub test_split: f64,

    /// Early stopping patience
    pub early_stopping_patience: u32,

    /// Gradient clipping threshold
    pub gradient_clip: Option<f64>,

    /// Print training progress every N epochs (1 = every epoch, 10 = every 10 epochs)
    pub print_every: u32,
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
    Auto {
        min_lr: f64,
        max_lr: f64,
    },
    Adaptive {
        initial_lr: f64,
        patience: u32,
        factor: f64,
    },
    Fixed(f64),
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
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LearningScheduleConfig {
    Constant,
    LinearDecay { decay_rate: f64 },
    ExponentialDecay { decay_rate: f64 },
    CosineAnnealing { t_max: u32 },
    WarmRestarts { t_0: u32, t_mult: u32 },
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

        Ok(())
    }

    /// Validate training parameters for correctness with symbol context
    pub fn validate_for_symbols(&self, _symbols: &[String]) -> Result<()> {
        // First run basic validation
        self.validate()?;
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

        // Validate combined splits don't exceed 1.0
        if self.validation_split + self.test_split >= 1.0 {
            return Err(crate::utils::error::VangaError::ConfigError(format!(
                "validation_split + test_split must be < 1.0, got: {} + {} = {}",
                self.validation_split,
                self.test_split,
                self.validation_split + self.test_split
            )));
        }

        // Validate gradient clipping
        if let Some(clip) = self.gradient_clip {
            if clip <= 0.0 {
                return Err(crate::utils::error::VangaError::ConfigError(format!(
                    "gradient_clip must be positive, got: {}",
                    clip
                )));
            }
        }

        // Validate early stopping patience
        if self.early_stopping_patience == 0 {
            return Err(crate::utils::error::VangaError::ConfigError(
                "early_stopping_patience must be greater than 0".to_string(),
            ));
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
            horizons: vec!["1h".to_string(), "4h".to_string(), "1d".to_string()],
            features: FeatureConfig::default(),
            model: ModelConfig::default(),
            training: TrainingParams::default(),
            data: DataConfig::default(),
            optimization: OptimizationConfig::default(),
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
            learning_rate: LearningRateConfig::Adaptive {
                initial_lr: 0.001,
                patience: 10,
                factor: 0.5,
            }, // Adaptive by default
            optimizer: OptimizerType::AdamW {
                weight_decay: 0.01,
                beta1: 0.9,
                beta2: 0.999,
            }, // AdamW by default for better performance
            warmup_epochs: 5,        // 5 epochs warmup by default
            learning_schedule: None, // No schedule by default
            validation_split: 0.2,   // 20% validation for early stopping
            test_split: 0.1,
            early_stopping_patience: 50, // Reasonable patience
            gradient_clip: Some(1.0),    // Prevent exploding gradients
            print_every: 1,              // Print every epoch by default for better monitoring
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

        // Configuration loaded successfully

        log::info!(
            "✅ Configuration loaded and validated from: {}",
            path.as_ref().display()
        );
        Ok(self)
    }
}
