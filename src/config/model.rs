use crate::model::loss::CryptoLossFunction;
use serde::{Deserialize, Serialize};

/// Unified number of classes for all target types in the 5-class system
pub const NUM_CLASSES: usize = 5;

/// TFT Variable Selection configuration for model config
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TFTVariableSelectionConfig {
    pub static_selection: bool,
    pub temporal_selection: bool,
    pub selection_threshold: f64,
    pub top_k_features: Option<usize>,
    pub enable_interpretability: bool,
}

impl Default for TFTVariableSelectionConfig {
    fn default() -> Self {
        Self {
            static_selection: true,
            temporal_selection: true,
            selection_threshold: 0.1,
            top_k_features: None,
            enable_interpretability: true,
        }
    }
}

/// TFT Quantile Output configuration for model config
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TFTQuantileOutputConfig {
    pub enabled: bool,
    pub quantiles: Vec<f64>,
    pub loss_weighting: String, // "equal", "extreme_weighted", "custom"
    pub uncertainty_calibration: bool,
}

impl Default for TFTQuantileOutputConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            quantiles: vec![0.1, 0.5, 0.9],
            loss_weighting: "equal".to_string(),
            uncertainty_calibration: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// LSTM architecture type
    pub architecture: LSTMArchitecture,

    /// Sequence length for LSTM input
    pub sequence_length: SequenceLengthConfig,

    /// Hidden units configuration
    pub hidden_units: HiddenUnitsConfig,

    /// Dropout configuration
    pub dropout: DropoutConfig,

    /// Attention mechanism configuration
    pub attention: AttentionConfig,

    /// XGBoost hybrid model configuration
    pub xgboost: XGBoostConfig,

    /// Output heads configuration
    pub output_heads: OutputHeadsConfig,

    /// TFT Quantile regression configuration
    pub quantile_outputs: Option<TFTQuantileOutputConfig>,

    /// Loss function configuration for multi-target training
    pub loss_function: CryptoLossFunction,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LSTMArchitecture {
    /// Multi-layer LSTM with shared representation
    MultiLSTM { layers: u32 },

    /// Stacked LSTM layers
    StackedLSTM { layers: u32 },

    /// Bidirectional LSTM
    BidirectionalLSTM { layers: u32 },

    /// LSTM with CNN feature extraction
    CNNLSTM { cnn_layers: u32, lstm_layers: u32 },

    /// Transformer-LSTM hybrid
    TransformerLSTM {
        transformer_layers: u32,
        lstm_layers: u32,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SequenceLengthConfig {
    Auto { min_length: u32, max_length: u32 },
    Fixed(u32),
    Adaptive,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HiddenUnitsConfig {
    Auto {
        min_units: u32,
        max_units: u32,
    },
    Fixed(Vec<u32>),
    Pyramid {
        base_units: u32,
        reduction_factor: f64,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DropoutConfig {
    pub enabled: bool,
    pub rate: DropoutRate,
    pub variational: bool,
    pub recurrent: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DropoutRate {
    Auto { min_rate: f64, max_rate: f64 },
    Fixed(f64),
    Adaptive,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionConfig {
    pub enabled: bool,
    pub mechanism: AttentionMechanism,
    pub heads: u32,
    pub head_dim: Option<u32>,              // Auto-optimized if None
    pub dropout_rate: f64,                  // Attention dropout
    pub temperature_scaling: f64,           // Crypto volatility adaptation
    pub use_relative_position: bool,        // Temporal modeling for crypto
    pub visualization: VisualizationConfig, // Analysis options
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AttentionMechanism {
    SelfAttention,
    MultiHeadAttention,
    AdditiveAttention,
    /// TFT Variable Selection Attention (builds on MultiHeadAttention)
    VariableSelection,
    None,
}

/// XGBoost hybrid model configuration following attention pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct XGBoostConfig {
    /// Enable/disable XGBoost hybrid mode
    pub enabled: bool,

    /// LSTM feature extraction dimension (k in paper, typically 64)
    pub feature_dim: usize,

    /// XGBoost hyperparameters
    pub n_estimators: usize, // Number of trees (M in equation 9)
    pub max_depth: usize,      // Maximum tree depth
    pub learning_rate: f64,    // Learning rate (eta)
    pub subsample: f64,        // Row sampling ratio
    pub colsample_bytree: f64, // Column sampling ratio
    pub reg_alpha: f64,        // L1 regularization
    pub reg_lambda: f64,       // L2 regularization (λ in equation 11)

    /// Training configuration
    pub early_stopping_rounds: Option<usize>,
    pub eval_metric: String, // "rmse", "mae", "logloss", etc.
    pub objective: String,   // "reg:squarederror", "multi:softprob", etc.

    /// Feature importance analysis
    pub save_feature_importance: bool,
    pub importance_type: String, // "weight", "gain", "cover"
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualizationConfig {
    pub save_heatmaps: bool,
    pub export_analysis: bool,
    pub output_dir: String,
}

impl Default for VisualizationConfig {
    fn default() -> Self {
        Self {
            save_heatmaps: false,   // Disabled by default for performance
            export_analysis: false, // Disabled by default
            output_dir: "attention_analysis".to_string(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputHeadsConfig {
    /// Price level classification head
    pub price_levels: PriceLevelHead,

    /// Direction prediction head
    pub direction: DirectionHead,

    /// Volatility prediction head
    pub volatility: VolatilityHead,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PriceLevelHead {
    pub enabled: bool,
    pub bandwidth_size: Option<f64>, // Optional for backward compatibility
    pub percentiles: Option<[f64; 2]>, // [lower, upper] percentiles e.g., [0.1, 0.9]
}

impl PriceLevelHead {
    /// Validate the price level head configuration
    pub fn validate(&self) -> Result<(), crate::utils::error::VangaError> {
        if let Some(bandwidth_size) = self.bandwidth_size {
            if bandwidth_size <= 0.0 {
                return Err(crate::utils::error::VangaError::config(
                    "Price level bandwidth_size must be positive",
                ));
            }
            if !bandwidth_size.is_finite() {
                return Err(crate::utils::error::VangaError::config(
                    "Price level bandwidth_size must be finite",
                ));
            }
        }

        // Validate percentiles if provided
        if let Some(percentiles) = self.percentiles {
            if percentiles[0] >= percentiles[1] {
                return Err(crate::utils::error::VangaError::config(
                    "Price level percentiles[0] must be less than percentiles[1]",
                ));
            }
            if percentiles[0] < 0.0 || percentiles[0] > 1.0 {
                return Err(crate::utils::error::VangaError::config(
                    "Price level percentiles[0] must be between 0.0 and 1.0",
                ));
            }
            if percentiles[1] < 0.0 || percentiles[1] > 1.0 {
                return Err(crate::utils::error::VangaError::config(
                    "Price level percentiles[1] must be between 0.0 and 1.0",
                ));
            }
        }

        // Unified 5-class system for all targets
        // bandwidth_size only affects breakout sensitivity, not number of bins
        // No validation needed for bins since it's architecturally fixed

        Ok(())
    }
}

impl DirectionHead {
    /// Validate the direction head configuration
    pub fn validate(&self) -> Result<(), crate::utils::error::VangaError> {
        if let Some(slope_sensitivity) = self.slope_sensitivity {
            if slope_sensitivity <= 0.0 {
                return Err(crate::utils::error::VangaError::config(
                    "Direction slope_sensitivity must be positive",
                ));
            }
        }
        if let Some(base_threshold) = self.base_threshold {
            if base_threshold <= 0.0 || base_threshold > 1.0 {
                return Err(crate::utils::error::VangaError::config(
                    "Direction base_threshold must be between 0.0 and 1.0",
                ));
            }
        }
        if let Some(extreme_multiplier) = self.extreme_multiplier {
            if extreme_multiplier <= 1.0 || extreme_multiplier > 10.0 {
                return Err(crate::utils::error::VangaError::config(
                    "Direction extreme_multiplier must be between 1.0 and 10.0",
                ));
            }
        }
        Ok(())
    }
}

impl VolatilityHead {
    /// Validate volatility head configuration
    pub fn validate(&self) -> Result<(), crate::utils::error::VangaError> {
        if let Some(bandwidth_size) = self.bandwidth_size {
            if bandwidth_size <= 0.0 {
                return Err(crate::utils::error::VangaError::config(
                    "Volatility bandwidth_size must be positive",
                ));
            }
        }
        if let Some(base_threshold) = self.base_threshold {
            if base_threshold <= 0.0 || base_threshold > 1.0 {
                return Err(crate::utils::error::VangaError::config(
                    "Volatility base_threshold must be between 0.0 and 1.0",
                ));
            }
        }
        if let Some(extreme_multiplier) = self.extreme_multiplier {
            if extreme_multiplier <= 1.0 || extreme_multiplier > 10.0 {
                return Err(crate::utils::error::VangaError::config(
                    "Volatility extreme_multiplier must be between 1.0 and 10.0",
                ));
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DirectionHead {
    pub enabled: bool,
    pub slope_sensitivity: Option<f64>, // Controls slope acceleration thresholds for trend momentum detection
    pub base_threshold: Option<f64>,    // Base momentum threshold (default: 0.12 = 12%)
    pub extreme_multiplier: Option<f64>, // Extreme class multiplier (default: 2.0)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VolatilityHead {
    pub enabled: bool,
    pub bandwidth_size: Option<f64>, // Unified sensitivity control (same as DirectionHead)
    pub base_threshold: Option<f64>, // Base ATR threshold (default: 0.15 = 15%)
    pub extreme_multiplier: Option<f64>, // Extreme class multiplier (default: 1.8)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VolatilityPredictionMethod {
    Direct,
    GARCH,
    Stochastic,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            architecture: LSTMArchitecture::MultiLSTM { layers: 2 },
            sequence_length: SequenceLengthConfig::Auto {
                min_length: 30,
                max_length: 120,
            },
            hidden_units: HiddenUnitsConfig::Auto {
                min_units: 64,
                max_units: 512,
            },
            dropout: DropoutConfig {
                enabled: true,
                rate: DropoutRate::Auto {
                    min_rate: 0.1,
                    max_rate: 0.5,
                },
                variational: true,
                recurrent: true,
            },
            attention: AttentionConfig {
                enabled: true,
                mechanism: AttentionMechanism::SelfAttention,
                heads: 8,
                head_dim: None,              // Auto-optimized
                dropout_rate: 0.1,           // Conservative for crypto
                temperature_scaling: 1.0,    // Standard temperature
                use_relative_position: true, // Better for time series
                visualization: VisualizationConfig::default(),
            },
            xgboost: XGBoostConfig::default(), // XGBoost disabled by default
            output_heads: OutputHeadsConfig {
                price_levels: PriceLevelHead {
                    enabled: true,
                    bandwidth_size: Some(1.0),     // Default bandwidth size
                    percentiles: Some([0.1, 0.9]), // Default 10th/90th percentiles
                },
                direction: DirectionHead {
                    enabled: true,
                    slope_sensitivity: Some(0.02), // Crypto-optimized for volatility normalization
                    base_threshold: Some(0.12),    // 12% momentum threshold
                    extreme_multiplier: Some(2.5), // 2.5x for better extreme detection
                },
                volatility: VolatilityHead {
                    enabled: true,
                    bandwidth_size: Some(1.2), // Less sensitive for volatility regimes
                    base_threshold: Some(0.15), // 15% ATR threshold
                    extreme_multiplier: Some(1.8), // 1.8x for extreme classes
                },
            },
            quantile_outputs: None, // Disabled by default for backward compatibility
            loss_function: CryptoLossFunction::MSE, // Use explicit MSE default
        }
    }
}

impl ModelConfig {
    /// Validate the model configuration
    pub fn validate(&self) -> Result<(), crate::utils::error::VangaError> {
        // Validate output heads
        if self.output_heads.price_levels.enabled {
            self.output_heads.price_levels.validate()?;
        }
        if self.output_heads.direction.enabled {
            self.output_heads.direction.validate()?;
        }
        if self.output_heads.volatility.enabled {
            self.output_heads.volatility.validate()?;
        }

        // Validate sequence length
        match &self.sequence_length {
            SequenceLengthConfig::Fixed(length) => {
                if *length == 0 {
                    return Err(crate::utils::error::VangaError::config(
                        "Fixed sequence length must be greater than 0",
                    ));
                }
                if *length > 1000 {
                    return Err(crate::utils::error::VangaError::config(
                        "Fixed sequence length should be <= 1000 for reasonable performance",
                    ));
                }
            }
            SequenceLengthConfig::Auto {
                min_length,
                max_length,
            } => {
                if *min_length == 0 {
                    return Err(crate::utils::error::VangaError::config(
                        "Auto sequence min_length must be greater than 0",
                    ));
                }
                if *min_length >= *max_length {
                    return Err(crate::utils::error::VangaError::config(
                        "Auto sequence min_length must be less than max_length",
                    ));
                }
                if *max_length > 1000 {
                    return Err(crate::utils::error::VangaError::config(
                        "Auto sequence max_length should be <= 1000 for reasonable performance",
                    ));
                }
            }
            SequenceLengthConfig::Adaptive => {
                // Adaptive is always valid
            }
        }

        // Validate hidden units
        match &self.hidden_units {
            HiddenUnitsConfig::Fixed(units) => {
                if units.is_empty() {
                    return Err(crate::utils::error::VangaError::config(
                        "Fixed hidden units cannot be empty",
                    ));
                }
                for &unit_count in units {
                    if unit_count == 0 {
                        return Err(crate::utils::error::VangaError::config(
                            "Fixed hidden units must be greater than 0",
                        ));
                    }
                }
            }
            HiddenUnitsConfig::Auto {
                min_units,
                max_units,
            } => {
                if *min_units == 0 {
                    return Err(crate::utils::error::VangaError::config(
                        "Auto hidden min_units must be greater than 0",
                    ));
                }
                if *min_units >= *max_units {
                    return Err(crate::utils::error::VangaError::config(
                        "Auto hidden min_units must be less than max_units",
                    ));
                }
            }
            HiddenUnitsConfig::Pyramid {
                base_units,
                reduction_factor,
            } => {
                if *base_units == 0 {
                    return Err(crate::utils::error::VangaError::config(
                        "Pyramid base_units must be greater than 0",
                    ));
                }
                if *reduction_factor <= 0.0 || *reduction_factor >= 1.0 {
                    return Err(crate::utils::error::VangaError::config(
                        "Pyramid reduction_factor must be between 0.0 and 1.0",
                    ));
                }
            }
        }

        Ok(())
    }
}

impl OutputHeadsConfig {
    /// Calculate total output size needed for all enabled prediction heads
    pub fn calculate_total_output_size(&self) -> usize {
        let mut total_size = 0;

        // Price level classification outputs (5 classes in unified system)
        if self.price_levels.enabled {
            total_size += NUM_CLASSES;
        }

        // Direction prediction outputs (5 classes: DUMP, DOWN, SIDEWAYS, UP, PUMP)
        if self.direction.enabled {
            total_size += NUM_CLASSES;
        }

        // Volatility prediction outputs (5 classes: VERY_LOW, LOW, MEDIUM, HIGH, VERY_HIGH)
        if self.volatility.enabled {
            total_size += NUM_CLASSES; // Unified 5-class volatility output
        }

        // Ensure at least one output
        if total_size == 0 {
            log::warn!("No prediction heads enabled, defaulting to single output");
            total_size = 1;
        }

        total_size
    }

    /// Get output segment information for parsing multi-target predictions
    pub fn get_output_segments(&self) -> OutputSegments {
        let mut segments = OutputSegments::new();
        let mut current_offset = 0;

        if self.price_levels.enabled {
            let size = NUM_CLASSES; // 5 classes: Strong Down, Moderate Down, Neutral, Moderate Up, Strong Up
            segments.price_levels = Some((current_offset, current_offset + size));
            current_offset += size;
        }

        if self.direction.enabled {
            let size = NUM_CLASSES; // 5 classes: Dump, Down, Sideways, Up, Pump
            segments.direction = Some((current_offset, current_offset + size));
            current_offset += size;
        }

        if self.volatility.enabled {
            let size = NUM_CLASSES; // Unified 5-class volatility output
            segments.volatility = Some((current_offset, current_offset + size));
        }

        segments
    }
}

/// Output segment information for parsing multi-target predictions
#[derive(Debug, Clone)]
pub struct OutputSegments {
    /// Price levels segment: (start_idx, end_idx)
    pub price_levels: Option<(usize, usize)>,
    /// Direction segment: (start_idx, end_idx)
    pub direction: Option<(usize, usize)>,
    /// Volatility segment: (start_idx, end_idx)
    pub volatility: Option<(usize, usize)>,
}

impl Default for OutputSegments {
    fn default() -> Self {
        Self::new()
    }
}

impl OutputSegments {
    pub fn new() -> Self {
        Self {
            price_levels: None,
            direction: None,
            volatility: None,
        }
    }
}

/// Default implementation for XGBoostConfig following crypto-optimized settings
impl Default for XGBoostConfig {
    fn default() -> Self {
        Self {
            enabled: false,                            // Disabled by default like attention
            feature_dim: 64,                           // k=64 as per paper
            n_estimators: 100,                         // Moderate number of trees
            max_depth: 6,                              // Good balance for crypto complexity
            learning_rate: 0.1,                        // Conservative learning rate
            subsample: 0.8,                            // Row sampling to prevent overfitting
            colsample_bytree: 0.8,                     // Column sampling
            reg_alpha: 0.0,                            // L1 regularization (disabled by default)
            reg_lambda: 1.0,                           // L2 regularization (λ in equation 11)
            early_stopping_rounds: Some(10),           // Early stopping for efficiency
            eval_metric: "rmse".to_string(),           // Default regression metric
            objective: "reg:squarederror".to_string(), // Default regression objective
            save_feature_importance: true,             // Enable feature analysis
            importance_type: "gain".to_string(),       // Information gain importance
        }
    }
}
