use crate::model::loss::CryptoLossFunction;
use serde::{Deserialize, Serialize};

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
    pub bins: u32,
    pub range_percent: f64,
    pub distribution_type: DistributionType,
    pub target_strategy: PriceLevelTargetStrategy,
}

impl PriceLevelHead {
    /// Validate the price level head configuration
    pub fn validate(&self) -> Result<(), crate::utils::error::VangaError> {
        if self.bins < 2 {
            return Err(crate::utils::error::VangaError::config(
                "Price level bins must be at least 2",
            ));
        }
        if self.bins > 50 {
            return Err(crate::utils::error::VangaError::config(
                "Price level bins should be <= 50 for reasonable performance",
            ));
        }
        if self.range_percent <= 0.0 {
            return Err(crate::utils::error::VangaError::config(
                "Price level range_percent must be greater than 0.0",
            ));
        }
        if self.range_percent > 1.0 {
            return Err(crate::utils::error::VangaError::config(
                "Price level range_percent should be <= 1.0 (100%)",
            ));
        }

        // Validate the target strategy
        self.target_strategy.validate()?;

        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DirectionHead {
    pub enabled: bool,
    pub threshold: f64,
    pub confidence_calibration: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VolatilityHead {
    pub enabled: bool,
    pub method: VolatilityPredictionMethod,
    pub horizons: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DistributionType {
    Categorical,
    Beta,
    Dirichlet,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum PriceLevelTargetStrategy {
    /// Current approach: single future price point
    Current,

    /// Standard VWAP over horizon period
    StandardVwap,

    /// Momentum-aware VWAP with directional bias
    MomentumVwap {
        momentum_window: usize,
        bias_strength: f64,
    },
}

impl Default for PriceLevelTargetStrategy {
    fn default() -> Self {
        Self::Current // Backward compatibility
    }
}

impl PriceLevelTargetStrategy {
    /// Validate the target strategy configuration
    pub fn validate(&self) -> Result<(), crate::utils::error::VangaError> {
        match self {
            Self::Current => Ok(()),
            Self::StandardVwap => Ok(()),
            Self::MomentumVwap {
                momentum_window,
                bias_strength,
            } => {
                if *momentum_window == 0 {
                    return Err(crate::utils::error::VangaError::config(
                        "MomentumVwap momentum_window must be greater than 0",
                    ));
                }
                if *momentum_window > 100 {
                    return Err(crate::utils::error::VangaError::config(
                        "MomentumVwap momentum_window should be <= 100 for reasonable performance",
                    ));
                }
                if *bias_strength < 0.0 || *bias_strength > 1.0 {
                    return Err(crate::utils::error::VangaError::config(
                        "MomentumVwap bias_strength must be between 0.0 and 1.0",
                    ));
                }
                Ok(())
            }
        }
    }

    /// Get a human-readable description of the strategy
    pub fn description(&self) -> &'static str {
        match self {
            Self::Current => "Current: Uses simple future price point",
            Self::StandardVwap => "Standard VWAP: Volume-weighted average price over horizon",
            Self::MomentumVwap { .. } => "Momentum VWAP: Momentum-aware VWAP with directional bias",
        }
    }
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
            output_heads: OutputHeadsConfig {
                price_levels: PriceLevelHead {
                    enabled: true,
                    bins: 10,
                    range_percent: 5.0,
                    distribution_type: DistributionType::Categorical,
                    target_strategy: PriceLevelTargetStrategy::default(),
                },
                direction: DirectionHead {
                    enabled: true,
                    threshold: 0.01,
                    confidence_calibration: true,
                },
                volatility: VolatilityHead {
                    enabled: true,
                    method: VolatilityPredictionMethod::Direct,
                    horizons: vec!["1h".to_string(), "4h".to_string(), "24h".to_string()],
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

        // Price level classification outputs (softmax probabilities)
        if self.price_levels.enabled {
            total_size += self.price_levels.bins as usize;
        }

        // Direction prediction outputs (3 classes: DOWN, SIDEWAYS, UP)
        if self.direction.enabled {
            total_size += 3;
        }

        // Volatility prediction outputs (one per horizon)
        if self.volatility.enabled {
            total_size += self.volatility.horizons.len();
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
            let size = self.price_levels.bins as usize;
            segments.price_levels = Some((current_offset, current_offset + size));
            current_offset += size;
        }

        if self.direction.enabled {
            let size = 3; // DOWN, SIDEWAYS, UP
            segments.direction = Some((current_offset, current_offset + size));
            current_offset += size;
        }

        if self.volatility.enabled {
            let size = self.volatility.horizons.len();
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
