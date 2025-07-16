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
