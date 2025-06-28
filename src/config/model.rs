use serde::{Deserialize, Serialize};

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
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AttentionMechanism {
    SelfAttention,
    MultiHeadAttention,
    AdditiveAttention,
    None,
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
        }
    }
}
