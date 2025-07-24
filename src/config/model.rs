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
    pub bandwidth_size: Option<f64>, // Optional for backward compatibility
    pub distribution_type: DistributionType,
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

        // Sequence-aware classification always uses 6 bins (fixed architecture)
        // bandwidth_size only affects breakout sensitivity, not number of bins
        // No validation needed for bins since it's architecturally fixed

        Ok(())
    }
}

impl DirectionHead {
    /// Validate the direction head configuration
    pub fn validate(&self) -> Result<(), crate::utils::error::VangaError> {
        let (down_threshold, up_threshold) = self.thresholds;
        if down_threshold >= 0.0 {
            return Err(crate::utils::error::VangaError::config(
                "Direction down_threshold must be negative",
            ));
        }
        if up_threshold <= 0.0 {
            return Err(crate::utils::error::VangaError::config(
                "Direction up_threshold must be positive",
            ));
        }
        if down_threshold.abs() != up_threshold {
            log::warn!(
                "Direction thresholds are not symmetric: {} vs {}",
                down_threshold,
                up_threshold
            );
        }
        Ok(())
    }
}

impl VolatilityHead {
    /// Validate the volatility head configuration
    pub fn validate(&self) -> Result<(), crate::utils::error::VangaError> {
        let (low_percentile, high_percentile) = self.thresholds;
        if low_percentile <= 0.0 || low_percentile >= 1.0 {
            return Err(crate::utils::error::VangaError::config(
                "Volatility low_percentile must be between 0.0 and 1.0",
            ));
        }
        if high_percentile <= 0.0 || high_percentile >= 1.0 {
            return Err(crate::utils::error::VangaError::config(
                "Volatility high_percentile must be between 0.0 and 1.0",
            ));
        }
        if low_percentile >= high_percentile {
            return Err(crate::utils::error::VangaError::config(
                "Volatility low_percentile must be less than high_percentile",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DirectionHead {
    pub enabled: bool,
    pub thresholds: (f64, f64), // (down_threshold, up_threshold) for consistency
    pub confidence_calibration: bool,
    pub use_adaptive_thresholds: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VolatilityHead {
    pub enabled: bool,
    pub method: VolatilityPredictionMethod,
    pub horizons: Vec<String>,
    pub thresholds: (f64, f64), // (low_percentile, high_percentile) for consistency
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
                    bandwidth_size: Some(1.0), // Default bandwidth size
                    distribution_type: DistributionType::Categorical,
                },
                direction: DirectionHead {
                    enabled: true,
                    thresholds: (-0.01, 0.01), // 1% symmetric thresholds
                    confidence_calibration: true,
                    use_adaptive_thresholds: true,
                },
                volatility: VolatilityHead {
                    enabled: true,
                    method: VolatilityPredictionMethod::Direct,
                    horizons: vec!["1h".to_string(), "4h".to_string(), "24h".to_string()],
                    thresholds: (0.33, 0.67), // 33rd and 67th percentiles
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

        // Price level classification outputs (6 bins for sequence-aware classification)
        if self.price_levels.enabled {
            total_size += 6; // Fixed: sequence-aware classification always uses 6 bins
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
            let size = 6; // Fixed: sequence-aware classification always uses 6 bins
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
