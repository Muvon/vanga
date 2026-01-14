use crate::model::bias_correction::BiasCorrection;
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

/// Position of LayerNorm relative to LSTM activation
///
/// Pre-LN: Normalize BEFORE the LSTM layer (better gradient flow for deep networks)
/// Post-LN: Normalize AFTER the LSTM layer (standard for LSTMs, Ba et al., 2016)
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum LayerNormPosition {
    Pre,
    Post,
}

impl Default for LayerNormPosition {
    fn default() -> Self {
        LayerNormPosition::Post
    }
}

impl std::fmt::Display for LayerNormPosition {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LayerNormPosition::Pre => write!(f, "pre"),
            LayerNormPosition::Post => write!(f, "post"),
        }
    }
}

impl std::str::FromStr for LayerNormPosition {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "pre" => Ok(LayerNormPosition::Pre),
            "post" => Ok(LayerNormPosition::Post),
            _ => Err(format!(
                "Invalid LayerNormPosition '{}'. Must be 'pre' or 'post'",
                s
            )),
        }
    }
}

/// Layer Normalization configuration for stabilizing LSTM training
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerNormConfig {
    /// Enable layer normalization
    pub enabled: bool,
    /// Epsilon value for numerical stability (default: 1e-5)
    pub epsilon: f64,
    /// Apply layer norm to LSTM cell outputs
    pub lstm_cell: bool,
    /// Apply layer norm before or after LSTM activation
    /// "pre" = before (better gradient flow for deep networks)
    /// "post" = after (standard for LSTMs)
    pub position: LayerNormPosition,
}

impl Default for LayerNormConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            epsilon: 1e-5,
            lstm_cell: true,
            position: LayerNormPosition::Post, // post-norm is standard for LSTMs
        }
    }
}

/// Model configuration for LSTM architecture
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// LSTM architecture type
    pub architecture: LSTMArchitecture,

    /// Sequence length for LSTM input
    pub sequence_length: SequenceLengthConfig,

    /// Hidden units configuration
    pub hidden_units: HiddenUnitsConfig,

    /// Layer normalization configuration
    #[serde(default)]
    pub layer_norm: LayerNormConfig,

    /// Dropout configuration
    pub dropout: DropoutConfig,

    /// Attention mechanism configuration
    pub attention: AttentionConfig,

    /// XGBoost hybrid model configuration
    pub xgboost: XGBoostConfig,

    /// TFT Quantile regression configuration
    pub quantile_outputs: Option<TFTQuantileOutputConfig>,

    /// Bias correction configuration
    pub bias_correction: BiasCorrection,
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

impl Default for SequenceLengthConfig {
    fn default() -> Self {
        Self::Auto {
            min_length: 24,
            max_length: 168,
        }
    }
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

impl Default for HiddenUnitsConfig {
    fn default() -> Self {
        Self::Auto {
            min_units: 32,
            max_units: 256,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DropoutRate {
    Auto { min_rate: f64, max_rate: f64 },
    Fixed(f64),
    Adaptive,
}

impl Default for DropoutRate {
    fn default() -> Self {
        Self::Fixed(0.2)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DropoutConfig {
    pub enabled: bool,
    pub rate: DropoutRate,
    pub variational: bool,
    pub recurrent: bool,
}

impl Default for DropoutConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            rate: DropoutRate::Auto {
                min_rate: 0.1,
                max_rate: 0.5,
            },
            variational: true, // Enable variational dropout by default for LSTM
            recurrent: true,   // Enable recurrent dropout by default for LSTM
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionConfig {
    pub enabled: bool,
    pub mechanism: AttentionMechanism,
    pub heads: u32,
    pub head_dim: Option<u32>,              // Auto-optimized if None
    pub dropout_rate: f64,                  // Attention dropout rate
    pub dropout_weights: bool,              // Apply dropout to attention weights
    pub dropout_output: bool,               // Apply dropout to attention output
    pub dropout_projections: bool,          // Apply dropout to Q, K, V projections
    pub dropout_scores: bool,               // Apply dropout to final attention scores
    pub temperature_scaling: f64,           // Crypto volatility adaptation
    pub use_relative_position: bool,        // Temporal modeling for crypto
    pub visualization: VisualizationConfig, // Analysis options
    /// Mixture-of-Head Attention configuration (only used when mechanism = MixtureOfHeads)
    pub moh: Option<MoHConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum AttentionMechanism {
    SelfAttention,
    MultiHeadAttention,
    AdditiveAttention,
    /// TFT Variable Selection Attention (builds on MultiHeadAttention)
    VariableSelection,
    /// Mixture-of-Head Attention (MoH) - Dynamic head routing for efficiency
    MixtureOfHeads,
    None,
}

/// SmartCore hybrid model configuration (maintains XGBoost name for compatibility)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct XGBoostConfig {
    /// Enable/disable SmartCore hybrid mode
    pub enabled: bool,

    /// LSTM feature extraction dimension (k in paper, typically 64)
    pub feature_dim: usize,

    /// SmartCore hyperparameters
    pub n_estimators: usize, // Number of trees in Random Forest
    pub max_depth: usize, // Maximum tree depth

    /// SmartCore algorithm and evaluation
    pub objective: String, // "RandomForest" or "DecisionTree"
    pub eval_metric: String, // "multiclass_accuracy", etc.

    /// Feature importance analysis
    pub save_feature_importance: bool,
    pub importance_method: String, // "permutation" for SmartCore
}

/// Mixture-of-Head Attention configuration for dynamic head routing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MoHConfig {
    /// Total number of attention heads (h in paper)
    pub total_heads: u32,
    /// Number of always-active shared heads (hs in paper)
    pub shared_heads: u32,
    /// Number of routed heads to activate via top-K selection (K in paper)
    pub top_k: u32,
    /// Weight for load balance loss (β in paper, default: 0.01)
    pub load_balance_weight: f64,
    /// Temperature for routing softmax (higher = more exploration)
    pub routing_temperature: f64,
    /// Enable routing score logging for analysis
    pub log_routing_decisions: bool,

    // ===== VOLATILITY-ADAPTIVE ROUTING =====
    /// Enable volatility-adaptive routing temperature
    #[serde(default)]
    pub volatility_adaptive: bool,

    /// Volatility multiplier for temperature scaling (default: 0.5)
    #[serde(default = "default_volatility_multiplier")]
    pub volatility_multiplier: f64,

    /// Volatility smoothing window (default: 10 timesteps)
    #[serde(default = "default_volatility_window")]
    pub volatility_window: usize,

    // ===== SPARSE ATTENTION =====
    /// Enable sparse attention mechanism
    #[serde(default)]
    pub sparse_attention: bool,

    /// Minimum sparsity ratio in high volatility (default: 0.3 = 30% tokens)
    #[serde(default = "default_min_sparse_ratio")]
    pub min_sparse_ratio: f32,

    /// Maximum sparsity ratio in low volatility (default: 0.7 = 70% tokens)
    #[serde(default = "default_max_sparse_ratio")]
    pub max_sparse_ratio: f32,

    // ===== DEFORMABLE ATTENTION =====
    /// Enable deformable attention with learnable offsets
    #[serde(default)]
    pub deformable_attention: bool,

    /// Number of learnable offset positions per head (default: 8)
    #[serde(default = "default_num_offsets")]
    pub num_offsets: usize,

    /// Enable importance-based token sampling
    #[serde(default)]
    pub learnable_sampling: bool,
}

// Default value functions
fn default_volatility_multiplier() -> f64 {
    0.5
}
fn default_volatility_window() -> usize {
    10
}
fn default_min_sparse_ratio() -> f32 {
    0.3
}
fn default_max_sparse_ratio() -> f32 {
    0.7
}
fn default_num_offsets() -> usize {
    8
}

impl Default for MoHConfig {
    fn default() -> Self {
        Self {
            total_heads: 16,              // More heads for better specialization
            shared_heads: 4,              // 25% shared heads for common patterns
            top_k: 4,                     // 25% routed heads active (50% total active)
            load_balance_weight: 0.01,    // Standard β from paper
            routing_temperature: 1.0,     // Standard temperature
            log_routing_decisions: false, // Disabled by default for performance

            // Advanced features disabled by default for backward compatibility
            volatility_adaptive: false,
            volatility_multiplier: 0.5,
            volatility_window: 10,

            sparse_attention: false,
            min_sparse_ratio: 0.3,
            max_sparse_ratio: 0.7,

            deformable_attention: false,
            num_offsets: 8,
            learnable_sampling: false,
        }
    }
}

impl MoHConfig {
    /// Validate MoH configuration parameters
    pub fn validate(&self) -> Result<(), String> {
        // Existing validations
        if self.shared_heads + self.top_k > self.total_heads {
            return Err(format!(
                "shared_heads ({}) + top_k ({}) cannot exceed total_heads ({})",
                self.shared_heads, self.top_k, self.total_heads
            ));
        }

        if self.shared_heads == 0 && self.top_k == 0 {
            return Err("At least one head must be active (shared_heads + top_k > 0)".to_string());
        }

        if self.load_balance_weight < 0.0 || self.load_balance_weight > 1.0 {
            return Err(format!(
                "load_balance_weight ({}) must be between 0.0 and 1.0",
                self.load_balance_weight
            ));
        }

        if self.routing_temperature <= 0.0 {
            return Err(format!(
                "routing_temperature ({}) must be positive",
                self.routing_temperature
            ));
        }

        // New validations for advanced features
        if self.volatility_multiplier < 0.0 || self.volatility_multiplier > 2.0 {
            return Err(format!(
                "volatility_multiplier must be in [0, 2], got {}",
                self.volatility_multiplier
            ));
        }

        if self.volatility_window < 2 {
            return Err(format!(
                "volatility_window must be >= 2, got {}",
                self.volatility_window
            ));
        }

        if self.min_sparse_ratio < 0.1 || self.min_sparse_ratio > 1.0 {
            return Err(format!(
                "min_sparse_ratio must be in [0.1, 1.0], got {}",
                self.min_sparse_ratio
            ));
        }

        if self.max_sparse_ratio < self.min_sparse_ratio || self.max_sparse_ratio > 1.0 {
            return Err(format!(
                "max_sparse_ratio must be in [min_sparse_ratio, 1.0], got {}",
                self.max_sparse_ratio
            ));
        }

        if self.deformable_attention && self.num_offsets < 2 {
            return Err(format!(
                "num_offsets must be >= 2 when deformable_attention enabled, got {}",
                self.num_offsets
            ));
        }

        Ok(())
    }

    /// Check if any advanced features are enabled
    pub fn has_advanced_features(&self) -> bool {
        self.volatility_adaptive
            || self.sparse_attention
            || self.deformable_attention
            || self.learnable_sampling
    }

    /// Get the total number of active heads (shared + routed)
    pub fn active_heads(&self) -> u32 {
        self.shared_heads + self.top_k
    }

    /// Get the number of inactive heads
    pub fn inactive_heads(&self) -> u32 {
        self.total_heads - self.active_heads()
    }

    /// Calculate efficiency ratio (active/total heads)
    pub fn efficiency_ratio(&self) -> f64 {
        self.active_heads() as f64 / self.total_heads as f64
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualizationConfig {
    pub save_heatmaps: bool,
    pub export_analysis: bool,
    pub output_dir: String,
}

impl Default for AttentionConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            mechanism: AttentionMechanism::MultiHeadAttention,
            heads: 8,                    // Auto-optimized default for crypto sequences
            head_dim: Some(64),          // Optimal for most crypto features (50-100)
            dropout_rate: 0.1,           // Conservative dropout for attention
            dropout_weights: true,       // Apply dropout to attention weights (standard)
            dropout_output: true,        // Apply dropout to attention output (recommended)
            dropout_projections: true,   // Apply dropout to Q, K, V projections (comprehensive)
            dropout_scores: true, // Apply dropout to final attention scores (additional regularization)
            temperature_scaling: 1.0, // Standard temperature
            use_relative_position: true, // Better for time series
            visualization: VisualizationConfig::default(),
            moh: None, // MoH config only when mechanism = MixtureOfHeads
        }
    }
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

/// Validate the model configuration
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
            // Layer Normalization for deep LSTM stabilization (Ba et al., 2016)
            // Enables faster convergence and better gradient flow in deep networks
            layer_norm: LayerNormConfig {
                enabled: false, // Disabled by default for backward compatibility
                epsilon: 1e-5,
                lstm_cell: true,
                position: LayerNormPosition::Post,
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
                dropout_weights: true,       // Apply dropout to attention weights
                dropout_output: true,        // Apply dropout to attention output
                dropout_projections: true,   // Apply dropout to Q, K, V projections
                dropout_scores: true,        // Apply dropout to final attention scores
                temperature_scaling: 1.0,    // Standard temperature
                use_relative_position: true, // Better for time series
                visualization: VisualizationConfig::default(),
                moh: None, // No MoH for default config
            },
            xgboost: XGBoostConfig::default(), // XGBoost disabled by default
            quantile_outputs: None,            // Disabled by default for backward compatibility
            bias_correction: BiasCorrection::default(), // Bias correction with default settings
        }
    }
}

impl ModelConfig {
    /// Validate the model configuration
    pub fn validate(&self) -> Result<(), crate::utils::error::VangaError> {
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

        // Validate LayerNormConfig position
        if self.layer_norm.enabled {
            // Position is now a type-safe enum, so it's always valid
            // Just log for debugging
            log::debug!(
                "LayerNorm enabled with position: {:?}",
                self.layer_norm.position
            );
        }

        Ok(())
    }
}

/// Default implementation for SmartCore config (maintains XGBoost name for compatibility)
impl Default for XGBoostConfig {
    fn default() -> Self {
        Self {
            enabled: false,                                 // Disabled by default like attention
            feature_dim: 64,                                // k=64 as per paper
            n_estimators: 100,                              // Number of trees in Random Forest
            max_depth: 6,                                   // Good balance for crypto complexity
            objective: "RandomForest".to_string(),          // SmartCore algorithm
            eval_metric: "multiclass_accuracy".to_string(), // SmartCore evaluation metric
            save_feature_importance: true,                  // Enable feature analysis
            importance_method: "permutation".to_string(),   // Use permutation-based importance
        }
    }
}
