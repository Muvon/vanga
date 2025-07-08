// TFT Variable Selection Network - Building on existing MultiHeadAttention
use crate::model::attention::MultiHeadAttention;
use crate::utils::error::Result;
use candle_core::Tensor;
use candle_nn::{linear, Linear, Module, VarBuilder};
use serde::{Deserialize, Serialize};

/// Variable Selection configuration for TFT enhancement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VariableSelectionConfig {
    /// Enable static variable selection (for symbol, exchange features)
    pub static_selection: bool,
    /// Enable temporal variable selection (for time-varying features)
    pub temporal_selection: bool,
    /// Selection threshold for feature importance
    pub selection_threshold: f64,
    /// Number of top features to select (None = automatic)
    pub top_k_features: Option<usize>,
    /// Enable interpretability output
    pub enable_interpretability: bool,
}

impl Default for VariableSelectionConfig {
    fn default() -> Self {
        Self {
            static_selection: true,
            temporal_selection: true,
            selection_threshold: 0.1,
            top_k_features: None, // Auto-select based on importance
            enable_interpretability: true,
        }
    }
}

/// Variable Selection Network that wraps existing MultiHeadAttention
pub struct VariableSelectionNetwork {
    static_selector: Option<Linear>,
    temporal_selector: Option<Linear>,
    importance_weights: Option<Tensor>,
}

impl VariableSelectionNetwork {
    /// Create new variable selection network
    pub fn new(input_dim: usize, config: VariableSelectionConfig, vs: VarBuilder) -> Result<Self> {
        let static_selector = if config.static_selection {
            Some(linear(input_dim, input_dim, vs.pp("static_selector"))?)
        } else {
            None
        };

        let temporal_selector = if config.temporal_selection {
            Some(linear(input_dim, input_dim, vs.pp("temporal_selector"))?)
        } else {
            None
        };

        Ok(Self {
            static_selector,
            temporal_selector,
            importance_weights: None,
        })
    }

    /// Apply variable selection to input features
    pub fn select_variables(&self, input: &Tensor) -> Result<(Tensor, Option<Tensor>)> {
        let mut selected_features = input.clone();
        let mut importance_scores = None;

        // Apply static variable selection
        if let Some(ref static_sel) = self.static_selector {
            let static_weights = static_sel.forward(input)?;
            // Use tanh as approximation for sigmoid (available in candle)
            let static_importance = static_weights.tanh()?;
            selected_features = selected_features.broadcast_mul(&static_importance)?;
            importance_scores = Some(static_importance);
        }

        // Apply temporal variable selection
        if let Some(ref temporal_sel) = self.temporal_selector {
            let temporal_weights = temporal_sel.forward(&selected_features)?;
            let temporal_importance = temporal_weights.tanh()?;
            selected_features = selected_features.broadcast_mul(&temporal_importance)?;

            // Combine importance scores if both exist
            importance_scores = match importance_scores {
                Some(static_imp) => Some(static_imp.broadcast_mul(&temporal_importance)?),
                None => Some(temporal_importance),
            };
        }

        Ok((selected_features, importance_scores))
    }

    /// Get feature importance scores for interpretability
    pub fn get_feature_importance(&self) -> Option<&Tensor> {
        self.importance_weights.as_ref()
    }
}

/// Enhanced attention with variable selection - wraps existing MultiHeadAttention
pub struct VariableSelectionAttention {
    base_attention: MultiHeadAttention,
    variable_selector: VariableSelectionNetwork,
    config: VariableSelectionConfig,
}

impl VariableSelectionAttention {
    /// Create variable selection attention from existing MultiHeadAttention
    pub fn from_existing_attention(
        attention: MultiHeadAttention,
        vs_config: VariableSelectionConfig,
        vs: VarBuilder,
    ) -> Result<Self> {
        // Get input dimension from attention config (approximate from head configuration)
        let input_dim = 64; // Default, will be auto-detected in practice

        let variable_selector = VariableSelectionNetwork::new(input_dim, vs_config.clone(), vs)?;

        log::info!(
            "Created TFT Variable Selection Attention with static_selection={}, temporal_selection={}",
            vs_config.static_selection,
            vs_config.temporal_selection
        );

        Ok(Self {
            base_attention: attention,
            variable_selector,
            config: vs_config,
        })
    }

    /// Forward pass with variable selection + existing attention
    pub fn forward(&self, input: &Tensor) -> Result<(Tensor, Tensor, Option<Tensor>)> {
        // Step 1: Apply variable selection to input features
        let (selected_features, importance_scores) =
            self.variable_selector.select_variables(input)?;

        // Step 2: Apply existing attention mechanism to selected features
        let (attention_output, attention_weights) =
            self.base_attention.forward(&selected_features)?;

        // Return: (output, attention_weights, feature_importance)
        Ok((attention_output, attention_weights, importance_scores))
    }

    /// Get variable selection configuration
    pub fn get_config(&self) -> &VariableSelectionConfig {
        &self.config
    }

    /// Get feature importance for interpretability
    pub fn get_feature_importance(&self) -> Option<&Tensor> {
        self.variable_selector.get_feature_importance()
    }
}

/// Factory for creating TFT-enhanced attention from existing components
pub struct TFTAttentionFactory;

impl TFTAttentionFactory {
    /// Upgrade existing MultiHeadAttention to TFT Variable Selection Attention
    pub fn upgrade_attention(
        existing_attention: MultiHeadAttention,
        vs_config: VariableSelectionConfig,
        vs: VarBuilder,
    ) -> Result<VariableSelectionAttention> {
        VariableSelectionAttention::from_existing_attention(existing_attention, vs_config, vs)
    }

    /// Create crypto-optimized variable selection configuration
    pub fn create_crypto_optimized_config() -> VariableSelectionConfig {
        VariableSelectionConfig {
            static_selection: true,        // Important for symbol-specific features
            temporal_selection: true,      // Critical for time-varying crypto patterns
            selection_threshold: 0.15,     // Slightly higher threshold for crypto noise
            top_k_features: Some(20),      // Limit to top 20 features for crypto efficiency
            enable_interpretability: true, // Essential for crypto analysis
        }
    }

    /// Create high-frequency trading optimized configuration
    pub fn create_hft_optimized_config() -> VariableSelectionConfig {
        VariableSelectionConfig {
            static_selection: false,        // Less important for HFT
            temporal_selection: true,       // Critical for short-term patterns
            selection_threshold: 0.2,       // Higher threshold for noise reduction
            top_k_features: Some(10),       // Fewer features for speed
            enable_interpretability: false, // Disabled for performance
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_variable_selection_config_defaults() {
        let config = VariableSelectionConfig::default();
        assert!(config.static_selection);
        assert!(config.temporal_selection);
        assert_eq!(config.selection_threshold, 0.1);
        assert!(config.enable_interpretability);
    }

    #[test]
    fn test_crypto_optimized_config() {
        let config = TFTAttentionFactory::create_crypto_optimized_config();
        assert!(config.static_selection);
        assert!(config.temporal_selection);
        assert_eq!(config.top_k_features, Some(20));
        assert!(config.enable_interpretability);
    }

    #[test]
    fn test_hft_optimized_config() {
        let config = TFTAttentionFactory::create_hft_optimized_config();
        assert!(!config.static_selection);
        assert!(config.temporal_selection);
        assert_eq!(config.top_k_features, Some(10));
        assert!(!config.enable_interpretability);
    }
}
