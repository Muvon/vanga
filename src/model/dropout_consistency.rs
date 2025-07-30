//! Dropout Consistency Configuration
//!
//! This module provides configuration options for handling dropout consistency
//! between training and validation phases to prevent validation loss jumping.

use serde::{Deserialize, Serialize};

/// Dropout consistency strategy for training vs validation
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum DropoutConsistencyStrategy {
    /// Standard approach: dropout enabled in training, disabled in validation
    Standard,
    /// Consistent approach: same dropout behavior in both training and validation
    Consistent,
    /// Validation-only: no dropout in training, enabled in validation (for testing)
    ValidationOnly,
    /// Disabled: no dropout in either training or validation
    Disabled,
}

/// Attention dropout component types
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AttentionDropoutComponent {
    /// Dropout applied to attention weights after softmax
    AttentionWeights,
    /// Dropout applied to attention output after projection
    AttentionOutput,
    /// Dropout applied to Q, K, V projections
    Projections,
}

impl Default for DropoutConsistencyStrategy {
    fn default() -> Self {
        Self::Standard // Keep existing behavior as default
    }
}

/// Enhanced dropout configuration with consistency options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DropoutConsistencyConfig {
    /// Dropout consistency strategy
    pub strategy: DropoutConsistencyStrategy,
    /// Whether to log dropout behavior changes
    pub log_dropout_changes: bool,
    /// Whether to warn about potential validation inconsistencies
    pub warn_validation_inconsistency: bool,
    /// Attention-specific dropout configuration
    pub attention_dropout_config: AttentionDropoutConfig,
}

/// Attention-specific dropout configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionDropoutConfig {
    /// Whether to apply dropout to attention weights
    pub apply_to_attention_weights: bool,
    /// Whether to apply dropout to attention output
    pub apply_to_attention_output: bool,
    /// Whether to apply dropout to attention projections (Q, K, V)
    pub apply_to_projections: bool,
    /// Scale factor for attention dropout relative to base dropout rate
    pub attention_dropout_scale: f64,
}

impl Default for AttentionDropoutConfig {
    fn default() -> Self {
        Self {
            apply_to_attention_weights: true, // Standard practice
            apply_to_attention_output: true,  // Additional regularization
            apply_to_projections: true,       // Comprehensive regularization
            attention_dropout_scale: 1.0,     // Same rate as base dropout
        }
    }
}

impl Default for DropoutConsistencyConfig {
    fn default() -> Self {
        Self {
            strategy: DropoutConsistencyStrategy::Standard,
            log_dropout_changes: false, // DISABLED by default to prevent log spam
            warn_validation_inconsistency: false, // DISABLED by default to prevent log spam
            attention_dropout_config: AttentionDropoutConfig::default(),
        }
    }
}

impl DropoutConsistencyConfig {
    /// Create configuration with detailed logging enabled (for debugging)
    pub fn with_detailed_logging() -> Self {
        Self {
            strategy: DropoutConsistencyStrategy::Standard,
            log_dropout_changes: true,
            warn_validation_inconsistency: true,
            attention_dropout_config: AttentionDropoutConfig::default(),
        }
    }

    /// Create configuration optimized for consistent validation behavior
    pub fn with_consistent_validation() -> Self {
        Self {
            strategy: DropoutConsistencyStrategy::Consistent,
            log_dropout_changes: false,
            warn_validation_inconsistency: false,
            attention_dropout_config: AttentionDropoutConfig::default(),
        }
    }
    /// Determine if dropout should be applied based on phase and strategy
    pub fn should_apply_dropout(&self, is_training: bool, dropout_enabled: bool) -> bool {
        if !dropout_enabled {
            return false; // Dropout disabled in model config
        }

        match self.strategy {
            DropoutConsistencyStrategy::Standard => is_training,
            DropoutConsistencyStrategy::Consistent => true, // Always apply
            DropoutConsistencyStrategy::ValidationOnly => !is_training,
            DropoutConsistencyStrategy::Disabled => false, // Never apply
        }
    }

    /// Validate dropout configuration consistency across model components
    pub fn validate_dropout_config(
        &self,
        lstm_dropout_enabled: bool,
        attention_dropout_rate: Option<f64>,
        component_name: &str,
    ) -> Result<(), String> {
        // Check for inconsistent dropout configurations
        if lstm_dropout_enabled && attention_dropout_rate.is_some() {
            let attention_rate = attention_dropout_rate.unwrap();
            if attention_rate <= 0.0 || attention_rate >= 1.0 {
                return Err(format!(
                    "Invalid attention dropout rate {:.3} in {}: must be between 0.0 and 1.0",
                    attention_rate, component_name
                ));
            }
        }

        // REMOVED: Strategy warning spam - this was causing log pollution
        // The Standard strategy is the default and expected behavior
        // Users can change to Consistent strategy if they want same dropout in validation

        Ok(())
    }

    /// Get recommended dropout rate based on training phase and strategy
    pub fn get_effective_dropout_rate(&self, base_rate: f64, is_training: bool) -> f64 {
        if !self.should_apply_dropout(is_training, base_rate > 0.0) {
            return 0.0;
        }

        match self.strategy {
            DropoutConsistencyStrategy::Standard => {
                if is_training {
                    base_rate
                } else {
                    0.0
                }
            }
            DropoutConsistencyStrategy::Consistent => base_rate,
            DropoutConsistencyStrategy::ValidationOnly => {
                if is_training {
                    0.0
                } else {
                    base_rate
                }
            }
            DropoutConsistencyStrategy::Disabled => 0.0,
        }
    }

    /// Get effective attention dropout rate with attention-specific scaling
    pub fn get_effective_attention_dropout_rate(&self, base_rate: f64, is_training: bool) -> f64 {
        let base_effective_rate = self.get_effective_dropout_rate(base_rate, is_training);
        base_effective_rate * self.attention_dropout_config.attention_dropout_scale
    }

    /// Check if dropout should be applied to specific attention component
    pub fn should_apply_attention_dropout(
        &self,
        component: AttentionDropoutComponent,
        is_training: bool,
        base_dropout_rate: f64,
    ) -> bool {
        if !self.should_apply_dropout(is_training, base_dropout_rate > 0.0) {
            return false;
        }

        match component {
            AttentionDropoutComponent::AttentionWeights => {
                self.attention_dropout_config.apply_to_attention_weights
            }
            AttentionDropoutComponent::AttentionOutput => {
                self.attention_dropout_config.apply_to_attention_output
            }
            AttentionDropoutComponent::Projections => {
                self.attention_dropout_config.apply_to_projections
            }
        }
    }

    /// Validate comprehensive dropout configuration including attention
    pub fn validate_comprehensive_dropout_config(
        &self,
        lstm_dropout_enabled: bool,
        lstm_dropout_rate: Option<f64>,
        attention_dropout_rate: Option<f64>,
        component_name: &str,
    ) -> Result<(), String> {
        // Validate LSTM dropout
        if let Some(rate) = lstm_dropout_rate {
            if rate <= 0.0 || rate >= 1.0 {
                return Err(format!(
                    "Invalid LSTM dropout rate {:.3} in {}: must be between 0.0 and 1.0",
                    rate, component_name
                ));
            }
        }

        // Validate attention dropout
        if let Some(rate) = attention_dropout_rate {
            if rate <= 0.0 || rate >= 1.0 {
                return Err(format!(
                    "Invalid attention dropout rate {:.3} in {}: must be between 0.0 and 1.0",
                    rate, component_name
                ));
            }
        }

        // Validate attention dropout scale
        if self.attention_dropout_config.attention_dropout_scale <= 0.0 {
            return Err(format!(
                "Invalid attention dropout scale {:.3} in {}: must be > 0.0",
                self.attention_dropout_config.attention_dropout_scale, component_name
            ));
        }

        // Check for potential inconsistencies
        if lstm_dropout_enabled && attention_dropout_rate.is_some() {
            let lstm_rate = lstm_dropout_rate.unwrap_or(0.2);
            let attention_rate = attention_dropout_rate.unwrap();
            let rate_diff = (lstm_rate - attention_rate).abs();

            if rate_diff > 0.3 {
                log::warn!(
                    "⚠️ Large dropout rate difference in {}: LSTM={:.3}, Attention={:.3}. Consider using similar rates for consistency.",
                    component_name, lstm_rate, attention_rate
                );
            }
        }

        // Strategy-specific warnings - ONLY log once during model initialization, not every forward pass
        if self.strategy == DropoutConsistencyStrategy::Standard
            && self.warn_validation_inconsistency
        {
            // Only warn during model setup, not during training loops
            // This prevents log spam during forward passes
            log::debug!(
                "📋 {} using Standard dropout strategy: dropout disabled during validation (expected behavior)",
                component_name
            );
        }

        Ok(())
    }

    /// Log dropout behavior for debugging
    pub fn log_dropout_behavior(&self, is_training: bool, dropout_applied: bool) {
        if !self.log_dropout_changes {
            return;
        }

        let phase = if is_training {
            "TRAINING"
        } else {
            "VALIDATION"
        };
        let status = if dropout_applied {
            "ENABLED"
        } else {
            "DISABLED"
        };

        log::debug!(
            "🔧 Dropout {} in {} phase (strategy: {:?})",
            status,
            phase,
            self.strategy
        );

        if self.warn_validation_inconsistency
            && self.strategy == DropoutConsistencyStrategy::Standard
            && !is_training
        {
            log::debug!(
                "⚠️ Standard dropout strategy: validation runs without dropout, may cause loss inconsistency"
            );
        }
    }

    /// Get strategy description for logging
    pub fn get_strategy_description(&self) -> &str {
        match self.strategy {
            DropoutConsistencyStrategy::Standard => {
                "Training: dropout ON, Validation: dropout OFF (may cause loss jumping)"
            }
            DropoutConsistencyStrategy::Consistent => {
                "Training: dropout ON, Validation: dropout ON (consistent behavior)"
            }
            DropoutConsistencyStrategy::ValidationOnly => {
                "Training: dropout OFF, Validation: dropout ON (testing only)"
            }
            DropoutConsistencyStrategy::Disabled => {
                "Training: dropout OFF, Validation: dropout OFF (no regularization)"
            }
        }
    }

    /// Check if strategy may cause validation loss inconsistency
    pub fn may_cause_validation_inconsistency(&self) -> bool {
        matches!(
            self.strategy,
            DropoutConsistencyStrategy::Standard | DropoutConsistencyStrategy::ValidationOnly
        )
    }

    /// Get recommended strategy for stable validation
    pub fn get_recommended_strategy() -> Self {
        Self {
            strategy: DropoutConsistencyStrategy::Consistent,
            log_dropout_changes: true,
            warn_validation_inconsistency: false, // No warnings for recommended strategy
            attention_dropout_config: AttentionDropoutConfig::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dropout_consistency_strategies() {
        let config = DropoutConsistencyConfig::default();

        // Standard strategy
        assert!(config.should_apply_dropout(true, true)); // Training: ON
        assert!(!config.should_apply_dropout(false, true)); // Validation: OFF

        // Consistent strategy
        let consistent_config = DropoutConsistencyConfig {
            strategy: DropoutConsistencyStrategy::Consistent,
            ..Default::default()
        };
        assert!(consistent_config.should_apply_dropout(true, true)); // Training: ON
        assert!(consistent_config.should_apply_dropout(false, true)); // Validation: ON

        // Disabled strategy
        let disabled_config = DropoutConsistencyConfig {
            strategy: DropoutConsistencyStrategy::Disabled,
            ..Default::default()
        };
        assert!(!disabled_config.should_apply_dropout(true, true)); // Training: OFF
        assert!(!disabled_config.should_apply_dropout(false, true)); // Validation: OFF
    }

    #[test]
    fn test_validation_inconsistency_detection() {
        let standard_config = DropoutConsistencyConfig::default();
        assert!(standard_config.may_cause_validation_inconsistency());

        let consistent_config = DropoutConsistencyConfig {
            strategy: DropoutConsistencyStrategy::Consistent,
            ..Default::default()
        };
        assert!(!consistent_config.may_cause_validation_inconsistency());
    }

    #[test]
    fn test_recommended_strategy() {
        let recommended = DropoutConsistencyConfig::get_recommended_strategy();
        assert_eq!(recommended.strategy, DropoutConsistencyStrategy::Consistent);
        assert!(!recommended.may_cause_validation_inconsistency());
    }
}
