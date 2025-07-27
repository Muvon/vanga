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
}

impl Default for DropoutConsistencyConfig {
    fn default() -> Self {
        Self {
            strategy: DropoutConsistencyStrategy::Standard,
            log_dropout_changes: true,
            warn_validation_inconsistency: true,
        }
    }
}

impl DropoutConsistencyConfig {
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
