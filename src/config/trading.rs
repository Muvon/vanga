use serde::{Deserialize, Serialize};

/// Adaptive trading configuration - NO HARDCODED THRESHOLDS!
/// All trading decisions are based purely on model predictions and data analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveTradingConfig {
    /// Enable adaptive trading (default: true)
    pub enabled: bool,

    /// Minimum confidence threshold for any trading decision (default: 0.25 = 25%)
    /// This is the only "threshold" but it's based on model's own confidence assessment
    pub min_confidence_threshold: f64,

    /// Position sizing configuration
    pub position_sizing: AdaptivePositionSizing,

    /// Risk management configuration
    pub risk_management: AdaptiveRiskManagement,
}

impl Default for AdaptiveTradingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            min_confidence_threshold: 0.25, // Only trade when model has >25% confidence in any direction
            position_sizing: AdaptivePositionSizing::default(),
            risk_management: AdaptiveRiskManagement::default(),
        }
    }
}

/// Adaptive position sizing based on model confidence and volatility
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptivePositionSizing {
    /// Base position size multiplier (default: 1.0)
    pub base_multiplier: f64,

    /// Confidence boost factor (default: 0.5)
    /// Higher confidence = larger position (up to base_multiplier * (1 + confidence_boost))
    pub confidence_boost: f64,

    /// Volatility adjustment factor (default: 0.3)
    /// Higher volatility = smaller position to manage risk
    pub volatility_adjustment: f64,

    /// Maximum position size multiplier (default: 2.0)
    pub max_multiplier: f64,

    /// Minimum position size multiplier (default: 0.3)
    pub min_multiplier: f64,
}

impl Default for AdaptivePositionSizing {
    fn default() -> Self {
        Self {
            base_multiplier: 1.0,
            confidence_boost: 0.5,
            volatility_adjustment: 0.3,
            max_multiplier: 2.0,
            min_multiplier: 0.3,
        }
    }
}

/// Adaptive risk management based on model predictions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveRiskManagement {
    /// Use sequence bandwidth for stop loss calculation (default: true)
    pub use_sequence_bandwidth: bool,

    /// Bandwidth fraction for stop loss (default: 0.4 = 40% of sequence bandwidth)
    pub bandwidth_stop_fraction: f64,

    /// Use model's expected move for targets (default: true)
    pub use_expected_moves: bool,

    /// Target adjustment factor (default: 1.0 = use full expected move)
    pub target_adjustment: f64,

    /// Minimum risk/reward ratio (default: 0.5)
    /// Much more realistic for crypto markets
    pub min_risk_reward: f64,
}

impl Default for AdaptiveRiskManagement {
    fn default() -> Self {
        Self {
            use_sequence_bandwidth: true,
            bandwidth_stop_fraction: 0.4,
            use_expected_moves: true,
            target_adjustment: 1.0,
            min_risk_reward: 0.5,
        }
    }
}

impl AdaptiveTradingConfig {
    /// Validate configuration values
    pub fn validate(&self) -> crate::utils::error::Result<()> {
        if self.min_confidence_threshold <= 0.0 || self.min_confidence_threshold > 1.0 {
            return Err(crate::utils::error::VangaError::ConfigError(
                "min_confidence_threshold must be between 0.0 and 1.0".to_string(),
            ));
        }

        if self.position_sizing.base_multiplier <= 0.0 {
            return Err(crate::utils::error::VangaError::ConfigError(
                "base_multiplier must be positive".to_string(),
            ));
        }

        if self.risk_management.min_risk_reward <= 0.0 {
            return Err(crate::utils::error::VangaError::ConfigError(
                "min_risk_reward must be positive".to_string(),
            ));
        }

        Ok(())
    }

    /// Calculate adaptive position size based on model predictions
    pub fn calculate_position_size(
        &self,
        base_size: f64,
        confidence: f64,
        volatility_multiplier: f64,
    ) -> f64 {
        let confidence_boost = confidence * self.position_sizing.confidence_boost;
        let volatility_penalty = volatility_multiplier * self.position_sizing.volatility_adjustment;

        let adjusted_size = base_size
            * self.position_sizing.base_multiplier
            * (1.0 + confidence_boost - volatility_penalty);

        adjusted_size
            .max(base_size * self.position_sizing.min_multiplier)
            .min(base_size * self.position_sizing.max_multiplier)
    }

    /// Calculate adaptive stop loss based on sequence bandwidth
    pub fn calculate_stop_loss(
        &self,
        current_price: f64,
        sequence_bandwidth_percent: f64,
        recommended_stop_percent: f64,
        is_long: bool,
    ) -> f64 {
        let stop_percent = if self.risk_management.use_sequence_bandwidth {
            (sequence_bandwidth_percent / 100.0) * self.risk_management.bandwidth_stop_fraction
        } else {
            recommended_stop_percent / 100.0
        };

        if is_long {
            current_price * (1.0 - stop_percent)
        } else {
            current_price * (1.0 + stop_percent)
        }
    }

    /// Calculate adaptive target based on expected moves
    pub fn calculate_target(
        &self,
        current_price: f64,
        expected_move_percent: f64,
        is_long: bool,
    ) -> f64 {
        let target_percent = if self.risk_management.use_expected_moves {
            (expected_move_percent / 100.0) * self.risk_management.target_adjustment
        } else {
            0.01 // 1% fallback
        };

        if is_long {
            current_price * (1.0 + target_percent)
        } else {
            current_price * (1.0 - target_percent)
        }
    }
}
