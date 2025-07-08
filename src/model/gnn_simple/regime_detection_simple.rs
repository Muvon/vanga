// Simplified Market Regime Detection - compilation ready
use crate::utils::error::Result;
use candle_core::Tensor;
use candle_nn::{linear, Linear, Module, VarBuilder};
use serde::{Deserialize, Serialize};

/// Market regime types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MarketRegime {
    /// Bull market - strong upward trend
    Bull,
    /// Bear market - strong downward trend
    Bear,
    /// Sideways/ranging market - no clear trend
    Sideways,
    /// High volatility regime - increased uncertainty
    HighVolatility,
    /// Low volatility regime - stable conditions
    LowVolatility,
    /// Crisis regime - extreme market stress
    Crisis,
    /// Recovery regime - post-crisis stabilization
    Recovery,
}

impl MarketRegime {
    /// Get regime as numeric encoding for model input
    pub fn to_numeric(&self) -> f32 {
        match self {
            MarketRegime::Bull => 1.0,
            MarketRegime::Bear => -1.0,
            MarketRegime::Sideways => 0.0,
            MarketRegime::HighVolatility => 2.0,
            MarketRegime::LowVolatility => -2.0,
            MarketRegime::Crisis => 3.0,
            MarketRegime::Recovery => -3.0,
        }
    }

    /// Get regime from numeric prediction
    pub fn from_numeric(value: f32) -> Self {
        if value > 2.5 {
            MarketRegime::Crisis
        } else if value > 1.5 {
            MarketRegime::HighVolatility
        } else if value > 0.5 {
            MarketRegime::Bull
        } else if value > -0.5 {
            MarketRegime::Sideways
        } else if value > -1.5 {
            MarketRegime::Bear
        } else if value > -2.5 {
            MarketRegime::LowVolatility
        } else {
            MarketRegime::Recovery
        }
    }
}

/// Regime detection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegimeConfig {
    /// Enable regime detection
    pub enabled: bool,
    /// Lookback window for regime analysis
    pub lookback_window: usize,
    /// Volatility threshold for high/low volatility regimes
    pub volatility_threshold: f64,
    /// Trend strength threshold for bull/bear detection
    pub trend_threshold: f64,
    /// Crisis detection sensitivity (0.0 to 1.0)
    pub crisis_sensitivity: f64,
}

impl Default for RegimeConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            lookback_window: 50,
            volatility_threshold: 0.02,
            trend_threshold: 0.01,
            crisis_sensitivity: 0.7,
        }
    }
}

impl RegimeConfig {
    /// Crypto-optimized configuration
    pub fn crypto_optimized() -> Self {
        Self {
            enabled: true,
            lookback_window: 30,
            volatility_threshold: 0.05,
            trend_threshold: 0.02,
            crisis_sensitivity: 0.8,
        }
    }

    /// Portfolio-optimized configuration
    pub fn portfolio_optimized() -> Self {
        Self {
            enabled: true,
            lookback_window: 100,
            volatility_threshold: 0.015,
            trend_threshold: 0.005,
            crisis_sensitivity: 0.9,
        }
    }
}

/// Simplified market regime detector
pub struct RegimeDetector {
    /// Regime classification network
    regime_classifier: Linear,
    /// Regime history for smoothing
    regime_history: Vec<MarketRegime>,
    /// Configuration
    config: RegimeConfig,
}

impl RegimeDetector {
    /// Create new regime detector
    pub fn new(config: RegimeConfig, vs: VarBuilder) -> Result<Self> {
        let regime_classifier = linear(256, 7, vs.pp("regime_classifier"))?; // 7 regime types

        log::info!(
            "Created simplified Regime Detector with lookback_window={}",
            config.lookback_window
        );

        Ok(Self {
            regime_classifier,
            regime_history: Vec::new(),
            config,
        })
    }

    /// Detect current market regime (simplified)
    pub fn detect_regime(&mut self, market_features: &Tensor) -> Result<Tensor> {
        // Simplified regime detection
        let regime_probabilities = self.regime_classifier.forward(market_features)?;

        // Update regime history (simplified)
        let predicted_regime = MarketRegime::Bull; // Placeholder
        self.regime_history.push(predicted_regime);

        // Keep history within reasonable bounds
        if self.regime_history.len() > self.config.lookback_window {
            self.regime_history.remove(0);
        }

        Ok(regime_probabilities)
    }

    /// Get current regime
    pub fn get_current_regime(&self) -> Option<&MarketRegime> {
        self.regime_history.last()
    }

    /// Get regime transition probability
    pub fn get_transition_probability(&self) -> f64 {
        if self.regime_history.len() < 2 {
            return 0.0;
        }

        let current = self.regime_history.last().unwrap();
        let previous = &self.regime_history[self.regime_history.len() - 2];

        if current != previous {
            1.0 // Transition occurred
        } else {
            0.0 // No transition
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_market_regime_numeric_conversion() {
        assert_eq!(MarketRegime::Bull.to_numeric(), 1.0);
        assert_eq!(MarketRegime::Bear.to_numeric(), -1.0);
        assert_eq!(MarketRegime::Crisis.to_numeric(), 3.0);

        assert_eq!(MarketRegime::from_numeric(1.5), MarketRegime::Bull);
        assert_eq!(MarketRegime::from_numeric(-1.5), MarketRegime::Bear);
        assert_eq!(MarketRegime::from_numeric(3.0), MarketRegime::Crisis);
    }

    #[test]
    fn test_regime_config_defaults() {
        let config = RegimeConfig::default();
        assert!(config.enabled);
        assert_eq!(config.lookback_window, 50);
        assert_eq!(config.volatility_threshold, 0.02);
    }

    #[test]
    fn test_regime_config_crypto_optimized() {
        let config = RegimeConfig::crypto_optimized();
        assert_eq!(config.lookback_window, 30);
        assert_eq!(config.volatility_threshold, 0.05);
        assert_eq!(config.crisis_sensitivity, 0.8);
    }
}
