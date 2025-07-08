// Market Regime Detection using Graph Neural Networks
use crate::utils::error::{Result, VangaError};
use candle_core::Tensor;
use candle_nn::{linear, Linear, Module, VarBuilder};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

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
    /// Enable multi-timeframe regime analysis
    pub multi_timeframe: bool,
    /// Timeframes for multi-timeframe analysis
    pub timeframes: Vec<String>,
    /// Regime smoothing factor (0.0 to 1.0)
    pub smoothing_factor: f64,
    /// Enable regime transition detection
    pub detect_transitions: bool,
}

impl Default for RegimeConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            lookback_window: 50,
            volatility_threshold: 0.02,
            trend_threshold: 0.01,
            crisis_sensitivity: 0.7,
            multi_timeframe: true,
            timeframes: vec!["1h".to_string(), "4h".to_string(), "1d".to_string()],
            smoothing_factor: 0.8,
            detect_transitions: true,
        }
    }
}

impl RegimeConfig {
    /// Crypto-optimized configuration
    pub fn crypto_optimized() -> Self {
        Self {
            enabled: true,
            lookback_window: 30,        // Shorter window for crypto volatility
            volatility_threshold: 0.05, // Higher threshold for crypto
            trend_threshold: 0.02,
            crisis_sensitivity: 0.8, // Higher sensitivity for crypto crashes
            multi_timeframe: true,
            timeframes: vec![
                "15m".to_string(),
                "1h".to_string(),
                "4h".to_string(),
                "1d".to_string(),
            ],
            smoothing_factor: 0.6, // Less smoothing for faster adaptation
            detect_transitions: true,
        }
    }

    /// Portfolio-optimized configuration
    pub fn portfolio_optimized() -> Self {
        Self {
            enabled: true,
            lookback_window: 100, // Longer window for portfolio stability
            volatility_threshold: 0.015,
            trend_threshold: 0.005,
            crisis_sensitivity: 0.9, // High sensitivity for portfolio protection
            multi_timeframe: true,
            timeframes: vec!["4h".to_string(), "1d".to_string(), "1w".to_string()],
            smoothing_factor: 0.9, // More smoothing for stability
            detect_transitions: true,
        }
    }
}

/// Market regime detector using GNN
pub struct RegimeDetector {
    /// Regime classification network
    regime_classifier: RegimeClassifier,
    /// Volatility analyzer
    volatility_analyzer: VolatilityAnalyzer,
    /// Trend analyzer
    trend_analyzer: TrendAnalyzer,
    /// Crisis detector
    crisis_detector: CrisisDetector,
    /// Regime history for smoothing
    regime_history: Vec<MarketRegime>,
    /// Configuration
    config: RegimeConfig,
}

/// Regime classification neural network
pub struct RegimeClassifier {
    /// Input feature processor
    feature_processor: Linear,
    /// Hidden layers
    hidden_layers: Vec<Linear>,
    /// Output layer (7 regimes)
    output_layer: Linear,
    /// Dropout rate
    dropout_rate: f64,
}

/// Volatility analysis component
pub struct VolatilityAnalyzer {
    /// Volatility feature extractor
    feature_extractor: Linear,
    /// Volatility predictor
    volatility_predictor: Linear,
    /// GARCH-like parameters
    garch_alpha: f64,
    garch_beta: f64,
}

/// Trend analysis component
pub struct TrendAnalyzer {
    /// Trend feature extractor
    feature_extractor: Linear,
    /// Trend strength predictor
    trend_predictor: Linear,
    /// Moving average periods
    ma_periods: Vec<usize>,
}

/// Crisis detection component
pub struct CrisisDetector {
    /// Crisis feature extractor
    feature_extractor: Linear,
    /// Crisis probability predictor
    crisis_predictor: Linear,
    /// Crisis indicators
    crisis_indicators: Vec<String>,
}

impl RegimeDetector {
    /// Create new regime detector
    pub fn new(config: RegimeConfig, vs: VarBuilder) -> Result<Self> {
        let regime_classifier = RegimeClassifier::new(256, vs.pp("regime_classifier"))?;
        let volatility_analyzer = VolatilityAnalyzer::new(vs.pp("volatility_analyzer"))?;
        let trend_analyzer = TrendAnalyzer::new(vs.pp("trend_analyzer"))?;
        let crisis_detector = CrisisDetector::new(vs.pp("crisis_detector"))?;

        log::info!(
            "Created Regime Detector with lookback_window={}, multi_timeframe={}",
            config.lookback_window,
            config.multi_timeframe
        );

        Ok(Self {
            regime_classifier,
            volatility_analyzer,
            trend_analyzer,
            crisis_detector,
            regime_history: Vec::new(),
            config,
        })
    }

    /// Detect current market regime
    pub fn detect_regime(&mut self, market_features: &Tensor) -> Result<Tensor> {
        // Extract regime features
        let regime_features = self.extract_regime_features(market_features)?;

        // Analyze volatility
        let volatility_features = self.volatility_analyzer.analyze(&regime_features)?;

        // Analyze trend
        let trend_features = self.trend_analyzer.analyze(&regime_features)?;

        // Detect crisis
        let crisis_features = self.crisis_detector.analyze(&regime_features)?;

        // Combine all features
        let combined_features = Tensor::cat(
            &[
                regime_features,
                volatility_features,
                trend_features,
                crisis_features,
            ],
            1,
        )?;

        // Classify regime
        let regime_probabilities = self.regime_classifier.forward(&combined_features)?;

        // Apply smoothing if enabled
        let smoothed_probabilities = if self.config.smoothing_factor > 0.0 {
            self.apply_regime_smoothing(&regime_probabilities)?
        } else {
            regime_probabilities
        };

        // Update regime history
        let predicted_regime = self.probabilities_to_regime(&smoothed_probabilities)?;
        self.regime_history.push(predicted_regime);

        // Keep history within reasonable bounds
        if self.regime_history.len() > self.config.lookback_window {
            self.regime_history.remove(0);
        }

        Ok(smoothed_probabilities)
    }

    /// Extract regime-specific features from market data
    fn extract_regime_features(&self, market_features: &Tensor) -> Result<Tensor> {
        let batch_size = market_features.dim(0)?;
        let seq_len = market_features.dim(1)?;
        let feature_dim = market_features.dim(2)?;

        // Calculate price-based features
        let prices = market_features.i((.., .., 0))?; // Assuming close price is first feature
        let returns = self.calculate_returns(&prices)?;
        let volatility = self.calculate_rolling_volatility(&returns, 20)?;

        // Calculate volume-based features
        let volumes = market_features.i((.., .., 1))?; // Assuming volume is second feature
        let volume_ma = self.calculate_moving_average(&volumes, 20)?;
        let volume_ratio = volumes.div(&volume_ma.add_scalar(1e-8)?)?;

        // Calculate momentum features
        let momentum_5 = self.calculate_momentum(&prices, 5)?;
        let momentum_20 = self.calculate_momentum(&prices, 20)?;

        // Calculate trend strength
        let trend_strength = self.calculate_trend_strength(&prices, 20)?;

        // Stack all regime features
        let regime_features = Tensor::stack(
            &[
                returns,
                volatility,
                volume_ratio,
                momentum_5,
                momentum_20,
                trend_strength,
            ],
            2,
        )?;

        Ok(regime_features)
    }

    /// Calculate price returns
    fn calculate_returns(&self, prices: &Tensor) -> Result<Tensor> {
        let shifted_prices = prices.narrow(1, 0, prices.dim(1)? - 1)?;
        let current_prices = prices.narrow(1, 1, prices.dim(1)? - 1)?;
        let returns = current_prices
            .div(&shifted_prices.add_scalar(1e-8)?)?
            .log()?;

        // Pad with zeros for first return
        let batch_size = prices.dim(0)?;
        let zero_returns = Tensor::zeros((batch_size, 1), prices.device())?;
        Tensor::cat(&[zero_returns, returns], 1)
    }

    /// Calculate rolling volatility
    fn calculate_rolling_volatility(&self, returns: &Tensor, window: usize) -> Result<Tensor> {
        let batch_size = returns.dim(0)?;
        let seq_len = returns.dim(1)?;
        let mut volatilities = Vec::new();

        for i in 0..seq_len {
            let start_idx = if i >= window { i - window } else { 0 };
            let window_returns = returns.narrow(1, start_idx, i - start_idx + 1)?;
            let mean_return = window_returns.mean(1)?;
            let centered_returns = window_returns.sub(&mean_return.unsqueeze(1)?)?;
            let variance = centered_returns.pow_tensor_scalar(2.0)?.mean(1)?;
            let volatility = variance.sqrt()?;
            volatilities.push(volatility);
        }

        Tensor::stack(&volatilities, 1)
    }

    /// Calculate moving average
    fn calculate_moving_average(&self, data: &Tensor, window: usize) -> Result<Tensor> {
        let seq_len = data.dim(1)?;
        let mut averages = Vec::new();

        for i in 0..seq_len {
            let start_idx = if i >= window { i - window } else { 0 };
            let window_data = data.narrow(1, start_idx, i - start_idx + 1)?;
            let average = window_data.mean(1)?;
            averages.push(average);
        }

        Tensor::stack(&averages, 1)
    }

    /// Calculate momentum
    fn calculate_momentum(&self, prices: &Tensor, period: usize) -> Result<Tensor> {
        let seq_len = prices.dim(1)?;
        let mut momentum = Vec::new();

        for i in 0..seq_len {
            if i >= period {
                let current_price = prices.i((.., i))?;
                let past_price = prices.i((.., i - period))?;
                let mom = current_price
                    .div(&past_price.add_scalar(1e-8)?)?
                    .sub_scalar(1.0)?;
                momentum.push(mom);
            } else {
                let zero_momentum = Tensor::zeros((prices.dim(0)?,), prices.device())?;
                momentum.push(zero_momentum);
            }
        }

        Tensor::stack(&momentum, 1)
    }

    /// Calculate trend strength
    fn calculate_trend_strength(&self, prices: &Tensor, window: usize) -> Result<Tensor> {
        let seq_len = prices.dim(1)?;
        let mut trend_strengths = Vec::new();

        for i in 0..seq_len {
            if i >= window {
                let window_prices = prices.narrow(1, i - window, window)?;

                // Calculate linear regression slope as trend strength
                let x_values = Tensor::arange(0f32, window as f32, prices.device())?;
                let y_values = window_prices.mean(0)?;

                // Simplified trend calculation (slope of linear fit)
                let x_mean = x_values.mean_all()?;
                let y_mean = y_values.mean_all()?;

                let numerator = x_values
                    .sub(&x_mean)?
                    .mul(&y_values.sub(&y_mean)?)?
                    .sum_all()?;
                let denominator = x_values.sub(&x_mean)?.pow_tensor_scalar(2.0)?.sum_all()?;

                let slope = numerator.div(&denominator.add_scalar(1e-8)?)?;
                let trend_strength = slope.abs()?;

                trend_strengths.push(trend_strength.unsqueeze(0)?.expand((prices.dim(0)?,))?);
            } else {
                let zero_trend = Tensor::zeros((prices.dim(0)?,), prices.device())?;
                trend_strengths.push(zero_trend);
            }
        }

        Tensor::stack(&trend_strengths, 1)
    }

    /// Apply regime smoothing based on history
    fn apply_regime_smoothing(&self, probabilities: &Tensor) -> Result<Tensor> {
        if self.regime_history.is_empty() {
            return Ok(probabilities.clone());
        }

        // Get last regime as one-hot encoding
        let last_regime = self.regime_history.last().unwrap();
        let last_regime_onehot = self.regime_to_onehot(last_regime, probabilities.device())?;

        // Apply exponential smoothing
        let alpha = 1.0 - self.config.smoothing_factor;
        let smoothed = probabilities
            .mul_scalar(alpha)?
            .add(&last_regime_onehot.mul_scalar(self.config.smoothing_factor)?)?;

        Ok(smoothed)
    }

    /// Convert regime to one-hot encoding
    fn regime_to_onehot(
        &self,
        regime: &MarketRegime,
        device: &candle_core::Device,
    ) -> Result<Tensor> {
        let mut onehot = vec![0.0f32; 7]; // 7 regime types
        let index = match regime {
            MarketRegime::Bull => 0,
            MarketRegime::Bear => 1,
            MarketRegime::Sideways => 2,
            MarketRegime::HighVolatility => 3,
            MarketRegime::LowVolatility => 4,
            MarketRegime::Crisis => 5,
            MarketRegime::Recovery => 6,
        };
        onehot[index] = 1.0;

        Tensor::from_slice(&onehot, (7,), device)
    }

    /// Convert probabilities to regime
    fn probabilities_to_regime(&self, probabilities: &Tensor) -> Result<MarketRegime> {
        let probs_vec = probabilities.to_vec1::<f32>()?;
        let max_index = probs_vec
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(index, _)| index)
            .unwrap_or(0);

        let regime = match max_index {
            0 => MarketRegime::Bull,
            1 => MarketRegime::Bear,
            2 => MarketRegime::Sideways,
            3 => MarketRegime::HighVolatility,
            4 => MarketRegime::LowVolatility,
            5 => MarketRegime::Crisis,
            6 => MarketRegime::Recovery,
            _ => MarketRegime::Sideways,
        };

        Ok(regime)
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

impl RegimeClassifier {
    fn new(input_dim: usize, vs: VarBuilder) -> Result<Self> {
        let feature_processor = linear(input_dim, 128, vs.pp("feature_processor"))?;

        let hidden_layers = vec![
            linear(128, 64, vs.pp("hidden_1"))?,
            linear(64, 32, vs.pp("hidden_2"))?,
        ];

        let output_layer = linear(32, 7, vs.pp("output"))?; // 7 regime types

        Ok(Self {
            feature_processor,
            hidden_layers,
            output_layer,
            dropout_rate: 0.1,
        })
    }

    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let mut x = self.feature_processor.forward(input)?;
        x = x.relu()?;

        for layer in &self.hidden_layers {
            x = layer.forward(&x)?;
            x = x.relu()?;
            // Note: Dropout would be applied here in training mode
        }

        let output = self.output_layer.forward(&x)?;
        output.softmax(1) // Convert to probabilities
    }
}

impl VolatilityAnalyzer {
    fn new(vs: VarBuilder) -> Result<Self> {
        let feature_extractor = linear(6, 32, vs.pp("feature_extractor"))?; // 6 regime features
        let volatility_predictor = linear(32, 16, vs.pp("volatility_predictor"))?;

        Ok(Self {
            feature_extractor,
            volatility_predictor,
            garch_alpha: 0.1,
            garch_beta: 0.85,
        })
    }

    fn analyze(&self, features: &Tensor) -> Result<Tensor> {
        let extracted = self.feature_extractor.forward(features)?;
        let volatility_features = self.volatility_predictor.forward(&extracted.relu()?)?;
        Ok(volatility_features)
    }
}

impl TrendAnalyzer {
    fn new(vs: VarBuilder) -> Result<Self> {
        let feature_extractor = linear(6, 32, vs.pp("feature_extractor"))?;
        let trend_predictor = linear(32, 16, vs.pp("trend_predictor"))?;

        Ok(Self {
            feature_extractor,
            trend_predictor,
            ma_periods: vec![5, 10, 20, 50],
        })
    }

    fn analyze(&self, features: &Tensor) -> Result<Tensor> {
        let extracted = self.feature_extractor.forward(features)?;
        let trend_features = self.trend_predictor.forward(&extracted.relu()?)?;
        Ok(trend_features)
    }
}

impl CrisisDetector {
    fn new(vs: VarBuilder) -> Result<Self> {
        let feature_extractor = linear(6, 32, vs.pp("feature_extractor"))?;
        let crisis_predictor = linear(32, 8, vs.pp("crisis_predictor"))?;

        Ok(Self {
            feature_extractor,
            crisis_predictor,
            crisis_indicators: vec![
                "extreme_volatility".to_string(),
                "volume_spike".to_string(),
                "correlation_breakdown".to_string(),
                "liquidity_crisis".to_string(),
            ],
        })
    }

    fn analyze(&self, features: &Tensor) -> Result<Tensor> {
        let extracted = self.feature_extractor.forward(features)?;
        let crisis_features = self.crisis_predictor.forward(&extracted.relu()?)?;
        Ok(crisis_features)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_nn::VarBuilder;

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
        assert_eq!(config.timeframes.len(), 4);
    }

    #[test]
    fn test_regime_config_portfolio_optimized() {
        let config = RegimeConfig::portfolio_optimized();
        assert_eq!(config.lookback_window, 100);
        assert_eq!(config.crisis_sensitivity, 0.9);
        assert_eq!(config.smoothing_factor, 0.9);
    }
}
