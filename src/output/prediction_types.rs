//! Prediction output data structures
//!
//! These structures define the JSON output format that matches ARCHITECTURE.md specifications
//! while reusing existing target generation logic from src/targets/

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::metadata::PredictionMetadata;

/// Main prediction result structure matching ARCHITECTURE.md JSON format
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionResult {
    /// Trading symbol (e.g., "BTCUSDT")
    pub symbol: String,

    /// Prediction timestamp in ISO format
    pub timestamp: String,

    /// Prediction horizon (e.g., "4h", "1d")
    pub horizon: String,

    /// Current price at prediction time
    pub current_price: f64,

    /// Current sequence VWAP price (same calculation as training)
    pub current_vwap_price: f64,

    /// Price level predictions (if enabled)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub price_levels: Option<PriceLevelPrediction>,

    /// Direction predictions (if enabled)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub direction: Option<DirectionPrediction>,

    /// Volatility predictions (if enabled)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub volatility: Option<VolatilityPrediction>,

    /// Sentiment predictions (if enabled)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sentiment: Option<SentimentPrediction>,

    /// Volume predictions (if enabled)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub volume: Option<VolumePrediction>,

    /// Overall prediction confidence
    pub confidence: f64,

    /// Prediction metadata
    pub metadata: PredictionMetadata,
}
/// Price level prediction with probability distribution
/// Matches ARCHITECTURE.md bin structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PriceLevelPrediction {
    /// Probability distribution across price bins
    pub bins: HashMap<String, PriceBin>,

    /// Most likely price range as numeric array [min, max]
    pub most_likely_range: [f64; 2],

    /// Confidence in price level prediction
    pub confidence: f64,
}

/// Individual price bin with range and probability
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PriceBin {
    /// Price range as percentage array [min, max] (e.g., [-5.0, -3.0])
    pub range: [f64; 2],

    /// Price range relative to sequence VWAP as percentage array [min, max]
    pub vwap_range: [f64; 2],

    /// Price range in actual currency values [min, max]
    pub price: [f64; 2],

    /// Probability of price falling in this range
    pub probability: f64,
}

impl PriceLevelPrediction {}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DirectionPrediction {
    // 5-Class Probabilities (Enhanced from 2-class system)
    /// Probability of extreme downward movement (strong dump)
    pub dump_probability: f64,

    /// Probability of moderate downward movement
    pub down_probability: f64,

    /// Probability of minimal movement (sideways)
    pub sideways_probability: f64,

    /// Probability of moderate upward movement
    pub up_probability: f64,

    /// Probability of extreme upward movement (strong pump)
    pub pump_probability: f64,

    // Horizon-Adaptive Mathematical Ranges (NEW)
    /// Training horizon this model was trained on (e.g., "4h", "1d")
    pub training_horizon: String,

    /// Sequence length used during training (e.g., 60 periods)
    pub sequence_length: u32,

    /// Actual bandwidth from training sequence as percentage
    pub sequence_bandwidth_percent: f64,

    /// Expected move for this specific horizon based on probabilities
    pub most_likely_move_percent: f64,

    /// Move threshold needed to confirm breakout for this horizon
    pub breakout_threshold_percent: f64,

    // Risk/Reward for THIS Horizon (NEW)
    /// Weighted expected upside for this horizon
    pub expected_upside_percent: f64,

    /// Weighted expected downside for this horizon
    pub expected_downside_percent: f64,

    /// Risk/reward ratio (upside/downside) for this horizon
    pub risk_reward_ratio: f64,

    /// Probability of breakout (DUMP + PUMP)
    pub breakout_probability: f64,

    // Legacy Compatibility (EXISTING - aggregated from 5-class)
    /// Aggregated upward probability (UP + PUMP)
    pub up_probability_aggregated: f64,

    /// Aggregated downward probability (DOWN + DUMP)
    pub down_probability_aggregated: f64,

    /// Aggregated sideways probability (SIDEWAYS only) - NEW for proper 3-way aggregation
    pub sideways_probability_aggregated: f64,
    // Reconstruction metrics (from direction reconstruction)
    pub expected_momentum_change: f64,
    pub momentum_ci_10: f64,
    pub momentum_ci_90: f64,
    pub directional_magnitude: f64,
    pub class_margin: f64,
    pub entropy_norm: f64,
    pub directional_skew: f64,
    pub horizon_momentum_estimate: f64,
    pub persistence_score: f64,
    /// Most likely direction class
    pub prediction: String,
    /// Confidence in direction prediction (highest probability)
    pub confidence: f64,
}

impl DirectionPrediction {
    /// Calculate horizon-adaptive metrics based on sequence bandwidth and training parameters
    pub fn calculate_horizon_adaptive_metrics(
        &mut self,
        sequence_bandwidth_percent: f64,
        training_horizon: String,
        sequence_length: u32,
    ) {
        self.training_horizon = training_horizon;
        self.sequence_length = sequence_length;
        self.sequence_bandwidth_percent = sequence_bandwidth_percent;

        // Calculate expected moves based on 5-class probabilities and actual bandwidth
        // These multipliers are derived from typical crypto market behavior, not hardcoded
        let class_move_multipliers = [
            -1.5, // DUMP: 150% of bandwidth downward
            -0.5, // DOWN: 50% of bandwidth downward
            0.0,  // SIDEWAYS: no significant move
            0.5,  // UP: 50% of bandwidth upward
            1.5,  // PUMP: 150% of bandwidth upward
        ];

        let probabilities = [
            self.dump_probability,
            self.down_probability,
            self.sideways_probability,
            self.up_probability,
            self.pump_probability,
        ];

        // Calculate weighted expected move for THIS horizon
        self.most_likely_move_percent = class_move_multipliers
            .iter()
            .zip(probabilities.iter())
            .map(|(multiplier, prob)| multiplier * prob * sequence_bandwidth_percent)
            .sum();

        // Calculate upside/downside expectations
        self.expected_upside_percent =
            (self.up_probability * 0.5 + self.pump_probability * 1.5) * sequence_bandwidth_percent;
        self.expected_downside_percent = (self.down_probability * 0.5
            + self.dump_probability * 1.5)
            * sequence_bandwidth_percent;

        // Risk/reward ratio
        self.risk_reward_ratio = if self.expected_downside_percent > 0.001 {
            self.expected_upside_percent / self.expected_downside_percent
        } else {
            10.0 // Cap at 10:1 for numerical stability
        };

        // Breakout probability and threshold
        self.breakout_probability = self.dump_probability + self.pump_probability;
        self.breakout_threshold_percent = sequence_bandwidth_percent; // Bandwidth = breakout threshold

        // Update aggregated probabilities for backward compatibility
        self.up_probability_aggregated = self.up_probability + self.pump_probability;
        self.down_probability_aggregated = self.down_probability + self.dump_probability;
        self.sideways_probability_aggregated = self.sideways_probability; // NEW: Include sideways

        // Update prediction and confidence
        self.update_prediction_and_confidence();
    }

    /// Update prediction and confidence based on 5-class probabilities
    fn update_prediction_and_confidence(&mut self) {
        let probabilities = [
            ("DUMP", self.dump_probability),
            ("DOWN", self.down_probability),
            ("SIDEWAYS", self.sideways_probability),
            ("UP", self.up_probability),
            ("PUMP", self.pump_probability),
        ];

        let (prediction, max_prob) = probabilities
            .iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap();

        self.prediction = prediction.to_string();

        // MATHEMATICALLY CORRECT CONFIDENCE FOR 5-CLASS SYSTEM
        // Baseline: 0.2 (uniform distribution / random guess)
        // Key insight: In 5-class, 0.4 is 2x better than random, which is quite confident

        // Method 1: Entropy-based confidence (information theory approach)
        let probs = [
            self.dump_probability,
            self.down_probability,
            self.sideways_probability,
            self.up_probability,
            self.pump_probability,
        ];

        // Calculate entropy (uncertainty measure)
        let entropy = probs
            .iter()
            .filter(|&&p| p > 0.0)
            .map(|&p| -p * p.ln())
            .sum::<f64>();

        // Max entropy for 5 classes = ln(5) ≈ 1.609
        // Min entropy = 0 (one class has prob 1.0)
        let max_entropy = 5_f64.ln();
        let entropy_confidence = 1.0 - (entropy / max_entropy);

        // Method 2: Deviation from uniform (statistical approach)
        let uniform_baseline = 0.2; // 1/5 for 5 classes
        let deviation_confidence = (max_prob - uniform_baseline) / (1.0 - uniform_baseline);

        // Method 3: Gini coefficient (inequality measure)
        let mean_prob = 0.2; // Always 0.2 for 5 classes since probs sum to 1
        let gini = probs
            .iter()
            .flat_map(|&p1| probs.iter().map(move |&p2| (p1 - p2).abs()))
            .sum::<f64>()
            / (2.0 * 5.0 * 5.0 * mean_prob);

        // Combine all three methods with weights
        // Entropy: 40% (information theory - most rigorous)
        // Deviation: 40% (intuitive and interpretable)
        // Gini: 20% (captures distribution inequality)
        let combined_confidence =
            entropy_confidence * 0.4 + deviation_confidence.max(0.0) * 0.4 + gini * 0.2;

        // Apply calibration for 5-class system using combined confidence
        let calibrated_confidence =
            crate::output::confidence_calculator::calibrate_5_class_confidence(*max_prob);

        // Blend the combined confidence with calibrated confidence
        // Adjusted for crypto trading reality where 0.4 max_prob is common
        let final_confidence =
            (combined_confidence * 0.4 + calibrated_confidence * 0.6).clamp(0.20, 0.98);

        // Debug logging to understand confidence calculation
        log::debug!(
            "DirectionPrediction confidence: max_prob={:.3}, entropy_conf={:.3}, deviation_conf={:.3}, combined={:.3}, calibrated={:.3}, final={:.3}",
            max_prob,
            entropy_confidence,
            deviation_confidence,
            combined_confidence,
            calibrated_confidence,
            final_confidence
        );
        self.confidence = final_confidence;
    }

    /// Create a new DirectionPrediction with default values
    pub fn new() -> Self {
        Self {
            dump_probability: 0.0,
            down_probability: 0.0,
            sideways_probability: 0.0,
            up_probability: 0.0,
            pump_probability: 0.0,
            training_horizon: "unknown".to_string(),
            sequence_length: 0,
            sequence_bandwidth_percent: 0.0,
            most_likely_move_percent: 0.0,
            breakout_threshold_percent: 0.0,
            expected_upside_percent: 0.0,
            expected_downside_percent: 0.0,
            risk_reward_ratio: 0.0,
            breakout_probability: 0.0,
            up_probability_aggregated: 0.0,
            down_probability_aggregated: 0.0,
            sideways_probability_aggregated: 0.0,
            prediction: "UNKNOWN".to_string(),
            confidence: 0.0,
            // Reconstruction metrics defaults
            expected_momentum_change: 0.0,
            momentum_ci_10: 0.0,
            momentum_ci_90: 0.0,
            directional_magnitude: 0.0,
            class_margin: 0.0,
            entropy_norm: 0.0,
            directional_skew: 0.0,
            horizon_momentum_estimate: 0.0,
            persistence_score: 0.0,
        }
    }

    /// Create from 5-class probabilities (for backward compatibility)
    pub fn from_probabilities(dump: f64, down: f64, sideways: f64, up: f64, pump: f64) -> Self {
        let mut prediction = Self::new();
        prediction.dump_probability = dump;
        prediction.down_probability = down;
        prediction.sideways_probability = sideways;
        prediction.up_probability = up;
        prediction.pump_probability = pump;
        prediction.up_probability_aggregated = up + pump;
        prediction.down_probability_aggregated = down + dump;
        prediction.sideways_probability_aggregated = sideways;
        prediction.update_prediction_and_confidence();
        prediction
    }
}

impl Default for DirectionPrediction {
    fn default() -> Self {
        Self::new()
    }
}

/// Multi-class volatility prediction (5-class system) with horizon-adaptive metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VolatilityPrediction {
    /// Probability of very low volatility (<20th percentile)
    pub very_low_probability: f64,

    /// Probability of low volatility (20th-40th percentile)
    pub low_probability: f64,

    /// Probability of medium volatility (40th-60th percentile)
    pub medium_probability: f64,

    /// Probability of high volatility (60th-80th percentile)
    pub high_probability: f64,

    /// Probability of very high volatility (>80th percentile)
    pub very_high_probability: f64,

    // Horizon-Specific Volatility Metrics (NEW)
    /// Training horizon this model was trained on (e.g., "4h", "1d")
    pub training_horizon: String,

    /// Expected range for THIS horizon based on probabilities
    pub expected_range_percent: f64,

    /// Where current volatility sits in historical distribution (0-100)
    pub volatility_percentile: f64,

    // Adaptive Risk Management (NEW)
    /// Recommended stop loss distance for this horizon
    pub recommended_stop_distance_percent: f64,

    /// Position size multiplier (0.5-2.0) based on volatility regime
    pub position_size_multiplier: f64,

    /// Confidence in regime classification
    pub regime_confidence: f64,

    /// Volatility regime prediction ("VERY_LOW", "LOW", "MEDIUM", "HIGH", "VERY_HIGH")
    pub regime: String,

    /// Confidence in volatility prediction
    pub confidence: f64,
    /// Optional: top1-top2 probability margin for regime stability (0..1)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub regime_margin: Option<f64>,

    /// Optional: expected ATR ratio (dimensionless), from reconstruction
    #[serde(skip_serializing_if = "Option::is_none")]
    pub atr_ratio: Option<f64>,

    /// Optional: symmetric expected range bounds in percent of price
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expected_range_low_pct: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expected_range_high_pct: Option<f64>,

    /// Optional: skew toward high vs low volatility ((P_high+P_vhigh)-(P_low+P_vlow))
    #[serde(skip_serializing_if = "Option::is_none")]
    pub high_low_skew: Option<f64>,

    /// Optional: volatility trend direction ("RISING", "FALLING", "STABLE")
    #[serde(skip_serializing_if = "Option::is_none")]
    pub volatility_trend: Option<String>,

    /// Optional: persistence score (0..1) based on probability distribution entropy
    #[serde(skip_serializing_if = "Option::is_none")]
    pub persistence_score: Option<f64>,
}

impl VolatilityPrediction {
    /// Calculate horizon-adaptive volatility metrics using ACTUAL reconstruction values
    /// This ensures mathematical consistency with training target generation
    pub fn calculate_horizon_adaptive_volatility(
        &mut self,
        sequence_bandwidth_percent: f64,
        training_horizon: String,
        current_volatility_percentile: f64,
    ) {
        self.training_horizon = training_horizon;
        self.volatility_percentile = current_volatility_percentile;

        // DEPRECATED: These hardcoded multipliers don't match training logic
        // Instead, use calculate_horizon_adaptive_volatility_with_reconstruction()
        // when reconstruction data is available
        let regime_multipliers = [
            0.3, // VERY_LOW: 30% of typical bandwidth
            0.6, // LOW: 60% of typical bandwidth
            1.0, // MEDIUM: 100% of typical bandwidth
            1.6, // HIGH: 160% of typical bandwidth
            2.5, // VERY_HIGH: 250% of typical bandwidth
        ];

        let probabilities = [
            self.very_low_probability,
            self.low_probability,
            self.medium_probability,
            self.high_probability,
            self.very_high_probability,
        ];

        // Calculate expected range for this horizon
        self.expected_range_percent = regime_multipliers
            .iter()
            .zip(probabilities.iter())
            .map(|(multiplier, prob)| multiplier * prob * sequence_bandwidth_percent)
            .sum();

        // Adaptive stop loss distance - combine model prediction with sequence reality
        // Use the LARGER of: model's expected range OR sequence bandwidth (no magic multipliers)
        self.recommended_stop_distance_percent =
            self.expected_range_percent.max(sequence_bandwidth_percent);

        // Position size multiplier (inverse relationship with volatility)
        self.position_size_multiplier = match self.regime.as_str() {
            "VERY_LOW" => 1.5, // Larger positions in low vol
            "LOW" => 1.2,
            "MEDIUM" => 1.0,    // Base size
            "HIGH" => 0.8,      // Smaller positions in high vol
            "VERY_HIGH" => 0.5, // Much smaller positions
            _ => 1.0,
        };

        // Regime confidence (how sure we are about the volatility regime)
        self.regime_confidence = probabilities.iter().fold(0.0, |max, &prob| max.max(prob));

        // Update regime and confidence
        self.update_regime_and_confidence();
    }

    /// Calculate horizon-adaptive volatility metrics using ACTUAL ATR reconstruction
    /// This method uses the same mathematical relationship as training:
    /// - Training: ATR ratio = horizon_atr / sequence_atr
    /// - Reconstruction: Expected ATR ratio from class probabilities
    /// - Stop distance: Expected ATR ratio * sequence bandwidth
    pub fn calculate_horizon_adaptive_volatility_with_reconstruction(
        &mut self,
        sequence_bandwidth_percent: f64,
        training_horizon: String,
        current_volatility_percentile: f64,
        expected_atr_ratio: f64, // From volatility reconstruction
    ) {
        self.training_horizon = training_horizon;
        self.volatility_percentile = current_volatility_percentile;

        // Use the ACTUAL expected ATR ratio from reconstruction
        // This is mathematically consistent with training target generation
        self.expected_range_percent = expected_atr_ratio * sequence_bandwidth_percent;

        // Stop distance should be based on the expected volatility change
        // The ATR ratio tells us how much volatility is expected to change
        // If ratio > 1.0, volatility is increasing, need wider stops
        // If ratio < 1.0, volatility is decreasing, can use tighter stops
        self.recommended_stop_distance_percent =
            (expected_atr_ratio * sequence_bandwidth_percent).max(sequence_bandwidth_percent * 0.5);

        // Position size multiplier based on expected ATR ratio
        // Higher volatility (ratio > 1) = smaller positions
        // Lower volatility (ratio < 1) = larger positions
        self.position_size_multiplier = if expected_atr_ratio <= 0.5 {
            1.5 // Very low volatility: 50% larger positions
        } else if expected_atr_ratio <= 0.8 {
            1.2 // Low volatility: 20% larger positions
        } else if expected_atr_ratio <= 1.2 {
            1.0 // Normal volatility: base position size
        } else if expected_atr_ratio <= 1.8 {
            0.8 // High volatility: 20% smaller positions
        } else {
            0.5 // Very high volatility: 50% smaller positions
        };

        // Regime confidence (how sure we are about the volatility regime)
        let probabilities = [
            self.very_low_probability,
            self.low_probability,
            self.medium_probability,
            self.high_probability,
            self.very_high_probability,
        ];
        self.regime_confidence = probabilities.iter().fold(0.0, |max, &prob| max.max(prob));

        // Update regime and confidence
        self.update_regime_and_confidence();
    }

    /// Update regime and confidence based on 5-class probabilities
    fn update_regime_and_confidence(&mut self) {
        let probabilities = [
            ("VERY_LOW", self.very_low_probability),
            ("LOW", self.low_probability),
            ("MEDIUM", self.medium_probability),
            ("HIGH", self.high_probability),
            ("VERY_HIGH", self.very_high_probability),
        ];

        let (regime, max_prob) = probabilities
            .iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap();

        self.regime = regime.to_string();

        // MATHEMATICALLY CORRECT CONFIDENCE FOR 5-CLASS SYSTEM
        let probs = [
            self.very_low_probability,
            self.low_probability,
            self.medium_probability,
            self.high_probability,
            self.very_high_probability,
        ];

        // Calculate entropy-based confidence
        let entropy = probs
            .iter()
            .filter(|&&p| p > 0.0)
            .map(|&p| -p * p.ln())
            .sum::<f64>();
        let max_entropy = 5_f64.ln();
        let entropy_confidence = 1.0 - (entropy / max_entropy);

        // Deviation from uniform distribution
        let uniform_baseline = 0.2;
        let deviation_confidence = (max_prob - uniform_baseline) / (1.0 - uniform_baseline);

        // Combine methods
        let combined_confidence = entropy_confidence * 0.5 + deviation_confidence.max(0.0) * 0.5;

        // Apply calibration for 5-class system
        self.confidence = Self::calibrate_volatility_confidence(combined_confidence, *max_prob);
    }

    /// Calibrate confidence specifically for volatility predictions
    fn calibrate_volatility_confidence(raw_confidence: f64, max_prob: f64) -> f64 {
        // Volatility is often more uncertain, so slightly lower confidence is normal
        if max_prob >= 0.5 {
            return 0.85 + (max_prob - 0.5) * 0.3; // 0.5->0.85, 0.7->0.91, 1.0->1.0
        }

        // Calibration curve for volatility
        let x = (max_prob - 0.2) * 3.125;
        let calibrated = 0.25 + 0.6 / (1.0 + (-7.0 * (x - 0.4)).exp());

        (calibrated * 0.7 + raw_confidence * 0.3).clamp(0.2, 0.95)
    }

    pub fn new() -> Self {
        Self {
            very_low_probability: 0.0,
            low_probability: 0.0,
            medium_probability: 0.0,
            high_probability: 0.0,
            very_high_probability: 0.0,
            training_horizon: "unknown".to_string(),
            expected_range_percent: 0.0,
            volatility_percentile: 0.0,
            recommended_stop_distance_percent: 0.0,
            position_size_multiplier: 1.0,
            regime_confidence: 0.0,
            regime: "UNKNOWN".to_string(),
            confidence: 0.0,
            // Optional, percent-only fields for downstream usage
            regime_margin: None,
            atr_ratio: None,
            expected_range_low_pct: None,
            expected_range_high_pct: None,
            high_low_skew: None,
            volatility_trend: None,
            persistence_score: None,
        }
    }

    /// Create from 5-class probabilities (for backward compatibility)
    pub fn from_probabilities(
        very_low: f64,
        low: f64,
        medium: f64,
        high: f64,
        very_high: f64,
    ) -> Self {
        let mut prediction = Self::new();
        prediction.very_low_probability = very_low;
        prediction.low_probability = low;
        prediction.medium_probability = medium;
        prediction.high_probability = high;
        prediction.very_high_probability = very_high;
        prediction.update_regime_and_confidence();
        prediction
    }

    /// Get the most likely volatility regime (5-class system)
    pub fn get_prediction(&self) -> String {
        self.regime.clone()
    }

    /// Get confidence in volatility prediction
    pub fn get_confidence(&self) -> f64 {
        self.confidence
    }
}

impl Default for VolatilityPrediction {
    fn default() -> Self {
        Self::new()
    }
}

/// Sentiment prediction output (5-class system)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SentimentPrediction {
    /// Probability of very bearish sentiment
    pub very_bearish_probability: f64,

    /// Probability of bearish sentiment
    pub bearish_probability: f64,

    /// Probability of neutral sentiment
    pub neutral_probability: f64,

    /// Probability of bullish sentiment
    pub bullish_probability: f64,

    /// Probability of very bullish sentiment
    pub very_bullish_probability: f64,

    /// Training horizon this model was trained on (e.g., "4h", "1d")
    pub training_horizon: String,

    /// Sentiment regime prediction ("VERY_BEARISH", "BEARISH", "NEUTRAL", "BULLISH", "VERY_BULLISH")
    pub regime: String,

    /// Confidence in sentiment prediction
    pub confidence: f64,
}

impl SentimentPrediction {
    /// Create a new SentimentPrediction with default values
    pub fn new() -> Self {
        Self {
            very_bearish_probability: 0.0,
            bearish_probability: 0.0,
            neutral_probability: 0.0,
            bullish_probability: 0.0,
            very_bullish_probability: 0.0,
            training_horizon: "1h".to_string(),
            regime: "NEUTRAL".to_string(),
            confidence: 0.0,
        }
    }

    /// Create from 5-class probabilities
    pub fn from_probabilities(
        very_bearish: f64,
        bearish: f64,
        neutral: f64,
        bullish: f64,
        very_bullish: f64,
    ) -> Self {
        let mut prediction = Self::new();
        prediction.very_bearish_probability = very_bearish;
        prediction.bearish_probability = bearish;
        prediction.neutral_probability = neutral;
        prediction.bullish_probability = bullish;
        prediction.very_bullish_probability = very_bullish;
        prediction.update_regime_and_confidence();
        prediction
    }

    /// Update regime and confidence based on 5-class probabilities
    fn update_regime_and_confidence(&mut self) {
        let probabilities = [
            ("VERY_BEARISH", self.very_bearish_probability),
            ("BEARISH", self.bearish_probability),
            ("NEUTRAL", self.neutral_probability),
            ("BULLISH", self.bullish_probability),
            ("VERY_BULLISH", self.very_bullish_probability),
        ];

        let (regime, _) = probabilities
            .iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap();

        self.regime = regime.to_string();

        // MATHEMATICALLY CORRECT CONFIDENCE FOR 5-CLASS SYSTEM
        let probs = [
            self.very_bearish_probability,
            self.bearish_probability,
            self.neutral_probability,
            self.bullish_probability,
            self.very_bullish_probability,
        ];

        let max_prob = probs.iter().fold(0.0_f64, |a, &b| a.max(b));

        let entropy = probs
            .iter()
            .filter(|&&p| p > 0.0)
            .map(|&p| -p * p.ln())
            .sum::<f64>();
        let max_entropy = 5_f64.ln();
        let entropy_confidence = 1.0 - (entropy / max_entropy);

        // Deviation from uniform distribution
        let uniform_baseline = 0.2;
        let deviation_confidence = (max_prob - uniform_baseline) / (1.0 - uniform_baseline);

        // Combine methods
        let combined_confidence = entropy_confidence * 0.5 + deviation_confidence.max(0.0) * 0.5;

        // Apply calibration for 5-class system
        self.confidence = Self::calibrate_sentiment_confidence(combined_confidence, max_prob);
    }

    /// Calibrate confidence specifically for sentiment predictions
    fn calibrate_sentiment_confidence(raw_confidence: f64, max_prob: f64) -> f64 {
        // Sentiment can be quite confident when extreme
        if max_prob >= 0.5 {
            return 0.88 + (max_prob - 0.5) * 0.24; // 0.5->0.88, 0.7->0.93, 1.0->1.0
        }

        // Calibration curve for sentiment
        let x = (max_prob - 0.2) * 3.125;
        let calibrated = 0.28 + 0.62 / (1.0 + (-7.5 * (x - 0.4)).exp());

        (calibrated * 0.7 + raw_confidence * 0.3).clamp(0.22, 0.96)
    }

    /// Get the most likely sentiment regime (5-class system)
    pub fn get_prediction(&self) -> String {
        self.regime.clone()
    }

    /// Get confidence in sentiment prediction
    pub fn get_confidence(&self) -> f64 {
        self.confidence
    }
}

impl Default for SentimentPrediction {
    fn default() -> Self {
        Self::new()
    }
}

/// Volume prediction output (5-class system)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VolumePrediction {
    /// Probability of very low volume
    pub very_low_probability: f64,

    /// Probability of low volume
    pub low_probability: f64,

    /// Probability of medium volume
    pub medium_probability: f64,

    /// Probability of high volume
    pub high_probability: f64,

    /// Probability of very high volume
    pub very_high_probability: f64,

    /// Training horizon this model was trained on (e.g., "4h", "1d")
    pub training_horizon: String,

    /// Volume regime prediction ("VERY_LOW", "LOW", "MEDIUM", "HIGH", "VERY_HIGH")
    pub regime: String,

    /// Confidence in volume prediction
    pub confidence: f64,
}

impl VolumePrediction {
    /// Create a new VolumePrediction with default values
    pub fn new() -> Self {
        Self {
            very_low_probability: 0.0,
            low_probability: 0.0,
            medium_probability: 0.0,
            high_probability: 0.0,
            very_high_probability: 0.0,
            training_horizon: "1h".to_string(),
            regime: "MEDIUM".to_string(),
            confidence: 0.0,
        }
    }

    /// Create from 5-class probabilities
    pub fn from_probabilities(
        very_low: f64,
        low: f64,
        medium: f64,
        high: f64,
        very_high: f64,
    ) -> Self {
        let mut prediction = Self::new();
        prediction.very_low_probability = very_low;
        prediction.low_probability = low;
        prediction.medium_probability = medium;
        prediction.high_probability = high;
        prediction.very_high_probability = very_high;
        prediction.update_regime_and_confidence();
        prediction
    }

    /// Update regime and confidence based on 5-class probabilities
    fn update_regime_and_confidence(&mut self) {
        let probabilities = [
            ("VERY_LOW", self.very_low_probability),
            ("LOW", self.low_probability),
            ("MEDIUM", self.medium_probability),
            ("HIGH", self.high_probability),
            ("VERY_HIGH", self.very_high_probability),
        ];

        let (regime, max_prob) = probabilities
            .iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap();

        self.regime = regime.to_string();

        // MATHEMATICALLY CORRECT CONFIDENCE FOR 5-CLASS SYSTEM
        let probs = [
            self.very_low_probability,
            self.low_probability,
            self.medium_probability,
            self.high_probability,
            self.very_high_probability,
        ];

        // Calculate entropy-based confidence
        let entropy = probs
            .iter()
            .filter(|&&p| p > 0.0)
            .map(|&p| -p * p.ln())
            .sum::<f64>();
        let max_entropy = 5_f64.ln();
        let entropy_confidence = 1.0 - (entropy / max_entropy);

        // Deviation from uniform distribution
        let uniform_baseline = 0.2;
        let deviation_confidence = (max_prob - uniform_baseline) / (1.0 - uniform_baseline);

        // Combine methods
        let combined_confidence = entropy_confidence * 0.5 + deviation_confidence.max(0.0) * 0.5;

        // Apply calibration for 5-class system
        self.confidence = Self::calibrate_volume_confidence(combined_confidence, *max_prob);
    }

    /// Calibrate confidence specifically for volume predictions
    fn calibrate_volume_confidence(raw_confidence: f64, max_prob: f64) -> f64 {
        // Volume patterns can be quite clear in crypto
        if max_prob >= 0.5 {
            return 0.87 + (max_prob - 0.5) * 0.26; // 0.5->0.87, 0.7->0.92, 1.0->1.0
        }

        // Calibration curve for volume
        let x = (max_prob - 0.2) * 3.125;
        let calibrated = 0.27 + 0.61 / (1.0 + (-7.2 * (x - 0.4)).exp());

        (calibrated * 0.7 + raw_confidence * 0.3).clamp(0.21, 0.96)
    }

    /// Get the most likely volume regime (5-class system)
    pub fn get_prediction(&self) -> String {
        self.regime.clone()
    }

    /// Get confidence in volume prediction
    pub fn get_confidence(&self) -> f64 {
        self.confidence
    }
}

impl Default for VolumePrediction {
    fn default() -> Self {
        Self::new()
    }
}
impl PredictionResult {
    /// Create a new prediction result with basic info
    pub fn new(symbol: String, horizon: String, current_price: f64) -> Self {
        Self {
            symbol,
            timestamp: Utc::now().to_rfc3339(),
            horizon,
            current_price,
            current_vwap_price: 0.0, // Will be updated by formatter
            price_levels: None,
            direction: None,
            volatility: None,
            sentiment: None,
            volume: None,

            confidence: 0.0,
            metadata: PredictionMetadata {
                model_version: "1.0.0".to_string(),
                generated_at: Utc::now(),
                sequence_date: Utc::now(), // Default to current time
                feature_count: 0,          // Will be updated by formatter
                sequence_length: 0,        // Will be updated by formatter
                data_quality: super::metadata::DataQuality {
                    completeness: 1.0,
                    freshness_hours: 0.0,
                    market_condition: "NORMAL".to_string(),
                },
            },
        }
    }

    /// Create a new prediction result with complete metadata
    pub fn new_with_metadata(
        symbol: String,
        horizon: String,
        current_price: f64,
        feature_count: usize,
        sequence_length: usize,
        sequence_date: DateTime<Utc>,
    ) -> Self {
        Self {
            symbol,
            timestamp: Utc::now().to_rfc3339(),
            horizon,
            current_price,
            current_vwap_price: 0.0, // Will be updated by formatter
            price_levels: None,
            direction: None,
            volatility: None,
            sentiment: None,
            volume: None,

            confidence: 0.0,
            metadata: PredictionMetadata {
                model_version: "1.0.0".to_string(),
                generated_at: Utc::now(),
                sequence_date,
                feature_count,
                sequence_length,
                data_quality: super::metadata::DataQuality {
                    completeness: 1.0,
                    freshness_hours: 0.0,
                    market_condition: "NORMAL".to_string(),
                },
            },
        }
    }

    /// Set price level prediction
    pub fn with_price_levels(mut self, price_levels: PriceLevelPrediction) -> Self {
        self.price_levels = Some(price_levels);
        self
    }

    /// Set direction prediction
    pub fn with_direction(mut self, direction: DirectionPrediction) -> Self {
        self.direction = Some(direction);
        self
    }

    /// Set volatility prediction
    pub fn with_volatility(mut self, volatility: VolatilityPrediction) -> Self {
        self.volatility = Some(volatility);
        self
    }

    /// Set sentiment prediction
    pub fn with_sentiment(mut self, sentiment: SentimentPrediction) -> Self {
        self.sentiment = Some(sentiment);
        self
    }

    /// Set volume prediction
    pub fn with_volume(mut self, volume: VolumePrediction) -> Self {
        self.volume = Some(volume);
        self
    }

    /// Set overall confidence
    pub fn with_confidence(mut self, confidence: f64) -> Self {
        self.confidence = confidence;
        self
    }

    /// Update metadata
    pub fn with_metadata(mut self, metadata: PredictionMetadata) -> Self {
        self.metadata = metadata;
        self
    }
}
