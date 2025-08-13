//! Prediction output data structures
//!
//! These structures define the JSON output format that matches ARCHITECTURE.md specifications
//! while reusing existing target generation logic from src/targets/

use chrono::Utc;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// Import types that will be defined in other modules
use super::metadata::PredictionMetadata;
use super::trading_orders::{OrderLevel, TradingOrders};

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

    /// Trading orders with dynamic position sizing (always included)
    pub orders: TradingOrders,

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

impl PriceLevelPrediction {
    /// Extract price level ranges as percentage arrays for order generation
    /// Returns ranges in the order: [strong_down, moderate_down, neutral, moderate_up, strong_up]
    pub fn extract_ranges_for_orders(&self) -> [[f64; 2]; 5] {
        let mut ranges = [[0.0, 0.0]; 5];

        // Map bin names to array indices
        let bin_order = [
            "strong_down",   // 0
            "moderate_down", // 1
            "neutral",       // 2
            "moderate_up",   // 3
            "strong_up",     // 4
        ];

        for (i, bin_name) in bin_order.iter().enumerate() {
            if let Some(bin) = self.bins.get(*bin_name) {
                ranges[i] = bin.range;
            }
        }

        ranges
    }

    /// Extract probability-based order portions for adaptive allocation
    /// Returns (up_portions, down_portions) for SHORT position logic
    pub fn extract_probability_portions(&self) -> ([f64; 3], [f64; 3]) {
        // For SHORT: entries use UP probabilities, exits use DOWN probabilities
        let moderate_up_prob = self
            .bins
            .get("moderate_up")
            .map(|b| b.probability)
            .unwrap_or(0.0);
        let strong_up_prob = self
            .bins
            .get("strong_up")
            .map(|b| b.probability)
            .unwrap_or(0.0);
        let moderate_down_prob = self
            .bins
            .get("moderate_down")
            .map(|b| b.probability)
            .unwrap_or(0.0);
        let strong_down_prob = self
            .bins
            .get("strong_down")
            .map(|b| b.probability)
            .unwrap_or(0.0);

        let up_total = moderate_up_prob + strong_up_prob;
        let down_total = moderate_down_prob + strong_down_prob;

        // Entry portions (based on UP probabilities - we enter when price moves against us)
        let entry_portions = if up_total > 0.0 {
            let moderate_portion = moderate_up_prob / up_total;
            let strong_portion = strong_up_prob / up_total;
            // Split into 3 orders: moderate_lower, moderate_upper, strong_lower
            [
                moderate_portion * 0.6, // 60% of moderate_up probability
                moderate_portion * 0.4, // 40% of moderate_up probability
                strong_portion,         // 100% of strong_up probability
            ]
        } else {
            [0.33, 0.33, 0.34] // Fallback to equal distribution
        };

        // Exit portions (based on DOWN probabilities - we exit when price moves with us)
        let exit_portions = if down_total > 0.0 {
            let moderate_portion = moderate_down_prob / down_total;
            let strong_portion = strong_down_prob / down_total;
            // Split into 3 orders: moderate_center, strong_upper, strong_lower
            [
                moderate_portion,     // 100% of moderate_down probability
                strong_portion * 0.5, // 50% of strong_down probability
                strong_portion * 0.5, // 50% of strong_down probability
            ]
        } else {
            [0.33, 0.33, 0.34] // Fallback to equal distribution
        };

        (entry_portions, exit_portions)
    }

    /// Get natural price positioning from range centers and boundaries
    pub fn get_natural_price_positions(&self) -> (Vec<f64>, Vec<f64>) {
        let moderate_up = self.bins.get("moderate_up");
        let strong_up = self.bins.get("strong_up");
        let moderate_down = self.bins.get("moderate_down");
        let strong_down = self.bins.get("strong_down");

        // Entry positions (UP ranges for SHORT)
        let entry_positions = if let (Some(mod_up), Some(str_up)) = (moderate_up, strong_up) {
            vec![
                mod_up.range[0],                           // moderate_up lower
                (mod_up.range[0] + mod_up.range[1]) / 2.0, // moderate_up center
                str_up.range[0],                           // strong_up lower
            ]
        } else {
            vec![0.8, 1.2, 1.8] // Fallback percentages
        };

        // Exit positions (DOWN ranges for SHORT)
        let exit_positions = if let (Some(mod_down), Some(str_down)) = (moderate_down, strong_down)
        {
            vec![
                (mod_down.range[0] + mod_down.range[1]) / 2.0, // moderate_down center
                str_down.range[1], // strong_down upper (closer to current)
                (str_down.range[0] + str_down.range[1]) / 2.0, // strong_down center
            ]
        } else {
            vec![-1.5, -2.5, -3.5] // Fallback percentages
        };

        (entry_positions, exit_positions)
    }

    /// Validate probability-driven orders for consistency and eliminate duplicates
    pub fn validate_orders(
        &self,
        entry_levels: &[OrderLevel; 3],
        exit_levels: &[OrderLevel; 3],
        stop_levels: &[OrderLevel; 3],
    ) -> crate::utils::error::Result<()> {
        // 1. Validate probability portions sum to 1.0
        let (entry_portions, exit_portions) = self.extract_probability_portions();
        let entry_sum: f64 = entry_portions.iter().sum();
        let exit_sum: f64 = exit_portions.iter().sum();

        if (entry_sum - 1.0).abs() > 0.001 {
            return Err(crate::utils::error::VangaError::PredictionError(format!(
                "Entry portions sum to {:.6}, expected 1.0",
                entry_sum
            )));
        }

        if (exit_sum - 1.0).abs() > 0.001 {
            return Err(crate::utils::error::VangaError::PredictionError(format!(
                "Exit portions sum to {:.6}, expected 1.0",
                exit_sum
            )));
        }

        // 2. Validate no duplicate prices
        let entry_prices: Vec<f64> = entry_levels.iter().map(|l| l.price).collect();
        let exit_prices: Vec<f64> = exit_levels.iter().map(|l| l.price).collect();
        let stop_prices: Vec<f64> = stop_levels.iter().map(|l| l.price).collect();

        for i in 0..3 {
            for j in (i + 1)..3 {
                if (entry_prices[i] - entry_prices[j]).abs() < 0.01 {
                    return Err(crate::utils::error::VangaError::PredictionError(format!(
                        "Duplicate entry prices: {:.2} and {:.2}",
                        entry_prices[i], entry_prices[j]
                    )));
                }
                if (exit_prices[i] - exit_prices[j]).abs() < 0.01 {
                    return Err(crate::utils::error::VangaError::PredictionError(format!(
                        "Duplicate exit prices: {:.2} and {:.2}",
                        exit_prices[i], exit_prices[j]
                    )));
                }
                if (stop_prices[i] - stop_prices[j]).abs() < 0.01 {
                    return Err(crate::utils::error::VangaError::PredictionError(format!(
                        "Duplicate stop prices: {:.2} and {:.2}",
                        stop_prices[i], stop_prices[j]
                    )));
                }
            }
        }

        // 3. Validate proper order sequencing for SHORT
        let current_price =
            entry_levels[0].price / (1.0 + self.get_natural_price_positions().0[0] / 100.0);

        for level in entry_levels {
            if level.price <= current_price {
                return Err(crate::utils::error::VangaError::PredictionError(format!(
                    "SHORT entry price {:.2} must be above current price {:.2}",
                    level.price, current_price
                )));
            }
        }

        for level in exit_levels {
            if level.price >= current_price {
                return Err(crate::utils::error::VangaError::PredictionError(format!(
                    "SHORT exit price {:.2} must be below current price {:.2}",
                    level.price, current_price
                )));
            }
        }

        for (i, level) in stop_levels.iter().enumerate() {
            if level.price <= entry_levels[i].price {
                return Err(crate::utils::error::VangaError::PredictionError(format!(
                    "SHORT stop price {:.2} must be above entry price {:.2}",
                    level.price, entry_levels[i].price
                )));
            }
        }

        Ok(())
    }
}
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

        let (prediction, confidence) = probabilities
            .iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap();

        self.prediction = prediction.to_string();
        self.confidence = *confidence;
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
}

impl VolatilityPrediction {
    /// Calculate horizon-adaptive volatility metrics
    pub fn calculate_horizon_adaptive_volatility(
        &mut self,
        sequence_bandwidth_percent: f64,
        training_horizon: String,
        current_volatility_percentile: f64,
    ) {
        self.training_horizon = training_horizon;
        self.volatility_percentile = current_volatility_percentile;

        // Map 5-class probabilities to expected range for THIS horizon
        // Based on the actual sequence bandwidth, not hardcoded values
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

        // Adaptive stop loss distance (based on expected range)
        self.recommended_stop_distance_percent = self.expected_range_percent * 0.6; // 60% of expected range

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

    /// Update regime and confidence based on 5-class probabilities
    fn update_regime_and_confidence(&mut self) {
        let probabilities = [
            ("VERY_LOW", self.very_low_probability),
            ("LOW", self.low_probability),
            ("MEDIUM", self.medium_probability),
            ("HIGH", self.high_probability),
            ("VERY_HIGH", self.very_high_probability),
        ];

        let (regime, confidence) = probabilities
            .iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap();

        self.regime = regime.to_string();
        self.confidence = *confidence;
    }

    /// Create a new VolatilityPrediction with default values
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

        let (regime, confidence) = probabilities
            .iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap();

        self.regime = regime.to_string();
        self.confidence = *confidence;
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

        let (regime, confidence) = probabilities
            .iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap();

        self.regime = regime.to_string();
        self.confidence = *confidence;
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
            orders: TradingOrders::default(),
            confidence: 0.0,
            metadata: PredictionMetadata {
                model_version: "1.0.0".to_string(),
                generated_at: Utc::now(),
                feature_count: 0,   // Will be updated by formatter
                sequence_length: 0, // Will be updated by formatter
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
            orders: TradingOrders::default(),
            confidence: 0.0,
            metadata: PredictionMetadata {
                model_version: "1.0.0".to_string(),
                generated_at: Utc::now(),
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

    /// Set trading orders
    pub fn with_orders(mut self, orders: TradingOrders) -> Self {
        self.orders = orders;
        self
    }
}
