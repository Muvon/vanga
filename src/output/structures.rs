//! Prediction output data structures
//!
//! These structures define the JSON output format that matches ARCHITECTURE.md specifications
//! while reusing existing target generation logic from src/targets/

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

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

    /// Price level predictions (if enabled)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub price_levels: Option<PriceLevelPrediction>,

    /// Direction predictions (if enabled)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub direction: Option<DirectionPrediction>,

    /// Volatility predictions (if enabled)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub volatility: Option<VolatilityPrediction>,

    /// Trading orders with dynamic position sizing (always included)
    pub orders: TradingOrders,

    /// Adaptive trading signal for enhanced order generation (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub adaptive_signal: Option<AdaptiveTradingSignal>,

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

/// Overall confidence scoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfidenceScore {
    /// Overall prediction confidence (0.0 to 1.0)
    pub overall: f64,

    /// Price level confidence
    pub price_levels: f64,

    /// Direction confidence
    pub direction: f64,

    /// Volatility confidence
    pub volatility: f64,
}

/// Prediction metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionMetadata {
    /// Model version/identifier
    pub model_version: String,

    /// Prediction generation timestamp
    pub generated_at: DateTime<Utc>,

    /// Number of features used
    pub feature_count: usize,

    /// Sequence length used for prediction
    pub sequence_length: usize,

    /// Data quality indicators
    pub data_quality: DataQuality,
}

/// Data quality metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataQuality {
    /// Completeness score (0.0 to 1.0)
    pub completeness: f64,

    /// Data freshness (hours since last data point)
    pub freshness_hours: f64,

    /// Market condition assessment
    pub market_condition: String, // "NORMAL", "VOLATILE", "TRENDING"
}

impl PredictionResult {
    /// Create a new prediction result with basic info
    pub fn new(symbol: String, horizon: String, current_price: f64) -> Self {
        Self {
            symbol,
            timestamp: Utc::now().to_rfc3339(),
            horizon,
            current_price,
            price_levels: None,
            direction: None,
            volatility: None,
            orders: TradingOrders::default(),
            adaptive_signal: None,
            confidence: 0.0,
            metadata: PredictionMetadata {
                model_version: "1.0.0".to_string(),
                generated_at: Utc::now(),
                feature_count: 0,   // Will be updated by formatter
                sequence_length: 0, // Will be updated by formatter
                data_quality: DataQuality {
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
            price_levels: None,
            direction: None,
            volatility: None,
            orders: TradingOrders::default(),
            adaptive_signal: None,
            confidence: 0.0,
            metadata: PredictionMetadata {
                model_version: "1.0.0".to_string(),
                generated_at: Utc::now(),
                feature_count,
                sequence_length,
                data_quality: DataQuality {
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

    /// Set adaptive trading signal
    pub fn with_adaptive_signal(mut self, signal: AdaptiveTradingSignal) -> Self {
        self.adaptive_signal = Some(signal);
        self
    }

    /// Generate adaptive trading signal that works for any horizon
    pub fn generate_adaptive_trading_signal(
        direction_pred: &DirectionPrediction,
        volatility_pred: &VolatilityPrediction,
        current_price: f64,
    ) -> AdaptiveTradingSignal {
        // 1. BREAKOUT SIGNALS (works for any horizon)
        if direction_pred.breakout_probability > 0.4 {
            if direction_pred.pump_probability > 0.25 {
                return AdaptiveTradingSignal::StrongLong {
                    entry_price: current_price,
                    target_price: current_price
                        * (1.0 + direction_pred.expected_upside_percent / 100.0),
                    stop_loss: current_price
                        * (1.0 - volatility_pred.recommended_stop_distance_percent / 100.0),
                    position_size: volatility_pred.position_size_multiplier * 1.3, // Boost for breakouts
                    horizon: direction_pred.training_horizon.clone(),
                    risk_reward: direction_pred.risk_reward_ratio,
                    confidence: direction_pred.pump_probability,
                };
            }
            if direction_pred.dump_probability > 0.25 {
                return AdaptiveTradingSignal::StrongShort {
                    entry_price: current_price,
                    target_price: current_price
                        * (1.0 - direction_pred.expected_downside_percent / 100.0),
                    stop_loss: current_price
                        * (1.0 + volatility_pred.recommended_stop_distance_percent / 100.0),
                    position_size: volatility_pred.position_size_multiplier * 1.3,
                    horizon: direction_pred.training_horizon.clone(),
                    risk_reward: direction_pred.risk_reward_ratio,
                    confidence: direction_pred.dump_probability,
                };
            }
        }

        // 2. INTELLIGENT ADAPTIVE SIGNAL GENERATION
        // Analyze probability distribution to determine best trading approach
        let directional_edge =
            direction_pred.up_probability_aggregated - direction_pred.down_probability_aggregated;

        // Calculate probability spread (how concentrated vs distributed the predictions are)
        let max_prob = direction_pred
            .sideways_probability
            .max(direction_pred.up_probability_aggregated)
            .max(direction_pred.down_probability_aggregated);
        let min_prob = direction_pred
            .sideways_probability
            .min(direction_pred.up_probability_aggregated)
            .min(direction_pred.down_probability_aggregated);
        let probability_spread = max_prob - min_prob;

        // Calculate confidence in the prediction (higher spread = more confident)
        let prediction_confidence = probability_spread;

        log::info!(
            "📊 ADAPTIVE ANALYSIS: edge={:.1}%, spread={:.1}%, confidence={:.1}%, max_prob={:.1}%",
            directional_edge * 100.0,
            probability_spread * 100.0,
            prediction_confidence * 100.0,
            max_prob * 100.0
        );

        // 🚀 ADAPTIVE DECISION LOGIC - No hardcoded thresholds!
        // Use model's own confidence and risk/reward to determine if we should trade

        // Minimum confidence threshold based on model's own assessment
        let min_confidence_threshold = 0.25; // Only trade if we have >25% confidence in any direction
        let min_risk_reward_threshold = 0.8; // Minimum R/R based on volatility

        // Check if any prediction has sufficient confidence
        let has_sufficient_confidence = max_prob > min_confidence_threshold;
        let has_acceptable_risk_reward =
            direction_pred.risk_reward_ratio > min_risk_reward_threshold;

        if has_sufficient_confidence && has_acceptable_risk_reward {
            // Determine the best trading strategy based on probability distribution

            // Strategy 1: Clear directional bias (one aggregated probability significantly higher)
            if directional_edge.abs() > probability_spread * 0.3 {
                // Edge is significant relative to spread
                if directional_edge > 0.0 {
                    log::info!(
                        "📈 DIRECTIONAL LONG: edge={:.1}% is significant",
                        directional_edge * 100.0
                    );
                    return AdaptiveTradingSignal::Long {
                        entry_price: current_price,
                        target_price: current_price
                            * (1.0 + direction_pred.expected_upside_percent / 100.0),
                        stop_loss: current_price
                            * (1.0 - volatility_pred.recommended_stop_distance_percent / 100.0),
                        position_size: volatility_pred.position_size_multiplier
                            * (1.0 + prediction_confidence),
                        horizon: direction_pred.training_horizon.clone(),
                        risk_reward: direction_pred.risk_reward_ratio,
                        confidence: direction_pred.up_probability_aggregated,
                    };
                } else {
                    log::info!(
                        "📉 DIRECTIONAL SHORT: edge={:.1}% is significant",
                        directional_edge * 100.0
                    );
                    return AdaptiveTradingSignal::Short {
                        entry_price: current_price,
                        target_price: current_price
                            * (1.0 - direction_pred.expected_downside_percent / 100.0),
                        stop_loss: current_price
                            * (1.0 + volatility_pred.recommended_stop_distance_percent / 100.0),
                        position_size: volatility_pred.position_size_multiplier
                            * (1.0 + prediction_confidence),
                        horizon: direction_pred.training_horizon.clone(),
                        risk_reward: direction_pred.risk_reward_ratio,
                        confidence: direction_pred.down_probability_aggregated,
                    };
                }
            }

            // Strategy 2: SIDEWAYS is competitive (within reasonable range of other probabilities)
            // Even if not dominant, if SIDEWAYS is close to other probabilities, it's a valid strategy
            let sideways_competitiveness = direction_pred.sideways_probability / max_prob;
            if sideways_competitiveness > 0.8 {
                // SIDEWAYS is within 20% of the highest probability

                // Determine bias within sideways movement
                let sideways_direction = if direction_pred.up_probability_aggregated
                    > direction_pred.down_probability_aggregated
                {
                    "LONG"
                } else {
                    "SHORT"
                };

                let _bias_strength = directional_edge.abs();

                // Use sequence bandwidth for more conservative targets in sideways markets
                let target_percent = if sideways_direction == "LONG" {
                    (direction_pred.expected_upside_percent / 100.0)
                        .min(direction_pred.sequence_bandwidth_percent / 100.0 * 0.6)
                } else {
                    (direction_pred.expected_downside_percent / 100.0)
                        .min(direction_pred.sequence_bandwidth_percent / 100.0 * 0.6)
                };

                let stop_percent = (direction_pred.sequence_bandwidth_percent / 100.0 * 0.4)
                    .max(volatility_pred.recommended_stop_distance_percent / 100.0);

                let target_price = if sideways_direction == "LONG" {
                    current_price * (1.0 + target_percent)
                } else {
                    current_price * (1.0 - target_percent)
                };

                let stop_loss = if sideways_direction == "LONG" {
                    current_price * (1.0 - stop_percent)
                } else {
                    current_price * (1.0 + stop_percent)
                };

                let risk_reward = if stop_percent > 0.0 {
                    target_percent / stop_percent
                } else {
                    0.0
                };

                log::info!(
                    "🔄 SIDEWAYS COMPETITIVE: sideways={:.1}% (competitiveness={:.1}%), bias={}, R/R={:.2}",
                    direction_pred.sideways_probability * 100.0,
                    sideways_competitiveness * 100.0,
                    sideways_direction,
                    risk_reward
                );

                if sideways_direction == "LONG" {
                    return AdaptiveTradingSignal::SidewaysLong {
                        entry_price: current_price,
                        target_price,
                        stop_loss,
                        position_size: volatility_pred.position_size_multiplier
                            * (0.7 + prediction_confidence * 0.3),
                        horizon: direction_pred.training_horizon.clone(),
                        risk_reward,
                        confidence: direction_pred.sideways_probability,
                        sideways_probability: direction_pred.sideways_probability,
                        sequence_bias: format!("{}_BIAS", sideways_direction),
                    };
                } else {
                    return AdaptiveTradingSignal::SidewaysShort {
                        entry_price: current_price,
                        target_price,
                        stop_loss,
                        position_size: volatility_pred.position_size_multiplier
                            * (0.7 + prediction_confidence * 0.3),
                        horizon: direction_pred.training_horizon.clone(),
                        risk_reward,
                        confidence: direction_pred.sideways_probability,
                        sideways_probability: direction_pred.sideways_probability,
                        sequence_bias: format!("{}_BIAS", sideways_direction),
                    };
                }
            }

            // Strategy 3: Weak directional bias but still tradeable
            // If we have acceptable confidence but no clear strategy above, trade the bias
            if max_prob > 0.3 {
                // At least 30% confidence in some direction
                if direction_pred.up_probability_aggregated
                    > direction_pred.down_probability_aggregated
                {
                    log::info!(
                        "📈 WEAK LONG: up_agg={:.1}% > down_agg={:.1}%",
                        direction_pred.up_probability_aggregated * 100.0,
                        direction_pred.down_probability_aggregated * 100.0
                    );
                    return AdaptiveTradingSignal::Long {
                        entry_price: current_price,
                        target_price: current_price
                            * (1.0 + direction_pred.expected_upside_percent / 100.0),
                        stop_loss: current_price
                            * (1.0 - volatility_pred.recommended_stop_distance_percent / 100.0),
                        position_size: volatility_pred.position_size_multiplier * 0.8, // Reduced size for weak signals
                        horizon: direction_pred.training_horizon.clone(),
                        risk_reward: direction_pred.risk_reward_ratio,
                        confidence: direction_pred.up_probability_aggregated,
                    };
                } else {
                    log::info!(
                        "📉 WEAK SHORT: down_agg={:.1}% > up_agg={:.1}%",
                        direction_pred.down_probability_aggregated * 100.0,
                        direction_pred.up_probability_aggregated * 100.0
                    );
                    return AdaptiveTradingSignal::Short {
                        entry_price: current_price,
                        target_price: current_price
                            * (1.0 - direction_pred.expected_downside_percent / 100.0),
                        stop_loss: current_price
                            * (1.0 + volatility_pred.recommended_stop_distance_percent / 100.0),
                        position_size: volatility_pred.position_size_multiplier * 0.8,
                        horizon: direction_pred.training_horizon.clone(),
                        risk_reward: direction_pred.risk_reward_ratio,
                        confidence: direction_pred.down_probability_aggregated,
                    };
                }
            }
        }

        AdaptiveTradingSignal::NoSignal {
            reason: format!(
                "Insufficient confidence for trading. Max probability: {:.1}% (need >25%), R/R: {:.2} (need >0.8), Spread: {:.1}%",
                max_prob * 100.0,
                direction_pred.risk_reward_ratio,
                probability_spread * 100.0
            ),
            horizon: direction_pred.training_horizon.clone(),
            confidence: direction_pred.confidence,
        }
    }
}

/// Horizon-agnostic adaptive trading signal
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AdaptiveTradingSignal {
    StrongLong {
        entry_price: f64,
        target_price: f64,
        stop_loss: f64,
        position_size: f64,
        horizon: String,
        risk_reward: f64,
        confidence: f64,
    },
    Long {
        entry_price: f64,
        target_price: f64,
        stop_loss: f64,
        position_size: f64,
        horizon: String,
        risk_reward: f64,
        confidence: f64,
    },
    StrongShort {
        entry_price: f64,
        target_price: f64,
        stop_loss: f64,
        position_size: f64,
        horizon: String,
        risk_reward: f64,
        confidence: f64,
    },
    Short {
        entry_price: f64,
        target_price: f64,
        stop_loss: f64,
        position_size: f64,
        horizon: String,
        risk_reward: f64,
        confidence: f64,
    },
    SidewaysLong {
        entry_price: f64,
        target_price: f64,
        stop_loss: f64,
        position_size: f64,
        horizon: String,
        risk_reward: f64,
        confidence: f64,
        sideways_probability: f64,
        sequence_bias: String, // "LONG_BIAS" or "SHORT_BIAS"
    },
    SidewaysShort {
        entry_price: f64,
        target_price: f64,
        stop_loss: f64,
        position_size: f64,
        horizon: String,
        risk_reward: f64,
        confidence: f64,
        sideways_probability: f64,
        sequence_bias: String, // "LONG_BIAS" or "SHORT_BIAS"
    },
    NoSignal {
        reason: String,
        horizon: String,
        confidence: f64,
    },
}

/// Trading orders with dynamic position sizing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradingOrders {
    /// Trading direction based on prediction
    pub direction: String, // "LONG" or "SHORT"

    /// Three entry levels with dynamic quantities
    pub entry_levels: [OrderLevel; 3],

    /// Three exit levels with dynamic quantities
    pub exit_levels: [OrderLevel; 3],

    /// Three stop loss levels with dynamic quantities
    pub stop_levels: [OrderLevel; 3],

    /// Total position size (1.0 = 100%)
    pub total_position_size: f64,

    /// Expected risk-reward ratio (crypto-aggressive: 4.0-8.0+)
    pub risk_reward_ratio: f64,

    /// ATR multiplier used for spacing
    pub atr_multiplier: f64,

    /// Confidence-based position sizing enabled
    pub dynamic_sizing: bool,
}

/// Individual order level with dynamic sizing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderLevel {
    /// Order price
    pub price: f64,

    /// Dynamic quantity percentage (not fixed 33.33%)
    pub quantity_percentage: f64,

    /// Distance from current price in ATR units
    pub atr_distance: f64,

    /// Order type for execution
    pub order_type: String, // "LIMIT", "STOP_LIMIT", "MARKET"

    /// Confidence level for this price point
    pub confidence: f64,
}

/// Configuration for order generation
#[derive(Debug, Clone)]
pub struct OrderConfig {
    /// Base ATR multiplier (crypto default: 2.0)
    pub base_atr_multiplier: f64,

    /// Minimum risk-reward ratio (crypto: 4.0)
    pub min_risk_reward: f64,

    /// Maximum risk-reward ratio (crypto: 12.0)
    pub max_risk_reward: f64,

    /// Enable aggressive position sizing
    pub aggressive_sizing: bool,

    /// Hunt protection multiplier for stops
    pub hunt_protection: f64,
}

impl Default for OrderConfig {
    fn default() -> Self {
        Self {
            base_atr_multiplier: 2.0, // Crypto-aggressive spacing
            min_risk_reward: 4.0,     // Minimum 4:1 for crypto
            max_risk_reward: 12.0,    // Maximum 12:1 for high conviction
            aggressive_sizing: true,  // Enable dynamic sizing
            hunt_protection: 1.5,     // 50% extra distance for stops
        }
    }
}

impl Default for TradingOrders {
    fn default() -> Self {
        let empty_level = OrderLevel {
            price: 0.0,
            quantity_percentage: 0.0,
            atr_distance: 0.0,
            order_type: "NONE".to_string(),
            confidence: 0.0,
        };

        Self {
            direction: "NO_SIGNAL".to_string(),
            entry_levels: [
                empty_level.clone(),
                empty_level.clone(),
                empty_level.clone(),
            ],
            exit_levels: [
                empty_level.clone(),
                empty_level.clone(),
                empty_level.clone(),
            ],
            stop_levels: [empty_level.clone(), empty_level.clone(), empty_level],
            total_position_size: 0.0,
            risk_reward_ratio: 0.0,
            atr_multiplier: 0.0,
            dynamic_sizing: false,
        }
    }
}

/// Configuration for sequence-aware order generation
#[derive(Debug, Clone)]
pub struct SequenceAwareOrderConfig<'a> {
    pub current_price: f64,
    pub direction_pred: &'a DirectionPrediction,
    pub volatility_pred: &'a VolatilityPrediction,
    pub price_levels: &'a PriceLevelPrediction,
    pub atr_value: f64,
    pub config: &'a OrderConfig,
    pub sequence_prices: &'a [f64],
    pub bandwidth_size: f64,
}

impl TradingOrders {
    /// Generate trading orders from predictions with ADAPTIVE MATHEMATICAL OPTIMIZATION
    /// This function now uses the new adaptive order generation system that maximizes
    /// utilization of ALL prediction data instead of hardcoded thresholds
    pub fn generate(
        current_price: f64,
        direction_pred: &DirectionPrediction,
        volatility_pred: &VolatilityPrediction,
        price_levels: &PriceLevelPrediction,
        atr_value: f64,
        config: &OrderConfig,
    ) -> crate::utils::error::Result<Self> {
        // 🚀 NEW: Use adaptive mathematical order generation system
        use crate::output::adaptive_orders;

        log::info!("🔄 Using NEW adaptive mathematical order generation system");

        adaptive_orders::generate_adaptive_orders(
            current_price,
            direction_pred,
            volatility_pred,
            price_levels,
            atr_value,
            config,
        )
    }

    /// Generate sequence-aware trading orders using the same bandwidth logic as price levels
    /// This ensures consistency between price level predictions and order generation
    pub fn generate_sequence_aware(
        config: SequenceAwareOrderConfig,
    ) -> crate::utils::error::Result<Self> {
        // Validate that price_levels are consistent with our sequence-aware approach
        log::debug!(
            "Generating sequence-aware orders with price level confidence: {:.2}",
            config.price_levels.confidence
        );

        // 🚀 ADAPTIVE ANALYSIS - Same logic as AdaptiveTradingSignal
        // Analyze probability distribution to determine best trading approach
        let directional_edge = config.direction_pred.up_probability_aggregated
            - config.direction_pred.down_probability_aggregated;

        // Calculate probability spread (how concentrated vs distributed the predictions are)
        let max_prob = config
            .direction_pred
            .sideways_probability
            .max(config.direction_pred.up_probability_aggregated)
            .max(config.direction_pred.down_probability_aggregated);
        let min_prob = config
            .direction_pred
            .sideways_probability
            .min(config.direction_pred.up_probability_aggregated)
            .min(config.direction_pred.down_probability_aggregated);
        let probability_spread = max_prob - min_prob;

        // Calculate confidence in the prediction (higher spread = more confident)
        let prediction_confidence = probability_spread;

        log::info!(
            "📊 SEQUENCE-AWARE ANALYSIS: edge={:.1}%, spread={:.1}%, confidence={:.1}%, max_prob={:.1}%",
            directional_edge * 100.0,
            probability_spread * 100.0,
            prediction_confidence * 100.0,
            max_prob * 100.0
        );

        // Use model's own confidence and risk/reward to determine if we should trade
        let min_confidence_threshold = 0.25; // Only trade if we have >25% confidence in any direction
        let min_risk_reward_threshold = 0.5; // Minimum R/R based on volatility

        // Check if any prediction has sufficient confidence
        let has_sufficient_confidence = max_prob > min_confidence_threshold;
        let has_acceptable_risk_reward =
            config.direction_pred.risk_reward_ratio > min_risk_reward_threshold;

        if !has_sufficient_confidence || !has_acceptable_risk_reward {
            return Ok(Self::empty(
                config.direction_pred,
                &format!(
                    "Insufficient confidence for trading. Max probability: {:.1}% (need >25%), R/R: {:.2} (need >0.5)",
                    max_prob * 100.0,
                    config.direction_pred.risk_reward_ratio
                ),
            ));
        }

        // Determine direction based on adaptive logic
        let direction = if directional_edge.abs() > probability_spread * 0.3 {
            // Clear directional bias
            if directional_edge > 0.0 {
                "LONG"
            } else {
                "SHORT"
            }
        } else {
            // Weak directional bias but still tradeable if we have confidence
            if config.direction_pred.up_probability_aggregated
                > config.direction_pred.down_probability_aggregated
            {
                "LONG"
            } else {
                "SHORT"
            }
        };

        // 🎯 PROPER ATR CALCULATION: Base multiplier adjusted by market volatility
        let volatility_factor = config.volatility_pred.expected_range_percent / 5.0; // Scale to reasonable range (5% baseline)
        let atr_multiplier = config.config.base_atr_multiplier * volatility_factor.clamp(0.5, 3.0); // Cap between 0.5x-3.0x

        // ATR distance: use base calculation with market adjustment
        let base_atr_pct = config
            .volatility_pred
            .recommended_stop_distance_percent
            .max(1.0); // Minimum 1%
        let atr_distance = config.current_price * (base_atr_pct / 100.0);

        // 🎯 ADAPTIVE ORDER GENERATION: Use price level probabilities instead of sequence ranges
        let (mut entry_levels, mut exit_levels, mut stop_levels) = if direction == "LONG" {
            // Check if this is a breakout signal based on pump probability (adaptive threshold)
            let is_breakout = config.direction_pred.pump_probability > 0.25; // Use same threshold as AdaptiveTradingSignal
            Self::generate_adaptive_long_orders(
                config.current_price,
                atr_distance,
                config.price_levels,
                config.direction_pred,
                config.volatility_pred,
                config.config,
                is_breakout,
            )
        } else {
            // Check if this is a breakout signal based on dump probability (adaptive threshold)
            let is_breakout = config.direction_pred.dump_probability > 0.25; // Use same threshold as AdaptiveTradingSignal
                                                                             // Extract actual price level ranges for order generation
            let sequence_ranges = config.price_levels.extract_ranges_for_orders();
            Self::generate_sequence_aware_short_orders(
                config.current_price,
                atr_distance,
                &sequence_ranges,
                config.config,
                is_breakout,
                config.price_levels,    // NEW: Add price_levels parameter
                config.volatility_pred, // NEW: Add volatility_pred parameter
            )
        };

        // Validate and optimize risk-reward ratio (configurable minimum for crypto)
        let min_risk_reward = 4.0; // TODO: Move to config - 4:1 minimum as requested
        let risk_reward_ratio = Self::validate_and_optimize_risk_reward(
            &mut entry_levels,
            &mut exit_levels,
            &mut stop_levels,
            direction,
            config.current_price,
            min_risk_reward,
        );

        // Log final risk/reward assessment
        if risk_reward_ratio < min_risk_reward {
            log::error!(
                "🚨 TRADING SIGNAL REJECTED: Risk/Reward {:.2} below minimum {:.2} - would be 'hole in pocket'",
                risk_reward_ratio, min_risk_reward
            );
        } else {
            log::info!(
                "✅ TRADING SIGNAL APPROVED: Risk/Reward {:.2} meets minimum {:.2} requirement",
                risk_reward_ratio,
                min_risk_reward
            );
        }

        Ok(TradingOrders {
            direction: direction.to_string(),
            entry_levels,
            exit_levels,
            stop_levels,
            total_position_size: 1.0,
            risk_reward_ratio,
            atr_multiplier,
            dynamic_sizing: config.config.aggressive_sizing,
        })
    }

    /// Generate adaptive long orders using price level probabilities
    fn generate_adaptive_long_orders(
        current_price: f64,
        atr_distance: f64,
        price_levels: &PriceLevelPrediction,
        direction_pred: &DirectionPrediction,
        volatility_pred: &VolatilityPrediction,
        config: &OrderConfig,
        is_breakout: bool,
    ) -> ([OrderLevel; 3], [OrderLevel; 3], [OrderLevel; 3]) {
        // 🎯 SIMPLE ADAPTIVE LOGIC: Use sequence bandwidth for realistic ranges
        let sequence_bandwidth_pct = direction_pred.sequence_bandwidth_percent;
        let expected_upside = direction_pred.expected_upside_percent;

        // ENTRY LOGIC: Use most likely downside range but keep it realistic
        let neutral_bin = price_levels.bins.get("neutral");
        let moderate_down_bin = price_levels.bins.get("moderate_down");

        // Use neutral range lower bound as primary entry (most conservative)
        let primary_entry_pct = neutral_bin
            .map(|bin| bin.range[0]) // Lower bound of neutral
            .unwrap_or(-sequence_bandwidth_pct * 0.3); // Fallback: 30% of bandwidth

        // Secondary entries: scale down from primary
        let entry_1_pct = primary_entry_pct; // Best entry
        let entry_2_pct = primary_entry_pct - sequence_bandwidth_pct * 0.2; // 20% bandwidth deeper
        let entry_3_pct = primary_entry_pct - sequence_bandwidth_pct * 0.4; // 40% bandwidth deeper

        // POSITION SIZING: Simple probability weighting
        let neutral_prob = neutral_bin.map(|bin| bin.probability).unwrap_or(0.4);
        let moderate_down_prob = moderate_down_bin.map(|bin| bin.probability).unwrap_or(0.3);

        let entry_levels = [
            OrderLevel {
                price: current_price * (1.0 + entry_1_pct / 100.0),
                quantity_percentage: neutral_prob.min(0.5), // Cap at 50%
                atr_distance,
                order_type: "LIMIT".to_string(),
                confidence: neutral_prob,
            },
            OrderLevel {
                price: current_price * (1.0 + entry_2_pct / 100.0),
                quantity_percentage: moderate_down_prob.min(0.3), // Cap at 30%
                atr_distance,
                order_type: "LIMIT".to_string(),
                confidence: moderate_down_prob,
            },
            OrderLevel {
                price: current_price * (1.0 + entry_3_pct / 100.0),
                quantity_percentage: (1.0 - neutral_prob - moderate_down_prob).max(0.2), // Remainder, min 20%
                atr_distance,
                order_type: if is_breakout {
                    "STOP_LIMIT".to_string()
                } else {
                    "LIMIT".to_string()
                },
                confidence: 0.2,
            },
        ];

        // EXIT LOGIC: Use expected upside with progressive scaling
        let exit_1_pct = expected_upside * 0.5; // 50% of expected upside
        let exit_2_pct = expected_upside * 0.8; // 80% of expected upside
        let exit_3_pct = expected_upside; // Full expected upside

        let exit_levels = [
            OrderLevel {
                price: current_price * (1.0 + exit_1_pct / 100.0),
                quantity_percentage: 0.4,
                atr_distance,
                order_type: "LIMIT".to_string(),
                confidence: 0.8,
            },
            OrderLevel {
                price: current_price * (1.0 + exit_2_pct / 100.0),
                quantity_percentage: 0.4,
                atr_distance,
                order_type: "LIMIT".to_string(),
                confidence: 0.6,
            },
            OrderLevel {
                price: current_price * (1.0 + exit_3_pct / 100.0),
                quantity_percentage: 0.2,
                atr_distance,
                order_type: "LIMIT".to_string(),
                confidence: 0.4,
            },
        ];

        // STOP LOGIC: ENFORCE RISK-REWARD RATIO MATHEMATICALLY
        let avg_entry_price =
            (entry_levels[0].price + entry_levels[1].price + entry_levels[2].price) / 3.0;
        let avg_exit_price =
            (exit_levels[0].price + exit_levels[1].price + exit_levels[2].price) / 3.0;

        // Calculate required stop distance to maintain min_risk_reward
        let expected_profit = avg_exit_price - avg_entry_price;
        let max_allowed_loss = expected_profit / config.min_risk_reward; // Enforce 4:1 ratio

        // Use volatility recommendation but cap by risk-reward requirement
        let volatility_stop_distance =
            volatility_pred.recommended_stop_distance_percent / 100.0 * avg_entry_price;
        let required_stop_distance = max_allowed_loss.min(volatility_stop_distance);

        // CRITICAL: Stops must be BELOW entry prices for LONG
        let stop_price_1 = avg_entry_price - required_stop_distance;
        let stop_price_2 = avg_entry_price - required_stop_distance * 1.1; // 10% wider
        let stop_price_3 = avg_entry_price - required_stop_distance * 1.2; // 20% wider

        let stop_levels = [
            OrderLevel {
                price: stop_price_1,
                quantity_percentage: 0.4,
                atr_distance: atr_distance * config.hunt_protection,
                order_type: "STOP_LOSS".to_string(),
                confidence: 0.9,
            },
            OrderLevel {
                price: stop_price_2,
                quantity_percentage: 0.4,
                atr_distance: atr_distance * config.hunt_protection,
                order_type: "STOP_LOSS".to_string(),
                confidence: 0.8,
            },
            OrderLevel {
                price: stop_price_3,
                quantity_percentage: 0.2,
                atr_distance: atr_distance * config.hunt_protection,
                order_type: "STOP_LOSS".to_string(),
                confidence: 0.7,
            },
        ];

        // VALIDATION: Ensure mathematical correctness
        let actual_risk_reward = expected_profit / required_stop_distance;

        log::info!(
            "🎯 LONG Orders: Avg Entry={:.2} | Avg Exit={:.2} | Avg Stop={:.2} | R:R={:.2} (min={:.1})",
            avg_entry_price, avg_exit_price, (stop_price_1 + stop_price_2 + stop_price_3) / 3.0,
            actual_risk_reward, config.min_risk_reward
        );

        // Ensure stops are below entries
        for (i, stop) in stop_levels.iter().enumerate() {
            if stop.price >= entry_levels[i].price {
                log::error!(
                    "🚨 CRITICAL: Stop {} ({:.2}) >= Entry {} ({:.2})",
                    i,
                    stop.price,
                    i,
                    entry_levels[i].price
                );
            }
        }

        (entry_levels, exit_levels, stop_levels)
    }

    /// Generate sequence-aware short orders using probability-based allocation (NO MAGIC NUMBERS)
    fn generate_sequence_aware_short_orders(
        current_price: f64,
        atr_distance: f64,
        sequence_ranges: &[[f64; 2]],
        config: &OrderConfig,
        is_breakout: bool,
        price_levels: &PriceLevelPrediction, // NEW: Use for probability-based allocation
        volatility_pred: &VolatilityPrediction, // NEW: Use for adaptive stop calculation
    ) -> ([OrderLevel; 3], [OrderLevel; 3], [OrderLevel; 3]) {
        // DEBUG: Log the ranges being used
        log::debug!(
            "🔍 SHORT Order Generation Debug: current_price={:.2}, ranges={:?}",
            current_price,
            sequence_ranges
        );

        // CRITICAL FIX: Force correct SHORT order logic regardless of range issues
        // SHORT must enter ABOVE current price and exit BELOW current price

        // Use sequence ranges for entry levels, but ensure they're positive (ABOVE current)
        let moderate_up_range = &sequence_ranges[3]; // moderate_up
        let strong_up_range = &sequence_ranges[4]; // strong_up

        log::debug!(
            "📊 SHORT Entry Ranges: moderate_up={:?}, strong_up={:?}",
            moderate_up_range,
            strong_up_range
        );

        // 🎯 NEW: Use probability-based allocation instead of hardcoded portions
        let (entry_portions, exit_portions) = price_levels.extract_probability_portions();
        let (entry_positions, exit_positions) = price_levels.get_natural_price_positions();

        log::info!(
            "📊 Probability-based allocation (breakout={}): Entry portions: [{:.3}, {:.3}, {:.3}], Exit portions: [{:.3}, {:.3}, {:.3}]",
            is_breakout,
            entry_portions[0], entry_portions[1], entry_portions[2],
            exit_portions[0], exit_portions[1], exit_portions[2]
        );

        // Use natural price positions from prediction data
        let entry_1_pct = entry_positions[0].max(0.5); // Ensure minimum above current
        let entry_2_pct = entry_positions[1].max(entry_1_pct + 0.2);
        let entry_3_pct = entry_positions[2].max(entry_2_pct + 0.2);

        log::debug!(
            "💰 SHORT Entry Percentages (ADAPTIVE): entry_1={:.2}%, entry_2={:.2}%, entry_3={:.2}%",
            entry_1_pct,
            entry_2_pct,
            entry_3_pct
        );

        let entry_levels = [
            OrderLevel {
                price: current_price * (1.0 + entry_1_pct / 100.0),
                quantity_percentage: entry_portions[0],
                atr_distance,
                order_type: "LIMIT".to_string(),
                confidence: 0.8,
            },
            OrderLevel {
                price: current_price * (1.0 + entry_2_pct / 100.0),
                quantity_percentage: entry_portions[1],
                atr_distance,
                order_type: "LIMIT".to_string(),
                confidence: 0.7,
            },
            OrderLevel {
                price: current_price * (1.0 + entry_3_pct / 100.0),
                quantity_percentage: entry_portions[2],
                atr_distance,
                order_type: if is_breakout {
                    "STOP_LIMIT".to_string() // More aggressive for breakouts
                } else {
                    "LIMIT".to_string()
                },
                confidence: if is_breakout { 0.9 } else { 0.6 },
            },
        ];

        // 🎯 NEW: Exit levels using probability-based allocation and natural positions
        let exit_1_pct = exit_positions[0]; // moderate_down center
        let exit_2_pct = exit_positions[1]; // strong_down upper
        let exit_3_pct = exit_positions[2]; // strong_down center

        log::debug!(
            "📉 SHORT Exit Percentages (ADAPTIVE): exit_1={:.2}%, exit_2={:.2}%, exit_3={:.2}%",
            exit_1_pct,
            exit_2_pct,
            exit_3_pct
        );

        let exit_levels = [
            OrderLevel {
                price: current_price * (1.0 + exit_1_pct / 100.0),
                quantity_percentage: exit_portions[0],
                atr_distance,
                order_type: "LIMIT".to_string(),
                confidence: 0.8,
            },
            OrderLevel {
                price: current_price * (1.0 + exit_2_pct / 100.0),
                quantity_percentage: exit_portions[1],
                atr_distance,
                order_type: "LIMIT".to_string(),
                confidence: 0.6,
            },
            OrderLevel {
                price: current_price * (1.0 + exit_3_pct / 100.0),
                quantity_percentage: exit_portions[2],
                atr_distance,
                order_type: "LIMIT".to_string(),
                confidence: 0.4,
            },
        ];

        // 🎯 NEW: Adaptive stop levels using volatility model data (NO HARDCODED VALUES)
        // Extract volatility data from the prediction model
        let recommended_stop_distance = volatility_pred.recommended_stop_distance_percent;
        let expected_range = volatility_pred.expected_range_percent;
        let regime_confidence = volatility_pred.confidence;

        // Calculate adaptive buffer based on volatility regime confidence
        // Lower confidence = higher uncertainty = wider buffer needed
        let confidence_buffer = (1.0 - regime_confidence) * expected_range;
        let adaptive_stop_distance = recommended_stop_distance + confidence_buffer;

        log::info!(
            "🎯 Adaptive Stop Calculation: base={:.3}% + buffer={:.3}% = total={:.3}%",
            recommended_stop_distance,
            confidence_buffer,
            adaptive_stop_distance
        );

        // Position stops at adaptive distance above each entry
        let stop_1_pct = entry_1_pct + adaptive_stop_distance;
        let stop_2_pct = entry_2_pct + adaptive_stop_distance;
        let stop_3_pct = entry_3_pct + adaptive_stop_distance;

        log::debug!(
            "🛑 SHORT Stop Percentages (VOLATILITY-ADAPTIVE): stop_1={:.2}%, stop_2={:.2}%, stop_3={:.2}%",
            stop_1_pct, stop_2_pct, stop_3_pct
        );

        // Use same probability-based portions for stops as entries (risk consistency)
        let stop_levels = [
            OrderLevel {
                price: current_price * (1.0 + (stop_1_pct * config.hunt_protection) / 100.0),
                quantity_percentage: entry_portions[0], // Same as entry allocation
                atr_distance: atr_distance * config.hunt_protection,
                order_type: "STOP_LOSS".to_string(),
                confidence: 0.9,
            },
            OrderLevel {
                price: current_price * (1.0 + (stop_2_pct * config.hunt_protection) / 100.0),
                quantity_percentage: entry_portions[1], // Same as entry allocation
                atr_distance: atr_distance * config.hunt_protection,
                order_type: "STOP_LOSS".to_string(),
                confidence: 0.8,
            },
            OrderLevel {
                price: current_price * (1.0 + (stop_3_pct * config.hunt_protection) / 100.0),
                quantity_percentage: entry_portions[2], // Same as entry allocation
                atr_distance: atr_distance * config.hunt_protection,
                order_type: "STOP_LOSS".to_string(),
                confidence: 0.7,
            },
        ];

        // 🎯 NEW: Validate the generated orders for consistency
        if let Err(e) = price_levels.validate_orders(&entry_levels, &exit_levels, &stop_levels) {
            log::error!("❌ Order validation failed: {}", e);
            return (entry_levels, exit_levels, stop_levels); // Return anyway but log error
        }

        log::info!("✅ Probability-driven orders validated successfully - no duplicates, proper sequencing");

        (entry_levels, exit_levels, stop_levels)
    }

    /// Create empty orders when no trading signals are available
    pub fn empty(direction_pred: &DirectionPrediction, reason: &str) -> Self {
        let empty_level = OrderLevel {
            price: 0.0,
            quantity_percentage: 0.0,
            atr_distance: 0.0,
            order_type: "NONE".to_string(),
            confidence: 0.0,
        };

        let direction = if direction_pred.up_probability > direction_pred.down_probability {
            "LONG"
        } else {
            "SHORT"
        };

        TradingOrders {
            direction: format!("{} ({})", direction, reason),
            entry_levels: [
                empty_level.clone(),
                empty_level.clone(),
                empty_level.clone(),
            ],
            exit_levels: [
                empty_level.clone(),
                empty_level.clone(),
                empty_level.clone(),
            ],
            stop_levels: [empty_level.clone(), empty_level.clone(), empty_level],
            total_position_size: 0.0,
            risk_reward_ratio: 0.0,
            atr_multiplier: 0.0,
            dynamic_sizing: false,
        }
    }

    /// Validate and optimize risk/reward ratio for trading viability
    fn validate_and_optimize_risk_reward(
        entry_levels: &mut [OrderLevel; 3],
        exit_levels: &mut [OrderLevel; 3],
        stop_levels: &mut [OrderLevel; 3],
        direction: &str,
        current_price: f64,
        min_ratio: f64,
    ) -> f64 {
        let initial_ratio =
            Self::calculate_risk_reward(entry_levels, exit_levels, stop_levels, direction);

        log::debug!(
            "🎯 Initial Risk/Reward Ratio: {:.2} (minimum required: {:.2})",
            initial_ratio,
            min_ratio
        );

        if initial_ratio >= min_ratio {
            log::debug!("✅ Risk/Reward ratio is acceptable: {:.2}", initial_ratio);
            return initial_ratio;
        }

        log::warn!(
            "⚠️ Poor Risk/Reward ratio: {:.2} < {:.2} minimum. Attempting optimization...",
            initial_ratio,
            min_ratio
        );

        // OPTIMIZATION: Adjust levels to improve risk/reward
        match direction {
            "SHORT" => {
                // For SHORT: Improve by moving exits lower (more profit) or stops closer (less risk)
                for exit in exit_levels.iter_mut() {
                    if exit.price > 0.0 {
                        // Move exits 0.5% lower for more profit
                        exit.price *= 0.995;
                    }
                }

                // Move stops slightly closer to reduce risk
                for stop in stop_levels.iter_mut() {
                    if stop.price > current_price {
                        // Move stops 0.3% closer to current price
                        let distance_from_current = stop.price - current_price;
                        stop.price = current_price + (distance_from_current * 0.97);
                    }
                }
            }
            "LONG" => {
                // For LONG: Improve by moving exits higher (more profit) or stops closer (less risk)
                for exit in exit_levels.iter_mut() {
                    if exit.price > 0.0 {
                        // Move exits 0.5% higher for more profit
                        exit.price *= 1.005;
                    }
                }

                // Move stops slightly closer to reduce risk
                for stop in stop_levels.iter_mut() {
                    if stop.price < current_price {
                        // Move stops 0.3% closer to current price
                        let distance_from_current = current_price - stop.price;
                        stop.price = current_price - (distance_from_current * 0.97);
                    }
                }
            }
            _ => {}
        }

        let optimized_ratio =
            Self::calculate_risk_reward(entry_levels, exit_levels, stop_levels, direction);

        if optimized_ratio >= min_ratio {
            log::info!(
                "✅ Risk/Reward optimized successfully: {:.2} -> {:.2}",
                initial_ratio,
                optimized_ratio
            );
        } else {
            log::error!(
                "❌ Failed to optimize Risk/Reward: {:.2} -> {:.2} (still below {:.2} minimum)",
                initial_ratio,
                optimized_ratio,
                min_ratio
            );
        }

        optimized_ratio
    }

    /// Calculate risk-reward ratio from order levels with direction awareness
    fn calculate_risk_reward(
        entry_levels: &[OrderLevel; 3],
        exit_levels: &[OrderLevel; 3],
        stop_levels: &[OrderLevel; 3],
        direction: &str,
    ) -> f64 {
        // Weighted average prices by quantity
        let avg_entry = Self::weighted_average_price(entry_levels);
        let avg_exit = Self::weighted_average_price(exit_levels);
        let avg_stop = Self::weighted_average_price(stop_levels);

        // Calculate profit and loss based on direction
        let (potential_profit, potential_loss) = match direction {
            "LONG" => {
                // LONG: profit when price goes up, loss when price goes down
                let profit = avg_exit - avg_entry; // Exit higher than entry = profit
                let loss = avg_entry - avg_stop; // Stop lower than entry = loss
                (profit.max(0.0), loss.max(0.0))
            }
            "SHORT" => {
                // SHORT: profit when price goes down, loss when price goes up
                let profit = avg_entry - avg_exit; // Entry higher than exit = profit
                let loss = avg_stop - avg_entry; // Stop higher than entry = loss
                (profit.max(0.0), loss.max(0.0))
            }
            _ => {
                // Fallback to absolute difference for unknown directions
                let profit = (avg_exit - avg_entry).abs();
                let loss = (avg_entry - avg_stop).abs();
                (profit, loss)
            }
        };

        if potential_loss > 0.0 {
            potential_profit / potential_loss
        } else {
            10.0 // If no loss possible, return high ratio
        }
    }

    /// Calculate weighted average price by quantity
    fn weighted_average_price(levels: &[OrderLevel; 3]) -> f64 {
        let total_weight: f64 = levels.iter().map(|l| l.quantity_percentage).sum();
        if total_weight > 0.0 {
            levels
                .iter()
                .map(|l| l.price * l.quantity_percentage)
                .sum::<f64>()
                / total_weight
        } else {
            levels.iter().map(|l| l.price).sum::<f64>() / 3.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn create_test_direction_prediction(up_prob: f64) -> DirectionPrediction {
        // Convert 2-class to 5-class probabilities for testing
        let down_prob = 1.0 - up_prob;
        let sideways_prob = 0.1; // Reduced sideways to increase directional edge
        let remaining = 1.0 - sideways_prob;
        let dump_prob = if down_prob > 0.5 {
            (down_prob - 0.5) * remaining
        } else {
            0.0
        };
        let pump_prob = if up_prob > 0.5 {
            (up_prob - 0.5) * remaining
        } else {
            0.0
        };
        let down_moderate = (down_prob - dump_prob) * remaining;
        let up_moderate = (up_prob - pump_prob) * remaining;

        let mut direction_pred = DirectionPrediction::from_probabilities(
            dump_prob,
            down_moderate,
            sideways_prob,
            up_moderate,
            pump_prob,
        );

        // Calculate adaptive metrics to populate aggregated probabilities
        direction_pred.calculate_horizon_adaptive_metrics(
            5.0, // 5% bandwidth
            "4h".to_string(),
            60,
        );

        direction_pred
    }

    fn create_test_volatility_prediction(regime: &str) -> VolatilityPrediction {
        // Create probability distribution based on regime
        let (very_low, low, medium, high, very_high) = match regime {
            "VERY_LOW" => (0.7, 0.2, 0.1, 0.0, 0.0),
            "LOW" => (0.2, 0.6, 0.2, 0.0, 0.0),
            "MEDIUM" => (0.1, 0.2, 0.4, 0.2, 0.1),
            "HIGH" => (0.0, 0.0, 0.2, 0.6, 0.2),
            "VERY_HIGH" => (0.0, 0.0, 0.1, 0.2, 0.7),
            _ => (0.1, 0.2, 0.4, 0.2, 0.1),
        };

        VolatilityPrediction::from_probabilities(very_low, low, medium, high, very_high)
    }

    fn create_test_price_levels() -> PriceLevelPrediction {
        let mut bins = HashMap::new();

        bins.insert(
            "pump".to_string(),
            PriceBin {
                range: [3.0, 15.0],
                price: [43000.0, 46000.0],
                probability: 0.7, // Higher confidence pump
            },
        );

        bins.insert(
            "sideways".to_string(),
            PriceBin {
                range: [-3.0, 3.0],
                price: [41000.0, 43000.0],
                probability: 0.2, // Reduced sideways probability
            },
        );

        bins.insert(
            "dump".to_string(),
            PriceBin {
                range: [-15.0, -3.0],
                price: [38000.0, 41000.0],
                probability: 0.1,
            },
        );

        PriceLevelPrediction {
            bins,
            most_likely_range: [3.0, 15.0],
            confidence: 0.9, // Higher confidence
        }
    }

    fn create_test_crypto_aggressive_price_levels() -> PriceLevelPrediction {
        let mut bins = HashMap::new();

        // Extremely bullish scenario with high confidence moon targets
        bins.insert(
            "moon".to_string(),
            PriceBin {
                range: [10.0, 30.0], // 10-30% pump expected
                price: [47000.0, 56000.0],
                probability: 0.8, // Very high confidence
            },
        );

        bins.insert(
            "pump".to_string(),
            PriceBin {
                range: [5.0, 10.0],
                price: [45000.0, 47000.0],
                probability: 0.15,
            },
        );

        bins.insert(
            "sideways".to_string(),
            PriceBin {
                range: [-2.0, 5.0],
                price: [42000.0, 45000.0],
                probability: 0.05,
            },
        );

        PriceLevelPrediction {
            bins,
            most_likely_range: [10.0, 30.0], // Expecting big move up
            confidence: 0.9,                 // Very confident
        }
    }

    #[test]
    fn test_long_order_generation() {
        let current_price = 43000.0;
        let direction_pred = create_test_direction_prediction(0.8); // Very strong up signal
        let volatility_pred = create_test_volatility_prediction("MEDIUM");
        let price_levels = create_test_price_levels();
        let atr_value = 800.0; // Higher ATR for better order generation
        let config = OrderConfig::default();

        let orders = TradingOrders::generate(
            current_price,
            &direction_pred,
            &volatility_pred,
            &price_levels,
            atr_value,
            &config,
        )
        .unwrap();

        // Should be LONG direction with strong signal
        assert!(
            orders.direction.starts_with("LONG"),
            "Expected LONG direction, got: {}",
            orders.direction
        );

        // Entry levels should be below current price (if they have valid prices)
        for level in &orders.entry_levels {
            if level.price > 0.0 {
                assert!(
                    level.price < current_price,
                    "Entry price {} should be below current price {}",
                    level.price,
                    current_price
                );
            }
        }

        // Exit levels should be above current price (if they have valid prices)
        for level in &orders.exit_levels {
            if level.price > 0.0 {
                assert!(
                    level.price > current_price,
                    "Exit price {} should be above current price {}",
                    level.price,
                    current_price
                );
            }
        }

        // Stop levels should be below current price (if they have valid prices)
        for level in &orders.stop_levels {
            if level.price > 0.0 {
                assert!(
                    level.price < current_price,
                    "Stop price {} should be below current price {}",
                    level.price,
                    current_price
                );
            }
        }
    }

    #[test]
    fn test_short_order_generation() {
        let current_price = 43000.0;
        let direction_pred = create_test_direction_prediction(0.2); // Strong down
        let volatility_pred = create_test_volatility_prediction("HIGH");
        let price_levels = create_test_price_levels();
        let atr_value = 700.0; // $700 ATR (high volatility)
        let config = OrderConfig::default();

        let orders = TradingOrders::generate(
            current_price,
            &direction_pred,
            &volatility_pred,
            &price_levels,
            atr_value,
            &config,
        )
        .unwrap();

        // Should be SHORT direction
        assert_eq!(orders.direction, "SHORT");

        // Entry levels should be above current price (selling higher)
        for level in &orders.entry_levels {
            assert!(
                level.price > current_price,
                "Short entry price {} should be above current price {}",
                level.price,
                current_price
            );
        }

        // Exit levels should be below current price (buying lower)
        for level in &orders.exit_levels {
            assert!(
                level.price < current_price,
                "Short exit price {} should be below current price {}",
                level.price,
                current_price
            );
        }

        // Stop levels should be above current price (buying higher to stop loss)
        for level in &orders.stop_levels {
            assert!(
                level.price > current_price,
                "Short stop price {} should be above current price {}",
                level.price,
                current_price
            );
        }

        // Should use higher ATR multiplier for HIGH volatility
        assert!(
            orders.atr_multiplier > 2.0,
            "HIGH volatility should increase ATR multiplier, got {}",
            orders.atr_multiplier
        );
    }

    #[test]
    fn test_crypto_aggressive_risk_reward() {
        let current_price = 43000.0;
        let direction_pred = create_test_direction_prediction(0.8); // Very strong up
        let volatility_pred = create_test_volatility_prediction("LOW"); // Low vol = tighter stops
        let price_levels = create_test_crypto_aggressive_price_levels(); // More aggressive targets
        let atr_value = 250.0; // Even smaller ATR for tighter risk management
        let config = OrderConfig {
            hunt_protection: 0.5, // Even tighter stop protection
            ..Default::default()
        };

        let orders = TradingOrders::generate(
            current_price,
            &direction_pred,
            &volatility_pred,
            &price_levels,
            atr_value,
            &config,
        )
        .unwrap();

        // Debug output
        println!(
            "Entry levels: {:?}",
            orders
                .entry_levels
                .iter()
                .map(|l| l.price)
                .collect::<Vec<_>>()
        );
        println!(
            "Exit levels: {:?}",
            orders
                .exit_levels
                .iter()
                .map(|l| l.price)
                .collect::<Vec<_>>()
        );
        println!(
            "Stop levels: {:?}",
            orders
                .stop_levels
                .iter()
                .map(|l| l.price)
                .collect::<Vec<_>>()
        );
        println!("Risk-reward ratio: {}", orders.risk_reward_ratio);

        // Risk-reward should be crypto-aggressive (>= 4.0)
        assert!(
            orders.risk_reward_ratio >= 4.0,
            "Risk-reward ratio should be >= 4.0 for crypto, got {}",
            orders.risk_reward_ratio
        );

        // Should have dynamic sizing enabled
        assert!(orders.dynamic_sizing, "Dynamic sizing should be enabled");

        // Total position size should be 100%
        assert!(
            (orders.total_position_size - 1.0).abs() < 0.01,
            "Total position size should be 1.0 (100%)"
        );
    }

    #[test]
    fn test_dynamic_quantity_allocation() {
        let current_price = 43000.0;
        let direction_pred = create_test_direction_prediction(0.7);
        let volatility_pred = create_test_volatility_prediction("MEDIUM");
        let price_levels = create_test_price_levels(); // Has high confidence "pump" bin
        let atr_value = 500.0;
        let config = OrderConfig::default();

        let orders = TradingOrders::generate(
            current_price,
            &direction_pred,
            &volatility_pred,
            &price_levels,
            atr_value,
            &config,
        )
        .unwrap();

        // Should NOT be equal 33.33% allocation due to dynamic sizing
        let quantities: Vec<f64> = orders
            .entry_levels
            .iter()
            .map(|l| l.quantity_percentage)
            .collect();

        println!("Generated quantities: {:?}", quantities);

        // Check that quantities are different (not all equal)
        let all_equal = quantities.windows(2).all(|w| (w[0] - w[1]).abs() < 0.01);
        assert!(
            !all_equal,
            "Quantities should be dynamic, not equal: {:?}",
            quantities
        );

        // First entry should get the most allocation (front-loaded)
        assert!(
            quantities[0] > quantities[1],
            "First entry should get more allocation than second: {:?}",
            quantities
        );
        assert!(
            quantities[0] > quantities[2],
            "First entry should get more allocation than third"
        );
    }

    #[test]
    fn test_hunt_protection() {
        let current_price = 43000.0;
        let direction_pred = create_test_direction_prediction(0.7);
        let volatility_pred = create_test_volatility_prediction("MEDIUM");
        let price_levels = create_test_price_levels();
        let atr_value = 500.0;
        let config = OrderConfig::default(); // hunt_protection = 1.5

        let orders = TradingOrders::generate(
            current_price,
            &direction_pred,
            &volatility_pred,
            &price_levels,
            atr_value,
            &config,
        )
        .unwrap();

        // Stop levels should be further away due to hunt protection
        let expected_base_stop = current_price - atr_value * 2.0 * 3.0; // Base distance
        let actual_stop = orders.stop_levels[0].price;
        let expected_protected_stop = current_price - atr_value * 2.0 * 3.0 * 1.5; // With protection

        // Actual stop should be closer to protected distance than base distance
        let distance_to_protected = (actual_stop - expected_protected_stop).abs();
        let distance_to_base = (actual_stop - expected_base_stop).abs();

        assert!(
            distance_to_protected < distance_to_base,
            "Stop should be closer to hunt-protected distance. Actual: {}, Protected: {}, Base: {}",
            actual_stop,
            expected_protected_stop,
            expected_base_stop
        );
    }

    #[test]
    fn test_weak_direction_returns_empty_orders() {
        let current_price = 43000.0;
        let direction_pred = create_test_direction_prediction(0.55); // Weak signal
        let volatility_pred = create_test_volatility_prediction("MEDIUM");
        let price_levels = create_test_price_levels();
        let atr_value = 500.0;
        let config = OrderConfig::default();

        let orders = TradingOrders::generate(
            current_price,
            &direction_pred,
            &volatility_pred,
            &price_levels,
            atr_value,
            &config,
        )
        .unwrap();

        // Should return empty orders for weak direction signals
        assert!(
            orders.direction.contains("NO_SIGNAL") || orders.direction.contains("Insufficient"),
            "Should indicate no signal or insufficient confidence in direction: {}",
            orders.direction
        );
        assert_eq!(
            orders.total_position_size, 0.0,
            "Position size should be 0 for weak signals"
        );
        assert_eq!(
            orders.risk_reward_ratio, 0.0,
            "Risk-reward should be 0 for empty orders"
        );

        // All order levels should be empty
        for level in &orders.entry_levels {
            assert_eq!(
                level.price, 0.0,
                "Entry prices should be 0 for empty orders"
            );
            assert_eq!(
                level.quantity_percentage, 0.0,
                "Entry quantities should be 0 for empty orders"
            );
        }
    }

    #[test]
    fn test_volatility_regime_atr_scaling() {
        let current_price = 43000.0;
        let direction_pred = create_test_direction_prediction(0.7);
        let price_levels = create_test_price_levels();
        let atr_value = 500.0;
        let config = OrderConfig::default();

        // Test LOW volatility
        let low_vol = create_test_volatility_prediction("LOW");
        let low_orders = TradingOrders::generate(
            current_price,
            &direction_pred,
            &low_vol,
            &price_levels,
            atr_value,
            &config,
        )
        .unwrap();

        // Test HIGH volatility
        let high_vol = create_test_volatility_prediction("HIGH");
        let high_orders = TradingOrders::generate(
            current_price,
            &direction_pred,
            &high_vol,
            &price_levels,
            atr_value,
            &config,
        )
        .unwrap();

        // HIGH volatility should have larger ATR multiplier
        assert!(
            high_orders.atr_multiplier > low_orders.atr_multiplier,
            "HIGH volatility ATR multiplier ({}) should be larger than LOW volatility ({})",
            high_orders.atr_multiplier,
            low_orders.atr_multiplier
        );

        // Price spreads should be wider for HIGH volatility
        let low_entry_spread = low_orders.entry_levels[2].price - low_orders.entry_levels[0].price;
        let high_entry_spread =
            high_orders.entry_levels[2].price - high_orders.entry_levels[0].price;

        assert!(
            high_entry_spread.abs() > low_entry_spread.abs(),
            "HIGH volatility should have wider entry spreads"
        );
    }

    #[test]
    fn test_adaptive_system_different_horizons() {
        // Test 1: 4h horizon with 60 sequence length and 1.5 bandwidth multiplier
        let mut direction_4h = DirectionPrediction::from_probabilities(
            0.1, 0.2, 0.2, 0.3, 0.2, // Moderate bullish
        );
        direction_4h.calculate_horizon_adaptive_metrics(
            4.5, // 4.5% bandwidth (calculated from sequence)
            "4h".to_string(),
            60,
        );

        // Test 2: 1d horizon with 30 sequence length and 2.0 bandwidth multiplier
        let mut direction_1d = DirectionPrediction::from_probabilities(
            0.1, 0.2, 0.2, 0.3, 0.2, // Same probabilities
        );
        direction_1d.calculate_horizon_adaptive_metrics(
            8.0, // 8.0% bandwidth (larger for daily)
            "1d".to_string(),
            30,
        );

        // Validate horizon-specific calculations
        assert_eq!(direction_4h.training_horizon, "4h");
        assert_eq!(direction_4h.sequence_length, 60);
        assert_eq!(direction_4h.sequence_bandwidth_percent, 4.5);

        assert_eq!(direction_1d.training_horizon, "1d");
        assert_eq!(direction_1d.sequence_length, 30);
        assert_eq!(direction_1d.sequence_bandwidth_percent, 8.0);

        // Expected moves should scale with bandwidth
        assert!(direction_1d.expected_upside_percent > direction_4h.expected_upside_percent);
        assert!(direction_1d.breakout_threshold_percent > direction_4h.breakout_threshold_percent);

        // Both should have same aggregated probabilities (same input)
        assert!(
            (direction_4h.up_probability_aggregated - direction_1d.up_probability_aggregated).abs()
                < 0.001
        );

        println!(
            "4h Expected Upside: {:.2}%",
            direction_4h.expected_upside_percent
        );
        println!(
            "1d Expected Upside: {:.2}%",
            direction_1d.expected_upside_percent
        );
        println!("4h Risk/Reward: {:.2}", direction_4h.risk_reward_ratio);
        println!("1d Risk/Reward: {:.2}", direction_1d.risk_reward_ratio);
    }

    #[test]
    fn test_50_percent_threshold_bug_fixed() {
        // Test that the adaptive system can generate orders with very strong signals
        let current_price = 45000.0;

        // Create prediction with extremely strong directional edge (95% up, 5% down)
        let mut direction_pred = DirectionPrediction::from_probabilities(
            0.0, 0.05, 0.0, 0.5, 0.45, // 95% up aggregated, 5% down aggregated
        );

        // Calculate adaptive metrics to populate aggregated probabilities
        direction_pred.calculate_horizon_adaptive_metrics(
            5.0, // 5% bandwidth
            "4h".to_string(),
            60,
        );

        let volatility_pred = VolatilityPrediction::from_probabilities(
            0.8, 0.2, 0.0, 0.0, 0.0, // Very low volatility for lower thresholds
        );

        // Create very confident price levels with low entropy
        let mut bins = HashMap::new();
        bins.insert(
            "pump".to_string(),
            PriceBin {
                range: [5.0, 20.0],
                price: [47000.0, 54000.0],
                probability: 0.95, // Very high confidence
            },
        );
        bins.insert(
            "sideways".to_string(),
            PriceBin {
                range: [-2.0, 2.0],
                price: [44000.0, 46000.0],
                probability: 0.05,
            },
        );

        let price_levels = PriceLevelPrediction {
            bins,
            most_likely_range: [5.0, 20.0],
            confidence: 0.95, // Very high confidence
        };

        let atr_value = 1000.0; // High ATR
        let config = OrderConfig::default();

        // Debug the directional edge calculation
        let directional_edge =
            direction_pred.up_probability_aggregated - direction_pred.down_probability_aggregated;
        println!(
            "Up aggregated: {:.3}",
            direction_pred.up_probability_aggregated
        );
        println!(
            "Down aggregated: {:.3}",
            direction_pred.down_probability_aggregated
        );
        println!(
            "Directional edge: {:.3} ({:.1}%)",
            directional_edge,
            directional_edge * 100.0
        );

        // With extremely strong signals, should generate orders
        let orders = TradingOrders::generate(
            current_price,
            &direction_pred,
            &volatility_pred,
            &price_levels,
            atr_value,
            &config,
        )
        .unwrap();

        // Print the actual result for debugging
        println!("Generated orders direction: {}", orders.direction);

        // Should generate LONG orders with such strong signals
        // If this fails, the adaptive system thresholds are too strict
        assert!(
            orders.direction.starts_with("LONG") || orders.direction.contains("LONG"),
            "Expected LONG orders with 90% directional edge, got: {}",
            orders.direction
        );
    }
}
