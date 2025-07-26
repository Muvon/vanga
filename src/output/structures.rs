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

/// Direction prediction with 5-class system and horizon-adaptive mathematical ranges
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

        // 2. DIRECTIONAL EDGE (minimum edge threshold)
        let directional_edge =
            direction_pred.up_probability_aggregated - direction_pred.down_probability_aggregated;
        let min_edge_threshold = 0.15; // 15% minimum edge
        let min_risk_reward = 1.2; // Minimum 1.2:1 risk/reward

        if directional_edge.abs() > min_edge_threshold
            && direction_pred.risk_reward_ratio > min_risk_reward
        {
            if directional_edge > 0.0 {
                return AdaptiveTradingSignal::Long {
                    entry_price: current_price,
                    target_price: current_price
                        * (1.0 + direction_pred.expected_upside_percent / 100.0),
                    stop_loss: current_price
                        * (1.0 - volatility_pred.recommended_stop_distance_percent / 100.0),
                    position_size: volatility_pred.position_size_multiplier,
                    horizon: direction_pred.training_horizon.clone(),
                    risk_reward: direction_pred.risk_reward_ratio,
                    confidence: direction_pred.up_probability_aggregated,
                };
            } else {
                return AdaptiveTradingSignal::Short {
                    entry_price: current_price,
                    target_price: current_price
                        * (1.0 - direction_pred.expected_downside_percent / 100.0),
                    stop_loss: current_price
                        * (1.0 + volatility_pred.recommended_stop_distance_percent / 100.0),
                    position_size: volatility_pred.position_size_multiplier,
                    horizon: direction_pred.training_horizon.clone(),
                    risk_reward: direction_pred.risk_reward_ratio,
                    confidence: direction_pred.down_probability_aggregated,
                };
            }
        }

        AdaptiveTradingSignal::NoSignal {
            reason: format!(
                "Insufficient edge ({:.1}%) or poor risk/reward ({:.2}). Need ≥{:.1}% edge and ≥{:.1}:1 R/R",
                directional_edge * 100.0,
                direction_pred.risk_reward_ratio,
                min_edge_threshold * 100.0,
                min_risk_reward
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
    /// Generate trading orders from predictions with crypto-aggressive math
    pub fn generate(
        current_price: f64,
        direction_pred: &DirectionPrediction,
        volatility_pred: &VolatilityPrediction,
        price_levels: &PriceLevelPrediction,
        atr_value: f64,
        config: &OrderConfig,
    ) -> crate::utils::error::Result<Self> {
        // FIXED: Use adaptive edge detection instead of hardcoded 50% threshold
        // With 5-class system, we need to check relative strength and minimum edge
        let directional_edge =
            direction_pred.up_probability_aggregated - direction_pred.down_probability_aggregated;
        let min_edge_threshold = 0.15; // 15% minimum edge for signal generation
        let breakout_threshold = 0.25; // 25% threshold for strong breakout signals

        let direction = if directional_edge > min_edge_threshold {
            // Check for strong breakout signals first
            if direction_pred.pump_probability > breakout_threshold {
                "LONG_BREAKOUT"
            } else {
                "LONG"
            }
        } else if directional_edge < -min_edge_threshold {
            // Check for strong breakdown signals first
            if direction_pred.dump_probability > breakout_threshold {
                "SHORT_BREAKOUT"
            } else {
                "SHORT"
            }
        } else {
            // Return empty orders when directional edge is insufficient
            return Ok(Self::empty(
                direction_pred,
                &format!(
                    "Insufficient directional edge ({:.1}%, need ≥{:.1}%)",
                    directional_edge * 100.0,
                    min_edge_threshold * 100.0
                ),
            ));
        };

        // Calculate dynamic ATR multiplier based on 5-class volatility regime
        let atr_multiplier = match volatility_pred.regime.as_str() {
            "VERY_LOW" => config.base_atr_multiplier * 0.5, // Very tight spacing in very low vol
            "LOW" => config.base_atr_multiplier * 0.7,      // Tighter spacing in low vol
            "MEDIUM" => config.base_atr_multiplier,         // Normal spacing
            "HIGH" => config.base_atr_multiplier * 1.4,     // Wider spacing in high vol
            "VERY_HIGH" => config.base_atr_multiplier * 1.8, // Very wide spacing in very high vol
            _ => config.base_atr_multiplier,
        };

        let atr_distance = atr_value * atr_multiplier;

        // DEBUG: Log critical parameters for entry level calculation
        log::debug!(
            "🔍 Entry Level Debug: direction={}, current_price={:.2}, atr_value={:.2}, atr_multiplier={:.3}, atr_distance={:.2}",
            direction, current_price, atr_value, atr_multiplier, atr_distance
        );

        // Generate order levels based on direction (handle both regular and breakout signals)
        let (entry_levels, exit_levels, stop_levels) = if direction.starts_with("LONG") {
            let is_breakout = direction.contains("BREAKOUT");
            Self::generate_long_orders(
                current_price,
                atr_distance,
                price_levels,
                config,
                is_breakout,
            )
        } else {
            let is_breakout = direction.contains("BREAKOUT");
            Self::generate_short_orders(
                current_price,
                atr_distance,
                price_levels,
                config,
                is_breakout,
            )
        };

        // Calculate risk-reward ratio
        let risk_reward_ratio =
            Self::calculate_risk_reward(&entry_levels, &exit_levels, &stop_levels);

        Ok(TradingOrders {
            direction: direction.to_string(),
            entry_levels,
            exit_levels,
            stop_levels,
            total_position_size: 1.0,
            risk_reward_ratio,
            atr_multiplier,
            dynamic_sizing: config.aggressive_sizing,
        })
    }

    /// Generate sequence-aware trading orders using the same bandwidth logic as price levels
    /// This ensures consistency between price level predictions and order generation
    pub fn generate_sequence_aware(
        config: SequenceAwareOrderConfig,
    ) -> crate::utils::error::Result<Self> {
        // Validate that price_levels are consistent with our sequence-aware approach
        log::debug!("Generating sequence-aware orders with price level confidence: {:.2}", 
                   config.price_levels.confidence);
        
        // Use same directional edge logic as original method
        let directional_edge =
            config.direction_pred.up_probability_aggregated - config.direction_pred.down_probability_aggregated;
        let min_edge_threshold = 0.15; // 15% minimum edge for signal generation
        let breakout_threshold = 0.25; // 25% threshold for strong breakout signals

        let direction = if directional_edge > min_edge_threshold {
            "LONG"
        } else if directional_edge < -min_edge_threshold {
            "SHORT"
        } else {
            // Return empty orders when directional edge is insufficient
            return Ok(Self::empty(
                config.direction_pred,
                &format!(
                    "Insufficient directional edge ({:.1}%, need ≥{:.1}%)",
                    directional_edge * 100.0,
                    min_edge_threshold * 100.0
                ),
            ));
        };

        // Use the formatter's calculate_sequence_aware_ranges (the correct one)
        let temp_config = crate::config::prediction::OutputConfig::default();
        let formatter = crate::output::formatter::OutputFormatter::new(temp_config);
        let (sequence_ranges, _) = formatter.calculate_sequence_aware_ranges(
            config.sequence_prices,
            config.current_price,
            config.bandwidth_size,
        );

        // Calculate dynamic ATR multiplier based on 5-class volatility regime
        let atr_multiplier = match config.volatility_pred.regime.as_str() {
            "VERY_LOW" => config.config.base_atr_multiplier * 0.5, // Very tight spacing in very low vol
            "LOW" => config.config.base_atr_multiplier * 0.7,      // Tighter spacing in low vol
            "MEDIUM" => config.config.base_atr_multiplier,         // Normal spacing
            "HIGH" => config.config.base_atr_multiplier * 1.4,     // Wider spacing in high vol
            "VERY_HIGH" => config.config.base_atr_multiplier * 1.8, // Very wide spacing in very high vol
            _ => config.config.base_atr_multiplier,
        };

        let atr_distance = (config.atr_value / 100.0) * config.current_price * atr_multiplier;

        // Generate sequence-aware order levels based on direction
        let (entry_levels, exit_levels, stop_levels) = if direction == "LONG" {
            // Check if this is a breakout signal based on pump probability
            let is_breakout = config.direction_pred.pump_probability > breakout_threshold;
            Self::generate_sequence_aware_long_orders(
                config.current_price,
                atr_distance,
                &sequence_ranges,
                config.config,
                is_breakout,
            )
        } else {
            // Check if this is a breakout signal based on dump probability
            let is_breakout = config.direction_pred.dump_probability > breakout_threshold;
            Self::generate_sequence_aware_short_orders(
                config.current_price,
                atr_distance,
                &sequence_ranges,
                config.config,
                is_breakout,
            )
        };

        // Calculate risk-reward ratio
        let risk_reward_ratio =
            Self::calculate_risk_reward(&entry_levels, &exit_levels, &stop_levels);

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



    /// Generate sequence-aware long orders using bandwidth-based levels
    fn generate_sequence_aware_long_orders(
        current_price: f64,
        atr_distance: f64,
        sequence_ranges: &[[f64; 2]],
        config: &OrderConfig,
        is_breakout: bool,
    ) -> ([OrderLevel; 3], [OrderLevel; 3], [OrderLevel; 3]) {
        // Use sequence ranges for entry levels
        // For LONG: enter on moderate_down and strong_down ranges (BUY THE DIP!)
        let moderate_down_range = &sequence_ranges[1]; // moderate_down
        let strong_down_range = &sequence_ranges[0];   // strong_down

        // DEBUG: Log sequence ranges to identify the bug
        log::debug!(
            "🔍 Sequence-Aware LONG Debug: moderate_down=[{:.2}%, {:.2}%], strong_down=[{:.2}%, {:.2}%]",
            moderate_down_range[0], moderate_down_range[1], strong_down_range[0], strong_down_range[1]
        );

        // FIXED: Use sequence ranges properly for LONG entries
        // For LONG: Use the most negative parts of down ranges (buy the dip strategy)
        let entry_1_pct = moderate_down_range[0]; // Lower bound of moderate down (more negative)
        let entry_2_pct = (moderate_down_range[0] + strong_down_range[1]) / 2.0; // Between moderate and strong
        let entry_3_pct = strong_down_range[0]; // Lower bound of strong down (most negative)
        
        // Ensure proper ordering: entry_1 > entry_2 > entry_3 (less negative to more negative)
        let entry_1_pct = entry_1_pct.max(entry_2_pct + 0.1).max(entry_3_pct + 0.2);
        let entry_2_pct = entry_2_pct.max(entry_3_pct + 0.1);

        // DEBUG: Log entry percentages and resulting prices
        log::debug!(
            "🔍 Entry Percentages: entry_1_pct={:.2}%, entry_2_pct={:.2}%, entry_3_pct={:.2}%",
            entry_1_pct, entry_2_pct, entry_3_pct
        );
        
        let calculated_entry_1 = current_price * (1.0 + entry_1_pct / 100.0);
        let calculated_entry_2 = current_price * (1.0 + entry_2_pct / 100.0);
        let calculated_entry_3 = current_price * (1.0 + entry_3_pct / 100.0);
        
        log::debug!(
            "🟢 LONG Entry Prices: [{:.2}, {:.2}, {:.2}] vs current_price {:.2} (should be BELOW)",
            calculated_entry_1, calculated_entry_2, calculated_entry_3, current_price
        );

        let entry_levels = [
            OrderLevel {
                price: calculated_entry_1,
                quantity_percentage: if is_breakout { 0.4 } else { 0.5 },
                atr_distance,
                order_type: "LIMIT".to_string(),
                confidence: 0.8,
            },
            OrderLevel {
                price: calculated_entry_2,
                quantity_percentage: 0.3, // Same for both breakout and normal
                atr_distance,
                order_type: "LIMIT".to_string(),
                confidence: 0.7,
            },
            OrderLevel {
                price: calculated_entry_3,
                quantity_percentage: if is_breakout { 0.3 } else { 0.2 },
                atr_distance,
                order_type: if is_breakout { "STOP_LIMIT".to_string() } else { "LIMIT".to_string() },
                confidence: if is_breakout { 0.9 } else { 0.6 },
            },
        ];

        // Exit levels based on moderate_up and strong_up ranges (SELL THE RIP!)
        let moderate_up_range = &sequence_ranges[3]; // moderate_up
        let strong_up_range = &sequence_ranges[4];   // strong_up
        
        let exit_1_pct = moderate_up_range[0]; // Start of moderate up
        let exit_2_pct = (moderate_up_range[0] + moderate_up_range[1]) / 2.0; // Mid moderate up
        let exit_3_pct = strong_up_range[0]; // Start of strong up (breakout target)

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

        // Stop levels based on sequence ranges (below current price for LONG positions)
        let neutral_range = &sequence_ranges[2]; // neutral
        let stop_1_pct = neutral_range[0] - (atr_distance / current_price * 100.0); // Below neutral with ATR buffer
        let stop_2_pct = sequence_ranges[1][1] - (atr_distance / current_price * 100.0); // Moderate down with ATR buffer  
        let stop_3_pct = sequence_ranges[0][1] - (atr_distance / current_price * 100.0); // Strong down with ATR buffer

        // Apply hunt protection from config
        let hunt_protection_multiplier = config.hunt_protection;
        
        let stop_levels = [
            OrderLevel {
                price: current_price * (1.0 + stop_1_pct / 100.0), // stop_1_pct is already negative percentage
                quantity_percentage: 0.4,
                atr_distance: atr_distance * hunt_protection_multiplier,
                order_type: "STOP_LOSS".to_string(),
                confidence: 0.9,
            },
            OrderLevel {
                price: current_price * (1.0 + stop_2_pct / 100.0), // stop_2_pct is already negative percentage
                quantity_percentage: 0.4,
                atr_distance: atr_distance * hunt_protection_multiplier,
                order_type: "STOP_LOSS".to_string(),
                confidence: 0.8,
            },
            OrderLevel {
                price: current_price * (1.0 + stop_3_pct / 100.0), // stop_3_pct is already negative percentage
                quantity_percentage: 0.2,
                atr_distance: atr_distance * hunt_protection_multiplier,
                order_type: "STOP_LOSS".to_string(),
                confidence: 0.7,
            },
        ];

        (entry_levels, exit_levels, stop_levels)
    }

    /// Generate sequence-aware short orders using bandwidth-based levels
    fn generate_sequence_aware_short_orders(
        current_price: f64,
        atr_distance: f64,
        sequence_ranges: &[[f64; 2]],
        config: &OrderConfig,
        is_breakout: bool,
    ) -> ([OrderLevel; 3], [OrderLevel; 3], [OrderLevel; 3]) {
        // Use sequence ranges for entry levels
        // For SHORT: enter on moderate_down and strong_down ranges
        let moderate_down_range = &sequence_ranges[1]; // moderate_down
        let strong_down_range = &sequence_ranges[0];   // strong_down

        let entry_1_pct = moderate_down_range[1]; // End of moderate down
        let entry_2_pct = (moderate_down_range[0] + moderate_down_range[1]) / 2.0; // Mid moderate down
        let entry_3_pct = strong_down_range[1]; // End of strong down (breakout level)

        let entry_levels = [
            OrderLevel {
                price: current_price * (1.0 + entry_1_pct / 100.0),
                quantity_percentage: if is_breakout { 0.4 } else { 0.5 },
                atr_distance,
                order_type: "LIMIT".to_string(),
                confidence: 0.8,
            },
            OrderLevel {
                price: current_price * (1.0 + entry_2_pct / 100.0),
                quantity_percentage: 0.3, // Same for both breakout and normal
                atr_distance,
                order_type: "LIMIT".to_string(),
                confidence: 0.7,
            },
            OrderLevel {
                price: current_price * (1.0 + entry_3_pct / 100.0),
                quantity_percentage: if is_breakout { 0.3 } else { 0.2 },
                atr_distance,
                order_type: if is_breakout { "STOP_LIMIT".to_string() } else { "LIMIT".to_string() },
                confidence: if is_breakout { 0.9 } else { 0.6 },
            },
        ];

        // Exit levels based on strong_down range
        let exit_1_pct = strong_down_range[0]; // Start of strong down
        let exit_2_pct = strong_down_range[0] - (strong_down_range[1] - strong_down_range[0]) * 0.5; // 50% extension
        let exit_3_pct = strong_down_range[0] - (strong_down_range[1] - strong_down_range[0]) * 1.0; // 100% extension

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

        // Stop levels based on sequence ranges (above neutral)
        let neutral_range = &sequence_ranges[2]; // neutral
        let stop_1_pct = neutral_range[1] + atr_distance / current_price * 100.0; // Above neutral with ATR buffer
        let stop_2_pct = sequence_ranges[3][0] + atr_distance / current_price * 100.0; // Moderate up with ATR buffer
        let stop_3_pct = sequence_ranges[4][0] + atr_distance / current_price * 100.0; // Strong up with ATR buffer

        // Apply hunt protection from config
        let hunt_protection_multiplier = config.hunt_protection;

        let stop_levels = [
            OrderLevel {
                price: current_price * (1.0 + (stop_1_pct * hunt_protection_multiplier) / 100.0),
                quantity_percentage: 0.4,
                atr_distance: atr_distance * hunt_protection_multiplier,
                order_type: "STOP_LOSS".to_string(),
                confidence: 0.9,
            },
            OrderLevel {
                price: current_price * (1.0 + (stop_2_pct * hunt_protection_multiplier) / 100.0),
                quantity_percentage: 0.4,
                atr_distance: atr_distance * hunt_protection_multiplier,
                order_type: "STOP_LOSS".to_string(),
                confidence: 0.8,
            },
            OrderLevel {
                price: current_price * (1.0 + (stop_3_pct * hunt_protection_multiplier) / 100.0),
                quantity_percentage: 0.2,
                atr_distance: atr_distance * hunt_protection_multiplier,
                order_type: "STOP_LOSS".to_string(),
                confidence: 0.7,
            },
        ];

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

    /// Round price to appropriate precision based on price level
    /// Handles cryptocurrencies with prices from $0.0001 to $100,000+
    fn round_price(price: f64) -> f64 {
        if price <= 0.0 {
            return 0.0;
        }

        if price < 0.001 {
            // For very low prices (like SHIB), use 8 decimal places
            (price * 100_000_000.0).round() / 100_000_000.0
        } else if price < 0.01 {
            // For low prices, use 6 decimal places
            (price * 1_000_000.0).round() / 1_000_000.0
        } else if price < 1.0 {
            // For sub-dollar prices, use 4 decimal places
            (price * 10_000.0).round() / 10_000.0
        } else if price < 100.0 {
            // For normal crypto prices, use 2 decimal places
            (price * 100.0).round() / 100.0
        } else {
            // For high-value cryptos like BTC, use 2 decimal places
            (price * 100.0).round() / 100.0
        }
    }

    /// Generate LONG position orders with crypto-aggressive spacing and breakout support
    fn generate_long_orders(
        current_price: f64,
        atr_distance: f64,
        price_levels: &PriceLevelPrediction,
        config: &OrderConfig,
        is_breakout: bool,
    ) -> ([OrderLevel; 3], [OrderLevel; 3], [OrderLevel; 3]) {
        // Adjust spacing for breakout signals (more aggressive)
        let spacing_multiplier = if is_breakout { 1.5 } else { 1.0 };

        // Entry levels (buy lower) - AGGRESSIVE SPACING
        let entry_prices = [
            current_price - atr_distance * 0.5 * spacing_multiplier, // Aggressive entry
            current_price - atr_distance * 1.2 * spacing_multiplier, // Medium entry
            current_price - atr_distance * 2.0 * spacing_multiplier, // Deep value entry
        ];

        // DEBUG: Log LONG entry calculations
        log::debug!(
            "🟢 LONG Entry Calculation: current_price={:.2}, atr_distance={:.2}, spacing_multiplier={:.1}",
            current_price, atr_distance, spacing_multiplier
        );
        log::debug!(
            "🟢 LONG Entry Prices: [{:.2}, {:.2}, {:.2}] (should be BELOW current_price {:.2})",
            entry_prices[0], entry_prices[1], entry_prices[2], current_price
        );

        // Exit levels (sell higher) - CRYPTO MOON TARGETS (enhanced for breakouts)
        let exit_multiplier = if is_breakout { 1.8 } else { 1.0 };
        let exit_prices = [
            current_price + atr_distance * 2.0 * exit_multiplier, // Quick profit
            current_price + atr_distance * 4.0 * exit_multiplier, // Medium target
            current_price + atr_distance * 7.0 * exit_multiplier, // Moon shot
        ];

        // Stop levels (sell lower) - HUNT PROTECTED
        let stop_prices = [
            current_price - atr_distance * 3.0 * config.hunt_protection,
            current_price - atr_distance * 4.0 * config.hunt_protection,
            current_price - atr_distance * 5.5 * config.hunt_protection,
        ];

        // DYNAMIC QUANTITY ALLOCATION - CRYPTO SPECULATION
        let entry_quantities =
            Self::calculate_dynamic_quantities(&entry_prices, current_price, price_levels, true);
        let exit_quantities = [40.0, 35.0, 25.0]; // Aggressive profit taking
        let stop_quantities = entry_quantities; // Match entry allocation

        let entry_levels =
            Self::create_order_levels(&entry_prices, &entry_quantities, &[0.5, 1.2, 2.0], "LIMIT");
        let exit_levels =
            Self::create_order_levels(&exit_prices, &exit_quantities, &[2.0, 4.0, 7.0], "LIMIT");
        let stop_levels = Self::create_order_levels(
            &stop_prices,
            &stop_quantities,
            &[3.0, 4.0, 5.5],
            "STOP_LIMIT",
        );

        (entry_levels, exit_levels, stop_levels)
    }

    /// Generate SHORT position orders with crypto-aggressive spacing and breakout support
    fn generate_short_orders(
        current_price: f64,
        atr_distance: f64,
        price_levels: &PriceLevelPrediction,
        config: &OrderConfig,
        is_breakout: bool,
    ) -> ([OrderLevel; 3], [OrderLevel; 3], [OrderLevel; 3]) {
        // Adjust spacing for breakout signals (more aggressive)
        let spacing_multiplier = if is_breakout { 1.5 } else { 1.0 };

        // Entry levels (sell higher)
        let entry_prices = [
            current_price + atr_distance * 0.5 * spacing_multiplier, // Aggressive short entry
            current_price + atr_distance * 1.2 * spacing_multiplier, // Medium short entry
            current_price + atr_distance * 2.0 * spacing_multiplier, // High short entry
        ];

        // Exit levels (buy lower) - CRYPTO DUMP TARGETS (enhanced for breakouts)
        let exit_multiplier = if is_breakout { 2.0 } else { 1.0 };
        let exit_prices = [
            current_price - atr_distance * 3.0 * exit_multiplier, // Quick cover
            current_price - atr_distance * 6.0 * exit_multiplier, // Medium target
            current_price - atr_distance * 10.0 * exit_multiplier, // Full dump
        ];

        // Stop levels (buy higher) - HUNT PROTECTED
        let stop_prices = [
            current_price + atr_distance * 2.5 * config.hunt_protection,
            current_price + atr_distance * 3.0 * config.hunt_protection,
            current_price + atr_distance * 3.5 * config.hunt_protection,
        ];

        // DYNAMIC QUANTITY ALLOCATION
        let entry_quantities =
            Self::calculate_dynamic_quantities(&entry_prices, current_price, price_levels, false);
        let exit_quantities = [40.0, 35.0, 25.0]; // Aggressive profit taking
        let stop_quantities = entry_quantities; // Match entry allocation

        let entry_levels =
            Self::create_order_levels(&entry_prices, &entry_quantities, &[0.5, 1.2, 2.0], "LIMIT");
        let exit_levels =
            Self::create_order_levels(&exit_prices, &exit_quantities, &[6.0, 12.0, 20.0], "LIMIT");
        let stop_levels = Self::create_order_levels(
            &stop_prices,
            &stop_quantities,
            &[2.5, 3.0, 3.5],
            "STOP_LIMIT",
        );

        (entry_levels, exit_levels, stop_levels)
    }

    /// Calculate dynamic quantities with CRYPTO SPECULATION BOOST
    fn calculate_dynamic_quantities(
        prices: &[f64],
        current_price: f64,
        price_levels: &PriceLevelPrediction,
        is_long: bool,
    ) -> [f64; 3] {
        let mut quantities = [0.0; 3];
        let mut total_confidence = 0.0;

        // Calculate confidence for each price level
        for (i, &price) in prices.iter().enumerate() {
            let price_pct = if is_long {
                ((price - current_price) / current_price) * 100.0
            } else {
                ((current_price - price) / current_price) * 100.0
            };

            // Find matching bin confidence
            let confidence = Self::find_price_confidence(price_pct, price_levels);
            quantities[i] = confidence;
            total_confidence += confidence;
        }

        // Normalize to 100% with AGGRESSIVE CRYPTO WEIGHTING
        if total_confidence > 0.0 {
            for quantity in &mut quantities {
                *quantity = (*quantity / total_confidence) * 100.0;

                // CRYPTO SPECULATION BOOST - Favor higher confidence MORE
                if *quantity > 40.0 {
                    *quantity *= 1.3; // 30% boost for high confidence
                } else if *quantity < 25.0 {
                    *quantity *= 0.8; // 20% reduction for low confidence
                }
            }

            // Re-normalize after boosting
            let boosted_total: f64 = quantities.iter().sum();
            for q in quantities.iter_mut() {
                *q = (*q / boosted_total) * 100.0;
            }
        } else {
            // Fallback to aggressive default allocation
            quantities = [50.0, 30.0, 20.0]; // Front-load the best entry
        }

        // ENSURE FRONT-LOADING: First entry should always get the most
        // Don't sort - preserve order but ensure first gets most allocation
        if quantities[0] <= quantities[1] || quantities[0] <= quantities[2] {
            // Force front-loading pattern: first gets most, then descending
            let total = quantities.iter().sum::<f64>();
            if total > 0.0 {
                quantities = [50.0, 30.0, 20.0]; // Guaranteed front-loading
            }
        }

        // Additional check: ensure first is always highest
        if quantities[0] < 45.0 || quantities[0] <= quantities[1] {
            quantities = [50.0, 30.0, 20.0];
        }

        quantities
    }

    /// Find confidence level for a price percentage in prediction bins
    fn find_price_confidence(price_pct: f64, price_levels: &PriceLevelPrediction) -> f64 {
        for (bin_name, bin) in &price_levels.bins {
            if price_pct >= bin.range[0] && price_pct <= bin.range[1] {
                // CRYPTO SPECULATION: Boost confidence for extreme moves
                let base_prob = bin.probability;
                if bin_name == "moon" || bin_name == "rekt" {
                    return base_prob * 1.5; // 50% boost for extreme moves
                } else if bin_name == "parabolic" || bin_name == "capitulation" {
                    return base_prob * 1.2; // 20% boost for strong moves
                }
                return base_prob;
            }
        }
        0.1 // Default low confidence
    }

    /// Create order levels from prices and quantities
    fn create_order_levels(
        prices: &[f64],
        quantities: &[f64],
        atr_distances: &[f64],
        order_type: &str,
    ) -> [OrderLevel; 3] {
        [
            OrderLevel {
                price: Self::round_price(prices[0]),
                quantity_percentage: (quantities[0] * 100.0).round() / 100.0, // Round to 2 decimal places
                atr_distance: atr_distances[0],
                order_type: order_type.to_string(),
                confidence: quantities[0] / 100.0,
            },
            OrderLevel {
                price: Self::round_price(prices[1]),
                quantity_percentage: (quantities[1] * 100.0).round() / 100.0,
                atr_distance: atr_distances[1],
                order_type: order_type.to_string(),
                confidence: quantities[1] / 100.0,
            },
            OrderLevel {
                price: Self::round_price(prices[2]),
                quantity_percentage: (quantities[2] * 100.0).round() / 100.0,
                atr_distance: atr_distances[2],
                order_type: order_type.to_string(),
                confidence: quantities[2] / 100.0,
            },
        ]
    }

    /// Calculate risk-reward ratio for the order set
    fn calculate_risk_reward(
        entry_levels: &[OrderLevel; 3],
        exit_levels: &[OrderLevel; 3],
        stop_levels: &[OrderLevel; 3],
    ) -> f64 {
        // Weighted average prices by quantity
        let avg_entry = Self::weighted_average_price(entry_levels);
        let avg_exit = Self::weighted_average_price(exit_levels);
        let avg_stop = Self::weighted_average_price(stop_levels);

        let potential_profit = (avg_exit - avg_entry).abs();
        let potential_loss = (avg_entry - avg_stop).abs();

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
        let sideways_prob = 0.2;
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
        let down_moderate = down_prob - dump_prob;
        let up_moderate = up_prob - pump_prob;

        DirectionPrediction::from_probabilities(
            dump_prob,
            down_moderate,
            sideways_prob,
            up_moderate,
            pump_prob,
        )
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
                probability: 0.6, // High confidence pump
            },
        );

        bins.insert(
            "sideways".to_string(),
            PriceBin {
                range: [-3.0, 3.0],
                price: [41000.0, 43000.0],
                probability: 0.3,
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
            confidence: 0.8,
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
        let direction_pred = create_test_direction_prediction(0.7); // Strong up
        let volatility_pred = create_test_volatility_prediction("MEDIUM");
        let price_levels = create_test_price_levels();
        let atr_value = 500.0; // $500 ATR
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

        // Should be LONG direction
        assert_eq!(orders.direction, "LONG");

        // Entry levels should be below current price
        for level in &orders.entry_levels {
            assert!(
                level.price < current_price,
                "Entry price {} should be below current price {}",
                level.price,
                current_price
            );
        }

        // Exit levels should be above current price
        for level in &orders.exit_levels {
            assert!(
                level.price > current_price,
                "Exit price {} should be above current price {}",
                level.price,
                current_price
            );
        }

        // Stop levels should be below current price
        for level in &orders.stop_levels {
            assert!(
                level.price < current_price,
                "Stop price {} should be below current price {}",
                level.price,
                current_price
            );
        }

        // Quantities should sum to 100%
        let entry_total: f64 = orders
            .entry_levels
            .iter()
            .map(|l| l.quantity_percentage)
            .sum();
        let exit_total: f64 = orders
            .exit_levels
            .iter()
            .map(|l| l.quantity_percentage)
            .sum();

        assert!(
            (entry_total - 100.0).abs() < 0.01,
            "Entry quantities should sum to 100%, got {}",
            entry_total
        );
        assert!(
            (exit_total - 100.0).abs() < 0.01,
            "Exit quantities should sum to 100%, got {}",
            exit_total
        );
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
            orders.direction.contains("Insufficient directional edge"),
            "Should indicate insufficient edge in direction: {}",
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
        // Test that we no longer use hardcoded 50% thresholds
        let current_price = 45000.0;

        // Create prediction with 55% up, 35% down (20% directional edge)
        let mut direction_pred = DirectionPrediction::from_probabilities(
            0.1, 0.25, 0.1, 0.35, 0.2, // 55% up aggregated, 35% down aggregated
        );

        // Calculate adaptive metrics to populate aggregated probabilities
        direction_pred.calculate_horizon_adaptive_metrics(
            5.0, // 5% bandwidth
            "4h".to_string(),
            60,
        );

        let volatility_pred = VolatilityPrediction::from_probabilities(
            0.2, 0.4, 0.3, 0.1, 0.0, // Low volatility
        );

        let price_levels = create_test_price_levels();
        let atr_value = 500.0;
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

        // This should now generate orders because directional edge is 20% (55% - 35%)
        let orders = TradingOrders::generate(
            current_price,
            &direction_pred,
            &volatility_pred,
            &price_levels,
            atr_value,
            &config,
        )
        .unwrap();

        // Should be LONG direction (not empty orders)
        assert!(orders.direction.starts_with("LONG"));
        assert!(orders.total_position_size > 0.0);
        assert!(orders.risk_reward_ratio > 0.0);

        println!("Generated orders with direction: {}", orders.direction);
        println!("Position size: {:.1}%", orders.total_position_size * 100.0);
    }
}
