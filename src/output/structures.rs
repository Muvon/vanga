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

/// Direction prediction matching ARCHITECTURE.md format
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DirectionPrediction {
    /// Probability of upward movement
    pub up_probability: f64,

    /// Probability of downward movement
    pub down_probability: f64,

    /// Predicted direction ("UP", "DOWN", "SIDEWAYS")
    pub prediction: String,

    /// Confidence in direction prediction
    pub confidence: f64,
}

/// Multi-horizon volatility prediction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VolatilityPrediction {
    /// Expected 1-hour volatility
    pub expected_1h: f64,

    /// Expected 4-hour volatility
    pub expected_4h: f64,

    /// Expected 24-hour volatility
    pub expected_24h: f64,

    /// Volatility regime prediction ("LOW", "MEDIUM", "HIGH")
    pub regime: String,

    /// Confidence in volatility prediction
    pub confidence: f64,
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
        // Check if direction prediction is strong enough for trading
        // With 5-class system, combined probabilities (up+pump, down+dump) are typically 0.3-0.8
        // Lowered threshold from 0.6 to 0.5 to account for more distributed probabilities
        let direction = if direction_pred.up_probability > 0.5 {
            "LONG"
        } else if direction_pred.down_probability > 0.5 {
            "SHORT"
        } else {
            // Return empty orders when direction is not strong enough
            return Ok(Self::empty(
                direction_pred,
                "Direction prediction not strong enough for trading orders",
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

        // Generate order levels based on direction
        let (entry_levels, exit_levels, stop_levels) = if direction == "LONG" {
            Self::generate_long_orders(current_price, atr_distance, price_levels, config)
        } else {
            Self::generate_short_orders(current_price, atr_distance, price_levels, config)
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
            "LONG_WEAK"
        } else {
            "SHORT_WEAK"
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

    /// Generate LONG position orders with crypto-aggressive spacing
    fn generate_long_orders(
        current_price: f64,
        atr_distance: f64,
        price_levels: &PriceLevelPrediction,
        config: &OrderConfig,
    ) -> ([OrderLevel; 3], [OrderLevel; 3], [OrderLevel; 3]) {
        // Entry levels (buy lower) - AGGRESSIVE SPACING
        let entry_prices = [
            current_price - atr_distance * 0.5, // Aggressive entry
            current_price - atr_distance * 1.2, // Medium entry
            current_price - atr_distance * 2.0, // Deep value entry
        ];

        // Exit levels (sell higher) - CRYPTO MOON TARGETS
        let exit_prices = [
            current_price + atr_distance * 2.0, // Quick profit
            current_price + atr_distance * 4.0, // Medium target
            current_price + atr_distance * 7.0, // Moon shot
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

    /// Generate SHORT position orders with crypto-aggressive spacing
    fn generate_short_orders(
        current_price: f64,
        atr_distance: f64,
        price_levels: &PriceLevelPrediction,
        config: &OrderConfig,
    ) -> ([OrderLevel; 3], [OrderLevel; 3], [OrderLevel; 3]) {
        // Entry levels (sell higher)
        let entry_prices = [
            current_price + atr_distance * 0.5, // Aggressive short entry
            current_price + atr_distance * 1.2, // Medium short entry
            current_price + atr_distance * 2.0, // High short entry
        ];

        // Exit levels (buy lower) - CRYPTO DUMP TARGETS
        let exit_prices = [
            current_price - atr_distance * 3.0, // Quick cover (increased from 2.0)
            current_price - atr_distance * 6.0, // Medium target (increased from 4.0)
            current_price - atr_distance * 10.0, // Full dump (increased from 7.0)
        ];

        // Stop levels (buy higher) - HUNT PROTECTED
        let stop_prices = [
            current_price + atr_distance * 2.5 * config.hunt_protection, // Reduced from 3.0
            current_price + atr_distance * 3.0 * config.hunt_protection, // Reduced from 4.0
            current_price + atr_distance * 3.5 * config.hunt_protection, // Reduced from 5.5
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
        DirectionPrediction {
            up_probability: up_prob,
            down_probability: 1.0 - up_prob,
            prediction: if up_prob > 0.5 {
                "UP".to_string()
            } else {
                "DOWN".to_string()
            },
            confidence: up_prob.max(1.0 - up_prob),
        }
    }

    fn create_test_volatility_prediction(regime: &str) -> VolatilityPrediction {
        VolatilityPrediction {
            expected_1h: 2.5,
            expected_4h: 5.0,
            expected_24h: 10.0,
            regime: regime.to_string(),
            confidence: 0.8,
        }
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
            orders.direction.contains("WEAK"),
            "Should indicate weak signal in direction: {}",
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
}
