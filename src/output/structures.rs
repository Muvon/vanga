//! Backward compatibility re-exports for prediction output structures
//!
//! This module maintains backward compatibility by re-exporting all types
//! from the new modular structure. New code should import directly from
//! the specific modules (prediction_types, trading_orders, etc.).

// Re-export all types from the new modular structure
pub use super::metadata::{ConfidenceScore, DataQuality, PredictionMetadata};
pub use super::prediction_types::{
    DirectionPrediction, PredictionResult, PriceBin, PriceLevelPrediction, VolatilityPrediction,
};
pub use super::trading_orders::{OrderConfig, OrderLevel, SequenceAwareOrderConfig, TradingOrders};

// Include all tests from the original file to ensure they still work
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
            "strong_down".to_string(),
            PriceBin {
                range: [-15.0, -8.0],
                vwap_range: [-15.0, -8.0],
                price: [36000.0, 39000.0],
                probability: 0.15,
            },
        );

        bins.insert(
            "moderate_down".to_string(),
            PriceBin {
                range: [-8.0, -3.0],
                vwap_range: [-8.0, -3.0],
                price: [39000.0, 41000.0],
                probability: 0.25,
            },
        );

        bins.insert(
            "neutral".to_string(),
            PriceBin {
                range: [-3.0, 3.0],
                vwap_range: [-3.0, 3.0],
                price: [41000.0, 44000.0],
                probability: 0.20,
            },
        );

        bins.insert(
            "moderate_up".to_string(),
            PriceBin {
                range: [3.0, 8.0],
                vwap_range: [3.0, 8.0],
                price: [44000.0, 46000.0],
                probability: 0.25,
            },
        );

        bins.insert(
            "strong_up".to_string(),
            PriceBin {
                range: [8.0, 15.0],
                vwap_range: [8.0, 15.0],
                price: [46000.0, 49000.0],
                probability: 0.15,
            },
        );

        PriceLevelPrediction {
            bins,
            most_likely_range: [3.0, 15.0],
            confidence: 0.9,
        }
    }

    fn create_test_crypto_aggressive_price_levels() -> PriceLevelPrediction {
        let mut bins = HashMap::new();

        bins.insert(
            "strong_down".to_string(),
            PriceBin {
                range: [-20.0, -10.0],
                vwap_range: [-20.0, -10.0],
                price: [34000.0, 38000.0],
                probability: 0.02,
            },
        );

        bins.insert(
            "moderate_down".to_string(),
            PriceBin {
                range: [-10.0, -5.0],
                vwap_range: [-10.0, -5.0],
                price: [38000.0, 40000.0],
                probability: 0.03,
            },
        );

        bins.insert(
            "neutral".to_string(),
            PriceBin {
                range: [-5.0, 5.0],
                vwap_range: [-5.0, 5.0],
                price: [40000.0, 45000.0],
                probability: 0.05,
            },
        );

        bins.insert(
            "moderate_up".to_string(),
            PriceBin {
                range: [5.0, 10.0],
                vwap_range: [5.0, 10.0],
                price: [45000.0, 47000.0],
                probability: 0.15,
            },
        );

        bins.insert(
            "strong_up".to_string(),
            PriceBin {
                range: [10.0, 30.0], // 10-30% pump expected
                vwap_range: [10.0, 30.0],
                price: [47000.0, 56000.0],
                probability: 0.75, // Very high confidence
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

        // Create sequence prices for testing (simulate recent price history)
        let sequence_prices = vec![current_price * 0.98, current_price * 0.99, current_price];
        let bandwidth_size = 0.02; // 2% bandwidth for testing

        let order_config = SequenceAwareOrderConfig {
            current_price,
            direction_pred: &direction_pred,
            volatility_pred: &volatility_pred,
            price_levels: &price_levels,
            atr_value,
            config: &config,
            sequence_prices: &sequence_prices,
            bandwidth_size,
            dynamic_entry_sizes: None,
            dynamic_exit_sizes: None,
            overall_confidence: None,
        };

        let orders = TradingOrders::generate(order_config).unwrap();

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

        // Create sequence prices for testing (simulate recent price history)
        let sequence_prices = vec![current_price * 1.01, current_price * 1.005, current_price];
        let bandwidth_size = 0.02; // 2% bandwidth for testing

        let order_config = SequenceAwareOrderConfig {
            current_price,
            direction_pred: &direction_pred,
            volatility_pred: &volatility_pred,
            price_levels: &price_levels,
            atr_value,
            config: &config,
            sequence_prices: &sequence_prices,
            bandwidth_size,
            dynamic_entry_sizes: None,
            dynamic_exit_sizes: None,
            overall_confidence: None,
        };

        let orders = TradingOrders::generate(order_config).unwrap();

        // Should be SHORT direction
        assert_eq!(orders.direction, "SHORT");

        // If there's a note, it means orders are empty due to insufficient confidence
        if orders.note.is_some() {
            // Just verify it's an empty order set
            assert_eq!(orders.entry_levels[0].price, 0.0);
            return; // Skip further assertions for empty orders
        }

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

        // Create sequence prices for testing (simulate recent price history)
        let sequence_prices = vec![current_price * 0.995, current_price * 0.998, current_price];
        let bandwidth_size = 0.01; // 1% bandwidth for low volatility

        let order_config_seq = SequenceAwareOrderConfig {
            current_price,
            direction_pred: &direction_pred,
            volatility_pred: &volatility_pred,
            price_levels: &price_levels,
            atr_value,
            config: &config,
            sequence_prices: &sequence_prices,
            bandwidth_size,
            dynamic_entry_sizes: None,
            dynamic_exit_sizes: None,
            overall_confidence: None,
        };

        let orders = TradingOrders::generate(order_config_seq).unwrap();

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

        // Create sequence prices for testing (simulate recent price history)
        let sequence_prices = vec![current_price * 0.99, current_price * 0.995, current_price];
        let bandwidth_size = 0.015; // 1.5% bandwidth for medium volatility

        let order_config = SequenceAwareOrderConfig {
            current_price,
            direction_pred: &direction_pred,
            volatility_pred: &volatility_pred,
            price_levels: &price_levels,
            atr_value,
            config: &config,
            sequence_prices: &sequence_prices,
            bandwidth_size,
            dynamic_entry_sizes: None,
            dynamic_exit_sizes: None,
            overall_confidence: None,
        };

        let orders = TradingOrders::generate(order_config).unwrap();

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
}
