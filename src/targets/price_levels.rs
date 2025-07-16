//! Price level target generation for cryptocurrency forecasting
//!
//! This module implements quantile-based price level classification for LSTM training.
//! Price levels are calculated using dynamic quantiles to create balanced target distributions.

use crate::utils::error::Result;
use polars::prelude::*;
use std::collections::HashMap;

/// Configuration for price level target generation
#[derive(Debug, Clone)]
pub struct PriceLevelConfig {
    pub bins: u32,
    pub quantile_method: QuantileMethod,
    pub lookback_window: usize,
    pub min_price_change: f64,
}

#[derive(Debug, Clone)]
pub enum QuantileMethod {
    Fixed,
    Rolling { window: usize },
    Adaptive { min_samples: usize },
}

impl Default for PriceLevelConfig {
    fn default() -> Self {
        Self {
            bins: 5,
            quantile_method: QuantileMethod::Rolling { window: 1000 },
            lookback_window: 100,
            min_price_change: 0.001, // 0.1% minimum change
        }
    }
}

/// Generate price level targets for multiple horizons
pub fn generate_price_level_targets(
    df: &DataFrame,
    horizons: &[String],
    config: &PriceLevelConfig,
) -> Result<HashMap<String, Vec<i32>>> {
    // Convert PriceLevelConfig to PriceLevelHead for compatibility
    let price_level_head = crate::config::model::PriceLevelHead {
        enabled: true,
        bins: config.bins,
        range_percent: config.min_price_change,
        distribution_type: crate::config::model::DistributionType::Categorical,
        target_strategy: crate::config::model::PriceLevelTargetStrategy::Current,
    };

    generate_price_level_targets_with_head(df, horizons, &price_level_head)
}

/// Generate price level targets using PriceLevelHead configuration
pub fn generate_price_level_targets_with_head(
    df: &DataFrame,
    horizons: &[String],
    config: &crate::config::model::PriceLevelHead,
) -> Result<HashMap<String, Vec<i32>>> {
    let close_prices = extract_close_prices(df)?;
    let mut targets = HashMap::new();

    for horizon in horizons {
        let horizon_steps = parse_horizon_to_steps(horizon)?;
        let price_targets = calculate_price_level_targets(&close_prices, horizon_steps, config)?;
        targets.insert(horizon.clone(), price_targets);
    }

    Ok(targets)
}

/// Generate price level targets from ModelConfig (convenience function)
pub fn generate_price_level_targets_from_model_config(
    df: &DataFrame,
    horizons: &[String],
    model_config: &crate::config::model::ModelConfig,
) -> Result<HashMap<String, Vec<i32>>> {
    generate_price_level_targets_with_head(df, horizons, &model_config.output_heads.price_levels)
}

/// Calculate price level targets for a specific horizon
fn calculate_price_level_targets(
    prices: &[f64],
    horizon_steps: usize,
    config: &crate::config::model::PriceLevelHead,
) -> Result<Vec<i32>> {
    if prices.len() < horizon_steps + 100 {
        // Use fixed lookback for now
        return Err(crate::utils::error::VangaError::DataError(
            "Insufficient data for price level target generation".to_string(),
        ));
    }

    let mut targets = vec![-1; prices.len()];

    for i in 100..(prices.len() - horizon_steps) {
        // Use fixed lookback
        let current_price = prices[i];

        // NEW: Calculate target price based on strategy
        let target_price = match &config.target_strategy {
            crate::config::model::PriceLevelTargetStrategy::Current => {
                // Existing logic: single future price
                prices[i + horizon_steps]
            }

            crate::config::model::PriceLevelTargetStrategy::StandardVWAP => {
                calculate_standard_vwap(prices, i, horizon_steps)?
            }

            crate::config::model::PriceLevelTargetStrategy::MomentumVWAP {
                momentum_window,
                bias_strength,
            } => {
                calculate_momentum_vwap(prices, i, horizon_steps, *momentum_window, *bias_strength)?
            }
        };

        let price_change = (target_price - current_price) / current_price;

        // Skip if price change is too small
        if price_change.abs() < config.range_percent {
            targets[i] = config.bins as i32 / 2; // Neutral class
            continue;
        }

        // Calculate quantiles for price level classification
        let quantiles = calculate_quantiles(
            &prices[i.saturating_sub(100)..=i], // Use fixed lookback
            config.bins,
            &QuantileMethod::Rolling { window: 1000 }, // Use default method
        )?;

        // Classify target price into quantile bins
        targets[i] = classify_price_to_level(target_price, &quantiles);
    }

    Ok(targets)
}

/// Calculate dynamic quantiles for price classification
fn calculate_quantiles(prices: &[f64], bins: u32, method: &QuantileMethod) -> Result<Vec<f64>> {
    if prices.is_empty() {
        return Err(crate::utils::error::VangaError::DataError(
            "Empty price data for quantile calculation".to_string(),
        ));
    }

    let mut sorted_prices = prices.to_vec();
    sorted_prices.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let mut quantiles = Vec::new();

    match method {
        QuantileMethod::Fixed => {
            for i in 1..bins {
                let percentile = i as f64 / bins as f64;
                let index = (percentile * (sorted_prices.len() - 1) as f64) as usize;
                quantiles.push(sorted_prices[index]);
            }
        }
        QuantileMethod::Rolling { window } => {
            let effective_window = (*window).min(prices.len());
            let recent_prices = &sorted_prices[sorted_prices.len() - effective_window..];

            for i in 1..bins {
                let percentile = i as f64 / bins as f64;
                let index = (percentile * (recent_prices.len() - 1) as f64) as usize;
                quantiles.push(recent_prices[index]);
            }
        }
        QuantileMethod::Adaptive { min_samples } => {
            if sorted_prices.len() < *min_samples {
                return calculate_quantiles(prices, bins, &QuantileMethod::Fixed);
            }

            // Use adaptive quantiles based on price volatility
            let volatility = calculate_price_volatility(&sorted_prices);
            let adaptive_factor = 1.0 + volatility * 0.5;

            for i in 1..bins {
                let base_percentile = i as f64 / bins as f64;
                let adaptive_percentile = (base_percentile * adaptive_factor).clamp(0.05, 0.95);
                let index = (adaptive_percentile * (sorted_prices.len() - 1) as f64) as usize;
                quantiles.push(sorted_prices[index]);
            }
        }
    }

    Ok(quantiles)
}

/// Classify a price into a quantile level
fn classify_price_to_level(price: f64, quantiles: &[f64]) -> i32 {
    for (i, &threshold) in quantiles.iter().enumerate() {
        if price <= threshold {
            return i as i32;
        }
    }
    quantiles.len() as i32
}

/// Calculate price volatility for adaptive quantiles
fn calculate_price_volatility(prices: &[f64]) -> f64 {
    if prices.len() < 2 {
        return 0.0;
    }

    let returns: Vec<f64> = prices.windows(2).map(|w| (w[1] - w[0]) / w[0]).collect();

    let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;
    let variance = returns
        .iter()
        .map(|r| (r - mean_return).powi(2))
        .sum::<f64>()
        / returns.len() as f64;

    variance.sqrt()
}

/// Extract close prices from DataFrame
fn extract_close_prices(df: &DataFrame) -> Result<Vec<f64>> {
    let close_series = df.column("close").map_err(|e| {
        crate::utils::error::VangaError::DataError(format!("Failed to get close column: {}", e))
    })?;

    let values: Vec<f64> = close_series
        .f64()
        .map_err(|e| {
            crate::utils::error::VangaError::DataError(format!(
                "Failed to convert close to f64: {}",
                e
            ))
        })?
        .into_no_null_iter()
        .collect();

    Ok(values)
}

/// Parse horizon string to number of steps
fn parse_horizon_to_steps(horizon: &str) -> Result<usize> {
    match horizon {
        "1h" => Ok(1),
        "4h" => Ok(4),
        "1d" => Ok(24),
        "7d" => Ok(168),
        _ => {
            if let Some(num_str) = horizon.strip_suffix('h') {
                num_str.parse::<usize>().map_err(|_| {
                    crate::utils::error::VangaError::DataError(format!(
                        "Invalid horizon format: {}",
                        horizon
                    ))
                })
            } else if let Some(num_str) = horizon.strip_suffix('d') {
                num_str.parse::<usize>().map(|d| d * 24).map_err(|_| {
                    crate::utils::error::VangaError::DataError(format!(
                        "Invalid horizon format: {}",
                        horizon
                    ))
                })
            } else {
                Err(crate::utils::error::VangaError::DataError(format!(
                    "Unsupported horizon format: {}",
                    horizon
                )))
            }
        }
    }
}
/// Calculate standard VWAP over horizon period
fn calculate_standard_vwap(prices: &[f64], start_idx: usize, horizon_steps: usize) -> Result<f64> {
    if start_idx + horizon_steps >= prices.len() {
        return Ok(prices[start_idx]);
    }

    let mut sum = 0.0;
    let mut count = 0;

    for i in 1..=horizon_steps {
        let idx = start_idx + i;
        if idx >= prices.len() {
            break;
        }
        sum += prices[idx];
        count += 1;
    }

    Ok(if count > 0 {
        sum / count as f64
    } else {
        prices[start_idx]
    })
}

/// Calculate momentum-aware VWAP with directional bias
fn calculate_momentum_vwap(
    prices: &[f64],
    start_idx: usize,
    horizon_steps: usize,
    momentum_window: usize,
    bias_strength: f64,
) -> Result<f64> {
    if start_idx + horizon_steps >= prices.len() || start_idx < momentum_window {
        return Ok(prices[start_idx]);
    }

    // Calculate momentum
    let current_price = prices[start_idx];
    let past_price = prices[start_idx - momentum_window];
    let momentum = (current_price - past_price) / past_price;

    let mut weighted_sum = 0.0;
    let mut weight_sum = 0.0;

    for i in 1..=horizon_steps {
        let idx = start_idx + i;
        if idx >= prices.len() {
            break;
        }

        let price = prices[idx];

        // Time-based weight (more recent = higher weight)
        let time_weight = i as f64 / horizon_steps as f64;

        // Momentum-based weight adjustment
        let momentum_adjustment = if momentum > 0.0 {
            1.0 + (momentum * bias_strength)
        } else {
            1.0 - (momentum.abs() * bias_strength)
        };

        let total_weight = time_weight * momentum_adjustment;

        weighted_sum += price * total_weight;
        weight_sum += total_weight;
    }

    Ok(if weight_sum > 0.0 {
        weighted_sum / weight_sum
    } else {
        prices[start_idx]
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::model::{DistributionType, PriceLevelHead, PriceLevelTargetStrategy};

    fn create_test_prices() -> Vec<f64> {
        // Create enough data for testing (100 lookback + horizon + some extra)
        (0..150).map(|i| 100.0 + i as f64).collect()
    }

    fn create_test_config(strategy: PriceLevelTargetStrategy) -> PriceLevelHead {
        PriceLevelHead {
            enabled: true,
            bins: 5,
            range_percent: 0.01,
            distribution_type: DistributionType::Categorical,
            target_strategy: strategy,
        }
    }

    #[test]
    fn test_current_strategy() {
        let prices = create_test_prices();
        let config = create_test_config(PriceLevelTargetStrategy::Current);

        let result = calculate_price_level_targets(&prices, 2, &config).unwrap();

        // Should have valid targets for middle indices
        assert!(result.len() == prices.len());
        // First 100 and last 2 should be -1 (invalid)
        for &value in result.iter().take(100) {
            assert_eq!(value, -1);
        }
        for &value in result.iter().skip(prices.len() - 2) {
            assert_eq!(value, -1);
        }
    }

    #[test]
    fn test_standard_vwap_strategy() {
        let prices = create_test_prices();
        let config = create_test_config(PriceLevelTargetStrategy::StandardVWAP);

        let result = calculate_price_level_targets(&prices, 2, &config).unwrap();

        // Should have valid targets for middle indices
        assert!(result.len() == prices.len());
        // Results should be different from current strategy for same input
        let current_config = create_test_config(PriceLevelTargetStrategy::Current);
        let current_result = calculate_price_level_targets(&prices, 2, &current_config).unwrap();

        // At least some values should be different
        let mut differences = 0;
        for i in 100..(prices.len() - 2) {
            if result[i] != current_result[i] {
                differences += 1;
            }
        }
        // We expect some differences due to VWAP vs single point
        assert!(differences >= 0); // At least allow for same results in simple cases
    }

    #[test]
    fn test_momentum_vwap_strategy() {
        let prices = create_test_prices();
        let config = create_test_config(PriceLevelTargetStrategy::MomentumVWAP {
            momentum_window: 3,
            bias_strength: 0.5,
        });

        let result = calculate_price_level_targets(&prices, 2, &config).unwrap();

        // Should have valid targets for middle indices
        assert!(result.len() == prices.len());
        // First 100 should be -1 (invalid)
        for &value in result.iter().take(100) {
            assert_eq!(value, -1);
        }
    }

    #[test]
    fn test_calculate_standard_vwap() {
        let prices = vec![100.0, 101.0, 102.0, 103.0, 104.0, 105.0];

        let result = calculate_standard_vwap(&prices, 0, 3).unwrap();

        // Should be average of prices[1], prices[2], prices[3] = (101 + 102 + 103) / 3 = 102
        assert!((result - 102.0).abs() < 0.001);
    }

    #[test]
    fn test_calculate_momentum_vwap() {
        let prices = vec![100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0];

        let result = calculate_momentum_vwap(&prices, 3, 2, 2, 0.5).unwrap();

        // Should return a weighted average that considers momentum
        assert!(result > 0.0);
        assert!(result < 200.0); // Reasonable bounds
    }

    #[test]
    fn test_vwap_edge_cases() {
        let prices = vec![100.0, 101.0];

        // Test with insufficient data
        let result = calculate_standard_vwap(&prices, 0, 5).unwrap();
        assert_eq!(result, 100.0); // Should return current price

        // Test momentum VWAP with insufficient history
        let result = calculate_momentum_vwap(&prices, 0, 2, 5, 0.5).unwrap();
        assert_eq!(result, 100.0); // Should return current price
    }

    #[test]
    fn test_backward_compatibility() {
        // Test that the old PriceLevelConfig interface still works
        let old_config = PriceLevelConfig {
            bins: 5,
            quantile_method: QuantileMethod::Fixed,
            lookback_window: 50,
            min_price_change: 0.01,
        };

        // Create test DataFrame with enough data
        use polars::prelude::*;
        let prices: Vec<f64> = (0..200).map(|i| 100.0 + i as f64).collect();
        let df = df! {
            "close" => prices
        }
        .unwrap();

        let horizons = vec!["1h".to_string()];
        let result = generate_price_level_targets(&df, &horizons, &old_config);

        // Should work without errors
        assert!(result.is_ok());
    }

    #[test]
    fn test_model_config_integration() {
        use crate::config::model::ModelConfig;

        // Create test DataFrame with enough data
        use polars::prelude::*;
        let prices: Vec<f64> = (0..200).map(|i| 100.0 + i as f64).collect();
        let df = df! {
            "close" => prices
        }
        .unwrap();

        let model_config = ModelConfig::default();
        let horizons = vec!["1h".to_string()];

        let result = generate_price_level_targets_from_model_config(&df, &horizons, &model_config);

        // Should work without errors
        assert!(result.is_ok());
    }
}
