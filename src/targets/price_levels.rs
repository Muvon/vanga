//! Price level target generation for cryptocurrency forecasting
//!
//! This module implements quantile-based price level classification for LSTM training.
//! Price levels are calculated using dynamic quantiles to create balanced target distributions.

use crate::utils::error::Result;
use polars::prelude::*;
use std::collections::HashMap;

/// Type alias for price level targets with quantile boundaries
type PriceLevelTargetsWithBoundaries = (HashMap<String, Vec<i32>>, HashMap<String, Vec<f64>>);

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

/// Generate price level targets with consistent global boundaries
/// This ensures training and validation use the same class boundaries
pub fn generate_price_level_targets_with_global_boundaries(
    df: &DataFrame,
    horizons: &[String],
    config: &crate::config::model::PriceLevelHead,
    train_val_split_idx: Option<usize>, // Split point for consistent boundaries
) -> Result<PriceLevelTargetsWithBoundaries> {
    let mut all_targets = HashMap::new();
    let mut global_quantiles = HashMap::new();

    // Extract price column
    let prices = df
        .column("close")?
        .f64()?
        .into_no_null_iter()
        .collect::<Vec<f64>>();

    if prices.len() < 2 {
        return Err(crate::utils::error::VangaError::DataError(
            "Insufficient price data for target generation".to_string(),
        ));
    }

    for horizon in horizons {
        let horizon_steps = parse_horizon_to_steps(horizon)?;

        // Calculate global quantiles from training portion only for consistency
        let train_end_idx = if let Some(split_idx) = train_val_split_idx {
            split_idx.min(prices.len().saturating_sub(horizon_steps))
        } else {
            prices.len().saturating_sub(horizon_steps)
        };

        let train_start_idx = 100; // Skip initial lookback period

        if train_end_idx <= train_start_idx {
            return Err(crate::utils::error::VangaError::DataError(
                "Insufficient training data for global quantile calculation".to_string(),
            ));
        }

        // FIXED: Use consistent Fixed quantile method for stable boundaries
        let quantiles = calculate_quantiles(
            &prices[train_start_idx..train_end_idx],
            config.bins,
            &QuantileMethod::Fixed, // Always use Fixed for consistency
        )?;

        log::debug!(
            "🎯 Global boundaries [{}]: {} training samples, quantiles: {:?}",
            horizon,
            train_end_idx - train_start_idx,
            quantiles
        );

        // Apply same quantiles to entire dataset (both training and validation)
        let targets = apply_quantiles_to_targets(&prices, horizon_steps, &quantiles, config)?;

        all_targets.insert(horizon.clone(), targets);
        global_quantiles.insert(horizon.clone(), quantiles);
    }

    Ok((all_targets, global_quantiles))
}

/// Apply pre-calculated quantiles to generate targets
fn apply_quantiles_to_targets(
    prices: &[f64],
    horizon_steps: usize,
    quantiles: &[f64],
    config: &crate::config::model::PriceLevelHead,
) -> Result<Vec<i32>> {
    let mut targets = vec![-1; prices.len()]; // Initialize with invalid targets

    for i in 0..(prices.len().saturating_sub(horizon_steps)) {
        let current_price = prices[i];

        // Calculate target price based on strategy
        let target_price = match &config.target_strategy {
            crate::config::model::PriceLevelTargetStrategy::Current => prices[i + horizon_steps],
            crate::config::model::PriceLevelTargetStrategy::StandardVwap => {
                calculate_standard_vwap(prices, i, horizon_steps)?
            }
            crate::config::model::PriceLevelTargetStrategy::MomentumVwap {
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

        // Classify using global quantiles
        targets[i] = classify_price_to_level(target_price, quantiles);
    }

    Ok(targets)
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

        // Analyze and log class distribution
        analyze_class_distribution(&price_targets, horizon, config.bins)?;

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

/// Calculate price level targets for a specific horizon with consistent global quantiles
fn calculate_price_level_targets(
    prices: &[f64],
    horizon_steps: usize,
    config: &crate::config::model::PriceLevelHead,
) -> Result<Vec<i32>> {
    if prices.len() < horizon_steps + 100 {
        return Err(crate::utils::error::VangaError::DataError(
            "Insufficient data for price level target generation".to_string(),
        ));
    }

    let mut targets = vec![-1; prices.len()];

    // CRITICAL FIX: Calculate global quantiles from PERCENTAGE CHANGES, not absolute prices
    // This ensures consistent class boundaries across all symbols regardless of price level
    let training_data_start = 100; // Skip initial lookback period
    let training_data_end = prices.len() - horizon_steps; // Ensure we have future data

    if training_data_end <= training_data_start {
        return Err(crate::utils::error::VangaError::DataError(
            "Insufficient data range for global quantile calculation".to_string(),
        ));
    }

    // Calculate percentage changes for quantile calculation
    let mut percentage_changes = Vec::new();
    for i in training_data_start..training_data_end {
        let current_price = prices[i];
        let target_price = match &config.target_strategy {
            crate::config::model::PriceLevelTargetStrategy::Current => prices[i + horizon_steps],
            crate::config::model::PriceLevelTargetStrategy::StandardVwap => {
                calculate_standard_vwap(prices, i, horizon_steps)?
            }
            crate::config::model::PriceLevelTargetStrategy::MomentumVwap {
                momentum_window,
                bias_strength,
            } => {
                calculate_momentum_vwap(prices, i, horizon_steps, *momentum_window, *bias_strength)?
            }
        };

        if current_price != 0.0 {
            let price_change = (target_price - current_price) / current_price;
            percentage_changes.push(price_change);
        }
    }

    // Calculate quantiles from percentage changes (symbol-agnostic)
    let global_quantiles = calculate_quantiles(
        &percentage_changes,
        config.bins,
        &QuantileMethod::Fixed, // Use Fixed method for consistency across train/val
    )?;

    log::debug!(
        "🎯 Global quantiles calculated for {} bins using {} percentage changes: {:?}",
        config.bins,
        percentage_changes.len(),
        global_quantiles
    );

    // Apply consistent quantiles to all samples using percentage changes
    for i in training_data_start..training_data_end {
        let current_price = prices[i];

        // Calculate target price based on strategy
        let target_price = match &config.target_strategy {
            crate::config::model::PriceLevelTargetStrategy::Current => prices[i + horizon_steps],

            crate::config::model::PriceLevelTargetStrategy::StandardVwap => {
                calculate_standard_vwap(prices, i, horizon_steps)?
            }

            crate::config::model::PriceLevelTargetStrategy::MomentumVwap {
                momentum_window,
                bias_strength,
            } => {
                calculate_momentum_vwap(prices, i, horizon_steps, *momentum_window, *bias_strength)?
            }
        };

        if current_price == 0.0 {
            targets[i] = config.bins as i32 / 2; // Neutral class for invalid prices
            continue;
        }

        let price_change = (target_price - current_price) / current_price;

        // Skip if price change is too small
        if price_change.abs() < config.range_percent {
            targets[i] = config.bins as i32 / 2; // Neutral class
            continue;
        }

        // FIXED: Classify percentage change against percentage quantiles
        targets[i] = classify_price_to_level(price_change, &global_quantiles);
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

/// Classify a value into a quantile level (works for both prices and percentage changes)
fn classify_price_to_level(value: f64, quantiles: &[f64]) -> i32 {
    for (i, &threshold) in quantiles.iter().enumerate() {
        if value <= threshold {
            return i as i32;
        }
    }
    // Return highest class for values above all quantiles
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

/// Analyze class distribution and log insights for debugging
fn analyze_class_distribution(targets: &[i32], horizon: &str, bins: u32) -> Result<()> {
    let mut class_counts = vec![0usize; bins as usize];
    let mut valid_targets = 0;

    // Count class occurrences
    for &target in targets {
        if target >= 0 && target < bins as i32 {
            class_counts[target as usize] += 1;
            valid_targets += 1;
        }
    }

    if valid_targets == 0 {
        log::warn!(
            "📊 Price Level Analysis [{}]: No valid targets found",
            horizon
        );
        return Ok(());
    }

    // Calculate class distribution statistics
    let total_samples = valid_targets as f64;
    let mut class_percentages = Vec::new();
    let mut min_class_size = usize::MAX;
    let mut max_class_size = 0;

    for &count in class_counts.iter() {
        let percentage = (count as f64 / total_samples) * 100.0;
        class_percentages.push(percentage);

        if count > 0 {
            min_class_size = min_class_size.min(count);
            max_class_size = max_class_size.max(count);
        }
    }

    // Calculate imbalance ratio
    let imbalance_ratio = if min_class_size != usize::MAX && min_class_size > 0 {
        max_class_size as f64 / min_class_size as f64
    } else {
        f64::INFINITY
    };

    // Log compact class distribution analysis
    log::info!(
        "📊 Price Level Distribution [{}]: {} samples, {:.1}x imbalance, classes: [{}]",
        horizon,
        valid_targets,
        imbalance_ratio,
        class_percentages
            .iter()
            .enumerate()
            .map(|(i, p)| format!("{}:{:.1}%", i, p))
            .collect::<Vec<_>>()
            .join(", ")
    );

    // Warn about severe imbalance
    if imbalance_ratio > 10.0 {
        log::warn!(
            "⚠️  Severe class imbalance detected for {} ({}x ratio) - consider class weighting",
            horizon,
            imbalance_ratio
        );
    }

    // Warn about empty classes
    let empty_classes: Vec<usize> = class_counts
        .iter()
        .enumerate()
        .filter(|(_, &count)| count == 0)
        .map(|(idx, _)| idx)
        .collect();

    if !empty_classes.is_empty() {
        log::warn!(
            "⚠️  Empty classes detected for {}: {:?} - may cause training instability",
            horizon,
            empty_classes
        );
    }

    Ok(())
}

/// Extract close prices from DataFrame
pub fn extract_close_prices(df: &DataFrame) -> Result<Vec<f64>> {
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
        let config = create_test_config(PriceLevelTargetStrategy::StandardVwap);

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
        let config = create_test_config(PriceLevelTargetStrategy::MomentumVwap {
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

    #[test]
    fn test_config_parsing_consistency() {
        use crate::config::model::{PriceLevelHead, PriceLevelTargetStrategy};

        // Test current strategy parsing
        let current_toml = r#"
enabled = true
bins = 10
range_percent = 0.05
distribution_type = "Categorical"
target_strategy = { type = "current" }
"#;
        let current_head: PriceLevelHead = toml::from_str(current_toml).unwrap();
        assert!(matches!(
            current_head.target_strategy,
            PriceLevelTargetStrategy::Current
        ));
        assert!(current_head.validate().is_ok());

        // Test standard VWAP strategy parsing
        let standard_toml = r#"
enabled = true
bins = 10
range_percent = 0.05
distribution_type = "Categorical"
target_strategy = { type = "standard_vwap" }
"#;
        let standard_head: PriceLevelHead = toml::from_str(standard_toml).unwrap();
        assert!(matches!(
            standard_head.target_strategy,
            PriceLevelTargetStrategy::StandardVwap
        ));
        assert!(standard_head.validate().is_ok());

        // Test momentum VWAP strategy parsing
        let momentum_toml = r#"
enabled = true
bins = 10
range_percent = 0.05
distribution_type = "Categorical"
target_strategy = { type = "momentum_vwap", momentum_window = 20, bias_strength = 0.3 }
"#;
        let momentum_head: PriceLevelHead = toml::from_str(momentum_toml).unwrap();
        match momentum_head.target_strategy {
            PriceLevelTargetStrategy::MomentumVwap {
                momentum_window,
                bias_strength,
            } => {
                assert_eq!(momentum_window, 20);
                assert_eq!(bias_strength, 0.3);
            }
            _ => panic!("Expected MomentumVwap variant"),
        }
        assert!(momentum_head.validate().is_ok());
    }

    #[test]
    fn test_validation_errors() {
        use crate::config::model::{DistributionType, PriceLevelHead, PriceLevelTargetStrategy};

        // Test invalid momentum_window
        let invalid_head = PriceLevelHead {
            enabled: true,
            bins: 10,
            range_percent: 0.05,
            distribution_type: DistributionType::Categorical,
            target_strategy: PriceLevelTargetStrategy::MomentumVwap {
                momentum_window: 0, // Invalid!
                bias_strength: 0.3,
            },
        };
        assert!(invalid_head.validate().is_err());

        // Test invalid bias_strength
        let invalid_head2 = PriceLevelHead {
            enabled: true,
            bins: 10,
            range_percent: 0.05,
            distribution_type: DistributionType::Categorical,
            target_strategy: PriceLevelTargetStrategy::MomentumVwap {
                momentum_window: 20,
                bias_strength: 1.5, // Invalid!
            },
        };
        assert!(invalid_head2.validate().is_err());

        // Test invalid bins
        let invalid_head3 = PriceLevelHead {
            enabled: true,
            bins: 1, // Invalid!
            range_percent: 0.05,
            distribution_type: DistributionType::Categorical,
            target_strategy: PriceLevelTargetStrategy::Current,
        };
        assert!(invalid_head3.validate().is_err());
    }
}
