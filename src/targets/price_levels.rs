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
    /// Bandwidth multiplier for breakout sensitivity (default: 1.0)
    /// - 0.5: More sensitive (smaller breakout thresholds)
    /// - 1.0: Standard behavior
    /// - 1.5: Less sensitive (larger breakout thresholds)
    pub bandwidth_size: f64,
}

impl Default for PriceLevelConfig {
    fn default() -> Self {
        Self {
            bandwidth_size: 1.0,
        }
    }
}

impl PriceLevelConfig {
    /// Validate the configuration parameters
    pub fn validate(&self) -> Result<()> {
        if self.bandwidth_size <= 0.0 {
            return Err(crate::utils::error::VangaError::ConfigError(format!(
                "bandwidth_size must be positive, got: {}",
                self.bandwidth_size
            )));
        }

        if !self.bandwidth_size.is_finite() {
            return Err(crate::utils::error::VangaError::ConfigError(
                "bandwidth_size must be a finite number".to_string(),
            ));
        }

        Ok(())
    }
}

/// Generate price level targets using PriceLevelConfig
pub fn generate_price_level_targets(
    df: &DataFrame,
    horizons: &[String],
    config: &PriceLevelConfig,
) -> Result<HashMap<String, Vec<i32>>> {
    // Validate configuration
    config.validate()?;

    let close_prices = extract_close_prices(df)?;
    let mut targets = HashMap::new();

    for horizon in horizons {
        let horizon_steps = parse_horizon_to_steps(horizon)?;
        let price_targets = calculate_price_level_targets(
            &close_prices,
            horizon_steps,
            config,
            None,
            None, // No sequence length override - use default behavior
        )?;

        // Analyze and log class distribution (5 classes)
        analyze_class_distribution(&price_targets, horizon, 5)?;

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
    let config = PriceLevelConfig {
        bandwidth_size: model_config
            .output_heads
            .price_levels
            .bandwidth_size
            .unwrap_or(1.0),
    };
    generate_price_level_targets(df, horizons, &config)
}

/// Calculate minimum data points required for target generation (fallback when FeatureConfig not available)
fn calculate_min_data_points(_config: &PriceLevelConfig) -> usize {
    // Default maximum feature window from technical indicators
    // Based on maximum periods used in feature engineering:
    // - SMA/EMA periods: up to 200
    // - RSI, MACD, Bollinger Bands: ~26-50 periods
    // - Volume indicators: ~20-50 periods
    let max_feature_window = 250; // Conservative estimate
    let stability_buffer = 50;
    max_feature_window + stability_buffer
}

/// Classify price level using sequence-aware 5-class classification
///
/// **5-Class System:**
/// - 0: Strong Breakout Down (< min - bandwidth)
/// - 1: Moderate Down (min - bandwidth ≤ x < min)
/// - 2: Neutral (min ≤ x < max) - Merged from previous Range Low + Range High
/// - 3: Moderate Up (max ≤ x < max + bandwidth)
/// - 4: Strong Breakout Up (≥ max + bandwidth)
///
/// # Arguments
/// * `target_price` - The future price to classify
/// * `sequence_prices` - The input sequence prices (min length: 2)
/// * `config` - Configuration for bandwidth sensitivity
///
/// # Returns
/// * `i32` - Classification bin [0-4]
fn classify_price_level_sequence_aware(
    target_price: f64,
    sequence_prices: &[f64],
    config: &PriceLevelConfig,
) -> i32 {
    if sequence_prices.is_empty() {
        return 2; // Default to neutral class
    }

    let sequence_min = sequence_prices.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let sequence_max = sequence_prices
        .iter()
        .fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let base_bandwidth = sequence_max - sequence_min;
    let bandwidth = base_bandwidth * config.bandwidth_size;

    // Handle edge case: flat sequence (bandwidth = 0)
    if bandwidth == 0.0 {
        return if target_price >= sequence_min { 3 } else { 2 };
    }

    // 5-class classification
    if target_price < sequence_min - bandwidth {
        0 // Strong Breakout Down
    } else if target_price < sequence_min {
        1 // Moderate Down
    } else if target_price < sequence_max {
        2 // Neutral (merged Range Low + Range High)
    } else if target_price < sequence_max + bandwidth {
        3 // Moderate Up
    } else {
        4 // Strong Breakout Up
    }
}

/// Calculate price level targets for a specific horizon with consistent global quantiles
fn calculate_price_level_targets(
    prices: &[f64],
    horizon_steps: usize,
    config: &PriceLevelConfig,
    feature_window: Option<usize>,
    _sequence_length_override: Option<usize>,
) -> Result<Vec<i32>> {
    // Validate configuration
    config.validate()?;

    // Use provided feature_window or fallback to default calculation for sequence length
    let sequence_length = feature_window.unwrap_or_else(|| calculate_min_data_points(config));

    if prices.len() < horizon_steps + sequence_length {
        return Err(crate::utils::error::VangaError::DataError(format!(
            "Insufficient data for price level target generation. Need {} points, have {}",
            horizon_steps + sequence_length,
            prices.len()
        )));
    }

    // Only sequence-aware approach now
    calculate_sequence_aware_targets(prices, horizon_steps, sequence_length, config)
}

/// Calculate sequence-aware price level targets using sequence context
///
/// This function generates targets using the sequence-aware approach where each prediction
/// point uses its own sequence context (min, max, current) to define adaptive boundaries.
/// No global quantiles are needed - each sequence defines its own classification boundaries.
fn calculate_sequence_aware_targets(
    prices: &[f64],
    horizon_steps: usize,
    sequence_length: usize,
    config: &PriceLevelConfig,
) -> Result<Vec<i32>> {
    let mut targets = vec![-1; prices.len()];

    // Start from sequence_length to have enough history
    let start_idx = sequence_length;
    let end_idx = prices.len().saturating_sub(horizon_steps);

    if end_idx <= start_idx {
        return Err(crate::utils::error::VangaError::DataError(
            "Insufficient data for sequence-aware target generation".to_string(),
        ));
    }

    log::debug!(
        "🔄 Generating sequence-aware targets: sequence_length={}, start_idx={}, end_idx={}, total_targets={}",
        sequence_length,
        start_idx,
        end_idx,
        end_idx - start_idx
    );

    for i in start_idx..end_idx {
        // Extract sequence for this prediction point
        let sequence_start = i.saturating_sub(sequence_length);
        let sequence = &prices[sequence_start..i];

        // Get target price
        let target_price = prices[i + horizon_steps];

        // Classify using sequence-aware approach
        targets[i] = classify_price_level_sequence_aware(target_price, sequence, config);
    }

    Ok(targets)
}

/// Analyze class distribution and log insights for debugging with imbalance mitigation
fn analyze_class_distribution(targets: &[i32], horizon: &str, bins: u32) -> Result<()> {
    use crate::targets::imbalance_mitigation::{
        ClassDistributionAnalysis, ImbalanceMitigationConfig, ImbalanceMitigator,
    };

    // Perform advanced analysis
    let mitigation_config = ImbalanceMitigationConfig::default();
    let analysis = ClassDistributionAnalysis::analyze(targets, bins as usize, &mitigation_config);

    // Generate and log recommendations if imbalance is severe
    if analysis.imbalance_ratio > mitigation_config.max_imbalance_ratio {
        let current_config = PriceLevelConfig::default();
        let recommendations = ImbalanceMitigator::generate_recommendations(
            &analysis,
            &current_config,
            &mitigation_config,
        );
        recommendations.log_recommendations(horizon);
    }

    // Continue with existing logging for compatibility
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

    // Check for problematic imbalance
    if imbalance_ratio > 100.0 {
        log::warn!(
            "⚠️  Severe class imbalance detected for {} ({:.0}x ratio) - consider class weighting",
            horizon,
            imbalance_ratio
        );
    }

    // Identify empty classes
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

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_prices() -> Vec<f64> {
        // Create enough data for testing (sequence_length + horizon + some extra)
        (0..400).map(|i| 100.0 + i as f64).collect()
    }

    fn create_test_config() -> PriceLevelConfig {
        PriceLevelConfig {
            bandwidth_size: 1.0,
        }
    }

    #[test]
    fn test_sequence_aware_classification_5_classes() {
        // Test case: sequence [10, 15, 12, 18, 14]
        // min=10, max=18, base_bandwidth=8, bandwidth=8*1.0=8
        let sequence = vec![10.0, 15.0, 12.0, 18.0, 14.0];
        let config = create_test_config();

        // Test each of the 5 classes
        assert_eq!(
            classify_price_level_sequence_aware(1.0, &sequence, &config),
            0
        ); // < 10-8 = 2 (Strong Breakout Down)
        assert_eq!(
            classify_price_level_sequence_aware(5.0, &sequence, &config),
            1
        ); // 2 ≤ x < 10 (Moderate Down)
        assert_eq!(
            classify_price_level_sequence_aware(14.0, &sequence, &config),
            2
        ); // 10 ≤ x < 18 (Neutral - merged range)
        assert_eq!(
            classify_price_level_sequence_aware(20.0, &sequence, &config),
            3
        ); // 18 ≤ x < 26 (Moderate Up)
        assert_eq!(
            classify_price_level_sequence_aware(30.0, &sequence, &config),
            4
        ); // ≥ 26 (Strong Breakout Up)
    }

    #[test]
    fn test_flat_sequence_edge_case() {
        let sequence = vec![10.0, 10.0, 10.0, 10.0, 10.0]; // bandwidth = 0
        let config = create_test_config();

        assert_eq!(
            classify_price_level_sequence_aware(9.0, &sequence, &config),
            2
        ); // < current
        assert_eq!(
            classify_price_level_sequence_aware(11.0, &sequence, &config),
            3
        ); // ≥ current
    }

    #[test]
    fn test_empty_sequence_edge_case() {
        let sequence = vec![];
        let config = create_test_config();

        assert_eq!(
            classify_price_level_sequence_aware(100.0, &sequence, &config),
            2
        ); // Default neutral
    }

    #[test]
    fn test_bandwidth_size_multiplier() {
        // Test case: sequence [10, 15, 12, 18, 14]
        // min=10, max=18, current=14, base_bandwidth=8
        let sequence = vec![10.0, 15.0, 12.0, 18.0, 14.0];

        // Test with sensitive configuration (bandwidth_size = 0.5)
        let sensitive_config = PriceLevelConfig {
            bandwidth_size: 0.5,
        };
        // bandwidth = 8 * 0.5 = 4
        // Breakout thresholds: < 6.0 (bin 0), ≥ 22.0 (bin 4)
        assert_eq!(
            classify_price_level_sequence_aware(5.0, &sequence, &sensitive_config),
            0
        ); // < 10-4 = 6
        assert_eq!(
            classify_price_level_sequence_aware(23.0, &sequence, &sensitive_config),
            4
        ); // ≥ 18+4 = 22

        // Test with conservative configuration (bandwidth_size = 1.5)
        let conservative_config = PriceLevelConfig {
            bandwidth_size: 1.5,
        };
        // bandwidth = 8 * 1.5 = 12
        // Breakout thresholds: < -2.0 (bin 0), ≥ 30.0 (bin 4)
        assert_eq!(
            classify_price_level_sequence_aware(-3.0, &sequence, &conservative_config),
            0
        ); // < 10-12 = -2
        assert_eq!(
            classify_price_level_sequence_aware(31.0, &sequence, &conservative_config),
            4
        ); // ≥ 18+12 = 30

        // Same price should be classified differently with different bandwidth_size
        let test_price = 25.0;
        assert_eq!(
            classify_price_level_sequence_aware(test_price, &sequence, &sensitive_config),
            4
        ); // Strong breakout (25 ≥ 22)
        assert_eq!(
            classify_price_level_sequence_aware(test_price, &sequence, &conservative_config),
            4
        ); // Moderate up (18 ≤ 25 < 30)
    }

    #[test]
    fn test_sequence_aware_targets_generation() {
        let prices = create_test_prices();
        let config = create_test_config();

        let result = calculate_price_level_targets(&prices, 2, &config, Some(50), None).unwrap();

        // Should have valid targets for middle indices
        assert!(result.len() == prices.len());

        // Check that we have some valid targets (not all -1)
        let valid_targets: Vec<_> = result.iter().filter(|&&x| x != -1).collect();
        assert!(!valid_targets.is_empty());

        // All valid targets should be in range [0, 4] for 5-class system
        for &target in &valid_targets {
            assert!((0..=4).contains(target));
        }
    }

    #[test]
    fn test_integration_with_dataframe() {
        // Test the full pipeline with sample data
        use polars::prelude::*;
        let prices: Vec<f64> = (0..400).map(|i| 100.0 + i as f64).collect();
        let df = df! {
            "close" => prices
        }
        .unwrap();

        let horizons = vec!["1h".to_string()];
        let config = create_test_config();

        let targets = generate_price_level_targets(&df, &horizons, &config).unwrap();

        // Verify targets are in valid range [0, 4] for 5-class system
        for target_vec in targets.values() {
            for &target in target_vec {
                if target != -1 {
                    assert!((0..=4).contains(&target));
                }
            }
        }
    }

    #[test]
    fn test_config_validation() {
        // Test valid config
        let valid_config = create_test_config();
        assert_eq!(valid_config.bandwidth_size, 1.0);
        assert!(valid_config.validate().is_ok());

        // Test different bandwidth_size values
        let sensitive_config = PriceLevelConfig {
            bandwidth_size: 0.5,
        };
        assert_eq!(sensitive_config.bandwidth_size, 0.5);
        assert!(sensitive_config.validate().is_ok());

        let conservative_config = PriceLevelConfig {
            bandwidth_size: 1.5,
        };
        assert_eq!(conservative_config.bandwidth_size, 1.5);
        assert!(conservative_config.validate().is_ok());

        // Test invalid configs
        let zero_config = PriceLevelConfig {
            bandwidth_size: 0.0,
        };
        assert!(zero_config.validate().is_err());

        let negative_config = PriceLevelConfig {
            bandwidth_size: -1.0,
        };
        assert!(negative_config.validate().is_err());

        let infinite_config = PriceLevelConfig {
            bandwidth_size: f64::INFINITY,
        };
        assert!(infinite_config.validate().is_err());

        let nan_config = PriceLevelConfig {
            bandwidth_size: f64::NAN,
        };
        assert!(nan_config.validate().is_err());
    }
}
