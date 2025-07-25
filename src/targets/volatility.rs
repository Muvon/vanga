//! Volatility target generation for cryptocurrency market regime classification
//!
//! This module implements volatility regime classification for risk assessment:
//! - 0: VeryLow (minimal volatility)
//! - 1: Low (below average volatility)
//! - 2: Medium (average volatility)
//! - 3: High (above average volatility)
//! - 4: VeryHigh (extreme volatility)

use crate::config::model::VolatilityHead;
use crate::utils::error::Result;
use polars::prelude::*;
use std::collections::HashMap;

/// Configuration for volatility target generation (for consistency with direction pattern)
#[derive(Debug, Clone)]
pub struct VolatilityConfig {
    /// Bandwidth multiplier for volatility sensitivity (default: 1.0)
    /// - 0.5: More sensitive (smaller volatility thresholds, more classes in extreme ranges)
    /// - 1.0: Standard behavior
    /// - 1.5: Less sensitive (larger volatility thresholds, more balanced distribution)
    pub bandwidth_size: f64,
}

impl Default for VolatilityConfig {
    fn default() -> Self {
        Self {
            bandwidth_size: 1.0,
        }
    }
}

impl VolatilityConfig {
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

/// Generate volatility targets for multiple horizons (ENHANCED: supports both configs)
pub fn generate_volatility_targets(
    df: &DataFrame,
    horizons: &[String],
    config: Option<&VolatilityConfig>,
    model_config: Option<&VolatilityHead>,
) -> Result<HashMap<String, Vec<i32>>> {
    let close_prices = extract_close_prices(df)?;
    let mut targets = HashMap::new();

    for horizon in horizons {
        let horizon_steps = parse_horizon_to_steps(horizon)?;

        let volatility_targets = if let Some(model_cfg) = model_config {
            // Use model config (NEW: eliminates hardcoded values)
            let bandwidth_size = model_cfg.bandwidth_size.unwrap_or(1.0);
            let base_percentiles = model_cfg.base_percentiles;

            // Apply bandwidth sensitivity to percentiles
            let center = 0.5;
            let adaptive_percentiles: [f64; 4] = base_percentiles.map(|p| {
                let distance = p - center;
                let sensitivity = 1.0 / bandwidth_size;
                center + (distance * sensitivity)
            });

            apply_volatility_classification(&close_prices, horizon_steps, &adaptive_percentiles)?
        } else if let Some(legacy_cfg) = config {
            // Legacy path for backward compatibility
            calculate_volatility_targets(&close_prices, horizon_steps, legacy_cfg)?
        } else {
            return Err(crate::utils::error::VangaError::config(
                "Either VolatilityConfig or VolatilityHead must be provided",
            ));
        };

        // Analyze and log class distribution for this specific horizon
        analyze_class_distribution(&volatility_targets, 5, horizon)?;

        targets.insert(horizon.clone(), volatility_targets);
    }

    Ok(targets)
}

/// Apply volatility classification using model config
fn apply_volatility_classification(
    prices: &[f64],
    horizon_steps: usize,
    adaptive_percentiles: &[f64; 4],
) -> Result<Vec<i32>> {
    let volatility_window = 24; // 24-hour rolling window for volatility calculation

    if prices.len() < volatility_window + horizon_steps {
        return Err(crate::utils::error::VangaError::DataError(format!(
            "Insufficient data for volatility target generation: need {}, got {}",
            volatility_window + horizon_steps,
            prices.len()
        )));
    }

    let mut targets = vec![-1; prices.len()];

    // Calculate current volatility series for threshold determination
    let current_volatility = calculate_realized_volatility(prices, volatility_window)?;

    // Calculate regime boundaries from current volatility for consistent classification
    let regime_boundaries =
        calculate_percentile_boundaries(&current_volatility, adaptive_percentiles)?;

    // For each valid position, calculate future volatility and classify
    for (i, target) in targets
        .iter_mut()
        .enumerate()
        .take(prices.len() - horizon_steps)
        .skip(volatility_window)
    {
        // Calculate future volatility window starting at horizon
        let future_start = i + horizon_steps;
        let future_end = (future_start + volatility_window).min(prices.len());

        if future_end - future_start < volatility_window / 2 {
            // Skip if insufficient future data for reliable volatility calculation
            continue;
        }

        // Calculate future volatility for this horizon
        let future_prices = &prices[future_start..future_end];
        let future_volatility = calculate_future_volatility(future_prices)?;

        // Classify future volatility using current regime boundaries
        let volatility_class = classify_volatility_regime(future_volatility, &regime_boundaries);
        *target = volatility_class;
    }

    Ok(targets)
}

/// Calculate volatility targets using legacy config (for backward compatibility)
fn calculate_volatility_targets(
    prices: &[f64],
    horizon_steps: usize,
    config: &VolatilityConfig,
) -> Result<Vec<i32>> {
    // Use default percentiles for legacy path
    let base_percentiles = [0.20, 0.40, 0.60, 0.80];
    let sensitivity = 1.0 / config.bandwidth_size;
    let center = 0.5;

    // Apply bandwidth sensitivity to percentiles
    let adaptive_percentiles: [f64; 4] = base_percentiles.map(|p| {
        let distance = p - center;
        center + (distance * sensitivity)
    });

    apply_volatility_classification(prices, horizon_steps, &adaptive_percentiles)
}

/// Calculate future volatility for horizon-specific prediction
fn calculate_future_volatility(prices: &[f64]) -> Result<f64> {
    if prices.len() < 2 {
        return Ok(0.0); // Default for insufficient data
    }

    // Calculate returns for the future period
    let mut returns = Vec::with_capacity(prices.len() - 1);
    for i in 1..prices.len() {
        if prices[i] > 0.0 && prices[i - 1] > 0.0 {
            returns.push((prices[i] / prices[i - 1]).ln());
        } else {
            returns.push(0.0);
        }
    }

    if returns.is_empty() {
        return Ok(0.0);
    }

    // Calculate standard deviation of returns (volatility)
    let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;
    let variance = returns
        .iter()
        .map(|&r| (r - mean_return).powi(2))
        .sum::<f64>()
        / returns.len() as f64;

    Ok(variance.sqrt())
}

/// Classify volatility into regime using boundaries
fn classify_volatility_regime(volatility: f64, boundaries: &[f64; 4]) -> i32 {
    if volatility <= boundaries[0] {
        0 // VeryLow
    } else if volatility <= boundaries[1] {
        1 // Low
    } else if volatility <= boundaries[2] {
        2 // Medium
    } else if volatility <= boundaries[3] {
        3 // High
    } else {
        4 // VeryHigh
    }
}

/// Parse horizon string to number of steps
pub fn parse_horizon_to_steps(horizon: &str) -> Result<usize> {
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

/// Volatility regime classes (5-class system)
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum VolatilityRegime {
    VeryLow = 0,  // <20th percentile
    Low = 1,      // 20th-40th percentile
    Medium = 2,   // 40th-60th percentile
    High = 3,     // 60th-80th percentile
    VeryHigh = 4, // >80th percentile
}

/// Analyze class distribution and log insights for volatility targets per horizon
fn analyze_class_distribution(targets: &[i32], bins: u32, horizon: &str) -> Result<()> {
    let mut class_counts = vec![0usize; bins as usize];
    let mut valid_targets = 0;

    for &target in targets {
        if target >= 0 && target < bins as i32 {
            class_counts[target as usize] += 1;
            valid_targets += 1;
        }
    }

    if valid_targets == 0 {
        log::warn!(
            "⚠️ No valid volatility targets found for horizon {}",
            horizon
        );
        return Ok(());
    }

    log::info!(
        "📊 Volatility Class Distribution for {} (n={})",
        horizon,
        valid_targets
    );
    for (class, &count) in class_counts.iter().enumerate() {
        let percentage = (count as f64 / valid_targets as f64) * 100.0;
        let class_name = match class {
            0 => "VeryLow",
            1 => "Low",
            2 => "Medium",
            3 => "High",
            4 => "VeryHigh",
            _ => "Unknown",
        };
        log::info!(
            "   {} Class {} ({}): {} ({:.1}%)",
            horizon,
            class,
            class_name,
            count,
            percentage
        );
    }

    Ok(())
}

/// Extract close prices from DataFrame
pub fn extract_close_prices(df: &DataFrame) -> Result<Vec<f64>> {
    let close_series = df.column("close").map_err(|e| {
        crate::utils::error::VangaError::DataError(format!(
            "Failed to extract 'close' column: {}",
            e
        ))
    })?;

    let values: Vec<f64> = close_series
        .f64()
        .map_err(|e| {
            crate::utils::error::VangaError::DataError(format!(
                "Failed to convert close prices to f64: {}",
                e
            ))
        })?
        .into_no_null_iter()
        .collect();

    if values.is_empty() {
        return Err(crate::utils::error::VangaError::DataError(
            "No valid close prices found".to_string(),
        ));
    }

    Ok(values)
}

/// Calculate realized volatility using rolling window
fn calculate_realized_volatility(prices: &[f64], window: usize) -> Result<Vec<f64>> {
    if prices.len() < window {
        return Err(crate::utils::error::VangaError::DataError(format!(
            "Insufficient data for volatility calculation: need {}, got {}",
            window,
            prices.len()
        )));
    }

    let mut volatilities = Vec::new();

    for i in window..prices.len() {
        let window_prices = &prices[i - window..i];
        let volatility = calculate_future_volatility(window_prices)?;
        volatilities.push(volatility);
    }

    Ok(volatilities)
}

/// Calculate percentile boundaries for classification
fn calculate_percentile_boundaries(values: &[f64], percentiles: &[f64; 4]) -> Result<[f64; 4]> {
    if values.is_empty() {
        return Err(crate::utils::error::VangaError::DataError(
            "Cannot calculate percentiles from empty data".to_string(),
        ));
    }

    let mut sorted_values = values.to_vec();
    sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let mut boundaries = [0.0; 4];
    for (i, &percentile) in percentiles.iter().enumerate() {
        let index = ((sorted_values.len() - 1) as f64 * percentile) as usize;
        boundaries[i] = sorted_values[index.min(sorted_values.len() - 1)];
    }

    Ok(boundaries)
}
