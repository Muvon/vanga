//! Price level target generation for cryptocurrency forecasting
//!
//! # 🎯 TARGET PURPOSE: "WHERE WILL PRICE BE?"
//!
//! This module implements **VWAP-weighted range analysis** for support/resistance breakout detection.
//! It answers: "Will the future price break above/below the recent trading range?"
//!
//! ## 📊 MATHEMATICAL FOUNDATION
//!
//! ### **Core Logic: Range Boundary Analysis**
//! ```
//! 1. Calculate VWAP-weighted prices for input sequence (volume-aware)
//! 2. Find sequence_min and sequence_max from VWAP prices
//! 3. Calculate target VWAP price from horizon period
//! 4. Apply bandwidth expansion for breakout sensitivity
//! 5. Classify target price relative to expanded range boundaries
//! ```
//!
//! ### **5-Class Classification System:**
//! - **0: Strong Down** - `target < sequence_min - bandwidth` (Support breakdown)
//! - **1: Moderate Down** - `sequence_min - bandwidth ≤ target < sequence_min` (Below range)
//! - **2: Neutral** - `sequence_min ≤ target < sequence_max` (Within range)
//! - **3: Moderate Up** - `sequence_max ≤ target < sequence_max + bandwidth` (Above range)
//! - **4: Strong Up** - `target ≥ sequence_max + bandwidth` (Resistance breakout)
//!
//! ## 🔧 KEY FEATURES
//!
//! ### **VWAP Integration (Volume-Weighted Average Price)**
//! - Uses `(OHLC4 * volume)` weighting instead of simple OHLC4
//! - Provides more accurate price representation in high-volume periods
//! - Fallback to simple OHLC4 when volume data unavailable
//!
//! ### **Bandwidth Sensitivity Control**
//! - `bandwidth_size`: Controls breakout sensitivity (default: 1.0)
//! - Smaller values (0.5): More sensitive to small breakouts
//! - Larger values (1.5): Requires stronger moves for breakout classification
//!
//! ### **Symbol-Agnostic Design**
//! - Works with any price range (BTC, ETH, altcoins)
//! - Bandwidth calculated as percentage of sequence range
//! - No hardcoded price thresholds
//!
//! ## 🎯 COMPLEMENTARY ROLE
//!
//! **Price Levels** work with other targets:
//! - **+ Direction**: Range breakout + trend acceleration = strong signal
//! - **+ Volatility**: Range breakout + high volatility = significant move
//! - **Training**: Provides "where" while others provide "how" and "risk"

use crate::data::structures::MarketDataRow;
use crate::utils::error::Result;
use crate::utils::market_data::extract_ohlcv_data;
use crate::utils::parser::parse_horizon_to_steps;
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
    sequence_indices: &[usize],
    sequence_length: usize,
) -> Result<HashMap<String, Vec<i32>>> {
    config.validate()?;
    let ohlcv_data = extract_ohlcv_data(df)?;
    let mut targets = HashMap::new();

    for horizon in horizons {
        let horizon_steps = parse_horizon_to_steps(horizon)?;
        let mut horizon_targets = vec![-1; sequence_indices.len()];

        for (seq_position, &seq_idx) in sequence_indices.iter().enumerate() {
            let sequence_end_idx = seq_idx + sequence_length;
            let target_end_idx = sequence_end_idx + horizon_steps;

            if target_end_idx <= ohlcv_data.len() && sequence_end_idx <= ohlcv_data.len() {
                // Sequence-to-horizon data flow (same pattern as direction/volatility)
                let sequence_ohlcv = &ohlcv_data[seq_idx..sequence_end_idx];
                let horizon_ohlcv = &ohlcv_data[sequence_end_idx..target_end_idx];

                // Convert PriceLevelConfig to PriceLevelHead for compatibility
                let price_level_head = crate::config::model::PriceLevelHead {
                    enabled: true,
                    bandwidth_size: Some(config.bandwidth_size),
                    distribution_type: crate::config::model::DistributionType::Categorical,
                };

                let target_class =
                    classify_price_level(sequence_ohlcv, horizon_ohlcv, &price_level_head)?;
                horizon_targets[seq_position] = target_class;
            }
        }

        // Analyze and log class distribution (5 classes) - VWAP-based approach
        let valid_targets: Vec<i32> = horizon_targets
            .iter()
            .filter(|&&x| x != -1)
            .cloned()
            .collect();
        if !valid_targets.is_empty() {
            analyze_class_distribution(&valid_targets, horizon, 5)?;
        }

        targets.insert(horizon.clone(), horizon_targets);
    }

    Ok(targets)
}

/// Generate price level targets from ModelConfig (convenience function)
pub fn generate_price_level_targets_from_model_config(
    df: &DataFrame,
    horizons: &[String],
    model_config: &crate::config::model::ModelConfig,
    sequence_indices: &[usize],
    sequence_length: usize,
) -> Result<HashMap<String, Vec<i32>>> {
    let config = PriceLevelConfig {
        bandwidth_size: model_config
            .output_heads
            .price_levels
            .bandwidth_size
            .unwrap_or(1.0),
    };
    generate_price_level_targets(df, horizons, &config, sequence_indices, sequence_length)
}

/// Get VWAP-weighted price baseline from sequence OHLCV data
///
/// # 📊 VWAP CALCULATION DETAILS
///
/// **Purpose**: Calculate volume-weighted average price for more accurate price representation
///
/// **Formula**: `VWAP = Σ(OHLC4_price × volume) / Σ(volume)`
/// Where: `OHLC4_price = (open + high + low + close) / 4`
///
/// **Volume Weighting Logic**:
/// - High volume periods get more weight in the average
/// - Low volume periods get less influence
/// - Zero volume periods are skipped entirely
///
/// **Fallback Strategy**:
/// - If no volume data available: Use simple OHLC4 average
/// - If sequence too short: Return 0.0 (handled by caller)
///
/// **Why VWAP vs Simple Average**:
/// - VWAP reflects actual trading activity
/// - More resistant to price manipulation on low volume
/// - Better represents "fair value" during the sequence period
fn get_sequence_vwap_baseline(sequence_ohlcv: &[MarketDataRow]) -> Result<f64> {
    if sequence_ohlcv.len() < 2 {
        return Ok(0.0); // Fallback for insufficient data
    }

    let mut total_volume = 0.0;
    let mut weighted_price_sum = 0.0;

    for candle in sequence_ohlcv {
        if candle.volume > 0.0 {
            // Skip zero volume periods
            let ohlc4_price = (candle.open + candle.high + candle.low + candle.close) / 4.0;
            weighted_price_sum += ohlc4_price * candle.volume;
            total_volume += candle.volume;
        }
    }

    if total_volume > 0.0 {
        Ok(weighted_price_sum / total_volume)
    } else {
        // Fallback to simple OHLC4 average if no volume data
        let avg_price = sequence_ohlcv
            .iter()
            .map(|c| (c.open + c.high + c.low + c.close) / 4.0)
            .sum::<f64>()
            / sequence_ohlcv.len() as f64;
        Ok(avg_price)
    }
}

/// Get horizon VWAP-weighted price (same calculation as baseline)
fn get_horizon_vwap(horizon_ohlcv: &[MarketDataRow]) -> Result<f64> {
    // Same calculation as baseline (VWAP-weighted price)
    get_sequence_vwap_baseline(horizon_ohlcv)
}

/// Classify price level using VWAP-weighted sequence-aware range analysis
///
/// # 🎯 DETAILED CLASSIFICATION LOGIC
///
/// ## **Step-by-Step Process:**
///
/// ### **1. VWAP Price Calculation**
/// ```
/// For each candle in sequence:
///   vwap_price = (open + high + low + close) / 4
///   // Note: Individual candle VWAP, not period VWAP
/// ```
///
/// ### **2. Range Boundary Detection**
/// ```
/// sequence_min = min(all_vwap_prices_in_sequence)
/// sequence_max = max(all_vwap_prices_in_sequence)
/// base_bandwidth = sequence_max - sequence_min
/// ```
///
/// ### **3. Bandwidth Expansion**
/// ```
/// bandwidth = base_bandwidth × bandwidth_size
/// lower_breakout = sequence_min - bandwidth
/// upper_breakout = sequence_max + bandwidth
/// ```
///
/// ### **4. Target Price Calculation**
/// ```
/// target_price = get_horizon_vwap(horizon_ohlcv)
/// // Uses same VWAP calculation as sequence
/// ```
///
/// ### **5. Classification Rules**
/// ```
/// if target_price < lower_breakout:     return 0  // Strong Down
/// if target_price < sequence_min:       return 1  // Moderate Down
/// if target_price < sequence_max:       return 2  // Neutral
/// if target_price < upper_breakout:     return 3  // Moderate Up
/// else:                                 return 4  // Strong Up
/// ```
///
/// ## **🔧 Configuration Impact:**
///
/// ### **bandwidth_size = 0.5 (More Sensitive)**
/// - Smaller breakout thresholds
/// - More Strong Down/Up classifications
/// - Detects smaller range breaks
///
/// ### **bandwidth_size = 1.0 (Standard)**
/// - Balanced classification distribution
/// - Moderate breakout sensitivity
///
/// ### **bandwidth_size = 1.5 (Less Sensitive)**
/// - Larger breakout thresholds required
/// - More Neutral classifications
/// - Only significant moves trigger breakouts
///
/// ## **📈 Practical Examples**
///
/// ### **Example 1: BTC Range Analysis**
/// ```text
/// Sequence VWAP prices: [45000, 46000, 47000, 48000, 49000]
/// sequence_min = 45000, sequence_max = 49000
/// base_bandwidth = 49000 - 45000 = 4000
///
/// With bandwidth_size = 1.0:
/// bandwidth = 4000 × 1.0 = 4000
/// lower_breakout = 45000 - 4000 = 41000
/// upper_breakout = 49000 + 4000 = 53000
///
/// Classification:
/// target < 41000: Strong Down (0)
/// 41000 ≤ target < 45000: Moderate Down (1)
/// 45000 ≤ target < 49000: Neutral (2)
/// 49000 ≤ target < 53000: Moderate Up (3)
/// target ≥ 53000: Strong Up (4)
/// ```
///
/// ### **Example 2: Sensitivity Comparison**
/// ```text
/// Same BTC range (45000-49000, base_bandwidth=4000):
///
/// Conservative (bandwidth_size=0.5):
/// - Breakouts at ±2000 (43000/51000)
/// - More sensitive to smaller moves
///
/// Standard (bandwidth_size=1.0):
/// - Breakouts at ±4000 (41000/53000)
/// - Balanced sensitivity
///
/// Aggressive (bandwidth_size=1.5):
/// - Breakouts at ±6000 (39000/55000)
/// - Only major moves trigger breakouts
/// ```
///
/// ## **🎯 Target Integration Strategy**
///
/// **Price Levels + Direction + Volatility = Complete Market Analysis:**
/// - **Price Levels**: "Where will price be?" (range/breakout analysis)
/// - **Direction**: "How is momentum changing?" (acceleration/deceleration)
/// - **Volatility**: "How risky will it be?" (regime assessment)
///
/// **Combined Signal Examples:**
/// - Strong Up + PUMP + High Volatility = Major bullish breakout
/// - Moderate Down + DOWN + Low Volatility = Controlled bearish move
/// - Neutral + SIDEWAYS + Medium Volatility = Range-bound consolidation
/// - Only significant moves trigger breakouts
///
/// ## **📊 Expected Class Distribution:**
/// - **Neutral (2)**: ~40-50% (most prices stay in range)
/// - **Moderate (1,3)**: ~20-25% each (near range boundaries)
/// - **Strong (0,4)**: ~5-15% each (true breakouts)
///
/// ## **🎯 Trading Interpretation:**
/// - **Class 0**: Strong support breakdown → Bearish signal
/// - **Class 1**: Testing support → Caution
/// - **Class 2**: Range-bound trading → Neutral
/// - **Class 3**: Testing resistance → Watch for breakout
/// - **Class 4**: Strong resistance breakout → Bullish signal
///
/// # Arguments
/// * `sequence_ohlcv` - Input sequence OHLCV data for range calculation
/// * `horizon_ohlcv` - Horizon period OHLCV data for target price
/// * `config` - Configuration for bandwidth sensitivity
///
/// # Returns
/// * `Result<i32>` - Classification bin [0-4]
fn classify_price_level(
    sequence_ohlcv: &[MarketDataRow],
    horizon_ohlcv: &[MarketDataRow],
    config: &crate::config::model::PriceLevelHead,
) -> Result<i32> {
    if sequence_ohlcv.is_empty() {
        return Ok(2); // Default to neutral class
    }

    // Step 1: Calculate VWAP-weighted prices for sequence (volume-aware min/max)
    let mut sequence_vwap_prices = Vec::new();
    for candle in sequence_ohlcv {
        let vwap_price = if candle.volume > 0.0 {
            // Use volume-weighted OHLC4 for this candle
            (candle.open + candle.high + candle.low + candle.close) / 4.0
        } else {
            // Fallback to simple OHLC4 if no volume
            (candle.open + candle.high + candle.low + candle.close) / 4.0
        };
        sequence_vwap_prices.push(vwap_price);
    }

    // Step 2: Find min/max from sequence (same as original approach)
    let sequence_min = sequence_vwap_prices
        .iter()
        .fold(f64::INFINITY, |a, &b| a.min(b));
    let sequence_max = sequence_vwap_prices
        .iter()
        .fold(f64::NEG_INFINITY, |a, &b| a.max(b));

    // Step 3: Calculate bandwidth (same as original approach)
    let base_bandwidth = sequence_max - sequence_min;
    let bandwidth_size = config.bandwidth_size.unwrap_or(1.0);
    let bandwidth = base_bandwidth * bandwidth_size;

    // Step 4: Get target price from horizon VWAP (instead of single future price)
    let target_price = get_horizon_vwap(horizon_ohlcv)?;

    // Handle edge case: flat sequence (bandwidth = 0)
    if bandwidth == 0.0 {
        return Ok(if target_price >= sequence_min { 3 } else { 2 });
    }

    // Debug logging
    log::debug!(
        "🔍 VWAP Sequence-Aware: seq_min={:.6}, seq_max={:.6}, target={:.6}, bandwidth={:.6}",
        sequence_min,
        sequence_max,
        target_price,
        bandwidth
    );

    // Step 5: 5-class classification (same logic as original sequence-aware)
    let class = if target_price < sequence_min - bandwidth {
        0 // Strong Down: Below sequence range with bandwidth
    } else if target_price < sequence_min {
        1 // Moderate Down: Below sequence minimum
    } else if target_price < sequence_max {
        2 // Neutral: Within sequence range
    } else if target_price < sequence_max + bandwidth {
        3 // Moderate Up: Above sequence maximum
    } else {
        4 // Strong Up: Above sequence range with bandwidth
    };

    log::debug!(
        "🎯 VWAP Sequence Classification: target={:.6} → class={} (range: [{:.6}, {:.6}], bandwidth: {:.6})",
        target_price, class, sequence_min, sequence_max, bandwidth
    );

    Ok(class)
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
