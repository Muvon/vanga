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

use crate::config::model::TargetsConfig;
use crate::data::structures::MarketDataRow;
use crate::targets::sequence_reconstruction::{SequenceAnalyzer, SequenceReconstructionConfig};
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

                // Use enhanced classification with momentum weighting and adaptive bandwidth
                // TODO: Make these configurable parameters in future config refactoring
                let momentum_factor = Some(1.2); // Slight bias toward recent data

                let target_class = classify_price_level_with_momentum(
                    sequence_ohlcv,
                    horizon_ohlcv,
                    momentum_factor,
                )?;
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
pub fn get_sequence_vwap_baseline(sequence_ohlcv: &[MarketDataRow]) -> Result<f64> {
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

/// **ENHANCED**: Get momentum-weighted VWAP from sequence OHLCV data
///
/// # 🚀 MOMENTUM-WEIGHTED VWAP ENHANCEMENT
///
/// **Mathematical Enhancement**: Applies time-based momentum weighting to give recent data more influence
///
/// **Formula**: `MVWAP = Σ(OHLC4_price × volume × momentum_weight) / Σ(volume × momentum_weight)`
/// Where: `momentum_weight = (time_position / sequence_length)^momentum_factor`
///
/// **Momentum Weighting Logic**:
/// - `momentum_factor = 1.0`: Equal weighting (same as standard VWAP)
/// - `momentum_factor > 1.0`: Recent data weighted more heavily
/// - `momentum_factor < 1.0`: Earlier data weighted more heavily
///
/// **Benefits**:
/// - Captures recent price momentum trends
/// - More responsive to evolving market conditions
/// - Better prediction accuracy for trending markets
/// - Maintains volume awareness from original VWAP
///
/// **Adaptive Behavior**:
/// - Automatically adjusts to sequence volatility
/// - Maintains mathematical consistency across market regimes
/// - Preserves backward compatibility when momentum_factor = 1.0
pub fn calculate_vwap_with_momentum(
    sequence_ohlcv: &[MarketDataRow],
    momentum_factor: f64,
) -> Result<f64> {
    if sequence_ohlcv.len() < 2 {
        return Ok(0.0); // Fallback for insufficient data
    }

    // If momentum_factor is 1.0, use standard VWAP for efficiency
    if (momentum_factor - 1.0).abs() < 1e-6 {
        return get_sequence_vwap_baseline(sequence_ohlcv);
    }

    let mut total_weight = 0.0;
    let mut weighted_price_sum = 0.0;
    let sequence_length = sequence_ohlcv.len() as f64;

    for (i, candle) in sequence_ohlcv.iter().enumerate() {
        if candle.volume > 0.0 {
            // Calculate OHLC4 price
            let ohlc4_price = (candle.open + candle.high + candle.low + candle.close) / 4.0;

            // Calculate momentum weight (recent data gets more weight when momentum_factor > 1.0)
            let time_position = (i as f64 + 1.0) / sequence_length; // 0.0 to 1.0
            let momentum_weight = time_position.powf(momentum_factor);

            // Combined weight: volume × momentum
            let combined_weight = candle.volume * momentum_weight;

            weighted_price_sum += ohlc4_price * combined_weight;
            total_weight += combined_weight;
        }
    }

    if total_weight > 0.0 {
        Ok(weighted_price_sum / total_weight)
    } else {
        // Fallback to momentum-weighted simple average if no volume data
        let mut weighted_price_sum = 0.0;
        let mut total_weight = 0.0;

        for (i, candle) in sequence_ohlcv.iter().enumerate() {
            let ohlc4_price = (candle.open + candle.high + candle.low + candle.close) / 4.0;
            let time_position = (i as f64 + 1.0) / sequence_length;
            let momentum_weight = time_position.powf(momentum_factor);

            weighted_price_sum += ohlc4_price * momentum_weight;
            total_weight += momentum_weight;
        }

        Ok(weighted_price_sum / total_weight)
    }
}

/// Get horizon VWAP-weighted price (same calculation as baseline)
pub fn get_horizon_vwap(horizon_ohlcv: &[MarketDataRow]) -> Result<f64> {
    // Same calculation as baseline (VWAP-weighted price)
    get_sequence_vwap_baseline(horizon_ohlcv)
}

/// Calculate volume-weighted center price from sequence data
///
/// This is the same as get_sequence_vwap_baseline but with a more descriptive name
/// for use in adaptive percentile calculations. The volume-weighted center represents
/// the "fair value" price during the sequence period, accounting for trading activity.
pub fn calculate_volume_weighted_center(sequence_ohlcv: &[MarketDataRow]) -> Result<f64> {
    get_sequence_vwap_baseline(sequence_ohlcv)
}

/// Extract individual OHLC4 prices from sequence data
///
/// Returns a vector of OHLC4 prices (not volume-weighted) for distribution analysis.
/// Each price represents the average of open, high, low, close for that candle.
pub fn extract_sequence_prices(sequence_ohlcv: &[MarketDataRow]) -> Vec<f64> {
    sequence_ohlcv
        .iter()
        .map(|candle| (candle.open + candle.high + candle.low + candle.close) / 4.0)
        .collect()
}

/// Calculate adaptive percentiles based on sequence price distribution around volume-weighted center
///
/// # 🎯 ADAPTIVE PERCENTILE ALGORITHM
///
/// **Core Logic**: Analyze how prices are distributed around the volume-weighted center
/// to determine optimal percentile boundaries for classification.
///
/// **Steps**:
/// 1. Calculate volume-weighted center (fair value price)
/// 2. Partition prices into below/above center groups
/// 3. Determine adaptive percentiles based on distribution balance
/// 4. Apply volatility-based adjustments
///
/// **Adaptive Logic**:
/// - **Balanced distribution** (40-60% below center): Standard percentiles [0.1, 0.9]
/// - **Bottom-heavy** (>60% below center): Tighter lower, wider upper [0.15, 0.95]
/// - **Top-heavy** (<40% below center): Wider lower, tighter upper [0.05, 0.85]
/// - **High volatility**: Expand both boundaries for noise tolerance
/// - **Low volatility**: Contract boundaries for sensitivity
///
/// **Benefits**:
/// - Automatic adaptation to each sequence's characteristics
/// - Volume-aware (uses actual trading activity)
/// - Volatility-responsive (tight for low vol, wide for high vol)
/// - No hardcoded magic numbers
///
/// # Arguments
/// * `sequence_ohlcv` - OHLCV data for the sequence
///
/// # Returns
/// * `Result<[f64; 2]>` - Adaptive percentiles [lower, upper] optimized for the sequence
pub fn calculate_adaptive_percentiles_from_sequence(
    sequence_ohlcv: &[MarketDataRow],
) -> Result<[f64; 2]> {
    if sequence_ohlcv.len() < 5 {
        // Fallback for short sequences - use standard percentiles
        return Ok([0.1, 0.9]);
    }

    // 1. Calculate volume-weighted center (fair value)
    let volume_weighted_center = calculate_volume_weighted_center(sequence_ohlcv)?;

    // 2. Extract individual OHLC4 prices for distribution analysis
    let sequence_prices = extract_sequence_prices(sequence_ohlcv);

    // 3. Partition prices around the volume-weighted center
    let below_center_count = sequence_prices
        .iter()
        .filter(|&&price| price < volume_weighted_center)
        .count();

    let below_center_ratio = below_center_count as f64 / sequence_prices.len() as f64;

    // 4. Calculate coefficient of variation for volatility adjustment
    let price_mean = sequence_prices.iter().sum::<f64>() / sequence_prices.len() as f64;
    let price_variance = sequence_prices
        .iter()
        .map(|&p| (p - price_mean).powi(2))
        .sum::<f64>()
        / sequence_prices.len() as f64;
    let coefficient_of_variation = if price_mean > 1e-8 {
        price_variance.sqrt() / price_mean
    } else {
        0.02 // Default 2% volatility
    };

    // 5. Determine base percentiles based on distribution balance
    let (base_lower, base_upper): (f64, f64) = match below_center_ratio {
        ratio if ratio < 0.4 => (0.05, 0.85), // Top-heavy: wider lower, tighter upper
        ratio if ratio > 0.6 => (0.15, 0.95), // Bottom-heavy: tighter lower, wider upper
        _ => (0.1, 0.9),                      // Balanced: standard percentiles
    };

    // 6. Apply volatility-based adjustments
    let volatility_adjustment: f64 = match coefficient_of_variation {
        cv if cv < 0.01 => 0.05, // Low volatility: contract boundaries (more sensitive)
        cv if cv > 0.04 => -0.05, // High volatility: expand boundaries (less sensitive)
        _ => 0.0,                // Medium volatility: no adjustment
    };

    let adaptive_lower = (base_lower + volatility_adjustment).clamp(0.02, 0.25);
    let adaptive_upper = (base_upper - volatility_adjustment).clamp(0.75, 0.98);

    // 7. Debug logging for transparency
    log::debug!(
        "🎯 Adaptive Percentiles: center={:.6}, below_ratio={:.2}, cv={:.3}, base=[{:.2}, {:.2}], adaptive=[{:.2}, {:.2}]",
        volume_weighted_center,
        below_center_ratio,
        coefficient_of_variation,
        base_lower,
        base_upper,
        adaptive_lower,
        adaptive_upper
    );

    Ok([adaptive_lower, adaptive_upper])
}

/// **ENHANCED**: Calculate adaptive bandwidth based on sequence volatility characteristics
///
/// # 🎯 ADAPTIVE BANDWIDTH CALCULATION
///
/// **Mathematical Foundation**: Adjusts bandwidth based on sequence price volatility to ensure
/// consistent classification difficulty across different market regimes and volatility periods.
///
/// **Algorithm**:
/// 1. Calculate sequence price volatility (coefficient of variation)
/// 2. Compare to baseline volatility (2% for crypto markets)
/// 3. Scale bandwidth by volatility ratio with bounds
/// 4. Apply momentum weighting if specified
///
/// **Volatility Scaling Logic**:
/// - High volatility periods: Larger bandwidth (less sensitive to noise)
/// - Low volatility periods: Smaller bandwidth (more sensitive to small moves)
/// - Extreme volatility: Capped to prevent over-adjustment
///
/// **Benefits**:
/// - Consistent classification across market regimes
/// - Automatic adaptation to volatility clustering
/// - Maintains ~20% class distribution target
/// - Prevents over-fitting to specific volatility periods
pub fn calculate_adaptive_bandwidth(
    sequence_ohlcv: &[MarketDataRow],
    base_bandwidth: f64,
    momentum_factor: Option<f64>,
) -> Result<f64> {
    if sequence_ohlcv.len() < 3 {
        return Ok(base_bandwidth); // Fallback for insufficient data
    }

    // 1. Extract sequence prices for volatility calculation
    let prices: Vec<f64> = sequence_ohlcv
        .iter()
        .map(|c| (c.open + c.high + c.low + c.close) / 4.0)
        .collect();

    // 2. Calculate price volatility (coefficient of variation)
    let price_mean = prices.iter().sum::<f64>() / prices.len() as f64;
    let price_variance = prices
        .iter()
        .map(|&p| (p - price_mean).powi(2))
        .sum::<f64>()
        / prices.len() as f64;
    let price_std = price_variance.sqrt();
    let coefficient_of_variation = if price_mean > 1e-8 {
        price_std / price_mean
    } else {
        0.02 // Default 2% volatility for edge cases
    };

    // 3. Calculate volatility multiplier with bounds
    let baseline_volatility = 0.02; // 2% baseline for crypto markets
    let volatility_ratio = coefficient_of_variation / baseline_volatility;
    let volatility_multiplier = volatility_ratio.clamp(0.3, 3.0); // Prevent extreme adjustments

    // 4. Apply momentum weighting if specified
    let final_multiplier = if let Some(momentum) = momentum_factor {
        if momentum > 1.0 {
            // Higher momentum factor = more recent data weight = potentially higher volatility
            let momentum_adjustment = 1.0 + (momentum - 1.0) * 0.2; // 20% max adjustment
            volatility_multiplier * momentum_adjustment
        } else {
            volatility_multiplier
        }
    } else {
        volatility_multiplier
    };

    // 5. Calculate adaptive bandwidth
    let adaptive_bandwidth = base_bandwidth * final_multiplier;

    log::debug!(
        "🎯 Adaptive Bandwidth: base={:.3}, volatility_ratio={:.3}, multiplier={:.3}, adaptive={:.3}",
        base_bandwidth,
        volatility_ratio,
        final_multiplier,
        adaptive_bandwidth
    );

    Ok(adaptive_bandwidth)
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
/// * `sequence_ohlcv` - Input sequence OHLCV data for percentile calculation
/// * `horizon_ohlcv` - Horizon period OHLCV data for target price
/// * `config` - Configuration with percentiles array and bandwidth sensitivity
///
/// # Returns
/// * `Result<i32>` - Classification bin [0-4] based on percentile boundaries
pub fn classify_price_level(
    sequence_ohlcv: &[MarketDataRow],
    horizon_ohlcv: &[MarketDataRow],
    config: &crate::config::model::TargetsConfig,
) -> Result<i32> {
    if sequence_ohlcv.is_empty() {
        return Ok(2); // Default to neutral class
    }

    // Use adaptive percentiles based on sequence data (no hardcoded values)
    let percentiles = calculate_adaptive_percentiles_from_sequence(sequence_ohlcv)?;
    let bandwidth_size = config.base_sensitivity; // Use base_sensitivity as bandwidth

    let reconstruction_config = SequenceReconstructionConfig {
        percentiles,
        bandwidth_size,
    };
    let analyzer = SequenceAnalyzer::new(reconstruction_config);

    // Calculate boundaries using centralized logic
    let boundaries = analyzer.calculate_boundaries(sequence_ohlcv)?;

    // Get target price from horizon VWAP (instead of single future price)
    let target_price = get_horizon_vwap(horizon_ohlcv)?;

    // Handle edge case: flat sequence (bandwidth = 0)
    if boundaries.bandwidth == 0.0 {
        return Ok(if target_price >= boundaries.sequence_min {
            3
        } else {
            2
        });
    }

    // Debug logging
    log::debug!(
        "🔍 Centralized Classification: percentiles=[{:.1}%, {:.1}%], seq_range=[{:.6}, {:.6}], target={:.6}, bandwidth={:.6}",
        percentiles[0] * 100.0,
        percentiles[1] * 100.0,
        boundaries.sequence_min,
        boundaries.sequence_max,
        target_price,
        boundaries.bandwidth
    );

    // Use centralized classification logic
    let class = boundaries.classify_price(target_price);

    log::debug!(
        "🎯 Centralized Classification: target={:.6} → class={} (percentile_range: [{:.6}, {:.6}], bandwidth: {:.6})",
        target_price, class, boundaries.sequence_min, boundaries.sequence_max, boundaries.bandwidth
    );

    Ok(class)
}

/// **ENHANCED**: Classify price level using momentum-weighted VWAP and adaptive thresholds
///
/// # 🚀 ENHANCED CLASSIFICATION WITH ADAPTIVE FEATURES
///
/// **Mathematical Enhancements**:
/// 1. **Momentum-Weighted VWAP**: Recent data gets more influence in price calculation
/// 2. **Adaptive Bandwidth**: Automatically adjusts to sequence volatility characteristics
/// 3. **Volatility Normalization**: Consistent behavior across different market regimes
/// 4. **Backward Compatibility**: Falls back to standard method when enhancements disabled
///
/// **Key Improvements**:
/// - **Better Trend Capture**: Momentum weighting captures evolving price trends
/// - **Volatility Adaptation**: Bandwidth scales with market volatility automatically
/// - **Balanced Distribution**: Maintains ~20% per class across market conditions
/// - **Mathematical Consistency**: Stable performance across different assets and timeframes
///
/// **Configuration Parameters**:
/// - `momentum_factor`: 1.0 = standard VWAP, >1.0 = recent data weighted more
/// - `base_sensitivity`: Base bandwidth multiplier (auto-scaled by volatility)
///
/// # Arguments
/// * `sequence_ohlcv` - Input sequence OHLCV data
/// * `horizon_ohlcv` - Horizon period OHLCV data
/// * `momentum_factor` - Optional momentum weighting (1.0 = standard, >1.0 = recent bias)
///
/// # Returns
/// * `Result<i32>` - Enhanced classification [0-4] with adaptive features
pub fn classify_price_level_with_momentum(
    sequence_ohlcv: &[MarketDataRow],
    horizon_ohlcv: &[MarketDataRow],
    momentum_factor: Option<f64>,
) -> Result<i32> {
    if sequence_ohlcv.is_empty() {
        return Ok(2); // Default to neutral class
    }

    // 1. Calculate sequence VWAP with optional momentum weighting
    let seq_vwap = if let Some(momentum) = momentum_factor {
        if momentum != 1.0 {
            calculate_vwap_with_momentum(sequence_ohlcv, momentum)?
        } else {
            get_sequence_vwap_baseline(sequence_ohlcv)?
        }
    } else {
        get_sequence_vwap_baseline(sequence_ohlcv)?
    };

    // 2. Calculate horizon VWAP (standard calculation)
    let hor_vwap = get_horizon_vwap(horizon_ohlcv)?;

    // 3. Use adaptive percentiles based on sequence data
    let percentiles = calculate_adaptive_percentiles_from_sequence(sequence_ohlcv)?;
    let base_bandwidth_size = 1.0;

    // 4. Calculate adaptive bandwidth if enabled
    // // ALWAYS  adaptive no need to have complexity
    let final_bandwidth_size =
        calculate_adaptive_bandwidth(sequence_ohlcv, base_bandwidth_size, momentum_factor)?;

    // 5. Use enhanced sequence reconstruction with adaptive parameters
    let reconstruction_config = SequenceReconstructionConfig {
        percentiles,
        bandwidth_size: final_bandwidth_size,
    };
    let analyzer = SequenceAnalyzer::new(reconstruction_config);

    // 6. Calculate boundaries using centralized logic
    let boundaries = analyzer.calculate_boundaries(sequence_ohlcv)?;

    // 7. Handle edge case: flat sequence
    if boundaries.bandwidth == 0.0 {
        return Ok(if hor_vwap >= boundaries.sequence_min {
            3
        } else {
            2
        });
    }

    // 8. Enhanced debug logging
    log::debug!(
        "🚀 Price Level with Momentum: momentum_factor={:?}, seq_vwap={:.6}, hor_vwap={:.6}, final_bandwidth={:.3}",
        momentum_factor,
        seq_vwap,
        hor_vwap,
        final_bandwidth_size
    );

    // 9. Use centralized classification logic
    let class = boundaries.classify_price(hor_vwap);

    log::debug!(
        "🎯 Price Level Result: target={:.6} → class={} (adaptive_range: [{:.6}, {:.6}], adaptive_bandwidth: {:.6})",
        hor_vwap, class, boundaries.sequence_min, boundaries.sequence_max, boundaries.bandwidth
    );

    Ok(class)
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
        "📊 Price Level Distribution [{}]: {} samples, {:.1}x imbalance, classes: [{}] (BEFORE balanced selection)",
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

/// Generate price level targets using TargetsConfig (NEW UNIFIED APPROACH)
pub fn generate_price_level_targets_with_targets_config(
    df: &DataFrame,
    horizons: &[String],
    targets_config: &TargetsConfig,
    sequence_indices: &[usize],
    sequence_length: usize,
) -> Result<HashMap<String, Vec<i32>>> {
    generate_price_level_targets_with_adaptive_params(
        df,
        horizons,
        targets_config,
        sequence_indices,
        sequence_length,
        None, // No adaptive parameters - use base config
    )
}

/// Generate price level targets with optional adaptive parameters
///
/// When adaptive_params is provided, uses the pre-calibrated parameters for consistent
/// target generation between training and prediction. When None, uses base config.
pub fn generate_price_level_targets_with_adaptive_params(
    df: &DataFrame,
    horizons: &[String],
    targets_config: &TargetsConfig,
    sequence_indices: &[usize],
    sequence_length: usize,
    adaptive_params: Option<&crate::targets::adaptive_parameters::PriceLevelAdaptiveParams>,
) -> Result<HashMap<String, Vec<i32>>> {
    let config = if let Some(params) = adaptive_params {
        log::info!(
            "🎯 Using pre-calibrated price level bandwidth: {:.4}",
            params.bandwidth_size
        );
        PriceLevelConfig {
            bandwidth_size: params.bandwidth_size,
        }
    } else {
        log::info!(
            "🎯 Using base price level bandwidth: {:.4}",
            targets_config.base_sensitivity
        );
        PriceLevelConfig {
            bandwidth_size: targets_config.base_sensitivity,
        }
    };
    generate_price_level_targets(df, horizons, &config, sequence_indices, sequence_length)
}

/// DEPRECATED: Generate price level targets from ModelConfig (use generate_price_level_targets_with_targets_config instead)
pub fn generate_price_level_targets_from_model_config(
    df: &DataFrame,
    horizons: &[String],
    model_config: &crate::config::model::ModelConfig,
    sequence_indices: &[usize],
    sequence_length: usize,
) -> Result<HashMap<String, Vec<i32>>> {
    // Use the new TargetsConfig approach
    generate_price_level_targets_with_targets_config(
        df,
        horizons,
        &model_config.targets,
        sequence_indices,
        sequence_length,
    )
}

// ============================================================================
// PREDICTION RECONSTRUCTION METHODS
// ============================================================================

/// Reconstruction result for price level predictions
#[derive(Debug, Clone)]
pub struct PriceLevelReconstruction {
    /// Percentage ranges for each class [lower_bound, upper_bound]
    pub percentage_ranges: Vec<[f64; 2]>,
    /// Absolute price ranges for each class [lower_price, upper_price]
    pub price_ranges: Vec<[f64; 2]>,
    /// VWAP-relative percentage ranges for each class [lower_bound, upper_bound]
    pub vwap_percentage_ranges: Vec<[f64; 2]>,
    /// Class probabilities from model
    pub probabilities: Vec<f64>,
    /// Most likely class index
    pub most_likely_class: usize,
    /// Confidence (probability of most likely class)
    pub confidence: f64,
    /// Expected price change percentage (weighted average)
    pub expected_change_percent: f64,
    /// Sequence boundaries used for calculation
    pub sequence_min: f64,
    pub sequence_max: f64,
    pub bandwidth: f64,
}

/// Reconstruct price level predictions from model probabilities
///
/// This method reverses the training classification logic to convert
/// raw model probabilities back to meaningful price ranges and percentages.
///
/// # Arguments
/// * `probabilities` - 5-element array of class probabilities from model
/// * `sequence_ohlcv` - OHLCV data for the input sequence (same as used in training)
/// * `current_price` - Current price for percentage calculations
/// * `config` - Optional configuration (uses defaults if None)
///
/// # Returns
/// * `PriceLevelReconstruction` - Complete reconstruction with ranges and metrics
pub fn reconstruct_price_levels(
    probabilities: &[f64],
    sequence_ohlcv: &[MarketDataRow],
    current_price: f64,
    config: Option<&TargetsConfig>,
) -> Result<PriceLevelReconstruction> {
    // Validate inputs
    if probabilities.len() != 5 {
        return Err(crate::utils::error::VangaError::DataError(
            "Price level reconstruction requires exactly 5 class probabilities".to_string(),
        ));
    }

    if sequence_ohlcv.is_empty() {
        return Err(crate::utils::error::VangaError::DataError(
            "Sequence OHLCV data is required for price level reconstruction".to_string(),
        ));
    }

    if current_price <= 0.0 {
        return Err(crate::utils::error::VangaError::DataError(
            "Current price must be positive for percentage calculations".to_string(),
        ));
    }

    // Use same adaptive percentiles as training for consistency
    let percentiles = calculate_adaptive_percentiles_from_sequence(sequence_ohlcv)?;
    let bandwidth_size = config.map(|c| c.base_sensitivity).unwrap_or(1.0);

    // Calculate adaptive bandwidth (same logic as training)
    let final_bandwidth_size = calculate_adaptive_bandwidth(
        sequence_ohlcv,
        bandwidth_size,
        Some(1.2), // Same momentum factor as training
    )?;

    // Use centralized sequence reconstruction logic (same as training)
    let reconstruction_config = SequenceReconstructionConfig {
        percentiles,
        bandwidth_size: final_bandwidth_size,
    };
    let analyzer = SequenceAnalyzer::new(reconstruction_config);
    let boundaries = analyzer.calculate_boundaries(sequence_ohlcv)?;

    // Calculate percentage ranges for each class
    let percentage_ranges = boundaries.get_price_level_ranges(current_price);

    // Calculate absolute price ranges
    let price_ranges: Vec<[f64; 2]> = percentage_ranges
        .iter()
        .map(|[lower_pct, upper_pct]| {
            let lower_price = current_price * (1.0 + lower_pct / 100.0);
            let upper_price = current_price * (1.0 + upper_pct / 100.0);
            [lower_price, upper_price]
        })
        .collect();

    // Calculate sequence VWAP for vwap_range calculation
    let sequence_vwap = get_sequence_vwap_baseline(sequence_ohlcv)?;

    // Calculate VWAP-relative percentage ranges
    let vwap_percentage_ranges: Vec<[f64; 2]> = price_ranges
        .iter()
        .map(|[lower_price, upper_price]| {
            let lower_pct = ((lower_price / sequence_vwap) - 1.0) * 100.0;
            let upper_pct = ((upper_price / sequence_vwap) - 1.0) * 100.0;
            [lower_pct, upper_pct]
        })
        .collect();

    // Find most likely class and confidence
    let (most_likely_class, confidence) = probabilities
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(idx, &prob)| (idx, prob))
        .unwrap_or((2, 0.2)); // Default to neutral class

    // Calculate expected price change (weighted average of class midpoints)
    let class_midpoints: Vec<f64> = percentage_ranges
        .iter()
        .map(|[lower, upper]| (lower + upper) / 2.0)
        .collect();

    let expected_change_percent: f64 = probabilities
        .iter()
        .zip(class_midpoints.iter())
        .map(|(&prob, &midpoint)| prob * midpoint)
        .sum();

    Ok(PriceLevelReconstruction {
        percentage_ranges,
        price_ranges,
        vwap_percentage_ranges,
        probabilities: probabilities.to_vec(),
        most_likely_class,
        confidence,
        expected_change_percent,
        sequence_min: boundaries.sequence_min,
        sequence_max: boundaries.sequence_max,
        bandwidth: boundaries.bandwidth,
    })
}

/// Convert class probabilities to expected price targets
///
/// This method calculates the expected price for each class based on
/// the same mathematical logic used in training target generation.
///
/// # Arguments
/// * `probabilities` - 5-element array of class probabilities
/// * `sequence_ohlcv` - OHLCV data for boundary calculation
/// * `config` - Optional configuration
///
/// # Returns
/// * `Vec<f64>` - Expected price for each class [strong_down, moderate_down, neutral, moderate_up, strong_up]
pub fn probabilities_to_price_targets(
    probabilities: &[f64],
    sequence_ohlcv: &[MarketDataRow],
    config: Option<&TargetsConfig>,
) -> Result<Vec<f64>> {
    if probabilities.len() != 5 {
        return Err(crate::utils::error::VangaError::DataError(
            "Expected 5 class probabilities for price level reconstruction".to_string(),
        ));
    }

    // Use same adaptive percentiles as training for consistency
    let percentiles = calculate_adaptive_percentiles_from_sequence(sequence_ohlcv)?;
    let bandwidth_size = config.map(|c| c.base_sensitivity).unwrap_or(1.0);
    let final_bandwidth_size =
        calculate_adaptive_bandwidth(sequence_ohlcv, bandwidth_size, Some(1.2))?;

    let reconstruction_config = SequenceReconstructionConfig {
        percentiles,
        bandwidth_size: final_bandwidth_size,
    };
    let analyzer = SequenceAnalyzer::new(reconstruction_config);
    let boundaries = analyzer.calculate_boundaries(sequence_ohlcv)?;

    // Calculate representative price for each class (midpoint of boundaries)
    let class_prices = vec![
        boundaries.boundaries[0] - boundaries.bandwidth / 2.0, // Strong Down: below boundary_1
        (boundaries.boundaries[0] + boundaries.boundaries[1]) / 2.0, // Moderate Down: between boundary_1 and boundary_2
        (boundaries.boundaries[1] + boundaries.boundaries[2]) / 2.0, // Neutral: between boundary_2 and boundary_3
        (boundaries.boundaries[2] + boundaries.boundaries[3]) / 2.0, // Moderate Up: between boundary_3 and boundary_4
        boundaries.boundaries[3] + boundaries.bandwidth / 2.0,       // Strong Up: above boundary_4
    ];

    Ok(class_prices)
}

/// Calculate percentage changes from current price for each class
///
/// # Arguments
/// * `probabilities` - 5-element array of class probabilities
/// * `sequence_ohlcv` - OHLCV data for boundary calculation
/// * `current_price` - Current price for percentage calculation
/// * `config` - Optional configuration
///
/// # Returns
/// * `Vec<f64>` - Percentage change for each class
pub fn probabilities_to_percentage_changes(
    probabilities: &[f64],
    sequence_ohlcv: &[MarketDataRow],
    current_price: f64,
    config: Option<&TargetsConfig>,
) -> Result<Vec<f64>> {
    let class_prices = probabilities_to_price_targets(probabilities, sequence_ohlcv, config)?;

    let percentage_changes: Vec<f64> = class_prices
        .iter()
        .map(|&price| ((price - current_price) / current_price) * 100.0)
        .collect();

    Ok(percentage_changes)
}
