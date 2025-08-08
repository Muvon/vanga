//! Adaptive Target Parameters System
//!
//! This module implements automatic parameter optimization for all target types
//! to achieve balanced class distributions across different market conditions.
//!
//! ## Core Philosophy: "Sweet Spot" Optimization
//!
//! Instead of using fixed parameters, this system:
//! 1. **Analyzes actual market data** to understand distribution patterns
//! 2. **Finds optimal parameters** that maximize balanced class distribution
//! 3. **Stores adaptive parameters** with the model for consistent prediction
//! 4. **Ensures reproducibility** between training and inference
//!
//! ## Key Benefits
//!
//! - **Automatic Optimization**: No manual parameter tuning required
//! - **Market Adaptive**: Parameters adjust to specific market characteristics
//! - **Balanced Distribution**: Targets ~20% per class across all conditions
//! - **Prediction Consistency**: Same parameters used in training and inference
//! - **Symbol Agnostic**: Works across different trading pairs and timeframes

use crate::config::model::TargetsConfig;
use crate::data::structures::MarketDataRow;
use crate::targets::volatility::{calculate_atr_distribution_stats, AtrDistributionStats};
use crate::utils::error::Result;
use polars::frame::DataFrame;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Comprehensive adaptive parameters for all target types
///
/// This struct stores the calibrated parameters found during training
/// that produce optimal balanced class distributions. These parameters
/// are saved with the model and reused during prediction to ensure
/// consistent target generation.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AdaptiveTargetParameters {
    /// Direction target parameters (momentum-based)
    pub direction: DirectionAdaptiveParams,

    /// Price level target parameters (VWAP-weighted)
    pub price_levels: PriceLevelAdaptiveParams,

    /// Volatility target parameters (ATR distribution-based)
    pub volatility: VolatilityAdaptiveParams,

    /// Sentiment target parameters (candle body analysis)
    pub sentiment: SentimentAdaptiveParams,

    /// Volume target parameters (logarithmic volume analysis)
    pub volume: VolumeAdaptiveParams,

    /// Calibration metadata
    pub calibration_info: CalibrationMetadata,
}

/// Adaptive parameters for direction targets (momentum-based classification)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DirectionAdaptiveParams {
    /// Calibrated base sensitivity for momentum change thresholds
    pub base_sensitivity: f64,

    /// Extreme multiplier for DUMP/PUMP vs DOWN/UP boundaries
    pub extreme_multiplier: f64,

    /// Momentum weighting factor for recent data emphasis
    pub momentum_weighting: f64,

    /// Trend consistency normalization factor
    pub trend_consistency_factor: f64,

    /// Distribution balance achieved with these parameters
    pub achieved_balance: ClassDistributionBalance,
}

impl Default for DirectionAdaptiveParams {
    fn default() -> Self {
        Self {
            base_sensitivity: 0.02,
            extreme_multiplier: 2.0,
            momentum_weighting: 1.2,
            trend_consistency_factor: 1.0,
            achieved_balance: ClassDistributionBalance::default(),
        }
    }
}

/// Adaptive parameters for price level targets (VWAP-weighted classification)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PriceLevelAdaptiveParams {
    /// Calibrated bandwidth size for breakout sensitivity
    pub bandwidth_size: f64,

    /// Adaptive percentiles for range boundaries [lower, upper]
    pub adaptive_percentiles: [f64; 2],

    /// Momentum factor for VWAP weighting
    pub momentum_factor: f64,

    /// Volatility adjustment factor for bandwidth scaling
    pub volatility_adjustment: f64,

    /// Distribution balance achieved with these parameters
    pub achieved_balance: ClassDistributionBalance,
}

impl Default for PriceLevelAdaptiveParams {
    fn default() -> Self {
        Self {
            bandwidth_size: 1.0,
            adaptive_percentiles: [0.1, 0.9],
            momentum_factor: 1.2,
            volatility_adjustment: 1.0,
            achieved_balance: ClassDistributionBalance::default(),
        }
    }
}

/// Adaptive parameters for volatility targets (ATR distribution-based classification)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VolatilityAdaptiveParams {
    /// Calibrated bandwidth for logarithmic ratio thresholds
    pub bandwidth_size: f64,

    /// Extreme multiplier for VeryLow/VeryHigh vs Low/High boundaries
    pub extreme_multiplier: f64,

    /// ATR distribution characteristics
    pub atr_distribution_stats: AtrDistributionStats,

    /// Coefficient of variation adjustment factor
    pub cv_adjustment_factor: f64,

    /// Distribution balance achieved with these parameters
    pub achieved_balance: ClassDistributionBalance,
}

impl Default for VolatilityAdaptiveParams {
    fn default() -> Self {
        Self {
            bandwidth_size: 0.4,
            extreme_multiplier: 2.0,
            atr_distribution_stats: AtrDistributionStats::default(),
            cv_adjustment_factor: 1.0,
            achieved_balance: ClassDistributionBalance::default(),
        }
    }
}

/// Sentiment adaptive parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SentimentAdaptiveParams {
    /// Body analysis sensitivity for sentiment calculation
    pub body_sensitivity: f64,

    /// Volume confirmation weight in sentiment score
    pub volume_weight: f64,

    /// Consistency factor for adaptive threshold scaling
    pub consistency_factor: f64,

    /// Distribution balance achieved with these parameters
    pub achieved_balance: ClassDistributionBalance,
}

impl Default for SentimentAdaptiveParams {
    fn default() -> Self {
        Self {
            body_sensitivity: 1.0,
            volume_weight: 0.3,
            consistency_factor: 1.0,
            achieved_balance: ClassDistributionBalance::default(),
        }
    }
}

/// Volume adaptive parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VolumeAdaptiveParams {
    /// Volume bandwidth size for threshold calculation
    pub bandwidth_size: f64,

    /// Extreme threshold multiplier
    pub extreme_multiplier: f64,

    /// Volume smoothing periods for noise reduction
    pub smoothing_periods: usize,

    /// Distribution balance achieved with these parameters
    pub achieved_balance: ClassDistributionBalance,
}

impl Default for VolumeAdaptiveParams {
    fn default() -> Self {
        Self {
            bandwidth_size: 0.4,
            extreme_multiplier: 2.0,
            smoothing_periods: 3,
            achieved_balance: ClassDistributionBalance::default(),
        }
    }
}

/// Class distribution balance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassDistributionBalance {
    /// Percentage of samples in each class [0, 1, 2, 3, 4]
    pub class_percentages: [f64; 5],

    /// Imbalance ratio (max_class_size / min_class_size)
    pub imbalance_ratio: f64,

    /// Total valid samples used for calibration
    pub total_samples: usize,

    /// Standard deviation of class percentages (lower = more balanced)
    pub balance_score: f64,
}

impl Default for ClassDistributionBalance {
    fn default() -> Self {
        Self {
            class_percentages: [20.0, 20.0, 20.0, 20.0, 20.0], // Perfect balance
            imbalance_ratio: 1.0,
            total_samples: 0,
            balance_score: 0.0,
        }
    }
}

/// Calibration metadata for tracking and debugging
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationMetadata {
    /// Data characteristics used for calibration
    pub data_length: usize,
    pub sequence_length: usize,
    pub horizon_steps: usize,

    /// Calibration process information
    pub calibration_samples: usize,
    pub calibration_iterations: usize,
    pub optimization_time_ms: u64,

    /// Target balance configuration used
    pub target_balance: f64,

    /// Overall calibration success metrics
    pub overall_balance_score: f64,
    pub calibration_success: bool,
}

impl Default for CalibrationMetadata {
    fn default() -> Self {
        Self {
            data_length: 0,
            sequence_length: 60,
            horizon_steps: 24,
            calibration_samples: 0,
            calibration_iterations: 0,
            optimization_time_ms: 0,
            target_balance: 0.2,
            overall_balance_score: 0.0,
            calibration_success: false,
        }
    }
}

/// Adaptive parameter calibration orchestrator
pub struct AdaptiveParameterCalibrator {
    /// Base configuration for calibration
    base_config: TargetsConfig,

    /// Optimization settings
    max_iterations: usize,
    tolerance: f64,
    target_balance: f64,
}

/// Parameters for price level evaluation
#[derive(Debug, Clone)]
struct PriceLevelEvaluationParams {
    bandwidth_size: f64,
    percentiles: [f64; 2],
    momentum_factor: f64,
}

impl AdaptiveParameterCalibrator {
    /// Create new calibrator with configuration
    pub fn new(base_config: TargetsConfig) -> Self {
        Self {
            target_balance: base_config.balance_target,
            base_config,
            max_iterations: 50,
            tolerance: 0.01, // 1% tolerance for balance optimization
        }
    }

    /// Get the base configuration
    pub fn get_base_config(&self) -> &TargetsConfig {
        &self.base_config
    }

    /// Calibrate price level parameters specifically
    pub async fn calibrate_price_levels(
        &self,
        ohlcv_data: &[MarketDataRow],
        sequence_length: usize,
        horizon_steps: usize,
        sequence_indices: &[usize],
    ) -> Result<PriceLevelAdaptiveParams> {
        log::debug!("🎯 Starting specialized price level calibration...");

        // Grid search for optimal price level parameters
        let bandwidth_candidates = vec![0.5, 0.7, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0];
        let percentile_candidates = vec![
            [0.05, 0.95],
            [0.1, 0.9],
            [0.15, 0.85],
            [0.2, 0.8],
            [0.25, 0.75],
        ];
        let momentum_factor_candidates = vec![1.0, 1.1, 1.2, 1.3];

        let mut best_params = PriceLevelAdaptiveParams::default();
        let mut best_balance_score = f64::INFINITY;

        for &bandwidth in &bandwidth_candidates {
            for &percentiles in &percentile_candidates {
                for &momentum_factor in &momentum_factor_candidates {
                    let eval_params = PriceLevelEvaluationParams {
                        bandwidth_size: bandwidth,
                        percentiles,
                        momentum_factor,
                    };

                    let balance = self
                        .evaluate_price_level_parameters(
                            ohlcv_data,
                            sequence_indices,
                            sequence_length,
                            horizon_steps,
                            &eval_params,
                        )
                        .await?;

                    // MIN-CLASS OPTIMIZATION: Prioritize parameters that maximize minimum class representation
                    let min_class_ratio = balance
                        .class_percentages
                        .iter()
                        .min_by(|a, b| a.partial_cmp(b).unwrap())
                        .unwrap()
                        / 100.0;
                    let min_class_threshold = 0.15; // 15% minimum per class (vs 20% ideal)

                    // Heavy penalty if any class falls below threshold
                    let min_class_penalty = if min_class_ratio < min_class_threshold {
                        (min_class_threshold - min_class_ratio) * 50.0 // 50x penalty for under-representation
                    } else {
                        0.0
                    };

                    let score = balance.balance_score
                        + (balance.imbalance_ratio - 1.0) * 0.1
                        + min_class_penalty;

                    if score < best_balance_score {
                        best_balance_score = score;
                        best_params = PriceLevelAdaptiveParams {
                            bandwidth_size: bandwidth,
                            adaptive_percentiles: percentiles,
                            momentum_factor,
                            volatility_adjustment: 1.0, // Default adjustment
                            achieved_balance: balance,
                        };
                    }
                }
            }
        }

        let min_class_ratio = best_params
            .achieved_balance
            .class_percentages
            .iter()
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap()
            / 100.0;

        log::info!(
            "✅ Price level calibration complete: bandwidth={:.3}, balance_score={:.4}, min_class={:.1}%",
            best_params.bandwidth_size,
            best_params.achieved_balance.balance_score,
            min_class_ratio * 100.0
        );

        Ok(best_params)
    }

    /// Calibrate volatility parameters specifically
    pub async fn calibrate_volatility(
        &self,
        ohlcv_data: &[MarketDataRow],
        sequence_length: usize,
        horizon_steps: usize,
        sequence_indices: &[usize],
    ) -> Result<VolatilityAdaptiveParams> {
        log::debug!("🎯 Starting specialized volatility calibration...");

        // First calculate ATR values using the same logic as calibrate_volatility_parameters
        use crate::targets::volatility::get_sequence_atr_baseline;

        let mut sequence_atr_values = Vec::new();
        let mut horizon_atr_values = Vec::new();

        for &seq_idx in sequence_indices
            .iter()
            .take(sequence_indices.len().min(1000))
        {
            let sequence_end_idx = seq_idx + sequence_length;
            let target_end_idx = sequence_end_idx + horizon_steps;

            if target_end_idx <= ohlcv_data.len() {
                let sequence_candles = &ohlcv_data[seq_idx..sequence_end_idx];
                let horizon_candles = &ohlcv_data[sequence_end_idx..target_end_idx];

                if sequence_candles.len() >= 2 && horizon_candles.len() >= 2 {
                    if let (Ok(seq_atr), Ok(hor_atr)) = (
                        get_sequence_atr_baseline(sequence_candles),
                        get_sequence_atr_baseline(horizon_candles),
                    ) {
                        if seq_atr > 0.0 && hor_atr > 0.0 {
                            sequence_atr_values.push(seq_atr);
                            horizon_atr_values.push(hor_atr);
                        }
                    }
                }
            }
        }

        if sequence_atr_values.is_empty() {
            log::warn!("No valid ATR values found, using defaults");
            return Ok(VolatilityAdaptiveParams::default());
        }

        // Grid search for optimal volatility parameters
        let bandwidth_candidates = vec![0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0];
        let extreme_multiplier_candidates = vec![1.5, 2.0, 2.5, 3.0];

        let mut best_params = VolatilityAdaptiveParams::default();
        let mut best_balance_score = f64::INFINITY;

        for &bandwidth in &bandwidth_candidates {
            for &extreme_multiplier in &extreme_multiplier_candidates {
                let balance = self.evaluate_volatility_parameters(
                    &sequence_atr_values,
                    &horizon_atr_values,
                    bandwidth,
                    extreme_multiplier,
                )?;

                // MIN-CLASS OPTIMIZATION: Prioritize parameters that maximize minimum class representation
                let min_class_ratio = balance
                    .class_percentages
                    .iter()
                    .min_by(|a, b| a.partial_cmp(b).unwrap())
                    .unwrap()
                    / 100.0;
                let min_class_threshold = 0.15; // 15% minimum per class (vs 20% ideal)

                // Heavy penalty if any class falls below threshold
                let min_class_penalty = if min_class_ratio < min_class_threshold {
                    (min_class_threshold - min_class_ratio) * 50.0 // 50x penalty for under-representation
                } else {
                    0.0
                };

                let score = balance.balance_score
                    + (balance.imbalance_ratio - 1.0) * 0.1
                    + min_class_penalty;

                if score < best_balance_score {
                    best_balance_score = score;
                    best_params = VolatilityAdaptiveParams {
                        bandwidth_size: bandwidth,
                        extreme_multiplier,
                        atr_distribution_stats: AtrDistributionStats::default(),
                        cv_adjustment_factor: 1.0,
                        achieved_balance: balance,
                    };
                }
            }
        }

        let min_class_ratio = best_params
            .achieved_balance
            .class_percentages
            .iter()
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap()
            / 100.0;

        log::info!(
            "✅ Volatility calibration complete: bandwidth={:.3}, balance_score={:.4}, min_class={:.1}%",
            best_params.bandwidth_size,
            best_params.achieved_balance.balance_score,
            min_class_ratio * 100.0
        );

        Ok(best_params)
    }

    /// Calibrate adaptive parameters for all target types
    ///
    /// This is the main orchestration method that finds optimal parameters
    /// for all three target types simultaneously, ensuring balanced
    /// class distributions across the entire multi-target system.
    ///
    /// ## Algorithm
    /// 1. **Sample Data Analysis**: Analyze market data characteristics
    /// 2. **Individual Calibration**: Optimize each target type separately
    /// 3. **Cross-Target Validation**: Ensure parameters work well together
    /// 4. **Balance Optimization**: Fine-tune for overall system balance
    /// 5. **Validation**: Verify calibration quality and consistency
    ///
    /// ## Parameters
    /// - `ohlcv_data`: Market data for calibration analysis
    /// - `sequence_length`: Length of input sequences
    /// - `horizon_steps`: Prediction horizon length
    /// - `sequence_indices`: Sequence positions for analysis
    ///
    /// ## Returns
    /// Fully calibrated adaptive parameters optimized for balanced distribution
    pub async fn calibrate_all_targets(
        &self,
        ohlcv_data: &[MarketDataRow],
        sequence_length: usize,
        horizon_steps: usize,
        sequence_indices: &[usize],
    ) -> Result<AdaptiveTargetParameters> {
        let start_time = std::time::Instant::now();

        log::info!(
            "🎯 Starting adaptive parameter calibration for {} sequences",
            sequence_indices.len()
        );

        // Step 1: Calibrate direction parameters (momentum-based)
        log::info!("📊 Calibrating direction parameters (momentum-based)...");
        let direction_params = self
            .calibrate_direction_parameters(
                ohlcv_data,
                sequence_length,
                horizon_steps,
                sequence_indices,
            )
            .await?;

        // Step 2: Calibrate price level parameters (VWAP-weighted)
        log::info!("📊 Calibrating price level parameters (VWAP-weighted)...");
        let price_level_params = self
            .calibrate_price_level_parameters(
                ohlcv_data,
                sequence_length,
                horizon_steps,
                sequence_indices,
            )
            .await?;

        // Step 3: Calibrate volatility parameters (ATR distribution-based)
        log::info!("📊 Calibrating volatility parameters (ATR distribution-based)...");
        let volatility_params = self
            .calibrate_volatility_parameters(
                ohlcv_data,
                sequence_length,
                horizon_steps,
                sequence_indices,
            )
            .await?;

        // Step 4: Create calibration metadata
        let calibration_time = start_time.elapsed().as_millis() as u64;
        let overall_balance_score = (direction_params.achieved_balance.balance_score
            + price_level_params.achieved_balance.balance_score
            + volatility_params.achieved_balance.balance_score)
            / 3.0;

        let calibration_info = CalibrationMetadata {
            data_length: ohlcv_data.len(),
            sequence_length,
            horizon_steps,
            calibration_samples: sequence_indices.len(),
            calibration_iterations: self.max_iterations,
            optimization_time_ms: calibration_time,
            target_balance: self.target_balance,
            overall_balance_score,
            calibration_success: overall_balance_score < self.tolerance * 3.0, // 3x tolerance for combined system
        };

        let adaptive_params = AdaptiveTargetParameters {
            direction: direction_params,
            price_levels: price_level_params,
            volatility: volatility_params,
            sentiment: SentimentAdaptiveParams::default(), // TODO: Add calibration
            volume: VolumeAdaptiveParams::default(),       // TODO: Add calibration
            calibration_info,
        };

        // Step 5: Log calibration results
        self.log_calibration_results(&adaptive_params);

        log::info!(
            "✅ Adaptive parameter calibration completed in {}ms with overall balance score: {:.4}",
            calibration_time,
            overall_balance_score
        );

        Ok(adaptive_params)
    }

    /// Calibrate direction parameters for optimal momentum-based classification
    ///
    /// This method finds the optimal sensitivity parameters for direction targets
    /// by analyzing momentum change patterns and optimizing for balanced class distribution.
    ///
    /// ## Algorithm
    /// 1. **Sample Momentum Changes**: Calculate momentum changes across all sequences
    /// 2. **Distribution Analysis**: Analyze momentum change distribution characteristics
    /// 3. **Parameter Grid Search**: Test different sensitivity values
    /// 4. **Balance Optimization**: Find parameters that achieve target balance
    /// 5. **Validation**: Verify calibration quality and consistency
    ///
    /// ## Optimization Strategy
    /// - **Grid Search**: Test sensitivity values from 0.005 to 0.1
    /// - **Balance Scoring**: Minimize standard deviation from 20% per class
    /// - **Imbalance Penalty**: Penalize extreme imbalance ratios
    /// - **Sample Size Validation**: Ensure sufficient samples for each class
    async fn calibrate_direction_parameters(
        &self,
        ohlcv_data: &[MarketDataRow],
        sequence_length: usize,
        horizon_steps: usize,
        sequence_indices: &[usize],
    ) -> Result<DirectionAdaptiveParams> {
        log::debug!("🎯 Starting direction parameter calibration...");

        // Step 1: Extract close prices for momentum analysis
        let close_prices: Vec<f64> = ohlcv_data.iter().map(|row| row.close).collect();

        if close_prices.len() < sequence_length + horizon_steps + 10 {
            log::warn!("Insufficient data for direction calibration, using defaults");
            return Ok(DirectionAdaptiveParams::default());
        }

        // Step 2: Sample momentum changes for distribution analysis
        let mut momentum_changes = Vec::new();
        let mut trend_consistencies = Vec::new();

        for &seq_idx in sequence_indices
            .iter()
            .take(sequence_indices.len().min(1000))
        {
            // Limit for performance
            let sequence_end_idx = seq_idx + sequence_length;
            let target_end_idx = sequence_end_idx + horizon_steps;

            if target_end_idx <= close_prices.len() {
                let sequence_prices = &close_prices[seq_idx..sequence_end_idx];
                let horizon_prices = &close_prices[sequence_end_idx..target_end_idx];

                if sequence_prices.len() >= 2 && horizon_prices.len() >= 2 {
                    // Calculate momentum change (same logic as direction.rs)
                    let seq_start = sequence_prices[0];
                    let seq_end = sequence_prices[sequence_prices.len() - 1];
                    let sequence_momentum = (seq_end - seq_start) / seq_start;

                    let hor_start = horizon_prices[0];
                    let hor_end = horizon_prices[horizon_prices.len() - 1];
                    let horizon_momentum = (hor_end - hor_start) / hor_start;

                    let momentum_change = horizon_momentum - sequence_momentum;

                    // Calculate trend consistency (same logic as direction.rs)
                    let trend_consistency = self.calculate_trend_consistency(sequence_prices)?;

                    if momentum_change.is_finite()
                        && trend_consistency.is_finite()
                        && trend_consistency > 0.0
                    {
                        momentum_changes.push(momentum_change);
                        trend_consistencies.push(trend_consistency);
                    }
                }
            }
        }

        if momentum_changes.is_empty() {
            log::warn!("No valid momentum changes found, using defaults");
            return Ok(DirectionAdaptiveParams::default());
        }

        log::debug!(
            "📊 Analyzed {} momentum changes for calibration",
            momentum_changes.len()
        );

        // Step 3: Calculate distribution characteristics
        let mean_trend_consistency =
            trend_consistencies.iter().sum::<f64>() / trend_consistencies.len() as f64;

        // Step 4: Grid search for optimal sensitivity
        let sensitivity_candidates = vec![
            0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.06, 0.08, 0.1,
        ];

        let extreme_multiplier_candidates = vec![1.5, 2.0, 2.5, 3.0];

        let mut best_params = DirectionAdaptiveParams::default();
        let mut best_balance_score = f64::INFINITY;

        for &sensitivity in &sensitivity_candidates {
            for &extreme_mult in &extreme_multiplier_candidates {
                // Test this parameter combination
                let balance = self.evaluate_direction_parameters(
                    &momentum_changes,
                    &trend_consistencies,
                    sensitivity,
                    extreme_mult,
                    mean_trend_consistency,
                )?;

                // Score this configuration (lower is better)
                // MIN-CLASS OPTIMIZATION: Prioritize parameters that maximize minimum class representation
                let min_class_ratio = balance
                    .class_percentages
                    .iter()
                    .min_by(|a, b| a.partial_cmp(b).unwrap())
                    .unwrap()
                    / 100.0;
                let min_class_threshold = 0.15; // 15% minimum per class (vs 20% ideal)

                // Heavy penalty if any class falls below threshold
                let min_class_penalty = if min_class_ratio < min_class_threshold {
                    (min_class_threshold - min_class_ratio) * 50.0 // 50x penalty for under-representation
                } else {
                    0.0
                };

                let score = balance.balance_score
                    + (balance.imbalance_ratio - 1.0) * 0.1
                    + min_class_penalty;

                if score < best_balance_score {
                    best_balance_score = score;
                    best_params = DirectionAdaptiveParams {
                        base_sensitivity: sensitivity,
                        extreme_multiplier: extreme_mult,
                        momentum_weighting: self.base_config.momentum_weighting,
                        trend_consistency_factor: mean_trend_consistency,
                        achieved_balance: balance,
                    };
                }
            }
        }

        let min_class_ratio = best_params
            .achieved_balance
            .class_percentages
            .iter()
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap()
            / 100.0;

        log::info!(
            "🎯 Direction calibration: sensitivity={:.6}, extreme_mult={:.2}, balance_score={:.4}, imbalance={:.1}x, min_class={:.1}%",
            best_params.base_sensitivity,
            best_params.extreme_multiplier,
            best_params.achieved_balance.balance_score,
            best_params.achieved_balance.imbalance_ratio,
            min_class_ratio * 100.0
        );

        Ok(best_params)
    }

    /// Calculate trend consistency for a price sequence (helper method)
    fn calculate_trend_consistency(&self, prices: &[f64]) -> Result<f64> {
        if prices.len() < 3 {
            return Ok(0.01); // Default consistency for short sequences
        }

        let mut momentum_changes = Vec::new();

        // Calculate momentum between consecutive segments
        let segment_size = (prices.len() / 3).max(2);
        for i in 0..(prices.len() - segment_size * 2) {
            let seg1_start = prices[i];
            let seg1_end = prices[i + segment_size];
            let seg2_start = seg1_end;
            let seg2_end = prices[i + segment_size * 2];

            if seg1_start != 0.0 && seg2_start != 0.0 {
                let seg1_momentum = (seg1_end - seg1_start) / seg1_start;
                let seg2_momentum = (seg2_end - seg2_start) / seg2_start;
                let momentum_change = seg2_momentum - seg1_momentum;

                if momentum_change.is_finite() {
                    momentum_changes.push(momentum_change);
                }
            }
        }

        if momentum_changes.is_empty() {
            return Ok(0.01);
        }

        // Calculate standard deviation of momentum changes (trend consistency)
        let mean = momentum_changes.iter().sum::<f64>() / momentum_changes.len() as f64;
        let variance = momentum_changes
            .iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>()
            / momentum_changes.len() as f64;
        let std_dev = variance.sqrt();

        Ok(std_dev.max(0.005)) // Minimum consistency threshold
    }

    /// Evaluate direction parameters by simulating classification
    fn evaluate_direction_parameters(
        &self,
        momentum_changes: &[f64],
        trend_consistencies: &[f64],
        base_sensitivity: f64,
        extreme_multiplier: f64,
        mean_trend_consistency: f64,
    ) -> Result<ClassDistributionBalance> {
        let mut class_counts = [0usize; 5];

        for (i, &momentum_change) in momentum_changes.iter().enumerate() {
            let trend_consistency = trend_consistencies
                .get(i)
                .unwrap_or(&mean_trend_consistency);

            // Apply same classification logic as direction.rs
            let base_multiplier = base_sensitivity * 20.0; // Scale for momentum changes
            let base_threshold = trend_consistency * base_multiplier;
            let extreme_threshold = trend_consistency * base_multiplier * extreme_multiplier;

            // Ensure reasonable minimum thresholds
            let min_base = 0.01; // 1% minimum momentum change
            let min_extreme = 0.03; // 3% minimum for extreme changes

            let final_base_threshold = base_threshold.max(min_base);
            let final_extreme_threshold = extreme_threshold.max(min_extreme);

            // Classify based on momentum change magnitude and direction
            let class = if momentum_change <= -final_extreme_threshold {
                0 // DUMP: Strong momentum reversal
            } else if momentum_change <= -final_base_threshold {
                1 // DOWN: Moderate momentum weakening
            } else if momentum_change.abs() <= final_base_threshold {
                2 // SIDEWAYS: Momentum continuation
            } else if momentum_change <= final_extreme_threshold {
                3 // UP: Moderate momentum strengthening
            } else {
                4 // PUMP: Strong momentum acceleration
            };

            class_counts[class] += 1;
        }

        Ok(calculate_class_distribution_balance(&class_counts))
    }

    /// Calibrate price level parameters for optimal VWAP-weighted classification
    ///
    /// This method finds the optimal parameters for price level targets by analyzing
    /// VWAP-weighted price distributions and optimizing for balanced range-based classification.
    ///
    /// ## Algorithm
    /// 1. **VWAP Analysis**: Calculate VWAP baselines and target prices across sequences
    /// 2. **Range Distribution**: Analyze price range characteristics and volatility patterns
    /// 3. **Percentile Optimization**: Find optimal adaptive percentiles for boundaries
    /// 4. **Bandwidth Calibration**: Optimize bandwidth size for breakout sensitivity
    /// 5. **Balance Validation**: Verify balanced class distribution achievement
    ///
    /// ## Optimization Strategy
    /// - **Adaptive Percentiles**: Test percentile combinations for optimal boundaries
    /// - **Bandwidth Scaling**: Test bandwidth sizes from 0.5 to 2.0
    /// - **Momentum Weighting**: Optimize momentum factor for VWAP calculation
    /// - **Volatility Adjustment**: Account for sequence volatility characteristics
    async fn calibrate_price_level_parameters(
        &self,
        ohlcv_data: &[MarketDataRow],
        sequence_length: usize,
        horizon_steps: usize,
        sequence_indices: &[usize],
    ) -> Result<PriceLevelAdaptiveParams> {
        use crate::targets::price_levels::{get_horizon_vwap, get_sequence_vwap_baseline};

        log::debug!("🎯 Starting price level parameter calibration...");

        if ohlcv_data.len() < sequence_length + horizon_steps + 10 {
            log::warn!("Insufficient data for price level calibration, using defaults");
            return Ok(PriceLevelAdaptiveParams::default());
        }

        // Step 1: Sample VWAP data and price ranges for analysis
        let mut sequence_ranges = Vec::new();
        let mut target_prices = Vec::new();
        let mut sequence_vwaps = Vec::new();
        let mut volatility_measures = Vec::new();

        for &seq_idx in sequence_indices
            .iter()
            .take(sequence_indices.len().min(1000))
        {
            // Limit for performance
            let sequence_end_idx = seq_idx + sequence_length;
            let target_end_idx = sequence_end_idx + horizon_steps;

            if target_end_idx <= ohlcv_data.len() {
                let sequence_ohlcv = &ohlcv_data[seq_idx..sequence_end_idx];
                let horizon_ohlcv = &ohlcv_data[sequence_end_idx..target_end_idx];

                if sequence_ohlcv.len() >= 2 && horizon_ohlcv.len() >= 2 {
                    // Calculate sequence VWAP and range
                    let seq_vwap = get_sequence_vwap_baseline(sequence_ohlcv)?;
                    let seq_prices: Vec<f64> = sequence_ohlcv
                        .iter()
                        .map(|c| (c.open + c.high + c.low + c.close) / 4.0)
                        .collect();

                    let seq_min = seq_prices.iter().fold(f64::INFINITY, |a, &b| a.min(b));
                    let seq_max = seq_prices.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                    let seq_range = seq_max - seq_min;

                    // Calculate target VWAP
                    let target_vwap = get_horizon_vwap(horizon_ohlcv)?;

                    // Calculate volatility measure (coefficient of variation)
                    let price_mean = seq_prices.iter().sum::<f64>() / seq_prices.len() as f64;
                    let price_variance = seq_prices
                        .iter()
                        .map(|&p| (p - price_mean).powi(2))
                        .sum::<f64>()
                        / seq_prices.len() as f64;
                    let volatility = if price_mean > 1e-8 {
                        price_variance.sqrt() / price_mean
                    } else {
                        0.02
                    };

                    if seq_vwap > 0.0 && target_vwap > 0.0 && seq_range > 0.0 {
                        sequence_ranges.push(seq_range / seq_vwap); // Normalized range
                        target_prices.push(target_vwap);
                        sequence_vwaps.push(seq_vwap);
                        volatility_measures.push(volatility);
                    }
                }
            }
        }

        if sequence_ranges.is_empty() {
            log::warn!("No valid price level data found, using defaults");
            return Ok(PriceLevelAdaptiveParams::default());
        }

        log::debug!(
            "📊 Analyzed {} price level sequences for calibration",
            sequence_ranges.len()
        );

        // Step 2: Calculate distribution characteristics
        let mean_volatility =
            volatility_measures.iter().sum::<f64>() / volatility_measures.len() as f64;
        let _mean_range = sequence_ranges.iter().sum::<f64>() / sequence_ranges.len() as f64;

        // Step 3: Grid search for optimal parameters
        let bandwidth_candidates = vec![0.5, 0.7, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0];
        let percentile_candidates = vec![
            [0.05, 0.95],
            [0.1, 0.9],
            [0.15, 0.85],
            [0.2, 0.8],
            [0.25, 0.75],
        ];
        let momentum_factor_candidates = vec![1.0, 1.1, 1.2, 1.3, 1.5];

        let mut best_params = PriceLevelAdaptiveParams::default();
        let mut best_balance_score = f64::INFINITY;

        for &bandwidth in &bandwidth_candidates {
            for &percentiles in &percentile_candidates {
                for &momentum_factor in &momentum_factor_candidates {
                    // Test this parameter combination
                    let eval_params = PriceLevelEvaluationParams {
                        bandwidth_size: bandwidth,
                        percentiles,
                        momentum_factor,
                    };
                    let balance = self
                        .evaluate_price_level_parameters(
                            ohlcv_data,
                            sequence_indices,
                            sequence_length,
                            horizon_steps,
                            &eval_params,
                        )
                        .await?;

                    // Score this configuration (lower is better)
                    let score = balance.balance_score + (balance.imbalance_ratio - 1.0) * 0.1;

                    if score < best_balance_score {
                        best_balance_score = score;
                        best_params = PriceLevelAdaptiveParams {
                            bandwidth_size: bandwidth,
                            adaptive_percentiles: percentiles,
                            momentum_factor,
                            volatility_adjustment: mean_volatility / 0.02, // Normalize to 2% baseline
                            achieved_balance: balance,
                        };
                    }
                }
            }
        }

        log::info!(
            "🎯 Price level calibration: bandwidth={:.3}, percentiles=[{:.2}, {:.2}], momentum={:.2}, balance_score={:.4}, imbalance={:.1}x",
            best_params.bandwidth_size,
            best_params.adaptive_percentiles[0],
            best_params.adaptive_percentiles[1],
            best_params.momentum_factor,
            best_params.achieved_balance.balance_score,
            best_params.achieved_balance.imbalance_ratio
        );

        Ok(best_params)
    }

    /// Evaluate price level parameters by simulating classification
    async fn evaluate_price_level_parameters(
        &self,
        ohlcv_data: &[MarketDataRow],
        sequence_indices: &[usize],
        sequence_length: usize,
        horizon_steps: usize,
        params: &PriceLevelEvaluationParams,
    ) -> Result<ClassDistributionBalance> {
        use crate::targets::price_levels::{calculate_vwap_with_momentum, get_horizon_vwap};
        use crate::targets::sequence_reconstruction::{
            SequenceAnalyzer, SequenceReconstructionConfig,
        };

        let mut class_counts = [0usize; 5];
        let sample_limit = sequence_indices.len().min(500); // Limit for performance

        for &seq_idx in sequence_indices.iter().take(sample_limit) {
            let sequence_end_idx = seq_idx + sequence_length;
            let target_end_idx = sequence_end_idx + horizon_steps;

            if target_end_idx <= ohlcv_data.len() {
                let sequence_ohlcv = &ohlcv_data[seq_idx..sequence_end_idx];
                let horizon_ohlcv = &ohlcv_data[sequence_end_idx..target_end_idx];

                if sequence_ohlcv.len() >= 2 && horizon_ohlcv.len() >= 2 {
                    // Calculate sequence VWAP with momentum weighting
                    let _seq_vwap = if params.momentum_factor != 1.0 {
                        calculate_vwap_with_momentum(sequence_ohlcv, params.momentum_factor)?
                    } else {
                        crate::targets::price_levels::get_sequence_vwap_baseline(sequence_ohlcv)?
                    };

                    // Calculate target VWAP
                    let target_vwap = get_horizon_vwap(horizon_ohlcv)?;

                    // Use sequence reconstruction for consistent classification
                    let reconstruction_config = SequenceReconstructionConfig {
                        percentiles: params.percentiles,
                        bandwidth_size: params.bandwidth_size,
                    };
                    let analyzer = SequenceAnalyzer::new(reconstruction_config);
                    let boundaries = analyzer.calculate_boundaries(sequence_ohlcv)?;

                    // Handle edge case: flat sequence
                    if boundaries.bandwidth == 0.0 {
                        let class = if target_vwap >= boundaries.sequence_min {
                            3
                        } else {
                            2
                        };
                        class_counts[class] += 1;
                        continue;
                    }

                    // Classify using centralized logic
                    let class = boundaries.classify_price(target_vwap);
                    if (0..5).contains(&class) {
                        class_counts[class as usize] += 1;
                    }
                }
            }
        }

        Ok(calculate_class_distribution_balance(&class_counts))
    }

    /// Calibrate volatility parameters for optimal ATR distribution-based classification
    ///
    /// This method finds the optimal parameters for volatility targets by analyzing
    /// ATR distribution patterns and optimizing for balanced volatility regime classification.
    ///
    /// ## Algorithm
    /// 1. **ATR Distribution Analysis**: Calculate ATR ratios across all sequences
    /// 2. **Logarithmic Ratio Sampling**: Analyze log ratio distribution characteristics
    /// 3. **Bandwidth Optimization**: Find optimal bandwidth for logarithmic thresholds
    /// 4. **Extreme Multiplier Tuning**: Optimize extreme class boundary multipliers
    /// 5. **Distribution Validation**: Verify balanced volatility regime classification
    ///
    /// ## Optimization Strategy
    /// - **Log Ratio Analysis**: Sample ATR ratios and convert to symmetric log space
    /// - **Percentile-Based Calibration**: Use distribution percentiles for threshold setting
    /// - **Coefficient of Variation**: Account for ATR distribution characteristics
    /// - **Balance Optimization**: Minimize deviation from 20% per class target
    async fn calibrate_volatility_parameters(
        &self,
        ohlcv_data: &[MarketDataRow],
        sequence_length: usize,
        horizon_steps: usize,
        sequence_indices: &[usize],
    ) -> Result<VolatilityAdaptiveParams> {
        use crate::targets::volatility::get_sequence_atr_baseline;

        log::debug!("🎯 Starting volatility parameter calibration...");

        if ohlcv_data.len() < sequence_length + horizon_steps + 10 {
            log::warn!("Insufficient data for volatility calibration, using defaults");
            return Ok(VolatilityAdaptiveParams::default());
        }

        // Step 1: Sample ATR ratios for distribution analysis
        let mut atr_ratios = Vec::new();
        let mut log_ratios = Vec::new();
        let mut sequence_atr_values = Vec::new();
        let mut horizon_atr_values = Vec::new();

        for &seq_idx in sequence_indices
            .iter()
            .take(sequence_indices.len().min(1000))
        {
            // Limit for performance
            let sequence_end_idx = seq_idx + sequence_length;
            let target_end_idx = sequence_end_idx + horizon_steps;

            if target_end_idx <= ohlcv_data.len() {
                let sequence_candles = &ohlcv_data[seq_idx..sequence_end_idx];
                let horizon_candles = &ohlcv_data[sequence_end_idx..target_end_idx];

                if sequence_candles.len() >= 2 && horizon_candles.len() >= 2 {
                    // Calculate sequence and horizon ATR
                    let seq_atr = get_sequence_atr_baseline(sequence_candles)?;
                    let hor_atr = get_sequence_atr_baseline(horizon_candles)?;

                    if seq_atr > 0.0 && hor_atr > 0.0 {
                        let atr_ratio = hor_atr / seq_atr;
                        let log_ratio = atr_ratio.ln();

                        if atr_ratio.is_finite() && log_ratio.is_finite() {
                            atr_ratios.push(atr_ratio);
                            log_ratios.push(log_ratio);
                            sequence_atr_values.push(seq_atr);
                            horizon_atr_values.push(hor_atr);
                        }
                    }
                }
            }
        }

        if atr_ratios.is_empty() {
            log::warn!("No valid ATR ratios found, using defaults");
            return Ok(VolatilityAdaptiveParams::default());
        }

        log::debug!(
            "📊 Analyzed {} ATR ratios for calibration",
            atr_ratios.len()
        );

        // Step 2: Calculate ATR distribution statistics
        let atr_stats = calculate_atr_distribution_stats(&sequence_atr_values);

        // Step 3: Analyze log ratio distribution for threshold calibration
        let mut sorted_log_ratios = log_ratios.clone();
        sorted_log_ratios.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let n = sorted_log_ratios.len();
        let log_ratio_std = {
            let mean = log_ratios.iter().sum::<f64>() / n as f64;
            let variance = log_ratios.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
            variance.sqrt()
        };

        // Step 4: Grid search for optimal parameters
        let bandwidth_candidates = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0];
        let extreme_multiplier_candidates = vec![1.5, 2.0, 2.5, 3.0];

        let mut best_params = VolatilityAdaptiveParams::default();
        let mut best_balance_score = f64::INFINITY;

        for &bandwidth in &bandwidth_candidates {
            for &extreme_mult in &extreme_multiplier_candidates {
                // Test this parameter combination
                let balance = self.evaluate_volatility_parameters(
                    &sequence_atr_values,
                    &horizon_atr_values,
                    bandwidth,
                    extreme_mult,
                )?;

                // Score this configuration (lower is better)
                let score = balance.balance_score + (balance.imbalance_ratio - 1.0) * 0.1;

                if score < best_balance_score {
                    best_balance_score = score;
                    best_params = VolatilityAdaptiveParams {
                        bandwidth_size: bandwidth,
                        extreme_multiplier: extreme_mult,
                        atr_distribution_stats: AtrDistributionStats {
                            mean: atr_stats.mean,
                            std_dev: atr_stats.std_dev,
                            median: atr_stats.median,
                            percentile_25: atr_stats.percentile_25,
                            percentile_75: atr_stats.percentile_75,
                            coefficient_of_variation: if atr_stats.mean > 1e-8 {
                                atr_stats.std_dev / atr_stats.mean
                            } else {
                                0.5
                            },
                        },
                        cv_adjustment_factor: log_ratio_std / 0.5, // Normalize to 0.5 baseline
                        achieved_balance: balance,
                    };
                }
            }
        }

        log::info!(
            "🎯 Volatility calibration: bandwidth={:.3}, extreme_mult={:.2}, cv={:.3}, balance_score={:.4}, imbalance={:.1}x",
            best_params.bandwidth_size,
            best_params.extreme_multiplier,
            best_params.atr_distribution_stats.coefficient_of_variation,
            best_params.achieved_balance.balance_score,
            best_params.achieved_balance.imbalance_ratio
        );

        Ok(best_params)
    }

    /// Evaluate volatility parameters by simulating classification
    fn evaluate_volatility_parameters(
        &self,
        sequence_atr_values: &[f64],
        horizon_atr_values: &[f64],
        bandwidth_size: f64,
        extreme_multiplier: f64,
    ) -> Result<ClassDistributionBalance> {
        use crate::targets::volatility::{classify_volatility_log_ratio, LogVolatilityThresholds};

        let mut class_counts = [0usize; 5];

        // Create thresholds using same logic as volatility.rs
        let half_bandwidth = bandwidth_size / 2.0;
        let extreme_bandwidth = bandwidth_size * extreme_multiplier;

        let thresholds = LogVolatilityThresholds {
            very_low_max: -extreme_bandwidth,
            low_max: -half_bandwidth,
            medium_max: half_bandwidth,
            high_max: extreme_bandwidth,
        };

        // Classify each ATR ratio pair
        for (i, &seq_atr) in sequence_atr_values.iter().enumerate() {
            if let Some(&hor_atr) = horizon_atr_values.get(i) {
                if seq_atr > 0.0 && hor_atr > 0.0 {
                    let class = classify_volatility_log_ratio(seq_atr, hor_atr, &thresholds);
                    if (0..5).contains(&class) {
                        class_counts[class as usize] += 1;
                    }
                }
            }
        }

        Ok(calculate_class_distribution_balance(&class_counts))
    }

    /// Log comprehensive calibration results
    fn log_calibration_results(&self, params: &AdaptiveTargetParameters) {
        log::info!("🎯 ADAPTIVE PARAMETER CALIBRATION RESULTS");
        log::info!("==========================================");

        // Direction results
        log::info!(
            "📊 Direction: sensitivity={:.6}, extreme_mult={:.2}, balance_score={:.4}, imbalance={:.1}x",
            params.direction.base_sensitivity,
            params.direction.extreme_multiplier,
            params.direction.achieved_balance.balance_score,
            params.direction.achieved_balance.imbalance_ratio
        );

        // Price level results
        log::info!(
            "📊 Price Levels: bandwidth={:.4}, percentiles=[{:.2}, {:.2}], balance_score={:.4}, imbalance={:.1}x",
            params.price_levels.bandwidth_size,
            params.price_levels.adaptive_percentiles[0],
            params.price_levels.adaptive_percentiles[1],
            params.price_levels.achieved_balance.balance_score,
            params.price_levels.achieved_balance.imbalance_ratio
        );

        // Volatility results
        log::info!(
            "📊 Volatility: bandwidth={:.4}, extreme_mult={:.2}, balance_score={:.4}, imbalance={:.1}x",
            params.volatility.bandwidth_size,
            params.volatility.extreme_multiplier,
            params.volatility.achieved_balance.balance_score,
            params.volatility.achieved_balance.imbalance_ratio
        );

        // Overall results
        log::info!(
            "🎯 Overall: balance_score={:.4}, calibration_time={}ms, success={}",
            params.calibration_info.overall_balance_score,
            params.calibration_info.optimization_time_ms,
            params.calibration_info.calibration_success
        );

        log::info!("==========================================");
    }
}

/// Calculate class distribution balance metrics
pub fn calculate_class_distribution_balance(class_counts: &[usize; 5]) -> ClassDistributionBalance {
    let total_samples: usize = class_counts.iter().sum();

    if total_samples == 0 {
        return ClassDistributionBalance::default();
    }

    // Calculate class percentages
    let mut class_percentages = [0.0; 5];
    for (i, &count) in class_counts.iter().enumerate() {
        class_percentages[i] = (count as f64 / total_samples as f64) * 100.0;
    }

    // Calculate imbalance ratio
    let min_class_size = class_counts.iter().filter(|&&c| c > 0).min().unwrap_or(&1);
    let max_class_size = class_counts.iter().max().unwrap_or(&1);
    let imbalance_ratio = *max_class_size as f64 / *min_class_size as f64;

    // Calculate balance score (standard deviation of percentages from 20%)
    let target_percentage = 20.0; // 20% per class for 5-class system
    let variance = class_percentages
        .iter()
        .map(|&p| (p - target_percentage).powi(2))
        .sum::<f64>()
        / 5.0;
    let balance_score = variance.sqrt();

    ClassDistributionBalance {
        class_percentages,
        imbalance_ratio,
        total_samples,
        balance_score,
    }
}

/// Optimization helper functions
pub mod optimization {
    use super::*;

    /// Grid search optimization for parameter tuning
    pub struct GridSearchOptimizer {
        pub parameter_ranges: HashMap<String, Vec<f64>>,
        pub evaluation_metric: String, // "balance_score", "imbalance_ratio", etc.
    }

    impl Default for GridSearchOptimizer {
        fn default() -> Self {
            Self {
                parameter_ranges: HashMap::new(),
                evaluation_metric: "balance_score".to_string(),
            }
        }
    }

    impl GridSearchOptimizer {
        /// Create new grid search optimizer
        pub fn new() -> Self {
            Self::default()
        }

        /// Add parameter range for optimization
        pub fn add_parameter_range(&mut self, name: String, values: Vec<f64>) {
            self.parameter_ranges.insert(name, values);
        }

        /// Find optimal parameters using grid search
        pub async fn optimize<F, T>(&self, _evaluation_fn: F) -> Result<HashMap<String, f64>>
        where
            F: Fn(&HashMap<String, f64>) -> Result<T>,
            T: PartialOrd + Copy,
        {
            // Implementation will be added when needed
            Ok(HashMap::new())
        }
    }

    /// Bayesian optimization for more efficient parameter search
    pub struct BayesianOptimizer {
        pub bounds: HashMap<String, (f64, f64)>,
        pub acquisition_function: String,
        pub max_iterations: usize,
    }

    impl Default for BayesianOptimizer {
        fn default() -> Self {
            Self {
                bounds: HashMap::new(),
                acquisition_function: "expected_improvement".to_string(),
                max_iterations: 50,
            }
        }
    }

    impl BayesianOptimizer {
        /// Create new Bayesian optimizer
        pub fn new() -> Self {
            Self::default()
        }

        /// Add parameter bounds for optimization
        pub fn add_parameter_bounds(&mut self, name: String, bounds: (f64, f64)) {
            self.bounds.insert(name, bounds);
        }
    }
}

// Additional calibration methods for new targets
impl AdaptiveParameterCalibrator {
    /// Calibrate sentiment parameters for balanced distribution
    pub async fn calibrate_sentiment_parameters(
        &self,
        df: &DataFrame,
        horizons: &[String],
        sequence_indices: &[usize],
        sequence_length: usize,
    ) -> Result<SentimentAdaptiveParams> {
        log::info!("🎯 Calibrating sentiment parameters for balanced distribution");

        let mut best_params = SentimentAdaptiveParams::default();
        let mut best_balance_score = f64::INFINITY;

        // Grid search for optimal sentiment parameters
        let body_sensitivity_values = vec![0.5, 0.75, 1.0, 1.25, 1.5, 2.0];
        let volume_weight_values = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        let consistency_factor_values = vec![0.8, 1.0, 1.2, 1.5];

        for &body_sensitivity in &body_sensitivity_values {
            for &volume_weight in &volume_weight_values {
                for &consistency_factor in &consistency_factor_values {
                    let test_params = SentimentAdaptiveParams {
                        body_sensitivity,
                        volume_weight,
                        consistency_factor,
                        achieved_balance: ClassDistributionBalance::default(),
                    };

                    match self
                        .evaluate_sentiment_parameters(
                            df,
                            horizons,
                            sequence_indices,
                            sequence_length,
                            &test_params,
                        )
                        .await
                    {
                        Ok(balance_score) => {
                            if balance_score < best_balance_score {
                                best_balance_score = balance_score;
                                best_params = test_params;
                            }
                        }
                        Err(e) => {
                            log::warn!("Failed to evaluate sentiment parameters: {}", e);
                            continue;
                        }
                    }
                }
            }
        }

        // Update achieved balance
        best_params.achieved_balance.balance_score = best_balance_score;

        log::info!(
            "🎯 Sentiment calibration complete: body_sensitivity={:.3}, volume_weight={:.3}, consistency_factor={:.3}, balance_score={:.4}",
            best_params.body_sensitivity, best_params.volume_weight, best_params.consistency_factor, best_balance_score
        );

        Ok(best_params)
    }

    /// Calibrate volume parameters for balanced distribution
    pub async fn calibrate_volume_parameters(
        &self,
        df: &DataFrame,
        horizons: &[String],
        sequence_indices: &[usize],
        sequence_length: usize,
    ) -> Result<VolumeAdaptiveParams> {
        log::info!("🎯 Calibrating volume parameters for balanced distribution");

        let mut best_params = VolumeAdaptiveParams::default();
        let mut best_balance_score = f64::INFINITY;

        // Grid search for optimal volume parameters
        let bandwidth_values = vec![0.2, 0.3, 0.4, 0.5, 0.6, 0.8];
        let extreme_multiplier_values = vec![1.5, 2.0, 2.5, 3.0];
        let smoothing_periods_values = vec![1, 2, 3, 5];

        for &bandwidth_size in &bandwidth_values {
            for &extreme_multiplier in &extreme_multiplier_values {
                for &smoothing_periods in &smoothing_periods_values {
                    let test_params = VolumeAdaptiveParams {
                        bandwidth_size,
                        extreme_multiplier,
                        smoothing_periods,
                        achieved_balance: ClassDistributionBalance::default(),
                    };

                    match self
                        .evaluate_volume_parameters(
                            df,
                            horizons,
                            sequence_indices,
                            sequence_length,
                            &test_params,
                        )
                        .await
                    {
                        Ok(balance_score) => {
                            if balance_score < best_balance_score {
                                best_balance_score = balance_score;
                                best_params = test_params;
                            }
                        }
                        Err(e) => {
                            log::warn!("Failed to evaluate volume parameters: {}", e);
                            continue;
                        }
                    }
                }
            }
        }

        // Update achieved balance
        best_params.achieved_balance.balance_score = best_balance_score;

        log::info!(
            "🎯 Volume calibration complete: bandwidth={:.3}, extreme_multiplier={:.2}, smoothing_periods={}, balance_score={:.4}",
            best_params.bandwidth_size, best_params.extreme_multiplier, best_params.smoothing_periods, best_balance_score
        );

        Ok(best_params)
    }

    /// Evaluate sentiment parameters by simulating classification
    async fn evaluate_sentiment_parameters(
        &self,
        df: &DataFrame,
        horizons: &[String],
        sequence_indices: &[usize],
        sequence_length: usize,
        params: &SentimentAdaptiveParams,
    ) -> Result<f64> {
        use crate::targets::sentiment::generate_sentiment_targets_with_adaptive_params;

        // Generate targets with test parameters
        let targets = generate_sentiment_targets_with_adaptive_params(
            df,
            horizons,
            &self.base_config,
            sequence_indices,
            sequence_length,
            Some(params),
        )?;

        // Calculate balance score across all horizons
        let mut total_balance_score = 0.0;
        let mut horizon_count = 0;

        for (_, horizon_targets) in targets {
            if !horizon_targets.is_empty() {
                let mut class_counts = [0usize; 5];
                for &target in &horizon_targets {
                    if (0..5).contains(&target) {
                        class_counts[target as usize] += 1;
                    }
                }

                let balance = calculate_class_distribution_balance(&class_counts);
                total_balance_score += balance.balance_score;
                horizon_count += 1;
            }
        }

        if horizon_count > 0 {
            Ok(total_balance_score / horizon_count as f64)
        } else {
            Ok(f64::INFINITY) // No valid targets
        }
    }

    /// Evaluate volume parameters by simulating classification
    async fn evaluate_volume_parameters(
        &self,
        df: &DataFrame,
        horizons: &[String],
        sequence_indices: &[usize],
        sequence_length: usize,
        params: &VolumeAdaptiveParams,
    ) -> Result<f64> {
        use crate::targets::volume::generate_volume_targets_with_adaptive_params;

        // Generate targets with test parameters
        let targets = generate_volume_targets_with_adaptive_params(
            df,
            horizons,
            &self.base_config,
            sequence_indices,
            sequence_length,
            Some(params),
        )?;

        // Calculate balance score across all horizons
        let mut total_balance_score = 0.0;
        let mut horizon_count = 0;

        for (_, horizon_targets) in targets {
            if !horizon_targets.is_empty() {
                let mut class_counts = [0usize; 5];
                for &target in &horizon_targets {
                    if (0..5).contains(&target) {
                        class_counts[target as usize] += 1;
                    }
                }

                let balance = calculate_class_distribution_balance(&class_counts);
                total_balance_score += balance.balance_score;
                horizon_count += 1;
            }
        }

        if horizon_count > 0 {
            Ok(total_balance_score / horizon_count as f64)
        } else {
            Ok(f64::INFINITY) // No valid targets
        }
    }
}
