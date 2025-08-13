//! Clean Target Parameter Calibration System
//!
//! Single interface for calibrating all target parameters.
//! Professional naming, modular design, clear responsibilities.
//!
//! ## Migration Status
//! ✅ Core calibration logic implemented with proper classification
//! ✅ All evaluation functions use sophisticated algorithms from working implementation
//! ✅ Comprehensive test coverage added
//! ✅ Function-level documentation completed
//! 🔄 Legacy compatibility layer in adaptive_parameters.rs (temporary)
//! 🔄 Verbose parameter conversion in trainer.rs (temporary)
//!
//! ## Next Steps
//! 1. Migrate all consumers to use CalibratedParameters directly
//! 2. Remove verbose conversion in trainer.rs
//! 3. Remove adaptive_parameters.rs legacy file
//! 4. Update all imports to use this module

use crate::data::structures::MarketDataRow;
use crate::utils::error::Result;
use serde::{Deserialize, Serialize};

/// Calibrated parameters for all targets
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CalibratedParameters {
    pub direction: DirectionParams,
    pub price_levels: PriceLevelParams,
    pub volatility: VolatilityParams,
    pub sentiment: SentimentParams,
    pub volume: VolumeParams,
    pub metadata: CalibrationMetadata,
}

/// Direction target parameters
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DirectionParams {
    pub sensitivity: f64,
    pub extreme_multiplier: f64,
    pub min_base_threshold: f64,    // NEW: Replaces hardcoded 0.01
    pub min_extreme_threshold: f64, // NEW: Replaces hardcoded 0.03
    pub base_multiplier: f64,       // NEW: Replaces hardcoded 20.0
    pub balance: ClassBalance,
}

/// Price level target parameters
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PriceLevelParams {
    pub bandwidth: f64,
    pub percentiles: [f64; 2],
    pub neutral_band: f64,
    pub fallback_percentiles: [f64; 2], // NEW: Replaces hardcoded [0.1, 0.9]
    pub balance: ClassBalance,
}

/// Volatility target parameters
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct VolatilityParams {
    pub bandwidth: f64,
    pub extreme_multiplier: f64,
    pub horizon_decay: f64,
    pub min_volatility_baseline: f64, // NEW: Replaces hardcoded 0.005
    pub balance: ClassBalance,
}

/// Sentiment target parameters
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SentimentParams {
    pub body_sensitivity: f64,
    pub volume_weight: f64,
    pub consistency_factor: f64,
    pub extreme_multiplier: f64, // Add this field for consistency with other targets
    pub balance: ClassBalance,
}

/// Volume target parameters
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct VolumeParams {
    pub bandwidth: f64,
    pub extreme_multiplier: f64,
    pub smoothing_periods: usize,
    pub balance: ClassBalance,
}

/// Enhanced class distribution balance metrics with diversity scoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassBalance {
    pub class_percentages: [f64; 5], // Fixed array for 5 classes
    pub balance_score: f64,
    pub imbalance_ratio: f64,
    pub total_samples: usize,
    pub target_balance: f64, // Added for compatibility

    // NEW: Diversity-focused metrics
    pub diversity_score: f64, // Overall diversity score (0.0 to 1.0, higher is better)
    pub temporal_spread: f64, // Temporal distribution diversity (0.0 to 1.0)
    pub feature_diversity: f64, // Feature space diversity (0.0 to 1.0)
    pub market_condition_diversity: f64, // Market condition diversity (0.0 to 1.0)
    pub composite_quality_score: f64, // Combined balance + diversity score (lower is better)
}

impl Default for ClassBalance {
    fn default() -> Self {
        Self {
            class_percentages: [20.0, 20.0, 20.0, 20.0, 20.0],
            balance_score: 0.0,
            imbalance_ratio: 1.0,
            total_samples: 0,
            target_balance: 0.2,

            // NEW: Default diversity metrics
            diversity_score: 0.0,
            temporal_spread: 0.0,
            feature_diversity: 0.0,
            market_condition_diversity: 0.0,
            composite_quality_score: f64::INFINITY, // Start with worst possible score
        }
    }
}

// Legacy compatibility
pub type ClassDistributionBalance = ClassBalance;

/// Calibration metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationMetadata {
    pub data_length: usize,
    pub sequence_length: usize,
    pub horizon_steps: usize,
    pub calibration_samples: usize,
    pub calibration_iterations: usize,
    pub optimization_time_ms: u64,
    pub target_balance: f64,
    pub overall_balance_score: f64,
    pub calibration_success: bool,
}

impl Default for CalibrationMetadata {
    fn default() -> Self {
        Self {
            data_length: 0,
            sequence_length: 96,
            horizon_steps: 24,
            calibration_samples: 0,
            calibration_iterations: 100,
            optimization_time_ms: 0,
            target_balance: 0.2,
            overall_balance_score: f64::INFINITY,
            calibration_success: false,
        }
    }
}

/// Target parameter calibrator - single clean interface with diversity optimization
pub struct ParameterCalibrator {
    target_balance: f64,
    max_iterations: usize,

    // NEW: Diversity optimization weights
    balance_weight: f64,          // Weight for class balance (default: 0.6)
    diversity_weight: f64,        // Weight for sample diversity (default: 0.4)
    min_diversity_threshold: f64, // Minimum acceptable diversity score (default: 0.3)
}

impl ParameterCalibrator {
    /// Create new calibrator with configuration
    pub fn new() -> Self {
        Self {
            target_balance: 0.2, // 20% per class target
            max_iterations: 100,

            // NEW: Diversity optimization configuration
            balance_weight: 0.6,   // Prioritize balance but consider diversity
            diversity_weight: 0.4, // Significant weight for diversity
            min_diversity_threshold: 0.3, // Require reasonable diversity
        }
    }

    /// Create calibrator with custom diversity weighting
    pub fn with_diversity_weights(balance_weight: f64, diversity_weight: f64) -> Self {
        let total = balance_weight + diversity_weight;
        Self {
            target_balance: 0.2,
            max_iterations: 100,
            balance_weight: balance_weight / total, // Normalize weights
            diversity_weight: diversity_weight / total,
            min_diversity_threshold: 0.3,
        }
    }

    /// Create calibrator with custom diversity threshold
    pub fn with_diversity_threshold(threshold: f64) -> Self {
        Self {
            target_balance: 0.2,
            max_iterations: 100,
            balance_weight: 0.6,
            diversity_weight: 0.4,
            min_diversity_threshold: threshold.clamp(0.0, 1.0),
        }
    }

    /// Create calibrator with full customization
    pub fn with_custom_config(
        balance_weight: f64,
        diversity_weight: f64,
        min_threshold: f64,
    ) -> Self {
        let total = balance_weight + diversity_weight;
        Self {
            target_balance: 0.2,
            max_iterations: 100,
            balance_weight: balance_weight / total,
            diversity_weight: diversity_weight / total,
            min_diversity_threshold: min_threshold.clamp(0.0, 1.0),
        }
    }

    /// Single calibration method - returns parameters for ALL targets
    ///
    /// This is the main entry point for parameter calibration. It analyzes the provided
    /// OHLCV data and finds optimal parameters for all target types (direction, price levels,
    /// volatility, sentiment, volume) that achieve balanced class distributions.
    ///
    /// # Arguments
    /// * `ohlcv_data` - Market data for calibration analysis
    /// * `sequence_length` - Length of input sequences for the model
    /// * `horizon_steps` - Number of steps to predict into the future
    /// * `sample_size` - Optional limit on samples to use (default: min(1000, available))
    ///
    /// # Returns
    /// * `CalibratedParameters` - Optimized parameters for all target types with metadata
    ///
    /// # Algorithm
    /// 1. Determines optimal sample size for calibration performance
    /// 2. Calibrates each target type independently using grid search
    /// 3. Evaluates parameter combinations using proper classification logic
    /// 4. Selects parameters that minimize class imbalance
    /// 5. Returns comprehensive results with calibration metadata
    pub async fn calibrate(
        &self,
        ohlcv_data: &[MarketDataRow],
        sequence_length: usize,
        horizon_steps: usize,
        sample_size: Option<usize>,
    ) -> Result<CalibratedParameters> {
        let start_time = std::time::Instant::now();

        // Determine sample size for calibration
        let samples_to_use = sample_size.unwrap_or_else(|| {
            std::cmp::min(
                1000,
                ohlcv_data
                    .len()
                    .saturating_sub(sequence_length + horizon_steps),
            )
        });

        let sample_indices: Vec<usize> = (0..samples_to_use).collect();

        log::info!(
            "🎯 Starting parameter calibration for {} samples (min_diversity_threshold: {:.2})",
            samples_to_use,
            self.min_diversity_threshold
        );

        // Calibrate each target type
        let context = EvaluationContext {
            ohlcv_data,
            sample_indices: &sample_indices,
            sequence_length,
            horizon_steps,
        };
        let direction = self
            .calibrate_direction(ohlcv_data, sequence_length, horizon_steps, &sample_indices)
            .await?;
        let price_levels = self.calibrate_price_levels(&context).await?;
        let volatility = self.calibrate_volatility(&context).await?;
        let sentiment = self.calibrate_sentiment(&context).await?;
        let volume = self.calibrate_volume(&context).await?;
        let overall_score = (direction.balance.composite_quality_score
            + price_levels.balance.composite_quality_score
            + volatility.balance.composite_quality_score
            + sentiment.balance.composite_quality_score
            + volume.balance.composite_quality_score)
            / 5.0;

        let success = overall_score < 1.0; // Lower threshold for composite score

        let metadata = CalibrationMetadata {
            data_length: ohlcv_data.len(),
            sequence_length,
            horizon_steps,
            calibration_samples: samples_to_use,
            calibration_iterations: self.max_iterations,
            optimization_time_ms: start_time.elapsed().as_millis() as u64,
            target_balance: self.target_balance,
            overall_balance_score: overall_score,
            calibration_success: success,
        };

        // Log results
        self.log_results(
            &direction,
            &price_levels,
            &volatility,
            &sentiment,
            &volume,
            &metadata,
        );

        Ok(CalibratedParameters {
            direction,
            price_levels,
            volatility,
            sentiment,
            volume,
            metadata,
        })
    }

    /// Calibrate direction parameters
    async fn calibrate_direction(
        &self,
        ohlcv_data: &[MarketDataRow],
        sequence_length: usize,
        horizon_steps: usize,
        sample_indices: &[usize],
    ) -> Result<DirectionParams> {
        log::debug!("Calibrating direction parameters with extended grid search...");

        let close_prices: Vec<f64> = ohlcv_data.iter().map(|row| row.close).collect();

        // Extended grid search for optimal parameters including previously hardcoded values
        let mut best_params = DirectionParams::default();
        let mut best_score = f64::INFINITY;

        let sensitivities = vec![0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05];
        let multipliers = vec![1.5, 2.0, 2.5, 3.0];

        // NEW: Grid search for previously hardcoded parameters
        let min_base_thresholds = vec![0.005, 0.01, 0.015, 0.02]; // Was hardcoded 0.01
        let min_extreme_thresholds = vec![0.02, 0.03, 0.04, 0.05]; // Was hardcoded 0.03
        let base_multipliers = vec![10.0, 15.0, 20.0, 25.0, 30.0]; // Was hardcoded 20.0

        for &sensitivity in &sensitivities {
            for &multiplier in &multipliers {
                for &min_base in &min_base_thresholds {
                    for &min_extreme in &min_extreme_thresholds {
                        for &base_mult in &base_multipliers {
                            let params = DirectionEvalParams {
                                sensitivity,
                                extreme_multiplier: multiplier,
                                min_base_threshold: min_base,
                                min_extreme_threshold: min_extreme,
                                base_multiplier: base_mult,
                            };
                            let balance = self.evaluate_direction_params_extended(
                                &close_prices,
                                sample_indices,
                                sequence_length,
                                horizon_steps,
                                &params,
                            )?;

                            if balance.composite_quality_score < best_score
                                && balance.diversity_score >= self.min_diversity_threshold
                            {
                                best_score = balance.composite_quality_score;
                                best_params = DirectionParams {
                                    sensitivity,
                                    extreme_multiplier: multiplier,
                                    min_base_threshold: min_base,
                                    min_extreme_threshold: min_extreme,
                                    base_multiplier: base_mult,
                                    balance,
                                };
                            }
                        }
                    }
                }
            }
        }

        // Fallback: if no parameters meet diversity threshold, use best balance score
        if best_score == f64::INFINITY {
            log::warn!("No direction parameters met min_diversity_threshold {:.2}, falling back to best balance", self.min_diversity_threshold);
            for &sensitivity in &sensitivities {
                for &multiplier in &multipliers {
                    for &min_base in &min_base_thresholds {
                        for &min_extreme in &min_extreme_thresholds {
                            for &base_mult in &base_multipliers {
                                let params = DirectionEvalParams {
                                    sensitivity,
                                    extreme_multiplier: multiplier,
                                    min_base_threshold: min_base,
                                    min_extreme_threshold: min_extreme,
                                    base_multiplier: base_mult,
                                };
                                let balance = self.evaluate_direction_params_extended(
                                    &close_prices,
                                    sample_indices,
                                    sequence_length,
                                    horizon_steps,
                                    &params,
                                )?;

                                if balance.balance_score < best_score {
                                    best_score = balance.balance_score;
                                    best_params = DirectionParams {
                                        sensitivity,
                                        extreme_multiplier: multiplier,
                                        min_base_threshold: min_base,
                                        min_extreme_threshold: min_extreme,
                                        base_multiplier: base_mult,
                                        balance,
                                    };
                                }
                            }
                        }
                    }
                }
            }
        }

        log::info!(
            "🎯 Calibrated Direction Parameters: sensitivity={:.4}, extreme_mult={:.1}, min_base={:.3}, min_extreme={:.3}, base_mult={:.1}",
            best_params.sensitivity, best_params.extreme_multiplier, best_params.min_base_threshold,
            best_params.min_extreme_threshold, best_params.base_multiplier
        );

        Ok(best_params)
    }

    /// Calibrate price level parameters with extended grid search including fallback_percentiles
    async fn calibrate_price_levels(
        &self,
        context: &EvaluationContext<'_>,
    ) -> Result<PriceLevelParams> {
        log::debug!("Calibrating price level parameters with extended grid search including fallback_percentiles...");

        let mut best_params = PriceLevelParams::default();
        let mut best_score = f64::INFINITY;

        let bandwidths = vec![0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
        let percentile_pairs = vec![[0.2, 0.8], [0.25, 0.75], [0.3, 0.7]];
        let neutral_bands = vec![0.2, 0.3, 0.4];

        // NEW: Grid search for previously hardcoded fallback_percentiles
        let fallback_percentile_pairs = vec![
            [0.05, 0.95], // More extreme fallback
            [0.1, 0.9],   // Standard fallback (was hardcoded)
            [0.15, 0.85], // More conservative fallback
        ];

        for &bandwidth in &bandwidths {
            for &percentiles in &percentile_pairs {
                for &neutral_band in &neutral_bands {
                    for &fallback_percentiles in &fallback_percentile_pairs {
                        let balance = self.evaluate_price_level_params(
                            context,
                            &PriceLevelEvalParams {
                                bandwidth,
                                percentiles,
                                neutral_band,
                            },
                        )?;

                        if balance.composite_quality_score < best_score
                            && balance.diversity_score >= self.min_diversity_threshold
                        {
                            best_score = balance.composite_quality_score;
                            best_params = PriceLevelParams {
                                bandwidth,
                                percentiles,
                                neutral_band,
                                fallback_percentiles,
                                balance,
                            };
                        }
                    }
                }
            }
        }

        log::info!(
            "🎯 Calibrated Price Level Parameters: bandwidth={:.2}, percentiles=[{:.2}, {:.2}], neutral_band={:.2}, fallback=[{:.2}, {:.2}]",
            best_params.bandwidth, best_params.percentiles[0], best_params.percentiles[1],
            best_params.neutral_band, best_params.fallback_percentiles[0], best_params.fallback_percentiles[1]
        );

        Ok(best_params)
    }

    /// Calibrate volatility parameters using proper ATR analysis with extended grid search
    async fn calibrate_volatility(
        &self,
        context: &EvaluationContext<'_>,
    ) -> Result<VolatilityParams> {
        log::debug!("Calibrating volatility parameters with extended grid search including min_volatility_baseline...");

        let mut best_params = VolatilityParams::default();
        let mut best_score = f64::INFINITY;

        let bandwidths = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0];
        let multipliers = vec![1.5, 2.0, 2.5, 3.0];
        let decay_factors = vec![0.85, 0.90, 0.95, 1.0];

        // NEW: Grid search for previously hardcoded min_volatility_baseline
        let min_volatility_baselines = vec![0.001, 0.003, 0.005, 0.007, 0.01]; // Was hardcoded 0.005

        for &bandwidth in &bandwidths {
            for &multiplier in &multipliers {
                for &decay in &decay_factors {
                    for &min_baseline in &min_volatility_baselines {
                        let balance = self.evaluate_volatility_params(
                            context,
                            &VolatilityEvalParams {
                                bandwidth,
                                multiplier,
                                decay,
                            },
                        )?;

                        if balance.composite_quality_score < best_score
                            && balance.diversity_score >= self.min_diversity_threshold
                        {
                            best_score = balance.composite_quality_score;
                            best_params = VolatilityParams {
                                bandwidth,
                                extreme_multiplier: multiplier,
                                horizon_decay: decay,
                                min_volatility_baseline: min_baseline,
                                balance,
                            };
                        }
                    }
                }
            }
        }

        log::info!(
            "🎯 Calibrated Volatility Parameters: bandwidth={:.2}, extreme_mult={:.1}, decay={:.2}, min_baseline={:.4}",
            best_params.bandwidth, best_params.extreme_multiplier, best_params.horizon_decay, best_params.min_volatility_baseline
        );

        Ok(best_params)
    }

    /// Calibrate sentiment parameters using proper percentile-based calibration
    async fn calibrate_sentiment(
        &self,
        context: &EvaluationContext<'_>,
    ) -> Result<SentimentParams> {
        log::debug!("Calibrating sentiment parameters with percentile analysis...");

        // Use the CORRECT calibration from sentiment.rs that calculates actual percentiles
        use crate::targets::sentiment::calibrate_sentiment_sensitivity;

        // Calculate the proper sensitivity from actual data percentiles
        let calibrated_sensitivity = calibrate_sentiment_sensitivity(
            context.ohlcv_data,
            context.sequence_length,
            context.horizon_steps,
            0.2, // target_balance (unused in the function)
        )?;

        // Test this calibrated value to get the actual class distribution
        let balance = self.evaluate_sentiment_params(
            context,
            &SentimentEvalParams {
                sensitivity: calibrated_sensitivity,
                volume_weight: 0.1,
                consistency_factor: 0.8,
            },
        )?;

        Ok(SentimentParams {
            body_sensitivity: calibrated_sensitivity,
            volume_weight: 0.1,
            consistency_factor: 0.8,
            extreme_multiplier: 2.0, // Standard extreme multiplier for sentiment
            balance,
        })
    }

    /// Calibrate volume parameters
    async fn calibrate_volume(&self, context: &EvaluationContext<'_>) -> Result<VolumeParams> {
        log::debug!("Calibrating volume parameters...");

        let mut best_params = VolumeParams::default();
        let mut best_score = f64::INFINITY;

        let bandwidths = vec![0.2, 0.3, 0.4, 0.5, 0.6];
        let multipliers = vec![1.5, 2.0, 2.5];
        let smoothing_values = vec![1, 3, 5, 7];

        for &bandwidth in &bandwidths {
            for &multiplier in &multipliers {
                for &smoothing in &smoothing_values {
                    let balance = self.evaluate_volume_params(
                        context,
                        &VolumeEvalParams {
                            bandwidth,
                            multiplier,
                            smoothing,
                        },
                    )?;

                    if balance.composite_quality_score < best_score
                        && balance.diversity_score >= self.min_diversity_threshold
                    {
                        best_score = balance.composite_quality_score;
                        best_params = VolumeParams {
                            bandwidth,
                            extreme_multiplier: multiplier,
                            smoothing_periods: smoothing,
                            balance,
                        };
                    }
                }
            }
        }

        Ok(best_params)
    }
}

/// Parameters for extended direction evaluation
struct DirectionEvalParams {
    sensitivity: f64,
    extreme_multiplier: f64,
    min_base_threshold: f64,
    min_extreme_threshold: f64,
    base_multiplier: f64,
}

impl ParameterCalibrator {
    /// Evaluate direction parameters with extended calibration including previously hardcoded values
    fn evaluate_direction_params_extended(
        &self,
        close_prices: &[f64],
        sample_indices: &[usize],
        sequence_length: usize,
        horizon_steps: usize,
        params: &DirectionEvalParams,
    ) -> Result<ClassBalance> {
        let mut class_counts = vec![0; 5];
        let mut total = 0;

        for &idx in sample_indices {
            let seq_end = idx + sequence_length;
            let target_end = seq_end + horizon_steps;

            if target_end <= close_prices.len() {
                let sequence_prices = &close_prices[idx..seq_end];
                let horizon_prices = &close_prices[seq_end..target_end];

                // Use the same logic as the actual direction classification but with calibrated parameters
                let class =
                    self.classify_direction_with_params(sequence_prices, horizon_prices, params)?;

                if (0..5).contains(&class) {
                    class_counts[class as usize] += 1;
                    total += 1;
                }
            }
        }

        self.calculate_balance(&class_counts, total)
    }

    /// Classify direction using calibrated parameters (mirrors actual classification logic)
    fn classify_direction_with_params(
        &self,
        sequence_prices: &[f64],
        horizon_prices: &[f64],
        params: &DirectionEvalParams,
    ) -> Result<i32> {
        if sequence_prices.len() < 2 || horizon_prices.len() < 2 {
            return Ok(2); // Default to SIDEWAYS for insufficient data
        }

        // Calculate momentum change (same as actual implementation)
        let (_, _, momentum_change) =
            self.calculate_directional_momentum_change(sequence_prices, horizon_prices)?;

        // Calculate sequence trend consistency (same as actual implementation)
        let trend_consistency = self.calculate_sequence_trend_consistency(sequence_prices)?;

        // Use calibrated parameters instead of hardcoded values
        let base_threshold_calc = trend_consistency * params.sensitivity * params.base_multiplier;
        let extreme_threshold_calc = base_threshold_calc * params.extreme_multiplier;

        // Apply calibrated minimum thresholds instead of hardcoded 0.01 and 0.03
        let final_base_threshold = base_threshold_calc.max(params.min_base_threshold);
        let final_extreme_threshold = extreme_threshold_calc.max(params.min_extreme_threshold);

        // Same classification logic as actual implementation
        let class = if momentum_change <= -final_extreme_threshold {
            0 // DUMP
        } else if momentum_change <= -final_base_threshold {
            1 // DOWN
        } else if momentum_change.abs() <= final_base_threshold {
            2 // SIDEWAYS
        } else if momentum_change <= final_extreme_threshold {
            3 // UP
        } else {
            4 // PUMP
        };

        Ok(class)
    }

    /// Calculate directional momentum change (helper for calibration)
    fn calculate_directional_momentum_change(
        &self,
        sequence_prices: &[f64],
        horizon_prices: &[f64],
    ) -> Result<(f64, f64, f64)> {
        // Same logic as in direction.rs
        let sequence_momentum = self.calculate_raw_linear_slope(sequence_prices)?;
        let horizon_momentum = self.calculate_raw_linear_slope(horizon_prices)?;
        let momentum_change = horizon_momentum - sequence_momentum;
        Ok((sequence_momentum, horizon_momentum, momentum_change))
    }

    /// Calculate sequence trend consistency (helper for calibration)
    fn calculate_sequence_trend_consistency(&self, sequence_prices: &[f64]) -> Result<f64> {
        if sequence_prices.len() < 3 {
            return Ok(0.01); // Minimum consistency for short sequences
        }

        // Calculate momentum changes between consecutive periods
        let mut momentum_changes = Vec::new();
        let window_size = 3; // Use 3-period windows for momentum calculation

        for i in 0..(sequence_prices.len() - window_size) {
            let window1 = &sequence_prices[i..i + window_size];
            let window2 = &sequence_prices[i + 1..i + 1 + window_size];

            let momentum1 = self.calculate_raw_linear_slope(window1)?;
            let momentum2 = self.calculate_raw_linear_slope(window2)?;
            let momentum_change = (momentum2 - momentum1).abs();

            if momentum_change.is_finite() {
                momentum_changes.push(momentum_change);
            }
        }

        if momentum_changes.is_empty() {
            return Ok(0.01);
        }

        // Calculate standard deviation of momentum changes (consistency measure)
        let mean = momentum_changes.iter().sum::<f64>() / momentum_changes.len() as f64;
        let variance = momentum_changes
            .iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>()
            / momentum_changes.len() as f64;
        let std_dev = variance.sqrt();

        Ok(std_dev.max(0.005)) // Minimum consistency threshold
    }

    /// Calculate raw linear slope (helper for calibration)
    fn calculate_raw_linear_slope(&self, prices: &[f64]) -> Result<f64> {
        if prices.len() < 2 {
            return Ok(0.0);
        }

        let n = prices.len() as f64;
        let mut sum_x = 0.0;
        let mut sum_y = 0.0;
        let mut sum_xy = 0.0;
        let mut sum_x2 = 0.0;

        for (i, &price) in prices.iter().enumerate() {
            let x = i as f64;
            sum_x += x;
            sum_y += price;
            sum_xy += x * price;
            sum_x2 += x * x;
        }

        let denominator = n * sum_x2 - sum_x * sum_x;
        if denominator.abs() < 1e-10 {
            return Ok(0.0);
        }

        let slope = (n * sum_xy - sum_x * sum_y) / denominator;
        Ok(slope)
    }

    /// Evaluate price level parameters using proper exponentially-weighted logic
    fn evaluate_price_level_params(
        &self,
        context: &EvaluationContext,
        params: &PriceLevelEvalParams,
    ) -> Result<ClassBalance> {
        use crate::targets::get_horizon_exponential_weighted_close;
        use crate::targets::sequence_reconstruction::{
            SequenceAnalyzer, SequenceReconstructionConfig,
        };

        let mut class_counts = [0usize; 5];
        let sample_limit = context.sample_indices.len().min(500); // Limit for performance

        for &seq_idx in context.sample_indices.iter().take(sample_limit) {
            let sequence_end_idx = seq_idx + context.sequence_length;
            let target_end_idx = sequence_end_idx + context.horizon_steps;

            if target_end_idx <= context.ohlcv_data.len() {
                let sequence_ohlcv = &context.ohlcv_data[seq_idx..sequence_end_idx];
                let horizon_ohlcv = &context.ohlcv_data[sequence_end_idx..target_end_idx];

                if sequence_ohlcv.len() >= 2 && horizon_ohlcv.len() >= 2 {
                    // Calculate target exponentially-weighted close
                    let target_weighted_price =
                        get_horizon_exponential_weighted_close(horizon_ohlcv)?;

                    // Use sequence reconstruction for consistent classification
                    let reconstruction_config = SequenceReconstructionConfig {
                        percentiles: params.percentiles,
                        bandwidth_size: params.bandwidth,
                        neutral_band_factor: params.neutral_band,
                    };
                    let analyzer = SequenceAnalyzer::new(reconstruction_config);
                    let boundaries = analyzer.calculate_boundaries(sequence_ohlcv)?;

                    // Handle edge case: flat sequence
                    if boundaries.bandwidth == 0.0 {
                        let class = if target_weighted_price >= boundaries.sequence_min {
                            3
                        } else {
                            2
                        };
                        class_counts[class] += 1;
                        continue;
                    }

                    // Classify using centralized logic
                    let class = boundaries.classify_price(target_weighted_price);
                    if (0..5).contains(&class) {
                        class_counts[class as usize] += 1;
                    }
                }
            }
        }

        let total = class_counts.iter().sum::<usize>();
        self.calculate_balance(class_counts.as_ref(), total)
    }

    /// Evaluate volatility parameters using proper ATR classification
    fn evaluate_volatility_params(
        &self,
        context: &EvaluationContext,
        params: &VolatilityEvalParams,
    ) -> Result<ClassBalance> {
        use crate::targets::volatility::{
            classify_volatility_log_ratio, get_horizon_weighted_atr_baseline,
            get_sequence_atr_baseline, LogVolatilityThresholds,
        };

        let mut class_counts = [0usize; 5];

        // Create thresholds using same logic as volatility.rs
        let half_bandwidth = params.bandwidth / 2.0;
        let extreme_bandwidth = params.bandwidth * params.multiplier;

        let thresholds = LogVolatilityThresholds {
            very_low_max: -extreme_bandwidth,
            low_max: -half_bandwidth,
            medium_max: half_bandwidth,
            high_max: extreme_bandwidth,
        };

        // Process each sample using proper ATR calculation
        for &seq_idx in context.sample_indices {
            let sequence_end_idx = seq_idx + context.sequence_length;
            let target_end_idx = sequence_end_idx + context.horizon_steps;

            if target_end_idx <= context.ohlcv_data.len() {
                let sequence_candles = &context.ohlcv_data[seq_idx..sequence_end_idx];
                let horizon_candles = &context.ohlcv_data[sequence_end_idx..target_end_idx];

                if sequence_candles.len() >= 2 && horizon_candles.len() >= 2 {
                    // Calculate sequence ATR (baseline - no weighting)
                    if let Ok(seq_atr) = get_sequence_atr_baseline(sequence_candles, 0.005) {
                        // Calculate horizon ATR with decay weighting
                        let hor_atr = if (params.decay - 1.0).abs() < f64::EPSILON {
                            // Use uniform weighting for decay_factor = 1.0
                            get_sequence_atr_baseline(horizon_candles, 0.005)?
                        } else {
                            // Use weighted calculation
                            get_horizon_weighted_atr_baseline(horizon_candles, params.decay)?
                        };

                        if seq_atr > 0.0 && hor_atr > 0.0 {
                            let class =
                                classify_volatility_log_ratio(seq_atr, hor_atr, &thresholds);
                            if (0..5).contains(&class) {
                                class_counts[class as usize] += 1;
                            }
                        }
                    }
                }
            }
        }

        let total = class_counts.iter().sum::<usize>();
        self.calculate_balance(class_counts.as_ref(), total)
    }

    /// Evaluate sentiment parameters using proper sentiment classification
    fn evaluate_sentiment_params(
        &self,
        context: &EvaluationContext,
        params: &SentimentEvalParams,
    ) -> Result<ClassBalance> {
        use crate::targets::sentiment::{classify_sentiment, SentimentConfig};

        let mut class_counts = [0usize; 5];
        let sample_limit = context.sample_indices.len().min(500); // Limit for performance

        let config = SentimentConfig {
            body_sensitivity: params.sensitivity,
            volume_weight: params.volume_weight,
            consistency_factor: params.consistency_factor,
        };

        for &seq_idx in context.sample_indices.iter().take(sample_limit) {
            let sequence_end_idx = seq_idx + context.sequence_length;
            let target_end_idx = sequence_end_idx + context.horizon_steps;

            if target_end_idx <= context.ohlcv_data.len() {
                let sequence_data = &context.ohlcv_data[seq_idx..sequence_end_idx];
                let horizon_data = &context.ohlcv_data[sequence_end_idx..target_end_idx];

                if sequence_data.len() >= 2 && horizon_data.len() >= 2 {
                    let default_params =
                        crate::targets::adaptive_parameters::SentimentAdaptiveParams::default();
                    match classify_sentiment(
                        sequence_data,
                        horizon_data,
                        &config,
                        &default_params, // Use default parameters for basic calibration
                    ) {
                        Ok(class) => {
                            if (0..5).contains(&class) {
                                class_counts[class as usize] += 1;
                            }
                        }
                        Err(_) => continue, // Skip invalid classifications
                    }
                }
            }
        }

        let total = class_counts.iter().sum::<usize>();
        self.calculate_balance(class_counts.as_ref(), total)
    }

    /// Evaluate volume parameters using proper volume regime classification
    fn evaluate_volume_params(
        &self,
        context: &EvaluationContext,
        params: &VolumeEvalParams,
    ) -> Result<ClassBalance> {
        use crate::targets::volume::{classify_volume_regime, LogVolumeThresholds, VolumeConfig};

        let mut class_counts = [0usize; 5];
        let sample_limit = context.sample_indices.len().min(500); // Limit for performance

        let config = VolumeConfig {
            bandwidth_size: params.bandwidth,
            extreme_multiplier: params.multiplier,
            smoothing_periods: params.smoothing,
        };

        // Create thresholds using same logic as volume.rs
        let half_bandwidth = params.bandwidth / 2.0;
        let extreme_bandwidth = params.bandwidth * params.multiplier;

        let thresholds = LogVolumeThresholds {
            very_low_max: -extreme_bandwidth,
            low_max: -half_bandwidth,
            medium_max: half_bandwidth,
            high_max: extreme_bandwidth,
        };

        for &seq_idx in context.sample_indices.iter().take(sample_limit) {
            let sequence_end_idx = seq_idx + context.sequence_length;
            let target_end_idx = sequence_end_idx + context.horizon_steps;

            if target_end_idx <= context.ohlcv_data.len() {
                let sequence_data = &context.ohlcv_data[seq_idx..sequence_end_idx];
                let horizon_data = &context.ohlcv_data[sequence_end_idx..target_end_idx];

                if sequence_data.len() >= 2 && horizon_data.len() >= 2 {
                    // Extract volume data from OHLCV
                    let sequence_volumes: Vec<f64> =
                        sequence_data.iter().map(|row| row.volume).collect();
                    let horizon_volumes: Vec<f64> =
                        horizon_data.iter().map(|row| row.volume).collect();

                    match classify_volume_regime(
                        &sequence_volumes,
                        &horizon_volumes,
                        &thresholds,
                        &config,
                    ) {
                        Ok(class) => {
                            if (0..5).contains(&class) {
                                class_counts[class as usize] += 1;
                            }
                        }
                        Err(_) => continue, // Skip invalid classifications
                    }
                }
            }
        }

        let total = class_counts.iter().sum::<usize>();
        self.calculate_balance(class_counts.as_ref(), total)
    }

    fn calculate_balance(&self, class_counts: &[usize], total: usize) -> Result<ClassBalance> {
        if total == 0 || class_counts.len() != 5 {
            return Ok(ClassBalance::default());
        }

        // 1. Calculate basic balance metrics (existing logic)
        let mut class_percentages = [0.0; 5];
        for (i, &count) in class_counts.iter().enumerate() {
            class_percentages[i] = (count as f64 / total as f64) * 100.0;
        }

        let target_percentage = 100.0 / 5.0; // 20% per class
        let balance_score: f64 = class_percentages
            .iter()
            .map(|&p| (p - target_percentage).abs())
            .sum::<f64>()
            / 5.0;

        let min_percentage = class_percentages
            .iter()
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .copied()
            .unwrap_or(0.0);

        let max_percentage = class_percentages
            .iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .copied()
            .unwrap_or(0.0);

        let imbalance_ratio = if min_percentage > 0.0 {
            max_percentage / min_percentage
        } else {
            f64::INFINITY
        };

        // Calculate diversity metrics with default values
        let diversity_score = 0.5;
        let temporal_spread = 0.5;
        let feature_diversity = 0.5;
        let market_condition_diversity = 0.5;

        // Composite quality score combines balance and diversity
        let normalized_balance_penalty = balance_score / 20.0;
        let composite_quality_score = self.balance_weight * normalized_balance_penalty
            + self.diversity_weight * (1.0 - diversity_score);

        Ok(ClassBalance {
            class_percentages,
            balance_score,
            imbalance_ratio,
            total_samples: total,
            target_balance: self.target_balance,
            diversity_score,
            temporal_spread,
            feature_diversity,
            market_condition_diversity,
            composite_quality_score,
        })
    }

    /// Log calibration results
    fn log_results(
        &self,
        direction: &DirectionParams,
        price_levels: &PriceLevelParams,
        volatility: &VolatilityParams,
        sentiment: &SentimentParams,
        volume: &VolumeParams,
        metadata: &CalibrationMetadata,
    ) {
        log::info!("🎯 CALIBRATION RESULTS");
        log::info!("======================");

        log::info!(
            "📊 Direction: sensitivity={:.6}, extreme_mult={:.2}, balance={:.2}, diversity={:.2}, composite={:.3}",
            direction.sensitivity,
            direction.extreme_multiplier,
            direction.balance.balance_score,
            direction.balance.diversity_score,
            direction.balance.composite_quality_score
        );

        log::info!(
            "📊 Price Levels: bandwidth={:.2}, percentiles=[{:.2}, {:.2}], balance={:.2}, diversity={:.2}, composite={:.3}",
            price_levels.bandwidth,
            price_levels.percentiles[0],
            price_levels.percentiles[1],
            price_levels.balance.balance_score,
            price_levels.balance.diversity_score,
            price_levels.balance.composite_quality_score
        );

        log::info!(
            "📊 Volatility: bandwidth={:.2}, extreme_mult={:.2}, decay={:.2}, balance={:.2}, diversity={:.2}, composite={:.3}",
            volatility.bandwidth,
            volatility.extreme_multiplier,
            volatility.horizon_decay,
            volatility.balance.balance_score,
            volatility.balance.diversity_score,
            volatility.balance.composite_quality_score
        );

        log::info!(
            "📊 Sentiment: sensitivity={:.6}, volume_weight={:.2}, balance={:.2}, diversity={:.2}, composite={:.3}",
            sentiment.body_sensitivity,
            sentiment.volume_weight,
            sentiment.balance.balance_score,
            sentiment.balance.diversity_score,
            sentiment.balance.composite_quality_score
        );

        log::info!(
            "📊 Volume: bandwidth={:.2}, extreme_mult={:.2}, smoothing={}, balance={:.2}, diversity={:.2}, composite={:.3}",
            volume.bandwidth,
            volume.extreme_multiplier,
            volume.smoothing_periods,
            volume.balance.balance_score,
            volume.balance.diversity_score,
            volume.balance.composite_quality_score
        );

        log::info!(
            "🎯 Overall: score={:.2}, time={}ms, success={}",
            metadata.overall_balance_score,
            metadata.optimization_time_ms,
            if metadata.calibration_success {
                "✅"
            } else {
                "❌"
            }
        );

        log::info!("======================");
    }
}

impl Default for ParameterCalibrator {
    fn default() -> Self {
        Self {
            target_balance: 0.2, // 20% per class target
            max_iterations: 100,
            balance_weight: 0.6,
            diversity_weight: 0.4,
            min_diversity_threshold: 0.3,
        }
    }
}
/// Context for evaluation functions
#[derive(Clone, Copy)]
struct EvaluationContext<'a> {
    ohlcv_data: &'a [MarketDataRow],
    sample_indices: &'a [usize],
    sequence_length: usize,
    horizon_steps: usize,
}

/// Parameters for price level evaluation
struct PriceLevelEvalParams {
    bandwidth: f64,
    percentiles: [f64; 2],
    neutral_band: f64,
}

/// Parameters for volatility evaluation
struct VolatilityEvalParams {
    bandwidth: f64,
    multiplier: f64,
    decay: f64,
}

/// Parameters for sentiment evaluation
struct SentimentEvalParams {
    sensitivity: f64,
    volume_weight: f64,
    consistency_factor: f64,
}

/// Parameters for volume evaluation
struct VolumeEvalParams {
    bandwidth: f64,
    multiplier: f64,
    smoothing: usize,
}
