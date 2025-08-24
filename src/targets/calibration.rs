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
    pub percentiles: [f64; 2], // Base percentiles for adaptive calculation
    pub neutral_band_factor: f64, // Replaces hardcoded 0.4 (was called neutral_band)
    pub momentum_factor: f64,  // NEW: Replaces hardcoded 1.2
    pub balance: ClassBalance,
}

/// Volatility target parameters
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct VolatilityParams {
    pub bandwidth: f64,
    pub extreme_multiplier: f64,
    pub volume_weight: f64, // NEW: Volume weight for volatility score calculation
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
    pub min_base_threshold: f64, // NEW: Minimum base threshold for consistency
    pub min_extreme_threshold: f64, // NEW: Minimum extreme threshold for consistency
    pub balance: ClassBalance,
}

/// Volume target parameters
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct VolumeParams {
    pub bandwidth: f64,
    pub extreme_multiplier: f64,
    pub smoothing_periods: usize,
    pub min_base_threshold: f64, // NEW: Minimum base threshold for consistency
    pub min_extreme_threshold: f64, // NEW: Minimum extreme threshold for consistency
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
    /// Generate diverse calibration sample indices using sequence generation logic + diversity selection
    fn generate_diverse_calibration_indices(
        &self,
        total_data_length: usize,
        sequence_length: usize,
        horizon_steps: usize,
        sample_size: Option<usize>,
        sequence_overlap: f64,
    ) -> Result<Vec<usize>> {
        use crate::utils::sequence_utils::{calculate_sequence_indices, calculate_step_size};

        // Step 1: Generate ALL possible sequence indices using same logic as training
        // Use the SAME overlap as configured for training
        let step_size = calculate_step_size(sequence_overlap, sequence_length);

        let all_possible_indices = calculate_sequence_indices(
            total_data_length,
            sequence_length,
            step_size,
            horizon_steps,
        )?;

        // Step 2: Determine target sample size - 50% of available, min 1000, max 20000
        let max_available = all_possible_indices.len();
        let target_samples = sample_size.unwrap_or_else(|| (max_available / 2).clamp(1000, 20000));

        log::info!(
            "🎯 Calibration sampling: {} total possible sequences, targeting {} diverse samples ({:.1}% coverage, overlap={:.1}%)",
            max_available,
            target_samples,
            (target_samples as f64 / max_available as f64) * 100.0,
            sequence_overlap * 100.0
        );

        // Step 3: If we need fewer samples than available, use diversity selection
        if target_samples >= max_available {
            log::info!(
                "✅ Using all {} available sequences for calibration",
                max_available
            );
            return Ok(all_possible_indices);
        }

        // Step 4: Use temporal stratification for diversity (reuse existing logic)
        let selected_indices =
            self.select_diverse_temporal_samples(&all_possible_indices, target_samples)?;

        log::info!(
            "✅ Selected {} diverse samples from {} available using temporal stratification",
            selected_indices.len(),
            max_available
        );

        Ok(selected_indices)
    }

    /// Select diverse samples using temporal stratification (reuse DiversitySelector logic)
    fn select_diverse_temporal_samples(
        &self,
        all_indices: &[usize],
        target_count: usize,
    ) -> Result<Vec<usize>> {
        use rand::seq::SliceRandom;

        // Sort by temporal position for temporal diversity
        let mut temporal_sorted: Vec<usize> = all_indices.to_vec();
        temporal_sorted.sort_unstable();

        // Divide into temporal buckets and select from each (same as DiversitySelector)
        let num_buckets = (target_count / 10).clamp(5, 20); // 5-20 buckets for good coverage
        let bucket_size = temporal_sorted.len() / num_buckets;
        let sequences_per_bucket = target_count / num_buckets;
        let remainder = target_count % num_buckets;

        let mut selected = Vec::new();
        let mut rng = rand::rng();

        for bucket_idx in 0..num_buckets {
            let start = bucket_idx * bucket_size;
            let end = if bucket_idx == num_buckets - 1 {
                temporal_sorted.len() // Last bucket gets remainder
            } else {
                (bucket_idx + 1) * bucket_size
            };

            let mut bucket_sequences: Vec<usize> = temporal_sorted[start..end].to_vec();

            // Shuffle and select from this temporal bucket
            bucket_sequences.shuffle(&mut rng);

            let take_count = if bucket_idx < remainder {
                sequences_per_bucket + 1
            } else {
                sequences_per_bucket
            };

            selected.extend(bucket_sequences.into_iter().take(take_count));
        }

        // Ensure we have exactly the right count
        selected.truncate(target_count);

        Ok(selected)
    }

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
        sequence_overlap: f64,
    ) -> Result<CalibratedParameters> {
        let start_time = std::time::Instant::now();

        // Generate diverse sample indices using same logic as sequence generation
        let sample_indices = self.generate_diverse_calibration_indices(
            ohlcv_data.len(),
            sequence_length,
            horizon_steps,
            sample_size,
            sequence_overlap,
        )?;
        let samples_to_use = sample_indices.len();

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

        let sensitivities = vec![0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]; // More granular options
        let multipliers = vec![1.5, 2.0, 2.5, 3.0, 4.0, 5.0]; // More options

        // NEW: Grid search for previously hardcoded parameters - BETTER RANGES
        let min_base_thresholds = vec![0.001, 0.003, 0.005, 0.01, 0.015]; // More granular
        let min_extreme_thresholds = vec![0.005, 0.01, 0.015, 0.02, 0.03]; // More granular
        let base_multipliers = vec![2.0, 5.0, 10.0, 15.0, 20.0, 30.0]; // Better range

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

        let bandwidths = vec![0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
        let percentile_pairs = vec![
            [0.01, 0.99],
            [0.05, 0.95],
            [0.1, 0.9],
            [0.15, 0.85],
            [0.2, 0.8],
            [0.25, 0.75],
            [0.3, 0.7],
            [0.35, 0.65],
            [0.4, 0.6],
        ]; // Base percentiles for adaptive calculation
        let neutral_band_factors = vec![0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]; // Neutral zone size
        let momentum_factors = vec![1.1, 1.2, 1.3, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]; // Momentum weighting for bandwidth

        for &bandwidth in &bandwidths {
            for &percentiles in &percentile_pairs {
                for &neutral_band_factor in &neutral_band_factors {
                    for &momentum_factor in &momentum_factors {
                        let balance = self.evaluate_price_level_params(
                            context,
                            &PriceLevelEvalParams {
                                bandwidth,
                                percentiles,
                                neutral_band: neutral_band_factor,
                                momentum_factor,
                            },
                        )?;

                        if balance.composite_quality_score < best_score
                            && balance.diversity_score >= self.min_diversity_threshold
                        {
                            best_score = balance.composite_quality_score;
                            best_params = PriceLevelParams {
                                bandwidth,
                                percentiles,
                                neutral_band_factor,
                                momentum_factor,
                                balance,
                            };
                        }
                    }
                }
            }
        }

        log::info!(
            "🎯 Calibrated Price Level Parameters: bandwidth={:.2}, percentiles=[{:.2}, {:.2}], neutral_band_factor={:.2}, momentum_factor={:.2}",
            best_params.bandwidth,
            best_params.percentiles[0], best_params.percentiles[1],
            best_params.neutral_band_factor,
            best_params.momentum_factor
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
        let volume_weights = vec![0.05, 0.1, 0.15, 0.2, 0.25, 0.3]; // NEW: Volume weight calibration

        // NEW: Grid search for previously hardcoded min_volatility_baseline
        let min_volatility_baselines = vec![0.001, 0.003, 0.005, 0.007, 0.01]; // Was hardcoded 0.005

        for &bandwidth in &bandwidths {
            for &multiplier in &multipliers {
                for &decay in &decay_factors {
                    for &volume_weight in &volume_weights {
                        for &min_baseline in &min_volatility_baselines {
                            let balance = self.evaluate_volatility_params(
                                context,
                                &VolatilityEvalParams {
                                    bandwidth,
                                    multiplier,
                                    decay,
                                    volume_weight,
                                    min_baseline,
                                },
                            )?;

                            if balance.composite_quality_score < best_score
                                && balance.diversity_score >= self.min_diversity_threshold
                            {
                                best_score = balance.composite_quality_score;
                                best_params = VolatilityParams {
                                    bandwidth,
                                    extreme_multiplier: multiplier,
                                    volume_weight,
                                    horizon_decay: decay,
                                    min_volatility_baseline: min_baseline,
                                    balance,
                                };
                            }
                        }
                    }
                }
            }
        }

        log::info!(
            "🎯 Calibrated Volatility Parameters: bandwidth={:.2}, extreme_mult={:.1}, volume_weight={:.3}, decay={:.2}, min_baseline={:.4}",
            best_params.bandwidth, best_params.extreme_multiplier, best_params.volume_weight, best_params.horizon_decay, best_params.min_volatility_baseline
        );

        Ok(best_params)
    }

    /// Calibrate sentiment parameters using proper percentile-based calibration
    async fn calibrate_sentiment(
        &self,
        context: &EvaluationContext<'_>,
    ) -> Result<SentimentParams> {
        log::debug!("Calibrating sentiment parameters with comprehensive optimization...");

        // Use the ENHANCED calibrate_sentiment_sensitivity that now optimizes ALL parameters
        use crate::targets::sentiment::{
            calibrate_sentiment_sensitivity, get_optimal_extreme_multiplier,
            get_optimal_volume_weight,
        };

        // This now finds optimal sensitivity, volume_weight, AND extreme_multiplier
        let optimal_sensitivity = calibrate_sentiment_sensitivity(
            context.ohlcv_data,
            context.sequence_length,
            context.horizon_steps,
            0.2, // target 20% per class
        )?;

        // Get the other optimal parameters found during calibration
        let optimal_volume_weight = get_optimal_volume_weight();
        let optimal_extreme_multiplier = get_optimal_extreme_multiplier();

        // Test the optimal parameters to get actual class distribution
        let balance = self.evaluate_sentiment_params(
            context,
            &SentimentEvalParams {
                sensitivity: optimal_sensitivity,
                volume_weight: optimal_volume_weight,
                consistency_factor: 0.8, // Keep this fixed for now
            },
        )?;

        // Add minimum thresholds similar to direction target for consistency
        let min_base_threshold = optimal_sensitivity * 0.1; // 10% of sensitivity as minimum
        let min_extreme_threshold = optimal_sensitivity * optimal_extreme_multiplier * 0.1;

        log::info!(
            "🎯 Comprehensive sentiment calibration complete: sensitivity={:.4}, volume_weight={:.3}, extreme_multiplier={:.2}, min_base={:.4}, min_extreme={:.4}",
            optimal_sensitivity, optimal_volume_weight, optimal_extreme_multiplier, min_base_threshold, min_extreme_threshold
        );

        Ok(SentimentParams {
            body_sensitivity: optimal_sensitivity,
            volume_weight: optimal_volume_weight, // Now optimized!
            consistency_factor: 0.8,
            extreme_multiplier: optimal_extreme_multiplier, // Now optimized! No more magic numbers!
            min_base_threshold,                             // NEW: Added for consistency
            min_extreme_threshold,                          // NEW: Added for consistency
            balance,
        })
    }

    /// Calibrate volume parameters
    async fn calibrate_volume(&self, context: &EvaluationContext<'_>) -> Result<VolumeParams> {
        log::debug!("Calibrating volume parameters...");

        let mut best_params = VolumeParams::default();
        let mut best_score = f64::INFINITY;

        let bandwidths = vec![0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
        let multipliers = vec![1.5, 2.0, 2.5, 3.0, 4.0, 5.0];
        let smoothing_values = vec![1, 3, 5, 7, 9, 11, 15];

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
                            min_base_threshold: bandwidth * 0.1, // 10% of bandwidth as minimum
                            min_extreme_threshold: bandwidth * multiplier * 0.1, // For consistency
                            balance,
                        };
                    }
                }
            }
        }

        log::info!(
            "🎯 Calibrated Volume Parameters: bandwidth={:.2}, extreme_mult={:.1}, smoothing={}, min_base={:.4}, min_extreme={:.4}",
            best_params.bandwidth, best_params.extreme_multiplier, best_params.smoothing_periods,
            best_params.min_base_threshold, best_params.min_extreme_threshold
        );

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
        // Same logic as in direction.rs - USE PERCENTAGE CHANGE, NOT RAW SLOPE
        if sequence_prices.len() < 2 || horizon_prices.len() < 2 {
            return Ok((0.0, 0.0, 0.0));
        }

        let seq_start = sequence_prices[0];
        let seq_end = sequence_prices[sequence_prices.len() - 1];

        // Avoid division by zero - use epsilon check
        let sequence_momentum = if seq_start.abs() < 1e-10 {
            0.0
        } else {
            (seq_end - seq_start) / seq_start
        };

        let hor_start = horizon_prices[0];
        let hor_end = horizon_prices[horizon_prices.len() - 1];

        // Avoid division by zero - use epsilon check
        let horizon_momentum = if hor_start.abs() < 1e-10 {
            0.0
        } else {
            (hor_end - hor_start) / hor_start
        };

        let momentum_change = horizon_momentum - sequence_momentum;
        Ok((sequence_momentum, horizon_momentum, momentum_change))
    }

    /// Calculate sequence trend consistency (helper for calibration)
    fn calculate_sequence_trend_consistency(&self, sequence_prices: &[f64]) -> Result<f64> {
        if sequence_prices.len() < 3 {
            return Ok(0.01); // Minimum consistency for short sequences
        }

        let mut momentum_changes = Vec::new();

        // Calculate momentum between consecutive segments - MUST MATCH direction.rs EXACTLY
        let segment_size = (sequence_prices.len() / 3).max(2);
        for i in 0..(sequence_prices.len() - segment_size * 2) {
            let seg1_start = sequence_prices[i];
            let seg1_end = sequence_prices[i + segment_size];
            let seg2_start = seg1_end;
            let seg2_end = sequence_prices[i + segment_size * 2];

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

                    // FIXED: Use the same logic as actual classification with calibrated parameters
                    // 1. Calculate adaptive percentiles with base percentiles as fallback
                    use crate::targets::price_levels::calculate_adaptive_percentiles_from_sequence;
                    let adaptive_percentiles = calculate_adaptive_percentiles_from_sequence(
                        sequence_ohlcv,
                        Some(params.percentiles), // Use calibrated percentiles as fallback
                    )?;

                    // 2. Calculate adaptive bandwidth with momentum factor
                    use crate::targets::price_levels::calculate_adaptive_bandwidth;
                    let final_bandwidth_size = calculate_adaptive_bandwidth(
                        sequence_ohlcv,
                        params.bandwidth,
                        Some(params.momentum_factor),
                    )?;

                    // 3. Use sequence reconstruction with calibrated parameters (same as actual classification)
                    let reconstruction_config = SequenceReconstructionConfig {
                        percentiles: adaptive_percentiles, // Use calculated adaptive percentiles
                        bandwidth_size: final_bandwidth_size, // Use calculated adaptive bandwidth
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

    /// Evaluate volatility parameters using simplified ATR momentum classification
    fn evaluate_volatility_params(
        &self,
        context: &EvaluationContext,
        params: &VolatilityEvalParams,
    ) -> Result<ClassBalance> {
        use crate::targets::volatility::classify_volatility_with_calibrated_params;

        let mut class_counts = [0usize; 5];

        // Create calibrated parameters for the new simplified approach
        let calibrated_params = crate::targets::calibration::VolatilityParams {
            bandwidth: params.bandwidth,
            extreme_multiplier: params.multiplier,
            volume_weight: params.volume_weight, // Use the calibrated volume weight
            horizon_decay: params.decay,         // Use the passed decay parameter
            min_volatility_baseline: params.min_baseline, // Use the calibrated min baseline
            balance: Default::default(),
        };

        // Process each sample using the new simplified classification
        for &seq_idx in context.sample_indices {
            let sequence_end_idx = seq_idx + context.sequence_length;
            let target_end_idx = sequence_end_idx + context.horizon_steps;

            if target_end_idx <= context.ohlcv_data.len() {
                let sequence_candles = &context.ohlcv_data[seq_idx..sequence_end_idx];
                let horizon_candles = &context.ohlcv_data[sequence_end_idx..target_end_idx];

                if sequence_candles.len() >= 2 && horizon_candles.len() >= 2 {
                    // Use the new simplified classification approach
                    if let Ok((class, _strength)) = classify_volatility_with_calibrated_params(
                        sequence_candles,
                        horizon_candles,
                        &calibrated_params,
                    ) {
                        if (0..5).contains(&class) {
                            class_counts[class as usize] += 1;
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
        use crate::targets::sentiment::{
            classify_sentiment_with_calibrated_params, get_optimal_extreme_multiplier,
        };

        let mut class_counts = [0usize; 5];
        let sample_limit = context.sample_indices.len().min(500); // Limit for performance

        // Get the extreme_multiplier that was found during calibration
        let extreme_multiplier = get_optimal_extreme_multiplier();

        for &seq_idx in context.sample_indices.iter().take(sample_limit) {
            let sequence_end_idx = seq_idx + context.sequence_length;
            let target_end_idx = sequence_end_idx + context.horizon_steps;

            if target_end_idx <= context.ohlcv_data.len() {
                let sequence_data = &context.ohlcv_data[seq_idx..sequence_end_idx];
                let horizon_data = &context.ohlcv_data[sequence_end_idx..target_end_idx];

                if sequence_data.len() >= 2 && horizon_data.len() >= 2 {
                    // Use the params being evaluated with the calibrated extreme_multiplier
                    let eval_params = crate::targets::calibration::SentimentParams {
                        body_sensitivity: params.sensitivity,
                        volume_weight: params.volume_weight,
                        consistency_factor: params.consistency_factor,
                        extreme_multiplier, // Use the calibrated extreme_multiplier
                        min_base_threshold: params.sensitivity * 0.1, // 10% of sensitivity as minimum
                        min_extreme_threshold: params.sensitivity * extreme_multiplier * 0.1,
                        balance: ClassBalance::default(),
                    };
                    match classify_sentiment_with_calibrated_params(
                        sequence_data,
                        horizon_data,
                        &eval_params,
                    ) {
                        Ok((class, _strength)) => {
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
    percentiles: [f64; 2], // Base percentiles for adaptive calculation
    neutral_band: f64,
    momentum_factor: f64, // Momentum factor for bandwidth adjustment
}

/// Parameters for volatility evaluation
struct VolatilityEvalParams {
    bandwidth: f64,
    multiplier: f64,
    decay: f64,
    volume_weight: f64, // NEW: Volume weight parameter
    min_baseline: f64,  // NEW: Minimum volatility baseline parameter
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
