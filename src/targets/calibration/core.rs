//! Calibration Core Module
//!
//! Contains the main ParameterCalibrator struct and orchestration logic.
//! Handles diversity sampling, calibration coordination, and result logging.

use super::types::*;
use super::utils::CalibrationUtils;
use crate::data::structures::MarketDataRow;
use crate::utils::error::Result;

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

    /// Get utility helper for balance calculations
    pub fn get_utils(&self) -> CalibrationUtils {
        CalibrationUtils::new(
            self.balance_weight,
            self.diversity_weight,
            self.target_balance,
        )
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

// Forward declarations for calibration methods (implemented in separate modules)
impl ParameterCalibrator {
    pub async fn calibrate_direction(
        &self,
        ohlcv_data: &[MarketDataRow],
        sequence_length: usize,
        horizon_steps: usize,
        sample_indices: &[usize],
    ) -> Result<DirectionParams> {
        super::direction::calibrate_direction(
            self,
            ohlcv_data,
            sequence_length,
            horizon_steps,
            sample_indices,
        )
        .await
    }

    pub async fn calibrate_price_levels(
        &self,
        context: &EvaluationContext<'_>,
    ) -> Result<PriceLevelParams> {
        super::price_levels::calibrate_price_levels(self, context).await
    }

    pub async fn calibrate_volatility(
        &self,
        context: &EvaluationContext<'_>,
    ) -> Result<VolatilityParams> {
        super::volatility::calibrate_volatility(self, context).await
    }

    pub async fn calibrate_sentiment(
        &self,
        context: &EvaluationContext<'_>,
    ) -> Result<SentimentParams> {
        super::sentiment::calibrate_sentiment(self, context).await
    }

    pub async fn calibrate_volume(&self, context: &EvaluationContext<'_>) -> Result<VolumeParams> {
        super::volume::calibrate_volume(self, context).await
    }
}
