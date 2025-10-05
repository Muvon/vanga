//! Calibration Core Module
//!
//! Contains the main ParameterCalibrator struct and orchestration logic.
//! Handles diversity sampling, calibration coordination, and result logging.

use super::types::*;
use super::utils::CalibrationUtils;
use crate::data::structures::MarketDataRow;
use crate::utils::error::Result;
use std::sync::Arc;

/// Target parameter calibrator - single clean interface with diversity optimization
#[derive(Clone)]
pub struct ParameterCalibrator {
    target_balance: f64,
    max_iterations: usize,

    // NEW: Diversity optimization weights
    balance_weight: f64,   // Weight for class balance (default: 0.6)
    diversity_weight: f64, // Weight for sample diversity (default: 0.4)
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

        // Step 2: Use ALL available samples for QUALITY calibration (no artificial limits)
        // Only limit if explicitly requested via sample_size parameter
        let max_available = all_possible_indices.len();
        let target_samples = sample_size.unwrap_or(max_available); // Use ALL samples by default

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

    /// Select diverse samples using UNIFORM temporal stratification for maximum diversity
    fn select_diverse_temporal_samples(
        &self,
        all_indices: &[usize],
        target_count: usize,
    ) -> Result<Vec<usize>> {
        if target_count >= all_indices.len() {
            return Ok(all_indices.to_vec());
        }

        // Sort by temporal position
        let mut temporal_sorted: Vec<usize> = all_indices.to_vec();
        temporal_sorted.sort_unstable();

        // Use UNIFORM sampling for maximum temporal spread
        // This ensures samples are evenly distributed across time
        let step = all_indices.len() as f64 / target_count as f64;
        let mut selected = Vec::with_capacity(target_count);

        for i in 0..target_count {
            let idx = (i as f64 * step) as usize;
            if idx < temporal_sorted.len() {
                selected.push(temporal_sorted[idx]);
            }
        }

        log::debug!(
            "Uniform temporal sampling: selected {} samples with step size {:.2}",
            selected.len(),
            step
        );

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
        }
    }

    /// Create calibrator with custom diversity threshold (deprecated - kept for compatibility)
    pub fn with_diversity_threshold(_threshold: f64) -> Self {
        Self::default()
    }

    /// Create calibrator with full customization
    pub fn with_custom_config(
        balance_weight: f64,
        diversity_weight: f64,
        _min_threshold: f64,
    ) -> Self {
        Self::with_diversity_weights(balance_weight, diversity_weight)
    }

    /// Validate sample quality to ensure diverse, representative calibration data
    fn validate_sample_quality(
        &self,
        ohlcv_data: &[MarketDataRow],
        sample_indices: &[usize],
    ) -> Result<()> {
        use super::utils::CalibrationUtils;

        log::info!("🔍 Validating calibration sample quality...");

        // 1. Temporal coverage check
        let temporal_diversity = CalibrationUtils::calculate_temporal_diversity(sample_indices);
        log::info!(
            "  📅 Temporal diversity: {:.2}%",
            temporal_diversity * 100.0
        );

        if temporal_diversity < 0.5 {
            log::warn!(
                "  ⚠️  Low temporal diversity ({:.1}%) - samples may be clustered in time",
                temporal_diversity * 100.0
            );
        }

        // 2. Feature space coverage check
        let feature_diversity =
            CalibrationUtils::calculate_feature_space_diversity(ohlcv_data, sample_indices);
        log::info!(
            "  📊 Feature space diversity: {:.2}%",
            feature_diversity * 100.0
        );

        if feature_diversity < 0.5 {
            log::warn!(
                "  ⚠️  Low feature diversity ({:.1}%) - samples may not cover full price/volatility range",
                feature_diversity * 100.0
            );
        }

        // 3. Market condition balance check
        let market_diversity =
            CalibrationUtils::calculate_market_condition_diversity(ohlcv_data, sample_indices);
        log::info!(
            "  🎯 Market condition diversity: {:.2}%",
            market_diversity * 100.0
        );

        if market_diversity < 0.5 {
            log::warn!(
                "  ⚠️  Low market condition diversity ({:.1}%) - samples may be biased toward bull/bear/sideways",
                market_diversity * 100.0
            );
        }

        // 4. Overall quality assessment
        let overall_quality = (temporal_diversity + feature_diversity + market_diversity) / 3.0;
        log::info!(
            "  ✅ Overall sample quality: {:.2}%",
            overall_quality * 100.0
        );

        if overall_quality < 0.6 {
            log::warn!(
                "  ⚠️  Sample quality below recommended threshold (60%) - calibration may be suboptimal"
            );
        } else if overall_quality >= 0.8 {
            log::info!("  🌟 Excellent sample quality - calibration should be highly reliable");
        }

        Ok(())
    }

    /// PER-HORIZON PARALLEL calibration method - returns parameters for ALL targets × ALL horizons
    ///
    /// This is the main entry point for parameter calibration. It analyzes the provided
    /// OHLCV data and finds optimal parameters for all target types (direction, price levels,
    /// volatility, sentiment, volume) FOR EACH HORIZON SEPARATELY.
    ///
    /// **PARALLELIZATION**:
    /// - Level 1: All horizons calibrated in parallel
    /// - Level 2: Within each horizon, all 5 targets calibrated in parallel
    /// - Uses all available CPU cores for maximum performance
    ///
    /// # Arguments
    /// * `ohlcv_data` - Market data for calibration analysis
    /// * `sequence_length` - Length of input sequences for the model
    /// * `horizons` - All prediction horizons (e.g., ["1h", "4h", "24h"])
    /// * `sample_size` - Optional limit on samples to use (default: all available)
    ///
    /// # Returns
    /// * `CalibratedParameters` - Optimized parameters per horizon for all target types
    ///
    /// # Algorithm
    /// 1. For each horizon IN PARALLEL:
    ///    a. Generate diverse sample indices
    ///    b. Calibrate all 5 targets IN PARALLEL
    ///    c. Store parameters in HashMap<horizon, params>
    /// 2. Returns comprehensive results with per-horizon metadata
    pub async fn calibrate(
        &self,
        ohlcv_data: &[MarketDataRow],
        sequence_length: usize,
        horizons: &[String],
        sample_size: Option<usize>,
        sequence_overlap: f64,
    ) -> Result<CalibratedParameters> {
        let total_start = std::time::Instant::now();
        let num_cpus = num_cpus::get();

        log::info!(
            "🎯 Starting PARALLEL PER-HORIZON calibration for {} horizons: {:?}",
            horizons.len(),
            horizons
        );
        log::info!(
            "⚡ Parallelization: {} CPU cores available, {} horizons × 5 targets = {} parallel tasks",
            num_cpus,
            horizons.len(),
            horizons.len() * 5
        );

        // Wrap data in Arc for safe sharing across threads
        let ohlcv_data_arc = Arc::new(ohlcv_data.to_vec());

        // Calibrate all horizons in parallel
        let horizon_futures: Vec<_> = horizons
            .iter()
            .enumerate()
            .map(|(horizon_idx, horizon)| {
                let horizon = horizon.clone();
                let ohlcv_data = Arc::clone(&ohlcv_data_arc);
                let calibrator = self.clone();

                async move {
                    let horizon_start = std::time::Instant::now();
                    let prefix = format!("[H{}/{}:{}]", horizon_idx + 1, horizons.len(), horizon);

                    log::info!(
                        "{} 🕐 Starting horizon calibration",
                        prefix
                    );

                    // Parse horizon to steps
                    let horizon_steps = crate::utils::parser::parse_horizon_to_steps(&horizon)
                        .map_err(|e| {
                            crate::utils::error::VangaError::ConfigError(format!(
                                "Invalid horizon '{}': {}",
                                horizon, e
                            ))
                        })?;

                    // Generate diverse sample indices for this horizon
                    let sample_indices = calibrator.generate_diverse_calibration_indices(
                        ohlcv_data.len(),
                        sequence_length,
                        horizon_steps,
                        sample_size,
                        sequence_overlap,
                    )?;

                    log::info!(
                        "{} 📊 Calibrating with {} samples",
                        prefix,
                        sample_indices.len()
                    );

                    // Validate sample quality
                    calibrator.validate_sample_quality(&ohlcv_data, &sample_indices)?;

                    // Calibrate all 5 targets IN PARALLEL for this horizon using REAL CPU parallelization
                    log::info!("{} ⚡ Calibrating 5 targets in PARALLEL on separate CPU threads", prefix);

                    // Clone data for each parallel task
                    let ohlcv_clone1 = ohlcv_data.clone();
                    let ohlcv_clone2 = ohlcv_data.clone();
                    let ohlcv_clone3 = ohlcv_data.clone();
                    let ohlcv_clone4 = ohlcv_data.clone();
                    let ohlcv_clone5 = ohlcv_data.clone();

                    let indices_clone1 = sample_indices.clone();
                    let indices_clone2 = sample_indices.clone();
                    let indices_clone3 = sample_indices.clone();
                    let indices_clone4 = sample_indices.clone();
                    let indices_clone5 = sample_indices.clone();

                    let calibrator1 = calibrator.clone();
                    let calibrator2 = calibrator.clone();
                    let calibrator3 = calibrator.clone();
                    let calibrator4 = calibrator.clone();
                    let calibrator5 = calibrator.clone();

                    let prefix1 = prefix.clone();
                    let prefix2 = prefix.clone();
                    let prefix3 = prefix.clone();
                    let prefix4 = prefix.clone();
                    let prefix5 = prefix.clone();

                    // Spawn BLOCKING tasks for CPU-intensive work (runs on separate OS threads)
                    let direction_handle = tokio::task::spawn_blocking(move || {
                        log::info!("{} [Direction] 🚀 Starting Bayesian optimization on CPU thread...", prefix1);
                        let rt = tokio::runtime::Handle::current();
                        let result = rt.block_on(calibrator1.calibrate_direction(
                            &ohlcv_clone1,
                            sequence_length,
                            horizon_steps,
                            &indices_clone1,
                            &prefix1,
                        ));
                        match &result {
                            Ok(params) => log::info!(
                                "{} [Direction] ✅ Complete - score: {:.3}",
                                prefix1,
                                params.balance.composite_quality_score
                            ),
                            Err(e) => log::error!("{} [Direction] ❌ Failed: {}", prefix1, e),
                        }
                        result
                    });

                    let price_handle = tokio::task::spawn_blocking(move || {
                        log::info!("{} [PriceLevels] 🚀 Starting Bayesian optimization on CPU thread...", prefix2);
                        let context = EvaluationContext {
                            ohlcv_data: &ohlcv_clone2,
                            sample_indices: &indices_clone2,
                            sequence_length,
                            horizon_steps,
                        };
                        let rt = tokio::runtime::Handle::current();
                        let result = rt.block_on(calibrator2.calibrate_price_levels(&context, &prefix2));
                        match &result {
                            Ok(params) => log::info!(
                                "{} [PriceLevels] ✅ Complete - score: {:.3}",
                                prefix2,
                                params.balance.composite_quality_score
                            ),
                            Err(e) => log::error!("{} [PriceLevels] ❌ Failed: {}", prefix2, e),
                        }
                        result
                    });

                    let volatility_handle = tokio::task::spawn_blocking(move || {
                        log::info!("{} [Volatility] 🚀 Starting Bayesian optimization on CPU thread...", prefix3);
                        let context = EvaluationContext {
                            ohlcv_data: &ohlcv_clone3,
                            sample_indices: &indices_clone3,
                            sequence_length,
                            horizon_steps,
                        };
                        let rt = tokio::runtime::Handle::current();
                        let result = rt.block_on(calibrator3.calibrate_volatility(&context, &prefix3));
                        match &result {
                            Ok(params) => log::info!(
                                "{} [Volatility] ✅ Complete - score: {:.3}",
                                prefix3,
                                params.balance.composite_quality_score
                            ),
                            Err(e) => log::error!("{} [Volatility] ❌ Failed: {}", prefix3, e),
                        }
                        result
                    });

                    let sentiment_handle = tokio::task::spawn_blocking(move || {
                        log::info!("{} [Sentiment] 🚀 Starting Bayesian optimization on CPU thread...", prefix4);
                        let context = EvaluationContext {
                            ohlcv_data: &ohlcv_clone4,
                            sample_indices: &indices_clone4,
                            sequence_length,
                            horizon_steps,
                        };
                        let rt = tokio::runtime::Handle::current();
                        let result = rt.block_on(calibrator4.calibrate_sentiment(&context, &prefix4));
                        match &result {
                            Ok(params) => log::info!(
                                "{} [Sentiment] ✅ Complete - score: {:.3}",
                                prefix4,
                                params.balance.composite_quality_score
                            ),
                            Err(e) => log::error!("{} [Sentiment] ❌ Failed: {}", prefix4, e),
                        }
                        result
                    });

                    let volume_handle = tokio::task::spawn_blocking(move || {
                        log::info!("{} [Volume] 🚀 Starting Bayesian optimization on CPU thread...", prefix5);
                        let context = EvaluationContext {
                            ohlcv_data: &ohlcv_clone5,
                            sample_indices: &indices_clone5,
                            sequence_length,
                            horizon_steps,
                        };
                        let rt = tokio::runtime::Handle::current();
                        let result = rt.block_on(calibrator5.calibrate_volume(&context, &prefix5));
                        match &result {
                            Ok(params) => log::info!(
                                "{} [Volume] ✅ Complete - score: {:.3}",
                                prefix5,
                                params.balance.composite_quality_score
                            ),
                            Err(e) => log::error!("{} [Volume] ❌ Failed: {}", prefix5, e),
                        }
                        result
                    });

                    // Wait for all CPU threads to complete
                    let (direction_res, price_res, volatility_res, sentiment_res, volume_res) = tokio::join!(
                        direction_handle,
                        price_handle,
                        volatility_handle,
                        sentiment_handle,
                        volume_handle
                    );

                    // Unwrap JoinHandle results and propagate errors
                    let direction = direction_res.map_err(|e| {
                        crate::utils::error::VangaError::OptimizationError(format!("Direction task failed: {}", e))
                    })??;
                    let price_levels = price_res.map_err(|e| {
                        crate::utils::error::VangaError::OptimizationError(format!("PriceLevels task failed: {}", e))
                    })??;
                    let volatility = volatility_res.map_err(|e| {
                        crate::utils::error::VangaError::OptimizationError(format!("Volatility task failed: {}", e))
                    })??;
                    let sentiment = sentiment_res.map_err(|e| {
                        crate::utils::error::VangaError::OptimizationError(format!("Sentiment task failed: {}", e))
                    })??;
                    let volume = volume_res.map_err(|e| {
                        crate::utils::error::VangaError::OptimizationError(format!("Volume task failed: {}", e))
                    })??;

                    // Calculate horizon score
                    let horizon_score = (direction.balance.composite_quality_score
                        + price_levels.balance.composite_quality_score
                        + volatility.balance.composite_quality_score
                        + sentiment.balance.composite_quality_score
                        + volume.balance.composite_quality_score)
                        / 5.0;

                    let horizon_time = horizon_start.elapsed().as_millis() as u64;

                    log::info!(
                        "{} ✅ Horizon calibrated in {}ms (score: {:.3})",
                        prefix,
                        horizon_time,
                        horizon_score
                    );

                    Ok::<_, crate::utils::error::VangaError>((
                        horizon,
                        direction,
                        price_levels,
                        volatility,
                        sentiment,
                        volume,
                        horizon_score,
                        horizon_time,
                    ))
                }
            })
            .collect();

        // Wait for all horizons to complete
        let results = futures::future::try_join_all(horizon_futures).await?;

        // Collect results into HashMaps
        let mut direction_params = std::collections::HashMap::new();
        let mut price_level_params = std::collections::HashMap::new();
        let mut volatility_params = std::collections::HashMap::new();
        let mut sentiment_params = std::collections::HashMap::new();
        let mut volume_params = std::collections::HashMap::new();
        let mut overall_scores = Vec::new();
        let mut total_optimization_time = 0u64;

        for (horizon, direction, price_levels, volatility, sentiment, volume, score, time) in
            results
        {
            direction_params.insert(horizon.clone(), direction);
            price_level_params.insert(horizon.clone(), price_levels);
            volatility_params.insert(horizon.clone(), volatility);
            sentiment_params.insert(horizon.clone(), sentiment);
            volume_params.insert(horizon.clone(), volume);
            overall_scores.push(score);
            total_optimization_time += time;
        }

        // Calculate overall statistics
        let overall_score = overall_scores.iter().sum::<f64>() / overall_scores.len() as f64;
        let success = overall_score < 1.0;
        let total_elapsed = total_start.elapsed().as_millis() as u64;
        let speedup = if horizons.len() > 1 {
            total_optimization_time as f64 / total_elapsed as f64
        } else {
            1.0
        };

        let metadata = CalibrationMetadata {
            data_length: ohlcv_data.len(),
            sequence_length,
            horizons: horizons.to_vec(),
            calibration_samples: 0, // Will be set per-horizon
            calibration_iterations: self.max_iterations,
            optimization_time_ms: total_elapsed, // Use actual wall-clock time
            target_balance: self.target_balance,
            overall_balance_score: overall_score,
            calibration_success: success,
        };

        log::info!("\n{}", "=".repeat(60));
        log::info!("🎯 PARALLEL PER-HORIZON CALIBRATION COMPLETE");
        log::info!("{}", "=".repeat(60));
        log::info!("  Wall-clock time: {}ms", total_elapsed);
        log::info!("  CPU time (sum): {}ms", total_optimization_time);
        log::info!("  Speedup: {:.2}x", speedup);
        log::info!("  Overall score: {:.3}", overall_score);
        log::info!("  Success: {}", if success { "✅" } else { "❌" });
        log::info!("{}\n", "=".repeat(60));

        Ok(CalibratedParameters {
            direction: direction_params,
            price_levels: price_level_params,
            volatility: volatility_params,
            sentiment: sentiment_params,
            volume: volume_params,
            metadata,
        })
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
        prefix: &str,
    ) -> Result<DirectionParams> {
        super::direction::calibrate_direction(
            self,
            ohlcv_data,
            sequence_length,
            horizon_steps,
            sample_indices,
            prefix,
        )
        .await
    }

    pub async fn calibrate_price_levels(
        &self,
        context: &EvaluationContext<'_>,
        prefix: &str,
    ) -> Result<PriceLevelParams> {
        super::price_levels::calibrate_price_levels(self, context, prefix).await
    }

    pub async fn calibrate_volatility(
        &self,
        context: &EvaluationContext<'_>,
        prefix: &str,
    ) -> Result<VolatilityParams> {
        super::volatility::calibrate_volatility(self, context, prefix).await
    }

    pub async fn calibrate_sentiment(
        &self,
        context: &EvaluationContext<'_>,
        prefix: &str,
    ) -> Result<SentimentParams> {
        super::sentiment::calibrate_sentiment(self, context, prefix).await
    }

    pub async fn calibrate_volume(
        &self,
        context: &EvaluationContext<'_>,
        prefix: &str,
    ) -> Result<VolumeParams> {
        super::volume::calibrate_volume(self, context, prefix).await
    }

    /// Calibrate parameters using Bayesian Optimization
    ///
    /// This method uses Gaussian Process-based Bayesian Optimization to find
    /// optimal parameters efficiently. It's significantly faster than grid search
    /// and finds better parameters by modeling the objective function.
    ///
    /// # Arguments
    /// * `param_bounds` - Min/max bounds for each parameter
    /// * `param_names` - Names for logging
    /// * `objective_fn` - Function to minimize (returns score, lower is better)
    /// * `config` - Bayesian optimization configuration
    /// * `prefix` - Log prefix for parallel execution tracking
    ///
    /// # Returns
    /// Best parameters found
    pub async fn calibrate_with_bayesian<F>(
        &self,
        param_bounds: Vec<(f64, f64)>,
        param_names: Vec<String>,
        objective_fn: F,
        config: super::bayesian::BayesianConfig,
        prefix: &str,
    ) -> Result<Vec<f64>>
    where
        F: Fn(&[f64]) -> Result<f64>,
    {
        use super::bayesian::BayesianOptimizer;

        log::info!(
            "{} 🔬 Starting Bayesian Optimization with {} parameters",
            prefix,
            param_names.len()
        );
        log::info!(
            "{}    Initial samples: {}, Max iterations: {}, Tolerance: {:.6}",
            prefix,
            config.n_initial,
            config.max_iterations,
            config.tolerance
        );

        let mut optimizer = BayesianOptimizer::new(param_bounds, param_names, &config);

        // Phase 1: Initial random exploration (Latin Hypercube Sampling)
        log::info!(
            "{} 📊 Phase 1: Initial exploration ({} samples)",
            prefix,
            config.n_initial
        );
        let initial_samples = optimizer.initialize_latin_hypercube(config.n_initial, prefix);

        for (i, params) in initial_samples.iter().enumerate() {
            let score = objective_fn(params)?;
            optimizer.add_observation(params.clone(), score);
            log::debug!(
                "{}   Sample {}/{}: score = {:.6}",
                prefix,
                i + 1,
                config.n_initial,
                score
            );
        }

        if let Some((_, best_score)) = optimizer.get_best() {
            log::info!(
                "{}   Initial best score: {:.6} (from {} samples)",
                prefix,
                best_score,
                config.n_initial
            );
        }

        // Phase 2: Bayesian optimization iterations with SMART convergence
        log::info!(
            "{} 📊 Phase 2: Bayesian optimization (up to {} iterations)",
            prefix,
            config.max_iterations
        );

        let mut prev_best_score = f64::INFINITY;
        let mut no_improvement_count = 0;
        let max_patience = 15; // Increased from 5 for quality
        let n_params = optimizer.param_names.len();
        let min_iterations = if n_params >= 6 { 50 } else { 30 }; // Minimum iterations based on dimensionality

        for iteration in 0..config.max_iterations {
            // Suggest next point to evaluate
            let next_params = optimizer.suggest_next()?;

            // Evaluate objective function
            let score = objective_fn(&next_params)?;
            optimizer.add_observation(next_params, score);

            // Log progress every 5 iterations
            if (iteration + 1) % 5 == 0 || iteration == 0 {
                optimizer.log_progress(optimizer.n_observations(), prefix);
            }

            // Check convergence (but enforce minimum iterations)
            if iteration + 1 >= min_iterations {
                if let Some((_, best_score)) = optimizer.get_best() {
                    let absolute_improvement = prev_best_score - best_score;
                    let relative_improvement = if prev_best_score.abs() > 1e-10 {
                        absolute_improvement / prev_best_score.abs()
                    } else {
                        absolute_improvement
                    };

                    // Check multiple convergence criteria
                    let absolute_converged = absolute_improvement < config.tolerance;
                    let relative_converged = relative_improvement < 0.001; // 0.1% relative improvement

                    if absolute_converged && relative_converged {
                        no_improvement_count += 1;

                        if no_improvement_count >= max_patience {
                            log::info!(
                                "{} ✅ Converged after {} iterations (no improvement for {} iterations)",
                                prefix,
                                iteration + 1,
                                max_patience
                            );
                            log::info!(
                                "{}    Final score: {:.6}, Absolute improvement: {:.6}, Relative: {:.4}%",
                                prefix,
                                best_score,
                                absolute_improvement,
                                relative_improvement * 100.0
                            );
                            break;
                        }
                    } else {
                        no_improvement_count = 0; // Reset counter on improvement

                        if absolute_improvement > 0.0 {
                            log::debug!(
                                "{}   Improvement: {:.6} ({:.4}% relative)",
                                prefix,
                                absolute_improvement,
                                relative_improvement * 100.0
                            );
                        }
                    }

                    prev_best_score = best_score;
                }
            } else {
                // Still in minimum iteration phase
                if let Some((_, best_score)) = optimizer.get_best() {
                    prev_best_score = best_score;
                }
            }
        }

        // Return best parameters found
        if let Some((best_params, best_score)) = optimizer.get_best() {
            log::info!("{} 🎯 Bayesian Optimization Complete!", prefix);
            log::info!(
                "{}   Total evaluations: {}",
                prefix,
                optimizer.n_observations()
            );
            log::info!("{}   Best Score: {:.6}", prefix, best_score);
            log::info!("{}   Best Parameters:", prefix);
            for (name, &value) in optimizer.param_names.iter().zip(best_params.iter()) {
                log::info!("{}     {}: {:.6}", prefix, name, value);
            }

            Ok(best_params)
        } else {
            Err(crate::utils::error::VangaError::ConfigError(format!(
                "{} Bayesian optimization failed to find any valid parameters",
                prefix
            )))
        }
    }
}
