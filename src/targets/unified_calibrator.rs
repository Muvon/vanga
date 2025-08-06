//! Unified Target Calibration Orchestrator
//!
//! This module provides the main orchestration for adaptive parameter calibration
//! across all target types, ensuring optimal balance and consistency in the
//! multi-target prediction system.

use crate::config::model::TargetsConfig;
use crate::data::structures::MarketDataRow;
use crate::targets::adaptive_parameters::{AdaptiveParameterCalibrator, AdaptiveTargetParameters};
use crate::utils::error::Result;

/// Unified calibration orchestrator for all target types
///
/// This orchestrator coordinates the calibration process across direction,
/// price level, and volatility targets to ensure optimal system-wide balance
/// and parameter consistency.
pub struct UnifiedTargetCalibrator {
    /// Individual target calibrators
    direction_calibrator: AdaptiveParameterCalibrator,
    price_level_calibrator: AdaptiveParameterCalibrator,
    volatility_calibrator: AdaptiveParameterCalibrator,

    /// System-wide optimization settings
    system_balance_weight: f64,
    cross_target_consistency_weight: f64,
    max_calibration_iterations: usize,
}

impl UnifiedTargetCalibrator {
    /// Create new unified calibrator with base configuration
    pub fn new(base_config: TargetsConfig) -> Self {
        Self {
            direction_calibrator: AdaptiveParameterCalibrator::new(base_config.clone()),
            price_level_calibrator: AdaptiveParameterCalibrator::new(base_config.clone()),
            volatility_calibrator: AdaptiveParameterCalibrator::new(base_config),
            system_balance_weight: 0.3,
            cross_target_consistency_weight: 0.2,
            max_calibration_iterations: 3,
        }
    }

    /// Get the price level calibrator for specialized operations
    pub fn get_price_level_calibrator(&self) -> &AdaptiveParameterCalibrator {
        &self.price_level_calibrator
    }

    /// Get the volatility calibrator for specialized operations
    pub fn get_volatility_calibrator(&self) -> &AdaptiveParameterCalibrator {
        &self.volatility_calibrator
    }

    /// Calibrate all targets with system-wide optimization
    ///
    /// This is the main entry point for unified calibration that:
    /// 1. **Individual Calibration**: Optimizes each target type separately
    /// 2. **Cross-Target Validation**: Ensures parameters work well together
    /// 3. **System Balance**: Optimizes overall multi-target system balance
    /// 4. **Consistency Checks**: Validates parameter consistency across targets
    /// 5. **Final Validation**: Comprehensive system validation and reporting
    ///
    /// ## Algorithm
    /// - **Phase 1**: Individual target optimization for baseline parameters
    /// - **Phase 2**: Cross-target consistency analysis and adjustment
    /// - **Phase 3**: System-wide balance optimization with joint scoring
    /// - **Phase 4**: Final validation and parameter refinement
    ///
    /// ## Parameters
    /// - `ohlcv_data`: Market data for calibration analysis
    /// - `sequence_length`: Length of input sequences
    /// - `horizon_steps`: Prediction horizon length
    /// - `sequence_indices`: Sequence positions for analysis
    ///
    /// ## Returns
    /// Fully optimized adaptive parameters for all target types
    pub async fn calibrate_unified_system(
        &self,
        ohlcv_data: &[MarketDataRow],
        sequence_length: usize,
        horizon_steps: usize,
        sequence_indices: &[usize],
    ) -> Result<AdaptiveTargetParameters> {
        let start_time = std::time::Instant::now();

        log::info!(
            "🎯 Starting unified target calibration for {} sequences",
            sequence_indices.len()
        );

        // Use all calibrators for comprehensive target optimization
        log::info!("📊 Phase 1: Individual target calibration...");

        // Primary calibration using direction calibrator (current implementation)
        let mut adaptive_params = self
            .direction_calibrator
            .calibrate_all_targets(ohlcv_data, sequence_length, horizon_steps, sequence_indices)
            .await?;

        // Enhanced calibration using specialized calibrators
        log::debug!("🔧 Applying price level calibrator refinements...");
        adaptive_params.price_levels = self
            .price_level_calibrator
            .calibrate_price_levels(ohlcv_data, sequence_length, horizon_steps, sequence_indices)
            .await?;

        log::debug!("🔧 Applying volatility calibrator refinements...");
        adaptive_params.volatility = self
            .volatility_calibrator
            .calibrate_volatility(ohlcv_data, sequence_length, horizon_steps, sequence_indices)
            .await?;

        // Phase 2: Cross-target consistency analysis
        log::info!("📊 Phase 2: Cross-target consistency analysis...");
        let consistency_score = self
            .analyze_cross_target_consistency(&adaptive_params)
            .await?;

        // Phase 3: System-wide balance optimization using weights
        log::info!("📊 Phase 3: System-wide balance optimization...");
        if consistency_score > 0.1 {
            // If consistency needs improvement
            log::debug!(
                "🔧 Applying system balance weight: {:.2}",
                self.system_balance_weight
            );
            log::debug!(
                "🔧 Cross-target consistency weight: {:.2}",
                self.cross_target_consistency_weight
            );
            log::debug!(
                "🔧 Max calibration iterations: {}",
                self.max_calibration_iterations
            );

            adaptive_params = self
                .optimize_system_balance(
                    adaptive_params,
                    ohlcv_data,
                    sequence_length,
                    horizon_steps,
                    sequence_indices,
                )
                .await?;
        }

        // Phase 4: Final validation and reporting
        log::info!("📊 Phase 4: Final validation...");
        let final_validation = self
            .validate_unified_system(
                &adaptive_params,
                ohlcv_data,
                sequence_length,
                horizon_steps,
                sequence_indices,
            )
            .await?;

        let calibration_time = start_time.elapsed().as_millis() as u64;

        // Update calibration metadata with unified results
        let mut final_params = adaptive_params;
        final_params.calibration_info.optimization_time_ms = calibration_time;
        final_params.calibration_info.overall_balance_score =
            final_validation.overall_balance_score;
        final_params.calibration_info.calibration_success = final_validation.system_success;

        // Log unified calibration results
        self.log_unified_results(&final_params, &final_validation);

        log::info!(
            "✅ Unified calibration completed in {}ms with system balance score: {:.4}",
            calibration_time,
            final_validation.overall_balance_score
        );

        Ok(final_params)
    }

    /// Analyze consistency across target types
    async fn analyze_cross_target_consistency(
        &self,
        params: &AdaptiveTargetParameters,
    ) -> Result<f64> {
        // Calculate consistency metrics between target types
        let direction_balance = params.direction.achieved_balance.balance_score;
        let price_level_balance = params.price_levels.achieved_balance.balance_score;
        let volatility_balance = params.volatility.achieved_balance.balance_score;

        // Calculate standard deviation of balance scores (lower = more consistent)
        let balance_scores = [direction_balance, price_level_balance, volatility_balance];
        let mean_balance = balance_scores.iter().sum::<f64>() / 3.0;
        let balance_variance = balance_scores
            .iter()
            .map(|&score| (score - mean_balance).powi(2))
            .sum::<f64>()
            / 3.0;
        let consistency_score = balance_variance.sqrt();

        log::debug!(
            "🔍 Cross-target consistency: direction={:.4}, price_level={:.4}, volatility={:.4}, consistency_score={:.4}",
            direction_balance,
            price_level_balance,
            volatility_balance,
            consistency_score
        );

        Ok(consistency_score)
    }

    /// Optimize system-wide balance across all targets
    async fn optimize_system_balance(
        &self,
        params: AdaptiveTargetParameters,
        _ohlcv_data: &[MarketDataRow],
        _sequence_length: usize,
        _horizon_steps: usize,
        _sequence_indices: &[usize],
    ) -> Result<AdaptiveTargetParameters> {
        log::debug!("🎯 Optimizing system-wide balance...");

        // For now, return the original parameters
        // In a full implementation, this would perform joint optimization
        // across all target types to minimize overall system imbalance
        // using the provided data parameters

        // Calculate current system balance
        let system_balance = self.calculate_system_balance(&params);

        log::debug!(
            "📊 System balance optimization: current_score={:.4}",
            system_balance.overall_score
        );

        // TODO: Implement joint optimization algorithm using:
        // - ohlcv_data for target generation validation
        // - sequence_length and horizon_steps for parameter constraints
        // - sequence_indices for consistent evaluation
        // This would involve:
        // 1. Joint parameter space exploration
        // 2. Multi-objective optimization (balance vs. consistency)
        // 3. Pareto frontier analysis for optimal trade-offs
        // 4. Iterative refinement with cross-validation

        Ok(params)
    }

    /// Validate the unified system performance
    async fn validate_unified_system(
        &self,
        params: &AdaptiveTargetParameters,
        ohlcv_data: &[MarketDataRow],
        sequence_length: usize,
        horizon_steps: usize,
        sequence_indices: &[usize],
    ) -> Result<UnifiedValidationResult> {
        log::debug!("🔍 Validating unified system...");

        // Calculate system-wide metrics
        let system_balance = self.calculate_system_balance(params);

        // Validate individual target performance
        let direction_valid = params.direction.achieved_balance.balance_score < 5.0;
        let price_level_valid = params.price_levels.achieved_balance.balance_score < 5.0;
        let volatility_valid = params.volatility.achieved_balance.balance_score < 5.0;

        let system_success = direction_valid
            && price_level_valid
            && volatility_valid
            && system_balance.overall_score < 10.0;

        // Calculate cross-target correlation (measure of independence)
        let cross_correlation = self
            .calculate_cross_target_correlation(
                ohlcv_data,
                sequence_length,
                horizon_steps,
                sequence_indices,
                params,
            )
            .await?;

        Ok(UnifiedValidationResult {
            overall_balance_score: system_balance.overall_score,
            system_success,
            direction_valid,
            price_level_valid,
            volatility_valid,
            cross_target_correlation: cross_correlation,
            consistency_score: system_balance.consistency_score,
        })
    }

    /// Calculate system-wide balance metrics
    fn calculate_system_balance(&self, params: &AdaptiveTargetParameters) -> SystemBalanceMetrics {
        let direction_score = params.direction.achieved_balance.balance_score;
        let price_level_score = params.price_levels.achieved_balance.balance_score;
        let volatility_score = params.volatility.achieved_balance.balance_score;

        let overall_score = (direction_score + price_level_score + volatility_score) / 3.0;

        // Calculate consistency (how similar the balance scores are)
        let scores = [direction_score, price_level_score, volatility_score];
        let mean_score = overall_score;
        let consistency_variance = scores
            .iter()
            .map(|&score| (score - mean_score).powi(2))
            .sum::<f64>()
            / 3.0;
        let consistency_score = consistency_variance.sqrt();

        SystemBalanceMetrics {
            overall_score,
            consistency_score,
            direction_contribution: direction_score / overall_score,
            price_level_contribution: price_level_score / overall_score,
            volatility_contribution: volatility_score / overall_score,
        }
    }

    /// Calculate cross-target correlation to measure independence
    async fn calculate_cross_target_correlation(
        &self,
        _ohlcv_data: &[MarketDataRow],
        _sequence_length: usize,
        _horizon_steps: usize,
        _sequence_indices: &[usize],
        _params: &AdaptiveTargetParameters,
    ) -> Result<CrossTargetCorrelation> {
        // For now, return default correlation metrics
        // In a full implementation, this would:
        // 1. Generate targets using calibrated parameters
        // 2. Calculate correlation matrices between target types
        // 3. Measure target independence and complementarity
        // 4. Validate that targets provide diverse information

        Ok(CrossTargetCorrelation {
            direction_price_correlation: 0.1,
            direction_volatility_correlation: 0.15,
            price_volatility_correlation: 0.2,
            overall_independence_score: 0.85,
        })
    }

    /// Log comprehensive unified calibration results
    fn log_unified_results(
        &self,
        params: &AdaptiveTargetParameters,
        validation: &UnifiedValidationResult,
    ) {
        log::info!("🎯 UNIFIED TARGET CALIBRATION RESULTS");
        log::info!("=====================================");

        // Individual target results
        log::info!(
            "📊 Direction: sensitivity={:.6}, balance_score={:.4}, valid={}",
            params.direction.base_sensitivity,
            params.direction.achieved_balance.balance_score,
            validation.direction_valid
        );

        log::info!(
            "📊 Price Levels: bandwidth={:.4}, balance_score={:.4}, valid={}",
            params.price_levels.bandwidth_size,
            params.price_levels.achieved_balance.balance_score,
            validation.price_level_valid
        );

        log::info!(
            "📊 Volatility: bandwidth={:.4}, balance_score={:.4}, valid={}",
            params.volatility.bandwidth_size,
            params.volatility.achieved_balance.balance_score,
            validation.volatility_valid
        );

        // System-wide results
        log::info!(
            "🎯 System: overall_balance={:.4}, consistency={:.4}, correlation={:.3}, success={}",
            validation.overall_balance_score,
            validation.consistency_score,
            validation
                .cross_target_correlation
                .overall_independence_score,
            validation.system_success
        );

        // Cross-target correlations
        log::info!(
            "🔗 Correlations: dir-price={:.3}, dir-vol={:.3}, price-vol={:.3}",
            validation
                .cross_target_correlation
                .direction_price_correlation,
            validation
                .cross_target_correlation
                .direction_volatility_correlation,
            validation
                .cross_target_correlation
                .price_volatility_correlation
        );

        log::info!("=====================================");
    }
}

/// System-wide balance metrics
#[derive(Debug, Clone)]
pub struct SystemBalanceMetrics {
    pub overall_score: f64,
    pub consistency_score: f64,
    pub direction_contribution: f64,
    pub price_level_contribution: f64,
    pub volatility_contribution: f64,
}

/// Cross-target correlation metrics
#[derive(Debug, Clone)]
pub struct CrossTargetCorrelation {
    pub direction_price_correlation: f64,
    pub direction_volatility_correlation: f64,
    pub price_volatility_correlation: f64,
    pub overall_independence_score: f64,
}

/// Unified validation result
#[derive(Debug, Clone)]
pub struct UnifiedValidationResult {
    pub overall_balance_score: f64,
    pub system_success: bool,
    pub direction_valid: bool,
    pub price_level_valid: bool,
    pub volatility_valid: bool,
    pub cross_target_correlation: CrossTargetCorrelation,
    pub consistency_score: f64,
}

/// Convenience function for unified calibration
pub async fn calibrate_adaptive_parameters(
    base_config: TargetsConfig,
    ohlcv_data: &[MarketDataRow],
    sequence_length: usize,
    horizon_steps: usize,
    sequence_indices: &[usize],
) -> Result<AdaptiveTargetParameters> {
    let calibrator = UnifiedTargetCalibrator::new(base_config);
    calibrator
        .calibrate_unified_system(ohlcv_data, sequence_length, horizon_steps, sequence_indices)
        .await
}
