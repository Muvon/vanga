//! Calibration Parameter Types
//!
//! This module contains all parameter structures used in the calibration system.
//! Each target type has its own parameter struct with calibrated values that
//! ensure balanced 5-class classification across all targets.

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

/// Context for evaluation functions
#[derive(Clone, Copy)]
pub struct EvaluationContext<'a> {
    pub ohlcv_data: &'a [crate::data::structures::MarketDataRow],
    pub sample_indices: &'a [usize],
    pub sequence_length: usize,
    pub horizon_steps: usize,
}

/// Parameters for direction evaluation
#[derive(Debug, Clone)]
pub struct DirectionEvalParams {
    pub sensitivity: f64,
    pub extreme_multiplier: f64,
    pub min_base_threshold: f64,
    pub min_extreme_threshold: f64,
    pub base_multiplier: f64,
}

/// Parameters for price level evaluation
#[derive(Debug, Clone)]
pub struct PriceLevelEvalParams {
    pub bandwidth: f64,
    pub percentiles: [f64; 2], // Base percentiles for adaptive calculation
    pub neutral_band: f64,
    pub momentum_factor: f64, // Momentum factor for bandwidth adjustment
}

/// Parameters for volatility evaluation
#[derive(Debug, Clone)]
pub struct VolatilityEvalParams {
    pub bandwidth: f64,
    pub multiplier: f64,
    pub decay: f64,
    pub volume_weight: f64, // NEW: Volume weight parameter
    pub min_baseline: f64,  // NEW: Minimum volatility baseline parameter
}

/// Parameters for sentiment evaluation
#[derive(Debug, Clone)]
pub struct SentimentEvalParams {
    pub sensitivity: f64,
    pub volume_weight: f64,
    pub consistency_factor: f64,
}

/// Parameters for volume evaluation
#[derive(Debug, Clone)]
pub struct VolumeEvalParams {
    pub bandwidth: f64,
    pub multiplier: f64,
    pub smoothing: usize,
}
