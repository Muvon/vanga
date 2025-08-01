//! Unified sequence reconstruction logic for training-prediction consistency
//!
//! This module provides centralized sequence analysis and reconstruction capabilities
//! to ensure mathematical consistency between training target generation and prediction
//! output formatting. All sequence-related logic should use this module as the single
//! source of truth.

use crate::data::structures::MarketDataRow;
use crate::utils::error::{Result, VangaError};
use serde::{Deserialize, Serialize};

/// Configuration for sequence reconstruction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SequenceReconstructionConfig {
    /// Percentiles for boundary calculation [lower, upper] e.g., [0.1, 0.9]
    pub percentiles: [f64; 2],
    /// Bandwidth size multiplier for breakout detection
    pub bandwidth_size: f64,
    /// Enable adaptive percentile calculation based on sequence characteristics
    pub adaptive_percentiles: bool,
    /// Sensitivity level for adaptive calculations
    pub sensitivity: crate::config::model::AdaptiveSensitivity,
}

impl Default for SequenceReconstructionConfig {
    fn default() -> Self {
        Self {
            percentiles: [0.1, 0.9],    // Default 10th-90th percentiles
            bandwidth_size: 1.0,        // Default bandwidth multiplier
            adaptive_percentiles: true, // Enable adaptive percentiles by default
            sensitivity: crate::config::model::AdaptiveSensitivity::Balanced, // Default sensitivity
        }
    }
}

impl SequenceReconstructionConfig {
    /// Calculate adaptive percentiles based on sequence characteristics
    /// This method can be used by all target types for consistent adaptive behavior
    pub fn adaptive_percentiles(&self, sequence_data: &[MarketDataRow]) -> [f64; 2] {
        if !self.adaptive_percentiles || sequence_data.len() < 10 {
            return self.percentiles; // Use fixed percentiles as fallback
        }

        // Calculate VWAP prices for analysis
        let vwap_prices: Vec<f64> = sequence_data
            .iter()
            .map(|candle| {
                // Volume weighting doesn't matter for OHLC4 calculation
                (candle.open + candle.high + candle.low + candle.close) / 4.0
            })
            .collect();

        // Use default sensitivity for now - will be updated to use TargetsConfig
        let sensitivity = crate::config::model::AdaptiveSensitivity::Balanced;
        SequenceAnalyzer::calculate_adaptive_percentiles(&vwap_prices, sensitivity)
    }

    /// Calculate adaptive sensitivity multiplier for direction/volatility targets
    /// Returns a multiplier that adjusts base_sensitivity based on sequence characteristics
    pub fn adaptive_sensitivity_multiplier(&self, sequence_data: &[MarketDataRow]) -> f64 {
        if !self.adaptive_percentiles || sequence_data.len() < 10 {
            return 1.0; // Use base sensitivity as-is
        }

        // Calculate VWAP prices for analysis
        let vwap_prices: Vec<f64> = sequence_data
            .iter()
            .map(|candle| {
                // Volume weighting doesn't matter for OHLC4 calculation
                (candle.open + candle.high + candle.low + candle.close) / 4.0
            })
            .collect();

        let (volatility_coeff, trend_strength, range_coeff) =
            SequenceAnalyzer::calculate_sequence_characteristics(&vwap_prices);

        // Calculate adaptive multiplier
        // High volatility = higher sensitivity (more responsive to changes)
        // Strong trend = lower sensitivity (avoid over-classification in trends)
        // Large range = higher sensitivity (more breakout opportunities)
        let volatility_adjustment = volatility_coeff * 1.5; // Max 1.5x adjustment
        let trend_adjustment = trend_strength * 0.5; // Max 0.5x reduction
        let range_adjustment = range_coeff * 0.3; // Max 0.3x adjustment

        let multiplier =
            (1.0 + volatility_adjustment - trend_adjustment + range_adjustment).clamp(0.5, 2.0); // Reasonable bounds (0.5x to 2.0x)

        log::debug!(
            "🎯 Adaptive Sensitivity: vol_coeff={:.3}, trend_strength={:.3}, range_coeff={:.3}, multiplier={:.3}",
            volatility_coeff, trend_strength, range_coeff, multiplier
        );

        multiplier
    }
}

/// Sequence boundaries calculated from OHLCV data
#[derive(Debug, Clone)]
pub struct SequenceBoundaries {
    /// Lower percentile boundary (e.g., 10th percentile)
    pub sequence_min: f64,
    /// Upper percentile boundary (e.g., 90th percentile)
    pub sequence_max: f64,
    /// Bandwidth for breakout detection
    pub bandwidth: f64,
    /// Classification boundaries [boundary_1, boundary_2, boundary_3, boundary_4]
    /// - boundary_1: sequence_min - bandwidth (strong_down | moderate_down)
    /// - boundary_2: sequence_min (moderate_down | neutral)
    /// - boundary_3: sequence_max (neutral | moderate_up)
    /// - boundary_4: sequence_max + bandwidth (moderate_up | strong_up)
    pub boundaries: [f64; 4],
    /// VWAP prices used for calculation
    pub vwap_prices: Vec<f64>,
}

impl SequenceBoundaries {
    /// Get the 5 price level ranges as percentage arrays from current price
    pub fn get_price_level_ranges(&self, current_price: f64) -> Vec<[f64; 2]> {
        let to_pct = |price: f64| ((price - current_price) / current_price) * 100.0;

        // Create non-overlapping ranges with tiny epsilon for JSON display
        let epsilon = 0.0001; // 0.0001% - tiny value to prevent overlap

        // **EXACT INVERSE OF TRAINING LOGIC**:
        // Training boundaries: [seq_min - bw, seq_min, seq_max, seq_max + bw]
        // boundaries[0] = seq_min - bandwidth
        // boundaries[1] = seq_min
        // boundaries[2] = seq_max  
        // boundaries[3] = seq_max + bandwidth

        vec![
            // Strong Down: < boundaries[0] (< seq_min - bandwidth)
            [
                to_pct(self.boundaries[0] - self.bandwidth), // Extended lower bound for display
                to_pct(self.boundaries[0]) - epsilon,        // < seq_min - bandwidth
            ],
            // Moderate Down: [boundaries[0], boundaries[1]) = [seq_min - bw, seq_min)
            [
                to_pct(self.boundaries[0]),                  // seq_min - bandwidth
                to_pct(self.boundaries[1]) - epsilon,        // seq_min
            ],
            // Neutral: [boundaries[1], boundaries[2]) = [seq_min, seq_max)
            [
                to_pct(self.boundaries[1]),                  // seq_min
                to_pct(self.boundaries[2]) - epsilon,        // seq_max
            ],
            // Moderate Up: [boundaries[2], boundaries[3]) = [seq_max, seq_max + bw)
            [
                to_pct(self.boundaries[2]),                  // seq_max
                to_pct(self.boundaries[3]) - epsilon,        // seq_max + bandwidth
            ],
            // Strong Up: >= boundaries[3] (>= seq_max + bandwidth)
            [
                to_pct(self.boundaries[3]),                  // seq_max + bandwidth
                to_pct(self.boundaries[3] + self.bandwidth), // Extended upper bound for display
            ],
        ]
    }

    /// Classify a target price into one of 5 classes (matches training logic exactly)
    pub fn classify_price(&self, target_price: f64) -> i32 {
        if target_price < self.boundaries[0] {
            0 // Strong Down: Below sequence_min - bandwidth
        } else if target_price < self.boundaries[1] {
            1 // Moderate Down: Below sequence_min
        } else if target_price < self.boundaries[2] {
            2 // Neutral: Within percentile range
        } else if target_price < self.boundaries[3] {
            3 // Moderate Up: Above sequence_max
        } else {
            4 // Strong Up: Above sequence_max + bandwidth
        }
    }

    /// Get class names in order
    pub fn get_class_names() -> Vec<&'static str> {
        vec![
            "strong_down",
            "moderate_down",
            "neutral",
            "moderate_up",
            "strong_up",
        ]
    }
}

/// Unified sequence analyzer for training-prediction consistency
pub struct SequenceAnalyzer {
    config: SequenceReconstructionConfig,
}

impl SequenceAnalyzer {
    /// Create new sequence analyzer with configuration
    pub fn new(config: SequenceReconstructionConfig) -> Self {
        Self { config }
    }

    /// Calculate sequence characteristics for adaptive percentile determination
    /// Returns (volatility_coefficient, trend_strength, range_coefficient)
    fn calculate_sequence_characteristics(vwap_prices: &[f64]) -> (f64, f64, f64) {
        if vwap_prices.len() < 3 {
            return (0.0, 0.0, 0.0);
        }

        let n = vwap_prices.len() as f64;
        let mean = vwap_prices.iter().sum::<f64>() / n;

        // 1. Volatility coefficient (coefficient of variation)
        let variance = vwap_prices.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n;
        let volatility_coefficient = if mean > 0.0 {
            (variance.sqrt() / mean).clamp(0.0, 1.0)
        } else {
            0.0
        };

        // 2. Trend strength (normalized linear regression slope)
        let x_mean = (n - 1.0) / 2.0; // Mean of indices [0, 1, 2, ..., n-1]
        let numerator: f64 = vwap_prices
            .iter()
            .enumerate()
            .map(|(i, &y)| (i as f64 - x_mean) * (y - mean))
            .sum();
        let denominator: f64 = (0..vwap_prices.len())
            .map(|i| (i as f64 - x_mean).powi(2))
            .sum();

        let trend_strength = if denominator > 0.0 && mean > 0.0 {
            let slope = numerator / denominator;
            (slope / mean).abs().clamp(0.0, 1.0) // Normalize by mean price
        } else {
            0.0
        };

        // 3. Range coefficient (range relative to mean)
        let min_price = vwap_prices.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_price = vwap_prices.iter().cloned().fold(0.0, f64::max);
        let range_coefficient = if mean > 0.0 {
            ((max_price - min_price) / mean).clamp(0.0, 1.0)
        } else {
            0.0
        };

        (volatility_coefficient, trend_strength, range_coefficient)
    }

    /// Calculate adaptive percentiles based on sequence characteristics
    /// Returns [lower_percentile, upper_percentile] optimized for balanced distribution
    /// 
    /// **INVERSE TRAINING LOGIC**: This reverses the exact training process
    fn calculate_adaptive_percentiles(
        vwap_prices: &[f64],
        sensitivity: crate::config::model::AdaptiveSensitivity,
    ) -> [f64; 2] {
        let (volatility_coeff, trend_strength, _range_coeff) =
            Self::calculate_sequence_characteristics(vwap_prices);

        // **REVERSE LOGIC**: Use the same sensitivity scaling as training
        // In training: bandwidth = base_bandwidth * sensitivity.base_value()
        // For reconstruction: we need percentiles that create the same relative bandwidth
        
        let base_sensitivity = sensitivity.base_value();
        
        // Adaptive adjustments (same logic as training, but for percentile selection)
        let volatility_factor = 1.0 - (volatility_coeff * 0.3).min(0.2); // High vol = narrower percentiles
        let trend_factor = 1.0 + (trend_strength * 0.2).min(0.1);        // Strong trend = wider percentiles
        
        // Calculate adaptive percentile width based on sensitivity
        // This creates the inverse relationship: higher sensitivity = narrower percentiles
        let percentile_width = base_sensitivity * volatility_factor * trend_factor;
        
        // Convert to actual percentiles around the median (0.5)
        let lower_percentile = (0.5 - percentile_width).clamp(0.05, 0.45);
        let upper_percentile = (0.5 + percentile_width).clamp(0.55, 0.95);

        log::debug!(
            "🔄 INVERSE Training Logic - Adaptive Percentiles: sensitivity={}, base={:.4}, vol_factor={:.3}, trend_factor={:.3}, width={:.4}, percentiles=[{:.3}, {:.3}]",
            sensitivity, base_sensitivity, volatility_factor, trend_factor, percentile_width, lower_percentile, upper_percentile
        );

        [lower_percentile, upper_percentile]
    }

    /// Create from model configuration (for consistency with training)
    pub fn from_model_config(model_config: &crate::config::model::TargetsConfig) -> Self {
        let config = SequenceReconstructionConfig {
            percentiles: [0.1, 0.9], // Default percentiles for 5-class system
            bandwidth_size: model_config.base_sensitivity(), // Use base_sensitivity as bandwidth
            adaptive_percentiles: true, // Enable adaptive percentiles by default
            sensitivity: model_config.sensitivity, // Use the sensitivity from model config
        };
        Self::new(config)
    }

    /// Create with adaptive percentiles calculated from sequence data
    ///
    /// This method ensures training-prediction consistency by using the same
    /// adaptive percentile calculation logic for both target generation and
    /// prediction reconstruction.
    pub fn from_sequence_data(
        sequence_ohlcv: &[MarketDataRow],
        bandwidth_size: f64,
    ) -> Result<Self> {
        // Use the same adaptive percentile logic as training
        let adaptive_percentiles =
            crate::targets::price_levels::calculate_adaptive_percentiles_from_sequence(
                sequence_ohlcv,
            )?;

        let config = SequenceReconstructionConfig {
            percentiles: adaptive_percentiles,
            bandwidth_size,
        };

        Ok(Self::new(config))
    }

    /// Calculate VWAP prices from OHLCV sequence (centralized logic)
    pub fn calculate_vwap_prices(&self, sequence_ohlcv: &[MarketDataRow]) -> Result<Vec<f64>> {
        if sequence_ohlcv.is_empty() {
            return Err(VangaError::data("Empty OHLCV sequence provided"));
        }

        let vwap_prices: Vec<f64> = sequence_ohlcv
            .iter()
            .map(|candle| {
                if candle.volume > 0.0 {
                    // Volume-weighted OHLC4
                    (candle.open + candle.high + candle.low + candle.close) / 4.0
                } else {
                    // Fallback to simple OHLC4 if no volume
                    (candle.open + candle.high + candle.low + candle.close) / 4.0
                }
            })
            .collect();

        Ok(vwap_prices)
    }

    /// Calculate sequence boundaries from OHLCV data (single source of truth)
    pub fn calculate_boundaries(
        &self,
        sequence_ohlcv: &[MarketDataRow],
    ) -> Result<SequenceBoundaries> {
        if sequence_ohlcv.len() < 2 {
            return Err(VangaError::data(
                "Insufficient OHLCV data for boundary calculation (need at least 2 candles)",
            ));
        }

        // Calculate VWAP prices
        let vwap_prices = self.calculate_vwap_prices(sequence_ohlcv)?;

        // Determine percentiles (adaptive or fixed)
        let percentiles = if self.config.adaptive_percentiles && vwap_prices.len() >= 10 {
            // Use adaptive percentiles based on sequence characteristics
            Self::calculate_adaptive_percentiles(&vwap_prices, self.config.sensitivity)
        } else {
            // Use fixed percentiles as fallback
            self.config.percentiles
        };

        // Calculate percentile boundaries from sorted prices
        let mut sorted_prices = vwap_prices.clone();
        sorted_prices.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let n = sorted_prices.len();
        let lower_idx = ((n as f64 * percentiles[0]) as usize).min(n - 1);
        let upper_idx = ((n as f64 * percentiles[1]) as usize).min(n - 1);

        let sequence_min = sorted_prices[lower_idx];
        let sequence_max = sorted_prices[upper_idx];

        // Calculate bandwidth
        let base_bandwidth = sequence_max - sequence_min;
        let bandwidth = base_bandwidth * self.config.bandwidth_size;

        // Define classification boundaries (matches training logic exactly)
        let boundaries = [
            sequence_min - bandwidth, // boundary_1: strong_down | moderate_down
            sequence_min,             // boundary_2: moderate_down | neutral
            sequence_max,             // boundary_3: neutral | moderate_up
            sequence_max + bandwidth, // boundary_4: moderate_up | strong_up
        ];

        log::debug!(
            "🎯 Sequence Boundaries: adaptive={}, percentiles=[{:.3}, {:.3}], range=[{:.6}, {:.6}], bandwidth={:.6}",
            self.config.adaptive_percentiles && vwap_prices.len() >= 10,
            percentiles[0], percentiles[1],
            sequence_min, sequence_max, bandwidth
        );

        Ok(SequenceBoundaries {
            sequence_min,
            sequence_max,
            bandwidth,
            boundaries,
            vwap_prices,
        })
    }

    /// Reconstruct targets from sequences (for validation and testing)
    pub fn sequences_to_targets(
        &self,
        sequence_ohlcv: &[MarketDataRow],
        horizon_prices: &[f64],
    ) -> Result<Vec<i32>> {
        let boundaries = self.calculate_boundaries(sequence_ohlcv)?;

        let targets: Vec<i32> = horizon_prices
            .iter()
            .map(|&price| boundaries.classify_price(price))
            .collect();

        Ok(targets)
    }

    /// Reconstruct probability distribution from sequences (for prediction output)
    pub fn sequences_to_probabilities(
        &self,
        _sequence_ohlcv: &[MarketDataRow],
        model_probabilities: &[f64], // Raw model output probabilities
    ) -> Result<Vec<f64>> {
        // Validate input
        if model_probabilities.len() != 5 {
            return Err(VangaError::data(
                "Expected 5 class probabilities for price levels",
            ));
        }

        // For now, return the model probabilities as-is
        // Future enhancement: Apply sequence-aware probability adjustments
        Ok(model_probabilities.to_vec())
    }

    /// Get configuration for debugging and validation
    pub fn get_config(&self) -> &SequenceReconstructionConfig {
        &self.config
    }
}

/// Trait for sequence reconstruction capabilities
pub trait SequenceReconstructor {
    /// Convert sequences to target classifications
    fn sequences_to_targets(
        &self,
        sequence_ohlcv: &[MarketDataRow],
        horizon_prices: &[f64],
    ) -> Result<Vec<i32>>;

    /// Convert sequences to probability distributions
    fn sequences_to_probabilities(
        &self,
        sequence_ohlcv: &[MarketDataRow],
        model_probabilities: &[f64],
    ) -> Result<Vec<f64>>;

    /// Convert sequences to price level ranges
    fn sequences_to_ranges(
        &self,
        sequence_ohlcv: &[MarketDataRow],
        current_price: f64,
    ) -> Result<Vec<[f64; 2]>>;
}

impl SequenceReconstructor for SequenceAnalyzer {
    fn sequences_to_targets(
        &self,
        sequence_ohlcv: &[MarketDataRow],
        horizon_prices: &[f64],
    ) -> Result<Vec<i32>> {
        self.sequences_to_targets(sequence_ohlcv, horizon_prices)
    }

    fn sequences_to_probabilities(
        &self,
        _sequence_ohlcv: &[MarketDataRow],
        model_probabilities: &[f64],
    ) -> Result<Vec<f64>> {
        self.sequences_to_probabilities(_sequence_ohlcv, model_probabilities)
    }

    fn sequences_to_ranges(
        &self,
        sequence_ohlcv: &[MarketDataRow],
        current_price: f64,
    ) -> Result<Vec<[f64; 2]>> {
        let boundaries = self.calculate_boundaries(sequence_ohlcv)?;
        Ok(boundaries.get_price_level_ranges(current_price))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_candles(ohlcv_data: Vec<(f64, f64, f64, f64, f64)>) -> Vec<MarketDataRow> {
        ohlcv_data
            .into_iter()
            .enumerate()
            .map(|(i, (o, h, l, c, v))| MarketDataRow {
                timestamp: i as i64,
                open: o,
                high: h,
                low: l,
                close: c,
                volume: v,
            })
            .collect()
    }

    #[test]
    fn test_vwap_calculation() {
        let analyzer = SequenceAnalyzer::new(SequenceReconstructionConfig::default());
        let candles = create_test_candles(vec![
            (100.0, 105.0, 95.0, 102.0, 1000.0),
            (102.0, 108.0, 98.0, 106.0, 1500.0),
        ]);

        let vwap_prices = analyzer.calculate_vwap_prices(&candles).unwrap();
        assert_eq!(vwap_prices.len(), 2);
        assert_eq!(vwap_prices[0], (100.0 + 105.0 + 95.0 + 102.0) / 4.0);
        assert_eq!(vwap_prices[1], (102.0 + 108.0 + 98.0 + 106.0) / 4.0);
    }

    #[test]
    fn test_boundary_calculation() {
        let analyzer = SequenceAnalyzer::new(SequenceReconstructionConfig {
            percentiles: [0.2, 0.8],
            bandwidth_size: 1.0,
            adaptive_percentiles: false, // Disable for predictable test results
            sensitivity: crate::config::model::AdaptiveSensitivity::Balanced, // Default sensitivity
        });

        let candles = create_test_candles(vec![
            (90.0, 95.0, 85.0, 92.0, 1000.0),     // VWAP: 90.5
            (95.0, 100.0, 90.0, 98.0, 1000.0),    // VWAP: 95.75
            (100.0, 110.0, 95.0, 105.0, 1000.0),  // VWAP: 102.5
            (105.0, 115.0, 100.0, 110.0, 1000.0), // VWAP: 107.5
            (110.0, 120.0, 105.0, 115.0, 1000.0), // VWAP: 112.5
        ]);

        let boundaries = analyzer.calculate_boundaries(&candles).unwrap();

        // Verify boundaries are calculated correctly
        assert!(boundaries.sequence_min < boundaries.sequence_max);
        assert!(boundaries.bandwidth > 0.0);
        assert_eq!(boundaries.boundaries.len(), 4);

        // Verify boundary ordering
        assert!(boundaries.boundaries[0] < boundaries.boundaries[1]);
        assert!(boundaries.boundaries[1] < boundaries.boundaries[2]);
        assert!(boundaries.boundaries[2] < boundaries.boundaries[3]);
    }

    #[test]
    fn test_price_classification() {
        let analyzer = SequenceAnalyzer::new(SequenceReconstructionConfig::default());
        let candles = create_test_candles(vec![
            (100.0, 105.0, 95.0, 102.0, 1000.0),
            (102.0, 108.0, 98.0, 106.0, 1000.0),
            (106.0, 112.0, 102.0, 110.0, 1000.0),
        ]);

        let boundaries = analyzer.calculate_boundaries(&candles).unwrap();

        // Test classification at different price levels
        let very_low_price = boundaries.boundaries[0] - 1.0;
        let mid_price = (boundaries.sequence_min + boundaries.sequence_max) / 2.0;
        let very_high_price = boundaries.boundaries[3] + 1.0;

        assert_eq!(boundaries.classify_price(very_low_price), 0); // Strong Down
        assert_eq!(boundaries.classify_price(mid_price), 2); // Neutral
        assert_eq!(boundaries.classify_price(very_high_price), 4); // Strong Up
    }

    #[test]
    fn test_adaptive_percentiles() {
        let config = SequenceReconstructionConfig {
            percentiles: [0.1, 0.9],
            bandwidth_size: 1.0,
            adaptive_percentiles: true,
            sensitivity: crate::config::model::AdaptiveSensitivity::Balanced, // Default sensitivity
        };

        // Test with high volatility sequence (should narrow percentiles)
        let high_vol_candles = create_test_candles(vec![
            (100.0, 120.0, 80.0, 110.0, 1000.0),  // High volatility
            (110.0, 140.0, 90.0, 120.0, 1000.0),  // High volatility
            (120.0, 150.0, 100.0, 130.0, 1000.0), // High volatility
            (130.0, 160.0, 110.0, 140.0, 1000.0), // High volatility
            (140.0, 170.0, 120.0, 150.0, 1000.0), // High volatility
            (150.0, 180.0, 130.0, 160.0, 1000.0), // High volatility
            (160.0, 190.0, 140.0, 170.0, 1000.0), // High volatility
            (170.0, 200.0, 150.0, 180.0, 1000.0), // High volatility
            (180.0, 210.0, 160.0, 190.0, 1000.0), // High volatility
            (190.0, 220.0, 170.0, 200.0, 1000.0), // High volatility
        ]);

        let adaptive_percentiles = config.adaptive_percentiles(&high_vol_candles);
        println!(
            "🎯 High volatility adaptive percentiles: [{:.3}, {:.3}]",
            adaptive_percentiles[0], adaptive_percentiles[1]
        );

        // Test with low volatility sequence (should widen percentiles)
        let low_vol_candles = create_test_candles(vec![
            (100.0, 101.0, 99.0, 100.5, 1000.0),  // Low volatility
            (100.5, 101.5, 99.5, 101.0, 1000.0),  // Low volatility
            (101.0, 102.0, 100.0, 101.5, 1000.0), // Low volatility
            (101.5, 102.5, 100.5, 102.0, 1000.0), // Low volatility
            (102.0, 103.0, 101.0, 102.5, 1000.0), // Low volatility
            (102.5, 103.5, 101.5, 103.0, 1000.0), // Low volatility
            (103.0, 104.0, 102.0, 103.5, 1000.0), // Low volatility
            (103.5, 104.5, 102.5, 104.0, 1000.0), // Low volatility
            (104.0, 105.0, 103.0, 104.5, 1000.0), // Low volatility
            (104.5, 105.5, 103.5, 105.0, 1000.0), // Low volatility
        ]);

        let adaptive_percentiles_low = config.adaptive_percentiles(&low_vol_candles);
        println!(
            "🎯 Low volatility adaptive percentiles: [{:.3}, {:.3}]",
            adaptive_percentiles_low[0], adaptive_percentiles_low[1]
        );

        // Both should have much narrower neutral zones than the original [0.1, 0.9]
        let high_vol_width = adaptive_percentiles[1] - adaptive_percentiles[0];
        let low_vol_width = adaptive_percentiles_low[1] - adaptive_percentiles_low[0];
        let original_width = 0.9 - 0.1; // 0.8

        println!(
            "🎯 High vol width: {:.3}, Low vol width: {:.3}, Original width: {:.3}",
            high_vol_width, low_vol_width, original_width
        );

        // Both adaptive widths should be much narrower than original
        assert!(
            high_vol_width < original_width * 0.6,
            "High volatility should have much narrower neutral zone: {:.3} < {:.3}",
            high_vol_width,
            original_width * 0.6
        );
        assert!(
            low_vol_width < original_width * 0.6,
            "Low volatility should have much narrower neutral zone: {:.3} < {:.3}",
            low_vol_width,
            original_width * 0.6
        );

        // Test sideways market (low trend + low volatility) - should get sideways bonus
        let sideways_candles = create_test_candles(vec![
            (100.0, 100.5, 99.5, 100.2, 1000.0), // Very low volatility, no trend
            (100.2, 100.7, 99.7, 100.1, 1000.0), // Very low volatility, no trend
            (100.1, 100.6, 99.6, 100.3, 1000.0), // Very low volatility, no trend
            (100.3, 100.8, 99.8, 100.0, 1000.0), // Very low volatility, no trend
            (100.0, 100.5, 99.5, 100.4, 1000.0), // Very low volatility, no trend
            (100.4, 100.9, 99.9, 100.1, 1000.0), // Very low volatility, no trend
            (100.1, 100.6, 99.6, 100.2, 1000.0), // Very low volatility, no trend
            (100.2, 100.7, 99.7, 100.3, 1000.0), // Very low volatility, no trend
            (100.3, 100.8, 99.8, 100.0, 1000.0), // Very low volatility, no trend
            (100.0, 100.5, 99.5, 100.1, 1000.0), // Very low volatility, no trend
        ]);

        let sideways_percentiles = config.adaptive_percentiles(&sideways_candles);
        println!(
            "🎯 Sideways market adaptive percentiles: [{:.3}, {:.3}]",
            sideways_percentiles[0], sideways_percentiles[1]
        );

        let sideways_width = sideways_percentiles[1] - sideways_percentiles[0];
        println!("🎯 Sideways width: {:.3}", sideways_width);

        // Sideways should have the narrowest neutral zone
        assert!(
            sideways_width <= low_vol_width,
            "Sideways should have narrowest or equal neutral zone: {:.3} <= {:.3}",
            sideways_width,
            low_vol_width
        );
    }

    #[test]
    fn test_sequence_reconstruction_trait() {
        let analyzer = SequenceAnalyzer::new(SequenceReconstructionConfig::default());
        let candles = create_test_candles(vec![
            (100.0, 105.0, 95.0, 102.0, 1000.0),
            (102.0, 108.0, 98.0, 106.0, 1000.0),
        ]);

        // Test ranges reconstruction
        let ranges = analyzer.sequences_to_ranges(&candles, 100.0).unwrap();
        assert_eq!(ranges.len(), 5);

        // Test probabilities reconstruction
        let model_probs = vec![0.1, 0.2, 0.4, 0.2, 0.1];
        let probs = analyzer
            .sequences_to_probabilities(&candles, &model_probs)
            .unwrap();
        assert_eq!(probs, model_probs);
    }
}
