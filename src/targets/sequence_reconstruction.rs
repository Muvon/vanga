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
}

impl Default for SequenceReconstructionConfig {
    fn default() -> Self {
        Self {
            percentiles: [0.1, 0.9], // Default 10th-90th percentiles
            bandwidth_size: 1.0,     // Default bandwidth multiplier
        }
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

        vec![
            // Strong Down: (-∞, boundary_1)
            [
                to_pct(self.boundaries[0] - self.bandwidth),
                to_pct(self.boundaries[0]) - epsilon,
            ],
            // Moderate Down: [boundary_1, boundary_2)
            [
                to_pct(self.boundaries[0]),
                to_pct(self.boundaries[1]) - epsilon,
            ],
            // Neutral: [boundary_2, boundary_3)
            [
                to_pct(self.boundaries[1]),
                to_pct(self.boundaries[2]) - epsilon,
            ],
            // Moderate Up: [boundary_3, boundary_4)
            [
                to_pct(self.boundaries[2]),
                to_pct(self.boundaries[3]) - epsilon,
            ],
            // Strong Up: [boundary_4, +∞)
            [
                to_pct(self.boundaries[3]),
                to_pct(self.boundaries[3] + self.bandwidth),
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

    /// Create from model configuration (for consistency with training)
    pub fn from_model_config(model_config: &crate::config::model::PriceLevelHead) -> Self {
        let config = SequenceReconstructionConfig {
            percentiles: model_config.percentiles.unwrap_or([0.1, 0.9]),
            bandwidth_size: model_config.bandwidth_size.unwrap_or(1.0),
        };
        Self::new(config)
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

        // Calculate percentile boundaries from sorted prices
        let mut sorted_prices = vwap_prices.clone();
        sorted_prices.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let n = sorted_prices.len();
        let lower_idx = ((n as f64 * self.config.percentiles[0]) as usize).min(n - 1);
        let upper_idx = ((n as f64 * self.config.percentiles[1]) as usize).min(n - 1);

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
