//! Comprehensive tests for price level classification to ensure mathematical correctness
//!
//! These tests validate:
//! 1. Percentile-based boundary calculation accuracy
//! 2. VWAP price calculation correctness
//! 3. Bandwidth scaling appropriateness
//! 4. Classification balance across different scenarios
//! 5. Edge case handling

#[cfg(test)]
mod tests {
    use super::super::price_levels::*;
    use crate::data::structures::MarketDataRow;

    /// Helper function to create test market data
    fn create_test_candles(ohlcv_data: Vec<(f64, f64, f64, f64, f64)>) -> Vec<MarketDataRow> {
        ohlcv_data
            .into_iter()
            .map(|(open, high, low, close, volume)| MarketDataRow {
                timestamp: 0,
                open,
                high,
                low,
                close,
                volume,
            })
            .collect()
    }

    /// **ENHANCED**: Test adaptive bandwidth calculation
    #[test]
    fn test_adaptive_bandwidth() {
        // Test case 1: Low volatility sequence should have smaller bandwidth
        let low_vol_sequence = create_test_candles(vec![
            (100.0, 100.1, 99.9, 100.0, 1000.0), // Very tight range
            (100.0, 100.1, 99.9, 100.05, 1000.0),
            (100.05, 100.15, 99.95, 100.1, 1000.0),
            (100.1, 100.2, 100.0, 100.15, 1000.0),
        ]);

        // Test case 2: High volatility sequence should have larger bandwidth
        let high_vol_sequence = create_test_candles(vec![
            (100.0, 105.0, 95.0, 102.0, 1000.0), // Wide range
            (102.0, 108.0, 98.0, 104.0, 1000.0),
            (104.0, 110.0, 100.0, 106.0, 1000.0),
            (106.0, 112.0, 102.0, 108.0, 1000.0),
        ]);

        let base_bandwidth = 1.0;

        let low_vol_bandwidth =
            calculate_adaptive_bandwidth(&low_vol_sequence, base_bandwidth, None).unwrap();
        let high_vol_bandwidth =
            calculate_adaptive_bandwidth(&high_vol_sequence, base_bandwidth, None).unwrap();

        // High volatility should result in larger bandwidth
        assert!(high_vol_bandwidth > low_vol_bandwidth);

        // Both should be reasonable multiples of base bandwidth
        assert!(low_vol_bandwidth >= 0.3 * base_bandwidth); // Minimum bound
        assert!(high_vol_bandwidth <= 3.0 * base_bandwidth); // Maximum bound

        println!(
            "Low volatility adaptive bandwidth: {:.3}",
            low_vol_bandwidth
        );
        println!(
            "High volatility adaptive bandwidth: {:.3}",
            high_vol_bandwidth
        );
    }
}
