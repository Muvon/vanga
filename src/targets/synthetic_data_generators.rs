//! Synthetic market data generators for comprehensive target balance testing
//!
//! Provides realistic market data generation for various scenarios:
//! - Trending markets (bull/bear with different strengths)
//! - Sideways/consolidation periods
//! - Volatility regimes (low/high/expanding/contracting)
//! - Momentum patterns (accelerating/decelerating/reversing)
//! - Mixed market conditions

use crate::data::structures::MarketDataRow;
use std::f64::consts::PI;

/// Comprehensive market data generator with realistic patterns
pub struct SyntheticMarketGenerator {
    /// Random seed for reproducible results
    pub seed: u64,
    /// Current random state
    rng_state: u64,
}

impl SyntheticMarketGenerator {
    /// Create new generator with seed
    pub fn new(seed: u64) -> Self {
        Self {
            seed,
            rng_state: seed,
        }
    }

    /// Simple linear congruential generator for reproducible randomness
    fn next_random(&mut self) -> f64 {
        self.rng_state = self.rng_state.wrapping_mul(1103515245).wrapping_add(12345);
        (self.rng_state % 2147483647) as f64 / 2147483647.0
    }

    /// Generate random value in range [-1, 1]
    fn random_centered(&mut self) -> f64 {
        self.next_random() * 2.0 - 1.0
    }

    /// Generate realistic OHLC from close price and volatility
    fn generate_ohlc(&mut self, close: f64, volatility: f64) -> (f64, f64, f64, f64) {
        let vol_factor = volatility * 0.01; // Convert to percentage

        // Generate realistic intraday range
        let range_size = close * vol_factor * (0.5 + self.next_random() * 1.5);
        let mid_point = close + self.random_centered() * range_size * 0.3;

        let high = mid_point + range_size * (0.3 + self.next_random() * 0.7);
        let low = mid_point - range_size * (0.3 + self.next_random() * 0.7);

        // Open is typically close to previous close (handled by caller)
        let open = close + self.random_centered() * range_size * 0.2;

        (
            open,
            high.max(close).max(open),
            low.min(close).min(open),
            close,
        )
    }

    /// Generate realistic volume based on price movement and volatility
    fn generate_volume(&mut self, price_change_pct: f64, volatility: f64, base_volume: f64) -> f64 {
        let volume_multiplier = 1.0 + price_change_pct.abs() * 2.0 + volatility * 0.1;
        let noise = 0.8 + self.next_random() * 0.4; // 0.8 to 1.2 multiplier
        base_volume * volume_multiplier * noise
    }

    /// Generate trending market with configurable parameters
    pub fn generate_trending_market(
        &mut self,
        start_price: f64,
        length: usize,
        trend_strength: f64,    // Price change per period
        trend_consistency: f64, // 0.0 = very noisy, 1.0 = very consistent
        base_volatility: f64,   // Base volatility level
        base_volume: f64,       // Base volume level
    ) -> Vec<MarketDataRow> {
        let mut data: Vec<MarketDataRow> = Vec::with_capacity(length);
        let mut current_price = start_price;

        for i in 0..length {
            // Trend component with consistency factor
            let trend_component = trend_strength
                * (trend_consistency + (1.0 - trend_consistency) * self.random_centered());

            // Volatility noise
            let volatility_noise = self.random_centered() * base_volatility * current_price * 0.01;

            // Mean reversion component (prevents unrealistic price explosion)
            let mean_reversion = if i > 10 {
                let price_change_pct = (current_price - start_price) / start_price;
                -price_change_pct * 0.1 * current_price // Weak mean reversion
            } else {
                0.0
            };

            // Calculate new price
            let price_change = trend_component + volatility_noise + mean_reversion;
            current_price += price_change;

            // Ensure price stays positive
            current_price = current_price.max(start_price * 0.1);

            // Generate OHLC
            let (open, high, low, _close) = self.generate_ohlc(current_price, base_volatility);

            // Generate volume
            let price_change_pct = price_change / current_price;
            let volume = self.generate_volume(price_change_pct, base_volatility, base_volume);

            data.push(MarketDataRow {
                timestamp: i as i64,
                open: if i == 0 { start_price } else { open },
                high,
                low,
                close: current_price,
                volume,
            });
        }

        data
    }

    /// Generate sideways/consolidation market
    pub fn generate_sideways_market(
        &mut self,
        center_price: f64,
        length: usize,
        range_size_pct: f64,  // Range as percentage of center price
        cycle_period: f64,    // Oscillation period
        base_volatility: f64, // Base volatility level
        base_volume: f64,     // Base volume level
    ) -> Vec<MarketDataRow> {
        let mut data: Vec<MarketDataRow> = Vec::with_capacity(length);
        let range_size = center_price * range_size_pct * 0.01;

        for i in 0..length {
            // Cyclical component
            let cycle_phase = (i as f64) * 2.0 * PI / cycle_period;
            let cycle_component =
                (cycle_phase.sin() + 0.3 * (cycle_phase * 2.0).sin()) * range_size * 0.4;

            // Random walk within range
            let random_component = self.random_centered() * range_size * 0.3;

            // Mean reversion to center
            let current_deviation = if i > 0 {
                data[i - 1].close - center_price
            } else {
                0.0
            };
            let mean_reversion = -current_deviation * 0.1;

            // Calculate price
            let price = center_price + cycle_component + random_component + mean_reversion;

            // Generate OHLC
            let (open, high, low, _close) = self.generate_ohlc(price, base_volatility);

            // Generate volume (lower in sideways markets)
            let volume = self.generate_volume(0.0, base_volatility * 0.8, base_volume * 0.9);

            data.push(MarketDataRow {
                timestamp: i as i64,
                open: if i == 0 { center_price } else { open },
                high,
                low,
                close: price,
                volume,
            });
        }

        data
    }

    /// Generate market with changing volatility regime
    pub fn generate_volatility_regime_market(
        &mut self,
        start_price: f64,
        length: usize,
        initial_volatility: f64,
        volatility_trend: f64, // Change in volatility per period
        base_volume: f64,
    ) -> Vec<MarketDataRow> {
        let mut data: Vec<MarketDataRow> = Vec::with_capacity(length);
        let mut current_price = start_price;
        let mut current_volatility = initial_volatility;

        for i in 0..length {
            // Update volatility
            current_volatility += volatility_trend + self.random_centered() * 0.1;
            current_volatility = current_volatility.clamp(0.5, 10.0); // Reasonable bounds

            // Price movement with current volatility
            let price_change = self.random_centered() * current_volatility * current_price * 0.01;
            current_price += price_change;
            current_price = current_price.max(start_price * 0.1);

            // Generate OHLC with current volatility
            let (open, high, low, _close) = self.generate_ohlc(current_price, current_volatility);

            // Volume increases with volatility
            let volume = self.generate_volume(
                price_change / current_price,
                current_volatility,
                base_volume,
            );

            data.push(MarketDataRow {
                timestamp: i as i64,
                open: if i == 0 { start_price } else { open },
                high,
                low,
                close: current_price,
                volume,
            });
        }

        data
    }

    /// Generate market with momentum changes
    pub fn generate_momentum_change_market(
        &mut self,
        start_price: f64,
        length: usize,
        momentum_periods: Vec<(usize, f64)>, // (period_length, momentum_strength)
        base_volatility: f64,
        base_volume: f64,
    ) -> Vec<MarketDataRow> {
        let mut data: Vec<MarketDataRow> = Vec::with_capacity(length);
        let mut current_price = start_price;
        let mut period_index = 0;
        let mut period_progress = 0;

        for i in 0..length {
            // Determine current momentum
            let current_momentum = if period_index < momentum_periods.len() {
                let (period_length, _momentum_strength) = momentum_periods[period_index];

                // Check if we need to move to next period
                if period_progress >= period_length {
                    period_index += 1;
                    period_progress = 0;
                }

                if period_index < momentum_periods.len() {
                    momentum_periods[period_index].1
                } else {
                    0.0 // No more periods defined
                }
            } else {
                0.0
            };

            // Apply momentum with some noise
            let momentum_component = current_momentum * (0.8 + self.next_random() * 0.4);
            let volatility_noise = self.random_centered() * base_volatility * current_price * 0.01;

            current_price += momentum_component + volatility_noise;
            current_price = current_price.max(start_price * 0.1);

            // Generate OHLC
            let (open, high, low, _close) = self.generate_ohlc(current_price, base_volatility);

            // Generate volume
            let price_change_pct = (momentum_component + volatility_noise) / current_price;
            let volume = self.generate_volume(price_change_pct, base_volatility, base_volume);

            data.push(MarketDataRow {
                timestamp: i as i64,
                open: if i == 0 { start_price } else { open },
                high,
                low,
                close: current_price,
                volume,
            });

            period_progress += 1;
        }

        data
    }

    /// Generate realistic crypto-like market with multiple regimes
    pub fn generate_realistic_crypto_market(
        &mut self,
        start_price: f64,
        length: usize,
        base_volume: f64,
    ) -> Vec<MarketDataRow> {
        let mut data: Vec<MarketDataRow> = Vec::with_capacity(length);
        let mut current_price = start_price;

        // Define market regimes
        let regime_length = length / 5; // 5 different regimes

        for i in 0..length {
            let regime = i / regime_length;

            let (trend_strength, volatility, volume_multiplier) = match regime {
                0 => (2.0, 1.5, 1.0),  // Slow uptrend, low vol
                1 => (5.0, 3.0, 1.5),  // Strong uptrend, medium vol
                2 => (0.0, 1.0, 0.8),  // Sideways, low vol
                3 => (-3.0, 4.0, 1.8), // Downtrend, high vol
                _ => (1.0, 2.0, 1.2),  // Recovery, medium vol
            };

            // Apply regime characteristics
            let trend_component = trend_strength + self.random_centered() * trend_strength * 0.5;
            let volatility_noise = self.random_centered() * volatility * current_price * 0.01;

            // Occasional volatility spikes (crypto-like)
            let spike_probability = 0.05; // 5% chance
            let volatility_spike = if self.next_random() < spike_probability {
                self.random_centered() * current_price * 0.05 // Up to 5% spike
            } else {
                0.0
            };

            current_price += trend_component + volatility_noise + volatility_spike;
            current_price = current_price.max(start_price * 0.1);

            // Generate OHLC
            let (open, high, low, _close) = self.generate_ohlc(current_price, volatility);

            // Generate volume
            let total_change = trend_component + volatility_noise + volatility_spike;
            let price_change_pct = total_change / current_price;
            let volume = self.generate_volume(
                price_change_pct,
                volatility,
                base_volume * volume_multiplier,
            );

            data.push(MarketDataRow {
                timestamp: i as i64,
                open: if i == 0 { start_price } else { open },
                high,
                low,
                close: current_price,
                volume,
            });
        }

        data
    }
}

/// Predefined market scenarios for testing
pub struct MarketScenarios;

impl MarketScenarios {
    /// Generate BTC-like bull market
    pub fn btc_bull_market(length: usize) -> Vec<MarketDataRow> {
        let mut generator = SyntheticMarketGenerator::new(12345);
        generator.generate_trending_market(
            50000.0, // Start price
            length, 100.0,  // Strong uptrend
            0.7,    // Fairly consistent
            2.5,    // Moderate volatility
            1000.0, // Base volume
        )
    }

    /// Generate ETH-like bear market
    pub fn eth_bear_market(length: usize) -> Vec<MarketDataRow> {
        let mut generator = SyntheticMarketGenerator::new(54321);
        generator.generate_trending_market(
            3000.0, // Start price
            length, -50.0,  // Strong downtrend
            0.6,    // Somewhat consistent
            3.5,    // Higher volatility
            1500.0, // Base volume
        )
    }

    /// Generate altcoin sideways market
    pub fn altcoin_sideways_market(length: usize) -> Vec<MarketDataRow> {
        let mut generator = SyntheticMarketGenerator::new(98765);
        generator.generate_sideways_market(
            1.0, // Start price
            length, 5.0,   // 5% range
            20.0,  // 20-period cycle
            2.0,   // Moderate volatility
            800.0, // Base volume
        )
    }

    /// Generate high volatility market (like during news events)
    pub fn high_volatility_market(length: usize) -> Vec<MarketDataRow> {
        let mut generator = SyntheticMarketGenerator::new(13579);
        generator.generate_volatility_regime_market(
            25000.0, // Start price
            length, 5.0,    // High initial volatility
            0.1,    // Increasing volatility
            1200.0, // Base volume
        )
    }

    /// Generate momentum reversal market
    pub fn momentum_reversal_market(length: usize) -> Vec<MarketDataRow> {
        let mut generator = SyntheticMarketGenerator::new(24680);

        let momentum_periods = vec![
            (length / 3, 80.0),   // Strong up momentum
            (length / 3, -120.0), // Strong down momentum
            (length / 3, 20.0),   // Weak up momentum
        ];

        generator.generate_momentum_change_market(
            40000.0, // Start price
            length,
            momentum_periods,
            3.0,    // Moderate volatility
            1100.0, // Base volume
        )
    }

    /// Generate realistic mixed market conditions
    pub fn realistic_mixed_market(length: usize) -> Vec<MarketDataRow> {
        let mut generator = SyntheticMarketGenerator::new(11111);
        generator.generate_realistic_crypto_market(
            35000.0, // Start price
            length, 1000.0, // Base volume
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_synthetic_generator_reproducibility() {
        let mut gen1 = SyntheticMarketGenerator::new(12345);
        let mut gen2 = SyntheticMarketGenerator::new(12345);

        let data1 = gen1.generate_trending_market(1000.0, 100, 10.0, 0.8, 2.0, 1000.0);
        let data2 = gen2.generate_trending_market(1000.0, 100, 10.0, 0.8, 2.0, 1000.0);

        assert_eq!(data1.len(), data2.len());

        for (row1, row2) in data1.iter().zip(data2.iter()) {
            assert!(
                (row1.close - row2.close).abs() < 1e-10,
                "Reproducibility failed"
            );
        }
    }

    #[test]
    fn test_market_scenarios() {
        let length = 100;

        let scenarios = vec![
            ("BTC Bull", MarketScenarios::btc_bull_market(length)),
            ("ETH Bear", MarketScenarios::eth_bear_market(length)),
            (
                "Altcoin Sideways",
                MarketScenarios::altcoin_sideways_market(length),
            ),
            (
                "High Volatility",
                MarketScenarios::high_volatility_market(length),
            ),
            (
                "Momentum Reversal",
                MarketScenarios::momentum_reversal_market(length),
            ),
            (
                "Realistic Mixed",
                MarketScenarios::realistic_mixed_market(length),
            ),
        ];

        for (name, data) in scenarios {
            assert_eq!(data.len(), length, "{} scenario length mismatch", name);

            // Verify data integrity
            for (i, row) in data.iter().enumerate() {
                assert!(
                    row.high >= row.low,
                    "{} scenario: high < low at index {}",
                    name,
                    i
                );
                assert!(
                    row.high >= row.close,
                    "{} scenario: high < close at index {}",
                    name,
                    i
                );
                assert!(
                    row.low <= row.close,
                    "{} scenario: low > close at index {}",
                    name,
                    i
                );
                assert!(
                    row.volume > 0.0,
                    "{} scenario: zero volume at index {}",
                    name,
                    i
                );
                assert!(
                    row.close > 0.0,
                    "{} scenario: zero price at index {}",
                    name,
                    i
                );
            }
        }
    }
}
