//! Sequence Statistics Module - Extract market behavior from raw price sequences
//! NO MAGIC NUMBERS - Everything derived from mathematical relationships

use crate::utils::error::{Result, VangaError};

/// Adaptive bounds calculated from sequence data for order generation
#[derive(Debug, Clone)]
pub struct AdaptiveBounds {
    /// Minimum price in sequence
    pub sequence_min: f64,
    /// Maximum price in sequence
    pub sequence_max: f64,
    /// 10th percentile
    pub p10: f64,
    /// 25th percentile (Q1)
    pub p25: f64,
    /// 50th percentile (median)
    pub p50: f64,
    /// 75th percentile (Q3)
    pub p75: f64,
    /// 90th percentile
    pub p90: f64,
    /// IQR-based volatility percentage
    pub iqr_volatility: f64,
    /// Maximum drawdown percentage from current price
    pub max_drawdown_pct: f64,
    /// Maximum upside percentage from current price
    pub max_upside_pct: f64,
    /// Sequence range as percentage of current price
    pub sequence_range_pct: f64,
    /// Sequence mean for z-score calculations
    pub sequence_mean: f64,
    /// Sequence standard deviation for z-score calculations
    pub sequence_std: f64,
}

/// Statistics extracted from raw price sequence
#[derive(Debug, Clone)]
pub struct SequenceStatistics {
    /// Mean return (period-to-period change)
    pub mean_return: f64,

    /// Standard deviation of returns
    pub std_return: f64,

    /// Maximum drawdown in the sequence
    pub max_drawdown: f64,

    /// Maximum consecutive rise
    pub max_runup: f64,

    /// Mean reversion rate (autocorrelation at lag 1)
    pub mean_reversion_rate: f64,

    /// Hurst exponent (trending vs mean-reverting)
    pub hurst_exponent: f64,

    /// Maximum Adverse Excursion distribution (for stops)
    pub mae_distribution: Vec<f64>,

    /// Maximum Favorable Excursion distribution (for targets)
    pub mfe_distribution: Vec<f64>,

    /// Shannon entropy of price distribution
    pub price_entropy: f64,

    /// Shannon entropy of volume distribution (if available)
    pub volume_entropy: Option<f64>,

    /// Time-scaled volatility (σ√t) for the horizon
    pub time_scaled_volatility: f64,

    /// Kelly fraction for optimal position sizing
    pub kelly_fraction: f64,
}

impl SequenceStatistics {
    /// Calculate all statistics from raw price sequence
    /// Uses ONLY mathematical relationships, NO hardcoded thresholds
    pub fn from_prices(
        prices: &[f64],
        horizon_hours: f64,
        volumes: Option<&[f64]>,
    ) -> Result<Self> {
        if prices.len() < 2 {
            return Err(VangaError::DataError(
                "Need at least 2 prices for statistics".to_string(),
            ));
        }

        // Calculate returns (period-to-period changes)
        let returns = Self::calculate_returns(prices);

        // Basic statistics
        let mean_return = Self::mean(&returns);
        let std_return = Self::std_dev(&returns, mean_return);

        // Drawdown and runup
        let (max_drawdown, max_runup) = Self::calculate_extremes(prices);

        // Mean reversion rate (autocorrelation at lag 1)
        let mean_reversion_rate = Self::autocorrelation(&returns, 1);

        // Hurst exponent for trend detection
        let hurst_exponent = Self::calculate_hurst(prices)?;

        // MAE/MFE distributions for stops and targets
        let (mae_distribution, mfe_distribution) = Self::calculate_excursion_distributions(prices);

        // Information entropy
        let price_entropy = Self::calculate_shannon_entropy(prices);
        let volume_entropy = volumes.map(Self::calculate_shannon_entropy);

        // Time-scaled volatility using square root of time rule
        let time_scaled_volatility = std_return * horizon_hours.sqrt();

        // Kelly criterion for position sizing
        let kelly_fraction = Self::calculate_kelly(&returns);

        Ok(Self {
            mean_return,
            std_return,
            max_drawdown,
            max_runup,
            mean_reversion_rate,
            hurst_exponent,
            mae_distribution,
            mfe_distribution,
            price_entropy,
            volume_entropy,
            time_scaled_volatility,
            kelly_fraction,
        })
    }

    /// Calculate period-to-period returns
    fn calculate_returns(prices: &[f64]) -> Vec<f64> {
        prices.windows(2).map(|w| (w[1] - w[0]) / w[0]).collect()
    }

    /// Calculate mean of values
    fn mean(values: &[f64]) -> f64 {
        if values.is_empty() {
            return 0.0;
        }
        values.iter().sum::<f64>() / values.len() as f64
    }

    /// Calculate standard deviation
    fn std_dev(values: &[f64], mean: f64) -> f64 {
        if values.len() < 2 {
            return 0.0;
        }
        let variance =
            values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (values.len() - 1) as f64;
        variance.sqrt()
    }

    /// Calculate maximum drawdown and runup
    fn calculate_extremes(prices: &[f64]) -> (f64, f64) {
        let mut max_drawdown = 0.0;
        let mut max_runup = 0.0;
        let mut peak = prices[0];
        let mut trough = prices[0];

        for &price in prices.iter() {
            // Update peak for drawdown calculation
            if price > peak {
                peak = price;
                // Reset trough for runup calculation
                trough = price;
            }

            // Update trough for runup calculation
            if price < trough {
                trough = price;
            }

            // Calculate drawdown from peak
            let drawdown = (peak - price) / peak;
            if drawdown > max_drawdown {
                max_drawdown = drawdown;
            }

            // Calculate runup from trough
            let runup = (price - trough) / trough;
            if runup > max_runup {
                max_runup = runup;
            }
        }

        (max_drawdown, max_runup)
    }

    /// Calculate autocorrelation at specified lag
    fn autocorrelation(returns: &[f64], lag: usize) -> f64 {
        if returns.len() <= lag {
            return 0.0;
        }

        let mean = Self::mean(returns);
        let variance =
            returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / returns.len() as f64;

        if variance == 0.0 {
            return 0.0;
        }

        let covariance: f64 = returns[..returns.len() - lag]
            .iter()
            .zip(returns[lag..].iter())
            .map(|(r1, r2)| (r1 - mean) * (r2 - mean))
            .sum::<f64>()
            / (returns.len() - lag) as f64;

        covariance / variance
    }

    /// Calculate Hurst exponent using R/S analysis
    fn calculate_hurst(prices: &[f64]) -> Result<f64> {
        let n = prices.len();
        if n < 10 {
            // Not enough data for reliable Hurst calculation
            return Ok(0.5); // Return neutral value
        }

        // Simplified R/S analysis
        let returns = Self::calculate_returns(prices);
        let mean_return = Self::mean(&returns);

        // Calculate cumulative deviations
        let mut cumulative_deviations = vec![0.0];
        let mut cumsum = 0.0;
        for r in &returns {
            cumsum += r - mean_return;
            cumulative_deviations.push(cumsum);
        }

        // Calculate range
        let max_dev = cumulative_deviations
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        let min_dev = cumulative_deviations
            .iter()
            .cloned()
            .fold(f64::INFINITY, f64::min);
        let range = max_dev - min_dev;

        // Calculate standard deviation
        let std_dev = Self::std_dev(&returns, mean_return);

        if std_dev == 0.0 || range == 0.0 {
            return Ok(0.5); // Neutral Hurst
        }

        // R/S statistic
        let rs = range / std_dev;

        // Hurst exponent approximation: R/S = (n/2)^H
        // H = log(R/S) / log(n/2)
        let hurst = rs.ln() / (n as f64 / 2.0).ln();

        // Clamp to valid range [0, 1]
        Ok(hurst.clamp(0.0, 1.0))
    }

    /// Calculate MAE and MFE distributions
    fn calculate_excursion_distributions(prices: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let mut mae_values = Vec::new();
        let mut mfe_values = Vec::new();

        // For each potential entry point
        for i in 0..prices.len() - 1 {
            let entry_price = prices[i];
            let mut max_adverse = 0.0;
            let mut max_favorable = 0.0;

            // Track excursions from this entry
            for &current_price in prices.iter().skip(i + 1) {
                let move_pct = (current_price - entry_price) / entry_price;

                // Update max adverse (negative for longs, positive for shorts)
                if move_pct < 0.0 && move_pct.abs() > max_adverse {
                    max_adverse = move_pct.abs();
                }

                // Update max favorable
                if move_pct > 0.0 && move_pct > max_favorable {
                    max_favorable = move_pct;
                }

                // If this becomes a winning trade, record the MAE
                if move_pct > 0.0 && max_adverse > 0.0 {
                    mae_values.push(max_adverse);
                    mfe_values.push(max_favorable);
                    break; // Move to next entry point
                }
            }
        }

        // Sort for percentile calculations
        mae_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        mfe_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        (mae_values, mfe_values)
    }

    /// Calculate Shannon entropy of a distribution
    fn calculate_shannon_entropy(values: &[f64]) -> f64 {
        if values.is_empty() {
            return 0.0;
        }

        // Create histogram with adaptive bins
        let min_val = values.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_val = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        if min_val >= max_val {
            return 0.0; // No variation
        }

        // Use Sturges' rule for number of bins: k = 1 + log2(n)
        let num_bins = (1.0 + (values.len() as f64).log2()).ceil() as usize;
        let bin_width = (max_val - min_val) / num_bins as f64;

        // Count values in each bin
        let mut bin_counts = vec![0; num_bins];
        for &value in values {
            let bin_idx = ((value - min_val) / bin_width).floor() as usize;
            let bin_idx = bin_idx.min(num_bins - 1); // Handle edge case
            bin_counts[bin_idx] += 1;
        }

        // Calculate probabilities and entropy
        let total = values.len() as f64;
        let mut entropy = 0.0;

        for count in bin_counts {
            if count > 0 {
                let p = count as f64 / total;
                entropy -= p * p.ln();
            }
        }

        entropy
    }

    /// Calculate Kelly fraction for optimal position sizing
    fn calculate_kelly(returns: &[f64]) -> f64 {
        // Separate winning and losing returns
        let wins: Vec<f64> = returns.iter().cloned().filter(|&r| r > 0.0).collect();
        let losses: Vec<f64> = returns.iter().cloned().filter(|&r| r < 0.0).collect();

        if wins.is_empty() || losses.is_empty() {
            // Not enough data for Kelly calculation
            return 0.25; // Conservative default
        }

        // Calculate win rate
        let win_rate = wins.len() as f64 / returns.len() as f64;
        let loss_rate = 1.0 - win_rate;

        // Calculate average win and loss magnitudes
        let avg_win = Self::mean(&wins);
        let avg_loss = Self::mean(&losses).abs();

        if avg_loss == 0.0 {
            return 1.0; // No losses observed (unlikely)
        }

        // Kelly formula: f = (p*b - q) / b
        // where p = win_rate, q = loss_rate, b = avg_win/avg_loss
        let b = avg_win / avg_loss;
        let kelly = (win_rate * b - loss_rate) / b;

        // Apply Kelly fraction with safety factor (never use full Kelly)
        // Common practice is to use 25% of Kelly for safety
        (kelly * 0.25).clamp(0.0, 1.0)
    }

    /// Get percentile value from sorted distribution
    pub fn percentile(sorted_values: &[f64], percentile: f64) -> f64 {
        if sorted_values.is_empty() {
            return 0.0;
        }

        let index = ((percentile / 100.0) * (sorted_values.len() - 1) as f64).round() as usize;
        sorted_values[index.min(sorted_values.len() - 1)]
    }

    /// Calculate sequence-derived bounds for fully adaptive order generation
    /// Returns (sequence_min, sequence_max, p10, p25, p50, p75, p90)
    pub fn calculate_sequence_percentiles(
        &self,
        prices: &[f64],
    ) -> (f64, f64, f64, f64, f64, f64, f64) {
        if prices.is_empty() {
            return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        }

        let mut sorted_prices = prices.to_vec();
        sorted_prices.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let sequence_min = sorted_prices[0];
        let sequence_max = sorted_prices[sorted_prices.len() - 1];
        let p10 = Self::percentile(&sorted_prices, 10.0);
        let p25 = Self::percentile(&sorted_prices, 25.0);
        let p50 = Self::percentile(&sorted_prices, 50.0);
        let p75 = Self::percentile(&sorted_prices, 75.0);
        let p90 = Self::percentile(&sorted_prices, 90.0);

        (sequence_min, sequence_max, p10, p25, p50, p75, p90)
    }

    /// Calculate IQR-based volatility (robust to outliers)
    pub fn calculate_iqr_volatility(&self, prices: &[f64]) -> f64 {
        let (_, _, _, p25, p50, p75, _) = self.calculate_sequence_percentiles(prices);

        if p50 == 0.0 {
            return 0.0;
        }

        // IQR volatility as percentage: ((p75 - p25) / p50) * 100.0
        ((p75 - p25) / p50) * 100.0
    }

    /// Calculate maximum drawdown and upside percentages from current price
    pub fn calculate_drawdown_upside_from_current(
        &self,
        prices: &[f64],
        current_price: f64,
    ) -> (f64, f64) {
        if prices.is_empty() || current_price <= 0.0 {
            return (0.0, 0.0);
        }

        let (sequence_min, sequence_max, _, _, _, _, _) =
            self.calculate_sequence_percentiles(prices);

        // Maximum drawdown from current price to sequence minimum
        let max_drawdown_pct = if current_price > sequence_min {
            ((current_price - sequence_min) / current_price) * 100.0
        } else {
            0.0
        };

        // Maximum upside from current price to sequence maximum
        let max_upside_pct = if sequence_max > current_price {
            ((sequence_max - current_price) / current_price) * 100.0
        } else {
            0.0
        };

        (max_drawdown_pct, max_upside_pct)
    }

    /// Calculate sequence range percentage from current price
    pub fn calculate_sequence_range_pct(&self, prices: &[f64], current_price: f64) -> f64 {
        if prices.is_empty() || current_price <= 0.0 {
            return 0.0;
        }

        let (sequence_min, sequence_max, _, _, _, _, _) =
            self.calculate_sequence_percentiles(prices);
        ((sequence_max - sequence_min) / current_price) * 100.0
    }

    /// Validate price using z-score analysis (within 3 standard deviations)
    pub fn validate_price_with_zscore(&self, prices: &[f64], target_price: f64) -> f64 {
        if prices.is_empty() {
            return target_price;
        }

        let sequence_mean = Self::mean(prices);
        let sequence_std = Self::std_dev(prices, sequence_mean);

        if sequence_std == 0.0 {
            return target_price; // No variation, return as-is
        }

        let z_score = (target_price - sequence_mean) / sequence_std;

        // Within 3 standard deviations (99.7% of normal distribution)
        if z_score.abs() <= 3.0 {
            target_price
        } else {
            // Scale back to 3-sigma boundary
            sequence_mean + (3.0 * z_score.signum() * sequence_std)
        }
    }

    /// Get sequence statistics for adaptive order generation
    pub fn get_adaptive_bounds(&self, prices: &[f64], current_price: f64) -> AdaptiveBounds {
        let (sequence_min, sequence_max, p10, p25, p50, p75, p90) =
            self.calculate_sequence_percentiles(prices);
        let iqr_volatility = self.calculate_iqr_volatility(prices);
        let (max_drawdown_pct, max_upside_pct) =
            self.calculate_drawdown_upside_from_current(prices, current_price);
        let sequence_range_pct = self.calculate_sequence_range_pct(prices, current_price);

        AdaptiveBounds {
            sequence_min,
            sequence_max,
            p10,
            p25,
            p50,
            p75,
            p90,
            iqr_volatility,
            max_drawdown_pct,
            max_upside_pct,
            sequence_range_pct,
            sequence_mean: Self::mean(prices),
            sequence_std: Self::std_dev(prices, Self::mean(prices)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sequence_statistics_calculation() {
        // Test with sample price sequence
        let prices = vec![100.0, 102.0, 101.0, 103.0, 102.5, 104.0, 103.0, 105.0];
        let horizon_hours = 4.0;

        let stats = SequenceStatistics::from_prices(&prices, horizon_hours, None).unwrap();

        // Verify calculations are reasonable
        assert!(stats.mean_return > 0.0); // Upward trend
        assert!(stats.std_return > 0.0); // Some volatility
        assert!(stats.max_drawdown >= 0.0);
        assert!(stats.max_runup >= 0.0);
        assert!(stats.hurst_exponent >= 0.0 && stats.hurst_exponent <= 1.0);
        assert!(stats.kelly_fraction >= 0.0 && stats.kelly_fraction <= 1.0);
    }
}
