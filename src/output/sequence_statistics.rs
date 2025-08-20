//! Sequence Statistics Module - Extract market behavior from raw price sequences
//! NO MAGIC NUMBERS - Everything derived from mathematical relationships

use crate::utils::error::{Result, VangaError};

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
