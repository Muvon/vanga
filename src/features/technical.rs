// Technical indicators implementation with comprehensive indicators
use crate::config::features::TechnicalIndicatorsConfig;
use crate::features::ta_helpers::*; // Import TA crate helpers
use crate::utils::error::{Result, VangaError};
use polars::prelude::*;
use rayon::prelude::*;

/// OHLCV data structure to reduce function parameter count
#[derive(Clone)]
pub struct OhlcvData<'a> {
    pub open: &'a [f64],
    pub high: &'a [f64],
    pub low: &'a [f64],
    pub close: &'a [f64],
    pub volume: &'a [f64],
}

/// Generate comprehensive technical indicators for cryptocurrency data - PARALLELIZED
pub async fn generate_technical_indicators(
    mut df: DataFrame,
    config: &TechnicalIndicatorsConfig,
) -> Result<DataFrame> {
    log::info!("Generating comprehensive technical indicators with parallel processing...");

    // Detect timeframe for window adjustments
    let timeframe_minutes = crate::utils::parser::detect_timeframe_minutes(&df)? as f64;
    let timeframe_multiplier = timeframe_minutes / 60.0; // Relative to 1h baseline

    // Extract OHLCV data for calculations - PARALLEL EXTRACTION
    let close_prices = extract_numeric_column(&df, "close")?;
    let high_prices = extract_numeric_column(&df, "high")?;
    let low_prices = extract_numeric_column(&df, "low")?;
    let open_prices = extract_numeric_column(&df, "open")?;
    let volume = extract_numeric_column(&df, "volume")?;

    // Create OHLCV data structure
    let ohlcv = OhlcvData {
        open: &open_prices,
        high: &high_prices,
        low: &low_prices,
        close: &close_prices,
        volume: &volume,
    };

    // PARALLEL INDICATOR PROCESSING: Process all indicator groups concurrently
    let results = rayon::join(
        || {
            // Trend indicators group
            let mut trend_results = Vec::new();

            if !config.moving_averages.sma_periods.is_empty() {
                trend_results.push((
                    "sma",
                    add_sma_indicators(
                        df.clone(),
                        &close_prices,
                        &config.moving_averages.sma_periods,
                    ),
                ));
            }

            if !config.moving_averages.ema_periods.is_empty() {
                trend_results.push((
                    "ema",
                    add_ema_indicators(
                        df.clone(),
                        &close_prices,
                        &config.moving_averages.ema_periods,
                    ),
                ));
            }

            if !config.moving_averages.dema_periods.is_empty() {
                trend_results.push((
                    "dema",
                    add_dema_indicators(
                        df.clone(),
                        &close_prices,
                        &config.moving_averages.dema_periods,
                    ),
                ));
            }

            if !config.moving_averages.tema_periods.is_empty() {
                trend_results.push((
                    "tema",
                    add_tema_indicators(
                        df.clone(),
                        &close_prices,
                        &config.moving_averages.tema_periods,
                    ),
                ));
            }

            if !config.moving_averages.kama_periods.is_empty() {
                trend_results.push((
                    "kama",
                    add_kama_indicators(
                        df.clone(),
                        &close_prices,
                        &config.moving_averages.kama_periods,
                    ),
                ));
            }

            if !config.moving_averages.zlema_periods.is_empty() {
                trend_results.push((
                    "zlema",
                    add_zlema_indicators(
                        df.clone(),
                        &close_prices,
                        &config.moving_averages.zlema_periods,
                    ),
                ));
            }

            trend_results
        },
        || {
            // Momentum indicators group
            let mut momentum_results = Vec::new();

            if !config.momentum.rsi_periods.is_empty() {
                momentum_results.push((
                    "rsi",
                    add_rsi_indicators(df.clone(), &close_prices, &config.momentum.rsi_periods),
                ));
            }

            momentum_results
        },
    );

    // Apply trend indicators
    for (name, result) in results.0 {
        match result {
            Ok(updated_df) => {
                df = updated_df;
                log::debug!("Applied {} indicators", name);
            }
            Err(e) => log::warn!("Failed to apply {} indicators: {}", name, e),
        }
    }

    // Apply momentum indicators
    for (name, result) in results.1 {
        match result {
            Ok(updated_df) => {
                df = updated_df;
                log::debug!("Applied {} indicators", name);
            }
            Err(e) => log::warn!("Failed to apply {} indicators: {}", name, e),
        }
    }

    if config.trend.macd.enabled {
        df = add_macd_indicators(
            df,
            &close_prices,
            config.trend.macd.fast_period,
            config.trend.macd.slow_period,
            config.trend.macd.signal_period,
        )?;
    }

    if config.volatility.bollinger_bands.enabled {
        df = add_bollinger_bands(
            df,
            &close_prices,
            config.volatility.bollinger_bands.period,
            config.volatility.bollinger_bands.std_dev,
        )?;
    }

    // Momentum Indicators
    if !config.momentum.rsi_periods.is_empty() {
        df = add_rsi_indicators(df, &close_prices, &config.momentum.rsi_periods)?;
    }

    if config.momentum.stochastic {
        df = add_stochastic_indicators(df, &ohlcv, 14, 3)?;
    }

    if config.momentum.williams_r {
        df = add_williams_r(df, &high_prices, &low_prices, &close_prices, 14)?;
    }

    if !config.momentum.cci_periods.is_empty() {
        for &period in &config.momentum.cci_periods {
            df = add_cci_indicator(
                df,
                &open_prices,
                &high_prices,
                &low_prices,
                &close_prices,
                &volume,
                period,
            )?;
        }
    }

    // Volume Indicators
    if !config.volume.volume_sma_periods.is_empty() {
        df = add_volume_indicators(
            df,
            &close_prices,
            &volume,
            &config.volume.volume_sma_periods,
            &config.momentum,
        )?;
    }

    if config.volume.obv {
        df = add_obv_indicator(df, &close_prices, &volume)?;
    }

    if !config.volume.mfi_periods.is_empty() {
        for &period in &config.volume.mfi_periods {
            df = add_mfi_indicator(
                df,
                &high_prices,
                &low_prices,
                &close_prices,
                &volume,
                period,
            )?;
        }
    }

    // Volatility Indicators
    if !config.volatility.atr_periods.is_empty() {
        df = add_atr_indicators(
            df,
            &open_prices,
            &high_prices,
            &low_prices,
            &close_prices,
            &volume,
            &config.volatility.atr_periods,
            &config.momentum,
        )?;
    }

    if config.volatility.keltner_channels {
        df = add_keltner_channels(df, &ohlcv, 20, 2.0)?;
    }

    // Cryptocurrency-specific indicators (always enabled for crypto markets)
    df = add_crypto_specific_indicators(
        df,
        CryptoIndicatorParams {
            open: &open_prices,
            high: &high_prices,
            low: &low_prices,
            close: &close_prices,
            volume: &volume,
            advanced_config: &config.trend.advanced,
            timeframe_multiplier,
        },
    )?;

    // Consolidation/Neutral-detection features (CRITICAL for Class 2 learning)
    if config.trend.consolidation.enabled {
        df = add_consolidation_features(df, &ohlcv, config)?;
    }

    let total_features = df.width() - 6; // Subtract OHLCV + timestamp
    log::info!("Generated {} technical indicators", total_features);

    // Log momentum features if enabled
    if config.momentum.atr_momentum_enabled
        || config.momentum.volume_momentum_enabled
        || config.momentum.pv_divergence_enabled
    {
        let mut momentum_count = 0;
        if config.momentum.atr_momentum_enabled {
            momentum_count += config.volatility.atr_periods.len()
                * (config.momentum.momentum_periods.len() * 2 + 1);
        }
        if config.momentum.volume_momentum_enabled {
            momentum_count += config.volume.volume_sma_periods.len()
                * (config.momentum.momentum_periods.len() * 2 + 1);
        }
        if config.momentum.pv_divergence_enabled {
            momentum_count +=
                config.volume.volume_sma_periods.len() * 2 + config.momentum.momentum_periods.len();
        }
        log::info!(
            "  └─ Momentum features: {} (ATR: {}, Volume: {}, PV Divergence: {})",
            momentum_count,
            config.momentum.atr_momentum_enabled,
            config.momentum.volume_momentum_enabled,
            config.momentum.pv_divergence_enabled
        );
    }

    Ok(df)
}

/// Extract numeric column as Vec<f64>
pub fn extract_numeric_column(df: &DataFrame, column_name: &str) -> Result<Vec<f64>> {
    let series = df
        .column(column_name)
        .map_err(|_| VangaError::FeatureError(format!("Column '{}' not found", column_name)))?;

    let values: Result<Vec<f64>> = series
        .f64()
        .map_err(|_| VangaError::FeatureError(format!("Column '{}' is not numeric", column_name)))?
        .into_iter()
        .map(|opt_val| {
            opt_val.ok_or_else(|| {
                VangaError::FeatureError(format!("Null value in column '{}'", column_name))
            })
        })
        .collect();

    values
}

/// Calculate simple moving average
pub(crate) fn calculate_sma(data: &[f64], period: usize) -> Vec<f64> {
    let mut result = vec![f64::NAN; data.len()];

    if period > data.len() {
        return result;
    }

    for i in (period - 1)..data.len() {
        let sum: f64 = data[(i + 1 - period)..=i].iter().sum();
        result[i] = sum / period as f64;
    }

    result
}

/// Calculate momentum: (current - previous) / previous
/// Returns rate of change over specified period
pub(crate) fn calculate_momentum(data: &[f64], period: usize) -> Vec<f64> {
    let mut result = vec![f64::NAN; data.len()];

    if period == 0 || period >= data.len() {
        return result;
    }

    for i in period..data.len() {
        let previous = data[i - period];
        let current = data[i];

        if previous.abs() > 1e-10 && previous.is_finite() && current.is_finite() {
            result[i] = (current - previous) / previous;
        }
    }

    result
}

/// Calculate acceleration: change in momentum
pub(crate) fn calculate_acceleration(momentum: &[f64]) -> Vec<f64> {
    let mut result = vec![f64::NAN; momentum.len()];

    for i in 1..momentum.len() {
        if momentum[i].is_finite() && momentum[i - 1].is_finite() {
            result[i] = momentum[i] - momentum[i - 1];
        }
    }

    result
}

/// Calculate trend slope using linear regression over rolling window
pub(crate) fn calculate_trend_slope(data: &[f64], window: usize) -> Vec<f64> {
    let mut result = vec![f64::NAN; data.len()];

    if window < 2 || window > data.len() {
        return result;
    }

    for i in (window - 1)..data.len() {
        let window_data = &data[(i + 1 - window)..=i];

        // Linear regression: y = mx + b, we want slope m
        let n = window as f64;
        let mut sum_x = 0.0;
        let mut sum_y = 0.0;
        let mut sum_xy = 0.0;
        let mut sum_x2 = 0.0;

        let mut valid = true;
        for (j, &y) in window_data.iter().enumerate() {
            if !y.is_finite() {
                valid = false;
                break;
            }
            let x = j as f64;
            sum_x += x;
            sum_y += y;
            sum_xy += x * y;
            sum_x2 += x * x;
        }

        if valid {
            let denominator = n * sum_x2 - sum_x * sum_x;
            if denominator.abs() > 1e-10 {
                let slope = (n * sum_xy - sum_x * sum_y) / denominator;
                result[i] = slope;
            }
        }
    }

    result
}

/// Calculate On-Balance Volume (OBV)
pub(crate) fn calculate_obv(close: &[f64], volume: &[f64]) -> Vec<f64> {
    let mut result = vec![0.0; close.len()];

    if close.is_empty() {
        return result;
    }

    result[0] = volume[0];

    for i in 1..close.len() {
        if close[i] > close[i - 1] {
            result[i] = result[i - 1] + volume[i];
        } else if close[i] < close[i - 1] {
            result[i] = result[i - 1] - volume[i];
        } else {
            result[i] = result[i - 1];
        }
    }

    result
}

/// Calculate Money Flow Index (MFI)
fn calculate_mfi(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    period: usize,
) -> Vec<f64> {
    if close.len() < 2 {
        return vec![f64::NAN; close.len()];
    }

    let mut typical_price = vec![0.0; close.len()];
    let mut raw_money_flow = vec![0.0; close.len()];
    let mut positive_flow = vec![0.0; close.len()];
    let mut negative_flow = vec![0.0; close.len()];

    for i in 0..close.len() {
        typical_price[i] = (high[i] + low[i] + close[i]) / 3.0;
        raw_money_flow[i] = typical_price[i] * volume[i];

        if i > 0 {
            if typical_price[i] > typical_price[i - 1] {
                positive_flow[i] = raw_money_flow[i];
            } else if typical_price[i] < typical_price[i - 1] {
                negative_flow[i] = raw_money_flow[i];
            }
        }
    }

    let mut result = vec![f64::NAN; close.len()];

    for i in period..close.len() {
        let pos_sum = positive_flow[(i + 1 - period)..=i].iter().sum::<f64>();
        let neg_sum = negative_flow[(i + 1 - period)..=i].iter().sum::<f64>();

        if neg_sum == 0.0 && pos_sum == 0.0 {
            // No money flow - neutral MFI
            result[i] = 50.0;
        } else if neg_sum == 0.0 {
            // Only positive flow - maximum MFI
            result[i] = 100.0;
        } else {
            // Normal MFI calculation
            let money_ratio = pos_sum / neg_sum;
            result[i] = 100.0 - (100.0 / (1.0 + money_ratio));
        }
    }

    result
}

// Helper functions for DataFrame integration - CRITICAL for compilation

/// Add SMA indicators to DataFrame - PARALLELIZED
fn add_sma_indicators(
    mut df: DataFrame,
    close_prices: &[f64],
    periods: &[u32],
) -> Result<DataFrame> {
    // Compute all SMA periods in parallel
    let sma_results: Vec<_> = periods
        .par_iter()
        .map(|&period| {
            let sma = calculate_sma_ta(close_prices, period as usize).unwrap_or_else(|e| {
                log::error!("SMA calculation failed for period {}: {}", period, e);
                vec![
                    close_prices.iter().sum::<f64>() / close_prices.len() as f64;
                    close_prices.len()
                ] // Default to mean
            });
            (format!("sma_{}", period), sma)
        })
        .collect();

    // Add all computed SMAs to DataFrame
    for (column_name, sma_values) in sma_results {
        let series = Series::new(column_name.clone().into(), sma_values).into_column();
        df = df
            .with_column(series)
            .map_err(|e| {
                VangaError::FeatureError(format!("Failed to add {} column: {}", column_name, e))
            })?
            .clone();
    }
    Ok(df)
}

/// Add EMA indicators to DataFrame - PARALLELIZED
fn add_ema_indicators(
    mut df: DataFrame,
    close_prices: &[f64],
    periods: &[u32],
) -> Result<DataFrame> {
    // Compute all EMA periods in parallel
    let ema_results: Vec<_> = periods
        .par_iter()
        .map(|&period| {
            let ema = calculate_ema_ta(close_prices, period as usize).unwrap_or_else(|e| {
                log::error!("EMA calculation failed for period {}: {}", period, e);
                vec![
                    close_prices.iter().sum::<f64>() / close_prices.len() as f64;
                    close_prices.len()
                ] // Default to mean
            });
            (format!("ema_{}", period), ema)
        })
        .collect();

    // Add all computed EMAs to DataFrame
    for (column_name, ema_values) in ema_results {
        let series = Series::new(column_name.clone().into(), ema_values).into_column();
        df = df
            .with_column(series)
            .map_err(|e| {
                VangaError::FeatureError(format!("Failed to add {} column: {}", column_name, e))
            })?
            .clone();
    }
    Ok(df)
}

/// Add DEMA (Double Exponential Moving Average) indicators to DataFrame - PARALLELIZED
fn add_dema_indicators(
    mut df: DataFrame,
    close_prices: &[f64],
    periods: &[u32],
) -> Result<DataFrame> {
    let dema_results: Vec<_> = periods
        .par_iter()
        .map(|&period| {
            let dema = calculate_dema(close_prices, period as usize);
            (format!("dema_{}", period), dema)
        })
        .collect();

    for (column_name, dema_values) in dema_results {
        let series = Series::new(column_name.clone().into(), dema_values).into_column();
        df = df
            .with_column(series)
            .map_err(|e| {
                VangaError::FeatureError(format!("Failed to add {} column: {}", column_name, e))
            })?
            .clone();
    }
    Ok(df)
}

/// Add TEMA (Triple Exponential Moving Average) indicators to DataFrame - PARALLELIZED
fn add_tema_indicators(
    mut df: DataFrame,
    close_prices: &[f64],
    periods: &[u32],
) -> Result<DataFrame> {
    let tema_results: Vec<_> = periods
        .par_iter()
        .map(|&period| {
            let tema = calculate_tema(close_prices, period as usize);
            (format!("tema_{}", period), tema)
        })
        .collect();

    for (column_name, tema_values) in tema_results {
        let series = Series::new(column_name.clone().into(), tema_values).into_column();
        df = df
            .with_column(series)
            .map_err(|e| {
                VangaError::FeatureError(format!("Failed to add {} column: {}", column_name, e))
            })?
            .clone();
    }
    Ok(df)
}

/// Add KAMA (Kaufman's Adaptive Moving Average) indicators to DataFrame - PARALLELIZED
fn add_kama_indicators(
    mut df: DataFrame,
    close_prices: &[f64],
    periods: &[u32],
) -> Result<DataFrame> {
    let kama_results: Vec<_> = periods
        .par_iter()
        .map(|&period| {
            let kama = calculate_kama(close_prices, period as usize, 2, 30);
            (format!("kama_{}", period), kama)
        })
        .collect();

    for (column_name, kama_values) in kama_results {
        let series = Series::new(column_name.clone().into(), kama_values).into_column();
        df = df
            .with_column(series)
            .map_err(|e| {
                VangaError::FeatureError(format!("Failed to add {} column: {}", column_name, e))
            })?
            .clone();
    }
    Ok(df)
}

/// Add ZLEMA (Zero-Lag Exponential Moving Average) indicators to DataFrame - PARALLELIZED
fn add_zlema_indicators(
    mut df: DataFrame,
    close_prices: &[f64],
    periods: &[u32],
) -> Result<DataFrame> {
    let zlema_results: Vec<_> = periods
        .par_iter()
        .map(|&period| {
            let zlema = calculate_zlema(close_prices, period as usize);
            (format!("zlema_{}", period), zlema)
        })
        .collect();

    for (column_name, zlema_values) in zlema_results {
        let series = Series::new(column_name.clone().into(), zlema_values).into_column();
        df = df
            .with_column(series)
            .map_err(|e| {
                VangaError::FeatureError(format!("Failed to add {} column: {}", column_name, e))
            })?
            .clone();
    }
    Ok(df)
}

/// Calculate DEMA (Double Exponential Moving Average)
/// Formula: DEMA = 2 * EMA(n) - EMA(EMA(n))
fn calculate_dema(prices: &[f64], period: usize) -> Vec<f64> {
    let mut result = vec![f64::NAN; prices.len()];

    if prices.len() < period || period < 2 {
        return result;
    }

    // Calculate EMA
    let ema1 = match calculate_ema_ta(prices, period) {
        Ok(ema) => ema,
        Err(_) => return result,
    };

    // Calculate EMA of EMA
    let ema2 = match calculate_ema_ta(&ema1, period) {
        Ok(ema) => ema,
        Err(_) => return result,
    };

    // DEMA = 2 * EMA - EMA(EMA)
    for i in (2 * period - 1)..prices.len() {
        if ema1[i].is_finite() && ema2[i].is_finite() {
            result[i] = 2.0 * ema1[i] - ema2[i];
        }
    }

    result
}

/// Calculate TEMA (Triple Exponential Moving Average)
/// Formula: TEMA = 3 * EMA - 3 * EMA(EMA) + EMA(EMA(EMA))
fn calculate_tema(prices: &[f64], period: usize) -> Vec<f64> {
    let mut result = vec![f64::NAN; prices.len()];

    if prices.len() < period || period < 2 {
        return result;
    }

    // Calculate EMA
    let ema1 = match calculate_ema_ta(prices, period) {
        Ok(ema) => ema,
        Err(_) => return result,
    };

    // Calculate EMA of EMA
    let ema2 = match calculate_ema_ta(&ema1, period) {
        Ok(ema) => ema,
        Err(_) => return result,
    };

    // Calculate EMA of EMA of EMA
    let ema3 = match calculate_ema_ta(&ema2, period) {
        Ok(ema) => ema,
        Err(_) => return result,
    };

    // TEMA = 3 * EMA - 3 * EMA(EMA) + EMA(EMA(EMA))
    for i in (3 * period - 2)..prices.len() {
        if ema1[i].is_finite() && ema2[i].is_finite() && ema3[i].is_finite() {
            result[i] = 3.0 * ema1[i] - 3.0 * ema2[i] + ema3[i];
        }
    }

    result
}

/// Calculate KAMA (Kaufman's Adaptive Moving Average)
/// Adapts to market volatility using Efficiency Ratio
fn calculate_kama(
    prices: &[f64],
    period: usize,
    fast_period: usize,
    slow_period: usize,
) -> Vec<f64> {
    let mut result = vec![f64::NAN; prices.len()];

    if prices.len() < period || period < 2 {
        return result;
    }

    let fast_sc = 2.0 / (fast_period as f64 + 1.0); // Fast smoothing constant
    let slow_sc = 2.0 / (slow_period as f64 + 1.0); // Slow smoothing constant

    // Initialize KAMA with first valid price
    let mut kama = prices[period - 1];
    result[period - 1] = kama;

    for i in period..prices.len() {
        // Calculate Efficiency Ratio (ER)
        let change = (prices[i] - prices[i - period]).abs();
        let volatility: f64 = (0..period)
            .map(|j| (prices[i - j] - prices[i - j - 1]).abs())
            .sum();

        let er = if volatility > 1e-10 {
            change / volatility
        } else {
            0.0
        };

        // Calculate Smoothing Constant (SC)
        let sc = (er * (fast_sc - slow_sc) + slow_sc).powi(2);

        // Calculate KAMA
        kama = kama + sc * (prices[i] - kama);
        result[i] = kama;
    }

    result
}

/// Calculate ZLEMA (Zero-Lag Exponential Moving Average)
/// Formula: ZLEMA = EMA(Data + (Data - Data[Lag]))
/// where Lag = (Period - 1) / 2
fn calculate_zlema(prices: &[f64], period: usize) -> Vec<f64> {
    let mut result = vec![f64::NAN; prices.len()];

    if prices.len() < period || period < 2 {
        return result;
    }

    let lag = (period - 1) / 2;

    // Create lag-adjusted data
    let mut ema_data = vec![f64::NAN; prices.len()];
    for i in lag..prices.len() {
        ema_data[i] = prices[i] + (prices[i] - prices[i - lag]);
    }

    // Calculate EMA on lag-adjusted data
    let zlema = match calculate_ema_ta(&ema_data, period) {
        Ok(ema) => ema,
        Err(_) => return result,
    };

    // Copy valid values
    for i in (period + lag - 1)..prices.len() {
        if zlema[i].is_finite() {
            result[i] = zlema[i];
        }
    }

    result
}

/// Add MACD indicators to DataFrame
fn add_macd_indicators(
    mut df: DataFrame,
    close_prices: &[f64],
    fast: u32,
    slow: u32,
    signal: u32,
) -> Result<DataFrame> {
    let (macd_line, signal_line, histogram) =
        calculate_macd_ta(close_prices, fast as usize, slow as usize, signal as usize)
            .unwrap_or_else(|e| {
                log::error!("MACD calculation failed: {}", e);
                let len = close_prices.len();
                (vec![0.0; len], vec![0.0; len], vec![0.0; len]) // Default to zeros
            });

    df = df
        .with_column(Series::new("macd".into(), macd_line).into_column())
        .map_err(|e| VangaError::FeatureError(format!("Failed to add MACD column: {}", e)))?
        .clone();

    df = df
        .with_column(Series::new("macd_signal".into(), signal_line).into_column())
        .map_err(|e| VangaError::FeatureError(format!("Failed to add MACD signal column: {}", e)))?
        .clone();

    df = df
        .with_column(Series::new("macd_histogram".into(), histogram).into_column())
        .map_err(|e| {
            VangaError::FeatureError(format!("Failed to add MACD histogram column: {}", e))
        })?
        .clone();

    Ok(df)
}

/// Add Bollinger Bands to DataFrame
fn add_bollinger_bands(
    mut df: DataFrame,
    close_prices: &[f64],
    period: u32,
    std_dev: f64,
) -> Result<DataFrame> {
    let (upper, middle, lower) =
        calculate_bollinger_bands_ta(close_prices, period as usize, std_dev).unwrap_or_else(|e| {
            log::error!("Bollinger Bands calculation failed: {}", e);
            let len = close_prices.len();
            (vec![0.0; len], vec![0.0; len], vec![0.0; len]) // Default to zeros
        });

    df = df
        .with_column(Series::new("bb_upper".into(), upper).into_column())
        .map_err(|e| {
            VangaError::FeatureError(format!("Failed to add Bollinger upper band: {}", e))
        })?
        .clone();

    df = df
        .with_column(Series::new("bb_middle".into(), middle).into_column())
        .map_err(|e| {
            VangaError::FeatureError(format!("Failed to add Bollinger middle band: {}", e))
        })?
        .clone();

    df = df
        .with_column(Series::new("bb_lower".into(), lower).into_column())
        .map_err(|e| {
            VangaError::FeatureError(format!("Failed to add Bollinger lower band: {}", e))
        })?
        .clone();

    Ok(df)
}

/// Add RSI indicators to DataFrame - PARALLELIZED
fn add_rsi_indicators(
    mut df: DataFrame,
    close_prices: &[f64],
    periods: &[u32],
) -> Result<DataFrame> {
    // Debug: Check input data quality
    log::debug!(
        "RSI calculation: input data length = {}",
        close_prices.len()
    );
    let nan_count = close_prices.iter().filter(|&&x| !x.is_finite()).count();
    if nan_count > 0 {
        log::error!(
            "RSI calculation: {} non-finite values in input close_prices",
            nan_count
        );
        return Err(VangaError::FeatureError(format!(
            "RSI calculation failed: {} non-finite values in input close prices",
            nan_count
        )));
    }

    // Compute all RSI periods in parallel
    let rsi_results: Vec<_> = periods
        .par_iter()
        .map(|&period| {
            log::debug!("Calculating RSI for period {}", period);
            let rsi = calculate_rsi_ta(close_prices, period as usize).unwrap_or_else(|e| {
                log::error!("RSI calculation failed for period {}: {}", period, e);
                vec![50.0; close_prices.len()] // Default to neutral RSI
            });
            let nan_count = rsi.iter().filter(|&&x| !x.is_finite()).count();
            log::debug!(
                "RSI period {}: {} NaN values out of {} total",
                period,
                nan_count,
                rsi.len()
            );
            (format!("rsi_{}", period), rsi)
        })
        .collect();

    // Add all computed RSIs to DataFrame
    for (column_name, rsi_values) in rsi_results {
        let series = Series::new(column_name.clone().into(), rsi_values).into_column();
        df = df
            .with_column(series)
            .map_err(|e| {
                VangaError::FeatureError(format!("Failed to add {} column: {}", column_name, e))
            })?
            .clone();
    }
    Ok(df)
}

/// Add Stochastic indicators to DataFrame
#[allow(clippy::too_many_arguments)]
fn add_stochastic_indicators(
    mut df: DataFrame,
    ohlcv: &OhlcvData,
    k_period: u32,
    d_period: u32,
) -> Result<DataFrame> {
    let (k_values, d_values) = calculate_stochastic_ta(
        ohlcv.open,
        ohlcv.high,
        ohlcv.low,
        ohlcv.close,
        ohlcv.volume,
        k_period as usize,
        d_period as usize,
    )
    .unwrap_or_else(|e| {
        log::error!("Stochastic calculation failed: {}", e);
        let len = ohlcv.close.len();
        (vec![f64::NAN; len], vec![f64::NAN; len]) // Use NaN for filtering
    });

    df = df
        .with_column(Series::new("stoch_k".into(), k_values).into_column())
        .map_err(|e| VangaError::FeatureError(format!("Failed to add Stochastic %K: {}", e)))?
        .clone();

    df = df
        .with_column(Series::new("stoch_d".into(), d_values).into_column())
        .map_err(|e| VangaError::FeatureError(format!("Failed to add Stochastic %D: {}", e)))?
        .clone();

    Ok(df)
}

/// Add Williams %R indicator to DataFrame
fn add_williams_r(
    mut df: DataFrame,
    high: &[f64],
    low: &[f64],
    close: &[f64],
    period: u32,
) -> Result<DataFrame> {
    let williams_r_values = calculate_williams_r_ta(high, low, close, period as usize)
        .unwrap_or_else(|e| {
            log::error!("Williams %R calculation failed: {}", e);
            vec![-50.0; close.len()] // Default to neutral value
        });
    df = df
        .with_column(Series::new("williams_r".into(), williams_r_values).into_column())
        .map_err(|e| VangaError::FeatureError(format!("Failed to add Williams %R: {}", e)))?
        .clone();
    Ok(df)
}

/// Add CCI indicator to DataFrame
fn add_cci_indicator(
    mut df: DataFrame,
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    period: u32,
) -> Result<DataFrame> {
    let cci_values = calculate_cci_ta(open, high, low, close, volume, period as usize)
        .unwrap_or_else(|e| {
            log::error!("CCI calculation failed: {}", e);
            vec![f64::NAN; close.len()] // Use NaN for filtering
        });
    let column_name = format!("cci_{}", period);
    let series = Series::new(column_name.clone().into(), cci_values).into_column();
    df = df
        .with_column(series)
        .map_err(|e| VangaError::FeatureError(format!("Failed to add {}: {}", column_name, e)))?
        .clone();
    Ok(df)
}

/// Add Volume indicators to DataFrame with optional momentum features
pub(crate) fn add_volume_indicators(
    mut df: DataFrame,
    close: &[f64],
    volume: &[f64],
    periods: &[u32],
    momentum_config: &crate::config::features::MomentumConfig,
) -> Result<DataFrame> {
    for &period in periods {
        let volume_sma = calculate_sma(volume, period as usize);
        let column_name = format!("volume_sma_{}", period);
        let series = Series::new(column_name.clone().into(), volume_sma.clone()).into_column();
        df = df
            .with_column(series)
            .map_err(|e| VangaError::FeatureError(format!("Failed to add {}: {}", column_name, e)))?
            .clone();

        // Add price-volume correlation indicator
        let pv_correlation = calculate_price_volume_correlation(close, volume, period as usize);
        let corr_column_name = format!("pv_correlation_{}", period);
        let series = Series::new(corr_column_name.clone().into(), pv_correlation).into_column();
        df = df
            .with_column(series)
            .map_err(|e| {
                VangaError::FeatureError(format!("Failed to add {}: {}", corr_column_name, e))
            })?
            .clone();

        // Add volume-weighted price indicator
        let vwap = calculate_volume_weighted_average_price(close, volume, period as usize);
        let vwap_column_name = format!("vwap_{}", period);
        let series = Series::new(vwap_column_name.clone().into(), vwap).into_column();
        df = df
            .with_column(series)
            .map_err(|e| {
                VangaError::FeatureError(format!("Failed to add {}: {}", vwap_column_name, e))
            })?
            .clone();

        // Add volume momentum features if enabled
        if momentum_config.volume_momentum_enabled {
            for &mom_period in &momentum_config.momentum_periods {
                // Volume momentum (relative to SMA)
                let mut volume_momentum = vec![f64::NAN; volume.len()];
                for i in (period as usize)..volume.len() {
                    if volume_sma[i].is_finite() && volume_sma[i].abs() > 1e-10 {
                        volume_momentum[i] = (volume[i] - volume_sma[i]) / volume_sma[i];
                    }
                }
                let mom_name = format!("volume_{}_momentum_{}", period, mom_period);
                df = df
                    .with_column(
                        Series::new(mom_name.clone().into(), volume_momentum.clone()).into_column(),
                    )
                    .map_err(|e| {
                        VangaError::FeatureError(format!("Failed to add {}: {}", mom_name, e))
                    })?
                    .clone();

                // Volume acceleration
                let volume_acceleration = calculate_acceleration(&volume_momentum);
                let accel_name = format!("volume_{}_acceleration_{}", period, mom_period);
                df = df
                    .with_column(
                        Series::new(accel_name.clone().into(), volume_acceleration).into_column(),
                    )
                    .map_err(|e| {
                        VangaError::FeatureError(format!("Failed to add {}: {}", accel_name, e))
                    })?
                    .clone();
            }

            // Volume trend slope
            let volume_trend = calculate_trend_slope(volume, period as usize);
            let trend_name = format!("volume_{}_trend", period);
            df = df
                .with_column(Series::new(trend_name.clone().into(), volume_trend).into_column())
                .map_err(|e| {
                    VangaError::FeatureError(format!("Failed to add {}: {}", trend_name, e))
                })?
                .clone();
        }

        // Add price-volume divergence features if enabled
        if momentum_config.pv_divergence_enabled {
            // Calculate price momentum
            let price_momentum = calculate_momentum(close, period as usize);

            // Calculate volume momentum for divergence
            let volume_momentum_raw = calculate_momentum(volume, period as usize);

            // PV divergence = volume_momentum - price_momentum
            let mut pv_divergence = vec![f64::NAN; close.len()];
            for i in 0..close.len() {
                if price_momentum[i].is_finite() && volume_momentum_raw[i].is_finite() {
                    pv_divergence[i] = volume_momentum_raw[i] - price_momentum[i];
                }
            }
            let div_name = format!("pv_divergence_{}", period);
            df = df
                .with_column(
                    Series::new(div_name.clone().into(), pv_divergence.clone()).into_column(),
                )
                .map_err(|e| {
                    VangaError::FeatureError(format!("Failed to add {}: {}", div_name, e))
                })?
                .clone();

            // PV divergence trend
            let pv_div_trend = calculate_trend_slope(&pv_divergence, period as usize);
            let div_trend_name = format!("pv_divergence_{}_trend", period);
            df = df
                .with_column(Series::new(div_trend_name.clone().into(), pv_div_trend).into_column())
                .map_err(|e| {
                    VangaError::FeatureError(format!("Failed to add {}: {}", div_trend_name, e))
                })?
                .clone();
        }
    }

    // Add AD line momentum if PV divergence is enabled
    if momentum_config.pv_divergence_enabled {
        let obv_values = calculate_obv(close, volume);
        for &mom_period in &momentum_config.momentum_periods {
            let ad_momentum = calculate_momentum(&obv_values, mom_period as usize);
            let ad_mom_name = format!("ad_momentum_{}", mom_period);
            df = df
                .with_column(Series::new(ad_mom_name.clone().into(), ad_momentum).into_column())
                .map_err(|e| {
                    VangaError::FeatureError(format!("Failed to add {}: {}", ad_mom_name, e))
                })?
                .clone();
        }
    }

    Ok(df)
}

/// Add OBV indicator to DataFrame
fn add_obv_indicator(mut df: DataFrame, close: &[f64], volume: &[f64]) -> Result<DataFrame> {
    let obv_values = calculate_obv(close, volume);
    df = df
        .with_column(Series::new("obv".into(), obv_values).into_column())
        .map_err(|e| VangaError::FeatureError(format!("Failed to add OBV: {}", e)))?
        .clone();
    Ok(df)
}

/// Add MFI indicator to DataFrame
fn add_mfi_indicator(
    mut df: DataFrame,
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    period: u32,
) -> Result<DataFrame> {
    let mfi_values = calculate_mfi(high, low, close, volume, period as usize);
    let column_name = format!("mfi_{}", period);
    let series = Series::new(column_name.clone().into(), mfi_values).into_column();
    df = df
        .with_column(series)
        .map_err(|e| VangaError::FeatureError(format!("Failed to add {}: {}", column_name, e)))?
        .clone();
    Ok(df)
}

/// Add ATR indicators to DataFrame with optional momentum features
#[allow(clippy::too_many_arguments)]
pub(crate) fn add_atr_indicators(
    mut df: DataFrame,
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    periods: &[u32],
    momentum_config: &crate::config::features::MomentumConfig,
) -> Result<DataFrame> {
    for &period in periods {
        let atr_values = calculate_atr_ta(open, high, low, close, volume, period as usize)
            .unwrap_or_else(|e| {
                log::error!("ATR calculation failed for period {}: {}", period, e);
                vec![f64::NAN; close.len()] // Use NaN for filtering
            });
        let column_name = format!("atr_{}", period);
        let series = Series::new(column_name.clone().into(), atr_values.clone()).into_column();
        df = df
            .with_column(series)
            .map_err(|e| VangaError::FeatureError(format!("Failed to add {}: {}", column_name, e)))?
            .clone();

        // Add ATR momentum features if enabled
        if momentum_config.atr_momentum_enabled {
            for &mom_period in &momentum_config.momentum_periods {
                // ATR momentum
                let atr_momentum = calculate_momentum(&atr_values, mom_period as usize);
                let mom_name = format!("atr_{}_momentum_{}", period, mom_period);
                df = df
                    .with_column(
                        Series::new(mom_name.clone().into(), atr_momentum.clone()).into_column(),
                    )
                    .map_err(|e| {
                        VangaError::FeatureError(format!("Failed to add {}: {}", mom_name, e))
                    })?
                    .clone();

                // ATR acceleration
                let atr_acceleration = calculate_acceleration(&atr_momentum);
                let accel_name = format!("atr_{}_acceleration_{}", period, mom_period);
                df = df
                    .with_column(
                        Series::new(accel_name.clone().into(), atr_acceleration).into_column(),
                    )
                    .map_err(|e| {
                        VangaError::FeatureError(format!("Failed to add {}: {}", accel_name, e))
                    })?
                    .clone();
            }

            // ATR trend slope
            let atr_trend = calculate_trend_slope(&atr_values, period as usize);
            let trend_name = format!("atr_{}_trend", period);
            df = df
                .with_column(Series::new(trend_name.clone().into(), atr_trend).into_column())
                .map_err(|e| {
                    VangaError::FeatureError(format!("Failed to add {}: {}", trend_name, e))
                })?
                .clone();
        }
    }
    Ok(df)
}

/// Add Keltner Channels to DataFrame
#[allow(clippy::too_many_arguments)]
fn add_keltner_channels(
    mut df: DataFrame,
    ohlcv: &OhlcvData,
    period: u32,
    multiplier: f64,
) -> Result<DataFrame> {
    let ema_values = calculate_ema_ta(ohlcv.close, period as usize).unwrap_or_else(|e| {
        log::error!("EMA calculation failed for Keltner Channels: {}", e);
        vec![ohlcv.close.iter().sum::<f64>() / ohlcv.close.len() as f64; ohlcv.close.len()]
    });
    let atr_values = calculate_atr_ta(
        ohlcv.open,
        ohlcv.high,
        ohlcv.low,
        ohlcv.close,
        ohlcv.volume,
        period as usize,
    )
    .unwrap_or_else(|e| {
        log::error!("ATR calculation failed for Keltner Channels: {}", e);
        vec![f64::NAN; ohlcv.close.len()]
    });

    let mut keltner_upper = vec![f64::NAN; ohlcv.close.len()];
    let mut keltner_lower = vec![f64::NAN; ohlcv.close.len()];

    for i in 0..ohlcv.close.len() {
        if !ema_values[i].is_nan() && !atr_values[i].is_nan() {
            keltner_upper[i] = ema_values[i] + multiplier * atr_values[i];
            keltner_lower[i] = ema_values[i] - multiplier * atr_values[i];
        }
    }

    df = df
        .with_column(Series::new("keltner_upper".into(), keltner_upper).into_column())
        .map_err(|e| VangaError::FeatureError(format!("Failed to add Keltner upper: {}", e)))?
        .clone();

    df = df
        .with_column(Series::new("keltner_lower".into(), keltner_lower).into_column())
        .map_err(|e| VangaError::FeatureError(format!("Failed to add Keltner lower: {}", e)))?
        .clone();

    Ok(df)
}

/// Parameters for cryptocurrency-specific indicators
struct CryptoIndicatorParams<'a> {
    open: &'a [f64],
    high: &'a [f64],
    low: &'a [f64],
    close: &'a [f64],
    volume: &'a [f64],
    advanced_config: &'a crate::config::features::AdvancedIndicatorsConfig,
    timeframe_multiplier: f64,
}

/// Add cryptocurrency-specific indicators
fn add_crypto_specific_indicators(
    mut df: DataFrame,
    params: CryptoIndicatorParams,
) -> Result<DataFrame> {
    let CryptoIndicatorParams {
        open,
        high,
        low,
        close,
        volume,
        advanced_config,
        timeframe_multiplier,
    } = params;
    // Price velocity (rate of change)
    let mut price_velocity = vec![f64::NAN; close.len()];
    for i in 1..close.len() {
        price_velocity[i] = (close[i] - close[i - 1]) / close[i - 1] * 100.0;
    }

    // VWAP (Volume Weighted Average Price)
    let mut vwap = vec![f64::NAN; close.len()];
    let mut cum_volume = 0.0;
    let mut cum_price_volume = 0.0;

    for i in 0..close.len() {
        let typical_price = (high[i] + low[i] + close[i]) / 3.0;
        cum_price_volume += typical_price * volume[i];
        cum_volume += volume[i];

        if cum_volume > 0.0 {
            vwap[i] = cum_price_volume / cum_volume;
        }
    }

    // Price inefficiency: Body-to-Range ratio (measures directional strength vs rejection)
    // For 24/7 crypto markets, inter-candle gaps don't exist (open[i] == close[i-1])
    // Instead, measure intrabar inefficiency: how much of the range was directional movement
    // Range: -100 to +100 (negative = bearish body, positive = bullish body)
    let mut price_gaps = vec![0.0; close.len()];
    for i in 0..close.len() {
        let range = high[i] - low[i];
        if range > 0.0 {
            let body = close[i] - open[i];
            // Body-to-range ratio as percentage: (body / range) * 100
            // +100 = full bullish candle (no wicks), -100 = full bearish candle
            // 0 = doji or all wicks (high rejection)
            price_gaps[i] = (body / range * 100.0).clamp(-100.0, 100.0);
        }
    }

    // Wick imbalance: measures buying vs selling pressure rejection
    // Positive = upper wick dominance (selling pressure), Negative = lower wick dominance (buying pressure)
    let mut gap_volatility = vec![0.0; close.len()];
    for i in 0..close.len() {
        let body_top = open[i].max(close[i]);
        let body_bottom = open[i].min(close[i]);
        let upper_wick = high[i] - body_top;
        let lower_wick = body_bottom - low[i];
        let total_wick = upper_wick + lower_wick;

        if total_wick > 0.0 {
            // Wick imbalance: (upper - lower) / total * 100
            // +100 = all upper wick (strong selling rejection)
            // -100 = all lower wick (strong buying rejection)
            gap_volatility[i] =
                ((upper_wick - lower_wick) / total_wick * 100.0).clamp(-100.0, 100.0);
        }
    }

    // Intraday range (high - low) as percentage of close
    let mut intraday_range = vec![0.0; close.len()];
    for i in 0..close.len() {
        intraday_range[i] = (high[i] - low[i]) / close[i] * 100.0;
    }

    // Advanced mathematical indicators (completing AUTO_INDICATORS)
    if advanced_config.enabled {
        let hurst_values = calculate_hurst_exponent(close, advanced_config.hurst_window);
        let fractal_dims =
            calculate_fractal_dimension_higuchi(close, advanced_config.fractal_window);
        let regime_values = calculate_regime_indicator(
            close,
            volume,
            advanced_config.regime_window,
            timeframe_multiplier,
        );
        let clustering_values =
            calculate_volatility_clustering(close, advanced_config.clustering_window);
        let reversion_values =
            calculate_mean_reversion_strength(close, advanced_config.reversion_window);

        // Add advanced mathematical indicators to DataFrame
        df = df
            .with_column(Series::new("hurst_exponent".into(), hurst_values).into_column())
            .map_err(|e| VangaError::FeatureError(format!("Failed to add hurst_exponent: {}", e)))?
            .clone();

        df = df
            .with_column(Series::new("fractal_dimension".into(), fractal_dims).into_column())
            .map_err(|e| {
                VangaError::FeatureError(format!("Failed to add fractal_dimension: {}", e))
            })?
            .clone();

        df = df
            .with_column(Series::new("regime_indicator".into(), regime_values).into_column())
            .map_err(|e| {
                VangaError::FeatureError(format!("Failed to add regime_indicator: {}", e))
            })?
            .clone();

        df = df
            .with_column(
                Series::new("volatility_clustering".into(), clustering_values).into_column(),
            )
            .map_err(|e| {
                VangaError::FeatureError(format!("Failed to add volatility_clustering: {}", e))
            })?
            .clone();

        df = df
            .with_column(
                Series::new("mean_reversion_strength".into(), reversion_values).into_column(),
            )
            .map_err(|e| {
                VangaError::FeatureError(format!("Failed to add mean_reversion_strength: {}", e))
            })?
            .clone();

        log::debug!("Added {} advanced mathematical indicators", 5);
    }

    // Add indicators
    df = df
        .with_column(Series::new("price_velocity".into(), price_velocity).into_column())
        .map_err(|e| VangaError::FeatureError(format!("Failed to add price velocity: {}", e)))?
        .clone();

    df = df
        .with_column(Series::new("vwap".into(), vwap).into_column())
        .map_err(|e| VangaError::FeatureError(format!("Failed to add VWAP: {}", e)))?
        .clone();

    df = df
        .with_column(Series::new("price_gaps".into(), price_gaps).into_column())
        .map_err(|e| VangaError::FeatureError(format!("Failed to add price gaps: {}", e)))?
        .clone();

    df = df
        .with_column(Series::new("gap_volatility".into(), gap_volatility).into_column())
        .map_err(|e| VangaError::FeatureError(format!("Failed to add gap volatility: {}", e)))?
        .clone();

    df = df
        .with_column(Series::new("intraday_range".into(), intraday_range).into_column())
        .map_err(|e| VangaError::FeatureError(format!("Failed to add intraday range: {}", e)))?
        .clone();

    Ok(df)
}

/// Calculate price-volume correlation over a rolling window
fn calculate_price_volume_correlation(close: &[f64], volume: &[f64], period: usize) -> Vec<f64> {
    let mut correlation = vec![f64::NAN; close.len()];

    if close.len() != volume.len() || period < 2 {
        return correlation;
    }

    for i in period..close.len() {
        let price_window = &close[i - period..i];
        let volume_window = &volume[i - period..i];

        // Calculate Pearson correlation coefficient
        let price_mean = price_window.iter().sum::<f64>() / period as f64;
        let volume_mean = volume_window.iter().sum::<f64>() / period as f64;

        let mut numerator = 0.0;
        let mut price_variance = 0.0;
        let mut volume_variance = 0.0;

        for j in 0..period {
            let price_diff = price_window[j] - price_mean;
            let volume_diff = volume_window[j] - volume_mean;

            numerator += price_diff * volume_diff;
            price_variance += price_diff * price_diff;
            volume_variance += volume_diff * volume_diff;
        }

        let denominator = (price_variance * volume_variance).sqrt();
        correlation[i] = if denominator > 0.0 {
            numerator / denominator
        } else {
            0.0
        };
    }

    correlation
}

/// Calculate Volume Weighted Average Price (VWAP) over a rolling window
fn calculate_volume_weighted_average_price(
    close: &[f64],
    volume: &[f64],
    period: usize,
) -> Vec<f64> {
    let mut vwap = vec![f64::NAN; close.len()];

    if close.len() != volume.len() || period == 0 {
        return vwap;
    }

    for i in period..close.len() {
        let mut total_volume = 0.0;
        let mut weighted_price_sum = 0.0;

        for j in (i - period)..i {
            if volume[j] > 0.0 {
                weighted_price_sum += close[j] * volume[j];
                total_volume += volume[j];
            }
        }

        vwap[i] = if total_volume > 0.0 {
            weighted_price_sum / total_volume
        } else {
            close[i] // Fallback to current price if no volume
        };
    }

    vwap
}

/// Calculate Hurst Exponent using R/S analysis for regime detection
/// H > 0.65: Trending regime, H < 0.45: Mean-reverting regime
fn calculate_hurst_exponent(prices: &[f64], window: usize) -> Vec<f64> {
    let mut hurst_values = vec![f64::NAN; prices.len()];

    if prices.len() < window || window < 20 {
        return hurst_values;
    }

    for i in window..prices.len() {
        let price_window = &prices[i - window..i];
        let returns: Vec<f64> = price_window
            .windows(2)
            .map(|w| (w[1] / w[0]).ln())
            .collect();

        if returns.len() < 10 {
            continue;
        }

        // R/S Analysis with multiple lags
        let max_lag = (returns.len() / 4).min(20);
        let lags: Vec<usize> = (2..=max_lag).collect();
        let mut rs_values = Vec::new();

        for &lag in &lags {
            if lag >= returns.len() {
                continue;
            }

            let chunks = returns.len() / lag;
            if chunks == 0 {
                continue;
            }

            let mut rs_sum = 0.0;
            let mut valid_chunks = 0;

            for chunk in 0..chunks {
                let start = chunk * lag;
                let end = start + lag;
                let subset = &returns[start..end];

                // Calculate mean and cumulative deviations
                let mean = subset.iter().sum::<f64>() / lag as f64;
                let mut cumulative_devs = vec![0.0; lag];
                cumulative_devs[0] = subset[0] - mean;

                for j in 1..lag {
                    cumulative_devs[j] = cumulative_devs[j - 1] + (subset[j] - mean);
                }

                // Calculate range
                let max_dev = cumulative_devs
                    .iter()
                    .fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                let min_dev = cumulative_devs.iter().fold(f64::INFINITY, |a, &b| a.min(b));
                let range = max_dev - min_dev;

                // Calculate standard deviation
                let variance = subset.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / lag as f64;
                let std_dev = variance.sqrt();

                if std_dev > 1e-10 && range > 0.0 {
                    rs_sum += range / std_dev;
                    valid_chunks += 1;
                }
            }

            if valid_chunks > 0 {
                rs_values.push(rs_sum / valid_chunks as f64);
            }
        }

        // Linear regression to find Hurst exponent
        if rs_values.len() >= 3 {
            let log_lags: Vec<f64> = lags[..rs_values.len()]
                .iter()
                .map(|&x| (x as f64).ln())
                .collect();
            let log_rs: Vec<f64> = rs_values.iter().map(|&x| x.ln()).collect();

            let n = log_lags.len() as f64;
            let sum_x = log_lags.iter().sum::<f64>();
            let sum_y = log_rs.iter().sum::<f64>();
            let sum_xy = log_lags
                .iter()
                .zip(log_rs.iter())
                .map(|(x, y)| x * y)
                .sum::<f64>();
            let sum_x2 = log_lags.iter().map(|x| x * x).sum::<f64>();

            let denominator = n * sum_x2 - sum_x * sum_x;
            if denominator.abs() > 1e-10 {
                let slope = (n * sum_xy - sum_x * sum_y) / denominator;
                // Clamp to reasonable range for crypto markets
                hurst_values[i] = slope.clamp(0.1, 0.9);
            }
        }
    }

    hurst_values
}

/// Calculate Fractal Dimension using Higuchi method
/// Better suited for time series than box-counting
/// Returns values typically in range [1.0, 2.0]
/// Uses 1.5 as default for windows where calculation fails (low variation, constant prices, etc.)
fn calculate_fractal_dimension_higuchi(prices: &[f64], max_window: usize) -> Vec<f64> {
    let mut fractal_dims = vec![f64::NAN; prices.len()];

    if prices.len() < max_window || max_window < 10 {
        return fractal_dims;
    }

    let k_max = 8.min(max_window / 4);

    for idx in max_window..prices.len() {
        let window = &prices[idx - max_window..idx];
        let n = window.len();

        // Check for constant prices (all same value)
        let min_price = window.iter().copied().fold(f64::INFINITY, f64::min);
        let max_price = window.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let price_range = max_price - min_price;

        // If prices are constant or near-constant, skip calculation
        if price_range < 1e-10 {
            continue;
        }

        let mut l_k_values = Vec::with_capacity(k_max);

        for k in 1..=k_max {
            let mut l_m_values = Vec::with_capacity(k);

            for m in 0..k {
                let num_segments = (n - m - 1) / k;

                if num_segments == 0 {
                    continue;
                }

                let mut curve_length = 0.0;
                for i in 1..=num_segments {
                    let idx1 = m + i * k;
                    let idx2 = m + (i - 1) * k;
                    curve_length += (window[idx1] - window[idx2]).abs();
                }

                // Skip if curve length is zero (constant segment)
                if curve_length < 1e-10 {
                    continue;
                }

                let normalization = (n - 1) as f64 / (num_segments as f64 * k as f64 * k as f64);
                let l_m = curve_length * normalization;

                l_m_values.push(l_m);
            }

            if !l_m_values.is_empty() {
                let l_k = l_m_values.iter().sum::<f64>() / l_m_values.len() as f64;
                // Only add if l_k is positive and finite
                if l_k > 1e-10 && l_k.is_finite() {
                    l_k_values.push(l_k);
                }
            }
        }

        if l_k_values.len() >= 3 {
            let mut log_inv_k = Vec::with_capacity(l_k_values.len());
            let mut log_l_k = Vec::with_capacity(l_k_values.len());

            for (k_idx, &l_k) in l_k_values.iter().enumerate() {
                let k = (k_idx + 1) as f64;
                let log_val = l_k.ln();
                if log_val.is_finite() {
                    log_inv_k.push((1.0 / k).ln());
                    log_l_k.push(log_val);
                }
            }

            if log_inv_k.len() >= 3 {
                let n_points = log_inv_k.len() as f64;
                let sum_x = log_inv_k.iter().sum::<f64>();
                let sum_y = log_l_k.iter().sum::<f64>();
                let sum_xy = log_inv_k
                    .iter()
                    .zip(log_l_k.iter())
                    .map(|(x, y)| x * y)
                    .sum::<f64>();
                let sum_x2 = log_inv_k.iter().map(|x| x * x).sum::<f64>();

                let denominator = n_points * sum_x2 - sum_x * sum_x;

                if denominator.abs() > 1e-10 {
                    let slope = (n_points * sum_xy - sum_x * sum_y) / denominator;

                    // Clamp to reasonable range for time series (0.5 to 2.5)
                    if slope.is_finite() && slope > 0.0 {
                        fractal_dims[idx] = slope.clamp(0.5, 2.5);
                    }
                }
            }
        }
    }

    fractal_dims
}

/// Calculate Regime Indicator combining volatility, trend, and volume signals
/// Returns 0-3 scale: 0=stable/ranging, 3=high volatility/trending/high volume
fn calculate_regime_indicator(
    prices: &[f64],
    volume: &[f64],
    window: usize,
    _timeframe_multiplier: f64,
) -> Vec<f64> {
    let mut regime_values = vec![f64::NAN; prices.len()];

    if prices.len() < window || volume.len() != prices.len() || window < 5 {
        log::error!(
            "Insufficient data for regime indicator: prices={}, volume={}, window={}, min_window=5",
            prices.len(),
            volume.len(),
            window
        );
        return regime_values;
    }

    // Calculate historical statistics for adaptive thresholds
    let mut all_volatilities = Vec::new();
    let mut all_trend_strengths = Vec::new();

    for i in window..prices.len() {
        let price_window = &prices[i - window..i];

        let returns: Vec<f64> = price_window
            .windows(2)
            .map(|w| (w[1] / w[0]).ln())
            .collect();

        if !returns.is_empty() {
            let volatility = returns.iter().map(|r| r.powi(2)).sum::<f64>() / returns.len() as f64;
            all_volatilities.push(volatility);

            let price_start = price_window[0];
            let price_end = price_window[price_window.len() - 1];
            let price_change = (price_end - price_start) / price_start;
            all_trend_strengths.push(price_change.abs());
        }
    }

    if all_volatilities.is_empty() || all_trend_strengths.is_empty() {
        log::error!("Failed to calculate regime statistics: no valid returns computed");
        return regime_values;
    }

    // Use 75th percentile as threshold (adaptive to market conditions)
    let mut sorted_vols = all_volatilities.clone();
    sorted_vols.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let vol_threshold = sorted_vols[(sorted_vols.len() as f64 * 0.75) as usize];

    let mut sorted_trends = all_trend_strengths.clone();
    sorted_trends.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let trend_threshold = sorted_trends[(sorted_trends.len() as f64 * 0.75) as usize];

    for i in window..prices.len() {
        let price_window = &prices[i - window..i];
        let volume_window = &volume[i - window..i];

        // Calculate returns for volatility analysis
        let returns: Vec<f64> = price_window
            .windows(2)
            .map(|w| (w[1] / w[0]).ln())
            .collect();

        if returns.is_empty() {
            continue;
        }

        // 1. Volatility regime (high/low volatility)
        let volatility = returns.iter().map(|r| r.powi(2)).sum::<f64>() / returns.len() as f64;
        let vol_regime = if volatility > vol_threshold { 1.0 } else { 0.0 };

        // 2. Trend regime (trending/ranging)
        let price_start = price_window[0];
        let price_end = price_window[price_window.len() - 1];
        let price_change = (price_end - price_start) / price_start;
        let trend_strength = price_change.abs();
        let trend_regime = if trend_strength > trend_threshold {
            1.0
        } else {
            0.0
        };

        // 3. Volume regime (high/low volume) - use median instead of mean for robustness
        let mut sorted_volume = volume_window.to_vec();
        sorted_volume.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median_volume = sorted_volume[sorted_volume.len() / 2];
        let recent_volume = volume_window[volume_window.len() - 1];
        let volume_regime = if recent_volume > median_volume * 1.5 {
            1.0
        } else {
            0.0
        };

        // Combine regimes (0-3 scale)
        regime_values[i] = vol_regime + trend_regime + volume_regime;
    }

    regime_values
}

/// Calculate Volatility Clustering using autocorrelation of squared returns
/// Higher values indicate stronger volatility clustering (GARCH effects)
fn calculate_volatility_clustering(prices: &[f64], window: usize) -> Vec<f64> {
    let mut clustering_values = vec![f64::NAN; prices.len()];

    if prices.len() < window || window < 10 {
        return clustering_values;
    }

    for i in window..prices.len() {
        let price_window = &prices[i - window..i];
        let returns: Vec<f64> = price_window
            .windows(2)
            .map(|w| (w[1] / w[0]).ln())
            .collect();

        if returns.len() < 5 {
            continue;
        }

        // Calculate squared returns (volatility proxy)
        let squared_returns: Vec<f64> = returns.iter().map(|r| r.powi(2)).collect();

        // Calculate autocorrelation of squared returns (lag 1)
        let mean_sq = squared_returns.iter().sum::<f64>() / squared_returns.len() as f64;

        let mut numerator = 0.0;
        let mut denominator = 0.0;
        let mut count = 0;

        for j in 1..squared_returns.len() {
            let current_dev = squared_returns[j] - mean_sq;
            let prev_dev = squared_returns[j - 1] - mean_sq;

            numerator += current_dev * prev_dev;
            denominator += current_dev.powi(2);
            count += 1;
        }

        clustering_values[i] = if denominator > 1e-10 && count > 0 {
            (numerator / denominator).clamp(-1.0, 1.0) // Clamp correlation
        } else {
            0.0
        };
    }

    clustering_values
}

/// Calculate Mean Reversion Strength using deviation analysis
/// Higher values indicate stronger tendency to revert to mean
fn calculate_mean_reversion_strength(prices: &[f64], window: usize) -> Vec<f64> {
    let mut reversion_values = vec![f64::NAN; prices.len()];

    if prices.len() < window || window < 5 {
        return reversion_values;
    }

    for i in window..prices.len() {
        let price_window = &prices[i - window..i];

        // Calculate moving average
        let mean_price = price_window.iter().sum::<f64>() / price_window.len() as f64;

        // Calculate normalized deviations from mean
        let deviations: Vec<f64> = price_window
            .iter()
            .map(|&p| (p - mean_price) / mean_price)
            .collect();

        // Calculate mean reversion coefficient
        let mut reversion_sum = 0.0;
        let mut count = 0;

        for j in 1..deviations.len() {
            let current_dev = deviations[j];
            let prev_dev = deviations[j - 1];

            // Check if price moves back toward mean
            let moving_toward_mean = if prev_dev > 0.0 {
                current_dev < prev_dev // Positive deviation decreasing
            } else if prev_dev < 0.0 {
                current_dev > prev_dev // Negative deviation increasing (toward zero)
            } else {
                false
            };

            if moving_toward_mean {
                reversion_sum += (prev_dev - current_dev).abs();
                count += 1;
            }
        }

        reversion_values[i] = if count > 0 {
            (reversion_sum / count as f64).min(1.0) // Cap at 1.0 for normalization
        } else {
            0.0
        };
    }

    reversion_values
}

/// Add consolidation/neutral-detection features to help model learn Class 2 (Neutral) patterns
///
/// These features specifically detect sideways/consolidation markets where price is NOT trending.
/// Critical for improving Class 2 (Neutral) classification accuracy.
///
/// NEW Features added (non-duplicates):
/// 1. Range Tightness: (high - low) / close - measures price compression
/// 2. Bollinger Band Squeeze: (upper_bb - lower_bb) / close - detects volatility contraction
/// 3. Choppiness Index: High values (>61.8) indicate consolidation
/// 4. Price Efficiency Ratio: Measures how efficiently price moves (low = choppy)
/// 5. Directional Movement Balance: |+DI - -DI| / (+DI + -DI) - low = balanced/sideways
///
/// NOTE: ADX already exists in trend indicators, so we don't duplicate it here
pub fn add_consolidation_features(
    mut df: DataFrame,
    ohlcv: &OhlcvData,
    config: &TechnicalIndicatorsConfig,
) -> Result<DataFrame> {
    log::info!("🎯 Adding consolidation/neutral-detection features for Class 2 learning...");

    let close = ohlcv.close;
    let high = ohlcv.high;
    let low = ohlcv.low;

    // 1. Range Tightness: (high - low) / close
    // Low values indicate tight consolidation
    let range_tightness = calculate_range_tightness(high, low, close);
    df = df
        .with_column(Series::new("range_tightness".into(), range_tightness).into_column())
        .map_err(|e| VangaError::FeatureError(format!("Failed to add range_tightness: {}", e)))?
        .clone();

    // 2. Bollinger Band Squeeze: (upper_bb - lower_bb) / close
    // Low values indicate volatility contraction (consolidation)
    if config.volatility.bollinger_bands.enabled {
        let period = config.volatility.bollinger_bands.period;
        let std_dev = config.volatility.bollinger_bands.std_dev;
        let bb_squeeze = calculate_bollinger_squeeze(close, period, std_dev);
        df = df
            .with_column(
                Series::new(format!("bb_squeeze_{}", period).into(), bb_squeeze).into_column(),
            )
            .map_err(|e| VangaError::FeatureError(format!("Failed to add bb_squeeze: {}", e)))?
            .clone();
    }

    // 3. Choppiness Index
    // High values (>61.8) indicate consolidation, low values (<38.2) indicate trending
    for &period in &config.trend.consolidation.choppiness_periods {
        let choppiness = calculate_choppiness_index(high, low, close, period as usize);
        df = df
            .with_column(
                Series::new(format!("choppiness_{}", period).into(), choppiness).into_column(),
            )
            .map_err(|e| {
                VangaError::FeatureError(format!("Failed to add choppiness_{}: {}", period, e))
            })?
            .clone();
    }

    // 4. Price Efficiency Ratio
    // Low values indicate choppy/sideways movement
    for &period in &config.trend.consolidation.efficiency_periods {
        let efficiency = calculate_price_efficiency_ratio(close, period as usize);
        df = df
            .with_column(
                Series::new(format!("price_efficiency_{}", period).into(), efficiency)
                    .into_column(),
            )
            .map_err(|e| {
                VangaError::FeatureError(format!(
                    "Failed to add price_efficiency_{}: {}",
                    period, e
                ))
            })?
            .clone();
    }

    // 5. Directional Movement Balance
    // Low values indicate balanced directional movement (sideways)
    for &period in &config.trend.consolidation.dm_balance_periods {
        let dm_balance = calculate_directional_movement_balance(high, low, close, period as usize);
        df = df
            .with_column(
                Series::new(format!("dm_balance_{}", period).into(), dm_balance).into_column(),
            )
            .map_err(|e| {
                VangaError::FeatureError(format!("Failed to add dm_balance_{}: {}", period, e))
            })?
            .clone();
    }

    log::info!("✅ Added consolidation features: range_tightness, bb_squeeze, choppiness, price_efficiency, dm_balance");
    Ok(df)
}

/// Calculate Range Tightness: (high - low) / close
/// Low values indicate tight consolidation (neutral market)
pub fn calculate_range_tightness(high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
    high.iter()
        .zip(low.iter())
        .zip(close.iter())
        .map(
            |((&h, &l), &c)| {
                if c > 0.0 {
                    (h - l) / c
                } else {
                    f64::NAN
                }
            },
        )
        .collect()
}

/// Calculate Bollinger Band Squeeze: (upper_bb - lower_bb) / close
/// Low values indicate volatility contraction (consolidation phase)
pub fn calculate_bollinger_squeeze(close: &[f64], period: u32, std_dev: f64) -> Vec<f64> {
    let period_usize = period as usize;
    let mut result = vec![f64::NAN; close.len()];

    if close.len() < period_usize {
        return result;
    }

    for i in (period_usize - 1)..close.len() {
        let window = &close[(i + 1 - period_usize)..=i];
        let mean = window.iter().sum::<f64>() / period_usize as f64;

        let variance =
            window.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / period_usize as f64;
        let std = variance.sqrt();

        let upper_bb = mean + (std_dev * std);
        let lower_bb = mean - (std_dev * std);

        if close[i] > 0.0 {
            result[i] = (upper_bb - lower_bb) / close[i];
        }
    }

    result
}

/// Calculate Choppiness Index
/// High values (>61.8) indicate consolidation/sideways
/// Low values (<38.2) indicate trending market
pub fn calculate_choppiness_index(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    period: usize,
) -> Vec<f64> {
    let mut result = vec![f64::NAN; close.len()];

    if close.len() < period + 1 {
        return result;
    }

    for i in period..close.len() {
        let window_high = &high[(i + 1 - period)..=i];
        let window_low = &low[(i + 1 - period)..=i];

        let highest_high = window_high.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let lowest_low = window_low.iter().fold(f64::INFINITY, |a, &b| a.min(b));

        // Calculate sum of True Ranges
        let mut tr_sum = 0.0;
        for j in (i + 1 - period + 1)..=i {
            let h_l = high[j] - low[j];
            let h_c = (high[j] - close[j - 1]).abs();
            let l_c = (low[j] - close[j - 1]).abs();
            tr_sum += h_l.max(h_c).max(l_c);
        }

        let range = highest_high - lowest_low;
        if range > 0.0 && tr_sum > 0.0 {
            result[i] = 100.0 * (tr_sum / range).ln() / (period as f64).ln();
        }
    }

    result
}

/// Calculate Price Efficiency Ratio
/// Measures how efficiently price moves from start to end
/// Low values indicate choppy/sideways movement
pub fn calculate_price_efficiency_ratio(close: &[f64], period: usize) -> Vec<f64> {
    let mut result = vec![f64::NAN; close.len()];

    if close.len() < period {
        return result;
    }

    for i in (period - 1)..close.len() {
        let start_price = close[i + 1 - period];
        let end_price = close[i];

        // Net price change (direction)
        let net_change = (end_price - start_price).abs();

        // Sum of absolute price changes (volatility)
        let mut total_change = 0.0;
        for j in (i + 1 - period + 1)..=i {
            total_change += (close[j] - close[j - 1]).abs();
        }

        if total_change > 0.0 {
            result[i] = net_change / total_change;
        } else {
            result[i] = 0.0; // No movement = no efficiency
        }
    }

    result
}

/// Calculate Directional Movement Balance
/// Measures balance between +DI and -DI
/// Low values indicate balanced directional movement (sideways)
pub fn calculate_directional_movement_balance(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    period: usize,
) -> Vec<f64> {
    let mut result = vec![f64::NAN; close.len()];

    if close.len() < period + 1 {
        return result;
    }

    // Calculate +DM and -DM
    let mut plus_dm = vec![0.0; close.len()];
    let mut minus_dm = vec![0.0; close.len()];
    for i in 1..close.len() {
        let up_move = high[i] - high[i - 1];
        let down_move = low[i - 1] - low[i];

        if up_move > down_move && up_move > 0.0 {
            plus_dm[i] = up_move;
        }
        if down_move > up_move && down_move > 0.0 {
            minus_dm[i] = down_move;
        }
    }

    // Calculate smoothed +DM and -DM
    for i in period..close.len() {
        let plus_sum = plus_dm[(i + 1 - period)..=i].iter().sum::<f64>();
        let minus_sum = minus_dm[(i + 1 - period)..=i].iter().sum::<f64>();

        let total = plus_sum + minus_sum;
        if total > 0.0 {
            // Balance: 0 = perfectly balanced (sideways), 1 = completely one-directional
            result[i] = (plus_sum - minus_sum).abs() / total;
        } else {
            result[i] = 0.0; // No movement = balanced
        }
    }

    result
}
