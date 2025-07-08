// Technical indicators implementation with comprehensive indicators
use crate::config::features::TechnicalIndicatorsConfig;
use crate::utils::error::{Result, VangaError};
use polars::prelude::*;
use rayon::prelude::*;

/// Generate comprehensive technical indicators for cryptocurrency data - PARALLELIZED
pub async fn generate_technical_indicators(
    mut df: DataFrame,
    config: &TechnicalIndicatorsConfig,
) -> Result<DataFrame> {
    log::info!("Generating comprehensive technical indicators with parallel processing...");

    // Extract OHLCV data for calculations - PARALLEL EXTRACTION
    let close_prices = extract_numeric_column(&df, "close")?;
    let high_prices = extract_numeric_column(&df, "high")?;
    let low_prices = extract_numeric_column(&df, "low")?;
    let open_prices = extract_numeric_column(&df, "open")?;
    let volume = extract_numeric_column(&df, "volume")?;

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
        df = add_stochastic_indicators(df, &high_prices, &low_prices, &close_prices, 14, 3)?;
    }

    if config.momentum.williams_r {
        df = add_williams_r(df, &high_prices, &low_prices, &close_prices, 14)?;
    }

    if !config.momentum.cci_periods.is_empty() {
        for &period in &config.momentum.cci_periods {
            df = add_cci_indicator(df, &high_prices, &low_prices, &close_prices, period)?;
        }
    }

    // Volume Indicators
    if !config.volume.volume_sma_periods.is_empty() {
        df = add_volume_indicators(
            df,
            &close_prices,
            &volume,
            &config.volume.volume_sma_periods,
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
            &high_prices,
            &low_prices,
            &close_prices,
            &config.volatility.atr_periods,
        )?;
    }

    if config.volatility.keltner_channels {
        df = add_keltner_channels(df, &high_prices, &low_prices, &close_prices, 20, 2.0)?;
    }

    // Cryptocurrency-specific indicators (always enabled for crypto markets)
    df = add_crypto_specific_indicators(
        df,
        &open_prices,
        &high_prices,
        &low_prices,
        &close_prices,
        &volume,
    )?;

    // Portfolio-specific indicators (single-asset versions)
    if config.liquidity_stress_indicators {
        df = add_liquidity_stress_indicators(df, &close_prices, &volume, 20)?;
    }

    // Note: Multi-asset indicators (relative_strength_vs_btc, relative_strength_eth_vs_btc, volume_ratio_vs_market,
    // cross_asset_correlation, sector_rotation_signals, correlation_breakdown_detection)
    // require additional data and should be implemented at the portfolio level
    if config.relative_strength_vs_btc {
        log::warn!(
            "relative_strength_vs_btc requires BTC price data - implement at portfolio level"
        );
    }
    if config.relative_strength_eth_vs_btc {
        log::warn!("relative_strength_eth_vs_btc requires ETH and BTC price data - implement at portfolio level");
    }
    if config.volume_ratio_vs_market {
        log::warn!(
            "volume_ratio_vs_market requires market volume data - implement at portfolio level"
        );
    }
    if config.cross_asset_correlation {
        log::warn!(
            "cross_asset_correlation requires multiple assets - implement at portfolio level"
        );
    }
    if config.sector_rotation_signals {
        log::warn!(
            "sector_rotation_signals requires multiple assets - implement at portfolio level"
        );
    }
    if config.correlation_breakdown_detection {
        log::warn!("correlation_breakdown_detection requires multiple assets - implement at portfolio level");
    }

    log::info!("Generated {} technical indicators", df.width() - 6); // Subtract OHLCV + timestamp
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
fn calculate_sma(data: &[f64], period: usize) -> Vec<f64> {
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

/// Calculate exponential moving average
fn calculate_ema(data: &[f64], period: usize) -> Vec<f64> {
    let mut result = vec![f64::NAN; data.len()];

    if data.is_empty() || period == 0 {
        return result;
    }

    let alpha = 2.0 / (period as f64 + 1.0);
    result[0] = data[0];

    for i in 1..data.len() {
        result[i] = alpha * data[i] + (1.0 - alpha) * result[i - 1];
    }

    result
}

/// Calculate RSI (Relative Strength Index)
fn calculate_rsi(data: &[f64], period: usize) -> Vec<f64> {
    let mut result = vec![f64::NAN; data.len()];

    if data.len() < period + 1 {
        return result;
    }

    let mut gains = Vec::new();
    let mut losses = Vec::new();

    // Calculate price changes
    for i in 1..data.len() {
        let change = data[i] - data[i - 1];
        if change > 0.0 {
            gains.push(change);
            losses.push(0.0);
        } else {
            gains.push(0.0);
            losses.push(-change);
        }
    }

    // Calculate average gains and losses
    for i in (period - 1)..gains.len() {
        let avg_gain = gains[(i + 1 - period)..=i].iter().sum::<f64>() / period as f64;
        let avg_loss = losses[(i + 1 - period)..=i].iter().sum::<f64>() / period as f64;

        if avg_loss == 0.0 {
            result[i + 1] = 100.0;
        } else {
            let rs = avg_gain / avg_loss;
            result[i + 1] = 100.0 - (100.0 / (1.0 + rs));
        }
    }

    result
}

/// Calculate MACD (Moving Average Convergence Divergence)
fn calculate_macd(
    data: &[f64],
    fast: usize,
    slow: usize,
    signal: usize,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let ema_fast = calculate_ema(data, fast);
    let ema_slow = calculate_ema(data, slow);

    let mut macd_line = vec![f64::NAN; data.len()];
    for i in 0..data.len() {
        if !ema_fast[i].is_nan() && !ema_slow[i].is_nan() {
            macd_line[i] = ema_fast[i] - ema_slow[i];
        }
    }

    let signal_line = calculate_ema(&macd_line, signal);

    let mut histogram = vec![f64::NAN; data.len()];
    for i in 0..data.len() {
        if !macd_line[i].is_nan() && !signal_line[i].is_nan() {
            histogram[i] = macd_line[i] - signal_line[i];
        }
    }

    (macd_line, signal_line, histogram)
}

/// Calculate Bollinger Bands
fn calculate_bollinger_bands(
    data: &[f64],
    period: usize,
    std_dev: f64,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let sma = calculate_sma(data, period);
    let mut upper = vec![f64::NAN; data.len()];
    let mut lower = vec![f64::NAN; data.len()];

    for i in (period - 1)..data.len() {
        let window = &data[(i + 1 - period)..=i];
        let mean = sma[i];

        if !mean.is_nan() {
            let variance = window.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / period as f64;
            let std = variance.sqrt();

            upper[i] = mean + std_dev * std;
            lower[i] = mean - std_dev * std;
        }
    }

    (upper, sma, lower)
}

/// Calculate Stochastic Oscillator
fn calculate_stochastic(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    k_period: usize,
    d_period: usize,
) -> (Vec<f64>, Vec<f64>) {
    let mut k_values = vec![f64::NAN; close.len()];

    for i in (k_period - 1)..close.len() {
        let window_high = high[(i + 1 - k_period)..=i]
            .iter()
            .fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let window_low = low[(i + 1 - k_period)..=i]
            .iter()
            .fold(f64::INFINITY, |a, &b| a.min(b));

        if window_high != window_low {
            k_values[i] = 100.0 * (close[i] - window_low) / (window_high - window_low);
        }
    }

    let d_values = calculate_sma(&k_values, d_period);
    (k_values, d_values)
}

/// Calculate Williams %R
fn calculate_williams_r(high: &[f64], low: &[f64], close: &[f64], period: usize) -> Vec<f64> {
    let mut result = vec![f64::NAN; close.len()];

    for i in (period - 1)..close.len() {
        let window_high = high[(i + 1 - period)..=i]
            .iter()
            .fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let window_low = low[(i + 1 - period)..=i]
            .iter()
            .fold(f64::INFINITY, |a, &b| a.min(b));

        if window_high != window_low {
            result[i] = -100.0 * (window_high - close[i]) / (window_high - window_low);
        }
    }

    result
}

/// Calculate Commodity Channel Index (CCI)
fn calculate_cci(high: &[f64], low: &[f64], close: &[f64], period: usize) -> Vec<f64> {
    let mut typical_price = vec![0.0; close.len()];
    for i in 0..close.len() {
        typical_price[i] = (high[i] + low[i] + close[i]) / 3.0;
    }

    let sma_tp = calculate_sma(&typical_price, period);
    let mut result = vec![f64::NAN; close.len()];

    for i in (period - 1)..close.len() {
        let window = &typical_price[(i + 1 - period)..=i];
        let mean = sma_tp[i];

        if !mean.is_nan() {
            let mean_deviation =
                window.iter().map(|&x| (x - mean).abs()).sum::<f64>() / period as f64;

            if mean_deviation != 0.0 {
                result[i] = (typical_price[i] - mean) / (0.015 * mean_deviation);
            }
        }
    }

    result
}

/// Calculate Average True Range (ATR)
fn calculate_atr(high: &[f64], low: &[f64], close: &[f64], period: usize) -> Vec<f64> {
    if close.len() < 2 {
        return vec![f64::NAN; close.len()];
    }

    let mut true_range = vec![f64::NAN; close.len()];

    for i in 1..close.len() {
        let tr1 = high[i] - low[i];
        let tr2 = (high[i] - close[i - 1]).abs();
        let tr3 = (low[i] - close[i - 1]).abs();
        true_range[i] = tr1.max(tr2).max(tr3);
    }

    calculate_sma(&true_range, period)
}

/// Calculate On-Balance Volume (OBV)
fn calculate_obv(close: &[f64], volume: &[f64]) -> Vec<f64> {
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

        if neg_sum != 0.0 {
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
            let sma = calculate_sma(close_prices, period as usize);
            (format!("sma_{}", period), sma)
        })
        .collect();

    // Add all computed SMAs to DataFrame
    for (column_name, sma_values) in sma_results {
        let series = Series::new(&column_name, sma_values);
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
            let ema = calculate_ema(close_prices, period as usize);
            (format!("ema_{}", period), ema)
        })
        .collect();

    // Add all computed EMAs to DataFrame
    for (column_name, ema_values) in ema_results {
        let series = Series::new(&column_name, ema_values);
        df = df
            .with_column(series)
            .map_err(|e| {
                VangaError::FeatureError(format!("Failed to add {} column: {}", column_name, e))
            })?
            .clone();
    }
    Ok(df)
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
        calculate_macd(close_prices, fast as usize, slow as usize, signal as usize);

    df = df
        .with_column(Series::new("macd", macd_line))
        .map_err(|e| VangaError::FeatureError(format!("Failed to add MACD column: {}", e)))?
        .clone();

    df = df
        .with_column(Series::new("macd_signal", signal_line))
        .map_err(|e| VangaError::FeatureError(format!("Failed to add MACD signal column: {}", e)))?
        .clone();

    df = df
        .with_column(Series::new("macd_histogram", histogram))
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
    let (upper, middle, lower) = calculate_bollinger_bands(close_prices, period as usize, std_dev);

    df = df
        .with_column(Series::new("bb_upper", upper))
        .map_err(|e| {
            VangaError::FeatureError(format!("Failed to add Bollinger upper band: {}", e))
        })?
        .clone();

    df = df
        .with_column(Series::new("bb_middle", middle))
        .map_err(|e| {
            VangaError::FeatureError(format!("Failed to add Bollinger middle band: {}", e))
        })?
        .clone();

    df = df
        .with_column(Series::new("bb_lower", lower))
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
    // Compute all RSI periods in parallel
    let rsi_results: Vec<_> = periods
        .par_iter()
        .map(|&period| {
            let rsi = calculate_rsi(close_prices, period as usize);
            (format!("rsi_{}", period), rsi)
        })
        .collect();

    // Add all computed RSIs to DataFrame
    for (column_name, rsi_values) in rsi_results {
        let series = Series::new(&column_name, rsi_values);
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
fn add_stochastic_indicators(
    mut df: DataFrame,
    high: &[f64],
    low: &[f64],
    close: &[f64],
    k_period: u32,
    d_period: u32,
) -> Result<DataFrame> {
    let (k_values, d_values) =
        calculate_stochastic(high, low, close, k_period as usize, d_period as usize);

    df = df
        .with_column(Series::new("stoch_k", k_values))
        .map_err(|e| VangaError::FeatureError(format!("Failed to add Stochastic %K: {}", e)))?
        .clone();

    df = df
        .with_column(Series::new("stoch_d", d_values))
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
    let williams_r_values = calculate_williams_r(high, low, close, period as usize);
    df = df
        .with_column(Series::new("williams_r", williams_r_values))
        .map_err(|e| VangaError::FeatureError(format!("Failed to add Williams %R: {}", e)))?
        .clone();
    Ok(df)
}

/// Add CCI indicator to DataFrame
fn add_cci_indicator(
    mut df: DataFrame,
    high: &[f64],
    low: &[f64],
    close: &[f64],
    period: u32,
) -> Result<DataFrame> {
    let cci_values = calculate_cci(high, low, close, period as usize);
    let column_name = format!("cci_{}", period);
    df = df
        .with_column(Series::new(&column_name, cci_values))
        .map_err(|e| VangaError::FeatureError(format!("Failed to add {}: {}", column_name, e)))?
        .clone();
    Ok(df)
}

/// Add Volume indicators to DataFrame
fn add_volume_indicators(
    mut df: DataFrame,
    close: &[f64],
    volume: &[f64],
    periods: &[u32],
) -> Result<DataFrame> {
    for &period in periods {
        let volume_sma = calculate_sma(volume, period as usize);
        let column_name = format!("volume_sma_{}", period);
        df = df
            .with_column(Series::new(&column_name, volume_sma))
            .map_err(|e| VangaError::FeatureError(format!("Failed to add {}: {}", column_name, e)))?
            .clone();

        // Add price-volume correlation indicator
        let pv_correlation = calculate_price_volume_correlation(close, volume, period as usize);
        let corr_column_name = format!("pv_correlation_{}", period);
        df = df
            .with_column(Series::new(&corr_column_name, pv_correlation))
            .map_err(|e| {
                VangaError::FeatureError(format!("Failed to add {}: {}", corr_column_name, e))
            })?
            .clone();

        // Add volume-weighted price indicator
        let vwap = calculate_volume_weighted_average_price(close, volume, period as usize);
        let vwap_column_name = format!("vwap_{}", period);
        df = df
            .with_column(Series::new(&vwap_column_name, vwap))
            .map_err(|e| {
                VangaError::FeatureError(format!("Failed to add {}: {}", vwap_column_name, e))
            })?
            .clone();
    }
    Ok(df)
}

/// Add OBV indicator to DataFrame
fn add_obv_indicator(mut df: DataFrame, close: &[f64], volume: &[f64]) -> Result<DataFrame> {
    let obv_values = calculate_obv(close, volume);
    df = df
        .with_column(Series::new("obv", obv_values))
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
    df = df
        .with_column(Series::new(&column_name, mfi_values))
        .map_err(|e| VangaError::FeatureError(format!("Failed to add {}: {}", column_name, e)))?
        .clone();
    Ok(df)
}

/// Add ATR indicators to DataFrame
fn add_atr_indicators(
    mut df: DataFrame,
    high: &[f64],
    low: &[f64],
    close: &[f64],
    periods: &[u32],
) -> Result<DataFrame> {
    for &period in periods {
        let atr_values = calculate_atr(high, low, close, period as usize);
        let column_name = format!("atr_{}", period);
        df = df
            .with_column(Series::new(&column_name, atr_values))
            .map_err(|e| VangaError::FeatureError(format!("Failed to add {}: {}", column_name, e)))?
            .clone();
    }
    Ok(df)
}

/// Add Keltner Channels to DataFrame
fn add_keltner_channels(
    mut df: DataFrame,
    high: &[f64],
    low: &[f64],
    close: &[f64],
    period: u32,
    multiplier: f64,
) -> Result<DataFrame> {
    let ema_values = calculate_ema(close, period as usize);
    let atr_values = calculate_atr(high, low, close, period as usize);

    let mut keltner_upper = vec![f64::NAN; close.len()];
    let mut keltner_lower = vec![f64::NAN; close.len()];

    for i in 0..close.len() {
        if !ema_values[i].is_nan() && !atr_values[i].is_nan() {
            keltner_upper[i] = ema_values[i] + multiplier * atr_values[i];
            keltner_lower[i] = ema_values[i] - multiplier * atr_values[i];
        }
    }

    df = df
        .with_column(Series::new("keltner_upper", keltner_upper))
        .map_err(|e| VangaError::FeatureError(format!("Failed to add Keltner upper: {}", e)))?
        .clone();

    df = df
        .with_column(Series::new("keltner_lower", keltner_lower))
        .map_err(|e| VangaError::FeatureError(format!("Failed to add Keltner lower: {}", e)))?
        .clone();

    Ok(df)
}

/// Add cryptocurrency-specific indicators
fn add_crypto_specific_indicators(
    mut df: DataFrame,
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
) -> Result<DataFrame> {
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

    // Price gaps (difference between current open and previous close)
    let mut price_gaps = vec![0.0; close.len()];
    for i in 1..close.len() {
        price_gaps[i] = (open[i] - close[i - 1]) / close[i - 1] * 100.0;
    }

    // Intraday range (high - low) as percentage of close
    let mut intraday_range = vec![0.0; close.len()];
    for i in 0..close.len() {
        intraday_range[i] = (high[i] - low[i]) / close[i] * 100.0;
    }

    // Add indicators
    df = df
        .with_column(Series::new("price_velocity", price_velocity))
        .map_err(|e| VangaError::FeatureError(format!("Failed to add price velocity: {}", e)))?
        .clone();

    df = df
        .with_column(Series::new("vwap", vwap))
        .map_err(|e| VangaError::FeatureError(format!("Failed to add VWAP: {}", e)))?
        .clone();

    df = df
        .with_column(Series::new("price_gaps", price_gaps))
        .map_err(|e| VangaError::FeatureError(format!("Failed to add price gaps: {}", e)))?
        .clone();

    df = df
        .with_column(Series::new("intraday_range", intraday_range))
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

/// Calculate relative strength vs BTC (price ratio normalized)
fn calculate_relative_strength_vs_btc(asset_close: &[f64], btc_close: &[f64]) -> Vec<f64> {
    let mut relative_strength = vec![f64::NAN; asset_close.len()];

    if asset_close.len() != btc_close.len() {
        return relative_strength;
    }

    for i in 0..asset_close.len() {
        if btc_close[i] > 0.0 && !btc_close[i].is_nan() && !asset_close[i].is_nan() {
            relative_strength[i] = asset_close[i] / btc_close[i];
        }
    }

    relative_strength
}

/// Calculate ETH/BTC dominance ratio for crypto market cycle analysis
fn calculate_eth_btc_dominance_ratio(eth_close: &[f64], btc_close: &[f64]) -> Vec<f64> {
    let mut eth_btc_ratio = vec![f64::NAN; eth_close.len()];

    if eth_close.len() != btc_close.len() {
        return eth_btc_ratio;
    }

    for i in 0..eth_close.len() {
        if btc_close[i] > 0.0 && !btc_close[i].is_nan() && !eth_close[i].is_nan() {
            eth_btc_ratio[i] = eth_close[i] / btc_close[i];
        }
    }

    eth_btc_ratio
}

/// Calculate ETH/BTC cycle phase detection (alt season vs BTC dominance)
fn calculate_eth_btc_cycle_phases(eth_btc_ratio: &[f64]) -> Vec<f64> {
    let mut cycle_phase = vec![f64::NAN; eth_btc_ratio.len()];

    // Crypto market proven thresholds
    const ALT_SEASON_THRESHOLD: f64 = 0.08; // ETH/BTC > 0.08 = alt season
    const BTC_DOMINANCE_THRESHOLD: f64 = 0.05; // ETH/BTC < 0.05 = BTC dominance

    for i in 0..eth_btc_ratio.len() {
        if !eth_btc_ratio[i].is_nan() {
            cycle_phase[i] = if eth_btc_ratio[i] > ALT_SEASON_THRESHOLD {
                1.0 // Alt season
            } else if eth_btc_ratio[i] < BTC_DOMINANCE_THRESHOLD {
                -1.0 // BTC dominance
            } else {
                0.0 // Neutral/transition zone
            };
        }
    }

    cycle_phase
}

/// Calculate ETH/BTC momentum (rate of change in dominance)
fn calculate_eth_btc_momentum(eth_btc_ratio: &[f64], period: usize) -> Vec<f64> {
    let mut momentum = vec![f64::NAN; eth_btc_ratio.len()];

    if period == 0 || eth_btc_ratio.len() <= period {
        return momentum;
    }

    for i in period..eth_btc_ratio.len() {
        if !eth_btc_ratio[i].is_nan()
            && !eth_btc_ratio[i - period].is_nan()
            && eth_btc_ratio[i - period] > 0.0
        {
            momentum[i] =
                (eth_btc_ratio[i] - eth_btc_ratio[i - period]) / eth_btc_ratio[i - period] * 100.0;
        }
    }

    momentum
}

/// Calculate ETH/BTC volatility regime detection
fn calculate_eth_btc_volatility_regime(eth_btc_ratio: &[f64], period: usize) -> Vec<f64> {
    let mut volatility_regime = vec![f64::NAN; eth_btc_ratio.len()];

    if period < 2 || eth_btc_ratio.len() <= period {
        return volatility_regime;
    }

    for i in period..eth_btc_ratio.len() {
        let window = &eth_btc_ratio[i - period..i];
        let valid_values: Vec<f64> = window.iter().filter(|&&x| !x.is_nan()).copied().collect();

        if valid_values.len() >= 2 {
            let mean = valid_values.iter().sum::<f64>() / valid_values.len() as f64;
            let variance = valid_values
                .iter()
                .map(|&x| (x - mean).powi(2))
                .sum::<f64>()
                / valid_values.len() as f64;
            let volatility = variance.sqrt();

            // Normalize volatility relative to price level
            volatility_regime[i] = if mean > 0.0 {
                volatility / mean * 100.0 // Coefficient of variation as percentage
            } else {
                f64::NAN
            };
        }
    }

    volatility_regime
}

/// Calculate volume ratio vs market average (portfolio-level function)
pub fn calculate_volume_ratio_vs_market(volume: &[f64], market_volume: &[f64]) -> Vec<f64> {
    let mut volume_ratio = vec![f64::NAN; volume.len()];

    if volume.len() != market_volume.len() {
        return volume_ratio;
    }

    for i in 0..volume.len() {
        if market_volume[i] > 0.0 && !market_volume[i].is_nan() && !volume[i].is_nan() {
            volume_ratio[i] = volume[i] / market_volume[i];
        }
    }

    volume_ratio
}

/// Calculate cross-asset correlation matrix features (portfolio-level function)
pub fn calculate_cross_asset_correlation(prices: &[Vec<f64>], period: usize) -> Vec<f64> {
    if prices.len() < 2 {
        return vec![f64::NAN; prices[0].len()];
    }

    let data_len = prices[0].len();
    let mut avg_correlation = vec![f64::NAN; data_len];

    for (i, correlation_value) in avg_correlation
        .iter_mut()
        .enumerate()
        .take(data_len)
        .skip(period)
    {
        let mut correlations = Vec::new();

        // Calculate pairwise correlations for current window
        for j in 0..prices.len() {
            for k in (j + 1)..prices.len() {
                let window_j = &prices[j][i - period..i];
                let window_k = &prices[k][i - period..i];

                // Reuse existing correlation calculation logic
                let mean_j = window_j.iter().sum::<f64>() / period as f64;
                let mean_k = window_k.iter().sum::<f64>() / period as f64;

                let mut numerator = 0.0;
                let mut var_j = 0.0;
                let mut var_k = 0.0;

                for l in 0..period {
                    let diff_j = window_j[l] - mean_j;
                    let diff_k = window_k[l] - mean_k;

                    numerator += diff_j * diff_k;
                    var_j += diff_j * diff_j;
                    var_k += diff_k * diff_k;
                }

                let denominator = (var_j * var_k).sqrt();
                if denominator > 0.0 {
                    correlations.push(numerator / denominator);
                }
            }
        }

        // Average correlation across all pairs
        if !correlations.is_empty() {
            *correlation_value = correlations.iter().sum::<f64>() / correlations.len() as f64;
        }
    }

    avg_correlation
}

/// Calculate sector rotation signals (momentum divergence between assets) (portfolio-level function)
pub fn calculate_sector_rotation_signals(prices: &[Vec<f64>], period: usize) -> Vec<f64> {
    if prices.len() < 2 {
        return vec![f64::NAN; prices[0].len()];
    }

    let data_len = prices[0].len();
    let mut rotation_signal = vec![f64::NAN; data_len];

    for i in period..data_len {
        let mut momentum_values = Vec::new();

        // Calculate momentum for each asset
        for price_series in prices {
            if i >= period && price_series[i - period] > 0.0 {
                let momentum =
                    (price_series[i] - price_series[i - period]) / price_series[i - period];
                momentum_values.push(momentum);
            }
        }

        if momentum_values.len() >= 2 {
            // Calculate momentum dispersion (standard deviation)
            let mean_momentum = momentum_values.iter().sum::<f64>() / momentum_values.len() as f64;
            let variance = momentum_values
                .iter()
                .map(|&x| (x - mean_momentum).powi(2))
                .sum::<f64>()
                / momentum_values.len() as f64;

            rotation_signal[i] = variance.sqrt(); // Higher values = more rotation
        }
    }

    rotation_signal
}

/// Calculate correlation breakdown detection (portfolio-level function)
pub fn calculate_correlation_breakdown_detection(
    prices: &[Vec<f64>],
    short_period: usize,
    long_period: usize,
) -> Vec<f64> {
    let short_corr = calculate_cross_asset_correlation(prices, short_period);
    let long_corr = calculate_cross_asset_correlation(prices, long_period);

    let mut breakdown_signal = vec![f64::NAN; short_corr.len()];

    for i in 0..short_corr.len() {
        if !short_corr[i].is_nan() && !long_corr[i].is_nan() {
            // Breakdown signal = difference between short and long-term correlation
            breakdown_signal[i] = (short_corr[i] - long_corr[i]).abs();
        }
    }

    breakdown_signal
}

/// Calculate liquidity stress indicators (volume volatility and price impact)
fn calculate_liquidity_stress_indicators(close: &[f64], volume: &[f64], period: usize) -> Vec<f64> {
    let mut stress_indicator = vec![f64::NAN; close.len()];

    if close.len() != volume.len() || period < 2 {
        return stress_indicator;
    }

    for i in period..close.len() {
        let volume_window = &volume[i - period..i];
        let price_window = &close[i - period..i];

        // Calculate volume volatility
        let volume_mean = volume_window.iter().sum::<f64>() / period as f64;

        // Calculate price volatility
        let mut price_changes = Vec::new();
        for j in 1..price_window.len() {
            if price_window[j - 1] > 0.0 {
                price_changes.push((price_window[j] - price_window[j - 1]) / price_window[j - 1]);
            }
        }

        if !price_changes.is_empty() && volume_mean > 0.0 {
            let price_volatility = {
                let mean = price_changes.iter().sum::<f64>() / price_changes.len() as f64;
                let variance = price_changes
                    .iter()
                    .map(|&x| (x - mean).powi(2))
                    .sum::<f64>()
                    / price_changes.len() as f64;
                variance.sqrt()
            };

            // Stress = price volatility / volume (higher when low volume + high volatility)
            stress_indicator[i] =
                price_volatility / (volume_mean / volume_window.iter().sum::<f64>()).max(0.001);
        }
    }

    stress_indicator
}

/// Add liquidity stress indicators to DataFrame
fn add_liquidity_stress_indicators(
    mut df: DataFrame,
    close: &[f64],
    volume: &[f64],
    period: u32,
) -> Result<DataFrame> {
    let stress_values = calculate_liquidity_stress_indicators(close, volume, period as usize);
    let column_name = format!("liquidity_stress_{}", period);
    df = df
        .with_column(Series::new(&column_name, stress_values))
        .map_err(|e| VangaError::FeatureError(format!("Failed to add {}: {}", column_name, e)))?
        .clone();
    Ok(df)
}

/// Add ETH/BTC dominance indicators to DataFrame (portfolio-level function)
/// This function should be called when both ETH and BTC data are available
pub fn add_eth_btc_dominance_indicators(
    mut df: DataFrame,
    eth_close: &[f64],
    btc_close: &[f64],
) -> Result<DataFrame> {
    // Basic ETH/BTC ratio
    let eth_btc_ratio = calculate_eth_btc_dominance_ratio(eth_close, btc_close);
    df = df
        .with_column(Series::new("eth_btc_ratio", eth_btc_ratio.clone()))
        .map_err(|e| VangaError::FeatureError(format!("Failed to add eth_btc_ratio: {}", e)))?
        .clone();

    // Cycle phase detection
    let cycle_phases = calculate_eth_btc_cycle_phases(&eth_btc_ratio);
    df = df
        .with_column(Series::new("eth_btc_cycle_phase", cycle_phases))
        .map_err(|e| VangaError::FeatureError(format!("Failed to add eth_btc_cycle_phase: {}", e)))?
        .clone();

    // Momentum analysis (10-period default)
    let momentum_10 = calculate_eth_btc_momentum(&eth_btc_ratio, 10);
    df = df
        .with_column(Series::new("eth_btc_momentum_10", momentum_10))
        .map_err(|e| VangaError::FeatureError(format!("Failed to add eth_btc_momentum_10: {}", e)))?
        .clone();

    // Momentum analysis (20-period)
    let momentum_20 = calculate_eth_btc_momentum(&eth_btc_ratio, 20);
    df = df
        .with_column(Series::new("eth_btc_momentum_20", momentum_20))
        .map_err(|e| VangaError::FeatureError(format!("Failed to add eth_btc_momentum_20: {}", e)))?
        .clone();

    // Volatility regime (20-period)
    let volatility_regime = calculate_eth_btc_volatility_regime(&eth_btc_ratio, 20);
    df = df
        .with_column(Series::new("eth_btc_volatility_regime", volatility_regime))
        .map_err(|e| {
            VangaError::FeatureError(format!("Failed to add eth_btc_volatility_regime: {}", e))
        })?
        .clone();

    log::info!("Added ETH/BTC dominance indicators: ratio, cycle_phase, momentum_10, momentum_20, volatility_regime");
    Ok(df)
}

/// Add relative strength vs BTC indicators to DataFrame (portfolio-level function)
/// This function should be called when BTC data is available
pub fn add_relative_strength_vs_btc_indicators(
    mut df: DataFrame,
    asset_close: &[f64],
    btc_close: &[f64],
    asset_symbol: &str,
) -> Result<DataFrame> {
    let relative_strength = calculate_relative_strength_vs_btc(asset_close, btc_close);
    let column_name = format!("{}_btc_ratio", asset_symbol.to_lowercase());

    df = df
        .with_column(Series::new(&column_name, relative_strength))
        .map_err(|e| VangaError::FeatureError(format!("Failed to add {}: {}", column_name, e)))?
        .clone();

    log::info!("Added relative strength indicator: {}", column_name);
    Ok(df)
}

/// Add multi-timeframe ETH/BTC dominance analysis
/// Analyzes dominance across multiple timeframes for comprehensive market cycle detection
pub fn add_multi_timeframe_eth_btc_analysis(
    mut df: DataFrame,
    eth_close: &[f64],
    btc_close: &[f64],
    timeframes: &[u32], // periods like [10, 20, 50, 100] for different timeframes
) -> Result<DataFrame> {
    let eth_btc_ratio = calculate_eth_btc_dominance_ratio(eth_close, btc_close);

    // Add basic ratio first
    df = df
        .with_column(Series::new("eth_btc_ratio", eth_btc_ratio.clone()))
        .map_err(|e| VangaError::FeatureError(format!("Failed to add eth_btc_ratio: {}", e)))?
        .clone();

    // Add cycle phase detection
    let cycle_phases = calculate_eth_btc_cycle_phases(&eth_btc_ratio);
    df = df
        .with_column(Series::new("eth_btc_cycle_phase", cycle_phases))
        .map_err(|e| VangaError::FeatureError(format!("Failed to add eth_btc_cycle_phase: {}", e)))?
        .clone();

    // Multi-timeframe momentum analysis
    for &period in timeframes {
        let momentum = calculate_eth_btc_momentum(&eth_btc_ratio, period as usize);
        let column_name = format!("eth_btc_momentum_{}", period);
        df = df
            .with_column(Series::new(&column_name, momentum))
            .map_err(|e| VangaError::FeatureError(format!("Failed to add {}: {}", column_name, e)))?
            .clone();
    }

    // Multi-timeframe volatility regime analysis
    for &period in timeframes {
        let volatility_regime =
            calculate_eth_btc_volatility_regime(&eth_btc_ratio, period as usize);
        let column_name = format!("eth_btc_volatility_{}", period);
        df = df
            .with_column(Series::new(&column_name, volatility_regime))
            .map_err(|e| VangaError::FeatureError(format!("Failed to add {}: {}", column_name, e)))?
            .clone();
    }

    // Add dominance trend strength (SMA of momentum)
    for &period in timeframes {
        let momentum = calculate_eth_btc_momentum(&eth_btc_ratio, period as usize);
        let trend_strength = calculate_sma(&momentum, period as usize);
        let column_name = format!("eth_btc_trend_strength_{}", period);
        df = df
            .with_column(Series::new(&column_name, trend_strength))
            .map_err(|e| VangaError::FeatureError(format!("Failed to add {}: {}", column_name, e)))?
            .clone();
    }

    log::info!(
        "Added multi-timeframe ETH/BTC dominance analysis for periods: {:?}",
        timeframes
    );
    Ok(df)
}
