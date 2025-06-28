// Technical indicators implementation with comprehensive indicators
use crate::config::features::TechnicalIndicatorsConfig;
use crate::utils::error::{Result, VangaError};
use polars::prelude::*;

/// Generate comprehensive technical indicators for cryptocurrency data
pub async fn generate_technical_indicators(
    mut df: DataFrame,
    config: &TechnicalIndicatorsConfig,
) -> Result<DataFrame> {
    log::info!("Generating comprehensive technical indicators...");

    // Extract OHLCV data for calculations
    let close_prices = extract_numeric_column(&df, "close")?;
    let high_prices = extract_numeric_column(&df, "high")?;
    let low_prices = extract_numeric_column(&df, "low")?;
    let open_prices = extract_numeric_column(&df, "open")?;
    let volume = extract_numeric_column(&df, "volume")?;

    // Trend Indicators - using actual config structure
    if !config.moving_averages.sma_periods.is_empty() {
        df = add_sma_indicators(df, &close_prices, &config.moving_averages.sma_periods)?;
    }

    if !config.moving_averages.ema_periods.is_empty() {
        df = add_ema_indicators(df, &close_prices, &config.moving_averages.ema_periods)?;
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

/// Add SMA indicators to DataFrame
fn add_sma_indicators(
    mut df: DataFrame,
    close_prices: &[f64],
    periods: &[u32],
) -> Result<DataFrame> {
    for &period in periods {
        let sma_values = calculate_sma(close_prices, period as usize);
        let column_name = format!("sma_{}", period);
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

/// Add EMA indicators to DataFrame
fn add_ema_indicators(
    mut df: DataFrame,
    close_prices: &[f64],
    periods: &[u32],
) -> Result<DataFrame> {
    for &period in periods {
        let ema_values = calculate_ema(close_prices, period as usize);
        let column_name = format!("ema_{}", period);
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

/// Add RSI indicators to DataFrame
fn add_rsi_indicators(
    mut df: DataFrame,
    close_prices: &[f64],
    periods: &[u32],
) -> Result<DataFrame> {
    for &period in periods {
        let rsi_values = calculate_rsi(close_prices, period as usize);
        let column_name = format!("rsi_{}", period);
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
    _close: &[f64],
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
