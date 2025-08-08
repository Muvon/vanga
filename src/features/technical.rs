// Technical indicators implementation with comprehensive indicators
use crate::config::features::TechnicalIndicatorsConfig;
use crate::features::ta_helpers::*; // Import TA crate helpers
use crate::utils::error::{Result, VangaError};
use polars::prelude::*;
use rayon::prelude::*;

/// OHLCV data structure to reduce function parameter count
#[derive(Clone)]
struct OhlcvData<'a> {
    open: &'a [f64],
    high: &'a [f64],
    low: &'a [f64],
    close: &'a [f64],
    volume: &'a [f64],
}

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
        )?;
    }

    if config.volatility.keltner_channels {
        df = add_keltner_channels(df, &ohlcv, 20, 2.0)?;
    }

    // Cryptocurrency-specific indicators (always enabled for crypto markets)
    df = add_crypto_specific_indicators(
        df,
        &open_prices,
        &high_prices,
        &low_prices,
        &close_prices,
        &volume,
        &config.trend.advanced,
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
        calculate_macd_ta(close_prices, fast as usize, slow as usize, signal as usize)
            .unwrap_or_else(|e| {
                log::error!("MACD calculation failed: {}", e);
                let len = close_prices.len();
                (vec![0.0; len], vec![0.0; len], vec![0.0; len]) // Default to zeros
            });

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
    let (upper, middle, lower) =
        calculate_bollinger_bands_ta(close_prices, period as usize, std_dev).unwrap_or_else(|e| {
            log::error!("Bollinger Bands calculation failed: {}", e);
            let len = close_prices.len();
            (vec![0.0; len], vec![0.0; len], vec![0.0; len]) // Default to zeros
        });

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
    let williams_r_values = calculate_williams_r_ta(high, low, close, period as usize)
        .unwrap_or_else(|e| {
            log::error!("Williams %R calculation failed: {}", e);
            vec![-50.0; close.len()] // Default to neutral value
        });
    df = df
        .with_column(Series::new("williams_r", williams_r_values))
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
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    periods: &[u32],
) -> Result<DataFrame> {
    for &period in periods {
        let atr_values = calculate_atr_ta(open, high, low, close, volume, period as usize)
            .unwrap_or_else(|e| {
                log::error!("ATR calculation failed for period {}: {}", period, e);
                vec![f64::NAN; close.len()] // Use NaN for filtering
            });
        let column_name = format!("atr_{}", period);
        df = df
            .with_column(Series::new(&column_name, atr_values))
            .map_err(|e| VangaError::FeatureError(format!("Failed to add {}: {}", column_name, e)))?
            .clone();
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
    advanced_config: &crate::config::features::AdvancedIndicatorsConfig,
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
    // Improved to handle edge cases and provide more meaningful values
    let mut price_gaps = vec![0.0; close.len()];
    for i in 1..close.len() {
        if close[i - 1] > 0.0 {
            // Avoid division by zero
            let gap = (open[i] - close[i - 1]) / close[i - 1] * 100.0;
            // Clamp extreme gaps to reasonable range (-50% to +50%)
            price_gaps[i] = gap.clamp(-50.0, 50.0);
        }
    }

    // Add gap volatility as additional feature to make it more informative
    let mut gap_volatility = vec![0.0; close.len()];
    if close.len() > 20 {
        for i in 20..close.len() {
            let recent_gaps = &price_gaps[i - 19..i + 1];
            let mean_gap = recent_gaps.iter().sum::<f64>() / 20.0;
            let variance = recent_gaps
                .iter()
                .map(|&x| (x - mean_gap).powi(2))
                .sum::<f64>()
                / 20.0;
            gap_volatility[i] = variance.sqrt();
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
        let fractal_dims = calculate_fractal_dimension(close, advanced_config.fractal_window);
        let regime_values =
            calculate_regime_indicator(close, volume, advanced_config.regime_window);
        let clustering_values =
            calculate_volatility_clustering(close, advanced_config.clustering_window);
        let reversion_values =
            calculate_mean_reversion_strength(close, advanced_config.reversion_window);

        // Add advanced mathematical indicators to DataFrame
        df = df
            .with_column(Series::new("hurst_exponent", hurst_values))
            .map_err(|e| VangaError::FeatureError(format!("Failed to add hurst_exponent: {}", e)))?
            .clone();

        df = df
            .with_column(Series::new("fractal_dimension", fractal_dims))
            .map_err(|e| {
                VangaError::FeatureError(format!("Failed to add fractal_dimension: {}", e))
            })?
            .clone();

        df = df
            .with_column(Series::new("regime_indicator", regime_values))
            .map_err(|e| {
                VangaError::FeatureError(format!("Failed to add regime_indicator: {}", e))
            })?
            .clone();

        df = df
            .with_column(Series::new("volatility_clustering", clustering_values))
            .map_err(|e| {
                VangaError::FeatureError(format!("Failed to add volatility_clustering: {}", e))
            })?
            .clone();

        df = df
            .with_column(Series::new("mean_reversion_strength", reversion_values))
            .map_err(|e| {
                VangaError::FeatureError(format!("Failed to add mean_reversion_strength: {}", e))
            })?
            .clone();

        log::debug!("Added {} advanced mathematical indicators", 5);
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
        .with_column(Series::new("gap_volatility", gap_volatility))
        .map_err(|e| VangaError::FeatureError(format!("Failed to add gap volatility: {}", e)))?
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

/// Calculate Fractal Dimension using box-counting method
/// Higher values indicate more complex, chaotic price movements
fn calculate_fractal_dimension(prices: &[f64], window: usize) -> Vec<f64> {
    let mut fractal_dims = vec![f64::NAN; prices.len()]; // Start with NaN for filtering

    if prices.len() < window || window < 10 {
        log::warn!(
            "Fractal dimension: insufficient data (len={}, window={}, min_window=10)",
            prices.len(),
            window
        );
        return fractal_dims;
    }

    for i in window..prices.len() {
        let price_window = &prices[i - window..i];

        // Find price range
        let min_price = price_window.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_price = price_window
            .iter()
            .fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let price_range = max_price - min_price;

        // Handle constant prices with small random variation to avoid zero range
        let effective_range = if price_range <= 1e-10 {
            // For constant prices, use a small fraction of the price level as range
            let avg_price = price_window.iter().sum::<f64>() / price_window.len() as f64;
            avg_price * 1e-6 // 0.0001% of price as minimum range
        } else {
            price_range
        };

        // Box-counting method with multiple scales (increased from 7 to 12 scales)
        let mut scales = Vec::new();
        let mut counts = Vec::new();

        for scale_exp in 1..13 {
            let scale = effective_range / (2_f64.powi(scale_exp));
            if scale <= 0.0 {
                continue;
            }

            let boxes = (effective_range / scale).ceil() as usize;
            if boxes == 0 {
                continue;
            }

            let mut occupied_boxes = std::collections::HashSet::new();
            for &price in price_window {
                let box_index = ((price - min_price) / scale).floor() as usize;
                occupied_boxes.insert(box_index.min(boxes - 1));
            }

            if !occupied_boxes.is_empty() {
                scales.push(scale.ln());
                counts.push((occupied_boxes.len() as f64).ln());
            }
        }

        // Linear regression to find fractal dimension
        if scales.len() >= 4 {
            // Require at least 4 points for better regression
            let n = scales.len() as f64;
            let sum_x = scales.iter().sum::<f64>();
            let sum_y = counts.iter().sum::<f64>();
            let sum_xy = scales
                .iter()
                .zip(counts.iter())
                .map(|(x, y)| x * y)
                .sum::<f64>();
            let sum_x2 = scales.iter().map(|x| x * x).sum::<f64>();

            let denominator = n * sum_x2 - sum_x * sum_x;
            if denominator.abs() > 1e-10 {
                let slope = (n * sum_xy - sum_x * sum_y) / denominator;
                // Fractal dimension is negative slope, clamped to [1.0, 2.0]
                let fractal_dim = (-slope).clamp(1.0, 2.0);

                // Validate the result
                if fractal_dim.is_finite() {
                    fractal_dims[i] = fractal_dim;
                }
            }
        }
        // If regression fails, keep NaN for filtering
    }

    fractal_dims
}

/// Calculate Regime Indicator combining volatility, trend, and volume signals
/// Returns 0-3 scale: 0=stable/ranging, 3=high volatility/trending/high volume
fn calculate_regime_indicator(prices: &[f64], volume: &[f64], window: usize) -> Vec<f64> {
    let mut regime_values = vec![f64::NAN; prices.len()];

    if prices.len() < window || volume.len() != prices.len() || window < 5 {
        return regime_values;
    }

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
        let vol_threshold = 0.0004; // Crypto-optimized threshold
        let vol_regime = if volatility > vol_threshold { 1.0 } else { 0.0 };

        // 2. Trend regime (trending/ranging)
        let price_start = price_window[0];
        let price_end = price_window[price_window.len() - 1];
        let price_change = (price_end - price_start) / price_start;
        let trend_strength = price_change.abs();
        let trend_threshold = 0.02; // 2% threshold for crypto
        let trend_regime = if trend_strength > trend_threshold {
            1.0
        } else {
            0.0
        };

        // 3. Volume regime (high/low volume)
        let avg_volume = volume_window.iter().sum::<f64>() / volume_window.len() as f64;
        let recent_volume = volume_window[volume_window.len() - 1];
        let volume_regime = if recent_volume > avg_volume * 1.5 {
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
