// TA crate integration helpers
use crate::features::validation::*;
use crate::utils::{Result, VangaError};
use ta::indicators::*;
use ta::{DataItem, Next};

/// Calculate RSI using ta crate with proper error handling
pub fn calculate_rsi_ta(data: &[f64], period: usize) -> Result<Vec<f64>> {
    // Validate inputs
    validate_ohlcv_data(None, None, None, data, None)?;
    validate_period(period, data.len(), "RSI")?;
    check_data_variation(data, "RSI", 1e-6)?;

    let mut rsi = RelativeStrengthIndex::new(period)
        .map_err(|e| VangaError::FeatureError(format!("RSI initialization failed: {}", e)))?;

    let mut result = Vec::with_capacity(data.len());

    for (i, &price) in data.iter().enumerate() {
        let rsi_value = rsi.next(price);

        // Return NaN for the warm-up period (first 'period' values)
        // This maintains compatibility with the original filtering system
        if i < period {
            result.push(f64::NAN);
        } else {
            // Handle constant prices: RSI returns NaN when there's no price movement
            // In this case, use 50.0 (neutral RSI) as it indicates no momentum
            if rsi_value.is_nan() {
                result.push(50.0);
            } else {
                result.push(rsi_value);
            }
        }
    }

    // Sanitize output (but preserve NaN values for filtering)
    let result = sanitize_indicator_output(result, "RSI", 50.0, Some((0.0, 100.0)));
    log_indicator_stats(&result, "RSI");

    Ok(result)
}

/// Calculate SMA using ta crate
pub fn calculate_sma_ta(data: &[f64], period: usize) -> Result<Vec<f64>> {
    // Validate inputs
    validate_ohlcv_data(None, None, None, data, None)?;
    validate_period(period, data.len(), "SMA")?;

    let mut sma = SimpleMovingAverage::new(period)
        .map_err(|e| VangaError::FeatureError(format!("SMA initialization failed: {}", e)))?;

    let mut result = Vec::with_capacity(data.len());

    for (i, &price) in data.iter().enumerate() {
        let sma_value = sma.next(price);

        // Return NaN for the warm-up period (first 'period-1' values)
        if i < period - 1 {
            result.push(f64::NAN);
        } else {
            result.push(sma_value);
        }
    }

    log_indicator_stats(&result, "SMA");
    Ok(result)
}

/// Calculate EMA using ta crate
pub fn calculate_ema_ta(data: &[f64], period: usize) -> Result<Vec<f64>> {
    // Validate inputs
    validate_ohlcv_data(None, None, None, data, None)?;
    validate_period(period, data.len(), "EMA")?;

    let mut ema = ExponentialMovingAverage::new(period)
        .map_err(|e| VangaError::FeatureError(format!("EMA initialization failed: {}", e)))?;

    let mut result = Vec::with_capacity(data.len());

    for (i, &price) in data.iter().enumerate() {
        let ema_value = ema.next(price);

        // EMA starts immediately but we can optionally add a small warm-up
        // For consistency with other indicators, let's add a minimal warm-up
        if i == 0 {
            result.push(f64::NAN); // First value as NaN for consistency
        } else {
            result.push(ema_value);
        }
    }

    log_indicator_stats(&result, "EMA");
    Ok(result)
}

/// Calculate MACD using ta crate
pub fn calculate_macd_ta(
    data: &[f64],
    fast: usize,
    slow: usize,
    signal: usize,
) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>)> {
    // Validate inputs
    validate_ohlcv_data(None, None, None, data, None)?;
    validate_macd_params(fast, slow, signal, data.len())?;
    check_data_variation(data, "MACD", 1e-6)?;

    let mut macd = MovingAverageConvergenceDivergence::new(fast, slow, signal)
        .map_err(|e| VangaError::FeatureError(format!("MACD initialization failed: {}", e)))?;

    let mut macd_line = Vec::with_capacity(data.len());
    let mut signal_line = Vec::with_capacity(data.len());
    let mut histogram = Vec::with_capacity(data.len());

    for (i, &price) in data.iter().enumerate() {
        let macd_output = macd.next(price);

        // Return NaN for the warm-up period (slow period + signal period)
        let warmup_period = slow + signal - 1;
        if i < warmup_period {
            macd_line.push(f64::NAN);
            signal_line.push(f64::NAN);
            histogram.push(f64::NAN);
        } else {
            macd_line.push(macd_output.macd);
            signal_line.push(macd_output.signal);
            histogram.push(macd_output.histogram);
        }
    }

    log_indicator_stats(&macd_line, "MACD Line");
    log_indicator_stats(&signal_line, "MACD Signal");
    log_indicator_stats(&histogram, "MACD Histogram");

    Ok((macd_line, signal_line, histogram))
}

/// Calculate Bollinger Bands using ta crate
pub fn calculate_bollinger_bands_ta(
    data: &[f64],
    period: usize,
    std_dev: f64,
) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>)> {
    // Validate inputs
    validate_ohlcv_data(None, None, None, data, None)?;
    validate_bollinger_params(period, std_dev, data.len())?;

    let mut bb = BollingerBands::new(period, std_dev).map_err(|e| {
        VangaError::FeatureError(format!("Bollinger Bands initialization failed: {}", e))
    })?;

    let mut upper = Vec::with_capacity(data.len());
    let mut middle = Vec::with_capacity(data.len());
    let mut lower = Vec::with_capacity(data.len());

    for (i, &price) in data.iter().enumerate() {
        let bb_output = bb.next(price);

        // Return NaN for the warm-up period
        if i < period - 1 {
            upper.push(f64::NAN);
            middle.push(f64::NAN);
            lower.push(f64::NAN);
        } else {
            upper.push(bb_output.upper);
            middle.push(bb_output.average);
            lower.push(bb_output.lower);
        }
    }

    log_indicator_stats(&upper, "Bollinger Upper");
    log_indicator_stats(&middle, "Bollinger Middle");
    log_indicator_stats(&lower, "Bollinger Lower");

    Ok((upper, middle, lower))
}

/// Calculate Stochastic using ta crate
pub fn calculate_stochastic_ta(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    k_period: usize,
    d_period: usize,
) -> Result<(Vec<f64>, Vec<f64>)> {
    if high.len() != low.len()
        || low.len() != close.len()
        || close.len() != open.len()
        || open.len() != volume.len()
    {
        return Err(VangaError::FeatureError(
            "Stochastic: mismatched array lengths".to_string(),
        ));
    }

    validate_ohlcv_data(Some(open), Some(high), Some(low), close, Some(volume))?;
    validate_stochastic_params(k_period, d_period, close.len())?;

    let mut stoch = FastStochastic::new(k_period).map_err(|e| {
        VangaError::FeatureError(format!("Stochastic initialization failed: {}", e))
    })?;

    let mut k_values = Vec::with_capacity(high.len());

    for i in 0..high.len() {
        let data_item = DataItem::builder()
            .open(open[i])
            .high(high[i])
            .low(low[i])
            .close(close[i])
            .volume(volume[i])
            .build()
            .map_err(|e| {
                VangaError::FeatureError(format!("Stochastic DataItem build failed: {}", e))
            })?;

        let k_value = stoch.next(&data_item);

        // Return NaN for the warm-up period
        if i < k_period - 1 {
            k_values.push(f64::NAN);
        } else {
            k_values.push(k_value);
        }
    }

    // Calculate %D as SMA of %K, but preserve NaN values
    let mut d_values = Vec::with_capacity(k_values.len());
    let mut sma_d = SimpleMovingAverage::new(d_period).map_err(|e| {
        VangaError::FeatureError(format!("Stochastic %D SMA initialization failed: {}", e))
    })?;

    for (i, &k_val) in k_values.iter().enumerate() {
        if k_val.is_nan() {
            d_values.push(f64::NAN);
        } else {
            let d_value = sma_d.next(k_val);
            // Additional warm-up for %D
            if i < k_period + d_period - 2 {
                d_values.push(f64::NAN);
            } else {
                d_values.push(d_value);
            }
        }
    }

    log_indicator_stats(&k_values, "Stochastic %K");
    log_indicator_stats(&d_values, "Stochastic %D");

    Ok((k_values, d_values))
}

/// Calculate Williams %R using manual implementation (not available in ta crate)
pub fn calculate_williams_r_ta(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    period: usize,
) -> Result<Vec<f64>> {
    if high.len() != low.len() || low.len() != close.len() {
        return Err(VangaError::FeatureError(
            "Williams %R: mismatched array lengths".to_string(),
        ));
    }

    validate_ohlcv_data(None, Some(high), Some(low), close, None)?;
    validate_period(period, close.len(), "Williams %R")?;

    let mut result = vec![f64::NAN; high.len()]; // Initialize with NaN

    for i in period..high.len() {
        let window_high = high[i + 1 - period..=i]
            .iter()
            .fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let window_low = low[i + 1 - period..=i]
            .iter()
            .fold(f64::INFINITY, |a, &b| a.min(b)); // Fix: use low instead of high

        let range = window_high - window_low;
        if range > 0.0 {
            result[i] = -100.0 * (window_high - close[i]) / range;
        } else {
            // Constant prices (flat consolidation) → neutral value (-50)
            // Williams %R ranges from -100 (oversold) to 0 (overbought)
            // Neutral middle value indicates no clear overbought/oversold signal
            result[i] = -50.0;
        }
    }

    log_indicator_stats(&result, "Williams %R");

    Ok(result)
}

/// Calculate ATR using ta crate
pub fn calculate_atr_ta(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    period: usize,
) -> Result<Vec<f64>> {
    if high.len() != low.len()
        || low.len() != close.len()
        || close.len() != open.len()
        || open.len() != volume.len()
    {
        return Err(VangaError::FeatureError(
            "ATR: mismatched array lengths".to_string(),
        ));
    }

    validate_ohlcv_data(Some(open), Some(high), Some(low), close, Some(volume))?;
    validate_period(period, close.len(), "ATR")?;

    let mut atr = AverageTrueRange::new(period)
        .map_err(|e| VangaError::FeatureError(format!("ATR initialization failed: {}", e)))?;

    let mut result = Vec::with_capacity(high.len());

    for i in 0..high.len() {
        let data_item = DataItem::builder()
            .open(open[i])
            .high(high[i])
            .low(low[i])
            .close(close[i])
            .volume(volume[i])
            .build()
            .map_err(|e| VangaError::FeatureError(format!("ATR DataItem build failed: {}", e)))?;

        let atr_value = atr.next(&data_item);

        // Return NaN for the warm-up period
        if i < period {
            result.push(f64::NAN);
        } else {
            result.push(atr_value);
        }
    }

    log_indicator_stats(&result, "ATR");
    Ok(result)
}

/// Calculate CCI using ta crate
pub fn calculate_cci_ta(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    period: usize,
) -> Result<Vec<f64>> {
    if high.len() != low.len()
        || low.len() != close.len()
        || close.len() != open.len()
        || open.len() != volume.len()
    {
        return Err(VangaError::FeatureError(
            "CCI: mismatched array lengths".to_string(),
        ));
    }

    validate_ohlcv_data(Some(open), Some(high), Some(low), close, Some(volume))?;
    validate_period(period, close.len(), "CCI")?;

    let mut cci = CommodityChannelIndex::new(period)
        .map_err(|e| VangaError::FeatureError(format!("CCI initialization failed: {}", e)))?;

    let mut result = Vec::with_capacity(high.len());

    for i in 0..high.len() {
        let data_item = DataItem::builder()
            .open(open[i])
            .high(high[i])
            .low(low[i])
            .close(close[i])
            .volume(volume[i])
            .build()
            .map_err(|e| VangaError::FeatureError(format!("CCI DataItem build failed: {}", e)))?;

        let cci_value = cci.next(&data_item);

        // Return NaN for the warm-up period
        if i < period - 1 {
            result.push(f64::NAN);
        } else {
            result.push(cci_value);
        }
    }

    log_indicator_stats(&result, "CCI");
    Ok(result)
}
