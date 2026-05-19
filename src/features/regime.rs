//! Regime features: range position and Bollinger/Keltner squeeze.
//!
//! Markets alternate between compression (low volatility, tight range) and
//! expansion (breakouts). Long compressions are where directional moves are
//! "loaded"; the breakout direction is often opposite to what the consolidation
//! visually suggests (Wyckoff spring / upthrust). These features expose that
//! cycle to the model.
//!
//! All outputs are scale-invariant and survive per-sequence normalization:
//! - `regime_position_in_range_<N>` ∈ [0, 1] (fractional position in the N-bar range)
//! - `regime_squeeze_on_<N>` ∈ {0.0, 1.0} (BB inside KC over period N — "TTM squeeze")
//! - `regime_squeeze_duration_<N>` (consecutive bars in squeeze, integer-valued f64)
//! - `regime_range_compression_<N>_<M>` (current N-bar range / longer M-bar range)
//!
//! ## Rationale per target
//!
//! - **volatility**: squeeze flag + duration directly inform the volatility regime
//!   the model is trying to classify. A long squeeze about to release is the
//!   highest-information moment for vol forecasting.
//! - **direction**: `position_in_range` near 0 or 1 signals proximity to S/R,
//!   where directional moves are most likely to resolve.
//! - **price_level**: range-compression ratio quantifies whether the recent
//!   percentile range is contracting (price target tightens) or expanding.

use crate::config::features::RegimeFeaturesConfig;
use crate::features::technical::extract_numeric_column;
use crate::utils::error::Result;
use polars::prelude::*;

const RANGE_EPSILON: f64 = 1e-12;

/// Generate regime features and append them as columns to `df`.
pub async fn generate_regime_features(
    mut df: DataFrame,
    config: &RegimeFeaturesConfig,
) -> Result<DataFrame> {
    if !config.enabled {
        log::debug!("Regime features disabled, skipping");
        return Ok(df);
    }

    log::info!("Generating regime features with config: {:?}", config);

    let high = extract_numeric_column(&df, "high")?;
    let low = extract_numeric_column(&df, "low")?;
    let close = extract_numeric_column(&df, "close")?;

    // Range-position features.
    for &window in &config.range_position_windows {
        let w = window as usize;
        let pos = compute_position_in_range(&high, &low, &close, w);
        df = with_f64_column(df, &format!("regime_position_in_range_{}", window), pos)?;
    }

    // Squeeze features for each configured period.
    for &period in &config.squeeze_periods {
        let p = period as usize;
        let (squeeze_on, squeeze_duration) = compute_squeeze(
            &high,
            &low,
            &close,
            p,
            config.bb_std_dev,
            config.kc_atr_mult,
        );
        df = with_f64_column(df, &format!("regime_squeeze_on_{}", period), squeeze_on)?;
        df = with_f64_column(
            df,
            &format!("regime_squeeze_duration_{}", period),
            squeeze_duration,
        )?;
    }

    // Range compression: short range / long range. Captures whether the recent
    // window is tighter than the broader context.
    if let (Some(short_w), Some(long_w)) = (
        config.range_compression_short,
        config.range_compression_long,
    ) {
        if short_w > 0 && long_w > short_w {
            let comp = compute_range_compression(&high, &low, short_w as usize, long_w as usize);
            df = with_f64_column(
                df,
                &format!("regime_range_compression_{}_{}", short_w, long_w),
                comp,
            )?;
        } else {
            log::warn!(
                "Skipping range compression: invalid windows short={}, long={}",
                short_w,
                long_w
            );
        }
    }

    Ok(df)
}

/// Fractional position of `close` within the trailing `window`-bar range.
/// 0.0 = at the period low, 1.0 = at the period high. Flat windows produce 0.5
/// (no information).
fn compute_position_in_range(high: &[f64], low: &[f64], close: &[f64], window: usize) -> Vec<f64> {
    let n = close.len();
    let mut out = vec![0.5; n];

    if window < 2 || n < window {
        return out;
    }

    for i in (window - 1)..n {
        let slice_h = &high[i + 1 - window..=i];
        let slice_l = &low[i + 1 - window..=i];
        let hi = slice_h.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let lo = slice_l.iter().copied().fold(f64::INFINITY, f64::min);
        let span = hi - lo;
        out[i] = if span > RANGE_EPSILON {
            ((close[i] - lo) / span).clamp(0.0, 1.0)
        } else {
            0.5
        };
    }

    out
}

/// "TTM-style" squeeze: Bollinger Bands contained inside Keltner Channels.
///
/// BB = SMA(close, period) ± std_dev * stdev(close, period)
/// KC = SMA(close, period) ± atr_mult * ATR_proxy(period)
///   where ATR_proxy is the simple mean of (high - low) over `period` bars
///   (avoids the dependency cycle with calculate_atr_ta which needs OHLCV
///   matched against ta crate's stateful builder for every column).
///
/// Returns two vectors:
///   `squeeze_on[i]`        = 1.0 when both BB bands fall inside KC, else 0.0
///   `squeeze_duration[i]`  = consecutive count of `squeeze_on == 1.0` up to and
///                            including bar i (reset to 0 on first non-squeeze bar)
///
/// Warmup bars (< period - 1) emit zeros.
fn compute_squeeze(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    period: usize,
    bb_std_dev: f64,
    kc_atr_mult: f64,
) -> (Vec<f64>, Vec<f64>) {
    let n = close.len();
    let mut on = vec![0.0; n];
    let mut duration = vec![0.0; n];

    if period < 2 || n < period {
        return (on, duration);
    }

    let period_f = period as f64;
    let mut run = 0.0;

    for i in (period - 1)..n {
        let c_slice = &close[i + 1 - period..=i];
        let mean = c_slice.iter().sum::<f64>() / period_f;
        let var = c_slice.iter().map(|c| (c - mean).powi(2)).sum::<f64>() / period_f;
        let std = var.sqrt();

        // Mean true range as a stateless ATR proxy aligned to the same window.
        let h_slice = &high[i + 1 - period..=i];
        let l_slice = &low[i + 1 - period..=i];
        let mut tr_sum = 0.0;
        for j in 0..period {
            let h = h_slice[j];
            let l = l_slice[j];
            tr_sum += h - l;
        }
        let atr_proxy = tr_sum / period_f;

        let bb_upper = mean + bb_std_dev * std;
        let bb_lower = mean - bb_std_dev * std;
        let kc_upper = mean + kc_atr_mult * atr_proxy;
        let kc_lower = mean - kc_atr_mult * atr_proxy;

        let in_squeeze = bb_upper <= kc_upper && bb_lower >= kc_lower;
        if in_squeeze {
            on[i] = 1.0;
            run += 1.0;
        } else {
            run = 0.0;
        }
        duration[i] = run;
    }

    (on, duration)
}

/// Range compression: ratio of the `short` window's range over the `long`
/// window's range, both computed from high/low extremes. Values below 1.0
/// indicate the recent range is tighter than the broader context (compression);
/// values above 1.0 indicate expansion. Long-window warmup bars emit 1.0
/// (neutral / no information).
fn compute_range_compression(high: &[f64], low: &[f64], short: usize, long: usize) -> Vec<f64> {
    let n = high.len();
    let mut out = vec![1.0; n];

    if short < 2 || long <= short || n < long {
        return out;
    }

    for i in (long - 1)..n {
        let short_h = high[i + 1 - short..=i]
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        let short_l = low[i + 1 - short..=i]
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min);
        let long_h = high[i + 1 - long..=i]
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        let long_l = low[i + 1 - long..=i]
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min);

        let short_range = (short_h - short_l).max(RANGE_EPSILON);
        let long_range = (long_h - long_l).max(RANGE_EPSILON);
        out[i] = short_range / long_range;
    }

    out
}

fn with_f64_column(mut df: DataFrame, name: &str, values: Vec<f64>) -> Result<DataFrame> {
    df = df
        .with_column(Series::new(name.into(), values).into_column())
        .map_err(|e| {
            crate::utils::error::VangaError::DataError(format!(
                "Failed to add {} column: {}",
                name, e
            ))
        })?
        .clone();
    Ok(df)
}
