//! Liquidity-aware features: stop-hunt detection, wick anatomy, and order-flow proxy.
//!
//! These features capture how markets work *against* normal patterns — taking out
//! obvious stops at prior highs/lows before reversing, rejecting at S/R with long
//! wicks, and showing buy/sell pressure imbalances that diverge from price action.
//!
//! All features are scale-invariant (ratios, normalized z-scores, ATR-normalized
//! distances, binary flags) so they survive the per-sequence normalization used
//! in VANGA's pipeline. None depend on raw price magnitude, keeping the system
//! symbol-agnostic.
//!
//! ## Output columns
//!
//! For each `lookback` in `sweep_lookbacks`:
//! - `liq_sweep_up_<N>`         : 1.0 when `high` broke the prior N-bar high but `close` failed back inside (bull-trap / stop hunt above resistance)
//! - `liq_sweep_down_<N>`       : 1.0 when `low` broke the prior N-bar low but `close` failed back inside (bear-trap / stop hunt below support)
//! - `liq_sweep_strength_<N>`   : how far the sweep penetrated past the prior extreme, normalized by ATR. Signed: positive for up-sweep, negative for down-sweep
//!
//! Always (per-candle, no warmup):
//! - `liq_upper_wick_ratio`     : `(high - max(open, close)) / max(high - low, ε)` ∈ [0, 1]
//! - `liq_lower_wick_ratio`     : `(min(open, close) - low) / max(high - low, ε)` ∈ [0, 1]
//! - `liq_body_ratio`           : `|close - open| / max(high - low, ε)` ∈ [0, 1]
//! - `liq_wick_asymmetry`       : `upper - lower` ∈ [-1, 1] (positive = top rejection, negative = bottom rejection)
//!
//! Cumulative Volume Delta (per `cvd_windows`):
//! - `liq_cvd_slope_<W>`        : rolling-window z-score of CVD slope (order-flow momentum)
//! - `liq_cvd_price_div_<W>`    : signed divergence between rolling price change and rolling CVD change
//!
//! ## Rationale per target
//!
//! - **stop_level**: sweep flags + wick ratios directly inform where adverse moves cluster.
//!   A fresh sweep is the canonical "worst-adverse" moment, exactly what stop_level labels.
//! - **direction**: CVD slope + CVD/price divergence are leading indicators of trend
//!   exhaustion and continuation — orthogonal to MA-based momentum.
//! - **price_level**: sweep strength quantifies overshoots beyond the recent range,
//!   which is the same boundary structure price_level classifies against.

use crate::config::features::LiquidityFeaturesConfig;
use crate::features::ta_helpers::calculate_atr_ta;
use crate::features::technical::extract_numeric_column;
use crate::utils::error::Result;
use polars::prelude::*;

/// Small epsilon to keep division stable on zero-range candles.
const RANGE_EPSILON: f64 = 1e-12;

/// Generate liquidity-aware features and append them as columns to `df`.
pub async fn generate_liquidity_features(
    mut df: DataFrame,
    config: &LiquidityFeaturesConfig,
) -> Result<DataFrame> {
    if !config.enabled {
        log::debug!("Liquidity features disabled, skipping");
        return Ok(df);
    }

    log::info!("Generating liquidity features with config: {:?}", config);

    let open = extract_numeric_column(&df, "open")?;
    let high = extract_numeric_column(&df, "high")?;
    let low = extract_numeric_column(&df, "low")?;
    let close = extract_numeric_column(&df, "close")?;
    let volume = extract_numeric_column(&df, "volume")?;

    // Wick anatomy (no warmup, per-candle).
    let (upper_wick, lower_wick, body, asymmetry) = compute_wick_ratios(&open, &high, &low, &close);

    df = with_f64_column(df, "liq_upper_wick_ratio", upper_wick)?;
    df = with_f64_column(df, "liq_lower_wick_ratio", lower_wick)?;
    df = with_f64_column(df, "liq_body_ratio", body)?;
    df = with_f64_column(df, "liq_wick_asymmetry", asymmetry)?;

    // Liquidity sweeps need ATR for the strength normalization. Compute ATR once
    // at the configured period and reuse for every lookback.
    let atr = calculate_atr_ta(&open, &high, &low, &close, &volume, config.atr_period)?;

    for &lookback in &config.sweep_lookbacks {
        let lb = lookback as usize;
        let (sweep_up, sweep_down, strength) =
            compute_liquidity_sweeps(&high, &low, &close, &atr, lb);
        df = with_f64_column(df, &format!("liq_sweep_up_{}", lookback), sweep_up)?;
        df = with_f64_column(df, &format!("liq_sweep_down_{}", lookback), sweep_down)?;
        df = with_f64_column(df, &format!("liq_sweep_strength_{}", lookback), strength)?;
    }

    // CVD slope and CVD/price divergence per configured window.
    let cvd = compute_cumulative_volume_delta(&high, &low, &close, &volume);

    for &window in &config.cvd_windows {
        let w = window as usize;
        let cvd_slope = compute_normalized_rolling_slope(&cvd, w);
        let cvd_price_div = compute_rolling_divergence(&close, &cvd, w);
        df = with_f64_column(df, &format!("liq_cvd_slope_{}", window), cvd_slope)?;
        df = with_f64_column(df, &format!("liq_cvd_price_div_{}", window), cvd_price_div)?;
    }

    Ok(df)
}

/// Decompose each candle into upper-wick / lower-wick / body ratios and wick asymmetry.
/// All four returned vectors have length `open.len()` and values in `[0, 1]` (except
/// `asymmetry` which is in `[-1, 1]`). For zero-range candles (high == low) the
/// epsilon guard makes every ratio equal to zero — a defensible "no signal" value.
fn compute_wick_ratios(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let n = open.len();
    let mut upper = Vec::with_capacity(n);
    let mut lower = Vec::with_capacity(n);
    let mut body = Vec::with_capacity(n);
    let mut asym = Vec::with_capacity(n);

    for i in 0..n {
        let h = high[i];
        let l = low[i];
        let o = open[i];
        let c = close[i];
        let range = (h - l).max(RANGE_EPSILON);

        let body_top = if o > c { o } else { c };
        let body_bottom = if o < c { o } else { c };

        let upper_wick = ((h - body_top) / range).clamp(0.0, 1.0);
        let lower_wick = ((body_bottom - l) / range).clamp(0.0, 1.0);
        let body_ratio = ((c - o).abs() / range).clamp(0.0, 1.0);

        upper.push(upper_wick);
        lower.push(lower_wick);
        body.push(body_ratio);
        asym.push(upper_wick - lower_wick);
    }

    (upper, lower, body, asym)
}

/// Detect liquidity sweeps against the prior `lookback`-bar high/low.
///
/// A bullish-side sweep (`sweep_up = 1.0`) fires when the current `high` exceeds
/// the maximum high over the previous `lookback` bars but the `close` settles
/// back below that prior high — a classic stop-hunt above resistance that failed.
/// `sweep_down` is the symmetric case below support.
///
/// `sweep_strength` is the signed penetration distance past the swept boundary,
/// normalized by ATR so it's symbol- and regime-agnostic:
///   `strength = (high - prior_high) / atr` for up-sweeps (positive)
///   `strength = -(prior_low - low) / atr` for down-sweeps (negative)
///   `strength = 0.0` when no sweep fired
///
/// Warmup: first `lookback` candles are zero (no prior window to compare). NaN
/// values from the ATR helper during its own warmup also produce zero strength.
fn compute_liquidity_sweeps(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    atr: &[f64],
    lookback: usize,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let n = high.len();
    let mut sweep_up = vec![0.0; n];
    let mut sweep_down = vec![0.0; n];
    let mut strength = vec![0.0; n];

    if lookback == 0 || n <= lookback {
        return (sweep_up, sweep_down, strength);
    }

    for i in lookback..n {
        let window_start = i - lookback;
        let prior_high = high[window_start..i]
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        let prior_low = low[window_start..i]
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min);

        // ATR is the normalizer; zero or NaN ATR means no strength can be computed.
        let atr_i = atr[i];
        let atr_usable = atr_i.is_finite() && atr_i > 0.0;

        if high[i] > prior_high && close[i] < prior_high {
            sweep_up[i] = 1.0;
            if atr_usable {
                strength[i] = (high[i] - prior_high) / atr_i;
            }
        }

        if low[i] < prior_low && close[i] > prior_low {
            sweep_down[i] = 1.0;
            if atr_usable {
                // Signed negative so the model sees up-sweeps and down-sweeps
                // as distinct directions in a single column.
                strength[i] = -((prior_low - low[i]) / atr_i);
            }
        }
    }

    (sweep_up, sweep_down, strength)
}

/// Compute a per-candle Cumulative Volume Delta (CVD) proxy from OHLCV.
///
/// Without trade-by-trade data we approximate the buy/sell split using the
/// candle's close position within its range: closes near the high imply
/// aggressive buying, closes near the low imply aggressive selling. This is
/// the standard "candle balance" estimator and is the closest OHLCV gets to
/// true order flow.
///
///   buy_share  = (close - low) / range
///   sell_share = (high - close) / range
///   delta      = (buy_share - sell_share) * volume
///   CVD[i]     = CVD[i-1] + delta[i]
fn compute_cumulative_volume_delta(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
) -> Vec<f64> {
    let n = close.len();
    let mut cvd = Vec::with_capacity(n);
    let mut running = 0.0;

    for i in 0..n {
        let range = (high[i] - low[i]).max(RANGE_EPSILON);
        let buy_share = ((close[i] - low[i]) / range).clamp(0.0, 1.0);
        let sell_share = ((high[i] - close[i]) / range).clamp(0.0, 1.0);
        let delta = (buy_share - sell_share) * volume[i];
        running += delta;
        cvd.push(running);
    }

    cvd
}

/// Compute a rolling-window normalized slope of `series`.
///
/// For each index i ≥ window, we fit a least-squares slope over the past
/// `window` samples, then divide by the window's standard deviation to make
/// the result scale-free. The output is unitless and centered near zero, so
/// it composes well with per-sequence z-score normalization.
///
/// NaN-safe: if the window has zero variance (flat series) the slope is set
/// to zero rather than infinity.
fn compute_normalized_rolling_slope(series: &[f64], window: usize) -> Vec<f64> {
    let n = series.len();
    let mut out = vec![0.0; n];

    if window < 2 || n < window {
        return out;
    }

    let window_f = window as f64;
    let x_mean = (window_f - 1.0) / 2.0;
    // Σ (x - x_mean)² for x = 0..window-1. Constant across windows.
    let x_var_sum: f64 = (0..window)
        .map(|j| {
            let dx = j as f64 - x_mean;
            dx * dx
        })
        .sum();

    if x_var_sum <= 0.0 {
        return out;
    }

    for i in (window - 1)..n {
        let slice = &series[i + 1 - window..=i];
        let y_mean = slice.iter().sum::<f64>() / window_f;

        let mut cov = 0.0;
        let mut y_var = 0.0;
        for (j, &y) in slice.iter().enumerate() {
            let dx = j as f64 - x_mean;
            let dy = y - y_mean;
            cov += dx * dy;
            y_var += dy * dy;
        }

        let slope = cov / x_var_sum;
        let y_std = (y_var / window_f).sqrt();

        out[i] = if y_std > RANGE_EPSILON {
            slope / y_std
        } else {
            0.0
        };
    }

    out
}

/// Compute rolling divergence between two series: the difference between their
/// per-window standardized changes. Positive values mean `a` rose more than `b`
/// (relative to their own intra-window volatility); negative means the inverse.
/// Used as `compute_rolling_divergence(price, cvd, window)` to surface classic
/// "price up, order flow down" exhaustion patterns.
///
/// Each change is normalized by the window's standard deviation, which makes
/// the metric scale-invariant *and* sensitive to small absolute moves on
/// large-magnitude series (like price). When the window is flat (std ≈ 0)
/// the contribution from that series is zero rather than infinity.
fn compute_rolling_divergence(a: &[f64], b: &[f64], window: usize) -> Vec<f64> {
    let n = a.len();
    let mut out = vec![0.0; n];

    if window < 2 || n < window || a.len() != b.len() {
        return out;
    }

    let window_f = window as f64;

    for i in (window - 1)..n {
        let a_now = a[i];
        let a_then = a[i + 1 - window];
        let b_now = b[i];
        let b_then = b[i + 1 - window];

        let a_slice = &a[i + 1 - window..=i];
        let b_slice = &b[i + 1 - window..=i];

        let a_mean = a_slice.iter().sum::<f64>() / window_f;
        let b_mean = b_slice.iter().sum::<f64>() / window_f;

        let a_var = a_slice.iter().map(|v| (v - a_mean).powi(2)).sum::<f64>() / window_f;
        let b_var = b_slice.iter().map(|v| (v - b_mean).powi(2)).sum::<f64>() / window_f;

        let a_std = a_var.sqrt();
        let b_std = b_var.sqrt();

        let a_change = if a_std > RANGE_EPSILON {
            (a_now - a_then) / a_std
        } else {
            0.0
        };
        let b_change = if b_std > RANGE_EPSILON {
            (b_now - b_then) / b_std
        } else {
            0.0
        };

        out[i] = a_change - b_change;
    }

    out
}

/// Append an f64 column to the DataFrame, returning an owned DataFrame.
/// Mirrors the with_column-and-clone pattern used elsewhere in this module so
/// failure paths surface as proper VangaError::DataError values.
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
