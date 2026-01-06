// Custom features processing
use crate::config::features::CustomFeaturesConfig;
use crate::utils::error::Result;
use polars::prelude::*;

pub async fn process_custom_features(
    mut df: DataFrame,
    config: &CustomFeaturesConfig,
) -> Result<DataFrame> {
    log::info!("Processing custom features with config: {:?}", config);

    // Extract price data
    let close_prices = crate::features::technical::extract_numeric_column(&df, "close")?;
    let volume = crate::features::technical::extract_numeric_column(&df, "volume")?;

    // Calculate custom momentum features
    let momentum_3 = calculate_momentum(&close_prices, 3);
    let momentum_7 = calculate_momentum(&close_prices, 7);
    let momentum_14 = calculate_momentum(&close_prices, 14);

    // Calculate volume-price trend
    let vpt = calculate_volume_price_trend(&close_prices, &volume);

    // Calculate price acceleration
    let price_acceleration = calculate_price_acceleration(&close_prices, 5);

    // Calculate volume momentum
    let volume_momentum = calculate_momentum(&volume, 5);

    // Add custom features to DataFrame one by one
    df = df
        .with_column(Series::new("momentum_3".into(), momentum_3).into_column())
        .map_err(|e| {
            crate::utils::error::VangaError::DataError(format!(
                "Failed to add momentum_3 column: {}",
                e
            ))
        })?
        .clone();
    df = df
        .with_column(Series::new("momentum_7".into(), momentum_7).into_column())
        .map_err(|e| {
            crate::utils::error::VangaError::DataError(format!(
                "Failed to add momentum_7 column: {}",
                e
            ))
        })?
        .clone();
    df = df
        .with_column(Series::new("momentum_14".into(), momentum_14).into_column())
        .map_err(|e| {
            crate::utils::error::VangaError::DataError(format!(
                "Failed to add momentum_14 column: {}",
                e
            ))
        })?
        .clone();
    df = df
        .with_column(Series::new("volume_price_trend".into(), vpt).into_column())
        .map_err(|e| {
            crate::utils::error::VangaError::DataError(format!(
                "Failed to add volume_price_trend column: {}",
                e
            ))
        })?
        .clone();
    df = df
        .with_column(Series::new("price_acceleration".into(), price_acceleration).into_column())
        .map_err(|e| {
            crate::utils::error::VangaError::DataError(format!(
                "Failed to add price_acceleration column: {}",
                e
            ))
        })?
        .clone();
    df = df
        .with_column(Series::new("volume_momentum".into(), volume_momentum).into_column())
        .map_err(|e| {
            crate::utils::error::VangaError::DataError(format!(
                "Failed to add volume_momentum column: {}",
                e
            ))
        })?
        .clone();

    Ok(df)
}

/// Calculate momentum over a period
fn calculate_momentum(prices: &[f64], period: usize) -> Vec<f64> {
    let mut momentum = vec![0.0; prices.len()];

    for i in period..prices.len() {
        if prices[i - period] != 0.0 {
            momentum[i] = (prices[i] - prices[i - period]) / prices[i - period];
        }
    }

    momentum
}

/// Calculate Volume Price Trend
fn calculate_volume_price_trend(prices: &[f64], volume: &[f64]) -> Vec<f64> {
    let mut vpt = vec![0.0; prices.len()];

    for i in 1..prices.len() {
        if prices[i - 1] != 0.0 {
            let price_change = (prices[i] - prices[i - 1]) / prices[i - 1];
            vpt[i] = vpt[i - 1] + volume[i] * price_change;
        }
    }

    vpt
}

/// Calculate price acceleration (second derivative)
fn calculate_price_acceleration(prices: &[f64], period: usize) -> Vec<f64> {
    let mut acceleration = vec![0.0; prices.len()];

    for i in (period * 2)..prices.len() {
        if prices[i - period] != 0.0 && prices[i - period * 2] != 0.0 {
            let current_momentum = (prices[i] - prices[i - period]) / prices[i - period];
            let previous_momentum =
                (prices[i - period] - prices[i - period * 2]) / prices[i - period * 2];
            acceleration[i] = current_momentum - previous_momentum;
        }
    }

    acceleration
}
