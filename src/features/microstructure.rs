// Market microstructure features
use crate::config::features::MarketMicrostructureConfig;
use crate::utils::error::Result;
use polars::prelude::*;

pub async fn generate_microstructure_features(
    mut df: DataFrame,
    config: &MarketMicrostructureConfig,
) -> Result<DataFrame> {
    log::info!(
        "Generating market microstructure features with config: {:?}",
        config
    );

    // Extract OHLCV data
    let close_prices = crate::features::technical::extract_numeric_column(&df, "close")?;
    let volume = crate::features::technical::extract_numeric_column(&df, "volume")?;
    let high_prices = crate::features::technical::extract_numeric_column(&df, "high")?;
    let low_prices = crate::features::technical::extract_numeric_column(&df, "low")?;

    // Calculate bid-ask spread proxy (high-low spread)
    let spread: Vec<f64> = high_prices
        .iter()
        .zip(low_prices.iter())
        .map(|(h, l)| (h - l) / ((h + l) / 2.0))
        .collect();

    // Calculate volume-weighted spread
    let vw_spread: Vec<f64> = spread
        .iter()
        .zip(volume.iter())
        .map(|(s, v)| s * v)
        .collect();

    // Calculate price impact (simplified)
    let price_impact: Vec<f64> = close_prices
        .windows(2)
        .zip(volume.iter().skip(1))
        .map(|(prices, vol)| {
            if *vol > 0.0 {
                (prices[1] - prices[0]).abs() / vol.sqrt()
            } else {
                0.0
            }
        })
        .chain(std::iter::once(0.0))
        .collect();

    // Add microstructure features to DataFrame one by one
    df = df
        .with_column(Series::new("spread".into(), spread))
        .map_err(|e| {
            crate::utils::error::VangaError::DataError(format!(
                "Failed to add spread column: {}",
                e
            ))
        })?
        .clone();
    df = df
        .with_column(Series::new("volume_weighted_spread".into(), vw_spread))
        .map_err(|e| {
            crate::utils::error::VangaError::DataError(format!(
                "Failed to add volume_weighted_spread column: {}",
                e
            ))
        })?
        .clone();
    df = df
        .with_column(Series::new("price_impact".into(), price_impact))
        .map_err(|e| {
            crate::utils::error::VangaError::DataError(format!(
                "Failed to add price_impact column: {}",
                e
            ))
        })?
        .clone();

    Ok(df)
}
