// Feature engineering
use crate::config::features::FeatureEngineeringConfig;
use crate::utils::error::Result;
use polars::prelude::*;

pub async fn apply_feature_engineering(
    mut df: DataFrame,
    config: &FeatureEngineeringConfig,
) -> Result<DataFrame> {
    log::info!("Applying feature engineering with config: {:?}", config);

    // Extract price data
    let close_prices = crate::features::technical::extract_numeric_column(&df, "close")?;
    let volume = crate::features::technical::extract_numeric_column(&df, "volume")?;
    let high_prices = crate::features::technical::extract_numeric_column(&df, "high")?;
    let low_prices = crate::features::technical::extract_numeric_column(&df, "low")?;

    // Generate interaction features
    let price_volume_ratio: Vec<f64> = close_prices
        .iter()
        .zip(volume.iter())
        .map(|(p, v)| if *v > 0.0 { p / v } else { 0.0 })
        .collect();

    // Generate polynomial features (price squared, cubed)
    let price_squared: Vec<f64> = close_prices.iter().map(|p| p.powi(2)).collect();
    let price_log: Vec<f64> = close_prices
        .iter()
        .map(|p| if *p > 0.0 { p.ln() } else { 0.0 })
        .collect();

    // Generate lag features
    let price_lag_1 = create_lag_feature(&close_prices, 1);
    let price_lag_5 = create_lag_feature(&close_prices, 5);
    let volume_lag_1 = create_lag_feature(&volume, 1);

    // Generate rolling statistics
    let price_rolling_mean = calculate_rolling_mean(&close_prices, 10);
    let price_rolling_std = calculate_rolling_std(&close_prices, 10);

    // Generate relative features
    let high_low_ratio: Vec<f64> = high_prices
        .iter()
        .zip(low_prices.iter())
        .map(|(h, l)| if *l > 0.0 { h / l } else { 1.0 })
        .collect();

    // Add engineered features to DataFrame one by one
    df = df
        .with_column(Series::new("price_volume_ratio", price_volume_ratio))
        .map_err(|e| {
            crate::utils::error::VangaError::DataError(format!(
                "Failed to add price_volume_ratio column: {}",
                e
            ))
        })?
        .clone();
    df = df
        .with_column(Series::new("price_squared", price_squared))
        .map_err(|e| {
            crate::utils::error::VangaError::DataError(format!(
                "Failed to add price_squared column: {}",
                e
            ))
        })?
        .clone();
    df = df
        .with_column(Series::new("price_log", price_log))
        .map_err(|e| {
            crate::utils::error::VangaError::DataError(format!(
                "Failed to add price_log column: {}",
                e
            ))
        })?
        .clone();
    df = df
        .with_column(Series::new("price_lag_1", price_lag_1))
        .map_err(|e| {
            crate::utils::error::VangaError::DataError(format!(
                "Failed to add price_lag_1 column: {}",
                e
            ))
        })?
        .clone();
    df = df
        .with_column(Series::new("price_lag_5", price_lag_5))
        .map_err(|e| {
            crate::utils::error::VangaError::DataError(format!(
                "Failed to add price_lag_5 column: {}",
                e
            ))
        })?
        .clone();
    df = df
        .with_column(Series::new("volume_lag_1", volume_lag_1))
        .map_err(|e| {
            crate::utils::error::VangaError::DataError(format!(
                "Failed to add volume_lag_1 column: {}",
                e
            ))
        })?
        .clone();
    df = df
        .with_column(Series::new("price_rolling_mean_10", price_rolling_mean))
        .map_err(|e| {
            crate::utils::error::VangaError::DataError(format!(
                "Failed to add price_rolling_mean_10 column: {}",
                e
            ))
        })?
        .clone();
    df = df
        .with_column(Series::new("price_rolling_std_10", price_rolling_std))
        .map_err(|e| {
            crate::utils::error::VangaError::DataError(format!(
                "Failed to add price_rolling_std_10 column: {}",
                e
            ))
        })?
        .clone();
    df = df
        .with_column(Series::new("high_low_ratio", high_low_ratio))
        .map_err(|e| {
            crate::utils::error::VangaError::DataError(format!(
                "Failed to add high_low_ratio column: {}",
                e
            ))
        })?
        .clone();

    Ok(df)
}

/// Create lag feature
fn create_lag_feature(data: &[f64], lag: usize) -> Vec<f64> {
    let mut lagged = vec![0.0; lag];
    lagged.extend_from_slice(&data[..data.len().saturating_sub(lag)]);
    lagged
}

/// Calculate rolling mean
fn calculate_rolling_mean(data: &[f64], window: usize) -> Vec<f64> {
    let mut rolling_mean = vec![0.0; data.len()];

    for i in window..data.len() {
        let sum: f64 = data[i - window..i].iter().sum();
        rolling_mean[i] = sum / window as f64;
    }

    rolling_mean
}

/// Calculate rolling standard deviation
fn calculate_rolling_std(data: &[f64], window: usize) -> Vec<f64> {
    let mut rolling_std = vec![0.0; data.len()];

    for i in window..data.len() {
        let window_data = &data[i - window..i];
        let mean = window_data.iter().sum::<f64>() / window as f64;
        let variance = window_data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / window as f64;
        rolling_std[i] = variance.sqrt();
    }

    rolling_std
}
