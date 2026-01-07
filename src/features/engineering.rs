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

    // Generate interaction features if enabled
    if config.interaction_features.enabled {
        let price_volume_ratio: Vec<f64> = close_prices
            .iter()
            .zip(volume.iter())
            .map(|(p, v)| if *v > 0.0 { p / v } else { 0.0 })
            .collect();

        df = df
            .with_column(Series::new("price_volume_ratio".into(), price_volume_ratio).into_column())
            .map_err(|e| {
                crate::utils::error::VangaError::DataError(format!(
                    "Failed to add price_volume_ratio column: {}",
                    e
                ))
            })?
            .clone();
    }

    // Generate polynomial features if enabled
    if config.polynomial_features.enabled {
        let price_squared: Vec<f64> = close_prices.iter().map(|p| p.powi(2)).collect();
        let price_log: Vec<f64> = close_prices
            .iter()
            .map(|p| if *p > 0.0 { p.ln() } else { 0.0 })
            .collect();

        df = df
            .with_column(Series::new("price_squared".into(), price_squared).into_column())
            .map_err(|e| {
                crate::utils::error::VangaError::DataError(format!(
                    "Failed to add price_squared column: {}",
                    e
                ))
            })?
            .clone();

        df = df
            .with_column(Series::new("price_log".into(), price_log).into_column())
            .map_err(|e| {
                crate::utils::error::VangaError::DataError(format!(
                    "Failed to add price_log column: {}",
                    e
                ))
            })?
            .clone();
    }

    // Generate lag features based on configuration
    if config.lag_features.enabled {
        for feature_name in &config.lag_features.features_to_lag {
            // Check if the column exists in the DataFrame
            if let Ok(column_data) =
                crate::features::technical::extract_numeric_column(&df, feature_name)
            {
                for &lag_period in &config.lag_features.lag_periods {
                    let lag_feature = create_lag_feature(&column_data, lag_period as usize);
                    let lag_column_name = format!("{}_lag_{}", feature_name, lag_period);
                    let series =
                        Series::new(lag_column_name.clone().into(), lag_feature).into_column();

                    df = df
                        .with_column(series)
                        .map_err(|e| {
                            crate::utils::error::VangaError::DataError(format!(
                                "Failed to add {} column: {}",
                                lag_column_name, e
                            ))
                        })?
                        .clone();
                }
            } else {
                log::warn!(
                    "Column '{}' not found for lag feature generation, skipping",
                    feature_name
                );
            }
        }
    }

    // Generate rolling statistics if enabled
    if config.rolling_features.enabled {
        for &window_size in &config.rolling_features.window_sizes {
            let price_rolling_mean = calculate_rolling_mean(&close_prices, window_size as usize);
            let price_rolling_std = calculate_rolling_std(&close_prices, window_size as usize);

            df = df
                .with_column(
                    Series::new(
                        format!("price_rolling_mean_{}", window_size).into(),
                        price_rolling_mean,
                    )
                    .into_column(),
                )
                .map_err(|e| {
                    crate::utils::error::VangaError::DataError(format!(
                        "Failed to add price_rolling_mean_{} column: {}",
                        window_size, e
                    ))
                })?
                .clone();

            df = df
                .with_column(
                    Series::new(
                        format!("price_rolling_std_{}", window_size).into(),
                        price_rolling_std,
                    )
                    .into_column(),
                )
                .map_err(|e| {
                    crate::utils::error::VangaError::DataError(format!(
                        "Failed to add price_rolling_std_{} column: {}",
                        window_size, e
                    ))
                })?
                .clone();
        }
    }

    // Generate relative features
    let high_low_ratio: Vec<f64> = high_prices
        .iter()
        .zip(low_prices.iter())
        .map(|(h, l)| if *l > 0.0 { h / l } else { 1.0 })
        .collect();

    df = df
        .with_column(Series::new("high_low_ratio".into(), high_low_ratio).into_column())
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
    let mut lagged = vec![f64::NAN; lag]; // Use NaN for lag padding, not 0.0
    lagged.extend_from_slice(&data[..data.len().saturating_sub(lag)]);
    lagged
}

/// Calculate rolling mean
fn calculate_rolling_mean(data: &[f64], window: usize) -> Vec<f64> {
    let mut rolling_mean = vec![f64::NAN; data.len()]; // Use NaN for initial values

    for i in window..data.len() {
        let sum: f64 = data[i - window..i].iter().sum();
        rolling_mean[i] = sum / window as f64;
    }

    rolling_mean
}

/// Calculate rolling standard deviation
fn calculate_rolling_std(data: &[f64], window: usize) -> Vec<f64> {
    let mut rolling_std = vec![f64::NAN; data.len()]; // Use NaN for initial values

    for i in window..data.len() {
        let window_data = &data[i - window..i];
        let mean = window_data.iter().sum::<f64>() / window as f64;
        let variance = window_data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / window as f64;
        rolling_std[i] = variance.sqrt();
    }

    rolling_std
}
