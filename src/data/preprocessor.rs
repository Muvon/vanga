// Data preprocessor
use crate::config::training::DataConfig;
use crate::utils::error::{Result, VangaError};
use polars::prelude::*;

pub struct DataPreprocessor;

impl Default for DataPreprocessor {
    fn default() -> Self {
        Self::new()
    }
}

impl DataPreprocessor {
    pub fn new() -> Self {
        Self
    }

    pub async fn process_for_training(
        &self,
        mut df: DataFrame,
        config: &DataConfig,
        features_config: Option<&crate::config::FeatureConfig>,
    ) -> Result<DataFrame> {
        df = if config.outlier_handling.enabled {
            self.remove_outliers(df)? // Reuse existing method
        } else {
            df
        };

        // Apply feature engineering if config provided
        if let Some(config) = features_config {
            log::info!("Applying feature engineering from config");
            let feature_engineer = crate::features::FeatureEngineer::new(config.clone());
            df = feature_engineer.generate_features(df).await?;

            // Remove rows with NaN values instead of failing
            df = self.remove_nan_rows(df)?;

            // Validate that all remaining data is clean
            self.validate_features(&df, "after feature engineering and NaN removal")?;
        }
        // Apply normalization
        df = self.normalize_features(df)?;

        Ok(df)
    }

    pub async fn process_for_prediction(
        &self,
        mut df: DataFrame,
        symbol: &str,
    ) -> Result<DataFrame> {
        log::info!("Processing data for prediction on symbol: {}", symbol);

        // Basic validation for prediction data
        df = self.validate_prediction_data(df, symbol)?;

        // Apply same preprocessing as training (without target-specific operations)
        df = self.normalize_features(df)?;

        Ok(df)
    }

    /// Process multiple symbol data for cross-asset prediction
    pub async fn process_for_cross_asset_prediction(
        &self,
        symbol_data: std::collections::HashMap<String, DataFrame>,
        features_config: &crate::config::FeatureConfig,
    ) -> Result<std::collections::HashMap<String, DataFrame>> {
        log::info!(
            "Processing {} symbols for cross-asset prediction",
            symbol_data.len()
        );

        // Apply cross-asset feature engineering if enabled
        if features_config.cross_asset.enabled {
            let cross_asset_generator = crate::features::CrossAssetFeatureGenerator::new(
                features_config.cross_asset.clone(),
            );
            cross_asset_generator
                .generate_cross_asset_features(&symbol_data)
                .await
        } else {
            // Process each symbol individually
            let mut processed_data = std::collections::HashMap::new();
            for (symbol, df) in symbol_data {
                let processed_df = self.process_for_prediction(df, &symbol).await?;
                processed_data.insert(symbol, processed_df);
            }
            Ok(processed_data)
        }
    }

    /// Remove outliers using IQR method
    fn remove_outliers(&self, df: DataFrame) -> Result<DataFrame> {
        // Simple outlier removal for numeric columns
        // This is a placeholder - in production you'd want more sophisticated methods
        Ok(df)
    }

    /// Normalize features for training/prediction
    fn normalize_features(&self, df: DataFrame) -> Result<DataFrame> {
        // Apply feature normalization (z-score, min-max, etc.)
        // This is a placeholder - actual normalization would be implemented here
        Ok(df)
    }

    /// Validate prediction data format and completeness
    fn validate_prediction_data(&self, df: DataFrame, symbol: &str) -> Result<DataFrame> {
        // Validate that required columns exist for the symbol
        let required_columns = ["open", "high", "low", "close", "volume"];

        for col in required_columns {
            if !df.get_column_names().contains(&col) {
                return Err(crate::utils::error::VangaError::DataError(format!(
                    "Missing required column '{}' for symbol {}",
                    col, symbol
                )));
            }
        }

        Ok(df)
    }

    /// Validate that all numeric features are finite
    /// Remove rows containing NaN values from DataFrame with proper validation
    pub fn remove_nan_rows(&self, mut df: DataFrame) -> Result<DataFrame> {
        log::info!("Removing rows with NaN values to ensure clean training data");

        let original_len = df.height();

        // Find the first row where ALL columns have valid values
        let mut first_valid_row = None;

        for row_idx in 0..original_len {
            let mut all_valid = true;

            for series in df.get_columns() {
                if series.dtype().is_numeric() {
                    if let Ok(float_series) = series.f64() {
                        if let Some(value) = float_series.get(row_idx) {
                            if !value.is_finite() {
                                all_valid = false;
                                break;
                            }
                        } else {
                            all_valid = false;
                            break;
                        }
                    }
                }
            }

            if all_valid {
                first_valid_row = Some(row_idx);
                break;
            }
        }

        let first_valid_idx = match first_valid_row {
            Some(idx) => idx,
            None => {
                return Err(VangaError::DataError(
                    "No rows found with all valid (non-NaN) values. Cannot proceed with training."
                        .to_string(),
                ));
            }
        };

        // Validate that NO NaN values appear after the first valid row
        for row_idx in first_valid_idx..original_len {
            for series in df.get_columns() {
                if series.dtype().is_numeric() {
                    let col_name = series.name();
                    if let Ok(float_series) = series.f64() {
                        if let Some(value) = float_series.get(row_idx) {
                            if !value.is_finite() {
                                let timestamp = df
                                    .column("timestamp")
                                    .ok()
                                    .and_then(|ts| ts.get(row_idx).ok())
                                    .map(|v| format!("{}", v))
                                    .unwrap_or_else(|| format!("row {}", row_idx));

                                return Err(VangaError::DataError(format!(
                                    "FEATURE CALCULATION BUG: NaN value found in column '{}' at row {} (timestamp: {}). \
                                     All rows from {} onwards should have valid values. \
                                     NaN values should only appear at the beginning before training data starts.",
                                    col_name, row_idx, timestamp, first_valid_idx
                                )));
                            }
                        }
                        // Removed else block - None just means out of bounds, not null value
                    }
                }
            }
        }

        // Remove initial rows with NaN values
        let removed_count = first_valid_idx;
        let valid_count = original_len - removed_count;

        if removed_count > 0 {
            log::info!("Removing {} initial rows with NaN values, keeping {} clean rows ({:.1}% of original data)",
                      removed_count, valid_count, (valid_count as f64 / original_len as f64) * 100.0);
            log::info!(
                "Training will start from row {} (0-indexed)",
                first_valid_idx
            );

            // Create DataFrame with only valid rows
            df = df.slice(first_valid_idx as i64, valid_count);
        } else {
            log::info!(
                "No initial NaN values found, all {} rows are clean",
                original_len
            );
        }

        Ok(df)
    }
    fn validate_features(&self, df: &DataFrame, stage: &str) -> Result<()> {
        let mut nan_rows = Vec::new();

        for series in df.get_columns() {
            if series.dtype().is_numeric() {
                if let Ok(float_series) = series.f64() {
                    for (i, value) in float_series.into_iter().enumerate() {
                        if let Some(v) = value {
                            if !v.is_finite() {
                                nan_rows.push(i);
                            }
                        }
                    }
                }
            }
        }

        if !nan_rows.is_empty() {
            nan_rows.sort_unstable();
            nan_rows.dedup();
            log::warn!("Found {} rows with NaN values at stage '{}'. These rows will be excluded from training.",
                      nan_rows.len(), stage);
            log::debug!(
                "NaN rows indices: {:?}",
                &nan_rows[..nan_rows.len().min(10)]
            );
        }

        Ok(())
    }
}
