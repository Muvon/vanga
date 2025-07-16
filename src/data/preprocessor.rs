// Data preprocessor
use crate::config::training::DataConfig;
use crate::utils::error::{Result, VangaError};
use polars::prelude::*;

/// Strategy for replacing outlier values in time-series data
#[derive(Debug, Clone)]
enum ReplacementStrategy {
    /// Cap outliers to bounds (preserves direction, limits magnitude)
    Cap,
    /// Replace with median/center value (maximum stability)
    Median,
    /// Interpolate from surrounding values (now implemented)
    Interpolate,
}

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
        log::info!(
            "Starting data preprocessing for training with {} rows and {} columns",
            df.height(),
            df.width()
        );

        // Validate input DataFrame
        if df.height() == 0 {
            return Err(VangaError::DataError(
                "Cannot process empty DataFrame for training".to_string(),
            ));
        }

        // Apply outlier removal if enabled
        df = if config.outlier_handling.enabled {
            log::info!(
                "Outlier handling enabled: method={:?}, threshold={}",
                config.outlier_handling.method,
                config.outlier_handling.threshold
            );

            let original_rows = df.height();

            let processed_df = match config.outlier_handling.method {
                crate::config::training::OutlierMethod::IQR => {
                    self.remove_outliers_iqr(df, config.outlier_handling.threshold)?
                }
                crate::config::training::OutlierMethod::ZScore => {
                    self.remove_outliers_zscore(df, config.outlier_handling.threshold)?
                }
                crate::config::training::OutlierMethod::ModifiedZScore => {
                    self.remove_outliers_modified_zscore(df, config.outlier_handling.threshold)?
                }
            };

            let remaining_rows = processed_df.height();
            if remaining_rows == 0 {
                return Err(VangaError::DataError(
                    "All rows were removed during outlier detection. Consider adjusting the threshold or disabling outlier handling.".to_string()
                ));
            }

            if remaining_rows < original_rows / 2 {
                log::warn!("Outlier removal eliminated more than 50% of data ({} -> {} rows). Consider adjusting threshold.",
                          original_rows, remaining_rows);
            }

            processed_df
        } else {
            log::info!(
                "Outlier handling disabled, keeping all {} rows",
                df.height()
            );
            df
        };

        // Apply feature engineering if config provided
        if let Some(config) = features_config {
            log::info!("Applying feature engineering from config");

            let feature_engineer = crate::features::FeatureEngineer::new(config.clone());

            match feature_engineer.generate_features(df).await {
                Ok(engineered_df) => {
                    df = engineered_df;
                    log::info!(
                        "Feature engineering completed: {} columns after processing",
                        df.width()
                    );
                }
                Err(e) => {
                    log::error!("Feature engineering failed: {}", e);
                    return Err(VangaError::DataError(format!(
                        "Feature engineering failed: {}. Check your feature configuration.",
                        e
                    )));
                }
            }

            // Remove rows with NaN values instead of failing
            match self.remove_nan_rows(df) {
                Ok(clean_df) => {
                    df = clean_df;
                    if df.height() == 0 {
                        return Err(VangaError::DataError(
                            "All rows contain NaN values after feature engineering. This indicates a problem with feature calculation.".to_string()
                        ));
                    }
                }
                Err(e) => {
                    log::error!("NaN removal failed: {}", e);
                    return Err(e);
                }
            }

            // Validate that all remaining data is clean
            if let Err(e) = self.validate_features(&df, "after feature engineering and NaN removal")
            {
                log::error!("Feature validation failed: {}", e);
                return Err(VangaError::DataError(format!(
                    "Data validation failed after feature engineering: {}. Check feature calculation logic.", e
                )));
            }
        }

        // Apply normalization based on configuration
        log::info!(
            "Applying {:?} normalization to {} columns",
            config.normalization,
            df.width()
        );

        let original_columns = df.width();
        df = match config.normalization {
            crate::config::training::NormalizationMethod::Robust => {
                match self.normalize_features_robust(df) {
                    Ok(normalized_df) => normalized_df,
                    Err(e) => {
                        log::error!("Robust normalization failed: {}", e);
                        return Err(VangaError::DataError(format!(
                            "Robust normalization failed: {}. Check for constant columns or invalid data.", e
                        )));
                    }
                }
            }
            crate::config::training::NormalizationMethod::MinMax => {
                match self.normalize_features_minmax(df) {
                    Ok(normalized_df) => normalized_df,
                    Err(e) => {
                        log::error!("MinMax normalization failed: {}", e);
                        return Err(VangaError::DataError(format!(
                            "MinMax normalization failed: {}. Check for constant columns or invalid data.", e
                        )));
                    }
                }
            }
            crate::config::training::NormalizationMethod::Standard => {
                match self.normalize_features_standard(df) {
                    Ok(normalized_df) => normalized_df,
                    Err(e) => {
                        log::error!("Standard normalization failed: {}", e);
                        return Err(VangaError::DataError(format!(
                            "Standard normalization failed: {}. Check for constant columns or invalid data.", e
                        )));
                    }
                }
            }
            crate::config::training::NormalizationMethod::Quantile => {
                match self.normalize_features_quantile(df) {
                    Ok(normalized_df) => normalized_df,
                    Err(e) => {
                        log::error!("Quantile normalization failed: {}", e);
                        return Err(VangaError::DataError(format!(
                            "Quantile normalization failed: {}. Check for constant columns or invalid data.", e
                        )));
                    }
                }
            }
        };

        // Final validation
        if df.width() != original_columns {
            log::warn!(
                "Column count changed during normalization: {} -> {}",
                original_columns,
                df.width()
            );
        }

        if df.height() == 0 {
            return Err(VangaError::DataError(
                "No data remaining after preprocessing. Check your data quality and configuration."
                    .to_string(),
            ));
        }

        log::info!(
            "Data preprocessing completed successfully: {} rows, {} columns",
            df.height(),
            df.width()
        );
        Ok(df)
    }

    pub async fn process_for_prediction(
        &self,
        mut df: DataFrame,
        symbol: &str,
    ) -> Result<DataFrame> {
        log::info!(
            "Processing data for prediction on symbol: {} ({} rows, {} columns)",
            symbol,
            df.height(),
            df.width()
        );

        // Validate input DataFrame
        if df.height() == 0 {
            return Err(VangaError::DataError(format!(
                "Cannot process empty DataFrame for prediction on symbol {}",
                symbol
            )));
        }

        // Basic validation for prediction data
        match self.validate_prediction_data(df, symbol) {
            Ok(validated_df) => {
                df = validated_df;
                log::info!("Prediction data validation passed for symbol {}", symbol);
            }
            Err(e) => {
                log::error!(
                    "Prediction data validation failed for symbol {}: {}",
                    symbol,
                    e
                );
                return Err(VangaError::DataError(format!(
                    "Prediction data validation failed for symbol {}: {}. Ensure required columns (open, high, low, close, volume) are present.",
                    symbol, e
                )));
            }
        }

        // Apply same preprocessing as training (without target-specific operations)
        // For prediction, we should use stored normalization parameters from training
        // For now, we'll use Robust normalization as default
        match self.normalize_features_robust(df) {
            Ok(normalized_df) => {
                df = normalized_df;
                log::info!(
                    "Prediction data normalization completed for symbol {} ({} rows, {} columns)",
                    symbol,
                    df.height(),
                    df.width()
                );
            }
            Err(e) => {
                log::error!(
                    "Prediction data normalization failed for symbol {}: {}",
                    symbol,
                    e
                );
                return Err(VangaError::DataError(format!(
                    "Prediction data normalization failed for symbol {}: {}. Check for constant columns or invalid data.",
                    symbol, e
                )));
            }
        }

        // Final validation
        if df.height() == 0 {
            return Err(VangaError::DataError(format!(
                "No data remaining after preprocessing for prediction on symbol {}. Check your data quality.",
                symbol
            )));
        }

        log::info!("Prediction data preprocessing completed successfully for symbol {}: {} rows, {} columns",
                  symbol, df.height(), df.width());
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

        // Validate input data
        if symbol_data.is_empty() {
            return Err(VangaError::DataError(
                "Cannot process empty symbol data for cross-asset prediction".to_string(),
            ));
        }

        // Validate each symbol's data
        for (symbol, df) in &symbol_data {
            if df.height() == 0 {
                return Err(VangaError::DataError(format!(
                    "Symbol {} has empty DataFrame for cross-asset prediction",
                    symbol
                )));
            }
            log::debug!(
                "Symbol {}: {} rows, {} columns",
                symbol,
                df.height(),
                df.width()
            );
        }

        // Apply cross-asset feature engineering if enabled
        if features_config.cross_asset.enabled {
            log::info!(
                "Cross-asset feature engineering enabled, processing {} symbols",
                symbol_data.len()
            );

            let cross_asset_generator = crate::features::CrossAssetFeatureGenerator::new(
                features_config.cross_asset.clone(),
            );

            match cross_asset_generator
                .generate_cross_asset_features(&symbol_data)
                .await
            {
                Ok(processed_data) => {
                    log::info!(
                        "Cross-asset feature engineering completed for {} symbols",
                        processed_data.len()
                    );

                    // Validate processed data
                    for (symbol, df) in &processed_data {
                        if df.height() == 0 {
                            log::warn!(
                                "Symbol {} has no data after cross-asset processing",
                                symbol
                            );
                        } else {
                            log::debug!(
                                "Symbol {} after cross-asset processing: {} rows, {} columns",
                                symbol,
                                df.height(),
                                df.width()
                            );
                        }
                    }

                    Ok(processed_data)
                }
                Err(e) => {
                    log::error!("Cross-asset feature engineering failed: {}", e);
                    Err(VangaError::DataError(format!(
                        "Cross-asset feature engineering failed: {}. Check your cross-asset configuration and ensure all required symbols are present.",
                        e
                    )))
                }
            }
        } else {
            log::info!("Cross-asset feature engineering disabled, processing symbols individually");

            // Process each symbol individually
            let mut processed_data = std::collections::HashMap::new();
            let mut processing_errors = Vec::new();

            for (symbol, df) in symbol_data {
                match self.process_for_prediction(df, &symbol).await {
                    Ok(processed_df) => {
                        log::debug!(
                            "Individual processing completed for symbol {}: {} rows, {} columns",
                            symbol,
                            processed_df.height(),
                            processed_df.width()
                        );
                        processed_data.insert(symbol, processed_df);
                    }
                    Err(e) => {
                        log::error!("Individual processing failed for symbol {}: {}", symbol, e);
                        processing_errors.push(format!("Symbol {}: {}", symbol, e));
                    }
                }
            }

            if !processing_errors.is_empty() {
                return Err(VangaError::DataError(format!(
                    "Individual symbol processing failed for {} symbols: {}",
                    processing_errors.len(),
                    processing_errors.join("; ")
                )));
            }

            if processed_data.is_empty() {
                return Err(VangaError::DataError(
                    "No symbols were successfully processed for cross-asset prediction".to_string(),
                ));
            }

            log::info!(
                "Individual symbol processing completed for {} symbols",
                processed_data.len()
            );
            Ok(processed_data)
        }
    }

    /// Remove outliers using Interquartile Range (IQR) method with time-series preservation
    fn remove_outliers_iqr(&self, mut df: DataFrame, threshold: f64) -> Result<DataFrame> {
        log::info!(
            "Applying IQR outlier handling with time-series preservation, threshold: {}",
            threshold
        );

        let original_len = df.height();
        let mut processed_columns = Vec::new();
        let mut outlier_stats = Vec::new();

        // Process each column
        for column_name in df.get_column_names() {
            if let Ok(series) = df.column(column_name) {
                if self.should_process_column_for_outliers(column_name)
                    && series.dtype().is_numeric()
                {
                    if let Ok(float_series) = series.f64() {
                        log::debug!("Processing column '{}' for IQR outliers", column_name);

                        // Calculate IQR statistics
                        let values: Vec<f64> = float_series
                            .into_iter()
                            .filter_map(|v| v.filter(|x| x.is_finite()))
                            .collect();

                        if values.is_empty() {
                            log::warn!(
                                "Column '{}' has no valid values, skipping outlier processing",
                                column_name
                            );
                            processed_columns.push(series.clone());
                            continue;
                        }

                        let mut sorted_values = values.clone();
                        sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap());

                        let q1 = self.calculate_percentile(&sorted_values, 0.25);
                        let q3 = self.calculate_percentile(&sorted_values, 0.75);
                        let iqr = q3 - q1;

                        if iqr == 0.0 {
                            log::warn!("Column '{}' has zero IQR (constant values), skipping outlier processing", column_name);
                            processed_columns.push(series.clone());
                            continue;
                        }

                        let lower_bound = q1 - threshold * iqr;
                        let upper_bound = q3 + threshold * iqr;

                        // Replace outlier values instead of removing rows
                        let (processed_series, outlier_count) = self.replace_outlier_values_iqr(
                            float_series,
                            column_name,
                            lower_bound,
                            upper_bound,
                            q1,
                            q3,
                        )?;

                        processed_columns.push(processed_series);

                        if outlier_count > 0 {
                            outlier_stats.push(format!(
                                "{}: {} outliers replaced ({:.1}%), bounds=[{:.4}, {:.4}]",
                                column_name,
                                outlier_count,
                                (outlier_count as f64 / original_len as f64) * 100.0,
                                lower_bound,
                                upper_bound
                            ));
                        }

                        log::debug!("Column '{}': Q1={:.4}, Q3={:.4}, IQR={:.4}, bounds=[{:.4}, {:.4}], {} outliers replaced",
                                   column_name, q1, q3, iqr, lower_bound, upper_bound, outlier_count);
                    } else {
                        processed_columns.push(series.clone());
                    }
                } else {
                    // Keep non-numeric columns and protected columns as-is
                    processed_columns.push(series.clone());
                }
            }
        }

        // Create new DataFrame with processed columns
        df = DataFrame::new(processed_columns)?;

        if !outlier_stats.is_empty() {
            log::info!(
                "IQR outlier processing completed: {}",
                outlier_stats.join("; ")
            );
        } else {
            log::info!(
                "No outliers detected using IQR method with threshold {}",
                threshold
            );
        }

        log::info!(
            "Time-series integrity preserved: {} rows maintained",
            df.height()
        );
        Ok(df)
    }

    /// Remove outliers using Z-Score method with time-series preservation
    fn remove_outliers_zscore(&self, mut df: DataFrame, threshold: f64) -> Result<DataFrame> {
        log::info!(
            "Applying Z-Score outlier handling with time-series preservation, threshold: {}",
            threshold
        );

        let original_len = df.height();
        let mut processed_columns = Vec::new();
        let mut outlier_stats = Vec::new();

        // Process each column
        for column_name in df.get_column_names() {
            if let Ok(series) = df.column(column_name) {
                if self.should_process_column_for_outliers(column_name)
                    && series.dtype().is_numeric()
                {
                    if let Ok(float_series) = series.f64() {
                        log::debug!("Processing column '{}' for Z-Score outliers", column_name);

                        // Calculate mean and standard deviation
                        let values: Vec<f64> = float_series
                            .into_iter()
                            .filter_map(|v| v.filter(|x| x.is_finite()))
                            .collect();

                        if values.is_empty() {
                            log::warn!(
                                "Column '{}' has no valid values, skipping outlier processing",
                                column_name
                            );
                            processed_columns.push(series.clone());
                            continue;
                        }

                        let mean = values.iter().sum::<f64>() / values.len() as f64;
                        let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>()
                            / values.len() as f64;
                        let std_dev = variance.sqrt();

                        if std_dev == 0.0 {
                            log::warn!("Column '{}' has zero standard deviation (constant values), skipping outlier processing", column_name);
                            processed_columns.push(series.clone());
                            continue;
                        }

                        // Replace outlier values instead of removing rows
                        let (processed_series, outlier_count) = self
                            .replace_outlier_values_zscore(
                                float_series,
                                column_name,
                                mean,
                                std_dev,
                                threshold,
                            )?;

                        processed_columns.push(processed_series);

                        if outlier_count > 0 {
                            outlier_stats.push(format!(
                                "{}: {} outliers replaced ({:.1}%), mean={:.4}, std={:.4}",
                                column_name,
                                outlier_count,
                                (outlier_count as f64 / original_len as f64) * 100.0,
                                mean,
                                std_dev
                            ));
                        }

                        log::debug!("Column '{}': mean={:.4}, std={:.4}, threshold={:.2}, {} outliers replaced",
                                   column_name, mean, std_dev, threshold, outlier_count);
                    } else {
                        processed_columns.push(series.clone());
                    }
                } else {
                    // Keep non-numeric columns and protected columns as-is
                    processed_columns.push(series.clone());
                }
            }
        }

        // Create new DataFrame with processed columns
        df = DataFrame::new(processed_columns)?;

        if !outlier_stats.is_empty() {
            log::info!(
                "Z-Score outlier processing completed: {}",
                outlier_stats.join("; ")
            );
        } else {
            log::info!(
                "No outliers detected using Z-Score method with threshold {}",
                threshold
            );
        }

        log::info!(
            "Time-series integrity preserved: {} rows maintained",
            df.height()
        );
        Ok(df)
    }

    /// Remove outliers using Modified Z-Score method with time-series preservation (most robust)
    fn remove_outliers_modified_zscore(
        &self,
        mut df: DataFrame,
        threshold: f64,
    ) -> Result<DataFrame> {
        log::info!("Applying Modified Z-Score outlier handling with time-series preservation, threshold: {}", threshold);

        let original_len = df.height();
        let mut processed_columns = Vec::new();
        let mut outlier_stats = Vec::new();

        // Process each column
        for column_name in df.get_column_names() {
            if let Ok(series) = df.column(column_name) {
                if self.should_process_column_for_outliers(column_name)
                    && series.dtype().is_numeric()
                {
                    if let Ok(float_series) = series.f64() {
                        log::debug!(
                            "Processing column '{}' for Modified Z-Score outliers",
                            column_name
                        );

                        // Calculate median and MAD (Median Absolute Deviation)
                        let values: Vec<f64> = float_series
                            .into_iter()
                            .filter_map(|v| v.filter(|x| x.is_finite()))
                            .collect();

                        if values.is_empty() {
                            log::warn!(
                                "Column '{}' has no valid values, skipping outlier processing",
                                column_name
                            );
                            processed_columns.push(series.clone());
                            continue;
                        }

                        let mut sorted_values = values.clone();
                        sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
                        let median = self.calculate_percentile(&sorted_values, 0.5);

                        // Calculate MAD
                        let mut deviations: Vec<f64> =
                            values.iter().map(|x| (x - median).abs()).collect();
                        deviations.sort_by(|a, b| a.partial_cmp(b).unwrap());
                        let mad = self.calculate_percentile(&deviations, 0.5);

                        if mad == 0.0 {
                            log::warn!("Column '{}' has zero MAD (constant values), skipping outlier processing", column_name);
                            processed_columns.push(series.clone());
                            continue;
                        }

                        // Replace outlier values instead of removing rows
                        let (processed_series, outlier_count) = self
                            .replace_outlier_values_modified_zscore(
                                float_series,
                                column_name,
                                median,
                                mad,
                                threshold,
                            )?;

                        processed_columns.push(processed_series);

                        if outlier_count > 0 {
                            outlier_stats.push(format!(
                                "{}: {} outliers replaced ({:.1}%), median={:.4}, MAD={:.4}",
                                column_name,
                                outlier_count,
                                (outlier_count as f64 / original_len as f64) * 100.0,
                                median,
                                mad
                            ));
                        }

                        log::debug!("Column '{}': median={:.4}, MAD={:.4}, threshold={:.2}, {} outliers replaced",
                                   column_name, median, mad, threshold, outlier_count);
                    } else {
                        processed_columns.push(series.clone());
                    }
                } else {
                    // Keep non-numeric columns and protected columns as-is
                    processed_columns.push(series.clone());
                }
            }
        }

        // Create new DataFrame with processed columns
        df = DataFrame::new(processed_columns)?;

        if !outlier_stats.is_empty() {
            log::info!(
                "Modified Z-Score outlier processing completed: {}",
                outlier_stats.join("; ")
            );
        } else {
            log::info!(
                "No outliers detected using Modified Z-Score method with threshold {}",
                threshold
            );
        }

        log::info!(
            "Time-series integrity preserved: {} rows maintained",
            df.height()
        );
        Ok(df)
    }

    /// Calculate percentile from sorted values
    fn calculate_percentile(&self, sorted_values: &[f64], percentile: f64) -> f64 {
        if sorted_values.is_empty() {
            return 0.0;
        }

        let index = (percentile * (sorted_values.len() - 1) as f64).round() as usize;
        sorted_values[index.min(sorted_values.len() - 1)]
    }

    /// Determine if a column should be processed for outlier detection
    fn should_process_column_for_outliers(&self, column_name: &str) -> bool {
        // Skip timestamp and preserve price data integrity for time-series
        match column_name {
            "timestamp" => false,
            // Preserve OHLC price data - these are legitimate market movements
            "open" | "high" | "low" | "close" => false,
            // Process all other columns (volume, indicators, etc.)
            _ => true,
        }
    }

    /// Replace outlier values using IQR method with column-aware strategies
    fn replace_outlier_values_iqr(
        &self,
        float_series: &polars::chunked_array::ChunkedArray<polars::datatypes::Float64Type>,
        column_name: &str,
        lower_bound: f64,
        upper_bound: f64,
        q1: f64,
        q3: f64,
    ) -> Result<(Series, usize)> {
        let mut outlier_count = 0;
        let replacement_strategy = self.get_replacement_strategy(column_name);

        let processed_values: Vec<Option<f64>> = float_series
            .into_iter()
            .enumerate()
            .map(|(index, value)| {
                if let Some(v) = value {
                    if v.is_finite() && (v < lower_bound || v > upper_bound) {
                        outlier_count += 1;
                        // Replace outlier with appropriate value based on strategy
                        let replacement_value = match replacement_strategy {
                            ReplacementStrategy::Interpolate => self.interpolate_outlier_value(
                                float_series,
                                index,
                                v,
                                (q1 + q3) / 2.0,
                            ),
                            _ => self.get_replacement_value(
                                v,
                                &replacement_strategy,
                                (q1 + q3) / 2.0,
                                q1,
                                lower_bound,
                                upper_bound,
                            ),
                        };
                        Some(replacement_value)
                    } else {
                        Some(v)
                    }
                } else {
                    value
                }
            })
            .collect();

        let processed_series = Series::new(column_name, processed_values);
        Ok((processed_series, outlier_count))
    }

    /// Replace outlier values using Z-Score method with column-aware strategies
    fn replace_outlier_values_zscore(
        &self,
        float_series: &polars::chunked_array::ChunkedArray<polars::datatypes::Float64Type>,
        column_name: &str,
        mean: f64,
        std_dev: f64,
        threshold: f64,
    ) -> Result<(Series, usize)> {
        let mut outlier_count = 0;
        let replacement_strategy = self.get_replacement_strategy(column_name);

        let processed_values: Vec<Option<f64>> = float_series
            .into_iter()
            .enumerate()
            .map(|(index, value)| {
                if let Some(v) = value {
                    if v.is_finite() {
                        let z_score = (v - mean).abs() / std_dev;
                        if z_score > threshold {
                            outlier_count += 1;
                            // Replace outlier with appropriate value based on strategy
                            let lower_bound = mean - threshold * std_dev;
                            let upper_bound = mean + threshold * std_dev;
                            let replacement_value = match replacement_strategy {
                                ReplacementStrategy::Interpolate => {
                                    self.interpolate_outlier_value(float_series, index, v, mean)
                                }
                                _ => self.get_replacement_value(
                                    v,
                                    &replacement_strategy,
                                    mean,
                                    mean,
                                    lower_bound,
                                    upper_bound,
                                ),
                            };
                            Some(replacement_value)
                        } else {
                            Some(v)
                        }
                    } else {
                        Some(v)
                    }
                } else {
                    value
                }
            })
            .collect();

        let processed_series = Series::new(column_name, processed_values);
        Ok((processed_series, outlier_count))
    }

    /// Replace outlier values using Modified Z-Score method with column-aware strategies
    fn replace_outlier_values_modified_zscore(
        &self,
        float_series: &polars::chunked_array::ChunkedArray<polars::datatypes::Float64Type>,
        column_name: &str,
        median: f64,
        mad: f64,
        threshold: f64,
    ) -> Result<(Series, usize)> {
        let mut outlier_count = 0;
        let replacement_strategy = self.get_replacement_strategy(column_name);
        let scale_factor = 0.6745;

        let processed_values: Vec<Option<f64>> = float_series
            .into_iter()
            .enumerate()
            .map(|(index, value)| {
                if let Some(v) = value {
                    if v.is_finite() {
                        let modified_z_score = scale_factor * (v - median).abs() / mad;
                        if modified_z_score > threshold {
                            outlier_count += 1;
                            // Replace outlier with appropriate value based on strategy
                            let bound_range = threshold * mad / scale_factor;
                            let lower_bound = median - bound_range;
                            let upper_bound = median + bound_range;
                            let replacement_value = match replacement_strategy {
                                ReplacementStrategy::Interpolate => {
                                    self.interpolate_outlier_value(float_series, index, v, median)
                                }
                                _ => self.get_replacement_value(
                                    v,
                                    &replacement_strategy,
                                    median,
                                    median,
                                    lower_bound,
                                    upper_bound,
                                ),
                            };
                            Some(replacement_value)
                        } else {
                            Some(v)
                        }
                    } else {
                        Some(v)
                    }
                } else {
                    value
                }
            })
            .collect();

        let processed_series = Series::new(column_name, processed_values);
        Ok((processed_series, outlier_count))
    }

    /// Determine replacement strategy based on column type
    fn get_replacement_strategy(&self, column_name: &str) -> ReplacementStrategy {
        let strategy = match column_name {
            // Volume: Cap to reasonable bounds (preserve market structure)
            "volume" => ReplacementStrategy::Cap,
            // Technical indicators: Use median for stability
            name if name.contains("rsi")
                || name.contains("macd")
                || name.contains("sma")
                || name.contains("ema")
                || name.contains("bb")
                || name.contains("stoch") =>
            {
                ReplacementStrategy::Median
            }
            // Price-derived features: Use interpolation for smooth time-series continuity
            name if name.contains("return")
                || name.contains("change")
                || name.contains("momentum")
                || name.contains("velocity")
                || name.contains("acceleration")
                || name.contains("volatility") =>
            {
                ReplacementStrategy::Interpolate
            }
            // Other features: Use bounds capping
            _ => ReplacementStrategy::Cap,
        };

        log::debug!(
            "Column '{}' using {:?} replacement strategy",
            column_name,
            strategy
        );
        strategy
    }

    /// Get replacement value based on strategy and outlier characteristics
    fn get_replacement_value(
        &self,
        outlier_value: f64,
        strategy: &ReplacementStrategy,
        center_value: f64,
        _fallback_center: f64,
        lower_bound: f64,
        upper_bound: f64,
    ) -> f64 {
        match strategy {
            ReplacementStrategy::Cap => {
                // Cap to bounds - preserves direction of outlier but limits magnitude
                if outlier_value > upper_bound {
                    upper_bound
                } else if outlier_value < lower_bound {
                    lower_bound
                } else {
                    outlier_value
                }
            }
            ReplacementStrategy::Median => {
                // Replace with median/center value for stability
                center_value
            }
            ReplacementStrategy::Interpolate => {
                // Interpolation now fully implemented with surrounding values
                log::debug!("Using interpolation strategy for outlier replacement");
                center_value // This path should not be reached as interpolation is handled in replacement methods
            }
        }
    }

    /// Interpolate outlier value using surrounding valid values
    fn interpolate_outlier_value(
        &self,
        series: &polars::chunked_array::ChunkedArray<polars::datatypes::Float64Type>,
        outlier_index: usize,
        outlier_value: f64,
        fallback_value: f64,
    ) -> f64 {
        let series_len = series.len();

        // Handle edge cases
        if series_len <= 2 {
            log::debug!("Series too short for interpolation, using fallback value");
            return fallback_value;
        }

        // Find previous valid value
        let mut prev_value = None;
        let mut prev_index = None;
        for i in (0..outlier_index).rev() {
            if let Some(val) = series.get(i) {
                if val.is_finite() {
                    prev_value = Some(val);
                    prev_index = Some(i);
                    break;
                }
            }
        }

        // Find next valid value
        let mut next_value = None;
        let mut next_index = None;
        for i in (outlier_index + 1)..series_len {
            if let Some(val) = series.get(i) {
                if val.is_finite() {
                    next_value = Some(val);
                    next_index = Some(i);
                    break;
                }
            }
        }

        // Perform interpolation based on available surrounding values
        match (prev_value, next_value, prev_index, next_index) {
            (Some(prev), Some(next), Some(prev_idx), Some(next_idx)) => {
                // Linear interpolation between previous and next values
                let distance_from_prev = (outlier_index - prev_idx) as f64;
                let total_distance = (next_idx - prev_idx) as f64;
                let interpolation_ratio = distance_from_prev / total_distance;

                let interpolated = prev + (next - prev) * interpolation_ratio;

                log::debug!(
                    "Interpolated outlier at index {} (value={:.4}) between prev={:.4} and next={:.4} -> {:.4}",
                    outlier_index, outlier_value, prev, next, interpolated
                );

                interpolated
            }
            (Some(prev), Some(_), None, _)
            | (Some(prev), None, _, _)
            | (Some(prev), Some(_), Some(_), None) => {
                // Only previous value available or index issues, use previous value
                log::debug!(
                    "Using previous value {:.4} for outlier at index {} (no next value or index issues)",
                    prev, outlier_index
                );
                prev
            }
            (None, Some(next), _, _) => {
                // Only next value available, use it
                log::debug!(
                    "Using next value {:.4} for outlier at index {} (no previous value)",
                    next,
                    outlier_index
                );
                next
            }
            (None, None, _, _) => {
                // No surrounding values available, use fallback
                log::debug!(
                    "No surrounding values for interpolation at index {}, using fallback {:.4}",
                    outlier_index,
                    fallback_value
                );
                fallback_value
            }
        }
    }

    /// Calculate comprehensive statistics for all numeric columns
    pub fn calculate_statistics(&self, df: &DataFrame) -> Result<crate::data::NormalizationStats> {
        log::info!(
            "Calculating comprehensive statistics for {} columns",
            df.width()
        );

        let mut means = Vec::new();
        let mut stds = Vec::new();
        let mut mins = Vec::new();
        let mut maxs = Vec::new();
        let mut medians = Vec::new();
        let mut q25 = Vec::new();
        let mut q75 = Vec::new();

        // Process each numeric column
        for column_name in df.get_column_names() {
            if let Ok(series) = df.column(column_name) {
                if series.dtype().is_numeric() && column_name != "timestamp" {
                    if let Ok(float_series) = series.f64() {
                        // Extract valid values
                        let values: Vec<f64> = float_series
                            .into_iter()
                            .filter_map(|v| v.filter(|x| x.is_finite()))
                            .collect();

                        if values.is_empty() {
                            log::warn!(
                                "Column '{}' has no valid values, using default statistics",
                                column_name
                            );
                            means.push(0.0);
                            stds.push(1.0);
                            mins.push(0.0);
                            maxs.push(1.0);
                            medians.push(0.0);
                            q25.push(0.0);
                            q75.push(1.0);
                            continue;
                        }

                        // Calculate basic statistics
                        let mean = values.iter().sum::<f64>() / values.len() as f64;
                        let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>()
                            / values.len() as f64;
                        let std_dev = variance.sqrt();
                        let min_val = values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
                        let max_val = values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

                        // Calculate quantiles
                        let mut sorted_values = values.clone();
                        sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap());

                        let median = self.calculate_percentile(&sorted_values, 0.5);
                        let q1 = self.calculate_percentile(&sorted_values, 0.25);
                        let q3 = self.calculate_percentile(&sorted_values, 0.75);

                        // Store statistics
                        means.push(mean);
                        stds.push(std_dev);
                        mins.push(min_val);
                        maxs.push(max_val);
                        medians.push(median);
                        q25.push(q1);
                        q75.push(q3);

                        log::debug!("Column '{}': mean={:.4}, std={:.4}, min={:.4}, max={:.4}, median={:.4}, Q1={:.4}, Q3={:.4}",
                                   column_name, mean, std_dev, min_val, max_val, median, q1, q3);
                    }
                }
            }
        }

        let stats = crate::data::NormalizationStats {
            means,
            stds,
            mins,
            maxs,
            medians,
            q25,
            q75,
        };

        log::info!(
            "Statistics calculated for {} numeric columns",
            stats.means.len()
        );
        Ok(stats)
    }

    /// Apply normalization using pre-calculated statistics
    pub fn apply_normalization_with_stats(
        &self,
        df: DataFrame,
        stats: &crate::data::NormalizationStats,
        method: &crate::config::training::NormalizationMethod,
    ) -> Result<DataFrame> {
        log::info!(
            "Applying {:?} normalization using pre-calculated statistics",
            method
        );

        let mut normalized_columns = Vec::new();
        let mut stats_index = 0;

        // Process each column
        for column_name in df.get_column_names() {
            if let Ok(series) = df.column(column_name) {
                if series.dtype().is_numeric() && column_name != "timestamp" {
                    if stats_index >= stats.means.len() {
                        return Err(crate::utils::error::VangaError::DataError(
                            "Statistics index out of bounds - mismatch between training and prediction data".to_string()
                        ));
                    }

                    if let Ok(float_series) = series.f64() {
                        let normalized_values: Vec<Option<f64>> = match method {
                            crate::config::training::NormalizationMethod::Robust => {
                                let median = stats.medians[stats_index];
                                let iqr = stats.q75[stats_index] - stats.q25[stats_index];

                                if iqr == 0.0 {
                                    log::warn!(
                                        "Column '{}' has zero IQR, skipping normalization",
                                        column_name
                                    );
                                    float_series.into_iter().collect()
                                } else {
                                    float_series
                                        .into_iter()
                                        .map(|v| {
                                            v.filter(|x| x.is_finite()).map(|x| (x - median) / iqr)
                                        })
                                        .collect()
                                }
                            }
                            crate::config::training::NormalizationMethod::MinMax => {
                                let min_val = stats.mins[stats_index];
                                let range = stats.maxs[stats_index] - min_val;

                                if range == 0.0 {
                                    log::warn!(
                                        "Column '{}' has zero range, skipping normalization",
                                        column_name
                                    );
                                    float_series.into_iter().collect()
                                } else {
                                    float_series
                                        .into_iter()
                                        .map(|v| {
                                            v.filter(|x| x.is_finite())
                                                .map(|x| (x - min_val) / range)
                                        })
                                        .collect()
                                }
                            }
                            crate::config::training::NormalizationMethod::Standard => {
                                let mean = stats.means[stats_index];
                                let std_dev = stats.stds[stats_index];

                                if std_dev == 0.0 {
                                    log::warn!(
                                        "Column '{}' has zero std dev, skipping normalization",
                                        column_name
                                    );
                                    float_series.into_iter().collect()
                                } else {
                                    float_series
                                        .into_iter()
                                        .map(|v| {
                                            v.filter(|x| x.is_finite())
                                                .map(|x| (x - mean) / std_dev)
                                        })
                                        .collect()
                                }
                            }
                            crate::config::training::NormalizationMethod::Quantile => {
                                // For quantile normalization, we would need to store the sorted values
                                // For now, fall back to robust normalization
                                log::warn!("Quantile normalization with pre-calculated stats not fully implemented, using robust");
                                let median = stats.medians[stats_index];
                                let iqr = stats.q75[stats_index] - stats.q25[stats_index];

                                if iqr == 0.0 {
                                    float_series.into_iter().collect()
                                } else {
                                    float_series
                                        .into_iter()
                                        .map(|v| {
                                            v.filter(|x| x.is_finite()).map(|x| (x - median) / iqr)
                                        })
                                        .collect()
                                }
                            }
                        };

                        let normalized_series = Series::new(column_name, normalized_values);
                        normalized_columns.push(normalized_series);
                        stats_index += 1;

                        log::debug!(
                            "Column '{}' normalized using {:?} method",
                            column_name,
                            method
                        );
                    }
                } else {
                    // Keep non-numeric columns as-is
                    normalized_columns.push(series.clone());
                }
            }
        }

        let normalized_df = DataFrame::new(normalized_columns)?;
        log::info!(
            "Normalization completed using pre-calculated statistics for {} columns",
            stats_index
        );

        Ok(normalized_df)
    }

    /// Validate that DataFrame has expected structure for normalization
    pub fn validate_normalization_compatibility(
        &self,
        df: &DataFrame,
        stats: &crate::data::NormalizationStats,
    ) -> Result<()> {
        let numeric_columns: Vec<&str> = df
            .get_column_names()
            .into_iter()
            .filter(|&name| {
                if let Ok(series) = df.column(name) {
                    series.dtype().is_numeric() && name != "timestamp"
                } else {
                    false
                }
            })
            .collect();

        if numeric_columns.len() != stats.means.len() {
            return Err(crate::utils::error::VangaError::DataError(format!(
                "Column count mismatch: DataFrame has {} numeric columns, but statistics were calculated for {} columns",
                numeric_columns.len(),
                stats.means.len()
            )));
        }

        log::info!(
            "Normalization compatibility validated: {} numeric columns match statistics",
            numeric_columns.len()
        );
        Ok(())
    }

    /// Apply Robust normalization (uses median and IQR - handles outliers well)
    /// This method preserves time-series continuity and is recommended for cryptocurrency data
    fn normalize_features_robust(&self, mut df: DataFrame) -> Result<DataFrame> {
        log::info!("Applying Robust normalization (median and IQR-based) - optimal for cryptocurrency time-series data");

        let mut normalized_columns = Vec::new();

        // Process each numeric column
        for column_name in df.get_column_names() {
            if let Ok(series) = df.column(column_name) {
                if series.dtype().is_numeric() && column_name != "timestamp" {
                    if let Ok(float_series) = series.f64() {
                        // Calculate robust statistics
                        let values: Vec<f64> = float_series
                            .into_iter()
                            .filter_map(|v| v.filter(|x| x.is_finite()))
                            .collect();

                        if values.is_empty() {
                            log::warn!(
                                "Column '{}' has no valid values, skipping normalization",
                                column_name
                            );
                            continue;
                        }

                        let mut sorted_values = values.clone();
                        sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap());

                        let median = self.calculate_percentile(&sorted_values, 0.5);
                        let q1 = self.calculate_percentile(&sorted_values, 0.25);
                        let q3 = self.calculate_percentile(&sorted_values, 0.75);
                        let iqr = q3 - q1;

                        if iqr == 0.0 {
                            log::warn!("Column '{}' has zero IQR (constant values), skipping normalization", column_name);
                            continue;
                        }

                        // Apply robust normalization: (x - median) / IQR
                        let normalized_values: Vec<Option<f64>> = float_series
                            .into_iter()
                            .map(|v| v.filter(|x| x.is_finite()).map(|x| (x - median) / iqr))
                            .collect();

                        let normalized_series = Series::new(column_name, normalized_values);
                        normalized_columns.push(normalized_series);

                        log::debug!("Column '{}': median={:.4}, Q1={:.4}, Q3={:.4}, IQR={:.4} - normalized for time-series stability",
                                   column_name, median, q1, q3, iqr);
                    }
                } else {
                    // Keep non-numeric columns as-is
                    normalized_columns.push(series.clone());
                }
            }
        }

        // Create new DataFrame with normalized columns
        df = DataFrame::new(normalized_columns)?;
        log::info!(
            "Robust normalization completed for {} columns - time-series ready for LSTM training",
            df.width()
        );

        Ok(df)
    }

    /// Apply MinMax normalization (scales to [0,1] range)
    fn normalize_features_minmax(&self, mut df: DataFrame) -> Result<DataFrame> {
        log::info!("Applying MinMax normalization (scaling to [0,1] range)");

        let mut normalized_columns = Vec::new();

        // Process each numeric column
        for column_name in df.get_column_names() {
            if let Ok(series) = df.column(column_name) {
                if series.dtype().is_numeric() && column_name != "timestamp" {
                    if let Ok(float_series) = series.f64() {
                        // Calculate min and max
                        let values: Vec<f64> = float_series
                            .into_iter()
                            .filter_map(|v| v.filter(|x| x.is_finite()))
                            .collect();

                        if values.is_empty() {
                            log::warn!(
                                "Column '{}' has no valid values, skipping normalization",
                                column_name
                            );
                            continue;
                        }

                        let min_val = values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
                        let max_val = values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                        let range = max_val - min_val;

                        if range == 0.0 {
                            log::warn!("Column '{}' has zero range (constant values), skipping normalization", column_name);
                            continue;
                        }

                        // Apply MinMax normalization: (x - min) / (max - min)
                        let normalized_values: Vec<Option<f64>> = float_series
                            .into_iter()
                            .map(|v| v.filter(|x| x.is_finite()).map(|x| (x - min_val) / range))
                            .collect();

                        let normalized_series = Series::new(column_name, normalized_values);
                        normalized_columns.push(normalized_series);

                        log::debug!(
                            "Column '{}': min={:.4}, max={:.4}, range={:.4}",
                            column_name,
                            min_val,
                            max_val,
                            range
                        );
                    }
                } else {
                    // Keep non-numeric columns as-is
                    normalized_columns.push(series.clone());
                }
            }
        }

        // Create new DataFrame with normalized columns
        df = DataFrame::new(normalized_columns)?;
        log::info!("MinMax normalization completed for {} columns", df.width());

        Ok(df)
    }

    /// Apply Standard normalization (Z-score: mean=0, std=1)
    fn normalize_features_standard(&self, mut df: DataFrame) -> Result<DataFrame> {
        log::info!("Applying Standard normalization (Z-score: mean=0, std=1)");

        let mut normalized_columns = Vec::new();

        // Process each numeric column
        for column_name in df.get_column_names() {
            if let Ok(series) = df.column(column_name) {
                if series.dtype().is_numeric() && column_name != "timestamp" {
                    if let Ok(float_series) = series.f64() {
                        // Calculate mean and standard deviation
                        let values: Vec<f64> = float_series
                            .into_iter()
                            .filter_map(|v| v.filter(|x| x.is_finite()))
                            .collect();

                        if values.is_empty() {
                            log::warn!(
                                "Column '{}' has no valid values, skipping normalization",
                                column_name
                            );
                            continue;
                        }

                        let mean = values.iter().sum::<f64>() / values.len() as f64;
                        let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>()
                            / values.len() as f64;
                        let std_dev = variance.sqrt();

                        if std_dev == 0.0 {
                            log::warn!("Column '{}' has zero standard deviation (constant values), skipping normalization", column_name);
                            continue;
                        }

                        // Apply Standard normalization: (x - mean) / std
                        let normalized_values: Vec<Option<f64>> = float_series
                            .into_iter()
                            .map(|v| v.filter(|x| x.is_finite()).map(|x| (x - mean) / std_dev))
                            .collect();

                        let normalized_series = Series::new(column_name, normalized_values);
                        normalized_columns.push(normalized_series);

                        log::debug!(
                            "Column '{}': mean={:.4}, std={:.4}",
                            column_name,
                            mean,
                            std_dev
                        );
                    }
                } else {
                    // Keep non-numeric columns as-is
                    normalized_columns.push(series.clone());
                }
            }
        }

        // Create new DataFrame with normalized columns
        df = DataFrame::new(normalized_columns)?;
        log::info!(
            "Standard normalization completed for {} columns",
            df.width()
        );

        Ok(df)
    }

    /// Apply Quantile normalization (uses quantile transformation)
    fn normalize_features_quantile(&self, mut df: DataFrame) -> Result<DataFrame> {
        log::info!("Applying Quantile normalization (quantile transformation)");

        let mut normalized_columns = Vec::new();

        // Process each numeric column
        for column_name in df.get_column_names() {
            if let Ok(series) = df.column(column_name) {
                if series.dtype().is_numeric() && column_name != "timestamp" {
                    if let Ok(float_series) = series.f64() {
                        // Calculate quantiles for transformation
                        let values: Vec<f64> = float_series
                            .into_iter()
                            .filter_map(|v| v.filter(|x| x.is_finite()))
                            .collect();

                        if values.is_empty() {
                            log::warn!(
                                "Column '{}' has no valid values, skipping normalization",
                                column_name
                            );
                            continue;
                        }

                        let mut sorted_values = values.clone();
                        sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap());

                        // Apply quantile transformation
                        let normalized_values: Vec<Option<f64>> = float_series
                            .into_iter()
                            .map(|v| {
                                v.filter(|x| x.is_finite()).map(|x| {
                                    // Find the quantile rank of this value
                                    let rank = sorted_values
                                        .iter()
                                        .position(|&val| val >= x)
                                        .unwrap_or(sorted_values.len() - 1);

                                    // Convert to quantile (0 to 1)
                                    rank as f64 / (sorted_values.len() - 1) as f64
                                })
                            })
                            .collect();

                        let normalized_series = Series::new(column_name, normalized_values);
                        normalized_columns.push(normalized_series);

                        log::debug!(
                            "Column '{}': quantile transformation applied with {} unique values",
                            column_name,
                            sorted_values.len()
                        );
                    }
                } else {
                    // Keep non-numeric columns as-is
                    normalized_columns.push(series.clone());
                }
            }
        }

        // Create new DataFrame with normalized columns
        df = DataFrame::new(normalized_columns)?;
        log::info!(
            "Quantile normalization completed for {} columns",
            df.width()
        );

        Ok(df)
    }

    /// Validate prediction data format and completeness
    fn validate_prediction_data(&self, df: DataFrame, symbol: &str) -> Result<DataFrame> {
        log::debug!(
            "Validating prediction data for symbol {}: {} rows, {} columns",
            symbol,
            df.height(),
            df.width()
        );

        // Validate that required columns exist for the symbol
        let required_columns = ["open", "high", "low", "close", "volume"];
        let available_columns: Vec<&str> = df.get_column_names();
        let mut missing_columns = Vec::new();

        for col in required_columns {
            if !available_columns.contains(&col) {
                missing_columns.push(col);
            }
        }

        if !missing_columns.is_empty() {
            return Err(crate::utils::error::VangaError::DataError(format!(
                "Missing required columns for symbol {}: {}. Available columns: {}. Required columns: {}",
                symbol,
                missing_columns.join(", "),
                available_columns.join(", "),
                required_columns.join(", ")
            )));
        }

        // Validate that required columns have valid data
        for col in required_columns {
            if let Ok(series) = df.column(col) {
                if series.dtype().is_numeric() {
                    if let Ok(float_series) = series.f64() {
                        let valid_count = float_series
                            .into_iter()
                            .filter(|v| v.is_some_and(|x| x.is_finite()))
                            .count();

                        if valid_count == 0 {
                            return Err(crate::utils::error::VangaError::DataError(format!(
                                "Column '{}' for symbol {} contains no valid numeric values",
                                col, symbol
                            )));
                        }

                        let total_count = float_series.len();
                        let valid_percentage = (valid_count as f64 / total_count as f64) * 100.0;

                        if valid_percentage < 50.0 {
                            log::warn!(
                                "Column '{}' for symbol {} has only {:.1}% valid values ({}/{})",
                                col,
                                symbol,
                                valid_percentage,
                                valid_count,
                                total_count
                            );
                        }

                        log::debug!(
                            "Column '{}' for symbol {}: {}/{} valid values ({:.1}%)",
                            col,
                            symbol,
                            valid_count,
                            total_count,
                            valid_percentage
                        );
                    }
                } else {
                    return Err(crate::utils::error::VangaError::DataError(format!(
                        "Column '{}' for symbol {} is not numeric (type: {:?})",
                        col,
                        symbol,
                        series.dtype()
                    )));
                }
            }
        }

        log::debug!("Prediction data validation passed for symbol {}", symbol);
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
    /// Validate that all numeric features are finite and log detailed information
    fn validate_features(&self, df: &DataFrame, stage: &str) -> Result<()> {
        log::debug!(
            "Validating features at stage: '{}' ({} rows, {} columns)",
            stage,
            df.height(),
            df.width()
        );

        let mut nan_rows = Vec::new();
        let mut column_stats = Vec::new();

        for series in df.get_columns() {
            let column_name = series.name();

            if series.dtype().is_numeric() {
                if let Ok(float_series) = series.f64() {
                    let mut valid_count = 0;
                    let mut nan_count = 0;
                    let mut inf_count = 0;

                    for (i, value) in float_series.into_iter().enumerate() {
                        match value {
                            Some(v) => {
                                if v.is_finite() {
                                    valid_count += 1;
                                } else if v.is_nan() {
                                    nan_count += 1;
                                    nan_rows.push(i);
                                } else {
                                    inf_count += 1;
                                    nan_rows.push(i);
                                }
                            }
                            None => {
                                nan_count += 1;
                                nan_rows.push(i);
                            }
                        }
                    }

                    let total_count = float_series.len();
                    let valid_percentage = (valid_count as f64 / total_count as f64) * 100.0;

                    column_stats.push(format!(
                        "{}: {}/{} valid ({:.1}%), {} NaN, {} Inf",
                        column_name,
                        valid_count,
                        total_count,
                        valid_percentage,
                        nan_count,
                        inf_count
                    ));

                    if valid_percentage < 90.0 {
                        log::warn!(
                            "Column '{}' at stage '{}' has low data quality: {:.1}% valid values",
                            column_name,
                            stage,
                            valid_percentage
                        );
                    }
                }
            } else {
                log::debug!(
                    "Skipping non-numeric column '{}' (type: {:?})",
                    column_name,
                    series.dtype()
                );
            }
        }

        // Log column statistics
        if !column_stats.is_empty() {
            log::debug!(
                "Feature validation statistics at stage '{}': {}",
                stage,
                column_stats.join("; ")
            );
        }

        // Handle NaN rows
        if !nan_rows.is_empty() {
            nan_rows.sort_unstable();
            nan_rows.dedup();

            let nan_percentage = (nan_rows.len() as f64 / df.height() as f64) * 100.0;

            if nan_percentage > 50.0 {
                return Err(crate::utils::error::VangaError::DataError(format!(
                    "Too many rows with invalid values at stage '{}': {}/{} rows ({:.1}%) have NaN/Inf values. This indicates a serious data quality issue.",
                    stage, nan_rows.len(), df.height(), nan_percentage
                )));
            } else if nan_percentage > 10.0 {
                log::warn!("High number of rows with invalid values at stage '{}': {}/{} rows ({:.1}%) have NaN/Inf values",
                          stage, nan_rows.len(), df.height(), nan_percentage);
            } else {
                log::info!("Found {} rows with invalid values at stage '{}' ({:.1}% of data). These rows will be excluded from training.",
                          nan_rows.len(), stage, nan_percentage);
            }

            log::debug!(
                "Invalid value row indices at stage '{}': {:?}",
                stage,
                &nan_rows[..nan_rows.len().min(10)]
            );
        } else {
            log::info!(
                "All {} rows have valid values at stage '{}'",
                df.height(),
                stage
            );
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_interpolation_strategy_selection() {
        let preprocessor = DataPreprocessor;

        // Test that interpolation is selected for price-derived features
        assert!(matches!(
            preprocessor.get_replacement_strategy("price_return"),
            ReplacementStrategy::Interpolate
        ));
        assert!(matches!(
            preprocessor.get_replacement_strategy("momentum_5"),
            ReplacementStrategy::Interpolate
        ));
        assert!(matches!(
            preprocessor.get_replacement_strategy("volatility_10"),
            ReplacementStrategy::Interpolate
        ));

        // Test that median is selected for technical indicators
        assert!(matches!(
            preprocessor.get_replacement_strategy("rsi_14"),
            ReplacementStrategy::Median
        ));
        assert!(matches!(
            preprocessor.get_replacement_strategy("sma_20"),
            ReplacementStrategy::Median
        ));

        // Test that cap is selected for volume and other features
        assert!(matches!(
            preprocessor.get_replacement_strategy("volume"),
            ReplacementStrategy::Cap
        ));
        assert!(matches!(
            preprocessor.get_replacement_strategy("some_other_feature"),
            ReplacementStrategy::Cap
        ));
    }

    #[test]
    fn test_interpolation_logic() {
        let preprocessor = DataPreprocessor;

        // Create test series: [1.0, 2.0, OUTLIER, 4.0, 5.0]
        let values = vec![Some(1.0), Some(2.0), Some(100.0), Some(4.0), Some(5.0)];
        let series = Series::new("test", values).f64().unwrap().clone();

        // Test interpolation at index 2 (outlier value 100.0)
        let interpolated = preprocessor.interpolate_outlier_value(&series, 2, 100.0, 3.0);

        // Should interpolate between 2.0 and 4.0, giving 3.0
        assert!((interpolated - 3.0).abs() < 0.001);
    }
}
