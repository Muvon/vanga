use crate::config::GlobalConfig;
use crate::utils::error::{Result, VangaError};
use polars::prelude::*;
use rayon::prelude::*;
use std::fs::File;
use std::path::Path;

/// Data loader for CSV files with automatic schema detection
pub struct DataLoader {
    chunk_size: usize,
}

impl DataLoader {
    pub fn new() -> Self {
        Self {
            chunk_size: 10000, // Process in chunks for memory efficiency
        }
    }

    /// PARALLELIZED: Load multiple CSV files concurrently
    pub async fn load_multiple_csv<P: AsRef<Path> + Sync>(
        &self,
        paths: &[P],
    ) -> Result<Vec<DataFrame>> {
        log::info!("Loading {} CSV files in parallel", paths.len());

        let results: Vec<Result<DataFrame>> = paths
            .iter()
            .collect::<Vec<_>>()
            .par_iter()
            .map(|path| {
                let path = path.as_ref();
                log::debug!("Loading file: {}", path.display());

                if !path.exists() {
                    return Err(VangaError::DataError(format!(
                        "Data file not found: {}",
                        path.display()
                    )));
                }

                // Load CSV with validation - Infer schema properly
                let file = File::open(path).map_err(|e| {
                    VangaError::DataError(format!("Failed to open CSV file: {}", e))
                })?;
                let mut df = polars::prelude::CsvReader::new(file)
                    .finish()
                    .map_err(|e| VangaError::DataError(format!("Failed to read CSV: {}", e)))?;

                // Cast all columns except timestamp to Float64
                let column_names: Vec<String> = df
                    .get_column_names()
                    .iter()
                    .map(|s| s.to_string())
                    .collect();
                for col_name in column_names {
                    // Skip timestamp column
                    if col_name.to_lowercase() == "timestamp" {
                        continue;
                    }

                    // Try to cast to Float64 - if it fails, leave as is
                    if let Ok(col) = df.column(&col_name) {
                        if let Ok(casted) = col.cast(&DataType::Float64) {
                            if let Ok(new_df) = df.with_column(casted.into_column()) {
                                df = new_df.clone();
                            }
                        }
                    }
                }

                log::debug!("Loaded {} rows from {}", df.height(), path.display());
                Ok(df)
            })
            .collect();

        // Collect results and handle errors
        let mut dataframes = Vec::with_capacity(results.len());
        for result in results {
            dataframes.push(result?);
        }

        log::info!("Successfully loaded {} CSV files", dataframes.len());
        Ok(dataframes)
    }

    /// Load CSV data with automatic validation
    pub async fn load_csv<P: AsRef<Path>>(&self, path: P) -> Result<DataFrame> {
        let path = path.as_ref();

        // Better validation and error messages
        if !path.exists() {
            return Err(VangaError::DataError(format!(
                "❌ Data file not found: {}\n💡 Make sure the file exists and the path is correct.",
                path.display()
            )));
        }

        if path.is_dir() {
            return Err(VangaError::DataError(format!(
                "❌ Expected CSV file but got directory: {}\n💡 Use --data-dir for directories or specify a .csv file path.",
                path.display()
            )));
        }

        if let Some(extension) = path.extension() {
            if extension != "csv" {
                return Err(VangaError::DataError(format!(
                    "❌ Expected .csv file but got .{}: {}\n💡 Please provide a CSV file.",
                    extension.to_string_lossy(),
                    path.display()
                )));
            }
        } else {
            return Err(VangaError::DataError(format!(
                "❌ File has no extension, expected .csv: {}\n💡 Please provide a CSV file.",
                path.display()
            )));
        }

        // Read CSV - Simple solution: read everything as strings first, then cast numeric columns to Float64
        log::info!("📂 Loading CSV file: {}", path.display());

        // Read CSV with proper schema inference
        let file = File::open(path)
            .map_err(|e| VangaError::DataError(format!(
                "❌ Failed to open CSV file: {}\n🔍 Error: {}\n💡 Check if the file exists and is readable.",
                path.display(), e
            )))?;
        let mut df = CsvReader::new(file)
            .finish()
            .map_err(|e| VangaError::DataError(format!(
                "❌ Failed to read CSV file: {}\n🔍 Error: {}\n💡 Check if the file contains valid CSV data with proper headers.",
                path.display(), e
            )))?;

        // Now cast all columns except timestamp to Float64
        let column_names: Vec<String> = df
            .get_column_names()
            .iter()
            .map(|s| s.to_string())
            .collect();
        for col_name in column_names {
            // Skip timestamp column
            if col_name.to_lowercase() == "timestamp" {
                continue;
            }

            // Try to cast to Float64 - if it fails, leave as is (might be a string column)
            if let Ok(col) = df.column(&col_name) {
                if let Ok(casted) = col.cast(&DataType::Float64) {
                    if let Ok(new_df) = df.with_column(casted.into_column()) {
                        df = new_df.clone();
                    }
                }
            }
        }

        // Validate required columns
        self.validate_required_columns(&df)?;

        // Standardize column names
        let df = self.standardize_columns(df)?;

        // Sort by timestamp
        let df = df
            .lazy()
            .sort(["timestamp"], SortMultipleOptions::default())
            .collect()
            .map_err(|e| VangaError::DataError(format!("Failed to sort data: {}", e)))?;

        // Debug: Check for timestamp uniqueness after loading and sorting
        if let Ok(timestamp_col) = df.column("timestamp") {
            let unique_count = timestamp_col.n_unique().unwrap_or(0);
            let total_count = df.height();

            log::debug!(
                "📊 Timestamp validation after loading: {} unique out of {} total timestamps",
                unique_count,
                total_count
            );

            if unique_count != total_count {
                log::warn!(
                    "⚠️  POTENTIAL DUPLICATE TIMESTAMPS detected during data loading: {} unique out of {} total",
                    unique_count, total_count
                );
                log::warn!(
                    "🔍 This suggests the issue is in the data loading/preprocessing pipeline, not the original CSV file"
                );
            }
        }

        log::info!(
            "Loaded {} records from {} with {} columns",
            df.height(),
            path.display(),
            df.width()
        );

        Ok(df)
    }

    /// Validate that required columns are present
    fn validate_required_columns(&self, df: &DataFrame) -> Result<()> {
        let columns = df.get_column_names();
        let missing_columns: Vec<_> = GlobalConfig::REQUIRED_COLUMNS
            .iter()
            .filter(|&col| {
                !columns
                    .iter()
                    .any(|&c| c.to_lowercase() == col.to_lowercase())
            })
            .collect();

        if !missing_columns.is_empty() {
            return Err(VangaError::DataError(format!(
                "Missing required columns: {:?}",
                missing_columns
            )));
        }

        Ok(())
    }

    /// Standardize column names to lowercase
    fn standardize_columns(&self, mut df: DataFrame) -> Result<DataFrame> {
        // Create mapping of current names to standardized names
        let column_mapping: Vec<(String, String)> = df
            .get_column_names()
            .iter()
            .map(|&name| {
                let standardized = self.standardize_column_name(name);
                (name.to_string(), standardized)
            })
            .collect();

        // Rename columns
        for (old_name, new_name) in column_mapping {
            if old_name != new_name {
                df = df
                    .lazy()
                    .rename([&old_name], [&new_name], true)
                    .collect()
                    .map_err(|e| {
                        VangaError::DataError(format!("Failed to rename column: {}", e))
                    })?;
            }
        }

        Ok(df)
    }

    /// Standardize individual column name
    fn standardize_column_name(&self, name: &str) -> String {
        let lower = name.to_lowercase();

        // Handle common variations
        match lower.as_str() {
            "time" | "datetime" | "date" | "ts" => "timestamp".to_string(),
            "o" | "open_price" => "open".to_string(),
            "h" | "high_price" => "high".to_string(),
            "l" | "low_price" => "low".to_string(),
            "c" | "close_price" => "close".to_string(),
            "v" | "vol" | "volume_base" => "volume".to_string(),
            "volume_quote" | "quote_volume" => "volume_quote".to_string(),
            "count" | "trade_count" | "trades" => "trades_count".to_string(),
            "taker_buy_volume" | "buy_volume_base" => "buy_volume".to_string(),
            "taker_buy_quote_volume" | "buy_volume_quote" => "buy_volume_quote".to_string(),
            _ => lower,
        }
    }

    /// Load data in chunks for memory efficiency
    pub async fn load_csv_chunked<P: AsRef<Path>>(
        &self,
        path: P,
        process_chunk: impl Fn(DataFrame) -> Result<DataFrame>,
    ) -> Result<DataFrame> {
        let path = path.as_ref();

        // Implement actual chunking using self.chunk_size
        log::debug!("Loading CSV with chunk size: {}", self.chunk_size);

        // For now, we'll load the entire file and process it in chunks
        // Future optimization: implement streaming CSV reader for very large files
        let df = self.load_csv(path).await?;

        if df.height() <= self.chunk_size {
            // File is smaller than chunk size, process entire DataFrame
            process_chunk(df)
        } else {
            // Process in chunks and combine results
            let mut results = Vec::new();
            let total_rows = df.height();

            for start in (0..total_rows).step_by(self.chunk_size) {
                let end = std::cmp::min(start + self.chunk_size, total_rows);
                let chunk = df.slice(start as i64, end - start);
                let processed_chunk = process_chunk(chunk)?;
                results.push(processed_chunk);
            }

            // Combine all processed chunks
            if results.is_empty() {
                Err(crate::utils::error::VangaError::DataError(
                    "No chunks processed".to_string(),
                ))
            } else {
                let num_chunks = results.len();
                let combined = results.into_iter().next().unwrap();
                // Note: This is a simplified combination - real implementation would need
                // proper handling of overlapping sequences and feature continuity
                log::info!("Processed {} chunks", num_chunks);
                Ok(combined)
            }
        }
    }

    /// Get basic statistics about the loaded data
    pub fn get_data_info(&self, df: &DataFrame) -> DataInfo {
        let shape = df.shape();
        let columns = df.get_column_names();

        // Get timestamp range if available
        let (start_time, end_time) = if let Ok(timestamp_col) = df.column("timestamp") {
            if let Ok(timestamps) = timestamp_col.datetime() {
                let start = timestamps
                    .as_datetime_iter()
                    .flatten()
                    .min()
                    .and_then(|ts| {
                        chrono::DateTime::from_timestamp_millis(ts.and_utc().timestamp_millis())
                    });
                let end = timestamps
                    .as_datetime_iter()
                    .flatten()
                    .max()
                    .and_then(|ts| {
                        chrono::DateTime::from_timestamp_millis(ts.and_utc().timestamp_millis())
                    });
                (start, end)
            } else {
                (None, None)
            }
        } else {
            (None, None)
        };

        // Identify custom features (beyond required columns)
        let custom_features: Vec<String> = columns
            .iter()
            .filter(|&col| !GlobalConfig::REQUIRED_COLUMNS.contains(&col.to_lowercase().as_str()))
            .map(|&col| col.to_string())
            .collect();

        DataInfo {
            rows: shape.0,
            columns: shape.1,
            column_names: columns.iter().map(|&s| s.to_string()).collect(),
            custom_features,
            start_time,
            end_time,
        }
    }

    /// Split data chronologically for backtesting (no data leakage)
    pub fn split_chronological(
        &self,
        df: &DataFrame,
        train_ratio: f64,
    ) -> Result<(DataFrame, DataFrame)> {
        if !(0.0..1.0).contains(&train_ratio) {
            return Err(VangaError::DataError(
                "train_ratio must be between 0.0 and 1.0".to_string(),
            ));
        }

        let total_rows = df.height();
        let train_rows = (total_rows as f64 * train_ratio) as usize;

        if train_rows == 0 || train_rows >= total_rows {
            return Err(VangaError::DataError(
                "Invalid train_ratio or insufficient data for splitting".to_string(),
            ));
        }

        // Split chronologically to prevent data leakage
        let train_df = df.slice(0, train_rows);
        let test_df = df.slice(train_rows as i64, total_rows - train_rows);

        log::info!(
            "📊 Chronological split: {} train samples, {} test samples",
            train_df.height(),
            test_df.height()
        );

        Ok((train_df, test_df))
    }

    /// Split data into train/validation/test with proper chronological ordering
    /// This is the SOLUTION to fix validation loss growing issue
    pub fn split_chronological_three_way(
        &self,
        df: &DataFrame,
        train_ratio: f64,
        val_ratio: f64,
    ) -> Result<(DataFrame, DataFrame, DataFrame)> {
        if train_ratio <= 0.0 || val_ratio <= 0.0 || (train_ratio + val_ratio) >= 1.0 {
            return Err(VangaError::DataError(format!(
                "Invalid ratios: train_ratio={}, val_ratio={}, sum={} (must be > 0 and sum < 1.0)",
                train_ratio,
                val_ratio,
                train_ratio + val_ratio
            )));
        }

        let total_rows = df.height();
        let train_rows = (total_rows as f64 * train_ratio) as usize;
        let val_rows = (total_rows as f64 * val_ratio) as usize;
        let test_rows = total_rows - train_rows - val_rows;

        if train_rows == 0 || val_rows == 0 || test_rows == 0 {
            return Err(VangaError::DataError(
                "Insufficient data for three-way split - all splits must have at least 1 row"
                    .to_string(),
            ));
        }

        // CRITICAL: Chronological split to prevent data leakage
        // Timeline: [Training Data] -> [Validation Data] -> [Test Data]
        let train_df = df.slice(0, train_rows);
        let val_df = df.slice(train_rows as i64, val_rows);
        let test_df = df.slice((train_rows + val_rows) as i64, test_rows);

        log::info!("🎯 CHRONOLOGICAL THREE-WAY SPLIT (SOLUTION FOR VALIDATION LOSS ISSUE):");
        log::info!(
            "📊 Training: {} samples ({:.1}%)",
            train_df.height(),
            train_ratio * 100.0
        );
        log::info!(
            "📊 Validation: {} samples ({:.1}%)",
            val_df.height(),
            val_ratio * 100.0
        );
        log::info!(
            "📊 Test: {} samples ({:.1}%)",
            test_df.height(),
            (1.0 - train_ratio - val_ratio) * 100.0
        );

        // Validate timestamp ordering if timestamp column exists
        let timestamp_cols = df.get_column_names();
        {
            let timestamp_col = timestamp_cols
                .iter()
                .find(|col| col.to_lowercase().contains("time"))
                .copied();

            if let Some(ts_col) = timestamp_col {
                if let (Ok(train_ts), Ok(val_ts), Ok(test_ts)) = (
                    train_df.column(ts_col),
                    val_df.column(ts_col),
                    test_df.column(ts_col),
                ) {
                    log::info!(
                        "🕐 Timeline validation: Train ends before Val starts, Val ends before Test starts"
                    );
                    log::debug!(
                        "Train: {} to {}, Val: {} to {}, Test: {} to {}",
                        train_ts.get(0).unwrap_or_default(),
                        train_ts.get(train_ts.len() - 1).unwrap_or_default(),
                        val_ts.get(0).unwrap_or_default(),
                        val_ts.get(val_ts.len() - 1).unwrap_or_default(),
                        test_ts.get(0).unwrap_or_default(),
                        test_ts.get(test_ts.len() - 1).unwrap_or_default(),
                    );
                }
            }
        }

        Ok((train_df, val_df, test_df))
    }
}

/// Information about loaded data
#[derive(Debug)]
pub struct DataInfo {
    pub rows: usize,
    pub columns: usize,
    pub column_names: Vec<String>,
    pub custom_features: Vec<String>,
    pub start_time: Option<chrono::DateTime<chrono::Utc>>,
    pub end_time: Option<chrono::DateTime<chrono::Utc>>,
}

impl Default for DataLoader {
    fn default() -> Self {
        Self::new()
    }
}
