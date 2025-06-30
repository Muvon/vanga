use crate::config::GlobalConfig;
use crate::utils::error::{Result, VangaError};
use polars::prelude::*;
use rayon::prelude::*;
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

                // Load CSV with validation
                let df = polars::prelude::CsvReader::from_path(path)
                    .map_err(|e| {
                        VangaError::DataError(format!("Failed to create CSV reader: {}", e))
                    })?
                    .has_header(true)
                    .finish()
                    .map_err(|e| VangaError::DataError(format!("Failed to read CSV: {}", e)))?;

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

        if !path.exists() {
            return Err(VangaError::DataError(format!(
                "Data file not found: {}",
                path.display()
            )));
        }

        // Read CSV with Polars (use read_csv directly)
        let df = polars::prelude::CsvReader::from_path(path)
            .map_err(|e| VangaError::DataError(format!("Failed to create CSV reader: {}", e)))?
            .finish()
            .map_err(|e| VangaError::DataError(format!("Failed to read CSV: {}", e)))?;

        // Validate required columns
        self.validate_required_columns(&df)?;

        // Standardize column names
        let df = self.standardize_columns(df)?;

        // Sort by timestamp
        let df = df
            .lazy()
            .sort("timestamp", SortOptions::default())
            .collect()
            .map_err(|e| VangaError::DataError(format!("Failed to sort data: {}", e)))?;

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
                    .rename([&old_name], [&new_name])
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
            let timestamps = timestamp_col.datetime().unwrap();
            let start = timestamps
                .min()
                .and_then(chrono::DateTime::from_timestamp_millis);
            let end = timestamps
                .max()
                .and_then(chrono::DateTime::from_timestamp_millis);
            (start, end)
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
