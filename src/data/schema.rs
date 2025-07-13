use crate::config::GlobalConfig;
use crate::utils::error::{Result, VangaError};
use polars::prelude::*;
use std::ops::BitAnd;

/// Schema validation for cryptocurrency data
pub struct CryptoDataSchema;

impl CryptoDataSchema {
    /// Validate that the DataFrame contains required columns with correct types
    pub fn validate(df: &DataFrame) -> Result<()> {
        // Check required columns exist
        Self::validate_required_columns(df)?;

        // Validate data types
        Self::validate_data_types(df)?;

        // Validate data quality
        Self::validate_data_quality(df)?;

        Ok(())
    }

    /// Validate required columns are present
    fn validate_required_columns(df: &DataFrame) -> Result<()> {
        let columns = df.get_column_names();
        let missing_columns: Vec<_> = GlobalConfig::REQUIRED_COLUMNS
            .iter()
            .filter(|&col| !columns.contains(col))
            .collect();

        if !missing_columns.is_empty() {
            return Err(VangaError::DataValidation(
                DataValidationError::MissingColumns(
                    missing_columns.iter().map(|&s| s.to_string()).collect(),
                ),
            ));
        }

        Ok(())
    }

    /// Validate data types are appropriate
    fn validate_data_types(df: &DataFrame) -> Result<()> {
        let schema = df.schema();

        // Timestamp should be datetime or string (parseable to datetime)
        if let Some(timestamp_dtype) = schema.get("timestamp") {
            match timestamp_dtype {
                DataType::Datetime(_, _) | DataType::Utf8 | DataType::Int64 | DataType::UInt64 => {}
                _ => {
                    return Err(VangaError::DataValidation(
                        DataValidationError::InvalidDataType {
                            column: "timestamp".to_string(),
                            expected: "datetime, string, or integer".to_string(),
                            found: format!("{:?}", timestamp_dtype),
                        },
                    ))
                }
            }
        }

        // OHLCV columns should be numeric
        for &col in &["open", "high", "low", "close", "volume"] {
            if let Some(dtype) = schema.get(col) {
                if !dtype.is_numeric() {
                    return Err(VangaError::DataValidation(
                        DataValidationError::InvalidDataType {
                            column: col.to_string(),
                            expected: "numeric".to_string(),
                            found: format!("{:?}", dtype),
                        },
                    ));
                }
            }
        }

        Ok(())
    }

    /// Validate data quality (no negative prices, volume >= 0, etc.)
    fn validate_data_quality(df: &DataFrame) -> Result<()> {
        // Check for negative prices
        for &price_col in &["open", "high", "low", "close"] {
            if let Ok(col) = df.column(price_col) {
                if let Ok(series) = col.cast(&DataType::Float64) {
                    if let Some(min_val) = series.min::<f64>() {
                        if min_val < 0.0 {
                            return Err(VangaError::DataValidation(
                                DataValidationError::InvalidData {
                                    column: price_col.to_string(),
                                    issue: "Negative prices detected".to_string(),
                                },
                            ));
                        }
                    }
                }
            }
        }

        // Check for negative volume
        if let Ok(volume_col) = df.column("volume") {
            if let Ok(series) = volume_col.cast(&DataType::Float64) {
                if let Some(min_val) = series.min::<f64>() {
                    if min_val < 0.0 {
                        return Err(VangaError::DataValidation(
                            DataValidationError::InvalidData {
                                column: "volume".to_string(),
                                issue: "Negative volume detected".to_string(),
                            },
                        ));
                    }
                }
            }
        }

        // Check OHLC relationship (High >= Low, High >= Open, High >= Close, Low <= Open, Low <= Close)
        Self::validate_ohlc_relationships(df)?;

        // Check for duplicate timestamps
        log::debug!("🔍 Starting timestamp uniqueness validation...");
        Self::validate_unique_timestamps(df)?;
        log::debug!("✅ Timestamp uniqueness validation passed");

        Ok(())
    }

    /// Validate OHLC price relationships
    fn validate_ohlc_relationships(df: &DataFrame) -> Result<()> {
        let high = df.column("high")?.cast(&DataType::Float64)?;
        let low = df.column("low")?.cast(&DataType::Float64)?;
        let open = df.column("open")?.cast(&DataType::Float64)?;
        let close = df.column("close")?.cast(&DataType::Float64)?;

        // High should be >= all other prices
        let high_ge_low = high.gt_eq(&low)?;
        let high_ge_open = high.gt_eq(&open)?;
        let high_ge_close = high.gt_eq(&close)?;

        // Low should be <= all other prices
        let low_le_open = low.lt_eq(&open)?;
        let low_le_close = low.lt_eq(&close)?;

        // Check if all conditions are satisfied
        let all_conditions = high_ge_low
            .bitand(high_ge_open)
            .bitand(high_ge_close)
            .bitand(low_le_open)
            .bitand(low_le_close);

        // Check for violations (this is just a warning, not an error)
        if let Some(sum) = all_conditions.sum() {
            let total_count = df.height() as u32;
            let violations = total_count - sum;
            if violations > 0 {
                log::warn!(
                    "Found {} rows with invalid OHLC relationships out of {} total rows",
                    violations,
                    total_count
                );
            }
        }

        Ok(())
    }

    /// Validate timestamp uniqueness
    fn validate_unique_timestamps(df: &DataFrame) -> Result<()> {
        let timestamp_col = df.column("timestamp")?;
        let unique_count = timestamp_col.n_unique().map_err(|e| {
            VangaError::DataValidation(DataValidationError::InvalidData {
                column: "timestamp".to_string(),
                issue: format!("Failed to count unique timestamps: {}", e),
            })
        })?;

        if unique_count != df.height() {
            // Find and report the actual duplicate timestamps with row numbers
            let mut duplicate_info = Vec::new();

            // Create a temporary DataFrame with row indices for duplicate detection
            let df_with_index = df
                .clone()
                .lazy()
                .with_row_count("row_index", Some(1)) // 1-based indexing
                .collect()
                .map_err(|e| {
                    VangaError::DataValidation(DataValidationError::InvalidData {
                        column: "timestamp".to_string(),
                        issue: format!("Failed to add row indices for duplicate detection: {}", e),
                    })
                })?;

            // Group by timestamp and find groups with more than one row
            let duplicates = df_with_index
                .lazy()
                .group_by([col("timestamp")])
                .agg([
                    col("row_index").alias("rows"),
                    col("row_index").count().alias("count"),
                ])
                .filter(col("count").gt(lit(1)))
                .sort("timestamp", SortOptions::default())
                .collect()
                .map_err(|e| {
                    VangaError::DataValidation(DataValidationError::InvalidData {
                        column: "timestamp".to_string(),
                        issue: format!("Failed to detect duplicate timestamps: {}", e),
                    })
                })?;

            // Extract duplicate information
            if duplicates.height() > 0 {
                let timestamp_series = duplicates.column("timestamp")?;
                let rows_series = duplicates.column("rows")?;
                let count_series = duplicates.column("count")?;

                for i in 0..duplicates.height() {
                    let timestamp = timestamp_series.get(i).map_err(|e| {
                        VangaError::DataValidation(DataValidationError::InvalidData {
                            column: "timestamp".to_string(),
                            issue: format!("Failed to get timestamp value: {}", e),
                        })
                    })?;

                    let rows_list = rows_series.get(i).map_err(|e| {
                        VangaError::DataValidation(DataValidationError::InvalidData {
                            column: "timestamp".to_string(),
                            issue: format!("Failed to get row indices: {}", e),
                        })
                    })?;

                    let count = count_series.get(i).map_err(|e| {
                        VangaError::DataValidation(DataValidationError::InvalidData {
                            column: "timestamp".to_string(),
                            issue: format!("Failed to get duplicate count: {}", e),
                        })
                    })?;

                    duplicate_info.push(format!(
                        "  • Timestamp '{}' appears {} times at rows: {}",
                        timestamp, count, rows_list
                    ));
                }
            }

            let detailed_message = if duplicate_info.is_empty() {
                format!(
                    "❌ DUPLICATE TIMESTAMPS DETECTED\n\
                     📊 Statistics: {} unique timestamps out of {} total rows\n\
                     🔍 Unable to extract specific duplicate details (this may indicate a data processing bug)",
                    unique_count, df.height()
                )
            } else {
                format!(
                    "❌ DUPLICATE TIMESTAMPS DETECTED\n\
                     📊 Statistics: {} unique timestamps out of {} total rows\n\
                     🔍 Duplicate timestamps found:\n{}\n\
                     💡 This error may be caused by:\n\
                     • Data processing bug introducing duplicates during loading/preprocessing\n\
                     • Timestamp parsing/conversion issues\n\
                     • Sorting or merging operations creating duplicates\n\
                     • Check the data loading pipeline in src/data/loader.rs",
                    unique_count,
                    df.height(),
                    duplicate_info.join("\n")
                )
            };

            return Err(VangaError::DataValidation(
                DataValidationError::InvalidData {
                    column: "timestamp".to_string(),
                    issue: detailed_message,
                },
            ));
        }

        Ok(())
    }

    /// Get schema information for the DataFrame
    pub fn get_schema_info(df: &DataFrame) -> SchemaInfo {
        let schema = df.schema();
        let shape = df.shape();

        let required_columns: Vec<ColumnInfo> = GlobalConfig::REQUIRED_COLUMNS
            .iter()
            .map(|&col| {
                let dtype = schema.get(col).cloned();
                let present = dtype.is_some();
                ColumnInfo {
                    name: col.to_string(),
                    data_type: dtype,
                    present,
                    null_count: if present {
                        df.column(col).map(|c| c.null_count()).unwrap_or(0)
                    } else {
                        0
                    },
                }
            })
            .collect();

        let custom_columns: Vec<ColumnInfo> = schema
            .iter()
            .filter(|(name, _)| !GlobalConfig::REQUIRED_COLUMNS.contains(&name.as_str()))
            .map(|(name, dtype)| {
                let null_count = df.column(name).map(|c| c.null_count()).unwrap_or(0);
                ColumnInfo {
                    name: name.to_string(),
                    data_type: Some(dtype.clone()),
                    present: true,
                    null_count,
                }
            })
            .collect();

        SchemaInfo {
            total_rows: shape.0,
            total_columns: shape.1,
            required_columns,
            custom_columns,
        }
    }
}

/// Data validation errors
#[derive(Debug, thiserror::Error)]
pub enum DataValidationError {
    #[error("Missing required columns: {0:?}")]
    MissingColumns(Vec<String>),

    #[error("Invalid data type for column '{column}': expected {expected}, found {found}")]
    InvalidDataType {
        column: String,
        expected: String,
        found: String,
    },

    #[error("Invalid data in column '{column}': {issue}")]
    InvalidData { column: String, issue: String },
}

/// Information about a column
#[derive(Debug)]
pub struct ColumnInfo {
    pub name: String,
    pub data_type: Option<DataType>,
    pub present: bool,
    pub null_count: usize,
}

/// Schema information for the dataset
#[derive(Debug)]
pub struct SchemaInfo {
    pub total_rows: usize,
    pub total_columns: usize,
    pub required_columns: Vec<ColumnInfo>,
    pub custom_columns: Vec<ColumnInfo>,
}
