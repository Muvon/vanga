// Sequence generator for LSTM training and prediction
use crate::data::{DataMetadata, PreparedData, PreparedPredictionData};
use crate::utils::error::{Result, VangaError};
use chrono::Utc;
use ndarray::{s, Array2, Array3, Axis};
use polars::prelude::*;
use rayon::prelude::*;

pub struct SequenceGenerator {
    // Sequence generation logic - overlap controlled via DataConfig
}

impl Default for SequenceGenerator {
    fn default() -> Self {
        Self::new()
    }
}

impl SequenceGenerator {
    pub fn new() -> Self {
        Self {}
    }

    pub async fn generate_training_sequences(
        &self,
        df: DataFrame, // RAW data with features, NOT pre-normalized
        horizons: &[String],
        model_config: &crate::config::ModelConfig,
        data_config: &crate::config::training::DataConfig,
        feature_config: &crate::config::FeatureConfig,
    ) -> Result<PreparedData> {
        self.generate_training_sequences_with_adaptive_params(
            df,
            horizons,
            model_config,
            data_config,
            feature_config,
            None,
        )
        .await
    }

    pub async fn generate_training_sequences_with_adaptive_params(
        &self,
        df: DataFrame, // RAW data with features, NOT pre-normalized
        horizons: &[String],
        model_config: &crate::config::ModelConfig,
        data_config: &crate::config::training::DataConfig,
        feature_config: &crate::config::FeatureConfig,
        adaptive_params: Option<&crate::targets::adaptive_parameters::AdaptiveTargetParameters>,
    ) -> Result<PreparedData> {
        log::info!(
            "Generating training sequences for LSTM with {} horizons...",
            horizons.len()
        );

        // Validate DataFrame for multi-horizon processing
        self.validate_dataframe_for_horizons(&df, horizons)?;

        // Get basic info
        let total_records = df.height();
        let sequence_length = match &model_config.sequence_length {
            crate::config::model::SequenceLengthConfig::Fixed(len) => *len as usize,
            crate::config::model::SequenceLengthConfig::Auto {
                min_length,
                max_length: _,
            } => *min_length as usize,
            crate::config::model::SequenceLengthConfig::Adaptive => 60,
        };

        // NEW: Validate feature window requirements
        self.validate_feature_window_requirements(
            total_records,
            sequence_length,
            horizons,
            feature_config,
        )?;

        // Extract feature columns (exclude timestamp and target columns)
        let feature_columns: Vec<String> = df
            .get_column_names()
            .iter()
            .filter(|&col| {
                ![
                    "timestamp",
                    "target_price",
                    "target_direction",
                    "target_volatility",
                ]
                .contains(col)
            })
            .map(|s| s.to_string())
            .collect();

        let feature_count = feature_columns.len();

        // Extract feature data as matrix
        let feature_data = self.extract_feature_matrix(&df, &feature_columns)?;

        // Generate sequences using per-sequence normalization (CORRECT approach)
        let (sequences, targets, sequence_indices) = self
            .create_sliding_windows_with_normalization(
                &feature_data,
                sequence_length,
                horizons,
                &df,
                data_config,
                model_config,
                adaptive_params,
            )
            .await?;

        log::info!("✅ Using per-sequence normalization (each sequence normalized independently)");

        // Create metadata
        let metadata = DataMetadata {
            symbol: "".to_string(), // Will be set by caller
            start_time: Utc::now(),
            end_time: Utc::now(),
            total_records,
            feature_count,
            sequence_length,
            horizons: horizons.to_vec(),
        };

        log::info!(
            "Generated {} sequences with {} features",
            sequences.len_of(Axis(0)),
            feature_count
        );

        // Final data utilization and efficiency summary
        log::info!("🏁 FINAL DATA PIPELINE SUMMARY:");
        log::info!(
            "   • Total Data Processed: {} rows → {} sequences → {} targets",
            total_records,
            sequences.len_of(Axis(0)),
            targets.data_length
        );
        log::info!(
            "   • Memory Efficiency: {:.1}% data utilization",
            (sequences.len_of(Axis(0)) as f64 / total_records as f64) * 100.0
        );
        log::info!(
            "   • Feature Engineering: {} features per sequence",
            feature_count
        );
        log::info!(
            "   • Multi-Horizon Targets: {} prediction horizons",
            horizons.len()
        );
        log::info!("   • Data Quality: Normalized, validated, ready for training");

        Ok(PreparedData {
            sequences,
            targets,
            feature_names: feature_columns,
            metadata,
            sequence_indices,
        })
    }

    pub async fn generate_prediction_sequences(
        &self,
        df: DataFrame,
        symbol: &str,
        model_config: &crate::config::ModelConfig,
    ) -> Result<PreparedPredictionData> {
        log::info!("Generating prediction sequences for symbol: {}", symbol);

        // Extract sequence length from config
        let sequence_length = match &model_config.sequence_length {
            crate::config::model::SequenceLengthConfig::Fixed(len) => *len as usize,
            crate::config::model::SequenceLengthConfig::Auto {
                min_length,
                max_length: _,
            } => *min_length as usize,
            crate::config::model::SequenceLengthConfig::Adaptive => 60,
        };

        // Get basic info
        let num_rows = df.height();
        let num_cols = df.width();

        log::debug!("Input data shape: {} rows, {} columns", num_rows, num_cols);
        log::debug!("Using sequence length: {}", sequence_length);

        // Validate feature window requirements for prediction sequences
        // Note: This is a secondary check - primary validation should happen in prepare_prediction_data
        if num_rows < sequence_length {
            return Err(VangaError::DataError(format!(
                "Not enough data for prediction sequences: {} rows < {} sequence_length required\n\
                 This suggests the data wasn't properly validated in prepare_prediction_data()",
                num_rows, sequence_length
            )));
        }

        // Extract feature columns (exclude timestamp and target columns)
        let exclude_columns = ["timestamp", "price_level", "direction", "volatility"];
        let feature_columns: Vec<String> = df
            .get_column_names()
            .iter()
            .filter(|&col| !exclude_columns.contains(col))
            .map(|&col| col.to_string())
            .collect();

        log::debug!(
            "Using {} feature columns for prediction",
            feature_columns.len()
        );

        // Extract feature data as matrix from original DataFrame (before normalization)
        let start_idx = num_rows.saturating_sub(sequence_length);

        // FIXED: Ensure prediction pipeline uses same per-sequence normalization as training
        // Extract the last sequence_length rows and normalize only that sequence
        let sequence_df = df.slice(start_idx as i64, sequence_length);

        // Extract OHLC data from the sequence for ATR calculation - REQUIRED!
        let sequence_ohlc = self.extract_ohlc_from_sequence(&sequence_df)
            .map_err(|e| VangaError::DataError(format!(
                "FATAL: Cannot extract OHLC data from sequence for order generation. This is required for proper ATR calculation and sequence-aware orders: {}", e
            )))?;

        let normalized_sequence = self.normalize_sequence_window(&sequence_df)?;
        let sequences =
            self.create_prediction_sequences(&normalized_sequence, 0, sequence_length)?;

        // Create metadata - use actual horizons from model if available
        // Note: model_config is ModelConfig, not Option, but it doesn't contain training horizons
        // For now, we'll use default "1h" until we properly store training config in models
        let horizons = vec!["1h".to_string()]; // TODO: Get actual trained horizons from model

        let metadata = crate::data::DataMetadata {
            symbol: symbol.to_string(),
            start_time: chrono::Utc::now(),
            end_time: chrono::Utc::now(),
            total_records: num_rows,
            feature_count: feature_columns.len(),
            sequence_length,
            horizons, // Use actual horizons instead of hardcoded "1h"
        };

        Ok(PreparedPredictionData {
            sequences,
            feature_names: feature_columns,
            metadata,
            sequence_ohlc: Some(sequence_ohlc),
        })
    }

    // Helper methods for sequence generation
    fn extract_feature_matrix(
        &self,
        df: &DataFrame,
        feature_columns: &[String],
    ) -> Result<Array2<f64>> {
        let rows = df.height();
        let cols = feature_columns.len();
        let mut matrix = Array2::zeros((rows, cols));

        for (col_idx, col_name) in feature_columns.iter().enumerate() {
            let series = df.column(col_name).map_err(|e| {
                VangaError::DataError(format!("Column '{}' not found: {}", col_name, e))
            })?;

            let values: Vec<f64> = series
                .f64()
                .map_err(|e| {
                    VangaError::DataError(format!(
                        "Failed to convert column '{}' to f64: {}",
                        col_name, e
                    ))
                })?
                .into_no_null_iter()
                .collect();

            for (row_idx, &value) in values.iter().enumerate() {
                if row_idx < rows {
                    matrix[[row_idx, col_idx]] = value;
                }
            }
        }

        Ok(matrix)
    }

    /// Create prediction sequences from feature data
    fn create_prediction_sequences(
        &self,
        normalized_df: &DataFrame,
        start_idx: usize,
        sequence_length: usize,
    ) -> Result<Array3<f64>> {
        // Extract feature columns (exclude timestamp)
        let feature_columns: Vec<String> = normalized_df
            .get_column_names()
            .iter()
            .filter(|&col| *col != "timestamp")
            .map(|s| s.to_string())
            .collect();

        let feature_data = self.extract_feature_matrix(normalized_df, &feature_columns)?;
        let num_features = feature_data.ncols();
        let available_rows = feature_data.nrows() - start_idx;

        if available_rows < sequence_length {
            return Err(VangaError::DataError(format!(
                "Not enough data for sequence: {} available < {} required",
                available_rows, sequence_length
            )));
        }

        // Create single sequence for prediction (batch size = 1)
        let mut sequences = Array3::zeros((1, sequence_length, num_features));

        for seq_idx in 0..sequence_length {
            let data_idx = start_idx + seq_idx;
            for feature_idx in 0..num_features {
                sequences[[0, seq_idx, feature_idx]] = feature_data[[data_idx, feature_idx]];
            }
        }

        Ok(sequences)
    }

    /// Validate DataFrame structure for multi-horizon processing
    fn validate_dataframe_for_horizons(&self, df: &DataFrame, horizons: &[String]) -> Result<()> {
        // Check minimum data requirements for all horizons
        let min_required_rows = horizons
            .iter()
            .map(|h| crate::utils::parser::parse_horizon_to_steps(h).unwrap_or(1))
            .max()
            .unwrap_or(1)
            * 2; // At least 2x the largest horizon for reliable training

        if df.height() < min_required_rows {
            return Err(crate::utils::error::VangaError::DataError(
                format!(
                    "Insufficient data for multi-horizon processing: need {} rows, got {}. Largest horizon requires {} steps.",
                    min_required_rows, df.height(), min_required_rows / 2
                )
            ));
        }

        // Validate required columns exist
        let required_columns = ["open", "close", "high", "low", "volume"];
        for col in required_columns {
            if df.column(col).is_err() {
                return Err(crate::utils::error::VangaError::DataError(format!(
                    "Missing required column '{}' for multi-horizon sequence generation",
                    col
                )));
            }
        }

        // Check for excessive missing values that could affect horizon alignment
        for column in df.get_column_names() {
            if let Ok(series) = df.column(column) {
                let null_count = series.null_count();
                let null_ratio = null_count as f64 / df.height() as f64;

                if null_ratio > 0.1 {
                    // More than 10% missing values
                    log::warn!(
                        "Column '{}' has {:.1}% missing values, may affect multi-horizon alignment",
                        column,
                        null_ratio * 100.0
                    );
                }
            }
        }

        log::info!(
            "DataFrame validation passed: {} rows, {} horizons, {} columns",
            df.height(),
            horizons.len(),
            df.width()
        );

        Ok(())
    }

    /// Validate feature window requirements for training data using actual feature configuration
    fn validate_feature_window_requirements(
        &self,
        total_records: usize,
        sequence_length: usize,
        horizons: &[String],
        feature_config: &crate::config::FeatureConfig,
    ) -> Result<()> {
        // Use proper feature window calculation
        let requirements = crate::utils::feature_window::calculate_min_data_requirements(
            feature_config,
            sequence_length,
            horizons,
        )?;

        // Validate using the proper requirements
        requirements.validate(total_records)?;

        // Log summary for debugging
        requirements.log_summary(total_records, "TRAINING DATA VALIDATION");

        Ok(())
    }

    /// Normalize a sequence window using only data within that window
    /// This is the CORRECT approach for time series ML - each sequence is self-contained
    /// 🚀 OPTIMIZED: Parallel column processing for maximum performance
    fn normalize_sequence_window(&self, window_df: &DataFrame) -> Result<DataFrame> {
        let column_names: Vec<String> = window_df
            .get_column_names()
            .iter()
            .map(|s| s.to_string())
            .collect();

        // 🚀 PARALLEL COLUMN PROCESSING: Process each column independently in parallel
        let normalized_columns: Result<Vec<Series>> = column_names
            .par_iter() // ← PARALLEL PROCESSING: Each column on different thread
            .map(|column_name| {
                if column_name == "timestamp" {
                    // Keep timestamp as-is
                    Ok(window_df.column(column_name)?.clone())
                } else if let Ok(series) = window_df.column(column_name) {
                    if series.dtype().is_numeric() {
                        // Normalize using only THIS WINDOW's data
                        self.normalize_column_in_window(series)
                    } else {
                        Ok(series.clone())
                    }
                } else {
                    Err(VangaError::DataError(format!(
                        "Column '{}' not found",
                        column_name
                    )))
                }
            })
            .collect();

        let normalized_columns = normalized_columns?;

        // Reconstruct DataFrame with normalized columns
        DataFrame::new(normalized_columns).map_err(|e| {
            VangaError::DataError(format!("Failed to create normalized DataFrame: {}", e))
        })
    }

    /// Normalize a single column using only values within the sequence window
    fn normalize_column_in_window(&self, series: &Series) -> Result<Series> {
        if let Ok(float_series) = series.f64() {
            // Since we've already removed initial NaN rows, we should have valid data
            let values: Vec<f64> = float_series
                .into_iter()
                .filter_map(|v| v.filter(|x| x.is_finite()))
                .collect();

            // If no finite values (shouldn't happen after remove_nan_rows), return original
            if values.is_empty() {
                log::warn!(
                    "Column '{}' has no finite values after NaN removal - this shouldn't happen",
                    series.name()
                );
                return Ok(series.clone());
            }

            // If only one unique value, return zeros (normalized constant)
            if values.len() == 1 {
                let zeros: Vec<Option<f64>> = (0..float_series.len()).map(|_| Some(0.0)).collect();
                return Ok(Series::new(series.name(), zeros));
            }

            // Calculate mean and std from finite values only
            let mean = values.iter().sum::<f64>() / values.len() as f64;
            let variance =
                values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / values.len() as f64;
            let std_dev = variance.sqrt();

            // Robust division by zero protection
            let std_dev = if std_dev < 1e-6 {
                1e-6 // Very small but not zero
            } else {
                std_dev
            };

            // Normalize: (x - mean) / std, preserving structure
            let normalized_values: Vec<Option<f64>> = float_series
                .into_iter()
                .map(|v| {
                    match v {
                        Some(x) if x.is_finite() => Some((x - mean) / std_dev),
                        _ => None, // Keep NaN as None
                    }
                })
                .collect();

            Ok(Series::new(series.name(), normalized_values))
        } else {
            Ok(series.clone())
        }
    }

    /// Create sliding windows with per-sequence normalization
    /// Each window: [sequence_length + gap + horizon_steps] normalized together
    /// Returns: (sequences, targets, sequence_indices)
    #[allow(clippy::too_many_arguments)]
    async fn create_sliding_windows_with_normalization(
        &self,
        _feature_data: &Array2<f64>, // Unused - we work with DataFrame directly
        sequence_length: usize,
        horizons: &[String],
        df: &DataFrame,
        data_config: &crate::config::training::DataConfig,
        model_config: &crate::config::ModelConfig,
        adaptive_params: Option<&crate::targets::adaptive_parameters::AdaptiveTargetParameters>,
    ) -> Result<(
        Array3<f64>,
        crate::targets::PreparedTargets,
        Vec<(usize, usize)>,
    )> {
        // Calculate maximum horizon steps
        let max_horizon_steps = horizons
            .iter()
            .map(|h| crate::utils::parser::parse_horizon_to_steps(h).unwrap_or(1))
            .max()
            .unwrap_or(1);

        // FIXED: Use unified step size calculation
        let step_size = crate::utils::sequence_utils::calculate_step_size(
            data_config.sequence_overlap,
            sequence_length,
        );

        // Total window size: sequence + horizon (no gap needed)
        let total_window_size = sequence_length + max_horizon_steps;

        log::info!(
            "🔧 Per-sequence normalization: sequence_length({}) + max_horizon({}) = {} total window",
            sequence_length, max_horizon_steps, total_window_size
        );

        let total_records = df.height();
        if total_records < total_window_size {
            return Err(VangaError::DataError(format!(
                "Insufficient data: need {} records, have {}",
                total_window_size, total_records
            )));
        }

        // FIXED: Calculate sequence indices for proper alignment
        let sequence_indices = crate::utils::sequence_utils::calculate_sequence_indices(
            total_records,
            sequence_length,
            step_size,
            max_horizon_steps,
        )?;

        // FIXED: Validate sequence overlap parameter
        crate::utils::sequence_utils::validate_sequence_overlap(data_config.sequence_overlap)?;

        // FIXED: Log detailed synchronization information
        crate::utils::sequence_utils::log_synchronization_details(
            &sequence_indices,
            step_size,
            sequence_length,
            max_horizon_steps,
            data_config.sequence_overlap,
        );

        log::info!(
            "📊 Sequence generation plan: step_size={}, total_sequences={}, overlap={:.1}%",
            step_size,
            sequence_indices.len(),
            data_config.sequence_overlap * 100.0
        );

        // 🚀 PARALLEL SEQUENCE PROCESSING: Process all sequences in parallel for maximum performance
        let feature_columns: Vec<String> = df
            .get_column_names()
            .iter()
            .filter(|&col| *col != "timestamp")
            .map(|s| s.to_string())
            .collect();

        log::info!(
            "🚀 Starting PARALLEL sequence processing: {} sequences across {} CPU cores",
            sequence_indices.len(),
            rayon::current_num_threads()
        );

        // Start timing for performance measurement
        let start_time = std::time::Instant::now();

        // Generate sequences using parallel processing - each sequence processed independently
        let all_sequences: Result<Vec<Array2<f64>>> = sequence_indices
            .par_iter() // ← PARALLEL PROCESSING: Each sequence on different thread
            .map(|&start_idx| {
                // Extract the complete window (sequence + horizon)
                let window_df = df.slice(start_idx as i64, total_window_size);

                // FIXED: Only normalize the input sequence part, NOT targets
                let sequence_df = window_df.slice(0, sequence_length);
                let normalized_sequence = self.normalize_sequence_window(&sequence_df)?;
                let sequence_matrix =
                    self.extract_feature_matrix(&normalized_sequence, &feature_columns)?;

                Ok(sequence_matrix)
            })
            .collect();

        let all_sequences = all_sequences?;
        let processing_time = start_time.elapsed();

        log::info!(
            "✅ Generated {} sequences with PARALLEL per-sequence normalization using unified step calculation",
            all_sequences.len()
        );
        log::info!(
            "🚀 Performance: Parallel processing completed in {:.2}ms using {} CPU cores ({:.2} sequences/ms)",
            processing_time.as_millis(),
            rayon::current_num_threads(),
            all_sequences.len() as f64 / processing_time.as_millis() as f64
        );

        // Convert sequences to Array3
        let sequences = self.convert_sequences_to_array3(all_sequences)?;

        // FIXED: Generate targets using aligned sequence indices
        let target_config =
            crate::targets::MultiTargetConfig::from_model_config(model_config, horizons.to_vec());

        let target_generator = crate::targets::TargetGenerator::new(target_config);
        let targets = target_generator
            .generate_all_targets_with_adaptive_params(
                df,
                Some(model_config),
                &sequence_indices,
                sequence_length,
                adaptive_params,
            )
            .await?;

        // Create sequence position indices
        let sequence_position_indices: Vec<(usize, usize)> = sequence_indices
            .iter()
            .map(|&start_idx| (start_idx, start_idx + sequence_length))
            .collect();

        Ok((sequences, targets, sequence_position_indices))
    }

    /// Convert list of sequence matrices to Array3
    fn convert_sequences_to_array3(&self, sequences: Vec<Array2<f64>>) -> Result<Array3<f64>> {
        if sequences.is_empty() {
            return Err(VangaError::DataError("No sequences to convert".to_string()));
        }

        let num_sequences = sequences.len();
        let sequence_length = sequences[0].nrows();
        let num_features = sequences[0].ncols();

        let mut array3 = Array3::zeros((num_sequences, sequence_length, num_features));

        for (i, sequence) in sequences.iter().enumerate() {
            array3.slice_mut(s![i, .., ..]).assign(sequence);
        }

        Ok(array3)
    }

    /// Extract OHLC data from sequence DataFrame for ATR calculation
    fn extract_ohlc_from_sequence(
        &self,
        sequence_df: &DataFrame,
    ) -> Result<Vec<crate::data::structures::MarketDataRow>> {
        let mut ohlc_data = Vec::new();

        // Get required columns
        let timestamp_col = sequence_df.column("timestamp").map_err(|_| {
            VangaError::DataError("Missing timestamp column in sequence data".to_string())
        })?;
        let open_col = sequence_df.column("open").map_err(|_| {
            VangaError::DataError("Missing open column in sequence data".to_string())
        })?;
        let high_col = sequence_df.column("high").map_err(|_| {
            VangaError::DataError("Missing high column in sequence data".to_string())
        })?;
        let low_col = sequence_df.column("low").map_err(|_| {
            VangaError::DataError("Missing low column in sequence data".to_string())
        })?;
        let close_col = sequence_df.column("close").map_err(|_| {
            VangaError::DataError("Missing close column in sequence data".to_string())
        })?;
        let volume_col = sequence_df.column("volume").map_err(|_| {
            VangaError::DataError("Missing volume column in sequence data".to_string())
        })?;

        // Extract data row by row
        for i in 0..sequence_df.height() {
            let timestamp = timestamp_col.get(i).map_err(|e| {
                VangaError::DataError(format!("Failed to extract timestamp at row {}: {}", i, e))
            })?;
            let open = open_col.get(i).map_err(|e| {
                VangaError::DataError(format!("Failed to extract open at row {}: {}", i, e))
            })?;
            let high = high_col.get(i).map_err(|e| {
                VangaError::DataError(format!("Failed to extract high at row {}: {}", i, e))
            })?;
            let low = low_col.get(i).map_err(|e| {
                VangaError::DataError(format!("Failed to extract low at row {}: {}", i, e))
            })?;
            let close = close_col.get(i).map_err(|e| {
                VangaError::DataError(format!("Failed to extract close at row {}: {}", i, e))
            })?;
            let volume = volume_col.get(i).map_err(|e| {
                VangaError::DataError(format!("Failed to extract volume at row {}: {}", i, e))
            })?;

            // Convert AnyValue to appropriate types
            let timestamp_i64 = match timestamp {
                polars::prelude::AnyValue::Datetime(dt, _, _) => dt / 1_000_000, // Convert microseconds to seconds
                polars::prelude::AnyValue::Int64(ts) => ts,
                polars::prelude::AnyValue::Utf8(s) => {
                    // Parse ISO timestamp string
                    chrono::DateTime::parse_from_rfc3339(s)
                        .map_err(|e| {
                            VangaError::DataError(format!(
                                "Failed to parse timestamp '{}': {}",
                                s, e
                            ))
                        })?
                        .timestamp()
                }
                _ => {
                    return Err(VangaError::DataError(format!(
                        "Unsupported timestamp type: {:?}",
                        timestamp
                    )))
                }
            };

            let open_f64 = match open {
                polars::prelude::AnyValue::Float64(f) => f,
                polars::prelude::AnyValue::Float32(f) => f as f64,
                polars::prelude::AnyValue::Int64(i) => i as f64,
                _ => {
                    return Err(VangaError::DataError(format!(
                        "Unsupported open price type: {:?}",
                        open
                    )))
                }
            };

            let high_f64 = match high {
                polars::prelude::AnyValue::Float64(f) => f,
                polars::prelude::AnyValue::Float32(f) => f as f64,
                polars::prelude::AnyValue::Int64(i) => i as f64,
                _ => {
                    return Err(VangaError::DataError(format!(
                        "Unsupported high price type: {:?}",
                        high
                    )))
                }
            };

            let low_f64 = match low {
                polars::prelude::AnyValue::Float64(f) => f,
                polars::prelude::AnyValue::Float32(f) => f as f64,
                polars::prelude::AnyValue::Int64(i) => i as f64,
                _ => {
                    return Err(VangaError::DataError(format!(
                        "Unsupported low price type: {:?}",
                        low
                    )))
                }
            };

            let close_f64 = match close {
                polars::prelude::AnyValue::Float64(f) => f,
                polars::prelude::AnyValue::Float32(f) => f as f64,
                polars::prelude::AnyValue::Int64(i) => i as f64,
                _ => {
                    return Err(VangaError::DataError(format!(
                        "Unsupported close price type: {:?}",
                        close
                    )))
                }
            };

            let volume_f64 = match volume {
                polars::prelude::AnyValue::Float64(f) => f,
                polars::prelude::AnyValue::Float32(f) => f as f64,
                polars::prelude::AnyValue::Int64(i) => i as f64,
                _ => {
                    return Err(VangaError::DataError(format!(
                        "Unsupported volume type: {:?}",
                        volume
                    )))
                }
            };

            ohlc_data.push(crate::data::structures::MarketDataRow {
                timestamp: timestamp_i64,
                open: open_f64,
                high: high_f64,
                low: low_f64,
                close: close_f64,
                volume: volume_f64,
            });
        }

        log::debug!(
            "Extracted {} OHLC rows from sequence for ATR calculation",
            ohlc_data.len()
        );
        Ok(ohlc_data)
    }
}
