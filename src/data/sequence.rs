// Sequence generator for LSTM training and prediction
use crate::data::{DataMetadata, PreparedData, PreparedPredictionData};
use crate::utils::error::{Result, VangaError};
use chrono::Utc;
use ndarray::{s, Array2, Array3, Axis};
use polars::prelude::*;

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
            data_config,
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
        let (sequences, targets) = self
            .create_sliding_windows_with_normalization(
                &feature_data,
                sequence_length,
                horizons,
                &df,
                data_config,
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
            // No global normalization stats needed - each sequence normalized independently
            normalization_stats: crate::data::NormalizationStats {
                means: vec![],
                stds: vec![],
                mins: vec![],
                maxs: vec![],
                medians: vec![],
                q25: vec![],
                q75: vec![],
            },
            metadata,
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
        let normalized_sequence = self.normalize_sequence_window(&sequence_df)?;
        let sequences =
            self.create_prediction_sequences(&normalized_sequence, 0, sequence_length)?;

        // Create metadata
        let metadata = crate::data::DataMetadata {
            symbol: symbol.to_string(),
            start_time: chrono::Utc::now(),
            end_time: chrono::Utc::now(),
            total_records: num_rows,
            feature_count: feature_columns.len(),
            sequence_length,
            horizons: vec!["1h".to_string()], // Default horizon
        };

        Ok(PreparedPredictionData {
            sequences,
            feature_names: feature_columns,
            metadata,
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
            .map(|h| crate::targets::volatility::parse_horizon_to_steps(h).unwrap_or(1))
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
        let required_columns = ["close", "high", "low", "volume"];
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

    /// Parse horizon string to steps (reuses existing volatility module function)
    /// Validate feature window requirements for training data
    fn validate_feature_window_requirements(
        &self,
        total_records: usize,
        sequence_length: usize,
        horizons: &[String],
        _data_config: &crate::config::training::DataConfig,
    ) -> Result<()> {
        // Calculate actual feature window from config if available
        // TODO: Pass feature config to this method for accurate calculation
        let estimated_max_window = 200; // Conservative estimate - should be replaced with actual config

        // Calculate minimum data requirements
        let max_horizon_steps = horizons
            .iter()
            .map(|h| crate::targets::volatility::parse_horizon_to_steps(h).unwrap_or(1))
            .max()
            .unwrap_or(1);

        let min_required = estimated_max_window + sequence_length + max_horizon_steps;

        if total_records < min_required {
            return Err(VangaError::DataError(format!(
                "Insufficient data for training: {} records available, {} required\n\
                 Breakdown:\n\
                 • Feature window: {} periods (for technical indicators like SMA, EMA)\n\
                 • Sequence length: {} periods (for LSTM input)\n\
                 • Horizon buffer: {} periods (for target calculation)\n\
                 • Total required: {} periods\n\
                 \n\
                 Solution: Provide at least {} rows of historical data",
                total_records,
                min_required,
                estimated_max_window,
                sequence_length,
                max_horizon_steps,
                min_required,
                min_required
            )));
        }

        log::info!("📊 TRAINING DATA VALIDATION:");
        log::info!("   • Dataset size: {} rows", total_records);
        log::info!(
            "   • Feature window: {} periods (estimated)",
            estimated_max_window
        );
        log::info!("   • Sequence length: {} periods", sequence_length);
        log::info!("   • Horizon buffer: {} periods", max_horizon_steps);
        log::info!("   • Total required: {} periods", min_required);
        log::info!(
            "   • Effective training data: {} rows",
            total_records.saturating_sub(estimated_max_window)
        );
        log::info!("   ✅ Sufficient data available");

        Ok(())
    }

    /// Normalize a sequence window using only data within that window
    /// This is the CORRECT approach for time series ML - each sequence is self-contained
    fn normalize_sequence_window(&self, window_df: &DataFrame) -> Result<DataFrame> {
        let mut normalized_columns = Vec::new();

        // Process each column independently
        for column_name in window_df.get_column_names() {
            if column_name == "timestamp" {
                // Keep timestamp as-is
                normalized_columns.push(window_df.column(column_name)?.clone());
                continue;
            }

            if let Ok(series) = window_df.column(column_name) {
                if series.dtype().is_numeric() {
                    // Normalize using only THIS WINDOW's data
                    let normalized_series = self.normalize_column_in_window(series)?;
                    normalized_columns.push(normalized_series);
                } else {
                    normalized_columns.push(series.clone());
                }
            }
        }

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
    async fn create_sliding_windows_with_normalization(
        &self,
        _feature_data: &Array2<f64>, // Unused - we work with DataFrame directly
        sequence_length: usize,
        horizons: &[String],
        df: &DataFrame,
        data_config: &crate::config::training::DataConfig,
    ) -> Result<(Array3<f64>, crate::targets::PreparedTargets)> {
        // Calculate maximum horizon steps
        let max_horizon_steps = horizons
            .iter()
            .map(|h| crate::targets::volatility::parse_horizon_to_steps(h).unwrap_or(1))
            .max()
            .unwrap_or(1);

        // Calculate gap size (same logic as walk-forward windows)
        let gap_size = max_horizon_steps; // Gap to prevent data leakage

        // Total window size: sequence + gap + horizon
        let total_window_size = sequence_length + gap_size + max_horizon_steps;

        log::info!(
            "🔧 Per-sequence normalization: sequence_length({}) + gap({}) + max_horizon({}) = {} total window",
            sequence_length, gap_size, max_horizon_steps, total_window_size
        );

        let total_records = df.height();
        if total_records < total_window_size {
            return Err(VangaError::DataError(format!(
                "Insufficient data: need {} records, have {}",
                total_window_size, total_records
            )));
        }

        let mut all_sequences = Vec::new();

        // Generate sequences with proper normalization
        let step_size = if data_config.sequence_overlap > 0.0 {
            1
        } else {
            sequence_length
        };
        let mut i = 0;

        while i + total_window_size <= total_records {
            // Extract the complete window (sequence + gap + horizon)
            let window_df = df.slice(i as i64, total_window_size);

            // FIXED: Only normalize the input sequence part, NOT gap+targets
            let sequence_df = window_df.slice(0, sequence_length);
            let normalized_sequence = self.normalize_sequence_window(&sequence_df)?;
            let feature_columns: Vec<String> = df
                .get_column_names()
                .iter()
                .filter(|&col| *col != "timestamp")
                .map(|s| s.to_string())
                .collect();
            let sequence_matrix =
                self.extract_feature_matrix(&normalized_sequence, &feature_columns)?;

            // Convert to sequence format [sequence_length, features]
            all_sequences.push(sequence_matrix);

            i += step_size;
        }

        log::info!(
            "✅ Generated {} sequences with per-sequence normalization",
            all_sequences.len()
        );

        // Convert sequences to Array3
        let sequences = self.convert_sequences_to_array3(all_sequences)?;

        // Generate targets using the original target generation logic
        // Note: Targets are generated from the original raw data, not normalized data
        let target_generator = crate::targets::TargetGenerator::with_defaults();
        let targets = target_generator.generate_all_targets(df, None).await?;

        Ok((sequences, targets))
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
}
