// Sequence generator for LSTM training and prediction
use crate::data::{DataMetadata, NormalizationStats, PreparedData, PreparedPredictionData};
use crate::targets::PreparedTargets;
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
        df: DataFrame,
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

        // Ensure we have enough data
        if total_records < sequence_length + 10 {
            return Err(VangaError::DataError(format!(
                "Not enough data: {} records, need at least {}",
                total_records,
                sequence_length + 10
            )));
        }

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

        // Generate sequences using sliding window with config overlap
        let (sequences, targets) = self
            .create_sliding_windows(&feature_data, sequence_length, horizons, &df, data_config)
            .await?;

        // Calculate normalization statistics
        let normalization_stats = self.calculate_normalization_stats(&feature_data)?;

        // Apply normalization to sequences
        let normalized_sequences = self.normalize_sequences(&sequences, &normalization_stats)?;

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
            normalized_sequences.len_of(Axis(0)),
            feature_count
        );

        // Final data utilization and efficiency summary
        log::info!("🏁 FINAL DATA PIPELINE SUMMARY:");
        log::info!(
            "   • Total Data Processed: {} rows → {} sequences → {} targets",
            total_records,
            normalized_sequences.len_of(Axis(0)),
            targets.data_length
        );
        log::info!(
            "   • Memory Efficiency: {:.1}% data utilization",
            (normalized_sequences.len_of(Axis(0)) as f64 / total_records as f64) * 100.0
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
            sequences: normalized_sequences,
            targets,
            feature_names: feature_columns,
            normalization_stats,
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

        // Ensure we have enough data for at least one sequence
        if num_rows < sequence_length {
            return Err(VangaError::DataError(format!(
                "Not enough data for prediction: {} rows < {} required",
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

        // Extract feature data as matrix
        let feature_data = self.extract_feature_matrix(&df, &feature_columns)?;

        // Generate prediction sequences (use the last sequence_length rows)
        let start_idx = num_rows.saturating_sub(sequence_length);
        let sequences =
            self.create_prediction_sequences(&feature_data, start_idx, sequence_length)?;

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

    async fn create_sliding_windows(
        &self,
        feature_data: &Array2<f64>,
        sequence_length: usize,
        horizons: &[String],
        df: &DataFrame,
        data_config: &crate::config::training::DataConfig,
    ) -> Result<(Array3<f64>, PreparedTargets)> {
        let total_rows = feature_data.nrows();
        let feature_count = feature_data.ncols();

        if total_rows < sequence_length + 1 {
            return Err(VangaError::DataError(format!(
                "Not enough data for sequences: {} rows, need {}",
                total_rows,
                sequence_length + 1
            )));
        }

        // Calculate maximum horizon offset for proper sequence alignment
        let max_horizon_steps = horizons
            .iter()
            .map(|h| crate::targets::volatility::parse_horizon_to_steps(h).unwrap_or(1))
            .max()
            .unwrap_or(1);

        // Calculate step size based on config overlap ratio
        let sequence_overlap = data_config.sequence_overlap;
        let step_size = if sequence_overlap == 0.0 {
            sequence_length // No overlap - sequences don't share data
        } else {
            std::cmp::max(
                1,
                (sequence_length as f64 * (1.0 - sequence_overlap)) as usize,
            )
        };

        // Adjust sequence count to account for multi-horizon targets and step size
        let effective_rows = total_rows.saturating_sub(max_horizon_steps);
        let max_start_idx = effective_rows.saturating_sub(sequence_length);

        // Calculate number of sequences with step_size (prevents 99% overlap!)
        let num_sequences = if max_start_idx == 0 {
            0
        } else {
            (max_start_idx / step_size) + 1
        };

        if num_sequences == 0 {
            return Err(VangaError::DataError(format!(
                "Insufficient data for sequences: {} total rows, {} max horizon steps, {} sequence length, {} step size",
                total_rows, max_horizon_steps, sequence_length, step_size
            )));
        }

        // Calculate data efficiency metrics
        let theoretical_max_sequences = max_start_idx; // With step_size=1
        let data_efficiency = (num_sequences as f64 / theoretical_max_sequences as f64) * 100.0;
        let total_data_points = num_sequences * sequence_length;
        let unique_data_points = max_start_idx + sequence_length - 1;
        let data_reuse_factor = total_data_points as f64 / unique_data_points as f64;

        log::info!("📊 SEQUENCE GENERATION SUMMARY:");
        log::info!(
            "   • Dataset: {} rows → {} effective rows (after horizon offset: {})",
            total_rows,
            effective_rows,
            max_horizon_steps
        );
        log::info!(
            "   • Overlap Config: {:.1}% → Step Size: {} (every {} rows)",
            sequence_overlap * 100.0,
            step_size,
            step_size
        );
        log::info!(
            "   • Sequences: {} generated (vs {} theoretical max with step=1)",
            num_sequences,
            theoretical_max_sequences
        );
        log::info!(
            "   • Data Efficiency: {:.1}% ({} sequences / {} max possible)",
            data_efficiency,
            num_sequences,
            theoretical_max_sequences
        );
        log::info!(
            "   • Data Reuse: {:.2}x (each data point used ~{:.1} times)",
            data_reuse_factor,
            data_reuse_factor
        );
        log::info!(
            "   • Features: {} per sequence, Horizons: {}",
            feature_count,
            horizons.len()
        );

        // Generate targets using DataFrame for all horizons
        let prepared_targets = self.generate_multi_horizon_targets(df, horizons).await?;

        let mut sequences = Array3::zeros((num_sequences, sequence_length, feature_count));

        // FIXED: Create sequences with configurable step size (no more 99% overlap!)
        let sequences_vec: Vec<Array2<f64>> = (0..num_sequences)
            .into_par_iter()
            .map(|i| {
                let start_idx = i * step_size;
                feature_data
                    .slice(s![start_idx..start_idx + sequence_length, ..])
                    .to_owned()
            })
            .collect();

        // Convert parallel results to Array3
        for (i, sequence) in sequences_vec.into_iter().enumerate() {
            sequences.slice_mut(s![i, .., ..]).assign(&sequence);
        }

        // Return both sequences and the actual generated targets
        Ok((sequences, prepared_targets))
    }

    fn calculate_normalization_stats(
        &self,
        feature_data: &Array2<f64>,
    ) -> Result<NormalizationStats> {
        let feature_count = feature_data.ncols();
        let mut means = Vec::with_capacity(feature_count);
        let mut stds = Vec::with_capacity(feature_count);
        let mut mins = Vec::with_capacity(feature_count);
        let mut maxs = Vec::with_capacity(feature_count);
        let mut medians = Vec::with_capacity(feature_count);
        let mut q25 = Vec::with_capacity(feature_count);
        let mut q75 = Vec::with_capacity(feature_count);

        for col_idx in 0..feature_count {
            let column = feature_data.column(col_idx);
            // Filter out NaN values before sorting and statistics
            let mut sorted_values: Vec<f64> = column
                .to_vec()
                .into_iter()
                .filter(|x| x.is_finite())
                .collect();
            sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap());

            let mean = column.mean().unwrap_or(0.0);
            let std = column.std(0.0);
            let min_val = sorted_values.first().copied().unwrap_or(0.0);
            let max_val = sorted_values.last().copied().unwrap_or(0.0);

            let len = sorted_values.len();
            let (median, q25_val, q75_val) = if len == 0 {
                // Handle case where all values were NaN
                (0.0, 0.0, 0.0)
            } else {
                let median = if len % 2 == 0 && len > 1 {
                    (sorted_values[len / 2 - 1] + sorted_values[len / 2]) / 2.0
                } else {
                    sorted_values[len / 2]
                };

                let q25_val = if len > 4 {
                    sorted_values[len / 4]
                } else {
                    sorted_values[0]
                };
                let q75_val = if len > 4 {
                    sorted_values[3 * len / 4]
                } else {
                    sorted_values[len - 1]
                };

                (median, q25_val, q75_val)
            };

            means.push(mean);
            stds.push(std);
            mins.push(min_val);
            maxs.push(max_val);
            medians.push(median);
            q25.push(q25_val);
            q75.push(q75_val);
        }

        Ok(NormalizationStats {
            means,
            stds,
            mins,
            maxs,
            medians,
            q25,
            q75,
        })
    }

    fn normalize_sequences(
        &self,
        sequences: &Array3<f64>,
        stats: &NormalizationStats,
    ) -> Result<Array3<f64>> {
        let mut normalized = sequences.clone();
        let feature_count = sequences.len_of(Axis(2));

        for feature_idx in 0..feature_count {
            let mean = stats.means[feature_idx];
            let std = stats.stds[feature_idx];

            if std > 0.0 {
                // Z-score normalization
                normalized
                    .slice_mut(s![.., .., feature_idx])
                    .mapv_inplace(|x| (x - mean) / std);
            }
        }

        Ok(normalized)
    }

    /// Create prediction sequences from feature data
    fn create_prediction_sequences(
        &self,
        feature_data: &Array2<f64>,
        start_idx: usize,
        sequence_length: usize,
    ) -> Result<Array3<f64>> {
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
            .map(|h| self.parse_horizon_to_steps(h).unwrap_or(1))
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
    fn parse_horizon_to_steps(&self, horizon: &str) -> Result<usize> {
        // Reuse the existing function from volatility module
        crate::targets::volatility::parse_horizon_to_steps(horizon)
    }

    /// Generate targets for multiple horizons using DataFrame
    async fn generate_multi_horizon_targets(
        &self,
        df: &DataFrame,
        horizons: &[String],
    ) -> Result<PreparedTargets> {
        log::info!(
            "Generating targets for {} specific horizons",
            horizons.len()
        );

        // Create target generator with horizon-specific configuration
        let config = crate::targets::MultiTargetConfig {
            horizons: horizons.to_vec(),
            ..Default::default()
        };

        let target_generator = crate::targets::TargetGenerator::new(config);
        target_generator.generate_all_targets(df).await
    }
}
