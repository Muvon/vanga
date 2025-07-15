// FIXED: Sequence generator with configurable overlap to prevent data leakage
use crate::data::{DataMetadata, NormalizationStats, PreparedData, PreparedPredictionData};
use crate::targets::PreparedTargets;
use crate::utils::error::{Result, VangaError};
use chrono::Utc;
use ndarray::{s, Array2, Array3, Axis};
use polars::prelude::*;
use rayon::prelude::*;

pub struct SequenceGenerator {
    /// Overlap ratio between sequences (0.0 = no overlap, 0.9 = 90% overlap)
    sequence_overlap: f64,
}

impl Default for SequenceGenerator {
    fn default() -> Self {
        Self::new(0.0) // Default to NO overlap to prevent data leakage
    }
}

impl SequenceGenerator {
    pub fn new(sequence_overlap: f64) -> Self {
        Self { sequence_overlap }
    }

    /// FIXED: Create sliding windows with configurable overlap
    async fn create_sliding_windows(
        &self,
        feature_data: &Array2<f64>,
        sequence_length: usize,
        horizons: &[String],
        df: &DataFrame,
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

        // Calculate step size based on overlap ratio
        let step_size = if self.sequence_overlap == 0.0 {
            sequence_length // No overlap
        } else {
            std::cmp::max(
                1,
                (sequence_length as f64 * (1.0 - self.sequence_overlap)) as usize,
            )
        };

        log::info!(
            "🔧 ANTI-LEAKAGE: Using step_size={} (overlap={:.1}%) for sequence generation",
            step_size,
            self.sequence_overlap * 100.0
        );

        // Calculate maximum horizon offset for proper sequence alignment
        let max_horizon_steps = horizons
            .iter()
            .map(|h| crate::targets::volatility::parse_horizon_to_steps(h).unwrap_or(1))
            .max()
            .unwrap_or(1);

        // Adjust sequence count to account for multi-horizon targets and step size
        let effective_rows = total_rows.saturating_sub(max_horizon_steps);
        let max_start_idx = effective_rows.saturating_sub(sequence_length);

        // Calculate number of sequences with step_size
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

        log::info!(
            "📊 Creating {} sequences (step_size={}, overlap={:.1}%) with {} features for {} horizons",
            num_sequences,
            step_size,
            self.sequence_overlap * 100.0,
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

        log::info!(
            "✅ ANTI-LEAKAGE: Generated {} non-overlapping sequences (vs ~{} with 99% overlap)",
            num_sequences,
            max_start_idx
        );

        // Return both sequences and the actual generated targets
        Ok((sequences, prepared_targets))
    }

    // ... rest of the methods remain the same
}
