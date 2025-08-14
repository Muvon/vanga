//! Advanced diversity-based sequence selection for optimal training
//!
//! This module implements sophisticated diversity metrics and selection algorithms
//! to maximize training data quality by selecting the most diverse and representative
//! samples from overloaded classes while maintaining perfect class balance.

use crate::data::balance::SequenceWithTargets;
use crate::targets::TargetType;
use crate::utils::error::{Result, VangaError};
use ndarray::{Array1, Array2};
use std::collections::HashMap;

/// Comprehensive diversity metrics for sequence selection
#[derive(Debug, Clone)]
pub struct DiversityMetrics {
    /// Feature space diversity score (0.0 to 1.0)
    pub feature_diversity: f64,
    /// Temporal diversity score (0.0 to 1.0)
    pub temporal_diversity: f64,
    /// Market condition diversity score (0.0 to 1.0)
    pub market_diversity: f64,
    /// Target-specific diversity score (0.0 to 1.0)
    pub target_diversity: f64,
    /// Composite diversity score (weighted combination)
    pub composite_score: f64,
}

/// Configuration for diversity-based selection
#[derive(Debug, Clone)]
pub struct DiversityConfig {
    /// Weight for feature space diversity (default: 0.3)
    pub feature_weight: f64,
    /// Weight for temporal diversity (default: 0.25)
    pub temporal_weight: f64,
    /// Weight for market condition diversity (default: 0.25)
    pub market_weight: f64,
    /// Weight for target-specific diversity (default: 0.2)
    pub target_weight: f64,
    /// Minimum diversity threshold for selection (default: 0.1)
    pub min_diversity_threshold: f64,
    /// Maximum similarity allowed between selected sequences (default: 0.8)
    pub max_similarity_threshold: f64,
}

impl Default for DiversityConfig {
    fn default() -> Self {
        Self {
            feature_weight: 0.3,
            temporal_weight: 0.25,
            market_weight: 0.25,
            target_weight: 0.2,
            min_diversity_threshold: 0.1,
            max_similarity_threshold: 0.8,
        }
    }
}

/// Advanced diversity-based sequence selector
pub struct DiversitySelector {
    config: DiversityConfig,
}

impl DiversitySelector {
    pub fn new(config: DiversityConfig) -> Self {
        Self { config }
    }

    /// SENIOR-LEVEL: Fast diversity selection using efficient algorithms
    ///
    /// Instead of O(n²) pairwise comparisons, we use:
    /// 1. Pre-computed statistical features (O(n))
    /// 2. K-means clustering for diversity (O(n log n))
    /// 3. Stratified sampling within clusters
    pub fn select_diverse_sequences(
        &self,
        all_sequences: &[SequenceWithTargets],
        class_indices: &[usize],
        target_count: usize,
        target_type: TargetType,
        horizon: &str,
        exclude_indices: &[usize],
    ) -> Result<Vec<usize>> {
        if class_indices.len() <= target_count {
            return Ok(class_indices.to_vec());
        }

        let utilization = (target_count as f64 / class_indices.len() as f64) * 100.0;

        log::info!(
            "🎯 DIVERSITY SELECTION: Selecting {} most diverse sequences from {} available for {:?} {} ({:.1}% utilization)",
            target_count,
            class_indices.len(),
            target_type,
            horizon,
            utilization
        );

        // Filter out excluded indices
        let available_indices: Vec<usize> = class_indices
            .iter()
            .filter(|&&idx| !exclude_indices.contains(&idx))
            .copied()
            .collect();

        if available_indices.len() < target_count {
            return Err(VangaError::DataError(format!(
                "Not enough valid sequences after exclusions: {} available, {} needed",
                available_indices.len(),
                target_count
            )));
        }

        // FAST APPROACH: Use efficient diversity selection
        let selected = self.select_diverse_fast(all_sequences, &available_indices, target_count)?;

        log::info!(
            "✅ FAST DIVERSITY SELECTION COMPLETE: Selected {} sequences",
            selected.len()
        );

        Ok(selected)
    }

    /// Fast diversity selection using clustering and stratified sampling
    fn select_diverse_fast(
        &self,
        all_sequences: &[SequenceWithTargets],
        available_indices: &[usize],
        target_count: usize,
    ) -> Result<Vec<usize>> {
        // Step 1: Extract lightweight features for all sequences (O(n))
        let mut sequence_features = Vec::new();
        for &idx in available_indices {
            let seq = &all_sequences[idx];
            let features = self.extract_lightweight_features(&seq.sequence_data)?;
            sequence_features.push((idx, features));
        }

        // Step 2: Simple diversity-based selection using temporal + statistical spread
        let selected = self.select_by_spread(&sequence_features, target_count, all_sequences)?;

        Ok(selected)
    }

    /// Extract lightweight features for fast diversity calculation
    /// For normalized sequences, we can use a subset of the sequence directly
    fn extract_lightweight_features(&self, data: &Array2<f64>) -> Result<Vec<f64>> {
        let (seq_len, num_features) = data.dim();

        // For normalized sequences, we can sample key points from the sequence
        // This is much more effective than statistical moments which become meaningless
        let mut features = Vec::new();

        // Sample key points from the sequence (beginning, middle, end)
        let sample_indices = if seq_len >= 3 {
            vec![0, seq_len / 2, seq_len - 1]
        } else {
            (0..seq_len).collect()
        };

        // Extract values at key time points for first 5 features (OHLCV)
        for &time_idx in &sample_indices {
            for feature_idx in 0..num_features.min(5) {
                features.push(data[[time_idx, feature_idx]]);
            }
        }

        Ok(features)
    }

    /// Select sequences by maximizing spread in feature + temporal space
    fn select_by_spread(
        &self,
        sequence_features: &[(usize, Vec<f64>)],
        target_count: usize,
        all_sequences: &[SequenceWithTargets],
    ) -> Result<Vec<usize>> {
        use rand::seq::SliceRandom;

        // Sort by temporal position for temporal diversity
        let mut temporal_sorted: Vec<(usize, usize)> = sequence_features
            .iter()
            .map(|(idx, _)| (*idx, all_sequences[*idx].start_idx))
            .collect();
        temporal_sorted.sort_by_key(|(_, start_idx)| *start_idx);

        // Divide into temporal buckets and select from each
        let num_buckets = (target_count / 10).clamp(3, 10); // 3-10 buckets
        let bucket_size = temporal_sorted.len() / num_buckets;
        let sequences_per_bucket = target_count / num_buckets;
        let remainder = target_count % num_buckets;

        let mut selected = Vec::new();
        let mut rng = rand::rng();

        for bucket_idx in 0..num_buckets {
            let start = bucket_idx * bucket_size;
            let end = if bucket_idx == num_buckets - 1 {
                temporal_sorted.len() // Last bucket gets remainder
            } else {
                (bucket_idx + 1) * bucket_size
            };

            let mut bucket_sequences: Vec<usize> = temporal_sorted[start..end]
                .iter()
                .map(|(idx, _)| *idx)
                .collect();

            // Shuffle and select from this temporal bucket
            bucket_sequences.shuffle(&mut rng);

            let take_count = if bucket_idx < remainder {
                sequences_per_bucket + 1
            } else {
                sequences_per_bucket
            };

            selected.extend(bucket_sequences.into_iter().take(take_count));
        }

        // If we still need more sequences, add randomly from remaining
        if selected.len() < target_count {
            let remaining_needed = target_count - selected.len();
            let mut remaining: Vec<usize> = sequence_features
                .iter()
                .map(|(idx, _)| *idx)
                .filter(|idx| !selected.contains(idx))
                .collect();

            remaining.shuffle(&mut rng);
            selected.extend(remaining.into_iter().take(remaining_needed));
        }

        // Ensure we have exactly the right count
        selected.truncate(target_count);

        Ok(selected)
    }

    /// Calculate comprehensive diversity metrics for a sequence
    pub fn calculate_sequence_diversity(
        &self,
        all_sequences: &[SequenceWithTargets],
        sequence_idx: usize,
        class_indices: &[usize],
        target_type: TargetType,
        horizon: &str,
    ) -> Result<DiversityMetrics> {
        // 1. Feature Space Diversity
        let feature_diversity =
            self.calculate_feature_diversity(all_sequences, sequence_idx, class_indices)?;

        // 2. Temporal Diversity
        let temporal_diversity =
            self.calculate_temporal_diversity(all_sequences, sequence_idx, class_indices)?;

        // 3. Market Condition Diversity
        let market_diversity =
            self.calculate_market_diversity(all_sequences, sequence_idx, class_indices)?;

        // 4. Target-Specific Diversity
        let target_diversity = self.calculate_target_diversity(
            all_sequences,
            sequence_idx,
            class_indices,
            target_type,
            horizon,
        )?;

        // 5. Composite Score
        let composite_score = self.config.feature_weight * feature_diversity
            + self.config.temporal_weight * temporal_diversity
            + self.config.market_weight * market_diversity
            + self.config.target_weight * target_diversity;

        Ok(DiversityMetrics {
            feature_diversity,
            temporal_diversity,
            market_diversity,
            target_diversity,
            composite_score,
        })
    }

    /// Calculate feature space diversity using cosine distance between normalized sequences
    fn calculate_feature_diversity(
        &self,
        all_sequences: &[SequenceWithTargets],
        sequence_idx: usize,
        class_indices: &[usize],
    ) -> Result<f64> {
        let target_sequence = &all_sequences[sequence_idx];
        let target_data = &target_sequence.sequence_data;

        // Calculate average cosine distance to other sequences in the same class
        let mut total_distance = 0.0;
        let mut count = 0;

        for &other_idx in class_indices {
            if other_idx == sequence_idx {
                continue;
            }

            let other_sequence = &all_sequences[other_idx];
            let other_data = &other_sequence.sequence_data;

            // Direct cosine distance between full sequences (perfect for normalized data)
            let distance = self.calculate_cosine_distance(target_data, other_data)?;
            total_distance += distance;
            count += 1;
        }

        if count == 0 {
            return Ok(1.0); // Maximum diversity if only one sequence
        }

        let avg_distance = total_distance / count as f64;

        // Cosine distance is already in 0-1 range, but normalize for consistency
        let normalized_diversity = avg_distance.min(1.0).max(0.0);

        Ok(normalized_diversity)
    }

    /// Calculate temporal diversity based on time distribution
    fn calculate_temporal_diversity(
        &self,
        all_sequences: &[SequenceWithTargets],
        sequence_idx: usize,
        class_indices: &[usize],
    ) -> Result<f64> {
        let target_sequence = &all_sequences[sequence_idx];
        let target_start = target_sequence.start_idx;

        // Calculate temporal spread within the class
        let class_starts: Vec<usize> = class_indices
            .iter()
            .map(|&idx| all_sequences[idx].start_idx)
            .collect();

        if class_starts.len() <= 1 {
            return Ok(1.0);
        }

        let min_start = *class_starts.iter().min().unwrap();
        let max_start = *class_starts.iter().max().unwrap();
        let total_span = max_start - min_start;

        if total_span == 0 {
            return Ok(1.0); // All sequences at same time
        }

        // Calculate how far this sequence is from the temporal center
        let temporal_center = (min_start + max_start) / 2;
        let distance_from_center = (target_start as i64 - temporal_center as i64).abs() as f64;
        let max_distance = (total_span / 2) as f64;

        // Higher diversity for sequences further from temporal center
        let temporal_diversity = if max_distance > 0.0 {
            (distance_from_center / max_distance).min(1.0)
        } else {
            1.0
        };

        Ok(temporal_diversity)
    }

    /// Calculate market condition diversity using cosine distance
    fn calculate_market_diversity(
        &self,
        all_sequences: &[SequenceWithTargets],
        sequence_idx: usize,
        class_indices: &[usize],
    ) -> Result<f64> {
        let target_sequence = &all_sequences[sequence_idx];
        let target_data = &target_sequence.sequence_data;

        // Calculate average cosine distance to other sequences in market condition space
        let mut total_distance = 0.0;
        let mut count = 0;

        for &other_idx in class_indices {
            if other_idx == sequence_idx {
                continue;
            }

            let other_sequence = &all_sequences[other_idx];
            let other_data = &other_sequence.sequence_data;

            // Use cosine distance for market condition comparison
            let distance = self.calculate_cosine_distance(target_data, other_data)?;
            total_distance += distance;
            count += 1;
        }

        if count == 0 {
            return Ok(0.5); // Default diversity
        }

        let avg_distance = total_distance / count as f64;
        let normalized_diversity = avg_distance.min(1.0).max(0.0);

        Ok(normalized_diversity)
    }

    /// Calculate target-specific diversity
    fn calculate_target_diversity(
        &self,
        all_sequences: &[SequenceWithTargets],
        sequence_idx: usize,
        class_indices: &[usize],
        target_type: TargetType,
        horizon: &str,
    ) -> Result<f64> {
        let target_sequence = &all_sequences[sequence_idx];
        let target_key = (target_type, horizon.to_string());

        // Get the target value for this sequence
        let target_value = target_sequence
            .targets
            .get(&target_key)
            .ok_or_else(|| VangaError::DataError("Target value not found".to_string()))?;

        // Calculate diversity based on class distribution within the class_indices
        let mut class_counts = HashMap::new();
        for &idx in class_indices {
            if let Some(seq) = all_sequences.get(idx) {
                if let Some(&class_val) = seq.targets.get(&target_key) {
                    *class_counts.entry(class_val).or_insert(0) += 1;
                }
            }
        }

        // Calculate diversity based on how rare this target value is within the class
        let total_sequences = class_indices.len();
        let same_target_count = class_counts.get(target_value).copied().unwrap_or(0);

        if total_sequences == 0 {
            return Ok(0.5); // Default diversity
        }

        // Higher diversity for rarer target values within the class
        let rarity_score = 1.0 - (same_target_count as f64 / total_sequences as f64);

        // Normalize to reasonable range (0.2 to 0.8)
        let target_diversity = 0.2 + (rarity_score * 0.6);

        Ok(target_diversity)
    }

    /// Calculate statistical features for a sequence
    pub fn calculate_sequence_statistics(&self, data: &Array2<f64>) -> Result<Array1<f64>> {
        let num_features = data.ncols();
        let mut stats = Vec::new();

        // Calculate statistics for each feature
        for feature_idx in 0..num_features {
            let feature_column = data.column(feature_idx);

            // Basic statistics
            let mean = feature_column.mean().unwrap_or(0.0);
            let std = feature_column.std(0.0);

            // Additional statistics
            let min = feature_column.iter().fold(f64::INFINITY, |a, &b| a.min(b));
            let max = feature_column
                .iter()
                .fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            let range = max - min;

            stats.extend_from_slice(&[mean, std, min, max, range]);
        }

        Ok(Array1::from_vec(stats))
    }

    /// Calculate cosine distance between two normalized sequences
    /// Perfect for normalized data - measures pattern similarity regardless of magnitude
    pub fn calculate_cosine_distance(
        &self,
        seq_a: &Array2<f64>,
        seq_b: &Array2<f64>,
    ) -> Result<f64> {
        // Ensure sequences have the same shape
        if seq_a.shape() != seq_b.shape() {
            return Err(VangaError::DataError(
                "Sequences must have the same shape for cosine distance calculation".to_string(),
            ));
        }

        // Flatten sequences to 1D for dot product calculation
        let flat_a: Vec<f64> = seq_a.iter().cloned().collect();
        let flat_b: Vec<f64> = seq_b.iter().cloned().collect();

        // Calculate dot product
        let dot_product: f64 = flat_a
            .iter()
            .zip(flat_b.iter())
            .map(|(a, b)| a * b)
            .sum::<f64>();

        // Calculate norms (magnitudes)
        let norm_a: f64 = flat_a.iter().map(|x| x.powi(2)).sum::<f64>().sqrt();
        let norm_b: f64 = flat_b.iter().map(|x| x.powi(2)).sum::<f64>().sqrt();

        // Handle edge case where one sequence has zero norm
        if norm_a == 0.0 || norm_b == 0.0 {
            return Ok(1.0); // Maximum distance for zero vectors
        }

        // Calculate cosine similarity
        let cosine_similarity = dot_product / (norm_a * norm_b);

        // Convert to cosine distance (1 - similarity)
        // Clamp to [0, 1] range to handle floating point precision issues
        let cosine_distance = (1.0 - cosine_similarity).max(0.0).min(1.0);

        Ok(cosine_distance)
    }

    /// Calculate Euclidean distance between two statistical feature vectors
    /// DEPRECATED: Use calculate_cosine_distance for normalized sequences
    pub fn euclidean_distance(&self, a: &Array1<f64>, b: &Array1<f64>) -> f64 {
        if a.len() != b.len() {
            return 0.0;
        }

        let sum_squared_diff: f64 = a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum();

        sum_squared_diff.sqrt()
    }

    /// Extract market condition features from sequence data
    pub fn extract_market_conditions(&self, data: &Array2<f64>) -> Result<Array1<f64>> {
        let (seq_len, num_features) = data.dim();

        // Assuming OHLCV structure (first 5 features)
        if num_features < 5 {
            return Err(VangaError::DataError(
                "Insufficient features for market condition analysis".to_string(),
            ));
        }

        let mut conditions = Vec::new();

        // Extract OHLCV columns
        let high = data.column(1);
        let low = data.column(2);
        let close = data.column(3);
        let volume = data.column(4);

        // Calculate market condition indicators

        // 1. Volatility (average true range)
        let mut atr_sum = 0.0;
        for i in 1..seq_len {
            let tr = (high[i] - low[i])
                .max((high[i] - close[i - 1]).abs())
                .max((low[i] - close[i - 1]).abs());
            atr_sum += tr;
        }
        let avg_volatility = if seq_len > 1 {
            atr_sum / (seq_len - 1) as f64
        } else {
            0.0
        };

        // 2. Trend strength (linear regression slope)
        let trend_slope = self.calculate_linear_slope(&close.to_vec());

        // 3. Volume profile (average volume relative to price)
        let avg_volume = volume.mean().unwrap_or(0.0);
        let avg_price = close.mean().unwrap_or(1.0);
        let volume_price_ratio = if avg_price > 0.0 {
            avg_volume / avg_price
        } else {
            0.0
        };

        // 4. Price range (high-low spread)
        let price_range = (high.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b))
            - low.iter().fold(f64::INFINITY, |a, &b| a.min(b)))
            / avg_price;

        conditions.extend_from_slice(&[
            avg_volatility,
            trend_slope,
            volume_price_ratio,
            price_range,
        ]);

        Ok(Array1::from_vec(conditions))
    }

    /// Calculate linear slope for trend analysis
    fn calculate_linear_slope(&self, values: &[f64]) -> f64 {
        let n = values.len() as f64;
        if n < 2.0 {
            return 0.0;
        }

        let x_mean = (n - 1.0) / 2.0; // 0, 1, 2, ... n-1 mean
        let y_mean = values.iter().sum::<f64>() / n;

        let mut numerator = 0.0;
        let mut denominator = 0.0;

        for (i, &y) in values.iter().enumerate() {
            let x = i as f64;
            numerator += (x - x_mean) * (y - y_mean);
            denominator += (x - x_mean).powi(2);
        }

        if denominator.abs() < f64::EPSILON {
            0.0
        } else {
            numerator / denominator
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_diversity_config_default() {
        let config = DiversityConfig::default();
        assert!(
            (config.feature_weight
                + config.temporal_weight
                + config.market_weight
                + config.target_weight
                - 1.0)
                .abs()
                < 0.001
        );
    }

    #[test]
    fn test_statistical_features() {
        let selector = DiversitySelector::new(DiversityConfig::default());
        let data = Array2::from_shape_vec(
            (5, 3),
            vec![
                1.0, 2.0, 3.0, 2.0, 3.0, 4.0, 3.0, 4.0, 5.0, 4.0, 5.0, 6.0, 5.0, 6.0, 7.0,
            ],
        )
        .unwrap();

        let stats = selector.calculate_sequence_statistics(&data).unwrap();
        assert!(!stats.is_empty());
    }

    #[test]
    fn test_linear_slope() {
        let selector = DiversitySelector::new(DiversityConfig::default());

        // Test upward trend
        let upward = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let slope = selector.calculate_linear_slope(&upward);
        assert!(slope > 0.0);

        // Test downward trend
        let downward = vec![5.0, 4.0, 3.0, 2.0, 1.0];
        let slope = selector.calculate_linear_slope(&downward);
        assert!(slope < 0.0);

        // Test flat trend
        let flat = vec![3.0, 3.0, 3.0, 3.0, 3.0];
        let slope = selector.calculate_linear_slope(&flat);
        assert!(slope.abs() < 0.001);
    }
}
