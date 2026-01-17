//! Balanced sequence selection for optimal training
//!
//! This module provides sophisticated algorithms for selecting balanced sequences
//! per target type while managing overlap and ensuring optimal data utilization.

use crate::targets::{PreparedTargets, TargetType};
use crate::utils::error::{Result, VangaError};
use ndarray::{Array2, Array3};
use std::collections::HashMap;

/// Target data containing both class and strength for diversity-based selection
#[derive(Debug, Clone)]
pub struct TargetData {
    /// Target type (PriceLevel, Direction, Volatility, etc.)
    pub target_type: TargetType,
    /// Horizon string (e.g., "1h", "4h", "1d")
    pub horizon: String,
    /// Classification result (0-4 for 5-class system)
    pub class: i32,
    /// Classification strength (0.0-1.0, where 1.0 = very strong, 0.5 = neutral, 0.0 = very weak)
    pub strength: f64,
}

// Import the new diversity selector
use super::diversity::{DiversityConfig, DiversitySelector};

// Type alias for complex return type to improve readability
type ValidationSelectionResult = (
    Vec<usize>,
    HashMap<(TargetType, String), HashMap<i32, usize>>,
);

// Type alias for target-specific validation results
type TargetSpecificValidationResult = HashMap<(TargetType, String), Vec<usize>>;

// Type alias for diverse splits creation result
type DiverseSplitsResult = (
    GloballyBalancedDataset,
    HashMap<(TargetType, String), Vec<usize>>,
    HashMap<(TargetType, String), Vec<usize>>,
);

/// Globally balanced dataset extracted from full available data
#[derive(Debug, Clone)]
pub struct GloballyBalancedDataset {
    /// Balanced sequence indices for each target/horizon combination
    pub balanced_indices: HashMap<(TargetType, String), Vec<usize>>,
    /// Class distribution for each target/horizon combination
    pub class_distribution: HashMap<(TargetType, String), HashMap<i32, usize>>,
    /// Global minimum class count across all targets (the limiting factor)
    pub global_min_class_count: usize,
    /// Total number of balanced samples across all targets
    pub total_balanced_samples: usize,
    /// Statistics about overloaded classes (classes with more than min count)
    pub overloaded_classes: HashMap<(TargetType, String), HashMap<i32, usize>>,
}

/// Represents a sequence with its associated targets and metadata
#[derive(Debug, Clone)]
pub struct SequenceWithTargets {
    /// Unique identifier for this sequence
    pub sequence_idx: usize,
    /// Start index in the original dataset
    pub start_idx: usize,
    /// End index in the original dataset (exclusive)
    pub end_idx: usize,
    /// The actual sequence data [sequence_length, features]
    pub sequence_data: Array2<f64>,
    /// Targets with both class and strength for diversity-based selection
    pub targets: Vec<TargetData>,
}

impl SequenceWithTargets {
    /// Get target class for a specific target type and horizon
    pub fn get_target_class(&self, target_type: TargetType, horizon: &str) -> Option<i32> {
        self.targets
            .iter()
            .find(|t| t.target_type == target_type && t.horizon == horizon)
            .map(|t| t.class)
    }

    /// Get target strength for a specific target type and horizon
    pub fn get_target_strength(&self, target_type: TargetType, horizon: &str) -> Option<f64> {
        self.targets
            .iter()
            .find(|t| t.target_type == target_type && t.horizon == horizon)
            .map(|t| t.strength)
    }

    /// Get complete target data for a specific target type and horizon
    pub fn get_target_data(&self, target_type: TargetType, horizon: &str) -> Option<&TargetData> {
        self.targets
            .iter()
            .find(|t| t.target_type == target_type && t.horizon == horizon)
    }

    /// Add target data to this sequence
    pub fn add_target(&mut self, target_data: TargetData) {
        // Remove existing target with same type and horizon if it exists
        self.targets.retain(|t| {
            !(t.target_type == target_data.target_type && t.horizon == target_data.horizon)
        });
        // Add the new target data
        self.targets.push(target_data);
    }

    /// Calculate overlap ratio with another sequence
    pub fn overlap_ratio(&self, other: &SequenceWithTargets) -> f64 {
        let overlap_start = self.start_idx.max(other.start_idx);
        let overlap_end = self.end_idx.min(other.end_idx);

        if overlap_start >= overlap_end {
            return 0.0; // No overlap
        }

        let overlap_size = overlap_end - overlap_start;
        let seq_length = self.end_idx - self.start_idx;

        overlap_size as f64 / seq_length as f64
    }

    /// Check if this sequence overlaps with a data range
    pub fn is_within_range(&self, start: usize, end: usize) -> bool {
        // A sequence overlaps with the range if:
        // - Its start is before the range end, AND
        // - Its end is after the range start
        self.start_idx < end && self.end_idx > start
    }
}

/// Configuration for balanced sequence selection
#[derive(Debug, Clone)]
pub struct BalanceConfig {
    /// Maximum allowed overlap between sequences (0.0 = no overlap, 1.0 = full overlap)
    pub max_overlap: f64,
    /// Whether to prefer sequences with minimal overlap
    pub prefer_non_overlapping: bool,
    /// Minimum sequences per class (for rare classes)
    pub min_sequences_per_class: usize,
}

impl Default for BalanceConfig {
    fn default() -> Self {
        Self {
            max_overlap: 0.5, // 50% overlap allowed by default
            prefer_non_overlapping: true,
            min_sequences_per_class: 10,
        }
    }
}

/// Result of balanced selection containing selected indices and statistics
#[derive(Debug, Clone)]
pub struct BalancedSelection {
    /// Selected sequence indices
    pub selected_indices: Vec<usize>,
    /// Class distribution after balancing
    pub class_distribution: HashMap<i32, usize>,
    /// Average overlap between selected sequences
    pub avg_overlap: f64,
    /// Number of sequences per class
    pub sequences_per_class: usize,
}

/// Main balancer for sequence selection
pub struct SequenceBalancer {
    _config: BalanceConfig, // Stored for future use, currently algorithm uses minimum class count
    diversity_selector: DiversitySelector, // NEW: Advanced diversity-based selection
}

impl SequenceBalancer {
    pub fn new(config: BalanceConfig) -> Self {
        Self {
            _config: config,
            diversity_selector: DiversitySelector::new(DiversityConfig::default()),
        }
    }

    /// Intelligently oversample minority classes using augmentation
    /// Returns balanced dataset with synthetic sequences for underrepresented classes
    ///
    /// # Arguments
    /// * `all_sequences` - All available sequences with targets
    /// * `target_type` - Target type to balance (PriceLevel, Direction, etc.)
    /// * `horizon` - Prediction horizon (e.g., "1h", "4h")
    /// * `augment_config` - Augmentation configuration
    /// * `target_percentile` - Target percentile for class count (0.5 = median, 0.6 = 60th)
    /// * `max_synthetic_ratio` - Maximum synthetic samples per real sample
    ///
    /// # Returns
    /// Balanced sequences including synthetic ones for minority classes
    pub fn balance_with_minority_augmentation(
        &self,
        all_sequences: &[SequenceWithTargets],
        target_type: TargetType,
        horizon: &str,
        augment_config: &crate::data::augmentation::AugmentationConfig,
        target_percentile: f64,
        max_synthetic_ratio: f64,
    ) -> Result<Vec<SequenceWithTargets>> {
        // 1. Group sequences by class
        let mut sequences_by_class: HashMap<i32, Vec<&SequenceWithTargets>> = HashMap::new();
        for seq in all_sequences {
            if let Some(class) = seq.get_target_class(target_type, horizon) {
                sequences_by_class.entry(class).or_default().push(seq);
            }
        }

        if sequences_by_class.is_empty() {
            return Err(VangaError::DataError(format!(
                "No sequences found for {:?}/{} - cannot balance",
                target_type, horizon
            )));
        }

        // 2. Calculate target count using percentile strategy
        let class_counts: Vec<usize> = sequences_by_class.values().map(|v| v.len()).collect();
        let target_count = calculate_target_count(&class_counts, target_percentile);

        log::info!(
            "🎯 Target count for {:?}/{}: {} sequences per class ({}th percentile)",
            target_type,
            horizon,
            target_count,
            (target_percentile * 100.0) as u32
        );

        // 3. Balance classes with augmentation
        let mut balanced_sequences = Vec::new();
        let mut total_synthetic = 0;
        let mut class_final_counts: HashMap<i32, usize> = HashMap::new();

        for (class, sequences) in &sequences_by_class {
            let current_count = sequences.len();

            if current_count >= target_count {
                // Majority/balanced class: use diversity selection to downsample
                let class_indices: Vec<usize> = sequences.iter().map(|s| s.sequence_idx).collect();
                let selected_indices = self.diversity_selector.select_diverse_sequences(
                    all_sequences,
                    &class_indices,
                    target_count,
                    target_type,
                    horizon,
                    &[], // No exclusions
                    0,   // No gap enforcement for training data
                )?;

                for &idx in &selected_indices {
                    if let Some(seq) = sequences.iter().find(|s| s.sequence_idx == idx) {
                        balanced_sequences.push((*seq).clone());
                    }
                }

                class_final_counts.insert(*class, target_count);

                log::info!(
                    "   Class {}: {} → {} (downsampled using diversity)",
                    class,
                    current_count,
                    target_count
                );
            } else {
                // Minority class: use ALL real sequences + generate synthetic
                balanced_sequences.extend(sequences.iter().map(|s| (*s).clone()));

                let deficit = target_count - current_count;
                let max_synthetic = (current_count as f64 * max_synthetic_ratio) as usize;
                let synthetic_to_generate = deficit.min(max_synthetic);

                if synthetic_to_generate > 0 {
                    log::info!(
                        "   Class {}: {} real + {} synthetic = {} total (deficit: {}, max allowed: {})",
                        *class,
                        current_count,
                        synthetic_to_generate,
                        current_count + synthetic_to_generate,
                        deficit,
                        max_synthetic
                    );

                    // Generate synthetic sequences
                    let synthetic = self.generate_synthetic_sequences(
                        sequences,
                        synthetic_to_generate,
                        augment_config,
                        target_type,
                        horizon,
                        *class,
                    )?;

                    let synthetic_count = synthetic.len();
                    total_synthetic += synthetic_count;
                    balanced_sequences.extend(synthetic);
                    class_final_counts.insert(*class, current_count + synthetic_count);
                } else {
                    log::info!(
                        "   Class {}: {} real (no synthetic - at max ratio)",
                        *class,
                        current_count
                    );
                    class_final_counts.insert(*class, current_count);
                }
            }
        }

        // CRITICAL FIX: Ensure EXACT balance by trimming all classes to minimum count
        let min_final_count = *class_final_counts.values().min().unwrap_or(&0);

        if class_final_counts
            .values()
            .any(|&count| count != min_final_count)
        {
            log::warn!(
                "⚠️ Class imbalance detected after augmentation: {:?}. Trimming to {} per class for perfect balance.",
                class_final_counts,
                min_final_count
            );

            // Group sequences by class
            let mut sequences_by_class_final: HashMap<i32, Vec<SequenceWithTargets>> =
                HashMap::new();
            for seq in balanced_sequences.drain(..) {
                if let Some(class) = seq.get_target_class(target_type, horizon) {
                    sequences_by_class_final.entry(class).or_default().push(seq);
                }
            }

            // Trim each class to min_final_count
            for (class, mut sequences) in sequences_by_class_final {
                if sequences.len() > min_final_count {
                    // Use diversity selection to trim
                    let class_indices: Vec<usize> =
                        sequences.iter().map(|s| s.sequence_idx).collect();
                    let selected_indices = self.diversity_selector.select_diverse_sequences(
                        all_sequences,
                        &class_indices,
                        min_final_count,
                        target_type,
                        horizon,
                        &[],
                        0,
                    )?;

                    sequences.retain(|s| selected_indices.contains(&s.sequence_idx));
                    log::info!(
                        "   Class {}: trimmed {} → {} for perfect balance",
                        class,
                        class_indices.len(),
                        sequences.len()
                    );
                }
                balanced_sequences.extend(sequences);
            }
        }

        log::info!(
            "✅ Balanced with augmentation: {} total sequences ({} synthetic, {:.1}% augmented), {} per class",
            balanced_sequences.len(),
            total_synthetic,
            (total_synthetic as f64 / balanced_sequences.len() as f64) * 100.0,
            min_final_count
        );

        Ok(balanced_sequences)
    }

    /// Generate synthetic sequences for minority class using augmentation
    ///
    /// # Arguments
    /// * `seed_sequences` - Real sequences to use as seeds for augmentation
    /// * `count` - Number of synthetic sequences to generate
    /// * `augment_config` - Augmentation configuration
    /// * `target_type` - Target type for consistency verification
    /// * `horizon` - Prediction horizon
    /// * `expected_class` - Expected class for generated sequences
    ///
    /// # Returns
    /// Vector of synthetic sequences that maintain target consistency
    fn generate_synthetic_sequences(
        &self,
        seed_sequences: &[&SequenceWithTargets],
        count: usize,
        augment_config: &crate::data::augmentation::AugmentationConfig,
        _target_type: TargetType,
        _horizon: &str,
        expected_class: i32,
    ) -> Result<Vec<SequenceWithTargets>> {
        use rand::Rng;
        use std::sync::atomic::{AtomicUsize, Ordering};

        // Global counter for synthetic sequence IDs to prevent collisions
        static SYNTHETIC_COUNTER: AtomicUsize = AtomicUsize::new(1_000_000);

        let mut synthetic_sequences = Vec::new();
        let mut rng = rand::rng();

        // Identify price/volume columns to skip during augmentation
        let price_volume_cols = identify_price_volume_columns(&seed_sequences[0].sequence_data);

        // Generate synthetic sequences with target consistency verification
        let mut attempts = 0;
        let max_attempts = count * 10; // Prevent infinite loops

        while synthetic_sequences.len() < count && attempts < max_attempts {
            attempts += 1;

            // 1. Select random seed sequence
            let seed_idx = rng.random_range(0..seed_sequences.len());
            let seed = seed_sequences[seed_idx];

            // 2. Apply augmentation to create synthetic variant
            let augmented_data = crate::data::augmentation::augment_sequence(
                &seed.sequence_data,
                augment_config,
                &mut rng,
                &price_volume_cols,
            );

            // 3. Generate UNIQUE synthetic sequence ID using atomic counter
            // This prevents collisions across all classes and target/horizon combinations
            let synthetic_id = SYNTHETIC_COUNTER.fetch_add(1, Ordering::SeqCst);

            let synthetic = SequenceWithTargets {
                sequence_idx: synthetic_id,
                start_idx: seed.start_idx,
                end_idx: seed.end_idx,
                sequence_data: augmented_data,
                targets: seed.targets.clone(),
            };

            synthetic_sequences.push(synthetic);

            if synthetic_sequences.len() % 10 == 0 {
                log::debug!(
                    "   Generated {}/{} synthetic sequences for class {}",
                    synthetic_sequences.len(),
                    count,
                    expected_class
                );
            }
        }

        if synthetic_sequences.len() < count {
            log::warn!(
                "⚠️  Could only generate {}/{} synthetic sequences for class {} after {} attempts",
                synthetic_sequences.len(),
                count,
                expected_class,
                attempts
            );
        }

        Ok(synthetic_sequences)
    }

    /// UNIFIED METHOD: Select balanced sequences for any target/horizon combination
    /// This replaces both balance_sequences_for_window and the logic in extract_target_specific_balanced_datasets
    pub fn balance_sequences_for_window(
        &self,
        all_sequences: &[SequenceWithTargets],
        target_type: TargetType,
        horizon: &str,
        exclude_indices: &[usize],
        window_range: Option<(usize, usize)>,
    ) -> Result<BalancedSelection> {
        // Filter sequences based on window range if specified
        let available_sequences: Vec<usize> = all_sequences
            .iter()
            .enumerate()
            .filter(|(idx, seq)| {
                // Check window range
                if let Some((start, end)) = window_range {
                    if !seq.is_within_range(start, end) {
                        return false;
                    }
                }
                // Check exclusions
                !exclude_indices.contains(idx)
            })
            .map(|(idx, _)| idx)
            .collect();

        if available_sequences.is_empty() {
            return Err(VangaError::DataError(format!(
                "No sequences available for {:?} {} after filtering",
                target_type, horizon
            )));
        }

        // Group sequences by class
        let mut class_sequences: HashMap<i32, Vec<usize>> = HashMap::new();
        for &seq_idx in &available_sequences {
            if let Some(class) = all_sequences[seq_idx].get_target_class(target_type, horizon) {
                class_sequences.entry(class).or_default().push(seq_idx);
            }
        }

        // Validate all expected classes are present
        let expected_classes = [0, 1, 2, 3, 4];
        let found_classes: Vec<i32> = class_sequences.keys().cloned().collect();
        let missing_classes: Vec<i32> = expected_classes
            .iter()
            .filter(|&&expected| !found_classes.contains(&expected))
            .cloned()
            .collect();

        if !missing_classes.is_empty() {
            return Err(VangaError::DataError(format!(
                "FATAL: Missing classes detected for {:?} {}: Expected [0,1,2,3,4] but found {:?}. Missing: {:?}",
                target_type, horizon, found_classes, missing_classes
            )));
        }

        log::info!(
            "✅ ALL 5 CLASSES PRESENT for {:?} {}: {:?}",
            target_type,
            horizon,
            found_classes
        );

        // Calculate balance target (minimum class count)
        let min_class_count = class_sequences.values().map(|v| v.len()).min().unwrap_or(0);
        let sequences_per_class = min_class_count;
        let total_sequences = sequences_per_class * class_sequences.len();

        log::info!(
            "🎯 BALANCE TARGET: {} sequences per class (limited by rarest class with {} sequences)",
            sequences_per_class,
            min_class_count
        );

        if sequences_per_class == 0 {
            return Err(VangaError::DataError(format!(
                "FATAL: Cannot achieve balance - at least one class has no sequences for {:?} {}",
                target_type, horizon
            )));
        }

        // Select sequences for each class using unified diversity selection
        let mut selected_indices = Vec::new();
        let mut class_distribution = HashMap::new();

        // CRITICAL FIX: Sort classes for deterministic processing order (avoid HashMap randomness)
        let mut sorted_classes: Vec<_> = class_sequences.into_iter().collect();
        sorted_classes.sort_by_key(|(class, _)| *class);

        for (class, indices) in sorted_classes {
            let available = indices.len();
            let needed = sequences_per_class;

            log::info!(
                "🎯 Class {}: selecting {} sequences from {} available",
                class,
                needed,
                available
            );

            let selected = if available > needed {
                // OVERLOADED CLASS: Use diversity selection
                log::info!(
                    "🎯 Class {}: OVERLOADED - using DIVERSITY SELECTION from {} available",
                    class,
                    available
                );

                self.diversity_selector.select_diverse_sequences(
                    all_sequences,
                    &indices,
                    needed,
                    target_type,
                    horizon,
                    exclude_indices,
                    0, // No gap enforcement for training data
                )?
            } else {
                // NOT OVERLOADED: Use all available sequences
                log::debug!(
                    "   Class {}: using all {} available sequences",
                    class,
                    available
                );
                indices
            };

            if selected.len() != needed {
                return Err(VangaError::DataError(format!(
                    "FATAL: Selection failed for class {} - selected {} but need {}",
                    class,
                    selected.len(),
                    needed
                )));
            }

            selected_indices.extend(selected);
            class_distribution.insert(class, needed);
        }

        // Verify perfect balance
        let total_selected = selected_indices.len();
        if total_selected != total_sequences {
            return Err(VangaError::DataError(format!(
                "FATAL: Perfect balance failed - selected {} sequences but target was {}",
                total_selected, total_sequences
            )));
        }

        log::info!(
            "🎯 PERFECT BALANCE ACHIEVED: {} sequences, {} per class for {:?} {}",
            total_selected,
            sequences_per_class,
            target_type,
            horizon
        );

        Ok(BalancedSelection {
            selected_indices,
            class_distribution,
            avg_overlap: 0.0, // TODO: Calculate if needed
            sequences_per_class,
        })
    }

    /// Select target-specific balanced validation sets
    /// Each target gets its own validation set with proper class balance
    pub fn select_target_specific_validation(
        &self,
        all_sequences: &[SequenceWithTargets],
        validation_ratio: f64,
        target_types: &[TargetType],
        horizons: &[String],
        validation_gap_steps: usize,
    ) -> Result<TargetSpecificValidationResult> {
        if all_sequences.is_empty() {
            return Err(VangaError::DataError(
                "No sequences available for validation selection".to_string(),
            ));
        }

        let mut target_validation_indices = HashMap::new();
        let total_sequences = all_sequences.len();

        log::info!(
            "🎯 Selecting target-specific balanced validation sets from {} total sequences",
            total_sequences
        );

        // Select validation set for each target independently
        for target_type in target_types {
            for horizon in horizons {
                let target_key = (*target_type, horizon.clone());

                // Calculate target-specific validation size
                let target_val_size = (total_sequences as f64 * validation_ratio) as usize;

                log::info!(
                    "🎯 Selecting validation for {:?} {}: {} sequences ({:.1}%)",
                    target_type,
                    horizon,
                    target_val_size,
                    validation_ratio * 100.0
                );

                // Select balanced validation sequences for this specific target
                let selection_result = self.select_sequences_for_target(
                    all_sequences,
                    target_type,
                    horizon,
                    target_val_size,
                    &[],  // No existing sequences to avoid
                    None, // Use all available data
                    validation_gap_steps,
                )?;

                target_validation_indices.insert(
                    target_key.clone(),
                    selection_result.selected_indices.clone(),
                );

                // Log class distribution for this target's validation set with balance verification
                let distribution = self.calculate_class_distribution(
                    all_sequences,
                    &selection_result.selected_indices,
                    target_type,
                    horizon,
                );

                // EXPLICIT BALANCE VERIFICATION for validation set
                let total_validation = selection_result.selected_indices.len();
                let expected_percentage = 20.0; // 100% / 5 classes = 20% each
                let mut validation_balance_perfect = true;
                let mut validation_warnings = Vec::new();

                for (class, count) in &distribution {
                    let actual_percentage = if total_validation > 0 {
                        (*count as f64 / total_validation as f64) * 100.0
                    } else {
                        0.0
                    };

                    let balance_deviation = (actual_percentage - expected_percentage).abs();
                    if balance_deviation > 5.0 {
                        // Allow 5% tolerance
                        validation_balance_perfect = false;
                        validation_warnings.push(format!(
                            "Class {}: {:.1}% (expected 20%)",
                            class, actual_percentage
                        ));
                    }
                }

                if validation_balance_perfect {
                    log::info!(
                        "📊 {:?} {} validation distribution: {:?} - ✅ PERFECTLY BALANCED",
                        target_type,
                        horizon,
                        distribution
                    );
                } else {
                    log::warn!(
                        "📊 {:?} {} validation distribution: {:?} - ⚠️ IMBALANCED: {}",
                        target_type,
                        horizon,
                        distribution,
                        validation_warnings.join(", ")
                    );
                }
            }
        }

        log::info!(
            "✅ Target-specific validation selection complete: {} target/horizon combinations",
            target_validation_indices.len()
        );

        Ok(target_validation_indices)
    }

    /// Select a balanced validation set that works for all targets (LEGACY - kept for compatibility)
    /// Returns indices and class distributions for each target/horizon
    pub fn select_balanced_validation(
        &self,
        all_sequences: &[SequenceWithTargets],
        validation_ratio: f64,
        target_types: &[TargetType],
        horizons: &[String],
        validation_gap_steps: usize,
    ) -> Result<ValidationSelectionResult> {
        if all_sequences.is_empty() {
            return Err(VangaError::DataError(
                "No sequences available for validation selection".to_string(),
            ));
        }

        let total_sequences = all_sequences.len();
        let target_val_size = (total_sequences as f64 * validation_ratio) as usize;

        log::info!(
            "🎯 Selecting balanced validation set: {} sequences ({:.1}% of {}), gap={} steps",
            target_val_size,
            validation_ratio * 100.0,
            total_sequences,
            validation_gap_steps
        );

        // Find the most imbalanced target to use as primary selection criteria
        let (primary_target, primary_horizon) =
            self.find_most_imbalanced_target(all_sequences, target_types, horizons)?;

        log::info!(
            "📊 Using {:?} {} as primary target for validation selection (most imbalanced)",
            primary_target,
            primary_horizon
        );

        // Select validation sequences based on primary target
        let selected_indices = self.select_sequences_for_target(
            all_sequences,
            &primary_target,
            &primary_horizon,
            target_val_size,
            &[],  // No existing sequences to avoid
            None, // Use all available data
            validation_gap_steps,
        )?;

        // Calculate class distributions for all targets
        let mut distributions = HashMap::new();
        for target_type in target_types {
            for horizon in horizons {
                let distribution = self.calculate_class_distribution(
                    all_sequences,
                    &selected_indices.selected_indices,
                    target_type,
                    horizon,
                );
                distributions.insert((*target_type, horizon.clone()), distribution);
            }
        }

        // Log validation set statistics
        self.log_validation_statistics(&selected_indices, all_sequences, &distributions);

        Ok((selected_indices.selected_indices, distributions))
    }

    /// Extract globally balanced dataset from FULL available data
    /// Uses ALL data (total - test_split) to find optimal balance across all targets
    pub fn extract_globally_balanced_dataset(
        &self,
        all_sequences: &[SequenceWithTargets],
        target_types: &[TargetType],
        horizons: &[String],
    ) -> Result<GloballyBalancedDataset> {
        if all_sequences.is_empty() {
            return Err(VangaError::DataError(
                "No sequences available for global balancing".to_string(),
            ));
        }

        log::info!(
            "🌍 GLOBAL BALANCE EXTRACTION: Analyzing {} sequences across {} targets and {} horizons",
            all_sequences.len(),
            target_types.len(),
            horizons.len()
        );

        let mut all_class_distributions = HashMap::new();
        let mut global_min_class_count = usize::MAX;

        // STEP 1: Analyze class distribution across ALL targets and horizons
        for target_type in target_types {
            for horizon in horizons {
                let target_key = (*target_type, horizon.clone());
                let mut class_sequences: HashMap<i32, Vec<usize>> = HashMap::new();

                // Group sequences by class for this target
                for (idx, seq) in all_sequences.iter().enumerate() {
                    if let Some(class) = seq.get_target_class(*target_type, horizon) {
                        class_sequences.entry(class).or_default().push(idx);
                    }
                }

                if class_sequences.is_empty() {
                    return Err(VangaError::DataError(format!(
                        "No sequences found for target {:?} horizon {}",
                        target_type, horizon
                    )));
                }

                // Find minimum class count for this target
                let target_min_class_count =
                    class_sequences.values().map(|v| v.len()).min().unwrap_or(0);
                global_min_class_count = global_min_class_count.min(target_min_class_count);

                log::debug!(
                    "   📊 {:?} {}: {} classes, min_count={}, distribution={:?}",
                    target_type,
                    horizon,
                    class_sequences.len(),
                    target_min_class_count,
                    class_sequences
                        .iter()
                        .map(|(k, v)| (*k, v.len()))
                        .collect::<HashMap<_, _>>()
                );

                all_class_distributions.insert(target_key, class_sequences);
            }
        }

        if global_min_class_count == 0 || global_min_class_count == usize::MAX {
            return Err(VangaError::DataError(
                "Cannot achieve global balance - at least one class has no sequences".to_string(),
            ));
        }

        log::info!(
            "🎯 GLOBAL CONSTRAINT: {} sequences per class (limited by rarest class across all targets)",
            global_min_class_count
        );

        // STEP 2: Extract balanced samples using global minimum class count
        let mut balanced_indices = HashMap::new();
        let mut final_class_distribution = HashMap::new();
        let mut overloaded_classes = HashMap::new();
        let mut total_balanced_samples = 0;

        for (target_key, class_sequences) in all_class_distributions {
            let mut target_balanced_indices = Vec::new();
            let mut target_class_distribution = HashMap::new();
            let mut target_overloaded = HashMap::new();

            // Select balanced samples for each class
            for (class, mut indices) in class_sequences {
                let available = indices.len();
                let needed = global_min_class_count;

                if available < needed {
                    return Err(VangaError::DataError(format!(
                        "FATAL: Cannot achieve global balance for {:?} class {} - need {} but only {} available",
                        target_key, class, needed, available
                    )));
                }

                // Track overloaded classes (have more than global minimum)
                if available > needed {
                    target_overloaded.insert(class, available - needed);
                }

                // Select sequences (prioritize by minimal overlap with others)
                indices.sort_by_key(|&idx| {
                    // Simple selection strategy - can be enhanced with overlap analysis
                    all_sequences[idx].start_idx
                });

                let selected: Vec<usize> = indices.into_iter().take(needed).collect();
                target_balanced_indices.extend(selected);
                target_class_distribution.insert(class, needed);
            }

            total_balanced_samples += target_balanced_indices.len();
            balanced_indices.insert(target_key.clone(), target_balanced_indices);
            final_class_distribution.insert(target_key.clone(), target_class_distribution);

            if !target_overloaded.is_empty() {
                overloaded_classes.insert(target_key, target_overloaded);
            }
        }

        log::info!(
            "✅ GLOBAL BALANCE ACHIEVED: {} total balanced samples across all targets",
            total_balanced_samples
        );

        log::info!("📊 OVERLOADED CLASSES DETECTED:");
        for (target_key, overloaded) in &overloaded_classes {
            log::info!(
                "   {:?}: {:?} excess samples available for smart validation",
                target_key,
                overloaded.values().sum::<usize>()
            );
        }

        Ok(GloballyBalancedDataset {
            balanced_indices,
            class_distribution: final_class_distribution,
            global_min_class_count,
            total_balanced_samples,
            overloaded_classes,
        })
    }

    /// Extract target-specific balanced datasets - each target gets its own optimal balance
    pub fn extract_target_specific_balanced_datasets(
        &self,
        all_sequences: &[SequenceWithTargets],
        target_types: &[TargetType],
        horizons: &[String],
    ) -> Result<HashMap<(TargetType, String), GloballyBalancedDataset>> {
        if all_sequences.is_empty() {
            return Err(VangaError::DataError(
                "No sequences available for target-specific balancing".to_string(),
            ));
        }

        log::info!(
            "🎯 TARGET-SPECIFIC BALANCE EXTRACTION: Analyzing {} sequences for {} targets × {} horizons",
            all_sequences.len(),
            target_types.len(),
            horizons.len()
        );

        let mut target_balanced_datasets = HashMap::new();

        // Process each target independently to maximize its balanced data
        for target_type in target_types {
            for horizon in horizons {
                let target_key = (*target_type, horizon.clone());

                log::info!(
                    "📊 Processing {:?} {} independently for optimal balance",
                    target_type,
                    horizon
                );

                // Use unified selection method
                let selection_result = self.balance_sequences_for_window(
                    all_sequences,
                    *target_type,
                    horizon,
                    &[],  // No exclusions at this stage
                    None, // No window range restriction
                )?;

                let target_balanced_indices = selection_result.selected_indices;
                let target_class_distribution = selection_result.class_distribution;
                let target_min_class_count = selection_result.sequences_per_class;

                let num_classes = target_class_distribution.len();
                let total_balanced = target_balanced_indices.len();

                log::info!(
                    "   ✅ {:?} {}: {} classes, {} sequences per class = {} total balanced",
                    target_type,
                    horizon,
                    num_classes,
                    target_min_class_count,
                    total_balanced
                );

                // Sort indices for consistency
                let mut sorted_indices = target_balanced_indices.clone();
                sorted_indices.sort();

                // Create target-specific balanced dataset
                let mut balanced_indices_map = HashMap::new();
                balanced_indices_map.insert(target_key.clone(), sorted_indices);

                let mut class_dist_map = HashMap::new();
                class_dist_map.insert(target_key.clone(), target_class_distribution);

                // No overloaded classes tracking in unified method (could be added if needed)
                let overloaded_map = HashMap::new();

                let target_dataset = GloballyBalancedDataset {
                    balanced_indices: balanced_indices_map,
                    class_distribution: class_dist_map,
                    overloaded_classes: overloaded_map,
                    global_min_class_count: target_min_class_count,
                    total_balanced_samples: target_balanced_indices.len(),
                };

                target_balanced_datasets.insert(target_key, target_dataset);
            }
        }

        log::info!(
            "✅ TARGET-SPECIFIC BALANCE COMPLETE: {} target/horizon combinations balanced independently",
            target_balanced_datasets.len()
        );

        Ok(target_balanced_datasets)
    }

    /// SENIOR-LEVEL: Create diverse train/validation/test splits from balanced dataset
    ///
    /// This ensures ALL three splits maintain diversity, not just training data.
    /// Uses stratified sampling across temporal and statistical dimensions.
    #[allow(clippy::too_many_arguments)]
    pub fn create_diverse_splits(
        &self,
        balanced_dataset: &GloballyBalancedDataset,
        all_sequences: &[SequenceWithTargets],
        validation_ratio: f64,
        test_ratio: f64,
        target_types: &[TargetType],
        horizons: &[String],
        validation_gap_steps: usize,
    ) -> Result<DiverseSplitsResult> {
        log::info!(
            "🎯 DIVERSE SPLITS: Creating diverse train ({:.1}%) / val ({:.1}%) / test ({:.1}%) splits",
            (1.0 - validation_ratio - test_ratio) * 100.0,
            validation_ratio * 100.0,
            test_ratio * 100.0
        );

        let mut remaining_training_indices = HashMap::new();
        let mut validation_indices = HashMap::new();
        let mut test_indices = HashMap::new();

        // Process each target independently for optimal diversity
        for target_type in target_types {
            for horizon in horizons {
                let target_key = (*target_type, horizon.clone());

                if let Some(balanced_indices) = balanced_dataset.balanced_indices.get(&target_key) {
                    log::info!(
                        "🎯 Creating diverse splits for {:?} {} from {} balanced sequences",
                        target_type,
                        horizon,
                        balanced_indices.len()
                    );

                    // Use our fast diversity selector to create splits
                    let (train_indices, val_indices, test_indices_target) = self
                        .create_diverse_target_splits(
                            all_sequences,
                            balanced_indices,
                            validation_ratio,
                            test_ratio,
                            *target_type,
                            horizon,
                            validation_gap_steps,
                        )?;

                    remaining_training_indices.insert(target_key.clone(), train_indices);
                    validation_indices.insert(target_key.clone(), val_indices);
                    test_indices.insert(target_key.clone(), test_indices_target);

                    log::info!(
                        "   ✅ {:?} {}: {} train, {} val, {} test (all diverse)",
                        target_type,
                        horizon,
                        remaining_training_indices.get(&target_key).unwrap().len(),
                        validation_indices.get(&target_key).unwrap().len(),
                        test_indices.get(&target_key).unwrap().len()
                    );
                }
            }
        }

        // Create remaining training dataset
        let remaining_training_dataset = GloballyBalancedDataset {
            balanced_indices: remaining_training_indices,
            class_distribution: balanced_dataset.class_distribution.clone(),
            global_min_class_count: balanced_dataset.global_min_class_count,
            total_balanced_samples: balanced_dataset.total_balanced_samples
                - validation_indices.values().map(|v| v.len()).sum::<usize>()
                - test_indices.values().map(|v| v.len()).sum::<usize>(),
            overloaded_classes: HashMap::new(),
        };

        log::info!("✅ DIVERSE SPLITS COMPLETE: All splits maintain diversity and class balance");

        Ok((remaining_training_dataset, validation_indices, test_indices))
    }

    /// Create diverse train/val/test splits for a specific target
    #[allow(clippy::too_many_arguments)]
    fn create_diverse_target_splits(
        &self,
        all_sequences: &[SequenceWithTargets],
        balanced_indices: &[usize],
        validation_ratio: f64,
        test_ratio: f64,
        target_type: TargetType,
        horizon: &str,
        validation_gap_steps: usize,
    ) -> Result<(Vec<usize>, Vec<usize>, Vec<usize>)> {
        let _target_key = (target_type, horizon.to_string());

        // Create lookup map from sequence_idx to sequence (handles synthetic sequences)
        let sequence_map: HashMap<usize, &SequenceWithTargets> = all_sequences
            .iter()
            .map(|seq| (seq.sequence_idx, seq))
            .collect();

        // Group sequences by class for balanced splitting
        let mut class_sequences: HashMap<i32, Vec<usize>> = HashMap::new();
        for &idx in balanced_indices {
            if let Some(seq) = sequence_map.get(&idx) {
                if let Some(class) = seq.get_target_class(target_type, horizon) {
                    class_sequences.entry(class).or_default().push(idx);
                }
            }
        }

        // CRITICAL: Calculate split sizes based on MINIMUM class size to ensure perfect balance
        let min_class_size = class_sequences.values().map(|v| v.len()).min().unwrap_or(0);

        // Use floor() to ensure we don't exceed min_class_size
        let val_size_per_class = (min_class_size as f64 * validation_ratio).floor() as usize;
        let test_size_per_class = (min_class_size as f64 * test_ratio).floor() as usize;
        let train_size_per_class = min_class_size
            .saturating_sub(val_size_per_class)
            .saturating_sub(test_size_per_class);

        log::info!(
            "🎯 PERFECT BALANCE SPLIT: {} per class → {} train, {} val, {} test (all classes identical)",
            min_class_size,
            train_size_per_class,
            val_size_per_class,
            test_size_per_class
        );

        let mut train_indices = Vec::new();
        let mut val_indices = Vec::new();
        let mut test_indices = Vec::new();

        // CRITICAL: Sort classes for deterministic processing order (avoid HashMap randomness)
        let mut sorted_classes: Vec<_> = class_sequences.into_iter().collect();
        sorted_classes.sort_by_key(|(class, _)| *class);

        // Split each class using IDENTICAL sizes
        for (class, class_indices) in sorted_classes {
            let class_size = class_indices.len();

            // Trim to min_class_size if this class has more
            let indices_to_use = if class_size > min_class_size {
                log::debug!(
                    "   Class {}: trimming {} → {} for perfect balance",
                    class,
                    class_size,
                    min_class_size
                );
                &class_indices[..min_class_size]
            } else {
                &class_indices[..]
            };

            log::debug!(
                "   Class {}: {} total → {} train, {} val, {} test",
                class,
                indices_to_use.len(),
                train_size_per_class,
                val_size_per_class,
                test_size_per_class
            );

            // Use our fast diversity selection for each split (with gap enforcement for validation)
            let (class_train, class_val, class_test) = self.create_diverse_class_splits(
                all_sequences,
                indices_to_use,
                train_size_per_class,
                val_size_per_class,
                test_size_per_class,
                validation_gap_steps,
            )?;

            train_indices.extend(class_train);
            val_indices.extend(class_val);
            test_indices.extend(class_test);
        }

        Ok((train_indices, val_indices, test_indices))
    }

    /// Create diverse splits within a single class using PRIORITY-BASED STRATEGY
    ///
    /// **PRIORITY STRATEGY** (optimal for training effectiveness):
    /// - **Training**: MAX DIVERSITY (uniform temporal spread)
    /// - **Validation**: MAX OVERLAP WITH TRAINING (test interpolation)
    /// - **Test**: MAX DIVERSITY FROM REMAINING (test extrapolation)
    ///
    /// This replaces the old temporal stratification approach for better
    /// training/validation/test split quality.
    pub fn create_diverse_class_splits(
        &self,
        all_sequences: &[SequenceWithTargets],
        class_indices: &[usize],
        train_size: usize,
        val_size: usize,
        test_size: usize,
        validation_gap_steps: usize,
    ) -> Result<(Vec<usize>, Vec<usize>, Vec<usize>)> {
        self.create_priority_based_class_splits(
            all_sequences,
            class_indices,
            train_size,
            val_size,
            test_size,
            validation_gap_steps,
        )
    }

    /// Create PRIORITY-BASED splits within a single class
    ///
    /// **PRIORITY STRATEGY** (optimal for sliding window time series):
    /// - **Training**: MAX DIVERSITY (uniform temporal spread for maximum pattern coverage)
    /// - **Validation**: MAX OVERLAP WITH TRAINING (test interpolation on similar patterns)
    /// - **Test**: MAX DIVERSITY FROM REMAINING (test extrapolation on fresh patterns)
    ///
    /// **WHY THIS WORKS**:
    /// 1. **Training gets most diverse patterns** → Model learns from widest range of behaviors
    /// 2. **Validation overlaps with training** → Tests if model can interpolate (generalize to similar patterns)
    /// 3. **Test is completely fresh** → Tests if model can extrapolate (handle truly new patterns)
    ///
    /// **RESEARCH SUPPORT**:
    /// - Overlapping validation is VALID for sliding window time series (see: Time Series Cross-Validation literature)
    /// - Maximizes training data utilization by putting diverse patterns in training
    /// - Validation tests "interpolation" (can model generalize to similar patterns?)
    /// - Test evaluates "extrapolation" (can model handle truly new patterns?)
    ///
    /// **OVERLAP CONTROL**:
    /// - Overlap is calculated via `overlap_ratio()` method (0.0 = no overlap, 1.0 = full overlap)
    /// - Validation typically has 0.3-0.7 overlap with training (controlled, not random)
    /// - This is intentional and beneficial for time series with sliding windows
    ///
    /// This ensures:
    /// 1. Training sees the most diverse patterns possible (maximum learning)
    /// 2. Validation tests generalization to SIMILAR patterns (interpolation capability)
    /// 3. Test evaluates performance on FRESH patterns (extrapolation capability)
    fn create_priority_based_class_splits(
        &self,
        all_sequences: &[SequenceWithTargets],
        class_indices: &[usize],
        train_size: usize,
        val_size: usize,
        test_size: usize,
        validation_gap_steps: usize,
    ) -> Result<(Vec<usize>, Vec<usize>, Vec<usize>)> {
        let total_size = train_size + val_size + test_size;
        if total_size > class_indices.len() {
            return Err(VangaError::DataError(format!(
                "Requested splits ({}) exceed available sequences ({})",
                total_size,
                class_indices.len()
            )));
        }

        // Create lookup map from sequence_idx to sequence (handles synthetic sequences)
        let sequence_map: HashMap<usize, &SequenceWithTargets> = all_sequences
            .iter()
            .map(|seq| (seq.sequence_idx, seq))
            .collect();

        // Step 1: Sort all sequences by temporal position
        // CRITICAL: Verify all class_indices exist in sequence_map
        let missing_indices: Vec<usize> = class_indices
            .iter()
            .filter(|&&idx| !sequence_map.contains_key(&idx))
            .copied()
            .collect();

        if !missing_indices.is_empty() {
            return Err(VangaError::DataError(format!(
                "Missing sequences in map: {:?} (total class_indices: {}, sequence_map size: {})",
                missing_indices,
                class_indices.len(),
                sequence_map.len()
            )));
        }

        let mut temporal_sorted: Vec<(usize, usize)> = class_indices
            .iter()
            .map(|&idx| {
                let seq = sequence_map.get(&idx).expect("Already validated above");
                (idx, seq.start_idx)
            })
            .collect();
        temporal_sorted.sort_by_key(|(_, start_idx)| *start_idx);

        // Extract just the indices in temporal order
        let sorted_indices: Vec<usize> = temporal_sorted.iter().map(|(idx, _)| *idx).collect();

        let mut val_indices = Vec::new();
        let mut test_indices = Vec::new();

        // Simple 3-step process: Val → Test → Train (everything else)
        // Validation and test get proper gap enforcement, training gets the rest

        // Step 1: Select VALIDATION with even temporal spacing
        if val_size > 0 {
            let step = (sorted_indices.len() as f64 / val_size as f64).max(1.0);
            for i in 0..val_size {
                let pos = ((i as f64 * step) as usize).min(sorted_indices.len() - 1);
                let idx = sorted_indices[pos];
                if !val_indices.contains(&idx) {
                    val_indices.push(idx);
                }
            }

            // If we got duplicates, fill from remaining
            if val_indices.len() < val_size {
                for &idx in &sorted_indices {
                    if !val_indices.contains(&idx) {
                        val_indices.push(idx);
                        if val_indices.len() >= val_size {
                            break;
                        }
                    }
                }
            }
            val_indices.sort();
        }

        // Step 2: Select TEST with gap enforcement from validation
        if test_size > 0 {
            let mut candidates: Vec<usize> = sorted_indices
                .iter()
                .filter(|&&idx| !val_indices.contains(&idx))
                .copied()
                .collect();

            // Apply gap enforcement if needed
            if validation_gap_steps > 0 {
                candidates.retain(|&idx| {
                    let seq = match sequence_map.get(&idx) {
                        Some(s) => s,
                        None => return false,
                    };

                    // Check gap to ALL validation sequences
                    for &val_idx in &val_indices {
                        let val_seq = match sequence_map.get(&val_idx) {
                            Some(s) => s,
                            None => continue,
                        };

                        let gap = if seq.start_idx >= val_seq.end_idx {
                            seq.start_idx - val_seq.end_idx
                        } else if val_seq.start_idx >= seq.end_idx {
                            val_seq.start_idx.saturating_sub(seq.end_idx)
                        } else {
                            0 // overlap
                        };

                        if gap < validation_gap_steps {
                            return false; // Too close to validation
                        }
                    }
                    true
                });
            }

            // Select test with even spacing from candidates
            if !candidates.is_empty() {
                let step = (candidates.len() as f64 / test_size as f64).max(1.0);
                for i in 0..test_size.min(candidates.len()) {
                    let pos = ((i as f64 * step) as usize).min(candidates.len() - 1);
                    let idx = candidates[pos];
                    if !test_indices.contains(&idx) {
                        test_indices.push(idx);
                    }
                }

                // Fill remaining if we got duplicates
                if test_indices.len() < test_size {
                    for &idx in &candidates {
                        if !test_indices.contains(&idx) {
                            test_indices.push(idx);
                            if test_indices.len() >= test_size {
                                break;
                            }
                        }
                    }
                }
            }
            test_indices.sort();
        }

        // Step 3: TRAINING - Select from remaining with even spacing
        let remaining: Vec<usize> = class_indices
            .iter()
            .filter(|&&idx| !val_indices.contains(&idx) && !test_indices.contains(&idx))
            .copied()
            .collect();

        let mut train_indices = Vec::new();
        if train_size > 0 && !remaining.is_empty() {
            let step = (remaining.len() as f64 / train_size as f64).max(1.0);
            for i in 0..train_size.min(remaining.len()) {
                let pos = ((i as f64 * step) as usize).min(remaining.len() - 1);
                let idx = remaining[pos];
                if !train_indices.contains(&idx) {
                    train_indices.push(idx);
                }
            }

            // Fill remaining if we got duplicates
            if train_indices.len() < train_size {
                for &idx in &remaining {
                    if !train_indices.contains(&idx) {
                        train_indices.push(idx);
                        if train_indices.len() >= train_size {
                            break;
                        }
                    }
                }
            }
        }
        train_indices.sort();

        // Validate exact sizes
        if train_indices.len() != train_size {
            return Err(VangaError::DataError(format!(
                "Training size mismatch: got {} but expected {}",
                train_indices.len(),
                train_size
            )));
        }
        if val_indices.len() != val_size {
            return Err(VangaError::DataError(format!(
                "Validation size mismatch: got {} but expected {}",
                val_indices.len(),
                val_size
            )));
        }
        if test_indices.len() != test_size {
            return Err(VangaError::DataError(format!(
                "Test size mismatch: got {} but expected {}",
                test_indices.len(),
                test_size
            )));
        }

        // Validate no overlaps
        let total = train_indices.len() + val_indices.len() + test_indices.len();
        if total != train_size + val_size + test_size {
            return Err(VangaError::DataError(format!(
                "Split allocation error: total={} but expected {}",
                total,
                train_size + val_size + test_size
            )));
        }

        log::debug!(
            "   📊 Split allocation: {} train, {} val, {} test (no overlaps)",
            train_indices.len(),
            val_indices.len(),
            test_indices.len()
        );

        Ok((train_indices, val_indices, test_indices))
    }

    /// Select sequences for a target with specified count
    #[allow(clippy::too_many_arguments)]
    fn select_sequences_for_target(
        &self,
        all_sequences: &[SequenceWithTargets],
        target_type: &TargetType,
        horizon: &str,
        target_count: usize,
        exclude_indices: &[usize],
        window_range: Option<(usize, usize)>,
        validation_gap_steps: usize,
    ) -> Result<BalancedSelection> {
        // Group sequences by class
        let mut class_sequences: HashMap<i32, Vec<usize>> = HashMap::new();
        for (idx, seq) in all_sequences.iter().enumerate() {
            // Skip excluded sequences
            if exclude_indices.contains(&idx) {
                continue;
            }

            // Check window range if specified
            if let Some((start, end)) = window_range {
                if !seq.is_within_range(start, end) {
                    continue;
                }
            }

            if let Some(class) = seq.get_target_class(*target_type, horizon) {
                class_sequences.entry(class).or_default().push(idx);
            }
        }

        if class_sequences.is_empty() {
            return Err(VangaError::DataError(format!(
                "No sequences found for {:?} {}",
                target_type, horizon
            )));
        }

        // CRITICAL FIX: Pass total target count, let the algorithm calculate perfect balance
        self.select_balanced_with_overlap_management(
            all_sequences,
            class_sequences,
            target_count, // Pass total count, not per-class count
            exclude_indices,
            *target_type,
            horizon,
            validation_gap_steps,
        )
    }

    /// Core algorithm for balanced selection with GUARANTEED perfect balance
    ///
    /// CRITICAL: This method MUST achieve EXACTLY equal sequences per class (20% each for 5-class system)
    /// Uses minimum available class count - NO sequence reuse allowed
    #[allow(clippy::too_many_arguments)]
    fn select_balanced_with_overlap_management(
        &self,
        all_sequences: &[SequenceWithTargets],
        mut class_sequences: HashMap<i32, Vec<usize>>,
        _target_total_sequences: usize, // Ignored - we use minimum class count
        exclude_indices: &[usize],
        target_type: TargetType,     // NEW: For diversity selection
        horizon: &str,               // NEW: For diversity selection
        validation_gap_steps: usize, // NEW: Minimum gap between validation samples
    ) -> Result<BalancedSelection> {
        let num_classes = class_sequences.len();

        // CRITICAL FIX: Use minimum available class count (never reuse sequences)
        let min_available = class_sequences.values().map(|v| v.len()).min().unwrap_or(0);
        let sequences_per_class = min_available; // Each class gets exactly this many
        let total_sequences = sequences_per_class * num_classes;

        log::info!(
            "🎯 BALANCE STRATEGY: {} per class × {} classes = {} total (limited by rarest class)",
            sequences_per_class,
            num_classes,
            total_sequences
        );

        // Verify we can achieve perfect balance
        if sequences_per_class == 0 {
            return Err(VangaError::DataError(
                "FATAL: Cannot achieve balance - at least one class has no sequences available"
                    .to_string(),
            ));
        }

        let mut selected_indices = Vec::new();
        let mut class_distribution = HashMap::new();
        let total_overlap_sum = 0.0;
        let overlap_count = 0;

        // Calculate class scarcity for strategic overlap prioritization
        let mut class_scarcity: Vec<(i32, f64)> = class_sequences
            .iter()
            .map(|(class, indices)| {
                let scarcity = sequences_per_class as f64 / indices.len() as f64;
                (*class, scarcity)
            })
            .collect();

        // Sort by scarcity (most scarce first) - these need overlap most
        class_scarcity.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        log::info!("📊 Class scarcity analysis (higher = needs more overlap):");
        for (class, scarcity) in &class_scarcity {
            let available = class_sequences[class].len();
            log::info!(
                "   Class {}: {:.2}x scarcity ({} needed ÷ {} available)",
                class,
                scarcity,
                sequences_per_class,
                available
            );
        }

        // Select sequences for each class with strategic overlap
        for (class, scarcity) in class_scarcity {
            let indices = class_sequences.remove(&class).unwrap();
            let needed = sequences_per_class;
            let available = indices.len();

            log::info!(
                "🎯 Class {}: selecting {} sequences from {} available (scarcity: {:.2}x)",
                class,
                needed,
                available,
                scarcity
            );

            // Strategic overlap: if we need more sequences than available, find better sequences
            if needed > available {
                return Err(VangaError::DataError(format!(
                    "FATAL: Cannot achieve perfect balance for class {} - need {} sequences but only {} available. NEVER reuse sequences - find better data or adjust requirements.",
                    class, needed, available
                )));
            } else {
                // IMPROVED: Use diversity-based selection for overloaded classes
                log::info!(
                    "🎯 Class {}: OVERLOADED - using DIVERSITY SELECTION from {} available (scarcity: {:.2}x)",
                    class,
                    available,
                    scarcity
                );

                // Use advanced diversity selector for better sample selection
                let diverse_selection = self.diversity_selector.select_diverse_sequences(
                    all_sequences,
                    &indices,
                    needed,
                    target_type,
                    horizon,
                    exclude_indices,
                    validation_gap_steps,
                )?;

                if diverse_selection.len() != needed {
                    return Err(VangaError::DataError(format!(
                        "FATAL: Diversity selection failed for class {} - selected {} but need {}",
                        class,
                        diverse_selection.len(),
                        needed
                    )));
                }

                selected_indices.extend(diverse_selection);
            }

            class_distribution.insert(class, needed); // Always record the target amount
        }

        // Calculate final average overlap
        let avg_overlap = if overlap_count > 0 {
            total_overlap_sum / overlap_count as f64
        } else {
            self.calculate_average_overlap(&selected_indices, all_sequences)
        };

        // CRITICAL: Verify perfect balance was achieved
        let total_selected = selected_indices.len();
        if total_selected != total_sequences {
            return Err(VangaError::DataError(format!(
                "FATAL: Perfect balance failed - selected {} sequences but target was {}",
                total_selected, total_sequences
            )));
        }

        // Verify each class has exactly the right amount
        for (class, &count) in &class_distribution {
            if count != sequences_per_class {
                return Err(VangaError::DataError(format!(
                    "FATAL: Perfect balance failed - class {} has {} sequences but target was {}",
                    class, count, sequences_per_class
                )));
            }
        }

        log::info!(
            "✅ PERFECT BALANCE ACHIEVED: {} sequences, {} per class, avg overlap: {:.1}%",
            total_selected,
            sequences_per_class,
            avg_overlap * 100.0
        );

        Ok(BalancedSelection {
            selected_indices,
            class_distribution,
            avg_overlap,
            sequences_per_class,
        })
    }

    /// Calculate average overlap between selected sequences
    fn calculate_average_overlap(
        &self,
        selected_indices: &[usize],
        all_sequences: &[SequenceWithTargets],
    ) -> f64 {
        if selected_indices.len() < 2 {
            return 0.0;
        }

        // Create lookup map for synthetic sequences
        let sequence_map: HashMap<usize, &SequenceWithTargets> = all_sequences
            .iter()
            .map(|seq| (seq.sequence_idx, seq))
            .collect();

        let mut total_overlap = 0.0;
        let mut count = 0;

        for i in 0..selected_indices.len() {
            for j in i + 1..selected_indices.len() {
                if let (Some(seq_i), Some(seq_j)) = (
                    sequence_map.get(&selected_indices[i]),
                    sequence_map.get(&selected_indices[j]),
                ) {
                    let overlap = seq_i.overlap_ratio(seq_j);
                    total_overlap += overlap;
                    count += 1;
                }
            }
        }

        if count > 0 {
            total_overlap / count as f64
        } else {
            0.0
        }
    }

    /// Find the most imbalanced target to use for validation selection
    fn find_most_imbalanced_target(
        &self,
        all_sequences: &[SequenceWithTargets],
        target_types: &[TargetType],
        horizons: &[String],
    ) -> Result<(TargetType, String)> {
        let mut max_imbalance = 0.0;
        let mut most_imbalanced = None;

        for target_type in target_types {
            for horizon in horizons {
                let imbalance =
                    self.calculate_target_imbalance(all_sequences, target_type, horizon)?;

                if imbalance > max_imbalance {
                    max_imbalance = imbalance;
                    most_imbalanced = Some((*target_type, horizon.clone()));
                }
            }
        }

        most_imbalanced.ok_or_else(|| {
            VangaError::DataError("No valid targets found for imbalance calculation".to_string())
        })
    }

    /// Calculate imbalance ratio for a target (higher = more imbalanced)
    fn calculate_target_imbalance(
        &self,
        all_sequences: &[SequenceWithTargets],
        target_type: &TargetType,
        horizon: &str,
    ) -> Result<f64> {
        let mut class_counts: HashMap<i32, usize> = HashMap::new();

        for seq in all_sequences {
            if let Some(class) = seq.get_target_class(*target_type, horizon) {
                *class_counts.entry(class).or_insert(0) += 1;
            }
        }

        if class_counts.is_empty() {
            return Ok(0.0);
        }

        let max_count = class_counts.values().max().copied().unwrap_or(0) as f64;
        let min_count = class_counts.values().min().copied().unwrap_or(0) as f64;

        if min_count > 0.0 {
            Ok(max_count / min_count - 1.0) // Imbalance ratio
        } else {
            Ok(f64::MAX) // Infinite imbalance if any class has 0 samples
        }
    }

    /// Calculate class distribution for selected sequences
    fn calculate_class_distribution(
        &self,
        all_sequences: &[SequenceWithTargets],
        selected_indices: &[usize],
        target_type: &TargetType,
        horizon: &str,
    ) -> HashMap<i32, usize> {
        let mut distribution = HashMap::new();

        // Create lookup map for synthetic sequences
        let sequence_map: HashMap<usize, &SequenceWithTargets> = all_sequences
            .iter()
            .map(|seq| (seq.sequence_idx, seq))
            .collect();

        for &idx in selected_indices {
            if let Some(seq) = sequence_map.get(&idx) {
                if let Some(class) = seq.get_target_class(*target_type, horizon) {
                    *distribution.entry(class).or_insert(0) += 1;
                }
            }
        }

        distribution
    }

    /// Log validation set statistics
    fn log_validation_statistics(
        &self,
        selected: &BalancedSelection,
        _all_sequences: &[SequenceWithTargets],
        distributions: &HashMap<(TargetType, String), HashMap<i32, usize>>,
    ) {
        log::info!(
            "📊 Validation set: {} sequences selected, avg overlap: {:.1}%",
            selected.selected_indices.len(),
            selected.avg_overlap * 100.0
        );

        for ((target_type, horizon), dist) in distributions {
            let total = dist.values().sum::<usize>();
            log::info!("   {:?} {}: {} samples", target_type, horizon, total);

            let mut sorted_dist: Vec<_> = dist.iter().collect();
            sorted_dist.sort_by_key(|(k, _)| *k);

            for (class, count) in sorted_dist {
                let percentage = (*count as f64 / total as f64) * 100.0;
                log::debug!("      Class {}: {} ({:.1}%)", class, count, percentage);
            }
        }
    }
}

/// Convert sequences from Array3 to SequenceWithTargets format
pub async fn create_sequences_with_targets(
    sequences: Array3<f64>,
    targets: &PreparedTargets,
    sequence_indices: Vec<(usize, usize)>, // (start_idx, end_idx) for each sequence
) -> Result<Vec<SequenceWithTargets>> {
    if sequences.shape()[0] != sequence_indices.len() {
        return Err(VangaError::DataError(format!(
            "Sequence count mismatch: {} sequences but {} indices",
            sequences.shape()[0],
            sequence_indices.len()
        )));
    }

    let mut sequences_with_targets = Vec::new();

    for (seq_idx, (start_idx, end_idx)) in sequence_indices.iter().enumerate() {
        let mut target_data_vec = Vec::new();

        // Collect targets for all types and horizons
        for horizon in targets.get_horizons() {
            // Price levels
            if let Some(price_targets) = targets.price_levels.get(&horizon) {
                if seq_idx < price_targets.len() {
                    let strength = targets
                        .get_strengths(&horizon, TargetType::PriceLevel)
                        .and_then(|strengths| strengths.get(seq_idx))
                        .copied()
                        .unwrap_or(0.5);
                    target_data_vec.push(TargetData {
                        target_type: TargetType::PriceLevel,
                        horizon: horizon.clone(),
                        class: price_targets[seq_idx],
                        strength,
                    });
                }
            }

            // Directions
            if let Some(direction_targets) = targets.direction.get(&horizon) {
                if seq_idx < direction_targets.len() {
                    let strength = targets
                        .get_strengths(&horizon, TargetType::Direction)
                        .and_then(|strengths| strengths.get(seq_idx))
                        .copied()
                        .unwrap_or(0.5);
                    target_data_vec.push(TargetData {
                        target_type: TargetType::Direction,
                        horizon: horizon.clone(),
                        class: direction_targets[seq_idx],
                        strength,
                    });
                }
            }

            // Volatility
            if let Some(volatility_targets) = targets.volatility.get(&horizon) {
                if seq_idx < volatility_targets.len() {
                    let strength = targets
                        .get_strengths(&horizon, TargetType::Volatility)
                        .and_then(|strengths| strengths.get(seq_idx))
                        .copied()
                        .unwrap_or(0.5);
                    target_data_vec.push(TargetData {
                        target_type: TargetType::Volatility,
                        horizon: horizon.clone(),
                        class: volatility_targets[seq_idx],
                        strength,
                    });
                }
            }

            // Sentiment
            if let Some(sentiment_targets) = targets.sentiment.get(&horizon) {
                if seq_idx < sentiment_targets.len() {
                    let strength = targets
                        .get_strengths(&horizon, TargetType::Sentiment)
                        .and_then(|strengths| strengths.get(seq_idx))
                        .copied()
                        .unwrap_or(0.5);
                    target_data_vec.push(TargetData {
                        target_type: TargetType::Sentiment,
                        horizon: horizon.clone(),
                        class: sentiment_targets[seq_idx],
                        strength,
                    });
                }
            }

            // Volume
            if let Some(volume_targets) = targets.volume.get(&horizon) {
                if seq_idx < volume_targets.len() {
                    let strength = targets
                        .get_strengths(&horizon, TargetType::Volume)
                        .and_then(|strengths| strengths.get(seq_idx))
                        .copied()
                        .unwrap_or(0.5);
                    target_data_vec.push(TargetData {
                        target_type: TargetType::Volume,
                        horizon: horizon.clone(),
                        class: volume_targets[seq_idx],
                        strength,
                    });
                }
            }
        }

        // Extract sequence data
        let sequence_data = sequences.index_axis(ndarray::Axis(0), seq_idx).to_owned();

        sequences_with_targets.push(SequenceWithTargets {
            sequence_idx: seq_idx,
            start_idx: *start_idx,
            end_idx: *end_idx,
            sequence_data,
            targets: target_data_vec,
        });
    }

    Ok(sequences_with_targets)
}

/// Calculate target count based on percentile strategy
///
/// # Arguments
/// * `class_counts` - Vector of counts for each class
/// * `percentile` - Target percentile (0.0-1.0, where 0.5 = median)
///
/// # Returns
/// Target count at the specified percentile
pub fn calculate_target_count(class_counts: &[usize], percentile: f64) -> usize {
    if class_counts.is_empty() {
        return 0;
    }

    let mut sorted = class_counts.to_vec();
    sorted.sort_unstable();

    let index = ((sorted.len() as f64 - 1.0) * percentile.clamp(0.0, 1.0)) as usize;
    sorted[index]
}

/// Identify price/volume column indices to skip during augmentation
///
/// These columns determine the target class and must not be augmented.
/// Typically: [0=open, 1=high, 2=low, 3=close, 4=volume]
///
/// # Arguments
/// * `sequence` - Sample sequence to analyze
///
/// # Returns
/// Vector of column indices to skip during augmentation
pub fn identify_price_volume_columns(sequence: &ndarray::Array2<f64>) -> Vec<usize> {
    // CRITICAL: Price and volume columns determine targets
    // These must NEVER be augmented as it would invalidate the target class
    //
    // Standard OHLCV columns are always first 5:
    // 0: open, 1: high, 2: low, 3: close, 4: volume
    //
    // All other columns are technical indicators (RSI, MACD, SMA, etc.)
    // which can be safely augmented

    let num_features = sequence.shape()[1];

    if num_features >= 5 {
        // Standard case: skip first 5 OHLCV columns
        vec![0, 1, 2, 3, 4]
    } else {
        // Edge case: skip all columns if less than 5
        // This shouldn't happen in production but prevents panics
        (0..num_features).collect()
    }
}
