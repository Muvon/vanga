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
    ) -> Result<ValidationSelectionResult> {
        if all_sequences.is_empty() {
            return Err(VangaError::DataError(
                "No sequences available for validation selection".to_string(),
            ));
        }

        let total_sequences = all_sequences.len();
        let target_val_size = (total_sequences as f64 * validation_ratio) as usize;

        log::info!(
            "🎯 Selecting balanced validation set: {} sequences ({:.1}% of {})",
            target_val_size,
            validation_ratio * 100.0,
            total_sequences
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
    pub fn create_diverse_splits(
        &self,
        balanced_dataset: &GloballyBalancedDataset,
        all_sequences: &[SequenceWithTargets],
        validation_ratio: f64,
        test_ratio: f64,
        target_types: &[TargetType],
        horizons: &[String],
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
    fn create_diverse_target_splits(
        &self,
        all_sequences: &[SequenceWithTargets],
        balanced_indices: &[usize],
        validation_ratio: f64,
        test_ratio: f64,
        target_type: TargetType,
        horizon: &str,
    ) -> Result<(Vec<usize>, Vec<usize>, Vec<usize>)> {
        let _target_key = (target_type, horizon.to_string());

        // Group sequences by class for balanced splitting
        let mut class_sequences: HashMap<i32, Vec<usize>> = HashMap::new();
        for &idx in balanced_indices {
            if let Some(class) = all_sequences[idx].get_target_class(target_type, horizon) {
                class_sequences.entry(class).or_default().push(idx);
            }
        }

        let mut train_indices = Vec::new();
        let mut val_indices = Vec::new();
        let mut test_indices = Vec::new();

        // Split each class independently to maintain balance
        for (class, class_indices) in class_sequences {
            let class_size = class_indices.len();
            let val_size = (class_size as f64 * validation_ratio) as usize;
            let test_size = (class_size as f64 * test_ratio) as usize;
            let train_size = class_size - val_size - test_size;

            log::debug!(
                "   Class {}: {} total → {} train, {} val, {} test",
                class,
                class_size,
                train_size,
                val_size,
                test_size
            );

            // Use our fast diversity selection for each split
            let (class_train, class_val, class_test) = self.create_diverse_class_splits(
                all_sequences,
                &class_indices,
                train_size,
                val_size,
                test_size,
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
    /// - **Validation**: MAX OVERLAP WITH TRAINING (similar patterns)
    /// - **Test**: MAX DIVERSITY FROM REMAINING (fresh patterns)
    ///
    /// This replaces the old temporal stratification approach for better
    /// training/validation/test split quality.
    fn create_diverse_class_splits(
        &self,
        all_sequences: &[SequenceWithTargets],
        class_indices: &[usize],
        train_size: usize,
        val_size: usize,
        test_size: usize,
    ) -> Result<(Vec<usize>, Vec<usize>, Vec<usize>)> {
        self.create_priority_based_class_splits(
            all_sequences,
            class_indices,
            train_size,
            val_size,
            test_size,
        )
    }

    /// Create PRIORITY-BASED splits within a single class
    ///
    /// **PRIORITY STRATEGY**:
    /// - **Training**: MAX DIVERSITY (uniform temporal spread for maximum pattern coverage)
    /// - **Validation**: MAX OVERLAP WITH TRAINING (similar patterns for generalization testing)
    /// - **Test**: MAX DIVERSITY FROM REMAINING (fresh patterns, no overlap with validation)
    ///
    /// This ensures:
    /// 1. Training sees the most diverse patterns possible
    /// 2. Validation tests generalization to SIMILAR patterns (overlapping)
    /// 3. Test evaluates performance on FRESH patterns (non-overlapping)
    fn create_priority_based_class_splits(
        &self,
        all_sequences: &[SequenceWithTargets],
        class_indices: &[usize],
        train_size: usize,
        val_size: usize,
        test_size: usize,
    ) -> Result<(Vec<usize>, Vec<usize>, Vec<usize>)> {
        let total_size = train_size + val_size + test_size;
        if total_size > class_indices.len() {
            return Err(VangaError::DataError(format!(
                "Requested splits ({}) exceed available sequences ({})",
                total_size,
                class_indices.len()
            )));
        }

        // Step 1: Sort all sequences by temporal position
        let mut temporal_sorted: Vec<(usize, usize)> = class_indices
            .iter()
            .map(|&idx| (idx, all_sequences[idx].start_idx))
            .collect();
        temporal_sorted.sort_by_key(|(_, start_idx)| *start_idx);

        // Extract just the indices in temporal order
        let sorted_indices: Vec<usize> = temporal_sorted.iter().map(|(idx, _)| *idx).collect();

        // Store temporal positions for overlap calculations
        let temporal_positions: Vec<usize> = temporal_sorted.iter().map(|(_, pos)| *pos).collect();
        let _min_pos = temporal_positions[0];

        let mut train_indices = Vec::new();
        let mut val_indices = Vec::new();
        let mut test_indices = Vec::new();

        // Step 2: TRAINING - MAX DIVERSITY (uniform temporal spread)
        // Select sequences at regular intervals across the entire temporal range
        if train_size > 0 {
            let train_interval = if train_size > 1 {
                (sorted_indices.len() - 1) as f64 / (train_size - 1) as f64
            } else {
                0.0
            };

            for i in 0..train_size {
                let idx = if train_size > 1 {
                    (i as f64 * train_interval) as usize
                } else {
                    sorted_indices.len() / 2 // Middle for single sample
                };
                train_indices.push(sorted_indices[idx]);
            }

            // Sort training indices for consistency
            train_indices.sort();

            log::debug!(
                "   🎯 TRAINING (Max Diversity): {} sequences at uniform temporal intervals",
                train_indices.len()
            );
        }

        // Step 3: VALIDATION - MAX OVERLAP WITH TRAINING
        // From remaining sequences, select those that OVERLAP most with training windows
        let remaining: Vec<usize> = sorted_indices
            .iter()
            .filter(|&&idx| !train_indices.contains(&idx))
            .copied()
            .collect();

        if val_size > 0 && !remaining.is_empty() {
            // Calculate overlap score for each remaining sequence with training set
            let mut scored: Vec<(usize, f64)> = remaining
                .iter()
                .map(|&idx| {
                    let overlap_score =
                        self.calculate_overlap_with_set(idx, &train_indices, all_sequences);
                    (idx, overlap_score)
                })
                .collect();

            // Sort by overlap score (highest first) for validation
            scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            // Select top val_size for validation
            val_indices.extend(
                scored
                    .iter()
                    .take(val_size.min(scored.len()))
                    .map(|(idx, _)| *idx),
            );

            // Sort validation indices for consistency
            val_indices.sort();

            let avg_overlap: f64 = if !val_indices.is_empty() {
                let total: f64 = val_indices
                    .iter()
                    .map(|&idx| self.calculate_overlap_with_set(idx, &train_indices, all_sequences))
                    .sum();
                total / val_indices.len() as f64
            } else {
                0.0
            };

            log::debug!(
                "   🎯 VALIDATION (Max Overlap): {} sequences, avg overlap with train: {:.2}",
                val_indices.len(),
                avg_overlap
            );
        }

        // Step 4: TEST - MAX DIVERSITY FROM REMAINING
        // From remaining sequences, select those that maximize temporal diversity
        let remaining_after_val: Vec<usize> = remaining
            .iter()
            .filter(|&&idx| !val_indices.contains(&idx))
            .copied()
            .collect();

        if test_size > 0 && !remaining_after_val.is_empty() {
            // Score by temporal distance from nearest training OR validation sequence
            let mut scored: Vec<(usize, f64)> = remaining_after_val
                .iter()
                .map(|&idx| {
                    let diversity_score = self.calculate_diversity_from_selected(
                        idx,
                        &train_indices,
                        &val_indices,
                        all_sequences,
                    );
                    (idx, diversity_score)
                })
                .collect();

            // Sort by diversity score (highest first) - prefer sequences far from others
            scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            // Select top test_size for test (they are already the most diverse)
            test_indices.extend(
                scored
                    .iter()
                    .take(test_size.min(scored.len()))
                    .map(|(idx, _)| *idx),
            );

            // Sort test indices for consistency
            test_indices.sort();

            log::debug!(
                "   🎯 TEST (Max Diversity from Remaining): {} sequences",
                test_indices.len()
            );
        }

        // Step 5: Handle any remaining due to rounding or exact size requirements
        let mut remaining_final: Vec<usize> = class_indices
            .iter()
            .filter(|&&idx| {
                !train_indices.contains(&idx)
                    && !val_indices.contains(&idx)
                    && !test_indices.contains(&idx)
            })
            .copied()
            .collect();
        remaining_final.sort();

        // Distribute remaining to reach exact target sizes
        while train_indices.len() < train_size && !remaining_final.is_empty() {
            train_indices.push(remaining_final.remove(0));
        }
        while val_indices.len() < val_size && !remaining_final.is_empty() {
            val_indices.push(remaining_final.remove(0));
        }
        while test_indices.len() < test_size && !remaining_final.is_empty() {
            test_indices.push(remaining_final.remove(0));
        }

        // Final validation
        if train_indices.len() != train_size
            || val_indices.len() != val_size
            || test_indices.len() != test_size
        {
            return Err(VangaError::DataError(format!(
                "Split allocation mismatch: train={}/{}, val={}/{}, test={}/{}",
                train_indices.len(),
                train_size,
                val_indices.len(),
                val_size,
                test_indices.len(),
                test_size
            )));
        }

        // Sort all indices for consistency
        train_indices.sort();
        val_indices.sort();
        test_indices.sort();

        log::debug!(
            "   📊 Split allocation: {} train, {} val, {} test (no overlaps)",
            train_indices.len(),
            val_indices.len(),
            test_indices.len()
        );

        Ok((train_indices, val_indices, test_indices))
    }

    /// Calculate overlap score between a sequence and a set of training indices
    /// Returns value between 0.0 (no overlap) and 1.0 (maximum overlap)
    fn calculate_overlap_with_set(
        &self,
        seq_idx: usize,
        reference_indices: &[usize],
        all_sequences: &[SequenceWithTargets],
    ) -> f64 {
        if reference_indices.is_empty() {
            return 0.0;
        }

        let seq = &all_sequences[seq_idx];
        let mut total_overlap = 0.0;
        let mut count = 0;

        for &ref_idx in reference_indices {
            let ref_seq = &all_sequences[ref_idx];
            total_overlap += seq.overlap_ratio(ref_seq);
            count += 1;
        }

        if count > 0 {
            // Normalize: average overlap across all reference sequences
            (total_overlap / count as f64).clamp(0.0, 1.0)
        } else {
            0.0
        }
    }

    /// Calculate diversity score - how far this sequence is from any selected sequence
    /// Higher score = more diverse (far from all selected)
    fn calculate_diversity_from_selected(
        &self,
        seq_idx: usize,
        train_indices: &[usize],
        val_indices: &[usize],
        all_sequences: &[SequenceWithTargets],
    ) -> f64 {
        let all_selected: Vec<usize> = train_indices
            .iter()
            .chain(val_indices.iter())
            .copied()
            .collect();

        if all_selected.is_empty() {
            return 1.0; // Maximum diversity if nothing selected yet
        }

        let seq = &all_sequences[seq_idx];
        let seq_start = seq.start_idx;
        let seq_end = seq.end_idx;

        if seq_end <= seq_start {
            return 0.0;
        }

        // Find minimum distance to any selected sequence
        let mut min_distance = usize::MAX;

        for &sel_idx in &all_selected {
            let sel_seq = &all_sequences[sel_idx];
            let sel_start = sel_seq.start_idx;
            let sel_end = sel_seq.end_idx;

            // Calculate gap between sequences
            let gap = if seq_start >= sel_end {
                seq_start - sel_end // seq after sel
            } else {
                sel_start.saturating_sub(seq_end) // seq before sel or overlapping
            };

            min_distance = min_distance.min(gap);
        }

        // Get temporal range for normalization
        let temporal_positions: Vec<usize> = all_selected
            .iter()
            .map(|&idx| all_sequences[idx].start_idx)
            .collect();
        let min_pos = *temporal_positions.iter().min().unwrap_or(&seq_start);
        let max_pos = *temporal_positions.iter().max().unwrap_or(&seq_start);
        let total_span = max_pos.saturating_sub(min_pos).max(1);

        // Normalize by total span
        let normalized_distance = min_distance as f64 / total_span as f64;

        // Also consider temporal position diversity (prefer extremes over center)
        let temporal_center = (min_pos + max_pos) as f64 / 2.0;
        let pos_from_center = (seq_start as f64 - temporal_center).abs();
        let max_distance_from_center = (max_pos - min_pos) as f64 / 2.0;
        let temporal_diversity = if max_distance_from_center > 0.0 {
            pos_from_center / max_distance_from_center
        } else {
            0.5
        };

        // Combined score: weighted average of distance and temporal position
        // Higher weight on distance to prefer non-overlapping sequences
        let combined_score = 0.7 * normalized_distance + 0.3 * temporal_diversity;

        combined_score.clamp(0.0, 1.0)
    }

    /// Select sequences for a target with specified count
    fn select_sequences_for_target(
        &self,
        all_sequences: &[SequenceWithTargets],
        target_type: &TargetType,
        horizon: &str,
        target_count: usize,
        exclude_indices: &[usize],
        window_range: Option<(usize, usize)>,
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
        )
    }

    /// Core algorithm for balanced selection with GUARANTEED perfect balance
    ///
    /// CRITICAL: This method MUST achieve EXACTLY equal sequences per class (20% each for 5-class system)
    /// Uses minimum available class count - NO sequence reuse allowed
    fn select_balanced_with_overlap_management(
        &self,
        all_sequences: &[SequenceWithTargets],
        mut class_sequences: HashMap<i32, Vec<usize>>,
        _target_total_sequences: usize, // Ignored - we use minimum class count
        exclude_indices: &[usize],
        target_type: TargetType, // NEW: For diversity selection
        horizon: &str,           // NEW: For diversity selection
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

        let mut total_overlap = 0.0;
        let mut count = 0;

        for i in 0..selected_indices.len() {
            for j in i + 1..selected_indices.len() {
                let overlap = all_sequences[selected_indices[i]]
                    .overlap_ratio(&all_sequences[selected_indices[j]]);
                total_overlap += overlap;
                count += 1;
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

        for &idx in selected_indices {
            if let Some(class) = all_sequences[idx].get_target_class(*target_type, horizon) {
                *distribution.entry(class).or_insert(0) += 1;
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
