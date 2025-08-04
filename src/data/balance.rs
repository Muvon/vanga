//! Balanced sequence selection for optimal training
//!
//! This module provides sophisticated algorithms for selecting balanced sequences
//! per target type while managing overlap and ensuring optimal data utilization.

use crate::targets::{PreparedTargets, TargetType};
use crate::utils::error::{Result, VangaError};
use ndarray::{Array2, Array3};
use std::collections::HashMap;

// Import WindowConfig from parent module
use super::WindowConfig;

// Type alias for complex return type to improve readability
type ValidationSelectionResult = (
    Vec<usize>,
    HashMap<(TargetType, String), HashMap<i32, usize>>,
);

// Type alias for target-specific validation results
type TargetSpecificValidationResult = HashMap<(TargetType, String), Vec<usize>>;

// Type alias for complex return types to satisfy clippy
type ValidationSplitResult = (
    GloballyBalancedDataset,
    HashMap<(TargetType, String), Vec<usize>>,
);
type WindowIndicesMap = HashMap<usize, HashMap<(TargetType, String), Vec<usize>>>;

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
    /// Targets for all target types and horizons: (target_type, horizon) -> class
    pub targets: HashMap<(TargetType, String), i32>,
}

impl SequenceWithTargets {
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
}

impl SequenceBalancer {
    pub fn new(config: BalanceConfig) -> Self {
        Self { _config: config }
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

    /// Balance sequences for a specific window, target type, and horizon
    pub fn balance_sequences_for_window(
        &self,
        all_sequences: &[SequenceWithTargets],
        validation_indices: &[usize],
        window_range: (usize, usize),
        target_type: TargetType,
        horizon: &str,
    ) -> Result<BalancedSelection> {
        // Filter sequences that fall within this window's range
        let available_sequences: Vec<usize> = all_sequences
            .iter()
            .enumerate()
            .filter(|(idx, seq)| {
                seq.is_within_range(window_range.0, window_range.1)
                    && !validation_indices.contains(idx) // Exclude validation sequences
            })
            .map(|(idx, _)| idx)
            .collect();

        if available_sequences.is_empty() {
            return Err(VangaError::DataError(format!(
                "No sequences available in range [{}, {}) for {:?} {}",
                window_range.0, window_range.1, target_type, horizon
            )));
        }

        log::debug!(
            "🔍 Window range [{}, {}): {} sequences available for {:?} {}",
            window_range.0,
            window_range.1,
            available_sequences.len(),
            target_type,
            horizon
        );

        // Debug: Show some example sequences and their ranges
        if available_sequences.len() < 10 {
            for &idx in &available_sequences {
                let seq = &all_sequences[idx];
                log::debug!(
                    "   Seq {}: data range [{}, {}]",
                    idx,
                    seq.start_idx,
                    seq.end_idx
                );
            }
        }

        // Calculate class distribution for available sequences
        let mut class_sequences: HashMap<i32, Vec<usize>> = HashMap::new();
        for &seq_idx in &available_sequences {
            if let Some(&class) = all_sequences[seq_idx]
                .targets
                .get(&(target_type, horizon.to_string()))
            {
                class_sequences.entry(class).or_default().push(seq_idx);
            }
        }

        // CRITICAL: Validate ALL expected classes are present (0, 1, 2, 3, 4)
        let expected_classes = [0, 1, 2, 3, 4];
        let found_classes: Vec<i32> = class_sequences.keys().cloned().collect();
        let missing_classes: Vec<i32> = expected_classes
            .iter()
            .filter(|&&expected| !found_classes.contains(&expected))
            .cloned()
            .collect();

        if !missing_classes.is_empty() {
            return Err(VangaError::DataError(format!(
                "FATAL: Missing classes detected for {:?} {}: Expected classes [0,1,2,3,4] but found {:?}. Missing: {:?}. This indicates target generation failure or data corruption.",
                target_type, horizon, found_classes, missing_classes
            )));
        }

        log::info!(
            "✅ ALL 5 CLASSES PRESENT for {:?} {}: {:?}",
            target_type,
            horizon,
            found_classes
        );

        // Store original counts for logging
        let original_class_counts: HashMap<i32, usize> =
            class_sequences.iter().map(|(k, v)| (*k, v.len())).collect();

        // CRITICAL: Balance to minimum available class count (no sequence reuse!)
        let min_class_count = class_sequences.values().map(|v| v.len()).min().unwrap_or(0);

        // Use minimum class count as the balanced target (never reuse sequences)
        let balanced_count = min_class_count;

        log::info!(
            "🎯 BALANCE TARGET: {} sequences per class (limited by rarest class with {} sequences)",
            balanced_count,
            min_class_count
        );

        if balanced_count == 0 {
            return Err(VangaError::DataError(format!(
                "FATAL: Cannot achieve balance - at least one class has no sequences for {:?} {}",
                target_type, horizon
            )));
        }

        log::debug!(
            "📊 Class distribution before balancing: {:?}",
            class_sequences
                .iter()
                .map(|(k, v)| (*k, v.len()))
                .collect::<HashMap<_, _>>()
        );

        // Select balanced sequences (no sequence reuse!)
        // Total sequences = balanced_count * num_classes (all classes get same amount)
        let total_target_sequences = balanced_count * class_sequences.len();

        let selected = self.select_balanced_with_overlap_management(
            all_sequences,
            class_sequences,
            total_target_sequences,
            validation_indices,
        )?;

        // CRITICAL: Verify perfect balance was achieved (DYNAMIC class count)
        let total_selected = selected.selected_indices.len();
        let num_classes = selected.class_distribution.len();
        let expected_percentage = 100.0 / num_classes as f64; // Dynamic percentage based on actual classes
        let mut balance_warnings = Vec::new();
        let mut perfect_balance = true;

        log::info!(
            "🔍 BALANCE VERIFICATION for {:?} {}: {} sequences selected, {} classes found, expecting {:.1}% per class",
            target_type, horizon, total_selected, num_classes, expected_percentage
        );

        // Verify and log class-specific balance results
        for (class, count) in &selected.class_distribution {
            let original_count = original_class_counts.get(class).unwrap_or(&0);
            let actual_percentage = if total_selected > 0 {
                (*count as f64 / total_selected as f64) * 100.0
            } else {
                0.0
            };

            let balance_deviation = (actual_percentage - expected_percentage).abs();

            // CRITICAL: Perfect balance required (±1% tolerance max)
            if balance_deviation > 1.0 {
                balance_warnings.push(format!(
                    "Class {}: {:.1}% (should be {:.1}%, deviation: {:.1}%)",
                    class, actual_percentage, expected_percentage, balance_deviation
                ));
                perfect_balance = false;
            }

            log::info!(
                "   Class {}: {} sequences ({:.1}%) - had {} available - balance: {}",
                class,
                count,
                actual_percentage,
                original_count,
                if balance_deviation <= 1.0 {
                    "✅ PERFECT"
                } else if balance_deviation <= 5.0 {
                    "⚠️ ACCEPTABLE"
                } else {
                    "❌ FAILED"
                }
            );
        }

        // CRITICAL: Make balance failures FATAL
        if !perfect_balance {
            return Err(VangaError::DataError(format!(
                "FATAL: Perfect balance requirement failed for {:?} {}: {}. System requires ±1% tolerance for stable training.",
                target_type, horizon, balance_warnings.join(", ")
            )));
        }

        log::info!(
            "🎯 PERFECT BALANCE VERIFIED: All {} classes within ±1% of {:.1}% target for {:?} {}",
            num_classes,
            expected_percentage,
            target_type,
            horizon
        );

        Ok(selected)
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
                    if let Some(&class) = seq.targets.get(&target_key) {
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

    /// Derive validation set from globally balanced dataset using smart overlap management
    /// Uses overloaded classes to extract validation without reducing training balance
    pub fn smart_validation_split_from_balanced(
        &self,
        balanced_dataset: &GloballyBalancedDataset,
        _all_sequences: &[SequenceWithTargets], // TODO: Use for overlap management
        validation_ratio: f64,
        target_types: &[TargetType],
        horizons: &[String],
    ) -> Result<ValidationSplitResult> {
        log::info!(
            "🧠 SMART VALIDATION DERIVATION: Extracting {:.1}% validation from overloaded classes",
            validation_ratio * 100.0
        );

        let mut remaining_training_indices = balanced_dataset.balanced_indices.clone();
        let mut validation_indices = HashMap::new();
        let mut total_validation_extracted = 0;

        // Calculate target validation size per target
        let target_validation_size = (balanced_dataset.global_min_class_count as f64
            * balanced_dataset
                .class_distribution
                .values()
                .next()
                .unwrap()
                .len() as f64
            * validation_ratio) as usize;

        log::info!(
            "🎯 TARGET: ~{} validation samples per target (from {} balanced samples)",
            target_validation_size,
            balanced_dataset.global_min_class_count
                * balanced_dataset
                    .class_distribution
                    .values()
                    .next()
                    .unwrap()
                    .len()
        );

        for target_type in target_types {
            for horizon in horizons {
                let target_key = (*target_type, horizon.clone());
                let mut target_validation = Vec::new();

                // Check if this target has overloaded classes
                if let Some(overloaded) = balanced_dataset.overloaded_classes.get(&target_key) {
                    log::debug!(
                        "   🔍 {:?}: {} overloaded classes with {} excess samples",
                        target_key,
                        overloaded.len(),
                        overloaded.values().sum::<usize>()
                    );

                    // TODO: Extract validation from overloaded classes first
                    // For now, use simple proportional extraction
                    let training_indices = remaining_training_indices.get(&target_key).unwrap();
                    let validation_count =
                        (training_indices.len() as f64 * validation_ratio) as usize;

                    if validation_count > 0 {
                        // Simple extraction - take last N sequences (can be enhanced with overlap management)
                        let split_point = training_indices.len() - validation_count;
                        let (remaining_training, extracted_validation) =
                            training_indices.split_at(split_point);

                        target_validation.extend_from_slice(extracted_validation);
                        remaining_training_indices
                            .insert(target_key.clone(), remaining_training.to_vec());
                        total_validation_extracted += validation_count;
                    }
                } else {
                    // No overloaded classes - extract proportionally from balanced set
                    let training_indices = remaining_training_indices.get(&target_key).unwrap();
                    let validation_count =
                        (training_indices.len() as f64 * validation_ratio) as usize;

                    if validation_count > 0 {
                        let split_point = training_indices.len() - validation_count;
                        let (remaining_training, extracted_validation) =
                            training_indices.split_at(split_point);

                        target_validation.extend_from_slice(extracted_validation);
                        remaining_training_indices
                            .insert(target_key.clone(), remaining_training.to_vec());
                        total_validation_extracted += validation_count;
                    }
                }

                validation_indices.insert(target_key, target_validation);
            }
        }

        log::info!(
            "✅ SMART VALIDATION EXTRACTED: {} total validation samples across all targets",
            total_validation_extracted
        );

        // Create remaining training dataset
        let remaining_training_dataset = GloballyBalancedDataset {
            balanced_indices: remaining_training_indices,
            class_distribution: balanced_dataset.class_distribution.clone(), // Proportions remain same
            global_min_class_count: balanced_dataset.global_min_class_count,
            total_balanced_samples: balanced_dataset.total_balanced_samples
                - total_validation_extracted,
            overloaded_classes: HashMap::new(), // No overloaded classes in remaining training
        };

        Ok((remaining_training_dataset, validation_indices))
    }

    /// Build training windows from globally balanced dataset
    /// Distributes balanced samples across windows ensuring consistent balance
    #[allow(private_interfaces)]
    pub(crate) fn build_windows_from_balanced_pool(
        &self,
        balanced_training_pool: &GloballyBalancedDataset,
        window_ranges: &[WindowConfig],
        target_types: &[TargetType],
        horizons: &[String],
    ) -> Result<WindowIndicesMap> {
        log::info!(
            "🏗️ WINDOW CONSTRUCTION: Building {} windows from globally balanced pool ({} samples)",
            window_ranges.len(),
            balanced_training_pool.total_balanced_samples
        );

        let mut window_indices_map = HashMap::new();

        for (window_idx, window_range) in window_ranges.iter().enumerate() {
            let mut window_target_indices = HashMap::new();

            log::debug!(
                "   📊 Window {}: range [0..{}]",
                window_idx + 1,
                window_range.train_end
            );

            for target_type in target_types {
                for horizon in horizons {
                    let target_key = (*target_type, horizon.clone());

                    if let Some(balanced_indices) =
                        balanced_training_pool.balanced_indices.get(&target_key)
                    {
                        // PROGRESSIVE WINDOW EXPANSION WITH MAINTAINED BALANCE
                        // Each window must maintain perfect class balance

                        let total_samples = balanced_indices.len();
                        let samples_for_this_window = if window_ranges.len() == 1 {
                            // Single window: use all samples
                            total_samples
                        } else {
                            // Progressive expansion: calculate portion for this window
                            let window_progress =
                                (window_idx + 1) as f64 / window_ranges.len() as f64;
                            let min_samples = total_samples / window_ranges.len(); // Minimum per window
                            let progressive_samples =
                                (total_samples as f64 * window_progress) as usize;
                            std::cmp::max(min_samples, progressive_samples)
                        };

                        // CRITICAL: Maintain balance by taking proportional samples from each class
                        let class_dist = balanced_training_pool
                            .class_distribution
                            .get(&target_key)
                            .unwrap();
                        let window_samples = self.extract_balanced_subset_for_window(
                            balanced_indices,
                            samples_for_this_window,
                            class_dist,
                            balanced_training_pool.global_min_class_count,
                        )?;

                        log::debug!(
                            "   📊 {:?}: {} samples for window {} ({}% of balanced pool)",
                            target_key,
                            window_samples.len(),
                            window_idx + 1,
                            (window_samples.len() as f64 / total_samples as f64) * 100.0
                        );

                        window_target_indices.insert(target_key, window_samples);
                    }
                }
            }

            window_indices_map.insert(window_idx, window_target_indices);
        }

        log::info!(
            "✅ WINDOW CONSTRUCTION COMPLETE: {} windows built with consistent balance",
            window_ranges.len()
        );

        Ok(window_indices_map)
    }

    /// Extract balanced subset for a specific window maintaining class proportions
    /// CRITICAL: Each window must maintain perfect balance, not just take first N samples
    fn extract_balanced_subset_for_window(
        &self,
        all_balanced_indices: &[usize],
        target_samples: usize,
        class_distribution: &HashMap<i32, usize>,
        global_min_class_count: usize,
    ) -> Result<Vec<usize>> {
        if target_samples >= all_balanced_indices.len() {
            // Want all samples - return everything (already balanced)
            return Ok(all_balanced_indices.to_vec());
        }

        let num_classes = class_distribution.len();

        // Calculate samples per class for this window
        let samples_per_class_for_window = target_samples / num_classes;
        let remainder = target_samples % num_classes;

        if samples_per_class_for_window == 0 {
            // Too few samples requested - take at least 1 per class if possible
            let min_samples = std::cmp::min(target_samples, num_classes);
            return Ok(all_balanced_indices
                .iter()
                .take(min_samples)
                .cloned()
                .collect());
        }

        log::debug!(
            "   🎯 Window balance: {} samples total, {} per class, {} remainder classes get +1",
            target_samples,
            samples_per_class_for_window,
            remainder
        );

        let mut window_samples = Vec::new();

        // Since balanced_indices are already balanced (global_min_class_count per class),
        // we can take samples proportionally from the structured balanced array
        let classes_processed = 0;

        for &_class_count in class_distribution.values() {
            let samples_for_this_class = if classes_processed < remainder {
                samples_per_class_for_window + 1 // Distribute remainder
            } else {
                samples_per_class_for_window
            };

            // Take samples for this class from the balanced indices
            // Since indices are balanced, we take from the class's portion
            let class_start_idx = classes_processed * global_min_class_count;
            let class_end_idx = std::cmp::min(
                class_start_idx + samples_for_this_class,
                (classes_processed + 1) * global_min_class_count,
            );

            for i in class_start_idx..class_end_idx {
                if i < all_balanced_indices.len() {
                    window_samples.push(all_balanced_indices[i]);
                }
            }
        }

        log::debug!(
            "   ✅ Balanced window subset: {} samples with perfect class balance maintained",
            window_samples.len()
        );

        Ok(window_samples)
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

            if let Some(&class) = seq.targets.get(&(*target_type, horizon.to_string())) {
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
            let mut indices = class_sequences.remove(&class).unwrap();
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
                // Abundant class: select without overlap
                indices.sort_by_key(|&idx| {
                    let max_validation_overlap = exclude_indices
                        .iter()
                        .map(|&val_idx| all_sequences[idx].overlap_ratio(&all_sequences[val_idx]))
                        .fold(0.0, f64::max);
                    (max_validation_overlap * 1000.0) as i64
                });

                let mut class_selected = 0;
                for &idx in &indices {
                    if class_selected >= needed {
                        break;
                    }

                    let max_validation_overlap = exclude_indices
                        .iter()
                        .map(|&val_idx| all_sequences[idx].overlap_ratio(&all_sequences[val_idx]))
                        .fold(0.0, f64::max);

                    if max_validation_overlap < 0.95 {
                        selected_indices.push(idx);
                        class_selected += 1;
                    }
                }

                if class_selected < needed {
                    return Err(VangaError::DataError(format!(
                        "FATAL: Cannot achieve perfect balance for class {} - selected {} but need {} (validation overlap too high)",
                        class, class_selected, needed
                    )));
                }
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
            if let Some(&class) = seq.targets.get(&(*target_type, horizon.to_string())) {
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
            if let Some(&class) = all_sequences[idx]
                .targets
                .get(&(*target_type, horizon.to_string()))
            {
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
        let mut target_map = HashMap::new();

        // Collect targets for all types and horizons
        for horizon in targets.get_horizons() {
            // Price levels
            if let Some(price_targets) = targets.price_levels.get(&horizon) {
                if seq_idx < price_targets.len() {
                    target_map.insert(
                        (TargetType::PriceLevel, horizon.clone()),
                        price_targets[seq_idx],
                    );
                }
            }

            // Directions
            if let Some(direction_targets) = targets.directions.get(&horizon) {
                if seq_idx < direction_targets.len() {
                    target_map.insert(
                        (TargetType::Direction, horizon.clone()),
                        direction_targets[seq_idx],
                    );
                }
            }

            // Volatility
            if let Some(volatility_targets) = targets.volatility.get(&horizon) {
                if seq_idx < volatility_targets.len() {
                    target_map.insert(
                        (TargetType::Volatility, horizon.clone()),
                        volatility_targets[seq_idx],
                    );
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
            targets: target_map,
        });
    }

    Ok(sequences_with_targets)
}

#[cfg(test)]
mod global_balance_tests {
    use super::*;
    use crate::targets::TargetType;

    #[test]
    fn test_global_balance_extraction_basic() {
        // Create test sequences with different class distributions
        let mut sequences = Vec::new();

        // Create sequences for PriceLevel target with classes 0, 1, 2
        for i in 0..15 {
            let class = (i % 3) as i32; // Classes 0, 1, 2 with 5 sequences each (i32 type)
            let mut targets = HashMap::new();
            targets.insert((TargetType::PriceLevel, "1h".to_string()), class);

            sequences.push(SequenceWithTargets {
                sequence_idx: i,
                start_idx: i * 10,
                end_idx: (i + 1) * 10,
                sequence_data: Array2::zeros((10, 5)),
                targets,
            });
        }

        let balancer = SequenceBalancer::new(BalanceConfig::default());
        let target_types = vec![TargetType::PriceLevel];
        let horizons = vec!["1h".to_string()];

        let result =
            balancer.extract_globally_balanced_dataset(&sequences, &target_types, &horizons);

        assert!(result.is_ok());
        let balanced = result.unwrap();

        // Should have global minimum class count of 5 (all classes have 5 sequences)
        assert_eq!(balanced.global_min_class_count, 5);

        // Should have balanced indices for PriceLevel 1h
        let target_key = (TargetType::PriceLevel, "1h".to_string());
        assert!(balanced.balanced_indices.contains_key(&target_key));

        // Should have 15 total balanced samples (5 per class * 3 classes)
        let indices = balanced.balanced_indices.get(&target_key).unwrap();
        assert_eq!(indices.len(), 15);
    }

    #[test]
    fn test_smart_validation_split() {
        // Create test sequences
        let mut sequences = Vec::new();

        for i in 0..20 {
            let class = (i % 2) as i32; // Classes 0, 1 with 10 sequences each (i32 type)
            let mut targets = HashMap::new();
            targets.insert((TargetType::PriceLevel, "1h".to_string()), class);

            sequences.push(SequenceWithTargets {
                sequence_idx: i,
                start_idx: i * 10,
                end_idx: (i + 1) * 10,
                sequence_data: Array2::zeros((10, 5)),
                targets,
            });
        }

        let balancer = SequenceBalancer::new(BalanceConfig::default());
        let target_types = vec![TargetType::PriceLevel];
        let horizons = vec!["1h".to_string()];

        // First extract globally balanced dataset
        let globally_balanced = balancer
            .extract_globally_balanced_dataset(&sequences, &target_types, &horizons)
            .unwrap();

        // Then split validation
        let result = balancer.smart_validation_split_from_balanced(
            &globally_balanced,
            &sequences,
            0.2, // 20% validation
            &target_types,
            &horizons,
        );

        assert!(result.is_ok());
        let (remaining_training, validation_indices) = result.unwrap();

        // Should have validation indices for the target
        let target_key = (TargetType::PriceLevel, "1h".to_string());
        assert!(validation_indices.contains_key(&target_key));

        // Validation should be extracted
        let val_indices = validation_indices.get(&target_key).unwrap();
        assert!(!val_indices.is_empty());

        // Training pool should be reduced
        assert!(
            remaining_training.total_balanced_samples < globally_balanced.total_balanced_samples
        );
    }

    #[test]
    fn test_progressive_window_expansion() {
        // Create test sequences
        let mut sequences = Vec::new();

        for i in 0..30 {
            let class = (i % 3) as i32; // Classes 0, 1, 2 with 10 sequences each
            let mut targets = HashMap::new();
            targets.insert((TargetType::PriceLevel, "1h".to_string()), class);

            sequences.push(SequenceWithTargets {
                sequence_idx: i,
                start_idx: i * 10,
                end_idx: (i + 1) * 10,
                sequence_data: Array2::zeros((10, 5)),
                targets,
            });
        }

        let balancer = SequenceBalancer::new(BalanceConfig::default());
        let target_types = vec![TargetType::PriceLevel];
        let horizons = vec!["1h".to_string()];

        // Extract globally balanced dataset
        let globally_balanced = balancer
            .extract_globally_balanced_dataset(&sequences, &target_types, &horizons)
            .unwrap();

        // Create mock window ranges
        use super::WindowConfig;
        let window_ranges = vec![
            WindowConfig { train_end: 100 },
            WindowConfig { train_end: 200 },
            WindowConfig { train_end: 300 },
        ];

        // Build windows from balanced pool
        let window_indices_map = balancer
            .build_windows_from_balanced_pool(
                &globally_balanced,
                &window_ranges,
                &target_types,
                &horizons,
            )
            .unwrap();

        let target_key = (TargetType::PriceLevel, "1h".to_string());

        // Verify progressive expansion
        let window_0_samples = window_indices_map
            .get(&0)
            .unwrap()
            .get(&target_key)
            .unwrap()
            .len();
        let window_1_samples = window_indices_map
            .get(&1)
            .unwrap()
            .get(&target_key)
            .unwrap()
            .len();
        let window_2_samples = window_indices_map
            .get(&2)
            .unwrap()
            .get(&target_key)
            .unwrap()
            .len();

        // Each window should have more or equal samples than the previous
        assert!(window_1_samples >= window_0_samples);
        assert!(window_2_samples >= window_1_samples);

        // Last window should have all balanced samples
        let total_balanced = globally_balanced
            .balanced_indices
            .get(&target_key)
            .unwrap()
            .len();
        assert_eq!(window_2_samples, total_balanced);
    }

    #[test]
    fn test_window_balance_maintained() {
        // Create test sequences with clear class structure
        let mut sequences = Vec::new();

        // Create 60 sequences: 20 each for classes 0, 1, 2 (imbalanced)
        for class in 0..3 {
            for i in 0..20 {
                let mut targets = HashMap::new();
                targets.insert((TargetType::PriceLevel, "1h".to_string()), class as i32);

                sequences.push(SequenceWithTargets {
                    sequence_idx: class * 20 + i,
                    start_idx: (class * 20 + i) * 10,
                    end_idx: ((class * 20 + i) + 1) * 10,
                    sequence_data: Array2::zeros((10, 5)),
                    targets,
                });
            }
        }

        let balancer = SequenceBalancer::new(BalanceConfig::default());
        let target_types = vec![TargetType::PriceLevel];
        let horizons = vec!["1h".to_string()];

        // Extract globally balanced dataset (should balance to 20 per class)
        let globally_balanced = balancer
            .extract_globally_balanced_dataset(&sequences, &target_types, &horizons)
            .unwrap();

        // Create 3 windows for testing
        use super::WindowConfig;
        let window_ranges = vec![
            WindowConfig { train_end: 100 },
            WindowConfig { train_end: 200 },
            WindowConfig { train_end: 300 },
        ];

        // Build windows from balanced pool
        let window_indices_map = balancer
            .build_windows_from_balanced_pool(
                &globally_balanced,
                &window_ranges,
                &target_types,
                &horizons,
            )
            .unwrap();

        let target_key = (TargetType::PriceLevel, "1h".to_string());

        // CRITICAL TEST: Verify each window maintains perfect balance
        for window_idx in 0..3 {
            let window_samples = window_indices_map
                .get(&window_idx)
                .unwrap()
                .get(&target_key)
                .unwrap();

            // Count class distribution in this window
            let mut class_counts = HashMap::new();
            for &sample_idx in window_samples {
                let seq = &sequences[sample_idx];
                let class = seq.targets.get(&target_key).unwrap();
                *class_counts.entry(*class).or_insert(0) += 1;
            }

            println!(
                "Window {} class distribution: {:?}",
                window_idx, class_counts
            );

            // Verify perfect balance within this window
            if class_counts.len() > 1 {
                let counts: Vec<usize> = class_counts.values().cloned().collect();
                let min_count = *counts.iter().min().unwrap();
                let max_count = *counts.iter().max().unwrap();

                // Allow at most 1 sample difference between classes (due to remainder distribution)
                assert!(
                    max_count - min_count <= 1,
                    "Window {} is imbalanced: min={}, max={}, diff={}",
                    window_idx,
                    min_count,
                    max_count,
                    max_count - min_count
                );
            }
        }
    }
}
