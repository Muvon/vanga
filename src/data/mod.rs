pub mod balance;
#[cfg(test)]
mod balance_critical_test;
#[cfg(test)]
mod balance_test;
pub mod loader;
pub mod preprocessor;
pub mod schema;
pub mod sequence;
pub mod structures;
pub mod target_converter;

use serde::{Deserialize, Serialize};

pub use loader::DataLoader;
pub use preprocessor::DataPreprocessor;
pub use schema::{CryptoDataSchema, DataValidationError};
pub use sequence::SequenceGenerator;
pub use target_converter::TargetConverter;

use crate::config::training::ClassWeightStrategy;
use crate::targets::PreparedTargets;
use crate::targets::TargetType;
use crate::utils::error::{Result, VangaError};

use std::collections::{HashMap, HashSet};
use std::path::Path;

/// Main data pipeline orchestrator
pub struct DataPipeline {
    loader: DataLoader,
    preprocessor: DataPreprocessor,
    sequence_generator: SequenceGenerator,
}

impl Default for DataPipeline {
    fn default() -> Self {
        Self::new()
    }
}

impl DataPipeline {
    pub fn new() -> Self {
        Self {
            loader: DataLoader::new(),
            preprocessor: DataPreprocessor::new(),
            sequence_generator: SequenceGenerator::default(), // Uses no overlap by default
        }
    }

    /// Load and preprocess data for training with walk-forward analysis and balanced sampling
    pub async fn prepare_training_data<P: AsRef<Path>>(
        &self,
        data_path: P,
        config: &crate::config::TrainingConfig,
    ) -> Result<TargetSpecificWindows> {
        // Load raw data
        let raw_data = self.loader.load_csv(data_path).await?;

        // Validate schema
        CryptoDataSchema::validate(&raw_data)?;

        // Apply feature engineering but NO global normalization
        let processed_data = self
            .preprocessor
            .process_features_only(raw_data, &config.data, Some(&config.features))
            .await?;

        // Use target-specific balanced approach
        let target_windows = self
            .create_target_specific_balanced_windows(processed_data, config)
            .await?;

        log::info!(
            "🎯 Target-specific balanced analysis: {} targets with independent balancing",
            target_windows.windows_by_target.len()
        );

        Ok(target_windows)
    }

    /// Calculate class weights for a specific training window
    /// Reuses the same logic as the LSTM model's class weight calculation
    fn calculate_window_class_weights(
        &self,
        train_data: &PreparedData,
        target_type: &TargetType,
        horizon: &str,
        _config: &crate::config::TrainingConfig,
    ) -> Result<Option<Vec<f32>>> {
        // Get the target data for the specific target type and horizon
        let targets = match target_type {
            TargetType::PriceLevel => train_data.targets.price_levels.get(horizon),
            TargetType::Direction => train_data.targets.directions.get(horizon),
            TargetType::Volatility => train_data.targets.volatility.get(horizon),
        };

        let targets = match targets {
            Some(t) => t,
            None => {
                log::warn!(
                    "⚠️ No target data available for {:?} horizon {}, skipping class weights",
                    target_type,
                    horizon
                );
                return Ok(None);
            }
        };

        if targets.is_empty() {
            log::warn!(
                "⚠️ Empty target data for {:?} horizon {}, skipping class weights",
                target_type,
                horizon
            );
            return Ok(None);
        }

        // Get the correct number of classes from model configuration (same logic as LSTM model)
        let num_classes = match target_type {
            TargetType::PriceLevel => crate::config::model::NUM_CLASSES, // Always 5-class system
            TargetType::Direction => crate::config::model::NUM_CLASSES,  // Always 5-class system
            TargetType::Volatility => crate::config::model::NUM_CLASSES, // Always 5-class system
        };

        // Count class frequencies
        let mut class_counts: HashMap<i32, usize> = HashMap::new();
        let mut total_samples = 0;

        for &target in targets.iter() {
            let class_id = target;
            *class_counts.entry(class_id).or_insert(0) += 1;
            total_samples += 1;
        }

        // Debug: Log detailed class distribution for this window
        log::debug!(
            "🔍 Window class distribution for {:?} horizon {}: {} total samples",
            target_type,
            horizon,
            total_samples
        );
        for (class_id, count) in &class_counts {
            let percentage = (*count as f64 / total_samples as f64) * 100.0;
            log::debug!(
                "   Class {}: {} samples ({:.2}%)",
                class_id,
                count,
                percentage
            );
        }

        if num_classes < 2 {
            log::warn!(
                "⚠️ Only {} classes configured for {:?} horizon {}, skipping class weights",
                num_classes,
                target_type,
                horizon
            );
            return Ok(None);
        }

        // Use advanced class weighting (same as price levels) for all target types
        use crate::targets::imbalance_mitigation::{
            AdvancedClassWeighter, ClassDistributionAnalysis, ImbalanceMitigationConfig,
        };

        let mitigation_config = ImbalanceMitigationConfig::default();
        let analysis = ClassDistributionAnalysis::analyze(targets, num_classes, &mitigation_config);
        let weights = AdvancedClassWeighter::calculate_weights(
            &analysis,
            &mitigation_config.weighting_strategy,
        )?;

        log::debug!(
            "🎯 Window class weights for {:?} horizon {}: {:?} (from {} samples, {} classes configured)",
            target_type,
            horizon,
            weights,
            total_samples,
            num_classes
        );

        Ok(Some(weights))
    }

    /// Calculate class weights for all target types and horizons
    fn calculate_all_target_class_weights(
        &self,
        train_data: &PreparedData,
        config: &crate::config::TrainingConfig,
    ) -> Result<HashMap<String, Vec<f32>>> {
        let mut target_weights = HashMap::new();

        // Define all target types to calculate weights for
        let target_types = [
            TargetType::PriceLevel,
            TargetType::Direction,
            TargetType::Volatility,
        ];

        for target_type in &target_types {
            for horizon in &config.horizons {
                // Calculate weights for this specific target type and horizon
                if let Ok(Some(weights)) =
                    self.calculate_window_class_weights(train_data, target_type, horizon, config)
                {
                    let key = format!("{:?}_{}", target_type, horizon);
                    target_weights.insert(key, weights);

                    log::debug!(
                        "📊 Calculated class weights for {:?} horizon {}: {} classes",
                        target_type,
                        horizon,
                        target_weights
                            .get(&format!("{:?}_{}", target_type, horizon))
                            .unwrap()
                            .len()
                    );
                }
            }
        }

        log::info!(
            "🎯 Calculated class weights for {} target-horizon combinations",
            target_weights.len()
        );

        Ok(target_weights)
    }

    /// Calculate optimal walk-forward window configuration for maximum data utilization
    /// Balances between data efficiency and validation quality
    fn calculate_optimal_window_configuration(
        available_for_training: usize,
        base_validation_size: usize,
        min_train_size: usize,
        gap_steps: usize,
        min_increment_ratio: f64, // NEW: Minimum increment ratio for sufficient new data
    ) -> OptimalWindowConfig {
        let min_validation_size = std::cmp::max(base_validation_size / 2, 1000);
        let max_validation_size = base_validation_size * 2;
        let data_for_expansion = available_for_training - min_train_size;

        let mut best_config = None;
        let mut best_score = 0.0;

        // 🚀 EFFICIENCY-FOCUSED WINDOW ALGORITHM
        // Try different window counts with efficiency considerations
        let max_reasonable_windows = std::cmp::min(6, available_for_training / 1000); // Cap based on data size
        let window_range = 2..=max_reasonable_windows;

        log::info!(
            "🧠 Efficiency-focused algorithm: testing {}-{} windows (capped at {} for efficiency)",
            window_range.start(),
            window_range.end(),
            max_reasonable_windows
        );

        for window_count in window_range {
            // 🚀 FIXED: Progressive increment calculation based on PREVIOUS window size
            // Each increment should be min_increment_ratio of the IMMEDIATE PREVIOUS window

            let mut progressive_increments = Vec::new();
            let mut previous_window_size = min_train_size; // Start with first window size
            let mut total_increment_needed = 0;

            // Calculate progressive increments for each subsequent window
            for window_idx in 1..window_count {
                // ✅ CORRECT: Calculate increment based on PREVIOUS window size
                let min_increment_for_this_window =
                    (previous_window_size as f64 * min_increment_ratio) as usize;

                progressive_increments.push(min_increment_for_this_window);
                total_increment_needed += min_increment_for_this_window;

                // Update previous_window_size for next iteration
                previous_window_size += min_increment_for_this_window;

                log::debug!(
                    "📈 Window {} increment: {} samples ({:.1}% of previous window size {})",
                    window_idx + 1, // Window number (1-indexed)
                    min_increment_for_this_window,
                    min_increment_ratio * 100.0,
                    previous_window_size - min_increment_for_this_window // Previous window size
                );
            }

            // Check if we have enough data for progressive increments
            if total_increment_needed > data_for_expansion {
                log::debug!(
                    "⚠️  Skipping {} windows: progressive increments need {} > available {} samples",
                    window_count, total_increment_needed, data_for_expansion
                );
                continue;
            }

            // Also check the old validation for backward compatibility
            let avg_increment = data_for_expansion / window_count;
            if avg_increment < available_for_training / 20 {
                continue;
            }

            let mut windows = Vec::new();
            let mut train_end = min_train_size;
            let mut total_used = min_train_size;
            let mut next_fresh_validation_start = min_train_size + gap_steps;

            for i in 0..window_count {
                // Calculate remaining data after this window's training
                let remaining_after_train = available_for_training - train_end - gap_steps;

                if remaining_after_train == 0 {
                    break;
                }

                // For final window, use all remaining data for validation
                let validation_size = if i == window_count - 1 {
                    remaining_after_train
                } else {
                    // Distribute remaining data across remaining windows
                    let remaining_windows = window_count - i;
                    let avg_val_per_remaining = remaining_after_train / remaining_windows;
                    std::cmp::min(
                        std::cmp::max(avg_val_per_remaining, min_validation_size),
                        max_validation_size,
                    )
                };

                // Check if we have enough data
                if train_end + gap_steps + validation_size > available_for_training {
                    break;
                }

                // 🧠 SMART VALIDATION SCHEDULING
                if next_fresh_validation_start + validation_size <= available_for_training {
                    // Use fresh validation data
                    next_fresh_validation_start += validation_size + (gap_steps * 2);
                    // Reserve space
                }

                windows.push(WindowConfig { train_end });

                total_used = train_end + gap_steps + validation_size;

                // 🚀 NEW: Use progressive increments instead of fixed avg_increment
                if i < window_count - 1 {
                    if i < progressive_increments.len() {
                        let progressive_increment = progressive_increments[i];
                        train_end += progressive_increment;

                        log::debug!(
                            "📈 Window {}: train_end={}, increment=+{} ({:.1}% of previous window)",
                            i + 2, // Next window number
                            train_end,
                            progressive_increment,
                            (progressive_increment as f64
                                / (train_end - progressive_increment) as f64)
                                * 100.0
                        );
                    } else {
                        // Fallback to avg_increment for safety (shouldn't happen with new logic)
                        train_end += avg_increment;
                    }
                }
            }

            // Only consider complete configurations
            if windows.len() != window_count {
                continue;
            }

            let utilization = (total_used as f64 / available_for_training as f64) * 100.0;

            // 🚀 EFFICIENCY-FOCUSED SCORING FUNCTION
            // Balance data utilization with training efficiency

            // Base quality score with stronger diminishing returns
            let window_quality_score = if window_count <= 3 {
                window_count as f64 // Linear for 2-3 windows
            } else if window_count <= 5 {
                3.0 + (window_count as f64 - 3.0) * 0.7 // Moderate returns for 4-5 windows
            } else {
                4.4 + (window_count as f64 - 5.0) * 0.3 // Strong diminishing returns for 6+ windows
            };

            // Efficiency bonus: favor 4-5 windows (sweet spot)
            let efficiency_bonus = match window_count {
                4 | 5 => 0.3, // Sweet spot bonus
                3 | 6 => 0.1, // Slight bonus for reasonable choices
                _ => 0.0,     // No bonus for extreme choices
            };

            // Training time penalty for excessive windows
            let time_penalty = if window_count > 5 {
                (window_count as f64 - 5.0) * 0.2 // Penalty for > 5 windows
            } else {
                0.0
            };

            // Data utilization bonus (encourage high utilization)
            let utilization_bonus = if utilization > 95.0 {
                0.2
            } else if utilization > 90.0 {
                0.1
            } else {
                0.0
            };

            // Final efficiency-focused score
            let score =
                (utilization * window_quality_score / 100.0) + efficiency_bonus + utilization_bonus
                    - time_penalty;

            log::debug!(
                "   {} windows: util={:.1}%, quality={:.2}, efficiency_bonus={:.2}, time_penalty={:.2}, final_score={:.3}",
                window_count, utilization, window_quality_score, efficiency_bonus, time_penalty, score
            );

            if score > best_score {
                best_score = score;

                // Progressive increments calculated and used above

                best_config = Some(OptimalWindowConfig { windows });

                log::debug!(
                    "   🏆 NEW BEST: {} windows, score={:.3}, progressive_increments={:?}",
                    window_count,
                    score,
                    progressive_increments
                );
            }
        }

        // Fallback to simple 3-window configuration if no optimal found
        best_config.unwrap_or_else(|| {
            let simple_increment = data_for_expansion / 3;
            let mut windows = Vec::new();
            let mut train_end = min_train_size;
            let mut next_fresh_validation_start = min_train_size + gap_steps;

            for i in 0..3 {
                let validation_size = if i == 2 {
                    available_for_training - train_end - gap_steps
                } else {
                    base_validation_size
                };

                // 🧠 SMART VALIDATION SCHEDULING (Fallback)
                if next_fresh_validation_start + validation_size <= available_for_training {
                    // Use fresh validation data
                    next_fresh_validation_start += validation_size + (gap_steps * 2);
                    // Reserve space
                }

                windows.push(WindowConfig { train_end });

                if i < 2 {
                    train_end += simple_increment;
                }
            }

            OptimalWindowConfig {
                windows: vec![WindowConfig {
                    train_end: available_for_training,
                }],
            }
        })
    }

    /// Create target-specific walk-forward analysis windows with independent balancing per target
    /// Create target-specific balanced windows with 100% balanced training and validation
    ///
    /// CRITICAL ARCHITECTURE:
    /// 1. Each target (PriceLevel, Direction, Volatility) gets independent balanced datasets
    /// 2. Each target's balanced dataset is split using smart_validation_split_from_balanced()
    /// 3. Both training and validation maintain perfect class balance for each target
    /// 4. Progressive windows use portions of balanced training data (still balanced)
    /// 5. Result: Each target trains on optimal balanced data without waste
    async fn create_target_specific_balanced_windows(
        &self,
        raw_processed_data: polars::prelude::DataFrame,
        config: &crate::config::TrainingConfig,
    ) -> Result<TargetSpecificWindows> {
        let total_samples = raw_processed_data.height();

        // STEP 1: Use ALL data for sequence generation (test extraction happens later in balanced pipeline)
        // CRITICAL FIX: Don't waste data with early test reservation - balanced pipeline handles test split properly
        let available_for_training = total_samples; // Use ALL data
        let test_size = 0; // No early reservation

        log::info!(
            "📊 Data split: total={}, early_test_reserved={} (FIXED: no early waste), available_for_training={}",
            total_samples,
            test_size,
            available_for_training
        );

        log::info!("🔧 Test data will be extracted later using balanced pipeline (no data waste)");

        // STEP 2: Generate ALL sequences from ALL available data
        log::info!("🔄 Generating all possible sequences with overlap from ALL data...");

        let training_df = raw_processed_data.clone(); // Use ALL data, not sliced
        let all_sequences_data = self
            .sequence_generator
            .generate_training_sequences(
                training_df,
                &config.horizons,
                &config.model,
                &config.data,
                &config.features,
            )
            .await?;

        // STEP 3: Convert to SequenceWithTargets format
        let all_sequences = crate::data::balance::create_sequences_with_targets(
            all_sequences_data.sequences.clone(),
            &all_sequences_data.targets,
            all_sequences_data.sequence_indices.clone(),
        )
        .await?;

        log::info!(
            "✅ Generated {} total sequences from {} available samples",
            all_sequences.len(),
            available_for_training
        );

        // STEP 4: Create sequence balancer
        let balance_config = crate::data::balance::BalanceConfig {
            max_overlap: config.data.sequence_overlap,
            prefer_non_overlapping: config.data.sequence_overlap < 0.3,
            min_sequences_per_class: 10,
        };

        let balancer = crate::data::balance::SequenceBalancer::new(balance_config);

        // STEP 5: Extract target-specific balanced datasets
        // CRITICAL: Each target gets its own optimal balance (no global minimum constraint)
        let target_types = vec![
            crate::targets::TargetType::PriceLevel,
            crate::targets::TargetType::Direction,
            crate::targets::TargetType::Volatility,
        ];

        log::info!("🎯 TARGET-SPECIFIC BALANCE EXTRACTION: Each target balanced independently");
        let target_balanced_datasets = balancer.extract_target_specific_balanced_datasets(
            &all_sequences,
            &target_types,
            &config.horizons,
        )?;

        // Log per-target balance summary
        for ((target_type, horizon), dataset) in &target_balanced_datasets {
            log::info!(
                "   {:?} {}: {} balanced sequences ({} per class × {} classes)",
                target_type,
                horizon,
                dataset.total_balanced_samples,
                dataset.global_min_class_count,
                dataset
                    .class_distribution
                    .get(&(*target_type, horizon.clone()))
                    .map(|d| d.len())
                    .unwrap_or(0)
            );
        }

        // STEP 6: Calculate window configuration
        let first_target_validation_size = target_balanced_datasets
            .values()
            .next()
            .map(|dataset| {
                (dataset.total_balanced_samples as f64 * config.training.validation_split) as usize
            })
            .unwrap_or(0);

        let window_config = self.calculate_window_configuration_for_balanced(
            available_for_training,
            first_target_validation_size,
            config,
        )?;

        // STEP 7: Create target-specific windows with 100% balanced train/val splits
        let mut target_specific_windows: HashMap<
            (crate::targets::TargetType, String),
            Vec<TrainingWindow>,
        > = HashMap::new();
        let validation_ratio = config.training.validation_split;

        log::info!("🏗️ CREATING TARGET-SPECIFIC BALANCED WINDOWS:");

        // Process each target independently with balanced train/val splits
        // CRITICAL: Sort keys for deterministic processing order (avoid random HashMap iteration)
        let mut sorted_targets: Vec<_> = target_balanced_datasets.keys().collect();
        sorted_targets.sort_by(|a, b| {
            // Sort by target type first, then by horizon
            match a.0.cmp(&b.0) {
                std::cmp::Ordering::Equal => a.1.cmp(&b.1),
                other => other,
            }
        });

        for (target_type, horizon) in sorted_targets {
            let target_dataset = target_balanced_datasets
                .get(&(*target_type, horizon.clone()))
                .unwrap();
            let target_key = (*target_type, horizon.clone());

            log::info!(
                "🎯 Creating windows for {:?} {} with {} balanced sequences",
                target_type,
                horizon,
                target_dataset.total_balanced_samples
            );

            let mut target_windows = Vec::new();

            // Get balanced indices for this target (used for validation in smart split)
            if target_dataset.balanced_indices.contains_key(&target_key) {
                // Create windows for this specific target
                for (window_idx, window_range) in window_config.windows.iter().enumerate() {
                    log::info!(
                        "📊 {:?} {} Window {}/{}: range [0..{}]",
                        target_type,
                        horizon,
                        window_idx + 1,
                        window_config.windows.len(),
                        window_range.train_end
                    );

                    // VALIDATION SPLIT: Mixed chronological - can be from anywhere in time series
                    // Uses balanced extraction while maintaining class distribution
                    let (balanced_training_dataset, target_validation_indices) = balancer
                        .smart_validation_split_from_balanced(
                            target_dataset,
                            &all_sequences,
                            validation_ratio,
                            &[*target_type],
                            &[horizon.clone()],
                            false, // Validation: mixed chronological
                        )?;

                    // Extract training indices for this specific target from the balanced dataset
                    let train_indices = balanced_training_dataset
                        .balanced_indices
                        .get(&target_key)
                        .cloned()
                        .unwrap_or_default();

                    // Extract validation indices for this specific target
                    let val_indices = target_validation_indices
                        .get(&target_key)
                        .cloned()
                        .unwrap_or_default();

                    log::debug!(
                        "   🎯 {:?} {} balanced split: {} train (balanced), {} val (balanced)",
                        target_type,
                        horizon,
                        train_indices.len(),
                        val_indices.len()
                    );

                    // CRITICAL: Progressive windows create CUMULATIVE data (Window1: 2880, Window2: 3744, etc.)
                    // This is CORRECT - continuation training uses expanded data while preserving weights
                    let samples_for_window = if window_config.windows.len() == 1 {
                        train_indices.len()
                    } else {
                        let window_progress =
                            (window_idx + 1) as f64 / window_config.windows.len() as f64;
                        let min_samples = train_indices.len() / window_config.windows.len();
                        let progressive_samples =
                            (train_indices.len() as f64 * window_progress) as usize;
                        std::cmp::max(min_samples, progressive_samples)
                    };

                    // CRITICAL FIX: Take BALANCED subset for progressive windows
                    // Maintain perfect balance while taking progressive amounts
                    let window_train = if samples_for_window >= train_indices.len() {
                        // Use all data if requesting more than available
                        train_indices.clone()
                    } else {
                        // Take balanced subset: samples_for_window / 5 classes = samples per class
                        let samples_per_class = samples_for_window / 5;
                        let mut balanced_subset = Vec::new();

                        // Group indices by class and take equal amounts from each
                        let mut class_indices: std::collections::HashMap<i32, Vec<usize>> =
                            std::collections::HashMap::new();
                        for &idx in &train_indices {
                            if let Some(seq) = all_sequences.get(idx) {
                                if let Some(&target_value) = seq.targets.get(&target_key) {
                                    class_indices.entry(target_value).or_default().push(idx);
                                }
                            }
                        }

                        // Take samples_per_class from each class to maintain balance
                        for class_value in 0..5 {
                            if let Some(class_idx_list) = class_indices.get(&class_value) {
                                let take_count = samples_per_class.min(class_idx_list.len());
                                balanced_subset.extend(&class_idx_list[..take_count]);
                            }
                        }

                        log::debug!(
                            "🎯 Progressive Window {}: Taking {} samples ({} per class × 5) from {} total balanced",
                            window_idx + 1, balanced_subset.len(), samples_per_class, train_indices.len()
                        );

                        balanced_subset
                    };

                    // Create target-specific indices map
                    let mut train_indices_map = HashMap::new();
                    train_indices_map.insert(target_key.clone(), window_train);

                    let mut val_indices_map = HashMap::new();
                    val_indices_map.insert(target_key.clone(), val_indices.clone());

                    // Create window data using target-specific indices
                    let train_data = self.create_data_from_indices(
                        &all_sequences_data,
                        &train_indices_map,
                        &all_sequences,
                    )?;

                    let val_data = self.create_data_from_indices(
                        &all_sequences_data,
                        &val_indices_map,
                        &all_sequences,
                    )?;

                    // TEST SPLIT: End chronological - from most recent data for realistic evaluation
                    // Uses same balanced extraction but takes from end of time series
                    let test_data = if config.training.test_split > 0.0 {
                        // Extract balanced test split from the remaining training data after validation
                        let (_final_training_dataset, target_test_indices) = balancer
                            .smart_validation_split_from_balanced(
                                &balanced_training_dataset,
                                &all_sequences,
                                config.training.test_split,
                                &[*target_type],
                                &[horizon.clone()],
                                true, // Test: end chronological (most recent data)
                            )?;

                        // Extract test indices for this specific target
                        let test_indices = target_test_indices
                            .get(&target_key)
                            .cloned()
                            .unwrap_or_default();

                        if !test_indices.is_empty() {
                            // Create test indices map
                            let mut test_indices_map = HashMap::new();
                            test_indices_map.insert(target_key.clone(), test_indices.clone());

                            // Create test data using target-specific indices
                            let test_data = self.create_data_from_indices(
                                &all_sequences_data,
                                &test_indices_map,
                                &all_sequences,
                            )?;

                            log::info!(
                                "🧪 {:?} {} balanced test split: {} test samples (balanced)",
                                target_type,
                                horizon,
                                test_indices.len()
                            );

                            test_data
                        } else {
                            // No test data available - create empty
                            let mut empty_targets = crate::targets::PreparedTargets::new(0);
                            empty_targets.valid_indices = Vec::new();
                            PreparedData {
                                sequences: ndarray::Array3::zeros((
                                    0,
                                    train_data.sequences.shape()[1],
                                    train_data.sequences.shape()[2],
                                )),
                                targets: empty_targets,
                                feature_names: train_data.feature_names.clone(),
                                metadata: train_data.metadata.clone(),
                                sequence_indices: Vec::new(),
                            }
                        }
                    } else {
                        // No test split configured - create empty test data
                        let mut empty_targets = crate::targets::PreparedTargets::new(0);
                        empty_targets.valid_indices = Vec::new();
                        PreparedData {
                            sequences: ndarray::Array3::zeros((
                                0,
                                train_data.sequences.shape()[1],
                                train_data.sequences.shape()[2],
                            )),
                            targets: empty_targets,
                            feature_names: train_data.feature_names.clone(),
                            metadata: train_data.metadata.clone(),
                            sequence_indices: Vec::new(),
                        }
                    };

                    // Calculate class weights if needed
                    let target_class_weights = match config.training.class_weight_strategy {
                        ClassWeightStrategy::PerWindow => {
                            self.calculate_all_target_class_weights(&train_data, config)?
                        }
                        _ => HashMap::new(),
                    };

                    let val_samples = val_data.sequences.shape()[0];
                    let train_sequences = train_data.sequences.shape()[0];
                    let num_targets = train_data.targets.target_names.len();

                    let window = TrainingWindow {
                        train_data,
                        val_data,
                        test_data: test_data.clone(),
                        window_id: window_idx,
                        train_samples: train_sequences, // ✅ FIX: Use actual training sequence count
                        val_samples,
                        test_samples: test_data.sequences.shape()[0],
                        target_class_weights,
                    };

                    target_windows.push(window);

                    // Log target-specific balanced window summary
                    log::info!(
                        "   → {:?} {} W{}: train {} seq (balanced), val {} seq (balanced), targets: {}",
                        target_type, horizon, window_idx + 1,
                        train_sequences,
                        val_samples,
                        num_targets
                    );
                }
            } else {
                log::error!(
                    "❌ CRITICAL: No balanced indices found for {:?} {} in target_dataset.balanced_indices",
                    target_type, horizon
                );
                log::debug!(
                    "Available keys in balanced_indices: {:?}",
                    target_dataset.balanced_indices.keys().collect::<Vec<_>>()
                );
                return Err(VangaError::DataError(format!(
                    "No balanced indices found for target {:?} horizon {}. This indicates a balance extraction failure.",
                    target_type, horizon
                )));
            }

            // Only insert if we have windows (should always have windows after the check above)
            if !target_windows.is_empty() {
                target_specific_windows.insert(target_key, target_windows);
            } else {
                log::error!(
                    "❌ CRITICAL: Empty target_windows for {:?} {} - this should not happen",
                    target_type,
                    horizon
                );
            }
        }

        log::info!("✅ Created target-specific balanced windows:");

        // Log per-target window summary
        for ((target_type, horizon), windows) in &target_specific_windows {
            if let Some(dataset) = target_balanced_datasets.get(&(*target_type, horizon.clone())) {
                log::info!(
                    "   {:?} {}: {} windows, {} balanced sequences total ({} per class × {} classes)",
                    target_type,
                    horizon,
                    windows.len(),
                    dataset.total_balanced_samples,
                    dataset.global_min_class_count,
                    dataset.class_distribution.get(&(*target_type, horizon.clone()))
                        .map(|d| d.len()).unwrap_or(0)
                );
            }
        }

        // Extract shared metadata from first window
        let feature_names = target_specific_windows
            .values()
            .next()
            .and_then(|windows| windows.first())
            .map(|window| window.train_data.feature_names.clone())
            .unwrap_or_default();

        Ok(TargetSpecificWindows {
            windows_by_target: target_specific_windows,
            feature_names,
        })
    }

    /// Calculate window configuration for balanced approach
    fn calculate_window_configuration_for_balanced(
        &self,
        available_samples: usize,
        validation_size: usize,
        config: &crate::config::TrainingConfig,
    ) -> Result<OptimalWindowConfig> {
        // Similar to existing logic but adapted for balanced approach
        let min_train_size = (available_samples as f64 * config.training.min_train_ratio) as usize;
        let data_for_expansion = available_samples.saturating_sub(min_train_size);

        let use_single_window =
            config.training.min_train_ratio >= 0.8 || data_for_expansion < (available_samples / 10);

        if use_single_window {
            Ok(OptimalWindowConfig {
                windows: vec![WindowConfig {
                    train_end: available_samples,
                }],
            })
        } else {
            // Use existing optimal window calculation
            Ok(Self::calculate_optimal_window_configuration(
                available_samples,
                validation_size,
                min_train_size,
                0, // No gap needed with balanced approach
                config.training.min_increment_ratio,
            ))
        }
    }

    /// Create PreparedData from selected sequence indices with PROPER target alignment
    fn create_data_from_indices(
        &self,
        all_data: &PreparedData,
        indices_by_target: &HashMap<(crate::targets::TargetType, String), Vec<usize>>,
        all_sequences: &[crate::data::balance::SequenceWithTargets],
    ) -> Result<PreparedData> {
        // Get unique indices across all targets
        let mut unique_indices: HashSet<usize> = HashSet::new();
        for indices in indices_by_target.values() {
            unique_indices.extend(indices);
        }

        let mut sorted_indices: Vec<usize> = unique_indices.into_iter().collect();
        sorted_indices.sort();

        // Validate indices are within bounds
        let max_available_idx = all_data.sequences.shape()[0];
        let invalid_indices: Vec<usize> = sorted_indices
            .iter()
            .filter(|&&idx| idx >= max_available_idx)
            .copied()
            .collect();

        if !invalid_indices.is_empty() {
            log::error!("❌ Invalid indices found: {:?}", invalid_indices);
            log::error!("   Max available index: {}", max_available_idx - 1);
            return Err(VangaError::DataError(format!(
                "Invalid sequence indices: {:?} (max available: {})",
                invalid_indices,
                max_available_idx - 1
            )));
        }

        // Extract sequences
        let num_sequences = sorted_indices.len();
        let sequence_length = all_data.sequences.shape()[1];
        let num_features = all_data.sequences.shape()[2];

        log::debug!(
            "   • Creating array with shape: ({}, {}, {})",
            num_sequences,
            sequence_length,
            num_features
        );

        let mut sequences = ndarray::Array3::zeros((num_sequences, sequence_length, num_features));
        let mut sequence_indices = Vec::new();

        for (new_idx, &orig_idx) in sorted_indices.iter().enumerate() {
            sequences
                .slice_mut(ndarray::s![new_idx, .., ..])
                .assign(&all_data.sequences.slice(ndarray::s![orig_idx, .., ..]));
            sequence_indices.push(all_data.sequence_indices[orig_idx]);
        }

        // Create targets using EMBEDDED targets from SequenceWithTargets (CRITICAL FIX!)
        let mut targets = crate::targets::PreparedTargets::new(num_sequences);

        // CRITICAL FIX: Create target-specific PreparedTargets with ONLY current target data
        // Determine which target types are present in indices_by_target
        let target_types_present: std::collections::HashSet<crate::targets::TargetType> =
            indices_by_target
                .keys()
                .map(|(target_type, _)| *target_type)
                .collect();

        log::debug!(
            "🎯 Creating target-specific PreparedTargets for: {:?}",
            target_types_present
        );

        // Only initialize and populate target arrays for the SPECIFIC target types present
        for (target_type, horizon) in indices_by_target.keys() {
            let target_name = match target_type {
                crate::targets::TargetType::PriceLevel => format!("price_level_{}", horizon),
                crate::targets::TargetType::Direction => format!("direction_{}", horizon),
                crate::targets::TargetType::Volatility => format!("volatility_{}", horizon),
            };
            targets.target_names.push(target_name);

            match target_type {
                crate::targets::TargetType::PriceLevel => {
                    targets
                        .price_levels
                        .insert(horizon.clone(), vec![-1; num_sequences]);
                }
                crate::targets::TargetType::Direction => {
                    targets
                        .directions
                        .insert(horizon.clone(), vec![-1; num_sequences]);
                }
                crate::targets::TargetType::Volatility => {
                    targets
                        .volatility
                        .insert(horizon.clone(), vec![-1; num_sequences]);
                }
            }
        }

        // Extract targets from SequenceWithTargets (TARGET-SPECIFIC FILTERING!)
        for (new_idx, &orig_idx) in sorted_indices.iter().enumerate() {
            if orig_idx < all_sequences.len() {
                let seq_with_targets = &all_sequences[orig_idx];

                // Copy ONLY the target types that are present in indices_by_target
                for ((target_type, horizon), &target_value) in &seq_with_targets.targets {
                    // CRITICAL FIX: Only copy targets that are in indices_by_target
                    if indices_by_target.contains_key(&(*target_type, horizon.clone())) {
                        match target_type {
                            crate::targets::TargetType::PriceLevel => {
                                if let Some(targets_vec) = targets.price_levels.get_mut(horizon) {
                                    targets_vec[new_idx] = target_value;
                                }
                            }
                            crate::targets::TargetType::Direction => {
                                if let Some(targets_vec) = targets.directions.get_mut(horizon) {
                                    targets_vec[new_idx] = target_value;
                                }
                            }
                            crate::targets::TargetType::Volatility => {
                                if let Some(targets_vec) = targets.volatility.get_mut(horizon) {
                                    targets_vec[new_idx] = target_value;
                                }
                            }
                        }
                    }
                }
            }
        }

        // CRITICAL FIX: Populate valid_indices with all sequence indices
        // All sequences in balanced selection are valid by definition
        targets.valid_indices = (0..num_sequences).collect();

        log::debug!(
            "Balanced data: {} sequences, {} targets/horizon",
            num_sequences,
            targets.target_names.len()
        );

        Ok(PreparedData {
            sequences,
            targets,
            feature_names: all_data.feature_names.clone(),
            metadata: all_data.metadata.clone(),
            sequence_indices,
        })
    }

    /// Load and preprocess data for prediction
    pub async fn prepare_prediction_data<P: AsRef<Path>>(
        &self,
        data_path: P,
        config: &crate::config::PredictionConfig,
    ) -> Result<PreparedPredictionData> {
        // Load raw data
        let raw_data = self.loader.load_csv(data_path).await?;

        // Validate schema
        CryptoDataSchema::validate(&raw_data)?;

        // Load model to get training config
        let model_path = crate::utils::model_path::get_model_path(&config.symbols[0]);
        let model = crate::model::multi_target::MultiTargetLSTMModel::load(&model_path)?;

        // Use stored training config for consistent preprocessing
        let processed_data = if let Some(training_config) = model.get_training_config() {
            log::info!("Using stored training config for consistent preprocessing");

            // Apply EXACT same preprocessing as training (feature engineering + remove_nan_rows)
            let df = self
                .preprocessor
                .process_features_only(
                    raw_data,
                    &training_config.data,
                    Some(&training_config.features),
                )
                .await?;

            log::info!(
                "✅ Applied same preprocessing as training: {} rows, {} columns",
                df.height(),
                df.width()
            );
            log::info!("✅ Per-sequence normalization will be applied during sequence generation");

            df
        } else {
            // Fallback for old models without stored training config
            log::warn!("No training config found in model - using basic preprocessing (may cause feature mismatch)");
            self.preprocessor
                .process_for_prediction(raw_data, &config.symbols[0], None)
                .await?
        };

        // Generate prediction sequences using model config from training
        let model_config = if let Some(training_config) = model.get_training_config() {
            &training_config.model
        } else {
            // Fallback for old models
            &crate::config::ModelConfig::default()
        };

        let sequences = self
            .sequence_generator
            .generate_prediction_sequences(processed_data, &config.symbols[0], model_config)
            .await?;

        Ok(sequences)
    }

    /// Load and preprocess data for multi-symbol cross-asset prediction
    pub async fn prepare_cross_asset_prediction_data(
        &self,
        symbol_paths: &std::collections::HashMap<String, std::path::PathBuf>,
        _config: &crate::config::PredictionConfig,
        features_config: &crate::config::FeatureConfig,
    ) -> Result<std::collections::HashMap<String, PreparedPredictionData>> {
        log::info!(
            "Preparing cross-asset prediction data for {} symbols",
            symbol_paths.len()
        );

        // Load raw data for all symbols
        let mut symbol_data = std::collections::HashMap::new();
        for (symbol, path) in symbol_paths {
            let raw_data = self.loader.load_csv(path).await?;
            CryptoDataSchema::validate(&raw_data)?;
            symbol_data.insert(symbol.clone(), raw_data);
        }

        // Apply cross-asset preprocessing
        let processed_symbol_data = self
            .preprocessor
            .process_for_cross_asset_prediction(symbol_data, features_config)
            .await?;

        // Generate prediction sequences for each symbol
        let mut prepared_data = std::collections::HashMap::new();
        let default_model_config = crate::config::ModelConfig::default();

        for (symbol, processed_df) in processed_symbol_data {
            let sequences = self
                .sequence_generator
                .generate_prediction_sequences(processed_df, &symbol, &default_model_config)
                .await?;
            prepared_data.insert(symbol, sequences);
        }

        Ok(prepared_data)
    }
}

/// Prepared training data with sequences and targets
#[derive(Debug, Clone)]
pub struct PreparedData {
    pub sequences: ndarray::Array3<f64>, // [batch, sequence, features]
    pub targets: PreparedTargets,
    pub feature_names: Vec<String>,
    pub metadata: DataMetadata,
    /// Start and end indices for each sequence in the original data
    pub sequence_indices: Vec<(usize, usize)>,
}

/// Prepared prediction data
#[derive(Debug)]
pub struct PreparedPredictionData {
    pub sequences: ndarray::Array3<f64>, // [batch, sequence, features]
    pub feature_names: Vec<String>,
    pub metadata: DataMetadata,
    /// OHLC data for the sequence used in prediction (for ATR calculation)
    pub sequence_ohlc: Option<Vec<crate::data::structures::MarketDataRow>>,
}

/// Normalization statistics for features
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct NormalizationStats {
    pub means: Vec<f64>,
    pub stds: Vec<f64>,
    pub mins: Vec<f64>,
    pub maxs: Vec<f64>,
    pub medians: Vec<f64>,
    pub q25: Vec<f64>,
    pub q75: Vec<f64>,
}

/// Data metadata
#[derive(Debug, Clone)]
pub struct DataMetadata {
    pub symbol: String,
    pub start_time: chrono::DateTime<chrono::Utc>,
    pub end_time: chrono::DateTime<chrono::Utc>,
    pub total_records: usize,
    pub feature_count: usize,
    pub sequence_length: usize,
    pub horizons: Vec<String>,
}

/// Configuration for a single walk-forward window
#[derive(Debug, Clone)]
struct WindowConfig {
    pub train_end: usize,
}

/// Optimal walk-forward window configuration
#[derive(Debug)]
struct OptimalWindowConfig {
    pub windows: Vec<WindowConfig>,
}

/// Walk-forward training window with proper three-way split and sequence tracking
#[derive(Debug, Clone)]
pub struct TrainingWindow {
    pub train_data: PreparedData,
    pub val_data: PreparedData,
    /// Test data - empty for intermediate windows, populated for final evaluation
    pub test_data: PreparedData,
    pub window_id: usize,
    pub train_samples: usize,
    pub val_samples: usize,
    pub test_samples: usize,
    /// Target-specific class weights for balanced training
    /// Key format: "{target_type}_{horizon}" (e.g., "PriceLevel_1h", "Direction_4h")
    pub target_class_weights: HashMap<String, Vec<f32>>,
}

/// Container for target-specific balanced windows
#[derive(Debug)]
pub struct TargetSpecificWindows {
    /// Windows organized by target: (TargetType, Horizon) -> Vec<TrainingWindow>
    pub windows_by_target: HashMap<(crate::targets::TargetType, String), Vec<TrainingWindow>>,
    /// Shared metadata
    pub feature_names: Vec<String>,
}
