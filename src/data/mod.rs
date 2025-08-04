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
use crate::utils::error::Result;

use std::collections::HashMap;
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

    /// Load and preprocess data for training with walk-forward analysis (default)
    pub async fn prepare_training_data<P: AsRef<Path>>(
        &self,
        data_path: P,
        config: &crate::config::TrainingConfig,
    ) -> Result<Vec<TrainingWindow>> {
        // Load raw data
        let raw_data = self.loader.load_csv(data_path).await?;

        // Validate schema
        CryptoDataSchema::validate(&raw_data)?;

        // Apply feature engineering but NO global normalization
        let processed_data = self
            .preprocessor
            .process_features_only(raw_data, &config.data, Some(&config.features))
            .await?;

        // Create windows with raw data - normalization happens per-sequence
        let windows = self
            .create_walk_forward_windows(processed_data, config)
            .await?;

        log::info!(
            "📊 Walk-forward analysis: {} windows created for progressive training",
            windows.len()
        );

        Ok(windows)
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

    /// Create expanding window validation with proper chronological split
    /// FIXED: Training data uses [0..train_end], validation uses [train_end + gap..train_end + gap + val_size]
    /// Returns (training_data, validation_data)
    fn create_distributed_validation(
        data: &polars::prelude::DataFrame,
        train_end: usize, // FIXED: This is the END of training data (not size)
        val_size: usize,
        gap_steps: usize,
        validation_start_override: Option<usize>, // NEW: Override default validation position
    ) -> Result<(polars::prelude::DataFrame, polars::prelude::DataFrame)> {
        let total_data_size = data.height();

        // Use smart validation start if provided
        let val_start = validation_start_override.unwrap_or(train_end + gap_steps);
        let val_end = val_start + val_size;

        // Validate we have enough data
        if val_end > total_data_size {
            return Err(crate::utils::error::VangaError::DataError(
                format!(
                    "Insufficient data for expanding window: train_end={}, gap={}, val_size={}, total={}, need={}",
                    train_end, gap_steps, val_size, total_data_size, val_end
                )
            ));
        }

        // CORRECT EXPANDING WINDOW LOGIC:
        // Training data: ALL data from start to train_end
        let train_df = data.slice(0, train_end);

        // Validation data: NEXT chunk after gap
        let val_df = data.slice(val_start as i64, val_size);

        log::info!(
            "✅ Smart validation: train=[0..{}] ({} samples), val=[{}..{}] ({} samples), gap_maintained={}",
            train_end,
            train_df.height(),
            val_start,
            val_end,
            val_df.height(),
            val_start >= train_end + gap_steps
        );

        // Validate the split worked correctly
        if train_df.height() != train_end {
            return Err(crate::utils::error::VangaError::DataError(format!(
                "Training data size mismatch: expected {}, got {}",
                train_end,
                train_df.height()
            )));
        }

        if val_df.height() != val_size {
            return Err(crate::utils::error::VangaError::DataError(format!(
                "Validation data size mismatch: expected {}, got {}",
                val_size,
                val_df.height()
            )));
        }

        log::debug!(
            "🔧 Expanding window validation: train_samples={}, val_samples={}, gap={}, chronological_order=preserved",
            train_df.height(),
            val_df.height(),
            gap_steps
        );

        Ok((train_df, val_df))
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
            // 🚀 NEW: Progressive increment validation approach
            // Instead of using fixed avg_increment, calculate increments that maintain min_increment_ratio

            let mut progressive_increments = Vec::new();
            let mut current_train_size = min_train_size;
            let mut total_increment_needed = 0;

            // Calculate progressive increments for each window
            for _window_idx in 1..window_count {
                let min_increment_for_this_window =
                    (current_train_size as f64 * min_increment_ratio) as usize;
                progressive_increments.push(min_increment_for_this_window);
                total_increment_needed += min_increment_for_this_window;
                current_train_size += min_increment_for_this_window;
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
                let (validation_start, is_fresh) =
                    if next_fresh_validation_start + validation_size <= available_for_training {
                        // Use fresh validation data
                        let start = next_fresh_validation_start;
                        next_fresh_validation_start += validation_size + (gap_steps * 2); // Reserve space
                        (Some(start), true)
                    } else {
                        // Use default validation positioning (may reuse, but with proper gap)
                        (None, false)
                    };

                windows.push(WindowConfig {
                    train_end,
                    validation_size,
                    validation_start,
                    is_fresh_validation: is_fresh,
                });

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

                // Calculate average increment for reporting (using progressive increments)
                let avg_progressive_increment = if progressive_increments.is_empty() {
                    avg_increment
                } else {
                    progressive_increments.iter().sum::<usize>() / progressive_increments.len()
                };

                best_config = Some(OptimalWindowConfig {
                    window_count,
                    windows,
                    data_utilization: utilization,
                    avg_increment: avg_progressive_increment,
                });

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
                let (validation_start, is_fresh) =
                    if next_fresh_validation_start + validation_size <= available_for_training {
                        // Use fresh validation data
                        let start = next_fresh_validation_start;
                        next_fresh_validation_start += validation_size + (gap_steps * 2); // Reserve space
                        (Some(start), true)
                    } else {
                        // Use default validation positioning (may reuse, but with proper gap)
                        (None, false)
                    };

                windows.push(WindowConfig {
                    train_end,
                    validation_size,
                    validation_start,
                    is_fresh_validation: is_fresh,
                });

                if i < 2 {
                    train_end += simple_increment;
                }
            }

            OptimalWindowConfig {
                window_count: 3,
                windows,
                data_utilization: 95.0, // Approximate
                avg_increment: simple_increment,
            }
        })
    }

    /// Create walk-forward analysis windows with proper three-way split
    /// Reserves test_split for final evaluation while maximizing training data utilization
    async fn create_walk_forward_windows(
        &self,
        raw_processed_data: polars::prelude::DataFrame, // Has features but NOT normalized
        config: &crate::config::TrainingConfig,
    ) -> Result<Vec<TrainingWindow>> {
        let total_samples = raw_processed_data.height();

        // STEP 1: Reserve test set (never touched during training/validation)
        let test_size = (total_samples as f64 * config.training.test_split) as usize;
        let available_for_training = total_samples - test_size;

        // STEP 2: Calculate validation size from remaining data
        let validation_size =
            (available_for_training as f64 * config.training.validation_split) as usize;

        // 🚀 EFFICIENCY-FOCUSED: Configurable minimum training size (default 40% for faster training)
        let min_train_size =
            (available_for_training as f64 * config.training.min_train_ratio) as usize;

        // 🎯 SINGLE-WINDOW DETECTION: When min_train_ratio leaves no expansion room
        let data_for_expansion = available_for_training.saturating_sub(min_train_size);

        let use_single_window = config.training.min_train_ratio >= 0.8
            || data_for_expansion < (available_for_training / 10);

        if validation_size == 0 {
            return Err(crate::utils::error::VangaError::DataError(
                format!(
                    "Invalid validation_split: results in 0 validation samples from {} available samples",
                    available_for_training
                )
            ));
        }

        if use_single_window {
            log::info!(
                "🎯 SINGLE-WINDOW MODE: min_train_ratio={:.1}% leaves only {} samples for expansion (< 10% threshold)",
                config.training.min_train_ratio * 100.0,
                data_for_expansion
            );
            log::info!(
                "   Using single training window with all {} available samples",
                available_for_training
            );
        }

        let mut windows = Vec::new();

        log::info!(
            "📊 Efficiency-focused three-way split: total={}, test_reserved={} ({:.1}%), available_for_training={}, val_size={} ({:.1}%)",
            total_samples,
            test_size,
            config.training.test_split * 100.0,
            available_for_training,
            validation_size,
            config.training.validation_split * 100.0
        );

        log::info!(
            "🚀 Efficiency settings: min_train_ratio={:.1}% ({} samples), optimized for faster training",
            config.training.min_train_ratio * 100.0,
            min_train_size
        );

        // Parse validation gap
        let gap_steps = if config.training.validation_gap == "0" {
            0
        } else {
            crate::utils::parser::parse_horizon_to_steps(&config.training.validation_gap)
                .unwrap_or_else(|e| {
                    log::warn!(
                        "Invalid validation_gap '{}': {}. Using 0.",
                        config.training.validation_gap,
                        e
                    );
                    0
                })
        };

        log::info!(
            "🔄 Expanding window validation: gap={} steps, validation occurs AFTER training period",
            gap_steps
        );

        // 🚀 EFFICIENCY-FOCUSED WALK-FORWARD ALGORITHM
        let optimal_config = if use_single_window {
            // 🎯 SINGLE-WINDOW MODE: Use all available data in one training window
            log::info!("🎯 Creating single training window with all available data");

            let single_train_size = available_for_training - validation_size - gap_steps;

            if single_train_size < 1000 {
                return Err(crate::utils::error::VangaError::DataError(
                    format!(
                        "Insufficient training data after reserving validation: {} samples < 1000 minimum. \
                        Available: {}, validation: {}, gap: {}, remaining for training: {}",
                        single_train_size, available_for_training, validation_size, gap_steps, single_train_size
                    )
                ));
            }

            OptimalWindowConfig {
                window_count: 1,
                windows: vec![WindowConfig {
                    train_end: single_train_size,
                    validation_size,
                    validation_start: Some(single_train_size + gap_steps),
                    is_fresh_validation: true,
                }],
                data_utilization: 100.0,
                avg_increment: 0, // No increments in single-window mode
            }
        } else {
            // 🚀 MULTI-WINDOW MODE: Calculate optimal window configuration
            Self::calculate_optimal_window_configuration(
                available_for_training,
                validation_size,
                min_train_size,
                gap_steps,
                config.training.min_increment_ratio,
            )
        };

        if use_single_window {
            log::info!(
                "🎯 Single-window result: 1 window, {:.1}% data utilization, training_size={} samples",
                optimal_config.data_utilization,
                optimal_config.windows[0].train_end
            );
        } else {
            log::info!(
                "🚀 Multi-window result: {} windows planned, {:.1}% data utilization, avg_increment={} samples",
                optimal_config.window_count,
                optimal_config.data_utilization,
                optimal_config.avg_increment
            );
        }

        // Create windows using the optimal configuration
        for (window_idx, window_config) in optimal_config.windows.iter().enumerate() {
            let train_end = window_config.train_end;
            let current_validation_size = window_config.validation_size;
            let is_final_window = window_idx == optimal_config.windows.len() - 1;

            // Log smart validation scheduling
            log::info!(
                "📊 Window {}: train=[0..{}], validation={} ({})",
                window_idx,
                train_end,
                if let Some(start) = window_config.validation_start {
                    format!("[{}..{}]", start, start + current_validation_size)
                } else {
                    format!(
                        "[{}..{}]",
                        train_end + gap_steps,
                        train_end + gap_steps + current_validation_size
                    )
                },
                if window_config.is_fresh_validation {
                    "FRESH"
                } else {
                    "REUSED"
                }
            );

            // Create expanding window validation with proper chronological split
            let (train_df, val_df) = Self::create_distributed_validation(
                &raw_processed_data,
                train_end,
                current_validation_size,
                gap_steps,
                window_config.validation_start, // NEW: Pass smart validation start
            )?;

            // Store DataFrame heights before moving them
            let train_df_height = train_df.height();
            let val_df_height = val_df.height();

            // Test data is reserved - only include in final window
            let _test_df = if is_final_window && test_size > 0 {
                // Final window - include test data for final evaluation
                Some(raw_processed_data.slice(available_for_training as i64, test_size))
            } else {
                // Intermediate window - no test data
                None
            };

            // Generate sequences with per-sequence normalization
            let train_sequences = self
                .sequence_generator
                .generate_training_sequences(
                    train_df, // RAW data
                    &config.horizons,
                    &config.model,
                    &config.data,
                )
                .await?;

            let val_sequences = self
                .sequence_generator
                .generate_training_sequences(
                    val_df, // RAW data
                    &config.horizons,
                    &config.model,
                    &config.data,
                )
                .await?;

            // Generate test sequences - empty for intermediate windows, populated for final window
            let test_sequences = if is_final_window && test_size > 0 {
                // Final window - include test data for final evaluation
                self.sequence_generator
                    .generate_training_sequences(
                        raw_processed_data.slice(available_for_training as i64, test_size),
                        &config.horizons,
                        &config.model,
                        &config.data,
                    )
                    .await?
            } else {
                // Intermediate window - create empty test data with same structure
                PreparedData {
                    sequences: ndarray::Array3::zeros((
                        0,
                        train_sequences.sequences.shape()[1],
                        train_sequences.sequences.shape()[2],
                    )),
                    targets: crate::targets::PreparedTargets::new(0),
                    feature_names: train_sequences.feature_names.clone(),
                    normalization_stats: train_sequences.normalization_stats.clone(),
                    metadata: train_sequences.metadata.clone(),
                }
            };

            // Calculate target-specific per-window class weights based on configuration strategy
            let target_class_weights = match config.training.class_weight_strategy {
                ClassWeightStrategy::PerWindow => self
                    .calculate_all_target_class_weights(&train_sequences, config)
                    .unwrap_or_else(|e| {
                        log::warn!(
                            "⚠️ Failed to calculate target-specific class weights: {}",
                            e
                        );
                        HashMap::new()
                    }),
                ClassWeightStrategy::Global => {
                    // Global weights will be calculated once in the LSTM model
                    HashMap::new()
                }
                ClassWeightStrategy::None => {
                    // No class weighting
                    HashMap::new()
                }
                ClassWeightStrategy::Advanced => {
                    // Use advanced imbalance mitigation strategies
                    self.calculate_all_target_class_weights(&train_sequences, config)
                        .unwrap_or_else(|e| {
                            log::warn!("⚠️ Failed to calculate advanced class weights: {}", e);
                            HashMap::new()
                        })
                }
            };

            // Log target class weights summary for this window
            if !target_class_weights.is_empty() {
                log::info!(
                    "🎯 Window {} class weights: {} target-horizon combinations calculated",
                    windows.len() + 1,
                    target_class_weights.len()
                );
                for (key, weights) in &target_class_weights {
                    log::debug!("   {}: {:?}", key, weights);
                }
            } else {
                log::info!(
                    "🎯 Window {}: No class weights calculated (strategy: {:?})",
                    windows.len() + 1,
                    config.training.class_weight_strategy
                );
            }

            // Log expanding window details BEFORE creating the window
            log::info!(
                "📊 ADAPTIVE Window {}: train_data=[0..{}] ({} samples → {} sequences), val_data=[{}..{}] ({} samples → {} sequences), val_size={}",
                window_idx + 1,
                train_end,
                train_df_height,
                train_sequences.sequences.shape()[0],
                train_end + gap_steps,
                train_end + gap_steps + current_validation_size,
                val_df_height,
                val_sequences.sequences.shape()[0],
                current_validation_size
            );

            windows.push(TrainingWindow {
                train_data: train_sequences,
                val_data: val_sequences,
                test_data: test_sequences.clone(),
                window_id: window_idx,
                train_samples: train_end,
                val_samples: current_validation_size,
                test_samples: test_sequences.sequences.shape()[0],
                target_class_weights,
            });
        }

        if windows.is_empty() {
            return Err(crate::utils::error::VangaError::DataError(
                "No valid walk-forward windows could be created".to_string(),
            ));
        }

        // Log comprehensive window expansion summary
        log::info!(
            "📊 EXPANDING Walk-forward windows created: {} windows with per-sequence normalization",
            windows.len()
        );

        // Log expansion pattern for verification
        for (i, window) in windows.iter().enumerate() {
            let expansion_from_first = if i == 0 {
                0
            } else {
                window.train_samples - windows[0].train_samples
            };
            let expansion_from_previous = if i == 0 {
                0
            } else {
                window.train_samples - windows[i - 1].train_samples
            };
            // Calculate percentage increase from previous window
            let percentage_increase = if i == 0 {
                0.0
            } else {
                (expansion_from_previous as f64 / windows[i - 1].train_samples as f64) * 100.0
            };

            log::info!(
                "   Window {}: train_samples={} (+{} from first, +{} from previous, {:.1}% increase), val_samples={}, test_samples={}",
                i + 1,
                window.train_samples,
                expansion_from_first,
                expansion_from_previous,
                percentage_increase,
                window.val_samples,
                window.test_samples
            );
        }

        // Calculate actual data utilization for verification
        let total_data_used = windows
            .last()
            .map(|w| w.train_samples + w.val_samples)
            .unwrap_or(0);
        let actual_utilization = (total_data_used as f64 / available_for_training as f64) * 100.0;
        let data_saved = total_data_used.saturating_sub((available_for_training * 917) / 1000); // vs 91.7%

        log::info!(
            "✅ ADAPTIVE ALGORITHM: {} windows created with {:.1}% data utilization (was 91.7% with fixed increment)",
            optimal_config.window_count,
            optimal_config.data_utilization
        );

        log::info!(
            "📈 Data efficiency: {}/{} samples used ({:.1}% actual), {} samples saved vs old algorithm",
            total_data_used,
            available_for_training,
            actual_utilization,
            data_saved
        );

        Ok(windows)
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
    pub normalization_stats: NormalizationStats,
    pub metadata: DataMetadata,
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
    pub validation_size: usize,
    /// Override default validation position for smart scheduling
    pub validation_start: Option<usize>,
    /// Track whether this window uses fresh validation data
    pub is_fresh_validation: bool,
}

/// Optimal walk-forward window configuration
#[derive(Debug)]
struct OptimalWindowConfig {
    pub window_count: usize,
    pub windows: Vec<WindowConfig>,
    pub data_utilization: f64,
    pub avg_increment: usize,
}

/// Walk-forward training window with proper three-way split
#[derive(Debug)]
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
