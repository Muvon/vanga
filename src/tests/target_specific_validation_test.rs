//! Test to verify target-specific validation is properly used in training

use crate::config::model::ModelConfig;
use crate::config::training::{
    BatchSizeConfig, ClassWeightStrategy, DataConfig, DeviceConfig, EarlyStoppingConfig,
    EpochConfig, OptimizerType, TrainingConfig, TrainingParams,
};
use crate::config::FeatureConfig;
use crate::data::balance::{BalanceConfig, SequenceBalancer};
use crate::model::multi_target::{MultiTargetLSTMModel, TrainingContext};
use crate::targets::{PreparedTargets, TargetType};
use ndarray::{Array2, Array3};
use std::collections::HashMap;

#[tokio::test]
async fn test_target_specific_validation_in_training() {
    // Create test data
    let num_sequences = 100;
    let sequence_length = 30;
    let num_features = 5;
    let num_targets = 3; // PriceLevel, Direction, Volatility

    // Create sequences
    let sequences = Array3::zeros((num_sequences, sequence_length, num_features));

    // Create targets with imbalanced distribution
    let mut targets = PreparedTargets::new(num_sequences);
    targets.target_names = vec![
        "price_level_1h".to_string(),
        "direction_1h".to_string(),
        "volatility_1h".to_string(),
    ];

    // Create imbalanced class distributions for each target
    // PriceLevel: mostly class 2 (neutral)
    targets.price_levels.insert(
        "1h".to_string(),
        (0..num_sequences)
            .map(|i| {
                if i < 10 {
                    0
                }
                // 10% class 0
                else if i < 20 {
                    1
                }
                // 10% class 1
                else if i < 70 {
                    2
                }
                // 50% class 2 (imbalanced)
                else if i < 85 {
                    3
                }
                // 15% class 3
                else {
                    4
                } // 15% class 4
            })
            .collect(),
    );

    // Direction: mostly class 1 (down)
    targets.directions.insert(
        "1h".to_string(),
        (0..num_sequences)
            .map(|i| {
                if i < 5 {
                    0
                }
                // 5% class 0
                else if i < 60 {
                    1
                }
                // 55% class 1 (imbalanced)
                else if i < 75 {
                    2
                }
                // 15% class 2
                else if i < 90 {
                    3
                }
                // 15% class 3
                else {
                    4
                } // 10% class 4
            })
            .collect(),
    );

    // Volatility: mostly class 3 (high)
    targets.volatility.insert(
        "1h".to_string(),
        (0..num_sequences)
            .map(|i| {
                if i < 15 {
                    0
                }
                // 15% class 0
                else if i < 25 {
                    1
                }
                // 10% class 1
                else if i < 35 {
                    2
                }
                // 10% class 2
                else if i < 80 {
                    3
                }
                // 45% class 3 (imbalanced)
                else {
                    4
                } // 20% class 4
            })
            .collect(),
    );

    targets.valid_indices = (0..num_sequences).collect();

    // Convert to sequences with targets
    let sequence_indices: Vec<(usize, usize)> = (0..num_sequences)
        .map(|i| (i * 10, i * 10 + sequence_length))
        .collect();

    let all_sequences = crate::data::balance::create_sequences_with_targets(
        sequences.clone(),
        &targets,
        sequence_indices.clone(),
    )
    .await
    .expect("Failed to create sequences with targets");

    // Create balancer and select target-specific validation
    let balance_config = BalanceConfig::default();
    let balancer = SequenceBalancer::new(balance_config);

    let target_types = vec![
        TargetType::PriceLevel,
        TargetType::Direction,
        TargetType::Volatility,
    ];
    let horizons = vec!["1h".to_string()];
    let validation_ratio = 0.2;

    let target_validation_indices = balancer
        .select_target_specific_validation(
            &all_sequences,
            validation_ratio,
            &target_types,
            &horizons,
        )
        .expect("Failed to select validation");

    // Verify each target has balanced validation
    for target_type in &target_types {
        let key = (*target_type, "1h".to_string());
        let val_indices = target_validation_indices
            .get(&key)
            .expect("Missing validation indices for target");

        println!(
            "Target {:?}: {} validation sequences",
            target_type,
            val_indices.len()
        );

        // Check class distribution in validation
        let mut class_counts = HashMap::new();
        for &idx in val_indices {
            if let Some(&class) = all_sequences[idx].targets.get(&key) {
                *class_counts.entry(class).or_insert(0) += 1;
            }
        }

        println!("  Class distribution: {:?}", class_counts);

        // Verify balanced distribution (all classes should have similar counts)
        let counts: Vec<usize> = class_counts.values().copied().collect();
        let min_count = *counts.iter().min().unwrap_or(&0);
        let max_count = *counts.iter().max().unwrap_or(&0);

        // Allow some variance but ensure reasonable balance
        assert!(
            max_count <= min_count * 2,
            "Validation for {:?} is not balanced: min={}, max={}",
            target_type,
            min_count,
            max_count
        );
    }

    // Create training configuration
    let config = TrainingConfig {
        symbol: "TEST".to_string(),
        data_path: std::path::PathBuf::from("test.csv"),
        fresh_training: true,
        continue_training: false,
        horizons: vec!["1h".to_string()],
        features: FeatureConfig::default(),
        model: ModelConfig::default(),
        training: TrainingParams {
            device: DeviceConfig::CPU,
            epochs: EpochConfig::Fixed(1), // Just one epoch for testing
            batch_size: BatchSizeConfig::Fixed(32),
            learning_rate: 0.001,
            optimizer: OptimizerType::Adam {
                beta1: 0.9,
                beta2: 0.999,
                eps: 1e-8,
                weight_decay: None,
                amsgrad: false,
            },
            warmup_epochs: 0,
            learning_schedule: None,
            validation_split: 0.2,
            validation_gap: "0".to_string(),
            test_split: 0.0,
            window_decay: 1.0,
            early_stopping: EarlyStoppingConfig {
                patience: 10,
                min_delta: 0.0001,
            },
            gradient_clip: Some(1.0),
            print_every: 1,
            class_weight_strategy: ClassWeightStrategy::None,
            min_train_ratio: 0.5,
            min_increment_ratio: 0.1,
            seed: 42, // Add seed for reproducibility
        },
        data: DataConfig::default(),
        optimization: Default::default(),
    };

    // Create multi-target model
    let target_names = vec![
        "price_level_1h".to_string(),
        "direction_1h".to_string(),
        "volatility_1h".to_string(),
    ];
    let mut model =
        MultiTargetLSTMModel::new(&config.model, num_features, target_names, horizons.clone())
            .expect("Failed to create model");

    // Convert targets to Array2
    let targets_array = Array2::from_shape_fn((num_sequences, num_targets), |(i, j)| match j {
        0 => targets.price_levels.get("1h").unwrap()[i] as f64,
        1 => targets.directions.get("1h").unwrap()[i] as f64,
        2 => targets.volatility.get("1h").unwrap()[i] as f64,
        _ => unreachable!(),
    });

    // Train with target-specific validation
    let result = model
        .train(
            TrainingContext::Standard {
                sequences: &sequences,
                targets: &targets_array,
                val_sequences: Some(&sequences),
                val_targets: Some(&targets_array),
                target_class_weights: None,
                target_validation_indices: Some(&target_validation_indices),
            },
            &config,
        )
        .await;

    assert!(result.is_ok(), "Training failed: {:?}", result.err());

    println!("✅ Target-specific validation training completed successfully!");
}
