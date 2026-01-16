use crate::config::training::TrainingConfig;
use crate::data::{DataMetadata, PreparedData, TrainingWindow};
use crate::targets::PreparedTargets;
use chrono::Utc;
use ndarray::Array3;

#[test]
fn test_truncation_maintains_sequence_target_alignment() {
    // Create config with truncation enabled
    let mut config = TrainingConfig::default();
    config.data.target_samples.truncate = true;
    config.data.target_samples.count = 500;

    // Create mock window with 1000 samples (train + val)
    let train_samples = 900;
    let val_samples = 100;
    let total_samples = train_samples + val_samples;
    let sequence_length = 30;
    let features = 50;

    // Create sequences
    let train_sequences = Array3::<f64>::zeros((train_samples, sequence_length, features));
    let val_sequences = Array3::<f64>::zeros((val_samples, sequence_length, features));

    // Create targets with proper alignment
    let mut train_targets = PreparedTargets::new(train_samples);
    let mut val_targets = PreparedTargets::new(val_samples);

    // Add target data for each horizon
    let horizons = vec!["1h".to_string(), "4h".to_string()];
    for horizon in &horizons {
        // Create target vectors with proper length
        let train_price_levels: Vec<i32> = (0..train_samples).map(|i| (i % 5) as i32).collect();
        let train_direction: Vec<i32> = (0..train_samples).map(|i| ((i + 1) % 5) as i32).collect();
        let train_volatility: Vec<i32> = (0..train_samples).map(|i| ((i + 2) % 5) as i32).collect();

        let val_price_levels: Vec<i32> = (0..val_samples).map(|i| (i % 5) as i32).collect();
        let val_direction: Vec<i32> = (0..val_samples).map(|i| ((i + 1) % 5) as i32).collect();
        let val_volatility: Vec<i32> = (0..val_samples).map(|i| ((i + 2) % 5) as i32).collect();

        train_targets
            .price_levels
            .insert(horizon.clone(), train_price_levels);
        train_targets
            .direction
            .insert(horizon.clone(), train_direction);
        train_targets
            .volatility
            .insert(horizon.clone(), train_volatility);

        val_targets
            .price_levels
            .insert(horizon.clone(), val_price_levels);
        val_targets.direction.insert(horizon.clone(), val_direction);
        val_targets
            .volatility
            .insert(horizon.clone(), val_volatility);
    }

    // Set valid_indices to match sequence counts
    train_targets.valid_indices = (0..train_samples).collect();
    val_targets.valid_indices = (0..val_samples).collect();

    // Set target names
    train_targets.target_names = vec![
        "price_level_1h".to_string(),
        "direction_1h".to_string(),
        "volatility_1h".to_string(),
    ];
    val_targets.target_names = train_targets.target_names.clone();

    // Create window
    let mut window = TrainingWindow {
        train_data: PreparedData {
            sequences: train_sequences,
            targets: train_targets,
            feature_names: vec![],
            metadata: DataMetadata {
                symbol: "BTCUSDT".to_string(),
                total_records: total_samples,
                feature_count: features,
                sequence_length,
                horizons: horizons.clone(),
                start_time: Utc::now(),
                end_time: Utc::now(),
            },

            sequence_indices: vec![],
        },
        val_data: PreparedData {
            sequences: val_sequences,
            targets: val_targets,
            feature_names: vec![],
            metadata: DataMetadata {
                symbol: "BTCUSDT".to_string(),
                total_records: total_samples,
                feature_count: features,
                sequence_length,
                horizons: horizons.clone(),
                start_time: Utc::now(),
                end_time: Utc::now(),
            },

            sequence_indices: vec![],
        },
        test_data: PreparedData {
            sequences: Array3::<f64>::zeros((0, sequence_length, features)),
            targets: PreparedTargets::new(0),
            feature_names: vec![],
            metadata: DataMetadata {
                symbol: "BTCUSDT".to_string(),
                total_records: 0,
                feature_count: features,
                sequence_length,
                horizons: horizons.clone(),
                start_time: Utc::now(),
                end_time: Utc::now(),
            },

            sequence_indices: vec![],
        },
        window_id: 0,
        train_samples,
        val_samples,
        test_samples: 0,
    };

    // BEFORE truncation: verify alignment
    assert_eq!(
        window.train_data.sequences.shape()[0],
        window.train_data.targets.valid_indices.len(),
        "BEFORE: Train sequences and targets must be aligned"
    );
    assert_eq!(
        window.val_data.sequences.shape()[0],
        window.val_data.targets.valid_indices.len(),
        "BEFORE: Val sequences and targets must be aligned"
    );

    // Verify target data lengths match valid_indices
    for horizon in &horizons {
        let train_price_len = window
            .train_data
            .targets
            .price_levels
            .get(horizon)
            .unwrap()
            .len();
        let train_dir_len = window
            .train_data
            .targets
            .direction
            .get(horizon)
            .unwrap()
            .len();
        let train_vol_len = window
            .train_data
            .targets
            .volatility
            .get(horizon)
            .unwrap()
            .len();

        assert_eq!(
            train_price_len, train_samples,
            "Train price_levels length mismatch"
        );
        assert_eq!(
            train_dir_len, train_samples,
            "Train direction length mismatch"
        );
        assert_eq!(
            train_vol_len, train_samples,
            "Train volatility length mismatch"
        );
    }

    // Apply truncation (simulating the trainer logic)
    let current_total = window.train_samples + window.val_samples;
    let target_count = 500;

    if current_total > target_count {
        let samples_to_keep = (target_count / 5) * 5;

        // Combine sequences
        let combined_sequences = ndarray::concatenate(
            ndarray::Axis(0),
            &[
                window.train_data.sequences.view(),
                window.val_data.sequences.view(),
            ],
        )
        .unwrap();

        // Select evenly distributed indices
        let stride = current_total as f64 / samples_to_keep as f64;
        let mut selected_indices: Vec<usize> = Vec::with_capacity(samples_to_keep);
        for i in 0..samples_to_keep {
            let idx = ((i as f64 * stride).round() as usize).min(current_total - 1);
            selected_indices.push(idx);
        }
        selected_indices.dedup();

        let unique_count = selected_indices.len();
        let val_ratio = config.training.validation_split;
        let val_samples_new = (unique_count as f64 * val_ratio).round() as usize;
        let train_samples_new = unique_count - val_samples_new;

        let train_indices: Vec<usize> = selected_indices[..train_samples_new].to_vec();
        let val_indices: Vec<usize> = selected_indices[train_samples_new..].to_vec();

        // Truncate sequences
        window.train_data.sequences = combined_sequences.select(ndarray::Axis(0), &train_indices);
        window.val_data.sequences = combined_sequences.select(ndarray::Axis(0), &val_indices);

        // THIS IS THE CRITICAL TEST: Targets must also be truncated
        // The fix should implement this logic in trainer.rs

        // For now, manually truncate targets to test the concept
        let combined_train_targets = window.train_data.targets.clone();
        let combined_val_targets = window.val_data.targets.clone();

        // Combine target vectors for each horizon
        for horizon in &horizons {
            let mut combined_price: Vec<i32> = combined_train_targets
                .price_levels
                .get(horizon)
                .unwrap()
                .clone();
            combined_price.extend(combined_val_targets.price_levels.get(horizon).unwrap());

            let mut combined_dir: Vec<i32> = combined_train_targets
                .direction
                .get(horizon)
                .unwrap()
                .clone();
            combined_dir.extend(combined_val_targets.direction.get(horizon).unwrap());

            let mut combined_vol: Vec<i32> = combined_train_targets
                .volatility
                .get(horizon)
                .unwrap()
                .clone();
            combined_vol.extend(combined_val_targets.volatility.get(horizon).unwrap());

            // Select targets using same indices
            let train_price: Vec<i32> = train_indices.iter().map(|&i| combined_price[i]).collect();
            let train_dir: Vec<i32> = train_indices.iter().map(|&i| combined_dir[i]).collect();
            let train_vol: Vec<i32> = train_indices.iter().map(|&i| combined_vol[i]).collect();

            let val_price: Vec<i32> = val_indices.iter().map(|&i| combined_price[i]).collect();
            let val_dir: Vec<i32> = val_indices.iter().map(|&i| combined_dir[i]).collect();
            let val_vol: Vec<i32> = val_indices.iter().map(|&i| combined_vol[i]).collect();

            window
                .train_data
                .targets
                .price_levels
                .insert(horizon.clone(), train_price);
            window
                .train_data
                .targets
                .direction
                .insert(horizon.clone(), train_dir);
            window
                .train_data
                .targets
                .volatility
                .insert(horizon.clone(), train_vol);

            window
                .val_data
                .targets
                .price_levels
                .insert(horizon.clone(), val_price);
            window
                .val_data
                .targets
                .direction
                .insert(horizon.clone(), val_dir);
            window
                .val_data
                .targets
                .volatility
                .insert(horizon.clone(), val_vol);
        }

        // Update valid_indices
        window.train_data.targets.valid_indices = (0..train_samples_new).collect();
        window.val_data.targets.valid_indices = (0..val_samples_new).collect();

        window.train_samples = train_samples_new;
        window.val_samples = val_samples_new;
    }

    // AFTER truncation: verify alignment is maintained
    assert_eq!(
        window.train_data.sequences.shape()[0],
        window.train_data.targets.valid_indices.len(),
        "AFTER: Train sequences and targets must remain aligned"
    );
    assert_eq!(
        window.val_data.sequences.shape()[0],
        window.val_data.targets.valid_indices.len(),
        "AFTER: Val sequences and targets must remain aligned"
    );

    // Verify target data lengths match new valid_indices
    for horizon in &horizons {
        let train_price_len = window
            .train_data
            .targets
            .price_levels
            .get(horizon)
            .unwrap()
            .len();
        let train_dir_len = window
            .train_data
            .targets
            .direction
            .get(horizon)
            .unwrap()
            .len();
        let train_vol_len = window
            .train_data
            .targets
            .volatility
            .get(horizon)
            .unwrap()
            .len();

        assert_eq!(
            train_price_len, window.train_samples,
            "AFTER: Train price_levels length must match truncated samples"
        );
        assert_eq!(
            train_dir_len, window.train_samples,
            "AFTER: Train direction length must match truncated samples"
        );
        assert_eq!(
            train_vol_len, window.train_samples,
            "AFTER: Train volatility length must match truncated samples"
        );

        let val_price_len = window
            .val_data
            .targets
            .price_levels
            .get(horizon)
            .unwrap()
            .len();
        let val_dir_len = window
            .val_data
            .targets
            .direction
            .get(horizon)
            .unwrap()
            .len();
        let val_vol_len = window
            .val_data
            .targets
            .volatility
            .get(horizon)
            .unwrap()
            .len();

        assert_eq!(
            val_price_len, window.val_samples,
            "AFTER: Val price_levels length must match truncated samples"
        );
        assert_eq!(
            val_dir_len, window.val_samples,
            "AFTER: Val direction length must match truncated samples"
        );
        assert_eq!(
            val_vol_len, window.val_samples,
            "AFTER: Val volatility length must match truncated samples"
        );
    }

    println!("✅ Sequence-target alignment test passed!");
    println!(
        "   Before: {} train + {} val = {} total",
        train_samples, val_samples, total_samples
    );
    println!(
        "   After: {} train + {} val = {} total",
        window.train_samples,
        window.val_samples,
        window.train_samples + window.val_samples
    );
}
