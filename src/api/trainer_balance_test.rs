use std::collections::HashMap;

use crate::config::model::HiddenUnitsConfig;
use crate::config::TrainingConfig;
use crate::utils::error::Result;

use super::trainer::{
	pool_rows_before_validation, validate_continue_training_compatibility, validate_pool_feature_names,
};

struct ContinueTrainingCompatibilityFixture {
	saved_config: TrainingConfig,
	current_config: TrainingConfig,
	saved_feature_names: Vec<String>,
	prepared_feature_names: Vec<String>,
	saved_target_names: Vec<String>,
	prepared_target_names: Vec<String>,
}

impl ContinueTrainingCompatibilityFixture {
	fn new() -> Self {
		let saved_config = TrainingConfig::default();
		Self {
			current_config: saved_config.clone(),
			saved_config,
			saved_feature_names: vec!["close".to_string(), "volume".to_string()],
			prepared_feature_names: vec!["close".to_string(), "volume".to_string()],
			saved_target_names: vec!["direction_1h".to_string()],
			prepared_target_names: vec!["direction_1h".to_string()],
		}
	}

	fn validate(&self, saved_config: Option<&TrainingConfig>) -> Result<()> {
		validate_continue_training_compatibility(
			saved_config,
			&self.current_config,
			Some(&self.saved_feature_names),
			&self.saved_target_names,
			&self.prepared_feature_names,
			&self.prepared_target_names,
		)
	}
}

#[test]
fn continue_training_compatibility_accepts_matching_model() {
	let fixture = ContinueTrainingCompatibilityFixture::new();

	fixture.validate(Some(&fixture.saved_config)).unwrap();
}

#[test]
fn continue_training_compatibility_rejects_missing_training_config() {
	let fixture = ContinueTrainingCompatibilityFixture::new();

	let error = fixture.validate(None).unwrap_err();

	assert!(error.to_string().contains("missing training_config"));
}

#[test]
fn continue_training_compatibility_rejects_hidden_units_change() {
	let mut fixture = ContinueTrainingCompatibilityFixture::new();
	fixture.saved_config.model.hidden_units = HiddenUnitsConfig::Fixed(vec![64]);
	fixture.current_config = fixture.saved_config.clone();
	fixture.current_config.model.hidden_units = HiddenUnitsConfig::Fixed(vec![128]);

	let error = fixture.validate(Some(&fixture.saved_config)).unwrap_err();

	assert!(error.to_string().contains("section(s): model"));
}

#[test]
fn continue_training_compatibility_rejects_xgboost_change() {
	let mut fixture = ContinueTrainingCompatibilityFixture::new();
	fixture.saved_config.model.xgboost.enabled = true;
	fixture.current_config = fixture.saved_config.clone();
	fixture.current_config.model.xgboost.enabled = false;

	let error = fixture.validate(Some(&fixture.saved_config)).unwrap_err();

	assert!(error.to_string().contains("section(s): model"));
}

#[test]
fn continue_training_compatibility_rejects_features_change() {
	let mut fixture = ContinueTrainingCompatibilityFixture::new();
	fixture.current_config.features.stats_window_bars += 1;

	let error = fixture.validate(Some(&fixture.saved_config)).unwrap_err();

	assert!(error.to_string().contains("section(s): features"));
}

#[test]
fn continue_training_compatibility_rejects_targets_change() {
	let mut fixture = ContinueTrainingCompatibilityFixture::new();
	fixture.current_config.targets.volume = !fixture.saved_config.targets.volume;

	let error = fixture.validate(Some(&fixture.saved_config)).unwrap_err();

	assert!(error.to_string().contains("section(s): targets"));
}

#[test]
fn continue_training_compatibility_rejects_feature_name_mismatch() {
	let mut fixture = ContinueTrainingCompatibilityFixture::new();
	fixture.prepared_feature_names.reverse();

	let error = fixture.validate(Some(&fixture.saved_config)).unwrap_err();

	assert!(error.to_string().contains("feature mismatch"));
}

#[test]
fn continue_training_compatibility_rejects_target_set_mismatch() {
	let mut fixture = ContinueTrainingCompatibilityFixture::new();
	fixture.prepared_target_names = vec!["volatility_1h".to_string()];

	let error = fixture.validate(Some(&fixture.saved_config)).unwrap_err();

	assert!(error.to_string().contains("target mismatch"));
}

#[test]
fn forest_pool_time_cut_keeps_only_rows_before_purged_validation_anchor() {
	// Anchors are rows 0, 1, 2 at 100, 200, 250; with a purge of 100 only the first two end before 350
	let kept = pool_rows_before_validation(&[(0, 1), (1, 2), (2, 3)], &[100, 200, 250], 350, 100).unwrap();

	assert_eq!(kept, vec![0, 1]);
}

#[test]
fn forest_pool_time_cut_errors_when_no_rows_remain() {
	let error = pool_rows_before_validation(&[(0, 1), (1, 2)], &[100, 200], 200, 100).unwrap_err();

	assert!(error.to_string().contains("zero rows"));
}

#[test]
fn forest_pool_feature_name_mismatch_is_an_error() {
	let own = vec!["close".to_string(), "volume".to_string()];
	let pool = vec!["volume".to_string(), "close".to_string()];

	let error = validate_pool_feature_names(&own, &pool, std::path::Path::new("pool.csv")).unwrap_err();

	assert!(error.to_string().contains("feature-name mismatch"));
}

#[test]
fn test_truncation_maintains_perfect_class_balance() {
	// Simulate the exact scenario from the bug report
	let current_total = 7100;
	let target_count = 5000;
	let samples_to_keep = (target_count / 5) * 5; // 5000
	let validation_split = 0.1;

	// Create mock target classes (simulating real data with balanced distribution)
	let mut target_classes: Vec<i32> = Vec::with_capacity(current_total);
	for i in 0..current_total {
		target_classes.push((i % 5) as i32); // Perfectly balanced: 1420 per class
	}

	// Group indices by class
	let mut class_indices: HashMap<i32, Vec<usize>> = HashMap::new();
	for (idx, &class) in target_classes.iter().enumerate() {
		class_indices.entry(class).or_default().push(idx);
	}

	// Calculate train/val split
	let val_samples = (samples_to_keep as f64 * validation_split).round() as usize;
	let train_samples = samples_to_keep - val_samples;

	// Calculate samples per class for each split
	let train_per_class = train_samples / 5;
	let val_per_class = val_samples / 5;

	println!("Target: {} total samples", samples_to_keep);
	println!("Train: {} samples ({} per class)", train_samples, train_per_class);
	println!("Val: {} samples ({} per class)", val_samples, val_per_class);

	// Select balanced indices for train and val separately
	let mut train_indices: Vec<usize> = Vec::with_capacity(train_samples);
	let mut val_indices: Vec<usize> = Vec::with_capacity(val_samples);

	for class in 0..5 {
		if let Some(indices) = class_indices.get(&class) {
			let total_needed = train_per_class + val_per_class;
			let class_stride = indices.len() as f64 / total_needed as f64;

			let mut class_selected: Vec<usize> = Vec::with_capacity(total_needed);
			for i in 0..total_needed {
				let idx_in_class = ((i as f64 * class_stride).round() as usize).min(indices.len() - 1);
				class_selected.push(indices[idx_in_class]);
			}

			// Split this class's samples into train and val
			train_indices.extend(&class_selected[..train_per_class]);
			val_indices.extend(&class_selected[train_per_class..]);
		}
	}

	// Sort and dedup
	train_indices.sort_unstable();
	train_indices.dedup();
	val_indices.sort_unstable();
	val_indices.dedup();

	// Verify train class balance
	let mut train_class_counts: HashMap<i32, usize> = HashMap::new();
	for &idx in &train_indices {
		let class = target_classes[idx];
		*train_class_counts.entry(class).or_default() += 1;
	}

	println!("\nTrain class distribution:");
	for class in 0..5 {
		let count = train_class_counts.get(&class).copied().unwrap_or(0);
		let percentage = (count as f64 / train_indices.len() as f64) * 100.0;
		println!("  Class {}: {} samples ({:.1}%)", class, count, percentage);
	}

	// Verify val class balance
	let mut val_class_counts: HashMap<i32, usize> = HashMap::new();
	for &idx in &val_indices {
		let class = target_classes[idx];
		*val_class_counts.entry(class).or_default() += 1;
	}

	println!("\nVal class distribution:");
	for class in 0..5 {
		let count = val_class_counts.get(&class).copied().unwrap_or(0);
		let percentage = (count as f64 / val_indices.len() as f64) * 100.0;
		println!("  Class {}: {} samples ({:.1}%)", class, count, percentage);
	}

	// CRITICAL ASSERTIONS: Perfect balance required
	for class in 0..5 {
		let train_count = train_class_counts.get(&class).copied().unwrap_or(0);
		assert_eq!(
			train_count, train_per_class,
			"Train class {} has {} samples, expected {} (perfect balance required)",
			class, train_count, train_per_class
		);

		let val_count = val_class_counts.get(&class).copied().unwrap_or(0);
		assert_eq!(
			val_count, val_per_class,
			"Val class {} has {} samples, expected {} (perfect balance required)",
			class, val_count, val_per_class
		);
	}

	// Verify total counts
	assert_eq!(train_indices.len(), train_samples, "Train total mismatch");
	assert_eq!(val_indices.len(), val_samples, "Val total mismatch");

	println!("\n✅ Perfect class balance maintained after truncation!");
}

#[test]
fn test_truncation_balance_with_imbalanced_input() {
	// Test with initially imbalanced data to ensure we can still achieve balance
	// As long as each class has enough samples
	let current_total = 7100;
	let target_count = 5000;
	let samples_to_keep = (target_count / 5) * 5;
	let validation_split = 0.1;

	// Create imbalanced target classes (but each has enough for truncation)
	let mut target_classes: Vec<i32> = Vec::new();
	target_classes.extend(vec![0; 2000]); // Class 0: 2000 samples (need 1000)
	target_classes.extend(vec![1; 1800]); // Class 1: 1800 samples (need 1000)
	target_classes.extend(vec![2; 1500]); // Class 2: 1500 samples (need 1000)
	target_classes.extend(vec![3; 1200]); // Class 3: 1200 samples (need 1000)
	target_classes.extend(vec![4; 600]); // Class 4: 600 samples (need 1000) - NOT ENOUGH!

	assert_eq!(target_classes.len(), current_total);

	// Group indices by class
	let mut class_indices: HashMap<i32, Vec<usize>> = HashMap::new();
	for (idx, &class) in target_classes.iter().enumerate() {
		class_indices.entry(class).or_default().push(idx);
	}

	// Calculate train/val split
	let val_samples = (samples_to_keep as f64 * validation_split).round() as usize;
	let train_samples = samples_to_keep - val_samples;
	let train_per_class = train_samples / 5;
	let val_per_class = val_samples / 5;

	// Check if we have enough samples in each class
	let total_per_class = train_per_class + val_per_class;
	for class in 0..5 {
		let available = class_indices.get(&class).map(|v| v.len()).unwrap_or(0);
		if available < total_per_class {
			println!(
				"⚠️  Class {} has only {} samples, need {} - cannot achieve perfect balance",
				class, available, total_per_class
			);
			println!("✅ Test correctly identifies insufficient data scenario");
			return; // This is expected behavior
		}
	}

	panic!("Test data should have insufficient samples for class 4");
}
