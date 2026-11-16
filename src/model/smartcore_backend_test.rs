use crate::config::model::{XGBoostConfig, NUM_CLASSES};
use crate::model::smartcore_backend::SmartCoreRegressor;
use crate::utils::error::Result;

use candle_core::{Device, Tensor};
use tempfile::tempdir;

const SAMPLES_PER_CLASS: usize = 20;
const FEATURE_DIM: usize = 4;
const TREE_COUNT: usize = 15;
const TREE_DEPTH: usize = 4;
const PROBABILITY_TOLERANCE: f32 = 1e-5;
const CLASS_SPACING: f32 = 10.0;
const NOISE_SCALE: f32 = 0.1;
const NOISE_PERIOD: usize = 7;

fn small_config() -> XGBoostConfig {
	XGBoostConfig {
		feature_dim: FEATURE_DIM,
		n_estimators: TREE_COUNT,
		max_depth: TREE_DEPTH,
		save_feature_importance: false,
		..XGBoostConfig::default()
	}
}

/// Class `c` sits around `c * CLASS_SPACING` in every feature, so a forest separates the classes exactly.
fn separable_data(device: &Device) -> Result<(Tensor, Tensor, Vec<usize>)> {
	let mut features = Vec::new();
	let mut targets = Vec::new();
	let mut classes = Vec::new();
	for class in 0..NUM_CLASSES {
		for sample in 0..SAMPLES_PER_CLASS {
			for feature in 0..FEATURE_DIM {
				let noise = ((sample + feature) % NOISE_PERIOD) as f32 * NOISE_SCALE;
				features.push(class as f32 * CLASS_SPACING + noise);
			}
			let mut one_hot = vec![0.0f32; NUM_CLASSES];
			one_hot[class] = 1.0;
			targets.extend(one_hot);
			classes.push(class);
		}
	}
	let rows = classes.len();
	Ok((
		Tensor::from_vec(features, (rows, FEATURE_DIM), device)?,
		Tensor::from_vec(targets, (rows, NUM_CLASSES), device)?,
		classes,
	))
}

fn trained_regressor() -> Result<(SmartCoreRegressor, Tensor, Vec<usize>)> {
	let device = Device::Cpu;
	let (features, targets, classes) = separable_data(&device)?;
	let mut regressor = SmartCoreRegressor::new(small_config(), device);
	regressor.train(&features, &targets, None, None)?;
	Ok((regressor, features, classes))
}

fn predict_rows(regressor: &SmartCoreRegressor, features: &Tensor) -> Result<Vec<Vec<f32>>> {
	Ok(regressor.predict(features)?.to_vec2::<f32>()?)
}

#[test]
fn forest_predictions_are_positive_vote_shares_that_sum_to_one() -> Result<()> {
	let (regressor, features, classes) = trained_regressor()?;
	let rows = predict_rows(&regressor, &features)?;

	assert_eq!(rows.len(), classes.len());
	for (row, &class) in rows.iter().zip(&classes) {
		assert_eq!(row.len(), NUM_CLASSES);
		assert!((row.iter().sum::<f32>() - 1.0).abs() < PROBABILITY_TOLERANCE);
		assert!(row.iter().all(|&p| p > 0.0));
		// Separable data: the forest's hard class is the true class, and it holds the most votes
		let arg_max = row
			.iter()
			.enumerate()
			.max_by(|a, b| a.1.total_cmp(b.1))
			.map(|(index, _)| index)
			.expect("row is not empty");
		assert_eq!(arg_max, class);
	}
	Ok(())
}

#[test]
fn predictions_are_identical_after_save_and_load() -> Result<()> {
	let (regressor, features, _) = trained_regressor()?;
	let before = predict_rows(&regressor, &features)?;

	let dir = tempdir().expect("temp dir");
	let path = dir.path().join("forest").to_string_lossy().to_string();
	regressor.save_model(&path)?;
	let loaded = SmartCoreRegressor::load_model(&path, Device::Cpu)?;

	assert_eq!(predict_rows(&loaded, &features)?, before);
	Ok(())
}

#[test]
fn fitted_blend_weight_survives_save_and_load() -> Result<()> {
	const FITTED_WEIGHT: f32 = 0.37;
	let (mut regressor, _, _) = trained_regressor()?;
	regressor.set_lstm_blend_weight(FITTED_WEIGHT);

	let dir = tempdir().expect("temp dir");
	let path = dir.path().join("forest").to_string_lossy().to_string();
	regressor.save_model(&path)?;
	let loaded = SmartCoreRegressor::load_model(&path, Device::Cpu)?;

	assert_eq!(loaded.get_config().lstm_blend_weight, FITTED_WEIGHT);
	Ok(())
}

#[test]
fn vote_power_flattens_the_shares_and_survives_save_and_load() -> Result<()> {
	const FLATTENING_POWER: f32 = 0.5;
	let (mut regressor, features, _) = trained_regressor()?;
	let counted = predict_rows(&regressor, &features)?;
	regressor.set_forest_vote_power(FLATTENING_POWER);
	let flattened = predict_rows(&regressor, &features)?;

	for (counted_row, flattened_row) in counted.iter().zip(&flattened) {
		assert!((flattened_row.iter().sum::<f32>() - 1.0).abs() < PROBABILITY_TOLERANCE);
		let counted_top = counted_row.iter().cloned().fold(f32::MIN, f32::max);
		let flattened_top = flattened_row.iter().cloned().fold(f32::MIN, f32::max);
		assert!(flattened_top < counted_top);
	}

	let dir = tempdir().expect("temp dir");
	let path = dir.path().join("forest").to_string_lossy().to_string();
	regressor.save_model(&path)?;
	let loaded = SmartCoreRegressor::load_model(&path, Device::Cpu)?;

	assert_eq!(loaded.get_config().forest_vote_power, FLATTENING_POWER);
	assert_eq!(predict_rows(&loaded, &features)?, flattened);
	Ok(())
}
