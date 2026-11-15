use super::SequenceGenerator;
use crate::config::model::SequenceLengthConfig;
use crate::config::{training::DataConfig, training::TrainingConfig, FeatureConfig};
use polars::prelude::*;

#[cfg(test)]
mod tests {
	use super::*;

	fn make_df(n: usize) -> DataFrame {
		let mut df = DataFrame::new(vec![]).unwrap();
		let ts: Vec<String> = (0..n).map(|i| format!("2024-01-01T00:{:02}:00Z", i)).collect();
		df.with_column(Series::new("timestamp".into(), ts)).unwrap();
		df.with_column(Series::new("open".into(), vec![42000.0; n])).unwrap();
		df.with_column(Series::new("high".into(), vec![42500.0; n])).unwrap();
		df.with_column(Series::new("low".into(), vec![41800.0; n])).unwrap();
		df.with_column(Series::new("close".into(), vec![42300.0; n])).unwrap();
		df.with_column(Series::new("volume".into(), vec![1000.0; n])).unwrap();
		df
	}

	#[tokio::test]
	async fn generate_training_sequences_fixed_ok() {
		let df = make_df(32);
		let horizons = vec!["1m".to_string()];
		let mut training_config = TrainingConfig {
			horizons: horizons.clone(),
			model: crate::config::model::ModelConfig {
				sequence_length: SequenceLengthConfig::Fixed(10),
				..Default::default()
			},
			..Default::default()
		};
		training_config.targets.price_level = false;
		training_config.targets.direction = false;
		training_config.targets.volatility = false;
		training_config.targets.sentiment = false;
		training_config.targets.volume = false;
		let gen = SequenceGenerator::default(); // Use default (no overlap)
		let data_config = DataConfig::default();
		let mut feature_config = FeatureConfig::default();
		feature_config.technical_indicators.enabled = false;
		feature_config.volatility_features.enabled = false;
		feature_config.engineering.lag_features.enabled = false;
		feature_config.engineering.rolling_features.enabled = false;
		let mut calibrated = crate::targets::calibration::CalibratedParameters::default();
		calibrated.stop_levels.insert(
			"1m".to_string(),
			crate::targets::calibration::StopLevelParams {
				bandwidth: 1.0,
				percentiles: [0.1, 0.9],
				neutral_band_factor: 0.4,
				momentum_factor: 1.0,
				..Default::default()
			},
		);
		let res = gen
			.generate_training_sequences_with_calibrated_params(
				df,
				&horizons,
				&training_config,
				&data_config,
				&feature_config,
				Some(&calibrated),
			)
			.await;
		assert!(res.is_ok(), "{res:?}");
	}

	#[tokio::test]
	async fn error_on_too_short_df() {
		let df = make_df(2);
		let horizons = vec!["1h".to_string()];
		let training_config = TrainingConfig {
			model: crate::config::model::ModelConfig {
				sequence_length: SequenceLengthConfig::Fixed(10),
				..Default::default()
			},
			..Default::default()
		};
		let gen = SequenceGenerator::default(); // Use default (no overlap)
		let data_config = DataConfig::default();
		let feature_config = FeatureConfig::default();
		let res = gen
			.generate_training_sequences(df, &horizons, &training_config, &data_config, &feature_config)
			.await;
		assert!(res.is_err());
	}
}

#[tokio::test]
async fn saved_feature_order_restores_identical_prediction_tensor() {
	let frame = df!(
		"timestamp" => vec![1704067200_i64, 1704070800, 1704074400],
		"open" => vec![10.0, 11.0, 12.0],
		"high" => vec![12.0, 14.0, 13.0],
		"low" => vec![9.0, 10.0, 11.0],
		"close" => vec![11.0, 12.0, 12.5],
		"volume" => vec![100.0, 140.0, 120.0]
	)
	.unwrap();
	let names = ["open", "high", "low", "close", "volume"].map(str::to_string);
	let reordered = frame
		.select(["volume", "close", "timestamp", "low", "high", "open"])
		.unwrap();
	let restored = SequenceGenerator::reorder_features(reordered, &names).unwrap();
	let config = crate::config::ModelConfig {
		sequence_length: SequenceLengthConfig::Fixed(3),
		..Default::default()
	};
	let generator = SequenceGenerator::default();
	let a = generator
		.generate_prediction_sequences(frame.clone(), "T", &config)
		.await
		.unwrap();
	let b = generator
		.generate_prediction_sequences(restored, "T", &config)
		.await
		.unwrap();
	assert_eq!(a.sequences, b.sequences);
	let mut wrong = names.to_vec();
	wrong[0] = "missing_open".to_string();
	let error = SequenceGenerator::reorder_features(frame, &wrong)
		.unwrap_err()
		.to_string();
	assert!(
		error.contains("missing_open") && error.contains("extra=[\"open\"]"),
		"{error}"
	);
}

#[test]
fn window_normalisation_leaves_context_columns_untouched() {
	let level = vec![0.4, 0.5, 0.6, 0.7];
	let window = df!(
		"timestamp" => [0i64, 3600, 7200, 10800],
		"close" => [10.0, 12.0, 14.0, 16.0],
		"ctx_vol_1" => level.clone(),
	)
	.unwrap();
	let normalised = SequenceGenerator::new().normalize_sequence_window(&window).unwrap();
	let kept: Vec<f64> = normalised
		.column("ctx_vol_1")
		.unwrap()
		.f64()
		.unwrap()
		.into_no_null_iter()
		.collect();
	assert_eq!(kept, level);
	let close: Vec<f64> = normalised
		.column("close")
		.unwrap()
		.f64()
		.unwrap()
		.into_no_null_iter()
		.collect();
	assert!(close.iter().sum::<f64>().abs() < 1e-9, "close is centred: {close:?}");
	assert!(close[0] < 0.0 && close[3] > 0.0);
}
