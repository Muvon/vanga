use super::{DataPreprocessor, ReplacementStrategy};
use crate::config::training::OutlierMethod;
use polars::prelude::*;

#[test]
fn all_outlier_statistics_and_replacements_are_causal() {
	let preprocessor = DataPreprocessor::new();
	let prefix = vec![10.0, 11.0, 9.0, 12.0, 8.0, 10.0, 11.0, 9.0, 1000.0];
	let mut values = prefix.clone();
	values.extend([1e9, -1e9, 1e12, -1e12]);
	for method in [OutlierMethod::IQR, OutlierMethod::ZScore, OutlierMethod::ModifiedZScore] {
		for name in ["volume", "rsi_14", "price_return"] {
			let a = DataFrame::new(vec![Series::new(name.into(), prefix.clone()).into_column()]).unwrap();
			let b = DataFrame::new(vec![Series::new(name.into(), values.clone()).into_column()]).unwrap();
			let a = preprocessor.remove_outliers_causal(a, &method, 1.5, 7).unwrap();
			let b = preprocessor.remove_outliers_causal(b, &method, 1.5, 7).unwrap();
			let left: Vec<_> = a.column(name).unwrap().f64().unwrap().into_iter().collect();
			let right: Vec<_> = b
				.column(name)
				.unwrap()
				.f64()
				.unwrap()
				.into_iter()
				.take(prefix.len())
				.collect();
			assert_eq!(left, right, "{method:?} {name}");
			assert!(left[8].unwrap() < 1000.0, "outlier unchanged for {method:?} {name}");
			if name == "price_return" {
				assert_eq!(left[8], Some(9.0));
			}
		}
	}
}

#[test]
fn statistics_window_defaults_and_rejects_zero() {
	let config = crate::config::FeatureConfig::default();
	assert_eq!(config.stats_window_bars, 720);
	let mut json = serde_json::to_value(&config).unwrap();
	json.as_object_mut().unwrap().remove("stats_window_bars");
	let restored: crate::config::FeatureConfig = serde_json::from_value(json).unwrap();
	assert_eq!(restored.stats_window_bars, 720);
	let frame = df!("volume" => [1.0, 2.0]).unwrap();
	assert!(DataPreprocessor::new()
		.remove_outliers_causal(frame, &OutlierMethod::IQR, 1.5, 0)
		.is_err());
}

#[test]
fn outlier_capping_skips_context_columns() {
	let values = vec![0.1, 0.2, 0.1, 0.2, 0.1, 0.2, 0.1, 0.2, 9.0];
	let frame = df!("ctx_vol_1" => values.clone(), "volume" => values.clone()).unwrap();
	let cleaned = DataPreprocessor::new()
		.remove_outliers_causal(frame, &OutlierMethod::ModifiedZScore, 4.0, 7)
		.unwrap();
	let read = |name: &str| -> Vec<f64> {
		cleaned
			.column(name)
			.unwrap()
			.f64()
			.unwrap()
			.into_no_null_iter()
			.collect()
	};
	assert_eq!(read("ctx_vol_1"), values);
	assert!(
		read("volume")[8] < 9.0,
		"the same spike is capped in an ordinary column"
	);
}

#[tokio::test]
async fn context_columns_are_added_only_when_enabled() {
	let rows = 120usize;
	let closes: Vec<f64> = (0..rows)
		.map(|i| 100.0 + (i as f64 * 0.7).sin() * 3.0 + i as f64 * 0.05)
		.collect();
	let frame = df!(
		"timestamp" => (0..rows as i64).map(|i| 1_767_571_200 + i * 3600).collect::<Vec<i64>>(),
		"open" => closes.clone(),
		"high" => closes.iter().map(|close| close + 1.0).collect::<Vec<f64>>(),
		"low" => closes.iter().map(|close| close - 1.0).collect::<Vec<f64>>(),
		"close" => closes.clone(),
		"volume" => (0..rows).map(|i| 10.0 + (i % 5) as f64).collect::<Vec<f64>>(),
	)
	.unwrap();
	let data_config = crate::config::training::DataConfig::default();
	let mut features = crate::config::FeatureConfig::default();
	features.technical_indicators.enabled = false;
	features.stats_window_bars = 27;
	let count_context = |frame: &DataFrame| {
		frame
			.get_column_names()
			.iter()
			.filter(|name| crate::features::context::is_context_column(name))
			.count()
	};
	let without = DataPreprocessor::new()
		.process_features_only(frame.clone(), &data_config, Some(&features))
		.await
		.unwrap();
	assert_eq!(count_context(&without), 0);
	features.context.enabled = true;
	let with = DataPreprocessor::new()
		.process_features_only(frame, &data_config, Some(&features))
		.await
		.unwrap();
	assert_eq!(count_context(&with), 11);
	assert_eq!(with.width(), without.width() + 11);
	for name in with.get_column_names() {
		if let Ok(values) = with.column(name).unwrap().f64() {
			assert!(
				values.into_iter().all(|value| value.is_some_and(f64::is_finite)),
				"{name} has a gap"
			);
		}
	}
}

#[test]
fn test_interpolation_strategy_selection() {
	let preprocessor = DataPreprocessor;

	// Test that interpolation is selected for price-derived features
	assert!(matches!(
		preprocessor.get_replacement_strategy("price_return"),
		ReplacementStrategy::Interpolate
	));
	assert!(matches!(
		preprocessor.get_replacement_strategy("momentum_5"),
		ReplacementStrategy::Interpolate
	));
	assert!(matches!(
		preprocessor.get_replacement_strategy("volatility_10"),
		ReplacementStrategy::Interpolate
	));

	// Test that median is selected for technical indicators
	assert!(matches!(
		preprocessor.get_replacement_strategy("rsi_14"),
		ReplacementStrategy::Median
	));
	assert!(matches!(
		preprocessor.get_replacement_strategy("sma_20"),
		ReplacementStrategy::Median
	));

	// Test that cap is selected for volume and other features
	assert!(matches!(
		preprocessor.get_replacement_strategy("volume"),
		ReplacementStrategy::Cap
	));
	assert!(matches!(
		preprocessor.get_replacement_strategy("some_other_feature"),
		ReplacementStrategy::Cap
	));
}

#[test]
fn test_extract_recent_clean_data() {
	use polars::prelude::*;

	let preprocessor = DataPreprocessor::new();

	// Create test data with NaN in middle but clean at end
	let data = vec![
		1.0,
		2.0,
		f64::NAN,
		4.0,
		5.0,
		6.0,
		7.0,
		8.0,
		9.0,
		10.0, // 10 rows
	];
	let df = DataFrame::new(
		vec![
			Series::new("price".into(), data),
			Series::new("volume".into(), vec![100.0; 10]),
		]
		.into_iter()
		.map(|s| s.into_column())
		.collect(),
	)
	.unwrap();

	// Extract 5 most recent clean rows
	let result = preprocessor.extract_recent_clean_data(df, 5).unwrap();

	// Should get rows 5-9 (6.0, 7.0, 8.0, 9.0, 10.0)
	assert_eq!(result.height(), 5);
	let price_col = result.column("price").unwrap().f64().unwrap();
	assert_eq!(price_col.get(0).unwrap(), 6.0);
	assert_eq!(price_col.get(4).unwrap(), 10.0);
}

#[test]
fn test_extract_recent_clean_data_insufficient() {
	use polars::prelude::*;

	let preprocessor = DataPreprocessor::new();

	// Create test data with only 3 rows
	let df = DataFrame::new(
		vec![Series::new("price".into(), vec![1.0, 2.0, 3.0])]
			.into_iter()
			.map(|s| s.into_column())
			.collect(),
	)
	.unwrap();

	// Try to extract 5 rows - should fail
	let result = preprocessor.extract_recent_clean_data(df, 5);
	assert!(result.is_err());
}
