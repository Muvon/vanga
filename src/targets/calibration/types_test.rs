use super::types::{CalibratedParameters, CalibrationMetadata, ClassBalance, PriceLevelParams, StopLevelParams};

fn balance_of(counts: [usize; 5]) -> ClassBalance {
	let total: usize = counts.iter().sum();
	ClassBalance {
		class_percentages: counts.map(|count| count as f64 / total as f64 * 100.0),
		total_samples: total,
		..ClassBalance::default()
	}
}

#[test]
fn most_even_split_allows_one_sample_of_difference() {
	assert!(balance_of([180, 180, 180, 180, 180]).is_most_even_split());
	// 901 samples cannot be split evenly: one class with one extra sample is the best possible.
	assert!(balance_of([181, 180, 180, 180, 180]).is_most_even_split());
	assert!(balance_of([181, 181, 180, 181, 181]).is_most_even_split());
}

#[test]
fn most_even_split_rejects_larger_differences_and_empty_input() {
	assert!(!balance_of([182, 180, 180, 180, 179]).is_most_even_split());
	assert!(!balance_of([2, 1, 1, 1, 0]).is_most_even_split());
	assert!(!ClassBalance::default().is_most_even_split());
}

#[test]
fn horizons_do_not_depend_on_direction_being_enabled() {
	let mut parameters = CalibratedParameters::default();
	parameters
		.price_levels
		.insert("3d".to_string(), PriceLevelParams::default());
	parameters
		.stop_levels
		.insert("3d".to_string(), StopLevelParams::default());

	assert_eq!(parameters.get_horizons(), vec!["3d"]);
	assert!(parameters.validate().is_ok());
}

#[test]
fn metadata_preserves_configured_horizon_order_without_duplicates() {
	let parameters = CalibratedParameters {
		metadata: CalibrationMetadata {
			horizons: vec!["3d".to_string(), "1d".to_string(), "3d".to_string()],
			..CalibrationMetadata::default()
		},
		..CalibratedParameters::default()
	};

	assert_eq!(parameters.get_horizons(), vec!["3d", "1d"]);
}
