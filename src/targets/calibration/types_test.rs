use super::types::{CalibratedParameters, CalibrationMetadata, PriceLevelParams, StopLevelParams};

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
