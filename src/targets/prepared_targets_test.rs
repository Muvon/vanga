use super::PreparedTargets;

#[test]
fn active_target_subset_validates_with_stop_level() {
	let mut targets = PreparedTargets::new(2);
	targets.price_levels.insert("3d".to_string(), vec![0, 4]);
	targets.stop_levels.insert("3d".to_string(), vec![1, 3]);
	targets.direction.insert("3d".to_string(), vec![2, 2]);

	assert!(targets.validate().is_ok());
}

#[test]
fn stop_level_length_mismatch_is_rejected() {
	let mut targets = PreparedTargets::new(2);
	targets.stop_levels.insert("3d".to_string(), vec![1]);

	let error = targets.validate().unwrap_err().to_string();
	assert!(error.contains("stop_level target length 1"));
}
