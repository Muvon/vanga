use super::*;

#[test]
fn prediction_history_guard_uses_saved_stats_and_sequence_lengths() {
    let mut training_config = crate::config::TrainingConfig::default();
    training_config.features.stats_window_bars = 720;
    training_config.model.sequence_length = crate::config::model::SequenceLengthConfig::Fixed(32);

    let error = validate_prediction_history(751, &training_config)
        .unwrap_err()
        .to_string();
    assert!(error.contains("rows given 751"), "{error}");
    assert!(error.contains("required 752"), "{error}");
    assert!(error.contains("stats_window_bars 720"), "{error}");
    assert!(error.contains("sequence_length 32"), "{error}");
    assert!(validate_prediction_history(752, &training_config).is_ok());
}
