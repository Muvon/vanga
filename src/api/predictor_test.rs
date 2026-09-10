#[test]
fn prediction_history_guard_uses_saved_stats_and_sequence_lengths() {
    let mut training_config = crate::config::TrainingConfig::default();
    training_config.features.stats_window_bars = 720;
    training_config.model.sequence_length = crate::config::model::SequenceLengthConfig::Fixed(32);

    let error = crate::data::validate_prediction_history(751, &training_config)
        .unwrap_err()
        .to_string();
    assert!(error.contains("rows given 751"), "{error}");
    assert!(error.contains("required 752"), "{error}");
    assert!(error.contains("stats_window_bars 720"), "{error}");
    assert!(error.contains("sequence_length 32"), "{error}");
    assert!(crate::data::validate_prediction_history(752, &training_config).is_ok());
}

#[test]
fn retained_prediction_history_ends_at_sequence_anchor() {
    use crate::data::structures::MarketDataRow;

    let rows: Vec<MarketDataRow> = (0..5)
        .map(|index| MarketDataRow {
            timestamp: index,
            open: index as f64 + 1.0,
            high: index as f64 + 1.0,
            low: index as f64 + 1.0,
            close: index as f64 + 1.0,
            volume: 1.0,
        })
        .collect();
    let sequence_anchor = rows.last().unwrap().clone();
    let mut retained = Some(rows[2..].to_vec());
    crate::data::retain_prediction_history(&mut retained, rows.clone()).unwrap();
    let retained = retained.unwrap();
    assert_eq!(retained.len(), rows.len());
    assert_eq!(
        retained.last().unwrap().timestamp,
        sequence_anchor.timestamp
    );
    assert_eq!(retained.last().unwrap().close, sequence_anchor.close);

    let mut wrong_timestamp = Some(rows[2..].to_vec());
    wrong_timestamp
        .as_mut()
        .unwrap()
        .last_mut()
        .unwrap()
        .timestamp += 1;
    assert!(crate::data::retain_prediction_history(&mut wrong_timestamp, rows).is_err());
}
