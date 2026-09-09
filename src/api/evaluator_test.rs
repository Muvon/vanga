use crate::api::evaluator::*;
use approx::assert_abs_diff_eq;
use chrono::{DateTime, Utc};
use std::collections::BTreeMap;

fn row(true_bin: usize, pred_bin: usize, max_prob: f64, confidence: f64) -> ScoredRow {
    let mut probs = [(1.0 - max_prob) / 4.0; 5];
    probs[pred_bin] = max_prob;
    ScoredRow {
        ts: "2026-01-01T00:00:00Z".into(),
        true_bin,
        pred_bin,
        probs,
        confidence,
        no_signal: false,
        close: 100.0,
        fut_max_high_rel: 0.1,
        fut_min_low_rel: -0.1,
        fut_close_rel: 0.05,
    }
}

#[test]
fn embargo_uses_original_window_start_and_keeps_every_valid_anchor() {
    // Features removed 13 warmup rows. split=40, seq=5, max_h=8 => start>=53.
    let original: Vec<usize> = (13..100).collect();
    for horizon in [4, 8] {
        let starts = holdout_starts(&original, 40, 5, 8, horizon, &[]).unwrap();
        assert_eq!(original[starts[0]], 53);
        assert_eq!(starts.len(), 100 - 53 - 5 - horizon + 1);
        assert_eq!(original[*starts.last().unwrap()] + 5 + horizon, 100);
        assert!(starts.iter().all(|&start| original[start] >= 40 + 5 + 8));
        assert!(starts.windows(2).all(|pair| pair[1] == pair[0] + 1));
        // Start 52 has an anchor after the embargo but must still be excluded.
        assert!(!starts.contains(&(52 - 13)));
        // A missing candle between raw rows 60 and 61 removes every window
        // covering both, i.e. starts 61-window+1..=60, and nothing else.
        let window = 5 + horizon;
        let gapped = holdout_starts(&original, 40, 5, 8, horizon, &[60]).unwrap();
        // Starts 48..=60 span the gap for horizon 8, but the embargo already
        // removed everything before 53, so exactly 53..=60 disappear.
        assert_eq!(starts.len() - gapped.len(), 60 - 53 + 1);
        assert!(gapped
            .iter()
            .all(|&s| original[s] > 60 || original[s] + window - 1 <= 60));
        assert!(gapped.contains(&(61 - 13)));
    }
    assert!(holdout_starts(&original, 95, 5, 8, 8, &[])
        .unwrap()
        .is_empty());
    assert!(holdout_starts(&original, usize::MAX, 5, 8, 8, &[]).is_err());
}

#[test]
fn handmade_ordinal_side_wide_f1_confusion_and_ece() {
    let metrics = compute_metrics(
        vec![
            row(0, 1, 0.6, 0.3), // ordinal 1, same side, too narrow, wrong
            row(4, 3, 0.6, 0.4), // ordinal 1, same side, wide enough, wrong
            row(2, 2, 0.8, 0.5), // ordinal 0, excluded from side, correct
            row(1, 4, 0.8, 0.6), // ordinal 3, wrong side, too narrow, wrong
            row(3, 3, 1.0, 0.9), // ordinal 0, same side, wide enough, correct
        ],
        true,
    )
    .unwrap();
    assert_eq!(metrics.n, 5);
    assert_eq!(
        metrics.confusion,
        [
            [0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0]
        ]
    );
    assert_abs_diff_eq!(metrics.accuracy, 0.4);
    assert_abs_diff_eq!(metrics.majority_class_baseline_accuracy, 0.2);
    assert_abs_diff_eq!(metrics.macro_f1, (1.0 + 2.0 / 3.0) / 5.0);
    assert_abs_diff_eq!(metrics.ordinal_mae, 1.0);
    assert_eq!(metrics.side_count, 4);
    assert_abs_diff_eq!(metrics.side_accuracy.unwrap(), 0.75);
    assert_abs_diff_eq!(metrics.wide_enough_rate.unwrap(), 0.6);
    // [.6,.7): 2 wrong at .6; [.8,.9): 1/2 correct at .8; [ .9,1]: 1 correct at 1.
    assert_abs_diff_eq!(metrics.ece, (2.0 * 0.6 + 2.0 * 0.3) / 5.0, epsilon = 1e-14);
    assert_eq!(metrics.confidence_deciles.len(), 10);
    assert_eq!(
        metrics
            .confidence_deciles
            .iter()
            .map(|d| d.n)
            .sum::<usize>(),
        5
    );
    assert_eq!(metrics.confidence_deciles[0].acc, None);
    assert_eq!(metrics.confidence_deciles[9].n, 1);
    assert_abs_diff_eq!(metrics.confidence_deciles[9].mean_confidence.unwrap(), 0.9);
    assert_abs_diff_eq!(metrics.top_decile_acc, 1.0);
}

#[test]
fn decile_ties_use_chronological_order_and_missing_side_is_null() {
    let rows = (0..20)
        .map(|i| row(2, if i >= 18 { 2 } else { 0 }, 0.5, 0.7))
        .collect();
    let m = compute_metrics(rows, false).unwrap();
    assert_eq!(m.side_count, 0);
    assert_eq!(m.side_accuracy, None);
    assert_eq!(m.wide_enough_rate, None);
    assert_abs_diff_eq!(m.top_decile_acc, 1.0);
    assert!(m.confidence_deciles.iter().all(|d| d.n == 2));
    assert!(compute_metrics(Vec::new(), false).is_err());
    assert!(compute_metrics(vec![row(5, 0, 0.5, 0.5)], false).is_err());
}

#[test]
fn fractional_and_iso_splits_are_exact() {
    let times: Vec<DateTime<Utc>> = (0..11)
        .map(|i| DateTime::from_timestamp(1_767_225_600 + i * 3600, 0).unwrap())
        .collect();
    assert_eq!(split_row(&times, "0.8").unwrap(), 8);
    assert_eq!(split_row(&times, "2026-01-01T07:30:00Z").unwrap(), 8);
    assert_eq!(split_row(&times, "2026-01-01T09:00:00+01:00").unwrap(), 8);
    assert_eq!(split_row(&times, "2026-01-01").unwrap(), 0);
    for invalid in ["NaN", "-0.1", "1.0", "nonsense", "2027-01-01"] {
        assert!(split_row(&times, invalid).is_err());
    }
}

#[test]
fn sha1_matches_known_vectors_including_multiblock_padding() {
    assert_eq!(sha1_hex(b""), "da39a3ee5e6b4b0d3255bfef95601890afd80709");
    assert_eq!(sha1_hex(b"abc"), "a9993e364706816aba3e25717850c26c9cd0d89d");
    assert_eq!(
        sha1_hex(b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"),
        "84983e441c3bd26ebaae4aa1f95129e5e54670f1"
    );
}

#[test]
fn json_is_stable_and_csv_appends_escaped_fields_with_one_header() {
    let make_report = || Evaluation {
        meta: EvaluationMeta {
            symbol: "ETHUSDT".into(),
            run_id: "000-base".into(),
            change: "a,b\nquoted \"change\"".into(),
            config_sha1: sha1_hex(b"config"),
            seed: 42,
            split_row: 80,
            split_ts: "2026-01-01T00:00:00Z".into(),
            holdout_from: "0.8".into(),
            horizons: vec!["8h".into()],
            sequence_length: 5,
            input_size: 6,
            n_rows: 100,
        },
        heads: BTreeMap::from([(
            "stop_level_8h".into(),
            compute_metrics(vec![row(2, 2, 0.8, 0.7)], true).unwrap(),
        )]),
    };
    let report = make_report();
    assert_eq!(
        serde_json::to_vec_pretty(&report).unwrap(),
        serde_json::to_vec_pretty(&make_report()).unwrap()
    );
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("results.csv");
    append_results(&path, &report).unwrap();
    append_results(&path, &report).unwrap();
    let mut reader = csv::Reader::from_path(&path).unwrap();
    assert_eq!(reader.headers().unwrap().len(), 19);
    let rows: Vec<_> = reader.records().map(|r| r.unwrap()).collect();
    assert_eq!(rows.len(), 2);
    for record in rows {
        assert_eq!(&record[7], report.meta.change);
        assert_eq!(&record[17], "");
        assert_eq!(&record[18], "");
    }
}

#[tokio::test]
async fn saved_model_round_trip_is_byte_identical_and_preserves_full_holdout() {
    use crate::config::model::{
        AttentionMechanism, DropoutRate, HiddenUnitsConfig, LSTMArchitecture, MoHConfig,
        SequenceLengthConfig,
    };
    use crate::config::TrainingConfig;
    use crate::data::{DataLoader, DataPreprocessor};
    use crate::model::multi_target::MultiTargetLSTMModel;
    use crate::targets::calibration::{
        CalibratedParameters, CalibrationMetadata, ClassBalance, DirectionParams, PriceLevelParams,
        StopLevelParams,
    };
    use std::io::Write;

    let temp = tempfile::tempdir().unwrap();
    let csv_path = temp.path().join("ETHUSDT.csv");
    let mut csv = std::fs::File::create(&csv_path).unwrap();
    writeln!(csv, "timestamp,open,high,low,close,volume").unwrap();
    for i in 0..200 {
        let close = 100.0 + i as f64 * 0.05 + (i as f64 * 0.4).sin();
        writeln!(
            csv,
            "{},{},{},{},{},100",
            1_767_225_600 + i * 3600,
            close - 0.1,
            close + 1.0,
            close - 1.0,
            close
        )
        .unwrap();
    }
    drop(csv);
    let mut config = TrainingConfig {
        horizons: vec!["2h".into(), "4h".into()],
        ..Default::default()
    };
    config.model.sequence_length = SequenceLengthConfig::Fixed(5);
    config.model.architecture = LSTMArchitecture::MultiLSTM { layers: 1 };
    config.model.hidden_units = HiddenUnitsConfig::Fixed(vec![8]);
    config.model.attention.enabled = true;
    config.model.attention.mechanism = AttentionMechanism::MixtureOfHeads;
    config.model.attention.heads = 2;
    config.model.attention.head_dim = Some(4);
    config.model.attention.moh = Some(MoHConfig {
        total_heads: 2,
        shared_heads: 1,
        top_k: 1,
        sparse_attention: true,
        volatility_adaptive: true,
        ..Default::default()
    });
    config.model.layer_norm.enabled = false;
    config.model.dain = None;
    config.model.xgboost.enabled = false;
    config.model.dropout.enabled = true;
    config.model.dropout.rate = DropoutRate::Fixed(0.5);
    config.features.technical_indicators.enabled = false;
    config.features.engineering.polynomial_features.enabled = false;
    config.features.engineering.interaction_features.enabled = false;
    config.features.engineering.rolling_features.enabled = false;
    config.features.engineering.lag_features.enabled = true;
    config.features.engineering.lag_features.lag_periods = vec![1];
    config.features.engineering.lag_features.features_to_lag = vec!["close".into()];
    config.data.outlier_handling.enabled = false;
    // These training-only settings must have no effect on the scorer.
    config.data.sequence_overlap = 0.0;
    config.data.target_samples.count = 1;
    config.data.target_samples.truncate = true;
    let raw = DataLoader::new().load_csv(&csv_path).await.unwrap();
    let processed = DataPreprocessor::new()
        .process_features_only(raw, &config.data, Some(&config.features))
        .await
        .unwrap();
    let features = processed.width() - 1;
    let names: Vec<String> = config
        .horizons
        .iter()
        .flat_map(|h| {
            [
                format!("price_level_{h}"),
                format!("stop_level_{h}"),
                format!("direction_{h}"),
            ]
        })
        .collect();
    let mut model = MultiTargetLSTMModel::new_with_seed(
        &config.model,
        features,
        names,
        config.horizons.clone(),
        Some(42),
        Some(candle_core::Device::Cpu),
    )
    .unwrap();
    // Save fixed initialized weights, without any training or calibration call.
    for head in &mut model.models {
        head.initialize_network(None).unwrap();
    }
    let mut calibration = CalibratedParameters {
        metadata: CalibrationMetadata {
            overall_balance_score: 0.0,
            ..Default::default()
        },
        ..Default::default()
    };
    let balance = ClassBalance {
        composite_quality_score: 0.0,
        ..Default::default()
    };
    for h in &config.horizons {
        calibration.direction.insert(
            h.clone(),
            DirectionParams {
                sensitivity: 0.4,
                extreme_multiplier: 2.0,
                min_base_threshold: 0.01,
                min_extreme_threshold: 0.03,
                base_multiplier: 20.0,
                balance: balance.clone(),
            },
        );
        calibration.price_levels.insert(
            h.clone(),
            PriceLevelParams {
                bandwidth: 1.0,
                percentiles: [0.1, 0.9],
                neutral_band_factor: 0.4,
                momentum_factor: 1.2,
                balance: balance.clone(),
            },
        );
        calibration.stop_levels.insert(
            h.clone(),
            StopLevelParams {
                bandwidth: 1.0,
                percentiles: [0.1, 0.9],
                neutral_band_factor: 0.4,
                momentum_factor: 1.2,
                balance: balance.clone(),
            },
        );
    }
    model.set_training_config(config.clone());
    model.set_calibrated_parameters(calibration);
    model.save(temp.path().join("ETHUSDT")).unwrap();
    // Production veto: confident price-up / direction-down is refused even at
    // min_confidence=0. Correct OHLCV, price and target names do not bypass it.
    let ohlcv = crate::utils::market_data::extract_ohlcv_data(&processed).unwrap();
    let formatter = crate::output::OutputFormatter::new(Default::default())
        .with_target_names(model.get_target_names())
        .with_sequence_ohlcv(ohlcv[..5].to_vec())
        .with_metadata(features, 5)
        .with_min_confidence(0.0)
        .with_calibrated_parameters(model.get_calibrated_parameters().unwrap().clone());
    let price_probs = [0.01, 0.01, 0.01, 0.02, 0.95];
    let stop_probs = [0.01, 0.02, 0.94, 0.02, 0.01];
    let direction_probs = [0.95, 0.02, 0.01, 0.01, 0.01];
    let local: Vec<f64> = [price_probs, stop_probs, direction_probs].concat();
    let predictions =
        ndarray::Array2::from_shape_vec((1, 30), [local.clone(), local.clone()].concat()).unwrap();
    let refused = formatter
        .format_predictions_with_targets(
            &predictions,
            "ETHUSDT",
            "2h",
            ohlcv[4].close,
            None,
            Some(model.get_target_names()),
        )
        .unwrap();
    assert!(
        refused.is_empty(),
        "Production must veto opposing confident signals"
    );
    let parser = crate::output::multi_target_parser::MultiTargetParser::with_target_names(
        &model.get_target_names()[..3],
    );
    let parsed = parser
        .parse_output(ndarray::ArrayView1::from(&local))
        .unwrap();
    assert_eq!(parsed.price_levels.unwrap(), price_probs);
    let confidence =
        head_confidence(refused.first(), crate::targets::TargetType::PriceLevel).unwrap();
    assert_eq!(confidence, 0.0);
    let mut no_signal = row(4, 4, 0.95, confidence);
    no_signal.probs = price_probs;
    no_signal.no_signal = refused.is_empty();
    let metrics = compute_metrics(vec![no_signal], false).unwrap();
    assert_eq!(metrics.n, 1);
    assert_eq!(metrics.n_no_signal, 1);
    assert_eq!(metrics.confusion[4][4], 1);
    assert_eq!(metrics.rows[0].probs, price_probs);

    let config_path = temp.path().join("config.toml");
    // Saved horizons and sequence length win over the evaluation config.
    config.horizons = vec!["99h".into()];
    config.training.seed = 99;
    config.model.sequence_length = SequenceLengthConfig::Fixed(100);
    std::fs::write(&config_path, toml::to_string(&config).unwrap()).unwrap();
    let output = temp.path().join("eval.json");
    let options = || EvaluationOptions {
        symbol: "ETHUSDT".into(),
        test_data: csv_path.clone(),
        holdout_from: "0.5".into(),
        config: config_path.clone(),
        models_dir: temp.path().into(),
        horizons: None,
        run_id: "000-base".into(),
        change: "none".into(),
        results_csv: None,
        eval_json: output.clone(),
    };
    evaluate(options()).await.unwrap();
    let first = std::fs::read(&output).unwrap();
    evaluate(options()).await.unwrap();
    assert_eq!(first, std::fs::read(&output).unwrap());
    let report: serde_json::Value = serde_json::from_slice(&first).unwrap();
    assert_eq!(report["meta"]["split_row"], 100);
    assert_eq!(report["meta"]["seed"], 99);
    assert_eq!(report["meta"]["sequence_length"], 5);
    assert_eq!(report["meta"]["horizons"], serde_json::json!(["2h", "4h"]));
    assert_eq!(report["heads"]["direction_2h"]["n"], 85);
    assert_eq!(report["heads"]["direction_4h"]["n"], 83);
    let first_row = &report["heads"]["price_level_2h"]["rows"][0];
    // split 100 + embargo(5+4) => input start 109 => anchor 113.
    let anchor_ts = DateTime::from_timestamp(1_767_225_600 + 113 * 3600, 0)
        .unwrap()
        .to_rfc3339_opts(chrono::SecondsFormat::Secs, true);
    assert_eq!(first_row["ts"], anchor_ts);
    let close_at = |i: usize| 100.0 + i as f64 * 0.05 + (i as f64 * 0.4).sin();
    assert_abs_diff_eq!(
        first_row["fut_close_rel"].as_f64().unwrap(),
        close_at(115) / close_at(113) - 1.0,
        epsilon = 1e-14
    );
    for head in report["heads"].as_object().unwrap().values() {
        let rows = head["rows"].as_array().unwrap();
        let refused = rows.iter().filter(|row| row["no_signal"] == true).count();
        assert_eq!(head["n_no_signal"], refused);
        for row in rows.iter().filter(|row| row["no_signal"] == true) {
            assert_eq!(row["confidence"], 0.0);
        }
    }
    // MoH sparse routing must match an isolated production predict call exactly.
    let loaded = MultiTargetLSTMModel::load(temp.path().join("ETHUSDT")).unwrap();
    let seq_start = processed
        .column("timestamp")
        .unwrap()
        .i64()
        .unwrap()
        .into_no_null_iter()
        .position(|ts| ts == 1_767_225_600 + 109 * 3600)
        .unwrap();
    let generator = crate::data::SequenceGenerator::new();
    let normalized = generator
        .normalize_sequence_window(&processed.slice(seq_start as i64, 5))
        .unwrap();
    let feature_names = processed
        .get_column_names()
        .iter()
        .filter(|name| name.as_str() != "timestamp")
        .map(|name| name.to_string())
        .collect::<Vec<_>>();
    let matrix = generator
        .extract_feature_matrix(&normalized, &feature_names)
        .unwrap();
    let mut sequence = ndarray::Array3::zeros((1, 5, features));
    sequence.slice_mut(ndarray::s![0, .., ..]).assign(&matrix);
    let prediction = loaded.predict(&sequence).await.unwrap();
    let parsed = parser
        .parse_output(prediction.slice(ndarray::s![0, 0..15]))
        .unwrap();
    for (expected, actual) in parsed
        .price_levels
        .unwrap()
        .iter()
        .zip(first_row["probs"].as_array().unwrap())
    {
        assert_abs_diff_eq!(*expected, actual.as_f64().unwrap(), epsilon = 1e-12);
    }
    // Selecting a shorter horizon must retain the longest TRAINED embargo, with
    // identical rows, predictions, labels, no-signal counts and all other metrics.
    let mut subset_options = options();
    subset_options.horizons = Some(vec!["2h".into()]);
    evaluate(subset_options).await.unwrap();
    let subset: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&output).unwrap()).unwrap();
    assert_eq!(subset["heads"].as_object().unwrap().len(), 3);
    for name in ["price_level_2h", "stop_level_2h", "direction_2h"] {
        assert_eq!(subset["heads"][name], report["heads"][name]);
    }
    // Changed feature identity is rejected BEFORE preprocessing, even when the
    // column count stays the same (close_lag_1 -> close_lag_2).
    config.features.engineering.lag_features.lag_periods = vec![2];
    std::fs::write(&config_path, toml::to_string(&config).unwrap()).unwrap();
    let error = evaluate(options()).await.unwrap_err().to_string();
    assert!(
        error.contains("features.engineering.lag_features.lag_periods"),
        "{error}"
    );
    // A corrupt saved feature contract still fails with both feature counts.
    config.features.engineering.lag_features.lag_periods = vec![1];
    config.features.engineering.lag_features.enabled = false;
    std::fs::write(&config_path, toml::to_string(&config).unwrap()).unwrap();
    let mut incompatible_saved = model.get_training_config().unwrap().clone();
    incompatible_saved.features = config.features.clone();
    model.set_training_config(incompatible_saved);
    model.save(temp.path().join("ETHUSDT")).unwrap();
    let error = evaluate(options()).await.unwrap_err().to_string();
    assert!(
        error.contains(&format!("saved input_size={features}")),
        "{error}"
    );
    assert!(error.contains("computed feature count=6"), "{error}");
    std::fs::remove_file(temp.path().join("ETHUSDT_0.safetensors")).unwrap();
    assert!(evaluate(options())
        .await
        .unwrap_err()
        .to_string()
        .contains("Weights file not found"));
}

#[test]
fn preprocessing_contract_reports_all_differing_keys() {
    let saved = crate::config::TrainingConfig::default();
    let mut requested = saved.clone();
    requested.training.seed += 1;
    requested.horizons = vec!["99h".into()];
    validate_preprocessing_config(&saved, &requested).unwrap();
    requested.features.engineering.lag_features.lag_periods = vec![2, 4];
    requested.data.outlier_handling.threshold += 0.5;
    let error = validate_preprocessing_config(&saved, &requested)
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("features.engineering.lag_features.lag_periods"),
        "{error}"
    );
    assert!(error.contains("data.outlier_handling.threshold"), "{error}");
    assert!(
        error.contains("saved=") && error.contains("--config="),
        "{error}"
    );
    assert!(!error.contains("training.seed"));
    assert!(!error.contains("horizons"));
}

#[test]
fn cadence_guard_rejects_changed_detector_window_and_gaps_after_first_100_rows() {
    use crate::utils::parser::detect_timeframe_minutes;
    use polars::prelude::*;
    let epoch = 1_767_225_600;
    let dense: Vec<_> = (0..200)
        .map(|i| DateTime::from_timestamp(epoch + i * 3600, 0).unwrap())
        .collect();
    assert!(validate_cadence(&dense, 60, 60).unwrap().is_empty());
    let mut gaps = dense.clone();
    for ts in &mut gaps[150..] {
        *ts += chrono::Duration::hours(1);
    }
    let frame =
        df!("timestamp" => gaps.iter().map(DateTime::timestamp).collect::<Vec<_>>()).unwrap();
    let raw_minutes = detect_timeframe_minutes(&frame).unwrap();
    let processed_minutes = detect_timeframe_minutes(&frame.slice(1, 199)).unwrap();
    assert_eq!((raw_minutes, processed_minutes), (60, 60));
    // One whole missing candle is reported as a gap index, not an error.
    assert_eq!(
        validate_cadence(&gaps, raw_minutes, processed_minutes).unwrap(),
        vec![149]
    );
    // Raw detector samples the initial hourly region; after warmup the processed
    // detector instead sees half-hour candles. Both values must be reported.
    let changed: Vec<_> = (0..200)
        .map(|i| {
            epoch
                + if i < 100 {
                    i * 3600
                } else {
                    100 * 3600 + (i - 100) * 1800
                }
        })
        .collect();
    let frame = df!("timestamp" => &changed).unwrap();
    let raw_minutes = detect_timeframe_minutes(&frame).unwrap();
    let processed_minutes = detect_timeframe_minutes(&frame.slice(100, 100)).unwrap();
    let ts = changed
        .iter()
        .map(|epoch| DateTime::from_timestamp(*epoch, 0).unwrap())
        .collect::<Vec<_>>();
    let error = validate_cadence(&ts, raw_minutes, processed_minutes)
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("raw=60 minutes, processed=30 minutes"),
        "{error}"
    );
    // Exact gaps, not rounded minute differences: subsecond jitter also fails.
    let mut jitter = dense;
    jitter[180] += chrono::Duration::milliseconds(1);
    assert!(validate_cadence(&jitter, 60, 60).is_err());
}
