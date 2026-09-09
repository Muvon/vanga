use super::formatter::infer_stop_is_bullish;
use super::prediction_types::{PriceBin, PriceLevelPrediction};
use std::collections::HashMap;

fn price_level_prediction(price: [f64; 2]) -> PriceLevelPrediction {
    let mut bins = HashMap::new();
    bins.insert(
        "predicted".to_string(),
        PriceBin {
            range: [0.0, 0.0],
            vwap_range: [0.0, 0.0],
            price,
            probability: 1.0,
        },
    );
    PriceLevelPrediction {
        bins,
        most_likely_range: [0.0, 0.0],
        confidence: 1.0,
    }
}

#[test]
fn stop_direction_uses_predicted_future_price_level() {
    let bullish = price_level_prediction([104.0, 106.0]);
    let bearish = price_level_prediction([94.0, 96.0]);

    assert_eq!(infer_stop_is_bullish(&bullish, 100.0), Some(true));
    assert_eq!(infer_stop_is_bullish(&bearish, 100.0), Some(false));
}

#[test]
fn probabilities_reject_nonfinite_and_negative_for_every_head() {
    use crate::output::multi_target_parser::MultiTargetParser;
    for target in [
        "price_level",
        "stop_level",
        "direction",
        "volatility",
        "sentiment",
        "volume",
    ] {
        let parser = MultiTargetParser::with_target_names(&[format!("{target}_8h")]);
        for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -0.01] {
            let values = ndarray::array![0.2, 0.2, value, 0.2, 0.2];
            assert!(
                parser.parse_output(values.view()).is_err(),
                "{target} accepted {value}"
            );
        }
        assert!(parser
            .parse_output(ndarray::array![0.1, 0.2, 0.3, 0.2, 0.2].view())
            .is_ok());
    }
}

#[test]
fn stop_side_ties_are_long_and_missing_price_signal_has_no_side() {
    let flat = price_level_prediction([99.0, 101.0]);
    assert_eq!(infer_stop_is_bullish(&flat, 100.0), Some(true));
    assert_eq!(infer_stop_is_bullish(&flat, f64::NAN), None);
    let mut missing = flat;
    missing.bins.clear();
    assert_eq!(infer_stop_is_bullish(&missing, 100.0), None);
}

#[test]
fn serving_skips_stop_only_predictions_without_aborting_horizon() {
    let names = vec!["stop_level_8h".to_string()];
    let formatter = crate::output::OutputFormatter::new(Default::default())
        .with_target_names(&names)
        .with_min_confidence(0.0);
    let predictions = ndarray::Array2::from_elem((1, 5), 0.2);
    let results = formatter
        .format_predictions_with_targets(&predictions, "T", "8h", 100.0, None, Some(&names))
        .unwrap();
    assert_eq!(results.len(), 1);
    assert!(results[0].price_levels.is_none());
    assert!(results[0].stop_levels.is_none());
}
