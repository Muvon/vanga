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
