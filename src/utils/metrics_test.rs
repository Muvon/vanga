use super::*;

#[test]
fn test_classification_metrics() {
    let pred = vec![1, 0, 1, 1];
    let tgt = vec![1, 0, 0, 1];
    let metrics = calculate_classification_metrics(&pred, &tgt).unwrap();
    assert!(metrics.accuracy >= 0.5);
}

#[test]
fn test_regression_metrics() {
    let pred = vec![1.0, 2.0, 3.0];
    let tgt = vec![1.1, 1.9, 3.2];
    let metrics = calculate_regression_metrics(&pred, &tgt).unwrap();
    assert!(metrics.rmse > 0.0);
}
