use crate::utils::metrics::*;

#[test]
fn test_classification_metrics() {
    let pred = vec![1, 0, 1, 1];
    let tgt = vec![1, 0, 0, 1];
    let metrics = calculate_classification_metrics(&pred, &tgt).unwrap();

    // Accuracy should be 3/4 = 0.75 (3 correct predictions out of 4)
    assert_eq!(metrics.accuracy, 0.75);

    // Verify we have per-class metrics
    assert!(!metrics.precision.is_empty());
    assert!(!metrics.recall.is_empty());
    assert!(!metrics.f1_score.is_empty());

    // Macro F1 and Weighted F1 should be different from accuracy
    // (unless by coincidence, but they use different calculations)
    assert!(metrics.macro_f1 >= 0.0 && metrics.macro_f1 <= 1.0);
    assert!(metrics.weighted_f1 >= 0.0 && metrics.weighted_f1 <= 1.0);
}

#[test]
fn test_classification_metrics_perfect_prediction() {
    let pred = vec![0, 1, 2, 0, 1];
    let tgt = vec![0, 1, 2, 0, 1];
    let metrics = calculate_classification_metrics(&pred, &tgt).unwrap();

    // Perfect prediction should have all metrics = 1.0
    assert_eq!(metrics.accuracy, 1.0);
    assert_eq!(metrics.macro_f1, 1.0);
    assert_eq!(metrics.weighted_f1, 1.0);

    // All per-class metrics should be 1.0
    for &precision in metrics.precision.values() {
        assert_eq!(precision, 1.0);
    }
    for &recall in metrics.recall.values() {
        assert_eq!(recall, 1.0);
    }
    for &f1 in metrics.f1_score.values() {
        assert_eq!(f1, 1.0);
    }
}

#[test]
fn test_classification_metrics_imbalanced_classes() {
    // Imbalanced dataset: mostly class 0, few class 1
    let pred = vec![0, 0, 0, 1, 0, 0, 1, 0];
    let tgt = vec![0, 0, 0, 0, 1, 1, 1, 0];
    let metrics = calculate_classification_metrics(&pred, &tgt).unwrap();

    // Accuracy should be 5/8 = 0.625 (correct predictions: indices 0,1,2,7,4 wrong: 3,5,6)
    // pred: [0, 0, 0, 1, 0, 0, 1, 0]
    // tgt:  [0, 0, 0, 0, 1, 1, 1, 0]
    // matches: 0==0, 0==0, 0==0, 1!=0, 0!=1, 0!=1, 1==1, 0==0 = 5 correct out of 8
    assert_eq!(metrics.accuracy, 0.625);

    // Macro F1 and Weighted F1 should be different due to class imbalance
    assert!(metrics.macro_f1 != metrics.weighted_f1);
    assert!(metrics.macro_f1 != metrics.accuracy);
    assert!(metrics.weighted_f1 != metrics.accuracy);

    // Should have metrics for both classes 0 and 1
    assert!(metrics.precision.contains_key(&0));
    assert!(metrics.precision.contains_key(&1));
    assert!(metrics.recall.contains_key(&0));
    assert!(metrics.recall.contains_key(&1));
    assert!(metrics.f1_score.contains_key(&0));
    assert!(metrics.f1_score.contains_key(&1));
}

#[test]
fn test_classification_metrics_five_classes() {
    // Test with 5 classes (like VANGA's price level targets)
    let pred = vec![0, 1, 2, 3, 4, 0, 1, 2, 3, 4];
    let tgt = vec![0, 1, 2, 3, 3, 1, 2, 3, 4, 0];
    let metrics = calculate_classification_metrics(&pred, &tgt).unwrap();

    // Accuracy should be 4/10 = 0.4 (classes 0,1,2,3 correct, others wrong)
    assert_eq!(metrics.accuracy, 0.4);

    // Should have metrics for all 5 classes
    for class in 0..5 {
        assert!(metrics.precision.contains_key(&class));
        assert!(metrics.recall.contains_key(&class));
        assert!(metrics.f1_score.contains_key(&class));
    }

    // Macro F1 and Weighted F1 should be valid
    assert!(metrics.macro_f1 >= 0.0 && metrics.macro_f1 <= 1.0);
    assert!(metrics.weighted_f1 >= 0.0 && metrics.weighted_f1 <= 1.0);
}

#[test]
fn test_regression_metrics() {
    let pred = vec![1.0, 2.0, 3.0];
    let tgt = vec![1.1, 1.9, 3.2];
    let metrics = calculate_regression_metrics(&pred, &tgt).unwrap();
    assert!(metrics.rmse > 0.0);
}
