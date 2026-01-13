use crate::model::calibration::ece::*;
use ndarray::Array2;

#[test]
fn test_calculate_ece_perfect_calibration() {
    // Perfect predictions: 100% confidence on correct class
    let predictions = Array2::from_shape_vec(
        (5, 5),
        vec![
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ],
    )
    .unwrap();

    let targets = predictions.clone();

    let ece = calculate_ece(&predictions, &targets).unwrap();

    // Perfect calibration should have ECE close to 0
    assert!(ece < 0.01, "ECE for perfect calibration: {}", ece);
}

#[test]
fn test_calculate_ece_empty_data() {
    let predictions = Array2::zeros((0, 5));
    let targets = Array2::zeros((0, 5));

    let ece = calculate_ece(&predictions, &targets).unwrap();

    assert_eq!(ece, 0.0);
}

#[test]
fn test_calculate_ece_mismatched_samples() {
    let predictions = Array2::zeros((10, 5));
    let targets = Array2::zeros((15, 5));

    let result = calculate_ece(&predictions, &targets);

    assert!(result.is_err());
}

#[test]
fn test_calculate_ece_overconfident() {
    // Overconfident predictions (high confidence, low accuracy)
    let mut predictions_data = Vec::new();
    let mut targets_data = Vec::new();

    for i in 0..20 {
        // Always predict class 0 with high confidence
        predictions_data.extend_from_slice(&[0.9, 0.025, 0.025, 0.025, 0.025]);

        // But actual class varies
        let actual_class = i % 5;
        let mut target_row = vec![0.0; 5];
        target_row[actual_class] = 1.0;
        targets_data.extend_from_slice(&target_row);
    }

    let predictions = Array2::from_shape_vec((20, 5), predictions_data).unwrap();
    let targets = Array2::from_shape_vec((20, 5), targets_data).unwrap();

    let ece = calculate_ece(&predictions, &targets).unwrap();

    // Overconfident predictions should have high ECE
    assert!(ece > 0.1, "ECE for overconfident predictions: {}", ece);
}

#[test]
fn test_calculate_per_class_ece() {
    let mut predictions_data = Vec::new();
    let mut targets_data = Vec::new();

    for i in 0..50 {
        let class = i % 5;

        // Different confidence per class
        let confidence = match class {
            0 => 0.9, // Very confident
            1 => 0.7, // Confident
            2 => 0.5, // Moderate
            3 => 0.3, // Low
            4 => 0.2, // Very low
            _ => 0.5,
        };

        let mut pred_row = vec![(1.0 - confidence) / 4.0; 5];
        pred_row[class] = confidence;
        predictions_data.extend_from_slice(&pred_row);

        let mut target_row = vec![0.0; 5];
        target_row[class] = 1.0;
        targets_data.extend_from_slice(&target_row);
    }

    let predictions = Array2::from_shape_vec((50, 5), predictions_data).unwrap();
    let targets = Array2::from_shape_vec((50, 5), targets_data).unwrap();

    let per_class_ece = calculate_per_class_ece(&predictions, &targets).unwrap();

    // All classes should have ECE values
    for &ece in &per_class_ece {
        assert!(ece >= 0.0);
    }
}

#[test]
fn test_calculate_per_class_ece_empty() {
    let predictions = Array2::zeros((0, 5));
    let targets = Array2::zeros((0, 5));

    let per_class_ece = calculate_per_class_ece(&predictions, &targets).unwrap();

    assert_eq!(per_class_ece, [0.0; 5]);
}

#[test]
fn test_calculate_per_class_ece_invalid_dimensions() {
    let predictions = Array2::zeros((10, 3));
    let targets = Array2::zeros((10, 5));

    let result = calculate_per_class_ece(&predictions, &targets);

    assert!(result.is_err());
}

#[test]
fn test_generate_reliability_diagram() {
    let mut predictions_data = Vec::new();
    let mut targets_data = Vec::new();

    for i in 0..100 {
        let class = i % 5;

        let confidence = 0.5 + (i as f64 / 200.0);
        let mut pred_row = vec![(1.0 - confidence) / 4.0; 5];
        pred_row[class] = confidence;
        predictions_data.extend_from_slice(&pred_row);

        let mut target_row = vec![0.0; 5];
        target_row[class] = 1.0;
        targets_data.extend_from_slice(&target_row);
    }

    let predictions = Array2::from_shape_vec((100, 5), predictions_data).unwrap();
    let targets = Array2::from_shape_vec((100, 5), targets_data).unwrap();

    let diagram = generate_reliability_diagram(&predictions, &targets).unwrap();

    // Check diagram structure
    assert_eq!(diagram.bin_boundaries.len(), 16); // NUM_BINS + 1
    assert_eq!(diagram.bin_confidences.len(), 15); // NUM_BINS
    assert_eq!(diagram.bin_accuracies.len(), 15);
    assert_eq!(diagram.bin_counts.len(), 15);
    assert!(diagram.ece >= 0.0);
}

#[test]
fn test_generate_reliability_diagram_empty() {
    let predictions = Array2::zeros((0, 5));
    let targets = Array2::zeros((0, 5));

    let diagram = generate_reliability_diagram(&predictions, &targets).unwrap();

    assert!(diagram.bin_boundaries.is_empty());
    assert!(diagram.bin_confidences.is_empty());
    assert!(diagram.bin_accuracies.is_empty());
    assert!(diagram.bin_counts.is_empty());
    assert_eq!(diagram.ece, 0.0);
}

#[test]
fn test_ece_bins_coverage() {
    // Create predictions spanning all confidence ranges
    let mut predictions_data = Vec::new();
    let mut targets_data = Vec::new();

    for i in 0..150 {
        let confidence = (i as f64 / 150.0).min(0.99);
        let class = i % 5;

        let mut pred_row = vec![(1.0 - confidence) / 4.0; 5];
        pred_row[class] = confidence;
        predictions_data.extend_from_slice(&pred_row);

        let mut target_row = vec![0.0; 5];
        target_row[class] = 1.0;
        targets_data.extend_from_slice(&target_row);
    }

    let predictions = Array2::from_shape_vec((150, 5), predictions_data).unwrap();
    let targets = Array2::from_shape_vec((150, 5), targets_data).unwrap();

    let diagram = generate_reliability_diagram(&predictions, &targets).unwrap();

    // Most bins should have samples
    let non_empty_bins = diagram.bin_counts.iter().filter(|&&c| c > 0).count();
    assert!(
        non_empty_bins > 10,
        "Only {} bins have samples",
        non_empty_bins
    );
}

#[test]
fn test_ece_monotonicity() {
    // Test that ECE increases with worse calibration

    // Well-calibrated predictions
    let good_predictions = Array2::from_shape_vec(
        (20, 5),
        (0..20)
            .flat_map(|_| vec![0.6, 0.1, 0.1, 0.1, 0.1])
            .collect::<Vec<_>>(),
    )
    .unwrap();

    // Poorly calibrated (overconfident)
    let bad_predictions = Array2::from_shape_vec(
        (20, 5),
        (0..20)
            .flat_map(|_| vec![0.95, 0.0125, 0.0125, 0.0125, 0.0125])
            .collect::<Vec<_>>(),
    )
    .unwrap();

    // Targets: only 60% are actually class 0
    let mut targets_data = Vec::new();
    for i in 0..20 {
        let actual_class = if i < 12 { 0 } else { i % 5 };
        let mut target_row = vec![0.0; 5];
        target_row[actual_class] = 1.0;
        targets_data.extend_from_slice(&target_row);
    }
    let targets = Array2::from_shape_vec((20, 5), targets_data).unwrap();

    let ece_good = calculate_ece(&good_predictions, &targets).unwrap();
    let ece_bad = calculate_ece(&bad_predictions, &targets).unwrap();

    // Overconfident predictions should have higher ECE
    assert!(
        ece_bad > ece_good,
        "Bad ECE: {}, Good ECE: {}",
        ece_bad,
        ece_good
    );
}
