//! Tests for loss calculation module

use super::loss::*;
use candle_core::{Device, Tensor};
use candle_nn;

#[test]
fn test_ordinal_loss_basic() {
    // Test that ordinal loss works and produces valid gradients
    let device = Device::Cpu;

    // Create simple predictions (logits) and targets
    let batch_size = 4;
    let num_classes = 5;

    // Create predictions as logits (before softmax)
    // Shape: [batch_size, num_classes]
    let predictions = Tensor::randn(0.0f32, 1.0f32, (batch_size, num_classes), &device)
        .expect("Failed to create predictions");

    // Create targets as class indices
    // Classes: 0, 1, 3, 4 (testing various ordinal positions)
    let targets = Tensor::from_vec(vec![0.0f32, 1.0, 3.0, 4.0], (batch_size, 1), &device)
        .expect("Failed to create targets");

    // Create loss calculator
    let config = crate::model::lstm::config::LSTMConfig {
        input_size: 10,
        hidden_size: 20,
        output_size: num_classes,
        num_layers: 1,
        dropout: 0.0,
        bidirectional: false,
        attention_heads: None,
    };

    let loss_calc = LossCalculator::new(config);

    // Calculate ordinal loss
    let training_config = crate::config::TrainingConfig::default();
    let loss = loss_calc.calculate_single_target_loss(
        &predictions,
        &targets,
        crate::targets::TargetType::PriceLevels,
        &training_config,
        false,
    );

    // Verify loss is computed successfully
    assert!(
        loss.is_ok(),
        "Ordinal loss calculation failed: {:?}",
        loss.err()
    );

    let loss_tensor = loss.unwrap();

    // Verify loss is a scalar
    assert_eq!(loss_tensor.dims(), &[], "Loss should be a scalar");

    // Verify loss value is reasonable (not NaN or infinite)
    let loss_value = loss_tensor
        .to_scalar::<f32>()
        .expect("Failed to get scalar");
    assert!(!loss_value.is_nan(), "Loss is NaN");
    assert!(!loss_value.is_infinite(), "Loss is infinite");
    assert!(loss_value >= 0.0, "Loss should be non-negative");

    // Typical ordinal loss range for random predictions should be around 0.5-2.0
    assert!(
        loss_value < 10.0,
        "Loss is unreasonably high: {}",
        loss_value
    );
}

#[test]
fn test_ordinal_loss_gradient_flow() {
    // Test that gradients flow properly through the ordinal loss
    let device = Device::Cpu;

    let batch_size = 8;
    let num_classes = 5;

    // Create predictions with requires_grad for gradient tracking
    let predictions = candle_core::Var::new(
        Tensor::randn(0.0f32, 1.0f32, (batch_size, num_classes), &device).unwrap(),
        &device,
    )
    .unwrap();

    // Create diverse targets to test all ordinal thresholds
    let targets = Tensor::from_vec(
        vec![0.0f32, 0.0, 1.0, 2.0, 2.0, 3.0, 4.0, 4.0],
        (batch_size, 1),
        &device,
    )
    .expect("Failed to create targets");

    let config = crate::model::lstm::config::LSTMConfig {
        input_size: 10,
        hidden_size: 20,
        output_size: num_classes,
        num_layers: 1,
        dropout: 0.0,
        bidirectional: false,
        attention_heads: None,
    };

    let loss_calc = LossCalculator::new(config);
    let training_config = crate::config::TrainingConfig::default();

    // Calculate loss
    let loss = loss_calc
        .calculate_single_target_loss(
            &predictions,
            &targets,
            crate::targets::TargetType::Direction,
            &training_config,
            false,
        )
        .expect("Loss calculation failed");

    // Compute gradients
    let grads = loss.backward().expect("Backward pass failed");

    // Verify gradients exist and are valid
    let pred_grad = grads
        .get(&predictions)
        .expect("No gradient for predictions");

    // Check gradient shape matches predictions
    assert_eq!(
        pred_grad.dims(),
        predictions.dims(),
        "Gradient shape mismatch"
    );

    // Verify gradients are not all zeros (model should be learning)
    let grad_sum = pred_grad.sum_all().unwrap().to_scalar::<f32>().unwrap();
    assert!(grad_sum.abs() > 1e-6, "Gradients are too small or zero");

    // Verify gradients are reasonable (not exploding)
    assert!(
        grad_sum.abs() < 1000.0,
        "Gradients are exploding: {}",
        grad_sum
    );
}

#[test]
fn test_ordinal_loss_perfect_predictions() {
    // Test that perfect predictions yield low loss
    let device = Device::Cpu;

    let batch_size = 4;
    let num_classes = 5;

    // Create perfect predictions (high confidence for correct class)
    // For class 2: logits should be low for 0,1 and high for 2,3,4
    let perfect_logits = vec![
        // Target class 0: high confidence for class 0
        vec![10.0, -5.0, -5.0, -5.0, -5.0],
        // Target class 2: high confidence for class 2
        vec![-5.0, -5.0, 10.0, -5.0, -5.0],
        // Target class 4: high confidence for class 4
        vec![-5.0, -5.0, -5.0, -5.0, 10.0],
        // Target class 1: high confidence for class 1
        vec![-5.0, 10.0, -5.0, -5.0, -5.0],
    ];

    let flat_logits: Vec<f32> = perfect_logits.into_iter().flatten().collect();
    let predictions = Tensor::from_vec(flat_logits, (batch_size, num_classes), &device)
        .expect("Failed to create predictions");

    let targets = Tensor::from_vec(vec![0.0f32, 2.0, 4.0, 1.0], (batch_size, 1), &device)
        .expect("Failed to create targets");

    let config = crate::model::lstm::config::LSTMConfig {
        input_size: 10,
        hidden_size: 20,
        output_size: num_classes,
        num_layers: 1,
        dropout: 0.0,
        bidirectional: false,
        attention_heads: None,
    };

    let loss_calc = LossCalculator::new(config);
    let training_config = crate::config::TrainingConfig::default();

    let loss = loss_calc
        .calculate_single_target_loss(
            &predictions,
            &targets,
            crate::targets::TargetType::Volatility,
            &training_config,
            false,
        )
        .expect("Loss calculation failed");

    let loss_value = loss.to_scalar::<f32>().expect("Failed to get scalar");

    // Perfect predictions should have very low loss (close to 0)
    assert!(
        loss_value < 0.1,
        "Perfect predictions should have low loss, got: {}",
        loss_value
    );
}

#[test]
fn test_ordinal_loss_worst_predictions() {
    // Test that completely wrong predictions yield high loss
    let device = Device::Cpu;

    let batch_size = 2;
    let num_classes = 5;

    // Create worst predictions (opposite of targets)
    // Target is class 0, predict class 4 with high confidence
    // Target is class 4, predict class 0 with high confidence
    let worst_logits = vec![
        vec![-10.0, -5.0, -5.0, -5.0, 10.0], // Target 0, predict 4
        vec![10.0, -5.0, -5.0, -5.0, -10.0], // Target 4, predict 0
    ];

    let flat_logits: Vec<f32> = worst_logits.into_iter().flatten().collect();
    let predictions = Tensor::from_vec(flat_logits, (batch_size, num_classes), &device)
        .expect("Failed to create predictions");

    let targets = Tensor::from_vec(vec![0.0f32, 4.0], (batch_size, 1), &device)
        .expect("Failed to create targets");

    let config = crate::model::lstm::config::LSTMConfig {
        input_size: 10,
        hidden_size: 20,
        output_size: num_classes,
        num_layers: 1,
        dropout: 0.0,
        bidirectional: false,
        attention_heads: None,
    };

    let loss_calc = LossCalculator::new(config);
    let training_config = crate::config::TrainingConfig::default();

    let loss = loss_calc
        .calculate_single_target_loss(
            &predictions,
            &targets,
            crate::targets::TargetType::Volume,
            &training_config,
            false,
        )
        .expect("Loss calculation failed");

    let loss_value = loss.to_scalar::<f32>().expect("Failed to get scalar");

    // Worst predictions should have high loss
    assert!(
        loss_value > 1.0,
        "Worst predictions should have high loss, got: {}",
        loss_value
    );
}

#[test]
fn test_ordinal_loss_ordering_property() {
    // Test that ordinal loss respects trading logic:
    // Wrong direction (across middle point 2) should have higher loss than wrong magnitude
    let device = Device::Cpu;

    let batch_size = 1;
    let num_classes = 5;

    // Test Case 1: Target is class 0 (VeryDown)
    let targets_down =
        Tensor::from_vec(vec![0.0f32], (batch_size, 1), &device).expect("Failed to create targets");

    let config = crate::model::lstm::config::LSTMConfig {
        input_size: 10,
        hidden_size: 20,
        output_size: num_classes,
        num_layers: 1,
        dropout: 0.0,
        bidirectional: false,
        attention_heads: None,
    };

    let loss_calc = LossCalculator::new(config);
    let training_config = crate::config::TrainingConfig::default();

    // Prediction 1: Predict class 1 (Down - same direction, small error)
    let same_direction_pred = Tensor::from_vec(
        vec![-5.0f32, 10.0, -5.0, -5.0, -5.0], // High confidence for class 1
        (batch_size, num_classes),
        &device,
    )
    .expect("Failed to create same direction prediction");

    let same_dir_loss = loss_calc
        .calculate_single_target_loss(
            &same_direction_pred,
            &targets_down,
            crate::targets::TargetType::PriceLevels,
            &training_config,
            false,
        )
        .expect("Loss calculation failed");

    let same_dir_loss_value = same_dir_loss.to_scalar::<f32>().unwrap();

    // Prediction 2: Predict class 4 (VeryUp - opposite direction, worst error)
    let opposite_direction_pred = Tensor::from_vec(
        vec![-5.0f32, -5.0, -5.0, -5.0, 10.0], // High confidence for class 4
        (batch_size, num_classes),
        &device,
    )
    .expect("Failed to create opposite direction prediction");

    let opposite_dir_loss = loss_calc
        .calculate_single_target_loss(
            &opposite_direction_pred,
            &targets_down,
            crate::targets::TargetType::PriceLevels,
            &training_config,
            false,
        )
        .expect("Loss calculation failed");

    let opposite_dir_loss_value = opposite_dir_loss.to_scalar::<f32>().unwrap();

    // Trading logic: predicting UP when market goes DOWN is much worse
    assert!(
        same_dir_loss_value < opposite_dir_loss_value,
        "Trading property violated: same direction loss {} >= opposite direction loss {}",
        same_dir_loss_value,
        opposite_dir_loss_value
    );

    // Test Case 2: Target is class 2 (Sideways - middle point)
    let targets_middle = Tensor::from_vec(vec![2.0f32], (batch_size, 1), &device)
        .expect("Failed to create middle targets");

    // Prediction 3: Predict class 1 (mild deviation)
    let mild_pred = Tensor::from_vec(
        vec![-5.0f32, 10.0, -5.0, -5.0, -5.0], // High confidence for class 1
        (batch_size, num_classes),
        &device,
    )
    .expect("Failed to create mild prediction");

    let mild_loss = loss_calc
        .calculate_single_target_loss(
            &mild_pred,
            &targets_middle,
            crate::targets::TargetType::PriceLevels,
            &training_config,
            false,
        )
        .expect("Loss calculation failed");

    let mild_loss_value = mild_loss.to_scalar::<f32>().unwrap();

    // Prediction 4: Predict class 0 (stronger deviation)
    let strong_pred = Tensor::from_vec(
        vec![10.0f32, -5.0, -5.0, -5.0, -5.0], // High confidence for class 0
        (batch_size, num_classes),
        &device,
    )
    .expect("Failed to create strong prediction");

    let strong_loss = loss_calc
        .calculate_single_target_loss(
            &strong_pred,
            &targets_middle,
            crate::targets::TargetType::PriceLevels,
            &training_config,
            false,
        )
        .expect("Loss calculation failed");

    let strong_loss_value = strong_loss.to_scalar::<f32>().unwrap();

    // From middle point, stronger deviations should have higher loss
    assert!(
        mild_loss_value < strong_loss_value,
        "Middle point property violated: mild loss {} >= strong loss {}",
        mild_loss_value,
        strong_loss_value
    );
}
