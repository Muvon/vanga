use crate::model::lstm::*;
use ndarray::{Array2, Array3};

#[test]
fn test_convert_sequences_to_tensors_one_hot() {
    // Create a simple LSTM model
    let config = LSTMConfig {
        input_size: 10,
        hidden_sizes: vec![32],
        output_size: 5,
        sequence_length: 20,
        learning_rate: 0.001,
        num_layers: 1,
    };

    let model = LSTMModel::new(config).unwrap();

    // Create test data with one-hot encoded targets
    let batch_size = 4;
    let seq_len = 20;
    let features = 10;

    // Create dummy sequences
    let sequences = Array3::<f64>::zeros((batch_size, seq_len, features));

    // Create one-hot encoded targets for 5 classes
    // Each sample has a different class: [0, 1, 2, 3]
    let mut targets = Array2::<f64>::zeros((batch_size, 5));
    targets[[0, 0]] = 1.0; // Class 0
    targets[[1, 1]] = 1.0; // Class 1
    targets[[2, 2]] = 1.0; // Class 2
    targets[[3, 3]] = 1.0; // Class 3

    // Convert to tensors
    let (_, target_tensor) = model
        .convert_sequences_to_tensors(&sequences, &targets)
        .unwrap();

    // Verify the target tensor has correct shape
    assert_eq!(target_tensor.shape().dims(), &[batch_size, 1]);

    // Get the values and verify they are class indices
    let target_values = target_tensor.to_vec2::<f32>().unwrap();
    assert_eq!(target_values[0][0], 0.0); // Class 0
    assert_eq!(target_values[1][0], 1.0); // Class 1
    assert_eq!(target_values[2][0], 2.0); // Class 2
    assert_eq!(target_values[3][0], 3.0); // Class 3

    println!("✅ One-hot to class index conversion working correctly!");
}

#[test]
fn test_convert_sequences_to_tensors_raw_indices() {
    // Create a simple LSTM model
    let config = LSTMConfig {
        input_size: 10,
        hidden_sizes: vec![32],
        output_size: 5,
        sequence_length: 20,
        learning_rate: 0.001,
        num_layers: 1,
    };

    let model = LSTMModel::new(config).unwrap();

    // Create test data with raw class indices
    let batch_size = 4;
    let seq_len = 20;
    let features = 10;

    // Create dummy sequences
    let sequences = Array3::<f64>::zeros((batch_size, seq_len, features));

    // Create raw class indices
    let mut targets = Array2::<f64>::zeros((batch_size, 1));
    targets[[0, 0]] = 0.0; // Class 0
    targets[[1, 0]] = 2.0; // Class 2
    targets[[2, 0]] = 4.0; // Class 4
    targets[[3, 0]] = 1.0; // Class 1

    // Convert to tensors
    let (_, target_tensor) = model
        .convert_sequences_to_tensors(&sequences, &targets)
        .unwrap();

    // Verify the target tensor has correct shape
    assert_eq!(target_tensor.shape().dims(), &[batch_size, 1]);

    // Get the values and verify they match input
    let target_values = target_tensor.to_vec2::<f32>().unwrap();
    assert_eq!(target_values[0][0], 0.0); // Class 0
    assert_eq!(target_values[1][0], 2.0); // Class 2
    assert_eq!(target_values[2][0], 4.0); // Class 4
    assert_eq!(target_values[3][0], 1.0); // Class 1

    println!("✅ Raw class indices pass-through working correctly!");
}

#[test]
fn test_convert_sequences_to_tensors_edge_cases() {
    // Create a simple LSTM model
    let config = LSTMConfig {
        input_size: 10,
        hidden_sizes: vec![32],
        output_size: 5,
        sequence_length: 20,
        learning_rate: 0.001,
        num_layers: 1,
    };

    let model = LSTMModel::new(config).unwrap();

    // Test with label smoothing (values not exactly 0 or 1)
    let batch_size = 2;
    let seq_len = 20;
    let features = 10;

    let sequences = Array3::<f64>::zeros((batch_size, seq_len, features));

    // Create targets with label smoothing
    let mut targets = Array2::<f64>::zeros((batch_size, 5));
    // First sample: class 2 with label smoothing
    targets[[0, 0]] = 0.02; // Smoothed
    targets[[0, 1]] = 0.02; // Smoothed
    targets[[0, 2]] = 0.92; // Main class (highest value)
    targets[[0, 3]] = 0.02; // Smoothed
    targets[[0, 4]] = 0.02; // Smoothed

    // Second sample: class 4
    targets[[1, 0]] = 0.1;
    targets[[1, 1]] = 0.1;
    targets[[1, 2]] = 0.1;
    targets[[1, 3]] = 0.1;
    targets[[1, 4]] = 0.6; // Highest value

    // Convert to tensors
    let (_, target_tensor) = model
        .convert_sequences_to_tensors(&sequences, &targets)
        .unwrap();

    // Get the values and verify argmax worked correctly
    let target_values = target_tensor.to_vec2::<f32>().unwrap();
    assert_eq!(target_values[0][0], 2.0); // Class 2 (highest at 0.92)
    assert_eq!(target_values[1][0], 4.0); // Class 4 (highest at 0.6)

    println!("✅ Label smoothing and argmax handling working correctly!");
}
