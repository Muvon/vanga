use super::*;
use ndarray::{Array2, Array3};

#[tokio::test]
async fn test_lstm_train_predict_shape() {
    let config = LSTMConfig::default();
    let mut model = LSTMModel::new(config).unwrap();
    let sequences = Array3::<f64>::zeros((2, 10, 5));
    let targets = Array2::<f64>::zeros((2, 1));
    model.train(&sequences, &targets).await.unwrap();
    let preds = model.predict(&sequences).await.unwrap();
    assert_eq!(preds.shape(), &[2, 1]);
}
