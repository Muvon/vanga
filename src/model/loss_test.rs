use super::*;
use ndarray::Array2;

#[test]
fn test_calculate_loss_mse() {
    let pred = Array2::<f64>::from_elem((2, 2), 1.0);
    let tgt = Array2::<f64>::from_elem((2, 2), 0.0);
    let lf = LossFunctions::default();
    let loss = lf.calculate_loss(&pred, &tgt, LossType::MSE).unwrap();
    assert!(loss > 0.9 && loss < 1.1);
}
