use super::*;

#[tokio::test]
async fn test_predictor_smoke() {
    let config = PredictionConfig::default();
    let model = LSTMModel::new(Default::default()).unwrap();
    let predictor = Predictor::new(config);
    let result = predictor.predict(&model).await;
    assert!(result.is_ok() || format!("{}", result.unwrap_err()).contains("data"));
}
