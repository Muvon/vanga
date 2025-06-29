use super::*;

#[tokio::test]
async fn test_train_model_smoke() {
    let config = TrainingConfig::default();
    let result = train_model(config).await;
    assert!(result.is_ok() || format!("{}", result.unwrap_err()).contains("data"));
}
