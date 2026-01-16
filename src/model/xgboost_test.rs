// Tests for xgboost module
use crate::config::model::XGBoostConfig;
use crate::model::xgboost::{get_eval_metric_for_target, get_objective_for_target, XGBoostRegressor};
use candle_core::Device;

#[test]
fn test_xgboost_config_default() {
    let config = XGBoostConfig::default();
    assert!(!config.enabled);
    assert_eq!(config.feature_dim, 64);
    assert_eq!(config.n_estimators, 100);
    assert_eq!(config.max_depth, 6);
}

#[test]
fn test_objective_selection() {
    assert_eq!(
        get_objective_for_target("price_level_1h", 5),
        "multi:softprob"
    );
    assert_eq!(
        get_objective_for_target("direction_4h", 5),
        "multi:softprob"
    );
    assert_eq!(
        get_objective_for_target("volatility_1d", 1),
        "reg:squarederror"
    );
}

#[test]
fn test_eval_metric_selection() {
    assert_eq!(get_eval_metric_for_target("price_level_1h", 5), "mlogloss");
    assert_eq!(get_eval_metric_for_target("direction_4h", 5), "mlogloss");
    assert_eq!(get_eval_metric_for_target("volatility_1d", 1), "rmse");
}

#[tokio::test]
async fn test_xgboost_regressor_creation() {
    let config = XGBoostConfig::default();
    let device = Device::Cpu;
    let regressor = XGBoostRegressor::new(config, device);

    assert!(!regressor.is_trained());
    assert!(regressor.get_feature_importance().is_none());
}

#[test]
fn test_determine_num_classes() {
    let config = XGBoostConfig::default();
    let device = Device::Cpu;
    let regressor = XGBoostRegressor::new(config, device.clone());

    // Test 1D tensor (regression)
    let targets_1d = Tensor::zeros((10,), candle_core::DType::F32, &device).unwrap();
    assert_eq!(regressor.determine_num_classes(&targets_1d).unwrap(), 1);

    // Test 2D tensor with 1 column (regression/binary)
    let targets_2d_1 = Tensor::zeros((10, 1), candle_core::DType::F32, &device).unwrap();
    assert_eq!(regressor.determine_num_classes(&targets_2d_1).unwrap(), 1);

    // Test 2D tensor with multiple columns (multi-class)
    let targets_2d_5 = Tensor::zeros((10, 5), candle_core::DType::F32, &device).unwrap();
    assert_eq!(regressor.determine_num_classes(&targets_2d_5).unwrap(), 5);
}