// Tests for attention_moh_wrapper module
use crate::config::model::{AttentionConfig, AttentionMechanism, MoHConfig};
use crate::model::attention_moh_wrapper::{
    EnhancedAttentionFactory, MoHAttentionWrapper, MoHTrainingLoss,
};
use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};

#[tokio::test]
async fn test_moh_wrapper_creation() {
    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vs = VarBuilder::from_varmap(&varmap, DType::F32, &device);

    let moh_config = MoHConfig::default();
    let attention_config = AttentionConfig {
        mechanism: AttentionMechanism::MixtureOfHeads,
        moh: Some(moh_config),
        ..AttentionConfig::default()
    };

    let wrapper = MoHAttentionWrapper::new(64, attention_config, vs, device);
    assert!(wrapper.is_ok());
}

#[tokio::test]
async fn test_moh_training_loss() {
    let device = Device::Cpu;
    let task_loss = Tensor::new(1.5f32, &device).unwrap();

    // Test without MoH
    let loss_without_moh = MoHTrainingLoss::new(task_loss.clone(), None, 0.01).unwrap();
    assert_eq!(loss_without_moh.task_loss_value().unwrap(), 1.5);
    assert!(loss_without_moh
        .load_balance_loss_value()
        .unwrap()
        .is_none());
    assert_eq!(loss_without_moh.total_loss_value().unwrap(), 1.5);
}

#[tokio::test]
async fn test_enhanced_attention_factory() {
    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vs = VarBuilder::from_varmap(&varmap, DType::F32, &device);

    let moh_config = MoHConfig::default();
    let attention_config = AttentionConfig {
        mechanism: AttentionMechanism::MixtureOfHeads,
        moh: Some(moh_config),
        ..AttentionConfig::default()
    };

    let attention = EnhancedAttentionFactory::create_attention(
        &AttentionMechanism::MixtureOfHeads,
        64,
        attention_config,
        vs,
        device,
    );

    assert!(attention.is_ok());
}
