// Tests for Mixture-of-Head Attention (MoH) implementation
use crate::config::model::{AttentionConfig, AttentionMechanism, MoHConfig};
use crate::model::attention_moh::MixtureOfHeadAttention;
use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};

#[test]
fn test_moh_config_validation() {
    // Valid configuration
    let valid_config = MoHConfig {
        total_heads: 16,
        shared_heads: 4,
        top_k: 4,
        load_balance_weight: 0.01,
        routing_temperature: 1.0,
        log_routing_decisions: false,
    };
    assert!(valid_config.validate().is_ok());

    // Invalid: shared_heads + top_k > total_heads
    let invalid_config = MoHConfig {
        total_heads: 8,
        shared_heads: 6,
        top_k: 4,
        ..valid_config.clone()
    };
    assert!(invalid_config.validate().is_err());

    // Invalid: no active heads
    let no_active_config = MoHConfig {
        total_heads: 8,
        shared_heads: 0,
        top_k: 0,
        ..valid_config.clone()
    };
    assert!(no_active_config.validate().is_err());

    // Invalid: negative load balance weight
    let negative_weight_config = MoHConfig {
        load_balance_weight: -0.1,
        ..valid_config.clone()
    };
    assert!(negative_weight_config.validate().is_err());

    // Invalid: zero routing temperature
    let zero_temp_config = MoHConfig {
        routing_temperature: 0.0,
        ..valid_config.clone()
    };
    assert!(zero_temp_config.validate().is_err());
}

#[test]
fn test_moh_config_calculations() {
    let config = MoHConfig {
        total_heads: 16,
        shared_heads: 4,
        top_k: 6,
        ..MoHConfig::default()
    };

    assert_eq!(config.active_heads(), 10); // 4 + 6
    assert_eq!(config.inactive_heads(), 6); // 16 - 10
    assert_eq!(config.efficiency_ratio(), 0.625); // 10/16
}

#[test]
fn test_moh_config_defaults() {
    let config = MoHConfig::default();

    assert_eq!(config.total_heads, 16);
    assert_eq!(config.shared_heads, 4);
    assert_eq!(config.top_k, 4);
    assert_eq!(config.load_balance_weight, 0.01);
    assert_eq!(config.routing_temperature, 1.0);
    assert!(!config.log_routing_decisions);

    // Should be valid
    assert!(config.validate().is_ok());

    // Should have 50% efficiency (8 active out of 16 total)
    assert_eq!(config.efficiency_ratio(), 0.5);
}

#[tokio::test]
async fn test_moh_attention_creation() {
    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vs = VarBuilder::from_varmap(&varmap, DType::F32, &device);

    let moh_config = MoHConfig::default();
    let attention_config = AttentionConfig {
        mechanism: AttentionMechanism::MixtureOfHeads,
        moh: Some(moh_config),
        ..AttentionConfig::default()
    };

    let attention = MixtureOfHeadAttention::new(64, attention_config, vs, device);
    assert!(attention.is_ok());

    let attention = attention.unwrap();
    assert_eq!(
        attention.get_config().mechanism,
        AttentionMechanism::MixtureOfHeads
    );
}

#[tokio::test]
async fn test_moh_head_dimension_optimization() {
    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vs = VarBuilder::from_varmap(&varmap, DType::F32, &device);

    // Test different input dimensions
    let test_cases = vec![
        (10, 16),  // Small input
        (50, 16),  // Medium input
        (150, 16), // Large input
    ];

    for (input_dim, total_heads) in test_cases {
        let moh_config = MoHConfig {
            total_heads,
            ..MoHConfig::default()
        };

        let attention_config = AttentionConfig {
            mechanism: AttentionMechanism::MixtureOfHeads,
            head_dim: None, // Auto-optimize
            moh: Some(moh_config),
            ..AttentionConfig::default()
        };

        let attention = MixtureOfHeadAttention::new(
            input_dim,
            attention_config,
            vs.pp(format!("test_{}", input_dim)),
            device.clone(),
        );
        assert!(attention.is_ok(), "Failed for input_dim={}", input_dim);

        // Verify the attention was created successfully
        let _attention = attention.unwrap();
        // Note: We can't directly access head_dim from the struct, but creation success indicates proper optimization
    }
}

#[tokio::test]
async fn test_moh_forward_pass() {
    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vs = VarBuilder::from_varmap(&varmap, DType::F32, &device);

    let moh_config = MoHConfig {
        total_heads: 8,
        shared_heads: 2,
        top_k: 2,
        log_routing_decisions: true,
        ..MoHConfig::default()
    };

    let attention_config = AttentionConfig {
        mechanism: AttentionMechanism::MixtureOfHeads,
        moh: Some(moh_config),
        ..AttentionConfig::default()
    };

    let mut attention =
        MixtureOfHeadAttention::new(64, attention_config, vs, device.clone()).unwrap();

    // Create test input: [batch_size=2, seq_len=10, input_dim=64]
    let input = Tensor::randn(0f32, 1f32, (2, 10, 64), &device).unwrap();

    // Forward pass - use direct method by being explicit about the call
    let result = MixtureOfHeadAttention::forward(&mut attention, &input, true);
    assert!(result.is_ok(), "Forward pass failed: {:?}", result.err());

    let (output, routing_scores) = result.unwrap();

    // Check output shape
    assert_eq!(output.dims(), &[2, 10, 64]);

    // Check routing scores shape: [batch=1, seq_len=10, num_heads=8]
    assert_eq!(routing_scores.dims(), &[1, 10, 8]);

    // Verify routing statistics
    let stats = attention.get_routing_stats();
    assert_eq!(stats["total_heads"], 8.0);
    assert_eq!(stats["shared_heads"], 2.0);
    assert_eq!(stats["top_k"], 2.0);
    assert_eq!(stats["efficiency_ratio"], 0.5); // (2+2)/8 = 0.5
}

#[tokio::test]
async fn test_moh_load_balance_loss() {
    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vs = VarBuilder::from_varmap(&varmap, DType::F32, &device);

    let moh_config = MoHConfig {
        total_heads: 8,
        shared_heads: 2,
        top_k: 2,
        load_balance_weight: 0.05,
        ..MoHConfig::default()
    };

    let attention_config = AttentionConfig {
        mechanism: AttentionMechanism::MixtureOfHeads,
        moh: Some(moh_config),
        ..AttentionConfig::default()
    };

    let mut attention =
        MixtureOfHeadAttention::new(32, attention_config, vs, device.clone()).unwrap();

    // Run several forward passes to build routing history
    let input = Tensor::randn(0f32, 1f32, (1, 5, 32), &device).unwrap();

    for _ in 0..10 {
        let _ = MixtureOfHeadAttention::forward(&mut attention, &input, true).unwrap();
    }

    // Calculate load balance loss
    let load_balance_loss = attention.calculate_load_balance_loss();
    assert!(load_balance_loss.is_ok());

    let loss_tensor = load_balance_loss.unwrap();
    let loss_value = loss_tensor.to_scalar::<f32>().unwrap();

    // Loss should be non-negative
    assert!(loss_value >= 0.0);

    // Verify routing history was recorded
    let stats = attention.get_routing_stats();
    assert!(stats["routing_history_length"] > 0.0);
}

#[tokio::test]
async fn test_moh_routing_efficiency() {
    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vs = VarBuilder::from_varmap(&varmap, DType::F32, &device);

    // Test different efficiency ratios
    let test_configs = vec![
        (16, 4, 4, 0.5),   // 50% efficiency
        (16, 2, 6, 0.5),   // 50% efficiency
        (12, 3, 3, 0.5),   // 50% efficiency
        (20, 5, 10, 0.75), // 75% efficiency
    ];

    for (total_heads, shared_heads, top_k, expected_efficiency) in test_configs {
        let moh_config = MoHConfig {
            total_heads,
            shared_heads,
            top_k,
            ..MoHConfig::default()
        };

        // Validate the config first
        assert!(
            moh_config.validate().is_ok(),
            "Config validation failed for {}/{}/{}",
            total_heads,
            shared_heads,
            top_k
        );
        assert_eq!(moh_config.efficiency_ratio(), expected_efficiency);

        let attention_config = AttentionConfig {
            mechanism: AttentionMechanism::MixtureOfHeads,
            moh: Some(moh_config),
            ..AttentionConfig::default()
        };

        let attention = MixtureOfHeadAttention::new(
            64,
            attention_config,
            vs.pp(format!(
                "eff_test_{}_{}_{}_{}",
                total_heads, shared_heads, top_k, expected_efficiency
            )),
            device.clone(),
        );
        if let Err(e) = &attention {
            println!(
                "Error for config {}/{}/{}: {:?}",
                total_heads, shared_heads, top_k, e
            );
        }
        assert!(
            attention.is_ok(),
            "Failed for config: {}/{}/{}",
            total_heads,
            shared_heads,
            top_k
        );
    }
}

#[test]
fn test_moh_top_k_selection() {
    // This tests the internal top-K selection logic
    // We'll create a simple version to test the algorithm

    fn select_top_k(scores: &[f32], k: usize) -> Vec<f32> {
        let mut indexed_scores: Vec<(usize, f32)> =
            scores.iter().enumerate().map(|(i, &s)| (i, s)).collect();
        indexed_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut result = vec![0.0; scores.len()];
        for (idx, score) in indexed_scores.iter().take(k.min(scores.len())) {
            result[*idx] = *score;
        }

        result
    }

    let scores = vec![0.1, 0.8, 0.3, 0.9, 0.2];
    let top_2 = select_top_k(&scores, 2);

    // Should select indices 3 (0.9) and 1 (0.8)
    assert_eq!(top_2, vec![0.0, 0.8, 0.0, 0.9, 0.0]);

    let top_3 = select_top_k(&scores, 3);
    // Should select indices 3 (0.9), 1 (0.8), and 2 (0.3)
    assert_eq!(top_3, vec![0.0, 0.8, 0.3, 0.9, 0.0]);
}

#[tokio::test]
async fn test_moh_memory_management() {
    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vs = VarBuilder::from_varmap(&varmap, DType::F32, &device);

    let moh_config = MoHConfig::default();
    let attention_config = AttentionConfig {
        mechanism: AttentionMechanism::MixtureOfHeads,
        moh: Some(moh_config),
        ..AttentionConfig::default()
    };

    let mut attention =
        MixtureOfHeadAttention::new(32, attention_config, vs, device.clone()).unwrap();
    let input = Tensor::randn(0f32, 1f32, (1, 5, 32), &device).unwrap();

    // Run many forward passes to test memory management
    for i in 0..1500 {
        let _ = MixtureOfHeadAttention::forward(&mut attention, &input, true).unwrap();

        // Check that routing history doesn't grow indefinitely
        let stats = attention.get_routing_stats();
        let history_length = stats["routing_history_length"];

        if i > 1000 {
            // Should be capped at 1000
            assert!(
                history_length <= 1000.0,
                "History length {} exceeds limit at iteration {}",
                history_length,
                i
            );
        }
    }

    // Test manual clearing
    attention.clear_routing_history();
    let stats = attention.get_routing_stats();
    assert_eq!(stats["routing_history_length"], 0.0);
}

#[tokio::test]
async fn test_moh_vs_standard_attention_compatibility() {
    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vs = VarBuilder::from_varmap(&varmap, DType::F32, &device);

    // Test that MoH produces reasonable outputs compared to standard attention
    let input = Tensor::randn(0f32, 1f32, (2, 8, 64), &device).unwrap();

    // Create MoH attention
    let moh_config = MoHConfig {
        total_heads: 8,
        shared_heads: 8, // All heads shared = similar to standard attention
        top_k: 0,        // No routed heads
        ..MoHConfig::default()
    };

    let attention_config = AttentionConfig {
        mechanism: AttentionMechanism::MixtureOfHeads,
        moh: Some(moh_config),
        ..AttentionConfig::default()
    };

    let mut moh_attention =
        MixtureOfHeadAttention::new(64, attention_config, vs, device.clone()).unwrap();

    // Forward pass
    let (moh_output, _) =
        MixtureOfHeadAttention::forward(&mut moh_attention, &input, false).unwrap();

    // Check output properties
    assert_eq!(moh_output.dims(), input.dims());

    // Check if the tensor is not empty before calling sum_all
    let output_dims = moh_output.dims();
    let total_elements: usize = output_dims.iter().product();
    assert!(
        total_elements > 0,
        "MoH output tensor is empty: dims={:?}",
        output_dims
    );

    // Output should not be all zeros
    let output_sum = moh_output.sum_all().unwrap().to_scalar::<f32>().unwrap();
    assert!(
        output_sum.abs() > 1e-6,
        "MoH output appears to be all zeros: sum={}",
        output_sum
    );
}
