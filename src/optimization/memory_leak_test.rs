//! Memory leak tests for fractional optimizers and attention mechanisms

use crate::optimization::frac_adam::{FracAdam, ParamsFracAdam};
use crate::optimization::frac_nadam::{FracNAdam, ParamsFracNAdam};
use crate::optimization::fractional::FractionalConfig;
use candle_core::{Device, Tensor, Var};
use candle_nn::optim::Optimizer;

#[test]
fn test_frac_adam_memory_management() {
    let device = Device::Cpu;

    // Create test variables
    let var1 = Var::new(&[2.0, 3.0, 4.0], &device).unwrap();
    let var2 = Var::new(&[1.0, 2.0, 3.0], &device).unwrap();
    let vars = vec![var1.clone(), var2.clone()];

    // Create optimizer with small memory window
    let params = ParamsFracAdam {
        lr: 0.001,
        beta_1: 0.9,
        beta_2: 0.999,
        eps: 1e-8,
        weight_decay: None,
        fractional: FractionalConfig {
            alpha: 0.5,
            memory_window: 5, // Small window for testing
            step_size: 1.0,
        },
    };

    let mut optimizer = FracAdam::new(vars, params).unwrap();

    // Simulate multiple training steps
    for i in 0..20 {
        // Create dummy gradients
        let grad1 =
            Tensor::new(&[0.1 * i as f32, 0.2 * i as f32, 0.3 * i as f32], &device).unwrap();
        let grad2 =
            Tensor::new(&[0.05 * i as f32, 0.1 * i as f32, 0.15 * i as f32], &device).unwrap();

        // Create gradient store
        let mut grads = candle_core::backprop::GradStore::new();
        grads.insert(&var1, grad1);
        grads.insert(&var2, grad2);

        // Step optimizer
        optimizer.step(&grads).unwrap();

        // Check memory usage periodically
        if i % 5 == 0 {
            let memory_usage = optimizer.memory_usage();
            // Memory usage should be bounded by window size + moments
            assert!(
                memory_usage <= 20,
                "Memory usage too high: {}",
                memory_usage
            );
        }
    }

    // Test memory cleanup
    let initial_memory = optimizer.memory_usage();
    optimizer.compact_memory();
    let compacted_memory = optimizer.memory_usage();
    assert!(
        compacted_memory <= initial_memory,
        "Compact should not increase memory"
    );

    // Test state clearing
    optimizer.clear_state();
    let cleared_memory = optimizer.memory_usage();
    assert_eq!(cleared_memory, 0, "Clear state should free all memory");
}

#[test]
fn test_frac_nadam_memory_management() {
    let device = Device::Cpu;

    // Create test variables
    let var = Var::new(&[1.0, 2.0, 3.0, 4.0], &device).unwrap();
    let vars = vec![var.clone()];

    // Create optimizer
    let params = ParamsFracNAdam {
        lr: 0.002,
        beta_1: 0.9,
        beta_2: 0.999,
        eps: 1e-8,
        weight_decay: None,
        momentum_decay: 0.004,
        fractional: FractionalConfig {
            alpha: 0.7,
            memory_window: 10,
            step_size: 1.0,
        },
    };

    let mut optimizer = FracNAdam::new(vars, params).unwrap();

    // Simulate training with memory checks
    for i in 0..30 {
        let grad = Tensor::new(
            &[
                0.1 * i as f32,
                0.2 * i as f32,
                0.3 * i as f32,
                0.4 * i as f32,
            ],
            &device,
        )
        .unwrap();

        let mut grads = candle_core::backprop::GradStore::new();
        grads.insert(&var, grad);

        optimizer.step(&grads).unwrap();

        // Memory should be bounded
        let memory_usage = optimizer.memory_usage();
        assert!(
            memory_usage <= 25,
            "Memory leak detected at step {}: usage = {}",
            i,
            memory_usage
        );
    }

    // Verify cleanup works
    optimizer.clear_state();
    assert_eq!(optimizer.memory_usage(), 0, "Memory not properly cleared");
}

#[test]
fn test_fractional_derivative_memory_bounds() {
    use crate::optimization::fractional::FractionalDerivative;

    let device = Device::Cpu;
    let num_params = 3;
    let memory_window = 10;

    let mut frac_deriv = FractionalDerivative::new(0.5, memory_window, 1.0, num_params).unwrap();

    // Create test gradients
    let gradients: Vec<Tensor> = (0..num_params)
        .map(|_| Tensor::new(&[1.0, 2.0, 3.0], &device).unwrap())
        .collect();

    // Update history many times
    for _ in 0..50 {
        frac_deriv.update_history(&gradients).unwrap();
    }

    // Check that history is bounded
    for i in 0..num_params {
        let history_len = frac_deriv.history_length(i);
        assert!(
            history_len <= memory_window,
            "History exceeded window: {} > {}",
            history_len,
            memory_window
        );
    }

    // Test memory compaction
    let initial_usage = frac_deriv.memory_usage();
    frac_deriv.compact_history();
    let compacted_usage = frac_deriv.memory_usage();
    assert!(compacted_usage <= initial_usage, "Compaction failed");

    // Test clear
    frac_deriv.clear_history();
    assert_eq!(frac_deriv.memory_usage(), 0, "History not cleared");
}

#[test]
fn test_memory_manager() {
    use crate::utils::memory_manager::MemoryManager;

    let manager = MemoryManager::new(8192, 10, false);

    // Test step counting
    for i in 0..25 {
        manager.increment_step();

        if i % 10 == 9 {
            assert!(manager.should_cleanup(), "Should cleanup at step {}", i + 1);
        }
    }

    assert_eq!(manager.get_step_count(), 25);

    // Test memory limit getter
    assert_eq!(manager.max_memory_mb(), 8192);
}

#[test]
fn test_adaptive_batch_size() {
    use crate::utils::memory_manager::AdaptiveBatchSize;

    let mut adaptive = AdaptiveBatchSize::new(32, 8, 128);

    // Initial size
    assert_eq!(adaptive.get_batch_size(), 32);

    // High memory should reduce
    adaptive.adjust(90.0);
    assert!(adaptive.get_batch_size() < 32);

    // Low memory should increase
    adaptive.adjust(40.0);
    let size_after_increase = adaptive.get_batch_size();
    assert!(size_after_increase > 8);

    // Reset
    adaptive.reset();
    assert_eq!(adaptive.get_batch_size(), 32);
}
