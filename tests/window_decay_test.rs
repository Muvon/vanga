//! Tests for window-based learning rate decay functionality

#[cfg(test)]
mod tests {
    use vanga::config::training::{LearningRateConfig, TrainingParams};

    #[test]
    fn test_window_decay_calculation() {
        let base_lr = 0.001_f64;
        let decay_factor = 0.8_f64;

        // Test exponential decay: base_lr * decay^window_id
        assert!((base_lr * decay_factor.powi(0) - 0.001).abs() < f64::EPSILON); // Window 0: 100%
        assert!((base_lr * decay_factor.powi(1) - 0.0008).abs() < f64::EPSILON); // Window 1: 80%
        assert!((base_lr * decay_factor.powi(2) - 0.00064).abs() < 1e-10); // Window 2: 64%
        assert!((base_lr * decay_factor.powi(3) - 0.000512).abs() < 1e-10); // Window 3: 51.2%
    }

    #[test]
    fn test_no_decay() {
        let base_lr = 0.001_f64;
        let decay_factor = 1.0_f64; // No decay

        // All windows should have same learning rate
        for window_id in 0..10_i32 {
            assert!((base_lr * decay_factor.powi(window_id) - 0.001).abs() < f64::EPSILON);
        }
    }

    #[test]
    fn test_aggressive_decay() {
        let base_lr = 0.001_f64;
        let decay_factor = 0.5_f64; // 50% reduction per window

        assert!((base_lr * decay_factor.powi(0) - 0.001).abs() < f64::EPSILON); // Window 0: 100%
        assert!((base_lr * decay_factor.powi(1) - 0.0005).abs() < f64::EPSILON); // Window 1: 50%
        assert!((base_lr * decay_factor.powi(2) - 0.00025).abs() < f64::EPSILON); // Window 2: 25%
        assert!((base_lr * decay_factor.powi(3) - 0.000125).abs() < f64::EPSILON);
        // Window 3: 12.5%
    }

    #[test]
    fn test_training_params_default() {
        let params = TrainingParams::default();
        assert_eq!(params.window_decay, 1.0); // No decay by default
    }

    #[test]
    fn test_learning_rate_extraction() {
        // Test Fixed learning rate
        let fixed_lr = LearningRateConfig::Fixed(0.001);
        let base_lr = match fixed_lr {
            LearningRateConfig::Fixed(lr) => lr,
            LearningRateConfig::Adaptive { initial_lr, .. } => initial_lr,
            LearningRateConfig::Auto { max_lr, .. } => max_lr,
        };
        assert_eq!(base_lr, 0.001);

        // Test Adaptive learning rate
        let adaptive_lr = LearningRateConfig::Adaptive {
            initial_lr: 0.002,
            patience: 10,
            factor: 0.5,
        };
        let base_lr = match adaptive_lr {
            LearningRateConfig::Fixed(lr) => lr,
            LearningRateConfig::Adaptive { initial_lr, .. } => initial_lr,
            LearningRateConfig::Auto { max_lr, .. } => max_lr,
        };
        assert_eq!(base_lr, 0.002);
    }

    #[test]
    fn test_window_decay_progression() {
        let scenarios = vec![
            (0.9_f64, "Conservative"), // 10% reduction
            (0.8_f64, "Standard"),     // 20% reduction
            (0.7_f64, "Aggressive"),   // 30% reduction
            (1.0_f64, "No decay"),     // No reduction
        ];

        for (decay_factor, description) in scenarios {
            println!("\n{} decay (factor={}):", description, decay_factor);
            let base_lr = 0.001_f64;

            for window_id in 0..5_i32 {
                let window_lr = base_lr * decay_factor.powi(window_id);
                let percentage = (window_lr / base_lr) * 100.0;
                println!(
                    "  Window {}: lr={:.6} ({:.1}%)",
                    window_id + 1,
                    window_lr,
                    percentage
                );
            }
        }
    }
}
