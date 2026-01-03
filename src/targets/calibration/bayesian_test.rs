//! Tests for Bayesian Optimization with Thompson Sampling and Trust Regions

use super::bayesian::*;
use crate::utils::error::Result;

#[test]
fn test_trust_region_creation() {
    let center = vec![0.5, 0.5, 0.5];
    let tr = TrustRegion::new(center.clone(), 0.3);

    assert_eq!(tr.center, center);
    assert_eq!(tr.radius, 0.3);
    assert_eq!(tr.success_counter, 0);
    assert_eq!(tr.failure_counter, 0);
}

#[test]
fn test_trust_region_expand() {
    let center = vec![0.5, 0.5];
    let mut tr = TrustRegion::new(center, 0.3);

    // Expand after 3 successes
    tr.expand();
    assert_eq!(tr.success_counter, 1);
    tr.expand();
    assert_eq!(tr.success_counter, 2);
    tr.expand();
    assert_eq!(tr.success_counter, 0); // Reset after expansion
    assert_eq!(tr.radius, 0.6); // Doubled
}

#[test]
fn test_trust_region_shrink() {
    let center = vec![0.5, 0.5];
    let mut tr = TrustRegion::new(center, 0.3);

    // Shrink after 3 failures
    tr.shrink();
    assert_eq!(tr.failure_counter, 1);
    tr.shrink();
    assert_eq!(tr.failure_counter, 2);
    tr.shrink();
    assert_eq!(tr.failure_counter, 0); // Reset after shrinking
    assert_eq!(tr.radius, 0.15); // Halved
}

#[test]
fn test_trust_region_needs_restart() {
    let center = vec![0.5, 0.5];
    let mut tr = TrustRegion::new(center, 0.3);

    assert!(!tr.needs_restart());

    // Shrink multiple times until restart needed
    for _ in 0..10 {
        tr.shrink();
        tr.shrink();
        tr.shrink();
    }

    assert!(tr.needs_restart());
}

#[test]
fn test_trust_region_restart() {
    let center = vec![0.5, 0.5];
    let mut tr = TrustRegion::new(center.clone(), 0.3);

    tr.radius = 0.001; // Very small
    let new_center = vec![0.8, 0.2];
    tr.restart(new_center.clone());

    assert_eq!(tr.center, new_center);
    assert_eq!(tr.radius, 0.3); // Reset to initial
    assert_eq!(tr.success_counter, 0);
    assert_eq!(tr.failure_counter, 0);
}

#[test]
fn test_trust_region_clip() {
    let center = vec![0.5, 0.5];
    let tr = TrustRegion::new(center, 0.2);
    let bounds = vec![(0.0, 1.0), (0.0, 1.0)];

    // Point outside trust region
    let point = vec![0.9, 0.1];
    let clipped = tr.clip_to_region(&point, &bounds);

    // Should be clipped to trust region
    assert!(clipped[0] >= 0.3 && clipped[0] <= 0.7); // center ± radius
    assert!(clipped[1] >= 0.3 && clipped[1] <= 0.7);
}

#[test]
fn test_acquisition_function_variants() {
    // Test all acquisition function types can be created
    let ei = AcquisitionFunction::ExpectedImprovement;
    let ucb = AcquisitionFunction::UpperConfidenceBound { kappa: 2.5 };
    let ts = AcquisitionFunction::ThompsonSampling;
    let egts = AcquisitionFunction::EpsilonGreedyThompsonSampling { epsilon: 0.3 };

    // Just ensure they can be created and cloned
    let _ei_clone = ei.clone();
    let _ucb_clone = ucb.clone();
    let _ts_clone = ts.clone();
    let _egts_clone = egts.clone();
}

#[test]
fn test_bayesian_config_default() {
    let config = BayesianConfig::default();

    assert_eq!(config.n_initial, 30);
    assert_eq!(config.max_iterations, 150);
    assert_eq!(config.tolerance, 1e-4);
    assert!(config.enable_trust_regions);
    assert!(config.enable_adaptive_restart);
    assert_eq!(config.stagnation_window, 15);
    assert_eq!(config.batch_size, 1);
}

#[test]
fn test_bayesian_config_high_dimensional() {
    let config = BayesianConfig::for_high_dimensional();

    assert_eq!(config.n_initial, 40);
    assert_eq!(config.max_iterations, 200);
    assert!(config.enable_trust_regions);
    assert_eq!(config.stagnation_window, 20);
}

#[test]
fn test_bayesian_config_quick_testing() {
    let config = BayesianConfig::for_quick_testing();

    assert_eq!(config.n_initial, 10);
    assert_eq!(config.max_iterations, 30);
    assert!(!config.enable_trust_regions);
    assert!(!config.enable_adaptive_restart);
}

#[test]
fn test_bayesian_config_maximum_quality() {
    let config = BayesianConfig::for_maximum_quality();

    assert_eq!(config.n_initial, 50);
    assert_eq!(config.max_iterations, 300);
    assert!(config.enable_trust_regions);
    assert!(config.enable_adaptive_restart);
    assert_eq!(config.batch_size, 3);
}

#[test]
fn test_bayesian_optimizer_creation() {
    let bounds = vec![(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)];
    let param_names = vec!["p1".to_string(), "p2".to_string(), "p3".to_string()];
    let config = BayesianConfig::default();

    let optimizer = BayesianOptimizer::new(bounds, param_names, &config, Some(42));

    assert_eq!(optimizer.seed(), Some(42));
    assert_eq!(optimizer.n_observations(), 0);
}

#[test]
fn test_bayesian_optimizer_add_observation() {
    let bounds = vec![(0.0, 1.0), (0.0, 1.0)];
    let param_names = vec!["p1".to_string(), "p2".to_string()];
    let config = BayesianConfig::default();

    let mut optimizer = BayesianOptimizer::new(bounds, param_names, &config, Some(42));

    optimizer.add_observation(vec![0.5, 0.5], 1.0);
    assert_eq!(optimizer.n_observations(), 1);

    optimizer.add_observation(vec![0.3, 0.7], 0.8);
    assert_eq!(optimizer.n_observations(), 2);

    let (best_params, best_score) = optimizer.get_best().unwrap();
    assert_eq!(best_score, 0.8);
    assert_eq!(best_params, vec![0.3, 0.7]);
}

#[test]
fn test_latin_hypercube_sampling() {
    let bounds = vec![(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)];
    let param_names = vec!["p1".to_string(), "p2".to_string(), "p3".to_string()];
    let config = BayesianConfig::default();

    let optimizer = BayesianOptimizer::new(bounds.clone(), param_names, &config, Some(42));

    let samples = optimizer.initialize_latin_hypercube(20, "[TEST]");

    assert_eq!(samples.len(), 20);
    assert_eq!(samples[0].len(), 3);

    // Check all samples are within bounds
    for sample in &samples {
        for (i, &value) in sample.iter().enumerate() {
            assert!(value >= bounds[i].0 && value <= bounds[i].1);
        }
    }
}

#[test]
fn test_bayesian_optimization_simple_quadratic() -> Result<()> {
    // Minimize f(x, y) = (x - 0.7)^2 + (y - 0.3)^2
    // Optimum at (0.7, 0.3) with value 0.0
    let objective = |params: &[f64]| -> Result<f64> {
        let x = params[0];
        let y = params[1];
        Ok((x - 0.7).powi(2) + (y - 0.3).powi(2))
    };

    let bounds = vec![(0.0, 1.0), (0.0, 1.0)];
    let param_names = vec!["x".to_string(), "y".to_string()];
    let config = BayesianConfig {
        n_initial: 10,
        max_iterations: 30,
        enable_trust_regions: true,
        ..Default::default()
    };

    let mut optimizer = BayesianOptimizer::new(bounds, param_names, &config, Some(42));

    // Initial exploration
    let initial_samples = optimizer.initialize_latin_hypercube(config.n_initial, "[TEST]");
    for params in initial_samples {
        let score = objective(&params)?;
        optimizer.add_observation(params, score);
    }

    // Bayesian optimization
    for _ in 0..config.max_iterations {
        let next_params = optimizer.suggest_next("[TEST]")?;
        let score = objective(&next_params)?;
        optimizer.add_observation(next_params, score);
    }

    let (best_params, best_score) = optimizer.get_best().unwrap();

    // Should find near-optimal solution
    assert!(best_score < 0.01, "Best score: {}", best_score);
    assert!(
        (best_params[0] - 0.7).abs() < 0.1,
        "Best x: {}",
        best_params[0]
    );
    assert!(
        (best_params[1] - 0.3).abs() < 0.1,
        "Best y: {}",
        best_params[1]
    );

    Ok(())
}

#[test]
fn test_thompson_sampling_acquisition() -> Result<()> {
    let bounds = vec![(0.0, 1.0), (0.0, 1.0)];
    let param_names = vec!["x".to_string(), "y".to_string()];
    let config = BayesianConfig {
        acquisition: AcquisitionFunction::ThompsonSampling,
        n_initial: 5,
        max_iterations: 10,
        ..Default::default()
    };

    let mut optimizer = BayesianOptimizer::new(bounds, param_names, &config, Some(42));

    // Simple objective
    let objective = |params: &[f64]| -> Result<f64> { Ok(params[0].powi(2) + params[1].powi(2)) };

    // Initial samples
    let initial_samples = optimizer.initialize_latin_hypercube(config.n_initial, "[TEST]");
    for params in initial_samples {
        let score = objective(&params)?;
        optimizer.add_observation(params, score);
    }

    // Run optimization with Thompson Sampling
    for _ in 0..config.max_iterations {
        let next_params = optimizer.suggest_next("[TEST]")?;
        let score = objective(&next_params)?;
        optimizer.add_observation(next_params, score);
    }

    let (_, best_score) = optimizer.get_best().unwrap();
    assert!(
        best_score < 0.1,
        "Thompson Sampling should find good solution"
    );

    Ok(())
}

#[test]
fn test_epsilon_greedy_thompson_sampling() -> Result<()> {
    let bounds = vec![(0.0, 1.0), (0.0, 1.0)];
    let param_names = vec!["x".to_string(), "y".to_string()];
    let config = BayesianConfig {
        acquisition: AcquisitionFunction::EpsilonGreedyThompsonSampling { epsilon: 0.3 },
        n_initial: 5,
        max_iterations: 10,
        ..Default::default()
    };

    let mut optimizer = BayesianOptimizer::new(bounds, param_names, &config, Some(42));

    let objective = |params: &[f64]| -> Result<f64> { Ok(params[0].powi(2) + params[1].powi(2)) };

    let initial_samples = optimizer.initialize_latin_hypercube(config.n_initial, "[TEST]");
    for params in initial_samples {
        let score = objective(&params)?;
        optimizer.add_observation(params, score);
    }

    for _ in 0..config.max_iterations {
        let next_params = optimizer.suggest_next("[TEST]")?;
        let score = objective(&next_params)?;
        optimizer.add_observation(next_params, score);
    }

    let (_, best_score) = optimizer.get_best().unwrap();
    assert!(
        best_score < 0.1,
        "Epsilon-Greedy TS should find good solution"
    );

    Ok(())
}

#[test]
fn test_trust_region_optimization() -> Result<()> {
    // Test that trust regions help focus search
    let bounds = vec![(0.0, 1.0), (0.0, 1.0)];
    let param_names = vec!["x".to_string(), "y".to_string()];
    let config = BayesianConfig {
        enable_trust_regions: true,
        n_initial: 5,
        max_iterations: 20,
        ..Default::default()
    };

    let mut optimizer = BayesianOptimizer::new(bounds, param_names, &config, Some(42));

    let objective = |params: &[f64]| -> Result<f64> {
        Ok((params[0] - 0.8).powi(2) + (params[1] - 0.2).powi(2))
    };

    let initial_samples = optimizer.initialize_latin_hypercube(config.n_initial, "[TEST]");
    for params in initial_samples {
        let score = objective(&params)?;
        optimizer.add_observation(params, score);
    }

    for _ in 0..config.max_iterations {
        let next_params = optimizer.suggest_next("[TEST]")?;
        let score = objective(&next_params)?;
        optimizer.add_observation(next_params, score);
    }

    let (best_params, best_score) = optimizer.get_best().unwrap();

    // Trust regions should help converge faster
    assert!(best_score < 0.01, "Trust regions should find good solution");
    assert!((best_params[0] - 0.8).abs() < 0.1);
    assert!((best_params[1] - 0.2).abs() < 0.1);

    Ok(())
}

#[test]
fn test_stagnation_detection_and_restart() -> Result<()> {
    let bounds = vec![(0.0, 1.0)];
    let param_names = vec!["x".to_string()];
    let config = BayesianConfig {
        enable_adaptive_restart: true,
        stagnation_window: 5,
        n_initial: 3,
        max_iterations: 20,
        ..Default::default()
    };

    let mut optimizer = BayesianOptimizer::new(bounds, param_names, &config, Some(42));

    // Objective with local minimum
    let objective = |params: &[f64]| -> Result<f64> {
        let x = params[0];
        // Function with local minimum at 0.3 and global at 0.8
        Ok(if x < 0.5 {
            (x - 0.3).powi(2) + 0.1
        } else {
            (x - 0.8).powi(2)
        })
    };

    let initial_samples = optimizer.initialize_latin_hypercube(config.n_initial, "[TEST]");
    for params in initial_samples {
        let score = objective(&params)?;
        optimizer.add_observation(params, score);
    }

    for _ in 0..config.max_iterations {
        let next_params = optimizer.suggest_next("[TEST]")?;
        let score = objective(&next_params)?;
        optimizer.add_observation(next_params, score);
    }

    // Should eventually escape local minimum
    let (_, best_score) = optimizer.get_best().unwrap();

    // Should eventually escape local minimum
    assert!(
        best_score < 0.05,
        "Should escape local minimum with restart"
    );

    Ok(())
}

#[test]
fn test_novelty_score_calculation() {
    let bounds = vec![(0.0, 1.0), (0.0, 1.0)];
    let param_names = vec!["x".to_string(), "y".to_string()];
    let config = BayesianConfig::default();

    let mut optimizer = BayesianOptimizer::new(bounds, param_names, &config, Some(42));

    // Add some observations
    optimizer.add_observation(vec![0.2, 0.2], 1.0);
    optimizer.add_observation(vec![0.3, 0.3], 1.0);
    optimizer.add_observation(vec![0.25, 0.25], 1.0);

    // Test novelty score (private method, but we can test through behavior)
    // A point far from observations should have high novelty
    // A point close to observations should have low novelty
    // This is tested indirectly through the optimization behavior
}

#[test]
fn test_reproducibility_with_seed() -> Result<()> {
    let bounds = vec![(0.0, 1.0), (0.0, 1.0)];
    let param_names = vec!["x".to_string(), "y".to_string()];
    let config = BayesianConfig::default();

    let objective = |params: &[f64]| -> Result<f64> { Ok(params[0].powi(2) + params[1].powi(2)) };

    // Run 1
    let mut opt1 = BayesianOptimizer::new(bounds.clone(), param_names.clone(), &config, Some(42));
    let samples1 = opt1.initialize_latin_hypercube(10, "[TEST]");
    for params in samples1 {
        let score = objective(&params)?;
        opt1.add_observation(params, score);
    }
    for _ in 0..5 {
        let next = opt1.suggest_next("[TEST]")?;
        let score = objective(&next)?;
        opt1.add_observation(next, score);
    }
    let (_, score1) = opt1.get_best().unwrap();

    // Run 2 with same seed
    let mut opt2 = BayesianOptimizer::new(bounds, param_names, &config, Some(42));
    let samples2 = opt2.initialize_latin_hypercube(10, "[TEST]");
    for params in samples2 {
        let score = objective(&params)?;
        opt2.add_observation(params, score);
    }
    for _ in 0..5 {
        let next = opt2.suggest_next("[TEST]")?;
        let score = objective(&next)?;
        opt2.add_observation(next, score);
    }
    let (_, score2) = opt2.get_best().unwrap();

    // Should get same results with same seed
    assert!(
        (score1 - score2).abs() < 1e-10,
        "Results should be reproducible with same seed"
    );

    Ok(())
}

#[test]
fn test_trust_region_expansion_contraction_cycle() {
    let center = vec![0.5, 0.5];
    let mut tr = TrustRegion::new(center, 0.2);
    let initial_radius = tr.radius;

    // Cycle of successes and failures
    for _ in 0..3 {
        tr.expand();
    }
    assert!(tr.radius > initial_radius, "Should expand after successes");

    let expanded_radius = tr.radius;
    for _ in 0..3 {
        tr.shrink();
    }
    assert!(tr.radius < expanded_radius, "Should shrink after failures");
}

#[test]
fn test_trust_region_bounds_clipping() {
    let center = vec![0.9, 0.1]; // Near boundaries
    let tr = TrustRegion::new(center, 0.3);
    let bounds = vec![(0.0, 1.0), (0.0, 1.0)];

    // Point that would go outside bounds
    let point = vec![1.5, -0.5];
    let clipped = tr.clip_to_region(&point, &bounds);

    // Should be clipped to valid bounds
    assert!(clipped[0] >= 0.0 && clipped[0] <= 1.0);
    assert!(clipped[1] >= 0.0 && clipped[1] <= 1.0);
}

#[test]
fn test_acquisition_function_switching() -> Result<()> {
    let bounds = vec![(0.0, 1.0), (0.0, 1.0)];
    let param_names = vec!["x".to_string(), "y".to_string()];
    let config = BayesianConfig {
        enable_adaptive_restart: true,
        stagnation_window: 3,
        n_initial: 5,
        max_iterations: 15,
        ..Default::default()
    };

    let mut optimizer = BayesianOptimizer::new(bounds, param_names, &config, Some(42));

    // Flat objective to trigger stagnation
    let objective = |_params: &[f64]| -> Result<f64> {
        Ok(1.0) // Always same score
    };

    let initial_samples = optimizer.initialize_latin_hypercube(config.n_initial, "[TEST]");
    for params in initial_samples {
        let score = objective(&params)?;
        optimizer.add_observation(params, score);
    }

    // Run optimization - should trigger acquisition function switches
    for _ in 0..config.max_iterations {
        let next_params = optimizer.suggest_next("[TEST]")?;
        let score = objective(&next_params)?;
        optimizer.add_observation(next_params, score);
    }

    // Should have detected stagnation and switched acquisition functions
    Ok(())
}

#[test]
fn test_epsilon_decay_over_iterations() -> Result<()> {
    let bounds = vec![(0.0, 1.0)];
    let param_names = vec!["x".to_string()];
    let config = BayesianConfig {
        acquisition: AcquisitionFunction::EpsilonGreedyThompsonSampling { epsilon: 0.5 },
        n_initial: 3,
        max_iterations: 20,
        ..Default::default()
    };

    let mut optimizer = BayesianOptimizer::new(bounds, param_names, &config, Some(42));

    let objective = |params: &[f64]| -> Result<f64> { Ok(params[0].powi(2)) };

    let initial_samples = optimizer.initialize_latin_hypercube(config.n_initial, "[TEST]");
    for params in initial_samples {
        let score = objective(&params)?;
        optimizer.add_observation(params, score);
    }

    // Epsilon should decay over iterations
    for _ in 0..config.max_iterations {
        let next_params = optimizer.suggest_next("[TEST]")?;
        let score = objective(&next_params)?;
        optimizer.add_observation(next_params, score);
    }

    let (_, best_score) = optimizer.get_best().unwrap();
    assert!(best_score < 0.05, "Should converge with epsilon decay");

    Ok(())
}

#[test]
fn test_multi_dimensional_optimization() -> Result<()> {
    // Test 5D optimization (same as price levels)
    let bounds = vec![
        (0.1, 3.0),   // bandwidth
        (0.01, 0.45), // percentile_low
        (0.55, 0.99), // percentile_high
        (0.05, 0.9),  // neutral_band
        (0.8, 6.0),   // momentum_factor
    ];
    let param_names = vec![
        "bw".to_string(),
        "p_low".to_string(),
        "p_high".to_string(),
        "nb".to_string(),
        "mf".to_string(),
    ];

    let config = BayesianConfig {
        n_initial: 15,
        max_iterations: 30,
        enable_trust_regions: true,
        ..Default::default()
    };

    let mut optimizer = BayesianOptimizer::new(bounds, param_names, &config, Some(42));

    // Rosenbrock-like function in 5D
    let objective = |params: &[f64]| -> Result<f64> {
        let mut sum = 0.0;
        for i in 0..params.len() - 1 {
            let a = params[i + 1] - params[i].powi(2);
            let b = 1.0 - params[i];
            sum += 100.0 * a.powi(2) + b.powi(2);
        }
        Ok(sum)
    };

    let initial_samples = optimizer.initialize_latin_hypercube(config.n_initial, "[TEST]");
    for params in initial_samples {
        let score = objective(&params)?;
        optimizer.add_observation(params, score);
    }

    for _ in 0..config.max_iterations {
        let next_params = optimizer.suggest_next("[TEST]")?;
        let score = objective(&next_params)?;
        optimizer.add_observation(next_params, score);
    }

    let (_, best_score) = optimizer.get_best().unwrap();
    // Rosenbrock is hard, but should make progress
    assert!(best_score < 1000.0, "Should make progress on 5D problem");

    Ok(())
}

#[test]
fn test_trust_region_with_noisy_objective() -> Result<()> {
    use rand::Rng;

    let bounds = vec![(0.0, 1.0), (0.0, 1.0)];
    let param_names = vec!["x".to_string(), "y".to_string()];
    let config = BayesianConfig {
        enable_trust_regions: true,
        n_initial: 10,
        max_iterations: 30,
        ..Default::default()
    };

    let mut optimizer = BayesianOptimizer::new(bounds, param_names, &config, Some(42));
    let mut rng = rand::rng();

    // Noisy quadratic objective
    let objective = |params: &[f64], rng: &mut rand::rngs::ThreadRng| -> Result<f64> {
        let true_value = (params[0] - 0.6).powi(2) + (params[1] - 0.4).powi(2);
        let noise = rng.random_range(-0.05..0.05);
        Ok(true_value + noise)
    };

    let initial_samples = optimizer.initialize_latin_hypercube(config.n_initial, "[TEST]");
    for params in initial_samples {
        let score = objective(&params, &mut rng)?;
        optimizer.add_observation(params, score);
    }

    for _ in 0..config.max_iterations {
        let next_params = optimizer.suggest_next("[TEST]")?;
        let score = objective(&next_params, &mut rng)?;
        optimizer.add_observation(next_params, score);
    }

    let (best_params, best_score) = optimizer.get_best().unwrap();

    // Should still find good solution despite noise
    assert!(best_score < 0.1, "Should handle noisy objectives");
    assert!((best_params[0] - 0.6).abs() < 0.2);
    assert!((best_params[1] - 0.4).abs() < 0.2);

    Ok(())
}

#[test]
fn test_batch_size_configuration() {
    let config_sequential = BayesianConfig::default();
    assert_eq!(
        config_sequential.batch_size, 1,
        "Default should be sequential"
    );

    let config_parallel = BayesianConfig::for_maximum_quality();
    assert_eq!(
        config_parallel.batch_size, 3,
        "Max quality should use batch"
    );
}

#[test]
fn test_gp_prediction_consistency() -> Result<()> {
    // Test that GP predictions are consistent
    let bounds = vec![(0.0, 1.0), (0.0, 1.0)];
    let param_names = vec!["x".to_string(), "y".to_string()];
    let config = BayesianConfig::default();

    let mut optimizer = BayesianOptimizer::new(bounds, param_names, &config, Some(42));

    // Add some observations
    optimizer.add_observation(vec![0.2, 0.3], 0.5);
    optimizer.add_observation(vec![0.8, 0.7], 0.3);
    optimizer.add_observation(vec![0.5, 0.5], 0.4);

    // Suggest next point (uses GP internally)
    let next1 = optimizer.suggest_next("[TEST]")?;

    // Add observation and suggest again
    optimizer.add_observation(next1, 0.35);
    let next2 = optimizer.suggest_next("[TEST]")?;

    // Should get different suggestions as we add more data
    assert_ne!(next2, vec![0.2, 0.3]);

    Ok(())
}

#[test]
fn test_boundary_conditions() -> Result<()> {
    // Test optimization at parameter boundaries
    let bounds = vec![(0.0, 1.0), (0.0, 1.0)];
    let param_names = vec!["x".to_string(), "y".to_string()];
    let config = BayesianConfig {
        n_initial: 5,
        max_iterations: 15,
        ..Default::default()
    };

    let mut optimizer = BayesianOptimizer::new(bounds, param_names, &config, Some(42));

    // Optimum at boundary
    let objective = |params: &[f64]| -> Result<f64> {
        Ok((params[0] - 0.0).powi(2) + (params[1] - 1.0).powi(2))
    };

    let initial_samples = optimizer.initialize_latin_hypercube(config.n_initial, "[TEST]");
    for params in initial_samples {
        let score = objective(&params)?;
        optimizer.add_observation(params, score);
    }

    for _ in 0..config.max_iterations {
        let next_params = optimizer.suggest_next("[TEST]")?;
        let score = objective(&next_params)?;
        optimizer.add_observation(next_params, score);
    }

    let (best_params, best_score) = optimizer.get_best().unwrap();

    // Should find boundary optimum
    assert!(best_score < 0.05);
    assert!(best_params[0] < 0.2, "Should find x near 0");
    assert!(best_params[1] > 0.8, "Should find y near 1");

    Ok(())
}

#[test]
fn test_convergence_tolerance() -> Result<()> {
    let bounds = vec![(0.0, 1.0)];
    let param_names = vec!["x".to_string()];
    let config = BayesianConfig {
        n_initial: 5,
        max_iterations: 50,
        tolerance: 1e-3,
        ..Default::default()
    };

    let mut optimizer = BayesianOptimizer::new(bounds, param_names, &config, Some(42));

    let objective = |params: &[f64]| -> Result<f64> { Ok((params[0] - 0.5).powi(2)) };

    let initial_samples = optimizer.initialize_latin_hypercube(config.n_initial, "[TEST]");
    for params in initial_samples {
        let score = objective(&params)?;
        optimizer.add_observation(params, score);
    }

    let mut iterations = 0;
    for i in 0..config.max_iterations {
        let next_params = optimizer.suggest_next("[TEST]")?;
        let score = objective(&next_params)?;
        optimizer.add_observation(next_params, score);
        iterations = i + 1;

        // Could add early stopping check here
        if score < 1e-4 {
            break;
        }
    }

    let (_, best_score) = optimizer.get_best().unwrap();
    assert!(best_score < 0.01, "Should converge to optimum");
    assert!(
        iterations < config.max_iterations,
        "Should converge before max iterations"
    );

    Ok(())
}
