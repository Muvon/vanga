use crate::optimization::{FracAdam, ParamsFracAdam};
use candle_core::{Device, Result, Tensor, Var};
use candle_nn::optim::Optimizer;

fn simple_quadratic_loss(w: &Var, device: &Device) -> Result<Tensor> {
    // Loss: (w - 1.0)^2
    let target = Tensor::new(&[1.0f32], device)?;
    let diff = w.as_tensor().sub(&target)?;
    diff.sqr()?.mean_all()
}

#[test]
fn test_frac_adam_learns_simple_quadratic() -> Result<()> {
    let device = Device::Cpu;
    // Start at 0.0, target is 1.0
    let w = Var::new(&[0.0f32], &device)?;

    // Use a modest LR; fractional history will warm up automatically
    let params = ParamsFracAdam {
        lr: 0.05,
        ..Default::default()
    };
    let mut opt = FracAdam::new(vec![w.clone()], params)?;

    // Initial loss
    let initial_loss = simple_quadratic_loss(&w, &device)?
        .to_scalar::<f32>()
        .expect("initial loss scalar");

    // Run a few update steps
    for _ in 0..50 {
        let loss = simple_quadratic_loss(&w, &device)?;
        opt.backward_step(&loss)?;
    }

    let final_loss = simple_quadratic_loss(&w, &device)?
        .to_scalar::<f32>()
        .expect("final loss scalar");

    assert!(
        final_loss < initial_loss,
        "FracAdam should decrease the loss: {} -> {}",
        initial_loss,
        final_loss
    );
    Ok(())
}

#[test]
fn test_frac_adam_weight_updates() -> Result<()> {
    let device = Device::Cpu;
    // Start at 0.0, target is 1.0
    let w = Var::new(&[0.0f32], &device)?;

    let params = ParamsFracAdam {
        lr: 0.1,
        fractional: crate::optimization::FractionalConfig {
            alpha: 0.9, // High fractional order
            memory_window: 30,
            step_size: 1.0,
        },
        ..Default::default()
    };
    let mut opt = FracAdam::new(vec![w.clone()], params)?;

    // Track weight values
    let initial_weight = w.as_tensor().to_vec1::<f32>()?[0];

    // Run several update steps and verify weights are changing
    let mut prev_weight = initial_weight;
    let mut weight_changes = Vec::new();

    for i in 0..20 {
        let loss = simple_quadratic_loss(&w, &device)?;
        opt.backward_step(&loss)?;

        let current_weight = w.as_tensor().to_vec1::<f32>()?[0];
        let change = (current_weight - prev_weight).abs();
        weight_changes.push(change);

        // After warmup, weights should be changing
        if i > 5 {
            assert!(
                change > 1e-6,
                "Weight should be updating at step {}: prev={:.6}, curr={:.6}, change={:.9}",
                i,
                prev_weight,
                current_weight,
                change
            );
        }

        prev_weight = current_weight;
    }

    // Verify we're moving toward the target (1.0)
    let final_weight = w.as_tensor().to_vec1::<f32>()?[0];
    assert!(
        final_weight > initial_weight,
        "Weight should move toward target: {} -> {}",
        initial_weight,
        final_weight
    );
    assert!(
        (final_weight - 1.0).abs() < (initial_weight - 1.0).abs(),
        "Should be closer to target 1.0: initial_dist={}, final_dist={}",
        (initial_weight - 1.0).abs(),
        (final_weight - 1.0).abs()
    );

    Ok(())
}

#[test]
fn test_frac_adam_convergence_comparison() -> Result<()> {
    // Compare convergence with different alpha values
    let device = Device::Cpu;

    let test_alpha = |alpha: f64, iterations: usize| -> Result<(f32, f32)> {
        let w = Var::new(&[0.0f32], &device)?;
        let params = ParamsFracAdam {
            lr: 0.05,
            fractional: crate::optimization::FractionalConfig {
                alpha,
                memory_window: 30,
                step_size: 1.0,
            },
            ..Default::default()
        };
        let mut opt = FracAdam::new(vec![w.clone()], params)?;

        for _ in 0..iterations {
            let loss = simple_quadratic_loss(&w, &device)?;
            opt.backward_step(&loss)?;
        }

        let final_loss = simple_quadratic_loss(&w, &device)?.to_scalar::<f32>()?;
        let final_weight = w.as_tensor().to_vec1::<f32>()?[0];
        Ok((final_loss, final_weight))
    };

    // Test different fractional orders with appropriate iterations
    let (loss_05, weight_05) = test_alpha(0.5, 200)?;
    let (loss_08, weight_08) = test_alpha(0.8, 300)?;
    let (loss_095, weight_095) = test_alpha(0.95, 500)?; // More iterations for high alpha

    println!("Alpha=0.5: loss={:.6}, weight={:.6}", loss_05, weight_05);
    println!("Alpha=0.8: loss={:.6}, weight={:.6}", loss_08, weight_08);
    println!("Alpha=0.95: loss={:.6}, weight={:.6}", loss_095, weight_095);

    // All should converge reasonably well
    // Note: With correct Grünwald-Letnikov weights, convergence characteristics differ
    assert!(
        loss_05 < 1.0,
        "Alpha=0.5 should show progress: loss={}",
        loss_05
    );
    assert!(
        loss_08 < 0.5,
        "Alpha=0.8 should converge better: loss={}",
        loss_08
    );
    assert!(
        loss_095 < 0.5,
        "Alpha=0.95 needs more iterations but should improve: loss={}",
        loss_095
    );

    // Verify weights are moving toward target (1.0)
    assert!(
        weight_05 > 0.0,
        "Alpha=0.5 weight should move toward 1.0: {}",
        weight_05
    );
    assert!(
        weight_08 > 0.0,
        "Alpha=0.8 weight should move toward 1.0: {}",
        weight_08
    );
    assert!(
        weight_095 > 0.0,
        "Alpha=0.95 weight should move toward 1.0: {}",
        weight_095
    );

    Ok(())
}
