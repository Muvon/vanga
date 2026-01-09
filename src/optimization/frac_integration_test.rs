use crate::optimization::{
    FracAdam, FracNAdam, FracProdigy, ParamsFracAdam, ParamsFracNAdam, ParamsFracProdigy,
};
use candle_core::{Device, Result, Tensor, Var};
use candle_nn::optim::Optimizer;

/// Test with a simple linear layer (y = Wx + b)
fn create_linear_layer(
    input_size: usize,
    output_size: usize,
    device: &Device,
) -> Result<(Var, Var)> {
    // Initialize weights with small random values
    let w_data = Tensor::randn(0.0f32, 0.1, &[output_size, input_size], device)?;
    let w = Var::from_tensor(&w_data)?;

    let b_data = Tensor::zeros(&[output_size], candle_core::DType::F32, device)?;
    let b = Var::from_tensor(&b_data)?;

    Ok((w, b))
}

fn linear_forward(x: &Tensor, w: &Var, b: &Var) -> Result<Tensor> {
    // y = Wx + b
    let wx = x.matmul(&w.as_tensor().t()?)?;
    wx.broadcast_add(b.as_tensor())
}

fn mse_loss(predictions: &Tensor, targets: &Tensor) -> Result<Tensor> {
    let diff = predictions.sub(targets)?;
    let squared = diff.sqr()?;
    squared.mean_all()
}

#[test]
fn test_frac_adam_with_linear_layer() -> Result<()> {
    let device = Device::Cpu;
    let batch_size = 4;
    let input_size = 3;
    let output_size = 2;

    // Create a simple linear layer
    let (w, b) = create_linear_layer(input_size, output_size, &device)?;

    // Create optimizer
    let params = ParamsFracAdam {
        lr: 0.01,
        ..Default::default()
    };
    let mut optimizer = FracAdam::new(vec![w.clone(), b.clone()], params)?;

    // Create some training data
    let x = Tensor::randn(0.0f32, 1.0, &[batch_size, input_size], &device)?;
    let y_true = Tensor::randn(0.0f32, 1.0, &[batch_size, output_size], &device)?;

    // Track initial loss
    let y_pred_initial = linear_forward(&x, &w, &b)?;
    let initial_loss = mse_loss(&y_pred_initial, &y_true)?.to_scalar::<f32>()?;

    // Training loop
    for _ in 0..100 {
        let y_pred = linear_forward(&x, &w, &b)?;
        let loss = mse_loss(&y_pred, &y_true)?;
        optimizer.backward_step(&loss)?;
    }

    // Check final loss
    let y_pred_final = linear_forward(&x, &w, &b)?;
    let final_loss = mse_loss(&y_pred_final, &y_true)?.to_scalar::<f32>()?;

    assert!(
        final_loss < initial_loss,
        "FracAdam should reduce loss: {} -> {}",
        initial_loss,
        final_loss
    );

    // Verify significant improvement
    let improvement = (initial_loss - final_loss) / initial_loss;
    assert!(
        improvement > 0.1,
        "Should have at least 10% improvement: {:.2}%",
        improvement * 100.0
    );

    Ok(())
}

#[test]
fn test_frac_nadam_with_linear_layer() -> Result<()> {
    let device = Device::Cpu;
    let batch_size = 4;
    let input_size = 3;
    let output_size = 2;

    // Create a simple linear layer
    let (w, b) = create_linear_layer(input_size, output_size, &device)?;

    // Create optimizer
    let params = ParamsFracNAdam {
        lr: 0.01,
        ..Default::default()
    };
    let mut optimizer = FracNAdam::new(vec![w.clone(), b.clone()], params)?;

    // Create some training data
    let x = Tensor::randn(0.0f32, 1.0, &[batch_size, input_size], &device)?;
    let y_true = Tensor::randn(0.0f32, 1.0, &[batch_size, output_size], &device)?;

    // Track initial loss
    let y_pred_initial = linear_forward(&x, &w, &b)?;
    let initial_loss = mse_loss(&y_pred_initial, &y_true)?.to_scalar::<f32>()?;

    // Training loop
    for _ in 0..100 {
        let y_pred = linear_forward(&x, &w, &b)?;
        let loss = mse_loss(&y_pred, &y_true)?;
        optimizer.backward_step(&loss)?;
    }

    // Check final loss
    let y_pred_final = linear_forward(&x, &w, &b)?;
    let final_loss = mse_loss(&y_pred_final, &y_true)?.to_scalar::<f32>()?;

    assert!(
        final_loss < initial_loss,
        "FracNAdam should reduce loss: {} -> {}",
        initial_loss,
        final_loss
    );

    // Verify significant improvement
    let improvement = (initial_loss - final_loss) / initial_loss;
    assert!(
        improvement > 0.1,
        "Should have at least 10% improvement: {:.2}%",
        improvement * 100.0
    );

    Ok(())
}

#[test]
fn test_frac_optimizers_comparison() -> Result<()> {
    let device = Device::Cpu;

    // Test function that trains a model and returns final loss
    let train_with_adam = || -> Result<f32> {
        let (w, b) = create_linear_layer(5, 3, &device)?;

        let mut optimizer = FracAdam::new(vec![w.clone(), b.clone()], ParamsFracAdam::default())?;

        // Fixed training data for fair comparison
        let x = Tensor::new(
            &[
                [1.0f32, 0.5, -0.3, 0.8, -0.2],
                [0.2, -0.7, 0.4, -0.1, 0.9],
                [-0.5, 0.3, 0.6, -0.4, 0.1],
            ],
            &device,
        )?;
        let y_true = Tensor::new(
            &[[0.5f32, -0.2, 0.8], [-0.3, 0.7, 0.1], [0.2, -0.5, 0.4]],
            &device,
        )?;

        // Train for fixed iterations
        for _ in 0..200 {
            let y_pred = linear_forward(&x, &w, &b)?;
            let loss = mse_loss(&y_pred, &y_true)?;
            optimizer.backward_step(&loss)?;
        }

        // Return final loss
        let y_pred_final = linear_forward(&x, &w, &b)?;
        mse_loss(&y_pred_final, &y_true)?.to_scalar::<f32>()
    };

    let train_with_nadam = || -> Result<f32> {
        let (w, b) = create_linear_layer(5, 3, &device)?;

        let mut optimizer = FracNAdam::new(vec![w.clone(), b.clone()], ParamsFracNAdam::default())?;

        // Fixed training data for fair comparison
        let x = Tensor::new(
            &[
                [1.0f32, 0.5, -0.3, 0.8, -0.2],
                [0.2, -0.7, 0.4, -0.1, 0.9],
                [-0.5, 0.3, 0.6, -0.4, 0.1],
            ],
            &device,
        )?;
        let y_true = Tensor::new(
            &[[0.5f32, -0.2, 0.8], [-0.3, 0.7, 0.1], [0.2, -0.5, 0.4]],
            &device,
        )?;

        // Train for fixed iterations
        for _ in 0..200 {
            let y_pred = linear_forward(&x, &w, &b)?;
            let loss = mse_loss(&y_pred, &y_true)?;
            optimizer.backward_step(&loss)?;
        }

        // Return final loss
        let y_pred_final = linear_forward(&x, &w, &b)?;
        mse_loss(&y_pred_final, &y_true)?.to_scalar::<f32>()
    };

    let adam_loss = train_with_adam()?;
    let nadam_loss = train_with_nadam()?;

    println!("FracAdam final loss: {:.6}", adam_loss);
    println!("FracNAdam final loss: {:.6}", nadam_loss);

    // Both should converge to low loss
    assert!(adam_loss < 0.5, "FracAdam should converge: {}", adam_loss);
    assert!(
        nadam_loss < 0.5,
        "FracNAdam should converge: {}",
        nadam_loss
    );

    Ok(())
}

#[test]
fn test_frac_prodigy_with_linear_layer() -> Result<()> {
    let device = Device::Cpu;
    let batch_size = 4;
    let input_size = 3;
    let output_size = 2;

    // Create a simple linear layer
    let (w, b) = create_linear_layer(input_size, output_size, &device)?;

    // Create optimizer - Prodigy uses lr=1.0 for automatic adaptation
    let params = ParamsFracProdigy {
        lr: 1.0, // CRITICAL: Must be 1.0 for Prodigy auto-adaptation
        ..Default::default()
    };
    let mut optimizer = FracProdigy::new(vec![w.clone(), b.clone()], params)?;

    // Create some training data
    let x = Tensor::randn(0.0f32, 1.0, &[batch_size, input_size], &device)?;
    let y_true = Tensor::randn(0.0f32, 1.0, &[batch_size, output_size], &device)?;

    // Track initial loss
    let y_pred_initial = linear_forward(&x, &w, &b)?;
    let initial_loss = mse_loss(&y_pred_initial, &y_true)?.to_scalar::<f32>()?;

    // Training loop - FracProdigy needs slightly more steps due to warmup
    for step in 0..150 {
        let y_pred = linear_forward(&x, &w, &b)?;
        let loss = mse_loss(&y_pred, &y_true)?;
        optimizer.backward_step(&loss)?;

        // Log progress for first few steps and periodically
        if step == 0 || (step < 10 && step % 2 == 0) || step % 50 == 0 {
            let current_loss = mse_loss(&y_pred, &y_true)?.to_scalar::<f32>()?;
            let effective_lr = optimizer.learning_rate();
            log::trace!(
                "FracProdigy step {}: loss={:.6}, effective_lr={:.2e}, D={:.2e}",
                step,
                current_loss,
                effective_lr,
                optimizer.get_d_estimate()
            );
        }
    }

    // Check final loss
    let y_pred_final = linear_forward(&x, &w, &b)?;
    let final_loss = mse_loss(&y_pred_final, &y_true)?.to_scalar::<f32>()?;

    assert!(
        final_loss < initial_loss,
        "FracProdigy should reduce loss: {} -> {}",
        initial_loss,
        final_loss
    );

    // Verify significant improvement (may need more steps than FracAdam)
    let improvement = (initial_loss - final_loss) / initial_loss;
    assert!(
        improvement > 0.05,
        "Should have at least 5% improvement: {:.2}%",
        improvement * 100.0
    );

    // Verify D-estimate is adapting
    let d_estimate = optimizer.get_d_estimate();
    assert!(
        d_estimate > 0.0,
        "D-estimate should be positive: {}",
        d_estimate
    );

    Ok(())
}
