
#[cfg(test)]
mod tests {
    use crate::model::lstm::LSTMModel;
    use crate::config::model::{LayerNormConfig, ModelConfig};
    use candle_core::{Device, Tensor, DType};

    #[test]
    fn test_layer_norm_numerical_stability() -> anyhow::Result<()> {
        let device = Device::Cpu;
        
        // Create a config with LayerNorm enabled
        let ln_config = LayerNormConfig {
            enabled: true,
            epsilon: 1e-5,
            lstm_cell: true,
            position: "post".to_string(),
        };
        
        // Create a dummy model instance just to access the method
        let config = ModelConfig {
            layer_norm: Some(ln_config.clone()),
            input_size: 3,
            hidden_size: 3,
            num_layers: 1,
            output_size: 1,
            ..Default::default()
        };
        
        let model = LSTMModel::new(&config)?;
        
        // Case 1: Large mean, small variance (Catastrophic cancellation candidate)
        // Values around 1,000,000 with small deviations
        // Using f32 might show the issue more easily than f64, but let's try f64 first as in the code
        // The code handles both, but usually inputs are f64 in this codebase? 
        // Let's check the code again. It matches input dtype.
        
        // Test with F32 which has less precision and is more prone to this error
        let data: Vec<f32> = vec![10000.0, 10000.1, 9999.9];
        let input = Tensor::new(data.as_slice(), &device)?.reshape((1, 3))?;
        
        // This should NOT panic or return NaNs
        let output = model.apply_layer_norm(&input, &ln_config, 0)?;
        
        println!("Input: {:?}", input.to_vec2::<f32>()?);
        println!("Output: {:?}", output.to_vec2::<f32>()?);
        
        let out_vec = output.flatten_all()?.to_vec1::<f32>()?;
        for v in out_vec {
            assert!(!v.is_nan(), "Output contained NaN with large mean input");
            assert!(!v.is_infinite(), "Output contained Inf with large mean input");
        }

        Ok(())
    }
}
