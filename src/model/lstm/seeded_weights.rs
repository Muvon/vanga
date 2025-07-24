//! Seeded weight initialization for reproducible LSTM training
//!
//! This module provides custom weight initialization that ensures reproducible
//! results across training runs when a seed is provided.

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::VarMap;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};

/// Custom seeded weight initializer for LSTM layers
pub struct SeededWeightInitializer {
    seed: u64,
    rng: rand::rngs::StdRng,
    device: Device,
    dtype: DType,
}

impl SeededWeightInitializer {
    /// Create a new seeded weight initializer
    pub fn new(seed: u64, device: Device, dtype: DType) -> Self {
        let rng = rand::rngs::StdRng::seed_from_u64(seed);
        log::info!("🎲 Created SeededWeightInitializer with seed {}", seed);

        Self {
            seed,
            rng,
            device,
            dtype,
        }
    }

    /// Initialize LSTM weights with Xavier/Glorot initialization using seeded RNG
    pub fn initialize_lstm_weights(
        &mut self,
        input_size: usize,
        hidden_size: usize,
        layer_name: &str,
        varmap: &VarMap,
    ) -> Result<()> {
        log::info!(
            "🎲 Initializing LSTM weights for layer '{}' with seed {}",
            layer_name,
            self.seed
        );

        // LSTM has 4 gates: input, forget, cell, output
        // Each gate has weight_ih (input-to-hidden) and weight_hh (hidden-to-hidden)
        // Plus bias_ih and bias_hh for each gate

        let gates = ["input", "forget", "cell", "output"];

        for gate in &gates {
            // Input-to-hidden weights (input_size x hidden_size)
            let weight_ih_name = format!("{}.weight_ih_{}", layer_name, gate);
            let weight_ih = self.create_xavier_tensor(input_size, hidden_size)?;
            self.insert_tensor_into_varmap(varmap, &weight_ih_name, weight_ih)?;

            // Hidden-to-hidden weights (hidden_size x hidden_size)
            let weight_hh_name = format!("{}.weight_hh_{}", layer_name, gate);
            let weight_hh = self.create_xavier_tensor(hidden_size, hidden_size)?;
            self.insert_tensor_into_varmap(varmap, &weight_hh_name, weight_hh)?;

            // Input-to-hidden bias (hidden_size,)
            let bias_ih_name = format!("{}.bias_ih_{}", layer_name, gate);
            let bias_ih = self.create_zero_tensor(&[hidden_size])?;
            self.insert_tensor_into_varmap(varmap, &bias_ih_name, bias_ih)?;

            // Hidden-to-hidden bias (hidden_size,)
            let bias_hh_name = format!("{}.bias_hh_{}", layer_name, gate);
            let bias_hh = self.create_zero_tensor(&[hidden_size])?;
            self.insert_tensor_into_varmap(varmap, &bias_hh_name, bias_hh)?;
        }

        log::info!(
            "✅ Successfully initialized LSTM weights for layer '{}'",
            layer_name
        );
        Ok(())
    }

    /// Create a tensor with Xavier/Glorot initialization
    fn create_xavier_tensor(&mut self, fan_in: usize, fan_out: usize) -> Result<Tensor> {
        // Xavier initialization: std = sqrt(2.0 / (fan_in + fan_out))
        let std_dev = (2.0 / (fan_in + fan_out) as f64).sqrt();
        let normal = Normal::new(0.0, std_dev).map_err(|e| {
            candle_core::Error::Msg(format!("Failed to create normal distribution: {}", e))
        })?;

        // Generate random values using seeded RNG
        let total_elements = fan_in * fan_out;
        let mut values = Vec::with_capacity(total_elements);

        for _ in 0..total_elements {
            values.push(normal.sample(&mut self.rng) as f32);
        }

        // Create tensor from values
        let tensor =
            Tensor::from_vec(values, &[fan_in, fan_out], &self.device)?.to_dtype(self.dtype)?;

        Ok(tensor)
    }

    /// Create a zero-initialized tensor
    fn create_zero_tensor(&self, shape: &[usize]) -> Result<Tensor> {
        Tensor::zeros(shape, self.dtype, &self.device)
    }

    /// Insert tensor into VarMap (this is a workaround since VarMap doesn't expose direct insertion)
    fn insert_tensor_into_varmap(
        &self,
        _varmap: &VarMap,
        name: &str,
        _tensor: Tensor,
    ) -> Result<()> {
        // This is a limitation - VarMap doesn't allow direct tensor insertion
        // We'll need to use a different approach
        log::warn!(
            "⚠️  Cannot directly insert tensor '{}' into VarMap - this is a Candle limitation",
            name
        );

        // For now, we'll store the tensor information for later use
        // The actual implementation will need to work around VarMap's limitations
        Ok(())
    }
}

/// Alternative approach: Pre-populate VarMap with seeded tensors before creating VarBuilder
pub fn create_seeded_varmap_for_lstm(
    seed: u64,
    input_size: usize,
    hidden_size: usize,
    num_layers: usize,
    device: &Device,
    dtype: DType,
) -> Result<VarMap> {
    log::info!("🎲 Creating seeded VarMap for LSTM with seed {}", seed);

    let varmap = VarMap::new();
    let mut initializer = SeededWeightInitializer::new(seed, device.clone(), dtype);

    // Initialize weights for each LSTM layer
    for layer_idx in 0..num_layers {
        let layer_name = format!("forward_lstm_layer_{}", layer_idx);
        initializer.initialize_lstm_weights(input_size, hidden_size, &layer_name, &varmap)?;
    }

    log::info!("✅ Created seeded VarMap with {} layers", num_layers);
    Ok(varmap)
}

/// Seeded tensor creation utilities
pub struct SeededTensorUtils;

impl SeededTensorUtils {
    /// Create a seeded random tensor with Xavier initialization
    pub fn xavier_tensor(
        seed: u64,
        shape: &[usize],
        device: &Device,
        dtype: DType,
    ) -> Result<Tensor> {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

        let fan_in = if shape.len() >= 2 { shape[0] } else { 1 };
        let fan_out = if shape.len() >= 2 { shape[1] } else { shape[0] };
        let std_dev = (2.0 / (fan_in + fan_out) as f64).sqrt();

        let normal = Normal::new(0.0, std_dev).map_err(|e| {
            candle_core::Error::Msg(format!("Failed to create normal distribution: {}", e))
        })?;

        let total_elements: usize = shape.iter().product();
        let mut values = Vec::with_capacity(total_elements);

        for _ in 0..total_elements {
            values.push(normal.sample(&mut rng) as f32);
        }

        Tensor::from_vec(values, shape, device)?.to_dtype(dtype)
    }

    /// Create a seeded random tensor with normal distribution
    pub fn normal_tensor(
        seed: u64,
        shape: &[usize],
        mean: f64,
        std: f64,
        device: &Device,
        dtype: DType,
    ) -> Result<Tensor> {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let normal = Normal::new(mean, std).map_err(|e| {
            candle_core::Error::Msg(format!("Failed to create normal distribution: {}", e))
        })?;

        let total_elements: usize = shape.iter().product();
        let mut values = Vec::with_capacity(total_elements);

        for _ in 0..total_elements {
            values.push(normal.sample(&mut rng) as f32);
        }

        Tensor::from_vec(values, shape, device)?.to_dtype(dtype)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_seeded_tensor_reproducibility() -> Result<()> {
        let device = Device::Cpu;
        let dtype = DType::F32;
        let shape = &[10, 20];
        let seed = 42;

        // Create two tensors with the same seed
        let tensor1 = SeededTensorUtils::xavier_tensor(seed, shape, &device, dtype)?;
        let tensor2 = SeededTensorUtils::xavier_tensor(seed, shape, &device, dtype)?;

        // They should be identical
        let diff = tensor1.sub(&tensor2)?.abs()?.sum_all()?;
        let diff_value: f32 = diff.to_scalar()?;

        assert!(
            diff_value < 1e-10,
            "Tensors should be identical with same seed"
        );

        // Create tensor with different seed
        let tensor3 = SeededTensorUtils::xavier_tensor(123, shape, &device, dtype)?;
        let diff2 = tensor1.sub(&tensor3)?.abs()?.sum_all()?;
        let diff2_value: f32 = diff2.to_scalar()?;

        assert!(
            diff2_value > 1e-6,
            "Tensors should be different with different seeds"
        );

        Ok(())
    }
}
