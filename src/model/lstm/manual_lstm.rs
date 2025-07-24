//! Manual LSTM implementation with seeded weight initialization
//!
//! This module provides a manual LSTM implementation that bypasses Candle's
//! default weight initialization to ensure reproducible results.

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{Linear, VarBuilder, VarMap};
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};

/// Manual LSTM cell with seeded weight initialization
pub struct SeededLSTMCell {
    pub input_to_hidden: Linear,
    pub hidden_to_hidden: Linear,
    pub input_size: usize,
    pub hidden_size: usize,
}

impl SeededLSTMCell {
    /// Create a new LSTM cell with seeded weights
    pub fn new_with_seed(
        input_size: usize,
        hidden_size: usize,
        seed: u64,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        log::info!(
            "🎲 Creating seeded LSTM cell ({}→{}) with seed {}",
            input_size,
            hidden_size,
            seed
        );

        // Create seeded weights manually
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

        // Create VarMap and VarBuilder for weight storage
        let varmap = VarMap::new();
        let vs = VarBuilder::from_varmap(&varmap, dtype, device);

        // Create linear layers with seeded initialization
        let input_to_hidden = Self::create_seeded_linear(
            &mut rng,
            input_size,
            hidden_size * 4, // 4 gates
            vs.pp("input_to_hidden"),
        )?;

        let hidden_to_hidden = Self::create_seeded_linear(
            &mut rng,
            hidden_size,
            hidden_size * 4, // 4 gates
            vs.pp("hidden_to_hidden"),
        )?;

        Ok(Self {
            input_to_hidden,
            hidden_to_hidden,
            input_size,
            hidden_size,
        })
    }

    /// Create a linear layer with seeded Xavier initialization
    fn create_seeded_linear(
        rng: &mut rand::rngs::StdRng,
        input_size: usize,
        output_size: usize,
        vs: VarBuilder,
    ) -> Result<Linear> {
        // Xavier initialization
        let std_dev = (2.0 / (input_size + output_size) as f64).sqrt();
        let normal = Normal::new(0.0, std_dev).map_err(|e| {
            candle_core::Error::Msg(format!("Failed to create normal distribution: {}", e))
        })?;

        // Generate weight values
        let weight_size = input_size * output_size;
        let mut weight_values = Vec::with_capacity(weight_size);
        for _ in 0..weight_size {
            weight_values.push(normal.sample(rng) as f32);
        }

        // Generate bias values (zero initialization)
        let bias_values = vec![0.0f32; output_size];

        // Create tensors
        let weight_tensor =
            Tensor::from_vec(weight_values, &[output_size, input_size], vs.device())?
                .to_dtype(vs.dtype())?;

        let bias_tensor =
            Tensor::from_vec(bias_values, &[output_size], vs.device())?.to_dtype(vs.dtype())?;

        // Create linear layer manually
        Ok(Linear::new(weight_tensor, Some(bias_tensor)))
    }

    /// Forward pass through LSTM cell
    pub fn forward(
        &self,
        input: &Tensor,
        hidden: &Tensor,
        cell: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        // Input gates: input_gate, forget_gate, cell_gate, output_gate
        let input_gates = self.input_to_hidden.forward(input)?;
        let hidden_gates = self.hidden_to_hidden.forward(hidden)?;
        let gates = input_gates.add(&hidden_gates)?;

        // Split gates
        let gate_size = self.hidden_size;
        let input_gate = candle_nn::ops::sigmoid(&gates.narrow(1, 0, gate_size)?)?;
        let forget_gate = candle_nn::ops::sigmoid(&gates.narrow(1, gate_size, gate_size)?)?;
        let cell_gate = gates.narrow(1, gate_size * 2, gate_size)?.tanh()?;
        let output_gate = candle_nn::ops::sigmoid(&gates.narrow(1, gate_size * 3, gate_size)?)?;

        // Update cell state
        let new_cell = forget_gate.mul(cell)?.add(&input_gate.mul(&cell_gate)?)?;

        // Update hidden state
        let new_hidden = output_gate.mul(&new_cell.tanh()?)?;

        Ok((new_hidden, new_cell))
    }
}

/// Manual LSTM layer with multiple cells
pub struct SeededLSTMLayer {
    pub cells: Vec<SeededLSTMCell>,
    pub hidden_size: usize,
    pub num_layers: usize,
}

impl SeededLSTMLayer {
    /// Create a new LSTM layer with seeded weights
    pub fn new_with_seed(
        input_size: usize,
        hidden_size: usize,
        num_layers: usize,
        seed: u64,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        log::info!(
            "🎲 Creating seeded LSTM layer with {} layers, seed {}",
            num_layers,
            seed
        );

        let mut cells = Vec::new();
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

        for layer_idx in 0..num_layers {
            let layer_input_size = if layer_idx == 0 {
                input_size
            } else {
                hidden_size
            };

            // Generate unique seed for each layer
            let layer_seed = seed.wrapping_add(layer_idx as u64 * 1000);

            let cell = SeededLSTMCell::new_with_seed(
                layer_input_size,
                hidden_size,
                layer_seed,
                device,
                dtype,
            )?;

            cells.push(cell);
        }

        Ok(Self {
            cells,
            hidden_size,
            num_layers,
        })
    }

    /// Forward pass through all LSTM layers
    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let batch_size = input.dim(0)?;
        let seq_len = input.dim(1)?;

        // Initialize hidden and cell states
        let mut hidden_states = Vec::new();
        let mut cell_states = Vec::new();

        for _ in 0..self.num_layers {
            let hidden = Tensor::zeros(
                &[batch_size, self.hidden_size],
                input.dtype(),
                input.device(),
            )?;
            let cell = Tensor::zeros(
                &[batch_size, self.hidden_size],
                input.dtype(),
                input.device(),
            )?;
            hidden_states.push(hidden);
            cell_states.push(cell);
        }

        let mut layer_input = input.clone();

        // Process through each layer
        for layer_idx in 0..self.num_layers {
            let mut layer_outputs = Vec::new();

            // Process each time step
            for t in 0..seq_len {
                let step_input = layer_input.narrow(1, t, 1)?.squeeze(1)?;

                let (new_hidden, new_cell) = self.cells[layer_idx].forward(
                    &step_input,
                    &hidden_states[layer_idx],
                    &cell_states[layer_idx],
                )?;

                hidden_states[layer_idx] = new_hidden.clone();
                cell_states[layer_idx] = new_cell;

                layer_outputs.push(new_hidden.unsqueeze(1)?);
            }

            // Concatenate time steps
            layer_input = Tensor::cat(&layer_outputs, 1)?;
        }

        // Return the last hidden state
        Ok(hidden_states[self.num_layers - 1].clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_seeded_lstm_reproducibility() -> Result<()> {
        let device = Device::Cpu;
        let dtype = DType::F32;
        let input_size = 10;
        let hidden_size = 20;
        let num_layers = 2;
        let seed = 42;

        // Create two LSTM layers with the same seed
        let lstm1 = SeededLSTMLayer::new_with_seed(
            input_size,
            hidden_size,
            num_layers,
            seed,
            &device,
            dtype,
        )?;
        let lstm2 = SeededLSTMLayer::new_with_seed(
            input_size,
            hidden_size,
            num_layers,
            seed,
            &device,
            dtype,
        )?;

        // Create test input
        let batch_size = 2;
        let seq_len = 5;
        let input_data: Vec<f32> = (0..batch_size * seq_len * input_size)
            .map(|i| (i as f32) * 0.01)
            .collect();

        let input = Tensor::from_vec(input_data, &[batch_size, seq_len, input_size], &device)?
            .to_dtype(dtype)?;

        // Forward pass through both models
        let output1 = lstm1.forward(&input)?;
        let output2 = lstm2.forward(&input)?;

        // Check if outputs are identical
        let diff = output1.sub(&output2)?.abs()?.sum_all()?;
        let diff_value: f32 = diff.to_scalar()?;

        assert!(
            diff_value < 1e-6,
            "LSTM outputs should be identical with same seed, got diff: {}",
            diff_value
        );

        // Test with different seed
        let lstm3 = SeededLSTMLayer::new_with_seed(
            input_size,
            hidden_size,
            num_layers,
            123,
            &device,
            dtype,
        )?;
        let output3 = lstm3.forward(&input)?;

        let diff2 = output1.sub(&output3)?.abs()?.sum_all()?;
        let diff2_value: f32 = diff2.to_scalar()?;

        assert!(
            diff2_value > 1e-4,
            "LSTM outputs should be different with different seeds"
        );

        Ok(())
    }
}
