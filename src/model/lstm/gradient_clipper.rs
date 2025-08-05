//! Proper gradient clipping implementation with direct gradient scaling
//!
//! This module implements mathematically correct gradient clipping that scales
//! gradients directly without modifying learning rates, preserving optimizer
//! state integrity for momentum-based optimizers like Adam/AdamW.

use crate::utils::error::{Result, VangaError};
use candle_core::{backprop::GradStore, Device, Tensor, Var};
use candle_nn::VarMap;
use std::collections::HashMap;

/// Custom gradient store that enables gradient modification for proper clipping
///
/// This wrapper around Candle's read-only GradStore allows us to:
/// 1. Calculate gradient norms correctly
/// 2. Scale gradients directly when clipping is needed
/// 3. Preserve original learning rates and optimizer state
/// 4. Maintain mathematical equivalence to PyTorch's clip_grad_norm_
#[derive(Debug)]
pub struct ClippableGradStore {
    /// Original gradients from backward pass
    original_grads: HashMap<String, Tensor>,
    /// Clipped gradients (only populated when clipping is applied)
    clipped_grads: HashMap<String, Tensor>,
    /// Whether gradients have been clipped
    is_clipped: bool,
    /// Original gradient norm before clipping
    original_norm: f64,
    /// Effective gradient norm after clipping
    effective_norm: f64,
    /// Device for tensor operations
    device: Device,
}

impl ClippableGradStore {
    /// Create a new ClippableGradStore from Candle's GradStore
    ///
    /// This extracts all gradients from the read-only GradStore and stores them
    /// in a modifiable format for potential clipping operations.
    pub fn from_grad_store(
        grad_store: &GradStore,
        varmap: &VarMap,
        device: &Device,
    ) -> Result<Self> {
        let mut original_grads = HashMap::new();

        // Extract all gradients from the VarMap
        for var in varmap.all_vars().iter() {
            if let Some(grad) = grad_store.get(var) {
                // Create a unique key for this variable
                let var_key = Self::create_var_key(var)?;

                // Clone the gradient tensor for modification
                let grad_clone = grad.clone();
                original_grads.insert(var_key, grad_clone);
            }
        }

        if original_grads.is_empty() {
            return Err(VangaError::ModelError(
                "No gradients found in GradStore - this indicates a problem with backpropagation"
                    .to_string(),
            ));
        }

        Ok(Self {
            original_grads,
            clipped_grads: HashMap::new(),
            is_clipped: false,
            original_norm: 0.0,
            effective_norm: 0.0,
            device: device.clone(),
        })
    }

    /// Apply gradient clipping using L2 norm (equivalent to PyTorch's clip_grad_norm_)
    ///
    /// Mathematical formula:
    /// 1. Calculate total norm: ||g|| = sqrt(sum(||g_i||²)) for all parameters i
    /// 2. If ||g|| > threshold: g_clipped = g * (threshold / ||g||)
    /// 3. Else: g_clipped = g (no modification)
    ///
    /// This preserves the gradient direction while limiting the magnitude,
    /// which is the standard approach used in PyTorch, TensorFlow, etc.
    pub fn clip_gradients_by_norm(&mut self, clip_value: f64) -> Result<(f64, f64)> {
        // Step 1: Calculate the L2 norm of all gradients
        let grad_norm = self.calculate_gradient_norm()?;
        self.original_norm = grad_norm;

        log::debug!(
            "🔍 Gradient norm calculated: {:.6} from {} parameters (threshold: {:.3})",
            grad_norm,
            self.original_grads.len(),
            clip_value
        );

        // Step 2: Apply clipping if needed
        if grad_norm > clip_value && grad_norm > 0.0 {
            let clip_ratio = clip_value / grad_norm;
            self.effective_norm = clip_value; // Clipped norm is exactly the threshold

            // Step 3: Scale all gradients by the clip ratio
            self.clipped_grads.clear();
            for (var_key, original_grad) in &self.original_grads {
                // Scale gradient: g_clipped = g * (threshold / ||g||)
                let clipped_grad = original_grad
                    .broadcast_mul(&Tensor::new(&[clip_ratio as f32], &self.device)?)?
                    .contiguous()?;

                self.clipped_grads.insert(var_key.clone(), clipped_grad);
            }

            self.is_clipped = true;

            log::debug!(
                "✂️ GRADIENT CLIPPING APPLIED: original_norm={:.6} > threshold={:.6} (clip_ratio={:.6}, effective_norm={:.6})",
                grad_norm,
                clip_value,
                clip_ratio,
                self.effective_norm
            );
        } else {
            // No clipping needed
            self.effective_norm = grad_norm;
            self.is_clipped = false;

            log::trace!(
                "✅ No gradient clipping needed: norm={:.6} <= threshold={:.6}",
                grad_norm,
                clip_value
            );
        }

        Ok((self.original_norm, self.effective_norm))
    }

    /// Calculate the L2 norm of all gradients
    ///
    /// Formula: ||g|| = sqrt(sum(||g_i||²)) where g_i are individual parameter gradients
    /// This matches PyTorch's implementation exactly.
    pub fn calculate_gradient_norm(&self) -> Result<f64> {
        let mut total_norm_squared = 0.0f64;

        for (var_key, grad) in &self.original_grads {
            let grad_norm_squared = self.calculate_tensor_norm_squared(grad)?;
            total_norm_squared += grad_norm_squared;

            log::trace!(
                "Gradient norm for param {}: {:.6e}",
                var_key,
                grad_norm_squared.sqrt()
            );
        }

        let total_norm = total_norm_squared.sqrt();
        log::debug!(
            "🔍 Total gradient norm: {:.6e} across {} parameters",
            total_norm,
            self.original_grads.len()
        );

        Ok(total_norm)
    }

    /// Calculate the squared L2 norm of a single tensor
    ///
    /// Formula: ||tensor||² = sum(tensor_i²) for all elements i
    fn calculate_tensor_norm_squared(&self, tensor: &Tensor) -> Result<f64> {
        let squared = tensor.sqr().map_err(|e| {
            VangaError::ModelError(format!(
                "Failed to square tensor for norm calculation: {}",
                e
            ))
        })?;

        let sum = squared.sum_all().map_err(|e| {
            VangaError::ModelError(format!("Failed to sum tensor for norm calculation: {}", e))
        })?;

        // Handle both F32 and F64 tensors
        let norm_squared: f64 = match sum.dtype() {
            candle_core::DType::F32 => {
                let val: f32 = sum.to_scalar().map_err(|e| {
                    VangaError::ModelError(format!("Failed to convert F32 norm to scalar: {}", e))
                })?;
                val as f64
            }
            candle_core::DType::F64 => sum.to_scalar().map_err(|e| {
                VangaError::ModelError(format!("Failed to convert F64 norm to scalar: {}", e))
            })?,
            _ => {
                return Err(VangaError::ModelError(format!(
                    "Unsupported tensor dtype for norm calculation: {:?}",
                    sum.dtype()
                )));
            }
        };

        Ok(norm_squared)
    }

    /// Create a unique key for a variable for HashMap storage
    fn create_var_key(var: &Var) -> Result<String> {
        // Use tensor shape and device as a unique identifier
        let tensor = var.as_tensor();
        let shape_str = tensor
            .shape()
            .dims()
            .iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
            .join("x");

        // Include device info for uniqueness across devices
        let device_str = format!("{:?}", tensor.device());

        Ok(format!("{}_{}", shape_str, device_str))
    }

    /// Get the appropriate gradients for optimizer step
    ///
    /// Returns clipped gradients if clipping was applied, otherwise original gradients.
    /// This ensures the optimizer always receives the correct gradients without
    /// needing to know about the clipping implementation.
    pub fn get_gradients_for_optimizer(&self) -> &HashMap<String, Tensor> {
        if self.is_clipped {
            &self.clipped_grads
        } else {
            &self.original_grads
        }
    }

    /// Check if gradients were clipped
    pub fn is_clipped(&self) -> bool {
        self.is_clipped
    }

    /// Get the original gradient norm (before clipping)
    pub fn original_norm(&self) -> f64 {
        self.original_norm
    }

    /// Get the effective gradient norm (after clipping)
    pub fn effective_norm(&self) -> f64 {
        self.effective_norm
    }

    /// Get the number of parameters with gradients
    pub fn parameter_count(&self) -> usize {
        self.original_grads.len()
    }

    /// Validate gradient flow to catch issues early
    ///
    /// This performs the same validation as the original implementation
    /// but uses the effective (post-clipping) gradient norm.
    pub fn validate_gradient_flow(&self) -> Result<bool> {
        let effective_norm = self.effective_norm;
        let original_norm = self.original_norm;

        // Check for NaN gradients (use effective norm)
        if effective_norm.is_nan() {
            return Err(VangaError::ModelError(
                "🚨 NaN gradients detected! This indicates numerical instability in loss calculation or model architecture.".to_string()
            ));
        }

        // Check for infinite gradients (use effective norm)
        if effective_norm.is_infinite() {
            return Err(VangaError::ModelError(
                "🚨 Infinite gradients detected! This indicates exploding gradients - consider gradient clipping or lower learning rate.".to_string()
            ));
        }

        // Check for zero gradients (no learning) - use effective norm
        if effective_norm < 1e-12 {
            log::warn!(
                "⚠️ Very small effective gradient norm ({:.2e}) - model may not be learning effectively",
                effective_norm
            );
            return Ok(false);
        }

        // Check for exploding gradients - NOW USES EFFECTIVE NORM (the actual training impact)
        if effective_norm > 100.0 {
            if original_norm != effective_norm {
                // Clipping was applied but still too large
                log::warn!(
                    "⚠️ Large effective gradient norm ({:.2e}) after clipping from original ({:.2e}) - consider lower clipping threshold or learning rate",
                    effective_norm,
                    original_norm
                );
            } else {
                // No clipping was applied
                log::warn!(
                    "⚠️ Large gradient norm ({:.2e}) - consider gradient clipping",
                    effective_norm
                );
            }
            return Ok(false);
        }

        // Log successful validation
        if self.is_clipped {
            log::debug!(
                "✂️ Gradient clipping working: original={:.2e} -> effective={:.2e}",
                original_norm,
                effective_norm
            );
        }

        log::debug!(
            "✅ Gradient flow validation passed - effective_norm: {:.6e}, original_norm: {:.6e}",
            effective_norm,
            original_norm
        );

        Ok(true)
    }
}

/// Custom GradStore implementation that works with Candle optimizers
///
/// This struct implements the same interface as Candle's GradStore but allows
/// us to provide modified (clipped) gradients to the optimizer while maintaining
/// full compatibility with all optimizer types.
pub struct ClippedGradStoreAdapter {
    /// The clippable grad store containing our modified gradients
    clipper: ClippableGradStore,
    /// Mapping from Var to our gradient tensors
    var_to_grad: HashMap<String, Tensor>,
}

impl ClippedGradStoreAdapter {
    /// Create a new adapter from a ClippableGradStore
    pub fn new(clipper: ClippableGradStore) -> Self {
        let gradients = clipper.get_gradients_for_optimizer();
        let var_to_grad = gradients.clone();

        Self {
            clipper,
            var_to_grad,
        }
    }

    /// Get gradient for a specific variable (mimics GradStore::get)
    pub fn get(&self, var: &Var) -> Option<&Tensor> {
        let var_key = ClippableGradStore::create_var_key(var).ok()?;
        self.var_to_grad.get(&var_key)
    }

    /// Get the underlying clipper for access to clipping statistics
    pub fn clipper(&self) -> &ClippableGradStore {
        &self.clipper
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::DType;
    use candle_nn::VarBuilder;

    #[test]
    fn test_gradient_norm_calculation() {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        // Create test variables
        let _var1 = vb.get((2, 2), "test_var1").unwrap();
        let _var2 = vb.get((3,), "test_var2").unwrap();

        // Create test gradients (this is a simplified test)
        // In real usage, gradients come from loss.backward()
        let grad1 = Tensor::new(&[[1.0f32, 2.0], [3.0, 4.0]], &device).unwrap();
        let grad2 = Tensor::new(&[1.0f32, 1.0, 1.0], &device).unwrap();

        // Test norm calculation
        let mut original_grads = HashMap::new();
        original_grads.insert("2x2_Cpu".to_string(), grad1);
        original_grads.insert("3_Cpu".to_string(), grad2);

        let clipper = ClippableGradStore {
            original_grads,
            clipped_grads: HashMap::new(),
            is_clipped: false,
            original_norm: 0.0,
            effective_norm: 0.0,
            device,
        };

        let norm = clipper.calculate_gradient_norm().unwrap();

        // Expected norm: sqrt(1² + 2² + 3² + 4² + 1² + 1² + 1²) = sqrt(33) ≈ 5.745
        assert!((norm - 5.745).abs() < 0.01);
    }

    #[test]
    fn test_gradient_clipping_application() {
        let device = Device::Cpu;
        let grad = Tensor::new(&[3.0f32, 4.0], &device).unwrap(); // norm = 5.0

        let mut original_grads = HashMap::new();
        original_grads.insert("test_key".to_string(), grad);

        let mut clipper = ClippableGradStore {
            original_grads,
            clipped_grads: HashMap::new(),
            is_clipped: false,
            original_norm: 0.0,
            effective_norm: 0.0,
            device,
        };

        // Apply clipping with threshold 2.0
        let (original_norm, effective_norm) = clipper.clip_gradients_by_norm(2.0).unwrap();

        assert!((original_norm - 5.0).abs() < 0.01);
        assert!((effective_norm - 2.0).abs() < 0.01);
        assert!(clipper.is_clipped());

        // Check that clipped gradients have the correct norm
        let clipped_grad = clipper.clipped_grads.get("test_key").unwrap();
        let clipped_values: Vec<f32> = clipped_grad.to_vec1().unwrap();

        // Expected: [3, 4] * (2/5) = [1.2, 1.6]
        assert!((clipped_values[0] - 1.2).abs() < 0.01);
        assert!((clipped_values[1] - 1.6).abs() < 0.01);
    }
}
