//! Bayesian Optimization for Target Calibration
//!
//! Implements Gaussian Process-based Bayesian Optimization for finding
//! optimal calibration parameters across all targets.
//!
//! ## Algorithm Overview
//!
//! 1. **Initialization**: Latin Hypercube Sampling for space coverage
//! 2. **Gaussian Process**: Model objective function with uncertainty
//! 3. **Acquisition Function**: Expected Improvement for next point selection
//! 4. **Iteration**: Repeat until convergence or max iterations
//!
//! ## Key Features
//!
//! - Smart parameter space exploration
//! - Uncertainty-aware optimization
//! - Automatic convergence detection
//! - Detailed logging for transparency

use crate::utils::error::{Result, VangaError};
use ndarray::{Array1, Array2};
use rand::Rng;
use statrs::distribution::{ContinuousCDF, Normal};

/// Bayesian Optimizer for target calibration
pub struct BayesianOptimizer {
    /// Parameter bounds (min, max) for each parameter
    bounds: Vec<(f64, f64)>,
    /// Parameter names for logging
    pub param_names: Vec<String>,
    /// Observed parameter combinations
    observations_x: Vec<Vec<f64>>,
    /// Observed objective values (lower is better)
    observations_y: Vec<f64>,
    /// Gaussian Process hyperparameters
    gp_length_scale: f64,
    gp_noise: f64,
    /// Acquisition function type
    acquisition: AcquisitionFunction,
}

/// Acquisition function types for Bayesian optimization
#[derive(Debug, Clone)]
pub enum AcquisitionFunction {
    /// Expected Improvement (default, balanced exploration/exploitation)
    ExpectedImprovement,
    /// Upper Confidence Bound (more exploration)
    UpperConfidenceBound { kappa: f64 },
}

/// Configuration for Bayesian Optimization
#[derive(Debug, Clone)]
pub struct BayesianConfig {
    /// Number of initial random samples (Latin Hypercube)
    pub n_initial: usize,
    /// Maximum number of iterations
    pub max_iterations: usize,
    /// Convergence tolerance (stop if improvement < tolerance)
    pub tolerance: f64,
    /// Acquisition function to use
    pub acquisition: AcquisitionFunction,
    /// Gaussian Process length scale
    pub gp_length_scale: f64,
    /// Gaussian Process noise level
    pub gp_noise: f64,
}

impl Default for BayesianConfig {
    fn default() -> Self {
        Self {
            n_initial: 15,
            max_iterations: 50,
            tolerance: 1e-4,
            acquisition: AcquisitionFunction::ExpectedImprovement,
            gp_length_scale: 0.5,
            gp_noise: 1e-6,
        }
    }
}

impl BayesianOptimizer {
    /// Create new Bayesian optimizer with parameter bounds
    pub fn new(bounds: Vec<(f64, f64)>, param_names: Vec<String>, config: &BayesianConfig) -> Self {
        Self {
            bounds,
            param_names,
            observations_x: Vec::new(),
            observations_y: Vec::new(),
            gp_length_scale: config.gp_length_scale,
            gp_noise: config.gp_noise,
            acquisition: config.acquisition.clone(),
        }
    }

    /// Initialize with Latin Hypercube Sampling for better space coverage
    pub fn initialize_latin_hypercube(&self, n_samples: usize) -> Vec<Vec<f64>> {
        let mut rng = rand::rng();
        let n_params = self.bounds.len();

        // Latin Hypercube Sampling: divide each dimension into n_samples bins
        let mut samples = vec![vec![0.0; n_params]; n_samples];

        for param_idx in 0..n_params {
            // Create shuffled indices for this parameter
            let mut indices: Vec<usize> = (0..n_samples).collect();
            for i in (1..n_samples).rev() {
                let j = rng.random_range(0..=i);
                indices.swap(i, j);
            }

            // Assign values within bins
            let (min, max) = self.bounds[param_idx];
            let bin_size = (max - min) / n_samples as f64;

            for (sample_idx, &bin_idx) in indices.iter().enumerate() {
                let bin_start = min + bin_idx as f64 * bin_size;
                let bin_end = bin_start + bin_size;
                let value = rng.random_range(bin_start..bin_end);
                samples[sample_idx][param_idx] = value;
            }
        }

        log::debug!(
            "🎲 Generated {} Latin Hypercube samples for {} parameters",
            n_samples,
            n_params
        );

        samples
    }

    /// Add observation (parameter values + objective score)
    pub fn add_observation(&mut self, params: Vec<f64>, score: f64) {
        self.observations_x.push(params);
        self.observations_y.push(score);
    }

    /// Suggest next parameter combination to evaluate
    pub fn suggest_next(&self) -> Result<Vec<f64>> {
        if self.observations_x.is_empty() {
            return Err(VangaError::ConfigError(
                "No observations yet. Call initialize_latin_hypercube() first.".to_string(),
            ));
        }

        // Build Gaussian Process model
        let gp = self.build_gaussian_process()?;

        // Optimize acquisition function to find next best point
        let next_params = self.optimize_acquisition(&gp)?;

        Ok(next_params)
    }

    /// Build Gaussian Process model from observations
    fn build_gaussian_process(&self) -> Result<GaussianProcess> {
        let n_obs = self.observations_x.len();
        let n_params = self.bounds.len();

        // Convert observations to ndarray
        let mut x_matrix = Array2::zeros((n_obs, n_params));
        for (i, obs) in self.observations_x.iter().enumerate() {
            for (j, &val) in obs.iter().enumerate() {
                x_matrix[[i, j]] = val;
            }
        }

        let y_vector = Array1::from_vec(self.observations_y.clone());

        // Compute kernel matrix (RBF/Squared Exponential)
        let kernel_matrix = self.compute_kernel_matrix(&x_matrix)?;

        Ok(GaussianProcess {
            x_train: x_matrix,
            y_train: y_vector,
            kernel_matrix,
            length_scale: self.gp_length_scale,
        })
    }

    /// Compute RBF kernel matrix
    fn compute_kernel_matrix(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        let n = x.nrows();
        let mut kernel = Array2::zeros((n, n));

        for i in 0..n {
            for j in 0..n {
                let dist_sq = self.squared_distance(&x.row(i).to_vec(), &x.row(j).to_vec());
                kernel[[i, j]] =
                    (-0.5 * dist_sq / (self.gp_length_scale * self.gp_length_scale)).exp();
            }
        }

        // Add noise to diagonal for numerical stability
        for i in 0..n {
            kernel[[i, i]] += self.gp_noise;
        }

        Ok(kernel)
    }

    /// Squared Euclidean distance between two points
    fn squared_distance(&self, a: &[f64], b: &[f64]) -> f64 {
        a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum()
    }

    /// Optimize acquisition function to find next best point
    fn optimize_acquisition(&self, gp: &GaussianProcess) -> Result<Vec<f64>> {
        let mut rng = rand::rng();
        let n_restarts = 10; // Multiple random restarts
        let n_candidates = 2000; // Random candidates per restart

        let mut best_params = Vec::new();
        let mut best_acquisition_value = f64::NEG_INFINITY;

        // Find best of current observations
        let best_y = self
            .observations_y
            .iter()
            .fold(f64::INFINITY, |a, &b| a.min(b));

        for _ in 0..n_restarts {
            // Generate random candidates
            for _ in 0..n_candidates {
                let mut candidate = Vec::new();
                for (min, max) in &self.bounds {
                    candidate.push(rng.random_range(*min..*max));
                }

                // Predict mean and std at candidate point
                let (mean, std) = gp.predict(&candidate)?;

                // Calculate acquisition function value
                let acquisition_value = match &self.acquisition {
                    AcquisitionFunction::ExpectedImprovement => {
                        self.expected_improvement(mean, std, best_y)
                    }
                    AcquisitionFunction::UpperConfidenceBound { kappa } => {
                        mean - kappa * std // Minimize, so negative UCB
                    }
                };

                if acquisition_value > best_acquisition_value {
                    best_acquisition_value = acquisition_value;
                    best_params = candidate;
                }
            }
        }

        Ok(best_params)
    }

    /// Expected Improvement acquisition function
    fn expected_improvement(&self, mean: f64, std: f64, best_y: f64) -> f64 {
        if std < 1e-10 {
            return 0.0;
        }

        let improvement = best_y - mean; // Minimize
        let z = improvement / std;

        // EI = improvement * Φ(z) + std * φ(z)
        let normal = Normal::new(0.0, 1.0).unwrap();
        let phi = normal.cdf(z);

        // Calculate PDF manually: φ(z) = (1/√(2π)) * exp(-z²/2)
        let pdf = (-0.5 * z * z).exp() / (2.0 * std::f64::consts::PI).sqrt();

        improvement * phi + std * pdf
    }

    /// Get best parameters found so far
    pub fn get_best(&self) -> Option<(Vec<f64>, f64)> {
        if self.observations_x.is_empty() {
            return None;
        }

        let (best_idx, &best_score) = self
            .observations_y
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())?;

        Some((self.observations_x[best_idx].clone(), best_score))
    }

    /// Log optimization progress
    pub fn log_progress(&self, iteration: usize) {
        if let Some((best_params, best_score)) = self.get_best() {
            log::info!("🔍 Iteration {}: Best Score = {:.6}", iteration, best_score);

            for (name, &value) in self.param_names.iter().zip(best_params.iter()) {
                log::debug!("    {}: {:.6}", name, value);
            }
        }
    }

    /// Get number of observations
    pub fn n_observations(&self) -> usize {
        self.observations_x.len()
    }
}

/// Gaussian Process model for Bayesian Optimization
struct GaussianProcess {
    x_train: Array2<f64>,
    y_train: Array1<f64>,
    kernel_matrix: Array2<f64>,
    length_scale: f64,
}

impl GaussianProcess {
    /// Predict mean and standard deviation at new point
    fn predict(&self, x_new: &[f64]) -> Result<(f64, f64)> {
        let n_train = self.x_train.nrows();

        // Compute kernel vector between x_new and training points
        let mut k_star = Array1::zeros(n_train);
        for i in 0..n_train {
            let dist_sq = self.squared_distance(x_new, &self.x_train.row(i).to_vec());
            k_star[i] = (-0.5 * dist_sq / (self.length_scale * self.length_scale)).exp();
        }

        // Solve K * alpha = y for alpha using Cholesky decomposition
        let alpha = self.solve_linear_system(&self.y_train)?;

        // Predictive mean: k_star^T * alpha
        let mean = k_star.dot(&alpha);

        // Predictive variance: k(x*, x*) - k_star^T * K^-1 * k_star
        let k_star_star = 1.0; // RBF kernel at zero distance
        let k_inv_k_star = self.solve_linear_system(&k_star)?;
        let variance = k_star_star - k_star.dot(&k_inv_k_star);
        let std = variance.max(0.0).sqrt();

        Ok((mean, std))
    }

    /// Solve linear system K * x = b using Cholesky decomposition
    fn solve_linear_system(&self, b: &Array1<f64>) -> Result<Array1<f64>> {
        let n = self.kernel_matrix.nrows();

        // Cholesky decomposition: K = L * L^T
        let l = self.cholesky_decomposition()?;

        // Forward substitution: L * y = b
        let mut y = Array1::zeros(n);
        for i in 0..n {
            let mut sum = b[i];
            for j in 0..i {
                sum -= l[[i, j]] * y[j];
            }
            y[i] = sum / l[[i, i]];
        }

        // Backward substitution: L^T * x = y
        let mut x = Array1::zeros(n);
        for i in (0..n).rev() {
            let mut sum = y[i];
            for j in (i + 1)..n {
                sum -= l[[j, i]] * x[j];
            }
            x[i] = sum / l[[i, i]];
        }

        Ok(x)
    }

    /// Cholesky decomposition of kernel matrix
    fn cholesky_decomposition(&self) -> Result<Array2<f64>> {
        let n = self.kernel_matrix.nrows();
        let mut l = Array2::zeros((n, n));

        for i in 0..n {
            for j in 0..=i {
                let mut sum = self.kernel_matrix[[i, j]];

                for k in 0..j {
                    sum -= l[[i, k]] * l[[j, k]];
                }

                if i == j {
                    if sum <= 0.0 {
                        return Err(VangaError::ConfigError(
                            "Kernel matrix is not positive definite".to_string(),
                        ));
                    }
                    l[[i, j]] = sum.sqrt();
                } else {
                    l[[i, j]] = sum / l[[j, j]];
                }
            }
        }

        Ok(l)
    }

    fn squared_distance(&self, a: &[f64], b: &[f64]) -> f64 {
        a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum()
    }
}
