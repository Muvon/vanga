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
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
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
    /// Random seed for reproducible optimization (None = random, Some(0) = random, Some(n) = seeded)
    seed: Option<u64>,
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
    /// Default configuration optimized for QUALITY over speed
    /// Suitable for 4D parameter spaces (direction, price_levels, volatility, volume)
    fn default() -> Self {
        Self {
            n_initial: 30,       // Increased from 15 for better initial exploration
            max_iterations: 100, // Increased from 50 for thorough optimization
            tolerance: 1e-5,     // Stricter from 1e-4 for better convergence
            acquisition: AcquisitionFunction::ExpectedImprovement,
            gp_length_scale: 0.5,
            gp_noise: 1e-5, // Increased from 1e-6 for better numerical stability
        }
    }
}

impl BayesianConfig {
    /// Configuration for high-dimensional spaces (6D sentiment with 4 weights + 2 thresholds)
    /// Uses more initial samples and iterations for complex parameter spaces
    pub fn for_high_dimensional() -> Self {
        Self {
            n_initial: 40,       // More samples for 6D space
            max_iterations: 120, // More iterations for complex optimization
            tolerance: 1e-5,
            acquisition: AcquisitionFunction::ExpectedImprovement,
            gp_length_scale: 0.5,
            gp_noise: 1e-5,
        }
    }

    /// Configuration for quick calibration (testing/development)
    /// Trades quality for speed - NOT recommended for production
    pub fn for_quick_testing() -> Self {
        Self {
            n_initial: 10,
            max_iterations: 30,
            tolerance: 1e-4,
            acquisition: AcquisitionFunction::ExpectedImprovement,
            gp_length_scale: 0.5,
            gp_noise: 1e-5,
        }
    }

    /// Configuration for maximum quality (research/final calibration)
    /// Uses extensive exploration - very slow but finds best parameters
    pub fn for_maximum_quality() -> Self {
        Self {
            n_initial: 50,
            max_iterations: 200,
            tolerance: 1e-6,
            acquisition: AcquisitionFunction::ExpectedImprovement,
            gp_length_scale: 0.5,
            gp_noise: 1e-5,
        }
    }
}

impl BayesianOptimizer {
    /// Create new Bayesian optimizer with parameter bounds
    pub fn new(
        bounds: Vec<(f64, f64)>,
        param_names: Vec<String>,
        config: &BayesianConfig,
        seed: Option<u64>,
    ) -> Self {
        Self {
            bounds,
            param_names,
            observations_x: Vec::new(),
            observations_y: Vec::new(),
            gp_length_scale: config.gp_length_scale,
            gp_noise: config.gp_noise,
            acquisition: config.acquisition.clone(),
            seed,
        }
    }

    /// Initialize with Enhanced Latin Hypercube Sampling using maximin criterion
    /// This provides superior space coverage compared to basic LHS
    pub fn initialize_latin_hypercube(&self, n_samples: usize, prefix: &str) -> Vec<Vec<f64>> {
        // Use seeded RNG if seed is provided, otherwise random
        let mut rng: Box<dyn rand::RngCore> = match self.seed {
            Some(0) | None => {
                // seed=0 or None means random
                Box::new(rand::rng())
            }
            Some(seed_value) => {
                // Use seeded RNG for reproducibility
                log::debug!(
                    "{} 🎲 Using seeded RNG (seed={}) for reproducible LHS",
                    prefix,
                    seed_value
                );
                Box::new(StdRng::seed_from_u64(seed_value))
            }
        };

        let n_params = self.bounds.len();

        // Generate multiple LHS candidates and select best using maximin criterion
        let n_candidates = 5; // Generate 5 candidates, pick best
        let mut best_samples = Vec::new();
        let mut best_min_distance = 0.0;

        for candidate_idx in 0..n_candidates {
            // Generate one LHS candidate
            let mut samples = vec![vec![0.0; n_params]; n_samples];

            for (param_idx, &(min, max)) in self.bounds.iter().enumerate() {
                // Create shuffled indices for this parameter
                let mut indices: Vec<usize> = (0..n_samples).collect();
                for i in (1..n_samples).rev() {
                    let j = rng.random_range(0..=i);
                    indices.swap(i, j);
                }

                // Assign values within bins
                let bin_size = (max - min) / n_samples as f64;

                for (sample_idx, &bin_idx) in indices.iter().enumerate() {
                    let bin_start = min + bin_idx as f64 * bin_size;
                    let bin_end = bin_start + bin_size;
                    let value = rng.random_range(bin_start..bin_end);
                    samples[sample_idx][param_idx] = value;
                }
            }

            // Calculate minimum pairwise distance (maximin criterion)
            let min_distance = self.calculate_min_pairwise_distance(&samples);

            if candidate_idx == 0 || min_distance > best_min_distance {
                best_min_distance = min_distance;
                best_samples = samples;
            }
        }

        // Calculate quality metrics
        let (min_dist, avg_dist, max_dist) = self.calculate_distance_statistics(&best_samples);

        log::info!(
            "{} 🎲 Enhanced LHS: {} samples, {} params | Min dist: {:.4}, Avg dist: {:.4}, Max dist: {:.4}",
            prefix,
            n_samples,
            n_params,
            min_dist,
            avg_dist,
            max_dist
        );

        best_samples
    }

    /// Calculate minimum pairwise distance between samples (for maximin criterion)
    fn calculate_min_pairwise_distance(&self, samples: &[Vec<f64>]) -> f64 {
        if samples.len() < 2 {
            return 0.0;
        }

        let mut min_distance = f64::INFINITY;

        for i in 0..samples.len() {
            for j in (i + 1)..samples.len() {
                let dist = self.squared_distance(&samples[i], &samples[j]).sqrt();
                if dist < min_distance {
                    min_distance = dist;
                }
            }
        }

        min_distance
    }

    /// Calculate distance statistics for quality assessment
    fn calculate_distance_statistics(&self, samples: &[Vec<f64>]) -> (f64, f64, f64) {
        if samples.len() < 2 {
            return (0.0, 0.0, 0.0);
        }

        let mut distances = Vec::new();

        for i in 0..samples.len() {
            for j in (i + 1)..samples.len() {
                let dist = self.squared_distance(&samples[i], &samples[j]).sqrt();
                distances.push(dist);
            }
        }

        if distances.is_empty() {
            return (0.0, 0.0, 0.0);
        }

        let min_dist = distances.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_dist = distances.iter().fold(0.0_f64, |a, &b| a.max(b));
        let avg_dist = distances.iter().sum::<f64>() / distances.len() as f64;

        (min_dist, avg_dist, max_dist)
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

    /// Optimize acquisition function to find next best point with QUALITY-FIRST approach
    /// Uses 250k evaluations (50 restarts × 5000 candidates) for thorough exploration
    fn optimize_acquisition(&self, gp: &GaussianProcess) -> Result<Vec<f64>> {
        // Use seeded RNG if seed is provided, otherwise random
        let mut rng: Box<dyn rand::RngCore> = match self.seed {
            Some(0) | None => {
                // seed=0 or None means random
                Box::new(rand::rng())
            }
            Some(seed_value) => {
                // Use seeded RNG for reproducibility
                // Add offset to avoid same samples as LHS
                Box::new(StdRng::seed_from_u64(seed_value.wrapping_add(1000)))
            }
        };

        let n_restarts = 50; // Increased from 10 for quality
        let n_candidates = 5000; // Increased from 2000 for quality

        let mut best_params = Vec::new();
        let mut best_acquisition_value = f64::NEG_INFINITY;
        let mut all_acquisition_values = Vec::new();

        // Find best of current observations
        let best_y = self
            .observations_y
            .iter()
            .fold(f64::INFINITY, |a, &b| a.min(b));

        for restart_idx in 0..n_restarts {
            // Generate random candidates for this restart
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

                // Add diversity penalty to avoid clustering near previous observations
                let diversity_penalty = self.calculate_diversity_penalty(&candidate);
                let adjusted_acquisition = acquisition_value * (1.0 - 0.1 * diversity_penalty);

                all_acquisition_values.push(adjusted_acquisition);

                if adjusted_acquisition > best_acquisition_value {
                    best_acquisition_value = adjusted_acquisition;
                    best_params = candidate;
                }
            }

            // Log progress every 10 restarts
            if (restart_idx + 1) % 10 == 0 {
                log::debug!(
                    "  Acquisition optimization: {}/{} restarts, best EI: {:.6}",
                    restart_idx + 1,
                    n_restarts,
                    best_acquisition_value
                );
            }
        }

        // Log acquisition statistics
        if !all_acquisition_values.is_empty() {
            let min_acq = all_acquisition_values
                .iter()
                .fold(f64::INFINITY, |a, &b| a.min(b));
            let max_acq = all_acquisition_values
                .iter()
                .fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            let avg_acq =
                all_acquisition_values.iter().sum::<f64>() / all_acquisition_values.len() as f64;

            log::debug!(
                "📊 Acquisition stats: min={:.6}, avg={:.6}, max={:.6} (from {} evaluations)",
                min_acq,
                avg_acq,
                max_acq,
                all_acquisition_values.len()
            );
        }

        Ok(best_params)
    }

    /// Calculate diversity penalty to avoid clustering near previous observations
    /// Returns 0.0 (far from all observations) to 1.0 (very close to an observation)
    fn calculate_diversity_penalty(&self, candidate: &[f64]) -> f64 {
        if self.observations_x.is_empty() {
            return 0.0;
        }

        // Find minimum distance to any previous observation
        let mut min_distance = f64::INFINITY;
        for obs in &self.observations_x {
            let dist = self.squared_distance(candidate, obs).sqrt();
            if dist < min_distance {
                min_distance = dist;
            }
        }

        // Normalize by parameter space diagonal
        let diagonal = self
            .bounds
            .iter()
            .map(|(min, max)| (max - min).powi(2))
            .sum::<f64>()
            .sqrt();

        let normalized_distance = min_distance / diagonal;

        // Convert to penalty (closer = higher penalty)
        (1.0 - normalized_distance.min(1.0)).clamp(0.0, 1.0)
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
    pub fn log_progress(&self, iteration: usize, prefix: &str) {
        if let Some((best_params, best_score)) = self.get_best() {
            log::info!(
                "{} 🔍 Iteration {}: Best Score = {:.6}",
                prefix,
                iteration,
                best_score
            );

            for (name, &value) in self.param_names.iter().zip(best_params.iter()) {
                log::debug!("{}     {}: {:.6}", prefix, name, value);
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
