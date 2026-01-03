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

/// Trust Region for local Bayesian optimization (TuRBO-inspired)
#[derive(Debug, Clone)]
pub struct TrustRegion {
    /// Center of trust region
    pub center: Vec<f64>,
    /// Radius of trust region (normalized to [0, 1])
    pub radius: f64,
    /// Success counter (improvements found)
    pub success_counter: usize,
    /// Failure counter (no improvements)
    pub failure_counter: usize,
    /// Initial radius
    initial_radius: f64,
    /// Minimum radius before restart
    min_radius: f64,
}

impl TrustRegion {
    /// Create new trust region centered at given point
    pub fn new(center: Vec<f64>, initial_radius: f64) -> Self {
        Self {
            center,
            radius: initial_radius,
            success_counter: 0,
            failure_counter: 0,
            initial_radius,
            min_radius: initial_radius * 0.01, // 1% of initial
        }
    }

    /// Expand trust region after success
    pub fn expand(&mut self) {
        self.success_counter += 1;
        self.failure_counter = 0;

        // Expand by 2x after 3 consecutive successes
        if self.success_counter >= 3 {
            self.radius = (self.radius * 2.0).min(1.0);
            self.success_counter = 0;
            log::debug!("🔼 Trust region EXPANDED to radius {:.4}", self.radius);
        }
    }

    /// Shrink trust region after failure
    pub fn shrink(&mut self) {
        self.failure_counter += 1;
        self.success_counter = 0;

        // Shrink by 0.5x after 3 consecutive failures
        if self.failure_counter >= 3 {
            self.radius *= 0.5;
            self.failure_counter = 0;
            log::debug!("🔽 Trust region SHRUNK to radius {:.4}", self.radius);
        }
    }

    /// Check if trust region is too small and needs restart
    pub fn needs_restart(&self) -> bool {
        self.radius < self.min_radius
    }

    /// Restart trust region at new random location
    pub fn restart(&mut self, new_center: Vec<f64>) {
        log::info!(
            "🔄 Trust region RESTART at new location (old radius: {:.4})",
            self.radius
        );
        self.center = new_center;
        self.radius = self.initial_radius;
        self.success_counter = 0;
        self.failure_counter = 0;
    }

    /// Update center to best point found
    pub fn update_center(&mut self, new_center: Vec<f64>) {
        self.center = new_center;
    }

    /// Clip point to trust region bounds
    pub fn clip_to_region(&self, point: &[f64], bounds: &[(f64, f64)]) -> Vec<f64> {
        point
            .iter()
            .zip(self.center.iter())
            .zip(bounds.iter())
            .map(|((&p, &c), &(min, max))| {
                let lower = (c - self.radius * (max - min)).max(min);
                let upper = (c + self.radius * (max - min)).min(max);
                p.clamp(lower, upper)
            })
            .collect()
    }
}

/// Bayesian Optimizer for target calibration
pub struct BayesianOptimizer {
    /// Parameter bounds (min, max) for each parameter
    pub bounds: Vec<(f64, f64)>,
    /// Parameter names for logging
    pub param_names: Vec<String>,
    /// Observed parameter combinations
    observations_x: Vec<Vec<f64>>,
    /// Observed objective values (lower is better)
    observations_y: Vec<f64>,
    /// Gaussian Process hyperparameters
    gp_length_scale: f64,
    gp_noise: f64,
    /// Random seed for reproducible optimization (None = random, Some(0) = random, Some(n) = seeded)
    seed: Option<u64>,
    /// Trust region for local optimization (TuRBO-inspired)
    trust_region: Option<TrustRegion>,
    /// Enable trust region optimization
    enable_trust_regions: bool,
    /// Best score history for stagnation detection
    best_score_history: Vec<f64>,
    /// Stagnation detection window
    stagnation_window: usize,
    /// Enable adaptive restart
    enable_adaptive_restart: bool,
    /// Restart counter
    restart_count: usize,
    /// Current acquisition function (can change during optimization)
    current_acquisition: AcquisitionFunction,
    /// Iteration counter
    iteration: usize,
}

/// Acquisition function types for Bayesian optimization
#[derive(Debug, Clone)]
pub enum AcquisitionFunction {
    /// Expected Improvement (default, balanced exploration/exploitation)
    ExpectedImprovement,
    /// Upper Confidence Bound (more exploration)
    UpperConfidenceBound { kappa: f64 },
    /// Thompson Sampling (samples from GP posterior, excellent exploration)
    ThompsonSampling,
    /// Epsilon-Greedy Thompson Sampling (2024 paper: robust hybrid approach)
    /// With probability epsilon: random exploration, with 1-epsilon: Thompson Sampling
    EpsilonGreedyThompsonSampling { epsilon: f64 },
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
    /// Enable trust region optimization (TuRBO-inspired)
    pub enable_trust_regions: bool,
    /// Enable adaptive restart when stuck
    pub enable_adaptive_restart: bool,
    /// Stagnation detection window (iterations without improvement)
    pub stagnation_window: usize,
    /// Batch size for parallel acquisition (1 = sequential)
    pub batch_size: usize,
}

impl Default for BayesianConfig {
    /// Default configuration with STATE-OF-THE-ART 2024-2025 research
    /// Based on: Epsilon-Greedy TS (2024), TuRBO (NeurIPS 2019), Trust Regions (2024)
    /// Optimized for QUALITY + EXPLORATION with adaptive restart
    fn default() -> Self {
        Self {
            n_initial: 30,       // 6D optimal for 5 params (5*D + 5)
            max_iterations: 150, // More time for quality exploration
            tolerance: 1e-4,     // Adaptive tolerance
            acquisition: AcquisitionFunction::EpsilonGreedyThompsonSampling { epsilon: 0.3 }, // 30% exploration
            gp_length_scale: 0.8, // Slightly shorter for local structure
            gp_noise: 1e-5,       // Lower noise for deterministic objectives
            enable_trust_regions: true,
            enable_adaptive_restart: true,
            stagnation_window: 15,
            batch_size: 1, // Sequential by default
        }
    }
}

impl BayesianConfig {
    /// Configuration for high-dimensional spaces (6D sentiment with 4 weights + 2 thresholds)
    /// Uses more initial samples and iterations for complex parameter spaces
    pub fn for_high_dimensional() -> Self {
        Self {
            n_initial: 40,       // More samples for 6D space
            max_iterations: 200, // More iterations for complex optimization
            tolerance: 1e-5,
            acquisition: AcquisitionFunction::EpsilonGreedyThompsonSampling { epsilon: 0.35 },
            gp_length_scale: 0.5,
            gp_noise: 1e-5,
            enable_trust_regions: true,
            enable_adaptive_restart: true,
            stagnation_window: 20,
            batch_size: 1,
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
            enable_trust_regions: false,
            enable_adaptive_restart: false,
            stagnation_window: 10,
            batch_size: 1,
        }
    }

    /// Configuration for maximum quality (research/final calibration)
    /// Uses extensive exploration with Epsilon-Greedy TS and trust regions
    pub fn for_maximum_quality() -> Self {
        Self {
            n_initial: 50,       // Extensive initial exploration
            max_iterations: 300, // Maximum exploration time
            tolerance: 1e-5,     // Very strict convergence
            acquisition: AcquisitionFunction::EpsilonGreedyThompsonSampling { epsilon: 0.4 }, // 40% exploration
            gp_length_scale: 0.8,
            gp_noise: 1e-6, // Very low noise
            enable_trust_regions: true,
            enable_adaptive_restart: true,
            stagnation_window: 25,
            batch_size: 3, // Batch parallel for quality
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
            seed,
            trust_region: None,
            enable_trust_regions: config.enable_trust_regions,
            best_score_history: Vec::new(),
            stagnation_window: config.stagnation_window,
            enable_adaptive_restart: config.enable_adaptive_restart,
            restart_count: 0,
            current_acquisition: config.acquisition.clone(),
            iteration: 0,
        }
    }

    /// Get the random seed for reproducibility
    pub fn seed(&self) -> Option<u64> {
        self.seed
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
        // Research: Fewer candidates = less diversity = more room for Bayesian exploration
        let n_candidates = 2; // Reduced from 5 to allow Bayesian phase to explore
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
        // Update trust region if enabled (before adding observation)
        if self.enable_trust_regions {
            let current_best = self.get_best_score();
            if let Some(ref mut tr) = self.trust_region {
                if score < current_best {
                    // Improvement found
                    tr.expand();
                    tr.update_center(params.clone());
                } else {
                    // No improvement
                    tr.shrink();
                }
            } else {
                // Initialize trust region at first observation
                self.trust_region = Some(TrustRegion::new(params.clone(), 0.3));
                // 30% initial radius
            }
        }

        self.observations_x.push(params);
        self.observations_y.push(score);
        self.best_score_history.push(score);
    }

    /// Get current best score
    fn get_best_score(&self) -> f64 {
        self.observations_y
            .iter()
            .fold(f64::INFINITY, |a, &b| a.min(b))
    }

    /// Detect stagnation (no improvement for N iterations)
    fn detect_stagnation(&self) -> bool {
        if !self.enable_adaptive_restart || self.best_score_history.len() < self.stagnation_window {
            return false;
        }

        let recent_scores =
            &self.best_score_history[self.best_score_history.len() - self.stagnation_window..];
        let best_recent = recent_scores.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let oldest_recent = recent_scores[0];

        // Stagnation if improvement < 0.1% in last N iterations
        let improvement = (oldest_recent - best_recent).abs() / oldest_recent.max(1e-10);
        improvement < 0.001
    }

    /// Handle stagnation with adaptive restart
    fn handle_stagnation(&mut self, prefix: &str) {
        self.restart_count += 1;
        log::warn!(
            "{} ⚠️  STAGNATION DETECTED (restart #{}) - triggering adaptive restart",
            prefix,
            self.restart_count
        );

        // Strategy 1: Restart trust region at new random location
        if self.enable_trust_regions {
            if let Some(ref mut tr) = self.trust_region {
                let mut rng: Box<dyn rand::RngCore> = match self.seed {
                    Some(0) | None => Box::new(rand::rng()),
                    Some(seed_value) => Box::new(StdRng::seed_from_u64(
                        seed_value.wrapping_add(self.restart_count as u64 * 1000),
                    )),
                };

                let new_center: Vec<f64> = self
                    .bounds
                    .iter()
                    .map(|(min, max)| rng.random_range(*min..*max))
                    .collect();

                tr.restart(new_center);
            }
        }

        // Strategy 2: Switch acquisition function
        self.current_acquisition = match &self.current_acquisition {
            AcquisitionFunction::UpperConfidenceBound { .. } => {
                log::info!("{} 🔄 Switching from UCB to Thompson Sampling", prefix);
                AcquisitionFunction::ThompsonSampling
            }
            AcquisitionFunction::ExpectedImprovement => {
                log::info!("{} 🔄 Switching from EI to Epsilon-Greedy TS", prefix);
                AcquisitionFunction::EpsilonGreedyThompsonSampling { epsilon: 0.4 }
            }
            AcquisitionFunction::ThompsonSampling => {
                log::info!(
                    "{} 🔄 Switching from TS to UCB with high exploration",
                    prefix
                );
                AcquisitionFunction::UpperConfidenceBound { kappa: 3.0 }
            }
            AcquisitionFunction::EpsilonGreedyThompsonSampling { .. } => {
                log::info!("{} 🔄 Switching from Epsilon-Greedy TS to pure TS", prefix);
                AcquisitionFunction::ThompsonSampling
            }
        };
    }

    /// Suggest next parameter combination to evaluate
    pub fn suggest_next(&mut self, prefix: &str) -> Result<Vec<f64>> {
        if self.observations_x.is_empty() {
            return Err(VangaError::ConfigError(
                "No observations yet. Call initialize_latin_hypercube() first.".to_string(),
            ));
        }

        self.iteration += 1;

        // Check for stagnation and handle adaptively
        if self.detect_stagnation() {
            self.handle_stagnation(prefix);
        }

        // Check if trust region needs restart
        if self.enable_trust_regions {
            if let Some(ref mut tr) = self.trust_region {
                if tr.needs_restart() {
                    let mut rng: Box<dyn rand::RngCore> = match self.seed {
                        Some(0) | None => Box::new(rand::rng()),
                        Some(seed_value) => Box::new(StdRng::seed_from_u64(
                            seed_value.wrapping_add(self.restart_count as u64 * 1000),
                        )),
                    };

                    let new_center: Vec<f64> = self
                        .bounds
                        .iter()
                        .map(|(min, max)| rng.random_range(*min..*max))
                        .collect();

                    tr.restart(new_center);
                }
            }
        }

        // Build Gaussian Process model
        let gp = self.build_gaussian_process()?;

        // Optimize acquisition function to find next best point
        let next_params = self.optimize_acquisition(&gp, prefix)?;

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
    /// Uses adaptive evaluation budget: 250k when exploring, 100k when exploiting
    fn optimize_acquisition(&self, gp: &GaussianProcess, prefix: &str) -> Result<Vec<f64>> {
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

        // Adaptive evaluation budget based on trust region state
        let (n_restarts, n_candidates) = if self.enable_trust_regions {
            if let Some(ref tr) = self.trust_region {
                if tr.radius < 0.1 {
                    // Small trust region = local exploitation = fewer evaluations
                    (20, 2000) // 40k evaluations
                } else {
                    // Large trust region = exploration = more evaluations
                    (50, 5000) // 250k evaluations (QUALITY-FIRST)
                }
            } else {
                (50, 5000) // Default: 250k evaluations
            }
        } else {
            (50, 5000) // No trust regions: always use 250k evaluations
        };

        log::debug!(
            "{} 🔍 Acquisition optimization: {} restarts × {} candidates = {} evaluations",
            prefix,
            n_restarts,
            n_candidates,
            n_restarts * n_candidates
        );

        let mut best_params = Vec::new();
        let mut best_acquisition_value = f64::NEG_INFINITY;
        let mut all_acquisition_values = Vec::new();

        // Find best of current observations
        let best_y = self
            .observations_y
            .iter()
            .fold(f64::INFINITY, |a, &b| a.min(b));

        // Decay epsilon for Epsilon-Greedy TS
        let current_epsilon =
            if let AcquisitionFunction::EpsilonGreedyThompsonSampling { epsilon } =
                &self.current_acquisition
            {
                // Decay from initial epsilon to 0.1 over iterations
                let decay_rate = 0.7; // Decay to 70% of initial
                epsilon
                    * (1.0 - decay_rate * (self.iteration as f64 / 150.0).min(1.0))
                        .max(0.1 / epsilon)
            } else {
                0.0
            };

        for restart_idx in 0..n_restarts {
            // Generate random candidates for this restart
            for _ in 0..n_candidates {
                let mut candidate = Vec::new();

                // Generate candidate within trust region if enabled
                if self.enable_trust_regions {
                    if let Some(ref tr) = self.trust_region {
                        // Sample within trust region
                        for (i, (min, max)) in self.bounds.iter().enumerate() {
                            let center = tr.center[i];
                            let lower = (center - tr.radius * (max - min)).max(*min);
                            let upper = (center + tr.radius * (max - min)).min(*max);
                            candidate.push(rng.random_range(lower..upper));
                        }
                    } else {
                        // No trust region yet, sample from full bounds
                        for (min, max) in &self.bounds {
                            candidate.push(rng.random_range(*min..*max));
                        }
                    }
                } else {
                    // No trust regions, sample from full bounds
                    for (min, max) in &self.bounds {
                        candidate.push(rng.random_range(*min..*max));
                    }
                }

                // Calculate acquisition function value
                let acquisition_value = match &self.current_acquisition {
                    AcquisitionFunction::ExpectedImprovement => {
                        let (mean, std) = gp.predict(&candidate)?;
                        self.expected_improvement(mean, std, best_y)
                    }
                    AcquisitionFunction::UpperConfidenceBound { kappa } => {
                        let (mean, std) = gp.predict(&candidate)?;
                        // For MINIMIZATION: we want to explore low mean with high uncertainty
                        // Negate to convert to maximization problem for acquisition
                        -(mean - kappa * std)
                    }
                    AcquisitionFunction::ThompsonSampling => {
                        // Sample from GP posterior
                        let sample_value = gp.sample_from_posterior(&candidate, &mut rng)?;
                        // Negate for minimization (we want low values)
                        -sample_value
                    }
                    AcquisitionFunction::EpsilonGreedyThompsonSampling { .. } => {
                        // Epsilon-greedy: with probability epsilon, pure exploration
                        if rng.random_range(0.0..1.0) < current_epsilon {
                            // Pure exploration: random value (all candidates equally good)
                            rng.random_range(0.0..1.0)
                        } else {
                            // Thompson Sampling
                            let sample_value = gp.sample_from_posterior(&candidate, &mut rng)?;
                            -sample_value
                        }
                    }
                };

                // Add diversity penalty and novelty bonus
                let diversity_penalty = self.calculate_diversity_penalty(&candidate);
                let novelty_score = self.calculate_novelty_score(&candidate);
                let adjusted_acquisition = acquisition_value
                    * (1.0 - 0.05 * diversity_penalty)
                    * (1.0 + 0.1 * novelty_score);

                all_acquisition_values.push(adjusted_acquisition);

                if adjusted_acquisition > best_acquisition_value {
                    best_acquisition_value = adjusted_acquisition;
                    best_params = candidate;
                }
            }

            // Log progress every 10 restarts
            if (restart_idx + 1) % 10 == 0 {
                log::debug!(
                    "{}   Acquisition optimization: {}/{} restarts, best value: {:.6}",
                    prefix,
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
                "{} 📊 Acquisition stats: min={:.6}, avg={:.6}, max={:.6} (from {} evaluations)",
                prefix,
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

    /// Calculate novelty score (reward for being far from ALL observations)
    /// Returns 0.0 (clustered with observations) to 1.0 (very novel region)
    fn calculate_novelty_score(&self, candidate: &[f64]) -> f64 {
        if self.observations_x.is_empty() {
            return 1.0; // Maximum novelty if no observations
        }

        // Calculate distances to all observations
        let mut distances: Vec<f64> = self
            .observations_x
            .iter()
            .map(|obs| self.squared_distance(candidate, obs).sqrt())
            .collect();

        // Sort to find K nearest neighbors
        distances.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // Average distance to K=5 nearest neighbors
        let k = 5.min(distances.len());
        let avg_distance = distances.iter().take(k).sum::<f64>() / k as f64;

        // Normalize by parameter space diagonal
        let diagonal = self
            .bounds
            .iter()
            .map(|(min, max)| (max - min).powi(2))
            .sum::<f64>()
            .sqrt();

        let normalized_distance = avg_distance / diagonal;

        // Convert to novelty score (farther = higher novelty)
        normalized_distance.min(1.0).clamp(0.0, 1.0)
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

    /// Sample from GP posterior at given point (Thompson Sampling)
    /// Returns a sample from N(mean, variance) distribution
    fn sample_from_posterior(
        &self,
        x_new: &[f64],
        rng: &mut Box<dyn rand::RngCore>,
    ) -> Result<f64> {
        let (mean, std) = self.predict(x_new)?;

        // Sample from N(0, 1)
        let z = self.sample_standard_normal(rng);

        // Transform to N(mean, std^2)
        let sample = mean + std * z;

        Ok(sample)
    }

    /// Sample from standard normal distribution N(0, 1) using Box-Muller transform
    fn sample_standard_normal(&self, rng: &mut Box<dyn rand::RngCore>) -> f64 {
        let u1: f64 = rng.random_range(1e-10..1.0); // Avoid log(0)
        let u2: f64 = rng.random_range(0.0..1.0);

        // Box-Muller transform
        (-2.0_f64 * u1.ln()).sqrt() * (2.0_f64 * std::f64::consts::PI * u2).cos()
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
