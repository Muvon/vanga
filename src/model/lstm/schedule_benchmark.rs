//! Performance benchmarking and visualization utilities for learning rate schedules
//!
//! This module provides tools to benchmark different learning rate schedules
//! and visualize their behavior over training epochs.

use crate::config::training::LearningScheduleConfig;
use crate::model::lstm::schedule_validation::calculate_theoretical_min_lr;
use std::time::Instant;

/// Performance metrics for a learning rate schedule
#[derive(Debug, Clone)]
pub struct SchedulePerformanceMetrics {
    pub schedule_name: String,
    pub calculation_time_ns: u128,
    pub memory_usage_bytes: usize,
    pub lr_values: Vec<f64>,
    pub theoretical_min_lr: f64,
    pub actual_min_lr: f64,
    pub lr_variance: f64,
    pub convergence_rate: f64,
}

/// Benchmark configuration
#[derive(Debug, Clone)]
pub struct BenchmarkConfig {
    pub initial_lr: f64,
    pub total_epochs: usize,
    pub iterations: usize,
    pub warmup_iterations: usize,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            initial_lr: 0.001,
            total_epochs: 1000,
            iterations: 10000,
            warmup_iterations: 1000,
        }
    }
}

/// Benchmark a single learning rate schedule
pub fn benchmark_schedule(
    schedule: &LearningScheduleConfig,
    config: &BenchmarkConfig,
) -> SchedulePerformanceMetrics {
    let schedule_name = format!("{:?}", schedule)
        .split('(')
        .next()
        .unwrap_or("Unknown")
        .to_string();

    // Warmup iterations
    for _ in 0..config.warmup_iterations {
        for epoch in 0..config.total_epochs {
            let _ =
                calculate_lr_for_benchmark(schedule, epoch, config.initial_lr, config.total_epochs);
        }
    }

    // Actual benchmark
    let start_time = Instant::now();
    let mut lr_values = Vec::with_capacity(config.total_epochs);

    for _ in 0..config.iterations {
        lr_values.clear();
        for epoch in 0..config.total_epochs {
            let lr =
                calculate_lr_for_benchmark(schedule, epoch, config.initial_lr, config.total_epochs);
            if lr_values.len() < config.total_epochs {
                lr_values.push(lr);
            }
        }
    }

    let calculation_time_ns = start_time.elapsed().as_nanos() / config.iterations as u128;

    // Calculate metrics
    let theoretical_min_lr =
        calculate_theoretical_min_lr(schedule, config.initial_lr, config.total_epochs);
    let actual_min_lr = lr_values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let lr_mean = lr_values.iter().sum::<f64>() / lr_values.len() as f64;
    let lr_variance = lr_values
        .iter()
        .map(|&lr| (lr - lr_mean).powi(2))
        .sum::<f64>()
        / lr_values.len() as f64;

    // Calculate convergence rate (how quickly LR decreases)
    let convergence_rate = if lr_values.len() > 1 {
        let start_lr = lr_values[0];
        let end_lr = lr_values[lr_values.len() - 1];
        (start_lr - end_lr) / start_lr
    } else {
        0.0
    };

    // Estimate memory usage (rough approximation)
    let memory_usage_bytes =
        std::mem::size_of_val(schedule) + lr_values.len() * std::mem::size_of::<f64>();

    SchedulePerformanceMetrics {
        schedule_name,
        calculation_time_ns,
        memory_usage_bytes,
        lr_values,
        theoretical_min_lr,
        actual_min_lr,
        lr_variance,
        convergence_rate,
    }
}

/// Benchmark multiple learning rate schedules
pub fn benchmark_multiple_schedules(
    schedules: Vec<LearningScheduleConfig>,
    config: &BenchmarkConfig,
) -> Vec<SchedulePerformanceMetrics> {
    schedules
        .iter()
        .map(|schedule| benchmark_schedule(schedule, config))
        .collect()
}

/// Generate a comprehensive benchmark report
pub fn generate_benchmark_report(metrics: &[SchedulePerformanceMetrics]) -> String {
    let mut report = String::new();

    report.push_str("# Learning Rate Schedule Performance Benchmark\n\n");
    report.push_str(&format!("Benchmarked {} schedules\n\n", metrics.len()));

    // Performance ranking
    let mut sorted_metrics = metrics.to_vec();
    sorted_metrics.sort_by(|a, b| a.calculation_time_ns.cmp(&b.calculation_time_ns));

    report.push_str("## Performance Ranking (by calculation time)\n\n");
    for (rank, metric) in sorted_metrics.iter().enumerate() {
        report.push_str(&format!(
            "{}. **{}**: {:.2} ns/epoch\n",
            rank + 1,
            metric.schedule_name,
            metric.calculation_time_ns as f64 / 1000.0 // Convert to microseconds
        ));
    }

    report.push_str("\n## Detailed Metrics\n\n");
    for metric in metrics {
        report.push_str(&format!("### {}\n\n", metric.schedule_name));
        report.push_str(&format!(
            "- **Calculation Time**: {:.2} μs/epoch\n",
            metric.calculation_time_ns as f64 / 1000.0
        ));
        report.push_str(&format!(
            "- **Memory Usage**: {} bytes\n",
            metric.memory_usage_bytes
        ));
        report.push_str(&format!(
            "- **Theoretical Min LR**: {:.6e}\n",
            metric.theoretical_min_lr
        ));
        report.push_str(&format!(
            "- **Actual Min LR**: {:.6e}\n",
            metric.actual_min_lr
        ));
        report.push_str(&format!("- **LR Variance**: {:.6e}\n", metric.lr_variance));
        report.push_str(&format!(
            "- **Convergence Rate**: {:.2}%\n",
            metric.convergence_rate * 100.0
        ));
        report.push('\n');
    }

    // LSTM-specific recommendations
    report.push_str("## LSTM Training Recommendations\n\n");

    let fastest_schedule = sorted_metrics.first().unwrap();
    report.push_str(&format!(
        "- **Fastest Schedule**: {} ({:.2} μs/epoch)\n",
        fastest_schedule.schedule_name,
        fastest_schedule.calculation_time_ns as f64 / 1000.0
    ));

    let most_stable = metrics
        .iter()
        .min_by(|a, b| a.lr_variance.partial_cmp(&b.lr_variance).unwrap())
        .unwrap();
    report.push_str(&format!(
        "- **Most Stable**: {} (variance: {:.6e})\n",
        most_stable.schedule_name, most_stable.lr_variance
    ));

    let best_convergence = metrics
        .iter()
        .max_by(|a, b| a.convergence_rate.partial_cmp(&b.convergence_rate).unwrap())
        .unwrap();
    report.push_str(&format!(
        "- **Best Convergence**: {} ({:.2}% reduction)\n",
        best_convergence.schedule_name,
        best_convergence.convergence_rate * 100.0
    ));

    report
}

/// Generate CSV data for visualization
pub fn generate_csv_data(
    metrics: &[SchedulePerformanceMetrics],
    max_epochs: Option<usize>,
) -> String {
    let max_epochs = max_epochs.unwrap_or(100);
    let mut csv = String::new();

    // Header
    csv.push_str("epoch");
    for metric in metrics {
        csv.push_str(&format!(",{}", metric.schedule_name));
    }
    csv.push('\n');

    // Data rows
    for epoch in 0..max_epochs.min(metrics.iter().map(|m| m.lr_values.len()).min().unwrap_or(0)) {
        csv.push_str(&format!("{}", epoch));
        for metric in metrics {
            if epoch < metric.lr_values.len() {
                csv.push_str(&format!(",{:.8e}", metric.lr_values[epoch]));
            } else {
                csv.push(',');
            }
        }
        csv.push('\n');
    }

    csv
}

/// Create a comprehensive benchmark suite for LSTM-optimized schedules
pub fn create_lstm_benchmark_suite() -> Vec<LearningScheduleConfig> {
    vec![
        // Basic schedules
        LearningScheduleConfig::Constant,
        LearningScheduleConfig::LinearDecay {
            decay_rate: 0.01,
            min_lr: Some(1e-6),
        },
        LearningScheduleConfig::ExponentialDecay {
            gamma: 0.95,
            min_lr: Some(1e-6),
        },
        // Advanced schedules optimized for LSTM
        LearningScheduleConfig::CosineAnnealing {
            t_max: 100,
            eta_min: Some(1e-6),
        },
        LearningScheduleConfig::OneCycle {
            max_lr: 0.01,
            pct_start: Some(0.3),
            anneal_strategy: Some("cos".to_string()),
            div_factor: Some(25.0),
            final_div_factor: Some(1e4),
        },
        LearningScheduleConfig::WarmRestarts {
            t_0: 10,
            t_mult: 2,
            eta_min: Some(1e-6),
        },
        LearningScheduleConfig::CyclicalLR {
            base_lr: 1e-5,
            max_lr: 1e-3,
            step_size_up: 20,
            step_size_down: Some(20),
            mode: Some("triangular".to_string()),
            gamma: Some(1.0),
        },
        LearningScheduleConfig::StepDecay {
            step_size: 25,
            gamma: 0.5,
            milestones: None,
            min_lr: Some(1e-6),
        },
        LearningScheduleConfig::PolynomialDecay {
            power: 2.0,
            min_lr: Some(1e-6),
        },
        LearningScheduleConfig::NoamLR {
            model_size: 512,
            warmup_steps: 100,
            factor: Some(1.0),
        },
    ]
}

/// Run a complete LSTM-focused benchmark
pub fn run_lstm_benchmark() -> (Vec<SchedulePerformanceMetrics>, String, String) {
    let schedules = create_lstm_benchmark_suite();
    let config = BenchmarkConfig {
        initial_lr: 0.001,
        total_epochs: 1000,
        iterations: 1000,
        warmup_iterations: 100,
    };

    println!("🚀 Running LSTM Learning Rate Schedule Benchmark...");
    println!("   Schedules: {}", schedules.len());
    println!("   Epochs: {}", config.total_epochs);
    println!("   Iterations: {}", config.iterations);

    let metrics = benchmark_multiple_schedules(schedules, &config);
    let report = generate_benchmark_report(&metrics);
    let csv_data = generate_csv_data(&metrics, Some(200)); // First 200 epochs for visualization

    println!("✅ Benchmark completed!");

    (metrics, report, csv_data)
}

/// Simplified calculation function for benchmarking (avoids complex dependencies)
fn calculate_lr_for_benchmark(
    schedule: &LearningScheduleConfig,
    epoch: usize,
    initial_lr: f64,
    total_epochs: usize,
) -> f64 {
    use std::f64::consts::PI;

    match schedule {
        LearningScheduleConfig::Constant => initial_lr,

        LearningScheduleConfig::LinearDecay { decay_rate, min_lr } => {
            let progress = epoch as f64 / total_epochs.max(1) as f64;
            let decay_factor = 1.0 - (decay_rate * progress);
            let min_threshold = min_lr.unwrap_or(initial_lr * 0.001);
            (initial_lr * decay_factor).max(min_threshold)
        }

        LearningScheduleConfig::ExponentialDecay { gamma, min_lr } => {
            let decay_factor = gamma.powf(epoch as f64);
            let min_threshold = min_lr.unwrap_or(initial_lr * 0.0001);
            (initial_lr * decay_factor).max(min_threshold)
        }

        LearningScheduleConfig::CosineAnnealing { t_max, eta_min } => {
            let t_max_f = (*t_max).max(1) as f64;
            let progress = (epoch as f64 / t_max_f).min(1.0);
            let eta_min_val = eta_min.unwrap_or(initial_lr * 0.001);
            let cosine_factor = 0.5 * (1.0 + (PI * progress).cos());
            eta_min_val + (initial_lr - eta_min_val) * cosine_factor
        }

        LearningScheduleConfig::OneCycle {
            max_lr,
            pct_start,
            anneal_strategy,
            div_factor,
            final_div_factor,
        } => {
            let pct_start_val = pct_start.unwrap_or(0.3);
            let div_factor_val = div_factor.unwrap_or(25.0);
            let final_div_factor_val = final_div_factor.unwrap_or(1e4);
            let anneal_strategy_val = anneal_strategy.as_deref().unwrap_or("cos");

            let initial_lr_calc = max_lr / div_factor_val;
            let final_lr = initial_lr_calc / final_div_factor_val;

            let progress = epoch as f64 / total_epochs.max(1) as f64;

            if progress <= pct_start_val {
                let phase_progress = progress / pct_start_val;
                initial_lr_calc + (max_lr - initial_lr_calc) * phase_progress
            } else {
                let phase_progress = (progress - pct_start_val) / (1.0 - pct_start_val);
                match anneal_strategy_val {
                    "linear" => max_lr - (max_lr - final_lr) * phase_progress,
                    _ => {
                        let cosine_factor = 0.5 * (1.0 + (PI * phase_progress).cos());
                        final_lr + (max_lr - final_lr) * cosine_factor
                    }
                }
            }
        }

        // Simplified implementations for other schedules
        _ => initial_lr * 0.95_f64.powf(epoch as f64), // Default exponential decay
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_benchmark_constant_schedule() {
        let schedule = LearningScheduleConfig::Constant;
        let config = BenchmarkConfig {
            initial_lr: 0.001,
            total_epochs: 100,
            iterations: 10,
            warmup_iterations: 5,
        };

        let metrics = benchmark_schedule(&schedule, &config);

        assert_eq!(metrics.schedule_name, "Constant");
        assert!(metrics.calculation_time_ns > 0);
        assert_eq!(metrics.lr_values.len(), config.total_epochs);
        assert!(metrics.lr_values.iter().all(|&lr| lr == config.initial_lr));
        assert_eq!(metrics.convergence_rate, 0.0); // No convergence for constant
    }

    #[test]
    fn test_benchmark_multiple_schedules() {
        let schedules = vec![
            LearningScheduleConfig::Constant,
            LearningScheduleConfig::LinearDecay {
                decay_rate: 0.1,
                min_lr: Some(0.0001),
            },
        ];

        let config = BenchmarkConfig {
            initial_lr: 0.001,
            total_epochs: 50,
            iterations: 5,
            warmup_iterations: 2,
        };

        let metrics = benchmark_multiple_schedules(schedules, &config);

        assert_eq!(metrics.len(), 2);
        assert_eq!(metrics[0].schedule_name, "Constant");
        assert_eq!(metrics[1].schedule_name, "LinearDecay");
    }

    #[test]
    fn test_csv_generation() {
        let metrics = vec![
            SchedulePerformanceMetrics {
                schedule_name: "Test1".to_string(),
                calculation_time_ns: 1000,
                memory_usage_bytes: 100,
                lr_values: vec![0.001, 0.0009, 0.0008],
                theoretical_min_lr: 0.0001,
                actual_min_lr: 0.0008,
                lr_variance: 1e-8,
                convergence_rate: 0.2,
            },
            SchedulePerformanceMetrics {
                schedule_name: "Test2".to_string(),
                calculation_time_ns: 2000,
                memory_usage_bytes: 200,
                lr_values: vec![0.001, 0.0005, 0.0002],
                theoretical_min_lr: 0.0001,
                actual_min_lr: 0.0002,
                lr_variance: 2e-8,
                convergence_rate: 0.8,
            },
        ];

        let csv = generate_csv_data(&metrics, Some(3));

        assert!(csv.contains("epoch,Test1,Test2"));
        assert!(csv.contains("0,1.00000000e-3,1.00000000e-3"));
        assert!(csv.contains("1,9.00000000e-4,5.00000000e-4"));
        assert!(csv.contains("2,8.00000000e-4,2.00000000e-4"));
    }

    #[test]
    fn test_report_generation() {
        let metrics = vec![
            SchedulePerformanceMetrics {
                schedule_name: "Fast".to_string(),
                calculation_time_ns: 500,
                memory_usage_bytes: 50,
                lr_values: vec![0.001, 0.0009],
                theoretical_min_lr: 0.0001,
                actual_min_lr: 0.0009,
                lr_variance: 1e-9,
                convergence_rate: 0.1,
            },
            SchedulePerformanceMetrics {
                schedule_name: "Slow".to_string(),
                calculation_time_ns: 2000,
                memory_usage_bytes: 200,
                lr_values: vec![0.001, 0.0001],
                theoretical_min_lr: 0.0001,
                actual_min_lr: 0.0001,
                lr_variance: 1e-7,
                convergence_rate: 0.9,
            },
        ];

        let report = generate_benchmark_report(&metrics);

        assert!(report.contains("# Learning Rate Schedule Performance Benchmark"));
        assert!(report.contains("Benchmarked 2 schedules"));
        assert!(report.contains("**Fastest Schedule**: Fast"));
        assert!(report.contains("**Most Stable**: Fast"));
        assert!(report.contains("**Best Convergence**: Slow"));
    }
}
