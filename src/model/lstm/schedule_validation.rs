//! Mathematical validation functions for learning rate schedules
//!
//! This module provides comprehensive validation functions to ensure
//! learning rate schedules are mathematically correct and follow best practices.

use crate::config::training::LearningScheduleConfig;
use crate::utils::error::{Result, VangaError};

/// Comprehensive validation for learning schedule configurations
pub fn validate_learning_schedule(config: &LearningScheduleConfig) -> Result<()> {
    match config {
        LearningScheduleConfig::Constant => {
            // No validation needed for constant schedule
            Ok(())
        }

        LearningScheduleConfig::ReduceOnPlateau {
            patience,
            factor,
            min_lr,
            monitor,
            threshold,
        } => {
            validate_positive_u32(*patience, "patience")?;
            validate_factor(*factor)?;
            validate_optional_positive_f64(*min_lr, "min_lr")?;
            validate_monitor_metric(monitor.as_deref())?;
            validate_optional_positive_f64(*threshold, "threshold")?;
            Ok(())
        }

        LearningScheduleConfig::LinearDecay { decay_rate, min_lr } => {
            validate_decay_rate(*decay_rate)?;
            validate_optional_positive_f64(*min_lr, "min_lr")?;
            Ok(())
        }

        LearningScheduleConfig::ExponentialDecay { gamma, min_lr } => {
            validate_gamma(*gamma)?;
            validate_optional_positive_f64(*min_lr, "min_lr")?;
            Ok(())
        }

        LearningScheduleConfig::StepDecay {
            step_size,
            gamma,
            milestones,
            min_lr,
        } => {
            validate_positive_u32(*step_size, "step_size")?;
            validate_gamma(*gamma)?;
            validate_optional_positive_f64(*min_lr, "min_lr")?;
            if let Some(milestones) = milestones {
                validate_milestones(milestones)?;
            }
            Ok(())
        }

        LearningScheduleConfig::PolynomialDecay { power, min_lr } => {
            validate_positive_f64(*power, "power")?;
            validate_optional_positive_f64(*min_lr, "min_lr")?;
            Ok(())
        }

        LearningScheduleConfig::CosineAnnealing { t_max, eta_min } => {
            validate_positive_u32(*t_max, "t_max")?;
            validate_optional_positive_f64(*eta_min, "eta_min")?;
            Ok(())
        }

        LearningScheduleConfig::WarmRestarts {
            t_0,
            t_mult,
            eta_min,
        } => {
            validate_positive_u32(*t_0, "t_0")?;
            validate_positive_u32(*t_mult, "t_mult")?;
            validate_optional_positive_f64(*eta_min, "eta_min")?;
            Ok(())
        }

        LearningScheduleConfig::OneCycle {
            max_lr,
            pct_start,
            anneal_strategy,
            div_factor,
            final_div_factor,
        } => {
            validate_positive_f64(*max_lr, "max_lr")?;
            validate_optional_percentage(*pct_start, "pct_start")?;
            validate_anneal_strategy(anneal_strategy.as_deref())?;
            validate_optional_positive_f64(*div_factor, "div_factor")?;
            validate_optional_positive_f64(*final_div_factor, "final_div_factor")?;
            Ok(())
        }

        LearningScheduleConfig::CyclicalLR {
            base_lr,
            max_lr,
            step_size_up,
            step_size_down,
            mode,
            gamma,
        } => {
            validate_positive_f64(*base_lr, "base_lr")?;
            validate_positive_f64(*max_lr, "max_lr")?;
            if *max_lr <= *base_lr {
                return Err(VangaError::ConfigError(
                    "max_lr must be greater than base_lr for CyclicalLR".to_string(),
                ));
            }
            validate_positive_u32(*step_size_up, "step_size_up")?;
            validate_optional_positive_u32(*step_size_down, "step_size_down")?;
            validate_cyclical_mode(mode.as_deref())?;
            validate_optional_positive_f64(*gamma, "gamma")?;
            Ok(())
        }

        LearningScheduleConfig::NoamLR {
            model_size,
            warmup_steps,
            factor,
        } => {
            validate_positive_u32(*model_size, "model_size")?;
            validate_positive_u32(*warmup_steps, "warmup_steps")?;
            validate_optional_positive_f64(*factor, "factor")?;
            Ok(())
        }
    }
}

/// Validate that a factor is between 0 and 1 (exclusive of 0, inclusive of 1)
fn validate_factor(factor: f64) -> Result<()> {
    if factor <= 0.0 || factor > 1.0 {
        return Err(VangaError::ConfigError(format!(
            "Factor must be in range (0, 1], got: {}",
            factor
        )));
    }
    Ok(())
}

/// Validate decay rate is between 0 and 1
fn validate_decay_rate(decay_rate: f64) -> Result<()> {
    if !(0.0..=1.0).contains(&decay_rate) {
        return Err(VangaError::ConfigError(format!(
            "Decay rate must be in range [0, 1], got: {}",
            decay_rate
        )));
    }
    Ok(())
}

/// Validate gamma parameter (typically between 0 and 1 for decay)
fn validate_gamma(gamma: f64) -> Result<()> {
    if gamma <= 0.0 || gamma > 1.0 {
        return Err(VangaError::ConfigError(format!(
            "Gamma must be in range (0, 1], got: {}",
            gamma
        )));
    }
    Ok(())
}

/// Validate positive f64 value
fn validate_positive_f64(value: f64, name: &str) -> Result<()> {
    if value <= 0.0 || !value.is_finite() {
        return Err(VangaError::ConfigError(format!(
            "{} must be positive and finite, got: {}",
            name, value
        )));
    }
    Ok(())
}

/// Validate optional positive f64 value
fn validate_optional_positive_f64(value: Option<f64>, name: &str) -> Result<()> {
    if let Some(val) = value {
        validate_positive_f64(val, name)?;
    }
    Ok(())
}

/// Validate positive u32 value
fn validate_positive_u32(value: u32, name: &str) -> Result<()> {
    if value == 0 {
        return Err(VangaError::ConfigError(format!(
            "{} must be positive, got: {}",
            name, value
        )));
    }
    Ok(())
}

/// Validate optional positive u32 value
fn validate_optional_positive_u32(value: Option<u32>, name: &str) -> Result<()> {
    if let Some(val) = value {
        validate_positive_u32(val, name)?;
    }
    Ok(())
}

/// Validate percentage value (0.0 to 1.0)
fn validate_optional_percentage(value: Option<f64>, name: &str) -> Result<()> {
    if let Some(val) = value {
        if !(0.0..=1.0).contains(&val) || !val.is_finite() {
            return Err(VangaError::ConfigError(format!(
                "{} must be in range [0, 1], got: {}",
                name, val
            )));
        }
    }
    Ok(())
}

/// Validate monitor metric string
fn validate_monitor_metric(monitor: Option<&str>) -> Result<()> {
    if let Some(metric) = monitor {
        match metric {
            "loss" | "accuracy" | "f1_score" | "precision" | "recall" => Ok(()),
            _ => Err(VangaError::ConfigError(
                format!("Invalid monitor metric: {}. Valid options: loss, accuracy, f1_score, precision, recall", metric)
            ))
        }
    } else {
        Ok(())
    }
}

/// Validate annealing strategy
fn validate_anneal_strategy(strategy: Option<&str>) -> Result<()> {
    if let Some(strat) = strategy {
        match strat {
            "cos" | "linear" => Ok(()),
            _ => Err(VangaError::ConfigError(format!(
                "Invalid anneal strategy: {}. Valid options: cos, linear",
                strat
            ))),
        }
    } else {
        Ok(())
    }
}

/// Validate cyclical mode
fn validate_cyclical_mode(mode: Option<&str>) -> Result<()> {
    if let Some(m) = mode {
        match m {
            "triangular" | "triangular2" | "exp_range" => Ok(()),
            _ => Err(VangaError::ConfigError(format!(
                "Invalid cyclical mode: {}. Valid options: triangular, triangular2, exp_range",
                m
            ))),
        }
    } else {
        Ok(())
    }
}

/// Validate milestones are in ascending order
fn validate_milestones(milestones: &[u32]) -> Result<()> {
    if milestones.is_empty() {
        return Err(VangaError::ConfigError(
            "Milestones cannot be empty".to_string(),
        ));
    }

    for window in milestones.windows(2) {
        if window[0] >= window[1] {
            return Err(VangaError::ConfigError(format!(
                "Milestones must be in ascending order, found {} >= {}",
                window[0], window[1]
            )));
        }
    }
    Ok(())
}

/// Calculate theoretical minimum learning rate for a schedule
pub fn calculate_theoretical_min_lr(
    config: &LearningScheduleConfig,
    initial_lr: f64,
    total_epochs: usize,
) -> f64 {
    match config {
        LearningScheduleConfig::Constant => initial_lr,

        LearningScheduleConfig::ReduceOnPlateau { min_lr, .. } => {
            min_lr.unwrap_or(initial_lr * 0.001)
        }

        LearningScheduleConfig::LinearDecay { min_lr, .. } => min_lr.unwrap_or(initial_lr * 0.001),

        LearningScheduleConfig::ExponentialDecay { gamma, min_lr } => {
            let theoretical_min = initial_lr * gamma.powf(total_epochs as f64);
            min_lr.unwrap_or(theoretical_min.max(initial_lr * 0.0001))
        }

        LearningScheduleConfig::StepDecay { gamma, min_lr, .. } => {
            // Assume worst case: decay every epoch
            let theoretical_min = initial_lr * gamma.powf(total_epochs as f64);
            min_lr.unwrap_or(theoretical_min.max(initial_lr * 0.0001))
        }

        LearningScheduleConfig::PolynomialDecay { min_lr, .. } => {
            min_lr.unwrap_or(initial_lr * 0.001)
        }

        LearningScheduleConfig::CosineAnnealing { eta_min, .. } => {
            eta_min.unwrap_or(initial_lr * 0.001)
        }

        LearningScheduleConfig::WarmRestarts { eta_min, .. } => {
            eta_min.unwrap_or(initial_lr * 0.001)
        }

        LearningScheduleConfig::OneCycle {
            max_lr,
            final_div_factor,
            div_factor,
            ..
        } => {
            let div_factor_val = div_factor.unwrap_or(25.0);
            let final_div_factor_val = final_div_factor.unwrap_or(1e4);
            let initial_lr_calc = max_lr / div_factor_val;
            initial_lr_calc / final_div_factor_val
        }

        LearningScheduleConfig::CyclicalLR { base_lr, .. } => *base_lr,

        LearningScheduleConfig::NoamLR { .. } => {
            // NoamLR approaches zero as steps increase
            initial_lr * 0.0001
        }
    }
}

/// Check if a learning rate schedule is suitable for LSTM training
pub fn validate_lstm_suitability(
    config: &LearningScheduleConfig,
    initial_lr: f64,
    total_epochs: usize,
) -> Result<Vec<String>> {
    let mut warnings = Vec::new();

    match config {
        LearningScheduleConfig::OneCycle { max_lr, .. } => {
            if *max_lr > initial_lr * 10.0 {
                warnings.push(format!(
                    "OneCycle max_lr ({}) is very high compared to initial_lr ({}). Consider reducing for LSTM stability.",
                    max_lr, initial_lr
                ));
            }
        }

        LearningScheduleConfig::CyclicalLR {
            base_lr, max_lr, ..
        } => {
            let ratio = max_lr / base_lr;
            if ratio > 10.0 {
                warnings.push(format!(
                    "CyclicalLR ratio ({:.1}) is very high. LSTM training typically benefits from ratios < 10.",
                    ratio
                ));
            }
        }

        LearningScheduleConfig::ExponentialDecay { gamma, .. } => {
            if *gamma < 0.9 {
                warnings.push(format!(
                    "ExponentialDecay gamma ({}) is quite aggressive. LSTM training often benefits from slower decay (gamma > 0.9).",
                    gamma
                ));
            }
        }

        LearningScheduleConfig::LinearDecay { decay_rate, .. } => {
            if *decay_rate > 0.1 {
                warnings.push(format!(
                    "LinearDecay rate ({}) is quite aggressive. Consider slower decay for LSTM convergence.",
                    decay_rate
                ));
            }
        }

        LearningScheduleConfig::CosineAnnealing { t_max, .. } => {
            if *t_max as usize > total_epochs {
                warnings.push(format!(
                    "CosineAnnealing t_max ({}) exceeds total_epochs ({}). Schedule will not complete full cycle.",
                    t_max, total_epochs
                ));
            }
        }

        _ => {} // Other schedules are generally LSTM-friendly
    }

    // Check minimum learning rate
    let min_lr = calculate_theoretical_min_lr(config, initial_lr, total_epochs);
    if min_lr < initial_lr * 1e-6 {
        warnings.push(format!(
            "Theoretical minimum LR ({:.2e}) is extremely small. LSTM training may stagnate.",
            min_lr
        ));
    }

    Ok(warnings)
}
