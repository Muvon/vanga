//! Window-aware learning rate scheduling system
//!
//! This module provides a unified approach to apply window decay to all learning rate schedules,
//! ensuring consistent behavior across different schedulers in walk-forward training.

use crate::config::training::LearningScheduleConfig;
use crate::utils::error::Result;

/// Window-aware learning rate calculation that properly integrates window decay
/// with all learning rate schedules
pub struct WindowAwareLearningRate {
    pub original_schedule: LearningScheduleConfig,
    pub window_decay: f64,
    pub window_id: usize,
}

impl WindowAwareLearningRate {
    /// Create a new window-aware learning rate calculator
    pub fn new(
        original_schedule: LearningScheduleConfig,
        window_decay: f64,
        window_id: usize,
    ) -> Self {
        Self {
            original_schedule,
            window_decay,
            window_id,
        }
    }

    /// Apply window decay to a learning rate schedule, returning a modified schedule
    /// that properly accounts for the window decay factor
    pub fn apply_window_decay(&self) -> Result<LearningScheduleConfig> {
        let decay_factor = self.window_decay.powi(self.window_id as i32);

        match &self.original_schedule {
            LearningScheduleConfig::Constant => {
                // For constant schedule, we can't modify the schedule itself,
                // so we return the original and handle decay in the training loop
                Ok(self.original_schedule.clone())
            }

            LearningScheduleConfig::ReduceOnPlateau {
                patience,
                factor,
                min_lr,
                monitor,
                threshold,
            } => {
                // Apply decay to min_lr if present
                let decayed_min_lr = min_lr.map(|lr| lr * decay_factor);

                Ok(LearningScheduleConfig::ReduceOnPlateau {
                    patience: *patience,
                    factor: *factor,
                    min_lr: decayed_min_lr,
                    monitor: monitor.clone(),
                    threshold: *threshold,
                })
            }

            LearningScheduleConfig::LinearDecay { decay_rate, min_lr } => {
                let decayed_min_lr = min_lr.map(|lr| lr * decay_factor);

                Ok(LearningScheduleConfig::LinearDecay {
                    decay_rate: *decay_rate,
                    min_lr: decayed_min_lr,
                })
            }

            LearningScheduleConfig::ExponentialDecay { gamma, min_lr } => {
                let decayed_min_lr = min_lr.map(|lr| lr * decay_factor);

                Ok(LearningScheduleConfig::ExponentialDecay {
                    gamma: *gamma,
                    min_lr: decayed_min_lr,
                })
            }

            LearningScheduleConfig::StepDecay {
                step_size,
                gamma,
                milestones,
                min_lr,
            } => {
                let decayed_min_lr = min_lr.map(|lr| lr * decay_factor);

                Ok(LearningScheduleConfig::StepDecay {
                    step_size: *step_size,
                    gamma: *gamma,
                    milestones: milestones.clone(),
                    min_lr: decayed_min_lr,
                })
            }

            LearningScheduleConfig::PolynomialDecay { power, min_lr } => {
                let decayed_min_lr = min_lr.map(|lr| lr * decay_factor);

                Ok(LearningScheduleConfig::PolynomialDecay {
                    power: *power,
                    min_lr: decayed_min_lr,
                })
            }

            LearningScheduleConfig::CosineAnnealing { t_max, eta_min } => {
                let decayed_eta_min = eta_min.map(|lr| lr * decay_factor);

                Ok(LearningScheduleConfig::CosineAnnealing {
                    t_max: *t_max,
                    eta_min: decayed_eta_min,
                })
            }

            LearningScheduleConfig::WarmRestarts {
                t_0,
                t_mult,
                eta_min,
            } => {
                let decayed_eta_min = eta_min.map(|lr| lr * decay_factor);

                Ok(LearningScheduleConfig::WarmRestarts {
                    t_0: *t_0,
                    t_mult: *t_mult,
                    eta_min: decayed_eta_min,
                })
            }

            LearningScheduleConfig::OneCycle {
                max_lr,
                pct_start,
                anneal_strategy,
                div_factor,
                final_div_factor,
            } => {
                // CRITICAL: Apply decay to max_lr for OneCycle
                let decayed_max_lr = max_lr * decay_factor;

                Ok(LearningScheduleConfig::OneCycle {
                    max_lr: decayed_max_lr,
                    pct_start: *pct_start,
                    anneal_strategy: anneal_strategy.clone(),
                    div_factor: *div_factor,
                    final_div_factor: *final_div_factor,
                })
            }

            LearningScheduleConfig::CyclicalLR {
                base_lr,
                max_lr,
                step_size_up,
                step_size_down,
                mode,
                gamma,
            } => {
                // Apply decay to both base_lr and max_lr
                let decayed_base_lr = base_lr * decay_factor;
                let decayed_max_lr = max_lr * decay_factor;

                Ok(LearningScheduleConfig::CyclicalLR {
                    base_lr: decayed_base_lr,
                    max_lr: decayed_max_lr,
                    step_size_up: *step_size_up,
                    step_size_down: *step_size_down,
                    mode: mode.clone(),
                    gamma: *gamma,
                })
            }

            LearningScheduleConfig::NoamLR {
                model_size,
                warmup_steps,
                factor,
            } => {
                // Apply decay to the factor parameter
                let decayed_factor = factor.map(|f| f * decay_factor).or(Some(decay_factor));

                Ok(LearningScheduleConfig::NoamLR {
                    model_size: *model_size,
                    warmup_steps: *warmup_steps,
                    factor: decayed_factor,
                })
            }
        }
    }

    /// Calculate the effective base learning rate for schedules that don't define their own
    pub fn calculate_effective_base_lr(&self, config_lr: f64) -> f64 {
        let decay_factor = self.window_decay.powi(self.window_id as i32);

        match &self.original_schedule {
            // These schedules use config.learning_rate as base
            LearningScheduleConfig::Constant
            | LearningScheduleConfig::ReduceOnPlateau { .. }
            | LearningScheduleConfig::LinearDecay { .. }
            | LearningScheduleConfig::ExponentialDecay { .. }
            | LearningScheduleConfig::StepDecay { .. }
            | LearningScheduleConfig::PolynomialDecay { .. }
            | LearningScheduleConfig::CosineAnnealing { .. }
            | LearningScheduleConfig::WarmRestarts { .. } => config_lr * decay_factor,

            // These schedules define their own learning rates
            LearningScheduleConfig::OneCycle { .. }
            | LearningScheduleConfig::CyclicalLR { .. }
            | LearningScheduleConfig::NoamLR { .. } => {
                // For these, the decay is applied within the schedule itself
                config_lr // Return original config LR for logging purposes
            }
        }
    }

    /// Get a description of how window decay is applied for this schedule
    pub fn get_decay_description(&self) -> String {
        let decay_factor = self.window_decay.powi(self.window_id as i32);
        let decay_percentage = decay_factor * 100.0;

        match &self.original_schedule {
            LearningScheduleConfig::Constant => {
                format!(
                    "Constant LR scaled by {:.1}% (window decay)",
                    decay_percentage
                )
            }

            LearningScheduleConfig::ReduceOnPlateau { .. } => {
                format!(
                    "ReduceOnPlateau with base LR scaled by {:.1}% and min_lr adjusted",
                    decay_percentage
                )
            }

            LearningScheduleConfig::LinearDecay { .. } => {
                format!(
                    "LinearDecay with base LR scaled by {:.1}% and min_lr adjusted",
                    decay_percentage
                )
            }

            LearningScheduleConfig::ExponentialDecay { .. } => {
                format!(
                    "ExponentialDecay with base LR scaled by {:.1}% and min_lr adjusted",
                    decay_percentage
                )
            }

            LearningScheduleConfig::StepDecay { .. } => {
                format!(
                    "StepDecay with base LR scaled by {:.1}% and min_lr adjusted",
                    decay_percentage
                )
            }

            LearningScheduleConfig::PolynomialDecay { .. } => {
                format!(
                    "PolynomialDecay with base LR scaled by {:.1}% and min_lr adjusted",
                    decay_percentage
                )
            }

            LearningScheduleConfig::CosineAnnealing { .. } => {
                format!(
                    "CosineAnnealing with base LR scaled by {:.1}% and eta_min adjusted",
                    decay_percentage
                )
            }

            LearningScheduleConfig::WarmRestarts { .. } => {
                format!(
                    "WarmRestarts with base LR scaled by {:.1}% and eta_min adjusted",
                    decay_percentage
                )
            }

            LearningScheduleConfig::OneCycle { max_lr, .. } => {
                let decayed_max_lr = max_lr * decay_factor;
                format!(
                    "OneCycle with max_lr scaled from {:.6} to {:.6} ({:.1}%)",
                    max_lr, decayed_max_lr, decay_percentage
                )
            }

            LearningScheduleConfig::CyclicalLR {
                base_lr, max_lr, ..
            } => {
                let decayed_base_lr = base_lr * decay_factor;
                let decayed_max_lr = max_lr * decay_factor;
                format!(
                    "CyclicalLR with base_lr: {:.6}→{:.6}, max_lr: {:.6}→{:.6} ({:.1}%)",
                    base_lr, decayed_base_lr, max_lr, decayed_max_lr, decay_percentage
                )
            }

            LearningScheduleConfig::NoamLR { factor, .. } => {
                let original_factor = factor.unwrap_or(1.0);
                let decayed_factor = original_factor * decay_factor;
                format!(
                    "NoamLR with factor scaled from {:.6} to {:.6} ({:.1}%)",
                    original_factor, decayed_factor, decay_percentage
                )
            }
        }
    }

    /// Validate that the window decay configuration is reasonable for this schedule
    pub fn validate_window_decay_compatibility(&self) -> Result<Vec<String>> {
        let mut warnings = Vec::new();
        let decay_factor = self.window_decay.powi(self.window_id as i32);

        // General validation
        if self.window_decay < 0.5 && self.window_id > 3 {
            warnings.push(format!(
                "Aggressive window decay ({}) may cause learning rates to become too small after window {} (factor: {:.6})",
                self.window_decay, self.window_id, decay_factor
            ));
        }

        if self.window_decay > 1.0 {
            warnings.push(format!(
                "Window decay > 1.0 ({}) will increase learning rates over time, which may cause instability",
                self.window_decay
            ));
        }

        // Schedule-specific validation
        match &self.original_schedule {
            LearningScheduleConfig::OneCycle {
                max_lr, div_factor, ..
            } => {
                let decayed_max_lr = max_lr * decay_factor;
                let initial_lr = decayed_max_lr / div_factor.unwrap_or(25.0);

                if decayed_max_lr < 1e-5 {
                    warnings.push(format!(
                        "OneCycle max_lr becomes very small ({:.2e}) after window decay. Consider reducing window_decay or increasing max_lr.",
                        decayed_max_lr
                    ));
                }

                if initial_lr < 1e-7 {
                    warnings.push(format!(
                        "OneCycle initial_lr becomes extremely small ({:.2e}) after window decay. Training may stagnate.",
                        initial_lr
                    ));
                }
            }

            LearningScheduleConfig::CyclicalLR {
                base_lr, max_lr, ..
            } => {
                let decayed_base_lr = base_lr * decay_factor;
                let decayed_max_lr = max_lr * decay_factor;

                if decayed_base_lr < 1e-6 {
                    warnings.push(format!(
                        "CyclicalLR base_lr becomes very small ({:.2e}) after window decay.",
                        decayed_base_lr
                    ));
                }

                if (decayed_max_lr - decayed_base_lr) < 1e-6 {
                    warnings.push(format!(
                        "CyclicalLR range becomes too narrow after window decay: {:.2e} to {:.2e}",
                        decayed_base_lr, decayed_max_lr
                    ));
                }
            }

            _ => {} // Other schedules are generally more robust to decay
        }

        Ok(warnings)
    }
}

/// Helper function to create window-aware configuration for training
pub fn create_window_aware_config(
    original_config: &crate::config::TrainingConfig,
    window_id: usize,
) -> Result<crate::config::TrainingConfig> {
    let mut window_config = original_config.clone();

    if let Some(schedule) = &original_config.training.learning_schedule {
        let window_aware = WindowAwareLearningRate::new(
            schedule.clone(),
            original_config.training.window_decay,
            window_id,
        );

        // Apply window decay to the schedule
        let decayed_schedule = window_aware.apply_window_decay()?;
        window_config.training.learning_schedule = Some(decayed_schedule);

        // Update base learning rate for schedules that use it
        let effective_base_lr =
            window_aware.calculate_effective_base_lr(original_config.training.learning_rate);
        window_config.training.learning_rate = effective_base_lr;

        // Log the decay application
        log::info!(
            "🔄 Window {} decay applied: {}",
            window_id + 1,
            window_aware.get_decay_description()
        );

        // Validate and log warnings
        let warnings = window_aware.validate_window_decay_compatibility()?;
        for warning in warnings {
            log::warn!("⚠️ Window decay warning: {}", warning);
        }
    } else {
        // No schedule, just apply simple decay to base learning rate
        let decay_factor = original_config.training.window_decay.powi(window_id as i32);
        window_config.training.learning_rate =
            original_config.training.learning_rate * decay_factor;

        log::info!(
            "🔄 Window {} simple decay: {:.6} → {:.6} ({:.1}%)",
            window_id + 1,
            original_config.training.learning_rate,
            window_config.training.learning_rate,
            decay_factor * 100.0
        );
    }

    Ok(window_config)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_onecycle_window_decay() {
        let original_schedule = LearningScheduleConfig::OneCycle {
            max_lr: 0.01,
            pct_start: Some(0.3),
            anneal_strategy: Some("cos".to_string()),
            div_factor: Some(25.0),
            final_div_factor: Some(1e4),
        };

        let window_aware = WindowAwareLearningRate::new(original_schedule, 0.8, 2);
        let decayed_schedule = window_aware.apply_window_decay().unwrap();

        if let LearningScheduleConfig::OneCycle { max_lr, .. } = decayed_schedule {
            // Window 2 with decay 0.8: 0.01 * 0.8^2 = 0.01 * 0.64 = 0.0064
            assert!((max_lr - 0.0064).abs() < 1e-6);
        } else {
            panic!("Expected OneCycle schedule");
        }
    }

    #[test]
    fn test_cyclical_lr_window_decay() {
        let original_schedule = LearningScheduleConfig::CyclicalLR {
            base_lr: 1e-5,
            max_lr: 1e-3,
            step_size_up: 20,
            step_size_down: Some(20),
            mode: Some("triangular".to_string()),
            gamma: Some(1.0),
        };

        let window_aware = WindowAwareLearningRate::new(original_schedule, 0.9, 1);
        let decayed_schedule = window_aware.apply_window_decay().unwrap();

        if let LearningScheduleConfig::CyclicalLR {
            base_lr, max_lr, ..
        } = decayed_schedule
        {
            // Window 1 with decay 0.9: both LRs scaled by 0.9
            assert!((base_lr - 1e-5 * 0.9).abs() < 1e-8);
            assert!((max_lr - 1e-3 * 0.9).abs() < 1e-6);
        } else {
            panic!("Expected CyclicalLR schedule");
        }
    }

    #[test]
    fn test_cosine_annealing_window_decay() {
        let original_schedule = LearningScheduleConfig::CosineAnnealing {
            t_max: 100,
            eta_min: Some(1e-6),
        };

        let window_aware = WindowAwareLearningRate::new(original_schedule, 0.7, 3);
        let decayed_schedule = window_aware.apply_window_decay().unwrap();

        if let LearningScheduleConfig::CosineAnnealing { eta_min, .. } = decayed_schedule {
            // Window 3 with decay 0.7: eta_min scaled by 0.7^3 = 0.343
            let expected_eta_min = 1e-6 * 0.7_f64.powi(3);
            assert!((eta_min.unwrap() - expected_eta_min).abs() < 1e-9);
        } else {
            panic!("Expected CosineAnnealing schedule");
        }
    }

    #[test]
    fn test_window_decay_warnings() {
        let original_schedule = LearningScheduleConfig::OneCycle {
            max_lr: 1e-4, // Very small max_lr
            pct_start: Some(0.3),
            anneal_strategy: Some("cos".to_string()),
            div_factor: Some(25.0),
            final_div_factor: Some(1e4),
        };

        let window_aware = WindowAwareLearningRate::new(original_schedule, 0.5, 5); // Aggressive decay
        let warnings = window_aware.validate_window_decay_compatibility().unwrap();

        assert!(
            !warnings.is_empty(),
            "Should generate warnings for aggressive decay"
        );
        assert!(warnings
            .iter()
            .any(|w| w.contains("max_lr becomes very small")));
    }
}
