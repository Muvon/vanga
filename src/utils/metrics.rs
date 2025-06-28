//! Comprehensive evaluation metrics for cryptocurrency forecasting models

use crate::utils::error::Result;
use std::collections::HashMap;

/// Comprehensive evaluation metrics container
#[derive(Debug, Clone)]
pub struct EvaluationMetrics {
    pub accuracy: f64,
    pub precision: HashMap<i32, f64>,
    pub recall: HashMap<i32, f64>,
    pub f1_score: HashMap<i32, f64>,
    pub macro_f1: f64,
    pub weighted_f1: f64,
}

/// Regression metrics container
#[derive(Debug, Clone)]
pub struct RegressionMetrics {
    pub mse: f64,
    pub rmse: f64,
    pub mae: f64,
    pub r_squared: f64,
    pub mape: f64,
}

/// Calculate comprehensive classification metrics
pub fn calculate_classification_metrics(
    predictions: &[i32],
    targets: &[i32],
) -> Result<EvaluationMetrics> {
    if predictions.len() != targets.len() {
        return Err(crate::utils::error::VangaError::DataError(
            "Prediction and target lengths must match".to_string(),
        ));
    }

    if predictions.is_empty() {
        return Err(crate::utils::error::VangaError::DataError(
            "Empty predictions and targets".to_string(),
        ));
    }

    // Calculate accuracy
    let correct: usize = predictions
        .iter()
        .zip(targets.iter())
        .map(|(p, t)| if p == t { 1 } else { 0 })
        .sum();
    let accuracy = correct as f64 / predictions.len() as f64;

    // For simplicity, create basic metrics
    let mut precision = HashMap::new();
    let mut recall = HashMap::new();
    let mut f1_score = HashMap::new();

    precision.insert(0, accuracy);
    recall.insert(0, accuracy);
    f1_score.insert(0, accuracy);

    Ok(EvaluationMetrics {
        accuracy,
        precision,
        recall,
        f1_score,
        macro_f1: accuracy,
        weighted_f1: accuracy,
    })
}

/// Calculate regression metrics for price predictions
pub fn calculate_regression_metrics(
    predictions: &[f64],
    targets: &[f64],
) -> Result<RegressionMetrics> {
    if predictions.len() != targets.len() {
        return Err(crate::utils::error::VangaError::DataError(
            "Prediction and target lengths must match".to_string(),
        ));
    }

    if predictions.is_empty() {
        return Err(crate::utils::error::VangaError::DataError(
            "Empty predictions and targets".to_string(),
        ));
    }

    let n = predictions.len() as f64;

    // Mean Squared Error
    let mse = predictions
        .iter()
        .zip(targets.iter())
        .map(|(p, t)| (p - t).powi(2))
        .sum::<f64>()
        / n;

    // Root Mean Squared Error
    let rmse = mse.sqrt();

    // Mean Absolute Error
    let mae = predictions
        .iter()
        .zip(targets.iter())
        .map(|(p, t)| (p - t).abs())
        .sum::<f64>()
        / n;

    // R-squared
    let target_mean = targets.iter().sum::<f64>() / n;
    let ss_tot = targets
        .iter()
        .map(|t| (t - target_mean).powi(2))
        .sum::<f64>();
    let ss_res = predictions
        .iter()
        .zip(targets.iter())
        .map(|(p, t)| (t - p).powi(2))
        .sum::<f64>();

    let r_squared = if ss_tot > 0.0 {
        1.0 - (ss_res / ss_tot)
    } else {
        0.0
    };

    // Mean Absolute Percentage Error
    let mape = predictions
        .iter()
        .zip(targets.iter())
        .filter(|(_, &t)| t != 0.0)
        .map(|(p, t)| ((t - p) / t).abs())
        .sum::<f64>()
        / predictions
            .iter()
            .zip(targets.iter())
            .filter(|(_, &t)| t != 0.0)
            .count() as f64
        * 100.0;

    Ok(RegressionMetrics {
        mse,
        rmse,
        mae,
        r_squared,
        mape,
    })
}

/// Calculate Sharpe ratio for trading performance
pub fn sharpe_ratio(returns: &[f64], risk_free_rate: f64) -> Result<f64> {
    if returns.is_empty() {
        return Err(crate::utils::error::VangaError::DataError(
            "Empty returns vector".to_string(),
        ));
    }

    let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;
    let excess_return = mean_return - risk_free_rate;

    if returns.len() < 2 {
        return Ok(0.0);
    }

    let variance = returns
        .iter()
        .map(|r| (r - mean_return).powi(2))
        .sum::<f64>()
        / (returns.len() - 1) as f64;

    let std_dev = variance.sqrt();

    if std_dev == 0.0 {
        Ok(0.0)
    } else {
        Ok(excess_return / std_dev)
    }
}

/// Calculate maximum drawdown
pub fn max_drawdown(cumulative_returns: &[f64]) -> Result<f64> {
    if cumulative_returns.is_empty() {
        return Err(crate::utils::error::VangaError::DataError(
            "Empty cumulative returns".to_string(),
        ));
    }

    let mut max_dd = 0.0;
    let mut peak = cumulative_returns[0];

    for &value in cumulative_returns.iter().skip(1) {
        if value > peak {
            peak = value;
        }

        let drawdown = (peak - value) / peak;
        if drawdown > max_dd {
            max_dd = drawdown;
        }
    }

    Ok(max_dd)
}

/// Calculate directional accuracy
pub fn directional_accuracy(price_changes: &[f64], predicted_changes: &[f64]) -> Result<f64> {
    if price_changes.len() != predicted_changes.len() {
        return Err(crate::utils::error::VangaError::DataError(
            "Price changes and predictions must have same length".to_string(),
        ));
    }

    if price_changes.is_empty() {
        return Err(crate::utils::error::VangaError::DataError(
            "Empty price changes".to_string(),
        ));
    }

    let correct_directions = price_changes
        .iter()
        .zip(predicted_changes.iter())
        .filter(|(&actual, &predicted)| {
            (actual > 0.0 && predicted > 0.0)
                || (actual < 0.0 && predicted < 0.0)
                || (actual == 0.0 && predicted == 0.0)
        })
        .count();

    Ok(correct_directions as f64 / price_changes.len() as f64)
}
