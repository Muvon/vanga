//! Backtesting results reporting and formatting
//!
//! Provides comprehensive reporting capabilities for backtesting results
//! with clean, readable output formats.

use crate::api::backtester::BacktestResults;
use crate::utils::error::Result;
use std::path::Path;

/// Backtest results reporter
pub struct BacktestReporter;

impl BacktestReporter {
    /// Generate comprehensive backtest report
    pub fn generate_report(results: &[BacktestResults]) -> String {
        if results.is_empty() {
            return "No backtesting results to display.".to_string();
        }

        let mut report = String::new();

        // Header
        report.push_str("📊 VANGA BACKTESTING RESULTS\n");
        report.push_str(&"=".repeat(80));
        report.push('\n');

        // Summary table
        if results.len() > 1 {
            report.push_str(&Self::generate_summary_table(results));
            report.push('\n');
        }

        // Detailed results for each symbol
        for (i, result) in results.iter().enumerate() {
            if i > 0 {
                report.push_str(&"-".repeat(80));
                report.push('\n');
            }
            report.push_str(&Self::generate_detailed_report(result));
        }

        // Footer
        report.push_str(&"=".repeat(80));
        report.push_str("\n🚀 VANGA Backtesting Complete\n");

        report
    }

    /// Generate summary comparison table
    fn generate_summary_table(results: &[BacktestResults]) -> String {
        let mut table = String::new();

        table.push_str("📈 SUMMARY COMPARISON\n\n");
        table.push_str(&format!(
            "{:<12} {:<8} {:<8} {:<8} {:<8} {:<8} {:<8}\n",
            "Symbol", "RMSE", "MAE", "R²", "MAPE%", "Dir.Acc%", "Samples"
        ));
        table.push_str(&"-".repeat(70));
        table.push('\n');

        for result in results {
            table.push_str(&format!(
                "{:<12} {:<8.4} {:<8.4} {:<8.3} {:<8.2} {:<8.1} {:<8}\n",
                result.symbol,
                result.regression_metrics.rmse,
                result.regression_metrics.mae,
                result.regression_metrics.r_squared,
                result.regression_metrics.mape,
                result.directional_accuracy * 100.0,
                result.test_samples
            ));
        }

        table.push('\n');
        table
    }

    /// Generate detailed report for single symbol
    fn generate_detailed_report(result: &BacktestResults) -> String {
        let mut report = String::new();

        // Header
        report.push_str(&format!("🎯 {} BACKTESTING RESULTS\n", result.symbol));
        report.push_str(&format!(
            "Model: {} | Test Samples: {}\n\n",
            result.model_type, result.test_samples
        ));

        // Time periods
        report.push_str("📅 TIME PERIODS:\n");
        report.push_str(&format!(
            "├─ Training:   {} → {}\n",
            result.train_period.0, result.train_period.1
        ));
        report.push_str(&format!(
            "└─ Testing:    {} → {}\n\n",
            result.test_period.0, result.test_period.1
        ));

        // Performance metrics
        report.push_str("📊 PERFORMANCE METRICS:\n");
        report.push_str("├─ Regression Metrics:\n");
        report.push_str(&format!(
            "│  ├─ RMSE:     {:.6}\n",
            result.regression_metrics.rmse
        ));
        report.push_str(&format!(
            "│  ├─ MAE:      {:.6}\n",
            result.regression_metrics.mae
        ));
        report.push_str(&format!(
            "│  ├─ R²:       {:.4}\n",
            result.regression_metrics.r_squared
        ));
        report.push_str(&format!(
            "│  └─ MAPE:     {:.2}%\n",
            result.regression_metrics.mape
        ));
        report.push_str("└─ Trading Metrics:\n");
        report.push_str(&format!(
            "   ├─ Directional Accuracy: {:.1}%\n",
            result.directional_accuracy * 100.0
        ));
        report.push_str(&format!(
            "   └─ Predictions Made:     {}\n",
            result.prediction_count
        ));

        // Performance assessment
        report.push_str(&Self::generate_performance_assessment(result));

        report.push('\n');
        report
    }

    /// Generate performance assessment
    fn generate_performance_assessment(result: &BacktestResults) -> String {
        let mut assessment = String::new();

        assessment.push_str("\n🎯 PERFORMANCE ASSESSMENT:\n");

        // R² assessment
        let r2_rating = match result.regression_metrics.r_squared {
            r2 if r2 >= 0.8 => ("🟢 Excellent", "Strong predictive power"),
            r2 if r2 >= 0.6 => ("🟡 Good", "Moderate predictive power"),
            r2 if r2 >= 0.4 => ("🟠 Fair", "Limited predictive power"),
            _ => ("🔴 Poor", "Weak predictive power"),
        };

        // Directional accuracy assessment
        let dir_rating = match result.directional_accuracy {
            acc if acc >= 0.65 => ("🟢 Strong", "Good trend prediction"),
            acc if acc >= 0.55 => ("🟡 Moderate", "Decent trend prediction"),
            acc if acc >= 0.50 => ("🟠 Weak", "Limited trend prediction"),
            _ => ("🔴 Poor", "Poor trend prediction"),
        };

        // MAPE assessment
        let mape_rating = match result.regression_metrics.mape {
            mape if mape <= 5.0 => ("🟢 Excellent", "Very accurate predictions"),
            mape if mape <= 10.0 => ("🟡 Good", "Reasonably accurate"),
            mape if mape <= 20.0 => ("🟠 Fair", "Moderate accuracy"),
            _ => ("🔴 Poor", "Low accuracy"),
        };

        assessment.push_str(&format!(
            "├─ Prediction Quality: {} - {}\n",
            r2_rating.0, r2_rating.1
        ));
        assessment.push_str(&format!(
            "├─ Trend Detection:    {} - {}\n",
            dir_rating.0, dir_rating.1
        ));
        assessment.push_str(&format!(
            "└─ Price Accuracy:     {} - {}\n",
            mape_rating.0, mape_rating.1
        ));

        assessment
    }

    /// Save results to JSON file (simplified format)
    pub fn save_to_json(results: &[BacktestResults], path: &Path) -> Result<()> {
        let mut json_content = String::from("[\n");

        for (i, result) in results.iter().enumerate() {
            if i > 0 {
                json_content.push_str(",\n");
            }
            json_content.push_str(&format!(
                "  {{\n    \"symbol\": \"{}\",\n    \"model_type\": \"{}\",\n    \"test_samples\": {},\n    \"rmse\": {:.6},\n    \"mae\": {:.6},\n    \"r_squared\": {:.4},\n    \"mape\": {:.2},\n    \"directional_accuracy\": {:.4}\n  }}",
                result.symbol,
                result.model_type,
                result.test_samples,
                result.regression_metrics.rmse,
                result.regression_metrics.mae,
                result.regression_metrics.r_squared,
                result.regression_metrics.mape,
                result.directional_accuracy
            ));
        }

        json_content.push_str("\n]");

        std::fs::write(path, json_content).map_err(|e| {
            crate::utils::error::VangaError::DataError(format!("Failed to write JSON: {}", e))
        })?;

        Ok(())
    }

    /// Generate CSV summary
    pub fn generate_csv_summary(results: &[BacktestResults]) -> String {
        let mut csv = String::new();

        // Header
        csv.push_str("symbol,model_type,test_samples,rmse,mae,r_squared,mape,directional_accuracy,prediction_count\n");

        // Data rows
        for result in results {
            csv.push_str(&format!(
                "{},{},{},{:.6},{:.6},{:.4},{:.2},{:.4},{}\n",
                result.symbol,
                result.model_type,
                result.test_samples,
                result.regression_metrics.rmse,
                result.regression_metrics.mae,
                result.regression_metrics.r_squared,
                result.regression_metrics.mape,
                result.directional_accuracy,
                result.prediction_count
            ));
        }

        csv
    }

    /// Save CSV summary to file
    pub fn save_csv_summary(results: &[BacktestResults], path: &Path) -> Result<()> {
        let csv = Self::generate_csv_summary(results);

        std::fs::write(path, csv).map_err(|e| {
            crate::utils::error::VangaError::DataError(format!("Failed to write CSV: {}", e))
        })?;

        Ok(())
    }

    /// Generate markdown report
    pub fn generate_markdown_report(results: &[BacktestResults]) -> String {
        let mut md = String::new();

        md.push_str("# VANGA Backtesting Results\n\n");

        if results.len() > 1 {
            md.push_str("## Summary\n\n");
            md.push_str("| Symbol | RMSE | MAE | R² | MAPE% | Dir.Acc% | Samples |\n");
            md.push_str("|--------|------|-----|----|----|----|---------|\n");

            for result in results {
                md.push_str(&format!(
                    "| {} | {:.4} | {:.4} | {:.3} | {:.2} | {:.1} | {} |\n",
                    result.symbol,
                    result.regression_metrics.rmse,
                    result.regression_metrics.mae,
                    result.regression_metrics.r_squared,
                    result.regression_metrics.mape,
                    result.directional_accuracy * 100.0,
                    result.test_samples
                ));
            }
            md.push('\n');
        }

        for result in results {
            md.push_str(&format!("## {} Results\n\n", result.symbol));
            md.push_str(&format!("- **Model**: {}\n", result.model_type));
            md.push_str(&format!("- **Test Samples**: {}\n", result.test_samples));
            md.push_str(&format!(
                "- **RMSE**: {:.6}\n",
                result.regression_metrics.rmse
            ));
            md.push_str(&format!(
                "- **MAE**: {:.6}\n",
                result.regression_metrics.mae
            ));
            md.push_str(&format!(
                "- **R²**: {:.4}\n",
                result.regression_metrics.r_squared
            ));
            md.push_str(&format!(
                "- **MAPE**: {:.2}%\n",
                result.regression_metrics.mape
            ));
            md.push_str(&format!(
                "- **Directional Accuracy**: {:.1}%\n\n",
                result.directional_accuracy * 100.0
            ));
        }

        md
    }
}

/// Convenience function for quick reporting
pub fn print_backtest_results(results: &[BacktestResults]) {
    println!("{}", BacktestReporter::generate_report(results));
}

/// Save comprehensive backtest report
pub fn save_backtest_report(
    results: &[BacktestResults],
    output_dir: &Path,
    format: &str,
) -> Result<()> {
    std::fs::create_dir_all(output_dir).map_err(|e| {
        crate::utils::error::VangaError::DataError(format!(
            "Failed to create output directory: {}",
            e
        ))
    })?;

    match format.to_lowercase().as_str() {
        "json" => {
            let json_path = output_dir.join("backtest_results.json");
            BacktestReporter::save_to_json(results, &json_path)?;
            log::info!("📄 JSON report saved to: {:?}", json_path);
        }
        "csv" => {
            let csv_path = output_dir.join("backtest_summary.csv");
            BacktestReporter::save_csv_summary(results, &csv_path)?;
            log::info!("📊 CSV summary saved to: {:?}", csv_path);
        }
        "markdown" | "md" => {
            let md_path = output_dir.join("backtest_report.md");
            let md_content = BacktestReporter::generate_markdown_report(results);
            std::fs::write(&md_path, md_content).map_err(|e| {
                crate::utils::error::VangaError::DataError(format!(
                    "Failed to write markdown: {}",
                    e
                ))
            })?;
            log::info!("📝 Markdown report saved to: {:?}", md_path);
        }
        _ => {
            return Err(crate::utils::error::VangaError::DataError(format!(
                "Unsupported format: {}. Use 'json', 'csv', or 'markdown'",
                format
            )));
        }
    }

    Ok(())
}
