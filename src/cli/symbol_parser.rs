// CLI argument parsing with unified symbol interface
use clap::{Arg, ArgMatches, Command};
use std::collections::HashMap;

/// Parse symbols from CLI argument - supports both single and multiple symbols
pub fn parse_symbols(symbol_arg: &str) -> Vec<String> {
    symbol_arg
        .split(',')
        .map(|s| s.trim().to_uppercase())
        .filter(|s| !s.is_empty())
        .collect()
}

/// Data path resolution for single vs multi-symbol scenarios
#[derive(Debug, Clone)]
pub enum DataPaths {
    /// Single symbol: direct file path
    Single(String),
    /// Multi-symbol: symbol -> file path mapping
    Multi(HashMap<String, String>),
}

/// Resolve data paths based on symbol count and CLI arguments
pub fn resolve_data_paths(
    symbols: &[String],
    data_arg: Option<&str>,
    data_dir_arg: Option<&str>,
) -> Result<DataPaths, String> {
    match symbols.len() {
        0 => Err("No symbols provided".to_string()),
        1 => {
            // Single symbol: require --data file
            if let Some(data_file) = data_arg {
                Ok(DataPaths::Single(data_file.to_string()))
            } else {
                Err("Single symbol requires --data <file> argument".to_string())
            }
        }
        _ => {
            // Multi-symbol: require --data-dir directory
            if let Some(data_dir) = data_dir_arg {
                let mut paths = HashMap::new();
                for symbol in symbols {
                    let file_path = format!("{}/{}_1h.csv", data_dir, symbol);
                    paths.insert(symbol.clone(), file_path);
                }
                Ok(DataPaths::Multi(paths))
            } else {
                Err("Multi-symbol requires --data-dir <directory> argument".to_string())
            }
        }
    }
}

/// Auto-select appropriate configuration based on symbol count
pub fn select_config(symbols: &[String], config_arg: Option<&str>) -> String {
    if let Some(config) = config_arg {
        return config.to_string();
    }

    match symbols.len() {
        1 => "configs/tft_enhanced.toml".to_string(),
        2..=4 => "configs/tft_gnn_small_portfolio.toml".to_string(),
        5..=8 => "configs/tft_gnn_multi_asset.toml".to_string(),
        _ => "configs/tft_gnn_large_portfolio.toml".to_string(),
    }
}

/// Build CLI command with unified symbol interface
pub fn build_train_command() -> Command {
    Command::new("train")
        .about("Train VANGA model with single or multiple symbols")
        .arg(
            Arg::new("symbol")
                .long("symbol")
                .short('s')
                .value_name("SYMBOLS")
                .help("Symbol(s) to train on. Single: BTCUSDT, Multiple: BTCUSDT,ETHUSDT,ADAUSDT")
                .required(true),
        )
        .arg(
            Arg::new("data")
                .long("data")
                .short('d')
                .value_name("FILE")
                .help("Data file for single symbol training")
                .conflicts_with("data-dir"),
        )
        .arg(
            Arg::new("data-dir")
                .long("data-dir")
                .value_name("DIRECTORY")
                .help("Data directory for multi-symbol training (contains SYMBOL_1h.csv files)")
                .conflicts_with("data"),
        )
        .arg(
            Arg::new("config")
                .long("config")
                .short('c')
                .value_name("FILE")
                .help("Configuration file (auto-selected if not provided)"),
        )
        .arg(
            Arg::new("output")
                .long("output")
                .short('o')
                .value_name("FILE")
                .help("Output model file")
                .required(true),
        )
        .arg(
            Arg::new("auto-optimize")
                .long("auto-optimize")
                .help("Enable automatic parameter optimization")
                .action(clap::ArgAction::SetTrue),
        )
        .arg(
            Arg::new("strategy")
                .long("strategy")
                .value_name("STRATEGY")
                .help("Optimization strategy: crypto_optimized, conservative, portfolio_optimized")
                .default_value("crypto_optimized")
                .requires("auto-optimize"),
        )
}

/// Build CLI command for prediction
pub fn build_predict_command() -> Command {
    Command::new("predict")
        .about("Predict with VANGA model using single or multiple symbols")
        .arg(
            Arg::new("symbol")
                .long("symbol")
                .short('s')
                .value_name("SYMBOLS")
                .help("Symbol(s) to predict. Must match training symbols.")
                .required(true),
        )
        .arg(
            Arg::new("input")
                .long("input")
                .short('i')
                .value_name("FILE")
                .help("Input data file for single symbol prediction")
                .conflicts_with("input-dir"),
        )
        .arg(
            Arg::new("input-dir")
                .long("input-dir")
                .value_name("DIRECTORY")
                .help("Input data directory for multi-symbol prediction")
                .conflicts_with("input"),
        )
        .arg(
            Arg::new("model")
                .long("model")
                .short('m')
                .value_name("FILE")
                .help("Trained model file")
                .required(true),
        )
        .arg(
            Arg::new("output")
                .long("output")
                .short('o')
                .value_name("FILE")
                .help("Output predictions file (JSON format)"),
        )
        .arg(
            Arg::new("quantiles")
                .long("quantiles")
                .value_name("LEVELS")
                .help("Quantile levels for uncertainty (e.g., 0.05,0.95)")
                .default_value("0.05,0.25,0.5,0.75,0.95"),
        )
        .arg(
            Arg::new("include-regime")
                .long("include-regime")
                .help("Include market regime detection in output")
                .action(clap::ArgAction::SetTrue),
        )
        .arg(
            Arg::new("include-correlations")
                .long("include-correlations")
                .help("Include cross-asset correlations (multi-symbol only)")
                .action(clap::ArgAction::SetTrue),
        )
}

/// Parse and validate training arguments
pub fn parse_train_args(matches: &ArgMatches) -> Result<TrainArgs, String> {
    let symbol_arg = matches.get_one::<String>("symbol").unwrap();
    let symbols = parse_symbols(symbol_arg);

    if symbols.is_empty() {
        return Err("No valid symbols provided".to_string());
    }

    let data_paths = resolve_data_paths(
        &symbols,
        matches.get_one::<String>("data").map(|s| s.as_str()),
        matches.get_one::<String>("data-dir").map(|s| s.as_str()),
    )?;

    let config_path = select_config(
        &symbols,
        matches.get_one::<String>("config").map(|s| s.as_str()),
    );

    let output_path = matches.get_one::<String>("output").unwrap().clone();

    let auto_optimize = matches.get_flag("auto-optimize");
    let strategy = if auto_optimize {
        Some(matches.get_one::<String>("strategy").unwrap().clone())
    } else {
        None
    };

    Ok(TrainArgs {
        symbols,
        data_paths,
        config_path,
        output_path,
        auto_optimize,
        strategy,
    })
}

/// Parse and validate prediction arguments
pub fn parse_predict_args(matches: &ArgMatches) -> Result<PredictArgs, String> {
    let symbol_arg = matches.get_one::<String>("symbol").unwrap();
    let symbols = parse_symbols(symbol_arg);

    if symbols.is_empty() {
        return Err("No valid symbols provided".to_string());
    }

    let input_paths = resolve_data_paths(
        &symbols,
        matches.get_one::<String>("input").map(|s| s.as_str()),
        matches.get_one::<String>("input-dir").map(|s| s.as_str()),
    )?;

    let model_path = matches.get_one::<String>("model").unwrap().clone();

    let output_path = matches
        .get_one::<String>("output")
        .cloned()
        .unwrap_or_else(|| {
            if symbols.len() == 1 {
                format!("{}_predictions.json", symbols[0])
            } else {
                "portfolio_predictions.json".to_string()
            }
        });

    let quantiles = parse_quantiles(matches.get_one::<String>("quantiles").unwrap())?;

    let include_regime = matches.get_flag("include-regime");
    let include_correlations = matches.get_flag("include-correlations");

    Ok(PredictArgs {
        symbols,
        input_paths,
        model_path,
        output_path,
        quantiles,
        include_regime,
        include_correlations,
    })
}

/// Parse quantile levels from string
fn parse_quantiles(quantiles_str: &str) -> Result<Vec<f64>, String> {
    let mut quantiles: Vec<f64> = quantiles_str
        .split(',')
        .map(|s| s.trim().parse::<f64>())
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| format!("Invalid quantile format: {}", e))?
        .into_iter()
        .filter(|&q| q > 0.0 && q < 1.0)
        .collect();

    if quantiles.is_empty() {
        return Err("No valid quantiles provided (must be between 0 and 1)".to_string());
    }

    quantiles.sort_by(|a, b| a.partial_cmp(b).unwrap());
    Ok(quantiles)
}

/// Training arguments structure
#[derive(Debug, Clone)]
pub struct TrainArgs {
    pub symbols: Vec<String>,
    pub data_paths: DataPaths,
    pub config_path: String,
    pub output_path: String,
    pub auto_optimize: bool,
    pub strategy: Option<String>,
}

/// Prediction arguments structure
#[derive(Debug, Clone)]
pub struct PredictArgs {
    pub symbols: Vec<String>,
    pub input_paths: DataPaths,
    pub model_path: String,
    pub output_path: String,
    pub quantiles: Vec<f64>,
    pub include_regime: bool,
    pub include_correlations: bool,
}

/// Validate symbol compatibility between training and prediction
pub fn validate_symbol_compatibility(
    train_symbols: &[String],
    predict_symbols: &[String],
) -> Result<(), String> {
    // For single symbol models, prediction must use same symbol
    // For single symbol models, prediction must use same symbol
    if train_symbols.len() == 1
        && predict_symbols.len() == 1
        && train_symbols[0] != predict_symbols[0]
    {
        return Err(format!(
            "Symbol mismatch: model trained on {}, prediction requested for {}",
            train_symbols[0], predict_symbols[0]
        ));
    }

    // For multi-symbol models, prediction symbols must be subset of training symbols
    if train_symbols.len() > 1 {
        for predict_symbol in predict_symbols {
            if !train_symbols.contains(predict_symbol) {
                return Err(format!(
                    "Symbol {} not found in training symbols: {:?}",
                    predict_symbol, train_symbols
                ));
            }
        }
    }

    Ok(())
}

/// Generate example commands for help
pub fn generate_examples() -> Vec<(&'static str, &'static str)> {
    vec![
        (
            "Single symbol TFT training",
            "vanga train --symbol BTCUSDT --data data/BTCUSDT_1h.csv --output models/BTCUSDT_tft.model"
        ),
        (
            "Multi-symbol GNN training",
            "vanga train --symbol BTCUSDT,ETHUSDT,ADAUSDT --data-dir data/multi_asset/ --output models/portfolio_gnn.model"
        ),
        (
            "Auto-optimized training",
            "vanga train --symbol BTCUSDT --data data/BTCUSDT_1h.csv --auto-optimize --strategy crypto_optimized --output models/BTCUSDT_optimized.model"
        ),
        (
            "Single symbol prediction",
            "vanga predict --symbol BTCUSDT --input data/BTCUSDT_recent.csv --model models/BTCUSDT_tft.model"
        ),
        (
            "Multi-symbol portfolio prediction",
            "vanga predict --symbol BTCUSDT,ETHUSDT,ADAUSDT --input-dir data/recent/ --model models/portfolio_gnn.model --include-regime --include-correlations"
        ),
        (
            "Prediction with custom quantiles",
            "vanga predict --symbol BTCUSDT --input data/BTCUSDT_recent.csv --model models/BTCUSDT_tft.model --quantiles 0.1,0.9"
        ),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_single_symbol() {
        let symbols = parse_symbols("BTCUSDT");
        assert_eq!(symbols, vec!["BTCUSDT"]);
    }

    #[test]
    fn test_parse_multiple_symbols() {
        let symbols = parse_symbols("BTCUSDT,ETHUSDT,ADAUSDT");
        assert_eq!(symbols, vec!["BTCUSDT", "ETHUSDT", "ADAUSDT"]);
    }

    #[test]
    fn test_parse_symbols_with_spaces() {
        let symbols = parse_symbols("BTCUSDT, ETHUSDT , ADAUSDT");
        assert_eq!(symbols, vec!["BTCUSDT", "ETHUSDT", "ADAUSDT"]);
    }

    #[test]
    fn test_resolve_single_symbol_data_path() {
        let symbols = vec!["BTCUSDT".to_string()];
        let paths = resolve_data_paths(&symbols, Some("data/BTCUSDT.csv"), None).unwrap();

        match paths {
            DataPaths::Single(path) => assert_eq!(path, "data/BTCUSDT.csv"),
            _ => panic!("Expected single path"),
        }
    }

    #[test]
    fn test_resolve_multi_symbol_data_paths() {
        let symbols = vec!["BTCUSDT".to_string(), "ETHUSDT".to_string()];
        let paths = resolve_data_paths(&symbols, None, Some("data/multi_asset")).unwrap();

        match paths {
            DataPaths::Multi(paths) => {
                assert_eq!(
                    paths.get("BTCUSDT"),
                    Some(&"data/multi_asset/BTCUSDT_1h.csv".to_string())
                );
                assert_eq!(
                    paths.get("ETHUSDT"),
                    Some(&"data/multi_asset/ETHUSDT_1h.csv".to_string())
                );
            }
            _ => panic!("Expected multi paths"),
        }
    }

    #[test]
    fn test_config_selection() {
        assert_eq!(
            select_config(&["BTCUSDT".to_string()], None),
            "configs/tft_enhanced.toml"
        );
        assert_eq!(
            select_config(&["BTCUSDT".to_string(), "ETHUSDT".to_string()], None),
            "configs/tft_gnn_small_portfolio.toml"
        );
        assert_eq!(
            select_config(&vec!["BTC".to_string(); 6], None),
            "configs/tft_gnn_multi_asset.toml"
        );
    }

    #[test]
    fn test_symbol_compatibility_validation() {
        // Single symbol compatibility
        let train_symbols = vec!["BTCUSDT".to_string()];
        let predict_symbols = vec!["BTCUSDT".to_string()];
        assert!(validate_symbol_compatibility(&train_symbols, &predict_symbols).is_ok());

        // Single symbol mismatch
        let predict_symbols = vec!["ETHUSDT".to_string()];
        assert!(validate_symbol_compatibility(&train_symbols, &predict_symbols).is_err());

        // Multi-symbol subset
        let train_symbols = vec![
            "BTCUSDT".to_string(),
            "ETHUSDT".to_string(),
            "ADAUSDT".to_string(),
        ];
        let predict_symbols = vec!["BTCUSDT".to_string(), "ETHUSDT".to_string()];
        assert!(validate_symbol_compatibility(&train_symbols, &predict_symbols).is_ok());

        // Multi-symbol invalid
        let predict_symbols = vec!["DOTUSDT".to_string()];
        assert!(validate_symbol_compatibility(&train_symbols, &predict_symbols).is_err());
    }
}
