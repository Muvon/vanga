use crate::utils::file_discovery::*;
use std::fs;
use tempfile::TempDir;

#[test]
fn test_resolve_single_symbol_file() {
    let temp_dir = TempDir::new().unwrap();
    let csv_file = temp_dir.path().join("test.csv");
    fs::write(&csv_file, "test data").unwrap();

    let result = resolve_symbol_data_path(&csv_file, "BTCUSDT").unwrap();
    assert_eq!(result, csv_file);
}

#[test]
fn test_resolve_single_symbol_directory() {
    let temp_dir = TempDir::new().unwrap();
    let csv_file = temp_dir.path().join("BTCUSDT.csv");
    fs::write(&csv_file, "test data").unwrap();

    let result = resolve_symbol_data_path(temp_dir.path(), "BTCUSDT").unwrap();
    assert_eq!(result, csv_file);
}

#[test]
fn test_resolve_multi_symbol_directory() {
    let temp_dir = TempDir::new().unwrap();
    let btc_file = temp_dir.path().join("BTCUSDT.csv");
    let eth_file = temp_dir.path().join("ETHUSDT.csv");
    fs::write(&btc_file, "btc data").unwrap();
    fs::write(&eth_file, "eth data").unwrap();

    let symbols = vec!["BTCUSDT".to_string(), "ETHUSDT".to_string()];
    let result = resolve_multi_symbol_data_paths(temp_dir.path(), &symbols).unwrap();

    assert_eq!(result.len(), 2);
    assert!(result.contains(&("BTCUSDT".to_string(), btc_file)));
    assert!(result.contains(&("ETHUSDT".to_string(), eth_file)));
}
