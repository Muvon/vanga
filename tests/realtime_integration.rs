//! Integration tests for real-time streaming prediction functionality

use std::io::Write;
use std::path::PathBuf;
use std::time::Duration;
use tempfile::NamedTempFile;
use tokio::time::timeout;

use vanga::realtime::{start_realtime_prediction, OutputFormat, RealtimeConfig};

/// Test data for integration tests
const TEST_CSV_HEADER: &str = "timestamp,open,high,low,close,volume";
const TEST_CSV_DATA: &[&str] = &[
    "1640995200,47000.5,47100.0,46900.0,47050.0,1234.56",
    "1640995260,47050.0,47150.0,47000.0,47080.0,1456.78",
    "1640995320,47080.0,47200.0,47050.0,47150.0,1678.90",
    "1640995380,47150.0,47250.0,47100.0,47200.0,1890.12",
    "1640995440,47200.0,47300.0,47150.0,47250.0,2012.34",
];

/// Create a test CSV file with initial data
async fn create_test_csv_file() -> NamedTempFile {
    let mut temp_file = NamedTempFile::new().expect("Failed to create temp file");

    // Write header
    writeln!(temp_file, "{}", TEST_CSV_HEADER).unwrap();

    // Write initial data
    for line in TEST_CSV_DATA {
        writeln!(temp_file, "{}", line).unwrap();
    }

    temp_file.flush().unwrap();
    temp_file
}

/// Create test configuration
fn create_test_config(file_path: PathBuf) -> RealtimeConfig {
    RealtimeConfig {
        file_path,
        symbol: "BTCUSDT".to_string(),
        poll_interval: Duration::from_millis(100), // Fast polling for tests
        buffer_size: 50,
        feature_window: 5, // Small window for faster tests
        output_format: OutputFormat::Json,
        debug: true,
    }
}

#[tokio::test]
#[ignore] // Ignore by default as it requires a trained model
async fn test_realtime_prediction_integration() {
    // This test requires a trained model to exist
    // Run with: cargo test test_realtime_prediction_integration -- --ignored

    let temp_file = create_test_csv_file().await;
    let config = create_test_config(temp_file.path().to_path_buf());

    // Start real-time prediction with timeout
    let result = timeout(Duration::from_secs(5), start_realtime_prediction(config)).await;

    // Should timeout (which is expected behavior for continuous streaming)
    assert!(result.is_err());
}

#[tokio::test]
async fn test_realtime_config_creation() {
    let temp_file = create_test_csv_file().await;
    let config = create_test_config(temp_file.path().to_path_buf());

    assert_eq!(config.symbol, "BTCUSDT");
    assert_eq!(config.buffer_size, 50);
    assert_eq!(config.feature_window, 5);
    assert!(matches!(config.output_format, OutputFormat::Json));
}

#[tokio::test]
async fn test_output_format_variants() {
    let temp_file = create_test_csv_file().await;

    // Test JSON format
    let config_json = RealtimeConfig {
        output_format: OutputFormat::Json,
        ..create_test_config(temp_file.path().to_path_buf())
    };
    assert!(matches!(config_json.output_format, OutputFormat::Json));

    // Test CSV format
    let config_csv = RealtimeConfig {
        output_format: OutputFormat::Csv,
        ..create_test_config(temp_file.path().to_path_buf())
    };
    assert!(matches!(config_csv.output_format, OutputFormat::Csv));

    // Test Pretty format
    let config_pretty = RealtimeConfig {
        output_format: OutputFormat::Pretty,
        ..create_test_config(temp_file.path().to_path_buf())
    };
    assert!(matches!(config_pretty.output_format, OutputFormat::Pretty));
}
