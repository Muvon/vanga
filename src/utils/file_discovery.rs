// File discovery utilities for training and prediction
use crate::utils::error::{Result, VangaError};
use std::path::{Path, PathBuf};

/// Resolve data file path for a symbol
///
/// # Arguments
/// * `base_path` - Base path (file or directory)
/// * `symbol` - Trading symbol (e.g., "BTCUSDT")
///
/// # Returns
/// * For directory: `base_path/SYMBOL.csv`
/// * For file: `base_path` (assumes single symbol)
pub fn resolve_symbol_data_path<P: AsRef<Path>>(base_path: P, symbol: &str) -> Result<PathBuf> {
    let base_path = base_path.as_ref();

    if base_path.is_dir() {
        // Auto-discovery: look for SYMBOL.csv in directory
        let csv_path = base_path.join(format!("{}.csv", symbol));
        if !csv_path.exists() {
            return Err(VangaError::DataError(format!(
                "❌ Data file not found: {}\n💡 Expected file: {}",
                base_path.display(),
                csv_path.display()
            )));
        }
        Ok(csv_path)
    } else if base_path.is_file() {
        // Single file - assume it's for the requested symbol
        Ok(base_path.to_path_buf())
    } else {
        Err(VangaError::DataError(format!(
            "❌ Data path not found: {}\n💡 Provide a valid CSV file or directory with symbol files",
            base_path.display()
        )))
    }
}

/// Resolve data file paths for multiple symbols
///
/// # Arguments
/// * `base_path` - Base path (must be directory for multiple symbols)
/// * `symbols` - List of trading symbols
///
/// # Returns
/// * Map of symbol -> file path
pub fn resolve_multi_symbol_data_paths<P: AsRef<Path>>(
    base_path: P,
    symbols: &[String],
) -> Result<Vec<(String, PathBuf)>> {
    let base_path = base_path.as_ref();

    if symbols.len() == 1 {
        // Single symbol - can be file or directory
        let symbol = &symbols[0];
        let path = resolve_symbol_data_path(base_path, symbol)?;
        return Ok(vec![(symbol.clone(), path)]);
    }

    // Multiple symbols - must be directory
    if !base_path.is_dir() {
        return Err(VangaError::DataError(format!(
            "❌ Multiple symbols specified but data is not a directory: {}\n💡 Use a directory with individual CSV files for each symbol: {}/SYMBOL.csv",
            base_path.display(),
            base_path.display()
        )));
    }

    let mut symbol_paths = Vec::new();
    let mut missing_files = Vec::new();

    for symbol in symbols {
        let csv_path = base_path.join(format!("{}.csv", symbol));
        if csv_path.exists() {
            symbol_paths.push((symbol.clone(), csv_path));
        } else {
            missing_files.push(format!("{}.csv", symbol));
        }
    }

    if !missing_files.is_empty() {
        return Err(VangaError::DataError(format!(
            "❌ Missing data files in {}: {}\n💡 Expected files: {}",
            base_path.display(),
            missing_files.join(", "),
            symbols
                .iter()
                .map(|s| format!("{}.csv", s))
                .collect::<Vec<_>>()
                .join(", ")
        )));
    }

    Ok(symbol_paths)
}

/// Validate data path requirements for symbols
pub fn validate_data_path_for_symbols<P: AsRef<Path>>(
    base_path: P,
    symbols: &[String],
) -> Result<()> {
    let base_path = base_path.as_ref();

    if !base_path.exists() {
        return Err(VangaError::DataError(format!(
            "❌ Data path not found: {}\n💡 Create the directory or check the path.",
            base_path.display()
        )));
    }

    if base_path.is_file() && symbols.len() > 1 {
        return Err(VangaError::DataError(format!(
            "❌ Multiple symbols specified but data is a single file: {}\n💡 Use a directory with individual CSV files for each symbol: {}/SYMBOL.csv",
            base_path.display(),
            base_path.parent().unwrap_or_else(|| std::path::Path::new(".")).display()
        )));
    }

    Ok(())
}
