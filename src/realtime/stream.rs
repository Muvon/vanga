//! CSV streaming functionality for incremental file reading
//!
//! This module provides efficient incremental CSV reading capabilities for real-time
//! data processing. It tracks file position to read only new lines as they are appended.

use crate::data::structures::MarketDataRow;
use crate::utils::error::{Result, VangaError};
use chrono::{DateTime, Utc};
use std::fs::File;
use std::io::{BufRead, BufReader, Seek, SeekFrom};
use std::path::PathBuf;

/// Incremental CSV file reader with position tracking
///
/// This struct efficiently reads only new lines from a CSV file by maintaining
/// the last read position. It's designed for real-time streaming scenarios where
/// data is continuously appended to a file.
pub struct CsvStreamer {
    file_path: PathBuf,
    file_position: u64,
    line_count: usize,
    last_read_time: DateTime<Utc>,
}

impl CsvStreamer {
    /// Create a new CSV streamer for the specified file
    ///
    /// # Arguments
    /// * `file_path` - Path to the CSV file to stream
    ///
    /// # Returns
    /// * `Self` - New CsvStreamer instance
    ///
    /// # Example
    /// ```rust,no_run
    /// use vanga::realtime::CsvStreamer;
    /// use std::path::PathBuf;
    ///
    /// let streamer = CsvStreamer::new(PathBuf::from("data/live.csv"));
    /// ```
    pub fn new(file_path: PathBuf) -> Self {
        Self {
            file_path,
            file_position: 0,
            line_count: 0,
            last_read_time: Utc::now(),
        }
    }

    /// Read new lines that have been appended since the last read
    ///
    /// This method efficiently reads only the new content by seeking to the last
    /// known position in the file. It parses each line as a MarketDataRow.
    ///
    /// # Returns
    /// * `Result<Vec<MarketDataRow>>` - Vector of new market data rows or error
    ///
    /// # Example
    /// ```rust,no_run
    /// use vanga::realtime::CsvStreamer;
    /// use std::path::PathBuf;
    ///
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let mut streamer = CsvStreamer::new(PathBuf::from("data/live.csv"));
    /// let new_rows = streamer.read_new_lines().await?;
    /// println!("Read {} new rows", new_rows.len());
    /// # Ok(())
    /// # }
    /// ```
    pub async fn read_new_lines(&mut self) -> Result<Vec<MarketDataRow>> {
        // Check if file exists
        if !self.file_path.exists() {
            return Err(VangaError::IoError(format!(
                "CSV file does not exist: {}",
                self.file_path.display()
            )));
        }

        let mut file = File::open(&self.file_path).map_err(|e| {
            VangaError::IoError(format!(
                "Failed to open CSV file {}: {}",
                self.file_path.display(),
                e
            ))
        })?;

        // Get current file size to check if file has grown
        let file_size = file
            .metadata()
            .map_err(|e| VangaError::IoError(format!("Failed to get file metadata: {}", e)))?
            .len();

        if file_size <= self.file_position {
            // No new data available
            log::debug!(
                "No new data in file (size: {}, position: {})",
                file_size,
                self.file_position
            );
            return Ok(Vec::new());
        }

        // Seek to last known position
        file.seek(SeekFrom::Start(self.file_position))
            .map_err(|e| {
                VangaError::IoError(format!(
                    "Failed to seek to position {}: {}",
                    self.file_position, e
                ))
            })?;

        let mut reader = BufReader::new(file);
        let mut new_rows = Vec::new();
        let mut line = String::new();

        loop {
            line.clear();
            match reader.read_line(&mut line) {
                Ok(0) => break, // EOF reached
                Ok(bytes_read) => {
                    self.file_position += bytes_read as u64;

                    // Skip empty lines and comments
                    let trimmed = line.trim();
                    if trimmed.is_empty() || trimmed.starts_with('#') {
                        continue;
                    }

                    // Skip header line if this is the first read
                    if self.line_count == 0 && trimmed.to_lowercase().contains("timestamp") {
                        self.line_count += 1;
                        continue;
                    }

                    // Parse CSV line to MarketDataRow
                    match self.parse_csv_line(trimmed) {
                        Ok(data_row) => {
                            new_rows.push(data_row);
                            self.line_count += 1;
                        }
                        Err(e) => {
                            log::warn!(
                                "Failed to parse CSV line {}: '{}', error: {}",
                                self.line_count + 1,
                                trimmed,
                                e
                            );
                            // Continue processing other lines
                            continue;
                        }
                    }
                }
                Err(e) => {
                    return Err(VangaError::IoError(format!("Failed to read line: {}", e)));
                }
            }
        }

        self.last_read_time = Utc::now();

        if !new_rows.is_empty() {
            log::info!(
                "Read {} new rows from CSV file (total lines processed: {})",
                new_rows.len(),
                self.line_count
            );
        }

        Ok(new_rows)
    }

    /// Parse a single CSV line into a MarketDataRow
    ///
    /// Expected CSV format: timestamp,open,high,low,close,volume
    /// Supports both Unix timestamps and ISO 8601 datetime strings.
    ///
    /// # Arguments
    /// * `line` - CSV line to parse
    ///
    /// # Returns
    /// * `Result<MarketDataRow>` - Parsed market data or error
    fn parse_csv_line(&self, line: &str) -> Result<MarketDataRow> {
        let parts: Vec<&str> = line.split(',').collect();

        if parts.len() < 6 {
            return Err(VangaError::DataError(format!(
                "Invalid CSV format: expected 6 columns, got {}. Line: '{}'",
                parts.len(),
                line
            )));
        }

        // Parse timestamp (support both Unix timestamp and ISO 8601)
        let timestamp = if parts[0].contains('T') || parts[0].contains('-') {
            // ISO 8601 format
            DateTime::parse_from_rfc3339(parts[0])
                .map_err(|e| {
                    VangaError::DataError(format!("Invalid ISO timestamp '{}': {}", parts[0], e))
                })?
                .timestamp()
        } else {
            // Unix timestamp
            parts[0].parse::<i64>().map_err(|e| {
                VangaError::DataError(format!("Invalid Unix timestamp '{}': {}", parts[0], e))
            })?
        };

        // Parse OHLCV data
        let open = parts[1].parse::<f64>().map_err(|e| {
            VangaError::DataError(format!("Invalid open price '{}': {}", parts[1], e))
        })?;

        let high = parts[2].parse::<f64>().map_err(|e| {
            VangaError::DataError(format!("Invalid high price '{}': {}", parts[2], e))
        })?;

        let low = parts[3].parse::<f64>().map_err(|e| {
            VangaError::DataError(format!("Invalid low price '{}': {}", parts[3], e))
        })?;

        let close = parts[4].parse::<f64>().map_err(|e| {
            VangaError::DataError(format!("Invalid close price '{}': {}", parts[4], e))
        })?;

        let volume = parts[5]
            .parse::<f64>()
            .map_err(|e| VangaError::DataError(format!("Invalid volume '{}': {}", parts[5], e)))?;

        // Validate OHLC relationships
        if high < low {
            return Err(VangaError::DataError(format!(
                "Invalid OHLC: high ({}) < low ({})",
                high, low
            )));
        }
        if open < 0.0 || high < 0.0 || low < 0.0 || close < 0.0 {
            return Err(VangaError::DataError(
                "Invalid OHLC: negative prices not allowed".to_string(),
            ));
        }
        if volume < 0.0 {
            return Err(VangaError::DataError(
                "Invalid volume: negative values not allowed".to_string(),
            ));
        }

        Ok(MarketDataRow {
            timestamp,
            open,
            high,
            low,
            close,
            volume,
        })
    }

    /// Get the current file position
    pub fn get_position(&self) -> u64 {
        self.file_position
    }

    /// Get the number of lines processed
    pub fn get_line_count(&self) -> usize {
        self.line_count
    }

    /// Get the last read time
    pub fn get_last_read_time(&self) -> DateTime<Utc> {
        self.last_read_time
    }

    /// Reset the streamer to read from the beginning of the file
    pub fn reset(&mut self) {
        self.file_position = 0;
        self.line_count = 0;
        self.last_read_time = Utc::now();
        log::info!("CSV streamer reset to beginning of file");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[tokio::test]
    async fn test_csv_streamer_creation() {
        let temp_file = NamedTempFile::new().unwrap();
        let streamer = CsvStreamer::new(temp_file.path().to_path_buf());
        assert_eq!(streamer.get_position(), 0);
        assert_eq!(streamer.get_line_count(), 0);
    }

    #[tokio::test]
    async fn test_read_new_lines_empty_file() {
        let temp_file = NamedTempFile::new().unwrap();
        let mut streamer = CsvStreamer::new(temp_file.path().to_path_buf());
        let result = streamer.read_new_lines().await.unwrap();
        assert!(result.is_empty());
    }

    #[tokio::test]
    async fn test_read_new_lines_with_data() {
        let mut temp_file = NamedTempFile::new().unwrap();
        writeln!(temp_file, "timestamp,open,high,low,close,volume").unwrap();
        writeln!(
            temp_file,
            "1640995200,47000.5,47100.0,46900.0,47050.0,1234.56"
        )
        .unwrap();
        temp_file.flush().unwrap();

        let mut streamer = CsvStreamer::new(temp_file.path().to_path_buf());
        let result = streamer.read_new_lines().await.unwrap();

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].timestamp, 1640995200);
        assert_eq!(result[0].open, 47000.5);
        assert_eq!(result[0].close, 47050.0);
    }

    #[tokio::test]
    async fn test_incremental_reading() {
        let mut temp_file = NamedTempFile::new().unwrap();
        writeln!(temp_file, "timestamp,open,high,low,close,volume").unwrap();
        writeln!(
            temp_file,
            "1640995200,47000.5,47100.0,46900.0,47050.0,1234.56"
        )
        .unwrap();
        temp_file.flush().unwrap();

        let mut streamer = CsvStreamer::new(temp_file.path().to_path_buf());

        // First read
        let result1 = streamer.read_new_lines().await.unwrap();
        assert_eq!(result1.len(), 1);

        // Add more data
        writeln!(
            temp_file,
            "1640995260,47050.0,47150.0,47000.0,47080.0,1456.78"
        )
        .unwrap();
        temp_file.flush().unwrap();

        // Second read should only get new data
        let result2 = streamer.read_new_lines().await.unwrap();
        assert_eq!(result2.len(), 1);
        assert_eq!(result2[0].timestamp, 1640995260);
    }

    #[test]
    fn test_parse_csv_line_valid() {
        let streamer = CsvStreamer::new(PathBuf::from("test.csv"));
        let line = "1640995200,47000.5,47100.0,46900.0,47050.0,1234.56";
        let result = streamer.parse_csv_line(line).unwrap();

        assert_eq!(result.timestamp, 1640995200);
        assert_eq!(result.open, 47000.5);
        assert_eq!(result.high, 47100.0);
        assert_eq!(result.low, 46900.0);
        assert_eq!(result.close, 47050.0);
        assert_eq!(result.volume, 1234.56);
    }

    #[test]
    fn test_parse_csv_line_invalid_format() {
        let streamer = CsvStreamer::new(PathBuf::from("test.csv"));
        let line = "1640995200,47000.5,47100.0"; // Missing columns
        let result = streamer.parse_csv_line(line);
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_csv_line_invalid_ohlc() {
        let streamer = CsvStreamer::new(PathBuf::from("test.csv"));
        let line = "1640995200,47000.5,46900.0,47100.0,47050.0,1234.56"; // high < low
        let result = streamer.parse_csv_line(line);
        assert!(result.is_err());
    }
}
