//! Data structures for market data and related types
//!
//! This module defines the core data structures used throughout the VANGA system
//! for representing cryptocurrency market data, predictions, and related information.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Core market data structure representing OHLCV data for a single time period
///
/// This structure represents a single candlestick/bar of market data including
/// timestamp, open/high/low/close prices, and volume.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MarketDataRow {
    /// Unix timestamp (seconds since epoch)
    pub timestamp: i64,
    /// Opening price for the period
    pub open: f64,
    /// Highest price during the period
    pub high: f64,
    /// Lowest price during the period
    pub low: f64,
    /// Closing price for the period
    pub close: f64,
    /// Trading volume during the period
    pub volume: f64,
}

impl MarketDataRow {
    /// Create a new MarketDataRow
    ///
    /// # Arguments
    /// * `timestamp` - Unix timestamp in seconds
    /// * `open` - Opening price
    /// * `high` - Highest price
    /// * `low` - Lowest price
    /// * `close` - Closing price
    /// * `volume` - Trading volume
    ///
    /// # Returns
    /// * `Self` - New MarketDataRow instance
    pub fn new(timestamp: i64, open: f64, high: f64, low: f64, close: f64, volume: f64) -> Self {
        Self {
            timestamp,
            open,
            high,
            low,
            close,
            volume,
        }
    }

    /// Validate the market data for consistency
    ///
    /// Checks that:
    /// - High >= Low
    /// - All prices are positive
    /// - Volume is non-negative
    ///
    /// # Returns
    /// * `bool` - True if data is valid
    pub fn is_valid(&self) -> bool {
        self.high >= self.low
            && self.open > 0.0
            && self.high > 0.0
            && self.low > 0.0
            && self.close > 0.0
            && self.volume >= 0.0
    }

    /// Get the typical price (HLC/3)
    pub fn typical_price(&self) -> f64 {
        (self.high + self.low + self.close) / 3.0
    }

    /// Get the price range (high - low)
    pub fn price_range(&self) -> f64 {
        self.high - self.low
    }

    /// Get the price change (close - open)
    pub fn price_change(&self) -> f64 {
        self.close - self.open
    }

    /// Get the price change percentage
    pub fn price_change_percent(&self) -> f64 {
        if self.open == 0.0 {
            0.0
        } else {
            ((self.close - self.open) / self.open) * 100.0
        }
    }

    /// Convert to DateTime<Utc> from timestamp
    pub fn datetime(&self) -> DateTime<Utc> {
        DateTime::from_timestamp(self.timestamp, 0).unwrap_or_default()
    }
}

impl Default for MarketDataRow {
    fn default() -> Self {
        Self {
            timestamp: 0,
            open: 0.0,
            high: 0.0,
            low: 0.0,
            close: 0.0,
            volume: 0.0,
        }
    }
}

/// Extended market data with additional computed fields
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtendedMarketData {
    /// Base market data
    #[serde(flatten)]
    pub base: MarketDataRow,
    /// Volume-weighted average price
    pub vwap: Option<f64>,
    /// Number of trades (if available)
    pub trade_count: Option<u64>,
    /// Bid-ask spread (if available)
    pub spread: Option<f64>,
}

impl ExtendedMarketData {
    /// Create from base MarketDataRow
    pub fn from_base(base: MarketDataRow) -> Self {
        Self {
            base,
            vwap: None,
            trade_count: None,
            spread: None,
        }
    }
}
