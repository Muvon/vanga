// Cross-asset feature engineering for multi-symbol cryptocurrency analysis
use crate::config::features::CrossAssetConfig;
use crate::utils::error::{Result, VangaError};
use polars::prelude::*;
use std::collections::HashMap;

/// Cross-asset features calculated from multiple cryptocurrency symbols
#[derive(Debug, Clone)]
pub struct CrossAssetFeatures {
    /// BTC dominance in the portfolio (0.0-1.0)
    pub btc_dominance: Option<f64>,
    /// Internal fear/greed index (0.0=extreme fear, 1.0=extreme greed)
    pub market_sentiment: Option<f64>,
    /// ETH/BTC price ratio (when both symbols available)
    pub eth_btc_ratio: Option<f64>,
    /// Cross-symbol price correlation (-1.0 to 1.0)
    pub price_correlation: Option<f64>,
    /// Overall market momentum (-1.0=bearish, 1.0=bullish)
    pub market_momentum: Option<f64>,
    /// Market volatility clustering (0.0-1.0)
    pub volatility_clustering: Option<f64>,
}

/// Cross-asset feature generator for multi-symbol analysis
pub struct CrossAssetFeatureGenerator {
    config: CrossAssetConfig,
}

impl CrossAssetFeatureGenerator {
    pub fn new(config: CrossAssetConfig) -> Self {
        Self { config }
    }

    /// Generate cross-asset features from multiple symbol DataFrames
    pub async fn generate_cross_asset_features(
        &self,
        symbol_data: &HashMap<String, DataFrame>,
    ) -> Result<HashMap<String, DataFrame>> {
        if !self.config.enabled {
            log::debug!("Cross-asset features disabled, returning original data");
            return Ok(symbol_data.clone());
        }

        log::info!(
            "Generating cross-asset features for {} symbols",
            symbol_data.len()
        );

        // Validate required symbols are present
        self.validate_required_symbols(symbol_data)?;

        // Calculate cross-asset features from all symbol data
        let cross_features_vec = self.calculate_cross_asset_features(symbol_data).await?;

        // Add cross-asset features to each symbol's DataFrame
        let mut enhanced_data = HashMap::new();
        for (symbol, df) in symbol_data.iter() {
            let enhanced_df =
                self.add_cross_features_to_dataframe(df.clone(), &cross_features_vec)?;
            enhanced_data.insert(symbol.clone(), enhanced_df);
        }

        log::info!("Successfully added cross-asset features to all symbols");
        Ok(enhanced_data)
    }

    /// Validate that required symbols are present for cross-asset analysis
    fn validate_required_symbols(&self, symbol_data: &HashMap<String, DataFrame>) -> Result<()> {
        if self.config.min_symbols_required > 0
            && symbol_data.len() < self.config.min_symbols_required
        {
            return Err(VangaError::ConfigError(format!(
                "Cross-asset analysis requires at least {} symbols, but only {} provided",
                self.config.min_symbols_required,
                symbol_data.len()
            )));
        }

        // Check for required symbols
        for required_symbol in &self.config.required_symbols {
            if !symbol_data.contains_key(required_symbol) {
                log::warn!(
                    "Required symbol '{}' not found in data. Cross-asset features may be incomplete.",
                    required_symbol
                );
            }
        }

        Ok(())
    }

    /// Calculate cross-asset features from all symbol data
    async fn calculate_cross_asset_features(
        &self,
        symbol_data: &HashMap<String, DataFrame>,
    ) -> Result<Vec<CrossAssetFeatures>> {
        log::debug!(
            "Calculating cross-asset features from {} symbols",
            symbol_data.len()
        );

        // Extract price and volume data for all symbols
        let mut symbol_prices = HashMap::new();
        let mut symbol_volumes = HashMap::new();
        let mut min_length = usize::MAX;

        for (symbol, df) in symbol_data.iter() {
            let prices = extract_numeric_column(df, "close")?;
            let volumes = extract_numeric_column(df, "volume")?;

            min_length = min_length.min(prices.len());
            symbol_prices.insert(symbol.clone(), prices);
            symbol_volumes.insert(symbol.clone(), volumes);
        }

        // Calculate features for each time step
        let mut cross_features = Vec::with_capacity(min_length);

        for i in 0..min_length {
            let features = CrossAssetFeatures {
                btc_dominance: self.calculate_btc_dominance(&symbol_volumes, i),
                market_sentiment: self.calculate_market_sentiment(
                    &symbol_prices,
                    &symbol_volumes,
                    i,
                ),
                eth_btc_ratio: self.calculate_eth_btc_ratio(&symbol_prices, i),
                price_correlation: self.calculate_price_correlation(&symbol_prices, i),
                market_momentum: self.calculate_market_momentum(&symbol_prices, i),
                volatility_clustering: self.calculate_volatility_clustering(&symbol_prices, i),
            };
            cross_features.push(features);
        }

        log::debug!(
            "Calculated {} cross-asset feature vectors",
            cross_features.len()
        );
        Ok(cross_features)
    }

    /// Calculate BTC dominance based on volume
    fn calculate_btc_dominance(
        &self,
        symbol_volumes: &HashMap<String, Vec<f64>>,
        index: usize,
    ) -> Option<f64> {
        if !self.config.btc_dominance_enabled {
            return None;
        }

        let btc_volume = symbol_volumes.get("BTCUSDT")?.get(index)?;
        let total_volume: f64 = symbol_volumes
            .values()
            .filter_map(|volumes| volumes.get(index))
            .sum();

        if total_volume > 0.0 {
            Some(btc_volume / total_volume)
        } else {
            Some(0.0)
        }
    }

    /// Calculate internal fear/greed index from price velocity and volume spikes
    fn calculate_market_sentiment(
        &self,
        symbol_prices: &HashMap<String, Vec<f64>>,
        symbol_volumes: &HashMap<String, Vec<f64>>,
        index: usize,
    ) -> Option<f64> {
        if index < 20 {
            // Need at least 20 periods for calculation
            return None;
        }

        let mut sentiment_scores = Vec::new();

        for (symbol, prices) in symbol_prices.iter() {
            if let Some(volumes) = symbol_volumes.get(symbol) {
                // Calculate price velocity (rate of change)
                let current_price = prices.get(index)?;
                let prev_price = prices.get(index - 1)?;
                let price_change = if *prev_price > 0.0 {
                    (current_price - prev_price) / prev_price
                } else {
                    0.0
                };

                // Calculate volume spike (current vs 20-period average)
                let current_volume = volumes.get(index)?;
                let avg_volume: f64 = volumes
                    .iter()
                    .skip(index.saturating_sub(20))
                    .take(20)
                    .sum::<f64>()
                    / 20.0;

                let volume_spike = if avg_volume > 0.0 {
                    current_volume / avg_volume
                } else {
                    1.0
                };

                // Calculate volatility (20-period standard deviation)
                let recent_prices: Vec<f64> = prices
                    .iter()
                    .skip(index.saturating_sub(20))
                    .take(20)
                    .cloned()
                    .collect();

                let mean_price = recent_prices.iter().sum::<f64>() / recent_prices.len() as f64;
                let variance = recent_prices
                    .iter()
                    .map(|p| (p - mean_price).powi(2))
                    .sum::<f64>()
                    / recent_prices.len() as f64;
                let volatility = variance.sqrt() / mean_price;

                // Combine factors into sentiment score
                // Positive price change + high volume = greed
                // Negative price change + high volume = fear
                // High volatility = uncertainty (neutral)
                let sentiment = if volatility > 0.05 {
                    // High volatility = uncertainty
                    0.5
                } else if price_change > 0.0 && volume_spike > 1.5 {
                    // Greed
                    0.5 + (price_change * volume_spike).min(0.5)
                } else if price_change < 0.0 && volume_spike > 1.5 {
                    // Fear
                    0.5 + (price_change * volume_spike).max(-0.5)
                } else {
                    // Neutral
                    0.5
                };

                sentiment_scores.push(sentiment);
            }
        }

        if sentiment_scores.is_empty() {
            None
        } else {
            Some(sentiment_scores.iter().sum::<f64>() / sentiment_scores.len() as f64)
        }
    }

    /// Calculate ETH/BTC ratio when both symbols are available
    fn calculate_eth_btc_ratio(
        &self,
        symbol_prices: &HashMap<String, Vec<f64>>,
        index: usize,
    ) -> Option<f64> {
        if !self.config.eth_btc_ratio_enabled {
            return None;
        }

        let eth_price = symbol_prices.get("ETHUSDT")?.get(index)?;
        let btc_price = symbol_prices.get("BTCUSDT")?.get(index)?;

        if *btc_price > 0.0 {
            Some(eth_price / btc_price)
        } else {
            None
        }
    }

    /// Calculate cross-symbol price correlation (20-period rolling)
    fn calculate_price_correlation(
        &self,
        symbol_prices: &HashMap<String, Vec<f64>>,
        index: usize,
    ) -> Option<f64> {
        if index < 20 || symbol_prices.len() < 2 {
            return None;
        }

        let symbols: Vec<&String> = symbol_prices.keys().collect();
        let mut correlations = Vec::new();

        // Calculate pairwise correlations
        for i in 0..symbols.len() {
            for j in (i + 1)..symbols.len() {
                let prices1 = symbol_prices.get(symbols[i])?;
                let prices2 = symbol_prices.get(symbols[j])?;

                let recent1: Vec<f64> = prices1
                    .iter()
                    .skip(index.saturating_sub(20))
                    .take(20)
                    .cloned()
                    .collect();
                let recent2: Vec<f64> = prices2
                    .iter()
                    .skip(index.saturating_sub(20))
                    .take(20)
                    .cloned()
                    .collect();

                if let Some(corr) = calculate_correlation(&recent1, &recent2) {
                    correlations.push(corr);
                }
            }
        }

        if correlations.is_empty() {
            None
        } else {
            Some(correlations.iter().sum::<f64>() / correlations.len() as f64)
        }
    }

    /// Calculate overall market momentum
    fn calculate_market_momentum(
        &self,
        symbol_prices: &HashMap<String, Vec<f64>>,
        index: usize,
    ) -> Option<f64> {
        if index < 10 {
            return None;
        }

        let mut momentum_scores = Vec::new();

        for prices in symbol_prices.values() {
            let current_price = prices.get(index)?;
            let prev_price = prices.get(index - 10)?; // 10-period momentum

            if *prev_price > 0.0 {
                let momentum = (current_price - prev_price) / prev_price;
                momentum_scores.push(momentum);
            }
        }

        if momentum_scores.is_empty() {
            None
        } else {
            let avg_momentum = momentum_scores.iter().sum::<f64>() / momentum_scores.len() as f64;
            // Normalize to -1.0 to 1.0 range
            Some(avg_momentum.tanh())
        }
    }

    /// Calculate volatility clustering across symbols
    fn calculate_volatility_clustering(
        &self,
        symbol_prices: &HashMap<String, Vec<f64>>,
        index: usize,
    ) -> Option<f64> {
        if index < 20 {
            return None;
        }

        let mut clustering_scores = Vec::new();

        for prices in symbol_prices.values() {
            // Calculate recent volatility (10-period)
            let recent_prices: Vec<f64> = prices
                .iter()
                .skip(index.saturating_sub(10))
                .take(10)
                .cloned()
                .collect();

            // Calculate historical volatility (20-period)
            let historical_prices: Vec<f64> = prices
                .iter()
                .skip(index.saturating_sub(20))
                .take(20)
                .cloned()
                .collect();

            let recent_vol = calculate_volatility(&recent_prices);
            let historical_vol = calculate_volatility(&historical_prices);

            if historical_vol > 0.0 {
                let clustering = recent_vol / historical_vol;
                clustering_scores.push(clustering);
            }
        }

        if clustering_scores.is_empty() {
            None
        } else {
            let avg_clustering =
                clustering_scores.iter().sum::<f64>() / clustering_scores.len() as f64;
            // Normalize to 0.0-1.0 range
            Some((avg_clustering - 1.0).abs().min(1.0))
        }
    }

    /// Add cross-asset features to a DataFrame
    fn add_cross_features_to_dataframe(
        &self,
        mut df: DataFrame,
        cross_features: &[CrossAssetFeatures],
    ) -> Result<DataFrame> {
        let num_rows = df.height();
        if num_rows == 0 {
            return Ok(df.clone());
        }

        // Ensure the number of features matches the number of rows in the dataframe
        let features_to_apply = if cross_features.len() < num_rows {
            log::warn!(
                "Cross-asset features count ({}) is less than DataFrame rows ({}). Padding with defaults.",
                cross_features.len(),
                num_rows
            );
            // Create a padded feature vector
            let mut padded = cross_features.to_vec();
            let default_feature = CrossAssetFeatures {
                btc_dominance: None,
                market_sentiment: None,
                eth_btc_ratio: None,
                price_correlation: None,
                market_momentum: None,
                volatility_clustering: None,
            };
            padded.resize(num_rows, default_feature);
            padded
        } else {
            cross_features.to_vec()
        };

        let btc_dominance_values: Vec<f64> = features_to_apply
            .iter()
            .map(|f| f.btc_dominance.unwrap_or(0.0))
            .collect();
        let market_sentiment_values: Vec<f64> = features_to_apply
            .iter()
            .map(|f| f.market_sentiment.unwrap_or(0.5))
            .collect();
        let eth_btc_ratio_values: Vec<f64> = features_to_apply
            .iter()
            .map(|f| f.eth_btc_ratio.unwrap_or(0.0))
            .collect();
        let price_correlation_values: Vec<f64> = features_to_apply
            .iter()
            .map(|f| f.price_correlation.unwrap_or(0.0))
            .collect();
        let market_momentum_values: Vec<f64> = features_to_apply
            .iter()
            .map(|f| f.market_momentum.unwrap_or(0.0))
            .collect();
        let volatility_clustering_values: Vec<f64> = features_to_apply
            .iter()
            .map(|f| f.volatility_clustering.unwrap_or(0.0))
            .collect();

        if self.config.btc_dominance_enabled {
            df = df
                .with_column(Series::new("cross_btc_dominance".into(), btc_dominance_values).into_column())
                .map_err(|e| {
                    VangaError::FeatureError(format!("Failed to add cross_btc_dominance: {}", e))
                })?
                .clone();
        }

        // Only add cross_market_sentiment if sentiment analysis is enabled
        if self.config.sentiment_analysis.enabled {
            df = df
                .with_column(Series::new("cross_market_sentiment".into(), market_sentiment_values).into_column())
                .map_err(|e| {
                    VangaError::FeatureError(format!("Failed to add cross_market_sentiment: {}", e))
                })?
                .clone();
        }
        if self.config.eth_btc_ratio_enabled {
            df = df
                .with_column(Series::new("cross_eth_btc_ratio".into(), eth_btc_ratio_values).into_column())
                .map_err(|e| {
                    VangaError::FeatureError(format!("Failed to add cross_eth_btc_ratio: {}", e))
                })?
                .clone();
        }
        df = df
            .with_column(Series::new("cross_price_correlation".into(), price_correlation_values).into_column())
            .map_err(|e| {
                VangaError::FeatureError(format!("Failed to add cross_price_correlation: {}", e))
            })?
            .clone();
        df = df
            .with_column(Series::new("cross_market_momentum".into(), market_momentum_values).into_column())
            .map_err(|e| {
                VangaError::FeatureError(format!("Failed to add cross_market_momentum: {}", e))
            })?
            .clone();
        df = df
            .with_column(Series::new("cross_volatility_clustering".into(), volatility_clustering_values).into_column())
            .map_err(|e| {
                VangaError::FeatureError(format!(
                    "Failed to add cross_volatility_clustering: {}",
                    e
                ))
            })?
            .clone();

        Ok(df)
    }
}

/// Extract numeric column as Vec<f64> (reused from technical.rs pattern)
fn extract_numeric_column(df: &DataFrame, column_name: &str) -> Result<Vec<f64>> {
    let series = df
        .column(column_name)
        .map_err(|e| VangaError::DataError(format!("Column '{}' not found: {}", column_name, e)))?;

    let values: Result<Vec<f64>> = series
        .f64()
        .map_err(|e| {
            VangaError::DataError(format!(
                "Failed to convert column '{}' to f64: {}",
                column_name, e
            ))
        })?
        .into_iter()
        .map(|opt_val| {
            opt_val.ok_or_else(|| {
                VangaError::DataError(format!("Null value found in column '{}'", column_name))
            })
        })
        .collect();

    values
}

/// Calculate correlation between two price series
fn calculate_correlation(x: &[f64], y: &[f64]) -> Option<f64> {
    if x.len() != y.len() || x.len() < 2 {
        return None;
    }

    let n = x.len() as f64;
    let mean_x = x.iter().sum::<f64>() / n;
    let mean_y = y.iter().sum::<f64>() / n;

    let mut numerator = 0.0;
    let mut sum_sq_x = 0.0;
    let mut sum_sq_y = 0.0;

    for (xi, yi) in x.iter().zip(y.iter()) {
        let dx = xi - mean_x;
        let dy = yi - mean_y;
        numerator += dx * dy;
        sum_sq_x += dx * dx;
        sum_sq_y += dy * dy;
    }

    let denominator = (sum_sq_x * sum_sq_y).sqrt();
    if denominator > 0.0 {
        Some(numerator / denominator)
    } else {
        None
    }
}

/// Calculate volatility (standard deviation of returns)
fn calculate_volatility(prices: &[f64]) -> f64 {
    if prices.len() < 2 {
        return 0.0;
    }

    let returns: Vec<f64> = prices
        .windows(2)
        .map(|w| {
            if w[0] > 0.0 {
                (w[1] - w[0]) / w[0]
            } else {
                0.0
            }
        })
        .collect();

    if returns.is_empty() {
        return 0.0;
    }

    let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;
    let variance = returns
        .iter()
        .map(|r| (r - mean_return).powi(2))
        .sum::<f64>()
        / returns.len() as f64;

    variance.sqrt()
}
