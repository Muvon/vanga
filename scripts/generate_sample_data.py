#!/usr/bin/env python3
"""
Generate sample cryptocurrency data for VANGA LSTM training examples.
Creates realistic OHLCV data with custom features for testing.
"""

import csv
import random
import math
from datetime import datetime, timedelta
import sys

def generate_crypto_data(rows=1000, start_price=42000.0):
    """Generate realistic cryptocurrency data with custom features."""
    
    start_date = datetime(2024, 1, 1)
    data = []
    price = start_price
    
    # Market state variables for more realistic data
    trend = 1.0  # 1.0 = neutral, >1.0 = bullish, <1.0 = bearish
    volatility = 0.02
    volume_base = 2000
    
    for i in range(rows):
        timestamp = (start_date + timedelta(hours=i)).strftime('%Y-%m-%dT%H:%M:%SZ')
        
        # Evolve market conditions
        if i % 100 == 0:  # Change trend every 100 hours
            trend = random.uniform(0.98, 1.02)
            volatility = random.uniform(0.01, 0.04)
            volume_base = random.uniform(1500, 3500)
        
        # Price movement with trend and volatility
        price_change = random.gauss(0, volatility) * trend
        price *= (1 + price_change)
        
        # OHLC generation
        open_price = price
        high_variance = abs(random.gauss(0, volatility * 0.5))
        low_variance = abs(random.gauss(0, volatility * 0.5))
        
        high_price = price * (1 + high_variance)
        low_price = price * (1 - low_variance)
        close_price = price * (1 + random.gauss(0, volatility * 0.3))
        
        # Volume with realistic patterns
        volume_multiplier = 1 + abs(price_change) * 10  # Higher volume on big moves
        volume = volume_base * volume_multiplier * random.uniform(0.5, 2.0)
        
        # Custom features with realistic correlations
        
        # Social sentiment (0.2 to 1.0) - correlates with price trend
        sentiment_base = 0.6 + (trend - 1.0) * 2
        social_sentiment = max(0.2, min(1.0, sentiment_base + random.gauss(0, 0.1)))
        
        # Funding rate (-5% to 5%) - anti-correlates with price in short term
        funding_rate = -price_change * 0.5 + random.gauss(0, 0.01)
        funding_rate = max(-0.05, min(0.05, funding_rate))
        
        # Whale activity (large transactions) - increases during high volatility
        whale_base = 1000000 * (1 + abs(price_change) * 20)
        whale_activity = whale_base * random.uniform(0.5, 2.0)
        
        # Fear & Greed Index (0-100) - correlates with sentiment
        fear_greed_base = social_sentiment * 100
        fear_greed_index = max(0, min(100, fear_greed_base + random.gauss(0, 10)))
        
        data.append([
            timestamp, open_price, high_price, low_price, close_price, volume,
            social_sentiment, funding_rate, whale_activity, fear_greed_index
        ])
    
    return data

def generate_onchain_data(rows=1000, start_price=42000.0):
    """Generate cryptocurrency data with on-chain metrics."""
    
    start_date = datetime(2024, 1, 1)
    data = []
    price = start_price
    
    # Base values for on-chain metrics
    base_addresses = 950000
    base_tx_count = 300000
    base_realized_cap = 850000000000
    
    for i in range(rows):
        timestamp = (start_date + timedelta(hours=i)).strftime('%Y-%m-%dT%H:%M:%SZ')
        
        # Price movement
        price_change = random.gauss(0, 0.02)
        price *= (1 + price_change)
        
        # OHLC with proper relationships
        open_price = price
        
        # Generate realistic OHLC
        daily_range = abs(random.gauss(0, 0.015))
        high_price = price * (1 + daily_range)
        low_price = price * (1 - daily_range)
        close_price = price
        
        # Ensure proper OHLC constraints
        low_price = min(low_price, open_price, close_price)
        high_price = max(high_price, open_price, close_price)
        volume = random.uniform(1000, 5000)
        
        # On-chain metrics with realistic growth trends
        
        # Active addresses - slow growth with daily variance
        addresses_growth = 1 + (i * 0.0001) + random.gauss(0, 0.01)
        active_addresses = int(base_addresses * addresses_growth)
        
        # Transaction count - correlates with price activity
        tx_multiplier = 1 + abs(price_change) * 5
        transaction_count = int(base_tx_count * tx_multiplier * random.uniform(0.8, 1.2))
        
        # NVT Ratio (Network Value to Transactions) - inversely related to tx volume
        market_cap = price * 19700000  # Approximate BTC supply
        nvt_ratio = (market_cap / (transaction_count * price)) * 1000
        nvt_ratio = max(20, min(80, nvt_ratio + random.gauss(0, 5)))
        
        # MVRV Ratio (Market Value to Realized Value)
        realized_cap = base_realized_cap * (1 + i * 0.0001)
        mvrv_ratio = market_cap / realized_cap
        mvrv_ratio = max(0.8, min(3.0, mvrv_ratio + random.gauss(0, 0.1)))
        
        data.append([
            timestamp, open_price, high_price, low_price, close_price, volume,
            active_addresses, transaction_count, nvt_ratio, mvrv_ratio, realized_cap
        ])
    
    return data

def write_csv(filename, headers, data):
    """Write data to CSV file."""
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(data)
    print(f"Generated {len(data)} rows in {filename}")

def main():
    """Generate sample data files."""
    
    # Default to 1000 rows, but allow override
    rows = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    
    print(f"Generating {rows} rows of sample data...")
    
    # Generate sentiment data
    sentiment_data = generate_crypto_data(rows)
    sentiment_headers = [
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'social_sentiment', 'funding_rate', 'whale_activity', 'fear_greed_index'
    ]
    write_csv('examples/btc_with_sentiment.csv', sentiment_headers, sentiment_data)
    
    # Generate on-chain data
    onchain_data = generate_onchain_data(rows)
    onchain_headers = [
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'active_addresses', 'transaction_count', 'nvt_ratio', 'mvrv_ratio', 'realized_cap'
    ]
    write_csv('examples/btc_with_onchain.csv', onchain_headers, onchain_data)
    
    print("Sample data generation complete!")
    print("Files created:")
    print("  - examples/btc_with_sentiment.csv (sentiment features)")
    print("  - examples/btc_with_onchain.csv (on-chain features)")
    print(f"Each file contains {rows} rows of realistic cryptocurrency data")

if __name__ == '__main__':
    main()