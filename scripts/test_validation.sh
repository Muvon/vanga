#!/bin/bash

# Feature validation and testing script
set -e

echo "🔍 Testing Feature Validation Tools"
echo "==================================="

# Test the Python validation script
echo "📊 Testing validation script with example data..."

python3 scripts/validate_features.py examples/btc_with_sentiment.csv --verbose

echo
echo "📝 Generating configuration for sentiment data..."
python3 scripts/validate_features.py examples/btc_with_sentiment.csv --generate-config /tmp/generated_sentiment.toml

echo
echo "📊 Testing validation script with on-chain data..."
python3 scripts/validate_features.py examples/btc_with_onchain.csv --verbose

echo
echo "📝 Generating configuration for on-chain data..."
python3 scripts/validate_features.py examples/btc_with_onchain.csv --generate-config /tmp/generated_onchain.toml

echo
echo "✅ Validation tools tested successfully!"
echo "Generated configurations:"
echo "  - /tmp/generated_sentiment.toml"
echo "  - /tmp/generated_onchain.toml"