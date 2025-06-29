#!/bin/bash

# VANGA LSTM Custom Features Workflow Examples
set -e

echo "🚀 VANGA LSTM Custom Features Workflow Examples"
echo "================================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_step() {
    echo -e "${BLUE}📋 $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Create directories
mkdir -p models predictions

# Generate sample data
echo
print_step "Generating sample data..."
python3 scripts/generate_sample_data.py 2000
print_success "Sample data generated!"

echo
print_step "Example 1: Training with Custom Features"
echo "========================================"

print_step "Training BTCUSDT model with sentiment features..."
if cargo run -- train \
    --symbol BTCUSDT \
    --data examples/btc_with_sentiment.csv \
    --horizons 1h; then
    print_success "Custom features training completed!"
else
    print_error "Custom features training failed"
    exit 1
fi

echo
print_step "Making predictions..."
if cargo run -- predict \
    --symbol BTCUSDT \
    --input examples/btc_with_sentiment.csv \
    --output predictions/btc_predictions.json; then
    print_success "Predictions completed!"
else
    print_error "Predictions failed"
fi

echo
print_step "Example 2: Fresh Training with On-Chain Data"
echo "============================================="

print_step "Training fresh model with on-chain features..."
if cargo run -- train \
    --symbol BTCUSDT \
    --data examples/btc_with_onchain.csv \
    --features-config examples/basic_features.toml \
    --fresh; then
    print_success "Fresh training completed!"
else
    print_error "Fresh training failed"
fi

echo
print_step "Example 3: Continue Training"
echo "============================"

print_step "Continuing training with more sentiment data..."
if cargo run -- train \
    --symbol BTCUSDT \
    --data examples/btc_with_sentiment.csv \
    --horizons 1h; then
    print_success "Continue training completed!"
else
    print_error "Continue training failed"
fi

echo
print_step "Example 4: Model Management"
echo "==========================="

print_step "Listing available models..."
if cargo run -- models list; then
    print_success "Model listing completed!"
else
    print_error "Model listing failed"
fi

echo
print_success "All examples completed successfully!"
print_step "Check outputs:"
echo "  - Models: ./models/"
echo "  - Predictions: ./predictions/"
echo "  - Sample data: ./examples/"

echo
print_step "Usage Summary:"
echo "1. Generate data: python3 scripts/generate_sample_data.py [rows]"
echo "2. Train model: vanga train --symbol SYMBOL --data data.csv"
echo "3. Continue training: vanga train --symbol SYMBOL --data more_data.csv"
echo "4. Fresh training: vanga train --symbol SYMBOL --data data.csv --fresh"
echo "5. Predict: vanga predict --symbol SYMBOL --input data.csv"
