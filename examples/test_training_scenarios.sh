#!/bin/bash

# Test script for VANGA training continuation scenarios
set -e

echo "🧪 Testing VANGA Training Continuation Scenarios"
echo "================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_step() {
    echo -e "${YELLOW}▶ $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Ensure we have sample data
if [ ! -f "examples/btc_with_sentiment.csv" ]; then
    print_step "Generating sample data..."
    python3 scripts/generate_sample_data.py
fi

# Clean up any existing models
rm -rf models/
mkdir -p models/

echo
print_step "Test 1: Fresh Training (--fresh flag)"
echo "======================================"
if timeout 60s cargo run -- train --symbol TESTCOIN --data examples/btc_with_sentiment.csv --horizons 1h --fresh; then
    print_success "Fresh training completed successfully"
else
    print_error "Fresh training failed"
    exit 1
fi

echo
print_step "Test 2: Default Behavior (should continue existing model)"
echo "========================================================"
if timeout 60s cargo run -- train --symbol TESTCOIN --data examples/btc_with_sentiment.csv --horizons 1h; then
    print_success "Default training continuation completed"
else
    print_error "Default training continuation failed"
    exit 1
fi

echo
print_step "Test 3: Explicit Continue Training (--continue-training flag)"
echo "=============================================================="
if timeout 60s cargo run -- train --symbol TESTCOIN --data examples/btc_with_sentiment.csv --horizons 1h --continue-training; then
    print_success "Explicit continue training completed"
else
    print_error "Explicit continue training failed"
    exit 1
fi

echo
print_step "Test 4: Continue Training with No Model (should fail)"
echo "====================================================="
if timeout 30s cargo run -- train --symbol NEWCOIN --data examples/btc_with_sentiment.csv --horizons 1h --continue-training 2>&1 | grep -q "no existing model found"; then
    print_success "Correctly failed when no model exists for continue training"
else
    print_error "Should have failed when no model exists for continue training"
    exit 1
fi

echo
print_step "Test 5: Fresh Training Override (--fresh should ignore existing model)"
echo "====================================================================="
if timeout 60s cargo run -- train --symbol TESTCOIN --data examples/btc_with_sentiment.csv --horizons 1h --fresh; then
    print_success "Fresh training override completed successfully"
else
    print_error "Fresh training override failed"
    exit 1
fi

echo
echo "🎉 All training continuation scenarios completed successfully!"
echo
echo "Summary:"
echo "✅ Fresh training works"
echo "✅ Default continuation works"
echo "✅ Explicit continuation works"
echo "✅ Proper error handling for missing models"
echo "✅ Fresh training override works"
echo
echo "Models created:"
ls -la models/
