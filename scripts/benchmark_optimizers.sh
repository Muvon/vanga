#!/bin/bash
# VANGA Optimizer Benchmark Script (Shell Version)
# Simple shell script to benchmark all optimizers and compare results

set -e

# Default values
DATA_PATH=""
SYMBOL=""
OUTPUT_DIR="benchmark_results"
QUICK_MODE=false
OPTIMIZERS=("AdamW" "SGD" "Adam" "RMSprop" "NAdam" "RAdam" "AdaMax" "AdaDelta" "AdaGrad")

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to show usage
show_usage() {
    cat << EOF
VANGA Optimizer Benchmark Script

Usage: $0 --data <path> --symbol <symbol> [options]

Required:
  --data <path>      Path to CSV data file
  --symbol <symbol>  Trading symbol (e.g., BTCUSDT)

Options:
  --output <dir>     Output directory (default: benchmark_results)
  --quick           Quick benchmark with reduced epochs
  --optimizers <list> Space-separated list of optimizers to test
                     Available: AdamW SGD Adam RMSprop NAdam RAdam AdaMax AdaDelta AdaGrad
  --help            Show this help message

Examples:
  $0 --data data/BTCUSDT_1h.csv --symbol BTCUSDT
  $0 --data data/BTCUSDT_1h.csv --symbol BTCUSDT --quick
  $0 --data data/BTCUSDT_1h.csv --symbol BTCUSDT --optimizers "AdamW RMSprop NAdam"
EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --data)
            DATA_PATH="$2"
            shift 2
            ;;
        --symbol)
            SYMBOL="$2"
            shift 2
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --quick)
            QUICK_MODE=true
            shift
            ;;
        --optimizers)
            shift
            OPTIMIZERS=()
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                OPTIMIZERS+=("$1")
                shift
            done
            ;;
        --help)
            show_usage
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Validate required arguments
if [[ -z "$DATA_PATH" ]]; then
    print_error "Data path is required"
    show_usage
    exit 1
fi

if [[ -z "$SYMBOL" ]]; then
    print_error "Symbol is required"
    show_usage
    exit 1
fi

if [[ ! -f "$DATA_PATH" ]]; then
    print_error "Data file not found: $DATA_PATH"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Initialize results file
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULTS_FILE="$OUTPUT_DIR/benchmark_results_${SYMBOL}_${TIMESTAMP}.csv"
REPORT_FILE="$OUTPUT_DIR/benchmark_report_${SYMBOL}_${TIMESTAMP}.md"

# Create CSV header
echo "Optimizer,Config,Status,FinalValLoss,BestValLoss,Epochs,TrainingTime,Converged,EarlyStopped,Error" > "$RESULTS_FILE"

print_status "Starting VANGA Optimizer Benchmark"
print_status "Symbol: $SYMBOL"
print_status "Data: $DATA_PATH"
print_status "Mode: $([ "$QUICK_MODE" = true ] && echo "Quick" || echo "Full")"
print_status "Output: $OUTPUT_DIR"
print_status "Testing optimizers: ${OPTIMIZERS[*]}"
echo

# Function to run single optimizer benchmark
run_optimizer_benchmark() {
    local optimizer=$1
    local config_file="configs/optimizer_examples/${optimizer,,}_"

    # Map optimizer names to config files
    case $optimizer in
        "AdamW")
            config_file="configs/optimizer_examples/adamw_crypto_optimized.toml"
            ;;
        "SGD")
            config_file="configs/optimizer_examples/sgd_fine_tuning.toml"
            ;;
        "Adam")
            config_file="configs/optimizer_examples/adam_general_purpose.toml"
            ;;
        "RMSprop")
            config_file="configs/optimizer_examples/rmsprop_volatile_markets.toml"
            ;;
        "NAdam")
            config_file="configs/optimizer_examples/nadam_momentum_markets.toml"
            ;;
        "RAdam")
            config_file="configs/optimizer_examples/radam_stable_convergence.toml"
            ;;
        "AdaMax")
            config_file="configs/optimizer_examples/adamax_large_gradients.toml"
            ;;
        "AdaDelta")
            config_file="configs/optimizer_examples/adadelta_sparse_data.toml"
            ;;
        "AdaGrad")
            config_file="configs/optimizer_examples/adagrad_short_training.toml"
            ;;
        *)
            print_error "Unknown optimizer: $optimizer"
            return 1
            ;;
    esac

    if [[ ! -f "$config_file" ]]; then
        print_error "Config file not found: $config_file"
        echo "$optimizer,$config_file,ERROR,inf,inf,0,0,false,false,Config file not found" >> "$RESULTS_FILE"
        return 1
    fi

    print_status "Benchmarking $optimizer optimizer..."

    # Create temporary config for benchmarking
    local temp_config="temp_benchmark_${optimizer}_${TIMESTAMP}.toml"
    cp "$config_file" "$temp_config"

    # Modify config for benchmarking if quick mode
    if [[ "$QUICK_MODE" = true ]]; then
        # Reduce epochs for quick benchmarking (requires toml manipulation)
        # For now, we'll use the config as-is and rely on timeout
        print_warning "Quick mode: using original config with timeout"
    fi

    # Run VANGA training
    local start_time=$(date +%s)
    local timeout_duration=3600  # 1 hour
    if [[ "$QUICK_MODE" = true ]]; then
        timeout_duration=600  # 10 minutes for quick mode
    fi

    local log_file="$OUTPUT_DIR/${optimizer}_${TIMESTAMP}.log"

    if timeout "$timeout_duration" cargo run --release -- train \
        --symbol "$SYMBOL" \
        --data "$DATA_PATH" \
        --config "$temp_config" \
        --fresh > "$log_file" 2>&1; then

        local end_time=$(date +%s)
        local training_time=$((end_time - start_time))

        # Parse results from log file
        local final_val_loss=$(grep "Val Loss:" "$log_file" | tail -1 | sed -n 's/.*Val Loss: \([0-9.]*\).*/\1/p' || echo "inf")
        local best_val_loss=$(grep "best validation loss" "$log_file" | sed -n 's/.*: \([0-9.]*\).*/\1/p' || echo "$final_val_loss")
        local epochs=$(grep "Epoch " "$log_file" | wc -l || echo "0")
        local converged=$(grep -q "Training completed" "$log_file" && echo "true" || echo "false")
        local early_stopped=$(grep -q "Early stopping" "$log_file" && echo "true" || echo "false")

        print_success "$optimizer: $final_val_loss val_loss in $epochs epochs (${training_time}s)"
        echo "$optimizer,$config_file,SUCCESS,$final_val_loss,$best_val_loss,$epochs,$training_time,$converged,$early_stopped," >> "$RESULTS_FILE"

    else
        local end_time=$(date +%s)
        local training_time=$((end_time - start_time))

        # Check if it was a timeout or error
        if [[ $? -eq 124 ]]; then
            print_warning "$optimizer timed out after ${timeout_duration}s"
            echo "$optimizer,$config_file,TIMEOUT,inf,inf,0,$training_time,false,false,Training timeout" >> "$RESULTS_FILE"
        else
            local error_msg=$(tail -5 "$log_file" | tr '\n' ' ' | sed 's/,/;/g')
            print_error "$optimizer failed: $error_msg"
            echo "$optimizer,$config_file,ERROR,inf,inf,0,$training_time,false,false,$error_msg" >> "$RESULTS_FILE"
        fi
    fi

    # Cleanup
    rm -f "$temp_config"

    # Brief pause between runs
    sleep 2
}

# Run benchmarks for all optimizers
for optimizer in "${OPTIMIZERS[@]}"; do
    run_optimizer_benchmark "$optimizer"
done

# Generate summary report
print_status "Generating benchmark report..."

cat > "$REPORT_FILE" << EOF
# VANGA Optimizer Benchmark Report

**Symbol:** $SYMBOL
**Data:** $DATA_PATH
**Generated:** $(date)
**Mode:** $([ "$QUICK_MODE" = true ] && echo "Quick" || echo "Full")

## Results Summary

EOF

# Add CSV results to report
echo "| Optimizer | Status | Best Val Loss | Epochs | Time (s) | Converged | Early Stopped |" >> "$REPORT_FILE"
echo "|-----------|--------|---------------|--------|----------|-----------|---------------|" >> "$REPORT_FILE"

# Parse CSV and create markdown table
tail -n +2 "$RESULTS_FILE" | while IFS=',' read -r optimizer config status final_val best_val epochs time converged early_stopped error; do
    if [[ "$status" == "SUCCESS" ]]; then
        echo "| $optimizer | ✅ $status | $best_val | $epochs | $time | $converged | $early_stopped |" >> "$REPORT_FILE"
    else
        echo "| $optimizer | ❌ $status | - | - | $time | false | false |" >> "$REPORT_FILE"
    fi
done

# Add recommendations
cat >> "$REPORT_FILE" << EOF

## Recommendations

Based on the benchmark results:

1. **Best Performance:** Check the optimizer with the lowest validation loss
2. **Fastest Training:** Look for the optimizer with the shortest training time
3. **Most Stable:** Consider optimizers that converged without early stopping

## Files Generated

- **Detailed Results:** $RESULTS_FILE
- **Individual Logs:** $OUTPUT_DIR/*_${TIMESTAMP}.log
- **This Report:** $REPORT_FILE

## Usage

To use the best performing optimizer, copy its configuration file and use it for training:

\`\`\`bash
# Example: If AdamW performed best
vanga train --symbol $SYMBOL --data $DATA_PATH --config configs/optimizer_examples/adamw_crypto_optimized.toml
\`\`\`

EOF

print_success "Benchmark completed!"
echo
print_status "Results saved to:"
print_status "  CSV: $RESULTS_FILE"
print_status "  Report: $REPORT_FILE"
print_status "  Logs: $OUTPUT_DIR/*_${TIMESTAMP}.log"
echo

# Show quick summary
print_status "Quick Summary:"
echo
successful_runs=$(grep -c "SUCCESS" "$RESULTS_FILE" || echo "0")
total_runs=$(tail -n +2 "$RESULTS_FILE" | wc -l)
print_status "Successful runs: $successful_runs/$total_runs"

if [[ $successful_runs -gt 0 ]]; then
    best_optimizer=$(tail -n +2 "$RESULTS_FILE" | grep "SUCCESS" | sort -t',' -k4 -n | head -1 | cut -d',' -f1)
    best_loss=$(tail -n +2 "$RESULTS_FILE" | grep "SUCCESS" | sort -t',' -k4 -n | head -1 | cut -d',' -f4)
    print_success "Best performer: $best_optimizer (val_loss: $best_loss)"
fi

print_status "View full report: cat $REPORT_FILE"
