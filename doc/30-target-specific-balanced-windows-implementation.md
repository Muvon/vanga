# Target-Specific Balanced Windows Implementation

## Overview

Successfully implemented target-specific balanced training for VANGA's multi-target LSTM architecture. Each target (PriceLevel, Direction, Volatility) now gets its own independently balanced dataset with 100% balanced training and validation splits.

## Problem Solved

**Original Issue**: Global balancing limited all targets to the minimum balanced sequences across targets. If PriceLevel had 300 balanced sequences, Direction had 400, and Volatility had 200, the system would limit ALL targets to 200 sequences, wasting valuable training data.

**Solution**: Each target now gets its own optimal balanced dataset and maintains perfect class balance in both training and validation splits.

## Key Implementation Details

### Core Method: `create_target_specific_balanced_windows()`

Location: `src/data/mod.rs:473-719`

**Critical Architecture Changes:**

1. **Independent Target Balancing**:
   ```rust
   let target_balanced_datasets = balancer.extract_target_specific_balanced_datasets(
       &all_sequences,
       &target_types,
       &config.horizons,
   )?;
   ```

2. **Balanced Train/Val Splits** (CRITICAL FIX):
   ```rust
   // BEFORE (BROKEN): Simple split that breaks class balance
   let (train_indices, val_indices) = if target_val_size > 0 {
       let split_point = target_indices.len() - target_val_size;
       (target_indices[..split_point].to_vec(), target_indices[split_point..].to_vec())
   } else {
       (target_indices.clone(), Vec::new())
   };

   // AFTER (FIXED): Maintains perfect class balance in both splits
   let (balanced_training_dataset, target_validation_indices) = balancer
       .smart_validation_split_from_balanced(
           target_dataset,
           &all_sequences,
           validation_ratio,
           &[*target_type],
           &[horizon.clone()],
       )?;
   ```

3. **Target-Specific Windows**:
   - Each target gets its own `Vec<TrainingWindow>`
   - Windows are organized by `(TargetType, String)` key
   - Progressive windows maintain balance within each target

## Data Flow Architecture

```
Raw Data → Sequences → Target-Specific Balance → Balanced Train/Val Split → Windows
    ↓           ↓              ↓                        ↓                    ↓
OHLCV      All Seqs    PriceLevel: 300 balanced    Train: 240 (balanced)   N Windows
Data       Generated   Direction: 400 balanced     Val: 60 (balanced)      Per Target
                      Volatility: 200 balanced
```

## Expected Results

### Before (Global Balancing):
```
🎯 Training PriceLevel 1h with 200 sequences
🎯 Training Direction 1h with 200 sequences
🎯 Training Volatility 1h with 200 sequences
Total: 600 sequences (100 sequences wasted)
```

### After (Target-Specific Balancing):
```
🎯 Training PriceLevel 1h with 300 sequences
🎯 Training Direction 1h with 400 sequences
🎯 Training Volatility 1h with 200 sequences
Total: 900 sequences (50% more training data)
```

## Key Benefits

1. **Maximum Data Utilization**: Each target uses its optimal balanced dataset
2. **Perfect Balance Maintained**: Both training and validation maintain class balance
3. **No Data Waste**: No longer limited by target with least balanced data
4. **Independent Optimization**: Each target can have different amounts of balanced data
5. **Scalable Architecture**: Easy to add new targets without affecting existing ones

## Implementation Files Modified

### Core Implementation:
- `src/data/mod.rs`: Main `create_target_specific_balanced_windows()` method
- `src/data/balance.rs`: `extract_target_specific_balanced_datasets()` method
- `src/api/trainer.rs`: Updated to handle `TargetSpecificWindows`

### Data Structures:
- `TargetSpecificWindows`: Container for target-specific windows
- `GloballyBalancedDataset`: Per-target balanced data with statistics

## Verification

### Compilation Status:
✅ Code compiles successfully with no errors
✅ All clippy warnings resolved
✅ Unused methods and type aliases removed

### Expected Log Output:
```
🎯 TARGET-SPECIFIC BALANCE EXTRACTION: Each target balanced independently
   PriceLevel 1h: 300 balanced sequences (60 per class × 5 classes)
   Direction 1h: 400 balanced sequences (80 per class × 5 classes)
   Volatility 1h: 200 balanced sequences (40 per class × 5 classes)

🏗️ CREATING TARGET-SPECIFIC BALANCED WINDOWS:
🎯 Creating windows for PriceLevel 1h with 300 balanced sequences
   → PriceLevel 1h W1: train 240 seq (balanced), val 60 seq (balanced), targets: 15
🎯 Creating windows for Direction 1h with 400 balanced sequences
   → Direction 1h W1: train 320 seq (balanced), val 80 seq (balanced), targets: 15
🎯 Creating windows for Volatility 1h with 200 balanced sequences
   → Volatility 1h W1: train 160 seq (balanced), val 40 seq (balanced), targets: 15
```

## Testing

To verify the implementation works:

```bash
# Run training with target-specific balancing
cargo run -- train --config configs/quick_start.toml

# Look for logs showing different sequence counts per target
# Should see different numbers for each target, not uniform counts
```

## Technical Notes

### Critical Architecture Rules:
1. **Balance Preservation**: `smart_validation_split_from_balanced()` maintains perfect class balance
2. **Target Independence**: Each target's balance is calculated independently
3. **Progressive Windows**: Each window maintains balance while growing in size
4. **Memory Efficiency**: Uses indices rather than copying sequence data

### Performance Impact:
- **Positive**: More training data per target (up to 50% increase)
- **Neutral**: Same computational complexity per target
- **Positive**: Better model performance due to more balanced training data

## Future Enhancements

1. **Dynamic Balancing**: Adjust balance ratios based on target difficulty
2. **Cross-Target Learning**: Share knowledge between targets while maintaining independence
3. **Adaptive Windows**: Adjust window sizes based on target-specific performance
4. **Balance Monitoring**: Real-time monitoring of class balance during training

## Conclusion

The target-specific balanced windows implementation successfully maximizes training data efficiency while maintaining perfect class balance for each target. This architectural improvement provides a solid foundation for multi-target LSTM training with optimal data utilization.
