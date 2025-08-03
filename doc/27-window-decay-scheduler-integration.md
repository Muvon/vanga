# Window Decay Integration with Learning Rate Schedulers

## Overview

The window decay system in VANGA has been enhanced to properly integrate with all learning rate schedulers. Previously, window decay only affected the base `learning_rate` parameter, but many schedulers have their own internal learning rate parameters (like `max_lr` in OneCycle, `base_lr` and `max_lr` in CyclicalLR, etc.) that were not being scaled.

## Problem Solved

**Before**: Window decay only scaled the base learning rate, but schedulers with internal LR parameters (OneCycle, CyclicalLR, etc.) used their original unscaled values, leading to inconsistent decay behavior.

**After**: Window decay is now applied to ALL relevant learning rate parameters within each scheduler, ensuring consistent decay behavior across all scheduler types.

## Architecture

### Core Components

1. **WindowAwareLearningRate**: Main class that handles window decay application
2. **create_window_aware_config()**: Factory function that creates window-specific configurations
3. **Scheduler-specific decay logic**: Each scheduler type has custom logic for which parameters to scale

### Integration Points

- **Training Loop**: Uses `create_window_aware_config()` instead of manual LR scaling
- **Scheduler Parameters**: All LR-related parameters are scaled by `window_decay^window_id`
- **Validation & Warnings**: Built-in validation for aggressive decay scenarios

## Scheduler-Specific Behavior

### OneCycle Scheduler
```toml
learning_schedule = { OneCycle = {
    max_lr = 0.01,           # ← Scaled by window decay
    pct_start = 0.3,         # ← Not scaled
    div_factor = 25.0        # ← Not scaled
}}
```
- **Scaled**: `max_lr`
- **Not Scaled**: `pct_start`, `div_factor`, `final_div_factor`, `anneal_strategy`

### CyclicalLR Scheduler
```toml
learning_schedule = { CyclicalLR = {
    base_lr = 1e-5,          # ← Scaled by window decay
    max_lr = 1e-3,           # ← Scaled by window decay
    step_size_up = 20        # ← Not scaled
}}
```
- **Scaled**: `base_lr`, `max_lr`
- **Not Scaled**: `step_size_up`, `step_size_down`, `mode`, `gamma`

### CosineAnnealing Scheduler
```toml
learning_schedule = { CosineAnnealing = {
    t_max = 100,             # ← Not scaled
    eta_min = 1e-6           # ← Scaled by window decay
}}
```
- **Scaled**: `eta_min` (minimum learning rate)
- **Not Scaled**: `t_max`
- **Also Scaled**: Base `learning_rate` from config

### Other Schedulers
- **Constant**: Only base `learning_rate` is scaled
- **ReduceOnPlateau**: Base `learning_rate` and `min_lr` are scaled
- **LinearDecay/ExponentialDecay/StepDecay/PolynomialDecay**: Base `learning_rate` and `min_lr` are scaled
- **WarmRestarts**: Base `learning_rate` and `eta_min` are scaled
- **NoamLR**: `factor` parameter is scaled

## Usage Example

### Configuration
```toml
[training]
learning_rate = 0.001
window_decay = 0.8  # 20% reduction per window

learning_schedule = { OneCycle = {
    max_lr = 0.01,
    pct_start = 0.3
}}
```

### Window Progression
- **Window 0**: `max_lr = 0.01` (100% of original)
- **Window 1**: `max_lr = 0.008` (80% of original)
- **Window 2**: `max_lr = 0.0064` (64% of original)
- **Window 3**: `max_lr = 0.00512` (51.2% of original)

## Implementation Details

### Window-Aware Config Creation
```rust
// Old approach (incorrect)
let mut window_config = self.config.clone();
window_config.training.learning_rate = window_lr;

// New approach (correct)
let window_config = create_window_aware_config(&self.config, window_id)?;
```

### Decay Factor Calculation
```rust
let decay_factor = window_decay.powi(window_id as i32);
let scaled_max_lr = original_max_lr * decay_factor;
```

### Validation & Warnings
The system includes built-in validation that warns about:
- Aggressive decay factors that may cause learning rates to become too small
- Window decay > 1.0 that increases learning rates over time
- Scheduler-specific issues (e.g., OneCycle max_lr becoming too small)

## Testing

Comprehensive integration tests verify:
- OneCycle scheduler parameter scaling
- CyclicalLR scheduler parameter scaling
- CosineAnnealing scheduler parameter scaling
- Constant scheduler behavior
- Validation warning generation
- Description string generation

## Benefits

1. **Consistent Decay**: All schedulers now properly respect window decay
2. **Scheduler Agnostic**: Works with any learning rate scheduler
3. **Validation**: Built-in warnings for problematic configurations
4. **Backward Compatible**: Existing configurations continue to work
5. **Comprehensive**: Handles all scheduler types with appropriate parameter scaling

## Migration Guide

### For Existing Configurations
No changes required - existing configurations will automatically use the new system.

### For Custom Schedulers
If you add new schedulers, update the `WindowAwareLearningRate::apply_window_decay()` method to handle the new scheduler's parameters.

## Logging Output

The system provides detailed logging:
```
🔄 Window 2/5: effective_lr=0.000640 (64.0% of base) → 1000 train samples, 200 validation samples
   📊 Window decay will be applied to ALL scheduler parameters (max_lr, base_lr, etc.)
🔄 Window 2 decay applied: OneCycle with max_lr scaled from 0.010000 to 0.006400 (64.0%)
```

## Performance Impact

Minimal performance impact:
- Configuration creation happens once per window
- No impact on training loop performance
- Validation runs only once per window

## Future Enhancements

Potential future improvements:
- Per-parameter decay factors
- Adaptive decay based on validation performance
- Scheduler-specific decay strategies
- Integration with hyperparameter optimization
