# Learning Rate Schedules in VANGA LSTM

## Overview

VANGA LSTM provides a comprehensive suite of 11 learning rate schedules optimized for cryptocurrency time series forecasting. Each schedule is mathematically validated and includes LSTM-specific optimizations for better convergence and stability.

## Available Schedules

### 1. Constant Schedule
**Formula**: `lr(t) = lr_initial`

**Use Case**: Baseline training, fine-tuning pre-trained models

**Configuration**:
```toml
learning_schedule = "Constant"
```

**Pros**: Simple, predictable, good for fine-tuning
**Cons**: No adaptation to training progress

---

### 2. Reduce on Plateau
**Formula**: `lr(t+1) = lr(t) × factor` when validation loss plateaus

**Use Case**: Adaptive training with validation monitoring

**Configuration**:
```toml
learning_schedule = { ReduceOnPlateau = {
    patience = 10,
    factor = 0.5,
    min_lr = 1e-6,
    monitor = "loss",
    threshold = 0.01
}}
```

**Parameters**:
- `patience`: Epochs to wait before reducing LR
- `factor`: Multiplication factor (0 < factor < 1)
- `min_lr`: Minimum learning rate threshold
- `monitor`: Metric to monitor ("loss", "accuracy", "f1_score")
- `threshold`: Minimum improvement threshold

**Pros**: Adaptive, prevents overfitting, validation-aware
**Cons**: Requires validation data, can be slow to adapt

---

### 3. Linear Decay
**Formula**: `lr(t) = lr_initial × (1 - decay_rate × progress)`

Where `progress = epoch / total_epochs`

**Use Case**: Gradual learning rate reduction over training

**Configuration**:
```toml
learning_schedule = { LinearDecay = {
    decay_rate = 0.01,
    min_lr = 1e-6
}}
```

**Parameters**:
- `decay_rate`: Rate of linear decay (0 ≤ decay_rate ≤ 1)
- `min_lr`: Minimum learning rate threshold

**Pros**: Smooth decay, predictable, good for long training
**Cons**: May decay too slowly or quickly depending on total epochs

---

### 4. Exponential Decay
**Formula**: `lr(t) = lr_initial × γ^epoch`

**Use Case**: Fast initial learning with rapid decay

**Configuration**:
```toml
learning_schedule = { ExponentialDecay = {
    gamma = 0.95,
    min_lr = 1e-6
}}
```

**Parameters**:
- `gamma`: Decay factor per epoch (0 < gamma ≤ 1)
- `min_lr`: Minimum learning rate threshold

**Pros**: Fast convergence, mathematically elegant
**Cons**: Can decay too aggressively, may need careful tuning

**LSTM Recommendation**: Use gamma > 0.9 for stable LSTM training

---

### 5. Step Decay
**Formula**: `lr(t) = lr_initial × γ^floor(epoch/step_size)` or milestone-based

**Use Case**: Discrete learning rate reductions at specific points

**Configuration**:
```toml
# Regular step decay
learning_schedule = { StepDecay = {
    step_size = 25,
    gamma = 0.5,
    min_lr = 1e-6
}}

# Milestone-based decay
learning_schedule = { StepDecay = {
    step_size = 10,  # Ignored when milestones provided
    gamma = 0.1,
    milestones = [50, 100, 150],
    min_lr = 1e-6
}}
```

**Parameters**:
- `step_size`: Epochs between decay steps (ignored if milestones provided)
- `gamma`: Decay factor at each step (0 < gamma ≤ 1)
- `milestones`: Specific epochs for decay (optional, must be ascending)
- `min_lr`: Minimum learning rate threshold

**Pros**: Precise control, good for curriculum learning
**Cons**: Requires domain knowledge for milestone selection

---

### 6. Polynomial Decay
**Formula**: `lr(t) = min_lr + (lr_initial - min_lr) × (1 - progress)^power`

**Use Case**: Smooth non-linear decay with configurable curve shape

**Configuration**:
```toml
learning_schedule = { PolynomialDecay = {
    power = 2.0,
    min_lr = 1e-6
}}
```

**Parameters**:
- `power`: Polynomial power (> 0, higher = steeper decay)
- `min_lr`: Minimum learning rate threshold

**Pros**: Flexible decay curve, smooth transition to min_lr
**Cons**: Requires tuning of power parameter

**LSTM Recommendation**: Use power = 1.0-3.0 for balanced decay

---

### 7. Cosine Annealing
**Formula**: `lr(t) = η_min + (lr_initial - η_min) × 0.5 × (1 + cos(π × epoch / t_max))`

**Use Case**: Smooth periodic decay, good for finding global minima

**Configuration**:
```toml
learning_schedule = { CosineAnnealing = {
    t_max = 100,
    eta_min = 1e-6
}}
```

**Parameters**:
- `t_max`: Period of cosine cycle (should ≤ total_epochs)
- `eta_min`: Minimum learning rate at cycle end

**Pros**: Smooth decay, helps escape local minima, mathematically elegant
**Cons**: Requires careful t_max selection

**LSTM Recommendation**: Set t_max = total_epochs for single cycle

---

### 8. Warm Restarts (SGDR)
**Formula**: Cosine annealing with periodic restarts

**Use Case**: Escaping local minima, exploring loss landscape

**Configuration**:
```toml
learning_schedule = { WarmRestarts = {
    t_0 = 10,
    t_mult = 2,
    eta_min = 1e-6
}}
```

**Parameters**:
- `t_0`: Length of first restart cycle
- `t_mult`: Factor for increasing cycle length (≥ 1)
- `eta_min`: Minimum learning rate in each cycle

**Cycle Lengths**: t_0, t_0×t_mult, t_0×t_mult², ...

**Pros**: Excellent for escaping local minima, robust convergence
**Cons**: Can be unstable, requires careful hyperparameter tuning

**LSTM Recommendation**: Use t_mult = 1 or 2 for stability

---

### 9. One Cycle Learning Rate ⭐ **LSTM Optimized**
**Formula**: Triangular cycle with warm-up and cool-down phases

**Use Case**: Super-convergence for LSTM training, fastest convergence

**Configuration**:
```toml
learning_schedule = { OneCycle = {
    max_lr = 0.01,
    pct_start = 0.3,
    anneal_strategy = "cos",
    div_factor = 25.0,
    final_div_factor = 1e4
}}
```

**Parameters**:
- `max_lr`: Peak learning rate
- `pct_start`: Percentage of cycle for LR increase (0 < pct_start < 1)
- `anneal_strategy`: "cos" or "linear" for decay phase
- `div_factor`: initial_lr = max_lr / div_factor
- `final_div_factor`: final_lr = initial_lr / final_div_factor

**Phases**:
1. **Warm-up** (0 to pct_start): initial_lr → max_lr
2. **Cool-down** (pct_start to 1.0): max_lr → final_lr

**Pros**: Fastest convergence, excellent for LSTM, proven super-convergence
**Cons**: Requires careful max_lr tuning, can be unstable if max_lr too high

**LSTM Recommendation**: Start with max_lr = 10×initial_lr, adjust based on results

---

### 10. Cyclical Learning Rate
**Formula**: Triangular waves between base_lr and max_lr

**Use Case**: Exploring loss landscape, avoiding local minima

**Configuration**:
```toml
learning_schedule = { CyclicalLR = {
    base_lr = 1e-5,
    max_lr = 1e-3,
    step_size_up = 20,
    step_size_down = 20,
    mode = "triangular",
    gamma = 1.0
}}
```

**Parameters**:
- `base_lr`: Minimum learning rate in cycle
- `max_lr`: Maximum learning rate in cycle
- `step_size_up`: Epochs for LR increase
- `step_size_down`: Epochs for LR decrease (optional, defaults to step_size_up)
- `mode`: "triangular", "triangular2", "exp_range"
- `gamma`: Decay factor for "exp_range" mode

**Modes**:
- **triangular**: Constant amplitude
- **triangular2**: Amplitude halves each cycle
- **exp_range**: Exponential amplitude decay

**Pros**: Good exploration, helps escape local minima
**Cons**: Can be unstable, requires careful amplitude tuning

**LSTM Recommendation**: Use max_lr/base_lr ratio < 10 for stability

---

### 11. Noam Scheduler (Transformer-style)
**Formula**: `lr(t) = factor × model_size^(-0.5) × min(step^(-0.5), step × warmup_steps^(-1.5))`

**Use Case**: Sequence models, attention mechanisms, transformer-style training

**Configuration**:
```toml
learning_schedule = { NoamLR = {
    model_size = 512,
    warmup_steps = 100,
    factor = 1.0
}}
```

**Parameters**:
- `model_size`: Model dimension (affects scaling)
- `warmup_steps`: Linear warmup duration
- `factor`: Additional scaling factor

**Pros**: Proven for sequence models, automatic warmup and decay
**Cons**: May not be optimal for pure LSTM without attention

**LSTM Recommendation**: Use with attention-enhanced LSTM models

---

## LSTM-Specific Recommendations

### For Cryptocurrency Trading:

1. **Best Overall**: OneCycle with max_lr = 0.01, pct_start = 0.3
2. **Most Stable**: CosineAnnealing with t_max = total_epochs
3. **Fastest**: OneCycle with careful max_lr tuning
4. **Most Robust**: WarmRestarts with t_0 = 20, t_mult = 1

### Parameter Guidelines:

- **Initial LR**: 0.001 for most crypto datasets
- **Min LR**: 1e-6 to prevent complete stagnation
- **Decay Factors**: Use gamma > 0.9 for stable LSTM training
- **Cycle Lengths**: 20-50 epochs for crypto volatility patterns

### Validation Integration:

```toml
# Combine with validation for best results
validation_split = 0.2
early_stopping = { patience = 20, min_delta = 0.001 }
learning_schedule = { OneCycle = { max_lr = 0.01, pct_start = 0.3 }}
```

## Mathematical Properties

### Monotonicity:
- **Monotonic Decreasing**: LinearDecay, ExponentialDecay, PolynomialDecay
- **Non-Monotonic**: CosineAnnealing, WarmRestarts, OneCycle, CyclicalLR

### Convergence Guarantees:
- **Guaranteed Convergence**: Constant, LinearDecay, ExponentialDecay
- **Conditional Convergence**: All others (depends on parameters)

### Memory Efficiency:
- **Most Efficient**: Constant, LinearDecay, ExponentialDecay
- **Least Efficient**: WarmRestarts, CyclicalLR (due to cycle calculations)

## Performance Benchmarks

Based on VANGA's internal benchmarks on cryptocurrency data:

| Schedule | Convergence Speed | Stability | Final Loss | Memory Usage |
|----------|------------------|-----------|------------|--------------|
| OneCycle | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| CosineAnnealing | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| WarmRestarts | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| ExponentialDecay | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| ReduceOnPlateau | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |

## Configuration Examples

### Conservative (Stable Training):
```toml
learning_schedule = { CosineAnnealing = { t_max = 200, eta_min = 1e-6 }}
```

### Aggressive (Fast Convergence):
```toml
learning_schedule = { OneCycle = {
    max_lr = 0.02,
    pct_start = 0.25,
    anneal_strategy = "cos"
}}
```

### Exploration (Escape Local Minima):
```toml
learning_schedule = { WarmRestarts = {
    t_0 = 25,
    t_mult = 2,
    eta_min = 1e-6
}}
```

### Production (Reliable Results):
```toml
learning_schedule = { ReduceOnPlateau = {
    patience = 15,
    factor = 0.5,
    min_lr = 1e-6,
    monitor = "loss",
    threshold = 0.001
}}
```

## Troubleshooting

### Common Issues:

1. **Training Instability**: Reduce max_lr, increase patience, use more conservative schedules
2. **Slow Convergence**: Try OneCycle or reduce decay rates
3. **Overfitting**: Use ReduceOnPlateau with validation monitoring
4. **Loss Plateaus**: Try WarmRestarts or CyclicalLR for exploration

### Warning Messages:

The system provides automatic warnings for LSTM-unsuitable configurations:
- OneCycle max_lr too high compared to initial_lr
- CyclicalLR ratio too aggressive for LSTM stability
- ExponentialDecay gamma too aggressive
- CosineAnnealing t_max exceeding total epochs

## Integration with VANGA Training

All schedules integrate seamlessly with VANGA's training pipeline:

```rust
// Automatic validation and LSTM suitability checking
let config = TrainingConfig {
    learning_schedule: Some(LearningScheduleConfig::OneCycle {
        max_lr: 0.01,
        pct_start: Some(0.3),
        anneal_strategy: Some("cos".to_string()),
        div_factor: Some(25.0),
        final_div_factor: Some(1e4),
    }),
    // ... other config
};

// Training with schedule
model.train(&sequences, &targets, &config).await?;
```

The system automatically:
- Validates all parameters
- Provides LSTM-specific warnings
- Integrates with warmup epochs
- Combines with early stopping
- Logs learning rate changes during training

## References

1. Smith, L. N. (2017). "Cyclical Learning Rates for Training Neural Networks"
2. Loshchilov, I., & Hutter, F. (2016). "SGDR: Stochastic Gradient Descent with Warm Restarts"
3. Smith, L. N., & Topin, N. (2019). "Super-convergence: Very fast training of neural networks using large learning rates"
4. Vaswani, A., et al. (2017). "Attention is All You Need" (Noam Scheduler)
5. VANGA Internal Benchmarks on Cryptocurrency Time Series (2024)
