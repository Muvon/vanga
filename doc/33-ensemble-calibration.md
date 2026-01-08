# Ensemble Calibration System for VANGA LSTM

## 🎯 Overview

The Ensemble Calibration System is a comprehensive post-training calibration framework that improves model confidence estimates and prediction reliability through multiple complementary techniques.

### Key Components
- **Temperature Scaling**: Single shared temperature optimized via NLL minimization
- **Label Smoothing**: Adaptive smoothing based on per-class overconfidence
- **Mixup Augmentation**: ECE-based data augmentation during training (uniform application)
- **ECE Tracking**: Expected Calibration Error monitoring with 15-bin standard
- **Reliability Diagrams**: Visualization of calibration quality

### Benefits
- ✅ **Improved Confidence Estimates**: Better alignment between predicted probabilities and actual accuracy
- ✅ **Reduced Overconfidence**: Prevents model from being overly confident on uncertain predictions
- ✅ **Better Generalization**: Regularization through label smoothing and mixup
- ✅ **Gradient-Preserving**: Tensor-based operations maintain end-to-end differentiability
- ✅ **Stable Training**: Single shared temperature prevents class distribution distortion

## 🏗️ Architecture

### Calibration Pipeline

```
Training Completion
    ↓
Validation Set Predictions
    ↓
ECE Calculation (15-bin standard)
    ↓
Temperature Scaling (NLL minimization with binary search)
    ↓
Label Smoothing (per-class adaptive)
    ↓
Mixup Alpha Tuning (ECE-based, uniform application)
    ↓
Calibrated Model
```

### Module Structure

```
src/model/calibration/
├── ensemble.rs         # Orchestrates all calibration methods
├── temperature.rs      # Single shared temperature with NLL optimization
├── label_smoothing.rs  # Adaptive label smoothing
├── mixup.rs           # ECE-based mixup augmentation (uniform)
└── ece.rs             # Expected Calibration Error calculation
```

## ⚙️ Configuration

### Basic Configuration

```toml
[training.calibration]
enable_ensemble_calibration = true
ramp_up_epochs = 10                  # Gradual calibration application
```

### Advanced Configuration

```toml
[training.calibration]
enable_ensemble_calibration = true
ramp_up_epochs = 15

# Label smoothing
adaptive_label_smoothing = true
max_smoothing_epsilon = 0.2

# Mixup augmentation
enable_mixup = true
mixup_alpha_range = [0.1, 0.4]
```

## 🔬 Technical Details

### Temperature Scaling

Temperature scaling adjusts the softmax temperature to calibrate confidence:

```
Calibrated Probabilities = softmax(logits / T)
```

Where `T` is a **single shared temperature** parameter optimized via NLL minimization.

**Optimization Method:**
- Uses binary search on Negative Log-Likelihood (NLL) loss
- Single shared temperature for all classes (standard approach from calibration research)
- Search range: [0.5, 5.0] with precision 0.01

**Why NLL Instead of ECE?**
- NLL is a proper scoring rule that directly optimizes probability estimates
- For perfectly balanced 5-class datasets, NLL optimization is the mathematically correct approach
- ECE is used for monitoring, not optimization

### Label Smoothing

Adaptive label smoothing prevents overconfidence by smoothing target distributions:

```
Smoothed Target = (1 - ε) * one_hot + ε / num_classes
```

Where `ε` is determined per-class based on overconfidence:
- Higher ε when class confidence > accuracy (overconfident)
- Lower ε when class confidence ≈ accuracy (well-calibrated)

### Mixup Augmentation

Mixup creates virtual training examples by interpolating between samples:

```
Mixed Sample = λ * sample_i + (1 - λ) * sample_j
Mixed Target = λ * target_i + (1 - λ) * target_j
```

Where `λ ~ Beta(α, α)` and α is tuned based on ECE.

**Key Design Decision:**
- Mixup applies **uniformly to all samples** (not class-selective)
- This ensures consistent regularization across all classes
- Prevents training instability from inconsistent class treatment

### Expected Calibration Error (ECE)

ECE measures calibration quality by comparing confidence to accuracy:

```
ECE = Σ(b=1 to B) (n_b / n) * |acc(b) - conf(b)|
```

Where:
- B = 15 bins (research standard)
- n_b = samples in bin b
- acc(b) = accuracy in bin b
- conf(b) = average confidence in bin b

**Interpretation:**
- ECE < 0.05: Excellent calibration
- ECE 0.05-0.10: Good calibration
- ECE 0.10-0.15: Acceptable calibration
- ECE > 0.15: Poor calibration (needs improvement)

## 🚀 Usage

### Training with Ensemble Calibration

```bash
# Enable ensemble calibration in training
vanga train \
  --symbol BTCUSDT \
  --data data/btc_1h.csv \
  --config configs/training_with_calibration.toml
```

### Configuration Example

```toml
# configs/training_with_calibration.toml

[training]
epochs = 100
batch_size = 32
optimizer = { AdamW = { weight_decay = 0.01 } }

[training.calibration]
enable_ensemble_calibration = true
ramp_up_epochs = 10
adaptive_label_smoothing = true
enable_mixup = true
```

### Programmatic Usage

```rust
use vanga::model::calibration::EnsembleCalibrator;

// Create calibrator
let mut calibrator = EnsembleCalibrator::new();

// Calibrate from validation data
calibrator.calibrate_from_validation(
    &val_predictions,
    &val_targets,
)?;

// Apply to inference logits (uses single shared temperature)
let calibrated_logits = calibrator.apply_to_logits(&raw_logits)?;

// Apply to training targets
let smoothed_targets = calibrator.apply_label_smoothing(&targets)?;

// Apply mixup to training batch (uniform application)
let (mixed_sequences, mixed_targets) = calibrator.apply_mixup(
    &sequences,
    &targets,
    &mut rng_state,
)?;
```

## 📊 Monitoring and Evaluation

### Calibration Metrics

The system tracks several metrics:

```rust
pub struct CalibrationMetrics {
    pub overall_ece: f64,
    pub per_class_ece: [f64; 5],
    pub temperature: f64,
    pub label_smoothing_epsilons: [f64; 5],
    pub mixup_alpha: f64,
    pub mixup_enabled_classes: [bool; 5],
}
```

### Logging Output

During calibration, you'll see:

```
🎯 Starting ensemble calibration...
📊 Initial ECE: 0.1234
🌡️  Optimizing single shared temperature (NLL minimization)...
   Initial: NLL=1.2345, ECE=0.1234
   T=1.4500: NLL 1.2345 → 1.0890 (11.8% reduction), ECE=0.0987
✅ Optimal temperature: 1.4500 (NLL: 1.0890, ECE: 0.0987)
🏷️  Calibrating label smoothing...
   Optimal ε = 0.10 (adaptive per class)
🔀 Tuning mixup alpha...
   Optimal α = 0.25
✅ Ensemble calibration complete!
    Final ECE: 0.0890 (Good)
    Final NLL: 1.0890 (11.8% improvement)
```

## 🔧 Troubleshooting

### High ECE After Calibration

**Problem**: ECE remains high (>0.10) after calibration

**Solutions:**
1. Increase `calibration_iterations` (try 100-200)
2. Widen `temperature_range` (try [0.1, 10.0])
3. Increase `exploration_factor` (try 0.2-0.3)
4. Check validation set size (need 1000+ samples)
5. Try different `bayesian_method` (TuRBO-2 vs BORE)

### Calibration Instability

**Problem**: Calibration metrics fluctuate during training

**Solutions:**
1. Increase `ramp_up_epochs` for gradual application
2. Use larger validation set for stable ECE estimates
3. Enable early stopping with `early_stopping_patience`
4. Reduce `exploration_factor` for more exploitation

### Overfitting with Mixup

**Problem**: Training loss decreases but validation loss increases

**Solutions:**
1. Reduce `mixup_alpha_range` upper bound
2. Disable mixup for well-calibrated classes
3. Increase regularization (weight decay, dropout)
4. Use more training data

## 🎯 Best Practices

### When to Use Ensemble Calibration

✅ **Use when:**
- Model shows overconfidence (high confidence on wrong predictions)
- ECE > 0.10 on validation set
- Trading decisions require accurate confidence estimates
- Working with imbalanced datasets

❌ **Skip when:**
- Model already well-calibrated (ECE < 0.05)
- Very small validation set (<500 samples)
- Training time is critical constraint
- Simple baseline model for comparison

### Calibration vs Bias Correction

**Ensemble Calibration:**
- Post-training confidence adjustment
- Improves probability estimates
- Uses Bayesian optimization
- More computationally expensive
- Better for final production models

**Linear Bias Correction:**
- Simple linear adjustment
- Faster to compute
- Good for quick iterations
- Suitable for development/testing

**Recommendation:** Use linear bias correction during development, switch to ensemble calibration for production models.

### Configuration Guidelines

**Conservative (Stable):**
```toml
calibration_iterations = 30
exploration_factor = 0.05
ramp_up_epochs = 15
```

**Balanced (Recommended):**
```toml
calibration_iterations = 50
exploration_factor = 0.1
ramp_up_epochs = 10
```

**Aggressive (Maximum Quality):**
```toml
calibration_iterations = 100
exploration_factor = 0.15
ramp_up_epochs = 5
```

## 📚 References

- **Temperature Scaling**: Guo et al., "On Calibration of Modern Neural Networks" (ICML 2017)
- **Label Smoothing**: Müller et al., "When Does Label Smoothing Help?" (NeurIPS 2019)
- **Mixup**: Zhang et al., "mixup: Beyond Empirical Risk Minimization" (ICLR 2018)
- **ECE**: Naeini et al., "Obtaining Well Calibrated Probabilities Using Bayesian Binning" (AAAI 2015)
- **TuRBO**: Eriksson et al., "Scalable Global Optimization via Local Bayesian Optimization" (NeurIPS 2019)
- **BORE**: Tiao et al., "BORE: Bayesian Optimization by Density-Ratio Estimation" (ICML 2021)

## 🔗 Related Documentation

- [Training Guide](04-training.md) - Main training pipeline
- [Configuration Guide](20-configuration.md) - Complete configuration reference
- [Optimizer Selection](22-optimizer-selection-guide.md) - Optimizer choices
- [Technical Implementation](10-technical-implementation.md) - Implementation details

---

**Last Updated**: January 2026
**Module**: `src/model/calibration/`
**Status**: Production-ready
