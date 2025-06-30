# VANGA - LSTM Cryptocurrency Forecasting System

## 📚 **Complete Documentation Index**

### **🚀 Getting Started**
1. **[Introduction](01-introduction.md)** - System overview, features, and concepts
2. **[Installation](02-installation.md)** - Setup, dependencies, and build instructions
3. **[Data Preparation](03-data-preparation.md)** - Data format, preprocessing, and validation
4. **[Training](04-training.md)** - Model training, configuration, and optimization
5. **[Predictions](05-predictions.md)** - Making predictions, output formats, and evaluation

### **🔧 Technical Reference**
6. **[Technical Indicators](06-technical-indicators.md)** - 50+ indicators implementation and usage
7. **[System Architecture](07-architecture.md)** - Complete system architecture and design
8. **[Multi-Target System](08-targets.md)** - Prediction targets and configuration
9. **[Evaluation](09-evaluation.md)** - Model performance evaluation and metrics

### **✅ Final Implementation**
10. **[Technical Implementation Guide](10-technical-implementation.md)** - Complete technical specifications
11. **[Usage Examples](11-usage-examples.md)** - Comprehensive usage guide with real-world examples
12. **[Quick Start Guide](12-quick-start.md)** - Fast-track setup with intelligent training
13. **[Complete Usage Guide](13-usage-guide.md)** - Detailed training and custom features guide

### **🧠 Intelligent Training System**
- **[Auto-Optimization](05-auto-optimization.md)** - Intelligent training features and configuration
- **[Training Guide](04-training.md)** - Updated with early stopping and adaptive learning rate

## 🎯 **Quick Navigation**

### **For New Users**
Start with: [Introduction](01-introduction.md) → [Installation](02-installation.md) → [Usage Examples](11-usage-examples.md)

### **For Developers**
Read: [Technical Implementation Guide](10-technical-implementation.md)

### **For Production Use**
Check: [Usage Examples](11-usage-examples.md) → [Technical Indicators](06-technical-indicators.md)

## 📊 **System Capabilities**

### **Core Features**
- ✅ **LSTM Neural Networks**: Complete Candle framework integration
- ✅ **50+ Technical Indicators**: Professional-grade technical analysis
- ✅ **Multi-Target Prediction**: Price levels, direction, volatility
- ✅ **CLI Interface**: Complete train/predict/manage workflow
- ✅ **Model Persistence**: Save/load functionality with error handling
- ✅ **Configuration System**: Flexible TOML-based configuration

### **Performance Specifications**
- **Technical Indicators**: ~3ms for all 50+ indicators per 1000 data points
- **Memory Usage**: <10MB for 100k data points with full indicator suite
- **Build Status**: Zero compilation errors, optimized release build
- **CLI Commands**: All commands functional with comprehensive help text

## 🏗️ **Architecture Overview**

```
CSV Data → Polars DataFrame → Technical Indicators (50+) → Feature Matrix →
LSTM Sequences → Multi-Target Prediction → CSV Output
```

### **Key Components**
- **Data Pipeline**: High-performance Polars-based processing
- **Feature Engineering**: Comprehensive technical analysis suite
- **LSTM Model**: Candle framework integration with persistence
- **API Layer**: High-level training and prediction functions
- **CLI Interface**: Complete command-line interface
- **Configuration**: TOML-based parameter management
