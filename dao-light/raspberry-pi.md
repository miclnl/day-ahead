# Raspberry Pi 4 Optimization Guide

## Hardware Requirements
- Raspberry Pi 4 CM with 8GB RAM (optimal performance)
- Fast microSD card (Class 10/A2) or SSD via USB (highly recommended)
- Active cooling recommended for sustained ML operations

## 8GB RAM Optimizations
With 8GB RAM, your Pi can run most ML features at near-desktop performance:

### ML Libraries (8GB Optimized)
- Full scikit-learn with all algorithms
- TensorFlow Lite for neural networks (lighter than full TensorFlow)
- XGBoost with full feature set
- LightGBM for fast gradient boosting
- Joblib for parallel processing

### Performance Settings
Add these to your Home Assistant `configuration.yaml`:

```yaml
# Optimize for Raspberry Pi
http:
  use_x_forwarded_for: true
  trusted_proxies:
    - 127.0.0.1

# Reduce memory usage
recorder:
  auto_purge: true
  purge_keep_days: 7
```

### DAO Settings for 8GB Pi
In your DAO `options.json`, add:

```json
{
  "performance": {
    "max_workers": 4,
    "memory_limit_mb": 3072,
    "use_lightweight_ml": false,
    "cache_size_mb": 512
  },
  "smart_optimization": {
    "advanced_prediction": {
      "enabled": true,
      "model_type": "full",
      "prediction_horizon": 60
    },
    "weather_forecast_days": 6
  }
}
```

## Performance Expectations (8GB Pi)
- **Multi-day optimization**: 2-5 minutes (vs 30s desktop)
- **ML training**: 30-60 seconds (vs 10s desktop)  
- **Real-time prediction**: <1 second
- **Memory usage**: ~2-3GB under full load

## Monitoring
Watch memory usage: `htop` or HA System Monitor integration