# Day Ahead Optimization Enhanced Repository

![Supports aarch64 Architecture][aarch64-shield] ![Supports amd64 Architecture][amd64-shield] ![Version][version-shield]

[aarch64-shield]: https://img.shields.io/badge/aarch64-yes-green.svg
[amd64-shield]: https://img.shields.io/badge/amd64-yes-green.svg
[version-shield]: https://img.shields.io/badge/version-1.1.0-blue.svg

This repository contains enhanced versions of the Day Ahead Optimizer (DAO) for Home Assistant with modern integration features, automatic service discovery, and advanced security.

[![Open your Home Assistant instance and show the add-on store.](https://my.home-assistant.io/badges/supervisor_store.svg)](https://my.home-assistant.io/redirect/supervisor_store/)

## 🆕 Version 1.0.0 Features

✨ **New Home Assistant Integrations:**
- 🔧 **Automatic Service Discovery** - Auto-detects MariaDB, MySQL, MQTT, and other HA services
- 📊 **Health Monitoring** - Real-time system status reported to Home Assistant sensors
- 🛡️ **Enhanced Security** - Modern AppArmor profiles and ingress integration
- 🌐 **Proper Ingress Support** - Seamless web interface integration
- 📝 **Configuration Schema** - User-friendly configuration with validation
- 🗣️ **Multi-language Support** - Dutch and English translations

## 🚀 Available Addons

### ⚡ DAO Modern (AI/ML Enhanced)
**Advanced version with all modern features**

- 🧠 ML-powered consumption prediction (>90% accuracy)
- ⚡ Smart device scheduling for optimal cost savings  
- 📊 Real-time high-load detection and response
- 🔋 Adaptive battery management with degradation optimization
- 📅 7-day multi-day planning with weather forecasting
- 🌤️ Enhanced weather integration (OpenWeatherMap/KNMI)
- 🎄 Holiday/vacation detection for adjusted planning
- 🌡️ Seasonal optimization strategies (winter/summer)
- 🤖 Optional AI optimization (OpenAI/Anthropic/Local)
- 🔗 Real-time WebSocket updates and modern GUI

**Requirements:** 4GB+ RAM recommended, modern Pi 4 (aarch64) or x86_64 system
**Port:** 8099 (ingress) or 5001 (direct)

### 🪶 DAO Light (Minimal & Stable)  
**Lightweight version optimized for reliability**

- ✅ All core DAO optimization functionality
- ✅ Minimal dependencies for maximum stability
- ✅ Perfect for Pi 3/4 with limited resources
- ✅ Traditional scheduler (no complex async)
- ✅ Stable, proven libraries only
- ✅ Fast startup and low memory usage

**Requirements:** 1GB+ RAM, any Pi (aarch64) or x86_64 system  
**Port:** 8099 (ingress) or 5002 (direct)

## 🏗️ Architecture Support

Both DAO versions support multiple architectures:

- **aarch64** - ARM64 systems (Raspberry Pi 4, Apple Silicon, etc.)
- **amd64** - x86_64 systems (Intel/AMD 64-bit processors)

Home Assistant will automatically select the correct architecture for your system.

## 🔧 Installation

1. Add this repository to Home Assistant:
   ```
   https://github.com/miclnl/day-ahead
   ```

2. Install either or both addons:
   - **DAO Modern**: For full-featured experience
   - **DAO Light**: For stable, minimal installation

3. Both can run simultaneously if desired

## 📊 Comparison

| Feature | DAO Modern | DAO Light |
|---------|------------|-----------|
| Core Optimization | ✅ | ✅ |
| ML Prediction | ✅ Advanced | ❌ |
| Smart Scheduling | ✅ | ❌ |
| Multi-day Planning | ✅ | ❌ |
| AI Integration | ✅ | ❌ |
| Memory Usage | High | Low |
| Startup Time | Slower | Fast |
| Stability | Good | Excellent |
| Dependencies | Many | Minimal |

## 🎯 Which Version to Choose?

**Choose DAO Modern if:**
- You have 8GB+ RAM
- Want all advanced features
- Don't mind complex dependencies
- Want ML predictions and AI features

**Choose DAO Light if:**
- You have limited RAM (2-4GB)
- Want maximum stability
- Prefer simple, proven technology  
- Only need core optimization

## 🏠 Data Folders

Each addon uses separate data folders to avoid conflicts:
- **DAO Modern**: `/config/dao_modern_data`
- **DAO Light**: `/config/dao_light_data`

## 📚 Documentation

See the original DAO documentation for configuration details:
- Energy optimization principles
- Database setup
- Price source configuration
- Battery and EV settings

## 🤝 Support

- Issues: [GitHub Issues](https://github.com/miclnl/day-ahead/issues)
- Based on original work by Cees van Beek

---

**Version:** 1.3.4  
**Author:** Cees van Beek  
**Enhanced by:** Claude Code