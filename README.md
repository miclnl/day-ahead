# Day Ahead Optimization Enhanced Repository

![Supports aarch64 Architecture][aarch64-shield] ![Supports amd64 Architecture][amd64-shield] ![Version][version-shield]

[aarch64-shield]: https://img.shields.io/badge/aarch64-yes-green.svg
[amd64-shield]: https://img.shields.io/badge/amd64-yes-green.svg
[version-shield]: https://img.shields.io/badge/version-1.1.1-blue.svg

This repository contains enhanced versions of the Day Ahead Optimizer (DAO) for Home Assistant with modern integration features, automatic service discovery, and advanced security.

[![Open your Home Assistant instance and show the add-on store.](https://my.home-assistant.io/badges/supervisor_store.svg)](https://my.home-assistant.io/redirect/supervisor_store/)

## 🏠 About This Project

This project started as a personal energy optimization solution for my own Home Assistant setup, driven by the need for intelligent battery management and cost-effective energy usage. What began as a custom solution for optimizing solar panels, home batteries, and dynamic electricity pricing has evolved into a comprehensive energy management system.

**Built on Excellent Foundations:**
I want to emphasize that this enhanced version is built upon an already excellent and well-functioning original DAO project. I was extremely satisfied with the original implementation - it worked beautifully for energy optimization and saved significant costs on my electricity bills. I claim no credit for the brilliant core algorithms, mathematical optimization, or the fundamental architecture that made this possible.

This enhanced repository represents extensions and modernizations developed with the assistance of AI coding tools (Claude Code) to add Home Assistant integration features, security improvements, and deployment optimizations. The core energy optimization intelligence remains the work of the original developers.

**Personal Origins, Open Source Spirit:**
While this repository is primarily developed and maintained for my own use case, I believe in the power of open source collaboration. Everyone is welcome to use this code, adapt it to their needs, learn from it, or contribute improvements back to the community.

**Why Share This?**
- **Transparency**: Energy optimization should be accessible and understandable
- **Community Learning**: Others might benefit from the approaches and solutions developed here
- **Collaborative Improvement**: Different use cases and perspectives make the software better
- **Knowledge Sharing**: The intersection of Home Assistant, energy management, and machine learning offers valuable learning opportunities

**Use at Your Own Risk:**
This is a personal project that works well for my specific setup (Dutch energy market with dynamic pricing, solar panels, home battery, electric vehicle charging). While I've made efforts to make it configurable and robust, you may need to adapt certain aspects for your environment:

- **Energy Markets**: Primarily tested with Dutch/European energy providers (Nord Pool, ENTSO-E)
- **Hardware**: Optimized for common EU solar/battery setups
- **Regulations**: Tax calculations and grid limitations based on Dutch regulations
- **Language**: Interface supports Dutch and English

I welcome feedback, bug reports, and contributions, but keep in mind that my time for extensive support is limited.

**Contributing:**
If you find bugs, have improvements, or want to adapt the code for other energy markets or hardware configurations, pull requests are very welcome! Even if it's just documentation improvements or configuration examples for different setups.

Feel free to fork, modify, and make it your own - that's the beauty of open source! 🚀

## 🆕 Version 1.1.0 Features

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