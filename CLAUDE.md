# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Day Ahead Optimizer (DAO) is a Home Assistant add-on that optimizes energy consumption, production, and battery storage using dynamic electricity pricing. The system uses Mixed-Integer Linear Programming (MIP) to calculate optimal usage of batteries, electric vehicles, boilers, and heat pumps based on energy prices, weather forecasts, and consumption patterns.

## Core Architecture

### Main Components
- **da_scheduler.py**: Main scheduler that runs optimization tasks at configured intervals
- **day_ahead.py**: Core optimization engine using MIP algorithms for energy planning
- **da_base.py**: Base class providing Home Assistant API integration and common functionality
- **db_manager.py**: Database abstraction layer supporting SQLite, MySQL, PostgreSQL
- **da_prices.py**: Energy price data fetching from Entsoe and NordPool APIs
- **da_meteo.py**: Weather data integration for solar panel production forecasting
- **da_report.py**: Reporting and data analysis functionality

### Web Interface
- **webserver/da_server.py**: Flask application entry point
- **webserver/app/routes.py**: Web dashboard routes and API endpoints
- **webserver/gunicorn_config.py**: Production WSGI server configuration

## Configuration System

The system uses JSON-based configuration:
- **data/options.json**: Main configuration file with energy sources, pricing, and optimization settings
- **data/secrets.json**: API keys and sensitive configuration
- **config.yaml**: Home Assistant add-on metadata

Key configuration sections:
- `database ha`: Home Assistant database connection
- `database da`: DAO-specific database settings
- `prices`: Energy pricing configuration and tax rates
- `battery`: Battery characteristics and limits
- `electric vehicle`: EV charging optimization settings
- `boiler`: Water heater scheduling parameters
- `heating`: Heat pump configuration

## Development Commands

### Running the System
```bash
# Main entry point (production)
cd dao/prog && python3 da_scheduler.py

# Test mode
cd dao/run && bash run_test.sh

# Web server only
cd dao/webserver && python3 da_server.py
```

### Testing
```bash
# Run all tests
cd dao/tests/prog && python3 test_dao.py

# Database connectivity test
cd dao/prog && python3 check_db.py
```

### Dependencies
```bash
# Install requirements
pip install -r dao/requirements.txt
```

## Database Architecture

The system supports multiple database backends:
- **SQLite**: Default for development and small installations
- **MySQL/MariaDB**: Production deployments
- **PostgreSQL**: Enterprise installations

Key tables managed by db_manager.py handle:
- Energy price history
- Weather data
- Optimization results
- Home Assistant entity states

## Optimization Algorithm

The core `DaCalc` class in day_ahead.py implements:
1. **Data Collection**: Fetches prices, weather, and consumption history
2. **Model Building**: Creates MIP constraints for batteries, EVs, heating
3. **Optimization**: Solves for minimal cost or grid independence
4. **Result Processing**: Generates hourly schedules for all energy devices

Variables optimized:
- Battery charge/discharge cycles
- EV charging schedules  
- Heat pump operation timing
- Boiler heating periods

## Home Assistant Integration

Built on hassapi for:
- Entity state reading/writing
- Service calls for device control
- Notification sending
- Configuration management through HA secrets

## File Structure Patterns

- `dao/prog/`: Core calculation and scheduling logic
- `dao/webserver/`: Flask web interface
- `dao/data/`: Configuration and data storage
- `dao/tests/`: Unit and integration tests
- `dao/run/`: Startup and management scripts

## Key Dependencies

- **mip**: Mixed-Integer Programming solver
- **pandas**: Data manipulation and analysis
- **hassapi**: Home Assistant API integration
- **sqlalchemy**: Database ORM
- **flask**: Web dashboard framework
- **nordpool/entsoe-py**: Energy price APIs

## Logging and Monitoring

Structured logging to:
- `data/log/`: Application logs with rotation
- Home Assistant notification entities for alerts
- Web dashboard for real-time status monitoring
- Test de docker builds voordat je aangeeft dat de functionaliteit werkt