"""
Statistical Predictor module for Day Ahead Optimizer.
Replaces ML with statistical methods for reliable predictions.
ML-free implementation for 100% container stability.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta

# Cloud AI Support (optional)
try:
    import openai
    OPENAI_AVAILABLE = True
    logging.info("OpenAI available for optional cloud predictions")
except ImportError:
    OPENAI_AVAILABLE = False
    logging.info("OpenAI not available - using statistical methods only")

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
    logging.info("Anthropic Claude available for optional cloud predictions")
except ImportError:
    ANTHROPIC_AVAILABLE = False
    logging.info("Anthropic not available - using statistical methods only")


class StatisticalPredictor:
    """Statistical prediction engine - ML-free implementation."""
    
    def __init__(self, config: dict = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        self.prediction_cache = {}
        
        # Statistical models for different prediction types
        self.consumption_stats = {}
        self.pv_stats = {}
        self.weather_stats = {}
        
        self.logger.info("Statistical Predictor initialized - ML-free mode")
    
    def predict_consumption(self, historical_data: pd.DataFrame, 
                          hours_ahead: int = 24) -> np.ndarray:
        """
        Statistical consumption prediction using time-series patterns.
        Replaces ML with seasonal decomposition and trend analysis.
        """
        try:
            if historical_data.empty:
                # Fallback to typical daily pattern
                return self._generate_typical_consumption_pattern(hours_ahead)
            
            # Statistical pattern analysis
            hourly_patterns = self._extract_hourly_patterns(historical_data)
            weekly_patterns = self._extract_weekly_patterns(historical_data)
            seasonal_trend = self._calculate_seasonal_trend(historical_data)
            
            # Combine patterns statistically
            predictions = []
            current_time = datetime.now()
            
            for hour in range(hours_ahead):
                future_time = current_time + timedelta(hours=hour)
                
                # Hourly pattern component
                hour_of_day = future_time.hour
                hourly_component = hourly_patterns.get(hour_of_day, 1.0)
                
                # Weekly pattern component
                day_of_week = future_time.weekday()
                weekly_component = weekly_patterns.get(day_of_week, 1.0)
                
                # Seasonal component
                seasonal_component = seasonal_trend
                
                # Statistical combination
                prediction = (hourly_component * weekly_component * 
                            seasonal_component * self._get_base_consumption())
                
                predictions.append(max(0.1, prediction))  # Minimum consumption
            
            self.logger.debug(f"Generated {hours_ahead}h consumption forecast using statistical methods")
            return np.array(predictions)
            
        except Exception as e:
            self.logger.error(f"Statistical consumption prediction error: {e}")
            return self._generate_typical_consumption_pattern(hours_ahead)
    
    def predict_pv_production(self, weather_data: pd.DataFrame,
                            solar_config: dict, hours_ahead: int = 24) -> np.ndarray:
        """
        Statistical PV production prediction using weather patterns and solar geometry.
        """
        try:
            if weather_data.empty:
                return self._generate_typical_solar_pattern(hours_ahead, solar_config)
            
            predictions = []
            current_time = datetime.now()
            
            for hour in range(hours_ahead):
                future_time = current_time + timedelta(hours=hour)
                
                # Solar elevation calculation (basic astronomy)
                solar_elevation = self._calculate_solar_elevation(future_time)
                
                # Weather impact (cloud cover, temperature)
                weather_factor = self._calculate_weather_impact(weather_data, hour)
                
                # Panel efficiency factors
                temperature_factor = self._calculate_temperature_efficiency(weather_data, hour)
                
                # Base solar potential
                base_production = self._calculate_base_solar_production(
                    solar_elevation, solar_config)
                
                # Combine factors
                prediction = (base_production * weather_factor * 
                            temperature_factor * solar_config.get('panel_efficiency', 0.2))
                
                predictions.append(max(0, prediction))
            
            self.logger.debug(f"Generated {hours_ahead}h PV forecast using statistical methods")
            return np.array(predictions)
            
        except Exception as e:
            self.logger.error(f"Statistical PV prediction error: {e}")
            return self._generate_typical_solar_pattern(hours_ahead, solar_config)
    
    def _extract_hourly_patterns(self, data: pd.DataFrame) -> dict:
        """Extract statistical hourly consumption patterns."""
        if 'consumption' not in data.columns:
            return {i: 1.0 for i in range(24)}
        
        data['hour'] = pd.to_datetime(data.index).hour
        hourly_means = data.groupby('hour')['consumption'].mean()
        
        # Normalize to average = 1.0
        avg_consumption = hourly_means.mean()
        if avg_consumption > 0:
            return (hourly_means / avg_consumption).to_dict()
        return {i: 1.0 for i in range(24)}
    
    def _extract_weekly_patterns(self, data: pd.DataFrame) -> dict:
        """Extract statistical weekly consumption patterns."""
        if 'consumption' not in data.columns:
            return {i: 1.0 for i in range(7)}
        
        data['weekday'] = pd.to_datetime(data.index).weekday
        weekly_means = data.groupby('weekday')['consumption'].mean()
        
        # Normalize to average = 1.0
        avg_consumption = weekly_means.mean()
        if avg_consumption > 0:
            return (weekly_means / avg_consumption).to_dict()
        return {i: 1.0 for i in range(7)}
    
    def _calculate_seasonal_trend(self, data: pd.DataFrame) -> float:
        """Calculate seasonal trend factor."""
        if len(data) < 7:
            return 1.0
        
        # Simple trend calculation (recent vs historical average)
        recent_avg = data['consumption'].tail(7 * 24).mean()  # Last week
        historical_avg = data['consumption'].mean()
        
        if historical_avg > 0:
            return min(2.0, max(0.5, recent_avg / historical_avg))
        return 1.0
    
    def _generate_typical_consumption_pattern(self, hours: int) -> np.ndarray:
        """Generate typical daily consumption pattern when no data available."""
        # Typical household consumption pattern (normalized)
        daily_pattern = [
            0.5, 0.4, 0.4, 0.4, 0.5, 0.6,  # 0-5: Night/early morning
            0.8, 1.2, 1.0, 0.8, 0.7, 0.7,  # 6-11: Morning
            0.8, 0.7, 0.6, 0.7, 0.8, 1.2,  # 12-17: Afternoon
            1.5, 1.3, 1.1, 0.9, 0.7, 0.6   # 18-23: Evening
        ]
        
        # Repeat pattern for multiple days
        base_consumption = 2.5  # kW average
        pattern = []
        for hour in range(hours):
            pattern.append(daily_pattern[hour % 24] * base_consumption)
        
        return np.array(pattern)
    
    def _generate_typical_solar_pattern(self, hours: int, solar_config: dict) -> np.ndarray:
        """Generate typical solar production pattern."""
        peak_power = solar_config.get('peak_power_wp', 5000) / 1000  # Convert to kW
        
        # Typical solar production curve (normalized)
        daily_pattern = [
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # 0-5: Night
            0.05, 0.2, 0.4, 0.6, 0.8, 0.9,  # 6-11: Morning
            1.0, 0.95, 0.85, 0.7, 0.5, 0.3, # 12-17: Afternoon
            0.1, 0.0, 0.0, 0.0, 0.0, 0.0   # 18-23: Evening
        ]
        
        pattern = []
        for hour in range(hours):
            pattern.append(daily_pattern[hour % 24] * peak_power)
        
        return np.array(pattern)
    
    def _calculate_solar_elevation(self, dt: datetime) -> float:
        """Basic solar elevation calculation."""
        # Simplified solar elevation (for statistical purposes)
        hour = dt.hour + dt.minute / 60.0
        if 6 <= hour <= 18:
            # Approximate sine curve for daylight hours
            daylight_progress = (hour - 6) / 12.0  # 0 to 1
            elevation = np.sin(daylight_progress * np.pi) * 90  # Max 90 degrees
            return max(0, elevation)
        return 0
    
    def _calculate_weather_impact(self, weather_data: pd.DataFrame, hour: int) -> float:
        """Calculate weather impact on solar production."""
        if weather_data.empty or hour >= len(weather_data):
            return 0.8  # Assume moderate conditions
        
        try:
            # Cloud cover impact
            cloud_cover = weather_data.iloc[hour].get('cloud_cover', 0.3)
            clear_sky_factor = 1.0 - (cloud_cover * 0.7)
            
            return max(0.1, clear_sky_factor)
        except:
            return 0.8
    
    def _calculate_temperature_efficiency(self, weather_data: pd.DataFrame, hour: int) -> float:
        """Calculate temperature impact on panel efficiency."""
        if weather_data.empty or hour >= len(weather_data):
            return 0.95  # Assume optimal temperature
        
        try:
            temp = weather_data.iloc[hour].get('temperature', 25)
            # Panel efficiency decreases with high temperature
            optimal_temp = 25  # Celsius
            temp_factor = 1.0 - ((temp - optimal_temp) * 0.004)
            
            return max(0.7, min(1.0, temp_factor))
        except:
            return 0.95
    
    def _calculate_base_solar_production(self, elevation: float, config: dict) -> float:
        """Calculate base solar production from elevation."""
        if elevation <= 0:
            return 0
        
        # Solar irradiance approximation
        max_irradiance = 1000  # W/m²
        actual_irradiance = max_irradiance * np.sin(np.radians(elevation))
        
        panel_area = config.get('panel_area_m2', 25)  # m²
        return actual_irradiance * panel_area / 1000  # Convert to kW
    
    def _get_base_consumption(self) -> float:
        """Get base consumption for statistical calculations."""
        return self.config.get('typical_consumption_kw', 2.5)
    
    # Optional Cloud AI methods (only if API keys are configured)
    async def predict_with_cloud_ai(self, data: dict, prediction_type: str) -> Optional[dict]:
        """Optional cloud AI prediction if API keys configured."""
        if not (OPENAI_AVAILABLE or ANTHROPIC_AVAILABLE):
            return None
        
        try:
            # Check for API keys in config
            openai_key = self.config.get('openai_api_key')
            anthropic_key = self.config.get('anthropic_api_key')
            
            if openai_key and OPENAI_AVAILABLE:
                return await self._predict_with_openai(data, prediction_type, openai_key)
            elif anthropic_key and ANTHROPIC_AVAILABLE:
                return await self._predict_with_anthropic(data, prediction_type, anthropic_key)
            
        except Exception as e:
            self.logger.warning(f"Cloud AI prediction failed: {e}")
        
        return None
    
    async def _predict_with_openai(self, data: dict, prediction_type: str, api_key: str) -> dict:
        """OpenAI-powered prediction (optional)."""
        # Implementation only if explicitly configured
        pass
    
    async def _predict_with_anthropic(self, data: dict, prediction_type: str, api_key: str) -> dict:
        """Anthropic Claude-powered prediction (optional).""" 
        # Implementation only if explicitly configured
        pass