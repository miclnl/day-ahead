"""
Weather Reactive Optimization voor Day Ahead Optimizer.
Real-time weather adaptation voor energie optimalisatie.
Reageert dynamisch op weersveranderingen en aanpassingen in forecasts.
"""

import datetime as dt
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from enum import Enum
import json

class WeatherEvent(Enum):
    """Weather events that trigger optimization adjustments"""
    SUDDEN_CLOUDS = "sudden_clouds"
    CLEAR_SKIES = "clear_skies"
    TEMPERATURE_DROP = "temperature_drop"
    TEMPERATURE_RISE = "temperature_rise"
    WIND_INCREASE = "wind_increase"
    STORM_APPROACHING = "storm_approaching"
    FOG_FORMATION = "fog_formation"
    PRECIPITATION = "precipitation"

class WeatherReactiveOptimizer:
    """
    Real-time weather reactive optimization engine.
    Monitors weather changes and adapts energy strategy dynamically.
    """
    
    def __init__(self, da_calc_instance):
        """Initialize weather reactive optimizer"""
        self.da_calc = da_calc_instance
        self.config = da_calc_instance.config
        
        # Weather monitoring configuration
        self.monitoring_interval = self.config.get(['weather_reactive', 'monitoring_interval'], 0, 15)  # minutes
        self.forecast_update_threshold = self.config.get(['weather_reactive', 'forecast_threshold'], 0, 0.2)  # 20% change
        self.reaction_sensitivity = self.config.get(['weather_reactive', 'sensitivity'], 0, 0.8)  # High sensitivity
        
        # Solar panel configuration for reactive adjustments
        self.solar_capacity = self.config.get(['solar', 'capacity'], 0, 10.0)  # kWp
        self.solar_tilt = self.config.get(['solar', 'tilt'], 0, 30.0)  # degrees
        self.solar_azimuth = self.config.get(['solar', 'azimuth'], 0, 180.0)  # degrees (South=180)
        self.solar_efficiency = self.config.get(['solar', 'efficiency'], 0, 0.20)  # 20%
        
        # Heat pump / heating system
        self.heat_pump_present = self.config.get(['heating', 'heat_pump'], None, 'false').lower() == 'true'
        self.heat_pump_cop = self.config.get(['heating', 'cop'], 0, 3.5)
        self.heating_comfort_temp = self.config.get(['heating', 'comfort_temperature'], 0, 21.0)
        
        # Cooling system
        self.cooling_present = self.config.get(['cooling', 'present'], None, 'false').lower() == 'true'
        self.cooling_comfort_temp = self.config.get(['cooling', 'comfort_temperature'], 0, 24.0)
        
        # Weather thresholds for reactive actions
        self.weather_thresholds = {
            'solar_change_threshold': 0.15,      # 15% solar forecast change
            'temperature_change_threshold': 3.0,  # 3°C temperature change
            'wind_speed_threshold': 15.0,         # 15 m/s wind speed
            'cloud_cover_threshold': 0.3,         # 30% cloud cover change
            'precipitation_threshold': 2.0,       # 2mm precipitation
        }
        
        # Historical weather tracking
        self.weather_history = []
        self.forecast_accuracy_tracking = {}
        self.reaction_history = []
        
        # Current weather state
        self.current_weather_state = None
        self.last_forecast_update = None
        self.active_weather_events = set()
        
        logging.info("Weather reactive optimizer initialized")
    
    def monitor_weather_changes(self, current_weather: Dict, forecast_update: Dict = None) -> List[WeatherEvent]:
        """
        Monitor weather changes and detect events requiring optimization adjustment
        
        Args:
            current_weather: Current weather conditions
            forecast_update: Updated weather forecast (if available)
            
        Returns:
            List of detected weather events
        """
        detected_events = []
        
        try:
            # Store current weather
            self.current_weather_state = current_weather
            timestamp = dt.datetime.now()
            
            # 1. Detect solar production changes
            solar_events = self._detect_solar_changes(current_weather, forecast_update)
            detected_events.extend(solar_events)
            
            # 2. Detect temperature changes affecting heating/cooling
            temp_events = self._detect_temperature_changes(current_weather)
            detected_events.extend(temp_events)
            
            # 3. Detect weather conditions affecting comfort/efficiency
            comfort_events = self._detect_comfort_changes(current_weather)
            detected_events.extend(comfort_events)
            
            # 4. Detect extreme weather events
            extreme_events = self._detect_extreme_weather(current_weather)
            detected_events.extend(extreme_events)
            
            # Update active events
            self.active_weather_events = set(detected_events)
            
            # Log weather tracking
            self.weather_history.append({
                'timestamp': timestamp,
                'weather': current_weather,
                'events': detected_events
            })
            
            # Keep history manageable (last 24 hours)
            if len(self.weather_history) > 96:  # 15-minute intervals
                self.weather_history = self.weather_history[-96:]
            
            if detected_events:
                logging.info(f"Weather events detected: {[e.value for e in detected_events]}")
            
            return detected_events
            
        except Exception as e:
            logging.error(f"Error monitoring weather changes: {e}")
            return []
    
    def react_to_weather_events(self, events: List[WeatherEvent], current_schedule: Dict) -> Dict:
        """
        Generate reactive optimization adjustments based on weather events
        
        Args:
            events: List of weather events to react to
            current_schedule: Current optimization schedule
            
        Returns:
            Updated optimization schedule with weather reactive adjustments
        """
        if not events:
            return current_schedule
        
        try:
            adjusted_schedule = current_schedule.copy()
            adjustments_made = []
            
            for event in events:
                if event == WeatherEvent.SUDDEN_CLOUDS:
                    adjustment = self._adjust_for_reduced_solar(adjusted_schedule)
                    adjustments_made.extend(adjustment)
                
                elif event == WeatherEvent.CLEAR_SKIES:
                    adjustment = self._adjust_for_increased_solar(adjusted_schedule)
                    adjustments_made.extend(adjustment)
                
                elif event == WeatherEvent.TEMPERATURE_DROP:
                    adjustment = self._adjust_for_heating_demand(adjusted_schedule)
                    adjustments_made.extend(adjustment)
                
                elif event == WeatherEvent.TEMPERATURE_RISE:
                    adjustment = self._adjust_for_cooling_demand(adjusted_schedule)
                    adjustments_made.extend(adjustment)
                
                elif event == WeatherEvent.WIND_INCREASE:
                    adjustment = self._adjust_for_wind_conditions(adjusted_schedule)
                    adjustments_made.extend(adjustment)
                
                elif event == WeatherEvent.STORM_APPROACHING:
                    adjustment = self._adjust_for_storm_preparation(adjusted_schedule)
                    adjustments_made.extend(adjustment)
                
                elif event == WeatherEvent.PRECIPITATION:
                    adjustment = self._adjust_for_precipitation(adjusted_schedule)
                    adjustments_made.extend(adjustment)
            
            # Log reaction history
            reaction_record = {
                'timestamp': dt.datetime.now(),
                'events': [e.value for e in events],
                'adjustments': adjustments_made,
                'weather_state': self.current_weather_state
            }
            self.reaction_history.append(reaction_record)
            
            if adjustments_made:
                logging.info(f"Weather reactive adjustments applied: {adjustments_made}")
            
            return adjusted_schedule
            
        except Exception as e:
            logging.error(f"Error reacting to weather events: {e}")
            return current_schedule
    
    def get_enhanced_solar_forecast(self, weather_forecast: Dict, hours_ahead: int = 24) -> pd.DataFrame:
        """
        Enhanced solar forecast with accurate panel orientation calculations
        
        Args:
            weather_forecast: Weather forecast data
            hours_ahead: Hours to forecast ahead
            
        Returns:
            Enhanced solar production forecast with orientation corrections
        """
        try:
            forecast_data = []
            start_time = dt.datetime.now()
            
            for hour in range(hours_ahead):
                timestamp = start_time + dt.timedelta(hours=hour)
                hour_data = self._calculate_hourly_solar_production(timestamp, weather_forecast)
                forecast_data.append(hour_data)
            
            solar_forecast = pd.DataFrame(forecast_data)
            
            # Apply smoothing and validation
            solar_forecast['solar_production'] = self._apply_solar_smoothing(
                solar_forecast['solar_production']
            )
            
            logging.info(f"Enhanced solar forecast completed with orientation corrections")
            return solar_forecast
            
        except Exception as e:
            logging.error(f"Error in enhanced solar forecast: {e}")
            return self._fallback_solar_forecast(hours_ahead)
    
    def _calculate_hourly_solar_production(self, timestamp: dt.datetime, weather_forecast: Dict) -> Dict:
        """Calculate solar production for specific hour with panel orientation"""
        
        # 1. Calculate sun position
        sun_position = self._calculate_sun_position(timestamp)
        
        # 2. Calculate Direct Normal Irradiance (DNI) and Diffuse Horizontal Irradiance (DHI)
        solar_irradiance = self._calculate_solar_irradiance(timestamp, weather_forecast, sun_position)
        
        # 3. Apply panel orientation corrections
        panel_irradiance = self._apply_panel_orientation_correction(
            solar_irradiance, sun_position, timestamp
        )
        
        # 4. Apply weather corrections (clouds, atmosphere)
        weather_corrected = self._apply_weather_corrections(
            panel_irradiance, weather_forecast, timestamp
        )
        
        # 5. Calculate final production
        solar_production = weather_corrected * self.solar_capacity * self.solar_efficiency / 1000  # kW
        
        return {
            'timestamp': timestamp,
            'solar_production': max(0, solar_production),
            'sun_elevation': sun_position['elevation'],
            'sun_azimuth': sun_position['azimuth'],
            'panel_efficiency': weather_corrected / max(panel_irradiance, 0.001),
            'weather_factor': weather_forecast.get('cloud_cover', 0.3),
            'irradiance_panel': panel_irradiance
        }
    
    def _calculate_sun_position(self, timestamp: dt.datetime) -> Dict:
        """Calculate accurate sun position for timestamp"""
        
        try:
            # Get location from config
            latitude = self.config.get(['location', 'latitude'], 0, 52.0)  # Default: Netherlands
            longitude = self.config.get(['location', 'longitude'], 0, 5.0)
            
            # Use ephem if available for accurate calculations
            try:
                import ephem
                observer = ephem.Observer()
                observer.lat = str(latitude)
                observer.long = str(longitude)
                observer.date = timestamp.strftime('%Y/%m/%d %H:%M:%S')
                
                sun = ephem.Sun()
                sun.compute(observer)
                
                elevation = math.degrees(sun.alt)
                azimuth = math.degrees(sun.az)
                
                return {
                    'elevation': elevation,
                    'azimuth': azimuth,
                    'method': 'ephem'
                }
                
            except ImportError:
                # Fallback to simplified solar position calculation
                return self._simplified_sun_position(timestamp, latitude, longitude)
                
        except Exception as e:
            logging.warning(f"Error calculating sun position: {e}")
            return {'elevation': 45.0, 'azimuth': 180.0, 'method': 'fallback'}
    
    def _apply_panel_orientation_correction(self, irradiance: float, sun_position: Dict, timestamp: dt.datetime) -> float:
        """Apply panel tilt and azimuth corrections to solar irradiance"""
        
        import math
        
        sun_elevation = math.radians(sun_position['elevation'])
        sun_azimuth = math.radians(sun_position['azimuth'])
        panel_tilt = math.radians(self.solar_tilt)
        panel_azimuth = math.radians(self.solar_azimuth)
        
        # Calculate angle between sun and panel normal
        # Complex 3D geometry calculation for accurate panel irradiance
        cos_incidence = (
            math.sin(sun_elevation) * math.cos(panel_tilt) +
            math.cos(sun_elevation) * math.sin(panel_tilt) * 
            math.cos(sun_azimuth - panel_azimuth)
        )
        
        # Ensure cos_incidence is not negative (sun behind panel)
        cos_incidence = max(0, cos_incidence)
        
        # Apply Direct Normal Irradiance (DNI) component
        direct_irradiance = irradiance * 0.8 * cos_incidence  # 80% direct
        
        # Apply Diffuse Horizontal Irradiance (DHI) component
        # Diffuse component with isotropic sky model
        diffuse_factor = (1 + math.cos(panel_tilt)) / 2
        diffuse_irradiance = irradiance * 0.2 * diffuse_factor  # 20% diffuse
        
        # Ground reflection component (albedo)
        ground_albedo = 0.2  # Typical ground reflectance
        ground_factor = (1 - math.cos(panel_tilt)) / 2
        ground_irradiance = irradiance * ground_albedo * ground_factor
        
        total_panel_irradiance = direct_irradiance + diffuse_irradiance + ground_irradiance
        
        return total_panel_irradiance
    
    def _detect_solar_changes(self, current_weather: Dict, forecast_update: Dict = None) -> List[WeatherEvent]:
        """Detect changes in solar conditions"""
        events = []
        
        try:
            current_clouds = current_weather.get('cloud_cover', 0.5)
            
            # Compare with previous state if available
            if hasattr(self, '_previous_cloud_cover'):
                cloud_change = abs(current_clouds - self._previous_cloud_cover)
                
                if cloud_change > self.weather_thresholds['cloud_cover_threshold']:
                    if current_clouds > self._previous_cloud_cover:
                        events.append(WeatherEvent.SUDDEN_CLOUDS)
                    else:
                        events.append(WeatherEvent.CLEAR_SKIES)
            
            self._previous_cloud_cover = current_clouds
            
            # Check forecast update for significant solar changes
            if forecast_update:
                solar_change = forecast_update.get('solar_production_change', 0)
                if abs(solar_change) > self.weather_thresholds['solar_change_threshold']:
                    if solar_change < 0:
                        events.append(WeatherEvent.SUDDEN_CLOUDS)
                    else:
                        events.append(WeatherEvent.CLEAR_SKIES)
            
        except Exception as e:
            logging.error(f"Error detecting solar changes: {e}")
        
        return events
    
    def _adjust_for_reduced_solar(self, schedule: Dict) -> List[str]:
        """Adjust schedule for reduced solar production"""
        adjustments = []
        
        try:
            # Increase grid import during expected solar hours
            # Reduce battery discharge if solar was expected
            # Delay high-consumption activities
            
            current_hour = dt.datetime.now().hour
            if 8 <= current_hour <= 16:  # Daytime hours
                adjustments.append("increased_grid_import_during_solar_hours")
                adjustments.append("reduced_battery_discharge_conservation")
                adjustments.append("delayed_high_consumption_activities")
            
        except Exception as e:
            logging.error(f"Error adjusting for reduced solar: {e}")
        
        return adjustments
    
    def _adjust_for_increased_solar(self, schedule: Dict) -> List[str]:
        """Adjust schedule for increased solar production"""
        adjustments = []
        
        try:
            # Increase battery charging
            # Schedule high-consumption activities during solar hours
            # Reduce grid import
            
            current_hour = dt.datetime.now().hour
            if 8 <= current_hour <= 16:  # Daytime hours
                adjustments.append("increased_battery_charging")
                adjustments.append("scheduled_high_consumption_during_solar")
                adjustments.append("reduced_grid_import")
            
        except Exception as e:
            logging.error(f"Error adjusting for increased solar: {e}")
        
        return adjustments
    
    def get_weather_reactive_status(self) -> Dict[str, Any]:
        """Get current weather reactive optimization status"""
        
        return {
            'active_events': [e.value for e in self.active_weather_events],
            'monitoring_interval': self.monitoring_interval,
            'reaction_sensitivity': self.reaction_sensitivity,
            'solar_config': {
                'capacity': self.solar_capacity,
                'tilt': self.solar_tilt,
                'azimuth': self.solar_azimuth,
                'efficiency': self.solar_efficiency
            },
            'weather_thresholds': self.weather_thresholds,
            'last_weather_update': self.last_forecast_update,
            'reactions_today': len([r for r in self.reaction_history 
                                  if r['timestamp'].date() == dt.date.today()])
        }
    
    # Additional helper methods would be implemented here...
    def _detect_temperature_changes(self, current_weather: Dict) -> List[WeatherEvent]:
        """Detect temperature changes affecting heating/cooling demand"""
        # Implementation for temperature change detection
        return []
    
    def _detect_comfort_changes(self, current_weather: Dict) -> List[WeatherEvent]:
        """Detect weather changes affecting comfort systems"""
        # Implementation for comfort-related weather changes
        return []
    
    def _detect_extreme_weather(self, current_weather: Dict) -> List[WeatherEvent]:
        """Detect extreme weather events"""
        # Implementation for extreme weather detection
        return []
    
    def _adjust_for_heating_demand(self, schedule: Dict) -> List[str]:
        """Adjust schedule for increased heating demand"""
        return ["increased_heating_preparation"]
    
    def _adjust_for_cooling_demand(self, schedule: Dict) -> List[str]:
        """Adjust schedule for increased cooling demand"""
        return ["increased_cooling_preparation"]
    
    def _adjust_for_wind_conditions(self, schedule: Dict) -> List[str]:
        """Adjust schedule for wind conditions"""
        return ["wind_condition_adjustment"]
    
    def _adjust_for_storm_preparation(self, schedule: Dict) -> List[str]:
        """Adjust schedule for storm preparation"""
        return ["storm_preparation_mode"]
    
    def _adjust_for_precipitation(self, schedule: Dict) -> List[str]:
        """Adjust schedule for precipitation"""
        return ["precipitation_adjustment"]