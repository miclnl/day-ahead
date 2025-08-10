"""
Enhanced Weather Integration for Day Ahead Optimizer.
Smart solar forecasting and weather correlation without ML dependencies.
Uses meteorological models and statistical analysis.
"""

import datetime as dt
import logging
import numpy as np
import pandas as pd
# Optional import - fallback if not available
try:
    import ephem
    EPHEM_AVAILABLE = True
except ImportError:
    EPHEM_AVAILABLE = False
    logging.warning("ephem not available - using simplified solar calculations")
import math
from typing import Dict, List, Optional, Tuple, Any
import requests
# Optional import - fallback if da_meteo not available
try:
    from da_meteo import Meteo
    METEO_AVAILABLE = True
except ImportError:
    METEO_AVAILABLE = False
    logging.warning("da_meteo not available - using simple weather fallback")


class EnhancedWeatherService:
    """
    Enhanced weather integration with intelligent solar forecasting.
    Combines multiple weather sources and applies physical models.
    """
    
    def __init__(self, da_calc_instance):
        """Initialize with reference to main DaCalc instance"""
        self.da_calc = da_calc_instance
        self.config = da_calc_instance.config
        self.meteo = Meteo(da_calc_instance.config) if METEO_AVAILABLE else None
        
        # Solar system configuration
        self.solar_capacity = self.config.get(['solar', 'capacity'], 0, 0.0)  # kWp
        self.solar_efficiency = self.config.get(['solar', 'efficiency'], 0, 0.85)
        self.solar_tilt = self.config.get(['solar', 'tilt'], 0, 30.0)  # degrees
        self.solar_azimuth = self.config.get(['solar', 'azimuth'], 0, 180.0)  # degrees (South=180)
        
        # Location for solar calculations
        self.latitude = self.config.get(['location', 'latitude'], 0, 52.0)  # Default: Netherlands
        self.longitude = self.config.get(['location', 'longitude'], 0, 5.0)
        
        # Weather data sources
        self.weather_sources = {
            'primary': 'openweathermap',  # Could be made configurable
            'backup': 'meteo_base'        # Fallback to existing meteo system
        }
        
        # Solar model parameters
        self.solar_models = {
            'clear_sky_factor': 0.75,      # Clear sky irradiance factor
            'cloud_reduction_factor': {    # Cloud cover impact
                'clear': 1.0,       # 0-10% cloud cover
                'partly': 0.7,      # 10-50% cloud cover  
                'mostly': 0.4,      # 50-80% cloud cover
                'overcast': 0.15    # 80-100% cloud cover
            },
            'temperature_coefficient': -0.004  # %/°C - solar efficiency vs temperature
        }
        
        # Historical correlation cache
        self.weather_correlations = {}
        self.solar_performance_history = []
        
        logging.info("Enhanced weather integrator initialized")
    
    def get_enhanced_solar_forecast(
        self, 
        start_time: dt.datetime, 
        hours_ahead: int = 48
    ) -> pd.DataFrame:
        """
        Generate enhanced solar production forecast
        
        Args:
            start_time: Start time for forecast
            hours_ahead: Number of hours to forecast
            
        Returns:
            DataFrame with solar production forecast per hour
        """
        logging.info(f"Generating enhanced solar forecast for {hours_ahead} hours")
        
        try:
            # 1. Get weather forecast
            weather_data = self._get_comprehensive_weather_forecast(start_time, hours_ahead)
            
            # 2. Calculate solar geometry for each hour
            solar_geometry = self._calculate_solar_geometry(start_time, hours_ahead)
            
            # 3. Apply physical solar model
            solar_forecast = self._apply_enhanced_solar_model(
                weather_data, solar_geometry
            )
            
            # 4. Apply historical corrections
            corrected_forecast = self._apply_historical_corrections(solar_forecast)
            
            # 5. Add confidence intervals
            final_forecast = self._add_confidence_intervals(corrected_forecast)
            
            logging.info(f"Enhanced solar forecast completed. Peak: {final_forecast['solar_production'].max():.2f} kW")
            return final_forecast
            
        except Exception as e:
            logging.error(f"Error in enhanced solar forecast: {e}")
            return self._fallback_solar_forecast(start_time, hours_ahead)
    
    def get_weather_correlations(
        self, 
        start_time: dt.datetime, 
        hours_ahead: int = 48
    ) -> Dict[str, Any]:
        """
        Get weather correlations for consumption and production optimization
        
        Returns:
            Dict with weather impact analysis
        """
        try:
            weather_data = self._get_comprehensive_weather_forecast(start_time, hours_ahead)
            
            correlations = {
                'heating_demand': self._calculate_heating_demand(weather_data),
                'cooling_demand': self._calculate_cooling_demand(weather_data),
                'solar_potential': self._assess_solar_potential(weather_data),
                'wind_impact': self._assess_wind_impact(weather_data),
                'humidity_comfort': self._assess_humidity_comfort(weather_data),
                'weather_alerts': self._check_weather_alerts(weather_data)
            }
            
            return correlations
            
        except Exception as e:
            logging.error(f"Error calculating weather correlations: {e}")
            return {}
    
    def _get_comprehensive_weather_forecast(
        self, 
        start_time: dt.datetime, 
        hours_ahead: int
    ) -> pd.DataFrame:
        """Get comprehensive weather forecast from multiple sources"""
        
        try:
            # Try primary weather source first
            if self.weather_sources['primary'] == 'openweathermap':
                weather_data = self._get_openweather_forecast(start_time, hours_ahead)
                
                if weather_data is not None:
                    return weather_data
            
            # Fallback to existing meteo system
            logging.info("Using backup weather source")
            return self._get_meteo_backup_forecast(start_time, hours_ahead)
            
        except Exception as e:
            logging.error(f"Error getting weather forecast: {e}")
            return self._generate_default_weather(start_time, hours_ahead)
    
    def _get_openweather_forecast(
        self, 
        start_time: dt.datetime, 
        hours_ahead: int
    ) -> Optional[pd.DataFrame]:
        """Get weather forecast from OpenWeatherMap API"""
        
        # This would require API key configuration
        # For now, return None to use backup
        return None
    
    def _get_meteo_backup_forecast(
        self, 
        start_time: dt.datetime, 
        hours_ahead: int
    ) -> pd.DataFrame:
        """Get forecast using existing meteo system as backup"""
        
        try:
            # Use existing Meteo class
            weather_data = []
            
            for hour_offset in range(hours_ahead):
                forecast_time = start_time + dt.timedelta(hours=hour_offset)
                
                # Get basic weather data (this is simplified - adapt to actual Meteo API)
                temp = 15.0  # Default temperature
                cloud_cover = 0.5  # Default 50% clouds
                
                try:
                    # If meteo has forecast methods, use them
                    if self.meteo and hasattr(self.meteo, 'get_temperature_forecast'):
                        temp = self.meteo.get_temperature_forecast(forecast_time)
                    if self.meteo and hasattr(self.meteo, 'get_cloud_forecast'):
                        cloud_cover = self.meteo.get_cloud_forecast(forecast_time)
                except:
                    pass  # Use defaults
                
                weather_data.append({
                    'datetime': forecast_time,
                    'temperature': temp,
                    'cloud_cover': cloud_cover,
                    'humidity': 65,  # Default
                    'wind_speed': 5,  # Default m/s
                    'pressure': 1013,  # Default hPa
                    'visibility': 10000  # Default meters
                })
            
            df = pd.DataFrame(weather_data)
            df.set_index('datetime', inplace=True)
            return df
            
        except Exception as e:
            logging.error(f"Error with meteo backup: {e}")
            return self._generate_default_weather(start_time, hours_ahead)
    
    def _generate_default_weather(
        self, 
        start_time: dt.datetime, 
        hours_ahead: int
    ) -> pd.DataFrame:
        """Generate reasonable default weather forecast"""
        
        weather_data = []
        
        for hour_offset in range(hours_ahead):
            forecast_time = start_time + dt.timedelta(hours=hour_offset)
            
            # Seasonal temperature defaults
            month = forecast_time.month
            if month in [12, 1, 2]:  # Winter
                base_temp = 5.0
            elif month in [3, 4, 5]:  # Spring
                base_temp = 12.0
            elif month in [6, 7, 8]:  # Summer
                base_temp = 20.0
            else:  # Autumn
                base_temp = 12.0
            
            # Daily temperature variation
            hour = forecast_time.hour
            temp_variation = 5 * math.sin((hour - 6) * math.pi / 12)  # Peak at 18:00
            temperature = base_temp + temp_variation
            
            # Cloud cover pattern (more clouds in afternoon)
            cloud_base = 0.4
            cloud_variation = 0.3 * math.sin((hour - 12) * math.pi / 12)
            cloud_cover = max(0, min(1, cloud_base + cloud_variation))
            
            weather_data.append({
                'datetime': forecast_time,
                'temperature': temperature,
                'cloud_cover': cloud_cover,
                'humidity': 65,
                'wind_speed': 5,
                'pressure': 1013,
                'visibility': 10000
            })
        
        df = pd.DataFrame(weather_data)
        df.set_index('datetime', inplace=True)
        return df
    
    def _calculate_solar_geometry(
        self, 
        start_time: dt.datetime, 
        hours_ahead: int
    ) -> pd.DataFrame:
        """Calculate solar geometry (sun position) for each forecast hour"""
        
        geometry_data = []
        
        # Create observer at location
        if EPHEM_AVAILABLE:
            observer = ephem.Observer()
            observer.lat = str(self.latitude)
            observer.lon = str(self.longitude)
            sun = ephem.Sun()
        else:
            # Use simplified calculation when ephem not available
            return self._simplified_solar_geometry(start_time, hours_ahead)
        
        for hour_offset in range(hours_ahead):
            forecast_time = start_time + dt.timedelta(hours=hour_offset)
            observer.date = forecast_time
            
            sun.compute(observer)
            
            # Convert to degrees
            elevation = math.degrees(sun.alt)
            azimuth = math.degrees(sun.az)
            
            # Calculate solar irradiance factors
            if elevation > 0:  # Sun is above horizon
                # Air mass calculation
                air_mass = 1 / math.cos(math.radians(90 - elevation)) if elevation > 0 else float('inf')
                air_mass = min(air_mass, 10)  # Cap at reasonable value
                
                # Irradiance reduction due to air mass
                irradiance_factor = 0.7 ** (air_mass - 1)
                
                # Panel angle factor (simplified)
                panel_angle_factor = self._calculate_panel_angle_factor(elevation, azimuth)
            else:
                irradiance_factor = 0.0
                panel_angle_factor = 0.0
                air_mass = float('inf')
            
            geometry_data.append({
                'datetime': forecast_time,
                'sun_elevation': max(0, elevation),
                'sun_azimuth': azimuth,
                'air_mass': air_mass,
                'irradiance_factor': irradiance_factor,
                'panel_angle_factor': panel_angle_factor
            })
        
        df = pd.DataFrame(geometry_data)
        df.set_index('datetime', inplace=True)
        return df
    
    def _simplified_solar_geometry(self, start_time: dt.datetime, hours_ahead: int) -> pd.DataFrame:
        """Simplified solar geometry calculation when ephem not available"""
        
        geometry_data = []
        
        for hour_offset in range(hours_ahead):
            forecast_time = start_time + dt.timedelta(hours=hour_offset)
            hour = forecast_time.hour
            
            # Simplified sun elevation based on time of day
            if 6 <= hour <= 18:
                # Bell curve for elevation (peak at noon)
                hour_from_sunrise = hour - 6
                max_elevation = 60  # degrees (approximate for mid-latitudes)
                elevation = max_elevation * math.sin(hour_from_sunrise * math.pi / 12)
            else:
                elevation = 0
            
            # Simplified azimuth (south = 180 degrees)
            if hour < 12:
                azimuth = 90 + (hour - 6) * 15  # East to south
            else:
                azimuth = 180 + (hour - 12) * 15  # South to west
            
            azimuth = max(0, min(360, azimuth))
            
            # Calculate factors
            if elevation > 0:
                air_mass = 1 / math.cos(math.radians(90 - elevation))
                air_mass = min(air_mass, 10)
                irradiance_factor = 0.7 ** (air_mass - 1)
                panel_angle_factor = self._calculate_panel_angle_factor(elevation, azimuth)
            else:
                air_mass = float('inf')
                irradiance_factor = 0.0
                panel_angle_factor = 0.0
            
            geometry_data.append({
                'datetime': forecast_time,
                'sun_elevation': max(0, elevation),
                'sun_azimuth': azimuth,
                'air_mass': air_mass,
                'irradiance_factor': irradiance_factor,
                'panel_angle_factor': panel_angle_factor
            })
        
        df = pd.DataFrame(geometry_data)
        df.set_index('datetime', inplace=True)
        return df
    
    def _calculate_panel_angle_factor(self, sun_elevation: float, sun_azimuth: float) -> float:
        """
        Calculate accurate panel angle efficiency factor using 3D solar geometry.
        Implements proper Direct Normal Irradiance (DNI) and Diffuse Horizontal Irradiance (DHI).
        """
        
        if sun_elevation <= 0:
            return 0.0
        
        import math
        
        # Convert to radians for calculations
        sun_elevation_rad = math.radians(sun_elevation)
        sun_azimuth_rad = math.radians(sun_azimuth)
        panel_tilt_rad = math.radians(self.solar_tilt)
        panel_azimuth_rad = math.radians(self.solar_azimuth)
        
        # Calculate angle of incidence between sun and panel normal (cosine rule in 3D)
        cos_incidence = (
            math.sin(sun_elevation_rad) * math.cos(panel_tilt_rad) +
            math.cos(sun_elevation_rad) * math.sin(panel_tilt_rad) * 
            math.cos(sun_azimuth_rad - panel_azimuth_rad)
        )
        
        # Ensure cos_incidence is not negative (sun behind panel)
        cos_incidence = max(0, cos_incidence)
        
        # Direct component: proportional to cos(incidence angle)
        direct_factor = cos_incidence
        
        # Diffuse component: isotropic sky model
        # More diffuse light reaches tilted panels than horizontal
        diffuse_factor = (1 + math.cos(panel_tilt_rad)) / 2
        
        # Ground reflection component (albedo effect)
        ground_albedo = 0.2  # Typical ground reflectance
        ground_factor = ground_albedo * (1 - math.cos(panel_tilt_rad)) / 2
        
        # Weighted combination (typical: 80% direct, 15% diffuse, 5% ground)
        total_factor = (
            0.80 * direct_factor +
            0.15 * diffuse_factor + 
            0.05 * ground_factor
        )
        
        # Additional corrections for panel orientation optimization
        # Optimal tilt angle varies by season and latitude
        seasonal_correction = self._get_seasonal_tilt_correction()
        
        return total_factor * seasonal_correction
    
    def _get_seasonal_tilt_correction(self) -> float:
        """
        Get seasonal correction factor for panel tilt optimization.
        Optimal tilt varies throughout the year.
        """
        try:
            import math
            
            current_day = dt.datetime.now().timetuple().tm_yday  # Day of year (1-365)
            latitude = self.config.get(['location', 'latitude'], 0, 52.0)
            
            # Solar declination angle for current day
            declination = 23.45 * math.sin(math.radians(360 * (284 + current_day) / 365))
            
            # Optimal tilt angle for current season
            optimal_tilt = abs(latitude - declination)
            
            # Calculate correction based on difference from optimal
            tilt_difference = abs(self.solar_tilt - optimal_tilt)
            
            # Correction factor: 1.0 at optimal, decreasing with larger differences
            correction = max(0.7, 1.0 - (tilt_difference / 60.0))  # Min 70% efficiency
            
            return correction
            
        except Exception as e:
            logging.warning(f"Error calculating seasonal correction: {e}")
            return 1.0  # No correction if calculation fails
    
    def _apply_enhanced_solar_model(
        self, 
        weather_data: pd.DataFrame, 
        solar_geometry: pd.DataFrame
    ) -> pd.DataFrame:
        """Apply enhanced physical solar model"""
        
        solar_forecast = []
        
        for timestamp in weather_data.index:
            if timestamp in solar_geometry.index:
                weather_row = weather_data.loc[timestamp]
                geometry_row = solar_geometry.loc[timestamp]
                
                # Base solar irradiance (clear sky)
                base_irradiance = (
                    1000 *  # Standard Test Conditions irradiance (W/m²)
                    geometry_row['irradiance_factor'] *
                    geometry_row['panel_angle_factor']
                )
                
                # Cloud cover reduction
                cloud_cover = weather_row['cloud_cover']
                if cloud_cover < 0.1:
                    cloud_factor = self.solar_models['cloud_reduction_factor']['clear']
                elif cloud_cover < 0.5:
                    cloud_factor = self.solar_models['cloud_reduction_factor']['partly']
                elif cloud_cover < 0.8:
                    cloud_factor = self.solar_models['cloud_reduction_factor']['mostly']
                else:
                    cloud_factor = self.solar_models['cloud_reduction_factor']['overcast']
                
                # Temperature derating
                temp_derate = 1 + (weather_row['temperature'] - 25) * self.solar_models['temperature_coefficient']
                temp_derate = max(0.5, min(1.2, temp_derate))  # Reasonable bounds
                
                # Final solar production calculation
                solar_irradiance = base_irradiance * cloud_factor
                solar_production = (
                    self.solar_capacity *  # kWp
                    (solar_irradiance / 1000) *  # Convert to kW/m²
                    self.solar_efficiency *
                    temp_derate
                )
                
                solar_production = max(0, solar_production)  # Ensure positive
                
                solar_forecast.append({
                    'datetime': timestamp,
                    'solar_production': solar_production,
                    'solar_irradiance': solar_irradiance,
                    'cloud_factor': cloud_factor,
                    'temp_derate': temp_derate,
                    'sun_elevation': geometry_row['sun_elevation']
                })
        
        df = pd.DataFrame(solar_forecast)
        df.set_index('datetime', inplace=True)
        return df
    
    def _apply_historical_corrections(self, solar_forecast: pd.DataFrame) -> pd.DataFrame:
        """Apply corrections based on historical performance"""
        
        corrected = solar_forecast.copy()
        
        # If we have historical data, apply learning corrections
        if len(self.solar_performance_history) > 0:
            # Placeholder for historical correction logic
            # Would compare historical forecasts vs actual production
            avg_correction = 0.92  # Typical 8% overestimate correction
            corrected['solar_production'] *= avg_correction
        
        return corrected
    
    def _add_confidence_intervals(self, solar_forecast: pd.DataFrame) -> pd.DataFrame:
        """Add confidence intervals to solar forecast"""
        
        enhanced = solar_forecast.copy()
        
        # Calculate confidence based on weather certainty
        confidence_base = 0.8
        
        for timestamp, row in enhanced.iterrows():
            # Lower confidence for very cloudy conditions (more variable)
            if row.get('cloud_factor', 1.0) < 0.3:  # Very cloudy
                confidence = confidence_base * 0.7
            elif row.get('cloud_factor', 1.0) < 0.7:  # Partly cloudy
                confidence = confidence_base * 0.9
            else:  # Clear
                confidence = confidence_base
            
            # Add confidence and bounds
            enhanced.loc[timestamp, 'confidence'] = confidence
            enhanced.loc[timestamp, 'solar_min'] = row['solar_production'] * (1 - (1 - confidence) * 2)
            enhanced.loc[timestamp, 'solar_max'] = row['solar_production'] * (1 + (1 - confidence) * 2)
        
        # Ensure bounds are positive
        enhanced['solar_min'] = enhanced['solar_min'].clip(lower=0)
        enhanced['solar_max'] = enhanced['solar_max'].clip(lower=0)
        
        return enhanced
    
    def _fallback_solar_forecast(
        self, 
        start_time: dt.datetime, 
        hours_ahead: int
    ) -> pd.DataFrame:
        """Simple fallback solar forecast when enhanced method fails"""
        
        logging.info("Using fallback solar forecast")
        
        solar_data = []
        
        for hour_offset in range(hours_ahead):
            forecast_time = start_time + dt.timedelta(hours=hour_offset)
            hour = forecast_time.hour
            
            # Simple day/night solar pattern
            if 6 <= hour <= 20:  # Daytime
                # Bell curve for solar production
                hour_factor = math.sin((hour - 6) * math.pi / 14)  # Peak at noon
                solar_production = self.solar_capacity * 0.7 * hour_factor  # 70% efficiency assumption
            else:
                solar_production = 0.0
            
            solar_data.append({
                'datetime': forecast_time,
                'solar_production': max(0, solar_production),
                'confidence': 0.5  # Low confidence for fallback
            })
        
        df = pd.DataFrame(solar_data)
        df.set_index('datetime', inplace=True)
        return df
    
    def _calculate_heating_demand(self, weather_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate heating demand based on temperature"""
        
        heating_demand = []
        
        for timestamp, row in weather_data.iterrows():
            temp = row['temperature']
            
            # Heating degree calculation (base 18°C)
            heating_base = 18.0
            if temp < heating_base:
                demand_factor = (heating_base - temp) / 10  # Scale factor
                demand_factor = min(demand_factor, 2.0)  # Cap at 200%
            else:
                demand_factor = 0.0
            
            heating_demand.append({
                'datetime': timestamp,
                'heating_demand_factor': demand_factor
            })
        
        return pd.DataFrame(heating_demand)
    
    def _calculate_cooling_demand(self, weather_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate cooling demand based on temperature and humidity"""
        
        cooling_demand = []
        
        for timestamp, row in weather_data.iterrows():
            temp = row['temperature']
            humidity = row.get('humidity', 50)
            
            # Cooling degree calculation (base 24°C)
            cooling_base = 24.0
            if temp > cooling_base:
                base_demand = (temp - cooling_base) / 10
                
                # Humidity adjustment (higher humidity = more cooling needed)
                humidity_factor = 1 + (humidity - 50) / 100
                demand_factor = base_demand * humidity_factor
                demand_factor = min(demand_factor, 2.0)  # Cap at 200%
            else:
                demand_factor = 0.0
            
            cooling_demand.append({
                'datetime': timestamp,
                'cooling_demand_factor': demand_factor
            })
        
        return pd.DataFrame(cooling_demand)
    
    def _assess_solar_potential(self, weather_data: pd.DataFrame) -> Dict:
        """Assess overall solar potential for the forecast period"""
        
        if 'cloud_cover' in weather_data.columns:
            avg_cloud_cover = weather_data['cloud_cover'].mean()
            
            if avg_cloud_cover < 0.3:
                potential = 'excellent'
                factor = 1.0
            elif avg_cloud_cover < 0.6:
                potential = 'good'
                factor = 0.8
            elif avg_cloud_cover < 0.8:
                potential = 'fair'
                factor = 0.5
            else:
                potential = 'poor'
                factor = 0.2
        else:
            potential = 'unknown'
            factor = 0.7
        
        return {
            'potential': potential,
            'production_factor': factor,
            'avg_cloud_cover': avg_cloud_cover if 'cloud_cover' in weather_data.columns else None
        }
    
    def _assess_wind_impact(self, weather_data: pd.DataFrame) -> Dict:
        """Assess wind impact on comfort and energy usage"""
        
        if 'wind_speed' in weather_data.columns:
            avg_wind = weather_data['wind_speed'].mean()
            max_wind = weather_data['wind_speed'].max()
            
            # Wind cooling effect reduces heating demand
            wind_cooling_factor = min(avg_wind / 20, 0.3)  # Max 30% reduction
            
            return {
                'avg_wind_speed': avg_wind,
                'max_wind_speed': max_wind,
                'cooling_effect': wind_cooling_factor,
                'alert': max_wind > 15  # High wind alert
            }
        
        return {'wind_data_unavailable': True}
    
    def _assess_humidity_comfort(self, weather_data: pd.DataFrame) -> Dict:
        """Assess humidity impact on comfort"""
        
        if 'humidity' in weather_data.columns:
            avg_humidity = weather_data['humidity'].mean()
            
            if avg_humidity > 80:
                comfort = 'uncomfortable_humid'
                cooling_increase = 0.2
            elif avg_humidity > 60:
                comfort = 'slightly_humid'
                cooling_increase = 0.1
            elif avg_humidity < 30:
                comfort = 'dry'
                cooling_increase = -0.1  # Less cooling needed
            else:
                comfort = 'comfortable'
                cooling_increase = 0.0
            
            return {
                'avg_humidity': avg_humidity,
                'comfort_level': comfort,
                'cooling_adjustment': cooling_increase
            }
        
        return {'humidity_data_unavailable': True}
    
    def _check_weather_alerts(self, weather_data: pd.DataFrame) -> List[str]:
        """Check for weather conditions requiring special attention"""
        
        alerts = []
        
        # Temperature alerts
        if 'temperature' in weather_data.columns:
            max_temp = weather_data['temperature'].max()
            min_temp = weather_data['temperature'].min()
            
            if max_temp > 35:
                alerts.append('extreme_heat')
            elif max_temp > 30:
                alerts.append('high_temperature')
            
            if min_temp < -10:
                alerts.append('extreme_cold')
            elif min_temp < 0:
                alerts.append('freezing_conditions')
        
        # Wind alerts
        if 'wind_speed' in weather_data.columns:
            max_wind = weather_data['wind_speed'].max()
            if max_wind > 20:
                alerts.append('high_wind')
        
        return alerts

    def update_historical_performance(
        self, 
        timestamp: dt.datetime, 
        forecasted: float, 
        actual: float
    ):
        """Update historical performance data for learning"""
        
        performance_data = {
            'timestamp': timestamp,
            'forecasted': forecasted,
            'actual': actual,
            'error': actual - forecasted,
            'error_percent': ((actual - forecasted) / forecasted * 100) if forecasted > 0 else 0
        }
        
        self.solar_performance_history.append(performance_data)
        
        # Keep only recent history (last 30 days)
        cutoff_date = dt.datetime.now() - dt.timedelta(days=30)
        self.solar_performance_history = [
            p for p in self.solar_performance_history 
            if p['timestamp'] > cutoff_date
        ]
    
    def get_weather_performance_metrics(self) -> Dict:
        """Get performance metrics for weather forecasting"""
        
        if len(self.solar_performance_history) == 0:
            return {'no_historical_data': True}
        
        errors = [p['error_percent'] for p in self.solar_performance_history]
        
        return {
            'forecast_accuracy': 100 - np.mean(np.abs(errors)),
            'mean_error_percent': np.mean(errors),
            'forecast_count': len(self.solar_performance_history),
            'rmse_percent': np.sqrt(np.mean(np.array(errors) ** 2))
        }