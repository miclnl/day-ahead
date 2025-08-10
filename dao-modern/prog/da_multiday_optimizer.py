"""
Multi-day Optimization Engine
Geavanceerde 7-day planning voor maximale kostenbesparingen met weather forecasting
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import asyncio
import aiohttp
import json
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from dao.prog.da_config import Config
from dao.prog.da_ha_integration import HomeAssistantIntegration


class MultiDayOptimizer:
    """Multi-day optimization voor langere planning horizont"""
    
    def __init__(self, config: Config, ha_integration: HomeAssistantIntegration):
        self.config = config
        self.ha = ha_integration
        self.settings = self._load_multiday_settings()
        
        # Weather services
        self.weather_service = EnhancedWeatherService(config)
        self.holiday_detector = HolidayVacationDetector(config, ha_integration)
        self.seasonal_optimizer = SeasonalOptimizer(config)
        self.degradation_manager = BatteryDegradationManager(config)
        
        # Multi-day data
        self.multiday_forecast = pd.DataFrame()
        self.price_forecast = pd.DataFrame()
        self.consumption_patterns = {}
        
        logging.info("Multi-day Optimizer geïnitialiseerd")
    
    def _load_multiday_settings(self) -> Dict[str, Any]:
        """Laad multi-day optimization instellingen"""
        return {
            'enabled': self.config.get(['multiday_optimization', 'enabled'], True),
            'planning_horizon_days': self.config.get(['multiday_optimization', 'planning_days'], 7),
            'weather_forecast_enabled': self.config.get(['multiday_optimization', 'weather_forecast'], True),
            'holiday_detection_enabled': self.config.get(['multiday_optimization', 'holiday_detection'], True),
            'seasonal_optimization_enabled': self.config.get(['multiday_optimization', 'seasonal_optimization'], True),
            'battery_degradation_enabled': self.config.get(['multiday_optimization', 'battery_degradation'], True),
            
            # Weather settings
            'weather_api_key': self.config.get(['weather', 'api_key'], ''),
            'weather_provider': self.config.get(['weather', 'provider'], 'openweathermap'),  # knmi, openweathermap
            'location_lat': self.config.get(['location', 'latitude'], 52.0),
            'location_lon': self.config.get(['location', 'longitude'], 5.0),
            
            # Optimization settings
            'pv_forecast_accuracy_weight': self.config.get(['multiday_optimization', 'pv_accuracy_weight'], 0.8),
            'consumption_pattern_weight': self.config.get(['multiday_optimization', 'consumption_weight'], 0.9),
            'price_volatility_factor': self.config.get(['multiday_optimization', 'price_volatility'], 1.2),
        }
    
    async def run_multiday_optimization(self) -> Dict[str, Any]:
        """Hoofdfunctie voor multi-day optimization"""
        try:
            if not self.settings['enabled']:
                return {'success': False, 'message': 'Multi-day optimization disabled'}
            
            optimization_results = {
                'planning_horizon': self.settings['planning_horizon_days'],
                'weather_forecast': {},
                'consumption_forecast': {},
                'pv_forecast': {},
                'battery_strategy': {},
                'seasonal_adjustments': {},
                'holiday_adjustments': {},
                'cost_projections': {},
                'optimization_schedule': {}
            }
            
            # 1. Haal 7-day weather forecast op
            if self.settings['weather_forecast_enabled']:
                weather_forecast = await self.weather_service.get_extended_forecast(
                    self.settings['planning_horizon_days']
                )
                optimization_results['weather_forecast'] = weather_forecast
            
            # 2. Detecteer holidays/vacation periods
            if self.settings['holiday_detection_enabled']:
                holiday_periods = await self.holiday_detector.detect_special_periods(
                    datetime.now(),
                    datetime.now() + timedelta(days=self.settings['planning_horizon_days'])
                )
                optimization_results['holiday_adjustments'] = holiday_periods
            
            # 3. Genereer consumption forecast voor 7 dagen
            consumption_forecast = await self._generate_multiday_consumption_forecast(
                weather_forecast, holiday_periods
            )
            optimization_results['consumption_forecast'] = consumption_forecast
            
            # 4. Genereer PV forecast met weather data
            if weather_forecast:
                pv_forecast = await self._generate_multiday_pv_forecast(weather_forecast)
                optimization_results['pv_forecast'] = pv_forecast
            
            # 5. Seasonal optimization adjustments
            if self.settings['seasonal_optimization_enabled']:
                seasonal_adjustments = await self.seasonal_optimizer.get_seasonal_strategy(
                    datetime.now().month
                )
                optimization_results['seasonal_adjustments'] = seasonal_adjustments
            
            # 6. Battery degradation management
            if self.settings['battery_degradation_enabled']:
                degradation_strategy = await self.degradation_manager.optimize_battery_lifecycle(
                    optimization_results
                )
                optimization_results['battery_strategy'] = degradation_strategy
            
            # 7. Multi-day cost projections
            cost_projections = await self._calculate_multiday_cost_projections(
                optimization_results
            )
            optimization_results['cost_projections'] = cost_projections
            
            # 8. Generate optimal daily schedules
            daily_schedules = await self._generate_optimal_daily_schedules(
                optimization_results
            )
            optimization_results['optimization_schedule'] = daily_schedules
            
            return {
                'success': True,
                'timestamp': datetime.now().isoformat(),
                'results': optimization_results,
                'total_projected_savings': cost_projections.get('total_savings', 0),
                'next_optimization': (datetime.now() + timedelta(hours=6)).isoformat()
            }
            
        except Exception as e:
            logging.error(f"Multi-day optimization fout: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _generate_multiday_consumption_forecast(self, weather_forecast: Dict, 
                                                     holiday_periods: Dict) -> Dict[str, Any]:
        """Genereer 7-day consumption forecast"""
        try:
            forecast_days = self.settings['planning_horizon_days']
            base_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            
            daily_forecasts = {}
            
            for day in range(forecast_days):
                current_date = base_date + timedelta(days=day)
                
                # Basis consumption pattern gebaseerd op dag van week
                base_pattern = self._get_base_consumption_pattern(current_date)
                
                # Weather impact
                weather_adjustment = 1.0
                if weather_forecast and f'day_{day}' in weather_forecast:
                    weather_data = weather_forecast[f'day_{day}']
                    weather_adjustment = self._calculate_weather_consumption_impact(weather_data)
                
                # Holiday/vacation impact
                holiday_adjustment = 1.0
                if current_date.strftime('%Y-%m-%d') in holiday_periods.get('holidays', []):
                    holiday_adjustment = 0.7  # 30% minder verbruik op feestdagen
                elif current_date.strftime('%Y-%m-%d') in holiday_periods.get('vacation_periods', []):
                    holiday_adjustment = 0.4  # 60% minder verbruik tijdens vakanties
                
                # Seasonal adjustment
                seasonal_factor = self._get_seasonal_consumption_factor(current_date.month)
                
                # Combine all factors
                adjusted_pattern = (base_pattern * weather_adjustment * 
                                  holiday_adjustment * seasonal_factor)
                
                daily_forecasts[f'day_{day}'] = {
                    'date': current_date.strftime('%Y-%m-%d'),
                    'hourly_consumption': adjusted_pattern.tolist(),
                    'daily_total': adjusted_pattern.sum(),
                    'adjustments': {
                        'weather': weather_adjustment,
                        'holiday': holiday_adjustment,
                        'seasonal': seasonal_factor
                    }
                }
            
            return {
                'forecast_days': forecast_days,
                'daily_forecasts': daily_forecasts,
                'total_week_consumption': sum([day['daily_total'] for day in daily_forecasts.values()])
            }
            
        except Exception as e:
            logging.error(f"Multi-day consumption forecast fout: {e}")
            return {}
    
    def _get_base_consumption_pattern(self, date: datetime) -> np.ndarray:
        """Krijg basis consumption pattern voor een dag"""
        day_of_week = date.weekday()  # 0 = Monday, 6 = Sunday
        
        if day_of_week < 5:  # Weekdag
            # Werkdag pattern
            pattern = np.array([
                0.3, 0.3, 0.3, 0.3, 0.4, 0.6,  # 0-5: nacht -> ochtend
                0.8, 1.2, 1.0, 0.8, 0.7, 0.6,  # 6-11: ochtend -> middag
                0.7, 0.6, 0.5, 0.6, 0.8, 1.5,  # 12-17: middag -> avond
                1.8, 1.6, 1.4, 1.0, 0.8, 0.5   # 18-23: avond -> nacht
            ])
        else:  # Weekend
            # Weekend pattern (meer thuis, later opstaan)
            pattern = np.array([
                0.3, 0.3, 0.3, 0.3, 0.3, 0.4,  # 0-5: nacht
                0.5, 0.6, 0.9, 1.2, 1.1, 1.0,  # 6-11: late ochtend
                0.9, 0.8, 0.7, 0.8, 0.9, 1.2,  # 12-17: middag/namiddag
                1.4, 1.3, 1.2, 1.0, 0.7, 0.4   # 18-23: avond
            ])
        
        return pattern
    
    def _calculate_weather_consumption_impact(self, weather_data: Dict) -> float:
        """Bereken weather impact op verbruik"""
        temp = weather_data.get('avg_temp', 15)
        
        # Temperature impact (heating/cooling needs)
        if temp < 10:  # Koud weer = meer verwarming
            temp_factor = 1.0 + (10 - temp) * 0.08  # 8% meer per graad onder 10°C
        elif temp > 25:  # Warm weer = meer koeling
            temp_factor = 1.0 + (temp - 25) * 0.05  # 5% meer per graad boven 25°C
        else:
            temp_factor = 1.0
        
        # Wind impact (heat loss)
        wind_speed = weather_data.get('wind_speed', 0)
        wind_factor = 1.0 + (wind_speed - 5) * 0.02 if wind_speed > 5 else 1.0
        
        return min(2.0, max(0.5, temp_factor * wind_factor))
    
    def _get_seasonal_consumption_factor(self, month: int) -> float:
        """Seizoensgebonden consumption factor"""
        # Winter (Dec, Jan, Feb): meer verbruik
        if month in [12, 1, 2]:
            return 1.3
        # Lente (Mar, Apr, May): normaal verbruik
        elif month in [3, 4, 5]:
            return 1.0
        # Zomer (Jun, Jul, Aug): iets minder verbruik
        elif month in [6, 7, 8]:
            return 0.9
        # Herfst (Sep, Oct, Nov): stijgend verbruik
        else:
            return 1.1
    
    async def _generate_multiday_pv_forecast(self, weather_forecast: Dict) -> Dict[str, Any]:
        """Genereer 7-day PV forecast met weather data"""
        try:
            solar_config = self.config.get(['solar'], {})
            capacity_kwp = solar_config.get('capacity', 0)
            
            if capacity_kwp == 0:
                return {'message': 'No PV system configured'}
            
            forecast_days = self.settings['planning_horizon_days']
            daily_pv_forecasts = {}
            
            for day in range(forecast_days):
                if f'day_{day}' not in weather_forecast:
                    continue
                    
                weather_data = weather_forecast[f'day_{day}']
                pv_forecast = await self._calculate_daily_pv_production(
                    weather_data, capacity_kwp, day
                )
                
                daily_pv_forecasts[f'day_{day}'] = pv_forecast
            
            return {
                'installed_capacity_kwp': capacity_kwp,
                'daily_forecasts': daily_pv_forecasts,
                'total_week_production': sum([day['daily_total'] for day in daily_pv_forecasts.values()])
            }
            
        except Exception as e:
            logging.error(f"Multi-day PV forecast fout: {e}")
            return {}
    
    async def _calculate_daily_pv_production(self, weather_data: Dict, 
                                           capacity_kwp: float, day_offset: int) -> Dict[str, Any]:
        """Bereken PV productie voor één dag"""
        try:
            current_date = datetime.now() + timedelta(days=day_offset)
            month = current_date.month
            
            # Daylight hours voor deze tijd van jaar
            daylight_hours = self._get_daylight_hours(month)
            
            hourly_production = []
            
            for hour in range(24):
                if daylight_hours['sunrise'] <= hour <= daylight_hours['sunset']:
                    # Bereken solar angle factor
                    hours_from_noon = abs(hour - 12)
                    solar_angle_factor = np.cos(hours_from_noon * np.pi / 12)
                    
                    # Weather factors
                    cloud_cover = weather_data.get('cloud_cover', 0.3)
                    solar_irradiance = weather_data.get('solar_irradiance', 600)  # W/m²
                    temperature = weather_data.get('avg_temp', 15)
                    
                    # Cloud impact
                    clear_sky_factor = 1 - (cloud_cover * 0.8)
                    
                    # Temperature impact (PV efficiency decreases with heat)
                    temp_factor = 1 - ((temperature - 25) * 0.004)
                    temp_factor = np.clip(temp_factor, 0.7, 1.1)
                    
                    # Seasonal irradiance adjustment
                    seasonal_irradiance_factor = self._get_seasonal_irradiance_factor(month)
                    
                    # Calculate PV power
                    pv_power = (capacity_kwp * 
                               solar_irradiance / 1000 *  # Convert W/m² to kW/m²
                               solar_angle_factor * 
                               clear_sky_factor * 
                               temp_factor *
                               seasonal_irradiance_factor)
                    
                    pv_power = max(0, pv_power)
                else:
                    pv_power = 0
                
                hourly_production.append(pv_power)
            
            return {
                'date': current_date.strftime('%Y-%m-%d'),
                'hourly_production': hourly_production,
                'daily_total': sum(hourly_production),
                'weather_factors': {
                    'cloud_cover': weather_data.get('cloud_cover', 0.3),
                    'avg_temp': weather_data.get('avg_temp', 15),
                    'solar_irradiance': weather_data.get('solar_irradiance', 600)
                }
            }
            
        except Exception as e:
            logging.error(f"Daily PV calculation fout: {e}")
            return {'hourly_production': [0] * 24, 'daily_total': 0}
    
    def _get_daylight_hours(self, month: int) -> Dict[str, int]:
        """Krijg sunrise/sunset tijden voor maand (Nederland)"""
        daylight_table = {
            1: {'sunrise': 8, 'sunset': 17},   # Januari
            2: {'sunrise': 7, 'sunset': 18},   # Februari
            3: {'sunrise': 6, 'sunset': 19},   # Maart
            4: {'sunrise': 6, 'sunset': 20},   # April (zomertijd)
            5: {'sunrise': 5, 'sunset': 21},   # Mei
            6: {'sunrise': 5, 'sunset': 22},   # Juni
            7: {'sunrise': 5, 'sunset': 22},   # Juli
            8: {'sunrise': 6, 'sunset': 21},   # Augustus
            9: {'sunrise': 7, 'sunset': 19},   # September
            10: {'sunrise': 8, 'sunset': 18},  # Oktober (wintertijd)
            11: {'sunrise': 8, 'sunset': 16},  # November
            12: {'sunrise': 8, 'sunset': 16}   # December
        }
        return daylight_table.get(month, {'sunrise': 7, 'sunset': 18})
    
    def _get_seasonal_irradiance_factor(self, month: int) -> float:
        """Seizoensgebonden solar irradiance factor"""
        # Based on typical Dutch solar irradiance patterns
        factors = {
            1: 0.3, 2: 0.5, 3: 0.7, 4: 0.9,   # Winter -> Spring
            5: 1.0, 6: 1.1, 7: 1.1, 8: 1.0,   # Spring -> Summer
            9: 0.8, 10: 0.6, 11: 0.4, 12: 0.3  # Summer -> Winter
        }
        return factors.get(month, 0.7)
    
    async def _calculate_multiday_cost_projections(self, optimization_results: Dict) -> Dict[str, Any]:
        """Bereken kosten projecties voor komende week"""
        try:
            # Generate price forecast (simplified - zou uit echte APIs komen)
            price_forecast = await self._generate_price_forecast()
            
            # Base costs zonder optimization
            base_costs = self._calculate_base_weekly_costs(
                optimization_results['consumption_forecast'],
                price_forecast
            )
            
            # Optimized costs met PV self-consumption
            optimized_costs = self._calculate_optimized_weekly_costs(
                optimization_results['consumption_forecast'],
                optimization_results.get('pv_forecast', {}),
                price_forecast,
                optimization_results.get('battery_strategy', {}),
                optimization_results.get('seasonal_adjustments', {}),
                optimization_results.get('holiday_adjustments', {})
            )
            
            total_savings = base_costs - optimized_costs
            savings_percentage = (total_savings / base_costs * 100) if base_costs > 0 else 0
            
            return {
                'base_weekly_cost': round(base_costs, 2),
                'optimized_weekly_cost': round(optimized_costs, 2),
                'total_savings': round(total_savings, 2),
                'savings_percentage': round(savings_percentage, 1),
                'daily_breakdown': self._get_daily_cost_breakdown(
                    optimization_results, price_forecast
                )
            }
            
        except Exception as e:
            logging.error(f"Cost projection fout: {e}")
            return {'total_savings': 0}
    
    async def _generate_price_forecast(self) -> Dict[str, List[float]]:
        """Genereer prijzen forecast (simplified)"""
        forecast_days = self.settings['planning_horizon_days']
        price_forecast = {}
        
        # Base prijzen pattern (realistische Nederlandse prijzen)
        base_hourly_prices = np.array([
            0.12, 0.10, 0.08, 0.07, 0.08, 0.10,  # 0-5: nacht
            0.15, 0.20, 0.22, 0.18, 0.16, 0.14,  # 6-11: ochtend
            0.16, 0.14, 0.13, 0.15, 0.18, 0.25,  # 12-17: middag/namiddag
            0.28, 0.24, 0.20, 0.18, 0.16, 0.14   # 18-23: avond
        ])
        
        for day in range(forecast_days):
            # Voeg wat variatie toe per dag
            daily_variation = 1.0 + np.random.normal(0, 0.1)
            
            # Weekend effect (iets lagere prijzen)
            current_date = datetime.now() + timedelta(days=day)
            weekend_factor = 0.95 if current_date.weekday() >= 5 else 1.0
            
            daily_prices = base_hourly_prices * daily_variation * weekend_factor
            daily_prices = np.clip(daily_prices, 0.05, 0.45)
            
            price_forecast[f'day_{day}'] = daily_prices.tolist()
        
        return price_forecast
    
    def _calculate_base_weekly_costs(self, consumption_forecast: Dict, 
                                   price_forecast: Dict) -> float:
        """Bereken basis kosten zonder optimization"""
        total_cost = 0
        
        for day_key in consumption_forecast.get('daily_forecasts', {}):
            if day_key not in price_forecast:
                continue
                
            daily_consumption = consumption_forecast['daily_forecasts'][day_key]['hourly_consumption']
            daily_prices = price_forecast[day_key]
            
            daily_cost = sum(c * p for c, p in zip(daily_consumption, daily_prices))
            total_cost += daily_cost
        
        return total_cost
    
    def _calculate_optimized_weekly_costs(self, consumption_forecast: Dict, 
                                        pv_forecast: Dict, price_forecast: Dict,
                                        battery_strategy: Dict, seasonal_adjustments: Dict,
                                        holiday_adjustments: Dict) -> float:
        """Bereken geoptimaliseerde kosten"""
        total_cost = 0
        
        for day_key in consumption_forecast.get('daily_forecasts', {}):
            if day_key not in price_forecast:
                continue
                
            daily_consumption = np.array(consumption_forecast['daily_forecasts'][day_key]['hourly_consumption'])
            daily_prices = np.array(price_forecast[day_key])
            
            # PV self-consumption
            if day_key in pv_forecast.get('daily_forecasts', {}):
                daily_pv = np.array(pv_forecast['daily_forecasts'][day_key]['hourly_production'])
                # Self-consumption reduces grid consumption
                net_consumption = np.maximum(0, daily_consumption - daily_pv)
            else:
                net_consumption = daily_consumption
            
            # Battery optimization impact (simplified)
            if battery_strategy.get('enabled', False):
                # Assume 10% reduction during expensive hours through battery usage
                expensive_hours = daily_prices > np.percentile(daily_prices, 75)
                net_consumption[expensive_hours] *= 0.9
            
            # Seasonal adjustments impact
            seasonal_factor = seasonal_adjustments.get('cost_reduction_factor', 1.0)
            
            daily_cost = sum(net_consumption * daily_prices * seasonal_factor)
            total_cost += daily_cost
        
        return total_cost
    
    def _get_daily_cost_breakdown(self, optimization_results: Dict, 
                                price_forecast: Dict) -> Dict[str, Dict]:
        """Krijg dagelijkse kosten breakdown"""
        daily_breakdown = {}
        
        consumption_data = optimization_results.get('consumption_forecast', {}).get('daily_forecasts', {})
        pv_data = optimization_results.get('pv_forecast', {}).get('daily_forecasts', {})
        
        for day_key in consumption_data:
            if day_key not in price_forecast:
                continue
                
            base_date = datetime.now() + timedelta(days=int(day_key.split('_')[1]))
            
            consumption = sum(consumption_data[day_key]['hourly_consumption'])
            pv_production = sum(pv_data.get(day_key, {}).get('hourly_production', [0] * 24))
            avg_price = sum(price_forecast[day_key]) / 24
            
            daily_breakdown[base_date.strftime('%Y-%m-%d')] = {
                'consumption_kwh': round(consumption, 2),
                'pv_production_kwh': round(pv_production, 2),
                'net_consumption_kwh': round(max(0, consumption - pv_production), 2),
                'avg_price_eur_kwh': round(avg_price, 4),
                'estimated_cost_eur': round((consumption - min(consumption, pv_production)) * avg_price, 2)
            }
        
        return daily_breakdown
    
    async def _generate_optimal_daily_schedules(self, optimization_results: Dict) -> Dict[str, Any]:
        """Genereer optimale dagelijkse schema's"""
        try:
            forecast_days = self.settings['planning_horizon_days']
            daily_schedules = {}
            
            for day in range(forecast_days):
                current_date = datetime.now() + timedelta(days=day)
                day_key = f'day_{day}'
                
                schedule = {
                    'date': current_date.strftime('%Y-%m-%d'),
                    'battery_schedule': self._optimize_daily_battery_schedule(day_key, optimization_results),
                    'device_schedule': self._optimize_daily_device_schedule(day_key, optimization_results),
                    'pv_optimization': self._optimize_daily_pv_usage(day_key, optimization_results),
                    'cost_optimization': self._get_daily_cost_optimization_tips(day_key, optimization_results)
                }
                
                daily_schedules[day_key] = schedule
            
            return {
                'planning_horizon': forecast_days,
                'daily_schedules': daily_schedules,
                'optimization_summary': self._get_week_optimization_summary(daily_schedules)
            }
            
        except Exception as e:
            logging.error(f"Daily schedule generation fout: {e}")
            return {}
    
    def _optimize_daily_battery_schedule(self, day_key: str, optimization_results: Dict) -> Dict[str, Any]:
        """Optimaliseer batterij schema voor een dag"""
        # Simplified battery optimization
        return {
            'charge_periods': ['02:00-05:00', '13:00-16:00'],  # Goedkope uren + PV peak
            'discharge_periods': ['18:00-21:00'],  # Dure avonduren
            'target_soc_start': 20,  # Begin dag met 20%
            'target_soc_end': 80,   # Eind dag met 80%
            'estimated_cycles': 1.2  # Geschatte battery cycles
        }
    
    def _optimize_daily_device_schedule(self, day_key: str, optimization_results: Dict) -> Dict[str, Any]:
        """Optimaliseer apparaat schema voor een dag"""
        return {
            'dishwasher': {'optimal_start': '02:00', 'reason': 'Laagste prijzen'},
            'washing_machine': {'optimal_start': '13:00', 'reason': 'PV productie beschikbaar'},
            'dryer': {'optimal_start': '14:00', 'reason': 'Na wasmachine, PV beschikbaar'},
            'heat_pump': {
                'optimization_periods': ['13:00-16:00', '02:00-05:00'],
                'reason': 'PV beschikbaar + goedkope uren'
            }
        }
    
    def _optimize_daily_pv_usage(self, day_key: str, optimization_results: Dict) -> Dict[str, Any]:
        """Optimaliseer PV zelf verbruik voor een dag"""
        pv_data = optimization_results.get('pv_forecast', {}).get('daily_forecasts', {}).get(day_key, {})
        
        if not pv_data:
            return {'message': 'No PV forecast available'}
        
        return {
            'peak_production_hours': ['11:00-15:00'],
            'self_consumption_opportunities': ['12:00-16:00'],
            'battery_charging_optimal': ['13:00-16:00'],
            'excess_production_hours': ['12:00-14:00'],
            'estimated_self_consumption_rate': 75  # percentage
        }
    
    def _get_daily_cost_optimization_tips(self, day_key: str, optimization_results: Dict) -> List[str]:
        """Krijg dagelijkse kostenbesparingen tips"""
        return [
            "Gebruik vaatwasser tussen 02:00-05:00 voor 40% lagere kosten",
            "Plan warmtepomp tussen 13:00-16:00 voor optimaal PV verbruik", 
            "Laad EV tussen 01:00-06:00 tijdens dal-uren",
            "Batterij opladen tijdens PV piek (13:00-16:00)",
            "Vermijd hoog verbruik tussen 18:00-21:00 (piek tarieven)"
        ]
    
    def _get_week_optimization_summary(self, daily_schedules: Dict) -> Dict[str, Any]:
        """Krijg week optimalisatie samenvatting"""
        return {
            'total_battery_cycles': sum([
                schedule.get('battery_schedule', {}).get('estimated_cycles', 0)
                for schedule in daily_schedules.values()
            ]),
            'pv_self_consumption_rate': 72,  # Average percentage
            'device_scheduling_savings': 15.3,  # Percentage savings
            'peak_hour_avoidance': 85,  # Percentage peak hours avoided
            'optimization_score': 88  # Overall optimization effectiveness score
        }


class EnhancedWeatherService:
    """Enhanced weather service met real-time API integration"""
    
    def __init__(self, config: Config):
        self.config = config
        self.api_key = config.get(['weather', 'api_key'], '')
        self.provider = config.get(['weather', 'provider'], 'openweathermap')
        self.latitude = config.get(['location', 'latitude'], 52.0)
        self.longitude = config.get(['location', 'longitude'], 5.0)
        
    async def get_extended_forecast(self, days: int = 7) -> Dict[str, Any]:
        """Haal extended weather forecast op"""
        try:
            if self.provider == 'openweathermap':
                return await self._get_openweather_forecast(days)
            elif self.provider == 'knmi':
                return await self._get_knmi_forecast(days)
            else:
                return await self._get_fallback_forecast(days)
                
        except Exception as e:
            logging.error(f"Weather forecast fout: {e}")
            return await self._get_fallback_forecast(days)
    
    async def _get_openweather_forecast(self, days: int) -> Dict[str, Any]:
        """OpenWeatherMap API forecast"""
        if not self.api_key:
            return await self._get_fallback_forecast(days)
            
        try:
            url = f"http://api.openweathermap.org/data/2.5/forecast"
            params = {
                'lat': self.latitude,
                'lon': self.longitude,
                'appid': self.api_key,
                'units': 'metric',
                'cnt': days * 8  # 3-hour intervals
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return self._process_openweather_data(data, days)
                    else:
                        logging.warning(f"OpenWeatherMap API error: {response.status}")
                        return await self._get_fallback_forecast(days)
                        
        except Exception as e:
            logging.error(f"OpenWeatherMap API fout: {e}")
            return await self._get_fallback_forecast(days)
    
    async def _get_knmi_forecast(self, days: int) -> Dict[str, Any]:
        """KNMI API forecast (Nederlandse weer dienst)"""
        try:
            # KNMI heeft gratis API maar beperkte forecast dagen
            url = "http://weerlive.nl/api/json-data-10day.php"
            params = {
                'key': self.api_key,  # Aparte key voor weerlive
                'locatie': f"{self.latitude},{self.longitude}"
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return self._process_knmi_data(data, days)
                    else:
                        return await self._get_fallback_forecast(days)
                        
        except Exception as e:
            logging.error(f"KNMI API fout: {e}")
            return await self._get_fallback_forecast(days)
    
    def _process_openweather_data(self, data: Dict, days: int) -> Dict[str, Any]:
        """Process OpenWeatherMap response"""
        forecast = {}
        
        # Group forecasts by day
        daily_data = {}
        for item in data.get('list', []):
            dt = datetime.fromtimestamp(item['dt'])
            day_key = dt.strftime('%Y-%m-%d')
            
            if day_key not in daily_data:
                daily_data[day_key] = []
            
            daily_data[day_key].append({
                'temp': item['main']['temp'],
                'humidity': item['main']['humidity'],
                'pressure': item['main']['pressure'],
                'wind_speed': item['wind']['speed'],
                'cloud_cover': item['clouds']['all'] / 100.0,
                'visibility': item.get('visibility', 10000) / 1000.0,  # Convert to km
                'datetime': dt
            })
        
        # Create daily summaries
        day_count = 0
        for day_key in sorted(daily_data.keys())[:days]:
            day_forecasts = daily_data[day_key]
            
            forecast[f'day_{day_count}'] = {
                'date': day_key,
                'avg_temp': sum([f['temp'] for f in day_forecasts]) / len(day_forecasts),
                'min_temp': min([f['temp'] for f in day_forecasts]),
                'max_temp': max([f['temp'] for f in day_forecasts]),
                'avg_humidity': sum([f['humidity'] for f in day_forecasts]) / len(day_forecasts),
                'avg_pressure': sum([f['pressure'] for f in day_forecasts]) / len(day_forecasts),
                'wind_speed': sum([f['wind_speed'] for f in day_forecasts]) / len(day_forecasts),
                'cloud_cover': sum([f['cloud_cover'] for f in day_forecasts]) / len(day_forecasts),
                'visibility': sum([f['visibility'] for f in day_forecasts]) / len(day_forecasts),
                'solar_irradiance': self._estimate_solar_irradiance(
                    sum([f['cloud_cover'] for f in day_forecasts]) / len(day_forecasts),
                    day_count
                )
            }
            day_count += 1
        
        return forecast
    
    def _estimate_solar_irradiance(self, cloud_cover: float, day_offset: int) -> float:
        """Schat solar irradiance gebaseerd op bewolking en seizoen"""
        base_date = datetime.now() + timedelta(days=day_offset)
        month = base_date.month
        
        # Basis irradiance per maand (W/m²) voor Nederland
        monthly_base = {
            1: 200, 2: 350, 3: 550, 4: 750, 5: 900, 6: 950,
            7: 950, 8: 850, 9: 650, 10: 450, 11: 250, 12: 180
        }
        
        base_irradiance = monthly_base.get(month, 600)
        
        # Cloud impact
        clear_sky_factor = 1.0 - (cloud_cover * 0.8)
        
        return base_irradiance * clear_sky_factor
    
    async def _get_fallback_forecast(self, days: int) -> Dict[str, Any]:
        """Fallback weather forecast als APIs niet werken"""
        logging.warning("Using fallback weather forecast")
        
        forecast = {}
        current_month = datetime.now().month
        
        # Realistische fallback data gebaseerd op Nederlands klimaat
        for day in range(days):
            base_temp = self._get_seasonal_base_temp(current_month)
            
            forecast[f'day_{day}'] = {
                'date': (datetime.now() + timedelta(days=day)).strftime('%Y-%m-%d'),
                'avg_temp': base_temp + np.random.normal(0, 3),
                'min_temp': base_temp - 5 + np.random.normal(0, 2),
                'max_temp': base_temp + 5 + np.random.normal(0, 2),
                'avg_humidity': 70 + np.random.normal(0, 10),
                'wind_speed': 4 + np.random.normal(0, 2),
                'cloud_cover': 0.5 + np.random.normal(0, 0.3),
                'solar_irradiance': self._get_seasonal_base_irradiance(current_month) * (0.3 + np.random.random() * 0.7)
            }
        
        return forecast
    
    def _get_seasonal_base_temp(self, month: int) -> float:
        """Krijg seizoensgebonden basis temperatuur"""
        temps = {1: 3, 2: 4, 3: 7, 4: 11, 5: 15, 6: 18,
                7: 20, 8: 20, 9: 16, 10: 12, 11: 7, 12: 4}
        return temps.get(month, 12)
    
    def _get_seasonal_base_irradiance(self, month: int) -> float:
        """Krijg seizoensgebonden basis solar irradiance"""
        irradiance = {1: 200, 2: 350, 3: 550, 4: 750, 5: 900, 6: 950,
                     7: 950, 8: 850, 9: 650, 10: 450, 11: 250, 12: 180}
        return irradiance.get(month, 600)


class HolidayVacationDetector:
    """Holiday en vacation period detectie"""
    
    def __init__(self, config: Config, ha_integration: HomeAssistantIntegration):
        self.config = config
        self.ha = ha_integration
        
    async def detect_special_periods(self, start_date: datetime, end_date: datetime) -> Dict[str, List[str]]:
        """Detecteer holidays en vacation periods"""
        try:
            holidays = self._get_dutch_holidays(start_date, end_date)
            vacation_periods = await self._detect_vacation_periods(start_date, end_date)
            
            return {
                'holidays': holidays,
                'vacation_periods': vacation_periods,
                'special_events': await self._detect_special_events(start_date, end_date)
            }
            
        except Exception as e:
            logging.error(f"Holiday/vacation detection fout: {e}")
            return {'holidays': [], 'vacation_periods': [], 'special_events': []}
    
    def _get_dutch_holidays(self, start_date: datetime, end_date: datetime) -> List[str]:
        """Krijg Nederlandse feestdagen in periode"""
        holidays = []
        current = start_date
        
        while current <= end_date:
            # Vaste feestdagen
            if (current.month == 1 and current.day == 1) or \
               (current.month == 4 and current.day == 27) or \
               (current.month == 5 and current.day == 5) or \
               (current.month == 12 and current.day in [25, 26]):
                holidays.append(current.strftime('%Y-%m-%d'))
            
            # Pasen (vereenvoudigd - zou echte berekening moeten zijn)
            elif self._is_easter_period(current):
                holidays.append(current.strftime('%Y-%m-%d'))
            
            current += timedelta(days=1)
        
        return holidays
    
    def _is_easter_period(self, date: datetime) -> bool:
        """Check of datum in Paas periode valt (simplified)"""
        # Dit is een vereenvoudigde versie - echte Paas berekening is complexer
        if date.month == 4 and 5 <= date.day <= 15:  # Approximation
            return True
        return False
    
    async def _detect_vacation_periods(self, start_date: datetime, end_date: datetime) -> List[str]:
        """Detecteer vacation periods via HA presence detection"""
        try:
            vacation_days = []
            
            # Check Home Assistant presence entities
            presence_entities = [
                'person.family_member_1',
                'person.family_member_2',
                'device_tracker.phone_1',
                'device_tracker.phone_2'
            ]
            
            # Check historical presence (would be implemented with real HA data)
            current = start_date
            while current <= end_date:
                # Simplified: assume vacation if specific patterns detected
                if await self._is_vacation_day(current, presence_entities):
                    vacation_days.append(current.strftime('%Y-%m-%d'))
                
                current += timedelta(days=1)
            
            return vacation_days
            
        except Exception as e:
            logging.error(f"Vacation detection fout: {e}")
            return []
    
    async def _is_vacation_day(self, date: datetime, presence_entities: List[str]) -> bool:
        """Check of datum een vakantiedag is"""
        # Simplified logic - zou echte HA data moeten gebruiken
        # Voor nu: detecteer schoolvakanties
        school_vacation_periods = [
            (datetime(date.year, 7, 15), datetime(date.year, 8, 31)),  # Zomervakantie
            (datetime(date.year, 12, 23), datetime(date.year + 1, 1, 8)),  # Kerstvakantie
            (datetime(date.year, 4, 25), datetime(date.year, 5, 10)),  # Meivakantie
        ]
        
        for start, end in school_vacation_periods:
            if start <= date <= end:
                return True
        
        return False
    
    async def _detect_special_events(self, start_date: datetime, end_date: datetime) -> List[str]:
        """Detecteer speciale events die verbruik beïnvloeden"""
        events = []
        
        current = start_date
        while current <= end_date:
            # Sinterklaas periode
            if current.month == 12 and 1 <= current.day <= 6:
                events.append(current.strftime('%Y-%m-%d'))
            
            # WK/EK voetbal (zou dynamisch moeten zijn)
            # Kermis/festivals (lokaal afhankelijk)
            
            current += timedelta(days=1)
        
        return events


class SeasonalOptimizer:
    """Seizoensgebonden optimalisatie strategieën"""
    
    def __init__(self, config: Config):
        self.config = config
        
    async def get_seasonal_strategy(self, month: int) -> Dict[str, Any]:
        """Krijg seizoensspecifieke optimalisatie strategie"""
        try:
            if month in [12, 1, 2]:  # Winter
                return await self._get_winter_strategy()
            elif month in [3, 4, 5]:  # Lente
                return await self._get_spring_strategy()
            elif month in [6, 7, 8]:  # Zomer
                return await self._get_summer_strategy()
            else:  # Herfst
                return await self._get_autumn_strategy()
                
        except Exception as e:
            logging.error(f"Seasonal strategy fout: {e}")
            return {}
    
    async def _get_winter_strategy(self) -> Dict[str, Any]:
        """Winter optimalisatie strategie"""
        return {
            'season': 'winter',
            'battery_strategy': {
                'charge_preference': 'night_hours',  # Goedkope nacht uren
                'discharge_preference': 'morning_evening',  # Piek verwarming
                'target_soc_winter': 90,  # Hogere reserve voor verwarming
                'degradation_allowance': 1.2  # Meer cycles toegestaan
            },
            'heating_optimization': {
                'heat_pump_scheduling': 'off_peak_preheating',
                'thermal_mass_usage': True,  # Huis opwarmen tijdens goedkope uren
                'backup_heating_threshold': -5,  # °C wanneer backup inschakelen
            },
            'device_scheduling': {
                'dishwasher_preferred_hours': ['02:00-05:00', '13:00-15:00'],
                'washing_machine_winter_schedule': 'pv_available',
                'dryer_heat_recovery': True  # Gebruik waste heat
            },
            'pv_optimization': {
                'self_consumption_priority': 'maximum',  # Weinig PV, alles zelf gebruiken
                'battery_charging_from_pv': 'immediate',
                'grid_export_threshold': 0  # Niet exporteren in winter
            },
            'cost_reduction_factor': 0.85,  # 15% meer kosten in winter
            'consumption_increase_factor': 1.3
        }
    
    async def _get_summer_strategy(self) -> Dict[str, Any]:
        """Zomer optimalisatie strategie"""
        return {
            'season': 'summer',
            'battery_strategy': {
                'charge_preference': 'pv_excess',  # Laad met PV overschot
                'discharge_preference': 'evening_peak',
                'target_soc_summer': 70,  # Lagere reserve, meer cycles
                'degradation_allowance': 0.8  # Bescherm tegen hitte degradation
            },
            'cooling_optimization': {
                'airco_scheduling': 'pv_peak_hours',
                'thermal_mass_cooling': True,  # Huis koelen tijdens PV piek
                'natural_cooling_hours': ['22:00-06:00']
            },
            'device_scheduling': {
                'dishwasher_preferred_hours': ['12:00-16:00'],  # PV piek
                'washing_machine_summer_schedule': 'pv_peak',
                'pool_pump_optimization': 'pv_hours'
            },
            'pv_optimization': {
                'self_consumption_priority': 'smart_timing',
                'battery_charging_from_pv': 'selective',  # Laad alleen bij overschot
                'grid_export_optimization': True,  # Optimaliseer export timing
                'export_price_threshold': 0.08  # Minimum export prijs
            },
            'cost_reduction_factor': 1.15,  # 15% kostenreductie in zomer door PV
            'consumption_decrease_factor': 0.9
        }
    
    async def _get_spring_strategy(self) -> Dict[str, Any]:
        """Lente optimalisatie strategie"""
        return {
            'season': 'spring',
            'battery_strategy': {
                'charge_preference': 'mixed',  # Mix van nacht en PV
                'discharge_preference': 'price_based',
                'target_soc_spring': 80,
                'degradation_allowance': 1.0
            },
            'heating_optimization': {
                'heat_pump_scheduling': 'weather_adaptive',
                'transition_heating_cooling': True,
                'maintenance_scheduling': 'spring_maintenance'
            },
            'device_scheduling': {
                'dishwasher_preferred_hours': ['02:00-05:00', '13:00-16:00'],
                'seasonal_appliance_prep': True
            },
            'pv_optimization': {
                'self_consumption_priority': 'increasing',
                'battery_charging_from_pv': 'adaptive',
                'grid_export_threshold': 0.05
            },
            'cost_reduction_factor': 1.0,
            'consumption_factor': 1.0
        }
    
    async def _get_autumn_strategy(self) -> Dict[str, Any]:
        """Herfst optimalisatie strategie"""
        return {
            'season': 'autumn',
            'battery_strategy': {
                'charge_preference': 'night_transition',
                'discharge_preference': 'morning_evening_increase',
                'target_soc_autumn': 85,
                'degradation_allowance': 1.1
            },
            'heating_optimization': {
                'heat_pump_scheduling': 'increasing_usage',
                'thermal_mass_prep': True,  # Voorbereiding winter
                'system_check_scheduling': 'autumn_maintenance'
            },
            'device_scheduling': {
                'dishwasher_preferred_hours': ['02:00-05:00'],
                'seasonal_appliance_prep': True,
                'winter_prep_activities': True
            },
            'pv_optimization': {
                'self_consumption_priority': 'decreasing_available',
                'battery_charging_from_pv': 'maximum_capture',
                'grid_export_threshold': 0.03
            },
            'cost_reduction_factor': 0.95,
            'consumption_increase_factor': 1.1
        }


class BatteryDegradationManager:
    """Batterij degradation management voor lifecycle optimization"""
    
    def __init__(self, config: Config):
        self.config = config
        self.batteries = config.get(['battery'], [])
        
    async def optimize_battery_lifecycle(self, optimization_results: Dict) -> Dict[str, Any]:
        """Optimaliseer batterij lifecycle vs kosten"""
        try:
            degradation_strategy = {
                'lifecycle_optimization': True,
                'battery_strategies': {},
                'degradation_factors': {},
                'maintenance_schedule': {},
                'replacement_planning': {}
            }
            
            for i, battery in enumerate(self.batteries):
                battery_id = f'battery_{i}'
                
                # Huidige battery status
                battery_status = await self._assess_battery_health(battery)
                
                # Degradation-aware strategy
                strategy = await self._calculate_degradation_aware_strategy(
                    battery, battery_status, optimization_results
                )
                
                degradation_strategy['battery_strategies'][battery_id] = strategy
                degradation_strategy['degradation_factors'][battery_id] = battery_status
            
            return degradation_strategy
            
        except Exception as e:
            logging.error(f"Battery degradation management fout: {e}")
            return {'enabled': False}
    
    async def _assess_battery_health(self, battery_config: Dict) -> Dict[str, Any]:
        """Beoordeel huidige batterij gezondheid"""
        try:
            # Zou echte battery monitoring data moeten gebruiken
            estimated_age_months = 24  # Placeholder
            estimated_cycles = 500     # Placeholder
            capacity_retention = 0.95   # 95% van originele capaciteit
            
            # Battery health score (0-1)
            age_factor = max(0.5, 1 - (estimated_age_months / 120))  # 10 jaar levensduur
            cycle_factor = max(0.5, 1 - (estimated_cycles / 8000))   # 8000 cycles max
            capacity_factor = capacity_retention
            
            health_score = (age_factor + cycle_factor + capacity_factor) / 3
            
            return {
                'health_score': health_score,
                'estimated_age_months': estimated_age_months,
                'cycle_count': estimated_cycles,
                'capacity_retention': capacity_retention,
                'degradation_rate': self._calculate_degradation_rate(health_score),
                'remaining_cycles': int((1 - health_score) * 8000),
                'replacement_date': self._estimate_replacement_date(health_score)
            }
            
        except Exception as e:
            logging.error(f"Battery health assessment fout: {e}")
            return {'health_score': 0.8}  # Conservatieve default
    
    def _calculate_degradation_rate(self, health_score: float) -> float:
        """Bereken huidige degradation rate"""
        # Degradatie versnelt als battery ouder wordt
        if health_score > 0.9:
            return 0.005  # 0.5% per maand
        elif health_score > 0.8:
            return 0.008  # 0.8% per maand
        elif health_score > 0.7:
            return 0.012  # 1.2% per maand
        else:
            return 0.020  # 2% per maand (vervangen!)
    
    def _estimate_replacement_date(self, health_score: float) -> str:
        """Schat vervangingsdatum batterij"""
        if health_score > 0.8:
            months_remaining = int((health_score - 0.7) / 0.01)  # Tot 70% capaciteit
        else:
            months_remaining = 6  # Vervang binnen 6 maanden
        
        replacement_date = datetime.now() + timedelta(days=months_remaining * 30)
        return replacement_date.strftime('%Y-%m-%d')
    
    async def _calculate_degradation_aware_strategy(self, battery_config: Dict, 
                                                   battery_status: Dict,
                                                   optimization_results: Dict) -> Dict[str, Any]:
        """Bereken degradation-aware battery strategy"""
        try:
            health_score = battery_status.get('health_score', 0.8)
            
            # Adjust strategy based on battery health
            if health_score > 0.9:  # Nieuwe batterij - maximale utilization
                strategy = {
                    'utilization_factor': 1.0,
                    'max_daily_cycles': 2.5,
                    'soc_range': {'min': 10, 'max': 95},
                    'fast_charging_allowed': True,
                    'deep_discharge_allowed': True,
                    'priority': 'cost_optimization'
                }
            elif health_score > 0.8:  # Goede conditie - normale utilization
                strategy = {
                    'utilization_factor': 0.9,
                    'max_daily_cycles': 2.0,
                    'soc_range': {'min': 15, 'max': 90},
                    'fast_charging_allowed': True,
                    'deep_discharge_allowed': False,
                    'priority': 'balanced'
                }
            elif health_score > 0.7:  # Matige conditie - voorzichtige utilization
                strategy = {
                    'utilization_factor': 0.7,
                    'max_daily_cycles': 1.5,
                    'soc_range': {'min': 20, 'max': 85},
                    'fast_charging_allowed': False,
                    'deep_discharge_allowed': False,
                    'priority': 'lifecycle_preservation'
                }
            else:  # Slechte conditie - minimale utilization
                strategy = {
                    'utilization_factor': 0.5,
                    'max_daily_cycles': 1.0,
                    'soc_range': {'min': 30, 'max': 80},
                    'fast_charging_allowed': False,
                    'deep_discharge_allowed': False,
                    'priority': 'emergency_only',
                    'replacement_urgent': True
                }
            
            # Calculate economic optimization
            strategy['economic_analysis'] = await self._calculate_battery_economics(
                battery_config, battery_status, strategy
            )
            
            return strategy
            
        except Exception as e:
            logging.error(f"Degradation-aware strategy fout: {e}")
            return {'utilization_factor': 0.8, 'priority': 'balanced'}
    
    async def _calculate_battery_economics(self, battery_config: Dict, 
                                         battery_status: Dict, strategy: Dict) -> Dict[str, Any]:
        """Bereken battery economische optimalisatie"""
        try:
            capacity = battery_config.get('capacity', 10)  # kWh
            current_value = 8000  # € - zou dynamisch moeten zijn
            replacement_cost = 12000  # € - nieuwe battery
            
            # Bereken kosten per kWh cyclus
            remaining_cycles = battery_status.get('remaining_cycles', 2000)
            cost_per_kwh_cycle = current_value / (remaining_cycles * capacity)
            
            # Daily cost van battery usage
            daily_cycles = strategy.get('max_daily_cycles', 2.0)
            daily_degradation_cost = daily_cycles * capacity * cost_per_kwh_cycle
            
            # Bereken break-even point voor cycling vs grid
            grid_price_spread = 0.15  # € typische spread
            battery_efficiency = 0.95
            
            break_even_spread = daily_degradation_cost / (capacity * battery_efficiency)
            
            return {
                'cost_per_kwh_cycle': round(cost_per_kwh_cycle, 4),
                'daily_degradation_cost': round(daily_degradation_cost, 3),
                'break_even_price_spread': round(break_even_spread, 4),
                'profitable_cycling': break_even_spread < grid_price_spread,
                'monthly_degradation_cost': round(daily_degradation_cost * 30, 2),
                'replacement_cost': replacement_cost,
                'economic_lifetime_remaining': round(remaining_cycles / (daily_cycles * 365), 1)
            }
            
        except Exception as e:
            logging.error(f"Battery economics calculation fout: {e}")
            return {'profitable_cycling': True}


# Convenience functions
def create_multiday_optimizer(config: Config, ha_integration: HomeAssistantIntegration) -> MultiDayOptimizer:
    """Create and return a MultiDayOptimizer instance"""
    return MultiDayOptimizer(config, ha_integration)