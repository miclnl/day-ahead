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
# Statistical replacement for sklearn
# Using numpy-based statistical methods instead of ML dependencies
from da_statistical_predictor import StatisticalPredictor
from da_enhanced_weather import EnhancedWeatherService


class MultiDayOptimizer:
    """Multi-day optimization voor langere planning horizont"""
    
    def __init__(self, da_calc_instance):
        """Initialize with reference to main DaCalc instance"""
        self.da_calc = da_calc_instance
        self.config = da_calc_instance.config
        self.settings = self._load_multiday_settings()
        
        # Initialize statistical components
        self.predictor = StatisticalPredictor(da_calc_instance)
        self.weather_service = EnhancedWeatherService(da_calc_instance)
        self.holiday_detector = HolidayVacationDetector(da_calc_instance)
        self.seasonal_optimizer = SeasonalOptimizer(da_calc_instance)
        self.degradation_manager = BatteryDegradationManager(da_calc_instance)
        
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


class HolidayVacationDetector:
    """
    Holiday and vacation detection system for multiday optimization.
    Detects periods with different consumption patterns.
    """
    
    def __init__(self, da_calc_instance):
        self.da_calc = da_calc_instance
        self.config = da_calc_instance.config
        
        # Holiday detection settings
        self.vacation_detection_enabled = self.config.get(['multiday', 'vacation_detection'], 0, True)
        self.holiday_consumption_factor = self.config.get(['multiday', 'holiday_consumption_factor'], 0, 0.7)
        
        # Known holidays (Netherlands)
        self.fixed_holidays = [
            (1, 1),   # New Year
            (4, 27),  # King's Day
            (5, 5),   # Liberation Day
            (12, 25), # Christmas
            (12, 26)  # Boxing Day
        ]
        
        logging.info("Holiday/Vacation detector initialized")
    
    def detect_holiday_periods(self, start_date: datetime, days_ahead: int) -> List[Dict[str, Any]]:
        """Detect holiday periods in planning horizon"""
        
        holiday_periods = []
        
        try:
            for day_offset in range(days_ahead):
                check_date = start_date + timedelta(days=day_offset)
                
                # Check if it's a weekend
                is_weekend = check_date.weekday() >= 5
                
                # Check if it's a fixed holiday
                is_fixed_holiday = (check_date.month, check_date.day) in self.fixed_holidays
                
                # Check if it's Easter (calculated)
                is_easter = self._is_easter_period(check_date)
                
                # Check for vacation mode (from HA sensor or config)
                is_vacation = self._check_vacation_mode(check_date)
                
                if is_weekend or is_fixed_holiday or is_easter or is_vacation:
                    holiday_periods.append({
                        'date': check_date,
                        'type': self._get_holiday_type(is_weekend, is_fixed_holiday, is_easter, is_vacation),
                        'consumption_factor': self._get_consumption_factor(is_weekend, is_fixed_holiday, is_vacation),
                        'heating_factor': self._get_heating_factor(is_weekend, is_vacation)
                    })
            
            return holiday_periods
            
        except Exception as e:
            logging.error(f"Error detecting holiday periods: {e}")
            return []
    
    def _is_easter_period(self, check_date: datetime) -> bool:
        """Check if date is in Easter period (simplified)"""
        # Simplified Easter calculation - would use proper algorithm in production
        year = check_date.year
        easter_dates = {
            2024: (3, 31),
            2025: (4, 20),
            2026: (4, 5),
        }
        easter_date = easter_dates.get(year)
        if easter_date:
            easter = datetime(year, easter_date[0], easter_date[1])
            # Easter period: Good Friday to Easter Monday
            return abs((check_date - easter).days) <= 3
        return False
    
    def _check_vacation_mode(self, check_date: datetime) -> bool:
        """Check if vacation mode is active"""
        try:
            # Check HA sensor for vacation mode
            vacation_sensor = self.config.get(['vacation', 'sensor_entity'], None)
            if vacation_sensor and hasattr(self.da_calc, 'get_state'):
                vacation_state = self.da_calc.get_state(vacation_sensor)
                return vacation_state and vacation_state.lower() in ['on', 'true', 'away']
            
            return False
        except:
            return False
    
    def _get_holiday_type(self, weekend: bool, holiday: bool, easter: bool, vacation: bool) -> str:
        """Determine holiday type"""
        if vacation:
            return 'vacation'
        elif holiday or easter:
            return 'public_holiday'
        elif weekend:
            return 'weekend'
        return 'regular'
    
    def _get_consumption_factor(self, weekend: bool, holiday: bool, vacation: bool) -> float:
        """Get consumption adjustment factor"""
        if vacation:
            return 0.3  # Much lower consumption during vacation
        elif holiday:
            return 0.8  # Slightly lower during holidays
        elif weekend:
            return 0.9  # Slightly lower on weekends
        return 1.0
    
    def _get_heating_factor(self, weekend: bool, vacation: bool) -> float:
        """Get heating adjustment factor"""
        if vacation:
            return 0.5  # Lower heating when away
        elif weekend:
            return 1.1  # Slightly higher heating when home more
        return 1.0


class SeasonalOptimizer:
    """
    Seasonal optimization adjustments for multiday planning.
    Handles seasonal variations in consumption, production, and strategy.
    """
    
    def __init__(self, da_calc_instance):
        self.da_calc = da_calc_instance
        self.config = da_calc_instance.config
        
        # Seasonal settings
        self.seasonal_optimization_enabled = self.config.get(['multiday', 'seasonal_optimization'], 0, True)
        
        # Seasonal factors
        self.seasonal_consumption_factors = {
            'winter': 1.4,   # Higher heating demand
            'spring': 0.9,   # Moderate consumption
            'summer': 0.8,   # Lower heating, possible AC
            'autumn': 1.1    # Increasing heating demand
        }
        
        self.seasonal_solar_factors = {
            'winter': 0.3,   # Low sun angle, short days
            'spring': 0.8,   # Good production
            'summer': 1.0,   # Peak production
            'autumn': 0.6    # Decreasing production
        }
        
        logging.info("Seasonal optimizer initialized")
    
    def get_seasonal_adjustments(self, date: datetime) -> Dict[str, Any]:
        """Get seasonal adjustments for given date"""
        
        try:
            season = self._get_season(date)
            
            adjustments = {
                'season': season,
                'consumption_factor': self.seasonal_consumption_factors[season],
                'solar_factor': self.seasonal_solar_factors[season],
                'heating_priority': self._get_heating_priority(season),
                'battery_strategy': self._get_seasonal_battery_strategy(season),
                'daylight_hours': self._get_daylight_hours(date),
                'temperature_impact': self._get_temperature_impact_factor(season)
            }
            
            return adjustments
            
        except Exception as e:
            logging.error(f"Error calculating seasonal adjustments: {e}")
            return {'season': 'unknown', 'consumption_factor': 1.0, 'solar_factor': 1.0}
    
    def _get_season(self, date: datetime) -> str:
        """Determine season for given date"""
        month = date.month
        if month in [12, 1, 2]:
            return 'winter'
        elif month in [3, 4, 5]:
            return 'spring'
        elif month in [6, 7, 8]:
            return 'summer'
        else:  # 9, 10, 11
            return 'autumn'
    
    def _get_heating_priority(self, season: str) -> float:
        """Get heating priority factor for season"""
        priorities = {
            'winter': 1.0,   # Highest priority
            'autumn': 0.7,   # High priority
            'spring': 0.3,   # Low priority
            'summer': 0.0    # No heating priority
        }
        return priorities.get(season, 0.5)
    
    def _get_seasonal_battery_strategy(self, season: str) -> str:
        """Get optimal battery strategy for season"""
        strategies = {
            'winter': 'heating_support',     # Support heating with stored energy
            'spring': 'balanced',            # Balanced approach
            'summer': 'solar_storage',       # Store excess solar
            'autumn': 'cost_optimization'    # Focus on cost savings
        }
        return strategies.get(season, 'balanced')
    
    def _get_daylight_hours(self, date: datetime) -> Dict[str, int]:
        """Calculate daylight hours for date (simplified)"""
        month = date.month
        daylight_hours = {
            1: 8, 2: 9, 3: 11, 4: 13, 5: 15, 6: 16,
            7: 16, 8: 14, 9: 12, 10: 10, 11: 8, 12: 7
        }
        hours = daylight_hours.get(month, 12)
        return {
            'daylight_hours': hours,
            'sunrise_hour': 12 - hours // 2,
            'sunset_hour': 12 + hours // 2
        }
    
    def _get_temperature_impact_factor(self, season: str) -> float:
        """Get temperature impact factor for consumption"""
        impacts = {
            'winter': 1.5,   # High temperature impact
            'autumn': 1.2,   # Moderate impact
            'spring': 1.1,   # Low impact
            'summer': 1.3    # Moderate impact (cooling)
        }
        return impacts.get(season, 1.0)


class BatteryDegradationManager:
    """
    Battery degradation tracking and optimization for multiday planning.
    Balances cycling benefits with battery longevity.
    """
    
    def __init__(self, da_calc_instance):
        self.da_calc = da_calc_instance
        self.config = da_calc_instance.config
        
        # Battery degradation settings
        self.degradation_tracking_enabled = self.config.get(['multiday', 'degradation_tracking'], 0, True)
        
        # Battery specifications
        self.battery_capacity = self.config.get(['battery', 'capacity'], 0, 10.0)  # kWh
        self.battery_cost = self.config.get(['battery', 'replacement_cost'], 0, 8000.0)  # €
        self.cycle_life = self.config.get(['battery', 'cycle_life'], 0, 6000)  # cycles
        self.calendar_life = self.config.get(['battery', 'calendar_life_years'], 0, 10)  # years
        
        # Degradation tracking
        self.total_cycles = self._load_cycle_count()
        self.installation_date = self._load_installation_date()
        self.degradation_history = []
        
        logging.info("Battery degradation manager initialized")
    
    def calculate_degradation_impact(self, planned_cycles_per_day: float, days_ahead: int) -> Dict[str, Any]:
        """Calculate degradation impact of planned cycling"""
        
        try:
            total_planned_cycles = planned_cycles_per_day * days_ahead
            
            # Calculate current degradation state
            calendar_age_years = (datetime.now() - self.installation_date).days / 365.25
            cycle_degradation = self.total_cycles / self.cycle_life
            calendar_degradation = calendar_age_years / self.calendar_life
            
            # Combined degradation (worst of both)
            current_degradation = max(cycle_degradation, calendar_degradation)
            
            # Calculate impact of planned cycles
            additional_cycle_degradation = total_planned_cycles / self.cycle_life
            new_total_degradation = min(1.0, current_degradation + additional_cycle_degradation)
            
            # Economic analysis
            degradation_cost = additional_cycle_degradation * self.battery_cost
            cost_per_cycle = degradation_cost / max(total_planned_cycles, 0.001)
            
            return {
                'current_degradation_percent': round(current_degradation * 100, 1),
                'additional_degradation_percent': round(additional_cycle_degradation * 100, 3),
                'total_degradation_after_plan': round(new_total_degradation * 100, 1),
                'degradation_cost_euros': round(degradation_cost, 2),
                'cost_per_cycle_euros': round(cost_per_cycle, 3),
                'remaining_capacity_percent': round((1 - new_total_degradation) * 100, 1),
                'estimated_replacement_months': self._estimate_replacement_time(new_total_degradation),
                'cycling_recommended': degradation_cost < 50.0,  # Arbitrary threshold
                'cycle_budget_remaining': max(0, self.cycle_life - self.total_cycles - total_planned_cycles)
            }
            
        except Exception as e:
            logging.error(f"Error calculating degradation impact: {e}")
            return {'cycling_recommended': True}
    
    def _load_cycle_count(self) -> float:
        """Load current cycle count from storage"""
        try:
            # Would load from database or file in production
            return self.config.get(['battery', 'total_cycles'], 0, 0.0)
        except:
            return 0.0
    
    def _load_installation_date(self) -> datetime:
        """Load battery installation date"""
        try:
            install_date_str = self.config.get(['battery', 'installation_date'], None, None)
            if install_date_str:
                return datetime.strptime(install_date_str, '%Y-%m-%d')
            else:
                return datetime.now() - timedelta(days=365)  # Default: 1 year ago
        except:
            return datetime.now() - timedelta(days=365)
    
    def _estimate_replacement_time(self, degradation_level: float) -> int:
        """Estimate months until battery replacement needed"""
        if degradation_level >= 0.8:  # 80% degradation threshold
            return 0
        
        remaining_degradation = 0.8 - degradation_level
        # Estimate based on current degradation rate
        degradation_per_month = degradation_level / max(1, (datetime.now() - self.installation_date).days / 30.4)
        
        if degradation_per_month <= 0:
            return 120  # 10 years default
        
        return int(remaining_degradation / degradation_per_month)


# Convenience functions
def create_multiday_optimizer(da_calc_instance) -> MultiDayOptimizer:
    """Create and return a MultiDayOptimizer instance"""
    return MultiDayOptimizer(da_calc_instance)