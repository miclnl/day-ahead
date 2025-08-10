"""
Smart Optimization Engine voor advanced cost minimization
Focus op real-world kostenbesparingen door betere voorspellingen en automatische scheduling
"""

import logging
import pandas as pd
import numpy as np
import platform
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import asyncio
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
import json
import aiohttp
from dao.prog.da_config import Config
from dao.prog.da_ha_integration import HomeAssistantIntegration


def is_raspberry_pi() -> bool:
    """Detecteer of we draaien op een Raspberry Pi"""
    try:
        with open('/proc/cpuinfo', 'r') as f:
            cpuinfo = f.read()
        return 'BCM' in cpuinfo or 'Raspberry Pi' in cpuinfo
    except:
        return platform.machine().startswith('arm') or platform.machine().startswith('aarch64')


class SmartOptimizationEngine:
    """Geavanceerde optimalisatie engine voor kosten minimalisatie"""
    
    def __init__(self, config: Config, ha_integration: HomeAssistantIntegration):
        self.config = config
        self.ha = ha_integration
        self.is_pi = is_raspberry_pi()
        self.settings = self._load_smart_settings()
        
        if self.is_pi:
            logging.info("Raspberry Pi gedetecteerd - lightweight optimalisaties actief")
        
        # Prediction models (Pi-optimized)
        self.consumption_model = None
        self.pv_model = None
        self.load_detector = HighLoadDetector(config, ha_integration, lightweight=self.is_pi)
        self.device_scheduler = SmartDeviceScheduler(config, ha_integration)
        self.battery_manager = AdaptiveBatteryManager(config, ha_integration)
        
        # Historical data cache (optimized for Pi with 8GB)
        if self.is_pi:
            # With 8GB Pi we can handle more data
            cache_size = 300
            logging.info("Pi met voldoende RAM gedetecteerd - verhoogde cache size")
        else:
            cache_size = 500
        
        self.consumption_history = pd.DataFrame()
        self.weather_history = pd.DataFrame()
        self.price_history = pd.DataFrame()
        self.max_cache_size = cache_size
        
        logging.info(f"Smart Optimization Engine geïnitialiseerd (Pi mode: {self.is_pi})")
    
    def _load_smart_settings(self) -> Dict[str, Any]:
        """Laad smart optimization instellingen"""
        # Pi-specific optimizations (8GB variant can handle more)
        if self.is_pi:
            prediction_horizon = self.config.get(['smart_optimization', 'prediction_horizon'], 60)  # Good for 8GB Pi
            forecast_days = self.config.get(['smart_optimization', 'weather_forecast_days'], 6)  # Good for 8GB Pi
        else:
            prediction_horizon = self.config.get(['smart_optimization', 'prediction_horizon'], 72)
            forecast_days = self.config.get(['smart_optimization', 'weather_forecast_days'], 7)
        
        return {
            # Core Features
            'advanced_prediction_enabled': self.config.get(['smart_optimization', 'advanced_prediction', 'enabled'], True),
            'device_scheduling_enabled': self.config.get(['smart_optimization', 'device_scheduling', 'enabled'], True),
            'high_load_detection_enabled': self.config.get(['smart_optimization', 'high_load_detection', 'enabled'], True),
            'adaptive_battery_enabled': self.config.get(['smart_optimization', 'adaptive_battery', 'enabled'], True),
            
            # Advanced Prediction Settings (Pi-optimized)
            'prediction_horizon_hours': prediction_horizon,
            'weather_forecast_days': forecast_days,
            'consumption_pattern_learning': self.config.get(['smart_optimization', 'consumption_learning'], True),
            'lightweight_models': self.is_pi,
            
            # Device Scheduling Settings
            'auto_schedule_dishwasher': self.config.get(['smart_optimization', 'devices', 'dishwasher'], True),
            'auto_schedule_washing_machine': self.config.get(['smart_optimization', 'devices', 'washing_machine'], True),
            'auto_schedule_dryer': self.config.get(['smart_optimization', 'devices', 'dryer'], True),
            'auto_schedule_heat_pump': self.config.get(['smart_optimization', 'devices', 'heat_pump'], True),
            
            # High Load Detection Settings
            'high_load_threshold_multiplier': self.config.get(['smart_optimization', 'high_load_threshold'], 2.5),
            'high_load_reaction_time_minutes': self.config.get(['smart_optimization', 'reaction_time'], 5),
            'battery_emergency_discharge': self.config.get(['smart_optimization', 'emergency_discharge'], True),
            
            # Battery Management Settings
            'dynamic_soc_targets': self.config.get(['smart_optimization', 'battery', 'dynamic_soc'], True),
            'battery_health_priority': self.config.get(['smart_optimization', 'battery', 'health_priority'], 0.8),
            'seasonal_battery_strategy': self.config.get(['smart_optimization', 'battery', 'seasonal'], True)
        }
    
    async def run_smart_optimization(self, prices: pd.DataFrame, weather_forecast: pd.DataFrame) -> Dict[str, Any]:
        """Hoofdfunctie voor smart optimization"""
        try:
            optimization_results = {}
            
            # 1. Update models met recente data
            if self.settings['advanced_prediction_enabled']:
                await self._update_prediction_models()
                optimization_results['prediction_update'] = True
            
            # 2. Voorspel consumption en PV productie
            consumption_forecast = await self._predict_consumption(prices.index)
            pv_forecast = await self._predict_pv_production(weather_forecast)
            
            # 3. Detecteer en plan voor verwacht groot verbruik
            if self.settings['high_load_detection_enabled']:
                high_load_events = await self.load_detector.predict_high_loads(consumption_forecast)
                optimization_results['predicted_high_loads'] = len(high_load_events)
            
            # 4. Plan slimme apparaat scheduling
            if self.settings['device_scheduling_enabled']:
                device_schedule = await self.device_scheduler.create_optimal_schedule(
                    prices, consumption_forecast, pv_forecast
                )
                optimization_results['device_schedule'] = device_schedule
            
            # 5. Optimaliseer batterij strategie
            if self.settings['adaptive_battery_enabled']:
                battery_strategy = await self.battery_manager.optimize_battery_schedule(
                    prices, consumption_forecast, pv_forecast
                )
                optimization_results['battery_strategy'] = battery_strategy
            
            # 6. Bereken potentiële besparingen
            savings_analysis = await self._calculate_potential_savings(
                prices, consumption_forecast, pv_forecast, optimization_results
            )
            optimization_results['potential_savings'] = savings_analysis
            
            return {
                'success': True,
                'timestamp': datetime.now().isoformat(),
                'optimization_results': optimization_results,
                'next_optimization': (datetime.now() + timedelta(minutes=15)).isoformat()
            }
            
        except Exception as e:
            logging.error(f"Smart optimization fout: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _update_prediction_models(self):
        """Update ML modellen met recente data"""
        try:
            # Haal recente consumptie data op
            end_time = datetime.now()
            start_time = end_time - timedelta(days=30)
            
            consumption_data = await self._get_consumption_history(start_time, end_time)
            weather_data = await self._get_weather_history(start_time, end_time)
            
            if len(consumption_data) > 100:  # Genoeg data voor training
                # Train consumption model
                self.consumption_model = self._train_consumption_model(consumption_data, weather_data)
                logging.info("Consumption prediction model bijgewerkt")
            
            if len(weather_data) > 50:
                # Train PV model
                pv_data = await self._get_pv_history(start_time, end_time)
                if len(pv_data) > 50:
                    self.pv_model = self._train_pv_model(pv_data, weather_data)
                    logging.info("PV prediction model bijgewerkt")
                    
        except Exception as e:
            logging.error(f"Model update fout: {e}")
    
    def _train_consumption_model(self, consumption_data: pd.DataFrame, weather_data: pd.DataFrame) -> RandomForestRegressor:
        """Train geavanceerd consumption prediction model"""
        try:
            # Feature engineering
            features = pd.DataFrame()
            
            # Time features
            features['hour'] = consumption_data.index.hour
            features['day_of_week'] = consumption_data.index.dayofweek
            features['month'] = consumption_data.index.month
            features['is_weekend'] = (consumption_data.index.dayofweek >= 5).astype(int)
            
            # Weather features (if available)
            if not weather_data.empty:
                # Resample weather to match consumption frequency
                weather_resampled = weather_data.resample('1H').mean().reindex(consumption_data.index, method='nearest')
                features['temperature'] = weather_resampled.get('temperature', 15)  # Default 15°C
                features['solar_radiation'] = weather_resampled.get('solar_radiation', 0)
                features['wind_speed'] = weather_resampled.get('wind_speed', 0)
            else:
                features['temperature'] = 15  # Default values
                features['solar_radiation'] = 0
                features['wind_speed'] = 0
            
            # Historical consumption features (lag features)
            features['consumption_lag_1h'] = consumption_data['consumption'].shift(1)
            features['consumption_lag_24h'] = consumption_data['consumption'].shift(24)
            features['consumption_lag_168h'] = consumption_data['consumption'].shift(168)  # 1 week
            
            # Rolling averages
            features['consumption_avg_24h'] = consumption_data['consumption'].rolling(24, min_periods=1).mean()
            features['consumption_avg_7d'] = consumption_data['consumption'].rolling(168, min_periods=1).mean()
            
            # Seasonal features
            features['season'] = ((consumption_data.index.month % 12 + 3) // 3).map(
                {1: 0, 2: 1, 3: 2, 4: 3}  # Winter=0, Spring=1, Summer=2, Autumn=3
            )
            
            # Holiday detection (simplified)
            features['is_holiday'] = self._detect_holidays(consumption_data.index)
            
            # Drop rows with NaN values
            features = features.dropna()
            target = consumption_data.loc[features.index, 'consumption']
            
            if len(features) < 50:
                logging.warning("Onvoldoende data voor model training")
                return None
            
            # Train model
            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
            
            model.fit(features, target)
            
            # Log feature importance
            feature_importance = dict(zip(features.columns, model.feature_importances_))
            logging.info(f"Top features: {sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]}")
            
            return model
            
        except Exception as e:
            logging.error(f"Consumption model training fout: {e}")
            return None
    
    async def _predict_consumption(self, future_index: pd.DatetimeIndex) -> pd.DataFrame:
        """Voorspel toekomstig verbruik"""
        try:
            if self.consumption_model is None:
                # Fallback: gebruik huidige baseload
                current_baseload = self.config.get(['baseload'], {}).get('current', 0.5)
                return pd.DataFrame({
                    'predicted_consumption': [current_baseload] * len(future_index)
                }, index=future_index)
            
            # Prepare features voor voorspelling
            features = pd.DataFrame()
            
            # Time features
            features['hour'] = future_index.hour
            features['day_of_week'] = future_index.dayofweek
            features['month'] = future_index.month
            features['is_weekend'] = (future_index.dayofweek >= 5).astype(int)
            
            # Weather forecast features (zou van weather API komen)
            features['temperature'] = 15  # Placeholder
            features['solar_radiation'] = 0
            features['wind_speed'] = 0
            
            # Historical features (van recente data)
            recent_consumption = await self._get_recent_consumption()
            if not recent_consumption.empty:
                last_consumption = recent_consumption['consumption'].iloc[-1]
                features['consumption_lag_1h'] = last_consumption
                features['consumption_lag_24h'] = recent_consumption['consumption'].iloc[-24] if len(recent_consumption) >= 24 else last_consumption
                features['consumption_lag_168h'] = recent_consumption['consumption'].iloc[-168] if len(recent_consumption) >= 168 else last_consumption
                features['consumption_avg_24h'] = recent_consumption['consumption'].tail(24).mean()
                features['consumption_avg_7d'] = recent_consumption['consumption'].tail(168).mean()
            else:
                features['consumption_lag_1h'] = 0.5
                features['consumption_lag_24h'] = 0.5
                features['consumption_lag_168h'] = 0.5
                features['consumption_avg_24h'] = 0.5
                features['consumption_avg_7d'] = 0.5
            
            # Seasonal features
            features['season'] = ((future_index.month % 12 + 3) // 3).map(
                {1: 0, 2: 1, 3: 2, 4: 3}
            )
            
            features['is_holiday'] = self._detect_holidays(future_index)
            
            # Voorspelling
            predicted_consumption = self.consumption_model.predict(features)
            
            # Sanity checks
            predicted_consumption = np.clip(predicted_consumption, 0.1, 10.0)  # Redelijke grenzen
            
            return pd.DataFrame({
                'predicted_consumption': predicted_consumption
            }, index=future_index)
            
        except Exception as e:
            logging.error(f"Consumption prediction fout: {e}")
            # Fallback
            baseload = self.config.get(['baseload'], {}).get('current', 0.5)
            return pd.DataFrame({
                'predicted_consumption': [baseload] * len(future_index)
            }, index=future_index)
    
    async def _predict_pv_production(self, weather_forecast: pd.DataFrame) -> pd.DataFrame:
        """Voorspel PV productie op basis van weer voorspelling"""
        try:
            solar_config = self.config.get(['solar'], {})
            capacity_kwp = solar_config.get('capacity', 0)
            
            if capacity_kwp == 0:
                # Geen PV systeem
                return pd.DataFrame({'predicted_pv': [0] * len(weather_forecast)}, index=weather_forecast.index)
            
            # Basis PV model (kan later vervangen door trained model)
            pv_production = []
            
            for timestamp, row in weather_forecast.iterrows():
                hour = timestamp.hour
                solar_radiation = row.get('solar_radiation', 0)  # W/m²
                cloud_cover = row.get('cloud_cover', 0)  # 0-1
                temperature = row.get('temperature', 15)  # °C
                
                # Basis berekening
                if 6 <= hour <= 20:  # Daglicht uren
                    # Solar angle factor (simplified)
                    solar_angle_factor = np.sin((hour - 6) * np.pi / 14)
                    
                    # Cloud impact
                    clear_sky_factor = 1 - (cloud_cover * 0.8)
                    
                    # Temperature impact (PV efficiency daalt bij hogere temperaturen)
                    temp_factor = 1 - ((temperature - 25) * 0.004)
                    temp_factor = np.clip(temp_factor, 0.7, 1.1)
                    
                    # PV power calculation
                    pv_power = (capacity_kwp * solar_radiation / 1000 * 
                               solar_angle_factor * clear_sky_factor * temp_factor)
                    pv_power = max(0, pv_power)
                else:
                    pv_power = 0
                
                pv_production.append(pv_power)
            
            return pd.DataFrame({
                'predicted_pv': pv_production
            }, index=weather_forecast.index)
            
        except Exception as e:
            logging.error(f"PV prediction fout: {e}")
            return pd.DataFrame({'predicted_pv': [0] * len(weather_forecast)}, index=weather_forecast.index)
    
    def _detect_holidays(self, dates: pd.DatetimeIndex) -> List[int]:
        """Detecteer feestdagen (simplified Nederlandse feestdagen)"""
        holidays = []
        for date in dates:
            is_holiday = 0
            
            # Nederlandse feestdagen (simplified)
            if (date.month == 1 and date.day == 1) or \
               (date.month == 4 and date.day == 27) or \
               (date.month == 5 and date.day == 5) or \
               (date.month == 12 and date.day in [25, 26]):
                is_holiday = 1
            
            holidays.append(is_holiday)
        
        return holidays
    
    async def _get_consumption_history(self, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        """Haal historische consumptie data op"""
        try:
            # Via HA integration
            consumption_entity = self.config.get(['consumption', 'entity'])
            if consumption_entity:
                history_result = await self.ha.get_history([consumption_entity], start_time, end_time)
                if history_result['success']:
                    # Process history data to DataFrame
                    return self._process_history_to_df(history_result['history'], 'consumption')
            
            # Fallback: genereer dummy data voor testing
            logging.warning("Geen historische consumptie data beschikbaar, gebruik dummy data")
            return self._generate_dummy_consumption_data(start_time, end_time)
            
        except Exception as e:
            logging.error(f"Fout bij ophalen consumptie historie: {e}")
            return pd.DataFrame()
    
    async def _get_recent_consumption(self) -> pd.DataFrame:
        """Haal recente consumptie data op (laatste 7 dagen)"""
        end_time = datetime.now()
        start_time = end_time - timedelta(days=7)
        return await self._get_consumption_history(start_time, end_time)
    
    def _process_history_to_df(self, history_data: List, column_name: str) -> pd.DataFrame:
        """Converteer HA history data naar DataFrame"""
        try:
            records = []
            for entity_history in history_data:
                if not entity_history:
                    continue
                    
                for state in entity_history:
                    try:
                        timestamp = pd.to_datetime(state['last_changed'])
                        value = float(state['state'])
                        records.append({'timestamp': timestamp, column_name: value})
                    except (ValueError, KeyError):
                        continue
            
            if not records:
                return pd.DataFrame()
            
            df = pd.DataFrame(records)
            df = df.set_index('timestamp')
            df = df.sort_index()
            
            # Resample to hourly
            df = df.resample('1H').mean()
            
            return df
            
        except Exception as e:
            logging.error(f"Fout bij verwerken history data: {e}")
            return pd.DataFrame()
    
    def _generate_dummy_consumption_data(self, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        """Genereer realistische dummy consumption data voor testing"""
        time_range = pd.date_range(start_time, end_time, freq='1H')
        
        consumption_data = []
        for timestamp in time_range:
            hour = timestamp.hour
            day_of_week = timestamp.dayofweek
            
            # Basis consumption pattern
            if 6 <= hour <= 23:  # Dag
                base_consumption = 0.8
                if 17 <= hour <= 21:  # Avondpiek
                    base_consumption = 1.5
            else:  # Nacht
                base_consumption = 0.3
            
            # Weekend pattern
            if day_of_week >= 5:  # Weekend
                base_consumption *= 1.2
            
            # Random variatie
            consumption = base_consumption * (1 + np.random.normal(0, 0.2))
            consumption = max(0.1, consumption)  # Minimum 0.1 kW
            
            consumption_data.append(consumption)
        
        return pd.DataFrame({
            'consumption': consumption_data
        }, index=time_range)
    
    async def _calculate_potential_savings(self, prices: pd.DataFrame, consumption_forecast: pd.DataFrame, 
                                         pv_forecast: pd.DataFrame, optimization_results: Dict) -> Dict[str, float]:
        """Bereken potentiële kostenbesparingen van optimalizaties"""
        try:
            # Bereken baseline kosten (zonder optimalizatie)
            baseline_costs = (consumption_forecast['predicted_consumption'] * prices['price']).sum()
            
            # Bereken geoptimaliseerde kosten
            optimized_consumption = consumption_forecast['predicted_consumption'].copy()
            
            # Device scheduling impact
            if 'device_schedule' in optimization_results:
                device_impact = optimization_results['device_schedule'].get('energy_shifted', 0)
                # Simplified: assume 10% of consumption can be shifted to cheaper hours
                potential_shift = optimized_consumption.sum() * 0.1
                shifted_savings = potential_shift * (prices['price'].max() - prices['price'].min()) * 0.5
            else:
                shifted_savings = 0
            
            # Battery optimization impact
            battery_savings = 0
            if 'battery_strategy' in optimization_results:
                battery_capacity = self._get_total_battery_capacity()
                if battery_capacity > 0:
                    # Simplified: battery can capture price differences
                    daily_cycles = min(2, battery_capacity / optimized_consumption.mean())
                    price_spread = prices['price'].max() - prices['price'].min()
                    battery_savings = daily_cycles * battery_capacity * price_spread * 0.9  # 90% efficiency
            
            # PV self-consumption optimization
            pv_savings = 0
            if not pv_forecast.empty and pv_forecast['predicted_pv'].sum() > 0:
                # More PV self-consumption means less grid purchases at higher prices
                pv_production = pv_forecast['predicted_pv'].sum()
                avg_price = prices['price'].mean()
                pv_savings = pv_production * avg_price * 0.1  # 10% improvement in self-consumption
            
            total_potential_savings = shifted_savings + battery_savings + pv_savings
            
            return {
                'baseline_daily_cost': round(baseline_costs, 2),
                'device_scheduling_savings': round(shifted_savings, 2),
                'battery_optimization_savings': round(battery_savings, 2),
                'pv_optimization_savings': round(pv_savings, 2),
                'total_potential_savings': round(total_potential_savings, 2),
                'percentage_savings': round((total_potential_savings / baseline_costs * 100), 1) if baseline_costs > 0 else 0
            }
            
        except Exception as e:
            logging.error(f"Savings calculation fout: {e}")
            return {'total_potential_savings': 0, 'percentage_savings': 0}
    
    def _get_total_battery_capacity(self) -> float:
        """Get total battery capacity in kWh"""
        batteries = self.config.get(['battery'], [])
        return sum(battery.get('capacity', 0) for battery in batteries)
    
    async def _get_weather_history(self, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        """Haal historische weer data op"""
        # Placeholder - zou echte weather API gebruiken
        return pd.DataFrame()
    
    async def _get_pv_history(self, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        """Haal historische PV productie data op"""
        try:
            solar_entity = self.config.get(['solar', 'entity'])
            if solar_entity:
                history_result = await self.ha.get_history([solar_entity], start_time, end_time)
                if history_result['success']:
                    return self._process_history_to_df(history_result['history'], 'pv_production')
            return pd.DataFrame()
        except Exception as e:
            logging.error(f"Fout bij ophalen PV historie: {e}")
            return pd.DataFrame()


class HighLoadDetector:
    """Detecteer groot verbruik en pas automatisch strategieën aan"""
    
    def __init__(self, config: Config, ha_integration: HomeAssistantIntegration):
        self.config = config
        self.ha = ha_integration
        self.baseline_consumption = 0.5  # kW
        self.high_load_threshold = 2.0  # kW
        self.detection_active = False
        
    async def start_monitoring(self):
        """Start real-time monitoring voor groot verbruik"""
        if not self.config.get(['smart_optimization', 'high_load_detection', 'enabled'], True):
            return
            
        self.detection_active = True
        logging.info("High load detection gestart")
        
        # Background task voor monitoring
        asyncio.create_task(self._monitor_consumption())
    
    async def _monitor_consumption(self):
        """Monitor real-time consumption voor spikes"""
        while self.detection_active:
            try:
                # Haal huidige consumption op
                consumption_entity = self.config.get(['consumption', 'entity'])
                if consumption_entity:
                    current_state = await self.ha.get_entity_state(consumption_entity)
                    if current_state['success']:
                        current_consumption = float(current_state['state'])
                        
                        # Check voor high load event
                        if current_consumption > self.high_load_threshold:
                            await self._handle_high_load_event(current_consumption)
                
                # Check elke 30 seconden
                await asyncio.sleep(30)
                
            except Exception as e:
                logging.error(f"High load monitoring fout: {e}")
                await asyncio.sleep(60)
    
    async def _handle_high_load_event(self, consumption: float):
        """Handle detected high load event"""
        logging.warning(f"High load gedetecteerd: {consumption:.2f} kW")
        
        # Bepaal response strategy
        response_actions = []
        
        # 1. Battery discharge indien beschikbaar
        if self.config.get(['smart_optimization', 'emergency_discharge'], True):
            battery_capacity = self._get_available_battery_capacity()
            if battery_capacity > 0:
                discharge_power = min(consumption * 0.5, battery_capacity)
                response_actions.append({
                    'type': 'battery_discharge',
                    'power': discharge_power,
                    'duration': 60  # minutes
                })
        
        # 2. Delay non-essential devices
        if self.config.get(['smart_optimization', 'device_scheduling', 'enabled'], True):
            delayed_devices = await self._identify_delayable_devices()
            response_actions.extend(delayed_devices)
        
        # 3. Send notification
        response_actions.append({
            'type': 'notification',
            'message': f"Groot verbruik gedetecteerd: {consumption:.2f} kW. Automatische aanpassingen actief."
        })
        
        # Execute response actions
        for action in response_actions:
            await self._execute_response_action(action)
    
    def _get_available_battery_capacity(self) -> float:
        """Get current available battery discharge capacity"""
        # Simplified - zou echte battery status moeten checken
        batteries = self.config.get(['battery'], [])
        total_available = 0
        
        for battery in batteries:
            capacity = battery.get('capacity', 0)
            current_soc = 0.7  # Placeholder - zou uit HA moeten komen
            min_soc = battery.get('min_soc', 0.2)
            
            available = capacity * (current_soc - min_soc)
            total_available += max(0, available)
        
        return total_available
    
    async def _identify_delayable_devices(self) -> List[Dict]:
        """Identificeer apparaten die uitgesteld kunnen worden"""
        delayable = []
        
        # Check common delayable devices
        device_entities = {
            'dishwasher': self.config.get(['devices', 'dishwasher', 'entity']),
            'washing_machine': self.config.get(['devices', 'washing_machine', 'entity']),
            'dryer': self.config.get(['devices', 'dryer', 'entity'])
        }
        
        for device_name, entity_id in device_entities.items():
            if entity_id:
                state = await self.ha.get_entity_state(entity_id)
                if state['success'] and state['state'] == 'on':
                    delayable.append({
                        'type': 'device_delay',
                        'device': device_name,
                        'entity_id': entity_id,
                        'delay_minutes': 30
                    })
        
        return delayable
    
    async def _execute_response_action(self, action: Dict):
        """Execute a response action"""
        try:
            if action['type'] == 'battery_discharge':
                # Zou battery discharge commando sturen
                logging.info(f"Battery discharge gestart: {action['power']:.2f} kW voor {action['duration']} min")
                
            elif action['type'] == 'device_delay':
                # Zou device uitschakelen/pauzeren
                entity_id = action['entity_id']
                await self.ha.call_service('homeassistant', 'turn_off', entity_id)
                logging.info(f"Device {action['device']} gepauzeerd voor {action['delay_minutes']} min")
                
            elif action['type'] == 'notification':
                # Stuur notificatie naar HA
                await self.ha.call_service('notify', 'persistent_notification', data={
                    'message': action['message'],
                    'title': 'DAO Smart Optimization'
                })
                
        except Exception as e:
            logging.error(f"Response action fout: {e}")
    
    async def predict_high_loads(self, consumption_forecast: pd.DataFrame) -> List[Dict]:
        """Voorspel wanneer high load events gaan plaatsvinden"""
        predicted_events = []
        
        threshold = self.high_load_threshold
        
        for timestamp, row in consumption_forecast.iterrows():
            if row['predicted_consumption'] > threshold:
                predicted_events.append({
                    'timestamp': timestamp,
                    'predicted_load': row['predicted_consumption'],
                    'severity': 'high' if row['predicted_consumption'] > threshold * 1.5 else 'medium'
                })
        
        return predicted_events


class SmartDeviceScheduler:
    """Automatische scheduling van huishoudelijke apparaten"""
    
    def __init__(self, config: Config, ha_integration: HomeAssistantIntegration):
        self.config = config
        self.ha = ha_integration
        
    async def create_optimal_schedule(self, prices: pd.DataFrame, consumption_forecast: pd.DataFrame, 
                                    pv_forecast: pd.DataFrame) -> Dict[str, Any]:
        """Maak optimale schema voor apparaten"""
        try:
            schedule = {
                'devices': {},
                'energy_shifted': 0,
                'cost_savings': 0
            }
            
            # Get schedulable devices
            schedulable_devices = self._get_schedulable_devices()
            
            for device_name, device_config in schedulable_devices.items():
                if not device_config.get('enabled', True):
                    continue
                
                optimal_time = await self._find_optimal_time_slot(
                    device_config, prices, consumption_forecast, pv_forecast
                )
                
                if optimal_time:
                    schedule['devices'][device_name] = optimal_time
                    
                    # Calculate energy shifted
                    device_power = device_config.get('power', 1.0)  # kW
                    device_duration = device_config.get('duration', 1.0)  # hours
                    schedule['energy_shifted'] += device_power * device_duration
            
            # Calculate total cost savings
            if schedule['energy_shifted'] > 0:
                price_difference = prices['price'].max() - prices['price'].min()
                schedule['cost_savings'] = schedule['energy_shifted'] * price_difference * 0.5
            
            return schedule
            
        except Exception as e:
            logging.error(f"Device scheduling fout: {e}")
            return {'devices': {}, 'energy_shifted': 0, 'cost_savings': 0}
    
    def _get_schedulable_devices(self) -> Dict[str, Dict]:
        """Get lijst van apparaten die automatisch gepland kunnen worden"""
        return {
            'dishwasher': {
                'enabled': self.config.get(['smart_optimization', 'devices', 'dishwasher'], True),
                'entity': self.config.get(['devices', 'dishwasher', 'entity']),
                'power': self.config.get(['devices', 'dishwasher', 'power'], 1.5),  # kW
                'duration': self.config.get(['devices', 'dishwasher', 'duration'], 2.0),  # hours
                'earliest_start': 22,  # 22:00
                'latest_start': 6,     # 06:00
                'priority': 1
            },
            'washing_machine': {
                'enabled': self.config.get(['smart_optimization', 'devices', 'washing_machine'], True),
                'entity': self.config.get(['devices', 'washing_machine', 'entity']),
                'power': self.config.get(['devices', 'washing_machine', 'power'], 2.0),
                'duration': self.config.get(['devices', 'washing_machine', 'duration'], 1.5),
                'earliest_start': 23,
                'latest_start': 7,
                'priority': 2
            },
            'dryer': {
                'enabled': self.config.get(['smart_optimization', 'devices', 'dryer'], True),
                'entity': self.config.get(['devices', 'dryer', 'entity']),
                'power': self.config.get(['devices', 'dryer', 'power'], 2.5),
                'duration': self.config.get(['devices', 'dryer', 'duration'], 2.0),
                'earliest_start': 23,
                'latest_start': 8,
                'priority': 3
            }
        }
    
    async def _find_optimal_time_slot(self, device_config: Dict, prices: pd.DataFrame,
                                    consumption_forecast: pd.DataFrame, pv_forecast: pd.DataFrame) -> Optional[Dict]:
        """Vind optimale tijd slot voor apparaat"""
        try:
            if not device_config.get('entity'):
                return None
            
            # Check if device needs scheduling
            entity_state = await self.ha.get_entity_state(device_config['entity'])
            if not entity_state['success'] or entity_state['state'] != 'ready':
                return None
            
            power = device_config['power']
            duration = device_config['duration']
            earliest = device_config['earliest_start']
            latest = device_config['latest_start']
            
            best_time = None
            lowest_cost = float('inf')
            
            # Evaluate possible start times
            for hour in range(24):
                # Check if within allowed window
                if earliest <= latest:  # Same day
                    if not (earliest <= hour <= latest):
                        continue
                else:  # Spans midnight
                    if not (hour >= earliest or hour <= latest):
                        continue
                
                # Calculate cost for this time slot
                end_hour = (hour + duration) % 24
                if hour < end_hour:
                    time_slice = prices.iloc[hour:hour + int(duration)]
                else:
                    # Spans midnight
                    time_slice = pd.concat([prices.iloc[hour:], prices.iloc[:end_hour]])
                
                if len(time_slice) == 0:
                    continue
                
                # Calculate total cost
                avg_price = time_slice['price'].mean()
                total_cost = power * duration * avg_price
                
                # Bonus for PV overlap (if during day)
                if 8 <= hour <= 16 and not pv_forecast.empty:
                    pv_overlap = pv_forecast.iloc[hour:hour + int(duration)]['predicted_pv'].sum()
                    # Reduce cost if PV is available
                    total_cost *= (1 - min(0.5, pv_overlap / (power * duration)))
                
                if total_cost < lowest_cost:
                    lowest_cost = total_cost
                    best_time = {
                        'start_time': f"{hour:02d}:00",
                        'duration_hours': duration,
                        'estimated_cost': round(total_cost, 3),
                        'power_kw': power
                    }
            
            return best_time
            
        except Exception as e:
            logging.error(f"Optimal time finding fout: {e}")
            return None


class AdaptiveBatteryManager:
    """Geavanceerd batterij management voor kosten optimalisatie"""
    
    def __init__(self, config: Config, ha_integration: HomeAssistantIntegration):
        self.config = config
        self.ha = ha_integration
        
    async def optimize_battery_schedule(self, prices: pd.DataFrame, consumption_forecast: pd.DataFrame,
                                      pv_forecast: pd.DataFrame) -> Dict[str, Any]:
        """Optimaliseer batterij schema voor maximale kosten besparing"""
        try:
            if not self.config.get(['smart_optimization', 'adaptive_battery', 'enabled'], True):
                return {'strategy': 'disabled'}
            
            batteries = self.config.get(['battery'], [])
            if not batteries:
                return {'strategy': 'no_batteries'}
            
            strategy = {
                'charge_periods': [],
                'discharge_periods': [],
                'soc_targets': {},
                'estimated_savings': 0
            }
            
            # Analyse price patterns
            price_analysis = self._analyze_prices(prices)
            
            # Voor elke batterij
            total_savings = 0
            for i, battery in enumerate(batteries):
                battery_strategy = await self._optimize_single_battery(
                    battery, prices, consumption_forecast, pv_forecast, price_analysis
                )
                
                strategy['soc_targets'][f'battery_{i}'] = battery_strategy['target_soc']
                strategy['charge_periods'].extend(battery_strategy['charge_periods'])
                strategy['discharge_periods'].extend(battery_strategy['discharge_periods'])
                total_savings += battery_strategy['estimated_savings']
            
            strategy['estimated_savings'] = total_savings
            
            return strategy
            
        except Exception as e:
            logging.error(f"Battery optimization fout: {e}")
            return {'strategy': 'error', 'error': str(e)}
    
    def _analyze_prices(self, prices: pd.DataFrame) -> Dict[str, Any]:
        """Analyseer prijspatronen voor batterij optimalisatie"""
        analysis = {
            'min_price': prices['price'].min(),
            'max_price': prices['price'].max(),
            'avg_price': prices['price'].mean(),
            'price_spread': prices['price'].max() - prices['price'].min(),
            'cheap_hours': [],
            'expensive_hours': []
        }
        
        # Determine cheap and expensive periods
        price_threshold_low = analysis['min_price'] + (analysis['price_spread'] * 0.3)
        price_threshold_high = analysis['max_price'] - (analysis['price_spread'] * 0.3)
        
        for idx, row in prices.iterrows():
            hour = idx.hour if hasattr(idx, 'hour') else idx
            
            if row['price'] <= price_threshold_low:
                analysis['cheap_hours'].append(hour)
            elif row['price'] >= price_threshold_high:
                analysis['expensive_hours'].append(hour)
        
        return analysis
    
    async def _optimize_single_battery(self, battery_config: Dict, prices: pd.DataFrame,
                                     consumption_forecast: pd.DataFrame, pv_forecast: pd.DataFrame,
                                     price_analysis: Dict) -> Dict[str, Any]:
        """Optimaliseer strategie voor één batterij"""
        try:
            capacity = battery_config.get('capacity', 10)  # kWh
            max_power = battery_config.get('max_power', 5)  # kW
            efficiency = battery_config.get('efficiency', 95) / 100
            current_soc = await self._get_battery_soc(battery_config.get('entity', ''))
            
            strategy = {
                'target_soc': current_soc,
                'charge_periods': [],
                'discharge_periods': [],
                'estimated_savings': 0
            }
            
            # Simple strategy: charge during cheap hours, discharge during expensive hours
            price_spread = price_analysis['price_spread']
            
            if price_spread > 0.05:  # Significant price difference (5 cent/kWh)
                # Charge periods
                for hour in price_analysis['cheap_hours']:
                    if current_soc < 0.9:  # Don't overcharge
                        charge_power = min(max_power, (capacity * (0.9 - current_soc)))
                        if charge_power > 0:
                            strategy['charge_periods'].append({
                                'hour': hour,
                                'power': charge_power,
                                'duration': 1
                            })
                            current_soc += (charge_power * efficiency / capacity)
                
                # Discharge periods
                for hour in price_analysis['expensive_hours']:
                    if current_soc > 0.2:  # Keep minimum charge
                        discharge_power = min(max_power, capacity * (current_soc - 0.2))
                        if discharge_power > 0:
                            strategy['discharge_periods'].append({
                                'hour': hour,
                                'power': discharge_power,
                                'duration': 1
                            })
                            current_soc -= (discharge_power / capacity)
                
                # Calculate estimated savings
                total_energy_cycled = sum(p['power'] for p in strategy['charge_periods'])
                strategy['estimated_savings'] = total_energy_cycled * price_spread * efficiency
            
            strategy['target_soc'] = current_soc
            
            return strategy
            
        except Exception as e:
            logging.error(f"Single battery optimization fout: {e}")
            return {'target_soc': 0.5, 'charge_periods': [], 'discharge_periods': [], 'estimated_savings': 0}
    
    async def _get_battery_soc(self, entity_id: str) -> float:
        """Haal huidige batterij SOC op"""
        if not entity_id:
            return 0.5  # Default 50%
        
        try:
            state = await self.ha.get_entity_state(entity_id)
            if state['success']:
                soc = float(state['state']) / 100  # Convert percentage to decimal
                return max(0.1, min(0.9, soc))  # Clamp between 10% and 90%
        except Exception as e:
            logging.error(f"Battery SOC ophalen fout: {e}")
        
        return 0.5  # Default fallback