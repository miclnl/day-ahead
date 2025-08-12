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
import json
import aiohttp

# Statistical replacements for ML libraries
from collections import defaultdict
from statistics import mean, median, stdev

# Fallback ML implementation (no external dependencies)
class FallbackRandomForestRegressor:
    """Fallback Random Forest implementation zonder externe ML libraries"""

    def __init__(self, n_estimators=100, max_depth=15, min_samples_split=5, min_samples_leaf=2, random_state=42, n_jobs=-1):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.feature_importances_ = None
        self.is_fitted = False

    def fit(self, X, y):
        """Train een eenvoudig statistisch model als fallback"""
        if len(X) < 10:
            # Te weinig data, gebruik gemiddelde
            self.prediction_value = y.mean()
            self.feature_importances_ = [1.0/len(X.columns)] * len(X.columns)
        else:
            # Eenvoudige lineaire combinatie van features
            self.prediction_value = y.mean()
            # Bereken feature importance gebaseerd op correlatie
            correlations = []
            for col in X.columns:
                try:
                    corr = abs(X[col].corr(y))
                    correlations.append(corr if not pd.isna(corr) else 0.0)
                except:
                    correlations.append(0.0)

            # Normaliseer feature importances
            total_corr = sum(correlations)
            if total_corr > 0:
                self.feature_importances_ = [c/total_corr for c in correlations]
            else:
                self.feature_importances_ = [1.0/len(X.columns)] * len(X.columns)

        self.is_fitted = True
        return self

    def predict(self, X):
        """Voorspel met eenvoudig statistisch model"""
        if not self.is_fitted:
            return [self.prediction_value] * len(X)

        predictions = []
        for _, row in X.iterrows():
            # Eenvoudige voorspelling gebaseerd op gemiddelde + feature effecten
            pred = self.prediction_value
            for i, col in enumerate(X.columns):
                if i < len(self.feature_importances_):
                    # Voeg feature effect toe (vereenvoudigd)
                    feature_val = row[col]
                    if pd.notna(feature_val):
                        pred += (feature_val - X[col].mean()) * self.feature_importances_[i] * 0.1

            predictions.append(max(0, pred))  # Verbruik kan niet negatief zijn

        return predictions

# Try to import real RandomForestRegressor, fallback to our implementation
try:
    from sklearn.ensemble import RandomForestRegressor
    logging.info("✅ Scikit-learn RandomForestRegressor beschikbaar - gebruik echte ML algoritmes")
    USE_REAL_ML = True
except ImportError:
    RandomForestRegressor = FallbackRandomForestRegressor
    logging.info("⚠️ Scikit-learn niet beschikbaar - gebruik fallback statistische implementatie")
    USE_REAL_ML = False
except Exception as e:
    # Fallback bij andere errors (zoals SIGILL op Raspberry Pi)
    RandomForestRegressor = FallbackRandomForestRegressor
    logging.warning(f"⚠️ Scikit-learn error ({e}) - gebruik fallback implementatie")
    USE_REAL_ML = False

# Optional cloud AI support
try:
    import openai
    CLOUD_AI_AVAILABLE = True
except ImportError:
    CLOUD_AI_AVAILABLE = False

# Dummy classes as fallback
class DummyHighLoadDetector:
    """Dummy class als fallback voor HighLoadDetector"""
    async def predict_high_loads(self, consumption_forecast):
        return []

    async def start_monitoring(self):
        pass

class DummySmartDeviceScheduler:
    """Dummy class als fallback voor SmartDeviceScheduler"""
    async def create_optimal_schedule(self, prices, consumption_forecast, pv_forecast):
        return {'status': 'dummy', 'devices': []}

class DummyAdaptiveBatteryManager:
    """Dummy class als fallback voor AdaptiveBatteryManager"""
    async def optimize_battery_schedule(self, prices, consumption_forecast, pv_forecast):
        return {'status': 'dummy', 'strategy': 'basic'}

# Import local modules
try:
    from dao.prog.da_config import Config
    from dao.prog.da_ha_integration import HomeAssistantIntegration
except ImportError:
    # Fallback for testing
    Config = dict
    HomeAssistantIntegration = object


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

        # Statistical prediction models (ML-free)
        self.consumption_stats = defaultdict(list)
        self.pv_stats = defaultdict(list)

        # Initialize helper classes later to avoid NameError
        self.load_detector = None
        self.device_scheduler = None
        self.battery_manager = None

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

        # Initialize helper classes after they are defined
        self._initialize_helper_classes()

    def _initialize_helper_classes(self):
        """Initialize helper classes after they are defined"""
        try:
            self.load_detector = HighLoadDetector(self.config, self.ha, lightweight=self.is_pi)
            self.device_scheduler = SmartDeviceScheduler(self.config, self.ha)
            self.battery_manager = AdaptiveBatteryManager(self.config, self.ha)
            logging.debug("Helper classes geïnitialiseerd")
        except Exception as e:
            logging.error(f"Fout bij initialiseren helper classes: {e}")
            # Create dummy classes as fallback
            self.load_detector = DummyHighLoadDetector()
            self.device_scheduler = DummySmartDeviceScheduler()
            self.battery_manager = DummyAdaptiveBatteryManager()

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

            # Emergency Discharge Price Analysis Settings
            'emergency_discharge_price_threshold': self.config.get(['smart_optimization', 'emergency_discharge', 'price_threshold'], 0.02),
            'emergency_discharge_time_threshold': self.config.get(['smart_optimization', 'emergency_discharge', 'time_threshold'], 2),
            'emergency_discharge_lookahead_hours': self.config.get(['smart_optimization', 'emergency_discharge', 'lookahead_hours'], 4),

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
            if USE_REAL_ML:
                logging.info("🎯 Training consumption model met echte Random Forest (scikit-learn)")
            else:
                logging.info("📊 Training consumption model met statistische fallback implementatie")

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

    def _train_pv_model(self, pv_data: pd.DataFrame, weather_data: pd.DataFrame) -> RandomForestRegressor:
        """Train PV production prediction model"""
        try:
            if USE_REAL_ML:
                logging.info("🎯 Training PV model met echte Random Forest (scikit-learn)")
            else:
                logging.info("📊 Training PV model met statistische fallback implementatie")

            # Feature engineering voor PV voorspelling
            features = pd.DataFrame()

            # Time features
            features['hour'] = pv_data.index.hour
            features['day_of_week'] = pv_data.index.dayofweek
            features['month'] = pv_data.index.month
            features['day_of_year'] = pv_data.index.dayofyear

            # Solar position features (vereenvoudigd)
            features['solar_noon'] = abs(features['hour'] - 12)  # Afstand tot zonne-noon
            features['is_daylight'] = ((features['hour'] >= 6) & (features['hour'] <= 20)).astype(int)

            # Weather features (if available)
            if not weather_data.empty:
                # Resample weather to match PV frequency
                weather_resampled = weather_data.resample('1H').mean().reindex(pv_data.index, method='nearest')
                features['temperature'] = weather_resampled.get('temperature', 15)
                features['solar_radiation'] = weather_resampled.get('solar_radiation', 0)
                features['cloud_cover'] = weather_resampled.get('cloud_cover', 0)
                features['wind_speed'] = weather_resampled.get('wind_speed', 0)
            else:
                # Default values
                features['temperature'] = 15
                features['solar_radiation'] = 0
                features['cloud_cover'] = 0
                features['wind_speed'] = 0

            # Historical PV features (lag features)
            if 'production' in pv_data.columns:
                features['pv_lag_1h'] = pv_data['production'].shift(1)
                features['pv_lag_24h'] = pv_data['production'].shift(24)
                features['pv_avg_24h'] = pv_data['production'].rolling(24, min_periods=1).mean()
                features['pv_avg_7d'] = pv_data['production'].rolling(168, min_periods=1).mean()
            else:
                # Fallback als 'production' kolom ontbreekt
                features['pv_lag_1h'] = 0
                features['pv_lag_24h'] = 0
                features['pv_avg_24h'] = 0
                features['pv_avg_7d'] = 0

            # Seasonal features
            features['season'] = ((pv_data.index.month % 12 + 3) // 3).map(
                {1: 0, 2: 1, 3: 2, 4: 3}  # Winter=0, Spring=1, Summer=2, Autumn=3
            )

            # Drop rows with NaN values
            features = features.dropna()

            # Bepaal target kolom
            if 'production' in pv_data.columns:
                target = pv_data.loc[features.index, 'production']
            else:
                # Fallback: gebruik dummy data
                target = pd.Series([0.5] * len(features), index=features.index)

            if len(features) < 20:
                logging.warning("Onvoldoende PV data voor model training")
                return None

            # Train model
            model = RandomForestRegressor(
                n_estimators=50,  # Minder estimators voor PV (meestal eenvoudiger patroon)
                max_depth=10,
                min_samples_split=3,
                min_samples_leaf=1,
                random_state=42,
                n_jobs=-1
            )

            model.fit(features, target)

            # Log feature importance
            feature_importance = dict(zip(features.columns, model.feature_importances_))
            logging.info(f"PV model top features: {sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]}")

            return model

        except Exception as e:
            logging.error(f"PV model training fout: {e}")
            return None

    async def _predict_consumption(self, future_index: pd.DatetimeIndex) -> pd.DataFrame:
        """Voorspel toekomstig verbruik"""
        try:
            if self.consumption_model is None:
                # Geen model beschikbaar - gebruik baseline uit config
                current_baseload = self.config.get(['baseload'], {}).get('current', 0.5)
                logging.warning("⚠️ Geen consumption model beschikbaar - gebruik baseline verbruik")
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
            features['temperature'] = 15  # Placeholder - zou echte weer data moeten zijn
            features['solar_radiation'] = 0
            features['wind_speed'] = 0

            # Voorspel met model
            predictions = self.consumption_model.predict(features)

            return pd.DataFrame({
                'predicted_consumption': predictions
            }, index=future_index)

        except Exception as e:
            logging.error(f"❌ Consumption prediction error: {e}")
            # Fallback: gebruik baseline verbruik
            current_baseload = self.config.get(['baseload'], {}).get('current', 0.5)
            return pd.DataFrame({
                'predicted_consumption': [current_baseload] * len(future_index)
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
        """Haal historische verbruik data op"""
        try:
            # Probeer echte verbruik data op te halen
            from dao.prog.da_report import Report
            report = Report()
            consumption_data = report.get_consumption_data(start_time, end_time)

            if consumption_data is not None and not consumption_data.empty:
                logging.info("✅ Echte verbruik data opgehaald uit database")
                return consumption_data
            else:
                logging.warning("⚠️ Geen echte verbruik data beschikbaar in database")
                return pd.DataFrame()  # Lege DataFrame

        except Exception as e:
            logging.error(f"❌ Fout bij ophalen verbruik data: {e}")
            return pd.DataFrame()  # Lege DataFrame

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

    def __init__(self, config: Config, ha_integration: HomeAssistantIntegration, lightweight: bool):
        self.config = config
        self.ha = ha_integration
        self.baseline_consumption = 0.5  # kW
        self.high_load_threshold = 2.0  # kW
        self.detection_active = False
        self.lightweight = lightweight

    async def start_monitoring(self):
        """Start high load monitoring"""
        if self.detection_active:
            return

        self.detection_active = True
        logging.info("High load detection gestart - monitoring elke 30 seconden")

        # Start monitoring in background
        asyncio.create_task(self._monitor_consumption())

    async def _get_current_consumption(self) -> float:
        """Haal huidig verbruik op uit Home Assistant"""
        try:
            # Probeer eerst de geconfigureerde consumption entity
            consumption_entity = self.config.get(['consumption', 'entity'])
            if consumption_entity:
                current_state = await self.ha.get_entity_state(consumption_entity)
                if current_state.get('success'):
                    return float(current_state['state'])

            # Fallback: gebruik gemiddelde verbruik uit config
            baseline_consumption = self.config.get(['baseload', 'current'], 0.5)
            logging.debug(f"Using fallback consumption: {baseline_consumption} kW")
            return baseline_consumption

        except Exception as e:
            logging.error(f"Error getting current consumption: {e}")
            # Return baseline als fallback
            return self.config.get(['baseload', 'current'], 0.5)

    async def _monitor_consumption(self):
        """Monitor verbruik elke 30 seconden voor high load detectie"""
        while self.detection_active:
            try:
                # Haal huidig verbruik op
                current_consumption = await self._get_current_consumption()

                if current_consumption > self.high_load_threshold:
                    logging.warning(f"🚨 HIGH LOAD DETECTED: {current_consumption:.2f} kW (threshold: {self.high_load_threshold:.2f} kW)")

                    # Trigger immediate response
                    await self._handle_high_load_event(current_consumption)

                # Wacht 30 seconden voor volgende check
                await asyncio.sleep(30)

            except Exception as e:
                logging.error(f"High load monitoring error: {e}")
                await asyncio.sleep(30)  # Continue monitoring ondanks errors

    async def _handle_high_load_event(self, consumption: float):
        """Handle high load event met prijsafweging en apparaat-specifieke optimalisatie"""
        try:
            logging.warning(f"🚨 HIGH LOAD EVENT: {consumption:.2f} kW - Context analyse en optimalisatie")

            # Haal huidige prijzen op voor de komende uren
            current_prices = await self._get_current_and_future_prices()
            if not current_prices:
                logging.warning("⚠️ Geen echte prijsdata beschikbaar - kan geen prijsafweging maken")
                logging.info("🔄 Fallback naar basis emergency discharge zonder prijsoptimalisatie")
                await self._execute_emergency_discharge(consumption)
                return

            # Bereken prijsafweging
            price_analysis = await self._analyze_discharge_timing(consumption, current_prices)

            if price_analysis['should_discharge_now']:
                logging.info(f"✅ PRIJS GUNSTIG: Ontladen nu (prijs: €{price_analysis['current_price']:.3f}/kWh)")
                await self._execute_emergency_discharge(consumption)
            else:
                logging.info(f"⏰ PRIJS ONGUNSTIG: Wachten tot {price_analysis['optimal_time']} (prijs: €{price_analysis['optimal_price']:.3f}/kWh)")
                await self._execute_delayed_response(consumption, price_analysis)

        except Exception as e:
            logging.error(f"❌ High load event handling error: {e}")
            # Fallback: direct ontladen bij errors
            await self._execute_emergency_discharge(consumption)

    async def _get_current_and_future_prices(self) -> Optional[pd.DataFrame]:
        """Haal huidige en toekomstige prijzen op voor de komende 4 uur"""
        try:
            from datetime import datetime, timedelta

            now = datetime.now()
            end_time = now + timedelta(hours=4)

            # Probeer prijsdata op te halen uit database
            try:
                from dao.prog.da_report import Report
                report = Report()
                price_data = report.get_price_data(now, end_time)

                if price_data is not None and not price_data.empty and 'da_ex' in price_data.columns:
                    logging.info("✅ Echte prijsdata opgehaald voor emergency discharge analyse")
                    return price_data[['da_ex']].rename(columns={'da_ex': 'price'})
                else:
                    logging.warning("⚠️ Geen echte prijsdata beschikbaar voor emergency discharge analyse")
                    return None

            except Exception as e:
                logging.error(f"❌ Fout bij ophalen prijsdata: {e}")
                return None

        except Exception as e:
            logging.error(f"❌ Error getting prices for emergency discharge: {e}")
            return None

    async def _analyze_discharge_timing(self, consumption: float, prices: pd.DataFrame) -> Dict[str, Any]:
        """Analyseer of het gunstig is om nu te ontladen of te wachten"""
        try:
            current_price = prices.iloc[0]['price']
            future_prices = prices.iloc[1:6]  # Komende 5 uur

            # Bereken gemiddelde prijs in de komende uren
            avg_future_price = future_prices['price'].mean()
            min_future_price = future_prices['price'].min()

            # Zoek het beste moment in de komende uren
            optimal_idx = future_prices['price'].idxmin()
            optimal_time = optimal_idx.strftime('%H:%M')
            optimal_price = min_future_price

            # Prijsafweging logica
            price_threshold = 0.02  # €0.02/kWh verschil
            time_threshold = 2  # 2 uur wachten

            # Bepaal of ontladen nu gunstig is
            price_difference = current_price - min_future_price
            hours_until_optimal = (optimal_idx - prices.index[0]).total_seconds() / 3600

            should_discharge_now = (
                price_difference < price_threshold or  # Prijsverschil te klein
                hours_until_optimal > time_threshold or  # Te lang wachten
                current_price < avg_future_price * 0.9  # Huidige prijs is al laag
            )

            analysis = {
                'should_discharge_now': should_discharge_now,
                'current_price': current_price,
                'avg_future_price': avg_future_price,
                'min_future_price': min_future_price,
                'optimal_time': optimal_time,
                'optimal_price': optimal_price,
                'price_difference': price_difference,
                'hours_until_optimal': hours_until_optimal,
                'reasoning': self._get_discharge_reasoning(should_discharge_now, price_difference, hours_until_optimal)
            }

            logging.info(f"�� PRIJSANALYSE: Nu: €{current_price:.3f}, Beste: €{optimal_price:.3f} om {optimal_time}, Verschil: €{price_difference:.3f}")

            return analysis

        except Exception as e:
            logging.error(f"Error analyzing discharge timing: {e}")
            # Fallback: ontladen nu bij errors
            return {
                'should_discharge_now': True,
                'current_price': 0.20,
                'reasoning': 'Error in price analysis - fallback to immediate discharge'
            }

    def _get_discharge_reasoning(self, should_discharge_now: bool, price_difference: float, hours_until_optimal: float) -> str:
        """Genereer uitleg voor discharge beslissing"""
        if should_discharge_now:
            if price_difference < 0.02:
                return "Prijsverschil te klein om te wachten"
            elif hours_until_optimal > 2:
                return "Te lang wachten voor kleine besparing"
            else:
                return "Huidige prijs is al gunstig"
        else:
            return f"Wachten tot beste prijs (€{price_difference:.3f} besparing over {hours_until_optimal:.1f} uur)"

    async def _execute_emergency_discharge(self, consumption: float):
        """Execute emergency discharge action"""
        try:
            logging.warning(f"🚨 EMERGENCY DISCHARGE: Ontladen nu (consumptie: {consumption:.2f} kW)")

            # Haal huidige batterij SOC op
            battery_entity = self.config.get(['battery', 'entity'])
            if battery_entity:
                current_soc = await self._get_battery_soc(battery_entity)
                logging.info(f"Huidige SOC: {current_soc:.2f}")

                # Bepaal maximale ontlaad capaciteit
                batteries = self.config.get(['battery'], [])
                total_available_capacity = sum(battery.get('capacity', 0) * (0.7 - battery.get('min_soc', 0.2)) for battery in batteries)
                logging.info(f"Totale beschikbare ontlaad capaciteit: {total_available_capacity:.2f} kWh")

                # Ontlaad tot SOC van 0.2
                discharge_power = min(consumption, total_available_capacity)
                logging.info(f"Ontlaad: {discharge_power:.2f} kW")

                # Voeg ontlaad actie toe aan planning
                self.ha.add_action({
                    'type': 'battery_discharge',
                    'power': discharge_power,
                    'duration': 1 # Ontlaad voor 1 uur
                })

                # Stuur notificatie naar HA
                await self.ha.call_service('notify', 'persistent_notification', data={
                    'message': f"EMERGENCY DISCHARGE: Ontladen nu {discharge_power:.2f} kW om hoge consumptie te compenseren.",
                    'title': 'DAO Smart Optimization'
                })

                logging.info("EMERGENCY DISCHARGE actie uitgevoerd.")

        except Exception as e:
            logging.error(f"Emergency discharge execution error: {e}")

    async def _execute_delayed_response(self, consumption: float, price_analysis: Dict):
        """Execute delayed response action (e.g., wait for better price)"""
        try:
            logging.info(f"⏰ WACHTEN OP BETERE PRIJS: Ontladen niet nu (consumptie: {consumption:.2f} kW), wachten tot {price_analysis['optimal_time']}")

            # Voeg wachten actie toe aan planning
            self.ha.add_action({
                'type': 'device_delay',
                'device': 'heat_pump', # Stel een apparaat in dat uitgesteld kan worden
                'entity_id': self.config.get(['devices', 'heat_pump', 'entity']),
                'delay_minutes': 60 # Wacht 1 uur
            })

            # Stuur notificatie naar HA
            await self.ha.call_service('notify', 'persistent_notification', data={
                'message': f"EMERGENCY DISCHARGE: Ontladen niet nu, wachten tot {price_analysis['optimal_time']} om {price_analysis['optimal_price']:.3f} €/kWh.",
                'title': 'DAO Smart Optimization'
            })

            logging.info("EMERGENCY DISCHARGE wachten actie uitgevoerd.")

        except Exception as e:
            logging.error(f"Delayed response execution error: {e}")

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