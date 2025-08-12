"""
Statistical Consumption Predictor voor Day Ahead Optimizer.
Vervangt ML-based prediction met intelligente statistische analyse.
Geen ML dependencies - pure pandas/numpy statistiek.
"""

import datetime as dt
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import math

# HA statistics client (synchronous HTTP)
try:
    from .ha_client import get_energy_preferences, get_statistics_period
except Exception:
    try:
        from ha_client import get_energy_preferences, get_statistics_period  # type: ignore
    except Exception:
        get_energy_preferences = None  # type: ignore
        get_statistics_period = None  # type: ignore

# Eenvoudige in-memory cache voor HA statistics (reduceert API load)
_STAT_CACHE: Dict[str, Tuple[float, Any]] = {}
_CACHE_TTL_S = 300  # 5 minuten

def _cache_key(statistic_id: str, start_iso: str, period: str) -> str:
    return f"{statistic_id}|{start_iso}|{period}"

def _get_cached_or_fetch(config, statistic_id: str, start: dt.datetime, period: str = 'hour') -> Any:
    if get_statistics_period is None:
        return None
    try:
        key = _cache_key(statistic_id, start.replace(microsecond=0).isoformat(), period)
        now = dt.datetime.now().timestamp()
        entry = _STAT_CACHE.get(key)
        if entry and (now - entry[0]) < _CACHE_TTL_S:
            return entry[1]
        data = get_statistics_period(config, statistic_id, start=start, period=period)
        _STAT_CACHE[key] = (now, data)
        return data
    except Exception:
        return None


class StatisticalPredictor:
    """
    Statistical consumption prediction using historical patterns, weather correlation,
    and seasonal adjustments. Replaces ML models with robust statistical methods.
    """

    def __init__(self, da_calc_instance):
        """Initialize with reference to main DaCalc instance"""
        self.da_calc = da_calc_instance
        self.db_da = da_calc_instance.db_da
        self.config = da_calc_instance.config

        # Configuration
        self.prediction_window = 48  # hours ahead
        self.history_days = 30      # days of history for patterns
        self.seasonal_window = 90   # days for seasonal analysis

        # Pattern storage
        self.hourly_patterns = {}
        self.weekday_patterns = {}
        self.seasonal_patterns = {}
        self.weather_correlations = {}

        # Confidence thresholds
        self.min_data_points = 7  # minimum days of data needed

        logging.info("Statistical predictor initialized")

    def predict_consumption(
        self,
        start_time: dt.datetime,
        hours_ahead: int = 48,
        weather_data: Dict = None
    ) -> pd.DataFrame:
        """
        Main prediction method using statistical analysis

        Args:
            start_time: Start time for prediction
            hours_ahead: Number of hours to predict
            weather_data: Optional weather forecast data

        Returns:
            DataFrame with predicted consumption per hour
        """
        logging.info(f"Starting statistical consumption prediction for {hours_ahead} hours")

        try:
            # 1. Get historical consumption data
            historical_data = self._get_historical_consumption(start_time)
            if historical_data is None or len(historical_data) < self.min_data_points * 24:
                logging.warning("Insufficient historical data for reliable prediction")
                return self._fallback_prediction(start_time, hours_ahead)

            # 2. Analyze patterns
            patterns = self._analyze_consumption_patterns(historical_data)

            # 3. Generate predictions
            predictions = []
            for hour_offset in range(hours_ahead):
                prediction_time = start_time + dt.timedelta(hours=hour_offset)

                # Base prediction from patterns
                base_prediction = self._get_pattern_based_prediction(
                    prediction_time, patterns, historical_data
                )

                # Weather adjustment
                if weather_data:
                    weather_adjusted = self._apply_weather_correction(
                        base_prediction, prediction_time, weather_data, historical_data
                    )
                else:
                    weather_adjusted = base_prediction

                # Confidence scoring
                confidence = self._calculate_prediction_confidence(
                    prediction_time, historical_data, patterns
                )

                predictions.append({
                    'datetime': prediction_time,
                    'predicted_consumption': max(0, weather_adjusted),  # Ensure positive
                    'base_pattern': base_prediction,
                    'confidence': confidence
                })

            results_df = pd.DataFrame(predictions)
            results_df.set_index('datetime', inplace=True)

            logging.info(f"Generated {len(predictions)} consumption predictions")
            return results_df

        except Exception as e:
            logging.error(f"Error in statistical prediction: {e}")
            return self._fallback_prediction(start_time, hours_ahead)

    def _get_historical_consumption(self, reference_time: dt.datetime) -> pd.DataFrame:
        """Retrieve historical consumption data from HA Statistics API (external mode) or DB (internal)."""
        try:
            mode = None
            try:
                mode = str(self.config.get(["energy", "storage_mode"], None, "external")).lower()
            except Exception:
                mode = "external"

            end_time = reference_time
            start_time = end_time - dt.timedelta(days=self.history_days)

            if mode == "external" and get_statistics_period is not None:
                # Prefer HA Energy Dashboard statistics
                cons_stat = None
                try:
                    prefs = get_energy_preferences(self.config) if get_energy_preferences else None
                    if isinstance(prefs, dict):
                        sources = prefs.get('energy_sources', []) or []
                        for src in sources:
                            for flow in src.get('flow_from', []) or []:
                                sid = flow.get('stat_energy_from')
                                if isinstance(sid, str):
                                    cons_stat = sid
                                    break
                            if cons_stat:
                                break
                except Exception:
                    cons_stat = None
                # Fallback naar handmatige entity
                if not cons_stat:
                    try:
                        cons_stat = self.config.get(['energy', 'consumption_entity'])
                    except Exception:
                        cons_stat = None
                if not cons_stat:
                    raise RuntimeError("Geen consumption statistic/entity geconfigureerd in HA")

                # Haal uur-statistieken op
                series = _get_cached_or_fetch(self.config, cons_stat, start=start_time, period='hour')
                if not isinstance(series, list) or len(series) == 0:
                    raise RuntimeError("Geen statistics data ontvangen van HA")
                # Home Assistant retourneert list of list; pak binnenste lijst indien nodig
                items = series[0] if isinstance(series[0], list) else series
                records: List[Dict[str, Any]] = []
                for row in items:
                    try:
                        ts_str = row.get('start') or row.get('start_time')
                        val = row.get('sum') or row.get('change')
                        if ts_str is None or not isinstance(val, (int, float)):
                            continue
                        ts = pd.to_datetime(ts_str)
                        records.append({
                            'datetime': ts,
                            'consumption': float(val)
                        })
                    except Exception:
                        continue
                if not records:
                    raise RuntimeError("Lege statistics lijst van HA")
                df = pd.DataFrame.from_records(records)
                df.sort_values(by='datetime', inplace=True)
                df.set_index('datetime', inplace=True)
                return df

            # INTERNAL modus: lees uit eigen database
            query = f"""
                SELECT datetime, consumption
                FROM energy_data
                WHERE datetime BETWEEN '{start_time}' AND '{end_time}'
                ORDER BY datetime
            """
            df = pd.read_sql(query, self.db_da.connection)
            df['datetime'] = pd.to_datetime(df['datetime'])
            df.set_index('datetime', inplace=True)
            return df

        except Exception as e:
            logging.error(f"Error retrieving historical consumption: {e}")
            return None

    def _analyze_consumption_patterns(self, data: pd.DataFrame) -> Dict:
        """Analyze historical data for patterns"""
        patterns = {}

        # Add time-based features
        data_analysis = data.copy()
        data_analysis['hour'] = data_analysis.index.hour
        data_analysis['weekday'] = data_analysis.index.weekday
        data_analysis['month'] = data_analysis.index.month
        data_analysis['week_of_year'] = data_analysis.index.isocalendar().week

        # 1. Hourly patterns (24-hour cycle)
        patterns['hourly'] = data_analysis.groupby('hour')['consumption'].agg([
            'mean', 'std', 'median', 'quantile'
        ]).to_dict()

        # 2. Weekday patterns (0=Monday to 6=Sunday)
        patterns['weekday'] = data_analysis.groupby('weekday')['consumption'].agg([
            'mean', 'std', 'median'
        ]).to_dict()

        # 3. Monthly seasonal patterns
        patterns['seasonal'] = data_analysis.groupby('month')['consumption'].agg([
            'mean', 'std', 'median'
        ]).to_dict()

        # 4. Weekly patterns (work week vs weekend)
        data_analysis['is_weekend'] = data_analysis['weekday'].isin([5, 6])
        patterns['weekend'] = data_analysis.groupby('is_weekend')['consumption'].agg([
            'mean', 'std', 'median'
        ]).to_dict()

        # 5. Combined patterns (hour + weekday interaction)
        patterns['hour_weekday'] = data_analysis.groupby(['hour', 'weekday'])['consumption'].mean().to_dict()

        logging.info("Consumption patterns analyzed successfully")
        return patterns

    def _get_pattern_based_prediction(
        self,
        prediction_time: dt.datetime,
        patterns: Dict,
        historical_data: pd.DataFrame
    ) -> float:
        """Generate base prediction from statistical patterns"""

        hour = prediction_time.hour
        weekday = prediction_time.weekday
        month = prediction_time.month
        is_weekend = weekday in [5, 6]

        # Weight different pattern sources
        weights = {
            'hourly': 0.4,
            'weekday': 0.3,
            'seasonal': 0.2,
            'combined': 0.1
        }

        predictions = {}

        # Hourly pattern
        if hour in patterns['hourly']['mean']:
            predictions['hourly'] = patterns['hourly']['mean'][hour]
        else:
            predictions['hourly'] = historical_data['consumption'].mean()

        # Weekday pattern
        if weekday in patterns['weekday']['mean']:
            predictions['weekday'] = patterns['weekday']['mean'][weekday]
        else:
            predictions['weekday'] = historical_data['consumption'].mean()

        # Seasonal pattern
        if month in patterns['seasonal']['mean']:
            predictions['seasonal'] = patterns['seasonal']['mean'][month]
        else:
            predictions['seasonal'] = historical_data['consumption'].mean()

        # Combined hour+weekday pattern
        combined_key = (hour, weekday)
        if combined_key in patterns['hour_weekday']:
            predictions['combined'] = patterns['hour_weekday'][combined_key]
        else:
            predictions['combined'] = (predictions['hourly'] + predictions['weekday']) / 2

        # Weighted combination
        weighted_prediction = sum(
            weights[pattern] * predictions[pattern]
            for pattern in weights
            if pattern in predictions
        )

        return weighted_prediction

    def _apply_weather_correction(
        self,
        base_prediction: float,
        prediction_time: dt.datetime,
        weather_data: Dict,
        historical_data: pd.DataFrame
    ) -> float:
        """Apply weather-based corrections to base prediction"""

        corrected_prediction = base_prediction

        try:
            # Temperature correlation (heating/cooling)
            if 'temperature' in weather_data:
                temp = weather_data['temperature']
                temp_correction = self._calculate_temperature_correction(
                    temp, prediction_time, historical_data
                )
                corrected_prediction *= temp_correction

            # Solar/cloud cover correlation (reduced consumption during high solar)
            if 'cloud_cover' in weather_data and 'solar_capacity' in self.config:
                cloud_cover = weather_data['cloud_cover']
                solar_correction = self._calculate_solar_correction(
                    cloud_cover, prediction_time
                )
                corrected_prediction *= solar_correction

        except Exception as e:
            logging.warning(f"Weather correction failed: {e}")

        return corrected_prediction

    def _calculate_temperature_correction(
        self,
        temperature: float,
        time: dt.datetime,
        historical_data: pd.DataFrame
    ) -> float:
        """Calculate temperature-based consumption adjustment"""

        # Comfort temperature ranges
        comfort_min, comfort_max = 18, 22

        if comfort_min <= temperature <= comfort_max:
            return 1.0  # No adjustment needed

        # Heating season (Oct-Apr in Northern Hemisphere)
        is_heating_season = time.month in [10, 11, 12, 1, 2, 3, 4]

        # Cooling season (May-Sep)
        is_cooling_season = time.month in [5, 6, 7, 8, 9]

        if is_heating_season and temperature < comfort_min:
            # Colder = more heating = higher consumption
            temp_diff = comfort_min - temperature
            heating_factor = 1 + (temp_diff * 0.05)  # 5% per degree below comfort
            return min(heating_factor, 2.0)  # Cap at 100% increase

        elif is_cooling_season and temperature > comfort_max:
            # Hotter = more cooling = higher consumption
            temp_diff = temperature - comfort_max
            cooling_factor = 1 + (temp_diff * 0.03)  # 3% per degree above comfort
            return min(cooling_factor, 1.5)  # Cap at 50% increase

        return 1.0

    def _calculate_solar_correction(self, cloud_cover: float, time: dt.datetime) -> float:
        """Calculate solar production impact on consumption"""

        # Only apply during daylight hours
        hour = time.hour
        if hour < 6 or hour > 20:
            return 1.0

        # High solar production (low cloud cover) can reduce grid consumption
        if cloud_cover < 0.3:  # Clear sky
            return 0.85  # 15% reduction in consumption from grid
        elif cloud_cover < 0.6:  # Partly cloudy
            return 0.95  # 5% reduction

        return 1.0  # No solar benefit

    def _calculate_prediction_confidence(
        self,
        prediction_time: dt.datetime,
        historical_data: pd.DataFrame,
        patterns: Dict
    ) -> float:
        """Calculate confidence score for prediction (0-1)"""

        confidence_factors = []

        # Data availability
        data_days = len(historical_data) / 24
        data_confidence = min(data_days / self.min_data_points, 1.0)
        confidence_factors.append(data_confidence)

        # Pattern consistency (low std = high confidence)
        hour = prediction_time.hour
        if hour in patterns['hourly']['std']:
            hour_std = patterns['hourly']['std'][hour]
            hour_mean = patterns['hourly']['mean'][hour]
            if hour_mean > 0:
                hour_confidence = 1 - min(hour_std / hour_mean, 1.0)
                confidence_factors.append(hour_confidence)

        # Time distance (closer = more confident)
        # This would be calculated relative to current time in real implementation
        time_confidence = 0.9  # Placeholder
        confidence_factors.append(time_confidence)

        return sum(confidence_factors) / len(confidence_factors)

    def _fallback_prediction(self, start_time: dt.datetime, hours_ahead: int) -> pd.DataFrame:
        """Fallback prediction when insufficient data available"""

        logging.info("Using fallback prediction method")

        # Simple average-based prediction
        try:
            # Get limited recent data
            recent_data = self._get_recent_consumption_sample()
            if recent_data is not None:
                avg_consumption = recent_data['consumption'].mean()
            else:
                avg_consumption = 2.5  # Default assumption (kW)
        except:
            avg_consumption = 2.5

        predictions = []
        for hour_offset in range(hours_ahead):
            prediction_time = start_time + dt.timedelta(hours=hour_offset)

            # Simple day/night pattern
            hour = prediction_time.hour
            if 6 <= hour <= 22:  # Daytime
                consumption = avg_consumption * 1.2
            else:  # Nighttime
                consumption = avg_consumption * 0.8

            predictions.append({
                'datetime': prediction_time,
                'predicted_consumption': consumption,
                'base_pattern': consumption,
                'confidence': 0.3  # Low confidence for fallback
            })

        df = pd.DataFrame(predictions)
        df.set_index('datetime', inplace=True)
        return df

    def _get_recent_consumption_sample(self) -> pd.DataFrame:
        """Get small sample of recent data for fallback"""
        try:
            query = """
                SELECT datetime, consumption
                FROM energy_data
                ORDER BY datetime DESC
                LIMIT 168  -- Last week
            """
            df = pd.read_sql(query, self.db_da.connection)
            df['datetime'] = pd.to_datetime(df['datetime'])
            return df
        except:
            return None

    def get_prediction_performance_metrics(self) -> Dict:
        """Calculate how well predictions are performing (for monitoring)"""
        # This would compare recent predictions vs actual consumption
        # Implementation depends on how predictions are stored
        return {
            'accuracy': 0.85,  # Placeholder - implement actual comparison
            'mae': 0.3,        # Mean Absolute Error
            'predictions_made': 100,
            'confidence_avg': 0.78
        }