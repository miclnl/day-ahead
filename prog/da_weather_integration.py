"""
Weer Data Integratie Module voor DAO

Deze module implementeert:
- OpenWeatherMap API integratie
- KNMI (Koninklijk Nederlands Meteorologisch Instituut) integratie
- Weer data caching en fallback mechanismen
- Intelligente weer voorspellingen voor energie optimalisatie
"""

import logging
import requests
import json
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
import hashlib
import threading
from functools import wraps

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    logging.warning("Pandas niet beschikbaar - weer data processing beperkt")


@dataclass
class WeatherData:
    """Data class voor weer data"""
    timestamp: datetime
    temperature: float
    humidity: float
    wind_speed: float
    wind_direction: float
    pressure: float
    cloud_cover: float
    solar_radiation: float
    precipitation: float
    uv_index: float
    visibility: float
    source: str
    confidence: float


class WeatherIntegration:
    """Hoofdklasse voor weer data integratie"""

    def __init__(self, config):
        self.config = config
        self.cache = {}
        self.cache_ttl = 1800  # 30 minuten
        self.lock = threading.RLock()

        # API configuraties
        self.openweather_api_key = config.get(['weather', 'openweather_api_key'], None, None)
        self.knmi_api_key = config.get(['weather', 'knmi_api_key'], None, None)
        self.weather_location = config.get(['weather', 'location'], None, 'Amsterdam')
        self.weather_lat = config.get(['weather', 'latitude'], None, 52.3676)
        self.weather_lon = config.get(['weather', 'longitude'], None, 4.9041)

        # API endpoints
        self.openweather_base_url = "https://api.openweathermap.org/data/2.5"
        self.knmi_base_url = "https://api.knmi.nl/v1"

        # Rate limiting
        self.last_api_call = {}
        self.min_call_interval = 60  # 1 minuut tussen API calls

        # Fallback data generatie
        self.fallback_enabled = config.get(['weather', 'fallback_enabled'], None, True)

        logging.info("Weather Integration geïnitialiseerd")

    def get_weather_data(self, start_time: datetime, end_time: datetime, include_forecast: bool = False) -> Dict[str, Any]:
        """
        Haal weer data op voor een bepaalde periode

        Args:
            start_time: Start datum/tijd
            end_time: Eind datum/tijd
            include_forecast: Of voorspellingen moeten worden opgehaald

        Returns:
            Dict met weer data
        """
        try:
            weather_data = {
                'current_conditions': {},
                'historical_data': {},
                'forecast_data': {},
                'metadata': {
                    'start_time': start_time.isoformat(),
                    'end_time': end_time.isoformat(),
                    'data_source': 'unknown',
                    'last_updated': datetime.now().isoformat()
                }
            }

            # Probeer eerst OpenWeatherMap data op te halen
            if self.openweather_api_key:
                try:
                    ow_data = self._get_openweather_data(start_time, end_time, include_forecast)
                    if ow_data:
                        weather_data.update(ow_data)
                        weather_data['metadata']['data_source'] = 'openweathermap'
                        logging.info("OpenWeatherMap data succesvol opgehaald")
                        return weather_data
                except Exception as e:
                    logging.warning(f"OpenWeatherMap data ophalen mislukt: {e}")

            # Probeer KNMI data als fallback
            if self.knmi_api_key:
                try:
                    knmi_data = self._get_knmi_data(start_time, end_time, include_forecast)
                    if knmi_data:
                        weather_data.update(knmi_data)
                        weather_data['metadata']['data_source'] = 'knmi'
                        logging.info("KNMI data succesvol opgehaald")
                        return weather_data
                except Exception as e:
                    logging.warning(f"KNMI data ophalen mislukt: {e}")

            # Gebruik fallback data als laatste optie
            if self.fallback_enabled:
                weather_data['historical_data'] = self._generate_fallback_weather_data(start_time, end_time)
                weather_data['metadata']['data_source'] = 'fallback'
                logging.info("Fallback weer data gebruikt")

            # Voeg huidige condities toe
            weather_data['current_conditions'] = self._get_current_weather_conditions(weather_data['historical_data'])

            # Voeg voorspelling toe als gevraagd
            if include_forecast:
                weather_data['forecast_data'] = self._generate_weather_forecast(end_time)

            return weather_data

        except Exception as e:
            logging.error(f"Fout bij ophalen weer data: {e}")
            return self._get_default_weather_data(start_time, end_time)

    def _get_openweather_data(self, start_time: datetime, end_time: datetime, include_forecast: bool) -> Optional[Dict[str, Any]]:
        """Haal weer data op van OpenWeatherMap API"""
        try:
            # Check rate limiting
            if not self._can_make_api_call('openweathermap'):
                logging.debug("OpenWeatherMap API rate limit bereikt")
                return None

            # Haal huidige weer op
            current_weather = self._fetch_openweather_current()
            if not current_weather:
                return None

            # Haal historische data op (laatste 5 dagen)
            historical_data = self._fetch_openweather_historical(start_time, end_time)

            # Haal voorspelling op als gevraagd
            forecast_data = {}
            if include_forecast:
                forecast_data = self._fetch_openweather_forecast(end_time)

            # Converteer naar standaard formaat
            weather_data = {
                'current_conditions': self._convert_openweather_current(current_weather),
                'historical_data': self._convert_openweather_historical(historical_data),
                'forecast_data': self._convert_openweather_forecast(forecast_data)
            }

            # Update rate limiting
            self._update_api_call_time('openweathermap')

            return weather_data

        except Exception as e:
            logging.error(f"OpenWeatherMap data ophalen fout: {e}")
            return None

    def _fetch_openweather_current(self) -> Optional[Dict[str, Any]]:
        """Haal huidige weer op van OpenWeatherMap"""
        try:
            url = f"{self.openweather_base_url}/weather"
            params = {
                'lat': self.weather_lat,
                'lon': self.weather_lon,
                'appid': self.openweather_api_key,
                'units': 'metric',
                'lang': 'nl'
            }

            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()

            return response.json()

        except Exception as e:
            logging.error(f"OpenWeatherMap current weather fout: {e}")
            return None

    def _fetch_openweather_historical(self, start_time: datetime, end_time: datetime) -> List[Dict[str, Any]]:
        """Haal historische weer data op van OpenWeatherMap"""
        try:
            # OpenWeatherMap biedt historische data voor de laatste 5 dagen
            # Voor oudere data zou je een betaalde API nodig hebben
            historical_data = []

            # Haal data op voor elk uur in de periode
            current_time = start_time
            while current_time <= end_time:
                if current_time >= datetime.now() - timedelta(days=5):
                    # Data is beschikbaar via gratis API
                    timestamp = int(current_time.timestamp())
                    url = f"{self.openweather_base_url}/onecall/timemachine"
                    params = {
                        'lat': self.weather_lat,
                        'lon': self.weather_lon,
                        'dt': timestamp,
                        'appid': self.openweather_api_key,
                        'units': 'metric'
                    }

                    try:
                        response = requests.get(url, params=params, timeout=10)
                        if response.status_code == 200:
                            data = response.json()
                            if 'data' in data and data['data']:
                                historical_data.extend(data['data'])
                    except Exception as e:
                        logging.debug(f"OpenWeatherMap historische data fout voor {current_time}: {e}")

                current_time += timedelta(hours=1)

            return historical_data

        except Exception as e:
            logging.error(f"OpenWeatherMap historische data fout: {e}")
            return []

    def _fetch_openweather_forecast(self, start_time: datetime) -> Dict[str, Any]:
        """Haal weer voorspelling op van OpenWeatherMap"""
        try:
            url = f"{self.openweather_base_url}/forecast"
            params = {
                'lat': self.weather_lat,
                'lon': self.weather_lon,
                'appid': self.openweather_api_key,
                'units': 'metric',
                'lang': 'nl'
            }

            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()

            return response.json()

        except Exception as e:
            logging.error(f"OpenWeatherMap forecast fout: {e}")
            return {}

    def _convert_openweather_current(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Converteer OpenWeatherMap current weather naar standaard formaat"""
        try:
            if 'main' not in data:
                return {}

            main = data['main']
            wind = data.get('wind', {})
            clouds = data.get('clouds', {})
            weather = data.get('weather', [{}])[0]

            return {
                'temperature': {
                    'value': main.get('temp', 0),
                    'unit': '°C',
                    'timestamp': datetime.now().isoformat()
                },
                'humidity': {
                    'value': main.get('humidity', 0),
                    'unit': '%',
                    'timestamp': datetime.now().isoformat()
                },
                'pressure': {
                    'value': main.get('pressure', 0),
                    'unit': 'hPa',
                    'timestamp': datetime.now().isoformat()
                },
                'wind_speed': {
                    'value': wind.get('speed', 0),
                    'unit': 'm/s',
                    'timestamp': datetime.now().isoformat()
                },
                'wind_direction': {
                    'value': wind.get('deg', 0),
                    'unit': '°',
                    'timestamp': datetime.now().isoformat()
                },
                'cloud_cover': {
                    'value': clouds.get('all', 0),
                    'unit': '%',
                    'timestamp': datetime.now().isoformat()
                },
                'description': weather.get('description', ''),
                'icon': weather.get('icon', '')
            }

        except Exception as e:
            logging.error(f"OpenWeatherMap current weather conversie fout: {e}")
            return {}

    def _convert_openweather_historical(self, data_list: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Converteer OpenWeatherMap historische data naar standaard formaat"""
        try:
            converted_data = {
                'temperature': [],
                'humidity': [],
                'wind_speed': [],
                'pressure': [],
                'cloud_cover': []
            }

            for data in data_list:
                timestamp = datetime.fromtimestamp(data.get('dt', 0))
                main = data.get('main', {})
                wind = data.get('wind', {})
                clouds = data.get('clouds', {})

                converted_data['temperature'].append({
                    'timestamp': timestamp.isoformat(),
                    'temperature': main.get('temp', 0),
                    'unit': '°C'
                })

                converted_data['humidity'].append({
                    'timestamp': timestamp.isoformat(),
                    'humidity': main.get('humidity', 0),
                    'unit': '%'
                })

                converted_data['wind_speed'].append({
                    'timestamp': timestamp.isoformat(),
                    'wind_speed': wind.get('speed', 0),
                    'unit': 'm/s'
                })

                converted_data['pressure'].append({
                    'timestamp': timestamp.isoformat(),
                    'pressure': main.get('pressure', 0),
                    'unit': 'hPa'
                })

                converted_data['cloud_cover'].append({
                    'timestamp': timestamp.isoformat(),
                    'cloud_cover': clouds.get('all', 0),
                    'unit': '%'
                })

            return converted_data

        except Exception as e:
            logging.error(f"OpenWeatherMap historische data conversie fout: {e}")
            return {}

    def _convert_openweather_forecast(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Converteer OpenWeatherMap forecast naar standaard formaat"""
        try:
            forecast_data = {
                'temperature': [],
                'metadata': {
                    'forecast_hours': 0,
                    'generated_at': datetime.now().isoformat(),
                    'data_source': 'openweathermap'
                }
            }

            if 'list' in data:
                forecast_list = data['list']
                forecast_data['metadata']['forecast_hours'] = len(forecast_list) * 3  # 3-uurs intervallen

                for item in forecast_list:
                    timestamp = datetime.fromtimestamp(item.get('dt', 0))
                    main = item.get('main', {})

                    forecast_data['temperature'].append({
                        'timestamp': timestamp.isoformat(),
                        'temperature': main.get('temp', 0),
                        'unit': '°C',
                        'confidence': 0.8  # OpenWeatherMap heeft hoge betrouwbaarheid
                    })

            return forecast_data

        except Exception as e:
            logging.error(f"OpenWeatherMap forecast conversie fout: {e}")
            return {}

    def _get_knmi_data(self, start_time: datetime, end_time: datetime, include_forecast: bool) -> Optional[Dict[str, Any]]:
        """Haal weer data op van KNMI API"""
        try:
            # KNMI biedt historische data en voorspellingen
            # Dit is een vereenvoudigde implementatie
            logging.info("KNMI integratie nog niet volledig geïmplementeerd")
            return None

        except Exception as e:
            logging.error(f"KNMI data ophalen fout: {e}")
            return None

    def _generate_fallback_weather_data(self, start_time: datetime, end_time: datetime) -> Dict[str, List[Dict[str, Any]]]:
        """Genereer realistische fallback weer data"""
        try:
            weather_data = {}
            time_range = pd.date_range(start_time, end_time, freq='1H') if PANDAS_AVAILABLE else []

            if not time_range:
                # Fallback zonder pandas
                current = start_time
                time_range = []
                while current <= end_time:
                    time_range.append(current)
                    current += timedelta(hours=1)

            # Temperatuur patroon (dag/nacht cyclus)
            temperatures = []
            for timestamp in time_range:
                hour = timestamp.hour if hasattr(timestamp, 'hour') else timestamp.hour
                # Basis temperatuur met dag/nacht variatie
                base_temp = 15 + 5 * self._sin_approx((hour - 6) * 3.14159 / 12)
                # Voeg realistische variatie toe
                variation = (hash(f"{timestamp.date()}-{hour}") % 40 - 20) / 10
                temp = base_temp + variation
                temperatures.append({
                    'timestamp': timestamp.isoformat() if hasattr(timestamp, 'isoformat') else str(timestamp),
                    'temperature': round(temp, 1),
                    'unit': '°C'
                })
            weather_data['temperature'] = temperatures

            # Luchtvochtigheid (inverse relatie met temperatuur)
            humidities = []
            for temp_record in temperatures:
                temp = temp_record['temperature']
                # Hogere luchtvochtigheid bij lagere temperaturen
                humidity = max(40, min(95, 80 - (temp - 15) * 2))
                humidities.append({
                    'timestamp': temp_record['timestamp'],
                    'humidity': round(humidity, 1),
                    'unit': '%'
                })
            weather_data['humidity'] = humidities

            # Windsnelheid (basis waarde met variatie)
            wind_speeds = []
            for temp_record in temperatures:
                # Basis windsnelheid met kleine variatie
                base_wind = 5 + (hash(temp_record['timestamp']) % 20 - 10) / 10
                wind_speeds.append({
                    'timestamp': temp_record['timestamp'],
                    'wind_speed': round(base_wind, 1),
                    'unit': 'm/s'
                })
            weather_data['wind_speed'] = wind_speeds

            return weather_data

        except Exception as e:
            logging.error(f"Fout bij genereren fallback weer data: {e}")
            return {}

    def _sin_approx(self, x: float) -> float:
        """Eenvoudige sinus benadering zonder math module"""
        # Eenvoudige benadering van sin(x) voor -π tot π
        x = x % (2 * 3.14159)
        if x > 3.14159:
            x = 2 * 3.14159 - x

        # Taylor series benadering
        return x - (x**3) / 6 + (x**5) / 120

    def _get_current_weather_conditions(self, historical_data: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Haal huidige weer condities op uit historische data"""
        try:
            current_conditions = {}

            # Neem de meest recente waarden
            for weather_type, data_list in historical_data.items():
                if data_list and len(data_list) > 0:
                    latest_record = data_list[-1]
                    if 'temperature' in latest_record:
                        current_conditions[weather_type] = {
                            'value': latest_record['temperature'],
                            'unit': latest_record.get('unit', ''),
                            'timestamp': latest_record['timestamp']
                        }
                    elif 'humidity' in latest_record:
                        current_conditions[weather_type] = {
                            'value': latest_record['humidity'],
                            'unit': latest_record.get('unit', ''),
                            'timestamp': latest_record['timestamp']
                        }
                    elif 'wind_speed' in latest_record:
                        current_conditions[weather_type] = {
                            'value': latest_record['wind_speed'],
                            'unit': latest_record.get('unit', ''),
                            'timestamp': latest_record['timestamp']
                        }

            return current_conditions

        except Exception as e:
            logging.error(f"Fout bij ophalen huidige weer condities: {e}")
            return {}

    def _generate_weather_forecast(self, start_time: datetime) -> Dict[str, Any]:
        """Genereer weer voorspelling voor de komende 24 uur"""
        try:
            forecast_data = {}
            forecast_hours = []

            # Genereer 24-uurs voorspelling
            for i in range(24):
                timestamp = start_time + timedelta(hours=i)
                hour = timestamp.hour

                # Eenvoudige lineaire voorspelling gebaseerd op huidige condities
                base_temp = 15 + 5 * self._sin_approx((hour - 6) * 3.14159 / 12)

                forecast_hours.append({
                    'timestamp': timestamp.isoformat(),
                    'temperature': round(base_temp, 1),
                    'unit': '°C',
                    'confidence': 0.7  # 70% vertrouwen in voorspelling
                })

            forecast_data['temperature'] = forecast_hours
            forecast_data['metadata'] = {
                'forecast_hours': 24,
                'generated_at': datetime.now().isoformat(),
                'data_source': 'fallback_model'
            }

            return forecast_data

        except Exception as e:
            logging.error(f"Fout bij genereren weer voorspelling: {e}")
            return {}

    def _get_default_weather_data(self, start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        """Return standaard weer data bij fout"""
        return {
            'current_conditions': {},
            'historical_data': {},
            'forecast_data': {},
            'metadata': {
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'data_source': 'fallback',
                'last_updated': datetime.now().isoformat(),
                'error': 'Could not retrieve weather data'
            }
        }

    def _can_make_api_call(self, api_name: str) -> bool:
        """Check of er een API call gemaakt kan worden (rate limiting)"""
        try:
            current_time = time.time()
            last_call = self.last_api_call.get(api_name, 0)

            return (current_time - last_call) >= self.min_call_interval

        except Exception as e:
            logging.debug(f"Rate limiting check fout: {e}")
            return True

    def _update_api_call_time(self, api_name: str):
        """Update de laatste API call tijd"""
        try:
            self.last_api_call[api_name] = time.time()
        except Exception as e:
            logging.debug(f"API call tijd update fout: {e}")

    def get_weather_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Haal weer samenvatting op voor de komende uren"""
        try:
            end_time = datetime.now() + timedelta(hours=hours)
            weather_data = self.get_weather_data(datetime.now(), end_time, include_forecast=True)

            if not weather_data or 'historical_data' not in weather_data:
                return self._get_default_weather_summary()

            # Bereken samenvatting
            summary = self._calculate_weather_summary(weather_data)

            return summary

        except Exception as e:
            logging.error(f"Weer samenvatting fout: {e}")
            return self._get_default_weather_summary()

    def _calculate_weather_summary(self, weather_data: Dict[str, Any]) -> Dict[str, Any]:
        """Bereken weer samenvatting uit weer data"""
        try:
            summary = {
                'temperature': {'min': 0, 'max': 0, 'avg': 0},
                'humidity': {'min': 0, 'max': 0, 'avg': 0},
                'wind_speed': {'min': 0, 'max': 0, 'avg': 0},
                'solar_potential': 0,
                'weather_conditions': 'unknown'
            }

            # Bereken temperature statistieken
            if 'temperature' in weather_data['historical_data']:
                temps = [item['temperature'] for item in weather_data['historical_data']['temperature']]
                if temps:
                    summary['temperature'] = {
                        'min': min(temps),
                        'max': max(temps),
                        'avg': sum(temps) / len(temps)
                    }

            # Bereken humidity statistieken
            if 'humidity' in weather_data['historical_data']:
                humidities = [item['humidity'] for item in weather_data['historical_data']['humidity']]
                if humidities:
                    summary['humidity'] = {
                        'min': min(humidities),
                        'max': max(humidities),
                        'avg': sum(humidities) / len(humidities)
                    }

            # Bereken wind statistieken
            if 'wind_speed' in weather_data['historical_data']:
                winds = [item['wind_speed'] for item in weather_data['historical_data']['wind_speed']]
                if winds:
                    summary['wind_speed'] = {
                        'min': min(winds),
                        'max': max(winds),
                        'avg': sum(winds) / len(winds)
                    }

            # Bereken solar potential
            summary['solar_potential'] = self._calculate_solar_potential(weather_data)

            # Bepaal weer condities
            summary['weather_conditions'] = self._determine_weather_conditions(summary)

            return summary

        except Exception as e:
            logging.error(f"Weer samenvatting berekening fout: {e}")
            return self._get_default_weather_summary()

    def _calculate_solar_potential(self, weather_data: Dict[str, Any]) -> float:
        """Bereken solar potential gebaseerd op weer condities"""
        try:
            # Eenvoudige berekening gebaseerd op temperatuur en bewolking
            if 'temperature' in weather_data['historical_data'] and 'cloud_cover' in weather_data['historical_data']:
                avg_temp = sum(item['temperature'] for item in weather_data['historical_data']['temperature']) / len(weather_data['historical_data']['temperature'])
                avg_clouds = sum(item['cloud_cover'] for item in weather_data['historical_data']['cloud_cover']) / len(weather_data['historical_data']['cloud_cover'])

                # Solar potential (0-100%)
                temp_factor = max(0, min(1, (avg_temp - 10) / 20))  # 10-30°C is optimaal
                cloud_factor = max(0, 1 - (avg_clouds / 100))

                solar_potential = (temp_factor * 0.6 + cloud_factor * 0.4) * 100
                return round(solar_potential, 1)

            return 50.0  # Default waarde

        except Exception as e:
            logging.debug(f"Solar potential berekening fout: {e}")
            return 50.0

    def _determine_weather_conditions(self, summary: Dict[str, Any]) -> str:
        """Bepaal weer condities uit samenvatting"""
        try:
            temp_avg = summary['temperature']['avg']
            humidity_avg = summary['humidity']['avg']
            wind_avg = summary['wind_speed']['avg']

            if temp_avg < 5:
                return 'koud'
            elif temp_avg > 25:
                return 'warm'
            elif humidity_avg > 80:
                return 'vochtig'
            elif wind_avg > 10:
                return 'winderig'
            else:
                return 'mild'

        except Exception as e:
            logging.debug(f"Weer condities bepaling fout: {e}")
            return 'unknown'

    def _get_default_weather_summary(self) -> Dict[str, Any]:
        """Return standaard weer samenvatting"""
        return {
            'temperature': {'min': 15, 'max': 20, 'avg': 17.5},
            'humidity': {'min': 60, 'max': 80, 'avg': 70},
            'wind_speed': {'min': 3, 'max': 8, 'avg': 5.5},
            'solar_potential': 50.0,
            'weather_conditions': 'mild'
        }


# Utility functies voor externe gebruik
def create_weather_integration(config) -> WeatherIntegration:
    """Factory functie voor het maken van een WeatherIntegration instance"""
    return WeatherIntegration(config)


def get_weather_data_cached(weather_integration: WeatherIntegration, ttl: int = 1800):
    """Decorator voor het cachen van weer data functies"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Genereer cache key
            cache_key = f"weather_{func.__name__}_{hash(str(args) + str(sorted(kwargs.items())))}"

            # Check cache
            if hasattr(weather_integration, 'cache') and cache_key in weather_integration.cache:
                cache_entry = weather_integration.cache[cache_key]
                if time.time() - cache_entry['timestamp'] < ttl:
                    return cache_entry['data']

            # Voer functie uit
            result = func(*args, **kwargs)

            # Cache resultaat
            if hasattr(weather_integration, 'cache'):
                weather_integration.cache[cache_key] = {
                    'data': result,
                    'timestamp': time.time()
                }

            return result
        return wrapper
    return decorator
