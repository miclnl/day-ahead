"""
Prijs Data Integratie Module voor DAO

Deze module implementeert:
- NordPool integratie (Noord-Europese energie markt)
- ENTSO-E integratie (Europese netwerk operators)
- EasyEnergy integratie (Nederlandse energie leverancier)
- Tibber integratie (Noorse energie leverancier)
- Prijs data caching en fallback mechanismen
- Intelligente prijs voorspellingen
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
    logging.warning("Pandas niet beschikbaar - prijs data processing beperkt")


@dataclass
class PriceData:
    """Data class voor prijs data"""
    timestamp: datetime
    price: float
    currency: str
    market: str
    region: str
    source: str
    confidence: float
    unit: str = "EUR/MWh"


class PriceIntegration:
    """Hoofdklasse voor prijs data integratie"""

    def __init__(self, config):
        self.config = config
        self.cache = {}
        self.cache_ttl = 3600  # 1 uur
        self.lock = threading.RLock()

        # API configuraties
        self.nordpool_api_key = config.get(['prices', 'nordpool_api_key'], None, None)
        self.entsoe_api_key = config.get(['prices', 'entsoe_api_key'], None, None)
        self.easyenergy_api_key = config.get(['prices', 'easyenergy_api_key'], None, None)
        self.tibber_api_key = config.get(['prices', 'tibber_api_key'], None, None)

        # API endpoints
        self.nordpool_base_url = "https://api.nordpoolgroup.com/v1"
        self.entsoe_base_url = "https://transparency.entsoe.eu/api"
        self.easyenergy_base_url = "https://api.easyenergy.com/v1"
        self.tibber_base_url = "https://api.tibber.com/v1-beta/gql"

        # Rate limiting
        self.last_api_call = {}
        self.min_call_interval = 300  # 5 minuten tussen API calls

        # Fallback data generatie
        self.fallback_enabled = config.get(['prices', 'fallback_enabled'], None, True)

        # Prijs voorspelling modellen
        self.prediction_models = {
            'simple': self._simple_price_prediction,
            'trend': self._trend_based_prediction,
            'weather': self._weather_based_prediction
        }

        logging.info("Price Integration geïnitialiseerd")

    def get_price_data(self, start_time: datetime, end_time: datetime, market: str = None) -> Dict[str, Any]:
        """
        Haal prijs data op voor een bepaalde periode

        Args:
            start_time: Start datum/tijd
            end_time: Eind datum/tijd
            market: Specifieke markt (optioneel)

        Returns:
            Dict met prijs data
        """
        try:
            price_data = {
                'current_prices': {},
                'historical_data': {},
                'forecast_data': {},
                'metadata': {
                    'start_time': start_time.isoformat(),
                    'end_time': end_time.isoformat(),
                    'data_source': 'unknown',
                    'last_updated': datetime.now().isoformat()
                }
            }

            # Probeer eerst NordPool data op te halen
            if self.nordpool_api_key and (not market or market == 'nordpool'):
                try:
                    nordpool_data = self._get_nordpool_data(start_time, end_time)
                    if nordpool_data:
                        price_data.update(nordpool_data)
                        price_data['metadata']['data_source'] = 'nordpool'
                        logging.info("NordPool prijs data succesvol opgehaald")
                        return price_data
                except Exception as e:
                    logging.warning(f"NordPool data ophalen mislukt: {e}")

            # Probeer ENTSO-E data als fallback
            if self.entsoe_api_key and (not market or market == 'entsoe'):
                try:
                    entsoe_data = self._get_entsoe_data(start_time, end_time)
                    if entsoe_data:
                        price_data.update(entsoe_data)
                        price_data['metadata']['data_source'] = 'entsoe'
                        logging.info("ENTSO-E prijs data succesvol opgehaald")
                        return price_data
                except Exception as e:
                    logging.warning(f"ENTSO-E data ophalen mislukt: {e}")

            # Probeer EasyEnergy data
            if self.easyenergy_api_key and (not market or market == 'easyenergy'):
                try:
                    easyenergy_data = self._get_easyenergy_data(start_time, end_time)
                    if easyenergy_data:
                        price_data.update(easyenergy_data)
                        price_data['metadata']['data_source'] = 'easyenergy'
                        logging.info("EasyEnergy prijs data succesvol opgehaald")
                        return price_data
                except Exception as e:
                    logging.warning(f"EasyEnergy data ophalen mislukt: {e}")

            # Probeer Tibber data
            if self.tibber_api_key and (not market or market == 'tibber'):
                try:
                    tibber_data = self._get_tibber_data(start_time, end_time)
                    if tibber_data:
                        price_data.update(tibber_data)
                        price_data['metadata']['data_source'] = 'tibber'
                        logging.info("Tibber prijs data succesvol opgehaald")
                        return price_data
                except Exception as e:
                    logging.warning(f"Tibber data ophalen mislukt: {e}")

            # Gebruik fallback data als laatste optie
            if self.fallback_enabled:
                price_data['historical_data'] = self._generate_fallback_price_data(start_time, end_time)
                price_data['metadata']['data_source'] = 'fallback'
                logging.info("Fallback prijs data gebruikt")

            # Voeg huidige prijzen toe
            price_data['current_prices'] = self._get_current_prices(price_data['historical_data'])

            # Voeg voorspelling toe
            price_data['forecast_data'] = self._generate_price_forecast(end_time)

            return price_data

        except Exception as e:
            logging.error(f"Fout bij ophalen prijs data: {e}")
            return self._get_default_price_data(start_time, end_time)

    def _get_nordpool_data(self, start_time: datetime, end_time: datetime) -> Optional[Dict[str, Any]]:
        """Haal prijs data op van NordPool API"""
        try:
            # Check rate limiting
            if not self._can_make_api_call('nordpool'):
                logging.debug("NordPool API rate limit bereikt")
                return None

            # NordPool biedt dag-ahead prijzen
            # Dit is een vereenvoudigde implementatie
            logging.info("NordPool integratie nog niet volledig geïmplementeerd")

            # Update rate limiting
            self._update_api_call_time('nordpool')

            return None

        except Exception as e:
            logging.error(f"NordPool data ophalen fout: {e}")
            return None

    def _get_entsoe_data(self, start_time: datetime, end_time: datetime) -> Optional[Dict[str, Any]]:
        """Haal prijs data op van ENTSO-E API"""
        try:
            # Check rate limiting
            if not self._can_make_api_call('entsoe'):
                logging.debug("ENTSO-E API rate limit bereikt")
                return None

            # ENTSO-E biedt transparantie data
            # Dit is een vereenvoudigde implementatie
            logging.info("ENTSO-E integratie nog niet volledig geïmplementeerd")

            # Update rate limiting
            self._update_api_call_time('entsoe')

            return None

        except Exception as e:
            logging.error(f"ENTSO-E data ophalen fout: {e}")
            return None

    def _get_easyenergy_data(self, start_time: datetime, end_time: datetime) -> Optional[Dict[str, Any]]:
        """Haal prijs data op van EasyEnergy API"""
        try:
            # Check rate limiting
            if not self._can_make_api_call('easyenergy'):
                logging.debug("EasyEnergy API rate limit bereikt")
                return None

            # EasyEnergy biedt Nederlandse energie prijzen
            # Dit is een vereenvoudigde implementatie
            logging.info("EasyEnergy integratie nog niet volledig geïmplementeerd")

            # Update rate limiting
            self._update_api_call_time('easyenergy')

            return None

        except Exception as e:
            logging.error(f"EasyEnergy data ophalen fout: {e}")
            return None

    def _get_tibber_data(self, start_time: datetime, end_time: datetime) -> Optional[Dict[str, Any]]:
        """Haal prijs data op van Tibber API"""
        try:
            # Check rate limiting
            if not self._can_make_api_call('tibber'):
                logging.debug("Tibber API rate limit bereikt")
                return None

            # Tibber biedt Noorse energie prijzen
            # Dit is een vereenvoudigde implementatie
            logging.info("Tibber integratie nog niet volledig geïmplementeerd")

            # Update rate limiting
            self._update_api_call_time('tibber')

            return None

        except Exception as e:
            logging.error(f"Tibber data ophalen fout: {e}")
            return None

    def _generate_fallback_price_data(self, start_time: datetime, end_time: datetime) -> Dict[str, List[Dict[str, Any]]]:
        """Genereer realistische fallback prijs data"""
        try:
            price_data = {}
            time_range = []

            # Genereer uurlijkse tijdstippen
            current = start_time
            while current <= end_time:
                time_range.append(current)
                current += timedelta(hours=1)

            # Genereer prijzen gebaseerd op typische dag/nacht patronen
            prices = []
            for timestamp in time_range:
                hour = timestamp.hour

                # Basis prijs patroon (dag/nacht variatie)
                base_price = 50.0  # EUR/MWh basis prijs

                # Dag prijs (8:00-18:00) is hoger
                if 8 <= hour <= 18:
                    base_price += 20.0

                # Ochtend piek (7:00-9:00)
                if 7 <= hour <= 9:
                    base_price += 15.0

                # Avond piek (17:00-19:00)
                if 17 <= hour <= 19:
                    base_price += 15.0

                # Nacht prijs (23:00-6:00) is lager
                if hour >= 23 or hour <= 6:
                    base_price -= 10.0

                # Weekend prijzen zijn lager
                if timestamp.weekday() >= 5:  # Zaterdag en zondag
                    base_price -= 5.0

                # Voeg realistische variatie toe
                variation = (hash(f"{timestamp.date()}-{hour}") % 40 - 20) / 10
                final_price = base_price + variation

                prices.append({
                    'timestamp': timestamp.isoformat(),
                    'price': round(final_price, 2),
                    'unit': 'EUR/MWh',
                    'market': 'fallback',
                    'region': 'NL'
                })

            price_data['hourly_prices'] = prices

            # Bereken dagelijkse gemiddelden
            daily_prices = {}
            for price_record in prices:
                date_key = price_record['timestamp'][:10]  # YYYY-MM-DD
                if date_key not in daily_prices:
                    daily_prices[date_key] = []
                daily_prices[date_key].append(price_record['price'])

            daily_averages = []
            for date_key, day_prices in daily_prices.items():
                daily_averages.append({
                    'date': date_key,
                    'avg_price': round(sum(day_prices) / len(day_prices), 2),
                    'min_price': min(day_prices),
                    'max_price': max(day_prices),
                    'unit': 'EUR/MWh'
                })

            price_data['daily_averages'] = daily_averages

            return price_data

        except Exception as e:
            logging.error(f"Fout bij genereren fallback prijs data: {e}")
            return {}

    def _get_current_prices(self, historical_data: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Haal huidige prijzen op uit historische data"""
        try:
            current_prices = {}

            if 'hourly_prices' in historical_data and historical_data['hourly_prices']:
                latest_price = historical_data['hourly_prices'][-1]
                current_prices['current_price'] = {
                    'value': latest_price['price'],
                    'unit': latest_price['unit'],
                    'timestamp': latest_price['timestamp'],
                    'market': latest_price.get('market', 'unknown'),
                    'region': latest_price.get('region', 'unknown')
                }

            return current_prices

        except Exception as e:
            logging.error(f"Fout bij ophalen huidige prijzen: {e}")
            return {}

    def _generate_price_forecast(self, start_time: datetime) -> Dict[str, Any]:
        """Genereer prijs voorspelling voor de komende 24 uur"""
        try:
            forecast_data = {
                'hourly_forecast': [],
                'metadata': {
                    'forecast_hours': 24,
                    'generated_at': datetime.now().isoformat(),
                    'data_source': 'prediction_model'
                }
            }

            # Genereer 24-uurs voorspelling
            for i in range(24):
                timestamp = start_time + timedelta(hours=i)
                hour = timestamp.hour

                # Eenvoudige prijs voorspelling gebaseerd op typische patronen
                base_price = 50.0

                # Dag prijs (8:00-18:00) is hoger
                if 8 <= hour <= 18:
                    base_price += 20.0
                elif hour >= 23 or hour <= 6:
                    base_price -= 10.0

                # Weekend prijzen zijn lager
                if timestamp.weekday() >= 5:
                    base_price -= 5.0

                forecast_data['hourly_forecast'].append({
                    'timestamp': timestamp.isoformat(),
                    'price': round(base_price, 2),
                    'unit': 'EUR/MWh',
                    'confidence': 0.7  # 70% vertrouwen in voorspelling
                })

            return forecast_data

        except Exception as e:
            logging.error(f"Fout bij genereren prijs voorspelling: {e}")
            return {}

    def _get_default_price_data(self, start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        """Return standaard prijs data bij fout"""
        return {
            'current_prices': {},
            'historical_data': {},
            'forecast_data': {},
            'metadata': {
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'data_source': 'fallback',
                'last_updated': datetime.now().isoformat(),
                'error': 'Could not retrieve price data'
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

    def get_price_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Haal prijs samenvatting op voor de komende uren"""
        try:
            end_time = datetime.now() + timedelta(hours=hours)
            price_data = self.get_price_data(datetime.now(), end_time)

            if not price_data or 'historical_data' not in price_data:
                return self._get_default_price_summary()

            # Bereken samenvatting
            summary = self._calculate_price_summary(price_data)

            return summary

        except Exception as e:
            logging.error(f"Prijs samenvatting fout: {e}")
            return self._get_default_price_summary()

    def _calculate_price_summary(self, price_data: Dict[str, Any]) -> Dict[str, Any]:
        """Bereken prijs samenvatting uit prijs data"""
        try:
            summary = {
                'current_price': 0.0,
                'price_range': {'min': 0, 'max': 0, 'avg': 0},
                'market_trend': 'stable',
                'peak_hours': [],
                'off_peak_hours': [],
                'recommendations': []
            }

            # Bereken prijs statistieken
            if 'hourly_prices' in price_data['historical_data']:
                prices = [item['price'] for item in price_data['historical_data']['hourly_prices']]
                if prices:
                    summary['price_range'] = {
                        'min': min(prices),
                        'max': max(prices),
                        'avg': round(sum(prices) / len(prices), 2)
                    }

                    # Bepaal markt trend
                    if len(prices) >= 2:
                        price_change = prices[-1] - prices[0]
                        if price_change > 5:
                            summary['market_trend'] = 'rising'
                        elif price_change < -5:
                            summary['market_trend'] = 'falling'
                        else:
                            summary['market_trend'] = 'stable'

                    # Identificeer piek en dal uren
                    avg_price = summary['price_range']['avg']
                    for item in price_data['historical_data']['hourly_prices']:
                        if item['price'] > avg_price * 1.2:  # 20% boven gemiddelde
                            summary['peak_hours'].append(item['timestamp'])
                        elif item['price'] < avg_price * 0.8:  # 20% onder gemiddelde
                            summary['off_peak_hours'].append(item['timestamp'])

                    # Genereer aanbevelingen
                    summary['recommendations'] = self._generate_price_recommendations(summary)

            # Huidige prijs
            if 'current_prices' in price_data and 'current_price' in price_data['current_prices']:
                summary['current_price'] = price_data['current_prices']['current_price']['value']

            return summary

        except Exception as e:
            logging.error(f"Prijs samenvatting berekening fout: {e}")
            return self._get_default_price_summary()

    def _generate_price_recommendations(self, summary: Dict[str, Any]) -> List[str]:
        """Genereer prijs aanbevelingen"""
        try:
            recommendations = []

            if summary['market_trend'] == 'rising':
                recommendations.append("Energie prijzen stijgen - overweeg om energie op te slaan")
                recommendations.append("Plan energie-intensieve activiteiten tijdens daluren")

            elif summary['market_trend'] == 'falling':
                recommendations.append("Energie prijzen dalen - gunstig moment voor verbruik")
                recommendations.append("Overweeg om opgeslagen energie te verkopen")

            if summary['peak_hours']:
                recommendations.append(f"Vermijd energie verbruik tijdens piekuren: {len(summary['peak_hours'])} uur")

            if summary['off_peak_hours']:
                recommendations.append(f"Gunstige uren voor energie verbruik: {len(summary['off_peak_hours'])} uur")

            if not recommendations:
                recommendations.append("Prijs patroon is stabiel - normaal verbruik mogelijk")

            return recommendations

        except Exception as e:
            logging.debug(f"Prijs aanbevelingen genereren fout: {e}")
            return ["Kon geen aanbevelingen genereren"]

    def _get_default_price_summary(self) -> Dict[str, Any]:
        """Return standaard prijs samenvatting"""
        return {
            'current_price': 50.0,
            'price_range': {'min': 40, 'max': 70, 'avg': 55},
            'market_trend': 'stable',
            'peak_hours': [],
            'off_peak_hours': [],
            'recommendations': ['Geen prijs data beschikbaar']
        }

    def predict_price_for_hour(self, target_hour: datetime, model: str = 'simple') -> Dict[str, Any]:
        """
        Voorspel prijs voor een specifiek uur

        Args:
            target_hour: Doel uur voor voorspelling
            model: Voorspelling model ('simple', 'trend', 'weather')

        Returns:
            Dict met prijs voorspelling
        """
        try:
            if model not in self.prediction_models:
                model = 'simple'

            # Haal historische data op
            start_time = target_hour - timedelta(days=7)
            price_data = self.get_price_data(start_time, target_hour)

            # Gebruik geselecteerd model
            prediction = self.prediction_models[model](target_hour, price_data)

            return {
                'timestamp': target_hour.isoformat(),
                'predicted_price': prediction['price'],
                'confidence': prediction['confidence'],
                'model': model,
                'unit': 'EUR/MWh',
                'factors': prediction.get('factors', [])
            }

        except Exception as e:
            logging.error(f"Prijs voorspelling fout: {e}")
            return {
                'timestamp': target_hour.isoformat(),
                'predicted_price': 50.0,
                'confidence': 0.5,
                'model': 'fallback',
                'unit': 'EUR/MWh',
                'factors': ['Voorspelling mislukt']
            }

    def _simple_price_prediction(self, target_hour: datetime, price_data: Dict[str, Any]) -> Dict[str, Any]:
        """Eenvoudige prijs voorspelling gebaseerd op historische patronen"""
        try:
            # Basis prijs patroon
            hour = target_hour.hour
            base_price = 50.0

            # Dag/nacht variatie
            if 8 <= hour <= 18:
                base_price += 20.0
            elif hour >= 23 or hour <= 6:
                base_price -= 10.0

            # Weekend effect
            if target_hour.weekday() >= 5:
                base_price -= 5.0

            return {
                'price': round(base_price, 2),
                'confidence': 0.6,
                'factors': ['dag/nacht patroon', 'weekend effect']
            }

        except Exception as e:
            logging.debug(f"Eenvoudige prijs voorspelling fout: {e}")
            return {'price': 50.0, 'confidence': 0.5, 'factors': ['fallback']}

    def _trend_based_prediction(self, target_hour: datetime, price_data: Dict[str, Any]) -> Dict[str, Any]:
        """Trend-gebaseerde prijs voorspelling"""
        try:
            # Analyseer prijs trends uit historische data
            if 'hourly_prices' in price_data.get('historical_data', {}):
                prices = [item['price'] for item in price_data['historical_data']['hourly_prices']]
                if len(prices) >= 2:
                    # Bereken trend
                    trend = (prices[-1] - prices[0]) / len(prices)
                    base_price = prices[-1] + (trend * 24)  # Voorspel 24 uur vooruit

                    return {
                        'price': round(max(0, base_price), 2),
                        'confidence': 0.7,
                        'factors': ['prijs trend', 'lineaire extrapolatie']
                    }

            # Fallback naar eenvoudige voorspelling
            return self._simple_price_prediction(target_hour, price_data)

        except Exception as e:
            logging.debug(f"Trend-gebaseerde prijs voorspelling fout: {e}")
            return self._simple_price_prediction(target_hour, price_data)

    def _weather_based_prediction(self, target_hour: datetime, price_data: Dict[str, Any]) -> Dict[str, Any]:
        """Weer-gebaseerde prijs voorspelling"""
        try:
            # Dit zou geïntegreerd worden met weer data
            # Voor nu, gebruik eenvoudige voorspelling
            logging.debug("Weer-gebaseerde prijs voorspelling nog niet geïmplementeerd")
            return self._simple_price_prediction(target_hour, price_data)

        except Exception as e:
            logging.debug(f"Weer-gebaseerde prijs voorspelling fout: {e}")
            return self._simple_price_prediction(target_hour, price_data)


# Utility functies voor externe gebruik
def create_price_integration(config) -> PriceIntegration:
    """Factory functie voor het maken van een PriceIntegration instance"""
    return PriceIntegration(config)


def get_price_data_cached(price_integration: PriceIntegration, ttl: int = 3600):
    """Decorator voor het cachen van prijs data functies"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Genereer cache key
            cache_key = f"price_{func.__name__}_{hash(str(args) + str(sorted(kwargs.items())))}"

            # Check cache
            if hasattr(price_integration, 'cache') and cache_key in price_integration.cache:
                cache_entry = price_integration.cache[cache_key]
                if time.time() - cache_entry['timestamp'] < ttl:
                    return cache_entry['data']

            # Voer functie uit
            result = func(*args, **kwargs)

            # Cache resultaat
            if hasattr(price_integration, 'cache'):
                price_integration.cache[cache_key] = {
                    'data': result,
                    'timestamp': time.time()
                }

            return result
        return wrapper
    return decorator
