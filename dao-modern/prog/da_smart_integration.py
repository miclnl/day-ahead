"""
Smart Integration Module
Integreert alle smart optimization features in de bestaande DAO architectuur
"""

import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from dao.prog.da_config import Config
from dao.prog.da_ha_integration import HomeAssistantIntegration
from dao.prog.da_smart_optimization import SmartOptimizationEngine
from dao.prog.da_multiday_optimizer import MultiDayOptimizer
from dao.webserver.da_websocket import get_websocket_server


class SmartDAOIntegration:
    """Hoofdklasse die alle smart features integreert"""
    
    def __init__(self, config: Config):
        self.config = config
        self.ha_integration = None
        self.smart_engine = None
        self.multiday_optimizer = None
        self.is_running = False
        self.last_optimization = None
        self.last_multiday_optimization = None
        
        # Initialize components als features enabled zijn
        self._initialize_components()
        
    def _initialize_components(self):
        """Initialiseer componenten gebaseerd op instellingen"""
        try:
            # Altijd HA integratie initialiseren
            self.ha_integration = HomeAssistantIntegration(self.config)
            
            # Smart optimization engine als enabled
            if self.config.get(['smart_optimization', 'advanced_prediction', 'enabled'], True):
                self.smart_engine = SmartOptimizationEngine(self.config, self.ha_integration)
                logging.info("Smart Optimization Engine geïnitialiseerd")
            
            # Multi-day optimizer als enabled
            if self.config.get(['multiday_optimization', 'enabled'], True):
                self.multiday_optimizer = MultiDayOptimizer(self.config, self.ha_integration)
                logging.info("Multi-day Optimizer geïnitialiseerd")
            
        except Exception as e:
            logging.error(f"Component initialisatie fout: {e}")
    
    async def start_smart_services(self):
        """Start alle smart services"""
        if self.is_running:
            return
            
        try:
            self.is_running = True
            
            # Start HA WebSocket listener
            if self.ha_integration:
                asyncio.create_task(self.ha_integration.start_websocket_listener())
                logging.info("HA WebSocket listener gestart")
            
            # Start high load detection
            if (self.smart_engine and 
                self.config.get(['smart_optimization', 'high_load_detection', 'enabled'], True)):
                await self.smart_engine.load_detector.start_monitoring()
                logging.info("High load detection gestart")
            
            # Start periodic smart optimization
            asyncio.create_task(self._run_smart_optimization_loop())
            logging.info("Smart optimization loop gestart")
            
            # Start periodic multi-day optimization (runs less frequently)
            if self.multiday_optimizer:
                asyncio.create_task(self._run_multiday_optimization_loop())
                logging.info("Multi-day optimization loop gestart")
            
        except Exception as e:
            logging.error(f"Smart services start fout: {e}")
            self.is_running = False
    
    async def stop_smart_services(self):
        """Stop alle smart services"""
        self.is_running = False
        logging.info("Smart services gestopt")
    
    async def _run_smart_optimization_loop(self):
        """Hoofdloop voor smart optimization"""
        while self.is_running:
            try:
                # Check of het tijd is voor nieuwe optimalisatie
                if self._should_run_optimization():
                    await self._run_smart_optimization()
                
                # Wacht 5 minuten voor volgende check
                await asyncio.sleep(300)
                
            except Exception as e:
                logging.error(f"Smart optimization loop fout: {e}")
                await asyncio.sleep(600)  # Wacht langer bij fout
    
    def _should_run_optimization(self) -> bool:
        """Check of smart optimization moet draaien"""
        if not self.smart_engine:
            return False
            
        # Run elke 15 minuten
        if self.last_optimization is None:
            return True
            
        time_since_last = datetime.now() - self.last_optimization
        return time_since_last.total_seconds() > 900  # 15 minuten
    
    async def _run_smart_optimization(self):
        """Voer smart optimization uit"""
        try:
            logging.info("Smart optimization gestart")
            self.last_optimization = datetime.now()
            
            # Haal data op voor optimalisatie
            optimization_data = await self._gather_optimization_data()
            if not optimization_data:
                return
            
            # Run smart optimization
            results = await self.smart_engine.run_smart_optimization(
                optimization_data['prices'],
                optimization_data['weather']
            )
            
            if results.get('success'):
                # Broadcast resultaten naar WebSocket clients
                await self._broadcast_optimization_results(results)
                
                # Update HA sensors
                await self._update_ha_sensors(results)
                
                logging.info(f"Smart optimization voltooid: {results['optimization_results']['potential_savings']}")
            else:
                logging.error(f"Smart optimization fout: {results.get('error')}")
                
        except Exception as e:
            logging.error(f"Smart optimization uitvoering fout: {e}")
    
    async def _run_multiday_optimization_loop(self):
        """Loop voor multi-day optimization (runs elke 6 uur)"""
        while self.is_running:
            try:
                # Check of het tijd is voor multi-day optimization
                if self._should_run_multiday_optimization():
                    await self._run_multiday_optimization()
                
                # Wacht 1 uur voor volgende check (multi-day optimization runs minder frequent)
                await asyncio.sleep(3600)
                
            except Exception as e:
                logging.error(f"Multi-day optimization loop fout: {e}")
                await asyncio.sleep(3600)
    
    def _should_run_multiday_optimization(self) -> bool:
        """Check of multi-day optimization moet draaien"""
        if not self.multiday_optimizer:
            return False
            
        # Run elke 6 uur
        if self.last_multiday_optimization is None:
            return True
            
        time_since_last = datetime.now() - self.last_multiday_optimization
        return time_since_last.total_seconds() > 21600  # 6 uur
    
    async def _run_multiday_optimization(self):
        """Voer multi-day optimization uit"""
        try:
            logging.info("Multi-day optimization gestart")
            self.last_multiday_optimization = datetime.now()
            
            # Run multi-day optimization
            results = await self.multiday_optimizer.run_multiday_optimization()
            
            if results.get('success'):
                # Broadcast resultaten naar WebSocket clients
                await self._broadcast_multiday_results(results)
                
                # Update HA sensors met multi-day data
                await self._update_ha_multiday_sensors(results)
                
                logging.info(f"Multi-day optimization voltooid - Planning voor {results['results']['planning_horizon']} dagen")
            else:
                logging.error(f"Multi-day optimization fout: {results.get('error')}")
                
        except Exception as e:
            logging.error(f"Multi-day optimization uitvoering fout: {e}")
    
    async def _gather_optimization_data(self) -> Optional[Dict[str, Any]]:
        """Verzamel data voor optimization"""
        try:
            # Maak dummy data voor nu - zou uit echte bronnen moeten komen
            import pandas as pd
            import numpy as np
            
            # Price data (24 hours)
            now = datetime.now().replace(minute=0, second=0, microsecond=0)
            hours = pd.date_range(now, periods=24, freq='1H')
            
            # Genereer realistische prijzen (€/kWh)
            base_prices = np.array([
                0.15, 0.12, 0.10, 0.08, 0.07, 0.08,  # 0-5: nacht, goedkoop
                0.12, 0.18, 0.22, 0.20, 0.18, 0.16,  # 6-11: ochtend, stijgend
                0.18, 0.16, 0.15, 0.17, 0.19, 0.25,  # 12-17: middag/avond
                0.28, 0.24, 0.22, 0.20, 0.18, 0.16   # 18-23: piek en afname
            ])
            
            # Voeg wat variatie toe
            prices = base_prices * (1 + np.random.normal(0, 0.1, len(base_prices)))
            prices = np.clip(prices, 0.05, 0.50)
            
            prices_df = pd.DataFrame({'price': prices}, index=hours)
            
            # Weather data (simplified)
            weather_df = pd.DataFrame({
                'temperature': [15 + 5 * np.sin(i * np.pi / 12) for i in range(24)],
                'solar_radiation': [max(0, 800 * np.sin(max(0, (i-6) * np.pi / 12))) for i in range(24)],
                'cloud_cover': [0.3] * 24,
                'wind_speed': [5] * 24
            }, index=hours)
            
            return {
                'prices': prices_df,
                'weather': weather_df
            }
            
        except Exception as e:
            logging.error(f"Data gathering fout: {e}")
            return None
    
    async def _broadcast_optimization_results(self, results: Dict[str, Any]):
        """Broadcast optimization resultaten naar WebSocket clients"""
        try:
            ws_server = get_websocket_server()
            if ws_server:
                # Bereid data voor voor broadcast
                broadcast_data = {
                    'daily_savings': results['optimization_results'].get('potential_savings', {}).get('total_potential_savings', 0),
                    'device_schedules': len(results['optimization_results'].get('device_schedule', {}).get('devices', {})),
                    'battery_optimizations': len(results['optimization_results'].get('battery_strategy', {}).get('charge_periods', [])),
                    'predicted_high_loads': results['optimization_results'].get('predicted_high_loads', 0),
                    'next_optimization': results.get('next_optimization')
                }
                
                await ws_server.send_optimization_completed(broadcast_data)
                
        except Exception as e:
            logging.error(f"WebSocket broadcast fout: {e}")
    
    async def _update_ha_sensors(self, results: Dict[str, Any]):
        """Update HA sensors met optimization resultaten"""
        try:
            if not self.ha_integration:
                return
                
            # Prepare sensor updates
            sensor_data = {
                'status': 'Completed',
                'next_run': results.get('next_optimization'),
                'daily_savings': results['optimization_results'].get('potential_savings', {}).get('total_potential_savings', 0)
            }
            
            # Update DAO sensors in HA
            await self.ha_integration.update_dao_sensors(sensor_data)
            
        except Exception as e:
            logging.error(f"HA sensor update fout: {e}")
    
    async def _broadcast_multiday_results(self, results: Dict[str, Any]):
        """Broadcast multi-day optimization resultaten naar WebSocket clients"""
        try:
            ws_server = get_websocket_server()
            if ws_server:
                # Bereid data voor voor broadcast
                broadcast_data = {
                    'type': 'multiday_optimization_completed',
                    'planning_horizon_days': results['results']['planning_horizon'],
                    'total_projected_savings': results.get('total_projected_savings', 0),
                    'weather_forecast_available': bool(results['results'].get('weather_forecast')),
                    'holiday_periods': len(results['results'].get('holiday_adjustments', {}).get('holidays', [])),
                    'seasonal_optimization': bool(results['results'].get('seasonal_adjustments')),
                    'battery_degradation_optimized': bool(results['results'].get('battery_strategy')),
                    'next_multiday_optimization': results.get('next_optimization')
                }
                
                await ws_server.broadcast_to_all({
                    'type': 'multiday_update',
                    'timestamp': datetime.now().isoformat(),
                    'data': broadcast_data
                })
                
        except Exception as e:
            logging.error(f"Multi-day WebSocket broadcast fout: {e}")
    
    async def _update_ha_multiday_sensors(self, results: Dict[str, Any]):
        """Update HA sensors met multi-day optimization resultaten"""
        try:
            if not self.ha_integration:
                return
                
            # Prepare sensor updates voor multi-day data
            sensor_data = {
                'multiday_status': 'Completed',
                'planning_horizon': results['results']['planning_horizon'],
                'total_projected_savings': results.get('total_projected_savings', 0),
                'next_multiday_run': results.get('next_optimization'),
                'weather_integration': bool(results['results'].get('weather_forecast')),
                'seasonal_optimization': bool(results['results'].get('seasonal_adjustments'))
            }
            
            # Update DAO multi-day sensors in HA (zou custom entities moeten zijn)
            await self.ha_integration.update_dao_sensors(sensor_data)
            
        except Exception as e:
            logging.error(f"HA multi-day sensor update fout: {e}")
    
    async def handle_price_update(self, new_prices: Dict[str, Any]):
        """Handle nieuwe prijzen (voor toekomstige real-time price updates)"""
        try:
            if not self.smart_engine:
                return
                
            logging.info("Nieuwe prijzen ontvangen, re-optimalisatie overwegen")
            
            # Check of prijswijziging significant genoeg is
            # Voor nu: trigger altijd re-optimization bij nieuwe prijzen
            await self._run_smart_optimization()
            
        except Exception as e:
            logging.error(f"Price update handling fout: {e}")
    
    async def handle_high_consumption_detected(self, consumption: float, entity_id: str):
        """Handle hoog verbruik detectie"""
        try:
            if not self.smart_engine or not self.smart_engine.load_detector:
                return
                
            logging.warning(f"Hoog verbruik gedetecteerd: {consumption:.2f} kW van {entity_id}")
            
            # Trigger immediate response
            await self.smart_engine.load_detector._handle_high_load_event(consumption)
            
            # Send WebSocket notification
            ws_server = get_websocket_server()
            if ws_server:
                await ws_server.send_error_notification(
                    'high_consumption',
                    f"Hoog verbruik gedetecteerd: {consumption:.2f} kW. Automatische maatregelen genomen."
                )
                
        except Exception as e:
            logging.error(f"High consumption handling fout: {e}")
    
    async def get_smart_status(self) -> Dict[str, Any]:
        """Krijg status van alle smart features"""
        try:
            status = {
                'smart_services_running': self.is_running,
                'last_optimization': self.last_optimization.isoformat() if self.last_optimization else None,
                'ha_integration': {
                    'connected': False,
                    'websocket_active': False
                },
                'features': {
                    'advanced_prediction': self.config.get(['smart_optimization', 'advanced_prediction', 'enabled'], True),
                    'device_scheduling': self.config.get(['smart_optimization', 'device_scheduling', 'enabled'], True),
                    'high_load_detection': self.config.get(['smart_optimization', 'high_load_detection', 'enabled'], True),
                    'adaptive_battery': self.config.get(['smart_optimization', 'adaptive_battery', 'enabled'], True)
                }
            }
            
            # Check HA integration status
            if self.ha_integration:
                ha_test = await self.ha_integration.test_connection()
                status['ha_integration']['connected'] = ha_test.get('success', False)
                status['ha_integration']['websocket_active'] = self.ha_integration.ws_connection is not None
            
            return status
            
        except Exception as e:
            logging.error(f"Smart status fout: {e}")
            return {'error': str(e)}
    
    async def manual_optimization_trigger(self) -> Dict[str, Any]:
        """Trigger manual smart optimization"""
        try:
            logging.info("Handmatige smart optimization gestart")
            await self._run_smart_optimization()
            return {'success': True, 'message': 'Smart optimization gestart'}
            
        except Exception as e:
            logging.error(f"Manual optimization fout: {e}")
            return {'success': False, 'error': str(e)}


# Global instance
smart_dao: Optional[SmartDAOIntegration] = None


def initialize_smart_dao(config: Config) -> SmartDAOIntegration:
    """Initialiseer globale Smart DAO instance"""
    global smart_dao
    smart_dao = SmartDAOIntegration(config)
    return smart_dao


def get_smart_dao() -> Optional[SmartDAOIntegration]:
    """Krijg globale Smart DAO instance"""
    return smart_dao


async def start_smart_dao_services(config: Config):
    """Start alle Smart DAO services"""
    try:
        smart_integration = initialize_smart_dao(config)
        await smart_integration.start_smart_services()
        logging.info("Smart DAO services gestart")
    except Exception as e:
        logging.error(f"Smart DAO services start fout: {e}")


async def stop_smart_dao_services():
    """Stop alle Smart DAO services"""
    global smart_dao
    if smart_dao:
        await smart_dao.stop_smart_services()
        smart_dao = None
        logging.info("Smart DAO services gestopt")