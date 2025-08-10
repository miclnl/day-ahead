"""
AI-powered optimization module for Day Ahead Optimizer.
Provides configurable AI integration for advanced optimization strategies.

Now enhanced with DSPy for better structured AI interactions.
"""

import asyncio
import aiohttp
import json
import logging
import datetime
from typing import Dict, List, Optional, Any, Union
from abc import ABC, abstractmethod
import pandas as pd

from dao.prog.da_config import Config

# Import DSPy optimizer if available
try:
    from dao.prog.da_dspy_optimizer import (
        DSPyAIOptimizer, 
        EnergySystemData, 
        OptimizationResult,
        create_dspy_optimizer
    )
    DSPY_AVAILABLE = True
    logging.info("DSPy AI optimization available")
except ImportError as e:
    DSPY_AVAILABLE = False
    logging.warning(f"DSPy not available, falling back to basic AI: {e}")


class AIOptimizerBase(ABC):
    """Abstract base class for AI optimization providers"""
    
    @abstractmethod
    async def optimize(self, optimization_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Perform AI-powered optimization"""
        pass
    
    @abstractmethod
    async def validate_config(self) -> bool:
        """Validate AI provider configuration"""
        pass


class OpenAIOptimizer(AIOptimizerBase):
    """OpenAI GPT-4 based optimization"""
    
    def __init__(self, config: Dict[str, Any]):
        self.api_key = config.get('api_key')
        self.model = config.get('model', 'gpt-4o')
        self.base_url = config.get('endpoint', 'https://api.openai.com/v1')
        self.max_tokens = config.get('max_tokens', 4000)
        self.temperature = config.get('temperature', 0.1)
        self.timeout = config.get('timeout', 60)
    
    async def validate_config(self) -> bool:
        """Validate OpenAI configuration"""
        if not self.api_key:
            logging.error("OpenAI API key not configured")
            return False
        
        try:
            async with aiohttp.ClientSession() as session:
                headers = {
                    'Authorization': f'Bearer {self.api_key}',
                    'Content-Type': 'application/json'
                }
                
                # Test with a simple models request
                async with session.get(
                    f'{self.base_url}/models',
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    return response.status == 200
                    
        except Exception as e:
            logging.error(f"OpenAI config validation failed: {e}")
            return False
    
    async def optimize(self, optimization_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Perform OpenAI-powered optimization"""
        logging.info("Running OpenAI optimization")
        
        try:
            # Prepare prompt
            prompt = self._build_optimization_prompt(optimization_data)
            
            async with aiohttp.ClientSession() as session:
                headers = {
                    'Authorization': f'Bearer {self.api_key}',
                    'Content-Type': 'application/json'
                }
                
                payload = {
                    'model': self.model,
                    'messages': [
                        {
                            'role': 'system',
                            'content': self._get_system_prompt()
                        },
                        {
                            'role': 'user', 
                            'content': prompt
                        }
                    ],
                    'max_tokens': self.max_tokens,
                    'temperature': self.temperature,
                    'response_format': {'type': 'json_object'}
                }
                
                async with session.post(
                    f'{self.base_url}/chat/completions',
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=self.timeout)
                ) as response:
                    
                    if response.status != 200:
                        logging.error(f"OpenAI API error: {response.status}")
                        return None
                    
                    result = await response.json()
                    
                    # Extract and parse AI response
                    ai_content = result['choices'][0]['message']['content']
                    optimization_result = json.loads(ai_content)
                    
                    return self._process_ai_response(optimization_result, optimization_data)
                    
        except Exception as e:
            logging.error(f"OpenAI optimization error: {e}")
            return None
    
    def _get_system_prompt(self) -> str:
        """Get system prompt for energy optimization"""
        return """You are an expert energy optimization system for home energy management.
        
Your task is to optimize energy consumption, battery usage, EV charging, and heating schedules based on:
- Dynamic electricity prices
- Solar production forecasts
- Weather conditions
- User preferences and constraints
- Battery characteristics
- EV charging requirements
- Heating/cooling needs

You must respond with a valid JSON object containing optimized schedules.
Focus on minimizing costs while respecting all technical constraints.
Consider grid stability, battery life, and user comfort.

Always provide practical, implementable solutions."""
    
    def _build_optimization_prompt(self, data: Dict[str, Any]) -> str:
        """Build detailed optimization prompt from data"""
        prompt_parts = [
            "Please optimize the following energy system configuration:",
            "",
            "## Time Period",
            f"Optimization period: {data.get('time_periods', 24)} hours",
            f"Start time: {data.get('start_dt', 'Not specified')}",
            "",
            "## Energy Prices (€/kWh)",
        ]
        
        # Add price data
        if 'price_data' in data:
            price_data = data['price_data']
            for i, price in enumerate(price_data.get('consumption_prices', [])):
                timestamp = price_data.get('timestamps', [])[i] if i < len(price_data.get('timestamps', [])) else i
                prompt_parts.append(f"Hour {i}: €{price:.4f} (consumption), €{price_data.get('production_prices', [price])[i]:.4f} (production)")
        
        prompt_parts.extend([
            "",
            "## Solar Production Forecast (kW)",
        ])
        
        # Add solar data
        if 'solar_data' in data:
            solar_data = data['solar_data']
            for i, production_list in enumerate(solar_data.get('production_forecasts', [])):
                total_production = sum(production_list) if production_list else 0
                prompt_parts.append(f"Hour {i}: {total_production:.2f} kW")
        
        prompt_parts.extend([
            "",
            "## Baseload Consumption (kW)",
        ])
        
        # Add baseload data
        if 'baseload_data' in data:
            baseload = data['baseload_data'].get('baseload_consumption', [])
            for i, consumption in enumerate(baseload):
                prompt_parts.append(f"Hour {i}: {consumption:.2f} kW")
        
        # Add battery configuration
        if 'battery_data' in data and data['battery_data'].get('batteries'):
            prompt_parts.extend([
                "",
                "## Battery Configuration",
            ])
            
            for i, battery in enumerate(data['battery_data']['batteries']):
                prompt_parts.extend([
                    f"Battery {i}:",
                    f"  - Capacity: {battery['capacity']:.1f} kWh",
                    f"  - Max charge: {battery['max_charge_power']:.1f} kW", 
                    f"  - Max discharge: {battery['max_discharge_power']:.1f} kW",
                    f"  - Start SoC: {battery['start_soc']*100:.1f}%",
                    f"  - Min SoC: {battery['min_soc']*100:.1f}%",
                    f"  - Max SoC: {battery['max_soc']*100:.1f}%",
                ])
        
        # Add EV configuration
        if 'ev_data' in data and data['ev_data'].get('electric_vehicles'):
            prompt_parts.extend([
                "",
                "## Electric Vehicle Configuration",
            ])
            
            for i, ev in enumerate(data['ev_data']['electric_vehicles']):
                prompt_parts.extend([
                    f"EV {i}:",
                    f"  - Capacity: {ev['capacity']:.1f} kWh",
                    f"  - Charge power: {ev['charge_power']:.1f} kW",
                    f"  - Start SoC: {ev['start_soc']*100:.1f}%",
                    f"  - Target SoC: {ev['target_soc']*100:.1f}%",
                    f"  - Departure: {ev.get('departure_time', 'Not specified')}",
                ])
        
        # Add heating configuration
        if 'heating_data' in data:
            heating = data['heating_data']
            if heating.get('boiler_enabled') or heating.get('heat_pump_enabled'):
                prompt_parts.extend([
                    "",
                    "## Heating Configuration",
                ])
                
                if heating.get('boiler_enabled'):
                    prompt_parts.append(f"Boiler: {heating.get('boiler_power', 2000)}W")
                
                if heating.get('heat_pump_enabled'):
                    prompt_parts.extend([
                        f"Heat pump: {heating.get('heat_pump_power', 3000)}W",
                        f"COP: {heating.get('heat_pump_cop', 3.5)}",
                    ])
        
        prompt_parts.extend([
            "",
            "## Optimization Objective",
            "Minimize total energy costs while respecting all constraints.",
            "",
            "## Required Response Format",
            "Respond with a JSON object containing:",
            "```json",
            "{",
            '  "grid_import": [hourly_values_in_kW],',
            '  "grid_export": [hourly_values_in_kW],',
            '  "batteries": [',
            "    {",
            '      "charge_power": [hourly_values_in_kW],',
            '      "discharge_power": [hourly_values_in_kW],',
            '      "soc": [hourly_soc_percentages]',
            "    }",
            "  ],",
            '  "electric_vehicles": [',
            "    {",
            '      "charge_power": [hourly_values_in_kW],',
            '      "soc": [hourly_soc_percentages]',
            "    }",
            "  ],",
            '  "heating": {',
            '    "boiler_heating": [hourly_binary_values],',
            '    "heat_pump_power": [hourly_values_in_kW]',
            "  },",
            '  "total_cost": estimated_total_cost_in_euros,',
            '  "optimization_notes": "Brief explanation of key decisions"',
            "}",
            "```",
        ])
        
        return "\\n".join(prompt_parts)
    
    def _process_ai_response(self, ai_result: Dict, original_data: Dict) -> Dict[str, Any]:
        """Process and validate AI optimization response"""
        try:
            # Validate required fields
            required_fields = ['grid_import', 'grid_export']
            for field in required_fields:
                if field not in ai_result:
                    logging.error(f"AI response missing required field: {field}")
                    return None
            
            # Validate data lengths
            expected_length = original_data.get('time_periods', 24)
            
            for field in ['grid_import', 'grid_export']:
                if len(ai_result[field]) != expected_length:
                    logging.error(f"AI response {field} length mismatch: expected {expected_length}, got {len(ai_result[field])}")
                    return None
            
            # Add timestamps
            ai_result['timestamps'] = original_data.get('price_data', {}).get('timestamps', [])
            
            logging.info(f"AI optimization completed. Estimated cost: €{ai_result.get('total_cost', 'N/A')}")
            logging.info(f"AI notes: {ai_result.get('optimization_notes', 'None provided')}")
            
            return ai_result
            
        except Exception as e:
            logging.error(f"Error processing AI response: {e}")
            return None


class AnthropicOptimizer(AIOptimizerBase):
    """Anthropic Claude based optimization"""
    
    def __init__(self, config: Dict[str, Any]):
        self.api_key = config.get('api_key')
        self.model = config.get('model', 'claude-3-5-sonnet-20241022')
        self.base_url = config.get('endpoint', 'https://api.anthropic.com/v1')
        self.max_tokens = config.get('max_tokens', 4000)
        self.timeout = config.get('timeout', 60)
    
    async def validate_config(self) -> bool:
        """Validate Anthropic configuration"""
        if not self.api_key:
            logging.error("Anthropic API key not configured")
            return False
        return True  # Simplified validation
    
    async def optimize(self, optimization_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Perform Anthropic-powered optimization"""
        logging.info("Running Anthropic optimization")
        
        try:
            # Use similar approach as OpenAI but with Anthropic API format
            prompt = self._build_anthropic_prompt(optimization_data)
            
            async with aiohttp.ClientSession() as session:
                headers = {
                    'x-api-key': self.api_key,
                    'Content-Type': 'application/json',
                    'anthropic-version': '2023-06-01'
                }
                
                payload = {
                    'model': self.model,
                    'max_tokens': self.max_tokens,
                    'messages': [
                        {
                            'role': 'user',
                            'content': prompt
                        }
                    ]
                }
                
                async with session.post(
                    f'{self.base_url}/messages',
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=self.timeout)
                ) as response:
                    
                    if response.status != 200:
                        logging.error(f"Anthropic API error: {response.status}")
                        return None
                    
                    result = await response.json()
                    ai_content = result['content'][0]['text']
                    
                    # Extract JSON from response
                    json_start = ai_content.find('{')
                    json_end = ai_content.rfind('}') + 1
                    
                    if json_start >= 0 and json_end > json_start:
                        json_content = ai_content[json_start:json_end]
                        optimization_result = json.loads(json_content)
                        return self._process_ai_response(optimization_result, optimization_data)
                    
                    return None
                    
        except Exception as e:
            logging.error(f"Anthropic optimization error: {e}")
            return None
    
    def _build_anthropic_prompt(self, data: Dict[str, Any]) -> str:
        """Build Anthropic-specific prompt"""
        # Similar to OpenAI but adapted for Claude's preferences
        return f"""I need you to optimize an energy system. You are an expert energy optimization AI.

{self._build_optimization_context(data)}

Please provide an optimal energy schedule as a JSON object. Focus on minimizing costs while respecting all technical constraints."""
    
    def _build_optimization_context(self, data: Dict[str, Any]) -> str:
        """Build context for Anthropic prompt"""
        # Reuse OpenAI prompt building logic
        openai_optimizer = OpenAIOptimizer({})
        return openai_optimizer._build_optimization_prompt(data)
    
    def _process_ai_response(self, ai_result: Dict, original_data: Dict) -> Dict[str, Any]:
        """Process Anthropic AI response - same as OpenAI"""
        openai_optimizer = OpenAIOptimizer({})
        return openai_optimizer._process_ai_response(ai_result, original_data)


class LocalAIOptimizer(AIOptimizerBase):
    """Local AI model optimization (e.g., Ollama, local OpenAI API)"""
    
    def __init__(self, config: Dict[str, Any]):
        self.base_url = config.get('endpoint', 'http://localhost:11434')
        self.model = config.get('model', 'llama3')
        self.timeout = config.get('timeout', 120)
    
    async def validate_config(self) -> bool:
        """Validate local AI configuration"""
        try:
            async with aiohttp.ClientSession() as session:
                # Test connection to local AI endpoint
                async with session.get(
                    f'{self.base_url}/api/tags',
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as response:
                    return response.status == 200
        except Exception as e:
            logging.error(f"Local AI config validation failed: {e}")
            return False
    
    async def optimize(self, optimization_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Perform local AI optimization"""
        logging.info("Running local AI optimization")
        
        try:
            prompt = self._build_local_ai_prompt(optimization_data)
            
            async with aiohttp.ClientSession() as session:
                payload = {
                    'model': self.model,
                    'prompt': prompt,
                    'stream': False,
                    'format': 'json'
                }
                
                async with session.post(
                    f'{self.base_url}/api/generate',
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=self.timeout)
                ) as response:
                    
                    if response.status != 200:
                        logging.error(f"Local AI API error: {response.status}")
                        return None
                    
                    result = await response.json()
                    ai_content = result.get('response', '')
                    
                    try:
                        optimization_result = json.loads(ai_content)
                        return self._process_ai_response(optimization_result, optimization_data)
                    except json.JSONDecodeError:
                        logging.error("Local AI returned invalid JSON")
                        return None
                    
        except Exception as e:
            logging.error(f"Local AI optimization error: {e}")
            return None
    
    def _build_local_ai_prompt(self, data: Dict[str, Any]) -> str:
        """Build prompt for local AI"""
        # Simpler prompt for local models
        return f"""Optimize this energy system. Respond with JSON only.

Time periods: {data.get('time_periods', 24)}
Prices: {data.get('price_data', {}).get('consumption_prices', [])}
Solar production: Available
Batteries: {len(data.get('battery_data', {}).get('batteries', []))}
EVs: {len(data.get('ev_data', {}).get('electric_vehicles', []))}

Minimize cost. Return JSON with grid_import, grid_export arrays and total_cost."""
    
    def _process_ai_response(self, ai_result: Dict, original_data: Dict) -> Dict[str, Any]:
        """Process local AI response"""
        # Simplified processing for local AI
        try:
            if 'grid_import' in ai_result and 'grid_export' in ai_result:
                ai_result['timestamps'] = original_data.get('price_data', {}).get('timestamps', [])
                return ai_result
            return None
        except Exception as e:
            logging.error(f"Error processing local AI response: {e}")
            return None


class AIOptimizationManager:
    """Main AI optimization manager with DSPy and fallback strategies"""
    
    def __init__(self, config: Config):
        self.config = config
        self.ai_config = config.get(['optimization', 'ai'], {})
        self.enabled = self.ai_config.get('enabled', False)
        self.provider = self.ai_config.get('provider', 'openai').lower()
        self.fallback_to_local = self.ai_config.get('fallback_to_local', True)
        self.cost_threshold = self.ai_config.get('cost_threshold', 0.10)  # Max cost per optimization
        self.use_dspy = self.ai_config.get('use_dspy', True)  # Prefer DSPy when available
        
        self.optimizer = None
        self.dspy_optimizer = None
        self._initialize_optimizer()
    
    def _initialize_optimizer(self):
        """Initialize the appropriate AI optimizer (DSPy first, then fallback)"""
        if not self.enabled:
            return
        
        # Try DSPy first if available and preferred
        if DSPY_AVAILABLE and self.use_dspy:
            try:
                self.dspy_optimizer = DSPyAIOptimizer(self.config)
                if self.dspy_optimizer.validate_configuration():
                    logging.info(f"DSPy optimizer initialized successfully with {self.provider}")
                    return
                else:
                    logging.warning("DSPy optimizer validation failed, falling back to legacy")
            except Exception as e:
                logging.warning(f"DSPy initialization failed: {e}, falling back to legacy")
        
        # Fallback to legacy optimizers
        try:
            if self.provider == 'openai':
                self.optimizer = OpenAIOptimizer(self.ai_config.get('openai', {}))
            elif self.provider == 'anthropic':
                self.optimizer = AnthropicOptimizer(self.ai_config.get('anthropic', {}))
            elif self.provider == 'local':
                self.optimizer = LocalAIOptimizer(self.ai_config.get('local', {}))
            else:
                logging.error(f"Unknown AI provider: {self.provider}")
                
        except Exception as e:
            logging.error(f"Error initializing AI optimizer: {e}")
            self.optimizer = None
    
    async def is_available(self) -> bool:
        """Check if AI optimization is available"""
        if not self.enabled or not self.optimizer:
            return False
        
        try:
            return await self.optimizer.validate_config()
        except Exception as e:
            logging.error(f"AI availability check failed: {e}")
            return False
    
    async def optimize(self, optimization_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Perform AI optimization with fallback strategies.
        Returns None if AI optimization fails and local fallback should be used.
        """
        if not self.enabled:
            logging.debug("AI optimization disabled")
            return None
        
        if not await self.is_available():
            logging.warning("AI optimization not available")
            return None
        
        try:
            # Estimate cost before proceeding
            estimated_cost = self._estimate_optimization_cost(optimization_data)
            if estimated_cost > self.cost_threshold:
                logging.warning(f"AI optimization cost ${estimated_cost:.3f} exceeds threshold ${self.cost_threshold:.3f}")
                return None
            
            # Use DSPy optimizer if available
            if self.dspy_optimizer:
                result = await self._optimize_with_dspy(optimization_data)
                if result:
                    logging.info("DSPy AI optimization completed successfully")
                    return result
                else:
                    logging.warning("DSPy optimization failed, trying legacy")
            
            # Fallback to legacy optimizer
            if self.optimizer:
                result = await self.optimizer.optimize(optimization_data)
                
                if result:
                    logging.info("Legacy AI optimization completed successfully")
                    return result
                else:
                    logging.error("AI optimization failed")
                    
                    # Try fallback to local AI if configured
                    if self.fallback_to_local and self.provider != 'local':
                        logging.info("Attempting fallback to local AI")
                        local_optimizer = LocalAIOptimizer(self.ai_config.get('local', {}))
                        return await local_optimizer.optimize(optimization_data)
            
            return None
                
        except Exception as e:
            logging.error(f"AI optimization error: {e}")
            return None
    
    async def _optimize_with_dspy(self, optimization_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Use DSPy optimizer for structured AI optimization"""
        try:
            # Convert optimization data to EnergySystemData format
            system_data = self._convert_to_energy_system_data(optimization_data)
            
            # Perform DSPy optimization
            result = await self.dspy_optimizer.optimize_energy_system(
                system_data=system_data,
                objective="minimize_cost"
            )
            
            if result and result.battery_schedule:
                # Convert DSPy result back to legacy format
                return self._convert_dspy_result_to_legacy(result, optimization_data)
            else:
                logging.warning("DSPy optimization returned no valid result")
                return None
                
        except Exception as e:
            logging.error(f"DSPy optimization failed: {e}")
            return None
    
    def _convert_to_energy_system_data(self, data: Dict[str, Any]) -> EnergySystemData:
        """Convert legacy optimization data to EnergySystemData format"""
        time_periods = data.get('time_periods', 24)
        start_dt = data.get('start_dt', datetime.datetime.now().isoformat())
        
        # Extract price data
        price_data = data.get('price_data', {})
        electricity_prices = price_data.get('consumption_prices', [0.20] * time_periods)
        
        # Extract solar forecast
        solar_data = data.get('solar_data', {})
        solar_forecasts = solar_data.get('production_forecasts', [])
        solar_forecast = []
        for hour_forecasts in solar_forecasts:
            if hour_forecasts:
                solar_forecast.append(sum(hour_forecasts))
            else:
                solar_forecast.append(0.0)
        
        # Pad or truncate to correct length
        while len(solar_forecast) < time_periods:
            solar_forecast.append(0.0)
        solar_forecast = solar_forecast[:time_periods]
        
        # Extract consumption forecast
        baseload_data = data.get('baseload_data', {})
        consumption_forecast = baseload_data.get('baseload_consumption', [1.5] * time_periods)
        
        # Extract battery configuration
        battery_data = data.get('battery_data', {})
        batteries = battery_data.get('batteries', [])
        battery_config = {}
        if batteries:
            # Use first battery for now
            battery = batteries[0]
            battery_config = {
                "capacity": battery.get('capacity', 10.0),
                "max_charge_power": battery.get('max_charge_power', 5.0),
                "max_discharge_power": battery.get('max_discharge_power', 5.0),
                "start_soc": battery.get('start_soc', 0.5),
                "min_soc": battery.get('min_soc', 0.1),
                "max_soc": battery.get('max_soc', 0.9),
                "efficiency": battery.get('efficiency', 0.95)
            }
        
        # Extract EV configuration
        ev_data = data.get('ev_data', {})
        ev_config = None
        if ev_data.get('electric_vehicles'):
            evs = ev_data['electric_vehicles']
            if evs:
                ev = evs[0]
                ev_config = {
                    "capacity": ev.get('capacity', 50.0),
                    "charge_power": ev.get('charge_power', 11.0),
                    "start_soc": ev.get('start_soc', 0.3),
                    "target_soc": ev.get('target_soc', 0.8),
                    "departure_time": ev.get('departure_time', '08:00')
                }
        
        # Extract heating configuration
        heating_data = data.get('heating_data', {})
        heating_config = None
        if heating_data:
            heating_config = {
                "boiler_enabled": heating_data.get('boiler_enabled', False),
                "heat_pump_enabled": heating_data.get('heat_pump_enabled', False),
                "boiler_power": heating_data.get('boiler_power', 2000),
                "heat_pump_power": heating_data.get('heat_pump_power', 3000),
                "heat_pump_cop": heating_data.get('heat_pump_cop', 3.5)
            }
        
        return EnergySystemData(
            time_periods=time_periods,
            start_dt=start_dt,
            electricity_prices=electricity_prices,
            solar_forecast=solar_forecast,
            consumption_forecast=consumption_forecast,
            battery_config=battery_config,
            ev_config=ev_config,
            heating_config=heating_config
        )
    
    def _convert_dspy_result_to_legacy(self, result: OptimizationResult, original_data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert DSPy optimization result to legacy format"""
        time_periods = original_data.get('time_periods', 24)
        timestamps = original_data.get('price_data', {}).get('timestamps', [])
        
        # Build legacy format result
        legacy_result = {
            'grid_import': [max(0, val) for val in result.battery_schedule],
            'grid_export': [max(0, -val) for val in result.battery_schedule],
            'total_cost': result.expected_cost,
            'optimization_notes': result.optimization_strategy,
            'timestamps': timestamps
        }
        
        # Add battery information
        if result.battery_schedule:
            legacy_result['batteries'] = [{
                'charge_power': [max(0, val) for val in result.battery_schedule],
                'discharge_power': [max(0, -val) for val in result.battery_schedule],
                'soc': [50.0] * time_periods  # Simplified SoC calculation
            }]
        
        # Add EV information
        if result.ev_charging_schedule:
            legacy_result['electric_vehicles'] = [{
                'charge_power': result.ev_charging_schedule,
                'soc': [30.0 + (i * 2.0) for i in range(time_periods)]  # Simplified
            }]
        
        # Add heating information
        if result.heating_schedule:
            legacy_result['heating'] = {
                'boiler_heating': [1 if val > 0 else 0 for val in result.heating_schedule],
                'heat_pump_power': result.heating_schedule
            }
        
        return legacy_result

    def _estimate_optimization_cost(self, data: Dict[str, Any]) -> float:
        """Estimate the cost of AI optimization"""
        if self.provider == 'local':
            return 0.0  # Local AI is free
        
        # Rough estimation based on data complexity
        time_periods = data.get('time_periods', 24)
        batteries = len(data.get('battery_data', {}).get('batteries', []))
        evs = len(data.get('ev_data', {}).get('electric_vehicles', []))
        
        # Estimate tokens (very rough)
        estimated_input_tokens = 1000 + (time_periods * 50) + (batteries * 100) + (evs * 100)
        estimated_output_tokens = 500 + (time_periods * 10)
        
        if self.provider == 'openai':
            # GPT-4o pricing (rough estimate)
            input_cost = (estimated_input_tokens / 1000) * 0.005
            output_cost = (estimated_output_tokens / 1000) * 0.015
            return input_cost + output_cost
        
        elif self.provider == 'anthropic':
            # Claude pricing (rough estimate)
            input_cost = (estimated_input_tokens / 1000) * 0.003
            output_cost = (estimated_output_tokens / 1000) * 0.015
            return input_cost + output_cost
        
        return 0.01  # Default small cost