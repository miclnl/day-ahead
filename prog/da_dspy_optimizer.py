"""
DSPy-powered AI optimization module for Day Ahead Optimizer.
Modern, structured approach to AI-driven energy optimization.
"""

import dspy
import json
import logging
import datetime
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
import pandas as pd
from abc import ABC, abstractmethod

# Config handled by da_calc_instance parameter


@dataclass
class EnergySystemData:
    """Structured representation of energy system data for optimization"""
    time_periods: int
    start_dt: str
    electricity_prices: List[float]
    solar_forecast: List[float]
    consumption_forecast: List[float]
    battery_config: Dict[str, Any]
    ev_config: Optional[Dict[str, Any]] = None
    heating_config: Optional[Dict[str, Any]] = None
    weather_data: Optional[Dict[str, Any]] = None
    constraints: Optional[Dict[str, Any]] = None


@dataclass
class OptimizationResult:
    """Structured optimization result"""
    battery_schedule: List[float]
    ev_charging_schedule: Optional[List[float]] = None
    heating_schedule: Optional[List[float]] = None
    expected_cost: float = 0.0
    expected_savings: float = 0.0
    confidence_score: float = 0.0
    optimization_strategy: str = ""
    warnings: List[str] = None


class EnergyOptimizationSignature(dspy.Signature):
    """DSPy signature for energy optimization tasks"""
    
    system_data = dspy.InputField(desc="Complete energy system configuration and forecasts")
    optimization_objective = dspy.InputField(desc="Primary optimization objective (cost, comfort, sustainability)")
    constraints = dspy.InputField(desc="Technical and user constraints")
    
    optimized_schedule = dspy.OutputField(desc="Optimized energy schedules in JSON format")
    cost_analysis = dspy.OutputField(desc="Cost analysis and expected savings")
    strategy_explanation = dspy.OutputField(desc="Explanation of optimization strategy")


class EnergyOptimizer(dspy.Module):
    """DSPy module for energy optimization"""
    
    def __init__(self):
        super().__init__()
        self.optimizer = dspy.ChainOfThought(EnergyOptimizationSignature)
        
    def forward(self, system_data: str, objective: str, constraints: str):
        """Forward pass through the optimization module"""
        return self.optimizer(
            system_data=system_data,
            optimization_objective=objective, 
            constraints=constraints
        )


class BatteryOptimizationSignature(dspy.Signature):
    """Specialized signature for battery optimization"""
    
    prices = dspy.InputField(desc="24-hour electricity prices")
    solar_forecast = dspy.InputField(desc="Solar production forecast")
    consumption_forecast = dspy.InputField(desc="Consumption forecast")
    battery_specs = dspy.InputField(desc="Battery specifications and constraints")
    
    charging_schedule = dspy.OutputField(desc="Optimal charging schedule (positive=charge, negative=discharge)")
    soc_schedule = dspy.OutputField(desc="State of charge schedule")
    cost_savings = dspy.OutputField(desc="Expected cost savings in euros")


class BatteryOptimizer(dspy.Module):
    """Specialized DSPy module for battery optimization"""
    
    def __init__(self):
        super().__init__()
        self.optimizer = dspy.ChainOfThought(BatteryOptimizationSignature)
        
    def forward(self, prices: str, solar: str, consumption: str, battery: str):
        return self.optimizer(
            prices=prices,
            solar_forecast=solar,
            consumption_forecast=consumption,
            battery_specs=battery
        )


class DSPyAIOptimizer:
    """DSPy-powered AI optimizer for DAO"""
    
    def __init__(self, da_calc_instance):
        self.da_calc = da_calc_instance
        self.config = da_calc_instance.config
        self.ai_config = self.config.get(['optimization', 'ai'], 0, {})
        self.provider = self.ai_config.get('provider', 'openai')
        
        # Initialize DSPy language model
        self.lm = self._initialize_language_model()
        if self.lm:
            dspy.settings.configure(lm=self.lm)
            
        # Initialize optimization modules
        self.energy_optimizer = EnergyOptimizer()
        self.battery_optimizer = BatteryOptimizer()
        
        logging.info(f"DSPy AI Optimizer initialized with {self.provider}")
    
    def _initialize_language_model(self) -> Optional[dspy.LM]:
        """Initialize the appropriate DSPy language model"""
        try:
            if self.provider == 'openai':
                api_key = self.ai_config.get('openai', {}).get('api_key')
                model = self.ai_config.get('openai', {}).get('model', 'gpt-4o')
                
                if not api_key:
                    logging.warning("OpenAI API key not found")
                    return None
                    
                return dspy.OpenAI(
                    model=model,
                    api_key=api_key,
                    max_tokens=4000,
                    temperature=0.1
                )
                
            elif self.provider == 'anthropic':
                api_key = self.ai_config.get('anthropic', {}).get('api_key')
                model = self.ai_config.get('anthropic', {}).get('model', 'claude-3-haiku-20240307')
                
                if not api_key:
                    logging.warning("Anthropic API key not found")
                    return None
                    
                return dspy.Claude(
                    model=model,
                    api_key=api_key,
                    max_tokens=4000
                )
                
            elif self.provider == 'local':
                # For local models (Ollama, etc.)
                endpoint = self.ai_config.get('local', {}).get('endpoint', 'http://localhost:11434')
                model = self.ai_config.get('local', {}).get('model', 'llama3.1')
                
                return dspy.OllamaLocal(
                    model=model,
                    base_url=endpoint,
                    max_tokens=4000
                )
                
            else:
                logging.error(f"Unsupported AI provider: {self.provider}")
                return None
                
        except Exception as e:
            logging.error(f"Failed to initialize language model: {e}")
            return None
    
    async def optimize_energy_system(self, system_data: EnergySystemData, 
                                   objective: str = "minimize_cost") -> Optional[OptimizationResult]:
        """Perform comprehensive energy system optimization"""
        
        if not self.lm:
            logging.error("No language model available for optimization")
            return None
            
        try:
            # Prepare system data as structured string
            system_str = self._format_system_data(system_data)
            constraints_str = self._format_constraints(system_data.constraints or {})
            
            # Run optimization
            result = self.energy_optimizer(
                system_data=system_str,
                objective=objective,
                constraints=constraints_str
            )
            
            # Parse results
            return self._parse_optimization_result(result, system_data)
            
        except Exception as e:
            logging.error(f"Energy optimization failed: {e}")
            return None
    
    async def optimize_battery_only(self, prices: List[float], solar_forecast: List[float],
                                  consumption_forecast: List[float], 
                                  battery_config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Specialized battery optimization using DSPy"""
        
        if not self.lm:
            logging.error("No language model available for battery optimization")
            return None
            
        try:
            # Format inputs for DSPy
            prices_str = json.dumps(prices)
            solar_str = json.dumps(solar_forecast)
            consumption_str = json.dumps(consumption_forecast)
            battery_str = json.dumps(battery_config)
            
            # Run battery optimization
            result = self.battery_optimizer(
                prices=prices_str,
                solar=solar_str,
                consumption=consumption_str,
                battery=battery_str
            )
            
            return self._parse_battery_result(result)
            
        except Exception as e:
            logging.error(f"Battery optimization failed: {e}")
            return None
    
    def _format_system_data(self, data: EnergySystemData) -> str:
        """Format energy system data for AI processing"""
        formatted = {
            "time_config": {
                "periods": data.time_periods,
                "start_time": data.start_dt
            },
            "market_data": {
                "electricity_prices": data.electricity_prices,
                "price_unit": "EUR/kWh"
            },
            "forecasts": {
                "solar_production": data.solar_forecast,
                "consumption": data.consumption_forecast,
                "unit": "kWh"
            },
            "assets": {
                "battery": data.battery_config,
                "ev": data.ev_config,
                "heating": data.heating_config
            }
        }
        
        if data.weather_data:
            formatted["weather"] = data.weather_data
            
        return json.dumps(formatted, indent=2)
    
    def _format_constraints(self, constraints: Dict[str, Any]) -> str:
        """Format constraints for AI processing"""
        return json.dumps({
            "technical_constraints": constraints.get("technical", {}),
            "user_preferences": constraints.get("user", {}),
            "comfort_requirements": constraints.get("comfort", {}),
            "safety_limits": constraints.get("safety", {})
        }, indent=2)
    
    def _parse_optimization_result(self, result, system_data: EnergySystemData) -> OptimizationResult:
        """Parse DSPy optimization result into structured format"""
        try:
            # Extract JSON from the AI response
            schedule_data = json.loads(result.optimized_schedule)
            
            # Parse battery schedule
            battery_schedule = schedule_data.get("battery_schedule", [0] * system_data.time_periods)
            
            # Parse optional schedules
            ev_schedule = None
            if system_data.ev_config and "ev_schedule" in schedule_data:
                ev_schedule = schedule_data["ev_schedule"]
                
            heating_schedule = None
            if system_data.heating_config and "heating_schedule" in schedule_data:
                heating_schedule = schedule_data["heating_schedule"]
            
            # Extract cost analysis
            cost_info = result.cost_analysis
            expected_cost = float(cost_info.split("€")[0].replace(",", ".")) if "€" in cost_info else 0.0
            expected_savings = 0.0  # Calculate from baseline
            
            return OptimizationResult(
                battery_schedule=battery_schedule,
                ev_charging_schedule=ev_schedule,
                heating_schedule=heating_schedule,
                expected_cost=expected_cost,
                expected_savings=expected_savings,
                confidence_score=0.8,  # DSPy provides consistent quality
                optimization_strategy=result.strategy_explanation,
                warnings=[]
            )
            
        except Exception as e:
            logging.error(f"Failed to parse optimization result: {e}")
            return OptimizationResult(
                battery_schedule=[0] * system_data.time_periods,
                warnings=[f"Parsing error: {str(e)}"]
            )
    
    def _parse_battery_result(self, result) -> Dict[str, Any]:
        """Parse battery-specific optimization result"""
        try:
            charging_schedule = json.loads(result.charging_schedule)
            soc_schedule = json.loads(result.soc_schedule)
            
            return {
                "charging_schedule": charging_schedule,
                "soc_schedule": soc_schedule,
                "cost_savings": result.cost_savings,
                "optimization_type": "battery_only"
            }
            
        except Exception as e:
            logging.error(f"Failed to parse battery result: {e}")
            return {"error": str(e)}
    
    def validate_configuration(self) -> bool:
        """Validate DSPy AI configuration"""
        if not self.lm:
            logging.error("Language model not initialized")
            return False
            
        try:
            # Test with a simple query
            test_signature = dspy.Signature("question -> answer")
            test_module = dspy.ChainOfThought(test_signature)
            
            result = test_module(question="What is 2+2?")
            return "4" in result.answer
            
        except Exception as e:
            logging.error(f"Configuration validation failed: {e}")
            return False


# Convenience function for backward compatibility
async def create_dspy_optimizer(da_calc_instance) -> Optional[DSPyAIOptimizer]:
    """Create and validate DSPy AI optimizer instance"""
    optimizer = DSPyAIOptimizer(da_calc_instance)
    
    if optimizer.validate_configuration():
        return optimizer
    else:
        logging.error("Failed to create valid DSPy optimizer")
        return None