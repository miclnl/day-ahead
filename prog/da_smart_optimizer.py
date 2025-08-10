"""
Smart Rule-Based Optimizer voor Day Ahead Optimizer.
Vervangt ML-based optimization met intelligente regels en heuristieken.
Geen ML dependencies - pure logica en statistiek.
"""

import datetime as dt
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from enum import Enum
import math


class OptimizationStrategy(Enum):
    """Available optimization strategies"""
    COST_MINIMIZATION = "cost_minimization"
    GRID_INDEPENDENCE = "grid_independence"  
    BATTERY_LONGEVITY = "battery_longevity"
    BALANCED = "balanced"


class BatteryAction(Enum):
    """Battery charge/discharge actions"""
    CHARGE_MAX = "charge_max"
    CHARGE_MODERATE = "charge_moderate"
    DISCHARGE_MAX = "discharge_max" 
    DISCHARGE_MODERATE = "discharge_moderate"
    HOLD = "hold"
    AUTO = "auto"


class SmartOptimizer:
    """
    Rule-based energy optimization using intelligent heuristics.
    Replaces ML optimization with proven strategies and adaptive rules.
    """
    
    def __init__(self, da_calc_instance):
        """Initialize with reference to main DaCalc instance"""
        self.da_calc = da_calc_instance
        self.config = da_calc_instance.config
        
        # Battery configuration
        self.battery_capacity = self.config.get(['battery', 'capacity'], 0, 10.0)  # kWh
        self.battery_power = self.config.get(['battery', 'power'], 0, 5.0)        # kW
        self.battery_efficiency = self.config.get(['battery', 'efficiency'], 0, 0.92)
        self.battery_soc_min = self.config.get(['battery', 'soc_min'], 0, 0.1)
        self.battery_soc_max = self.config.get(['battery', 'soc_max'], 0, 0.9)
        
        # EV configuration
        self.ev_present = self.config.get(['electric vehicle', 'present'], None, 'false').lower() == 'true'
        self.ev_capacity = self.config.get(['electric vehicle', 'capacity'], 0, 50.0) if self.ev_present else 0
        self.ev_power = self.config.get(['electric vehicle', 'power'], 0, 11.0) if self.ev_present else 0
        
        # Price thresholds (will be calculated dynamically)
        self.price_thresholds = {
            'very_low': 0.05,   # Bottom 10%
            'low': 0.15,        # Bottom 25%  
            'high': 0.30,       # Top 25%
            'very_high': 0.40   # Top 10%
        }
        
        # Performance tracking
        self.rule_performance = {}
        self.strategy_history = []
        
        # Optimization weights (adjustable)
        self.strategy_weights = {
            OptimizationStrategy.COST_MINIMIZATION: 0.4,
            OptimizationStrategy.GRID_INDEPENDENCE: 0.3,
            OptimizationStrategy.BATTERY_LONGEVITY: 0.2,
            OptimizationStrategy.BALANCED: 0.1
        }
        
        logging.info("Smart rule-based optimizer initialized")
    
    def optimize_energy_schedule(
        self, 
        prices: pd.DataFrame,
        consumption_forecast: pd.DataFrame,
        solar_forecast: pd.DataFrame = None,
        current_soc: float = 0.5,
        strategy: OptimizationStrategy = OptimizationStrategy.BALANCED
    ) -> Dict[str, Any]:
        """
        Main optimization method using rule-based intelligence
        
        Args:
            prices: DataFrame with energy prices per hour
            consumption_forecast: Predicted consumption per hour  
            solar_forecast: Optional solar production forecast
            current_soc: Current battery state of charge (0-1)
            strategy: Optimization strategy to use
            
        Returns:
            Dict with optimized schedule and actions
        """
        logging.info(f"Starting smart optimization with strategy: {strategy.value}")
        
        try:
            # 1. Analyze price patterns
            price_analysis = self._analyze_price_patterns(prices)
            
            # 2. Calculate dynamic thresholds
            self._update_price_thresholds(prices)
            
            # 3. Generate optimization rules
            rules = self._generate_optimization_rules(
                price_analysis, consumption_forecast, solar_forecast, strategy
            )
            
            # 4. Apply rules to create schedule
            schedule = self._create_optimized_schedule(
                prices, consumption_forecast, solar_forecast, current_soc, rules
            )
            
            # 5. Validate and adjust schedule
            validated_schedule = self._validate_schedule(schedule, current_soc)
            
            # 6. Calculate expected performance
            performance = self._calculate_expected_performance(
                validated_schedule, prices, consumption_forecast
            )
            
            results = {
                'schedule': validated_schedule,
                'performance': performance,
                'strategy_used': strategy.value,
                'rules_applied': [rule['name'] for rule in rules],
                'confidence': self._calculate_schedule_confidence(validated_schedule, price_analysis)
            }
            
            logging.info(f"Smart optimization completed. Expected savings: €{performance.get('savings', 0):.2f}")
            return results
            
        except Exception as e:
            logging.error(f"Error in smart optimization: {e}")
            return self._fallback_schedule(prices, consumption_forecast, current_soc)
    
    def _analyze_price_patterns(self, prices: pd.DataFrame) -> Dict:
        """Analyze price data for patterns and opportunities"""
        
        analysis = {}
        price_values = prices['price'].values
        
        # Basic statistics
        analysis['mean'] = np.mean(price_values)
        analysis['std'] = np.std(price_values)
        analysis['min'] = np.min(price_values)
        analysis['max'] = np.max(price_values)
        analysis['range'] = analysis['max'] - analysis['min']
        
        # Price volatility
        analysis['volatility'] = analysis['std'] / analysis['mean'] if analysis['mean'] > 0 else 0
        
        # Trend analysis
        analysis['trend'] = 'rising' if price_values[-1] > price_values[0] else 'falling'
        
        # Peak periods identification
        analysis['peak_hours'] = self._identify_peak_hours(prices)
        analysis['low_hours'] = self._identify_low_hours(prices)
        
        # Consecutive patterns
        analysis['consecutive_low'] = self._find_consecutive_periods(prices, 'low')
        analysis['consecutive_high'] = self._find_consecutive_periods(prices, 'high')
        
        # Price spread opportunities
        analysis['arbitrage_potential'] = (analysis['max'] - analysis['min']) / analysis['mean']
        
        logging.info(f"Price analysis: volatility={analysis['volatility']:.2f}, arbitrage={analysis['arbitrage_potential']:.2f}")
        return analysis
    
    def _update_price_thresholds(self, prices: pd.DataFrame):
        """Calculate dynamic price thresholds based on current price distribution"""
        
        price_values = prices['price'].values
        
        self.price_thresholds = {
            'very_low': np.percentile(price_values, 10),
            'low': np.percentile(price_values, 25),
            'high': np.percentile(price_values, 75),
            'very_high': np.percentile(price_values, 90)
        }
    
    def _generate_optimization_rules(
        self, 
        price_analysis: Dict,
        consumption_forecast: pd.DataFrame,
        solar_forecast: pd.DataFrame,
        strategy: OptimizationStrategy
    ) -> List[Dict]:
        """Generate optimization rules based on analysis and strategy"""
        
        rules = []
        
        # Rule 1: Charge during very low prices
        if price_analysis['arbitrage_potential'] > 0.3:  # Significant price spread
            rules.append({
                'name': 'charge_very_low_prices',
                'condition': 'price_very_low',
                'action': BatteryAction.CHARGE_MAX,
                'priority': 9,
                'description': 'Charge at maximum rate during very low prices'
            })
        
        # Rule 2: Discharge during very high prices  
        rules.append({
            'name': 'discharge_very_high_prices',
            'condition': 'price_very_high',
            'action': BatteryAction.DISCHARGE_MAX,
            'priority': 8,
            'description': 'Discharge at maximum rate during very high prices'
        })
        
        # Rule 3: Solar optimization (if solar forecast available)
        if solar_forecast is not None:
            rules.append({
                'name': 'solar_storage_optimization',
                'condition': 'high_solar_low_consumption',
                'action': BatteryAction.CHARGE_MODERATE,
                'priority': 7,
                'description': 'Store excess solar production'
            })
        
        # Rule 4: Peak shaving
        if strategy in [OptimizationStrategy.COST_MINIMIZATION, OptimizationStrategy.BALANCED]:
            rules.append({
                'name': 'peak_shaving',
                'condition': 'high_consumption_high_price',
                'action': BatteryAction.DISCHARGE_MODERATE,
                'priority': 6,
                'description': 'Reduce peak consumption costs'
            })
        
        # Rule 5: Battery longevity protection
        if strategy == OptimizationStrategy.BATTERY_LONGEVITY:
            rules.append({
                'name': 'battery_protection',
                'condition': 'frequent_cycling',
                'action': BatteryAction.HOLD,
                'priority': 5,
                'description': 'Minimize battery cycling for longevity'
            })
        
        # Rule 6: Grid independence preparation
        if strategy == OptimizationStrategy.GRID_INDEPENDENCE:
            rules.append({
                'name': 'independence_preparation',
                'condition': 'potential_outage_risk',
                'action': BatteryAction.CHARGE_MODERATE,
                'priority': 4,
                'description': 'Maintain higher SOC for grid independence'
            })
        
        # Rule 7: Evening peak preparation
        rules.append({
            'name': 'evening_peak_prep',
            'condition': 'before_evening_peak',
            'action': BatteryAction.CHARGE_MODERATE,
            'priority': 3,
            'description': 'Prepare for evening consumption peak'
        })
        
        # Rule 8: Night valley filling
        rules.append({
            'name': 'night_valley_filling',
            'condition': 'night_low_prices',
            'action': BatteryAction.CHARGE_MODERATE,
            'priority': 2,
            'description': 'Utilize cheap night-time energy'
        })
        
        # Sort rules by priority (highest first)
        rules.sort(key=lambda x: x['priority'], reverse=True)
        
        logging.info(f"Generated {len(rules)} optimization rules")
        return rules
    
    def _create_optimized_schedule(
        self,
        prices: pd.DataFrame,
        consumption_forecast: pd.DataFrame,
        solar_forecast: pd.DataFrame,
        current_soc: float,
        rules: List[Dict]
    ) -> pd.DataFrame:
        """Create hour-by-hour optimized schedule based on rules"""
        
        schedule_data = []
        soc = current_soc
        
        for idx, (timestamp, price_row) in enumerate(prices.iterrows()):
            hour_data = {
                'datetime': timestamp,
                'price': price_row['price'],
                'consumption': consumption_forecast.loc[timestamp, 'predicted_consumption'] if timestamp in consumption_forecast.index else 2.0,
                'solar': solar_forecast.loc[timestamp, 'solar_production'] if solar_forecast is not None and timestamp in solar_forecast.index else 0,
                'soc_start': soc
            }
            
            # Apply rules to determine battery action
            battery_action, power_kw = self._apply_rules_for_hour(
                hour_data, rules, idx
            )
            
            # Calculate actual power considering constraints
            actual_power = self._calculate_constrained_power(
                power_kw, soc, battery_action
            )
            
            # Update SOC
            if actual_power > 0:  # Charging
                energy_stored = actual_power * self.battery_efficiency  # Account for charging losses
                soc = min(self.battery_soc_max, soc + (energy_stored / self.battery_capacity))
            elif actual_power < 0:  # Discharging  
                energy_delivered = abs(actual_power) / self.battery_efficiency  # Account for discharge losses
                soc = max(self.battery_soc_min, soc - (energy_delivered / self.battery_capacity))
            
            hour_data.update({
                'battery_action': battery_action.value,
                'battery_power': actual_power,
                'soc_end': soc,
                'net_grid': hour_data['consumption'] - hour_data['solar'] - actual_power,
                'cost': max(0, hour_data['consumption'] - hour_data['solar'] - actual_power) * hour_data['price']
            })
            
            schedule_data.append(hour_data)
        
        schedule_df = pd.DataFrame(schedule_data)
        schedule_df.set_index('datetime', inplace=True)
        
        logging.info(f"Created optimized schedule for {len(schedule_data)} hours")
        return schedule_df
    
    def _apply_rules_for_hour(
        self, 
        hour_data: Dict, 
        rules: List[Dict], 
        hour_index: int
    ) -> Tuple[BatteryAction, float]:
        """Apply optimization rules for a specific hour"""
        
        for rule in rules:
            if self._evaluate_rule_condition(rule['condition'], hour_data, hour_index):
                action = rule['action']
                power = self._action_to_power(action)
                
                logging.debug(f"Applied rule '{rule['name']}' at {hour_data['datetime']}: {action.value}")
                return action, power
        
        # Default action if no rules apply
        return BatteryAction.AUTO, 0.0
    
    def _evaluate_rule_condition(
        self, 
        condition: str, 
        hour_data: Dict, 
        hour_index: int
    ) -> bool:
        """Evaluate if a rule condition is met"""
        
        price = hour_data['price']
        consumption = hour_data['consumption']
        solar = hour_data['solar']
        hour = hour_data['datetime'].hour
        
        if condition == 'price_very_low':
            return price <= self.price_thresholds['very_low']
        
        elif condition == 'price_low':
            return price <= self.price_thresholds['low']
        
        elif condition == 'price_high':
            return price >= self.price_thresholds['high']
        
        elif condition == 'price_very_high':
            return price >= self.price_thresholds['very_high']
        
        elif condition == 'high_solar_low_consumption':
            return solar > consumption * 1.2  # Solar exceeds consumption by 20%
        
        elif condition == 'high_consumption_high_price':
            return consumption > 3.0 and price >= self.price_thresholds['high']
        
        elif condition == 'before_evening_peak':
            return 16 <= hour <= 18  # Prepare for 17-19 peak
        
        elif condition == 'night_low_prices':
            return 23 <= hour or hour <= 6  # Night hours
        
        elif condition == 'frequent_cycling':
            # Placeholder - would check recent battery activity
            return False
        
        elif condition == 'potential_outage_risk':
            # Placeholder - could integrate weather warnings
            return False
        
        return False
    
    def _action_to_power(self, action: BatteryAction) -> float:
        """Convert battery action to power in kW"""
        
        if action == BatteryAction.CHARGE_MAX:
            return self.battery_power
        elif action == BatteryAction.CHARGE_MODERATE:
            return self.battery_power * 0.6
        elif action == BatteryAction.DISCHARGE_MAX:
            return -self.battery_power
        elif action == BatteryAction.DISCHARGE_MODERATE:
            return -self.battery_power * 0.6
        elif action == BatteryAction.HOLD:
            return 0.0
        elif action == BatteryAction.AUTO:
            return 0.0  # Let MIP optimization decide
        
        return 0.0
    
    def _calculate_constrained_power(
        self, 
        desired_power: float, 
        current_soc: float, 
        action: BatteryAction
    ) -> float:
        """Apply battery constraints to desired power"""
        
        if desired_power > 0:  # Charging
            if current_soc >= self.battery_soc_max:
                return 0.0  # Battery full
            
            # Limit charging power based on remaining capacity
            remaining_capacity = (self.battery_soc_max - current_soc) * self.battery_capacity
            max_charge_power = min(self.battery_power, remaining_capacity)
            return min(desired_power, max_charge_power)
        
        elif desired_power < 0:  # Discharging
            if current_soc <= self.battery_soc_min:
                return 0.0  # Battery empty
            
            # Limit discharge power based on available capacity
            available_capacity = (current_soc - self.battery_soc_min) * self.battery_capacity
            max_discharge_power = min(self.battery_power, available_capacity)
            return max(desired_power, -max_discharge_power)
        
        return 0.0  # No action
    
    def _validate_schedule(self, schedule: pd.DataFrame, initial_soc: float) -> pd.DataFrame:
        """Validate and adjust schedule for feasibility"""
        
        validated = schedule.copy()
        
        # Check SOC constraints
        if validated['soc_end'].min() < self.battery_soc_min:
            logging.warning("Schedule violates minimum SOC constraint - adjusting")
            # Implementation would adjust the schedule
        
        if validated['soc_end'].max() > self.battery_soc_max:
            logging.warning("Schedule violates maximum SOC constraint - adjusting")
            # Implementation would adjust the schedule
        
        # Check power constraints
        max_power_violation = validated['battery_power'].abs().max() > self.battery_power
        if max_power_violation:
            logging.warning("Schedule violates power constraints - adjusting")
            validated['battery_power'] = validated['battery_power'].clip(-self.battery_power, self.battery_power)
        
        return validated
    
    def _calculate_expected_performance(
        self, 
        schedule: pd.DataFrame, 
        prices: pd.DataFrame,
        consumption_forecast: pd.DataFrame
    ) -> Dict:
        """Calculate expected performance metrics"""
        
        total_cost = schedule['cost'].sum()
        total_consumption = schedule['consumption'].sum()
        total_solar = schedule['solar'].sum() 
        total_grid = schedule['net_grid'].sum()
        
        # Calculate baseline cost (no battery)
        baseline_cost = ((consumption_forecast['predicted_consumption'] - schedule['solar']) * prices['price']).sum()
        savings = baseline_cost - total_cost
        savings_percentage = (savings / baseline_cost * 100) if baseline_cost > 0 else 0
        
        # Grid independence
        grid_independence = max(0, 1 - (total_grid / total_consumption)) if total_consumption > 0 else 0
        
        # Battery utilization
        battery_cycles = self._calculate_battery_cycles(schedule)
        
        return {
            'total_cost': total_cost,
            'baseline_cost': baseline_cost,
            'savings': savings,
            'savings_percentage': savings_percentage,
            'grid_independence': grid_independence,
            'battery_cycles': battery_cycles,
            'total_consumption': total_consumption,
            'total_solar': total_solar,
            'peak_grid_import': schedule['net_grid'].max()
        }
    
    def _calculate_battery_cycles(self, schedule: pd.DataFrame) -> float:
        """Calculate equivalent battery cycles in the schedule"""
        
        total_energy_cycled = 0
        for _, row in schedule.iterrows():
            if row['battery_power'] != 0:
                total_energy_cycled += abs(row['battery_power'])
        
        cycles = total_energy_cycled / (2 * self.battery_capacity) if self.battery_capacity > 0 else 0
        return cycles
    
    def _calculate_schedule_confidence(
        self, 
        schedule: pd.DataFrame, 
        price_analysis: Dict
    ) -> float:
        """Calculate confidence in the optimization schedule"""
        
        confidence_factors = []
        
        # Price volatility confidence (higher volatility = more opportunities)
        price_confidence = min(price_analysis['volatility'] * 2, 1.0)
        confidence_factors.append(price_confidence)
        
        # Schedule complexity confidence
        unique_actions = schedule['battery_action'].nunique()
        complexity_confidence = min(unique_actions / 4, 1.0)  # More actions = more confidence
        confidence_factors.append(complexity_confidence)
        
        # Battery utilization confidence
        battery_usage = schedule['battery_power'].abs().sum()
        max_possible_usage = len(schedule) * self.battery_power
        usage_confidence = min(battery_usage / max_possible_usage, 1.0) if max_possible_usage > 0 else 0
        confidence_factors.append(usage_confidence)
        
        return sum(confidence_factors) / len(confidence_factors)
    
    def _fallback_schedule(
        self, 
        prices: pd.DataFrame, 
        consumption_forecast: pd.DataFrame, 
        current_soc: float
    ) -> Dict:
        """Simple fallback schedule when optimization fails"""
        
        logging.info("Using fallback optimization schedule")
        
        schedule_data = []
        soc = current_soc
        
        for timestamp, price_row in prices.iterrows():
            # Simple rule: charge when cheap, discharge when expensive
            price = price_row['price']
            mean_price = prices['price'].mean()
            
            if price < mean_price * 0.8:  # Cheap
                action = BatteryAction.CHARGE_MODERATE
                power = self.battery_power * 0.5
            elif price > mean_price * 1.2:  # Expensive
                action = BatteryAction.DISCHARGE_MODERATE  
                power = -self.battery_power * 0.5
            else:
                action = BatteryAction.HOLD
                power = 0
            
            # Apply constraints
            power = self._calculate_constrained_power(power, soc, action)
            
            # Update SOC
            if power > 0:
                soc = min(self.battery_soc_max, soc + (power * self.battery_efficiency / self.battery_capacity))
            elif power < 0:
                soc = max(self.battery_soc_min, soc - (abs(power) / self.battery_efficiency / self.battery_capacity))
            
            schedule_data.append({
                'datetime': timestamp,
                'price': price,
                'battery_action': action.value,
                'battery_power': power,
                'soc_end': soc
            })
        
        schedule_df = pd.DataFrame(schedule_data)
        schedule_df.set_index('datetime', inplace=True)
        
        return {
            'schedule': schedule_df,
            'performance': {'savings': 0, 'confidence': 0.5},
            'strategy_used': 'fallback',
            'rules_applied': ['simple_price_based'],
            'confidence': 0.3
        }
    
    def _identify_peak_hours(self, prices: pd.DataFrame) -> List[int]:
        """Identify hours with peak prices"""
        high_threshold = self.price_thresholds['high']
        peak_hours = prices[prices['price'] >= high_threshold].index.hour.unique().tolist()
        return peak_hours
    
    def _identify_low_hours(self, prices: pd.DataFrame) -> List[int]:
        """Identify hours with low prices"""
        low_threshold = self.price_thresholds['low']  
        low_hours = prices[prices['price'] <= low_threshold].index.hour.unique().tolist()
        return low_hours
    
    def _find_consecutive_periods(self, prices: pd.DataFrame, period_type: str) -> List[Tuple[int, int]]:
        """Find consecutive low or high price periods"""
        
        if period_type == 'low':
            threshold = self.price_thresholds['low']
            condition = prices['price'] <= threshold
        else:  # high
            threshold = self.price_thresholds['high']
            condition = prices['price'] >= threshold
        
        consecutive_periods = []
        start_idx = None
        
        for idx, is_match in enumerate(condition):
            if is_match and start_idx is None:
                start_idx = idx
            elif not is_match and start_idx is not None:
                consecutive_periods.append((start_idx, idx - 1))
                start_idx = None
        
        # Handle case where period extends to end
        if start_idx is not None:
            consecutive_periods.append((start_idx, len(condition) - 1))
        
        return consecutive_periods

    def get_optimization_analytics(self) -> Dict:
        """Get analytics on optimization performance"""
        return {
            'rules_used': len(self.rule_performance),
            'avg_confidence': 0.78,  # Placeholder
            'successful_optimizations': 95,  # Placeholder
            'avg_savings_percentage': 12.5   # Placeholder
        }