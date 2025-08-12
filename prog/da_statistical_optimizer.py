"""
Statistical Optimization Engine for Day Ahead Optimizer.
Replaces ML-based optimization with statistical prediction and rule-based optimization.
Uses da_statistical_predictor, da_smart_optimizer, da_enhanced_weather, and da_performance_monitor.
"""

import datetime as dt
import logging
import math
import pandas as pd
# Optional import - MIP optimization (fallback if not available)
try:
    from mip import Model, xsum, minimize, BINARY, CONTINUOUS
    MIP_AVAILABLE = True
except ImportError:
    MIP_AVAILABLE = False
    logging.warning("MIP not available - using pure rule-based optimization")
from typing import List, Dict, Tuple, Optional, Any

# Import new statistical modules (replace ML imports)
from da_statistical_predictor import StatisticalPredictor
from da_smart_optimizer import SmartOptimizer, OptimizationStrategy
from da_enhanced_weather import EnhancedWeatherService
from da_performance_monitor import PerformanceMonitor
from da_weather_reactive import WeatherReactiveOptimizer
from da_multiday_optimizer import MultiDayOptimizer
from ha_client import turn_switch


class StatisticalEnergyOptimizer:
    """
    Statistical energy optimization engine using intelligent heuristics.
    Replaces the ML-based EnergyOptimizer with proven statistical methods.
    """

    def __init__(self, da_calc_instance):
        """Initialize with reference to main DaCalc instance for access to config and data"""
        self.da_calc = da_calc_instance
        self.config = da_calc_instance.config
        self.db_da = da_calc_instance.db_da

        # Initialize statistical components (replace AI/ML managers)
        self.predictor = StatisticalPredictor(da_calc_instance)
        self.smart_optimizer = SmartOptimizer(da_calc_instance)
        self.weather_integrator = EnhancedWeatherService(da_calc_instance)
        self.performance_monitor = PerformanceMonitor(da_calc_instance)
        self.weather_reactive = WeatherReactiveOptimizer(da_calc_instance)
        self.multiday_optimizer = MultiDayOptimizer(da_calc_instance)

        # MIP optimization model
        self.model = None

        # Optimization configuration
        self.optimization_strategy = self._get_optimization_strategy()
        self.prediction_hours = self.config.get(['optimization', 'hours_ahead'], 0, 48)
        # Force pure rule-based if MIP not available
        self.hybrid_mode = (self.config.get(['optimization', 'hybrid_mode'], None, 'true').lower() == 'true'
                           and MIP_AVAILABLE)

        logging.info("Statistical energy optimizer initialized")

    def optimize_energy_schedule(
        self,
        start_dt: dt.datetime = None,
        start_soc: float = None
    ) -> Dict[str, Any]:
        """
        Main optimization entry point using statistical intelligence

        Args:
            start_dt: Start datetime for optimization
            start_soc: Starting state of charge

        Returns:
            Dict containing optimization results
        """
        optimization_start = dt.datetime.now()
        logging.info(f"Starting statistical energy optimization")

        try:
            # Step 1: Data collection and preparation
            optimization_data = self._collect_and_prepare_data(start_dt, start_soc)
            if optimization_data is None:
                return None

            # Step 2: Statistical prediction
            predictions = self._generate_statistical_predictions(optimization_data)

            # Step 3: Enhanced weather analysis
            weather_analysis = self._get_weather_analysis(optimization_data)

            # Step 3.5: Weather reactive monitoring and adjustments
            weather_events = self._monitor_weather_reactive_conditions(weather_analysis)

            # Step 4: Rule-based optimization
            if self.hybrid_mode:
                # Hybrid approach: Smart rules + MIP refinement
                results = self._hybrid_optimization(
                    optimization_data, predictions, weather_analysis
                )
            else:
                # Pure rule-based optimization
                results = self._rule_based_optimization(
                    optimization_data, predictions, weather_analysis
                )

            # Step 5: Apply weather reactive adjustments
            weather_adjusted_results = self._apply_weather_reactive_adjustments(
                results, weather_events, weather_analysis
            )

            # Step 6: Validate and refine results
            validated_results = self._validate_and_refine(weather_adjusted_results, optimization_data)

            # Step 6: Performance logging
            self._log_optimization_performance(
                optimization_start, validated_results, optimization_data
            )

            # Step 7: Generate reports
            self._generate_reports(validated_results)

            optimization_duration = dt.datetime.now() - optimization_start
            logging.info(f"Statistical optimization completed in {optimization_duration.total_seconds():.2f}s")

            return validated_results

        except Exception as e:
            logging.error(f"Error in statistical optimization: {e}")
            return self._fallback_optimization(start_dt, start_soc)

    def _collect_and_prepare_data(
        self,
        start_dt: dt.datetime = None,
        start_soc: float = None
    ) -> Dict[str, Any]:
        """Collect and prepare all necessary data for optimization"""

        try:
            # Set defaults
            if start_dt is None:
                start_dt = dt.datetime.now().replace(minute=0, second=0, microsecond=0)

            if start_soc is None:
                start_soc = self._get_current_battery_soc()

            # Get energy prices
            prices = self._get_energy_prices(start_dt, self.prediction_hours)
            if prices is None or len(prices) == 0:
                logging.error("No energy prices available for optimization")
                return None

            # Get current system status
            system_status = self._get_system_status()

            return {
                'start_time': start_dt,
                'start_soc': start_soc,
                'prediction_hours': self.prediction_hours,
                'prices': prices,
                'system_status': system_status,
                'battery_config': self._get_battery_config(),
                'solar_config': self._get_solar_config(),
                'ev_config': self._get_ev_config() if self._is_ev_present() else None
            }

        except Exception as e:
            logging.error(f"Error collecting optimization data: {e}")
            return None

    def _generate_statistical_predictions(self, optimization_data: Dict) -> Dict[str, pd.DataFrame]:
        """Generate statistical predictions for consumption and production"""

        predictions = {}

        try:
            start_time = optimization_data['start_time']
            hours_ahead = optimization_data['prediction_hours']

            # Consumption prediction
            logging.info("Generating consumption prediction...")
            consumption_forecast = self.predictor.predict_consumption(
                start_time, hours_ahead
            )
            predictions['consumption'] = consumption_forecast

            # Solar production prediction (if solar system present)
            if optimization_data['solar_config']['capacity'] > 0:
                logging.info("Generating solar production forecast...")
                solar_forecast = self.weather_integrator.get_enhanced_solar_forecast(
                    start_time, hours_ahead
                )
                predictions['solar'] = solar_forecast

            # Weather correlations
            weather_correlations = self.weather_integrator.get_weather_correlations(
                start_time, hours_ahead
            )
            predictions['weather_impact'] = weather_correlations

            logging.info("Statistical predictions generated successfully")
            return predictions

        except Exception as e:
            logging.error(f"Error generating statistical predictions: {e}")
            return {}

    def _get_weather_analysis(self, optimization_data: Dict) -> Dict:
        """Get enhanced weather analysis for optimization"""

        try:
            start_time = optimization_data['start_time']
            hours_ahead = optimization_data['prediction_hours']

            weather_analysis = self.weather_integrator.get_weather_correlations(
                start_time, hours_ahead
            )

            return weather_analysis

        except Exception as e:
            logging.error(f"Error getting weather analysis: {e}")
            return {}

    def _hybrid_optimization(
        self,
        optimization_data: Dict,
        predictions: Dict,
        weather_analysis: Dict
    ) -> Dict[str, Any]:
        """Hybrid optimization: Smart rules + MIP refinement"""

        try:
            # Phase 1: Rule-based initial solution
            logging.info("Phase 1: Generating rule-based solution...")
            rule_solution = self._rule_based_optimization(
                optimization_data, predictions, weather_analysis
            )

            # Phase 2: MIP refinement of rule solution
            logging.info("Phase 2: Refining with MIP optimization...")
            refined_solution = self._refine_with_mip(rule_solution, optimization_data)

            # Combine best aspects of both solutions
            final_solution = self._combine_solutions(rule_solution, refined_solution)

            final_solution['optimization_method'] = 'hybrid_statistical_mip'
            return final_solution

        except Exception as e:
            logging.error(f"Error in hybrid optimization: {e}")
            # Fallback to pure rule-based
            return self._rule_based_optimization(optimization_data, predictions, weather_analysis)

    def _rule_based_optimization(
        self,
        optimization_data: Dict,
        predictions: Dict,
        weather_analysis: Dict
    ) -> Dict[str, Any]:
        """Pure rule-based optimization using smart heuristics"""

        try:
            prices = optimization_data['prices']
            consumption_forecast = predictions.get('consumption')
            solar_forecast = predictions.get('solar')
            current_soc = optimization_data['start_soc']

            # Use smart optimizer
            optimization_results = self.smart_optimizer.optimize_energy_schedule(
                prices=prices,
                consumption_forecast=consumption_forecast,
                solar_forecast=solar_forecast,
                current_soc=current_soc,
                strategy=self.optimization_strategy
            )

            # Add metadata
            optimization_results['optimization_method'] = 'rule_based_statistical'
            optimization_results['predictions_used'] = list(predictions.keys())
            optimization_results['weather_analysis'] = weather_analysis

            # Apply dynamic solar control decision if enabled
            try:
                solar_ctrl = self._get_dynamic_solar_control_config()
                if solar_ctrl.get('enabled') and solar_ctrl.get('switch_entity'):
                    self._apply_dynamic_solar_control(
                        solar_ctrl=solar_ctrl,
                        prices=prices,
                        schedule=optimization_results.get('schedule'),
                        consumption_forecast=consumption_forecast
                    )
                    optimization_results['solar_dynamic_control_applied'] = True
            except Exception as e:
                logging.warning(f"Dynamic solar control skipped: {e}")

            return optimization_results

        except Exception as e:
            logging.error(f"Error in rule-based optimization: {e}")
            return self._simple_fallback_optimization(optimization_data)

    def _refine_with_mip(
        self,
        rule_solution: Dict,
        optimization_data: Dict
    ) -> Dict[str, Any]:
        """Refine rule-based solution using MIP optimization"""

        try:
            # This would implement MIP refinement
            # For now, return the rule solution (MIP integration complex)
            logging.info("MIP refinement not yet implemented - using rule solution")
            return rule_solution

        except Exception as e:
            logging.error(f"Error in MIP refinement: {e}")
            return rule_solution

    def _combine_solutions(self, solution1: Dict, solution2: Dict) -> Dict:
        """Combine best aspects of multiple solutions"""

        # For now, prefer the first solution
        # In full implementation, would analyze and combine optimally
        return solution1

    def _get_dynamic_solar_control_config(self) -> Dict[str, Any]:
        mode = self.da_calc.config.get(['solar', 'dynamic_control', 'mode'], None, 'all_in_zero').lower()
        enabled = self.da_calc.config.get(['solar', 'dynamic_control', 'enabled'], None, False)
        switch_entity = self.da_calc.config.get(['solar', 'dynamic_control', 'switch_entity'], None, '')
        # Optional vaste toeslagen (€/kWh) voor all-in prijs
        taxes = float(self.da_calc.config.get(['prices', 'taxes_per_kwh'], None, 0.0) or 0.0)
        grid_fee = float(self.da_calc.config.get(['prices', 'grid_fee_per_kwh'], None, 0.0) or 0.0)
        supplier_fee = float(self.da_calc.config.get(['prices', 'supplier_fee_per_kwh'], None, 0.0) or 0.0)
        hysteresis_min = int(self.da_calc.config.get(['solar', 'dynamic_control', 'hysteresis_minutes'], None, 10) or 10)
        return {
            'enabled': bool(enabled),
            'mode': mode,
            'switch_entity': switch_entity,
            'taxes': taxes,
            'grid_fee': grid_fee,
            'supplier_fee': supplier_fee,
            'hysteresis_min': hysteresis_min,
        }

    def _apply_dynamic_solar_control(
        self,
        solar_ctrl: Dict[str, Any],
        prices: pd.DataFrame,
        schedule: Optional[pd.DataFrame],
        consumption_forecast: pd.DataFrame,
    ) -> None:
        """Decide PV on/off based on all-in price or net-zero mode and call HA switch accordingly.
        We gebruiken uur-resolutie en een eenvoudige hysterese om frequent schakelen te voorkomen.
        """
        if prices is None or prices.empty:
            return
        try:
            now = dt.datetime.now().replace(minute=0, second=0, microsecond=0)
            if now not in prices.index:
                # Align to nearest hour present in prices
                nearest = prices.index.get_indexer([now], method='nearest')[0]
                ts = prices.index[nearest]
            else:
                ts = now

            market_price = float(prices.loc[ts, 'price'])  # €/kWh
            all_in = market_price + float(solar_ctrl['taxes']) + float(solar_ctrl['grid_fee']) + float(solar_ctrl['supplier_fee'])

            # Net consumption at this hour (approx)
            cons = float(consumption_forecast.loc[ts, 'predicted_consumption']) if ts in consumption_forecast.index else 0.0
            solar = 0.0
            if schedule is not None and 'solar' in schedule.columns and ts in schedule.index:
                solar = float(schedule.loc[ts, 'solar'])

            mode = solar_ctrl['mode']
            turn_on = True
            if mode == 'all_in_zero':
                # Simple rule: PV off if all-in ≤ 0
                turn_on = not (all_in <= 0)
            elif mode == 'net_zero':
                # Strive for ~net zero export/import at the meter
                # If solar significantly exceeds current consumption while price incentives are poor (>0 all-in), turn off to avoid paid export
                if solar > cons and all_in > 0:
                    turn_on = False
                else:
                    turn_on = True

            # Simple hysteresis: only toggle if last action older than hysteresis_min
            last_decision_attr = getattr(self, '_pv_last_decision', None)
            last_time_attr = getattr(self, '_pv_last_time', None)
            too_soon = False
            if last_time_attr is not None:
                delta = (dt.datetime.now() - last_time_attr).total_seconds() / 60.0
                too_soon = delta < float(solar_ctrl['hysteresis_min'])

            if last_decision_attr is not None and too_soon and last_decision_attr == turn_on:
                # Skip toggle; keep current state
                return

            # Issue HA switch
            entity = solar_ctrl['switch_entity']
            ok = turn_switch(self.da_calc.config, entity, on=turn_on)
            if ok:
                logging.info(f"PV dynamic control: set {entity} -> {'ON' if turn_on else 'OFF'} (all_in={all_in:.4f} €/kWh, mode={mode})")
                self._pv_last_decision = turn_on
                self._pv_last_time = dt.datetime.now()
            else:
                logging.warning(f"PV dynamic control: failed to toggle {entity}")
        except Exception as e:
            logging.warning(f"PV dynamic control error: {e}")

    def _validate_and_refine(self, results: Dict, optimization_data: Dict) -> Dict:
        """Validate optimization results and apply refinements"""

        try:
            validated = results.copy()

            # Validate schedule feasibility
            schedule = validated.get('schedule')
            if schedule is not None:
                # Check battery constraints
                validated_schedule = self._validate_battery_constraints(
                    schedule, optimization_data['battery_config']
                )
                validated['schedule'] = validated_schedule

            # Calculate final performance metrics
            performance = self._calculate_comprehensive_performance(validated, optimization_data)
            validated['performance'] = performance

            return validated

        except Exception as e:
            logging.error(f"Error validating results: {e}")
            return results

    def _validate_battery_constraints(
        self,
        schedule: pd.DataFrame,
        battery_config: Dict
    ) -> pd.DataFrame:
        """Validate and fix battery constraint violations"""

        validated = schedule.copy()

        # Check SOC bounds
        min_soc = battery_config.get('soc_min', 0.1)
        max_soc = battery_config.get('soc_max', 0.9)

        validated['soc_end'] = validated['soc_end'].clip(min_soc, max_soc)

        # Check power bounds
        max_power = battery_config.get('power', 5.0)
        validated['battery_power'] = validated['battery_power'].clip(-max_power, max_power)

        return validated

    def _calculate_comprehensive_performance(
        self,
        results: Dict,
        optimization_data: Dict
    ) -> Dict:
        """Calculate comprehensive performance metrics"""

        performance = results.get('performance', {})

        # Add statistical performance metrics
        performance.update({
            'optimization_method': results.get('optimization_method', 'unknown'),
            'prediction_confidence': self._calculate_overall_prediction_confidence(results),
            'weather_impact_score': self._calculate_weather_impact_score(results),
            'rule_effectiveness_score': self._calculate_rule_effectiveness_score(results),
            'complexity_score': len(results.get('rules_applied', [])) / 10  # Normalize to 0-1
        })

        return performance

    def _log_optimization_performance(
        self,
        start_time: dt.datetime,
        results: Dict,
        optimization_data: Dict
    ):
        """Log optimization performance for monitoring"""

        try:
            strategy_used = results.get('optimization_method', 'unknown')
            rules_applied = results.get('rules_applied', [])
            performance = results.get('performance', {})
            predicted_savings = performance.get('savings', 0)

            self.performance_monitor.log_optimization_performance(
                timestamp=start_time,
                strategy_used=strategy_used,
                rules_applied=rules_applied,
                predicted_savings=predicted_savings,
                actual_savings=None,  # Will be updated later when actual data available
                battery_cycles=performance.get('battery_cycles', 0),
                grid_independence=performance.get('grid_independence', 0)
            )

        except Exception as e:
            logging.error(f"Error logging performance: {e}")

    def _fallback_optimization(
        self,
        start_dt: dt.datetime,
        start_soc: float
    ) -> Dict[str, Any]:
        """Complete fallback when all optimization methods fail"""

        logging.warning("Using complete fallback optimization")

        try:
            # Get minimal required data
            prices = self._get_energy_prices(start_dt, 24)
            if prices is None:
                return None

            # Simple cost-based strategy
            schedule_data = []
            soc = start_soc or 0.5

            mean_price = prices['price'].mean()

            for timestamp, price_row in prices.iterrows():
                price = price_row['price']

                # Simple rule: charge when cheap, discharge when expensive
                if price < mean_price * 0.8:
                    action = 'charge'
                    power = 2.0  # kW
                elif price > mean_price * 1.2:
                    action = 'discharge'
                    power = -2.0  # kW
                else:
                    action = 'hold'
                    power = 0.0

                # Update SOC (simplified)
                soc_change = power / 10.0 * 0.9  # Assume 10kWh battery, 90% efficiency
                soc = max(0.1, min(0.9, soc + soc_change))

                schedule_data.append({
                    'datetime': timestamp,
                    'price': price,
                    'battery_action': action,
                    'battery_power': power,
                    'soc_end': soc
                })

            schedule_df = pd.DataFrame(schedule_data)
            schedule_df.set_index('datetime', inplace=True)

            return {
                'schedule': schedule_df,
                'performance': {'savings': 0, 'confidence': 0.3},
                'optimization_method': 'simple_fallback',
                'rules_applied': ['simple_price_based'],
                'confidence': 0.3
            }

        except Exception as e:
            logging.error(f"Error in fallback optimization: {e}")
            return None

    def _simple_fallback_optimization(self, optimization_data: Dict) -> Dict:
        """Simple fallback when rule-based optimization fails"""

        return self._fallback_optimization(
            optimization_data['start_time'],
            optimization_data['start_soc']
        )

    def _get_optimization_strategy(self) -> OptimizationStrategy:
        """Get optimization strategy from configuration"""

        strategy_name = self.config.get(['optimization', 'strategy'], None, 'balanced').lower()

        strategy_mapping = {
            'cost_minimization': OptimizationStrategy.COST_MINIMIZATION,
            'grid_independence': OptimizationStrategy.GRID_INDEPENDENCE,
            'battery_longevity': OptimizationStrategy.BATTERY_LONGEVITY,
            'balanced': OptimizationStrategy.BALANCED
        }

        return strategy_mapping.get(strategy_name, OptimizationStrategy.BALANCED)

    def _get_current_battery_soc(self) -> float:
        """Get current battery state of charge"""
        try:
            # This would query Home Assistant for current SOC
            # For now, return a default
            return 0.5
        except:
            return 0.5

    def _get_energy_prices(self, start_time: dt.datetime, hours: int) -> pd.DataFrame:
        """Get energy prices for optimization period"""
        try:
            # Use existing price fetching logic
            if hasattr(self.da_calc, 'get_prices'):
                return self.da_calc.get_prices(start_time, hours)
            else:
                # Generate mock prices for testing
                return self._generate_mock_prices(start_time, hours)
        except Exception as e:
            logging.error(f"Error getting energy prices: {e}")
            return self._generate_mock_prices(start_time, hours)

    def _generate_mock_prices(self, start_time: dt.datetime, hours: int) -> pd.DataFrame:
        """Fallback functie - probeer echte prijzen uit database te halen"""
        try:
            # Probeer eerst echte prijzen uit de Report class te halen
            from .da_report import Report
            report = Report()

            try:
                price_data = report.get_price_data(start_time, start_time + dt.timedelta(hours=hours))
                if price_data is not None and not price_data.empty and 'da_ex' in price_data.columns:
                    logging.info("Echte prijs data opgehaald uit database")
                    # Converteer naar gewenste formaat
                    df = price_data.copy()
                    df['price'] = df['da_ex']  # Gebruik day-ahead prijs
                    df = df[['price']]
                    return df
            except Exception as e:
                logging.debug(f"Kon geen echte prijzen ophalen uit Report class: {e}")

            # Als fallback, genereer realistische prijzen gebaseerd op typische patronen
            logging.warning("Gebruik fallback prijs patroon - geen echte data beschikbaar")
            prices = []
            base_price = 0.20  # €0.20 per kWh

            for hour_offset in range(hours):
                timestamp = start_time + dt.timedelta(hours=hour_offset)
                hour = timestamp.hour

                # Realistische prijs patroon gebaseerd op Nederlandse energiemarkt
                if 17 <= hour <= 20:  # Avondpiek
                    price = base_price * 1.5
                elif 23 <= hour or hour <= 6:  # Nacht dal
                    price = base_price * 0.6
                elif 7 <= hour <= 9:  # Ochtendpiek
                    price = base_price * 1.3
                else:
                    price = base_price

                # Voeg realistische variatie toe (geen random, maar deterministisch)
                variation = 0.9 + 0.2 * (hash(f"{timestamp.date()}-{hour}") % 20) / 100
                price *= variation

                prices.append({
                    'datetime': timestamp,
                    'price': round(price, 4)
                })

            df = pd.DataFrame(prices)
            df.set_index('datetime', inplace=True)
            return df

        except Exception as e:
            logging.error(f"Fout bij genereren van prijs data: {e}")
            # Laatste fallback: lege DataFrame
            return pd.DataFrame()

    def _get_system_status(self) -> Dict:
        """Get current system status"""
        return {
            'battery_available': True,
            'solar_available': self._get_solar_config()['capacity'] > 0,
            'ev_available': self._is_ev_present(),
            'heating_available': True
        }

    def _get_battery_config(self) -> Dict:
        """Get battery configuration"""
        return {
            'capacity': self.config.get(['battery', 'capacity'], 0, 10.0),
            'power': self.config.get(['battery', 'power'], 0, 5.0),
            'efficiency': self.config.get(['battery', 'efficiency'], 0, 0.92),
            'soc_min': self.config.get(['battery', 'soc_min'], 0, 0.1),
            'soc_max': self.config.get(['battery', 'soc_max'], 0, 0.9)
        }

    def _get_solar_config(self) -> Dict:
        """Get solar configuration"""
        return {
            'capacity': self.config.get(['solar', 'capacity'], 0, 0.0),
            'efficiency': self.config.get(['solar', 'efficiency'], 0, 0.85),
            'tilt': self.config.get(['solar', 'tilt'], 0, 30.0),
            'azimuth': self.config.get(['solar', 'azimuth'], 0, 180.0)
        }

    def _get_ev_config(self) -> Dict:
        """Get EV configuration"""
        return {
            'capacity': self.config.get(['electric vehicle', 'capacity'], 0, 50.0),
            'power': self.config.get(['electric vehicle', 'power'], 0, 11.0),
            'efficiency': self.config.get(['electric vehicle', 'efficiency'], 0, 0.9)
        }

    def _is_ev_present(self) -> bool:
        """Check if EV is configured"""
        return self.config.get(['electric vehicle', 'present'], None, 'false').lower() == 'true'

    def _calculate_overall_prediction_confidence(self, results: Dict) -> float:
        """Calculate overall confidence in predictions used"""
        # Placeholder implementation
        return results.get('confidence', 0.8)

    def _calculate_weather_impact_score(self, results: Dict) -> float:
        """Calculate weather impact score"""
        weather_analysis = results.get('weather_analysis', {})
        if not weather_analysis:
            return 0.5

        # Simple scoring based on available weather data
        score = 0.0
        factors = ['solar_potential', 'heating_demand', 'cooling_demand']
        available_factors = sum(1 for factor in factors if factor in weather_analysis)

        return available_factors / len(factors)

    def _calculate_rule_effectiveness_score(self, results: Dict) -> float:
        """Calculate effectiveness score of applied rules"""
        rules_applied = results.get('rules_applied', [])
        if not rules_applied:
            return 0.0

        # Get rule effectiveness from performance monitor
        rule_stats = self.performance_monitor.rule_effectiveness

        if not rule_stats:
            return 0.7  # Default assumption

        total_effectiveness = sum(
            rule_stats.get(rule, {}).get('success_rate', 0.7)
            for rule in rules_applied
        )

        return total_effectiveness / len(rules_applied)

    def _generate_reports(self, results: Dict):
        """Generate optimization reports"""
        try:
            logging.info("Generating optimization reports...")

            # Log summary
            performance = results.get('performance', {})
            savings = performance.get('savings', 0)
            confidence = results.get('confidence', 0)
            method = results.get('optimization_method', 'unknown')

            logging.info(f"Optimization completed: method={method}, savings=€{savings:.2f}, confidence={confidence:.2f}")

            # More detailed reporting would go here

        except Exception as e:
            logging.error(f"Error generating reports: {e}")

    def _monitor_weather_reactive_conditions(self, weather_analysis: Dict) -> List:
        """Monitor current weather for reactive optimization events"""

        try:
            current_weather = weather_analysis.get('current_conditions', {})
            forecast_update = weather_analysis.get('forecast_changes', {})

            # Use weather reactive optimizer to detect events
            weather_events = self.weather_reactive.monitor_weather_changes(
                current_weather, forecast_update
            )

            if weather_events:
                logging.info(f"Weather reactive events detected: {[e.value for e in weather_events]}")

            return weather_events

        except Exception as e:
            logging.error(f"Error monitoring weather reactive conditions: {e}")
            return []

    def _apply_weather_reactive_adjustments(self, results: Dict, weather_events: List, weather_analysis: Dict) -> Dict:
        """Apply weather reactive adjustments to optimization results"""

        if not weather_events:
            return results

        try:
            # Get current schedule from results
            current_schedule = {
                'battery_schedule': results.get('battery_schedule', {}),
                'solar_forecast': results.get('solar_forecast', {}),
                'consumption_forecast': results.get('consumption_forecast', {}),
                'optimization_strategy': results.get('optimization_strategy', 'balanced')
            }

            # Apply weather reactive adjustments
            adjusted_schedule = self.weather_reactive.react_to_weather_events(
                weather_events, current_schedule
            )

            # Merge adjustments back into results
            if adjusted_schedule != current_schedule:
                results.update(adjusted_schedule)
                results['weather_reactive_applied'] = True
                results['weather_events'] = [e.value for e in weather_events]
                logging.info("Weather reactive adjustments applied to optimization")
            else:
                results['weather_reactive_applied'] = False

            return results

        except Exception as e:
            logging.error(f"Error applying weather reactive adjustments: {e}")
            return results

    def get_optimizer_status(self) -> Dict:
        """Get current optimizer status"""
        return {
            'predictor_status': 'active',
            'smart_optimizer_status': 'active',
            'weather_integrator_status': 'active',
            'performance_monitor_status': 'active',
            'weather_reactive_status': 'active',
            'multiday_optimizer_status': 'active',
            'optimization_strategy': self.optimization_strategy.value,
            'prediction_hours': self.prediction_hours,
            'hybrid_mode': self.hybrid_mode,
            'multiday_planning_available': True
        }