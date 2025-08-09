"""
Refactored optimization engine for Day Ahead Optimizer.
Split from the monolithic calc_optimum() function for better maintainability.
"""

import datetime as dt
import logging
import math
import pandas as pd
from mip import Model, xsum, minimize, BINARY, CONTINUOUS
from typing import List, Dict, Tuple, Optional, Any
from da_ai_optimizer import AIOptimizationManager
from da_ml_predictor import MLPredictionManager


class EnergyOptimizer:
    """
    Modular energy optimization engine using Mixed-Integer Programming.
    Replaces the monolithic calc_optimum() method with smaller, focused methods.
    """
    
    def __init__(self, da_calc_instance):
        """Initialize with reference to main DaCalc instance for access to config and data"""
        self.da_calc = da_calc_instance
        self.model = None
        
        # Initialize AI and ML components
        self.ai_manager = AIOptimizationManager(da_calc_instance.config)
        self.ml_manager = MLPredictionManager(da_calc_instance.config, da_calc_instance.db_da)
        
    def optimize_energy_schedule(
        self, 
        start_dt: dt.datetime = None, 
        start_soc: float = None
    ) -> Dict[str, Any]:
        """
        Main optimization entry point - replaces the monolithic calc_optimum()
        
        Args:
            start_dt: Start datetime for optimization
            start_soc: Starting state of charge
            
        Returns:
            Dict containing optimization results
        """
        logging.info(f"Starting modular energy optimization")
        
        # Step 1: Data collection and preparation
        optimization_data = self._collect_and_prepare_data(start_dt, start_soc)
        if optimization_data is None:
            return None
            
        # Step 2: Try AI optimization first (if configured)
        if await self.ai_manager.is_available():
            logging.info("Attempting AI-powered optimization")
            ai_solution = await self.ai_manager.optimize(optimization_data)
            
            if ai_solution:
                logging.info("AI optimization successful, using AI solution")
                results = ai_solution
                self._generate_reports(results)
                return results
            else:
                logging.info("AI optimization failed, falling back to traditional MIP")
        
        # Step 3: Build traditional MIP optimization model
        self.model = self._build_optimization_model(optimization_data)
        
        # Step 4: Solve optimization
        solution = self._solve_optimization()
        if solution is None:
            return None
            
        # Step 4: Process and save results
        results = self._process_results(solution, optimization_data)
        
        # Step 5: Generate reports and notifications
        self._generate_reports(results)
        
        logging.info("Modular optimization completed successfully")
        return results
    
    def _collect_and_prepare_data(self, start_dt: dt.datetime, start_soc: float) -> Optional[Dict]:
        """
        Collect and prepare all data needed for optimization.
        Replaces the data preparation section of calc_optimum().
        """
        logging.info("Collecting and preparing optimization data")
        
        # Determine optimization period
        if start_dt is None:
            start_dt = dt.datetime.now()
            
        start_ts = int(start_dt.timestamp())
        interval_s = self.da_calc.interval_s
        modulo = start_ts % interval_s
        
        if modulo > (interval_s - 10):
            start_ts = start_ts + interval_s - modulo
            
        start_dt = dt.datetime.fromtimestamp(start_ts)
        start_h = int(interval_s * math.floor(start_ts / interval_s))
        fraction_first_interval = 1 - (start_ts - start_h) / interval_s
        
        # Fetch forecast data
        prog_data = self.da_calc.db_da.get_prognose_data(
            start=start_h, end=None, interval=self.da_calc.interval
        )
        
        if len(prog_data) <= 2:
            logging.error("Insufficient forecast data for optimization")
            return None
            
        if len(prog_data) <= 8:
            logging.warning("Limited forecast data available")
            
        # Prepare pricing data
        price_data = self._prepare_pricing_data(prog_data)
        
        # Prepare solar data
        solar_data = self._prepare_solar_data(prog_data, start_ts, fraction_first_interval)
        
        # Prepare baseload data (with ML enhancement)
        baseload_data = await self._prepare_baseload_data_enhanced(len(price_data['consumption_prices']))
        
        # Prepare battery data
        battery_data = self._prepare_battery_data(start_soc)
        
        # Prepare EV data
        ev_data = self._prepare_ev_data()
        
        # Prepare heating/boiler data
        heating_data = self._prepare_heating_data(prog_data)
        
        return {
            'start_dt': start_dt,
            'start_ts': start_ts,
            'fraction_first_interval': fraction_first_interval,
            'time_periods': len(prog_data),
            'price_data': price_data,
            'solar_data': solar_data,
            'baseload_data': baseload_data,
            'battery_data': battery_data,
            'ev_data': ev_data,
            'heating_data': heating_data,
            'raw_prog_data': prog_data
        }
    
    def _prepare_pricing_data(self, prog_data: pd.DataFrame) -> Dict:
        """Extract and calculate pricing information"""
        prog_data = prog_data.reset_index()
        
        # Vectorized operations for pricing
        consumption_prices = prog_data['da_cons'].tolist()
        production_prices = prog_data['da_prod'].tolist() 
        spot_prices = prog_data['da_price'].tolist()
        timestamps = prog_data['tijd'].tolist()
        
        avg_price = sum(consumption_prices) / len(consumption_prices)
        avg_prices = [avg_price] * len(consumption_prices)
        
        return {
            'consumption_prices': consumption_prices,
            'production_prices': production_prices,
            'spot_prices': spot_prices,
            'average_prices': avg_prices,
            'timestamps': timestamps
        }
    
    def _prepare_solar_data(self, prog_data: pd.DataFrame, start_ts: int, fraction_first_interval: float) -> Dict:
        """Prepare solar production forecasts"""
        solar_data = {
            'timestamps': [],
            'hour_fractions': [],
            'global_radiation': [],
            'production_forecasts': []
        }
        
        solar_configs = getattr(self.da_calc, 'solar', [])
        battery_solar_configs = []
        
        # Extract solar configs from batteries  
        for battery_config in self.da_calc.battery_options:
            if 'solar' in battery_config:
                battery_solar_configs.extend(battery_config['solar'])
        
        prog_data = prog_data.reset_index()
        first_hour = True
        
        for _, row in prog_data.iterrows():
            row_time = row['time']
            dtime = dt.datetime.fromtimestamp(row_time)
            
            solar_data['timestamps'].append(row_time)
            solar_data['global_radiation'].append(row['glob_rad'])
            
            if first_hour:
                solar_data['hour_fractions'].append(fraction_first_interval)
                first_hour = False
            else:
                solar_data['hour_fractions'].append(1.0)
            
            # Calculate production for each solar installation
            hour_production = []
            hour_fraction = solar_data['hour_fractions'][-1]
            
            # Main solar installations
            for solar_config in solar_configs:
                production = self.da_calc.calc_prod_solar(
                    solar_config, row_time, row['glob_rad'], hour_fraction
                )
                hour_production.append(production)
            
            # Battery solar installations  
            for solar_config in battery_solar_configs:
                production = self.da_calc.calc_prod_solar(
                    solar_config, row_time, row['glob_rad'], hour_fraction
                )
                hour_production.append(production)
                
            solar_data['production_forecasts'].append(hour_production)
        
        return solar_data
    
    async def _prepare_baseload_data_enhanced(self, num_periods: int) -> Dict:
        """Prepare baseload data with ML enhancement"""
        try:
            # Try ML-enhanced adaptive baseload first
            adaptive_baseload = await self.ml_manager.get_adaptive_baseload()
            
            if adaptive_baseload and len(adaptive_baseload) >= 24:
                logging.info("Using ML-enhanced adaptive baseload")
                # Extend to cover full optimization period
                extended_baseload = adaptive_baseload * ((num_periods // 24) + 1)
                return {'baseload_consumption': extended_baseload[:num_periods]}
            else:
                logging.info("ML adaptive baseload not available, using traditional method")
                return self._prepare_baseload_data_traditional(num_periods)
                
        except Exception as e:
            logging.error(f"Error in enhanced baseload preparation: {e}")
            return self._prepare_baseload_data_traditional(num_periods)
    
    def _prepare_baseload_data_traditional(self, num_periods: int) -> Dict:
        """Prepare baseload consumption data"""
        if self.da_calc.use_calc_baseload:
            logging.info("Using calculated baseload")
            weekday = dt.datetime.weekday(dt.datetime.now())
            baseload = self.da_calc.get_calculated_baseload(weekday)
            
            if num_periods > 24:
                # Add next day
                next_weekday = (weekday + 1) % 7
                baseload.extend(self.da_calc.get_calculated_baseload(next_weekday))
        else:
            logging.info("Using configured baseload")
            baseload_config = self.da_calc.config.get(['baseload'], [])
            baseload = baseload_config * ((num_periods // 24) + 1)
            baseload = baseload[:num_periods]
        
        return {'baseload_consumption': baseload}
    
    def _prepare_battery_data(self, start_soc: float) -> Dict:
        """Prepare battery configuration and constraints"""
        battery_options = self.da_calc.battery_options
        
        if not battery_options:
            return {'batteries': []}
        
        batteries = []
        for i, battery_config in enumerate(battery_options):
            battery_data = {
                'index': i,
                'capacity': battery_config.get('capacity', 0),
                'max_charge_power': battery_config.get('max_charge_power', 0),
                'max_discharge_power': battery_config.get('max_discharge_power', 0),
                'charge_efficiency': battery_config.get('charge_efficiency', 0.95),
                'discharge_efficiency': battery_config.get('discharge_efficiency', 0.95),
                'min_soc': battery_config.get('min_soc', 10) / 100,
                'max_soc': battery_config.get('max_soc', 90) / 100,
                'start_soc': start_soc if start_soc is not None else battery_config.get('start_soc', 50) / 100,
            }
            batteries.append(battery_data)
        
        return {'batteries': batteries}
    
    def _prepare_ev_data(self) -> Dict:
        """Prepare electric vehicle charging data"""
        ev_options = self.da_calc.ev_options
        
        if not ev_options:
            return {'electric_vehicles': []}
        
        # Process EV configurations
        electric_vehicles = []
        for i, ev_config in enumerate(ev_options):
            ev_data = {
                'index': i,
                'capacity': ev_config.get('capacity', 0),
                'charge_power': ev_config.get('charge_power', 0),
                'charge_efficiency': ev_config.get('charge_efficiency', 0.9),
                'start_soc': ev_config.get('start_soc', 20) / 100,
                'target_soc': ev_config.get('target_soc', 80) / 100,
                'departure_time': ev_config.get('departure_time', '07:00'),
                'arrival_time': ev_config.get('arrival_time', '17:00'),
            }
            electric_vehicles.append(ev_data)
        
        return {'electric_vehicles': electric_vehicles}
    
    def _prepare_heating_data(self, prog_data: pd.DataFrame) -> Dict:
        """Prepare heating/boiler data including weather-based adjustments"""
        heating_data = {
            'boiler_enabled': False,
            'heat_pump_enabled': False,
            'temperature_forecast': []
        }
        
        # Extract temperature data
        if 'temp' in prog_data.columns:
            heating_data['temperature_forecast'] = prog_data['temp'].tolist()
        
        # Boiler configuration
        if self.da_calc.boiler_options and len(self.da_calc.boiler_options) > 0:
            heating_data['boiler_enabled'] = True
            heating_data['boiler_power'] = self.da_calc.boiler_options.get('power', 2000)
            heating_data['boiler_volume'] = self.da_calc.boiler_options.get('volume', 120)
            heating_data['boiler_setpoint'] = self.da_calc.boiler_options.get('setpoint', 60)
        
        # Heat pump configuration  
        if self.da_calc.heating_options and len(self.da_calc.heating_options) > 0:
            heating_data['heat_pump_enabled'] = True
            heating_data['heat_pump_power'] = self.da_calc.heating_options.get('power', 3000)
            heating_data['heat_pump_cop'] = self.da_calc.heating_options.get('cop', 3.5)
        
        return heating_data
    
    def _build_optimization_model(self, data: Dict) -> Model:
        """
        Build the MIP optimization model.
        Replaces the model building section of calc_optimum().
        """
        logging.info("Building optimization model")
        
        model = Model("EnergyOptimization")
        
        U = data['time_periods']  # Number of time periods
        B = len(data['battery_data']['batteries'])  # Number of batteries
        E = len(data['ev_data']['electric_vehicles'])  # Number of EVs
        
        # Decision variables
        variables = {}
        
        # Grid import/export variables
        variables['grid_import'] = [model.add_var(var_type=CONTINUOUS, lb=0) for u in range(U)]
        variables['grid_export'] = [model.add_var(var_type=CONTINUOUS, lb=0) for u in range(U)]
        
        # Battery variables
        if B > 0:
            variables['battery_charge'] = [[model.add_var(var_type=CONTINUOUS, lb=0) for u in range(U)] for b in range(B)]
            variables['battery_discharge'] = [[model.add_var(var_type=CONTINUOUS, lb=0) for u in range(U)] for b in range(B)]
            variables['battery_soc'] = [[model.add_var(var_type=CONTINUOUS, lb=0, ub=1) for u in range(U)] for b in range(B)]
            variables['battery_charge_binary'] = [[model.add_var(var_type=BINARY) for u in range(U)] for b in range(B)]
        
        # EV variables
        if E > 0:
            variables['ev_charge'] = [[model.add_var(var_type=CONTINUOUS, lb=0) for u in range(U)] for e in range(E)]
            variables['ev_soc'] = [[model.add_var(var_type=CONTINUOUS, lb=0, ub=1) for u in range(U)] for e in range(E)]
        
        # Heating variables
        if data['heating_data']['boiler_enabled']:
            variables['boiler_heating'] = [model.add_var(var_type=BINARY) for u in range(U)]
            variables['boiler_temp'] = [model.add_var(var_type=CONTINUOUS, lb=20, ub=80) for u in range(U)]
        
        if data['heating_data']['heat_pump_enabled']:
            variables['heat_pump_power'] = [model.add_var(var_type=CONTINUOUS, lb=0) for u in range(U)]
        
        # Add constraints
        self._add_energy_balance_constraints(model, variables, data)
        self._add_battery_constraints(model, variables, data)
        self._add_ev_constraints(model, variables, data)  
        self._add_heating_constraints(model, variables, data)
        self._add_grid_constraints(model, variables, data)
        
        # Set objective function
        self._set_objective_function(model, variables, data)
        
        return model
    
    def _add_energy_balance_constraints(self, model: Model, variables: Dict, data: Dict):
        """Add energy balance constraints to the model"""
        U = data['time_periods']
        
        for u in range(U):
            # Energy balance: Production + Grid Import = Consumption + Grid Export + Storage Charging
            solar_production = sum(data['solar_data']['production_forecasts'][u]) if data['solar_data']['production_forecasts'][u] else 0
            baseload = data['baseload_data']['baseload_consumption'][u] if u < len(data['baseload_data']['baseload_consumption']) else 0
            
            total_consumption = baseload
            
            # Add battery charging to consumption
            if 'battery_charge' in variables:
                for b in range(len(data['battery_data']['batteries'])):
                    total_consumption += variables['battery_charge'][b][u]
            
            # Add EV charging to consumption  
            if 'ev_charge' in variables:
                for e in range(len(data['ev_data']['electric_vehicles'])):
                    total_consumption += variables['ev_charge'][e][u]
            
            # Add heating consumption
            if 'boiler_heating' in variables:
                boiler_power = data['heating_data'].get('boiler_power', 2000) / 1000  # Convert to kW
                total_consumption += variables['boiler_heating'][u] * boiler_power
            
            if 'heat_pump_power' in variables:
                total_consumption += variables['heat_pump_power'][u]
            
            total_production = solar_production
            
            # Add battery discharging to production
            if 'battery_discharge' in variables:
                for b in range(len(data['battery_data']['batteries'])):
                    total_production += variables['battery_discharge'][b][u] * data['battery_data']['batteries'][b]['discharge_efficiency']
            
            # Energy balance constraint
            model.add_constr(
                total_production + variables['grid_import'][u] == 
                total_consumption + variables['grid_export'][u]
            )
    
    def _add_battery_constraints(self, model: Model, variables: Dict, data: Dict):
        """Add battery-specific constraints"""
        if 'battery_charge' not in variables:
            return
            
        batteries = data['battery_data']['batteries']
        U = data['time_periods']
        
        for b, battery in enumerate(batteries):
            for u in range(U):
                # Charge/discharge power limits
                model.add_constr(variables['battery_charge'][b][u] <= battery['max_charge_power'])
                model.add_constr(variables['battery_discharge'][b][u] <= battery['max_discharge_power'])
                
                # SOC limits
                model.add_constr(variables['battery_soc'][b][u] >= battery['min_soc'])
                model.add_constr(variables['battery_soc'][b][u] <= battery['max_soc'])
                
                # SOC evolution
                if u == 0:
                    # First period
                    model.add_constr(
                        variables['battery_soc'][b][u] == battery['start_soc'] + 
                        (variables['battery_charge'][b][u] * battery['charge_efficiency'] - 
                         variables['battery_discharge'][b][u]) / battery['capacity']
                    )
                else:
                    # Subsequent periods
                    model.add_constr(
                        variables['battery_soc'][b][u] == variables['battery_soc'][b][u-1] + 
                        (variables['battery_charge'][b][u] * battery['charge_efficiency'] - 
                         variables['battery_discharge'][b][u]) / battery['capacity']
                    )
                
                # Prevent simultaneous charge/discharge using binary variable
                M = max(battery['max_charge_power'], battery['max_discharge_power'])  # Big M
                model.add_constr(variables['battery_charge'][b][u] <= M * variables['battery_charge_binary'][b][u])
                model.add_constr(variables['battery_discharge'][b][u] <= M * (1 - variables['battery_charge_binary'][b][u]))
    
    def _add_ev_constraints(self, model: Model, variables: Dict, data: Dict):
        """Add electric vehicle charging constraints"""
        if 'ev_charge' not in variables:
            return
            
        evs = data['ev_data']['electric_vehicles']
        U = data['time_periods']
        
        for e, ev in enumerate(evs):
            for u in range(U):
                # Charge power limits
                model.add_constr(variables['ev_charge'][e][u] <= ev['charge_power'])
                
                # SOC limits
                model.add_constr(variables['ev_soc'][e][u] >= 0)
                model.add_constr(variables['ev_soc'][e][u] <= 1)
                
                # SOC evolution
                if u == 0:
                    model.add_constr(
                        variables['ev_soc'][e][u] == ev['start_soc'] + 
                        (variables['ev_charge'][e][u] * ev['charge_efficiency']) / ev['capacity']
                    )
                else:
                    model.add_constr(
                        variables['ev_soc'][e][u] == variables['ev_soc'][e][u-1] + 
                        (variables['ev_charge'][e][u] * ev['charge_efficiency']) / ev['capacity']
                    )
            
            # Target SOC constraint (must reach target before departure)
            # This would need time-based logic based on departure_time
            # For now, ensure target is reached by end of optimization period
            if U > 0:
                model.add_constr(variables['ev_soc'][e][U-1] >= ev['target_soc'])
    
    def _add_heating_constraints(self, model: Model, variables: Dict, data: Dict):
        """Add heating and boiler constraints"""
        heating_data = data['heating_data']
        
        if heating_data['boiler_enabled'] and 'boiler_heating' in variables:
            # Add boiler temperature evolution constraints
            # Simplified model - could be enhanced with thermal dynamics
            pass
        
        if heating_data['heat_pump_enabled'] and 'heat_pump_power' in variables:
            # Add heat pump constraints based on COP and temperature
            # Simplified for now
            max_hp_power = heating_data.get('heat_pump_power', 3000) / 1000  # Convert to kW
            U = data['time_periods']
            
            for u in range(U):
                model.add_constr(variables['heat_pump_power'][u] <= max_hp_power)
    
    def _add_grid_constraints(self, model: Model, variables: Dict, data: Dict):
        """Add grid connection constraints"""
        U = data['time_periods']
        max_grid_power = self.da_calc.grid_max_power
        
        for u in range(U):
            # Grid import/export limits
            model.add_constr(variables['grid_import'][u] <= max_grid_power)
            model.add_constr(variables['grid_export'][u] <= max_grid_power)
    
    def _set_objective_function(self, model: Model, variables: Dict, data: Dict):
        """Set the optimization objective function"""
        U = data['time_periods']
        price_data = data['price_data']
        
        # Choose strategy
        strategy = self.da_calc.config.get(['strategy'], 'minimize cost')
        
        if strategy == 'minimize cost':
            # Minimize total energy costs
            total_cost = xsum(
                variables['grid_import'][u] * price_data['consumption_prices'][u] - 
                variables['grid_export'][u] * price_data['production_prices'][u]
                for u in range(U)
            )
            model.objective = minimize(total_cost)
            
        elif strategy == 'maximize self_consumption':
            # Maximize self-consumption (minimize grid exchange)
            grid_exchange = xsum(
                variables['grid_import'][u] + variables['grid_export'][u]
                for u in range(U)
            )
            model.objective = minimize(grid_exchange)
        
        else:
            # Default to cost minimization
            total_cost = xsum(
                variables['grid_import'][u] * price_data['consumption_prices'][u] - 
                variables['grid_export'][u] * price_data['production_prices'][u]
                for u in range(U)
            )
            model.objective = minimize(total_cost)
    
    def _solve_optimization(self) -> Optional[Dict]:
        """
        Solve the optimization model.
        Replaces the solving section of calc_optimum().
        """
        logging.info("Solving optimization model")
        
        try:
            # Set solver parameters
            self.model.solver_name = 'CBC'  # Can be configured
            self.model.threads = -1  # Use all available threads
            self.model.max_seconds = 300  # 5 minute timeout
            
            # Solve the model
            status = self.model.optimize()
            
            if status.value != 0:  # Not optimal
                logging.error(f"Optimization failed with status: {status}")
                return None
            
            logging.info(f"Optimization completed with status: {status}")
            logging.info(f"Objective value: {self.model.objective_value}")
            
            return {'status': status, 'objective_value': self.model.objective_value}
            
        except Exception as e:
            logging.error(f"Error during optimization: {e}")
            return None
    
    def _process_results(self, solution: Dict, data: Dict) -> Dict:
        """
        Process optimization results and prepare output data.
        Replaces the results processing section of calc_optimum().
        """
        logging.info("Processing optimization results")
        
        U = data['time_periods']
        results = {
            'timestamps': data['price_data']['timestamps'],
            'grid_import': [],
            'grid_export': [],
            'total_cost': solution['objective_value'],
            'batteries': [],
            'electric_vehicles': [],
            'heating': {}
        }
        
        # Extract grid import/export results
        model_vars = self.model.vars
        grid_import_vars = [var for var in model_vars if 'grid_import' in var.name]
        grid_export_vars = [var for var in model_vars if 'grid_export' in var.name]
        
        for u in range(U):
            import_var = next((var for var in grid_import_vars if f"[{u}]" in var.name), None)
            export_var = next((var for var in grid_export_vars if f"[{u}]" in var.name), None)
            
            results['grid_import'].append(import_var.x if import_var else 0)
            results['grid_export'].append(export_var.x if export_var else 0)
        
        # Extract battery results
        if data['battery_data']['batteries']:
            results['batteries'] = self._extract_battery_results(data)
        
        # Extract EV results
        if data['ev_data']['electric_vehicles']:
            results['electric_vehicles'] = self._extract_ev_results(data)
        
        # Extract heating results
        results['heating'] = self._extract_heating_results(data)
        
        return results
    
    def _extract_battery_results(self, data: Dict) -> List[Dict]:
        """Extract battery optimization results"""
        battery_results = []
        U = data['time_periods']
        B = len(data['battery_data']['batteries'])
        
        model_vars = self.model.vars
        
        for b in range(B):
            battery_result = {
                'charge_power': [],
                'discharge_power': [],
                'soc': []
            }
            
            for u in range(U):
                # Find corresponding variables
                charge_var = next((var for var in model_vars if f"battery_charge[{b}][{u}]" in var.name), None)
                discharge_var = next((var for var in model_vars if f"battery_discharge[{b}][{u}]" in var.name), None)
                soc_var = next((var for var in model_vars if f"battery_soc[{b}][{u}]" in var.name), None)
                
                battery_result['charge_power'].append(charge_var.x if charge_var else 0)
                battery_result['discharge_power'].append(discharge_var.x if discharge_var else 0)
                battery_result['soc'].append(soc_var.x if soc_var else 0)
            
            battery_results.append(battery_result)
        
        return battery_results
    
    def _extract_ev_results(self, data: Dict) -> List[Dict]:
        """Extract EV optimization results"""
        ev_results = []
        U = data['time_periods']
        E = len(data['ev_data']['electric_vehicles'])
        
        model_vars = self.model.vars
        
        for e in range(E):
            ev_result = {
                'charge_power': [],
                'soc': []
            }
            
            for u in range(U):
                charge_var = next((var for var in model_vars if f"ev_charge[{e}][{u}]" in var.name), None)
                soc_var = next((var for var in model_vars if f"ev_soc[{e}][{u}]" in var.name), None)
                
                ev_result['charge_power'].append(charge_var.x if charge_var else 0)
                ev_result['soc'].append(soc_var.x if soc_var else 0)
            
            ev_results.append(ev_result)
        
        return ev_results
    
    def _extract_heating_results(self, data: Dict) -> Dict:
        """Extract heating optimization results"""
        heating_results = {}
        U = data['time_periods']
        model_vars = self.model.vars
        
        if data['heating_data']['boiler_enabled']:
            heating_results['boiler_heating'] = []
            for u in range(U):
                boiler_var = next((var for var in model_vars if f"boiler_heating[{u}]" in var.name), None)
                heating_results['boiler_heating'].append(boiler_var.x if boiler_var else 0)
        
        if data['heating_data']['heat_pump_enabled']:
            heating_results['heat_pump_power'] = []
            for u in range(U):
                hp_var = next((var for var in model_vars if f"heat_pump_power[{u}]" in var.name), None)
                heating_results['heat_pump_power'].append(hp_var.x if hp_var else 0)
        
        return heating_results
    
    def _generate_reports(self, results: Dict):
        """
        Generate reports and send notifications.
        Replaces the reporting section of calc_optimum().
        """
        logging.info("Generating optimization reports")
        
        # Log summary statistics
        total_import = sum(results['grid_import'])
        total_export = sum(results['grid_export'])
        net_consumption = total_import - total_export
        
        logging.info(f"Optimization summary:")
        logging.info(f"  Total grid import: {total_import:.2f} kWh")  
        logging.info(f"  Total grid export: {total_export:.2f} kWh")
        logging.info(f"  Net consumption: {net_consumption:.2f} kWh")
        logging.info(f"  Total cost: €{results['total_cost']:.2f}")
        
        # Send notification if configured
        if hasattr(self.da_calc, 'notification_entity') and self.da_calc.notification_entity:
            message = f"Optimalisatie voltooid. Kosten: €{results['total_cost']:.2f}, Net verbruik: {net_consumption:.1f} kWh"
            self.da_calc.set_value(self.da_calc.notification_entity, message)