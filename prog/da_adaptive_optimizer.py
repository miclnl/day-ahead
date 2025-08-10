"""
Statistical Optimization Engine for DAO Modern.
Volledig gebaseerd op Statistical Intelligence voor maximale stabiliteit.
"""

import datetime as dt
import logging
from typing import Dict, Any, Optional

# Import statistical optimization engine
try:
    from da_statistical_optimizer import StatisticalEnergyOptimizer
    STATISTICAL_AVAILABLE = True
except ImportError as e:
    STATISTICAL_AVAILABLE = False
    logging.error(f"Statistical optimizer not available: {e}")
    raise ImportError("Statistical optimizer is required for DAO Modern")

class AdaptiveOptimizationEngine:
    """
    Statistical Optimization Engine voor DAO Modern.
    Volledig gebaseerd op statistical intelligence voor maximale stabiliteit.
    """
    
    def __init__(self, da_calc_instance):
        """Initialize with reference to main DaCalc instance"""
        self.da_calc = da_calc_instance
        self.config = da_calc_instance.config
        
        # Initialize statistical optimizer
        self.statistical_optimizer = None
        
        if STATISTICAL_AVAILABLE:
            try:
                self.statistical_optimizer = StatisticalEnergyOptimizer(da_calc_instance)
                logging.info("Statistical optimizer initialized successfully")
            except Exception as e:
                logging.error(f"Error initializing statistical optimizer: {e}")
                raise
        else:
            raise ImportError("Statistical optimizer is required but not available")
    
    def optimize_energy_schedule(
        self, 
        start_dt: dt.datetime = None, 
        start_soc: float = None
    ) -> Optional[Dict[str, Any]]:
        """
        Main optimization method using statistical intelligence
        
        Args:
            start_dt: Start datetime for optimization
            start_soc: Starting state of charge
            
        Returns:
            Dict containing optimization results
        """
        optimization_start = dt.datetime.now()
        
        try:
            if self.statistical_optimizer:
                logging.info("Running Statistical Intelligence optimization")
                results = self.statistical_optimizer.optimize_energy_schedule(start_dt, start_soc)
                
                if results:
                    results['optimizer_used'] = 'StatisticalEnergyOptimizer'
                    results['optimization_mode'] = 'statistical'
                    
                    duration = dt.datetime.now() - optimization_start
                    logging.info(f"Statistical optimization successful in {duration.total_seconds():.2f}s")
                    return results
                
                logging.warning("Statistical optimizer returned no results")
            
            logging.error("Statistical optimization failed")
            return None
            
        except Exception as e:
            logging.error(f"Error in statistical optimization: {e}", exc_info=True)
            return None
    
    def get_optimizer_status(self) -> Dict[str, Any]:
        """Get current optimizer status"""
        
        return {
            'optimization_mode': 'statistical',
            'statistical_available': STATISTICAL_AVAILABLE,
            'statistical_optimizer_ready': self.statistical_optimizer is not None,
            'active_optimizer': 'StatisticalEnergyOptimizer',
            'capabilities': {
                'statistical_prediction': STATISTICAL_AVAILABLE,
                'smart_rules': STATISTICAL_AVAILABLE,
                'weather_integration': STATISTICAL_AVAILABLE,
                'performance_monitoring': STATISTICAL_AVAILABLE,
                'ml_prediction': False,
                'ai_optimization': False
            }
        }
    
    def test_optimizers(self) -> Dict[str, bool]:
        """Test statistical optimizer"""
        
        results = {}
        
        # Test statistical optimizer
        if self.statistical_optimizer:
            try:
                # Quick test without actual optimization
                status = self.statistical_optimizer.get_optimizer_status()
                results['statistical'] = status.get('predictor_status') == 'active'
            except Exception as e:
                logging.error(f"Statistical optimizer test failed: {e}")
                results['statistical'] = False
        else:
            results['statistical'] = False
        
        logging.info(f"Optimizer test results: {results}")
        return results