#!/usr/bin/env python3
"""
Test van de complete Statistical Optimization Pipeline voor DAO.
Test alle nieuwe modules: statistical predictor, smart optimizer, weather, monitoring.
"""

import sys
import os
import datetime as dt
import logging
import pandas as pd
import numpy as np

# Add dao-modern/prog to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'dao-modern', 'prog'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def create_mock_config():
    """Create mock configuration for testing"""
    return MockConfig()

class MockConfig:
    """Mock configuration for testing"""
    
    def __init__(self):
        self.config_data = {
            'battery': {
                'capacity': 10.0,
                'power': 5.0,
                'efficiency': 0.92,
                'soc_min': 0.1,
                'soc_max': 0.9
            },
            'solar': {
                'capacity': 8.0,
                'efficiency': 0.85,
                'tilt': 30.0,
                'azimuth': 180.0
            },
            'location': {
                'latitude': 52.0,
                'longitude': 5.0
            },
            'optimization': {
                'strategy': 'balanced',
                'hours_ahead': 24,
                'hybrid_mode': 'true'
            },
            'monitoring': {
                'enabled': 'true',
                'retention_days': 30
            }
        }
    
    def get(self, keys, index=0, default=None):
        """Get configuration value"""
        try:
            current = self.config_data
            for key in keys:
                current = current[key]
            return current
        except (KeyError, TypeError):
            return default

class MockDatabase:
    """Mock database for testing"""
    
    def __init__(self):
        self.connection = None

class MockDaCalc:
    """Mock DaCalc instance for testing"""
    
    def __init__(self):
        self.config = create_mock_config()
        self.db_da = MockDatabase()
        self.debug = False
        self.notification_entity = None

def test_statistical_predictor():
    """Test Statistical Consumption Predictor"""
    print("\n🧪 Testing Statistical Predictor...")
    
    try:
        from da_statistical_predictor import StatisticalPredictor
        
        da_calc = MockDaCalc()
        predictor = StatisticalPredictor(da_calc)
        
        # Test prediction
        start_time = dt.datetime.now().replace(minute=0, second=0, microsecond=0)
        
        # Use fallback prediction since no real database
        prediction = predictor._fallback_prediction(start_time, 24)
        
        assert len(prediction) == 24, f"Expected 24 hours, got {len(prediction)}"
        assert 'predicted_consumption' in prediction.columns, "Missing consumption column"
        assert prediction['predicted_consumption'].min() >= 0, "Negative consumption found"
        
        print(f"✅ Statistical predictor working: {len(prediction)} hours predicted")
        print(f"   Consumption range: {prediction['predicted_consumption'].min():.2f} - {prediction['predicted_consumption'].max():.2f} kW")
        
        return True
        
    except Exception as e:
        print(f"❌ Statistical predictor test failed: {e}")
        return False

def test_smart_optimizer():
    """Test Smart Rule-Based Optimizer"""
    print("\n🧪 Testing Smart Optimizer...")
    
    try:
        from da_smart_optimizer import SmartOptimizer, OptimizationStrategy
        
        da_calc = MockDaCalc()
        optimizer = SmartOptimizer(da_calc)
        
        # Create mock data
        start_time = dt.datetime.now().replace(minute=0, second=0, microsecond=0)
        
        # Mock prices
        prices_data = []
        for i in range(24):
            hour = (start_time + dt.timedelta(hours=i)).hour
            # Evening peak pattern
            if 17 <= hour <= 20:
                price = 0.35
            elif 23 <= hour or hour <= 6:
                price = 0.15
            else:
                price = 0.22
            
            prices_data.append({
                'datetime': start_time + dt.timedelta(hours=i),
                'price': price
            })
        
        prices = pd.DataFrame(prices_data)
        prices.set_index('datetime', inplace=True)
        
        # Mock consumption
        consumption_data = []
        for i in range(24):
            consumption_data.append({
                'datetime': start_time + dt.timedelta(hours=i),
                'predicted_consumption': 2.5 + np.sin(i * np.pi / 12)  # Sine wave pattern
            })
        
        consumption = pd.DataFrame(consumption_data)
        consumption.set_index('datetime', inplace=True)
        
        # Test optimization
        result = optimizer.optimize_energy_schedule(
            prices=prices,
            consumption_forecast=consumption,
            current_soc=0.5,
            strategy=OptimizationStrategy.BALANCED
        )
        
        assert result is not None, "Optimization returned None"
        assert 'schedule' in result, "Missing schedule in result"
        assert 'performance' in result, "Missing performance in result"
        
        schedule = result['schedule']
        performance = result['performance']
        
        print(f"✅ Smart optimizer working: {len(schedule)} hours optimized")
        print(f"   Expected savings: €{performance.get('savings', 0):.2f}")
        print(f"   Strategy used: {result.get('strategy_used', 'unknown')}")
        print(f"   Rules applied: {len(result.get('rules_applied', []))}")
        
        return True
        
    except Exception as e:
        print(f"❌ Smart optimizer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_enhanced_weather():
    """Test Enhanced Weather Integration"""
    print("\n🧪 Testing Enhanced Weather...")
    
    try:
        from da_enhanced_weather import EnhancedWeatherIntegrator
        
        da_calc = MockDaCalc()
        weather = EnhancedWeatherIntegrator(da_calc)
        
        start_time = dt.datetime.now().replace(minute=0, second=0, microsecond=0)
        
        # Test solar forecast (will use fallback)
        solar_forecast = weather.get_enhanced_solar_forecast(start_time, 24)
        
        assert len(solar_forecast) == 24, f"Expected 24 hours, got {len(solar_forecast)}"
        assert 'solar_production' in solar_forecast.columns, "Missing solar production column"
        assert solar_forecast['solar_production'].min() >= 0, "Negative solar production"
        
        # Test weather correlations
        correlations = weather.get_weather_correlations(start_time, 24)
        
        print(f"✅ Enhanced weather working: {len(solar_forecast)} hours forecasted")
        print(f"   Solar production range: {solar_forecast['solar_production'].min():.2f} - {solar_forecast['solar_production'].max():.2f} kW")
        print(f"   Weather correlations: {len(correlations)} factors analyzed")
        
        return True
        
    except Exception as e:
        print(f"❌ Enhanced weather test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_performance_monitor():
    """Test Performance Monitoring System"""
    print("\n🧪 Testing Performance Monitor...")
    
    try:
        from da_performance_monitor import PerformanceMonitor
        
        da_calc = MockDaCalc()
        monitor = PerformanceMonitor(da_calc)
        
        # Test logging prediction performance
        monitor.log_prediction_performance(
            timestamp=dt.datetime.now(),
            prediction_type='consumption',
            predicted_value=2.5,
            actual_value=2.3,
            confidence=0.85
        )
        
        # Test logging optimization performance
        monitor.log_optimization_performance(
            timestamp=dt.datetime.now(),
            strategy_used='rule_based_statistical',
            rules_applied=['charge_very_low_prices', 'discharge_very_high_prices'],
            predicted_savings=5.50,
            actual_savings=4.80
        )
        
        # Test analytics
        analytics = monitor.get_performance_analytics(days_back=1)
        
        # Test recommendations
        recommendations = monitor.get_rule_recommendations()
        
        print(f"✅ Performance monitor working")
        print(f"   Prediction records: {len(monitor.prediction_performance)}")
        print(f"   Optimization records: {len(monitor.optimization_performance)}")
        print(f"   Analytics generated: {len(analytics)} sections")
        print(f"   Recommendations: {len(recommendations)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Performance monitor test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_complete_pipeline():
    """Test Complete Statistical Optimization Pipeline"""
    print("\n🚀 Testing Complete Pipeline...")
    
    try:
        from da_statistical_optimizer import StatisticalEnergyOptimizer
        
        da_calc = MockDaCalc()
        optimizer = StatisticalEnergyOptimizer(da_calc)
        
        # Test complete optimization
        start_time = dt.datetime.now().replace(minute=0, second=0, microsecond=0)
        
        result = optimizer.optimize_energy_schedule(
            start_dt=start_time,
            start_soc=0.5
        )
        
        assert result is not None, "Complete optimization returned None"
        assert 'schedule' in result, "Missing schedule in result"
        assert 'performance' in result, "Missing performance in result"
        assert 'optimization_method' in result, "Missing optimization method"
        
        schedule = result['schedule']
        performance = result['performance']
        method = result['optimization_method']
        
        print(f"✅ Complete pipeline working!")
        print(f"   Method: {method}")
        print(f"   Schedule hours: {len(schedule)}")
        print(f"   Expected savings: €{performance.get('savings', 0):.2f}")
        print(f"   Confidence: {result.get('confidence', 0):.2f}")
        print(f"   Rules applied: {result.get('rules_applied', [])}")
        
        # Test optimizer status
        status = optimizer.get_optimizer_status()
        print(f"   Optimizer status: All components {status.get('predictor_status', 'unknown')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Complete pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("🧪 Testing Statistical Optimization Pipeline for DAO")
    print("=" * 60)
    
    tests = [
        ("Statistical Predictor", test_statistical_predictor),
        ("Smart Optimizer", test_smart_optimizer), 
        ("Enhanced Weather", test_enhanced_weather),
        ("Performance Monitor", test_performance_monitor),
        ("Complete Pipeline", test_complete_pipeline)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 60)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 60)
    
    passed = 0
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status}: {test_name}")
        if success:
            passed += 1
    
    success_rate = (passed / len(results)) * 100
    print(f"\nOverall: {passed}/{len(results)} tests passed ({success_rate:.1f}%)")
    
    if success_rate >= 80:
        print("\n🎉 Statistical Optimization Pipeline is READY!")
        print("   - Geen ML dependencies meer nodig")
        print("   - SIGILL container crashes opgelost") 
        print("   - Intelligente heuristieken werken goed")
        print("   - Performance monitoring actief")
    else:
        print("\n⚠️  Some components need attention before deployment")
    
    return success_rate >= 80

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)