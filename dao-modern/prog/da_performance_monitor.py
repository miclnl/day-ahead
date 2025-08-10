"""
Performance Monitoring System for Day Ahead Optimizer.
Tracks effectiveness of statistical prediction and rule-based optimization.
Provides analytics and adaptive learning capabilities.
"""

import datetime as dt
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import json
import os
from sqlalchemy import Table, Column, DateTime, Float, String, Integer, Boolean, Text
from sqlalchemy.orm import sessionmaker


class PerformanceMonitor:
    """
    Monitors and analyzes performance of DAO optimization strategies.
    Tracks prediction accuracy, cost savings, and rule effectiveness.
    """
    
    def __init__(self, da_calc_instance):
        """Initialize with reference to main DaCalc instance"""
        self.da_calc = da_calc_instance
        self.db_da = da_calc_instance.db_da
        self.config = da_calc_instance.config
        
        # Performance tracking storage
        self.prediction_performance = []
        self.optimization_performance = []
        self.rule_effectiveness = {}
        self.cost_savings_history = []
        
        # Monitoring configuration
        self.monitoring_enabled = self.config.get(['monitoring', 'enabled'], None, 'true').lower() == 'true'
        self.retention_days = self.config.get(['monitoring', 'retention_days'], 0, 90)
        self.alert_thresholds = {
            'prediction_accuracy_min': 0.7,      # 70% minimum accuracy
            'cost_savings_min': 0.05,            # 5% minimum savings
            'rule_success_rate_min': 0.6         # 60% minimum rule success
        }
        
        # Performance metrics cache
        self.metrics_cache = {
            'last_updated': None,
            'cache_duration': dt.timedelta(minutes=30),
            'cached_metrics': {}
        }
        
        # Initialize database table if needed
        self._initialize_monitoring_tables()
        
        logging.info("Performance monitor initialized")
    
    def log_prediction_performance(
        self, 
        timestamp: dt.datetime,
        prediction_type: str,
        predicted_value: float,
        actual_value: float,
        confidence: float = None,
        weather_conditions: Dict = None
    ):
        """
        Log prediction performance for analysis
        
        Args:
            timestamp: Time of prediction
            prediction_type: Type of prediction (consumption, solar, etc.)
            predicted_value: Predicted value
            actual_value: Actual measured value
            confidence: Prediction confidence (0-1)
            weather_conditions: Weather conditions during prediction
        """
        if not self.monitoring_enabled:
            return
        
        try:
            error = actual_value - predicted_value
            error_percent = (abs(error) / actual_value * 100) if actual_value != 0 else 0
            accuracy = max(0, 1 - (abs(error) / max(abs(actual_value), abs(predicted_value))))
            
            performance_record = {
                'timestamp': timestamp,
                'prediction_type': prediction_type,
                'predicted_value': predicted_value,
                'actual_value': actual_value,
                'error': error,
                'error_percent': error_percent,
                'accuracy': accuracy,
                'confidence': confidence or 0.0,
                'weather_conditions': json.dumps(weather_conditions) if weather_conditions else None
            }
            
            self.prediction_performance.append(performance_record)
            
            # Save to database
            self._save_prediction_performance(performance_record)
            
            # Check for alerts
            self._check_prediction_alerts(performance_record)
            
        except Exception as e:
            logging.error(f"Error logging prediction performance: {e}")
    
    def log_optimization_performance(
        self,
        timestamp: dt.datetime,
        strategy_used: str,
        rules_applied: List[str],
        predicted_savings: float,
        actual_savings: float = None,
        battery_cycles: float = None,
        grid_independence: float = None
    ):
        """
        Log optimization strategy performance
        
        Args:
            timestamp: Time of optimization
            strategy_used: Optimization strategy name
            rules_applied: List of rules that were applied
            predicted_savings: Predicted cost savings
            actual_savings: Actual measured savings (if available)
            battery_cycles: Battery cycles used
            grid_independence: Grid independence achieved
        """
        if not self.monitoring_enabled:
            return
        
        try:
            performance_record = {
                'timestamp': timestamp,
                'strategy_used': strategy_used,
                'rules_applied': json.dumps(rules_applied),
                'predicted_savings': predicted_savings,
                'actual_savings': actual_savings,
                'battery_cycles': battery_cycles,
                'grid_independence': grid_independence,
                'success': actual_savings >= predicted_savings * 0.8 if actual_savings else None
            }
            
            self.optimization_performance.append(performance_record)
            
            # Update rule effectiveness
            self._update_rule_effectiveness(rules_applied, actual_savings, predicted_savings)
            
            # Save to database
            self._save_optimization_performance(performance_record)
            
        except Exception as e:
            logging.error(f"Error logging optimization performance: {e}")
    
    def log_cost_savings(
        self,
        timestamp: dt.datetime,
        baseline_cost: float,
        optimized_cost: float,
        savings_amount: float,
        savings_percent: float,
        energy_sources: Dict = None
    ):
        """Log cost savings achieved"""
        
        if not self.monitoring_enabled:
            return
        
        try:
            savings_record = {
                'timestamp': timestamp,
                'baseline_cost': baseline_cost,
                'optimized_cost': optimized_cost,
                'savings_amount': savings_amount,
                'savings_percent': savings_percent,
                'energy_sources': json.dumps(energy_sources) if energy_sources else None
            }
            
            self.cost_savings_history.append(savings_record)
            
            # Save to database
            self._save_cost_savings(savings_record)
            
            # Check savings alerts
            self._check_savings_alerts(savings_record)
            
        except Exception as e:
            logging.error(f"Error logging cost savings: {e}")
    
    def get_performance_analytics(self, days_back: int = 30) -> Dict[str, Any]:
        """
        Get comprehensive performance analytics
        
        Args:
            days_back: Number of days to analyze
            
        Returns:
            Dict with performance analytics
        """
        try:
            # Check cache first
            if self._is_cache_valid():
                return self.metrics_cache['cached_metrics']
            
            cutoff_date = dt.datetime.now() - dt.timedelta(days=days_back)
            
            analytics = {
                'prediction_analytics': self._analyze_prediction_performance(cutoff_date),
                'optimization_analytics': self._analyze_optimization_performance(cutoff_date),
                'cost_analytics': self._analyze_cost_performance(cutoff_date),
                'rule_analytics': self._analyze_rule_effectiveness(cutoff_date),
                'trend_analytics': self._analyze_performance_trends(cutoff_date),
                'alert_summary': self._get_alert_summary(cutoff_date),
                'generated_at': dt.datetime.now(),
                'period_days': days_back
            }
            
            # Update cache
            self.metrics_cache = {
                'last_updated': dt.datetime.now(),
                'cache_duration': dt.timedelta(minutes=30),
                'cached_metrics': analytics
            }
            
            return analytics
            
        except Exception as e:
            logging.error(f"Error generating performance analytics: {e}")
            return {'error': str(e)}
    
    def get_rule_recommendations(self) -> List[Dict]:
        """Get recommendations for improving rule effectiveness"""
        
        recommendations = []
        
        try:
            # Analyze rule performance
            for rule_name, stats in self.rule_effectiveness.items():
                success_rate = stats.get('success_rate', 0)
                usage_count = stats.get('usage_count', 0)
                
                if usage_count >= 5:  # Enough data points
                    if success_rate < self.alert_thresholds['rule_success_rate_min']:
                        recommendations.append({
                            'type': 'rule_improvement',
                            'rule_name': rule_name,
                            'current_success_rate': success_rate,
                            'usage_count': usage_count,
                            'recommendation': f'Consider adjusting rule "{rule_name}" - low success rate',
                            'priority': 'high' if success_rate < 0.4 else 'medium'
                        })
                    elif success_rate > 0.9:
                        recommendations.append({
                            'type': 'rule_expansion',
                            'rule_name': rule_name,
                            'current_success_rate': success_rate,
                            'usage_count': usage_count,
                            'recommendation': f'Consider expanding usage of successful rule "{rule_name}"',
                            'priority': 'low'
                        })
            
            # Check for missing rule coverage
            recommendations.extend(self._identify_coverage_gaps())
            
            return sorted(recommendations, key=lambda x: {'high': 3, 'medium': 2, 'low': 1}[x['priority']], reverse=True)
            
        except Exception as e:
            logging.error(f"Error generating rule recommendations: {e}")
            return []
    
    def _analyze_prediction_performance(self, cutoff_date: dt.datetime) -> Dict:
        """Analyze prediction accuracy and patterns"""
        
        recent_predictions = [
            p for p in self.prediction_performance 
            if p['timestamp'] > cutoff_date
        ]
        
        if not recent_predictions:
            return {'no_data': True}
        
        # Group by prediction type
        by_type = {}
        for pred in recent_predictions:
            pred_type = pred['prediction_type']
            if pred_type not in by_type:
                by_type[pred_type] = []
            by_type[pred_type].append(pred)
        
        analytics = {}
        
        for pred_type, predictions in by_type.items():
            accuracies = [p['accuracy'] for p in predictions]
            errors = [p['error_percent'] for p in predictions]
            confidences = [p['confidence'] for p in predictions if p['confidence'] is not None]
            
            analytics[pred_type] = {
                'count': len(predictions),
                'avg_accuracy': np.mean(accuracies),
                'avg_error_percent': np.mean(errors),
                'max_error_percent': np.max(errors),
                'avg_confidence': np.mean(confidences) if confidences else None,
                'accuracy_trend': self._calculate_trend([p['accuracy'] for p in predictions[-20:]]),
                'recent_performance': 'good' if np.mean(accuracies[-10:]) > 0.8 else 'poor'
            }
        
        return analytics
    
    def _analyze_optimization_performance(self, cutoff_date: dt.datetime) -> Dict:
        """Analyze optimization strategy effectiveness"""
        
        recent_optimizations = [
            o for o in self.optimization_performance 
            if o['timestamp'] > cutoff_date
        ]
        
        if not recent_optimizations:
            return {'no_data': True}
        
        # Strategy analysis
        strategies = {}
        for opt in recent_optimizations:
            strategy = opt['strategy_used']
            if strategy not in strategies:
                strategies[strategy] = []
            strategies[strategy].append(opt)
        
        strategy_analytics = {}
        for strategy, opts in strategies.items():
            successful = [o for o in opts if o.get('success') is True]
            predicted_savings = [o['predicted_savings'] for o in opts]
            actual_savings = [o['actual_savings'] for o in opts if o['actual_savings'] is not None]
            
            strategy_analytics[strategy] = {
                'usage_count': len(opts),
                'success_rate': len(successful) / len(opts) if opts else 0,
                'avg_predicted_savings': np.mean(predicted_savings),
                'avg_actual_savings': np.mean(actual_savings) if actual_savings else None,
                'prediction_accuracy': self._calculate_prediction_vs_actual(
                    [o['predicted_savings'] for o in opts],
                    [o['actual_savings'] for o in opts if o['actual_savings'] is not None]
                )
            }
        
        return {
            'total_optimizations': len(recent_optimizations),
            'strategy_performance': strategy_analytics,
            'overall_success_rate': len([o for o in recent_optimizations if o.get('success')]) / len(recent_optimizations)
        }
    
    def _analyze_cost_performance(self, cutoff_date: dt.datetime) -> Dict:
        """Analyze cost savings performance"""
        
        recent_savings = [
            s for s in self.cost_savings_history 
            if s['timestamp'] > cutoff_date
        ]
        
        if not recent_savings:
            return {'no_data': True}
        
        savings_amounts = [s['savings_amount'] for s in recent_savings]
        savings_percents = [s['savings_percent'] for s in recent_savings]
        
        return {
            'total_periods': len(recent_savings),
            'total_savings': sum(savings_amounts),
            'avg_savings_amount': np.mean(savings_amounts),
            'avg_savings_percent': np.mean(savings_percents),
            'max_savings_amount': max(savings_amounts),
            'max_savings_percent': max(savings_percents),
            'consistent_savings': len([s for s in savings_percents if s > 5]) / len(savings_percents),
            'savings_trend': self._calculate_trend(savings_amounts[-20:])
        }
    
    def _analyze_rule_effectiveness(self, cutoff_date: dt.datetime) -> Dict:
        """Analyze individual rule effectiveness"""
        
        rule_analytics = {}
        
        for rule_name, stats in self.rule_effectiveness.items():
            if stats.get('last_used', dt.datetime.min) > cutoff_date:
                rule_analytics[rule_name] = {
                    'usage_count': stats.get('usage_count', 0),
                    'success_rate': stats.get('success_rate', 0),
                    'avg_impact': stats.get('avg_impact', 0),
                    'last_used': stats.get('last_used'),
                    'effectiveness': 'high' if stats.get('success_rate', 0) > 0.8 else 'medium' if stats.get('success_rate', 0) > 0.6 else 'low'
                }
        
        return rule_analytics
    
    def _analyze_performance_trends(self, cutoff_date: dt.datetime) -> Dict:
        """Analyze performance trends over time"""
        
        # Get daily aggregated data
        daily_performance = self._aggregate_daily_performance(cutoff_date)
        
        if len(daily_performance) < 7:  # Need at least a week of data
            return {'insufficient_data': True}
        
        days = sorted(daily_performance.keys())
        savings_trend = [daily_performance[day]['savings'] for day in days]
        accuracy_trend = [daily_performance[day]['accuracy'] for day in days]
        
        return {
            'savings_trend': self._calculate_trend(savings_trend),
            'accuracy_trend': self._calculate_trend(accuracy_trend),
            'volatility': {
                'savings': np.std(savings_trend),
                'accuracy': np.std(accuracy_trend)
            },
            'recent_vs_historical': self._compare_recent_vs_historical(daily_performance)
        }
    
    def _update_rule_effectiveness(
        self, 
        rules_applied: List[str], 
        actual_savings: float, 
        predicted_savings: float
    ):
        """Update effectiveness metrics for applied rules"""
        
        if actual_savings is None:
            return
        
        success = actual_savings >= predicted_savings * 0.8  # 80% threshold
        impact = actual_savings - predicted_savings
        
        for rule_name in rules_applied:
            if rule_name not in self.rule_effectiveness:
                self.rule_effectiveness[rule_name] = {
                    'usage_count': 0,
                    'success_count': 0,
                    'total_impact': 0,
                    'success_rate': 0,
                    'avg_impact': 0,
                    'last_used': dt.datetime.now()
                }
            
            stats = self.rule_effectiveness[rule_name]
            stats['usage_count'] += 1
            stats['last_used'] = dt.datetime.now()
            
            if success:
                stats['success_count'] += 1
            
            stats['total_impact'] += impact
            stats['success_rate'] = stats['success_count'] / stats['usage_count']
            stats['avg_impact'] = stats['total_impact'] / stats['usage_count']
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction from values"""
        
        if len(values) < 2:
            return 'insufficient_data'
        
        # Simple linear trend
        x = list(range(len(values)))
        correlation = np.corrcoef(x, values)[0, 1] if len(values) > 2 else 0
        
        if correlation > 0.3:
            return 'improving'
        elif correlation < -0.3:
            return 'declining'
        else:
            return 'stable'
    
    def _calculate_prediction_vs_actual(
        self, 
        predicted: List[float], 
        actual: List[float]
    ) -> float:
        """Calculate accuracy of predictions vs actual values"""
        
        if not predicted or not actual or len(predicted) != len(actual):
            return 0.0
        
        errors = [abs(p - a) / max(abs(a), abs(p)) for p, a in zip(predicted, actual) if max(abs(a), abs(p)) > 0]
        
        if not errors:
            return 0.0
        
        return 1 - np.mean(errors)  # Convert error to accuracy
    
    def _check_prediction_alerts(self, performance_record: Dict):
        """Check if prediction performance triggers alerts"""
        
        accuracy = performance_record['accuracy']
        
        if accuracy < self.alert_thresholds['prediction_accuracy_min']:
            self._send_alert(
                'prediction_accuracy_low',
                f"Prediction accuracy below threshold: {accuracy:.2f} < {self.alert_thresholds['prediction_accuracy_min']}"
            )
    
    def _check_savings_alerts(self, savings_record: Dict):
        """Check if savings performance triggers alerts"""
        
        savings_percent = savings_record['savings_percent']
        
        if savings_percent < self.alert_thresholds['cost_savings_min'] * 100:
            self._send_alert(
                'cost_savings_low',
                f"Cost savings below threshold: {savings_percent:.2f}% < {self.alert_thresholds['cost_savings_min']*100}%"
            )
    
    def _send_alert(self, alert_type: str, message: str):
        """Send performance alert"""
        
        try:
            logging.warning(f"Performance Alert [{alert_type}]: {message}")
            
            # Send to Home Assistant notification if available
            if hasattr(self.da_calc, 'notification_entity') and self.da_calc.notification_entity:
                self.da_calc.set_value(
                    self.da_calc.notification_entity,
                    f"DAO Performance Alert: {message}"
                )
        
        except Exception as e:
            logging.error(f"Error sending alert: {e}")
    
    def _initialize_monitoring_tables(self):
        """Initialize database tables for monitoring if they don't exist"""
        
        try:
            # This would create tables in the database
            # Implementation depends on database schema
            pass
        
        except Exception as e:
            logging.error(f"Error initializing monitoring tables: {e}")
    
    def _save_prediction_performance(self, record: Dict):
        """Save prediction performance to database"""
        try:
            # Implementation would save to database
            pass
        except Exception as e:
            logging.error(f"Error saving prediction performance: {e}")
    
    def _save_optimization_performance(self, record: Dict):
        """Save optimization performance to database"""
        try:
            # Implementation would save to database
            pass
        except Exception as e:
            logging.error(f"Error saving optimization performance: {e}")
    
    def _save_cost_savings(self, record: Dict):
        """Save cost savings to database"""
        try:
            # Implementation would save to database
            pass
        except Exception as e:
            logging.error(f"Error saving cost savings: {e}")
    
    def _is_cache_valid(self) -> bool:
        """Check if metrics cache is still valid"""
        
        if self.metrics_cache['last_updated'] is None:
            return False
        
        age = dt.datetime.now() - self.metrics_cache['last_updated']
        return age < self.metrics_cache['cache_duration']
    
    def _aggregate_daily_performance(self, cutoff_date: dt.datetime) -> Dict:
        """Aggregate performance data by day"""
        
        daily_data = {}
        
        # Aggregate savings data
        for savings in self.cost_savings_history:
            if savings['timestamp'] > cutoff_date:
                day = savings['timestamp'].date()
                if day not in daily_data:
                    daily_data[day] = {'savings': 0, 'accuracy': []}
                daily_data[day]['savings'] += savings['savings_amount']
        
        # Aggregate accuracy data
        for prediction in self.prediction_performance:
            if prediction['timestamp'] > cutoff_date:
                day = prediction['timestamp'].date()
                if day not in daily_data:
                    daily_data[day] = {'savings': 0, 'accuracy': []}
                daily_data[day]['accuracy'].append(prediction['accuracy'])
        
        # Calculate daily averages
        for day in daily_data:
            if daily_data[day]['accuracy']:
                daily_data[day]['accuracy'] = np.mean(daily_data[day]['accuracy'])
            else:
                daily_data[day]['accuracy'] = 0
        
        return daily_data
    
    def _compare_recent_vs_historical(self, daily_performance: Dict) -> Dict:
        """Compare recent performance vs historical average"""
        
        days = sorted(daily_performance.keys())
        if len(days) < 14:
            return {'insufficient_data': True}
        
        # Split into recent (last 7 days) and historical
        split_point = len(days) - 7
        historical_days = days[:split_point]
        recent_days = days[split_point:]
        
        historical_savings = np.mean([daily_performance[day]['savings'] for day in historical_days])
        recent_savings = np.mean([daily_performance[day]['savings'] for day in recent_days])
        
        historical_accuracy = np.mean([daily_performance[day]['accuracy'] for day in historical_days])
        recent_accuracy = np.mean([daily_performance[day]['accuracy'] for day in recent_days])
        
        return {
            'savings_change': (recent_savings - historical_savings) / historical_savings if historical_savings > 0 else 0,
            'accuracy_change': recent_accuracy - historical_accuracy,
            'recent_performance': 'better' if recent_savings > historical_savings and recent_accuracy > historical_accuracy else 'worse',
            'historical_avg_savings': historical_savings,
            'recent_avg_savings': recent_savings,
            'historical_avg_accuracy': historical_accuracy,
            'recent_avg_accuracy': recent_accuracy
        }
    
    def _get_alert_summary(self, cutoff_date: dt.datetime) -> Dict:
        """Get summary of recent alerts"""
        
        # This would retrieve alerts from database or log
        return {
            'total_alerts': 0,
            'alert_types': {},
            'recent_alerts': []
        }
    
    def _identify_coverage_gaps(self) -> List[Dict]:
        """Identify gaps in rule coverage"""
        
        recommendations = []
        
        # Check if certain conditions are not covered by rules
        if 'solar_storage_optimization' not in self.rule_effectiveness:
            recommendations.append({
                'type': 'missing_rule',
                'rule_name': 'solar_storage_optimization',
                'recommendation': 'Consider adding solar storage optimization rules',
                'priority': 'medium'
            })
        
        if 'peak_shaving' not in self.rule_effectiveness:
            recommendations.append({
                'type': 'missing_rule',
                'rule_name': 'peak_shaving',
                'recommendation': 'Consider adding peak shaving rules for cost optimization',
                'priority': 'high'
            })
        
        return recommendations
    
    def cleanup_old_data(self):
        """Clean up old monitoring data based on retention policy"""
        
        cutoff_date = dt.datetime.now() - dt.timedelta(days=self.retention_days)
        
        # Clean up in-memory data
        self.prediction_performance = [
            p for p in self.prediction_performance 
            if p['timestamp'] > cutoff_date
        ]
        
        self.optimization_performance = [
            o for o in self.optimization_performance 
            if o['timestamp'] > cutoff_date
        ]
        
        self.cost_savings_history = [
            s for s in self.cost_savings_history 
            if s['timestamp'] > cutoff_date
        ]
        
        # Clean up rule effectiveness for unused rules
        for rule_name in list(self.rule_effectiveness.keys()):
            if self.rule_effectiveness[rule_name].get('last_used', dt.datetime.min) < cutoff_date:
                del self.rule_effectiveness[rule_name]
        
        logging.info(f"Cleaned up monitoring data older than {self.retention_days} days")

    def get_monitoring_status(self) -> Dict:
        """Get current status of the monitoring system"""
        
        return {
            'enabled': self.monitoring_enabled,
            'retention_days': self.retention_days,
            'prediction_records': len(self.prediction_performance),
            'optimization_records': len(self.optimization_performance),
            'cost_records': len(self.cost_savings_history),
            'rules_tracked': len(self.rule_effectiveness),
            'cache_status': 'valid' if self._is_cache_valid() else 'expired',
            'last_cleanup': 'not_implemented',  # Would track actual cleanup times
            'alert_thresholds': self.alert_thresholds
        }