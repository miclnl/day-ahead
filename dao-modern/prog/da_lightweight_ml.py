"""
Lightweight ML module using only built-in Python and basic numpy/pandas.
Alternative to heavy ML dependencies for basic prediction tasks.
"""

import numpy as np
import pandas as pd
import pickle
import logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta


class SimpleLinearRegression:
    """Simple linear regression without scikit-learn"""
    
    def __init__(self):
        self.weights = None
        self.bias = None
        self.trained = False
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train linear regression model"""
        try:
            # Add bias column
            X_with_bias = np.column_stack([np.ones(len(X)), X])
            
            # Solve normal equation: (X'X)^-1 X'y
            XTX = np.dot(X_with_bias.T, X_with_bias)
            XTy = np.dot(X_with_bias.T, y)
            
            # Add small regularization for numerical stability
            XTX += np.eye(XTX.shape[0]) * 1e-6
            
            params = np.linalg.solve(XTX, XTy)
            self.bias = params[0]
            self.weights = params[1:]
            self.trained = True
            
        except Exception as e:
            logging.error(f"Linear regression training error: {e}")
            self.trained = False
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        if not self.trained:
            return np.zeros(len(X))
        
        return self.bias + np.dot(X, self.weights)


class MovingAveragePredictor:
    """Simple moving average predictor"""
    
    def __init__(self, window_size: int = 24):
        self.window_size = window_size
        self.history = []
    
    def fit(self, data: List[float]):
        """Store historical data"""
        self.history = data[-self.window_size*2:] if len(data) > self.window_size*2 else data
    
    def predict(self, steps: int = 1) -> List[float]:
        """Predict next values using moving average"""
        if len(self.history) < self.window_size:
            return [np.mean(self.history)] * steps if self.history else [0.0] * steps
        
        predictions = []
        recent_data = self.history[-self.window_size:]
        
        for _ in range(steps):
            pred = np.mean(recent_data)
            predictions.append(pred)
            # Update for next prediction
            recent_data = recent_data[1:] + [pred]
        
        return predictions


class SeasonalPredictor:
    """Simple seasonal pattern predictor"""
    
    def __init__(self, season_length: int = 24):
        self.season_length = season_length
        self.seasonal_pattern = None
        self.trend = None
    
    def fit(self, data: List[float]):
        """Learn seasonal pattern"""
        if len(data) < self.season_length * 2:
            self.seasonal_pattern = [np.mean(data)] * self.season_length
            self.trend = 0
            return
        
        # Extract seasonal pattern
        data_array = np.array(data)
        seasons = len(data) // self.season_length
        
        pattern = []
        for i in range(self.season_length):
            seasonal_values = [data_array[j * self.season_length + i] 
                             for j in range(seasons) 
                             if j * self.season_length + i < len(data)]
            pattern.append(np.mean(seasonal_values))
        
        self.seasonal_pattern = pattern
        
        # Simple trend calculation
        if seasons > 1:
            first_half = data[:len(data)//2]
            second_half = data[len(data)//2:]
            self.trend = (np.mean(second_half) - np.mean(first_half)) / (len(data) / 2)
        else:
            self.trend = 0
    
    def predict(self, steps: int = 1) -> List[float]:
        """Predict using seasonal pattern"""
        if not self.seasonal_pattern:
            return [0.0] * steps
        
        predictions = []
        for i in range(steps):
            seasonal_idx = i % self.season_length
            seasonal_value = self.seasonal_pattern[seasonal_idx]
            trend_value = self.trend * i
            predictions.append(max(0, seasonal_value + trend_value))
        
        return predictions


class LightweightMLPredictor:
    """Lightweight ML predictor combining multiple simple methods"""
    
    def __init__(self):
        self.linear_model = SimpleLinearRegression()
        self.moving_avg = MovingAveragePredictor()
        self.seasonal_model = SeasonalPredictor()
        self.ensemble_weights = [0.4, 0.3, 0.3]  # linear, moving_avg, seasonal
        self.trained = False
    
    def prepare_features(self, df: pd.DataFrame, target_col: str) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare simple features from DataFrame"""
        try:
            # Basic time-based features
            df['hour'] = pd.to_datetime(df.index).hour
            df['day_of_week'] = pd.to_datetime(df.index).dayofweek
            df['month'] = pd.to_datetime(df.index).month
            
            # Rolling statistics
            df['rolling_mean_6h'] = df[target_col].rolling(6, min_periods=1).mean()
            df['rolling_std_6h'] = df[target_col].rolling(6, min_periods=1).std().fillna(0)
            
            # Lag features
            df['lag_1h'] = df[target_col].shift(1)
            df['lag_24h'] = df[target_col].shift(24)
            
            feature_cols = ['hour', 'day_of_week', 'month', 'rolling_mean_6h', 'rolling_std_6h', 'lag_1h', 'lag_24h']
            
            # Drop NaN rows
            df_clean = df.dropna()
            
            X = df_clean[feature_cols].values
            y = df_clean[target_col].values
            
            return X, y
            
        except Exception as e:
            logging.error(f"Feature preparation error: {e}")
            return np.array([]), np.array([])
    
    def train(self, df: pd.DataFrame, target_col: str = 'consumption'):
        """Train all models"""
        try:
            # Prepare data
            X, y = self.prepare_features(df, target_col)
            
            if len(X) == 0:
                logging.warning("No data available for training")
                return False
            
            # Train linear model
            self.linear_model.fit(X, y)
            
            # Train moving average
            self.moving_avg.fit(y.tolist())
            
            # Train seasonal model
            self.seasonal_model.fit(y.tolist())
            
            self.trained = True
            logging.info("Lightweight ML models trained successfully")
            return True
            
        except Exception as e:
            logging.error(f"Training error: {e}")
            return False
    
    def predict(self, df: pd.DataFrame, steps: int = 24) -> List[float]:
        """Make ensemble predictions"""
        if not self.trained:
            logging.warning("Models not trained, using default predictions")
            return [1.0] * steps  # Default consumption
        
        try:
            predictions = []
            
            # Get predictions from each model
            if len(df) > 0:
                X_last, _ = self.prepare_features(df.tail(50), df.columns[0])  # Use any numeric column
                if len(X_last) > 0:
                    linear_pred = self.linear_model.predict(X_last[-1:])
                    linear_predictions = [max(0, linear_pred[0])] * steps
                else:
                    linear_predictions = [1.0] * steps
            else:
                linear_predictions = [1.0] * steps
            
            moving_avg_pred = self.moving_avg.predict(steps)
            seasonal_pred = self.seasonal_model.predict(steps)
            
            # Ensemble prediction
            for i in range(steps):
                ensemble_pred = (
                    self.ensemble_weights[0] * linear_predictions[i] +
                    self.ensemble_weights[1] * moving_avg_pred[i] +
                    self.ensemble_weights[2] * seasonal_pred[i]
                )
                predictions.append(max(0, ensemble_pred))
            
            return predictions
            
        except Exception as e:
            logging.error(f"Prediction error: {e}")
            return [1.0] * steps


# Example usage functions
def predict_consumption(historical_data: pd.DataFrame, hours_ahead: int = 24) -> List[float]:
    """Simple function to predict consumption"""
    predictor = LightweightMLPredictor()
    
    if predictor.train(historical_data, 'consumption'):
        return predictor.predict(historical_data, hours_ahead)
    else:
        # Fallback to simple average
        avg_consumption = historical_data['consumption'].mean()
        return [avg_consumption] * hours_ahead


def predict_solar_production(weather_data: pd.DataFrame, pv_capacity_kw: float) -> List[float]:
    """Simple solar production prediction"""
    try:
        if 'solar_radiation' in weather_data.columns:
            # Simple model: production = radiation * efficiency * capacity
            efficiency = 0.2  # 20% efficiency
            production = weather_data['solar_radiation'] * efficiency * pv_capacity_kw / 1000
            return production.fillna(0).tolist()
        else:
            # Fallback: simple day/night pattern
            hours = len(weather_data)
            production = []
            for i in range(hours):
                hour_of_day = (datetime.now().hour + i) % 24
                if 6 <= hour_of_day <= 18:  # Daylight hours
                    # Simple sine wave for solar production
                    solar_factor = np.sin((hour_of_day - 6) * np.pi / 12)
                    production.append(max(0, pv_capacity_kw * solar_factor))
                else:
                    production.append(0.0)
            return production
            
    except Exception as e:
        logging.error(f"Solar prediction error: {e}")
        return [0.0] * len(weather_data)