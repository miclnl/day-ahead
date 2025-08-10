"""
Modern Machine Learning module for Day Ahead Optimizer.
Implements advanced ML techniques for consumption prediction, pattern recognition, and optimization.
"""

import asyncio
import datetime
import logging
import numpy as np
import pandas as pd
import pickle
import os
from typing import Dict, List, Optional, Tuple, Any, Union
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor

# ML Libraries
import sklearn
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline

# XGBoost (optional import with fallback)
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
    logging.info("XGBoost available for advanced ML models")
except ImportError:
    XGBOOST_AVAILABLE = False
    logging.warning("XGBoost not available, using sklearn models only")

# TensorFlow Lite (optional imports with fallback)
try:
    import tflite_runtime.interpreter as tflite
    TFLITE_AVAILABLE = True
    logging.info("TensorFlow Lite runtime available for inference")
except ImportError:
    try:
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers
        TENSORFLOW_AVAILABLE = True
        TFLITE_AVAILABLE = False
        logging.info("Full TensorFlow available")
    except ImportError:
        TENSORFLOW_AVAILABLE = False
        TFLITE_AVAILABLE = False
        logging.warning("Neither TensorFlow nor TensorFlow Lite available, deep learning features disabled")

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    logging.warning("PyTorch not available, neural network features limited")

from dao.prog.da_config import Config
from dao.prog.db_manager import DBmanagerObj


class MLModelBase(ABC):
    """Abstract base class for ML prediction models"""
    
    def __init__(self, model_name: str, config: Dict[str, Any]):
        self.model_name = model_name
        self.config = config
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_names = []
        self.model_path = config.get('model_path', '../data/models/')
        
        # Create model directory if it doesn't exist
        os.makedirs(self.model_path, exist_ok=True)
    
    @abstractmethod
    async def train(self, training_data: pd.DataFrame, target_column: str) -> bool:
        """Train the model with historical data"""
        pass
    
    @abstractmethod
    async def predict(self, input_data: pd.DataFrame) -> np.ndarray:
        """Make predictions on new data"""
        pass
    
    @abstractmethod
    async def evaluate(self, test_data: pd.DataFrame, target_column: str) -> Dict[str, float]:
        """Evaluate model performance"""
        pass
    
    def save_model(self) -> bool:
        """Save trained model to disk"""
        try:
            model_file = os.path.join(self.model_path, f"{self.model_name}.pkl")
            with open(model_file, 'wb') as f:
                pickle.dump({
                    'model': self.model,
                    'scaler': self.scaler,
                    'feature_names': self.feature_names,
                    'is_trained': self.is_trained,
                    'config': self.config
                }, f)
            logging.info(f"Model saved: {model_file}")
            return True
        except Exception as e:
            logging.error(f"Error saving model: {e}")
            return False
    
    def load_model(self) -> bool:
        """Load trained model from disk"""
        try:
            model_file = os.path.join(self.model_path, f"{self.model_name}.pkl")
            if os.path.exists(model_file):
                with open(model_file, 'rb') as f:
                    data = pickle.load(f)
                    self.model = data['model']
                    self.scaler = data['scaler']
                    self.feature_names = data['feature_names']
                    self.is_trained = data['is_trained']
                logging.info(f"Model loaded: {model_file}")
                return True
            else:
                logging.info(f"No saved model found: {model_file}")
                return False
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            return False


class ConsumptionPredictor(MLModelBase):
    """Advanced consumption prediction using multiple ML algorithms"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("consumption_predictor", config)
        self.algorithm = config.get('algorithm', 'ensemble')  # ensemble, xgboost, neural_network
        self.lookback_hours = config.get('lookback_hours', 168)  # 1 week
        self.prediction_horizon = config.get('prediction_horizon', 48)  # 2 days
    
    async def train(self, training_data: pd.DataFrame, target_column: str = 'consumption') -> bool:
        """Train consumption prediction model"""
        logging.info(f"Training consumption predictor with {len(training_data)} samples")
        
        try:
            # Prepare features
            features_df = await self._prepare_features(training_data)
            
            if features_df.empty:
                logging.error("No features could be prepared")
                return False
            
            # Prepare target variable
            y = features_df[target_column].values
            X = features_df.drop(columns=[target_column]).values
            self.feature_names = features_df.drop(columns=[target_column]).columns.tolist()
            
            # Split data (time series aware)
            train_size = int(0.8 * len(X))
            X_train, X_test = X[:train_size], X[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train model based on algorithm choice
            if self.algorithm == 'ensemble':
                self.model = await self._train_ensemble(X_train_scaled, y_train)
            elif self.algorithm == 'xgboost':
                if XGBOOST_AVAILABLE:
                    self.model = await self._train_xgboost(X_train_scaled, y_train)
                else:
                    logging.warning("XGBoost not available, using Gradient Boosting instead")
                    self.model = await self._train_gradient_boosting(X_train_scaled, y_train)
            elif self.algorithm == 'neural_network' and TENSORFLOW_AVAILABLE:
                self.model = await self._train_neural_network(X_train_scaled, y_train, X_test_scaled, y_test)
            else:
                logging.warning(f"Algorithm {self.algorithm} not available, using ensemble")
                self.model = await self._train_ensemble(X_train_scaled, y_train)
            
            # Evaluate model
            test_predictions = await self.predict(features_df.iloc[train_size:].drop(columns=[target_column]))
            test_score = r2_score(y_test, test_predictions)
            rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
            
            logging.info(f"Model training completed. R² score: {test_score:.4f}, RMSE: {rmse:.4f}")
            
            self.is_trained = True
            self.save_model()
            return True
            
        except Exception as e:
            logging.error(f"Error training consumption predictor: {e}")
            return False
    
    async def _train_ensemble(self, X_train: np.ndarray, y_train: np.ndarray) -> sklearn.ensemble.VotingRegressor:
        """Train ensemble of multiple algorithms"""
        from sklearn.ensemble import VotingRegressor
        
        # Individual models
        models = [
            ('rf', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)),
            ('gb', GradientBoostingRegressor(n_estimators=100, random_state=42)),
            ('ridge', Ridge(alpha=1.0)),
            ('elastic', ElasticNet(alpha=0.1, random_state=42))
        ]
        
        # Create ensemble
        ensemble = VotingRegressor(models)
        
        # Train with cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        
        # Use async to avoid blocking
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            ensemble = await loop.run_in_executor(executor, ensemble.fit, X_train, y_train)
        
        return ensemble
    
    async def _train_gradient_boosting(self, X_train: np.ndarray, y_train: np.ndarray) -> GradientBoostingRegressor:
        """Train Gradient Boosting model (sklearn alternative to XGBoost)"""
        
        # Gradient Boosting with similar parameters to XGBoost
        model = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42
        )
        
        # Train asynchronously
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            model = await loop.run_in_executor(executor, model.fit, X_train, y_train)
        
        return model
    
    async def _train_xgboost(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train XGBoost model (only if available)"""
        if not XGBOOST_AVAILABLE:
            raise RuntimeError("XGBoost not available")
            
        # XGBoost with optimal parameters for time series
        model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )
        
        # Train asynchronously
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            model = await loop.run_in_executor(executor, model.fit, X_train, y_train)
        
        return model
    
    async def _train_neural_network(self, X_train: np.ndarray, y_train: np.ndarray, 
                                   X_test: np.ndarray, y_test: np.ndarray) -> tf.keras.Model:
        """Train neural network model"""
        if not TENSORFLOW_AVAILABLE:
            raise RuntimeError("TensorFlow not available")
        
        # Define architecture
        model = keras.Sequential([
            layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(), 
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu'),
            layers.Dense(1)
        ])
        
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        # Callbacks for better training
        callbacks = [
            keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5)
        ]
        
        # Train asynchronously
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            await loop.run_in_executor(
                executor,
                model.fit,
                X_train, y_train,
                epochs=100,
                batch_size=32,
                validation_data=(X_test, y_test),
                callbacks=callbacks,
                verbose=0
            )
        
        return model
    
    async def predict(self, input_data: pd.DataFrame) -> np.ndarray:
        """Make consumption predictions"""
        if not self.is_trained:
            if not self.load_model():
                raise RuntimeError("Model not trained and no saved model found")
        
        try:
            # Prepare features
            features_df = await self._prepare_features(input_data, for_prediction=True)
            
            if features_df.empty:
                logging.error("No features could be prepared for prediction")
                return np.array([])
            
            # Scale features
            X_scaled = self.scaler.transform(features_df[self.feature_names].values)
            
            # Make predictions
            if isinstance(self.model, tf.keras.Model):
                predictions = self.model.predict(X_scaled, verbose=0).flatten()
            else:
                predictions = self.model.predict(X_scaled)
            
            return predictions
            
        except Exception as e:
            logging.error(f"Error making predictions: {e}")
            return np.array([])
    
    async def evaluate(self, test_data: pd.DataFrame, target_column: str = 'consumption') -> Dict[str, float]:
        """Evaluate model performance"""
        try:
            features_df = await self._prepare_features(test_data)
            y_true = features_df[target_column].values
            y_pred = await self.predict(features_df.drop(columns=[target_column]))
            
            return {
                'r2_score': r2_score(y_true, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
                'mae': mean_absolute_error(y_true, y_pred),
                'mape': np.mean(np.abs((y_true - y_pred) / y_true)) * 100
            }
            
        except Exception as e:
            logging.error(f"Error evaluating model: {e}")
            return {}
    
    async def _prepare_features(self, data: pd.DataFrame, for_prediction: bool = False) -> pd.DataFrame:
        """Prepare features for training or prediction"""
        try:
            # Ensure datetime index
            if 'tijd' in data.columns:
                data['tijd'] = pd.to_datetime(data['tijd'])
                data = data.set_index('tijd')
            
            # Time-based features
            features = pd.DataFrame(index=data.index)
            
            # Temporal features
            features['hour'] = data.index.hour
            features['day_of_week'] = data.index.dayofweek
            features['month'] = data.index.month
            features['is_weekend'] = (data.index.dayofweek >= 5).astype(int)
            features['is_workday'] = (~features['is_weekend'].astype(bool)).astype(int)
            
            # Cyclical encoding for temporal features
            features['hour_sin'] = np.sin(2 * np.pi * features['hour'] / 24)
            features['hour_cos'] = np.cos(2 * np.pi * features['hour'] / 24)
            features['day_sin'] = np.sin(2 * np.pi * features['day_of_week'] / 7)
            features['day_cos'] = np.cos(2 * np.pi * features['day_of_week'] / 7)
            features['month_sin'] = np.sin(2 * np.pi * features['month'] / 12)
            features['month_cos'] = np.cos(2 * np.pi * features['month'] / 12)
            
            # Weather features
            if 'temp' in data.columns:
                features['temperature'] = data['temp']
                features['temp_squared'] = data['temp'] ** 2  # Non-linear temperature effects
            
            if 'gr' in data.columns:
                features['global_radiation'] = data['gr']
                features['is_sunny'] = (data['gr'] > 200).astype(int)
            
            # Price features
            if 'da_price' in data.columns:
                features['price'] = data['da_price']
                features['price_rank'] = data['da_price'].rank(pct=True)  # Relative price level
                
                # Price change features
                features['price_diff'] = data['da_price'].diff()
                features['price_ma_3h'] = data['da_price'].rolling(window=3).mean()
                features['price_ma_24h'] = data['da_price'].rolling(window=24).mean()
            
            # Consumption features (for training)
            if 'consumption' in data.columns:
                # Lagged consumption features
                for lag in [1, 2, 3, 24, 48, 168]:  # 1h, 2h, 3h, 1d, 2d, 1w
                    features[f'consumption_lag_{lag}h'] = data['consumption'].shift(lag)
                
                # Rolling statistics
                features['consumption_ma_3h'] = data['consumption'].rolling(window=3).mean()
                features['consumption_ma_24h'] = data['consumption'].rolling(window=24).mean()
                features['consumption_std_24h'] = data['consumption'].rolling(window=24).std()
                
                # Add target variable
                if not for_prediction:
                    features['consumption'] = data['consumption']
            
            # Production features
            if 'production' in data.columns:
                features['production'] = data['production']
                features['net_consumption'] = features.get('consumption', 0) - data['production']
            
            # Special day indicators
            features['is_holiday'] = self._is_holiday(data.index)
            features['is_school_holiday'] = self._is_school_holiday(data.index)
            
            # Remove rows with NaN values (caused by lags and rolling windows)
            features = features.dropna()
            
            return features
            
        except Exception as e:
            logging.error(f"Error preparing features: {e}")
            return pd.DataFrame()
    
    def _is_holiday(self, dates: pd.DatetimeIndex) -> np.ndarray:
        """Detect Dutch holidays (simplified)"""
        # This would ideally use a holiday library
        # For now, simple approximation
        holidays = []
        for date in dates:
            is_holiday = (
                (date.month == 1 and date.day == 1) or  # New Year
                (date.month == 4 and date.day == 27) or  # King's Day
                (date.month == 5 and date.day == 5) or   # Liberation Day
                (date.month == 12 and date.day == 25) or # Christmas
                (date.month == 12 and date.day == 26)    # Boxing Day
            )
            holidays.append(int(is_holiday))
        return np.array(holidays)
    
    def _is_school_holiday(self, dates: pd.DatetimeIndex) -> np.ndarray:
        """Detect school holidays (simplified)"""
        # Simplified school holiday detection
        school_holidays = []
        for date in dates:
            is_school_holiday = (
                (date.month in [7, 8]) or  # Summer holidays
                (date.month == 12 and date.day >= 20) or  # Christmas holidays
                (date.month == 1 and date.day <= 7) or    # Christmas holidays
                (date.month == 2 and 15 <= date.day <= 21) or  # Spring break
                (date.month == 4 and 20 <= date.day <= 28) or  # May break
                (date.month == 10 and 15 <= date.day <= 23)    # Fall break
            )
            school_holidays.append(int(is_school_holiday))
        return np.array(school_holidays)


class PatternRecognizer(MLModelBase):
    """Pattern recognition for energy usage behaviors"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("pattern_recognizer", config)
        self.n_clusters = config.get('n_clusters', 5)
        self.pattern_length = config.get('pattern_length', 24)  # 24 hours
        
    async def train(self, training_data: pd.DataFrame, target_column: str = 'consumption') -> bool:
        """Train pattern recognition model"""
        logging.info("Training pattern recognition model")
        
        try:
            from sklearn.cluster import KMeans
            from sklearn.decomposition import PCA
            
            # Prepare patterns (daily consumption profiles)
            patterns = await self._extract_daily_patterns(training_data, target_column)
            
            if patterns.empty:
                logging.error("No patterns could be extracted")
                return False
            
            # Normalize patterns
            pattern_data = self.scaler.fit_transform(patterns.values)
            
            # Apply PCA for dimensionality reduction
            self.pca = PCA(n_components=min(10, pattern_data.shape[1]))
            pattern_data_pca = self.pca.fit_transform(pattern_data)
            
            # Cluster patterns
            self.model = KMeans(n_clusters=self.n_clusters, random_state=42)
            cluster_labels = self.model.fit_predict(pattern_data_pca)
            
            # Store pattern information
            self.pattern_centroids = self.model.cluster_centers_
            self.pattern_labels = cluster_labels
            
            logging.info(f"Identified {self.n_clusters} consumption patterns")
            self.is_trained = True
            self.save_model()
            return True
            
        except Exception as e:
            logging.error(f"Error training pattern recognizer: {e}")
            return False
    
    async def predict(self, input_data: pd.DataFrame) -> np.ndarray:
        """Predict consumption pattern for new data"""
        if not self.is_trained:
            if not self.load_model():
                raise RuntimeError("Pattern recognizer not trained")
        
        try:
            patterns = await self._extract_daily_patterns(input_data, 'consumption')
            if patterns.empty:
                return np.array([])
            
            # Transform and predict
            pattern_data = self.scaler.transform(patterns.values)
            pattern_data_pca = self.pca.transform(pattern_data)
            cluster_predictions = self.model.predict(pattern_data_pca)
            
            return cluster_predictions
            
        except Exception as e:
            logging.error(f"Error predicting patterns: {e}")
            return np.array([])
    
    async def evaluate(self, test_data: pd.DataFrame, target_column: str = 'consumption') -> Dict[str, float]:
        """Evaluate pattern recognition quality"""
        try:
            from sklearn.metrics import silhouette_score
            
            patterns = await self._extract_daily_patterns(test_data, target_column)
            pattern_data = self.scaler.transform(patterns.values)
            pattern_data_pca = self.pca.transform(pattern_data)
            
            cluster_labels = self.model.predict(pattern_data_pca)
            silhouette = silhouette_score(pattern_data_pca, cluster_labels)
            
            return {
                'silhouette_score': silhouette,
                'n_patterns': len(patterns),
                'inertia': self.model.inertia_
            }
            
        except Exception as e:
            logging.error(f"Error evaluating pattern recognizer: {e}")
            return {}
    
    async def _extract_daily_patterns(self, data: pd.DataFrame, target_column: str) -> pd.DataFrame:
        """Extract daily consumption patterns"""
        try:
            # Ensure datetime index
            if 'tijd' in data.columns:
                data['tijd'] = pd.to_datetime(data['tijd'])
                data = data.set_index('tijd')
            
            # Group by date and extract daily profiles
            daily_patterns = []
            
            for date, group in data.groupby(data.index.date):
                if len(group) >= self.pattern_length:  # Full day
                    # Resample to hourly if needed
                    hourly_data = group[target_column].resample('H').mean()
                    
                    if len(hourly_data) >= 24:
                        # Take first 24 hours as pattern
                        pattern = hourly_data.iloc[:24].values
                        daily_patterns.append(pattern)
            
            if daily_patterns:
                patterns_df = pd.DataFrame(daily_patterns)
                patterns_df.columns = [f'hour_{i:02d}' for i in range(24)]
                return patterns_df
            else:
                return pd.DataFrame()
                
        except Exception as e:
            logging.error(f"Error extracting daily patterns: {e}")
            return pd.DataFrame()


class AdaptiveBaseloadCalculator:
    """Adaptive baseload calculation using ML techniques"""
    
    def __init__(self, config: Config, db_da: DBmanagerObj):
        self.config = config
        self.db_da = db_da
        self.learning_rate = 0.1
        self.forgetting_factor = 0.95
        self.min_samples = 168  # 1 week minimum
        
    async def calculate_adaptive_baseload(self, current_time: datetime.datetime) -> List[float]:
        """Calculate adaptive baseload using recent consumption patterns"""
        logging.info("Calculating adaptive baseload")
        
        try:
            # Get historical consumption data
            end_time = current_time
            start_time = current_time - datetime.timedelta(days=28)  # 4 weeks
            
            consumption_data = self.db_da.get_column_data(
                'values', 'cons', start_time, end_time
            )
            
            if consumption_data.empty or len(consumption_data) < self.min_samples:
                logging.warning("Insufficient data for adaptive baseload, using configured values")
                return self.config.get(['baseload'], [0.5] * 24)
            
            # Convert to hourly DataFrame
            consumption_data['time'] = pd.to_datetime(consumption_data['time'], unit='s')
            consumption_data = consumption_data.set_index('time')
            consumption_data['consumption'] = consumption_data['value']
            
            # Resample to hourly
            hourly_consumption = consumption_data['consumption'].resample('H').mean()
            
            # Calculate baseload by hour using multiple methods
            baseload_methods = await self._calculate_multiple_baseload_methods(hourly_consumption)
            
            # Combine methods with adaptive weights
            final_baseload = await self._combine_baseload_methods(baseload_methods, hourly_consumption)
            
            logging.info(f"Adaptive baseload calculated: avg={np.mean(final_baseload):.3f} kW")
            return final_baseload.tolist()
            
        except Exception as e:
            logging.error(f"Error calculating adaptive baseload: {e}")
            return self.config.get(['baseload'], [0.5] * 24)
    
    async def _calculate_multiple_baseload_methods(self, consumption_data: pd.Series) -> Dict[str, np.ndarray]:
        """Calculate baseload using multiple methods"""
        methods = {}
        
        try:
            # Method 1: Rolling minimum with seasonal adjustment
            daily_consumption = consumption_data.groupby([consumption_data.index.hour, 
                                                        consumption_data.index.dayofweek]).median()
            
            workday_baseload = []
            weekend_baseload = []
            
            for hour in range(24):
                # Workday baseload (Mon-Fri)
                workday_values = [daily_consumption.get((hour, day), 0) for day in range(5)]
                workday_baseload.append(np.median([v for v in workday_values if v > 0]))
                
                # Weekend baseload (Sat-Sun)
                weekend_values = [daily_consumption.get((hour, day), 0) for day in range(5, 7)]
                weekend_baseload.append(np.median([v for v in weekend_values if v > 0]))
            
            # Use current day type
            now = datetime.datetime.now()
            if now.weekday() < 5:  # Workday
                methods['seasonal'] = np.array(workday_baseload)
            else:  # Weekend
                methods['seasonal'] = np.array(weekend_baseload)
            
            # Method 2: Statistical percentile approach
            hourly_percentiles = consumption_data.groupby(consumption_data.index.hour).quantile(0.1)
            methods['percentile'] = hourly_percentiles.reindex(range(24)).fillna(method='bfill').values
            
            # Method 3: ML-based minimum consumption prediction
            if len(consumption_data) > 168:  # 1 week
                methods['ml_based'] = await self._ml_baseload_prediction(consumption_data)
            
            # Method 4: Weather-adjusted baseload
            methods['weather_adjusted'] = await self._weather_adjusted_baseload(consumption_data)
            
        except Exception as e:
            logging.error(f"Error in baseload method calculation: {e}")
            # Fallback method
            methods['fallback'] = np.full(24, consumption_data.quantile(0.1))
        
        return methods
    
    async def _ml_baseload_prediction(self, consumption_data: pd.Series) -> np.ndarray:
        """Use ML to predict minimum consumption patterns"""
        try:
            from sklearn.ensemble import RandomForestRegressor
            
            # Create features
            df = pd.DataFrame({'consumption': consumption_data})
            df['hour'] = df.index.hour
            df['day_of_week'] = df.index.dayofweek
            df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
            
            # Target: rolling minimum consumption
            df['min_consumption'] = consumption_data.rolling(window=24, min_periods=1).min()
            
            # Remove NaN values
            df = df.dropna()
            
            if len(df) < 100:
                return np.full(24, consumption_data.quantile(0.1))
            
            # Prepare features and target
            feature_cols = ['hour', 'day_of_week', 'is_weekend']
            X = df[feature_cols].values
            y = df['min_consumption'].values
            
            # Train model
            model = RandomForestRegressor(n_estimators=50, random_state=42)
            model.fit(X, y)
            
            # Predict for each hour
            baseload_predictions = []
            current_day = datetime.datetime.now()
            is_weekend = 1 if current_day.weekday() >= 5 else 0
            
            for hour in range(24):
                features = np.array([[hour, current_day.weekday(), is_weekend]])
                prediction = model.predict(features)[0]
                baseload_predictions.append(max(prediction, 0.05))  # Minimum 50W
            
            return np.array(baseload_predictions)
            
        except Exception as e:
            logging.error(f"Error in ML baseload prediction: {e}")
            return np.full(24, consumption_data.quantile(0.1))
    
    async def _weather_adjusted_baseload(self, consumption_data: pd.Series) -> np.ndarray:
        """Calculate weather-adjusted baseload"""
        try:
            # Get temperature data for the same period
            temp_data = self.db_da.get_column_data(
                'values', 'temp', 
                consumption_data.index[0], 
                consumption_data.index[-1]
            )
            
            if temp_data.empty:
                return np.full(24, consumption_data.quantile(0.1))
            
            # Convert temperature data
            temp_data['time'] = pd.to_datetime(temp_data['time'], unit='s')
            temp_data = temp_data.set_index('time')
            temp_hourly = temp_data['value'].resample('H').mean()
            
            # Align with consumption data
            aligned_data = pd.DataFrame({
                'consumption': consumption_data,
                'temperature': temp_hourly
            }).dropna()
            
            if len(aligned_data) < 100:
                return np.full(24, consumption_data.quantile(0.1))
            
            # Calculate temperature-adjusted baseload
            baseload_by_hour = []
            
            for hour in range(24):
                hour_data = aligned_data[aligned_data.index.hour == hour]
                
                if len(hour_data) > 10:
                    # Find minimum consumption at comfortable temperatures (18-22°C)
                    comfortable_data = hour_data[
                        (hour_data['temperature'] >= 18) & 
                        (hour_data['temperature'] <= 22)
                    ]
                    
                    if len(comfortable_data) > 5:
                        baseload = comfortable_data['consumption'].quantile(0.1)
                    else:
                        baseload = hour_data['consumption'].quantile(0.1)
                    
                    baseload_by_hour.append(max(baseload, 0.05))
                else:
                    baseload_by_hour.append(0.5)  # Default 500W
            
            return np.array(baseload_by_hour)
            
        except Exception as e:
            logging.error(f"Error in weather-adjusted baseload: {e}")
            return np.full(24, consumption_data.quantile(0.1))
    
    async def _combine_baseload_methods(self, methods: Dict[str, np.ndarray], 
                                      consumption_data: pd.Series) -> np.ndarray:
        """Combine multiple baseload methods with adaptive weights"""
        try:
            if not methods:
                return np.full(24, consumption_data.quantile(0.1))
            
            # Calculate weights based on recent accuracy
            weights = await self._calculate_method_weights(methods, consumption_data)
            
            # Combine methods
            combined_baseload = np.zeros(24)
            total_weight = sum(weights.values())
            
            for method_name, baseload_values in methods.items():
                if method_name in weights and len(baseload_values) == 24:
                    weight = weights[method_name] / total_weight
                    combined_baseload += weight * baseload_values
            
            # Ensure reasonable bounds
            combined_baseload = np.clip(combined_baseload, 0.05, 2.0)  # 50W to 2kW
            
            return combined_baseload
            
        except Exception as e:
            logging.error(f"Error combining baseload methods: {e}")
            return np.full(24, consumption_data.quantile(0.1))
    
    async def _calculate_method_weights(self, methods: Dict[str, np.ndarray], 
                                      consumption_data: pd.Series) -> Dict[str, float]:
        """Calculate adaptive weights for different baseload methods"""
        weights = {}
        
        try:
            # Test each method against recent data
            recent_data = consumption_data.tail(168)  # Last week
            
            for method_name, baseload_values in methods.items():
                if len(baseload_values) != 24:
                    weights[method_name] = 0.1
                    continue
                
                # Calculate accuracy
                errors = []
                for hour in range(24):
                    hour_data = recent_data[recent_data.index.hour == hour]
                    if len(hour_data) > 0:
                        actual_min = hour_data.min()
                        predicted_min = baseload_values[hour]
                        error = abs(actual_min - predicted_min) / max(actual_min, 0.1)
                        errors.append(error)
                
                if errors:
                    avg_error = np.mean(errors)
                    # Convert error to weight (lower error = higher weight)
                    weights[method_name] = 1.0 / (1.0 + avg_error)
                else:
                    weights[method_name] = 0.5
            
            # Ensure at least some minimum weights
            min_weight = 0.1
            for method_name in weights:
                weights[method_name] = max(weights[method_name], min_weight)
            
        except Exception as e:
            logging.error(f"Error calculating method weights: {e}")
            # Equal weights fallback
            weights = {method: 1.0 for method in methods.keys()}
        
        return weights


class MLPredictionManager:
    """Main ML prediction manager coordinating all ML models"""
    
    def __init__(self, config: Config, db_da: DBmanagerObj):
        self.config = config
        self.db_da = db_da
        self.ml_config = config.get(['machine_learning'], {})
        self.enabled = self.ml_config.get('enabled', True)
        
        # Initialize models
        self.consumption_predictor = ConsumptionPredictor(
            self.ml_config.get('consumption_predictor', {})
        )
        
        self.pattern_recognizer = PatternRecognizer(
            self.ml_config.get('pattern_recognizer', {})
        )
        
        self.adaptive_baseload = AdaptiveBaseloadCalculator(config, db_da)
        
        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="ML-Worker")
    
    async def train_all_models(self) -> bool:
        """Train all ML models with historical data"""
        if not self.enabled:
            logging.info("ML models disabled in configuration")
            return False
        
        logging.info("Training all ML models")
        
        try:
            # Get training data
            training_data = await self._get_training_data()
            
            if training_data.empty:
                logging.error("No training data available")
                return False
            
            # Train models concurrently
            tasks = [
                self.consumption_predictor.train(training_data, 'consumption'),
                self.pattern_recognizer.train(training_data, 'consumption')
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            success_count = sum(1 for result in results if result is True)
            logging.info(f"ML model training completed: {success_count}/{len(tasks)} models successful")
            
            return success_count > 0
            
        except Exception as e:
            logging.error(f"Error training ML models: {e}")
            return False
    
    async def predict_consumption(self, forecast_data: pd.DataFrame) -> Optional[np.ndarray]:
        """Predict consumption using trained models"""
        try:
            predictions = await self.consumption_predictor.predict(forecast_data)
            
            if len(predictions) > 0:
                logging.info(f"Generated {len(predictions)} consumption predictions")
                return predictions
            else:
                logging.warning("No consumption predictions generated")
                return None
                
        except Exception as e:
            logging.error(f"Error predicting consumption: {e}")
            return None
    
    async def get_adaptive_baseload(self) -> List[float]:
        """Get adaptive baseload calculation"""
        try:
            baseload = await self.adaptive_baseload.calculate_adaptive_baseload(
                datetime.datetime.now()
            )
            logging.info(f"Adaptive baseload calculated: {[f'{b:.3f}' for b in baseload]}")
            return baseload
            
        except Exception as e:
            logging.error(f"Error calculating adaptive baseload: {e}")
            return self.config.get(['baseload'], [0.5] * 24)
    
    async def recognize_patterns(self, consumption_data: pd.DataFrame) -> Optional[np.ndarray]:
        """Recognize consumption patterns"""
        try:
            patterns = await self.pattern_recognizer.predict(consumption_data)
            
            if len(patterns) > 0:
                logging.info(f"Identified patterns: {patterns}")
                return patterns
            else:
                return None
                
        except Exception as e:
            logging.error(f"Error recognizing patterns: {e}")
            return None
    
    async def _get_training_data(self) -> pd.DataFrame:
        """Get historical data for training ML models"""
        try:
            # Get last 3 months of data for training
            end_time = datetime.datetime.now()
            start_time = end_time - datetime.timedelta(days=90)
            
            # Fetch consumption data
            consumption_data = self.db_da.get_column_data('values', 'cons', start_time, end_time)
            production_data = self.db_da.get_column_data('values', 'prod', start_time, end_time)
            temp_data = self.db_da.get_column_data('values', 'temp', start_time, end_time)
            price_data = self.db_da.get_column_data('values', 'da', start_time, end_time)
            
            # Combine all data
            all_data = []
            
            for data, column_name in [(consumption_data, 'consumption'), 
                                    (production_data, 'production'),
                                    (temp_data, 'temp'), 
                                    (price_data, 'da_price')]:
                if not data.empty:
                    data['time'] = pd.to_datetime(data['time'], unit='s')
                    data = data.rename(columns={'value': column_name})
                    data = data[['time', column_name]]
                    all_data.append(data)
            
            if not all_data:
                return pd.DataFrame()
            
            # Merge all data on time
            combined_data = all_data[0]
            for data in all_data[1:]:
                combined_data = pd.merge(combined_data, data, on='time', how='outer')
            
            # Set time as index and sort
            combined_data = combined_data.set_index('time').sort_index()
            
            # Forward fill missing values (limited)
            combined_data = combined_data.fillna(method='ffill', limit=2)
            
            # Drop remaining NaN values
            combined_data = combined_data.dropna()
            
            logging.info(f"Prepared {len(combined_data)} training samples")
            return combined_data
            
        except Exception as e:
            logging.error(f"Error getting training data: {e}")
            return pd.DataFrame()
    
    def __del__(self):
        """Cleanup executor on destruction"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)