"""
Database Optimalisatie Module voor DAO

Deze module implementeert:
- Database connection pooling
- Query caching
- Performance monitoring
- Database indexing optimalisatie
"""

import logging
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
from functools import wraps
import hashlib
import json
import sqlite3
from contextlib import contextmanager

try:
    from sqlalchemy import create_engine, text, inspect
    from sqlalchemy.pool import QueuePool
    from sqlalchemy.orm import sessionmaker
    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False
    logging.warning("SQLAlchemy niet beschikbaar - database optimalisatie beperkt")


class DatabaseOptimizer:
    """Database optimalisatie en caching systeem"""

    def __init__(self, config, db_path: str = None):
        self.config = config
        self.db_path = db_path or "../data/dao.db"
        self.cache = {}
        self.cache_stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0
        }
        self.performance_metrics = {}
        self.connection_pool = None
        self.session_factory = None
        self.max_cache_size = 1000
        self.cache_ttl = 300  # 5 minuten
        self.lock = threading.RLock()

        # Initialiseer database optimalisatie
        self._initialize_database_optimization()

        logging.info("Database Optimizer geïnitialiseerd")

    def _initialize_database_optimization(self):
        """Initialiseer database optimalisatie features"""
        try:
            if SQLALCHEMY_AVAILABLE and self.db_path:
                # Maak SQLAlchemy engine met connection pooling
                self.connection_pool = create_engine(
                    f'sqlite:///{self.db_path}',
                    poolclass=QueuePool,
                    pool_size=10,
                    max_overflow=20,
                    pool_timeout=30,
                    pool_recycle=3600,
                    pool_pre_ping=True
                )

                # Maak session factory
                self.session_factory = sessionmaker(bind=self.connection_pool)

                # Optimaliseer database schema
                self._optimize_database_schema()

                logging.info("Database connection pooling geactiveerd")
            else:
                logging.warning("SQLAlchemy niet beschikbaar - connection pooling uitgeschakeld")

        except Exception as e:
            logging.error(f"Database optimalisatie initialisatie fout: {e}")

    def _optimize_database_schema(self):
        """Optimaliseer database schema met indexes en optimalisaties"""
        try:
            if not self.connection_pool:
                return

            inspector = inspect(self.connection_pool)

            # Controleer bestaande indexes
            existing_indexes = {}
            for table_name in inspector.get_table_names():
                existing_indexes[table_name] = [idx['name'] for idx in inspector.get_indexes(table_name)]

            # Maak ontbrekende indexes aan
            self._create_missing_indexes(existing_indexes)

            # Optimaliseer database instellingen
            self._optimize_database_settings()

            logging.info("Database schema optimalisatie voltooid")

        except Exception as e:
            logging.error(f"Database schema optimalisatie fout: {e}")

    def _create_missing_indexes(self, existing_indexes: Dict[str, List[str]]):
        """Maak ontbrekende indexes aan voor betere performance"""
        try:
            # Definieer gewenste indexes per tabel
            desired_indexes = {
                'values': [
                    ('idx_values_timestamp', 'timestamp'),
                    ('idx_values_entity_id', 'entity_id'),
                    ('idx_values_timestamp_entity', ['timestamp', 'entity_id'])
                ],
                'energy_balance': [
                    ('idx_energy_timestamp', 'timestamp'),
                    ('idx_energy_type', 'type')
                ],
                'prices': [
                    ('idx_prices_timestamp', 'timestamp'),
                    ('idx_prices_type', 'price_type')
                ]
            }

            with self.get_session() as session:
                for table_name, indexes in desired_indexes.items():
                    if table_name not in existing_indexes:
                        continue

                    for index_name, columns in indexes:
                        if index_name not in existing_indexes[table_name]:
                            try:
                                if isinstance(columns, list):
                                    # Composite index
                                    columns_str = ', '.join(columns)
                                else:
                                    # Single column index
                                    columns_str = columns

                                create_index_sql = f"CREATE INDEX IF NOT EXISTS {index_name} ON {table_name} ({columns_str})"
                                session.execute(text(create_index_sql))
                                logging.info(f"Index aangemaakt: {index_name} op {table_name}")

                            except Exception as e:
                                logging.debug(f"Kon index {index_name} niet aanmaken: {e}")

        except Exception as e:
            logging.error(f"Index creatie fout: {e}")

    def _optimize_database_settings(self):
        """Optimaliseer database instellingen voor betere performance"""
        try:
            with self.get_session() as session:
                # SQLite optimalisaties
                session.execute(text("PRAGMA journal_mode = WAL"))
                session.execute(text("PRAGMA synchronous = NORMAL"))
                session.execute(text("PRAGMA cache_size = 10000"))
                session.execute(text("PRAGMA temp_store = MEMORY"))
                session.execute(text("PRAGMA mmap_size = 268435456"))  # 256MB

                logging.info("Database instellingen geoptimaliseerd")

        except Exception as e:
            logging.error(f"Database instellingen optimalisatie fout: {e}")

    @contextmanager
    def get_session(self):
        """Context manager voor database sessies met connection pooling"""
        if not self.session_factory:
            # Fallback naar directe database connectie
            conn = sqlite3.connect(self.db_path)
            try:
                yield conn
            finally:
                conn.close()
            return

        session = self.session_factory()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            raise
        finally:
            session.close()

    def cache_query(self, query_key: str, result: Any, ttl: int = None) -> None:
        """Cache een query resultaat"""
        try:
            with self.lock:
                # Verwijder oude cache entries als cache te groot wordt
                if len(self.cache) >= self.max_cache_size:
                    self._evict_old_cache_entries()

                # Voeg nieuwe cache entry toe
                cache_ttl = ttl or self.cache_ttl
                self.cache[query_key] = {
                    'data': result,
                    'timestamp': time.time(),
                    'ttl': cache_ttl
                }

        except Exception as e:
            logging.debug(f"Cache error: {e}")

    def get_cached_query(self, query_key: str) -> Optional[Any]:
        """Haal gecachte query op"""
        try:
            with self.lock:
                if query_key in self.cache:
                    cache_entry = self.cache[query_key]

                    # Check of cache entry nog geldig is
                    if time.time() - cache_entry['timestamp'] < cache_entry['ttl']:
                        self.cache_stats['hits'] += 1
                        return cache_entry['data']
                    else:
                        # Verwijder verlopen cache entry
                        del self.cache[query_key]
                        self.cache_stats['evictions'] += 1

                self.cache_stats['misses'] += 1
                return None

        except Exception as e:
            logging.debug(f"Cache retrieval error: {e}")
            return None

    def _evict_old_cache_entries(self):
        """Verwijder oude cache entries om ruimte vrij te maken"""
        try:
            current_time = time.time()
            expired_keys = []

            for key, entry in self.cache.items():
                if current_time - entry['timestamp'] > entry['ttl']:
                    expired_keys.append(key)

            # Verwijder expired entries
            for key in expired_keys:
                del self.cache[key]
                self.cache_stats['evictions'] += 1

            # Als nog steeds te groot, verwijder oudste entries
            if len(self.cache) >= self.max_cache_size:
                sorted_entries = sorted(
                    self.cache.items(),
                    key=lambda x: x[1]['timestamp']
                )

                # Verwijder 20% van de oudste entries
                entries_to_remove = int(self.max_cache_size * 0.2)
                for key, _ in sorted_entries[:entries_to_remove]:
                    del self.cache[key]
                    self.cache_stats['evictions'] += 1

        except Exception as e:
            logging.debug(f"Cache eviction error: {e}")

    def clear_cache(self, pattern: str = None) -> None:
        """Leeg cache (optioneel met pattern matching)"""
        try:
            with self.lock:
                if pattern:
                    # Verwijder alleen entries die matchen met pattern
                    keys_to_remove = [key for key in self.cache.keys() if pattern in key]
                    for key in keys_to_remove:
                        del self.cache[key]
                    logging.info(f"Cache gecleared voor pattern: {pattern} ({len(keys_to_remove)} entries)")
                else:
                    # Leeg hele cache
                    cleared_count = len(self.cache)
                    self.cache.clear()
                    logging.info(f"Cache volledig gecleared ({cleared_count} entries)")

        except Exception as e:
            logging.error(f"Cache clear error: {e}")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Haal cache statistieken op"""
        try:
            with self.lock:
                total_requests = self.cache_stats['hits'] + self.cache_stats['misses']
                hit_rate = (self.cache_stats['hits'] / total_requests * 100) if total_requests > 0 else 0

                return {
                    'cache_size': len(self.cache),
                    'max_cache_size': self.max_cache_size,
                    'hits': self.cache_stats['hits'],
                    'misses': self.cache_stats['misses'],
                    'evictions': self.cache_stats['evictions'],
                    'hit_rate_percentage': round(hit_rate, 2),
                    'total_requests': total_requests
                }

        except Exception as e:
            logging.error(f"Cache stats error: {e}")
            return {}

    def monitor_query_performance(self, query_name: str):
        """Decorator voor het monitoren van query performance"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    execution_time = time.time() - start_time

                    # Sla performance metrics op
                    self._record_performance_metric(query_name, execution_time, True)

                    return result

                except Exception as e:
                    execution_time = time.time() - start_time
                    self._record_performance_metric(query_name, execution_time, False)
                    raise

            return wrapper
        return decorator

    def _record_performance_metric(self, query_name: str, execution_time: float, success: bool):
        """Record performance metric voor een query"""
        try:
            if query_name not in self.performance_metrics:
                self.performance_metrics[query_name] = {
                    'total_calls': 0,
                    'successful_calls': 0,
                    'failed_calls': 0,
                    'total_execution_time': 0.0,
                    'min_execution_time': float('inf'),
                    'max_execution_time': 0.0,
                    'last_call': None
                }

            metric = self.performance_metrics[query_name]
            metric['total_calls'] += 1
            metric['total_execution_time'] += execution_time
            metric['min_execution_time'] = min(metric['min_execution_time'], execution_time)
            metric['max_execution_time'] = max(metric['max_execution_time'], execution_time)
            metric['last_call'] = datetime.now().isoformat()

            if success:
                metric['successful_calls'] += 1
            else:
                metric['failed_calls'] += 1

        except Exception as e:
            logging.debug(f"Performance metric recording error: {e}")

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Haal performance metrics op"""
        try:
            metrics = {}
            for query_name, metric in self.performance_metrics.items():
                total_calls = metric['total_calls']
                if total_calls > 0:
                    avg_execution_time = metric['total_execution_time'] / total_calls
                    success_rate = (metric['successful_calls'] / total_calls) * 100

                    metrics[query_name] = {
                        'total_calls': total_calls,
                        'successful_calls': metric['successful_calls'],
                        'failed_calls': metric['failed_calls'],
                        'success_rate_percentage': round(success_rate, 2),
                        'avg_execution_time_ms': round(avg_execution_time * 1000, 2),
                        'min_execution_time_ms': round(metric['min_execution_time'] * 1000, 2),
                        'max_execution_time_ms': round(metric['max_execution_time'] * 1000, 2),
                        'last_call': metric['last_call']
                    }

            return metrics

        except Exception as e:
            logging.error(f"Performance metrics error: {e}")
            return {}

    def generate_query_hash(self, query: str, params: Dict[str, Any] = None) -> str:
        """Genereer unieke hash voor een query en parameters"""
        try:
            # Combineer query en parameters
            query_string = query
            if params:
                # Sorteer parameters voor consistente hashing
                sorted_params = json.dumps(params, sort_keys=True)
                query_string += sorted_params

            # Genereer hash
            return hashlib.md5(query_string.encode('utf-8')).hexdigest()

        except Exception as e:
            logging.debug(f"Query hash generation error: {e}")
            return str(hash(query_string))

    def optimize_database_maintenance(self):
        """Voer database onderhoud en optimalisatie uit"""
        try:
            logging.info("Database onderhoud gestart")

            with self.get_session() as session:
                # VACUUM database (reorganiseer en comprimeer)
                session.execute(text("VACUUM"))

                # Update database statistieken
                session.execute(text("ANALYZE"))

                # Optimaliseer database instellingen opnieuw
                self._optimize_database_settings()

                # Clear cache om ruimte vrij te maken
                self.clear_cache()

            logging.info("Database onderhoud voltooid")

        except Exception as e:
            logging.error(f"Database onderhoud fout: {e}")

    def get_database_info(self) -> Dict[str, Any]:
        """Haal database informatie op"""
        try:
            info = {
                'database_path': self.db_path,
                'connection_pooling': SQLALCHEMY_AVAILABLE and self.connection_pool is not None,
                'cache_enabled': True,
                'cache_stats': self.get_cache_stats(),
                'performance_metrics': self.get_performance_metrics()
            }

            # Voeg database schema informatie toe
            if self.connection_pool:
                try:
                    inspector = inspect(self.connection_pool)
                    info['tables'] = inspector.get_table_names()
                    info['total_tables'] = len(info['tables'])
                except Exception as e:
                    logging.debug(f"Kon database schema niet ophalen: {e}")
                    info['tables'] = []
                    info['total_tables'] = 0

            return info

        except Exception as e:
            logging.error(f"Database info error: {e}")
            return {'error': str(e)}


# Utility functies voor externe gebruik
def create_database_optimizer(config, db_path: str = None) -> DatabaseOptimizer:
    """Factory functie voor het maken van een DatabaseOptimizer instance"""
    return DatabaseOptimizer(config, db_path)


def cache_query_result(optimizer: DatabaseOptimizer, ttl: int = 300):
    """Decorator voor het cachen van functie resultaten"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Genereer cache key gebaseerd op functie naam en parameters
            cache_key = f"{func.__name__}:{hash(str(args) + str(sorted(kwargs.items())))}"

            # Probeer gecachte resultaat op te halen
            cached_result = optimizer.get_cached_query(cache_key)
            if cached_result is not None:
                return cached_result

            # Voer functie uit en cache resultaat
            result = func(*args, **kwargs)
            optimizer.cache_query(cache_key, result, ttl)

            return result
        return wrapper
    return decorator
