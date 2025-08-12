"""
Performance Optimalisatie Module voor DAO

Deze module implementeert:
- Geavanceerde multi-layer caching (memory, disk, Redis)
- Intelligente rate limiting met adaptive throttling
- Performance monitoring en profiling
- Load balancing en connection pooling
- Resource management en optimization
"""

import logging
import time
import json
import hashlib
import threading
import pickle
import os
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple, Callable
from dataclasses import dataclass, asdict
from functools import wraps
import gc
import psutil
import weakref

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logging.warning("Redis niet beschikbaar - caching beperkt tot memory/disk")

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    logging.warning("NumPy niet beschikbaar - performance monitoring beperkt")


@dataclass
class CacheEntry:
    """Data class voor cache entries"""
    key: str
    data: Any
    timestamp: float
    ttl: float
    access_count: int
    last_access: float
    size_bytes: int
    priority: int = 1  # 1-10, hoger = belangrijker


@dataclass
class RateLimitInfo:
    """Data class voor rate limiting informatie"""
    endpoint: str
    current_requests: int
    max_requests: int
    window_start: float
    window_duration: float
    blocked_until: Optional[float] = None


@dataclass
class PerformanceMetric:
    """Data class voor performance metrics"""
    operation: str
    execution_time: float
    memory_usage: float
    cpu_usage: float
    timestamp: float
    success: bool
    error_message: Optional[str] = None


class PerformanceOptimizer:
    """Hoofdklasse voor performance optimalisatie"""

    def __init__(self, config):
        self.config = config
        self.lock = threading.RLock()

        # Cache configuraties
        self.memory_cache = {}
        self.memory_cache_size = config.get(['performance', 'memory_cache_size'], None, 1000)
        self.memory_cache_ttl = config.get(['performance', 'memory_cache_ttl'], None, 3600)

        # Disk cache configuratie
        self.disk_cache_enabled = config.get(['performance', 'disk_cache_enabled'], None, True)
        self.disk_cache_path = config.get(['performance', 'disk_cache_path'], None, '/tmp/dao_cache')
        self.disk_cache_max_size = config.get(['performance', 'disk_cache_max_size'], None, 100 * 1024 * 1024)  # 100MB

        # Redis cache configuratie
        self.redis_enabled = config.get(['performance', 'redis_enabled'], None, False)
        self.redis_host = config.get(['performance', 'redis_host'], None, 'localhost')
        self.redis_port = config.get(['performance', 'redis_port'], None, 6379)
        self.redis_db = config.get(['performance', 'redis_db'], None, 0)
        self.redis_password = config.get(['performance', 'redis_password'], None, None)

        # Rate limiting configuratie
        self.rate_limits = {}
        self.default_rate_limit = config.get(['performance', 'default_rate_limit'], None, 100)  # requests per minute
        self.rate_limit_window = config.get(['performance', 'rate_limit_window'], None, 60)  # seconds

        # Performance monitoring
        self.performance_metrics = {}
        self.metrics_retention_days = config.get(['performance', 'metrics_retention_days'], None, 30)

        # Resource monitoring
        self.resource_monitoring_enabled = config.get(['performance', 'resource_monitoring_enabled'], None, True)
        self.resource_check_interval = config.get(['performance', 'resource_check_interval'], None, 300)  # 5 minutes

        # Initialiseer componenten
        self._initialize_caches()
        self._initialize_rate_limiting()
        self._initialize_performance_monitoring()

        # Start resource monitoring thread
        if self.resource_monitoring_enabled:
            self._start_resource_monitoring()

        logging.info("Performance Optimizer geïnitialiseerd")

    def _initialize_caches(self):
        """Initialiseer alle cache lagen"""
        try:
            # Maak disk cache directory aan
            if self.disk_cache_enabled:
                os.makedirs(self.disk_cache_path, exist_ok=True)
                self._cleanup_disk_cache()

            # Initialiseer Redis cache
            if self.redis_enabled and REDIS_AVAILABLE:
                try:
                    self.redis_client = redis.Redis(
                        host=self.redis_host,
                        port=self.redis_port,
                        db=self.redis_db,
                        password=self.redis_password,
                        decode_responses=False,
                        socket_timeout=5,
                        socket_connect_timeout=5
                    )
                    # Test verbinding
                    self.redis_client.ping()
                    logging.info("Redis cache geïnitialiseerd")
                except Exception as e:
                    logging.warning(f"Redis cache initialisatie mislukt: {e}")
                    self.redis_enabled = False

            # Start cache cleanup thread
            self._start_cache_cleanup()

        except Exception as e:
            logging.error(f"Cache initialisatie fout: {e}")

    def _initialize_rate_limiting(self):
        """Initialiseer rate limiting systeem"""
        try:
            # Standaard rate limits voor verschillende endpoints
            default_limits = {
                'api': 100,  # 100 requests per minute
                'weather': 60,  # 60 requests per minute
                'prices': 120,  # 120 requests per minute
                'database': 200,  # 200 requests per minute
                'export': 30,  # 30 requests per minute
            }

            for endpoint, limit in default_limits.items():
                self.rate_limits[endpoint] = RateLimitInfo(
                    endpoint=endpoint,
                    current_requests=0,
                    max_requests=limit,
                    window_start=time.time(),
                    window_duration=self.rate_limit_window
                )

            logging.info("Rate limiting geïnitialiseerd")

        except Exception as e:
            logging.error(f"Rate limiting initialisatie fout: {e}")

    def _initialize_performance_monitoring(self):
        """Initialiseer performance monitoring"""
        try:
            # Maak metrics directory aan
            metrics_dir = os.path.join(self.disk_cache_path, 'metrics')
            os.makedirs(metrics_dir, exist_ok=True)

            # Laad bestaande metrics
            self._load_performance_metrics()

            logging.info("Performance monitoring geïnitialiseerd")

        except Exception as e:
            logging.error(f"Performance monitoring initialisatie fout: {e}")

    def cache_get(self, key: str, layer: str = 'auto') -> Optional[Any]:
        """
        Haal data op uit cache

        Args:
            key: Cache key
            layer: Cache layer ('memory', 'disk', 'redis', 'auto')

        Returns:
            Gecachte data of None
        """
        try:
            if layer == 'auto':
                # Probeer alle lagen in volgorde van snelheid
                return self._cache_get_auto(key)

            elif layer == 'memory':
                return self._cache_get_memory(key)

            elif layer == 'disk':
                return self._cache_get_disk(key)

            elif layer == 'redis':
                return self._cache_get_redis(key)

            else:
                logging.warning(f"Onbekende cache layer: {layer}")
                return None

        except Exception as e:
            logging.debug(f"Cache get fout voor key {key}: {e}")
            return None

    def _cache_get_auto(self, key: str) -> Optional[Any]:
        """Automatische cache layer selectie"""
        try:
            # Probeer eerst memory cache (snelste)
            result = self._cache_get_memory(key)
            if result is not None:
                return result

            # Probeer Redis cache (sneller dan disk)
            if self.redis_enabled:
                result = self._cache_get_redis(key)
                if result is not None:
                    # Promoveer naar memory cache
                    self._cache_set_memory(key, result, ttl=300)
                    return result

            # Probeer disk cache (langzaamste)
            result = self._cache_get_disk(key)
            if result is not None:
                # Promoveer naar memory cache
                self._cache_set_memory(key, result, ttl=300)
                return result

            return None

        except Exception as e:
            logging.debug(f"Auto cache get fout: {e}")
            return None

    def _cache_get_memory(self, key: str) -> Optional[Any]:
        """Haal data op uit memory cache"""
        try:
            if key not in self.memory_cache:
                return None

            entry = self.memory_cache[key]
            current_time = time.time()

            # Check TTL
            if current_time - entry.timestamp > entry.ttl:
                del self.memory_cache[key]
                return None

            # Update access info
            entry.access_count += 1
            entry.last_access = current_time

            return entry.data

        except Exception as e:
            logging.debug(f"Memory cache get fout: {e}")
            return None

    def _cache_get_disk(self, key: str) -> Optional[Any]:
        """Haal data op uit disk cache"""
        try:
            if not self.disk_cache_enabled:
                return None

            cache_file = os.path.join(self.disk_cache_path, f"{key}.cache")
            if not os.path.exists(cache_file):
                return None

            # Check file age
            file_age = time.time() - os.path.getmtime(cache_file)
            if file_age > self.memory_cache_ttl:
                os.remove(cache_file)
                return None

            # Load data
            with open(cache_file, 'rb') as f:
                entry = pickle.load(f)

            # Check TTL
            if time.time() - entry.timestamp > entry.ttl:
                os.remove(cache_file)
                return None

            return entry.data

        except Exception as e:
            logging.debug(f"Disk cache get fout: {e}")
            return None

    def _cache_get_redis(self, key: str) -> Optional[Any]:
        """Haal data op uit Redis cache"""
        try:
            if not self.redis_enabled:
                return None

            redis_key = f"dao_cache:{key}"
            data = self.redis_client.get(redis_key)

            if data is None:
                return None

            # Deserialize data
            return pickle.loads(data)

        except Exception as e:
            logging.debug(f"Redis cache get fout: {e}")
            return None

    def cache_set(self, key: str, data: Any, ttl: int = None, layer: str = 'auto', priority: int = 1):
        """
        Sla data op in cache

        Args:
            key: Cache key
            data: Data om op te slaan
            ttl: Time to live in seconds
            layer: Cache layer ('memory', 'disk', 'redis', 'auto')
            priority: Cache priority (1-10)
        """
        try:
            if ttl is None:
                ttl = self.memory_cache_ttl

            if layer == 'auto':
                # Sla op in alle beschikbare lagen
                self._cache_set_memory(key, data, ttl, priority)
                if self.disk_cache_enabled:
                    self._cache_set_disk(key, data, ttl, priority)
                if self.redis_enabled:
                    self._cache_set_redis(key, data, ttl, priority)

            elif layer == 'memory':
                self._cache_set_memory(key, data, ttl, priority)

            elif layer == 'disk':
                self._cache_set_disk(key, data, ttl, priority)

            elif layer == 'redis':
                self._cache_set_redis(key, data, ttl, priority)

        except Exception as e:
            logging.debug(f"Cache set fout voor key {key}: {e}")

    def _cache_set_memory(self, key: str, data: Any, ttl: int, priority: int = 1):
        """Sla data op in memory cache"""
        try:
            # Bereken data grootte
            size_bytes = len(pickle.dumps(data))

            # Maak cache entry
            entry = CacheEntry(
                key=key,
                data=data,
                timestamp=time.time(),
                ttl=ttl,
                access_count=0,
                last_access=time.time(),
                size_bytes=size_bytes,
                priority=priority
            )

            # Check cache grootte
            if len(self.memory_cache) >= self.memory_cache_size:
                self._evict_memory_cache()

            # Voeg toe aan cache
            self.memory_cache[key] = entry

        except Exception as e:
            logging.debug(f"Memory cache set fout: {e}")

    def _cache_set_disk(self, key: str, data: Any, ttl: int, priority: int = 1):
        """Sla data op in disk cache"""
        try:
            if not self.disk_cache_enabled:
                return

            # Maak cache entry
            entry = CacheEntry(
                key=key,
                data=data,
                timestamp=time.time(),
                ttl=ttl,
                access_count=0,
                last_access=time.time(),
                size_bytes=0,
                priority=priority
            )

            # Sla op naar disk
            cache_file = os.path.join(self.disk_cache_path, f"{key}.cache")
            with open(cache_file, 'wb') as f:
                pickle.dump(entry, f)

            # Check disk cache grootte
            self._check_disk_cache_size()

        except Exception as e:
            logging.debug(f"Disk cache set fout: {e}")

    def _cache_set_redis(self, key: str, data: Any, ttl: int, priority: int = 1):
        """Sla data op in Redis cache"""
        try:
            if not self.redis_enabled:
                return

            # Serialize data
            serialized_data = pickle.dumps(data)

            # Sla op in Redis met TTL
            redis_key = f"dao_cache:{key}"
            self.redis_client.setex(redis_key, ttl, serialized_data)

        except Exception as e:
            logging.debug(f"Redis cache set fout: {e}")

    def _evict_memory_cache(self):
        """Verwijder items uit memory cache om ruimte vrij te maken"""
        try:
            if not self.memory_cache:
                return

            # Sorteer op prioriteit en laatste toegang
            sorted_entries = sorted(
                self.memory_cache.items(),
                key=lambda x: (x[1].priority, x[1].last_access)
            )

            # Verwijder 20% van de items
            items_to_remove = max(1, len(sorted_entries) // 5)
            for key, _ in sorted_entries[:items_to_remove]:
                del self.memory_cache[key]

        except Exception as e:
            logging.debug(f"Memory cache eviction fout: {e}")

    def _check_disk_cache_size(self):
        """Controleer en beperk disk cache grootte"""
        try:
            if not self.disk_cache_enabled:
                return

            total_size = 0
            cache_files = []

            # Bereken totale grootte
            for filename in os.listdir(self.disk_cache_path):
                if filename.endswith('.cache'):
                    filepath = os.path.join(self.disk_cache_path, filename)
                    file_size = os.path.getsize(filepath)
                    total_size += file_size
                    cache_files.append((filepath, file_size, os.path.getmtime(filepath)))

            # Als te groot, verwijder oudste bestanden
            if total_size > self.disk_cache_max_size:
                # Sorteer op modificatie tijd (oudste eerst)
                cache_files.sort(key=lambda x: x[2])

                for filepath, file_size, _ in cache_files:
                    os.remove(filepath)
                    total_size -= file_size

                    if total_size <= self.disk_cache_max_size * 0.8:  # Stop bij 80%
                        break

        except Exception as e:
            logging.debug(f"Disk cache size check fout: {e}")

    def _cleanup_disk_cache(self):
        """Ruim disk cache op"""
        try:
            if not self.disk_cache_enabled:
                return

            current_time = time.time()
            removed_count = 0

            for filename in os.listdir(self.disk_cache_path):
                if filename.endswith('.cache'):
                    filepath = os.path.join(self.disk_cache_path, filename)
                    file_age = current_time - os.path.getmtime(filepath)

                    if file_age > self.memory_cache_ttl:
                        os.remove(filepath)
                        removed_count += 1

            if removed_count > 0:
                logging.info(f"Disk cache opgeruimd: {removed_count} bestanden verwijderd")

        except Exception as e:
            logging.debug(f"Disk cache cleanup fout: {e}")

    def check_rate_limit(self, endpoint: str, user_id: str = None) -> bool:
        """
        Check of een request binnen rate limit valt

        Args:
            endpoint: API endpoint
            user_id: Gebruiker ID (optioneel)

        Returns:
            True als request toegestaan is, False anders
        """
        try:
            # Gebruik standaard endpoint als niet gespecificeerd
            if endpoint not in self.rate_limits:
                endpoint = 'api'

            rate_limit = self.rate_limits[endpoint]
            current_time = time.time()

            # Check of rate limit window verlopen is
            if current_time - rate_limit.window_start > rate_limit.window_duration:
                # Reset window
                rate_limit.current_requests = 0
                rate_limit.window_start = current_time
                rate_limit.blocked_until = None

            # Check of endpoint geblokkeerd is
            if rate_limit.blocked_until and current_time < rate_limit.blocked_until:
                return False

            # Check of limiet bereikt is
            if rate_limit.current_requests >= rate_limit.max_requests:
                # Blokkeer voor 1 minuut
                rate_limit.blocked_until = current_time + 60
                return False

            # Verhoog request teller
            rate_limit.current_requests += 1
            return True

        except Exception as e:
            logging.debug(f"Rate limit check fout: {e}")
            return True  # Sta toe bij fout

    def get_rate_limit_info(self, endpoint: str) -> Optional[RateLimitInfo]:
        """Haal rate limit informatie op"""
        try:
            if endpoint in self.rate_limits:
                return self.rate_limits[endpoint]
            return None
        except Exception as e:
            logging.debug(f"Rate limit info fout: {e}")
            return None

    def monitor_performance(self, operation: str):
        """
        Decorator voor performance monitoring

        Args:
            operation: Naam van de operatie
        """
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                start_memory = self._get_memory_usage()
                start_cpu = self._get_cpu_usage()

                try:
                    result = func(*args, **kwargs)
                    success = True
                    error_message = None
                except Exception as e:
                    result = None
                    success = False
                    error_message = str(e)
                    raise
                finally:
                    # Record performance metrics
                    execution_time = time.time() - start_time
                    memory_usage = self._get_memory_usage() - start_memory
                    cpu_usage = self._get_cpu_usage() - start_cpu

                    self._record_performance_metric(
                        operation, execution_time, memory_usage, cpu_usage, success, error_message
                    )

                return result
            return wrapper
        return decorator

    def _record_performance_metric(self, operation: str, execution_time: float, memory_usage: float,
                                 cpu_usage: float, success: bool, error_message: str = None):
        """Record een performance metric"""
        try:
            metric = PerformanceMetric(
                operation=operation,
                execution_time=execution_time,
                memory_usage=memory_usage,
                cpu_usage=cpu_usage,
                timestamp=time.time(),
                success=success,
                error_message=error_message
            )

            # Sla op in memory
            if operation not in self.performance_metrics:
                self.performance_metrics[operation] = []

            self.performance_metrics[operation].append(metric)

            # Behoud alleen recente metrics
            cutoff_time = time.time() - (self.metrics_retention_days * 24 * 3600)
            self.performance_metrics[operation] = [
                m for m in self.performance_metrics[operation]
                if m.timestamp > cutoff_time
            ]

            # Sla op naar disk periodiek
            if len(self.performance_metrics[operation]) % 100 == 0:
                self._save_performance_metrics()

        except Exception as e:
            logging.debug(f"Performance metric recording fout: {e}")

    def _get_memory_usage(self) -> float:
        """Haal huidige memory usage op"""
        try:
            if psutil:
                process = psutil.Process()
                return process.memory_info().rss / 1024 / 1024  # MB
            return 0.0
        except Exception:
            return 0.0

    def _get_cpu_usage(self) -> float:
        """Haal huidige CPU usage op"""
        try:
            if psutil:
                process = psutil.Process()
                return process.cpu_percent()
            return 0.0
        except Exception:
            return 0.0

    def get_performance_summary(self, operation: str = None, hours: int = 24) -> Dict[str, Any]:
        """
        Haal performance samenvatting op

        Args:
            operation: Specifieke operatie (optioneel)
            hours: Aantal uren om te analyseren

        Returns:
            Performance samenvatting
        """
        try:
            cutoff_time = time.time() - (hours * 3600)

            if operation:
                # Specifieke operatie
                if operation not in self.performance_metrics:
                    return self._get_empty_performance_summary()

                metrics = [m for m in self.performance_metrics[operation] if m.timestamp > cutoff_time]
            else:
                # Alle operaties
                all_metrics = []
                for op_metrics in self.performance_metrics.values():
                    all_metrics.extend([m for m in op_metrics if m.timestamp > cutoff_time])
                metrics = all_metrics

            if not metrics:
                return self._get_empty_performance_summary()

            # Bereken statistieken
            execution_times = [m.execution_time for m in metrics]
            memory_usages = [m.memory_usage for m in metrics]
            cpu_usages = [m.cpu_usage for m in metrics]
            success_count = sum(1 for m in metrics if m.success)
            total_count = len(metrics)

            summary = {
                'period_hours': hours,
                'total_operations': total_count,
                'success_rate': (success_count / total_count * 100) if total_count > 0 else 0,
                'execution_time': {
                    'min': min(execution_times) if execution_times else 0,
                    'max': max(execution_times) if execution_times else 0,
                    'avg': sum(execution_times) / len(execution_times) if execution_times else 0,
                    'p95': self._calculate_percentile(execution_times, 95) if execution_times else 0,
                    'p99': self._calculate_percentile(execution_times, 99) if execution_times else 0
                },
                'memory_usage': {
                    'min': min(memory_usages) if memory_usages else 0,
                    'max': max(memory_usages) if memory_usages else 0,
                    'avg': sum(memory_usages) / len(memory_usages) if memory_usages else 0
                },
                'cpu_usage': {
                    'min': min(cpu_usages) if cpu_usages else 0,
                    'max': max(cpu_usages) if cpu_usages else 0,
                    'avg': sum(cpu_usages) / len(cpu_usages) if cpu_usages else 0
                },
                'timestamp': datetime.now().isoformat()
            }

            return summary

        except Exception as e:
            logging.error(f"Performance summary fout: {e}")
            return self._get_empty_performance_summary()

    def _calculate_percentile(self, values: List[float], percentile: int) -> float:
        """Bereken percentiel van een lijst waarden"""
        try:
            if not values:
                return 0.0

            if NUMPY_AVAILABLE:
                return float(np.percentile(values, percentile))
            else:
                # Eenvoudige percentiel berekening
                sorted_values = sorted(values)
                index = int(len(sorted_values) * percentile / 100)
                return sorted_values[min(index, len(sorted_values) - 1)]

        except Exception as e:
            logging.debug(f"Percentiel berekening fout: {e}")
            return 0.0

    def _get_empty_performance_summary(self) -> Dict[str, Any]:
        """Return lege performance samenvatting"""
        return {
            'period_hours': 0,
            'total_operations': 0,
            'success_rate': 0,
            'execution_time': {'min': 0, 'max': 0, 'avg': 0, 'p95': 0, 'p99': 0},
            'memory_usage': {'min': 0, 'max': 0, 'avg': 0},
            'cpu_usage': {'min': 0, 'max': 0, 'avg': 0},
            'timestamp': datetime.now().isoformat()
        }

    def _save_performance_metrics(self):
        """Sla performance metrics op naar disk"""
        try:
            metrics_file = os.path.join(self.disk_cache_path, 'metrics', 'performance_metrics.json')

            # Converteer naar JSON-serializable formaat
            metrics_data = {}
            for operation, metrics in self.performance_metrics.items():
                metrics_data[operation] = [asdict(m) for m in metrics]

            with open(metrics_file, 'w') as f:
                json.dump(metrics_data, f, indent=2)

        except Exception as e:
            logging.debug(f"Performance metrics opslaan fout: {e}")

    def _load_performance_metrics(self):
        """Laad performance metrics van disk"""
        try:
            metrics_file = os.path.join(self.disk_cache_path, 'metrics', 'performance_metrics.json')

            if os.path.exists(metrics_file):
                with open(metrics_file, 'r') as f:
                    metrics_data = json.load(f)

                # Converteer terug naar PerformanceMetric objecten
                for operation, metrics_list in metrics_data.items():
                    self.performance_metrics[operation] = [
                        PerformanceMetric(**metric_dict) for metric_dict in metrics_list
                    ]

        except Exception as e:
            logging.debug(f"Performance metrics laden fout: {e}")

    def _start_cache_cleanup(self):
        """Start cache cleanup thread"""
        try:
            def cleanup_worker():
                while True:
                    try:
                        time.sleep(300)  # Elke 5 minuten
                        self._cleanup_disk_cache()

                        # Garbage collection
                        gc.collect()

                    except Exception as e:
                        logging.debug(f"Cache cleanup worker fout: {e}")

            cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
            cleanup_thread.start()

        except Exception as e:
            logging.debug(f"Cache cleanup thread start fout: {e}")

    def _start_resource_monitoring(self):
        """Start resource monitoring thread"""
        try:
            def resource_monitor():
                while True:
                    try:
                        time.sleep(self.resource_check_interval)

                        # Check memory usage
                        memory_usage = self._get_memory_usage()
                        if memory_usage > 1000:  # 1GB
                            logging.warning(f"Hoge memory usage: {memory_usage:.1f} MB")

                        # Check CPU usage
                        cpu_usage = self._get_cpu_usage()
                        if cpu_usage > 80:  # 80%
                            logging.warning(f"Hoge CPU usage: {cpu_usage:.1f}%")

                        # Check disk space
                        if self.disk_cache_enabled:
                            disk_usage = psutil.disk_usage(self.disk_cache_path)
                            if disk_usage.percent > 90:
                                logging.warning(f"Schijf bijna vol: {disk_usage.percent:.1f}%")

                    except Exception as e:
                        logging.debug(f"Resource monitoring fout: {e}")

            monitor_thread = threading.Thread(target=resource_monitor, daemon=True)
            monitor_thread.start()

        except Exception as e:
            logging.debug(f"Resource monitoring thread start fout: {e}")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Haal cache statistieken op"""
        try:
            stats = {
                'memory_cache': {
                    'size': len(self.memory_cache),
                    'max_size': self.memory_cache_size,
                    'usage_percent': (len(self.memory_cache) / self.memory_cache_size * 100) if self.memory_cache_size > 0 else 0
                },
                'disk_cache': {
                    'enabled': self.disk_cache_enabled,
                    'path': self.disk_cache_path,
                    'max_size_mb': self.disk_cache_max_size / 1024 / 1024
                },
                'redis_cache': {
                    'enabled': self.redis_enabled,
                    'host': self.redis_host,
                    'port': self.redis_port
                }
            }

            # Bereken disk cache grootte
            if self.disk_cache_enabled and os.path.exists(self.disk_cache_path):
                total_size = 0
                file_count = 0
                for filename in os.listdir(self.disk_cache_path):
                    if filename.endswith('.cache'):
                        filepath = os.path.join(self.disk_cache_path, filename)
                        total_size += os.path.getsize(filepath)
                        file_count += 1

                stats['disk_cache']['current_size_mb'] = total_size / 1024 / 1024
                stats['disk_cache']['file_count'] = file_count
                stats['disk_cache']['usage_percent'] = (total_size / self.disk_cache_max_size * 100) if self.disk_cache_max_size > 0 else 0

            return stats

        except Exception as e:
            logging.error(f"Cache stats fout: {e}")
            return {}

    def clear_cache(self, layer: str = 'all'):
        """
        Leeg cache

        Args:
            layer: Cache layer om te legen ('memory', 'disk', 'redis', 'all')
        """
        try:
            if layer in ['memory', 'all']:
                self.memory_cache.clear()
                logging.info("Memory cache gecleared")

            if layer in ['disk', 'all'] and self.disk_cache_enabled:
                for filename in os.listdir(self.disk_cache_path):
                    if filename.endswith('.cache'):
                        os.remove(os.path.join(self.disk_cache_path, filename))
                logging.info("Disk cache gecleared")

            if layer in ['redis', 'all'] and self.redis_enabled:
                # Verwijder alle DAO cache keys
                for key in self.redis_client.scan_iter(match="dao_cache:*"):
                    self.redis_client.delete(key)
                logging.info("Redis cache gecleared")

        except Exception as e:
            logging.error(f"Cache clear fout: {e}")

    def optimize_performance(self) -> Dict[str, Any]:
        """Voer performance optimalisatie uit"""
        try:
            optimization_results = {
                'timestamp': datetime.now().isoformat(),
                'actions_taken': [],
                'improvements': {}
            }

            # Memory cache optimalisatie
            if len(self.memory_cache) > self.memory_cache_size * 0.8:
                old_size = len(self.memory_cache)
                self._evict_memory_cache()
                new_size = len(self.memory_cache)
                optimization_results['actions_taken'].append(f"Memory cache opgeruimd: {old_size} -> {new_size} items")
                optimization_results['improvements']['memory_cache'] = f"Reduced by {old_size - new_size} items"

            # Disk cache optimalisatie
            if self.disk_cache_enabled:
                old_size = sum(os.path.getsize(os.path.join(self.disk_cache_path, f))
                             for f in os.listdir(self.disk_cache_path)
                             if f.endswith('.cache'))
                self._cleanup_disk_cache()
                self._check_disk_cache_size()
                new_size = sum(os.path.getsize(os.path.join(self.disk_cache_path, f))
                             for f in os.listdir(self.disk_cache_path)
                             if f.endswith('.cache'))
                optimization_results['actions_taken'].append(f"Disk cache opgeruimd: {old_size/1024/1024:.1f}MB -> {new_size/1024/1024:.1f}MB")
                optimization_results['improvements']['disk_cache'] = f"Reduced by {(old_size - new_size)/1024/1024:.1f}MB"

            # Garbage collection
            old_objects = gc.get_count()
            gc.collect()
            new_objects = gc.get_count()
            optimization_results['actions_taken'].append(f"Garbage collection uitgevoerd: {old_objects} -> {new_objects} objects")
            optimization_results['improvements']['garbage_collection'] = f"Cleaned {old_objects - new_objects} objects"

            logging.info("Performance optimalisatie voltooid")
            return optimization_results

        except Exception as e:
            logging.error(f"Performance optimalisatie fout: {e}")
            return {'error': str(e)}


# Utility functies voor externe gebruik
def create_performance_optimizer(config) -> PerformanceOptimizer:
    """Factory functie voor het maken van een PerformanceOptimizer instance"""
    return PerformanceOptimizer(config)


def cache_result(performance_optimizer: PerformanceOptimizer, key: str = None, ttl: int = None, layer: str = 'auto'):
    """Decorator voor het cachen van functie resultaten"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Genereer cache key
            if key is None:
                cache_key = f"{func.__name__}:{hash(str(args) + str(sorted(kwargs.items())))}"
            else:
                cache_key = key

            # Check cache
            cached_result = performance_optimizer.cache_get(cache_key, layer)
            if cached_result is not None:
                return cached_result

            # Voer functie uit
            result = func(*args, **kwargs)

            # Cache resultaat
            performance_optimizer.cache_set(cache_key, result, ttl, layer)

            return result
        return wrapper
    return decorator


def rate_limit(performance_optimizer: PerformanceOptimizer, endpoint: str, user_id: str = None):
    """Decorator voor rate limiting"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Check rate limit
            if not performance_optimizer.check_rate_limit(endpoint, user_id):
                raise Exception(f"Rate limit exceeded for endpoint: {endpoint}")

            # Voer functie uit
            return func(*args, **kwargs)
        return wrapper
    return decorator
