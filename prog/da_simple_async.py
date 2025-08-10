"""
Simple async-like functionality using threading and requests.
Alternative to aiohttp for basic concurrent operations.
"""

import requests
import threading
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass


@dataclass
class FetchResult:
    """Result of a fetch operation"""
    url: str
    success: bool
    data: Any = None
    error: str = None
    duration: float = 0.0


class SimpleFetcher:
    """Simple concurrent fetcher using threading"""
    
    def __init__(self, max_workers: int = 4, timeout: int = 30):
        self.max_workers = max_workers
        self.timeout = timeout
        self.session = requests.Session()
        
        # Set reasonable defaults
        self.session.headers.update({
            'User-Agent': 'DAO/1.0'
        })
    
    def fetch_url(self, url: str, headers: Optional[Dict] = None, 
                  params: Optional[Dict] = None) -> FetchResult:
        """Fetch single URL"""
        start_time = time.time()
        
        try:
            response = self.session.get(
                url, 
                headers=headers or {}, 
                params=params or {},
                timeout=self.timeout
            )
            
            duration = time.time() - start_time
            
            if response.status_code == 200:
                try:
                    data = response.json()
                except:
                    data = response.text
                
                return FetchResult(
                    url=url,
                    success=True,
                    data=data,
                    duration=duration
                )
            else:
                return FetchResult(
                    url=url,
                    success=False,
                    error=f"HTTP {response.status_code}",
                    duration=duration
                )
        
        except Exception as e:
            duration = time.time() - start_time
            return FetchResult(
                url=url,
                success=False,
                error=str(e),
                duration=duration
            )
    
    def fetch_multiple(self, urls: List[str], 
                      headers: Optional[Dict] = None) -> List[FetchResult]:
        """Fetch multiple URLs concurrently"""
        results = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_url = {
                executor.submit(self.fetch_url, url, headers): url 
                for url in urls
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_url):
                result = future.result()
                results.append(result)
        
        # Sort results by original URL order
        url_to_result = {r.url: r for r in results}
        return [url_to_result[url] for url in urls if url in url_to_result]
    
    def fetch_with_callbacks(self, tasks: List[Dict[str, Any]]) -> List[FetchResult]:
        """Fetch URLs and execute callbacks"""
        results = []
        
        def fetch_and_callback(task):
            url = task['url']
            callback = task.get('callback')
            callback_args = task.get('callback_args', [])
            
            result = self.fetch_url(url, task.get('headers'))
            
            # Execute callback if provided and fetch was successful
            if callback and result.success:
                try:
                    callback(result.data, *callback_args)
                except Exception as e:
                    logging.error(f"Callback error for {url}: {e}")
            
            return result
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_task = {
                executor.submit(fetch_and_callback, task): task 
                for task in tasks
            }
            
            for future in as_completed(future_to_task):
                result = future.result()
                results.append(result)
        
        return results


class SimpleAsyncManager:
    """Simple async-like manager for coordinating operations"""
    
    def __init__(self, max_concurrent: int = 4):
        self.max_concurrent = max_concurrent
        self.fetcher = SimpleFetcher(max_workers=max_concurrent)
        self.tasks = []
    
    def add_fetch_task(self, url: str, callback: Optional[Callable] = None,
                      headers: Optional[Dict] = None):
        """Add a fetch task"""
        self.tasks.append({
            'type': 'fetch',
            'url': url,
            'callback': callback,
            'headers': headers or {}
        })
    
    def add_function_task(self, func: Callable, args: tuple = (), 
                         callback: Optional[Callable] = None):
        """Add a function execution task"""
        self.tasks.append({
            'type': 'function',
            'function': func,
            'args': args,
            'callback': callback
        })
    
    def execute_all(self) -> List[Any]:
        """Execute all tasks concurrently"""
        results = []
        
        # Separate fetch and function tasks
        fetch_tasks = [t for t in self.tasks if t['type'] == 'fetch']
        function_tasks = [t for t in self.tasks if t['type'] == 'function']
        
        # Execute fetch tasks
        if fetch_tasks:
            fetch_results = self.fetcher.fetch_with_callbacks(fetch_tasks)
            results.extend(fetch_results)
        
        # Execute function tasks
        if function_tasks:
            with ThreadPoolExecutor(max_workers=self.max_concurrent) as executor:
                future_to_task = {
                    executor.submit(self._execute_function_task, task): task 
                    for task in function_tasks
                }
                
                for future in as_completed(future_to_task):
                    result = future.result()
                    results.append(result)
        
        # Clear tasks
        self.tasks.clear()
        
        return results
    
    def _execute_function_task(self, task: Dict) -> Any:
        """Execute a function task"""
        try:
            func = task['function']
            args = task['args']
            callback = task.get('callback')
            
            result = func(*args)
            
            if callback:
                callback(result)
            
            return result
            
        except Exception as e:
            logging.error(f"Function task error: {e}")
            return None


# Convenience functions
def fetch_multiple_urls(urls: List[str], timeout: int = 30) -> List[FetchResult]:
    """Simple function to fetch multiple URLs"""
    fetcher = SimpleFetcher(timeout=timeout)
    return fetcher.fetch_multiple(urls)


def fetch_energy_data_concurrent(price_url: str, weather_url: str) -> Dict[str, Any]:
    """Fetch energy and weather data concurrently"""
    fetcher = SimpleFetcher()
    
    results = fetcher.fetch_multiple([price_url, weather_url])
    
    return {
        'prices': results[0].data if results[0].success else None,
        'weather': results[1].data if len(results) > 1 and results[1].success else None,
        'errors': [r.error for r in results if not r.success]
    }


# Example usage for DAO
def fetch_dao_data_simple():
    """Example of fetching DAO data without aiohttp"""
    manager = SimpleAsyncManager()
    
    # Add fetch tasks
    manager.add_fetch_task(
        'https://api.example.com/prices',
        callback=lambda data: logging.info(f"Got {len(data)} price entries")
    )
    
    manager.add_fetch_task(
        'https://api.example.com/weather',
        callback=lambda data: logging.info(f"Got weather data")
    )
    
    # Add function task
    manager.add_function_task(
        lambda: logging.info("Processing data..."),
        callback=lambda _: logging.info("Processing complete")
    )
    
    # Execute all tasks
    results = manager.execute_all()
    
    return results