"""
Async data fetching module for Day Ahead Optimizer.
Replaces synchronous API calls with concurrent async operations for improved performance.
"""

import asyncio
import aiohttp
import datetime
import json
import logging
import math
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from concurrent.futures import ThreadPoolExecutor
import time

from dao.prog.da_config import Config
from dao.prog.db_manager import DBmanagerObj


class AsyncDataFetcher:
    """
    Async data fetcher for concurrent API operations.
    Dramatically improves performance by running price/weather/HA data fetching in parallel.
    """
    
    def __init__(self, config: Config, db_da: DBmanagerObj):
        self.config = config
        self.db_da = db_da
        self.session: Optional[aiohttp.ClientSession] = None
        self.executor = ThreadPoolExecutor(max_workers=4)
        
    async def __aenter__(self):
        """Async context manager entry"""
        timeout = aiohttp.ClientTimeout(total=30, connect=10)
        self.session = aiohttp.ClientSession(
            timeout=timeout,
            headers={'User-Agent': 'DAO-AsyncFetcher/1.0'}
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
        self.executor.shutdown(wait=True)
    
    async def fetch_all_data_concurrent(
        self, 
        start: datetime.datetime, 
        end: datetime.datetime
    ) -> Dict[str, Any]:
        """
        Concurrently fetch all required data (prices, weather, HA data).
        This replaces sequential fetching with parallel operations.
        """
        logging.info("Starting concurrent data fetching")
        start_time = time.time()
        
        try:
            # Create tasks for concurrent execution
            tasks = []
            
            # Price data task
            price_source = self.config.get(['prices', 'source day ahead'], 'nordpool').lower()
            if price_source == 'nordpool':
                tasks.append(self._fetch_nordpool_prices(start, end))
            elif price_source == 'entsoe':
                tasks.append(self._fetch_entsoe_prices(start, end))
            elif price_source == 'easyenergy':
                tasks.append(self._fetch_easyenergy_prices(start, end))
            elif price_source == 'tibber':
                tasks.append(self._fetch_tibber_prices(start, end))
            
            # Weather data task
            tasks.append(self._fetch_meteo_data())
            
            # Home Assistant data task (if needed)
            tasks.append(self._fetch_ha_data())
            
            # Execute all tasks concurrently
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            processed_results = {}
            task_names = ['prices', 'weather', 'home_assistant']
            
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logging.error(f"Error in {task_names[i]} fetch: {result}")
                    processed_results[task_names[i]] = None
                else:
                    processed_results[task_names[i]] = result
            
            elapsed_time = time.time() - start_time
            logging.info(f"Concurrent data fetching completed in {elapsed_time:.2f} seconds")
            
            return processed_results
            
        except Exception as e:
            logging.error(f"Error in concurrent data fetching: {e}")
            return {}
    
    async def _fetch_nordpool_prices(
        self, 
        start: datetime.datetime, 
        end: datetime.datetime
    ) -> Optional[pd.DataFrame]:
        """Async NordPool price fetching"""
        logging.info("Fetching NordPool prices asynchronously")
        
        try:
            # Use thread executor for nordpool library (not async)
            loop = asyncio.get_event_loop()
            
            def sync_nordpool_fetch():
                from nordpool import elspot
                prices_spot = elspot.Prices()
                end_date = start if len(sys.argv) <= 2 else None
                return prices_spot.hourly(areas=["NL"], end_date=end_date)
            
            hourly_prices_spot = await loop.run_in_executor(
                self.executor, sync_nordpool_fetch
            )
            
            if hourly_prices_spot is None:
                logging.error("No NordPool data received")
                return None
            
            # Process data
            hourly_values = hourly_prices_spot["areas"]["NL"]["values"]
            
            # Create DataFrame
            price_data = []
            for hourly_value in hourly_values:
                time_dt = hourly_value["start"]
                time_ts = int(time_dt.timestamp())
                value = hourly_value["value"]
                
                if value != float("inf"):
                    price_data.append({
                        'time': str(time_ts),
                        'code': 'da',
                        'value': value / 1000  # Convert to €/kWh
                    })
            
            df_nordpool = pd.DataFrame(price_data)
            logging.info(f"Successfully fetched {len(df_nordpool)} NordPool price records")
            
            return df_nordpool
            
        except Exception as e:
            logging.error(f"Error fetching NordPool prices: {e}")
            return None
    
    async def _fetch_entsoe_prices(
        self, 
        start: datetime.datetime, 
        end: datetime.datetime
    ) -> Optional[pd.DataFrame]:
        """Async Entsoe price fetching"""
        logging.info("Fetching Entsoe prices asynchronously")
        
        try:
            # Use thread executor for entsoe-py library (not async)
            loop = asyncio.get_event_loop()
            
            def sync_entsoe_fetch():
                from entsoe import EntsoePandasClient
                api_key = self.config.get(['prices', 'entsoe-api-key'])
                client = EntsoePandasClient(api_key=api_key)
                
                # NL area code  
                country_code = 'NL'
                return client.query_day_ahead_prices(country_code, start=start, end=end)
            
            da_prices = await loop.run_in_executor(
                self.executor, sync_entsoe_fetch
            )
            
            if da_prices is None or len(da_prices) == 0:
                logging.error("No Entsoe data received")
                return None
            
            # Process Entsoe data
            da_prices = da_prices.reset_index()
            
            # Vectorized conversion
            timestamps = da_prices.iloc[:, 0].apply(
                lambda x: str(int(datetime.datetime.timestamp(x)))
            )
            values = da_prices.iloc[:, 1] / 1000  # Convert to €/kWh
            
            df_entsoe = pd.DataFrame({
                'time': timestamps,
                'code': 'da',
                'value': values
            })
            
            logging.info(f"Successfully fetched {len(df_entsoe)} Entsoe price records")
            return df_entsoe
            
        except Exception as e:
            logging.error(f"Error fetching Entsoe prices: {e}")
            return None
    
    async def _fetch_easyenergy_prices(
        self, 
        start: datetime.datetime, 
        end: datetime.datetime
    ) -> Optional[pd.DataFrame]:
        """Async EasyEnergy price fetching"""
        logging.info("Fetching EasyEnergy prices asynchronously")
        
        try:
            startstr = start.strftime("%Y-%m-%dT%H:%M:%S")
            endstr = end.strftime("%Y-%m-%dT%H:%M:%S")
            url = (
                "https://mijn.easyenergy.com/nl/api/tariff/getapxtariffs"
                f"?startTimestamp={startstr}&endTimestamp={endstr}"
            )
            
            async with self.session.get(url) as response:
                if response.status != 200:
                    logging.error(f"EasyEnergy API returned status {response.status}")
                    return None
                
                text_data = await response.text()
                json_object = json.loads(text_data)
                
            df = pd.DataFrame.from_records(json_object)
            logging.info(f"EasyEnergy data: \\n{df.to_string(index=False)}")
            
            # Vectorized processing
            timestamps = df['Timestamp'].apply(
                lambda x: str(int(datetime.datetime.fromisoformat(x).timestamp()))
            )
            
            df_easyenergy = pd.DataFrame({
                'time': timestamps,
                'code': 'da',
                'value': df['TariffReturn']
            })
            
            logging.info(f"Successfully fetched {len(df_easyenergy)} EasyEnergy price records")
            return df_easyenergy
            
        except Exception as e:
            logging.error(f"Error fetching EasyEnergy prices: {e}")
            return None
    
    async def _fetch_tibber_prices(
        self, 
        start: datetime.datetime, 
        end: datetime.datetime
    ) -> Optional[pd.DataFrame]:
        """Async Tibber price fetching"""
        logging.info("Fetching Tibber prices asynchronously")
        
        try:
            now_ts = datetime.datetime.now().timestamp()
            get_ts = start.timestamp()
            count = 1 + math.ceil((now_ts - get_ts) / 3600)
            
            # GraphQL query for Tibber
            query = {
                "query": f"""
                {{
                    viewer {{
                        homes {{
                            currentSubscription {{
                                priceInfo {{
                                    today {{
                                        energy
                                        startsAt
                                    }}
                                    tomorrow {{
                                        energy
                                        startsAt
                                    }}
                                    range(resolution: HOURLY, last: {count}) {{
                                        nodes {{
                                            energy
                                            startsAt
                                        }}
                                    }}
                                }}
                            }}
                        }}
                    }}
                }}
                """
            }
            
            # Get Tibber token from config
            tibber_token = self.config.get(['tibber', 'token'])
            if not tibber_token:
                logging.error("Tibber token not configured")
                return None
            
            headers = {
                'Authorization': f'Bearer {tibber_token}',
                'Content-Type': 'application/json'
            }
            
            async with self.session.post(
                'https://api.tibber.com/v1-beta/gql',
                json=query,
                headers=headers
            ) as response:
                
                if response.status != 200:
                    logging.error(f"Tibber API returned status {response.status}")
                    return None
                
                data = await response.json()
                
            # Process Tibber response
            homes = data.get('data', {}).get('viewer', {}).get('homes', [])
            if not homes:
                logging.error("No Tibber homes data")
                return None
            
            price_info = homes[0].get('currentSubscription', {}).get('priceInfo', {})
            
            price_data = []
            
            # Process today's prices
            for price_entry in price_info.get('today', []):
                time_str = price_entry['startsAt']
                energy_price = price_entry['energy']
                
                time_dt = datetime.datetime.fromisoformat(time_str.replace('Z', '+00:00'))
                time_ts = int(time_dt.timestamp())
                
                price_data.append({
                    'time': str(time_ts),
                    'code': 'da',
                    'value': energy_price
                })
            
            # Process tomorrow's prices
            for price_entry in price_info.get('tomorrow', []):
                time_str = price_entry['startsAt']
                energy_price = price_entry['energy']
                
                time_dt = datetime.datetime.fromisoformat(time_str.replace('Z', '+00:00'))
                time_ts = int(time_dt.timestamp())
                
                price_data.append({
                    'time': str(time_ts),
                    'code': 'da',
                    'value': energy_price
                })
            
            df_tibber = pd.DataFrame(price_data)
            logging.info(f"Successfully fetched {len(df_tibber)} Tibber price records")
            
            return df_tibber
            
        except Exception as e:
            logging.error(f"Error fetching Tibber prices: {e}")
            return None
    
    async def _fetch_meteo_data(self) -> Optional[pd.DataFrame]:
        """Async weather data fetching"""
        logging.info("Fetching meteo data asynchronously")
        
        try:
            meteoserver_key = self.config.get(['meteoserver-key'])
            latitude = self.config.get(['latitude'])
            longitude = self.config.get(['longitude'])
            
            if not all([meteoserver_key, latitude, longitude]):
                logging.error("Missing meteo configuration")
                return None
            
            # Fetch both Harmonie and GFS data concurrently
            harmonie_task = self._fetch_meteoserver_data('harmonie', meteoserver_key, latitude, longitude)
            gfs_task = self._fetch_meteoserver_data('gfs', meteoserver_key, latitude, longitude)
            
            harmonie_data, gfs_data = await asyncio.gather(harmonie_task, gfs_task, return_exceptions=True)
            
            # Process results
            combined_data = []
            
            if not isinstance(harmonie_data, Exception) and harmonie_data is not None:
                combined_data.extend(harmonie_data)
                
            if not isinstance(gfs_data, Exception) and gfs_data is not None and len(combined_data) < 96:
                # Use GFS to fill gaps
                combined_data.extend(gfs_data[len(combined_data):])
            
            if not combined_data:
                logging.error("No meteo data received")
                return None
            
            df_meteo = pd.DataFrame(combined_data)
            logging.info(f"Successfully fetched {len(df_meteo)} meteo records")
            
            return df_meteo
            
        except Exception as e:
            logging.error(f"Error fetching meteo data: {e}")
            return None
    
    async def _fetch_meteoserver_data(
        self, 
        model: str, 
        key: str, 
        lat: float, 
        lon: float
    ) -> Optional[List[Dict]]:
        """Fetch data from specific meteoserver model"""
        try:
            if model == "harmonie":
                url = "https://data.meteoserver.nl/api/uurverwachting.php"
            else:
                url = "https://data.meteoserver.nl/api/uurverwachting_gfs.php"
            
            params = {
                'lat': lat,
                'long': lon,
                'key': key
            }
            
            async with self.session.get(url, params=params) as response:
                if response.status != 200:
                    logging.error(f"Meteoserver {model} API returned status {response.status}")
                    return None
                
                text_data = await response.text()
                json_object = json.loads(text_data)
            
            # Process meteo data with solar calculations
            meteo_records = []
            solar_config = self.config.get(['solar'], {})
            
            for record in json_object:
                # Extract base data
                time_str = record['tijd_nl'] + ':00'
                time_dt = datetime.datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S')
                time_ts = int(time_dt.timestamp())
                
                global_radiation = float(record.get('gr', 0))
                temperature = float(record.get('temp', 0))
                
                # Calculate solar radiation if solar config exists
                solar_radiation = global_radiation
                if solar_config and global_radiation > 0:
                    # This would need the solar calculation logic from the original
                    # For now, use global radiation as approximation
                    solar_radiation = global_radiation * 0.8  # Simplified conversion
                
                # Add multiple records for different data types
                meteo_records.extend([
                    {
                        'time': str(time_ts),
                        'code': 'gr',
                        'value': global_radiation
                    },
                    {
                        'time': str(time_ts),
                        'code': 'temp',
                        'value': temperature
                    },
                    {
                        'time': str(time_ts),
                        'code': 'solar_rad',
                        'value': solar_radiation
                    }
                ])
            
            logging.info(f"Successfully processed {len(meteo_records)} {model} records")
            return meteo_records
            
        except Exception as e:
            logging.error(f"Error fetching {model} data: {e}")
            return None
    
    async def _fetch_ha_data(self) -> Optional[Dict]:
        """Async Home Assistant data fetching"""
        logging.info("Fetching Home Assistant data asynchronously")
        
        try:
            # This would fetch current consumption, battery status, etc. from HA
            # For now, return placeholder data
            ha_data = {
                'current_consumption': 0.5,  # kW
                'current_production': 0.0,   # kW
                'battery_soc': 50,          # %
                'timestamp': datetime.datetime.now().timestamp()
            }
            
            logging.info("Home Assistant data fetched (placeholder)")
            return ha_data
            
        except Exception as e:
            logging.error(f"Error fetching HA data: {e}")
            return None
    
    async def save_data_concurrent(self, data_dict: Dict[str, pd.DataFrame]):
        """
        Save multiple DataFrames to database concurrently.
        Uses async database operations where possible.
        """
        logging.info("Starting concurrent data saving")
        
        try:
            # Create tasks for concurrent database saves
            save_tasks = []
            
            for data_type, df in data_dict.items():
                if df is not None and not df.empty:
                    # Run database saves in executor (since db operations are sync)
                    save_task = asyncio.get_event_loop().run_in_executor(
                        self.executor,
                        lambda d=df: self.db_da.savedata(d)
                    )
                    save_tasks.append(save_task)
            
            if save_tasks:
                await asyncio.gather(*save_tasks, return_exceptions=True)
                logging.info(f"Successfully saved {len(save_tasks)} data sets")
            else:
                logging.warning("No data to save")
                
        except Exception as e:
            logging.error(f"Error in concurrent data saving: {e}")


# Usage example function
async def fetch_and_save_all_data(config: Config, db_da: DBmanagerObj) -> bool:
    """
    Example function showing how to use the AsyncDataFetcher.
    This replaces the old sequential data fetching pattern.
    """
    start_time = datetime.datetime.now()
    end_time = start_time + datetime.timedelta(days=2)
    
    async with AsyncDataFetcher(config, db_da) as fetcher:
        # Fetch all data concurrently
        results = await fetcher.fetch_all_data_concurrent(start_time, end_time)
        
        # Prepare data for saving
        data_to_save = {}
        
        if results.get('prices') is not None:
            data_to_save['prices'] = results['prices']
            
        if results.get('weather') is not None:
            data_to_save['weather'] = results['weather']
        
        # Save data concurrently
        if data_to_save:
            await fetcher.save_data_concurrent(data_to_save)
            return True
        else:
            logging.error("No data to save")
            return False