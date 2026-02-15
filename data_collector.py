"""
Data Collector for CoinMarketCap API
Fetches cryptocurrency data including latest prices and historical data
"""
import requests
import pandas as pd
import time
from datetime import datetime, timedelta
import config
from utils import setup_logger, save_json, load_json, APIError
import os

logger = setup_logger(__name__)

class CoinMarketCapAPI:
    """CoinMarketCap API client for fetching cryptocurrency data"""
    
    def __init__(self):
        self.api_key = config.COINMARKETCAP_API_KEY
        self.base_url = config.COINMARKETCAP_BASE_URL
        self.headers = {
            'Accepts': 'application/json',
            'X-CMC_PRO_API_KEY': self.api_key,
        }
        
    def test_connection(self):
        """Test API connection"""
        try:
            url = f"{self.base_url}/cryptocurrency/map"
            params = {'limit': 1}
            response = requests.get(url, headers=self.headers, params=params)
            
            if response.status_code == 200:
                logger.info("✓ API connection successful!")
                return True
            else:
                logger.error(f"✗ API connection failed: {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"✗ API connection error: {str(e)}")
            return False
    
    def get_latest_listings(self, limit=100):
        """Get latest cryptocurrency listings"""
        try:
            url = f"{self.base_url}/cryptocurrency/listings/latest"
            params = {
                'limit': limit,
                'convert': 'USD'
            }
            
            response = requests.get(url, headers=self.headers, params=params)
            
            if response.status_code == 200:
                data = response.json()
                return data['data']
            else:
                raise APIError(f"API request failed: {response.status_code} - {response.text}")
                
        except Exception as e:
            logger.error(f"Error fetching latest listings: {str(e)}")
            raise APIError(str(e))
    
    def get_cryptocurrency_info(self, symbols):
        """Get cryptocurrency metadata"""
        try:
            url = f"{self.base_url}/cryptocurrency/info"
            params = {'symbol': ','.join(symbols)}
            
            response = requests.get(url, headers=self.headers, params=params)
            
            if response.status_code == 200:
                data = response.json()
                return data['data']
            else:
                raise APIError(f"API request failed: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Error fetching crypto info: {str(e)}")
            raise APIError(str(e))
    
    def get_quotes_latest(self, symbols):
        """Get latest quotes for specific cryptocurrencies"""
        try:
            url = f"{self.base_url}/cryptocurrency/quotes/latest"
            params = {
                'symbol': ','.join(symbols),
                'convert': 'USD'
            }
            
            response = requests.get(url, headers=self.headers, params=params)
            
            if response.status_code == 200:
                data = response.json()
                return data['data']
            else:
                raise APIError(f"API request failed: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Error fetching quotes: {str(e)}")
            raise APIError(str(e))
    
    def get_ohlcv_historical(self, symbol, time_period='daily', count=365):
        """
        Get historical OHLCV data
        Note: This endpoint requires a paid plan. We'll simulate data for free tier.
        """
        logger.warning("Historical OHLCV requires paid API plan. Using quotes data instead.")
        
        # For free tier, we'll collect current data and build historical dataset over time
        # Or use alternative free data sources
        return None
    
    def parse_listings_to_dataframe(self, listings_data):
        """Parse listings data to pandas DataFrame"""
        parsed_data = []
        
        for crypto in listings_data:
            quote = crypto['quote']['USD']
            parsed_data.append({
                'symbol': crypto['symbol'],
                'name': crypto['name'],
                'price': quote['price'],
                'volume_24h': quote['volume_24h'],
                'volume_change_24h': quote['volume_change_24h'],
                'percent_change_1h': quote['percent_change_1h'],
                'percent_change_24h': quote['percent_change_24h'],
                'percent_change_7d': quote['percent_change_7d'],
                'percent_change_30d': quote['percent_change_30d'],
                'market_cap': quote['market_cap'],
                'market_cap_dominance': quote['market_cap_dominance'],
                'circulating_supply': crypto['circulating_supply'],
                'total_supply': crypto['total_supply'],
                'max_supply': crypto['max_supply'],
                'timestamp': datetime.now().isoformat()
            })
        
        df = pd.DataFrame(parsed_data)
        return df
    
    def collect_and_save_data(self, filename=None):
        """Collect current market data and save to CSV"""
        try:
            logger.info("Collecting cryptocurrency data...")
            
            # Get latest listings
            listings = self.get_latest_listings(limit=config.DATA_LIMIT)
            df = self.parse_listings_to_dataframe(listings)
            
            # Save to CSV
            if filename is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"{config.DATA_DIR}/crypto_data_{timestamp}.csv"
            
            df.to_csv(filename, index=False)
            logger.info(f"✓ Data saved to {filename}")
            logger.info(f"✓ Collected data for {len(df)} cryptocurrencies")
            
            return df
            
        except Exception as e:
            logger.error(f"Error collecting data: {str(e)}")
            raise

def collect_historical_data_simulation(symbols, days=365):
    """
    Simulate historical data collection for demonstration
    In production, you would use a paid API or alternative data source
    """
    logger.info(f"Simulating historical data for {len(symbols)} cryptocurrencies...")
    
    # This is a placeholder - in real implementation, you'd use:
    # 1. CoinMarketCap paid API
    # 2. Alternative free APIs (CoinGecko, Binance, etc.)
    # 3. Web scraping (with permission)
    # 4. Pre-downloaded datasets
    
    historical_data = {}
    
    for symbol in symbols:
        # Generate sample data structure
        dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
        
        # Placeholder - would be replaced with actual API calls
        historical_data[symbol] = {
            'dates': dates.tolist(),
            'note': 'Use alternative data source for historical data'
        }
    
    return historical_data

def main():
    """Test the data collector"""
    print("="*60)
    print("  CoinMarketCap Data Collector Test")
    print("="*60)
    
    api = CoinMarketCapAPI()
    
    # Test connection
    print("\n1. Testing API connection...")
    if api.test_connection():
        print("   ✓ Connection successful!")
    else:
        print("   ✗ Connection failed!")
        return
    
    # Collect data
    print("\n2. Collecting latest cryptocurrency data...")
    df = api.collect_and_save_data()
    
    # Display sample
    print("\n3. Sample data (top 10 cryptocurrencies):")
    print(df[['symbol', 'name', 'price', 'percent_change_24h', 'market_cap']].head(10))
    
    # Get specific quotes
    print("\n4. Getting quotes for specific cryptocurrencies...")
    quotes = api.get_quotes_latest(config.CRYPTOCURRENCIES[:5])
    
    for symbol, data in quotes.items():
        quote = data['quote']['USD']
        print(f"\n{symbol} ({data['name']}):")
        print(f"  Price: ${quote['price']:.2f}")
        print(f"  24h Change: {quote['percent_change_24h']:.2f}%")
        print(f"  Market Cap: ${quote['market_cap']:,.0f}")

if __name__ == "__main__":
    main()
