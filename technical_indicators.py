"""
Technical Indicators Calculator
Calculates various technical analysis indicators for cryptocurrency data
"""
import pandas as pd
import numpy as np
from ta.trend import MACD, EMAIndicator, SMAIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator, VolumeWeightedAveragePrice
import config
from utils import setup_logger, validate_dataframe

logger = setup_logger(__name__)

class TechnicalIndicators:
    """Calculate technical indicators for cryptocurrency price data"""
    
    def __init__(self, df):
        """
        Initialize with price dataframe
        Expected columns: open, high, low, close, volume
        """
        self.df = df.copy()
        
    def add_sma(self, periods=None):
        """Add Simple Moving Averages"""
        if periods is None:
            periods = config.SMA_PERIODS
        
        for period in periods:
            if len(self.df) >= period:
                indicator = SMAIndicator(close=self.df['close'], window=period)
                self.df[f'sma_{period}'] = indicator.sma_indicator()
        
        logger.info(f"✓ Added SMA indicators: {periods}")
        return self
    
    def add_ema(self, periods=None):
        """Add Exponential Moving Averages"""
        if periods is None:
            periods = config.EMA_PERIODS
        
        for period in periods:
            if len(self.df) >= period:
                indicator = EMAIndicator(close=self.df['close'], window=period)
                self.df[f'ema_{period}'] = indicator.ema_indicator()
        
        logger.info(f"✓ Added EMA indicators: {periods}")
        return self
    
    def add_rsi(self, period=None):
        """Add Relative Strength Index"""
        if period is None:
            period = config.RSI_PERIOD
        
        if len(self.df) >= period:
            indicator = RSIIndicator(close=self.df['close'], window=period)
            self.df['rsi'] = indicator.rsi()
        
        logger.info(f"✓ Added RSI indicator (period={period})")
        return self
    
    def add_macd(self, fast=None, slow=None, signal=None):
        """Add MACD indicator"""
        if fast is None:
            fast = config.MACD_FAST
        if slow is None:
            slow = config.MACD_SLOW
        if signal is None:
            signal = config.MACD_SIGNAL
        
        if len(self.df) >= slow:
            indicator = MACD(
                close=self.df['close'],
                window_fast=fast,
                window_slow=slow,
                window_sign=signal
            )
            self.df['macd'] = indicator.macd()
            self.df['macd_signal'] = indicator.macd_signal()
            self.df['macd_diff'] = indicator.macd_diff()
        
        logger.info(f"✓ Added MACD indicator (fast={fast}, slow={slow}, signal={signal})")
        return self
    
    def add_bollinger_bands(self, period=None, std=None):
        """Add Bollinger Bands"""
        if period is None:
            period = config.BB_PERIOD
        if std is None:
            std = config.BB_STD
        
        if len(self.df) >= period:
            indicator = BollingerBands(
                close=self.df['close'],
                window=period,
                window_dev=std
            )
            self.df['bb_high'] = indicator.bollinger_hband()
            self.df['bb_mid'] = indicator.bollinger_mavg()
            self.df['bb_low'] = indicator.bollinger_lband()
            self.df['bb_width'] = indicator.bollinger_wband()
        
        logger.info(f"✓ Added Bollinger Bands (period={period}, std={std})")
        return self
    
    def add_stochastic(self, period=14):
        """Add Stochastic Oscillator"""
        if len(self.df) >= period:
            indicator = StochasticOscillator(
                high=self.df['high'],
                low=self.df['low'],
                close=self.df['close'],
                window=period
            )
            self.df['stoch_k'] = indicator.stoch()
            self.df['stoch_d'] = indicator.stoch_signal()
        
        logger.info(f"✓ Added Stochastic Oscillator (period={period})")
        return self
    
    def add_atr(self, period=14):
        """Add Average True Range"""
        if len(self.df) >= period:
            indicator = AverageTrueRange(
                high=self.df['high'],
                low=self.df['low'],
                close=self.df['close'],
                window=period
            )
            self.df['atr'] = indicator.average_true_range()
        
        logger.info(f"✓ Added ATR indicator (period={period})")
        return self
    
    def add_obv(self):
        """Add On-Balance Volume"""
        indicator = OnBalanceVolumeIndicator(
            close=self.df['close'],
            volume=self.df['volume']
        )
        self.df['obv'] = indicator.on_balance_volume()
        
        logger.info("✓ Added OBV indicator")
        return self
    
    def add_price_changes(self):
        """Add price change indicators"""
        self.df['price_change'] = self.df['close'].pct_change()
        self.df['price_change_1d'] = self.df['close'].pct_change(periods=1)
        self.df['price_change_7d'] = self.df['close'].pct_change(periods=7)
        self.df['price_change_30d'] = self.df['close'].pct_change(periods=30)
        
        logger.info("✓ Added price change indicators")
        return self
    
    def add_volume_changes(self):
        """Add volume change indicators"""
        self.df['volume_change'] = self.df['volume'].pct_change()
        self.df['volume_sma_20'] = self.df['volume'].rolling(window=20).mean()
        self.df['volume_ratio'] = self.df['volume'] / self.df['volume_sma_20']
        
        logger.info("✓ Added volume change indicators")
        return self
    
    def add_momentum_indicators(self):
        """Add momentum indicators"""
        # Rate of Change
        self.df['roc_12'] = ((self.df['close'] - self.df['close'].shift(12)) / 
                             self.df['close'].shift(12)) * 100
        
        # Momentum
        self.df['momentum'] = self.df['close'] - self.df['close'].shift(10)
        
        logger.info("✓ Added momentum indicators")
        return self
    
    def add_all_indicators(self):
        """Add all technical indicators"""
        logger.info("Adding all technical indicators...")
        
        self.add_sma()
        self.add_ema()
        self.add_rsi()
        self.add_macd()
        self.add_bollinger_bands()
        self.add_stochastic()
        self.add_atr()
        self.add_obv()
        self.add_price_changes()
        self.add_volume_changes()
        self.add_momentum_indicators()
        
        # Remove NaN values
        initial_rows = len(self.df)
        self.df = self.df.dropna()
        removed_rows = initial_rows - len(self.df)
        
        logger.info(f"✓ All indicators added! Removed {removed_rows} rows with NaN values")
        logger.info(f"✓ Final dataset: {len(self.df)} rows, {len(self.df.columns)} columns")
        
        return self
    
    def get_dataframe(self):
        """Return the dataframe with indicators"""
        return self.df

def create_sample_data():
    """Create sample OHLCV data for testing"""
    dates = pd.date_range(end=pd.Timestamp.now(), periods=500, freq='D')
    
    # Generate sample price data (random walk)
    np.random.seed(42)
    price = 50000
    prices = [price]
    
    for _ in range(499):
        change = np.random.randn() * 1000
        price = max(price + change, 1000)  # Ensure positive price
        prices.append(price)
    
    df = pd.DataFrame({
        'date': dates,
        'open': prices,
        'high': [p * (1 + abs(np.random.randn() * 0.02)) for p in prices],
        'low': [p * (1 - abs(np.random.randn() * 0.02)) for p in prices],
        'close': [p * (1 + np.random.randn() * 0.01) for p in prices],
        'volume': [np.random.randint(1000000, 10000000) for _ in prices]
    })
    
    return df

def main():
    """Test technical indicators"""
    print("="*60)
    print("  Technical Indicators Calculator Test")
    print("="*60)
    
    # Create sample data
    print("\n1. Creating sample OHLCV data...")
    df = create_sample_data()
    print(f"   ✓ Created {len(df)} days of sample data")
    
    # Calculate indicators
    print("\n2. Calculating technical indicators...")
    ti = TechnicalIndicators(df)
    ti.add_all_indicators()
    
    # Get result
    result_df = ti.get_dataframe()
    
    # Display sample
    print("\n3. Sample data with indicators (last 5 rows):")
    display_cols = ['date', 'close', 'rsi', 'macd', 'bb_high', 'bb_low', 'volume_ratio']
    print(result_df[display_cols].tail())
    
    print(f"\n4. Total indicators calculated: {len(result_df.columns) - 6} indicators")
    print(f"   Columns: {', '.join(result_df.columns.tolist())}")
    
    # Save sample
    output_file = f"{config.DATA_DIR}/sample_with_indicators.csv"
    result_df.to_csv(output_file, index=False)
    print(f"\n✓ Sample data saved to {output_file}")

if __name__ == "__main__":
    main()
