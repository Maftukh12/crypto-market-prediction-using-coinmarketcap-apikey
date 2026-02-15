"""
Main Application - Crypto ML Prediction System
Interactive CLI interface for the crypto prediction system
"""
import sys
import os
from data_collector import CoinMarketCapAPI
from technical_indicators import TechnicalIndicators, create_sample_data
from feature_engineering import FeatureEngineer
from model_trainer import ModelTrainer
from predictor import CryptoPredictor
import config
from utils import setup_logger, print_section
import pandas as pd

logger = setup_logger(__name__)

class CryptoMLApp:
    """Main application class"""
    
    def __init__(self):
        """Initialize application"""
        self.api = CoinMarketCapAPI()
        
    def show_menu(self):
        """Display main menu"""
        print("\n" + "="*60)
        print("  CRYPTO ML PREDICTION SYSTEM")
        print("="*60)
        print("\n1. Test API Connection")
        print("2. Collect Market Data")
        print("3. Generate Sample Data with Indicators")
        print("4. Train ML Models")
        print("5. Make Predictions")
        print("6. View Model Performance")
        print("7. Exit")
        print("\n" + "="*60)
        
        choice = input("\nSelect option (1-7): ")
        return choice
    
    def test_api_connection(self):
        """Test CoinMarketCap API connection"""
        print_section("Testing API Connection")
        
        if self.api.test_connection():
            print("\n✓ API connection successful!")
            
            # Get sample quotes
            try:
                quotes = self.api.get_quotes_latest(['BTC', 'ETH', 'BNB'])
                print("\nSample Quotes:")
                for symbol, data in quotes.items():
                    quote = data['quote']['USD']
                    print(f"\n{symbol}:")
                    print(f"  Price: ${quote['price']:,.2f}")
                    print(f"  24h Change: {quote['percent_change_24h']:.2f}%")
            except Exception as e:
                logger.error(f"Error fetching quotes: {str(e)}")
        else:
            print("\n✗ API connection failed!")
    
    def collect_market_data(self):
        """Collect current market data"""
        print_section("Collecting Market Data")
        
        try:
            df = self.api.collect_and_save_data()
            print(f"\n✓ Collected data for {len(df)} cryptocurrencies")
            print("\nTop 10 by Market Cap:")
            print(df[['symbol', 'name', 'price', 'percent_change_24h', 'market_cap']].head(10))
        except Exception as e:
            logger.error(f"Error collecting data: {str(e)}")
    
    def generate_sample_data(self):
        """Generate sample data with technical indicators"""
        print_section("Generating Sample Data")
        
        print("\nGenerating 200 days of sample OHLCV data...")
        df = create_sample_data()
        
        print("Calculating technical indicators...")
        ti = TechnicalIndicators(df)
        ti.add_all_indicators()
        
        result_df = ti.get_dataframe()
        
        # Save
        output_file = f"{config.DATA_DIR}/sample_with_indicators.csv"
        result_df.to_csv(output_file, index=False)
        
        print(f"\n✓ Generated {len(result_df)} rows with {len(result_df.columns)} columns")
        print(f"✓ Saved to {output_file}")
        
        print("\nSample data (last 5 rows):")
        display_cols = ['date', 'close', 'rsi', 'macd', 'bb_high', 'bb_low']
        print(result_df[display_cols].tail())
    
    def train_models(self):
        """Train all ML models"""
        print_section("Training ML Models")
        
        data_file = f"{config.DATA_DIR}/sample_with_indicators.csv"
        
        if not os.path.exists(data_file):
            print("\n✗ Data file not found!")
            print("Please generate sample data first (Option 3)")
            return
        
        print("\nThis will train all ML models (LSTM, Random Forest, XGBoost)")
        print("This may take several minutes...")
        
        confirm = input("\nContinue? (y/n): ")
        if confirm.lower() != 'y':
            print("Training cancelled")
            return
        
        try:
            trainer = ModelTrainer(data_file)
            trainer.load_and_prepare_data()
            
            # Train models
            trainer.train_random_forest()
            trainer.train_xgboost_classifier()
            trainer.train_xgboost_regressor()
            trainer.train_lstm()
            
            # Save scaler
            trainer.save_scaler()
            
            # Generate report
            trainer.generate_report()
            
            print("\n✓ All models trained successfully!")
            
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
    
    def make_predictions(self):
        """Make market predictions"""
        print_section("Making Predictions")
        
        try:
            predictor = CryptoPredictor()
            
            print("\nSelect prediction mode:")
            print("1. Single cryptocurrency")
            print("2. Multiple cryptocurrencies")
            
            mode = input("\nSelect mode (1-2): ")
            
            if mode == '1':
                symbol = input("Enter cryptocurrency symbol (e.g., BTC): ").upper()
                predictor.predict_single(symbol)
            elif mode == '2':
                print("\nPredicting top 5 cryptocurrencies...")
                results = predictor.predict_multiple()
                
                print("\n" + "="*60)
                print("  SUMMARY")
                print("="*60)
                for symbol, result in results.items():
                    ensemble = result['ensemble']
                    print(f"\n{symbol}: {ensemble['direction']} (Confidence: {ensemble['confidence']:.2%})")
            
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            print("\n✗ Please train models first (Option 4)")
    
    def view_performance(self):
        """View model performance report"""
        print_section("Model Performance Report")
        
        report_file = f"{config.MODELS_DIR}/training_report.txt"
        
        if os.path.exists(report_file):
            with open(report_file, 'r') as f:
                print(f.read())
        else:
            print("\n✗ Performance report not found!")
            print("Please train models first (Option 4)")
    
    def run(self):
        """Run the application"""
        while True:
            try:
                choice = self.show_menu()
                
                if choice == '1':
                    self.test_api_connection()
                elif choice == '2':
                    self.collect_market_data()
                elif choice == '3':
                    self.generate_sample_data()
                elif choice == '4':
                    self.train_models()
                elif choice == '5':
                    self.make_predictions()
                elif choice == '6':
                    self.view_performance()
                elif choice == '7':
                    print("\nExiting... Goodbye!")
                    break
                else:
                    print("\n✗ Invalid option! Please select 1-7")
                
                input("\nPress Enter to continue...")
                
            except KeyboardInterrupt:
                print("\n\nExiting... Goodbye!")
                break
            except Exception as e:
                logger.error(f"Error: {str(e)}")
                input("\nPress Enter to continue...")

def main():
    """Main entry point"""
    print("="*60)
    print("  CRYPTO ML PREDICTION SYSTEM")
    print("  Machine Learning for Cryptocurrency Market Analysis")
    print("="*60)
    print("\nInitializing...")
    
    app = CryptoMLApp()
    app.run()

if __name__ == "__main__":
    main()
