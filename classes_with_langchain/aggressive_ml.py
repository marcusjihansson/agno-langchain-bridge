# Stock price analysis and trading signal generation

# This is a Python class that creates a trading strategy for a Aggressive Trading Strategy
# This is a Machine Learning trading model to maximize profits
# This does not use confidence intervals to determine if the model should trade or not 
# Further explanation on these confidence intervals, please check Defensive Trading Strategy

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import ta
from typing import Union, List, Dict, Tuple
import logging

import datetime
from datetime import timedelta, datetime
from zoneinfo import ZoneInfo
from alpaca.data.timeframe import TimeFrame
from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.historical.crypto import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from ta.momentum import RSIIndicator
from ta.trend import MACD, ADXIndicator, SMAIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator, AccDistIndexIndicator
import pprint 

from langchain_core.output_parsers.string import StrOutputParser
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_ollama import ChatOllama

# Safely access API keys using .get()
import os

alpaca_key = os.getenv("ALPACA_KEY")
alpaca_secret = os.getenv("ALPACA_SECRET")


class DataFetcher_Crypto:
    def __init__(self, alpaca_key: str, alpaca_secret: str):
        """
        Initialize the StockAnalyzer with Alpaca API credentials.
        """
        self.alpaca_key = alpaca_key
        self.alpaca_secret = alpaca_secret

    def fetch_ohlcv(self, symbol: str, days: int = 30, timeframe: TimeFrame = TimeFrame.Day, limit: int = None) -> Union[pd.DataFrame, str]:
        """
        Fetch historical stock bars from Alpaca API.
        """
        try:
            # Create Alpaca client
            client = CryptoHistoricalDataClient(self.alpaca_key, self.alpaca_secret)

            # Set timezone to New York
            now = datetime.now(ZoneInfo("America/New_York"))

            # Prepare request parameters
            if limit is None:
                limit = days

            req = CryptoBarsRequest(
                symbol_or_symbols=[symbol],
                timeframe=timeframe,
                start=now - timedelta(days=days),
                limit=limit
            )

            # Fetch stock bars
            bars = client.get_crypto_bars(req)
            df = bars.df

            # Ensure VWAP is included
            if "vwap" not in df.columns:
                return "Error: VWAP data not available in fetched data."

            # Reset index and parse timestamp as datetime
            df = df.reset_index()
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)  # Set timestamp as the index

            return df  # Return as a DataFrame for further processing

        except Exception as e:
            return f"Error fetching OHLCV data: {str(e)}"


class DataFetcher_Stock:
    def __init__(self, alpaca_key: str, alpaca_secret: str):
        """
        Initialize the StockAnalyzer with Alpaca API credentials.
        """
        self.alpaca_key = alpaca_key
        self.alpaca_secret = alpaca_secret

    def fetch_ohlcv(self, symbol: str, days: int = 30, timeframe: TimeFrame = TimeFrame.Day, limit: int = None) -> Union[pd.DataFrame, str]:
        """
        Fetch historical stock bars from Alpaca API.
        """
        try:
            # Create Alpaca client
            client = StockHistoricalDataClient(self.alpaca_key, self.alpaca_secret)

            # Set timezone to New York
            now = datetime.now(ZoneInfo("America/New_York"))

            # Prepare request parameters
            if limit is None:
                limit = days

            req = StockBarsRequest(
                symbol_or_symbols=[symbol],
                timeframe=timeframe,
                start=now - timedelta(days=days),
                limit=limit
            )

            # Fetch stock bars
            bars = client.get_stock_bars(req)
            df = bars.df

            # Ensure VWAP is included
            if "vwap" not in df.columns:
                return "Error: VWAP data not available in fetched data."

            # Reset index and parse timestamp as datetime
            df = df.reset_index()
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)  # Set timestamp as the index

            return df  # Return as a DataFrame for further processing

        except Exception as e:
            return f"Error fetching OHLCV data: {str(e)}"

class AggressiveMLTradingStrategy:
    def __init__(self, analyzer, symbol: str, days: int = 365, prediction_window: int = 5):
        """
        Initialize the ML-based trading strategy.
        
        Args:
            analyzer: An instance of StockAnalyzer
            symbol: The stock symbol to analyze
            days: Number of days of historical data to use
            prediction_window: Days ahead to predict (target variable)
        """
        self.analyzer = analyzer
        self.symbol = symbol
        self.days = days
        self.prediction_window = prediction_window
        self.model = None
        self.scaler = StandardScaler()
        self.logger = self._setup_logger()
        self.features = []
        
    def _setup_logger(self):
        """Set up logging for the strategy."""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger
    
    def fetch_and_prepare_data(self) -> Union[pd.DataFrame, str]:
        """
        Fetch data using the analyzer and prepare it for training.
        
        Returns:
            DataFrame with price data and technical indicators or error message
        """
        # Fetch historical data
        self.logger.info(f"Fetching {self.days} days of data for {self.symbol}")
        data = self.analyzer.fetch_ohlcv(self.symbol, days=self.days)
        
        if isinstance(data, str):
            self.logger.error(f"Failed to fetch data: {data}")
            return data
            
        # Ensure we have enough data
        if len(data) < 50:
            error_msg = f"Insufficient data points for analysis: {len(data)} < 50"
            self.logger.error(error_msg)
            return error_msg
            
        # Add technical indicators
        self.logger.info("Adding technical indicators")
        enriched_data = self._add_technical_indicators(data)
        
        # Create target variable (future price direction)
        self.logger.info(f"Creating target variable with {self.prediction_window}-day prediction window")
        enriched_data = self._create_target_variable(enriched_data)
        
        # Drop rows with NaN values (resulting from indicator calculations)
        cleaned_data = enriched_data.dropna()
        self.logger.info(f"Cleaned data shape: {cleaned_data.shape}")
        
        return cleaned_data
    
    def _add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add various technical indicators to the dataset.
        """
        df = data.copy()

        # Price-based indicators
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))

        # Moving Averages
        df['sma_5'] = SMAIndicator(df['close'], window=5).sma_indicator()
        df['sma_20'] = SMAIndicator(df['close'], window=20).sma_indicator()
        df['sma_50'] = SMAIndicator(df['close'], window=50).sma_indicator()

        # MA Crossover Features
        df['ma_crossover_5_20'] = (df['sma_5'] > df['sma_20']).astype(int)

        # Momentum Indicators
        df['rsi_14'] = RSIIndicator(df['close'], window=14).rsi()

        macd = MACD(df['close'], window_slow=26, window_fast=12, window_sign=9)
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_hist'] = macd.macd_diff()

        # Volatility Indicators
        df['atr_14'] = AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()

        bb = BollingerBands(df['close'], window=20, window_dev=2)
        df['bollinger_upper'] = bb.bollinger_hband()
        df['bollinger_middle'] = bb.bollinger_mavg()
        df['bollinger_lower'] = bb.bollinger_lband()

        # Volume-Based Indicators
        df['obv'] = OnBalanceVolumeIndicator(df['close'], df['volume']).on_balance_volume()
        df['ad'] = AccDistIndexIndicator(df['high'], df['low'], df['close'], df['volume']).acc_dist_index()

        # Trend Strength
        df['adx'] = ADXIndicator(df['high'], df['low'], df['close'], window=14).adx()

        # Price Relative to Indicators
        df['price_to_sma_20'] = df['close'] / df['sma_20'] - 1
        df['bb_position'] = (df['close'] - df['bollinger_lower']) / (df['bollinger_upper'] - df['bollinger_lower'])

        # Store feature names
        self.features = [
            'returns', 'log_returns',
            'sma_5', 'sma_20', 'sma_50', 'ma_crossover_5_20',
            'rsi_14', 'macd', 'macd_signal', 'macd_hist',
            'atr_14', 'bollinger_upper', 'bollinger_middle', 'bollinger_lower',
            'obv', 'ad', 'adx', 'price_to_sma_20', 'bb_position',
            'open', 'high', 'low', 'close', 'volume', 'vwap'
        ]

        # Drop NaNs caused by indicator calculations
        return df.dropna()

    
    def _create_target_variable(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create the target variable based on future price movement.
        
        Args:
            data: DataFrame with price and indicator data
            
        Returns:
            DataFrame with added target variable
        """
        df = data.copy()
        
        # Future price
        df['future_price'] = df['close'].shift(-self.prediction_window)
        
        # Target: 1 if price goes up, 0 if it goes down
        df['target'] = (df['future_price'] > df['close']).astype(int)
        
        return df
    
    def train_model(self, data: pd.DataFrame) -> Dict:
        """
        Train a machine learning model on the prepared data.
        
        Args:
            data: DataFrame with features and target
            
        Returns:
            Dictionary with training results
        """
        # Remove rows where we don't have a target
        df = data.dropna(subset=['target'])
        
        # Select features and target
        X = df[self.features]
        y = df['target']
        
        # Split into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        
        # Scale features
        self.logger.info("Scaling features")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train the model
        self.logger.info("Training Random Forest model")
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate the model
        y_pred = self.model.predict(X_test_scaled)
        
        results = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'feature_importance': dict(zip(self.features, self.model.feature_importances_)),
            'train_samples': len(X_train),
            'test_samples': len(X_test)
        }
        
        self.logger.info(f"Model training results: accuracy={results['accuracy']:.4f}, precision={results['precision']:.4f}")
        
        print(results.keys())
        
        return results 
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on model predictions.
        
        Args:
            data: DataFrame with features
            
        Returns:
            DataFrame with added prediction and signal columns
        """
        if self.model is None:
            self.logger.error("Model not trained. Call train_model first.")
            return data
            
        df = data.copy()
        
        # Select features and scale
        X = df[self.features].dropna()
        X_scaled = self.scaler.transform(X)
        
        # Generate predictions and probabilities
        df.loc[X.index, 'prediction'] = self.model.predict(X_scaled)
        proba = self.model.predict_proba(X_scaled)
        df.loc[X.index, 'prediction_probability'] = [p[1] for p in proba]
        
        # Generate signals (1 for buy, -1 for sell, 0 for hold)
        # Only generate signal when probability exceeds threshold
        df['signal'] = 0
        df.loc[df['prediction_probability'] > 0.6, 'signal'] = 1
        df.loc[df['prediction_probability'] < 0.4, 'signal'] = -1
        
        return df
    
    def backtest(self, data: pd.DataFrame, initial_capital: float = 10000.0) -> Dict:
        """
        Backtest the strategy on historical data.
        
        Args:
            data: DataFrame with trading signals
            initial_capital: Starting capital for the backtest
            
        Returns:
            Dictionary with backtest results
        """
        df = data.copy()
        
        # Initialize backtest columns
        df['position'] = 0
        df['position_value'] = 0.0
        df['cash'] = initial_capital
        df['portfolio_value'] = initial_capital
        
        # Get signals
        signals = df['signal'].fillna(0)
        
        # Iterate through the DataFrame to implement the strategy
        position = 0
        cash = initial_capital
        
        for i in range(1, len(df)):
            date = df.index[i]
            prev_date = df.index[i-1]
            
            # Get the closing price and signal
            close_price = df.loc[date, 'close']
            signal = signals[prev_date]  # Using previous day's signal
            
            # Update position based on signal
            if signal == 1 and position == 0:  # Buy signal
                # Buy as many shares as possible
                shares_to_buy = int(cash / close_price)
                position += shares_to_buy
                cash -= shares_to_buy * close_price
            elif signal == -1 and position > 0:  # Sell signal
                # Sell all shares
                cash += position * close_price
                position = 0
            
            # Update DataFrame
            df.loc[date, 'position'] = position
            df.loc[date, 'position_value'] = position * close_price
            df.loc[date, 'cash'] = cash
            df.loc[date, 'portfolio_value'] = position * close_price + cash
        
        # Calculate returns
        df['strategy_returns'] = df['portfolio_value'].pct_change()
        df['buy_hold_returns'] = df['close'].pct_change()
        
        # Calculate cumulative returns
        df['cumulative_strategy_returns'] = (1 + df['strategy_returns']).cumprod() - 1
        df['cumulative_buy_hold_returns'] = (1 + df['buy_hold_returns']).cumprod() - 1
        
        # Calculate performance metrics
        total_days = (df.index[-1] - df.index[0]).days
        years = total_days / 365.25
        
        strategy_return = df['cumulative_strategy_returns'].iloc[-1]
        buy_hold_return = df['cumulative_buy_hold_returns'].iloc[-1]
        
        strategy_annual_return = (1 + strategy_return) ** (1 / years) - 1
        buy_hold_annual_return = (1 + buy_hold_return) ** (1 / years) - 1
        
        strategy_volatility = df['strategy_returns'].std() * (252 ** 0.5)  # Annualized
        buy_hold_volatility = df['buy_hold_returns'].std() * (252 ** 0.5)  # Annualized
        
        strategy_sharpe = strategy_annual_return / strategy_volatility if strategy_volatility != 0 else 0
        buy_hold_sharpe = buy_hold_annual_return / buy_hold_volatility if buy_hold_volatility != 0 else 0
        
        # Count trades
        position_changes = df['position'].diff().fillna(0)
        buys = len(position_changes[position_changes > 0])
        sells = len(position_changes[position_changes < 0])
        
        backtest_results = {
            'final_portfolio_value': df['portfolio_value'].iloc[-1],
            'total_return': strategy_return,
            'buy_hold_return': buy_hold_return,
            'annual_return': strategy_annual_return,
            'buy_hold_annual_return': buy_hold_annual_return,
            'sharpe_ratio': strategy_sharpe,
            'buy_hold_sharpe_ratio': buy_hold_sharpe,
            'volatility': strategy_volatility,
            'buy_hold_volatility': buy_hold_volatility,
            'num_trades': buys + sells,
            'win_rate': None  # Requires individual trade analysis
        }
        
        return backtest_results, df
    
    def plot_results(self, backtest_df: pd.DataFrame, save_path: str = None):
        """
        Plot backtest results including portfolio value and cumulative returns comparison.
        
        Args:
            backtest_df: DataFrame with backtest results
            save_path: Optional path to save the plot
        """
        plt.figure(figsize=(12, 10))
        
        # Plot 1: Portfolio Value
        plt.subplot(2, 1, 1)
        plt.plot(backtest_df['portfolio_value'], label='Portfolio Value')
        plt.title('Portfolio Value Over Time')
        plt.xlabel('Date')
        plt.ylabel('Value ($)')
        plt.legend()
        plt.grid(True)
        
        # Plot 2: Strategy vs Buy & Hold
        plt.subplot(2, 1, 2)
        plt.plot(backtest_df['cumulative_strategy_returns'], label='ML Strategy')
        plt.plot(backtest_df['cumulative_buy_hold_returns'], label='Buy & Hold')
        plt.title('Cumulative Returns: Strategy vs Buy & Hold')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Returns (%)')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            
        plt.show()
    
    def run_strategy(self) -> Tuple[Dict, pd.DataFrame]:
        """
        Run the complete trading strategy workflow.
        
        Returns:
            Tuple of (combined results, backtest dataframe)
        """
        # Fetch and prepare data
        data = self.fetch_and_prepare_data()
        if isinstance(data, str):
            self.logger.error(f"Strategy failed: {data}")
            return None, None
        
        # Train model
        training_results = self.train_model(data)
        self.logger.info(f"Model training complete with accuracy: {training_results['accuracy']:.4f}")
        
        # Generate trading signals
        signal_data = self.generate_signals(data)
        self.logger.info(f"Generated signals for {len(signal_data)} data points")
        
        # Backtest the strategy
        backtest_results, backtest_df = self.backtest(signal_data)
        self.logger.info(f"Backtest complete. Final portfolio value: ${backtest_results['final_portfolio_value']:.2f}")
        self.logger.info(f"Strategy return: {backtest_results['total_return']*100:.2f}% vs Buy & Hold: {backtest_results['buy_hold_return']*100:.2f}%")
        
        # Combine results
        combined_results = {**backtest_results, **training_results}
        
        return combined_results, backtest_df


# Example usage
if __name__ == "__main__":
    from alpaca.data.timeframe import TimeFrame
    from zoneinfo import ZoneInfo
    from datetime import datetime, timedelta
    
    # Initialize the StockAnalyzer (assuming it's available)
    analyzer = DataFetcher_Crypto(alpaca_key, alpaca_secret)
    
    # Create and run the ML trading strategy
    strategy = AggressiveMLTradingStrategy(analyzer, symbol="ETH/USD", days=365*2)  # 2 years of data
    
    # Run the complete strategy
    combined_results, backtest_data = strategy.run_strategy()
    
    # Plot results
    if backtest_data is not None:
        strategy.plot_results(backtest_data)
        
        # Print key performance metrics
        print("\nPerformance Summary:")
        print(f"Final Portfolio Value: ${combined_results['final_portfolio_value']:.2f}")
        print(f"Total Return: {combined_results['total_return']*100:.2f}%")
        print(f"Buy & Hold Return: {combined_results['buy_hold_return']*100:.2f}%")
        print(f"Annual Return: {combined_results['annual_return']*100:.2f}%")
        print(f"Sharpe Ratio: {combined_results['sharpe_ratio']:.4f}")
        print(f"Number of Trades: {combined_results['num_trades']}")
        
        # Print top 5 most important features
        sorted_features = sorted(
            combined_results['feature_importance'].items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:5]
        print("\nTop 5 Important Features:")
        for feature, importance in sorted_features:
            print(f"{feature}: {importance:.4f}")