# This is a tool file where all Langchain based tools are converted to run with Agno-agi
# The conversion is done by running the langchain based class as a Python function 
# This Python function is then able to be run as a tool inside Agno-agi Agents

# This is a bunch of functions I have tested and run. These works but, remember: 
    # 1. This is not financial advice, do not blindly trust the tools as they need optimizations 
    # 2. These tools are moslty testing tools and in some cases are the data collection not completely fair, as;
        # in the case for the LegalAnalyzer is made by a very simple RAG and the data was collected from some PDF docs
        # in the case for the GeopoliticsAnalyzer is made by simple RAG on pandas dataframes and the data was collected by 
                        # finding data on the internet by sort of viable sources like the World Risk Index 
    # 3. I would argue that the data collection is fine, as the project as a whole was made as an test project for me to:
        # learn these tools and see if there was a possiblity to build Agents with different tools where an LLM was 
        # used both inside the tool and by the Agent itself.
    # 4. Interestingly enough was this doable as we could run these tools as Python functions with Agno-agi 

# Libraries
from typing import Dict
import time

from langchain_ollama import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
import json
import ccxt
import pandas as pd
from textblob import TextBlob

from lang_classes.insider_sentiment_tool import Senttiment_Data_Tool, InsiderAnalysisTool
from lang_classes.sentiment import ContextDataTool
from lang_classes.macro import MacroAnalyzer
from lang_classes.optiions import OptionsAnalysisLLM
from lang_classes.terminal_value import ValuationCalculator
from lang_classes.legal_analysis import LegalAnalyzer
from lang_classes.geo_analysis import GeopoliticsAnalyzer
from lang_classes.stocks_ta import StockAnalyzer
from lang_classes.fundamental import AlphaVantageClient
from lang_classes.blockchain_tool import BlockchainMetricsAnalyzer
from lang_classes.crypto_market_tool import CryptoMarketAnalyzer
from lang_classes.aggressive_ml import DataFetcher_Crypto, DataFetcher_Stock, AggressiveMLTradingStrategy
from lang_classes.defensive_ml import StockDataFetcher, CryptoDataFetcher, DefensiveTradingStrategy
from lang_classes.asset_price import AssetPriceFetcher
from lang_classes.supply_chain import SupplyChainAnalyzer
from lang_classes.text_embeddings import TextFileProcessor

# Safely access API keys using .get()
import os 

finnhub_key = os.getenv('FINNHUB_KEY')
news_key = os.getenv("FINLIGHT_API_KEY")
alpaca_key = os.getenv("ALPACA_KEY")
alpaca_secret = os.getenv("ALPACA_SECRET")
alpha_vantage_key = os.getenv("ALPHA_VANTAGE_API_KEY")

# Get API credentials (ensure these are strings, not tuples)
bybit_key = os.getenv('BYBIT_API_KEY')
bybit_secret = os.getenv('BYBIT_SECRET_KEY')

# API Key setup
exchange = ccxt.bybit({
    'apiKey': bybit_key,
    'secret': bybit_secret,
    'enableRateLimit': True,
})


# Initialize the gemma model
gemma = ChatOllama(
    model="gemma3:1b",
)

## OUTSIDE FACTORS: MACRO, LEGAL, GEOPOLITICS

# Macro Analysis Tool
def macro_analysis_tool(symbol: str) -> str:
    """
    Uses a LangChain-based LLM to perform macro analysis and returns the analysis.
    """
    try:
        llm = ChatOllama(model="qwen2.5:1.5b")
        tool = MacroAnalyzer(llm)
        tool.collect_data()
        macro_result = tool.analyze_ticker_macro_impact(symbol)
        return macro_result
    except Exception as e:
        return f"Data error: {str(e)}" 
    
# Legal Analyzer Tool
def legal_analysis_tool(symbol: str) -> str:
    """
    Uses a LangChain-based LLM to perform legal analysis and returns the analysis.
    """
    try:
        llm = ChatOllama(model="qwen2.5:1.5b")
        tool = LegalAnalyzer(llm)
        legal_result = tool.analyze(symbol)
        return legal_result['answer']
    except Exception as e:
        return f"Data error: {str(e)}" 

# Geopolitics Analyzer Tool
def geopolitics_analysis_tool(symbol: str) -> str:
    """Analyzes geopolitical risks and their impact on investments."""
    try: 
        llm = ChatOllama(model="qwen2.5:1.5b")
        tool = GeopoliticsAnalyzer(llm)
        geo_insights=tool.analyze(symbol)
        return geo_insights
    except Exception as e:
        return f"Data error: {str(e)}" 

# Supply Chain Analyzer Tool
def supply_chain_analysis_tool(symbol: str) -> str:
    """Analyzes geopolitical risks and their impact on investments."""
    try: 
        llm = ChatOllama(model="qwen2.5:1.5b")
        tool = SupplyChainAnalyzer(llm)
        supply_chain_analysis=tool.analyze(symbol)
        return supply_chain_analysis 
    except Exception as e:
        return f"Data error: {str(e)}" 



## SENTIMENT ANALYSIS (INSIDER SENTIMNET AND GENERAL SENTIMENT ANALYSIS FOR A TICKER)

# Insider Analysis Tool
def insider_analysis_tool(symbol: str, start_date: str, end_date: str, max_articles: int = 20) -> Dict:
    """Analyzes sentiment for a stock based on insider sentiment and text analysis."""
    try:
        llm = ChatOllama(model="qwen2.5:1.5b")
        data_tool = Senttiment_Data_Tool(api_key=finnhub_key)
        tool = InsiderAnalysisTool(data_tool, llm)
        insider_analysis = tool.analyze_insider_sentiment(symbol, start_date, end_date, max_articles)
        return {"analysis": insider_analysis}
    except Exception as e:
        return f"Data error: {str(e)}" 

# Sentiment Analysis Tool
def sentiment_analysis_tool(symbol: str, start_date: str, end_date: str) -> str:
    """Fetches news and analyzes context for a stock."""
    try:
        tool = ContextDataTool(news_key)
        llm = ChatOllama(model="qwen2.5:1.5b")
        context_data = tool.get_ticker_context(symbol, start_date, end_date, llm)
        return {"analysis": context_data}
    except Exception as e:
        return f"Data error: {str(e)}" 



# ASSET PRICE FETCHER TOOL 

# Fetch asset price for Stocks
def get_asset_price_stock(symbol: str) -> str:
    """Analyzes technical indicators for an asset to create a trading or investment strategy."""
    try:
        analyzer = AssetPriceFetcher(alpaca_key, alpaca_secret)
        asset_price=analyzer.fetch_ohlcv(symbol)
        return {"analysis": asset_price}
    except Exception as e:
        return f"Data error: {str(e)}" 

# Fetch asset price for Crypto
def get_asset_price_crypto(symbol: str) -> str:
    """Analyzes technical indicators for an asset to create a trading or investment strategy."""
    try:
        analyzer = AssetPriceFetcher(alpaca_key, alpaca_secret)
        asset_price=analyzer.fetch_ohlcv_crypto(symbol)
        return {"analysis": asset_price}
    except Exception as e:
        return f"Data error: {str(e)}" 


## TRADING BASED TOOLS

# Technical Analysis Tool
def technical_analysis_tool(symbol: str) -> str:
    """Analyzes technical indicators for an asset to create a trading or investment strategy."""
    try:
        analyzer = StockAnalyzer(alpaca_key, alpaca_secret)
        llm = ChatOllama(model="qwen2.5:1.5b")
        trading_signals=analyzer.get_technical_analysis(symbol, llm)
        return {"analysis": trading_signals}
    except Exception as e:
        return f"Data error: {str(e)}" 

# Options Analysis Tool
def options_analysis_tool(symbol: str, expiration_date: str) -> str:
    """Analyzes options data for a given stock and expiration date."""
    try:
        llm = ChatOllama(model="qwen2.5:1.5b")
        analyzer = OptionsAnalysisLLM(llm)
        insights = analyzer.run_analysis(symbol, expiration_date)
        return insights
    except Exception as e:
        return f"Data error: {str(e)}" 

# Machine Learning Models for trading (Maximize Profit Focused) (Stocks)
def aggressive_ml_trading_strategy_stock(symbol: str, backtesting_days: int = 365*2) -> dict:
    """
    Analyzes a stock using an aggressive ML trading strategy and returns performance metrics.
    
    Args:
        symbol (str): The stock ticker symbol (e.g., "AAPL")
        backtesting_days (int): Number of days to use for backtesting (default: 2 years)
        alpaca_key (str): Alpaca API key
        alpaca_secret (str): Alpaca API secret
    
    Returns:
        dict: Dictionary containing analysis results and backtest data
        
    Example:
        results = analyze_ml_trading_strategy("AAPL")
        print(f"Total Return: {results['analysis']['total_return']*100:.2f}%")
    """
    try:
        # Initialize the DataFetcher
        analyzer = DataFetcher_Stock(alpaca_key, alpaca_secret)
        
        # Create and run the ML trading strategy
        strategy = AggressiveMLTradingStrategy(analyzer, symbol, backtesting_days)
        
        # Run the complete strategy
        combined_results, backtest_data = strategy.run_strategy()
        
        # Plot results if backtest data is available
        if backtest_data is not None:
            strategy.plot_results(backtest_data)
        
        # Format performance summary
        performance_summary = {
            "final_portfolio_value": f"${combined_results['final_portfolio_value']:.2f}",
            "total_return": f"{combined_results['total_return']*100:.2f}%",
            "buy_hold_return": f"{combined_results['buy_hold_return']*100:.2f}%",
            "annual_return": f"{combined_results['annual_return']*100:.2f}%",
            "sharpe_ratio": f"{combined_results['sharpe_ratio']:.4f}",
            "num_trades": combined_results['num_trades']
        }
        
        # Format top 5 features
        sorted_features = sorted(
            combined_results['feature_importance'].items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        top_features = {}
        for feature, importance in sorted_features:
            top_features[feature] = f"{importance:.4f}"
        
        # Return comprehensive results
        return {
            "analysis": combined_results,
            "backtest_data": backtest_data,
            "performance_summary": performance_summary,
            "top_features": top_features,
            "status": "success"
        }
    
    except Exception as e:
        return {
            "status": "error",
            "message": f"Analysis failed: {str(e)}",
            "analysis": None,
            "backtest_data": None
        }

# Machine Learning Models for trading (Minimize Loss Focused) (Stocks)
def defensive_ml_trading_strategy_stock(symbol: str, backtesting_days: int = 365*2) -> dict:
    """
    Analyzes a stock using an aggressive ML trading strategy and returns performance metrics.
    
    Args:
        symbol (str): The stock ticker symbol (e.g., "AAPL")
        backtesting_days (int): Number of days to use for backtesting (default: 2 years)
        alpaca_key (str): Alpaca API key
        alpaca_secret (str): Alpaca API secret
    
    Returns:
        dict: Dictionary containing analysis results and backtest data
        
    Example:
        results = analyze_ml_trading_strategy("AAPL")
        print(f"Total Return: {results['analysis']['total_return']*100:.2f}%")
    """
    try:
        # Initialize the DataFetcher
        analyzer = StockDataFetcher(alpaca_key, alpaca_secret)
        
        # Create and run the ML trading strategy
        strategy = DefensiveTradingStrategy(analyzer, symbol, days=backtesting_days)
        
        # Run the complete strategy
        combined_results, backtest_data = strategy.run_strategy()
        
        # Plot results if backtest data is available
        if backtest_data is not None:
            strategy.plot_results(backtest_data)
        
        # Format performance summary
        performance_summary = {
            "final_portfolio_value": f"${combined_results['final_portfolio_value']:.2f}",
            "total_return": f"{combined_results['total_return']*100:.2f}%",
            "buy_hold_return": f"{combined_results['buy_hold_return']*100:.2f}%",
            "annual_return": f"{combined_results['annual_return']*100:.2f}%",
            "sharpe_ratio": f"{combined_results['sharpe_ratio']:.4f}",
            "num_trades": combined_results['num_trades']
        }
        
        # Format top 5 features
        sorted_features = sorted(
            combined_results['feature_importance'].items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        top_features = {}
        for feature, importance in sorted_features:
            top_features[feature] = f"{importance:.4f}"
        
        # Return comprehensive results
        return {
            "analysis": combined_results,
            "backtest_data": backtest_data,
            "performance_summary": performance_summary,
            "top_features": top_features,
            "status": "success"
        }
    
    except Exception as e:
        return {
            "status": "error",
            "message": f"Analysis failed: {str(e)}",
            "analysis": None,
            "backtest_data": None
        }


## MICRO ECONOMICS

# Financial Statements Tool with Rate Limit Handling
def fundamental_analysis_tool(symbol: str) -> str:
    """Analyzes fundamental compandy data (balance sheets, income statements, cash flow statements, earnings) for investment insights."""
    try:
        llm = ChatOllama(model="qwen2.5:1.5b")
        client = AlphaVantageClient(alpha_vantage_key)
        fundamental_data=client.get_fundamental_analysis(symbol, llm)
        return fundamental_data
    except Exception as e:
        return f"Data error: {str(e)}" 

# Valuation Calculator Tool
def valuation_calculator_tool(symbol: str) -> str:
    """Calculates the intrinsic value of a stock."""
    try:
        tool = ValuationCalculator(alpha_vantage_key, alpaca_key, alpaca_secret, symbol)
        llm = ChatOllama(model="qwen2.5:1.5b")
        tv = tool.company_valuation(llm)
        return tv
    except Exception as e:
        return f"Data error: {str(e)}" 


## BLOCKCHAIN BASED TOOLS

# On-Chain analysis 
def get_onchain_analysis(symbol_name: str, symbol_name_capital: str) -> str:
    """Fetches on-chain metrics for a token.
        This function needs the token name in two formats as it fetches data form both coingecko and defillama 
        eg. 'bitcoin', 'Bitcoin'
    """
    try:
        analyzer = BlockchainMetricsAnalyzer(symbol_name, symbol_name_capital)
        llm = ChatOllama(model="qwen2.5:1.5b")
        metrics = analyzer.run_analysis()
        on_chain_analysis = analyzer.analyze_metrics(metrics, llm) 
        return on_chain_analysis
    except Exception as e:
        return f"Data error: {str(e)}" 

# Market metrics, such as liquidity illiquidity analysis and risk
def get_market_metrics(symbol: str, timeframe: str = '1d', limit: int = 5) -> dict:
    """
    Tool function for Agno-AGI to perform cryptocurrency market analysis.
    
    Args:
        symbol (str): Trading pair symbol (e.g., 'BTC/USDT').
        timeframe (str): Candle timeframe (default: '1d').
        limit (int): Number of candles to fetch (default: 250).
    
    Returns:
        dict: Structured analysis results.
    """
    try:
        # Initialize the analyzer with the given symbol
        analyzer = CryptoMarketAnalyzer(
            exchange_name='bybit',
            symbol=symbol,
            bybit_key=os.getenv('BYBIT_API_KEY'),
            bybit_secret=os.getenv('BYBIT_SECRET_KEY')
        )
        
        # Perform comprehensive analysis
        analysis_results = analyzer.run_analysis(timeframe=timeframe, limit=limit)

        if not analysis_results:
            return {"error": "Analysis could not be completed."}

        # Convert Timestamps to strings (if any exist)
        def convert_timestamps(obj):
            if isinstance(obj, pd.Timestamp):
                return obj.isoformat()  # Convert to string
            return obj

        # Apply conversion recursively if needed
        def clean_dict(d):
            if isinstance(d, dict):
                return {key: clean_dict(value) for key, value in d.items()}
            elif isinstance(d, list):
                return [clean_dict(item) for item in d]
            else:
                return convert_timestamps(d)

        return clean_dict(analysis_results)
    
    except Exception as e:
        return {"error": f"An error occurred: {str(e)}"}

# Machine Learning Models for trading (Maximize Profit Focused) (Crypto)
def aggresive_ml_trading_strategy_crypto(symbol: str, backtesting_days: int = 365*2) -> dict:
    """
    Analyzes a stock using an aggressive ML trading strategy and returns performance metrics.
    
    Args:
        symbol (str): The stock ticker symbol (e.g., "AAPL")
        backtesting_days (int): Number of days to use for backtesting (default: 2 years)
        alpaca_key (str): Alpaca API key
        alpaca_secret (str): Alpaca API secret
    
    Returns:
        dict: Dictionary containing analysis results and backtest data
        
    Example:
        results = analyze_ml_trading_strategy("AAPL")
        print(f"Total Return: {results['analysis']['total_return']*100:.2f}%")
    """
    try:
        # Initialize the DataFetcher
        analyzer = DataFetcher_Crypto(alpaca_key, alpaca_secret)
        
        # Create and run the ML trading strategy
        strategy = AggressiveMLTradingStrategy(analyzer, symbol, days=backtesting_days)
        
        # Run the complete strategy
        combined_results, backtest_data = strategy.run_strategy()
        
        # Plot results if backtest data is available
        if backtest_data is not None:
            strategy.plot_results(backtest_data)
        
        # Format performance summary
        performance_summary = {
            "final_portfolio_value": f"${combined_results['final_portfolio_value']:.2f}",
            "total_return": f"{combined_results['total_return']*100:.2f}%",
            "buy_hold_return": f"{combined_results['buy_hold_return']*100:.2f}%",
            "annual_return": f"{combined_results['annual_return']*100:.2f}%",
            "sharpe_ratio": f"{combined_results['sharpe_ratio']:.4f}",
            "num_trades": combined_results['num_trades']
        }
        
        # Format top 5 features
        sorted_features = sorted(
            combined_results['feature_importance'].items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        top_features = {}
        for feature, importance in sorted_features:
            top_features[feature] = f"{importance:.4f}"
        
        # Return comprehensive results
        return {
            "analysis": combined_results,
            "backtest_data": backtest_data,
            "performance_summary": performance_summary,
            "top_features": top_features,
            "status": "success"
        }
    
    except Exception as e:
        return {
            "status": "error",
            "message": f"Analysis failed: {str(e)}",
            "analysis": None,
            "backtest_data": None
        }

# Machine Learning Models for trading (Minimize Loss Focused) (Crypto)   
def defensive_ml_trading_strategy_crypto(symbol: str, backtesting_days: int = 365*2) -> dict:
    """
    Analyzes a stock using an aggressive ML trading strategy and returns performance metrics.
    
    Args:
        symbol (str): The stock ticker symbol (e.g., "AAPL")
        backtesting_days (int): Number of days to use for backtesting (default: 2 years)
        alpaca_key (str): Alpaca API key
        alpaca_secret (str): Alpaca API secret
    
    Returns:
        dict: Dictionary containing analysis results and backtest data
        
    Example:
        results = analyze_ml_trading_strategy("AAPL")
        print(f"Total Return: {results['analysis']['total_return']*100:.2f}%")
    """
    try:
        # Initialize the DataFetcher
        analyzer = CryptoDataFetcher(alpaca_key, alpaca_secret)
        
        # Create and run the ML trading strategy
        strategy = DefensiveTradingStrategy(analyzer, symbol, days=backtesting_days)
        
        # Run the complete strategy
        combined_results, backtest_data = strategy.run_strategy()
        
        # Plot results if backtest data is available
        if backtest_data is not None:
            strategy.plot_results(backtest_data)
        
        # Format performance summary
        performance_summary = {
            "final_portfolio_value": f"${combined_results['final_portfolio_value']:.2f}",
            "total_return": f"{combined_results['total_return']*100:.2f}%",
            "buy_hold_return": f"{combined_results['buy_hold_return']*100:.2f}%",
            "annual_return": f"{combined_results['annual_return']*100:.2f}%",
            "sharpe_ratio": f"{combined_results['sharpe_ratio']:.4f}",
            "num_trades": combined_results['num_trades']
        }
        
        # Format top 5 features
        sorted_features = sorted(
            combined_results['feature_importance'].items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        top_features = {}
        for feature, importance in sorted_features:
            top_features[feature] = f"{importance:.4f}"
        
        # Return comprehensive results
        return {
            "analysis": combined_results,
            "backtest_data": backtest_data,
            "performance_summary": performance_summary,
            "top_features": top_features,
            "status": "success"
        }
    
    except Exception as e:
        return {
            "status": "error",
            "message": f"Analysis failed: {str(e)}",
            "analysis": None,
            "backtest_data": None
        }


# AGENT TO COMPARE THREE ANALYSES:
def embeddings_tool(directory_path: str, query: str) -> str:
    """
    Uses a LangChain-based Embeddings model to store and retrieve documents for further analysis.
    """
    try:
        # Initialize the TextFileProcessor with the nomic-embed-text model
        persist_directory = "chroma_db"
        processor = TextFileProcessor(
            chunk_size=1000,
            chunk_overlap=200,
            persist_directory=persist_directory
        )
        
        # Override the default embedding model
        processor.embedding_model = OllamaEmbeddings(model="nomic-embed-text:latest")
        
        # Load documents from the specified directory
        documents = processor.load_files_from_directory(directory_path=directory_path)
        
        # Process the documents
        processor.process_documents(documents)
        
        # Perform similarity search
        results = processor.similarity_search(query=query)
        
        # Format the results as a string
        formatted_results = ""
        for i, doc in enumerate(results):
            formatted_results += f"\n\nDocument {i+1}:\n"
            formatted_results += f"Source: {doc.metadata.get('source', 'Unknown')}\n"
            formatted_results += f"Content: {doc.page_content}\n"
            
        return formatted_results
    except Exception as e:
        return f"Data error: {str(e)}"


print("Tools Initialized")


# Testing tools, in the tools file: 
if __name__ == "__main__":
    # Initialize the Analyzer
    import pprint
   
    symbol = "TSLA"
    tool = embeddings_tool(directory_path="analysis_results", query="I am indecisive and want to choose the best stock to invest in. Pick one of these stocks?")
    pprint.pprint(tool)
