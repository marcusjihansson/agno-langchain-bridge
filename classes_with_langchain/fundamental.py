# This Langchain class collects fundamental stock data from Alphavantage, then uses an llm to analyze the results
# This fundamental stock data is: income statement, balance sheet, cash flow statement and earnings 

# Libraries
from pathlib import Path
from typing import TypedDict, List, Annotated, Dict, Any, Optional, Tuple
import pprint

# LangChain libraries for building the Agent structure
from langchain_core.tools import BaseTool
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.output_parsers.string import StrOutputParser
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_ollama import ChatOllama

import os
import time
import requests
import dotenv

alpha_vantage_key = os.getenv("ALPHA_VANTAGE_API_KEY")

# Initialize the model
qwen = ChatOllama(
    model="qwen2.5:1.5b",
)

# Analyze fundamental data for a compnay based on their financial statements
class AlphaVantageClient:
    BASE_URL = "https://www.alphavantage.co/query"
    RATE_LIMIT_DELAY = 15  # Alpha Vantage's free tier has a rate limit of 5 requests per minute

    def __init__(self, alpha_vantage_key: Optional[str] = None):
        if not alpha_vantage_key:
            raise ValueError("Alpha Vantage API key is required.")
        self.alpha_vantage_key = alpha_vantage_key  # Store API key in an instance variable

    def _fetch_data(self, function: str, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Generic method to fetch financial statement data from Alpha Vantage API.
        """
        params = {
            "function": function,
            "symbol": symbol,
            "apikey": self.alpha_vantage_key
        }
        
        response = requests.get(self.BASE_URL, params=params)
        
        if response.status_code == 200:
            data = response.json()
            if "Error Message" in data:
                print(f"Error: {data['Error Message']}")
                return None
            return data
        else:
            print(f"Error: HTTP status code {response.status_code}")
            return None

    def get_income_statement(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Fetch the last 2 quarterly income statements."""
        time.sleep(self.RATE_LIMIT_DELAY)  # Respect API rate limits
        data = self._fetch_data("INCOME_STATEMENT", symbol)
        if data and "quarterlyReports" in data:
            return {"symbol": symbol, "quarterlyReports": data["quarterlyReports"][:2]}
        return None

    def get_balance_sheet(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Fetch the last 2 quarterly balance sheets."""
        time.sleep(self.RATE_LIMIT_DELAY)
        data = self._fetch_data("BALANCE_SHEET", symbol)
        if data and "quarterlyReports" in data:
            return {"symbol": symbol, "quarterlyReports": data["quarterlyReports"][:2]}
        return None

    def get_cash_flow_statement(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Fetch the last 2 quarterly cash flow statements."""
        time.sleep(self.RATE_LIMIT_DELAY)
        data = self._fetch_data("CASH_FLOW", symbol)
        if data and "quarterlyReports" in data:
            return {"symbol": symbol, "quarterlyReports": data["quarterlyReports"][:2]}
        return None

    def get_earnings(self, symbol: str) -> Optional[Any]:
        """Fetch the last 2 quarterly earnings reports."""
        time.sleep(self.RATE_LIMIT_DELAY)
        data = self._fetch_data("EARNINGS", symbol)
        if data and "quarterlyEarnings" in data:
            return data["quarterlyEarnings"][:2]
        return None

    def run_analysis(self, symbol: str) -> dict:
        """Run analysis by fetching all financial data for a given ticker."""
        print(f"Fetching financial data for {symbol}...")
        data = {
            "income_statement": self.get_income_statement(symbol),
            "balance_sheet": self.get_balance_sheet(symbol),
            "cash_flow": self.get_cash_flow_statement(symbol),
            "earnings": self.get_earnings(symbol)
        }
        return data

    def get_fundamental_analysis(self, symbol: str, llm) -> dict:
        """Full analysis pipeline with summarization, including financial data from run_analysis."""
        financial_data = self.run_analysis(symbol)

        if not financial_data:
            return {"error": "No financial data available"}

        prompt = ChatPromptTemplate.from_messages([
            ("system", f"""You are a seasoned financial trader with expertise in analyzing fundamental company data such as income statements,
            balance sheets, cash flow statements, and earnings data.
            Your task is to evaluate the financial data and determine whether it could have a **direct or indirect impact** on the price of the asset identified by the ticker "{symbol}".
            Consider broader market trends, sector dynamics, and macroeconomic factors when evaluating the financial performance of the asset identified by the ticker "{symbol}".
            Ensure your recommendations are concise, well-reasoned, and aligned with the provided timeframe.
            The data collected is for the last two quarters and your analysis should be focused on how the company 
            is financially positioned over the past two quarters and how this could impact the future price of the asset identified by the ticker "{symbol}". 
            Your analysis should also include any potential risks or uncertainties associated with the company's financial performance."""),
            
            ("user", """Analyze the following financial data:
            {financial_data}

            Answer the following questions:
            - Key insights and how they relate to "{symbol}".
            - Direct or indirect impact on "{symbol}".
            - How these insights could influence the price in the future.
            - Sector, market, and industry trends affecting "{symbol}".
            - Risks or uncertainties associated with your recommendation.

            Make sure all your analysis considers both **direct** and **indirect** effects on "{symbol}".""")
        ])

        # 🔥 Fix: Properly pass financial_data and symbol to the prompt
        input_variables = {
            "symbol": symbol,
            "financial_data": financial_data  # Pass the financial data to the LLM
        }

        chain = prompt | llm | StrOutputParser()
        fundamental_analysis = chain.invoke(input_variables) 
        return fundamental_analysis

# Example usage
if __name__ == "__main__":
    # Analyze a company by its ticker symbol
    symbol = "GOOG"

    # Initialize the ChatOllama LLM
    llm = qwen

    # Initialize the GeopoliticsAnalyzer
    client = AlphaVantageClient(alpha_vantage_key)
    
    # Run the analysis
    data = client.get_fundamental_analysis(symbol, llm)
    
    # Print the analysis result
    print(f"Fundamental Analysis Result for {symbol}:")

    pprint.pprint(data)
