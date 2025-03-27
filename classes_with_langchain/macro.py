# This Langchain class that collects macroeconomic data from DBnomics and then feeds it to an LLM to analyze
#  This has two use cases where it both can analyze the macro sentiment of a company or generally 


# libraries 
import pandas as pd
from dbnomics import fetch_series
from typing import List, Optional, Dict, Tuple
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain_ollama import ChatOllama

qwen = ChatOllama(
    model="qwen2.5:1.5b",
)

llm = qwen

class MacroAnalyzer:
    def __init__(self, llm: ChatOllama):
        """
        Initialize the analyzer with a local LLM instance.

        Args:
            llm: A local language model instance (compatible with LangChain).
        """
        self.llm = llm
        self.dataframes: List[pd.DataFrame] = []
        self.series_descriptions = {
            "BEA/NIPA-T10701/A191RL-Q": "GDP Quarterly",
            "BLS/cu/CUSR0000SA0": "Monthly US CPI",
            "FED/H15/RIFLGFCY10_N.B": "US 10-year Treasury",
            "FED/H15/RIFLGFCY02_N.B": "US 2-year Treasury",
            "INSEE/IPI-2021/A.BDM.IPI_MOYENNE_ANNUELLE.MIG_CAG.SO.MOYENNE_ANNUELLE.FM.SO.BRUT.2021": "Capital Industrial Index",
            "INSEE/IPI-2021/A.BDM.IPI_MOYENNE_ANNUELLE.SO.26-1.MOYENNE_ANNUELLE.FM.SO.BRUT.2021": "Semiconductor Index",
            "INSEE/IPI-2021/A.BDM.IPI_MOYENNE_ANNUELLE.SO.26-2.MOYENNE_ANNUELLE.FM.SO.BRUT.2021": "Compute Index",
            "INSEE/IPI-2021/A.BDM.IPI_MOYENNE_ANNUELLE.MIG_NRG.SO.MOYENNE_ANNUELLE.FM.SO.BRUT.2021": "Energy Index"
        }
        # Common company profiles for sector-specific analysis
        self.company_profiles = {
            "AAPL": "Technology hardware manufacturer focusing on consumer electronics, software, and services",
            "MSFT": "Software and cloud computing company with diverse enterprise and consumer offerings",
            "GOOGL": "Technology company specializing in internet services, digital advertising, cloud computing, and AI",
            "META": "Social media and metaverse technology company with focus on digital advertising",
            "AMZN": "E-commerce, cloud computing, digital streaming, and AI company with diverse revenue streams",
            "TSLA": "Electric vehicle and clean energy company with significant manufacturing and AI components",
            "NVDA": "Semiconductor company specializing in graphics processing units and AI acceleration hardware",
            "AMD": "Semiconductor company producing processors for computing and graphics applications",
            "INTC": "Semiconductor manufacturer specializing in CPUs, data center, and other computing technologies",
            "IBM": "Technology and consulting company with focus on cloud computing, AI, and enterprise services"
        }

    def fetch_single_series(self, series_id: str) -> Optional[pd.DataFrame]:
        """
        Fetch a single series from DBnomics with error handling.

        Args:
            series_id: The DBnomics series identifier.

        Returns:
            DataFrame containing the series data or None if fetch fails.
        """
        try:
            df = fetch_series(series_id)
            df["series_id"] = series_id
            df["series_name"] = self.series_descriptions.get(series_id, "Unknown Series")
            return df[["original_period", "original_value", "series_name"]].tail()
        except Exception as e:
            print(f"Error fetching series {series_id}: {str(e)}")
            return None

    def collect_data(self) -> List[pd.DataFrame]:
        """
        Collect economic data series from various sources using dbnomics.
        Each series is trimmed to its tail (last few observations) and stored.

        Returns:
            List of pandas DataFrames with the collected data.
        """
        self.dataframes = []

        for series_id in self.series_descriptions.keys():
            df = self.fetch_single_series(series_id)
            if df is not None:
                self.dataframes.append(df)
                print(f"Successfully collected {self.series_descriptions[series_id]}")
                #print(df)
                print("\n")

        if not self.dataframes:
            print("Warning: No data was successfully collected!")
        else:
            print(f"Successfully collected {len(self.dataframes)} out of {len(self.series_descriptions)} series")

        return self.dataframes

    def _format_data(self) -> str:
        """
        Format collected data with series names for better context.

        Returns:
            Formatted string of collected data.
        """
        if not self.dataframes:
            return "No data available"
            
        return "\n\n".join(
            f"{df['series_name'].iloc[0]}:\n{df.drop('series_name', axis=1).to_string(index=False)}"
            for df in self.dataframes
        )

    def analyze_macroeconomics(self) -> Optional[str]:
        """
        Analyze the collected data using a LangChain chat prompt template.

        Returns:
            The textual analysis generated by the LLM or None if no data available.
        """
        if not self.dataframes:
            print("No data collected. Please run collect_data() first.")
            return None

        combined_data = self._format_data()

        prompt_template = ChatPromptTemplate.from_template("""
       
        # ROLE:
        You are an expert macroeconomics investor and analyst, specializing in evaluating macroeconomic factors for investment purposes. Your task is to interpret and analyze the following macroeconomic data and explain whether this environment is positive or negative for investors in technology companies.

        # TASK:
        In your analysis, please address the following:
        1. **Overall Economic Assessment:** Provide a summary of the current macroeconomic environment.
        2. **Positive Indicators:** Identify any economic factors (e.g., GDP growth, low inflation, favorable monetary policy) that suggest a beneficial climate for technology investments.
        3. **Negative Indicators:** Highlight any economic risks (e.g., high inflation, rising interest rates, geopolitical instability) that could adversely affect technology companies.
        4. **Key Drivers:** Explain the major factors driving your assessment, such as economic growth trends, consumer sentiment, fiscal policies, and global market conditions.
        5. **Actionable Insights:** Offer clear, actionable recommendations for investors considering technology stocks based on your analysis.
        6. **Timeframe:** Assume that you invest in the beginning of 2025, based on this data and your analysis would you continue your investment operations in this macroeconomic environment?

        Use the data provided below to support your evaluation:

        {data}

        # OUTPUT FORMAT:
        Provide a summary of the current macroeconimcal environment, positive indicators, negative indicators, key drivers, and actionable insights
        Assume your analysis is presented to the investor at the beginning of 2025, also please keep your analysis to 10 short bullet points. 
        Ensure your analysis is concise, objective, and grounded solely in the data provided. Your insights should help investors understand the potential impact of these macroeconomic factors on technology investments.
        """)

        try:
            chain = prompt_template | self.llm | StrOutputParser()
            return chain.invoke({"data": combined_data})
        except Exception as e:
            print(f"Error during analysis: {str(e)}")
            return None

    def get_company_profile(self, symbol: str) -> str:
        """
        Get company profile for a given ticker symbol, with fallback for unknown symbols.
        
        Args:
            symbol: Stock ticker symbol.
            
        Returns:
            Company profile description.
        """
        return self.company_profiles.get(
            symbol.upper(), 
            f"Technology company with ticker {symbol}"
        )

    def analyze_ticker_macro_impact(self, symbol: str) -> Optional[str]:
        """
        Analyze macroeconomic impact specifically for the given ticker symbol.
        
        Args:
            symbol: Stock ticker symbol to analyze.
            
        Returns:
            Ticker-specific macroeconomic analysis or None if no data available.
        """
        if not self.dataframes:
            print("No data collected. Please run collect_data() first.")
            return None
            
        combined_data = self._format_data()
        company_profile = self.get_company_profile(symbol)
        
        prompt_template = ChatPromptTemplate.from_template("""
        
        # ROLE:                                                  
        You are an expert macroeconomics investor and analyst, specializing in evaluating macroeconomic factors for investment purposes. Your task is to interpret and analyze the following macroeconomic data and explain how it specifically impacts {symbol}, a {company_profile}.

        #TASK:
        In your analysis, please address the following:
        1. **Company-Specific Impact:** How do the current macroeconomic conditions specifically affect {symbol}'s business model, revenue streams, and growth prospects?
        2. **Sector Correlation:** How strongly does {symbol} correlate with broader macroeconomic trends compared to its sector peers?
        3. **Sensitivity Analysis:** Which macroeconomic factors (e.g., interest rates, inflation, GDP growth) is {symbol} most sensitive to, and why?
        4. **Competitive Position:** How might the current macroeconomic environment affect {symbol}'s competitive position within its industry?
        5. **Risk Assessment:** Identify specific macroeconomic risks that could disproportionately impact {symbol}.
        6. **Investment Recommendation:** Based on the macroeconomic data, provide a specific investment recommendation for {symbol} (buy, hold, sell) with supporting rationale.

        Use the data provided below to support your evaluation:

        {data}

        # OUTPUT FORMAT:
        Provide a summary of the current macroeconimcal environment, positive indicators, negative indicators, key drivers, and actionable insights
        Assume your analysis is presented to the investor at the beginning of 2025 for {symbol}, 
        also please keep your analysis to 10 short bullet points.                                                 
        Ensure your analysis is concise, objective, and focuses specifically on how macroeconomic factors uniquely impact {symbol}, rather than just general technology sector trends.
        """)

        try:
            chain = prompt_template | self.llm | StrOutputParser()
            return chain.invoke({
                "data": combined_data,
                "symbol": symbol.upper(),
                "company_profile": company_profile
            })
        except Exception as e:
            print(f"Error during ticker analysis: {str(e)}")
            return None
            
    def full_analysis(self, symbol: str) -> Dict[str, str]:
        """
        Perform both general macroeconomic analysis and ticker-specific analysis.
        
        Args:
            symbol: Stock ticker symbol to analyze.
            
        Returns:
            Dictionary containing both general and ticker-specific analyses.
        """
        # Ensure data is collected
        if not self.dataframes:
            self.collect_data()
            
        # Run general analysis
        general_analysis = self.analyze_macroeconomics()
        
        # Run ticker-specific analysis
        ticker_analysis = self.analyze_ticker_macro_impact(symbol)
        
        return {
            "general_macro_analysis": general_analysis or "Analysis failed",
            "ticker_specific_analysis": ticker_analysis or f"Analysis for {symbol} failed"
        }


# Example usage
if __name__ == "__main__":
    # Initialize the MacroAnalyzer
    analyzer = MacroAnalyzer(llm=qwen)
    
    # Collect data once
    analyzer.collect_data()
    
    # Option 1: Get just the general analysis
    #general_analysis = analyzer.analyze_macroeconomics()
    #print("General Macroeconomic Analysis:")
    #print(general_analysis)
    #print("\n" + "="*80 + "\n")
    
    # Option 2: Get ticker-specific analysis
    symbol = "GOOG"
    ticker_analysis = analyzer.analyze_ticker_macro_impact(symbol)
    print(f"Ticker-Specific Analysis for {symbol}:")
    print(ticker_analysis)
    print("\n" + "="*80 + "\n")
    
    # Option 3: Get both analyses at once
    #symbol = "META"
    #results = analyzer.full_analysis(symbol)
    
    # Print the analysis result
    #print(results)
    #print(f"Full Analysis for {symbol}:")

