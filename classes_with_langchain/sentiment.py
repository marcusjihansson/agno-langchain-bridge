# This Langchain class analyzes the senitment for a ticker between 2 dates and then feeds it to an LLM 
# It collects data and after doing some data modeling can the user ask questions from the articles collected 
# The api used here is Finlight

# Libraries 
from typing import Optional
from finlight_client import FinlightApi
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
import os 
import dotenv
import pprint

news_key = os.getenv("FINLIGHT_API_KEY")

# Initialize the qwen model

qwen = ChatOllama(
    model="qwen2.5:1.5b",
)

class ContextDataTool:
    def __init__(self, news_key: Optional[str] = None):
        if not news_key:
            raise ValueError("Finlight API key is required.")
        self.news_key = news_key  # Store API key in an instance variable

    def get_news(self, symbol: str, start_date: str, end_date: str, limit: int = 20) -> Optional[str]:
        """
        Fetch financial news for a given stock symbol from Finlight API.

        :param symbol: Stock symbol of the company
        :param start_date: Start date of the news
        :param end_date: End date of the news
        :param limit: Number of articles to fetch
        :return: The news content as a string
        """
        config = {"api_key": self.news_key}
        client = FinlightApi(config)

        params = {
            "query": symbol,
            "language": "en",
            "from": start_date,
            "to": end_date,
            "limit": limit
        }

        # Fetch articles
        articles = client.articles.get_extended_articles(params)

        if not articles or "articles" not in articles:
            return None

        content_str = " ".join(article["content"] for article in articles["articles"] if "content" in article)

        print("Number of articles:", len(articles["articles"]))

        return content_str

    def get_ticker_context(self, symbol: str, start_date: str, end_date: str, llm) -> dict:
        """Full analysis pipeline with summarization"""
        # Fetch news content
        content_str = self.get_news(symbol, start_date, end_date)

        # Check if content is available
        if not content_str:
            return {"error": "No news data available"}

        prompt = ChatPromptTemplate.from_messages([
            ("system", f"""You are a seasoned financial trader with expertise in analyzing news and market data to inform trading decisions.
            Your task is to evaluate the provided data and determine whether it could have an **indirect impact** on the price of the asset identified by the ticker "{symbol}".
            While the data may not be directly related to "{symbol}", it could still influence the asset's price through broader market trends, sector dynamics, or macroeconomic factors.
            Keep in mind that the analysis should strictly consider data between {start_date} and {end_date}, as no future data is available.
            Ensure your recommendations are concise, well-reasoned, and aligned with the timeframe and context provided."""),

            ("user", f"""Analyze the following data:\n\n{content_str}\n\n

            Answer the following questions:
            - Key insights from the data and how they relate to "{symbol}".
            - Whether the data has a **direct** or **indirect** impact on "{symbol}".
            - How these insights could influence the price or market sentiment of "{symbol}".
            - What are the sector trends that could have an impact on "{symbol}". If the data has no direct or indirect impact on "{symbol}", then I will not make a recommendation.
            - What are the market trends and macroeconomic factors that could have an impact on "{symbol}"
            and which of those factors could have a **direct** or **indirect** impact on "{symbol}". If the data has no direct or indirect impact on "{symbol}", then I will not make a recommendation.
            - What are industry trends that could have an impact on "{symbol}". If the data has no direct or indirect impact on "{symbol}", then I will not make a recommendation.
            - Any risks or uncertainties associated with your recommendation.

            Remember, this analysis is based on data between {start_date} and {end_date}, and the recommendation assumes you would act at the {end_date} of this analysis.
            Make sure all your analysis considers both **direct** and **indirect** effects on "{symbol}." """)
        ])

        chain = prompt | llm | StrOutputParser()

        input_variables = {
            "symbol": symbol,
            "start_date": start_date,
            "end_date": end_date,
            "input": content_str
        }

        ticker_context = chain.invoke(input_variables)

        return ticker_context

# Example usage
if __name__ == "__main__":
    # Initialize the LegalAnalyzer
    analyzer = ContextDataTool(news_key)
    
    # Analyze a company by its ticker symbol
    symbol = "META"
    start_date = "2025-03-01"
    end_date = "2025-03-11"
    llm = qwen
    result = analyzer.get_ticker_context(symbol, start_date, end_date, llm)
    
    # Print the analysis result
    print(f"Sentiment Analysis Result for {symbol}:")
    pprint.pprint(result)


