# This Langchain class analyzes geopolitics data, then uses an llm to analyze the results
# The data here is collected by finding data sets online and then doing some data modeling 

# The data sources used are the following; 
# credit_risk_premium: https://pages.stern.nyu.edu/~adamodar/New_Home_Page/datafile/ctryprem.html
# Economic_Policy_Data: https://www.policyuncertainty.com/
# Geopolitical_Risk_Index: https://www.policyuncertainty.com/gpr.html
# wgidataset: https://www.worldbank.org/en/publication/worldwide-governance-indicators
# gwmo_business_env_sp: S&P Global Country Risk Service (WMO) and searched on Google for WMO23PV and click the link and get the excel file
# worldriskindex: https://data.humdata.org/dataset/worldriskindex#:~:text=The%20WorldRiskIndex%20is%20a%20statistical%20model%20that%20provides,events%20and%20the%20negative%20impacts%20of%20climate%20change.



# Libraries
from geopolitical_research.geopolitics import (
    geopolitical_risk_index,
    global_economic_policy_uncertainty_index,
    credit_risk_premium_data,
    control_of_corruption,
    government_effectiveness,
    political_stability,
    rule_of_law,
    regulatory_quality,
    voice_accountability,
    world_risk_trend,
    explanations_world_risk,
    analyst_expectations
)

from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OllamaEmbeddings
from langchain.vectorstores import Chroma
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_ollama import ChatOllama
from langchain.schema import Document
import pprint

# Initialize the qwen model

qwen = ChatOllama(
    model="qwen2.5:1.5b",
)

llm = qwen

class GeopoliticsAnalyzer:
    def __init__(self, llm):
        """Initialize the geopolitics analyzer by loading DataFrames, converting them to documents,
        and setting up the retrieval chain."""
        # Load the DataFrames from the geopolitics module.
        self.dataframes = [
            geopolitical_risk_index,
            global_economic_policy_uncertainty_index,
            credit_risk_premium_data,
            control_of_corruption,
            government_effectiveness,
            political_stability,
            rule_of_law,
            regulatory_quality,
            voice_accountability,
            world_risk_trend,
            explanations_world_risk,
            analyst_expectations
        ]

        # Convert DataFrames into LangChain Documents.
        self.documents = self._dataframes_to_documents(self.dataframes)

        # Initialize the embedding model.
        self.embedding_model = OllamaEmbeddings(model="nomic-embed-text:latest")

        # Index documents with Chroma vector store.
        self.vector_store = Chroma.from_documents(self.documents, self.embedding_model)

        # Initialize the ChatOllama LLM.
        self.llm = llm

        # Convert the vector store to a retriever.
        self.retriever = self.vector_store.as_retriever()

        # Define the system prompt with incorporated question details.
        system_message_content = """
You are an expert geopolitical analyst with extensive experience evaluating global political, economic, and regulatory risks. Your role is to assess the geopolitical environment and its implications for investing in {symbol}, using a wide array of data sources.

You have access to the following data sources:
1. geopolitical_risk_index: A quantitative measure of global geopolitical risks.
2. global_economic_policy_uncertainty_index: An indicator of worldwide economic policy uncertainty.
3. credit_risk_premium_data: Data assessing credit risk premiums in financial markets.
4. control_of_corruption: A metric evaluating the effectiveness of corruption control.
5. government_effectiveness: Measures of government efficiency and policy implementation.
6. political_stability: Indicators of political system stability.
7. rule_of_law: Evaluations of the consistency and fairness of legal frameworks.
8. regulatory_quality: Data on the quality and enforcement of regulations.
9. voice_accountability: Metrics on citizen participation and governmental accountability.
10. world_risk_trend: Trends in global risk levels over time.
11. explanations_world_risk: Qualitative explanations for changes in world risk.
12. analyst_expectations: Summaries of expert forecasts regarding future geopolitical risks.

Your task is to analyze these data sources and provide a comprehensive evaluation of the geopolitical risks associated with investing in {symbol}. You will:

1. Assess the overall global geopolitical environment and identify trends from the data that could influence {symbol}'s operations or stock performance.
2. Identify specific risks such as regulatory challenges, political instability, corruption, and other relevant factors.
3. Deliver clear, actionable insights and recommendations for investors, focusing solely on the geopolitical aspects of investing in {symbol}.
4. Clearly note if any data is missing or inconclusive.

Be sure to address all relevant data sources in your analysis, with particular attention to how each factor might impact {symbol}'s business model, operations, and market performance.

Your response should be in-depth (using as many data sources as possible), but concise, well-structured, data-driven, and actionable.

Use the following retrieved data to support your evaluation:

{context}
"""
        
        # Define the human prompt.
        human_message_content = "Analyze the geopolitical risks for investing in {symbol} based on the retrieved data:\n\n{context}\n\n"

        # Create a chat prompt template using the system and human messages.
        self.chat_prompt_template = ChatPromptTemplate(
            input_variables=["context", "symbol"],
            messages=[
                SystemMessagePromptTemplate.from_template(system_message_content),
                HumanMessagePromptTemplate.from_template(human_message_content)
            ]
        )

        # Build the chain to combine retrieved documents using the "stuff" method.
        combine_docs_chain = create_stuff_documents_chain(self.llm, self.chat_prompt_template)
        self.rag_chain = create_retrieval_chain(self.retriever, combine_docs_chain)

    def _dataframes_to_documents(self, dataframes):
        """
        Convert a list of pandas DataFrames into LangChain Document objects.
        """
        documents = []
        for df in dataframes:
            for _, row in df.iterrows():
                text = "\n".join([f"{col}: {value}" for col, value in row.items()])
                doc = Document(page_content=text, metadata={"source": "dataframe"})
                documents.append(doc)
        return documents

    def analyze(self, symbol: str) -> str:
        """
        Run a geopolitics analysis query for a given company ticker symbol.

        Parameters:
            symbol (str): The stock symbol or company identifier (e.g., "UBER", "META").

        Returns:
            str: The geopolitical analysis response generated by the retrieval chain.
        """
        query = {
            "input": f"What are the geopolitical risks for investing in {symbol}?",
            "symbol": symbol
        }
        response = self.rag_chain.invoke(query)
        return response


# Example usage
if __name__ == "__main__":
    # Initialize the ChatOllama LLM
    llm = qwen
    
    # Initialize the GeopoliticsAnalyzer
    analyzer = GeopoliticsAnalyzer(llm)
    
    # Analyze a company by its ticker symbol
    symbol = "META"
    result = analyzer.analyze(symbol=symbol)
    
    # Print the analysis result
    print(f"Geopolitical Analysis Result for {symbol}:")
    pprint.pprint(result)
