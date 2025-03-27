# This Langchain class analyzes "legal data" with an llm 
# This "legal data" is just some articles I have collected from AI Watch Global regulatory tracker

# Libraries
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
import pprint 

# Initialize the qwen model

qwen = ChatOllama(
    model="qwen2.5:1.5b",
)

class LegalAnalyzer:
    def __init__(self, llm):
        """Initialize the legal analyzer by loading documents, indexing them,
        and setting up the retrieval chain with a custom prompt."""

        # Initialize the ChatOllama LLM.
        self.llm = llm

        # Initialize the embedding model.
        self.embedding_model = OllamaEmbeddings(model="nomic-embed-text:latest")

        # List of PDF file paths to load.
        self.pdf_directories = [
            "/Users/marcusjohansson/Desktop/financial_research_agent/AI_law/AI Watch_ Global regulatory tracker _EU_Council_ White & Case LLP.pdf",
            "/Users/marcusjohansson/Desktop/financial_research_agent/AI_law/AI Watch_ Global regulatory tracker _G7_White & Case LLP.pdf",
            "/Users/marcusjohansson/Desktop/financial_research_agent/AI_law/AI Watch_ Global regulatory tracker - China White & Case LLP.pdf",
            "/Users/marcusjohansson/Desktop/financial_research_agent/AI_law/AI Watch_ Global regulatory tracker - Eu White & Case LLP.pdf",
            "/Users/marcusjohansson/Desktop/financial_research_agent/AI_law/AI Watch_ Global regulatory tracker - OECD _ White & Case LLP.pdf",
            "/Users/marcusjohansson/Desktop/financial_research_agent/AI_law/AI Watch_ Global regulatory tracker - United Kingdom _ White & Case LLP.pdf",
            "/Users/marcusjohansson/Desktop/financial_research_agent/AI_law/AI Watch_ Global regulatory tracker - United Nations _ White & Case LLP.pdf",
            "/Users/marcusjohansson/Desktop/financial_research_agent/AI_law/AI Watch_ Global regulatory tracker - United States _ White & Case LLP.pdf"
        ]

        # Load PDF documents.
        self.documents = []
        for directory in self.pdf_directories:
            loader = PyPDFLoader(directory)
            loaded_docs = loader.load()
            self.documents.extend(loaded_docs)
            print(f"Documents loaded successfully from: {directory}")

        # Index documents with Chroma vector store.
        self.vector_store = Chroma.from_documents(self.documents, self.embedding_model)

        # Convert the vector store to a retriever.
        self.retriever = self.vector_store.as_retriever()

        # Define the system prompt with placeholders.
        system_message_content = """
You are an expert legal analyst and you have worked for many years as a lawyer,
specializing in analyzing and evaluating legal frameworks for investors who ask you questions about the legal environment.
Your task is to interpret and analyze the following legal articles and explain whether this environment
is positive or negative for investors in technology companies like {symbol}.

In your analysis, please address the following:
1. **Overall legal Assessment:** Provide a summary of the current legal environment.
2. **Positive Indicators:** Identify any legal factors (e.g., favorable legal policy) that suggest a beneficial climate for technology investments.
3. **Negative Indicators:** Highlight any legal risks (e.g., strict legal policy) that could adversely affect technology companies.
4. **Key Drivers:** Explain the major factors driving your assessment, such as favorable or unfavorable legal policies.
5. **Actionable Insights:** Offer clear, actionable recommendations for investors considering technology stocks based on your analysis.

Use the following retrieved data to support your evaluation:

{context}

The legal risks should be sorted by country/region from this data:
1. AI Watch_ Global regulatory tracker - China,
2. AI Watch_ Global regulatory tracker - EU,
3. AI Watch_ Global regulatory tracker - United Kingdom,
4. AI Watch_ Global regulatory tracker - United States,
5. AI Watch_ Global regulatory tracker - EU_Council,
6. AI Watch_ Global regulatory tracker - G7,
7. AI Watch_ Global regulatory tracker - OECD,
8. AI Watch_ Global regulatory tracker - United Nations,
and then concluded from a holistic legal environment perspective.

Assume your analysis is presented to the investor at the beginning of 2025. Based solely on this data,
would you continue your investment operations in this legal environment for {symbol}?

Ensure your analysis is concise, objective, and grounded solely in the data provided.
Your insights should help investors understand the potential impact of these regulatory regimes on technology investments.
"""
        # Define the human prompt.
        human_message_content = "Analyze the legal risks of investing in {symbol} based on the retrieved data:\n\n{context}\n\n"

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

    def analyze(self, symbol: str) -> str:
        """
        Run a legal analysis query for a given company ticker symbol.

        Parameters:
            symbol (str): The stock symbol or company identifier (e.g., "UBER", "META").

        Returns:
            str: The legal analysis response generated by the retrieval chain.
        """
        # Build the query dictionary with just the company symbol.
        query = {
            "input": f"""Your goal with your analysis is to explain the legal risks of investing in global technology companies like {symbol}. 
            Your answer should be based on the data provided and be concise, objective, and grounded solely in the data provided.

            Provide a summary of the current legal environment, positive indicators, negative indicators, key drivers, and actionable insights
            Assume your analysis is presented to the investor at the beginning of 2025 for {symbol}, 
            also please keep your analysis to 10 short bullet points.""",

            "symbol": symbol
        }
        # Invoke the retrieval chain and return the response.
        response = self.rag_chain.invoke(query)
        return response


# Example usage
if __name__ == "__main__":
    # Initialize the LegalAnalyzer
    llm = qwen
    analyzer = LegalAnalyzer(llm)
    
    # Analyze a company by its ticker symbol
    symbol = "META"
    result = analyzer.analyze(symbol=symbol)
    
    # Print the analysis result
    print(f"Legal Analysis Result for {symbol}:")
    pprint.pprint(result['answer'])