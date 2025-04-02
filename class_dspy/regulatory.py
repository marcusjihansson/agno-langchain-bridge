import dspy
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

class RegulatoryAnalyzer:
    def __init__(self):
        """
        Initializes the LegalAnalyzer with embedded PDF paths and sets up the DSPy pipeline.
        Uses a two-stage approach: first understanding regulations, then applying to specific companies.
        """
        self.pdf_paths = [
            "AI_law/AI Watch_ Global regulatory tracker _EU_Council_ White & Case LLP.pdf",
            "AI_law/AI Watch_ Global regulatory tracker _G7_White & Case LLP.pdf",
            "AI_law/AI Watch_ Global regulatory tracker - China White & Case LLP.pdf",
            "AI_law/AI Watch_ Global regulatory tracker - Eu White & Case LLP.pdf",
            "AI_law/AI Watch_ Global regulatory tracker - OECD _ White & Case LLP.pdf",
            "AI_law/AI Watch_ Global regulatory tracker - United Kingdom _ White & Case LLP.pdf",
            "AI_law/AI Watch_ Global regulatory tracker - United Nations _ White & Case LLP.pdf",
            "AI_law/AI Watch_ Global regulatory tracker - United States _ White & Case LLP.pdf",
            "AI_law/AI Regulations around the World-Mind_Foundry.pdf",
            "AI_law/Global AI Regulations Tracker_ Europe, Americas & Asia-Pacific Overview-Legal_Nodes.pdf",
            "AI_law/global_ai_law_policy_tracker_iapp.pdf"
        ]
        # First stage questions about regulations generally
        self.regulation_questions = [
            "What are the key requirements of recent AI regulations?",
            "What industries or business activities face the most regulatory scrutiny?",
            "What compliance obligations do AI regulations impose on companies?",
            """Which countries (China, United Kingdom, United States), regions (EU Council, EU) and or groups (G7, OECD) 
                have the most strict AI regulations that could be imposed on companies?""",
            """Which countries (for example; China, United Kingdom, United States), regions (for example; EU Council, EU) and or groups (for example; G7, OECD) 
                have the most leanient AI regulations that could be imposed on companies?"""
        ]
        self._setup_pipeline()
        
    def _setup_pipeline(self):
        """Sets up a two-stage DSPy pipeline."""
        lm = dspy.LM('ollama_chat/qwen2.5:1.5b', api_base='http://localhost:11434', api_key='')
        dspy.configure(lm=lm)

        # Load and process documents
        loaders = [PyPDFLoader(path) for path in self.pdf_paths]
        documents = []
        for loader in loaders:
            documents.extend(loader.load())

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(documents)

        # Create embeddings and vectorstore
        embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")
        vectorstore = Chroma.from_documents(texts, embeddings)

        class LangchainRetriever(dspy.Module):
            def __init__(self, vectorstore, k=3):
                super().__init__()
                self.vectorstore = vectorstore
                self.k = k

            def forward(self, query):
                docs = self.vectorstore.similarity_search(query, k=self.k)
                return [doc.page_content for doc in docs]
        
        # Company Context Generator
        class CompanyContextGenerator(dspy.Signature):
            """Generate context about a company based on its ticker symbol."""
            ticker_symbol = dspy.InputField()
            company_context = dspy.OutputField(desc="Detailed context about the company, including sector, business models, AI usage, technology stack, and business practices")
        
        # STAGE 1: Regulations Understanding
        class RegulationAnalysis(dspy.Signature):
            """Extracts key regulatory information from documents."""
            question = dspy.InputField(desc="Question about AI regulations generally")
            documents = dspy.InputField(desc="Regulatory document content")
            key_findings = dspy.OutputField(desc="Key regulatory findings from the documents")
            scope_of_regulations = dspy.OutputField(desc="What companies, sectors, or activities these regulations apply to")
            compliance_requirements = dspy.OutputField(desc="What companies must do to comply")
        
        # STAGE 2: Company-Specific Impact Analysis
        class CompanyImpactAnalysis(dspy.Signature):
            """Analyzes how regulatory findings affect a specific company."""
            ticker_symbol = dspy.InputField()
            company_context = dspy.InputField()
            regulatory_findings = dspy.InputField()
            regulatory_scope = dspy.InputField()
            compliance_requirements = dspy.InputField()
            reasoning = dspy.OutputField(desc="Step-by-step analysis connecting regulations to this company")
            company_impact = dspy.OutputField(desc="How these regulations specifically impact this company")
            risk_assessment = dspy.OutputField(desc="Risk assessment with numerical rating (1-10) and justification")
        
        # Two-Stage Analysis Pipeline
        class TwoStageAnalyzer(dspy.Module):
            def __init__(self, retriever):
                super().__init__()
                self.retriever = retriever
                self.company_context_generator = dspy.ChainOfThought(CompanyContextGenerator)
                self.regulation_analyzer = dspy.ChainOfThought(RegulationAnalysis)
                self.company_impact_analyzer = dspy.ChainOfThought(CompanyImpactAnalysis)
                
            def forward(self, ticker_symbol, regulation_question):
                # Stage 1: Retrieve regulatory information and analyze it generally
                documents = self.retriever(regulation_question)
                reg_analysis = self.regulation_analyzer(
                    question=regulation_question,
                    documents=documents
                )
                
                # Generate company context
                context_result = self.company_context_generator(ticker_symbol=ticker_symbol)
                company_context = context_result.company_context
                
                # Stage 2: Apply regulatory findings to the specific company
                impact_analysis = self.company_impact_analyzer(
                    ticker_symbol=ticker_symbol,
                    company_context=company_context,
                    regulatory_findings=reg_analysis.key_findings,
                    regulatory_scope=reg_analysis.scope_of_regulations,
                    compliance_requirements=reg_analysis.compliance_requirements
                )
                
                # Return combined results
                return dspy.Prediction(
                    regulation_question=regulation_question,
                    regulatory_findings=reg_analysis.key_findings,
                    company_context=company_context,
                    reasoning=impact_analysis.reasoning,
                    company_impact=impact_analysis.company_impact,
                    risk_assessment=impact_analysis.risk_assessment
                )

        # Initialize pipeline components
        self.retriever = LangchainRetriever(vectorstore=vectorstore)
        self.analyzer = TwoStageAnalyzer(self.retriever)

    def analyze(self, ticker_symbol, custom_regulation_question=None):
        """
        Performs a two-stage analysis: first understanding regulations, then applying to the company.
        
        Args:
            ticker_symbol (str): The ticker symbol of the company to analyze
            custom_regulation_question (str, optional): A specific regulation question
            
        Returns:
            dict: Results containing analysis for each regulation question and company impact
        """
        results = {}
        
        # If custom question provided, only analyze that
        if custom_regulation_question:
            result = self.analyzer(ticker_symbol=ticker_symbol, regulation_question=custom_regulation_question)
            return result
        
        # Otherwise analyze all standard regulation questions
        for question in self.regulation_questions:
            result = self.analyzer(ticker_symbol=ticker_symbol, regulation_question=question)
            results[question] = {
                "regulatory_findings": result.regulatory_findings,
                "company_context": result.company_context,
                "reasoning": result.reasoning,
                "company_impact": result.company_impact,
                "risk_assessment": result.risk_assessment
            }
            
        return results

    def generate_report(self, ticker_symbol):
        """Generates a comprehensive regulatory impact report for a company."""
        results = self.analyze(ticker_symbol)
        
        report = f"# Regulatory Impact Analysis for {ticker_symbol}\n\n"
        
        # Extract company context from first result
        company_context = list(results.values())[0]["company_context"]
        report += f"## Company Profile\n{company_context}\n\n"
        
        # Add analysis for each regulatory area
        report += "## Regulatory Impact Assessment\n\n"
        for question, result in results.items():
            report += f"### {question}\n\n"
            report += f"**Key Regulatory Findings:**\n{result['regulatory_findings']}\n\n"
            report += f"**Impact on {ticker_symbol}:**\n{result['company_impact']}\n\n"
            report += f"**Risk Assessment:**\n{result['risk_assessment']}\n\n"
            
        # Add overall recommendations section
        overall_rec_prompt = f"Based on all regulatory analysis for {ticker_symbol}, what are the top 3-5 recommendations for compliance?"
        overall_rec_result = self.analyzer(ticker_symbol=ticker_symbol, regulation_question=overall_rec_prompt)
        
        report += "## Overall Recommendations\n\n"
        report += overall_rec_result.company_impact
            
        return report

# Example Usage
if __name__ == "__main__":
    analyzer = RegulatoryAnalyzer()
    
    # Option 1: Analyze all standard regulatory questions
    #results = analyzer.analyze(ticker_symbol)
    
    # Option 2: Generate a comprehensive report
    ticker_symbol = "META"
    report = analyzer.generate_report(ticker_symbol)
    print(report)
    
    # Option 3: Ask a custom regulation question
    # custom_result = analyzer.analyze("GOOG", "What data protection requirements exist in recent AI regulations?")
    # print(custom_result.company_impact)