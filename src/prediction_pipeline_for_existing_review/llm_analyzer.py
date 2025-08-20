import json
from typing import Dict, List
import logging
from tqdm import tqdm

# LangChain and Pydantic components
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

# Import your custom modules
# Make sure this is the correct, updated Pydantic model (e.g., DashboardAnalysis)
from .structured_output import DashboardAnalysis as ReviewAnalysis 
from .prompt import detailed_prompt_template

# Get a logger instance for this module
logger = logging.getLogger(__name__)

class LLMAnalysis:
    """
    Encapsulates the LLM analysis logic.
    """
    def __init__(self, model_name: str = "llama3"):
        """
        Initializes the LangChain chain once to be reused for all analyses.
        """
        logger.info(f"Initializing LLM chain with model: {model_name}")
        try:
            parser = PydanticOutputParser(pydantic_object=ReviewAnalysis)
            # Build a chat prompt and partially fill format instructions.
            chat_prompt = ChatPromptTemplate.from_template(detailed_prompt_template).partial(
                format_instructions=parser.get_format_instructions()
            )
            # Hint the model to produce JSON-only outputs
            model = ChatOllama(model=model_name, format="json")
            self.chain = chat_prompt | model | parser
            logger.info("LLM chain initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize LLM chain: {e}")
            raise

    def analyze_review(self, review_text: str) -> Dict:
        """Invokes the LLM chain for a single review text."""
        try:
            result = self.chain.invoke({"review_text": review_text})
            return result.dict()
        except Exception as e:
            logger.error(f"Error analyzing review text: '{review_text[:50]}...'. Error: {e}")
            return None

    def run_analysis_on_list(self, data_list: List[Dict]) -> List[Dict]:
        """
        Analyzes each review in a list and merges the original data with the
        LLM analysis result.
        """
        analysis_results = []
        logger.info(f"Starting analysis on a list of {len(data_list)} reviews...")

        for review_data in tqdm(data_list, desc="Analyzing Reviews"):
            # 'review_data' is a dictionary like {'id': 123, 'asin': '...', 'title': '...', 'region': '...', 'text': '...'}
            
            # 1. Analyze the 'text' part of the dictionary
            analysis_dict = self.analyze_review(review_data['text'])
            
            # 2. If the analysis is successful, merge it with the original data
            if analysis_dict:
                # --- THIS IS THE UPDATED PART ---
                # Attach all pass-through data from the input to the output
                analysis_dict['original_id'] = review_data.get('id')
                analysis_dict['asin'] = review_data.get('asin')
                analysis_dict['title'] = review_data.get('title')
                analysis_dict['region'] = review_data.get('region')
                analysis_dict['review_date'] = review_data.get('review_date')
                analysis_results.append(analysis_dict)

        logger.info(f"Analysis complete. Generated {len(analysis_results)} results.")
        return analysis_results