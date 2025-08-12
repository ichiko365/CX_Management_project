import json
from typing import Dict, List
import logging
from tqdm import tqdm

# LangChain and Pydantic components
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

# Import your custom modules
from .structured_output import ReviewAnalysis
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
        This is much more efficient than rebuilding it for every review.
        """
        logger.info(f"Initializing LLM chain with model: {model_name}")
        try:
            parser = PydanticOutputParser(pydantic_object=ReviewAnalysis)
            prompt = ChatPromptTemplate.from_template(
                template=detailed_prompt_template,
                partial_variables={"format_instructions": parser.get_format_instructions()}
            )
            model = ChatOllama(model=model_name)
            self.chain = prompt | model | parser
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

    # In your LLMAnalysis class inside llm_analyzer.py

    def run_analysis_on_list(self, data_list: List[Dict]) -> List[Dict]:
        analysis_results = []
        logger.info(f"Starting analysis on a list of {len(data_list)} reviews...")

        for review_data in tqdm(data_list, desc="Analyzing Reviews"):
            # 'review_data' is a dictionary like {'id': 123, 'text': '...'}
            
            # 1. Analyze the 'text' part of the dictionary
            analysis_dict = self.analyze_review(review_data['text'])
            
            # --- THIS IS THE CRUCIAL PART ---
            # 2. If the analysis is successful, attach the original 'id'
            if analysis_dict:
                # Get the 'id' from the original input data for this loop iteration
                analysis_dict['original_id'] = review_data.get('id')
                analysis_results.append(analysis_dict)

        logger.info(f"Analysis complete. Generated {len(analysis_results)} results.")
        return analysis_results