import json
from typing import Dict, List
import logging
from tqdm import tqdm
import os

# LangChain and Pydantic components
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

# Import both model providers
from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI

# Import your custom modules
from .structured_output import DashboardAnalysis
from .prompt import detailed_prompt_template

logger = logging.getLogger(__name__)

class LLMAnalysis:
    """
    Encapsulates the LLM analysis logic, with support for multiple providers.
    """
    def __init__(self, provider: str = "ollama", model_name: str = "llama3"):
        """
        Initializes the LangChain chain, selecting the model based on the provider.

        Args:
            provider (str): The LLM provider to use ('ollama' or 'deepseek').
            model_name (str): The specific model to use (e.g., 'llama3' or 'deepseek-r1:8b').
        """
        logger.info(f"Initializing LLM chain with provider: '{provider}', model: '{model_name}'")
        try:
            parser = PydanticOutputParser(pydantic_object=DashboardAnalysis)
            prompt = ChatPromptTemplate.from_template(
                template=detailed_prompt_template,
                partial_variables={"format_instructions": parser.get_format_instructions()}
            )

            # --- Select the model based on the provider ---
            if provider.lower() == 'ollama':
                model = ChatOllama(model=model_name)
            elif provider.lower() == 'deepseek':
                # This will automatically use the GOOGLE_API_KEY from your .env file
                # if not os.getenv("GOOGLE_API_KEY"):
                    # raise ValueError("GOOGLE_API_KEY not found in .env file.")
                # model = ChatGoogleGenerativeAI(model=model_name, temperature=0) # Just for now, ruf
                model = ChatOllama(model=model_name)
            else:
                raise ValueError(f"Unsupported provider: {provider}. Please choose 'ollama' or 'gemini'.")

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

    def run_analysis_on_list(self, data_list: List[Dict]) -> List[Dict]:
        """
        Analyzes each review in a list and merges the original data with the
        LLM analysis result.
        """
        analysis_results = []
        logger.info(f"Starting analysis on a list of {len(data_list)} reviews...")

        for review_data in tqdm(data_list, desc="Analyzing Reviews"):
            analysis_dict = self.analyze_review(review_data['text'])
            
            if analysis_dict:
                # Attach all pass-through data from the input to the output
                analysis_dict['original_id'] = review_data.get('id')
                analysis_dict['asin'] = review_data.get('asin')
                analysis_dict['title'] = review_data.get('title')
                analysis_dict['region'] = review_data.get('region')
                analysis_results.append(analysis_dict)

        logger.info(f"Analysis complete. Generated {len(analysis_results)} results.")
        return analysis_results