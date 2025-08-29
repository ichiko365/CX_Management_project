import logging
import pandas as pd
from from_root import from_root
import os
from dotenv import load_dotenv # --- NEW ---

# --- 1. Load environment variables from .env file ---
# This makes GOOGLE_API_KEY available to LangChain
load_dotenv(override=True) # --- NEW ---

# 2. Import your custom logger setup.
from src.logger import logger 

# 3. Import the components
from src.evaluating_llm.data_loading import load_data_from_csv
from src.evaluating_llm.data_transformation import DataTransformation
from src.evaluating_llm.llm_analyzer import LLMAnalysis

# 4. Get a logger instance for this specific script
log = logging.getLogger(__name__)

def main():
    """
    Orchestrates the full evaluation pipeline for multiple models on the training set.
    """
    log.info("--- Starting Multi-Model Evaluation Pipeline on Training Set ---")
    
    # === STAGE 1: LOAD & TRANSFORM TRAINING DATA ===
    try:
        train_set_path = os.path.join(from_root(),"src", "data", "train_set.csv")
        train_df = pd.read_csv(train_set_path).head(30)
        log.info(f"Loaded training set with {len(train_df)} reviews.")
    except FileNotFoundError:
        log.error(f"'{train_set_path}' not found. Please run data preparation pipeline first.")
        return

    transformer = DataTransformation()
    llm_ready_training_data = transformer.prepare_data_for_llm(train_df)

    # === STAGE 2: RUN ANALYSIS WITH LLAMA 3 ===
    log.info("--- Analyzing with Llama 3 (Ollama) ---")
    llama_analyzer = LLMAnalysis(provider="ollama", model_name="llama3")
    llama_results = llama_analyzer.run_analysis_on_list(llm_ready_training_data)

    # === STAGE 3: RUN ANALYSIS WITH DEEPSEEK ===
    log.info("--- Analyzing with DeepSeek ---")
    deepseek_analyzer = LLMAnalysis(provider="deepseek", model_name='deepseek-r1:8b')
    deepseek_results = deepseek_analyzer.run_analysis_on_list(llm_ready_training_data)

    # === STAGE 4: MERGE AND SAVE COMPARISON RESULTS ===
    log.info("--- Merging results for comparison ---")
    
    # Convert results to DataFrames, adding prefixes to the columns
    llama_df = pd.DataFrame(llama_results).rename(columns=lambda c: f"llama_{c}")
    deepseek_df = pd.DataFrame(deepseek_results).rename(columns=lambda c: f"deepseek_{c}")

    # Merge with the original training data based on the ID
    comparison_df = pd.merge(train_df, llama_df, left_on='id', right_on='llama_original_id', how='left')
    comparison_df = pd.merge(comparison_df, deepseek_df, left_on='id', right_on='deepseek_original_id', how='left')

    # Save the final comparison file
    output_path = os.path.join(from_root(),"src", "data", "training_comparison_results.csv")
    comparison_df.to_csv(output_path, index=False)
    
    log.info(f"✅ Comparison results for the training set saved to '{output_path}'")
    log.info("--- Multi-Model Evaluation Finished Successfully ---")

if __name__ == '__main__':
    main()