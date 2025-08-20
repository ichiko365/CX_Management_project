import logging
import pandas as pd
from from_root import from_root
import os
from sklearn.metrics import accuracy_score, classification_report

# 1. Import your custom logger setup.
# from logger import logger 
log = logging.getLogger(__name__)

def evaluate_models():
    """
    Loads the comparison results, aligns data formats, and calculates
    performance metrics for each model against the ground truth.
    """
    log.info("--- Starting Final Model Evaluation ---")
    
    # === STAGE 1: LOAD THE COMPARISON DATA ===
    try:
        comparison_path = os.path.join(from_root(),"cx_dashboard", "data", "training_comparison_results.csv")
        df = pd.read_csv(comparison_path)
        log.info(f"Loaded comparison results with {len(df)} reviews.")
    except FileNotFoundError:
        log.error(f"'{comparison_path}' not found. Please run the multi-model analysis pipeline first.")
        return

    # --- STAGE 2: DATA CLEANING AND PREPARATION ---
    log.info("Cleaning and aligning data formats for evaluation...")
    
    # Define the mapping from string to integer
    sentiment_map = {"Positive": 1, "Neutral": 0, "Negative": -1}

    # Apply the mapping to the prediction columns to convert them to numbers
    df['llama_sentiment_numeric'] = df['llama_sentiment'].map(sentiment_map)
    df['gemini_sentiment_numeric'] = df['gemini_sentiment'].map(sentiment_map)
    
    # Drop rows where any model failed or ground truth is missing
    df.dropna(subset=['llama_sentiment_numeric', 'gemini_sentiment_numeric', 'Sentiment'], inplace=True)
    
    # Ensure all columns are the correct integer type
    df['llama_sentiment_numeric'] = df['llama_sentiment_numeric'].astype(int)
    df['gemini_sentiment_numeric'] = df['gemini_sentiment_numeric'].astype(int)
    df['Sentiment'] = df['Sentiment'].astype(int)
    
    ground_truth = df['Sentiment']

    # === STAGE 3: EVALUATE LLAMA 3 PERFORMANCE ===
    log.info("\n" + "="*50)
    log.info("📊 Llama 3 (Ollama) Performance Report")
    log.info("="*50)
    llama_predictions = df['llama_sentiment_numeric']
    llama_accuracy = accuracy_score(ground_truth, llama_predictions)
    log.info(f"Overall Accuracy: {llama_accuracy * 100:.2f}%")
    
    llama_report = classification_report(ground_truth, llama_predictions, target_names=['Negative (-1)', 'Neutral (0)', 'Positive (1)'])
    print("\nClassification Report:\n", llama_report)

    # === STAGE 4: EVALUATE GEMINI PERFORMANCE ===
    log.info("\n" + "="*50)
    log.info("✨ Gemini Performance Report")
    log.info("="*50)
    gemini_predictions = df['gemini_sentiment_numeric']
    gemini_accuracy = accuracy_score(ground_truth, gemini_predictions)
    log.info(f"Overall Accuracy: {gemini_accuracy * 100:.2f}%")
    
    gemini_report = classification_report(ground_truth, gemini_predictions, target_names=['Negative (-1)', 'Neutral (0)', 'Positive (1)'])
    print("\nClassification Report:\n", gemini_report)
    
    log.info("--- Model Evaluation Finished ---")

if __name__ == '__main__':
    evaluate_models()