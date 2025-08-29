import pandas as pd
from typing import List, Dict
import logging

# Use your custom logger
from src.logger import logger

class DataTransformation:
    """
    Transforms a DataFrame of reviews into a list of dictionaries
    formatted for the LLM analysis stage.
    """
    def prepare_data_for_llm(self, df: pd.DataFrame) -> List[Dict]:
        """
        Takes a DataFrame (from train_set.csv or test_set.csv), selects the
        necessary columns, renames them, and returns a list of dictionaries.

        Args:
            df (pd.DataFrame): The DataFrame containing review data with 'id' and 'Review' columns.

        Returns:
            List[Dict]: A list of dictionaries, where each item is {'id': ..., 'text': ...}.
        """
        try:
            logger.info(f"Starting data transformation for {len(df)} rows...")
            
            # 1. Select the 'id' and 'Review' columns needed for the pipeline
            required_columns = ['id', 'Review']
            if not all(col in df.columns for col in required_columns):
                missing = [col for col in required_columns if col not in df.columns]
                logger.error(f"Input DataFrame is missing required columns: {missing}")
                return []
            
            df_selected = df[required_columns].copy()

            # 2. Rename 'Review' to 'text' for the LLM prompt
            df_renamed = df_selected.rename(columns={'Review': 'text'})
            
            # 3. Convert to a list of dictionaries
            records = df_renamed.to_dict(orient='records')
            logger.info(f"Successfully transformed {len(records)} records.")
            
            return records

        except Exception as e:
            logger.error(f"An error occurred during data transformation: {e}")
            return []