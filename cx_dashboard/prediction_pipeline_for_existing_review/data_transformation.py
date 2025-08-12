import pandas as pd
from typing import List, Dict
import logging
from datetime import datetime

# Get a logger instance for this module
logger = logging.getLogger(__name__)

class DataTransformation:
    """
    This class handles the transformation of raw data into a format
    ready for the LLM.
    """
    def __init__(self):
        # The initializer can be used for configuration in the future
        pass

    def prepare_data_for_llm(self, df: pd.DataFrame) -> List[Dict]:
        """
        Takes a raw DataFrame, selects and processes the required columns,
        and returns a list of dictionaries.

        Args:
            df (pd.DataFrame): The raw DataFrame from the database.

        Returns:
            List[Dict]: A list of clean dictionaries ready for the LLM.
        """
        try:
            logger.info("Starting data transformation process...")
            
            # --- 1. Select only the necessary columns ---
            # 'id' is crucial for updating the status after analysis.
            # 'ASIN' (product ID) provides useful context for the LLM.
            required_columns = ['id', 'ASIN', 'Review']
            if not all(col in df.columns for col in required_columns):
                logger.error(f"Input DataFrame is missing required columns. Expected: {required_columns}")
                return []
            
            df_selected = df[required_columns].copy()
            
            # --- 2. Rename columns for the LLM prompt ---
            # Using lowercase/snake_case is a good practice for keys.
            df_renamed = df_selected.rename(columns={
                'Review': 'text',
                'ASIN': 'asin'
            })
            
            # --- 3. Convert to list of dictionaries ---
            records = df_renamed.to_dict(orient='records')
            logger.info(f"Successfully transformed {len(records)} records.")
            
            return records

        except Exception as e:
            logger.error(f"An error occurred during data transformation: {e}", exc_info=True)
            raise

