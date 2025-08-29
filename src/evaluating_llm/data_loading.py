import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple, List
from src.logger import logger 

def load_data_from_csv(filepath: str) -> pd.DataFrame:
    """
    Reads a CSV file into a pandas DataFrame.
    """
    try:
        logger.info(f"Loading data from CSV file: {filepath}...")
        df = pd.read_csv(filepath)
        logger.info(f"Successfully loaded {len(df)} rows.")
        return df
    except FileNotFoundError:
        logger.error(f"File not found at path: {filepath}")
        return pd.DataFrame()

def create_evaluation_sets(df: pd.DataFrame, target_column: str, features_columns: List[str], test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits the DataFrame into training and testing sets. It performs a
    stratified split and handles classes with only one member.
    """
    logger.info("Splitting data into training and testing sets...")
    
    # Pre-split cleaning: Remove classes with only one member
    value_counts = df[target_column].value_counts()
    ratings_to_keep = value_counts[value_counts > 1].index
    
    original_rows = len(df)
    df_filtered = df[df[target_column].isin(ratings_to_keep)]
    
    if len(df) > len(df_filtered):
        logger.warning(f"Removed {original_rows - len(df_filtered)} rows with unique ratings for stratification.")

    if df_filtered.empty:
        logger.error("No data left after filtering for stratification. Cannot split.")
        return pd.DataFrame(), pd.DataFrame()

    # Define features (X) and the target for stratification (y)
    features = df_filtered[features_columns]
    target = df_filtered[target_column]

    # Perform the stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=42,
        stratify=target
    )

    # --- THIS IS THE CORRECTED PART ---
    # Reconstruct the final DataFrames using the index from the split.
    # This is the cleanest way and preserves ALL original columns.
    train_df = df_filtered.loc[X_train.index]
    test_df = df_filtered.loc[X_test.index]
    # --- END OF CORRECTION ---

    logger.info(f"Data split complete. Training set: {len(train_df)} rows. Testing set: {len(test_df)} rows.")
    return train_df, test_df