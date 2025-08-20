import os
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from typing import Optional
from urllib.parse import quote_plus
from dotenv import load_dotenv
import logging

# 1. Get a logger instance for this specific file.
# The configuration will be inherited from the setup in your main script.
logger = logging.getLogger(__name__)

def create_db_engine() -> Optional[Engine]:
    """
    Reads database credentials from the .env file and creates a SQLAlchemy engine.
    """
    load_dotenv()
    db_user = os.getenv("DB_USER")
    db_password = os.getenv("DB_PASSWORD")
    db_host = os.getenv("DB_HOST")
    db_port = os.getenv("DB_PORT")
    db_name = os.getenv("DB_NAME")

    if not all([db_user, db_password, db_host, db_port, db_name]):
        logger.error("Database configuration is missing in the .env file.")
        return None
    
    try:
        encoded_password = quote_plus(db_password)
        connection_url = f"postgresql://{db_user}:{encoded_password}@{db_host}:{db_port}/{db_name}"
        engine = create_engine(connection_url)
        # Test connection
        engine.connect()
        logger.info("Successfully created database engine.")
        return engine
    except Exception as e:
        logger.error(f"Error creating database engine: {e}")
        return None

def fetch_pending_reviews(engine: Engine, batch_size: int = 100) -> pd.DataFrame:
    """
    Fetches a batch of reviews with 'pending' status from the PostgreSQL database.
    """
    table_name = "raw_reviews"
    sql_query = f"""
        SELECT * FROM {table_name} 
        WHERE analysis_status = 'pending' 
        LIMIT {batch_size};
    """
    
    try:
        logger.info(f"Fetching up to {batch_size} pending reviews from table '{table_name}'...")
        df = pd.read_sql_query(sql_query, engine)
        logger.info(f"Successfully fetched {len(df)} reviews.")
        return df
    except Exception as e:
        logger.error(f"Error fetching data: {e}")
        return pd.DataFrame() # Return an empty DataFrame on error