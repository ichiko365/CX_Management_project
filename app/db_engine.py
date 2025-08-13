import os
import logging
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from typing import Optional
from urllib.parse import quote_plus
from dotenv import load_dotenv

# Get a logger for this utility module
logger = logging.getLogger(__name__)

def create_db_engine(db_name_env: str = "DB_NAME") -> Optional[Engine]:
    """
    Reads database credentials and a specific database name from the .env file
    and creates a robust SQLAlchemy engine.

    Args:
        db_name_env (str): The name of the environment variable that holds the
                           database name (e.g., "DB_NAME" or "RESULT_DB_NAME").

    Returns:
        Optional[Engine]: A SQLAlchemy engine instance if successful, otherwise None.
    """
    load_dotenv()
    
    # Load all necessary credentials from the .env file
    db_user = os.getenv("DB_USER")
    db_password_raw = os.getenv("DB_PASSWORD")
    db_host = os.getenv("DB_HOST")
    db_port = os.getenv("DB_PORT")
    db_name = os.getenv(db_name_env)

    # Check that all required variables are present
    if not all([db_user, db_password_raw, db_host, db_port, db_name]):
        logger.error(f"Database configuration is missing. Check .env for: DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, and {db_name_env}")
        return None
    
    try:
        # URL-encode the password to handle any special characters
        encoded_password = quote_plus(db_password_raw)
        
        # Construct the final connection URL
        connection_url = f"postgresql://{db_user}:{encoded_password}@{db_host}:{db_port}/{db_name}"
        
        engine = create_engine(connection_url)
        
        # Test the connection to ensure it's valid before returning
        with engine.connect() as connection:
            logger.info(f"Successfully created and tested database engine for: {db_name}")
        
        return engine
        
    except Exception as e:
        logger.error(f"Error creating database engine for {db_name}: {e}")
        return None
