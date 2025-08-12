import os
import logging
import json
from typing import List, Dict
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
from urllib.parse import quote_plus

logger = logging.getLogger(__name__)

class DataSaver:
    def __init__(self):
        """Initializes connections to both source and result databases."""
        load_dotenv()
        self.source_engine = self._create_engine(os.getenv("DB_NAME"))
        self.result_engine = self._create_engine(os.getenv("RESULT_DB_NAME"))
        
        if self.result_engine:
            self._create_results_table_if_not_exists()

    def _create_engine(self, db_name):
        """Helper function to create a SQLAlchemy engine."""
        try:
            db_user = os.getenv("DB_USER")
            # Ensure password is not None before quoting
            db_password_raw = os.getenv("DB_PASSWORD")
            db_password = quote_plus(db_password_raw) if db_password_raw else ''
            
            db_host = os.getenv("DB_HOST")
            db_port = os.getenv("DB_PORT")
            url = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
            engine = create_engine(url)
            engine.connect().close() # Use a short-lived connection to test
            logger.info(f"Successfully connected to database: {db_name}")
            return engine
        except Exception as e:
            logger.error(f"Failed to connect to database {db_name}: {e}")
            return None

    def _create_results_table_if_not_exists(self):
        """Creates the analysis_results table with a consistent schema."""
        # --- UPDATED TABLE SCHEMA: Removed FOREIGN KEY, renamed review_id ---
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS analysis_results (
            id SERIAL PRIMARY KEY,
            original_review_id INTEGER NOT NULL UNIQUE,
            asin VARCHAR(20),
            title TEXT,
            region VARCHAR(10),
            sentiment VARCHAR(20),
            summary TEXT,
            key_drivers JSONB,
            urgency_score INTEGER,
            issue_tags TEXT[],
            primary_category VARCHAR(50),
            analysis_timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        );
        """
        try:
            with self.result_engine.connect() as connection:
                connection.execute(text(create_table_sql))
                connection.commit()
            logger.info("Table 'analysis_results' is ready.")
        except Exception as e:
            logger.error(f"Failed to create 'analysis_results' table: {e}")

    def save_results(self, analysis_results: List[Dict]):
        """
        Saves analysis results idempotently and updates status in the source DB.
        """
        if not self.source_engine or not self.result_engine:
            logger.error("Database connections not available. Cannot save results.")
            return

        SourceSession = sessionmaker(bind=self.source_engine)
        ResultSession = sessionmaker(bind=self.result_engine)
        
        ids_to_update = [res.get('original_id') for res in analysis_results if res.get('original_id')]
        if not ids_to_update:
            logger.warning("No results with original_id found to save.")
            return

        # --- Use context managers for safer, cleaner session handling ---
        with ResultSession() as result_session:
            try:
                for result in analysis_results:
                    if not result.get('original_id'):
                        continue

                    insert_sql = text("""
                        INSERT INTO analysis_results (original_review_id, asin, title, region, sentiment, summary, key_drivers, urgency_score, issue_tags, primary_category)
                        VALUES (:original_id, :asin, :title, :region, :sentiment, :summary, :drivers, :urgency, :tags, :category)
                        ON CONFLICT (original_review_id) DO UPDATE SET
                            asin = EXCLUDED.asin, title = EXCLUDED.title, region = EXCLUDED.region,
                            sentiment = EXCLUDED.sentiment, summary = EXCLUDED.summary, key_drivers = EXCLUDED.key_drivers,
                            urgency_score = EXCLUDED.urgency_score, issue_tags = EXCLUDED.issue_tags,
                            primary_category = EXCLUDED.primary_category, analysis_timestamp = CURRENT_TIMESTAMP;
                    """)
                    
                    params = {
                        "original_id": result.get('original_id'),
                        "asin": result.get('asin'),
                        "title": result.get('title'),
                        "region": result.get('region'),
                        "sentiment": result.get('sentiment'),
                        "summary": result.get('summary'),
                        "drivers": json.dumps(result.get('key_drivers')),
                        "urgency": result.get('urgency_score'),
                        "tags": result.get('issue_tags'),
                        "category": result.get('primary_category')
                    }
                    result_session.execute(insert_sql, params)
                
                result_session.commit()
                logger.info(f"Successfully saved/updated {len(ids_to_update)} analysis results.")

            except Exception as e:
                logger.error(f"Error during result saving: {e}", exc_info=True)
                result_session.rollback()
                return # Stop if saving results fails

        # Update the status in the original database in one batch
        with SourceSession() as source_session:
            try:
                update_sql = text("UPDATE raw_reviews SET analysis_status = 'completed' WHERE id = ANY(:ids);")
                source_session.execute(update_sql, {"ids": ids_to_update})
                source_session.commit()
                logger.info(f"Updated status for {len(ids_to_update)} reviews in source database.")
            except Exception as e:
                logger.error(f"Error updating status in source database: {e}", exc_info=True)
                source_session.rollback()