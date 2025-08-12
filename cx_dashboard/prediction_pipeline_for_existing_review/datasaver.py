import os
import logging
from typing import List, Dict
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
from urllib.parse import quote_plus
import json
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
            db_password = quote_plus(os.getenv("DB_PASSWORD"))
            db_host = os.getenv("DB_HOST")
            db_port = os.getenv("DB_PORT")
            url = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
            engine = create_engine(url)
            engine.connect()
            logger.info(f"Successfully connected to database: {db_name}")
            return engine
        except Exception as e:
            logger.error(f"Failed to connect to database {db_name}: {e}")
            return None

    def _create_results_table_if_not_exists(self):
        """Creates the analysis_results table in the result DB."""
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS analysis_results (
            id SERIAL PRIMARY KEY,
            original_review_id INTEGER NOT NULL,
            sentiment VARCHAR(20),
            main_topic VARCHAR(50),
            key_drivers JSONB,
            is_actionable BOOLEAN,
            summary TEXT,
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
        Saves analysis results to the new DB and updates status in the source DB.
        """
        if not self.source_engine or not self.result_engine:
            logger.error("Database connections not available. Cannot save results.")
            return

        SourceSession = sessionmaker(bind=self.source_engine)
        ResultSession = sessionmaker(bind=self.result_engine)
        source_session = SourceSession()
        result_session = ResultSession()
        
        saved_count = 0
        for result in analysis_results:
            try:
                original_id = result.get('original_id')
                if not original_id:
                    logger.warning("Skipping result with no original_id.")
                    continue

                # 1. Insert into the new results database
                insert_sql = text("""
                    INSERT INTO analysis_results (original_review_id, sentiment, main_topic, key_drivers, is_actionable, summary)
                    VALUES (:id, :sentiment, :main_topic, :drivers, :actionable, :summary);
                """)
                result_session.execute(insert_sql, {
                    "id": original_id,
                    "sentiment": result['sentiment'],
                    "main_topic": result['main_topic'],
                    "drivers": json.dumps(result['key_drivers']), # Convert dict to JSON string
                    "actionable": result['is_actionable'],
                    "summary": result['summary']
                })

                # 2. Update the status in the original database
                update_sql = text("UPDATE raw_reviews SET analysis_status = 'completed' WHERE id = :id;")
                source_session.execute(update_sql, {"id": original_id})

                # Commit both transactions
                result_session.commit()
                source_session.commit()
                saved_count += 1
            except Exception as e:
                logger.error(f"Error saving result for review ID {original_id}: {e}")
                result_session.rollback()
                source_session.rollback()
        
        logger.info(f"Successfully saved {saved_count} analysis results.")
        source_session.close()
        result_session.close()