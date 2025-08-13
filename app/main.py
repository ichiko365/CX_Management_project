import logging
import subprocess
import sys
import threading
from fastapi import FastAPI, BackgroundTasks, HTTPException
from typing import Optional

# Import your Pydantic schemas from the same 'app' folder
from .schema import ReviewInput, ReviewDB

# Import your database connector and logger
from .db_engine import create_db_engine
from cx_dashboard.logger import logger
from sqlalchemy.orm import sessionmaker
from sqlalchemy import text

# --- Application Setup ---
log = logging.getLogger(__name__)
app = FastAPI(
    title="CX Review Ingestion API",
    description="API for submitting new reviews. Triggers the analysis pipeline after every 2 reviews."
)

# --- Database Connections ---
db_engine = create_db_engine(db_name_env="DB_NAME")
if not db_engine:
    log.critical("FATAL: Could not connect to the source database. The API cannot start.")
    raise RuntimeError("Failed to connect to the source database on startup.")
SourceSession = sessionmaker(bind=db_engine)


# --- State Management & Pipeline Trigger ---
BATCH_TRIGGER_SIZE = 2
new_reviews_counter = 0
counter_lock = threading.Lock()

def run_the_pipeline():
    """Triggers the main run_pipeline.py script as a background process."""
    log.info(f"--- TRIGGERING BACKGROUND PIPELINE for a batch of {BATCH_TRIGGER_SIZE} reviews ---")
    python_executable = sys.executable 
    script_path = "run_pipeline.py" 
    try:
        # Use Popen to run the script in a new, non-blocking process
        subprocess.Popen([python_executable, script_path])
        log.info(f"Successfully started '{script_path}' in the background.")
    except Exception as e:
        log.error(f"Failed to start run_pipeline.py: {e}")


import logging
import subprocess
import sys
import threading
from fastapi import FastAPI, BackgroundTasks, HTTPException
from typing import Optional

# Import your Pydantic schemas from the same 'app' folder
from .schema import ReviewInput, ReviewDB

# Import your database connector and logger
from .db_engine import create_db_engine
from cx_dashboard.logger import logger
from sqlalchemy.orm import sessionmaker
from sqlalchemy import text

# --- Application Setup ---
log = logging.getLogger(__name__)
app = FastAPI(
    title="CX Review Ingestion API",
    description="API for submitting new reviews. Triggers the analysis pipeline after every 2 reviews."
)

# --- Database Connections ---
db_engine = create_db_engine(db_name_env="DB_NAME")
if not db_engine:
    log.critical("FATAL: Could not connect to the source database. The API cannot start.")
    raise RuntimeError("Failed to connect to the source database on startup.")
SourceSession = sessionmaker(bind=db_engine)


# --- State Management & Pipeline Trigger ---
BATCH_TRIGGER_SIZE = 2
new_reviews_counter = 0
counter_lock = threading.Lock()

def run_the_pipeline():
    """Triggers the main run_pipeline.py script as a background process."""
    log.info(f"--- TRIGGERING BACKGROUND PIPELINE for a batch of {BATCH_TRIGGER_SIZE} reviews ---")
    python_executable = sys.executable 
    script_path = "run_pipeline.py" 
    try:
        # Use Popen to run the script in a new, non-blocking process
        subprocess.Popen([python_executable, script_path])
        log.info(f"Successfully started '{script_path}' in the background.")
    except Exception as e:
        log.error(f"Failed to start run_pipeline.py: {e}")


@app.post("/add_review/", response_model=ReviewDB, status_code=201)
def add_review_to_queue(review: ReviewInput, background_tasks: BackgroundTasks):
    global new_reviews_counter
    log.info(f"Received new review for ASIN: {review.ASIN}")

    db_session = SourceSession()
    try:
        # Insert review and get DB ID
        insert_sql = text("""
            INSERT INTO raw_reviews ("ASIN", "Title", "Region", "Review", "analysis_status")
            VALUES (:ASIN, :Title, :Region, :Review, 'pending')
            RETURNING id;
        """)
        result = db_session.execute(insert_sql, review.dict())
        db_id = result.scalar_one()  # This is the actual DB primary key ID
        db_session.commit()

        # Count reviews AFTER insertion
        count_sql = text("""SELECT COUNT(*) FROM raw_reviews;""")
        total_reviews = db_session.execute(count_sql).scalar_one()

        log.info(f"Successfully inserted review with ID: {db_id}. Total reviews now: {total_reviews}")

        # Increment batch counter
        with counter_lock:
            new_reviews_counter += 1
            log.info(f"New reviews in current batch: {new_reviews_counter}/{BATCH_TRIGGER_SIZE}")
            if new_reviews_counter >= BATCH_TRIGGER_SIZE:
                background_tasks.add_task(run_the_pipeline)
                new_reviews_counter = 0

        # Return DB ID in the `id` field so Pydantic validation passes
        return ReviewDB(id=db_id, **review.dict())

    except Exception as e:
        db_session.rollback()
        log.error(f"Failed to add new review: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to save review.")
    finally:
        db_session.close()
