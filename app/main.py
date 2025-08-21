"""
Temporary FastAPI entrypoint that ensures the working directory is the app folder
before importing local modules. Provides the same behavior as main.py:
- POST /add_review/ to append to DB (Title/Description auto-fill via schema).
- Triggers run_pipeline.py (one level above app/) after every BATCH_TRIGGER_SIZE reviews.
- POST /trigger_pipeline to manually run the pipeline.
"""

import os
import sys
import logging
import threading
import subprocess
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi import Query
from sqlalchemy import text
from sqlalchemy.orm import sessionmaker


# -------------------- Ensure CWD = this file's folder --------------------
APP_DIR = Path(__file__).resolve().parent
os.chdir(str(APP_DIR))


# ---------------------------- Imports ------------------------------------
# Support both package-style and script-style imports
os.chdir(os.path.dirname(os.path.abspath(__file__)))

try:
	from .db_engine import create_db_engine
	from .schema import ReviewInput, ReviewDB
except Exception:  # pragma: no cover - fallback when executed as a script
	from db_engine import create_db_engine
	from schema import ReviewInput, ReviewDB


# ---------------------------- App setup ----------------------------------
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("api-temp")

app = FastAPI(
	title="CX Review Ingestion API (temp)",
	description="Accepts new reviews and triggers the analysis pipeline every N reviews.",
)


# ---------------------------- DB setup -----------------------------------
engine = create_db_engine("DB_NAME")
if engine is None:
	log.critical("Cannot start API: database connection failed.")
	raise RuntimeError("Database connection failed on startup")

SessionLocal = sessionmaker(bind=engine)


# ------------------------ Ensure PK sequence is healthy -------------------
def _ensure_raw_reviews_id_sequence() -> None:
	"""Ensure raw_reviews.id has a nextval() default and the sequence is aligned.

	This repairs cases where the table was created without SERIAL or where the
	sequence fell behind the max(id), causing UNIQUE violations on insert.
	"""
	try:
		with engine.begin() as conn:  # begin() provides a txn and commits automatically
			# Check current default on id
			default_sql = text(
				"""
				SELECT column_default
				FROM information_schema.columns
				WHERE table_name = 'raw_reviews' AND column_name = 'id';
				"""
			)
			current_default = conn.execute(default_sql).scalar()

			# Create sequence if needed and set as default
			conn.execute(text("CREATE SEQUENCE IF NOT EXISTS raw_reviews_id_seq;"))
			conn.execute(text("ALTER SEQUENCE raw_reviews_id_seq OWNED BY raw_reviews.id;"))
			if not current_default or "nextval(" not in str(current_default):
				conn.execute(
					text("ALTER TABLE raw_reviews ALTER COLUMN id SET DEFAULT nextval('raw_reviews_id_seq');")
				)

			# Align sequence with current MAX(id)
			max_id = conn.execute(text("SELECT COALESCE(MAX(id), 0) FROM raw_reviews;")).scalar() or 0
			conn.execute(text("SELECT setval('raw_reviews_id_seq', :val)").bindparams(val=int(max_id)))
	except Exception as e:
		log.warning("Could not ensure raw_reviews.id sequence: %s", e)


# Attempt repair on startup
_ensure_raw_reviews_id_sequence()


# ------------------------ Pipeline trigger logic -------------------------
BATCH_TRIGGER_SIZE = int(os.getenv("BATCH_TRIGGER_SIZE", "2"))
_counter_lock = threading.Lock()
_new_reviews_counter = 0


def _start_pipeline_process() -> None:
	project_root = APP_DIR.parent
	script = project_root / "run_pipeline.py"
	if not script.exists():
		log.error("run_pipeline.py not found at %s", script)
		return

	env = os.environ.copy()
	env["PYTHONPATH"] = f"{project_root}{os.pathsep}{env.get('PYTHONPATH','')}".rstrip(os.pathsep)

	logs_dir = project_root / "logs"
	try:
		logs_dir.mkdir(exist_ok=True)
	except Exception:
		pass

	stdout = stderr = subprocess.DEVNULL
	log_file = None
	try:
		log_file = open(logs_dir / "pipeline.log", "a", buffering=1)
		stdout = stderr = log_file
	except Exception:
		pass

	try:
		subprocess.Popen(
			[sys.executable, str(script)],
			cwd=str(project_root),
			env=env,
			stdout=stdout,
			stderr=stderr,
		)
		log.info("Pipeline started in background")
	except Exception as e:
		log.error("Failed to start pipeline: %s", e)
	finally:
		try:
			if log_file and not log_file.closed:
				log_file.close()
		except Exception:
			pass


# ------------------------------ Endpoints --------------------------------
@app.get("/suggest_titles")
def suggest_titles(q: str = Query(..., min_length=2), limit: int = Query(10, ge=1, le=25)):
	"""Return up to `limit` title suggestions matching `q` (case-insensitive).
	For each distinct Title, returns the most recent ASIN/Description.
	"""
	eng = engine  # use existing engine
	try:
		# For each Title, take the latest row (by id) and fall back to the latest non-null Description if needed
		sql = text(
			"""
			WITH latest_per_title AS (
			  SELECT DISTINCT ON ("Title") id, "Title", "ASIN", "Description"
			  FROM raw_reviews
			  WHERE "Title" ILIKE :q
			  ORDER BY "Title", id DESC
			)
			SELECT l."Title",
			       l."ASIN",
			       COALESCE(l."Description", d."Description") AS "Description"
			FROM latest_per_title l
			LEFT JOIN LATERAL (
			  SELECT "Description"
			  FROM raw_reviews r2
			  WHERE r2."Title" = l."Title" AND r2."Description" IS NOT NULL
			  ORDER BY r2.id DESC
			  LIMIT 1
			) d ON true
			LIMIT :limit;
			"""
		)
		with eng.connect() as conn:
			rows = conn.execute(sql, {"q": f"%{q}%", "limit": int(limit)}).mappings().all()
		return [
			{
				"Title": r.get("Title"),
				"ASIN": r.get("ASIN"),
				"Description": r.get("Description"),
			}
			for r in rows
			if r.get("Title")
		]
	except Exception as e:
		log.error("suggest_titles failed: %s", e)
		raise HTTPException(status_code=500, detail="Suggestion lookup failed")

@app.post("/add_review/", response_model=ReviewDB, status_code=201)
def add_review_to_queue(review: ReviewInput, background_tasks: BackgroundTasks):
	global _new_reviews_counter

	# Pydantic v1/v2 compatibility
	try:
		payload: Dict[str, Any] = review.model_dump(by_alias=True)
	except Exception:
		payload = review.dict()

	log.info("New review received for ASIN=%s", payload.get("ASIN"))

	db = SessionLocal()
	try:
		insert_sql = text(
			"""
			INSERT INTO raw_reviews
				("ASIN", "Title", "Description", "Region", "Review", "analysis_status")
			VALUES
				(:ASIN, :Title, :Description, :Region, :Review, 'pending')
			RETURNING id;
			"""
		)
		result = db.execute(insert_sql, payload)
		new_id = result.scalar_one()
		db.commit()

		with _counter_lock:
			_new_reviews_counter += 1
			log.info("Batch counter %s/%s", _new_reviews_counter, BATCH_TRIGGER_SIZE)
			if _new_reviews_counter >= BATCH_TRIGGER_SIZE:
				background_tasks.add_task(_start_pipeline_process)
				_new_reviews_counter = 0

		payload_with_id = {**payload, "id": new_id}
		try:
			return ReviewDB(**payload_with_id)
		except Exception:
			return ReviewDB(
				id=new_id,
				ASIN=payload.get("ASIN"),
				Title=payload.get("Title"),
				Description=payload.get("Description"),
				Review=payload.get("Review"),
				Region=payload.get("Region"),
			)
	except Exception as e:
		db.rollback()
		log.error("Failed to insert review: %s", e, exc_info=True)
		raise HTTPException(status_code=500, detail="Failed to save review")
	finally:
		db.close()


@app.post("/trigger_pipeline", status_code=202)
def trigger_pipeline(background_tasks: BackgroundTasks):
	background_tasks.add_task(_start_pipeline_process)
	return {"status": "queued"}


