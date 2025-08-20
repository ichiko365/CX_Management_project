import os
import logging
from functools import lru_cache
from typing import Dict, Optional
from urllib.parse import quote_plus

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

# Get a logger for this utility module
logger = logging.getLogger(__name__)


# -----------------------------
# Connection URL and Engine API
# -----------------------------
def _build_connection_url(db_name_env: str = "DB_NAME") -> Optional[str]:
    """Build a SQLAlchemy connection URL from environment variables."""
    load_dotenv()

    db_user = os.getenv("DB_USER")
    db_password_raw = os.getenv("DB_PASSWORD")
    db_host = os.getenv("DB_HOST")
    db_port = os.getenv("DB_PORT")
    db_name = os.getenv(db_name_env)

    if not all([db_user, db_password_raw, db_host, db_port, db_name]):
        logger.error(
            "Database configuration is missing. Check .env for: DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, and %s",
            db_name_env,
        )
        return None

    encoded_password = quote_plus(db_password_raw)
    return f"postgresql://{db_user}:{encoded_password}@{db_host}:{db_port}/{db_name}"


@lru_cache(maxsize=2)
def create_db_engine(db_name_env: str = "DB_NAME") -> Optional[Engine]:
    """Create and cache a SQLAlchemy engine. Returns None on failure."""
    url = _build_connection_url(db_name_env)
    if not url:
        return None
    try:
        engine = create_engine(url)
        # Smoke test
        with engine.connect() as _:
            logger.info("Database connection OK for %s", db_name_env)
        return engine
    except Exception as e:
        logger.error("Error creating database engine for %s: %s", db_name_env, e)
        return None


# -----------------------------
# Lightweight data helpers
# -----------------------------
def fetch_product_metadata(asin: str) -> Dict[str, Optional[str]]:
    """
    Return product metadata for a given ASIN using the latest record found in
    'raw_reviews' table. Only returns Title/Description keys when available.

    This is intentionally read-only and lightweight so it can be used from
    Pydantic validators.
    """
    engine = create_db_engine("DB_NAME")
    if engine is None:
        return {}

    sql = text(
        """
        SELECT "Title", "Description"
        FROM raw_reviews
        WHERE "ASIN" = :asin
          AND ("Title" IS NOT NULL OR "Description" IS NOT NULL)
        ORDER BY id DESC
        LIMIT 1;
        """
    )

    try:
        with engine.connect() as conn:
            row = conn.execute(sql, {"asin": asin}).mappings().first()
            if not row:
                return {}
            return {
                "Title": row.get("Title"),
                "Description": row.get("Description"),
            }
    except Exception as e:
        logger.warning("fetch_product_metadata failed for ASIN %s: %s", asin, e)
        return {}