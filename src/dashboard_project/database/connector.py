import streamlit as st
import psycopg2
from typing import Optional, Dict, Any


@st.cache_resource(show_spinner=False)
def get_db_connection():
    """
    Establish and cache a PostgreSQL connection using Streamlit secrets.

    Supports either of these secrets structures:
    1) [connections.postgresql] with keys: user, password, host, port, database
    2) [database] with keys: user, password, host, port, name

    Returns:
        psycopg2 connection or None if connection fails.
    """
    conn_params: Optional[Dict[str, Any]] = None

    try:
        # Prefer the Streamlit connections style used in secrets.toml provided.
        if "connections" in st.secrets and "postgresql" in st.secrets["connections"]:
            cfg = st.secrets["connections"]["postgresql"]
            conn_params = {
                "host": cfg.get("host"),
                "port": cfg.get("port"),
                "dbname": cfg.get("database"),
                "user": cfg.get("user"),
                "password": cfg.get("password"),
            }
        # Backward-compatible fallback to a simple [database] section
        elif "database" in st.secrets:
            cfg = st.secrets["database"]
            conn_params = {
                "host": cfg.get("host"),
                "port": cfg.get("port"),
                "dbname": cfg.get("name"),
                "user": cfg.get("user"),
                "password": cfg.get("password"),
            }
        else:
            st.error(
                "Database credentials not found in .streamlit/secrets.toml.\n"
                "Expected [connections.postgresql] or [database] section."
            )
            return None

        # Attempt connection
        conn = psycopg2.connect(**conn_params)

        # Quick health check to fail fast on bad credentials
        with conn.cursor() as cur:
            cur.execute("SELECT 1;")
            cur.fetchone()

        return conn

    except Exception as e:
        st.error(f"Error connecting to PostgreSQL Database: {e}")
        return None


if __name__ == "__main__":
    # Simple connectivity check when running this file directly
    connection = get_db_connection()
    if connection:
        st.success("Connected to the database successfully!")
        try:
            connection.close()
        except Exception:
            pass
    else:
        st.error("Failed to connect to the database.")