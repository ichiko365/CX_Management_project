import streamlit as st
from sqlalchemy import create_engine, text
from typing import Optional, Dict, Any
from pathlib import Path
import sys
import types

try:
    import tomllib  # Python 3.11+
except Exception:  # pragma: no cover
    tomllib = None
    try:
        import toml  # type: ignore
    except Exception:
        toml = None  # type: ignore


@st.cache_resource(show_spinner=False)
def get_db_connection(db_key: str = "database"):
    """
    Establish and cache a PostgreSQL SQLAlchemy engine using Streamlit secrets.

    Parameters:
        db_key: Which DB name to use from secrets. For [connections.postgresql],
                valid keys are typically "database" or "customer_database".
                For legacy [database] section, "database" uses 'name', while
                "customer_database" will try 'customer_database' or 'customer_name'.

    Returns:
        SQLAlchemy Engine (safe for pandas.read_sql).
    """
    conn_params: Optional[Dict[str, Any]] = None

    def _print_err(msg: str) -> None:
        try:
            st.error(msg)
        except Exception:
            print(msg, file=sys.stderr)

    def _load_secrets() -> Optional[Dict[str, Any]]:
        # Try Streamlit secrets first
        try:
            if hasattr(st, "secrets") and isinstance(st.secrets, (dict, types.MappingProxyType)) and st.secrets:
                return dict(st.secrets)
        except Exception:
            pass
        # Fallback to local .streamlit/secrets.toml relative to this file
        try:
            base = Path(__file__).resolve().parents[1]  # .../src/dashboard_project
            secrets_path = base / ".streamlit" / "secrets.toml"
            if secrets_path.exists():
                data: Dict[str, Any]
                if tomllib is not None:
                    data = tomllib.loads(secrets_path.read_text())  # type: ignore[arg-type]
                elif 'toml' in globals() and toml is not None:  # type: ignore[name-defined]
                    data = toml.loads(secrets_path.read_text())  # type: ignore[attr-defined]
                else:
                    _print_err("No tomllib/toml available to parse secrets.toml")
                    return None
                return data
        except Exception as e:
            _print_err(f"Failed to read secrets.toml: {e}")
        return None

    try:
        secrets_obj = _load_secrets() or {}
        # Prefer the Streamlit connections style used in secrets.toml provided.
        if "connections" in secrets_obj and "postgresql" in secrets_obj["connections"]:
            cfg = secrets_obj["connections"]["postgresql"]
            dbname = cfg.get(db_key) or cfg.get("database")
            conn_params = {
                "host": cfg.get("host"),
                "port": cfg.get("port"),
                "dbname": dbname,
                "user": cfg.get("user"),
                "password": cfg.get("password"),
            }
        # Backward-compatible fallback to a simple [database] section
        elif "database" in secrets_obj:
            cfg = secrets_obj["database"]
            if db_key == "database":
                dbname = cfg.get("name") or cfg.get("database")
            else:
                dbname = cfg.get("customer_database") or cfg.get("customer_name") or cfg.get("name")
            conn_params = {
                "host": cfg.get("host"),
                "port": cfg.get("port"),
                "dbname": dbname,
                "user": cfg.get("user"),
                "password": cfg.get("password"),
            }
        else:
            _print_err(
                "Database credentials not found in .streamlit/secrets.toml.\n"
                "Expected [connections.postgresql] or [database] section."
            )
            return None

        if not conn_params or not conn_params.get("dbname"):
            st.error(f"Database name for key '{db_key}' not found in secrets.")
            return None

        # Build SQLAlchemy engine
        engine_url = (
            f"postgresql+psycopg2://{conn_params['user']}:{conn_params['password']}"
            f"@{conn_params['host']}:{conn_params['port']}/{conn_params['dbname']}"
        )
        engine = create_engine(engine_url)

        # Quick health check
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))

        return engine

    except Exception as e:
        _print_err(f"Error connecting to PostgreSQL ({db_key}): {e}")
        return None


def get_customer_db_connection():
    """Convenience wrapper to connect to the 'customer_database' from secrets."""
    return get_db_connection("customer_database")


if __name__ == "__main__":
    # Simple connectivity check when running this file directly
    for key in ("database", "customer_database"):
        conn = get_db_connection(key)
        if conn:
            st.success(f"Connected to '{key}' successfully!")
            try:
                with conn.connect() as cur:
                    dbn = cur.execute(text("SELECT current_database();")).scalar()
                st.write(f"current_database() = {dbn}")
            except Exception:
                pass
        else:
            st.error(f"Failed to connect to '{key}'.")
