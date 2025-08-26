import pandas as pd
import streamlit as st
from typing import Optional
from database.connector import get_db_connection


@st.cache_data(ttl=300, show_spinner=False)
def fetch_table(table_name: str = "analysis_results", database_name = "database") -> pd.DataFrame:
    """
    Fetch all rows from the specified table.

    Args:
        table_name: Name of the table to fetch.

    Returns:
        DataFrame of results (empty on error or if table has no rows).
    """
    conn = get_db_connection(database_name)
    if not conn:
        st.warning("Database connection is not available.")
        return pd.DataFrame()

    try:
        # Note: table_name is interpolated safely by whitelisting characters.
        if not table_name.replace("_", "").isalnum():
            raise ValueError("Invalid table name.")

        query = f'SELECT * FROM "{table_name}";'
        df = pd.read_sql(query, conn)
        # If an 'id' column exists, promote it to the index to avoid duplicate id + auto-index
        if "id" in df.columns:
            try:
                df.set_index("id", inplace=True, drop=True)
            except Exception:
                # If setting index fails for any reason, continue with original df
                pass
    except Exception as e:
        st.error(f"Error fetching data from '{table_name}': {e}")
        return pd.DataFrame()
    else:
        return df
    finally:
        # Do not close the cached connection here; psycopg2 connection will be
        # reused by Streamlit's cache. If needed, manage cursor-level resources
        # within the read_sql call implicitly.
        pass


@st.cache_data(ttl=120, show_spinner=False)
def fetch_count(table_name: str = "analysis_results", database_name = "database") -> Optional[int]:
    """Return COUNT(*) for the specified table, or None on error."""
    conn = get_db_connection(database_name)
    if not conn:
        return None
    try:
        if not table_name.replace("_", "").isalnum():
            raise ValueError("Invalid table name.")
        q = f'SELECT COUNT(*) FROM "{table_name}";'
        result = pd.read_sql(q, conn)
        return int(result.iloc[0, 0]) if not result.empty else 0
    except Exception:
        return None
    finally:
        # Preserve cached connection for reuse.
        pass

