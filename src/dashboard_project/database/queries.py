import pandas as pd
import streamlit as st
from typing import Optional
import re
from sqlalchemy import text
from database.connector import get_db_connection, get_customer_db_connection


@st.cache_data(ttl=60, show_spinner=False)  # Reduced TTL to avoid stale connections
def fetch_table(table_name: str = "analysis_results", database_name = "database") -> pd.DataFrame:
    """
    Fetch all rows from the specified table.

    Args:
        table_name: Name of the table to fetch.

    Returns:
        DataFrame of results (empty on error or if table has no rows).
    """
    engine = None
    try:
        engine = get_db_connection(database_name)
        if not engine:
            st.warning("Database connection is not available.")
            return pd.DataFrame()

        # Note: table_name is interpolated safely by whitelisting characters.
        if not table_name.replace("_", "").isalnum():
            raise ValueError("Invalid table name.")

        query = f'SELECT * FROM "{table_name}";'
        df = pd.read_sql(query, engine)
        # If an 'id' column exists, promote it to the index to avoid duplicate id + auto-index
        if "id" in df.columns:
            try:
                df.set_index("id", inplace=True, drop=True)
            except Exception:
                # If setting index fails for any reason, continue with original df
                pass
        return df
    except Exception as e:
        st.error(f"Error fetching data from '{table_name}': {e}")
        return pd.DataFrame()
    finally:
        # SQLAlchemy engines handle connection pooling, no need to close
        pass


@st.cache_data(ttl=120, show_spinner=False)
def fetch_count(table_name: str = "analysis_results", database_name = "database") -> Optional[int]:
    """Return COUNT(*) for the specified table, or None on error."""
    engine = get_db_connection(database_name)
    if not engine:
        return None
    try:
        if not table_name.replace("_", "").isalnum():
            raise ValueError("Invalid table name.")
        q = f'SELECT COUNT(*) FROM "{table_name}";'
        result = pd.read_sql(q, engine)
        return int(result.iloc[0, 0]) if not result.empty else 0
    except Exception:
        return None
    finally:
        # SQLAlchemy engines handle connection pooling, no need to close
        pass


def _sanitize_identifier(name: str) -> str:
    """Make a safe Postgres identifier. Keep alnum/underscore, lowercased.
    Always wrap with double quotes when used to preserve exact casing.
    """
    if not name or not isinstance(name, str):
        return "col"
    
    s = re.sub(r"[^a-zA-Z0-9_]+", "_", name.strip()).strip("_").lower()
    if not s:
        s = "col"
    return s


def sync_complaints_from_customer_db(target_table_name: str = "complaints") -> int:
    """
    Fetch joined support data from customer_database and sync to main database.
    
    This function:
    1. Fetches data from customer_database using the join query from complaints.py
    2. Updates or creates the complaints table in the main database
    3. Uses replace strategy to handle existing data
    
    Args:
        target_table_name: Name of the table to create/update in main database
        
    Returns:
        Number of rows processed
    """
    src_engine = None
    dst_engine = None
    
    try:
        # Clear Streamlit cache for database connections to get fresh connections
        if hasattr(st, 'cache_resource'):
            st.cache_resource.clear()
        
        # Connect to customer database to fetch data
        src_engine = get_customer_db_connection()
        if not src_engine:
            st.error("Unable to connect to customer_database")
            return 0
            
        # Execute the join query from complaints.py
        query = """
            SELECT
              st.id,
              st.user_id,
              COALESCE(u.name, 'Unknown User') AS user_name,
              st.order_id,
              tm.id AS team_member_id,
              COALESCE(tm.name, 'Unassigned') AS team_member_name,
              d.id AS department_id,
              COALESCE(d.name, 'No Department') AS department_name,
              COALESCE(st.status, 'unknown') AS status,
              COALESCE(st.summary, 'No summary') AS summary,
              st.created_at
            FROM support_tasks st
            LEFT JOIN team_members tm ON st.assigned_to_member_id = tm.id
            LEFT JOIN departments d ON tm.department_id = d.id
            LEFT JOIN users u ON st.user_id = u.id
            ORDER BY st.created_at DESC
        """
        
        # Use SQLAlchemy engine to fetch data
        df = pd.read_sql(query, src_engine)
        
        if df.empty:
            st.warning("No data found in customer database")
            return 0
            
        # Connect to main database
        dst_engine = get_db_connection("database")
        if not dst_engine:
            st.error("Unable to connect to main database")
            return 0
            
        # Sanitize column names
        safe_cols = [_sanitize_identifier(c) for c in df.columns]
        df.columns = safe_cols
        
        # Use SQLAlchemy engine with proper connection handling
        with dst_engine.connect() as conn:
            # Create table if it doesn't exist
            cols_ddl = ", ".join([f'"{c}" TEXT' for c in safe_cols])
            conn.execute(text(f'''
                CREATE TABLE IF NOT EXISTS "{target_table_name}" (
                    {cols_ddl},
                    UNIQUE(id)
                )
            '''))
            
            # Clear existing data and insert new data (replace strategy)
            conn.execute(text(f'DELETE FROM "{target_table_name}"'))
            conn.commit()
            
            # Use pandas to_sql for easier data insertion
            df.to_sql(target_table_name, conn, if_exists='append', index=False, method='multi')
            conn.commit()
            
        st.success(f"Successfully synced {len(df)} rows to '{target_table_name}' table")
        return len(df)
        
    except Exception as e:
        st.error(f"Error syncing complaints data: {e}")
        return 0
        
    finally:
        # SQLAlchemy engines handle connection pooling, no need to manually close
        pass


def fetch_complaints_with_departments() -> pd.DataFrame:
    """
    Fetch complaints data with department information using RIGHT JOIN.
    
    This function:
    1. Fetches complaints from main database (complaints table)
    2. RIGHT JOINs with departments from customer_database
    3. Returns all departments, including those without complaints
    
    Returns:
        DataFrame with complaints and department data
    """
    main_engine = None
    customer_engine = None
    
    try:
        # Clear Streamlit cache for database connections to get fresh connections
        if hasattr(st, 'cache_resource'):
            st.cache_resource.clear()
            
        # First, get complaints data from main database
        main_engine = get_db_connection("database")
        if not main_engine:
            st.warning("Unable to connect to main database")
            return pd.DataFrame()
            
        # Get customer database connection for departments
        customer_engine = get_customer_db_connection()
        if not customer_engine:
            st.warning("Unable to connect to customer database")
            return pd.DataFrame()
            
        # Since we can't do cross-database joins directly in PostgreSQL,
        # we'll fetch data from both databases and join in pandas
        
        # Fetch complaints from main database
        complaints_query = 'SELECT * FROM "complaints"'
        complaints_df = pd.read_sql(complaints_query, main_engine)
        
        # Fetch departments from customer database
        departments_query = 'SELECT id, name FROM "departments"'
        departments_df = pd.read_sql(departments_query, customer_engine)
        
        # Perform RIGHT JOIN in pandas (all departments, matching complaints where available)
        if not complaints_df.empty and not departments_df.empty:
            # Convert department_id to match type for join
            if 'department_id' in complaints_df.columns:
                complaints_df['department_id'] = complaints_df['department_id'].astype(str)
            departments_df['id'] = departments_df['id'].astype(str)
            
            # Right join - all departments, matching complaints where available
            result_df = complaints_df.merge(
                departments_df,
                left_on='department_id',
                right_on='id',
                how='right',
                suffixes=('_complaint', '_dept')
            )
            
            # Rename department columns for clarity
            if 'name_dept' in result_df.columns:
                result_df.rename(columns={'name_dept': 'department_name_actual'}, inplace=True)
            if 'id_dept' in result_df.columns:
                result_df.rename(columns={'id_dept': 'department_id_actual'}, inplace=True)
                
        elif not departments_df.empty:
            # No complaints but we have departments
            result_df = departments_df.copy()
            result_df.rename(columns={'name': 'department_name_actual', 'id': 'department_id_actual'}, inplace=True)
        else:
            # No data in either table
            result_df = pd.DataFrame()
            
        return result_df
        
    except Exception as e:
        st.error(f"Error fetching complaints with departments: {e}")
        return pd.DataFrame()
        
    finally:
        # SQLAlchemy engines handle connection pooling, no need to manually close
        pass