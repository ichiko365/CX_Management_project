import pandas as pd
import streamlit as st
from typing import Optional
import re
from psycopg2.extras import execute_values
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
    conn = None
    try:
        conn = get_db_connection(database_name)
        if not conn:
            st.warning("Database connection is not available.")
            return pd.DataFrame()

        # Test connection
        if hasattr(conn, 'closed') and conn.closed:
            st.warning(f"Database connection to '{database_name}' is closed.")
            return pd.DataFrame()

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
        return df
    except Exception as e:
        st.error(f"Error fetching data from '{table_name}': {e}")
        return pd.DataFrame()
    finally:
        # Close connection since we're using reduced TTL
        if conn:
            try:
                conn.close()
            except:
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
    src_conn = None
    dst_conn = None
    
    try:
        # Clear Streamlit cache for database connections to get fresh connections
        if hasattr(st, 'cache_resource'):
            st.cache_resource.clear()
        
        # Connect to customer database to fetch data
        src_conn = get_customer_db_connection()
        if not src_conn:
            st.error("Unable to connect to customer_database")
            return 0
            
        # Test connection
        if hasattr(src_conn, 'closed') and src_conn.closed:
            st.error("Customer database connection is closed")
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
        
        with src_conn.cursor() as cur:
            cur.execute(query)
            rows = cur.fetchall()
            cols = [desc[0] for desc in cur.description]
        
        if not cols or not rows:
            st.warning("No data found in customer database")
            return 0
            
        # Connect to main database
        dst_conn = get_db_connection("database")
        if not dst_conn:
            st.error("Unable to connect to main database")
            return 0
            
        # Test connection
        if hasattr(dst_conn, 'closed') and dst_conn.closed:
            st.error("Main database connection is closed")
            return 0
            
        # Sanitize column names
        safe_cols = [_sanitize_identifier(c) for c in cols]
        
        with dst_conn.cursor() as cur:
            # Create table if it doesn't exist
            cols_ddl = ", ".join([f'"{c}" TEXT' for c in safe_cols])
            cur.execute(f'''
                CREATE TABLE IF NOT EXISTS "{target_table_name}" (
                    {cols_ddl},
                    UNIQUE(id)
                )
            ''')
            
            # Clear existing data and insert new data (replace strategy)
            cur.execute(f'DELETE FROM "{target_table_name}"')
            
            # Insert new data
            if rows:
                col_names_sql = ", ".join([f'"{c}"' for c in safe_cols])
                insert_sql = f'INSERT INTO "{target_table_name}" ({col_names_sql}) VALUES %s'
                
                # Convert all row values to strings for TEXT columns
                data = [tuple(None if v is None else str(v) for v in row) for row in rows]
                execute_values(cur, insert_sql, data, page_size=500)
            
            dst_conn.commit()
            
        st.success(f"Successfully synced {len(rows)} rows to '{target_table_name}' table")
        return len(rows)
        
    except Exception as e:
        st.error(f"Error syncing complaints data: {e}")
        if dst_conn:
            try:
                dst_conn.rollback()
            except:
                pass
        return 0
        
    finally:
        # Always close connections
        if src_conn:
            try:
                src_conn.close()
            except:
                pass
        if dst_conn:
            try:
                dst_conn.close()
            except:
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
    main_conn = None
    customer_conn = None
    
    try:
        # Clear Streamlit cache for database connections to get fresh connections
        if hasattr(st, 'cache_resource'):
            st.cache_resource.clear()
            
        # First, get complaints data from main database
        main_conn = get_db_connection("database")
        if not main_conn:
            st.warning("Unable to connect to main database")
            return pd.DataFrame()
            
        # Test connection
        if hasattr(main_conn, 'closed') and main_conn.closed:
            st.warning("Main database connection is closed")
            return pd.DataFrame()
            
        # Get customer database connection for departments
        customer_conn = get_customer_db_connection()
        if not customer_conn:
            st.warning("Unable to connect to customer database")
            return pd.DataFrame()
            
        # Test connection
        if hasattr(customer_conn, 'closed') and customer_conn.closed:
            st.warning("Customer database connection is closed")
            return pd.DataFrame()
            
        # Since we can't do cross-database joins directly in PostgreSQL,
        # we'll fetch data from both databases and join in pandas
        
        # Fetch complaints from main database
        complaints_query = 'SELECT * FROM "complaints"'
        complaints_df = pd.read_sql(complaints_query, main_conn)
        
        # Fetch departments from customer database
        departments_query = 'SELECT id, name FROM "departments"'
        departments_df = pd.read_sql(departments_query, customer_conn)
        
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
        # Always close connections
        if main_conn:
            try:
                main_conn.close()
            except:
                pass
        if customer_conn:
            try:
                customer_conn.close()
            except:
                pass

