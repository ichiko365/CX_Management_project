from typing import List, Tuple, Optional
import re
import sys
import os
from psycopg2.extras import execute_values

# Add the current directory to sys.path to import connector
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from connector import get_db_connection, get_customer_db_connection

# Force refresh of imports to fix caching issue


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


def fetch_joined_support_data() -> Tuple[List[str], List[Tuple]]:
    """Pull rows by joining support_tasks -> team_members -> departments on customer DB.

    Returns:
        (columns, rows) where columns are the result column names and rows is a list of tuples.
    """
    src = None
    try:
        src = get_customer_db_connection()
        if not src:
            raise RuntimeError("Unable to connect to customer_database")
        
        # Check if connection is valid
        if hasattr(src, 'closed') and src.closed:
            raise RuntimeError("Customer database connection is closed")
        
        with src.cursor() as cur:
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
            
            cur.execute(query)
            rows = cur.fetchall()
            cols = [desc[0] for desc in cur.description]
            
            print(f"Fetched {len(rows)} rows with {len(cols)} columns from customer database")
            return cols, rows
            
    except Exception as e:
        print(f"Error fetching joined support data: {e}")
        raise
    finally:
        if src and hasattr(src, 'closed') and not src.closed:
            try:
                src.close()
            except Exception as e:
                print(f"Warning: Error closing customer database connection: {e}")


def sync_to_main_database(table_name: str = "complaints") -> int:
    """Create/replace 'complaints' table in main DB with joined data from customer DB.

    - Drops existing table if present.
    - Creates a new table with TEXT columns matching the source columns.
    - Inserts all rows using batch insert.

    Returns:
        Number of rows inserted.
    """
    dst = None
    try:
        # Fetch data from customer database
        cols, rows = fetch_joined_support_data()
        if not cols:
            print("No columns returned from source query.")
            return 0
        
        if not rows:
            print("No data rows returned from source query.")
            # Still create empty table for consistency
        
        # Connect to main database
        dst = get_db_connection("database")
        if not dst:
            raise RuntimeError("Unable to connect to main database (database)")
        
        # Check if connection is valid
        if hasattr(dst, 'closed') and dst.closed:
            raise RuntimeError("Main database connection is closed")
        
        # Process data
        safe_cols = [_sanitize_identifier(c) for c in cols]
        cols_ddl = ", ".join([f'"{c}" TEXT' for c in safe_cols])
        
        with dst.cursor() as cur:
            # Drop and recreate table
            print(f"Dropping existing table '{table_name}' if it exists...")
            cur.execute(f'DROP TABLE IF EXISTS "{table_name}"')
            
            print(f"Creating new table '{table_name}' with {len(safe_cols)} columns...")
            cur.execute(f'CREATE TABLE "{table_name}" ({cols_ddl})')
            
            # Insert data if available
            if rows:
                col_names_sql = ", ".join([f'"{c}"' for c in safe_cols])
                insert_sql = f'INSERT INTO "{table_name}" ({col_names_sql}) VALUES %s'
                
                # Convert all row values to strings (or None) for TEXT columns
                def _coerce_row(r: Tuple) -> Tuple:
                    return tuple(None if (v is None) else str(v) for v in r)
                
                data = [_coerce_row(r) for r in rows]
                print(f"Inserting {len(data)} rows into '{table_name}' table...")
                
                # Use smaller page size for better memory management
                execute_values(cur, insert_sql, data, page_size=500)
            
            # Commit the transaction
            dst.commit()
            print(f"Successfully synchronized {len(rows)} rows to '{table_name}' table")
        
        return len(rows)
        
    except Exception as e:
        print(f"Error syncing to main database: {e}")
        if dst:
            try:
                dst.rollback()
                print("Transaction rolled back due to error")
            except Exception as rollback_error:
                print(f"Error during rollback: {rollback_error}")
        raise
    finally:
        if dst and hasattr(dst, 'closed') and not dst.closed:
            try:
                dst.close()
            except Exception as e:
                print(f"Warning: Error closing main database connection: {e}")


def main() -> None:
    """Main function to sync complaints data from customer DB to main DB."""
    try:
        print("Starting complaints data synchronization...")
        print("=" * 50)
        
        inserted = sync_to_main_database("complaints")
        
        print("=" * 50)
        if inserted > 0:
            print(f"✓ Successfully synced {inserted} rows into 'complaints' table.")
        else:
            print("✓ Synchronization completed (no data to sync).")
            
    except Exception as e:
        print("=" * 50)
        print(f"✗ Sync failed: {e}")
        print("Please check your database connections and try again.")
        sys.exit(1)


if __name__ == "__main__":
    main()