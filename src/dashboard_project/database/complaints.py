from typing import List, Tuple
import re
from psycopg2.extras import execute_values

from connector import get_db_connection, get_customer_db_connection


def _sanitize_identifier(name: str) -> str:
    """Make a safe Postgres identifier. Keep alnum/underscore, lowercased.
    Always wrap with double quotes when used to preserve exact casing.
    """
    s = re.sub(r"[^a-zA-Z0-9_]+", "_", name.strip()).strip("_").lower()
    if not s:
        s = "col"
    return s


def fetch_joined_support_data() -> Tuple[List[str], List[Tuple]]:
    """Pull rows by joining support_tasks -> team_members -> departments on customer DB.

    Returns:
        (columns, rows) where columns are the result column names and rows is a list of tuples.
    """
    src = get_customer_db_connection()
    if not src:
        raise RuntimeError("Unable to connect to customer_database")
    try:
        with src.cursor() as cur:
            cur.execute(
                """
                SELECT
                  st.id,
                  st.user_id,
                  u.name AS user_name,
                  st.order_id,
                  tm.id AS team_member_id,
                  tm.name AS team_member_name,
                  d.id  AS department_id,
                  d.name  AS department_name,
                  st.status,
                  st.summary,
                  st.created_at
                FROM support_tasks st
                LEFT JOIN team_members tm ON st.assigned_to_member_id = tm.id
                LEFT JOIN departments d   ON tm.department_id = d.id
                LEFT JOIN users u ON st.user_id = u.id;
                """
            )
            rows = cur.fetchall()
            cols = [desc[0] for desc in cur.description]
            return cols, rows
    finally:
        try:
            src.close()
        except Exception:
            pass


def sync_to_main_database(table_name: str = "complaints") -> int:
    """Create/replace 'complaints' table in main DB with joined data from customer DB.

    - Drops existing table if present.
    - Creates a new table with TEXT columns matching the source columns.
    - Inserts all rows using batch insert.

    Returns:
        Number of rows inserted.
    """
    cols, rows = fetch_joined_support_data()
    if not cols:
        print("No columns returned from source query.")
        return 0

    dst = get_db_connection("database")
    if not dst:
        raise RuntimeError("Unable to connect to main database (database)")
    try:
        with dst:
            with dst.cursor() as cur:
                # Recreate table with TEXT columns for broad compatibility
                safe_cols = [_sanitize_identifier(c) for c in cols]
                cols_ddl = ", ".join([f'"{c}" TEXT' for c in safe_cols])
                cur.execute(f"DROP TABLE IF EXISTS \"{table_name}\";")
                cur.execute(f"CREATE TABLE \"{table_name}\" ({cols_ddl});")

                # Build INSERT statement
                col_names_sql = ", ".join([f'"{c}"' for c in safe_cols])
                insert_sql = f"INSERT INTO \"{table_name}\" ({col_names_sql}) VALUES %s"

                # Convert all row values to strings (or None) for TEXT columns
                def _coerce_row(r: Tuple) -> Tuple:
                    return tuple(None if (v is None) else str(v) for v in r)

                data = [_coerce_row(r) for r in rows]
                if data:
                    execute_values(cur, insert_sql, data, page_size=1000)
        return len(rows)
    finally:
        try:
            dst.close()
        except Exception:
            pass


def main() -> None:
    try:
        inserted = sync_to_main_database("complaints")
        print(f"Synced {inserted} rows into 'complaints' table.")
    except Exception as e:
        print(f"Sync failed: {e}")


if __name__ == "__main__":
    main()