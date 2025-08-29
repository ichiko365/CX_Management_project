# create postgresql tables and feed the data into it.
import os
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from urllib.parse import quote_plus
from pathlib import Path

def _ensure_databases(db_user: str, db_password: str, db_host: str, db_port: str, main_db: str, result_db: str | None = None) -> None:
    """
    Ensure the primary (main_db) and optional result_db databases exist.
    Uses a server-level connection to the default 'postgres' database to create them if missing.
    """
    try:
        encoded_password = quote_plus(db_password)
        server_url = f"postgresql://{db_user}:{encoded_password}@{db_host}:{db_port}/postgres"
        server_engine = create_engine(server_url, isolation_level="AUTOCOMMIT")
        with server_engine.connect() as conn:
            # Create main DB if not exists
            exists = conn.execute(text("SELECT 1 FROM pg_database WHERE datname = :name"), {"name": main_db}).scalar()
            if not exists:
                conn.execute(text(f'CREATE DATABASE "{main_db}"'))
                print(f"🆕 Created database: {main_db}")
            else:
                print(f"✅ Database exists: {main_db}")

            # Create result DB if requested
            if result_db:
                exists_r = conn.execute(text("SELECT 1 FROM pg_database WHERE datname = :name"), {"name": result_db}).scalar()
                if not exists_r:
                    conn.execute(text(f'CREATE DATABASE "{result_db}"'))
                    print(f"🆕 Created result database: {result_db}")
                else:
                    print(f"✅ Result database exists: {result_db}")
    except Exception as e:
        print(f"⚠️ Could not ensure databases (insufficient permissions or server not reachable): {e}")


def load_csv_to_postgres():
    """
    Reads a CSV file, adds 'analysis_status' column,
    and loads the contents into a PostgreSQL table.
    Lets Postgres handle the auto-increment `id`.
    """
    # --- 1. Load config ---
    project_root = Path(__file__).resolve().parent
    # Load .env from project root even if script is run from elsewhere
    load_dotenv(project_root / ".env", override=True)

    db_user = os.getenv("DB_USER")
    db_password = os.getenv("DB_PASSWORD")
    db_host = os.getenv("DB_HOST")
    db_port = os.getenv("DB_PORT")
    db_name = os.getenv("DB_NAME")
    result_db_name = os.getenv("RESULT_DB_NAME")

    # Allow override via env; default to a project-relative path
    csv_file_path = os.getenv("CSV_FILE_PATH", "data/amazon_data_final.csv")
    table_name = "raw_reviews"

    if not all([db_user, db_password, db_host, db_port, db_name]):
        print("Error: Database configuration is missing in the .env file.")
        return

    try:
        # --- 1b. Ensure databases exist (main and optional result) ---
        _ensure_databases(db_user, db_password, db_host, db_port, db_name, result_db_name)

        # --- 2. Read CSV ---
        # Resolve CSV path relative to the project root if not absolute
        csv_path = Path(csv_file_path)
        if not csv_path.is_absolute():
            csv_path = (project_root / csv_path).resolve()

        # Fallbacks if the default doesn't exist
        if not csv_path.exists():
            candidates = [
                project_root / "data" / "amazon_data_final.csv",
                project_root / "src" / "data" / "amazon_data_final.csv",
                project_root / "data" / "sampled_amazon_data.csv",
                project_root / "src" / "data" / "sampled_amazon_data.csv",
            ]
            csv_path = next((p for p in candidates if p.exists()), csv_path)

        print(f"Reading CSV file from '{csv_path}'...")
        df = pd.read_csv(csv_path)
        print(f"✅ Loaded {len(df)} rows from CSV.")

        # --- 3. Add 'analysis_status' column ---
        df["analysis_status"] = "pending"

    # --- 4. Connect to Postgres (main DB) ---
        encoded_password = quote_plus(db_password)
        connection_url = f"postgresql://{db_user}:{encoded_password}@{db_host}:{db_port}/{db_name}"
        engine = create_engine(connection_url)
        print(f"🔌 Connecting to '{db_name}'...")

        # --- 5. Create table if not exists ---
        with engine.connect() as conn:
            conn.execute(text(f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    id SERIAL PRIMARY KEY,
                    "ASIN" TEXT,
                    "Title" TEXT,
                    "Description" TEXT,
                    "ImageURL" TEXT,
                    "Rating" FLOAT,
                    "Verified" BOOLEAN,
                    "ReviewTime" TIMESTAMP,
                    "Review" TEXT,
                    "Summary" TEXT,
                    "Domestic Shipping" TEXT,
                    "International Shipping" TEXT,
                    "Sentiment" FLOAT,
                    "Region" TEXT,
                    analysis_status TEXT
                );
            """))
            conn.commit()

        # --- 6. Append data to table ---
        print(f"📥 Loading data into '{table_name}'...")

        # --- Change ReviewTime to to latest by shifting the time ---
        
        # convert ReviewTime into date format then find the lastest date from df['ReviewTime']
        df['ReviewTime'] = pd.to_datetime(df['ReviewTime'], errors='coerce')
        latest_date = df['ReviewTime'].max()

        # find the no. of days from today
        days_diff = (pd.Timestamp.now() - latest_date).days

        # increment the each ReviewTime by days_diff -1
        df['ReviewTime'] = df['ReviewTime'] + pd.Timedelta(days=days_diff - 1)
        # so, the latest ReviewTime is now the yesterday's date

        df.to_sql(
            name=table_name,
            con=engine,
            if_exists="append",
            index=False
        )

        print(f"✅ Success! {len(df)} rows inserted into '{table_name}'.")

        # --- 7. Align the sequence with MAX(id) to avoid duplicate key on future inserts ---
        try:
            with engine.begin() as conn:
                # Ensure sequence exists and is owned by the id column
                conn.execute(text("CREATE SEQUENCE IF NOT EXISTS raw_reviews_id_seq;"))
                conn.execute(text("ALTER SEQUENCE raw_reviews_id_seq OWNED BY raw_reviews.id;"))
                conn.execute(text("ALTER TABLE raw_reviews ALTER COLUMN id SET DEFAULT nextval('raw_reviews_id_seq');"))
                conn.execute(text("SELECT setval('raw_reviews_id_seq', COALESCE((SELECT MAX(id) FROM raw_reviews), 0));"))
            print("🛠️ Sequence raw_reviews_id_seq aligned with MAX(id).")
        except Exception as seq_err:
            print(f"⚠️ Could not adjust sequence: {seq_err}")

    except FileNotFoundError:
        print(f"❌ Error: CSV file '{csv_file_path}' not found.")
    except Exception as e:
        print(f"❌ An error occurred: {e}")

if __name__ == "__main__":
    load_csv_to_postgres()