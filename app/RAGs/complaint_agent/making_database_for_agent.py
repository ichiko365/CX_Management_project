import os
import logging as log
from urllib.parse import quote_plus
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# Basic logging setup
log.basicConfig(level=log.INFO)

def setup_agent_tables():
    """
    Connects to the database, drops existing agent-related tables for a clean
    setup, creates new tables (users, purchase_history), and inserts sample data.
    """
    log.info("--- Starting Database Setup for Agent ---")
    
    # --- 1. Connect to the Database ---
    load_dotenv()
    db_user = os.getenv("DB_USER")
    db_password_raw = os.getenv("DB_PASSWORD")
    db_host = os.getenv("DB_HOST")
    db_port = os.getenv("DB_PORT")
    db_name = os.getenv("DB_NAME") # We'll use the main DB for this

    if not all([db_user, db_password_raw, db_host, db_port, db_name]):
        log.error("Database configuration is missing in .env file. Aborting.")
        return

    try:
        encoded_password = quote_plus(db_password_raw)
        connection_url = f"postgresql://{db_user}:{encoded_password}@{db_host}:{db_port}/{db_name}"
        engine = create_engine(connection_url)
        
        with engine.connect() as connection:
            log.info("Successfully connected to the database.")
            
            # --- 2. Define SQL Commands ---
            # Use DROP TABLE IF EXISTS for a clean, re-runnable script
            # CORRECTED LINE: Added CASCADE to the DROP TABLE users command
            sql_commands = """
            DROP TABLE IF EXISTS purchase_history;
            DROP TABLE IF EXISTS users CASCADE;

            CREATE TABLE users (
                id SERIAL PRIMARY KEY,
                name VARCHAR(100),
                email VARCHAR(100) UNIQUE NOT NULL
            );

            CREATE TABLE purchase_history (
                id SERIAL PRIMARY KEY,
                user_id INTEGER NOT NULL,
                order_id VARCHAR(50) UNIQUE NOT NULL,
                asin VARCHAR(20) NOT NULL,
                purchase_date DATE NOT NULL,
                CONSTRAINT fk_user
                    FOREIGN KEY(user_id) 
                    REFERENCES users(id)
            );

            -- Insert Sample Data
            INSERT INTO users (id, name, email) VALUES 
            (1, 'Priya Sharma', 'priya@example.com'),
            (2, 'Amit Kumar', 'amit@example.com');

            -- Change the sequence to start after the manually inserted IDs
            ALTER SEQUENCE users_id_seq RESTART WITH 3;

            INSERT INTO purchase_history (user_id, order_id, asin, purchase_date)
            VALUES 
            (1, '123456789', 'B000ASDGK8', '2025-07-20'),
            (1, '112121007', 'B00176GSEI', '2025-06-15');
            """
            
            # --- 3. Execute the SQL ---
            log.info("Dropping old tables and creating new ones...")
            connection.execute(text(sql_commands))
            connection.commit()
            log.info("✅ Successfully created 'users' and 'purchase_history' tables.")
            log.info("✅ Successfully inserted sample data.")

    except Exception as e:
        log.error(f"An error occurred during database setup: {e}", exc_info=True)


def setup_task_tables():
    """
    Creates the tables required for task assignment.
    """
    log.info("--- Setting up Task Assignment tables ---")
    
    # --- 1. Connect to the Database ---
    load_dotenv()
    db_user = os.getenv("DB_USER")
    db_password_raw = os.getenv("DB_PASSWORD")
    db_host = os.getenv("DB_HOST")
    db_port = os.getenv("DB_PORT")
    db_name = os.getenv("DB_NAME") # We'll use the main DB for this

    if not all([db_user, db_password_raw, db_host, db_port, db_name]):
        log.error("Database configuration is missing in .env file. Aborting.")
        return
    
    try:
        encoded_password = quote_plus(db_password_raw)
        connection_url = f"postgresql://{db_user}:{encoded_password}@{db_host}:{db_port}/{db_name}"
        engine = create_engine(connection_url)
        with engine.connect() as connection:
            # NOTE: For this script to work, the 'users' table must exist first.
            # It's good practice to ensure setup_agent_tables() runs before this.
            sql = """
            DROP TABLE IF EXISTS support_tasks;
            DROP TABLE IF EXISTS team_members;
            DROP TABLE IF EXISTS departments;

            CREATE TABLE departments (
                id SERIAL PRIMARY KEY,
                name VARCHAR(100) UNIQUE NOT NULL
            );

            CREATE TABLE team_members (
                id SERIAL PRIMARY KEY,
                name VARCHAR(100),
                email VARCHAR(100) UNIQUE,
                department_id INTEGER NOT NULL REFERENCES departments(id)
            );

            CREATE TABLE support_tasks (
                id SERIAL PRIMARY KEY,
                user_id INTEGER NOT NULL REFERENCES users(id),
                order_id VARCHAR(50),
                assigned_to_member_id INTEGER REFERENCES team_members(id),
                summary TEXT NOT NULL,
                department VARCHAR(50),
                status VARCHAR(20) DEFAULT 'open',
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            );

            INSERT INTO departments (id, name) VALUES
            (1, 'Billing'),
            (2, 'Technical Support'),
            (3, 'Product Quality'),
            (4, 'General Inquiry');
            ALTER SEQUENCE departments_id_seq RESTART WITH 5;

            INSERT INTO team_members (id, name, email, department_id) VALUES
            (1, 'Rohan', 'rohan@support.com', 1),
            (2, 'Sunita', 'sunita@support.com', 2),
            (3, 'Karan', 'karan@support.com', 3),
            (4, 'Neha', 'neha@support.com', 4);
            ALTER SEQUENCE team_members_id_seq RESTART WITH 5;
            """
            connection.execute(text(sql))
            connection.commit()
            log.info("✅ Successfully created 'departments', 'team_members', and 'support_tasks' tables.")
    except Exception as e:
        log.error(f"An error occurred during task table setup: {e}", exc_info=True)

if __name__ == "__main__":
    setup_agent_tables()
    setup_task_tables()