from typing import Dict, Any
import os
import logging as log
from urllib.parse import quote_plus
from sqlalchemy import create_engine, text
# Generate a summary (using your existing summarization approach)
from langchain_core.messages import HumanMessage

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from model import LLMManager

# Basic logging setup
log.basicConfig(level=log.INFO)

def get_db_connection():
    """Create and return a database connection."""
    db_user = os.getenv("DB_USER")
    db_password_raw = os.getenv("DB_PASSWORD")
    db_host = os.getenv("DB_HOST")
    db_port = os.getenv("DB_PORT")
    db_name = os.getenv("DB_NAME")

    if not all([db_user, db_password_raw, db_host, db_port, db_name]):
        log.error("Database configuration is missing in .env file.")
        return None

    try:
        encoded_password = quote_plus(db_password_raw)
        connection_url = f"postgresql://{db_user}:{encoded_password}@{db_host}:{db_port}/{db_name}"
        engine = create_engine(connection_url)
        return engine
    except Exception as e:
        log.error(f"Failed to create database connection: {e}")
        return None

def verify_order_id_into_db(order_id: str) -> Dict[str, Any]:
    """
    Verify if an order ID exists in the purchase_history table.
    Returns a dictionary with verification status and order details.
    """
    log.info(f"Verifying order ID: {order_id}")
    
    engine = get_db_connection()
    if not engine:
        return {"valid": False, "error": "Database connection failed"}
    
    try:
        with engine.connect() as connection:
            # Query to check if order exists and get user details
            query = text("""
                SELECT ph.order_id, ph.asin, ph.purchase_date, u.name, u.email, u.id as user_id
                FROM purchase_history ph
                JOIN users u ON ph.user_id = u.id
                WHERE ph.order_id = :order_id
            """)
            
            result = connection.execute(query, {"order_id": order_id})
            order_data = result.mappings().first()
            
            if order_data:
                log.info(f"Order ID {order_id} verified successfully")
                return {
                    "valid": True,
                    "order_id": order_data["order_id"],
                    "asin": order_data["asin"],
                    "purchase_date": order_data["purchase_date"].isoformat() if order_data["purchase_date"] else None,
                    "customer_name": order_data["name"],
                    "customer_email": order_data["email"],
                    "user_id": order_data["user_id"]
                }
            else:
                log.warning(f"Order ID {order_id} not found in database")
                return {"valid": False, "error": "Order ID not found"}
                
    except Exception as e:
        log.error(f"Error verifying order ID {order_id}: {e}")
        return {"valid": False, "error": f"Database error: {str(e)}"}
    
    finally:
        engine.dispose()

def log_complaint_to_db_full(order_id: str, department: str, complaint_details: str, complaint_summary: str) -> Dict[str, Any]:
    """
    Log a complaint to the support_tasks table in the database.
    Returns a dictionary with logging status and task details.
    """
    log.info(f"Logging complaint for order {order_id} to {department} department")
    
    # First verify the order exists
    order_verification = verify_order_id_into_db(order_id)
    if not order_verification["valid"]:
        return {
            "success": False,
            "error": f"Cannot log complaint: {order_verification.get('error', 'Order verification failed')}",
            "task_id": None
        }
    
    engine = get_db_connection()
    if not engine:
        return {"success": False, "error": "Database connection failed", "task_id": None}
    
    try:
        with engine.connect() as connection:
            # Get user_id from order verification
            user_id = order_verification["user_id"]
            
            # Find an appropriate team member for the department
            team_member_query = text("""
                SELECT tm.id, tm.name, tm.email
                FROM team_members tm
                JOIN departments d ON tm.department_id = d.id
                WHERE d.name ILIKE :department
                LIMIT 1
            """)
            
            team_member_result = connection.execute(
                team_member_query, 
                {"department": f"%{department}%"}
            )
            team_member = team_member_result.mappings().first()
            
            assigned_to = team_member["id"] if team_member else None
            
            # Insert the complaint into support_tasks
            insert_query = text("""
                INSERT INTO support_tasks 
                (user_id, order_id, assigned_to_member_id, summary, department, status)
                VALUES (:user_id, :order_id, :assigned_to, :summary, :department, 'open')
                RETURNING id, created_at
            """)
            
            result = connection.execute(
                insert_query,
                {
                    "user_id": user_id,
                    "order_id": order_id,
                    "assigned_to": assigned_to,
                    "summary": complaint_summary,
                    "department": department
                }
            )
            
            task_data = result.mappings().first()
            connection.commit()
            
            log.info(f"Complaint logged successfully for order {order_id}, task ID: {task_data['id']}")
            
            return {
                "success": True,
                "task_id": task_data["id"],
                "created_at": task_data["created_at"].isoformat(),
                "assigned_to": team_member["name"] if team_member else "Unassigned",
                "department": department,
                "order_details": order_verification
            }
            
    except Exception as e:
        log.error(f"Error logging complaint for order {order_id}: {e}")
        connection.rollback()
        return {"success": False, "error": f"Database error: {str(e)}", "task_id": None}
    
    finally:
        engine.dispose()


# Define tools (these would be implemented with your actual backend)
def verify_order_id(order_id: str) -> bool:
    """
    Verify if an order ID is valid.
    Compatible with your existing agent's expected return type.
    """
    result = verify_order_id_into_db(order_id)
    return result["valid"]

def log_complaint_to_db(order_id: str, department: str, complaint_details: str) -> bool:
    """
    Log a complaint to the database.
    Compatible with your existing agent's expected return type.
    """
    llm = LLMManager().get_client()
    summary_prompt = f"Summarize this complaint in 1-2 sentences: {complaint_details}"
    response = llm.invoke([HumanMessage(content=summary_prompt)])
    complaint_summary = response.content
    
    # Log to database
    result = log_complaint_to_db_full(order_id, department, complaint_details, complaint_summary)
    return result["success"]