from arrow import get
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import sys
import os
from sqlalchemy import text

# Add the parent directory to sys.path to import from database module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.queries import fetch_table, sync_complaints_from_customer_db
from database.connector import get_customer_db_connection

# Try to import pytz, if not available, provide fallback
try:
    import pytz
except ImportError:
    pytz = None


def refresh_data() -> bool:
    """
    Refresh the data by syncing from customer database.
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Use the queries sync function to refresh data into main DB
        n = sync_complaints_from_customer_db("complaints")
        if n > 0:
            st.success(f"Successfully refreshed data ({n} rows) from customer database")
        else:
            st.info("Refresh completed but no rows were processed")
        return True
    except Exception as e:
        st.error(f"Failed to refresh data: {e}")
        return False


def get_support_data() -> pd.DataFrame:
    """
    Fetch support/complaints data from the database.
    
    Returns:
        pd.DataFrame: Support data
    """
    try:
        df = fetch_table("complaints", "database")
        if df.empty:
            st.warning("No support data found. Try refreshing the data.")
        return df
    except Exception as e:
        st.error(f"Error fetching support data: {e}")
        return pd.DataFrame()
    

def calculate_time_ago(timestamp: str) -> str:
    """
    Convert timestamp to 'X hours ago' format.
    Args:
        timestamp: Timestamp string (may include timezone)
    Returns:
        str: Human readable time difference
    """
    try:
        if isinstance(timestamp, str):
            # Try different datetime formats including timezone-aware formats
            formats_to_try = [
                '%Y-%m-%d %H:%M:%S.%f%z',      # 2025-08-26 13:13:05.174474+05:30
                '%Y-%m-%d %H:%M:%S%z',         # 2025-08-26 13:13:05+05:30  
                '%Y-%m-%d %H:%M:%S.%f',        # 2025-08-26 13:13:05.174474
                '%Y-%m-%d %H:%M:%S',           # 2025-08-26 13:13:05
                '%Y-%m-%dT%H:%M:%S.%f%z',      # ISO format with timezone
                '%Y-%m-%dT%H:%M:%S%z',         # ISO format with timezone
                '%Y-%m-%dT%H:%M:%S.%f',        # ISO format
                '%Y-%m-%dT%H:%M:%S'            # ISO format
            ]
            
            parsed_time = None
            for fmt in formats_to_try:
                try:
                    # Handle timezone colon format (e.g., +05:30 -> +0530)
                    timestamp_str = timestamp
                    if '+' in timestamp_str and ':' in timestamp_str.split('+')[-1]:
                        # Convert +05:30 to +0530 for strptime compatibility
                        parts = timestamp_str.rsplit('+', 1)
                        if len(parts) == 2:
                            tz_part = parts[1].replace(':', '')
                            timestamp_str = parts[0] + '+' + tz_part
                    elif '-' in timestamp_str and timestamp_str.count('-') > 2:
                        # Handle negative timezone like -05:30 -> -0530
                        last_dash = timestamp_str.rfind('-')
                        if ':' in timestamp_str[last_dash:]:
                            tz_part = timestamp_str[last_dash+1:].replace(':', '')
                            timestamp_str = timestamp_str[:last_dash] + '-' + tz_part
                    
                    parsed_time = datetime.strptime(timestamp_str, fmt)
                    break
                except ValueError:
                    continue
            
            if parsed_time is None:
                return "Unknown"
        else:
            parsed_time = timestamp
        
        # Convert timezone-aware datetime to UTC for comparison
        if parsed_time.tzinfo is not None and pytz is not None:
            # Convert to UTC using pytz
            parsed_time_utc = parsed_time.astimezone(pytz.UTC).replace(tzinfo=None)
            now_utc = datetime.now(pytz.UTC).replace(tzinfo=None)
            diff = now_utc - parsed_time_utc
        elif parsed_time.tzinfo is not None:
            # Fallback: convert to naive datetime by removing timezone info
            parsed_time_naive = parsed_time.replace(tzinfo=None)
            diff = datetime.now() - parsed_time_naive
        else:
            # Assume local time if no timezone info
            diff = datetime.now() - parsed_time
        
        total_seconds = int(diff.total_seconds())
        
        if diff.days > 0:
            return f"{diff.days}d ago"
        elif total_seconds >= 3600:  # 1 hour or more
            hours = total_seconds // 3600
            return f"{hours}h ago"
        elif total_seconds >= 60:   # 1 minute or more
            minutes = total_seconds // 60
            return f"{minutes}m ago"
        elif total_seconds > 0:     # Less than a minute but positive
            return f"{total_seconds}s ago"
        else:
            return "Just now"
            
    except Exception as e:
        # For debugging, you can uncomment the line below to see what's wrong
        # st.error(f"Time parsing error for '{timestamp}': {e}")
        return "Unknown"


def calculate_performance_metrics(df: pd.DataFrame) -> dict:
    """
    Calculate performance metrics from support data.
    Args:
        df: Support dataframe
    Returns:
        dict: Performance metrics
    """
    if df.empty:
        return {}
    
    try:
        metrics = {}
        
        # Total items
        metrics['total_items'] = len(df)
        
        # Status breakdown and unresolved count
        if 'status' in df.columns:
            status_counts = df['status'].value_counts().to_dict()
            metrics['status_breakdown'] = status_counts
            # Count unresolved items
            unresolved_statuses = ['open']
            unresolved_count = sum([count for status, count in status_counts.items() 
                                  if status.lower() in unresolved_statuses or 
                                  (status.lower() not in ['closed', 'resolved', 'completed'])])
            metrics['unresolved_items'] = unresolved_count
        else:
            metrics['unresolved_items'] = 0
        
        # Items by team member
        if 'team_member_name' in df.columns:
            team_counts = df['team_member_name'].value_counts().to_dict()
            metrics['team_workload'] = team_counts
        
        # Items by department
        if 'department_name' in df.columns:
            dept_counts = df['department_name'].value_counts().to_dict()
            metrics['department_breakdown'] = dept_counts
            metrics['departments_with_items'] = len(dept_counts)
        
        # Recent items (last 24 hours)
        if 'created_at' in df.columns:
            try:
                # Try to parse all timestamps robustly
                def parse_dt(val):
                    try:
                        return pd.to_datetime(val, utc=True, errors='coerce')
                    except Exception:
                        return pd.NaT
                df['created_at_dt'] = df['created_at'].apply(parse_dt)
                now_utc = pd.Timestamp.utcnow()
                recent_threshold = now_utc - pd.Timedelta(days=1)
                recent_items = df[df['created_at_dt'] >= recent_threshold]
                metrics['recent_24h'] = len(recent_items)
            except Exception:
                metrics['recent_24h'] = 0
        else:
            metrics['recent_24h'] = 0
        
        return metrics
        
    except Exception as e:
        st.error(f"Error calculating performance metrics: {e}")
        return {}

def fetch_team_performance_data() -> pd.DataFrame:
    """
    Get team performance data by joining team_members with support_tasks.
    Returns:
        pd.DataFrame: Team performance data with task counts and department names
    """
    conn = None
    try:
        conn = get_customer_db_connection()
        if not conn:
            # Only show error if running in Streamlit context
            try:
                st.error("Unable to connect to customer database")
            except:
                print("Unable to connect to customer database")
            return pd.DataFrame()
        
        query = """
        SELECT 
            tm.id,
            tm.name,
            tm.email,
            tm.department_id,
            COALESCE(d.name, 'No Department') as department_name,
            COUNT(st.id) as total_tasks,
            SUM(CASE WHEN st.status = 'open' THEN 1 ELSE 0 END) as open_tasks,
            SUM(CASE WHEN st.status = 'closed' THEN 1 ELSE 0 END) as completed_tasks,
            CASE 
                WHEN COUNT(st.id) > 0 
                THEN CAST(ROUND(CAST(SUM(CASE WHEN st.status = 'closed' THEN 1 ELSE 0 END) AS DECIMAL) / COUNT(st.id) * 100, 0) AS INTEGER)
                ELSE 0 
            END as completion_percentage
        FROM team_members tm
        LEFT JOIN departments d ON tm.department_id = d.id
        LEFT JOIN support_tasks st ON tm.id = st.assigned_to_member_id
        GROUP BY tm.id, tm.name, tm.email, tm.department_id, d.name
        ORDER BY total_tasks DESC, completion_percentage DESC
        """
        
        # Check if engine is still valid before using it
        try:
            with conn.connect() as test_conn:
                if test_conn.closed:
                    try:
                        st.error("Database connection was closed unexpectedly")
                    except:
                        print("Database connection was closed unexpectedly")
                    return pd.DataFrame()
        except Exception:
            try:
                st.error("Database connection test failed")
            except:
                print("Database connection test failed")
            return pd.DataFrame()
        
        df = pd.read_sql(query, conn)
        
        # Validate the dataframe
        if df.empty:
            try:
                st.warning("No team performance data found in database")
            except:
                print("No team performance data found in database")
        
        return df
        
    except Exception as e:
        error_msg = f"Error fetching team performance data: {e}"
        try:
            st.error(error_msg)
        except:
            print(error_msg)
        return pd.DataFrame()
    finally:
        # Ensure engine is properly disposed
        if conn:
            try:
                conn.dispose()
            except Exception:
                pass

def calculate_team_efficiency(team_performance_df: pd.DataFrame) -> int:
    """
    Calculate team efficiency as percentage of departments working on tasks.
    Args:
        team_performance_df: Team performance dataframe
    Returns:
        int: Efficiency percentage
    """
    if team_performance_df.empty:
        return 0
    try:
        # Get unique departments
        total_departments = len(team_performance_df['department_name'].unique())
        
        # Get departments that have team members with active tasks (total_tasks > 0)
        active_departments = len(team_performance_df[team_performance_df['total_tasks'] > 0]['department_name'].unique())
        
        if total_departments > 0:
            efficiency = round((active_departments / total_departments) * 100)
            return efficiency
        else:
            return 0
    except Exception as e:
        print(f"Error calculating team efficiency: {e}")
        return 0

def toggle_task_status(team_member_id: int) -> bool:
    """
    Toggle the status of the oldest task for a team member.
    - If the oldest open task exists → mark it closed.
    - Else, if the oldest closed task exists → reopen it.
    - Works only if the team member has tasks assigned (>0).
    
    Args:
        team_member_id: ID of the team member
    Returns:
        bool: True if successful, False otherwise
    """
    conn = None
    try:
        conn = get_customer_db_connection()
        if not conn:
            try:
                st.error("Unable to connect to customer database")
            except:
                print("Unable to connect to customer database")
            return False
        
        # Test connection validity
        try:
            with conn.connect() as test_conn:
                if test_conn.closed:
                    try:
                        st.error("Database connection was closed unexpectedly")
                    except:
                        print("Database connection was closed unexpectedly")
                    return False
        except Exception:
            try:
                st.error("Database connection test failed")
            except:
                print("Database connection test failed")
            return False
        
        # Use proper SQLAlchemy connection for operations
        with conn.connect() as db_conn:
            # Check if team member has any assigned tasks
            result = db_conn.execute(text("""
                SELECT COUNT(*) FROM support_tasks 
                WHERE assigned_to_member_id = :member_id
            """), {"member_id": team_member_id})
            total_tasks = result.scalar()

            if total_tasks == 0:
                try:
                    st.warning("This team member has no assigned tasks yet.")
                except:
                    print("This team member has no assigned tasks yet.")
                return False

            # Find oldest open task (lock to avoid race conditions)
            result = db_conn.execute(text("""
                SELECT id FROM support_tasks 
                WHERE assigned_to_member_id = :member_id AND status = 'open'
                ORDER BY created_at ASC 
                LIMIT 1
                FOR UPDATE SKIP LOCKED
            """), {"member_id": team_member_id})
            
            task = result.fetchone()
            if task:
                db_conn.execute(text("""
                    UPDATE support_tasks 
                    SET status = 'closed'
                    WHERE id = :task_id
                """), {"task_id": task[0]})
                db_conn.commit()
                try:
                    st.success("Task marked as completed!")
                except:
                    print("Task marked as completed!")
                return True

            # Otherwise try reopening the oldest closed task
            result = db_conn.execute(text("""
                SELECT id FROM support_tasks 
                WHERE assigned_to_member_id = :member_id AND status = 'closed'
                ORDER BY created_at ASC 
                LIMIT 1
                FOR UPDATE SKIP LOCKED
            """), {"member_id": team_member_id})
            
            task = result.fetchone()
            if task:
                db_conn.execute(text("""
                    UPDATE support_tasks 
                    SET status = 'open'
                    WHERE id = :task_id
                """), {"task_id": task[0]})
                db_conn.commit()
                try:
                    st.success("Task reopened!")
                except:
                    print("Task reopened!")
                return True
            else:
                try:
                    st.info("No tasks found to toggle.")
                except:
                    print("No tasks found to toggle.")
                return False

    except Exception as e:
        error_msg = f"Error toggling task status: {e}"
        try:
            st.error(error_msg)
        except:
            print(error_msg)
        return False
    finally:
        # Ensure engine is properly disposed
        if conn:
            try:
                conn.dispose()
            except Exception as e:
                print(f"Warning: failed to dispose engine properly: {e}")


if __name__ == "__main__":
    result = calculate_team_efficiency(fetch_team_performance_data())
    print("Here is the answer:")
    print(result)


    
# Utility functions for backward compatibility
def get_complaints_data() -> pd.DataFrame:
    """Backward compatibility wrapper for get_support_data."""
    return get_support_data()


def refresh_complaints_data() -> bool:
    """Backward compatibility wrapper for refresh_data."""
    return refresh_data()


def get_time_ago(created_at: str) -> str:
    """Backward compatibility wrapper for calculate_time_ago."""
    return calculate_time_ago(created_at)


def get_team_performance_metrics(df: pd.DataFrame) -> Dict:
    """Backward compatibility wrapper for calculate_performance_metrics."""
    return calculate_performance_metrics(df)

def get_team_performance_data() -> pd.DataFrame:
    """Backward compatibility wrapper for fetch_team_performance_data."""
    return fetch_team_performance_data()


def toggle_oldest_task_status(team_member_id: int) -> bool:
    """Backward compatibility wrapper for toggle_task_status."""
    return toggle_task_status(team_member_id)