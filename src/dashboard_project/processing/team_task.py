import pandas as pd
import streamlit as st
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import sys
import os

# Add the parent directory to sys.path to import from database module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.queries import fetch_table
from database.complaints import sync_to_main_database, main as complaints_main
from database.connector import get_customer_db_connection

# Try to import pytz, if not available, provide fallback
try:
    import pytz
except ImportError:
    pytz = None


def refresh_complaints_data() -> bool:
    """
    Refresh the complaints table by syncing data from customer database.
    Uses the main() function from complaints.py to properly sync data.
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Use the main function from complaints.py which handles the full sync process
        complaints_main()
        st.success("Successfully refreshed complaints data from customer database")
        return True
    except Exception as e:
        st.error(f"Failed to refresh complaints data: {e}")
        return False


def get_complaints_data() -> pd.DataFrame:
    """
    Fetch complaints data from the database.
    
    Returns:
        pd.DataFrame: Complaints data
    """
    try:
        df = fetch_table("complaints", "database")
        if df.empty:
            st.warning("No complaints data found. Try refreshing the data.")
        return df
    except Exception as e:
        st.error(f"Error fetching complaints data: {e}")
        return pd.DataFrame()


def categorize_priority(status: str, created_at: str) -> str:
    """
    Categorize priority based on status and how old the complaint is.
    
    Args:
        status: Current status of the complaint
        created_at: When the complaint was created
        
    Returns:
        str: Priority level (Critical, High, Medium, Low)
    """
    try:
        # Parse the created_at timestamp
        if isinstance(created_at, str):
            # Try different datetime formats
            for fmt in ['%Y-%m-%d %H:%M:%S', '%Y-%m-%d %H:%M:%S.%f', '%Y-%m-%dT%H:%M:%S']:
                try:
                    created_time = datetime.strptime(created_at, fmt)
                    break
                except ValueError:
                    continue
            else:
                # If all formats fail, assume it's recent
                created_time = datetime.now() - timedelta(hours=1)
        else:
            created_time = created_at
        
        # Calculate age in hours
        age_hours = (datetime.now() - created_time).total_seconds() / 3600
        
        # Priority logic
        if status and status.lower() in ['urgent', 'critical', 'escalated']:
            return 'Critical'
        elif age_hours < 2:  # Less than 2 hours old
            return 'Critical'
        elif age_hours < 6:  # Less than 6 hours old
            return 'High'
        elif age_hours < 24:  # Less than 24 hours old
            return 'Medium'
        else:
            return 'Low'
            
    except Exception:
        return 'Medium'  # Default priority


def get_time_ago(created_at: str) -> str:
    """
    Convert timestamp to 'X hours ago' format.
    
    Args:
        created_at: Timestamp string (may include timezone)
        
    Returns:
        str: Human readable time difference
    """
    try:
        if isinstance(created_at, str):
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
            
            created_time = None
            for fmt in formats_to_try:
                try:
                    # Handle timezone colon format (e.g., +05:30 -> +0530)
                    timestamp_str = created_at
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
                    
                    created_time = datetime.strptime(timestamp_str, fmt)
                    break
                except ValueError:
                    continue
            
            if created_time is None:
                return "Unknown"
        else:
            created_time = created_at
        
        # Convert timezone-aware datetime to UTC for comparison
        if created_time.tzinfo is not None and pytz is not None:
            # Convert to UTC using pytz
            created_time_utc = created_time.astimezone(pytz.UTC).replace(tzinfo=None)
            now_utc = datetime.now(pytz.UTC).replace(tzinfo=None)
            diff = now_utc - created_time_utc
        elif created_time.tzinfo is not None:
            # Fallback: convert to naive datetime by removing timezone info
            created_time_naive = created_time.replace(tzinfo=None)
            diff = datetime.now() - created_time_naive
        else:
            # Assume local time if no timezone info
            diff = datetime.now() - created_time
        
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
        # st.error(f"Time parsing error for '{created_at}': {e}")
        return "Unknown"


def process_urgent_feedback_queue(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process complaints data to create urgent feedback queue.
    
    Args:
        df: Raw complaints dataframe
        
    Returns:
        pd.DataFrame: Processed urgent feedback queue
    """
    if df.empty:
        return pd.DataFrame()
    
    try:
        # Create a copy to avoid modifying original
        processed_df = df.copy()
        
        # Add time_ago column
        processed_df['time_ago'] = processed_df['created_at'].apply(get_time_ago)
        
        # Sort by creation time (newest first)
        try:
            processed_df['created_at_dt'] = pd.to_datetime(processed_df['created_at'], errors='coerce')
            processed_df = processed_df.sort_values('created_at_dt', ascending=False)
        except Exception:
            # Fallback sorting if datetime conversion fails
            pass
        
        # Select and rename columns for display (removed priority column)
        display_columns = {
            'user_name': 'Customer',
            'summary': 'Issue', 
            'team_member_name': 'Assigned To',
            'department_name': 'Department',
            'time_ago': 'Time'
        }
        
        # Only keep columns that exist in the dataframe
        available_columns = {k: v for k, v in display_columns.items() if k in processed_df.columns}
        
        if available_columns:
            result_df = processed_df[list(available_columns.keys())].rename(columns=available_columns)
        else:
            # Fallback if expected columns don't exist
            result_df = processed_df.copy()
        
        return result_df.head(20)  # Return top 20 items
        
    except Exception as e:
        st.error(f"Error processing urgent feedback queue: {e}")
        return pd.DataFrame()


def get_team_performance_metrics(df: pd.DataFrame) -> Dict:
    """
    Calculate team performance metrics from complaints data.
    
    Args:
        df: Complaints dataframe
        
    Returns:
        dict: Performance metrics
    """
    if df.empty:
        return {}
    
    try:
        metrics = {}
        
        # Total complaints
        metrics['total_complaints'] = len(df)
        
        # Total unresolved complaints (status not 'closed' or 'resolved')
        if 'status' in df.columns:
            status_counts = df['status'].value_counts().to_dict()
            metrics['status_breakdown'] = status_counts
            # Count unresolved complaints
            unresolved_statuses = ['open', 'pending', 'in_progress', 'new', 'escalated']
            unresolved_count = sum([count for status, count in status_counts.items() 
                                  if status.lower() in unresolved_statuses or 
                                  (status.lower() not in ['closed', 'resolved', 'completed'])])
            metrics['unresolved_complaints'] = unresolved_count
        else:
            metrics['unresolved_complaints'] = 0
        
        # Complaints by team member
        if 'team_member_name' in df.columns:
            team_counts = df['team_member_name'].value_counts().to_dict()
            metrics['team_workload'] = team_counts
        
        # Complaints by department and team efficiency calculation
        if 'department_name' in df.columns:
            dept_counts = df['department_name'].value_counts().to_dict()
            metrics['department_breakdown'] = dept_counts
            
            # Calculate team efficiency: % of departments that have active complaints
            total_departments = len(dept_counts)  # Departments with complaints
            # We'll need to get total departments from team_performance_data later
            metrics['departments_with_complaints'] = total_departments
        
        # Recent complaints (last 24 hours)
        if 'created_at' in df.columns:
            try:
                df['created_at_dt'] = pd.to_datetime(df['created_at'], errors='coerce')
                recent_threshold = datetime.now() - timedelta(days=1)
                recent_complaints = df[df['created_at_dt'] >= recent_threshold]
                metrics['recent_24h'] = len(recent_complaints)
            except Exception:
                metrics['recent_24h'] = 0
        else:
            metrics['recent_24h'] = 0
        
        return metrics
        
    except Exception as e:
        st.error(f"Error calculating team metrics: {e}")
        return {}


def calculate_team_efficiency(team_performance_df: pd.DataFrame) -> int:
    """
    Calculate team efficiency as percentage of departments working on complaints.
    
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
        
        # Get departments that have team members with active complaints (total_tasks > 0)
        active_departments = len(team_performance_df[team_performance_df['total_tasks'] > 0]['department_name'].unique())
        
        if total_departments > 0:
            efficiency = round((active_departments / total_departments) * 100)
            return efficiency
        else:
            return 0
            
    except Exception as e:
        print(f"Error calculating team efficiency: {e}")
        return 0


def get_urgent_queue_summary(df: pd.DataFrame) -> Dict:
    """
    Get summary statistics for urgent queue.
    
    Args:
        df: Processed urgent queue dataframe
        
    Returns:
        dict: Summary statistics
    """
    if df.empty:
        return {'total_items': 0}
    
    try:
        summary = {
            'total_items': len(df)
        }
        
        return summary
        
    except Exception as e:
        st.error(f"Error getting urgent queue summary: {e}")
        return {'total_items': 0}


def get_team_performance_data() -> pd.DataFrame:
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
        
        # Check if connection is still valid before using it
        if conn.closed:
            try:
                st.error("Database connection was closed unexpectedly")
            except:
                print("Database connection was closed unexpectedly")
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
        # Ensure connection is properly closed only if it exists and is not already closed
        if conn and hasattr(conn, 'closed') and not conn.closed:
            try:
                conn.close()
            except Exception:
                pass


def toggle_oldest_task_status(team_member_id: int) -> bool:
    """
    Toggle the status of the oldest open task for a team member.
    Only works if the team member has tasks assigned (>0).
    
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
        
        # Check if connection is still valid
        if conn.closed:
            try:
                st.error("Database connection was closed unexpectedly")
            except:
                print("Database connection was closed unexpectedly")
            return False
        
        with conn.cursor() as cur:
            # First, check if this team member has any assigned tasks
            cur.execute("""
                SELECT COUNT(*) FROM support_tasks 
                WHERE assigned_to_member_id = %s
            """, (team_member_id,))
            
            total_tasks = cur.fetchone()[0]
            
            if total_tasks == 0:
                try:
                    st.warning("This team member has no assigned tasks yet.")
                except:
                    print("This team member has no assigned tasks yet.")
                return False
            
            # Try to find the oldest open task
            cur.execute("""
                SELECT id, status FROM support_tasks 
                WHERE assigned_to_member_id = %s AND status = 'open'
                ORDER BY created_at ASC 
                LIMIT 1
            """, (team_member_id,))
            
            task = cur.fetchone()
            
            if task:
                # Toggle open to closed
                cur.execute("""
                    UPDATE support_tasks 
                    SET status = 'closed'
                    WHERE id = %s
                """, (task[0],))
                conn.commit()  # Explicitly commit the transaction
                try:
                    st.success("Task marked as completed!")
                except:
                    print("Task marked as completed!")
                return True
            else:
                # Try to find the oldest closed task to reopen
                cur.execute("""
                    SELECT id, status FROM support_tasks 
                    WHERE assigned_to_member_id = %s AND status = 'closed'
                    ORDER BY created_at ASC 
                    LIMIT 1
                """, (team_member_id,))
                
                task = cur.fetchone()
                if task:
                    # Toggle closed to open
                    cur.execute("""
                        UPDATE support_tasks 
                        SET status = 'open'
                        WHERE id = %s
                    """, (task[0],))
                    conn.commit()  # Explicitly commit the transaction
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
        # Ensure connection is properly closed only if it exists and is not already closed
        if conn and hasattr(conn, 'closed') and not conn.closed:
            try:
                conn.close()
            except Exception:
                pass