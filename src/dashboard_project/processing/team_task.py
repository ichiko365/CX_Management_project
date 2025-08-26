import pandas as pd
import streamlit as st
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import sys
import os

# Add the parent directory to sys.path to import from database module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.queries import fetch_table
from database.complaints import sync_to_main_database


def refresh_complaints_data() -> bool:
    """
    Refresh the complaints table by syncing data from customer database.
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        rows_synced = sync_to_main_database("complaints")
        st.success(f"Successfully synced {rows_synced} rows to complaints table")
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
        created_at: Timestamp string
        
    Returns:
        str: Human readable time difference
    """
    try:
        if isinstance(created_at, str):
            # Try different datetime formats
            for fmt in ['%Y-%m-%d %H:%M:%S', '%Y-%m-%d %H:%M:%S.%f', '%Y-%m-%dT%H:%M:%S']:
                try:
                    created_time = datetime.strptime(created_at, fmt)
                    break
                except ValueError:
                    continue
            else:
                return "Unknown"
        else:
            created_time = created_at
        
        # Calculate time difference
        diff = datetime.now() - created_time
        
        if diff.days > 0:
            return f"{diff.days}d ago"
        elif diff.seconds > 3600:
            hours = diff.seconds // 3600
            return f"{hours}h ago"
        elif diff.seconds > 60:
            minutes = diff.seconds // 60
            return f"{minutes}m ago"
        else:
            return "Just now"
            
    except Exception:
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
        
        # Add priority column
        processed_df['priority'] = processed_df.apply(
            lambda row: categorize_priority(
                row.get('status', ''), 
                row.get('created_at', '')
            ), axis=1
        )
        
        # Add time_ago column
        processed_df['time_ago'] = processed_df['created_at'].apply(get_time_ago)
        
        # Filter for urgent items (Critical and High priority)
        urgent_df = processed_df[processed_df['priority'].isin(['Critical', 'High'])].copy()
        
        # Sort by priority (Critical first) and then by creation time (newest first)
        priority_order = {'Critical': 0, 'High': 1, 'Medium': 2, 'Low': 3}
        urgent_df['priority_rank'] = urgent_df['priority'].map(priority_order)
        
        # Convert created_at to datetime for proper sorting
        try:
            urgent_df['created_at_dt'] = pd.to_datetime(urgent_df['created_at'], errors='coerce')
            urgent_df = urgent_df.sort_values(['priority_rank', 'created_at_dt'], ascending=[True, False])
        except Exception:
            urgent_df = urgent_df.sort_values('priority_rank', ascending=True)
        
        # Select and rename columns for display
        display_columns = {
            'user_name': 'Customer',
            'summary': 'Issue', 
            'priority': 'Priority',
            'team_member_name': 'Assigned To',
            'time_ago': 'Time'
        }
        
        # Only keep columns that exist in the dataframe
        available_columns = {k: v for k, v in display_columns.items() if k in urgent_df.columns}
        
        if available_columns:
            result_df = urgent_df[list(available_columns.keys())].rename(columns=available_columns)
        else:
            # Fallback if expected columns don't exist
            result_df = urgent_df.copy()
        
        return result_df.head(20)  # Return top 20 urgent items
        
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
        
        # Complaints by status
        if 'status' in df.columns:
            status_counts = df['status'].value_counts().to_dict()
            metrics['status_breakdown'] = status_counts
        
        # Complaints by team member
        if 'team_member_name' in df.columns:
            team_counts = df['team_member_name'].value_counts().to_dict()
            metrics['team_workload'] = team_counts
        
        # Complaints by department
        if 'department_name' in df.columns:
            dept_counts = df['department_name'].value_counts().to_dict()
            metrics['department_breakdown'] = dept_counts
        
        # Recent complaints (last 24 hours)
        if 'created_at' in df.columns:
            try:
                df['created_at_dt'] = pd.to_datetime(df['created_at'], errors='coerce')
                recent_threshold = datetime.now() - timedelta(days=1)
                recent_complaints = df[df['created_at_dt'] >= recent_threshold]
                metrics['recent_complaints'] = len(recent_complaints)
            except Exception:
                metrics['recent_complaints'] = 0
        
        return metrics
        
    except Exception as e:
        st.error(f"Error calculating team metrics: {e}")
        return {}


def get_urgent_queue_summary(df: pd.DataFrame) -> Dict:
    """
    Get summary statistics for urgent queue.
    
    Args:
        df: Processed urgent queue dataframe
        
    Returns:
        dict: Summary statistics
    """
    if df.empty:
        return {'critical_count': 0, 'high_count': 0, 'total_urgent': 0}
    
    try:
        summary = {}
        
        if 'Priority' in df.columns:
            priority_counts = df['Priority'].value_counts()
            summary['critical_count'] = priority_counts.get('Critical', 0)
            summary['high_count'] = priority_counts.get('High', 0)
            summary['total_urgent'] = len(df)
        else:
            summary = {'critical_count': 0, 'high_count': 0, 'total_urgent': len(df)}
        
        return summary
        
    except Exception as e:
        st.error(f"Error getting urgent queue summary: {e}")
        return {'critical_count': 0, 'high_count': 0, 'total_urgent': 0}