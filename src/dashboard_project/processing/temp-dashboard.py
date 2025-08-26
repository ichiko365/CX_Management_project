import streamlit as st
import pandas as pd
import sys
import os
from datetime import datetime

# Add the parent directory to sys.path to import from database and processing modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from processing.team_task import (
    get_complaints_data,
    refresh_complaints_data,
    process_urgent_feedback_queue,
    get_team_performance_metrics,
    get_urgent_queue_summary
)


def apply_custom_css():
    """Apply custom CSS for better styling similar to team.png"""
    st.markdown("""
    <style>
    .urgent-header {
        background-color: #fff5f5;
        border-left: 4px solid #ef4444;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0.5rem;
    }
    
    .urgent-title {
        color: #dc2626;
        font-size: 1.5rem;
        font-weight: bold;
        margin: 0;
        display: flex;
        align-items: center;
    }
    
    .urgent-subtitle {
        color: #6b7280;
        font-size: 0.9rem;
        margin: 0.25rem 0 0 0;
    }
    
    .priority-critical {
        background-color: #dc2626;
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 1rem;
        font-size: 0.8rem;
        font-weight: bold;
    }
    
    .priority-high {
        background-color: #f59e0b;
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 1rem;
        font-size: 0.8rem;
        font-weight: bold;
    }
    
    .priority-medium {
        background-color: #10b981;
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 1rem;
        font-size: 0.8rem;
        font-weight: bold;
    }
    
    .customer-info {
        font-weight: 600;
        color: #1f2937;
    }
    
    .time-info {
        color: #6b7280;
        font-size: 0.8rem;
    }
    
    .issue-text {
        color: #374151;
        font-size: 0.9rem;
    }
    
    .assigned-to {
        color: #1f2937;
        font-weight: 500;
    }
    
    .metrics-card {
        background-color: #f8fafc;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #e2e8f0;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #1f2937;
    }
    
    .metric-label {
        color: #6b7280;
        font-size: 0.9rem;
    }
    </style>
    """, unsafe_allow_html=True)


def render_priority_badge(priority: str) -> str:
    """Render priority badge with appropriate styling"""
    priority_class = f"priority-{priority.lower()}"
    return f'<span class="{priority_class}">{priority}</span>'


def render_urgent_queue_table(df: pd.DataFrame):
    """Render the urgent feedback queue table with custom styling"""
    if df.empty:
        st.info("No urgent items in the queue.")
        return
    
    # Create custom table HTML
    table_html = """
    <table style="width: 100%; border-collapse: collapse; margin-top: 1rem;">
        <thead>
            <tr style="border-bottom: 2px solid #e5e7eb;">
                <th style="text-align: left; padding: 0.75rem; font-weight: 600; color: #374151;">Customer</th>
                <th style="text-align: left; padding: 0.75rem; font-weight: 600; color: #374151;">Issue</th>
                <th style="text-align: left; padding: 0.75rem; font-weight: 600; color: #374151;">Priority</th>
                <th style="text-align: left; padding: 0.75rem; font-weight: 600; color: #374151;">Assigned To</th>
            </tr>
        </thead>
        <tbody>
    """
    
    for idx, row in df.iterrows():
        customer_name = row.get('Customer', 'Unknown')
        time_info = row.get('Time', '')
        issue = row.get('Issue', 'No summary available')
        priority = row.get('Priority', 'Medium')
        assigned_to = row.get('Assigned To', 'Unassigned')
        
        # Truncate long issue text
        if len(issue) > 80:
            issue = issue[:80] + "..."
        
        priority_badge = render_priority_badge(priority)
        
        table_html += f"""
        <tr style="border-bottom: 1px solid #f3f4f6;">
            <td style="padding: 1rem 0.75rem;">
                <div class="customer-info">{customer_name}</div>
                <div class="time-info">{time_info}</div>
            </td>
            <td style="padding: 1rem 0.75rem;">
                <div class="issue-text">{issue}</div>
            </td>
            <td style="padding: 1rem 0.75rem;">
                {priority_badge}
            </td>
            <td style="padding: 1rem 0.75rem;">
                <div class="assigned-to">{assigned_to}</div>
            </td>
        </tr>
        """
    
    table_html += """
        </tbody>
    </table>
    """
    
    st.markdown(table_html, unsafe_allow_html=True)


def render_metrics_cards(metrics: dict, urgent_summary: dict):
    """Render metrics cards"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metrics-card">
            <div class="metric-value">{}</div>
            <div class="metric-label">Total Complaints</div>
        </div>
        """.format(metrics.get('total_complaints', 0)), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metrics-card">
            <div class="metric-value" style="color: #dc2626;">{}</div>
            <div class="metric-label">Critical Issues</div>
        </div>
        """.format(urgent_summary.get('critical_count', 0)), unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metrics-card">
            <div class="metric-value" style="color: #f59e0b;">{}</div>
            <div class="metric-label">High Priority</div>
        </div>
        """.format(urgent_summary.get('high_count', 0)), unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metrics-card">
            <div class="metric-value">{}</div>
            <div class="metric-label">Recent (24h)</div>
        </div>
        """.format(metrics.get('recent_complaints', 0)), unsafe_allow_html=True)


def render_team_workload(metrics: dict):
    """Render team workload breakdown"""
    team_workload = metrics.get('team_workload', {})
    if team_workload:
        st.subheader("Team Workload")
        
        # Create a dataframe for better display
        workload_df = pd.DataFrame(list(team_workload.items()), columns=['Team Member', 'Open Cases'])
        workload_df = workload_df.sort_values('Open Cases', ascending=False)
        
        # Display as horizontal bar chart
        st.bar_chart(workload_df.set_index('Team Member'))


def render_department_breakdown(metrics: dict):
    """Render department breakdown"""
    dept_breakdown = metrics.get('department_breakdown', {})
    if dept_breakdown:
        st.subheader("Department Breakdown")
        
        # Create a pie chart
        dept_df = pd.DataFrame(list(dept_breakdown.items()), columns=['Department', 'Cases'])
        
        # Display as donut chart using plotly if available, otherwise use st.pyplot
        try:
            import plotly.express as px
            fig = px.pie(dept_df, values='Cases', names='Department', title="Cases by Department")
            st.plotly_chart(fig, use_container_width=True)
        except ImportError:
            # Fallback to simple metrics
            col1, col2 = st.columns(2)
            for i, (dept, count) in enumerate(dept_breakdown.items()):
                if i % 2 == 0:
                    col1.metric(dept, count)
                else:
                    col2.metric(dept, count)


def main():
    """Main dashboard function"""
    st.set_page_config(
        page_title="CX Management Dashboard", 
        page_icon="🚨", 
        layout="wide"
    )
    
    apply_custom_css()
    
    # Header
    st.markdown("""
    <div class="urgent-header">
        <h1 class="urgent-title">🚨 Urgent Feedback Queue</h1>
        <p class="urgent-subtitle">Critical issues requiring immediate attention</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Control buttons
    col1, col2, col3 = st.columns([1, 1, 4])
    
    with col1:
        if st.button("🔄 Refresh Data", help="Sync data from customer database"):
            with st.spinner("Refreshing data..."):
                if refresh_complaints_data():
                    st.rerun()
    
    with col2:
        auto_refresh = st.checkbox("Auto Refresh (30s)", help="Automatically refresh data every 30 seconds")
    
    # Auto refresh
    if auto_refresh:
        st.rerun()
    
    # Load and process data
    with st.spinner("Loading complaints data..."):
        complaints_df = get_complaints_data()
    
    if complaints_df.empty:
        st.warning("No complaints data available. Please refresh the data or check database connection.")
        if st.button("Try Refreshing Data"):
            refresh_complaints_data()
            st.rerun()
        return
    
    # Process urgent queue
    urgent_queue_df = process_urgent_feedback_queue(complaints_df)
    
    # Get metrics
    metrics = get_team_performance_metrics(complaints_df)
    urgent_summary = get_urgent_queue_summary(urgent_queue_df)
    
    # Display metrics cards
    render_metrics_cards(metrics, urgent_summary)
    
    st.markdown("---")
    
    # Main urgent queue table
    st.markdown("### Urgent Issues Queue")
    render_urgent_queue_table(urgent_queue_df)
    
    # Additional insights
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        render_team_workload(metrics)
    
    with col2:
        render_department_breakdown(metrics)
    
    # Status breakdown
    status_breakdown = metrics.get('status_breakdown', {})
    if status_breakdown:
        st.markdown("---")
        st.subheader("Status Overview")
        status_cols = st.columns(len(status_breakdown))
        for i, (status, count) in enumerate(status_breakdown.items()):
            with status_cols[i]:
                st.metric(f"{status.title()} Cases", count)
    
    # Footer with last updated time
    st.markdown("---")
    st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()