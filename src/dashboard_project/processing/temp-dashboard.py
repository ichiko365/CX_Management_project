import streamlit as st
import pandas as pd
import sys
import os
from datetime import datetime

# Add the parent directory to sys.path to import from database and processing modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from team_task import (
    get_complaints_data,
    refresh_complaints_data,
    process_urgent_feedback_queue,
    get_team_performance_metrics,
    get_urgent_queue_summary,
    get_team_performance_data,
    calculate_team_efficiency
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
    
    .tooltip-container {
        position: relative;
        display: inline-block;
        cursor: pointer;
    }
    
    .tooltip-text {
        visibility: hidden;
        width: 300px;
        background-color: #1f2937;
        color: white;
        text-align: left;
        border-radius: 6px;
        padding: 8px;
        position: absolute;
        z-index: 1000;
        bottom: 125%;
        left: 50%;
        margin-left: -150px;
        opacity: 0;
        transition: opacity 0.3s;
        font-size: 0.8rem;
        line-height: 1.4;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .tooltip-text::after {
        content: "";
        position: absolute;
        top: 100%;
        left: 50%;
        margin-left: -5px;
        border-width: 5px;
        border-style: solid;
        border-color: #1f2937 transparent transparent transparent;
    }
    
    .tooltip-container:hover .tooltip-text {
        visibility: visible;
        opacity: 1;
    }
    
    .issue-preview {
        color: #374151;
        font-size: 0.9rem;
        font-style: italic;
    }
    
    .issue-preview:hover {
        color: #1f2937;
        text-decoration: underline;
    }
    </style>
    """, unsafe_allow_html=True)


def render_urgent_queue_card(df: pd.DataFrame):
    """Render the urgent feedback queue card similar to team performance card"""
    # Issues Queue Header
    st.markdown("""
    <div style="background-color: #f8fafc; padding: 1rem; border-radius: 0.5rem; border: 1px solid #e2e8f0; margin-bottom: 1rem;">
        <h3 style="color: #dc2626; margin: 0; font-size: 1.25rem; font-weight: bold;">
            🔔 Issues Queue
        </h3>
        <p style="color: #6b7280; font-size: 0.9rem; margin: 0.25rem 0 0 0;">
            Customer complaints and support requests
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    if df.empty:
        st.info("No items in the queue.")
        return
    
    # Table headers
    col1, col2, col3, col4 = st.columns([2, 4, 2, 1])
    with col1:
        st.markdown("**Customer**")
    with col2:
        st.markdown("**Issue**")
    with col3:
        st.markdown("**Assigned To**")
    with col4:
        st.markdown("**Time**")
    
    st.markdown("---")
    
    # Display each issue
    for idx, row in df.iterrows():
        customer_name = row.get('Customer', 'Unknown')
        time_info = row.get('Time', '')
        full_issue = row.get('Issue', 'No summary available')  # Keep original full text
        assigned_to = row.get('Assigned To', 'Unassigned')
        
        col1, col2, col3, col4 = st.columns([2, 4, 2, 1])
        
        with col1:
            st.markdown(f"**{customer_name}**")
        
        with col2:
            # Display truncated issue with tooltip for full text
            if len(full_issue) > 80:
                truncated_issue = full_issue[:80] + "..."
                st.markdown(f"""
                <div class="tooltip-container">
                    <span class="issue-preview">{truncated_issue}</span>
                    <span class="tooltip-text">{full_issue}</span>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"*{full_issue}*")
        
        with col3:
            if assigned_to and assigned_to != 'Unassigned':
                st.markdown(f"**{assigned_to}**")
                # Get department name from the row
                dept_name = row.get('Department', 'No Department')
                if dept_name and dept_name != 'No Department':
                    st.caption(dept_name)
            else:
                st.markdown("*Unassigned*")
        
        with col4:
            if time_info:
                st.caption(time_info)
        
        # Add some spacing between rows
        st.markdown("<br>", unsafe_allow_html=True)


def render_metrics_cards(metrics: dict, urgent_summary: dict, team_performance_df: pd.DataFrame):
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
            <div class="metric-label">Total Unresolved</div>
        </div>
        """.format(metrics.get('unresolved_complaints', 0)), unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metrics-card">
            <div class="metric-value" style="color: #059669;">{}</div>
            <div class="metric-label">Last 24h</div>
        </div>
        """.format(metrics.get('recent_24h', 0)), unsafe_allow_html=True)
    
    with col4:
        # Calculate team efficiency
        efficiency = calculate_team_efficiency(team_performance_df)
        st.markdown("""
        <div class="metrics-card">
            <div class="metric-value" style="color: #7c3aed;">{}</div>
            <div class="metric-label">Team Efficiency</div>
        </div>
        """.format(f"{efficiency}%"), unsafe_allow_html=True)


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


def render_team_performance_card(team_df: pd.DataFrame):
    """Render the team performance card similar to team2.png"""
    if team_df.empty:
        st.info("No team performance data available.")
        return
    
    # Team Performance Header
    st.markdown("""
    <div style="background-color: #f8fafc; padding: 1rem; border-radius: 0.5rem; border: 1px solid #e2e8f0; margin-bottom: 1rem;">
        <h3 style="color: #7c3aed; margin: 0; font-size: 1.25rem; font-weight: bold;">
            👥 Team Performance
        </h3>
        <p style="color: #6b7280; font-size: 0.9rem; margin: 0.25rem 0 0 0;">
            Task allocation and performance metrics
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Table headers
    col1, col2, col3 = st.columns([3, 2, 2])
    with col1:
        st.markdown("**Team Member**")
    with col2:
        st.markdown("**Tasks**")  
    with col3:
        st.markdown("**Completion**")
    
    st.markdown("---")
    
    # Display each team member
    for idx, row in team_df.iterrows():
        col1, col2, col3 = st.columns([3, 2, 2])
        
        with col1:
            # Team member name and department
            department = row.get('department_name', 'No Department')
            st.markdown(f"**{row['name']}**")
            st.caption(department)
        
        with col2:
            # Task counts
            total_tasks = int(row['total_tasks'])
            open_tasks = int(row['open_tasks'])
            completed_tasks = int(row['completed_tasks'])
            
            st.markdown(f"{total_tasks} assigned")
            st.caption(f"{completed_tasks} completed")
        
        with col3:
            # Completion percentage with progress bar
            completion_pct = int(row['completion_percentage'])
            
            # Create progress bar using HTML/CSS
            progress_color = "#10b981" if completion_pct >= 80 else "#f59e0b" if completion_pct >= 60 else "#ef4444"
            
            st.markdown(f"""
            <div style="background-color: #f3f4f6; border-radius: 0.5rem; height: 0.5rem; margin: 0.25rem 0;">
                <div style="background-color: {progress_color}; width: {completion_pct}%; height: 100%; border-radius: 0.5rem;"></div>
            </div>
            <small style="color: #6b7280;">{completion_pct}%</small>
            """, unsafe_allow_html=True)
        
        # Add some spacing between rows
        st.markdown("<br>", unsafe_allow_html=True)


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
            st.plotly_chart(fig, use_container_width=True, key="dept_breakdown_chart")
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
        <h1 class="urgent-title">🎧 Customer Support Queue</h1>
        <p class="urgent-subtitle">Customer complaints and support requests</p>
    </div>
    """, unsafe_allow_html=True)
    
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
    
    # Get team performance data
    team_performance_df = get_team_performance_data()
    
    # Get metrics
    metrics = get_team_performance_metrics(complaints_df)
    urgent_summary = get_urgent_queue_summary(urgent_queue_df)
    
    # Display metrics cards
    render_metrics_cards(metrics, urgent_summary, team_performance_df)
    
    st.markdown("---")
    
    # Main content area with Issues Queue and Team Performance side by side
    col_main1, col_main2 = st.columns([1, 1])  # Equal height columns
    
    with col_main1:
        # Issues Queue Card (styled like Team Performance)
        if not urgent_queue_df.empty:
            render_urgent_queue_card(urgent_queue_df)
        else:
            st.markdown("""
            <div style="background-color: #f8fafc; padding: 1rem; border-radius: 0.5rem; border: 1px solid #e2e8f0; margin-bottom: 1rem;">
                <h3 style="color: #dc2626; margin: 0; font-size: 1.25rem; font-weight: bold;">
                    🔔 Issues Queue
                </h3>
                <p style="color: #6b7280; font-size: 0.9rem; margin: 0.25rem 0 0 0;">
                    Customer complaints and support requests
                </p>
            </div>
            """, unsafe_allow_html=True)
            st.info("No items in the queue.")
    
    with col_main2:
        # Team Performance Card (on the right)
        render_team_performance_card(team_performance_df)
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