import os
import streamlit as st
import pandas as pd
from datetime import datetime

# Import the queries we added earlier
try:
    from database.queries import (
        fetch_complaints_with_departments,
        fetch_table,
        sync_complaints_from_customer_db,
    )
except Exception:
    # If imports fail because of path, try adjusting sys.path
    import sys
    sys.path.append(os.path.dirname(__file__))
    from database.queries import (
        fetch_complaints_with_departments,
        fetch_table,
        sync_complaints_from_customer_db,
    )

ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")
URGENT_IMG = os.path.join(ASSETS_DIR, "urgent_queue.png")
TEAM_IMG = os.path.join(ASSETS_DIR, "team_performance.png")

st.set_page_config(page_title="Support Dashboard", layout="wide")

st.title("Support Dashboard")
st.markdown("A lightweight dashboard showing urgent feedback and team performance. Columns 'Priority' and 'Rating' are omitted.")

# --- Top row: show images if available ---
cols = st.columns([1, 1])
def _is_valid_image(path: str) -> bool:
    """Quick check for PNG/JPEG headers to avoid trying to load invalid files."""
    try:
        if not os.path.exists(path):
            return False
        with open(path, "rb") as f:
            header = f.read(8)
            if not header:
                return False
            # PNG signature or JPEG start bytes
            if header.startswith(b"\x89PNG") or header.startswith(b"\xff\xd8\xff"):
                return True
        return False
    except Exception:
        return False


with cols[0]:
    if _is_valid_image(URGENT_IMG):
        try:
            st.image(URGENT_IMG, use_container_width=True)
        except Exception:
            st.info("Urgent queue image exists but could not be displayed.")
    else:
        st.info("Urgent queue image not found or invalid. Place a PNG/JPEG at: src/dashboard_project/assets/urgent_queue.png")

with cols[1]:
    if _is_valid_image(TEAM_IMG):
        try:
            st.image(TEAM_IMG, use_container_width=True)
        except Exception:
            st.info("Team performance image exists but could not be displayed.")
    else:
        st.info("Team performance image not found or invalid. Place a PNG/JPEG at: src/dashboard_project/assets/team_performance.png")

st.markdown("---")

# Control buttons
col1, col2 = st.columns([1, 1])

with col1:
    # Combined Sync and Refresh button
    if st.button("Sync and Refresh"):
        with st.spinner("Syncing and refreshing..."):
            n = sync_complaints_from_customer_db("complaints")
            # Clear Streamlit caches to ensure fresh connections/data
            try:
                st.cache_data.clear()
            except Exception:
                pass
            if hasattr(st, 'cache_resource'):
                try:
                    st.cache_resource.clear()
                except Exception:
                    pass
            if n > 0:
                st.success(f"Synced {n} rows and refreshed data")
            else:
                st.info("Sync completed; cache cleared")
            # Rerun to refresh UI with new data
            try:
                st.rerun()
            except Exception:
                try:
                    st.experimental_rerun()
                except Exception:
                    pass

with col2:
    # Placeholder column to keep layout balanced
    st.write("")

# Fetch complaints table (main DB)
try:
    complaints_df = fetch_table("complaints", database_name="database")
except Exception as e:
    st.error(f"Error fetching complaints table: {e}")
    complaints_df = pd.DataFrame()

# If fetch_table returned empty, try the combined fetch to get department mapping
if complaints_df.empty:
    try:
        merged = fetch_complaints_with_departments()
        # If result has complaint columns, use them; else create a friendly table
        if not merged.empty:
            # Try to pick complaint columns
            complaints_df = merged.copy()
    except Exception:
        complaints_df = pd.DataFrame()

# Fallback sample data if still empty
if complaints_df.empty:
    complaints_df = pd.DataFrame([
        {
            "id": "1",
            "user_name": "Sarah Johnson",
            "summary": "Product defect causing safety concerns",
            "team_member_name": "Mike Chen",
            "department_name": "Quality & Safety",
            "status": "open",
            "created_at": datetime.now().isoformat(),
        },
        {
            "id": "2",
            "user_name": "David Wilson",
            "summary": "Billing discrepancy - overcharged",
            "team_member_name": "Lisa Park",
            "department_name": "Billing",
            "status": "open",
            "created_at": (datetime.now()).isoformat(),
        },
        {
            "id": "3",
            "user_name": "Emma Davis",
            "summary": "Delivery never arrived",
            "team_member_name": "Tom Rodriguez",
            "department_name": "Logistics",
            "status": "resolved",
            "created_at": (datetime.now()).isoformat(),
        },
    ])

# Drop 'Priority' and 'Rating' if present
for col in ["Priority", "priority", "Rating", "rating"]:
    if col in complaints_df.columns:
        complaints_df = complaints_df.drop(columns=[col])

# Convert created_at to datetime if present
if "created_at" in complaints_df.columns:
    try:
        complaints_df["created_at"] = pd.to_datetime(complaints_df["created_at"])
    except Exception:
        pass

# Urgent Feedback Queue card
st.subheader("Urgent Feedback Queue")
# Show latest 10 by created_at if available
if "created_at" in complaints_df.columns:
    display_df = complaints_df.sort_values(by="created_at", ascending=False).head(10)
else:
    display_df = complaints_df.head(10)

# Select columns to display (prefer these if present)
preferred_cols = ["user_name", "summary", "team_member_name", "department_name", "status", "created_at"]
cols_present = [c for c in preferred_cols if c in display_df.columns]
if not cols_present:
    cols_present = list(display_df.columns)[:6]

st.table(display_df[cols_present].rename(columns={"user_name": "Customer", "summary": "Issue", "team_member_name": "Assigned To", "department_name": "Department", "created_at": "Created"}))

st.markdown("---")

# Team Performance card
st.subheader("Team Performance")

# Compute team stats
if "team_member_name" in complaints_df.columns:
    df = complaints_df.copy()
    df["is_completed"] = df["status"].astype(str).str.lower().isin(["resolved", "closed", "done", "completed"]) if "status" in df.columns else False
    # Use 'id' if present for counts, else use the index
    id_col = "id" if "id" in df.columns else None
    if id_col:
        group = df.groupby("team_member_name").agg(
            assigned=(id_col, "count"),
            completed=("is_completed", "sum"),
        ).reset_index()
    else:
        # fallback: count rows per team member
        assigned_series = df.groupby("team_member_name").size()
        completed_series = df.groupby("team_member_name")["is_completed"].sum()
        group = (
            pd.concat([assigned_series, completed_series], axis=1)
            .rename(columns={0: "assigned", "is_completed": "completed"})
            .reset_index()
        )
    group["completion_pct"] = (group["completed"] / group["assigned"]).fillna(0)

    # Display as table with progress bars
    for _, row in group.sort_values("completion_pct", ascending=False).iterrows():
        cols = st.columns([3, 2, 2])
        cols[0].markdown(f"**{row['team_member_name']}**")
        cols[1].markdown(f"{int(row['assigned'])} assigned\n{int(row['completed'])} completed")
        cols[2].progress(float(row["completion_pct"]))
else:
    st.info("No team member data available in complaints table.")

st.markdown("---")

st.caption("Dashboard generated by src/dashboard_project/dashboard.py")
