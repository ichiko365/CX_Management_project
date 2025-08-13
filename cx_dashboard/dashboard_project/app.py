import streamlit as st

from database.queries import fetch_table, fetch_count
from processing.calculation import get_filter_options, apply_filters, compute_kpis


st.title("Sales Data Dashboard")

# Manual refresh to clear cached data and fetch fresh rows
if st.button("Refresh data", type="primary", help="Clear cached results and reload from DB"):
    # Clear function-specific caches (Streamlit 1.25+). Fallback to global clear if needed.
    cleared_any = False
    try:
        fetch_table.clear()
        cleared_any = True
    except Exception:
        pass
    try:
        fetch_count.clear()
        cleared_any = True or cleared_any
    except Exception:
        pass
    if not cleared_any:
        try:
            st.cache_data.clear()
        except Exception:
            pass
    st.toast("Refreshing data…", icon="🔄")
    st.rerun()

# Load data once (cache-aware) and build filters
df = fetch_table("analysis_results")
filters = get_filter_options(df)

# Sidebar filters
st.sidebar.header("Filters")
asin_choices = filters.get("asin", [])
selected_asins = st.sidebar.multiselect("ASIN", options=asin_choices, default=[])

# Apply filters
filtered_df = apply_filters(df, asin_values=selected_asins)

# Tabs for Dashboard and Table
tab_dashboard, tab_table = st.tabs(["Dashboard", "Table"])

with tab_dashboard:
    # KPI from calculation utilities (after filters)
    kpis = compute_kpis(filtered_df)
    st.subheader("KPIs")
    col1 = st.columns(1)[0]
    col1.metric(label="Number of Rows", value=f"{kpis.get('rows', 0):,}")

    # Optional: quick status using COUNT(*) of the whole table
    count = fetch_count("analysis_results")
    if count is None:
        st.caption("Row count unavailable (DB error).")
    else:
        st.caption(f"Total rows in table: {count:,}")

with tab_table:
    st.subheader("Analysis Results")
    st.write("Displaying data from the analysis_results table (filtered):")
    if filtered_df is not None and not filtered_df.empty:
        st.dataframe(filtered_df, hide_index=True)
    else:
        st.info("No data to display for the selected filters.")
