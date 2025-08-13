import streamlit as st

from database.queries import fetch_table, fetch_count


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

# Optional: quick status using COUNT(*), with try/else messaging
count = fetch_count("analysis_results")
if count is None:
    st.warning("Could not determine row count for analysis_results.")
elif count == 0:
    st.info("No data found in analysis_results.")
else:
    st.success(f"analysis_results has {count} rows.")

# Main table fetch and display
df = fetch_table("analysis_results")
st.header("Analysis Results")
st.write("Displaying data from the analysis_results table:")

if df is not None and not df.empty:
    st.dataframe(df)
else:
    # Provide a structured empty-state message
    st.info("No data to display.")
