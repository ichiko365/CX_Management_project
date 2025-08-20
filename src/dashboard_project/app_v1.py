import streamlit as st
import pandas as pd

from database.queries import fetch_table, fetch_count
from processing.calculation import get_filter_options, apply_filters, KpiEngine


# THEME: Set colors for background and title gradient — change these to adjust look
bg_start = "#fff0f6"   # light pink start (background)
bg_end = "#f7f4ff"     # very light purple end (background)
title_start = "#d800b9"  # vibrant pink (title gradient start)
title_end = "#6a00ff"    # purple (title gradient end)

# THEME: Page background gradient injected via CSS
st.markdown(
        f"""
        <style>
        /* THEME: Page background gradient — edit bg_start/bg_end above */
        [data-testid="stAppViewContainer"] {{
            background: linear-gradient(180deg, {bg_start} 0%, {bg_end} 100%) !important;
        }}
        /* Make header transparent to let gradient show through */
        [data-testid="stHeader"] {{ background: transparent; }}
        </style>
        """,
        unsafe_allow_html=True,
)

# THEME: Gradient Dashboard title — edit title_start/title_end above
st.markdown(
        f"""
        <h1 style="margin:0 0 0.25rem 0; background:linear-gradient(90deg, {title_start}, {title_end}); -webkit-background-clip:text; background-clip:text; color:transparent;">
            Beauty CX Dashboard
        </h1>
        """,
        unsafe_allow_html=True,
)

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

"""Sidebar filters (except Category which is shown in Dashboard header)."""
st.sidebar.header("Filters")
asin_choices = filters.get("asin", [])
selected_asins = st.sidebar.multiselect("ASIN", options=asin_choices, default=[])

# Additional categorical filters
region_choices = filters.get("region", [])
selected_regions = st.sidebar.multiselect("Region", options=region_choices, default=[])

sentiment_choices = filters.get("sentiment", [])
selected_sentiments = st.sidebar.multiselect("Sentiment", options=sentiment_choices, default=[])

primary_category_choices = filters.get("primary_category", [])

# Numeric range filter for urgency_score
urgency_cfg = filters.get("urgency_score", {}) or {}
urgency_score_range = None
if isinstance(urgency_cfg, dict) and urgency_cfg.get("min") is not None and urgency_cfg.get("max") is not None:
    try:
        umin = float(urgency_cfg["min"])
        umax = float(urgency_cfg["max"])
        if umin < umax:
            urgency_score_range = st.sidebar.slider(
                "Urgency score",
                min_value=umin,
                max_value=umax,
                value=(umin, umax),
            )
        else:
            st.sidebar.caption("Urgency score range unavailable.")
    except Exception:
        st.sidebar.caption("Urgency score range unavailable.")

# Date range filter for review_date
review_date_cfg = filters.get("review_date", {}) or {}
review_date_range = None
if isinstance(review_date_cfg, dict) and review_date_cfg.get("min") is not None and review_date_cfg.get("max") is not None:
    start_ts = pd.to_datetime(review_date_cfg["min"], errors="coerce")
    end_ts = pd.to_datetime(review_date_cfg["max"], errors="coerce")
    if pd.notna(start_ts) and pd.notna(end_ts) and start_ts <= end_ts:
        start_date = start_ts.date()
        end_date = end_ts.date()
        review_date_range = st.sidebar.date_input(
            "Review date",
            value=(start_date, end_date),
            min_value=start_date,
            max_value=end_date,
        )
        # Normalize single-date selection to a (start, end) tuple
        if review_date_range is not None and not isinstance(review_date_range, (list, tuple)):
            review_date_range = (review_date_range, review_date_range)
        elif isinstance(review_date_range, (list, tuple)) and len(review_date_range) == 1:
            review_date_range = (review_date_range[0], review_date_range[0])

# Trend smoothing (moved to sidebar)
smoothing_window = st.sidebar.slider(
    "Trend smoothing window",
    min_value=0,
    max_value=12,
    value=3,
    help="Rolling window size; 0 disables smoothing",
)

"""Tabs for Dashboard and Table"""
tab_dashboard, tab_table = st.tabs(["Dashboard", "Table"])

with tab_dashboard:
    # Header controls: period selection and Category filter (moved from sidebar)
    left, mid, right = st.columns([1, 1, 2])
    with left:
        period_label = st.selectbox(
            "Period",
            options=["Last 7 days", "Last 30 days", "Last 90 days", "Use Review Date"],
            index=1,
        )
    with mid:
        selected_primary_categories = st.multiselect(
            "All Categories",
            options=primary_category_choices,
            default=st.session_state.get("primary_category_top", []),
            key="primary_category_top",
            help="Filter by primary category",
        )

    # Apply filters (including Category from header)
    use_review_date = period_label == "Use Review Date"
    effective_review_date_range = review_date_range if use_review_date else None
    filtered_df = apply_filters(
        df,
        asin_values=selected_asins,
        region_values=selected_regions,
        sentiment_values=selected_sentiments,
        primary_category_values=selected_primary_categories,
        urgency_score_range=urgency_score_range,
        review_date_range=effective_review_date_range,
    )

    # KPI cards
    days_map = {"Last 7 days": 7, "Last 30 days": 30, "Last 90 days": 90}
    period_days = days_map.get(period_label, 30)
    kpi_engine = KpiEngine(filtered_df)
    if use_review_date:
        # Compute KPIs over the filtered_df without time-window deltas
        # Sentiment score (overall)
        mapping = getattr(KpiEngine, "DEFAULT_MAPPING", {
            "Positive": 10.0, "Neutral": 5.0, "Mixed": 3.0, "Negative": 1.0,
        })
        if filtered_df is not None and not filtered_df.empty and "sentiment" in filtered_df.columns:
            sent_series = filtered_df["sentiment"].map(lambda s: mapping.get(str(s), None)).dropna()
            sent_score_overall = float(sent_series.mean()) if not sent_series.empty else None
        else:
            sent_score_overall = None

        # Review volume (overall)
        vol_overall = int(len(filtered_df)) if filtered_df is not None else 0

        # Urgent issues (overall) – reuse KpiEngine parsing logic
        if filtered_df is not None and not filtered_df.empty:
            urgency = pd.to_numeric(filtered_df.get("urgency_score", pd.Series(index=filtered_df.index)), errors="coerce")
            tags = filtered_df.get("issue_tags", pd.Series(index=filtered_df.index)).apply(KpiEngine._parse_issue_tags)
            urgent_mask = urgency.fillna(0) >= 3
            has_tags = tags.apply(lambda lst: isinstance(lst, list) and len(lst) > 0)
            urgent_mask = urgent_mask | has_tags
            total_u = int(urgent_mask.sum())
            critical_u = int((urgency >= 5).fillna(False).sum())
            high_u = int(((urgency >= 4) & (urgency < 5)).fillna(False).sum())
        else:
            total_u = critical_u = high_u = 0

        k = {
            "sentiment_score": {"score": sent_score_overall, "delta": None},
            "review_volume": {"count": vol_overall, "delta_pct": None},
            "urgent_issues": {"total": total_u, "critical": critical_u, "high": high_u},
            "team_utilization": None,
        }
    else:
        k = kpi_engine.compute_all(days=period_days)

    # Light CSS for gradient cards inspired by your template
    st.markdown(
        """
        <style>
        .metric-grid {display:grid; gap:1rem; grid-template-columns:repeat(4, minmax(200px, 1fr));}
        .card {background:#ffffff; border:1px solid #eee; border-radius:14px; padding:1rem 1.1rem; box-shadow:0 2px 4px rgba(0,0,0,0.04);} 
        .card.soft {background:linear-gradient(135deg,#fff,#f9f6ff);} 
        .kpi-value {font-size:1.9rem; font-weight:700; margin:0.1rem 0;}
        .kpi-sub {font-size:0.75rem; color:#0c8f3d; font-weight:500; margin-top:2px;}
        .kpi-sub.neg {color:#c22727;}
    .kpi-badge {background:#f1f5f9; padding:2px 7px; border-radius:20px; font-size:0.65rem; margin-left:6px;}
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Render KPI cards using the visualization pattern from ruf.py (lines 143–167)
    sent = k.get("sentiment_score", {}) or {}
    sent_score = sent.get("score")
    sent_delta = sent.get("delta")
    sent_val_html = f"{sent_score:.1f}" if sent_score is not None else "—"
    if sent_delta is None or use_review_date:
        sent_delta_html = ""
        sent_delta_cls = "kpi-sub"
    else:
        _arrow = "↑" if sent_delta >= 0 else "↓"
        _val = f"+{sent_delta:.1f}" if sent_delta >= 0 else f"{sent_delta:.1f}"
        sent_delta_html = f"{_arrow} {_val} from last month"
        sent_delta_cls = "kpi-sub" if sent_delta >= 0 else "kpi-sub neg"

    vol = k.get("review_volume", {}) or {}
    vol_count = vol.get("count", 0)
    vol_delta = vol.get("delta_pct")
    if vol_delta is None or use_review_date:
        vol_delta_html = ""
        vol_delta_cls = "kpi-sub"
    else:
        _varrow = "↑" if vol_delta >= 0 else "↓"
        _vval = f"+{vol_delta:.1f}%" if vol_delta >= 0 else f"{vol_delta:.1f}%"
        vol_delta_html = f"{_varrow} {_vval} from last month"
        vol_delta_cls = "kpi-sub" if vol_delta >= 0 else "kpi-sub neg"

    urg = k.get("urgent_issues", {}) or {}
    total_urg = urg.get("total", 0)
    crit = urg.get("critical", 0)
    high = urg.get("high", 0)

    team_val = k.get("team_utilization")
    team_val_html = f"{team_val:.0f}%" if team_val else "—"
    team_delta_html = ""  # no change to logic; leave blank if not available

    st.markdown(
            f"""
            <div class='metric-grid'>
                <div class='card soft'>
                    <div>Sentiment Score <span class='kpi-badge'>/10</span></div>
                    <p class='kpi-value'>{sent_val_html}</p>
                    <div class='{sent_delta_cls}'>{sent_delta_html}</div>
                </div>
                <div class='card soft'>
                    <div>Review Volume</div>
                    <p class='kpi-value'>{vol_count:,}</p>
                    <div class='{vol_delta_cls}'>{vol_delta_html}</div>
                </div>
                <div class='card soft'>
                    <div>Urgent Issues</div>
                    <p class='kpi-value'>{total_urg}</p>
                    <div class='kpi-sub'>{crit} critical, {high} high</div>
                </div>
                <div class='card soft'>
                    <div>Team Utilization</div>
                    <p class='kpi-value'>{team_val_html}</p>
                    <div class='kpi-sub'>{team_delta_html}</div>
                </div>
            </div>
                """,
                unsafe_allow_html=True,
        )

    # Historical Sentiment Trend with optional smoothing
    st.markdown("### Historical Sentiment Trend")
    st.caption("Beauty product sentiment over time")
    trend_container = st.container()
    with trend_container:
        # Align trend data with the period selection when not using Review Date
        trend_source_df = filtered_df
        if not use_review_date and filtered_df is not None and not filtered_df.empty and "review_date" in filtered_df.columns:
            dt = pd.to_datetime(filtered_df["review_date"], errors="coerce", utc=True)
            if dt.dropna().shape[0] > 0:
                end_current = dt.max().floor("D")
                start_current = end_current - pd.Timedelta(days=period_days - 1)
                mask = (dt >= start_current) & (dt <= end_current)
                trend_source_df = filtered_df[mask]

        trend_engine = KpiEngine(trend_source_df)
        trend_fn = getattr(trend_engine, "sentiment_trend", None)
        # Choose weekly for <=30 days, else monthly; review-date mode uses monthly
        trend_freq = "M" if (use_review_date or period_days > 30) else "W"
        trend_df = trend_fn(freq=trend_freq, smoothing_window=smoothing_window) if callable(trend_fn) else None
        if trend_df is not None and not trend_df.empty:
            try:
                import matplotlib.pyplot as plt  # lazy import; optional dependency
                import matplotlib.dates as mdates
                fig, ax = plt.subplots(figsize=(7, 3.2))
                ax.fill_between(trend_df["period"], trend_df["smoothed"], color="#ea489c22")
                ax.plot(trend_df["period"], trend_df["smoothed"], color="#ea489c", linewidth=2)
                ax.set_ylabel("Sentiment")
                ax.set_xlabel("Date")
                ax.set_ylim(0, 10)
                # De-clutter ticks: weekly or monthly
                if trend_freq == "W":
                    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
                else:
                    ax.xaxis.set_major_locator(mdates.MonthLocator())
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
                fig.autofmt_xdate()
                ax.grid(alpha=0.2)
                st.pyplot(fig, use_container_width=True)
            except Exception:
                # Fallback to Streamlit built-in area chart if matplotlib unavailable
                tdf = trend_df.set_index("period")["smoothed"]
                st.area_chart(tdf)
        else:
            st.info("Not enough dated sentiment data to plot.")

    # Optional: quick status using COUNT(*) of the whole table
    count = fetch_count("analysis_results")
    if count is None:
        st.caption("Row count unavailable (DB error).")
    else:
        st.caption(f"Total rows in table: {count:,}")

with tab_table:
    st.subheader("Analysis Results")
    st.write("Displaying data from the analysis_results table (filtered):")
    # Reuse filtered_df from Dashboard tab (computed with header Category)
    if 'filtered_df' in locals() and filtered_df is not None and not filtered_df.empty:
        st.dataframe(filtered_df, hide_index=False)
    else:
        st.info("No data to display for the selected filters.")
