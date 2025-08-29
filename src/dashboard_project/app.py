from math import e
import os
import streamlit as st
import pandas as pd
from datetime import datetime

from database.queries import fetch_table, fetch_count
from processing.calculation import get_filter_options, apply_filters, KpiEngine
from processing.team_task import (
    get_support_data,
    refresh_data,
    calculate_performance_metrics,
    fetch_team_performance_data,
    calculate_team_efficiency,
    calculate_time_ago
)

st.set_page_config(layout="wide")


# Always import sync_complaints_from_customer_db for refresh_data
try:
    from database.queries import sync_complaints_from_customer_db
except ImportError:
    sync_complaints_from_customer_db = None
# THEME: Set colors for background and title gradient — change these to adjust look
bg_start = "#e5fbff"   #  (background)
bg_end = "#fdfde9"     # (background)
title_start = "#d37fe3"  # vibrant pink (title gradient start)
title_end = "#008ae4"    # purple (title gradient end)

# THEME: Page background gradient injected via CSS
st.markdown(
        f"""
        <style>
        /* THEME: Page background gradient — edit bg_start/bg_end above */
        [data-testid="stAppViewContainer"] {{
            background: linear-gradient(90deg, {bg_start} 0%, {bg_end} 100%) !important;
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
        <h1 style="margin:0 0 0.75rem 0; text-align:center; font-size:4.4rem; font-weight:800; line-height:1.2; background:linear-gradient(90deg, {title_start}, {title_end}); -webkit-background-clip:text; background-clip:text; color:transparent;">
            🌟 Beauty CX Analytics Hub
        </h1>
        <p style="text-align:center; font-size:1.2rem; color:#64748b; margin-bottom:2rem; font-weight:400;">
            Your comprehensive customer experience intelligence platform
        </p>
        """,
        unsafe_allow_html=True,
)

# Load data once (cache-aware) and build filters
df = fetch_table("analysis_results")
filters = get_filter_options(df)

# Note: Navigation will be handled via st.Page/st.navigation below (with emoji icons).

# Compute KPI dict locally (mirrors calculation.KpiEngine semantics for the dashboard)
def _compute_kpis_for_dashboard(filtered_df: pd.DataFrame, use_review_date: bool, period_days: int) -> dict:
    if not isinstance(filtered_df, pd.DataFrame):
        filtered_df = pd.DataFrame()

    if use_review_date:
        # Sentiment score (overall)
        mapping = {
            "Positive": 10.0, "Neutral": 5.0, "Mixed": 3.0, "Negative": 1.0,
        }
        if not filtered_df.empty and "sentiment" in filtered_df.columns:
            sent_series = filtered_df["sentiment"].map(lambda s: mapping.get(str(s), None)).dropna()
            sent_score_overall = float(sent_series.mean()) if not sent_series.empty else None
        else:
            sent_score_overall = None

        # Review volume (overall)
        vol_overall = int(len(filtered_df)) if filtered_df is not None else 0

        # Urgent issues (overall) – based only on urgency_score thresholds
        if not filtered_df.empty:
            urgency = pd.to_numeric(filtered_df.get("urgency_score", pd.Series(index=filtered_df.index)), errors="coerce")
            urgent_mask = urgency.fillna(0) >= 3
            total_u = int(urgent_mask.sum())
            critical_u = int((urgency >= 5).fillna(False).sum())
            high_u = int(((urgency >= 4) & (urgency < 5)).fillna(False).sum())
        else:
            total_u = critical_u = high_u = 0

        return {
            "sentiment_score": {"score": sent_score_overall, "delta": None},
            "review_volume": {"count": vol_overall, "delta_pct": None},
            "urgent_issues": {"total": total_u, "critical": critical_u, "high": high_u},
        }
    else:
        engine = KpiEngine(filtered_df)
        return engine.compute_all(days=period_days)

def _dashboard_page():
    # Welcome message for first-time users
    st.markdown("""
    <div style="background: linear-gradient(135deg, #faccff 0%, #a178df 50%); padding: 1rem; border-radius: 0.85rem; margin-bottom: 0.5rem; color: white; text-align: center;">
        <h3 style="margin: 0; font-weight: 500; font-size: 1.3rem;">📈 Customer Experience Analytics</h3>
        <p style="margin: 0.3rem 0 0 0; opacity: 0.9; font-size: 0.9rem;">Monitor sentiment trends, track urgent issues, and discover key insights</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Manual refresh button - only on dashboard page
    col1, col2, col3 = st.columns([2, 1, 2])
    with col2:
        if st.button("🔄 Refresh Data", type="primary", help="Get the latest data from database", use_container_width=True):
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
            st.toast("✅ Data refreshed successfully!", icon="🔄")
            st.rerun()
    
    # Sidebar filters - only for Analytics Dashboard
    """📊 Smart Filters - Customize your view"""
    st.sidebar.markdown("### 🎯 Filter Options")

    # Replace ASIN with Title filter
    title_choices = filters.get("title")
    if (not title_choices) and isinstance(df, pd.DataFrame) and ("title" in df.columns):
        try:
            title_choices = sorted([t for t in df["title"].dropna().unique().tolist() if str(t).strip()])
        except Exception:
            title_choices = []
    selected_titles = st.sidebar.multiselect("🏷️ Product Titles", options=title_choices or [], default=[], help="Filter by specific product titles")
    
    # Additional categorical filters
    region_choices = filters.get("region", [])
    selected_regions = st.sidebar.multiselect("📍 Regions", options=region_choices, default=[], help="Select geographical regions")

    sentiment_choices = filters.get("sentiment", [])
    selected_sentiments = st.sidebar.multiselect("😊 Customer Sentiment", options=sentiment_choices, default=[], help="Filter by customer sentiment")

    primary_category_choices = filters.get("primary_category", [])

    # Numeric range filter for urgency_score
    urgency_cfg = filters.get("urgency_score", {}) or {}
    urgency_score_range = None
    if isinstance(urgency_cfg, dict) and urgency_cfg.get("min") is not None and urgency_cfg.get("max") is not None:
        try:
            umin = float(urgency_cfg["min"])
            umax = float(urgency_cfg["max"])
            if umin < umax:
                st.sidebar.markdown("---")
                urgency_score_range = st.sidebar.slider(
                    "⚠️ Urgency Level",
                    min_value=umin,
                    max_value=umax,
                    value=(umin, umax),
                    help="Filter by urgency score range (1=low, 5=critical)"
                )
            else:
                st.sidebar.info("ℹ️ Urgency score data not available")
        except Exception:
            st.sidebar.info("ℹ️ Urgency score data not available")

    # Date range filter for review_date
    review_date_cfg = filters.get("review_date", {}) or {}
    review_date_range = None
    if isinstance(review_date_cfg, dict) and review_date_cfg.get("min") is not None and review_date_cfg.get("max") is not None:
        start_ts = pd.to_datetime(review_date_cfg["min"], errors="coerce")
        end_ts = pd.to_datetime(review_date_cfg["max"], errors="coerce")
        if pd.notna(start_ts) and pd.notna(end_ts) and start_ts <= end_ts:
            start_date = start_ts.date()
            end_date = end_ts.date()
            st.sidebar.markdown("---")
            review_date_range = st.sidebar.date_input(
                "📅 Review Period",
                value=(start_date, end_date),
                min_value=start_date,
                max_value=end_date,
                help="Select date range for customer reviews"
            )
            # Normalize single-date selection to a (start, end) tuple
            if review_date_range is not None and not isinstance(review_date_range, (list, tuple)):
                review_date_range = (review_date_range, review_date_range)
            elif isinstance(review_date_range, (list, tuple)) and len(review_date_range) == 1:
                review_date_range = (review_date_range[0], review_date_range[0])
    
    # Header controls: period selection and Category filter (moved from sidebar)
    st.markdown("### ⏰ Analysis Period & Categories")
    left, mid, right = st.columns([2, 1, 1])
    with mid:
        period_label = st.selectbox(
            "📊 Time Period",
            options=["Last 7 days", "Last 30 days", "Last 90 days", "Use Review Date"],
            index=1,
            help="Choose your analysis time frame"
        )
    with right:
        selected_primary_categories = st.multiselect(
            "🏷️ Product Categories",
            options=primary_category_choices,
            default=st.session_state.get("primary_category_top", []),
            key="primary_category_top",
            help="Filter by product categories",
        )

    # Apply filters (including Category from header). Prefilter by Title and drop ASIN filter.
    base_df = df
    if selected_titles:
        try:
            base_df = base_df[base_df["title"].isin(selected_titles)]
        except Exception:
            pass
    use_review_date = period_label == "Use Review Date"
    effective_review_date_range = review_date_range if use_review_date else None
    # Expand Review date range by ±1 day on both sides for Dashboard filtering
    if use_review_date and isinstance(effective_review_date_range, (list, tuple)) and len(effective_review_date_range) == 2:
        try:
            _s, _e = effective_review_date_range
            _s_ts = pd.to_datetime(_s, errors="coerce")
            _e_ts = pd.to_datetime(_e, errors="coerce")
            if pd.notna(_s_ts) and pd.notna(_e_ts):
                effective_review_date_range = (
                    (_s_ts - pd.Timedelta(days=1)).date(),
                    (_e_ts + pd.Timedelta(days=1)).date(),
                )
        except Exception:
            pass
    filtered_df = apply_filters(
        base_df,
        asin_values=[],
        region_values=selected_regions,
        sentiment_values=selected_sentiments,
        primary_category_values=selected_primary_categories,
        urgency_score_range=urgency_score_range,
        review_date_range=effective_review_date_range,
    )

    # KPI cards
    days_map = {"Last 7 days": 7, "Last 30 days": 30, "Last 90 days": 90}
    period_days = days_map.get(period_label, 30)
    k = _compute_kpis_for_dashboard(filtered_df, use_review_date, period_days)

    # KPI CSS: soft blue background + gauge styles
    st.markdown(
        """
        <style>
    .metric-grid {display:grid; gap:1rem; grid-template-columns:repeat(4, minmax(200px, 1fr)); align-items:stretch; --kpi-card-h: 210px;}
        .card {background:#ffffff; border:1px solid #dbe6ff; border-radius:16px; padding:1rem 1.1rem; box-shadow:0 2px 8px rgba(30,64,175,0.08); min-height:var(--kpi-card-h); height:var(--kpi-card-h); display:flex; flex-direction:column;}
        .card.soft {background:linear-gradient(180deg,#eaf1ff,#edf2ff);} 
        .kpi-value {font-size:1.9rem; font-weight:700; margin:0.1rem 0;}
        .kpi-sub {font-size:0.8rem; color:#0c8f3d; font-weight:600; margin-top:2px;}
        .kpi-sub.neg {color:#c22727;}
    .kpi-badge {background:#e7ecff; padding:2px 7px; border-radius:20px; font-size:0.7rem; margin-left:6px;}
    /* Centered number card variant */
    .card.center {position:relative; align-items:center; justify-content:center; text-align:center;}
    .card.center > div:first-child {position:absolute; top:10px; left:12px; font-weight:600; font-size:0.9rem; color:#334155; opacity:0.9;}
    .card.center .kpi-value {font-size:5rem; line-height:1; margin:0;}
    .card.center .kpi-sub {position:absolute; bottom:10px; left:12px; right:12px; text-align:center;}
    /* Tinted badges for Urgent Issues */
    .badge {display:flex; flex-direction:column; align-items:flex-start; gap:2px; padding:8px 10px; border-radius:10px; border:1px solid transparent;}
    .badge-critical {background:#ffe8e8; border-color:#ffc9c9; color:#b91c1c;}
    .badge-high {background:#fff4e5; border-color:#ffe1b4; color:#92400e;}
        /* Gauge card */
        .kpi-gauge-card {position:relative; background:linear-gradient(180deg,#eaf1ff,#edf2ff); border:1px solid #dbe6ff; border-radius:16px; padding:1rem 1.1rem; box-shadow:0 2px 8px rgba(30,64,175,0.08); min-height:var(--kpi-card-h); height:var(--kpi-card-h);} 
        .kpi-pill {position:absolute; top:10px; right:12px; background:#e6e6ff; color:#111827; font-weight:800; padding:4px 10px; border-radius:14px; font-size:0.95rem;}
    .gauge-wrap {position:relative; width:100%; height:100px;}
        .gauge {position:absolute; left:0; right:0; top:0; bottom:0; display:flex; align-items:center; justify-content:center;}
        .gauge svg {width:100%; height:100%;}
    .gauge-center {position:absolute; left:0; right:0; top:36px; text-align:center;}
        .gauge-emoji {font-size:44px; line-height:44px;}
        .gauge-status {font-size:0.95rem; font-weight:700; margin-top:6px;}
    .gauge-labels {display:flex; justify-content:space-between; font-weight:700; margin-top:7px;}
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
        sent_delta_html = f"{_arrow} {_val} from last period"
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
        vol_delta_html = f"{_varrow} {_vval} from last period"
        vol_delta_cls = "kpi-sub" if vol_delta >= 0 else "kpi-sub neg"

    urg = k.get("urgent_issues", {}) or {}
    total_urg = int(urg.get("total", 0) or 0)
    crit = int(urg.get("critical", 0) or 0)
    high = int(urg.get("high", 0) or 0)

    # Build gauge KPI for Sentiment Average (normalize 0..10 -> -1..1)
    _norm = None
    if sent_score is not None:
        try:
            _norm = max(-1.0, min(1.0, (float(sent_score) - 5.0) / 5.0))
        except Exception:
            _norm = None
    _percent = ((_norm + 1) / 2) if _norm is not None else None
    _offset = (100.0 * (1.0 - _percent)) if _percent is not None else 100.0
    if _norm is None:
        _emoji, _status_text, _status_color, _pill_text = "❔", "—", "#64748b", "—"
    else:
        if _norm >= 0.25:
            _emoji, _status_text, _status_color = "😄", "Positive", "#16a34a"
        elif _norm <= -0.25:
            _emoji, _status_text, _status_color = "😞", "Negative", "#dc2626"
        else:
            _emoji, _status_text, _status_color = "😐", "Neutral", "#64748b"
        _pill_text = f"{_norm:.2f}"

    # Build optional delta block for the gauge
    sent_delta_block = (
        f"<div class='{sent_delta_cls}' style='margin-top:6px;'>{sent_delta_html}</div>"
        if sent_delta_html else ""
    )

    gauge_card_html = (
        f"<div class='kpi-gauge-card'>"
        f"<div style='font-weight:800;'>Sentiment Average</div>"
        f"<div class='kpi-pill'>{_pill_text}</div>"
        f"<div class='gauge-wrap'>"
        f"<div class='gauge'>"
        f"<svg viewBox='0 0 240 140'>"
        f"<path d='M20 120 A 100 100 0 0 1 220 120' fill='none' stroke='#d9d6ff' stroke-width='14' stroke-linecap='round' pathLength='100' />"
        f"<path d='M20 120 A 100 100 0 0 1 220 120' fill='none' stroke='#16a34a' stroke-width='14' stroke-linecap='round' pathLength='100' style='stroke-dasharray:100; stroke-dashoffset:{_offset};'/>"
        f"</svg>"
        f"</div>"
        f"<div class='gauge-center'>"
        f"<div class='gauge-emoji'>{_emoji}</div>"
        f"<div class='gauge-status' style='color:{_status_color}'>{_status_text}</div>"
        f"</div>"
    f"</div>"
    f"<div class='gauge-labels'><span>-1.00</span><span>1.00</span></div>"
    f"{sent_delta_block}"
        f"</div>"
    )

    # Construct KPI grid HTML without indent so Markdown doesn't treat as code
    kpi_grid_html = (
        "<div class='metric-grid' style='grid-template-columns:repeat(3, minmax(200px, 1fr));'>"
        f"{gauge_card_html}"
    "<div class='card soft center'>"
    "<div>Review Volume</div>"
    f"<p class='kpi-value'>{vol_count:,}</p>"
    f"<div class='{vol_delta_cls}'>{vol_delta_html}</div>"
    "</div>"
    "<div class='card soft' style='position:relative;'>"
    "<div style='font-weight:800;'>Urgent Issues</div>"
    f"<div class='kpi-pill' title='Urgency score ≥ 3'>{total_urg}</div>"
    f"<div style='display:flex;flex-direction:column;gap:8px;margin-top:12px;'>"
    f"<div class='badge badge-critical' style='height:70px; display:flex; flex-direction:row; align-items:center; justify-content:space-between; padding:10px 12px;'>"
    f"<div style='font-size:1.3rem;font-weight:500;'>Critical</div>"
    f"<div style='font-size:1.3rem;font-weight:500;'>{crit}</div>"
    f"</div>"
    f"<div class='badge badge-high' style='height:70px; display:flex; flex-direction:row; align-items:center; justify-content:space-between; padding:10px 12px;'>"
    f"<div style='font-size:1.3rem;font-weight:500;'>High</div>"
    f"<div style='font-size:1.3rem;font-weight:500;'>{high}</div>"
    f"</div>"
    f"</div>"
    "</div>"
        "</div>"
    )

    st.markdown(kpi_grid_html, unsafe_allow_html=True)

    # Add helpful information about KPIs
    with st.expander("ℹ️ Understanding Your Metrics", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            **🎯 Sentiment Average**
            - Range: -1.00 to 1.00
            - Higher = More positive feedback
            - Calculated from customer reviews
            """)
        with col2:
            st.markdown("""
            **📊 Review Volume**
            - Total customer reviews
            - Shows period-over-period change
            - Indicates engagement levels
            """)
        with col3:
            st.markdown("""
            **⚠️ Urgent Issues**
            - Critical: Urgency score ≥ 5
            - High: Urgency score 4-5
            - Requires immediate attention
            """)

    # Historical Sentiment Trend (left) and Region Map (right)
    st.markdown("---")
    st.markdown("### 📈 Data Insights & Trends")
    # st.caption("📊 Left: Customer urgency patterns • 🗺️ Right: Geographic distribution of reviews")
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

        left_col, right_col = st.columns(2)

    # --- Left: Urgency Score Distribution Bar Chart ---
        with left_col:
            st.caption("📊 Left: Customer urgency patterns")
            if trend_source_df is not None and not trend_source_df.empty and "urgency_score" in trend_source_df.columns:
                try:
                    import altair as alt  # lazy import
                    
                    # Process urgency score data
                    urgency_data = trend_source_df["urgency_score"].dropna()
                    
                    if not urgency_data.empty:
                        # Create urgency score bins/ranges (1-5 scale)
                        urgency_data = pd.to_numeric(urgency_data, errors="coerce").dropna()
                        
                        # Define bins for urgency scores (1-5 scale)
                        bins = [0, 1, 2, 3, 4, 5]
                        labels = ['1', '2', '3', '4', '5']
                        
                        # Bin the data
                        binned_data = pd.cut(urgency_data, bins=bins, labels=labels, include_lowest=True)
                        urgency_counts = binned_data.value_counts().reset_index()
                        urgency_counts.columns = ['urgency_score', 'count']
                        urgency_counts = urgency_counts.sort_values('urgency_score')
                        
                        # Create bar chart with Altair and blue background
                        chart = alt.Chart(urgency_counts).mark_bar(
                            color='#3b82f6',  # Blue bars
                            strokeWidth=1,
                            stroke='#1e40af'
                        ).encode(
                            x=alt.X('urgency_score:O', title='Urgency Score', axis=alt.Axis(labelAngle=0)),
                            y=alt.Y('count:Q', title='Number of Reviews'),
                            tooltip=['urgency_score:O', 'count:Q']
                        ).properties(
                            height=320,  # Increased to match map height
                            title='Customer Issue Urgency Distribution',
                            background='#eff6ff'  # Light blue background
                        ).configure(
                            background='#eff6ff'  # Chart background
                        ).configure_axis(
                            grid=True, 
                            gridOpacity=0.3, 
                            gridColor='#bfdbfe',
                            domain=False, 
                            labelPadding=6, 
                            titlePadding=12
                        ).configure_view(
                            strokeWidth=0,
                            fill='#eff6ff'  # View background
                        )
                        
                        st.altair_chart(chart, use_container_width=True)
                    else:
                        st.info("📊 No urgency score data available to display")
                        
                except Exception as e:
                    # Fallback to Streamlit built-in bar chart
                    try:
                        urgency_data = trend_source_df["urgency_score"].dropna()
                        if not urgency_data.empty:
                            urgency_data = pd.to_numeric(urgency_data, errors="coerce").dropna()
                            
                            # Create histogram-like data for bar chart (1-5 scale)
                            bins = [0, 1, 2, 3, 4, 5]
                            labels = ['1', '2', '3', '4', '5']
                            binned_data = pd.cut(urgency_data, bins=bins, labels=labels, include_lowest=True)
                            urgency_counts = binned_data.value_counts().sort_index()
                            
                            st.subheader("📊 Customer Issue Urgency Distribution")
                            st.bar_chart(urgency_counts)
                        else:
                            st.info("📊 No urgency score data available to display")
                    except Exception:
                        st.info("📊 Unable to generate urgency distribution chart")
            else:
                st.info("📊 No urgency score data available to display")

        # --- Right: India map for 5 cities with circle size by review count ---
        with right_col:
            st.caption("🗺️ Right: Geographic distribution of reviews")
            df_map_src = trend_source_df if trend_source_df is not None else pd.DataFrame()
            if df_map_src is not None and not df_map_src.empty and "region" in df_map_src.columns:
                # Use Folium (Leaflet) so tiles render without Mapbox token
                import folium
                from streamlit_folium import st_folium

                def _norm(r: str) -> str:
                    return str(r).strip().lower()

                # Canonical 5 cities and synonyms -> canonical key
                CANON = {
                    "delhi": "delhi", "Delhi": "delhi",
                    "mumbai": "mumbai", "Mumbai": "mumbai", "Bombay": "mumbai",
                    "chennai": "chennai", "Chennai": "chennai",
                    "kolkata": "kolkata", "Calcutta": "kolkata",
                    "bangalore": "bengaluru",
                }
                CITY_COORDS = {
                    "delhi": (28.6139, 77.2090),
                    "mumbai": (19.0760, 72.8777),
                    "chennai": (13.0827, 80.2707),
                    "kolkata": (22.5726, 88.3639),
                    "bengaluru": (12.9716, 77.5946),
                }

                # Map all region values to the 5 canonical cities
                mapped = df_map_src["region"].astype(str).apply(_norm).map(CANON).dropna()
                if mapped.empty:
                    st.info("🗺️ No supported cities found (Delhi, Mumbai, Chennai, Kolkata, Bengaluru)")
                else:
                    counts = mapped.value_counts().rename_axis("city").reset_index(name="count")
                    counts["lat"] = counts["city"].map(lambda c: CITY_COORDS[c][0])
                    counts["lon"] = counts["city"].map(lambda c: CITY_COORDS[c][1])

                    # Build map centered on India
                    m = folium.Map(location=[22.9734, 78.6569], zoom_start=3.8, tiles="CartoDB Positron")

                    # Scale radius by count (pixels). Keep minimum visible size.
                    max_c = max(int(counts["count"].max()), 1)

                    def _radius(c):
                        base = 8
                        return base + int(22 * (c / max_c))

                    for _, row in counts.iterrows():
                        folium.CircleMarker(
                            location=(row["lat"], row["lon"]),
                            radius=_radius(row["count"]),
                            color="#6a00ff",
                            weight=1,
                            fill=True,
                            fill_color="#d800b87a",
                            fill_opacity=0.6,
                            tooltip=f"{row['city'].title()}: {int(row['count'])} reviews",
                        ).add_to(m)
                        # Add a label with the count value at the center of the circle
                        folium.map.Marker(
                            [row["lat"], row["lon"]],
                            icon=folium.DivIcon(
                                html=f"""<div style="font-size: 16px; color: #222; font-weight: bold; text-align: center;">{int(row['count'])}</div>"""
                            ),
                        ).add_to(m)

                    st_folium(m, width=None, height=320)

    # Beauty Sentiment Drivers (computed in calculation.py)
    st.markdown("---")
    st.markdown("### 🎯 Key Customer Sentiment Drivers")
    st.caption("💡 Discover what makes customers happy or frustrated with your products")

    # Attach Positive/Negative list columns to filtered_df using engine
    drivers_engine = KpiEngine(filtered_df)
    # Respect period when not using 'Use Review Date'
    _driver_days = None if use_review_date else period_days
    lists_df = drivers_engine.key_driver_lists(days=_driver_days)
    try:
        _df_copy = filtered_df.copy()
        for col in ["Positive", "Negative"]:
            if col in lists_df.columns:
                _df_copy[col] = lists_df[col]
        filtered_df = _df_copy
    except Exception:
        pass

    # Get top 6 positive/negative with counts
    top = drivers_engine.top_key_drivers(n=6, days=_driver_days)
    pos_top = top.get("positive", [])
    neg_top = top.get("negative", [])

    def _pad(items, size: int = 6):
        items = items[:size]
        return items + [None] * (size - len(items))

    # Minimal card CSS (green for positive, red for negative) apart from that every thing should be same
    st.markdown(
        """
        <style>
        .driver-card {background:#f2fcf6; border:1px solid #d6f5e1; border-radius:12px; padding:0.9rem 1rem; min-height:90px;}
        .driver-card.blank {background:transparent; border:1px dashed #e8e8e8;}
        .driver-card.neg {background:#ffc4c4; border:1px solid #ffd7d7; border-radius:12px; padding:0.9rem 1rem; min-height:90px;}
        .driver-card.blank.neg {background:transparent; border:1px dashed #ffd7d7;}
        .driver-title {font-weight:600; margin-bottom:0.4rem;}
        .driver-meta {font-size:0.75rem; opacity:0.9;}
        </style>
        """,
        unsafe_allow_html=True,
    )

    def _render_topic_cards(items, subtitle: str, negative: bool = False):
        items = _pad(items, 6)
        # Render as 2 rows x 3 columns using Streamlit columns
        for r in range(2):
            cols = st.columns(3)
            for c in range(3):
                idx = r * 3 + c
                with cols[c]:
                    item = items[idx]
                    if item is None:
                        klass = "driver-card blank neg" if negative else "driver-card blank"
                        st.markdown(f"<div class='{klass}'></div>", unsafe_allow_html=True)
                    else:
                        title, cnt = item
                        klass = "driver-card neg" if negative else "driver-card"
                        st.markdown(
                            f"""
                            <div class='{klass}'>
                            <div class='driver-title'>{title}</div>
                            <div class='driver-meta'>{subtitle} <span style='float:right;font-weight:600'>{cnt}</span></div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )


    p_tabs = st.tabs(["✅ What Customers Love", "❌ Areas for Improvement"])
    with p_tabs[0]:
        _render_topic_cards(pos_top, "Mentions", negative=False)
    with p_tabs[1]:
        _render_topic_cards(neg_top, "Mentions", negative=True)

    # Recommended Actions (separate section below drivers)
    st.markdown("---")
    st.markdown("### 🚀 Recommended Actions")
    st.caption("🎯 AI-powered suggestions to improve your customer experience")
    colA, colB = st.columns(2)
    with colA:
        st.markdown("**🔥 Immediate Actions**")
        st.markdown(
                """
                <div style='background:#fff6f6; border:1px solid #ffd7d7; border-radius:12px; padding:0.9rem 1rem; margin-bottom:0.8rem;'>
                    <div style='font-weight:600; color:#c22727;'>⚠️ Address Allergic Reactions</div>
                    <div style='font-size:0.85rem; opacity:0.9;'>Contact customers with skin reactions immediately</div>
                </div>
                <div style='background:#fff9ee; border:1px solid #ffe3c2; border-radius:12px; padding:0.9rem 1rem;'>
                    <div style='font-weight:600; color:#b96a00;'>🎯 Improve Color Matching</div>
                    <div style='font-size:0.85rem; opacity:0.9;'>Review shade descriptions and add more photos</div>
                </div>
                """,
                unsafe_allow_html=True,
        )
    with colB:
        st.markdown("**📈 Strategic Improvements**")
        st.markdown(
                """
                <div style='background:#eef3ff; border:1px solid #d5e2ff; border-radius:12px; padding:0.9rem 1rem; margin-bottom:0.8rem;'>
                    <div style='font-weight:600; color:#1f4bff;'>📈 Expand Shade Range</div>
                    <div style='font-size:0.85rem; opacity:0.9;'>Customers requesting more inclusive shade options</div>
                </div>
                <div style='background:#f2fcf6; border:1px solid #d6f5e1; border-radius:12px; padding:0.9rem 1rem;'>
                    <div style='font-weight:600; color:#0c8f3d;'>✅ Highlight Longevity</div>
                    <div style='font-size:0.85rem; opacity:0.9;'>Promote long-lasting formulas in marketing</div>
                </div>
                """,
                unsafe_allow_html=True,
        )

    # Optional: quick status using COUNT(*) of the whole table
    st.markdown("---")
    count = fetch_count("analysis_results")
    if count is None:
        st.caption("ℹ️ Database connection status: Checking...")
    else:
        st.caption(f"📊 Total data points analyzed: **{count:,}** customer reviews")
        
    # Add footer with tips
    st.markdown("---")
    with st.expander("💡 Tips for Using This Dashboard", expanded=False):
        st.markdown("""
        **🎯 Getting Started:**
        - Use the sidebar filters to focus on specific products, regions, or time periods
        - Toggle between different time periods using the dropdown above
        - Hover over charts for detailed information
        
        **📊 Understanding Your Data:**
        - Green indicators = positive trends
        - Red indicators = areas needing attention
        - Higher urgency scores require immediate action
        
        **🔄 Keeping Data Fresh:**
        - Click "Refresh Data" to get the latest customer feedback
        - Data updates automatically reflect in all charts and metrics
        """)

def _table_page():
    st.markdown("""
    <div style="background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%); padding: 1rem; border-radius: 0.75rem; margin-bottom: 1.5rem; color: white; text-align: center;">
        <h3 style="margin: 0; font-weight: 500; font-size: 1.3rem;">📋 Raw Data Explorer</h3>
        <p style="margin: 0.3rem 0 0 0; opacity: 0.9; font-size: 0.9rem;">Dive deep into your customer feedback data</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 🔍 Detailed Analysis Results")
    st.write("📊 Explore the complete dataset:")
    # Show all data without filters for Table page
    table_df = df
    if table_df is not None and not table_df.empty:
        st.success(f"✅ Found **{len(table_df):,}** records in total")
        
        # Add download option
        csv = table_df.to_csv(index=False)
        st.download_button(
            label="💾 Download as CSV",
            data=csv,
            file_name=f"customer_analysis_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            help="Download complete dataset for external analysis"
        )
        
        st.dataframe(table_df, hide_index=False, use_container_width=True)
    else:
        st.info("ℹ️ No data available in the database. Please check your data source.")


try:
    from database.queries import (
        fetch_complaints_with_departments,
    )
    from database.connector import get_customer_db_connection
except Exception:
    # If imports fail because of path, try adjusting sys.path
    import sys
    sys.path.append(os.path.dirname(__file__))
    from database.connector import get_customer_db_connection

def refresh_data() -> bool:
    """
    Refresh the data by syncing from customer database (following temp_dashboard.py pattern).
    """
    try:
        # Use the same sync method as temp_dashboard.py
        n = sync_complaints_from_customer_db("complaints")
        
        # Clear Streamlit caches to ensure fresh connections/data - like temp_dashboard.py
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
            st.success(f"Successfully refreshed data ({n} rows) from customer database")
        else:
            st.info("Refresh completed but no rows were processed")
        return True
    except Exception as e:
        st.error(f"Failed to refresh data: {e}")
        return False

def _team_management_page():
    """Team Management page with customer support queue and team performance"""
    
    # Welcome header for team management
    st.markdown("""
    <div style="background: linear-gradient(135deg, #c0b0e8 0%, #008490 90%); padding: 1rem; border-radius: 0.75rem; margin-bottom: 1.5rem; color: white; text-align: center;">
        <h3 style="margin: 0; font-weight: 500; font-size: 1.3rem;">👥 Team Operations Center</h3>
        <p style="margin: 0.3rem 0 0 0; opacity: 0.9; font-size: 0.9rem;">Monitor team performance and manage customer support queue</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Apply custom CSS for team management styling
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
        """Render the urgent feedback queue card"""
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
        
        # Filter and prepare display data directly from support data
        try:
            # Filter for open/unresolved issues
            urgent_df = df[df['status'].str.lower().isin(['open', 'pending', 'new'])].copy()
            
            if urgent_df.empty:
                st.info("No urgent items in the queue.")
                return
            
            # Sort by urgency if available, otherwise by created_at (newest first)
            if 'urgency' in urgent_df.columns:
                urgent_df = urgent_df.sort_values('urgency', ascending=False)
            elif 'priority' in urgent_df.columns:
                urgent_df = urgent_df.sort_values('priority', ascending=False)
            elif 'created_at' in urgent_df.columns:
                urgent_df = urgent_df.sort_values('created_at', ascending=False)
            
            # Limit to top 5
            urgent_df = urgent_df.head(5)
            
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
            for idx, row in urgent_df.iterrows():
                # Get customer name (adapt to different column names)
                customer_name = row.get('user_name', row.get('customer_name', row.get('name', 'Unknown')))
                
                # Get issue description (adapt to different column names)
                full_issue = row.get('summary', row.get('description', row.get('issue', 'No summary available')))
                
                # Get assigned team member
                assigned_to = row.get('team_member_name', row.get('assigned_to', 'Unassigned'))
                
                # Get department
                dept_name = row.get('department_name', 'No Department')
                
                # Get time
                time_info = calculate_time_ago(str(row.get('created_at', ''))) if 'created_at' in row else ''
                
                col1, col2, col3, col4 = st.columns([2, 4, 2, 1])
                
                with col1:
                    st.markdown(f"**{customer_name}**")
                
                with col2:
                    # Display truncated issue with tooltip for full text
                    if len(str(full_issue)) > 80:
                        truncated_issue = str(full_issue)[:80] + "..."
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
                        if dept_name and dept_name != 'No Department':
                            st.caption(dept_name)
                    else:
                        st.markdown("*Unassigned*")
                
                with col4:
                    if time_info:
                        st.caption(time_info)
                
                st.markdown("<br>", unsafe_allow_html=True)
                
        except Exception as e:
            st.error(f"Error displaying urgent queue: {e}")
            st.info("No items in the queue.")

    def render_team_performance_card(team_df: pd.DataFrame):
        """Render the team performance card"""
        if team_df.empty:
            st.info("No team performance data available.")
            return
        
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
        
        # Display each team member (limited to 5)
        for idx, row in team_df.head(5).iterrows():
            col1, col2, col3 = st.columns([3, 2, 2])
            
            with col1:
                department = row.get('department_name', 'No Department')
                st.markdown(f"**{row['name']}**")
                st.caption(department)
            
            with col2:
                total_tasks = int(row['total_tasks'])
                completed_tasks = int(row['completed_tasks'])
                
                st.markdown(f"{total_tasks} assigned")
                st.caption(f"{completed_tasks} completed")
            
            with col3:
                completion_pct = int(row['completion_percentage'])
                progress_color = "#10b981" if completion_pct >= 80 else "#f59e0b" if completion_pct >= 60 else "#ef4444"
                
                st.markdown(f"""
                <div style="background-color: #f3f4f6; border-radius: 0.5rem; height: 0.5rem; margin: 0.25rem 0;">
                    <div style="background-color: {progress_color}; width: {completion_pct}%; height: 100%; border-radius: 0.5rem;"></div>
                </div>
                <small style="color: #6b7280;">{completion_pct}%</small>
                """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)

    def render_metrics_cards(metrics: dict, team_performance_df: pd.DataFrame):
        """Render metrics cards"""
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div class="metrics-card">
                <div class="metric-value">{}</div>
                <div class="metric-label">Total Support Items</div>
            </div>
            """.format(metrics.get('total_items', 0)), unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metrics-card">
                <div class="metric-value" style="color: #dc2626;">{}</div>
                <div class="metric-label">Total Unresolved</div>
            </div>
            """.format(metrics.get('unresolved_items', 0)), unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="metrics-card">
                <div class="metric-value" style="color: #059669;">{}</div>
                <div class="metric-label">Last 24h</div>
            </div>
            """.format(metrics.get('recent_24h', 0)), unsafe_allow_html=True)
        
        with col4:
            efficiency = calculate_team_efficiency(team_performance_df)
            st.markdown("""
            <div class="metrics-card">
                <div class="metric-value" style="color: #7c3aed;">{}</div>
                <div class="metric-label">Team Efficiency</div>
            </div>
            """.format(f"{efficiency}%"), unsafe_allow_html=True)

    # Main Team Management Page Content
    st.markdown("### 🎧 Live Support Dashboard")
    
    # Add refresh button at the top - following temp_dashboard.py pattern
    col_refresh1, col_refresh2, col_refresh3 = st.columns([1, 2, 1])
    with col_refresh2:
        if st.button("🔄 Sync and Refresh", type="primary", use_container_width=True):
            with st.spinner("🔄 Syncing and refreshing..."):
                success = refresh_data()
                if success:
                    # Rerun to refresh UI with new data - like temp_dashboard.py
                    try:
                        st.rerun()
                    except Exception:
                        try:
                            st.experimental_rerun()
                        except Exception:
                            pass

    # Load and process data - following temp_dashboard.py pattern
    with st.spinner("📊 Loading customer support data..."):
        support_df = get_support_data()
    
    # Even if empty, continue with fallback data (temp_dashboard.py approach)
    
    # Process team performance data
    try:
        team_performance_df = fetch_team_performance_data()
    except Exception as e:
        st.error(f"Failed to load team performance data: {e}")
        print(f"\n\n\n There is error \n\n\n")
        team_performance_df = pd.DataFrame()
    
    # Get metrics
    metrics = calculate_performance_metrics(support_df)
    
    # Display metrics cards
    st.markdown("### 📊 Key Performance Metrics")
    render_metrics_cards(metrics, team_performance_df)
    
    st.markdown("---")
    
    # Main content area with Issues Queue and Team Performance side by side
    st.markdown("### 🔥 Active Operations")
    col_main1, col_main2 = st.columns([1, 1])
    
    with col_main1:
        # Pass the support data directly to render_urgent_queue_card
        render_urgent_queue_card(support_df)
    
    with col_main2:
        render_team_performance_card(team_performance_df)
    
    # Footer with last updated time
    st.markdown("---")
    st.success(f"✅ Dashboard last updated: **{datetime.now().strftime('%Y-%m-%d at %H:%M:%S')}**")
    
    # Add team management tips
    with st.expander("💡 Team Management Tips", expanded=False):
        st.markdown("""
        **🎯 Prioritization:**
        - Focus on Critical (urgency 5) issues first
        - Monitor team efficiency percentage
        - Track department-wise complaint distribution
        
        **📈 Performance Optimization:**
        - Green completion rates (80%+) indicate high performance
        - Yellow rates (60-80%) need attention
        - Red rates (<60%) require immediate intervention
        
        **🔄 Best Practices:**
        - Monitor team efficiency and task completion rates
        - Hover over charts for detailed insights
        - Use filters to focus on specific time periods or regions
        """)

# --- Navigation: st.Page API with emoji icons (fallback to radio if unavailable) ---
if hasattr(st, "Page") and hasattr(st, "navigation"):
    pages = [
        st.Page(_dashboard_page, title="Analytics Dashboard", icon="📊"),
        st.Page(_team_management_page, title="Team Operations", icon="👥"),
        st.Page(_table_page, title="Data Explorer", icon="📋"),
    ]
    st.navigation(pages).run()
else:
    # Fallback to legacy sidebar radio if running on older Streamlit
    st.sidebar.markdown("### 🧭 Navigation")
    _page = st.sidebar.radio("Choose a view:", ["Analytics Dashboard", "Team Operations", "Data Explorer"], index=0)
    if _page == "Analytics Dashboard":
        _dashboard_page()
    elif _page == "Team Operations":
        _team_management_page()
    else:
        _table_page()
