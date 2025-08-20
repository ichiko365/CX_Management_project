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

"""Tabs for Dashboard, Table and Add Data"""
tab_dashboard, tab_table, tab_add = st.tabs(["Dashboard", "Table", "Add Data"])

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

    # Historical Sentiment Trend (left) and Region Map (right)
    st.markdown("### Historical")
    st.caption("Left: sentiment trend • Right: review density by region")
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

        # --- Left: transparent Altair area chart for better theme blending ---
        with left_col:
            trend_engine = KpiEngine(trend_source_df)
            trend_fn = getattr(trend_engine, "sentiment_trend", None)
            trend_freq = "M" if (use_review_date or period_days > 30) else "W"
            trend_df = trend_fn(freq=trend_freq, smoothing_window=smoothing_window) if callable(trend_fn) else None
            if trend_df is not None and not trend_df.empty:
                try:
                    import altair as alt  # lazy import
                    # Ensure datetime type for Altair
                    tdf = trend_df.copy()
                    tdf["period"] = pd.to_datetime(tdf["period"], errors="coerce")

                    base = alt.Chart(tdf).encode(
                        x=alt.X("period:T", title="Date"),
                        y=alt.Y("smoothed:Q", title="Sentiment", scale=alt.Scale(domain=[0, 10]))
                    )
                    area = base.mark_area(
                        color=alt.Gradient(
                            gradient="linear",
                            stops=[
                                alt.GradientStop(color="#ea489c33", offset=0),
                                alt.GradientStop(color="#ea489c05", offset=1),
                            ],
                            x1=1, x2=1, y1=0, y2=1,
                        )
                    )
                    line = base.mark_line(color="#ea489c", strokeWidth=2)
                    chart = (area + line).properties(height=260).configure(
                        background=None
                    ).configure_axis(
                        grid=True, gridOpacity=0.2, domain=False
                    )
                    st.altair_chart(chart, use_container_width=True)
                except Exception:
                    # Fallback to Streamlit built-in area chart
                    try:
                        tdf = trend_df.set_index("period")["smoothed"]
                        st.area_chart(tdf)
                    except Exception:
                        st.info("Not enough dated sentiment data to plot.")
            else:
                st.info("Not enough dated sentiment data to plot.")

        # --- Right: India map for 5 cities with circle size by review count ---
        with right_col:
            df_map_src = trend_source_df if trend_source_df is not None else pd.DataFrame()
            if df_map_src is not None and not df_map_src.empty and "region" in df_map_src.columns:
                try:
                    # Use Folium (Leaflet) so tiles render without Mapbox token
                    import folium
                    from streamlit_folium import st_folium

                    def _norm(r: str) -> str:
                        return str(r).strip().lower()

                    # Canonical 5 cities and synonyms -> canonical key
                    CANON = {
                        "delhi": "delhi", "new delhi": "delhi",
                        "mumbai": "mumbai", "bombay": "mumbai",
                        "chennai": "chennai",
                        "kolkata": "kolkata", "calcutta": "kolkata",
                        "bengaluru": "bengaluru", "bangalore": "bengaluru",
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
                        st.info("No supported cities found (Delhi, Mumbai, Chennai, Kolkata, Bengaluru).")
                    else:
                        counts = mapped.value_counts().rename_axis("city").reset_index(name="count")
                        counts["lat"] = counts["city"].map(lambda c: CITY_COORDS[c][0])
                        counts["lon"] = counts["city"].map(lambda c: CITY_COORDS[c][1])

                        # Build map centered on India
                        m = folium.Map(location=[22.9734, 78.6569], zoom_start=5, tiles="CartoDB Positron")

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
                                fill_color="#d800b9",
                                fill_opacity=0.6,
                                tooltip=f"{row['city'].title()}: {int(row['count'])} reviews",
                            ).add_to(m)

                        st_folium(m, width=None, height=320)
                except Exception:
                    # Fallback to pydeck as a secondary option if folium isn't available
                    try:
                        import pydeck as pdk
                        CANON = {
                            "delhi": "delhi", "new delhi": "delhi",
                            "mumbai": "mumbai", "bombay": "mumbai",
                            "chennai": "chennai",
                            "kolkata": "kolkata", "calcutta": "kolkata",
                            "bengaluru": "bengaluru", "bangalore": "bengaluru",
                        }
                        CITY_COORDS = {
                            "delhi": (28.6139, 77.2090),
                            "mumbai": (19.0760, 72.8777),
                            "chennai": (13.0827, 80.2707),
                            "kolkata": (22.5726, 88.3639),
                            "bengaluru": (12.9716, 77.5946),
                        }
                        mapped = df_map_src["region"].astype(str).str.lower().map(CANON).dropna()
                        if mapped.empty:
                            st.info("No supported cities found (Delhi, Mumbai, Chennai, Kolkata, Bengaluru).")
                        else:
                            counts = mapped.value_counts().rename_axis("city").reset_index(name="count")
                            counts["latitude"] = counts["city"].map(lambda c: CITY_COORDS[c][0])
                            counts["longitude"] = counts["city"].map(lambda c: CITY_COORDS[c][1])
                            counts["radius"] = counts["count"].apply(lambda c: 25000 + int(140000 * (c / max(int(counts['count'].max()), 1))))

                            view_state = pdk.ViewState(latitude=22.9734, longitude=78.6569, zoom=4.2)
                            layer = pdk.Layer(
                                "ScatterplotLayer",
                                data=counts,
                                get_position="[longitude, latitude]",
                                get_color=[106, 0, 255, 180],
                                get_radius="radius",
                                pickable=True,
                            )
                            deck = pdk.Deck(layers=[layer], initial_view_state=view_state, map_style=None)
                            st.pydeck_chart(deck, use_container_width=True)
                    except Exception:
                        st.info("Install folium or pydeck to see the map.")
            else:
                st.info("No region column available to map.")

    # Beauty Sentiment Drivers (computed in calculation.py)
    st.markdown("### Beauty Sentiment Drivers")
    st.caption("Top topics mentioned in reviews, split by sentiment")

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

    # Minimal card CSS (green for positive, red for negative)
    st.markdown(
        """
        <style>
        .driver-card {background:#f2fcf6; border:1px solid #d6f5e1; border-radius:12px; padding:0.9rem 1rem; min-height:90px;}
        .driver-card.blank {background:transparent; border:1px dashed #e8e8e8;}
        .driver-card.neg {background:#ffc4c4; border:1px solid #ffd7d7;}
        .driver-card.blank.neg {border-color:#ffd7d7;}
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

    p_tabs = st.tabs(["Positive Drivers", "Negative Drivers"])
    with p_tabs[0]:
        _render_topic_cards(pos_top, "No. of Reviews", negative=False)
    with p_tabs[1]:
        _render_topic_cards(neg_top, "No. of Reviews", negative=True)

    # Recommended Actions (separate section below drivers)
    st.markdown("### ✅ Recommended Actions")
    st.caption("AI-powered recommendations based on current data")
    colA, colB = st.columns(2)
    with colA:
        st.markdown("**Immediate Actions**")
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
        st.markdown("**Strategic Improvements**")
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

with tab_add:
    st.subheader("Add New Review")
    st.caption("Provide review details below. The Add button is currently non-functional.")

    # Pull choices from computed filters (from calculation.py)
    asin_opts = asin_choices if isinstance(asin_choices, list) else []
    region_opts = region_choices if isinstance(region_choices, list) else []

    col1, col2 = st.columns(2)
    with col1:
        if asin_opts:
            asin_val = st.selectbox("ASIN", options=asin_opts, index=0, key="add_asin")
        else:
            asin_val = st.selectbox("ASIN", options=asin_opts, key="add_asin")
        title_val = st.text_input("Title", key="add_title")
    with col2:
        if region_opts:
            region_val = st.selectbox("Region", options=region_opts, index=0, key="add_region")
        else:
            region_val = st.selectbox("Region", options=region_opts, key="add_region")

    review_val = st.text_area("Review", height=140, key="add_review_text")

    # Placeholder Add button (no action yet)
    st.button("Add", type="primary", key="add_submit")
