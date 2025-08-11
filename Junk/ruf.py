import streamlit as st
import pandas as pd
import json
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt

# --- Sidebar: App title & data loading options ---
st.sidebar.title("Amazon Reviews Explorer")

# (Optional future: file uploader)  # uploader = st.sidebar.file_uploader("Upload CSV", type=["csv"])  # if uploader: df = pd.read_csv(uploader)

# Load CSV into a DataFrame (could be replaced by uploaded file)
@st.cache_data(show_spinner=False)
def load_data(path: str):
    return pd.read_csv(path)

df = load_data('amazon_data_v2.csv')

# Ensure reviewTime parsed early for filters
if 'reviewTime' in df.columns:
    df['reviewTime'] = pd.to_datetime(df['reviewTime'], errors='coerce')

# --- Sidebar: Filters ---
with st.sidebar.expander("Filters", expanded=True):
    # Date range
    if 'reviewTime' in df.columns:
        min_date = df['reviewTime'].min()
        max_date = df['reviewTime'].max()
        date_range = st.date_input(
            "Review date range", 
            value=(min_date.date() if pd.notnull(min_date) else None, max_date.date() if pd.notnull(max_date) else None),
            min_value=min_date.date() if pd.notnull(min_date) else None,
            max_value=max_date.date() if pd.notnull(max_date) else None
        )
    else:
        date_range = None

    # Rating range
    if 'overall' in df.columns:
        min_rating = float(df['overall'].min())
        max_rating = float(df['overall'].max())
        rating_range = st.slider("Rating range", min_rating, max_rating, (min_rating, max_rating))
    else:
        rating_range = None

    # Brand multiselect (if brand column exists)
    if 'brand' in df.columns:
        brand_options = sorted([b for b in df['brand'].dropna().unique()][:300])  # limit options
        selected_brands = st.multiselect("Brands", brand_options)
    else:
        selected_brands = []

    # Keyword search (applies to reviewText + summary)
    keyword = st.text_input("Keyword in text/summary")

# --- Sidebar: Display & Smoothing Controls ---
with st.sidebar.expander("Display Options", expanded=True):
    enable_edit = st.toggle("Enable table editing")
    sigma = st.slider("Trend smoothing (Gaussian σ)", 0, 10, 2, help="0 disables smoothing")
    show_raw_points = st.checkbox("Show raw period average points", value=False)
    period_choice = st.radio("Trend period", ["Monthly","Yearly"], horizontal=True)

# Apply filters
filtered = df.copy()
if date_range and isinstance(date_range, (list, tuple)) and len(date_range) == 2 and all(date_range):
    start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1]) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    filtered = filtered[(filtered['reviewTime'] >= start) & (filtered['reviewTime'] <= end)]
if rating_range:
    rmin, rmax = rating_range
    filtered = filtered[(filtered['overall'] >= rmin) & (filtered['overall'] <= rmax)]
if selected_brands:
    filtered = filtered[filtered['brand'].isin(selected_brands)]
if keyword:
    kw = keyword.lower()
    mask = False
    if 'reviewText' in filtered.columns:
        mask = mask | filtered['reviewText'].fillna('').str.lower().str.contains(kw)
    if 'summary' in filtered.columns:
        mask = mask | filtered['summary'].fillna('').str.lower().str.contains(kw)
    filtered = filtered[mask]

# Derive simple category if not present (placeholder) for distribution
if 'category' not in filtered.columns:
    if 'title' in filtered.columns:
        filtered['category'] = filtered['title'].fillna('Unknown').str.split().str[0]
    else:
        filtered['category'] = 'Unknown'

# --- Custom CSS for dashboard style ---
st.markdown(
    """
    <style>
    .metric-grid {display:grid; gap:1rem; grid-template-columns:repeat(auto-fit,minmax(180px,1fr));}
    .card {background:#ffffff; border:1px solid #eee; border-radius:14px; padding:1rem 1.1rem; box-shadow:0 2px 4px rgba(0,0,0,0.04);} 
    .card.soft {background:linear-gradient(135deg,#fff,#f9f6ff);}
    .kpi-value {font-size:1.9rem; font-weight:600; margin:0;}
    .kpi-sub {font-size:0.75rem; color:#088a2d; font-weight:500; margin-top:2px;}
    .kpi-badge {background:#f1f5f9; padding:2px 7px; border-radius:20px; font-size:0.65rem; margin-left:6px;}
    .topic-grid {display:grid; gap:1rem; grid-template-columns:repeat(auto-fit,minmax(230px,1fr));}
    .topic-card {background:#fff; border:1px solid #eee; border-radius:12px; padding:0.8rem 0.9rem;} 
    .topic-score {font-size:1.3rem; font-weight:600; margin:0.3rem 0;}
    .good {color:#0c8f3d;} .warn {color:#c88700;} .trend-up {color:#0c8f3d;} .trend-down {color:#c22727;}
    .driver-tabs button {margin-right:4px;}
    .driver-grid {display:grid; gap:1rem; grid-template-columns:repeat(auto-fit,minmax(260px,1fr));}
    .driver-card {background:#f2fcf6; border:1px solid #d6f5e1; border-radius:12px; padding:0.9rem 1rem;}
    .driver-title {font-weight:600; margin-bottom:0.4rem;}
    .rec-section {display:grid; gap:1rem; grid-template-columns:repeat(auto-fit,minmax(300px,1fr));}
    .pill {display:inline-block; background:#f1f5f9; padding:2px 8px; border-radius:16px; font-size:0.65rem; margin:2px 4px 0 0;}
    .issue-col {background:#fff; border:1px solid #eee; border-radius:12px; padding:1rem;}
    .issue-item {border-bottom:1px solid #eee; padding:0.6rem 0;}
    .issue-item:last-child {border-bottom:none;}
    .severity-critical {background:#ffebeb; color:#d60000; padding:2px 8px; border-radius:14px; font-size:0.65rem;}
    .severity-high {background:#fff4e5; color:#b96a00; padding:2px 8px; border-radius:14px; font-size:0.65rem;}
    .team-card {background:#fff; border:1px solid #eee; border-radius:12px; padding:0.9rem 1rem; margin-bottom:0.8rem;}
    .progress-bar {height:6px; background:#eee; border-radius:4px; position:relative; margin:6px 0 4px;}
    .progress-fill {height:100%; background:#222; border-radius:4px;}
    </style>
    """,
    unsafe_allow_html=True
)

# --- High level KPIs ---
avg_rating = filtered['overall'].mean() if not filtered.empty else 0
sentiment_score = round((avg_rating/5)*10,1)
volume = len(filtered)
prev_volume = max(volume - int(volume*0.111), 1)  # placeholder for delta
volume_delta = (volume - prev_volume)/prev_volume*100 if prev_volume else 0
urgent_issues_placeholder = {"critical":1, "high":2}  # placeholders
team_utilization = 0.78  # placeholder
team_delta = 5  # +5% placeholder
sentiment_delta = 0.3  # placeholder

# Tabs: Dashboard & Data View
dash_tab, data_tab = st.tabs(["📊 Dashboard","🗂 Data View"])

with dash_tab:
    st.markdown("<h1 style='margin-bottom:0.2rem; background:linear-gradient(90deg,#d800b9,#6a00ff); -webkit-background-clip:text; color:transparent;'>Beauty CX Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("<p style='margin-top:0'>Comprehensive beauty product experience analytics</p>", unsafe_allow_html=True)

    # KPI Cards
    st.markdown(f"""
    <div class='metric-grid'>
      <div class='card soft'>
        <div>Sentiment Score <span class='kpi-badge'>/10</span></div>
        <p class='kpi-value'>{sentiment_score}</p>
        <div class='kpi-sub'>↑ +{sentiment_delta} from last month</div>
      </div>
      <div class='card soft'>
        <div>Review Volume</div>
        <p class='kpi-value'>{volume:,}</p>
        <div class='kpi-sub'>↑ {volume_delta:.1f}% from last month</div>
      </div>
      <div class='card soft'>
        <div>Urgent Issues</div>
        <p class='kpi-value'>{urgent_issues_placeholder['critical'] + urgent_issues_placeholder['high']}</p>
        <div class='kpi-sub'>{urgent_issues_placeholder['critical']} critical, {urgent_issues_placeholder['high']} high</div>
      </div>
      <div class='card soft'>
        <div>Team Utilization</div>
        <p class='kpi-value'>{int(team_utilization*100)}%</p>
        <div class='kpi-sub'>↑ +{team_delta}% from last week</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Trend Area Chart
    st.markdown("### 📈 Historical Sentiment Trend")
    st.caption("Beauty product sentiment over time")
    if 'reviewTime' in filtered.columns and filtered['reviewTime'].notna().sum()>0:
        trend_df = filtered.dropna(subset=['reviewTime']).copy()
        if period_choice == 'Monthly':
            grp = trend_df['reviewTime'].dt.to_period('M')
        else:
            grp = trend_df['reviewTime'].dt.to_period('Y')
        trend_df = trend_df.groupby(grp)['overall'].mean().reset_index()
        trend_df['period'] = trend_df['reviewTime'].dt.to_timestamp()
        y = trend_df['overall'].values
        if sigma>0 and len(y)>1:
            try: smoothed = gaussian_filter1d(y, sigma=sigma)
            except Exception: smoothed = y
        else: smoothed = y
        fig, ax = plt.subplots(figsize=(6.5,3.2))
        ax.fill_between(trend_df['period'], smoothed, color='#ff5bbd22')
        ax.plot(trend_df['period'], smoothed, color='#ff2d9b', linewidth=2)
        ax.set_ylim(0,5)
        ax.set_ylabel('Avg Rating')
        ax.set_xlabel('Date')
        ax.grid(alpha=0.15)
        st.pyplot(fig)
    else:
        st.info("No date data.")

    # Category Distribution (bar)
    st.markdown("### 📦 Category Distribution")
    cat_counts = filtered['category'].value_counts().head(12)
    if not cat_counts.empty:
        fig2, ax2 = plt.subplots(figsize=(5.5,3.2))
        ax2.bar(cat_counts.index, cat_counts.values, color='#6a00ff66')
        ax2.set_ylabel('Reviews')
        ax2.set_xticklabels(cat_counts.index, rotation=40, ha='right', fontsize=8)
        st.pyplot(fig2)
    else:
        st.info("No categories available.")

    # Topic Sentiment Analysis (placeholder heuristics)
    st.markdown("### 💠 Beauty Topic Sentiment Analysis")
    topics = [
        ('Color Match','color'),
        ('Skin Compatibility','skin'),
        ('Longevity/Wear Time','long'),
        ('Packaging Quality','pack'),
        ('Delivery Speed','deliver'),
        ('Value for Money','value')
    ]
    topic_cards = []
    base_mean = filtered['overall'].mean() if not filtered.empty else 0
    for label, kw in topics:
        if 'reviewText' in filtered.columns:
            subset = filtered[filtered['reviewText'].str.contains(kw, case=False, na=False)]
        else:
            subset = pd.DataFrame()
        score = subset['overall'].mean() if not subset.empty else base_mean
        count = len(subset)
        quality_tag = 'Excellent' if score>=4.2 else ('Good' if score>=3.5 else 'Fair')
        topic_cards.append({"label":label,"score":round((score/5)*10,1),"count":count,"tag":quality_tag})
    html_topics = "<div class='topic-grid'>" + "".join([
        f"""<div class='topic-card'><div>{c['label']}</div><div class='topic-score'>{c['score']}/10</div><div style='font-size:0.7rem;'>{c['count']} reviews</div><span class='kpi-badge'>{c['tag']}</span></div>"""
        for c in topic_cards]) + "</div>"
    st.markdown(html_topics, unsafe_allow_html=True)

    # Sentiment Drivers (placeholders)
    st.markdown("### 📈 Beauty Sentiment Drivers")
    driver_tabs = st.tabs(["Positive Drivers","Negative Drivers","Trending Topics"])
    pos = [
        {"title":"Perfect Color Match","impact":+2.8,"mentions":89,"pct":34},
        {"title":"Long-Lasting Formula","impact":+2.4,"mentions":76,"pct":29},
        {"title":"Gentle on Sensitive Skin","impact":+2.1,"mentions":68,"pct":26},
        {"title":"Beautiful Packaging","impact":+1.8,"mentions":54,"pct":21},
        {"title":"Great Coverage","impact":+1.6,"mentions":45,"pct":17},
    ]
    neg = [
        {"title":"Allergic Reactions","impact":-2.2,"mentions":14,"pct":5},
        {"title":"Wrong Shade","impact":-1.7,"mentions":19,"pct":7},
        {"title":"Drying Texture","impact":-1.5,"mentions":22,"pct":8},
    ]
    trend = [
        {"title":"Longevity Requests","impact":+0.9,"mentions":31,"pct":11},
        {"title":"Shade Expansion","impact":+0.7,"mentions":27,"pct":10},
    ]
    def render_driver(items):
        return ("<div class='driver-grid'>" + "".join([
            f"""<div class='driver-card'><div class='driver-title'>{d['title']}</div>
            <div style='font-size:0.7rem;'>Impact Score <span style='float:right;font-weight:600;color:{'#0c8f3d' if d['impact']>0 else '#c22727'}'>{d['impact']:+.1f}</span></div>
            <div style='font-size:0.7rem;'>Mentions <span style='float:right'>{d['mentions']}</span></div>
            <div style='font-size:0.7rem;'>In Reviews <span style='float:right'>{d['pct']}%</span></div></div>"""
        for d in items]) + "</div>")
    with driver_tabs[0]:
        st.markdown(render_driver(pos), unsafe_allow_html=True)
    with driver_tabs[1]:
        st.markdown(render_driver(neg), unsafe_allow_html=True)
    with driver_tabs[2]:
        st.markdown(render_driver(trend), unsafe_allow_html=True)

    # Recommended Actions (placeholders)
    st.markdown("### ✅ Recommended Actions")
    colA, colB = st.columns(2)
    with colA:
        st.markdown("**Immediate Actions**")
        st.markdown("- Address Allergic Reactions — contact affected customers\n- Improve Color Matching — enhance shade descriptions")
    with colB:
        st.markdown("**Strategic Improvements**")
        st.markdown("- Expand Shade Range — inclusive options\n- Highlight Longevity — emphasize marketing claims")

    # Two-column lower section: Issues Queue & Team Performance
    st.markdown("### 🚨 Operational Overview")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Urgent Beauty Issues Queue**")
        st.caption("Critical beauty product issues requiring immediate attention")
        issues = [
            {"customer":"Sarah M.","asin":"B08XYZ123","summary":"Severe allergic reaction - skin rash and irritation","severity":"Critical","tags":["allergic reaction","rash","irritation"],"ago":"45 min"},
            {"customer":"Jessica L.","asin":"B09ABC456","summary":"Wrong shade delivered - completely different color","severity":"High","tags":["wrong shade","color mismatch"],"ago":"2h"},
            {"customer":"Amanda K.","asin":"B07DEF789","summary":"Product expired before delivery date","severity":"High","tags":["expired","quality issue"],"ago":"4h"},
        ]
        for iss in issues:
            sev_class = 'severity-critical' if iss['severity']=="Critical" else 'severity-high'
            st.markdown(f"""
            <div class='issue-item'>
              <div style='display:flex; justify-content:space-between;'><strong>{iss['customer']}</strong><span class='{sev_class}'>{iss['severity']}</span></div>
              <div style='font-size:0.65rem; opacity:0.8;'>{iss['asin']} • {iss['ago']} ago</div>
              <div style='font-size:0.75rem; margin-top:4px;'>{iss['summary']}</div>
              <div>{''.join(f'<span class="pill">{t}</span>' for t in iss['tags'])}</div>
            </div>
            """, unsafe_allow_html=True)
    with col2:
        st.markdown("**Beauty Team Performance**")
        st.caption("Specialized beauty consultant performance metrics")
        team = [
            {"name":"Dr. Emily Chen","role":"Senior Beauty Consultant • Skin Issues","tasks_done":12,"tasks_total":15,"resp_time":"1.2h","rating":4.9,"skills":["Dermatology","Cosmetic Chemistry"]},
            {"name":"Mike Rodriguez","role":"Color Specialist • Color Matching","tasks_done":9,"tasks_total":10,"resp_time":"2.1h","rating":4.8,"skills":["Color Theory","Makeup Artistry"]},
            {"name":"Lisa Park","role":"Product Quality Expert • Quality Control","tasks_done":8,"tasks_total":12,"resp_time":"2.8h","rating":4.6,"skills":["Stability","QA"]},
        ]
        for m in team:
            pct = int(m['tasks_done']/m['tasks_total']*100)
            st.markdown(f"""
            <div class='team-card'>
              <div><strong>{m['name']}</strong></div>
              <div style='font-size:0.65rem; opacity:0.75;'>{m['role']}</div>
              <div style='display:flex; justify-content:space-between; font-size:0.7rem; margin-top:6px;'>
                <div>Tasks {m['tasks_done']}/{m['tasks_total']}</div><div>Response Time {m['resp_time']}</div></div>
              <div class='progress-bar'><div class='progress-fill' style='width:{pct}%;'></div></div>
              <div style='font-size:0.65rem;'>Completion Rate {pct}% • ⭐ {m['rating']}</div>
              <div style='margin-top:4px;'>{''.join(f'<span class="pill">{s}</span>' for s in m['skills'])}</div>
            </div>
            """, unsafe_allow_html=True)

with data_tab:
    st.subheader("Interactive Table View")
    # Column configuration (fix 'verified' typo)
    config = {
        "imageURLHighRes": st.column_config.ImageColumn(label="Image"),
        "verified": st.column_config.CheckboxColumn(label="Verified"),
        "overall": st.column_config.NumberColumn(label="Rating", help="1–5 stars"),
        # "reviewTime": st.column_config.DateColumn(label="Review Date"),
        "reviewText": st.column_config.TextColumn(label="Review Text"),
    }
    view_df = filtered.copy()
    if enable_edit:
        st.data_editor(view_df, hide_index=True, column_config=config, use_container_width=True, height=500)
    else:
        st.dataframe(view_df, hide_index=True, column_config=config, use_container_width=True, height=500)
    st.caption(f"Rows: {len(view_df):,} (Filtered from {len(df):,})")

    st.subheader("📈 Sentiment Trend (Detailed)")
    if 'reviewTime' in filtered.columns and filtered['reviewTime'].notna().sum()>0:
        trend_df2 = filtered.dropna(subset=['reviewTime']).copy()
        if period_choice == 'Monthly':
            grp = trend_df2['reviewTime'].dt.to_period('M')
        else:
            grp = trend_df2['reviewTime'].dt.to_period('Y')
        trend_df2 = trend_df2.groupby(grp)['overall'].mean().reset_index()
        trend_df2['period'] = trend_df2['reviewTime'].dt.to_timestamp()
        y2 = trend_df2['overall'].values
        if sigma>0 and len(y2)>1:
            try: smoothed2 = gaussian_filter1d(y2, sigma=sigma)
            except Exception: smoothed2 = y2
        else: smoothed2 = y2
        figd, axd = plt.subplots()
        axd.plot(trend_df2['period'], smoothed2, color='blue', label='Smoothed Avg')
        if show_raw_points:
            axd.scatter(trend_df2['period'], y2, color='gray', alpha=0.5, s=20, label='Raw Avg')
        axd.set_xlabel('Date')
        axd.set_ylabel('Avg Rating')
        axd.grid(alpha=0.3)
        axd.legend()
        st.pyplot(figd)
    else:
        st.info("No valid dates available to plot.")

# --- Optional Download ---
with st.sidebar.expander("Export"):
    csv_bytes = filtered.to_csv(index=False).encode('utf-8')
    st.download_button("Download filtered CSV", data=csv_bytes, file_name="filtered_reviews.csv", mime="text/csv")

# Footer
st.sidebar.markdown("---")
st.sidebar.caption("Built with Streamlit • Filters + Trend Visualization + Dashboard (placeholders for customization)")