import os
import sys
import time
import socket
import shutil
import subprocess
from pathlib import Path

import requests
import streamlit as st
import streamlit.components.v1 as components
from urllib.parse import urlsplit


# Optional extras (graceful fallback if not installed)
try:
    from streamlit_extras.stylable_container import stylable_container
    from streamlit_extras.metric_cards import style_metric_cards
    from streamlit_extras.badges import badge
    from streamlit_extras.let_it_rain import rain
    EXTRAS = True
except Exception:
    EXTRAS = False
    from contextlib import contextmanager

    @contextmanager
    def stylable_container(key: str, **kwargs):
        yield

    def style_metric_cards(**kwargs):
        return None

    def badge(*args, **kwargs):
        return None

    def rain(*args, **kwargs):
        return None


# ----------------------- Page setup -----------------------
st.set_page_config(page_title="CX Hub (Streamlit)", page_icon="🧭", layout="wide")

bg_start = "#f6f7ff"
bg_end = "#fff0f6"
st.markdown(
    f"""
    <style>
    [data-testid=\"stAppViewContainer\"] {{
        background: linear-gradient(180deg, {bg_start} 0%, {bg_end} 100%) !important;
    }}
    [data-testid=\"stHeader\"] {{ background: transparent; }}
    .card {{ background:#fff; border:1px solid #eee; border-radius:14px; padding:1rem 1.2rem; box-shadow:0 2px 4px rgba(0,0,0,0.04); }}
    .soft {{ background:linear-gradient(135deg,#fff,#f9f6ff); }}
    .status-badge {{ display:inline-block; padding:2px 8px; border-radius:999px; font-size:12px; font-weight:600; margin-top:4px; }}
    .status-badge.ok {{ background:#e8fff3; color:#0c8f3d; border:1px solid #b4f0cd; }}
    .status-badge.stop {{ background:#ffe8e8; color:#c22727; border:1px solid #ffc2c2; }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------------- Config -----------------------
BASE_DIR = Path(__file__).resolve().parent
PYTHON = sys.executable
HOST = os.getenv("ROUTER_HOST", "127.0.0.1")

# Default UI ports (we'll adapt if busy)
DEFAULT_CUSTOMER_UI_PORT = int(os.getenv("CUSTOMER_APP_PORT", "8501"))
DEFAULT_CLIENT_UI_PORT = int(os.getenv("CLIENT_APP_PORT", "8502"))

# App paths
CUSTOMER_APP_PATH = BASE_DIR / "app" / "app.py"
CLIENT_APP_PATH = BASE_DIR / "src" / "dashboard_project" / "app.py"

# Optional existing FastAPI router to piggyback on
ROUTER_PORT = int(os.getenv("ROUTER_PORT", "8000"))
ROUTER_URL = f"http://{HOST}:{ROUTER_PORT}"


def _router_alive() -> bool:
    try:
        r = requests.get(f"{ROUTER_URL}/healthz", timeout=0.6)
        return r.status_code == 200
    except Exception:
        return False


def _is_port_open(host: str, port: int, timeout: float = 0.35) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(timeout)
        return s.connect_ex((host, port)) == 0


def _st_health(host: str, port: int) -> bool:
    try:
        r = requests.get(f"http://{host}:{port}/_stcore/health", timeout=0.5)
        return r.status_code == 200
    except Exception:
        return _is_port_open(host, port)


def _find_free_port(start: int, end: int) -> int | None:
    for p in range(start, end + 1):
        if not _is_port_open(HOST, p):
            return p
    return None


def _start_streamlit(app_path: Path, ui_port: int, env_overrides: dict | None = None) -> None:
    if not app_path.exists():
        st.error(f"App not found: {app_path}")
        return
    cmd = [
        PYTHON,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--server.port",
        str(ui_port),
        "--server.address",
        HOST,
        "--browser.gatherUsageStats",
        "false",
    ]
    env = os.environ.copy()
    if env_overrides:
        env.update({k: str(v) for k, v in env_overrides.items()})
    try:
        subprocess.Popen(
            cmd,
            cwd=str(app_path.parent),
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception as e:
        st.error(f"Failed to start Streamlit: {e}")


# Persist chosen ports across reruns
if "cust_ui_port" not in st.session_state:
    st.session_state["cust_ui_port"] = (
        DEFAULT_CUSTOMER_UI_PORT
        if not _is_port_open(HOST, DEFAULT_CUSTOMER_UI_PORT)
        else (_find_free_port(DEFAULT_CUSTOMER_UI_PORT + 2, DEFAULT_CUSTOMER_UI_PORT + 40) or DEFAULT_CUSTOMER_UI_PORT)
    )
if "client_ui_port" not in st.session_state:
    st.session_state["client_ui_port"] = (
        DEFAULT_CLIENT_UI_PORT
        if not _is_port_open(HOST, DEFAULT_CLIENT_UI_PORT)
        else (_find_free_port(DEFAULT_CLIENT_UI_PORT + 2, DEFAULT_CLIENT_UI_PORT + 40) or DEFAULT_CLIENT_UI_PORT)
    )

# Ensure distinct ports (avoid accidental collisions if env defaults are equal)
if st.session_state["client_ui_port"] == st.session_state["cust_ui_port"]:
    # Prefer shifting client port to the next available
    start_from = max(st.session_state["client_ui_port"], DEFAULT_CLIENT_UI_PORT) + 1
    new_client = _find_free_port(start_from, start_from + 50) or _find_free_port(8500, 8700) or (st.session_state["cust_ui_port"] + 1)
    st.session_state["client_ui_port"] = new_client


st.markdown(
    """
    <h1 style="margin:0 0 .25rem 0; background:linear-gradient(90deg,#6a00ff,#d800b9); -webkit-background-clip:text; background-clip:text; color:transparent;">
      CX Hub
    </h1>
    <div style="opacity:.85">Start, embed, or open the Customer app and Client dashboard.</div>
    """,
    unsafe_allow_html=True,
)


def app_card(title: str, desc: str, emoji: str, url: str, healthy: bool, start_cb, embed_key: str):
    with stylable_container(
        f"card-{embed_key}",
        css_styles="""
            {
                background: white; border:1px solid #eee; border-radius:14px; padding: 1.0rem 1.2rem;
                box-shadow: 0 2px 4px rgba(0,0,0,0.04);
            }
        """,
    ):
        c1, c2 = st.columns([3, 2], vertical_alignment="center")
        with c1:
            st.markdown(f"### {emoji} {title}")
            st.caption(desc)
            if EXTRAS:
                lab = "Running" if healthy else "Stopped"
                cls = "ok" if healthy else "stop"
                st.markdown(f"<span class='status-badge {cls}'>{lab}</span>", unsafe_allow_html=True)
            else:
                st.write("Status:", "🟢" if healthy else "🔴")
        with c2:
            st.link_button("Open in new tab", url, type="primary", disabled=not healthy)
            start = st.button("Start / Restart", key=f"start-{embed_key}")
            if start:
                start_cb()
                # Immediately poll the specific app port for readiness so the status updates now
                try:
                    parsed = urlsplit(url)
                    port = parsed.port or (443 if parsed.scheme == "https" else 80)
                except Exception:
                    port = None
                if port:
                    with st.spinner("Starting..."):
                        deadline = time.time() + 10.0
                        while time.time() < deadline:
                            time.sleep(0.35)
                            if _st_health(HOST, int(port)):
                                break
                st.rerun()

        with st.expander("Embed preview"):
            if healthy:
                components.iframe(url, height=740, width=1280, scrolling=True)
            else:
                st.info("App not running. Click Start first.")


def start_customer():
    ui_port = st.session_state["cust_ui_port"]
    # Avoid backend collision with any FastAPI (e.g., router on 8000)
    be_port = _find_free_port(8100, 8300) or 8100
    if _router_alive():
        try:
            requests.get(f"{ROUTER_URL}/go/customer", timeout=1.2)
            return
        except Exception:
            pass
    _start_streamlit(
        CUSTOMER_APP_PATH,
        ui_port,
        env_overrides={"BACKEND_HOST": HOST, "BACKEND_PORT": be_port},
    )


def start_client():
    ui_port = st.session_state["client_ui_port"]
    if _router_alive():
        try:
            requests.get(f"{ROUTER_URL}/go/client", timeout=1.2)
            return
        except Exception:
            pass
    _start_streamlit(CLIENT_APP_PATH, ui_port)


# ----------------------- Layout -----------------------
cust_url = f"http://{HOST}:{st.session_state['cust_ui_port']}"
client_url = f"http://{HOST}:{st.session_state['client_ui_port']}"

cust_ok = _st_health(HOST, st.session_state["cust_ui_port"])
client_ok = _st_health(HOST, st.session_state["client_ui_port"])

if EXTRAS and (cust_ok or client_ok):
    rain(
        emoji="✨",
        font_size=16,
        falling_speed=3,
        animation_length=1,
    )

left, right = st.columns(2)
with left:
    app_card(
        "Customer App",
        "Submit reviews and trigger pipeline.",
        "🧑‍💼",
        cust_url,
        cust_ok,
        start_customer,
        "customer",
    )
with right:
    app_card(
        "Client Dashboard",
        "Explore KPIs and insights.",
        "📊",
        client_url,
        client_ok,
        start_client,
        "client",
    )


# ----------------------- System status -----------------------
st.markdown("---")
st.caption("System status")
colA, colB, colC, colD = st.columns(4)
colA.metric("Router alive", "Yes" if _router_alive() else "No")
colB.metric("Customer UI port", st.session_state["cust_ui_port"]) 
colC.metric("Client UI port", st.session_state["client_ui_port"]) 
colD.metric("Extras", "On" if EXTRAS else "Off")

if EXTRAS:
    style_metric_cards(border_left_color="#6a00ff")

st.info(
    "Tip: This Streamlit hub can start apps directly or via your FastAPI router if it's running."
)
