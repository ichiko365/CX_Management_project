import os
import sys
import time
import socket
import subprocess
from pathlib import Path
from typing import Optional

import requests
import streamlit as st
import streamlit.components.v1 as components

# Optional extras (graceful fallback)
try:
    from streamlit_extras.stylable_container import stylable_container
    from streamlit_extras.let_it_rain import rain
    EXTRAS = True
except Exception:
    EXTRAS = False
    from contextlib import contextmanager

    def rain(*args, **kwargs):
        return None

    @contextmanager
    def stylable_container(key: str, **kwargs):
        yield


# ----------------------- Page setup -----------------------
st.set_page_config(page_title="CX Hub", page_icon="🔐", layout="wide")

# ----------------------- Paths & Config -------------------
BASE_DIR = Path(__file__).resolve().parent
HOST = os.getenv("HUB_HOST", "127.0.0.1")
PYTHON = sys.executable

# Streamlit apps to launch
CUSTOMER_APP_PATH = BASE_DIR / "app" / "app.py"
CLIENT_APP_PATH = BASE_DIR / "src" / "dashboard_project" / "app.py"

# Default UI ports
DEFAULT_CUSTOMER_UI_PORT = int(os.getenv("CUSTOMER_APP_PORT", "8501"))
DEFAULT_CLIENT_UI_PORT = int(os.getenv("CLIENT_APP_PORT", "8502"))


def _is_port_open(host: str, port: int, timeout: float = 0.35) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(timeout)
        return s.connect_ex((host, port)) == 0


def _st_health(host: str, port: int) -> bool:
    try:
        r = requests.get(f"http://{host}:{port}/_stcore/health", timeout=0.6)
        return r.status_code == 200
    except Exception:
        return _is_port_open(host, port)


def _find_free_port(start: int, end: int) -> Optional[int]:
    for p in range(start, end + 1):
        if not _is_port_open(HOST, p):
            return p
    return None


def _start_streamlit(app_path: Path, ui_port: int, env_overrides: Optional[dict] = None) -> None:
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


# ----------------------- State & Ports --------------------
if "cust_ui_port" not in st.session_state:
    st.session_state["cust_ui_port"] = (
        DEFAULT_CUSTOMER_UI_PORT
        if not _is_port_open(HOST, DEFAULT_CUSTOMER_UI_PORT)
        else (_find_free_port(DEFAULT_CUSTOMER_UI_PORT + 1, DEFAULT_CUSTOMER_UI_PORT + 50) or DEFAULT_CUSTOMER_UI_PORT)
    )
if "client_ui_port" not in st.session_state:
    st.session_state["client_ui_port"] = (
        DEFAULT_CLIENT_UI_PORT
        if not _is_port_open(HOST, DEFAULT_CLIENT_UI_PORT)
        else (_find_free_port(DEFAULT_CLIENT_UI_PORT + 1, DEFAULT_CLIENT_UI_PORT + 50) or DEFAULT_CLIENT_UI_PORT)
    )
# ensure distinct
if st.session_state["client_ui_port"] == st.session_state["cust_ui_port"]:
    upper = max(st.session_state["client_ui_port"], st.session_state["cust_ui_port"]) + 1
    alt = _find_free_port(upper, upper + 60)
    st.session_state["client_ui_port"] = alt or (st.session_state["cust_ui_port"] + 1)

# Current view: login | client | customer
st.session_state.setdefault("view", "login")


# ----------------------- Theming & CSS --------------------
bg_url = "https://images.unsplash.com/photo-1526318472351-c75fcf070305?q=80&w=1600&auto=format&fit=crop"
css = """
    <style>
    body { overflow-x: hidden; }
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(120deg, rgba(0,0,0,0.25), rgba(0,0,0,0.25)),
                    url('REPLACE_BG') center/cover no-repeat fixed !important;
    }
    /* Frosted glass card */
    .glass {
        background: rgba(255,255,255,0.14);
        border: 1px solid rgba(255,255,255,0.25);
        box-shadow: 0 20px 40px rgba(0,0,0,0.25);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border-radius: 18px;
        padding: 304px 26px;
    }
    .center-wrap { display:flex; min-height: 0vh; align-items:center; justify-content:center; }
    .title-grad {
        background: linear-gradient(90deg, #6a00ff, #d800b9);
        -webkit-background-clip: text; background-clip: text; color: transparent;
        font-weight: 800; letter-spacing: .5px;
    }
    .small-link { font-size: 0.9rem; opacity: .95; }
    .pill { display:inline-block; padding:4px 10px; border-radius:999px; background:#ffffff22; border:1px solid #ffffff55; }
    /* Floating animation for subtle motion */
    @keyframes floaty {
      0% { transform: translateY(0); }
      50% { transform: translateY(-6px); }
      100% { transform: translateY(0); }
    }
    .floaty { animation: floaty 6s ease-in-out infinite; }
    /* top-right actions */
    .topbar { position: sticky; top: 0; z-index: 100; display:flex; justify-content:flex-end; padding: 0.4rem 0; }
    </style>
    """.replace("REPLACE_BG", bg_url)
st.markdown(css, unsafe_allow_html=True)


def _poll_until_healthy(port: int, timeout: float = 10.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        time.sleep(0.35)
        if _st_health(HOST, port):
            return True
    return _is_port_open(HOST, port)


def _login_card():
    # Animated confetti when page loads
    if EXTRAS:
        rain(emoji="✨", font_size=18, falling_speed=4, animation_length=1)

    st.markdown("<div class='center-wrap'>", unsafe_allow_html=True)
    with stylable_container("login_glass", css_styles=".glass{width:min(520px,92vw);} "):
        st.markdown("<div class='glass floaty'>", unsafe_allow_html=True)
        st.markdown("<h1 class='title-grad' style='text-align:center; margin:0 0 .25rem;'>Login</h1>", unsafe_allow_html=True)
        st.caption("Sign in to continue to CX Management")

        with st.form("login_form", clear_on_submit=False):
            u = st.text_input("Username", placeholder="Enter username")
            p = st.text_input("Password", type="password", placeholder="Enter password")
            cols = st.columns([1, 1])
            with cols[0]:
                _ = st.checkbox("Remember me", value=False)
            with cols[1]:
                st.markdown("<div style='text-align:right;' class='small-link'><a>Forgot password?</a></div>", unsafe_allow_html=True)
            submit = st.form_submit_button("Login", use_container_width=True, type="primary")

        st.markdown("<div style='display:flex; gap:14px; justify-content:center; margin-top:10px;'>"
                    "<span class='pill'>About us</span>"
                    "<span class='pill'>Features</span>"
                    "<span class='pill'>Contact us</span>"
                    "</div>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    if submit:
        # Validate simple credentials
        ok = False
        target = None
        if u == "Nikhil001" and p == "1234567890":
            ok = True
            target = "client"
        elif u == "Aryan001" and p == "1234567890":
            ok = True
            target = "customer"
        else:
            st.error("Invalid credentials")

        if ok and target:
            # Launch target app (on its own port) if not already running
            if target == "client":
                port = st.session_state["client_ui_port"]
                _start_streamlit(CLIENT_APP_PATH, port)
                _poll_until_healthy(port)
                st.session_state["view"] = "client"
            else:
                port = st.session_state["cust_ui_port"]
                # Give customer app its own internal backend port to avoid collisions
                api_port = _find_free_port(8100, 8300) or 8100
                _start_streamlit(CUSTOMER_APP_PATH, port, env_overrides={"BACKEND_HOST": HOST, "BACKEND_PORT": api_port})
                _poll_until_healthy(port)
                st.session_state["view"] = "customer"
            st.rerun()


def _dashboard_wrapper(title: str, port: int):
    # top-right logout
    st.markdown("<div class='topbar'>", unsafe_allow_html=True)
    _, right = st.columns([6, 1])
    with right:
        if st.button("Logout", type="secondary"):
            st.session_state["view"] = "login"
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(f"<h2 class='title-grad' style='margin:.25rem 0 .5rem'>{title}</h2>", unsafe_allow_html=True)
    # full-width iframe
    url = f"http://{HOST}:{port}"
    # Ensure app is up (if user bookmark jumped here)
    if not _st_health(HOST, port):
        with st.spinner("Starting application…"):
            _poll_until_healthy(port)
    components.iframe(url, height=900)


# ----------------------- Router ---------------------------
view = st.session_state.get("view", "login")
if view == "login":
    _login_card()
elif view == "client":
    _dashboard_wrapper("Client Dashboard", st.session_state["client_ui_port"])
elif view == "customer":
    _dashboard_wrapper("Customer Dashboard", st.session_state["cust_ui_port"])
else:
    st.session_state["view"] = "login"
    st.experimental_rerun()
