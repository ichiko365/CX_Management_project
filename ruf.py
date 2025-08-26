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

# Page config
st.set_page_config(page_title="Login Page", page_icon="🔒", layout="wide")

# Read query params for router base and any submitted credentials (new API)
try:
    params = dict(st.query_params)
except Exception:
    params = {}

# Credentials submitted via URL query (GET form)
q_user = params.get("user", "")
q_pass = params.get("pass", "")


BASE_DIR = Path(__file__).resolve().parent
HOST = os.getenv("HUB_HOST", "127.0.0.1")
PYTHON = sys.executable

# Streamlit apps to launch
CUSTOMER_APP_PATH = BASE_DIR / "app" / "app.py"
CLIENT_APP_PATH = BASE_DIR / "src" / "dashboard_project" / "app.py"

# Default UI ports
DEFAULT_CUSTOMER_UI_PORT = int(os.getenv("CUSTOMER_APP_PORT", "8501"))
DEFAULT_CLIENT_UI_PORT = int(os.getenv("CLIENT_APP_PORT", "8502"))


# ---------- Local helpers to manage Streamlit apps (use fixed ports above) ----------
def _is_port_open(host: str, port: int, timeout: float = 0.4) -> bool:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(timeout)
            return s.connect_ex((host, port)) == 0
    except Exception:
        return False


def _st_health(host: str, port: int, timeout: float = 0.7) -> bool:
    try:
        r = requests.get(f"http://{host}:{port}/_stcore/health", timeout=timeout)
        return r.status_code == 200
    except Exception:
        return _is_port_open(host, port)


def _start_streamlit(app_path: Path, ui_port: int, env_overrides: Optional[dict] = None) -> None:
    if not app_path.exists():
        # Surface error in UI but keep layout intact
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
        st.error(f"Failed to start Streamlit app: {e}")


def _poll_until_healthy(port: int, timeout: float = 10.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if _st_health(HOST, port):
            return True
        time.sleep(0.35)
    return _st_health(HOST, port)


def _find_free_port(start: int, end: int) -> Optional[int]:
    for p in range(start, end + 1):
        if not _is_port_open(HOST, p):
            return p
    return None


# -------------------- Choose stable, non-conflicting UI ports --------------------
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


# Background and custom CSS + HTML (form-based)
st.markdown("""
    <style>
        body {
            margin: 0;
            padding: 0;
        }
        .stApp {
            background: url("https://images.unsplash.com/photo-1612817288484-6f916006741a?q=80&w=2070&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D") no-repeat center center fixed;
            background-size: cover;
        }
        /* make the top Streamlit header transparent so wallpaper shows through */
        [data-testid="stHeader"], [data-testid="stToolbar"]{
            background: transparent !important;
            backdrop-filter: none !important;
        }
        /* ensure the main view container doesn't introduce a white background */
        [data-testid="stAppViewContainer"]{ background: transparent !important; }
        /* remove excess top padding to eliminate any visible white band */
        main .block-container{ padding-top: 0.5rem !important; }
        /* Floating animation for subtle motion */
        @keyframes floaty {
            0% { transform: translate(-50%, 0%) translateY(0); }
            50% { transform: translate(-50%, 0%) translateY(-6px); }
            100% { transform: translate(-50%, 0%) translateY(0); }
        }
        .glass-card {
            position: absolute;
            top: 12vh; /* move slightly down while staying centered */
            left: 50%;
            transform: translate(-50%, 0%);
            background: rgba(255, 255, 255, 0.15);
            border-radius: 20px;
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.3);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.25);
            padding: 3rem 2.5rem;
            width: auto;
            height: auto;
            text-align: center;
            animation: floaty 6s ease-in-out infinite;
        }
        .glass-card h2 {
            color: white;
            margin-bottom: 1.5rem;
        }
        .glass-card input, .glass-card button {
            width: 100%;
            padding: 0.8rem;
            margin: 0.6rem 0;
            border-radius: 12px;
            border: none;
            outline: none;
        }
        .glass-card button {
            background: rgba(255, 255, 255, 0.7);
            font-weight: bold;
            cursor: pointer;
            transition: 0.3s;
        }
        .glass-card button:hover {
            background: rgba(255, 255, 255, 0.9);
        }
        .glass-card form{ margin:0; }

        /* inline error message */
        .error-msg{
            margin-top: 0.25rem;
            padding: 8px 12px;
            border-left: 4px solid #e53935;
            background: rgba(229,57,53,0.15);
            color: #FFBDB0;
            font-weight: 600;
            border-radius: 8px;
            text-align: left;
        }

        /* bottom footer links - fixed so it doesn't disturb layout */
        .app-footer{
            position: fixed; left: 50%; bottom: 16px; transform: translateX(-50%);
            display: flex; align-items: center; justify-content: space-evenly; gap: 0; flex-wrap: nowrap;
            width: clamp(480px, 70vw, 700px);
            padding: 10px 22px; border-radius: 999px;
            background: rgba(0,0,0,0.28);
            border: 2px solid rgba(255,255,255,0.25);
            backdrop-filter: blur(8px);
            -webkit-backdrop-filter: blur(8px);
            color: #fff; z-index: 1000;
        }
        .app-footer a{ color:#f3f6ff; text-decoration: none; font-weight: 600; position: relative; }
        .app-footer a:hover{ text-decoration: underline; }
        /* Use CSS-generated separators so spacing stays equal */
        .app-footer .dot{ display:none; }
        .app-footer a:not(:last-child)::after{
            content: "•"; color:#f3f6ff; opacity:.65; margin-left:12px;
        }

        /* company name at top */
        .company-name{
            position: fixed;
            top: 20px;
            left: 50%;
            transform: translateX(-50%);
            color: white;
            font-size: 3rem;
            font-weight: 500;
            letter-spacing: 3px;
            text-shadow: 2px 2px 8px rgba(110,90,0,0.5);
            z-index: 1000;
        }
    </style>
""", unsafe_allow_html=True)

# Determine auth result & potential inline error (no page redirect)
target = None
error_html = ""
if q_user and q_pass:
    if q_pass == "1234":
        if q_user == "Aryan001":
            target = (CUSTOMER_APP_PATH, st.session_state["cust_ui_port"])
        elif q_user == "Nikhil001":
            target = (CLIENT_APP_PATH, st.session_state["client_ui_port"])
    if target is None:
        error_html = '<div class="error-msg">Invalid username or password.</div>'

# Card HTML with optional error message injected (keeps UI intact)
card_html = """
    <div class="company-name">method.</div>
    
    <div class="glass-card">
        <h2>Login (Client / Customer)</h2>
        <form id="login-form" method="get">
            <input name="user" id="username" type="text" placeholder="Username">
            <input name="pass" id="password" type="password" placeholder="Password">
            <button type="submit" id="login-btn">Login</button>
        </form>
        {error_html}
    </div>

    <div class="app-footer">
        <a href="#about">About us</a>
        <a href="#contact">Contact us</a>
        <a href="#privacy">Privacy Policy</a>
        <a href="#terms">Terms of Service</a>
    </div>
""".format(error_html=error_html)

st.markdown(card_html, unsafe_allow_html=True)

# Preserve router query param on submit (no UI change)
# No router param preservation needed; the router base is fixed

# Server-side credential handling: start target app locally, then open in a new tab (no redirect)
if target:
    app_path, port = target
    if not _st_health(HOST, port):
        _start_streamlit(app_path, port)
        _poll_until_healthy(port)
    url = f"http://{HOST}:{port}"
    key = f"{q_user}|{port}"
    if st.session_state.get("_opened_for") != key:
        components.html(
            f"""
            <script>
                window.open('{url}', '_blank');
            </script>
            """,
            height=0,
        )
        st.session_state["_opened_for"] = key
