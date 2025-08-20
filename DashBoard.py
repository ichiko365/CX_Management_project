import os
import sys
import socket
import shutil
import subprocess
from pathlib import Path
from typing import Optional

import requests
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse, PlainTextResponse


# ----------------------- Config -----------------------
BASE_DIR = Path(__file__).resolve().parent

# Paths to the two Streamlit apps
CUSTOMER_APP_PATH = BASE_DIR / "app" / "app.py"  # customer-facing app
CLIENT_APP_PATH = BASE_DIR / "src" / "dashboard_project" / "app.py"  # client dashboard

# Network settings
HOST = os.getenv("ROUTER_HOST", "127.0.0.1")  # where uvicorn serves this FastAPI
CUSTOMER_PORT = int(os.getenv("CUSTOMER_APP_PORT", "8501"))
CLIENT_PORT = int(os.getenv("CLIENT_APP_PORT", "8502"))

# If set, the router will start Streamlit apps on startup; otherwise, lazy-start on first visit
AUTOSTART = os.getenv("AUTOSTART_STREAMLIT", "false").lower() in {"1", "true", "yes"}

# Prefer the same Python interpreter running this process
PYTHON = sys.executable
STREAMLIT_BIN = shutil.which("streamlit")  # used only for presence check in messages


def _is_port_open(host: str, port: int, timeout: float = 0.4) -> bool:
	with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
		s.settimeout(timeout)
		return s.connect_ex((host, port)) == 0


def _streamlit_health(host: str, port: int, timeout: float = 0.6) -> bool:
	"""Best-effort health probe for Streamlit."""
	try:
		url = f"http://{host}:{port}/_stcore/health"
		r = requests.get(url, timeout=timeout)
		if r.status_code == 200:
			return True
	except Exception:
		pass
	return _is_port_open(host, port)


def _find_free_port(host: str, start: int = 8100, end: int = 8300) -> Optional[int]:
	for p in range(start, end + 1):
		if not _is_port_open(host, p):
			return p
	return None


def _start_streamlit(app_path: Path, port: int, env_overrides: Optional[dict] = None) -> subprocess.Popen:
	"""Start a Streamlit app on the given port using the current Python interpreter."""
	if not app_path.exists():
		raise FileNotFoundError(f"Streamlit app not found: {app_path}")

	# Use python -m streamlit run to ensure current venv is used
	cmd = [
		PYTHON,
		"-m",
		"streamlit",
		"run",
		str(app_path),
		"--server.port",
		str(port),
		"--server.address",
		HOST,
		"--browser.gatherUsageStats",
		"false",
	]

	env = os.environ.copy()
	if env_overrides:
		env.update({k: str(v) for k, v in env_overrides.items()})
	proc = subprocess.Popen(
		cmd,
		cwd=str(app_path.parent),
		env=env,
		stdout=subprocess.DEVNULL,
		stderr=subprocess.DEVNULL,
	)
	return proc


def _public_url(request: Request, port: int) -> str:
	"""Build a redirect URL respectful of proxies (x-forwarded-*)."""
	scheme = request.headers.get("x-forwarded-proto") or request.url.scheme
	host_hdr = request.headers.get("x-forwarded-host")
	if host_hdr:
		host_only = host_hdr.split(",")[0].strip()
	else:
		host_only = request.url.hostname or HOST
	return f"{scheme}://{host_only}:{port}"


# ----------------------- FastAPI app -----------------------
app = FastAPI(title="CX Router", version="1.0.0")


@app.on_event("startup")
async def _maybe_start_streamlit():
	if not AUTOSTART:
		return
	# Start both apps if not already running
	if not _streamlit_health(HOST, CUSTOMER_PORT):
		try:
			# Ensure customer's internal FastAPI backend doesn't collide with this router (often on 8000)
			free_api_port = _find_free_port(HOST, start=8100, end=8300) or 8100
			_start_streamlit(
				CUSTOMER_APP_PATH,
				CUSTOMER_PORT,
				env_overrides={
					# Hint the Streamlit app to use a non-8000 backend port
					"BACKEND_HOST": HOST,
					"BACKEND_PORT": free_api_port,
				},
			)
		except Exception:
			pass
	if not _streamlit_health(HOST, CLIENT_PORT):
		try:
			_start_streamlit(CLIENT_APP_PATH, CLIENT_PORT)
		except Exception:
			pass


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
	"""Landing page with links to Customer and Client apps."""
	customer_url = _public_url(request, CUSTOMER_PORT)
	client_url = _public_url(request, CLIENT_PORT)
	html = f"""
	<html>
	  <head>
		<meta charset='utf-8' />
		<meta name='viewport' content='width=device-width, initial-scale=1' />
		<title>CX Management Router</title>
		<style>
		  body {{ font-family: -apple-system, system-ui, Segoe UI, Roboto, Helvetica, Arial, sans-serif; padding: 2rem; }}
		  .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 1rem; }}
		  a.card {{ display:block; padding:1.1rem 1.2rem; border:1px solid #e6e6e6; border-radius:12px; text-decoration:none; color:#111; transition: box-shadow .2s, border-color .2s; background:#fff; }}
		  a.card:hover {{ box-shadow:0 6px 16px rgba(0,0,0,0.08); border-color:#d0d0d0; }}
		  .tag {{ display:inline-block; background:#f5f7ff; color:#384cff; font-size:12px; padding:.2rem .5rem; border-radius:999px; margin-left:.4rem; }}
		  .hint {{ color:#666; font-size: 13px; margin-top:.5rem; }}
		</style>
	  </head>
	  <body>
		<h1 style="margin-top:0">CX Management</h1>
		<p>Select an application:</p>
		<div class="grid">
		  <a class="card" href="/go/customer">
			<strong>Customer App</strong>
			<span class="tag">Streamlit</span>
			<div class="hint">{customer_url}</div>
		  </a>
		  <a class="card" href="/go/client">
			<strong>Client Dashboard</strong>
			<span class="tag">Streamlit</span>
			<div class="hint">{client_url}</div>
		  </a>
		</div>
		<p class="hint">This page is served by FastAPI. Ports and behavior can be configured with env vars: ROUTER_HOST, CUSTOMER_APP_PORT, CLIENT_APP_PORT, AUTOSTART_STREAMLIT.</p>
	  </body>
	</html>
	"""
	return HTMLResponse(content=html)


@app.get("/go/customer")
async def go_customer(request: Request):
	"""Start (if needed) and redirect to the customer Streamlit app."""
	if not _streamlit_health(HOST, CUSTOMER_PORT):
		try:
			free_api_port = _find_free_port(HOST, start=8100, end=8300) or 8100
			_start_streamlit(
				CUSTOMER_APP_PATH,
				CUSTOMER_PORT,
				env_overrides={
					"BACKEND_HOST": HOST,
					"BACKEND_PORT": free_api_port,
				},
			)
		except FileNotFoundError:
			return PlainTextResponse(
				f"Customer app not found at {CUSTOMER_APP_PATH}", status_code=500
			)
		except Exception as e:
			return PlainTextResponse(f"Failed to start customer app: {e}", status_code=500)

	return RedirectResponse(_public_url(request, CUSTOMER_PORT), status_code=307)


@app.get("/go/client")
async def go_client(request: Request):
	"""Start (if needed) and redirect to the client Streamlit app."""
	if not _streamlit_health(HOST, CLIENT_PORT):
		try:
			_start_streamlit(CLIENT_APP_PATH, CLIENT_PORT)
		except FileNotFoundError:
			return PlainTextResponse(
				f"Client app not found at {CLIENT_APP_PATH}", status_code=500
			)
		except Exception as e:
			return PlainTextResponse(f"Failed to start client app: {e}", status_code=500)

	return RedirectResponse(_public_url(request, CLIENT_PORT), status_code=307)


@app.get("/status")
async def status():
	data = {
		"router": {"host": HOST},
		"customer": {
			"path": str(CUSTOMER_APP_PATH),
			"port": CUSTOMER_PORT,
			"healthy": _streamlit_health(HOST, CUSTOMER_PORT),
		},
		"client": {
			"path": str(CLIENT_APP_PATH),
			"port": CLIENT_PORT,
			"healthy": _streamlit_health(HOST, CLIENT_PORT),
		},
		"streamlit_bin": STREAMLIT_BIN,
	}
	return JSONResponse(data)


# Optional: simple health endpoint for this router
@app.get("/healthz", response_class=PlainTextResponse)
async def healthz():
	return "ok"

