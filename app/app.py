import os
import sys
import time
import json
import socket
import subprocess
from pathlib import Path

import requests
import streamlit as st


# ----------------------- Page setup -----------------------
st.set_page_config(page_title="CX Management", page_icon="💬", layout="wide")


# ----------------------- Config ---------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
BACKEND_HOST = os.getenv("BACKEND_HOST", "127.0.0.1")
_default_port = int(os.getenv("BACKEND_PORT", "8000"))
# Keep backend port stable across reruns in the same Streamlit session
if "backend_port" in st.session_state:
	BACKEND_PORT = int(st.session_state["backend_port"])  # type: ignore
else:
	BACKEND_PORT = _default_port
BACKEND_URL = f"http://{BACKEND_HOST}:{BACKEND_PORT}"
UVICORN_APP = "app.main:app"  # requires app/__init__.py


def _is_port_open(host: str, port: int) -> bool:
	"""Quickly check if a TCP port is in use."""
	with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
		sock.settimeout(0.5)
		return sock.connect_ex((host, port)) == 0


def _is_backend_healthy_for(host: str, port: int) -> bool:
	try:
		r = requests.get(f"http://{host}:{port}/openapi.json", timeout=0.8)
		return r.status_code == 200
	except Exception:
		return False


def _is_backend_healthy() -> bool:
	return _is_backend_healthy_for(BACKEND_HOST, BACKEND_PORT)


def _find_free_port(start: int = 8001, end: int = 8100) -> int | None:
	for p in range(start, end + 1):
		if not _is_port_open(BACKEND_HOST, p):
			return p
	return None


def _start_backend_if_needed():
	"""Start the FastAPI backend via uvicorn if it's not already running.

	We keep a handle in st.session_state to avoid duplicate processes on reruns.
	"""
	global BACKEND_PORT, BACKEND_URL
	if _is_backend_healthy():
		return True

	# Only spawn once per Streamlit session
	if st.session_state.get("_backend_started"):
		# If we started it but it's not healthy yet, give it a moment
		for _ in range(12):  # ~3s
			if _is_backend_healthy():
				return True
			time.sleep(0.25)
		# Not healthy yet; clear flag so we can retry spawning on next run
		st.session_state["_backend_started"] = False
		return _is_port_open(BACKEND_HOST, BACKEND_PORT)

	# If the preferred port is already in use but not healthy, pick a free port and update globals
	if _is_port_open(BACKEND_HOST, BACKEND_PORT):
		alt = _find_free_port(start=BACKEND_PORT + 1)
		if alt:
			BACKEND_PORT = alt
			BACKEND_URL = f"http://{BACKEND_HOST}:{BACKEND_PORT}"
			st.session_state["backend_port"] = BACKEND_PORT
			st.info(f"Port in use; starting API on {BACKEND_PORT}.")

	# Prefer running from project root so relative imports & .env work
	env = os.environ.copy()
	# Ensure project root is importable for packages like `src` and `app`
	existing_pp = env.get("PYTHONPATH", "")
	env["PYTHONPATH"] = (
		f"{PROJECT_ROOT}{os.pathsep}{existing_pp}" if existing_pp else str(PROJECT_ROOT)
	)
	# Use the same Python interpreter running Streamlit
	python_exec = sys.executable
	cmd = [
		python_exec,
		"-m",
		"uvicorn",
		UVICORN_APP,
		"--host",
		BACKEND_HOST,
		"--port",
	str(BACKEND_PORT),
		"--reload",
	]

	# Direct backend logs to a file for debugging
	backend_log = None
	try:
		logs_dir = PROJECT_ROOT / "logs"
		logs_dir.mkdir(exist_ok=True)
		backend_log = open(logs_dir / "backend.log", "a", buffering=1)
	except Exception:
		backend_log = None

	try:
		subprocess.Popen(
			cmd,
			cwd=str(PROJECT_ROOT),
			env=env,
			stdout=(backend_log if backend_log else subprocess.DEVNULL),
			stderr=(backend_log if backend_log else subprocess.DEVNULL),
		)
		# We'll only mark as started once we see health/port
	except Exception as e:
		st.error(f"Failed to start backend server: {e}")
		return False

	# Wait briefly for startup
	for _ in range(16):  # up to ~4s
		if _is_backend_healthy():
			st.session_state["_backend_started"] = True
			return True
		time.sleep(0.25)
	# Final quick port check; if open, mark started and return
	if _is_port_open(BACKEND_HOST, BACKEND_PORT):
		st.session_state["_backend_started"] = True
		return True
	# Backend didn't come up; allow retry next rerun
	st.session_state["_backend_started"] = False
	return False


# -------------------- Start backend -----------------------
backend_ok = _start_backend_if_needed()
if not backend_ok:
	st.warning(
		"Backend API isn't reachable yet. You can still fill the form; submission will try to connect."
	)


# ----------------------- Header ---------------------------
st.title("CX Management")

left, right = st.columns([1, 1])
with left:
	st.image(
		"https://res.retailershakti.com/incom/images/product/Lakme-Face-Sheer-Sun-Kissed-1602502094-10039240-2.jpg",
		caption="Luminous Glow Serum",
		use_container_width=True,
	)
with right:
	st.video("https://youtu.be/AvpPocRT3JU?si=HujZcx88Qm-Fs-f2")

st.markdown("---")


# -------------------- Add Review Form ---------------------
st.subheader("Add Customer Review")
st.caption("Tip: Only ASIN and Review are required. Title and Description will auto-fill from the DB when possible.")

with st.form("add_review_form", clear_on_submit=True):
	asin = st.text_input("ASIN", placeholder="B002K6AHQY")
	cols = st.columns([1, 1])
	with cols[0]:
		rating = st.slider("Ratings", 1, 5, 4)
	with cols[1]:
		st.number_input("Numeric Rating", min_value=1, max_value=5, value=4)

	region = st.selectbox(
		"Region",
		[
			"Delhi",
			"Mumbai",
			"Bangalore",
			"Chennai",
			"Kolkata",
			"Other",
		],
		index=0,
	)
	title = st.text_input("Review Title (optional)", placeholder="Great Product!")
	description = st.text_area("Product Description (optional)", placeholder="Will auto-fill if known for this ASIN", height=80)
	review_text = st.text_area("Review", placeholder="Tell us what you think…", height=120)

	submitted = st.form_submit_button("Submit Review")

if submitted:
	payload = {
		"ASIN": asin.strip(),
		# Title/Description optional; backend/schema will auto-fill if missing based on ASIN
		"Title": title.strip() if title else None,
		"Description": description.strip() if description else None,
		"Review": review_text.strip(),
		"Region": region,
	}

	# Basic validation aligned with ReviewInput schema
	missing = [k for k in ("ASIN", "Review") if not payload[k]]
	if missing:
		st.error(f"Missing required fields: {', '.join(missing)}")
	else:
		try:
			# Use the current BACKEND_URL (may be dynamically updated if the default port was busy)
			res = requests.post(f"{BACKEND_URL}/add_review/", json=payload, timeout=8)
			if res.status_code == 201:
				data = res.json()
				st.success(f"Review submitted! DB id: {data.get('id')}")
				with st.expander("Response data"):
					st.json(data)
				# Note: 'rating' is collected for UX but not stored by current API
			else:
				try:
					detail = res.json()
				except Exception:
					detail = res.text
				st.error(f"Failed to submit review ({res.status_code}): {detail}")
		except Exception as e:
			st.error(f"Error contacting backend: {e}")


# -------------------- Simple Status Box -------------------
st.markdown("---")
st.caption("System status")
status_cols = st.columns(3)
status_cols[0].metric("Backend host", BACKEND_HOST)
status_cols[1].metric("Backend port", BACKEND_PORT)
status_cols[2].metric("API healthy", "✅" if _is_backend_healthy() else "❌")

# Quick controls and diagnostics
ctrl_col, _, _ = st.columns([1, 1, 1])
if ctrl_col.button("Run Pipeline Now"):
	try:
		r = requests.post(f"{BACKEND_URL}/trigger_pipeline", timeout=5)
		if r.status_code == 202:
			st.success("Pipeline queued to run in the background.")
		else:
			st.error(f"Failed to queue pipeline: {r.status_code} {r.text}")
	except Exception as e:
		st.error(f"Error contacting backend: {e}")

with st.expander("View pipeline log"):
	log_path = PROJECT_ROOT / "logs" / "pipeline.log"
	if log_path.exists():
		try:
			content = log_path.read_text(errors="ignore")[-800:]
			st.code(content or "(log is empty)", language="text")
		except Exception as e:
			st.warning(f"Could not read log file: {e}")
	else:
		st.info("No pipeline log found yet. Submit a review or run the pipeline to create it.")

with st.expander("View backend log"):
	be_log_path = PROJECT_ROOT / "logs" / "backend.log"
	if be_log_path.exists():
		try:
			content = be_log_path.read_text(errors="ignore")[-800:]
			st.code(content or "(log is empty)", language="text")
		except Exception as e:
			st.warning(f"Could not read backend log file: {e}")
	else:
		st.info("No backend log yet. It will appear after Streamlit starts the API server.")