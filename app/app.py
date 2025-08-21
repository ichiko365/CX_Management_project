import os
import sys
import time
import json
import socket
import subprocess
from pathlib import Path

import requests
from urllib.parse import urlparse, parse_qs
import streamlit as st
from typing import Optional, Dict


# ----------------------- Page setup -----------------------
st.set_page_config(page_title="CX Management", page_icon="💬", layout="wide")

# ----------------------- Theme & styles -------------------
st.markdown(
	"""
	<style>
	.stApp {
	  background: linear-gradient(135deg, #eef2ff 0%, #fff0f6 50%, #effdf6 100%);
	}
	.main-card { 
	  background: rgba(255,255,255,0.78);
	  border: 1px solid rgba(0,0,0,0.06);
	  border-radius: 16px; 
	  padding: 18px 18px 20px; 
	  box-shadow: 0 8px 24px rgba(0,0,0,0.08);
	}
	.media-frame { 
	  aspect-ratio: 16/9; 
	  width: 100%; 
	  border-radius: 16px; 
	  overflow: hidden; 
	  box-shadow: 0 8px 24px rgba(0,0,0,0.08);
	  border: 1px solid rgba(0,0,0,0.06);
	  background: #fff;
	}
	.media-frame img { width: 100%; height: 100%; object-fit: contain; object-position: center; display: block; }
	.media-frame iframe { width: 100% !important; height: 100% !important; border: 0; display:block; }
	</style>
	""",
	unsafe_allow_html=True,
)

# Optional: use streamlit-extras if present
try:
	from streamlit_extras.stylable_container import stylable_container
except Exception:
	stylable_container = None

# Optional: star rating component (fallback to slider if unavailable)
try:
	from streamlit_star_rating import st_star_rating  # type: ignore  # pip install streamlit-star-rating
except Exception:
	st_star_rating = None

# Optional: lightweight DB lookup to auto-fill ASIN/Description from Title
try:
	from .db_engine import create_db_engine
	from sqlalchemy import text
except Exception:
	try:
		from db_engine import create_db_engine
		from sqlalchemy import text
	except Exception:
		create_db_engine = None  # type: ignore
		text = None  # type: ignore


def youtube_embed(url: str) -> str:
	"""Return an embeddable YouTube iframe for various URL formats."""
	if not url:
		return ""
	parsed = urlparse(url)
	vid = ""
	if parsed.netloc in {"youtu.be"}:
		vid = parsed.path.lstrip("/")
	elif "youtube.com" in parsed.netloc:
		if parsed.path == "/watch":
			vid = parse_qs(parsed.query).get("v", [""])[0]
		elif parsed.path.startswith("/embed/"):
			vid = parsed.path.split("/")[-1]
	if not vid:
		return ""
	return f"""
	<div class=\"media-frame\">
	  <iframe src=\"https://www.youtube.com/embed/{vid}?rel=0\" allowfullscreen></iframe>
	</div>
	"""


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


def _lookup_asin_description_by_title(title: str) -> Optional[Dict[str, Optional[str]]]:
	"""Best-effort lookup of ASIN and Description for a given Title.

	Returns dict with keys {"ASIN", "Description"} when found, else None.
	Uses latest matching record by id.
	"""
	if not title or not create_db_engine or not text:
		return None
	try:
		eng = create_db_engine("DB_NAME")
		if eng is None:
			return None
		sql = text(
			"""
			SELECT "ASIN", "Description"
			FROM raw_reviews
			WHERE "Title" ILIKE :t AND "ASIN" IS NOT NULL
			ORDER BY id DESC
			LIMIT 1;
			"""
		)
		with eng.connect() as conn:
			row = conn.execute(sql, {"t": f"%{title.strip()}%"}).mappings().first()
			if not row:
				return None
			return {"ASIN": row.get("ASIN"), "Description": row.get("Description")}
	except Exception:
		return None


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

left, right = st.columns(2, gap="large")
with left:
	st.markdown(
		'''
		<div class="media-frame">
		  <img src="https://res.retailershakti.com/incom/images/product/Lakme-Face-Sheer-Sun-Kissed-1602502094-10039240-2.jpg" alt="Luminous Glow Serum" />
		</div>
		''',
		unsafe_allow_html=True,
	)
with right:
	html = youtube_embed("https://youtu.be/AvpPocRT3JU?si=HujZcx88Qm-Fs-f2")
	if html:
		st.markdown(html, unsafe_allow_html=True)
	else:
		st.video("https://youtu.be/AvpPocRT3JU?si=HujZcx88Qm-Fs-f2")

st.markdown("---")


# -------------------- Add Review Form ---------------------
st.subheader("Add Customer Review")
st.caption("Tip: Provide Title and Review. ASIN and Description will auto-fill from the DB when possible.")

container_ctx = stylable_container("review_form_card", css_styles="") if stylable_container else st.container()

with container_ctx:
	st.markdown('<div class="main-card">', unsafe_allow_html=True)
	# Apply any pending prefill BEFORE creating widgets
	_pending = st.session_state.pop("_pending_prefill", None)
	if _pending:
		if _pending.get("title") is not None:
			st.session_state["title_input"] = _pending.get("title")
		if _pending.get("desc") is not None:
			st.session_state["desc_input"] = _pending.get("desc")
	# Title input and live suggestions OUTSIDE the form for immediate reruns
	title = st.text_input("Product Title", placeholder="Type to search existing titles…", key="title_input")
	suggestions = []
	if title and len(title.strip()) >= 2 and backend_ok:
		try:
			resp = requests.get(f"{BACKEND_URL}/suggest_titles", params={"q": title.strip(), "limit": 8}, timeout=3)
			if resp.status_code == 200:
				suggestions = resp.json() or []
		except Exception:
			suggestions = []
	# Inline suggestions list (dropdown)
	selected = None
	if suggestions:
		options = ["Select a product…"] + [
			f"{(s.get('Title') or '(untitled)')[:90]}" + (f" · {s.get('ASIN')}" if s.get('ASIN') else "")
			for s in suggestions
		]
		choice = st.selectbox("Matches", options, index=0, key="title_matches")
		if choice and choice != "Select a product…":
			idx = options.index(choice) - 1
			if 0 <= idx < len(suggestions):
				selected = suggestions[idx]
	if selected:
		# Store resolved ASIN/Description for submit
		st.session_state["_resolved_from_suggestion"] = {
			"ASIN": selected.get("ASIN"),
			"Description": selected.get("Description"),
		}
		selected_title = (selected.get("Title") or "").strip()
		selected_desc = selected.get("Description")
		# If description missing, try to enrich before scheduling prefill
		if not selected_desc and selected_title:
			try:
				look = _lookup_asin_description_by_title(selected_title)
				selected_desc = (look or {}).get("Description")
			except Exception:
				selected_desc = None
		# Schedule prefill for next run (must be before widgets are created)
		st.session_state["_pending_prefill"] = {"title": selected_title, "desc": selected_desc}
		st.caption(f"Selected: {selected_title} · ASIN: {selected.get('ASIN')}")
		st.rerun()

	with st.form("add_review_form", clear_on_submit=True):
		cols = st.columns([1, 1])
		with cols[0]:
			if st_star_rating:
				rating = st_star_rating("Rating", maxValue=5, defaultValue=4, key="star_rating")
			else:
				rating = st.slider("Rating", 1, 5, 4)
		with cols[1]:
			# Placeholder to keep balanced layout; could add future fields here
			st.write("")

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
		description = st.text_area(
			"Product Description (auto if known)",
			placeholder="Will auto-fill from DB by Title if known",
			height=80,
			key="desc_input",
		)
		review_text = st.text_area("Review", placeholder="Tell us what you think…", height=120)
		submitted = st.form_submit_button("Submit Review")
	st.markdown('</div>', unsafe_allow_html=True)

if submitted:
	# Resolve ASIN/Description automatically from Title if needed
	resolved_asin = None
	resolved_desc = None
	from_sess = st.session_state.pop("_resolved_from_suggestion", None)
	if from_sess and from_sess.get("ASIN"):
		resolved_asin = from_sess.get("ASIN")
		resolved_desc = from_sess.get("Description")
	elif title:
		lookup = _lookup_asin_description_by_title(title)
		if lookup and lookup.get("ASIN"):
			resolved_asin = lookup.get("ASIN")
			resolved_desc = lookup.get("Description")

	# Compose payload
	# Use Title from session to reflect any suggestion prefill
	effective_title = (st.session_state.get("title_input") or title or "").strip()
	payload = {
		"ASIN": (resolved_asin or "").strip(),
		"Title": effective_title if effective_title else None,
		"Description": (description or resolved_desc or None),
		"Review": review_text.strip(),
		"Region": region,
	}

	# Basic validation: need Title and Review, and must resolve ASIN
	missing_core = [k for k in ("Title", "Review") if not payload.get(k)]
	if missing_core:
		st.error(f"Missing required fields: {', '.join(missing_core)}")
	elif not payload["ASIN"]:
		st.error("Couldn't auto-fill ASIN from Title. Please try a more specific Title.")
	else:
		try:
			res = requests.post(f"{BACKEND_URL}/add_review/", json=payload, timeout=8)
			if res.status_code == 201:
				data = res.json()
				st.success(f"Review submitted! DB id: {data.get('id')}")
				with st.expander("Resolved fields"):
					st.write({"ASIN": payload["ASIN"], "Description": payload.get("Description")})
				with st.expander("Response data"):
					st.json(data)
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