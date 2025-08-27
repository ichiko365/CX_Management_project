import os
import sys
import time
import json
import socket
import subprocess
from pathlib import Path
import sqlite3
import uuid

import requests
from urllib.parse import urlparse, parse_qs
import streamlit as st
from typing import Optional, Dict, List, Any


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
	
	/* Ensure consistent height for image and video containers */
	[data-testid="stImage"] > img {
	  height: 300px !important;
	  object-fit: contain !important;
	  width: 100% !important;
	}
	
	[data-testid="stVideo"] > video {
	  height: 300px !important;
	  object-fit: contain !important;
	  width: 100% !important;
	}

	/* Nudge header video down a bit so its visual center aligns with the image */
	.media-shift { margin-top: 105px; }

	/* Center the video caption and add a 90px gap from the video - using multiple selectors */
	.media-shift [data-testid="stCaptionContainer"] {
	  margin-top: 90px !important;
	  text-align: center !important;
	  width: 100% !important;
	  display: block !important;
	}
	.media-shift [data-testid="stCaptionContainer"] p {
	  text-align: center !important;
	  margin: 0 auto !important;
	}
	
	/* Alternative selector in case the above doesn't work */
	.media-shift .stCaption {
	  margin-top: 90px !important;
	  text-align: center !important;
	  width: 100% !important;
	}
	
	/* Another alternative - target any element containing "Product Video" */
	.media-shift *:contains("Product Video") {
	  text-align: center !important;
	  margin-top: 90px !important;
	  width: 100% !important;
	}
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

# -------------------- Chatbot Imports --------------------
# Ensure RAGs package root is importable for its internal absolute imports
APP_DIR = Path(__file__).resolve().parent
_rag_root = APP_DIR / "RAGs"
if str(_rag_root) not in sys.path:
	sys.path.insert(0, str(_rag_root))

# Try multiple import paths for IntentRouter to work both when run from repo root or /app
try:
	from RAGs.chatbot.intent_classifier import IntentRouter  # type: ignore
	ROUTER_AVAILABLE = True
except Exception:
	try:
		from app.RAGs.chatbot.intent_classifier import IntentRouter  # type: ignore
		ROUTER_AVAILABLE = True
	except Exception:
		IntentRouter = None  # type: ignore
		ROUTER_AVAILABLE = False

# Chat DB path (reuse existing DB in RAGs if present)
CHAT_DB_PATH = APP_DIR / "data" / "streamlit_conversation_memory.db"


# -------------------- Chat helper funcs ------------------
def _chat_load_previous_messages() -> List[Dict[str, str]]:
	if st.session_state.get("router") and hasattr(st.session_state.router, "state"):
		try:
			return st.session_state.router.state.get("messages", [])
		except Exception as e:
			st.error(f"Error loading previous messages: {e}")
			return []
	return []


def _chat_get_or_create_thread_id() -> str:
	if "thread_id" not in st.session_state:
		db_path = str(CHAT_DB_PATH)
		if os.path.exists(db_path):
			try:
				conn = sqlite3.connect(db_path)
				cursor = conn.cursor()
				cursor.execute(
					"""
					SELECT DISTINCT thread_id, MAX(updated_at) as last_updated
					FROM conversation_state 
					ORDER BY last_updated DESC
					LIMIT 10
					"""
				)
				existing_threads = cursor.fetchall()
				conn.close()
				if existing_threads:
					st.session_state.thread_id = existing_threads[0][0]
					st.session_state.is_continuing_conversation = True
				else:
					st.session_state.thread_id = (
						f"streamlit_user_{time.strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"
					)
					st.session_state.is_continuing_conversation = False
			except Exception:
				st.session_state.thread_id = (
					f"streamlit_user_{time.strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"
				)
				st.session_state.is_continuing_conversation = False
		else:
			st.session_state.thread_id = (
				f"streamlit_user_{time.strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"
			)
			st.session_state.is_continuing_conversation = False
	return st.session_state.thread_id


def chat_init_session_state():
	if "router" not in st.session_state:
		if ROUTER_AVAILABLE and IntentRouter is not None:
			thread_id = _chat_get_or_create_thread_id()
			st.session_state.router = IntentRouter(
				db_path=str(CHAT_DB_PATH),
				thread_id=thread_id,
			)
			if getattr(st.session_state, "is_continuing_conversation", False):
				previous_messages = _chat_load_previous_messages()
				if previous_messages:
					st.session_state.messages = previous_messages.copy()
					st.session_state.conversation_active = (
						st.session_state.router.is_conversation_active()
					)
				else:
					st.session_state.messages = []
			else:
				st.session_state.messages = []
		else:
			st.session_state.router = None
	if "messages" not in st.session_state:
		st.session_state.messages = []
	if "conversation_active" not in st.session_state:
		st.session_state.conversation_active = True
	if "show_history" not in st.session_state:
		st.session_state.show_history = False
	if "show_analytics" not in st.session_state:
		st.session_state.show_analytics = False


def chat_display_conversation_history():
	if (
		hasattr(st.session_state, "is_continuing_conversation")
		and st.session_state.is_continuing_conversation
		and st.session_state.messages
	):
		st.info(
			f"👋 Welcome back! Continuing previous conversation with {len(st.session_state.messages)} messages loaded."
		)
		st.markdown("---")
	for i, message in enumerate(st.session_state.messages):
		with st.chat_message(message["role"]):
			content = message["content"]
			if message["role"] == "user":
				st.markdown(f"**Message #{i//2 + 1}**")
			st.markdown(content)
			if message["role"] == "assistant" and i < len(st.session_state.messages) - 1:
				st.markdown("---")


def chat_get_conversation_analytics() -> Optional[Dict[str, Any]]:
	if not st.session_state.get("router"):
		return None
	try:
		history_summary = st.session_state.router.get_conversation_history_summary()
		current_state = st.session_state.router.get_conversation_state()
		return {"current_session": current_state, "historical_data": history_summary}
	except Exception as e:
		st.error(f"Error getting analytics: {e}")
		return None


def chat_show_conversation_selector():
	if not st.session_state.get("router"):
		return
	try:
		conn = sqlite3.connect(st.session_state.router.db_path)
		cursor = conn.cursor()
		cursor.execute(
			"""
			SELECT DISTINCT cs.thread_id, cs.updated_at, cs.state_data,
				   COALESCE(sum.summary, 'No summary available') as summary
			FROM conversation_state cs
			LEFT JOIN conversation_summaries sum ON cs.thread_id = sum.thread_id
			ORDER BY cs.updated_at DESC
			LIMIT 20
			"""
		)
		conversations = cursor.fetchall()
		conn.close()
		if conversations:
			st.subheader("🗂️ Available Conversations")
			for conv in conversations:
				thread_id = conv[0]
				updated_at = conv[1]
				summary = conv[3]
				try:
					state_data = json.loads(conv[2]) if conv[2] else {}
					message_count = len(state_data.get("messages", []))
				except Exception:
					message_count = 0
				display_id = thread_id[-12:] if len(thread_id) > 12 else thread_id
				col1, col2 = st.columns([3, 1])
				with col1:
					st.write(f"**{display_id}**")
					st.caption(f"{message_count} messages • {updated_at}")
					st.caption(f"{summary[:100]}..." if len(summary) > 100 else summary)
				with col2:
					if st.button("Load", key=f"load_{thread_id}"):
						chat_load_conversation(thread_id)
						st.rerun()
				st.markdown("---")
		else:
			st.info("No previous conversations found.")
	except Exception as e:
		st.error(f"Error loading conversations: {e}")


def chat_load_conversation(thread_id: str):
	try:
		if st.session_state.get("router"):
			st.session_state.router._save_state()
		st.session_state.thread_id = thread_id
		st.session_state.is_continuing_conversation = True
		if IntentRouter is not None:
			st.session_state.router = IntentRouter(
				db_path=str(CHAT_DB_PATH), thread_id=thread_id
			)
		prev = _chat_load_previous_messages()
		if prev:
			st.session_state.messages = prev.copy()
			st.session_state.conversation_active = (
				st.session_state.router.is_conversation_active()
			)
		else:
			st.session_state.messages = []
		st.success(f"Loaded conversation: {thread_id[-12:]}")
	except Exception as e:
		st.error(f"Error loading conversation: {e}")


def chat_start_new_conversation():
	try:
		new_thread_id = (
			f"streamlit_user_{time.strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"
		)
		st.session_state.thread_id = new_thread_id
		st.session_state.is_continuing_conversation = False
		if IntentRouter is not None:
			st.session_state.router = IntentRouter(
				db_path=str(CHAT_DB_PATH), thread_id=new_thread_id
			)
		st.session_state.messages = []
		st.session_state.conversation_active = True
		st.success("Started new conversation!")
		st.rerun()
	except Exception as e:
		st.error(f"Error starting new conversation: {e}")


def chat_search_history(query: str, search_type: str = "all") -> List[Dict[str, Any]]:
	if not st.session_state.get("router"):
		return []
	try:
		conn = sqlite3.connect(st.session_state.router.db_path)
		cursor = conn.cursor()
		results: List[Dict[str, Any]] = []
		if search_type in ["all", "qna"]:
			cursor.execute(
				"""
				SELECT thread_id, state_data, updated_at
				FROM conversation_state 
				WHERE state_data LIKE ?
				""",
				(f"%{query}%",),
			)
			for row in cursor.fetchall():
				try:
					state_data = json.loads(row[1])
					messages = state_data.get("messages", [])
					for msg in messages:
						if query.lower() in msg.get("content", "").lower():
							results.append(
								{
									"type": "conversation",
									"role": msg.get("role"),
									"content": msg.get("content"),
									"timestamp": row[2],
									"thread_id": row[0],
								}
							)
				except Exception:
					continue
		if search_type in ["all", "complaint"]:
			cursor.execute(
				"""
				SELECT complaint_id, complaint_type, description, status, created_at, thread_id
				FROM complaints_history 
				WHERE description LIKE ? OR complaint_type LIKE ?
				ORDER BY created_at DESC
				""",
				(f"%{query}%", f"%{query}%"),
			)
			for row in cursor.fetchall():
				results.append(
					{
						"type": "complaint",
						"complaint_id": row[0],
						"complaint_type": row[1],
						"description": row[2],
						"status": row[3],
						"timestamp": row[4],
						"thread_id": row[5],
					}
				)
		conn.close()
		return results
	except Exception as e:
		st.error(f"Error searching history: {e}")
		return []


def chat_display_search_results(results: List[Dict[str, Any]], query: str):
	if not results:
		st.info(f"No results found for '{query}'")
		return
	st.success(f"Found {len(results)} results for '{query}':")
	for i, result in enumerate(results):
		with st.expander(f"Result {i+1} - {result['type'].title()}", expanded=False):
			if result["type"] == "complaint":
				st.write(f"**Complaint ID:** {result['complaint_id']}")
				st.write(f"**Type:** {result['complaint_type']}")
				st.write(f"**Status:** {result['status']}")
				st.write(f"**Description:** {result['description']}")
				st.write(f"**Date:** {result['timestamp']}")
			else:
				st.write(f"**Role:** {result['role']}")
				st.write(f"**Content:** {result['content']}")
				st.write(f"**Date:** {result['timestamp']}")


def chat_enhance_input(user_input: str) -> str:
	if not st.session_state.get("messages"):
		return user_input
	if any(k in user_input.lower() for k in ["complaint", "issue", "problem"]):
		if st.session_state.get("router") and hasattr(st.session_state.router, "_get_complaint_history"):
			try:
				complaint_history = st.session_state.router._get_complaint_history()
			except Exception:
				complaint_history = []
			if complaint_history:
				context = " Previous complaints: "
				for i, complaint in enumerate(complaint_history[:3]):
					context += f"{i+1}. {complaint['complaint_type']} - {complaint['status']}; "
				return user_input + context
	return user_input


def chat_get_context_summary() -> str:
	if not st.session_state.get("messages"):
		return "No conversation history yet."
	try:
		user_messages = [m for m in st.session_state.messages if m["role"] == "user"]
		assistant_messages = [m for m in st.session_state.messages if m["role"] == "assistant"]
		complaint_count = 0
		if st.session_state.get("router") and hasattr(st.session_state.router, "state"):
			complaint_count = st.session_state.router.state.get("total_complaints", 0)
		summary = (
			f"""
		**Conversation Summary:**
		- Total messages: {len(st.session_state.messages)}
		- Your questions: {len(user_messages)}
		- My responses: {len(assistant_messages)}
		- Complaints discussed: {complaint_count}
		"""
		)
		if user_messages:
			recent_topics: List[str] = []
			for msg in user_messages[-3:]:
				if len(msg["content"]) < 100:
					recent_topics.append(msg["content"])
			if recent_topics:
				summary += "\n**Recent topics:**\n" + "\n".join([f"- {t}" for t in recent_topics])
		return summary
	except Exception as e:
		return f"Error generating summary: {e}"


def chat_context_panel():
	if not st.session_state.get("messages"):
		return
	with st.expander("🧠 Conversation Memory (What I Remember)", expanded=False):
		st.markdown(chat_get_context_summary())
		st.markdown("---")
		st.markdown("**Quick Access to Recent Topics:**")
		recent_user_msgs = [
			m["content"] for m in st.session_state.messages[-10:] if m["role"] == "user"
		]
		if recent_user_msgs:
			topics = [f"• {m}" for m in recent_user_msgs if len(m) < 100]
			for topic in topics[-5:]:
				st.markdown(topic)


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


def page_dashboard():
	# ----------------------- SIDEBAR ADS (Dashboard Only) ---------------------------
	with st.sidebar:
		st.markdown("### 🎯 Featured Product")
		st.markdown("---")
		
		# Product Image Ad
		st.image("../Media/Product.png", caption="✨ Premium Beauty Product", use_container_width=True)
		st.markdown(
			"""
			<div style='text-align: center; padding: 10px; background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 50%, #fecfef 100%); 
			border-radius: 10px; margin: 10px 0;'>
				<h4 style='color: #333; margin: 5px 0;'>🌟 Special Offer!</h4>
				<p style='color: #555; margin: 5px 0; font-size: 14px;'>Get 20% off on this amazing product</p>
			</div>
			""", 
			unsafe_allow_html=True
		)
		
		# Product Video Ad
		st.markdown("#### 📹 See it in Action")
		st.video("../Media/Product_Video.mp4")
		st.markdown(
			"""
			<div style='text-align: center; padding: 8px; background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%); 
			border-radius: 10px; margin: 10px 0;'>
				<p style='color: #333; margin: 5px 0; font-size: 12px;'>💝 Watch our product demo above!</p>
			</div>
			""", 
			unsafe_allow_html=True
		)
		
		st.markdown("---")
	
	# ----------------------- Header ---------------------------
	st.title("CX Management")

	left, right = st.columns(2, gap="small")
	with left:
		# Display product image from local Media folder with fixed height
		st.image("../Media/Product.png", caption="Product Image", use_container_width=True, width=None)
		
	with right:
		# Display product video from local Media folder with caption and slight vertical offset
		st.markdown('<div class="media-shift">', unsafe_allow_html=True)
		st.video("../Media/Product_Video.mp4")
		st.markdown('<div style="margin-top: 110px; text-align: center; width: 100%;"><p style="text-align: center; margin: 0; color: rgb(49, 51, 63); font-size: 14px; line-height: 1.6;">Product Video</p></div>', unsafe_allow_html=True)
		st.markdown('</div>', unsafe_allow_html=True)

	st.markdown("---")

	# -------------------- Add Review Form ---------------------
	st.subheader("Add Customer Review")
	st.caption("Tip: Provide Title and Review. ASIN and Description will auto-fill from the DB when possible.")

	container_ctx = stylable_container("review_form_card", css_styles="") if stylable_container else st.container()

	with container_ctx:
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
		
		# Only auto-apply selection if user explicitly chose from dropdown AND we're not in a rerun loop
		if selected and not st.session_state.get("_applying_suggestion", False):
			# Set flag to prevent rerun loops
			st.session_state["_applying_suggestion"] = True
			
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
		elif st.session_state.get("_applying_suggestion", False):
			# Clear the flag after rerun to allow future selections
			st.session_state["_applying_suggestion"] = False

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
		# If user didn't explicitly select but there's exactly one live suggestion, use it as a fallback
		elif "suggestions" in locals() and isinstance(suggestions, list) and len(suggestions) == 1:
			_only = suggestions[0] or {}
			sa = (_only.get("ASIN") or "").strip()
			if sa:
				resolved_asin = sa
				resolved_desc = _only.get("Description")
				# Optionally sync the title to the suggested one if present
				if _only.get("Title"):
					st.session_state["title_input"] = _only.get("Title")
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


def page_chat():
	# Chat session setup
	chat_init_session_state()

	# Sidebar controls for chatbot
	with st.sidebar:
		st.header("🔧 Chat Controls")
		if ROUTER_AVAILABLE and st.session_state.get("router"):
			st.info(f"**Session ID:** {st.session_state.thread_id[:20]}...")
			if hasattr(st.session_state, "is_continuing_conversation"):
				if st.session_state.is_continuing_conversation:
					st.success(
						f"📚 Continuing previous conversation ({len(st.session_state.messages)} messages loaded)"
					)
				else:
					st.info("🆕 Started new conversation")

		st.subheader("💬 Conversation Management")
		if st.button("🔄 Switch Conversation"):
			st.session_state.show_conversation_selector = True
		if getattr(st.session_state, "show_conversation_selector", False):
			chat_show_conversation_selector()
		if st.button("🆕 Start New Conversation"):
			chat_start_new_conversation()

		col_a, col_b = st.columns(2)
		with col_a:
			if st.button("🔄 Reset Current"):
				if st.session_state.get("router"):
					st.session_state.router.reset_conversation()
				st.session_state.messages = []
				st.rerun()
		with col_b:
			if st.button("📊 Analytics"):
				st.session_state.show_analytics = not st.session_state.get("show_analytics", False)
				st.rerun()

		st.markdown("---")
		st.header("🔍 Search History")
		search_query = st.text_input("Search conversations:", placeholder="Enter search terms...")
		search_type = st.selectbox("Search in:", ["all", "qna", "complaint"])
		if st.button("Search") and search_query:
			with st.spinner("Searching..."):
				results = chat_search_history(search_query, search_type)
				st.session_state.search_results = results
				st.session_state.search_query = search_query
		if hasattr(st.session_state, "search_results") and st.session_state.search_results:
			st.markdown("---")
			st.subheader(f"Search Results for '{st.session_state.search_query}'")
			for i, result in enumerate(st.session_state.search_results[:5]):
				if result["type"] == "complaint":
					st.write(f"**{i+1}.** Complaint: {result['complaint_type']}")
					st.caption(
						result["description"][:100] + "..."
						if len(result["description"]) > 100
						else result["description"]
					)
				else:
					st.write(f"**{i+1}.** {result['role']}: {result['content'][:50]}...")
				st.caption(f"📅 {result['timestamp']}")
				st.markdown("---")
			if len(st.session_state.search_results) > 5:
				st.info(
					f"+ {len(st.session_state.search_results) - 5} more results in main area"
				)

		st.markdown("---")
		# Removed chat width/visibility controls; chat now uses full width on this page

	# -------------------- Chat layout ---------------------
	st.title("🤖 Chat Assistant")
	st.markdown("---")

	# Conversation overview
	if st.session_state.get("messages"):
		with st.expander("📋 Conversation Overview", expanded=False):
			total_messages = len(st.session_state.messages)
			user_messages = len([m for m in st.session_state.messages if m["role"] == "user"])
			assistant_messages = len([m for m in st.session_state.messages if m["role"] == "assistant"])
			c1, c2, c3 = st.columns(3)
			with c1:
				st.metric("Total", total_messages)
			with c2:
				st.metric("You", user_messages)
			with c3:
				st.metric("Assistant", assistant_messages)
	# History and context
	chat_display_conversation_history()
	chat_context_panel()

	# Chat input
	if st.session_state.get("conversation_active", True) and st.session_state.get("router"):
		user_input = st.chat_input("Type your message here…")
		if user_input:
			st.session_state.messages.append({"role": "user", "content": user_input})
			enhanced = chat_enhance_input(user_input)
			with st.spinner("Processing…"):
				try:
					response = st.session_state.router.process_message(enhanced)
					st.session_state.messages.append({"role": "assistant", "content": response})
					if hasattr(st.session_state.router, "state") and "messages" in st.session_state.router.state:
						st.session_state.router.state["messages"] = st.session_state.messages.copy()
						st.session_state.router._save_state()
					if not st.session_state.router.is_conversation_active():
						st.session_state.conversation_active = False
						st.info("Conversation has ended. Reset to start a new one.")
					st.rerun()
				except Exception as e:
					st.error(f"Error: {e}")
					st.session_state.messages.append({
						"role": "assistant",
						"content": "I encountered an error processing your message. Please try again.",
					})
	elif not st.session_state.get("conversation_active", True):
		st.warning("Conversation has ended. Please reset to start a new conversation.")
	else:
		st.error("Chat system is not available.")

	st.markdown("---")
	st.subheader("ℹ️ Quick Info")
	if st.session_state.get("router"):
		current_state = st.session_state.router.get_conversation_state()
		st.write("**Current Status:**")
		st.write(f"Intent: {current_state.get('current_intent', 'None')}")
		st.write(f"Messages: {current_state.get('message_count', 0)}")
		st.write(f"Active: {'Yes' if current_state.get('conversation_active') else 'No'}")
		if current_state.get("complaint_active"):
			st.warning("🚨 Complaint in progress")

	st.markdown("---")
	st.subheader("📝 Sample Queries")
	sample_queries = [
		"What beauty products do you recommend?",
		"I have a complaint about my order",
		"What complaints have I filed before?",
		"What did we discuss about skincare?",
		"Reset our conversation",
		"Search previous conversations about returns",
	]
	for query in sample_queries:
		if st.button(f"💬 {query}", key=f"sample_{hash(query)}"):
			if st.session_state.get("router") and st.session_state.get("conversation_active", True):
				st.session_state.messages.append({"role": "user", "content": query})
				enhanced_q = chat_enhance_input(query)
				with st.spinner("Processing…"):
					try:
						response = st.session_state.router.process_message(enhanced_q)
						st.session_state.messages.append({"role": "assistant", "content": response})
						if hasattr(st.session_state.router, "state") and "messages" in st.session_state.router.state:
							st.session_state.router.state["messages"] = st.session_state.messages.copy()
							st.session_state.router._save_state()
						st.rerun()
					except Exception as e:
						st.error(f"Error: {e}")

	# Optional analytics section
	if st.session_state.get("show_analytics"):
		st.markdown("---")
		st.subheader("📊 Conversation Analytics")
		analytics = chat_get_conversation_analytics()
		if analytics:
			current = analytics["current_session"]
			col1, col2, col3, col4 = st.columns(4)
			with col1:
				st.metric("Current Messages", current.get("message_count", 0))
			with col2:
				st.metric("Session Complaints", current.get("total_complaints", 0))
			with col3:
				st.metric("Session Q&A", current.get("total_qna", 0))
			with col4:
				st.metric("Intent", current.get("current_intent", "None"))
			historical = analytics["historical_data"]
			st.subheader("Historical Overview")
			c5, c6, c7 = st.columns(3)
			with c5:
				st.metric("Total Sessions", historical.get("total_sessions", 0))
			with c6:
				st.metric("Total Complaints", historical.get("total_complaints", 0))
			with c7:
				st.metric("Total Q&A", historical.get("total_qna", 0))
			if historical.get("recent_sessions"):
				st.subheader("Recent Sessions")
				for session in historical["recent_sessions"][:3]:
					with st.expander(f"Session {session['date']}", expanded=False):
						st.write(session["summary"])
						st.write(
							f"Complaints: {session['complaints']}, Q&A: {session['qna']}"
						)

# -------------------- In-app Navigation ------------------
pages = [
	st.Page(page_dashboard, title="Dashboard", icon="🏠"),
	st.Page(page_chat, title="Chat Assistant", icon="🤖"),
]
st.navigation(pages).run()