import os
import sys
import time
import json
import socket
import subprocess
from pathlib import Path
import sqlite3
import uuid
import base64

import requests
from urllib.parse import urlparse, parse_qs
import streamlit as st
from typing import Optional, Dict, List, Any


# ----------------------- Page setup -----------------------
st.set_page_config(page_title="CX Management", page_icon="💬", layout="wide")

# ----------------------- Helper functions -----------------------
def _get_image_base64(image_path: str) -> str:
	"""Convert image to base64 string for HTML display."""
	try:
		with open(image_path, "rb") as img_file:
			return base64.b64encode(img_file.read()).decode()
	except Exception:
		return ""

# ----------------------- Theme & styles -------------------
st.markdown(
	"""
	<style>
	/* Full page background gradient covering entire viewport */
	.stApp {
	  background: linear-gradient(135deg, #eef2ff 0%, #fff0f6 50%, #effdf6 100%) !important;
	  min-height: 100vh !important;
	  height: 100% !important;
	}
	
	/* Extend background to cover all areas */
	.stApp > header {
	  background: transparent !important;
	}
	
	/* Cover the top toolbar area */
	.stApp > div[data-testid="stToolbar"] {
	  background: transparent !important;
	}
	
	/* Ensure background extends to main content area */
	.main .block-container {
	  background: transparent !important;
	}
	
	/* Cover sidebar background */
	.css-1d391kg, .css-1cypcdb {
	  background: rgba(255,255,255,0.85) !important;
	  backdrop-filter: blur(10px) !important;
	}
	
	/* Main container styling */
	.main-card { 
	  background: rgba(255,255,255,0.85);
	  border: 1px solid rgba(0,0,0,0.08);
	  border-radius: 20px; 
	  padding: 24px; 
	  box-shadow: 0 12px 32px rgba(0,0,0,0.1);
	  backdrop-filter: blur(10px);
	  margin: 16px 0;
	}
	
	/* Enhanced product image container */
	.product-showcase {
	  background: rgba(255,255,255,0.9);
	  border: 2px solid rgba(0,0,0,0.08);
	  border-radius: 16px;
	  padding: 16px;
	  box-shadow: 0 8px 24px rgba(0,0,0,0.12);
	  backdrop-filter: blur(5px);
	  margin-bottom: 24px;
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
	  border-radius: 12px !important;
	}
	
	/* Stronger override for product container images - multiple selectors */
	.product-image-container [data-testid="stImage"] > img,
	.product-image-container img {
	  height: 400px !important;
	  max-height: 400px !important;
	  min-height: 400px !important;
	  width: auto !important;
	  max-width: 100% !important;
	  object-fit: contain !important;
	  object-position: center !important;
	  display: block !important;
	  margin: 0 auto !important;
	}
	
	/* Target any img element inside product container regardless of nesting */
	.product-image-container * img {
	  height: 400px !important;
	  max-height: 400px !important;
	  min-height: 400px !important;
	  object-fit: contain !important;
	}
	
	[data-testid="stVideo"] > video {
	  height: 300px !important;
	  object-fit: contain !important;
	  width: 100% !important;
	}

	/* Fixed dimensions for product images to maintain layout integrity */
	.product-image-container {
	  height: 400px !important;
	  width: 100% !important;
	  display: flex !important;
	  align-items: center !important;
	  justify-content: center !important;
	  background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%) !important;
	  border-radius: 12px !important;
	  overflow: hidden !important;
	  box-shadow: 0 6px 20px rgba(0,0,0,0.1) !important;
	}
	
	/* Ensure ALL images inside product container have fixed dimensions */
	.product-image-container [data-testid="stImage"] {
	  height: 400px !important;
	  width: 100% !important;
	  display: flex !important;
	  align-items: center !important;
	  justify-content: center !important;
	}
	
	.product-image-container [data-testid="stImage"] > img {
	  height: 400px !important;
	  max-height: 400px !important;
	  width: auto !important;
	  max-width: 100% !important;
	  object-fit: contain !important;
	  object-position: center !important;
	  display: block !important;
	  border-radius: 8px !important;
	}
	
	/* Also target any direct img tags in the container */
	.product-image-container > img,
	.product-image-container img {
	  height: 400px !important;
	  max-height: 400px !important;
	  width: auto !important;
	  max-width: 100% !important;
	  object-fit: contain !important;
	  object-position: center !important;
	  display: block !important;
	  border-radius: 8px !important;
	}

	/* Form styling enhancements */
	.stForm {
	  background: rgba(255,255,255,0.7) !important;
	  border-radius: 16px !important;
	  padding: 20px !important;
	  border: 1px solid rgba(0,0,0,0.08) !important;
	}
	
	/* Button styling */
	.stButton > button {
	  border-radius: 12px !important;
	  border: none !important;
	  font-weight: 600 !important;
	  transition: all 0.3s ease !important;
	}
	
	/* Title styling */
	.main-title {
	  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
	  -webkit-background-clip: text !important;
	  -webkit-text-fill-color: transparent !important;
	  background-clip: text !important;
	  font-weight: bold !important;
	  text-align: center !important;
	  margin-bottom: 24px !important;
	}
	
	/* Remove any white backgrounds from containers */
	.stMainBlockContainer, .block-container {
	  background: transparent !important;
	}
	
	/* Ensure full height coverage */
	html, body, [data-testid="stAppViewContainer"], .main {
	  background: transparent !important;
	  min-height: 100vh !important;
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

# Ensure data directory exists
DATA_DIR = APP_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

# Chat DB path (reuse existing DB in RAGs if present)
CHAT_DB_PATH = DATA_DIR / "streamlit_conversation_memory.db"


# -------------------- Chat helper funcs ------------------
def _chat_load_previous_messages() -> List[Dict[str, str]]:
	"""Load previous messages for the current thread."""
	if not st.session_state.get("router"):
		return []
	
	try:
		# Load from database instead of router state
		conn = sqlite3.connect(str(CHAT_DB_PATH))
		cursor = conn.cursor()
		cursor.execute(
			"SELECT state_data FROM conversation_state WHERE thread_id = ?",
			(st.session_state.get("thread_id", ""),)
		)
		result = cursor.fetchone()
		conn.close()
		
		if result and result[0]:
			state_data = json.loads(result[0])
			return state_data.get("messages", [])
		return []
	except Exception as e:
		st.error(f"Error loading previous messages: {e}")
		return []


def _chat_get_or_create_thread_id() -> str:
	"""Get existing thread ID or create a new one."""
	if "thread_id" not in st.session_state:
		db_path = str(CHAT_DB_PATH)
		
		# Ensure database exists
		if not os.path.exists(db_path):
			# Create new thread ID if database doesn't exist
			st.session_state.thread_id = (
				f"streamlit_user_{time.strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"
			)
			st.session_state.is_continuing_conversation = False
		else:
			try:
				conn = sqlite3.connect(db_path)
				cursor = conn.cursor()
				cursor.execute(
					"""
					SELECT thread_id, updated_at
					FROM conversation_state 
					ORDER BY updated_at DESC
					LIMIT 1
					"""
				)
				result = cursor.fetchone()
				conn.close()
				
				if result:
					st.session_state.thread_id = result[0]
					st.session_state.is_continuing_conversation = True
				else:
					st.session_state.thread_id = (
						f"streamlit_user_{time.strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"
					)
					st.session_state.is_continuing_conversation = False
			except Exception as e:
				st.error(f"Database error: {e}")
				st.session_state.thread_id = (
					f"streamlit_user_{time.strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"
				)
				st.session_state.is_continuing_conversation = False
	return st.session_state.thread_id


def chat_init_session_state():
	"""Initialize chat session state."""
	if "router" not in st.session_state:
		if ROUTER_AVAILABLE and IntentRouter is not None:
			thread_id = _chat_get_or_create_thread_id()
			try:
				st.session_state.router = IntentRouter(
					db_path=str(CHAT_DB_PATH),
					thread_id=thread_id,
				)
				
				if getattr(st.session_state, "is_continuing_conversation", False):
					previous_messages = _chat_load_previous_messages()
					if previous_messages:
						st.session_state.messages = previous_messages.copy()
						# Sync messages with router state
						if hasattr(st.session_state.router, 'state'):
							st.session_state.router.state["messages"] = st.session_state.messages.copy()
						st.session_state.conversation_active = (
							st.session_state.router.is_conversation_active()
						)
					else:
						st.session_state.messages = []
				else:
					st.session_state.messages = []
			except Exception as e:
				st.error(f"Error initializing router: {e}")
				st.session_state.router = None
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
	"""Display conversation history with proper formatting."""
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
	"""Get conversation analytics."""
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
	"""Show conversation selector with improved error handling."""
	if not st.session_state.get("router"):
		st.error("Chat system not available.")
		return
		
	try:
		conn = sqlite3.connect(str(CHAT_DB_PATH))
		cursor = conn.cursor()
		
		# Get conversations with better error handling
		cursor.execute(
			"""
			SELECT DISTINCT cs.thread_id, cs.updated_at, cs.state_data
			FROM conversation_state cs
			ORDER BY cs.updated_at DESC
			LIMIT 20
			"""
		)
		conversations = cursor.fetchall()
		conn.close()
		
		if conversations:
			st.subheader("🗂️ Available Conversations")
			for i, conv in enumerate(conversations):
				thread_id = conv[0]
				updated_at = conv[1]
				state_data_str = conv[2]
				
				try:
					state_data = json.loads(state_data_str) if state_data_str else {}
					messages = state_data.get("messages", [])
					message_count = len(messages)
					
					# Create a simple summary from recent messages
					summary = "No messages"
					if messages:
						recent_user_messages = [
							msg["content"][:50] + "..." if len(msg["content"]) > 50 else msg["content"]
							for msg in messages[-3:] if msg["role"] == "user"
						]
						if recent_user_messages:
							summary = " | ".join(recent_user_messages)
						
				except Exception as e:
					message_count = 0
					summary = f"Error loading conversation: {str(e)[:50]}"
				
				display_id = thread_id[-12:] if len(thread_id) > 12 else thread_id
				
				with st.container():
					col1, col2 = st.columns([4, 1])
					with col1:
						st.write(f"**{display_id}**")
						st.caption(f"{message_count} messages • {updated_at}")
						st.caption(summary)
					with col2:
						# Use unique key with index to avoid conflicts
						if st.button("Load", key=f"load_conv_{i}_{hash(thread_id)}"):
							chat_load_conversation(thread_id)
							st.rerun()
					st.markdown("---")
		else:
			st.info("No previous conversations found.")
			
	except Exception as e:
		st.error(f"Error loading conversations: {e}")


def chat_load_conversation(thread_id: str):
	"""Load a specific conversation with improved error handling."""
	try:
		# Save current state first
		if st.session_state.get("router"):
			st.session_state.router._save_state()
		
		# Set new thread ID
		st.session_state.thread_id = thread_id
		st.session_state.is_continuing_conversation = True
		
		# Create new router instance for the thread
		if IntentRouter is not None:
			st.session_state.router = IntentRouter(
				db_path=str(CHAT_DB_PATH), 
				thread_id=thread_id
			)
			
			# Load messages
			previous_messages = _chat_load_previous_messages()
			if previous_messages:
				st.session_state.messages = previous_messages.copy()
				# Sync with router state
				if hasattr(st.session_state.router, 'state'):
					st.session_state.router.state["messages"] = st.session_state.messages.copy()
				st.session_state.conversation_active = (
					st.session_state.router.is_conversation_active()
				)
			else:
				st.session_state.messages = []
				
		# Close conversation selector
		if "show_conversation_selector" in st.session_state:
			st.session_state.show_conversation_selector = False
			
		display_id = thread_id[-12:] if len(thread_id) > 12 else thread_id
		st.success(f"Loaded conversation: {display_id}")
		
	except Exception as e:
		st.error(f"Error loading conversation: {e}")


def chat_start_new_conversation():
	"""Start a new conversation with proper cleanup."""
	try:
		# Save current state first
		if st.session_state.get("router"):
			st.session_state.router._save_state()
			
		# Generate new thread ID
		new_thread_id = (
			f"streamlit_user_{time.strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"
		)
		st.session_state.thread_id = new_thread_id
		st.session_state.is_continuing_conversation = False
		
		# Create new router instance
		if IntentRouter is not None:
			st.session_state.router = IntentRouter(
				db_path=str(CHAT_DB_PATH), 
				thread_id=new_thread_id
			)
			
		# Clear messages
		st.session_state.messages = []
		st.session_state.conversation_active = True
		
		# Close conversation selector
		if "show_conversation_selector" in st.session_state:
			st.session_state.show_conversation_selector = False
			
		st.success("Started new conversation!")
		st.rerun()
		
	except Exception as e:
		st.error(f"Error starting new conversation: {e}")


def chat_search_history(query: str, search_type: str = "all") -> List[Dict[str, Any]]:
	"""Search conversation history."""
	if not st.session_state.get("router"):
		return []
	try:
		conn = sqlite3.connect(str(CHAT_DB_PATH))
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
					state_data = json.loads(row[1]) if row[1] else {}
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
	"""Display search results."""
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
	"""Enhance user input with context."""
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
	"""Get conversation context summary."""
	if not st.session_state.get("messages"):
		return "No conversation history yet."
		
	try:
		user_messages = [m for m in st.session_state.messages if m["role"] == "user"]
		assistant_messages = [m for m in st.session_state.messages if m["role"] == "assistant"]
		
		complaint_count = 0
		if st.session_state.get("router") and hasattr(st.session_state.router, "state"):
			complaint_count = st.session_state.router.state.get("total_complaints", 0)
			
		summary = f"""
**Conversation Summary:**
- Total messages: {len(st.session_state.messages)}
- Your questions: {len(user_messages)}
- My responses: {len(assistant_messages)}
- Complaints discussed: {complaint_count}
"""
		
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
	"""Display conversation context panel."""
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
	"""Best-effort lookup of ASIN, Description, and ImageURL for a given Title.

	Returns dict with keys {"ASIN", "Description", "ImageURL"} when found, else None.
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
			SELECT "ASIN", "Description", "ImageURL"
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
			return {
				"ASIN": row.get("ASIN"), 
				"Description": row.get("Description"),
				"ImageURL": row.get("ImageURL")
			}
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
		product_image_path = str(PROJECT_ROOT / "Media" / "Product.png")
		st.image(product_image_path, caption="✨ Premium Beauty Product", use_container_width=True)
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
		product_video_path = str(PROJECT_ROOT / "Media" / "Product_Video.mp4")
		st.video(product_video_path)
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
	st.markdown('<h1 class="main-title">🎯 CX Management Dashboard</h1>', unsafe_allow_html=True)
	
	# Single column for full-width product image
	image_url = st.session_state.get("current_image_url")
	
	# Only show image container if we have a valid image URL
	if image_url and image_url.strip():
		# Use HTML with inline styles for maximum control over image dimensions
		st.markdown(f'''
		<div class="product-image-container" style="height: 400px; display: flex; align-items: center; justify-content: center; overflow: hidden;">
			<img src="{image_url}" style="height: 400px; max-height: 400px; width: auto; max-width: 100%; object-fit: contain; object-position: center; display: block; border-radius: 8px;" />
		</div>
		''', unsafe_allow_html=True)
		st.markdown('<h3 style="text-align: center; margin-top: 16px; color: #667eea; font-weight: 600;">Featured Product</h3>', unsafe_allow_html=True)
	
	st.markdown('</div>', unsafe_allow_html=True)

	st.markdown("---")

	# -------------------- Add Review Form ---------------------
	st.subheader("📝 Share Your Experience")
	st.markdown("💡 **Quick Tip:** Start typing the product name and we'll help you find it! ASIN and description will auto-fill.")

	container_ctx = stylable_container("review_form_card", css_styles="") if stylable_container else st.container()

	with container_ctx:
		# Apply any pending prefill BEFORE creating widgets
		_pending = st.session_state.pop("_pending_prefill", None)
		if _pending:
			if _pending.get("title") is not None:
				st.session_state["title_input"] = _pending.get("title")
			if _pending.get("desc") is not None:
				st.session_state["desc_input"] = _pending.get("desc")
			if _pending.get("image_url") is not None:
				st.session_state["current_image_url"] = _pending.get("image_url")
		# Title input and live suggestions OUTSIDE the form for immediate reruns
		title = st.text_input("🔍 Product Title", placeholder="Start typing to search products (e.g., 'Moisturizer', 'Lipstick')...", key="title_input")
		
		# Auto-fill ImageURL when title changes (for immediate visual feedback)
		if title and len(title.strip()) >= 3:  # Trigger after 3+ characters
			try:
				lookup = _lookup_asin_description_by_title(title)
				if lookup and lookup.get("ImageURL"):
					new_image_url = lookup.get("ImageURL")
					# Only update if it's different from current to avoid unnecessary reruns
					if st.session_state.get("current_image_url") != new_image_url:
						st.session_state["current_image_url"] = new_image_url
			except Exception:
				pass
		
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
			st.markdown("**🎯 Found these matching products:**")
			options = ["Select a product…"] + [
				f"{(s.get('Title') or '(untitled)')[:90]}" + (f" · {s.get('ASIN')}" if s.get('ASIN') else "")
				for s in suggestions
			]
			choice = st.selectbox("Choose your product", options, index=0, key="title_matches")
			if choice and choice != "Select a product…":
				idx = options.index(choice) - 1
				if 0 <= idx < len(suggestions):
					selected = suggestions[idx]
		
		# Only auto-apply selection if user explicitly chose from dropdown AND we're not in a rerun loop
		if selected and not st.session_state.get("_applying_suggestion", False):
			# Set flag to prevent rerun loops
			st.session_state["_applying_suggestion"] = True
			
			# Store resolved ASIN/Description/ImageURL for submit
			st.session_state["_resolved_from_suggestion"] = {
				"ASIN": selected.get("ASIN"),
				"Description": selected.get("Description"),
				"ImageURL": selected.get("ImageURL"),
			}
			selected_title = (selected.get("Title") or "").strip()
			selected_desc = selected.get("Description")
			selected_image = selected.get("ImageURL")
			# If description or image missing, try to enrich before scheduling prefill
			if (not selected_desc or not selected_image) and selected_title:
				try:
					look = _lookup_asin_description_by_title(selected_title)
					if look:
						selected_desc = selected_desc or look.get("Description")
						selected_image = selected_image or look.get("ImageURL")
				except Exception:
					pass
			# Schedule prefill for next run (must be before widgets are created)
			st.session_state["_pending_prefill"] = {
				"title": selected_title, 
				"desc": selected_desc,
				"image_url": selected_image
			}
			st.caption(f"✅ Selected: {selected_title} · ASIN: {selected.get('ASIN')}")
			# Force immediate image update by updating current_image_url right away
			if selected_image:
				st.session_state["current_image_url"] = selected_image
			st.rerun()
		elif st.session_state.get("_applying_suggestion", False):
			# Clear the flag after rerun to allow future selections
			st.session_state["_applying_suggestion"] = False

		with st.form("add_review_form", clear_on_submit=True):
			st.markdown("### 📊 Review Details")
			cols = st.columns([1, 1])
			with cols[0]:
				if st_star_rating:
					rating = st_star_rating("⭐ Rating", maxValue=5, defaultValue=4, key="star_rating")
				else:
					rating = st.slider("⭐ Rating", 1, 5, 4, help="How would you rate this product?")
			with cols[1]:
				region = st.selectbox(
					"📍 Your Location",
					[
						"Delhi",
						"Mumbai", 
						"Bangalore",
						"Chennai",
						"Kolkata",
						"Other",
					],
					index=0,
					help="This helps us provide location-specific insights"
				)

			description = st.text_area(
				"📝 Product Description",
				placeholder="This will auto-fill based on the product you select above...",
				height=80,
				key="desc_input",
				help="Product details will be filled automatically when you select a product"
			)
			review_text = st.text_area(
				"💭 Your Review", 
				placeholder="Share your experience with this product. What did you like? Any suggestions for improvement?", 
				height=120,
				help="Your detailed feedback helps other customers and improves our products"
			)
			
			# Enhanced submit button with col layout for better visual appeal
			col1, col2, col3 = st.columns([1, 2, 1])
			with col2:
				submitted = st.form_submit_button(
					"🚀 Submit My Review", 
					use_container_width=True,
					help="Share your experience with other customers!"
				)
		st.markdown('</div></div>', unsafe_allow_html=True)

	if submitted:
		# Resolve ASIN/Description automatically from Title if needed
		resolved_asin = None
		resolved_desc = None
		resolved_image = None
		from_sess = st.session_state.pop("_resolved_from_suggestion", None)
		if from_sess and from_sess.get("ASIN"):
			resolved_asin = from_sess.get("ASIN")
			resolved_desc = from_sess.get("Description")
			resolved_image = from_sess.get("ImageURL")
		# If user didn't explicitly select but there's exactly one live suggestion, use it as a fallback
		elif "suggestions" in locals() and isinstance(suggestions, list) and len(suggestions) == 1:
			_only = suggestions[0] or {}
			sa = (_only.get("ASIN") or "").strip()
			if sa:
				resolved_asin = sa
				resolved_desc = _only.get("Description")
				resolved_image = _only.get("ImageURL")
				# Optionally sync the title to the suggested one if present
				if _only.get("Title"):
					st.session_state["title_input"] = _only.get("Title")
		elif title:
			lookup = _lookup_asin_description_by_title(title)
			if lookup and lookup.get("ASIN"):
				resolved_asin = lookup.get("ASIN")
				resolved_desc = lookup.get("Description")
				resolved_image = lookup.get("ImageURL")

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
			st.error(f"⚠️ Missing required fields: {', '.join(missing_core)}")
		elif not payload["ASIN"]:
			st.error("⚠️ Couldn't find product details. Please select from the dropdown suggestions or try a more specific product name.")
		else:
			try:
				res = requests.post(f"{BACKEND_URL}/add_review/", json=payload, timeout=8)
				if res.status_code == 201:
					data = res.json()
					st.success(f"🎉 Thank you for your review! Successfully submitted with ID: {data.get('id')}")
					
					# Update current image URL for display if we have one
					if resolved_image:
						st.session_state["current_image_url"] = resolved_image
					
					# Enhanced success info with better styling
					st.markdown("### 📋 Review Summary")
					col1, col2, col3 = st.columns(3)
					with col1:
						st.info(f"🏷️ **Product ASIN**\n{payload['ASIN']}")
					with col2:
						st.info(f"⭐ **Your Rating**\n{rating}/5 stars")
					with col3:
						st.info(f"🔄 **Status**\nAuto-processed")
					
					st.balloons()  # Celebration animation
					
					# Helpful next steps
					st.markdown("### 🎯 What's Next?")
					st.markdown("""
					- ✅ Your review is now part of our system
					- 🔍 It will help improve our products and services
					- 🤖 Visit our **Chat Assistant** for any questions
					- 📝 Feel free to review more products!
					""")
				else:
					try:
						detail = res.json()
					except Exception:
						detail = res.text
					st.error(f"❌ Failed to submit review ({res.status_code}): {detail}")
			except Exception as e:
				st.error(f"🔌 Connection error: Please check your internet connection and try again.")
				st.error(f"Technical details: {e}")


def page_chat():
	"""Main chat page function with improved error handling and persistence."""
	# Chat session setup
	chat_init_session_state()

	# Sidebar controls for chatbot
	with st.sidebar:
		st.header("🔧 Chat Controls")
		if ROUTER_AVAILABLE and st.session_state.get("router"):
			thread_id = st.session_state.get("thread_id")
			if thread_id:
				display_id = thread_id[-20:] if len(thread_id) > 20 else thread_id
				st.info(f"**Session ID:** {display_id}")
			else:
				st.info("**Session ID:** Initializing...")
				
			if hasattr(st.session_state, "is_continuing_conversation"):
				if st.session_state.is_continuing_conversation:
					st.success(
						f"📚 Continuing previous conversation ({len(st.session_state.messages)} messages loaded)"
					)
				else:
					st.info("🆕 Started new conversation")
		else:
			st.warning("⚠️ Chat system not available")

		st.subheader("💬 Conversation Management")
		
		# Toggle conversation selector
		if st.button("🔄 Switch Conversation"):
			st.session_state.show_conversation_selector = not st.session_state.get("show_conversation_selector", False)
			
		# Show conversation selector if toggled
		if st.session_state.get("show_conversation_selector", False):
			chat_show_conversation_selector()
			
		if st.button("🆕 Start New Conversation"):
			chat_start_new_conversation()

		col_a, col_b = st.columns(2)
		with col_a:
			if st.button("🔄 Reset Current"):
				try:
					if st.session_state.get("router"):
						st.session_state.router.reset_conversation()
					st.session_state.messages = []
					st.success("Conversation reset!")
					st.rerun()
				except Exception as e:
					st.error(f"Error resetting: {e}")
					
		with col_b:
			if st.button("📊 Analytics"):
				st.session_state.show_analytics = not st.session_state.get("show_analytics", False)
				st.rerun()

		st.markdown("---")
		
		# Database status
		if os.path.exists(str(CHAT_DB_PATH)):
			st.success("✅ Database connected")
		else:
			st.error("❌ Database not found")

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

	# Chat input section
	if st.session_state.get("conversation_active", True) and st.session_state.get("router"):
		user_input = st.chat_input("Type your message here…")
		if user_input:
			# Add user message to session state immediately
			st.session_state.messages.append({"role": "user", "content": user_input})
			
			# Display user message immediately
			with st.chat_message("user"):
				st.markdown(f"**Message #{len([m for m in st.session_state.messages if m['role'] == 'user'])}**")
				st.markdown(user_input)
			
			# Process the message
			enhanced = chat_enhance_input(user_input)
			with st.spinner("Processing…"):
				try:
					response = st.session_state.router.process_message(enhanced)
					
					# Add assistant response to session state
					st.session_state.messages.append({"role": "assistant", "content": response})
					
					# Ensure router state is synced
					if hasattr(st.session_state.router, "state") and "messages" in st.session_state.router.state:
						st.session_state.router.state["messages"] = st.session_state.messages.copy()
						st.session_state.router._save_state()
					
					# Display assistant response
					with st.chat_message("assistant"):
						st.markdown(response)
					
					# Check if conversation is still active
					if not st.session_state.router.is_conversation_active():
						st.session_state.conversation_active = False
						st.info("Conversation has ended. Reset to start a new one.")
					
				except Exception as e:
					error_msg = f"Error processing message: {str(e)}"
					st.error(error_msg)
					st.session_state.messages.append({
						"role": "assistant",
						"content": "I encountered an error processing your message. Please try again.",
					})
					with st.chat_message("assistant"):
						st.markdown("I encountered an error processing your message. Please try again.")
						
	elif not st.session_state.get("conversation_active", True):
		st.warning("⚠️ Conversation has ended. Please reset to start a new conversation.")
	else:
		st.error("❌ Chat system is not available. Please check your configuration.")

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
	
	# Organize queries in columns for better space utilization
	cols = st.columns(3)  # 3 columns to fit 6 queries (2 per column)
	for i, query in enumerate(sample_queries):
		col_idx = i % 3  # Cycle through columns
		with cols[col_idx]:
			# Use a more unique key to avoid conflicts
			query_key = f"sample_{i}_{hash(query)}"
			if st.button(f"💬 {query}", key=query_key):
				if st.session_state.get("router") and st.session_state.get("conversation_active", True):
					# Add user message
					st.session_state.messages.append({"role": "user", "content": query})
					enhanced_q = chat_enhance_input(query)
					
					with st.spinner("Processing…"):
						try:
							response = st.session_state.router.process_message(enhanced_q)
							st.session_state.messages.append({"role": "assistant", "content": response})
							
							# Sync router state
							if hasattr(st.session_state.router, "state") and "messages" in st.session_state.router.state:
								st.session_state.router.state["messages"] = st.session_state.messages.copy()
								st.session_state.router._save_state()
							st.rerun()
						except Exception as e:
							st.error(f"Error: {e}")
				else:
					st.warning("Chat system not available or conversation ended.")

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
		else:
			st.info("No analytics available.")
# -------------------- In-app Navigation ------------------
pages = [
	st.Page(page_dashboard, title="Dashboard", icon="🏠"),
	st.Page(page_chat, title="Chat Assistant", icon="🤖"),
]
st.navigation(pages).run()