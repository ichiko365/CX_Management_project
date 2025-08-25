from __future__ import annotations

import os
from typing import List, Deque, Dict, Tuple, Optional
from collections import deque

from langchain_openai import ChatOpenAI
# import Ollama model
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
try:
	import tomllib as tomli  # Python 3.11+
except Exception:  # pragma: no cover
	tomli = None

# Robust import for both package and script execution
try:
	from .qa_chain import TOOLS, answer_product_question, recommend_products, compare_products, get_faq_guidance  # type: ignore
except Exception:
	try:
		from RAGs.qa_chain import TOOLS, answer_product_question, recommend_products, compare_products, get_faq_guidance  # type: ignore
	except Exception:
		import sys, pathlib
		sys.path.append(str(pathlib.Path(__file__).resolve().parent))
		from qa_chain import TOOLS, answer_product_question, recommend_products, compare_products, get_faq_guidance  # type: ignore

def _ensure_openai_api_key() -> None:
	"""Ensure OPENAI_API_KEY is available via env, .env, or Streamlit secrets."""
	if os.getenv("OPENAI_API_KEY"):
		return
	# Try default .env resolution
	load_dotenv()
	if os.getenv("OPENAI_API_KEY"):
		return
	# Try specific .env locations relative to app folder
	base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
	for p in [
		os.path.join(base_dir, ".env"),
		os.path.join(os.path.dirname(base_dir), ".env"),
		os.path.join(os.getcwd(), ".env"),
	]:
		try:
			if os.path.exists(p):
				load_dotenv(p)
		except Exception:
			pass
		if os.getenv("OPENAI_API_KEY"):
			return
	# Try Streamlit secrets
	secrets_path = os.path.join(base_dir, ".streamlit", "secrets.toml")
	if not os.getenv("OPENAI_API_KEY") and os.path.exists(secrets_path) and tomli is not None:
		try:
			with open(secrets_path, "rb") as f:
				data = tomli.load(f)
			key = data.get("OPENAI_API_KEY")
			if key:
				os.environ["OPENAI_API_KEY"] = key
			model = data.get("OPENAI_MODEL") or data.get("OPENAI_CHAT_MODEL")
			if model and not os.getenv("OPENAI_MODEL"):
				os.environ["OPENAI_MODEL"] = model
		except Exception:
			pass

def _llm() -> ChatOpenAI:
	_ensure_openai_api_key()
	model = os.getenv("OPENAI_MODEL", os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"))
	# Tool-calling enabled by default for OpenAI Chat models via LangChain
	return ChatOpenAI(model=model, temperature=0.2)

# def _llm() -> ChatOllama:
# 	model = os.getenv("OLLAMA_MODEL", "llama3")
# 	return ChatOllama(model='llama3-groq-tool-use')

AGENT_SYSTEM_PROMPT = (
    "You are a friendly and knowledgeable beauty advisor AI, here to help customers with product Q&A."
    "\nGoals:"
    "\n- Chat naturally, the way a helpful store consultant would. Keep it concise, but not robotic."
    "\n- Answer questions about beauty products using the retrieved context only. If something isn’t in the data, say so honestly and briefly."
    "\n- Detect the product from free-text mentions without requiring exact ASINs."
    "\n- Recommend similar products when asked, and explain in plain language why they are similar."
    "\n- Compare products side by side with clear, easy-to-scan bullet points."
    "\n- Be able to handle light conversation too (e.g., greetings, 'How can I help?')."
    "\nRules:"
    "\n- Always prefer using the provided tools (answer_product_question, recommend_products, compare_products) when relevant."
    "\n- Never invent details that are not in the product context. If context is insufficient, acknowledge it and give a helpful fallback."
	"\n- Never use your own knowledge or opinions of beauty products."
	"\n- Give note/acknowledgment when context is insufficient."
    "\n- Keep responses engaging and medium-length: clear, friendly, and useful without being wordy."
    "\n- When recommending or comparing, add a touch of helpful reasoning (e.g., 'this one mentions waterproof in the description')."
	"\n- Don't ask in the end something like ' If you have specific preferences like budget or features, let me know for more tailored suggestions!'"
)


# ---- Lightweight conversation memory (last N turns) ----
# Default: keep last 3 turns (human+ai pairs). Override via env or at call time.
_DEFAULT_TURNS = int(os.getenv("AGENT_HISTORY_TURNS", "3"))
_TURNS: int = max(0, _DEFAULT_TURNS)
_MEMORY: Dict[str, Deque[Tuple[str, str]]] = {}
_LAST_PRODUCT: Dict[str, Dict[str, str]] = {}

def set_history_limit(turns: int) -> None:
	"""Change memory limit globally (number of human+ai turns)."""
	global _TURNS
	_TURNS = max(0, int(turns))

def clear_history(session_id: str = "default") -> None:
	"""Clear stored conversation for a session."""
	_MEMORY.pop(session_id, None)
	_LAST_PRODUCT.pop(session_id, None)

def _get_buffer(session_id: str) -> Deque[Tuple[str, str]]:
	if session_id not in _MEMORY:
		_MEMORY[session_id] = deque()
	return _MEMORY[session_id]

def _append_turn(session_id: str, human: str, ai: str, turns_limit: Optional[int] = None) -> None:
	buf = _get_buffer(session_id)
	buf.append((human, ai))
	limit = _TURNS if turns_limit is None else max(0, int(turns_limit))
	# Keep only last `limit` turns
	while len(buf) > limit:
		buf.popleft()

def get_history_messages(session_id: str = "default", turns_limit: Optional[int] = None):
	"""Return history as LangChain messages [HumanMessage, AIMessage, ...]."""
	buf = list(_get_buffer(session_id))
	limit = _TURNS if turns_limit is None else max(0, int(turns_limit))
	if limit:
		buf = buf[-limit:]
	msgs = []
	for h, a in buf:
		msgs.append(HumanMessage(content=h))
		msgs.append(AIMessage(content=a))
	return msgs

def get_history_serialized(session_id: str = "default", turns_limit: Optional[int] = None):
	"""Return history as a list of dicts with role and content."""
	msgs = get_history_messages(session_id, turns_limit)
	out = []
	for m in msgs:
		role = "human" if isinstance(m, HumanMessage) else ("ai" if isinstance(m, AIMessage) else "system")
		out.append({"role": role, "content": m.content})
	return out

def _update_last_product(session_id: str, text: str) -> None:
	"""Parse and store last mentioned product from a standardized heading if present.

	Looks for lines like: 'Product: <Title> (ASIN: <ASIN>)'
	"""
	import re
	m = re.search(r"^\s*Product:\s*(.+?)\s*\(ASIN:\s*([A-Z0-9]+)\)\s*$", text, flags=re.I|re.M)
	if m:
		title = m.group(1).strip()
		asin = m.group(2).strip()
		_LAST_PRODUCT[session_id] = {"title": title, "asin": asin}

def get_last_product(session_id: str = "default") -> Optional[Dict[str, str]]:
	return _LAST_PRODUCT.get(session_id)

def _is_last_product_query(q: str) -> bool:
	import re
	# include common typo 'eariler'
	return bool(re.search(r"\b(what|which)\s+product\s+(did\s+)?(i|we)\s+(mention|say|talk\s+about)\s+(eariler|earlier|before|previously)\b", q, flags=re.I))

def _clean_faq_snippets(snips: str) -> str:
	"""Strip any leading 'Q:' lines and keep only guidance-like content."""
	if not snips:
		return ""
	lines = []
	for ln in snips.splitlines():
		if ln.strip().startswith("Q:"):
			continue
		# Remove a leading 'A: '
		if ln.strip().startswith("A:"):
			ln = ln.split(":", 1)[1].strip()
		lines.append(ln)
	return "\n".join(lines).strip()


def get_agent():
	"""Return an agent-like runnable that calls tools as needed."""
	llm = _llm().bind_tools(TOOLS)
	prompt = ChatPromptTemplate.from_messages([
		("system", AGENT_SYSTEM_PROMPT),
	("placeholder", "{history}"),
		("human", "{input}"),
	])
	return prompt | llm


def run_agent(query: str, session_id: str = "default", history_limit: Optional[int] = None) -> str:
	"""Single-turn interaction: routes to tools and returns final text."""
	# Handle recall of last-mentioned product directly from memory
	if _is_last_product_query(query):
		last = get_last_product(session_id)
		if not last:
			# try to salvage from text in existing history turns
			for h, a in reversed(list(_get_buffer(session_id))):
				import re
				m = re.search(r"^\s*Product:\s*(.+?)\s*\(ASIN:\s*([A-Z0-9]+)\)\s*$", a, flags=re.I|re.M)
				if m:
					last = {"title": m.group(1).strip(), "asin": m.group(2).strip()}
					_LAST_PRODUCT[session_id] = last
					break
		if last:
			ans = f"The previous product you mentioned was {last['title']} (ASIN: {last['asin']})."
		else:
			ans = "I couldn't find a previously mentioned product in this chat."
		_append_turn(session_id, query, ans, turns_limit=history_limit)
		return ans
	agent = get_agent()
	history_msgs = get_history_messages(session_id=session_id, turns_limit=history_limit)
	result = agent.invoke({"input": query, "history": history_msgs})

	# If the model decided to call a tool, LangChain returns a tool-call message.
	# We handle one tool call round-trip for simplicity; UI can loop if needed.
	if hasattr(result, "tool_calls") and result.tool_calls:
		call = result.tool_calls[0]
		name = call["name"]
		args = call.get("args", {})

		if name == "answer_product_question":
			output = answer_product_question.invoke(args)
		elif name == "recommend_products":
			output = recommend_products.invoke(args)
		elif name == "compare_products":
			output = compare_products.invoke(args)
		else:
			output = "I'm not sure how to help with that."

		# Optionally, send tool output back to the model for a final polish with FAQ guidance
		llm = _llm()
		faq_snippets = get_faq_guidance(query, k=3) if 'query' in locals() else ""
		faq_snippets = _clean_faq_snippets(faq_snippets)
		# If the user asked for features/benefits/specs, ensure bullet-pointed features
		features_mode = any(w in query.lower() for w in ["feature", "features", "benefit", "benefits", "spec", "specs", "highlights"]) 
		final = llm.invoke([
			SystemMessage(content=(
				"You will rewrite a tool result into a concise, user-friendly answer without adding new information. "
				"Respect these constraints strictly: do not invent facts, do not copy any 'Q:' or 'A:' labels, and do not start with a question. "
				"When helpful, align tone/structure with these FAQ snippets (if any). If they conflict with the tool result, ignore them.\n\n"
				f"FAQ style guidance:\n{faq_snippets}\n\n"
				+ ("If the user's request was for features/benefits/specs, present a short bullet list of key features extracted from the tool output, using the tool's wording where possible. " if features_mode else "")
				+ "If the tool output contains a line like 'Product: <Title> (ASIN: <ASIN>)', keep it as the first line and then provide the answer."
			)),
			HumanMessage(content=str(output)),
		])
		final_text = getattr(final, "content", str(output))
		_update_last_product(session_id, final_text)
		_append_turn(session_id, query, final_text, turns_limit=history_limit)
		return final_text

	# No tool call; return model's direct answer
	direct = getattr(result, "content", "I couldn't process that request.")
	_update_last_product(session_id, direct)
	_append_turn(session_id, query, direct, turns_limit=history_limit)
	return direct


# for testing
if __name__ == "__main__":
	query = input("Enter your query: ")
	print(run_agent(query))