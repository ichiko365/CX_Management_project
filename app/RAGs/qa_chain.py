from __future__ import annotations

import os
from functools import lru_cache
import re
from pathlib import Path
from typing import List, Optional, Tuple

try:
	# Prefer the new package per LangChain 0.2.9+
	from langchain_chroma import Chroma  # type: ignore
except Exception:  # pragma: no cover
	# Fallback for environments without langchain-chroma installed
	from langchain.vectorstores import Chroma  # deprecated path
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
# add ollama model
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from dotenv import load_dotenv

load_dotenv()
os.environ['HF_HOME'] = '/Users/nike/Documents/Data Science Work/Practice/Langchain/huggingface_cache'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'


# Paths: default to module directory; allow override to CWD for notebooks with ENV RAGS_USE_CWD=1
MODULE_DIR = Path(__file__).resolve().parent
if os.getenv("RAGS_USE_CWD", "0") == "1":
	BASE_DIR = Path(os.getcwd())
else:
	BASE_DIR = MODULE_DIR

QA_DB_PATH = str(BASE_DIR / ".chroma_qa")
CATALOG_DB_PATH = str(BASE_DIR / ".chroma_catalog")


@lru_cache(maxsize=1)
def _embeddings():
	# Keep model consistent with ingestion; force CPU to avoid MPS OOM on macOS
	# return OpenAIEmbeddings(model="text-embedding-3-large")
	return HuggingFaceEmbeddings(
		# model_name="sentence-transformers/all-MiniLM-L6-v2",
		model_name="Qwen/Qwen3-Embedding-0.6B",
		model_kwargs={"device": "cpu"},
		encode_kwargs={"batch_size": 16, "normalize_embeddings": True},
	)
FAQ_DB_PATH = str(BASE_DIR / ".chroma_faqs")
FAQS_JSON_PATH = MODULE_DIR.parent / "data" / "FAQs.json"


@lru_cache(maxsize=1)
def _qa_db() -> Chroma:
	return Chroma(persist_directory=QA_DB_PATH, embedding_function=_embeddings())


@lru_cache(maxsize=1)
def _catalog_db() -> Chroma:
	return Chroma(persist_directory=CATALOG_DB_PATH, embedding_function=_embeddings())



@lru_cache(maxsize=1)
def _faq_db() -> Optional[Chroma]:
	"""Load or build the FAQs vector store if data is available."""
	try:
		# If a persisted store exists, load it
		if Path(FAQ_DB_PATH).exists():
			return Chroma(persist_directory=FAQ_DB_PATH, embedding_function=_embeddings())
		# Else try to build from JSON if present
		if FAQS_JSON_PATH.exists():
			faqs = _load_faqs_json()
			if faqs:
				_build_faq_index(faqs)
				return Chroma(persist_directory=FAQ_DB_PATH, embedding_function=_embeddings())
	except Exception:
		pass
	return None


def _load_faqs_json() -> List[dict]:
	"""Load FAQs from JSON. Accepts list of {question, answer} or a dict of q->a."""
	try:
		import json
		with open(FAQS_JSON_PATH, "r", encoding="utf-8") as f:
			data = json.load(f)
		items: List[dict] = []
		if isinstance(data, dict):
			for k, v in data.items():
				if isinstance(v, str):
					items.append({"question": str(k), "answer": v})
		elif isinstance(data, list):
			for it in data:
				if isinstance(it, dict):
					q = it.get("question") or it.get("q") or it.get("Question")
					a = it.get("answer") or it.get("a") or it.get("Answer")
					if q and a:
						items.append({"question": str(q), "answer": str(a)})
		return items
	except Exception:
		return []


def _build_faq_index(faqs: List[dict]) -> None:
	"""Build and persist a Chroma index for FAQs from a list of {question, answer}."""
	if not faqs:
		return
	texts = [f"Q: {it['question']}\nA: {it['answer']}" for it in faqs]
	metadatas = [{"type": "faq", "idx": i} for i in range(len(texts))]
	try:
		Chroma.from_texts(texts, _embeddings(), metadatas=metadatas, persist_directory=FAQ_DB_PATH)
	except Exception:
		# If creation fails, ignore; FAQ guidance will be empty
		pass


def _retrieve_faq_guidance(question: str, k: int = 3) -> str:
	db = _faq_db()
	if not db:
		return ""
	try:
		docs = db.similarity_search(question, k=k)
		if not docs:
			return ""
		# Join a few Q/A snippets for style/structure guidance
		return "\n\n".join(d.page_content for d in docs)
	except Exception:
		return ""


def get_faq_guidance(question: str, k: int = 3) -> str:
	"""Public helper for agents/UI to fetch FAQ-style guidance for phrasing answers."""
	return _retrieve_faq_guidance(question, k=k)

def _llm() -> ChatOllama:
	# Do not change user's model: allow override via env, fallback is safe
	# model = os.getenv("OLLAMA_MODEL", "llama3")
	return ChatOllama(model='llama3-groq-tool-use')

# def _llm() -> ChatOpenAI:
# 	# Do not change user's model: allow override via env, fallback is safe
# 	model = os.getenv("OPENAI_MODEL", os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"))
# 	return ChatOpenAI(model=model, temperature=0.2)

def _resolve_product(query: str) -> Tuple[Optional[str], Optional[str], float]:
	"""Resolve a product mention to (ASIN, Title, score) using the catalog index.

	Returns (asin, title, score). Lower score is better. If not found, asin/title are None.
	"""
	catalog = _catalog_db()
	results = catalog.similarity_search_with_score(query, k=1)
	if not results:
		return None, None, 1e9
	doc, score = results[0]
	asin = doc.metadata.get("ASIN") if doc.metadata else None
	title = doc.metadata.get("Title") if doc.metadata else None
	return asin, title, score if score is not None else 0.0


def _retrieve_product_context(question: str, asin: Optional[str], k: int = 4) -> List[str]:
	"""Retrieve QA passages, optionally filtering by ASIN."""
	qa = _qa_db()
	if asin:
		docs = qa.similarity_search(question, k=k, filter={"ASIN": asin})
		if not docs:  # fallback if filter too strict
			docs = qa.similarity_search(question, k=k)
	else:
		docs = qa.similarity_search(question, k=k)
	return [d.page_content for d in docs]


def _join_context(chunks: List[str], max_chars: int = 6000) -> str:
	buf = []
	total = 0
	for ch in chunks:
		if total + len(ch) > max_chars:
			break
		buf.append(ch)
		total += len(ch)
	return "\n\n---\n\n".join(buf)


@tool("answer_product_question", return_direct=False)
def answer_product_question(question: str) -> str:
	"""Answer a user question about a product. Automatically detects the product from free text and uses product context. Input: question (string)."""
	asin, title, score = _resolve_product(question)

	context_chunks = _retrieve_product_context(question, asin)
	context_text = _join_context(context_chunks)

	# sys = (
	# 	"You are a helpful customer support assistant. Answer using only the provided beauty product context. "
	# 	"If the answer is not clearly supported by the context, say you are not sure. Keep responses concise."
	# )
	sys = (
		"You are answering a product-specific question. "
		"- Use ONLY the product’s ASIN, title, and description provided in context. "
		"- If the question asks about information not present (e.g., ingredients, price, cruelty-free, shades), reply: "
		"This information is not available from the product description I have."
		"- Always cite which part of the description you used."
	)
	prompt = ChatPromptTemplate.from_messages(
		[
			("system", sys),
			(
				"human",
				(
					"Product detected: ASIN={asin} Title={title} (confidence={score}).\n"
					"Context:\n{context}\n\nQuestion: {q}"
				),
			),
		]
	)
	llm = _llm()
	msg = prompt.format_messages(asin=asin, title=title, score=score, context=context_text, q=question)
	out = llm.invoke(msg)
	header = f"Product: {title or 'Unknown'} (ASIN: {asin or 'N/A'})\n"
	return header + (out.content or "")


def _parse_number_from_text(text: str, default: int, lo: int = 1, hi: int = 10) -> int:
	"""Extract an integer from text like 'top 5' or 'show 4 similar'. Falls back to default and clamps range."""
	if not text:
		return default
	# digits
	m = re.search(r"\b(\d{1,2})\b", text)
	if m:
		try:
			val = int(m.group(1))
			return max(lo, min(hi, val))
		except Exception:
			pass
	# simple number words 1-10
	words = {
		"one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
		"six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
	}
	for w, v in words.items():
		if re.search(rf"\b{w}\b", text, flags=re.I):
			return max(lo, min(hi, v))
	return default


@tool("recommend_products", return_direct=False)
def recommend_products(
	query: str,
	n: Optional[int] = None,
) -> str:
	"""Recommend N similar products for a given user query or product mention. Detects N from the query if present.

	Inputs:
	- query (string)
	- n (optional int)
	- prompt (optional ChatPromptTemplate): If provided, this template will be used to format the final response via the LLM.

	Template variables provided when `prompt` is used:
	- query: Here are the some products related to your request.
	- n: number of recommendations requested
	- anchor_asin: detected ASIN (may be None)
	- anchor_title: detected product title (may be None)
	- items: list of {"title": str, "asin": str, "score": float}
	- bullets: preformatted bullet list string
	"""
	asin, title, _ = _resolve_product(query)
	catalog = _catalog_db()

	# Detect N from query if not provided
	n_val = _parse_number_from_text(query, default=3) if n is None else n
	n_val = max(1, min(10, int(n_val)))

	anchor_text = title or query
	# Fetch a few extra to allow filtering out the same product
	results = catalog.similarity_search_with_score(anchor_text, k=max(n_val + 2, 5))

	items = []
	seen = set([asin]) if asin else set()
	for doc, score in results:
		cand_asin = (doc.metadata or {}).get("ASIN")
		cand_title = (doc.metadata or {}).get("Title")
		if not cand_asin or cand_asin in seen:
			continue
		seen.add(cand_asin)
		items.append((cand_title or "Unknown", cand_asin, score))
		if len(items) >= n_val:
			break

	if not items:
		return "No similar products found with enough confidence."

	bullets = [f"- {t} (ASIN: {a})" for t, a, _ in items]
	header = f"Similar products to '{title or query}':" if (title or query) else "Similar products:"

	# If a ChatPromptTemplate is supplied, use it to render a richer answer with the LLM.
	sys = (
		"You are recommending products. "
		"- Start with 'Here are some products related to your request.'"
		"- First, detect the category from query (mascara, eyeliner, etc.). "
		"- Only recommend products in the SAME category. "
		"- If budget, finish, or filters are mentioned but missing in data, explain that limitation clearly. "
		"- Do not recommend products outside the retrieved set."
		"- In the end, provide why are you giving this recommendation."
	)

	prompt = ChatPromptTemplate.from_messages(
		[
			("system", sys),
			("human", "{bullets}\n\nBased on the above, answer the user's query: {query}"),
		]
	)
	if prompt is not None:
		llm = _llm()
		# Build a flexible payload; the template can pick what it needs.
		payload = {
			"query": query,
			"n": n_val,
			"anchor_asin": asin,
			"anchor_title": title,
			"items": [
				{"title": t, "asin": a, "score": float(s) if s is not None else None}
				for (t, a, s) in items
			],
			"bullets": "\n".join([header, *bullets]),
		}
		messages = prompt.format_messages(**payload)
		out = llm.invoke(messages)
		anchor_line = f"Product: {title or 'Unknown'} (ASIN: {asin or 'N/A'})\n" if (title or asin) else ""
		return anchor_line + (out.content or "")

	# Fallback: simple bullet list
	return "\n".join([header, *bullets])


@tool("compare_products", return_direct=False)
def compare_products(
	query: str = "",
	product_a: Optional[str] = None,
	product_b: Optional[str] = None,
	n: Optional[int] = None,
) -> str:
	"""Compare N products detected from query or provided explicitly. Inputs: query (string), optional product_a/product_b, optional n (default 2 inferred from query). Returns bullet points grouped by each product."""
	# Determine desired number of items
	target_n = _parse_number_from_text(query, default=2) if n is None else int(n)
	target_n = max(2, min(6, target_n))

	catalog = _catalog_db()

	# Collect candidate product mentions
	candidates: List[str] = []
	for s in [product_a, product_b]:
		if s and s.strip():
			candidates.append(s.strip())

	if query:
		# Split by common comparators to extract mentions
		parts = re.split(r"\b(?:vs|versus|,| and | & )\b", query, flags=re.I)
		for p in parts:
			p = re.sub(r"\b(compare|between|among|products?)\b", " ", p, flags=re.I).strip()
			if p:
				candidates.append(p)

	# Resolve to unique ASINs
	resolved = []  # list of (asin, title)
	seen_asin = set()
	for cand in candidates:
		asin, title, _ = _resolve_product(cand)
		if asin and asin not in seen_asin:
			resolved.append((asin, title or cand))
			seen_asin.add(asin)
		if len(resolved) >= target_n:
			break

	# Fallback: fill from catalog by semantic search on the whole query
	if len(resolved) < target_n:
		results = catalog.similarity_search_with_score(query or "compare products", k=target_n + 2)
		for doc, _ in results:
			asin = (doc.metadata or {}).get("ASIN")
			title = (doc.metadata or {}).get("Title")
			if asin and asin not in seen_asin:
				resolved.append((asin, title or "Unknown"))
				seen_asin.add(asin)
			if len(resolved) >= target_n:
				break

	if len(resolved) < 2:
		return "I couldn't identify enough products to compare. Please specify at least two."

	# Retrieve contexts per product
	qa = _qa_db()
	contexts = []  # list of (asin, title, context)
	for asin, title in resolved[:target_n]:
		docs = qa.similarity_search(query or title, k=4, filter={"ASIN": asin})
		ctx = _join_context([d.page_content for d in docs])
		contexts.append((asin, title, ctx))

	# If any context missing, warn gracefully
	if any(not ctx for _, _, ctx in contexts):
		return (
			"I couldn't retrieve enough information for all requested products to compare them reliably. "
			"Try specifying different products or updating the product data."
		)

	# Build dynamic prompt for N products
	# sys = (
	# 	"You compare multiple products strictly using the provided contexts. "
	# 	"Output bullet points grouped by each product, focusing on key features, specs, and notable differences. "
	# 	"Do not fabricate details; if unknown, state it briefly. Keep it concise."
	# )
	sys = (
		"You are comparing two or more products. "
		"- First resolve which products the user means (match title/ASIN). "
		"- Only compare products from the same category. "
		"- If a requested product cannot be found, ask user to clarify. "
		"- Do not add unrelated products. "
	)

	human_lines = []
	for idx, (asin, title, ctx) in enumerate(contexts, start=1):
		human_lines.append(f"Product {idx}: {title} (ASIN: {asin})\nContext {idx}:\n{ctx}")
	human = "\n\n".join(human_lines)

	prompt = ChatPromptTemplate.from_messages([
		("system", sys),
		("human", human),
	])
	llm = _llm()
	out = llm.invoke(prompt.format_messages())
	anchor_line = f"Product: {resolved[0][1]} (ASIN: {resolved[0][0]})\n" if resolved else ""
	return anchor_line + (out.content or "")


# Optional convenience: expose a small registry for external imports
TOOLS = [answer_product_question, recommend_products, compare_products]