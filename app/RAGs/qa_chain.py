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
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from dotenv import load_dotenv

load_dotenv()


# Paths: default to module directory; allow override to CWD for notebooks with ENV RAGS_USE_CWD=1
MODULE_DIR = Path(__file__).resolve().parent
if os.getenv("RAGS_USE_CWD", "0") == "1":
	BASE_DIR = Path(os.getcwd())
else:
	BASE_DIR = MODULE_DIR

QA_DB_PATH = str(BASE_DIR / ".chroma_qa")
CATALOG_DB_PATH = str(BASE_DIR / ".chroma_catalog")


@lru_cache(maxsize=1)
def _embeddings() -> OpenAIEmbeddings:
	# Keep model consistent with ingestion
	return OpenAIEmbeddings(model="text-embedding-3-large")


@lru_cache(maxsize=1)
def _qa_db() -> Chroma:
	return Chroma(persist_directory=QA_DB_PATH, embedding_function=_embeddings())


@lru_cache(maxsize=1)
def _catalog_db() -> Chroma:
	return Chroma(persist_directory=CATALOG_DB_PATH, embedding_function=_embeddings())


def _llm() -> ChatOpenAI:
	# Do not change user's model: allow override via env, fallback is safe
	model = os.getenv("OPENAI_MODEL", os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"))
	return ChatOpenAI(model=model, temperature=0.2)


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

	sys = (
		"You are a helpful customer support assistant. Answer using only the provided beauty product context. "
		"If the answer is not clearly supported by the context, say you are not sure. Keep responses concise."
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
	return out.content


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
def recommend_products(query: str, n: Optional[int] = None) -> str:
	"""Recommend N similar products for a given user query or product mention. Detects N from the query if present. Inputs: query (string), n (optional int)."""
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
	sys = (
		"You compare multiple products strictly using the provided contexts. "
		"Output bullet points grouped by each product, focusing on key features, specs, and notable differences. "
		"Do not fabricate details; if unknown, state it briefly. Keep it concise."
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
	return out.content


# Optional convenience: expose a small registry for external imports
TOOLS = [answer_product_question, recommend_products, compare_products]