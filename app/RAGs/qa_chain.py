from __future__ import annotations

import os
import asyncio
from functools import lru_cache
import hashlib
import re
from pathlib import Path
from typing import List, Optional, Tuple
from sentence_transformers import SentenceTransformer
import numpy as np

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

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model import LLMManager

load_dotenv()
BASE_DIR = Path(__file__).resolve().parent

# 1. Create a path to a new cache directory inside your project
CACHE_DIR = BASE_DIR / ".huggingface_cache"

# 2. Make sure the directory actually exists
CACHE_DIR.mkdir(exist_ok=True)

# 3. Set HF_HOME to this new, portable path
os.environ['HF_HOME'] = str(CACHE_DIR)
os.environ['TOKENIZERS_PARALLELISM'] = 'false'


# Paths: default to module directory; allow override to CWD for notebooks with ENV RAGS_USE_CWD=1
MODULE_DIR = Path(__file__).resolve().parent
if os.getenv("RAGS_USE_CWD", "0") == "1":
    BASE_DIR = Path(os.getcwd())
else:
    BASE_DIR = MODULE_DIR

QA_DB_PATH = str(BASE_DIR / ".chroma_qa")
CATALOG_DB_PATH = str(BASE_DIR / ".chroma_catalog")

try:
    embedding_function = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    print("Embedding function loaded successfully.")
except Exception as e:
    print(f"Error loading embedding function: {e}")
    exit()
# --- Check the QA Database ---
print(f"\nVerifying QA Database at: {QA_DB_PATH}")
if os.path.exists(QA_DB_PATH):
    try:
        qa_db = Chroma(
            persist_directory=QA_DB_PATH, 
            embedding_function=embedding_function
        )
        # The ._collection.count() method is the most reliable way to check for content.
        doc_count = qa_db._collection.count()
        print(f"✅ SUCCESS: QA Database loaded. It contains {doc_count} documents.")
        if doc_count == 0:
            print("⚠️ WARNING: The database is empty! You need to run your data ingestion script.")
    except Exception as e:
        print(f"❌ ERROR: Failed to load QA Database. Error: {e}")
else:
    print("❌ ERROR: QA Database directory not found. Did you run the ingestion script to create it?")


@lru_cache(maxsize=1)
def _embeddings():
    # Keep model consistent with ingestion; force CPU to avoid MPS OOM on macOS
    # return OpenAIEmbeddings(model="text-embedding-3-large")
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",  # Faster, smaller model
        # model_name="Qwen/Qwen3-Embedding-0.6B",
        model_kwargs={"device": "mps"},  # Use CPU for stability
        encode_kwargs={"batch_size": 1, "normalize_embeddings": True},  # Reduced batch size for speed
    )

# Smart caching with query similarity
_SIMILARITY_CACHE = {}
_SIMILARITY_MODEL = None

def _get_similarity_model():
    global _SIMILARITY_MODEL
    if _SIMILARITY_MODEL is None:
        _SIMILARITY_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
    return _SIMILARITY_MODEL

def _get_similar_cached_query(query: str, threshold: float = 0.85) -> Optional[str]:
    """Check if query is similar to any cached query."""
    if not _SIMILARITY_CACHE:
        return None
    
    model = _get_similarity_model()
    query_embedding = model.encode([query])
    
    for cached_query, cached_embedding in _SIMILARITY_CACHE.items():
        similarity = np.dot(query_embedding[0], cached_embedding) / (
            np.linalg.norm(query_embedding[0]) * np.linalg.norm(cached_embedding)
        )
        if similarity > threshold:
            return cached_query
    return None

def _cache_query_embedding(query: str):
    """Cache query embedding for similarity matching."""
    if len(_SIMILARITY_CACHE) > 100:  # Limit cache size
        # Remove oldest entries
        oldest_keys = list(_SIMILARITY_CACHE.keys())[:20]
        for key in oldest_keys:
            del _SIMILARITY_CACHE[key]
    
    model = _get_similarity_model()
    embedding = model.encode([query])[0]
    _SIMILARITY_CACHE[query] = embedding

FAQ_DB_PATH = str(BASE_DIR / ".chroma_faqs")
FAQS_JSON_PATH = MODULE_DIR.parent / "data" / "FAQs.json"

# Keep Chroma connections alive instead of recreating
_QA_DB_INSTANCE = None
_CATALOG_DB_INSTANCE = None
_FAQ_DB_INSTANCE = None

@lru_cache(maxsize=1)
def _qa_db() -> Chroma:
    global _QA_DB_INSTANCE
    if _QA_DB_INSTANCE is None:
        _QA_DB_INSTANCE = Chroma(persist_directory=QA_DB_PATH, embedding_function=_embeddings())
    return _QA_DB_INSTANCE

@lru_cache(maxsize=1)
def _catalog_db() -> Chroma:
    global _CATALOG_DB_INSTANCE
    if _CATALOG_DB_INSTANCE is None:
        _CATALOG_DB_INSTANCE = Chroma(persist_directory=CATALOG_DB_PATH, embedding_function=_embeddings())
    return _CATALOG_DB_INSTANCE

@lru_cache(maxsize=1)
def _faq_db() -> Optional[Chroma]:
    """Load or build the FAQs vector store if data is available."""
    global _FAQ_DB_INSTANCE
    if _FAQ_DB_INSTANCE is not None:
        return _FAQ_DB_INSTANCE
        
    try:
        # If a persisted store exists, load it
        if Path(FAQ_DB_PATH).exists():
            _FAQ_DB_INSTANCE = Chroma(persist_directory=FAQ_DB_PATH, embedding_function=_embeddings())
            return _FAQ_DB_INSTANCE
        # Else try to build from JSON if present
        if FAQS_JSON_PATH.exists():
            faqs = _load_faqs_json()
            if faqs:
                _build_faq_index(faqs)
                _FAQ_DB_INSTANCE = Chroma(persist_directory=FAQ_DB_PATH, embedding_function=_embeddings())
                return _FAQ_DB_INSTANCE
    except Exception:
        pass
    return None

# Enhanced caching with similarity check
@lru_cache(maxsize=100)  # Increased cache size
def _cached_similarity_search(db_name: str, query: str, k: int = 4):
    # Check for similar cached queries first
    similar_query = _get_similar_cached_query(query)
    if similar_query:
        # Use cached result if available
        cache_key = f"{db_name}:{similar_query}:{k}"
        if hasattr(_cached_similarity_search, 'cache_info'):
            # Use the similar query for cache lookup
            return _cached_similarity_search(db_name, similar_query, k)
    
    # Cache the current query embedding for future similarity checks
    _cache_query_embedding(query)
    
    if db_name == "qa":
        return _qa_db().similarity_search(query, k=k)
    elif db_name == "catalog":
        return _catalog_db().similarity_search_with_score(query, k=k)

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

def _llm() -> ChatOpenAI:
    # Do not change user's model: allow override via env, fallback is safe
    model = LLMManager().get_client()
    return model

def _resolve_product(query: str) -> Tuple[Optional[str], Optional[str], float]:
    """Resolve a product mention to (ASIN, Title, score) using the catalog index.

    Returns (asin, title, score). Lower score is better. If not found, asin/title are None.
    """
    catalog = _catalog_db()
    results = catalog.similarity_search_with_score(query, k=5)  # Get more results for better matching
    if not results:
        return None, None, 1e9
    
    doc, score = results[0]
    asin = doc.metadata.get("ASIN") if doc.metadata else None
    title = doc.metadata.get("Title") if doc.metadata else None
    
    # More lenient threshold - accept matches even with higher scores
    return asin, title, score if score is not None else 0.0

def _retrieve_product_context(question: str, asin: Optional[str], k: int = 2) -> List[str]:
    """Retrieve QA passages, optionally filtering by ASIN."""
    # Try to use cached similarity search first
    try:
        if asin:
            docs = _cached_similarity_search("qa", question, k)
            if hasattr(docs[0], 'metadata') and docs[0].metadata.get("ASIN") != asin:
                # Fallback to direct DB call with filter
                qa = _qa_db()
                docs = qa.similarity_search(question, k=k, filter={"ASIN": asin})
                if not docs:  # fallback if filter too strict
                    docs = qa.similarity_search(question, k=k)
        else:
            docs = _cached_similarity_search("qa", question, k)
        
        if isinstance(docs, list) and docs and hasattr(docs[0], 'page_content'):
            return [d.page_content for d in docs]
        else:
            # Fallback to direct DB call
            qa = _qa_db()
            docs = qa.similarity_search(question, k=k)
            return [d.page_content for d in docs]
    except Exception:
        # Fallback to original implementation
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
    """Use this tool ONLY when a user asks a specific question about a product by mentioning its name, brand, or features. Do not use this tool for simple greetings like 'hi' or 'hello', or for general conversation."""
    asin, title, score = _resolve_product(question)

    context_chunks = _retrieve_product_context(question, asin)
    context_text = _join_context(context_chunks)

    sys = (
        "You are answering a product-specific question. "
        "- Use ONLY the product's ASIN, title, and description provided in context. "
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
    """Recommend N similar products for a given user query or product mention. Detects N from the query if present."""
    
    asin, title, score = _resolve_product(query)
    catalog = _catalog_db()

    # Detect N from query if not provided
    n_val = _parse_number_from_text(query, default=3) if n is None else n
    n_val = max(1, min(10, int(n_val)))

    # Use title if it's a good match, otherwise use the original query
    anchor_text = title if title and score < 1.0 else query
    # Fetch more results to ensure we have options after filtering
    k_fetch = max(n_val + 3, 8)
    results = catalog.similarity_search_with_score(anchor_text, k=k_fetch)

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

    # If no items found, try a broader search
    if not items:
        # Try broader search with common beauty terms
        fallback_terms = ["beauty product", "makeup", "skincare", "cosmetics"]
        for term in fallback_terms:
            fallback_results = catalog.similarity_search_with_score(term, k=n_val)
            for doc, score in fallback_results:
                cand_asin = (doc.metadata or {}).get("ASIN")
                cand_title = (doc.metadata or {}).get("Title")
                if cand_asin and cand_asin not in seen:
                    items.append((cand_title or "Beauty Product", cand_asin, score))
                    seen.add(cand_asin)
                if len(items) >= n_val:
                    break
            if items:
                break

    if not items:
        return (
            f"I couldn't find specific products matching '{query}'. "
            "This might be because the product database needs to be populated or "
            "the search terms don't match available products. "
            "Try asking for more general categories like 'lipstick', 'moisturizer', or 'mascara'."
        )

    bullets = [f"- {t} (ASIN: {a})" for t, a, _ in items]
    header = f"Here are {len(items)} products related to '{query}':"

    sys = (
    "You are recommending products. "
    "- Start with 'Here are some products related to your request.' "
    "- First, detect the category from query (mascara, eyeliner, etc.). "
    "- Only recommend products in the SAME category. "
    "- When showing each product, use the format: 'Product Name (ASIN: XXXXXXXX)'. "
    "- If budget, finish, or filters are mentioned but missing in data, explain that limitation clearly. "
    "- Do not recommend products outside the retrieved set. "
    "- In the end, provide why you are giving this recommendation."
    )


    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", sys),
            ("human", "{bullets}\n\nBased on the above, answer the user's query: {query}"),
        ]
    )
    
    try:
        llm = _llm()
        # Build payload for the template
        payload = {
            "query": query,
            "bullets": "\n".join([header] + bullets),
        }
        messages = prompt.format_messages(**payload)
        out = llm.invoke(messages)
        anchor_line = f"Product: {title or 'Unknown'} (ASIN: {asin or 'N/A'})\n" if (title or asin) else ""
        return anchor_line + (out.content or "")
    except Exception:
        # Fallback to simple bullet list if LLM fails
        return "\n".join([header] + bullets)

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
    "You are comparing two or more products. "
    "- First resolve which products the user means (match title/ASIN). "
    "- Only compare products from the same category. "
    "- When showing each product, use the format: 'Product Name (ASIN: XXXXXXXX)'. "
    "- If a requested product cannot be found, ask the user to clarify. "
    "- Do not add unrelated products. "
    "- Provide the comparison in a clear, structured way (e.g., bullet points or table style). "
    "- In the end, summarize which product may be better suited based on the available details."
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

def _is_greeting_or_casual(query: str) -> bool:
    """Use LLM to detect if the query is a greeting or casual conversation that doesn't need tools."""
    llm = _llm()
    
    # Create a focused prompt for greeting detection
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a conversation classifier. Your task is to determine if a message is a greeting, casual acknowledgment, or requires product-related assistance.

Classify as 'greeting' if the message is:
- A greeting (hello, hi, good morning, etc.)
- A farewell (bye, goodbye, see you later, etc.)
- A thank you or acknowledgment (thanks, okay, great, etc.)
- A casual social interaction (how are you, what's up, etc.)
- A simple acknowledgment (ok, cool, nice, perfect, etc.)

Classify as 'product_query' if the message:
- Asks about any product, feature, or service
- Contains questions that need information
- Requests recommendations, comparisons, or explanations
- Is anything other than a simple greeting or acknowledgment

Respond with ONLY one word: either 'greeting' or 'product_query'."""),
        ("human", query)
    ])
    
    try:
        result = llm.invoke(prompt.format_messages())
        classification = result.content.strip().lower()
        return classification == "greeting"
    except Exception:
        # Fallback to False to ensure product queries are handled
        return False

def _quick_intent_detection(query: str) -> Optional[str]:
    """Use LLM to classify the query intent."""
    # First check if it's a greeting/casual - these shouldn't trigger tools
    if _is_greeting_or_casual(query):
        return "greeting"
    
    # For product queries, use LLM to classify intent
    llm = _llm()
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an intent classifier for a beauty product Q&A system. Analyze the user's message and classify it into one of these categories:

1. 'recommend': User wants product recommendations, suggestions, or similar products
   - Examples: "recommend a moisturizer", "suggest something for dry skin", "show me similar products", "find me alternatives"

2. 'compare': User wants to compare products or understand differences
   - Examples: "compare these two", "which is better", "what's the difference between", "product A vs product B"

3. 'question': User has a specific question about products, features, or information
   - Examples: "what ingredients are in this", "how does it work", "tell me about", "explain the benefits"

4. 'uncertain': The intent is unclear or doesn't fit the above categories

Respond with ONLY one word: 'recommend', 'compare', 'question', or 'uncertain'."""),
        ("human", query)
    ])
    
    try:
        result = llm.invoke(prompt.format_messages())
        intent = result.content.strip().lower()
        
        if intent in ["recommend", "compare", "question"]:
            return intent
        else:
            return None  # Let the full agent handle uncertain cases
    except Exception:
        return None  # Fallback to full agent processing

# Optional convenience: expose a small registry for external imports
TOOLS = [answer_product_question, recommend_products, compare_products]