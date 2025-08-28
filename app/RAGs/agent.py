from __future__ import annotations

import os
import asyncio
from typing import List, Dict, Optional
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

try:
    import tomllib as tomli  # Python 3.11+
except Exception:
    tomli = None

# Robust import for both package and script execution
try:
    from .qa_chain import TOOLS, answer_product_question, recommend_products, compare_products, get_faq_guidance
except Exception:
    try:
        from RAGs.qa_chain import TOOLS, answer_product_question, recommend_products, compare_products, get_faq_guidance
    except Exception:
        import sys, pathlib
        sys.path.append(str(pathlib.Path(__file__).resolve().parent))
        from qa_chain import TOOLS, answer_product_question, recommend_products, compare_products, get_faq_guidance

def _ensure_openrouter_api_key() -> None:
    """Ensure OPENROUTER_API_KEY is available via env, .env, or Streamlit secrets."""
    if os.getenv("OPENROUTER_API_KEY"):
        return
    # Try default .env resolution
    load_dotenv()
    if os.getenv("OPENROUTER_API_KEY"):
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
        if os.getenv("OPENROUTER_API_KEY"):
            return
    # Try Streamlit secrets
    secrets_path = os.path.join(base_dir, ".streamlit", "secrets.toml")
    if not os.getenv("OPENROUTER_API_KEY") and os.path.exists(secrets_path) and tomli is not None:
        try:
            with open(secrets_path, "rb") as f:
                data = tomli.load(f)
            key = data.get("OPENROUTER_API_KEY")
            if key:
                os.environ["OPENROUTER_API_KEY"] = key
        except Exception:
            pass

def _llm() -> ChatOpenAI:
    _ensure_openrouter_api_key()
    model = os.getenv("OPENAI_MODEL", os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"))
    return ChatOpenAI(model=model, temperature=0.2)

AGENT_SYSTEM_PROMPT = (
    "You are a friendly and knowledgeable beauty advisor AI, here to help customers with product Q&A."
    "\nGoals:"
    "\n- Chat naturally, the way a helpful store consultant would. Keep it concise, but not robotic."
    "\n- When asked general questions like 'hello, hi, good morning', respond warmly and helpfully with greetings only, nothing more like give answer product related things."
    "\n- Answer questions about beauty products using the retrieved context only. If something isn't in the data, say so honestly and briefly."
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
    except Exception as e:
        print(f"Error in greeting detection: {e}")
        # Fallback to False to ensure product queries are handled
        return False

def _get_greeting_response(query: str) -> str:
    """Use LLM to generate an appropriate conversational response for greetings."""
    llm = _llm()
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a friendly beauty product advisor. Generate a warm, natural response to the customer's greeting or casual message.

Keep your response:
- Brief and conversational (1-2 sentences max)
- Focused on beauty products when offering help
- Natural and friendly, not robotic
- End with an offer to help with beauty products if appropriate

Do not ask multiple questions or be overly enthusiastic."""),
        ("human", query)
    ])
    
    try:
        result = llm.invoke(prompt.format_messages())
        return result.content.strip()
    except Exception as e:
        print(f"Error generating greeting response: {e}")
        # Fallback response
        return "Hello! I'm here to help you with beauty product questions. What would you like to know?"

def _clean_faq_snippets(snips: str) -> str:
    """Strip any leading 'Q:' lines and keep only guidance-like content."""
    if not snips:
        return ""
    lines = []
    for ln in snips.splitlines():
        if ln.strip().startswith("Q:"):
            continue
        if ln.strip().startswith("A:"):
            ln = ln.split(":", 1)[1].strip()
        lines.append(ln)
    return "\n".join(lines).strip()

def _quick_intent_detection(query: str) -> Optional[str]:
    """Use LLM-based intent detection to classify the query."""
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
    except Exception as e:
        print(f"Error in intent detection: {e}")
        return None  # Fallback to full agent processing

def get_agent():
    """Return an agent-like runnable that calls tools as needed."""
    llm = _llm().bind_tools(TOOLS)
    prompt = ChatPromptTemplate.from_messages([
        ("system", AGENT_SYSTEM_PROMPT),
        ("placeholder", "{history}"),
        ("human", "{input}"),
    ])
    return prompt | llm

def run_agent(query: str, history: List[Dict[str, str]] = None) -> str:
    """Single-turn interaction: routes to tools and returns final text."""
    
    # Convert history format if needed
    history_msgs = []
    if history:
        for msg in history:
            if msg["role"] == "user":
                history_msgs.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                history_msgs.append(AIMessage(content=msg["content"]))
    
    # Fast intent detection BEFORE LLM tool selection
    detected_intent = _quick_intent_detection(query)
    
    # Handle greetings and casual conversation without tools
    if detected_intent == "greeting":
        return _get_greeting_response(query)
    
    # Route directly to tools based on intent detection
    if detected_intent == "recommend":
        output = recommend_products.invoke({"query": query})
    elif detected_intent == "compare":
        output = compare_products.invoke({"query": query})
    elif detected_intent == "question":
        output = answer_product_question.invoke({"question": query})
    else:
        # Fallback to LLM tool selection when intent is uncertain
        agent = get_agent()
        result = agent.invoke({"input": query, "history": history_msgs})

        # If the model decided to call a tool
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
        else:
            # No tool call; return model's direct answer
            return getattr(result, "content", "I couldn't process that request.")

    # Polish tool output with FAQ guidance
    llm = _llm()
    faq_snippets = get_faq_guidance(query, k=2)
    faq_snippets = _clean_faq_snippets(faq_snippets)
    
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
    
    return getattr(final, "content", str(output))

# for testing
if __name__ == "__main__":
    query = input("Enter your query: ")
    print(run_agent(query))