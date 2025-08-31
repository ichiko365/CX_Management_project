from __future__ import annotations

import os
import asyncio
from typing import List, Dict, Optional
from langchain_openai import ChatOpenAI
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

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model import LLMManager

def _llm() -> ChatOpenAI:
    model = LLMManager().get_client()
    return model

AGENT_SYSTEM_PROMPT = (
    "You are a friendly and knowledgeable beauty advisor AI."
    "\nGoals:"
    "\n- Chat naturally, concise but not robotic."
    "\n- Respond to greetings/farewells/thanks warmly, without product info."
    "\n- Use retrieved context only for beauty product answers; say briefly if missing."
    "\n- Detect products from free text (no need for exact ASINs)."
    "\n- Recommend similar products with a plain-language reason."
    "\n- Compare products clearly with bullet points."
    "\n- Handle light conversation naturally."
    "\nRules:"
    "\n- Prefer using tools (answer_product_question, recommend_products, compare_products)."
    "\n- Never invent details or use outside knowledge."
    "\n- If context is insufficient, acknowledge it briefly."
    "\n- Responses should be medium-length, clear, and friendly."
    "\n- When recommending/comparing, explain reasoning simply."
    "\n- Do not end with generic follow-ups like 'let me know your budget'."
)

def _is_greeting_or_casual(query: str) -> bool:
    """Use LLM to detect if the query is a greeting or casual conversation that doesn't need tools."""
    llm = _llm()
    
    # Create a focused prompt for greeting detection
    prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "Classify the message as 'greeting' if it is a greeting, farewell, thanks, casual remark, "
         "or simple acknowledgment. Otherwise classify as 'product_query'. "
         "Respond with only 'greeting' or 'product_query'."),
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
    ("system",
     "If the message is a greeting, farewell, thanks, or casual remark, reply with a short, warm, natural response "
     "like a helpful store consultant (1–2 sentences max). "
     "Do not mention products unless the user asks. "
     "If the message is not a greeting/casual remark, respond with exactly one word: 'product_query'."),
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
    ("system", 
     "Classify the user's message into exactly one of these intents:\n"
     "- 'recommend' → asking for product suggestions or alternatives\n"
     "- 'compare' → asking to compare or find differences\n"
     "- 'question' → asking about product details or features\n"
     "- 'uncertain' → unclear or none of the above\n\n"
     "Respond with ONLY one word: recommend, compare, question, or uncertain."),
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

    # printing without refining
    return output


# for testing
if __name__ == "__main__":
    query = input("Enter your query: ")
    print(run_agent(query))