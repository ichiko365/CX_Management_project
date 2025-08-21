from __future__ import annotations

import os
from typing import List

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
try:
	import tomllib as tomli  # Python 3.11+
except Exception:  # pragma: no cover
	tomli = None

# Robust import for both package and script execution
try:
	from .qa_chain import TOOLS, answer_product_question, recommend_products, compare_products  # type: ignore
except Exception:
	try:
		from RAGs.qa_chain import TOOLS, answer_product_question, recommend_products, compare_products  # type: ignore
	except Exception:
		import sys, pathlib
		sys.path.append(str(pathlib.Path(__file__).resolve().parent))
		from qa_chain import TOOLS, answer_product_question, recommend_products, compare_products  # type: ignore

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


AGENT_SYSTEM_PROMPT = (
	"You are a concise customer support AI for product Q&A Specially for beauty products."
	"\nGoals:"
	"\n- You can answer general questions about beauty products."
	"\n- You can answer to general conversation like, 'Good morning', 'How can I help you today?'"
	"\n- Detect the product from natural language without asking for exact names/ASINs."
	"\n- Answer questions strictly from retrieved product context."
	"\n- Suggest similar products when asked."
	"\n- Compare products and present bullet points grouped by each product."
	"\nRules:"
	"\n- Prefer calling the provided tools when relevant (answer_product_question, recommend_products, compare_products)."
	"\n- If context is insufficient, state uncertainty briefly."
	"\n- Keep answers medium and helpful. Short if user questions are encouraged."
)


def get_agent():
	"""Return an agent-like runnable that calls tools as needed."""
	llm = _llm().bind_tools(TOOLS)
	prompt = ChatPromptTemplate.from_messages([
		("system", AGENT_SYSTEM_PROMPT),
		("human", "{input}"),
	])
	return prompt | llm


def run_agent(query: str) -> str:
	"""Single-turn interaction: routes to tools and returns final text."""
	agent = get_agent()
	result = agent.invoke({"input": query})

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

		# Optionally, send tool output back to the model for a final polish
		llm = _llm()
		final = llm.invoke([
			SystemMessage(content="Rewrite the following tool result into a concise, user-friendly answer without adding new information."),
			HumanMessage(content=str(output)),
		])
		return getattr(final, "content", str(output))

	# No tool call; return model's direct answer
	return getattr(result, "content", "I couldn't process that request.")


# for testing
if __name__ == "__main__":
	query = input("Enter your query: ")
	print(run_agent(query))