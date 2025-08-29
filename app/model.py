"""
LLM Loader Module
This module provides a class-based interface to load different LLMs
(ChatGPT, Gemini, OpenRouter/LLaMA) in a modular way.

Just for reference, here are the options:
- chatgpt -> gpt-4o-mini
- gemini -> gemini-2.5-pro
- openrouter -> meta-llama/llama-3-70b-instruct; 
- ollama -> llama3-groq-tool-use:latest
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama

try:
    import tomllib as tomli  # Python 3.11+
except Exception:  # pragma: no cover
    tomli = None


class LLMManager:
    def __init__(self, 
                 base_model: str = "openrouter", 
                 specific_model: str = "meta-llama/llama-3-70b-instruct",
                 temperature: float = 0.2):
        """
        Initialize the LLM Manager.
        :param base_model: one of ["chatgpt", "gemini", "openrouter"]
        :param specific_model: model name (if None, defaults per provider will be used)
        :param temperature: default=0.2
        """
        load_dotenv()
        self.base_model = base_model.lower()
        self.specific_model = specific_model
        self.temperature = temperature
        self.client = self._load_model()

    def _ensure_env_var(self, var_name: str):
        """Ensure required environment variable exists."""
        api_key = os.getenv(var_name)
        if not api_key:
            raise ValueError(f"{var_name} not found in environment variables or .env file")
        return api_key

    def _load_model(self):
        """Load the requested model based on base_model."""
        if self.base_model == "chatgpt" or self.base_model == "openai":
            api_key = self._ensure_env_var("OPENAI_API_KEY")
            model_name = self.specific_model or "gpt-3.5-turbo"
            return ChatOpenAI(
                model=model_name,
                temperature=self.temperature,
                openai_api_key=api_key
            )

        elif self.base_model == "gemini":
            api_key = self._ensure_env_var("GOOGLE_API_KEY")
            model_name = self.specific_model or "gemini-pro"
            return ChatGoogleGenerativeAI(
                model=model_name,
                temperature=self.temperature, 
                google_api_key=api_key
            )

        elif self.base_model == "openrouter":
            api_key = self._ensure_env_var("OPENROUTER_API_KEY")
            model_name = self.specific_model or "meta-llama/llama-3-70b-instruct"
            return ChatOpenAI(
                model=model_name,
                openai_api_base="https://openrouter.ai/api/v1",
                openai_api_key=api_key,
                temperature=self.temperature,
            )

        elif self.base_model == "ollama":
            model_name = self.specific_model or "llama3-groq-tool-use:latest"
            return ChatOllama(
                model=model_name,
                temperature=self.temperature
            )

        else:
            raise ValueError(f"Unsupported base_model: {self.base_model}")

    def get_client(self):
        """Return the loaded LLM client."""
        return self.client


if __name__ == "__main__":
    # Example usage
    llm_manager = LLMManager(base_model="openrouter")
    client = llm_manager.get_client()
    print(f"Loaded model: {client.model_name}")
    query = "What are the key benefits of using a modular architecture in software development?"
    print(client.invoke(query))
