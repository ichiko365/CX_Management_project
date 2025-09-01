from typing import List, Dict, Any, TypedDict, Literal
from pydantic import BaseModel, Field

class UserIntent(BaseModel):
    """The user's current intent in the conversation"""
    intent: Literal["complaint", "qna", "reset", "exit", "abort"]

class RouterState(TypedDict):
    """Router state for managing conversation flow."""
    messages: List[Dict[str, str]]
    current_intent: str | None
    complaint_handler_state: Dict[str, Any] | None
    conversation_active: bool