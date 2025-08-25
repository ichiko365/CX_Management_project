from typing import Literal
from langgraph.types import Command
from .tools import verify_order_id
from .schema import State 



def extract_order_id(text: str) -> str | None:
    """Extract order ID from text."""
    words = text.split()
    for word in words:
        # Remove common punctuation
        clean_word = word.strip(".,!?;:")
        if clean_word.isdigit() and len(clean_word) >= 6:
            return clean_word
    return None

def check_order_id(state: State) -> Command[Literal["verify_order", "__end__"]]:
    """Check if we have an order ID and route accordingly."""
    # Check if we already have a verified order ID
    if state.get("order_id_verified"):
        return Command(goto="ask_details")
    
    # Extract order ID from the last user message
    order_id = None
    for msg in reversed(state["messages"]):
        if isinstance(msg, dict) and msg.get("role") == "user":
            order_id = extract_order_id(msg["content"])
            break
        elif hasattr(msg, 'type') and msg.type == "human":
            order_id = extract_order_id(msg.content)
            break
    
    if order_id:
        # We have an order ID, need to verify it
        return Command(update={"order_id": order_id}, goto="verify_order")
    else:
        # No order ID found, ask for it
        response = "To help you with your complaint, I'll need your order ID. Could you please provide it?"
        new_message = {"role": "assistant", "content": response}
        return Command(
            update={"messages": [new_message]},
            goto="__end__"
        )

def verify_order(state: State) -> Command[Literal["ask_details", "__end__"]]:
    """Verify the order ID using the tool."""
    order_id = state["order_id"]
    is_valid = verify_order_id(order_id)
    
    if is_valid:
        response = f"Thank you! I've verified your order ID {order_id}. Please describe your complaint in detail so I can assist you."
        new_message = {"role": "assistant", "content": response}
        return Command(
            update={
                "messages": [new_message],
                "order_id_verified": True
            },
            goto="__end__"
        )
    else:
        response = f"I'm sorry, but the order ID {order_id} appears to be invalid. Please check and provide a valid order ID (should be at least 6 digits)."
        new_message = {"role": "assistant", "content": response}
        return Command(
            update={
                "messages": [new_message],
                "order_id_verified": False,
                "order_id": None  # Reset order ID
            },
            goto="__end__"
        )