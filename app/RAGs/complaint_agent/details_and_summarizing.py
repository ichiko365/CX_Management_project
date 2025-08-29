import os
from typing import Literal
from langgraph.types import Command
from .tools import log_complaint_to_db
from .schema import State   
from langchain_core.messages import HumanMessage
from .schema import DepartmentClassification

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from model import LLMManager


def ask_details(state: State) -> Command[Literal["summarize_complaint", "__end__"]]:
    """Check if we have complaint details and summarize or ask for them."""
    # Check if we already have complaint details
    if state.get("complaint_details"):
        return Command(goto="summarize_complaint")
    
    # Check if the user has provided details in the last message
    last_message = None
    for msg in reversed(state["messages"]):
        if isinstance(msg, dict) and msg.get("role") == "user":
            last_message = msg
            break
        elif hasattr(msg, 'type') and msg.type == "human":
            last_message = {"content": msg.content}
            break
    
    if last_message and len(last_message["content"].strip()) > 20:  # Reasonable detail length
        # User has provided details
        complaint_details = last_message["content"]
        return Command(
            update={"complaint_details": complaint_details},
            goto="summarize_complaint"
        )
    else:
        # Ask for details
        response = "Please describe your complaint in detail so I can assist you properly."
        new_message = {"role": "assistant", "content": response}
        return Command(
            update={"messages": [new_message]},
            goto="__end__"
        )

def summarize_complaint(state: State) -> Command[Literal["classify_department", "__end__"]]:
    """Summarize the complaint and ask for confirmation."""
    if state.get("complaint_summary"):
        return Command(goto="classify_department")
    
    # Generate a summary of the complaint
    summary_prompt = f"Summarize the following customer complaint in 1-2 sentences. Be concise but accurate:\n\n{state['complaint_details']}"
    # Initialize the chat model
    llm = LLMManager().get_client()

    # Create a classifier for department classification
    classifier_llm = llm.with_structured_output(DepartmentClassification)
    response = llm.invoke([HumanMessage(content=summary_prompt)])
    summary = response.content
    
    # Ask for confirmation
    confirmation_message = f"Just to make sure I understand correctly: {summary}\n\nIs this accurate? Please respond with 'yes' or 'no'."
    new_message = {"role": "assistant", "content": confirmation_message}
    
    return Command(
        update={
            "messages": [new_message],
            "complaint_summary": summary
        },
        goto="__end__"
    )

def classify_department(state: State) -> Command[Literal["log_complaint", "__end__"]]:
    """Classify the complaint into a department."""
    # Check if user confirmed the summary
    last_message = None
    for msg in reversed(state["messages"]):
        if isinstance(msg, dict) and msg.get("role") == "user":
            last_message = msg
            break
        elif hasattr(msg, 'type') and msg.type == "human":
            last_message = {"content": msg.content}
            break
    
    if last_message:
        user_response = last_message["content"].lower()
        if any(word in user_response for word in ["yes", "correct", "accurate", "right"]):
            # User confirmed, proceed with classification
            classification_prompt = f"""Classify this complaint into one of these departments:
- Billing: Issues with payments, refunds, charges, or invoices
- Technical Support: Problems with setup, functionality, or technical issues  
- Product Quality: Defective products, wrong items, or quality issues
- General Inquiry: Other non-urgent questions or concerns

Complaint: {state['complaint_details']}"""
            # Initialize the chat model
            llm = LLMManager().get_client()

            # Create a classifier for department classification
            classifier_llm = llm.with_structured_output(DepartmentClassification)
            classification = classifier_llm.invoke([HumanMessage(content=classification_prompt)])
            department = classification.department
            
            return Command(
                update={"department": department},
                goto="log_complaint"
            )
        else:
            # User didn't confirm, ask for clarification
            response = "I apologize for the misunderstanding. Could you please clarify your complaint?"
            new_message = {"role": "assistant", "content": response}
            return Command(
                update={
                    "messages": [new_message],
                    "complaint_details": None,
                    "complaint_summary": None
                },
                goto="__end__"
            )
    else:
        response = "Please confirm if my summary of your complaint is accurate by responding 'yes' or 'no'."
        new_message = {"role": "assistant", "content": response}
        return Command(
            update={"messages": [new_message]},
            goto="__end__"
        )

def log_complaint(state: State) -> Command[Literal["__end__"]]:
    """Log the complaint to the database."""
    success = log_complaint_to_db(
        order_id=state["order_id"],
        department=state["department"],
        complaint_details=state["complaint_details"]
    )
    
    if success:
        response = f"Thank you for your feedback. I've logged your complaint with our {state['department']} department. We'll get back to you shortly. Is there anything else I can help you with?"
    else:
        response = "I apologize, but there was an error logging your complaint. Please try again later or contact our support team directly."
    
    new_message = {"role": "assistant", "content": response}
    return Command(
        update={
            "messages": [new_message],
            "complaint_logged": success
        },
        goto="__end__"
    )
