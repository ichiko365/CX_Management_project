
from .agent_graph import create_complaint_graph

# Main handler class
class ComplaintHandler:
    def __init__(self):
        self.graph = create_complaint_graph()
        self.state = {
            "messages": [],
            "order_id": None,
            "order_id_verified": None,
            "complaint_details": None,
            "complaint_summary": None,
            "department": None,
            "complaint_logged": None
        }
    
    def handle_message(self, user_message: str) -> str:
        """Handle a user message and return the response."""
        # Add user message to state
        user_msg = {"role": "user", "content": user_message}
        self.state["messages"].append(user_msg)
        
        # Determine which node to start from based on current state
        if not self.state.get("order_id"):
            start_node = "check_order_id"
        elif not self.state.get("order_id_verified"):
            start_node = "verify_order"
        elif not self.state.get("complaint_details"):
            start_node = "ask_details"
        elif not self.state.get("complaint_summary"):
            start_node = "summarize_complaint"
        elif not self.state.get("department"):
            start_node = "classify_department"
        elif not self.state.get("complaint_logged"):
            start_node = "log_complaint"
        else:
            self.reset()
            
        
        # Run the graph from the appropriate node
        try:
            result = self.graph.invoke(self.state)
            self.state.update(result)
            
            # Get the last assistant message
            for msg in reversed(self.state["messages"]):
                if isinstance(msg, dict) and msg.get("role") == "assistant":
                    return msg["content"]
            
            return "I'm ready to help with your complaint. Please provide your order ID."
            
        except Exception as e:
            print(f"Error handling message: {e}")
            return "I apologize, but I encountered an error. Please try again."
    
    def reset(self):
        """Reset the conversation state."""
        self.state = {
            "messages": [],
            "order_id": None,
            "order_id_verified": None,
            "complaint_details": None,
            "complaint_summary": None,
            "department": None,
            "complaint_logged": None
        }



