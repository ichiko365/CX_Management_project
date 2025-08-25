from langgraph.graph import  StateGraph
from .details_and_summarizing import ask_details, summarize_complaint, classify_department, log_complaint
from .schema import State
from .verifying_id import check_order_id, verify_order

# Building the graph
def create_complaint_graph():
    """Create and compile the complaint handling graph."""
    graph_builder = StateGraph(State)
    
    # Add nodes
    graph_builder.add_node("check_order_id", check_order_id)
    graph_builder.add_node("verify_order", verify_order)
    graph_builder.add_node("ask_details", ask_details)
    graph_builder.add_node("summarize_complaint", summarize_complaint)
    graph_builder.add_node("classify_department", classify_department)
    graph_builder.add_node("log_complaint", log_complaint)
    
    # Set entry point
    graph_builder.set_entry_point("check_order_id")
    
    # Compile the graph
    return graph_builder.compile()
