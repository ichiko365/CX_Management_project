
from complaint_agent.complaint_handler import ComplaintHandler
from complaint_agent.making_database_for_agent import setup_agent_tables, setup_task_tables



def interactive_mode():
    """Run the system in interactive mode."""

    handler = ComplaintHandler()
    
    print("=== Customer Service Complaint System ===")
    print("Type 'quit' to exit, 'reset' to start over\n")
    
    while True:
        user_input = input("You: ").strip()
        
        if user_input.lower() == 'quit':
            break
        elif user_input.lower() == 'reset':
            handler.reset()
            print("Conversation reset. How can I help you today?\n")
            continue
        elif not user_input:
            continue
        
        try:
            response = handler.handle_message(user_input)
            print(f"Assistant: {response}\n")
        except Exception as e:
            print(f"Error: {e}\n")
    
if __name__ == "__main__":
    setup_agent_tables()
    setup_task_tables()
    interactive_mode()