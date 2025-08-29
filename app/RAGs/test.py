import streamlit as st
import sqlite3
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
import time
import functools
import uuid
import os
import sys

def time_llm_response(func):
    """Decorator to time LLM response generation"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        response_time = end_time - start_time
        return result, response_time
    return wrapper

# Import your actual IntentRouter (adjust the import path as needed)
try:
    from chatbot.intent_classifier import IntentRouter  # Replace with your actual import path
    ROUTER_AVAILABLE = True
except ImportError:
    st.error("Could not import IntentRouter. Please ensure the module is available.")
    ROUTER_AVAILABLE = False

def load_previous_messages():
    """Load previous messages from the router's conversation state"""
    if st.session_state.router and hasattr(st.session_state.router, 'state'):
        try:
            # Get messages from router's state
            router_messages = st.session_state.router.state.get("messages", [])
            return router_messages
        except Exception as e:
            st.error(f"Error loading previous messages: {e}")
            return []
    return []

def get_or_create_thread_id():
    """Get existing thread ID or create a new one, with option to use existing conversations"""
    if 'thread_id' not in st.session_state:
        # Check if there are existing conversations in the database
        db_path = "streamlit_conversation_memory.db"
        if os.path.exists(db_path):
            try:
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                
                # Get all existing thread IDs
                cursor.execute("""
                    SELECT DISTINCT thread_id, MAX(updated_at) as last_updated
                    FROM conversation_state 
                    ORDER BY last_updated DESC
                    LIMIT 10
                """)
                
                existing_threads = cursor.fetchall()
                conn.close()
                
                if existing_threads:
                    # For now, use the most recent thread (you can modify this logic)
                    st.session_state.thread_id = existing_threads[0][0]
                    st.session_state.is_continuing_conversation = True
                else:
                    # Create new thread ID
                    st.session_state.thread_id = f"streamlit_user_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"
                    st.session_state.is_continuing_conversation = False
            except Exception as e:
                # Create new thread ID if there's an error
                st.session_state.thread_id = f"streamlit_user_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"
                st.session_state.is_continuing_conversation = False
        else:
            # Create new thread ID if database doesn't exist
            st.session_state.thread_id = f"streamlit_user_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"
            st.session_state.is_continuing_conversation = False
    
    return st.session_state.thread_id

def init_session_state():
    """Initialize session state variables"""
    if 'router' not in st.session_state:
        if ROUTER_AVAILABLE:
            # Get or create thread ID (this will check for existing conversations)
            thread_id = get_or_create_thread_id()
            
            st.session_state.router = IntentRouter(
                db_path="../data/streamlit_conversation_memory.db",
                thread_id=thread_id
            )
            
            # Load previous messages if continuing a conversation
            if getattr(st.session_state, 'is_continuing_conversation', False):
                previous_messages = load_previous_messages()
                if previous_messages:
                    # Convert router messages to display format if needed
                    st.session_state.messages = previous_messages.copy()
                    st.session_state.conversation_active = st.session_state.router.is_conversation_active()
                else:
                    st.session_state.messages = []
            else:
                st.session_state.messages = []
        else:
            st.session_state.router = None
    
    # Initialize other session state variables
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    if 'conversation_active' not in st.session_state:
        st.session_state.conversation_active = True
    
    if 'show_history' not in st.session_state:
        st.session_state.show_history = False
    
    if 'show_analytics' not in st.session_state:
        st.session_state.show_analytics = False
    
    if 'response_times' not in st.session_state:
        st.session_state.response_times = []

def display_conversation_history():
    """Display conversation history in the main chat area"""
    # Show a welcome message if this is a continuing conversation
    if (hasattr(st.session_state, 'is_continuing_conversation') and 
        st.session_state.is_continuing_conversation and 
        st.session_state.messages):
        
        st.info(f"👋 Welcome back! Continuing previous conversation with {len(st.session_state.messages)} messages loaded.")
        st.markdown("---")
    
    # Display all messages
    for i, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            # Add timestamp if available (from router state)
            content = message["content"]
            
            # Add message number for reference
            if message["role"] == "user":
                st.markdown(f"**Message #{i//2 + 1}**")
            
            st.markdown(content)
            
            # Add a subtle separator between message pairs
            if message["role"] == "assistant" and i < len(st.session_state.messages) - 1:
                st.markdown("---")

def get_conversation_analytics():
    """Get analytics about conversations"""
    if not st.session_state.router:
        return None
    
    try:
        history_summary = st.session_state.router.get_conversation_history_summary()
        current_state = st.session_state.router.get_conversation_state()
        
        return {
            "current_session": current_state,
            "historical_data": history_summary
        }
    except Exception as e:
        st.error(f"Error getting analytics: {e}")
        return None

def show_conversation_selector():
    """Show available conversations to switch to"""
    if not st.session_state.router:
        return
    
    try:
        conn = sqlite3.connect(st.session_state.router.db_path)
        cursor = conn.cursor()
        
        # Get all conversations with their latest messages
        cursor.execute("""
            SELECT DISTINCT cs.thread_id, cs.updated_at, cs.state_data,
                   COALESCE(sum.summary, 'No summary available') as summary
            FROM conversation_state cs
            LEFT JOIN conversation_summaries sum ON cs.thread_id = sum.thread_id
            ORDER BY cs.updated_at DESC
            LIMIT 20
        """)
        
        conversations = cursor.fetchall()
        conn.close()
        
        if conversations:
            st.subheader("🗂️ Available Conversations")
            
            for conv in conversations:
                thread_id = conv[0]
                updated_at = conv[1]
                summary = conv[3]
                
                # Get message count from state_data
                try:
                    state_data = json.loads(conv[2]) if conv[2] else {}
                    message_count = len(state_data.get("messages", []))
                except:
                    message_count = 0
                
                # Create a more readable display
                display_id = thread_id[-12:] if len(thread_id) > 12 else thread_id
                
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"**{display_id}**")
                    st.caption(f"{message_count} messages • {updated_at}")
                    st.caption(f"{summary[:100]}..." if len(summary) > 100 else summary)
                
                with col2:
                    if st.button("Load", key=f"load_{thread_id}"):
                        load_conversation(thread_id)
                        st.rerun()
                
                st.markdown("---")
        else:
            st.info("No previous conversations found.")
            
    except Exception as e:
        st.error(f"Error loading conversations: {e}")

def load_conversation(thread_id: str):
    """Load a specific conversation"""
    try:
        # Save current state if needed
        if st.session_state.router:
            st.session_state.router._save_state()
        
        # Update session state
        st.session_state.thread_id = thread_id
        st.session_state.is_continuing_conversation = True
        
        # Create new router with the selected thread
        st.session_state.router = IntentRouter(
            db_path="streamlit_conversation_memory.db",
            thread_id=thread_id
        )
        
        # Load messages
        previous_messages = load_previous_messages()
        if previous_messages:
            st.session_state.messages = previous_messages.copy()
            st.session_state.conversation_active = st.session_state.router.is_conversation_active()
        else:
            st.session_state.messages = []
        
        st.success(f"Loaded conversation: {thread_id[-12:]}")
        
    except Exception as e:
        st.error(f"Error loading conversation: {e}")

def start_new_conversation():
    """Start a completely new conversation"""
    try:
        # Generate new thread ID
        new_thread_id = f"streamlit_user_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"
        
        # Update session state
        st.session_state.thread_id = new_thread_id
        st.session_state.is_continuing_conversation = False
        
        # Create new router
        st.session_state.router = IntentRouter(
            db_path="streamlit_conversation_memory.db",
            thread_id=new_thread_id
        )
        
        # Clear messages
        st.session_state.messages = []
        st.session_state.response_times = []  # Clear response times for new conversation
        st.session_state.conversation_active = True
        
        st.success("Started new conversation!")
        st.rerun()
        
    except Exception as e:
        st.error(f"Error starting new conversation: {e}")

def search_conversation_history(query: str, search_type: str = "all"):
    """Search through conversation history"""
    if not st.session_state.router:
        return []
    
    try:
        conn = sqlite3.connect(st.session_state.router.db_path)
        cursor = conn.cursor()
        
        results = []
        
        if search_type in ["all", "qna"]:
            # Search in conversation state messages
            cursor.execute("""
                SELECT thread_id, state_data, updated_at
                FROM conversation_state 
                WHERE state_data LIKE ?
            """, (f"%{query}%",))
            
            for row in cursor.fetchall():
                try:
                    state_data = json.loads(row[1])
                    messages = state_data.get("messages", [])
                    
                    for msg in messages:
                        if query.lower() in msg["content"].lower():
                            results.append({
                                "type": "conversation",
                                "role": msg["role"],
                                "content": msg["content"],
                                "timestamp": row[2],
                                "thread_id": row[0]
                            })
                except:
                    continue
        
        if search_type in ["all", "complaint"]:
            # Search in complaints history
            cursor.execute("""
                SELECT complaint_id, complaint_type, description, status, created_at, thread_id
                FROM complaints_history 
                WHERE description LIKE ? OR complaint_type LIKE ?
                ORDER BY created_at DESC
            """, (f"%{query}%", f"%{query}%"))
            
            for row in cursor.fetchall():
                results.append({
                    "type": "complaint",
                    "complaint_id": row[0],
                    "complaint_type": row[1],
                    "description": row[2],
                    "status": row[3],
                    "timestamp": row[4],
                    "thread_id": row[5]
                })
        
        conn.close()
        return results
        
    except Exception as e:
        st.error(f"Error searching history: {e}")
        return []

def display_search_results(results: List[Dict], query: str):
    """Display search results"""
    if not results:
        st.info(f"No results found for '{query}'")
        return
    
    st.success(f"Found {len(results)} results for '{query}':")
    
    for i, result in enumerate(results):
        with st.expander(f"Result {i+1} - {result['type'].title()}", expanded=False):
            if result["type"] == "complaint":
                st.write(f"**Complaint ID:** {result['complaint_id']}")
                st.write(f"**Type:** {result['complaint_type']}")
                st.write(f"**Status:** {result['status']}")
                st.write(f"**Description:** {result['description']}")
                st.write(f"**Date:** {result['timestamp']}")
            else:
                st.write(f"**Role:** {result['role']}")
                st.write(f"**Content:** {result['content']}")
                st.write(f"**Date:** {result['timestamp']}")

def enhance_user_input_with_context(user_input: str) -> str:
    """Enhance user input with conversation context if needed"""
    # This is a simple implementation - you can enhance it based on your needs
    if not st.session_state.messages:
        return user_input
    
    # Add context about previous complaints if relevant
    if any(keyword in user_input.lower() for keyword in ["complaint", "issue", "problem"]):
        if st.session_state.router:
            complaint_history = st.session_state.router._get_complaint_history()
            if complaint_history:
                # Add context about previous complaints
                context = " Previous complaints: "
                for i, complaint in enumerate(complaint_history[:3]):  # Last 3 complaints
                    context += f"{i+1}. {complaint['complaint_type']} - {complaint['status']}; "
                return user_input + context
    
    return user_input

def get_conversation_summary_for_context() -> str:
    """Generate a summary of the conversation for context display"""
    if not st.session_state.messages:
        return "No conversation history yet."
    
    try:
        # Count messages by type
        user_messages = [msg for msg in st.session_state.messages if msg["role"] == "user"]
        assistant_messages = [msg for msg in st.session_state.messages if msg["role"] == "assistant"]
        
        # Get complaint count if available
        complaint_count = 0
        if st.session_state.router and hasattr(st.session_state.router, 'state'):
            complaint_count = st.session_state.router.state.get("total_complaints", 0)
        
        summary = f"""
        **Conversation Summary:**
        - Total messages: {len(st.session_state.messages)}
        - Your questions: {len(user_messages)}
        - My responses: {len(assistant_messages)}
        - Complaints discussed: {complaint_count}
        """
        
        # Add recent topics if available
        if user_messages:
            recent_topics = []
            for msg in user_messages[-3:]:  # Last 3 user messages
                if len(msg["content"]) < 100:  # Only shorter messages
                    recent_topics.append(msg["content"])
            
            if recent_topics:
                summary += "\n**Recent topics:**\n"
                for topic in recent_topics:
                    summary += f"- {topic}\n"
        
        return summary
        
    except Exception as e:
        return f"Error generating summary: {e}"

def create_conversation_context_display():
    """Create a visual display of conversation context"""
    if not st.session_state.messages:
        return
    
    with st.expander("🧠 Conversation Memory (What I Remember)", expanded=False):
        context_summary = get_conversation_summary_for_context()
        st.markdown(context_summary)
        
        # Add quick stats
        st.markdown("---")
        st.markdown("**Quick Access to Recent Topics:**")
        
        # Extract recent topics mentioned
        recent_user_messages = [msg["content"] for msg in st.session_state.messages[-10:] if msg["role"] == "user"]
        
        if recent_user_messages:
            topics_mentioned = []
            for msg in recent_user_messages:
                if len(msg) < 100:  # Show shorter messages as topics
                    topics_mentioned.append(f"• {msg}")
            
            if topics_mentioned:
                for topic in topics_mentioned[-5:]:  # Last 5 topics
                    st.markdown(topic)

def main():
    st.set_page_config(
        page_title="Customer Service Chat",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 Customer Service Chat Assistant")
    st.markdown("---")
    
    # Initialize session state
    init_session_state()
    
    if not ROUTER_AVAILABLE:
        st.error("IntentRouter is not available. Please check your imports and dependencies.")
        return
    
    # Sidebar for controls and history
    with st.sidebar:
        st.header("🔧 Controls")
        
        # Session info with conversation status
        if st.session_state.router:
            st.info(f"**Session ID:** {st.session_state.thread_id[:20]}...")
            
            # Show conversation continuation status
            if hasattr(st.session_state, 'is_continuing_conversation'):
                if st.session_state.is_continuing_conversation:
                    st.success(f"📚 Continuing previous conversation ({len(st.session_state.messages)} messages loaded)")
                else:
                    st.info("🆕 Started new conversation")
        
        # Conversation selection
        st.subheader("💬 Conversation Management")
        
        # Button to load different conversation
        if st.button("🔄 Switch Conversation"):
            st.session_state.show_conversation_selector = True
        
        if getattr(st.session_state, 'show_conversation_selector', False):
            show_conversation_selector()
        
        # Button to start completely new conversation
        if st.button("🆕 Start New Conversation"):
            start_new_conversation()
        
        # Control buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Reset Current"):
                if st.session_state.router:
                    st.session_state.router.reset_conversation()
                st.session_state.messages = []
                st.session_state.response_times = []  # Clear response times
                st.rerun()
        
        with col2:
            if st.button("📊 Analytics"):
                st.session_state.show_analytics = not st.session_state.show_analytics
                st.rerun()
        
        st.markdown("---")
        
        # Search functionality
        st.header("🔍 Search History")
        search_query = st.text_input("Search conversations:", placeholder="Enter search terms...")
        search_type = st.selectbox("Search in:", ["all", "qna", "complaint"])
        
        if st.button("Search") and search_query:
            with st.spinner("Searching..."):
                results = search_conversation_history(search_query, search_type)
                st.session_state.search_results = results
                st.session_state.search_query = search_query
        
        # Display search results in sidebar
        if hasattr(st.session_state, 'search_results') and st.session_state.search_results:
            st.markdown("---")
            st.subheader(f"Search Results for '{st.session_state.search_query}'")
            
            for i, result in enumerate(st.session_state.search_results[:5]):  # Show first 5 in sidebar
                if result["type"] == "complaint":
                    st.write(f"**{i+1}.** Complaint: {result['complaint_type']}")
                    st.caption(result['description'][:100] + "..." if len(result['description']) > 100 else result['description'])
                else:
                    st.write(f"**{i+1}.** {result['role']}: {result['content'][:50]}...")
                st.caption(f"📅 {result['timestamp']}")
                st.markdown("---")
            
            if len(st.session_state.search_results) > 5:
                st.info(f"+ {len(st.session_state.search_results) - 5} more results in main area")
    
    # Main chat area
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Display analytics if requested
        if st.session_state.show_analytics:
            st.subheader("📊 Conversation Analytics")
            analytics = get_conversation_analytics()
            
            if analytics:
                # Current session stats
                current = analytics["current_session"]
                col_a, col_b, col_c, col_d = st.columns(4)
                
                with col_a:
                    st.metric("Current Messages", current.get("message_count", 0))
                with col_b:
                    st.metric("Session Complaints", current.get("total_complaints", 0))
                with col_c:
                    st.metric("Session Q&A", current.get("total_qna", 0))
                with col_d:
                    st.metric("Intent", current.get("current_intent", "None"))
                
                # Response time analytics
                if st.session_state.response_times:
                    st.subheader("⏱️ Response Time Analytics")
                    col_rt1, col_rt2, col_rt3, col_rt4 = st.columns(4)
                    
                    avg_response_time = sum(st.session_state.response_times) / len(st.session_state.response_times)
                    min_response_time = min(st.session_state.response_times)
                    max_response_time = max(st.session_state.response_times)
                    total_responses = len(st.session_state.response_times)
                    
                    with col_rt1:
                        st.metric("Total Responses", total_responses)
                    with col_rt2:
                        st.metric("Avg Response Time", f"{avg_response_time:.2f}s")
                    with col_rt3:
                        st.metric("Fastest Response", f"{min_response_time:.2f}s")
                    with col_rt4:
                        st.metric("Slowest Response", f"{max_response_time:.2f}s")
                    
                    # Response time chart
                    st.subheader("Response Time Trend")
                    st.line_chart(st.session_state.response_times)
                
                # Historical stats
                historical = analytics["historical_data"]
                st.subheader("Historical Overview")
                
                col_e, col_f, col_g = st.columns(3)
                with col_e:
                    st.metric("Total Sessions", historical.get("total_sessions", 0))
                with col_f:
                    st.metric("Total Complaints", historical.get("total_complaints", 0))
                with col_g:
                    st.metric("Total Q&A", historical.get("total_qna", 0))
                
                # Recent sessions
                if historical.get("recent_sessions"):
                    st.subheader("Recent Sessions")
                    for session in historical["recent_sessions"][:3]:
                        with st.expander(f"Session {session['date']}", expanded=False):
                            st.write(session["summary"])
                            st.write(f"Complaints: {session['complaints']}, Q&A: {session['qna']}")
            
            st.markdown("---")
        
        # Display search results in main area if available
        if hasattr(st.session_state, 'search_results') and st.session_state.search_results:
            st.subheader(f"🔍 Detailed Search Results")
            display_search_results(st.session_state.search_results, st.session_state.search_query)
            st.markdown("---")
        
        # Chat interface
        st.subheader("💬 Chat")
        
        # Show conversation overview at the top if there are previous messages
        if st.session_state.messages:
            with st.expander("📋 Conversation Overview", expanded=False):
                total_messages = len(st.session_state.messages)
                user_messages = len([msg for msg in st.session_state.messages if msg["role"] == "user"])
                assistant_messages = len([msg for msg in st.session_state.messages if msg["role"] == "assistant"])
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Messages", total_messages)
                with col2:
                    st.metric("Your Messages", user_messages)
                with col3:
                    st.metric("Assistant Replies", assistant_messages)
                
                # Show conversation type breakdown if available
                if st.session_state.router:
                    current_state = st.session_state.router.get_conversation_state()
                    if current_state.get("total_complaints", 0) > 0 or current_state.get("total_qna", 0) > 0:
                        st.write("**Interaction Types:**")
                        st.write(f"• Complaints: {current_state.get('total_complaints', 0)}")
                        st.write(f"• Q&A Sessions: {current_state.get('total_qna', 0)}")
        
        # Display conversation history
        display_conversation_history()
        
        # Add conversation context display
        create_conversation_context_display()
        
        # Chat input
        if st.session_state.conversation_active and st.session_state.router:
            user_input = st.chat_input("Type your message here...")
            
            if user_input:
                # Add user message to display
                st.session_state.messages.append({"role": "user", "content": user_input})
                
                # Enhance user input with conversation context if needed
                enhanced_input = enhance_user_input_with_context(user_input)
                
                # Get response from router
                with st.spinner("Processing..."):
                    try:
                        # Use enhanced input for processing but keep original for display
                        @time_llm_response
                        def get_timed_response():
                            return st.session_state.router.process_message(enhanced_input)
                        
                        response, response_time = get_timed_response()
                        
                        # Store response time for analytics
                        st.session_state.response_times.append(response_time)
                        
                        # Add response time to the assistant's message
                        response_with_timing = f"{response}\n\n*⏱️ Response time: {response_time:.2f} seconds*"
                        
                        # Add assistant response to display
                        st.session_state.messages.append({"role": "assistant", "content": response_with_timing})
                        
                        # Sync messages with router state to ensure consistency
                        if hasattr(st.session_state.router, 'state') and 'messages' in st.session_state.router.state:
                            # Update router state with our session messages to keep them in sync
                            st.session_state.router.state['messages'] = st.session_state.messages.copy()
                            st.session_state.router._save_state()
                        
                        # Check if conversation ended
                        if not st.session_state.router.is_conversation_active():
                            st.session_state.conversation_active = False
                            st.info("Conversation has ended. You can start a new conversation by resetting the chat.")
                        
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"Error processing message: {e}")
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": "I apologize, but I encountered an error processing your message. Please try again."
                        })
        
        elif not st.session_state.conversation_active:
            st.warning("Conversation has ended. Please reset to start a new conversation.")
        
        else:
            st.error("Chat system is not available.")
    
    with col2:
        # Quick actions and info
        st.subheader("ℹ️ Quick Info")
        
        if st.session_state.router:
            current_state = st.session_state.router.get_conversation_state()
            
            st.write("**Current Status:**")
            st.write(f"Intent: {current_state.get('current_intent', 'None')}")
            st.write(f"Messages: {current_state.get('message_count', 0)}")
            st.write(f"Active: {'Yes' if current_state.get('conversation_active') else 'No'}")
            
            if current_state.get('complaint_active'):
                st.warning("🚨 Complaint in progress")
        
        st.markdown("---")
        
        st.subheader("💡 Quick Actions")
        st.markdown("""
        **Try these commands:**
        - Ask a product question
        - Report a complaint
        - Say "reset" to start over
        - Say "search complaints about [topic]"
        - Ask "what have we discussed about [topic]?"
        """)
        
        st.markdown("---")
        
        st.subheader("📝 Sample Queries")
        sample_queries = [
            "What beauty products do you recommend?",
            "I have a complaint about my order",
            "What complaints have I filed before?",
            "What did we discuss about skincare?",
            "Reset our conversation",
            "Search previous conversations about returns"
        ]
        
        for query in sample_queries:
            if st.button(f"💬 {query}", key=f"sample_{hash(query)}"):
                if st.session_state.router and st.session_state.conversation_active:
                    # Add to messages and process
                    st.session_state.messages.append({"role": "user", "content": query})
                    
                    # Enhance query with context if needed
                    enhanced_query = enhance_user_input_with_context(query)
                    
                    with st.spinner("Processing..."):
                        try:
                            @time_llm_response
                            def get_timed_response():
                                return st.session_state.router.process_message(enhanced_query)
                            
                            response, response_time = get_timed_response()
                            
                            # Store response time for analytics
                            st.session_state.response_times.append(response_time)
                            
                            # Add response time to the assistant's message
                            response_with_timing = f"{response}\n\n*⏱️ Response time: {response_time:.2f} seconds*"
                            
                            st.session_state.messages.append({"role": "assistant", "content": response_with_timing})
                            
                            # Keep messages in sync
                            if hasattr(st.session_state.router, 'state') and 'messages' in st.session_state.router.state:
                                st.session_state.router.state['messages'] = st.session_state.messages.copy()
                                st.session_state.router._save_state()
                            
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error: {e}")

if __name__ == "__main__":
    main()