from typing import List, Dict, Any, Optional
from complaint_agent.complaint_handler import ComplaintHandler  # Adjust import path as needed
from chatbot.schema import UserIntent  # Adjust import path as needed
from agent import run_agent
import sqlite3
import json
from datetime import datetime
import os

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from model import LLMManager


class IntentRouter:
    """Main router class that manages intent classification and routing with SQLite persistence."""
    
    def __init__(self, db_path: str = "streamlit_conversation_memory.db", thread_id: str = "user_session"):
        self.llm = LLMManager().get_client()
        self.router_llm = self.llm.with_structured_output(UserIntent)
        
        # SQLite setup for persistence
        self.db_path = db_path
        self.thread_id = thread_id  # Single thread ID for the user
        
        # Initialize database tables
        self._init_database()
        
        self.complaint_handler = None  # Will be initialized when needed
        
        # Load existing conversation state or create new
        self.state = self._load_conversation_state()
        
        # Updated route instructions with context awareness
        self.route_instructions = """You are managing a customer service system that handles multiple types of user interactions. The system has access to full conversation history and can reference previous interactions to provide context-aware support.

Your task is to analyze the user's message and classify their intent into one of the following five categories:

(1) complaint

Classify as complaint when the user is:

- Reporting a problem with an order, billing, delivery, or product.  
- Experiencing technical issues or product quality concerns.  
- Wanting to file a formal complaint that needs to be logged and tracked.  
- Asking for updates on a previously submitted complaint.  
- Referring to a past complaint or issue in a way that requires follow-up.  
- Confirming a summarized complaint with "yes" or similar → this is a confirmation to proceed with logging the complaint (remain in complaint).  

(2) qna

Classify as qna when the user is:

- Asking general questions (especially about beauty products).  
- Seeking product recommendations or comparisons.  
- Making casual conversation or greetings.  
- Asking informational or exploratory queries.  
- Referring to previous general conversations.  

(3) abort

Classify as abort when the user wants to stop the complaint process without completing it, and switch to general Q&A.

Look for phrases like:  
- "stop"  
- "cancel"  
- "never mind"  
- "forget it"  
- "I changed my mind"  
- "let's do something else"  

(4) reset

Classify as reset when the user wants to completely reset the conversation, erase history, and start fresh.

Look for phrases like:  
- "start over"  
- "reset"  
- "clear history"  
- "begin again"  
- "new conversation"  

(5) exit

Classify as exit when the user clearly wants to end the conversation entirely.

Look for phrases like:  
- "bye"  
- "goodbye"  
- "exit"  
- "quit"  
- "end this"  
- "stop talking"  

Additional Instructions:

- Always consider the entire conversation history when determining intent.  
- Do not respond to the user or generate a reply. Simply return the classification.  
- Output one of the following exactly: complaint, qna, abort, reset, exit.  
"""

    def _init_database(self):
        """Initialize database tables for conversation history."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create table for conversation summaries
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversation_summaries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                thread_id TEXT,
                session_date TEXT,
                summary TEXT,
                complaint_count INTEGER DEFAULT 0,
                qna_count INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create table for complaints history
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS complaints_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                thread_id TEXT,
                complaint_id TEXT,
                complaint_type TEXT,
                description TEXT,
                status TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                resolved_at TIMESTAMP
            )
        """)
        
        # Create table for conversation state
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversation_state (
                thread_id TEXT PRIMARY KEY,
                state_data TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        conn.commit()
        conn.close()

    def _load_conversation_state(self) -> Dict[str, Any]:
        """Load existing conversation state from SQLite."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT state_data FROM conversation_state 
                WHERE thread_id = ?
            """, (self.thread_id,))
            
            result = cursor.fetchone()
            conn.close()
            
            if result:
                return json.loads(result[0])
        except Exception as e:
            print(f"Could not load conversation state: {e}")
        
        return self._get_default_state()

    def _get_default_state(self) -> Dict[str, Any]:
        """Get default conversation state."""
        return {
            "messages": [],
            "current_intent": None,
            "complaint_handler_state": None,
            "conversation_active": True,
            "session_start": datetime.now().isoformat(),
            "total_complaints": 0,
            "total_qna": 0
        }

    def _save_state(self):
        """Save current state to SQLite."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO conversation_state 
                (thread_id, state_data, updated_at)
                VALUES (?, ?, ?)
            """, (
                self.thread_id,
                json.dumps(self.state, default=str),
                datetime.now().isoformat()
            ))
            
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"Error saving state: {e}")

    def _get_conversation_context(self) -> str:
        """Generate conversation context for LLM including history."""
        context = []
        
        # Add recent conversation history (last 10 messages)
        recent_messages = self.state["messages"][-10:] if len(self.state["messages"]) > 10 else self.state["messages"]
        if recent_messages:
            context.append("Recent conversation history:")
            for msg in recent_messages:
                role = msg["role"]
                content = msg["content"][:100] + "..." if len(msg["content"]) > 100 else msg["content"]
                context.append(f"{role}: {content}")
        
        # Add complaint history
        complaint_history = self._get_complaint_history()
        if complaint_history:
            context.append("\nPrevious complaints:")
            for complaint in complaint_history[-3:]:  # Last 3 complaints
                context.append(f"- {complaint['complaint_type']}: {complaint['description'][:100]}... (Status: {complaint['status']})")
        
        # Add session stats
        context.append(f"\nSession stats: {self.state['total_complaints']} complaints, {self.state['total_qna']} Q&A interactions")
        
        return "\n".join(context)

    def _get_complaint_history(self) -> List[Dict]:
        """Get complaint history from database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT complaint_id, complaint_type, description, status, created_at, resolved_at
            FROM complaints_history 
            WHERE thread_id = ?
            ORDER BY created_at DESC
            LIMIT 10
        """, (self.thread_id,))
        
        results = cursor.fetchall()
        conn.close()
        
        return [
            {
                "complaint_id": row[0],
                "complaint_type": row[1], 
                "description": row[2],
                "status": row[3],
                "created_at": row[4],
                "resolved_at": row[5]
            }
            for row in results
        ]

    def _save_complaint_to_history(self, complaint_data: Dict):
        """Save completed complaint to history database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO complaints_history 
            (thread_id, complaint_id, complaint_type, description, status, resolved_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            self.thread_id,
            complaint_data.get("complaint_id", ""),
            complaint_data.get("type", ""),
            complaint_data.get("description", ""),
            "resolved",
            datetime.now().isoformat()
        ))
        
        conn.commit()
        conn.close()

    def classify_intent(self, messages: List[Dict[str, str]]) -> str:
        """Classify user intent based on conversation history using LLM."""
        # Add conversation context
        context = self._get_conversation_context()
        system_content = f"{self.route_instructions}\n\nConversation Context:\n{context}"
        
        system_msg = {"role": "system", "content": system_content}
        all_messages = [system_msg] + messages
        
        try:
            response = self.router_llm.invoke(all_messages)
            return response.intent
        except Exception as e:
            print(f"Error in intent classification: {e}")
            return "qna"  # Default to Q&A on error

    def handle_complaint(self, user_message: str) -> str:
        """Handle complaint using the existing ComplaintHandler with history context."""
        if not self.complaint_handler:
            try:
                self.complaint_handler = ComplaintHandler()
                # Pass conversation context to complaint handler if it supports it
                if hasattr(self.complaint_handler, 'set_context'):
                    complaint_history = self._get_complaint_history()
                    self.complaint_handler.set_context({
                        "previous_complaints": complaint_history,
                        "conversation_history": self.state["messages"][-5:]  # Last 5 messages
                    })
            except ImportError:
                return "I apologize, but the complaint system is currently unavailable. Please try again later."
        
        response = self.complaint_handler.handle_message(user_message)
        
        # Check if complaint is completed and save to history
        if (self.complaint_handler and 
            self.complaint_handler.state.get("complaint_logged")):
            
            # Save complaint to history
            complaint_data = self.complaint_handler.state.get("complaint_data", {})
            self._save_complaint_to_history(complaint_data)
            self.state["total_complaints"] += 1
        
        return response

    def handle_qna(self, user_message: str) -> str:
        """Handle Q&A by passing the message and conversation history to the agent."""
        try:
            # Get only the last 5 messages to avoid overwhelming the agent
            recent_history = self.state.get("messages", [])[-10:]  # Last 5 exchanges (10 messages)
            
            # Pass the current message and history to run_agent
            response = run_agent(query=user_message, history=recent_history)
            
            self.state["total_qna"] += 1
            return response
        except Exception as e:
            print(f"Error in handle_qna: {e}")
            return "I apologize, but I'm having trouble accessing the Q&A system. Please try again later."

    def process_message(self, user_message: str) -> str:
        """Main method to process user messages and return responses."""
        # Add user message to conversation history
        user_msg = {"role": "user", "content": user_message}
        self.state["messages"].append(user_msg)
        
        # Classify intent using LLM with context
        classified_intent = self.classify_intent(self.state["messages"])
        
        # Handle special intents first
        if classified_intent == "abort":
            # User wants to abort complaint and switch to Q&A
            if self.complaint_handler:
                self.complaint_handler.reset()
                self.complaint_handler = None
            
            self.state["current_intent"] = "qna"
            response = "Okay, I've stopped the complaint process. I can help you with other questions or discuss your previous interactions if needed."
            
        elif classified_intent == "reset":
            # User wants to reset everything but keep history in database
            if self.complaint_handler:
                self.complaint_handler.reset()
                self.complaint_handler = None
            
            # Save session summary before reset
            self._save_session_summary()
            
            # Reset state but keep database history
            self.state = self._get_default_state()
            response = "Conversation has been reset. Your previous conversations are still saved and I can reference them if needed. How can I help you today?"
            
        elif classified_intent == "exit":
            # User wants to exit
            self.state["conversation_active"] = False
            self._save_session_summary()
            response = "Thank you for chatting with us. Your conversation history has been saved. Have a great day!"
            
        elif classified_intent == "complaint":
            self.state["current_intent"] = "complaint"
            response = self.handle_complaint(user_message)
                
        else:  # qna
            # If switching from complaint to qna, reset complaint handler
            if self.state.get("current_intent") == "complaint":
                if self.complaint_handler:
                    self.complaint_handler.reset()
                    self.complaint_handler = None
            
            self.state["current_intent"] = "qna"
            response = self.handle_qna(user_message)
        
        # Add assistant response to conversation history
        assistant_msg = {"role": "assistant", "content": response}
        self.state["messages"].append(assistant_msg)
        
        # Save state to SQLite
        self._save_state()
        
        return response

    def _save_session_summary(self):
        """Save a summary of the current session."""
        if not self.state["messages"]:
            return
            
        # Create a simple summary
        summary = f"Session with {len(self.state['messages'])} messages, {self.state['total_complaints']} complaints, {self.state['total_qna']} Q&A interactions"
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO conversation_summaries 
            (thread_id, session_date, summary, complaint_count, qna_count)
            VALUES (?, ?, ?, ?, ?)
        """, (
            self.thread_id,
            datetime.now().strftime("%Y-%m-%d"),
            summary,
            self.state["total_complaints"],
            self.state["total_qna"]
        ))
        
        conn.commit()
        conn.close()

    def get_conversation_history_summary(self) -> Dict[str, Any]:
        """Get a summary of all conversation history."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get session summaries
        cursor.execute("""
            SELECT session_date, summary, complaint_count, qna_count, created_at
            FROM conversation_summaries 
            WHERE thread_id = ?
            ORDER BY created_at DESC
            LIMIT 10
        """, (self.thread_id,))
        
        sessions = cursor.fetchall()
        
        # Get total counts
        cursor.execute("""
            SELECT 
                COUNT(*) as total_sessions,
                SUM(complaint_count) as total_complaints,
                SUM(qna_count) as total_qna
            FROM conversation_summaries 
            WHERE thread_id = ?
        """, (self.thread_id,))
        
        totals = cursor.fetchone()
        conn.close()
        
        return {
            "recent_sessions": [
                {
                    "date": row[0],
                    "summary": row[1],
                    "complaints": row[2],
                    "qna": row[3],
                    "timestamp": row[4]
                }
                for row in sessions
            ],
            "total_sessions": totals[0] or 0,
            "total_complaints": totals[1] or 0,
            "total_qna": totals[2] or 0,
            "complaint_history": self._get_complaint_history()
        }

    def reset_conversation(self):
        """Reset the current conversation state but preserve history."""
        if self.complaint_handler:
            self.complaint_handler.reset()
        self.complaint_handler = None
        
        # Save current session before reset
        self._save_session_summary()
        
        # Reset to default state
        self.state = self._get_default_state()
        self._save_state()

    def get_conversation_state(self) -> Dict[str, Any]:
        """Get current conversation state for debugging/monitoring."""
        return {
            "current_intent": self.state.get("current_intent"),
            "message_count": len(self.state.get("messages", [])),
            "complaint_active": self.complaint_handler is not None,
            "complaint_logged": (self.complaint_handler.state.get("complaint_logged") 
                               if self.complaint_handler else None),
            "conversation_active": self.state.get("conversation_active", True),
            "session_start": self.state.get("session_start"),
            "total_complaints": self.state.get("total_complaints", 0),
            "total_qna": self.state.get("total_qna", 0),
            "thread_id": self.thread_id
        }

    def is_conversation_active(self) -> bool:
        """Check if conversation is still active."""
        return self.state.get("conversation_active", True)