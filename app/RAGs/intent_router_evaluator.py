"""
Comprehensive evaluation suite for IntentRouter system using Confident AI/DeepEval.
Fixed version with proper database initialization and authentication handling.
"""

import os
import json
import sqlite3
from typing import List, Dict, Any, Tuple
from datetime import datetime

import deepeval
from deepeval.test_case import LLMTestCase
from deepeval.metrics import GEval, AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.dataset import EvaluationDataset

# Import your IntentRouter (adjust path as needed)
from chatbot.intent_classifier import IntentRouter


class IntentRouterEvaluator:
    """Comprehensive evaluator for IntentRouter system using Confident AI."""
    
    def __init__(self, confident_api_key: str):
        """Initialize evaluator with Confident AI credentials."""
        os.environ['CONFIDENT_API_KEY'] = confident_api_key
        
        # Set up Confident AI authentication
        try:
            # Try the newer authentication method
            if hasattr(deepeval, 'set_confidence_api_key'):
                deepeval.set_confidence_api_key(confident_api_key)
            elif hasattr(deepeval, 'login'):
                # Older method - try with api_key parameter
                deepeval.login(api_key=confident_api_key)
        except Exception as e:
            print(f"⚠️ Authentication warning: {e}")
            print("Continuing with environment variable authentication")
        
        # Initialize router for testing with properly initialized in-memory DB
        self.router = self._create_initialized_router(":memory:", "eval_session")
        
        # Define custom evaluation metrics
        self._setup_custom_metrics()
        
        # Test datasets
        self.test_cases = self._create_test_cases()
    
    def _create_initialized_router(self, db_path: str, thread_id: str) -> IntentRouter:
        """Create an IntentRouter with properly initialized database tables."""
        
        # First create the database connection and initialize tables
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Create required tables
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS complaints_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                thread_id TEXT NOT NULL,
                complaint_summary TEXT NOT NULL,
                complaint_type TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS conversation_state (
                thread_id TEXT PRIMARY KEY,
                current_intent TEXT,
                complaint_in_progress BOOLEAN DEFAULT FALSE,
                complaint_details TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
        
        # Now create the router
        return IntentRouter(db_path=db_path, thread_id=thread_id)
    
    def _setup_custom_metrics(self):
        """Setup custom evaluation metrics for intent classification."""
        
        # Intent Classification Accuracy Metric
        self.intent_accuracy = GEval(
            name="Intent Classification Accuracy",
            criteria="""
            Evaluate if the classified intent correctly matches the user's actual need:
            - Score 1.0: Perfect classification that matches user intent exactly
            - Score 0.8-0.9: Mostly correct but minor interpretation differences  
            - Score 0.5-0.7: Somewhat correct but missing nuances
            - Score 0.2-0.4: Poor classification that misunderstands intent
            - Score 0.0-0.1: Completely wrong classification
            """,
            evaluation_params=["input", "actual_output", "expected_output"],
            evaluation_steps=[
                "Analyze the user's message to understand their true intent",
                "Compare the classified intent with the expected intent",
                "Consider context and conversation history if provided",
                "Determine if the classification would lead to appropriate handling"
            ]
        )
        
        # Context Awareness Metric
        self.context_awareness = GEval(
            name="Context Awareness",
            criteria="""
            Evaluate how well the system uses conversation history and context:
            - Score 1.0: Perfect use of context, references previous interactions appropriately
            - Score 0.8-0.9: Good context usage with minor gaps
            - Score 0.5-0.7: Some context awareness but misses important details
            - Score 0.2-0.4: Poor context usage, treats each message in isolation
            - Score 0.0-0.1: No context awareness, ignores conversation history
            """,
            evaluation_params=["input", "actual_output", "retrieval_context"],
            evaluation_steps=[
                "Check if previous conversation context is relevant",
                "Evaluate if the classification considers conversation history",
                "Assess if complaint history influences current classification",
                "Determine if session state affects routing decisions"
            ]
        )
        
        # Response Appropriateness Metric
        self.response_appropriateness = GEval(
            name="Response Appropriateness",
            criteria="""
            Evaluate if the final response is appropriate for the classified intent:
            - Score 1.0: Response perfectly matches intent and user needs
            - Score 0.8-0.9: Response is appropriate with minor issues
            - Score 0.5-0.7: Response somewhat fits but has notable problems
            - Score 0.2-0.4: Response poorly matches the intent
            - Score 0.0-0.1: Response completely inappropriate for the intent
            """,
            evaluation_params=["input", "actual_output", "expected_output"],
            evaluation_steps=[
                "Analyze if response addresses user's actual need",
                "Check if response tone matches intent type",
                "Evaluate if response provides appropriate next steps",
                "Assess overall user experience quality"
            ]
        )
    
    def _create_test_cases(self) -> List[LLMTestCase]:
        """Create comprehensive test cases for all intent types."""
        
        test_cases = []
        
        # COMPLAINT Intent Test Cases
        complaint_cases = [
            ("My order #12345 arrived damaged", "complaint", "Should route to complaint handler"),
            ("I've been waiting 2 weeks for my delivery", "complaint", "Delivery complaint"),
            ("The foundation I bought doesn't match my skin tone", "complaint", "Product quality issue"),
            ("I was charged twice for the same order", "complaint", "Billing complaint"),
            ("yes, that's exactly my complaint", "complaint", "Confirmation of complaint summary"),
            ("Can you update me on my previous complaint?", "complaint", "Following up on existing complaint"),
        ]
        
        # QNA Intent Test Cases  
        qna_cases = [
            ("What's the best moisturizer for dry skin?", "qna", "Product recommendation query"),
            ("Hello, how are you today?", "qna", "Greeting/casual conversation"),
            ("Can you recommend a good primer?", "qna", "General product question"),
            ("What's your return policy?", "qna", "Policy information request"),
            ("Tell me about your skincare routine tips", "qna", "General information request"),
            ("How do I apply concealer properly?", "qna", "How-to question"),
        ]
        
        # ABORT Intent Test Cases
        abort_cases = [
            ("Actually, never mind about the complaint", "abort", "User canceling complaint"),
            ("Stop, I don't want to file a complaint anymore", "abort", "Explicit stop request"),
            ("Forget it, let's talk about something else", "abort", "Switching away from complaint"),
            ("Cancel that", "abort", "Simple cancellation"),
            ("I changed my mind", "abort", "Mind change indication"),
        ]
        
        # RESET Intent Test Cases
        reset_cases = [
            ("Can we start over?", "reset", "Request to start fresh"),
            ("Reset this conversation", "reset", "Explicit reset request"),
            ("Clear our chat history", "reset", "History clearing request"),
            ("Let's begin again", "reset", "Fresh start request"),
            ("Start from the beginning", "reset", "Reset to initial state"),
        ]
        
        # EXIT Intent Test Cases
        exit_cases = [
            ("Goodbye", "exit", "Polite farewell"),
            ("I'm done, thanks", "exit", "Ending conversation"),
            ("Bye bye", "exit", "Casual goodbye"),
            ("Exit", "exit", "Direct exit command"),
            ("That's all, have a nice day", "exit", "Ending with pleasantries"),
        ]
        
        # Combine all test cases
        all_cases = [
            ("complaint", complaint_cases),
            ("qna", qna_cases), 
            ("abort", abort_cases),
            ("reset", reset_cases),
            ("exit", exit_cases)
        ]
        
        for intent_type, cases in all_cases:
            for user_input, expected_intent, description in cases:
                test_cases.append(
                    LLMTestCase(
                        input=user_input,
                        expected_output=expected_intent,
                        context=[description],
                        retrieval_context=[f"Testing {intent_type} intent classification"]
                    )
                )
        
        return test_cases
    
    def _create_contextual_test_cases(self) -> List[LLMTestCase]:
        """Create test cases that require conversation context."""
        
        contextual_cases = []
        
        # Test case 1: Following up on a complaint
        contextual_cases.append(
            LLMTestCase(
                input="Is there any update on that?",
                expected_output="complaint",
                context=["Previous message: 'My order arrived damaged'"],
                retrieval_context=["User previously filed a complaint about damaged order"]
            )
        )
        
        # Test case 2: Confirming complaint details
        contextual_cases.append(
            LLMTestCase(
                input="Yes, that's correct",
                expected_output="complaint", 
                context=["System summarized complaint details for confirmation"],
                retrieval_context=["Complaint handler provided summary, awaiting confirmation"]
            )
        )
        
        # Test case 3: Switching from complaint to general Q&A
        contextual_cases.append(
            LLMTestCase(
                input="Actually, can you recommend a good cleanser?",
                expected_output="qna",
                context=["User was in middle of filing complaint"],
                retrieval_context=["User switching from complaint to product question"]
            )
        )
        
        return contextual_cases
    
    def evaluate_intent_classification(self) -> Dict[str, Any]:
        """Evaluate intent classification accuracy."""
        
        print("🧪 Running Intent Classification Evaluation...")
        
        # Test intent classification only
        classification_results = []
        
        for test_case in self.test_cases:
            try:
                # Create a fresh router for each test with properly initialized DB
                test_router = self._create_initialized_router(":memory:", "test")
                
                # Classify intent
                messages = [{"role": "user", "content": test_case.input}]
                classified_intent = test_router.classify_intent(messages)
                
                # Create test case with actual output
                evaluated_case = LLMTestCase(
                    input=test_case.input,
                    actual_output=classified_intent,
                    expected_output=test_case.expected_output,
                    context=test_case.context,
                    retrieval_context=test_case.retrieval_context
                )
                
                classification_results.append(evaluated_case)
                
            except Exception as e:
                print(f"Error testing case '{test_case.input}': {e}")
                continue
        
        if not classification_results:
            print("⚠️ No test cases were successfully processed")
            return {"error": "No test cases processed"}
        
        # Evaluate using custom metrics
        try:
            # Use newer API format
            dataset = EvaluationDataset(test_cases=classification_results)
            results = deepeval.evaluate(
                test_cases=dataset.test_cases,
                metrics=[self.intent_accuracy, self.context_awareness]
            )
        except Exception as e:
            print(f"Error during evaluation: {e}")
            # Fallback: manual evaluation
            results = self._manual_evaluation(classification_results)
        
        return results
    
    def _manual_evaluation(self, test_cases: List[LLMTestCase]) -> Dict[str, Any]:
        """Manual fallback evaluation when DeepEval fails."""
        results = []
        for case in test_cases:
            score = 1.0 if case.actual_output == case.expected_output else 0.0
            results.append({
                "input": case.input,
                "actual_output": case.actual_output,
                "expected_output": case.expected_output,
                "score": score,
                "success": score >= 0.8
            })
        
        accuracy = sum(1 for r in results if r["success"]) / len(results) if results else 0
        return {
            "test_results": results,
            "metrics": {
                "accuracy": accuracy,
                "total_tests": len(results)
            }
        }
    
    def evaluate_end_to_end_flow(self) -> Dict[str, Any]:
        """Evaluate complete end-to-end system performance."""
        
        print("🔄 Running End-to-End Flow Evaluation...")
        
        e2e_results = []
        
        # Test complete message processing flow
        flow_test_cases = [
            {
                "input": "My lipstick arrived broken",
                "expected_intent": "complaint",
                "expected_response_type": "complaint_handling"
            },
            {
                "input": "What's a good mascara for sensitive eyes?", 
                "expected_intent": "qna",
                "expected_response_type": "product_recommendation"
            },
            {
                "input": "Never mind, I don't want to complain",
                "expected_intent": "abort", 
                "expected_response_type": "abort_confirmation"
            }
        ]
        
        for case in flow_test_cases:
            try:
                # Create fresh router with initialized DB
                test_router = self._create_initialized_router(":memory:", "e2e_test")
                
                # Process complete message flow
                response = test_router.process_message(case["input"])
                
                # Get the classified intent from router state
                classified_intent = test_router.state.get("current_intent", "unknown")
                
                e2e_case = LLMTestCase(
                    input=case["input"],
                    actual_output=response,
                    expected_output=case["expected_response_type"],
                    context=[f"Classified as: {classified_intent}"],
                    retrieval_context=[f"Expected intent: {case['expected_intent']}"]
                )
                
                e2e_results.append(e2e_case)
                
            except Exception as e:
                print(f"Error in E2E test for '{case['input']}': {e}")
                continue
        
        if not e2e_results:
            print("⚠️ No E2E test cases were successfully processed")
            return {"error": "No E2E test cases processed"}
        
        # Evaluate with response appropriateness metric
        try:
            dataset = EvaluationDataset(test_cases=e2e_results)
            results = deepeval.evaluate(
                test_cases=dataset.test_cases,
                metrics=[self.response_appropriateness, AnswerRelevancyMetric()]
            )
        except Exception as e:
            print(f"Error during E2E evaluation: {e}")
            results = self._manual_evaluation(e2e_results)
        
        return results
    
    def evaluate_conversation_memory(self) -> Dict[str, Any]:
        """Evaluate how well the system maintains conversation context."""
        
        print("🧠 Running Conversation Memory Evaluation...")
        
        memory_results = []
        
        # Multi-turn conversation scenarios
        conversation_scenarios = [
            {
                "turns": [
                    ("My order is late", "complaint"),
                    ("It was supposed to arrive yesterday", "complaint"), 
                    ("What's the status?", "complaint")
                ],
                "description": "Multi-turn complaint conversation"
            },
            {
                "turns": [
                    ("Hi there", "qna"),
                    ("I need skincare advice", "qna"),
                    ("Actually, I want to complain about a product", "complaint")
                ],
                "description": "Intent switching conversation"
            }
        ]
        
        for scenario in conversation_scenarios:
            try:
                # Create router for scenario with initialized DB
                test_router = self._create_initialized_router(":memory:", "memory_test")
                
                conversation_context = []
                
                for turn_input, expected_intent in scenario["turns"]:
                    # Process message
                    response = test_router.process_message(turn_input)
                    
                    # Get classified intent from state
                    classified_intent = test_router.state.get("current_intent", "unknown")
                    
                    conversation_context.append(f"User: {turn_input} | Classified: {classified_intent}")
                    
                    # Create test case for this turn
                    memory_case = LLMTestCase(
                        input=turn_input,
                        actual_output=classified_intent,
                        expected_output=expected_intent,
                        context=conversation_context.copy(),
                        retrieval_context=[scenario["description"]]
                    )
                    
                    memory_results.append(memory_case)
                    
            except Exception as e:
                print(f"Error in memory test for scenario '{scenario['description']}': {e}")
                continue
        
        if not memory_results:
            print("⚠️ No memory test cases were successfully processed")
            return {"error": "No memory test cases processed"}
        
        # Evaluate with context awareness metric
        try:
            dataset = EvaluationDataset(test_cases=memory_results)
            results = deepeval.evaluate(
                test_cases=dataset.test_cases,
                metrics=[self.context_awareness, self.intent_accuracy]
            )
        except Exception as e:
            print(f"Error during memory evaluation: {e}")
            results = self._manual_evaluation(memory_results)
        
        return results
    
    def evaluate_edge_cases(self) -> Dict[str, Any]:
        """Evaluate system performance on edge cases and ambiguous inputs."""
        
        print("⚠️ Running Edge Case Evaluation...")
        
        edge_cases = [
            # Ambiguous cases
            ("I have a question about my order", "complaint", "Could be complaint or general question"),
            ("Thanks", "qna", "Gratitude - should stay in current flow"),
            ("ok", "qna", "Acknowledgment - context dependent"),
            
            # Mixed intent cases
            ("I want to complain but also ask about products", "complaint", "Multiple intents - complaint takes priority"),
            ("Cancel my complaint and tell me about moisturizers", "abort", "Abort + new question"),
            
            # Spelling/grammar issues
            ("my oder is broekn", "complaint", "Misspelled complaint"),
            ("wat r ur best prodcuts?", "qna", "Informal spelling in question"),
            
            # Empty/minimal input
            ("", "qna", "Empty input"),
            ("?", "qna", "Just question mark"),
            ("help", "qna", "Generic help request"),
        ]
        
        edge_results = []
        
        for user_input, expected_intent, description in edge_cases:
            try:
                test_router = self._create_initialized_router(":memory:", "edge_test")
                
                # Test classification
                messages = [{"role": "user", "content": user_input}]
                classified_intent = test_router.classify_intent(messages)
                
                edge_case = LLMTestCase(
                    input=user_input,
                    actual_output=classified_intent,
                    expected_output=expected_intent,
                    context=[description],
                    retrieval_context=[f"Edge case: {description}"]
                )
                
                edge_results.append(edge_case)
                
            except Exception as e:
                print(f"Error testing edge case '{user_input}': {e}")
                continue
        
        if not edge_results:
            print("⚠️ No edge test cases were successfully processed")
            return {"error": "No edge test cases processed"}
        
        try:
            dataset = EvaluationDataset(test_cases=edge_results)
            results = deepeval.evaluate(
                test_cases=dataset.test_cases,
                metrics=[self.intent_accuracy]
            )
        except Exception as e:
            print(f"Error during edge case evaluation: {e}")
            results = self._manual_evaluation(edge_results)
        
        return results
    
    def run_comprehensive_evaluation(self) -> Dict[str, Any]:
        """Run all evaluation tests and generate comprehensive report."""
        
        print("🚀 Starting Comprehensive IntentRouter Evaluation")
        print("=" * 60)
        
        evaluation_results = {}
        
        try:
            # 1. Basic Intent Classification
            print("\n1️⃣ Intent Classification Evaluation")
            evaluation_results["intent_classification"] = self.evaluate_intent_classification()
            
            # 2. End-to-End Flow
            print("\n2️⃣ End-to-End Flow Evaluation") 
            evaluation_results["end_to_end"] = self.evaluate_end_to_end_flow()
            
            # 3. Conversation Memory
            print("\n3️⃣ Conversation Memory Evaluation")
            evaluation_results["conversation_memory"] = self.evaluate_conversation_memory()
            
            # 4. Edge Cases
            print("\n4️⃣ Edge Case Evaluation")
            evaluation_results["edge_cases"] = self.evaluate_edge_cases()
            
            # Generate summary report
            summary = self._generate_evaluation_summary(evaluation_results)
            evaluation_results["summary"] = summary
            
            print("\n" + "=" * 60)
            print("✅ Evaluation Complete!")
            print(f"📊 View detailed results on Confident AI dashboard")
            
            return evaluation_results
            
        except Exception as e:
            print(f"❌ Evaluation failed: {e}")
            return {"error": str(e)}
    
    def _generate_evaluation_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a summary of all evaluation results."""
        
        summary = {
            "timestamp": datetime.now().isoformat(),
            "total_test_cases": 0,
            "metrics_summary": {},
            "recommendations": []
        }
        
        # Count total test cases and calculate accuracy
        for category, result in results.items():
            if isinstance(result, dict) and "metrics" in result:
                summary["total_test_cases"] += result["metrics"].get("total_tests", 0)
                summary["metrics_summary"][category] = result["metrics"]
        
        # Add recommendations based on performance
        summary["recommendations"] = [
            "Monitor intent classification accuracy in production",
            "Add more edge case handling for ambiguous inputs",
            "Implement conversation context weighting for better classification",
            "Consider adding confidence scores to intent predictions",
            "Set up continuous evaluation pipeline for new test cases"
        ]
        
        return summary
    
    def save_evaluation_config(self, filepath: str = "evaluation_config.json"):
        """Save evaluation configuration for future use."""
        
        config = {
            "metrics": {
                "intent_accuracy": {
                    "name": "Intent Classification Accuracy",
                    "type": "G-Eval",
                    "criteria": "Evaluate intent classification correctness"
                },
                "context_awareness": {
                    "name": "Context Awareness", 
                    "type": "G-Eval",
                    "criteria": "Evaluate conversation context usage"
                },
                "response_appropriateness": {
                    "name": "Response Appropriateness",
                    "type": "G-Eval", 
                    "criteria": "Evaluate response quality for intent"
                }
            },
            "test_categories": [
                "intent_classification",
                "end_to_end_flow", 
                "conversation_memory",
                "edge_cases"
            ],
            "intents": ["complaint", "qna", "abort", "reset", "exit"]
        }
        
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"💾 Evaluation configuration saved to {filepath}")


def main():
    """Main function to run the evaluation."""
    
    # Get API key from environment
    api_key = os.getenv('CONFIDENT_API_KEY')
    if not api_key:
        print("❌ Please set CONFIDENT_API_KEY environment variable")
        print("Example: export CONFIDENT_API_KEY='your_api_key_here'")
        return
    
    try:
        # Initialize evaluator
        evaluator = IntentRouterEvaluator(api_key)
        
        # Run comprehensive evaluation
        results = evaluator.run_comprehensive_evaluation()
        
        # Save configuration
        evaluator.save_evaluation_config()
        
        # Save results to file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = f"evaluation_results_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n📋 Results saved to: {results_file}")
        print("📊 Check Confident AI dashboard for detailed analysis.")
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        print("💡 Try running: pip install --upgrade deepeval")
        print("💡 Or check your CONFIDENT_API_KEY is valid")


if __name__ == "__main__":
    main()


# Additional utility functions for ongoing evaluation

def quick_intent_test(user_message: str, api_key: str = None) -> Dict[str, Any]:
    """Quick test of a single user message."""
    
    if not api_key:
        api_key = os.getenv('CONFIDENT_API_KEY')
    
    evaluator = IntentRouterEvaluator(api_key)
    
    # Test single message
    test_router = evaluator._create_initialized_router(":memory:", "quick_test")
    messages = [{"role": "user", "content": user_message}]
    classified_intent = test_router.classify_intent(messages)
    
    response = test_router.process_message(user_message)
    
    return {
        "input": user_message,
        "classified_intent": classified_intent,
        "response": response,
        "router_state": test_router.get_conversation_state()
    }