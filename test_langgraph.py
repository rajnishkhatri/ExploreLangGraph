from lang_graph import app
from langchain_core.messages import HumanMessage, AIMessage, FunctionMessage
import json

def test_langgraph_prompt():
    # Test input
    test_input = "What is LangGraph?"
    
    print("Welcome to the LangGraph Chatbot Test!")
    print("-" * 50)
    print("\nTesting with input:", test_input)
    print("-" * 50)
    
    # Initialize messages with the test input
    messages = [HumanMessage(content=test_input)]
    
    # Stream responses
    for event in app.stream({
        "messages": messages,
        "next": "chatbot"
    }):
        for value in event.values():
            if "messages" in value and value["messages"]:
                for message in value["messages"]:
                    if isinstance(message, AIMessage):
                        print("\nAssistant:", message.content)
                        if message.additional_kwargs.get("function_call"):
                            print("\nFunction Call:", json.dumps(message.additional_kwargs["function_call"], indent=2))
                    elif isinstance(message, FunctionMessage):
                        print("\nSearch Results:", message.content)

if __name__ == "__main__":
    test_langgraph_prompt() 