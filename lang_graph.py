from dotenv import load_dotenv
import os
import json
from typing import Annotated, TypedDict, List, Dict, Any, Literal
from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, FunctionMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import Tool
from operator import itemgetter

# Load environment variables
load_dotenv()

# Define the state of our graph
class State(TypedDict):
    """The state of our graph."""
    messages: Annotated[list, add_messages]
    next: str

# Initialize tools and models
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.7,
)

# Create search tool
search_tool = TavilySearchResults()

# System message to give context to the LLM
SYSTEM_MESSAGE = """You are a helpful AI assistant with access to a search engine.
When you need to answer questions about current events or specific topics, you can use the search tool.
To use the search tool, you should specify a tool call in your response.
Always be truthful and if you don't know something, say so.

Available tools:
- tavily_search_results_json: Search the internet for current information
  Args: {"query": "your search query here"}
"""

class BasicToolNode:
    """A node that runs the tools requested in the last AIMessage."""

    def __init__(self, tools: list) -> None:
        self.tools_by_name = {tool.name: tool for tool in tools}

    def __call__(self, state: State) -> Dict[str, Any]:
        """Process the last message and run any requested tools."""
        if not state["messages"]:
            return {"next": "end"}
            
        message = state["messages"][-1]
        
        # If no tool calls or not an AI message, return empty
        if not isinstance(message, AIMessage) or not message.additional_kwargs.get("function_call"):
            return {"next": "end", "messages": []}
            
        # Get the function call
        function_call = message.additional_kwargs["function_call"]
        
        # Parse the arguments
        try:
            args = json.loads(function_call["arguments"])
        except:
            args = function_call["arguments"]
            
        # Call the tool
        tool_result = self.tools_by_name[function_call["name"]].invoke(args)
        
        # Return the result and continue to chatbot
        return {
            "messages": [FunctionMessage(content=str(tool_result), name=function_call["name"])],
            "next": "chatbot"
        }

def chatbot(state: State) -> Dict[str, Any]:
    """Generate a response using the LLM."""
    messages = [AIMessage(content=SYSTEM_MESSAGE)]
    messages.extend(state["messages"])
    
    # Generate response
    response = llm.invoke(messages)
    
    # Determine next step based on response
    if response.additional_kwargs.get("function_call"):
        next_step = "tools"
    else:
        next_step = "end"
    
    return {
        "messages": [response],
        "next": next_step
    }

# Create the graph
graph = StateGraph(State)

# Create tool node with our search tool
tool_node = BasicToolNode(tools=[search_tool])

# Add nodes
graph.add_node("chatbot", chatbot)
graph.add_node("tools", tool_node)

# Add edges
graph.add_edge(START, "chatbot")

# Add conditional edges based on next state
graph.add_conditional_edges(
    "chatbot",
    lambda x: x["next"],
    {
        "tools": "tools",
        "end": END,
    },
)

graph.add_conditional_edges(
    "tools",
    lambda x: x["next"],
    {
        "chatbot": "chatbot",
        "end": END,
    },
)

# Compile the graph
app = graph.compile()

def stream_graph_updates(user_input: str):
    """Stream updates from the graph for a given user input."""
    messages = [HumanMessage(content=user_input)]
    for event in app.stream({
        "messages": messages,
        "next": "chatbot"
    }):
        for value in event.values():
            if "messages" in value and value["messages"]:
                for message in value["messages"]:
                    if isinstance(message, AIMessage):
                        print("\nAssistant:", message.content)
                    elif isinstance(message, FunctionMessage):
                        print("\nSearch Results:", message.content)

def main():
    """Main chat loop."""
    print("Welcome to the LangGraph Chatbot with Search Capability!")
    print("Ask me anything - I can search the internet to help answer your questions.")
    print("-" * 50)
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            if user_input.lower() in ["quit", "exit", "q"]:
                print("\nGoodbye!")
                break
            
            if not user_input:
                continue
                
            stream_graph_updates(user_input)
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\nAn error occurred: {str(e)}")
            continue

if __name__ == "__main__":
    main()
