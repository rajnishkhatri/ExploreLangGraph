# LangGraph Support Chatbot

A sophisticated chatbot built using LangGraph that demonstrates advanced conversational AI capabilities. This implementation uses OpenAI's GPT-3.5-turbo model and Tavily's search API to provide intelligent responses with real-time web search capabilities.

## Features

- 🔍 Web search integration using Tavily API for current information
- 🧠 Intelligent response generation using GPT-3.5-turbo
- 📊 State management and conversation flow control
- 🛠️ Tool-based architecture for extensibility
- 🔄 Conditional routing between components

## Prerequisites

- Python 3.11 or higher
- OpenAI API key
- Tavily API key
- pipenv (for dependency management)

## Setup

1. Install pipenv if you haven't already:
```bash
pip install pipenv
```

2. Clone the repository:
```bash
git clone https://github.com/rajnishkhatri/ExploreLangGraph.git
cd ExploreLangGraph
```

3. Install dependencies:
```bash
pipenv install
```

4. Set up environment variables:
   - Copy `.env.example` to `.env`
   - Add your API keys to `.env`:
```
OPENAI_API_KEY=your_openai_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
```

5. Activate the virtual environment:
```bash
pipenv shell
```

6. Run the project:
```bash
python lang_graph.py
```

## Project Structure

- `lang_graph.py`: Main application file implementing the chatbot logic
- `Pipfile` & `Pipfile.lock`: Project dependencies and lock file
- `.env.example`: Template for environment variables
- `.gitignore`: Specifies which files Git should ignore

## Architecture

The chatbot is built using LangGraph's state machine architecture:

1. **State Management**: Uses TypedDict for maintaining conversation state
2. **Tool Node**: Implements BasicToolNode for handling search functionality
3. **Routing**: Conditional edges for intelligent flow control
4. **Message Handling**: Supports both AI and Function messages

## Development

This project follows the LangGraph framework and best practices. The implementation is based on LangGraph's official documentation and examples.

### Key Components:

- **ChatGPT Integration**: Uses OpenAI's GPT-3.5-turbo model for natural language understanding and generation
- **Tavily Search**: Integrates web search capabilities for real-time information
- **State Graph**: Implements a directed graph for managing conversation flow
- **Tool System**: Extensible tool-based architecture for adding new capabilities

## Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

## Resources

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/tutorials/introduction/)
- [OpenAI API Documentation](https://platform.openai.com/docs/api-reference)
- [Tavily API Documentation](https://docs.tavily.com/) 