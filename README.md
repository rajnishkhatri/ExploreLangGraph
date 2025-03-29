# LangGraph Support Chatbot

A sophisticated chatbot built using LangGraph that demonstrates advanced conversational AI capabilities.

## Features

- Web search integration for answering common questions
- State management across conversations
- Human-in-the-loop review for complex queries
- Custom state control
- Conversation path exploration and rewinding

## Setup

1. Install pipenv if you haven't already:
```bash
pip install pipenv
```

2. Install dependencies:
```bash
pipenv install
```

3. Create a `.env` file with your API keys:
```
ANTHROPIC_API_KEY=your_key_here
```

4. Activate the virtual environment:
```bash
pipenv shell
```

5. Run the project:
```bash
python lang_graph.py
```

## Project Structure

- `lang_graph.py`: Main application file
- `Pipfile`: Project dependencies
- `.env`: Environment variables (not tracked in git)

## Development

This project uses Python 3.11 and follows the tutorial from [LangGraph's official documentation](https://langchain-ai.github.io/langgraph/tutorials/introduction/). 