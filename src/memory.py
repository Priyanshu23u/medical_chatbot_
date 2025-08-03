# src/memory.py

from langchain_core.messages import HumanMessage, AIMessage

# Global history storage
_history = []

def get_chat_history():
    """Return history in LangChain format"""
    return ChatHistoryWrapper(_history)

def add_to_history(user_input, bot_output):
    _history.append(HumanMessage(content=user_input))
    _history.append(AIMessage(content=bot_output))

def clear_history():
    _history.clear()

class ChatHistoryWrapper:
    """Helper to make .messages available"""
    def __init__(self, messages):
        self.messages = messages
