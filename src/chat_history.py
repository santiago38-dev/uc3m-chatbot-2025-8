from langchain_community.chat_message_histories import ChatMessageHistory

_store = {}

def get_session_history(session_id):
    if session_id not in _store:
        _store[session_id] = ChatMessageHistory()
    return _store[session_id]

def clear_session_history(session_id):
    """Clear the chat history for a specific session."""
    if session_id in _store:
        del _store[session_id]

def clear_all_history():
    """Clear all chat history (useful for testing)."""
    global _store
    _store = {}