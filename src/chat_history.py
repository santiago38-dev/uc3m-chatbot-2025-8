from langchain_community.chat_message_histories import ChatMessageHistory

_store = {}

def get_session_history(session_id):
    if session_id not in _store:
        _store[session_id] = ChatMessageHistory()
    return _store[session_id]


def clear_session_history(session_id: str) -> bool:
    """
    Clear the chat history for a specific session.

    This is CRITICAL for preventing hallucinations caused by context pollution
    from prior Q&A pairs being injected into new queries.

    Args:
        session_id: The session ID to clear

    Returns:
        True if history was cleared, False if session didn't exist
    """
    if session_id in _store:
        _store[session_id].clear()
        del _store[session_id]
        return True
    return False


def clear_all_sessions() -> int:
    """
    Clear ALL session histories. Use with caution in multi-user environments.

    Returns:
        Number of sessions cleared
    """
    count = len(_store)
    _store.clear()
    return count