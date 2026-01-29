import json
from typing import Generator

import requests

from src.config import LLM_API_KEY


def call_llm_api(prompt_text: str) -> Generator[str, None, None]:
    """
    Sends the prompt to the LLM via the REST API and streams the result.

    Args:
        prompt_text: The formatted prompt string for the LLM.

    Yields:
        Tokens from the LLM's generated answer.
    """
    LLM_API_URL = "https://yiyuan.tsc.uc3m.es/api/generate"
    API_KEY = LLM_API_KEY

    headers = {
        "Content-Type": "application/json",
        "X-API-KEY": f"{API_KEY}"
    }

    payload = {
        "model": "llama3.1:8b",
        "prompt": prompt_text,
        "stream": True,
        "max_tokens": 1024,      # Consensus: 500 too low, 2048 overkill
        "temperature": 0.1       # Consensus: Low for factual RAG, not 0.0 (repetition risk)
    }

    # Track if we've yielded anything (for error handling)
    yielded_content = False

    try:
        # CRITICAL: Add timeout to prevent hanging forever on unresponsive server
        # connect_timeout=10s, read_timeout=120s (streaming can take time)
        with requests.post(
            LLM_API_URL,
            headers=headers,
            json=payload,
            stream=True,
            timeout=(10, 120)  # (connect_timeout, read_timeout)
        ) as response:
            response.raise_for_status()

            for line in response.iter_lines(decode_unicode=True):
                if line:
                    try:
                        json_data = json.loads(line)
                        if 'response' in json_data:
                            yielded_content = True
                            yield json_data['response']
                    except json.JSONDecodeError as e:
                        print(f"Warning: Could not decode line: {line}. Error: {e}")

    except requests.exceptions.Timeout as e:
        print(f"LLM API timeout: {e}")
        # Only yield error if nothing was streamed yet (avoid garbled response)
        if not yielded_content:
            yield "I'm sorry, the request timed out. Please try again."
    except requests.exceptions.RequestException as e:
        print(f"Error calling LLM API: {e}")
        # Only yield error if nothing was streamed yet (avoid garbled response)
        if not yielded_content:
            yield "I'm sorry, I couldn't get a response from the LLM at this moment."


def call_llm_api_full(prompt_text: str) -> str:
    """Collect all tokens from the streaming API into a single string."""
    return "".join(call_llm_api(prompt_text))