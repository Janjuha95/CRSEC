"""
LLM Router — routes calls to the appropriate Ollama model based on call_type.
"""
import os
import json
import datetime
import ollama

# Two-tier routing: primary handles high-volume simple calls,
# reasoning handles critical reasoning. Both currently point at 30b-a3b
# because the reasoning tier model (qwen3:32b dense) isn't pulled yet.
# When you pull qwen3:32b on VSC, flip REASONING_MODEL below.
PRIMARY_MODEL = "qwen3:30b-a3b"
REASONING_MODEL = "qwen3:32b-a3b"  

PRIMARY_CALL_TYPES = {
    "format_check",
    "type_check",
    "duplicate_check",
    "fact_consistency",
}

REASONING_CALL_TYPES = {
    "norm_creation",
    "norm_evaluation",
    "conflict_detection",
    "conversation",
    "defection_assessment",
    "violation_check",
}

# Qwen3 enters thinking mode by default. /no_think disables it via the
# system prompt; think=False on the API call is the belt-and-suspenders.
NO_THINK_SYSTEM = "/no_think"

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "llm_logs")
LOG_FILE = os.path.join(LOG_DIR, "calls.jsonl")


def _get_model(call_type: str) -> str:
    if call_type in REASONING_CALL_TYPES:
        return REASONING_MODEL
    if call_type in PRIMARY_CALL_TYPES:
        return PRIMARY_MODEL
    # Unknown call_type — default to reasoning tier to be safe, but log it.
    print(f"[llm_router] WARNING: unknown call_type '{call_type}', defaulting to REASONING_MODEL")
    return REASONING_MODEL


def _log_call(model, call_type, prompt, response, schema_enforced, error=None):
    os.makedirs(LOG_DIR, exist_ok=True)
    entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "model": model,
        "call_type": call_type,
        "prompt": prompt,
        "response": response,
        "schema_enforced": schema_enforced,
        "error": error,
    }
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def llm_call(prompt: str, call_type: str, json_schema: dict = None, max_retries: int = 2) -> str:
    """
    Route a prompt to the correct Ollama model based on call_type.
    If json_schema is provided, it is passed as format= to enforce structured output.
    Returns the response content as a string.
    """
    model = _get_model(call_type)
    messages = [
        {"role": "system", "content": NO_THINK_SYSTEM},
        {"role": "user", "content": prompt},
    ]

    kwargs = {"model": model, "messages": messages, "think": False}
    if json_schema is not None:
        kwargs["format"] = json_schema

    last_err = None
    for attempt in range(max_retries + 1):
        try:
            result = ollama.chat(**kwargs)
            content = result["message"]["content"]
            _log_call(model, call_type, prompt, content, json_schema is not None)
            return content
        except Exception as e:
            last_err = str(e)
            if attempt < max_retries:
                continue
            _log_call(model, call_type, prompt, "", json_schema is not None, error=last_err)
            raise


def openai_compat_call(prompt: str, call_type: str = "conversation", json_schema: dict = None) -> str:
    """
    Drop-in wrapper that mirrors the old OpenAI-style interface.
    """
    return llm_call(prompt, call_type, json_schema)