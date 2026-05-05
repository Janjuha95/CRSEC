"""
LLM Router — routes calls to the appropriate Ollama model based on call_type.
"""
import os
import json
import datetime
import ollama

PRIMARY_MODEL = "qwen3:8b"
REASONING_MODEL = "qwen3:8b"

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

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "llm_logs")
LOG_FILE = os.path.join(LOG_DIR, "calls.jsonl")


def _get_model(call_type: str) -> str:
    if call_type in REASONING_CALL_TYPES:
        return REASONING_MODEL
    if call_type in PRIMARY_CALL_TYPES:
        return REASONING_MODEL
    return REASONING_MODEL


def _log_call(model: str, call_type: str, prompt: str, response: str, schema_enforced: bool):
    os.makedirs(LOG_DIR, exist_ok=True)
    entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "model": model,
        "call_type": call_type,
        "prompt": prompt,
        "response": response,
        "schema_enforced": schema_enforced,
    }
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def llm_call(prompt: str, call_type: str, json_schema: dict = None) -> str:
    """
    Route a prompt to the correct Ollama model based on call_type.
    If json_schema is provided, it is passed as format= to enforce structured output.
    Returns the response content as a string.
    """
    model = _get_model(call_type)
    messages = [{"role": "user", "content": prompt}]

    kwargs = {"model": model, "messages": messages}
    if json_schema is not None:
        kwargs["format"] = json_schema

    result = ollama.chat(**kwargs)
    content = result["message"]["content"]

    _log_call(model, call_type, prompt, content, json_schema is not None)
    return content


def openai_compat_call(prompt: str, call_type: str = "conversation", json_schema: dict = None) -> str:
    """
    Drop-in wrapper that mirrors the old OpenAI-style interface.
    Calls llm_call and returns the string response.
    """
    return llm_call(prompt, call_type, json_schema)
