# CRSEC/reverie/backend_server/test_router.py
from llm_router import llm_call
from norm.schemas import NormEntry, NormUtilityRating

# Test 1: Basic call
print("=== Test 1: Basic call ===")
r = llm_call("Say hello in one sentence.", call_type="default")
print(r)

# Test 2: Structured output (this is the critical one)
print("\n=== Test 2: Structured JSON output ===")
r = llm_call(
    "Rate this norm: 'No smoking indoors'. Score 1-100 with reason.",
    call_type="norm_evaluation",
    json_schema=NormUtilityRating.model_json_schema()
)
print(r)
parsed = NormUtilityRating.model_validate_json(r)
print(f"Score: {parsed.score}, Reason: {parsed.reason}")

# Test 3: Two-tier routing (check logs)
print("\n=== Test 3: Model routing ===")
llm_call("test simple", call_type="format_check")
llm_call("test complex", call_type="defection_assessment")

import json
with open("llm_logs/calls.jsonl") as f:
    lines = f.readlines()[-2:]
    for l in lines:
        entry = json.loads(l)
        print(f"  call_type={entry['call_type']} → model={entry['model']}")

print("\nAll tests passed!")