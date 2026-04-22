# CRSEC/reverie/backend_server/test_crsec_prompts.py
from llm_router import llm_call
import json

# Test norm creation (the most complex structured output)
print("=== Norm Creation Prompt ===")
with open('./norm/creation_prompt/sys_prompt.txt') as f:
    sys_prompt = f.read()
with open('./norm/creation_prompt/usr_prompt_v6.txt') as f:
    usr_prompt = f.read()

# Use Abigail Chen's description as test input
with open('../../environment/frontend_server/storage/base_ville_n10_with_norm/personas/Abigail Chen/bootstrap_memory/scratch.json') as f:
    agent_desc = f.read()

response = llm_call(sys_prompt + "\n" + usr_prompt + "\n" + agent_desc,
                     call_type="norm_creation")
print(response[:500])

# Validate it's parseable JSON with 5 norms
try:
    norms = json.loads(response)
    assert "norm_1" in norms, "Missing norm_1"
    assert "norm_5" in norms, "Missing norm_5"
    for i in range(1, 6):
        n = norms[f"norm_{i}"]
        assert "content" in n and "type" in n and "utility" in n
        print(f"  norm_{i}: [{n['type']}] {n['content']} (utility={n['utility']})")
    print("PASS: Norm creation works!")
except Exception as e:
    print(f"FAIL: {e}")
    print("Raw response:", response[:300])

# Test conflict detection
print("\n=== Conflict Detection Prompt ===")
from persona.prompt_template.gpt_structure import generate_prompt
prompt_input = [
    "Carlos Gomez is smoking a cigarette inside the cafe",  # target desc
    "No smoking is allowed inside the cafe.",                # norms
    "entrepreneur",                                          # identity
    "passionate, organized, community-minded",               # innate
    "Carlos Gomez"                                           # target name
]
prompt = generate_prompt(prompt_input, 
    "norm/norm_retrieve_prompt/check_conflict_decide_talk_v5.txt")
response = llm_call(prompt, call_type="conflict_detection")
print(response[:500])
has_final = "FINAL OUTPUT" in response.upper() or "final output" in response.lower()
print(f"{'PASS' if has_final else 'WARN'}: Contains FINAL OUTPUT marker: {has_final}")

# Test norm utility rating
print("\n=== Norm Utility Rating ===")
prompt = generate_prompt(["No one is allowed to smoke inside the cafe."],
    "norm/norm_evaluate_prompt/specific_norm_utility_v2.txt")
response = llm_call(prompt, call_type="norm_evaluation")
print(response)
print("PASS" if any(c.isdigit() for c in response) else "WARN: No score found")