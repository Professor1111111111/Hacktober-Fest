from truthmesh.llm import chat
import json


def compare_claims(claim_a, claim_b):
    prompt = f"""
Compare these two claims.

Claim A:
{claim_a}

Claim B:
{claim_b}

Return ONLY valid JSON:

{{
    "relationship": "SUPPORTS",
    "confidence": 0.0,
    "reason": "short explanation"
}}

The relationship must be exactly one of:
- SUPPORTS
- CONTRADICTS
- NEUTRAL

Do not invent information.
"""

    result = chat(
        "llama3.1:8b",
        "You compare factual claims strictly using the supplied text.",
        prompt
    )

    return json.loads(result)