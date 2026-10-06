import json
from truthmesh.llm import chat


def extract_claim(question, evidence):
    prompt = f"""
Question:
{question}

Evidence:
{evidence}

Extract the main claim supported by the evidence.

Return ONLY valid JSON:

{{
  "claim": "short factual claim",
  "stance": "SUPPORTS",
  "confidence": 0.0
}}

Rules:
- stance must be SUPPORTS, CONTRADICTS, or NEUTRAL
- confidence must be between 0 and 1
- do not invent information
"""

    result = chat(
        "llama3.1:8b",
        "You extract factual claims from evidence. Never invent facts.",
        prompt
    )

    return json.loads(result)