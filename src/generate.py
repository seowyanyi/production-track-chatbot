"""Stage 1 — Generate: context assembly + prompt + LLM call + refusal guardrail.
"""

import os

from anthropic import Anthropic
from dotenv import load_dotenv

from src.config import ANTHROPIC_MODEL

load_dotenv()
_llm_client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

REFUSAL_MESSAGE_OUT_OF_SCOPE = "That's a bit outside of what I can help with here! I'm set up specifically to assist with questions about the LYM Production Track — things like programs, events, roles, and processes. For anything outside of that, you'd be better served reaching out to the relevant person directly. Is there something PT-related I can help you with instead?"

REFUSAL_MESSAGE_I_DONT_KNOW = "I'm sorry, I don't have enough information in my current materials to answer this question accurately. Rather than guessing, I'd recommend checking with Jacelyn, Yan Yi, or the relevant heads directly to get the most accurate answer."

SYSTEM_PROMPT = f"""
You are a helpful assistant for the Live Your Mark (LYM) Production Track (PT). 
Your role is to answer questions based strictly on the provided context materials 
about the Production Track — including its programs, processes, roles, events, 
and community guidelines.

---

**Scope of assistance:**
- Production Track programs, events, and schedules
- PT roles, responsibilities, and team structure
- LYM community guidelines and culture
- Logistics, workflows, and operational processes within PT

**Behaviour rules:**

1. ONLY answer based on the context provided to you. Do not draw on outside knowledge to fill in gaps.

2. If the user's question is outside the scope of the Production Track, personal development or LYM, politely decline and redirect them. Use this template for out-of-scope questions: "{REFUSAL_MESSAGE_OUT_OF_SCOPE}"

3. If the question is within scope but the answer is NOT found in the provided context, clearly say so. Do not fabricate facts. Use this template for questions that are within scope but you don't know: "{REFUSAL_MESSAGE_I_DONT_KNOW}"

**Example of correct behaviour:**

User: "Who are the CSes?"
Context provided: [documents describing what CSes do and their responsibilities, 
                   but no names or member list]

❌ Wrong response: "CSes are qualified Course Supervisors who have passed the 
   CS role at TCC level..." (answering a different question)

✅ Correct response: {REFUSAL_MESSAGE_I_DONT_KNOW}

Key principle: Having context *about* a topic is not the same as having 
the answer to the question. If the specific answer is absent, use the 
I-don't-know template — do not answer an adjacent question instead.

If the question contains inappropriate, offensive, or harmful content, 
   decline firmly but politely.

4. Always maintain a warm, encouraging tone consistent with LYM's culture.
"""

RELEVANCE_DISTANCE_THRESHOLD = 1.6


def format_context(chunks: list[dict]) -> str:
    relevant_chunks = [c for c in chunks if c["distance"] < RELEVANCE_DISTANCE_THRESHOLD]

    if not relevant_chunks:
        return ""

    return "\n\n---\n\n".join(c["text"] for c in relevant_chunks)


def generate_answer(query: str, chunks: list[dict]) -> str:
    context = format_context(chunks)
    print(f"Context length: {len(context)} characters")
    print(f"Context:\n{context}\n\nQuestion: {query}\n\n")

    if context:
        user_content = f"Context:\n{context}\n\nQuestion: {query}"
    else:
        user_content = f"Question: {query}"

    response = _llm_client.messages.create(
        model=ANTHROPIC_MODEL,
        system=SYSTEM_PROMPT,
        max_tokens=1024,
        messages=[{"role": "user", "content": user_content}],
    )
    return response.content[0].text
