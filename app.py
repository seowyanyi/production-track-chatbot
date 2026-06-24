"""Stage 2 — Chainlit chat UI over the Stage 1 RAG pipeline.

Run:
    uv run chainlit run app.py -w     # -w = auto-reload on save

The pipeline is unchanged: retrieve() -> stream_answer().
This file is ONLY the UI layer.
"""

import chainlit

from src.generate import stream_answer
from src.retrieve import retrieve

WELCOME = (
    "Hi! I'm the Production Track assistant. Ask me about PT roles, "
    "events, processes, or logistics."
)


@chainlit.on_chat_start
async def on_chat_start() -> None:
    """Fires once when a new chat session opens."""
    await chainlit.Message(content=WELCOME).send()


@chainlit.on_message
async def on_message(incoming: chainlit.Message) -> None:
    """Fires on every user message. `incoming.content` is the question."""
    query = incoming.content

    # 1. Retrieve. (Same call your CLI makes.)
    chunks = retrieve(query)

    # 2. Create an EMPTY assistant message we'll fill token-by-token.
    reply = chainlit.Message(content="")

    for token in stream_answer(query, chunks):
        await reply.stream_token(token)
    await reply.send()
