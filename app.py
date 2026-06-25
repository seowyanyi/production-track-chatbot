# uv run chainlit run app.py -w

import chainlit

from src.generate import stream_answer
from src.retrieve import retrieve

WELCOME = (
    "Hi! I'm the Production Track assistant. Ask me about PT roles, "
    "events, processes, or logistics."
)


HISTORY_KEY = "chat_history"


@chainlit.on_chat_start
async def on_chat_start() -> None:
    """Fires once when a new chat session opens."""
    chainlit.user_session.set(HISTORY_KEY, [])
    await chainlit.Message(content=WELCOME).send()


@chainlit.on_message
async def on_message(incoming: chainlit.Message) -> None:
    """Fires on every user message. `incoming.content` is the question."""
    query = incoming.content
    chat_history = chainlit.user_session.get(HISTORY_KEY) or []

    # 1. Retrieve relevant chunks for the current query.
    chunks = retrieve(query)

    # 2. Stream the answer, passing prior turns so the LLM has context.
    reply = chainlit.Message(content="")
    for token in stream_answer(query, chunks, chat_history=chat_history):
        await reply.stream_token(token)
    await reply.send()

    # 3. Append this turn to history for the next message.
    chat_history.append({"role": "user", "content": query})
    chat_history.append({"role": "assistant", "content": reply.content})
    chainlit.user_session.set(HISTORY_KEY, chat_history)
