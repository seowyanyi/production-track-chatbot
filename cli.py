"""Stage 1 CLI — the runnable end-to-end slice.

    uv run python cli.py "Who sets up the registration desk?"

Run order:
    1. uv run python -m src.ingest      (build the index)
    2. uv run python cli.py "..."       (ask a question)
"""

import sys

from src.generate import generate_answer
from src.retrieve import retrieve


def main() -> None:
    if len(sys.argv) < 2:
        print('Usage: uv run python cli.py "your question"')
        sys.exit(1)

    query = " ".join(sys.argv[1:])
    chunks = retrieve(query)
    answer = generate_answer(query, chunks)

    print(answer)


if __name__ == "__main__":
    main()
