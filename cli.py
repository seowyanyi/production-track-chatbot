# uv run python -m src.ingest  (build the index first)
# uv run python cli.py "Who sets up the registration desk?"

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
