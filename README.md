# PT RAG Chatbot

A RAG chatbot built for **Live Your Mark (LYM)**, a personal development community. Volunteers in the Production Track (PT) — the team that runs event logistics and operations — use it to query internal manuals, role guides, and process documentation in natural language.

## What it does

- Answers questions grounded strictly in PT documentation
- Refuses out-of-scope questions (e.g. unrelated topics) with a redirect
- Refuses in-scope questions it can't answer rather than hallucinating
- Maintains multi-turn conversation context across a session
- Streams responses token-by-token in a chat UI

---

## Architecture

```
User query
    │
    ▼
retrieve()          — embed query → cosine search → top-k chunks from Chroma
    │
    ▼
select_relevant_chunks()  — drop chunks above distance threshold (no junk context)
    │
    ▼
stream_answer()     — assemble context + chat history → Claude Sonnet → stream tokens
    │
    ▼
Chainlit UI         — renders streaming reply, maintains session history
```

**Key design choices:**

- **Distance threshold filtering** (`retrieve.py` → `generate.py`): retrieved chunks are filtered by cosine distance before being passed to the LLM. This reduces hallucination from low-relevance context at the cost of sometimes passing an empty context — handled by the guardrail.
- **Two distinct refusal cases**: out-of-scope questions and in-scope-but-unknown questions use different refusal messages.
- **Section-based chunking**: documents are split on markdown headings (`#`), and every chunk carries `doc_name` + `section_heading` in its text and metadata. This gives the LLM and retriever structural context without a fixed token window.

---

## Eval results

Scored with [Ragas](https://docs.ragas.io/) against a hand-written golden set (10 answerable, 4 refusal-expected). Claude acts as the judge LLM; embeddings reuse the same local `all-MiniLM-L6-v2` model as the retriever, so scores reflect the real retrieval space.

| Metric | Score | Layer | Reading |
|---|---|---|---|
| context_recall | **1.000** | retrieval | finds everything needed |
| context_precision | **0.684** | retrieval | drags in junk — the bottleneck |
| faithfulness | **0.953** | generation | rarely hallucinates |
| answer_relevancy | **0.855** | generation | answers address the question |
| refusal_accuracy | **1.000** | guardrail | refuses exactly when it should |

**Bottleneck identified:** `context_precision` of 0.684 means the retriever surfaces relevant chunks but ranks irrelevant ones too highly. The distance threshold partially mitigates this, but reranking or hybrid retrieval is the next step.

---

## Stack

| Concern | Choice |
|---|---|
| Language | Python 3.10+, `uv` |
| LLM | Claude Sonnet (`anthropic` SDK) |
| Embeddings | `all-MiniLM-L6-v2` via `sentence-transformers` (local, no API cost) |
| Vector store | Chroma (persistent local) |
| Framework | LangChain — LCEL and components only, no convenience chains |
| UI | Chainlit (streaming, multi-turn) |
| Eval | Ragas + `langchain-anthropic` |

---

## Repo layout

```
src/
  ingest.py         load → chunk → embed → store
  retrieve.py       query → top-k chunks (with distances)
  generate.py       context assembly, prompt, LLM call, refusal guardrail
  embeddings.py     local sentence-transformer wrapper
  config.py         paths, model name, hyperparams
  eval/
    golden.json     hand-written Q/A/source triples (answerable + refusal)
    harness.py      Ragas runner: faithfulness, relevancy, precision, recall, refusal accuracy
docs/               source markdown files (gitignored — contains internal LYM materials)
app.py              Chainlit entrypoint
cli.py              CLI entrypoint
```

---

## Running it

**Prerequisites:** Python 3.10+, [`uv`](https://github.com/astral-sh/uv), an `ANTHROPIC_API_KEY` in `.env`

```bash
# Install dependencies
uv sync

# 1. Ingest documents (run once, or after docs change)
uv run python -m src.ingest

# 2a. Chat via CLI
uv run python cli.py "What does a Course Supervisor do?"

# 2b. Chat via Chainlit UI
uv run chainlit run app.py

# 3. Run the eval suite
uv run --group eval python -m src.eval.harness
```

`.env` format:
```
ANTHROPIC_API_KEY=sk-ant-...
```

---

## Chunk metadata schema

Every stored chunk carries:

```python
{"source_url": str, "doc_name": str, "section_heading": str, "chunk_index": int}
```
