"""Eval harness — score the RAG pipeline against the golden set with Ragas.

    uv run python -m src.eval.harness

What it measures:
  - Ragas metrics (answerable questions only):
      faithfulness        : is the answer grounded in the retrieved context?
      answer_relevancy    : does the answer address the question?
      context_precision   : are the retrieved chunks relevant (ranked well)?
      context_recall      : did retrieval surface what the reference answer needs?
  - refusal_accuracy (all questions): did the bot refuse exactly when it should?
      Out-of-scope and unknown questions *should* refuse; answerable ones
      should NOT. This guards the Stage 1 grounding/refusal behaviour.

The Ragas judge is Claude (via langchain-anthropic); the embeddings reuse the
same local all-MiniLM-L6-v2 model the rest of the pipeline uses, so scores
reflect the real retrieval space.
"""

import json
import os

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.embeddings import Embeddings
from ragas import EvaluationDataset, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

from src.config import ANTHROPIC_MODEL, ROOT
from src.embeddings import embed_texts
from src.generate import (
    REFUSAL_MESSAGE_I_DONT_KNOW,
    REFUSAL_MESSAGE_OUT_OF_SCOPE,
    generate_answer,
    select_relevant_chunks,
)
from src.retrieve import retrieve

load_dotenv()

GOLDEN_PATH = ROOT / "src" / "eval" / "golden.json"

# Question types whose correct behaviour is to refuse rather than answer.
REFUSAL_TYPES = {"out_of_scope", "unknown"}

# Distinctive substrings of the two refusal templates in src/generate.py.
# Matching on these (rather than the full string) tolerates minor LLM rephrasing.
_REFUSAL_SIGNATURES = (
    REFUSAL_MESSAGE_OUT_OF_SCOPE[:40],
    REFUSAL_MESSAGE_I_DONT_KNOW[:40],
)

RAGAS_METRICS = [faithfulness, answer_relevancy, context_precision, context_recall]


class LocalEmbeddings(Embeddings):
    """LangChain Embeddings adapter over the project's local embed_texts().

    Lets Ragas score in the same vector space the retriever uses, with no
    extra model download or API calls.
    """

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [vector.tolist() for vector in embed_texts(texts)]

    def embed_query(self, text: str) -> list[float]:
        return embed_texts([text])[0].tolist()


def is_refusal(answer: str) -> bool:
    return any(signature in answer for signature in _REFUSAL_SIGNATURES)


def load_golden() -> list[dict]:
    with open(GOLDEN_PATH, encoding="utf-8") as golden_file:
        return json.load(golden_file)


def run_pipeline(golden_items: list[dict]) -> list[dict]:
    """Run retrieve + generate for every golden question once.

    Returns one record per question carrying everything both the Ragas
    metrics and the refusal check need downstream.
    """
    records: list[dict] = []
    for item in golden_items:
        question = item["question"]
        retrieved_chunks = retrieve(question)
        relevant_chunks = select_relevant_chunks(retrieved_chunks)
        answer = generate_answer(question, retrieved_chunks)

        records.append(
            {
                "question": question,
                "type": item["type"],
                "reference": item.get("reference"),
                "retrieved_contexts": [chunk["text"] for chunk in relevant_chunks],
                "answer": answer,
                "refused": is_refusal(answer),
            }
        )
    return records


def score_refusal_accuracy(records: list[dict]) -> float:
    """Fraction of questions where refuse/answer decision matched expectation."""
    correct = sum(
        1
        for record in records
        if record["refused"] == (record["type"] in REFUSAL_TYPES)
    )
    return correct / len(records) if records else 0.0


def score_ragas(records: list[dict]) -> dict[str, float]:
    """Run the Ragas metrics over the answerable questions only.

    Refusal questions have no reference answer and (correctly) no grounded
    response, so faithfulness/recall are meaningless for them.
    """
    answerable_records = [r for r in records if r["type"] == "answerable"]
    if not answerable_records:
        return {}

    dataset = EvaluationDataset.from_list(
        [
            {
                "user_input": record["question"],
                "retrieved_contexts": record["retrieved_contexts"] or [""],
                "response": record["answer"],
                "reference": record["reference"],
            }
            for record in answerable_records
        ]
    )

    judge_llm = LangchainLLMWrapper(
        ChatAnthropic(
            model=ANTHROPIC_MODEL,
            api_key=os.environ["ANTHROPIC_API_KEY"],
            # Generous headroom: Ragas judge prompts (per-claim verdicts) can be
            # long; too low truncates a verdict and drops that sample's score.
            max_tokens=2048,
            temperature=0.0,
        )
    )
    judge_embeddings = LangchainEmbeddingsWrapper(LocalEmbeddings())

    result = evaluate(
        dataset=dataset,
        metrics=RAGAS_METRICS,
        llm=judge_llm,
        embeddings=judge_embeddings,
        show_progress=True,
    )

    scores_frame = result.to_pandas()
    return {
        metric.name: float(scores_frame[metric.name].mean())
        for metric in RAGAS_METRICS
        if metric.name in scores_frame.columns
    }


def print_report(records: list[dict], ragas_scores: dict[str, float], refusal_accuracy: float) -> None:
    answerable_count = sum(1 for r in records if r["type"] == "answerable")
    refusal_count = len(records) - answerable_count

    print("\n" + "=" * 60)
    print("PT RAG — EVAL REPORT")
    print("=" * 60)
    print(f"Golden questions: {len(records)} "
          f"({answerable_count} answerable, {refusal_count} refusal-expected)")

    print(f"\nRagas metrics (answerable, n={answerable_count}):")
    if ragas_scores:
        for metric_name, score in ragas_scores.items():
            print(f"  {metric_name:<22} {score:.3f}")
    else:
        print("  (no answerable questions)")

    print(f"\nRefusal accuracy (all, n={len(records)}): {refusal_accuracy:.3f}")

    misclassified = [
        r for r in records if r["refused"] != (r["type"] in REFUSAL_TYPES)
    ]
    if misclassified:
        print("\n  Refusal mismatches:")
        for record in misclassified:
            expected = "refuse" if record["type"] in REFUSAL_TYPES else "answer"
            got = "refused" if record["refused"] else "answered"
            print(f"    [{record['type']}] expected={expected}, got={got} "
                  f"-> {record['question']}")
    print("=" * 60 + "\n")


def main() -> None:
    golden_items = load_golden()
    print(f"Running pipeline over {len(golden_items)} golden questions...")
    records = run_pipeline(golden_items)

    refusal_accuracy = score_refusal_accuracy(records)
    ragas_scores = score_ragas(records)

    print_report(records, ragas_scores, refusal_accuracy)


if __name__ == "__main__":
    main()
