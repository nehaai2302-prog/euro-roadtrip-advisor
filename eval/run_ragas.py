#!/usr/bin/env python3
"""
Offline RAG evaluation with Ragas.

Prerequisites:
  1. Run ``python ingest.py`` so ``chroma_db`` matches your knowledge base.
  2. Set ``OPENAI_API_KEY`` (e.g. in ``.env`` at project root).

Run from project root:
  uv run --extra eval python eval/run_ragas.py

Optional:
  RAGAS_JUDGE_MODEL=gpt-4o-mini   (default gpt-4o-mini; use a stronger model for stricter judging)
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Ragas eval extras (install: uv sync --extra eval)
try:
    from langchain_openai import ChatOpenAI
    from ragas import EvaluationDataset, evaluate
    from ragas.llms import LangchainLLMWrapper
    # Legacy metric objects (ragas.evaluate requires Metric subclasses, not collections v2)
    from ragas.metrics import FactualCorrectness, context_recall, faithfulness
except ImportError as e:
    raise SystemExit(
        "Missing eval dependencies. Install with: uv sync --extra eval\n" + str(e)
    ) from e

from dotenv import load_dotenv


def load_golden(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Ragas metrics on RAG golden set.")
    parser.add_argument(
        "--golden",
        type=Path,
        default=ROOT / "eval" / "golden_rag.jsonl",
        help="Path to JSONL with id, question, reference",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "eval" / "results",
        help="Directory for CSV exports",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate only the first N rows (for quick smoke tests)",
    )
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY is not set.", file=sys.stderr)
        return 1

    # Import pipeline after env (uses same Chroma + models as the app)
    from rag_pipeline import (
        default_eval_trip_context,
        generate_rag_answer,
        retrieve_rag_docs,
    )

    golden_rows = load_golden(args.golden)
    if args.limit is not None:
        golden_rows = golden_rows[: max(0, args.limit)]

    if not golden_rows:
        print("No golden rows to evaluate.", file=sys.stderr)
        return 1

    trip_ctx = default_eval_trip_context()
    chat_history: list = []

    dataset_rows: list[dict] = []
    for row in golden_rows:
        q = row["question"]
        ref = row["reference"]
        docs = retrieve_rag_docs(q)
        contexts = [d.page_content for d in docs]
        answer, _usage = generate_rag_answer(
            q,
            docs,
            chat_history,
            trip_ctx,
            traveler_type=trip_ctx["traveler_type"],
            intent="rag",
        )
        dataset_rows.append(
            {
                "user_input": q,
                "retrieved_contexts": contexts,
                "response": answer,
                "reference": ref,
                "golden_id": row.get("id", ""),
            }
        )

    evaluation_dataset = EvaluationDataset.from_list(
        [
            {
                "user_input": r["user_input"],
                "retrieved_contexts": r["retrieved_contexts"],
                "response": r["response"],
                "reference": r["reference"],
            }
            for r in dataset_rows
        ]
    )

    judge_model = os.environ.get("RAGAS_JUDGE_MODEL", "gpt-4o-mini")
    judge_chat = ChatOpenAI(model=judge_model, temperature=0)
    evaluator_llm = LangchainLLMWrapper(judge_chat)

    metrics = [faithfulness, context_recall, FactualCorrectness()]

    print(f"Evaluating {len(dataset_rows)} rows with judge model={judge_model!r} ...")
    result = evaluate(
        evaluation_dataset,
        metrics=metrics,
        llm=evaluator_llm,
        raise_exceptions=True,
    )

    df = result.to_pandas()
    # re-attach golden ids for readability
    if "golden_id" not in df.columns and len(dataset_rows) == len(df):
        df.insert(0, "golden_id", [r["golden_id"] for r in dataset_rows])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    per_row_path = args.out_dir / f"ragas_per_row_{stamp}.csv"
    df.to_csv(per_row_path, index=False)

    metric_cols = [
        c
        for c in df.columns
        if c == "faithfulness"
        or c == "context_recall"
        or c.startswith("factual_correctness")
    ]
    agg_path = args.out_dir / f"ragas_aggregate_{stamp}.csv"
    with agg_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "mean"])
        for col in metric_cols:
            mean_val = df[col].mean(skipna=True)
            w.writerow([col, f"{float(mean_val):.6f}" if mean_val == mean_val else ""])

    print("\n--- Aggregate (mean) ---")
    for col in metric_cols:
        m = df[col].mean(skipna=True)
        print(f"  {col}: {m:.4f}" if m == m else f"  {col}: nan")

    print(f"\nWrote: {per_row_path}")
    print(f"Wrote: {agg_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
