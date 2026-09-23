"""DeepEval Contextual Precision pytest suite for the BRD RAG pipeline.

Run alone or combined with the other RAG evaluation tests via the markers:

    pytest -m contextualPrecision -v
    pytest -m testRag -v --junitxml=results.xml --html=report.html --self-contained-html

Backend switch via the EVAL_BACKEND env var (same convention as test_rag_eval.py):
    (default) EVAL_BACKEND=ollama -> local, free (Ollama must be running on :11434)
    EVAL_BACKEND=openai           -> gpt-4o-mini; requires OPENAI_API_KEY in .env

The RAG pipeline (PDF load, chunking, Chroma, retriever) and the golden
dataset are shared with contextual_precision_eval.py. The retrieval_context
is fetched live by the retriever for every question, never hardcoded.

After the run, conftest.py uploads the collected results to the Confident AI
dashboard (requires CONFIDENT_API_KEY; a missing key only warns locally).
"""

import os
from pathlib import Path

# Must be set before deepeval is imported - its Settings singleton reads the
# env once, and the default ~88s per-attempt budget is too tight for local
# Ollama judge calls. Mirrors the override set by contextual_precision_eval.py.
os.environ.setdefault("DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE", "600")

import pytest
from dotenv import load_dotenv

from deepeval import assert_test
from deepeval.metrics import ContextualPrecisionMetric
from deepeval.test_case import LLMTestCase
from deepeval.test_run import log_hyperparameters

from contextual_precision_eval import (
    GOLDEN_DATASET,
    OLLAMA_APP_MODEL,
    OLLAMA_JUDGE_MODEL,
    OPENAI_APP_MODEL,
    OPENAI_JUDGE_MODEL,
    _build_app_llm,
    _build_judge_model,
    build_rag_chain,
    retrieve_and_format,
    validate_backend,
)

# Shared .env lives in Session1_Intro (single source of truth for secrets).
REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=REPO_ROOT / "Session1_Intro" / ".env")

# Backend switch: "ollama" (default, matches test_rag_eval.py) or "openai".
BACKEND = os.getenv("EVAL_BACKEND", "ollama").lower()
validate_backend(BACKEND)


@log_hyperparameters
def hyperparameters():
    """Model/backend attributes shown on the Confident AI test run."""
    return {
        "Backend": BACKEND,
        "App Model": OLLAMA_APP_MODEL if BACKEND == "ollama" else OPENAI_APP_MODEL,
        "Judge Model": OLLAMA_JUDGE_MODEL if BACKEND == "ollama" else OPENAI_JUDGE_MODEL,
    }


@pytest.fixture(scope="module")
def rag_pipeline():
    """Build the BRD RAG chain + retriever once for all parametrized cases."""
    return build_rag_chain(_build_app_llm(BACKEND))


@pytest.fixture(scope="module")
def precision_metric() -> ContextualPrecisionMetric:
    """ContextualPrecisionMetric judge: ranks retrieved chunks by relevance."""
    return ContextualPrecisionMetric(
        threshold=0.5,
        model=_build_judge_model(BACKEND),
        include_reason=True,
        async_mode=True,
        strict_mode=False,
    )


@pytest.mark.testRag
@pytest.mark.contextualPrecision
@pytest.mark.parametrize("item", GOLDEN_DATASET, ids=[q["source"] for q in GOLDEN_DATASET])
def test_contextual_precision(item, rag_pipeline, precision_metric):
    chain, retriever = rag_pipeline
    question = item["input"]
    retrieval_context, context_text = retrieve_and_format(retriever, question)
    actual_output = chain.invoke({"context": context_text, "question": question})

    test_case = LLMTestCase(
        name=question,
        input=question,
        actual_output=actual_output,
        expected_output=item["expected_output"],
        retrieval_context=retrieval_context,
    )
    # DeepEval 4.2.x: assert_test takes run_async directly (no async_config kwarg).
    assert_test(test_case, [precision_metric], run_async=False)
