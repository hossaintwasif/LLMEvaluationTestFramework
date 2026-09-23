"""DeepEval LLM-as-judge evaluation suite for LangChain answers.

Run locally or in CI (Jenkins):

    pytest test_rag_eval.py -v --junitxml=results.xml --html=report.html --self-contained-html

Backend switch via the EVAL_BACKEND env var:
    (default) EVAL_BACKEND=ollama  -> local, free (Ollama must be running on :11434)
    EVAL_BACKEND=openai            -> paid API; requires OPENAI_API_KEY in .env

On Windows PowerShell, if you see a UnicodeEncodeError from emoji output, set:

    $env:PYTHONIOENCODING = "utf-8"

The pass/fail gate is pytest's exit code: `assert_test` raises AssertionError
when a metric fails, which pytest turns into a failed test (non-zero exit).

After the run, conftest.py uploads the collected results to the Confident AI
dashboard (requires CONFIDENT_API_KEY; a missing key only warns locally).
"""

import os
import warnings
from pathlib import Path

import pytest
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from deepeval import assert_test
from deepeval.metrics import GEval
from deepeval.models import OllamaModel, OpenAIModel
from deepeval.test_case import LLMTestCase, SingleTurnParams
from deepeval.test_run import log_hyperparameters

# Shared .env lives in Session1_Intro (single source of truth for secrets).
REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=REPO_ROOT / "Session1_Intro" / ".env")

OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_APP_MODEL = "llama3.2"           # LangChain ChatOllama model name
OLLAMA_JUDGE_MODEL = "llama3.2:latest"  # DeepEval judge (needs the :latest tag)
OPENAI_APP_MODEL = "gpt-4o-mini"        # cheap model, fine for generating answers
OPENAI_JUDGE_MODEL = "gpt-4o-mini"      # cheap-but-reliable CI judge

# Backend switch: "ollama" (default) or "openai".
BACKEND = os.getenv("EVAL_BACKEND", "ollama").lower()
if BACKEND not in {"ollama", "openai"}:
    raise ValueError(f"Unknown EVAL_BACKEND: {BACKEND!r}. Use 'ollama' or 'openai'.")
if BACKEND == "openai" and not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError("EVAL_BACKEND=openai requires OPENAI_API_KEY in the environment or .env file.")

# CONFIDENT_API_KEY only uploads results to the Confident AI dashboard.
# The CI gate is the pytest exit code, so a missing key must not fail the run.
# The actual upload happens in conftest.py (pytest_sessionfinish hook).
if not os.getenv("CONFIDENT_API_KEY"):
    warnings.warn(
        "CONFIDENT_API_KEY is missing. Results will not be uploaded to Confident AI.",
        RuntimeWarning,
        stacklevel=2,
    )


@log_hyperparameters
def hyperparameters():
    """Model/backend attributes shown on the Confident AI test run."""
    return {
        "Backend": BACKEND,
        "App Model": OLLAMA_APP_MODEL if BACKEND == "ollama" else OPENAI_APP_MODEL,
        "Judge Model": OLLAMA_JUDGE_MODEL if BACKEND == "ollama" else OPENAI_JUDGE_MODEL,
    }


def _build_app_llm():
    """LLM that generates the answer being tested."""
    if BACKEND == "openai":
        return ChatOpenAI(model=OPENAI_APP_MODEL, temperature=0.5)
    return ChatOllama(
        base_url=OLLAMA_BASE_URL,
        model=OLLAMA_APP_MODEL,
        temperature=0.5,
        num_predict=512,
    )


def _build_judge_model():
    """LLM used by GEval to grade actual vs expected output."""
    if BACKEND == "openai":
        return OpenAIModel(model=OPENAI_JUDGE_MODEL)  # reads OPENAI_API_KEY from env
    return OllamaModel(model=OLLAMA_JUDGE_MODEL, base_url=OLLAMA_BASE_URL)


APP_LLM = _build_app_llm()


@pytest.fixture(scope="module")
def correctness_metric() -> GEval:
    """GEval judge: compares actual vs expected output."""
    judge = _build_judge_model()
    return GEval(
        name="Correctness",
        # DeepEval 4.x: list the test-case fields the judge is allowed to read.
        evaluation_params=[
            SingleTurnParams.INPUT,
            SingleTurnParams.ACTUAL_OUTPUT,
            SingleTurnParams.EXPECTED_OUTPUT,
            SingleTurnParams.RETRIEVAL_CONTEXT,
        ],
        criteria=(
            "The actual output must state the same core answer as the expected output. "
            "If the expected output names a person or fact, the actual output is correct "
            "when it identifies that same person or fact as the answer. Additional details, "
            "dates, or other names mentioned in the actual output do NOT make it incorrect."
        ),
        evaluation_steps=[
            "Read the input question.",
            "Read the expected output and identify the core answer (the person or fact it names).",
            "Read the actual output and check whether it states that same core answer.",
            "If the actual output states the same core answer, the test passes, even if the actual output adds extra details, dates, or other names.",
            "If the actual output states a different core answer, the test fails.",
        ],
        model=judge,
        threshold=0.5,
        async_mode=False,  # synchronous: avoids asyncio event-loop errors in CI/notebooks
        strict_mode=False,
    )


@pytest.mark.testRag
@pytest.mark.correctness
@pytest.mark.parametrize(
    "question, expected, context",
    [
        (
            "Who is the president of United States Of America in 2022?",
            "Joe Biden",
            ["The president of the United States is Joe Biden in 2022."],
        ),
        (
            "Who built the Taj Mahal?",
            "Shah Jahan",
            ["The Taj Mahal was built by Mughal Emperor Shah Jahan."],
        ),
    ],
)
def test_answer_correctness(question, expected, context, correctness_metric):
    test_case = LLMTestCase(
        input=question,
        actual_output=APP_LLM.invoke(question).content,
        expected_output=expected,
        retrieval_context=context,
    )
    # DeepEval 4.2.x: assert_test takes run_async directly (no async_config kwarg).
    assert_test(test_case, [correctness_metric], run_async=False)
