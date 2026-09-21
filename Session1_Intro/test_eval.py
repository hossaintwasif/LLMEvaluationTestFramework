"""DeepEval AnswerRelevancy evaluation (Session 1).

Run locally or in CI (Jenkins):

    pytest Session1_Intro/test_eval.py -v -m relevancy

Uses OpenAI (gpt-4o-mini) as the judge; OPENAI_API_KEY is loaded from the
shared .env in this folder.
"""

import os
import warnings
from pathlib import Path

import pytest
from dotenv import load_dotenv

from deepeval import assert_test
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.models import OpenAIModel
from deepeval.test_case import LLMTestCase

# Shared .env lives in this folder (single source of truth for secrets).
REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=REPO_ROOT / "Session1_Intro" / ".env")

openai_key = os.getenv("OPENAI_API_KEY")
if not openai_key:
    raise RuntimeError("OPENAI_API_KEY is missing. Set it in the environment or .env file.")

# CONFIDENT_API_KEY only uploads results to the Confident AI dashboard.
# The CI gate is the pytest exit code, so a missing key must not fail the run.
if not os.getenv("CONFIDENT_API_KEY"):
    warnings.warn(
        "CONFIDENT_API_KEY is missing. Results will not be uploaded to Confident AI.",
        RuntimeWarning,
        stacklevel=2,
    )

# Judge: graded by OpenAI. AnswerRelevancyMetric checks whether the actual
# output is self-contained and relevant to the input (it does NOT compare
# against an expected answer).
llm_model = OpenAIModel(model="gpt-4o-mini", api_key=openai_key)


@pytest.mark.testRag
@pytest.mark.relevancy
def test_answer_relevancy():
    metric = AnswerRelevancyMetric(
        threshold=0.5,
        model=llm_model,
        include_reason=True,
        async_mode=False,  # synchronous: avoids asyncio event-loop errors
    )
    test_case = LLMTestCase(
        input="Who is the current president of United States Of America?",
        actual_output="Joe Biden",
        retrieval_context=["The current president of the United States is Joe Biden."],
    )
    assert_test(test_case, [metric], run_async=False)
