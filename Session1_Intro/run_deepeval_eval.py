import os

from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase
from openai import RateLimitError
from tenacity import RetryError

from env_loader import load_runtime_env


def _extract_error_message(exc: Exception) -> str:
    if hasattr(exc, "body") and isinstance(exc.body, dict):
        return exc.body.get("error", {}).get("message", str(exc))

    if isinstance(exc, RetryError):
        last_attempt = getattr(exc, "last_attempt", None)
        if last_attempt is not None:
            inner = getattr(last_attempt, "exception", None)
            if callable(inner):
                inner = inner()
            if isinstance(inner, Exception):
                return _extract_error_message(inner)

    return str(exc)


def run_evaluation() -> float:
    # This function explicitly loads secrets from environment variables or local .env,
    # and keeps secrets out of source control.
    try:
        load_runtime_env()

        metric = AnswerRelevancyMetric(async_mode=False)
        test_case = LLMTestCase(
            input="Who is the current president of United States Of America?",
            actual_output="Joe Biden",
            retrieval_context=["The current president of the United States is Joe Biden."],
        )

        metric.measure(test_case)
        score = float(metric.score)
        print("Answer Relevancy Score:", score)
        return score
    except (RateLimitError, RetryError) as exc:
        print(f"DeepEval evaluation skipped: OpenAI quota exhausted. {_extract_error_message(exc)}")
        return 0.0
    except RuntimeError as exc:
        print(f"DeepEval evaluation skipped: {exc}")
        return 0.0


if __name__ == "__main__":
    run_evaluation()
