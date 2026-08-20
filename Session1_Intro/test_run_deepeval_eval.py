import types

from openai import RateLimitError

import run_deepeval_eval


class DummyMetric:
    def __init__(self, *args, **kwargs):
        self.score = 0.0

    def measure(self, test_case):
        request = types.SimpleNamespace(method="POST", url="https://api.openai.com/v1/chat/completions")
        headers = types.SimpleNamespace(get=lambda key, default=None: default)
        response = types.SimpleNamespace(request=request, status_code=429, headers=headers)
        raise RateLimitError("quota exhausted", response=response, body={"error": {"message": "insufficient_quota"}})


def test_run_evaluation_falls_back_when_api_is_quota_exhausted(monkeypatch):
    monkeypatch.setattr(run_deepeval_eval, "AnswerRelevancyMetric", DummyMetric)
    monkeypatch.setattr(run_deepeval_eval, "load_runtime_env", lambda: None)

    score = run_deepeval_eval.run_evaluation()

    assert score == 0.0
