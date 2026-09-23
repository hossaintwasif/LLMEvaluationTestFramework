"""Pytest hooks for the DeepEval suite in this folder.

``assert_test`` accumulates every result in DeepEval's global test run
manager but never uploads anything. This hook wraps the run up once
pytest finishes, which uploads the report to Confident AI when
CONFIDENT_API_KEY is set (otherwise the run is kept locally and
``deepeval view`` can open it).
"""

import sys

from deepeval.test_run import global_test_run_manager


def pytest_sessionfinish(session, exitstatus):
    manager = global_test_run_manager
    # Read the attribute directly: get_test_run() would create an empty
    # run when no DeepEval test executed (collection errors, no matches).
    test_run = manager.test_run
    if test_run is None:
        return
    if not test_run.test_cases and not test_run.conversational_test_cases:
        return

    # runDuration is only informational on the dashboard; pytest reports
    # session.duration once the run loop finishes.
    duration = float(getattr(session, "duration", 0.0) or 0.0)

    # An upload failure must never flip a finished build's exit code.
    try:
        manager.wrap_up_test_run(runDuration=duration)
    except Exception as exc:
        print(
            f"Warning: could not upload DeepEval test run to Confident AI: {exc}",
            file=sys.stderr,
        )
