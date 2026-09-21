# RAG Evaluation Tests (Session 2)

This folder contains the pytest suite that evaluates LLM answers with
DeepEval's GEval "Correctness" metric (LLM-as-judge).

- `test_rag_eval.py` — the test suite (tagged: `testRag`, `correctness`)
- `results.xml` — JUnit XML report for Jenkins (machine-readable)
- `report.html` — visual HTML report for humans (open in a browser)

## Prerequisites

1. **Virtual environment** — use the project venv `myenv314` (repo root), with:
   `deepeval`, `langchain-ollama`, `langchain-openai`, `pytest`, `pytest-html`
2. **Ollama backend (optional)** — Ollama running on `http://localhost:11434`
   with the `llama3.2` model. Needed only for the `ollama` backend.
3. **Secrets** — `Session1_Intro\.env` must exist with:
   - `OPENAI_API_KEY` — required for the `openai` backend
   - `CONFIDENT_API_KEY` — optional; only used to upload results to the
     Confident AI dashboard. A missing key is a warning, not an error.

## How to run

From this folder (or the repo root — pytest finds tests recursively):

```powershell
$env:PYTHONIOENCODING = "utf-8"   # avoids the emoji/cp1252 crash on Windows
pytest -m testRag -v              # run only the testRag-tagged tests
```

Run with **OpenAI** (paid API) for this terminal session:

```powershell
$env:PYTHONIOENCODING = "utf-8"   # avoids the emoji/cp1252 crash
$env:EVAL_BACKEND = "openai"      # switch to OpenAI for this terminal session
pytest -m testRag -v
```

That's it — `OPENAI_API_KEY` is picked up automatically from your `.env`.
The tag filter `-m testRag` selects only tagged tests; you can also use
`-m correctness` or run all with plain `pytest`.

### Notes on `EVAL_BACKEND`

- `$env:EVAL_BACKEND = "openai"` lasts for that PowerShell session only —
  close the terminal or run `Remove-Item Env:EVAL_BACKEND` and you're back
  to the Ollama default.
- If you want OpenAI **permanently** for local runs, add one line to
  `Session1_Intro\.env`:

  ```
  EVAL_BACKEND=openai
  ```

  Then plain `pytest -m testRag` always uses OpenAI — no terminal env needed.
  (A terminal `$env:` setting still overrides the `.env` value if you ever
  set both.)

## Environment variables

| Variable            | Values            | Default  | Purpose                                        |
| ------------------- | ----------------- | -------- | ---------------------------------------------- |
| `EVAL_BACKEND`      | `ollama`, `openai`| `ollama` | Which LLM generates and judges the answers     |
| `OPENAI_API_KEY`    | your API key      | —        | Required only when `EVAL_BACKEND=openai`       |
| `CONFIDENT_API_KEY` | your API key      | —        | Optional: uploads results to Confident AI      |
| `PYTHONIOENCODING`  | `utf-8`           | —        | Workaround for emoji crashes in Windows console|

## Test tags (markers)

Registered in `pytest.ini` at the repo root. Add new tags there first.

- `testRag` — RAG/LLM answer-quality evaluations
- `correctness` — LLM-as-judge correctness tests
- `slow` — reserved for long-running suites

Selection examples:

```powershell
pytest -m testRag                 # only tagged tests
pytest -m "not slow"              # everything except slow
pytest -m "testRag and not slow"  # combinations
pytest                            # run everything (no filter)
```

## Generating reports

```powershell
pytest test_rag_eval.py -v --junitxml=results.xml --html=report.html --self-contained-html
```

- `report.html` — double-click to view results in a browser.
- `results.xml` — feed to Jenkins ("Publish JUnit test result report").

## Jenkins / CI

CI agents have no local Ollama server, so the pipeline pins the OpenAI
backend and injects the API key from the Jenkins credential store:

```groovy
pipeline {
    agent any
    environment {
        EVAL_BACKEND = 'openai'
        PYTHONIOENCODING = 'utf-8'
    }
    stages {
        stage('RAG evals') {
            steps {
                withCredentials([string(credentialsId: 'openai-api-key', variable: 'OPENAI_API_KEY')]) {
                    bat 'myenv314\\Scripts\\python.exe -m pytest -m testRag --junitxml=results.xml --html=report.html --self-contained-html'
                }
            }
        }
    }
    post {
        always { junit 'results.xml' }
    }
}
```

The CI gate is the pytest exit code: a failing metric raises an
`AssertionError` inside `assert_test`, so a failed eval = failed build.

## Troubleshooting

- **`UnicodeEncodeError` (emoji output)** — set `$env:PYTHONIOENCODING = "utf-8"`.
- **`PytestUnknownMarkWarning`** — the tag isn't registered in `pytest.ini`.
  Add it under `markers =`.
- **Ollama run passes one time, fails the next** — llama3.2 (3B) is a weak,
  non-deterministic judge. Use `EVAL_BACKEND=openai` for reliable results.
- **`EVAL_BACKEND=openai` fails with RuntimeError** — `OPENAI_API_KEY` is
  missing from `.env` or the environment.
