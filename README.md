# LLM Evaluation Framework — DeepEval + OpenAI + Confident AI

A hands-on framework for evaluating LLM responses using **[DeepEval](https://deepeval.com)** with **OpenAI** (`gpt-4o-mini`) as the evaluation model and **Confident AI** as the reporting / dataset hub.

The project demonstrates:

- **Answer Relevancy** and **Contextual Precision** metrics
- Secure loading of API keys via a local `.env` file (dev) or Jenkins credentials (CI/CD)
- A reusable evaluation script (`run_deepeval_eval.py`) with graceful fallback when the OpenAI quota is exhausted
- A unit test that simulates quota exhaustion
- A Jenkins pipeline that installs dependencies, runs the test, and executes the evaluation

---

## Table of Contents

1. [Project Structure](#project-structure)
2. [Prerequisites](#prerequisites)
3. [Setup](#setup)
4. [Environment Variables](#environment-variables)
5. [Running Locally](#running-locally)
6. [Understanding the Metrics](#understanding-the-metrics)
7. [Confident AI Reporting](#confident-ai-reporting)
8. [Jenkins CI/CD Pipeline](#jenkins-cicd-pipeline)
9. [Security & .gitignore](#security--gitignore)
10. [Troubleshooting](#troubleshooting)

---

## Project Structure

```
Udemy_AI_project/
├── Jenkinsfile                    # CI/CD pipeline definition
├── README.md                      # This file
├── requirements.txt               # Pinned dependencies — install with one command
├── .gitignore                     # Secret + Python + IDE ignores
│
└── Session1_Intro/
    ├── first.ipynb                # Step-by-step Jupyter walkthrough
    ├── run_deepeval_eval.py       # Reusable evaluation entry point
    ├── test_run_deepeval_eval.py  # Unit test (quota-exhaustion fallback)
    ├── env_loader.py              # Loads keys from .env / environment
    ├── .env.example               # Template — safe to commit
    └── .env                       # Real keys — NEVER committed
```

> `myenv314/` is a local Python 3.14 virtual environment. It is git-ignored — never commit it.
> `Congig_Secret_File.txt` is a plain-text backup of your keys. It is git-ignored — never commit it.

---

## Prerequisites

- **Python 3.14+** (developed against 3.14)
- **pip**
- An **OpenAI API key** (used as the LLM judge model)
- A **Confident AI API key** (for login / reporting; free account at [confident-ai.com](https://www.confident-ai.com))

---

## Setup

Clone the repo and create a virtual environment:

```bash
git clone <your-repo-url>
cd Udemy_AI_project

# Create and activate a venv (Windows example)
python -m venv myenv314
myenv314\Scripts\activate

# Install dependencies (pinned versions in requirements.txt)
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

> **Why no venv in the repo?** Virtual environments are machine-specific (absolute paths, platform binaries, Python-version bindings). A clone that checked in a venv would be broken on any other machine. Instead, the repo ships the *recipe* (`requirements.txt`) so every person recreates an identical environment locally.

**If you add a dependency locally**, keep `requirements.txt` in sync:

```bash
python -m pip install <new-package>
python -m pip freeze > requirements.txt   # pins everything you installed
```

Then commit the updated `requirements.txt` — teammates and Jenkins pick it up automatically.

---

## Environment Variables

Two keys are required:

| Variable            | Purpose                                | Where to get it                          |
| ------------------- | -------------------------------------- | ---------------------------------------- |
| `OPENAI_API_KEY`    | LLM judge used by DeepEval metrics     | [platform.openai.com](https://platform.openai.com) |
| `CONFIDENT_API_KEY` | Confident AI login + result reporting  | Confident AI dashboard                   |

### Local development

1. Copy the template:

   ```bash
   copy Session1_Intro\.env.example Session1_Intro\.env
   ```

2. Fill in your real keys in `Session1_Intro/.env`:

   ```dotenv
   OPENAI_API_KEY=sk-...
   CONFIDENT_API_KEY=...
   ```

3. `env_loader.py` reads this file automatically. It **raises** `RuntimeError` if a key is missing, and prints whether each key was loaded.

### CI/CD (Jenkins)

Do **not** commit keys. Create two **secret-text credentials** in Jenkins:

| Credential ID     | Value                    |
| ----------------- | ------------------------ |
| `OPENAI_API_KEY`  | your OpenAI key          |
| `CONFIDENT_API_KEY` | your Confident AI key  |

The pipeline injects them as environment variables — no files or keys in the repo.

---

## Running Locally

### 1. Interactive walkthrough (Jupyter)

```bash
jupyter notebook Session1_Intro/first.ipynb
```

The notebook covers: secure env loading → Answer Relevancy → Confident AI login → Contextual Precision → `evaluate()` → Golden datasets → pushing datasets to Confident AI.

### 2. Scripted evaluation (recommended for CI)

```bash
cd Session1_Intro
python run_deepeval_eval.py
```

What it does:

- Loads keys via `env_loader.load_runtime_env()`
- Runs the **Answer Relevancy** metric on a sample test case
- Prints `Answer Relevancy Score: <float>`
- Returns the score as a float

Graceful fallbacks (returns `0.0` instead of crashing):

- OpenAI `RateLimitError` / tenacity `RetryError` (quota exhausted)
- Missing keys (`RuntimeError`)

### 3. Unit test

```bash
cd Session1_Intro
python -m pytest test_run_deepeval_eval.py -v
```

The test monkeypatches the metric to raise a 429 `RateLimitError` and asserts `run_evaluation()` falls back to `0.0`.

---

## Understanding the Metrics

| Metric                 | What it measures                                  | Keys used                              |
| ---------------------- | ------------------------------------------------ | -------------------------------------- |
| `AnswerRelevancyMetric` | Whether the output answers the input without irrelevant statements | `input`, `actual_output`, `retrieval_context` |
| `ContextualPrecisionMetric` | Whether relevant context is ranked above irrelevant context | `input`, `actual_output`, `retrieval_context`, `expected_output` |

Scores are `0.0 – 1.0`, with a default threshold of `0.5` (pass/fail). In the notebook you can inspect `score`, `success`, `verdicts`, and `reason` for full traceability.

---

## Confident AI Reporting

DeepEval integrates with **Confident AI** for hosted test runs and datasets:

```python
import deepeval

deepeval.login(CONFIDENT_API_KEY)   # one-time login
```

Then use `evaluate(test_cases=..., metrics=...)` — results are pushed to the Confident AI dashboard with a link like:

```
https://app.confident-ai.com/project/<project>/test-runs/<run-id>/test-cases
```

You can also build `EvaluationDataset` from `Golden` entries and push them:

```python
dataset = EvaluationDataset(goldens=[...])
dataset.push(alias="TestGoldenDataset", overwrite=True)
```

> **Note:** In Jupyter on Python 3.14, prefer `async_config=AsyncConfig(run_async=False)` in `evaluate()` — the default asyncio path conflicts with Jupyter's running event loop (see Troubleshooting).

---

## Jenkins CI/CD Pipeline

The `Jenkinsfile` defines three stages:

| Stage           | What it does                                                    |
| --------------- | --------------------------------------------------------------- |
| **Setup**       | Checks Python version, upgrades pip, installs `requirements.txt` (pinned) |
| **Unit test**   | Runs `pytest test_run_deepeval_eval.py -v` in `Session1_Intro`  |
| **Run evaluation** | Runs `python run_deepeval_eval.py` in `Session1_Intro`      |

Key points:

- `agent any` — runs on any available Jenkins agent
- Keys come from the Jenkins credential store via `credentials('OPENAI_API_KEY')` / `credentials('CONFIDENT_API_KEY')`
- The pipeline calls the project's own entry point — no duplicated inline logic
- Evaluation failures due to quota are **non-fatal** by design (`run_deepeval_eval.py` returns `0.0`), so the build won't break on a rate limit

### Setting up the Jenkins job

1. **New Item → Pipeline**
2. **Pipeline script from SCM** → Git → your repo URL
3. Script path: `Jenkinsfile`
4. Add the two **secret-text** credentials above (same IDs)
5. Save and run — the `Setup` stage installs everything, so no pre-configuration on the agent is needed.

---

## Security & .gitignore

The root `.gitignore` protects secrets and local artifacts:

```gitignore
# Secrets
.env
.env.*
!.env.example

# Plain-text secret files
Congig_Secret_File.txt

# Python
__pycache__/
*.pyc *.pyo *.pyd

# Virtual envs
.venv/ venv/ myenv314/

# IDE
.vscode/

# Tooling / runtime data
.commandcode/
.deepeval/
.ipynb_checkpoints/
```

- `.env` → ignored (real keys)
- `.env.*` → ignored, **except** `!.env.example` (the safe, placeholder template that SHOULD be committed)
- `Congig_Secret_File.txt` → ignored (plain-text key backup)
- Virtual envs, caches, and IDE folders → ignored

> **Before first push:** verify nothing sensitive is staged:

```bash
git status
git add .
git status          # confirm .env and Congig_Secret_File.txt are NOT listed
```

If you accidentally tracked them earlier, remove them from the index:

```bash
git rm --cached Session1_Intro/.env Congig_Secret_File.txt
```

---

## Troubleshooting

| Symptom | Cause & fix |
| ------- | ----------- |
| `OPENAI_API_KEY is missing` | `load_runtime_env()` couldn't find the key. Ensure `.env` exists with a real key, or the env var is set (CI). |
| `insufficient_quota` / 429 errors | OpenAI quota exhausted. `run_deepeval_eval.py` handles this and returns `0.0`; top up your OpenAI account or switch models. |
| `RuntimeError: Cannot enter into task ...` in Jupyter | DeepEval's async path conflicts with Jupyter's event loop on Python 3.14. Use `AsyncConfig(run_async=False)` in `evaluate()`, or run the scripted version instead. |
| Build fails at `python -m pip install` | Agent may lack network access or Python on PATH. Check the Jenkins agent's Python installation. |
| Secrets leaked to git | If `.env` was committed, revoke the keys and rotate them, then `git rm --cached` + re-commit. |

---

## License

Private / learning project — see your own licensing decisions before distributing.
