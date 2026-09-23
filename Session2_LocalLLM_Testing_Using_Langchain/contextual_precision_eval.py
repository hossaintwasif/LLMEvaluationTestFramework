"""Contextual Precision evaluation of the Customer Onboarding BRD RAG pipeline.

Simulates a real project workflow:
1. Business_Requirement_Document.pdf is the knowledge base.
2. The document is loaded, chunked and embedded into a Chroma vector store.
3. A golden dataset holds questions written against the BRD together with
   the expected answers extracted from the document.
4. For every question the retriever fetches the top-k chunks LIVE - the
   retrieval_context is never hardcoded - and the RAG chain answers from
   them, so actual_output also comes from the document.
5. DeepEval's ContextualPrecisionMetric judges whether the retriever ranked
   the relevant chunks above the irrelevant ones.

Backend switch via the EVAL_BACKEND env var:
    (default) EVAL_BACKEND=openai -> reliable judge; requires OPENAI_API_KEY in .env
    EVAL_BACKEND=ollama           -> local, free (Ollama must be running on :11434)

The Chroma embeddings always run locally through Ollama's nomic-embed-text
(same as the Session3/Session4 RAG notebooks), so Ollama needs to be up for
the retrieval step even when the app LLM and judge run on OpenAI.

Run:
    python contextual_precision_eval.py

The same pipeline is exposed as pytest tests (markers: testRag and
contextualPrecision) in test_contextual_precision.py, so it can run combined
with test_rag_eval.py and produce junit/html reports:

    pytest -m testRag --junitxml=results.xml --html=report.html --self-contained-html

On Windows PowerShell, if you see a UnicodeEncodeError from emoji output, set:

    $env:PYTHONIOENCODING = "utf-8"
"""

import os
import warnings
from pathlib import Path

# DeepEval 4.2.x gives each test case a 180s outer budget, sliced into ~88s
# per-attempt slices - too tight for local Ollama judge calls. Raise the
# override BEFORE importing deepeval so its Settings singleton picks it up
# (process env wins over DeepEval's own .env autoloading).
os.environ.setdefault("DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE", "600")

from dotenv import load_dotenv

from deepeval.evaluate import evaluate
from deepeval.evaluate.configs import AsyncConfig
from deepeval.metrics import ContextualPrecisionMetric
from deepeval.models import OllamaModel, OpenAIModel
from deepeval.test_case import LLMTestCase

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Shared .env lives in Session1_Intro (single source of truth for secrets).
REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=REPO_ROOT / "Session1_Intro" / ".env")

OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_APP_MODEL = "llama3.2"           # LangChain ChatOllama model name
OLLAMA_JUDGE_MODEL = "llama3.2:latest"  # DeepEval judge (needs the :latest tag)
OPENAI_APP_MODEL = "gpt-4o-mini"        # cheap model, fine for generating answers
OPENAI_JUDGE_MODEL = "gpt-4o-mini"      # cheap-but-reliable judge
EMBEDDING_MODEL = "nomic-embed-text"
TOP_K = 4

BRD_PATH = Path(__file__).resolve().parent / "Business_Requirement_Document.pdf"

# Backend switch: "openai" (default) or "ollama".
BACKEND = os.getenv("EVAL_BACKEND", "openai").lower()


def validate_backend(backend):
    """Raise a clear error for an unknown backend or a missing OpenAI key."""
    if backend not in {"ollama", "openai"}:
        raise ValueError(f"Unknown EVAL_BACKEND: {backend!r}. Use 'openai' or 'ollama'.")
    if backend == "openai" and not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "EVAL_BACKEND=openai requires OPENAI_API_KEY in the environment or .env file."
        )

# CONFIDENT_API_KEY only uploads results to the Confident AI dashboard.
if not os.getenv("CONFIDENT_API_KEY"):
    warnings.warn(
        "CONFIDENT_API_KEY is missing. Results will not be uploaded to Confident AI.",
        RuntimeWarning,
        stacklevel=2,
    )

# Golden dataset written against the BRD by QA/BA. Only the expected answers
# come from the document; retrieval_context is fetched live by the retriever.
GOLDEN_DATASET = [
    {
        "input": "What is the name of the project described in the business requirement document?",
        "expected_output": "Customer Onboarding Automation",
        "source": "BRD header - Project Name",
    },
    {
        "input": "Who is the author of the business requirement document?",
        "expected_output": "Twasif",
        "source": "BRD header - Author",
    },
    {
        "input": "What is out of scope for the customer onboarding automation project?",
        "expected_output": "Marketing automation and customer support ticketing",
        "source": "Scope - Out-of-Scope",
    },
    {
        "input": "How many onboarding requests per day must the system handle?",
        "expected_output": "10,000 onboarding requests per day",
        "source": "Non-Functional Requirements - Performance",
    },
    {
        "input": "Which compliance standards must the onboarding system follow?",
        "expected_output": "GDPR and ISO 27001",
        "source": "Non-Functional Requirements - Compliance",
    },
    {
        "input": "What are the acceptance criteria for the onboarding process?",
        "expected_output": (
            "95% accuracy in document validation and onboarding completed "
            "within 5 minutes per customer"
        ),
        "source": "Acceptance Criteria",
    },
]


def _build_app_llm(backend=None):
    """LLM that generates the RAG answer being tested."""
    backend = backend or BACKEND
    if backend == "openai":
        return ChatOpenAI(model=OPENAI_APP_MODEL, temperature=0.5)
    return ChatOllama(
        base_url=OLLAMA_BASE_URL,
        model=OLLAMA_APP_MODEL,
        temperature=0.5,
        num_predict=512,
    )


def _build_judge_model(backend=None):
    """LLM used by ContextualPrecisionMetric to rank the retrieved chunks."""
    backend = backend or BACKEND
    if backend == "openai":
        return OpenAIModel(model=OPENAI_JUDGE_MODEL)  # reads OPENAI_API_KEY from env
    return OllamaModel(
        model=OLLAMA_JUDGE_MODEL,
        base_url=OLLAMA_BASE_URL,
        # Cap generation: DeepEval sends the whole JSON schema as the Ollama
        # format, and an uncapped local llama3.2 run is slow and can time out.
        # Verdict reasons make the JSON long, so 512 tokens truncates it.
        generation_kwargs={"num_predict": 2048},
    )


def build_rag_chain(app_llm):
    """Load the BRD PDF, chunk it, embed into Chroma, return (chain, retriever)."""
    pages = PyPDFLoader(str(BRD_PATH)).load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=350, chunk_overlap=50)
    chunks = text_splitter.split_documents(pages)

    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)
    vector_store = Chroma.from_documents(documents=chunks, embedding=embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": TOP_K})

    prompt = ChatPromptTemplate.from_template(
        "You are a helpful assistant that answers questions using only the context below.\n"
        "If the context does not contain the answer, say you cannot find it in the document.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n"
        "Answer:"
    )
    return prompt | app_llm | StrOutputParser(), retriever


def retrieve_and_format(retriever, question):
    """Fetch top-k chunks live and return (list of chunk texts, joined text)."""
    docs = retriever.invoke(question)
    return [doc.page_content for doc in docs], "\n\n".join(doc.page_content for doc in docs)


def build_test_cases(chain, retriever):
    """Answer every golden question through the RAG chain and build test cases."""
    test_cases = []
    for item in GOLDEN_DATASET:
        question = item["input"]
        retrieval_context, context_text = retrieve_and_format(retriever, question)
        actual_output = chain.invoke({"context": context_text, "question": question})

        print(f"\nQuestion: {question}")
        print(f"Retrieved {len(retrieval_context)} chunks (k={TOP_K})")
        print(f"Actual:   {actual_output}")
        print(f"Expected: {item['expected_output']}   [{item['source']}]")

        test_cases.append(
            LLMTestCase(
                name=question,
                input=question,
                actual_output=actual_output,
                expected_output=item["expected_output"],
                retrieval_context=retrieval_context,
            )
        )
    return test_cases


def main():
    validate_backend(BACKEND)

    precision_metric = ContextualPrecisionMetric(
        threshold=0.5,
        model=_build_judge_model(),
        include_reason=True,
        async_mode=True,
        strict_mode=False,
    )

    chain, retriever = build_rag_chain(_build_app_llm())
    test_cases = build_test_cases(chain, retriever)

    print("\nEvaluating contextual precision...")
    results = evaluate(
        test_cases=test_cases,
        metrics=[precision_metric],
        # evaluate() resets the test run manager, so hyperparameters must be
        # passed here (the @log_hyperparameters decorator is pytest-only).
        hyperparameters={
            "Backend": BACKEND,
            "App Model": OLLAMA_APP_MODEL if BACKEND == "ollama" else OPENAI_APP_MODEL,
            "Judge Model": OLLAMA_JUDGE_MODEL if BACKEND == "ollama" else OPENAI_JUDGE_MODEL,
        },
        async_config=AsyncConfig(run_async=False),
    )

    for test_result in results.test_results:
        for metric_data in test_result.metrics_data:
            print(
                f"\n[{metric_data.name}] {test_result.name}\n"
                f"  score={metric_data.score:.2f}  passed={metric_data.success}\n"
                f"  reason: {metric_data.reason}"
            )
    return results


if __name__ == "__main__":
    main()
