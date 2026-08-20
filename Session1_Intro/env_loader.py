import os
from pathlib import Path
from typing import Dict, Optional

from dotenv import load_dotenv


DEFAULT_ENV_PATH = Path(__file__).resolve().parent / ".env"


def load_runtime_env(env_path: Optional[str | os.PathLike[str]] = None) -> Dict[str, str]:
    """Load env vars without hardcoding secrets in source control.

    Use a local .env file in development and Jenkins credentials in CI/CD.
    """
    env_file = Path(env_path) if env_path else DEFAULT_ENV_PATH
    load_dotenv(env_file, override=False)

    values = {
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY", ""),
        "CONFIDENT_API_KEY": os.getenv("CONFIDENT_API_KEY", ""),
    }

    if not values["OPENAI_API_KEY"]:
        raise RuntimeError("OPENAI_API_KEY is missing. Set it in the environment or .env file.")

    if not values["CONFIDENT_API_KEY"]:
        raise RuntimeError("CONFIDENT_API_KEY is missing. Set it in the environment or .env file.")

    os.environ["OPENAI_API_KEY"] = values["OPENAI_API_KEY"]
    os.environ["CONFIDENT_API_KEY"] = values["CONFIDENT_API_KEY"]

    return values


def get_confident_key() -> str:
    return os.getenv("CONFIDENT_API_KEY", "")


def get_openai_key() -> str:
    return os.getenv("OPENAI_API_KEY", "")
