from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


# ---------------------------------------------------------------------------
# Section 1 — Paths
# ---------------------------------------------------------------------------
# Path(__file__) is this file: .../core/config.py
# .parent      is: .../core/
# .parent.parent is: .../01_cannondale_bikes_assistant/   <- PROJECT_DIR
# .parent.parent.parent is: .../src/
# .parent.parent.parent.parent is: the repo root          <- REPO_ROOT
#
# We walk UP the tree instead of hardcoding a path so this works on any
# machine regardless of where the repo is cloned.

REPO_ROOT: Path = Path(__file__).resolve().parents[3]
PROJECT_DIR: Path = REPO_ROOT / "src" / "01_cannondale_bikes_assistant"


# ---------------------------------------------------------------------------
# Section 2 — Settings class
# ---------------------------------------------------------------------------
# BaseSettings (from pydantic-settings) automatically reads each field from
# environment variables. The alias="OPENAI_API_KEY" tells it which env var
# name to look for. If a required field is missing, it raises a clear error
# at startup — not silently later when you try to use it.

class Settings(BaseSettings):

    model_config = SettingsConfigDict(
        env_file=str(REPO_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",       # ignore unrelated keys in .env (e.g. YOUTUBE_API_KEY)
        case_sensitive=False,
    )

    # --- Required: the app cannot start without these ---
    openai_api_key: SecretStr = Field(..., alias="OPENAI_API_KEY")
    mongo_db_uri: SecretStr = Field(..., alias="MONGO_DB_URI")

    # --- Optional: needed for Phase 1+ features ---
    cohere_api_key: Optional[SecretStr] = Field(default=None, alias="COHERE_API_KEY")

    # --- MongoDB: which database, collection, and index to use ---
    mongo_db_name: str = Field(default="cannondale_bikes_db", alias="CANNONDALE_DB_NAME")
    mongo_collection: str = Field(default="bikes_collection", alias="CANNONDALE_COLLECTION")
    vector_index_name: str = Field(default="vector_index", alias="VECTOR_INDEX_NAME")

    # --- Model settings: sensible defaults, override via .env if needed ---
    llm_model: str = Field(default="gpt-4o", alias="LLM_MODEL")
    embedding_model: str = Field(default="text-embedding-ada-002", alias="EMBEDDING_MODEL")
    llm_temperature: float = Field(default=0.1, alias="LLM_TEMPERATURE")
    retriever_k: int = Field(default=5, alias="RETRIEVER_K")

    # --- Convenience properties ---
    # SecretStr hides the value in logs/repr. Call .get_secret_value() to
    # get the plain string when you actually need to pass it to OpenAI/Mongo.
    @property
    def openai_key(self) -> str:
        return self.openai_api_key.get_secret_value()

    @property
    def mongo_uri(self) -> str:
        return self.mongo_db_uri.get_secret_value()

    @property
    def cohere_key(self) -> Optional[str]:
        return self.cohere_api_key.get_secret_value() if self.cohere_api_key else None


# ---------------------------------------------------------------------------
# Section 3 — get_settings()
# ---------------------------------------------------------------------------
# @lru_cache means: run this function once, cache the result, return the
# same object on every subsequent call. So we only parse .env once per
# process, and every module that calls get_settings() gets the same instance.

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# Run this file directly to verify config loads correctly.
# In VS Code / Cursor: open this file and click "Run" or press Shift+Enter
# in an interactive window.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    s = get_settings()

    print("=== config.py smoke test ===\n")
    print(f"repo root      : {REPO_ROOT}")
    print(f"openai loaded  : {bool(s.openai_key)}")
    print(f"mongo loaded   : {bool(s.mongo_uri)}")
    print(f"llm model      : {s.llm_model}")
    print(f"embedding model: {s.embedding_model}")
    print(f"retriever k    : {s.retriever_k}")
    print(f"mongo db       : {s.mongo_db_name}")
    print(f"collection     : {s.mongo_collection}")
    print(f"vector index   : {s.vector_index_name}")
    print("\n=== OK ===")
