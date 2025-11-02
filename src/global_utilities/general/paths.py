import os
import sys
from pathlib import Path

def find_project_root(start: Path) -> Path:
    for parent in start.resolve().parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise FileNotFoundError("Could not find project root (missing pyproject.toml)")

ROOT_DIR = find_project_root(Path(__file__))

CANNONDALE_BIKES_ASSISTANT_DIR = ROOT_DIR / "src" / "01_cannondale_bikes_assistant"

LANGCHAIN_BEGINNER_MASTERCLASS_DIR = ROOT_DIR / "src" / "langchain_beginner_masterclass"

# DATABASE_DIR = ROOT_DIR / "database/"

# CRM_SQLITE_DATABASE_DIR = f"sqlite:///{DATABASE_DIR}/00_crm_database.sqlite"

SALES_PIPELINE_ASSISTANT_DIR = ROOT_DIR / "src" / "project_02_sales_pipeline_assistant"
SALES_PIPELINE_ASSISTANT_DB_DIR = ROOT_DIR / "database" / "sqlites" / "sales_pipeline.sqlite"


def find_project_root(current: Path, anchor: str = ".git"):
    for parent in current.parents:
        if (parent / anchor).exists():
            return parent
    return current  # fallback