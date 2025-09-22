import os
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.absolute().parent.parent.absolute()

CANNONDALE_BIKES_ASSISTANT_DIR = ROOT_DIR / "src" / "cannondale_bikes_assistant"

LANGCHAIN_BEGINNER_MASTERCLASS_DIR = ROOT_DIR / "src" / "langchain_beginner_masterclass"

# DATABASE_DIR = ROOT_DIR / "database/"

# CRM_SQLITE_DATABASE_DIR = f"sqlite:///{DATABASE_DIR}/00_crm_database.sqlite"