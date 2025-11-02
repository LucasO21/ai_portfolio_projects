

from langchain_core.output_parsers import BaseOutputParser

import re

def extract_sql_code(text: str):
    """
    Extracts the SQL query from a block of text. Handles:
    1) SQLQuery: ```sql ...``` fences
    2) ```sql ...``` fences
    3) ``` … ``` fences containing a "SELECT"
    4) SQLQuery: … (no fences)
    5) Bare "SELECT" …; up to semicolon
    6) SQL: SQLQuery: ...
    7) Just raw SQL starting with SELECT
    Returns the SQL (trimmed), or None if no query found.
    """
    patterns = [
        # 1) SQLQuery: ```sql ...```
        r"SQLQuery:\s*```sql\s*(?P<sql>[\s\S]+?)```",
        r"SQL: SQLQuery:\s*```sql\s*(?P<sql>[\s\S]+?)```",
        # 2) ```sql ...```
        r"```sql\s*(?P<sql>[\s\S]+?)```",
        # 3) ``` … ``` containing "SELECT"
        r"```(?:[\s\S]*?)\s*(?P<sql>SELECT[\s\S]+?)```",
        # 4) SQLQuery: … (grab until a blank line or end)
        r"SQLQuery:\s*(?P<sql>[\s\S]+?)(?=\n\s*\n|$)",
        # 5) Bare "SELECT" …; up to semicolon
        r"(?P<sql>SELECT[\s\S]+?;)(?=\s|$)",
        # 6) Raw SELECT without semicolon (NEW - catches your case!)
        r"(?P<sql>SELECT[\s\S]+?)(?=\n\s*\n|$)",
    ]

    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            sql = m.group("sql").strip()
            # strip any wrapping quotes
            if (sql.startswith(("'", '"')) and sql.endswith(("'", '"'))):
                sql = sql[1:-1].strip()
            return sql

    # If nothing matched, return the whole text if it starts with SELECT
    if text.strip().upper().startswith('SELECT'):
        return text.strip()

    return None

def audit_sql_code(text: str):
    sql = re.sub(r'^SQLQuery:\s*', '', text, flags=re.IGNORECASE)
    return sql


def extract_python_code(text: str):
    """
    Extracts Python code from a block of text. Handles:
      1) ```python ... ``` fences
      2) ``` ... ``` fences containing Python constructs
      3) Bare 'def' or 'import' blocks up to a blank line
    Returns the code (trimmed), or None if no code found.
    """
    patterns = [
        # 1) ```python ... ```
        r"```python\s*(?P<code>[\s\S]+?)```",
        # 2) ``` ... ``` containing a Python keyword
        r"```(?:[\s\S]*?)\s*(?P<code>(?:def |import |class )[\s\S]+?)```",
        # 3) Bare def/import/class up to the next blank line or end
        r"(?P<code>(?:def |import |class )[\s\S]+?)(?=\n\s*\n|$)",
    ]

    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            code = m.group("code").strip()
            # strip any wrapping quotes
            if (code.startswith(("'", '"')) and code.endswith(("'", '"'))):
                code = code[1:-1].strip()
            return code

    return None


class SQLOutputParser(BaseOutputParser):
    def parse(self, text: str):
        sql_code = extract_sql_code(text)
        if sql_code is not None:
            return sql_code
        else:
            # Assume ```sql wasn't used
            return text

class PythonOutputParser(BaseOutputParser):
    def parse(self, text: str):
        python_code = extract_python_code(text)
        if python_code is not None:
            return python_code
        else:
            # Assume ```python wasn't used
            return text
