"""
Shared NL-to-SQL core used by BOTH SQL agents.

NEXUS had two diverging implementations of the same loop: agents/sql_agent.py (Postgres, the
production warehouse) and evaluation/bibench/runner.py (SQLite, unseen external schemas). Fixes
made in one did not reach the other (the "keep the real DB error, not the echoed query" fix, the
markdown-fence stripping, the single retry). This module holds everything that is dialect-
independent so there is exactly one copy:

  * strip_fences()      clean an LLM reply down to bare SQL
  * real_error()        extract the part of a DB exception the retry actually needs
  * run_with_retry()    generate -> validate -> execute, one retry fed the real error

Dialect-specific parts stay with each caller: the prompt, the validator, and the executor.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable

MAX_ERROR_CHARS = 400


def strip_fences(raw: str) -> str:
    sql = (raw or "").strip()
    sql = re.sub(r"^```(?:sql)?|```$", "", sql, flags=re.IGNORECASE | re.MULTILINE).strip()
    return sql


def real_error(exc: Exception, sql: str | None) -> str:
    """DB drivers echo the whole failing query before the actual reason. Replace the echo and keep
    the TAIL, which is where the reason lives (head-truncation threw it away)."""
    msg = str(exc)
    if sql and sql in msg:
        msg = msg.replace(sql, "<query>")
    return msg[-MAX_ERROR_CHARS:]


class Rejected(Exception):
    """Raised by a validator to reject generated SQL before execution."""


@dataclass
class Attempt:
    sql: str
    status: str                 # "executed" | "rejected" | "error"
    reason: str | None = None


@dataclass
class Outcome:
    sql: str
    result: object | None            # whatever the executor returned
    error: str | None                # None on success
    attempts: list[Attempt] = field(default_factory=list)


def run_with_retry(
    generate: Callable[[str | None], str],
    execute: Callable[[str], object],
    validate: Callable[[str], None] | None = None,
    max_attempts: int = 2,
    on_attempt: Callable[[Attempt, int], None] | None = None,
) -> Outcome:
    """generate(retry_error)->sql ; validate(sql) raises Rejected/Exception ; execute(sql)->result."""
    error: str | None = None
    sql = ""
    attempts: list[Attempt] = []
    for n in range(1, max_attempts + 1):
        sql = strip_fences(generate(error))
        if validate is not None:
            try:
                validate(sql)
            except Exception as e:     # validator errors are fed back to the model like DB errors
                error = str(e)
                a = Attempt(sql, "rejected", error)
                attempts.append(a)
                if on_attempt:
                    on_attempt(a, n)
                continue
        try:
            result = execute(sql)
        except Exception as e:
            error = real_error(e, sql)
            a = Attempt(sql, "error", error)
            attempts.append(a)
            if on_attempt:
                on_attempt(a, n)
            continue
        a = Attempt(sql, "executed")
        attempts.append(a)
        if on_attempt:
            on_attempt(a, n)
        return Outcome(sql, result, None, attempts)
    return Outcome(sql, None, error, attempts)
