"""The shared NL2SQL loop used by both SQL agents."""
import pytest

import nl2sql_core as core


def test_strip_fences():
    assert core.strip_fences("```sql\nSELECT 1\n```") == "SELECT 1"
    assert core.strip_fences("  SELECT 2  ") == "SELECT 2"
    assert core.strip_fences("```\nSELECT 3\n```") == "SELECT 3"


def test_real_error_keeps_tail_and_drops_echoed_query():
    sql = "SELECT " + "x, " * 300 + "y FROM t"
    exc = Exception(f"(psycopg2.errors.UndefinedColumn) column \"zzz\" does not exist\n[SQL: {sql}]\nreal reason: column zzz")
    msg = core.real_error(exc, sql)
    assert "<query>" in msg and "column zzz" in msg and len(msg) <= core.MAX_ERROR_CHARS


def test_success_first_try():
    out = core.run_with_retry(lambda e: "SELECT 1", lambda s: [1])
    assert out.error is None and out.result == [1] and [a.status for a in out.attempts] == ["executed"]


def test_retry_gets_the_real_error_and_succeeds():
    seen = []
    def gen(err):
        seen.append(err)
        return "bad" if err is None else "good"
    def ex(sql):
        if sql == "bad":
            raise RuntimeError("syntax error near bad")
        return "rows"
    out = core.run_with_retry(gen, ex)
    assert out.result == "rows" and seen[0] is None and "syntax error" in seen[1]
    assert [a.status for a in out.attempts] == ["error", "executed"]


def test_validator_rejection_is_fed_back_and_can_recover():
    def validate(sql):
        if "dim_customer" in sql and "masked" not in sql:
            raise core.Rejected("use the masked view")
    gens = iter(["SELECT * FROM dim_customer", "SELECT * FROM dim_customer_masked"])
    out = core.run_with_retry(lambda e: next(gens), lambda s: "ok", validate)
    assert out.result == "ok" and out.attempts[0].status == "rejected"


def test_gives_up_after_max_attempts_and_reports_last_error():
    out = core.run_with_retry(lambda e: "bad", lambda s: (_ for _ in ()).throw(RuntimeError("boom")), max_attempts=2)
    assert out.result is None and "boom" in out.error and len(out.attempts) == 2


def test_executor_not_called_when_rejected():
    called = []
    def validate(sql):
        raise core.Rejected("no")
    core.run_with_retry(lambda e: "x", lambda s: called.append(s), validate)
    assert called == []


def test_on_attempt_callback_numbering():
    log = []
    core.run_with_retry(lambda e: "bad" if e is None else "ok",
                        lambda s: (_ for _ in ()).throw(RuntimeError("e")) if s == "bad" else 1,
                        on_attempt=lambda a, n: log.append((n, a.status)))
    assert log == [(1, "error"), (2, "executed")]
