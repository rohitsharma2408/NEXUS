"""Fails if the number-verification step is unwired. Run: python3 project1_agentic/tests/test_wiring.py"""
import os

AGENTS = os.path.join(os.path.dirname(__file__), "..", "agents")


def _read(name):
    return open(os.path.join(AGENTS, name)).read()


def test_supervisor_calls_verifier():
    s = _read("supervisor.py")
    assert "number_verifier.verify_answer(" in s, "supervisor does not call verify_answer"
    assert "evidence_checker.apply_verification(" in s, "supervisor does not call apply_verification"
    assert 'report["verification"]' in s, "supervisor does not attach verification to the report"


def test_checker_defines_what_supervisor_calls():
    assert "def apply_verification" in _read("evidence_checker.py")
    assert "def verify_answer" in _read("number_verifier.py")


if __name__ == "__main__":
    test_supervisor_calls_verifier()
    test_checker_defines_what_supervisor_calls()
    print("wiring ok")
