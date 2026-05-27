"""
test_phase1.py
==============
Phase 1 checkpoint tests — Model Abstraction Layer.

Tests:
  1. config.py loads correctly and validates backend
  2. ModelProvider.invoke() returns a non-empty string
  3. ModelProvider.batch() returns multiple responses
  4. get_provider() returns the same singleton on repeat calls
  5. inference_by_Ollama.get_response() returns correct dict shape
  6. inference_by_Ollama.summarize_dialog() returns a non-empty string
  7. Backend switching — same prompt, two providers, both return strings
  8. Invalid backend raises ValueError

Run:
    python test_phase1.py                   # tests active backend from .env
    python test_phase1.py --backend openai  # test a specific backend
    python test_phase1.py --all             # test ALL configured backends
"""

import sys
import argparse
import traceback
from typing import List, Tuple


PASS = "✓"
FAIL = "✗"
SKIP = "○"

results: List[Tuple[str, str, str]] = []   # (status, name, detail)


def record(status: str, name: str, detail: str = "") -> None:
    results.append((status, name, detail))
    icon = {"pass": PASS, "fail": FAIL, "skip": SKIP}[status]
    print(f"  {icon}  {name}" + (f"  →  {detail}" if detail else ""))


def section(title: str) -> None:
    print(f"\n{'─'*50}")
    print(f"  {title}")
    print(f"{'─'*50}")


# ---------------------------------------------------------------------------
# Test 1 — config.py
# ---------------------------------------------------------------------------

def test_config() -> None:
    section("1. config.py")
    try:
        from config import settings
        record("pass", "settings imported", f"backend={settings.model_backend!r}  model={settings.model_id!r}")
        record("pass", "settings.summary()", settings.summary().split('\n')[0])
    except Exception as e:
        record("fail", "settings import", str(e))
        return

    # Invalid backend detection
    try:
        from config import Settings
        Settings.__new__(Settings)   # bypass __init__
        bad = Settings.__new__(Settings)
        bad.model_backend      = "fakebackend"  # type: ignore
        bad.openai_api_key     = ""
        bad.anthropic_api_key  = ""
        bad.google_api_key     = ""
        bad.__post_init__()
        record("fail", "invalid backend raises ValueError", "no error raised!")
    except ValueError as e:
        record("pass", "invalid backend raises ValueError", str(e)[:60])
    except Exception as e:
        record("fail", "invalid backend raises ValueError", str(e))


# ---------------------------------------------------------------------------
# Test 2 — ModelProvider basics
# ---------------------------------------------------------------------------

def test_model_provider_basics() -> None:
    section("2. ModelProvider — invoke & batch")
    try:
        from model_provider import ModelProvider
        provider = ModelProvider()
        record("pass", "ModelProvider() constructed", str(provider))
    except Exception as e:
        record("fail", "ModelProvider() construction", str(e))
        traceback.print_exc()
        return

    # invoke
    try:
        resp = provider.invoke("Respond with exactly the word: PONG")
        assert isinstance(resp, str) and len(resp) > 0
        record("pass", "invoke() returns non-empty string", repr(resp[:60]))
    except Exception as e:
        record("fail", "invoke()", str(e))

    # batch
    try:
        prompts = [
            "Respond with exactly the word: ONE",
            "Respond with exactly the word: TWO",
        ]
        resps = provider.batch(prompts)
        assert isinstance(resps, list) and len(resps) == 2
        assert all(isinstance(r, str) and len(r) > 0 for r in resps)
        record("pass", "batch() returns list of strings", f"{resps[0][:30]} | {resps[1][:30]}")
    except Exception as e:
        record("fail", "batch()", str(e))

    # generate alias
    try:
        resps2 = provider.generate(["Say: ALIAS"])
        assert isinstance(resps2, list)
        record("pass", "generate() alias works")
    except Exception as e:
        record("fail", "generate() alias", str(e))


# ---------------------------------------------------------------------------
# Test 3 — singleton
# ---------------------------------------------------------------------------

def test_singleton() -> None:
    section("3. get_provider() singleton")
    try:
        from model_provider import get_provider
        p1 = get_provider()
        p2 = get_provider()
        assert p1 is p2, "Expected the same object"
        record("pass", "get_provider() returns same instance on repeat calls")
    except Exception as e:
        record("fail", "get_provider() singleton", str(e))


# ---------------------------------------------------------------------------
# Test 4 — inference_by_Ollama
# ---------------------------------------------------------------------------

def test_inference_module() -> None:
    section("4. inference_by_Ollama.py")
    try:
        from inference_by_Ollama import get_response, summarize_dialog
        record("pass", "inference_by_Ollama imported")
    except Exception as e:
        record("fail", "inference_by_Ollama import", str(e))
        return

    context = (
        "The annual leave policy allows employees 20 days of paid leave per year. "
        "Leave must be requested at least 2 weeks in advance."
    )
    question = "How many days of annual leave do employees get?"

    try:
        result = get_response(question, context)
        assert isinstance(result, dict), "Expected dict"
        assert result.get("role") == "assistant", f"role={result.get('role')!r}"
        assert isinstance(result.get("content"), str)
        assert len(result["content"]) > 0
        record("pass", "get_response() shape correct", f"role={result['role']!r}  len={len(result['content'])}")
    except Exception as e:
        record("fail", "get_response()", str(e))

    try:
        summary = summarize_dialog(question, "Employees get 20 days of annual leave.")
        assert isinstance(summary, str) and len(summary) > 0
        record("pass", "summarize_dialog() returns non-empty string", summary[:60])
    except Exception as e:
        record("fail", "summarize_dialog()", str(e))


# ---------------------------------------------------------------------------
# Test 5 — backend switching (only runs when --all or specific --backend)
# ---------------------------------------------------------------------------

def test_backend_switching(backends_to_test: List[str]) -> None:
    section(f"5. Backend switching — {backends_to_test}")
    from model_provider import ModelProvider
    from config import settings

    prompt = "Respond with exactly: 'Switch test passed.' and nothing else."

    for backend in backends_to_test:
        # Pick a sensible default model for each backend
        model_map = {
            "ollama":    settings.model_id or "llama3.1",
            "openai":    "gpt-4o-mini",
            "anthropic": "claude-haiku-4-5-20251001",
            "gemini":    "gemini-1.5-flash",
        }
        model_id = model_map.get(backend, settings.model_id)

        try:
            provider = ModelProvider(backend=backend, model_id=model_id)  # type: ignore
            resp = provider.invoke(prompt)
            assert isinstance(resp, str) and len(resp) > 0
            record("pass", f"backend={backend!r}  model={model_id!r}", repr(resp[:50]))
        except ImportError as e:
            record("skip", f"backend={backend!r}", f"package not installed: {e}")
        except Exception as e:
            record("fail", f"backend={backend!r}", str(e)[:80])


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary() -> bool:
    print(f"\n{'='*50}")
    print("  Summary")
    print(f"{'='*50}")
    passed = sum(1 for s, _, _ in results if s == "pass")
    failed = sum(1 for s, _, _ in results if s == "fail")
    skipped = sum(1 for s, _, _ in results if s == "skip")
    total = len(results)
    print(f"  {PASS} Passed : {passed}/{total}")
    if skipped:
        print(f"  {SKIP} Skipped: {skipped}  (missing packages for untested backends)")
    if failed:
        print(f"  {FAIL} Failed : {failed}")
        print("\n  Failed tests:")
        for s, name, detail in results:
            if s == "fail":
                print(f"    {FAIL}  {name}")
                if detail:
                    print(f"       {detail}")
    print(f"{'='*50}")

    if failed == 0:
        print("\n  ✅  Phase 1 checkpoint PASSED — model abstraction layer is working.")
    else:
        print("\n  ❌  Some tests failed. Fix the issues above before moving to Phase 2.")

    return failed == 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 1 checkpoint tests")
    parser.add_argument(
        "--backend",
        choices=["ollama", "openai", "anthropic", "gemini"],
        default=None,
        help="Test a specific backend (uses .env keys)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Test all four backends (skips any with missing packages/keys)",
    )
    args = parser.parse_args()

    print("=" * 50)
    print("  Phase 1 — Model Abstraction Layer Tests")
    print("=" * 50)

    test_config()
    test_model_provider_basics()
    test_singleton()
    test_inference_module()

    if args.all:
        test_backend_switching(["ollama", "openai", "anthropic", "gemini"])
    elif args.backend:
        test_backend_switching([args.backend])

    ok = print_summary()
    sys.exit(0 if ok else 1)
