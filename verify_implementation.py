# tests/verify_implementation.py
# AGPL v3 / VikaasLoop
#
# Smoke tests for core components

import os

os.environ["BITSANDBYTES_NO_CUDA"] = "1"
import asyncio
import gc
import json
import logging
import os
import tempfile
import time

import jwt

logging.basicConfig(level=logging.INFO)


def _remove_db_safe(path: str, retries: int = 8, delay: float = 0.3) -> None:
    """Windows safe SQLite file removal for WAL mode sidecars."""
    targets = [path, f"{path}-wal", f"{path}-shm"]
    for target in targets:
        if not os.path.exists(target):
            continue
        for attempt in range(retries):
            try:
                os.remove(target)
                break
            except PermissionError:
                if attempt < retries - 1:
                    time.sleep(delay)
                else:
                    print(f"  Warning: could not remove {target} locked.")


# ===========================================================================
# Formatter & DataGen
# ===========================================================================


async def verify_logic_units():
    print("\n=== Verifying Logic Units ===")
    from agents.datagen_agent import _deduplicate, clean_gemini_json_response
    from utils.formatter import format_training_pair

    # Formatter Test
    res = format_training_pair("Q", "A")
    assert "### Instruction:\nQ" in res and "### Response:\nA" in res
    print("  Formatter: PASS")

    # JSON Cleaner Test
    # FIXED: Defined the variable 'wrapped' correctly and separated the assertion.
    wrapped = '```json\n{"test": "data"}\n```'
    cleaned = clean_gemini_json_response(wrapped)
    assert "```" not in cleaned
    print("  JSON Cleaner: PASS")

    # Deduplication Test
    pairs = [
        {"instruction": "A", "response": "1"},
        {"instruction": "A", "response": "2"},
    ]
    assert len(_deduplicate(pairs)) == 1
    print("  Deduplicator: PASS")


# ===========================================================================
# Security & Config
# ===========================================================================


async def verify_security_and_config():
    print("\n=== Verifying Security & Config ===")
    from agents.orchestrator import sanitize_run_id
    from config import settings

    # Config Consistency Test
    s1 = settings.get_jwt_secret
    s2 = settings.get_jwt_secret
    assert s1 == s2 and len(s1) > 10
    print("  Config (JWT Consistency): PASS")

    # JWT Integrity Test
    payload = {"scope": "ws:loop"}
    token = jwt.encode(payload, settings.get_jwt_secret, algorithm="HS256")
    decoded = jwt.decode(token, settings.get_jwt_secret, algorithms=["HS256"])
    assert decoded["scope"] == "ws:loop"
    print("  JWT (Encode/Decode): PASS")

    # Sanitizer Test
    assert sanitize_run_id("valid_id_123") == "valid_id_123"
    try:
        sanitize_run_id("../evil")
        assert False
    except ValueError:
        print("  Orchestrator Sanitizer: PASS")


# ===========================================================================
# Data Components
# ===========================================================================


async def verify_data_components():
    print("\n=== Verifying Data Components ===")
    from agents.orchestrator import DataPartitioner
    from agents.skills_library import SkillsLibrary

    # DataPartitioner Test
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
    ) as f:
        tmp_path = f.name
        for i in range(20):
            f.write(json.dumps({"instruction": f"Q{i}", "response": f"A{i}"}) + "\n")

    try:
        eval_path = DataPartitioner.split(tmp_path, "test_run", eval_size=5)
        with open(eval_path) as ef:
            assert len(ef.readlines()) == 5
        print("  DataPartitioner: PASS")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        if "eval_path" in locals() and os.path.exists(eval_path):
            os.remove(eval_path)

    # Skills Library vectorized similarity test
    db_path = "data/test_verify.db"
    _remove_db_safe(db_path)
    lib = SkillsLibrary(db_path=db_path)
    lib.update_strategy_score("Python test", "StratA", "coding", 1, 0.9)
    lib.update_strategy_score("Python test", "StratA", "coding", 2, 0.95)

    tops = lib.get_top_strategies("Python coding")
    assert len(tops) > 0 and tops[0] == "StratA"
    print("  Skills Library (Vectorized): PASS")

    del lib
    gc.collect()
    time.sleep(0.5)
    _remove_db_safe(db_path)


# ===========================================================================
# Main Execution
# ===========================================================================
async def main():
    print("=" * 60)
    print("  VikaasLoop Component Verification Suite")
    print("=" * 60)
    try:
        await verify_logic_units()
        await verify_security_and_config()
        await verify_data_components()
        print("\n" + "=" * 60)
        print(" ALL VERIFICATIONS PASSED")
        print(" VikaasLoop is ready for deployment.")
        print("=" * 60)
    except Exception as e:
        print(f"!! VERIFICATION FAILED: {str(e)}")
        import traceback

        traceback.print_exc()  # This will print the exact line number and error
        import sys

        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
