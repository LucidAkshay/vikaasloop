# agents/datagen_agent.py
# AGPL v3 - VikaasLoop
#
# DataGenAgent: generates diverse, high-quality training pairs using Gemini Flash.
# Runs generation, deduplication, quality scoring, and JSONL persistence.

import os
import json
import uuid
import asyncio
import logging
import re
from typing import Any, Dict, List, Optional

from google import genai
from google.genai import types
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from utils.formatter import format_training_pair
from config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration & Lazy Globals
# ---------------------------------------------------------------------------

MODEL_ID = "gemini-2.0-flash"
CONCURRENT_REQUESTS = 5
BATCH_SIZE = 20

_client: Optional[genai.Client] = None
_semaphore: Optional[asyncio.Semaphore] = None


def _get_client() -> genai.Client:
    """Lazy load the Gemini client to ensure settings are fully parsed first."""
    global _client
    if _client is None:
        if not settings.GEMINI_API_KEY:
            logger.warning("GEMINI_API_KEY not found in settings!")
        _client = genai.Client(api_key=settings.GEMINI_API_KEY)
    return _client


def _get_semaphore() -> asyncio.Semaphore:
    """
    Return the semaphore for the current running event loop.
    Created lazily per loop to avoid "attached to a different loop" errors.
    """
    global _semaphore
    if _semaphore is None:
        _semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)
    return _semaphore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def clean_gemini_json_response(raw_text: str) -> str:
    """Safely strip markdown code-block wrappers, including '```json' tags."""
    text = raw_text.strip()
    # Remove starting ``` or ```json
    text = re.sub(r"\s*```$", "", text)
    return text.strip()

# ---------------------------------------------------------------------------
# Gemini API call with retry
# ---------------------------------------------------------------------------

@retry(
    retry=retry_if_exception_type(Exception),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    stop=stop_after_attempt(5),
    reraise=True,
)
async def _call_gemini(prompt: str, mime: str = "application/json") -> str:
    """Single Gemini async call with exponential backoff on any error."""
    client = _get_client()
    response = await client.aio.models.generate_content(
        model=MODEL_ID,
        contents=prompt,
        config=types.GenerateContentConfig(response_mime_type=mime),
    )
    return response.text


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

_EXAMPLE_PAIR = '''{
  "instruction": "Explain what a Python list comprehension is and give an example.",
  "response": "A list comprehension is a concise way to create lists in Python. Example: squares = [x**2 for x in range(10)]. You can also filter: evens = [x for x in range(20) if x % 2 == 0]"
}'''


async def _generate_batch(
    task_description: str, strategy_hint: str, batch_size: int = BATCH_SIZE
) -> List[Dict[str, str]]:
    """Generate one batch of training pairs. Returns [] on any failure."""
    prompt = f"""You are generating training data for an AI model.

TASK: {task_description}
STRATEGY: {strategy_hint}

Here is an example of a HIGH QUALITY training pair:
{_EXAMPLE_PAIR}

Generate {batch_size} high-quality training pairs following the exact same JSON format.
Each pair must be directly relevant to the TASK.

Return ONLY a JSON array of objects. Each object must have exactly two keys:
"instruction" (string) and "response" (string).
Do NOT wrap in markdown. Return ONLY the raw JSON array starting with [ and ending with ].

JSON array:"""

    async with _get_semaphore():
        try:
            raw = await _call_gemini(prompt)
            cleaned = clean_gemini_json_response(raw)
            pairs = json.loads(cleaned)
            return pairs if isinstance(pairs, list) else []
        except json.JSONDecodeError as exc:
            logger.error(f"_generate_batch JSON parse failed: {exc}")
            return []
        except Exception as exc:
            logger.error(f"_generate_batch failed: {exc}")
            return []


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def _deduplicate(pairs: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Exact-match deduplication on the instruction field.
    O(n) via a set to avoid the O(n squared) SequenceMatcher approach.
    Filters out malformed pairs before deduplication.
    """
    seen: set = set()
    result: List[Dict[str, str]] = []
    for pair in pairs:
        if not isinstance(pair, dict):
            continue
        instruction = pair.get("instruction", "").strip()
        response = pair.get("response", "").strip()
        
        if not instruction or not response:
            continue
            
        if instruction in seen:
            continue
            
        seen.add(instruction)
        result.append({"instruction": instruction, "response": response})
        
    return result


# ---------------------------------------------------------------------------
# Quality scoring
# ---------------------------------------------------------------------------

async def _score_batch(pairs: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    """Score a batch of pairs 1 to 5 using Gemini. Returns pairs with quality_score added."""
    if not pairs:
        return []

    # Using json.dumps ensures any quotes inside the pairs are properly escaped.
    safe_pairs_json = json.dumps(pairs, indent=2)

    prompt = f"""Score the following instruction-response pairs for quality on a scale of 1 to 5.
1 = Very Poor, 5 = Excellent.
Criteria: clarity, factual correctness, and usefulness.

Pairs:
{safe_pairs_json}

Return a JSON array of integer scores in the same order as the input pairs.
Example for 3 pairs: [5, 4, 2]
Return ONLY the JSON array. No markdown, no explanation."""

    async with _get_semaphore():
        try:
            raw = await _call_gemini(prompt)
            data = json.loads(clean_gemini_json_response(raw))

            if isinstance(data, dict):
                scores = data.get("scores", [])
                if not scores:
                    scores = next((v for v in data.values() if isinstance(v, list)), [])
            elif isinstance(data, list):
                scores = data
            else:
                scores = []

        except Exception as exc:
            logger.error(f"_score_batch failed: {exc}")
            scores = []

    result = []
    for i, pair in enumerate(pairs):
        p = pair.copy()
        raw_score = scores[i] if i < len(scores) else 3
        try:
            p["quality_score"] = max(1, min(5, int(raw_score)))
        except (ValueError, TypeError):
            p["quality_score"] = 3
            
        p["text"] = format_training_pair(p["instruction"], p["response"])
        result.append(p)

    return result


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def _save_jsonl(run_id: str, pairs: List[Dict[str, Any]]) -> str:
    """Write pairs to data/generated/run_id.jsonl. Returns the file path."""
    out_dir = os.path.join("data", "generated")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{run_id}.jsonl")

    with open(path, "w", encoding="utf-8") as f:
        for pair in pairs:
            # Ensure text field is up to date before writing
            pair["text"] = format_training_pair(
                pair.get("instruction", ""), pair.get("response", "")
            )
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    return path


# ---------------------------------------------------------------------------
# Public pipeline
# ---------------------------------------------------------------------------

async def run_datagen_pipeline(
    task_description: str,
    strategy_hint: str,
    target_count: int = 100,
) -> Dict[str, Any]:
    """
    Full data generation pipeline:
      1. Generate batches in parallel
      2. Deduplicate exact matches
      3. Score quality in parallel batches of 10
      4. Persist to JSONL
      5. Return metadata dict

    Returns:
        run_id, count, avg_quality_score, strategy_used, filename
    """
    batches_needed = max(1, -(-target_count // BATCH_SIZE))
    logger.info(
        f"DataGen: task={task_description[:50]!r}... strategy={strategy_hint[:50]!r}... "
        f"target={target_count} batches={batches_needed}"
    )

    # 1. Generate
    gen_tasks = [
        _generate_batch(task_description, strategy_hint) for _ in range(batches_needed)
    ]
    raw_batches = await asyncio.gather(*gen_tasks)
    all_pairs: List[Dict] = []
    for batch in raw_batches:
        all_pairs.extend(batch)
        
    logger.info(f"DataGen: generated {len(all_pairs)} raw pairs.")

    # 2. Deduplicate
    deduped = _deduplicate(all_pairs)
    logger.info(f"DataGen: {len(deduped)} pairs after deduplication.")

    # 3. Score in parallel batches of 10
    score_tasks = [
        _score_batch(deduped[i : i + 10]) for i in range(0, len(deduped), 10)
    ]
    scored_batches = await asyncio.gather(*score_tasks)
    final_pairs: List[Dict[str, Any]] = []
    for batch in scored_batches:
        final_pairs.extend(batch)

    # 4. Stats
    avg_score = (
        sum(int(p.get("quality_score", 3)) for p in final_pairs) / len(final_pairs)
        if final_pairs
        else 0.0
    )
    run_id = str(uuid.uuid4()).replace("-", "")[:16]

    # 5. Persist
    filename = await asyncio.to_thread(_save_jsonl, run_id, final_pairs)
    logger.info(f"DataGen: saved {len(final_pairs)} pairs to {filename}.")

    return {
        "run_id": run_id,
        "count": len(final_pairs),
        "avg_quality_score": round(avg_score, 2),
        "strategy_used": strategy_hint,
        "filename": filename,
    }

# EOF