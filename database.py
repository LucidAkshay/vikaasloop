# database.py
# AGPL v3 : VikaasLoop
#
# SQLite persistence for evaluation experiments and comparison results.
# Uses WAL journal mode for concurrent read/write access.

import logging
import os
import sqlite3
from contextlib import closing, contextmanager
from typing import Any, Dict, List, Optional

from config import settings

logger = logging.getLogger(__name__)


@contextmanager
def _get_conn():
    """
    Context managed SQLite connection.
    Guarantees physical connection closure via closing() and transaction safety.
    Always uses WAL mode and row_factory for dict like access.
    """
    db_path = os.path.abspath(settings.DB_PATH)

    # Ensure the directory exists before attempting connection
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    # CRITICAL FIX : sqlite3.connect context manager does NOT close the connection.
    # We wrap it in closing() to ensure file descriptors are released to the OS.
    with closing(sqlite3.connect(db_path, timeout=30.0)) as conn:
        conn.row_factory = sqlite3.Row
        # Enable WAL mode for high concurrency
        conn.execute("PRAGMA journal_mode=WAL;")
        try:
            yield conn
            conn.commit()
        except Exception as exc:
            conn.rollback()
            logger.error(f"Database transaction failed : {exc}")
            raise


def init_db() -> None:
    """Create tables if they do not exist. Safe to call on every startup."""
    with _get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS experiments (
                id                INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id            TEXT    UNIQUE NOT NULL,
                created_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                base_model        TEXT    NOT NULL,
                finetuned_model   TEXT    NOT NULL DEFAULT '',
                win_rate          REAL    NOT NULL,
                score_delta       REAL,
                total_comparisons INTEGER NOT NULL,
                a_wins            INTEGER NOT NULL,
                b_wins            INTEGER NOT NULL,
                ties              INTEGER NOT NULL,
                task_description  TEXT    NOT NULL DEFAULT ''
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS comparisons (
                id                 INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id      INTEGER NOT NULL,
                prompt             TEXT    NOT NULL,
                base_response      TEXT    NOT NULL,
                finetuned_response TEXT    NOT NULL,
                judge_verdict      TEXT    NOT NULL,
                created_at         TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (experiment_id) REFERENCES experiments (id)
            )
        """)
        # Index for performance on the React Dashboard sidebar
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_exp_created "
            "ON experiments (created_at DESC);"
        )
        logger.info("Database schema verified/initialized.")


def save_experiment(
    run_id: str,
    base_model: str,
    finetuned_model: str,
    win_rate: float,
    score_delta: Optional[float],
    total_comparisons: int,
    a_wins: int,
    b_wins: int,
    ties: int,
    comparisons: List[Dict[str, Any]],
    task_description: str = "",
) -> int:
    """
    Persist an evaluation run and its per prompt comparison data.
    Uses executemany for high performance batch insertion.
    """
    with _get_conn() as conn:
        cursor = conn.execute(
            """
            INSERT INTO experiments (
                run_id, base_model, finetuned_model, win_rate, score_delta,
                total_comparisons, a_wins, b_wins, ties, task_description
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                run_id,
                base_model,
                finetuned_model,
                win_rate,
                score_delta,
                total_comparisons,
                a_wins,
                b_wins,
                ties,
                task_description,
            ),
        )
        exp_id = cursor.lastrowid

        # Prepare batch for comparisons to avoid N execution overhead
        comp_data = []
        for comp in comparisons:
            # Robust key mapping for EvalAgent flexibility
            f_resp = (
                comp.get("finetuned_response") or comp.get("adapter_response") or ""
            )
            verdict = comp.get("verdict") or comp.get("judge_verdict") or "T"

            comp_data.append(
                (
                    exp_id,
                    comp.get("prompt", ""),
                    comp.get("base_response", ""),
                    f_resp,
                    verdict,
                )
            )

        if comp_data:
            conn.executemany(
                """
                INSERT INTO comparisons
                    (experiment_id, prompt, base_response, finetuned_response, judge_verdict)
                VALUES (?, ?, ?, ?, ?)
            """,
                comp_data,
            )

    logger.info(
        f"Saved experiment {run_id} (ID: {exp_id}) with {len(comp_data)} comparisons."
    )
    return exp_id or 0


def get_experiments() -> List[Dict[str, Any]]:
    """Return all experiments ordered newest first for the dashboard."""
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM experiments ORDER BY created_at DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def get_latest_experiment() -> Optional[Dict[str, Any]]:
    """Fetches the most recent run to calculate the win rate delta."""
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM experiments ORDER BY created_at DESC LIMIT 1"
        ).fetchone()
    return dict(row) if row else None


def get_experiment_details(run_id: str) -> Optional[Dict[str, Any]]:
    """Return a full experiment object including its nested list of comparisons."""
    with _get_conn() as conn:
        exp_row = conn.execute(
            "SELECT * FROM experiments WHERE run_id = ?", (run_id,)
        ).fetchone()

        if not exp_row:
            return None

        exp = dict(exp_row)
        comp_rows = conn.execute(
            "SELECT * FROM comparisons WHERE experiment_id = ? ORDER BY id ASC",
            (exp["id"],),
        ).fetchall()

        exp["comparisons"] = [dict(r) for r in comp_rows]
        # UI safety fallback
        exp["total_comparisons"] = exp.get("total_comparisons", len(exp["comparisons"]))

    return exp
