# agents/skills_library.py
# AGPL v3 - VikaasLoop
#
# SkillsLibrary: persistent store of "what data strategies worked on what tasks".
# Strategies are retrieved by cosine similarity of task embeddings, weighted by win rate.

import sqlite3
import numpy as np
import os
import logging
from contextlib import closing
from typing import List, Optional

logger = logging.getLogger(__name__)


class SkillsLibrary:
    """
    SQLite-backed store of (task_description, strategy_name, win_rate) triples.
    """

    def __init__(
        self,
        db_path: str = "data/skills.db",
        model_name: str = "all-MiniLM-L6-v2",
    ):
        self.db_path = db_path
        self._model_name = model_name
        self._model = None
        self._init_db()

    @property
    def _encoder(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading SentenceTransformer '{self._model_name}'")
            self._model = SentenceTransformer(self._model_name)
            logger.info("SentenceTransformer loaded.")
        return self._model

    def _encode(self, text: str) -> np.ndarray:
        return self._encoder.encode(text, normalize_embeddings=True).astype(np.float32)

    def _init_db(self) -> None:
        db_dir = os.path.dirname(self.db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
            
        with closing(sqlite3.connect(self.db_path, timeout=30.0)) as conn:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS strategies (
                    id               INTEGER PRIMARY KEY AUTOINCREMENT,
                    strategy_name    TEXT    NOT NULL,
                    task_type        TEXT    NOT NULL DEFAULT 'general',
                    win_rate         REAL    NOT NULL DEFAULT 0.0,
                    iteration_number INTEGER NOT NULL DEFAULT 0,
                    task_embedding   BLOB    NOT NULL,
                    task_description TEXT    NOT NULL,
                    UNIQUE (strategy_name, task_description)
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_task_type "
                "ON strategies (task_type);"
            )
            conn.commit()

    def get_top_strategies(
        self,
        task_description: str,
        task_type: Optional[str] = None,
        top_k: int = 3,
    ) -> List[str]:
        query_emb = self._encode(task_description)

        with closing(sqlite3.connect(self.db_path, timeout=30.0)) as conn:
            sql = "SELECT strategy_name, task_embedding, win_rate FROM strategies"
            params: tuple = ()
            if task_type:
                sql += " WHERE task_type = ?"
                params = (task_type,)
            rows = conn.execute(sql, params).fetchall()

        if not rows:
            return []

        names = []
        embs = []
        win_rates = []

        for name, emb_blob, win_rate in rows:
            emb = np.frombuffer(emb_blob, dtype=np.float32)
            if emb.shape == query_emb.shape:
                names.append(name)
                embs.append(emb)
                win_rates.append(float(win_rate))

        if not embs:
            return []

        embs_matrix = np.array(embs)
        win_rates_array = np.array(win_rates)

        similarities = np.dot(embs_matrix, query_emb)
        scores = similarities * (1.0 + win_rates_array)

        scored_indices = np.argsort(scores)[::-1]

        seen = set()
        top = []
        
        for idx in scored_indices:
            name = names[idx]
            if name not in seen:
                top.append(name)
                seen.add(name)
            if len(top) >= top_k:
                break

        return top

    def update_strategy_score(
        self,
        task_description: str,
        strategy_name: str,
        task_type: str,
        iteration_number: int,
        win_rate: float,
    ) -> None:
        embedding = self._encode(task_description).tobytes()

        with closing(sqlite3.connect(self.db_path, timeout=30.0)) as conn:
            conn.execute("""
                INSERT INTO strategies
                    (strategy_name, task_type, win_rate, iteration_number,
                     task_embedding, task_description)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT (strategy_name, task_description)
                DO UPDATE SET
                    win_rate         = excluded.win_rate,
                    iteration_number = excluded.iteration_number,
                    task_type        = excluded.task_type
            """, (strategy_name, task_type, win_rate,
                  iteration_number, embedding, task_description))
            conn.commit()

        logger.debug(
            f"Skills updated: strategy='{strategy_name}' "
            f"win_rate={win_rate:.3f} iteration={iteration_number}"
        )


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    lib = SkillsLibrary(db_path="data/skills_test.db")

    lib.update_strategy_score("Python coding task", "CodeOptimized", "coding", 1, 0.85)
    lib.update_strategy_score("Data analysis request", "DataPro", "analysis", 1, 0.92)
    # Update the same strategy with a new win rate to ensure no duplicate row is created
    lib.update_strategy_score("Python coding task", "CodeOptimized", "coding", 2, 0.91)

    print("Top strategies for 'Write a script for data cleaning':")
    print(lib.get_top_strategies("Write a script for data cleaning"))