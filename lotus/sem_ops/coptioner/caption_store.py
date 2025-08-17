# caption_store.py
import sqlite3
from typing import Iterable, List, Tuple, Optional

class CaptionFTSStore:
    """
    Simple SQLite FTS5-backed caption store.
    - Table: caps(rid TEXT UNINDEXED, caption TEXT)
    - Uses bm25() for ranking; lower is better.
    """
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        con = sqlite3.connect(self.db_path)
        try:
            con.execute("PRAGMA journal_mode=WAL;")
            con.execute("CREATE VIRTUAL TABLE IF NOT EXISTS caps USING fts5(rid UNINDEXED, caption);")
            con.commit()
        finally:
            con.close()

    def clear(self):
        con = sqlite3.connect(self.db_path)
        try:
            con.execute("DELETE FROM caps;")
            con.commit()
        finally:
            con.close()

    def index(self, pairs: Iterable[Tuple[str, str]]):
        pairs = list(pairs)           # ← materialize so we can iterate twice
        if not pairs:
            return

        con = sqlite3.connect(self.db_path)
        try:
            cur = con.cursor()
            rids = [(rid,) for rid, _ in pairs]
            cur.executemany("DELETE FROM caps WHERE rid=?;", rids)
            cur.executemany("INSERT INTO caps(rid, caption) VALUES (?, ?);", pairs)
            con.commit()
        finally:
            con.close()

    def search(self, query: str, limit: int = 1000) -> List[Tuple[str, float]]:
        """
        Returns [(rid, score)] with score=-bm25 (higher is better).
        Use FTS5 MATCH syntax: 'dog AND indoor NOT toy'
        """
        con = sqlite3.connect(self.db_path)
        try:
            cur = con.cursor()
            cur.execute(
                "SELECT rid, bm25(caps) AS rank FROM caps WHERE caps MATCH ? ORDER BY rank ASC LIMIT ?;",
                (query, int(limit)),
            )
            rows = cur.fetchall()
            # convert to a “higher is better” score
            return [(rid, -float(rank)) for (rid, rank) in rows]
        finally:
            con.close()
