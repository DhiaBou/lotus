from __future__ import annotations

import sqlite3
from typing import Dict, List, Optional, Sequence, Tuple

from lotus.fts_store.cs import CS


class SQLiteFTSStore(CS):
    def __init__(self) -> None:
        super().__init__()

    def _ensure(self):
        assert self.index_dir, "Index not loaded. Call load_index(index_dir) first."
        con = sqlite3.connect(self.index_dir)
        try:
            con.execute("PRAGMA journal_mode=WAL;")
            # Use unicode61 tokenizer with options: remove diacritics, lowercase
            con.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS caps 
                USING fts5(
                    rid UNINDEXED, 
                    caption, 
                    tokenize='unicode61 remove_diacritics 2'
                );
            """)
            con.commit()
        finally:
            con.close()

    def index(self, ids: Sequence[str], captions: Sequence[str], index_dir: str, **kwargs) -> None:
        if len(ids) != len(captions):
            raise ValueError("ids and captions must have same length")
        self.load_index(index_dir)
        con = sqlite3.connect(self.index_dir)
        try:
            cur = con.cursor()
            cur.executemany("DELETE FROM caps WHERE rid=?;", [(str(i),) for i in ids])
            cur.executemany("INSERT INTO caps(rid, caption) VALUES (?, ?);", list(zip(map(str, ids), captions)))
            con.commit()
        finally:
            con.close()

    def load_index(self, index_dir: str) -> None:
        self.index_dir = index_dir
        self._ensure()

    def __call__(self, query: str, K: int, ids: Optional[Sequence[str]] = None, **kwargs) -> List[Tuple[str, float]]:
        self._ensure()
        con = sqlite3.connect(self.index_dir)
        try:
            cur = con.cursor()
            base = "SELECT rid, bm25(caps) AS rank FROM caps WHERE caps MATCH ?"
            params: list = [query]
            if ids:
                marks = ",".join(["?"] * len(ids))
                base += f" AND rid IN ({marks})"
                params.extend([str(i) for i in ids])
            base += " ORDER BY rank ASC LIMIT ?;"
            params.append(int(K))
            cur.execute(base, params)
            return [(rid, -float(rank)) for rid, rank in cur.fetchall()]  # invert so higher=better
        finally:
            con.close()

    def get_captions_from_index(self, index_dir: str, ids: Sequence[str]) -> Dict[str, str]:
        self.load_index(index_dir)
        ids = list(dict.fromkeys(map(str, ids)))
        out: Dict[str, str] = {}
        if not ids:
            return out
        con = sqlite3.connect(self.index_dir)
        try:
            cur = con.cursor()
            for i in range(0, len(ids), 800):
                part = ids[i: i + 800]
                q = f"SELECT rid, caption FROM caps WHERE rid IN ({','.join(['?'] * len(part))});"
                cur.execute(q, part)
                for rid, cap in cur.fetchall():
                    out[rid] = cap
        finally:
            con.close()
        return out
