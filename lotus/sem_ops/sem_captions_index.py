from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

import lotus
from lotus.cache import operator_cache


@pd.api.extensions.register_dataframe_accessor("sem_captions_index")
class SemCaptionsIndexAccessor:
    """DataFrame accessor for caption-based indexing/search."""

    def __init__(self, pandas_obj: Any) -> None:
        self._validate(pandas_obj)
        self._obj = pandas_obj
        self._obj.attrs.setdefault("cap_index_dirs", {})  # {col_name: index_dir}

    @staticmethod
    def _validate(obj: Any) -> None:
        if not isinstance(obj, pd.DataFrame):
            raise AttributeError("Must be a DataFrame")

    @operator_cache
    def __call__(self, col_name: str, index_dir: str, batch_size: int = 10) -> pd.DataFrame:
        """
        Index a column of image paths/URLs by generating captions and storing them in a caption store.

        Args:
            col_name: Column containing image paths/URLs.
            index_dir: SQLite path (or store dir) used by the CaptionStore implementation.
            batch_size: Captioning batch size.

        Returns:
            The same DataFrame with the index directory recorded in attrs.
        """
        lotus.logger.warning(
            "Do not reset the dataframe index to ensure proper functionality of get_captions_from_index"
        )

        cap = getattr(lotus.settings, "cm", None)  # CM
        store = getattr(lotus.settings, "cs", None)  # CS
        if cap is None or store is None:
            raise ValueError(
                "The retrieval model must be an instance of RM, and the vector store must be an instance of VS. "
                "Please configure a valid retrieval model using lotus.settings.configure()"
            )

        paths: List[str] = self._obj[col_name].tolist()
        ids: List[str] = [str(i) for i in self._obj.index.tolist()]

        captions: List[str] = cap(paths, batch_size=batch_size)  # CM: list[str], same order as paths
        if len(captions) != len(ids):
            raise RuntimeError("Caption count mismatch. Check input column and batching.")

        store.index(ids, captions, index_dir)  # CS.index(ids, captions, index_dir)
        self._obj.attrs["cap_index_dirs"][col_name] = index_dir
        return self._obj

    def attach_index(self, col_name: str, index_dir: str) -> pd.DataFrame:
        """Attach an existing caption index without re-captioning."""
        self._obj.attrs.setdefault("cap_index_dirs", {})
        self._obj.attrs["cap_index_dirs"][col_name] = index_dir
        return self._obj

    def _dir_for(self, col_name: str) -> str:
        idx = self._obj.attrs.get("cap_index_dirs", {}).get(col_name)
        if not idx:
            raise ValueError(
                f"No caption index for column '{col_name}'. "
                f"Call df.sem_captions_index('{col_name}', index_dir) first or use attach_index()."
            )
        return idx

    def search(self, query: str, col_name: str, K: int = 100, ids: Optional[Sequence[Any]] = None) -> pd.DataFrame:
        """
        Search captions using FTS-like query syntax, returning matching rows ordered by score.

        Adds a 'cap_score' column (higher is better).
        """
        store = getattr(lotus.settings, "cs", None)
        if store is None:
            raise ValueError(
                "The retrieval model must be an instance of RM, and the vector store must be an instance of VS. "
                "Please configure a valid retrieval model using lotus.settings.configure()"
            )

        index_dir = self._dir_for(col_name)
        store.load_index(index_dir)

        subset_ids_str: Optional[List[str]] = None
        if ids is not None:
            subset_ids_str = [str(i) for i in ids]

        hits: List[Tuple[str, float]] = store(query, K, ids=subset_ids_str)  # [(rid_str, score)]
        if not hits:
            return self._obj.iloc[0:0].copy()

        # Map string rids back to original index labels
        str2label: Dict[str, Any] = {str(i): i for i in self._obj.index.tolist()}
        labels: List[Any] = [str2label[rid] for rid, _ in hits if rid in str2label]
        scores: List[float] = [score for rid, score in hits if rid in str2label]

        out = self._obj.loc[labels].copy()
        out["cap_score"] = scores
        # already in hit order; no need to sort unless you want to be explicit:
        # out = out.sort_values("cap_score", ascending=False)
        return out

    def load(
        self, col_name: str, out_col: Optional[str] = None, ids: Optional[Sequence[Any]] = None, default: str = ""
    ) -> pd.DataFrame:
        """
        Hydrate captions from the store into a DataFrame column.

        Args:
            col_name: The column that was indexed (used to locate the index_dir).
            out_col: Destination column to write captions into (defaults to f"{col_name}_cap").
            ids: Optional subset of DataFrame index labels to hydrate. Defaults to all rows.
            default: Value to use when a caption is missing.

        Returns:
            The DataFrame with captions written to out_col.
        """
        store = getattr(lotus.settings, "cs", None)  # CS

        if store is None:
            raise ValueError(
                "The retrieval model must be an instance of RM, and the vector store must be an instance of VS. "
                "Please configure a valid retrieval model using lotus.settings.configure()"
            )

        index_dir = self._dir_for(col_name)
        out_col = out_col or f"{col_name}_cap"

        if ids is None:
            ids = list(self._obj.index)
        id_strs = [str(i) for i in ids]

        got = store.get_captions_from_index(index_dir, id_strs)  # {rid_str: caption}
        series = [got.get(str(i), default) for i in self._obj.index.tolist()]
        self._obj[out_col] = pd.Series(series, index=self._obj.index)
        return self._obj
