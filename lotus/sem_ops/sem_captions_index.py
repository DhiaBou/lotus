# sem_captions_index.py
from typing import Any, Iterable, Optional, List, Tuple
from pathlib import Path
from PIL import Image
import pandas as pd

import lotus
from lotus.cache import operator_cache

# Optional: reuse your URL helpers if you caption URLs too
import io, re, requests
from urllib.parse import urlparse

from lotus.sem_ops.coptioner.caption_store import CaptionFTSStore
from lotus.sem_ops.coptioner.captioner import Captioner

IMG_EXT_RE = re.compile(r"\.(png|jpe?g|webp|bmp|gif|tiff?)($|\?)", re.IGNORECASE)
def looks_like_image_url(s: str) -> bool:
    try:
        u = urlparse(s)
        return (u.scheme in {"http", "https"}) and bool(u.netloc)
    except Exception:
        return False
def fetch_pil(url: str) -> Image.Image:
    r = requests.get(url, timeout=15); r.raise_for_status()
    return Image.open(io.BytesIO(r.content)).convert("RGB")

# local imports

@pd.api.extensions.register_dataframe_accessor("sem_captions_index")
class SemCaptionsIndexAccessor:
    """
    DataFrame accessor for:
      1) Captioning a column of images (paths or URLs)
      2) Storing captions in an FTS5 DB
      3) Retrieving “rows of importance” by caption query
    Keeps parity with your sem_index style.
    """
    def __init__(self, pandas_obj: Any) -> None:
        if not isinstance(pandas_obj, pd.DataFrame):
            raise AttributeError("Must be a DataFrame")
        self._obj = pandas_obj
        self._obj.attrs.setdefault("cap_index_dirs", {})  # map: out_col -> db_path

    # ---------- Build / index ----------

    @operator_cache
    def __call__(
            self,
            img_col: str,
            out_col: str,
            db_path: str,
            batch_size: int = 32,
            overwrite_db: bool = False,
            captioner: Optional[Captioner] = None,
    ) -> pd.DataFrame:
        """
        Caption images in `img_col`, write captions to `out_col`, and index them into FTS store at `db_path`.
        Supports local paths or URLs in `img_col`.
        """
        cap = captioner or getattr(lotus.settings, "captioner", None) or Captioner()
        store = getattr(lotus.settings, "cap_store", None) or CaptionFTSStore(db_path)
        if overwrite_db:
            store.clear()

        paths = self._obj[img_col].tolist()

        # Batch load images
        images, ids = [], []
        captions: List[str] = []

        def flush_batch():
            nonlocal images, captions
            if not images:
                return
            captions.extend(cap(images))
            images = []

        for rid, it in zip(self._obj.index.astype(str).tolist(), paths):
            if looks_like_image_url(str(it)):
                img = fetch_pil(str(it))
            else:
                img = Image.open(Path(str(it))).convert("RGB")
            images.append(img); ids.append(rid)
            if len(images) == batch_size:
                flush_batch()

        flush_batch()
        assert len(captions) == len(ids), "Batching bug: captions/ids mismatch"

        # Persist captions into dataframe
        self._obj[out_col] = pd.Series(captions, index=self._obj.index)

        # Index into FTS (rid, caption)
        store.index(zip(ids, captions))

        # keep pointers for later retrieval
        self._obj.attrs["cap_index_dirs"][out_col] = db_path
        return self._obj

    # ---------- Retrieval helpers ----------

    def _get_store(self, out_col: str) -> CaptionFTSStore:
        db = self._obj.attrs["cap_index_dirs"].get(out_col)
        if not db:
            raise ValueError(f"No caption index found for column '{out_col}'. Did you call df.sem_captions_index(...)?")
        return CaptionFTSStore(db)

    def search(self, query: str, out_col: str, limit: int = 1000) -> pd.DataFrame:
        """
        Caption-only recall: returns a new DataFrame containing the top matches,
        ordered by BM25 (best first), with a 'cap_score' column (higher is better).
        """
        store = self._get_store(out_col)
        hits: List[Tuple[str, float]] = store.search(query, limit=limit)
        if not hits:
            return self._obj.iloc[0:0].copy()

        # map rids (stringified original index) back to the dataframe
        idx = [type(self._obj.index.tolist()[0])(rid) if hasattr(self._obj.index, "dtype") else rid
               for rid, _ in hits]
        out = self._obj.loc[idx].copy()
        # preserve hit order
        order = {rid: i for i, (rid, _) in enumerate(hits)}
        out["cap_score"] = [hits[order[str(i)]][1] for i in out.index.astype(str)]
        out = out.sort_values("cap_score", ascending=False)
        return out

    def retrieve_rows(self, predicate_text: str, out_col: str, limit: int = 1000,
                      must: Iterable[str] = (), must_not: Iterable[str] = ()) -> pd.DataFrame:
        """
        Convenience: caption recall + cheap lexical gates.
        Use FTS MATCH syntax in `predicate_text` or pass plain words.
        """
        df = self.search(predicate_text, out_col=out_col, limit=limit).copy()
        if df.empty:
            return df
        if must:
            df = df[df[out_col].str.lower().apply(lambda t: all(w.lower() in t for w in must))]
        if must_not:
            df = df[df[out_col].str.lower().apply(lambda t: all(w.lower() not in t for w in must_not))]
        return df
