import gc
import io
import re
import requests
from urllib.parse import urlparse
from typing import List, Tuple
import os
import numpy as np
import torch
from numpy.typing import NDArray
from PIL import Image
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from lotus.dtype_extensions import convert_to_base_data
from lotus.models.rm import RM

IMG_EXT_RE = re.compile(r"\.(png|jpe?g|webp|bmp|gif|tiff?)($|\?)", re.IGNORECASE)

def looks_like_image_path(s: str) -> bool:
    return os.path.isfile(s) and IMG_EXT_RE.search(s) is not None

def looks_like_image_url(s: str) -> bool:
    try:
        u = urlparse(s)
        if u.scheme in {"http", "https"} and u.netloc:
            return True
    except Exception:
        pass
    return False

def fetch_pil(url: str) -> Image.Image:
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    return Image.open(io.BytesIO(r.content)).convert("RGB")

def fetch_pil_path(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")

class SentenceTransformersRM(RM):
    def __init__(
        self,
        model: str = "clip-ViT-L-14",   # multimodal-capable model
        max_batch_size: int = 64,
        normalize_embeddings: bool = True,
    ):
        self.model_name: str = model
        self.max_batch_size: int = max_batch_size
        self.normalize: bool = normalize_embeddings
        device: str = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

        self.model: SentenceTransformer = SentenceTransformer(model, device=device)

        # For CLIP text encoder (77 tokens). This makes ST truncate automatically.
        if not hasattr(self.model, "max_seq_length") or self.model.max_seq_length is None:
            self.model.max_seq_length = 77
        else:
            self.model.max_seq_length = min(self.model.max_seq_length, 77)

    # ---- encoding helpers ----

    def _encode_text_batch(self, batch: List[str]) -> NDArray[np.float32]:
        clean = convert_to_base_data(batch)  # keep parity with old code
        with torch.no_grad():
            e = self.model.encode(
                clean,
                convert_to_tensor=True,
                normalize_embeddings=self.normalize,
                show_progress_bar=False,
            )
        return e.cpu().numpy()

    def _encode_image_batch(self, batch_imgs: List[Image.Image]) -> NDArray[np.float32]:
        with torch.no_grad():
            e = self.model.encode(
                batch_imgs,
                convert_to_tensor=True,
                normalize_embeddings=self.normalize,
                show_progress_bar=False,
            )
        return e.cpu().numpy()

    # ---- public API ----

    def _embed(self, docs: List[str]) -> NDArray[np.float32]:
        image_indices: List[Tuple[int, str]] = []
        text_indices: List[int] = []

        for idx, d in enumerate(docs):
            if looks_like_image_url(d):
                image_indices.append((idx, "url"))
            elif looks_like_image_path(d):
                image_indices.append((idx, "path"))
            else:
                text_indices.append(idx)

        vecs: List[np.ndarray | None] = [None] * len(docs)

        for i in range(0, len(image_indices), self.max_batch_size):
            chunk = image_indices[i : i + self.max_batch_size]
            ids: List[int] = []
            imgs: List[Image.Image] = []
            for j, kind in chunk:
                src = docs[j]
                try:
                    img = fetch_pil(src) if kind == "url" else fetch_pil_path(src)
                    imgs.append(img)
                    ids.append(j)
                except Exception:
                    continue
            if imgs:
                emb = self._encode_image_batch(imgs)
                for j, v in zip(ids, emb):
                    vecs[j] = v
                for im in imgs:
                    try:
                        im.close()
                    except Exception:
                        pass
                del imgs
                gc.collect()

        for i in range(0, len(text_indices), self.max_batch_size):
            ids = text_indices[i : i + self.max_batch_size]
            txts = [docs[j] for j in ids]
            emb = self._encode_text_batch(txts)
            for j, v in zip(ids, emb):
                vecs[j] = v

        missing = [i for i, v in enumerate(vecs) if v is None]
        if missing:
            raise ValueError(f"Some documents failed to embed: indices {missing}")

        out = np.vstack(vecs)
        if out.dtype != np.float32:
            out = out.astype(np.float32, copy=False)
        return out






