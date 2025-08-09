import io
import re
import requests
from urllib.parse import urlparse
from typing import List, Tuple

import numpy as np
import torch
from numpy.typing import NDArray
from PIL import Image
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from lotus.dtype_extensions import convert_to_base_data
from lotus.models.rm import RM

IMG_EXT_RE = re.compile(r"\.(png|jpe?g|webp|bmp|gif|tiff?)($|\?)", re.IGNORECASE)

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


class SentenceTransformersRM(RM):
    def __init__(
        self,
        model: str = "clip-ViT-L-14",   # multimodal-capable model
        max_batch_size: int = 64,
        normalize_embeddings: bool = True,
        device: str | None = None,
    ):
        self.model_name: str = model
        self.max_batch_size: int = max_batch_size
        self.normalize: bool = normalize_embeddings

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
        text_items: List[Tuple[int, str]] = []
        image_items: List[Tuple[int, Image.Image]] = []

        # split / fetch
        for idx, d in enumerate(docs):
            if looks_like_image_url(d):
                img = fetch_pil(d)
                image_items.append((idx, img))
            else:
                text_items.append((idx, d))

        vecs: List[np.ndarray | None] = [None] * len(docs)

        # images
        for i in range(0, len(image_items), self.max_batch_size):
            chunk = image_items[i : i + self.max_batch_size]
            if not chunk:
                continue
            ids, imgs = zip(*chunk)
            emb = self._encode_image_batch(list(imgs))
            for j, v in zip(ids, emb):
                vecs[j] = v

        # text
        for i in range(0, len(text_items), self.max_batch_size):
            chunk = text_items[i : i + self.max_batch_size]
            if not chunk:
                continue
            ids, txts = zip(*chunk)
            emb = self._encode_text_batch(list(txts))
            for j, v in zip(ids, emb):
                vecs[j] = v

        # sanity check
        if any(v is None for v in vecs):
            missing = [i for i, v in enumerate(vecs) if v is None]
            raise ValueError(f"Some documents failed to embed: indices {missing}")

        return np.vstack(vecs)  # float32 by default from ST
