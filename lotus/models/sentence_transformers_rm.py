import numpy as np
import torch
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from lotus.dtype_extensions import convert_to_base_data
from lotus.models.rm import RM

import io
import re
import requests
from urllib.parse import urlparse
from typing import Iterable
from PIL import Image
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
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
        self.model_name = model
        self.max_batch_size = max_batch_size
        self.normalize = normalize_embeddings
        self.model = SentenceTransformer(model, device=device)

    def _encode_text_batch(self, batch: list[str]) -> np.ndarray:
        with torch.no_grad():
            e = self.model.encode(
                batch,
                convert_to_tensor=True,
                normalize_embeddings=self.normalize,
                show_progress_bar=False,
            )
        return e.cpu().numpy()

    def _encode_image_batch(self, batch_imgs: list[Image.Image]) -> np.ndarray:
        with torch.no_grad():
            e = self.model.encode(
                batch_imgs,
                convert_to_tensor=True,
                normalize_embeddings=self.normalize,
                show_progress_bar=False,
            )
        return e.cpu().numpy()

    def _embed(self, docs: list[str]) -> np.ndarray:
        text_items: list[tuple[int, str]] = []
        image_items: list[tuple[int, Image.Image]] = []

        # split
        for idx, d in enumerate(docs):
            if looks_like_image_url(d):
                img = fetch_pil(d)
                image_items.append((idx, img))
            else:
                text_items.append((idx, d))

        # encode in batches
        vecs = [None] * len(docs)

        # images
        for i in range(0, len(image_items), self.max_batch_size):
            chunk = image_items[i:i + self.max_batch_size]
            ids, imgs = zip(*chunk)
            emb = self._encode_image_batch(list(imgs))
            for j, v in zip(ids, emb):
                vecs[j] = v

        # text
        for i in range(0, len(text_items), self.max_batch_size):
            chunk = text_items[i:i + self.max_batch_size]
            ids, txts = zip(*chunk)
            emb = self._encode_text_batch(list(txts))
            for j, v in zip(ids, emb):
                vecs[j] = v

        return np.vstack(vecs)


