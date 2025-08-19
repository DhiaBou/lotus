from __future__ import annotations

import io
import re
from abc import ABC, abstractmethod
from typing import List, Sequence

import requests
from PIL import Image
from tqdm import tqdm


class CM(ABC):
    """
    Abstract captioner. Call with a list of image paths/URLs.
    Handles I/O + batching here; subclass only implements _caption_images(...)
    """

    @abstractmethod
    def _caption_images(self, images: Sequence[Image.Image]) -> List[str]:
        """Return captions for the given PIL images (same order)."""
        pass

    def __call__(self, paths: Sequence[str], batch_size: int = 16) -> List[str]:
        if not paths:
            return []
        out: List[str] = []
        batch: List[Image.Image] = []
        _IMG_EXT_RE = re.compile(r"\.(png|jpe?g|webp|bmp|gif|tiff?)($|\?)", re.IGNORECASE)

        def _is_url(s: str) -> bool:
            return s.startswith("http://") or s.startswith("https://")

        def _load_image(src: str) -> Image.Image:
            if _is_url(src):
                r = requests.get(src, timeout=15);
                r.raise_for_status()
                return Image.open(io.BytesIO(r.content)).convert("RGB")
            if not _IMG_EXT_RE.search(src):
                raise ValueError(f"Not an image path/URL: {src}")
            return Image.open(src).convert("RGB")

        for p in tqdm(paths, desc="Loading images"):
            batch.append(_load_image(p))
            if len(batch) >= batch_size:
                out.extend(self._caption_images(batch))
                batch = []
        if batch:
            out.extend(self._caption_images(batch))
        return out
