from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Sequence, Tuple


class CS(ABC):
    """Abstract caption store."""

    def __init__(self) -> None:
        self.index_dir: str | None = None

    @abstractmethod
    def index(self, ids: Sequence[str], captions: Sequence[str], index_dir: str, **kwargs) -> None:
        """Create/update the index at index_dir with given (id, caption) pairs."""
        pass

    @abstractmethod
    def load_index(self, index_dir: str) -> None:
        pass

    @abstractmethod
    def __call__(self, query: str, K: int, ids: Optional[Sequence[str]] = None, **kwargs) -> List[Tuple[str, float]]:
        """Caption search. Returns [(rid, score)] with higher=better."""
        pass

    @abstractmethod
    def get_captions_from_index(self, index_dir: str, ids: Sequence[str]) -> Dict[str, str]:
        pass
