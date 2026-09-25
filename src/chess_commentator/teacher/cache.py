"""Content-hash disk cache for expensive teacher LLM generations.

Implements path-independent SHA-256 caching to eliminate duplicate API costs.
"""

import hashlib
import json
import os
from typing import Optional, Dict, Any


class TeacherCache:
    """Disk cache for LLM generated commentary keyed by SHA-256 content hash."""

    def __init__(self, cache_dir: str = ".cache/teacher_commentary") -> None:
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    @staticmethod
    def compute_hash(
        fen: str,
        move: str,
        cp_loss: Optional[int],
        taxonomy: str,
        model: str,
        nonce: Optional[Any] = None,
    ) -> str:
        """Create a deterministic SHA-256 hash from position analysis details and optional nonce."""
        raw_key = f"{fen}|{move}|{cp_loss}|{taxonomy}|{model}"
        if nonce is not None:
            raw_key += f"|{nonce}"
        return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()

    def _get_path(self, content_hash: str) -> str:
        prefix = content_hash[:2]
        shard_dir = os.path.join(self.cache_dir, prefix)
        os.makedirs(shard_dir, exist_ok=True)
        return os.path.join(shard_dir, f"{content_hash}.json")

    def get(self, content_hash: str) -> Optional[str]:
        """Retrieve commentary by content hash if present on disk."""
        path = self._get_path(content_hash)
        if not os.path.isfile(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("commentary")
        except (json.JSONDecodeError, OSError):
            return None

    def set(
        self,
        content_hash: str,
        commentary: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Store commentary and metadata to disk cache."""
        path = self._get_path(content_hash)
        payload = {
            "hash": content_hash,
            "commentary": commentary,
            "metadata": metadata or {},
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def size(self) -> int:
        """Count cached items."""
        count = 0
        if not os.path.exists(self.cache_dir):
            return 0
        for root, _, files in os.walk(self.cache_dir):
            count += sum(1 for f in files if f.endswith(".json"))
        return count
