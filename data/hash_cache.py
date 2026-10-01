"""Scalable, resumable content hash caching.

Avoids re-hashing hundreds of thousands of images on repeated audits or scans.
Tracks file mtime and size to detect modified files.
Saves cache periodically and atomically to prevent corruption upon interruption.
"""
import hashlib
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional


class HashCache:
    """Manages persistent mtime/size-validated file sha256 hashes."""

    def __init__(self, cache_path: Optional[str] = None):
        self.cache_path = Path(cache_path).resolve() if cache_path else None
        self.entries: Dict[str, Dict[str, Any]] = {}
        self.dirty_count = 0
        self.save_interval = 2000
        self._load()

    def _load(self) -> None:
        if self.cache_path and self.cache_path.is_file():
            try:
                data = json.loads(self.cache_path.read_text(encoding='utf-8'))
                if isinstance(data, dict):
                    self.entries = data
            except Exception:
                # If cache is corrupted, start empty
                self.entries = {}

    def save(self) -> None:
        if not self.cache_path:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self.cache_path.with_suffix('.tmp')
        temp_path.write_text(json.dumps(self.entries, indent=2), encoding='utf-8')
        temp_path.replace(self.cache_path)
        self.dirty_count = 0

    def get_or_compute(self, file_path: str) -> str:
        """Return cached sha256 if file mtime and size match; otherwise compute and store."""
        p = Path(file_path).resolve()
        key = str(p)
        stat = p.stat()
        mtime = stat.st_mtime
        size = stat.st_size

        entry = self.entries.get(key)
        if entry and entry.get('size') == size and abs(entry.get('mtime', 0) - mtime) < 1e-4:
            return entry['sha256']

        # Compute hash
        digest = hashlib.sha256()
        with open(p, 'rb') as f:
            for block in iter(lambda: f.read(1024 * 1024), b''):
                digest.update(block)
        val = digest.hexdigest()

        self.entries[key] = {'sha256': val, 'size': size, 'mtime': mtime}
        self.dirty_count += 1
        if self.dirty_count >= self.save_interval:
            self.save()
        return val

    def close(self) -> None:
        if self.dirty_count > 0:
            self.save()
