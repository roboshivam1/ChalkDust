"""Content-addressed artifact store (D-004).

Every artifact lives at a path derived from the hash of everything that
determines it. Re-running a stage with unchanged inputs is a file-exists check.

Writes go to a temp path and are atomically renamed on commit, so a crash
mid-render leaves no half-written artifact that a later run would trust.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from chalkdust.core.models import BeatSpec, BuildContext, VoiceConfig

HASH_LEN = 16  # 64 bits of hex; collision risk is irrelevant at our volumes


def _stable_json(obj: Any) -> str:
    """Canonical JSON: sorted keys, no incidental whitespace. Two logically
    equal objects must always produce the same string, or caching is a lie."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def content_hash(*parts: Any) -> str:
    """Hash any collection of JSON-serialisable values into a short hex key."""
    h = hashlib.sha256()
    for part in parts:
        h.update(_stable_json(part).encode())
        h.update(b"\x00")  # separator, so ("ab","c") != ("a","bc")
    return h.hexdigest()[:HASH_LEN]


# --- key builders -----------------------------------------------------------
# Each key includes exactly what determines the artifact, and nothing else.
# Over-including causes needless re-work; under-including serves stale files.


def tts_key(narration: str, voice: VoiceConfig) -> str:
    """Audio depends only on the words and the voice."""
    return content_hash("tts", narration.strip(), voice.model_dump(mode="json"))


def beat_render_key(spec: BeatSpec, duration: float, ctx: BuildContext) -> str:
    """A rendered beat depends on the visual spec, how long it must run, and
    the build context.

    Deliberately excluded:
      - narration text: only affects the render via `duration`, already here
      - transition: applied at ffmpeg assembly, not baked into the clip
    """
    return content_hash(
        "beat",
        spec.component,
        spec.params,
        sorted(spec.carry_in),
        round(duration, 3),  # avoid float noise producing spurious misses
        ctx.model_dump(mode="json"),
    )


# --- store ------------------------------------------------------------------


@dataclass
class CacheSlot:
    """A reserved location in the cache.

    Usage:
        slot = cache.slot("tts", key, ".wav")
        if not slot.exists:
            generate_audio_to(slot.tmp)
            slot.commit()
        use(slot.path)
    """

    path: Path
    tmp: Path
    exists: bool

    def commit(self) -> Path:
        """Atomically move the temp file into place."""
        if not self.tmp.exists():
            raise FileNotFoundError(f"nothing written to {self.tmp}")
        os.replace(self.tmp, self.path)
        self.exists = True
        return self.path


class Cache:
    def __init__(self, root: Path | str = ".cache") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _dir(self, namespace: str) -> Path:
        d = self.root / namespace
        d.mkdir(parents=True, exist_ok=True)
        return d

    def slot(self, namespace: str, key: str, ext: str) -> CacheSlot:
        """Reserve a cache location. Does not create anything on disk."""
        final = self._dir(namespace) / f"{key}{ext}"
        return CacheSlot(path=final, tmp=final.with_suffix(final.suffix + ".tmp"),
                         exists=final.exists())

    def get(self, namespace: str, key: str, ext: str) -> Path | None:
        p = self._dir(namespace) / f"{key}{ext}"
        return p if p.exists() else None

    # JSON convenience, for research results and beat metadata.

    def get_json(self, namespace: str, key: str) -> Any | None:
        p = self.get(namespace, key, ".json")
        return json.loads(p.read_text()) if p else None

    def put_json(self, namespace: str, key: str, value: Any) -> Path:
        slot = self.slot(namespace, key, ".json")
        slot.tmp.write_text(_stable_json(value))
        return slot.commit()