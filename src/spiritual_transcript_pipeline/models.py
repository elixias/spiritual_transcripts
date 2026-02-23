from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class TranscriptLine:
    start: float
    end: float
    text: str


@dataclass
class ModuleSegment:
    start: float
    end: float
    text: str


@dataclass
class ModuleClip:
    title: str
    category: str
    subcategory: str | None
    context: str | None
    normalized_terms: dict[str, Any] | None
    segments: list[ModuleSegment]
