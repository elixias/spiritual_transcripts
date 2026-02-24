from __future__ import annotations

import json
import re
from pathlib import Path

from .models import TranscriptLine

TRANSCRIPT_LINE_RE = re.compile(
    r"^\[(?P<start>\d+(?:\.\d+)?)\s*-\s*(?P<end>\d+(?:\.\d+)?)\]\s*(?P<text>.*)$"
)


def format_transcript_lines(lines: list[TranscriptLine]) -> str:
    return "\n".join(
        f"[{line.start:.2f}-{line.end:.2f}] {line.text.strip()}".rstrip() for line in lines
    )


def save_transcript_txt(lines: list[TranscriptLine], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(format_transcript_lines(lines) + "\n", encoding="utf-8")


def save_json(data: object, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def parse_transcript_txt(path: Path) -> list[TranscriptLine]:
    return parse_transcript_text(path.read_text(encoding="utf-8"), source=str(path))


def parse_transcript_text(text: str, *, source: str = "<transcript>") -> list[TranscriptLine]:
    lines: list[TranscriptLine] = []
    for idx, raw in enumerate(text.splitlines(), start=1):
        stripped = raw.strip()
        if not stripped:
            continue
        match = TRANSCRIPT_LINE_RE.match(stripped)
        if not match:
            raise ValueError(f"Invalid transcript line format at {source}:{idx}: {raw}")
        start = float(match.group("start"))
        end = float(match.group("end"))
        text = match.group("text").strip()
        lines.append(TranscriptLine(start=start, end=end, text=text))
    return lines
