from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .models import ModuleClip, ModuleSegment, TranscriptLine


CODE_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL)


def load_prompt(template_path: Path, transcript_text: str) -> str:
    template = template_path.read_text(encoding="utf-8")
    if "{paste transcript}" in template:
        return template.replace("{paste transcript}", transcript_text)
    return template + "\n\n" + transcript_text


def parse_llm_json_response(raw_text: str) -> Any:
    match = CODE_FENCE_RE.match(raw_text)
    candidate = match.group(1) if match else raw_text
    return json.loads(candidate)


def _to_module_clip(item: dict[str, Any], idx: int) -> ModuleClip:
    if not isinstance(item.get("title"), str) or not item["title"].strip():
        raise ValueError(f"Module {idx}: missing/invalid title")
    if not isinstance(item.get("category"), str) or not item["category"].strip():
        raise ValueError(f"Module {idx}: missing/invalid category")

    raw_segments = item.get("segments")
    if not isinstance(raw_segments, list) or not raw_segments:
        raise ValueError(f"Module {idx}: missing/invalid segments")

    segments: list[ModuleSegment] = []
    for seg_idx, seg in enumerate(raw_segments, start=1):
        if not isinstance(seg, dict):
            raise ValueError(f"Module {idx} segment {seg_idx}: expected object")
        try:
            start = float(seg["start"])
            end = float(seg["end"])
            text = str(seg["text"]).strip()
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"Module {idx} segment {seg_idx}: invalid fields") from exc
        if end <= start:
            raise ValueError(f"Module {idx} segment {seg_idx}: end must be > start")
        if not text:
            raise ValueError(f"Module {idx} segment {seg_idx}: empty text")
        segments.append(ModuleSegment(start=start, end=end, text=text))

    subcategory = item.get("subcategory")
    if subcategory is not None and not isinstance(subcategory, str):
        raise ValueError(f"Module {idx}: subcategory must be string or null")
    context = item.get("context")
    if context is not None and not isinstance(context, str):
        raise ValueError(f"Module {idx}: context must be string or null")
    normalized_terms = item.get("normalized_terms")
    if normalized_terms is not None and not isinstance(normalized_terms, dict):
        raise ValueError(f"Module {idx}: normalized_terms must be object or null")

    return ModuleClip(
        title=item["title"].strip(),
        category=item["category"].strip(),
        subcategory=subcategory.strip() if isinstance(subcategory, str) else None,
        context=context.strip() if isinstance(context, str) else None,
        normalized_terms=normalized_terms,
        segments=segments,
    )


def validate_modules_payload(
    payload: Any,
    transcript_lines: list[TranscriptLine] | None = None,
    *,
    enforce_duration: bool = True,
) -> list[ModuleClip]:
    if not isinstance(payload, list):
        raise ValueError("LLM output must be a JSON array.")

    modules = [_to_module_clip(item, idx + 1) for idx, item in enumerate(payload)]

    all_intervals: list[tuple[float, float, int, int]] = []
    valid_starts: set[float] = set()
    valid_ends: set[float] = set()
    if transcript_lines is not None:
        valid_starts = {round(t.start, 2) for t in transcript_lines}
        valid_ends = {round(t.end, 2) for t in transcript_lines}

    for mod_idx, module in enumerate(modules, start=1):
        prev_end: float | None = None
        total_duration = 0.0
        for seg_idx, seg in enumerate(module.segments, start=1):
            if prev_end is not None and seg.start < prev_end:
                raise ValueError(
                    f"Module {mod_idx} segments overlap or are unordered at segment {seg_idx}."
                )
            prev_end = seg.end
            total_duration += seg.end - seg.start
            all_intervals.append((seg.start, seg.end, mod_idx, seg_idx))

            if transcript_lines is not None:
                if round(seg.start, 2) not in valid_starts:
                    raise ValueError(
                        f"Module {mod_idx} segment {seg_idx}: start {seg.start:.2f} is not a transcript boundary."
                    )
                if round(seg.end, 2) not in valid_ends:
                    raise ValueError(
                        f"Module {mod_idx} segment {seg_idx}: end {seg.end:.2f} is not a transcript boundary."
                    )

        if enforce_duration and (total_duration < 180 or total_duration > 1200):
            raise ValueError(
                f"Module {mod_idx} duration {total_duration:.1f}s is outside the 3-20 minute range."
            )

    all_intervals.sort(key=lambda x: (x[0], x[1]))
    prev: tuple[float, float, int, int] | None = None
    for current in all_intervals:
        if prev is not None and current[0] < prev[1]:
            raise ValueError(
                "Modules overlap across the full output "
                f"(module {prev[2]} segment {prev[3]} overlaps module {current[2]} segment {current[3]})."
            )
        prev = current

    return modules


def modules_to_jsonable(modules: list[ModuleClip]) -> list[dict[str, Any]]:
    return [
        {
            "title": module.title,
            "category": module.category,
            "subcategory": module.subcategory,
            "context": module.context,
            "normalized_terms": module.normalized_terms,
            "segments": [
                {"start": seg.start, "end": seg.end, "text": seg.text} for seg in module.segments
            ],
        }
        for module in modules
    ]
