from __future__ import annotations

from pathlib import Path
from typing import Any

from openai import OpenAI

from .config import StageLLMConfig
from .models import TranscriptLine


def _build_client(cfg: StageLLMConfig) -> OpenAI:
    if not cfg.api_key:
        raise ValueError("API key is missing for this stage. Set stage-specific or shared OPENAI_API_KEY.")
    return OpenAI(api_key=cfg.api_key)


def transcribe_audio_with_timestamps(audio_path: Path, cfg: StageLLMConfig) -> tuple[list[TranscriptLine], dict[str, Any]]:
    if not cfg.model:
        raise ValueError("TRANSCRIBE_MODEL is not set.")
    client = _build_client(cfg)
    with audio_path.open("rb") as audio_file:
        resp = client.audio.transcriptions.create(
            model=cfg.model,
            file=audio_file,
            response_format="verbose_json",
            timestamp_granularities=["segment"],
        )

    if hasattr(resp, "model_dump"):
        payload = resp.model_dump()
    elif isinstance(resp, dict):
        payload = resp
    else:
        payload = dict(resp)  # type: ignore[arg-type]

    segments = payload.get("segments") or []
    transcript_lines: list[TranscriptLine] = []
    for segment in segments:
        text = str(segment.get("text", "")).strip()
        if not text:
            continue
        transcript_lines.append(
            TranscriptLine(
                start=float(segment["start"]),
                end=float(segment["end"]),
                text=text,
            )
        )
    if not transcript_lines:
        raise RuntimeError("Transcription returned no timestamped segments.")
    return transcript_lines, payload
