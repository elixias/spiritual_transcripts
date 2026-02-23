from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


def _first_nonempty(*values: str | None) -> str | None:
    for value in values:
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def _float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return float(raw)


@dataclass
class StageLLMConfig:
    api_key: str | None
    provider: str | None
    model: str | None


@dataclass
class PipelineConfig:
    transcribe: StageLLMConfig
    segment: StageLLMConfig
    ffmpeg_bin: str
    ffprobe_bin: str
    segment_temperature: float
    segment_prompt_file: Path
    output_root: Path

    @classmethod
    def from_env(cls) -> "PipelineConfig":
        load_dotenv()
        shared_key = _first_nonempty(os.getenv("OPENAI_API_KEY"))

        transcribe = StageLLMConfig(
            api_key=_first_nonempty(os.getenv("TRANSCRIBE_API_KEY"), shared_key),
            provider=_first_nonempty(os.getenv("TRANSCRIBE_PROVIDER"), "openai"),
            model=_first_nonempty(os.getenv("TRANSCRIBE_MODEL"), "whisper-1"),
        )
        segment = StageLLMConfig(
            api_key=_first_nonempty(os.getenv("SEGMENT_API_KEY")),
            provider=_first_nonempty(os.getenv("SEGMENT_PROVIDER")),
            model=_first_nonempty(os.getenv("SEGMENT_MODEL")),
        )

        return cls(
            transcribe=transcribe,
            segment=segment,
            ffmpeg_bin=_first_nonempty(os.getenv("FFMPEG_BIN"), "ffmpeg") or "ffmpeg",
            ffprobe_bin=_first_nonempty(os.getenv("FFPROBE_BIN"), "ffprobe") or "ffprobe",
            segment_temperature=_float_env("SEGMENT_TEMPERATURE", 0.0),
            segment_prompt_file=Path(
                _first_nonempty(
                    os.getenv("SEGMENT_PROMPT_FILE"), "prompts/segment_modules_prompt.txt"
                )
                or "prompts/segment_modules_prompt.txt"
            ),
            output_root=Path(_first_nonempty(os.getenv("OUTPUT_ROOT"), "outputs") or "outputs"),
        )
