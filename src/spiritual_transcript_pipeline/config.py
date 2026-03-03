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


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return int(raw)


def _bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return default


@dataclass
class StageLLMConfig:
    api_key: str | None
    provider: str | None
    model: str | None


@dataclass
class CutRenderConfig:
    logo_path: Path | None
    logo_margin: int
    subtitles_enabled: bool
    subtitle_font_path: Path | None
    subtitle_font_size: int
    subtitle_color: str
    subtitle_outline_color: str
    subtitle_outline_width: int
    subtitle_margin: int
    video_encoder: str
    video_encoder_args: str


@dataclass
class PipelineConfig:
    transcribe: StageLLMConfig
    segment: StageLLMConfig
    ffmpeg_bin: str
    ffprobe_bin: str
    segment_temperature: float
    segment_prompt_file: Path
    output_root: Path
    cut: CutRenderConfig

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

        cut_logo_value = _first_nonempty(os.getenv("CUT_LOGO_PATH"))
        cut_font_value = _first_nonempty(os.getenv("CUT_SUBTITLE_FONT_PATH"))

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
            cut=CutRenderConfig(
                logo_path=Path(cut_logo_value) if cut_logo_value else None,
                logo_margin=_int_env("CUT_LOGO_MARGIN", 16),
                subtitles_enabled=_bool_env("CUT_ENABLE_SUBTITLES", True),
                subtitle_font_path=Path(cut_font_value) if cut_font_value else None,
                subtitle_font_size=_int_env("CUT_SUBTITLE_FONT_SIZE", 26),
                subtitle_color=_first_nonempty(os.getenv("CUT_SUBTITLE_COLOR"), "#FFFFFF")
                or "#FFFFFF",
                subtitle_outline_color=_first_nonempty(os.getenv("CUT_SUBTITLE_OUTLINE_COLOR"), "#000000")
                or "#000000",
                subtitle_outline_width=_int_env("CUT_SUBTITLE_OUTLINE_WIDTH", 3),
                subtitle_margin=_int_env("CUT_SUBTITLE_MARGIN", 40),
                video_encoder=_first_nonempty(os.getenv("CUT_VIDEO_ENCODER"), "libx264")
                or "libx264",
                video_encoder_args=_first_nonempty(
                    os.getenv("CUT_VIDEO_ENCODER_ARGS"), "-preset medium -crf 20"
                )
                or "-preset medium -crf 20",
            ),
        )
