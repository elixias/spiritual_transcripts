from __future__ import annotations

import json
import logging
import re
import shlex
import shutil
import tempfile
from pathlib import Path
from typing import Any

from .config import CutRenderConfig
from .ffmpeg_utils import concat_videos, cut_video_segment, _run
from .models import ModuleClip, ModuleSegment
from .segmentation import validate_modules_payload


logger = logging.getLogger(__name__)


def _slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = re.sub(r"_+", "_", value)
    return value.strip("_") or "clip"


def _load_modules(path: Path) -> list[ModuleClip]:
    payload: Any = json.loads(path.read_text(encoding="utf-8"))
    return validate_modules_payload(
        payload,
        transcript_lines=None,
        enforce_duration=False,
        enforce_global_overlap=False,
    )


def _module_output_name(index: int, module: ModuleClip) -> str:
    base = _slugify(module.title)
    return f"{index:03d}_{base}.mp4"


def cut_modules_to_clips(
    ffmpeg_bin: str,
    source_video: Path,
    modules_json: Path,
    out_dir: Path,
    render_config: CutRenderConfig,
) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    modules = _load_modules(modules_json)
    created: list[Path] = []

    for idx, module in enumerate(modules, start=1):
        module_total = sum(max(0.0, seg.end - seg.start) for seg in module.segments)
        logger.info(
            "[cut] Module %03d '%s' segments=%d total_duration=%.3fs",
            idx,
            module.title,
            len(module.segments),
            module_total,
        )
        output_path = out_dir / _module_output_name(idx, module)
        with tempfile.TemporaryDirectory(prefix=f"module_{idx:03d}_", dir=out_dir) as tmpdir:
            tmpdir_path = Path(tmpdir)
            part_files: list[Path] = []
            for seg_idx, seg in enumerate(module.segments, start=1):
                logger.info(
                    "[cut]  segment %03d start=%.3f end=%.3f duration=%.3f",
                    seg_idx,
                    seg.start,
                    seg.end,
                    max(0.0, seg.end - seg.start),
                )
                part_path = tmpdir_path / f"part_{seg_idx:03d}.mp4"
                _cut_exact_segment(ffmpeg_bin, source_video, seg, part_path)
                part_files.append(part_path)

            if len(part_files) == 1:
                intermediate_clip = part_files[0]
            else:
                concat_list = tmpdir_path / "concat.txt"
                intermediate_clip = tmpdir_path / "combined.mp4"
                concat_videos(ffmpeg_bin, part_files, concat_list, intermediate_clip)

            subtitles_path = (
                tmpdir_path / "subtitles.srt"
                if _write_subtitles(module, tmpdir_path / "subtitles.srt")
                else None
            ) if render_config.subtitles_enabled else None

            _render_final_clip(
                ffmpeg_bin,
                intermediate_clip,
                output_path,
                render_config,
                subtitles_path,
                module_total,
            )
        created.append(output_path)

    return created


def _cut_exact_segment(ffmpeg_bin: str, source_video: Path, seg: ModuleSegment, out_path: Path) -> None:
    cut_video_segment(
        ffmpeg_bin=ffmpeg_bin,
        input_video=source_video,
        start=seg.start,
        end=seg.end,
        output_path=out_path,
    )


def _render_final_clip(
    ffmpeg_bin: str,
    source_clip: Path,
    output_path: Path,
    render_config: CutRenderConfig,
    subtitles_path: Path | None,
    expected_duration: float,
) -> None:
    logo_input = (
        render_config.logo_path
        if render_config.logo_path and render_config.logo_path.exists()
        else None
    )
    needs_subtitles = subtitles_path is not None and subtitles_path.exists()
    if not logo_input and not needs_subtitles:
        shutil.copy2(source_clip, output_path)
        return

    filters: list[str] = []
    current_label = "[0:v]"

    if needs_subtitles and subtitles_path:
        filters.append(
            f"[0:v]{_build_subtitle_filter(render_config, subtitles_path)}[subbed]"
        )
        current_label = "[subbed]"

    if logo_input:
        filters.append(
            f"{current_label}[1:v]overlay=main_w-overlay_w-{render_config.logo_margin}:"
            f"{render_config.logo_margin}:shortest=1:eof_action=pass[logoed]"
        )
        current_label = "[logoed]"

    filter_complex = ";".join(filters)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(
        "[cut] Rendering final clip %s expected_duration=%.3fs subtitles=%s logo=%s",
        output_path.name,
        expected_duration,
        bool(needs_subtitles),
        bool(logo_input),
    )
    cmd = [
        ffmpeg_bin,
        "-y",
        "-i",
        str(source_clip),
    ]
    if logo_input:
        cmd.extend(["-loop", "1", "-i", str(logo_input)])
    cmd.extend(
        [
            "-filter_complex",
            filter_complex,
            "-map",
            current_label,
            "-map",
            "0:a?",
            "-t",
            f"{expected_duration:.3f}",
            "-c:v",
            render_config.video_encoder,
        ]
    )
    cmd.extend(_split_encoder_args(render_config.video_encoder_args))
    cmd.extend(
        [
            "-c:a",
            "aac",
            "-b:a",
            "160k",
            "-movflags",
            "+faststart",
            str(output_path),
        ]
    )
    _run(cmd)


def _build_subtitle_filter(config: CutRenderConfig, subtitles_path: Path) -> str:
    parts: list[str] = []
    quoted_subtitles = _quote_for_filter(str(subtitles_path.as_posix()))
    parts.append(f"subtitles=filename={quoted_subtitles}")
    style = _subtitle_force_style(config)
    if style:
        parts.append(f"force_style='{style}'")
    if config.subtitle_font_path and config.subtitle_font_path.exists():
        fonts_dir = _quote_for_filter(str(config.subtitle_font_path.as_posix()))
        parts.append(f"fontsdir={fonts_dir}")
    return ":".join(parts)


def _subtitle_force_style(config: CutRenderConfig) -> str:
    style_parts = [
        f"FontSize={config.subtitle_font_size}",
        f"PrimaryColour={_ass_color(config.subtitle_color)}",
        f"OutlineColour={_ass_color(config.subtitle_outline_color)}",
        f"Outline={config.subtitle_outline_width}",
        "BorderStyle=1",
        "Alignment=2",
        f"MarginV={config.subtitle_margin}",
    ]
    return ",".join(style_parts)


def _ass_color(color: str) -> str:
    normalized = color.strip()
    if normalized.startswith("#"):
        normalized = normalized[1:]
    if len(normalized) == 3:
        normalized = "".join(ch * 2 for ch in normalized)
    if len(normalized) != 6:
        normalized = "FFFFFF"
    r, g, b = normalized[0:2], normalized[2:4], normalized[4:6]
    return f"&H00{b.upper()}{g.upper()}{r.upper()}"


def _quote_for_filter(value: str) -> str:
    escaped = value.replace("'", r"'\''")
    escaped = escaped.replace(":", r"\:")
    return f"'{escaped}'"


def _split_encoder_args(args: str | None) -> list[str]:
    if not args:
        return []
    stripped = args.strip()
    if not stripped:
        return []
    return shlex.split(stripped)


def _write_subtitles(module: ModuleClip, out_path: Path) -> bool:
    lines: list[str] = []
    cursor = 0.0
    index = 1
    for segment in module.segments:
        duration = max(0.0, segment.end - segment.start)
        if duration <= 0:
            continue
        raw_text = segment.text.strip()
        safe_lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
        if safe_lines:
            text_block = "\n".join(safe_lines)
            lines.append(f"{index}")
            lines.append(f"{_srt_timestamp(cursor)} --> {_srt_timestamp(cursor + duration)}")
            lines.append(text_block)
            lines.append("")
            index += 1
        cursor += duration
    if not lines:
        return False
    out_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    return True


def _srt_timestamp(value: float) -> str:
    total_ms = int(round(value * 1000))
    hours = total_ms // 3_600_000
    remainder = total_ms % 3_600_000
    minutes = remainder // 60_000
    remainder %= 60_000
    seconds = remainder // 1000
    milliseconds = remainder % 1000
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"
