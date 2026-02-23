from __future__ import annotations

import json
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any

from .ffmpeg_utils import concat_videos, cut_video_segment
from .models import ModuleClip, ModuleSegment
from .segmentation import validate_modules_payload


def _slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = re.sub(r"_+", "_", value)
    return value.strip("_") or "clip"


def _load_modules(path: Path) -> list[ModuleClip]:
    payload: Any = json.loads(path.read_text(encoding="utf-8"))
    return validate_modules_payload(payload, transcript_lines=None, enforce_duration=False)


def _module_output_name(index: int, module: ModuleClip) -> str:
    base = _slugify(module.title)
    return f"{index:03d}_{base}.mp4"


def cut_modules_to_clips(
    ffmpeg_bin: str,
    source_video: Path,
    modules_json: Path,
    out_dir: Path,
) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    modules = _load_modules(modules_json)
    created: list[Path] = []

    for idx, module in enumerate(modules, start=1):
        output_path = out_dir / _module_output_name(idx, module)
        with tempfile.TemporaryDirectory(prefix=f"module_{idx:03d}_", dir=out_dir) as tmpdir:
            tmpdir_path = Path(tmpdir)
            part_files: list[Path] = []
            for seg_idx, seg in enumerate(module.segments, start=1):
                part_path = tmpdir_path / f"part_{seg_idx:03d}.mp4"
                _cut_exact_segment(ffmpeg_bin, source_video, seg, part_path)
                part_files.append(part_path)

            if len(part_files) == 1:
                shutil.copy2(part_files[0], output_path)
            else:
                concat_list = tmpdir_path / "concat.txt"
                concat_videos(ffmpeg_bin, part_files, concat_list, output_path)
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
