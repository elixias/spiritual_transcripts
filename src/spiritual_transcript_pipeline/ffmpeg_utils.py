from __future__ import annotations

import shlex
import subprocess
from pathlib import Path


def _run(cmd: list[str]) -> None:
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        rendered = " ".join(shlex.quote(part) for part in cmd)
        raise RuntimeError(f"Command failed ({exc.returncode}): {rendered}") from exc


def extract_audio_to_wav(
    ffmpeg_bin: str, input_video: Path, output_audio: Path, sample_rate: int = 16000
) -> None:
    output_audio.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg_bin,
        "-y",
        "-i",
        str(input_video),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-c:a",
        "pcm_s16le",
        str(output_audio),
    ]
    _run(cmd)


def cut_video_segment(
    ffmpeg_bin: str,
    input_video: Path,
    start: float,
    end: float,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    duration = max(0.0, end - start)
    if duration <= 0:
        raise ValueError(f"Invalid segment duration: start={start}, end={end}")
    cmd = [
        ffmpeg_bin,
        "-y",
        "-i",
        str(input_video),
        "-ss",
        f"{start:.3f}",
        "-t",
        f"{duration:.3f}",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "20",
        "-c:a",
        "aac",
        "-b:a",
        "160k",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    _run(cmd)


def concat_videos(ffmpeg_bin: str, files: list[Path], concat_list_file: Path, output_path: Path) -> None:
    concat_list_file.parent.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for file_path in files:
        escaped = str(file_path.resolve()).replace("'", "'\\''")
        lines.append(f"file '{escaped}'")
    concat_list_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    cmd = [
        ffmpeg_bin,
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat_list_file),
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "20",
        "-c:a",
        "aac",
        "-b:a",
        "160k",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    _run(cmd)
