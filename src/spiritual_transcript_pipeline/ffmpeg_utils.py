from __future__ import annotations

import re
import shlex
import subprocess
from pathlib import Path


def _run(cmd: list[str]) -> None:
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        rendered = " ".join(shlex.quote(part) for part in cmd)
        raise RuntimeError(f"Command failed ({exc.returncode}): {rendered}") from exc


def probe_media_duration(ffprobe_bin: str, input_path: Path) -> float:
    cmd = [
        ffprobe_bin,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(input_path),
    ]
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        rendered = " ".join(shlex.quote(part) for part in cmd)
        raise RuntimeError(f"Command failed ({exc.returncode}): {rendered}") from exc
    output = result.stdout.strip()
    if not output:
        raise RuntimeError(f"Unable to determine media duration for {input_path}")
    return float(output)


def probe_decodable_video_duration(ffmpeg_bin: str, input_path: Path) -> float:
    cmd = [
        ffmpeg_bin,
        "-hide_banner",
        "-i",
        str(input_path),
        "-map",
        "0:v:0",
        "-f",
        "null",
        "-",
    ]
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        rendered = " ".join(shlex.quote(part) for part in cmd)
        raise RuntimeError(f"Command failed ({exc.returncode}): {rendered}") from exc
    combined_output = f"{result.stdout}\n{result.stderr}"
    matches = re.findall(r"time=(\d{2}:\d{2}:\d{2}\.\d{2})", combined_output)
    if not matches:
        raise RuntimeError(f"Unable to determine decodable video duration for {input_path}")
    hours, minutes, seconds = matches[-1].split(":")
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


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
        "-ss",
        f"{start:.3f}",
        "-i",
        str(input_video),
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
        "-reset_timestamps",
        "1",
        "-avoid_negative_ts",
        "make_zero",
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
        escaped = file_path.resolve().as_posix().replace("'", "'\\''")
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
        "-c",
        "copy",
        "-fflags",
        "+genpts",
        "-avoid_negative_ts",
        "make_zero",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    _run(cmd)
