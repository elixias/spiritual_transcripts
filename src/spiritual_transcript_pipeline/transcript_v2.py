#!/usr/bin/env python3
"""
Lightweight CLI for transcribing a single video/audio asset.

Defaults to OpenAI's hosted Whisper API, but offers a `--backend` switch
to fall back to a local Hugging Face model if you prefer.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from dotenv import load_dotenv

DEFAULT_MODELS: Mapping[str, str] = {
    "openai": "whisper-1",
    "huggingface": "openai/whisper-small",
}

MAX_OPENAI_CHUNK_BYTES = 20 * 1024 * 1024  # 20 MiB
DEFAULT_SEGMENT_SECONDS = 300


@dataclass(frozen=True)
class TranscriptSegment:
    start: float
    end: float
    text: str

    def offset(self, delta: float) -> "TranscriptSegment":
        return TranscriptSegment(self.start + delta, self.end + delta, self.text)


@dataclass(frozen=True)
class CostRecord:
    model: str
    duration: float | None
    tokens: Mapping[str, int]
    cost: float


@dataclass(frozen=True)
class TranscriptionResult:
    backend: str
    model: str
    segments: list[TranscriptSegment]
    raw_payload: dict[str, Any]
    cost_records: list[CostRecord]


class CostTracker:
    def __init__(self) -> None:
        self.records: list[CostRecord] = []

    def add_openai_call(
        self,
        model: str,
        duration: float | None,
        usage: Mapping[str, int] | None,
    ) -> None:
        processed_usage = {}
        if usage:
            for key, value in usage.items():
                if isinstance(value, (int, float)):
                    processed_usage[key] = int(value)
        cost = self._estimate_cost(model, duration, processed_usage)
        self.records.append(CostRecord(model, duration, processed_usage, cost))

    def _estimate_cost(
        self,
        model: str,
        duration: float | None,
        usage: Mapping[str, int] | None,
    ) -> float:
        if duration and model in AUDIO_COST_PER_MINUTE:
            return (duration / 60.0) * AUDIO_COST_PER_MINUTE[model]
        if usage:
            rates = TEXT_TOKEN_RATES.get(model, {})
            cost = 0.0
            if rates:
                cost += (
                    usage.get("input_tokens", 0)
                    * rates.get("input", 0.0)
                    / 1_000_000
                )
                cost += (
                    usage.get("output_tokens", 0)
                    * rates.get("output", 0.0)
                    / 1_000_000
                )
            return cost
        return 0.0

    def summary_text(self) -> str:
        if not self.records:
            return ""
        lines = ["Cost summary:"]
        total = 0.0
        for rec in self.records:
            detail = f" - {rec.model}: ${rec.cost:.4f}"
            if rec.duration:
                detail += f" ({rec.duration:.1f}s audio)"
            if rec.tokens:
                detail += f" tokens=in:{rec.tokens.get('input_tokens',0)}, out:{rec.tokens.get('output_tokens',0)}"
            lines.append(detail)
            total += rec.cost
        lines.append(f"Total cost: ${total:.4f}")
        return "\n".join(lines)


AUDIO_COST_PER_MINUTE: Mapping[str, float] = {
    "whisper-1": 0.006,
    "gpt-4o-transcribe": 0.006,
    "gpt-4o-transcribe-diarize": 0.006,
    "gpt-4o-mini-transcribe": 0.003,
}

TEXT_TOKEN_RATES: Mapping[str, Mapping[str, float]] = {
    "gpt-4o-transcribe": {"input": 2.50, "output": 10.00},
    "gpt-4o-transcribe-diarize": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini-transcribe": {"input": 1.25, "output": 5.00},
}


def ensure_env_file(env_path: Path) -> None:
    """Make sure the requested env file exists and is loaded."""
    if not env_path.exists():
        env_path.write_text(
            "# Add your OpenAI credentials below\nOPENAI_API_KEY=\n",
            encoding="utf-8",
        )
        print(
            f"Created placeholder environment file at {env_path}. "
            "Open it, set OPENAI_API_KEY, and re-run if you want the OpenAI backend."
        )
    load_dotenv(env_path)


def transcribe_with_openai(
    media_path: Path,
    model_id: str,
    tracker: CostTracker,
    duration_override: float | None = None,
) -> tuple[list[TranscriptSegment], dict[str, Any]]:
    """Call the OpenAI Whisper endpoint (v1 API)."""
    from openai import OpenAI, OpenAIError

    api_key = os.getenv("OPENAI_API_KEY") or ""
    if not api_key.strip():
        raise RuntimeError(
            "OPENAI_API_KEY is missing. Set it in the env file or as an environment variable."
        )
    client = OpenAI(api_key=api_key)

    response_format = (
        "verbose_json" if not _is_gpt4o_transcribe_model(model_id) else "json"
    )
    with media_path.open("rb") as audio_file:
        try:
            result = client.audio.transcriptions.create(
                model=model_id,
                file=audio_file,
                response_format=response_format,
            )
        except OpenAIError as exc:
            msg = getattr(exc, "user_message", str(exc))
            if "response_format 'verbose_json' is not compatible" in msg:
                print(
                    "Verbose JSON format unsupported for this model; "
                    "falling back to default transcription output."
                )
                result = client.audio.transcriptions.create(
                    model=model_id, file=audio_file
                )
            else:
                raise

    if hasattr(result, "model_dump"):
        payload = result.model_dump()
    elif isinstance(result, dict):
        payload = result
    else:
        payload = vars(result)

    usage = payload.get("usage", {})
    duration = payload.get("duration")
    recorded_duration = duration_override if duration_override is not None else duration
    tracker.add_openai_call(model_id, float(recorded_duration) if recorded_duration else None, usage)

    segments = []
    for seg in payload.get("segments", []):
        start = seg.get("start")
        end = seg.get("end")
        text = seg.get("text", "").strip()
        if start is None or end is None or not text:
            continue
        segments.append(
            TranscriptSegment(float(start), float(end), text)
        )
    if not segments:
        fallback = getattr(result, "text", None) or result.get("text")
        if isinstance(fallback, str) and fallback.strip():
            segments.append(TranscriptSegment(0.0, 0.0, fallback.strip()))
    return segments, payload


def transcribe_with_huggingface(media_path: Path, model_id: str) -> tuple[list[TranscriptSegment], dict[str, Any]]:
    """Run a local Hugging Face Whisper variant."""
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if device == "cuda" else torch.float32

    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    )
    model.to(device)

    stt_pipeline = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        device=0 if device == "cuda" else -1,
        chunk_length_s=30,
        stride_length_s=5,
        return_timestamps=True,
        batch_size=4,
    )

    result = stt_pipeline(str(media_path), generate_kwargs={"task": "transcribe"})
    chunks = result.get("chunks") or []
    segments = []
    for chunk in chunks:
        timestamp = chunk.get("timestamp")
        text = chunk.get("text", "").strip()
        if not timestamp or len(timestamp) != 2 or not text:
            continue
        start, end = timestamp
        segments.append(TranscriptSegment(float(start), float(end), text))
    if not segments:
        fallback = result.get("text")
        if isinstance(fallback, str) and fallback.strip():
            segments.append(TranscriptSegment(0.0, 0.0, fallback.strip()))
    payload: dict[str, Any] = {"text": result.get("text"), "chunks": chunks}
    return segments, payload


def write_transcript(out_path: Path, text: str) -> None:
    """Persist transcript to disk."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    print(f"Transcript saved to {out_path}")


def format_timestamp(seconds: float) -> str:
    """Render seconds as HH:MM:SS.mmm."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"


def format_segment_line(segment: TranscriptSegment) -> str:
    return f"[{format_timestamp(segment.start)}-{format_timestamp(segment.end)}] {segment.text}"


def segments_to_text(segments: Iterable[TranscriptSegment]) -> str:
    return "\n".join(format_segment_line(seg) for seg in segments)


def append_segments_to_file(
    path: Path, segments: Iterable[TranscriptSegment], header: str | None = None
) -> None:
    with path.open("a", encoding="utf-8") as tmp:
        if header:
            tmp.write(f"{header}\n")
        for seg in segments:
            tmp.write(f"{format_segment_line(seg)}\n")
        tmp.write("\n")


def get_media_duration(media_path: Path, ffprobe_bin: str = "ffprobe") -> float:
    if shutil.which(ffprobe_bin) is None:
        raise RuntimeError(f"{ffprobe_bin} is required to measure chunk duration but was not found in PATH.")

    cmd = [
        ffprobe_bin,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(media_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    duration_text = result.stdout.strip()
    try:
        return float(duration_text)
    except ValueError as exc:
        raise RuntimeError(
            f"Unable to parse duration from ffprobe output: {duration_text}"
        ) from exc


def _is_gpt4o_transcribe_model(model_id: str) -> bool:
    normalized = model_id.lower()
    return normalized.startswith("gpt-4o-transcribe")


def transcribe_media(
    media_path: Path,
    *,
    backend: str = "openai",
    model_id: str | None = None,
    segment_duration: int = DEFAULT_SEGMENT_SECONDS,
    ffmpeg_bin: str = "ffmpeg",
    ffprobe_bin: str = "ffprobe",
    show_progress: bool = True,
) -> TranscriptionResult:
    selected_model = model_id or DEFAULT_MODELS[backend]
    tracker = CostTracker()

    def transcribe_target(
        path: Path, duration_override: float | None = None
    ) -> tuple[list[TranscriptSegment], dict[str, Any]]:
        if backend == "openai":
            return transcribe_with_openai(path, selected_model, tracker, duration_override)
        return transcribe_with_huggingface(path, selected_model)

    raw_payload: dict[str, Any]
    if backend == "openai" and media_path.stat().st_size > MAX_OPENAI_CHUNK_BYTES:
        if segment_duration < 10:
            raise RuntimeError("Segment duration must be at least 10 seconds.")
        with tempfile.TemporaryDirectory(prefix="transcribe_chunks_") as tmpdir:
            chunk_dir = Path(tmpdir)
            if show_progress:
                print(
                    f"File exceeds {MAX_OPENAI_CHUNK_BYTES // (1024 * 1024)} MB, splitting into "
                    f"{segment_duration}s chunks..."
                )
            chunk_paths = split_audio(media_path, segment_duration, chunk_dir, ffmpeg_bin=ffmpeg_bin)
            durations = [get_media_duration(chunk, ffprobe_bin=ffprobe_bin) for chunk in chunk_paths]
            if show_progress:
                print(f"Created {len(chunk_paths)} chunks. Transcribing each chunk...")
            temp_fd, temp_path = tempfile.mkstemp(
                prefix="transcribe_partial_",
                suffix=".txt",
            )
            os.close(temp_fd)
            temp_transcript_path = Path(temp_path)
            if show_progress:
                print(f"Intermediate transcript is being written to {temp_transcript_path}")
            segments_total: list[TranscriptSegment] = []
            chunk_payloads: list[dict[str, Any]] = []
            offset = 0.0
            try:
                for idx, (chunk_path, duration) in enumerate(zip(chunk_paths, durations), start=1):
                    if show_progress:
                        print(f"Transcribing chunk {idx}/{len(chunk_paths)}: {chunk_path.name}")
                    chunk_segments, chunk_payload = transcribe_target(chunk_path, duration_override=duration)
                    offset_segments = [seg.offset(offset) for seg in chunk_segments]
                    header = f"--- Chunk {idx}: {chunk_path.name} (start {format_timestamp(offset)}) ---"
                    append_segments_to_file(temp_transcript_path, offset_segments, header=header)
                    segments_total.extend(offset_segments)
                    chunk_payloads.append(
                        {
                            "chunk_index": idx,
                            "chunk_file": chunk_path.name,
                            "offset_seconds": offset,
                            "duration_seconds": duration,
                            "response": chunk_payload,
                        }
                    )
                    offset += duration
            except Exception:
                if show_progress:
                    print(f"Chunked transcript left at {temp_transcript_path} for inspection.")
                raise
            else:
                temp_transcript_path.unlink(missing_ok=True)

        raw_payload = {
            "backend": backend,
            "model": selected_model,
            "chunked": True,
            "chunks": chunk_payloads,
        }
        segments = segments_total
    else:
        segments, single_payload = transcribe_target(media_path)
        raw_payload = {
            "backend": backend,
            "model": selected_model,
            "chunked": False,
            "response": single_payload,
        }

    raw_payload["cost"] = {
        "records": [asdict(record) for record in tracker.records],
        "summary": tracker.summary_text(),
    }
    return TranscriptionResult(
        backend=backend,
        model=selected_model,
        segments=segments,
        raw_payload=raw_payload,
        cost_records=list(tracker.records),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Transcribe a single media asset and save the text."
    )
    parser.add_argument(
        "media_path",
        type=Path,
        help="Path to the input audio or video file to transcribe.",
    )
    parser.add_argument(
        "--backend",
        choices=list(DEFAULT_MODELS),
        default="openai",
        help="Which transcription backend to use (default: openai).",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Override the default model identifier for the chosen backend.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Where to write the transcript (defaults to <input>.txt).",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=Path(".env"),
        help="Location of the dotenv file to load API keys from.",
    )
    parser.add_argument(
        "--segment-duration",
        type=int,
        default=DEFAULT_SEGMENT_SECONDS,
        help="Segment length in seconds when splitting a large file for OpenAI.",
    )
    args = parser.parse_args()

    if not args.media_path.exists():
        print(f"{args.media_path} does not exist.", file=sys.stderr)
        sys.exit(1)
    if not args.media_path.is_file():
        print(f"{args.media_path} is not a file.", file=sys.stderr)
        sys.exit(1)

    ensure_env_file(args.env_file)

    model_id = args.model or DEFAULT_MODELS[args.backend]
    backend = args.backend

    print(f"Transcribing {args.media_path} with backend={backend} model={model_id}")

    result = transcribe_media(
        args.media_path,
        backend=backend,
        model_id=model_id,
        segment_duration=args.segment_duration,
        show_progress=True,
    )
    transcript = segments_to_text(result.segments)

    output_path = args.output or args.media_path.with_suffix(".txt")
    write_transcript(output_path, transcript)

    if summary := result.raw_payload.get("cost", {}).get("summary"):
        print(summary)


def split_audio(
    media_path: Path,
    segment_duration: int,
    output_dir: Path,
    *,
    ffmpeg_bin: str = "ffmpeg",
) -> list[Path]:
    """Use ffmpeg segmenter to cut the input into manageable chunks."""
    if shutil.which(ffmpeg_bin) is None:
        raise RuntimeError(f"{ffmpeg_bin} is required to split large files but was not found in PATH.")

    output_dir.mkdir(parents=True, exist_ok=True)
    pattern = output_dir / "chunk_%04d.mp3"
    cmd = [
        ffmpeg_bin,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(media_path),
        "-f",
        "segment",
        "-segment_time",
        str(segment_duration),
        "-reset_timestamps",
        "1",
        "-c:a",
        "libmp3lame",
        "-q:a",
        "2",
        str(pattern),
    ]
    subprocess.run(cmd, check=True)
    chunk_files = sorted(output_dir.glob("chunk_*.mp3"))
    if not chunk_files:
        raise RuntimeError("Failed to create any chunks. ffmpeg reported no output.")
    return chunk_files


if __name__ == "__main__":
    main()
